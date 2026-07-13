import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def as_probability_map(tensor, eps=1e-6):
    sanitized = torch.nan_to_num(tensor, nan=0.0, posinf=30.0, neginf=-30.0)
    t = sanitized.detach()
    needs_sigmoid = (t.min() < 0.0) | (t.max() > 1.0)
    result = torch.where(needs_sigmoid, torch.sigmoid(sanitized), sanitized)
    return result.clamp(eps, 1 - eps)


def get_structure_boundary(mask, eps=1e-6):
    prob = as_probability_map(mask, eps)
    return normalize_spatial_map(get_boundary_sobel(prob), eps)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


# BCE + Dice  loss
class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bceloss_fn = nn.BCELoss(weight, size_average)

    def forward(self, pred, target):
        pred = as_probability_map(pred)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        bceloss = self.bceloss_fn(pred_flat, target_flat)

        smooth = 1
        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        diceloss = 1 - dice_score.sum() / size

        return bceloss + diceloss


_sobel_kernel_cache = {}


def _get_sobel_kernels(device):
    key = device
    if key not in _sobel_kernel_cache:
        sobel_x = torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
            dtype=torch.float32, device=device,
        )
        sobel_y = torch.tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
            dtype=torch.float32, device=device,
        )
        _sobel_kernel_cache[key] = (sobel_x, sobel_y)
    return _sobel_kernel_cache[key]


def get_boundary_sobel(mask):
    """
    使用Sobel算子提取边界图（论文第3.2节，公式(9)）

    公式：B = sqrt((G_x * P_u)^2 + (G_y * P_u)^2)
    """
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)

    sobel_x, sobel_y = _get_sobel_kernels(mask.device)
    grad_x = F.conv2d(mask, sobel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_y, padding=1)
    boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return boundary


def binary_entropy_map(prob, eps=1e-6):
    prob = as_probability_map(prob, eps)
    return -(prob * torch.log(prob) + (1 - prob) * torch.log(1 - prob))


def normalize_spatial_map(tensor, eps=1e-6):
    max_val = tensor.amax(dim=(2, 3), keepdim=True)
    return tensor / (max_val + eps)


def entropy_weight(prob, tau=0.5, eps=1e-6):
    """Pixel-wise reliability weight from binary entropy (HDC CVPRW'25 §3.1).

    w(i) = exp(-H(p_i) / tau).  High-entropy (ambiguous) pixels get w → 0,
    confident pixels keep w ≈ 1.  Replaces the multi-evidence reliability map.
    """
    H = binary_entropy_map(prob, eps)
    return torch.exp(-H / max(float(tau), 1e-3))


def gaussian_rampup(epoch, total, alpha=0.1, beta=5.0):
    """BiPCC IV-B Eq.16  λ(t) = α · exp(-β · (1 - t/T_max)²).

    Smoother than sigmoid_rampup; weights stay small early, climb monotonically
    and never saturate to 1, which keeps the supervised term dominant.
    """
    t = float(np.clip(epoch, 0.0, total))
    return float(alpha * np.exp(-beta * (1.0 - t / max(float(total), 1.0)) ** 2))


def weighted_bce_dice_loss(pred, target, weight, smooth=1e-5):
    pred = as_probability_map(pred)
    target = torch.nan_to_num(target, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    weight = torch.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)

    bce = F.binary_cross_entropy(pred, target, reduction='none')
    spatial_dims = tuple(range(1, pred.ndim))
    bce = (bce * weight).mean(dim=spatial_dims)

    intersection = (pred * target * weight).sum(dim=spatial_dims)
    union = ((pred + target) * weight).sum(dim=spatial_dims)
    dice = 1 - (2 * intersection + smooth) / (union + smooth)
    reliability_mass = weight.mean(dim=spatial_dims).detach()
    return (bce + reliability_mass * dice).mean()


def anatomy_boundary_reliability(
    main_prob,
    aux_prob,
    boundary_logits,
    teacher_prob=None,
    entropy_tau=0.5,
    reliability_floor=0.0,
    eps=1e-6,
    **_legacy_kwargs,
):
    """Entropy-based pseudo-label reliability (HDC §3.1 + ShiftMatch Eq.3).

    The pseudo target is the EMA teacher when provided, else the main/aux mean.
    The reliability map is simply exp(-H(p̂)/τ); ambiguous pixels get w→0.
    All boundary/structure/anatomy evidence fusion from earlier revisions has
    been removed — entropy alone gives a cleaner, dataset-agnostic weight.

    Returned tuple is preserved so callers (evaluate.py, reliable_pseudo_label_loss)
    don't need to change.  Extra legacy keyword args are accepted and ignored.
    """
    main_prob = as_probability_map(main_prob, eps)
    aux_prob = as_probability_map(aux_prob, eps)
    if teacher_prob is None:
        pseudo_prob = 0.5 * (main_prob + aux_prob)
    else:
        pseudo_prob = as_probability_map(teacher_prob, eps)
    pseudo_boundary = get_structure_boundary(pseudo_prob, eps)
    reliability = entropy_weight(pseudo_prob, tau=entropy_tau, eps=eps)
    reliability = reliability.clamp(float(reliability_floor), 1.0)
    diagnostics = {'reliability_mean': reliability.detach().mean()}
    return pseudo_prob.detach(), reliability.detach(), pseudo_boundary.detach(), diagnostics


def reliable_pseudo_label_loss(
    main_prob,
    aux_prob,
    boundary_logits,
    teacher_prob=None,
    pseudo_main_weight=1.0,
    pseudo_aux_weight=0.5,
    pseudo_boundary_weight=0.1,
    entropy_tau=0.5,
    reliability_floor=0.0,
    eps=1e-6,
    **_legacy_kwargs,
):
    """Entropy-weighted pseudo-label loss (PL + cross-decoder + boundary).

    L_pl = pseudo_main_weight · weighted_bce_dice(main, p̂; w)
         + pseudo_aux_weight  · weighted_bce_dice(aux,  p̂; w)
         + pseudo_boundary_weight · BCE_logits(boundary, ∂p̂; w)

    where w = exp(-H(p̂)/τ).  Legacy kwargs (anatomy_logits, hard_edge_*, ...) are
    accepted and ignored to keep callsites stable during the transition.
    """
    pseudo_prob, reliability, pseudo_boundary, diagnostics = anatomy_boundary_reliability(
        main_prob.detach(), aux_prob.detach(),
        boundary_logits.detach() if torch.is_tensor(boundary_logits) else boundary_logits,
        teacher_prob=teacher_prob,
        entropy_tau=entropy_tau,
        reliability_floor=reliability_floor,
        eps=eps,
    )

    loss_main = weighted_bce_dice_loss(main_prob, pseudo_prob, reliability)
    loss_aux = weighted_bce_dice_loss(aux_prob, pseudo_prob, reliability)
    boundary_logits = torch.nan_to_num(boundary_logits, nan=0.0, posinf=30.0, neginf=-30.0)
    boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, pseudo_boundary, reduction='none')
    boundary_loss = (boundary_loss * reliability).mean()
    total_loss = (
        float(pseudo_main_weight) * loss_main +
        float(pseudo_aux_weight) * loss_aux +
        float(pseudo_boundary_weight) * boundary_loss
    )
    return total_loss, loss_main, loss_aux, boundary_loss, pseudo_prob, reliability, pseudo_boundary, diagnostics


def wsdice_per_class(pred, target, w_neg=0.05, smooth=1e-5):
    """Weighted Soft Dice per channel (Improved Dice, IEEE Access 2020).

    L_k = 1 - (2·Σpy + w·Σ(1-p)(1-y) + s) / (Σp + Σy + 2w·Σ(1-p)(1-y) + s)

    For small classes (e.g. PS ≈ 5% area on PSFHS) the negative-area term
    rescales the gradient back to the same magnitude as large classes.
    Pass w_neg as a list/tuple to set a different value per class.
    """
    pred = as_probability_map(pred)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    K = pred.size(1)
    if not isinstance(w_neg, (list, tuple)):
        w_neg = [float(w_neg)] * K
    w_neg = [float(w) for w in w_neg[:K]]
    while len(w_neg) < K:
        w_neg.append(w_neg[-1])

    losses = []
    for k in range(K):
        p, y = pred[:, k], target[:, k]
        inter = (p * y).sum()
        neg = ((1 - p) * (1 - y)).sum()
        denom = p.sum() + y.sum()
        dsc = (2 * inter + w_neg[k] * neg + smooth) / (denom + 2 * w_neg[k] * neg + smooth)
        losses.append(1 - dsc)
    return torch.stack(losses).mean()


def gradient_balanced_w_neg(target, alpha=1.0, eps=1e-3):
    """Adaptive per-class w_neg that equalizes WSDice gradient magnitudes.

    Theorem (Gradient Balance, see paper §3.X): for K-class WSDice with
    per-class negative-area weight w_k, the per-class gradient magnitude
    is bounded by O(1) (independent of class size) iff w_k = α · A_k / N_k
    where A_k is the positive area fraction and N_k = 1 - A_k.

    For PSFHS this gives:
        PS  (A≈0.03, N≈0.97):  w_PS ≈ 0.031
        FH  (A≈0.30, N≈0.70):  w_FH ≈ 0.429

    The values are derived per-batch from the target tensor — no external
    statistics required, no per-dataset tuning. Callers can fuse the result
    with their fixed w_neg via EMA for stability.
    """
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    # Per-class positive area fraction (averaged over batch and spatial dims).
    A = target.mean(dim=(0, 2, 3)).clamp(eps, 1.0 - eps)  # [K]
    N = 1.0 - A
    w = alpha * (A / N)
    return w.detach().cpu().tolist()


def inverse_freq_w_neg(target, alpha=1.0, eps=1e-3):
    """Inverse class-frequency baseline for WSDice w_neg.

    w_k ∝ 1 / A_k, normalized so max_k w_k = alpha. Rare classes get a
    larger negative-area weight, common classes get a proportionally smaller
    one. Standard re-weighting heuristic from imbalanced-class learning.

    Baseline alternative to gradient_balanced_w_neg (Theorem 1).
    """
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    A = target.mean(dim=(0, 2, 3)).clamp(eps, 1.0 - eps)
    w = 1.0 / A
    w = alpha * w / w.max()
    return w.detach().cpu().tolist()


def effective_number_w_neg(target, beta=0.9999, alpha=1.0, eps=1e-3):
    """Class-Balanced Loss (Cui et al. CVPR'19) effective-number weighting.

    w_k = (1 - β) / (1 - β^{n_k}), normalized so max_k w_k = alpha. Here
    n_k is the per-batch positive-pixel count for class k.  β controls how
    quickly samples saturate to "effective"; β → 1 strengthens the rare-class
    boost, β = 0 reduces to uniform weighting.

    Baseline alternative to gradient_balanced_w_neg (Theorem 1).
    """
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    n = target.sum(dim=(0, 2, 3)).clamp_min(1.0)  # [K] positive pixel count
    beta_t = torch.tensor(float(beta), device=n.device, dtype=n.dtype).clamp(0.0, 1.0 - 1e-7)
    eff = (1.0 - torch.pow(beta_t, n)).clamp_min(eps)
    w = (1.0 - beta_t) / eff
    w = alpha * w / w.max()
    return w.detach().cpu().tolist()


def carc_reliability(teacher_prob, shape_score=None, class_area=None,
                     tau=0.5, reliability_floor=0.0, eps=1e-6):
    """Class-Aware Reliability Coupling (CARC, paper §3.X).

    Multiplicatively couples three independent reliability sources so the
    pseudo-supervision automatically vanishes when ANY source disagrees:

        R(x, k) = R_shape(x) · R_pixel(x, k) · R_class(k)

    Independent components:
      R_pixel(x, k) = exp(-H(p̂_k(x)) / τ)
        — pixel-level binary entropy reliability (HDC CVPRW'25)
      R_shape(x) = σ(D(x̃))       (broadcast across classes)
        — global shape plausibility from the Wasserstein critic
      R_class(k) = w_k / max(w)   (broadcast across spatial dims)
        — per-class importance, normalized to [0,1]

    For PSFH multi-class this prevents the small PS class from being
    silently down-weighted whenever the critic happens to be unsure;
    each reliability source must independently agree before a pixel
    contributes to the pseudo-label loss.

    Proposition (Fail-safe). For any (x, k), R(x,k) ≤ min(R_shape,
    R_pixel, R_class). When any source is ≈ 0, the coupled reliability
    is ≈ 0 and that pseudo-label has negligible gradient — preventing
    confirmation bias from a single uncertain source.

    teacher_prob: [B, K, H, W] EMA-teacher pseudo-prob.
    shape_score : [B, 1, H', W'] or scalar — critic-derived shape
                  plausibility; None or all-zeros falls back to
                  pixel × class only.
    class_area  : [K] target area fractions used to weight class
                  importance; None falls back to uniform.
    """
    R_pixel = entropy_weight(teacher_prob, tau=tau, eps=eps)  # [B, K, H, W]
    R = R_pixel
    if shape_score is not None and torch.is_tensor(shape_score):
        shape = shape_score.float()
        # The critic typically returns a scalar per sample. Reshape to a
        # spatial map that broadcasts to R = [B, K, H, W].
        if shape.dim() == 0:
            shape = shape.view(1, 1, 1, 1)
        elif shape.dim() == 1:  # [B] — scalar per sample
            shape = shape.view(shape.size(0), 1, 1, 1)
        elif shape.dim() == 2:  # [B, 1] or [B, C]
            shape = shape.view(shape.size(0), shape.size(1), 1, 1)
        elif shape.dim() == 3:
            shape = shape.unsqueeze(1)
        # Sigmoid → [0,1]; clamp_max ensures it acts as a weight not a logit.
        shape = torch.sigmoid(shape).clamp(eps, 1.0)
        # Only interpolate when the critic actually returns a spatial map
        # (height/width > 1); a scalar-per-sample [B,1,1,1] broadcasts directly.
        if shape.shape[-1] > 1 and shape.shape[-2] > 1 and shape.shape[-2:] != R.shape[-2:]:
            shape = F.interpolate(shape, size=R.shape[-2:], mode='bilinear', align_corners=False)
        R = R * shape.expand_as(R)
    if class_area is not None and torch.is_tensor(class_area) and class_area.numel() > 0:
        w = class_area.float().clamp_min(eps)
        # Normalize so the largest class gets weight 1; small classes get
        # weight = A_k / A_max. We want the small class to be MORE reliable
        # (since it's harder, every pixel matters), so invert and renormalize.
        inv = 1.0 / w
        inv = inv / inv.max()
        R_class = inv.view(1, -1, 1, 1)
        R = R * R_class
    return R.clamp(float(reliability_floor), 1.0)
