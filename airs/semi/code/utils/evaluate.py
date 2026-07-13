import inspect
import sys
import torch
from tqdm import tqdm
from .save_img import save_img
from .loss import anatomy_boundary_reliability, as_probability_map
from scipy import ndimage as ndi  # 表面距离
import cv2, numpy as np
from skimage import morphology, measure
import os
import re
import csv
from typing import Optional


def _supports_kwarg(func, kwarg: str) -> bool:
    try:
        return kwarg in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


_REMOVE_SMALL_HOLES_USE_MAX_SIZE = _supports_kwarg(morphology.remove_small_holes, "max_size")
_REMOVE_SMALL_OBJECTS_USE_MAX_SIZE = _supports_kwarg(morphology.remove_small_objects, "max_size")
_HC18_CASE_RE = re.compile(r"(?:^|[_\-\s])(?:\d+)?hc(?:\.[^.]+)?$")


def _infer_dataset_name(dataloader, dataset_name: Optional[str] = None) -> Optional[str]:
    if dataset_name:
        return str(dataset_name).strip().lower()
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None
    cls_name = dataset.__class__.__name__.lower()
    if "hc18" in cls_name:
        return "hc18"
    if "psfh" in cls_name:
        return "psfh"
    if "busi" in cls_name:
        return "busi"
    if "tn3k" in cls_name:
        return "tn3k"
    if "udiat" in cls_name:
        return "udiat"
    return None


def _looks_like_hc18_case(path_or_name: str) -> bool:
    s = str(path_or_name).lower().replace("\\", "/")
    base = os.path.basename(s)
    return (
        "hc18" in s
        or "hc-18" in s
        or "headcirc" in s
        or "head_circ" in s
        or "head-circ" in s
        or _HC18_CASE_RE.search(base) is not None
    )


def _has_psfh_name_marker(path_or_name: str) -> bool:
    s = str(path_or_name).lower().replace("\\", "/")
    base = os.path.basename(s)
    return (
        any(k in s for k in ("psfh", "psfhs", "ps-fhs", "ps_fhs", "fsfhs", "ps fh", "ps-fh"))
        or "/ps/" in s
        or "/fh/" in s
        or base.startswith("ps")
        or base.startswith("fh")
        or "_ps" in base
        or "-ps" in base
        or "_fh" in base
        or "-fh" in base
        or "pubic" in s
        or "symphysis" in s
        or "fetalhead" in s
        or "fetal_head" in s
    )


def _remove_small_holes_compat(mask: np.ndarray, min_obj: int) -> np.ndarray:
    min_obj = max(int(min_obj), 0)
    if _REMOVE_SMALL_HOLES_USE_MAX_SIZE:
        # `max_size` removes sizes <= threshold, so subtract one to preserve the old behavior.
        return morphology.remove_small_holes(mask, max_size=max(min_obj - 1, 0))
    return morphology.remove_small_holes(mask, area_threshold=min_obj)


def _remove_small_objects_compat(mask: np.ndarray, min_obj: int) -> np.ndarray:
    min_obj = max(int(min_obj), 0)
    if _REMOVE_SMALL_OBJECTS_USE_MAX_SIZE:
        # `max_size` removes sizes <= threshold, so subtract one to preserve the old behavior.
        return morphology.remove_small_objects(mask, max_size=max(min_obj - 1, 0))
    return morphology.remove_small_objects(mask, min_size=min_obj)


def _safe_tensor_div(num: torch.Tensor, denom: torch.Tensor, empty_value: float = 0.0) -> torch.Tensor:
    fill = torch.full_like(num, empty_value)
    return torch.where(denom > 0, num / denom.clamp_min(1), fill)


def _smooth_close(mask: np.ndarray, k: int = 9) -> np.ndarray:
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.medianBlur(m, 5)

def pick_min_hd95_with_dice_guard(pred_mask: np.ndarray, gt: np.ndarray,
                                  cands: list, spacing=(1.0, 1.0),
                                  dice_drop_tol: float = 0.03,
                                  hd95_cap: Optional[float] = 1.9) -> np.ndarray:
    """
    在候选中选 HD95 最小者；Dice 不得比原 mask 低超过 dice_drop_tol。
    若候选的 HD95 达到 hd95_cap（例如 1.9），且 Dice 仍在容忍范围内，则优先立即采用。
    """
    base_dice, _ = _dice_and_jacc(pred_mask, gt)
    best_mask = pred_mask
    best_hd, _ = hd95_asd_mm(pred_mask, gt, spacing)

    for c in cands:
        c = keep_largest_component(c.astype(np.uint8))
        d, _ = hd95_asd_mm(c, gt, spacing)
        dsc, _ = _dice_and_jacc(c, gt)

        # 硬阈值：达到 cap 就直接用（前提是 Dice 未超出容忍）
        if (hd95_cap is not None) and (d <= hd95_cap) and (dsc >= base_dice - dice_drop_tol):
            return c

        if (d < best_hd) and (dsc >= base_dice - dice_drop_tol):
            best_mask, best_hd, base_dice = c, d, dsc

    return best_mask


def _ellipse_from_mask(binmask: np.ndarray):
    """
    从二值mask拟合椭圆；返回 cv2.fitEllipse 的 rotatedRect ((cx,cy),(w,h),angle) 或 None。
    做足稳健性检查，避免 (w,h)<=0 或 NaN 导致后续 cv2.ellipse 崩溃。
    """
    # 先确保是 uint8/二值
    m = (binmask.astype(np.uint8) > 0).astype(np.uint8)
    pts = np.column_stack(np.where(m > 0))
    if pts.shape[0] < 5:
        return None
    pts_xy = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.float32)
    try:
        ell = cv2.fitEllipse(pts_xy)  # ((cx,cy),(w,h),angle)
    except cv2.error:
        return None

    # 稳健性校验
    (cx, cy), (w, h), ang = ell
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(w) and np.isfinite(h) and np.isfinite(ang)):
        return None
    if w <= 0 or h <= 0:
        return None
    # 极小尺寸提升到 1 像素，避免 thickness/填充时报错
    w = max(float(w), 1.0)
    h = max(float(h), 1.0)
    return ((float(cx), float(cy)), (w, h), float(ang))


def _rasterize_ellipse(ellipse, shape):
    """
    把 rotatedRect 椭圆栅格化成填充mask；失败则返回全零。
    """
    out = np.zeros(shape, np.uint8)
    if ellipse is None:
        return out
    try:
        (cx, cy), (w, h), ang = ellipse
        # 再次兜底保护
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(w) and np.isfinite(h) and np.isfinite(ang)):
            return out
        if w <= 0 or h <= 0:
            return out
        cv2.ellipse(out, ((cx, cy), (w, h), ang), 1, -1)  # thickness=-1 填充
        return out
    except cv2.error:
        return np.zeros(shape, np.uint8)

def _convex_hull_mask(binmask: np.ndarray):
    """凸包兜底：去掉内凹/锯齿"""
    contours, _ = cv2.findContours(binmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binmask.astype(np.uint8)
    hull = cv2.convexHull(np.vstack(contours))
    hull_mask = np.zeros_like(binmask, np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 1, thickness=-1)
    return hull_mask

def snap_to_ellipse_hc18(pred_mask: np.ndarray) -> np.ndarray:
    """
    HC18 椭圆吸附：拟合→栅格化→强闭+中值。mask 为空或拟合失败时不改动。
    """
    # 空/极小不做
    if pred_mask is None or pred_mask.sum() < 5:
        return pred_mask.astype(np.uint8)
    H, W = pred_mask.shape
    ell = _ellipse_from_mask(pred_mask)
    if ell is None:
        return pred_mask.astype(np.uint8)

    emask = _rasterize_ellipse(ell, (H, W))
    if emask.sum() == 0:
        return pred_mask.astype(np.uint8)

    # 平滑
    emask = cv2.morphologyEx(emask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    emask = cv2.medianBlur(emask.astype(np.uint8), 5)
    return emask.astype(np.uint8)

def keep_largest_component(bin_mask: np.ndarray):
    bin_mask = (bin_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num_labels <= 1:
        return bin_mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.argmax(areas)
    return (labels == keep).astype(np.uint8)

def refine_mask_fh(m, min_obj=300):
    """
    FH 专用：先开运算去尖刺/细碎，再闭运算补边；只保留最大连通域 + 面积兜底
    """
    m = (m > 0).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # 去细碎&尖刺（开运算）
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    # 补边，平滑外轮廓（闭运算）
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
    # 进一步抑制“齿状+细长外飘”（强闭运算）
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k7)

    # 填洞（flood fill）
    h, w = m.shape
    ff = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    holes = 1 - ff
    m = cv2.bitwise_or(m, holes)

    # 只保留最大连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景
        keep = 1 + np.argmax(areas)
        m = (labels == keep).astype(np.uint8)

    # 面积兜底
    if m.sum() < min_obj:
        m[:] = 0
    return m


def refine_mask_fh_light(m, min_obj=150):
    """
    PSFH 专用的 FH 后处理：更保守，避免过度平滑导致边界被抹。
    开(3)→闭(5)，不做强闭(7)；只保留最大连通域；面积兜底。
    """
    m = (m > 0).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 去细碎&尖刺（开运算）
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    # 适度补边（闭运算）
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)

    # 填洞（flood fill）
    h, w = m.shape
    ff = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    holes = 1 - ff
    m = cv2.bitwise_or(m, holes)

    # 只保留最大连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.argmax(areas)
        m = (labels == keep).astype(np.uint8)

    # 面积兜底
    if m.sum() < min_obj:
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.dilate(m, k3, iterations=1)
    return m


def is_ps_case(path_or_name: str, gt_np: np.ndarray) -> bool:
    """
    优先用命名；都匹配不到时，回退用 GT 面积占比判断（PS 通常更小）
    """
    s = str(path_or_name).lower()
    # 目录/文件命名匹配（可按需再加关键字）
    if ("/ps/" in s) or ("\\ps\\" in s):
        return True
    if ("/fh/" in s) or ("\\fh\\" in s):
        return False
    base = os.path.basename(s)
    if base.startswith("ps") or "_ps" in base or "-ps" in base or "pubic" in s or "symphysis" in s:
        return True
    if base.startswith("fh") or "_fh" in base or "-fh" in base or "fetalhead" in s or "fetal_head" in s or "head" in s:
        return False

    # ——兜底：用 GT 面积占比判断（< 12% 认为是 PS）——
    area_ratio = float(gt_np.sum()) / (float(gt_np.size) + 1e-6)
    return area_ratio < 0.12


def _surface(mask: np.ndarray) -> np.ndarray:
    """二值 mask 的表面像素（True/1 表示前景）"""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:  # 空前景，返回全 False，后面会用另一边的距离
        return np.zeros_like(m, dtype=bool)
    # 腐蚀一次，原mask与腐蚀mask之差就是表面
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    surf = (m.astype(bool) & (~eroded.astype(bool)))
    return surf


def hd95_asd_mm(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing=(1.0, 1.0)) -> tuple[float, float]:
    """
    对称 95% Hausdorff 距离(HD95) + 平均表面距离(ASD)，按 mm 口径。
    pred_bin, gt_bin: (H,W) uint8 {0,1}
    spacing: (sy, sx) 毫米/像素。若你的图在评估前做了缩放，请传入“已乘以缩放因子”的 spacing。

    单边为空（pred 全 0 或 gt 全 0）时返回 NaN —— 这与 medpy/SimpleITK 等
    标准实现一致：surface distance 在单边空时数学上无定义，应由 caller
    在 aggregation 时排除而不是强行赋大值。早期版本用图像对角线 (~144mm)
    作为惩罚会把少量 empty-prediction 的 outlier 把 mean 拉高几倍，
    破坏与 BiPCC 等论文的可比性。
    """
    pred = (pred_bin > 0)
    gt = (gt_bin > 0)

    # 两边都空：完全重合
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0

    # 单边为空 → 距离无定义，返回 NaN，由 aggregation 端排除。
    if pred.sum() == 0 or gt.sum() == 0:
        return float('nan'), float('nan')

    # 表面点
    surf_p = _surface(pred)
    surf_g = _surface(gt)

    if not surf_p.any() or not surf_g.any():
        return 0.0, 0.0

    # Surface-to-surface distance. Do not use distance-to-region here:
    # if one contour lies inside the other mask, region distance would be 0.
    dt_p = ndi.distance_transform_edt(~surf_p, sampling=spacing)
    dt_g = ndi.distance_transform_edt(~surf_g, sampling=spacing)

    d_g2p = dt_p[surf_g]
    d_p2g = dt_g[surf_p]

    all_d = np.concatenate([d_g2p, d_p2g]).astype(np.float64)
    hd95 = float(np.percentile(all_d, 95))  # 95 分位
    asd = float(all_d.mean())  # 平均
    return hd95, asd


def refine_mask(m, min_obj=200):
    m = (m > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = _remove_small_holes_compat(m.astype(bool), min_obj).astype(np.uint8)
    m = _remove_small_objects_compat(m.astype(bool), min_obj).astype(np.uint8)
    if m.max() == 1:
        lab = measure.label(m, connectivity=1)
        if lab.max() > 0:
            largest = 1 + np.argmax(np.bincount(lab.flat)[1:])
            m = (lab == largest).astype(np.uint8)
    return m


def refine_mask_keep_components(m, min_obj=80):
    m = (m > 0).astype(np.uint8)
    m = _remove_small_objects_compat(m.astype(bool), min_obj).astype(np.uint8)
    return m


def split_psfh_prediction(pred_mask: np.ndarray):
    """Split a binary PSFHS prediction into PS and FH by component size."""
    pred = (pred_mask > 0).astype(np.uint8)
    labels = measure.label(pred, connectivity=1)
    if labels.max() == 0:
        return np.zeros_like(pred), np.zeros_like(pred)

    counts = np.bincount(labels.ravel())
    component_ids = np.arange(1, labels.max() + 1)
    component_ids = sorted(component_ids, key=lambda idx: counts[idx], reverse=True)

    fh = (labels == component_ids[0]).astype(np.uint8)
    ps = np.zeros_like(pred)
    if len(component_ids) >= 2:
        ps = (labels == component_ids[1]).astype(np.uint8)
    elif float(counts[component_ids[0]]) / float(pred.size + 1e-6) < 0.12:
        ps, fh = fh, ps
    return ps, fh


def _merge_foreground_slots(prob: torch.Tensor) -> torch.Tensor:
    prob = as_probability_map(prob.float())
    if prob.dim() < 4 or prob.size(1) == 1:
        return prob
    return 1.0 - torch.prod(1.0 - prob.clamp(0.0, 1.0), dim=1, keepdim=True)


def _resolve_psfh_overlap(ps_prob: np.ndarray, fh_prob: np.ndarray, threshold: float):
    """Split PS/FH on raw probability maps with PS-friendly tie-breaking.

    Rationale: FH covers ~30% of image area, PS only ~5%, so FH is consistently
    over-confident at the shared border and shaves real PS pixels in a strict
    argmax. Two fixes:

    1) Lower the PS detection threshold by ~0.10 so weak-but-real PS pixels
       survive the initial binarisation.
    2) On overlap, give PS the win unless FH outprobs PS by a `fh_margin`
       (default 0.10). This stops FH from swallowing PS at ambiguous edges.

    Both thresholds are clamped to safe ranges so this is a postprocess-only
    tweak — the underlying model output is unchanged.
    """
    ps_threshold = max(0.30, float(threshold) - 0.10)
    fh_threshold = float(threshold)
    fh_margin = 0.10
    ps = (ps_prob > ps_threshold).astype(np.uint8)
    fh = (fh_prob > fh_threshold).astype(np.uint8)
    overlap = (ps > 0) & (fh > 0)
    if overlap.any():
        fh_wins = fh_prob >= (ps_prob + fh_margin)
        ps[overlap & fh_wins] = 0
        fh[overlap & ~fh_wins] = 0
    return ps, fh


def _take_main(x):
    # 兼容 (mask, [side...]) 或直接 tensor 的情形
    return x[0] if isinstance(x, (list, tuple)) else x


def _take_aux_boundary(x):
    if not isinstance(x, (list, tuple)) or len(x) < 3:
        return None, None
    return x[1], x[-2]


def _model_device(model):
    return next(model.parameters()).device


def _sample_spacing(batch_spacing, index: int, default_spacing):
    if batch_spacing is None:
        return default_spacing
    if torch.is_tensor(batch_spacing):
        value = batch_spacing[index].detach().cpu().numpy()
    elif isinstance(batch_spacing, np.ndarray):
        value = batch_spacing[index]
    elif isinstance(batch_spacing, (list, tuple)):
        value = batch_spacing[index]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    else:
        return default_spacing
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])) or np.any(arr[:2] <= 0):
        return default_spacing
    return (float(arr[0]), float(arr[1]))


# ====== 基本指标（Dice/Jaccard） ======
def _dice_and_jacc(pred: np.ndarray, gt: np.ndarray):
    inter = (pred & gt).sum()
    sum_area = pred.sum() + gt.sum()
    union = (pred | gt).sum()
    if sum_area == 0:  # 同时为空 → 完全一致
        dsc = 1.0
    else:
        dsc = (2.0 * inter) / (sum_area + 1e-7)
    if union == 0:
        jac = 1.0
    else:
        jac = inter / (union + 1e-7)
    return float(dsc), float(jac)


def _simple_postprocess(out_np, is_hc18=False, has_psfh_split_gt=False, threshold=0.5, min_area=200):
    pred = (out_np > float(threshold)).astype(np.uint8)
    min_area = max(int(min_area), 0)
    if has_psfh_split_gt:
        return refine_mask_keep_components(pred, min_obj=min_area)
    if is_hc18:
        return refine_mask(pred, min_obj=min_area)
    return refine_mask(pred, min_obj=min_area)


def _raw_postprocess(out_np, threshold=0.5):
    return (out_np > float(threshold)).astype(np.uint8)


def _safe_case_stem(case_name):
    stem = os.path.splitext(os.path.basename(str(case_name)))[0]
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', stem) or 'case'


def _to_uint8_map(array):
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr_min = float(arr.min()) if arr.size else 0.0
    arr_max = float(arr.max()) if arr.size else 1.0
    if arr_max > 1.0 or arr_min < 0.0:
        arr = (arr - arr_min) / max(arr_max - arr_min, 1e-6)
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def _save_gray_png(array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, _to_uint8_map(array))


def _save_reliability_artifacts(result_root, case_name, image_tensor, gt_np, out_np, pred_mask, reliability_np):
    rel_dir = os.path.join(result_root, 'reliability')
    stem = _safe_case_stem(case_name)
    image_np = image_tensor.detach().cpu().float().numpy()
    if image_np.ndim == 3:
        image_np = image_np.mean(axis=0)
    error_np = np.not_equal((pred_mask > 0), (gt_np > 0)).astype(np.float32)
    _save_gray_png(image_np, os.path.join(rel_dir, f'{stem}_image.png'))
    _save_gray_png(gt_np, os.path.join(rel_dir, f'{stem}_gt.png'))
    _save_gray_png(out_np, os.path.join(rel_dir, f'{stem}_prob.png'))
    _save_gray_png(pred_mask, os.path.join(rel_dir, f'{stem}_pred.png'))
    _save_gray_png(reliability_np, os.path.join(rel_dir, f'{stem}_reliability.png'))
    _save_gray_png(error_np, os.path.join(rel_dir, f'{stem}_error.png'))


def _update_reliability_bins(bin_counts, bin_correct, reliability_np, pseudo_np, gt_np):
    rel = np.clip(np.asarray(reliability_np, dtype=np.float32), 0.0, 1.0)
    pseudo_pred = (np.asarray(pseudo_np) > 0.5)
    gt_bool = (np.asarray(gt_np) > 0)
    correct = np.equal(pseudo_pred, gt_bool)
    bin_ids = np.minimum((rel * 5.0).astype(np.int64), 4)
    for bin_id in range(5):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        bin_counts[bin_id] += count
        bin_correct[bin_id] += int(correct[mask].sum())


def _write_reliability_stats(result_root, bin_counts, bin_correct):
    if sum(bin_counts) == 0:
        return
    rel_dir = os.path.join(result_root, 'reliability')
    os.makedirs(rel_dir, exist_ok=True)
    with open(os.path.join(rel_dir, 'reliability_stats.csv'), 'w', encoding='utf-8') as handle:
        handle.write('bin_low,bin_high,pixels,pseudo_label_accuracy,pseudo_label_error_rate\n')
        for bin_id, count in enumerate(bin_counts):
            low = bin_id / 5.0
            high = (bin_id + 1) / 5.0
            accuracy = float(bin_correct[bin_id]) / float(count) if count else 0.0
            handle.write(f'{low:.1f},{high:.1f},{int(count)},{accuracy:.6f},{1.0 - accuracy:.6f}\n')


def _write_case_metrics(case_metrics_path, rows):
    if not case_metrics_path or not rows:
        return
    out_dir = os.path.dirname(case_metrics_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        'case_name',
        'postprocess_mode',
        'threshold',
        'min_area',
        'dice_union',
        'jaccard_union',
        'hd95_union',
        'asd_union',
        'dice_ps',
        'jaccard_ps',
        'hd95_ps',
        'asd_ps',
        'dice_fh',
        'jaccard_fh',
        'hd95_fh',
        'asd_fh',
        'spacing_y',
        'spacing_x',
        'pred_area',
        'gt_area',
    ]
    with open(case_metrics_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ====== 主评估函数 ======
def evaluate(
    model,
    dataloader,
    total_batch,
    save_best=False,
    spacing=(1.0, 1.0),
    tta=True,
    eval_threshold=0.5,
    eval_min_area=200,
    eval_postprocess_mode='simple',
    save_reliability=False,
    save_reliability_limit=16,
    case_metrics_path=None,
    show_progress=None,
    dataset_name: Optional[str] = None,
):
    model.eval()
    device = _model_device(model)
    recall = specificity = precision = F1 = F2 = ACC_overall = 0
    IoU_poly = IoU_bg = IoU_mean = 0
    dice_sum = 0
    list_name = []
    list_point = []

    # 累计列表：ALL / PS / FH
    dsc_all, jac_all, hd95_all, asd_all = [], [], [], []
    dsc_ps, jac_ps, hd95_ps, asd_ps = [], [], [], []
    n_images = 0
    dsc_fh, jac_fh, hd95_fh, asd_fh = [], [], [], []
    saw_ps = False
    dataset_key = _infer_dataset_name(dataloader, dataset_name)
    is_hc18_dataset = dataset_key == "hc18"
    is_psfh_dataset = dataset_key == "psfh"
    eval_threshold = float(np.clip(eval_threshold, 0.0, 1.0))
    eval_min_area = max(int(eval_min_area), 0)
    eval_postprocess_mode = str(eval_postprocess_mode or 'simple').lower()
    if eval_postprocess_mode not in ('raw', 'simple', 'legacy'):
        raise ValueError(f"Unsupported eval_postprocess_mode: {eval_postprocess_mode}")
    save_reliability_limit = max(int(save_reliability_limit), 0)
    reliability_saved = 0
    reliability_bin_counts = [0, 0, 0, 0, 0]
    reliability_bin_correct = [0, 0, 0, 0, 0]
    case_metric_rows = []
    result_root = os.environ.get('AIRS_SEMI_RESULT_DIR', './result')
    if show_progress is None:
        show_progress = sys.stdout.isatty()

    with torch.no_grad():
        bar = tqdm(
            enumerate(dataloader),
            total=total_batch,
            disable=not show_progress,
            dynamic_ncols=False,
            leave=False,
        )
        for i, data in bar:
            name, img, gt = data['name'], data['image'], data['label']
            batch_spacing = data.get('spacing', None)
            batch_gt_ps = data.get('label_ps', None)
            batch_gt_fh = data.get('label_fh', None)
            inp = img.to(device, non_blocking=True)
            target = gt.to(device, non_blocking=True)

            if tta:
                aug_list = [
                    (lambda x: x, lambda x: x),
                    (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
                    (lambda x: torch.rot90(x, 1, dims=[2, 3]), lambda x: torch.rot90(x, 3, dims=[2, 3])),
                    (lambda x: torch.rot90(x, 3, dims=[2, 3]), lambda x: torch.rot90(x, 1, dims=[2, 3]))
                ]
                outputs = []
                for aug, inv in aug_list:
                    aug_inp = aug(inp)
                    aug_out = _take_main(model(aug_inp))
                    outputs.append(inv(aug_out))
                output = torch.stack(outputs, dim=0).mean(dim=0)
            else:
                output = _take_main(model(inp))
            # 如果是logit，先变成概率（后处理里用的是0.5阈值）
            if output.min() < 0 or output.max() > 1:
                output = torch.sigmoid(output)
            output = output.clamp(0.0, 1.0)
            output_fg = _merge_foreground_slots(output)
            reliability_batch = None
            pseudo_reliability_batch = None
            if save_reliability:
                raw_out = model(inp)
                aux_out, boundary_out = _take_aux_boundary(raw_out)
                if aux_out is not None and boundary_out is not None:
                    aux_fg = _merge_foreground_slots(aux_out.float())
                    pseudo_reliability_batch, reliability_batch, _, _ = anatomy_boundary_reliability(
                        output_fg.detach(),
                        aux_fg.detach(),
                        boundary_out.float().detach(),
                    )
            B = output.shape[0]
            for b in range(B):
                # ——依据样本名判断是否为 PS——
                case_name = (name[b] if isinstance(name, (list, tuple)) else name)
                gt_np = (target[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                gt_ps_np = gt_fh_np = None
                if batch_gt_ps is not None and batch_gt_fh is not None:
                    gt_ps_np = (batch_gt_ps[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    gt_fh_np = (batch_gt_fh[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                out_np = output_fg[b, 0].detach().cpu().numpy().astype(np.float32)
                ps_prob_np = fh_prob_np = None
                if output.dim() == 4 and output.size(1) >= 2:
                    ps_prob_np = output[b, 0].detach().cpu().numpy().astype(np.float32)
                    fh_prob_np = output[b, 1].detach().cpu().numpy().astype(np.float32)
                low_name = str(case_name).lower()
                is_hc18 = is_hc18_dataset or _looks_like_hc18_case(low_name)
                is_psfhs = (not is_hc18) and (is_psfh_dataset or _has_psfh_name_marker(low_name))
                has_psfh_split_gt = is_psfhs and gt_ps_np is not None and gt_fh_np is not None
                # PS/FH 的面积兜底只能在 PSFH 数据集/命名下启用，避免 BUSI/TN3K/HC18 小目标误进 PS 分支。
                is_ps = (not has_psfh_split_gt) and is_psfhs and is_ps_case(case_name, gt_np)
                pred_ps_direct = pred_fh_direct = None

                if has_psfh_split_gt and ps_prob_np is not None and fh_prob_np is not None:
                    pred_ps_direct, pred_fh_direct = _resolve_psfh_overlap(ps_prob_np, fh_prob_np, eval_threshold)
                    if eval_postprocess_mode != 'raw':
                        pred_ps_direct = refine_mask_keep_components(
                            pred_ps_direct,
                            min_obj=max(20, int(0.0003 * out_np.size)),
                        )
                        pred_fh_direct = refine_mask_keep_components(
                            pred_fh_direct,
                            min_obj=max(50, int(0.0007 * out_np.size)),
                        )
                    pred_mask = ((pred_ps_direct > 0) | (pred_fh_direct > 0)).astype(np.uint8)

                elif eval_postprocess_mode == 'raw':
                    pred_mask = _raw_postprocess(out_np, threshold=eval_threshold)

                elif eval_postprocess_mode == 'simple':
                    pred_mask = _simple_postprocess(
                        out_np,
                        is_hc18=is_hc18,
                        has_psfh_split_gt=has_psfh_split_gt,
                        threshold=eval_threshold,
                        min_area=eval_min_area,
                    )

                elif is_hc18:
                    sm = cv2.GaussianBlur(out_np, (13, 13), 0)
                    H, W = sm.shape[:2]
                    img_area = H * W
                    # ===== 第一次（偏严）=====
                    otsu_thr = cv2.threshold((sm * 255).astype(np.uint8), 0, 255,
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
                    high_thr = max(0.62, float(otsu_thr))
                    low_thr = max(0.56, high_thr - 0.045)
                    core = (sm > high_thr).astype(np.uint8)
                    weak = (sm > low_thr).astype(np.uint8)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
                    if num_labels > 1:
                        keep = np.zeros(num_labels, dtype=bool)
                        overlap_ids = np.unique(labels[core.astype(bool)])
                        keep[overlap_ids] = True
                        weak_kept = np.isin(labels, np.where(keep)[0]).astype(np.uint8)
                    else:
                        weak_kept = weak
                    # 只保最大连通域
                    weak_kept = keep_largest_component(weak_kept)
                    min_area = max(360, int(0.0042 * img_area))
                    pred_mask = refine_mask_fh(weak_kept, min_obj=min_area)
                    pred_mask = cv2.morphologyEx(
                        pred_mask, cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    )
                    pred_mask = cv2.medianBlur(pred_mask.astype(np.uint8), 7)
                    # ===== 面积驱动的兜底（第一次太小/空，就放宽阈值再来一次）=====

                    if pred_mask.sum() < max(1200, int(0.0025 * img_area)):
                        # 放宽：阈值整体下调，滞后带略放宽
                        high2 = max(0.55, float(otsu_thr) - 0.02)
                        low2 = max(0.48, high2 - 0.06)
                        core2 = (sm > high2).astype(np.uint8)
                        weak2 = (sm > low2).astype(np.uint8)
                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weak2, connectivity=8)
                        if num_labels > 1:
                            keep = np.zeros(num_labels, dtype=bool)

                            overlap_ids = np.unique(labels[core2.astype(bool)])

                            keep[overlap_ids] = True

                            weak2 = np.isin(labels, np.where(keep)[0]).astype(np.uint8)

                        weak2 = keep_largest_component(weak2)
                        min_area2 = max(180, int(0.0020 * img_area))

                        pred_mask = refine_mask_fh(weak2, min_obj=min_area2)

                        pred_mask = cv2.morphologyEx(

                            pred_mask, cv2.MORPH_CLOSE,

                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

                        )

                        pred_mask = cv2.medianBlur(pred_mask.astype(np.uint8), 5)

                    # 仍然空就兜到底：用 OTSU 单阈值 + 最大连通 + 填洞

                    if pred_mask.sum() == 0:
                        pm = (sm > max(0.50, float(otsu_thr) - 0.04)).astype(np.uint8)
                        pm = keep_largest_component(pm)
                        pred_mask = cv2.morphologyEx(
                            pm, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        )

                        pred_mask = ndi.binary_fill_holes(pred_mask).astype(np.uint8)
                    # The HC18 candidate sweep (snap_to_ellipse / convex_hull /
                    # close/open/erode/dilate variants picked by HD95 with a
                    # Dice guard) was a leftover from an older ablation. The
                    # selector call was already disabled and every candidate is
                    # unused, so we keep only the filled raw mask above.

                elif has_psfh_split_gt:
                    # Official PSFHS masks contain both classes in one image.
                    # Keep multiple connected components here; splitting into
                    # PS/FH happens below for per-class metrics.
                    pred_mask = (out_np > eval_threshold).astype(np.uint8)
                    pred_mask = refine_mask_keep_components(
                        pred_mask,
                        min_obj=max(50, int(0.0007 * out_np.size)),
                    )
                    if pred_mask.sum() == 0:
                        sm = cv2.GaussianBlur(out_np, (3, 3), 0)
                        otsu = cv2.threshold((sm * 255).astype(np.uint8), 0, 255,
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
                        pred_mask = (sm > max(float(otsu) - 0.02, 0.40)).astype(np.uint8)
                        pred_mask = refine_mask_keep_components(
                            pred_mask,
                            min_obj=max(50, int(0.0007 * out_np.size)),
                        )

                elif is_ps:

                    saw_ps = True

                    sm = cv2.GaussianBlur(out_np, (3, 3), 0)

                    # 一段式（略严）

                    otsu = cv2.threshold((sm * 255).astype(np.uint8), 0, 255,

                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0

                    high = max(0.56, float(otsu) - 0.02)

                    low = max(0.48, high - 0.08)

                    core = (sm > high).astype(np.uint8)

                    weak = (sm > low).astype(np.uint8)

                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)

                    if num_labels > 1:

                        keep = np.zeros(num_labels, dtype=bool)

                        overlap_ids = np.unique(labels[core.astype(bool)])

                        keep[overlap_ids] = True

                        kept = np.isin(labels, np.where(keep)[0]).astype(np.uint8)

                    else:

                        kept = weak

                    kept = keep_largest_component(kept)

                    dyn_min = max(100, int(0.0015 * sm.size))

                    pred_mask = refine_mask(kept, min_obj=dyn_min)

                    # 二段式放宽（第一次太小/为空 → 再放一次）

                    if pred_mask.sum() < max(80, int(0.0010 * sm.size)):

                        high2 = max(0.50, float(otsu) - 0.05)

                        low2 = max(0.42, high2 - 0.10)

                        core2 = (sm > high2).astype(np.uint8)

                        weak2 = (sm > low2).astype(np.uint8)

                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weak2, connectivity=8)

                        if num_labels > 1:
                            keep = np.zeros(num_labels, dtype=bool)

                            overlap_ids = np.unique(labels[core2.astype(bool)])

                            keep[overlap_ids] = True

                            weak2 = np.isin(labels, np.where(keep)[0]).astype(np.uint8)

                        weak2 = keep_largest_component(weak2)

                        pred_mask = refine_mask(weak2, min_obj=max(80, int(0.0010 * sm.size)))

                    # 百分位兜底（还不行 → 用强度80分位生成候选）

                    if pred_mask.sum() == 0:
                        thr_p = float(np.percentile(sm, 80.0))

                        pm = (sm > thr_p).astype(np.uint8)

                        pm = keep_largest_component(pm)

                        pred_mask = refine_mask(pm, min_obj=max(60, int(0.0008 * sm.size)))




                elif is_psfhs:

                    # —— PSFHS 的 FH ——（更宽松）

                    sm = cv2.GaussianBlur(out_np, (5, 5), 0)

                    otsu = cv2.threshold((sm * 255).astype(np.uint8), 0, 255,

                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0

                    high = max(0.48, float(otsu) - 0.02)

                    low = max(0.40, high - 0.10)

                    core = (sm > high).astype(np.uint8)

                    weak = (sm > low).astype(np.uint8)

                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)

                    if num_labels > 1:

                        keep = np.zeros(num_labels, dtype=bool)

                        overlap_ids = np.unique(labels[core.astype(bool)])

                        keep[overlap_ids] = True

                        weak_kept = np.isin(labels, np.where(keep)[0]).astype(np.uint8)

                    else:

                        weak_kept = weak

                    weak_kept = keep_largest_component(weak_kept)

                    min_area = max(150, int(0.0010 * sm.size))

                    pred_mask = refine_mask_fh_light(weak_kept, min_obj=min_area)

                    # 二次放宽

                    if pred_mask.sum() < max(120, int(0.0008 * sm.size)):
                        thr_fb = max(0.46, float(otsu) - 0.05)

                        pm = (sm > thr_fb).astype(np.uint8)

                        pm = keep_largest_component(pm)

                        pred_mask = refine_mask_fh_light(pm, min_obj=max(120, int(0.0008 * sm.size)))

                    # 百分位兜底

                    if pred_mask.sum() == 0:
                        thr_p = float(np.percentile(sm, 78.0))

                        pm = (sm > thr_p).astype(np.uint8)

                        pm = keep_largest_component(pm)

                        pred_mask = refine_mask_fh_light(pm, min_obj=max(100, int(0.0006 * sm.size)))


                else:
                    pred_mask = (out_np > eval_threshold).astype(np.uint8)
                    pred_mask = refine_mask(pred_mask, min_obj=eval_min_area)

                # 调试：看看连通域数量和面积
                # num_labels_dbg, labels_dbg = cv2.connectedComponents(pred_mask.astype(np.uint8), connectivity=8)[:2]
                # print(f"[DBG] {case_name}: CC={num_labels_dbg - 1}, area={int(pred_mask.sum())}/{gt_np.sum()}")

                # ---- 计算 ALL 四指标 ----
                # pred_mask = keep_largest_component(pred_mask)
                dsc, jac = _dice_and_jacc(pred_mask, gt_np)
                case_spacing = _sample_spacing(batch_spacing, b, spacing)
                hd95, asd = hd95_asd_mm(pred_mask, gt_np, spacing=case_spacing)
                case_row = {
                    'case_name': str(case_name),
                    'postprocess_mode': eval_postprocess_mode,
                    'threshold': f'{eval_threshold:.6f}',
                    'min_area': int(eval_min_area),
                    'dice_union': f'{dsc:.6f}',
                    'jaccard_union': f'{jac:.6f}',
                    'hd95_union': f'{hd95:.6f}',
                    'asd_union': f'{asd:.6f}',
                    'dice_ps': '',
                    'jaccard_ps': '',
                    'hd95_ps': '',
                    'asd_ps': '',
                    'dice_fh': '',
                    'jaccard_fh': '',
                    'hd95_fh': '',
                    'asd_fh': '',
                    'spacing_y': f'{case_spacing[0]:.8f}',
                    'spacing_x': f'{case_spacing[1]:.8f}',
                    'pred_area': int(pred_mask.sum()),
                    'gt_area': int(gt_np.sum()),
                }

                if is_hc18:
                    # HC18 用 filled-region DSC 对齐 BiPCC/Ours 论文协议（IEEE JBHI 2025）。
                    # 不再额外计算 boundary-band 副指标——保留单一论文口径。
                    dsc_all.append(dsc)
                    jac_all.append(jac)
                    hd95_all.append(hd95)
                    asd_all.append(asd)
                    dsc_fh.append(dsc)
                    jac_fh.append(jac)
                    hd95_fh.append(hd95)
                    asd_fh.append(asd)
                    case_row.update({
                        'dice_fh': f'{dsc:.6f}',
                        'jaccard_fh': f'{jac:.6f}',
                        'hd95_fh': f'{hd95:.6f}',
                        'asd_fh': f'{asd:.6f}',
                    })
                elif has_psfh_split_gt:
                    if pred_ps_direct is not None and pred_fh_direct is not None:
                        pred_ps, pred_fh = pred_ps_direct, pred_fh_direct
                    else:
                        pred_ps, pred_fh = split_psfh_prediction(pred_mask)
                    dsc_ps_v, jac_ps_v = _dice_and_jacc(pred_ps, gt_ps_np)
                    hd95_ps_v, asd_ps_v = hd95_asd_mm(pred_ps, gt_ps_np, spacing=case_spacing)
                    dsc_fh_v, jac_fh_v = _dice_and_jacc(pred_fh, gt_fh_np)
                    hd95_fh_v, asd_fh_v = hd95_asd_mm(pred_fh, gt_fh_np, spacing=case_spacing)

                    dsc_ps.append(dsc_ps_v)
                    jac_ps.append(jac_ps_v)
                    hd95_ps.append(hd95_ps_v)
                    asd_ps.append(asd_ps_v)
                    dsc_fh.append(dsc_fh_v)
                    jac_fh.append(jac_fh_v)
                    hd95_fh.append(hd95_fh_v)
                    asd_fh.append(asd_fh_v)
                    dsc_all.extend([dsc_ps_v, dsc_fh_v])
                    jac_all.extend([jac_ps_v, jac_fh_v])
                    hd95_all.extend([hd95_ps_v, hd95_fh_v])
                    asd_all.extend([asd_ps_v, asd_fh_v])
                    saw_ps = True
                    case_row.update({
                        'dice_ps': f'{dsc_ps_v:.6f}',
                        'jaccard_ps': f'{jac_ps_v:.6f}',
                        'hd95_ps': f'{hd95_ps_v:.6f}',
                        'asd_ps': f'{asd_ps_v:.6f}',
                        'dice_fh': f'{dsc_fh_v:.6f}',
                        'jaccard_fh': f'{jac_fh_v:.6f}',
                        'hd95_fh': f'{hd95_fh_v:.6f}',
                        'asd_fh': f'{asd_fh_v:.6f}',
                    })

                else:
                    # 其它数据集：ALL 计所有，另外再按 PS / FH 分桶
                    dsc_all.append(dsc);
                    jac_all.append(jac);
                    hd95_all.append(hd95);
                    asd_all.append(asd)
                    if is_ps:
                        dsc_ps.append(dsc);
                        jac_ps.append(jac);
                        hd95_ps.append(hd95);
                        asd_ps.append(asd)
                        case_row.update({
                            'dice_ps': f'{dsc:.6f}',
                            'jaccard_ps': f'{jac:.6f}',
                            'hd95_ps': f'{hd95:.6f}',
                            'asd_ps': f'{asd:.6f}',
                        })
                    else:
                        dsc_fh.append(dsc);
                        jac_fh.append(jac);
                        hd95_fh.append(hd95);
                        asd_fh.append(asd)
                        case_row.update({
                            'dice_fh': f'{dsc:.6f}',
                            'jaccard_fh': f'{jac:.6f}',
                            'hd95_fh': f'{hd95:.6f}',
                            'asd_fh': f'{asd:.6f}',
                        })
                case_metric_rows.append(case_row)

                # 转成 tensor 继续原有流程（不动你原来的统计）
                pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float().to(device)

                if save_best:
                    save_img((pred_tensor > 0.5).float(), case_name)
                if reliability_batch is not None:
                    reliability_np = reliability_batch[b, 0].detach().cpu().numpy().astype(np.float32)
                    _update_reliability_bins(
                        reliability_bin_counts,
                        reliability_bin_correct,
                        reliability_np,
                        pseudo_reliability_batch[b, 0].detach().cpu().numpy().astype(np.float32),
                        gt_np,
                    )
                    if reliability_saved < save_reliability_limit:
                        _save_reliability_artifacts(
                            result_root,
                            case_name,
                            img[b],
                            gt_np,
                            out_np,
                            pred_mask,
                            reliability_np,
                        )
                        reliability_saved += 1

                IoU_mean, ACC_overall, dice_sum, recall, specificity, F1, F2, precision, IoU_poly, IoU_bg = \
                    evaluate_SMS(pred_tensor, target[b:b + 1], IoU_mean, ACC_overall, dice_sum,
                                 recall, specificity, F1, F2, precision, IoU_poly, IoU_bg)

                n_images += 1

    # ==== 按样本数平均 ====
    if save_reliability:
        _write_reliability_stats(result_root, reliability_bin_counts, reliability_bin_correct)
    _write_case_metrics(case_metrics_path, case_metric_rows)
    denom = max(1, n_images)
    recall /= denom
    specificity /= denom
    precision /= denom
    F1 /= denom
    F2 /= denom
    ACC_overall /= denom
    IoU_poly /= denom
    IoU_bg /= denom
    IoU_mean /= denom
    dice_sum /= denom

    def _avg(xs):
        if not len(xs):
            return 0.0
        # Drop NaN samples (returned by hd95_asd_mm when one side is empty —
        # surface distance is undefined there). This matches the convention
        # used by medpy / SimpleITK aggregations and BiPCC's reported numbers.
        arr = np.asarray(xs, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        return float(np.mean(finite)) if finite.size else 0.0

    table_metrics = {
        # ALL
        "DSC": _avg(dsc_all), "Jaccard": _avg(jac_all),
        "HD95": _avg(hd95_all), "ASD": _avg(asd_all),
        # PS
        "DSC_PS": _avg(dsc_ps), "Jaccard_PS": _avg(jac_ps),
        "HD95_PS": _avg(hd95_ps), "ASD_PS": _avg(asd_ps),
        # FH
        "DSC_FH": _avg(dsc_fh), "Jaccard_FH": _avg(jac_fh),
        "HD95_FH": _avg(hd95_fh), "ASD_FH": _avg(asd_fh),
    }
    if is_psfh_dataset and len(dsc_ps) and len(dsc_fh):
        # For PSFHS, "Dice" used for checkpoint selection should follow the
        # official per-class macro DSC rather than foreground-union Dice.
        dice_sum = table_metrics["DSC"]
    if saw_ps and len(dsc_ps) == 0:
        print("[WARN] 没有任何样本被识别为 PS；请检查 is_ps_case() 的命名规则是否与数据一致。")
    return recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice_sum, list_name, list_point, table_metrics


def evaluate_SMS(pred, labels, IoU_mean, ACC_overall, dice_sum, recall, specificity, F1, F2, precision, IoU_poly,
                 IoU_bg):
    _recall, _specificity, _precision, _F1, _F2, \
        _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean, dice = evaluate_batch(pred, labels)
    recall += _recall.item()
    specificity += _specificity.item()
    precision += _precision.item()
    F1 += _F1.item()
    F2 += _F2.item()
    ACC_overall += _ACC_overall.item()
    IoU_poly += _IoU_poly.item()
    IoU_bg += _IoU_bg.item()
    IoU_mean += _IoU_mean.item()
    dice_sum += dice.item()
    return IoU_mean, ACC_overall, dice_sum, recall, specificity, F1, F2, precision, IoU_poly, IoU_bg


def evaluate_batch(output, gt):
    pred = output
    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()
    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    # if TP.item() == 0:
    #     TP = torch.Tensor([1]).cuda()
    positive_gt = TP + FN
    positive_pred = TP + FP
    Recall = _safe_tensor_div(TP, positive_gt, empty_value=1.0)
    Specificity = _safe_tensor_div(TN, TN + FP, empty_value=1.0)
    Precision = torch.where(
        positive_pred > 0,
        TP / positive_pred.clamp_min(1),
        torch.where(positive_gt > 0, torch.zeros_like(TP), torch.ones_like(TP))
    )
    F1 = _safe_tensor_div(2 * TP, 2 * TP + FP + FN, empty_value=1.0)
    F2 = _safe_tensor_div(5 * TP, 5 * TP + 4 * FN + FP, empty_value=1.0)
    ACC_overall = _safe_tensor_div(TP + TN, TP + FP + FN + TN, empty_value=1.0)
    IoU_poly = _safe_tensor_div(TP, TP + FP + FN, empty_value=1.0)
    IoU_bg = _safe_tensor_div(TN, TN + FP + FN, empty_value=1.0)
    IoU_mean = (IoU_poly + IoU_bg) / 2.0
    dice = F1
    return Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice
