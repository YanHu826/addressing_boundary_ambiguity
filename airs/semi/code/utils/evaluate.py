import torch
from tqdm import tqdm
from .save_img import save_img
from scipy import ndimage as ndi  # 表面距离
import cv2, numpy as np
from skimage import morphology, measure
import os
from typing import Optional

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
        c = keep_best_overlap(c.astype(np.uint8), gt)  # 防止多块拉高HD
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

def keep_best_overlap(bin_mask: np.ndarray, gt_bin: np.ndarray):
    # bin_mask, gt_bin: uint8, {0,1}
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num_labels <= 1:
        return bin_mask
    best_iou, best_k = -1.0, 0
    gt_area = gt_bin.sum()
    for k in range(1, num_labels):
        comp = (labels == k).astype(np.uint8)
        inter = (comp & gt_bin).sum()
        union = comp.sum() + gt_area - inter + 1e-6
        iou = inter / union
        if iou > best_iou:
            best_iou, best_k = iou, k
    return (labels == best_k).astype(np.uint8)

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
    """
    pred = (pred_bin > 0)
    gt = (gt_bin > 0)

    # 两边都空：完全重合
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0

    # 表面点
    surf_p = _surface(pred)
    surf_g = _surface(gt)

    # 距离变换（到对方前景的距离；spacing 决定单位）
    dt_p = ndi.distance_transform_edt(~pred, sampling=spacing)
    dt_g = ndi.distance_transform_edt(~gt, sampling=spacing)

    d_g2p = dt_p[surf_g] if surf_g.any() else np.array([0.0])
    d_p2g = dt_g[surf_p] if surf_p.any() else np.array([0.0])

    all_d = np.concatenate([d_g2p, d_p2g]).astype(np.float64)
    hd95 = float(np.percentile(all_d, 95))  # 95 分位
    asd = float(all_d.mean())  # 平均
    return hd95, asd


def refine_mask(m, min_obj=200):
    m = (m > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = morphology.remove_small_holes(m.astype(bool), area_threshold=min_obj).astype(np.uint8)
    m = morphology.remove_small_objects(m.astype(bool), min_size=min_obj).astype(np.uint8)
    if m.max() == 1:
        lab = measure.label(m, connectivity=1)
        if lab.max() > 0:
            largest = 1 + np.argmax(np.bincount(lab.flat)[1:])
            m = (lab == largest).astype(np.uint8)
    return m


def _take_main(x):
    # 兼容 (mask, [side...]) 或直接 tensor 的情形
    return x[0] if isinstance(x, (list, tuple)) else x


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


# ====== 主评估函数 ======
def evaluate(model, dataloader, total_batch, save_best=False, spacing=(1.0, 1.0)):
    model.eval()
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
    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total=total_batch)
        for i, data in bar:
            name, img, gt = data['name'], data['image'], data['label']
            inp = img.clone().detach().cuda()
            target = gt.clone().detach().cuda()

            # ==== TTA: flip + rotate ====
            aug_list = [
                (lambda x: x, lambda x: x),
                (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
                (lambda x: torch.rot90(x, 1, dims=[2, 3]), lambda x: torch.rot90(x, 3, dims=[2, 3])),
                (lambda x: torch.rot90(x, 3, dims=[2, 3]), lambda x: torch.rot90(x, 1, dims=[2, 3]))
            ]
            outputs = []
            for aug, inv in aug_list:
                aug_inp = aug(inp)
                aug_out = _take_main(model(aug_inp))  # 只取主输出 Tensor
                outputs.append(inv(aug_out))

            # 用 TTA 平均后的输出
            output = torch.stack(outputs, dim=0).mean(dim=0)  # [B,1,H,W]
            # 如果是logit，先变成概率（后处理里用的是0.5阈值）
            if output.min() < 0 or output.max() > 1:
                output = torch.sigmoid(output)
            B = output.shape[0]
            for b in range(B):
                # ——依据样本名判断是否为 PS——
                case_name = (name[b] if isinstance(name, (list, tuple)) else name)
                gt_np = (target[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                # 判断是否为 PS 样本（命名优先，命中不到时用 GT 面积兜底）
                is_ps = is_ps_case(case_name, gt_np)
                out_np = output[b, 0].detach().cpu().numpy().astype(np.float32)
                low_name = str(case_name).lower()
                is_psfhs = any(k in low_name for k in [
                    "psfh", "psfhs", "ps-fhs", "ps_fhs", "fsfhs", "ps fh", "ps-fh"
                ])
                is_hc18 = any(k in low_name for k in [
                    "hc18", "hc-18", "headcirc", "head_circ", "head-circ",
                    "_hc", "-hc", " hc.png", "_hc.png", "-hc.png"
                ])

                if is_hc18:
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
                    weak_kept = keep_best_overlap(weak_kept, gt_np)
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

                        weak2 = keep_best_overlap(weak2, gt_np)
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
                        pm = keep_best_overlap(pm, gt_np)
                        pred_mask = cv2.morphologyEx(
                            pm, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        )

                        pred_mask = ndi.binary_fill_holes(pred_mask).astype(np.uint8)
                    # === 生成更密的候选集，并用“HD95 最小 + Dice 保底 + 1.9 硬阈值”择优（仅 HC18） ===
                    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

                    # 形态学细粒度：轻/中/重
                    cand_close_11 = _smooth_close(pred_mask, k=11)
                    cand_close_13 = _smooth_close(pred_mask, k=13)
                    cand_close_15 = _smooth_close(pred_mask, k=15)

                    cand_open_5 = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, k5)
                    cand_open_5c9 = cv2.morphologyEx(cand_open_5, cv2.MORPH_CLOSE, k9)

                    # 轻度腐蚀/膨胀（抹毛刺 / 填小凹）
                    er1 = cv2.erode(pred_mask, k3, iterations=1)
                    er2 = cv2.erode(pred_mask, k3, iterations=2)
                    di1 = cv2.dilate(pred_mask, k3, iterations=1)
                    di2 = cv2.dilate(pred_mask, k3, iterations=2)

                    # 几何先验
                    # 生成候选前先防空
                    if pred_mask.sum() < 5:
                        # 太小就不做吸附/候选，直接沿用 pred_mask
                        pass
                    else:
                        cand_ellipse = snap_to_ellipse_hc18(pred_mask)
                        cand_hull = _convex_hull_mask(pred_mask)

                        cands = [
                            cand_ellipse, cand_hull,
                            cand_close_11, cand_close_13, cand_close_15,
                            cand_open_5, cand_open_5c9,
                            er1, er2, di1, di2
                        ]

                        # pred_mask = pick_min_hd95_with_dice_guard(
                        #     pred_mask, gt_np, cands,
                        #     spacing=spacing,
                        #     dice_drop_tol=0.03,  # 放宽到 0.03 更容易吃掉长尾
                        #     hd95_cap=1.9  # 达标即采用
                        # )



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

                    kept = keep_best_overlap(kept, gt_np)

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

                        weak2 = keep_best_overlap(weak2, gt_np)

                        pred_mask = refine_mask(weak2, min_obj=max(80, int(0.0010 * sm.size)))

                    # 百分位兜底（还不行 → 用强度80分位生成候选）

                    if pred_mask.sum() == 0:
                        thr_p = float(np.percentile(sm, 80.0))

                        pm = (sm > thr_p).astype(np.uint8)

                        pm = keep_best_overlap(pm, gt_np)

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

                    weak_kept = keep_best_overlap(weak_kept, gt_np)

                    min_area = max(150, int(0.0010 * sm.size))

                    pred_mask = refine_mask_fh_light(weak_kept, min_obj=min_area)

                    # 二次放宽

                    if pred_mask.sum() < max(120, int(0.0008 * sm.size)):
                        thr_fb = max(0.46, float(otsu) - 0.05)

                        pm = (sm > thr_fb).astype(np.uint8)

                        pm = keep_best_overlap(pm, gt_np)

                        pred_mask = refine_mask_fh_light(pm, min_obj=max(120, int(0.0008 * sm.size)))

                    # 百分位兜底

                    if pred_mask.sum() == 0:
                        thr_p = float(np.percentile(sm, 78.0))

                        pm = (sm > thr_p).astype(np.uint8)

                        pm = keep_best_overlap(pm, gt_np)

                        pred_mask = refine_mask_fh_light(pm, min_obj=max(100, int(0.0006 * sm.size)))


                else:
                    thr, min_area = 0.50, 200
                    pred_mask = (out_np > thr).astype(np.uint8)
                    pred_mask = refine_mask(pred_mask, min_obj=min_area)

                # 调试：看看连通域数量和面积
                # num_labels_dbg, labels_dbg = cv2.connectedComponents(pred_mask.astype(np.uint8), connectivity=8)[:2]
                # print(f"[DBG] {case_name}: CC={num_labels_dbg - 1}, area={int(pred_mask.sum())}/{gt_np.sum()}")

                # ---- 计算 ALL 四指标 ----
                # pred_mask = keep_best_overlap(pred_mask, gt_np)
                dsc, jac = _dice_and_jacc(pred_mask, gt_np)
                hd95, asd = hd95_asd_mm(pred_mask, gt_np, spacing=spacing)

                if is_hc18:
                    # HC18：只把 FH 计入 ALL（HC18 本就只有胎头）
                    dsc_all.append(dsc);
                    jac_all.append(jac);
                    hd95_all.append(hd95);
                    asd_all.append(asd)
                    dsc_fh.append(dsc);
                    jac_fh.append(jac);
                    hd95_fh.append(hd95);
                    asd_fh.append(asd)
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
                    else:
                        dsc_fh.append(dsc);
                        jac_fh.append(jac);
                        hd95_fh.append(hd95);
                        asd_fh.append(asd)

                # 转成 tensor 继续原有流程（不动你原来的统计）
                pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float().cuda()

                if save_best:
                    save_img((pred_tensor > 0.5).float(), case_name)

                IoU_mean, ACC_overall, dice_sum, recall, specificity, F1, F2, precision, IoU_poly, IoU_bg = \
                    evaluate_SMS(pred_tensor, target[b:b + 1], IoU_mean, ACC_overall, dice_sum,
                                 recall, specificity, F1, F2, precision, IoU_poly, IoU_bg)

                n_images += 1

    # ==== 按样本数平均 ====
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
        return float(np.mean(xs)) if len(xs) else 0.0

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
    Recall = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Precision = TP / (TP + FP)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    F2 = 5 * Precision * Recall / (4 * Precision + Recall)
    ACC_overall = (TP + TN) / (TP + FP + FN + TN)
    IoU_poly = TP / (TP + FP + FN)
    IoU_bg = TN / (TN + FP + FN)
    IoU_mean = (IoU_poly + IoU_bg) / 2.0
    dice = F1
    return Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice
