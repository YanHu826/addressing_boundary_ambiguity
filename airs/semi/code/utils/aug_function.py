"""Augmentation utilities used by the semi-supervised pipeline.

历史上本文件来自 CCT (https://github.com/yassouali/CCT) 的实现，
携带了 7 个未被项目消费的扰动 decoder（MainDecoder/DropOutDecoder/
FeatureDropDecoder/FeatureNoiseDecoder/VATDecoder/ContextMaskingDecoder/
ObjectMaskingDecoder）与若干辅助层。当前 main.py / semi_self.py 仅用到
SORDecoder（其内部依赖 guided_cutout），其余符号已删除以避免误导后续
ablation。 -- cleanup 2026-05
"""

import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def guided_cutout(output, upscale, resize, erase=0.4, use_dropout=False):
    """Erase a random rectangular patch inside each foreground object.

    upscale 仅保留作为兼容参数，函数本身按 ``resize`` 直接做最近邻插值。
    """
    if len(output.shape) == 3:
        masks = (output > 0.5).float()
    elif output.size(1) == 1:
        prob = output
        if prob.detach().min().item() < 0.0 or prob.detach().max().item() > 1.0:
            prob = torch.sigmoid(prob)
        masks = (prob.squeeze(1) > 0.5).float()
    else:
        masks = (output.argmax(1) > 0).float()

    masks = F.interpolate(masks.unsqueeze(1), size=resize, mode='nearest').squeeze(1)

    if use_dropout:
        p_drop = random.randint(3, 6) / 10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        erased = False
        try:  # OpenCV 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:  # OpenCV 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue
            min_w, min_h, bb_w, bb_h = cv2.boundingRect(contour)
            if bb_w <= 1 or bb_h <= 1:
                continue
            erase_w = max(1, int(bb_w * erase))
            erase_h = max(1, int(bb_h * erase))
            rnd_start_w = random.randint(0, max(0, bb_w - erase_w))
            rnd_start_h = random.randint(0, max(0, bb_h - erase_h))
            h_start, h_end = min_h + rnd_start_h, min_h + rnd_start_h + erase_h
            w_start, w_end = min_w + rnd_start_w, min_w + rnd_start_w + erase_w
            mask_ones[h_start:h_end, w_start:w_end] = 0
            erased = True

        if not erased and mask_np.any():
            ys, xs = np.where(mask_np > 0)
            min_h, max_h = ys.min(), ys.max() + 1
            min_w, max_w = xs.min(), xs.max() + 1
            bb_h = max_h - min_h
            bb_w = max_w - min_w
            erase_h = max(1, int(bb_h * erase))
            erase_w = max(1, int(bb_w * erase))
            rnd_start_h = random.randint(0, max(0, bb_h - erase_h))
            rnd_start_w = random.randint(0, max(0, bb_w - erase_w))
            h_start, h_end = min_h + rnd_start_h, min_h + rnd_start_h + erase_h
            w_start, w_end = min_w + rnd_start_w, min_w + rnd_start_w + erase_w
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)


class SORDecoder(nn.Module):
    """Structure-Oriented Residual perturbation (论文第 3.5 节).

    使用主头预测的 soft mask 触发 guided cutout，按照前景置信度对原始特征
    做软门控的擦除，并以 ``residual_ratio`` 与原始特征做残差融合，避免
    在小目标 / 低置信度场景下退化为 no-op。
    """

    def __init__(
        self,
        erase=0.4,
        min_keep_ratio=0.7,
        residual_ratio=0.75,
        min_fg_ratio=0.01,
        full_gate_fg_ratio=0.08,
        min_confidence=0.65,
    ):
        super(SORDecoder, self).__init__()
        self.erase = erase
        self.min_keep_ratio = min_keep_ratio
        self.residual_ratio = residual_ratio
        self.min_fg_ratio = min_fg_ratio
        self.full_gate_fg_ratio = max(full_gate_fg_ratio, min_fg_ratio + 1e-6)
        self.min_confidence = min_confidence

    def forward(self, x, pred=None):
        if pred is None:
            return x

        prob = pred.detach().float()
        if prob.dim() == 3:
            prob = prob.unsqueeze(1)
        if prob.min().item() < 0.0 or prob.max().item() > 1.0:
            prob = torch.sigmoid(prob)
        prob = prob.clamp(0.0, 1.0)

        resize = (x.size(2), x.size(3))
        prob = F.interpolate(prob, size=resize, mode='bilinear', align_corners=False)
        hard_mask = (prob > 0.5).float()
        fg_pixels = hard_mask.sum(dim=(1, 2, 3), keepdim=True)
        fg_ratio = fg_pixels / float(hard_mask.shape[2] * hard_mask.shape[3])
        fg_conf = (prob * hard_mask).sum(dim=(1, 2, 3), keepdim=True) / fg_pixels.clamp_min(1.0)

        # Small or uncertain targets should stay close to the original feature.
        fg_gate = ((fg_ratio - self.min_fg_ratio) / (self.full_gate_fg_ratio - self.min_fg_ratio)).clamp(0.0, 1.0)
        conf_gate = ((fg_conf - self.min_confidence) / max(1e-6, 1.0 - self.min_confidence)).clamp(0.0, 1.0)
        gate = fg_gate * conf_gate

        maskcut = guided_cutout(prob, upscale=1, erase=self.erase, resize=resize).to(x.dtype)
        softened_mask = 1.0 - gate * (1.0 - maskcut) * (1.0 - self.min_keep_ratio)
        perturbed = x * softened_mask
        return self.residual_ratio * x + (1.0 - self.residual_ratio) * perturbed
