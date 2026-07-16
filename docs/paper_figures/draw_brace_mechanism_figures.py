#!/usr/bin/env python3
"""Draw reproducible BRACE mechanism figures for the KBS manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageFilter
import numpy as np


RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
}
plt.rcParams.update(RC_PARAMS)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "docs" / "kbs_submission_package" / "source"
ASSETS_DIR = PROJECT_ROOT / "assets"
INSET_DIR = PROJECT_ROOT / "docs" / "paper_figures" / "insets" / "real_semirun"

CANVAS_W = 1600
CANVAS_H = 900
PREVIEW_PPI = 100

COLORS = {
    "text": "#1F2328",
    "muted": "#5F6B76",
    "muted_fill": "#F7F8FA",
    "teacher": "#2F6FB3",
    "teacher_fill": "#EEF6FF",
    "fbwa": "#2F7D45",
    "fbwa_fill": "#F2FAF3",
    "sorp": "#7A4AA0",
    "sorp_fill": "#F7F1FB",
    "rct": "#C97921",
    "rct_fill": "#FFF7EC",
    "loss": "#B73A3A",
    "loss_fill": "#FFF1F1",
    "line": "#5F6B76",
}

PATHS = {
    "busi_image": INSET_DIR / "busi" / "busi_01_benign_434_crop_image.png",
    "busi_gt": INSET_DIR / "busi" / "busi_01_benign_434_crop_gt.png",
    "busi_pred": INSET_DIR / "busi" / "busi_01_benign_434_crop_pred.png",
    "busi_overlay": INSET_DIR / "busi" / "busi_01_benign_434_crop_overlay.png",
    "tn3k_image": INSET_DIR / "tn3k" / "tn3k_01_2301_crop_image.png",
    "tn3k_gt": INSET_DIR / "tn3k" / "tn3k_01_2301_crop_gt.png",
    "tn3k_pred": INSET_DIR / "tn3k" / "tn3k_01_2301_crop_pred.png",
    "tn3k_overlay": INSET_DIR / "tn3k" / "tn3k_01_2301_crop_overlay.png",
}


def new_canvas():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=PREVIEW_PPI)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    label="",
    *,
    ec=None,
    fc="white",
    lw=1.4,
    radius=8,
    fontsize=9,
    weight="normal",
    color=None,
    zorder=2,
    align="center",
):
    ec = ec or COLORS["line"]
    color = color or COLORS["text"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            zorder=zorder + 1,
            linespacing=1.12,
            multialignment=align,
        )
    return patch


def straight_arrow(
    ax,
    start,
    end,
    *,
    color=None,
    lw=1.6,
    dashed=False,
    zorder=5,
    mutation_scale=14,
    connectionstyle=None,
):
    color = color or COLORS["line"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=5,
        shrinkB=5,
        zorder=zorder,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)
    return patch


def elbow_arrow(ax, points, *, color=None, lw=1.6, dashed=False, zorder=5):
    color = color or COLORS["line"]
    if len(points) < 2:
        return
    linestyle = (0, (4, 3)) if dashed else "solid"
    for p0, p1 in zip(points[:-2], points[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, linestyle=linestyle, zorder=zorder)
    straight_arrow(ax, points[-2], points[-1], color=color, lw=lw, dashed=dashed, zorder=zorder)


def label(ax, x, y, text, *, fontsize=9, color=None, weight="normal", ha="center", va="center", zorder=8):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color or COLORS["text"],
        fontweight=weight,
        zorder=zorder,
        linespacing=1.1,
    )


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def image_card(
    ax,
    path: Path,
    x,
    y,
    w,
    h,
    label_text="",
    *,
    ec=None,
    fc="white",
    label_size=8,
    image_pad=9,
    label_h=26,
    zorder=3,
):
    ec = ec or COLORS["line"]
    rounded_box(ax, x, y, w, h, ec=ec, fc=fc, lw=1.3, radius=6, zorder=zorder)
    img = load_image(path)
    img_y0 = y + image_pad
    img_y1 = y + h - image_pad - (label_h if label_text else 0)
    ax.imshow(img, extent=(x + image_pad, x + w - image_pad, img_y1, img_y0), zorder=zorder + 1)
    if label_text:
        label(ax, x + w / 2, y + h - label_h / 2, label_text, fontsize=label_size, color=COLORS["text"], zorder=zorder + 2)


def tiny_map(ax, x, y, w, h, text="", *, ec=None, fc="white", color=None, grid=False, zorder=3):
    ec = ec or COLORS["line"]
    rounded_box(ax, x, y, w, h, text, ec=ec, fc=fc, lw=1.25, radius=5, fontsize=8, color=color, zorder=zorder)
    if grid:
        for i in range(1, 4):
            ax.plot([x + w * i / 4, x + w * i / 4], [y + 6, y + h - 6], color=ec, lw=0.45, alpha=0.35, zorder=zorder + 1)
            ax.plot([x + 6, x + w - 6], [y + h * i / 4, y + h * i / 4], color=ec, lw=0.45, alpha=0.35, zorder=zorder + 1)


def feature_strip(ax, x, y, w, h, *, ec=None, fc=None, n=4, label_text=""):
    ec = ec or COLORS["teacher"]
    fc = fc or COLORS["teacher_fill"]
    gap = w * 0.08
    block_w = (w - gap * (n - 1)) / n
    for i in range(n):
        height = h * (0.55 + 0.12 * (i % 2))
        yy = y + (h - height) / 2
        ax.add_patch(Rectangle((x + i * (block_w + gap), yy), block_w, height, ec=ec, fc=fc, lw=1.1, zorder=4))
    if label_text:
        label(ax, x + w / 2, y + h + 16, label_text, fontsize=8, color=COLORS["text"])


def reliability_array(mask_path: Path, size=220):
    mask = Image.open(mask_path).convert("L").resize((size, size))
    blur = mask.filter(ImageFilter.GaussianBlur(radius=13))
    arr = np.asarray(blur, dtype=float) / 255.0
    yy, xx = np.mgrid[0:size, 0:size]
    texture = 0.08 * np.sin(xx / 15.0) + 0.06 * np.cos(yy / 18.0)
    rel = np.clip(arr * (0.86 + texture), 0, 1)
    rgba = plt.get_cmap("YlOrBr")(rel)
    rgba[..., 3] = np.where(arr > 0.04, 1.0, 0.0)
    bg = np.zeros((size, size, 4))
    bg[..., :3] = 0.02
    bg[..., 3] = 1.0
    out = bg.copy()
    out[rgba[..., 3] > 0] = rgba[rgba[..., 3] > 0]
    return out


def heatmap_card(ax, mask_path: Path, x, y, w, h, label_text="", *, ec=None, label_size=8):
    ec = ec or COLORS["rct"]
    rounded_box(ax, x, y, w, h, ec=ec, fc="white", lw=1.3, radius=6, zorder=3)
    label_h = 24 if label_text else 0
    arr = reliability_array(mask_path)
    ax.imshow(arr, extent=(x + 9, x + w - 9, y + h - 9 - label_h, y + 9), zorder=4)
    if label_text:
        label(ax, x + w / 2, y + h - label_h / 2, label_text, fontsize=label_size)


def pill(ax, x, y, w, h, text, *, ec, fc, fontsize=8, weight="normal"):
    rounded_box(ax, x, y, w, h, text, ec=ec, fc=fc, lw=1.25, radius=h / 2, fontsize=fontsize, weight=weight)


def save_all(fig, stem: str, asset_name: str | None = None):
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    pixels_per_inch = PREVIEW_PPI
    tiff_dpi = 600
    png_path = SOURCE_DIR / f"{stem}.png"
    fig.savefig(png_path, dpi=pixels_per_inch, facecolor="white")
    fig.savefig(SOURCE_DIR / f"{stem}.svg", facecolor="white")
    fig.savefig(SOURCE_DIR / f"{stem}.pdf", facecolor="white")
    fig.savefig(
        SOURCE_DIR / f"{stem}.tiff",
        dpi=tiff_dpi,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    if asset_name:
        fig.savefig(ASSETS_DIR / asset_name, dpi=pixels_per_inch, facecolor="white")
    plt.close(fig)


def draw_overall_framework():
    fig, ax = new_canvas()

    # Inputs
    image_card(ax, PATHS["busi_overlay"], 48, 112, 155, 158, "x_l, y_l", ec=COLORS["muted"], label_size=8)
    image_card(ax, PATHS["tn3k_image"], 48, 580, 155, 158, "x_u", ec=COLORS["muted"], label_size=8)
    elbow_arrow(ax, [(203, 191), (252, 191), (252, 430), (292, 430)], color=COLORS["line"])
    elbow_arrow(ax, [(203, 659), (252, 659), (252, 500), (292, 500)], color=COLORS["line"])

    # Student backbone
    rounded_box(ax, 292, 380, 132, 140, "Shared\nencoder", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=11, weight="bold")
    rounded_box(ax, 485, 260, 116, 70, "Main\ndecoder", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=9)
    rounded_box(ax, 485, 415, 116, 70, "Boundary\nhead", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=9)
    rounded_box(ax, 485, 570, 116, 70, "Auxiliary\ndecoder", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=9)
    elbow_arrow(ax, [(424, 430), (455, 430), (455, 295), (485, 295)], color=COLORS["line"])
    straight_arrow(ax, (424, 450), (485, 450), color=COLORS["line"])
    elbow_arrow(ax, [(424, 500), (455, 500), (455, 605), (485, 605)], color=COLORS["line"])

    image_card(ax, PATHS["busi_pred"], 652, 240, 104, 104, "$P_u$", ec=COLORS["muted"], label_size=8, image_pad=7, label_h=20)
    image_card(ax, PATHS["busi_gt"], 652, 397, 104, 104, "$B(P_u)$", ec=COLORS["muted"], label_size=8, image_pad=7, label_h=20)
    image_card(ax, PATHS["busi_pred"], 652, 555, 104, 104, "$P_u^a$", ec=COLORS["muted"], label_size=8, image_pad=7, label_h=20)
    straight_arrow(ax, (601, 295), (652, 295), color=COLORS["line"])
    straight_arrow(ax, (601, 450), (652, 450), color=COLORS["line"])
    straight_arrow(ax, (601, 605), (652, 605), color=COLORS["line"])

    # Teacher path
    rounded_box(ax, 845, 105, 108, 72, "EMA\nteacher", ec=COLORS["teacher"], fc=COLORS["teacher_fill"], fontsize=10, weight="bold", color=COLORS["teacher"])
    image_card(ax, PATHS["tn3k_overlay"], 1010, 80, 126, 126, "$T_u$", ec=COLORS["teacher"], fc=COLORS["teacher_fill"], label_size=8, image_pad=8, label_h=22)
    straight_arrow(ax, (953, 141), (1010, 141), color=COLORS["teacher"])
    elbow_arrow(ax, [(756, 292), (792, 292), (792, 141), (845, 141)], color=COLORS["teacher"], dashed=True)
    label(ax, 780, 118, "EMA", fontsize=8, color=COLORS["teacher"], weight="bold")

    # SORP path
    rounded_box(ax, 790, 278, 330, 112, ec=COLORS["sorp"], fc=COLORS["sorp_fill"], lw=1.6, radius=10)
    tiny_map(ax, 815, 310, 76, 48, "$M_{sorp}$", ec=COLORS["sorp"], fc="white", color=COLORS["sorp"], grid=True)
    pill(ax, 924, 309, 74, 50, "perturb", ec=COLORS["sorp"], fc="white", fontsize=8)
    tiny_map(ax, 1030, 310, 66, 48, r"$\tilde{F}$", ec=COLORS["sorp"], fc="white", color=COLORS["sorp"], grid=True)
    elbow_arrow(ax, [(756, 292), (790, 292), (790, 334), (815, 334)], color=COLORS["sorp"])
    straight_arrow(ax, (891, 334), (924, 334), color=COLORS["sorp"])
    straight_arrow(ax, (998, 334), (1030, 334), color=COLORS["sorp"])
    elbow_arrow(ax, [(1096, 334), (1142, 334), (1142, 605), (756, 605)], color=COLORS["sorp"])

    # RCT path
    rounded_box(ax, 1180, 248, 210, 244, ec=COLORS["rct"], fc=COLORS["rct_fill"], lw=1.6, radius=10)
    pill(ax, 1204, 276, 78, 38, "entropy", ec=COLORS["rct"], fc="white", fontsize=8)
    pill(ax, 1204, 330, 78, 38, "boundary", ec=COLORS["rct"], fc="white", fontsize=8)
    pill(ax, 1204, 384, 78, 38, "agreement", ec=COLORS["rct"], fc="white", fontsize=8)
    heatmap_card(ax, PATHS["tn3k_pred"], 1302, 314, 68, 86, "$R_u$", ec=COLORS["rct"], label_size=8)
    straight_arrow(ax, (1282, 295), (1302, 350), color=COLORS["rct"])
    straight_arrow(ax, (1282, 349), (1302, 357), color=COLORS["rct"])
    straight_arrow(ax, (1282, 403), (1302, 364), color=COLORS["rct"])
    elbow_arrow(ax, [(1136, 143), (1164, 143), (1164, 295), (1204, 295)], color=COLORS["teacher"])
    elbow_arrow(ax, [(756, 292), (1164, 292), (1164, 349), (1204, 349)], color=COLORS["rct"])
    elbow_arrow(ax, [(756, 605), (1152, 605), (1152, 403), (1204, 403)], color=COLORS["rct"])

    # FBWA path
    rounded_box(ax, 790, 545, 430, 150, ec=COLORS["fbwa"], fc=COLORS["fbwa_fill"], lw=1.6, radius=10)
    tiny_map(ax, 820, 582, 76, 44, "$Z_l$", ec=COLORS["fbwa"], fc="white", color=COLORS["fbwa"], grid=True)
    tiny_map(ax, 820, 638, 76, 44, "$Z_u$", ec=COLORS["fbwa"], fc="white", color=COLORS["fbwa"], grid=True)
    pill(ax, 932, 600, 82, 48, "$D$", ec=COLORS["fbwa"], fc="white", fontsize=12, weight="bold")
    rounded_box(ax, 1050, 600, 122, 48, "$\\mathcal{L}_{fbwa}$", ec=COLORS["fbwa"], fc="white", fontsize=10, color=COLORS["fbwa"], weight="bold")
    elbow_arrow(ax, [(756, 450), (776, 450), (776, 604), (820, 604)], color=COLORS["fbwa"])
    elbow_arrow(ax, [(756, 450), (776, 450), (776, 660), (820, 660)], color=COLORS["fbwa"])
    straight_arrow(ax, (896, 604), (932, 618), color=COLORS["fbwa"])
    straight_arrow(ax, (896, 660), (932, 630), color=COLORS["fbwa"])
    straight_arrow(ax, (1014, 624), (1050, 624), color=COLORS["fbwa"])

    # Objective
    rounded_box(ax, 1430, 270, 128, 360, ec=COLORS["loss"], fc=COLORS["loss_fill"], lw=1.7, radius=10)
    label(ax, 1494, 315, "$\\mathcal{L}$", fontsize=14, color=COLORS["loss"], weight="bold")
    rounded_box(ax, 1446, 358, 96, 62, "$\\mathcal{L}_s$", ec=COLORS["loss"], fc="white", fontsize=12, color=COLORS["loss"], weight="bold")
    rounded_box(ax, 1446, 462, 96, 74, "$R_u \\odot \\mathcal{L}_u$", ec=COLORS["loss"], fc="white", fontsize=10, color=COLORS["loss"], weight="bold")
    rounded_box(ax, 1446, 576, 96, 34, "$\\lambda\\mathcal{L}_{fbwa}$", ec=COLORS["loss"], fc="white", fontsize=9, color=COLORS["loss"], weight="bold")
    elbow_arrow(ax, [(1370, 357), (1410, 357), (1410, 499), (1430, 499)], color=COLORS["rct"])
    elbow_arrow(ax, [(1172, 624), (1330, 624), (1330, 593), (1430, 593)], color=COLORS["fbwa"])
    elbow_arrow(ax, [(1494, 630), (1494, 775), (355, 775), (355, 520)], color=COLORS["loss"], dashed=True)
    label(ax, 880, 797, "optimization feedback", fontsize=8, color=COLORS["loss"])

    save_all(fig, "user_overall_framework_notitle", "framework.png")


def draw_fbwa_module():
    fig, ax = new_canvas()

    image_card(ax, PATHS["tn3k_image"], 115, 110, 150, 140, "$x_l$", ec=COLORS["muted"], label_size=9)
    image_card(ax, PATHS["tn3k_overlay"], 375, 110, 150, 140, "$y_l$", ec=COLORS["muted"], label_size=9)
    image_card(ax, PATHS["tn3k_image"], 1075, 110, 150, 140, "$x_u$", ec=COLORS["muted"], label_size=9)
    image_card(ax, PATHS["tn3k_pred"], 1335, 110, 150, 140, "$P_u$", ec=COLORS["muted"], label_size=9)

    rounded_box(ax, 100, 330, 180, 64, "$A(F_l)$", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=13)
    rounded_box(ax, 360, 330, 180, 64, "$B(y_l)$", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=13)
    rounded_box(ax, 1060, 330, 180, 64, "$A(F_u)$", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=13)
    rounded_box(ax, 1320, 330, 180, 64, "$B(P_u)$", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=13)

    straight_arrow(ax, (190, 250), (190, 330), color=COLORS["line"])
    straight_arrow(ax, (450, 250), (450, 330), color=COLORS["line"])
    straight_arrow(ax, (1150, 250), (1150, 330), color=COLORS["line"])
    straight_arrow(ax, (1410, 250), (1410, 330), color=COLORS["line"])

    rounded_box(ax, 180, 510, 360, 72, "$Z_l = [A(F_l), B(y_l)]$", ec=COLORS["fbwa"], fc=COLORS["fbwa_fill"], fontsize=15, color=COLORS["text"], weight="bold")
    rounded_box(ax, 1060, 510, 360, 72, "$Z_u = [A(F_u), B(P_u)]$", ec=COLORS["fbwa"], fc=COLORS["fbwa_fill"], fontsize=15, color=COLORS["text"], weight="bold")
    elbow_arrow(ax, [(190, 394), (190, 465), (360, 465), (360, 510)], color=COLORS["line"])
    elbow_arrow(ax, [(450, 394), (450, 465), (360, 465), (360, 510)], color=COLORS["line"])
    elbow_arrow(ax, [(1150, 394), (1150, 465), (1240, 465), (1240, 510)], color=COLORS["line"])
    elbow_arrow(ax, [(1410, 394), (1410, 465), (1240, 465), (1240, 510)], color=COLORS["line"])

    rounded_box(ax, 640, 610, 320, 135, ec=COLORS["fbwa"], fc=COLORS["fbwa_fill"], lw=1.7, radius=10)
    label(ax, 800, 645, "Shared critic $D$", fontsize=16, color=COLORS["fbwa"], weight="bold")
    ax.plot([675, 925], [670, 670], color=COLORS["fbwa"], lw=0.9, linestyle=(0, (5, 4)), alpha=0.8)
    label(ax, 800, 696, "Wasserstein distance", fontsize=10)
    label(ax, 800, 724, "gradient penalty", fontsize=10)
    elbow_arrow(ax, [(540, 546), (610, 546), (610, 666), (640, 666)], color=COLORS["fbwa"])
    elbow_arrow(ax, [(1060, 546), (990, 546), (990, 690), (960, 690)], color=COLORS["fbwa"])

    rounded_box(ax, 1110, 650, 170, 70, "$\\mathcal{L}_{fbwa}$", ec=COLORS["fbwa"], fc="white", fontsize=14, color=COLORS["fbwa"], weight="bold")
    straight_arrow(ax, (960, 678), (1110, 685), color=COLORS["fbwa"])
    elbow_arrow(ax, [(1280, 685), (1510, 685), (1510, 180), (1485, 180)], color=COLORS["fbwa"])
    label(ax, 1500, 285, "structural\nfeedback", fontsize=8, color=COLORS["fbwa"], ha="right")

    label(ax, 320, 86, "labeled feature-boundary pair", fontsize=14, weight="bold")
    label(ax, 1280, 86, "unlabeled pseudo-boundary pair", fontsize=14, weight="bold")

    save_all(fig, "user_fbwa_module_notitle", "fbwa_module.png")


def draw_sorp_rct_module():
    fig, ax = new_canvas()

    rounded_box(ax, 35, 45, 795, 790, ec=COLORS["sorp"], fc=COLORS["sorp_fill"], lw=1.5, radius=10)
    rounded_box(ax, 865, 45, 700, 790, ec=COLORS["rct"], fc=COLORS["rct_fill"], lw=1.5, radius=10)

    image_card(ax, PATHS["busi_image"], 80, 115, 135, 135, "$x_u$", ec=COLORS["muted"], label_size=9)
    rounded_box(ax, 270, 145, 110, 70, "student\nencoder", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=9)
    feature_strip(ax, 430, 134, 92, 86, ec=COLORS["sorp"], fc="#EEE2F6", label_text="$F$")
    rounded_box(ax, 570, 145, 110, 70, "main\ndecoder", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=9)
    image_card(ax, PATHS["busi_pred"], 700, 105, 116, 116, "$P_u$", ec=COLORS["sorp"], label_size=9, image_pad=7, label_h=20)
    straight_arrow(ax, (215, 182), (270, 182), color=COLORS["line"])
    straight_arrow(ax, (380, 182), (430, 182), color=COLORS["line"])
    straight_arrow(ax, (522, 182), (570, 182), color=COLORS["line"])
    straight_arrow(ax, (680, 182), (700, 182), color=COLORS["line"])

    image_card(ax, PATHS["busi_pred"], 95, 430, 116, 116, "$M_{sorp}$", ec=COLORS["sorp"], label_size=9, image_pad=7, label_h=20)
    heatmap_card(ax, PATHS["busi_pred"], 270, 430, 116, 116, "residual", ec=COLORS["sorp"], label_size=8)
    feature_strip(ax, 450, 435, 92, 86, ec=COLORS["sorp"], fc="#E8D6F2", label_text=r"$\tilde{F}$")
    rounded_box(ax, 588, 454, 110, 70, "auxiliary\ndecoder", ec=COLORS["muted"], fc=COLORS["muted_fill"], fontsize=9)
    image_card(ax, PATHS["busi_pred"], 735, 430, 116, 116, "$P_u^a$", ec=COLORS["sorp"], label_size=9, image_pad=7, label_h=20)
    elbow_arrow(ax, [(758, 221), (758, 310), (153, 310), (153, 430)], color=COLORS["sorp"])
    straight_arrow(ax, (211, 488), (270, 488), color=COLORS["sorp"])
    straight_arrow(ax, (386, 488), (450, 488), color=COLORS["sorp"])
    straight_arrow(ax, (542, 488), (588, 488), color=COLORS["sorp"])
    straight_arrow(ax, (698, 488), (735, 488), color=COLORS["sorp"])
    rounded_box(ax, 355, 610, 250, 46, r"$\tilde{F} = \rho F + (1-\rho)(F \odot M_{sorp})$", ec=COLORS["sorp"], fc="white", fontsize=10, color=COLORS["sorp"])
    straight_arrow(ax, (496, 521), (496, 610), color=COLORS["sorp"])

    # RCT side
    rounded_box(ax, 945, 122, 150, 78, "EMA\nteacher", ec=COLORS["teacher"], fc=COLORS["teacher_fill"], fontsize=10, color=COLORS["teacher"], weight="bold")
    image_card(ax, PATHS["tn3k_overlay"], 1185, 103, 116, 116, "$T_u$", ec=COLORS["teacher"], fc=COLORS["teacher_fill"], label_size=9, image_pad=7, label_h=20)
    straight_arrow(ax, (1095, 161), (1185, 161), color=COLORS["teacher"])
    elbow_arrow(ax, [(758, 162), (865, 162), (945, 162)], color=COLORS["teacher"], dashed=True)
    label(ax, 895, 137, "EMA", fontsize=8, color=COLORS["teacher"], weight="bold")

    rounded_box(ax, 905, 300, 610, 265, ec=COLORS["rct"], fc="white", lw=1.25, radius=8)
    heatmap_card(ax, PATHS["busi_pred"], 940, 340, 128, 128, "entropy", ec=COLORS["rct"], label_size=8)
    image_card(ax, PATHS["busi_overlay"], 1138, 340, 128, 128, "boundary", ec=COLORS["rct"], label_size=8, image_pad=7, label_h=20)
    image_card(ax, PATHS["busi_overlay"], 1336, 340, 128, 128, "agreement", ec=COLORS["rct"], label_size=8, image_pad=7, label_h=20)
    heatmap_card(ax, PATHS["busi_pred"], 1145, 575, 148, 120, "$R_u$", ec=COLORS["rct"], label_size=9)
    straight_arrow(ax, (1004, 468), (1190, 575), color=COLORS["rct"])
    straight_arrow(ax, (1202, 468), (1219, 575), color=COLORS["rct"])
    straight_arrow(ax, (1400, 468), (1260, 575), color=COLORS["rct"])
    elbow_arrow(ax, [(1301, 161), (1328, 161), (1328, 340), (1202, 340)], color=COLORS["teacher"])
    elbow_arrow(ax, [(758, 162), (875, 162), (875, 395), (940, 395)], color=COLORS["rct"])
    elbow_arrow(ax, [(851, 488), (875, 488), (875, 425), (1336, 425)], color=COLORS["sorp"])

    rounded_box(ax, 1010, 732, 360, 58, "$R_u \\odot \\mathcal{L}(T_u, P_u)\\; +\\; R_u \\odot \\mathcal{L}(T_u, P_u^a)$", ec=COLORS["loss"], fc=COLORS["loss_fill"], fontsize=12, color=COLORS["loss"], weight="bold")
    rounded_box(ax, 1415, 720, 118, 72, "$\\mathcal{L}_s$\n(labeled)", ec=COLORS["loss"], fc="white", fontsize=9, color=COLORS["loss"], weight="bold")
    rounded_box(ax, 1125, 817, 200, 40, "$\\mathcal{L}=\\mathcal{L}_s+\\mathcal{L}_u$", ec=COLORS["loss"], fc="white", fontsize=11, color=COLORS["loss"], weight="bold")
    straight_arrow(ax, (1219, 695), (1219, 732), color=COLORS["rct"])
    straight_arrow(ax, (1474, 720), (1370, 760), color=COLORS["loss"], dashed=True)
    straight_arrow(ax, (1219, 790), (1219, 817), color=COLORS["loss"])
    elbow_arrow(ax, [(1219, 857), (1219, 875), (175, 875), (175, 546)], color=COLORS["loss"], dashed=True)
    label(ax, 520, 850, "loss feedback", fontsize=8, color=COLORS["loss"])

    save_all(fig, "sorp_rct_module", "sorp_rct_module.png")


def main():
    draw_overall_framework()
    draw_fbwa_module()
    draw_sorp_rct_module()


if __name__ == "__main__":
    main()
