import cv2
import numpy as np
from PIL import Image


def load_rgb_image(path):
    with Image.open(path) as image:
        return image.convert('RGB').copy()


def load_grayscale_image(path):
    with Image.open(path) as image:
        return image.convert('L').copy()


def _fill_contour_region(boundary_array):
    _, thresholded = cv2.threshold(boundary_array, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(boundary_array)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _fill_ellipse_region(boundary_array):
    binary = (boundary_array > 127).astype(np.uint8)
    points_yx = np.column_stack(np.where(binary > 0))
    if points_yx.shape[0] < 5:
        return _fill_contour_region(boundary_array)

    points_xy = np.stack([points_yx[:, 1], points_yx[:, 0]], axis=1).astype(np.float32)
    try:
        ellipse = cv2.fitEllipse(points_xy)
        (cx, cy), (width, height), angle = ellipse
    except cv2.error:
        return _fill_contour_region(boundary_array)

    if not all(np.isfinite(v) for v in (cx, cy, width, height, angle)) or width <= 0 or height <= 0:
        return _fill_contour_region(boundary_array)

    filled = np.zeros_like(boundary_array)
    try:
        cv2.ellipse(filled, ellipse, 255, thickness=-1)
    except cv2.error:
        return _fill_contour_region(boundary_array)
    if filled.max() == 0:
        return _fill_contour_region(boundary_array)
    return filled


def build_region_mask(boundary_mask, mode='contour'):
    boundary_array = np.asarray(boundary_mask, dtype=np.uint8)
    if str(mode).lower() == 'ellipse':
        filled = _fill_ellipse_region(boundary_array)
    else:
        filled = _fill_contour_region(boundary_array)
    return Image.fromarray(filled, mode='L')
