import os
import warnings
import csv
import re
from torch.utils.data import Dataset
from torchvision import transforms
from utils.mytransforms import *
from PIL import Image
import numpy as np
import cv2
from utils.path_utils import AIRS_ROOT, get_split_file, resolve_split_entry

_DATA_SUBDIR = os.path.join('DATA', 'HC18')
_GT_MODE_ENV = 'AIRS_HC18_GT_MODE'  # 'fill' (default, paper baseline) | 'ellipse' (ablation)
_VALID_GT_MODES = ('fill', 'ellipse')
_OFFICIAL_PIXEL_SIZE_FILES = (
    ('training_set', 'training_set_pixel_size_and_HC.csv'),
    ('training_set', 'training_set_pixel_size.csv'),
    ('test_set', 'test_set_pixel_size.csv'),
)
def _entry_subset(raw_entry):
    parts = str(raw_entry).replace('\\', '/').split('/')
    for idx, part in enumerate(parts):
        if part == 'HC18' and idx + 1 < len(parts):
            return parts[idx + 1]
    if parts and parts[0] in ('training_set', 'test_set'):
        return parts[0]
    return 'unknown'


def _source_subset_counts(lines):
    counts = {}
    for line in lines:
        subset = _entry_subset(line)
        counts[subset] = counts.get(subset, 0) + 1
    return dict(sorted(counts.items()))


def _parse_pixel_size_rows(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
            file_name = (normalized.get('filename') or '').strip()
            pixel_size_raw = (
                normalized.get('pixel size(mm)')
                or normalized.get('pixel size (mm)')
                or normalized.get('pixel_size')
            )
            if not file_name or pixel_size_raw is None:
                continue
            try:
                pixel_size = float(pixel_size_raw)
            except ValueError:
                continue
            if pixel_size <= 0:
                continue
            rows.append((os.path.basename(file_name), pixel_size))
    return rows


def _filename_aliases(file_name):
    aliases = {os.path.basename(file_name)}
    match = re.match(r'^0*(\d+)(.*HC\.png)$', os.path.basename(file_name))
    if match:
        aliases.add(f'{int(match.group(1))}{match.group(2)}')
    return aliases


def _hc18_sort_key(file_name):
    match = re.match(r'^(\d+)', os.path.basename(file_name))
    number = int(match.group(1)) if match else 10**9
    return number, os.path.basename(file_name)


def _local_subset_filenames(data_root, subset):
    subset_dir = os.path.join(data_root, subset)
    if not os.path.isdir(subset_dir):
        return []
    names = []
    for name in os.listdir(subset_dir):
        if not name.lower().endswith('.png') or 'annotation' in name.lower():
            continue
        names.append(name)
    return sorted(names, key=_hc18_sort_key)


def _load_pixel_size_map(root):
    hc18_data_root = os.path.join(os.path.abspath(os.path.expanduser(str(root))), _DATA_SUBDIR)
    metadata_dirs = [
        hc18_data_root,
        str(AIRS_ROOT / 'data' / 'metadata' / 'HC18'),
    ]
    pixel_sizes = {}
    for subset, name in _OFFICIAL_PIXEL_SIZE_FILES:
        rows = []
        for metadata_dir in metadata_dirs:
            path = os.path.join(metadata_dir, name)
            if not os.path.exists(path):
                continue
            rows = _parse_pixel_size_rows(path)
            break
        if not rows:
            continue

        for file_name, pixel_size in rows:
            for alias in _filename_aliases(file_name):
                pixel_sizes[f'{subset}/{alias}'] = pixel_size

        # Some local HC18 mirrors renumber official images to 1..N while keeping
        # the official row order. Use the on-disk order when available.
        local_names = _local_subset_filenames(hc18_data_root, subset)
        if len(local_names) == len(rows):
            for local_name, (_, pixel_size) in zip(local_names, rows):
                pixel_sizes[f'{subset}/{local_name}'] = pixel_size
    return pixel_sizes


def _official_pixel_size(pixel_size_map, img_path):
    subset = os.path.basename(os.path.dirname(img_path))
    key = f'{subset}/{os.path.basename(img_path)}'
    if key in pixel_size_map:
        return pixel_size_map[key]
    raise KeyError(
        "Missing official HC18 pixel size for '{}'. Expected one of the official Zenodo CSV files "
        "under DATA/HC18 or airs/data/metadata/HC18.".format(key)
    )


def _label_hw(label):
    if hasattr(label, 'shape'):
        return int(label.shape[-2]), int(label.shape[-1])
    width, height = label.size
    return int(height), int(width)


def _resized_spacing(original_size, output_hw, pixel_size_mm):
    original_w, original_h = original_size
    out_h, out_w = output_hw
    return np.asarray([
        pixel_size_mm * float(original_h) / max(float(out_h), 1.0),
        pixel_size_mm * float(original_w) / max(float(out_w), 1.0),
    ], dtype=np.float32)


def _fill_contour_region(boundary_array):
    _, thresh = cv2.threshold(boundary_array, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(boundary_array)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _resolve_gt_mode():
    raw = os.environ.get(_GT_MODE_ENV)
    if raw is None or not str(raw).strip():
        return 'fill'
    mode = str(raw).strip().lower()
    if mode not in _VALID_GT_MODES:
        warnings.warn(
            "Unknown {}='{}'; valid: {}. Falling back to 'fill'.".format(
                _GT_MODE_ENV, raw, _VALID_GT_MODES,
            ),
            RuntimeWarning,
        )
        return 'fill'
    return mode


def _build_hc18_region_mask(boundary_mask):
    """Convert the official HC18 ellipse annotation into a filled region mask.

    Some HC18 annotations are visually elliptical but not topologically closed
    after rasterization. Filling contours then keeps only the thin boundary and
    makes Dice/HD95 meaningless for those cases. Fitting the official ellipse
    contour is robust to tiny gaps and matches the closed-contour cases.
    """
    boundary_array = np.asarray(boundary_mask, dtype=np.uint8)
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


def _resolve_eval_split_file(split_name, fallback_split=None, warn_on_fallback=False):
    candidate_splits = [split_name]
    if fallback_split and fallback_split not in candidate_splits:
        candidate_splits.append(fallback_split)

    for candidate_split in candidate_splits:
        split_file = get_split_file('HC18', candidate_split)
        if not os.path.exists(split_file):
            continue
        if candidate_split != split_name and warn_on_fallback:
            warnings.warn(
                "HC18 split '{}' is missing at {}; falling back to '{}'. Validation/test may reuse the same data, so keep final reporting explicit.".format(
                    split_name,
                    get_split_file('HC18', split_name),
                    candidate_split,
                ),
                RuntimeWarning,
            )
        return candidate_split, split_file

    searched = ', '.join(get_split_file('HC18', name) for name in candidate_splits)
    raise FileNotFoundError(f'No HC18 split file found. Searched: {searched}')


def _load_split_entries(*split_names):
    entries = set()
    for split_name in split_names:
        split_file = get_split_file('HC18', split_name)
        if not os.path.exists(split_file):
            continue
        with open(split_file, 'r') as handle:
            entries.update(line.strip() for line in handle if line.strip())
    return entries


class HC18Dataset(Dataset):
    def __init__(self, root, expID, mode='train', ratio=10, sign='label', transform=None, label_mode='region'):
        super().__init__()
        self.mode = mode
        self.sign = sign
        self.label_mode = label_mode
        self.requested_split_name = None
        self.resolved_split_name = None
        self.resolved_split_file = None
        self.raw_line_count = 0
        self.effective_line_count = 0
        self.filtered_holdout_count = 0
        self.filtered_official_test_set_count = 0
        self.source_subset_counts = {}
        self.gt_mode = _resolve_gt_mode()
        self.pixel_size_map = _load_pixel_size_map(root)
        # 选 split 文件
        if mode == 'train':
            if sign == 'label':
                self.requested_split_name = '107/labeled.txt' if expID == 1 else '214/labeled.txt'
                imgfile = get_split_file('HC18', '107' if expID == 1 else '214', 'labeled.txt')
            else:
                self.requested_split_name = '107/unlabeled.txt' if expID == 1 else '214/unlabeled.txt'
                imgfile = get_split_file('HC18', '107' if expID == 1 else '214', 'unlabeled.txt')
            self.resolved_split_name = self.requested_split_name
            self.resolved_split_file = imgfile
        elif mode == 'valid':
            self.requested_split_name = 'val.txt'
            self.resolved_split_name, imgfile = _resolve_eval_split_file(
                'val.txt',
            )
            self.resolved_split_file = imgfile
        else:  # test
            self.requested_split_name = 'test.txt'
            self.resolved_split_name, imgfile = _resolve_eval_split_file(
                'test.txt',
            )
            self.resolved_split_file = imgfile

        # 读取行（统一用 lines，不再对同一个 f 二次遍历）
        with open(imgfile, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        self.raw_line_count = len(lines)
        if mode == 'train':
            holdout_entries = _load_split_entries('val.txt', 'test.txt')
            if holdout_entries:
                before_holdout = len(lines)
                lines = [line for line in lines if line not in holdout_entries]
                self.filtered_holdout_count = before_holdout - len(lines)
            # BiPCC §IV-A transductive protocol: official test_set entries are
            # allowed as unlabeled training samples (images only, labels never
            # leaked via holdout filtering above).
            self.filtered_official_test_set_count = 0
        self.effective_line_count = len(lines)
        self.source_subset_counts = _source_subset_counts(lines)

        # 组织样本列表并做存在性校验
        if self.mode == 'train' and self.sign == 'unlabel':
            # 仅图片路径
            imgs = []
            for p in lines:
                p = resolve_split_entry(root, p, fallback_subdir=_DATA_SUBDIR)
                if os.path.exists(p):
                    imgs.append(p)
                else:
                    print(f"[警告] 找不到图片文件: {p}")
            self.imglist = imgs
        else:
            # 成对的 (img, mask)
            pairs = []
            for p in lines:
                img_path = resolve_split_entry(root, p, fallback_subdir=_DATA_SUBDIR)
                mask_path = p.replace('.png', '_Annotation.png')
                mask_path = resolve_split_entry(root, mask_path, fallback_subdir=_DATA_SUBDIR)
                if not os.path.exists(img_path):
                    print(f"[警告] 找不到图片文件: {img_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"[警告] 找不到标注文件: {mask_path}")
                    continue
                pairs.append((img_path, mask_path))
            self.imglist = pairs

        # transforms（保留你原来的）
        if transform is None:
            if mode == 'train' and sign == 'label':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomRotation(90),
                    RandomZoom((0.9, 1.1)),
                    AnatomyAwareRandomCrop((256, 256), focus_keys=('label',), focus_prob=0.6),
                    ToTensor()
                ])
            elif mode == 'train' and sign == 'unlabel':
                transform = transforms.Compose([
                    transforms.Resize((320, 320)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.RandomCrop((256, 256)),
                    transforms.ToTensor()
                ])
            else:  # valid/test
                transform = transforms.Compose([Resize((320, 320)), ToTensor()])
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train' and self.sign == 'unlabel':
            img_path = self.imglist[index]
            img = Image.open(img_path).convert('RGB')
            return self.transform(img) if self.transform else img
        else:
            img_path, gt_path = self.imglist[index]
            img = Image.open(img_path).convert('RGB')
            original_size = img.size
            pixel_size_mm = _official_pixel_size(self.pixel_size_map, img_path)

            # 原始边界 GT（灰度：细白边，黑背景）
            gt_boundary = Image.open(gt_path).convert('L')

            # 如果需要区域 GT，就把边界填充
            if self.label_mode in ('region', 'both'):
                boundary_array = np.asarray(gt_boundary, dtype=np.uint8)
                if self.gt_mode == 'ellipse':
                    region_array = _build_hc18_region_mask(gt_boundary)
                else:
                    region_array = _fill_contour_region(boundary_array)
                gt_region = Image.fromarray(region_array, mode='L')

            # 组装返回
            if self.label_mode == 'region':
                data = {'image': img, 'label': gt_region}
            elif self.label_mode == 'boundary':
                data = {'image': img, 'label': gt_boundary}
            else:  # 'both'
                data = {'image': img, 'label_region': gt_region, 'label_boundary': gt_boundary}

            if self.transform:
                data = self.transform(data)
            if self.mode != 'train':
                data['spacing'] = _resized_spacing(original_size, _label_hw(data['label']), pixel_size_mm)
            data['name'] = os.path.basename(img_path)
            return data

    def __len__(self):
        return len(self.imglist)
