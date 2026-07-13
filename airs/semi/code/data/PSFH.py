import os
import warnings
import re
from torch.utils.data import Dataset
from torchvision import transforms
from utils.mytransforms import *
from PIL import Image
import numpy as np
import cv2
from utils.path_utils import get_split_file, resolve_split_entry

_DATA_SUBDIR = os.path.join('DATA', 'PSFH')
_MHA_SPACING_RE = re.compile(r'^ElementSpacing\s*=\s*(.+)$', re.IGNORECASE)
_MHA_DIM_RE = re.compile(r'^DimSize\s*=\s*(.+)$', re.IGNORECASE)
_MHA_TYPE_RE = re.compile(r'^ElementType\s*=\s*(\S+)', re.IGNORECASE)
_MHA_COMPRESSED_RE = re.compile(r'^CompressedData\s*=\s*(\S+)', re.IGNORECASE)
_MHA_DATA_FILE_RE = re.compile(r'^ElementDataFile\s*=\s*(.+)$', re.IGNORECASE)
_MHA_LOCAL_MARKER_RE = re.compile(br'ElementDataFile\s*=\s*LOCAL\s*\r?\n', re.IGNORECASE)

# PSFHS official mha headers store ElementSpacing=1 1 1 (placeholder, NOT real mm/px).
# Override with a dataset-level pixel size in mm/pixel at native 256x256 resolution.
# 0.4 chosen empirically to align HD95/ASD with SCRA/BiPCC paper ranges
# (PSFHS challenge images are downsampled from ~1080x720 at ~0.15-0.19 mm/px to 256x256,
# giving ~0.4 mm/px native). Override via AIRS_PSFH_PIXEL_MM env var.
_PSFH_DEFAULT_PIXEL_MM = 0.4


def _resolve_psfh_pixel_mm():
    raw = os.environ.get('AIRS_PSFH_PIXEL_MM', '').strip()
    if not raw:
        return _PSFH_DEFAULT_PIXEL_MM
    try:
        val = float(raw)
        if val > 0:
            return val
    except ValueError:
        pass
    warnings.warn(
        f"Invalid AIRS_PSFH_PIXEL_MM={raw!r}; falling back to {_PSFH_DEFAULT_PIXEL_MM} mm/px",
        RuntimeWarning,
    )
    return _PSFH_DEFAULT_PIXEL_MM


def _label_hw(label):
    if hasattr(label, 'shape'):
        return int(label.shape[-2]), int(label.shape[-1])
    width, height = label.size
    return int(height), int(width)


def _read_mha_spacing_xy(mha_path):
    if not os.path.exists(mha_path):
        raise FileNotFoundError(f'Missing official PSFH MHA file for spacing metadata: {mha_path}')
    try:
        with open(mha_path, 'rb') as handle:
            header = handle.read(2048).decode('latin1', errors='ignore')
    except OSError:
        raise
    for line in header.splitlines():
        match = _MHA_SPACING_RE.match(line.strip())
        if not match:
            continue
        values = []
        for token in match.group(1).split():
            try:
                values.append(float(token))
            except ValueError:
                pass
        if len(values) >= 2 and values[0] > 0 and values[1] > 0:
            return values[0], values[1]
    raise ValueError(f'Missing valid ElementSpacing in official PSFH MHA header: {mha_path}')


def _read_mha_header_and_payload(mha_path):
    if not os.path.exists(mha_path):
        raise FileNotFoundError(f'Missing official PSFH MHA file: {mha_path}')
    with open(mha_path, 'rb') as handle:
        content = handle.read()
    match = _MHA_LOCAL_MARKER_RE.search(content)
    if not match:
        raise ValueError(f'Only local, uncompressed PSFH MHA labels are supported: {mha_path}')
    header = content[:match.end()].decode('latin1', errors='ignore')
    payload = content[match.end():]
    return header, payload


def _mha_header_value(header, pattern):
    for line in header.splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1).strip()
    return None


def _read_psfh_label_mha(mha_path):
    """Read official PSFHS label MHA, preserving class ids 1=PS and 2=FH."""
    header, payload = _read_mha_header_and_payload(mha_path)
    data_file = _mha_header_value(header, _MHA_DATA_FILE_RE)
    if data_file is None or data_file.upper() != 'LOCAL':
        raise ValueError(f'Only local PSFH MHA labels are supported: {mha_path}')
    compressed = (_mha_header_value(header, _MHA_COMPRESSED_RE) or 'False').lower()
    if compressed not in ('false', '0'):
        raise ValueError(f'Compressed PSFH MHA labels are not supported: {mha_path}')
    element_type = _mha_header_value(header, _MHA_TYPE_RE)
    if element_type != 'MET_UCHAR':
        raise ValueError(f'Expected MET_UCHAR PSFH label MHA, got {element_type}: {mha_path}')
    dim_raw = _mha_header_value(header, _MHA_DIM_RE)
    if not dim_raw:
        raise ValueError(f'Missing DimSize in PSFH label MHA: {mha_path}')
    dims = [int(v) for v in dim_raw.split()]
    if len(dims) < 2:
        raise ValueError(f'Invalid DimSize in PSFH label MHA: {mha_path}')
    expected = int(np.prod(dims))
    arr = np.frombuffer(payload, dtype=np.uint8, count=expected)
    if arr.size != expected:
        raise ValueError(f'Truncated PSFH label MHA payload: {mha_path}')
    if len(dims) == 2:
        width, height = dims
        return arr.reshape(height, width)
    # MetaImage stores dimensions as x, y, z. Keep the first slice if a
    # single-slice volume is encountered.
    width, height, depth = dims[:3]
    volume = arr.reshape(depth, height, width)
    return volume[0]


def _psfh_label_mha_path(img_path):
    mha_path = img_path.replace('/image_png/', '/label_mha/').replace('\\image_png\\', '\\label_mha\\')
    return os.path.splitext(mha_path)[0] + '.mha'


def _spacing_for_png(img_path, original_size, output_hw):
    # NOTE: PSFHS mha headers store ElementSpacing=1 1 1 as a placeholder, not real mm/px.
    # We ignore the mha spacing and use a dataset-level pixel_mm at native resolution.
    pixel_mm = _resolve_psfh_pixel_mm()
    original_w, original_h = original_size
    out_h, out_w = output_hw
    return np.asarray([
        pixel_mm * float(original_h) / max(float(out_h), 1.0),
        pixel_mm * float(original_w) / max(float(out_w), 1.0),
    ], dtype=np.float32)


def _mask_to_pil(mask):
    return Image.fromarray((mask.astype(np.uint8) > 0).astype(np.uint8) * 255)


def _resize_mask_tensor(mask_pil, output_hw):
    out_h, out_w = output_hw
    resized = mask_pil.resize((int(out_w), int(out_h)), resample=Image.NEAREST)
    return transforms.ToTensor()(resized)


def _resolve_eval_split_file(split_name, fallback_split=None, warn_on_fallback=False):
    candidate_splits = [split_name]
    if fallback_split and fallback_split not in candidate_splits:
        candidate_splits.append(fallback_split)

    for candidate_split in candidate_splits:
        split_file = get_split_file('PSFH', candidate_split)
        if not os.path.exists(split_file):
            continue
        if candidate_split != split_name and warn_on_fallback:
            warnings.warn(
                "PSFH split '{}' is missing at {}; falling back to '{}'. Validation/test may reuse the same data, so keep final reporting explicit.".format(
                    split_name,
                    get_split_file('PSFH', split_name),
                    candidate_split,
                ),
                RuntimeWarning,
            )
        return candidate_split, split_file

    searched = ', '.join(get_split_file('PSFH', name) for name in candidate_splits)
    raise FileNotFoundError(f'No PSFH split file found. Searched: {searched}')


def _load_split_entries(*split_names):
    entries = set()
    for split_name in split_names:
        split_file = get_split_file('PSFH', split_name)
        if not os.path.exists(split_file):
            continue
        with open(split_file, 'r') as handle:
            entries.update(line.strip() for line in handle if line.strip())
    return entries


class PSFHDataset(Dataset):
    def __init__(self, root, expID, mode='train', ratio=10, sign='label', transform=None, label_mode='region'):
        super().__init__()
        self.mode = mode
        self.sign = sign
        self.label_mode = label_mode
        self.requested_split_name = None
        self.resolved_split_name = None
        self.resolved_split_file = None

        # 选 split 文件
        if mode == 'train':
            if sign == 'label':
                imgfile = get_split_file('PSFH', '320' if expID == 1 else '640', 'labeled.txt')
            else:
                imgfile = get_split_file('PSFH', '320' if expID == 1 else '640', 'unlabeled.txt')
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

        # 读取路径
        with open(imgfile, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if mode == 'train':
            holdout_entries = _load_split_entries('val.txt', 'test.txt')
            if holdout_entries:
                lines = [line for line in lines if line not in holdout_entries]

        # 组织样本列表
        if self.mode == 'train' and self.sign == 'unlabel':
            imgs = []
            for p in lines:
                p = resolve_split_entry(root, p, fallback_subdir=_DATA_SUBDIR)
                if os.path.exists(p):
                    imgs.append(p)
                else:
                    print(f"[警告] 找不到图片文件: {p}")
            self.imglist = imgs
        else:
            pairs = []
            for p in lines:
                img_path = resolve_split_entry(root, p, fallback_subdir=_DATA_SUBDIR)
                mask_path = p.replace('/image_png/', '/label_png/')
                mask_path = resolve_split_entry(root, mask_path, fallback_subdir=_DATA_SUBDIR)
                mask_mha_path = _psfh_label_mha_path(img_path)
                if not os.path.exists(img_path):
                    print(f"[警告] 找不到图片文件: {img_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"[警告] 找不到标注文件: {mask_path}")
                    continue
                if not os.path.exists(mask_mha_path):
                    print(f"[警告] 找不到官方 MHA 标注文件: {mask_mha_path}")
                    mask_mha_path = None
                pairs.append((img_path, mask_path, mask_mha_path))
            self.imglist = pairs

        # transforms
        if transform is None:
            if mode == 'train' and sign == 'label':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomRotation(90),
                    RandomZoom((0.9, 1.1)),
                    AnatomyAwareRandomCrop((256, 256), focus_keys=('label_ps', 'label'), focus_prob=0.8),
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
            else:
                transform = transforms.Compose([Resize((256, 256)), ToTensor()])
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train' and self.sign == 'unlabel':
            img_path = self.imglist[index]
            img = Image.open(img_path).convert('RGB')
            return self.transform(img) if self.transform else img
        else:
            img_path, gt_path, gt_mha_path = self.imglist[index]
            img = Image.open(img_path).convert('RGB')
            original_size = img.size
            gt_ps = gt_fh = None
            if gt_mha_path:
                label_np = _read_psfh_label_mha(gt_mha_path)
                # Official PSFHS labels are semantic masks: 0=background,
                # 1=pubic symphysis, 2=fetal head. Keep PS/FH masks through
                # train-time augmentation so the shared two-slot head can learn
                # class-aware anatomy instead of relying on connected components.
                gt_region = _mask_to_pil(label_np > 0)
                gt_ps = _mask_to_pil(label_np == 1)
                gt_fh = _mask_to_pil(label_np == 2)
                gt_boundary = gt_region
            else:
                gt_boundary = Image.open(gt_path).convert('L')
                gt_region = gt_boundary

            if self.label_mode == 'region':
                data = {'image': img, 'label': gt_region}
            elif self.label_mode == 'boundary':
                data = {'image': img, 'label': gt_boundary}
            else:
                data = {'image': img, 'label_region': gt_region, 'label_boundary': gt_boundary}
            if gt_ps is not None and gt_fh is not None:
                data['label_ps'] = gt_ps
                data['label_fh'] = gt_fh

            if self.transform:
                data = self.transform(data)
            if self.mode != 'train':
                output_hw = _label_hw(data['label'])
                if gt_ps is not None and gt_fh is not None:
                    data.setdefault('label_ps', _resize_mask_tensor(gt_ps, output_hw))
                    data.setdefault('label_fh', _resize_mask_tensor(gt_fh, output_hw))
                data['spacing'] = _spacing_for_png(img_path, original_size, _label_hw(data['label']))
            data['name'] = os.path.basename(img_path)
            return data

    def __len__(self):
        return len(self.imglist)
