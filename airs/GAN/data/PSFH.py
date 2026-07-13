import os
import re
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from .utils.mytransforms import *
from .image_utils import build_region_mask, load_grayscale_image, load_rgb_image
from .path_utils import get_split_file, resolve_split_entry

_DATA_SUBDIR = os.path.join('DATA', 'PSFH')

_MHA_LOCAL_MARKER_RE = re.compile(br'ElementDataFile\s*=\s*LOCAL\s*\r?\n', re.IGNORECASE)
_MHA_DIM_RE = re.compile(r'^DimSize\s*=\s*(.+)$', re.IGNORECASE)
_MHA_TYPE_RE = re.compile(r'^ElementType\s*=\s*(\S+)', re.IGNORECASE)
_MHA_COMPRESSED_RE = re.compile(r'^CompressedData\s*=\s*(\S+)', re.IGNORECASE)
_MHA_DATA_FILE_RE = re.compile(r'^ElementDataFile\s*=\s*(.+)$', re.IGNORECASE)


def _psfh_label_mha_path(img_path):
    mha_path = img_path.replace('/image_png/', '/label_mha/').replace('\\image_png\\', '\\label_mha\\')
    return os.path.splitext(mha_path)[0] + '.mha'


def _mha_header_value(header, pattern):
    for line in header.splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1).strip()
    return None


def _read_psfh_label_mha(mha_path):
    """Read official PSFHS label MHA, preserving class ids 1=PS and 2=FH."""
    if not os.path.exists(mha_path):
        return None
    with open(mha_path, 'rb') as handle:
        content = handle.read()
    match = _MHA_LOCAL_MARKER_RE.search(content)
    if not match:
        return None
    header = content[:match.end()].decode('latin1', errors='ignore')
    payload = content[match.end():]
    data_file = _mha_header_value(header, _MHA_DATA_FILE_RE)
    if data_file is None or data_file.upper() != 'LOCAL':
        return None
    compressed = (_mha_header_value(header, _MHA_COMPRESSED_RE) or 'False').lower()
    if compressed not in ('false', '0'):
        return None
    element_type = _mha_header_value(header, _MHA_TYPE_RE)
    if element_type != 'MET_UCHAR':
        return None
    dim_raw = _mha_header_value(header, _MHA_DIM_RE)
    if not dim_raw:
        return None
    dims = [int(v) for v in dim_raw.split()]
    if len(dims) < 2:
        return None
    expected = int(np.prod(dims))
    arr = np.frombuffer(payload, dtype=np.uint8, count=expected)
    if arr.size != expected:
        return None
    if len(dims) == 2:
        width, height = dims
        return arr.reshape(height, width)
    width, height, depth = dims[:3]
    return arr.reshape(depth, height, width)[0]


def _mha_to_class_masks(mha_path):
    """Returns (ps_mask_PIL, fh_mask_PIL) or (None, None) when not available."""
    label = _read_psfh_label_mha(mha_path)
    if label is None:
        return None, None
    ps = Image.fromarray((label == 1).astype(np.uint8) * 255, mode='L')
    fh = Image.fromarray((label == 2).astype(np.uint8) * 255, mode='L')
    return ps, fh


def _load_holdout_entries():
    entries = set()
    for split_name in ('val.txt', 'test.txt'):
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

        # 选 split 文件
        if mode == 'train':
            if sign == 'label':
                imgfile = get_split_file('PSFH', '320' if expID == 1 else '640', 'labeled.txt')
            else:
                imgfile = get_split_file('PSFH', '320' if expID == 1 else '640', 'unlabeled.txt')
        else:  # valid/test
            imgfile = get_split_file('PSFH', 'test.txt')

        # 读取路径
        with open(imgfile, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if mode == 'train':
            holdout_entries = _load_holdout_entries()
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
                if not os.path.exists(img_path):
                    print(f"[警告] 找不到图片文件: {img_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"[警告] 找不到标注文件: {mask_path}")
                    continue
                pairs.append((img_path, mask_path))
            self.imglist = pairs

        self.region_mask_cache = {}
        if self.mode != 'train' or self.sign != 'unlabel':
            if self.label_mode in ('region', 'both'):
                unique_mask_paths = {mask_path for _, mask_path in self.imglist}
                self.region_mask_cache = {
                    mask_path: build_region_mask(load_grayscale_image(mask_path))
                    for mask_path in unique_mask_paths
                }

        # transforms
        if transform is None:
            if mode == 'train' and sign == 'label':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomRotation(90),
                    RandomZoom((0.9, 1.1)),
                    RandomCrop((256, 256)),
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
                transform = transforms.Compose([Resize((320, 320)), ToTensor()])
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train' and self.sign == 'unlabel':
            img_path = self.imglist[index]
            img = load_rgb_image(img_path)
            return self.transform(img) if self.transform else img
        else:
            img_path, gt_path = self.imglist[index]
            img = load_rgb_image(img_path)
            gt_boundary = load_grayscale_image(gt_path)

            if self.label_mode in ('region', 'both'):
                gt_region = self.region_mask_cache[gt_path].copy()

            # Read per-class PSFHS masks (1=PS, 2=FH) from the official MHA so
            # GAN pretrain sees per-class shape rather than a merged blob.
            # Encode region/ps/fh into a single 3-channel PIL Image so the
            # existing single-label transform pipeline carries all three through
            # resize/flip/rotate/crop atomically (avoids desync between channels).
            ps_pil, fh_pil = _mha_to_class_masks(_psfh_label_mha_path(img_path))
            if ps_pil is not None and fh_pil is not None and self.label_mode == 'region':
                region_arr = np.asarray(gt_region, dtype=np.uint8)
                ps_arr = np.asarray(ps_pil, dtype=np.uint8)
                fh_arr = np.asarray(fh_pil, dtype=np.uint8)
                stacked = np.stack([region_arr, ps_arr, fh_arr], axis=-1)
                packed_label = Image.fromarray(stacked, mode='RGB')
                data = {'image': img, 'label': packed_label}
            elif self.label_mode == 'region':
                data = {'image': img, 'label': gt_region}
            elif self.label_mode == 'boundary':
                data = {'image': img, 'label': gt_boundary}
            else:
                data = {'image': img, 'label_region': gt_region, 'label_boundary': gt_boundary}

            if self.transform:
                data = self.transform(data)
            data['name'] = os.path.basename(img_path)
            return data

    def __len__(self):
        return len(self.imglist)
