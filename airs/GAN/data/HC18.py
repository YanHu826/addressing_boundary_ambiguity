import os
from torch.utils.data import Dataset
from torchvision import transforms
from .utils.mytransforms import *
from PIL import Image
from .image_utils import build_region_mask, load_grayscale_image, load_rgb_image
from .path_utils import get_split_file, resolve_split_entry

_DATA_SUBDIR = os.path.join('DATA', 'HC18')
def _entry_subset(raw_entry):
    parts = str(raw_entry).replace('\\', '/').split('/')
    for idx, part in enumerate(parts):
        if part == 'HC18' and idx + 1 < len(parts):
            return parts[idx + 1]
    if parts and parts[0] in ('training_set', 'test_set'):
        return parts[0]
    return 'unknown'

def _load_holdout_entries():
    entries = set()
    for split_name in ('val.txt', 'test.txt'):
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
        self.filtered_official_test_set_count = 0
        # 选 split 文件
        if mode == 'train':
            if sign == 'label':
                imgfile = get_split_file('HC18', '107' if expID == 1 else '214', 'labeled.txt')
            else:
                imgfile = get_split_file('HC18', '107' if expID == 1 else '214', 'unlabeled.txt')
        else:  # valid/test
            imgfile = get_split_file('HC18', 'test.txt')

        # 读取行（统一用 lines，不再对同一个 f 二次遍历）
        with open(imgfile, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if mode == 'train':
            holdout_entries = _load_holdout_entries()
            if holdout_entries:
                lines = [line for line in lines if line not in holdout_entries]
            # Paper protocol: never use HC18 official test_set as unlabeled.
            before_test_filter = len(lines)
            lines = [line for line in lines if _entry_subset(line) != 'test_set']
            self.filtered_official_test_set_count = before_test_filter - len(lines)

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

        self.region_mask_cache = {}
        if self.mode != 'train' or self.sign != 'unlabel':
            if self.label_mode in ('region', 'both'):
                unique_mask_paths = {mask_path for _, mask_path in self.imglist}
                self.region_mask_cache = {
                    mask_path: build_region_mask(load_grayscale_image(mask_path), mode='ellipse')
                    for mask_path in unique_mask_paths
                }

        # transforms（保留你原来的）
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
            else:  # valid/test
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

            # 如果需要区域 GT，就把边界填充
            if self.label_mode in ('region', 'both'):
                gt_region = self.region_mask_cache[gt_path].copy()

            # 组装返回
            if self.label_mode == 'region':
                data = {'image': img, 'label': gt_region}
            elif self.label_mode == 'boundary':
                data = {'image': img, 'label': gt_boundary}
            else:  # 'both'
                data = {'image': img, 'label_region': gt_region, 'label_boundary': gt_boundary}

            if self.transform:
                data = self.transform(data)
            data['name'] = os.path.basename(img_path)
            return data

    def __len__(self):
        return len(self.imglist)
