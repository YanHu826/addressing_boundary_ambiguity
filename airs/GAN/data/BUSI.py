import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils.mytransforms import *
from .path_utils import get_split_file, resolve_split_entry

_TRAIN_SPLITS = {
    'label': {
        1: ('BUSI', '72', 'labeled.txt'),
        2: ('BUSI', '144', 'labeled.txt'),
        3: ('BUSI', '288', 'labeled.txt'),
    },
    'unlabel': {
        1: ('BUSI', '72', 'unlabeled.txt'),
        2: ('BUSI', '144', 'unlabeled.txt'),
        3: ('BUSI', '288', 'unlabeled.txt'),
    },
}
_DATA_SUBDIR = os.path.join('DATA', 'BUSI', 'Dataset_BUSI_with_GT')


def _mask_candidates(img_path):
    stem, ext = os.path.splitext(img_path)
    base_mask = stem + '_mask' + ext
    extra_masks = sorted(glob(stem + '_mask_[0-9]*' + ext))

    candidates = []
    if os.path.exists(base_mask):
        candidates.append(base_mask)
    candidates.extend(path for path in extra_masks if path not in candidates)
    if not candidates:
        raise FileNotFoundError(f'No BUSI mask found for image: {img_path}')
    return candidates


def _load_busi_mask(img_path):
    merged_mask = None
    for mask_path in _mask_candidates(img_path):
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
        merged_mask = mask if merged_mask is None else np.maximum(merged_mask, mask)
    return Image.fromarray(merged_mask, mode='L')


class BUSIDataSet(Dataset):
    def __init__(self, root, expID, mode='train', ratio=10, sign='label', transform=None):
        super(BUSIDataSet, self).__init__()
        self.mode = mode
        self.sign = sign
        if mode == 'train':
            split_parts = _TRAIN_SPLITS[sign].get(expID)
            if split_parts is None:
                raise ValueError(f'Unsupported expID {expID} for BUSI.')
            imgfile = get_split_file(*split_parts)
            with open(imgfile, 'r') as f:
                imglist = f.read().splitlines()
                self.imglist = [resolve_split_entry(root, img, fallback_subdir=_DATA_SUBDIR) for img in imglist]
                print("self.imglist: ", self.imglist)
        elif mode == 'valid':
            imgfile = get_split_file('BUSI', 'val.txt')
            with open(imgfile, 'r') as f:
                imglist = f.read().splitlines()
                self.imglist = [resolve_split_entry(root, img, fallback_subdir=_DATA_SUBDIR) for img in imglist]
        elif mode == 'test':
            imgfile = get_split_file('BUSI', 'val.txt')
            with open(imgfile, 'r') as f:
                imglist = f.read().splitlines()
                self.imglist = [resolve_split_entry(root, img, fallback_subdir=_DATA_SUBDIR) for img in imglist]

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
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((320, 320)),
                   ToTensor()
                ])
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train' and self.sign == 'unlabel':
            img_path = self.imglist[index]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                return self.transform(img)
        else:
            img_path = self.imglist[index]
            img = Image.open(img_path).convert('RGB')
            if '_mask' in img_path:
                gt = Image.open(img_path).convert('L')
            else:
                gt = _load_busi_mask(img_path)
            data = {'image': img, 'label': gt}
            if self.transform:
                data = self.transform(data)
            data['name'] = self.imglist[index].split('/')[-1]
            return data

    def __len__(self):
        return len(self.imglist)
    
    def get_all_files(self,directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(file_path)
        return file_paths
