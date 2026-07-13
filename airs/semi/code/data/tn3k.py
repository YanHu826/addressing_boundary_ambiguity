import os

from torch.utils.data import Dataset
from torchvision import transforms
from utils.mytransforms import *
from utils.path_utils import get_split_file, resolve_split_entry


_TRAIN_SPLITS = {
    'label': {
        1: ('322', 'labeled.txt'),
        2: ('644', 'labeled.txt'),
        3: ('1289', 'labeled.txt'),
    },
    'unlabel': {
        1: ('322', 'unlabeled.txt'),
        2: ('644', 'unlabeled.txt'),
        3: ('1289', 'unlabeled.txt'),
    },
}
_DATA_SUBDIR = os.path.join('DATA', 'TN3K')
_IMAGE_MASK_DIRS = (
    ('trainval-image', 'trainval-mask'),
    ('test-image', 'test-mask'),
)


def _tn3k_mask_path(img_path):
    for image_dir, mask_dir in _IMAGE_MASK_DIRS:
        if image_dir in img_path:
            return img_path.replace(image_dir, mask_dir)
    raise FileNotFoundError(f'Unable to derive TN3K mask path from image path: {img_path}')


class tn3kDataSet(Dataset):
    def __init__(self, root, expID, mode='train', ratio=10, sign='label', transform=None):
        super(tn3kDataSet, self).__init__()
        self.mode = mode
        self.sign = sign
        if mode == 'train':
            split_parts = _TRAIN_SPLITS[sign].get(expID)
            if split_parts is None:
                raise ValueError(f'Unsupported expID {expID} for TN3K.')
            imgfile = get_split_file('TN3K', *split_parts)
            with open(imgfile, 'r') as f:
                imglist = f.read().splitlines()
                self.imglist = [resolve_split_entry(root, img, fallback_subdir=_DATA_SUBDIR) for img in imglist]
                print("[INFO] Loaded image paths ({} mode):".format(self.mode))
                for i, p in enumerate(self.imglist[:5]):
                    print(f"  [{i}] {p}")
                print(f"[INFO] Total images loaded: {len(self.imglist)}")
        elif mode == 'valid':
            imgfile = get_split_file('TN3K', 'val.txt')
            with open(imgfile, 'r') as f:
                imglist = f.read().splitlines()
                self.imglist = [resolve_split_entry(root, img, fallback_subdir=_DATA_SUBDIR) for img in imglist]
                print("[INFO] Loaded image paths ({} mode):".format(self.mode))
                for i, p in enumerate(self.imglist[:5]):
                    print(f"  [{i}] {p}")
                print(f"[INFO] Total images loaded: {len(self.imglist)}")
        elif mode == 'test':
            imgfile = get_split_file('TN3K', 'test.txt')
            with open(imgfile, 'r') as f:
                imglist = f.read().splitlines()
                self.imglist = [resolve_split_entry(root, img, fallback_subdir=_DATA_SUBDIR) for img in imglist]
                print("[INFO] Loaded image paths ({} mode):".format(self.mode))
                for i, p in enumerate(self.imglist[:5]):
                    print(f"  [{i}] {p}")
                print(f"[INFO] Total images loaded: {len(self.imglist)}")

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
                    ImageRandomZoom((0.9, 1.1)),
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

            gt_path = _tn3k_mask_path(img_path)

            gt = Image.open(gt_path).convert('L')
            data = {'image': img, 'label': gt}

            if self.transform:
                data = self.transform(data)

            data['name'] = self.imglist[index].split('/')[-1]

            return data

    def __len__(self):
        return len(self.imglist)

    def get_all_files(self, directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(file_path)
        return file_paths
