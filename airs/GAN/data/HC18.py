import os
from torch.utils.data import Dataset
from torchvision import transforms
from .utils.mytransforms import *
from PIL import Image
import numpy as np
import cv2
# 改成你的 split 文件所在路径
data_address = r"/home/yh657/Shape-Prior-Semi-Seg/airs"

class HC18Dataset(Dataset):
    def __init__(self, root, expID, mode='train', ratio=10, sign='label', transform=None, label_mode='region'):
        super().__init__()
        self.mode = mode
        self.sign = sign
        self.label_mode = label_mode
        # 选 split 文件
        if mode == 'train':
            if sign == 'label':
                imgfile = data_address + ('/data/splits/HC18/107/labeled.txt' if expID == 1
                                          else '/data/splits/HC18/214/labeled.txt')
            else:
                imgfile = data_address + ('/data/splits/HC18/107/unlabeled.txt' if expID == 1
                                          else '/data/splits/HC18/214/unlabeled.txt')
        else:  # valid/test
            imgfile = data_address + '/data/splits/HC18/test.txt'

        # 读取行（统一用 lines，不再对同一个 f 二次遍历）
        with open(imgfile, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # 组织样本列表并做存在性校验
        if self.mode == 'train' and self.sign == 'unlabel':
            # 仅图片路径
            imgs = []
            for p in lines:
                if os.path.exists(p):
                    imgs.append(p)
                else:
                    print(f"[警告] 找不到图片文件: {p}")
            self.imglist = imgs
        else:
            # 成对的 (img, mask)
            pairs = []
            for p in lines:
                img_path = p
                mask_path = p.replace('.png', '_Annotation.png')
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
            img = Image.open(img_path).convert('RGB')
            return self.transform(img) if self.transform else img
        else:
            img_path, gt_path = self.imglist[index]
            img = Image.open(img_path).convert('RGB')

            # 原始边界 GT（灰度：细白边，黑背景）
            gt_boundary = Image.open(gt_path).convert('L')

            # 如果需要区域 GT，就把边界填充
            if self.label_mode in ('region', 'both'):
                gt_np = np.array(gt_boundary)
                _, thresh = cv2.threshold(gt_np, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(gt_np)
                cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
                gt_region = Image.fromarray(filled)

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

