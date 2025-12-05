import os

from torch.utils.data import Dataset
from torchvision import transforms
from utils.mytransforms import *

data_address = 'C:/Users/1/Desktop/master_degree/Shape-Prior-Semi-Seg/airs'
class tn3kDataSet(Dataset):
    def __init__(self, root, expID, mode='train', ratio=10, sign='label', transform=None):
        super(tn3kDataSet, self).__init__()
        self.mode = mode
        self.sign = sign
        if mode == 'train':
            if sign == 'label':
                if(expID == 1):
                    imgfile = data_address + '/data/splits/tn3k/322/labeled.txt'
                elif(expID == 2):
                    imgfile = data_address + '/data/splits/tn3k/644/labeled.txt'
                elif(expID == 3):
                    imgfile = data_address + '/data/splits/tn3k/1289/labeled.txt'

                with open(imgfile, 'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'DATA/TN3K', img) for img in imglist]
                    print("[INFO] Loaded image paths ({} mode):".format(self.mode))
                    for i, p in enumerate(self.imglist[:5]):
                        print(f"  [{i}] {p}")
                    print(f"[INFO] Total images loaded: {len(self.imglist)}")
            else:
                if(expID == 1):
                    imgfile = data_address + '/data/splits/tn3k/322/unlabeled.txt'
                elif(expID == 2):
                    imgfile = data_address + '/data/splits/tn3k/644/unlabeled.txt'
                elif(expID == 3):
                    imgfile = data_address + '/data/splits/tn3k/1289/unlabeled.txt'
                with open(imgfile, 'r') as f:
                    imglist = f.read().splitlines()
                    self.imglist = [os.path.join(root, 'DATA/TN3K', img) for img in imglist]
                    print("[INFO] Loaded image paths ({} mode):".format(self.mode))
                    for i, p in enumerate(self.imglist[:5]):
                        print(f"  [{i}] {p}")
                    print(f"[INFO] Total images loaded: {len(self.imglist)}")
        elif mode == 'valid':
            imgfile = data_address + '/data/splits/tn3k/val.txt'
            with open(imgfile, 'r') as f:
                imglist = f.read().splitlines()
                self.imglist = [os.path.join(root, 'DATA/TN3K', img) for img in imglist]
                print("[INFO] Loaded image paths ({} mode):".format(self.mode))
                for i, p in enumerate(self.imglist[:5]):
                    print(f"  [{i}] {p}")
                print(f"[INFO] Total images loaded: {len(self.imglist)}")
        elif mode == 'test':
            imgfile = data_address + '/data/splits/tn3k/val.txt'
            with open(imgfile, 'r') as f:
                imglist = f.read().splitlines()
                self.imglist = [os.path.join(root, 'DATA/TN3K', img) for img in imglist]
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

            gt_path = img_path.replace('trainval-image', 'trainval-mask')
            gt_path = gt_path.replace('.jpg', '.jpg')

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
