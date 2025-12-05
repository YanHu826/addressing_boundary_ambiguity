import os
import cv2
import albumentations as A
from albumentations.augmentations.dropout import CoarseDropout
from glob import glob
from tqdm import tqdm

# ======== 子集结构（图像目录, 掩膜目录）========
subsets = [
    ('trainval-image', 'trainval-mask'),
    ('test-image', 'test-mask')
]

base_input_dir = r'C:/Users/1/Desktop/master_degree/DATA/TN3K'
base_output_dir = r'C:/Users/1/Desktop/master_degree/DATA/TN3K_AUG'
num_aug_per_image = 5

# ======== 增强器配置 ========
transform = A.Compose([
    A.Rotate(limit=30, p=0.8),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    CoarseDropout(
        max_holes=1, max_height=32, max_width=32,
        min_holes=1, min_height=32, min_width=32,
        fill_value=0, p=0.5
    ),
    A.GaussianBlur(p=0.3)
])

# ======== 主处理流程 ========
for img_folder, mask_folder in subsets:
    img_in_dir = os.path.join(base_input_dir, img_folder)
    mask_in_dir = os.path.join(base_input_dir, mask_folder)

    img_out_dir = os.path.join(base_output_dir, img_folder)
    mask_out_dir = os.path.join(base_output_dir, mask_folder)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    labeled_txt = os.path.join(img_out_dir, 'labeled.txt')
    unlabeled_txt = os.path.join(img_out_dir, 'unlabeled.txt')
    labeled_list = []
    unlabeled_list = []

    image_paths = sorted(glob(os.path.join(img_in_dir, '*.jpg')))
    print(f"📂 正在处理：{img_folder}，图像数：{len(image_paths)}")

    for img_path in tqdm(image_paths):
        fname = os.path.basename(img_path)
        mask_path = os.path.join(mask_in_dir, fname)

        if not os.path.exists(mask_path):
            print(f"⚠️ 掩膜缺失，跳过：{mask_path}")
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 保存原图和掩膜
        cv2.imwrite(os.path.join(img_out_dir, fname), img)
        cv2.imwrite(os.path.join(mask_out_dir, fname), mask)
        unlabeled_list.append(fname)

        # 增强图像和掩膜
        base_name = os.path.splitext(fname)[0]
        for i in range(num_aug_per_image):
            aug = transform(image=img, mask=mask)
            aug_img = aug['image']
            aug_mask = aug['mask']

            aug_fname = f"{base_name}_aug{i}.jpg"
            cv2.imwrite(os.path.join(img_out_dir, aug_fname), aug_img)
            cv2.imwrite(os.path.join(mask_out_dir, aug_fname), aug_mask)
            labeled_list.append(aug_fname)

    # 写入清单
    with open(labeled_txt, 'w') as f:
        f.writelines([line + '\n' for line in labeled_list])
    with open(unlabeled_txt, 'w') as f:
        f.writelines([line + '\n' for line in unlabeled_list])

    print(f"✅ 完成增强：{img_folder}")
    print(f"📄 labeled: {len(labeled_list)}，unlabeled: {len(unlabeled_list)} → 存储于 {img_out_dir}")
