import os
import cv2
import albumentations as A
from albumentations.augmentations.dropout import CoarseDropout
from glob import glob
from tqdm import tqdm

# ========== 配置 ==========
base_input_dir = r'C:\Users\1\Desktop\master_degree\DATA\BUSI\Dataset_BUSI_with_GT'
base_output_dir = r'C:\Users\1\Desktop\master_degree\DATA\BUSI_AUG'
classes = ['benign', 'malignant', 'normal']

num_aug_per_image = 5  # 每张图增强几张

# ========== 增强器 ==========
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

# ========== 初始化TXT清单 ==========
labeled_txt = os.path.join(base_output_dir, 'labeled.txt')
unlabeled_txt = os.path.join(base_output_dir, 'unlabeled.txt')

labeled_list = []
unlabeled_list = []

# ========== 开始处理 ==========
for cls in classes:
    print(f"🧩 处理类别：{cls}")
    input_dir = os.path.join(base_input_dir, cls)
    output_dir = os.path.join(base_output_dir, cls)
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(glob(os.path.join(input_dir, '*.png')))
    images = [img for img in images if '_mask' not in img]

    for img_path in tqdm(images):
        mask_path = img_path.replace('.png', '_mask.png')

        # 读取原图与掩膜
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        base = os.path.splitext(os.path.basename(img_path))[0]

        # 保存原图（用于unlabeled清单）
        new_img_path = os.path.join(output_dir, base + '.png')
        new_mask_path = os.path.join(output_dir, base + '_mask.png')
        cv2.imwrite(new_img_path, img)
        cv2.imwrite(new_mask_path, mask)
        unlabeled_list.append(f"{cls}/{base}.png")

        # 开始增强
        for i in range(num_aug_per_image):
            aug = transform(image=img, mask=mask)
            aug_img = aug['image']
            aug_mask = aug['mask']

            aug_img_name = f"{base}_aug{i}.png"
            aug_mask_name = f"{base}_aug{i}_mask.png"

            cv2.imwrite(os.path.join(output_dir, aug_img_name), aug_img)
            cv2.imwrite(os.path.join(output_dir, aug_mask_name), aug_mask)

            labeled_list.append(f"{cls}/{aug_img_name}")

# ========== 写入TXT ==========
with open(labeled_txt, 'w') as f:
    for item in labeled_list:
        f.write(f"{item}\n")

with open(unlabeled_txt, 'w') as f:
    for item in unlabeled_list:
        f.write(f"{item}\n")

print("✅ 增强完成，文件已保存至 BUSI_AUG/")
print(f"📄 labeled.txt 条目数: {len(labeled_list)}")
print(f"📄 unlabeled.txt 条目数: {len(unlabeled_list)}")
