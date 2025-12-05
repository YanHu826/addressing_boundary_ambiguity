import os

# ========== 配置 ==========
LABELED_TXT = r'C:\Users\1\Desktop\master_degree\Shape-Prior-Semi-Seg\airs\data\splits\BUSI\288\labeled.txt'
UNLABELED_TXT = r'C:\Users\1\Desktop\master_degree\Shape-Prior-Semi-Seg\airs\data\splits\BUSI\288\unlabeled.txt'
VAL_TXT = r'C:\Users\1\Desktop\master_degree\Shape-Prior-Semi-Seg\airs\data\splits\BUSI\val.txt'

# ======== 输出文件路径 ========
OUTPUT_LABELED_1_8 = r'C:\Users\1\Desktop\master_degree\DATA\BUSI_AUG\labeled_1_2.txt'
OUTPUT_UNLABELED_1_8 = r'C:\Users\1\Desktop\master_degree\DATA\BUSI_AUG\unlabeled_1_2.txt'

NUM_AUG_PER_IMAGE = 5  # 每张增强多少张

# ========== 工具函数 ==========
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def write_txt(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

# ========== 主逻辑 ==========
def generate_labeled_unlabeled_txt():
    labeled_paths = read_txt(LABELED_TXT)
    unlabeled_paths = read_txt(UNLABELED_TXT)
    val_names = set(os.path.basename(line.strip()) for line in read_txt(VAL_TXT))

    labeled_augmented = []

    for path in labeled_paths:
        filename = os.path.basename(path)
        if filename in val_names:
            continue  # ✅ 跳过验证集图像

        base, ext = os.path.splitext(path)
        # 原图保留
        labeled_augmented.append(path)
        # 增强图追加
        for i in range(NUM_AUG_PER_IMAGE):
            labeled_augmented.append(f"{base}_aug{i}{ext}")

    # 写出文件
    write_txt(OUTPUT_LABELED_1_8, labeled_augmented)
    write_txt(OUTPUT_UNLABELED_1_8, unlabeled_paths)

    print(f"✅ labeled_1_8.txt: 共 {len(labeled_augmented)} 条（含原图 + 增强，已排除 val）")
    print(f"✅ unlabeled_1_8.txt: 共 {len(unlabeled_paths)} 条（未变动）")

# ========== 执行 ==========
if __name__ == '__main__':
    generate_labeled_unlabeled_txt()