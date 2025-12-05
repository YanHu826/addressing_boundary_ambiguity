import os

# ========== 配置 ==========

LABELED_TXT = r'C:\Users\1\Desktop\master_degree\Shape-Prior-Semi-Seg\airs\data\splits\UDIAT\73\labeled.txt'
UNLABELED_TXT = r'C:\Users\1\Desktop\master_degree\Shape-Prior-Semi-Seg\airs\data\splits\UDIAT\73\unlabeled.txt'
VAL_TXT = r'C:\Users\1\Desktop\master_degree\Shape-Prior-Semi-Seg\airs\data\splits\UDIAT\val.txt'

# ======== 输出文件路径 ========
OUTPUT_LABELED = r'C:/Users/1/Desktop/master_degree/DATA/UDIAT_AUG/labeled_1_2.txt'
OUTPUT_UNLABELED = r'C:/Users/1/Desktop/master_degree/DATA/UDIAT_AUG/unlabeled_1_2.txt'

NUM_AUG_PER_IMAGE = 5  # 每张原图增强几张

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
    val_names = set(os.path.basename(p) for p in read_txt(VAL_TXT))

    labeled_augmented = []

    for path in labeled_paths:
        filename = os.path.basename(path)
        if filename in val_names:
            continue  # ✅ 跳过验证集

        base, ext = os.path.splitext(path)
        labeled_augmented.append(path)  # 原图
        for i in range(NUM_AUG_PER_IMAGE):  # 增强图
            labeled_augmented.append(f"{base}_aug{i}{ext}")

    write_txt(OUTPUT_LABELED, labeled_augmented)
    write_txt(OUTPUT_UNLABELED, unlabeled_paths)

    print(f"✅ labeled_1_2.txt: 共 {len(labeled_augmented)} 条（含增强，排除 val）")
    print(f"✅ unlabeled_1_2.txt: 共 {len(unlabeled_paths)} 条（未变动）")

# ========== 执行 ==========
if __name__ == '__main__':
    generate_labeled_unlabeled_txt()