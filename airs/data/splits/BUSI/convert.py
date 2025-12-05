import os

def convert_line(line):
    filename = os.path.basename(line.strip())
    if 'benign' in line.lower():
        return f"C:/Users/1/Desktop/master_degree/DATA/BUSI/Dataset_BUSI_with_GT/benign/{filename}"
    elif 'malignant' in line.lower():
        return f"C:/Users/1/Desktop/master_degree/DATA/BUSI/Dataset_BUSI_with_GT/malignant/{filename}"
    elif 'normal' in line.lower():
        return f"C:/Users/1/Desktop/master_degree/DATA/BUSI/Dataset_BUSI_with_GT/normal/{filename}"
    else:
        return line.strip()

def convert_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        return
    print(f"✅ 正在处理：{file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = [convert_line(line) + '\n' for line in lines]

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前是 BUSI 目录
    split_dirs = ['72', '144', '288']
    files = ['labeled.txt', 'unlabeled.txt']

    # 处理 72/144/288 中的划分文件
    for subdir in split_dirs:
        for filename in files:
            file_path = os.path.join(current_dir, subdir, filename)
            convert_file(file_path)

    # 处理 BUSI 目录下的 val.txt
    convert_file(os.path.join(current_dir, 'val.txt'))

if __name__ == "__main__":
    main()
