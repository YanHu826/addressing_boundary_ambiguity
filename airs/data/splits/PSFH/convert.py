import os

# 本地路径前缀（Windows）
local_prefix = r"/home/yh657/DATA/PSFH/image_png"
# 服务器路径前缀（Linux）
server_prefix = "/home/yh657/DATA/PSFH/image_png"

# 要扫描的路径（包含 labeled.txt / unlabeled.txt / test.txt）
splits_dir = r"/content/drive/MyDrive/Shape-Prior-Semi-Seg/airs/data/splits/PSFH"

def convert_line(line):
    line = line.strip()
    if line.startswith(local_prefix):
        return line.replace(local_prefix, server_prefix).replace("\\", "/")
    return line.replace("\\", "/")

def convert_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = [convert_line(line) + '\n' for line in lines]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"✅ 已处理: {file_path}")

def main():
    for root, _, files in os.walk(splits_dir):
        for file in files:
            if file in ["labeled.txt", "unlabeled.txt", "test.txt"]:
                convert_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
