import os

# 本地路径前缀（Windows）
local_prefix_train = r"C:/Users/1/Desktop/master_degree/DATA/HC18/training_set"
local_prefix_test  = r"C:/Users/1/Desktop/master_degree/DATA/HC18/test_set"

# 服务器路径前缀（Linux）
server_prefix_train = "/home/yh657/DATA/HC18/training_set"
server_prefix_test  = "/home/yh657/DATA/HC18/test_set"

# 要扫描的路径
splits_dir = r"C:/Users/1/Desktop/master_degree/Shape-Prior-Semi-Seg/airs/data/splits/HC18"

def convert_line(line):
    line = line.strip()
    if line.startswith(local_prefix_train):
        return line.replace(local_prefix_train, server_prefix_train).replace("\\", "/")
    if line.startswith(local_prefix_test):
        return line.replace(local_prefix_test, server_prefix_test).replace("\\", "/")
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
            if file.endswith(".txt"):   # 处理所有 txt 文件
                convert_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
