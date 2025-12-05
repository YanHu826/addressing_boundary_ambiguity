import os

PATH_MAPPING = {
    'train': r'/home/yh657/DATA/TN3K/trainval-image',
    'val': r'/home/yh657/DATA/TN3K/trainval-image',
    'test': r'/home/yh657/DATA/TN3K/test-image'
}

SUBDIRS = ['322', '644', '1289']
FILES = ['labeled.txt', 'unlabeled.txt']

# ===== Main Splits Directory =====
BASE_SPLIT_DIR = r'C:/Users/1/Desktop/master_degree/Shape-Prior-Semi-Seg/airs/data/splits/tn3k'

# ===== Separate global val/test paths =====
GLOBAL_TXT_FILES = [
    os.path.join(BASE_SPLIT_DIR, 'val.txt'),
    os.path.join(BASE_SPLIT_DIR, 'test.txt')
]

def convert_line(line):
    line = line.strip().replace('\\', '/')
    filename = os.path.basename(line)
    lowercase_line = line.lower()

    if 'train' in lowercase_line:
        return os.path.join(PATH_MAPPING['train'], filename).replace('\\', '/')
    elif 'val' in lowercase_line:
        return os.path.join(PATH_MAPPING['val'], filename).replace('\\', '/')
    elif 'test' in lowercase_line:
        return os.path.join(PATH_MAPPING['test'], filename).replace('\\', '/')
    else:
        return line


def convert_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"File does not exist: {input_path}")
        return
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = [convert_line(line) + '\n' for line in lines]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Complete: {output_path}")


def main():
    # Batch process the labeled and unlabeled in 322/644/1289
    for subdir in SUBDIRS:
        subdir_path = os.path.join(BASE_SPLIT_DIR, subdir)
        for file in FILES:
            input_file = os.path.join(subdir_path, file)
            output_file = os.path.join(subdir_path, file.replace('.txt', '.txt'))
            convert_file(input_file, output_file)

    # Process the global val.txt and test.txt files
    for file_path in GLOBAL_TXT_FILES:
        output_file = file_path.replace('.txt', '.txt')
        convert_file(file_path, output_file)


if __name__ == '__main__':
    main()
