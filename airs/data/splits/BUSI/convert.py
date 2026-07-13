from pathlib import Path


DATASET_ROOT = Path('DATA') / 'BUSI' / 'Dataset_BUSI_with_GT'


def convert_line(line):
    entry = line.strip().replace('\\', '/')
    filename = Path(entry).name
    lower_entry = entry.lower()

    if 'benign' in lower_entry:
        return str(DATASET_ROOT / 'benign' / filename).replace('\\', '/')
    if 'malignant' in lower_entry:
        return str(DATASET_ROOT / 'malignant' / filename).replace('\\', '/')
    if 'normal' in lower_entry:
        return str(DATASET_ROOT / 'normal' / filename).replace('\\', '/')
    return entry


def convert_file(file_path):
    if not file_path.exists():
        print(f"File does not exist: {file_path}")
        return

    with file_path.open('r', encoding='utf-8') as handle:
        lines = handle.readlines()

    new_lines = [convert_line(line) + '\n' for line in lines]

    with file_path.open('w', encoding='utf-8') as handle:
        handle.writelines(new_lines)


def main():
    current_dir = Path(__file__).resolve().parent
    split_dirs = ['72', '144', '288']
    files = ['labeled.txt', 'unlabeled.txt']

    for subdir in split_dirs:
        for filename in files:
            convert_file(current_dir / subdir / filename)

    convert_file(current_dir / 'val.txt')

if __name__ == "__main__":
    main()
