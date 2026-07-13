from pathlib import Path


def convert_line(line):
    entry = line.strip().replace('\\', '/')
    lower_entry = entry.lower()

    for marker in ('image_png/', 'label_png/'):
        marker_index = lower_entry.find(marker)
        if marker_index != -1:
            suffix = entry[marker_index:]
            return f"DATA/PSFH/{suffix}"

    return entry


def convert_file(file_path):
    with file_path.open('r', encoding='utf-8') as handle:
        lines = handle.readlines()

    new_lines = [convert_line(line) + '\n' for line in lines]

    with file_path.open('w', encoding='utf-8') as handle:
        handle.writelines(new_lines)


def main():
    splits_dir = Path(__file__).resolve().parent
    for file_path in sorted(splits_dir.rglob('*.txt')):
        if file_path.name in {'labeled.txt', 'unlabeled.txt', 'test.txt'}:
            convert_file(file_path)

if __name__ == "__main__":
    main()
