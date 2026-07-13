from pathlib import Path


PATH_MAPPING = {
    'trainval-image/': 'DATA/TN3K/trainval-image',
    'test-image/': 'DATA/TN3K/test-image',
}
SUBDIRS = ['322', '644', '1289']
FILES = ['labeled.txt', 'unlabeled.txt']


def convert_line(line):
    entry = line.strip().replace('\\', '/')
    lower_entry = entry.lower()

    for marker, target_prefix in PATH_MAPPING.items():
        marker_index = lower_entry.find(marker)
        if marker_index != -1:
            suffix = entry[marker_index + len(marker):]
            return f"{target_prefix}/{suffix}"

    filename = Path(entry).name
    if 'test' in lower_entry:
        return f"DATA/TN3K/test-image/{filename}"
    if filename:
        return f"DATA/TN3K/trainval-image/{filename}"
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
    base_split_dir = Path(__file__).resolve().parent

    for subdir in SUBDIRS:
        for filename in FILES:
            convert_file(base_split_dir / subdir / filename)

    for filename in ('val.txt', 'test.txt'):
        convert_file(base_split_dir / filename)


if __name__ == '__main__':
    main()
