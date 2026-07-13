from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = PROJECT_ROOT.parent
INPUT_DIR = WORKSPACE_ROOT / 'DATA' / 'TN3K' / 'test-image'
OUTPUT_FILE = Path(__file__).resolve().parent / 'test.txt'


def list_files_in_directory(directory):
    return sorted(path for path in directory.rglob('*') if path.is_file())


def main():
    entries = []
    for path in list_files_in_directory(INPUT_DIR):
        relative_path = path.relative_to(INPUT_DIR)
        entries.append((Path('DATA') / 'TN3K' / 'test-image' / relative_path).as_posix())

    with OUTPUT_FILE.open('w', encoding='utf-8') as handle:
        for entry in entries:
            handle.write(entry + '\n')


if __name__ == '__main__':
    main()
