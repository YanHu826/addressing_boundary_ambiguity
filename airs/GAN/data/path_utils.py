import os
import re
from pathlib import Path


AIRS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = AIRS_ROOT.parent
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT.parent
_WINDOWS_ABS_RE = re.compile(r'^[A-Za-z]:[\\/]')


def normalize_root(root):
    if root is None or str(root).strip() == '':
        root = DEFAULT_WORKSPACE_ROOT
    return os.path.abspath(os.path.expanduser(str(root)))


def is_windows_absolute_path(path):
    return _WINDOWS_ABS_RE.match(str(path).strip()) is not None


def get_split_file(*parts):
    return str(AIRS_ROOT / 'data' / 'splits' / Path(*parts))


def extract_workspace_relative_path(raw_entry):
    entry = str(raw_entry).strip().replace('\\', '/')
    if entry == '':
        return None

    data_match = re.search(r'(?i)(?:^|/)DATA/(.+)$', entry)
    if data_match:
        return Path('DATA') / Path(*data_match.group(1).split('/'))

    project_data_match = re.search(r'(?i)(?:^|/)airs/data/(.+)$', entry)
    if project_data_match:
        return Path('airs') / 'data' / Path(*project_data_match.group(1).split('/'))

    return None


def resolve_split_entry(root, raw_entry, fallback_subdir=None):
    entry = str(raw_entry).strip().replace('\\', '/')
    workspace_root = Path(normalize_root(root))

    workspace_relative = extract_workspace_relative_path(entry)
    if workspace_relative is not None:
        return str(workspace_root / workspace_relative)

    if os.path.isabs(entry):
        return os.path.abspath(entry)

    if is_windows_absolute_path(entry):
        return entry

    if fallback_subdir:
        fallback_parts = str(fallback_subdir).replace('\\', '/').split('/')
        return str(workspace_root / Path(*fallback_parts) / Path(*entry.split('/')))

    return str(workspace_root / Path(*entry.split('/')))
