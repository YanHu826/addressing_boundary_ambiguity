import os
import re
from pathlib import Path


_CANONICAL_DATASETS = {
    'tn3k': 'TN3K',
    'busi': 'BUSI',
    'udiat': 'UDIAT',
    'hc18': 'HC18',
    'psfh': 'PSFH',
}

_SPLIT_DIRS = {
    'TN3K': 'tn3k',
    'BUSI': 'BUSI',
    'UDIAT': 'UDIAT',
    'HC18': 'HC18',
    'PSFH': 'PSFH',
}

AIRS_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = AIRS_ROOT.parent
SEMI_ROOT = AIRS_ROOT / 'semi'
CODE_ROOT = SEMI_ROOT / 'code'
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT.parent
_WINDOWS_ABS_RE = re.compile(r'^[A-Za-z]:[\\/]')


def resolve_env_path(env_var, default_path):
    override = os.getenv(env_var)
    if override is None or str(override).strip() == '':
        return str(default_path)
    return os.path.abspath(os.path.expanduser(str(override)))


def canonical_dataset_name(value):
    if value is None:
        raise ValueError('Dataset name is required.')

    key = str(value).strip().lower()
    if key not in _CANONICAL_DATASETS:
        expected = ', '.join(sorted(_CANONICAL_DATASETS.values()))
        raise ValueError(f"Unsupported dataset '{value}'. Expected one of: {expected}")

    return _CANONICAL_DATASETS[key]


def normalize_root(root):
    if root is None or str(root).strip() == '':
        root = DEFAULT_WORKSPACE_ROOT
    return os.path.abspath(os.path.expanduser(str(root)))


def is_windows_absolute_path(path):
    return _WINDOWS_ABS_RE.match(str(path).strip()) is not None


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


def get_split_file(dataset, *parts):
    dataset_name = canonical_dataset_name(dataset)
    return str(AIRS_ROOT / 'data' / 'splits' / _SPLIT_DIRS[dataset_name] / Path(*parts))


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


def get_checkpoint_root():
    return resolve_env_path('AIRS_SEMI_CHECKPOINT_ROOT', SEMI_ROOT / 'checkpoint')


def get_backbone_pretrain_path():
    return resolve_env_path('AIRS_BACKBONE_PRETRAIN_PATH', CODE_ROOT / 'pretrain' / 'backbone' / 'resnet34.pth')


def get_scd_pretrain_path():
    return resolve_env_path('AIRS_SCD_PRETRAIN_PATH', CODE_ROOT / 'models' / 'pretrain' / 'GAN' / 'netD_epoch_10000.pth')
