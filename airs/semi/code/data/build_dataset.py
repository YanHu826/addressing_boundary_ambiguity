from .tn3k import tn3kDataSet
from .BUSI import BUSIDataSet
from .UDIAT import UDIATDataSet
from .HC18 import HC18Dataset
from .PSFH import PSFHDataset
from utils.path_utils import canonical_dataset_name


def build_eval_dataset(args, split='test'):
    dataset = canonical_dataset_name(args.dataset)
    if split not in ('valid', 'test'):
        raise ValueError(f'Unsupported eval split: {split}')

    if dataset == 'TN3K':
        return tn3kDataSet(args.root, args.expID, mode=split)
    if dataset == 'BUSI':
        return BUSIDataSet(args.root, args.expID, mode=split)
    if dataset == 'UDIAT':
        return UDIATDataSet(args.root, args.expID, mode=split)
    if dataset == 'HC18':
        return HC18Dataset(args.root, args.expID, mode=split)
    if dataset == 'PSFH':
        return PSFHDataset(args.root, args.expID, mode=split)
    raise ValueError(f'Unsupported dataset: {dataset}')


def build_dataset(args):
    dataset = canonical_dataset_name(args.dataset)

    if args.manner == 'test':
        return build_eval_dataset(args, split='test')
    else:
        if dataset == 'TN3K':
            train_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = tn3kDataSet(args.root, args.expID, mode='valid')
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
            train_u_data = None
            if args.manner == 'semi' or args.manner == 'self':
                train_u_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif dataset == 'BUSI':
            train_data = BUSIDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = BUSIDataSet(args.root, args.expID, mode='valid')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = BUSIDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif dataset == 'UDIAT':
            train_data = UDIATDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = UDIATDataSet(args.root, args.expID, mode='valid')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = UDIATDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif dataset == 'HC18':
            train_data = HC18Dataset(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = HC18Dataset(args.root, args.expID, mode='valid')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = HC18Dataset(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif dataset == 'PSFH':
            train_data = PSFHDataset(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = PSFHDataset(args.root, args.expID, mode='valid')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = PSFHDataset(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        return train_data, train_u_data, valid_data
