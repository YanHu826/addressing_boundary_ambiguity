from .tn3k import tn3kDataSet
from .BUSI import BUSIDataSet
from .UDIAT import UDIATDataSet
from .HC18 import HC18Dataset
from .PSFH import PSFHDataset
def build_dataset(args):
    if args.manner == 'test':
        if args.dataset == 'tn3k':
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
        elif args.dataset == 'BUSI':
            test_data = BUSIDataSet(args.root, args.expID, mode='test')
        elif args.dataset == 'UDIAT':
            test_data = UDIATDataSet(args.root, args.expID, mode='test')
        elif args.dataset == 'HC18':
            test_data = HC18Dataset(args.root, args.expID, mode='test')
        elif args.dataset == 'PSFH':
            test_data = PSFHDataset(args.root, args.expID, mode='test')
        return test_data
    else:
        if args.dataset == 'tn3k':
            train_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = tn3kDataSet(args.root, args.expID, mode='valid')
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
            train_u_data = None
            if args.manner == 'semi' or args.manner == 'self':
                train_u_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'BUSI':
            train_data = BUSIDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = BUSIDataSet(args.root, args.expID, mode='valid')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = BUSIDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'UDIAT':
            train_data = UDIATDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = UDIATDataSet(args.root, args.expID, mode='valid')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = UDIATDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'HC18':
            train_data = HC18Dataset(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = HC18Dataset(args.root, args.expID, mode='valid')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = HC18Dataset(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'PSFH':
            train_data = PSFHDataset(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = PSFHDataset(args.root, args.expID, mode='valid')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = PSFHDataset(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        return train_data, train_u_data, valid_data


