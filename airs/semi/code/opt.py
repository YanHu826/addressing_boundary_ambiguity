import argparse

parse = argparse.ArgumentParser(description='PyTorch Semi-Medical-Seg Implement')

"-------------------GPU option----------------------------"
parse.add_argument('--GPUs', type=str, default='0')

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='/home/yh657/master_degree/')
parse.add_argument('--dataset', type=str, default='TN3K', choices=['TN3K','BUSI','UDIAT', 'HC18', 'PSFH'])
parse.add_argument('--ratio', type=int, default=10)

"-------------------training option-----------------------"
parse.add_argument('--manner', type=str, default='full', choices=['full', 'semi', 'test', 'self'])
parse.add_argument('--mode', type=str, default='train')
parse.add_argument('--nEpoch', type=int, default=200)
parse.add_argument('--batch_size', type=int, default=16)
parse.add_argument('--num_workers', type=int, default=2)
parse.add_argument('--load_ckpt', type=str, default=None)
parse.add_argument('--model', type=str, default='MyModel')
parse.add_argument('--expID', type=int, default=0) 
parse.add_argument('--ckpt_name', type=str, default=None)
parse.add_argument('--no_scd', action='store_true', help='Disable structure-aware discriminator')
parse.add_argument('--no_sor', action='store_true', help='Disable dropout decoder branch')
parse.add_argument('--no_ca', action='store_true', help='Disable CoordAttention in encoder')


"-------------------optimizer option-----------------------"
parse.add_argument('--lr', type=float, default=1e-4)
parse.add_argument('--power',type=float, default=0.9)
parse.add_argument('--betas', default=(0.9, 0.999))
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--eps', type=float, default=1e-8)
parse.add_argument('--mt', type=float, default=0.9)
parse.add_argument('--nclasses', type=int, default=1)
parse.add_argument('--band', type=int, default=3)

args = parse.parse_args()
