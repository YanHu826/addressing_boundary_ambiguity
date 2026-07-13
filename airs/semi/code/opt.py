import argparse
import sys
import torch

from utils.path_utils import (
    DEFAULT_WORKSPACE_ROOT,
    PROJECT_ROOT,
    SEMI_ROOT,
    canonical_dataset_name,
    get_checkpoint_root,
    normalize_root,
)

parse = argparse.ArgumentParser(description='PyTorch Semi-Medical-Seg Implement')

DEFAULT_VISIBLE_GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEFAULT_VISIBLE_GPU_IDS = ','.join(str(idx) for idx in range(DEFAULT_VISIBLE_GPU_COUNT))
DEFAULT_BATCH_SIZE = 24 if DEFAULT_VISIBLE_GPU_COUNT >= 8 else 16
if DEFAULT_VISIBLE_GPU_COUNT >= 8:
    DEFAULT_NUM_WORKERS = 8
elif DEFAULT_VISIBLE_GPU_COUNT >= 4:
    DEFAULT_NUM_WORKERS = 6
else:
    DEFAULT_NUM_WORKERS = 4
DEFAULT_EPOCHS = 240 if DEFAULT_VISIBLE_GPU_COUNT >= 8 else 200
DEFAULT_LR = 1.25e-4 if DEFAULT_VISIBLE_GPU_COUNT >= 8 else 1e-4

"-------------------GPU option----------------------------"
parse.add_argument('--GPUs', type=str, default=DEFAULT_VISIBLE_GPU_IDS)

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default=str(DEFAULT_WORKSPACE_ROOT),
                   help='Workspace root containing both the project directory and a sibling DATA directory.')
parse.add_argument('--dataset', type=canonical_dataset_name, default='TN3K',
                   choices=['TN3K', 'BUSI', 'UDIAT', 'HC18', 'PSFH'])
parse.add_argument('--ratio', type=int, default=10)

"-------------------training option-----------------------"
parse.add_argument('--manner', type=str, default='full', choices=['full', 'semi', 'test', 'self'])
parse.add_argument('--mode', type=str, default='train')
parse.add_argument('--nEpoch', type=int, default=DEFAULT_EPOCHS)
parse.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
parse.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
parse.add_argument('--precision', type=str, default='auto', choices=['auto', 'fp32', 'bf16'],
                   help='Numerical precision for CUDA execution. "auto" keeps semi training in fp32 for stability; use bf16 explicitly to opt in.')
parse.add_argument('--tf32', dest='tf32', action='store_true',
                   help='Enable TF32 matmul/cuDNN kernels on Ampere/Hopper GPUs for higher throughput.')
parse.add_argument('--no-tf32', dest='tf32', action='store_false',
                   help='Disable TF32 matmul/cuDNN kernels.')
parse.add_argument('--log_interval', type=int, default=10,
                   help='Print batch-level training logs every N batches. Set to 0 to disable periodic logging.')
parse.add_argument('--log_first_batches', type=int, default=3,
                   help='Always print the first N batches of each epoch.')
parse.add_argument('--disable_tqdm', action='store_true',
                   help='Disable tqdm progress bars and rely on plain-text logs only.')
parse.add_argument('--val_interval', type=int, default=1,
                   help='Run validation every N epochs during training. The last epoch always validates.')
parse.add_argument('--eval_threshold', type=float, default=0.5,
                   help='Binarization threshold used by the default BUSI/TN3K/UDIAT validation and test post-processing branch. '
                        'When omitted, BUSI defaults to 0.48, TN3K to 0.50, and other datasets keep 0.50.')
parse.add_argument('--eval_min_area', type=int, default=200,
                   help='Minimum connected-component area kept by the default BUSI/TN3K/UDIAT validation and test post-processing branch. '
                        'When omitted, BUSI defaults to 100, TN3K to 150, and other datasets keep 200.')
parse.add_argument('--eval_postprocess_mode', type=str, default='raw', choices=['raw', 'simple', 'legacy'],
                   help='Post-processing used for validation/test. "raw" only thresholds probabilities; "simple" adds minimal component cleanup; "legacy" reproduces older dataset-specific heuristics.')
parse.add_argument('--auto_tune_eval_postprocess', action='store_true',
                   help='Before final test, sweep threshold/min-area on the validation split and reuse the best pair for test evaluation. '
                        'BUSI/TN3K default to on when omitted.')
parse.add_argument('--no_auto_tune_eval_postprocess', dest='auto_tune_eval_postprocess', action='store_false',
                   help='Disable the validation-driven post-processing sweep before final test.')
parse.add_argument('--save_best_tta', action='store_true',
                   help='When saving best validation predictions during semi training, rerun validation with TTA enabled.')
parse.add_argument('--save_reliability_maps', action='store_true',
                   help='Save image/GT/prediction/probability/reliability/error maps during save_best/test evaluation.')
parse.add_argument('--save_reliability_limit', type=int, default=16,
                   help='Maximum number of reliability visualization cases to save per evaluation call.')
parse.add_argument('--benchmark_steps', type=int, default=0,
                   help='If > 0, run a short throughput benchmark and exit after the measured steps.')
parse.add_argument('--benchmark_warmup_steps', type=int, default=2,
                   help='Warmup batches excluded from benchmark timing.')
parse.add_argument('--signal_steps', type=int, default=0,
                   help='If > 0, run a short signal-profiling warmup and exit after the measured steps.')
parse.add_argument('--dataloader_start_method', type=str, default='auto',
                   choices=['auto', 'fork', 'spawn', 'forkserver'],
                   help='Multiprocessing start method for dataloader workers. "auto" keeps the platform default on Linux and uses spawn elsewhere.')
parse.add_argument('--signal_warmup_steps', type=int, default=10,
                   help='Warmup batches excluded from signal-profiling statistics.')
parse.add_argument('--seed', type=int, default=3407,
                   help='Random seed used for model initialization, sampling, and dataloader worker seeding.')
parse.add_argument('--deterministic', action='store_true',
                   help='Enable deterministic PyTorch algorithms for maximum reproducibility at the cost of speed.')
parse.add_argument('--load_ckpt', type=str, default=None)
parse.add_argument('--test_ckpt_role', type=str, default='student', choices=['student', 'teacher'],
                   help='Checkpoint role loaded during final test. Defaults to student so test matches the reported validation model unless overridden.')
parse.add_argument('--model', type=str, default='MyModel')
parse.add_argument('--expID', type=int, default=0) 
parse.add_argument('--ckpt_name', type=str, default='default')
parse.add_argument('--eval_tta', action='store_true',
                   help='Enable test-time augmentation (4x forward) during validation/test. '
                        'BUSI/TN3K validation also defaults to on when omitted.')
parse.add_argument('--no_eval_tta', dest='eval_tta', action='store_false',
                   help='Disable test-time augmentation during validation/test, overriding dataset-aware defaults.')
parse.add_argument('--no_scd', action='store_true', help='Disable structure-aware discriminator')
parse.add_argument('--no_sor', action='store_true', help='Disable dropout decoder branch')
parse.add_argument('--no_ca', action='store_true', help='Disable CoordAttention in encoder')
parse.add_argument('--no_psfh_wsdice', dest='no_psfh_wsdice', action='store_true', default=True,
                   help='Disable the legacy PSFHS-specific per-class weighted soft Dice on PS/FH slots. '
                        'This is the default for the merged-foreground diagnostic/release path.')
parse.add_argument('--use_psfh_wsdice', dest='no_psfh_wsdice', action='store_false',
                   help='Opt in to the legacy PSFHS per-class WSDice path for reproduction/diagnostic ablations.')
parse.add_argument('--aux_supervision_weight', type=float, default=0.5,
                   help='Supervision weight for the auxiliary SOR branch on labeled data.')
parse.add_argument('--boundary_weight', type=float, default=0.2,
                   help='Weight of explicit boundary supervision on labeled data.')
parse.add_argument('--pseudo_main_weight', type=float, default=1.0,
                   help='Weight of rectified pseudo-label supervision on the final inference head.')
parse.add_argument('--pseudo_aux_weight', type=float, default=0.3,
                   help='Weight of rectified pseudo-label supervision on the auxiliary branch.')
parse.add_argument('--sor_weight', type=float, default=0.05,
                   help='Deprecated compatibility flag. SOR now supplies an auxiliary view for reliability, not a separate loss.')
parse.add_argument('--edge_weight', type=float, default=0.05,
                   help='Deprecated compatibility flag. Boundary evidence is fused into pseudo-label reliability.')
parse.add_argument('--adv_weight', type=float, default=0.05,
                   help='Weight of the DSR (Wasserstein critic) shape-prior loss on the unlabeled branch.')
parse.add_argument('--fm_weight', type=float, default=0.5,
                   help='Deprecated compatibility flag. Discriminator feature matching is no longer used.')
parse.add_argument('--teacher_consistency_weight', type=float, default=0.5,
                   help='Deprecated compatibility flag. EMA teacher now provides the pseudo target directly.')
parse.add_argument('--cross_weight', type=float, default=0.3,
                   help='Coefficient for main↔aux cross-teaching (luo22b §3.2). '
                        'Effective weight is λ(t) · cross_weight on each step.')
parse.add_argument('--alpha_lam', type=float, default=0.1,
                   help='Peak value α of the Gaussian rampup λ(t) = α·exp(-β(1-t/T_max)²). '
                        'Matches BiPCC IV-B (α=0.1, β=5).')
parse.add_argument('--beta_lam', type=float, default=5.0,
                   help='Sharpness β of the Gaussian rampup. Higher = later, sharper rise.')
parse.add_argument('--consistency_rampup', type=int, default=40,
                   help='Legacy sigmoid ramp length; only used by scd_rampup now.')
parse.add_argument('--teacher_rampup', type=int, default=None,
                   help='Deprecated; replaced by the single Gaussian λ(t). Kept for argv compatibility.')
parse.add_argument('--structure_rampup', type=int, default=None,
                   help='Deprecated; replaced by the single Gaussian λ(t). Kept for argv compatibility.')
parse.add_argument('--scd_rampup', type=int, default=None,
                   help='Sigmoid ramp length used after SCD starts. Defaults to consistency_rampup.')
parse.add_argument('--scd_start_epoch', type=int, default=0,
                   help='Optional delayed start for SCD optimization. Defaults to 0 for the simpler prior-aware schedule.')
parse.add_argument('--scd_update_interval', type=int, default=1,
                   help='Update interval for discriminator optimization steps. WGAN-GP convention is D:G ≥ 1; '
                        'set to 1 (default) so the critic keeps up with the generator.')
parse.add_argument('--calibration_weight', type=float, default=0.1,
                   help='Weight of the unsupervised boundary-head consistency term inside the reliable pseudo-label objective.')
parse.add_argument('--entropy_tau', type=float, default=0.5,
                   help='Temperature for entropy-based pseudo-label reliability weight w=exp(-H(p)/tau). '
                        'Lower = stricter (only very confident pixels survive); higher = looser.')
parse.add_argument('--reliability_floor', type=float, default=0.0,
                   help='Lower clamp applied to the pseudo-label reliability map. Defaults to 0 for stricter pseudo-label gating.')
parse.add_argument('--adapter_lr', type=float, default=None,
                   help='Learning rate for the SCD feature adapter. Defaults to the segmentation learning rate.')
parse.add_argument('--scd_lr', type=float, default=1e-4,
                   help='Learning rate for the structure discriminator.')
parse.add_argument('--sor_erase', type=float, default=0.2,
                   help='Erasing ratio used by the SOR guided cutout module.')


"-------------------optimizer option-----------------------"
parse.add_argument('--optim', type=str, default='adam',
                   choices=['adam', 'sgd'],
                   help='Optimizer for the segmentation network. adam (default) or sgd. '
                        'BiPCC/Shape-Prior use SGD; switching to sgd reproduces their training dynamic.')
parse.add_argument('--momentum', type=float, default=0.9,
                   help='SGD momentum (only used when --optim sgd).')
parse.add_argument('--lr', type=float, default=DEFAULT_LR)
parse.add_argument('--power',type=float, default=0.9)
parse.add_argument('--betas', default=(0.9, 0.999))
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--eps', type=float, default=1e-8)
parse.add_argument('--mt', type=float, default=0.999)
parse.add_argument('--nclasses', type=int, default=2,
                   help='Number of shared foreground slots. Two slots keep one model across binary datasets and PSFHS PS/FH.')
parse.add_argument('--band', type=int, default=3)
parse.add_argument('--wsdice_w_ps', type=float, default=0.10,
                   help='Negative-area weight for the PS slot in per-class WSDice '
                        '(Improved Dice, IEEE Access 2020). Only used with --use_psfh_wsdice.')
parse.add_argument('--wsdice_w_fh', type=float, default=0.02,
                   help='Negative-area weight for the FH slot in per-class WSDice. '
                        'Only used with --use_psfh_wsdice.')
parse.add_argument('--adaptive_wsdice', action='store_true',
                   help='Enable gradient-balanced adaptive WSDice w_neg for legacy diagnostics. '
                        'Per-class w_neg is updated each batch via EMA toward α · A_k/N_k, '
                        'where A_k is the GT area fraction. Requires --use_psfh_wsdice. '
                        'Default off; when off, uses fixed '
                        '--wsdice_w_ps / --wsdice_w_fh.')
parse.add_argument('--adaptive_wsdice_alpha', type=float, default=1.0,
                   help='Theorem scale factor α in w_k = α·A_k/N_k. Default 1.0 (theoretical balance).')
parse.add_argument('--adaptive_wsdice_ema', type=float, default=0.99,
                   help='EMA momentum for adaptive w_neg updates; lower = faster adaptation.')
parse.add_argument('--wsdice_mode', type=str, default='auto',
                   choices=['auto', 'fixed', 'adaptive', 'inverse_freq', 'effective_number'],
                   help='Per-class w_neg weighting scheme for legacy WSDice; only active with --use_psfh_wsdice. '
                        '"auto" (default): use --adaptive_wsdice flag for back-compat. '
                        '"fixed": constant --wsdice_w_ps/--wsdice_w_fh. '
                        '"adaptive": gradient-balanced Theorem 1. '
                        '"inverse_freq": w ∝ 1/A_k baseline. '
                        '"effective_number": Cui CVPR\'19 baseline.')
parse.add_argument('--wsdice_eff_beta', type=float, default=0.9999,
                   help='β for effective_number weighting (Cui CVPR\'19). '
                        'Higher β = stronger rare-class boost.')
parse.add_argument('--use_carc', action='store_true',
                   help='Enable Class-Aware Reliability Coupling (CARC, paper §3.X): '
                        'multiplicative coupling of shape/pixel/class reliability. '
                        'Default off (additive reliability).')
parse.add_argument('--bwc_clamp', type=float, default=0.0,
                   help='Bounded Wasserstein Critic (BWC, paper §3.X Proposition 2): '
                        'soft tanh clamp on critic outputs to enforce |D̃(x)| ≤ c. '
                        'Default 0 = disabled (use raw WGAN-GP). Set to 3.0 to enable '
                        '(empirically prevents critic output drift on multi-class data).')
parse.add_argument('--use_strong_aug', action='store_true',
                   help='Enable FixMatch-style strong augmentation on the unlabeled student '
                        'path. Teacher path uses the weakly-augmented image; student must '
                        'produce a teacher-consistent prediction under strong perturbation. '
                        'Substantially boosts small-class PL robustness (paper §3.X).')
parse.add_argument('--strong_brightness', type=float, default=0.4,
                   help='Strong-augmentation brightness jitter magnitude (default 0.4).')
parse.add_argument('--strong_contrast', type=float, default=0.4,
                   help='Strong-augmentation contrast jitter magnitude (default 0.4).')
parse.add_argument('--strong_saturation', type=float, default=0.2,
                   help='Strong-augmentation saturation jitter magnitude (default 0.2).')
parse.add_argument('--strong_noise', type=float, default=0.1,
                   help='Strong-augmentation Gaussian noise sigma (default 0.1).')
parse.add_argument('--strong_cutout_prob', type=float, default=0.5,
                   help='Probability of applying random cutout per sample (default 0.5).')
parse.add_argument('--strong_cutout_ratio', type=float, default=0.3,
                   help='Maximum cutout box dimension as fraction of image (default 0.3).')
parse.add_argument('--use_bcp', action='store_true',
                   help='Enable Bidirectional Copy-Paste (BCP, paper §3.X). Injects '
                        'labeled PS regions into unlabeled images so PS receives '
                        'high-confidence supervision regardless of EMA-teacher '
                        'uncertainty. PSFH-specific; ignored on single-class datasets.')
parse.add_argument('--bcp_prob', type=float, default=0.5,
                   help='Per-unlabeled-sample probability of BCP PS injection (default 0.5).')
parse.add_argument('--bcp_weight', type=float, default=1.0,
                   help='Loss weight on the BCP-injected PS region (default 1.0).')
parse.add_argument('--ps_copy_paste_prob', type=float, default=0.5,
                   help='Probability of applying PS copy-paste augmentation on labeled PSFH batches.')
parse.set_defaults(tf32=DEFAULT_VISIBLE_GPU_COUNT > 0)


def parse_gpu_ids(raw_value):
    value = str(raw_value).strip()
    if value == '':
        return []
    return [int(item.strip()) for item in value.split(',') if item.strip() != '']


def sanitize_gpu_ids(gpu_ids, visible_gpu_count):
    if visible_gpu_count <= 0:
        return []

    sanitized = []
    for gpu_id in gpu_ids:
        if 0 <= gpu_id < visible_gpu_count:
            sanitized.append(gpu_id)
        else:
            print(f"[WARN] Ignoring GPU id {gpu_id}; only {visible_gpu_count} visible CUDA device(s).")
    return sanitized


def apply_dataset_eval_defaults(parsed_args, raw_argv):
    dataset_defaults = {
        'BUSI': {
            'eval_tta': True,
            'eval_threshold': 0.48,
            'eval_min_area': 100,
        },
        'TN3K': {
            'eval_tta': True,
            'eval_threshold': 0.50,
            'eval_min_area': 150,
        },
    }
    defaults = dataset_defaults.get(parsed_args.dataset, {})
    if '--eval_tta' not in raw_argv and '--no_eval_tta' not in raw_argv:
        parsed_args.eval_tta = defaults.get('eval_tta', bool(parsed_args.eval_tta))
    if '--eval_threshold' not in raw_argv:
        parsed_args.eval_threshold = defaults.get('eval_threshold', parsed_args.eval_threshold)
    if '--eval_min_area' not in raw_argv:
        parsed_args.eval_min_area = defaults.get('eval_min_area', parsed_args.eval_min_area)
    if '--auto_tune_eval_postprocess' not in raw_argv and '--no_auto_tune_eval_postprocess' not in raw_argv:
        parsed_args.auto_tune_eval_postprocess = defaults.get(
            'auto_tune_eval_postprocess',
            bool(parsed_args.auto_tune_eval_postprocess),
        )


def apply_dataset_semi_defaults(parsed_args, raw_argv):
    # Keep semi-supervised defaults dataset-agnostic. Dataset-specific values
    # should be explicit experiment choices, not hidden code-level tricks.
    dataset_defaults = {}
    defaults = dataset_defaults.get(parsed_args.dataset, {})
    if '--reliability_floor' not in raw_argv:
        parsed_args.reliability_floor = defaults.get(
            'reliability_floor',
            parsed_args.reliability_floor,
        )
    if '--scd_start_epoch' not in raw_argv:
        parsed_args.scd_start_epoch = defaults.get(
            'scd_start_epoch',
            parsed_args.scd_start_epoch,
        )


raw_argv = sys.argv[1:]
args = parse.parse_args()
if args.adapter_lr is None:
    args.adapter_lr = args.lr
if args.teacher_rampup is None:
    args.teacher_rampup = args.consistency_rampup
if args.structure_rampup is None:
    args.structure_rampup = args.consistency_rampup
if args.scd_rampup is None:
    args.scd_rampup = args.consistency_rampup
apply_dataset_eval_defaults(args, raw_argv)
apply_dataset_semi_defaults(args, raw_argv)
args.scd_start_epoch = max(int(args.scd_start_epoch), 0)
args.save_reliability_limit = max(int(args.save_reliability_limit), 0)
args.entropy_tau = max(float(args.entropy_tau), 1e-3)
args.alpha_lam = max(float(args.alpha_lam), 0.0)
args.beta_lam = max(float(args.beta_lam), 0.0)
args.cross_weight = max(float(args.cross_weight), 0.0)
args.reliability_floor = min(max(float(args.reliability_floor), 0.0), 1.0)
args.nclasses = max(int(args.nclasses), 1)
args.wsdice_w_ps = max(float(args.wsdice_w_ps), 0.0)
args.wsdice_w_fh = max(float(args.wsdice_w_fh), 0.0)
args.ps_copy_paste_prob = min(max(float(args.ps_copy_paste_prob), 0.0), 1.0)
args.root = normalize_root(args.root)
args.project_root = str(PROJECT_ROOT)
args.semi_root = str(SEMI_ROOT)
args.checkpoint_root = get_checkpoint_root()

import os as _os
_local_rank = int(_os.environ.get('LOCAL_RANK', -1))
args.local_rank = _local_rank
args.ddp = _local_rank >= 0
args.world_size = int(_os.environ.get('WORLD_SIZE', 1))

if args.ddp:
    args.gpu_ids = [_local_rank]
    args.primary_gpu = _local_rank
    args.use_data_parallel = False
    args.batch_size = max(1, args.batch_size // args.world_size)
else:
    args.gpu_ids = sanitize_gpu_ids(parse_gpu_ids(args.GPUs), DEFAULT_VISIBLE_GPU_COUNT)
    args.GPUs = ','.join(str(gpu_id) for gpu_id in args.gpu_ids)
    args.primary_gpu = args.gpu_ids[0] if args.gpu_ids else None
    args.use_data_parallel = len(args.gpu_ids) > 1
