import shutil
import hashlib
import json
from contextlib import nullcontext

import numpy as np
import random
import torch
import os
import sys
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
from data.build_dataset import build_dataset, build_eval_dataset
from models.build_model import (
    build_model,
    configure_cuda_model,
    get_model_state_dict,
    load_model_state_dict,
    unwrap_model,
)
from models.dc_gan import DCGAN_D
from utils.evaluate import evaluate
from opt import args
from utils.loss import (
    BceDiceLoss,
    sigmoid_rampup,
    gaussian_rampup,
    as_probability_map,
    get_structure_boundary,
    reliable_pseudo_label_loss,
    weighted_bce_dice_loss,
    entropy_weight,
    gradient_balanced_w_neg,
    inverse_freq_w_neg,
    effective_number_w_neg,
    carc_reliability,
)
from utils.path_utils import AIRS_ROOT, get_scd_pretrain_path, get_split_file
import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def is_main_process():
    return not getattr(args, 'ddp', False) or args.local_rank == 0


def log_info(message):
    if is_main_process():
        print(message, flush=True)


def scalar_value(value):
    if torch.is_tensor(value):
        return float(value.detach().mean().item())
    return float(value)


def mean_feature_matching_loss(real_features, fake_features):
    losses = []
    for real_feat, fake_feat in zip(real_features, fake_features):
        real_feat_mean = real_feat.float().mean(dim=0, keepdim=True)
        fake_feat_mean = fake_feat.float().mean(dim=0, keepdim=True)
        losses.append(F.l1_loss(fake_feat_mean, real_feat_mean))
    return sum(losses) / len(losses)


def set_optimizer_lr(optimizer, lr):
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group['lr'] = lr


def infer_scd_norm_type(state_dict):
    for key in state_dict.keys():
        if ':batchnorm.' in key:
            return 'batch'
        if ':norm.' in key:
            return 'instance'
    return 'instance'


def translate_scd_checkpoint_state(current_state, checkpoint_state):
    translated_state = {}
    stats = {
        'loaded': 0,
        'skipped_initial': 0,
        'skipped_missing': 0,
        'skipped_shape': 0,
    }
    for key, value in checkpoint_state.items():
        mapped_key = key.replace(':batchnorm.', ':norm.')
        if mapped_key.startswith('main.initial:') and mapped_key.endswith(':conv.weight'):
            stats['skipped_initial'] += 1
            continue
        if mapped_key not in current_state:
            stats['skipped_missing'] += 1
            continue
        if current_state[mapped_key].shape != value.shape:
            stats['skipped_shape'] += 1
            continue
        translated_state[mapped_key] = value
        stats['loaded'] += 1
    return translated_state, stats


def compute_unlabeled_trust(reliability_mean, main_aux_gap, student_teacher_gap):
    # ProPL-style pseudo-label calibration is uncertainty driven, so keep the
    # stronger unlabeled regularizers active only when region confidence and
    # cross-view agreement are both healthy.
    decoder_agreement = torch.exp(-2.0 * main_aux_gap.detach())
    teacher_agreement = torch.exp(-2.0 * student_teacher_gap.detach())
    trust = reliability_mean.detach() * decoder_agreement * teacher_agreement
    return trust.clamp(0.05, 1.0)


def update_running_stats(stats, **kwargs):
    tensor_keys = []
    tensor_vals = []
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            tensor_keys.append(key)
            tensor_vals.append(value.detach().mean())
        else:
            stats[key] = stats.get(key, 0.0) + float(value)
    if tensor_vals:
        floats = torch.stack(tensor_vals).float().cpu().tolist()
        for key, val in zip(tensor_keys, floats):
            stats[key] = stats.get(key, 0.0) + val


def surface_metric_mode(dataset_name=None):
    name = (dataset_name or args.dataset).lower()
    if name == 'hc18':
        return 'overall'
    if name == 'psfh':
        return 'split'
    return 'none'


def get_eval_spacing(dataset_name=None):
    """返回 evaluate/tune 时使用的 fallback (sy, sx) spacing。
    官方口径在 Dataset 里逐样本返回：HC18 用官方 pixel-size CSV，PSFH 用
    MHA ElementSpacing。这里的值只兜底给不携带 spacing 的临时/旧 Dataset。
    其他数据集不使用 surface metrics，spacing 无实质影响。
    """
    name = (dataset_name or args.dataset).lower()
    if name == 'hc18':
        return (0.2869, 0.4250)
    if name == 'psfh':
        return (1.0, 1.0)
    return (0.07, 0.07)


def should_log_batch(batch_id, total_batch):
    current_batch = batch_id + 1
    if args.log_first_batches > 0 and current_batch <= args.log_first_batches:
        return True
    if args.log_interval > 0 and current_batch % args.log_interval == 0:
        return True
    return current_batch == total_batch


def create_progress(total_batch):
    disable_tqdm = args.disable_tqdm or not sys.stdout.isatty() or not is_main_process()
    return tqdm(range(total_batch), disable=disable_tqdm, dynamic_ncols=False, leave=False)


def resolve_dataloader_start_method():
    if args.dataloader_start_method == 'auto':
        return None if os.name == 'posix' else 'spawn'
    return args.dataloader_start_method


def build_dataloader(dataset, batch_size, shuffle, num_workers, use_distributed_sampler=None):
    if use_distributed_sampler is None:
        use_distributed_sampler = args.ddp
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': bool(args.gpu_ids),
    }
    sampler = None
    if use_distributed_sampler:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        loader_kwargs['sampler'] = sampler
    else:
        loader_kwargs['shuffle'] = shuffle
    if num_workers > 0:
        start_method = resolve_dataloader_start_method()
        if start_method is not None:
            loader_kwargs['multiprocessing_context'] = start_method
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4
    loader = DataLoader(dataset, **loader_kwargs)
    loader.dist_sampler = sampler
    return loader


def resolve_effective_precision():
    requested = args.precision.lower()
    if not args.gpu_ids or not torch.cuda.is_available():
        return 'fp32'

    bf16_supported = hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
    if requested == 'auto':
        # BCE-style probability losses in semi training are noticeably less stable in bf16.
        return 'fp32'
    if requested == 'bf16' and not bf16_supported:
        log_info('[WARN] bf16 requested but not supported on this CUDA device; falling back to fp32.')
        return 'fp32'
    if requested == 'bf16':
        log_info('[WARN] bf16 requested for semi training; probability losses may become unstable late in training.')
    return requested


def autocast_context():
    if getattr(args, 'effective_precision', 'fp32') == 'bf16':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def configure_runtime():
    if args.ddp:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    args.effective_precision = resolve_effective_precision()
    if args.gpu_ids and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')


def set_random_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu_ids:
        torch.cuda.manual_seed_all(args.seed)

    if args.deterministic:
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log_info(
        "[INFO] Random seed: {} deterministic={}".format(
            args.seed,
            'on' if args.deterministic else 'off',
        )
    )


def log_runtime_configuration():
    loader_start_method = resolve_dataloader_start_method() or 'platform-default'
    log_info(
        '[INFO] Runtime config: precision={} requested_precision={} tf32={} batch_size={} num_workers={} '
        'loader_start={} epochs={} lr={:.6e} nclasses={} val_interval={} eval_tta={} save_best_tta={} '
        'eval_threshold={:.3f} eval_min_area={} eval_postprocess_mode={} auto_tune_eval_postprocess={} '
        'save_reliability_maps={} '
        'teacher_rampup={} structure_rampup={} scd_start_epoch={} scd_rampup={} '
        'entropy_tau={:.2f} rel_floor={:.2f} '
        'psfh_wsdice={} wsdice_w_neg=(ps={:.3f},fh={:.3f})'.format(
            getattr(args, 'effective_precision', 'fp32'),
            args.precision,
            'on' if args.tf32 else 'off',
            args.batch_size,
            args.num_workers,
            loader_start_method,
            args.nEpoch,
            args.lr,
            args.nclasses,
            args.val_interval,
            'on' if args.eval_tta else 'off',
            'on' if args.save_best_tta else 'off',
            args.eval_threshold,
            args.eval_min_area,
            args.eval_postprocess_mode,
            'on' if args.auto_tune_eval_postprocess else 'off',
            'on' if args.save_reliability_maps else 'off',
            args.teacher_rampup,
            args.structure_rampup,
            args.scd_start_epoch,
            args.scd_rampup,
            args.entropy_tau,
            args.reliability_floor,
            'off' if args.no_psfh_wsdice else 'on',
            args.wsdice_w_ps,
            args.wsdice_w_fh,
        )
    )


def eval_postprocess_kwargs():
    return {
        'eval_threshold': args.eval_threshold,
        'eval_min_area': args.eval_min_area,
        'eval_postprocess_mode': args.eval_postprocess_mode,
    }


def eval_reliability_kwargs():
    return {
        'save_reliability': bool(args.save_reliability_maps),
        'save_reliability_limit': args.save_reliability_limit,
    }


def eval_artifact_kwargs(split_name):
    return {
        **eval_reliability_kwargs(),
        'case_metrics_path': os.path.join(get_result_dir(), f'{split_name}_case_metrics.csv'),
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _split_file_summary(path):
    with open(path, 'r', encoding='utf-8') as handle:
        entries = [line.strip() for line in handle if line.strip()]
    return {
        'path': os.path.abspath(path),
        'count': len(entries),
        'sha256': _file_sha256(path),
    }


def collect_split_manifest(dataset_name):
    split_root = get_split_file(dataset_name)
    manifest = {}
    if not os.path.isdir(split_root):
        return manifest
    for dirpath, _, filenames in os.walk(split_root):
        for filename in filenames:
            if not filename.endswith('.txt'):
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, split_root).replace(os.sep, '/')
            manifest[rel] = _split_file_summary(path)
    return dict(sorted(manifest.items()))


def _json_safe_args():
    safe = {}
    for key, value in sorted(vars(args).items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = list(value)
        else:
            safe[key] = str(value)
    return safe


def write_eval_manifest(split_name, dataloader, postprocess_kwargs):
    result_dir = get_result_dir()
    os.makedirs(result_dir, exist_ok=True)
    split_file = resolved_eval_split_file(getattr(dataloader, 'dataset', None))
    manifest = {
        'dataset': args.dataset,
        'expID': args.expID,
        'ckpt_name': args.ckpt_name,
        'split_name': split_name,
        'resolved_split': describe_eval_split(getattr(dataloader, 'dataset', None)),
        'resolved_split_file': split_file,
        'resolved_split_sha256': _file_sha256(split_file) if split_file and os.path.exists(split_file) else None,
        'dataset_summary': dataset_split_summary(getattr(dataloader, 'dataset', None)),
        'git_commit': _git_commit(),
        'postprocess': dict(postprocess_kwargs),
        'args': _json_safe_args(),
        'split_manifest': collect_split_manifest(args.dataset),
    }
    with open(os.path.join(result_dir, f'{split_name}_eval_manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _git_commit():
    head_path = os.path.join(str(AIRS_ROOT.parent), '.git', 'HEAD')
    try:
        with open(head_path, 'r', encoding='utf-8') as handle:
            head = handle.read().strip()
        if head.startswith('ref: '):
            ref_path = os.path.join(str(AIRS_ROOT.parent), '.git', head.split(' ', 1)[1])
            with open(ref_path, 'r', encoding='utf-8') as handle:
                return handle.read().strip()
        return head
    except OSError:
        return None


def _unique_sorted_thresholds(values):
    cleaned = []
    seen = set()
    for value in values:
        clipped = round(float(np.clip(float(value), 0.0, 1.0)), 4)
        if clipped in seen:
            continue
        seen.add(clipped)
        cleaned.append(clipped)
    return sorted(cleaned)


def _unique_sorted_ints(values):
    cleaned = []
    seen = set()
    for value in values:
        normalized = max(int(value), 0)
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return sorted(cleaned)


def eval_postprocess_search_space():
    dataset = args.dataset.upper()
    if dataset == 'BUSI':
        thresholds = [0.44, 0.46, 0.48, 0.50, 0.52]
        min_areas = [50, 100, 150]
    elif dataset == 'TN3K':
        thresholds = [0.46, 0.48, 0.50, 0.52, 0.54]
        min_areas = [100, 150, 200]
    else:
        thresholds = [
            args.eval_threshold - 0.04,
            args.eval_threshold - 0.02,
            args.eval_threshold,
            args.eval_threshold + 0.02,
            args.eval_threshold + 0.04,
        ]
        min_areas = [
            max(args.eval_min_area // 2, 0),
            args.eval_min_area,
            max(args.eval_min_area * 2, 0),
        ]

    thresholds.append(args.eval_threshold)
    min_areas.append(args.eval_min_area)
    return _unique_sorted_thresholds(thresholds), _unique_sorted_ints(min_areas)


def refined_eval_postprocess_search_space(best_threshold, best_min_area):
    dataset = args.dataset.upper()
    if dataset == 'BUSI':
        threshold_offsets = [-0.02, -0.01, 0.0, 0.01, 0.02]
        area_offsets = [-50, -25, 0, 25, 50]
    elif dataset == 'TN3K':
        threshold_offsets = [-0.02, -0.01, 0.0, 0.01, 0.02]
        area_offsets = [-50, -25, 0, 25, 50]
    else:
        threshold_offsets = [-0.02, -0.01, 0.0, 0.01, 0.02]
        area_offsets = [-max(best_min_area // 4, 10), 0, max(best_min_area // 4, 10)]

    thresholds = _unique_sorted_thresholds(best_threshold + offset for offset in threshold_offsets)
    min_areas = _unique_sorted_ints(best_min_area + offset for offset in area_offsets)
    return thresholds, min_areas


def tune_eval_postprocess(model, dataloader, total_batch, spacing=(1.0, 1.0), split_name='valid'):
    base_kwargs = eval_postprocess_kwargs()
    if not args.auto_tune_eval_postprocess:
        return base_kwargs

    thresholds, min_areas = eval_postprocess_search_space()
    if len(thresholds) * len(min_areas) <= 1:
        return base_kwargs

    log_info(
        '[PostprocessTune] split={} dataset={} tta={} thresholds={} min_areas={}'.format(
            split_name,
            args.dataset,
            'on' if args.eval_tta else 'off',
            ','.join(f'{value:.2f}' for value in thresholds),
            ','.join(str(value) for value in min_areas),
        )
    )

    best_result = None
    base_result = None
    evaluated = {}

    def evaluate_candidate(threshold, min_area):
        key = (round(float(threshold), 4), int(min_area))
        if key in evaluated:
            return evaluated[key]

        recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice, *_, table_metrics = evaluate(
                model,
                dataloader,
                total_batch,
                spacing=spacing,
                tta=args.eval_tta,
                eval_threshold=key[0],
                eval_min_area=key[1],
                eval_postprocess_mode=args.eval_postprocess_mode,
                show_progress=False,
            )
        candidate = {
            'eval_threshold': key[0],
            'eval_min_area': key[1],
            'recall': recall,
            'specificity': specificity,
            'precision': precision,
            'F1': F1,
            'F2': F2,
            'ACC_overall': ACC_overall,
            'IoU_poly': IoU_poly,
            'IoU_bg': IoU_bg,
            'IoU_mean': IoU_mean,
            'dice': dice,
            'table_metrics': table_metrics,
        }
        evaluated[key] = candidate
        return candidate

    def pick_better(current_best, candidate):
        if current_best is None:
            return candidate
        if candidate['dice'] > current_best['dice'] + 1e-8:
            return candidate
        if abs(candidate['dice'] - current_best['dice']) <= 1e-8 and candidate['IoU_poly'] > current_best['IoU_poly'] + 1e-8:
            return candidate
        return current_best

    for threshold in thresholds:
        for min_area in min_areas:
            candidate = evaluate_candidate(threshold, min_area)
            if abs(candidate['eval_threshold'] - base_kwargs['eval_threshold']) <= 1e-8 and candidate['eval_min_area'] == base_kwargs['eval_min_area']:
                base_result = candidate
            best_result = pick_better(best_result, candidate)

    if base_result is not None:
        log_info(
            '[PostprocessTune] base threshold={:.2f} min_area={} dice={:.4f} IoU_poly={:.4f}'.format(
                base_result['eval_threshold'],
                base_result['eval_min_area'],
                base_result['dice'],
                base_result['IoU_poly'],
            )
        )
    if best_result is not None:
        delta_dice = 0.0 if base_result is None else best_result['dice'] - base_result['dice']
        delta_iou = 0.0 if base_result is None else best_result['IoU_poly'] - base_result['IoU_poly']
        log_info(
            '[PostprocessTune] best threshold={:.2f} min_area={} dice={:.4f} IoU_poly={:.4f} '
            'delta_dice={:+.4f} delta_iou={:+.4f}'.format(
                best_result['eval_threshold'],
                best_result['eval_min_area'],
                best_result['dice'],
                best_result['IoU_poly'],
                delta_dice,
                delta_iou,
            )
        )

        refined_thresholds, refined_min_areas = refined_eval_postprocess_search_space(
            best_result['eval_threshold'],
            best_result['eval_min_area'],
        )
        log_info(
            '[PostprocessTune] refine thresholds={} min_areas={}'.format(
                ','.join(f'{value:.2f}' for value in refined_thresholds),
                ','.join(str(value) for value in refined_min_areas),
            )
        )
        refined_best = best_result
        for threshold in refined_thresholds:
            for min_area in refined_min_areas:
                refined_best = pick_better(refined_best, evaluate_candidate(threshold, min_area))

        if refined_best is not best_result:
            log_info(
                '[PostprocessTune] refined best threshold={:.2f} min_area={} dice={:.4f} IoU_poly={:.4f} '
                'delta_vs_coarse={:+.4f}'.format(
                    refined_best['eval_threshold'],
                    refined_best['eval_min_area'],
                    refined_best['dice'],
                    refined_best['IoU_poly'],
                    refined_best['dice'] - best_result['dice'],
                )
            )
        best_result = refined_best
        return {
            'eval_threshold': best_result['eval_threshold'],
            'eval_min_area': best_result['eval_min_area'],
            'eval_postprocess_mode': args.eval_postprocess_mode,
        }
    return base_kwargs


def maybe_finish_benchmark(stage, measured_steps, measured_samples, start_time):
    if args.benchmark_steps <= 0 or measured_steps < args.benchmark_steps or start_time is None:
        return
    if args.gpu_ids:
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    log_info(
        "[AUTOTUNE_RESULT] stage={} batch_size={} workers={} steps={} samples={} elapsed_sec={:.6f} samples_per_sec={:.6f} batches_per_sec={:.6f}".format(
            stage,
            args.batch_size,
            args.num_workers,
            measured_steps,
            measured_samples,
            elapsed,
            measured_samples / elapsed,
            measured_steps / elapsed,
        )
    )
    raise SystemExit(0)


def emit_training_result(stage, best_f1, best_f1_epoch, best_dice, best_dice_epoch):
    log_info(
        "[TRAIN_RESULT] stage={} dataset={} ckpt_name={} batch_size={} workers={} epochs={} lr={:.6e} best_f1={:.6f} best_f1_epoch={} best_dice={:.6f} best_dice_epoch={}".format(
            stage,
            args.dataset,
            args.ckpt_name,
            args.batch_size,
            args.num_workers,
            args.nEpoch,
            args.lr,
            float(best_f1),
            int(best_f1_epoch),
            float(best_dice),
            int(best_dice_epoch),
        )
    )


def should_run_validation(epoch):
    interval = max(1, args.val_interval)
    return (epoch + 1) % interval == 0 or (epoch + 1) == args.nEpoch


def signal_mode_active():
    return args.signal_steps > 0


def init_signal_state():
    return {
        'seen_steps': 0,
        'measured_steps': 0,
        'measured_samples': 0,
        'sum_loss': 0.0,
        'sum_labeled': 0.0,
        'sum_unlabeled': 0.0,
        'stats': {},
        'first_total_loss': None,
        'last_total_loss': None,
        'first_labeled_loss': None,
        'last_labeled_loss': None,
    }


def update_signal_state(signal_state, batch_size, loss, loss_l, loss_u, **kwargs):
    signal_state['seen_steps'] += 1
    if signal_state['seen_steps'] <= args.signal_warmup_steps:
        return False

    loss_floats = torch.stack([
        loss.detach().mean(),
        loss_l.detach().mean(),
        loss_u.detach().mean(),
    ]).float().cpu().tolist()
    total_loss, labeled_loss, unlabeled_loss = loss_floats

    if signal_state['first_total_loss'] is None:
        signal_state['first_total_loss'] = total_loss
    if signal_state['first_labeled_loss'] is None:
        signal_state['first_labeled_loss'] = labeled_loss

    signal_state['last_total_loss'] = total_loss
    signal_state['last_labeled_loss'] = labeled_loss
    signal_state['measured_steps'] += 1
    signal_state['measured_samples'] += int(batch_size)
    signal_state['sum_loss'] += total_loss
    signal_state['sum_labeled'] += labeled_loss
    signal_state['sum_unlabeled'] += unlabeled_loss
    update_running_stats(signal_state['stats'], **kwargs)
    return signal_state['measured_steps'] >= args.signal_steps


def safe_metric_value(value, default=0.0):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def bounded_consistency_score(gap, scale=4.0):
    return max(0.0, 1.0 - scale * abs(gap))


def summarize_signal_state(signal_state):
    measured_steps = max(signal_state['measured_steps'], 1)
    stats = signal_state['stats']
    summary = {
        'avg_loss': signal_state['sum_loss'] / measured_steps,
        'avg_labeled': signal_state['sum_labeled'] / measured_steps,
        'avg_unlabeled': signal_state['sum_unlabeled'] / measured_steps,
        'loss_drop': safe_metric_value(signal_state['first_total_loss']) - safe_metric_value(signal_state['last_total_loss']),
        'labeled_loss_drop': safe_metric_value(signal_state['first_labeled_loss']) - safe_metric_value(signal_state['last_labeled_loss']),
        'measured_steps': signal_state['measured_steps'],
        'measured_samples': signal_state['measured_samples'],
    }
    for key in (
        'loss_l_main',
        'loss_l_aux',
        'loss_l_boundary',
        'loss_u_main',
        'loss_u_aux',
        'loss_calibration',
        'loss_edge',
        'loss_sor',
        'loss_teacher',
        'loss_adv',
        'teacher_weight',
        'unlabeled_trust',
        'scd_ramp',
        'reliability_mean',
        'reliability_high',
        'pseudo_fg',
        'pseudo_pos',
        'student_fg',
        'teacher_fg',
        'main_aux_gap',
        'student_teacher_gap',
        'scd_real_prob',
        'scd_fake_prob',
        'scd_d_loss',
        'psfh_ps_area_gt',
        'psfh_fh_area_gt',
        'psfh_ps_area_student',
        'psfh_fh_area_student',
        'psfh_ps_area_teacher',
        'psfh_fh_area_teacher',
        'psfh_ps_dsc_sup',
        'psfh_fh_dsc_sup',
        'psfh_critic_ps_ratio',
        'gp_grad_norm',
        'gp_value',
    ):
        summary[key] = safe_metric_value(stats.get(key, 0.0) / measured_steps)
    summary['pseudo_student_gap'] = abs(summary['pseudo_fg'] - summary['student_fg'])
    summary['pseudo_teacher_gap'] = abs(summary['pseudo_fg'] - summary['teacher_fg'])
    summary['scd_margin'] = summary['scd_real_prob'] - summary['scd_fake_prob']
    return summary


def compute_signal_score(signal_summary, val_f1=0.0, val_dice=0.0):
    val_f1 = safe_metric_value(val_f1)
    val_dice = safe_metric_value(val_dice)
    score = 0.0
    score += 2.0 * val_f1
    score += 1.5 * val_dice
    score += 0.15 * signal_summary['unlabeled_trust']
    score += 0.35 * signal_summary['reliability_mean']
    score += 0.20 * signal_summary['reliability_high']
    score += 0.25 * max(0.0, signal_summary['loss_drop'])
    score += 0.20 * max(0.0, signal_summary['labeled_loss_drop'])
    score += 0.10 * max(0.0, signal_summary['scd_margin'])
    score += 0.12 * bounded_consistency_score(signal_summary['main_aux_gap'])
    score += 0.12 * bounded_consistency_score(signal_summary['student_teacher_gap'])
    score += 0.08 * bounded_consistency_score(signal_summary['pseudo_student_gap'])
    score += 0.08 * bounded_consistency_score(signal_summary['pseudo_teacher_gap'])
    return score


def emit_signal_result(epoch, signal_summary, val_f1=0.0, val_dice=0.0):
    signal_score = compute_signal_score(signal_summary, val_f1=val_f1, val_dice=val_dice)
    log_info(
        "[SIGNAL_RESULT] stage=semi_signal dataset={} ckpt_name={} epoch={} batch_size={} workers={} lr={:.6e} "
        "measured_steps={} measured_samples={} avg_loss={:.4f} avg_labeled={:.4f} avg_unlabeled={:.4f} "
        "loss_drop={:.4f} labeled_loss_drop={:.4f} trust={:.4f} scd_ramp={:.4f} rel_mean={:.4f} rel_hi={:.4f} pseudo_fg={:.4f} pseudo_pos={:.4f} "
        "student_fg={:.4f} teacher_fg={:.4f} main_aux_gap={:.4f} st_gap={:.4f} pseudo_student_gap={:.4f} "
        "pseudo_teacher_gap={:.4f} scd_real={:.4f} scd_fake={:.4f} scd_margin={:.4f} scd_d={:.4f} "
        "val_f1={:.4f} val_dice={:.4f} signal_score={:.6f}".format(
            args.dataset,
            args.ckpt_name,
            epoch,
            args.batch_size,
            args.num_workers,
            args.lr,
            int(signal_summary['measured_steps']),
            int(signal_summary['measured_samples']),
            signal_summary['avg_loss'],
            signal_summary['avg_labeled'],
            signal_summary['avg_unlabeled'],
            signal_summary['loss_drop'],
            signal_summary['labeled_loss_drop'],
            signal_summary['unlabeled_trust'],
            signal_summary['scd_ramp'],
            signal_summary['reliability_mean'],
            signal_summary['reliability_high'],
            signal_summary['pseudo_fg'],
            signal_summary['pseudo_pos'],
            signal_summary['student_fg'],
            signal_summary['teacher_fg'],
            signal_summary['main_aux_gap'],
            signal_summary['student_teacher_gap'],
            signal_summary['pseudo_student_gap'],
            signal_summary['pseudo_teacher_gap'],
            signal_summary['scd_real_prob'],
            signal_summary['scd_fake_prob'],
            signal_summary['scd_margin'],
            signal_summary['scd_d_loss'],
            safe_metric_value(val_f1),
            safe_metric_value(val_dice),
            signal_score,
        )
    )


def log_full_train_batch(epoch, total_epoch, batch_id, total_batch, batch_time, lr, loss, loss_main, loss_aux, loss_boundary):
    log_info(
        "[Train][Epoch {}/{}][Batch {}/{}] loss={:.4f} main={:.4f} aux={:.4f} boundary={:.4f} lr={:.6e} dt={:.2f}s".format(
            epoch + 1,
            total_epoch,
            batch_id + 1,
            total_batch,
            scalar_value(loss),
            scalar_value(loss_main),
            scalar_value(loss_aux),
            scalar_value(loss_boundary),
            lr,
            batch_time,
        )
    )


def log_semi_train_batch(
    epoch,
    total_epoch,
    batch_id,
    total_batch,
    batch_time,
    lr,
    loss,
    loss_l,
    loss_u,
    loss_u_main,
    loss_u_aux,
    loss_calibration,
    loss_adv,
):
    log_info(
        "[TrainSemi][Epoch {}/{}][Batch {}/{}] loss={:.4f} labeled={:.4f} unlabeled={:.4f} "
        "u_main={:.4f} u_aux={:.4f} u_boundary={:.4f} shape={:.4f} "
        "lr={:.6e} dt={:.2f}s".format(
            epoch + 1,
            total_epoch,
            batch_id + 1,
            total_batch,
            scalar_value(loss),
            scalar_value(loss_l),
            scalar_value(loss_u),
            scalar_value(loss_u_main),
            scalar_value(loss_u_aux),
            scalar_value(loss_calibration),
            scalar_value(loss_adv),
            lr,
            batch_time,
        )
    )


def log_semi_diagnostics_batch(
    epoch,
    total_epoch,
    batch_id,
    total_batch,
    loss_l_main,
    loss_l_aux,
    loss_l_boundary,
    teacher_weight,
    unlabeled_trust,
    scd_ramp,
    reliability_mean,
    reliability_high,
    pseudo_fg,
    pseudo_pos,
    student_fg,
    teacher_fg,
    main_aux_gap,
    student_teacher_gap,
    scd_real_prob,
    scd_fake_prob,
    scd_d_loss,
):
    log_info(
        "[TrainSemiDiag][Epoch {}/{}][Batch {}/{}] l_main={:.4f} l_aux={:.4f} l_boundary={:.4f} "
        "teacher_w={:.4f} trust={:.4f} scd_ramp={:.4f} rel_mean={:.4f} rel_hi={:.4f} pseudo_fg={:.4f} pseudo_pos={:.4f} "
        "student_fg={:.4f} teacher_fg={:.4f} main_aux_gap={:.4f} st_gap={:.4f} "
        "scd_real={:.4f} scd_fake={:.4f} scd_d={:.4f}".format(
            epoch + 1,
            total_epoch,
            batch_id + 1,
            total_batch,
            scalar_value(loss_l_main),
            scalar_value(loss_l_aux),
            scalar_value(loss_l_boundary),
            scalar_value(teacher_weight),
            scalar_value(unlabeled_trust),
            scalar_value(scd_ramp),
            scalar_value(reliability_mean),
            scalar_value(reliability_high),
            scalar_value(pseudo_fg),
            scalar_value(pseudo_pos),
            scalar_value(student_fg),
            scalar_value(teacher_fg),
            scalar_value(main_aux_gap),
            scalar_value(student_teacher_gap),
            scalar_value(scd_real_prob),
            scalar_value(scd_fake_prob),
            scalar_value(scd_d_loss),
        )
    )


def log_validation_details(table_metrics):
    metric_mode = surface_metric_mode()
    if metric_mode == 'none':
        return
    log_info(
        "[ValidDetailAll] DSC_all={:.4f} Jacc_all={:.4f} HD95_all={:.4f} ASD_all={:.4f}".format(
            float(table_metrics.get('DSC', 0.0)),
            float(table_metrics.get('Jaccard', 0.0)),
            float(table_metrics.get('HD95', 0.0)),
            float(table_metrics.get('ASD', 0.0)),
        )
    )
    if metric_mode == 'split':
        log_info(
            "[ValidDetailPS] DSC_PS={:.4f} Jacc_PS={:.4f} HD95_PS={:.4f} ASD_PS={:.4f}".format(
                float(table_metrics.get('DSC_PS', 0.0)),
                float(table_metrics.get('Jaccard_PS', 0.0)),
                float(table_metrics.get('HD95_PS', 0.0)),
                float(table_metrics.get('ASD_PS', 0.0)),
            )
        )
        log_info(
            "[ValidDetailFH] DSC_FH={:.4f} Jacc_FH={:.4f} HD95_FH={:.4f} ASD_FH={:.4f}".format(
                float(table_metrics.get('DSC_FH', 0.0)),
                float(table_metrics.get('Jaccard_FH', 0.0)),
                float(table_metrics.get('HD95_FH', 0.0)),
                float(table_metrics.get('ASD_FH', 0.0)),
            )
        )


def get_result_dir():
    return os.environ.get('AIRS_SEMI_RESULT_DIR', './result')


def resolved_eval_split_file(dataset):
    split_file = getattr(dataset, 'resolved_split_file', None)
    if not split_file:
        return None
    return os.path.abspath(split_file)


def describe_eval_split(dataset):
    requested = getattr(dataset, 'requested_split_name', None)
    resolved = getattr(dataset, 'resolved_split_name', None)
    split_file = resolved_eval_split_file(dataset)
    if requested and resolved and requested != resolved:
        return f"{requested} -> {resolved} ({split_file})"
    if requested and split_file:
        return f"{requested} ({split_file})"
    if resolved and split_file:
        return f"{resolved} ({split_file})"
    return split_file or '<unknown>'


def dataset_split_summary(dataset):
    if dataset is None:
        return None
    try:
        length = len(dataset)
    except TypeError:
        length = None
    return {
        'class': dataset.__class__.__name__,
        'length': length,
        'split': describe_eval_split(dataset),
        'raw_line_count': getattr(dataset, 'raw_line_count', None),
        'effective_line_count': getattr(dataset, 'effective_line_count', None),
        'filtered_holdout_count': getattr(dataset, 'filtered_holdout_count', None),
        'filtered_official_test_set_count': getattr(dataset, 'filtered_official_test_set_count', None),
        'source_subset_counts': getattr(dataset, 'source_subset_counts', None),
    }


def _format_subset_counts(counts):
    if not counts:
        return '-'
    return ','.join('{}:{}'.format(key, value) for key, value in sorted(counts.items()))


def log_dataset_splits(stage, **datasets):
    for role, dataset in datasets.items():
        if dataset is None:
            continue
        summary = dataset_split_summary(dataset)
        log_info(
            "[DATA] stage={} role={} class={} n={} split={} raw={} effective={} "
            "filtered_holdout={} filtered_official_test_set={} sources={}".format(
                stage,
                role,
                summary.get('class'),
                summary.get('length'),
                summary.get('split'),
                summary.get('raw_line_count'),
                summary.get('effective_line_count'),
                summary.get('filtered_holdout_count'),
                summary.get('filtered_official_test_set_count'),
                _format_subset_counts(summary.get('source_subset_counts')),
            )
        )


def get_checkpoint_dir(role='student'):
    if role == 'teacher':
        return os.path.join(args.checkpoint_root, args.ckpt_name + "_teacher")
    return os.path.join(args.checkpoint_root, args.ckpt_name)


def resolve_test_checkpoint():
    role = getattr(args, 'test_ckpt_role', 'student')
    candidate_stems = [args.load_ckpt] if args.load_ckpt else ['best_dice', 'best']
    ckpt_dir = get_checkpoint_dir(role)

    for stem in candidate_stems:
        ckpt_path = os.path.join(ckpt_dir, stem + '.pth')
        if os.path.exists(ckpt_path):
            return role, ckpt_path, stem

    if role == 'student':
        teacher_dir = get_checkpoint_dir('teacher')
        teacher_hits = [
            stem + '.pth'
            for stem in candidate_stems
            if os.path.exists(os.path.join(teacher_dir, stem + '.pth'))
        ]
        if teacher_hits:
            log_info(
                "[Test] Note: found teacher checkpoint(s) {} under {}, but final test defaults to student checkpoints. "
                "Use --test_ckpt_role teacher to evaluate the EMA teacher explicitly.".format(
                    ', '.join(teacher_hits),
                    teacher_dir,
                )
            )

    return role, None, candidate_stems


def append_test_summary_markdown(metrics, table_metrics):
    summary_path = os.environ.get('AIRS_TEST_SUMMARY_FILE')
    if not summary_path:
        return

    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    def metric_value(name):
        metric_mode = surface_metric_mode()
        split_metric = name.endswith('_PS') or name.endswith('_FH')
        if metric_mode == 'none':
            return '-'
        if metric_mode != 'split' and split_metric:
            return '-'
        value = table_metrics.get(name)
        if value is None:
            return '-'
        return '{:.4f}'.format(float(value))

    row = (
        "| {dataset} | {expid} | {ckpt_name} | {recall:.4f} | {specificity:.4f} | {precision:.4f} | "
        "{F1:.4f} | {F2:.4f} | {ACC_overall:.4f} | {IoU_poly:.4f} | {IoU_bg:.4f} | {IoU_mean:.4f} | {dice:.4f} | "
        "{DSC} | {Jaccard} | {HD95} | {ASD} | {DSC_PS} | {Jaccard_PS} | {HD95_PS} | {ASD_PS} | "
        "{DSC_FH} | {Jaccard_FH} | {HD95_FH} | {ASD_FH} |\n"
    ).format(
        dataset=args.dataset,
        expid=args.expID,
        ckpt_name=args.ckpt_name,
        recall=metrics['recall'],
        specificity=metrics['specificity'],
        precision=metrics['precision'],
        F1=metrics['F1'],
        F2=metrics['F2'],
        ACC_overall=metrics['ACC_overall'],
        IoU_poly=metrics['IoU_poly'],
        IoU_bg=metrics['IoU_bg'],
        IoU_mean=metrics['IoU_mean'],
        dice=metrics['dice'],
        DSC=metric_value('DSC'),
        Jaccard=metric_value('Jaccard'),
        HD95=metric_value('HD95'),
        ASD=metric_value('ASD'),
        DSC_PS=metric_value('DSC_PS'),
        Jaccard_PS=metric_value('Jaccard_PS'),
        HD95_PS=metric_value('HD95_PS'),
        ASD_PS=metric_value('ASD_PS'),
        DSC_FH=metric_value('DSC_FH'),
        Jaccard_FH=metric_value('Jaccard_FH'),
        HD95_FH=metric_value('HD95_FH'),
        ASD_FH=metric_value('ASD_FH'),
    )

    with open(summary_path, 'a', encoding='utf-8') as summary_file:
        summary_file.write(row)


def set_requires_grad(module, requires_grad):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def gradient_penalty(critic, real, fake, gp_weight=10.0, return_diagnostics=False):
    """WGAN-GP gradient penalty (Gulrajani 2017; Shape-Prior MICCAI'25 Eq.1).

    Truncates real/fake to the smaller batch dimension so the interpolation
    α·real + (1-α)·fake is well-defined when labeled / unlabeled batch sizes
    differ (common in semi-supervised training).

    When return_diagnostics=True, also returns (mean ‖∇D‖₂, raw penalty)
    so callers can monitor whether the 1-Lipschitz constraint is actually
    being satisfied — divergence in PSFH appears as a mean grad norm
    drifting far from 1.0.
    """
    bs = min(real.size(0), fake.size(0))
    if bs == 0:
        zero = real.new_tensor(0.0)
        if return_diagnostics:
            return zero, zero, zero
        return zero
    real = real[:bs]
    fake = fake[:bs]
    alpha = torch.rand(bs, 1, 1, 1, device=real.device, dtype=real.dtype)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp)
    grads = torch.autograd.grad(
        outputs=d_interp.sum(), inputs=interp,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grad_norm = grads.reshape(bs, -1).norm(2, dim=1)
    penalty = gp_weight * ((grad_norm - 1.0) ** 2).mean()
    if return_diagnostics:
        return penalty, grad_norm.mean().detach(), penalty.detach()
    return penalty


_bce_dice_criterion = BceDiceLoss()


def merge_foreground_slots(prob):
    prob = torch.nan_to_num(prob.float(), nan=0.0, posinf=1.0, neginf=0.0)
    if prob.detach().min() < 0.0 or prob.detach().max() > 1.0:
        prob = torch.sigmoid(prob)
    prob = prob.clamp(0.0, 1.0)
    if prob.dim() < 4 or prob.size(1) == 1:
        return prob
    return 1.0 - torch.prod(1.0 - prob, dim=1, keepdim=True)


def is_psfh_training():
    return str(args.dataset).lower() == 'psfh'


def tensor_to_device(value, device):
    if torch.is_tensor(value) and device is not None:
        return value.to(device, non_blocking=True)
    return value


def prepare_supervised_target(batch, fallback_gt, device=None):
    gt = tensor_to_device(fallback_gt, device)
    if (
        is_psfh_training()
        and torch.is_tensor(batch.get('label_ps', None))
        and torch.is_tensor(batch.get('label_fh', None))
    ):
        ps = tensor_to_device(batch['label_ps'], device).float().clamp(0.0, 1.0)
        fh = tensor_to_device(batch['label_fh'], device).float().clamp(0.0, 1.0)
        return torch.cat([ps, fh], dim=1), True
    return gt.float().clamp(0.0, 1.0), False


def psfh_ps_copy_paste(img_l, target_l, prob=0.5):
    """PS-aware copy-paste augmentation for PSFH labeled batches.

    For each image in the batch, with probability `prob`, copy the PS region
    (slot 0 of target_l) from a randomly chosen sibling sample, paste it onto
    the current image (image pixels + target_l slot 0). FH (slot 1) is updated
    to ensure mutual exclusion (set to 0 wherever new PS is positive).

    Args:
        img_l:    [B, 3, H, W] labeled images
        target_l: [B, K, H, W] binary masks where K>=2 (slot 0 = PS, slot 1 = FH)
        prob:     per-sample paste probability

    Returns: (img_l_aug, target_l_aug) — same shapes, in-place safe copies.
    """
    if target_l.dim() != 4 or target_l.size(1) < 2 or img_l.size(0) < 2:
        return img_l, target_l
    B = img_l.size(0)
    img_aug = img_l.clone()
    target_aug = target_l.clone()
    perm = torch.randperm(B, device=img_l.device)
    for i in range(B):
        if torch.rand(1, device=img_l.device).item() >= prob:
            continue
        j = int(perm[i].item())
        if j == i:
            continue
        donor_ps = target_l[j, 0:1]
        if donor_ps.sum() < 16:
            continue
        donor_mask = (donor_ps > 0.5).float()
        # Paste image pixels of donor PS onto current image
        img_aug[i] = img_aug[i] * (1.0 - donor_mask) + img_l[j] * donor_mask
        # Add donor PS to current PS (union); clamp to {0,1}
        target_aug[i, 0:1] = (target_aug[i, 0:1] + donor_mask).clamp(0.0, 1.0)
        # Mutual exclusion: zero FH wherever donor PS overlaps
        target_aug[i, 1:2] = target_aug[i, 1:2] * (1.0 - donor_mask)
    return img_aug, target_aug


def psfh_bcp_inject(img_l, target_l, img_u, prob=0.5):
    """Bidirectional Copy-Paste (BCP, CVPR'23 Bai et al.) for PSFH PS slot.

    Inject labeled-PS regions into unlabeled images so the student receives
    high-confidence PS supervision even where the unlabeled teacher would
    have failed (the dominant failure mode for small PS class — see paper §3.X).

    For each unlabeled image, with probability `prob`, pick a random labeled
    image with a non-trivial PS region, paste that PS patch into the unlabeled
    image (in image space), and return a partial target tensor for the pasted
    region so the loss can be applied selectively.

    Args:
        img_l:    [Bl, 3, H, W] labeled images
        target_l: [Bl, K, H, W] labeled targets (K>=2; slot 0 = PS)
        img_u:    [Bu, 3, H, W] unlabeled images
        prob:     per-unlabeled-sample paste probability

    Returns:
        img_u_aug:    [Bu, 3, H, W] unlabeled images with pasted PS patches
        ps_inject:    [Bu, 1, H, W] {0,1} indicator of where PS was injected
                       (1 = supervised pixel, target value = 1 for PS slot)
    """
    Bu, _, H, W = img_u.shape
    img_aug = img_u.clone()
    ps_inject = torch.zeros(Bu, 1, H, W, device=img_u.device, dtype=img_u.dtype)
    if target_l.dim() != 4 or target_l.size(1) < 1 or img_l.size(0) == 0:
        return img_aug, ps_inject
    Bl = img_l.size(0)
    # Spatial alignment: labeled and unlabeled images should share spatial dims.
    # If sizes differ, resize the labeled donor on-the-fly (rare in practice).
    for i in range(Bu):
        if torch.rand(1, device=img_u.device).item() >= prob:
            continue
        j = int(torch.randint(0, Bl, (1,), device=img_u.device).item())
        donor_ps = target_l[j, 0:1]
        donor_img = img_l[j]
        # Resize if needed
        if donor_ps.shape[-2:] != (H, W):
            donor_ps = F.interpolate(donor_ps.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
            donor_img = F.interpolate(donor_img.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        if donor_ps.sum() < 16:
            continue
        donor_mask = (donor_ps > 0.5).float()
        img_aug[i] = img_aug[i] * (1.0 - donor_mask) + donor_img * donor_mask
        ps_inject[i] = donor_mask
    return img_aug, ps_inject


def psfh_multilabel_loss(pred, target, w_neg=(0.10, 0.02), smooth=1e-5):
    """Per-class weighted soft Dice + BCE for PSFHS multi-label segmentation.

    WSDice (Improved Dice, IEEE Access 2020) gives the negative area a small
    weight w_neg, which restores the gradient magnitude for the small PS class
    (≈5% of image area) without overwhelming FH (≈30%). Defaults: PS=0.10, FH=0.02.
    """
    pred = as_probability_map(pred)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    pred = pred[:, :target.size(1)]
    K = target.size(1)
    if not isinstance(w_neg, (list, tuple)):
        w_neg = [float(w_neg)] * K
    w_neg = [float(w) for w in w_neg[:K]]
    while len(w_neg) < K:
        w_neg.append(w_neg[-1])

    bce = F.binary_cross_entropy(pred, target, reduction='none').mean()

    dsc_terms = []
    for k in range(K):
        p, y = pred[:, k], target[:, k]
        inter = (p * y).sum()
        neg = ((1 - p) * (1 - y)).sum()
        denom = p.sum() + y.sum()
        dsc = (2 * inter + w_neg[k] * neg + smooth) / (denom + 2 * w_neg[k] * neg + smooth)
        dsc_terms.append(1 - dsc)
    dice = torch.stack(dsc_terms).mean()
    return bce + dice


def DeepSupSeg(pred, gt):
    return _bce_dice_criterion(pred, gt)


def get_boundary_map(mask_batch):
    """
    Build a normalized continuous boundary target shared by boundary supervision,
    SCD, SOR, and pseudo-label calibration.
    """
    return get_structure_boundary(mask_batch)


def apply_strong_augmentation(img, brightness=0.4, contrast=0.4, saturation=0.2,
                              gaussian_noise=0.1, cutout_prob=0.5, cutout_max_ratio=0.3):
    """FixMatch-style strong augmentation on a batched image tensor.

    Used only on the *student* path for unlabeled images so the network
    must produce a teacher-consistent prediction even under aggressive
    perturbation. The teacher path keeps the weakly-augmented image.

    Operations are applied in image space (no spatial perturbation here —
    pseudo-labels are computed in the original spatial frame and must
    align pixel-wise with the student prediction).

    img: [B, C, H, W] in [0, 1]
    Returns augmented [B, C, H, W] in [0, 1].
    """
    if img.dim() != 4:
        return img
    B, C, H, W = img.shape
    aug = img.clone()
    # Per-sample independent jitter — strong augmentation should differ across samples.
    # 1) Brightness: img * (1 + b)
    if brightness > 0:
        b = (torch.rand(B, 1, 1, 1, device=img.device) * 2 - 1) * brightness
        aug = aug * (1.0 + b)
    # 2) Contrast: (img - mean) * (1 + c) + mean
    if contrast > 0:
        c = (torch.rand(B, 1, 1, 1, device=img.device) * 2 - 1) * contrast
        m = aug.mean(dim=(2, 3), keepdim=True)
        aug = (aug - m) * (1.0 + c) + m
    # 3) Saturation: only meaningful for 3-channel RGB
    if saturation > 0 and C == 3:
        s = (torch.rand(B, 1, 1, 1, device=img.device) * 2 - 1) * saturation
        gray = aug.mean(dim=1, keepdim=True)
        aug = gray + (1.0 + s) * (aug - gray)
    # 4) Gaussian noise
    if gaussian_noise > 0:
        sigma = torch.rand(B, 1, 1, 1, device=img.device) * gaussian_noise
        aug = aug + torch.randn_like(aug) * sigma
    # 5) Cutout — random rectangle filled with batch mean color
    if cutout_prob > 0 and cutout_max_ratio > 0:
        do_cut = torch.rand(B, device=img.device) < cutout_prob
        for i in range(B):
            if not bool(do_cut[i]):
                continue
            ch = int(H * cutout_max_ratio * torch.rand(1).item() + 8)
            cw = int(W * cutout_max_ratio * torch.rand(1).item() + 8)
            ch = min(ch, H)
            cw = min(cw, W)
            top = torch.randint(0, max(1, H - ch + 1), (1,)).item()
            left = torch.randint(0, max(1, W - cw + 1), (1,)).item()
            mean_c = aug[i].mean(dim=(1, 2), keepdim=True)
            aug[i, :, top:top + ch, left:left + cw] = mean_c
    return aug.clamp(0.0, 1.0)


def split_batch_outputs(outputs, first_batch_size):
    if torch.is_tensor(outputs):
        return outputs[:first_batch_size], outputs[first_batch_size:]
    if isinstance(outputs, tuple):
        first_parts = []
        second_parts = []
        for item in outputs:
            first_item, second_item = split_batch_outputs(item, first_batch_size)
            first_parts.append(first_item)
            second_parts.append(second_item)
        return tuple(first_parts), tuple(second_parts)
    if isinstance(outputs, list):
        first_parts = []
        second_parts = []
        for item in outputs:
            first_item, second_item = split_batch_outputs(item, first_batch_size)
            first_parts.append(first_item)
            second_parts.append(second_item)
        return first_parts, second_parts
    raise TypeError(f'Unsupported model output type for batch split: {type(outputs)!r}')


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def adjust_lr_rate(argsimizer, iter, total_batch):
    lr = lr_poly(args.lr, iter, args.nEpoch * total_batch, args.power)
    argsimizer.param_groups[0]['lr'] = lr
    return lr


def compute_supervised_losses(main_prob, aux_prob, boundary_logits, target, boundary_target=None, psfh_slots=False, w_neg=None):
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if (
        psfh_slots
        and not args.no_psfh_wsdice
        and target.dim() == 4
        and target.size(1) > 1
        and main_prob.dim() == 4
        and main_prob.size(1) >= target.size(1)
    ):
        main_for_loss = as_probability_map(main_prob)[:, :target.size(1)]
        aux_for_loss = as_probability_map(aux_prob)[:, :target.size(1)]
        target_for_boundary = merge_foreground_slots(target)
        wsdice_w = w_neg if w_neg is not None else (args.wsdice_w_ps, args.wsdice_w_fh)
        loss_main = psfh_multilabel_loss(main_for_loss, target, wsdice_w)
        loss_aux = psfh_multilabel_loss(aux_for_loss, target, wsdice_w)
    else:
        main_for_loss = merge_foreground_slots(main_prob)
        aux_for_loss = merge_foreground_slots(aux_prob)
        target_for_boundary = merge_foreground_slots(target)
        loss_main = DeepSupSeg(main_for_loss, target_for_boundary)
        loss_aux = DeepSupSeg(aux_for_loss, target_for_boundary)
    if boundary_target is None:
        boundary_target = get_boundary_map(target_for_boundary)
    loss_boundary = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
    return loss_main, loss_aux, loss_boundary


def train():
    """load data"""
    train_l_data, _, valid_data = build_dataset(args)
    log_dataset_splits('train', train_l=train_l_data, valid=valid_data)
    train_l_dataloader = build_dataloader(
        train_l_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = build_dataloader(
            valid_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            use_distributed_sampler=False,
        )
        val_total_batch = math.ceil(len(valid_data) / args.batch_size)

    """load model"""
    model = build_model(args)

    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train
    log_info('\n---------------------------------')
    log_info('Start training')
    log_info('---------------------------------\n')

    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best_f1_epoch = -1
    best_dice = 0
    best_dice_epoch = -1
    benchmark_start_time = None
    benchmark_measured_steps = 0
    benchmark_measured_samples = 0
    for epoch in range(args.nEpoch):
        model.train()
        if getattr(train_l_dataloader, 'dist_sampler', None) is not None:
            train_l_dataloader.dist_sampler.set_epoch(epoch)

        log_info("Epoch: {}".format(epoch))
        total_batch = math.ceil(len(train_l_data) / args.batch_size)
        bar = tqdm(enumerate(train_l_dataloader), total=total_batch, disable=args.disable_tqdm or not sys.stdout.isatty() or not is_main_process(), dynamic_ncols=False, leave=False)
        epoch_loss_total = 0.0
        epoch_main_total = 0.0
        epoch_aux_total = 0.0
        epoch_boundary_total = 0.0
        for batch_id, data_l in bar:
            batch_start = time.perf_counter()
            itr = total_batch * epoch + batch_id
            img, gt = data_l['image'], data_l['label']
            device = None
            if args.gpu_ids:
                device = torch.device('cuda', args.primary_gpu)
                img = img.cuda(args.primary_gpu)
                gt = gt.cuda(args.primary_gpu)
            target_l, has_psfh_slots = prepare_supervised_target(data_l, gt, device)
            optim.zero_grad(set_to_none=True)
            with autocast_context():
                pred = model(img)
            mask = pred[0].float()
            aux_mask = pred[1].float()
            boundary_logits = pred[-2].float()
            loss_main, loss_aux, loss_boundary = compute_supervised_losses(
                mask,
                aux_mask,
                boundary_logits,
                target_l,
                psfh_slots=has_psfh_slots,
            )
            loss = (
                loss_main +
                args.aux_supervision_weight * loss_aux +
                args.boundary_weight * loss_boundary
            )
            loss.backward()
            optim.step()
            lr = adjust_lr_rate(optim, itr, total_batch)

            if args.benchmark_steps > 0:
                if benchmark_measured_steps == 0 and batch_id + 1 == args.benchmark_warmup_steps + 1:
                    if args.gpu_ids:
                        torch.cuda.synchronize()
                    benchmark_start_time = time.perf_counter()
                if batch_id + 1 > args.benchmark_warmup_steps:
                    benchmark_measured_steps += 1
                    benchmark_measured_samples += img.size(0)
                    maybe_finish_benchmark(
                        'semi_supervised_full',
                        benchmark_measured_steps,
                        benchmark_measured_samples,
                        benchmark_start_time,
                    )

            _batch_floats = torch.stack([
                loss.detach().mean(), loss_main.detach().mean(),
                loss_aux.detach().mean(), loss_boundary.detach().mean(),
            ]).float().cpu().tolist()
            epoch_loss_total += _batch_floats[0]
            epoch_main_total += _batch_floats[1]
            epoch_aux_total += _batch_floats[2]
            epoch_boundary_total += _batch_floats[3]

            if should_log_batch(batch_id, total_batch):
                log_full_train_batch(
                    epoch,
                    args.nEpoch,
                    batch_id,
                    total_batch,
                    time.perf_counter() - batch_start,
                    lr,
                    loss,
                    loss_main,
                    loss_aux,
                    loss_boundary,
                )

        if total_batch > 0:
            log_info(
                "[Train][Epoch {}/{} Summary] avg_loss={:.4f} avg_main={:.4f} avg_aux={:.4f} avg_boundary={:.4f}".format(
                    epoch + 1,
                    args.nEpoch,
                    epoch_loss_total / total_batch,
                    epoch_main_total / total_batch,
                    epoch_aux_total / total_batch,
                    epoch_boundary_total / total_batch,
                )
            )

        if valid_sign and is_main_process() and should_run_validation(epoch):
            recall, specificity, precision, F1, F2, \
                ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice, *_, table_metrics = evaluate(
                    model,
                    valid_dataloader,
                    val_total_batch,
                    tta=args.eval_tta,
                    **eval_postprocess_kwargs(),
                )

            log_info("Valid Result:")
            log_info(
                'recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice))
            log_validation_details(table_metrics)

            if dice > best_dice:
                best_dice = dice
                best_dice_epoch = epoch + 1
                torch.save(get_model_state_dict(model), os.path.join(args.checkpoint_root, args.ckpt_name, "best_dice.pth"))
            log_info("Best Dice:: {}".format(best_dice))

            if (F1 > F1_best):
                F1_best = F1
                best_f1_epoch = epoch + 1
                torch.save(get_model_state_dict(model), os.path.join(args.checkpoint_root, args.ckpt_name, "best.pth"))
            elif (F1 > F1_second_best):
                F1_second_best = F1
                torch.save(get_model_state_dict(model), os.path.join(args.checkpoint_root, args.ckpt_name, "second_best.pth"))
            elif (F1 > F1_third_best):
                F1_third_best = F1
                torch.save(get_model_state_dict(model), os.path.join(args.checkpoint_root, args.ckpt_name, "third_best.pth"))
        if args.ddp:
            torch.distributed.barrier()
    if is_main_process():
        emit_training_result('full', F1_best, best_f1_epoch, best_dice, best_dice_epoch)
    if args.ddp:
        torch.distributed.destroy_process_group()

def train_semi():
    """
    半监督训练主闭环：用一个统一的可靠性伪标签目标替代零散 loss 堆叠。

    无标注样本只围绕一个变量建模：pseudo-label reliability。可靠性同时
    读取 teacher 置信度、主/辅视图一致性、超声边界证据和可选解剖先验；
    Dice/HD95/ASD 的提升应来自更干净的伪标签监督，而不是指标保护技巧。
    """
    # ========== 数据加载 ==========
    train_l_data, train_u_data, valid_data = build_dataset(args)
    log_dataset_splits('train_semi', train_l=train_l_data, train_u=train_u_data, valid=valid_data)
    train_l_dataloader = build_dataloader(
        train_l_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    train_u_dataloader = build_dataloader(
        train_u_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = build_dataloader(
            valid_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            use_distributed_sampler=False,
        )
        val_total_batch = math.ceil(len(valid_data) / args.batch_size)
    """load model"""
    model = build_model(args)
    model_cps = build_model(args, wrap_distributed=False)
    load_model_state_dict(model_cps, get_model_state_dict(model))
    model_cps.eval()

    # ========== 初始化SCD判别器（论文第3.2节） ==========
    if not args.no_scd:
        netD_weight_path = get_scd_pretrain_path()
        netD_weight = None
        scd_norm_type = 'instance'
        if os.path.isfile(netD_weight_path):
            netD_weight = torch.load(netD_weight_path, map_location='cpu')
            scd_norm_type = infer_scd_norm_type(netD_weight)

        """
        Structure-Contrast Discriminator (SCD) 初始化
        论文第3.2节：使用DCGAN风格的判别器架构
        
        输入设计：
        - 论文公式(8): Z = Concat(F_u, B)，其中F_u是512维特征，B是1维边界图
        - 理论上输入通道数应为 512 + 1 = 513
        - 为使用预训练权重，使用特征适配器将512维降维到3维
        - 最终输入：3（特征）+ 1（边界）= 4通道，兼容预训练判别器
        """
        # 初始化判别器：DCGAN架构，输入4通道，输出64x64特征图
        # Keep the SCD branch on the process-local device. DataParallel can fail
        # when the discriminator returns intermediate feature lists.
        netD = configure_cuda_model(
            DCGAN_D(isize=64, nz=100, nc=5, ndf=64, ngpu=1, norm_type=scd_norm_type),
            args,
            wrap_distributed=False,
        )

        # 特征适配器：将512维编码器特征降维到3维（与 GAN pretrain 的 [image(3) + ps(1) + fh(1)] 5ch 输入对齐）
        feature_adapter = configure_cuda_model(nn.Sequential(
            nn.Conv2d(512, 3, kernel_size=1, bias=False),  # 1x1卷积降维
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        ), args, wrap_distributed=False)
        
        # 加载预训练的判别器权重（论文提到判别器需要预训练以稳定训练）
        if netD_weight is not None:
            current_state = unwrap_model(netD).state_dict()
            new_state_dict, load_stats = translate_scd_checkpoint_state(current_state, netD_weight)
            current_state.update(new_state_dict)
            unwrap_model(netD).load_state_dict(current_state)
            log_info(
                "[SCD] Loaded {} compatible tensors from {} "
                "(norm_type={}, skipped_initial={}, skipped_missing={}, skipped_shape={})".format(
                    load_stats['loaded'],
                    netD_weight_path,
                    scd_norm_type,
                    load_stats['skipped_initial'],
                    load_stats['skipped_missing'],
                    load_stats['skipped_shape'],
                )
            )
        else:
            print(f"[WARN] SCD pretrained weights not found at {netD_weight_path}; using random discriminator initialization.")
        netD.eval()  # 初始时设为评估模式

        # 特征适配器的优化器（需要单独优化）
        optim_adapter = torch.optim.Adam(feature_adapter.parameters(), lr=args.adapter_lr, weight_decay=args.weight_decay)
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.scd_lr, betas=(0.5, 0.999))
    else:
        feature_adapter = None
        optim_adapter = None
        optimizer_D = None

    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher_ckpt_dir = os.path.join(args.checkpoint_root, args.ckpt_name + "_teacher")
    os.makedirs(teacher_ckpt_dir, exist_ok=True)

    # train
    log_info('\n---------------------------------')
    log_info('Start training_semi')
    log_info('---------------------------------\n')
    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best_f1_epoch = -1
    best_dice = 0
    best_dice_epoch = -1
    benchmark_start_time = None
    benchmark_measured_steps = 0
    benchmark_measured_samples = 0
    signal_state = init_signal_state() if signal_mode_active() else None
    # Adaptive WSDice w_neg state (Theorem 1: gradient balance).
    # Initialised from the fixed --wsdice_w_ps / --wsdice_w_fh so the first
    # few batches behave like the static baseline before the EMA warms up.
    adaptive_w_neg = [float(args.wsdice_w_ps), float(args.wsdice_w_fh)]
    for epoch in range(args.nEpoch):
        model.train()
        if not args.no_scd:
            feature_adapter.train()
        if getattr(train_l_dataloader, 'dist_sampler', None) is not None:
            train_l_dataloader.dist_sampler.set_epoch(epoch)
        if getattr(train_u_dataloader, 'dist_sampler', None) is not None:
            train_u_dataloader.dist_sampler.set_epoch(epoch)
        log_info("Epoch: {}".format(epoch))
        loader = iter(zip(cycle(train_l_dataloader), train_u_dataloader))
        total_batch = len(train_u_dataloader)
        bar = create_progress(total_batch)
        epoch_loss_total = 0.0
        epoch_labeled_total = 0.0
        epoch_unlabeled_total = 0.0
        epoch_diag_stats = {}
        for batch_id in bar:
            batch_start = time.perf_counter()
            data_l, data_u = next(loader)
            itr = total_batch * epoch + batch_id
            img_l, gt = data_l['image'], data_l['label']
            img_u = data_u
            device = None
            if args.gpu_ids:
                device = torch.device('cuda', args.primary_gpu)
                img_l = img_l.cuda(args.primary_gpu)
                gt = gt.cuda(args.primary_gpu)
                img_u = img_u.cuda(args.primary_gpu)
            target_l, has_psfh_slots = prepare_supervised_target(data_l, gt, device)
            # PS-aware copy-paste augmentation on the labeled batch (PSFH only)
            if (
                has_psfh_slots
                and args.ps_copy_paste_prob > 0
                and target_l.size(1) >= 2
                and img_l.size(0) >= 2
            ):
                img_l, target_l = psfh_ps_copy_paste(img_l, target_l, prob=args.ps_copy_paste_prob)
            optim.zero_grad(set_to_none=True)
            if optim_adapter is not None:
                optim_adapter.zero_grad(set_to_none=True)

            # ========== 学生模型：合并标注/无标注前向，减少一次完整网络推理 ==========
            # FixMatch-style strong augmentation on the student's unlabeled input
            # (paper §3.X). Teacher path still uses the raw (weakly-augmented by
            # dataloader) image to produce stable pseudo-labels.
            if args.use_strong_aug and img_u.size(0) > 0:
                img_u_strong = apply_strong_augmentation(
                    img_u,
                    brightness=args.strong_brightness,
                    contrast=args.strong_contrast,
                    saturation=args.strong_saturation,
                    gaussian_noise=args.strong_noise,
                    cutout_prob=args.strong_cutout_prob,
                    cutout_max_ratio=args.strong_cutout_ratio,
                )
            else:
                img_u_strong = img_u
            # Bidirectional Copy-Paste (BCP, paper §3.X): inject labeled PS regions
            # into the student's unlabeled image so PS receives high-confidence
            # supervision regardless of EMA-teacher uncertainty.
            ps_inject = None
            if (
                args.use_bcp
                and has_psfh_slots
                and target_l.size(1) >= 1
                and img_u_strong.size(0) > 0
            ):
                img_u_strong, ps_inject = psfh_bcp_inject(
                    img_l, target_l, img_u_strong, prob=args.bcp_prob,
                )
            combined_img = torch.cat([img_l, img_u_strong], dim=0)
            with autocast_context():
                pred_combined = model(combined_img)
            pred_l, pred_u = split_batch_outputs(pred_combined, img_l.size(0))

            # ========== 标注数据：主头/辅助头/边界头联合监督 ==========
            mask_l = as_probability_map(pred_l[0].float())
            aux_mask_l = as_probability_map(pred_l[1].float())
            boundary_logits_l = torch.nan_to_num(pred_l[-2].float(), nan=0.0, posinf=30.0, neginf=-30.0)
            feat_l = pred_l[-1]
            target_l_fg = merge_foreground_slots(target_l)
            boundary_map_gt = get_boundary_map(target_l_fg)
            # Adaptive WSDice w_neg via Theorem 1 (gradient balance), EMA-smoothed.
            # Computed once per batch from the labeled targets and reused for
            # both supervised and unlabeled (teacher-target) WSDice calls.
            # Per-class w_neg dispatch. Each scheme produces a length-K list;
            # gradient_balanced is the paper's Theorem 1, inverse_freq and
            # effective_number are reviewer-facing baselines. The fixed path
            # uses static --wsdice_w_ps/--wsdice_w_fh.
            mode = args.wsdice_mode
            if mode == 'auto':
                mode = 'adaptive' if args.adaptive_wsdice else 'fixed'
            if (
                mode in ('adaptive', 'inverse_freq', 'effective_number')
                and has_psfh_slots
                and target_l.size(1) >= 2
            ):
                if mode == 'adaptive':
                    target_w = gradient_balanced_w_neg(
                        target_l[:, :2], alpha=args.adaptive_wsdice_alpha,
                    )
                elif mode == 'inverse_freq':
                    target_w = inverse_freq_w_neg(
                        target_l[:, :2], alpha=args.adaptive_wsdice_alpha,
                    )
                else:  # effective_number
                    target_w = effective_number_w_neg(
                        target_l[:, :2],
                        beta=args.wsdice_eff_beta,
                        alpha=args.adaptive_wsdice_alpha,
                    )
                beta = float(args.adaptive_wsdice_ema)
                adaptive_w_neg = [
                    beta * adaptive_w_neg[k] + (1.0 - beta) * float(target_w[k])
                    for k in range(min(2, len(target_w)))
                ]
                supervised_w_neg = tuple(adaptive_w_neg)
            else:
                supervised_w_neg = (args.wsdice_w_ps, args.wsdice_w_fh)
            loss_l_main, loss_l_aux, loss_l_boundary = compute_supervised_losses(
                mask_l,
                aux_mask_l,
                boundary_logits_l,
                target_l,
                boundary_target=boundary_map_gt,
                psfh_slots=has_psfh_slots,
                w_neg=supervised_w_neg,
            )
            loss_l = (
                loss_l_main +
                args.aux_supervision_weight * loss_l_aux +
                args.boundary_weight * loss_l_boundary
            )

            # ========== 无标注数据：结构感知伪标签校准 ==========
            mask_u = as_probability_map(pred_u[0].float())
            aux_mask_u = as_probability_map(pred_u[1].float())
            boundary_logits_u = torch.nan_to_num(pred_u[-2].float(), nan=0.0, posinf=30.0, neginf=-30.0)
            feat_u = pred_u[-1]
            mask_u_fg = merge_foreground_slots(mask_u)
            aux_mask_u_fg = merge_foreground_slots(aux_mask_u)

            boundary_map_mask_u = get_boundary_map(mask_u_fg)

            with torch.no_grad():
                with autocast_context():
                    teacher_pred_u = model_cps(img_u)[0]
                teacher_prob_u = as_probability_map(teacher_pred_u.float())
                teacher_prob_u_fg = merge_foreground_slots(teacher_prob_u)

            # Single Gaussian ramp (BiPCC IV-B Eq.16): λ(t) = α·exp(-β(1-t/T)²).
            # Drives every semi-supervised term uniformly so they cannot race each
            # other. The discriminator has its own sigmoid ramp (scd_ramp) since
            # the baseline (Shape-Prior) wants D to stabilize before driving G.
            lam = gaussian_rampup(epoch, args.nEpoch, args.alpha_lam, args.beta_lam)
            structure_ramp = lam
            teacher_weight = lam
            scd_active = (not args.no_scd) and epoch >= args.scd_start_epoch
            scd_ramp = sigmoid_rampup(epoch - args.scd_start_epoch, args.scd_rampup) if scd_active else 0.0

            loss_adv = torch.tensor(0.0, device=img_u.device)
            scd_real_prob = torch.tensor(0.0, device=img_u.device)
            scd_fake_prob = torch.tensor(0.0, device=img_u.device)
            scd_d_loss = torch.tensor(0.0, device=img_u.device)
            if scd_active:
                set_requires_grad(netD, True)
                netD.train()

                feat_l_resized = F.interpolate(feat_l, size=boundary_map_gt.shape[2:], mode='bilinear', align_corners=False)
                feat_u_resized = F.interpolate(feat_u, size=boundary_map_mask_u.shape[2:], mode='bilinear', align_corners=False)

                with autocast_context():
                    feat_l_adapted = feature_adapter(feat_l_resized)
                    feat_u_adapted = feature_adapter(feat_u_resized)

                # Per-class boundary inputs to the critic (2 channels).
                # Single-class datasets pad the second channel with zeros so the
                # critic input stays 4-channel (2 feat + 2 boundary) and matches
                # the pretrained DCGAN_D weights.
                if has_psfh_slots and target_l.size(1) >= 2 and mask_u.size(1) >= 2:
                    real_boundary_ps = get_boundary_map(target_l[:, 0:1])
                    real_boundary_fh = get_boundary_map(target_l[:, 1:2])
                    real_boundary_2ch = torch.cat([real_boundary_ps, real_boundary_fh], dim=1)
                    fake_boundary_ps = get_boundary_map(mask_u[:, 0:1])
                    fake_boundary_fh = get_boundary_map(mask_u[:, 1:2])
                    fake_boundary_2ch = torch.cat([fake_boundary_ps, fake_boundary_fh], dim=1)
                else:
                    zero_pad = torch.zeros_like(boundary_map_gt)
                    real_boundary_2ch = torch.cat([boundary_map_gt, zero_pad], dim=1)
                    zero_pad_u = torch.zeros_like(boundary_map_mask_u)
                    fake_boundary_2ch = torch.cat([boundary_map_mask_u, zero_pad_u], dim=1)

                real_feat_boundary = torch.cat([feat_l_adapted.detach(), real_boundary_2ch.detach()], dim=1)
                fake_feat_boundary = torch.cat([feat_u_adapted.detach(), fake_boundary_2ch.detach()], dim=1)

                # Wasserstein critic loss + gradient penalty (Shape-Prior MICCAI'25 Eq.1).
                # GP must be in fp32 with grad enabled; keep the critic forward out of autocast.
                real_pred = netD(real_feat_boundary).float()
                fake_pred = netD(fake_feat_boundary).float()
                # Bounded Wasserstein Critic (BWC, paper §3.X Proposition 2):
                # soft tanh clamp on critic outputs to enforce |D̃(x)| ≤ c.
                # WGAN-GP's gradient penalty enforces 1-Lipschitz only softly; we
                # empirically observed |D(x)| drifting to ±10 on PSFH/TN3K, which
                # makes the adversarial loss dominate the pseudo-label gradient.
                # BWC preserves the gradient direction of WGAN-GP near the
                # 1-Lipschitz region while providing an architectural upper bound.
                if args.bwc_clamp > 0:
                    c = float(args.bwc_clamp)
                    real_pred = c * torch.tanh(real_pred / c)
                    fake_pred = c * torch.tanh(fake_pred / c)
                gp, gp_grad_norm, gp_value = gradient_penalty(
                    netD, real_feat_boundary, fake_feat_boundary, return_diagnostics=True,
                )
                errD = fake_pred.mean() - real_pred.mean() + gp
                scd_real_prob = real_pred.mean()
                scd_fake_prob = fake_pred.mean()
                scd_d_loss = errD

                if (itr + 1) % max(1, args.scd_update_interval) == 0:
                    optimizer_D.zero_grad(set_to_none=True)
                    errD.backward()
                    optimizer_D.step()

                netD.eval()
                set_requires_grad(netD, False)

                fake_feat_boundary_G = torch.cat([feat_u_adapted, fake_boundary_2ch], dim=1)
                with autocast_context():
                    pred_fake_for_G = netD(fake_feat_boundary_G)
                pred_fake_for_G = pred_fake_for_G.float()
                # BWC clamp on the critic value used by the G objective so the
                # adversarial signal feeding back into the segmentation network
                # is bounded by |c| (matches the D-side clamp above).
                if args.bwc_clamp > 0:
                    c = float(args.bwc_clamp)
                    pred_fake_for_G = c * torch.tanh(pred_fake_for_G / c)
                # G objective: minimize -E[D(x̃)]  (Shape-Prior MICCAI'25 Eq.2)
                loss_adv = -pred_fake_for_G.mean()

            loss_u, loss_u_main, loss_u_aux, loss_calibration, rectified_pseudo, reliability_map, _, _ = reliable_pseudo_label_loss(
                mask_u_fg,
                aux_mask_u_fg,
                boundary_logits_u,
                teacher_prob=teacher_prob_u_fg,
                pseudo_main_weight=args.pseudo_main_weight,
                pseudo_aux_weight=args.pseudo_aux_weight,
                pseudo_boundary_weight=structure_ramp * args.calibration_weight,
                entropy_tau=args.entropy_tau,
                reliability_floor=args.reliability_floor,
            )
            loss_edge = torch.tensor(0.0, device=img_u.device)
            loss_sor = torch.tensor(0.0, device=img_u.device)
            # EMA-teacher MSE consistency removed: the PL term already supervises the
            # student against teacher_prob_u_fg with entropy reliability weighting, so
            # adding a separate MSE on the weak-augmented teacher only duplicates
            # gradient signal and dilutes the entropy-weighted PL objective (which
            # was the intended unsupervised driver per the plan).
            loss_teacher = torch.tensor(0.0, device=img_u.device)
            # PSFH unlabeled: per-class WSDice on student vs teacher slots (replaces
            # the previous slot_consistency + per-slot weighted BCE-Dice + topology stack).
            if has_psfh_slots and (not args.no_psfh_wsdice) and mask_u.size(1) > 1:
                # Reuse the adaptive w_neg already updated in the labeled-forward
                # block above. When --adaptive_wsdice is off, supervised_w_neg
                # falls back to the static --wsdice_w_ps / --wsdice_w_fh.
                w_neg = supervised_w_neg if has_psfh_slots else (args.wsdice_w_ps, args.wsdice_w_fh)
                loss_u_psfh_main = psfh_multilabel_loss(
                    mask_u, teacher_prob_u.detach(), w_neg=w_neg)
                loss_u_psfh_aux = psfh_multilabel_loss(
                    aux_mask_u, teacher_prob_u.detach(), w_neg=w_neg)
                loss_u = loss_u + structure_ramp * (
                    loss_u_psfh_main + args.pseudo_aux_weight * loss_u_psfh_aux
                )
                # BCP supervision (paper §3.X): pixels where labeled PS was
                # injected into the unlabeled image must be predicted as PS by
                # the student. This bypasses the EMA-teacher's PS uncertainty
                # and provides high-confidence PS gradient even when the rest
                # of the unlabeled batch is ambiguous.
                if ps_inject is not None and ps_inject.sum() > 0:
                    ps_pred = mask_u[:, 0:1]
                    bcp_target = ps_inject.detach()
                    # Weight only on injected pixels (sum > 0). BCE on the
                    # injected region; sum / (mask sum + ε) gives a per-pixel
                    # mean restricted to the supervised region.
                    bce_per_pixel = F.binary_cross_entropy(
                        ps_pred.clamp(1e-6, 1 - 1e-6), bcp_target, reduction='none')
                    bcp_loss = (bce_per_pixel * bcp_target).sum() / (bcp_target.sum() + 1e-6)
                    loss_u = loss_u + args.bcp_weight * bcp_loss

            if scd_active:
                loss_u = loss_u + scd_ramp * args.adv_weight * loss_adv

            # Cross-teaching (luo22b §3.2): scaled by the single λ(t) and a fixed
            # cross_weight (default 0.3 from the original paper) — no more
            # multi-piece magic-constant schedule.
            cross_teaching_weight = lam * args.cross_weight
            loss_cross = torch.tensor(0.0, device=img_u.device)
            if cross_teaching_weight > 0:
                # For PSFH we compute reliability per-channel from the teacher's
                # per-class probabilities, so PS doesn't get drowned out by FH.
                # For single-class datasets the broadcast 1-channel map suffices.
                if has_psfh_slots and mask_u.size(1) > 1 and teacher_prob_u.size(1) == mask_u.size(1):
                    if args.use_carc and scd_active:
                        # CARC (Class-Aware Reliability Coupling): multiplicative
                        # fusion of pixel-entropy reliability with shape-critic
                        # reliability and class-importance weights. Replaces the
                        # pure entropy weighting on the unlabeled path.
                        class_area_t = target_l[:, :2].mean(dim=(0, 2, 3)).detach() \
                            if target_l.size(1) >= 2 else None
                        rel_map = carc_reliability(
                            teacher_prob_u,
                            shape_score=pred_fake_for_G.detach(),
                            class_area=class_area_t,
                            tau=args.entropy_tau,
                            reliability_floor=args.reliability_floor,
                        ).detach()
                    else:
                        rel_map = entropy_weight(teacher_prob_u, tau=args.entropy_tau).clamp(
                            float(args.reliability_floor), 1.0).detach()
                else:
                    rel_map = reliability_map.expand_as(mask_u)
                loss_cross_main = weighted_bce_dice_loss(
                    mask_u, aux_mask_u.detach(), rel_map * cross_teaching_weight)
                loss_cross_aux = weighted_bce_dice_loss(
                    aux_mask_u, mask_u.detach(), rel_map * cross_teaching_weight)
                loss_cross = loss_cross_main + args.pseudo_aux_weight * loss_cross_aux
                loss_u = loss_u + loss_cross


            student_prob_u = mask_u_fg
            aux_prob_u = aux_mask_u_fg
            reliability_mean = reliability_map.mean()
            reliability_high = (reliability_map > 0.5).float().mean()
            pseudo_fg = rectified_pseudo.mean()
            pseudo_pos = (rectified_pseudo > 0.5).float().mean()
            student_fg = student_prob_u.mean()
            teacher_fg = teacher_prob_u_fg.mean()
            main_aux_gap = torch.abs(student_prob_u - aux_prob_u).mean()
            student_teacher_gap = torch.abs(student_prob_u - teacher_prob_u_fg).mean()
            unlabeled_trust = compute_unlabeled_trust(
                reliability_mean,
                main_aux_gap,
                student_teacher_gap,
            )

            # ===== PSFH-only per-class diagnostics =====
            # All zero on non-PSFH datasets so the same dict shape is logged.
            zero = torch.tensor(0.0, device=img_u.device)
            psfh_ps_area_gt = psfh_fh_area_gt = zero
            psfh_ps_area_student = psfh_fh_area_student = zero
            psfh_ps_area_teacher = psfh_fh_area_teacher = zero
            psfh_ps_dsc_sup = psfh_fh_dsc_sup = zero
            psfh_critic_ps_ratio = zero
            if has_psfh_slots and target_l.size(1) >= 2 and mask_u.size(1) >= 2:
                # GT vs student/teacher area fractions on the unlabeled batch
                # (PSFH-specific: separates PS imbalance from FH).
                psfh_ps_area_student = mask_u[:, 0].mean().detach()
                psfh_fh_area_student = mask_u[:, 1].mean().detach()
                if teacher_prob_u.size(1) >= 2:
                    psfh_ps_area_teacher = teacher_prob_u[:, 0].mean().detach()
                    psfh_fh_area_teacher = teacher_prob_u[:, 1].mean().detach()
                # GT area baseline (from labeled batch — known ratios)
                psfh_ps_area_gt = target_l[:, 0].mean().detach()
                psfh_fh_area_gt = target_l[:, 1].mean().detach()
                # Per-class DSC on labeled batch — does supervised loss actually
                # teach the student PS structure?
                eps_ = 1e-6
                ps_pred = (mask_l[:, 0] > 0.5).float().detach()
                ps_gt = (target_l[:, 0] > 0.5).float()
                fh_pred = (mask_l[:, 1] > 0.5).float().detach()
                fh_gt = (target_l[:, 1] > 0.5).float()
                psfh_ps_dsc_sup = (
                    (2 * (ps_pred * ps_gt).sum() + eps_)
                    / (ps_pred.sum() + ps_gt.sum() + eps_)
                ).detach()
                psfh_fh_dsc_sup = (
                    (2 * (fh_pred * fh_gt).sum() + eps_)
                    / (fh_pred.sum() + fh_gt.sum() + eps_)
                ).detach()
                # PS contribution to the critic input (merged boundary). If PS
                # is dominated by FH in the boundary, the critic effectively
                # ignores PS shape — explains why DSR can't help PS.
                ps_boundary_mass = get_boundary_map(mask_u[:, :1]).abs().mean().detach()
                fh_boundary_mass = get_boundary_map(mask_u[:, 1:2]).abs().mean().detach()
                total_mass = (ps_boundary_mass + fh_boundary_mass).clamp_min(1e-6)
                psfh_critic_ps_ratio = (ps_boundary_mass / total_mass).detach()

            # WGAN-GP grad-norm diagnostic (0 when SCD inactive)
            gp_grad_norm_diag = zero
            gp_value_diag = zero
            if scd_active:
                gp_grad_norm_diag = gp_grad_norm
                gp_value_diag = gp_value

            loss = loss_l + loss_u
            loss.backward()
            optim.step()
            if optim_adapter is not None:
                optim_adapter.step()
            if scd_active:
                set_requires_grad(netD, True)
            update_ema_variables(model, model_cps, args.mt)

            lr = adjust_lr_rate(optim, itr, total_batch)
            lr_scale = lr / max(args.lr, 1e-12)
            if optim_adapter is not None:
                set_optimizer_lr(optim_adapter, args.adapter_lr * lr_scale)
            if optimizer_D is not None:
                set_optimizer_lr(optimizer_D, args.scd_lr * lr_scale)

            if args.benchmark_steps > 0:
                if benchmark_measured_steps == 0 and batch_id + 1 == args.benchmark_warmup_steps + 1:
                    if args.gpu_ids:
                        torch.cuda.synchronize()
                    benchmark_start_time = time.perf_counter()
                if batch_id + 1 > args.benchmark_warmup_steps:
                    benchmark_measured_steps += 1
                    benchmark_measured_samples += img_u.size(0)
                    maybe_finish_benchmark(
                        'semi_supervised_semi',
                        benchmark_measured_steps,
                        benchmark_measured_samples,
                        benchmark_start_time,
                    )

            update_running_stats(
                epoch_diag_stats,
                _loss_total=loss,
                _loss_labeled=loss_l,
                _loss_unlabeled=loss_u,
                loss_l_main=loss_l_main,
                loss_l_aux=loss_l_aux,
                loss_l_boundary=loss_l_boundary,
                loss_u_main=loss_u_main,
                loss_u_aux=loss_u_aux,
                loss_calibration=loss_calibration,
                loss_edge=loss_edge,
                loss_sor=loss_sor,
                loss_teacher=loss_teacher,
                loss_adv=loss_adv,
                teacher_weight=teacher_weight,
                unlabeled_trust=unlabeled_trust,
                scd_ramp=scd_ramp,
                reliability_mean=reliability_mean,
                reliability_high=reliability_high,
                pseudo_fg=pseudo_fg,
                pseudo_pos=pseudo_pos,
                student_fg=student_fg,
                teacher_fg=teacher_fg,
                main_aux_gap=main_aux_gap,
                student_teacher_gap=student_teacher_gap,
                scd_real_prob=scd_real_prob,
                scd_fake_prob=scd_fake_prob,
                scd_d_loss=scd_d_loss,
                psfh_ps_area_gt=psfh_ps_area_gt,
                psfh_fh_area_gt=psfh_fh_area_gt,
                psfh_ps_area_student=psfh_ps_area_student,
                psfh_fh_area_student=psfh_fh_area_student,
                psfh_ps_area_teacher=psfh_ps_area_teacher,
                psfh_fh_area_teacher=psfh_fh_area_teacher,
                psfh_ps_dsc_sup=psfh_ps_dsc_sup,
                psfh_fh_dsc_sup=psfh_fh_dsc_sup,
                psfh_critic_ps_ratio=psfh_critic_ps_ratio,
                gp_grad_norm=gp_grad_norm_diag,
                gp_value=gp_value_diag,
            )
            epoch_loss_total += epoch_diag_stats.pop('_loss_total', 0.0)
            epoch_labeled_total += epoch_diag_stats.pop('_loss_labeled', 0.0)
            epoch_unlabeled_total += epoch_diag_stats.pop('_loss_unlabeled', 0.0)
            if signal_state is not None:
                signal_complete = update_signal_state(
                    signal_state,
                    img_u.size(0),
                    loss,
                    loss_l,
                    loss_u,
                    loss_l_main=loss_l_main,
                    loss_l_aux=loss_l_aux,
                    loss_l_boundary=loss_l_boundary,
                    loss_u_main=loss_u_main,
                    loss_u_aux=loss_u_aux,
                    loss_calibration=loss_calibration,
                    loss_edge=loss_edge,
                    loss_sor=loss_sor,
                    loss_teacher=loss_teacher,
                    loss_adv=loss_adv,
                    teacher_weight=teacher_weight,
                    unlabeled_trust=unlabeled_trust,
                    scd_ramp=scd_ramp,
                    reliability_mean=reliability_mean,
                    reliability_high=reliability_high,
                    pseudo_fg=pseudo_fg,
                    pseudo_pos=pseudo_pos,
                    student_fg=student_fg,
                    teacher_fg=teacher_fg,
                    main_aux_gap=main_aux_gap,
                    student_teacher_gap=student_teacher_gap,
                    scd_real_prob=scd_real_prob,
                    scd_fake_prob=scd_fake_prob,
                    scd_d_loss=scd_d_loss,
                )
            else:
                signal_complete = False

            if should_log_batch(batch_id, total_batch):
                log_semi_train_batch(
                    epoch,
                    args.nEpoch,
                    batch_id,
                    total_batch,
                    time.perf_counter() - batch_start,
                    lr,
                    loss,
                    loss_l,
                    loss_u,
                    loss_u_main,
                    loss_u_aux,
                    loss_calibration,
                    loss_adv,
                )
                log_semi_diagnostics_batch(
                    epoch,
                    args.nEpoch,
                    batch_id,
                    total_batch,
                    loss_l_main,
                    loss_l_aux,
                    loss_l_boundary,
                    teacher_weight,
                    unlabeled_trust,
                    scd_ramp,
                    reliability_mean,
                    reliability_high,
                    pseudo_fg,
                    pseudo_pos,
                    student_fg,
                    teacher_fg,
                    main_aux_gap,
                    student_teacher_gap,
                    scd_real_prob,
                    scd_fake_prob,
                    scd_d_loss,
                )
            if signal_complete:
                if is_main_process():
                    signal_summary = summarize_signal_state(signal_state)
                    val_f1 = 0.0
                    val_dice = 0.0
                    if valid_sign:
                        model.eval()
                        recall, specificity, precision, F1, F2, \
                            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice, *_, table_metrics = evaluate(
                                model,
                                valid_dataloader,
                                val_total_batch,
                                tta=args.eval_tta,
                                **eval_postprocess_kwargs(),
                            )
                        log_info("[Signal] Validation snapshot:")
                        log_info(
                            'recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f'
                            % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice)
                        )
                        log_validation_details(table_metrics)
                        val_f1 = F1
                        val_dice = dice
                    emit_signal_result(epoch + 1, signal_summary, val_f1=val_f1, val_dice=val_dice)
                if args.ddp:
                    torch.distributed.destroy_process_group()
                return
        if total_batch > 0:
            log_info(
                "[TrainSemi][Epoch {}/{} Summary] avg_loss={:.4f} avg_labeled={:.4f} avg_unlabeled={:.4f}".format(
                    epoch + 1,
                    args.nEpoch,
                    epoch_loss_total / total_batch,
                    epoch_labeled_total / total_batch,
                    epoch_unlabeled_total / total_batch,
                )
            )
            log_info(
                "[TrainSemiDiag][Epoch {}/{} Summary] avg_l_main={:.4f} avg_l_aux={:.4f} avg_l_boundary={:.4f} "
                "avg_u_main={:.4f} avg_u_aux={:.4f} avg_u_boundary={:.4f} avg_shape={:.4f} "
                "teacher_w={:.4f} trust={:.4f} scd_ramp={:.4f}".format(
                    epoch + 1,
                    args.nEpoch,
                    epoch_diag_stats.get('loss_l_main', 0.0) / total_batch,
                    epoch_diag_stats.get('loss_l_aux', 0.0) / total_batch,
                    epoch_diag_stats.get('loss_l_boundary', 0.0) / total_batch,
                    epoch_diag_stats.get('loss_u_main', 0.0) / total_batch,
                    epoch_diag_stats.get('loss_u_aux', 0.0) / total_batch,
                    epoch_diag_stats.get('loss_calibration', 0.0) / total_batch,
                    epoch_diag_stats.get('loss_adv', 0.0) / total_batch,
                    epoch_diag_stats.get('teacher_weight', 0.0) / total_batch,
                    epoch_diag_stats.get('unlabeled_trust', 0.0) / total_batch,
                    epoch_diag_stats.get('scd_ramp', 0.0) / total_batch,
                )
            )
            log_info(
                "[TrainSemiDiag][Epoch {}/{} Summary] pseudo_fg={:.4f} pseudo_pos={:.4f} student_fg={:.4f} "
                "teacher_fg={:.4f} rel_mean={:.4f} rel_hi={:.4f} main_aux_gap={:.4f} st_gap={:.4f} "
                "scd_real={:.4f} scd_fake={:.4f} scd_d={:.4f}".format(
                    epoch + 1,
                    args.nEpoch,
                    epoch_diag_stats.get('pseudo_fg', 0.0) / total_batch,
                    epoch_diag_stats.get('pseudo_pos', 0.0) / total_batch,
                    epoch_diag_stats.get('student_fg', 0.0) / total_batch,
                    epoch_diag_stats.get('teacher_fg', 0.0) / total_batch,
                    epoch_diag_stats.get('reliability_mean', 0.0) / total_batch,
                    epoch_diag_stats.get('reliability_high', 0.0) / total_batch,
                    epoch_diag_stats.get('main_aux_gap', 0.0) / total_batch,
                    epoch_diag_stats.get('student_teacher_gap', 0.0) / total_batch,
                    epoch_diag_stats.get('scd_real_prob', 0.0) / total_batch,
                    epoch_diag_stats.get('scd_fake_prob', 0.0) / total_batch,
                    epoch_diag_stats.get('scd_d_loss', 0.0) / total_batch,
                )
            )
            if has_psfh_slots:
                # Per-class diagnostics that target the PS underperformance:
                #   *_area_*  : class area fractions — does student match GT density?
                #   psfh_*_dsc_sup : labeled supervised DSC — is the labeled loss learning PS?
                #   critic_ps_ratio: how much PS contributes to the merged shape critic input
                #                    (< 0.15 means PS is invisible to DSR)
                #   gp_grad_norm   : WGAN-GP norm; should be ≈ 1.0; large deviation = Lipschitz violation
                log_info(
                    "[TrainSemiPSFH][Epoch {}/{} Summary] "
                    "ps_gt={:.4f} ps_stu={:.4f} ps_tea={:.4f}  fh_gt={:.4f} fh_stu={:.4f} fh_tea={:.4f}  "
                    "sup_dsc_ps={:.4f} sup_dsc_fh={:.4f}  critic_ps_ratio={:.4f}  gp_grad_norm={:.4f} gp_value={:.4f}".format(
                        epoch + 1,
                        args.nEpoch,
                        epoch_diag_stats.get('psfh_ps_area_gt', 0.0) / total_batch,
                        epoch_diag_stats.get('psfh_ps_area_student', 0.0) / total_batch,
                        epoch_diag_stats.get('psfh_ps_area_teacher', 0.0) / total_batch,
                        epoch_diag_stats.get('psfh_fh_area_gt', 0.0) / total_batch,
                        epoch_diag_stats.get('psfh_fh_area_student', 0.0) / total_batch,
                        epoch_diag_stats.get('psfh_fh_area_teacher', 0.0) / total_batch,
                        epoch_diag_stats.get('psfh_ps_dsc_sup', 0.0) / total_batch,
                        epoch_diag_stats.get('psfh_fh_dsc_sup', 0.0) / total_batch,
                        epoch_diag_stats.get('psfh_critic_ps_ratio', 0.0) / total_batch,
                        epoch_diag_stats.get('gp_grad_norm', 0.0) / total_batch,
                        epoch_diag_stats.get('gp_value', 0.0) / total_batch,
                    )
                )
        model.eval()
        if valid_sign and signal_state is None and is_main_process() and should_run_validation(epoch):
            recall, specificity, precision, F1, F2, \
                ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice, *_, table_metrics = evaluate(
                    model,
                    valid_dataloader,
                    val_total_batch,
                    tta=args.eval_tta,
                    **eval_postprocess_kwargs(),
                )
            save_output = False
            if dice > best_dice:
                best_dice = dice
                best_dice_epoch = epoch + 1
                save_output = True
                torch.save(get_model_state_dict(model), os.path.join(args.checkpoint_root, args.ckpt_name, "best_dice.pth"))
                torch.save(get_model_state_dict(model_cps), os.path.join(teacher_ckpt_dir, "best_dice.pth"))
            log_info("Best Dice:: {}".format(best_dice))

            if save_output:
                result_dir = get_result_dir()
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir)
                os.makedirs(result_dir)
                valid_postprocess_kwargs = eval_postprocess_kwargs()
                write_eval_manifest('valid', valid_dataloader, valid_postprocess_kwargs)
                evaluate(
                    model,
                    valid_dataloader,
                    val_total_batch,
                    save_best=True,
                    tta=args.save_best_tta,
                    **valid_postprocess_kwargs,
                    **eval_artifact_kwargs('valid'),
                )

            log_info("Valid Result:")
            log_info(
                'recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice))
            log_validation_details(table_metrics)

            if (F1 > F1_best):
                F1_best = F1
                best_f1_epoch = epoch + 1
                torch.save(get_model_state_dict(model), os.path.join(args.checkpoint_root, args.ckpt_name, "best.pth"))
                torch.save(get_model_state_dict(model_cps), os.path.join(teacher_ckpt_dir, "best.pth"))
            elif (F1 > F1_second_best):
                F1_second_best = F1
                torch.save(get_model_state_dict(model), os.path.join(args.checkpoint_root, args.ckpt_name, "second_best.pth"))
            elif (F1 > F1_third_best):
                F1_third_best = F1
                torch.save(get_model_state_dict(model), os.path.join(args.checkpoint_root, args.ckpt_name, "third_best.pth"))
        if args.ddp:
            torch.distributed.barrier()
    _signal_emitted = False
    if signal_state is not None and signal_state['measured_steps'] > 0 and is_main_process():
        val_f1 = 0.0
        val_dice = 0.0
        if valid_sign:
            recall, specificity, precision, F1, F2, \
                ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice, *_, table_metrics = evaluate(
                    model,
                    valid_dataloader,
                    val_total_batch,
                    tta=args.eval_tta,
                    **eval_postprocess_kwargs(),
                )
            log_info("[Signal] Final validation snapshot:")
            log_info(
                'recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f'
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice)
            )
            log_validation_details(table_metrics)
            val_f1 = F1
            val_dice = dice
        emit_signal_result(args.nEpoch, summarize_signal_state(signal_state), val_f1=val_f1, val_dice=val_dice)
        _signal_emitted = True
    if is_main_process() and not _signal_emitted:
        emit_training_result('semi', F1_best, best_f1_epoch, best_dice, best_dice_epoch)
    if args.ddp:
        torch.distributed.destroy_process_group()


def test():
    print('loading data......')
    test_data = build_dataset(args)
    log_dataset_splits('test', test=test_data)
    test_dataloader = build_dataloader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        use_distributed_sampler=False,
    )
    total_batch = math.ceil(len(test_data) / args.batch_size)
    model = build_model(args, load_ckpt=False)
    ckpt_role, ckpt_path, ckpt_stem = resolve_test_checkpoint()

    if ckpt_path is not None:
        print(f"[Test] Loading {ckpt_role} checkpoint '{ckpt_stem}.pth': {ckpt_path}")
        load_model_state_dict(model, torch.load(ckpt_path, map_location="cpu"))
    else:
        ckpt_dir = get_checkpoint_dir(ckpt_role)
        if args.load_ckpt is not None:
            print(
                f"[Test] WARNING: requested {ckpt_role} checkpoint '{args.load_ckpt}.pth' was not found in {ckpt_dir}; "
                "testing with randomly initialized weights."
            )
        else:
            print(
                f"[Test] WARNING: no {ckpt_role} checkpoint found in {ckpt_dir}; "
                "testing with randomly initialized weights."
            )

    model.eval()
    test_postprocess_kwargs = eval_postprocess_kwargs()
    if args.auto_tune_eval_postprocess:
        try:
            tune_data = build_eval_dataset(args, split='valid')
        except FileNotFoundError as exc:
            print(
                "[Test] WARNING: validation split is unavailable; skipping auto-tuned post-processing "
                "to avoid validation/test leakage. {}".format(exc)
            )
        else:
            tune_split_file = resolved_eval_split_file(tune_data)
            test_split_file = resolved_eval_split_file(test_data)
            tune_requested = getattr(tune_data, 'requested_split_name', None)
            tune_resolved = getattr(tune_data, 'resolved_split_name', None)
            if tune_requested and tune_resolved and tune_requested != tune_resolved:
                print(
                    "[Test] WARNING: validation split fell back to {}; skipping auto-tuned post-processing "
                    "to avoid validation/test leakage.".format(describe_eval_split(tune_data))
                )
            elif tune_split_file and test_split_file and tune_split_file == test_split_file:
                print(
                    "[Test] WARNING: validation split resolves to the same file as test ({}); skipping auto-tuned post-processing to avoid test-set leakage.".format(
                        describe_eval_split(tune_data),
                    )
                )
            elif len(tune_data) > 0:
                tune_dataloader = build_dataloader(
                    tune_data,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    use_distributed_sampler=False,
                )
                tune_total_batch = math.ceil(len(tune_data) / args.batch_size)
                test_postprocess_kwargs = tune_eval_postprocess(
                    model,
                    tune_dataloader,
                    tune_total_batch,
                    spacing=get_eval_spacing(),
                    split_name='valid',
                )

    write_eval_manifest('test', test_dataloader, test_postprocess_kwargs)

    recall, specificity, precision, F1, F2, \
        ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice, _, _, table_metrics = \
        evaluate(
            model,
            test_dataloader,
            total_batch,
            spacing=get_eval_spacing(),
            tta=args.eval_tta,
            **test_postprocess_kwargs,
            **eval_artifact_kwargs('test'),
        )
    metrics = {
        'recall': recall,
        'specificity': specificity,
        'precision': precision,
        'F1': F1,
        'F2': F2,
        'ACC_overall': ACC_overall,
        'IoU_poly': IoU_poly,
        'IoU_bg': IoU_bg,
        'IoU_mean': IoU_mean,
        'dice': dice,
    }

    metric_mode = surface_metric_mode()
    if metric_mode == "overall":
        # HC18 打印精简版
        print(
            'Valid Result: recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
            'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f, '
            'DSC_all: %.4f, Jacc_all: %.4f, HD95_all: %.4f, ASD_all: %.4f'
            % (recall, specificity, precision, F1, F2,
               ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice,
               table_metrics['DSC'], table_metrics['Jaccard'],
               table_metrics['HD95'], table_metrics['ASD'])
        )
    elif metric_mode == "split":
        # PSFH 打印全几何指标
        print(
            'Valid Result: recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
            'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f, '
            'DSC_all: %.4f, Jacc_all: %.4f, HD95_all: %.4f, ASD_all: %.4f, '
            'DSC_PS: %.4f, Jacc_PS: %.4f, HD95_PS: %.4f, ASD_PS: %.4f, '
            'DSC_FH: %.4f, Jacc_FH: %.4f, HD95_FH: %.4f, ASD_FH: %.4f'
            % (recall, specificity, precision, F1, F2,
               ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice,
               table_metrics['DSC'], table_metrics['Jaccard'], table_metrics['HD95'], table_metrics['ASD'],
               table_metrics['DSC_PS'], table_metrics['Jaccard_PS'], table_metrics['HD95_PS'], table_metrics['ASD_PS'],
               table_metrics['DSC_FH'], table_metrics['Jaccard_FH'], table_metrics['HD95_FH'], table_metrics['ASD_FH'])
        )
    else:
        print(
            'Valid Result: recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
            'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f'
            % (recall, specificity, precision, F1, F2,
               ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice)
        )

    log_validation_details(table_metrics)
    append_test_summary_markdown(metrics, table_metrics)


if __name__ == '__main__':
    configure_runtime()

    checkpoint_name = os.path.join(args.checkpoint_root, args.ckpt_name)
    if is_main_process() and not os.path.exists(checkpoint_name):
        os.makedirs(checkpoint_name)

    if args.gpu_ids:
        torch.cuda.set_device(args.primary_gpu)
        log_info(f"[INFO] Using GPU device ids: {args.gpu_ids}  ddp={args.ddp}  world_size={args.world_size}")
    set_random_seed()
    log_runtime_configuration()
    if args.manner == 'full':
        log_info('---{}-Seg Train---'.format(args.dataset))
        train()
    elif args.manner == 'semi':
        log_info('---{}-seg Semi-Train--'.format(args.dataset))
        train_semi()
    elif args.manner == 'test':
        log_info('---{}-Seg Test---'.format(args.dataset))
        test()
    log_info('Done')
