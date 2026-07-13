from __future__ import print_function

import argparse
from contextlib import nullcontext
import json
import os
import random
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.nn import functional as F

import models.dcgan as dcgan
import models.mlp as mlp
from data.BUSI import BUSIDataSet
from data.HC18 import HC18Dataset
from data.PSFH import PSFHDataset
from data.path_utils import DEFAULT_WORKSPACE_ROOT, normalize_root
from data.tn3k import tn3kDataSet

warnings.filterwarnings("ignore", category=UserWarning)
sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0)
DEFAULT_VISIBLE_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0


def log_info(message):
    print(message, flush=True)


def scalar_value(value):
    if torch.is_tensor(value):
        return float(value.detach().mean().item())
    return float(value)


def update_epoch_stats(stats, **kwargs):
    for key, value in kwargs.items():
        stats[key] = stats.get(key, 0.0) + scalar_value(value)


def compute_edges(tensor):
    kernel_x = sobel_x.to(device=tensor.device, dtype=tensor.dtype)
    kernel_y = sobel_y.to(device=tensor.device, dtype=tensor.dtype)
    # Sobel kernels expect a single input channel. For multi-channel inputs
    # (e.g. PS+FH stacked masks) run the convolution depthwise via `groups`,
    # so each channel keeps its own edge map.
    in_channels = tensor.size(1)
    if in_channels > 1:
        kernel_x = kernel_x.expand(in_channels, 1, -1, -1)
        kernel_y = kernel_y.expand(in_channels, 1, -1, -1)
        grad_x = F.conv2d(tensor, kernel_x, padding=1, groups=in_channels)
        grad_y = F.conv2d(tensor, kernel_y, padding=1, groups=in_channels)
    else:
        grad_x = F.conv2d(tensor, kernel_x, padding=1)
        grad_y = F.conv2d(tensor, kernel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)


def soft_dice_loss(prediction, target, eps=1e-6):
    prediction = prediction.float()
    target = target.float()
    dims = tuple(range(1, prediction.dim()))
    intersection = (prediction * target).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def dataloader_worker_init(_worker_id):
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    if hasattr(torch, 'set_num_interop_threads'):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    try:
        import cv2
        cv2.setNumThreads(1)
        if hasattr(cv2, 'ocl'):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def build_dataloader(dataset, opt):
    loader_kwargs = {
        'batch_size': opt.batchSize,
        'shuffle': True,
        'num_workers': int(opt.workers),
        'pin_memory': bool(opt.cuda),
    }
    if int(opt.workers) > 0:
        loader_kwargs['worker_init_fn'] = dataloader_worker_init
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def resolve_effective_precision(opt):
    requested = opt.precision.lower()
    if not opt.cuda:
        return 'fp32'

    bf16_supported = hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
    if requested == 'auto':
        # WGAN-style critic training is sensitive to coarse bf16 quantization.
        return 'fp32'
    if requested == 'bf16' and not bf16_supported:
        log_info('[WARN] bf16 requested but not supported on this CUDA device; falling back to fp32.')
        return 'fp32'
    if requested == 'bf16':
        log_info('[WARN] bf16 requested for GAN training; critic scores may become noticeably quantized.')
    return requested


def autocast_context(opt):
    if getattr(opt, 'effective_precision', 'fp32') == 'bf16':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def configure_runtime(opt):
    opt.effective_precision = resolve_effective_precision(opt)
    if opt.cuda and opt.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')


def build_dataset(opt):
    if opt.dataset == 'tn3k':
        return tn3kDataSet(opt.root, opt.expID, mode='train')
    if opt.dataset == 'busi':
        return BUSIDataSet(opt.root, opt.expID, mode='train')
    if opt.dataset == 'hc18':
        return HC18Dataset(opt.root, opt.expID, mode='train')
    if opt.dataset == 'psfh':
        return PSFHDataset(opt.root, opt.expID, mode='train')
    raise ValueError('Unsupported dataset: {}'.format(opt.dataset))


def maybe_resize(tensor, image_size):
    if tensor.shape[-2:] == (image_size, image_size):
        return tensor
    return F.interpolate(tensor, size=(image_size, image_size), mode='bilinear', align_corners=False)


def prepare_real_batch(data, opt, device):
    img_real = maybe_resize(data['image'], opt.imageSize)
    raw_label = maybe_resize(data['label'], opt.imageSize)
    if opt.cuda:
        img_real = img_real.to(device, non_blocking=True)
        raw_label = raw_label.to(device, non_blocking=True)
    # PSFH loader packs [region, ps, fh] into a 3-channel label so the
    # shared single-label transform pipeline keeps the three channels in
    # geometric sync. Single-class loaders return a 1-channel label.
    if raw_label.dim() == 4 and raw_label.size(1) == 3:
        real_mask = raw_label[:, 0:1]
        real_mask_ps = raw_label[:, 1:2]
        real_mask_fh = raw_label[:, 2:3]
    else:
        real_mask = raw_label
        # Treat the single foreground class as the PS slot and zero-pad FH so
        # the 5-channel (image+ps+fh) input is well-defined for all datasets.
        real_mask_ps = real_mask
        real_mask_fh = torch.zeros_like(real_mask)
    return img_real, real_mask, real_mask_ps, real_mask_fh


def normalize_mask_for_gan(mask):
    return mask.float().mul(2.0).sub(1.0)


def denormalize_mask_from_gan(mask):
    return mask.float().add(1.0).mul(0.5).clamp(0.0, 1.0)


def gradient_penalty(netD, real_data, fake_data, device, lambda_gp=10.0):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interpolates = netD(interpolates)
    grad_outputs = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp


def current_diters(opt, gen_iterations):
    if opt.warmup_gen_iterations > 0 and gen_iterations < opt.warmup_gen_iterations:
        return opt.warmup_diters
    if opt.extra_diters_every > 0 and gen_iterations > 0 and gen_iterations % opt.extra_diters_every == 0:
        return opt.warmup_diters
    return opt.Diters


def should_log_step(opt, gen_iterations, is_epoch_tail):
    if gen_iterations <= 1:
        return True
    if is_epoch_tail:
        return True
    return opt.log_interval > 0 and gen_iterations % opt.log_interval == 0


def save_checkpoints(epoch, opt, netG, netD):
    if opt.save_every <= 0:
        should_save = epoch == opt.niter - 1
    else:
        should_save = (epoch % opt.save_every == 0) or (epoch == opt.niter - 1)
    if not should_save:
        return
    torch.save(netG.state_dict(), os.path.join(opt.experiment, 'netG_epoch_{}.pth'.format(epoch)))
    torch.save(netD.state_dict(), os.path.join(opt.experiment, 'netD_epoch_{}.pth'.format(epoch)))


def maybe_finish_benchmark(opt, measured_steps, measured_samples, start_time):
    if opt.benchmark_steps <= 0 or measured_steps < opt.benchmark_steps or start_time is None:
        return
    if opt.cuda:
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    log_info(
        "[AUTOTUNE_RESULT] stage=gan batch_size={} workers={} steps={} samples={} elapsed_sec={:.6f} samples_per_sec={:.6f} batches_per_sec={:.6f}".format(
            opt.batchSize,
            opt.workers,
            measured_steps,
            measured_samples,
            elapsed,
            measured_samples / elapsed,
            measured_steps / elapsed,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataroot', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=2,
                        help='Number of class-aware mask channels generated by netG and consumed by netD. '
                             'Default 2 (PS slot + FH slot). Single-class datasets pad the FH channel with zeros.')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=3001, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='enable CUDA execution')
    parser.add_argument('--cpu', dest='cuda', action='store_false', help='force CPU execution')
    parser.add_argument('--ngpu', type=int, default=DEFAULT_VISIBLE_GPUS, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='First epoch index to run when continuing from netG/netD checkpoints.')
    parser.add_argument('--clamp_lower', type=float, default=-0.03)
    parser.add_argument('--clamp_upper', type=float, default=0.03)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--warmup_gen_iterations', type=int, default=5,
                        help='Use warmup_diters for the first N generator updates.')
    parser.add_argument('--warmup_diters', type=int, default=20,
                        help='Discriminator iterations used during warmup or periodic refreshes.')
    parser.add_argument('--extra_diters_every', type=int, default=0,
                        help='If > 0, periodically run warmup_diters every N generator updates.')
    parser.add_argument('--save_every', type=int, default=500, help='Save checkpoints every N epochs.')
    parser.add_argument('--log_interval', type=int, default=25, help='Print training logs every N generator updates.')
    parser.add_argument('--precision', type=str, default='auto', choices=['auto', 'fp32', 'bf16'],
                        help='Numerical precision for CUDA execution. "auto" resolves to fp32 for GAN stability.')
    parser.add_argument('--tf32', action='store_true',
                        help='Enable TF32 matmul/cuDNN kernels on Ampere/Hopper GPUs for higher throughput.')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default='result', help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--gradient_penalty', action='store_true',
                        help='Use gradient penalty (WGAN-GP) instead of weight clipping.')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='Gradient penalty coefficient for WGAN-GP.')
    parser.add_argument('--root', type=str, default=str(DEFAULT_WORKSPACE_ROOT),
                        help='Workspace root containing both the project directory and a sibling DATA directory.')
    parser.add_argument('--expID', type=int, default=1)
    parser.add_argument('--benchmark_steps', type=int, default=0,
                        help='If > 0, run a short throughput benchmark and exit after the measured steps.')
    parser.add_argument('--benchmark_warmup_steps', type=int, default=2,
                        help='Warmup generator steps excluded from benchmark timing.')
    parser.add_argument('--lambda_adv', type=float, default=0.5,
                        help='Generator adversarial loss weight.')
    parser.add_argument('--lambda_edge', type=float, default=1.0,
                        help='Generator edge reconstruction loss weight.')
    parser.add_argument('--lambda_fm', type=float, default=1.0,
                        help='Generator feature matching loss weight.')
    parser.add_argument('--lambda_mask', type=float, default=1.0,
                        help='Generator region reconstruction loss weight.')
    parser.add_argument('--manualSeed', type=int, default=None,
                        help='Optional fixed random seed for reproducible GAN/FBWA pretraining.')
    parser.set_defaults(cuda=torch.cuda.is_available())
    opt = parser.parse_args()
    opt.root = normalize_root(opt.root)
    opt.start_epoch = max(0, int(opt.start_epoch))

    if opt.experiment is None:
        opt.experiment = 'samples'
    os.makedirs(opt.experiment, exist_ok=True)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    available_gpu_count = torch.cuda.device_count()
    if available_gpu_count > 0 and not opt.cuda:
        log_info("WARNING: CUDA devices are available but CUDA execution was disabled explicitly.")
    if opt.cuda and available_gpu_count == 0:
        log_info("[WARN] CUDA was requested but no CUDA devices are available; falling back to CPU.")
    opt.cuda = bool(opt.cuda and available_gpu_count > 0)
    if opt.cuda:
        opt.ngpu = max(1, min(int(opt.ngpu), available_gpu_count))
    else:
        opt.ngpu = 0
    device = torch.device('cuda' if opt.cuda else 'cpu')
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = opt.cuda
    configure_runtime(opt)

    log_info(str(opt))
    log_info("Random Seed: {}".format(opt.manualSeed))
    log_info(
        "[INFO] Runtime config: precision={} requested_precision={} tf32={} batch_size={} workers={} niter={} Diters={} warmup_gen_iterations={} warmup_diters={} extra_diters_every={} lambda_adv={} lambda_edge={} lambda_fm={} lambda_mask={}".format(
            opt.effective_precision,
            opt.precision,
            'on' if opt.tf32 else 'off',
            opt.batchSize,
            opt.workers,
            opt.niter,
            opt.Diters,
            opt.warmup_gen_iterations,
            opt.warmup_diters,
            opt.extra_diters_every,
            opt.lambda_adv,
            opt.lambda_edge,
            opt.lambda_fm,
            opt.lambda_mask,
        )
    )

    dataset = build_dataset(opt)
    dataloader = build_dataloader(dataset, opt)

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    n_extra_layers = int(opt.n_extra_layers)

    generator_config = {
        "imageSize": opt.imageSize,
        "nz": nz,
        "nc": nc,
        "ngf": ngf,
        "ngpu": ngpu,
        "n_extra_layers": n_extra_layers,
        "noBN": opt.noBN,
        "mlp_G": opt.mlp_G,
    }
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config) + "\n")

    def weights_init(module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif 'Norm' in classname:
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.normal_(1.0, 0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG, map_location='cpu'))
        log_info("[INFO] Loaded netG checkpoint: {}".format(opt.netG))

    if opt.mlp_D:
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    elif opt.noBN:
        netD = dcgan.DCGAN_D_nobn(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD, map_location='cpu'))
        log_info("[INFO] Loaded netD checkpoint: {}".format(opt.netD))

    netD = netD.to(device)
    netG = netG.to(device)
    if opt.cuda:
        log_info("using cuda ===================================== ")

    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    noise = torch.empty(opt.batchSize, nz, 1, 1, device=device)
    if opt.start_epoch >= opt.niter:
        log_info("[INFO] start_epoch={} >= niter={}; nothing to train.".format(opt.start_epoch, opt.niter))
        raise SystemExit(0)
    if opt.start_epoch > 0:
        log_info("[INFO] Resuming GAN loop from epoch {} / {}".format(opt.start_epoch, opt.niter))

    gen_iterations = opt.start_epoch
    benchmark_start_time = None
    benchmark_measured_steps = 0
    benchmark_measured_samples = 0
    for epoch in range(opt.start_epoch, opt.niter):
        data_iter = iter(dataloader)
        i = 0
        epoch_stats = {}
        epoch_gen_steps = 0
        while i < len(dataloader):
            for parameter in netD.parameters():
                parameter.requires_grad = True

            Diters = current_diters(opt, gen_iterations)
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1
                optimizerD.zero_grad()

                data = next(data_iter)
                i += 1

                img_real, real_mask, real_mask_ps, real_mask_fh = prepare_real_batch(data, opt, device)
                real_mask_ps_gan = normalize_mask_for_gan(real_mask_ps)
                real_mask_fh_gan = normalize_mask_for_gan(real_mask_fh)
                input_real = torch.cat([img_real, real_mask_ps_gan, real_mask_fh_gan], dim=1)

                curr_batch_size = img_real.size(0)
                noise.resize_(curr_batch_size, nz, 1, 1).normal_(0, 1)

                with autocast_context(opt):
                    errD_real = netD(input_real)
                    fake = netG(noise)
                    fake = maybe_resize(fake, opt.imageSize)
                    # fake is in [-1,1] from tanh, already nc=2 channels.
                    input_fake = torch.cat([img_real, fake.detach()], dim=1)
                    errD_fake = netD(input_fake)

                errD = errD_fake.float().mean() - errD_real.float().mean()
                if opt.gradient_penalty:
                    gp = gradient_penalty(netD, input_real.detach(), input_fake.detach(), device, opt.lambda_gp)
                    (errD + gp).backward()
                else:
                    errD.backward()
                optimizerD.step()
                if not opt.gradient_penalty:
                    for parameter in netD.parameters():
                        parameter.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            for parameter in netD.parameters():
                parameter.requires_grad = False

            optimizerG.zero_grad()
            noise.resize_(curr_batch_size, nz, 1, 1).normal_(0, 1)
            with autocast_context(opt):
                fake = netG(noise)
                fake = maybe_resize(fake, opt.imageSize)
                input_fake = torch.cat([img_real, fake], dim=1)
                errG_main, fake_features = netD(input_fake, return_features=True, detach_features=False)
                with torch.no_grad():
                    _, real_features = netD(input_real, return_features=True, detach_features=True)

            fake_mask_unit = denormalize_mask_from_gan(fake)
            # Use per-class targets so each channel matches its class. The
            # FH channel for single-class datasets is zero-filled, which
            # supervises netG to also output a near-zero FH slot for those
            # datasets — keeping the same head usable across all four.
            real_target_2ch = torch.cat([real_mask_ps.float(), real_mask_fh.float()], dim=1)
            edge_loss = F.l1_loss(compute_edges(real_target_2ch), compute_edges(fake_mask_unit))
            fm_real_features = real_features[:-1] if len(real_features) > 1 else real_features
            fm_fake_features = fake_features[:-1] if len(fake_features) > 1 else fake_features
            if fm_fake_features:
                # Use intermediate critic activations only; matching the final score map makes FM dominate the adversarial objective.
                feature_matching_loss = sum(
                    F.l1_loss(fake_feature.float(), real_feature.float())
                    for real_feature, fake_feature in zip(fm_real_features, fm_fake_features)
                ) / len(fm_fake_features)
            else:
                feature_matching_loss = torch.tensor(0.0, device=device)
            mask_recon_loss = 0.5 * (
                F.binary_cross_entropy(fake_mask_unit.clamp(1e-6, 1.0 - 1e-6), real_target_2ch) +
                soft_dice_loss(fake_mask_unit, real_target_2ch)
            )

            generator_adv = -1.0 * errG_main.float().mean()
            critic_gap = errD_real.float().mean() - errD_fake.float().mean()
            real_fg = real_target_2ch.mean()
            fake_fg = fake_mask_unit.mean()
            real_pos = (real_target_2ch > 0.5).float().mean()
            fake_pos = (fake_mask_unit > 0.5).float().mean()
            fake_std = fake_mask_unit.float().std(unbiased=False)

            errG_total = (
                (opt.lambda_adv * generator_adv) +
                (opt.lambda_edge * edge_loss) +
                (opt.lambda_fm * feature_matching_loss) +
                (opt.lambda_mask * mask_recon_loss)
            )
            errG_total.backward()
            optimizerG.step()
            epoch_gen_steps += 1
            update_epoch_stats(
                epoch_stats,
                loss_d=errD,
                loss_g=errG_total,
                d_real=errD_real,
                d_fake=errD_fake,
                critic_gap=critic_gap,
                g_adv=generator_adv,
                edge=edge_loss,
                fm=feature_matching_loss,
                mask=mask_recon_loss,
                real_fg=real_fg,
                fake_fg=fake_fg,
                real_pos=real_pos,
                fake_pos=fake_pos,
                fake_std=fake_std,
            )

            gen_iterations += 1
            if opt.benchmark_steps > 0:
                if gen_iterations == opt.benchmark_warmup_steps + 1:
                    if opt.cuda:
                        torch.cuda.synchronize()
                    benchmark_start_time = time.perf_counter()
                if gen_iterations > opt.benchmark_warmup_steps:
                    benchmark_measured_steps += 1
                    benchmark_measured_samples += curr_batch_size
                    maybe_finish_benchmark(
                        opt,
                        benchmark_measured_steps,
                        benchmark_measured_samples,
                        benchmark_start_time,
                    )
            if should_log_step(opt, gen_iterations, i >= len(dataloader)):
                log_info(
                    '[{}/{}][{}/{}][{}] Loss_D: {:.6f} Loss_G: {:.6f} Loss_D_real: {:.6f} Loss_D_fake {:.6f} '
                    'CriticGap: {:.6f} G_adv: {:.6f} Edge: {:.6f} FM: {:.6f} Mask: {:.6f} '
                    'RealFG: {:.6f} FakeFG: {:.6f} RealPos: {:.6f} FakePos: {:.6f} FakeStd: {:.6f}'.format(
                        epoch,
                        opt.niter,
                        i,
                        len(dataloader),
                        gen_iterations,
                        scalar_value(errD),
                        scalar_value(errG_total),
                        scalar_value(errD_real),
                        scalar_value(errD_fake),
                        scalar_value(critic_gap),
                        scalar_value(generator_adv),
                        scalar_value(edge_loss),
                        scalar_value(feature_matching_loss),
                        scalar_value(mask_recon_loss),
                        scalar_value(real_fg),
                        scalar_value(fake_fg),
                        scalar_value(real_pos),
                        scalar_value(fake_pos),
                        scalar_value(fake_std),
                    )
                )

        if epoch_gen_steps > 0 and epoch_stats:
            denom = float(epoch_gen_steps)
            log_info(
                "[GAN][Epoch {}/{} Summary] avg_loss_d={:.6f} avg_loss_g={:.6f} avg_d_real={:.6f} avg_d_fake={:.6f} "
                "avg_critic_gap={:.6f} avg_g_adv={:.6f} avg_edge={:.6f} avg_fm={:.6f} avg_mask={:.6f} "
                "avg_real_fg={:.6f} avg_fake_fg={:.6f} avg_real_pos={:.6f} avg_fake_pos={:.6f} avg_fake_std={:.6f}".format(
                    epoch + 1,
                    opt.niter,
                    epoch_stats.get('loss_d', 0.0) / denom,
                    epoch_stats.get('loss_g', 0.0) / denom,
                    epoch_stats.get('d_real', 0.0) / denom,
                    epoch_stats.get('d_fake', 0.0) / denom,
                    epoch_stats.get('critic_gap', 0.0) / denom,
                    epoch_stats.get('g_adv', 0.0) / denom,
                    epoch_stats.get('edge', 0.0) / denom,
                    epoch_stats.get('fm', 0.0) / denom,
                    epoch_stats.get('mask', 0.0) / denom,
                    epoch_stats.get('real_fg', 0.0) / denom,
                    epoch_stats.get('fake_fg', 0.0) / denom,
                    epoch_stats.get('real_pos', 0.0) / denom,
                    epoch_stats.get('fake_pos', 0.0) / denom,
                    epoch_stats.get('fake_std', 0.0) / denom,
                )
            )
        save_checkpoints(epoch, opt, netG, netD)
