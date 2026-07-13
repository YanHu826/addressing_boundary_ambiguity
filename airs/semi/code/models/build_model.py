import os

import torch

import models


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def get_model_state_dict(model):
    return unwrap_model(model).state_dict()


def load_model_state_dict(model, checkpoint):
    base_model = unwrap_model(model)
    model_dict = base_model.state_dict()
    new_dict = {}

    for key, value in checkpoint.items():
        normalized_key = key[7:] if key.startswith('module.') else key
        if normalized_key in model_dict and model_dict[normalized_key].shape == value.shape:
            new_dict[normalized_key] = value

    model_dict.update(new_dict)
    base_model.load_state_dict(model_dict)


def configure_cuda_model(model, args, wrap_distributed=True):
    if not args.gpu_ids:
        return model

    torch.cuda.set_device(args.primary_gpu)
    model = model.cuda(args.primary_gpu)
    torch.backends.cudnn.benchmark = True

    if not wrap_distributed:
        return model

    if getattr(args, 'ddp', False):
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.primary_gpu],
            output_device=args.primary_gpu,
            find_unused_parameters=True,
        )
    elif args.use_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids, output_device=args.primary_gpu)

    return model


def build_model(args, wrap_distributed=True, load_ckpt=True):
    model = getattr(models, args.model)(args)
    model = configure_cuda_model(model, args, wrap_distributed=wrap_distributed)
    if load_ckpt and args.load_ckpt is not None:
        load_ckpt_path = os.path.join(args.checkpoint_root, str(args.ckpt_name), args.load_ckpt + '.pth')
        print(load_ckpt_path)
        assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
        print('Loading checkpoint......')
        checkpoint = torch.load(load_ckpt_path, map_location='cpu')
        load_model_state_dict(model, checkpoint)
        print('Done')

    return model
