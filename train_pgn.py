import argparse
import logging
import os
import pickle
import shutil
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from archs.deep_image_prior.skip import skip, default_skip_params
from archs.perceptual_loss import PerceptualLoss, PerceptualLossDeep
from dataloader import get_dataloaders
from image2stylegan.decoder_from_latents import DecoderFromLatents
from image2stylegan.model import StyledGenerator
from gen_envs import AutoencoderEnv
from losses import ScaledMSELoss, VggOneStepLoss, VggGradLoss, symmetric_jacobian_loss, total_variation_loss
from pgn import Pgn
from ssim import SSIM
from utils import (
    VideoWriter, MovingAverage, Timer,
    Normalizer, set_random_seeds, get_grad_norms, make_frame, torch_batch_to_numpy, percentile,
    l1_loss_batchwise, mse_loss_batchwise,
    ProcessGroup, DistributedSummaryWrapper
)

try:
    from apex import amp
except ImportError:
    amp = None


def get_multiple_percentiles(t, percentiles=(0, 1, 50, 99, 100)):
    return {
        q: percentile(t, q)
        for q in percentiles
    }


def get_grad_metrics(grad_pred, grad_true=None):
    def batchwise_flatten(tensor):
        b = tensor.shape[0]
        return tensor.reshape(b, -1)

    def l2_norm(a):
        return torch.norm(batchwise_flatten(a), p=2, dim=1)

    metrics = {}

    metrics['grads_pred_norm'] = l2_norm(grad_pred)

    metrics.update({
        f'grads_pred_q{q}': p
        for (q, p) in get_multiple_percentiles(grad_pred).items()
    })

    if grad_true is not None:
        metrics['grads_true_norm'] = l2_norm(grad_true)

        inf = torch.tensor(np.inf, device=metrics['grads_true_norm'].device, dtype=metrics['grads_true_norm'].dtype)

        metrics['grads_norm_ratio'] = torch.where(
            metrics['grads_true_norm'] != 0,
            metrics['grads_pred_norm'] / metrics['grads_true_norm'],
            inf,
        )

        metrics['grads_diff_norm'] = l2_norm(grad_pred - grad_true)
        metrics['loss_norm'] = torch.where(
            metrics['grads_true_norm'] != 0,
            metrics['grads_diff_norm'] / metrics['grads_true_norm'],
            inf,
        )

        metrics['cosine_similarity'] = F.cosine_similarity(
            batchwise_flatten(grad_pred),
            batchwise_flatten(grad_true),
        )

        metrics.update({
            f'grads_true_q{q}': p
            for (q, p) in get_multiple_percentiles(grad_true).items()
        })

    return metrics


def forward_backward(func, x):
    x = x.detach().requires_grad_()
    with torch.enable_grad():
        y = func(x)
        y.sum(dim=0).backward()
    return x.grad, y


def setup_dirs(args):
    args.run_dir.mkdir(exist_ok=True, parents=True)
    if next(args.run_dir.iterdir(), None) is not None:
        logging.warning(f'Run dir {args.run_dir} is not empty!')

    tb_dir = args.run_dir / 'tb'
    log_dir = args.run_dir / 'logs'
    model_dir = args.run_dir / 'models'

    tb_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    return tb_dir, log_dir, model_dir


def setup_args(args):
    if args.amp and amp is None:
        raise RuntimeError('--amp is provided but apex.amp is unavailable')

    if args.debug:
        args.tqdm = True
        args.batches_per_train_epoch = 8
        args.valid_num_iter = 100
        args.save_each = 1

    if args.valid_only:
        args.tqdm = True
        args.workers = 0

    if args.pgn_loss_grad_coef == 0 and args.pgn_backbone_to_grad != 'proxy':
        logging.warning('Grad coefficient is zero, forcing PGN proxy mode')
        args.pgn_backbone_to_grad = 'proxy'

    num_autoencoders = len(args.ae_max_layer)

    if args.resume_dir is not None:
        assert args.resume_epoch is not None

        args.start_epoch = args.resume_epoch + 1
        args.pgn_checkpoint = args.resume_dir / f'{args.resume_epoch}.pth'
        args.pgn_checkpoint_opt = args.resume_dir / f'{args.resume_epoch}_opt.pth'
        args.ae_checkpoints = [
            args.resume_dir / f'{args.resume_epoch}_ae_{i}.pth' for i in range(num_autoencoders)]
        args.ae_checkpoints_opt = [
            args.resume_dir / f'{args.resume_epoch}_ae_{i}_opt.pth' for i in range(num_autoencoders)]

    def parse_checkpoint_list(checkpoints):
        if checkpoints is None:
            return [None] * num_autoencoders
        else:
            return [
                (None if ckpt_path == Path('None') else ckpt_path)
                for ckpt_path in checkpoints
            ]

    args.ae_checkpoints = parse_checkpoint_list(args.ae_checkpoints)
    args.ae_checkpoints_opt = parse_checkpoint_list(args.ae_checkpoints_opt)
    assert len(args.ae_max_layer) == len(args.ae_checkpoints) == len(args.ae_checkpoints_opt)

    args.distributed = not args.no_distributed

    if args.distributed:
        if args.world_size is None:
            if args.dist_init_method == "env://":
                args.world_size = int(os.environ["WORLD_SIZE"])
                logging.debug(f'Read WORLD_SIZE={args.world_size} from environment variable')
            else:
                args.world_size = torch.cuda.device_count()
                logging.debug(f'Detected {args.world_size} CUDA devices')

        if args.world_size > 1:
            logging.warning(f'Disabling tqdm in distributed mode with world_size={args.world_size}')
            args.tqdm = False


def load_networks(args, pgn_device, ae_device, vgg_device):
    logging.info(f'Creating networks on devices: PGN->{pgn_device}, AEs->{ae_device}, VGG->{vgg_device}')

    normalizer = Normalizer.make('vgg').to(pgn_device)

    backbone_to_grad_params = {
        'type': args.pgn_proxy_type,
    }
    if args.pgn_proxy_type == 'raw':
        backbone_to_grad_params[args.pgn_proxy_type] = {}
    elif args.pgn_proxy_type == 'sigmoid':
        backbone_to_grad_params[args.pgn_proxy_type] = {
            'scale': args.pgn_proxy_sigmoid_scale,
        }
    elif args.pgn_proxy_type == 'warped_target':
        backbone_to_grad_params[args.pgn_proxy_type] = {
            'scale': args.pgn_proxy_warped_target_scale,
            'additive': args.pgn_proxy_warped_target_additive,
            'downscale_by': args.pgn_proxy_warped_target_downscale_by,
            'additive_scale': args.pgn_proxy_warped_target_additive_scale,
        }
    else:
        assert False

    if args.pgn_backbone_to_grad == 'direct':
        backbone_to_grad_params.update({
            'out_scale': args.pgn_out_scale,
            'grad_scale': args.pgn_proxy_grad_scale,
        })
    elif args.pgn_backbone_to_grad == 'proxy':
        backbone_to_grad_params.update({
            'grad_type': args.pgn_proxy_grad_type,
            'grad_scale': args.pgn_proxy_grad_scale,
        })
    else:
        assert False

    if args.pgn_proxy_type == 'warped_target':
        if args.pgn_proxy_warped_target_additive:
            pgn_out_channels = 5
        else:
            pgn_out_channels = 2
    else:
        pgn_out_channels = 3

    if args.pgn_arch == 'unet':
        backbone_params = {
            'block_type': args.block_type,
            'conv_type': args.conv_type,
            'norm_type': args.norm_type,
            'down_channels': args.unet_down_channels,
            'up_channels': args.unet_up_channels,
            'predict_value': args.pgn_predict_value,
            'out_channels': pgn_out_channels,
        }
    elif args.pgn_arch == 'resnet':
        assert not args.pgn_predict_value
        backbone_params = {
            'down_channels': args.resnet_down_channels,
            'up_channels': args.resnet_up_channels,
            'num_blocks': args.resnet_num_blocks,
            'output_nc': pgn_out_channels,
        }
    else:
        assert False

    pgn = Pgn(
        normalizer,
        backbone_type=args.pgn_arch, backbone_params=backbone_params,
        backbone_to_grad_type=args.pgn_backbone_to_grad, backbone_to_grad_params=backbone_to_grad_params,
        ignore_grad_scale_mismatch=args.pgn_ignore_grad_scale_mismatch,
        checkpoint_path=args.pgn_checkpoint)

    if args.ploss_type == 'full':
        ploss = PerceptualLoss(model=args.ploss_model)
    elif args.ploss_type == 'deep':
        use_bn = False
        if use_bn:
            feature_layer = 49
        else:
            feature_layer = 34
        ploss = PerceptualLossDeep(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=False, device=vgg_device)
    else:
        assert False

    autoencoders = [
        AutoencoderEnv(args.ae_block_type, args.ae_conv_type, max_layer, ckpt_path)
        for (max_layer, ckpt_path)
        in zip(args.ae_max_layer, args.ae_checkpoints)
    ]

    pgn = pgn.to(pgn_device)
    ploss = ploss.to(vgg_device)
    autoencoders = [ae.to(ae_device) for ae in autoencoders]

    if args.double:
        pgn.double()
        ploss.double()
        for ae in autoencoders:
            ae.double()

    for ae, opt_ckpt_path in zip(autoencoders, args.ae_checkpoints_opt):
        ae.init_opt(opt_ckpt_path)

    if args.pgn_optimizer == 'adam':
        opt_pgn = torch.optim.Adam(params=pgn.parameters())
    elif args.pgn_optimizer == 'sgd':
        opt_pgn = torch.optim.SGD(params=pgn.parameters(), lr=args.pgn_sgd_lr)
    else:
        assert False

    if args.pgn_checkpoint_opt is not None:
        logging.debug(f'Loading PGN optimizer checkpoint from {args.pgn_checkpoint_opt}')
        opt_pgn_state_dict = torch.load(args.pgn_checkpoint_opt, map_location='cpu')
        opt_pgn.load_state_dict(opt_pgn_state_dict)

    if args.amp:
        models = [pgn, *[ae.autoencoder for ae in autoencoders]]
        optimizers = [opt_pgn, *[ae.opt for ae in autoencoders]]
        models, optimizers = amp.initialize(models, optimizers, opt_level=args.amp_opt_level)
        pgn = models[0]
        opt_pgn = optimizers[0]
        for ae, autoencoder, opt in zip(autoencoders, models[1:], optimizers[1:]):
            ae.autoencoder = autoencoder
            ae.opt = opt

    for ae in autoencoders:
        ae.train()

    return normalizer, pgn, opt_pgn, autoencoders, ploss


def main(parser: argparse.ArgumentParser):
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)-15s] %(levelname)s: %(message)s')

    args = parser.parse_args()

    setup_args(args)
    tb_dir, log_dir, model_dir = setup_dirs(args)

    if args.distributed:
        logging.debug(f'Spawning {args.world_size} workers')
        mp.spawn(main_worker, nprocs=args.world_size, args=(args, tb_dir, log_dir, model_dir))
    else:
        main_worker(None, args, tb_dir, log_dir, model_dir)


def main_worker(rank: Optional[int], args, tb_dir: Path, log_dir: Path, model_dir: Path):
    set_random_seeds(args.seed)

    logging.basicConfig(
        level=logging.INFO if args.no_verbose else logging.DEBUG,
        format='[%(asctime)-15s' + (f' | {rank}' if args.distributed else '') + '] %(levelname)s: %(message)s'
    )
    # Prevent matplotlib from polluting logs with its initialization messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('stylegan').setLevel(logging.INFO)

    if args.distributed:
        assert rank is not None
        torch.cuda.set_device(rank)

        if args.pgn_device != 'cuda:0':
            logging.warning('Ignoring --pgn-device due to running in distributed multiprocessing mode.')
        args.pgn_device = f'cuda:{rank}'
        if args.ae_device != 'cuda:0':
            logging.warning('Ignoring --ae-device due to running in distributed multiprocessing mode.')
        args.ae_device = f'cuda:{rank}'
        if args.vgg_device != 'cuda:0':
            logging.warning('Ignoring --vgg-device due to running in distributed multiprocessing mode.')
        args.vgg_device = f'cuda:{rank}'

        args.batch_size = args.worker_batch_size * args.world_size
        args.workers = (args.workers + args.world_size - 1) // args.world_size
    else:
        args.batch_size = args.worker_batch_size

    pgn_device = torch.device(args.pgn_device)
    ae_device = torch.device(args.ae_device)
    vgg_device = torch.device(args.vgg_device)

    def use_true_grads(epoch):
        if args.pgn_backbone_to_grad == 'proxy':
            return False
        if epoch < args.warmup_epochs:
            return True
        else:
            if not args.no_alternate_gt_curr:
                return epoch % 2 == 0
            else:
                return False

    pgn_loss_coefs = {
        'proxy_prc_gt': args.pgn_loss_proxy_prc_gt_coef,
        'proxy_l1_gt': args.pgn_loss_proxy_l1_gt_coef,
        'proxy_mse_gt': args.pgn_loss_proxy_mse_gt_coef,
        'proxy_mse_gen': args.pgn_loss_proxy_mse_gen_coef,
        'grad': args.pgn_loss_grad_coef,
        'val': args.pgn_loss_value_coef,
        'jac': args.pgn_loss_jac_coef,
        'warping_grid_tv': args.pgn_loss_warping_grid_tv_coef,
        'warping_grid_l2': args.pgn_loss_warping_grid_l2_coef,
        'additive_l2': args.pgn_loss_additive_l2_coef,
    }

    calc_ssim = SSIM(size_average=False)

    with ExitStack() as stack:
        if args.distributed:
            stack.enter_context(ProcessGroup(
                backend=args.dist_backend,
                init_method=args.dist_init_method,
                world_size=args.world_size,
                rank=rank,
            ))

        logging.info('Initializing dataloaders...')
        train_dataloader, valid_dataloader = get_dataloaders(
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            train_batch_size=args.worker_batch_size,
            valid_batch_size=args.valid_batch_size,
            batches_per_train_epoch=args.batches_per_train_epoch,
            batches_per_valid_epoch=args.batches_per_valid_epoch,
            valid_first_samples=args.valid_first_samples,
            train_num_workers=args.workers,
            distributed=args.distributed,
        )

        normalizer, pgn, opt_pgn, autoencoders, ploss = load_networks(
            args, pgn_device, ae_device, vgg_device)

        if args.distributed:
            pgn = nn.parallel.DistributedDataParallel(pgn, device_ids=[rank])
            # It's fine if we don't sync autoencoders across GPUs. In fact, it might make training faster b/c PGN will
            # see more diverse samples. Also, ploss does not require gradients and thus does not need to be wrapped in
            # DistributedDataParallel.

        torch.backends.cudnn.benchmark = True

        pgn_grad_norm_avg = MovingAverage(args.pgn_grad_norm_avg)
        ae_grad_norm_avg = [MovingAverage(args.ae_grad_norm_avg)] * len(autoencoders)

        ae_best_l1 = [None] * len(autoencoders)
        ae_best_l1_it = [None] * len(autoencoders)

        pgn_loss_grad_func = {
            'mse': ScaledMSELoss(args.loss_scale),
            'vgg_one_step': VggOneStepLoss(ploss, args.vgg_one_step_lr),
            'vgg_grad': VggGradLoss(ploss),
        }[args.pgn_loss_grad]

        stack.enter_context(torch.no_grad())
        if args.distributed:
            if rank == 0:
                writer = stack.enter_context(SummaryWriter(str(tb_dir)))
            else:
                writer = None
            writer = DistributedSummaryWrapper(writer)
        else:
            writer = stack.enter_context(SummaryWriter(str(tb_dir)))

        for epoch in range(args.start_epoch, args.epochs):
            logging.info(f'Starting epoch {epoch}/{args.epochs}')

            if not args.valid_only:
                with Timer() as t_train:
                    train(
                        rank, args, writer, log_dir, epoch, train_dataloader, normalizer,
                        pgn, opt_pgn, pgn_device, pgn_grad_norm_avg, pgn_loss_grad_func, pgn_loss_coefs,
                        autoencoders, ae_device, ae_grad_norm_avg, ae_best_l1, ae_best_l1_it, use_true_grads,
                        ploss, vgg_device,
                    )

                writer.add_scalar('train/time', t_train.time(), epoch)
                logging.info(f'[epoch {epoch:>4d}/{args.epochs} | TRAIN] time {t_train.time():.0f}')

            with Timer() as t_valid:
                valid(
                    rank, args, writer, log_dir, epoch, valid_dataloader, normalizer, calc_ssim,
                    pgn.module if args.distributed else pgn, pgn_device, pgn_loss_grad_func,
                    ploss, vgg_device
                )

            writer.add_scalar('valid/time', t_valid.time(), epoch)
            logging.info(f'[epoch {epoch:>4d}/{args.epochs} | VALID] time {t_valid.time():.1f}')

            if not args.valid_only and (not args.distributed or rank == 0):
                if (epoch > 0 or args.save_each == 1) and ((epoch % args.save_each) == 0 or epoch == args.epochs - 1):
                    logging.info(f'Saving PGN+opt checkpoint to {model_dir}')
                    pgn_checkpoint = pgn.module.get_checkpoint() if args.distributed else pgn.get_checkpoint()
                    torch.save(pgn_checkpoint, model_dir / f'{epoch}.pth')
                    torch.save(opt_pgn.state_dict(), model_dir / f'{epoch}_opt.pth')

                    for ae_idx, autoencoder in enumerate(autoencoders):
                        logging.info(f'Saving AE#{ae_idx}+opt checkpoint to {model_dir}')
                        ae_state_dict, opt_ae_state_dict = autoencoder.state_dicts()
                        torch.save(ae_state_dict, model_dir / f'{epoch}_ae_{ae_idx}.pth')
                        torch.save(opt_ae_state_dict, model_dir / f'{epoch}_ae_{ae_idx}_opt.pth')

            if args.distributed:
                dist.barrier()

            if args.valid_only:
                break

    logging.info('Training complete, exiting.')


def train(rank: Optional[int], args, writer, log_dir: Optional[Path], epoch, train_dataloader, normalizer,
          pgn, opt_pgn, pgn_device, pgn_grad_norm_avg, pgn_loss_grad_func, pgn_loss_coefs,
          autoencoders, ae_device, ae_grad_norm_avg, ae_best_l1, ae_best_l1_it, use_true_grads,
          ploss, vgg_device):
    pgn.train()

    loading_start = time.time()
    for i, (gt, _) in enumerate(train_dataloader):
        batch_start = time.time()

        iteration_train = (epoch * args.batches_per_train_epoch + i) * args.batch_size

        time_loading = time.time() - loading_start
        writer.add_scalar('train/time_loading', time_loading, iteration_train)

        gt = gt.to(pgn_device, non_blocking=True)
        if args.double:
            gt = gt.double()

        with torch.enable_grad():
            assert args.worker_batch_size >= len(autoencoders)
            samples_per_ae = [
                args.worker_batch_size // len(autoencoders) +
                (1 if ae_idx < args.worker_batch_size % len(autoencoders) else 0)
                for ae_idx in range(len(autoencoders))
            ]
            assert sum(samples_per_ae) == args.worker_batch_size
            batch_indices = np.cumsum([0] + samples_per_ae)

            # AE forward
            gen = []
            for ae_idx, ae in enumerate(autoencoders):
                ae_input = gt[batch_indices[ae_idx]:batch_indices[ae_idx + 1]]
                gen.append(ae(ae_input.to(ae_device)).to(pgn_device))
            gen = torch.cat(gen, dim=0)
            gen.requires_grad_()  # it was detached in ae.forward(); grad is required for Jacobian loss

            # PGN forward on current input
            pred = pgn(gen, gt)

            assert not torch.isnan(pred["grad"]).any()

            if args.pgn_loss_grad_coef:
                # Ground truth gradient forward-backward

                # There is an option to not penalize the predicted gradients directly, in which case we
                # will only compute a loss on the proxy target (such as ploss(proxy, gt)). Otherwise, we
                # compute batchwise_loss_prc = ploss(gen, gt) and grads_true, its gradient with
                # respect to gen.

                def target_func(gen):
                    if args.pgn_target == 'vgg':
                        batchwise_loss = ploss(gen.to(vgg_device), gt.to(vgg_device)).to(pgn_device)
                    elif args.pgn_target == 'mse':
                        batchwise_loss = mse_loss_batchwise(gen, gt)
                    else:
                        assert False
                    if args.pgn_target_secondary is not None and args.pgn_target_secondary_coef:
                        if args.pgn_target_secondary == 'mse':
                            batchwise_loss_2 = mse_loss_batchwise(gen, gt)
                        else:
                            assert False
                        batchwise_loss = batchwise_loss + args.pgn_target_secondary_coef * batchwise_loss_2
                    return batchwise_loss

                grads_true, batchwise_loss_prc = forward_backward(target_func, gen)

            # AE backward
            ae_grads = grads_true if use_true_grads(epoch) else pred['grad']
            ae_grad_norms = []
            for ae_idx, autoencoder in enumerate(autoencoders):
                ae_grad_norm_limit = args.ae_grad_norm_clip_coef * ae_grad_norm_avg[ae_idx].get()
                ae_grad = ae_grads[batch_indices[ae_idx]:batch_indices[ae_idx + 1]]
                ae_grad_norm = autoencoders[ae_idx].step(ae_grad.to(ae_device), ae_grad_norm_limit)
                ae_grad_norms.append(ae_grad_norm)

            # PGN step
            opt_pgn.zero_grad()
            losses_pgn = {}
            if args.pgn_loss_proxy_prc_gt:
                losses_pgn['proxy_prc_gt'] = ploss(pred['proxy'].to(vgg_device), gt.to(vgg_device)).to(pgn_device)
            if args.pgn_loss_proxy_l1_gt:
                losses_pgn['proxy_l1_gt'] = l1_loss_batchwise(pred['proxy'], gt)
            if args.pgn_loss_proxy_mse_gt:
                losses_pgn['proxy_mse_gt'] = mse_loss_batchwise(pred['proxy'], gt)
            if args.pgn_loss_proxy_mse_gen:
                losses_pgn['proxy_mse_gen'] = mse_loss_batchwise(pred['proxy'], gen.detach())
            if args.pgn_loss_grad_coef:
                losses_pgn['grad'] = pgn_loss_grad_func(gen.detach(), gt, pred['grad'], grads_true)
            if args.pgn_predict_value:
                losses_pgn['val'] = mse_loss_batchwise(pred['val'], batchwise_loss_prc.detach())
            if args.symmetric_jacobian:
                losses_pgn['jac'] = symmetric_jacobian_loss(pred['grad'], gen)
            if args.pgn_loss_warping_grid_tv:
                losses_pgn['warping_grid_tv'] = total_variation_loss(pred['grid'])
            if args.pgn_loss_warping_grid_l2:
                losses_pgn['warping_grid_l2'] = (pred['grid']**2).mean(dim=(1, 2, 3))
            if args.pgn_loss_additive_l2:
                losses_pgn['additive_l2'] = (pred['additive']**2).mean(dim=(1, 2, 3))
            if not losses_pgn:
                raise RuntimeError('PGN has no loss!')
            loss_pgn = torch.stack([
                pgn_loss_coefs[k] * v if pgn_loss_coefs[k] != 1 else v
                for k, v in losses_pgn.items()
            ]).sum(dim=0)

            mem_pgn = torch.cuda.memory_allocated()
            with ExitStack() as stack_backward:
                t_backward = stack_backward.enter_context(Timer())
                if args.amp:
                    loss_pgn_scaled = stack_backward.enter_context(amp.scale_loss(loss_pgn, opt_pgn))
                else:
                    loss_pgn_scaled = loss_pgn
                loss_pgn_scaled.backward(torch.ones_like(loss_pgn))
            pgn_norms = get_grad_norms(pgn.module.backbone if args.distributed else pgn.backbone)
            pgn_grad_norm_limit = args.pgn_grad_norm_clip_coef * pgn_grad_norm_avg.get()
            pgn_grad_norm = nn.utils.clip_grad_norm_(pgn.parameters(), pgn_grad_norm_limit)
            with Timer() as t_step:
                opt_pgn.step()


        writer.add_scalar(f'train/time_backward', t_backward.time(), iteration_train)
        writer.add_scalar(f'train/time_step', t_step.time(), iteration_train)

        pgn_loss_val = loss_pgn.mean(dim=0).item()
        writer.add_scalar(f'train/loss', pgn_loss_val, iteration_train)

        writer.add_scalar(f'train/unet_grad_norm', pgn_grad_norm, iteration_train)
        writer.add_scalar(f'train/unet_grad_norm_avg', pgn_grad_norm_avg.get(), iteration_train)
        pgn_grad_norm_avg.update(pgn_grad_norm)

        # Metrics from autoencoders
        loss_l1 = l1_loss_batchwise(gen, gt)
        grad_metrics = get_grad_metrics(pred['grad'], grads_true if args.pgn_loss_grad_coef else None)

        writer.add_scalar(f'train/l1', loss_l1.mean(dim=0).item(), iteration_train)
        for k, v in grad_metrics.items():
            writer.add_scalar(f'train/{k}', v.mean(dim=0).item(), iteration_train)

        for ae_idx in range(len(autoencoders)):
            tb_ae_section = f'train_ae{ae_idx}'

            writer.add_scalar(f'{tb_ae_section}/grad_norm', ae_grad_norms[ae_idx], iteration_train)
            writer.add_scalar(f'{tb_ae_section}/grad_norm_avg', ae_grad_norm_avg[ae_idx].get(), iteration_train)
            ae_grad_norm_avg[ae_idx].update(ae_grad_norm)

            def log_tb_ae(key, value):
                val = value[batch_indices[ae_idx]:batch_indices[ae_idx + 1]].mean(dim=0).item()
                writer.add_scalar(f'{tb_ae_section}/{key}', val, iteration_train)
                return val

            loss_l1_val = log_tb_ae('l1', loss_l1)

            log_tb_ae('loss', loss_pgn)

            if args.pgn_loss_grad_coef:
                log_tb_ae('loss_grad', losses_pgn['grad'])
                log_tb_ae('perceptual', batchwise_loss_prc)

            if args.pgn_loss_proxy_prc_gt:
                log_tb_ae('proxy_loss_prc', losses_pgn['proxy_prc_gt'])
            if args.pgn_loss_proxy_l1_gt:
                log_tb_ae('proxy_loss_l1_gt', losses_pgn['proxy_l1_gt'])
            if args.pgn_loss_proxy_mse_gt:
                log_tb_ae('proxy_loss_mse_gt', losses_pgn['proxy_mse_gt'])
            if args.pgn_loss_proxy_mse_gen:
                log_tb_ae('proxy_loss_mse_gen', losses_pgn['proxy_mse_gen'])
            if args.symmetric_jacobian:
                log_tb_ae('loss_jac', losses_pgn['jac'])
            if args.pgn_loss_warping_grid_tv:
                log_tb_ae('warping_grid_tv', losses_pgn['warping_grid_tv'])
            if args.pgn_loss_warping_grid_l2:
                log_tb_ae('warping_grid_l2', losses_pgn['warping_grid_l2'])
            if args.pgn_loss_additive_l2:
                log_tb_ae('additive_l2', losses_pgn['additive_l2'])
            if args.pgn_predict_value:
                log_tb_ae('perceptual_pred', pred['val'])
                log_tb_ae('perceptual_pred_loss', losses_pgn['val'])

            for k, v in grad_metrics.items():
                log_tb_ae(k, v)

            # Save gt & gen once in a while
            if i == 0:
                if rank is None:
                    image_filename = f'{epoch:04d}_{i:03d}.jpg'
                else:
                    image_filename = f'{epoch:04d}_{i:03d}_{rank:02d}.jpg'
                frame = make_frame(normalizer, gt, gen)
                imageio.imwrite(log_dir / image_filename, frame)

            # Optionally reset autoencoder
            if ae_best_l1[ae_idx] is None or loss_l1_val < ae_best_l1[ae_idx]:
                ae_best_l1[ae_idx] = loss_l1_val
                ae_best_l1_it[ae_idx] = iteration_train
            if iteration_train - ae_best_l1_it[ae_idx] >= args.ae_min_samples_before_reset:
                reset_prob = args.ae_reset_prob
            elif loss_l1_val > args.ae_l1_hi_reset_threshold:
                reset_prob = 1
            else:
                reset_prob = 0
            if reset_prob:
                logging.info(f'Autoencoder {ae_idx} is eligible for reset with probability {reset_prob:.3f}')
                if np.random.rand() < reset_prob:
                    logging.info(f'Resetting autoencoder {ae_idx}')
                    autoencoders[ae_idx].reset()
                    ae_best_l1[ae_idx] = None
                    ae_best_l1_it[ae_idx] = None

        for field_name, field_norm in pgn_norms.items():
            writer.add_scalar(f'{args.pgn_arch}/norm_{field_name}', field_norm, iteration_train)
        if args.enable_histograms:
            writer.add_histogram(f'train/grads_pred', pred['grad'], iteration_train)
            if args.pgn_loss_grad_coef:
                writer.add_histogram(f'train/grads_true', grads_true, iteration_train)

        batch_time = time.time() - batch_start
        writer.add_scalar(f'train/batch_time', batch_time, iteration_train)
        writer.add_scalar(f'train/mem_pgn', mem_pgn, iteration_train)

        log_header = ' | '.join([
            f'epoch {epoch:>4d}/{args.epochs}',
            f'TRAIN',
            f'batch {i:>3d}/{len(train_dataloader)}',
        ])
        log_items = [
            f'time {batch_time:.2f}',
            f'load {time_loading:.2f}',
            f'loss {pgn_loss_val:.4f}',
        ]
        if args.pgn_loss_grad_coef:
            grad_log = ' | '.join([
                f'true {grad_metrics["grads_true_norm"].mean(dim=0).item():.4f}',
                f'pred {grad_metrics["grads_pred_norm"].mean(dim=0).item():.4f}',
            ])
            log_items.append(f'grad_norm {{{grad_log}}}')
        logging.debug(f'[{log_header}] {" | ".join(log_items)}')

        loading_start = time.time()


def valid(rank: Optional[int], args, writer, log_dir, epoch, valid_dataloader, normalizer, calc_ssim,
          pgn, pgn_device, pgn_loss_grad_func,
          ploss, vgg_device):
    pgn.eval()

    for i, (gt, _) in enumerate(valid_dataloader):
        if i < args.valid_skip_batches:
            logging.debug(f'Skipping validation batch {i}')
            continue

        batch_start = time.time()
        valid_detailed_metrics = [defaultdict(list) for _ in range(args.valid_batch_size)]

        if rank is None:
            log_filename_prefix = f'{epoch:04d}_{i:03d}'
        else:
            log_filename_prefix = f'{epoch:04d}_{i:03d}_{rank:02d}'

        if args.valid_task == 'dip':
            gt = gt.to(pgn_device, non_blocking=True)

            # Initialize DIPs
            skip_params = default_skip_params.copy()
            skip_params['need_sigmoid'] = args.dip_need_sigmoid

            dip_pgn = skip(**skip_params).to(pgn_device)
            if not args.valid_disable_model_prc:
                dip_prc = skip(**skip_params).to(pgn_device)
                dip_prc.load_state_dict(dip_pgn.state_dict())
            if not args.valid_disable_model_mse:
                dip_mse = skip(**skip_params).to(pgn_device)
                dip_mse.load_state_dict(dip_pgn.state_dict())

            if args.double:
                gt = gt.double()
                dip_pgn.double()
                if not args.valid_disable_model_prc:
                    dip_prc.double()
                if not args.valid_disable_model_mse:
                    dip_mse.double()

            parameters_pgn = dip_pgn.parameters()
            if not args.valid_disable_model_prc:
                parameters_prc = dip_prc.parameters()
            if not args.valid_disable_model_mse:
                parameters_mse = dip_mse.parameters()

            dip_input_mean = torch.rand_like(gt).mul_(args.dip_input_mean_scale).detach()
            dip_input_noise = torch.zeros_like(dip_input_mean)
            use_noise = args.dip_input_noise_std > 0 and np.random.random() < args.dip_input_noise_prob
        else:
            batch_dict = gt
            gt = batch_dict['images']
            latents = batch_dict['latents']

            gt = gt.to(pgn_device, non_blocking=True)
            gt = normalizer((gt + 1) / 2)
            latents = {k: v.to(pgn_device, non_blocking=True) for (k, v) in latents.items()}
            optimize_names = [name for name in latents if 'prime' in name]

            # Initialize StyleGAN
            checkpoint = torch.load(args.valid_stylegan_checkpoint_path, map_location='cpu')
            generator = StyledGenerator()
            generator.load_state_dict(checkpoint['g_running'])
            generator = generator.to(pgn_device)
            generator.eval()

            resolution = 256
            step = int(np.log2(resolution) - 2)
            decoder = DecoderFromLatents(generator, step=step)

            latents_pgn = {
                name: nn.Parameter(tensor.clone().detach().to(pgn_device))
                for name, tensor in latents.items()}
            latents_prc = {
                name: nn.Parameter(tensor.clone().detach().to(pgn_device))
                for name, tensor in latents.items()}
            latents_mse = {
                name: nn.Parameter(tensor.clone().detach().to(pgn_device))
                for name, tensor in latents.items()}

            parameters_pgn = [latents_pgn[name] for name in optimize_names]
            parameters_prc = [latents_prc[name] for name in optimize_names]
            parameters_mse = [latents_mse[name] for name in optimize_names]

        opt_valid_pgn = torch.optim.Adam(parameters_pgn, lr=args.valid_lr)
        if args.valid_task == 'stylegan' and 'opt_state' in batch_dict:
            opt_valid_pgn.load_state_dict(batch_dict['opt_state'])
        if not args.valid_disable_model_prc:
            opt_valid_prc = torch.optim.Adam(parameters_prc, lr=args.valid_lr)
            if args.valid_task == 'stylegan' and 'opt_state' in batch_dict:
                opt_valid_prc.load_state_dict(batch_dict['opt_state'])
        if not args.valid_disable_model_mse:
            opt_valid_mse = torch.optim.Adam(parameters_mse, lr=args.valid_lr)
            if args.valid_task == 'stylegan' and 'opt_state' in batch_dict:
                opt_valid_mse.load_state_dict(batch_dict['opt_state'])

        initial_prc_loss_val = None

        min_prc_loss_val = np.full(args.valid_batch_size, np.inf)
        min_prc_loss_it = np.zeros(args.valid_batch_size, dtype=int)

        with ExitStack() as stack:
            if not args.valid_disable_video:
                temp_dir = stack.enter_context(tempfile.TemporaryDirectory(dir='/dev/shm', prefix='dvgg-video-'))
                temp_dir = Path(temp_dir)
                if rank is None:
                    video_filename = f'{epoch:04d}_{i:03d}.mp4'
                else:
                    video_filename = f'{epoch:04d}_{i:03d}_{rank:02d}.mp4'
                stack.callback(shutil.move, temp_dir / video_filename, log_dir / video_filename)
                video_writer = stack.enter_context(VideoWriter(temp_dir / video_filename))

            progress_bar = stack.enter_context(tqdm(
                range(args.valid_num_iter), disable=not args.tqdm,
                desc=f'[epoch {epoch:>4d}/{args.epochs} | VALID | batch {i}/{len(valid_dataloader)}]'))

            for j in progress_bar:
                # TODO: log individual validation runs separately, with
                # batches_valid = (epoch * args.batches_per_valid_epoch + i) * args.world_size + rank
                batches_valid = epoch * args.batches_per_valid_epoch + i

                if args.valid_task == 'dip':
                    if use_noise:
                        dip_input = dip_input_mean + dip_input_noise.normal_(std=args.dip_input_noise_std)
                    else:
                        dip_input = dip_input_mean

                with torch.enable_grad():
                    opt_valid_pgn.zero_grad()
                    if args.valid_task == 'dip':
                        gen_pgn = normalizer(dip_pgn(dip_input))
                    else:
                        gen_pgn = normalizer((decoder(latents_pgn) + 1) / 2)
                    pred = pgn(gen_pgn.detach(), gt)
                    loss_dip_pgn = (gen_pgn * pred['grad']).sum(dim=(1, 2, 3))
                    if args.valid_pgn_mse_coef:
                        loss_dip_pgn_for_backward = loss_dip_pgn + args.valid_pgn_mse_coef * mse_loss_batchwise(gen_pgn, gt)
                    else:
                        loss_dip_pgn_for_backward = loss_dip_pgn
                    loss_dip_pgn_for_backward.backward(torch.ones_like(loss_dip_pgn))
                    if args.valid_clip_grad:
                        torch.nn.utils.clip_grad_norm_(parameters_pgn, args.valid_clip_grad_norm)
                    opt_valid_pgn.step()

                    if not args.valid_disable_model_prc:
                        opt_valid_prc.zero_grad()
                        if args.valid_task == 'dip':
                            gen_prc = normalizer(dip_prc(dip_input))
                        else:
                            gen_prc = normalizer((decoder(latents_prc) + 1) / 2)
                        loss_dip_prc = ploss(gen_prc.to(vgg_device), gt.to(vgg_device))
                        if args.valid_prc_mse_coef:
                            loss_dip_prc_for_backward = loss_dip_prc + args.valid_prc_mse_coef * mse_loss_batchwise(gen_prc, gt)
                        else:
                            loss_dip_prc_for_backward = loss_dip_prc
                        loss_dip_prc_for_backward.backward(torch.ones_like(loss_dip_prc))
                        if args.valid_clip_grad:
                            torch.nn.utils.clip_grad_norm_(parameters_prc, args.valid_clip_grad_norm)
                        opt_valid_prc.step()

                    if not args.valid_disable_model_mse:
                        opt_valid_mse.zero_grad()
                        if args.valid_task == 'dip':
                            gen_mse = normalizer(dip_mse(dip_input))
                        else:
                            gen_mse = normalizer((decoder(latents_mse) + 1) / 2)
                        if args.valid_mse_alternative == 'mse':
                            loss_dip_mse = mse_loss_batchwise(gen_mse, gt)
                        elif args.valid_mse_alternative == 'l1':
                            loss_dip_mse = l1_loss_batchwise(gen_mse, gt)
                        else:
                            assert False
                        loss_dip_mse.backward(torch.ones_like(loss_dip_mse))
                        if args.valid_clip_grad:
                            torch.nn.utils.clip_grad_norm_(parameters_mse, args.valid_clip_grad_norm)
                        opt_valid_mse.step()

                loss_pgn_l1_val = l1_loss_batchwise(gen_pgn, gt).cpu().numpy()
                loss_pgn_l2_val = mse_loss_batchwise(gen_pgn, gt).cpu().numpy()
                grads_true, loss_pgn_prc = forward_backward(
                    lambda gen: ploss(gen.to(vgg_device), gt.to(vgg_device)).to(pgn_device), gen_pgn)
                loss_pgn_prc_val = loss_pgn_prc.cpu().numpy()
                pgn_metrics = [
                    loss_pgn_l1_val,
                    loss_pgn_l2_val,
                    loss_pgn_prc_val,
                    calc_ssim(gen_pgn, gt).cpu().numpy(),
                ]

                if not args.valid_disable_model_prc:
                    prc_metrics = [
                        l1_loss_batchwise(gen_prc, gt).cpu().numpy(),
                        mse_loss_batchwise(gen_prc, gt).cpu().numpy(),
                        loss_dip_prc.cpu().numpy(),
                        calc_ssim(gen_prc, gt).cpu().numpy(),
                    ]

                if not args.valid_disable_model_mse:
                    mse_metrics = [
                        l1_loss_batchwise(gen_mse, gt).cpu().numpy(),
                        loss_dip_mse.cpu().numpy(),
                        ploss(gen_mse.to(vgg_device), gt.to(vgg_device)).cpu().numpy(),
                        calc_ssim(gen_mse, gt).cpu().numpy(),
                    ]

                def record_metrics(metrics, suffix=None):
                    for key, values in metrics.items():
                        if suffix is not None:
                            writer.add_scalar(f'{key}_{suffix}', values.mean(axis=0), batches_valid)
                        else:
                            assert isinstance(values, np.ndarray), (key, values)
                            assert len(values) == args.valid_batch_size, (key, values)
                            for b, value in enumerate(values):
                                valid_detailed_metrics[b][key].append(value)

                loss_pgn_grad_val = pgn_loss_grad_func(gen_pgn, gt, pred['grad'], grads_true).cpu().numpy()
                frame_metrics = {'valid_pgn/loss': loss_pgn_grad_val}

                if args.pgn_predict_value:
                    frame_metrics['valid_pgn/gen_perceptual_pred'] = pred['val'].cpu().numpy()

                image_metrics = dict(zip(
                    ['valid_' + key for key in [
                        'pgn/l1', 'pgn/l2', 'pgn/perceptual', 'pgn/ssim',
                        *(['prc/l1', 'prc/l2', 'prc/perceptual', 'prc/ssim']
                          if not args.valid_disable_model_prc else []),
                        *(['mse/l1', 'mse/l2', 'mse/perceptual', 'mse/ssim']
                          if not args.valid_disable_model_mse else []),
                        *(['ratio_prc/l1', 'ratio_prc/l2', 'ratio_prc/perceptual', 'ratio_prc/ssim']
                          if not args.valid_disable_model_prc else []),
                        *(['ratio_mse/l1', 'ratio_mse/l2', 'ratio_mse/perceptual', 'ratio_mse/ssim']
                          if not args.valid_disable_model_mse else []),
                        *(['ratio_prc_mse/l1', 'ratio_prc_mse/l2', 'ratio_prc_mse/perceptual', 'ratio_prc_mse/ssim']
                          if not args.valid_disable_model_prc or not args.valid_disable_model_mse else []),
                    ]],
                    pgn_metrics +
                    (prc_metrics if not args.valid_disable_model_prc else []) +
                    (mse_metrics if not args.valid_disable_model_mse else []) +
                    ([v_pgn / v_prc for (v_pgn, v_prc) in zip(pgn_metrics, prc_metrics)]
                     if not args.valid_disable_model_prc else []) +
                    ([v_pgn / v_mse for (v_pgn, v_mse) in zip(pgn_metrics, mse_metrics)]
                     if not args.valid_disable_model_mse else []) +
                    ([v_prc / v_mse for (v_prc, v_mse) in zip(prc_metrics, mse_metrics)]
                     if not args.valid_disable_model_prc and not args.valid_disable_model_mse else [])
                ))
                frame_metrics.update(image_metrics)

                grad_metrics = {
                    ('valid_pgn/' + key): value.cpu().numpy()
                    for (key, value) in get_grad_metrics(pred['grad'], grads_true).items()
                }
                frame_metrics.update(grad_metrics)

                proxy_pred = pred.get('proxy', None)
                if proxy_pred is not None:
                    frame_metrics.update({
                        f'valid_pgn/proxy_pred_q{q}': p.cpu().numpy()
                        for (q, p) in get_multiple_percentiles(proxy_pred).items()
                    })

                proxy_true = None
                if args.pgn_proxy_grad_type == 'mse':
                    _, c, h, w = gen_pgn.shape
                    proxy_true = gen_pgn - ((c * h * w) / 2) * grads_true
                    frame_metrics.update({
                        f'valid_pgn/proxy_true_q{q}': p.cpu().numpy()
                        for (q, p) in get_multiple_percentiles(proxy_true).items()
                    })

                record_metrics(frame_metrics)

                if j + 1 in args.valid_log_iter:
                    record_metrics(frame_metrics, str(j + 1))

                    if args.enable_histograms:
                        writer.add_histogram(f'valid_pgn/grads_pred_{j + 1}', pred['grad'], batches_valid)
                        writer.add_histogram(f'valid_pgn/grads_true_{j + 1}', grads_true, batches_valid)

                log_items = [
                    f'L1: {loss_pgn_l1_val.mean(axis=0):.4f}',
                    f'Prc: {loss_pgn_prc_val.mean(axis=0):.4f}',
                    f'MSE: {loss_pgn_grad_val.mean(axis=0):.4f}',
                    f'MSEn: {frame_metrics["valid_pgn/loss_norm"].mean(axis=0) * 100:.3f}%',
                    f'cos: {frame_metrics["valid_pgn/cosine_similarity"].mean(axis=0):.3f}',
                ]

                if args.tqdm:
                    progress_bar.set_postfix_str(' | '.join(log_items))
                elif j % 100 == 0 or j == args.valid_num_iter - 1:
                    header = ' | '.join([
                        f'epoch {epoch:>4d}/{args.epochs}',
                        f'VALID',
                        f'batch {i:>3d}/{len(valid_dataloader)}',
                        f'it {j}',
                    ])
                    logging.debug(f'[{header}] {" | ".join(log_items)}')

                if not args.valid_disable_video:
                    video_writer.write(make_frame(
                        normalizer, gt, gen_pgn, grads_true, pred['grad'],
                        proxy_true=proxy_true,
                        proxy_pred=proxy_pred,
                        gen_prc=gen_prc if not args.valid_disable_model_prc else None,
                        gen_mse=gen_mse if not args.valid_disable_model_mse else None,
                        nrow=2,
                    ))

                min_prc_loss_it = np.where(min_prc_loss_val <= loss_pgn_prc_val, min_prc_loss_it, j)
                min_prc_loss_val = np.minimum(min_prc_loss_val, loss_pgn_prc_val)

                need_to_break = False
                if initial_prc_loss_val is None:
                    initial_prc_loss_val = loss_pgn_prc_val
                elif (loss_pgn_prc_val >= args.valid_divergence_threshold * initial_prc_loss_val).any():
                    bad_index = loss_pgn_prc_val >= args.valid_divergence_threshold * initial_prc_loss_val
                    logging.debug(
                        f'Perceptual loss is too high ('
                        f'bad batch indices: {bad_index}, '
                        f'{loss_pgn_prc_val[bad_index]} >= {args.valid_divergence_threshold} * {initial_prc_loss_val[bad_index]}'
                        f'), breaking at iteration {j}')
                    need_to_break = True
                elif (min_prc_loss_it <= j - args.valid_plateau_patience).all():
                    logging.debug(
                        f'Perceptual loss has not been improved for {args.valid_plateau_patience} iterations ('
                        f'last improved: {min_prc_loss_it}'
                        f'), breaking at iteration {j}')
                    need_to_break = True

                if need_to_break:
                    for it in args.valid_log_iter:
                        if it > j + 1:
                            record_metrics(frame_metrics, str(it))
                    record_metrics(frame_metrics, 'abort')
                    writer.add_scalar('valid_pgn/num_iterations', j, batches_valid)
                    break
            else:
                record_metrics(frame_metrics, 'abort')
                writer.add_scalar('valid_pgn/num_iterations', args.valid_num_iter, batches_valid)

        images_to_save = {
            'gt': gt,
            'gen_pgn': gen_pgn,
        }
        if not args.valid_disable_model_prc:
            images_to_save['gen_prc'] = gen_prc
        if not args.valid_disable_model_mse:
            images_to_save['gen_mse'] = gen_mse
        if proxy_pred is not None:
            images_to_save['proxy_pred'] = proxy_pred
        for suffix, batch_torch in images_to_save.items():
            for b in range(args.valid_batch_size):
                image_filename = f'{log_filename_prefix}_{b:02d}_{suffix}.png'
                image_np = torch_batch_to_numpy(
                    batch_torch[b].unsqueeze(0), nrow=1, normalizer=normalizer)
                imageio.imwrite(log_dir / image_filename, image_np)

        batch_time = time.time() - batch_start
        writer.add_scalar('valid/batch_time', batch_time, epoch * args.batches_per_valid_epoch + 1)

        valid_detailed_metrics = [
            {key: np.array(value) for key, value in metrics_for_batch_element.items()}
            for metrics_for_batch_element in valid_detailed_metrics
        ]

        for b in range(args.valid_batch_size):
            with (log_dir / f'{log_filename_prefix}_{b:02d}.pkl').open('wb') as fp:
                pickle.dump(valid_detailed_metrics[b], fp)

            if args.valid_task == 'stylegan':
                for suffix, latents in [('pgn', latents_pgn), ('prc', latents_prc), ('mse', latents_mse)]:
                    torch.save(
                        {name: tensor.data[b] for name, tensor in latents.items()},
                        log_dir / f'{log_filename_prefix}_{b:02d}_latents_{suffix}.pth',
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--debug', action='store_true',
                        help='Enable options suitable for debugging')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to use for Python, Numpy, and PyTorch generators')
    parser.add_argument('--pgn-device', type=str, default='cuda:0')
    parser.add_argument('--ae-device', type=str, default='cuda:0')
    parser.add_argument('--vgg-device', type=str, default='cuda:0')

    parser.add_argument('--no-distributed', action='store_true',
                        help='Do not use multi-processing distributed training.')
    parser.add_argument('--world-size', type=int, default=None,
                        help='Number of processes for distributed training.')
    parser.add_argument('--dist-init-method', type=str, default='tcp://localhost:23456',
                        help='URL used to set up distributed training.')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='Distributed backend.')

    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision.')
    parser.add_argument('--amp-opt-level', type=str, default='O1')

    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--valid-only', action='store_true',
                        help='Run one epoch of validation and then exit')

    parser.add_argument('--dataset-name', type=str, default='ImageNet',
                        help='Name of training dataset')
    parser.add_argument('--data-dir', type=Path, default=Path('../datasets'),
                        help='Path to directory with dataset')
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Path to directory where output should be stored')

    parser.add_argument('--double', action='store_true',
                        help='Perform computations in float64 instead of float32')
    parser.add_argument('--worker-batch-size', type=int, default=48)
    parser.add_argument('--save-each', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Do not set log level to debug')
    parser.add_argument('--tqdm', action='store_true',
                        help='Use tqdm. Results in very large log files')
    parser.add_argument('--enable-histograms', action='store_true',
                        help='Log histograms to Tensorboard. Increases log size dramatically')

    parser.add_argument('--batches-per-train-epoch', type=int, metavar='N', default=512,
                        help='How many batches to sample during each epoch for training')
    parser.add_argument('--batches-per-valid-epoch', type=int, metavar='N', default=1,
                        help='How many batches to sample during each epoch for validation')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Stop after this many epochs')

    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Train autoencoder on ground truth trajectories for this many epochs')
    parser.add_argument('--no-alternate-gt-curr', action='store_true',
                        help='Do not train AEs on true gradients during even epochs when epoch > warmup_epochs')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='Manual epoch number (useful on restarts)')

    parser.add_argument('--ploss-type', type=str, default='full', choices=['full', 'deep'],
                        help='Implementation flavour of VGG-based perceptual loss')
    parser.add_argument('--ploss-model', type=str, default='vgg19', choices=['vgg19', 'vgg11'],
                        help='Backbone model for the perceptual loss')

    parser.add_argument('--pgn-arch', type=str, default='resnet', choices=['unet', 'resnet'],
                        help='PGN architecture')
    parser.add_argument('--block-type', type=str, default='doubleblaze',
                        choices=['singleres', 'doubleres', 'singleblaze', 'doubleblaze'],
                        help='Residual block type to use in UNet')
    parser.add_argument('--conv-type', type=str, default='sep', choices=['full', 'sep'],
                        help='Convolution type to use in UNet')
    parser.add_argument('--norm-type', type=str, default='batch', choices=['batch', 'instance'],
                        help='Normalization to use in UNet')
    parser.add_argument('--pgn-out-scale', type=float, default=512.0,
                        help='Divide PGN output by this constant. Helps with bad initialization scale')
    parser.add_argument('--loss-scale', type=float, default=67108864.0,  # 2**26
                        help='Multiply MSE loss by this constant. Helps with numerical stability')
    parser.add_argument('--pgn-grad-norm-avg', type=float, default=200.0,
                        help='Initial approximation for average gradient norm of PGN')
    parser.add_argument('--pgn-grad-norm-clip-coef', type=float, default=5.0,
                        help='PGN gradients are not allowed to exceed the average by a factor greater than this')
    parser.add_argument('--unet-down-channels', nargs='+', type=int, default=(64, 64, 128, 256, 512),
                        help='Number of channels in the output of each downsampling UNet layer')
    parser.add_argument('--unet-up-channels', nargs='+', type=int, default=(64, 128, 256, 256, 512),
                        help='Number of channels in the output of each upsampling UNet layer')
    parser.add_argument('--resnet-down-channels', nargs='+', type=int, default=(64, 128),
                        help='Number of channels in the output of each downsampling ResNet layer.')
    parser.add_argument('--resnet-up-channels', nargs='+', type=int, default=(64,),
                        help='Number of channels in the output of each upsampling ResNet layer.')
    parser.add_argument('--resnet-num-blocks', type=int, default=8,
                        help='Number of blocks in ResNet.')
    parser.add_argument('--pgn-optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer used for PGN')
    parser.add_argument('--pgn-sgd-lr', type=float, default=1.0,
                        help='Learning rate for SGD if used as PGN optimizer')

    parser.add_argument('--pgn-target', type=str, default='vgg', choices=['vgg', 'mse'],
                        help='Which gradients PGN learns to approximate')
    parser.add_argument('--pgn-target-secondary', type=str, default=None, choices=['mse'],
                        help='Other PGN gradients to add to the primary ones')
    parser.add_argument('--pgn-target-secondary-coef', type=float, default=0.01,
                        help='Coefficient for secondary gradients')
    parser.add_argument('--pgn-loss-grad', type=str, default='mse', choices=['mse', 'vgg_one_step', 'vgg_grad'],
                        help='Loss for the gradient predicted by the PGN')
    parser.add_argument('--vgg-one-step-lr', type=float, default=1.0,
                        help='Step size for the vgg_one_step loss')

    parser.add_argument('--pgn-loss-grad-coef', type=float, default=1.0,
                        help='Coefficient for PGN gradient loss')

    parser.add_argument('--pgn-predict-value', action='store_true',
                        help='Have PGN predict perceptual loss value in addition to gradients')
    parser.add_argument('--pgn-loss-value-coef', type=float, default=0.1,
                        help='Coefficient for the loss on prediction of perceptual loss value')

    parser.add_argument('--pgn-backbone-to-grad', type=str, choices=['direct', 'proxy'], default='proxy',
                        help='How gradients are computed from backbone output.')

    parser.add_argument('--pgn-proxy-type', type=str, choices=['raw', 'sigmoid', 'warped_target'], default='sigmoid',
                        help='How proxy is constructed from PGN output.')

    parser.add_argument('--pgn-proxy-sigmoid-scale', type=float, default=1.1,
                        help='Scale normalizer(sigmoid(proxy)) by this constant.')

    parser.add_argument('--pgn-proxy-warped-target-scale', type=float, default=3 / 112,
                        help='Scale tanh(grid_correction) by this constant.')
    parser.add_argument('--pgn-proxy-warped-target-additive', action='store_true',
                        help='Predict an additive component in PGN proxy warped target mode.')
    parser.add_argument('--pgn-proxy-warped-target-downscale-by', type=float, default=224 / 8,
                        help='Downscale warping grid [and additive component] by this factor before proceeding.')
    parser.add_argument('--pgn-proxy-warped-target-additive-scale', type=float, default=0.05,
                        help='Scale the additive component in PGN proxy warped target mode by this factor.')
    parser.add_argument('--pgn-loss-warping-grid-tv', action='store_true',
                        help='Enable the total variation loss on the PGN warping grid.')
    parser.add_argument('--pgn-loss-warping-grid-tv-coef', type=float, default=1.0,
                        help='Coefficient for the total variation loss on the PGN warping grid.')
    parser.add_argument('--pgn-loss-warping-grid-l2', action='store_true',
                        help='Enable the L2 loss on the PGN warping grid.')
    parser.add_argument('--pgn-loss-warping-grid-l2-coef', type=float, default=1.0,
                        help='Coefficient for the L2 loss on the PGN warping grid.')
    parser.add_argument('--pgn-loss-additive-l2', action='store_true',
                        help='Enable the L2 loss on the PGN additive component.')
    parser.add_argument('--pgn-loss-additive-l2-coef', type=float, default=1.0,
                        help='Coefficient for the L2 loss on the PGN additive component.')

    parser.add_argument('--pgn-proxy-grad-scale', type=float, default=1 / 40,
                        help='PGN gradient scale in proxy mode')

    parser.add_argument('--pgn-proxy-grad-type', type=str, default='mse',
                        choices=['mse', 'l1', 'logcosh', 'mse_logit', 'logcosh_logit'],
                        help='PGN gradient type between proxy and generated image in proxy mode')

    parser.add_argument('--pgn-loss-proxy-prc-gt', action='store_true',
                        help='Enable VGG(proxy, gt) PGN loss')
    parser.add_argument('--pgn-loss-proxy-prc-gt-coef', type=float, default=0.2,
                        help='Weight for VGG(proxy, gt) PGN loss')

    parser.add_argument('--pgn-loss-proxy-l1-gt', action='store_true',
                        help='Enable L1(proxy, gt) PGN loss')
    parser.add_argument('--pgn-loss-proxy-l1-gt-coef', type=float, default=0.5,
                        help='Weight for L1(proxy, gt) PGN loss')

    parser.add_argument('--pgn-loss-proxy-mse-gt', action='store_true',
                        help='Enable MSE(proxy, gt) PGN loss')
    parser.add_argument('--pgn-loss-proxy-mse-gt-coef', type=float, default=1.0,
                        help='Weight for MSE(proxy, gt) PGN loss')

    parser.add_argument('--pgn-loss-proxy-mse-gen', action='store_true',
                        help='Enable MSE(proxy, gen) PGN loss')
    parser.add_argument('--pgn-loss-proxy-mse-gen-coef', type=float, default=1e3,
                        help='Weight for MSE(proxy, gen) PGN loss')

    parser.add_argument('--symmetric-jacobian', action='store_true',
                        help='Enable PGN loss that steers its Jacobian to a symmetric matrix')
    parser.add_argument('--pgn-loss-jac-coef', type=float, default=1e4,
                        help='Coefficient for the symmetric Jacobian loss')

    parser.add_argument('--ae-max-layer', nargs='+', type=int, default=[0, 1, 2],
                        help='Maximum layer depth in each autoencoder')
    parser.add_argument('--ae-grad-norm-avg', type=float, default=200.0,
                        help='Initial approximation for average gradient norm of each autoencoder')
    parser.add_argument('--ae-grad-norm-clip-coef', type=float, default=5.0,
                        help='Gradients of each autoencoder are not allowed to exceed the average by a factor greater than this')
    parser.add_argument('--ae-block-type', type=str, default='doubleblaze',
                        choices=['singleres', 'doubleres', 'singleblaze', 'doubleblaze'],
                        help='Residual block type to use in each autoencoder')
    parser.add_argument('--ae-conv-type', type=str, default='sep', choices=['full', 'sep'],
                        help='Convolution type to use in each autoencoder')
    parser.add_argument('--ae-min-samples-before-reset', type=int, default=300000,
                        help='Autoencoders will have to plateau for this many samples before being considered for a reset')
    parser.add_argument('--ae-reset-prob', type=float, default=0.01,
                        help='Autoencoder reset probability after reaching plateau')
    parser.add_argument('--ae-l1-hi-reset-threshold', type=float, default=1e3,
                        help='Autoencoders will be forcibly reset if L1 loss becomes greater than this')

    parser.add_argument('--resume-dir', default=None, type=Path, metavar='PATH',
                        help='Directory to use for resuming. Overrides more specific parameters.')
    parser.add_argument('--resume-epoch', default=None, type=int, metavar='PATH',
                        help='Epoch to use for resuming. Overrides more specific parameters.')

    parser.add_argument('--pgn-checkpoint', default=None, type=Path, metavar='PATH',
                        help='Path to latest PGN checkpoint')
    parser.add_argument('--pgn-checkpoint-opt', default=None, type=Path, metavar='PATH',
                        help='Path to latest PGN optimizer checkpoint')
    parser.add_argument('--pgn-ignore-grad-scale-mismatch', action='store_true',
                        help='Do not abort if grad_scale in PGN checkpoint is different from grad_scale from CLI')
    parser.add_argument('--ae-checkpoints', nargs='+', type=Path, default=None,
                        help='Paths to checkpoints of autoencoders. Use \'None\' to initialize from scratch')
    parser.add_argument('--ae-checkpoints-opt', nargs='+', type=Path, default=None,
                        help='Paths to checkpoints of autoencoder optimizers. Use \'None\' to initialize from scratch')

    parser.add_argument('--valid-task', type=str, choices=['dip', 'stylegan'], default='dip',
                        help='Task for validation (either Deep Image Prior or Image2StyleGAN)')

    parser.add_argument('--valid-stylegan-checkpoint-path', type=Path,
                        default=Path('../checkpoints/stylegan-256px.model'),
                        help='Path to 256px StyleGAN checkpoint')

    parser.add_argument('--valid-batch-size', type=int, default=1,
                        help='During validation, the model will approximate this many images')
    parser.add_argument('--valid-num-iter', type=int, default=3000,
                        help='During validation, optimization will be performed for this many steps')
    parser.add_argument('--valid-log-iter', nargs='+', type=int, default=[1000, 2000, 3000],
                        help='Iterations to use for logging perceptual loss during optimization in validation')
    parser.add_argument('--valid-lr', type=float, default=0.01,
                        help='Learning rate for the optimizer in validation')

    parser.add_argument('--dip-input-mean-scale', type=float, default=0.1,
                        help='Mean input to DIP will be scaled by this factor')
    parser.add_argument('--dip-input-noise-std', type=float, default=1 / 30,
                        help='Extra noise for DIP will be scaled by this factor')
    parser.add_argument('--dip-input-noise-prob', type=float, default=0.5,
                        help='Probability of adding noise to DIP input for each run')
    parser.add_argument('--dip-need-sigmoid', action='store_true',
                        help='Add sigmoid at the end of the DIP network')

    parser.add_argument('--valid-pgn-mse-coef', type=float, default=0.0,
                        help='Coefficient for MSE loss during validation for models trained with PGN.')
    parser.add_argument('--valid-prc-mse-coef', type=float, default=0.0,
                        help='Coefficient for MSE loss during validation for models trained with perceptual loss.')

    parser.add_argument('--valid-divergence-threshold', type=float, default=2.0,
                        help='Break if perceptual loss exceeds this times its initial value.')
    parser.add_argument('--valid-plateau-patience', type=int, default=2000,
                        help='Break after this many batches if perceptual loss does not improve.')

    parser.add_argument('--valid-clip-grad', action='store_true',
                        help='Enable gradient clipping in validation')
    parser.add_argument('--valid-clip-grad-norm', type=float, default=10.0,
                        help='Max gradient norm in validation')

    parser.add_argument('--valid-disable-model-prc', action='store_true',
                        help='Do not fit a model using perceptual loss on validation')
    parser.add_argument('--valid-disable-model-mse', action='store_true',
                        help='Do not fit a model using MSE loss on validation')
    parser.add_argument('--valid-mse-alternative', type=str, choices=['mse', 'l1'], default='mse',
                        help='Alternative loss to use instead of MSE in validation')
    parser.add_argument('--valid-disable-video', action='store_true',
                        help='Do not record trajectory video in validation')

    parser.add_argument('--valid-skip-batches', type=int, default=0,
                        help='Skip this many first batches at each epoch in validation')

    parser.add_argument('--valid-first-samples', type=int, default=None,
                        help='Run on the first N samples from the validation set')

    main(parser)
