#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from torch import nn

from dataloader import get_dataloaders
from utils import set_random_seeds
from .decoder_from_latents import DecoderFromLatents
from .model import StyledGenerator


def load_generator(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path)
    generator = StyledGenerator()
    generator.load_state_dict(checkpoint['g_running'])
    generator.cuda()
    generator.eval()
    return generator


@torch.no_grad()
def get_mean_style(generator, device='cuda', code_size=512, batches_n=100):
    device = torch.device(device)
    mean_style = None
    batch_size = 1024

    with torch.no_grad():
        for i in range(batches_n):
            style = generator.mean_style(torch.randn(batch_size, code_size).to(device))

            if mean_style is None:
                mean_style = style

            else:
                mean_style += style

    mean_style /= batches_n
    return mean_style


def make_random_latents(batch_size=1, code_dim=512, step=8, device='cuda:0', make_primes=True, mean_style=None,
                        rand_weight=0):
    device = torch.device(device)
    result = {}
    if mean_style is not None:
        result[f'latent_w:0'] = mean_style + torch.randn(batch_size, code_dim).to(device) * rand_weight
    else:
        result[f'latent_w:0'] = torch.randn(batch_size, code_dim).to(device) * rand_weight

    for i in range(step + 1):
        spatial_size = 2 ** (i + 2)
        result[f'noise:{i}'] = torch.randn(batch_size, 1, spatial_size, spatial_size).to(device)
        if make_primes:
            result[f'latent_w_prime:{i}_1'] = result[f'latent_w:0'].clone().detach()
            result[f'latent_w_prime:{i}_2'] = result[f'latent_w:0'].clone().detach()
    return result


def batch2npimages(batch, nrow=None):
    if len(batch) == 1:
        nrow = 1
    elif nrow is None:
        raise ValueError('Need explicit nrow for batch size > 1')
    img = batch.detach().clone()
    img = torchvision.utils.make_grid(img, nrow=nrow)
    return img.add_(1).mul_(127.5).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('../datasets'))
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--generator', type=Path, default=Path('../checkpoints/stylegan-256px.model'))
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--reduce-lr-factor', type=float, default=0.5)
    parser.add_argument('--l2-coef', type=float, default=1.0)
    parser.add_argument('--clip-grad-norm', type=float, default=10.0)
    args = parser.parse_args()
    set_random_seeds(args.seed)

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)-15s] %(levelname)s: %(message)s',
    )
    logging.getLogger('stylegan').setLevel(logging.INFO)
    logging.getLogger('PIL').setLevel(logging.INFO)

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    generator = load_generator(args.generator)

    step = int(np.log2(args.resolution) - 2)
    decoder = DecoderFromLatents(generator, step=step)

    mean_style = get_mean_style(generator)
    latents_keys = list(make_random_latents(step=step, make_primes=True, mean_style=mean_style))
    optimize_names = [name for name in latents_keys if 'prime' in name]

    _, valid_dataloader = get_dataloaders(
        dataset_name='ffhq', data_dir=args.data_dir,
        train_batch_size=1, valid_batch_size=args.batch_size,
        replacement=False,
        batches_per_train_epoch=1, batches_per_valid_epoch=4,
        train_num_workers=1,
        normalize=False, resize_to=256, crop_to=256,
        valid_first_samples=100,
    )

    interrupted = False
    for batch_i, (images_batch, _) in enumerate(valid_dataloader):
        images_batch = images_batch * 2 - 1

        latents_batch = [
            make_random_latents(step=step, make_primes=True, mean_style=mean_style)
            for _ in range(args.batch_size)
        ]

        latents_batch = {
            name: torch.cat([latents[name] for latents in latents_batch])
            for name in latents_keys
        }

        if optimize_names is None:
            optimize_names = list(latents_batch.keys())

        latent_variables = {
            name: nn.Parameter(tensor.clone().detach().to(device))
            for name, tensor in latents_batch.items()
        }

        trainable_parameters = [latent_variables[name] for name in optimize_names]
        optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.iters, factor=args.reduce_lr_factor, verbose=True)

        gold_images = images_batch.to(device)

        try:
            with tqdm.tqdm(range(args.iters), total=args.iters, desc='Iter') as progress_bar:
                for _ in progress_bar:
                    optimizer.zero_grad()

                    pred_images = decoder(latent_variables)

                    coefs = {
                        'l2': args.l2_coef,
                    }

                    losses = {
                        'l2': F.mse_loss(pred_images, gold_images, reduction='none').mean(dim=(1, 2, 3))
                    }

                    loss = torch.stack([
                        coefs[k] * v if coefs[k] != 1 else v
                        for k, v in losses.items()
                    ]).sum(dim=0)
                    loss.sum(dim=0).backward()
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.clip_grad_norm)

                    optimizer.step()
                    loss_val = loss.mean(dim=0).item()
                    lr_scheduler.step(loss_val)

                    progress_bar.set_postfix_str(f'[Batch {batch_i + 1}/{len(valid_dataloader)}] Loss: {loss_val:.3f}')
        except KeyboardInterrupt:
            print('Interrupted')
            interrupted = True

        tuned_latents = {name: tensor.data.clone().detach()
                         for name, tensor in latent_variables.items()}

        opt_state_dict = optimizer.state_dict()

        batch_result = {
            'latents': tuned_latents,
            'images': images_batch,
            'opt_state': opt_state_dict,
        }
        torch.save(batch_result, os.path.join(args.run_dir, f'{batch_i:03d}.pth'))

        if interrupted:
            break


if __name__ == '__main__':
    main()
