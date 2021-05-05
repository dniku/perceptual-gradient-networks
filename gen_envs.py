import logging
import numpy as np
from pathlib import Path

import torch
from torch import nn

from archs.autoencoder import Autoencoder


class AutoencoderEnv(nn.Module):
    def __init__(self, ae_block_type, ae_conv_type, max_layer, ckpt_path: Path = None):
        super().__init__()

        self.autoencoder = Autoencoder(ae_block_type, ae_conv_type, max_layer)
        if ckpt_path is not None:
            logging.debug(f'Loading autoencoder checkpoint from {ckpt_path}...')
            state_dict = torch.load(ckpt_path, map_location='cpu')
            self.autoencoder.load_state_dict(state_dict)

        self.opt = None
        self.gen = None

    def init_opt(self, opt_ckpt_path: Path = None):
        self.opt = torch.optim.Adam(self.autoencoder.parameters())
        if opt_ckpt_path is not None:
            logging.debug(f'Loading autoencoder optimizer checkpoint from {opt_ckpt_path}...')
            opt_state_dict = torch.load(opt_ckpt_path, map_location='cpu')
            self.opt.load_state_dict(opt_state_dict)

    def reset(self):
        def reset_parameters(m):
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                m.reset_parameters()

        self.autoencoder.apply(reset_parameters)

    def step(self, grad, max_grad_norm=None):
        assert self.gen is not None
        self.opt.zero_grad()
        self.gen.backward(grad)
        if max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_grad_norm)
        else:
            grad_norm = None
        self.opt.step()
        self.gen = None
        return grad_norm

    def forward(self, gt):
        self.gen = self.autoencoder(gt)
        return self.gen.detach()

    def state_dicts(self, *args, **kwargs):
        return self.autoencoder.state_dict(*args, **kwargs), self.opt.state_dict()


class MixupEnv(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, gt):
        mixup_coefs = torch.tensor(
            np.random.beta(self.beta, self.beta, size=(gt.shape[0], 1, 1, 1)),
            dtype=torch.float32,
            device=gt.device,
        )
        p = np.random.permutation(gt.shape[0])
        gen = mixup_coefs * gt + (1 - mixup_coefs) * gt[p, :, :, :]
        return gen


class MixupZeroEnv(nn.Module):
    def forward(self, gt):
        return gt
