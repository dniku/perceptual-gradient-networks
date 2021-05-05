from pathlib import Path
import logging

import torch
from torch import nn
import torch.nn.functional as F

from archs.resnet_generator import ResnetGenerator
from archs.unet_custom import UNetCustom
from utils import \
    mse_loss_batchwise, l1_loss_batchwise, logcosh_loss_batchwise, \
    MseLogitLossBatchwise, LogcoshLogitLossBatchwise, \
    Normalizer


def read_checkpoint(checkpoint_path: Path, backbone_type):
    logging.debug(f'Loading PGN checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    checkpoint_version = checkpoint.get('version', 0)
    logging.debug(f'Loaded PGN checkpoint with version {checkpoint_version}')
    logging.debug(f'Fields that are not Tensors: {[k for k, v in checkpoint.items() if not isinstance(v, torch.Tensor)]}')

    if checkpoint_version == 0:
        backbone_params_saved = {k: v for k, v in checkpoint.items() if not isinstance(v, torch.Tensor)}
        backbone_to_grad_params_saved = {}

        if 'grad_scale' in backbone_params_saved:
            grad_scale = backbone_params_saved.pop('grad_scale')
            if backbone_type == 'unet' and grad_scale == 1e-3:
                # The constant 48 is due to the fact that I used to not have batch reduction in the MSE gradient
                # computation, and I have a whole lot of checkpoints trained with this batch size.
                grad_scale = 0.048
            backbone_to_grad_params_saved['grad_scale'] = grad_scale

        return {
            'state_dict': {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)},
            'backbone_type': None,
            'backbone_params': backbone_params_saved,
            'backbone_to_grad_type': None,
            'backbone_to_grad_params': backbone_to_grad_params_saved,
        }
    elif checkpoint_version == 1:
        return {
            'state_dict': checkpoint['state_dict'],
            'backbone_type': None,
            'backbone_params': checkpoint['params'],
            'backbone_to_grad_type': None,
            'backbone_to_grad_params': {},
        }
    elif checkpoint_version == 2:
        return checkpoint
    else:
        raise RuntimeError(f"Checkpoint version {checkpoint_version} is not supported.")


class Pgn(nn.Module):
    def __init__(self, normalizer=None,
                 backbone_type=None, backbone_params=None,
                 backbone_to_grad_type=None, backbone_to_grad_params=None,
                 ignore_grad_scale_mismatch=False,
                 checkpoint_path=None):
        super().__init__()

        if normalizer is None:
            normalizer = Normalizer.make('vgg')

        if backbone_params is None:
            backbone_params = {}
        if backbone_to_grad_params is None:
            backbone_to_grad_params = {}

        logging.debug('Args contain the following parameters:\n' + '\n'.join([
            f'    backbone_type: {backbone_type}',
            f'    backbone_params: {backbone_params}',
            f'    backbone_to_grad_type: {backbone_to_grad_type}',
            f'    backbone_to_grad_params: {backbone_to_grad_params}',
        ]))

        model_state_dict = None
        if checkpoint_path is not None:
            checkpoint = read_checkpoint(checkpoint_path, backbone_type)

            logging.debug('Read checkpoint with the following parameters:\n' + '\n'.join([
                f'    backbone_type: {checkpoint["backbone_type"]}',
                f'    backbone_params: {checkpoint["backbone_params"]}',
                f'    backbone_to_grad_type: {checkpoint["backbone_to_grad_type"]}',
                f'    backbone_to_grad_params: {checkpoint["backbone_to_grad_params"]}',
            ]))

            if backbone_type is None:
                backbone_type = checkpoint['backbone_type']
            elif checkpoint['backbone_type'] is not None:
                assert backbone_type == checkpoint['backbone_type'], (backbone_type, checkpoint['backbone_type'])

            if backbone_to_grad_type is None:
                backbone_to_grad_type = checkpoint['backbone_to_grad_type']
            elif checkpoint['backbone_to_grad_type'] is not None:
                assert backbone_to_grad_type == checkpoint['backbone_to_grad_type'], (backbone_to_grad_type, checkpoint['backbone_to_grad_type'])

            for key in (set(checkpoint['backbone_params'].keys()) & set(backbone_params)):
                value_ckpt = checkpoint['backbone_params'][key]
                value_args = backbone_params[key]
                assert value_args == value_ckpt, (key, value_args, value_ckpt)
            backbone_params.update(checkpoint['backbone_params'])

            for key in (set(checkpoint['backbone_to_grad_params'].keys()) & set(backbone_to_grad_params)):
                value_ckpt = checkpoint['backbone_to_grad_params'][key]
                value_args = backbone_to_grad_params[key]
                if key == 'grad_scale' and value_args != value_ckpt and ignore_grad_scale_mismatch:
                    logging.warning(f'grad_scale mismatch: provided {value_args}, but checkpoint has {value_ckpt}')
                    checkpoint['backbone_to_grad_params'].pop('grad_scale')  # safe since we're iterating over a copy
                else:
                    assert value_args == value_ckpt, (key, value_args, value_ckpt)
            backbone_to_grad_params.update(checkpoint['backbone_to_grad_params'])

            logging.debug('Final checkpoint parameters:\n' + '\n'.join([
                f'    backbone_type: {backbone_type}',
                f'    backbone_params: {backbone_params}',
                f'    backbone_to_grad_type: {backbone_to_grad_type}',
                f'    backbone_to_grad_params: {backbone_to_grad_params}',
            ]))

            model_state_dict = checkpoint['state_dict']

        assert backbone_type is not None
        assert backbone_to_grad_type is not None

        self.backbone = {
            'unet': UNetCustom,
            'resnet': ResnetGenerator,
        }[backbone_type](**backbone_params)

        proxy_type = backbone_to_grad_params['type']
        proxy_params = backbone_to_grad_params[proxy_type]
        make_proxy = {
            'raw': ProxyRaw,
            'sigmoid': ProxyAsSigmoid,
            'warped_target': ProxyAsWarpedTarget,
        }[proxy_type](normalizer, **proxy_params)

        if backbone_to_grad_type == 'direct':
            self.backbone_to_grad = PgnPredictGrad(
                make_proxy,
                backbone_to_grad_params['out_scale'],
                backbone_to_grad_params['grad_scale'],
            )
        elif backbone_to_grad_type == 'proxy':
            batchwise_loss_func = {
                'mse': mse_loss_batchwise,
                'l1': l1_loss_batchwise,
                'logcosh': logcosh_loss_batchwise,
                'mse_logit': MseLogitLossBatchwise(normalizer),
                'logcosh_logit': LogcoshLogitLossBatchwise(normalizer),
            }[backbone_to_grad_params['grad_type']]

            self.backbone_to_grad = PgnProxyToGrad(
                make_proxy,
                batchwise_loss_func,
                backbone_to_grad_params['grad_scale'],
            )
        else:
            assert False

        self.backbone_type = backbone_type
        self.backbone_params = backbone_params
        self.backbone_to_grad_type = backbone_to_grad_type
        self.backbone_to_grad_params = backbone_to_grad_params

        if model_state_dict is not None:
            self.backbone.load_state_dict(model_state_dict)

    def forward(self, input, target, **kwargs):
        pred = self.backbone(input, target, **kwargs)
        pred.update(self.backbone_to_grad(pred['out'], input, target))
        return pred

    def get_checkpoint(self):
        return {
            'state_dict': self.backbone.state_dict(),
            'backbone_type': self.backbone_type,
            'backbone_params': self.backbone_params,
            'backbone_to_grad_type': self.backbone_to_grad_type,
            'backbone_to_grad_params': self.backbone_to_grad_params,
            'version': 2,
        }


class PgnPredictGrad(nn.Module):
    def __init__(self, make_proxy, out_scale=None, grad_scale=None):
        super().__init__()
        self.make_proxy = make_proxy
        self.out_scale = out_scale
        self.grad_scale = grad_scale

    def forward(self, out, input, target):
        if self.out_scale is not None and self.out_scale != 1.0:
            out = out / self.out_scale
        result = self.make_proxy(input - out, target)
        proxy = result['proxy']
        _, c, h, w = input.shape
        grad = input - proxy  # MSE gradient
        grad_coef = 2 / (c * h * w)
        if self.grad_scale is not None and self.grad_scale != 1.0:
            grad_coef /= self.grad_scale
        grad = grad * grad_coef
        result['grad'] = grad
        return result


class PgnProxyToGrad(nn.Module):
    def __init__(self, make_proxy, batchwise_loss_func, grad_scale=None):
        super().__init__()
        self.make_proxy = make_proxy
        self.batchwise_loss_func = batchwise_loss_func
        self.grad_scale = grad_scale

    def forward(self, out, input, target):
        result = self.make_proxy(out, target)
        proxy = result['proxy']
        with torch.enable_grad():
            input_ = input.detach().requires_grad_()
            batchwise_loss = self.batchwise_loss_func(proxy, input_)
            grad = torch.autograd.grad(batchwise_loss.sum(dim=0), input_, create_graph=True)[0]
        if self.grad_scale is not None and self.grad_scale != 1.0:
            grad = grad / self.grad_scale
        result['grad'] = grad
        return result


class ProxyRaw(nn.Module):
    def __init__(self, normalizer):
        super().__init__()
        pass

    def forward(self, out, target):
        return {
            'proxy': out,
        }


class ProxyAsSigmoid(nn.Module):
    def __init__(self, normalizer, scale: float):
        super().__init__()
        self.normalizer = normalizer
        self.scale = scale

    def forward(self, out, target):
        return {
            'proxy': self.scale * self.normalizer(torch.sigmoid(out)),
        }


class ProxyAsWarpedTarget(nn.Module):
    def __init__(self, normalizer, scale: float, additive: bool, downscale_by: float, additive_scale: float):
        super().__init__()
        self.normalizer = normalizer
        self.scale = scale
        self.use_additive = additive
        self.downscale_by = downscale_by
        self.additive_scale = additive_scale

    def forward(self, out, target):
        b, _, h, w = target.shape
        grid_identity = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, h, device=target.device, dtype=target.dtype),
            torch.linspace(-1, 1, w, device=target.device, dtype=target.dtype),
        )[::-1], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

        if self.downscale_by is not None:
            ch, cw = out.shape[2:]
            nh, nw = round(ch / self.downscale_by), round(cw / self.downscale_by)
            out = F.interpolate(out, (nh, nw), mode='bilinear', align_corners=False)

        ch, cw = out.shape[2:]
        if (ch, cw) != (h, w):
            out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)

        if self.use_additive:
            grid_correction = out[:, :2, :, :]
            additive = out[:, 2:, :, :]

            additive = self.additive_scale * self.normalizer(torch.sigmoid(additive))
        else:
            grid_correction = out

        grid_correction = self.scale * torch.tanh(grid_correction)
        grid_correction = grid_correction.permute(0, 2, 3, 1)
        grid = torch.clamp(grid_identity + grid_correction, min=-1, max=1)

        proxy = F.grid_sample(target, grid, align_corners=False)
        if self.use_additive:
            proxy = proxy + additive
            # Here, I could choose to clamp the proxy to the range of the target image,
            # but I don't do that for the same reason as introducing --pgn-proxy-sigmoid-scale.

        result = {
            'proxy': proxy,
            'grid': grid_correction,
        }
        if self.use_additive:
            result['additive'] = additive

        return result
