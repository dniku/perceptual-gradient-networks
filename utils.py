import random
import time
from pathlib import Path
from typing import Optional, Union

import imageio_ffmpeg
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class VideoWriter:
    def __init__(self, path: Path, fps: float = 30.0, codec: str = 'libx264', bgr2rgb=False):
        self.path = path
        self.fps = float(fps)
        self.codec = codec
        self.out = None
        self.frame_size = None
        self.bgr2rgb = bgr2rgb

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.out is not None:
            self.out.close()

    def write(self, frame: np.ndarray):
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        frame_size = (frame.shape[1], frame.shape[0])
        if self.out is None:
            self.path.parent.mkdir(exist_ok=True, parents=True)
            self.frame_size = frame_size
            self.out = imageio_ffmpeg.write_frames(
                str(self.path), self.frame_size,
                fps=self.fps, codec=self.codec,
                macro_block_size=1, ffmpeg_log_level='error',
            )
            self.out.send(None)
        else:
            assert self.frame_size == frame_size, f"Wrong frame size: should be {self.frame_size}, got {frame_size}"
        if self.bgr2rgb:
            frame = frame[:, :, (2, 1, 0)]
        self.out.send(np.ascontiguousarray(frame))


class Normalizer(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()

        if mean is None:
            self.mean = None
        else:
            self.register_buffer('mean', mean)

        if std is None:
            self.std = None
        else:
            self.register_buffer('std', std)

    def forward(self, tensor):
        if self.mean is not None:
            tensor = tensor - self.mean
        if self.std is not None:
            tensor = tensor / self.std
        return tensor

    def backward(self, tensor):
        if self.std is not None:
            tensor = tensor / self.std
        return tensor

    def inverse(self, tensor):
        if self.std is not None:
            tensor = tensor * self.std
        if self.mean is not None:
            tensor = tensor + self.mean
        return tensor

    @classmethod
    def make(cls, kind='vgg'):
        if kind == 'vgg':
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1)
        elif kind == 'caffe':
            mean = torch.tensor([103.939, 116.779, 123.680], dtype=torch.float32).reshape(1, 3, 1, 1) / 255
            std = torch.tensor([1., 1., 1.], dtype=torch.float32).reshape(1, 3, 1, 1) / 255
        elif kind == 'none':
            mean = std = None
        else:
            assert False
        return cls(mean, std)


def percentile(t: torch.Tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    Source: https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    b = t.shape[0]
    t = t.reshape(b, -1)
    if q == 0:
        return torch.min(t, dim=-1).values
    elif q == 100:
        return torch.max(t, dim=-1).values
    elif q == 50:
        return torch.median(t, dim=-1).values
    else:
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(.01 * float(q) * (t.shape[1] - 1))
        result = t.kthvalue(k, dim=-1).values
        return result


def torch_image_to_numpy(image_torch):
    """Convert PyTorch tensor to Numpy array.
    :param image_torch: PyTorch float CHW Tensor in range [0..1].
    :returns: Numpy uint8 HWC array in range [0..255]."""
    return image_torch.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()


def torch_batch_to_numpy(batch_torch, nrow=8, normalizer=None):
    if normalizer is not None:
        batch_torch = normalizer.inverse(batch_torch)
    return torch_image_to_numpy(torchvision.utils.make_grid(batch_torch, nrow=nrow))


def make_frame(normalizer, gt, gen, grads_true=None, grads_pred=None, proxy_true=None, proxy_pred=None, gen_prc=None, gen_mse=None, nrow=8):
    grid_elems = [
        [normalizer.inverse(gt)],
        [normalizer.inverse(gen)],
    ]

    if grads_true is not None:
        assert grads_pred is not None
        grads_true_min = grads_true.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        grads_true_max = grads_true.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        grads_true_unit = ((grads_true - grads_true_min) / (grads_true_max - grads_true_min + 1e-8)).clamp(0, 1)
        grads_pred_unit = ((grads_pred - grads_true_min) / (grads_true_max - grads_true_min + 1e-8)).clamp(0, 1)

        grid_elems[0].append(grads_true_unit)
        grid_elems[1].append(grads_pred_unit)

    grid_elems[0].append(normalizer.inverse(proxy_true) if proxy_true is not None else torch.zeros_like(gen))
    grid_elems[1].append(normalizer.inverse(proxy_pred) if proxy_pred is not None else torch.zeros_like(gen))

    if gen_prc is not None or gen_mse is not None:
        grid_elems[0].append(normalizer.inverse(gen_prc) if gen_prc is not None else torch.zeros_like(gen))
        grid_elems[1].append(normalizer.inverse(gen_mse) if gen_mse is not None else torch.zeros_like(gen))

    grid = torch.cat([
        torch.cat([torchvision.utils.make_grid(elem, nrow=nrow) for elem in row], dim=2)
        for row in grid_elems
    ], dim=1)

    return torch_image_to_numpy(grid)


class MovingAverage:
    def __init__(self, initial_value, new_weight=0.01):
        self.value = initial_value
        self.new_weight = new_weight

    def get(self):
        return self.value

    def update(self, new_value):
        self.value = (1 - self.new_weight) * self.value + self.new_weight * new_value


def l1_loss_batchwise(input, target):
    return F.l1_loss(input, target, reduction='none').mean(dim=(1, 2, 3))


def mse_loss_batchwise(input, target):
    return F.mse_loss(input, target, reduction='none').mean(dim=(1, 2, 3))


def logcosh_loss_batchwise(input, target):
    # Implementation borrowed from Keras:
    # https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/losses.py#L918-L940
    def _logcosh(x):
        return x + F.softplus(-2. * x) - np.log(2.)

    return _logcosh(input - target).mean(dim=(1, 2, 3))


def logit(p, eps=1e-8):
    return -torch.log((1 / p.clamp(min=eps) - 1).clamp(min=eps))


class MseLogitLossBatchwise:
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, input, target):
        return mse_loss_batchwise(
            logit(self.normalizer.inverse(input)),
            logit(self.normalizer.inverse(target)),
        )


class LogcoshLogitLossBatchwise:
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, input, target):
        return logcosh_loss_batchwise(
            logit(self.normalizer.inverse(input)),
            logit(self.normalizer.inverse(target)),
        )


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_grad_norms(model, norm_type=2):
    result = {}
    for field_name in dir(model):
        m = getattr(model, field_name)
        if not isinstance(m, nn.Module):
            continue
        norms = [
            p.grad.data.norm(norm_type) ** norm_type
            for p in m.parameters()
            if p.grad is not None
        ]
        if norms:
            result[field_name] = (sum(norms) ** (1. / norm_type)).item()
    return result


class ReplayBuffer:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.storage = []
        self.total_items = 0

    def add(self, item):
        # Reservoir sampling
        self.total_items += 1
        if len(self.storage) < self.maxsize:
            self.storage.append(item)
        else:
            if np.random.random() < len(self.storage) / self.total_items:
                self.storage[np.random.randint(0, len(self.storage))] = item

    def add_batch(self, batch_gen, batch_gt, batch_grads_true):
        for gen, gt, grads_true in zip(batch_gen, batch_gt, batch_grads_true):
            self.add((gen.cpu().detach(), gt.cpu().detach(), grads_true.cpu().detach()))

    def get(self, size):
        return [self.storage[i] for i in np.random.randint(0, len(self.storage), size=size)]


class Timer:
    def __init__(self):
        self.time_start = None
        self.time_end = None

    def __enter__(self):
        self.time_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_end = time.time()

    def time(self):
        return self.time_end - self.time_start


class ProcessGroup:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        dist.init_process_group(*self.args, **self.kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        dist.destroy_process_group()


class DistributedSummaryWrapper:
    def __init__(self, writer: Optional[SummaryWriter], dst=0):
        self.writer = writer
        self.dst = dst

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, op='mean'):
        t = torch.tensor(scalar_value).cuda()
        if op == 'mean':
            t = t.to(torch.float32)
            dist.reduce(t, dst=self.dst, op=dist.ReduceOp.SUM)
        else:
            dist.reduce(t, dst=self.dst, op=op)
        if dist.get_rank() == self.dst:
            value = t.item()
            if op == 'mean':
                value /= dist.get_world_size()
            self.writer.add_scalar(tag, value, global_step, walltime)

    def add_histogram(self, *args, **kwargs):
        raise NotImplementedError
