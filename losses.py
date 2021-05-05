import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from archs.perceptual_loss import PerceptualLoss
from utils import mse_loss_batchwise


class ScaledMSELoss(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.input_scale = np.sqrt(scale)

    def forward(self, gen, gt, grads_pred, grads_true):
        return mse_loss_batchwise(self.input_scale * grads_pred, self.input_scale * grads_true)


class VggOneStepLoss(nn.Module):
    def __init__(self, ploss: PerceptualLoss, lr=0.1):
        super().__init__()
        self.ploss = ploss
        self.lr = lr

    def forward(self, gen, gt, grads_pred, grads_true):
        return self.ploss(gen + self.lr * grads_pred, gen + self.lr * grads_true)


class VggGradLoss(nn.Module):
    def __init__(self, ploss):
        super().__init__()
        self.ploss = ploss

    def forward(self, gen, gt, grads_pred, grads_true):
        return self.ploss(grads_pred, grads_true)


def symmetric_jacobian_loss(y, x, norm_type=1):
    v = torch.randn_like(y)
    v /= v.norm(norm_type)
    v.requires_grad_()

    vjp = torch.autograd.grad(y, x, v, create_graph=True, only_inputs=True)[0]
    jvp = torch.autograd.grad(vjp, v, v, create_graph=True, only_inputs=True)[0]

    return (vjp - jvp).norm(norm_type, dim=(1, 2, 3))


def total_variation_loss(t):
    v = mse_loss_batchwise(t[:, :, :-1, :], t[:, :, 1:, :])
    h = mse_loss_batchwise(t[:, :, :, :-1], t[:, :, :, 1:])
    return v + h
