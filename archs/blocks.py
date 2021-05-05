import math

import torch
from torch import nn as nn


def get_conv(conv_type):
    return {
        'full': full_convrelu,
        'sep': sep_convrelu,
    }[conv_type]


def full_convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


def sep_convrelu(in_channels, out_channels, kernel, padding):
    groups = math.gcd(in_channels, out_channels)

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, groups=groups),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
        nn.ReLU(inplace=True),
    )


def get_block(block_type):
    return {
        'singleres': SingleResBlock,
        'doubleres': DoubleResBlock,
        'singleblaze': SingleBlazeBlock,
        'doubleblaze': DoubleBlazeBlock,
    }[block_type]


def get_norm(norm_type):
    return {
        'batch': nn.BatchNorm2d,
        'instance': nn.InstanceNorm2d,
    }[norm_type]



class SingleResBlock(torch.nn.Module):
    def __init__(self, norm_type, in_channels, out_channels, stride=1):
        super(SingleResBlock, self).__init__()
        norm = get_norm(norm_type)
        self.residc = False
        self.c1 = torch.nn.Conv2d(
            kernel_size=3, padding=1, stride=stride, bias=False,
            in_channels=in_channels, out_channels=out_channels)
        self.c2 = torch.nn.Conv2d(
            kernel_size=3, padding=1, stride=1, bias=False,
            in_channels=out_channels, out_channels=out_channels)
        self.r1 = torch.nn.ReLU()
        self.r2 = torch.nn.ReLU()
        self.bn1 = norm(num_features=out_channels)
        self.bn2 = norm(num_features=out_channels)
        self.resid = True
        if stride == 2:
            self.residc = True
            self.cr1 = torch.nn.Conv2d(
                kernel_size=1, padding=0, stride=stride, bias=False,
                in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        res = self.c1(x)
        res = self.bn1(res)
        res = self.r1(res)
        res = self.c2(res)
        res = self.bn2(res)
        if self.resid:
            if self.residc:
                resid = self.cr1(x)
            else:
                resid = x
            res = res + resid
        res = self.r2(res)
        return res


class DoubleResBlock(torch.nn.Module):
    def __init__(self, norm_type, in_channels, out_channels, stride=1):
        super(DoubleResBlock, self).__init__()
        norm = get_norm(norm_type)
        self.residc = False
        self.c11 = torch.nn.Conv2d(
            kernel_size=3, padding=1, stride=stride, bias=False,
            in_channels=in_channels, out_channels=out_channels)
        self.c12 = torch.nn.Conv2d(
            kernel_size=3, padding=1, stride=1, bias=False,
            in_channels=out_channels, out_channels=out_channels)
        self.r11 = torch.nn.ReLU()
        self.r12 = torch.nn.ReLU()
        self.bn11 = norm(num_features=out_channels)
        self.bn12 = norm(num_features=out_channels)
        self.resid = True
        if stride == 2:
            self.residc = True
            self.cr11 = torch.nn.Conv2d(
                kernel_size=1, padding=0, stride=stride, bias=False,
                in_channels=in_channels, out_channels=out_channels)

        self.c21 = torch.nn.Conv2d(
            kernel_size=3, padding=1, stride=1, bias=False,
            in_channels=out_channels, out_channels=out_channels)
        self.c22 = torch.nn.Conv2d(
            kernel_size=3, padding=1, stride=1, bias=False,
            in_channels=out_channels, out_channels=out_channels)
        self.r21 = torch.nn.ReLU()
        self.r22 = torch.nn.ReLU()
        self.bn21 = norm(num_features=out_channels)
        self.bn22 = norm(num_features=out_channels)
        self.resid = True
        if stride == 2:
            self.residc = True
            self.cr21 = torch.nn.Conv2d(
                kernel_size=1, padding=0, stride=1, bias=False, in_channels=out_channels,
                out_channels=out_channels)

    def forward(self, x):
        res = self.c11(x)
        res = self.bn11(res)
        res = self.r11(res)
        res = self.c12(res)
        res = self.bn12(res)
        if self.resid:
            if self.residc:
                resid = self.cr11(x)
            else:
                resid = x
            res = res + resid
        res = self.r12(res)

        x = res

        res = self.c21(res)
        res = self.bn21(res)
        res = self.r21(res)
        res = self.c22(res)
        res = self.bn22(res)
        if self.resid:
            if self.residc:
                resid = self.cr21(x)
            else:
                resid = x
            res = res + resid
        res = self.r22(res)
        return res


class SingleBlazeBlock(torch.nn.Module):
    def __init__(self, norm_type, in_channels, out_channels, stride=1):
        super(SingleBlazeBlock, self).__init__()
        norm = get_norm(norm_type)
        self.residc = False
        self.cdw1 = torch.nn.Conv2d(
            kernel_size=5, padding=2, stride=stride, bias=False,
            in_channels=in_channels, out_channels=out_channels, groups=in_channels)
        self.cpw1 = torch.nn.Conv2d(
            kernel_size=1, padding=0, stride=1, bias=False,
            in_channels=out_channels, out_channels=out_channels, groups=1)
        self.r1 = torch.nn.ReLU()
        self.bn1 = norm(num_features=out_channels)
        self.resid = True
        if stride == 2:
            self.residc = True
            self.c1 = torch.nn.Conv2d(
                kernel_size=1, padding=0, stride=stride, bias=False,
                in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        res = self.cdw1(x)
        res = self.cpw1(res)
        res = self.bn1(res)
        if self.resid:
            if self.residc:
                resid = self.c1(x)
            else:
                resid = x
            res = res + resid
        res = self.r1(res)
        return res


class DoubleBlazeBlock(torch.nn.Module):
    def __init__(self, norm_type, in_channels, out_channels, stride=1):
        super(DoubleBlazeBlock, self).__init__()
        norm = get_norm(norm_type)
        self.residc = False
        self.cdw1 = torch.nn.Conv2d(
            kernel_size=5, padding=2, stride=stride, bias=False,
            in_channels=in_channels, out_channels=out_channels, groups=in_channels)
        self.cpw1 = torch.nn.Conv2d(
            kernel_size=1, padding=0, stride=1, bias=False,
            in_channels=out_channels, out_channels=out_channels, groups=1)
        self.cdw2 = torch.nn.Conv2d(
            kernel_size=5, padding=2, stride=1, bias=False,
            in_channels=out_channels, out_channels=out_channels, groups=out_channels)
        self.cpw2 = torch.nn.Conv2d(
            kernel_size=1, padding=0, stride=1, bias=False,
            in_channels=out_channels, out_channels=out_channels, groups=1)
        self.r1 = torch.nn.ReLU()
        self.r2 = torch.nn.ReLU()
        self.bn1 = norm(num_features=out_channels)
        self.bn2 = norm(num_features=out_channels)
        self.resid = True
        if stride == 2:
            self.residc = True
            self.c1 = torch.nn.Conv2d(
                kernel_size=1, padding=0, stride=stride, bias=False,
                in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        res = self.cdw1(x)
        res = self.cpw1(res)
        res = self.bn1(res)
        res = self.r1(res)
        res = self.cdw2(res)
        res = self.cpw2(res)
        res = self.bn2(res)
        if self.resid:
            if self.residc:
                resid = self.c1(x)
            else:
                resid = x
            res = res + resid
        res = self.r2(res)
        return res
