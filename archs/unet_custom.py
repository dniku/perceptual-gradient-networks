import os
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import get_conv, get_block, get_norm

try:
    import thop
except ImportError:
    thop = None


class UNetCustom(nn.Module):
    def __init__(self, out_channels=3,
                 block_type='doubleblaze', conv_type='sep', norm_type='batch',
                 down_channels=(64, 64, 128, 256, 512),
                 skip_channels=None,  # equal to down_channels by default
                 up_channels=(64, 128, 256, 256, 512),
                 predict_value=False):
        super().__init__()

        dch = down_channels
        uch = up_channels
        sch = down_channels if skip_channels is None else skip_channels
        top_skip = 64

        assert len(dch) == len(uch)
        n = len(dch)
        assert n >= 1

        self.block_type = block_type
        self.conv_type = conv_type
        self.norm_type = norm_type
        self.down_channels = dch
        self.up_channels = uch
        self.skip_channels = sch
        self.predict_value = predict_value

        block = get_block(block_type)
        conv = get_conv(conv_type)
        norm = get_norm(norm_type)

        # Down
        c1 = torch.nn.Conv2d(
            in_channels=6, out_channels=dch[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        bn1 = norm(dch[0])
        r1 = torch.nn.ReLU(inplace=True)
        self.layer0 = torch.nn.Sequential(c1, bn1, r1)

        if n > 1:
            m1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            dbb1 = block(norm_type, in_channels=dch[0], out_channels=dch[1], stride=1)
            self.layer1 = torch.nn.Sequential(m1, dbb1)

            for i in range(2, n):
                l = block(norm_type, in_channels=dch[i - 1], out_channels=dch[i], stride=2)
                setattr(self, f'layer{i}', l)

        # Skip
        self.conv_original_size0 = conv(6, top_skip, 3, 1)
        self.conv_original_size1 = conv(top_skip, top_skip, 3, 1)
        for i in range(n):
            l = conv(dch[i], sch[i], 1, 0)
            setattr(self, f'layer{i}_1x1', l)

        # Up
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_original_size2 = conv(top_skip + uch[1], uch[0], 3, 1)
        for i in range(1, n):
            in_channels = sch[i - 1] + (uch[i + 1] if i < n - 1 else sch[n - 1])
            l = conv(in_channels, uch[i], 3, 1)
            setattr(self, f'conv_up{i - 1}', l)

        if predict_value:
            for i in range(n):
                l = conv(uch[i], 128, 3, 1)
                setattr(self, f'conv_sobolev_{i}', l)
            self.linear_sobolev = nn.Linear(128 * n, 1)

        # Final
        self.conv_last = nn.Conv2d(uch[0], out_channels, 1)

    def profile(self, layer, inputs, name, profiles, debug=False):
        with open(os.devnull, "w") as null:
            with redirect_stdout(null):
                flops, params = thop.profile(layer, (inputs,))
        if debug:
            print(name, flops / 1e+9, "GFLOPS", params / 1e+6, "M params")
        profiles[name] = (flops, params)

    def forward(self, input, target, profile=False, debug=False):
        profiles = {}
        value_pred_list = []
        inputs = torch.cat([input, target], dim=1)

        if profile: self.profile(self.conv_original_size0, inputs, "conv_original_size0", profiles, debug)
        x_original = self.conv_original_size0(inputs)
        if profile: self.profile(self.conv_original_size1, x_original, "conv_original_size1", profiles, debug)
        x_original = self.conv_original_size1(x_original)

        x = inputs
        skips = [x_original]
        for i in range(len(self.down_channels)):
            layer_down_name = f'layer{i}'
            layer_skip_name = f'layer{i}_1x1'
            layer_down = getattr(self, layer_down_name)
            layer_skip = getattr(self, layer_skip_name)

            if profile: self.profile(layer_down, x, layer_down_name, profiles, debug)
            x = layer_down(x)
            if profile: self.profile(layer_skip, x, layer_skip_name, profiles, debug)
            out_skip = layer_skip(x)

            skips.append(out_skip)

        x = skips.pop()

        for i in reversed(range(len(self.up_channels))):
            layer_conv_name = f'conv_up{i - 1}' if i > 0 else f'conv_original_size2'
            layer_conv = getattr(self, layer_conv_name)

            if i == 5:
                if profile: self.profile(F.interpolate, x, "interpolate", profiles, debug)
                x = F.interpolate(x, (7, 7), mode='bilinear', align_corners=True)
            else:
                if profile: self.profile(self.upsample, x, "upsample", profiles, debug)
                x = self.upsample(x)

            in_skip = skips.pop()
            x = torch.cat([x, in_skip], dim=1)
            if profile: self.profile(layer_conv, x, layer_conv_name, profiles, debug)
            x = layer_conv(x)

            if self.predict_value:
                layer_value_name = f'conv_sobolev_{i}'
                layer_value = getattr(self, layer_value_name)

                if profile: self.profile(layer_value, x, layer_value_name, profiles, debug)
                layer_value_out = layer_value(x).mean(dim=[2, 3])
                value_pred_list.append(layer_value_out)

        assert not skips

        if profile: self.profile(self.conv_last, x, "conv_last", profiles, debug)
        out = self.conv_last(x)

        result = {
            'out': out,
        }

        if self.predict_value:
            value_pred = torch.cat(value_pred_list, dim=1)
            value_pred = self.linear_sobolev(value_pred).squeeze(dim=1)
            result['val'] = value_pred

        if profile:
            result['profiles'] = profiles

        return result
