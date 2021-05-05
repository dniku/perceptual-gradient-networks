import torch
from torch import nn

from archs.blocks import get_block, get_conv


class Autoencoder(nn.Module):
    def __init__(self, block_type, conv_type, max_layer=3):
        super().__init__()

        if max_layer < 0:
            raise ValueError(f"max_layer < 0 is not supported (received {max_layer})")

        block = get_block(block_type)
        conv = get_conv(conv_type)

        layers_down = [
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=3, out_channels=128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
            ),
        ]

        if max_layer > 0:
            layers_down.append(
                torch.nn.Sequential(
                    torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                    block('batch', in_channels=128, out_channels=128, stride=1),
                )
            )

        num_channels = 128
        for _ in range(2, max_layer + 1):
            layers_down.append(block('batch', in_channels=num_channels, out_channels=num_channels * 2, stride=2))
            num_channels *= 2

        layer_mid = conv(num_channels, num_channels, 1, 0)

        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        layers_up_rev = [
            nn.Conv2d(128, 3, 1),
            conv(128, 128, 3, 1),
            upsample,
        ]

        num_channels = 128
        for i in range(max_layer):
            next_num_channels = num_channels * 2 if i != max_layer - 1 else num_channels
            layers_up_rev.append(conv(next_num_channels, num_channels, 3, 1))
            layers_up_rev.append(upsample)
            num_channels = next_num_channels

        self.model = nn.Sequential(
            *layers_down,
            layer_mid,
            *layers_up_rev[::-1]
        )

    def forward(self, input):
        return self.model(input)
