import functools

import torch
from torch import nn


def conv_norm_act_full(norm_layer, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        norm_layer(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    ]


# Borrowed from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/29bbc96/models/networks.py#L315-L433
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=6, output_nc=3,
                 down_channels=(64, 128), up_channels=(64,),
                 conv_layer=conv_norm_act_full, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, num_blocks=8, padding_type='zero'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            down_channels (int) -- the number of filters in the initial downsampling layers
            up_channels (int)   -- the number of filters in the final upsampling layers
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            num_blocks (int)    -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(ResnetGenerator, self).__init__()

        assert num_blocks >= 0

        # Use bias only with InstanceNorm
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Initial layer
        model = conv_layer(norm_layer, input_nc, down_channels[0], bias=use_bias)

        in_ch = down_channels[0]

        # Downsampling layers
        for out_ch in down_channels[1:]:
            model += conv_layer(norm_layer, in_ch, out_ch, stride=2, bias=use_bias)
            in_ch = out_ch

        # ResNet blocks
        for i in range(num_blocks):
            model += [
                ResnetBlock(
                    in_ch, padding_type=padding_type,
                    conv_layer=conv_layer, norm_layer=norm_layer, use_dropout=use_dropout,
                    use_bias=True,
                )]

        # Upsampling layers
        for out_ch in up_channels:
            model += [nn.Upsample(scale_factor=2, mode='nearest')]
            model += conv_layer(norm_layer, in_ch, out_ch, bias=use_bias)
            in_ch = out_ch

        # Final layer
        model += [nn.Conv2d(in_ch, output_nc, kernel_size=3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input, target):
        """Standard forward"""
        inputs = torch.cat([input, target], dim=1)
        out = self.model(inputs)
        return {
            'out': out,
        }


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, conv_layer, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, conv_layer, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, conv_layer, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += conv_layer(norm_layer, dim, dim, padding=p, bias=use_bias)
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
