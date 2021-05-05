from torch import nn

from .common import Concat, conv, bn, act

default_skip_params = dict(
    num_input_channels=3, num_output_channels=3,
    num_channels_down=(8, 16, 32, 64, 128),
    num_channels_up=(8, 16, 32, 64, 128),
    num_channels_skip=(0, 0, 0, 4, 4),
    upsample_mode='bilinear',
    need_sigmoid=True,
    pad='reflection', act_fun='LeakyReLU',
)


def skip(
    num_input_channels=2, num_output_channels=3,
    num_channels_down=(16, 32, 64, 128, 128), num_channels_up=(16, 32, 64, 128, 128), num_channels_skip=(4, 4, 4, 4, 4),
    filter_size_down=3, filter_size_up=3, filter_skip_size=1,
    need_sigmoid=True, need_bias=True,
    pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
    need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not isinstance(upsample_mode, (tuple, list)):
        upsample_mode = [upsample_mode] * n_scales

    if not isinstance(downsample_mode, (tuple, list)):
        downsample_mode = [downsample_mode] * n_scales

    if not isinstance(filter_size_down, (tuple, list)):
        filter_size_down = [filter_size_down] * n_scales

    if not isinstance(filter_size_up, (tuple, list)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_curr_depth = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_curr_depth.add(Concat(1, skip, deeper))
        else:
            model_curr_depth.add(deeper)

        model_curr_depth.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        model_next_depth = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(model_next_depth)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=True))

        model_curr_depth.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_curr_depth.add(bn(num_channels_up[i]))
        model_curr_depth.add(act(act_fun))

        if need1x1_up:
            model_curr_depth.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_curr_depth.add(bn(num_channels_up[i]))
            model_curr_depth.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_curr_depth = model_next_depth

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
