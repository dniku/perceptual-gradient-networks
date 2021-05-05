import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PerceptualLoss(nn.Module):
    def __init__(self, model='vgg19', weights_source='pytorch', normalize_from=None):
        super().__init__()
        self.normalize_from = normalize_from

        self.num_layers = {
            'vgg19': 30,
            'vgg11': 18,
        }[model]

        if weights_source == 'pytorch':
            if model == 'vgg19':
                vgg = torchvision.models.vgg19(pretrained=True).features
            elif model == 'vgg11':
                vgg = torchvision.models.vgg11(pretrained=True).features
            self.mean_ = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1)
            self.std_ = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1)
        elif weights_source == 'caffe':
            assert model == 'vgg19'
            vgg_weights = torch.load(
                '/group-volume/orc_srr/violet/ivakhnenko.a/exp/gans_on_mobile/pretrained_weights/vgg19_imagenet_caffe.pth'
            )
            model = torchvision.models.vgg19()
            model.load_state_dict(vgg_weights)
            vgg = model.features
            self.mean_ = torch.tensor([103.939, 116.779, 123.680], dtype=torch.float32).reshape(1, 3, 1, 1) / 255
            self.std_ = torch.tensor([1., 1., 1.], dtype=torch.float32).reshape(1, 3, 1, 1) / 255
        else:
            assert False

        vgg_avg_pooling = []

        for weights in vgg.parameters():
            weights.requires_grad = False

        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)

        self.vgg = nn.Sequential(*vgg_avg_pooling[:self.num_layers])

    def norm_mean_std(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def forward(self, input, target):
        target = target.detach()

        if self.normalize_from is not None:
            lo, hi = self.normalize_from
            input = self.norm_mean_std((input - lo) / (hi - lo))
            target = self.norm_mean_std((target - lo) / (hi - lo))

        losses = []
        for layer in self.vgg:
            input = layer(input)
            target = layer(target)

            if layer.__class__.__name__ == 'ReLU':
                loss = F.l1_loss(input, target, reduction='none').mean(dim=(1, 2, 3))
                losses.append(loss)

        return torch.stack(losses).sum(dim=0)

    def perceptual_grads(self, input, target):
        # 771 Mb for 1 call with batch_size=1
        # 5703 Mb for 1 call with batch_size=32
        input = input.detach().requires_grad_()
        with torch.enable_grad():
            batchwise_loss = self.forward(input, target)
            batchwise_loss.backward(torch.ones_like(batchwise_loss))
        return input.grad


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class PerceptualLossDeep(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=False, device=torch.device('cpu')):
        super().__init__()
        self.extractor = VGGFeatureExtractor(feature_layer, use_bn, use_input_norm, device)

    def forward(self, input, target):
        features_input = self.extractor(input)
        features_target = self.extractor(target)

        return F.l1_loss(features_input, features_target, reduction='none').mean(dim=(1, 2, 3))
