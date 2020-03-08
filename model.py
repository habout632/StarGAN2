import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# class ResidualBlock(nn.Module):
#     """
#     Residual Block with instance normalization.
#     """
#
#     def __init__(self, dim_in, dim_out):
#         super(ResidualBlock, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
#         )
#         self.bn = nn.BatchNorm2d()
#         self.relu1 = nn.ReLU()
#         self.conv1 = nn.Conv2d()
#
#         self.bn = nn.BatchNorm2d()
#         self.relu1 = nn.ReLU()
#         self.conv1 = nn.Conv2d()
#
#     def forward(self, x):
#         return x + self.main(x)


class PreActBlock(nn.Module):
    """
    Pre-activation version of the BasicBlock.
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py

    set bias=True w.r.t paper says bias should be zero, bias=False w.r.t original github project
    """
    expansion = 1

    def __init__(self, dim_in, dim_out, stride=1, bias=True):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or dim_in != self.expansion * dim_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim_in, self.expansion * dim_out, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """
    Pre-activation version of the original Bottleneck module.
    """
    expansion = 4

    def __init__(self, dim_in, dim_out, stride=1, bias=True):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(dim_out)
        self.conv3 = nn.Conv2d(dim_out, self.expansion * dim_out, kernel_size=1, bias=bias)

        if stride != 1 or dim_in != self.expansion * dim_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim_in, self.expansion * dim_out, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    # assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class EqualLR:
    """

    """

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, content, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(content)
        out = gamma * out + beta

        return out


class Generator(nn.Module):
    """
    Generator network.
    """

    def __init__(self, conv_dim=64):
        super(Generator, self).__init__()

        # layers = [nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=3, bias=False)]
        self.input_conv = nn.Conv2d(3, 32, kernel_size=1)
        # layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        # layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        # encoder = []
        # for i in range(4):
        #     encoder.append(PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2))
        #     encoder.append(nn.AvgPool2d(2))
        #     encoder.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
        #     # layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim * 2
        # self.encoder = encoder
        self.resblk1 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool1 = nn.AvgPool2d(2)
        self.in1 = nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True)
        curr_dim = curr_dim * 2

        self.resblk2 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool2 = nn.AvgPool2d(2)
        self.in2 = nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True)
        curr_dim = curr_dim * 2

        self.resblk3 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool3 = nn.AvgPool2d(2)
        self.in3 = nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True)
        curr_dim = curr_dim * 2

        self.resblk4 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool4 = nn.AvgPool2d(2)
        self.in4 = nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True)
        curr_dim = curr_dim * 2

        # Bottleneck layers.
        # bottleneck = []
        # for i in range(2):
        #     bottleneck.append(PreActBottleneck(dim_in=curr_dim, dim_out=curr_dim))
        #     bottleneck.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
        self.resblk1_bottleneck = PreActBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.in1_bottleneck = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)

        self.resblk2_bottleneck = PreActBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.in2_bottleneck = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)

        self.resblk3_bottleneck = PreActBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.adain3_bottleneck = AdaptiveInstanceNorm(curr_dim, 64)

        self.resblk4_bottleneck = PreActBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.adain4_bottleneck = AdaptiveInstanceNorm(curr_dim, 64)

        # Up-sampling layers.
        self.resblk1_upsample = PreActBlock(dim_in=curr_dim, dim_out=curr_dim // 2)
        # self.upsample1 = nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)
        # self.upsample1 = F.interpolate(scale_factor=2, mode='nearest')
        self.adain1_upsample = AdaptiveInstanceNorm(curr_dim // 2, 64)
        curr_dim = curr_dim // 2

        self.resblk2_upsample = PreActBlock(dim_in=curr_dim, dim_out=curr_dim // 2)
        # self.upsample2 = F.interpolate(scale_factor=2, mode='nearest')
        self.adain2_upsample = AdaptiveInstanceNorm(curr_dim // 2, 64)
        curr_dim = curr_dim // 2

        self.resblk3_upsample = PreActBlock(dim_in=curr_dim, dim_out=curr_dim // 2)
        # self.upsample3 = F.interpolate(scale_factor=2, mode='nearest')
        self.adain3_upsample = AdaptiveInstanceNorm(curr_dim // 2, 64)
        curr_dim = curr_dim // 2

        self.resblk4_upsample = PreActBlock(dim_in=curr_dim, dim_out=curr_dim // 2)
        # self.upsample4 = F.interpolate(scale_factor=2, mode='nearest')
        self.adain4_upsample = AdaptiveInstanceNorm(curr_dim // 2, 64)
        curr_dim = curr_dim // 2

        # for i in range(2):
        #     layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        #     layers.append(nn.AdaptiveInstanceNorm(curr_dim // 2, affine=True, track_running_stats=True))

        # # Up-sampling layers.
        # for i in range(4):
        #     layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        #     layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        #     layers.append(nn.AdaptiveInstanceNorm(curr_dim // 2, affine=True, track_running_stats=True))
        #     # layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim // 2

        # layers.append(nn.Conv2d(curr_dim, 3, kernel_size=1, stride=1, padding=3, bias=False))
        # self.main = nn.Sequential(*layers)
        self.output_conv = nn.Conv2d(curr_dim, 3, kernel_size=1)

    def forward(self, x, style_code):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        output = self.input_conv(x)

        # encoder: downsampling
        # output = self.encoder(output)
        output = self.resblk1(output)
        output = self.avgpool1(output)
        output = self.in1(output)

        output = self.resblk2(output)
        output = self.avgpool2(output)
        output = self.in2(output)

        output = self.resblk3(output)
        output = self.avgpool3(output)
        output = self.in3(output)

        output = self.resblk4(output)
        output = self.avgpool4(output)
        output = self.in4(output)

        # bottleneck with IN and AdaIN
        output = self.resblk1_bottleneck(output)
        output = self.in1_bottleneck(output)

        output = self.resblk2_bottleneck(output)
        output = self.in2_bottleneck(output)

        output = self.resblk3_bottleneck(output)
        # output = adaptive_instance_normalization(output, style_code)
        output = self.adain3_bottleneck(output, style_code)

        output = self.resblk4_bottleneck(output)
        # output = adaptive_instance_normalization(output, style_code)
        output = self.adain4_bottleneck(output, style_code)

        # decoder: upsampling
        output = self.resblk1_upsample(output)
        output = F.interpolate(output, scale_factor=2, mode='nearest')
        # output = adaptive_instance_normalization(output, style_code)
        output = self.adain1_upsample(output, style_code)

        output = self.resblk2_upsample(output)
        output = F.interpolate(output, scale_factor=2, mode='nearest')
        # output = adaptive_instance_normalization(output, style_code)
        output = self.adain2_upsample(output, style_code)

        output = self.resblk3_upsample(output)
        output = F.interpolate(output, scale_factor=2, mode='nearest')
        # output = adaptive_instance_normalization(output, style_code)
        output = self.adain3_upsample(output, style_code)

        output = self.resblk4_upsample(output)
        output = F.interpolate(output, scale_factor=2, mode='nearest')
        # output = adaptive_instance_normalization(output, style_code)
        output = self.adain4_upsample(output, style_code)

        return self.output_conv(output)

        # return self.main(x)


class Discriminator(nn.Module):
    """
    Discriminator Network
    """

    def __init__(self, repeat_num=5, channel_multiplier=32, num_domains=2, dimension=1):
        """

        :param repeat_num:
        :param channel_multiplier: 16 for style encoder, 32 for discriminator
        :param num_domains:
        :param dimension: Style Encoder 64 for style code, Discriminator 1 for real/fake classification
        """
        super(Discriminator, self).__init__()
        # layers = [nn.Conv2d(3, channel_multiplier, kernel_size=1)]
        #
        curr_dim = channel_multiplier
        # for i in range(1, repeat_num):
        #     layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim * 2))
        #     layers.append(nn.AvgPool2d(4))
        #     curr_dim = curr_dim * 2
        #
        # layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        # layers.append(nn.AvgPool2d(4))
        #
        # layers.append(nn.LeakyReLU(0.01))
        # layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4))
        # layers.append(nn.LeakyReLU(0.01))
        #
        # self.main = nn.Sequential(*layers)

        self.conv1x1 = nn.Conv2d(3, channel_multiplier, kernel_size=1)

        self.resblk1 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool1 = nn.AvgPool2d(2)
        curr_dim = curr_dim * 2

        self.resblk2 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool2 = nn.AvgPool2d(2)
        curr_dim = curr_dim * 2

        self.resblk3 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool3 = nn.AvgPool2d(2)
        curr_dim = curr_dim * 2

        self.resblk4 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool4 = nn.AvgPool2d(2)
        curr_dim = curr_dim * 2

        self.resblk5 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool5 = nn.AvgPool2d(2)
        curr_dim = curr_dim * 2

        self.resblk6 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.avgpool6 = nn.AvgPool2d(2)
        # curr_dim = curr_dim * 2

        self.curr_dim = curr_dim

        self.lrelu1 = nn.LeakyReLU(0.01)
        self.conv4x4 = nn.Conv2d(curr_dim, curr_dim, kernel_size=4)
        self.lrelu2 = nn.LeakyReLU(0.01)

        self.out = nn.Linear(curr_dim, num_domains)

    def forward(self, x, num_domains=6):
        x = self.conv1x1(x)

        x = self.resblk1(x)
        x = self.avgpool1(x)

        x = self.resblk2(x)
        x = self.avgpool2(x)

        x = self.resblk3(x)
        x = self.avgpool3(x)

        x = self.resblk4(x)
        x = self.avgpool4(x)

        x = self.resblk5(x)
        x = self.avgpool5(x)

        x = self.resblk6(x)
        x = self.avgpool6(x)

        x = self.lrelu1(x)
        x = self.conv4x4(x)
        x = self.lrelu2(x)

        # h = self.main(x)
        x = x.view(-1, self.curr_dim)
        # return [self.out(x) for _ in range(num_domains)]
        out = self.out(x)
        return out


# class StyleEncoder(nn.Module):
#     """
#     Style Encoder with PatchGAN.
#     """
#
#     def __init__(self, repeat_num=5, channel_multiplier=16, dimension=64):
#         """
#
#         :param repeat_num:
#         :param channel_multiplier: 16 for style encoder, 32 for discriminator
#         :param num_domains:
#         :param dimension: Style Encoder 64 for style code, Discriminator 1 for real/fake classification
#         """
#         super(StyleEncoder, self).__init__()
#         layers = [nn.Conv2d(3, channel_multiplier, kernel_size=1, stride=2, padding=1)]
#
#         curr_dim = channel_multiplier
#         for i in range(1, repeat_num):
#             layers.append(PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2))
#             layers.append(nn.AvgPool2d(4))
#             curr_dim = curr_dim * 2
#
#         layers.append(PreActBlock(dim_in=curr_dim, dim_out=curr_dim))
#         layers.append(nn.AvgPool2d(4))
#
#         layers.append(nn.LeakyReLU(0.01))
#         layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1))
#         layers.append(nn.LeakyReLU(0.01))
#
#         self.main = nn.Sequential(*layers)
#         self.out = nn.Linear(curr_dim, dimension)
#
#     def forward(self, x, num_domains=6):
#         h = self.main(x)
#         return [self.out(h) for _ in range(num_domains)]


class StyleEncoder(nn.Module):
    """
    Style Encoder
    """

    def __init__(self, repeat_num=5, channel_multiplier=16, num_domains=2, dimension=64):
        """

        :param repeat_num:
        :param channel_multiplier: 16 for style encoder, 32 for discriminator
        :param num_domains:
        :param dimension: Style Encoder 64 for style code, Discriminator 1 for real/fake classification
        """
        super(StyleEncoder, self).__init__()
        # layers = [nn.Conv2d(3, channel_multiplier, kernel_size=1)]
        #
        curr_dim = channel_multiplier
        # for i in range(1, repeat_num):
        #     layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim * 2))
        #     layers.append(nn.AvgPool2d(4))
        #     curr_dim = curr_dim * 2
        #
        # layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        # layers.append(nn.AvgPool2d(4))
        #
        # layers.append(nn.LeakyReLU(0.01))
        # layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4))
        # layers.append(nn.LeakyReLU(0.01))
        #
        # self.main = nn.Sequential(*layers)

        self.conv1x1 = nn.Conv2d(3, channel_multiplier, kernel_size=1)

        self.resblk1 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool1 = nn.AvgPool2d(2)
        curr_dim = curr_dim * 2

        self.resblk2 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool2 = nn.AvgPool2d(2)
        curr_dim = curr_dim * 2

        self.resblk3 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool3 = nn.AvgPool2d(2)
        curr_dim = curr_dim * 2

        self.resblk4 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool4 = nn.AvgPool2d(2)
        curr_dim = curr_dim * 2

        self.resblk5 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim * 2)
        self.avgpool5 = nn.AvgPool2d(2)
        curr_dim = curr_dim * 2

        self.resblk6 = PreActBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.avgpool6 = nn.AvgPool2d(2)
        # curr_dim = curr_dim * 2

        self.curr_dim = curr_dim

        self.lrelu1 = nn.LeakyReLU(0.01)
        self.conv4x4 = nn.Conv2d(curr_dim, curr_dim, kernel_size=4)
        self.lrelu2 = nn.LeakyReLU(0.01)

        self.out = nn.Linear(curr_dim, dimension)

    def forward(self, x, num_domains=2):
        x = self.conv1x1(x)

        x = self.resblk1(x)
        x = self.avgpool1(x)

        x = self.resblk2(x)
        x = self.avgpool2(x)

        x = self.resblk3(x)
        x = self.avgpool3(x)

        x = self.resblk4(x)
        x = self.avgpool4(x)

        x = self.resblk5(x)
        x = self.avgpool5(x)

        x = self.resblk6(x)
        x = self.avgpool6(x)

        x = self.lrelu1(x)
        x = self.conv4x4(x)
        x = self.lrelu2(x)

        # h = self.main(x)
        x = x.view(-1, self.curr_dim)
        # return [self.out(x) for _ in range(num_domains)]
        # out = self.out(x)
        # return out
        return [self.out(x) for _ in range(num_domains)]


class Mapping(nn.Module):
    """
    Mapping network.
    """

    def __init__(self, image_size=128, repeat_num=6):
        """

        :param image_size:
        :param conv_dim:
        :param num_domains: nums of domains
        :param repeat_num: layer nums of linear FCN
        """
        super(Mapping, self).__init__()
        layers = [nn.Linear(16, 512), nn.ReLU()]

        for i in range(1, repeat_num):
            layers.append(nn.Linear(512, 512))
            layers.append(nn.ReLU())

        self.out = nn.Linear(512, 64)

        self.main = nn.Sequential(*layers)

    def forward(self, z, num_domains=6):
        """

        :param num_domains:
        :param z: latent code
        :return:
        """
        h = self.main(z)
        return [self.out(h) for _ in range(num_domains)]


def init_weights(m):
    """
    init weights with Kaiming He init and bias with 0
    https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    :param m:
    :return:
    """
    # print(m)
    if isinstance(m, nn.Module) and not isinstance(m, AdaptiveInstanceNorm):
        if not isinstance(m, nn.BatchNorm2d) and not isinstance(m, nn.AvgPool2d) and not isinstance(m, nn.InstanceNorm2d)\
                and not isinstance(m, nn.ReLU) and not isinstance(m, nn.LeakyReLU):
            # nn.init.kaiming_normal_(m.weight)
            modules = [f for f in m.children()]
            if modules:
                for s in modules:
                    # nn.init.kaiming_uniform_(s.weight)
                    # s.bias.data.zero_()
                    init_weights(s)
            else:
                try:
                    nn.init.kaiming_uniform_(m.weight)
                    m.bias.data.zero_()
                except Exception as e:
                    print(str(e))
            # print(m.weight)
            # print(m.bias)
    # if isinstance(m, AdaptiveInstanceNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.AvgPool2d) or isinstance(m, nn.InstanceNorm2d):
    #     pass
    # else:
    #     nn.init.kaiming_uniform_(m.weight)
    #     m.bias.data.zero_()
