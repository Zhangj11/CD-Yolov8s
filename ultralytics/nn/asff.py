# 代码已读√
import torch
import torch.nn.functional as F
import torch.nn as nn

class SiLU(nn.Module):
    """export-friendly inplace version of nn.SiLU()"""

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    @staticmethod
    def forward(x):
        # clone is not supported with nni 2.6.1
        # result = x.clone()
        # torch.sigmoid_(x)
        return x * torch.sigmoid(x)


class HSiLU(nn.Module):
    """
        export-friendly inplace version of nn.SiLU()
        hardsigmoid is better than sigmoid when used for edge model
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    @staticmethod
    def forward(x):
        # clone is not supported with nni 2.6.1
        # result = x.clone()
        # torch.hardsigmoid(x)
        return x * torch.hardsigmoid(x)


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        # @ to do nn.SiLU 1.7.0
        # module = nn.SiLU(inplace=inplace)
        module = SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'hsilu':
        module = HSiLU(inplace=inplace)
    elif name == 'identity':
        module = nn.Identity(inplace=inplace)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act='silu'):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ASFF(nn.Module):

    def __init__(self,
                 level,
                 type='ASFF',
                 asff_channel=2,
                 expand_kernel=3,
                 multiplier=1,
                 act='silu'):
        """
        Args:
            level(int): the level of the input feature
            type(str): ASFF or ASFF_sim
            asff_channel(int): the hidden channel of the attention layer in ASFF
            expand_kernel(int): expand kernel size of the expand layer
            multiplier: should be the same as width in the backbone
        """
        super(ASFF, self).__init__()
        self.level = level
        self.type = type

        self.dim = [
            int(1024 * multiplier),  # multiplier=widen_factor=0.5
            int(512 * multiplier),
            int(256 * multiplier)
        ]

        Conv = BaseConv

        self.inter_dim = self.dim[self.level]

        if self.type == 'ASFF':
            # 512
            if level == 0:
                # 256 -> 512
                self.stride_level_1 = Conv(
                    int(512 * multiplier), self.inter_dim, 3, 2, act=act)

                # 128 -> 512, 步距为2的最大池化和卷积进行下采样
                self.stride_level_2 = Conv(
                    int(256 * multiplier), self.inter_dim, 3, 2, act=act)

            # 256
            elif level == 1:
                # 512 -> 256
                self.compress_level_0 = Conv(
                    int(1024 * multiplier), self.inter_dim, 1, 1, act=act)
                # 128 -> 512
                self.stride_level_2 = Conv(
                    int(256 * multiplier), self.inter_dim, 3, 2, act=act)

            # 128
            elif level == 2:
                # 512 -> 128,1x1卷积降维
                self.compress_level_0 = Conv(
                    int(1024 * multiplier), self.inter_dim, 1, 1, act=act)
                # 256 -> 128
                self.compress_level_1 = Conv(
                    int(512 * multiplier), self.inter_dim, 1, 1, act=act)
            else:
                raise ValueError('Invalid level {}'.format(level))

        # 用于将加权后的三层特征图进一步融合
        self.expand = Conv(self.inter_dim, self.inter_dim, expand_kernel, 1, act=act)

        self.weight_level_0 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_1 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_2 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)

        self.weight_levels = Conv(asff_channel * 3, 3, 1, 1, act=act)

    def expand_channel(self, x):
        # [b,c,h,w]->[b,c*4,h/2,w/2],
        # v5Fcous层切片，通道*4，高宽减半
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x

    def mean_channel(self, x):
        # [b,c,h,w]->[b,c/2,h,w],通道减半
        x1 = x[:, ::2, :, :]
        x2 = x[:, 1::2, :, :]
        return (x1 + x2) / 2

    def forward(self, x):  # l,m,s
        """
        #
        128, 256, 512
        from small -> large
        """
        x_level_0 = x[2]  # max feature level [512,20,20]
        x_level_1 = x[1]  # mid feature level [256,40,40]
        x_level_2 = x[0]  # min feature level [128,80,80]

        if self.type == 'ASFF':
            if self.level == 0:
                # [512,20,20]
                level_0_resized = x_level_0
                # [256,40,40] -> [512,20,20]
                level_1_resized = self.stride_level_1(x_level_1)
                # [128,80,80] -> [128,40,40]
                level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
                # [128,40,40] -> [512,20,20]
                level_2_resized = self.stride_level_2(level_2_downsampled_inter)
            elif self.level == 1:
                # [512,20,20] -> [256,20,20]
                level_0_compressed = self.compress_level_0(x_level_0)
                # [256,20,20] -> [256,40,40]
                level_0_resized = F.interpolate(
                    level_0_compressed, scale_factor=2, mode='nearest')
                level_1_resized = x_level_1  # 不变
                # [128,80,80] -> [256,40,40]
                level_2_resized = self.stride_level_2(x_level_2)
            elif self.level == 2:
                # [512,20,20] -> [128,20,20]
                level_0_compressed = self.compress_level_0(x_level_0)
                # [128,20,20] -> [128,80,80]
                level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
                # [256,40,40] -> [128,40,40]
                x_level_1_compressed = self.compress_level_1(x_level_1)
                # [128,40,40] -> [128,80,80]
                level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
                # [128,80,80]不变
                level_2_resized = x_level_2
        else:
            # ASFF_sim
            if self.level == 0:
                # [512,20,20]
                level_0_resized = x_level_0
                # [256,40,40] -> [1024,20,20]
                level_1_resized = self.expand_channel(x_level_1)  # c*4，h/2, w/2
                # [1024,20,20] -> [512,20,20]
                level_1_resized = self.mean_channel(level_1_resized)  # c/2,h,w不变
                # [128,80,80] -> [512,40,40]
                level_2_resized = self.expand_channel(x_level_2)
                # [512,40,40] -> [512,20,20]
                level_2_resized = F.max_pool2d(level_2_resized, 3, stride=2, padding=1)

            elif self.level == 1:
                # [512,20,20] -> [512,40,40]
                level_0_resized = F.interpolate(x_level_0, scale_factor=2, mode='nearest')
                # [512,40,40] -> [256,40,40]
                level_0_resized = self.mean_channel(level_0_resized)
                # [256,40,40],不变
                level_1_resized = x_level_1
                # [128,80,80] -> [512,40,40]
                level_2_resized = self.expand_channel(x_level_2)
                # [512,40,40] -> [256,40,40]
                level_2_resized = self.mean_channel(level_2_resized)

            elif self.level == 2:
                # [512,20,20]->[512,80,80]
                level_0_resized = F.interpolate(x_level_0, scale_factor=4, mode='nearest')
                # [512,80,80]->[256,80,80]->[128,80,80]
                level_0_resized = self.mean_channel(self.mean_channel(level_0_resized))
                # [256,40,40]->[256,80,80]
                level_1_resized = F.interpolate(x_level_1, scale_factor=2, mode='nearest')
                # [256,80,80]->[128,80,80]
                level_1_resized = self.mean_channel(level_1_resized)  # 通道减半，宽高不变
                # [128,80,80]
                level_2_resized = x_level_2

        # 每层的权重,(1,C,H,W),例如(1,512,20,20)->(1,2,20,20)
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        # (1,6,20,20)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        # (1,3,20,20)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        # ex:(1,512,20,20)
        fused_out_reduced = level_0_resized * levels_weight[:, 0: 1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        # 用于将加权后的三层特征图进一步融合,ex:[1,512,20,20]->[1,512,20,20]
        out = self.expand(fused_out_reduced)

        return out

