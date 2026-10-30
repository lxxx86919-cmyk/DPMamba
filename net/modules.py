import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, act=True):
        super(ConvNormAct, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels)
        ]
        if act:
            layers.append(Mish())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act=True):
        super(DepthwiseSeparableConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.InstanceNorm2d(in_channels),
            Mish(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(out_channels)
        ]
        if act:
            layers.append(Mish())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LightweightEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(LightweightEncoder, self).__init__()
        self.conv1 = ConvNormAct(in_channels, base_channels, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(base_channels, base_channels * 2)
        self.conv3 = DepthwiseSeparableConv(base_channels * 2, base_channels * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LightweightPhysicsBranch(nn.Module):
    """
    Lightweight branch for t/tb and optional A prediction.
    It is used when use_lightweight=1 to reduce the cost of the physical branches.
    """
    def __init__(self, in_channels=3, base_channels=32, predict_a=True, use_a_residual_map=False, a_residual_scale=0.1):
        super(LightweightPhysicsBranch, self).__init__()
        self.predict_a = predict_a
        self.use_a_residual_map = use_a_residual_map
        self.a_residual_scale = a_residual_scale
        feat_channels = base_channels * 2

        self.encoder = LightweightEncoder(in_channels, base_channels)
        self.t_head = nn.Sequential(
            DepthwiseSeparableConv(feat_channels, feat_channels),
            nn.Conv2d(feat_channels, 3, 1, 1, 0),
            nn.Sigmoid()
        )
        self.tb_head = nn.Sequential(
            DepthwiseSeparableConv(feat_channels, feat_channels),
            nn.Conv2d(feat_channels, 3, 1, 1, 0),
            nn.Sigmoid()
        )
        if self.predict_a:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.a_global = nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, 1, 1, 0, bias=False),
                Mish(),
                nn.Conv2d(feat_channels, 3, 1, 1, 0),
                nn.Sigmoid()
            )
            if self.use_a_residual_map:
                self.a_residual = nn.Sequential(
                    DepthwiseSeparableConv(feat_channels, feat_channels),
                    nn.Conv2d(feat_channels, 3, 1, 1, 0)
                )
            else:
                self.a_residual = None
        else:
            self.pool = None
            self.a_global = None
            self.a_residual = None

    def forward(self, x):
        feat = self.encoder(x)
        t = self.t_head(feat)
        tb = self.tb_head(feat)
        if self.predict_a:
            A_global = self.a_global(self.pool(feat))
            if self.use_a_residual_map:
                A_residual = torch.tanh(self.a_residual(feat))
                A = torch.clamp(A_global + self.a_residual_scale * A_residual, 0.0, 1.0)
            else:
                A_residual = None
                A = A_global.expand(-1, -1, x.shape[2], x.shape[3])
        else:
            A_global = None
            A_residual = None
            A = None

        aux = {
            'features': feat,
            'A_global': A_global,
            'A_residual': A_residual,
            'A_map': A,
        }
        return t, tb, A, aux
