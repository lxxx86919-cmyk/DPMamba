import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def haar_dwt(x):
    """
    Haar wavelet decomposition.
    Input:  x, shape [B, C, H, W]
    Output: LL, LH, HL, HH, original_size
    """
    B, C, H, W = x.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    LL = (x00 + x01 + x10 + x11) * 0.5
    LH = (x00 - x01 + x10 - x11) * 0.5
    HL = (x00 + x01 - x10 - x11) * 0.5
    HH = (x00 - x01 - x10 + x11) * 0.5

    return LL, LH, HL, HH, (H, W)


def haar_idwt(LL, LH, HL, HH, output_size=None):
    """
    Haar inverse wavelet reconstruction.
    Input:  LL, LH, HL, HH, shape [B, C, H/2, W/2]
    Output: x, shape [B, C, H, W]
    """
    B, C, H, W = LL.shape
    x = LL.new_zeros(B, C, H * 2, W * 2)

    x[:, :, 0::2, 0::2] = (LL + LH + HL + HH) * 0.5
    x[:, :, 0::2, 1::2] = (LL - LH + HL - HH) * 0.5
    x[:, :, 1::2, 0::2] = (LL + LH - HL - HH) * 0.5
    x[:, :, 1::2, 1::2] = (LL - LH - HL + HH) * 0.5

    if output_size is not None:
        out_h, out_w = output_size
        x = x[:, :, :out_h, :out_w]

    return x


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        hidden_dim = max(dim // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            Mish(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(self.avg_pool(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, dim):
        super(DepthwiseSeparableConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.InstanceNorm2d(dim),
            Mish(),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(dim),
            Mish()
        )

    def forward(self, x):
        return self.conv(x)


class LowFrequencyColorCorrection(nn.Module):
    def __init__(self, dim):
        super(LowFrequencyColorCorrection, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(dim),
            Mish(),
            DepthwiseSeparableConv(dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(dim)
        )
        self.ca = ChannelAttention(dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = self.ca(self.block(x))
        return x + self.scale * residual


class HighFrequencyDetailEnhancement(nn.Module):
    def __init__(self, dim):
        super(HighFrequencyDetailEnhancement, self).__init__()
        hf_dim = dim * 3
        self.block = nn.Sequential(
            nn.Conv2d(hf_dim, hf_dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(hf_dim),
            Mish(),
            nn.Conv2d(hf_dim, hf_dim, 3, 1, 1, groups=hf_dim, bias=False),
            nn.InstanceNorm2d(hf_dim),
            Mish(),
            nn.Conv2d(hf_dim, hf_dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(hf_dim)
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, LH, HL, HH):
        hf = torch.cat([LH, HL, HH], dim=1)
        hf = hf + self.scale * self.block(hf)
        return torch.chunk(hf, 3, dim=1)


class FrequencyColorTextureCompensation(nn.Module):
    """
    Frequency Color-Texture Compensation Module.
    Low-frequency branch compensates color and illumination.
    High-frequency branch enhances edge and texture details.
    """
    def __init__(self, dim):
        super(FrequencyColorTextureCompensation, self).__init__()
        self.low_branch = LowFrequencyColorCorrection(dim)
        self.high_branch = HighFrequencyDetailEnhancement(dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(dim),
            Mish()
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        LL, LH, HL, HH, size = haar_dwt(x)
        LL = self.low_branch(LL)
        LH, HL, HH = self.high_branch(LH, HL, HH)
        freq = haar_idwt(LL, LH, HL, HH, output_size=size)
        freq = self.fusion(freq)
        return x + self.gamma * freq


def frequency_loss(pred, target, hf_weight=0.5):
    pred_LL, pred_LH, pred_HL, pred_HH, _ = haar_dwt(pred)
    target_LL, target_LH, target_HL, target_HH, _ = haar_dwt(target)

    loss_ll = F.l1_loss(pred_LL, target_LL)
    loss_hf = (
        F.l1_loss(pred_LH, target_LH) +
        F.l1_loss(pred_HL, target_HL) +
        F.l1_loss(pred_HH, target_HH)
    )
    return loss_ll + hf_weight * loss_hf
