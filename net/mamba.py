
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from net.frequency import FrequencyColorTextureCompensation

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

def _make_divisible(v, divisor=8, min_value=None):  ## 将通道数变成8的整数倍
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0,),
                Mish(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                nn.Sigmoid()  ## hardsigmoid
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DynamicMambaFusion(nn.Module):
    def __init__(self, dim, num_branches=4, reduction=4):
        super(DynamicMambaFusion, self).__init__()
        hidden_dim = _make_divisible(dim // reduction)
        self.num_branches = num_branches
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            Mish(),
            nn.Conv2d(hidden_dim, num_branches * dim, 1, 1, 0)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, branches, context, return_weights=False):
        B, K, C, H, W = branches.shape
        assert K == self.num_branches
        logits = self.gate(self.pool(context)).view(B, K, C, 1, 1)
        weights = F.softmax(logits, dim=1)
        out = torch.sum(branches * weights, dim=1)
        if return_weights:
            return out, weights
        return out


class MambaLayer(nn.Module):  ## input (1, 3, 256, 256). output (1, 3, 256, 256).
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, fusion_reduction=4, use_dynamic_mamba=True):
        super().__init__()
        self.dim = dim
        self.use_dynamic_mamba = use_dynamic_mamba
        self.last_scan_weights = None

        self.nin = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm = nn.InstanceNorm2d(dim) # LayerNorm
        self.relu = Mish()
        #
        self.nin2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.relu2 = Mish()



        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )
        self.fusion = DynamicMambaFusion(dim, num_branches=4, reduction=fusion_reduction)

    def forward(self, x):
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)
        act_x = x
        assert C == self.dim
        H, W = x.shape[2:]

        x_hw = x.reshape(B, C, H * W).transpose(1, 2).contiguous()
        x_hw_rev = torch.flip(x_hw, dims=[1])
        x_wh = x.permute(0, 1, 3, 2).contiguous().reshape(B, C, H * W).transpose(1, 2).contiguous()
        x_wh_rev = torch.flip(x_wh, dims=[1])

        y_hw = self.mamba(x_hw)
        y_hw_rev = torch.flip(self.mamba(x_hw_rev), dims=[1])
        y_wh = self.mamba(x_wh)
        y_wh_rev = torch.flip(self.mamba(x_wh_rev), dims=[1])

        y_hw = y_hw.transpose(1, 2).reshape(B, C, H, W)
        y_hw_rev = y_hw_rev.transpose(1, 2).reshape(B, C, H, W)
        y_wh = y_wh.transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2).contiguous()
        y_wh_rev = y_wh_rev.transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2).contiguous()

        branches = torch.stack([y_hw, y_hw_rev, y_wh, y_wh_rev], dim=1)
        if self.use_dynamic_mamba:
            out, weights = self.fusion(branches, act_x, return_weights=True)
        else:
            out = torch.mean(branches, dim=1)
            weights = None
        self.last_scan_weights = weights
        # act_x = self.relu3(x)
        out += act_x
        out = self.nin2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out


class JNet_mamba(torch.nn.Module):  ## mamba
    def __init__(self, use_dynamic_mamba=True, use_frequency_module=True):
        super().__init__()
        self.use_dynamic_mamba = use_dynamic_mamba
        self.use_frequency_module = use_frequency_module
        self.conv1 = torch.nn.Sequential(  ##  提升通道数
            # torch.nn.ReflectionPad2d(1),  ## 1x1 卷积前面不需要这个
            torch.nn.Conv2d(3, 64, 1, 1, 0),
            # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
            torch.nn.InstanceNorm2d(64),
            # torch.nn.ReLU()
            Mish()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 128, 3, 1, 0),
            torch.nn.InstanceNorm2d(128),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(128, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv5 = torch.nn.Sequential(
            SELayer(512, 512),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(512, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv6 = torch.nn.Sequential(
            SELayer(386, 384),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(384, 128, 3, 1, 0),
            torch.nn.InstanceNorm2d(128),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv7 = torch.nn.Sequential(
            SELayer(192, 192),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(192, 64, 3, 1, 0),
            torch.nn.InstanceNorm2d(64),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )
        self.mamba1 = MambaLayer(dim=64, use_dynamic_mamba=use_dynamic_mamba)
        self.mamba2 = MambaLayer(dim=128, use_dynamic_mamba=use_dynamic_mamba)
        self.mamba3 = MambaLayer(dim=256, use_dynamic_mamba=use_dynamic_mamba)
        if self.use_frequency_module:
            self.fctc = FrequencyColorTextureCompensation(dim=128)
        else:
            self.fctc = None

    def forward(self, input):
        x1 = self.conv1(input)
        x1_mamba = self.mamba1(x1)
        x2 = self.conv2(x1)
        x2_mamba = self.mamba2(x2)
        if self.use_frequency_module:
            x2_mamba = self.fctc(x2_mamba)
        x3 = self.conv3(x2)
        x3_mamba = self.mamba3(x3)

        x4 = self.conv4(x3)

        x5 = self.conv5(torch.cat((x4, x3_mamba), dim=1))
        x6 = self.conv6(torch.cat((x5, x2_mamba), dim=1))
        x7 = self.conv7(torch.cat((x6, x1_mamba), dim=1))
        x8 = self.final(x7)

        return x8

#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 实例化模型并移动到设备上
# # model = JNet().to(device)
# model = JNet_mamba().to(device)
# input_tensor = torch.randn(1, 3, 256, 256).to(device)
#
# output_tensor = model(input_tensor)
#
# print("Output shape:", output_tensor.shape)