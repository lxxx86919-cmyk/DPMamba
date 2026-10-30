import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def manual_A_torch(img):
    """
    Torch approximation of A.py/get_A.
    Input:  img [B, 3, H, W], values in [0, 1]
    Output: A_map [B, 3, H, W], values in [0, 1]
    """
    B, C, H, W = img.shape
    if C != 3:
        raise ValueError('manual_A_torch expects RGB input with 3 channels')

    x = img.clamp(0, 1) * 255.0
    flat = x.flatten(2)

    r = flat[:, 0]
    g = flat[:, 1]
    b = flat[:, 2]

    a_r = 140.0 / (1.0 + 14.4 * torch.exp(-0.034 * torch.median(r, dim=1).values))
    a_g = 1.13 * torch.mean(g, dim=1) + 1.11 * torch.std(g, dim=1, unbiased=False) - 25.6
    a_b = 1.13 * torch.mean(b, dim=1) + 1.11 * torch.std(b, dim=1, unbiased=False) - 25.6

    A = torch.stack([a_r, a_g, a_b], dim=1).view(B, 3, 1, 1)
    A = torch.clamp(A, 5.0, 250.0) / 255.0
    return A.expand(B, 3, H, W)


class LearnablePhysicsHead(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, use_a_residual_map=False, a_residual_scale=0.1):
        super(LearnablePhysicsHead, self).__init__()
        self.use_a_residual_map = use_a_residual_map
        self.a_residual_scale = a_residual_scale

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            Mish(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            Mish(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            Mish()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.a_global = nn.Conv2d(hidden_dim, 3, 1, 1, 0)

        if self.use_a_residual_map:
            self.a_residual = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                Mish(),
                nn.Conv2d(hidden_dim, 3, 1, 1, 0)
            )
        else:
            self.a_residual = None

    def forward(self, x):
        feat = self.encoder(x)
        A_global = torch.sigmoid(self.a_global(self.pool(feat)))

        if self.use_a_residual_map:
            A_residual = torch.tanh(self.a_residual(feat))
            A = torch.clamp(A_global + self.a_residual_scale * A_residual, 0.0, 1.0)
        else:
            A_residual = None
            A = A_global.expand(-1, -1, x.shape[2], x.shape[3])

        aux = {
            'A_global': A_global,
            'A_residual': A_residual,
            'A_map': A,
        }
        return A, aux


def physical_reconstruction(J, t, tb, A):
    return J * t + (1.0 - tb) * A


def smoothness_loss(t, tb=None):
    def gradient_loss(x):
        dx = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        dy = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        return dx + dy

    loss = gradient_loss(t)
    if tb is not None:
        loss = loss + gradient_loss(tb)
    return loss
