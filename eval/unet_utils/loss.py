"""
Loss functions ported from O2MAG (`eval/unet_utils/loss.py`).

* ``FocalLoss`` — focal loss for imbalanced anomaly segmentation.
* ``SSIMLoss``  — multi-scale structural similarity loss (neg-SSIM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation.

    Expects ``pred`` in [0, 1] (the model already applies ``Sigmoid``), so this
    uses ``binary_cross_entropy`` (NOT ``..._with_logits``); applying the
    logits variant on top of a sigmoid output would double-squash the values.
    """

    def __init__(self, gamma=2.0, alpha=1.0, beta=1.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, targets):
        pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        bce = F.binary_cross_entropy(pred, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        if self.alpha >= 0.0:
            loss = loss * self.alpha
        loss = loss * (targets + self.beta * (1 - targets))
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _gaussian(window_size, sigma=1.5):
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    return g / g.sum()


class SSIMLoss(nn.Module):
    """Multi-scale structural similarity loss (1 - SSIM), ported from O2MAG."""

    def __init__(self, window_size=11, max_val=1.0):
        super().__init__()
        self.window_size = window_size
        self.max_val = max_val
        self.C1 = (0.01 * max_val) ** 2
        self.C2 = (0.03 * max_val) ** 2

    def _create_window(self, channel, device):
        g1d = _gaussian(self.window_size)
        window_2d = g1d[:, None] @ g1d[None, :]
        window = window_2d.expand(channel, 1, self.window_size, self.window_size)
        return window.to(device)

    def forward(self, img1, img2):
        # Assume single-channel masks in [0,1]; expand to 3 channels if needed.
        channel = img1.size(1)
        device = img1.device
        window = self._create_window(channel, device)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(
                img1 * img1, window, padding=self.window_size // 2, groups=channel
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                img2 * img2, window, padding=self.window_size // 2, groups=channel
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                img1 * img2, window, padding=self.window_size // 2, groups=channel
            )
            - mu1_mu2
        )

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )
        return 1 - ssim_map.mean()
