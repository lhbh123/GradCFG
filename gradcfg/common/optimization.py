"""Shared attack helpers, preserving the original numerical operations."""
import os
import random

import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Additional settings for better reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def tv_loss(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Anisotropic TV loss: |dx| + |dy|
    img: (B, C, H, W)
    """
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]    # (B, C, H-1, W)
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]    # (B, C, H,   W-1)

    tv_y = dy.abs().mean()
    tv_x = dx.abs().mean()
    return tv_x + tv_y


def global_l2_clip(g: torch.Tensor, clip_norm: float, eps: float = 1e-12):
    norm = g.norm(p=2)
    scale = torch.clamp(clip_norm / (norm + eps), max=1.0)
    return g * scale


def add_dp_noise(g: torch.Tensor, clip_norm_used: float, sigma: float):
    if sigma <= 0:
        return g
    std = sigma * clip_norm_used
    print(torch.norm(torch.rand_like(g)))
    return g + torch.randn_like(g) * std


def topk_sparsify(g: torch.Tensor, ratio: float):
    if ratio <= 0 or ratio >= 1:
        return g
    flat = g.view(-1)
    k = max(1, int(flat.numel() * ratio))
    # Use the kth value as the threshold to retain the largest k entries.
    thresh = flat.abs().kthvalue(flat.numel() - k + 1).values
    mask = (flat.abs() >= thresh).to(flat.dtype)
    return (flat * mask).view_as(g)
