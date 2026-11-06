"""Explicit RGB metric protocols; historical compare.py is available as legacy."""
from functools import lru_cache
import math

import numpy as np


PROTOCOLS = ("standard", "legacy")


def align_images(img1, img2):
    import cv2

    if img1.ndim != 3 or img2.ndim != 3 or img1.shape[2] != 3 or img2.shape[2] != 3:
        raise ValueError("Expected RGB arrays of shape [H, W, 3].")
    if img1.dtype != np.uint8 or img2.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB images in [0, 255].")
    if img1.shape != img2.shape:
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
    return img1, img2


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * math.log10(255 ** 2 / mse))


@lru_cache(maxsize=8)
def build_lpips_model(device="cpu", net="alex"):
    import lpips

    return lpips.LPIPS(net=net).to(device).eval()


def to_lpips_tensor(image, device, protocol):
    import torch

    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown metric protocol: {protocol}")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if protocol == "standard":
        tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def calculate_image_similarity(img1, img2, device="cpu", protocol="standard", lpips_model=None, skip_lpips=False):
    """SSIM on grayscale, PSNR on RGB, LPIPS on RGB; resize to minimum size.

    legacy reproduces compare.py's BGR gray conversion on RGB and [0,1] LPIPS.
    Missing metric dependencies raise an error; SSIM is never replaced with NCC.
    """
    import cv2
    from skimage.metrics import structural_similarity

    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown metric protocol: {protocol}")
    img1, img2 = align_images(img1, img2)
    conversion = cv2.COLOR_RGB2GRAY if protocol == "standard" else cv2.COLOR_BGR2GRAY
    scores = {
        "ssim": float(structural_similarity(cv2.cvtColor(img1, conversion), cv2.cvtColor(img2, conversion), data_range=255)),
        "psnr": calculate_psnr(img1, img2),
        "lpips": None,
    }
    if not skip_lpips:
        import torch

        model = lpips_model if lpips_model is not None else build_lpips_model(str(device))
        with torch.no_grad():
            scores["lpips"] = float(model(to_lpips_tensor(img1, device, protocol), to_lpips_tensor(img2, device, protocol)).item())
    return scores


def calculate_legacy_similarity(img1, img2, device="cuda"):
    """Compatibility function used by historical reference matching."""
    return calculate_image_similarity(img1, img2, device=device, protocol="legacy")
