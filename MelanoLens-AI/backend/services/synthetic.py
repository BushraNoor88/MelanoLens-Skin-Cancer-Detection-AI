"""Synthetic lesion-like images with KNOWN reference masks (no dataset needed)."""
import cv2
import numpy as np


def _render(mask: np.ndarray, noise_sigma: float = 0.0, blur: float = 0.0, seed: int = 0):
    """Dark lesion (inside mask) on lighter skin-like background -> (bgr image, mask 0/255)."""
    rng = np.random.default_rng(seed)
    h, w = mask.shape
    gray = np.full((h, w), 190, np.float32)
    gray[mask > 0] = 90
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur)
    if noise_sigma > 0:
        gray += rng.normal(0, noise_sigma, gray.shape).astype(np.float32)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), mask


def circle(size=256, radius=70, **kw):
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), radius, 255, -1)
    return _render(m, **kw)


def irregular_polygon(size=256, **kw):
    m = np.zeros((size, size), np.uint8)
    pts = np.array([[60, 100], [110, 50], [170, 70], [200, 130], [170, 190], [110, 200], [70, 160]], np.int32)
    cv2.fillPoly(m, [pts], 255)
    return _render(m, **kw)


def noisy_shape(size=256, **kw):
    kw.setdefault("noise_sigma", 18.0)
    kw.setdefault("blur", 1.5)
    return circle(size, 65, **kw)
