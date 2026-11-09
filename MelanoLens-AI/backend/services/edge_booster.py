"""Step C: the Edge Booster (core idea of MelanoLens AI).

Idea: instead of detecting edges on the raw image, first ADD the gradient
(edge strength) back to the image so that boundaries become more prominent.

    I_enhanced = normalize(I + alpha * G)

Warning: noise also has gradients, so a large alpha amplifies noise. Enhancement
improves visual edge prominence; it does NOT guarantee a better segmentation.
"""
import cv2
import numpy as np

from .preprocessing import ParameterError, validate_odd_kernel

METHODS = ("sobel", "laplacian", "morphological", "multiscale")


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Linearly stretch any float array to 0..255 uint8. A constant array becomes all zeros."""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros(arr.shape, np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _scale_0_255(arr: np.ndarray) -> np.ndarray:
    """Scale a NON-negative response by its maximum to 0..255 (float)."""
    m = float(arr.max())
    return arr * (255.0 / m) if m > 1e-9 else np.zeros_like(arr)


def _check_alpha(alpha, name="alpha", maximum=5.0) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ParameterError(f"{name} must be a number.")
    if not (0.0 <= alpha <= maximum):
        raise ParameterError(f"{name} must be between 0 and {maximum}.")
    return float(alpha)


def sobel_gradient(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """G = sqrt(Gx^2 + Gy^2) as float32 (not normalised)."""
    k = validate_odd_kernel(ksize, "sobel ksize", minimum=1, maximum=7)
    f = gray.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=k)
    return cv2.magnitude(gx, gy)


def sobel_enhance(gray, alpha=1.0, ksize=3):
    alpha = _check_alpha(alpha)
    g = _scale_0_255(sobel_gradient(gray, ksize))          # gradient in 0..255
    enhanced = normalize_to_uint8(gray.astype(np.float32) + alpha * g)
    return enhanced, g.astype(np.uint8)


def laplacian_enhance(gray, strength=1.0, ksize=3, pre_blur=3):
    """Laplacian sharpening.

    Sign convention: cv2.Laplacian uses a kernel with a NEGATIVE centre
    (e.g. [0 1 0; 1 -4 1; 0 1 0]). So a bright spot gives a negative response,
    and sharpening is   I_sharp = I - strength * Laplacian(I).
    A small Gaussian pre-blur limits noise amplification.
    """
    strength = _check_alpha(strength, "strength")
    k = validate_odd_kernel(ksize, "laplacian ksize", minimum=1, maximum=7)
    base = gray.astype(np.float32)
    if pre_blur and pre_blur > 1:
        pb = validate_odd_kernel(pre_blur, "pre_blur")
        base = cv2.GaussianBlur(base, (pb, pb), 0)
    lap = cv2.Laplacian(base, cv2.CV_32F, ksize=k)
    sharpened = np.clip(gray.astype(np.float32) - strength * lap, 0, 255).astype(np.uint8)
    return sharpened, normalize_to_uint8(np.abs(lap))


def morphological_gradient(gray, kernel_size=3) -> np.ndarray:
    """G_morph = dilation(I) - erosion(I), with an elliptical structuring element."""
    k = validate_odd_kernel(kernel_size, "morphological kernel", minimum=3, maximum=21)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, se)


def morphological_enhance(gray, alpha=1.0, kernel_size=3):
    alpha = _check_alpha(alpha)
    g = _scale_0_255(morphological_gradient(gray, kernel_size).astype(np.float32))
    return normalize_to_uint8(gray.astype(np.float32) + alpha * g), g.astype(np.uint8)


def multiscale_gradient(gray, sigmas=(1.0, 2.0, 4.0), weights=None) -> np.ndarray:
    """Weighted sum of normalised Sobel responses computed at several blur scales (0..1)."""
    sigmas = list(sigmas)
    if not sigmas or any(s <= 0 for s in sigmas):
        raise ParameterError("Scales (sigmas) must be positive numbers.")
    weights = [1.0] * len(sigmas) if weights is None else list(weights)
    if len(weights) != len(sigmas) or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ParameterError("Weights must match the scales, be non-negative, and not all zero.")
    total = float(sum(weights))
    combined = np.zeros(gray.shape, np.float32)
    for s, w in zip(sigmas, weights):
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), s)
        resp = sobel_gradient(blurred, 3)
        m = float(resp.max())
        if m > 1e-9:
            resp = resp / m                                  # normalise BEFORE combining
        combined += (w / total) * resp
    return combined


def multiscale_enhance(gray, alpha=1.0, sigmas=(1.0, 2.0, 4.0), weights=None):
    alpha = _check_alpha(alpha)
    g = _scale_0_255(multiscale_gradient(gray, sigmas, weights))
    return normalize_to_uint8(gray.astype(np.float32) + alpha * g), g.astype(np.uint8)


def edge_boost(gray: np.ndarray, method: str = "sobel", alpha: float = 1.0,
               kernel_size: int = 3, sigmas=(1.0, 2.0, 4.0), weights=None) -> dict:
    """Single entry point. Returns {"enhanced": uint8 image, "gradient": uint8 map, "method": name}."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ParameterError("Edge Booster expects a single-channel uint8 image.")
    if method == "sobel":
        enhanced, grad = sobel_enhance(gray, alpha, kernel_size)
    elif method == "laplacian":
        enhanced, grad = laplacian_enhance(gray, alpha, kernel_size)
    elif method == "morphological":
        enhanced, grad = morphological_enhance(gray, alpha, kernel_size)
    elif method == "multiscale":
        enhanced, grad = multiscale_enhance(gray, alpha, sigmas, weights)
    else:
        raise ParameterError(f"method must be one of: {', '.join(METHODS)}.")
    return {"enhanced": enhanced, "gradient": grad, "method": method}
