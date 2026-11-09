"""Step D: edge detectors. Each returns a binary edge map (0 or 255) and a response map.

Sobel/Prewitt/Laplacian thresholds apply to the response scaled by its own
maximum to 0..255, so the same threshold behaves comparably on the
conventional and enhanced images.
"""
import cv2
import numpy as np

from .edge_booster import normalize_to_uint8
from .preprocessing import ParameterError, validate_odd_kernel

DETECTORS = ("canny", "sobel", "prewitt", "laplacian", "log")

PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)


def _check_gray(gray):
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ParameterError("Edge detectors expect a single-channel uint8 image.")


def _check_threshold(value, name="threshold"):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 <= value <= 255):
        raise ParameterError(f"{name} must be a number between 0 and 255.")
    return float(value)


def _scaled(arr):
    m = float(arr.max())
    return (arr * (255.0 / m)).astype(np.float32) if m > 1e-9 else np.zeros_like(arr, dtype=np.float32)


def _binary(response_255, threshold):
    return np.where(response_255 >= threshold, 255, 0).astype(np.uint8)


def canny(gray, low=50, high=150, blur_kernel=5):
    _check_gray(gray)
    low = _check_threshold(low, "Canny lower threshold")
    high = _check_threshold(high, "Canny upper threshold")
    if low >= high:
        raise ParameterError("Canny lower threshold must be smaller than the upper threshold.")
    k = validate_odd_kernel(blur_kernel, "canny blur kernel")
    src = cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray
    edges = cv2.Canny(src, low, high)
    return edges, edges.copy()


def canny_auto_thresholds(gray, percentile=85.0, blur_kernel=5):
    """Pick Canny thresholds from the image itself.

    high = the given percentile of the gradient magnitude (|Gx|+|Gy| after smoothing),
    low  = 0.4 * high. Using the SAME percentile for the conventional and the enhanced
    image gives both modes a comparable "edge budget", even though the Edge Booster
    changes the gradient scale. Floors (high >= 20, low >= 8) stop flat or synthetic images from getting near-zero thresholds."""
    _check_gray(gray)
    if not (50 <= percentile <= 99.5):
        raise ParameterError("canny_percentile must be between 50 and 99.5.")
    k = validate_odd_kernel(blur_kernel, "canny blur kernel")
    src = cv2.GaussianBlur(gray, (k, k), 0).astype(np.float32) if k > 1 else gray.astype(np.float32)
    mag = np.abs(cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3))
    high = max(float(np.percentile(mag, percentile)), 20.0)
    return max(0.4 * high, 8.0), high


def sobel(gray, threshold=60, ksize=3):
    _check_gray(gray)
    t = _check_threshold(threshold)
    k = validate_odd_kernel(ksize, "sobel ksize", minimum=1, maximum=7)
    f = gray.astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=k), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=k))
    resp = _scaled(mag)
    return _binary(resp, t), resp.astype(np.uint8)


def prewitt(gray, threshold=60):
    """Prewitt with explicit 3x3 kernels (equal weights, unlike Sobel's 1-2-1)."""
    _check_gray(gray)
    t = _check_threshold(threshold)
    f = gray.astype(np.float32)
    gx = cv2.filter2D(f, cv2.CV_32F, PREWITT_X)
    gy = cv2.filter2D(f, cv2.CV_32F, PREWITT_Y)
    resp = _scaled(cv2.magnitude(gx, gy))
    return _binary(resp, t), resp.astype(np.uint8)


def laplacian(gray, threshold=40, ksize=3):
    _check_gray(gray)
    t = _check_threshold(threshold)
    k = validate_odd_kernel(ksize, "laplacian ksize", minimum=1, maximum=7)
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=k))
    resp = _scaled(lap)
    return _binary(resp, t), resp.astype(np.uint8)


def laplacian_of_gaussian(gray, sigma=1.5, threshold=40):
    _check_gray(gray)
    if not (0.5 <= sigma <= 10):
        raise ParameterError("LoG sigma must be between 0.5 and 10.")
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), float(sigma))
    return laplacian(normalize_to_uint8(blurred), threshold, 3)


def run_detector(name: str, gray: np.ndarray, params: dict | None = None):
    """Run one detector by name. Returns (binary_edges, response)."""
    p = params or {}
    if name == "canny":
        low, high = p.get("canny_low", 50), p.get("canny_high", 150)
        if p.get("canny_auto"):
            low, high = canny_auto_thresholds(gray, p.get("canny_percentile", 85.0), p.get("canny_blur", 5))
        return canny(gray, min(low, 255), min(high, 255), p.get("canny_blur", 5))
    if name == "sobel":
        return sobel(gray, p.get("sobel_threshold", 60))
    if name == "prewitt":
        return prewitt(gray, p.get("prewitt_threshold", 60))
    if name == "laplacian":
        return laplacian(gray, p.get("laplacian_threshold", 40))
    if name == "log":
        return laplacian_of_gaussian(gray, p.get("log_sigma", 1.5), p.get("log_threshold", 40))
    raise ParameterError(f"detector must be one of: {', '.join(DETECTORS)}.")
