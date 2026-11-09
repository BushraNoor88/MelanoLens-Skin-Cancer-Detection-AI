"""Step B: preprocessing.

Why each step exists
- Grayscale      : edges are about brightness changes, so we drop colour.
- Gaussian blur  : smooths Gaussian-like sensor noise (also softens edges a bit).
- Median filter  : removes salt-and-pepper (impulse) noise and keeps edges fairly sharp.
- Bilateral      : smooths flat areas but keeps strong edges (weights by distance AND brightness).
- CLAHE          : boosts LOCAL contrast so faint borders become visible.
- Hair removal   : black-hat finds thin dark lines (hairs), inpainting fills them in.
"""
import cv2
import numpy as np

FILTERS = ("none", "gaussian", "median", "bilateral")


class ParameterError(ValueError):
    """Raised when a processing parameter is outside its valid range."""


def validate_odd_kernel(k, name="kernel_size", minimum=1, maximum=31) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ParameterError(f"{name} must be an integer.")
    k = int(k)
    if k < minimum or k > maximum:
        raise ParameterError(f"{name} must be between {minimum} and {maximum}.")
    if k % 2 == 0:
        raise ParameterError(f"{name} must be an odd number (got {k}).")
    return k


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr.copy()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = validate_odd_kernel(kernel_size)
    if k == 1:
        return gray.copy()
    return cv2.GaussianBlur(gray, (k, k), 0)


def median_filter(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = validate_odd_kernel(kernel_size, minimum=3)
    return cv2.medianBlur(gray, k)


def bilateral_filter(gray: np.ndarray, diameter: int = 9, sigma_color: float = 50,
                     sigma_space: float = 50) -> np.ndarray:
    d = validate_odd_kernel(diameter, "bilateral diameter", minimum=3, maximum=25)
    if not (1 <= sigma_color <= 250) or not (1 <= sigma_space <= 250):
        raise ParameterError("Bilateral sigma values must be between 1 and 250.")
    return cv2.bilateralFilter(gray, d, float(sigma_color), float(sigma_space))


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    if not (0.5 <= clip_limit <= 10):
        raise ParameterError("CLAHE clip limit must be between 0.5 and 10.")
    if isinstance(tile_grid, bool) or not isinstance(tile_grid, (int, np.integer)) or not (2 <= tile_grid <= 16):
        raise ParameterError("CLAHE tile grid must be an integer between 2 and 16.")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    return clahe.apply(gray)


def resize_keep_aspect(img: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink so the longest side is max_side. Never enlarges. Aspect ratio is kept."""
    if max_side < 32:
        raise ParameterError("max_side must be at least 32.")
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def remove_hair(bgr: np.ndarray, kernel_size: int = 17, threshold: int = 10) -> np.ndarray:
    """Reduce dark hair strands (DullRazor-style): black-hat -> mask -> inpaint."""
    k = validate_odd_kernel(kernel_size, "hair kernel", minimum=3, maximum=51)
    gray = to_grayscale(bgr)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    _, hair_mask = cv2.threshold(blackhat, int(threshold), 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8))
    return cv2.inpaint(bgr, hair_mask, 3, cv2.INPAINT_TELEA)


DEFAULT_PREPROCESS = {
    "filter": "gaussian", "kernel_size": 5,
    "bilateral_sigma_color": 50, "bilateral_sigma_space": 50,
    "clahe": True, "clahe_clip": 2.0, "clahe_tile": 8,
    "remove_hair": False, "max_side": 1024,
}


def preprocess(bgr: np.ndarray, params: dict | None = None) -> dict:
    """Run the full preprocessing chain.

    Returns {"bgr": resized colour image, "gray": grayscale, "preprocessed": final gray image}
    """
    p = {**DEFAULT_PREPROCESS, **(params or {})}
    if p["filter"] not in FILTERS:
        raise ParameterError(f"filter must be one of: {', '.join(FILTERS)}.")

    img = resize_keep_aspect(bgr, int(p["max_side"]))
    if p["remove_hair"]:
        img = remove_hair(img)
    gray = to_grayscale(img)
    out = gray

    if p["filter"] == "gaussian":
        out = gaussian_blur(out, p["kernel_size"])
    elif p["filter"] == "median":
        out = median_filter(out, p["kernel_size"])
    elif p["filter"] == "bilateral":
        out = bilateral_filter(out, p["kernel_size"], p["bilateral_sigma_color"], p["bilateral_sigma_space"])
    if p["clahe"]:
        out = apply_clahe(out, p["clahe_clip"], p["clahe_tile"])
    return {"bgr": img, "gray": gray, "preprocessed": out}
