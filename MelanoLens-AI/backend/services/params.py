"""Server-side validation of /api/process parameters (the frontend validates too, but never trusted)."""
from .boundary_extraction import SELECTIONS
from .edge_booster import METHODS
from .edge_detection import DETECTORS
from .preprocessing import FILTERS, ParameterError

MODES = ("conventional", "enhanced")

DEFAULTS = {
    # preprocessing
    "filter": "gaussian", "kernel_size": 5, "bilateral_sigma_color": 50, "bilateral_sigma_space": 50,
    "clahe": True, "clahe_clip": 2.0, "clahe_tile": 8, "remove_hair": False, "max_side": 1024,
    # edge booster
    "booster_method": "sobel", "booster_alpha": 1.0, "booster_kernel": 3,
    "multiscale_sigmas": [1.0, 2.0, 4.0], "multiscale_weights": None,
    # detectors (same values are used in conventional and enhanced mode)
    "detector": "canny", "mode": "enhanced",
    "canny_auto": True, "canny_percentile": 85.0, "canny_low": 50, "canny_high": 150, "canny_blur": 5,
    "sobel_threshold": 60, "prewitt_threshold": 60, "laplacian_threshold": 40,
    "log_sigma": 1.5, "log_threshold": 40,
    # boundary extraction
    "close_kernel": 5, "close_iterations": 1, "open_kernel": 0,
    "min_area_pct": 0.5, "max_area_pct": 95.0, "selection": "largest", "candidate_index": None,
    "boundary_tolerance_px": 2, "ignore_border_px": 5,
}


def _num(p, key, lo, hi, integer=False):
    v = p[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ParameterError(f"{key} must be a number.")
    if integer and float(v) != int(v):
        raise ParameterError(f"{key} must be a whole number.")
    if not lo <= v <= hi:
        raise ParameterError(f"{key} must be between {lo} and {hi}.")
    return int(v) if integer else float(v)


def _bool(p, key):
    if not isinstance(p[key], bool):
        raise ParameterError(f"{key} must be true or false.")
    return p[key]


def _choice(p, key, options):
    if p[key] not in options:
        raise ParameterError(f"{key} must be one of: {', '.join(options)}.")
    return p[key]


def parse_process_params(raw):
    """Validate a flat dict of user parameters. Returns the complete, validated flat dict."""
    if not isinstance(raw, dict):
        raise ParameterError("params must be a JSON object.")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ParameterError(f"Unknown parameter(s): {', '.join(unknown)}.")
    p = {**DEFAULTS, **raw}

    out = {
        "filter": _choice(p, "filter", FILTERS),
        "kernel_size": _num(p, "kernel_size", 1, 31, True),
        "bilateral_sigma_color": _num(p, "bilateral_sigma_color", 1, 250),
        "bilateral_sigma_space": _num(p, "bilateral_sigma_space", 1, 250),
        "clahe": _bool(p, "clahe"),
        "clahe_clip": _num(p, "clahe_clip", 0.5, 10),
        "clahe_tile": _num(p, "clahe_tile", 2, 16, True),
        "remove_hair": _bool(p, "remove_hair"),
        "max_side": _num(p, "max_side", 64, 2048, True),
        "booster_method": _choice(p, "booster_method", METHODS),
        "booster_alpha": _num(p, "booster_alpha", 0, 5),
        "booster_kernel": _num(p, "booster_kernel", 1, 21, True),
        "detector": _choice(p, "detector", DETECTORS),
        "mode": _choice(p, "mode", MODES),
        "canny_auto": _bool(p, "canny_auto"),
        "canny_percentile": _num(p, "canny_percentile", 50, 99.5),
        "canny_low": _num(p, "canny_low", 0, 255),
        "canny_high": _num(p, "canny_high", 0, 255),
        "canny_blur": _num(p, "canny_blur", 1, 31, True),
        "sobel_threshold": _num(p, "sobel_threshold", 0, 255),
        "prewitt_threshold": _num(p, "prewitt_threshold", 0, 255),
        "laplacian_threshold": _num(p, "laplacian_threshold", 0, 255),
        "log_sigma": _num(p, "log_sigma", 0.5, 10),
        "log_threshold": _num(p, "log_threshold", 0, 255),
        "close_kernel": _num(p, "close_kernel", 1, 31, True),
        "close_iterations": _num(p, "close_iterations", 1, 5, True),
        "open_kernel": _num(p, "open_kernel", 0, 15, True),
        "min_area_pct": _num(p, "min_area_pct", 0, 99),
        "max_area_pct": _num(p, "max_area_pct", 1, 100),
        "selection": _choice(p, "selection", SELECTIONS),
        "boundary_tolerance_px": _num(p, "boundary_tolerance_px", 0, 20),
        "ignore_border_px": _num(p, "ignore_border_px", 0, 30, True),
    }

    sig = p["multiscale_sigmas"]
    if (not isinstance(sig, list) or not 1 <= len(sig) <= 5
            or any(isinstance(s, bool) or not isinstance(s, (int, float)) or not 0 < s <= 10 for s in sig)):
        raise ParameterError("multiscale_sigmas must be a list of 1-5 numbers in (0, 10].")
    w = p["multiscale_weights"]
    if w is not None and (not isinstance(w, list) or len(w) != len(sig)
                          or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x < 0 for x in w)
                          or sum(w) <= 0):
        raise ParameterError("multiscale_weights must match multiscale_sigmas, be non-negative, and not all zero.")
    out["multiscale_sigmas"] = [float(s) for s in sig]
    out["multiscale_weights"] = None if w is None else [float(x) for x in w]

    ci = p["candidate_index"]
    if ci is not None and (isinstance(ci, bool) or not isinstance(ci, int) or ci < 0):
        raise ParameterError("candidate_index must be null or a non-negative integer.")
    out["candidate_index"] = ci

    if not out["canny_auto"] and out["canny_low"] >= out["canny_high"]:
        raise ParameterError("canny_low must be smaller than canny_high.")
    if out["min_area_pct"] >= out["max_area_pct"]:
        raise ParameterError("min_area_pct must be smaller than max_area_pct.")
    return out
