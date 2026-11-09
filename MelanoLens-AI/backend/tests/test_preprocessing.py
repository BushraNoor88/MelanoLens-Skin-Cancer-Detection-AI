import numpy as np
import pytest

from services import synthetic
from services.preprocessing import (ParameterError, apply_clahe, bilateral_filter, gaussian_blur,
                                    median_filter, preprocess, remove_hair, resize_keep_aspect,
                                    to_grayscale, validate_odd_kernel)


def test_grayscale_from_rgb_and_gray():
    bgr, _ = synthetic.circle()
    gray = to_grayscale(bgr)
    assert gray.ndim == 2 and gray.dtype == np.uint8
    assert to_grayscale(gray).shape == gray.shape        # already gray is accepted


def test_even_kernel_rejected():
    with pytest.raises(ParameterError):
        validate_odd_kernel(4)
    with pytest.raises(ParameterError):
        gaussian_blur(np.zeros((40, 40), np.uint8), 4)


def test_filters_keep_shape_and_reduce_noise():
    bgr, _ = synthetic.noisy_shape()
    gray = to_grayscale(bgr)
    clean_bg = gray[:40, :40].astype(float)
    for f in (gaussian_blur, median_filter):
        out = f(gray, 5)
        assert out.shape == gray.shape
        assert out[:40, :40].std() < clean_bg.std()
    assert bilateral_filter(gray, 9, 50, 50).shape == gray.shape


def test_clahe_validation():
    gray = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    assert apply_clahe(gray).shape == gray.shape
    with pytest.raises(ParameterError):
        apply_clahe(gray, clip_limit=100)


def test_resize_keeps_aspect_and_never_enlarges():
    img = np.zeros((200, 400, 3), np.uint8)
    out = resize_keep_aspect(img, 100)
    assert out.shape[:2] == (50, 100)
    assert resize_keep_aspect(img, 1000).shape == img.shape


def test_hair_removal_returns_same_shape():
    bgr, _ = synthetic.circle()
    bgr[100:103, :, :] = 0                                   # fake hair line
    out = remove_hair(bgr)
    assert out.shape == bgr.shape
    assert out[101, 10].mean() > 100                          # hair was filled in


def test_preprocess_pipeline_and_bad_filter():
    bgr, _ = synthetic.circle()
    res = preprocess(bgr, {"filter": "median", "kernel_size": 5, "clahe": True})
    assert res["preprocessed"].shape == res["gray"].shape
    with pytest.raises(ParameterError):
        preprocess(bgr, {"filter": "nonsense"})
