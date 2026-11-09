import numpy as np
import pytest

from services import synthetic
from services.edge_booster import (edge_boost, morphological_gradient, multiscale_gradient,
                                   normalize_to_uint8)
from services.preprocessing import ParameterError, to_grayscale


def _gray():
    return to_grayscale(synthetic.circle()[0])


def test_normalize_constant_image_is_zero_not_nan():
    out = normalize_to_uint8(np.full((10, 10), 7.0))
    assert out.dtype == np.uint8 and out.max() == 0


def test_all_methods_return_uint8_same_shape():
    g = _gray()
    for m in ("sobel", "laplacian", "morphological", "multiscale"):
        r = edge_boost(g, m, alpha=1.0, kernel_size=3)
        assert r["enhanced"].shape == g.shape and r["enhanced"].dtype == np.uint8
        assert r["gradient"].shape == g.shape


def test_alpha_zero_sobel_is_just_stretched_original():
    g = _gray()
    r = edge_boost(g, "sobel", alpha=0.0)
    assert np.array_equal(r["enhanced"], normalize_to_uint8(g))


def test_higher_alpha_changes_output():
    g = _gray()
    a = edge_boost(g, "sobel", alpha=0.5)["enhanced"]
    b = edge_boost(g, "sobel", alpha=3.0)["enhanced"]
    assert not np.array_equal(a, b)


def test_invalid_parameters():
    g = _gray()
    with pytest.raises(ParameterError):
        edge_boost(g, "sobel", alpha=-1)
    with pytest.raises(ParameterError):
        edge_boost(g, "sobel", alpha=99)
    with pytest.raises(ParameterError):
        edge_boost(g, "unknown")
    with pytest.raises(ParameterError):
        edge_boost(g, "morphological", kernel_size=4)
    with pytest.raises(ParameterError):
        edge_boost(np.zeros((10, 10, 3), np.uint8), "sobel")


def test_morphological_gradient_peaks_on_boundary():
    g = _gray()
    mg = morphological_gradient(g, 3)
    assert mg[128, 128] == 0                       # flat inside the lesion
    assert mg.max() > 50                           # strong response on the edge


def test_multiscale_weights_validation():
    g = _gray()
    out = multiscale_gradient(g, (1, 2), (1, 3))
    assert 0.0 <= out.min() and out.max() <= 1.0 + 1e-6
    with pytest.raises(ParameterError):
        multiscale_gradient(g, (1, 2), (1,))
    with pytest.raises(ParameterError):
        multiscale_gradient(g, (0, 2))
