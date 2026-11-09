import numpy as np
import pytest

from services import synthetic
from services.edge_detection import (DETECTORS, canny, laplacian, laplacian_of_gaussian, prewitt,
                                     run_detector, sobel)
from services.preprocessing import ParameterError, to_grayscale

CENTER, RADIUS = (128, 128), 70


def _gray(**kw):
    return to_grayscale(synthetic.circle(**kw)[0])


def _radii(edges):
    ys, xs = np.nonzero(edges)
    return np.hypot(ys - CENTER[0], xs - CENTER[1])


def test_each_detector_finds_edges_near_the_circle():
    g = _gray()
    for name in DETECTORS:
        edges, resp = run_detector(name, g, {})
        assert edges.dtype == np.uint8 and set(np.unique(edges)) <= {0, 255}
        assert resp.shape == g.shape
        r = _radii(edges)
        assert len(r) > 50, name
        assert np.median(np.abs(r - RADIUS)) < 4, name      # edges sit on the boundary


def test_flat_image_has_no_edges():
    flat = np.full((64, 64), 120, np.uint8)
    for fn in (canny, sobel, prewitt, laplacian):
        edges, _ = fn(flat)
        assert edges.max() == 0


def test_prewitt_is_not_canny():
    g = _gray()
    e_p, _ = prewitt(g, 60)
    e_c, _ = canny(g, 50, 150, 5)
    assert not np.array_equal(e_p, e_c)


def test_higher_threshold_gives_fewer_edge_pixels():
    g = _gray(noise_sigma=5)
    low = (sobel(g, 20)[0] > 0).sum()
    high = (sobel(g, 150)[0] > 0).sum()
    assert high <= low


def test_parameter_validation():
    g = _gray()
    with pytest.raises(ParameterError):
        canny(g, 150, 50)
    with pytest.raises(ParameterError):
        canny(g, -1, 100)
    with pytest.raises(ParameterError):
        canny(g, 50, 150, 4)
    with pytest.raises(ParameterError):
        sobel(g, 300)
    with pytest.raises(ParameterError):
        laplacian_of_gaussian(g, sigma=0)
    with pytest.raises(ParameterError):
        run_detector("nope", g)
    with pytest.raises(ParameterError):
        canny(np.zeros((10, 10, 3), np.uint8))


def test_auto_canny_thresholds_scale_with_the_image_and_have_floors():
    from services.edge_detection import canny_auto_thresholds
    g = _gray()
    low, high = canny_auto_thresholds(g, 85, 5)
    assert high >= 20 and low >= 8 and low < high
    flat = np.full((64, 64), 100, np.uint8)
    assert canny_auto_thresholds(flat, 85, 5) == (8.0, 20.0)                # floors on a flat image
    strong = np.clip(g.astype(int) * 2 - 100, 0, 255).astype(np.uint8)      # more contrast -> higher thresholds
    assert canny_auto_thresholds(strong, 60, 5)[1] >= canny_auto_thresholds(g, 60, 5)[1]
    with pytest.raises(ParameterError):
        canny_auto_thresholds(g, 20, 5)
    edges, _ = run_detector("canny", g, {"canny_auto": True, "canny_percentile": 85})
    assert np.median(np.abs(_radii(edges) - RADIUS)) < 4
