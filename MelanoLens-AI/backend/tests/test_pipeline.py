import numpy as np

from services import synthetic
from services.params import parse_process_params
from services.pipeline import run_pipeline


def _framed(width=2):
    img, ref = synthetic.circle()
    img = img.copy()
    img[:width] = 0; img[-width:] = 0; img[:, :width] = 0; img[:, -width:] = 0
    return img, ref


def test_frame_edges_are_ignored_by_default():
    img, ref = _framed(2)
    out = run_pipeline(img, parse_process_params({"mode": "conventional"}), ref)
    for name, e in out["images"].items():
        if name.startswith("edges_"):
            assert not e[:5].any() and not e[-5:].any() and not e[:, :5].any() and not e[:, -5:].any(), name
    b = out["meta"]["boundary"]
    assert b["selected_index"] is not None and not b["measurements"]["touches_border"]
    assert abs(b["measurements"]["area_px"] - np.pi * 70 ** 2) / (np.pi * 70 ** 2) < 0.08


def test_without_margin_a_frame_hijacks_the_contour_and_a_warning_is_given():
    img, ref = _framed(2)
    out = run_pipeline(img, parse_process_params({"ignore_border_px": 0}), ref)
    b = out["meta"]["boundary"]
    assert b["measurements"] is None or b["measurements"]["touches_border"]
    assert any("whole image" in w or "No reliable" in w for w in out["meta"]["warnings"])


def test_margin_parameter_is_validated():
    import pytest
    from services.preprocessing import ParameterError
    for bad in (-1, 31, 2.5, "3"):
        with pytest.raises(ParameterError):
            parse_process_params({"ignore_border_px": bad})


def test_pipeline_reports_which_canny_thresholds_were_used():
    img, ref = synthetic.circle()
    auto = run_pipeline(img, parse_process_params({}), ref)["meta"]["canny_thresholds_used"]
    assert auto["conventional"]["auto"] and auto["enhanced"]["high"] >= 20
    manual = run_pipeline(img, parse_process_params({"canny_auto": False, "canny_low": 40, "canny_high": 120}), ref)
    assert manual["meta"]["canny_thresholds_used"]["enhanced"] == {"low": 40.0, "high": 120.0, "auto": False}


def test_one_pixel_frame_on_a_real_sized_image_is_neutralised():
    img, ref = synthetic.circle(size=300)
    img = img.copy(); img[0] = 255; img[-1] = 255; img[:, 0] = 255; img[:, -1] = 255     # 1 px white frame
    out = run_pipeline(img, parse_process_params({"mode": "conventional", "kernel_size": 9, "canny_blur": 9}), ref)
    m = out["meta"]["boundary"]["measurements"]
    assert m is not None and not m["touches_border"] and m["area_px"] > 10000
