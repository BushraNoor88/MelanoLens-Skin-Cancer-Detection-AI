import cv2
import numpy as np
import pytest

from config import Config
from utils.image_validation import ImageValidationError, decode_image
from utils.file_utils import is_valid_result_id, new_result_id, safe_filename

MAX = 1024 * 1024


def _encode(arr, ext=".png"):
    ok, buf = cv2.imencode(ext, arr)
    assert ok
    return buf.tobytes()


def _decode(data, name="a.png", max_bytes=MAX):
    return decode_image(data, name, Config.ALLOWED_EXTENSIONS, max_bytes)


def test_valid_rgb_gray_and_rgba():
    rgb = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert _decode(_encode(rgb)).shape == (64, 64, 3)
    assert _decode(_encode(rgb[:, :, 0])).shape == (64, 64, 3)          # gray -> BGR
    rgba = np.dstack([rgb, np.full((64, 64), 255, np.uint8)])
    assert _decode(_encode(rgba)).shape == (64, 64, 3)
    assert _decode(_encode(rgb, ".jpg"), "a.jpg").dtype == np.uint8


def test_transparent_pixels_become_white():
    rgba = np.zeros((64, 64, 4), np.uint8)                                # fully transparent
    out = _decode(_encode(rgba))
    assert out.min() == 255


def test_invalid_extension_empty_big_corrupt_small():
    rgb = np.zeros((64, 64, 3), np.uint8)
    with pytest.raises(ImageValidationError) as e:
        _decode(_encode(rgb), "a.gif")
    assert e.value.code == "invalid_file_type"
    with pytest.raises(ImageValidationError):
        _decode(b"")
    with pytest.raises(ImageValidationError) as e:
        _decode(_encode(rgb), max_bytes=10)
    assert e.value.code == "file_too_large"
    with pytest.raises(ImageValidationError) as e:
        _decode(b"this is not an image", "fake.png")
    assert e.value.code == "corrupted_image"
    with pytest.raises(ImageValidationError) as e:
        _decode(_encode(np.zeros((8, 8, 3), np.uint8)))
    assert e.value.code == "image_too_small"


def test_truncated_png_is_rejected():
    data = _encode(np.random.default_rng(2).integers(0, 255, (128, 128, 3), dtype=np.uint8))
    with pytest.raises(ImageValidationError):
        _decode(data[: len(data) // 2])


def test_safe_filename_and_ids():
    assert "/" not in safe_filename("../../etc/passwd")
    assert safe_filename("") == "image"
    rid = new_result_id()
    assert is_valid_result_id(rid) and not is_valid_result_id("../x")
