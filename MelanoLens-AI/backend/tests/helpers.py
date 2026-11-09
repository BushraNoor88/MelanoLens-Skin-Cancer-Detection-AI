"""Shared helpers for API tests: isolated results folder and in-memory PNGs."""
import io
import shutil
import tempfile
from pathlib import Path

import cv2

from app import create_app
from config import Config


def make_client():
    tmp = Path(tempfile.mkdtemp(prefix="melanolens_test_"))

    class TestConfig(Config):
        RESULTS_DIR = tmp
        TESTING = True

    return create_app(TestConfig).test_client(), tmp


def cleanup(tmp):
    shutil.rmtree(tmp, ignore_errors=True)


def png_bytes(arr):
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def upload(client, image, mask=None, name="lesion.png"):
    data = {"image": (io.BytesIO(png_bytes(image)), name)}
    if mask is not None:
        data["mask"] = (io.BytesIO(png_bytes(mask)), "mask.png")
    return client.post("/api/upload", data=data, content_type="multipart/form-data")
