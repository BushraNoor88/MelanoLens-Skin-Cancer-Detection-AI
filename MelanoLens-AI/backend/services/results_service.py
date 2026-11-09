"""Controlled storage for uploads and generated results.

Layout:  <RESULTS_DIR>/<result_id>/{meta.json, _source.png, _reference.png, <image_type>.png ...}
Files starting with "_" are internal and are never served. Clients only ever
send a result_id (32 hex chars) and an image_type (lowercase letters/digits/_),
so they cannot point the server at arbitrary paths.
"""
import json
import re
import shutil
from pathlib import Path

import cv2

from utils.file_utils import is_valid_result_id

IMAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9_]{0,47}$")


class ResultNotFound(LookupError):
    pass


class ResultStateError(RuntimeError):
    pass


class ResultStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, rid):
        if not isinstance(rid, str) or not is_valid_result_id(rid) or not (self.root / rid).is_dir():
            raise ResultNotFound("Result not found. Upload an image first.")
        return self.root / rid

    def require(self, rid):
        self._dir(rid)

    def create(self, rid, source_bgr, meta, reference=None):
        d = self.root / rid
        d.mkdir(parents=True)
        self._write_png(d / "_source.png", source_bgr)
        if reference is not None:
            self._write_png(d / "_reference.png", reference)
        self.write_meta(rid, meta)

    @staticmethod
    def _write_png(path, arr):
        if not cv2.imwrite(str(path), arr):
            raise RuntimeError(f"Could not write {path.name}")

    def load_source(self, rid):
        return cv2.imread(str(self._dir(rid) / "_source.png"), cv2.IMREAD_COLOR)

    def load_reference(self, rid):
        p = self._dir(rid) / "_reference.png"
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None

    def clear_images(self, rid):
        """Remove classical-pipeline images. CNN outputs (cnn_*) belong to the CNN endpoints and are kept."""
        for p in self._dir(rid).glob("*.png"):
            if not p.name.startswith("_") and not p.name.startswith("cnn_"):
                p.unlink()

    def save_images(self, rid, images):
        d = self._dir(rid)
        for name, arr in images.items():
            if not IMAGE_NAME.match(name):
                raise ValueError(f"Bad image name: {name}")
            self._write_png(d / f"{name}.png", arr)

    def list_images(self, rid):
        return sorted(p.stem for p in self._dir(rid).glob("*.png") if not p.name.startswith("_"))

    def image_path(self, rid, name):
        d = self._dir(rid)
        if not isinstance(name, str) or not IMAGE_NAME.match(name):
            return None
        p = d / f"{name}.png"
        return p if p.is_file() else None

    def read_gray(self, rid, name):
        p = self.image_path(rid, name)
        return None if p is None else cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

    def read_meta(self, rid):
        return json.loads((self._dir(rid) / "meta.json").read_text(encoding="utf-8"))

    def write_meta(self, rid, meta):
        d = self.root / rid
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp.replace(d / "meta.json")

    def delete(self, rid):
        shutil.rmtree(self._dir(rid))
