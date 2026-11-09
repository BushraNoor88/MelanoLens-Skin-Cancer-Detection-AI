"""Safe image decoding and validation.

Everything returned from here is a 3-channel BGR uint8 array, the format
OpenCV expects. Grayscale, 16-bit and RGBA files are converted first, so the
rest of the pipeline only deals with one format.
"""
import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not an acceptable image."""

    def __init__(self, message: str, code: str = "invalid_image"):
        super().__init__(message)
        self.code = code


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if filename and "." in filename else ""


def to_bgr_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert any decoded array (gray / BGR / BGRA, 8 or 16 bit) to BGR uint8."""
    if arr.dtype == np.uint16:
        arr = (arr / 257.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 4:
        # Blend transparent pixels onto white so they do not become black.
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        return (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    raise ImageValidationError("Unsupported number of image channels.", "unsupported_channels")


def decode_image(data: bytes, filename: str, allowed_extensions, max_bytes: int,
                 min_side: int = 32) -> np.ndarray:
    """Validate raw upload bytes and return a BGR uint8 image."""
    if not data:
        raise ImageValidationError("The uploaded file is empty.", "empty_file")

    ext = _extension(filename)
    if ext not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions)).upper()
        raise ImageValidationError(f"Invalid file type '.{ext}'. Allowed types: {allowed}.",
                                   "invalid_file_type")

    if len(data) > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise ImageValidationError(f"File is larger than the {mb:.0f} MB limit.", "file_too_large")

    # Pillow checks that the file really is a JPEG/PNG and is not truncated.
    try:
        with Image.open(io.BytesIO(data)) as pil_img:
            fmt = pil_img.format
            pil_img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageValidationError("The file is corrupted or is not a valid image.", "corrupted_image")

    if fmt not in {"JPEG", "PNG"}:
        raise ImageValidationError("Only real JPEG or PNG content is accepted.", "invalid_file_type")

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ImageValidationError("The image could not be decoded.", "corrupted_image")

    img = to_bgr_uint8(arr)
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        raise ImageValidationError(f"Image is too small. Minimum side is {min_side}px.", "image_too_small")
    return img
