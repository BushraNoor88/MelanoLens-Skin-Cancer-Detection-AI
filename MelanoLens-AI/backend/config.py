"""Central configuration for MelanoLens AI.

Every value can be overridden with an environment variable (or a .env file),
so you never need to edit the code to change limits or folders.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # the melanolens-ai/ folder


def _path(env_name: str, default):
    value = os.getenv(env_name)
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else BASE_DIR / p


class Config:
    APP_NAME = "MelanoLens AI"
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024  # Flask rejects bigger requests
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    MIN_IMAGE_SIDE = 32
    MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1024"))  # larger images are resized
    ISIC_DIR = _path("ISIC_DIR", BASE_DIR / "data" / "isic")
    RESULTS_DIR = _path("RESULTS_DIR", BASE_DIR / "data" / "results")
    FRONTEND_DIR = BASE_DIR / "frontend"

    # ---- CNN modules (trained separately; see README "CNN modules") ----
    # Dataset locations. The labels CSV is auto-detected inside ISIC_DIR unless ISIC_LABELS_CSV is set.
    ISIC_IMAGES_DIR = _path("ISIC_IMAGES_DIR", ISIC_DIR / "images")
    ISIC_MASKS_DIR = _path("ISIC_MASKS_DIR", ISIC_DIR / "masks")
    ISIC_LABELS_CSV = _path("ISIC_LABELS_CSV", None)
    # Trained models + their sidecar files (metadata, history, split, evaluation) live here.
    MODELS_DIR = _path("MODELS_DIR", BASE_DIR / "backend" / "models")
    SEG_MODEL_PATH = _path("SEG_MODEL_PATH", MODELS_DIR / "unet_skin_lesion.keras")
    CLS_MODEL_PATH = _path("CLS_MODEL_PATH", MODELS_DIR / "skin_lesion_classifier.keras")
