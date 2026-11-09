"""Safe file and ID helpers. Clients never send file paths, only IDs."""
import re
import uuid

from werkzeug.utils import secure_filename

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def new_result_id() -> str:
    """Random 32-character hex ID used as a result folder name."""
    return uuid.uuid4().hex


def is_valid_result_id(value: str) -> bool:
    return bool(value) and bool(_ID_PATTERN.match(value))


def safe_filename(name: str, fallback: str = "image") -> str:
    """Strip directories and unsafe characters from a user-supplied filename."""
    cleaned = secure_filename(name or "")
    return cleaned or fallback
