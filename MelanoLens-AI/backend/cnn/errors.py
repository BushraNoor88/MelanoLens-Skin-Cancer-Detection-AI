"""Exceptions shared by the CNN modules (no TensorFlow import here)."""


class ModelNotAvailable(RuntimeError):
    """A CNN cannot be used. `code` is one of: tensorflow_missing, model_missing, model_invalid."""

    def __init__(self, message, code="model_unavailable"):
        super().__init__(message)
        self.code = code


class DatasetError(ValueError):
    """The dataset on disk cannot be used as-is. The message says what to fix."""
