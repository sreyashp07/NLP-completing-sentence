"""
Custom exception classes for the CustomerIntent API.

Using custom exceptions instead of generic ones gives:
- Better error messages in API responses
- Easier debugging and log filtering
- Consistent error handling across the app
"""


class CustomerIntentError(Exception):
    """Base exception for CustomerIntent application."""
    pass


class ModelNotLoadedError(CustomerIntentError):
    """Raised when prediction is attempted without a loaded model."""
    def __init__(self):
        super().__init__(
            "ML model is not loaded. Call load_model() before predicting."
        )


class ModelLoadError(CustomerIntentError):
    """Raised when model artifacts cannot be loaded from disk."""
    def __init__(self, path: str, reason: str):
        super().__init__(f"Failed to load model from {path}: {reason}")


class InvalidTextError(CustomerIntentError):
    """Raised when input text fails validation."""
    def __init__(self, reason: str):
        super().__init__(f"Invalid input text: {reason}")


class PredictionError(CustomerIntentError):
    """Raised when prediction pipeline fails unexpectedly."""
    def __init__(self, reason: str):
        super().__init__(f"Prediction failed: {reason}")
