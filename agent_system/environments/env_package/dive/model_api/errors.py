"""Exceptions never retain SDK requests, credentials, or response bodies."""

from contextlib import contextmanager


class ModelAPIError(RuntimeError):
    def __init__(self, message, *, error_type=None, status_code=None):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


@contextmanager
def sanitized_errors():
    try:
        yield
    except ModelAPIError:
        raise
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if type(status) is not int or not 100 <= status <= 599:
            status = None
        kind = type(exc).__name__
        raise ModelAPIError(f"Model API request failed ({kind})", error_type=kind, status_code=status) from None
