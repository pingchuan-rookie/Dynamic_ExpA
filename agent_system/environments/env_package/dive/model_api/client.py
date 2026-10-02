"""Lazy synchronous clients; each generation owns and closes its SDK resources."""

import math
from collections.abc import Sequence
from typing import Protocol

from .errors import sanitized_errors
from .types import GenerateOptions, Message, ModelConfig, ModelResponse


class ModelClient(Protocol):
    def generate(self, messages: Sequence[Message], *, options: GenerateOptions) -> ModelResponse: ...
    def close(self) -> None: ...


class _Client:
    def __init__(self, config, timeout_s, max_retries):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.config = config
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._closed = False

    def generate(self, messages: Sequence[Message], *, options: GenerateOptions) -> ModelResponse:
        if self._closed:
            raise RuntimeError("Model client is closed")
        if not messages:
            raise ValueError("At least one message is required")
        with sanitized_errors():
            if self.config.provider == "anthropic":
                from .providers.anthropic import generate
            else:
                from .providers.openai_compatible import generate
            return generate(self.config, messages, options, self.timeout_s, self.max_retries)

    def close(self):
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Model client is closed")
        return self

    def __exit__(self, *exc):
        self.close()


def create_model_client(config: ModelConfig, *, timeout_s: float = 120, max_retries: int = 0):
    """Create in the calling worker. No credentials or connections are opened until generate."""
    return _Client(config, timeout_s, max_retries)
