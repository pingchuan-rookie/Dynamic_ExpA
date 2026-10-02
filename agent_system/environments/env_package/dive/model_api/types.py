"""Nonsecret connection settings and synchronous text generation values."""

import math
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str | None = None

    def __post_init__(self):
        if self.provider not in {"openai_compatible", "trapi", "anthropic"}:
            raise ValueError("Unsupported model API provider")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Model name must be nonempty")
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Model URL must be HTTP(S) without credentials, query or fragment")
        if self.api_key_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("api_key_env must name an environment variable")
        if self.provider == "trapi" and self.api_key_env is not None:
            raise ValueError("TRAPI uses worker-local Azure credentials")


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self):
        if self.role not in {"system", "user", "assistant"} or not isinstance(self.content, str):
            raise ValueError("Text messages require a system/user/assistant role and string content")


@dataclass(frozen=True)
class GenerateOptions:
    max_output_tokens: int = 2048
    temperature: float | None = None

    def __post_init__(self):
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if self.temperature is not None and (not math.isfinite(self.temperature) or self.temperature < 0):
            raise ValueError("temperature must be finite and nonnegative")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ModelResponse:
    text: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    request_params: dict = field(default_factory=dict)
