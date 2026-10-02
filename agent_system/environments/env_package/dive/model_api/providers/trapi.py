"""TRAPI configuration and authenticated requests for DIVE judge and browse calls.

Only nonsecret configuration crosses worker boundaries. Credentials are constructed
in the calling worker and never stored in configuration, environment or results.
"""

from __future__ import annotations

import os
import re
from contextlib import ExitStack, contextmanager
from urllib.parse import urlsplit

from ..errors import ModelAPIError

TRAPI_ENV_VARS = ("OPENAI_MODEL", "OPENAI_BASE_URL", "TRAPI_HOST", "AZURE_CONFIG_DIR")
TRAPI_DEFAULT_MODEL = "gpt-5.4-mini_2026-03-17"
TRAPI_DEFAULT_BASE_URL = "https://trapi.research.microsoft.com/redmond/interactive/openai/v1"
TRAPI_SCOPE = "api://trapi/.default"


class TrapiTransportError(ModelAPIError):
    """Sanitized failure retaining only nonsecret classification metadata."""

    def __init__(self, message, *, error_type=None, status_code=None):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


def resolve_trapi_config(model=None, base_url=None, env=None):
    """Resolve explicit role overrides before shared TRAPI settings and defaults."""
    env = os.environ if env is None else env
    model = model or env.get("OPENAI_MODEL") or TRAPI_DEFAULT_MODEL
    base_url = base_url or env.get("OPENAI_BASE_URL")
    if not base_url:
        host = env.get("TRAPI_HOST")
        if host:
            host = host.rstrip("/")
            if "://" not in host:
                host = "https://" + host
            if urlsplit(host).path.rstrip("/"):
                raise ValueError("TRAPI_HOST must be an origin; use OPENAI_BASE_URL for a custom path")
            base_url = host + "/redmond/interactive/openai/v1"
        else:
            base_url = TRAPI_DEFAULT_BASE_URL
    url = urlsplit(base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("TRAPI base URL must be HTTP(S) without credentials, query or fragment")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("TRAPI model must be a nonempty deployment name")
    return {"model": model, "base_url": base_url.rstrip("/")}


def normalize_trapi_request(model, params):
    """Copy request arguments and apply only known GPT-5 compatibility rules.

    The GPT-5 profile uses max_completion_tokens and omits legacy sampling
    controls for compatibility with reasoning defaults. Keep budgets, tools and other args;
    unknown arguments still fail normally rather than being silently discarded.
    """
    params = dict(params)
    if re.match(r"^gpt-5(?:[.\-_]|$)", str(model).lower()):
        if "max_tokens" in params:
            budget = params.pop("max_tokens")
            if "max_completion_tokens" in params and params["max_completion_tokens"] != budget:
                raise ValueError("Conflicting TRAPI output token budgets")
            params["max_completion_tokens"] = budget
        for name in ("temperature", "top_p", "seed"):
            params.pop(name, None)
    return params


@contextmanager
def trapi_client(base_url, timeout, max_retries=0):
    """Yield a plain OpenAI v1 client with Azure CLI then Managed Identity auth.

    The SDK calls the provider per request; Azure Identity handles token refresh.
    Exceptions are sanitized before callers or upstream libraries can print them.
    BaseException deadlines remain intact while both resources are still closed.
    """
    try:
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )
        from openai import OpenAI

        with ExitStack() as resources:
            credential = ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential())
            resources.callback(credential.close)
            provider = get_bearer_token_provider(credential, TRAPI_SCOPE)
            client = resources.enter_context(
                OpenAI(
                    base_url=base_url,
                    api_key=provider,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )
            yield client
    except TrapiTransportError:
        raise
    except Exception as exc:
        error_type = type(exc).__name__
        status_code = getattr(exc, "status_code", None)
        if type(status_code) is not int or not 100 <= status_code <= 599:
            status_code = None
        raise TrapiTransportError(
            f"TRAPI request failed ({error_type})", error_type=error_type, status_code=status_code
        ) from None
