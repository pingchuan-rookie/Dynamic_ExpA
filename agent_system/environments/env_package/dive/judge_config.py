"""Non-secret DIVE judge configuration shared by launchers and isolated actors."""
from __future__ import annotations

import math
import os
import re
from urllib.parse import urlsplit

from .model_api.providers.trapi import TRAPI_SCOPE, resolve_trapi_config
JUDGE_ENV_VARS = ("DIVE_JUDGE_PROVIDER", "DIVE_JUDGE_MODEL", "DIVE_JUDGE_BASE_URL",
                  "DIVE_JUDGE_API_KEY_ENV", "OPENAI_MODEL", "OPENAI_BASE_URL", "TRAPI_HOST")
_FIELDS = {"provider", "model", "base_url", "api_key_env", "timeout_s", "max_tokens", "max_completion_tokens"}


def resolve_judge_config(config=None, environ=None):
    """Resolve once and persist the result, never a credential or token provider.

    A nonempty old saved configuration lacking provider means openai_compatible.
    It must not silently acquire the new default or an ambient provider override.
    """
    env = os.environ if environ is None else environ
    source = dict(config or {})
    if source.keys() - _FIELDS:
        raise ValueError("Unsupported DIVE judge configuration fields (credentials must stay outside config)")
    provider = source.get("provider") or ("openai_compatible" if source else env.get("DIVE_JUDGE_PROVIDER") or "trapi")
    if provider not in {"trapi", "openai_compatible", "anthropic"}:
        raise ValueError("Unsupported DIVE judge provider")
    model = source.get("model") or env.get("DIVE_JUDGE_MODEL")
    base_url = source.get("base_url") or env.get("DIVE_JUDGE_BASE_URL")
    if provider == "trapi":
        resolved = resolve_trapi_config(model=model, base_url=base_url, env=env)
        model, base_url = resolved["model"], resolved["base_url"]
    if not isinstance(model, str) or not model.strip() or not isinstance(base_url, str) or not base_url:
        raise ValueError("DIVE legacy judge requires explicit DIVE_JUDGE_MODEL and DIVE_JUDGE_BASE_URL")
    url = urlsplit(base_url)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Judge URL must be HTTP(S) without embedded credentials")
    timeout = float(source.get("timeout_s", 120))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Judge timeout must be finite and positive")
    result = {"provider": provider, "model": model, "base_url": base_url, "timeout_s": timeout}
    budget_key = "max_completion_tokens" if provider == "trapi" else "max_tokens"
    budget = source.get(budget_key, source.get("max_tokens", 2048))
    if isinstance(budget, bool) or int(budget) != float(budget) or int(budget) <= 0:
        raise ValueError("Judge token budget must be a positive integer")
    result[budget_key] = int(budget)
    if provider != "trapi":
        name = source.get("api_key_env") or env.get("DIVE_JUDGE_API_KEY_ENV") or "DIVE_JUDGE_API_KEY"
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("Judge api_key_env must name an environment variable")
        result["api_key_env"] = name
    return result


def restore_judge_config(saved):
    """Restore a saved identity without borrowing model/provider from the current host.

    Old saved environment mappings did not carry provider, whose historical default
    was openai_compatible. Only known non-secret judge fields are selected.
    """
    if "dive_judge" in saved:
        value = saved["dive_judge"]
        if not isinstance(value, dict) or not value.get("model") or not value.get("base_url"):
            raise ValueError("Saved DIVE judge identity is incomplete")
        return resolve_judge_config(value, {})
    env = saved.get("env", {})
    if not isinstance(env, dict):
        raise ValueError("Saved DIVE environment must be an object")
    if env.get("DIVE_JUDGE_CONFIG"):
        import json
        return restore_judge_config({"dive_judge": json.loads(env["DIVE_JUDGE_CONFIG"])})
    if env.get("DIVE_SESSION_CONFIG"):
        import json
        value = json.loads(env["DIVE_SESSION_CONFIG"]).get("judge_config")
        if value:
            return restore_judge_config({"dive_judge": value})
    fields = {"model": "DIVE_JUDGE_MODEL", "base_url": "DIVE_JUDGE_BASE_URL",
              "provider": "DIVE_JUDGE_PROVIDER", "api_key_env": "DIVE_JUDGE_API_KEY_ENV"}
    value = {field: env[key] for field, key in fields.items() if env.get(key)}
    if value:
        return restore_judge_config({"dive_judge": value})
    return None
