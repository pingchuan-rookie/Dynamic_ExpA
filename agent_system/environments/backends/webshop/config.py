"""Portable WebShop runtime configuration (standard library only)."""
from __future__ import annotations

import math
import os
from pathlib import Path


def resolve_webshop_config(config: dict) -> dict:
    # Code root is Dynamic_ExpA; repository/data roots may be mounted elsewhere.
    code_root = Path(__file__).resolve().parents[4]
    project_root = code_root
    backend = dict(config.get("backend_config", {}) or {})
    root = Path(os.path.expandvars(os.path.expanduser(str(
        os.environ.get("WEBSHOP_REPO") or os.environ.get("WEBSHOP_ROOT")
        or backend.get("source_dir") or config.get("webshop_root") or project_root / "agent_system/environments/env_package/webshop/source"
    ))))
    if not root.is_absolute():
        root = code_root / root

    def path_value(env_key, key, default, alias=None):
        path = Path(os.path.expandvars(os.path.expanduser(str(os.environ.get(env_key) or (os.environ.get(alias) if alias else None) or config.get(key) or default))))
        return str(path if path.is_absolute() else code_root / path)

    def positive(key, env_key, default, cast=float):
        value = cast(os.environ.get(env_key) or config.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        return value

    # Reject benchmark-changing options rather than silently using a reduced catalog.
    if backend.get("num_products") is not None or backend.get("human_goals", True) is not True:
        raise ValueError("WebShop experiments require the full catalog and human goals")
    backend.update({"webshop_root": str(root), "source_dir": str(root), "num_products": None, "human_goals": True})
    backend["assets_dir"] = path_value(
        "WEBSHOP_ASSETS", "webshop_data_dir", backend.get("assets_dir") or code_root / "data/webshop/assets",
        "WEBSHOP_DATA_DIR",
    )
    for key in ("assets_dir", "source_dir", "java_home"):
        if key in backend:
            value = Path(os.path.expandvars(os.path.expanduser(str(backend[key]))))
            backend[key] = str(value if value.is_absolute() else code_root / value)
    return {
        "python_executable": path_value("WEBSHOP_PYTHON_BIN", "python_executable", project_root / ".venvs/webshop/bin/python", "WEBSHOP_PYTHON"),
        "backend_script": str(Path(__file__).parent / "backend.py"),
        "backend_config": backend,
        "backend_replicas": positive("backend_replicas", "WEBSHOP_BACKEND_REPLICAS", 1, int),
        "sessions_per_backend": positive("sessions_per_backend", "WEBSHOP_SESSIONS_PER_BACKEND", 64, int),
        "num_cpus_per_worker": positive("num_cpus_per_worker", "WEBSHOP_CPUS_PER_BACKEND", 1.0),
        "startup_timeout": positive("startup_timeout", "WEBSHOP_STARTUP_TIMEOUT_S", 600.0),
        "request_timeout": positive("request_timeout", "WEBSHOP_REQUEST_TIMEOUT_S", 120.0),
        "lease_timeout": positive("lease_timeout", "WEBSHOP_LEASE_TIMEOUT_S", 600.0),
    }
