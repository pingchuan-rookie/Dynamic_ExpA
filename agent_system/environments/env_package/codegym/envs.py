# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Standalone codegym environment: reset, step, close and environment health checks."""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Optional


# Shared with CodeGymEnvPool so configuration and direct workers agree.
ACTION_TIMEOUT_DEFAULT_S = 10.0
# Two 5s framework phases plus bounded cleanup and scheduling headroom.
# Parent dill work and launch are ultimately bounded by the outer RPC watchdog.
ACTION_RPC_HEADROOM_S = 12.0


# =========================================================================
# env_str parsing + on-demand env class loading (mirrors CodeGym server_config.get_class)
# =========================================================================
def parse_codegym_env_str(env_str: str) -> tuple[str, str, str]:
    """Parse an ability string into (filename, class_name, inner_env_str).

    The input looks like ``codegym_v1@<source>__<EnvName>@<init_json>``
      - filename       : ``<source>__<EnvName>.py`` (the env file name)
      - class_name     : ``<EnvName>`` (the part after the first ``__`` in the file name, minus .py)
      - inner_env_str  : ``<EnvName>@<init_json>`` (source prefix removed, fed to from_env_str)
    Exactly matches env_server.start's ``env_str = "__".join(env_str.split("__")[1:])`` and the
    class-name extraction in server_config.get_class.
    """
    s = str(env_str)
    if s.startswith("codegym_v1@"):
        s = s[len("codegym_v1@"):]
    # s = "<source>__<EnvName>@<init_json>"
    head = s.split("@", 1)[0]                      # "<source>__<EnvName>"
    filename = head + ".py"
    parts = head.split("__")
    if len(parts) < 2:
        raise ValueError(f"cannot parse <source>__<EnvName> from env_str: {env_str!r}")
    class_name = "__".join(parts[1:])              # "<EnvName>"
    inner_env_str = "__".join(s.split("__")[1:])   # "<EnvName>@<init_json>"
    return filename, class_name, inner_env_str


def _load_env_class(envs_dir: str, filename: str, class_name: str, cache: dict[str, Any]) -> Any:
    """Load the env class by file name (cached per file name inside the worker to avoid re-running exec_module)."""
    cls = cache.get(filename)
    if cls is not None:
        return cls
    path = os.path.join(envs_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CodeGym env file does not exist: {path}")
    spec = importlib.util.spec_from_file_location(filename, path)
    module = importlib.util.module_from_spec(spec)
    source, repairs = source_compatibility().compatible_source(path)
    if repairs:
        # Keep raw dataset files intact; the preflight reports these exact compatibility repairs.
        exec(compile(source, path, "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"env module {filename} has no class {class_name}")
    cache[filename] = cls
    return cls


def source_compatibility():
    from . import source_compat
    return source_compat


def _transaction_helper():
    from . import transaction
    return transaction


# =========================================================================
# Single-env worker (plain Python; usable directly, or wrapped as an actor via ray.remote)
# =========================================================================
class CodeGymEnv:
    """Holds one CodeGym env, can be reset to different env_str values repeatedly and stepped.

    A worker serves at most one trajectory at a time (lease semantics). Under Ray actors it is a
    separate process, so even if the generated env code raises or crashes, only this one trajectory
    is affected and the pool can restart the policy LLM backbone.
    """

    def __init__(self, envs_dir: str, action_timeout_s: Optional[float] = None):
        self.envs_dir = os.path.expanduser(os.path.expandvars(envs_dir))
        self.action_timeout_s = _transaction_helper().validate_action_timeout(
            ACTION_TIMEOUT_DEFAULT_S if action_timeout_s is None else action_timeout_s
        )
        self._cls_cache: dict[str, Any] = {}
        self._env = None
        self._step_count = 0
        self._env_str: Optional[str] = None

    def reset(self, env_str: str) -> dict:
        self.close()
        filename, class_name, inner = parse_codegym_env_str(env_str)
        cls = _load_env_class(self.envs_dir, filename, class_name, self._cls_cache)
        # from_env_str validates the prefix ("<EnvName>@"); on mismatch it returns None, fall back to the constructor.
        env = cls.from_env_str(inner)
        if env is None:
            env = cls(env_str=inner)
        # Initial observation: prefer the env's own _get_obs/get_obs, otherwise use the same placeholder
        # string as env_server.
        obs = "successfully start"
        for name in ("_get_obs", "get_obs"):
            if hasattr(env, name):
                obs = getattr(env, name)()
                break
        self._env = env
        self._step_count = 0
        self._env_str = env_str
        return {
            "observation": str(obs),
            "reward": float(getattr(env, "reward", 0.0) or 0.0),
            "won": False,
            "available_actions": [],
            "done": bool(getattr(env, "finished", False)),
            "step_count": 0,
        }

    def step(self, action: str) -> dict:
        if self._env is None:
            raise RuntimeError("step called before reset")
        # Trial on a dill snapshot. Only a complete normal return commits;
        # action exceptions/timeouts preserve state and consume an attempt.
        candidate, status, observation, action_error, action_timed_out = (
            _transaction_helper().step_transaction(self._env, str(action), self.action_timeout_s)
        )
        reward = float(getattr(candidate, "reward", 0.0) or 0.0)
        done = bool(getattr(candidate, "finished", False))
        self._env = candidate
        self._step_count += 1
        return {
            "observation": str(observation),
            "reward": reward,
            # CodeGym has a terminal binary reward: after submitting Done, reward>0 means success.
            # Returned separately so callers can track success rate.
            "won": done and reward > 0,
            "available_actions": [],
            "done": done,
            "step_count": self._step_count,
            "status": bool(status),
            "action_error": action_error,
            "action_timed_out": action_timed_out,
            "action_failed": action_error is not None,
            "action_timeout": action_timed_out,
            "action_error_type": (
                "TimeoutError" if action_timed_out else action_error.split(":", 1)[0] if action_error else None
            ),
        }

    def get_ref_answer(self) -> Any:
        """Expose the env's reference answer (optional, for oracle / scoring use)."""
        if self._env is not None and hasattr(self._env, "get_ref_answer"):
            return self._env.get_ref_answer()
        return None

    def close(self) -> dict:
        if self._env is not None and hasattr(self._env, "close"):
            # Propagate failed cleanup so the pool discards unknown environment state.
            self._env.close()
        self._env = None
        self._env_str = None
        return {"closed": True}

    def health_check(self, env_str: str) -> dict:
        """Round-trip self-check reset(env_str)->step(Observe)->close; returns ok / error."""
        try:
            import json as _json
            r = self.reset(env_str)
            s = self.step(_json.dumps({"name": "Observe", "parameters": {}}))
            self.close()
            return {"ok": True, "reset_obs_len": len(r.get("observation", "")),
                    "step_obs_len": len(s.get("observation", ""))}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}
