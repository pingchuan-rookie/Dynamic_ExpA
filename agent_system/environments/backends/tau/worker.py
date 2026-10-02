"""Import-light Ray actor bridge to isolated official tau sessions.

Only standard-library and import-light worker code may be imported at module scope.
"""
from __future__ import annotations

from importlib import import_module
import json
import sys

from agent_system.environments.core.worker import BaseEnvWorker


class TauEnvWorker(BaseEnvWorker):
    def __init__(self, benchmark, session_config=None):
        if benchmark != "t2bench":
            raise ValueError("Unsupported tau benchmark")
        self.benchmark = benchmark
        self.session_config = dict(session_config or {})
        self._session = None

    def _session_class(self):
        return import_module(f"agent_system.environments.env_package.{self.benchmark}.envs").T2BenchEnv

    def health_check(self):
        try:
            self._session_class()
            return {"ok": True, "benchmark": self.benchmark, "python_executable": sys.executable,
                    "heavy_imports": [name for name in ("torch", "verl") if name in sys.modules]}
        except Exception as exc:
            # Dependency errors may include credentials from provider configuration.
            return {"ok": False, "error_type": type(exc).__name__, "benchmark": self.benchmark}

    @staticmethod
    def _response(value):
        if not isinstance(value, dict):
            raise TypeError("Tau session response must be a JSON object")
        # Validate a strict wire protocol, preserving all official nested schemas.
        json.dumps(value, allow_nan=False)
        return value

    def reset(self, **reset_spec):
        if self._session is not None:
            raise RuntimeError("Tau worker must close the previous session before reset")
        try:
            self._session = self._session_class()(**self.session_config)
            return self._response(self._session.reset(**reset_spec))
        except Exception as exc:
            raise RuntimeError(f"Tau reset failed ({type(exc).__name__})") from None

    def step(self, action):
        if self._session is None:
            raise RuntimeError("Tau worker has no active session")
        if not isinstance(action, dict):
            raise TypeError("Tau actions must remain JSON objects")
        try:
            return self._response(self._session.step(action))
        except Exception as exc:
            raise RuntimeError(f"Tau step failed ({type(exc).__name__})") from None

    def finalize(self, reason):
        if self._session is None:
            raise RuntimeError("Tau worker has no active session")
        try:
            return self._response(self._session.finalize(reason))
        except Exception as exc:
            raise RuntimeError(f"Tau finalize failed ({type(exc).__name__})") from None

    def close(self):
        session, self._session = self._session, None
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                raise RuntimeError(f"Tau close failed ({type(exc).__name__})") from None
        return {"ok": True}
