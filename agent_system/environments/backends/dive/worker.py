"""Import-light synchronous Ray bridge to a DIVE task session."""
from __future__ import annotations

import json
import sys


class DiveEnvWorker:
    def __init__(self, session_config=None):
        self.config = dict(session_config or {})
        self._session = None

    def health_check(self):
        try:
            from agent_system.environments.env_package.dive.runtime import DiveToolRuntime
            runtime = DiveToolRuntime(self.config.get("repo"), self.config.get("tool_timeout_s", 60),
                                      self.config.get("sandbox_url"))
            return {"ok": True, "benchmark": "dive", "registered_tools": len(runtime.names),
                    "python_executable": sys.executable,
                    "heavy_imports": [name for name in ("torch", "verl") if name in sys.modules]}
        except Exception as exc:
            return {"ok": False, "error_type": type(exc).__name__, "benchmark": "dive"}

    @staticmethod
    def _response(value):
        if not isinstance(value, dict):
            raise TypeError("DIVE response must be a JSON object")
        json.dumps(value, allow_nan=False)
        return value

    def reset(self, **spec):
        if self._session is not None:
            raise RuntimeError("DIVE worker must close before reset")
        from agent_system.environments.env_package.dive.envs import DiveEnv
        try:
            self._session = DiveEnv(**self.config)
            return self._response(self._session.reset(**spec))
        except Exception as exc:
            raise RuntimeError(f"DIVE reset failed ({type(exc).__name__})") from None

    def step(self, action):
        if self._session is None:
            raise RuntimeError("DIVE worker has no active session")
        try:
            return self._response(self._session.step(action))
        except Exception as exc:
            raise RuntimeError(f"DIVE step failed ({type(exc).__name__})") from None

    def finalize(self, reason):
        if self._session is None:
            raise RuntimeError("DIVE worker has no active session")
        try:
            return self._response(self._session.finalize(reason))
        except Exception as exc:
            raise RuntimeError(f"DIVE finalize failed ({type(exc).__name__})") from None

    def close(self):
        session, self._session = self._session, None
        if session is not None:
            session.close()
        return {"ok": True}
