"""Import-light bridge to one isolated SWE-bench session per worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def environment_interpreter(env=None, *, executable=None):
    """Use one absolute venv path without dereferencing its Python symlink."""
    env = os.environ if env is None else env
    canonical, legacy = env.get("SWEBENCH_PYTHON_BIN"), env.get("SWE_PYTHON_BIN")
    if canonical and legacy and os.path.abspath(os.path.expanduser(canonical)) != os.path.abspath(os.path.expanduser(legacy)):
        raise ValueError("SWEBENCH_PYTHON_BIN and SWE_PYTHON_BIN disagree")
    root = Path(__file__).resolve().parents[4]
    path = Path(executable or canonical or legacy or root / ".venvs/swebench/bin/python").expanduser()
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("SWE-bench interpreter must be an absolute existing executable")
    return os.path.abspath(path)


def verify_ray_runtime(expected=None):
    """Import Ray itself before scheduling, without importing policy libraries."""
    import ray
    identity = {"ray_version": ray.__version__, "python_version": ".".join(map(str, sys.version_info[:3]))}
    if expected is not None and any(identity.get(key) != expected.get(key) for key in identity):
        raise RuntimeError(f"SWE-bench Ray/Python runtime mismatch: expected {expected}, found {identity}")
    return identity


class SwebenchEnvWorker:
    def __init__(self, session_config=None):
        self.config = dict(session_config or {})
        self._session = None

    def health_check(self):
        try:
            from agent_system.environments.backends.swebench.session import SwebenchSession
            runtime = verify_ray_runtime(self.config.get("expected_ray_runtime"))
            heavy = [name for name in ("torch", "verl") if name in sys.modules]
            if heavy:
                raise RuntimeError("SWE-bench worker imported policy-only libraries")
            return {"ok": True, "benchmark": "swebench_verified", "python_executable": sys.executable,
                    "ray_runtime": runtime, "heavy_imports": heavy}
        except Exception as exc:
            return {"ok": False, "error_type": type(exc).__name__, "benchmark": "swebench_verified"}

    @staticmethod
    def _response(value):
        if not isinstance(value, dict):
            raise TypeError("SWE-bench session response must be a JSON object")
        json.dumps(value, allow_nan=False)
        return value

    def reset(self, **spec):
        if self._session is not None:
            raise RuntimeError("Close the prior SWE-bench session before resetting")
        from agent_system.environments.backends.swebench.session import SwebenchSession
        self._session = SwebenchSession(**self.config)
        return self._response(self._session.reset(**spec))

    def step(self, action):
        if self._session is None:
            raise RuntimeError("No active SWE-bench session")
        return self._response(self._session.step(action))

    def finalize(self, reason):
        if self._session is None:
            raise RuntimeError("No active SWE-bench session")
        return self._response(self._session.finalize(reason))

    def close(self):
        session, self._session = self._session, None
        if session is not None:
            session.close()
        return {"ok": True}
