"""Ray pool for isolated official tau sessions, without an environment server."""
from __future__ import annotations

import math
import os
from pathlib import Path

from agent_system.environments.core.pool import BaseEnvPool
from agent_system.environments.backends.tau.worker import TauEnvWorker


class TauEnvPool(BaseEnvPool):
    WORKER_CLS = TauEnvWorker
    LOG_NAME = "tau_env_pool"
    FAIL_ON_WORKER_LOSS = True
    STRUCTURED_ACTIONS = True

    def __init__(self, config: dict):
        self.config = dict(config)
        self.benchmark = self.config.get("benchmark", self.config.get("env_type"))
        if self.benchmark != "t2bench":
            raise ValueError("Tau pool requires benchmark=t2bench")
        size = int(self.config.get("pool_size", 1))
        if size <= 0:
            raise ValueError("Tau pool_size must be positive")
        super().__init__(size, float(self.config.get("num_cpus_per_worker", 0.25)))
        self._timeouts = {}
        for method, default in (("reset", 180), ("step", 300), ("finalize", 300), ("close", 30)):
            timeout = float(self.config.get(f"{method}_timeout_s", default))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError(f"{method}_timeout_s must be finite and positive")
            self._timeouts[method] = timeout

    def _worker_init_args(self):
        return (self.benchmark, dict(self.config.get("session_config", {})))

    def _rpc_timeout(self, method):
        return self._timeouts.get(method)

    def _worker_options(self):
        root = Path(__file__).resolve().parents[4]
        prefix = self.benchmark.upper()
        executable = self.config.get("python_executable") or os.environ.get(f"{prefix}_PYTHON_BIN")
        if executable is None:
            executable = str(root / ".venvs" / self.benchmark / "bin/python")
        executable = Path(executable).expanduser()
        if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("Tau actor python_executable must be an existing absolute executable")
        paths = [str(root)]
        if os.environ.get("PYTHONPATH"):
            paths.append(os.environ["PYTHONPATH"])
        env_vars = {"PYTHONPATH": os.pathsep.join(paths), "PYTHONDONTWRITEBYTECODE": "1"}
        # Pass names, never embed secrets in logged pool identities or task records.
        from agent_system.environments.env_package.t2bench.trapi import TRAPI_ENV_VARS
        for name in dict.fromkeys([*self.config.get("forward_env_vars", []), *TRAPI_ENV_VARS]):
            if name in os.environ:
                env_vars[name] = os.environ[name]
        return {"runtime_env": {"py_executable": str(executable), "env_vars": env_vars}}

    def _reset_log_extra(self, reset_spec, result):
        return {"benchmark": self.benchmark, "schema_hash": result.get("schema_hash"),
                "done": bool(result.get("done"))}
