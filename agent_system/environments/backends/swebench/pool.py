"""Bounded isolated CPU workers owning offline SWE-bench containers."""
from __future__ import annotations

import math
import os
from pathlib import Path

from agent_system.environments.core.pool import BaseEnvPool
from agent_system.environments.backends.swebench.worker import SwebenchEnvWorker, environment_interpreter


class SwebenchEnvPool(BaseEnvPool):
    WORKER_CLS = SwebenchEnvWorker
    LOG_NAME = "swebench_env_pool"
    FAIL_ON_WORKER_LOSS = True
    STRUCTURED_ACTIONS = True

    def __init__(self, config):
        self.config = dict(config)
        size = int(self.config.get("pool_size", 1))
        if size < 1:
            raise ValueError("SWE-bench pool size must be positive")
        super().__init__(size, float(self.config.get("num_cpus_per_worker", 1)))
        self._timeouts = {}
        for method, default in (("reset", 180), ("step", 300), ("finalize", 1800), ("close", 30)):
            value = float(self.config.get(f"{method}_timeout_s", default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{method}_timeout_s must be finite and positive")
            self._timeouts[method] = value

    def _worker_init_args(self):
        return (dict(self.config.get("session_config", {})),)

    def _rpc_timeout(self, method):
        return self._timeouts.get(method)

    def _worker_options(self):
        root = Path(__file__).resolve().parents[4]
        executable = environment_interpreter(executable=self.config.get("python_executable"))
        from agent_system.environments.env_package.swebench.offline import OFFLINE_ENV
        env_vars = {**OFFLINE_ENV, "PYTHONPATH": os.pathsep.join([str(root)]),
                    "SWEBENCH_PYTHON_BIN": executable, "SWE_PYTHON_BIN": executable}
        for name in ("DOCKER_HOST", "SWEBENCH_HARNESS_PATH"):
            if name in os.environ:
                env_vars[name] = os.environ[name]
        mini = self.config.get("session_config", {}).get("mini_swe_agent") or {}
        key_name = mini.get("api_key_env")
        if key_name:
            if not os.environ.get(key_name):
                raise ValueError("mini-swe-agent API key environment variable is unset")
            env_vars[key_name] = os.environ[key_name]
        return {"runtime_env": {"py_executable": str(executable), "env_vars": env_vars}}

    def _reset_log_extra(self, spec, response):
        return {"benchmark": "swebench_verified", "instance_id": spec.get("instance_id"),
                "schema_hash": response.get("schema_hash")}
