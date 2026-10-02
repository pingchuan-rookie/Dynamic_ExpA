"""CPU actor pool for pinned DIVE tools with isolated dependencies."""
from __future__ import annotations

import math
import os
from pathlib import Path

from agent_system.environments.core.pool import BaseEnvPool
from agent_system.environments.backends.dive.worker import DiveEnvWorker


class DiveEnvPool(BaseEnvPool):
    WORKER_CLS = DiveEnvWorker
    LOG_NAME = "dive_env_pool"
    FAIL_ON_WORKER_LOSS = True
    STRUCTURED_ACTIONS = True

    def __init__(self, config):
        self.config = dict(config)
        size = int(self.config.get("pool_size", os.environ.get("DIVE_ENV_POOL_SIZE", 4)))
        if size < 1:
            raise ValueError("DIVE pool_size must be positive")
        super().__init__(size, float(self.config.get("num_cpus_per_worker", 0.25)))
        self._timeouts = {}
        for method, default in (("reset", 120), ("step", 1800), ("finalize", 150), ("close", 15)):
            value = float(self.config.get(f"{method}_timeout_s", os.environ.get(f"DIVE_{method.upper()}_TIMEOUT_S", default)))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"DIVE {method}_timeout_s must be finite and positive")
            self._timeouts[method] = value

    def _worker_init_args(self):
        config = dict(self.config.get("session_config", {}))
        for key, env in (("repo", "DIVE_REPO"), ("sandbox_url", "SANDBOX_FUSION_URL")):
            if key not in config and os.environ.get(env):
                config[key] = os.environ[env]
        return (config,)

    def _rpc_timeout(self, method):
        return self._timeouts.get(method)

    def _worker_options(self):
        root = Path(__file__).resolve().parents[4]
        executable = Path(self.config.get("python_executable") or os.environ.get("DIVE_PYTHON_BIN") or root / ".venvs/dive/bin/python").expanduser()
        if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("DIVE actor requires an absolute executable DIVE_PYTHON_BIN")
        env_vars = {"PYTHONPATH": os.pathsep.join([str(root)]),
                    "PYTHONDONTWRITEBYTECODE": "1"}
        from agent_system.environments.env_package.dive.model_api.providers.trapi import TRAPI_ENV_VARS
        names = ("DIVE_REPO", "DIVE_JUDGE_MODEL", "DIVE_JUDGE_BASE_URL", "DIVE_JUDGE_PROVIDER", "DIVE_JUDGE_API_KEY",
                 "SANDBOX_FUSION_URL", "TUSHARE_TOKEN", "TUSHARE_API_KEY", "SERPER_API_KEY", "JINA_API_KEY",
                 "BROWSE_LLM_API_KEY", "BROWSE_LLM_BASE_URL", "BROWSE_LLM_MODEL", "BROWSE_LLM_PROVIDER",
                 "NCBI_API_KEY", "NCBI_EMAIL", "SEMANTIC_SCHOLAR_API_KEY", *TRAPI_ENV_VARS)
        judge_key = self.config.get("session_config", {}).get("judge_config", {}).get("api_key_env", "DIVE_JUDGE_API_KEY")
        for name in (*names, judge_key, *self.config.get("forward_env_vars", [])):
            if name in os.environ:
                env_vars[name] = os.environ[name]
        return {"runtime_env": {"py_executable": str(executable), "env_vars": env_vars}}

    def _reset_log_extra(self, spec, response):
        return {"benchmark": "dive", "schema_hash": response.get("schema_hash")}
