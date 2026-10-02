# CodeGym leases one Ray worker per trajectory and loads env_str-selected classes on demand.
# Classes are cached by filename; each action runs transactionally in a disposable subprocess.
# BaseEnvPool owns worker lifecycle. Worker arguments are envs_dir and action_timeout_s;
# health checks supply a real env_str, and reset_spec is forwarded to worker.reset.
# reset/step return observation, reward, available_actions, done and step_count;
# close returns {closed: True}.

from __future__ import annotations

import asyncio
import math
import os
from typing import Any, Optional

# env_str parsing (parse_codegym_env_str), env class loading and the single-env worker
# (CodeGymEnvWorker) live in the import-light agent_system.environments.backends.codegym.worker: the Ray actor process
# imports only that (never triggering verl/__init__.py -> torch), avoiding the raylet registration
# timeout storm when N actors start concurrently. The re-export keeps historical usage working.
from agent_system.environments.backends.codegym.worker import (  # noqa: F401
    ACTION_RPC_HEADROOM_S,
    ACTION_TIMEOUT_DEFAULT_S,
    CodeGymEnvWorker,
    parse_codegym_env_str,
)
from agent_system.environments.core.pool import TrajectoryEnvPool


# Default envs dir: data/codegym/dataset/envs/codegym_v1.
# The env sources are a generated artifact (rebuilt by experiments/shared/dataset/codegym_extract_envs.py from the
# HF dataset), so they live under data/ with the other generated data rather than inside the CodeGym
# submodule -- that submodule is kept byte-identical to upstream.
# On the cluster the envs live in blob, so the path can be overridden by config or CODEGYM_ENVS_DIR.
def _default_envs_dir() -> str:
    # This file is at agent_system/environments/backends/codegym/.
    # codegym -> backends -> environments -> agent_system -> repository root.
    here = os.path.dirname(os.path.abspath(__file__))
    dyad_verl_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    return os.path.join(dyad_verl_root, "data", "codegym", "dataset", "envs", "codegym_v1")


class CodeGymEnvPool(TrajectoryEnvPool):
    """Pool of N CodeGym env workers (external envs always run inside Ray). Worker management is inherited from
    BaseEnvPool.

    Automatic capacity follows the dispatched shard; explicit capacity must cover its live trajectories.
    unlike the ALFWorld pool, a CodeGym worker preloads no fixed env -- each session loads its env on
    the fly from env_str (task).
    The health self-check needs an env_str (hence the _health_check override); get_ref_answer is
    additionally exposed.
    """

    WORKER_CLS = CodeGymEnvWorker
    LOG_NAME = "codegym_env_pool"
    FAIL_ON_WORKER_LOSS = True
    RPC_TIMEOUT_DEFAULTS = {"reset": 120.0, "step": 30.0, "close": 10.0, "get_ref_answer": 30.0, "lease": 3600.0}

    @classmethod
    def resolve_timeouts(cls, config: Optional[dict] = None) -> dict[str, float]:
        config = config or {}
        timeouts = {}
        for method, default in cls.RPC_TIMEOUT_DEFAULTS.items():
            key = f"{method}_timeout_s"
            raw = os.environ.get(f"CODEGYM_{key.upper()}") or config.get(key, default)
            value = float(raw)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"CodeGym {key} must be finite and positive")
            timeouts[method] = value
        return timeouts

    @classmethod
    def resolve_action_timeout(cls, config: Optional[dict] = None) -> float:
        config = config or {}
        raw = os.environ.get("CODEGYM_ACTION_TIMEOUT_S") or config.get("action_timeout_s", ACTION_TIMEOUT_DEFAULT_S)
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("CodeGym action_timeout_s must be finite and positive")
        return value

    def __init__(
        self,
        envs_dir: str,
        pool_size: Optional[int],
        num_cpus_per_worker: float = 0.25,
        health_env_str: Optional[str] = None,
        timeouts: Optional[dict[str, float]] = None,
        action_timeout_s: Optional[float] = None,
    ):
        super().__init__(pool_size, num_cpus_per_worker)
        self._timeouts = self.resolve_timeouts() if timeouts is None else {
            method: float(timeouts.get(method, default)) for method, default in self.RPC_TIMEOUT_DEFAULTS.items()
        }
        if any(not math.isfinite(value) or value <= 0 for value in self._timeouts.values()):
            raise ValueError("CodeGym RPC timeouts must be finite and positive")
        self.action_timeout_s = self.resolve_action_timeout() if action_timeout_s is None else float(action_timeout_s)
        if not math.isfinite(self.action_timeout_s) or self.action_timeout_s <= 0:
            raise ValueError("CodeGym action_timeout_s must be finite and positive")
        if self._timeouts["step"] <= self.action_timeout_s + ACTION_RPC_HEADROOM_S:
            raise ValueError(
                f"CodeGym step_timeout_s must exceed action_timeout_s + {ACTION_RPC_HEADROOM_S}s "
                "to allow transaction cleanup before the outer RPC watchdog"
            )
        self.envs_dir = os.path.expanduser(os.path.expandvars(envs_dir))
        # env_str used by the health self-check: defaults to the first *Env.py in envs_dir, wrapped into a
        # minimal env_str.
        self.health_env_str = health_env_str

    def _lease_timeout(self) -> float:
        return self._timeouts["lease"]

    def _rpc_timeout(self, method: str) -> Optional[float]:
        return self._timeouts.get(method)

    async def _session_call(self, worker: Any, method: str, *args, **kwargs):
        try:
            return await super()._session_call(worker, method, *args, **kwargs)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"CodeGym {method} timed out after {self._rpc_timeout(method)}s "
                f"on worker={self._worker_idx.get(id(worker), -1)}"
            ) from exc

    def _worker_init_args(self) -> tuple:
        if self._pick_health_env_str() is None:
            raise FileNotFoundError(f"CodeGym has no health-check task in envs_dir={self.envs_dir!r}")
        return (self.envs_dir, self.action_timeout_s)

    def _pick_health_env_str(self) -> Optional[str]:
        if self.health_env_str:
            return self.health_env_str
        try:
            for fn in sorted(os.listdir(self.envs_dir)):
                if fn.endswith("Env.py") and not fn.startswith("__"):
                    head = fn[:-3]  # "<source>__<EnvName>"
                    # Empty init ({}) is fine: most envs have defaults for reset(options={}), we only need it to boot.
                    # env_str format: codegym_v1@<source>__<EnvName>@<init_json>
                    return f"codegym_v1@{head}@{{}}"
        except Exception:  # noqa: BLE001
            return None
        return None

    async def _health_check(self, worker: Any) -> dict:
        hes = self._pick_health_env_str()
        if hes is None:
            raise FileNotFoundError(f"CodeGym has no health-check task in envs_dir={self.envs_dir!r}")
        return await self._call(worker, "health_check", hes)

    def _reset_log_extra(self, reset_spec: dict, result: dict) -> dict:
        return {
            "env_str": str(reset_spec.get("env_str", ""))[:120],
            "obs_len": len(result.get("observation", "")),
        }

    def _pool_init_log_extra(self) -> dict:
        return {
            "envs_dir": self.envs_dir,
            "rpc_timeouts": self._timeouts,
            "action_timeout_s": self.action_timeout_s,
            "action_execution": "trial_then_overwrite",
        }

    async def get_ref_answer(self, session_id: str) -> Any:
        """Expose the env's reference answer (for oracle/scoring, optional). CodeGym-only, not part of the
        shared interface."""
        worker = self._sessions.get(session_id)
        if worker is None:
            raise KeyError(f"unknown session_id={session_id!r}")
        try:
            return await self._session_call(worker, "get_ref_answer")
        except BaseException:
            await self.abort_session(session_id, "get_ref_answer interrupted or failed")
            raise
