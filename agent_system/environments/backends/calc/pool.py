# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""GSM8K calculator env pool (external envs always run inside Ray), inheriting the shared BaseEnvPool.

The calculator is moved "into Ray": each trajectory leases one env worker (Ray actor = separate
process) that holds one CalcSession (one question = one session). No HTTP layer, no single-point server.
1:1 concurrency: pool_size should equal train_batch x rollout.n so every trajectory stays bound to a
dedicated env actor for its whole life (state kept isolated).

Worker management (start / create_session / step / close / abort / shutdown) is entirely inherited
from BaseEnvPool; this class only supplies the env-specific hooks: worker class = CalcEnvWorker,
constructor args = (max_turns,), extra diagnostic field = ground_truth.

Contract (reset_spec is forwarded to worker.reset(**reset_spec)):
  reset(ground_truth, max_turns) -> {observation, reward, available_actions, done, step_count}
  step(action)                   -> {observation, reward, available_actions, done, step_count, action_name}
  close()                        -> {closed: True}
"""

from __future__ import annotations

# The single-env worker used by the policy LLM backbone comes from the import-light package that sits next to verl:
# the policy LLM backbone process imports pure logic only and never triggers verl/__init__.py -> torch (~2.8s per
# process), which avoids the raylet registration timeout storm when N actors start concurrently
# (worker_pool.cc:590 ... not registered within the timeout). See agent_system/environments/README.md.
from agent_system.environments.backends.calc.worker import CalcEnvWorker

from agent_system.environments.core.pool import BaseEnvPool

# Legacy marker, kept for external imports and diagnostic comparison (backward compatibility).
CALC_ENV_POOL_IMPLEMENTATION = "2026-07-14.base-env-pool-v1"


class CalcEnvPool(BaseEnvPool):
    """Pool of N calculator env workers (always Ray actors). Worker management is inherited from BaseEnvPool.

    pool_size=None means auto: floor(available Ray CPU / num_cpus_per_worker);
    an explicit pool_size must fit the available Ray CPU budget or startup fails.
    """

    WORKER_CLS = CalcEnvWorker
    LOG_NAME = "calc_env_pool"

    def __init__(
        self,
        pool_size,
        num_cpus_per_worker: float = 0.1,
        max_turns: int = 20,
    ):
        super().__init__(pool_size, num_cpus_per_worker)
        self.max_turns = int(max_turns)

    def _worker_init_args(self) -> tuple:
        return (self.max_turns,)

    def _reset_log_extra(self, reset_spec: dict, result: dict) -> dict:
        return {"ground_truth": reset_spec.get("ground_truth")}
