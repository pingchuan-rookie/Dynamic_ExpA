# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Single-env worker for the Ray external env (import-light).

**Imports this package's pure logic only, never verl** -- so a Ray actor process does not execute
`verl/__init__.py` on startup (->torch, ~2.8s per process), and N actors starting concurrently never
trigger the raylet `worker_pool.cc:590 ... not registered within the timeout` storm.

`CalcEnvPool` in `agent_system/environments/backends/calc/pool.py` wraps this class as a policy LLM backbone via
`ray.remote(...)(CalcEnvWorker)`; each trajectory leases a worker, a worker holds one `CalcSession`
and can be reset to different questions repeatedly.
"""

from __future__ import annotations

from typing import Optional

from agent_system.environments.core.worker import BaseEnvWorker
from agent_system.environments.backends.calc.session import CalcSession


class CalcEnvWorker(BaseEnvWorker):
    """Holds one calculator session, can be reset to different questions (ground_truth) and stepped."""

    def __init__(self, max_turns: int = 20):
        self._session = CalcSession(max_turns=max_turns)

    def reset(self, ground_truth: Optional[str] = None, max_turns: Optional[int] = None) -> dict:
        return self._session.reset(ground_truth=ground_truth, max_turns=max_turns)

    def step(self, action: str) -> dict:
        return self._session.step(str(action))

    def close(self) -> dict:
        return self._session.close()

    def health_check(self) -> dict:
        return self._session.health_check()
