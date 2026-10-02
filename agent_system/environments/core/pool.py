# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Shared base class for external env pools (Calc / ALFWorld / CodeGym reuse one worker-management implementation).

Mirroring the verl-agent / GiGPO env pools, external envs are moved "into Ray": each trajectory
leases one Ray actor (separate process, isolated state, true parallelism), with no HTTP layer and
no single-point server. The three env pools differ only in their "env-specific hooks" (worker class /
constructor args / health args / per-task reset args / extra diagnostic fields); the
worker-management layer (start / create_session / step_session / close_session / abort_session /
shutdown + lease queue + stable indices + diagnostic logging) is identical for all three.

Shared contract:
  A worker actor must implement reset(**reset_spec) / step(action) / close() / health_check(*health_args);
  reset/step return a dict shaped {observation, reward, available_actions, done, step_count, ...}.

Subclasses only override the following (everything else is inherited):
  - WORKER_CLS: import-light single-env worker class (wrapped into an actor by ray.remote)
  - LOG_NAME  : diagnostic event name ("<env>_env_pool")
  - _worker_init_args()          -> tuple: actor constructor args
  - _health_check(worker)        -> dict : health self-check (CodeGym passes env_str)
  - _reset_log_extra(spec, res)  -> dict : extra env_reset diagnostic fields (optional)
  - _pool_init_log_extra()       -> dict : extra pool_init diagnostic fields (optional)
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from typing import Any, Optional

try:
    from agent_system.utils.diagnostics import full_dump as _diag_full
    from agent_system.utils.diagnostics import log_event
except Exception:  # noqa: BLE001 -- degrade to a no-op when diagnostics are unavailable; the env pool itself is unaffected.
    def log_event(*_args, **_kwargs):  # type: ignore
        return None

    def _diag_full() -> bool:  # type: ignore
        return False


# Cap on the action / observation strings mirrored into env_step_bind. The offline turns view
# compares this action_in against the agent side's action_sent verbatim, so a cap that cuts either
# one would read as "the action string was rewritten in transit" -- the exact bug the check exists
# to catch. `full` therefore keeps enough room for a real ALFWorld observation.
_BIND_TEXT_CAP = 512
_BIND_TEXT_CAP_FULL = 4096


class BaseEnvPool:
    """Pool of N Ray-actor env workers, leased/returned per session. Shared by all three external env pools.

    pool_size should be >= the max concurrent trajectories of one agent-loop worker
    (= TRAIN_BATCH_SIZE x ROLLOUT_N) to get true 1:1 with no queueing; surplus sessions block and
    queue inside create_session, reusing actors.
    pool_size=None means auto: floor(available Ray CPU / num_cpus_per_worker) (used by calc;
    alfworld/codegym pass a fixed int).
    """

    # ---- specified by subclasses ----
    WORKER_CLS: Any = None
    LOG_NAME: str = "env_pool"
    # Stateful conversational backends cannot reuse a worker after interrupted RPCs.
    FAIL_ON_WORKER_LOSS: bool = True
    STRUCTURED_ACTIONS: bool = False

    def __init__(self, pool_size: Optional[int], num_cpus_per_worker: float):
        self.pool_size = None if pool_size is None else int(pool_size)
        self.num_cpus_per_worker = float(num_cpus_per_worker)
        if self.pool_size is not None and self.pool_size <= 0:
            raise ValueError("pool_size must be positive or None")
        if not math.isfinite(self.num_cpus_per_worker) or self.num_cpus_per_worker <= 0:
            raise ValueError(f"num_cpus_per_worker must be finite and positive, got {self.num_cpus_per_worker}")

        self._workers: list[Any] = []
        self._free: Optional[asyncio.Queue] = None
        self._sessions: dict[str, Any] = {}            # session_id -> worker actor handle
        self._worker_idx: dict[int, int] = {}          # id(worker) -> stable index 0..pool_size-1
        self._session_worker: dict[str, int] = {}       # session_id -> worker index (proves the 1:1 traj<->env binding)
        self._peak_active = 0                           # peak number of concurrently leased workers
        self._started = False
        self._start_lock: Optional[asyncio.Lock] = None
        self._startup_error: Optional[str] = None
        self._capacity_error: Optional[str] = None
        self._lease_waiters = 0
        self._creating: dict[str, asyncio.Task] = {}

    # =====================================================================
    # Env-specific hooks (overridden by subclasses)
    # =====================================================================
    def _worker_init_args(self) -> tuple:
        """Positional args for constructing one worker actor (RemoteWorker.remote(*args))."""
        return ()

    def _worker_options(self) -> dict:
        """Optional Ray actor options, for example an isolated Python runtime."""
        return {}

    def _rpc_timeout(self, method: str) -> Optional[float]:
        """Optional per-operation watchdog; legacy environments remain unlimited."""
        return None

    def _lease_timeout(self) -> float:
        value = float(getattr(self, "config", {}).get("lease_timeout_s", 3600))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("lease_timeout_s must be finite and positive")
        return value

    async def _session_call(self, worker: Any, method: str, *args, **kwargs):
        timeout = self._rpc_timeout(method)
        call = self._call(worker, method, *args, **kwargs)
        if timeout is None:
            return await call
        try:
            return await asyncio.wait_for(call, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"{self.LOG_NAME} {method} exceeded {timeout}s on "
                f"worker={self._worker_idx.get(id(worker), -1)}"
            ) from exc

    def _lose_worker(self, worker: Any, reason: str) -> None:
        import ray
        try:
            ray.kill(worker, no_restart=True)
        except Exception:
            # The actor is already unusable; preserve the triggering RPC exception.
            pass
        finally:
            if self.FAIL_ON_WORKER_LOSS:
                self._capacity_error = f"{self.LOG_NAME} lost an actor ({reason}); pool cannot accept new sessions"
                if self._free is not None:
                    # Wake every queued lease. Active healthy sessions may finish normally.
                    for _ in range(self._lease_waiters):
                        self._free.put_nowait(None)

    async def _health_check(self, worker: Any) -> dict:
        """
        Run one health self-check on a worker; returns a dict containing 'ok'. Defaults to the no-arg health_check.
        """
        return await self._call(worker, "health_check")

    def _reset_log_extra(self, reset_spec: dict, result: dict) -> dict:
        """Env-specific extra fields for the env_reset diagnostic (e.g. ground_truth / game / env_str)."""
        return {}

    def _pool_init_log_extra(self) -> dict:
        """Env-specific extra fields for the pool_init diagnostic (e.g. num_games / envs_dir)."""
        return {}

    # =====================================================================
    # Shared worker management
    # =====================================================================
    async def _call(self, worker: Any, method: str, *args, **kwargs) -> Any:
        # External envs always run inside Ray: every call is one async remote to that worker actor (a separate process).
        return await getattr(worker, method).remote(*args, **kwargs)

    def _startup_batch_size(self) -> int:
        raw = (
            os.environ.get("ENV_POOL_STARTUP_BATCH_SIZE")
            or os.environ.get("CALC_ENV_STARTUP_BATCH_SIZE")  # legacy variable, kept for backward compatibility
            or "8"
        )
        return max(1, min(int(self.pool_size), int(raw)))

    async def start(self, minimum_size: Optional[int] = None) -> None:
        """
        Create the workers, materializing them in batches with a per-actor health check (limits raylet registration
        storms). Idempotent.
        """
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self._startup_error is not None:
                raise RuntimeError(self._startup_error)
            if self._capacity_error is not None:
                raise RuntimeError(self._capacity_error)
            if minimum_size is not None and minimum_size < 1:
                raise ValueError("minimum_size must be positive")
            if self._started and (minimum_size is None or minimum_size <= len(self._workers)):
                return
            if minimum_size is not None:
                self.pool_size = max(self.pool_size or 0, minimum_size)
            previous_count = len(self._workers)
            t0 = time.perf_counter()
            err = None
            startup_cause = None
            try:
                import ray
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                available_cpus = float(ray.available_resources().get("CPU", 0.0))
                # Existing actors already reserved their CPUs. Explicit capacity is a
                # contract: do not silently trade rollout parallelism for queueing.
                cpu_cap = previous_count + int((available_cpus + 1e-8) / self.num_cpus_per_worker)
                if cpu_cap <= 0:
                    raise RuntimeError(
                        f"insufficient available Ray CPU for {self.LOG_NAME} actor: "
                        f"available={available_cpus}, num_cpus_per_worker={self.num_cpus_per_worker}"
                    )
                if self.pool_size is None:
                    # auto: just use the actual available CPU capacity.
                    self.pool_size = cpu_cap
                    print(
                        f"[{type(self).__name__}] auto pool_size=floor({available_cpus} available CPU / "
                        f"{self.num_cpus_per_worker} CPU per actor)={self.pool_size}",
                        flush=True,
                    )
                elif self.pool_size > cpu_cap:
                    raise RuntimeError(
                        f"{self.LOG_NAME} requires {self.pool_size} actors, already has {previous_count}; "
                        f"needs {(self.pool_size - previous_count) * self.num_cpus_per_worker:g} additional "
                        f"Ray CPUs but only {available_cpus:g} are available "
                        f"(num_cpus_per_worker={self.num_cpus_per_worker}). "
                        "Capacity was not reduced. Increase resources or explicitly configure a smaller workload/pool."
                    )
                else:
                    print(
                        f"[{type(self).__name__}] pool_size={self.pool_size} "
                        f"(≤ cpu_cap={cpu_cap}, available_cpu={available_cpus})",
                        flush=True,
                    )
                RemoteWorker = ray.remote(num_cpus=self.num_cpus_per_worker)(self.WORKER_CLS)
                worker_options = self._worker_options()
                if worker_options:
                    RemoteWorker = RemoteWorker.options(**worker_options)
                # Do not submit every actor at once and probe only the first: inside blob/containers Python
                # worker registration is slow, so a not-yet-registered handle would be leased out early and
                # eventually raise ActorUnschedulableError at reset time.
                startup_batch_size = self._startup_batch_size()
                print(
                    f"[{type(self).__name__}] pool_size={self.pool_size} "
                    f"startup_batch_size={startup_batch_size}",
                    flush=True,
                )
                init_args = self._worker_init_args()
                startup_timeout = float(os.environ.get("DYAD_ENV_POOL_STARTUP_TIMEOUT_S", "120"))
                if not math.isfinite(startup_timeout) or startup_timeout <= 0:
                    raise ValueError("DYAD_ENV_POOL_STARTUP_TIMEOUT_S must be finite and positive")
                health_results: list[dict] = []
                for offset in range(previous_count, self.pool_size, startup_batch_size):
                    batch_size = min(startup_batch_size, self.pool_size - offset)
                    batch = [RemoteWorker.remote(*init_args) for _ in range(batch_size)]
                    self._workers.extend(batch)
                    # Only create the next batch after the whole current batch registered and
                    # passed health, to limit the worker startup storm.
                    try:
                        batch_health = await asyncio.wait_for(
                            asyncio.gather(*(self._health_check(w) for w in batch)),
                            timeout=startup_timeout,
                        )
                    except asyncio.TimeoutError as exc:
                        raise RuntimeError(
                            f"{self.LOG_NAME} startup timed out after {startup_timeout}s: "
                            f"ready={previous_count + len(health_results)}/{self.pool_size}, "
                            f"submitted={len(self._workers)}, "
                            f"num_cpus_per_worker={self.num_cpus_per_worker}, "
                            f"available_resources={ray.available_resources()}. "
                            "Pool sizes are per agent worker; check aggregate CPU demand "
                            "and Ray actor logs."
                        ) from exc
                    if not all(bool(item.get("ok")) for item in batch_health):
                        raise RuntimeError(f"{self.LOG_NAME} worker health check failed: {batch_health!r}")
                    health_results.extend(batch_health)
                    print(
                        f"[{type(self).__name__}] materialized {len(self._workers)}/{self.pool_size} actors "
                        f"(batch {offset // startup_batch_size + 1}, elapsed={time.perf_counter() - t0:.1f}s)",
                        flush=True,
                    )
                if self._capacity_error is not None:
                    raise RuntimeError(self._capacity_error)
                # Give every worker actor a stable index, used to prove "one trajectory <-> one dedicated env actor".
                self._worker_idx = {id(w): i for i, w in enumerate(self._workers)}
                if self._free is None:
                    self._free = asyncio.Queue()
                for w in self._workers[previous_count:]:
                    self._free.put_nowait(w)
                print(
                    f"[{type(self).__name__}] ✅ all {self.pool_size} actors registered + health passed, "
                    f"took {time.perf_counter() - t0:.1f}s; ready for isolated session leases",
                    flush=True,
                )
                health = {
                    "ok": True,
                    "checked_workers": previous_count + len(health_results),
                    "startup_batch_size": startup_batch_size,
                }
            except BaseException as exc:
                startup_cause = exc
                err = type(exc).__name__ if self.FAIL_ON_WORKER_LOSS else repr(exc)
                health = {"ok": False, "error": err}
                import ray
                for worker in self._workers:
                    try:
                        ray.kill(worker, no_restart=True)
                    except Exception:  # noqa: BLE001
                        pass
                self._workers = []
                self._capacity_error = f"{self.LOG_NAME} startup or capacity expansion failed"
                if self._free is not None:
                    for _ in range(self._lease_waiters):
                        self._free.put_nowait(None)
                if not isinstance(exc, Exception):
                    self._startup_error = f"{type(self).__name__} startup interrupted"
                    raise
            self._started = err is None and bool(health.get("ok"))
            log_event(
                self.LOG_NAME,
                "pool_init",
                backend="ray",
                pool_size=self.pool_size,
                duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                ok=self._started,
                health=health,
                error=err,
                **self._pool_init_log_extra(),
            )
            if not self._started:
                # All queued sessions must see the same failure, rather than each
                # retrying an unschedulable pool for another timeout interval.
                self._startup_error = f"{type(self).__name__} failed to start: health={health} error={err}"
                raise RuntimeError(self._startup_error) from startup_cause

    async def create_session(self, session_id: str, reset_spec: Optional[dict] = None) -> dict:
        if session_id in self._creating or session_id in self._sessions:
            raise ValueError(f"duplicate active session_id={session_id!r}")
        self._creating[session_id] = asyncio.current_task()
        try:
            return await self._create_session(session_id, reset_spec)
        finally:
            self._creating.pop(session_id, None)

    async def _create_session(self, session_id: str, reset_spec: Optional[dict] = None) -> dict:
        """Lease a free worker, reset it with reset_spec (env-specific task args), and bind it to session_id.

        Diagnostics record the worker index / queue wait / active count, proving "N concurrent
        trajectories <-> N dedicated env actors":
          - waited_ms ~= 0 and pool_size >= concurrency -> every trajectory gets a dedicated actor.
          - waited_ms > 0 -> pool_size is too small; surplus trajectories queue for a free actor.
        """
        lease_timeout = self._lease_timeout()
        if not self._started:
            await self.start()
        assert self._free is not None
        if session_id in self._sessions:
            raise ValueError(f"duplicate active session_id={session_id!r}")
        reset_spec = dict(reset_spec or {})
        _wait0 = time.perf_counter()
        if self._capacity_error is not None:
            raise RuntimeError(self._capacity_error)
        self._lease_waiters += 1
        try:
            worker = await asyncio.wait_for(self._free.get(), timeout=lease_timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"{self.LOG_NAME} lease for {session_id} exceeded {lease_timeout}s; "
                f"active={len(self._sessions)}, capacity={len(self._workers)}, waiters={self._lease_waiters}"
            ) from exc
        finally:
            self._lease_waiters -= 1
        if self._capacity_error is not None:
            if worker is not None and self._free is not None:
                self._free.put_nowait(worker)
            raise RuntimeError(self._capacity_error)
        if session_id in self._sessions:
            self._free.put_nowait(worker)
            raise ValueError(f"duplicate active session_id={session_id!r}")
        waited_ms = round((time.perf_counter() - _wait0) * 1000, 3)
        widx = self._worker_idx.get(id(worker), -1)
        self._sessions[session_id] = worker
        self._session_worker[session_id] = widx
        active = len(self._sessions)
        if active > self._peak_active:
            self._peak_active = active
        t0 = time.perf_counter()
        try:
            result = await self._session_call(worker, "reset", **reset_spec)
        except BaseException:
            self._sessions.pop(session_id, None)
            self._session_worker.pop(session_id, None)
            # An actor whose reset failed is in an unknown state; returning it to the pool would poison
            # the next trajectory. The caller fail-fasts the job.
            self._lose_worker(worker, "reset interrupted or failed")
            raise
        log_event(
            self.LOG_NAME,
            "env_reset",
            session_id=session_id,
            worker_idx=widx,                # index of the env actor bound to this trajectory (0..pool_size-1)
            waited_ms=waited_ms,            # time spent queueing for a free actor; 0 = obtained immediately (1:1)
            active_sessions=active,         # number of envs simultaneously active at logging time
            peak_active=self._peak_active,
            pool_size=self.pool_size,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            **self._reset_log_extra(reset_spec, result),
        )
        result.setdefault("worker_idx", widx)
        return result

    async def step_session(self, session_id: str, action: Any) -> dict:
        worker = self._sessions.get(session_id)
        if worker is None:
            raise KeyError(f"unknown session_id={session_id!r}")
        widx = self._session_worker.get(session_id, -1)
        t0 = time.perf_counter()
        try:
            result = await self._session_call(worker, "step", action if self.STRUCTURED_ACTIONS else str(action))
        except BaseException:
            if self.FAIL_ON_WORKER_LOSS:
                await self.abort_session(session_id, "step interrupted or failed")
            raise
        # Log per step which env actor received which action and returned which observation -- proves the
        # traj stays bound to one actor (state is maintained there).
        # End-to-end RPC latency includes scheduling and transport, not only env execution.
        _cap = _BIND_TEXT_CAP_FULL if _diag_full() else _BIND_TEXT_CAP
        log_event(
            self.LOG_NAME,
            "env_step_bind",
            session_id=session_id,
            worker_idx=widx,
            action_in=str(action)[:_cap],
            observation_out=str(result.get("observation", ""))[:_cap],
            reward=result.get("reward"),
            done=result.get("done"),
            server_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
        result.setdefault("worker_idx", widx)
        return result

    async def finalize_session(self, session_id: str, reason: str) -> dict:
        worker = self._sessions.get(session_id)
        if worker is None:
            raise KeyError(f"unknown session_id={session_id!r}")
        try:
            return await self._session_call(worker, "finalize", reason)
        except BaseException:
            if self.FAIL_ON_WORKER_LOSS:
                await self.abort_session(session_id, "finalize interrupted or failed")
            raise

    async def close_session(self, session_id: str) -> None:
        worker = self._sessions.pop(session_id, None)
        widx = self._session_worker.pop(session_id, -1)
        if worker is None:
            return
        t0 = time.perf_counter()
        ok = False
        error = None
        try:
            await self._session_call(worker, "close")
            ok = True
        except BaseException as exc:
            error = type(exc).__name__
            self._lose_worker(worker, "close interrupted or failed")
            raise
        finally:
            log_event(
                self.LOG_NAME,
                "env_close",
                session_id=session_id,
                worker_idx=widx,
                ok=ok,
                error=error,
                active_sessions=len(self._sessions),
                duration_ms=round((time.perf_counter() - t0) * 1000, 3),
            )
            # Only a worker that closed successfully may serve the next trajectory.
            if ok:
                assert self._free is not None
                self._free.put_nowait(worker)

    async def abort_session(self, session_id: str, reason: str) -> None:
        """Cancel queued creation and discard an actor whose environment state is unknown."""
        creating = self._creating.get(session_id)
        if creating is not None and creating is not asyncio.current_task():
            creating.cancel()
            await asyncio.gather(creating, return_exceptions=True)
        worker = self._sessions.pop(session_id, None)
        widx = self._session_worker.pop(session_id, -1)
        if worker is None:
            return
        self._lose_worker(worker, reason)
        log_event(
            self.LOG_NAME,
            "env_abort",
            session_id=session_id,
            worker_idx=widx,
            reason=reason,
            active_sessions=len(self._sessions),
        )

    async def shutdown(self) -> None:
        import ray
        if self.FAIL_ON_WORKER_LOSS:
            self._capacity_error = f"{self.LOG_NAME} has shut down"
            self._startup_error = self._capacity_error
            if self._free is not None:
                for _ in range(self._lease_waiters):
                    self._free.put_nowait(None)
        creating = [task for task in self._creating.values() if task is not asyncio.current_task()]
        for task in creating:
            task.cancel()
        if creating:
            await asyncio.gather(*creating, return_exceptions=True)
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            for w in self._workers:
                try:
                    ray.kill(w)
                except Exception:  # noqa: BLE001
                    pass
            self._workers = []
            self._sessions = {}
            self._session_worker = {}
            self._worker_idx = {}
            self._free = None
            self._started = False


class TrajectoryEnvPool(BaseEnvPool):
    """One actor per live rollout, sized from the shard dispatched to its owner."""

    def __init__(self, pool_size: Optional[int], num_cpus_per_worker: float):
        super().__init__(pool_size, num_cpus_per_worker)
        self._automatic_capacity = pool_size is None

    async def ensure_capacity(self, sessions: int) -> None:
        """Reserve full local rollout concurrency before launching any trajectories."""
        if not self._automatic_capacity and sessions > self.pool_size:
            raise ValueError(
                f"{self.LOG_NAME} assigned {sessions} concurrent sessions but explicit pool_size={self.pool_size}; "
                "increase it or remove the override to size from the dispatched batch"
            )
        await self.start(minimum_size=sessions)

    async def start(self, minimum_size: int | None = None) -> None:
        # Standalone tool use needs one actor, not every CPU on the machine.
        await super().start(minimum_size=minimum_size or (1 if self.pool_size is None else None))
