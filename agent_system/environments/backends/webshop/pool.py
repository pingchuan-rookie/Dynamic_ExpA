"""Bounded Ray backend replicas, each sharing one full catalog across leased sessions."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from uuid import uuid4

from agent_system.environments.backends.webshop.config import resolve_webshop_config
from agent_system.environments.backends.webshop.worker import WebShopEnvWorker
from agent_system.utils.diagnostics import full_dump, log_event


@dataclass
class _Lease:
    worker: object
    token: str
    session_id: str
    worker_idx: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class WebShopEnvPool:
    WORKER_CLS = WebShopEnvWorker
    LOG_NAME = "webshop_env_pool"

    def __init__(self, config: dict):
        self.config = resolve_webshop_config(config)
        self.pool_size = self.config["backend_replicas"]
        self._workers = []
        self._sessions: dict[str, _Lease] = {}
        self._pending: set[str] = set()
        self._create_tasks = {}
        self._free = asyncio.Queue()
        self._start_lock = asyncio.Lock()
        self._worker_locks = {}
        self._started = False
        self._error = None
        self._waiters = 0
        self._peak_active = 0

    def _log(self, event, **fields):
        log_event(self.LOG_NAME, event, backend="ray", pool_size=self.pool_size,
                  backend_replicas=self.pool_size, sessions_per_backend=self.config["sessions_per_backend"],
                  **fields)

    @staticmethod
    def _memory_fields(response):
        return {key: response.get(key) for key in ("backend_pid", "backend_rss_bytes", "backend_peak_rss_bytes")}

    async def _call(self, worker, method, *args, **kwargs):
        return await getattr(worker, method).remote(*args, **kwargs)

    async def _rpc(self, worker, method, *args, **kwargs):
        # Sidecar is serial; avoid starting the RPC watchdog while queued behind
        # another session's legitimate request. The sidecar has its own deadline.
        async with self._worker_locks[id(worker)]:
            self._check()
            return await asyncio.wait_for(
                self._call(worker, method, *args, **kwargs),
                timeout=self.config["request_timeout"] + 10,
            )

    async def start(self):
        async with self._start_lock:
            if self._error:
                raise RuntimeError(self._error)
            if self._started:
                return
            import ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            started_at = time.perf_counter()
            try:
                cpus = self.config["num_cpus_per_worker"]
                available = float(ray.available_resources().get("CPU", 0))
                if available + 1e-8 < cpus * self.pool_size:
                    raise RuntimeError(f"WebShop needs {cpus * self.pool_size} available Ray CPUs, got {available}")
                # Do not set py_executable/runtime_env: Ray stays in dyad-verl.
                remote = ray.remote(num_cpus=cpus, max_restarts=0)(self.WORKER_CLS)
                backend_memory = []
                for _ in range(self.pool_size):
                    worker = remote.remote(self.config)
                    self._workers.append(worker)
                    self._worker_locks[id(worker)] = asyncio.Lock()
                    health = await asyncio.wait_for(
                        self._call(worker, "health_check"), self.config["startup_timeout"] + 10
                    )
                    if not health.get("ok"):
                        raise RuntimeError(f"WebShop backend health failed: {health}")
                    backend_memory.append({"worker_idx": len(self._workers) - 1, **self._memory_fields(health)})
                # Round-robin slots avoid concentrating every session on replica 0.
                for _ in range(self.config["sessions_per_backend"]):
                    for worker in self._workers:
                        self._free.put_nowait(worker)
                self._started = True
                self._log("pool_init", ok=True, health={"ok": True, "checked_workers": len(self._workers)},
                          backend_memory=backend_memory,
                          duration_ms=round((time.perf_counter() - started_at) * 1000, 3))
            except BaseException as exc:
                self._log("pool_init", ok=False, error=type(exc).__name__,
                          duration_ms=round((time.perf_counter() - started_at) * 1000, 3))
                self._error = "WebShop backend pool startup failed; see originating exception"
                await self._shutdown_workers()
                raise

    def _check(self):
        if self._error:
            raise RuntimeError(self._error)

    async def create_session(self, session_id: str, reset_spec: dict | None = None) -> dict:
        if session_id in self._sessions or session_id in self._pending:
            raise ValueError(f"duplicate active session_id={session_id!r}")
        self._pending.add(session_id)
        self._create_tasks[session_id] = asyncio.current_task()
        worker = None
        lease = None
        try:
            await self.start()
            self._check()
            waited_at = time.perf_counter()
            self._waiters += 1
            try:
                worker = await asyncio.wait_for(self._free.get(), self.config["lease_timeout"])
            except asyncio.TimeoutError as exc:
                raise TimeoutError("WebShop session lease timed out; release completed trajectories") from exc
            finally:
                self._waiters -= 1
            self._check()
            lease = _Lease(worker, uuid4().hex, session_id, self._workers.index(worker))
            self._sessions[session_id] = lease
            self._peak_active = max(self._peak_active, len(self._sessions))
            waited_ms = round((time.perf_counter() - waited_at) * 1000, 3)
            reset_at = time.perf_counter()
            async with lease.lock:
                response = await self._rpc(worker, "reset", session_id=lease.token, **dict(reset_spec or {}))
                expected = (reset_spec or {}).get("initial_observation")
                if expected is not None and response.get("observation") != expected:
                    task_id = (reset_spec or {}).get("task_id")
                    raise ValueError(f"WebShop initial observation mismatch for task_id={task_id}")
                self._log("env_reset", session_id=session_id, worker_idx=lease.worker_idx,
                          task_id=(reset_spec or {}).get("task_id"), waited_ms=waited_ms,
                          active_sessions=len(self._sessions), peak_active=self._peak_active,
                          duration_ms=round((time.perf_counter() - reset_at) * 1000, 3),
                          ok=True, obs_len=len(response.get("observation", "")), **self._memory_fields(response))
                response.setdefault("worker_idx", lease.worker_idx)
                return response
        except BaseException:
            if lease is not None:
                await asyncio.shield(self.close_session(session_id))
            elif worker is not None and not self._error:
                self._free.put_nowait(worker)
            raise
        finally:
            self._pending.discard(session_id)
            self._create_tasks.pop(session_id, None)

    async def step_session(self, session_id: str, action: str) -> dict:
        lease = self._sessions.get(session_id)
        if lease is None:
            raise KeyError(f"unknown session_id={session_id!r}")
        async with lease.lock:
            if self._sessions.get(session_id) is not lease:
                raise KeyError(f"released session_id={session_id!r}")
            started_at = time.perf_counter()
            try:
                response = await self._rpc(lease.worker, "step", lease.token, action)
                cap = 4096 if full_dump() else 512
                self._log("env_step_bind", session_id=session_id, worker_idx=lease.worker_idx,
                          action_in=action[:cap], observation_out=response.get("observation", "")[:cap],
                          reward=response.get("reward"), done=response.get("done"),
                          server_ms=round((time.perf_counter() - started_at) * 1000, 3))
                response.setdefault("worker_idx", lease.worker_idx)
                return response
            except BaseException as exc:
                self._log("env_abort", session_id=session_id, worker_idx=lease.worker_idx,
                          reason=f"step failed: {type(exc).__name__}", active_sessions=len(self._sessions) - 1)
                # The actor serializes close after any interrupted request. Cleanup
                # runs to completion before this slot can be leased again.
                self._sessions.pop(session_id, None)
                await asyncio.shield(self._close_lease(lease))
                raise

    async def _close_lease(self, lease):
        started_at = time.perf_counter()
        error = None
        response = {}
        try:
            response = await self._rpc(lease.worker, "close", lease.token)
        except BaseException as exc:
            error = type(exc).__name__
            self._error = f"WebShop backend session cleanup failed ({type(exc).__name__}: {exc}); pool unavailable"
            # Do not leave a wedged backend or waiters alive.
            await asyncio.shield(self._shutdown_workers())
            for _ in range(self._waiters):
                self._free.put_nowait(None)
            raise
        else:
            if not self._error:
                self._free.put_nowait(lease.worker)
        finally:
            self._log("env_close", session_id=lease.session_id, worker_idx=lease.worker_idx,
                      ok=error is None, error=error, active_sessions=len(self._sessions),
                      **self._memory_fields(response), duration_ms=round((time.perf_counter() - started_at) * 1000, 3))

    async def close_session(self, session_id: str):
        lease = self._sessions.get(session_id)
        if lease is None:
            return
        async with lease.lock:
            if self._sessions.get(session_id) is not lease:
                return
            self._sessions.pop(session_id, None)
            await asyncio.shield(self._close_lease(lease))

    async def abort_session(self, session_id: str, reason: str):
        lease = self._sessions.get(session_id)
        if lease is not None or session_id in self._pending:
            self._log("env_abort", session_id=session_id,
                      worker_idx=lease.worker_idx if lease is not None else -1, reason=reason,
                      active_sessions=len(self._sessions) - int(lease is not None))
        # A queued create has no lease yet; cancel it rather than returning and
        # allowing it to materialize an unowned session after the caller aborts.
        task = self._create_tasks.get(session_id)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        # Abort drops this browser/session, never a healthy shared full catalog.
        await self.close_session(session_id)

    async def _shutdown_workers(self):
        import ray
        workers, self._workers = self._workers, []
        for worker in workers:
            try:
                await asyncio.wait_for(self._call(worker, "shutdown"), 5)
            except BaseException:
                pass
            finally:
                try:
                    ray.kill(worker, no_restart=True)
                except Exception:
                    pass

    async def shutdown(self):
        self._error = "WebShop backend pool has shut down"
        self._started = False
        for _ in range(self._waiters):
            self._free.put_nowait(None)
        # A queued create owns no session yet, but must not outlive shutdown.
        creating = [task for task in self._create_tasks.values() if task is not asyncio.current_task()]
        for task in creating:
            task.cancel()
        if creating:
            await asyncio.gather(*creating, return_exceptions=True)
        # Also serialize against a direct start(), which may be registering actors.
        async with self._start_lock:
            await self._shutdown_workers()
            self._started = False
            self._sessions.clear()
            while not self._free.empty():
                self._free.get_nowait()
