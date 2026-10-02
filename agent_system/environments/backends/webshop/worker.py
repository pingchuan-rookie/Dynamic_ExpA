"""Import-light Ray actor owning one JSONL sidecar, not one catalog per episode."""
from __future__ import annotations

import atexit
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time


# Set parent-death behavior after exec, not via unsafe preexec_fn in a threaded
# Ray actor. Check the original parent PID to cover death before prctl executes.
_SIDECAR_BOOTSTRAP = """
import ctypes, os, runpy, signal, sys
script, parent_pid = sys.argv[1], int(sys.argv[2])
if sys.platform == 'linux':
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Unable to set WebShop parent-death signal')
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
sys.argv = [script]
runpy.run_path(script, run_name='__main__')
"""


class WebShopBackendError(RuntimeError):
    """A backend operation failed with a structured protocol error."""


class WebShopBackendOperationError(WebShopBackendError):
    """A synchronized backend response rejected only this operation."""


class WebShopEnvWorker:
    def __init__(self, config: dict):
        self.config = dict(config)
        self._process = None
        self._stderr = None
        self._buffer = b""
        self._sequence = 0
        self._closed = False
        self._initialized = False
        atexit.register(self.shutdown)

    def _start(self):
        if self._closed:
            raise WebShopBackendError("WebShop sidecar has shut down")
        if self._process is not None:
            return
        self._stderr = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                [self.config["python_executable"], "-u", "-c", _SIDECAR_BOOTSTRAP,
                 self.config["backend_script"], str(os.getpid())],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                start_new_session=True, bufsize=0,
            )
            os.set_blocking(self._process.stdin.fileno(), False)
            os.set_blocking(self._process.stdout.fileno(), False)
        except BaseException:
            self.shutdown()
            raise

    def _stderr_context(self):
        if self._stderr is None:
            return ""
        self._stderr.seek(0, os.SEEK_END)
        self._stderr.seek(max(0, self._stderr.tell() - 4096))
        return self._stderr.read().decode("utf-8", errors="replace")

    def _exchange(self, request: dict, timeout: float):
        payload = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if len(payload) > 1024 * 1024:
            raise ValueError("WebShop request exceeds 1 MiB")
        process = self._process
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdin, selectors.EVENT_WRITE)
            sent = 0
            while sent < len(payload):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(f"WebShop {request['op']} timed out after {timeout}s")
                sent += os.write(process.stdin.fileno(), payload[sent:])
            selector.unregister(process.stdin)
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(f"WebShop {request['op']} timed out after {timeout}s")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise WebShopBackendError(
                        f"WebShop backend exited (code={process.poll()}) during {request['op']}; {self._stderr_context()}"
                    )
                self._buffer += chunk
                if len(self._buffer) > 16 * 1024 * 1024:
                    raise WebShopBackendError("WebShop response exceeds 16 MiB")
        line, self._buffer = self._buffer.split(b"\n", 1)
        response = json.loads(line)
        if not isinstance(response, dict) or response.get("id") != request["id"]:
            raise WebShopBackendError("WebShop JSONL response has mismatched request id")
        if response.get("ok") is not True:
            error = response.get("error", {})
            raise WebShopBackendOperationError(f"WebShop {request['op']} failed: {error}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise WebShopBackendError("WebShop JSONL result must be an object")
        return result

    def _request(self, op, timeout=None, **payload):
        self._start()
        self._sequence += 1
        try:
            return self._exchange({"id": self._sequence, "op": op, **payload},
                                  timeout or self.config["request_timeout"])
        except (TimeoutError, OSError, ValueError) as exc:
            context = self._stderr_context()
            self.shutdown()
            raise WebShopBackendError(f"WebShop {op} transport failed ({type(exc).__name__}: {exc}); {context}") from exc
        except WebShopBackendOperationError:
            raise
        except WebShopBackendError:
            self.shutdown()
            raise

    def _memory_sample(self):
        # Linux process RSS includes the in-process JVM/Lucene plus full catalog.
        # Sampling /proc is cheap; unavailable metrics remain null, never zero.
        pid = self._process.pid if self._process is not None else None
        result = {"backend_pid": pid, "backend_rss_bytes": None, "backend_peak_rss_bytes": None}
        if pid is not None:
            try:
                with open(f"/proc/{pid}/status", encoding="utf-8") as stream:
                    for line in stream:
                        key, _, value = line.partition(":")
                        if key in {"VmRSS", "VmHWM"}:
                            name = "backend_rss_bytes" if key == "VmRSS" else "backend_peak_rss_bytes"
                            result[name] = int(value.split()[0]) * 1024
            except (OSError, ValueError):
                pass
        return result

    def health_check(self):
        if not self._initialized:
            result = self._request("init", timeout=self.config["startup_timeout"], config=self.config["backend_config"])
            self._initialized = bool(result.get("ready"))
        else:
            result = self._request("health")
        return {**result, "ok": bool(result.get("ready")),
                "worker_python": sys.executable, "backend_python": self.config["python_executable"],
                **self._memory_sample()}

    def reset(self, session_id, **reset_spec):
        if not self._initialized:
            self.health_check()
        task_id = reset_spec["task_id"]
        split = reset_spec.get("split") or ("test" if task_id < 500 else "dev" if task_id < 1500 else "train")
        result = self._request("create", session_id=session_id, task_id=task_id, split=split)
        expected = reset_spec.get("initial_observation")
        if expected is not None and result.get("observation") != expected:
            self.close(session_id)
            raise ValueError(f"WebShop initial observation mismatch for task_id={task_id}")
        return {**result, **self._memory_sample()}

    def step(self, session_id, action):
        return self._request("step", session_id=session_id, action=action)

    def close(self, session_id):
        # Does not terminate the sidecar, its server or any other leased session.
        return {**self._request("close", session_id=session_id), **self._memory_sample()}

    def shutdown(self):
        self._closed = True
        process, self._process = self._process, None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2)
            if process.stdout is not None:
                process.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        atexit.unregister(self.shutdown)
        return {"closed": True}
