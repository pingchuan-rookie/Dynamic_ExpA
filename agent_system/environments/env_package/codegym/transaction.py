"""One CodeGym step on a disposable dill snapshot, without importing Ray or verl.

Clean helper processes use inherited anonymous files for payloads and a
tiny socket protocol for deadlines. A cleanup-only supervisor behind a Ray-safe
bootstrap kills the private process group on actor loss or trial completion.
No pipe-sized payloads, queue feeder threads, Python preexec_fn, or forked copy
of a multithreaded Ray actor are involved.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time


# Separate framework phases from the action deadline. The outer RPC watchdog
# also covers parent-side dill serialization/deserialization and process launch.
_FRAMEWORK_TIMEOUT_S = 5.0
_CLEANUP_TIMEOUT_S = 0.5


class TransactionError(RuntimeError):
    """A launch, serialization, child crash, or protocol failure (not an action error)."""


def validate_action_timeout(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("CodeGym action_timeout_s must be finite and positive")
    return value


def _receive(control: socket.socket, timeout_s: float, phase: str) -> bytes:
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([control], [], [], remaining)[0]:
            raise TimeoutError(f"CodeGym transaction {phase} timed out")
        token = control.recv(1)
        if not token:
            raise TransactionError(f"CodeGym transaction child exited during {phase}")
        return token


def _cleanup(process: subprocess.Popen) -> None:
    # start_new_session gives every trial its own group. Kill even after a
    # successful exit so subprocesses started by an action cannot outlive it.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_CLEANUP_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise TransactionError("CodeGym transaction child could not be reaped") from exc


def step_transaction(env, action: str, action_timeout_s: float):
    """Return (candidate, status, observation, action_error, action_timed_out).

    Only an uncaught exception from the call to env.step, or expiration of its
    execution deadline, is recoverable. In both cases candidate is the original
    object. Any normally returned pair commits regardless of status or text.
    """
    action_timeout_s = validate_action_timeout(action_timeout_s)
    import dill

    if sys.platform != "linux":
        raise TransactionError("CodeGym transaction isolation requires Linux parent-death signaling")
    # Anonymous files survive exec through pass_fds, but leave no artifacts even
    # when the actor is SIGKILLed. Large input/output cannot fill a control pipe.
    with tempfile.TemporaryFile() as request, tempfile.TemporaryFile() as response:
        dill.dump(list(sys.path), request)
        dill.dump((env, action), request)
        request.flush()
        request.seek(0)
        parent_control, child_control = socket.socketpair()
        with parent_control, child_control:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--bootstrap",
                 str(os.getpid()), str(request.fileno()), str(response.fileno()),
                 str(child_control.fileno())],
                pass_fds=(request.fileno(), response.fileno(), child_control.fileno()),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            child_control.close()
            try:
                try:
                    if _receive(parent_control, _FRAMEWORK_TIMEOUT_S, "startup") != b"R":
                        raise TransactionError("CodeGym transaction invalid startup response")
                    parent_control.sendall(b"G")
                    try:
                        returned = _receive(parent_control, action_timeout_s, "execution")
                    except TimeoutError:
                        message = f"CodeGym action timed out after {action_timeout_s:g}s"
                        return env, False, f"Error: {message}", message, True
                    if returned not in (b"S", b"E"):
                        raise TransactionError("CodeGym transaction invalid execution response")
                    if _receive(parent_control, _FRAMEWORK_TIMEOUT_S, "serialization") != b"D":
                        raise TransactionError("CodeGym transaction invalid serialization response")
                    # The cleanup-only supervisor confirms the trial's zero
                    # exit, then kills its own group including any descendants.
                    if _receive(parent_control, _CLEANUP_TIMEOUT_S, "trial exit") != b"X":
                        raise TransactionError("CodeGym transaction child failed before clean exit")
                    if process.wait(timeout=_CLEANUP_TIMEOUT_S) != -signal.SIGKILL:
                        raise TransactionError(f"CodeGym transaction supervisor failed (exit {process.returncode})")
                except (TimeoutError, OSError, subprocess.TimeoutExpired) as exc:
                    raise TransactionError(f"CodeGym transaction framework failure: {exc}") from exc
                response.seek(0)
                result = dill.load(response)
                if returned == b"E":
                    if not isinstance(result, str):
                        raise TransactionError("CodeGym transaction invalid action exception")
                    return env, False, f"Error: {result}", result, False
                candidate, status, observation = result
                return candidate, status, observation, None, False
            finally:
                _cleanup(process)


def _arm_parent_death(expected_parent_pid: int, death_signal: int = signal.SIGKILL) -> None:
    # Run only in clean helper processes, never a Popen preexec_fn in the
    # multithreaded actor. The supervisor uses SIGTERM to kill the whole group.
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(death_signal), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # The parent may have died between Popen and prctl. Do not start an orphan.
    if os.getppid() != expected_parent_pid:
        os._exit(1)


def _spawn_role(role, role_name: str, args: tuple[int, ...], pass_fds: tuple[int, ...]):
    """Reuse the clean shim interpreter, never fork the multithreaded Ray actor.

    Only the freshly exec'd bootstrap and its cleanup-only supervisor call this.
    Check native threads too: site customization may have started threads even
    before this stdlib-only module runs. In that case retain the exec path.
    """
    try:
        single_threaded = __name__ == "__main__" and len(os.listdir("/proc/self/task")) == 1
    except OSError:
        single_threaded = False
    if single_threaded:
        pid = os.fork()
        if pid == 0:
            # exec used to reset the supervisor's cleanup signal handler. Keep
            # that boundary: trial code must not inherit the parent's handler.
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            try:
                role(*args)
            finally:
                os._exit(1)

        def wait():
            return os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])

        return wait
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), role_name, *(str(value) for value in args)],
        pass_fds=pass_fds, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return process.wait


def _bootstrap(expected_parent_pid: int, request_fd: int, response_fd: int, control_fd: int) -> None:
    # Ray explicitly SIGKILLs a worker's direct children before worker exit.
    # Keep the cleanup supervisor one generation deeper so it can handle this
    # direct child's death, rather than itself being killed before cleanup.
    _arm_parent_death(expected_parent_pid)
    wait_supervisor = _spawn_role(
        _supervisor, "--supervisor", (os.getpid(), request_fd, response_fd, control_fd),
        (request_fd, response_fd, control_fd),
    )
    for fd in (request_fd, response_fd, control_fd):
        os.close(fd)
    wait_supervisor()
    os.killpg(os.getpgrp(), signal.SIGKILL)


def _supervisor(expected_parent_pid: int, request_fd: int, response_fd: int, control_fd: int) -> None:
    # This process never imports dill or executes env code, so its Python signal
    # handler cannot be stuck inside an action/native extension. The bootstrap
    # is the private session/group leader, known even during the launch race.
    group_pid = os.getpgrp()
    if group_pid != expected_parent_pid:
        raise TransactionError("CodeGym supervisor requires a private process group")

    def kill_group(*_):
        os.killpg(group_pid, signal.SIGKILL)

    signal.signal(signal.SIGTERM, kill_group)
    # Ray execution threads may block SIGTERM; exec preserves that mask.
    # Unblock only after installing the cleanup handler in this fresh process.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
    _arm_parent_death(expected_parent_pid, signal.SIGTERM)
    # Do not launch a trial until both parent-death handling and race checking
    # are active. All descendants inherit this group, not a new session.
    try:
        with socket.socket(fileno=control_fd) as control:
            wait_trial = _spawn_role(
                _child, "--child", (os.getpid(), request_fd, response_fd, control_fd),
                (request_fd, response_fd, control_fd),
            )
            os.close(request_fd)
            os.close(response_fd)
            # Only a clean trial exit can authorize commit. Child crashes send
            # an invalid token promptly even if descendants kept the FD open.
            control.sendall(b"X" if wait_trial() == 0 else b"F")
    finally:
        kill_group()


def _child(expected_parent_pid: int, request_fd: int, response_fd: int, control_fd: int) -> None:
    _arm_parent_death(expected_parent_pid)
    import dill

    with socket.socket(fileno=control_fd) as control:
        with os.fdopen(request_fd, "rb") as request:
            # Make referenced env modules importable before deserializing, but
            # never import the parent's Ray main module.
            sys.path[:] = dill.load(request)
            env, action = dill.load(request)
        control.sendall(b"R")
        if _receive(control, _FRAMEWORK_TIMEOUT_S, "dispatch") != b"G":
            raise TransactionError("CodeGym transaction invalid dispatch")
        try:
            result = env.step(action)
        except Exception as exc:
            control.sendall(b"E")
            payload = f"{type(exc).__name__}: {exc}"
        else:
            control.sendall(b"S")
            # Malformed return values, conversions, and serialization are
            # framework failures, not exceptions raised by env.step itself.
            status, observation = result
            payload = env, bool(status), str(observation)
        with os.fdopen(response_fd, "wb") as response:
            dill.dump(payload, response)
        control.sendall(b"D")
    # Avoid environment atexit hooks or non-daemon threads delaying exit.
    os._exit(0)


if __name__ == "__main__":
    entries = {"--child": _child, "--supervisor": _supervisor, "--bootstrap": _bootstrap}
    if len(sys.argv) != 6 or sys.argv[1] not in entries:
        raise SystemExit("This module is an internal CodeGym transaction child")
    entry = entries[sys.argv[1]]
    entry(*(int(value) for value in sys.argv[2:]))
