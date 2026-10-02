"""Process-local network isolation checks and explicit Docker launch preparation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_MODE": "disabled", "RAY_USAGE_STATS_ENABLED": "0",
    "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "PYTHONDONTWRITEBYTECODE": "1",
}


def verify_network_isolation(net_dev="/proc/net/dev"):
    """Fail closed unless this process's Linux network namespace is loopback-only.

    Offline library flags or a caller-provided environment variable alone do not
    prove isolation. This checks the namespace, without attempting internet I/O.
    """
    lines = Path(net_dev).read_text().splitlines()[2:]
    interfaces = sorted(line.split(":", 1)[0].strip() for line in lines if ":" in line)
    if interfaces != ["lo"]:
        raise RuntimeError(f"Offline SWE driver requires a loopback-only network namespace; found {interfaces}")
    return {"driver_network": "loopback_only"}


def _mount(path, *, writable=False):
    path = Path(path).expanduser().resolve(strict=True)
    if "," in str(path) or "\n" in str(path):
        raise ValueError("Docker mount paths cannot contain commas or newlines")
    return f"type=bind,src={path},dst={path}" + ("" if writable else ",readonly")


def launch_command(args, *, inspect_image=True):
    root = Path(__file__).resolve().parents[4]
    artifact = Path(args.artifact_root).expanduser().resolve(strict=True)
    if not artifact.is_dir() or artifact == root or root.is_relative_to(artifact):
        raise ValueError("Use a dedicated existing artifact directory, not an ancestor of project source")
    socket = Path(args.docker_socket).resolve(strict=True)
    image = args.image
    if inspect_image:
        # Resolve only a locally cached image. docker run is forbidden to pull.
        image = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
    command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--init",
               "--name", "dyad-swe-driver-" + uuid.uuid4().hex[:12],
               "--security-opt=no-new-privileges", "--shm-size", args.shm_size,
               "--mount", _mount(root), "--mount", _mount(artifact, writable=True),
               "--mount", f"type=bind,src={socket},dst=/var/run/docker.sock",
               "--workdir", str(root)]
    for path in args.mount_readonly:
        resolved = Path(path).expanduser().resolve(strict=True)
        if not resolved.is_relative_to(root):
            command.extend(["--mount", _mount(resolved)])
    if args.gpus:
        # Docker parses this argument as CSV even without a shell involved.
        gpu_request = args.gpus
        if gpu_request.startswith("device=") and "," in gpu_request:
            gpu_request = '"' + gpu_request + '"'
        command.extend(["--gpus", gpu_request])
    environment = dict(OFFLINE_ENV, ARTIFACT_ROOT=str(artifact), RUN_SITE="local",
                       HOST_UID=str(os.getuid()), HOST_GID=str(os.getgid()),
                       HF_HOME=str(artifact / "offline_cache/huggingface"),
                       XDG_CACHE_HOME=str(artifact / "offline_cache"),
                       TRITON_CACHE_DIR=str(artifact / "offline_cache/triton"),
                       TMPDIR="/tmp", RAY_TMPDIR="/tmp/ray", RAY_NODE_IP_ADDRESS="127.0.0.1",
                       PYTHONPATH=str(root), PYTHON_BIN="/opt/venv-expa-verl/bin/python",
                       SWE_PYTHON_BIN="/opt/venv-swebench/bin/python",
                       SWEBENCH_PYTHON_BIN="/opt/venv-swebench/bin/python", DOCKER_HOST="unix:///var/run/docker.sock")
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator or not key or key in environment or key in ("LD_PRELOAD", "PYTHONHOME"):
            raise ValueError(f"Invalid or reserved environment override: {key}")
        if any(token in key.upper() for token in ("TOKEN", "PASSWORD", "SECRET", "API_KEY", "PROXY")):
            raise ValueError("Credentials and proxies are not accepted by the offline launcher")
        environment[key] = value
    for key, value in environment.items():
        command.extend(["--env", f"{key}={value}"])
    evaluator = root / "experiments/shared/train_eval/evaluate.sh"
    check = Path(__file__).resolve()
    command.extend(["--entrypoint", "/bin/bash", image, "-c",
                    '"$SWE_PYTHON_BIN" "$1" check && shift && exec bash "$@"',
                    "swe-offline", str(check), str(evaluator), "swebench_verified", args.algorithm, *args.evaluation_args])
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Verify this process is isolated; does not contact external endpoints")
    launch = sub.add_parser("launch", help="Run the shared evaluator in a network-disabled driver container")
    launch.add_argument("--image", required=True, help="Existing local Dyad image containing /opt/venv-swebench")
    launch.add_argument("--artifact-root", required=True)
    launch.add_argument("--docker-socket", default="/var/run/docker.sock")
    launch.add_argument("--mount-readonly", action="append", default=[])
    launch.add_argument("--env", action="append", default=[])
    launch.add_argument("--gpus", help="Explicit Docker GPU selection, e.g. device=0,1")
    launch.add_argument("--shm-size", default="8g")
    launch.add_argument("--algorithm", choices=("grpo_react", "gigpo", "dyad-grpo", "dyad-gigpo"), default="grpo_react")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("evaluation_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            print(json.dumps(verify_network_isolation(), sort_keys=True))
            return 0
        if args.evaluation_args[:1] == ["--"]:
            args.evaluation_args = args.evaluation_args[1:]
        command = launch_command(args)
        if args.dry_run:
            print(json.dumps(command, ensure_ascii=False, indent=2))
            return 0
        return subprocess.call(command)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"SWE offline: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
