"""Offline Docker controller. Agent bytes never execute on the host.

Only immutable, already-local images are accepted. No build/pull API, host bind
mount, or global Docker cleanup is used. Workspace diffs are produced by trusted
host Git from bounded Docker archives, not by agent-controlled Git or hooks.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid

from .tools import FILE_TOOL_PROGRAM, action_tools, build_initial_messages, scaffold_identity


class SweBenchInfrastructureError(RuntimeError):
    """An attempt cannot be officially scored; never convert this into reward zero."""


@dataclass(frozen=True)
class RuntimeConfig:
    max_steps: int = 30
    tool_timeout_s: float = 60
    episode_timeout_s: float = 1800
    test_timeout_s: float = 1800
    cpus: float = 2
    memory_bytes: int = 4 * 1024**3
    pids_limit: int = 256
    max_output_bytes: int = 65536
    max_snapshot_bytes: int = 1024**3
    max_patch_bytes: int = 16 * 1024**2
    max_test_output_bytes: int = 64 * 1024**2
    arch: str = "x86_64"

    def __post_init__(self):
        for field in fields(self):
            if field.name == "arch":
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be positive and finite")
            if field.name not in {"cpus", "tool_timeout_s", "episode_timeout_s", "test_timeout_s"} and not isinstance(value, int):
                raise ValueError(f"{field.name} must be an integer")
        if self.arch not in {"x86_64", "arm64"}:
            raise ValueError("Unsupported architecture")


def local_client():
    import docker
    return docker.from_env(timeout=30)


class DockerSandbox:
    """One owned container. The SDK creates, but never pulls, the fixed image."""
    def __init__(self, client, image, config, *, purpose):
        self.client, self.config = client, config
        self.container = None
        if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise SweBenchInfrastructureError("Runtime requires an immutable local Docker image ID")
        try:
            found = client.images.get(image)
            if found.id != image:
                raise ValueError("Local image identity mismatch")
            expected = "amd64" if config.arch == "x86_64" else "arm64"
            if found.attrs.get("Architecture") != expected or found.attrs.get("Os") != "linux":
                raise ValueError("Local image platform mismatch")
            image_config = found.attrs.get("Config") or {}
            if image_config.get("Volumes"):
                raise ValueError("Image-declared volumes are not permitted")
            self.container = client.containers.create(
                image=image, name=f"dyad-swe-{purpose}-{uuid.uuid4().hex}",
                command=["/bin/sleep", "infinity"], entrypoint=[], working_dir="/testbed",
                user="root", network_mode="none", network_disabled=True, privileged=False,
                cap_drop=["ALL"], security_opt=["no-new-privileges:true"],
                nano_cpus=int(config.cpus * 1_000_000_000), mem_limit=config.memory_bytes,
                memswap_limit=config.memory_bytes, pids_limit=config.pids_limit,
                init=True, detach=True, labels={"dyad.swebench.owned": "true", "dyad.swebench.purpose": purpose},
                environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                             "WANDB_MODE": "disabled", "DO_NOT_TRACK": "1"},
            )
            self.container.start()
            self.inspection = self.inspect_offline()
        except Exception as exc:
            self.close()
            raise SweBenchInfrastructureError("Cannot start required local offline container; no pull/build attempted") from exc

    def inspect_offline(self):
        """Verify the daemon's actual isolation state, not merely create kwargs."""
        self.container.reload()
        raw = self.container.attrs
        host = raw.get("HostConfig") or {}
        networks = (raw.get("NetworkSettings") or {}).get("Networks") or {}
        if (host.get("NetworkMode") != "none" or host.get("Privileged") is not False
                or "ALL" not in {str(cap).upper() for cap in host.get("CapDrop", [])}
                or not any(str(opt) in {"no-new-privileges", "no-new-privileges:true"}
                           for opt in host.get("SecurityOpt", []))
                or host.get("NanoCpus", 0) <= 0 or host.get("Memory", 0) <= 0
                or host.get("PidsLimit", 0) <= 0 or host.get("Binds") or raw.get("Mounts")
                or set(networks) - {"none"}):
            raise SweBenchInfrastructureError("Docker daemon did not enforce required container isolation")
        return {"Id": raw["Id"], "Image": raw["Image"],
                "Config": {"Labels": deepcopy((raw.get("Config") or {}).get("Labels") or {})},
                "HostConfig": {key: deepcopy(host.get(key)) for key in (
                    "NetworkMode", "Privileged", "CapDrop", "SecurityOpt", "NanoCpus",
                    "Memory", "MemorySwap", "PidsLimit", "Binds")},
                "Mounts": deepcopy(raw.get("Mounts") or []),
                "NetworkSettings": {"Networks": deepcopy(networks)}}

    def exec(self, command, *, timeout=None, max_bytes=None, workdir="/testbed"):
        if self.container is None:
            raise SweBenchInfrastructureError("Container is closed")
        timeout = self.config.tool_timeout_s if timeout is None else timeout
        limit = self.config.max_output_bytes if max_bytes is None else max_bytes
        output = bytearray()
        state = {"truncated": False}
        container = self.container

        def execute():
            try:
                api = self.client.api
                execution = api.exec_create(container.id, command, workdir=workdir, user="root")
                for chunk in api.exec_start(execution["Id"], stream=True):
                    if isinstance(chunk, str):
                        chunk = chunk.encode()
                    room = max(0, limit - len(output))
                    output.extend(chunk[:room])
                    if len(chunk) > room:
                        state["truncated"] = True
                state["exit_code"] = api.exec_inspect(execution["Id"])["ExitCode"]
            except Exception as exc:
                state["error"] = exc

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self.close()
            thread.join(5)
            raise SweBenchInfrastructureError("Container execution exceeded its hard deadline")
        if "error" in state or state.get("exit_code") is None:
            raise SweBenchInfrastructureError("Docker execution failed") from state.get("error")
        return {"exit_code": state["exit_code"], "output": bytes(output).decode("utf-8", errors="replace"),
                "truncated": state["truncated"]}

    def checked(self, command, **kwargs):
        result = self.exec(command, **kwargs)
        if result["exit_code"] != 0 or result["truncated"]:
            # No raw diagnostics here: trusted setup/grading can contain hidden data.
            raise SweBenchInfrastructureError("Trusted container setup command failed")
        return result["output"]

    def put_file(self, path, content):
        path = PurePosixPath(path)
        data = content.encode() if isinstance(content, str) else content
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as stream:
            member = tarfile.TarInfo(path.name)
            member.size, member.mode = len(data), 0o600
            stream.addfile(member, io.BytesIO(data))
        if not self.container.put_archive(str(path.parent), archive.getvalue()):
            raise SweBenchInfrastructureError("Docker archive upload failed")

    def snapshot(self, destination):
        """Bounded regular files/symlinks only; never tar.extract on untrusted bytes."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            deadline = time.monotonic() + self.config.tool_timeout_s
            chunks, _ = self.container.get_archive("/testbed")
            # Spool the archive to disk, not RAM. Includes an allowance for tar metadata.
            with tempfile.TemporaryFile() as archive:
                total = 0
                for chunk in chunks:
                    if time.monotonic() > deadline:
                        raise SweBenchInfrastructureError("Workspace snapshot exceeded its deadline")
                    total += len(chunk)
                    if total > self.config.max_snapshot_bytes + 64 * 1024**2:
                        raise SweBenchInfrastructureError("Workspace archive exceeds byte budget")
                    archive.write(chunk)
                archive.seek(0)
                with tarfile.open(fileobj=archive, mode="r|*") as stream:
                    seen, total, count = set(), 0, 0
                    for member in stream:
                        if time.monotonic() > deadline:
                            raise SweBenchInfrastructureError("Workspace extraction exceeded its deadline")
                        parts = PurePosixPath(member.name).parts
                        if not parts or parts[0] != "testbed" or ".." in parts or member.name.startswith("/"):
                            raise SweBenchInfrastructureError("Unsafe workspace archive path")
                        relative = PurePosixPath(*parts[1:])
                        if ".git" in relative.parts or not relative.parts:
                            continue
                        count += 1
                        if count > 200000:
                            raise SweBenchInfrastructureError("Workspace has too many entries")
                        target = destination / str(relative)
                        if relative in seen:
                            raise SweBenchInfrastructureError("Duplicate workspace archive entry")
                        seen.add(relative)
                        if any(parent.is_symlink() for parent in target.parents if parent != destination.parent):
                            raise SweBenchInfrastructureError("Workspace archive traverses a symlink")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if member.isdir():
                            target.mkdir(exist_ok=True)
                        elif member.isfile():
                            total += member.size
                            if total > self.config.max_snapshot_bytes:
                                raise SweBenchInfrastructureError("Workspace exceeds byte budget")
                            with target.open("xb") as out:
                                shutil.copyfileobj(stream.extractfile(member), out)
                            target.chmod(0o755 if member.mode & 0o111 else 0o644)
                        elif member.issym():
                            # Preserve link text. Never dereference it on the host.
                            target.symlink_to(member.linkname)
                        else:
                            raise SweBenchInfrastructureError("Unsupported workspace archive file type")
        except SweBenchInfrastructureError:
            raise
        except Exception as exc:
            raise SweBenchInfrastructureError("Cannot snapshot container workspace") from exc

    def close(self):
        container, self.container = self.container, None
        if container is not None:
            try:
                container.remove(force=True, v=True)
            except Exception as exc:
                # Retain ownership so a later close can retry. Never prune other resources.
                self.container = container
                raise SweBenchInfrastructureError("Cannot remove owned SWE-bench container") from exc


def validate_repository_base(sandbox, base_commit):
    """Accept the pinned harness's single setup commit without discarding it.

    v4.1.0 commits tracked installation changes with `-am SWE-bench`, and can
    leave untracked build outputs. Those bytes belong to the immutable image
    baseline, not the candidate patch. Future issue commits are not accepted.
    """
    git = ["/usr/bin/git", "-c", "safe.directory=/testbed"]
    head = sandbox.checked([*git, "rev-parse", "HEAD"]).strip()
    if head != base_commit:
        metadata = sandbox.checked([*git, "show", "-s", "--format=%P%n%an%n%ae%n%s", "HEAD"]).strip().splitlines()
        # Published task images predate the release's .config email spelling.
        if (len(metadata) != 4 or metadata[0] != base_commit or metadata[1] != "SWE-bench"
                or metadata[2] not in {"setup@swebench.config", "setup@swebench.com"}
                or metadata[3] != "SWE-bench"):
            raise SweBenchInfrastructureError("Image HEAD is neither base_commit nor its official setup commit")
    if sandbox.checked([*git, "status", "--porcelain", "--untracked-files=no"]).strip():
        raise SweBenchInfrastructureError("Image repository has residual tracked workspace changes")
    return head


class TrustedWorkspace:
    """A private host index ignores all agent Git metadata, hooks and filters."""
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        self.tree = self.root / "tree"
        self.gitdir = self.root / "index.git"
        self.tree.mkdir()
        self.env = {"PATH": "/usr/bin:/bin", "HOME": str(self.root), "LC_ALL": "C.UTF-8",
                    "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
                    "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_ATTR_NOSYSTEM": "1",
                    "GIT_TERMINAL_PROMPT": "0"}
        self.git("init")
        (self.gitdir / "info" / "attributes").write_text("* -filter -text -ident -working-tree-encoding\n")

    def git(self, *args, output_limit=None):
        command = ["/usr/bin/git", "--git-dir", str(self.gitdir), "--work-tree", str(self.tree),
                   "-c", "core.hooksPath=/dev/null", "-c", "core.attributesFile=/dev/null",
                   "-c", "core.excludesFile=/dev/null", "-c", "core.autocrlf=false",
                   "-c", "user.name=Offline baseline", "-c", "user.email=offline@invalid", *args]
        try:
            # Disk-backed capture bounds host RAM even for adversarial binary diffs.
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
                subprocess.run(command, cwd=self.root, env=self.env, check=True,
                               stdout=output, stderr=errors, timeout=self.config.tool_timeout_s)
                size = output.tell()
                limit = self.config.max_output_bytes if output_limit is None else output_limit
                if size > limit:
                    raise SweBenchInfrastructureError("Trusted Git output exceeds byte budget")
                output.seek(0)
                return output.read()
        except (subprocess.SubprocessError, OSError) as exc:
            raise SweBenchInfrastructureError("Trusted workspace Git operation failed") from exc

    def baseline(self, sandbox):
        sandbox.snapshot(self.tree)
        self.git("add", "--all", "--force", "--", ".")
        self.git("commit", "--quiet", "--allow-empty", "--no-gpg-sign", "-m", "Base tree")

    def patch(self, sandbox):
        shutil.rmtree(self.tree)
        self.tree.mkdir()
        sandbox.snapshot(self.tree)
        self.git("add", "--all", "--force", "--", ".")
        patch = self.git("diff", "--cached", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", "HEAD", "--", ".",
                         output_limit=self.config.max_patch_bytes)
        if len(patch) > self.config.max_patch_bytes:
            raise SweBenchInfrastructureError("Submission patch exceeds byte budget")
        try:
            return patch.decode("utf-8")
        except UnicodeError as exc:
            raise SweBenchInfrastructureError("Submission patch is not valid UTF-8") from exc


class SweBenchEnv:
    protocol = "native_function_call"
    supports_gigpo = False

    def __init__(self, instance, image, *, config=None, docker_client=None, grader=None,
                 mini_agent=None, mini_artifact_dir=None):
        self.instance = deepcopy(dict(instance))
        for key in ("instance_id", "repo", "problem_statement", "base_commit"):
            if not isinstance(self.instance.get(key), str) or not self.instance[key]:
                raise ValueError(f"Missing public task field: {key}")
        if not re.fullmatch(r"[0-9a-f]{40}", self.instance["base_commit"]):
            raise ValueError("Invalid base_commit")
        self.config = config if isinstance(config, RuntimeConfig) else RuntimeConfig(**dict(config or {}))
        self.mini_agent = deepcopy(mini_agent)
        self.mini_artifact_dir = mini_artifact_dir
        self.mini_runs = []
        if mini_agent is not None:
            from .mini_agent import MiniConfig, verify_installation
            from dataclasses import asdict
            self.mini_agent = asdict(MiniConfig(**mini_agent))
            verify_installation()
            if mini_artifact_dir is None:
                raise ValueError("mini-swe-agent requires a persistent artifact directory")
        self.action_tools = action_tools(mini_agent is not None)
        self.scaffold_version, self.schema_hash = scaffold_identity(mini_agent is not None)
        self.image, self.client, self.grader = image, docker_client, grader
        self._own_client = docker_client is None
        self.sandbox = self.temporary = self.workspace = None
        self._active = self.done = False
        self.result = None
        self.step_count = self.tool_calls_count = 0
        self.task_description = (f"Repository: {self.instance['repo']}\n"
                                 f"Task: {self.instance['instance_id']}\n\n{self.instance['problem_statement']}")
        self.current_observation = "No tool results yet."
        self.initial_messages = build_initial_messages(self.instance, mini_agent is not None)
        self.messages = deepcopy(self.initial_messages)
        self.call_ids = set()
        self.termination_reason = None

    def start(self):
        if self._active or self.result is not None:
            raise RuntimeError("SWE-bench session cannot be restarted")
        try:
            self.client = self.client or local_client()
            self.sandbox = DockerSandbox(self.client, self.image, self.config, purpose="agent")
            base = self.instance["base_commit"]
            validate_repository_base(self.sandbox, base)
            # Remove harness artifacts and all history before exposing any tool.
            # Setup runs only against the pristine image, before agent code exists.
            self.sandbox.checked(["/bin/bash", "-c", "set -eu; rm -rf /testbed/.git; "
                "rm -f /eval.sh /test.patch /patch.diff /tmp/test.patch /root/eval.sh /root/test.patch; "
                "git init --quiet /testbed; cd /testbed; git -c core.hooksPath=/dev/null add -A; "
                "git -c core.hooksPath=/dev/null -c user.name=Offline -c user.email=offline@invalid "
                "commit --quiet --allow-empty --no-gpg-sign -m 'Base tree'"])
            self.temporary = tempfile.TemporaryDirectory(prefix="dyad-swe-workspace-")
            self.workspace = TrustedWorkspace(self.temporary.name, self.config)
            self.workspace.baseline(self.sandbox)
            self.started = time.monotonic()
            self._active = True
            return self._context([])
        except Exception as exc:
            self.close()
            if isinstance(exc, SweBenchInfrastructureError):
                raise
            raise SweBenchInfrastructureError("SWE-bench startup failed") from exc

    boot = start

    def _context(self, delta, **extra):
        if extra.get("format_error"):
            if not self.current_observation.endswith("\nFormat error."):
                self.current_observation += "\nFormat error."
        elif delta:
            self.current_observation = "\n".join(
                f"{message['name']}: {message['content']}" if message.get("name") else message["content"]
                for message in delta)
        observation = json.dumps({"messages": self.messages, "schema_hash": self.schema_hash},
                                 sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return {"protocol": self.protocol, "supports_gigpo": False, "action_tools": deepcopy(self.action_tools),
                "initial_messages": deepcopy(self.initial_messages), "agent_messages": deepcopy(self.messages),
                "messages_delta": deepcopy(delta), "observation": observation, "done": self.done,
                "task_description": self.task_description, "current_observation": self.current_observation,
                "schema_hash": self.schema_hash, "step_count": self.step_count,
                "tool_calls_count": self.tool_calls_count, "format_error": False, "executed": False, **extra}

    def _normalize(self, action):
        from jsonschema import validate
        if not isinstance(action, dict) or action.get("format_error") or action.get("missing_expanded_head"):
            raise ValueError("Invalid assistant message")
        content = action.get("content", action.get("raw_text", "")) or ""
        if not isinstance(content, str):
            raise ValueError("Assistant content must be text")
        raw = action.get("tool_calls") or []
        if not isinstance(raw, list) or not raw or len(raw) > 16:
            raise ValueError("tool_calls must be a nonempty list of at most 16 calls")
        calls, used = [], set()
        schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in self.action_tools}
        for index, call in enumerate(raw):
            fn = call.get("function", call)
            name = fn.get("name")
            arguments = fn.get("arguments", fn.get("parameters", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            validate(arguments, schemas[name])
            if name == "editor" and arguments["operation"] != "read":
                if "text" not in arguments or (arguments["operation"] == "replace" and not arguments.get("old_text")):
                    raise ValueError("Editor write requires text, and replace requires old_text")
            identity = call.get("id", call.get("tool_call_id")) or f"swe_{self.step_count}_{index}"
            if not isinstance(identity, str) or identity in used or identity in self.call_ids:
                raise ValueError("Invalid or duplicate tool call ID")
            used.add(identity)
            calls.append({"id": identity, "name": name, "arguments": arguments})
        if any(call["name"] == "finish" for call in calls[:-1]):
            raise ValueError("finish must be the final call")
        return content, calls

    def step(self, action):
        from jsonschema import ValidationError
        if not self._active or self.done:
            raise RuntimeError("No active nonterminal SWE-bench attempt")
        self.step_count += 1
        if time.monotonic() - self.started >= self.config.episode_timeout_s:
            self.done, self.termination_reason = True, "episode_timeout"
            return self._context([])
        try:
            content, calls = self._normalize(action)
        except (ValueError, KeyError, TypeError, AttributeError, ValidationError):
            feedback = {"role": "user", "content": "Format error."}
            self.messages.append(feedback)
            self.done = self.step_count >= self.config.max_steps
            return self._context([feedback], format_error=True)
        assistant = {"role": "assistant", "content": content}
        if calls:
            assistant["tool_calls"] = [{"id": call["id"], "type": "function", "function": {
                "name": call["name"], "arguments": json.dumps(call["arguments"])}} for call in calls]
        self.messages.append(assistant)
        delta = []
        for call in calls:
            name, arguments = call["name"], call["arguments"]
            self.call_ids.add(call["id"])
            remaining = self.config.episode_timeout_s - (time.monotonic() - self.started)
            if name == "finish":
                self.done, self.termination_reason = True, "finish"
                output = "Submission accepted. Evaluation is performed separately."
            elif remaining <= 0:
                self.done, self.termination_reason = True, "episode_timeout"
                output = "Episode budget exhausted; command not executed."
            elif name == "mini_swe_agent":
                from .mini_agent import run_mini_agent
                run = run_mini_agent(self, arguments["instruction"], self.mini_agent,
                                     Path(self.mini_artifact_dir) / f"call_{len(self.mini_runs) + 1:04d}")
                self.mini_runs.append({key: value for key, value in run.items() if key != "patch"})
                output = json.dumps(run, ensure_ascii=False)
            else:
                timeout = min(self.config.tool_timeout_s, remaining)
                shell = ["/bin/bash", "-c", "if [ -f /opt/miniconda3/bin/activate ]; then source /opt/miniconda3/bin/activate testbed; fi; " + arguments["command"]] if name == "bash" else ["/opt/miniconda3/bin/python", "-c", FILE_TOOL_PROGRAM, name, json.dumps(arguments)]
                command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=2", str(timeout), *shell]
                output = json.dumps(self.sandbox.exec(command, timeout=timeout + 10), ensure_ascii=False)
            feedback = {"role": "tool", "name": name, "tool_call_id": call["id"], "content": output}
            self.messages.append(feedback)
            delta.append(feedback)
            self.tool_calls_count += 1
        if self.step_count >= self.config.max_steps:
            self.done = True
            self.termination_reason = self.termination_reason or "max_steps"
        return self._context(delta, executed=bool(calls))

    def finalize(self, reason="rollout_terminated"):
        if self.result is not None:
            return deepcopy(self.result)
        if not self._active:
            raise RuntimeError("No active SWE-bench attempt")
        self.done = True
        identity = {"benchmark": "swebench_verified", "protocol": self.protocol,
                    "task_id": self.instance["instance_id"], "instance_id": self.instance["instance_id"],
                    "termination_reason": self.termination_reason or reason, "step_count": self.step_count,
                    "tool_calls_count": self.tool_calls_count, "schema_hash": self.schema_hash}
        if self.mini_agent is not None:
            identity.update(mini_swe_agent=deepcopy(self.mini_agent), mini_swe_runs=deepcopy(self.mini_runs),
                            scaffold_version=self.scaffold_version)
        patch = None
        try:
            # Freeze concurrent/background agent writes for a coherent submission.
            self.sandbox.container.pause()
            patch = self.workspace.patch(self.sandbox)
            self.sandbox.close()
            if self.grader is None:
                from .grading import OfficialGrader
                self.grader = OfficialGrader(self.client, self.image, self.config)
            score = self.grader.score(self.instance, patch)
            self.result = {**identity, **score, "patch": patch}
        except Exception as exc:
            self.result = {**identity, "status": "infrastructure_error", "metric_valid": False,
                           "official_scored": False, "official_reward": None, "resolved": None,
                           "error_type": type(exc).__name__}
        finally:
            try:
                self.close()
            except SweBenchInfrastructureError:
                self.result = {**identity, "status": "infrastructure_error", "metric_valid": False,
                               "official_scored": False, "official_reward": None, "resolved": None,
                               "error_type": "ContainerCleanupError"}
        # Submission evidence remains useful when grading or cleanup fails.
        # None means extraction failed; an empty string is a valid empty patch.
        if patch is not None:
            self.result["patch"] = patch
        if self.sandbox is not None and hasattr(self.sandbox, "inspection"):
            inspection = deepcopy(self.sandbox.inspection)
            self.result["agent_container_inspect"] = inspection
            self.result["agent_container_inspect_sha256"] = hashlib.sha256(
                json.dumps(inspection, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return deepcopy(self.result)

    def close(self):
        try:
            if self.sandbox is not None:
                self.sandbox.close()
        finally:
            if self.temporary is not None:
                self.temporary.cleanup()
                self.temporary = None
            self._active = False
        if self._own_client and self.client is not None:
            self.client.close()


# Conventional capitalization is accepted by callers without duplicating behavior.
SWEBenchSession = SweBenchEnv
