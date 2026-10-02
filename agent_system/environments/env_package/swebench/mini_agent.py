"""Pinned mini-swe-agent delegation in the existing isolated task workspace."""
from __future__ import annotations

import ipaddress
import json
import math
import os
import platform
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

MINI_VERSION = "2.4.6"


@dataclass(frozen=True)
class MiniConfig:
    model: str
    base_url: str
    api_key_env: str | None = None
    max_steps: int = 30
    max_tokens: int = 4096
    timeout_s: float = 600
    request_timeout_s: float = 120
    temperature: float = 0

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("mini-swe-agent requires a nonempty model name")
        url = urlsplit(self.base_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("mini-swe-agent base_url must be HTTP(S) without credentials/query/fragment")
        if url.port == 0:
            raise ValueError("mini-swe-agent model endpoint must use a nonzero port")
        # Formal SWE workers have no external network. Do not silently change that protocol.
        try:
            local = ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            local = url.hostname == "localhost"
        if not local:
            raise ValueError("Offline SWE mini-swe-agent requires a loopback model endpoint")
        if self.api_key_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("mini-swe-agent api_key_env must name an environment variable")
        for name in ("max_steps", "max_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"mini-swe-agent {name} must be a positive integer")
        for name in ("timeout_s", "request_timeout_s", "temperature"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"mini-swe-agent {name} must be finite and nonnegative")
        if not self.timeout_s or not self.request_timeout_s or self.temperature > 2:
            raise ValueError("mini-swe-agent requires positive timeouts and temperature <= 2")


def load_config(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("--mini-swe-config must contain a model configuration mapping")
    try:
        return asdict(MiniConfig(**value))
    except TypeError as exc:
        raise ValueError("Invalid mini-swe-agent configuration fields") from exc


def verify_installation():
    if version("mini-swe-agent") != MINI_VERSION:
        raise RuntimeError(f"Install mini-swe-agent=={MINI_VERSION} in the SWE interpreter")
    # mini's import otherwise reads a user-global .env and prints to protocol stdout.
    names = ("MSWEA_GLOBAL_CONFIG_DIR", "MSWEA_SILENT_STARTUP")
    previous = {name: os.environ.get(name) for name in names}
    try:
        with tempfile.TemporaryDirectory(prefix="dyad-mini-config-") as directory:
            os.environ.update(MSWEA_GLOBAL_CONFIG_DIR=directory, MSWEA_SILENT_STARTUP="1")
            import openai  # noqa: F401
            from minisweagent.agents.default import DefaultAgent  # noqa: F401
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return MINI_VERSION


class SandboxEnvironment:
    """mini's Environment protocol; reuse the task container, never create a second workspace."""
    def __init__(self, session, deadline):
        self.session, self.deadline = session, deadline
        self.calls = 0

    def get_template_vars(self):
        return {**platform.uname()._asdict(), "cwd": "/testbed"}

    def serialize(self):
        return {"info": {"environment_type": "dyad_offline_swe_workspace"}}

    def execute(self, action):
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.exceptions import TimeExceeded
        command = action.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("mini-swe-agent command must be nonempty text")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeExceeded({"role": "exit", "content": "TimeExceeded",
                                "extra": {"exit_status": "TimeExceeded", "submission": ""}})
        timeout = min(self.session.config.tool_timeout_s, remaining)
        shell = ["/bin/bash", "-c", "if [ -f /opt/miniconda3/bin/activate ]; then "
                 "source /opt/miniconda3/bin/activate testbed; fi; " + command]
        result = self.session.sandbox.exec(
            ["/usr/bin/timeout", "--signal=TERM", "--kill-after=2", str(timeout), *shell],
            timeout=timeout + 10)
        self.calls += 1
        output = {"output": result["output"], "returncode": result["exit_code"],
                  "exception_info": "Output truncated" if result.get("truncated") else ""}
        DockerEnvironment._check_finished(self, output)
        return output


class ApiModel:
    """OpenAI-compatible adapter using mini's native action parser and observations."""
    def __init__(self, config, deadline):
        import httpx
        from openai import OpenAI
        self.config, self.deadline = config, deadline
        key = os.environ.get(config.api_key_env) if config.api_key_env else "local"
        if not key:
            raise ValueError("mini-swe-agent API key environment variable is unset")
        self.client = OpenAI(base_url=config.base_url, api_key=key, max_retries=0,
                             http_client=httpx.Client(trust_env=False, follow_redirects=False))
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def query(self, messages):
        from minisweagent.exceptions import TimeExceeded
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeExceeded({"role": "exit", "content": "TimeExceeded",
                                "extra": {"exit_status": "TimeExceeded", "submission": ""}})
        try:
            response = self.client.chat.completions.create(
                model=self.config.model, messages=[{k: v for k, v in m.items() if k != "extra"} for m in messages],
                tools=[BASH_TOOL], max_tokens=self.config.max_tokens, temperature=self.config.temperature,
                timeout=min(remaining, self.config.request_timeout_s))
        except Exception:
            # SDK errors can contain credentials, response bodies or private service details.
            raise RuntimeError("mini-swe-agent model request failed") from None
        if response.usage:
            for key in self.usage:
                self.usage[key] += getattr(response.usage, key) or 0
        message = response.choices[0].message
        actions = parse_toolcall_actions(message.tool_calls or [], format_error_template="{{ error }}")
        return {"role": "assistant", "content": message.content,
                "tool_calls": [c.model_dump() for c in message.tool_calls or []],
                "extra": {"actions": actions, "cost": 0.0}}

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages
        return format_toolcall_observation_messages(
            actions=message["extra"]["actions"], outputs=outputs,
            observation_template="{{ output | tojson }}", template_vars=template_vars)

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {"info": {"model": asdict(self.config), "usage": self.usage, "cost_tracking": "unavailable"}}


def run_mini_agent(session, instruction, config, artifact_dir):
    """Run upstream DefaultAgent, save full evidence, return bounded patch feedback."""
    import hashlib
    from importlib.resources import files

    import yaml
    verify_installation()
    from minisweagent.agents.default import DefaultAgent
    cfg = MiniConfig(**config)
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=False)
    deadline = min(session.started + session.config.episode_timeout_s, time.monotonic() + cfg.timeout_s)
    model = ApiModel(cfg, deadline)
    environment = SandboxEnvironment(session, deadline)
    templates = yaml.safe_load(files("minisweagent").joinpath("config/mini.yaml").read_text())["agent"]
    agent = DefaultAgent(model, environment, system_template=templates["system_template"],
                         instance_template=templates["instance_template"],
                         step_limit=cfg.max_steps, cost_limit=0,
                         wall_time_limit_seconds=max(1, math.ceil(deadline - time.monotonic())))
    try:
        result = agent.run(task=session.task_description + "\n\nDelegated instructions:\n" + instruction)
    finally:
        try:
            agent.save(directory / "trajectory.json")
        finally:
            model.client.close()
    # Snapshot a paused workspace, including new files, through the trusted host-side differ.
    session.sandbox.container.pause()
    try:
        patch = session.workspace.patch(session.sandbox)
    finally:
        session.sandbox.container.unpause()
    patch_path = directory / "model.patch"
    patch_path.write_text(patch)
    encoded = patch.encode()
    limit = session.config.max_output_bytes
    evidence = {"mini_swe_agent_version": MINI_VERSION, "config": asdict(cfg),
                "exit_status": result.get("exit_status"), "model_calls": agent.n_calls,
                "command_calls": environment.calls, "usage": model.usage,
                "patch_path": str(patch_path), "patch_sha256": hashlib.sha256(encoded).hexdigest(),
                "patch_bytes": len(encoded), "patch_truncated": len(encoded) > limit,
                "trajectory_path": str(directory / "trajectory.json"),
                "trajectory_sha256": hashlib.sha256((directory / "trajectory.json").read_bytes()).hexdigest(),
                "official_scored": False}
    (directory / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return {**evidence, "patch": encoded[:limit].decode("utf-8", errors="replace"),
            "patch_truncated": len(encoded) > limit}
