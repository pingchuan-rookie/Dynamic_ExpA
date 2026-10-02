"""One DIVE task, its public interaction history and private terminal judgement."""
from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy

from agent_system.environments.prompts.dive import build_dive_messages

from .runtime import DiveInfrastructureError, DiveToolRuntime


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class DiveEnv:
    protocol = "dive"
    supports_gigpo = True

    def __init__(self, repo=None, tool_timeout_s=60, episode_timeout_s=1800,
                 max_steps=30, sandbox_url=None, judge_config=None, runtime=None, judge=None, **kwargs):
        self.runtime = runtime or DiveToolRuntime(repo, tool_timeout_s, sandbox_url)
        self.repo = repo
        self.judge = judge
        self._model_client = None
        self.judge_config = dict(judge_config or {})
        self.episode_timeout_s = float(episode_timeout_s)
        self.max_steps = int(max_steps)
        if not math.isfinite(self.episode_timeout_s) or self.episode_timeout_s <= 0 or self.max_steps <= 0:
            raise ValueError("DIVE episode timeout and max_steps must be positive")
        self._active = False

    def reset(self, task=None, **spec):
        if self._active:
            raise RuntimeError("Close a DIVE session before resetting")
        task = deepcopy(task or spec.get("task") or spec)
        for key in ("trace_id", "query", "answer", "tools"):
            if not task.get(key):
                raise ValueError(f"DIVE task missing {key}")
        if task["query"].startswith("Failed to evolve") or task["answer"].startswith("Failed to evolve"):
            raise ValueError("DIVE generation-failure placeholder is not a task")
        self.runtime.validate_tools(task["tools"])
        self.task = task
        self.tools = deepcopy(task["tools"])
        self.by_name = {t["function"]["name"]: t["function"] for t in self.tools}
        self.schema_hash = hashlib.sha256(canonical(self.tools).encode()).hexdigest()
        # The supplied prompt is algorithm-neutral; never synthesize it from private fields.
        custom_messages = spec.get("initial_messages") or task.get("initial_messages")
        self.custom_initial_messages = bool(custom_messages)
        self.current_observation = "No tool results yet."
        self.initial_messages = deepcopy(custom_messages or build_dive_messages(task["query"], self.tools))
        self.messages = deepcopy(self.initial_messages)
        self.step_count = 0
        self.tool_calls_count = 0
        self.answer = None
        self.result = None
        self.done = False
        self.started = time.monotonic()
        self._active = True
        self.call_ids = set()
        self.spec = {k: deepcopy(v) for k, v in spec.items() if k in (
            "split", "trial", "seed", "episode_id", "source_revision", "source_line",
            "source_sha256", "task_sha256",
        )}
        return self._response([])

    def _response(self, delta, **extra):
        if delta:
            self.current_observation = "\n".join(
                f"{message['name']}: {message['content']}" if message.get("name") else message["content"]
                for message in delta)
        # IDs bind responses to calls but request-generated UUIDs are not policy state.
        # Canonicalize them by occurrence so identical public histories can share GiGPO anchors.
        public = deepcopy(self.messages)
        identities = {}
        for message in public:
            if "tool_call_id" in message:
                original = message["tool_call_id"]
                message["tool_call_id"] = identities.setdefault(original, f"call_{len(identities)}")
        state = canonical({"messages": public, "schema_hash": self.schema_hash})
        return {"protocol": "dive", "supports_gigpo": True, "initial_messages": deepcopy(self.initial_messages),
                "task_description": self.task["query"], "current_observation": self.current_observation,
                "custom_initial_messages": self.custom_initial_messages,
                "action_tools": deepcopy(self.tools), "schema_hash": self.schema_hash,
                "agent_messages": deepcopy(self.messages), "messages_delta": deepcopy(delta),
                "observation": state, "gigpo_anchor": state, "step_count": self.step_count,
                "tool_calls_count": self.tool_calls_count, "done": self.done,
                "result": deepcopy(self.result), "episode_result": deepcopy(self.result),
                "format_error": False, "executed": False, **extra}

    @staticmethod
    def _calls(action):
        calls = action.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        normalized = []
        for i, call in enumerate(calls):
            if not isinstance(call, dict):
                raise ValueError("Tool call must be an object")
            fn = call.get("function", call)
            name = fn.get("name")
            arguments = fn.get("arguments", fn.get("parameters", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("Tool call requires name and object arguments")
            normalized.append({"id": call.get("id", call.get("tool_call_id")), "name": name,
                               "arguments": arguments})
        return normalized

    def step(self, action):
        if not self._active or self.done:
            raise RuntimeError("No active nonterminal DIVE session")
        if time.monotonic() - self.started >= self.episode_timeout_s:
            raise DiveInfrastructureError("DIVE episode deadline exceeded")
        if not isinstance(action, dict):
            raise ValueError("DIVE requires an assistant message object")
        self.step_count += 1
        raw_text = action.get("raw_text", action.get("content", ""))
        if not isinstance(raw_text, str):
            raise ValueError("Assistant content must be text")
        assistant = {"role": "assistant", "content": raw_text}
        try:
            if action.get("format_error") or action.get("missing_expanded_head"):
                raise ValueError("Invalid policy action")
            calls = self._calls(action)
            answer = action.get("content", raw_text)
            if not calls and (not isinstance(answer, str) or not answer.strip()):
                raise ValueError("Empty assistant response")
            from jsonschema.validators import validator_for
            used = set()
            for call in calls:
                name = call["name"]
                if name not in self.by_name:
                    raise ValueError("Tool outside task schema")
                schema = self.by_name[name]["parameters"]
                validator_for(schema)(schema).validate(call["arguments"])
                call_id = call["id"] or f"dive_{self.step_count}_{len(used)}"
                if not isinstance(call_id, str) or call_id in used or call_id in self.call_ids:
                    raise ValueError("Duplicate or invalid tool call ID")
                call["id"] = call_id
                used.add(call_id)
        except Exception:
            # Only validation runs in this block; execution errors must not become format errors.
            self.messages.append(assistant)
            feedback = {"role": "user", "content": "Format error."}
            self.messages.append(feedback)
            self.done = self.step_count >= self.max_steps
            return self._response([feedback], format_error=True, executed=False)
        self.messages.append(assistant)
        if not calls:
            self.answer = action.get("content", raw_text)
            self.done = True
            return self._response([], executed=True)
        delta = []
        for call in calls:
            if time.monotonic() - self.started >= self.episode_timeout_s:
                raise DiveInfrastructureError("DIVE episode deadline exceeded")
            self.call_ids.add(call["id"])
            output = self.runtime.execute(call["name"], call["arguments"])
            if time.monotonic() - self.started >= self.episode_timeout_s:
                raise DiveInfrastructureError("DIVE episode deadline exceeded")
            message = {"role": "tool", "name": call["name"], "tool_call_id": call["id"], "content": output}
            self.messages.append(message)
            delta.append(message)
            self.tool_calls_count += 1
        self.done = self.step_count >= self.max_steps
        return self._response(delta, executed=True)

    def finalize(self, reason="rollout_terminated"):
        if not self._active:
            raise RuntimeError("No active DIVE session")
        if self.result is None:
            judgement = None
            if self.answer is not None:
                if self.judge is None:
                    from .judge import DiveJudge
                    from .judge_config import resolve_judge_config
                    from .model_api import ModelConfig, create_model_client

                    config = resolve_judge_config(self.judge_config)
                    client = create_model_client(
                        ModelConfig(provider=config["provider"], model=config["model"],
                                    base_url=config["base_url"], api_key_env=config.get("api_key_env")),
                        timeout_s=config["timeout_s"], max_retries=0,
                    )
                    try:
                        self.judge = DiveJudge(config, self.repo, model_client=client)
                    except BaseException:
                        client.close()
                        raise
                    self._model_client = client
                judgement = self.judge.score(self.task["query"], self.task["answer"], self.answer)
            reward = float(judgement["reward"]) if judgement is not None else 0.0
            self.result = {"benchmark": "dive", "protocol": "dive", "task_id": self.task["trace_id"],
                           "trace_id": self.task["trace_id"], "domain": self.task.get("metadata", {}).get("domain"),
                           **self.spec, "status": "completed", "metric_valid": True,
                           "official_scored": judgement is not None, "official_reward": reward,
                           "attempt_reward": reward, "termination_reason": "final_answer" if self.answer is not None else reason,
                           "final_answer": self.answer, "verification": judgement,
                           "steps": self.step_count, "tool_calls_count": self.tool_calls_count,
                           "reward_rule": "strict_correct", "schema_hash": self.schema_hash}
            self.done = True
        return self._response([])

    def close(self):
        if self._model_client is not None:
            self._model_client.close()
            self._model_client = None
            self.judge = None
        self._active = False
        self.task = None
        self.messages = []
        self.answer = None
        return {"ok": True}
