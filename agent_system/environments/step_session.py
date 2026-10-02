# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Copyright 2026 Dyad contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Environment contracts for the shared independent-decision collector.

ALFWorld and WebShop use verl-agent reference profiles. DIVE and CodeGym are
project adapters, not environments implemented by the reference repository.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math

from agent_system.parsers.alfworld import project_alfworld_action
from agent_system.environments.prompts.alfworld import build_alfworld_prompt
from agent_system.environments.prompts.dive import build_dive_messages
from agent_system.environments.prompts.webshop import (
    WEBSHOP_TEMPLATE as WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS as WEBSHOP_TEMPLATE_NO_HIS,
    build_webshop_prompt as build_webshop_prompt,
)


def project_webshop_action(text):
    if "<action>" not in text.lower() or "</action>" not in text.lower():
        return text.lower()[-20:], False
    return project_alfworld_action(text)


@dataclass
class Transition:
    reward: float
    valid: bool
    action: object
    executed: bool = True


def check_state(state):
    if not isinstance(state, dict) or not isinstance(state.get("observation"), str) or "done" not in state:
        raise ValueError("Environment step requires a raw observation and done flag")
    if any(state.get(key) for key in ("env_actor_error", "service_error", "timed_out", "http_failure")):
        raise RuntimeError("Environment returned an infrastructure failure")
    if "reward" in state and not math.isfinite(float(state["reward"])):
        raise ValueError("Environment returned a nonfinite reward")


class PoolStepSession:
    """Direct pool lifecycle bypasses legacy tool parsing/reward shaping."""
    def __init__(self, tool, lease_id, settings, initial_messages, environment):
        self.tool, self.lease_id, self.environment = tool, lease_id, environment
        self.settings = copy.deepcopy(settings)
        create = copy.deepcopy(tool._default_create_kwargs)
        create.update(settings.get("create_kwargs", {}))
        if environment == "alfworld" and not any(
            "game" in part for part in (settings.get("create_kwargs", {}),
                                        settings.get("create_kwargs", {}).get("reset_payload", {}) or {})
        ):
            raise ValueError("Step rollout requires an explicit dataset ALFWorld game index")
        self.reset_spec = tool._build_reset_spec(create, settings)
        self.initial_messages = copy.deepcopy(list(initial_messages))
        self.state = None
        self.task = ""
        self.history = []
        self.turns = []
        self.context = {}

    async def reset(self):
        self.state = await self.tool.pool.create_session(self.lease_id, self.reset_spec)
        check_state(self.state)
        if "reward" not in self.state:
            raise ValueError("Environment reset omitted raw reward")
        if self.environment == "alfworld":
            marker = "Your task is to: "
            if marker not in self.state["observation"]:
                raise ValueError("Task description not found in ALFWorld reset observation")
            self.task = self.state["observation"].split(marker, 1)[1].strip()
        elif self.environment == "webshop":
            self.task = self.state.get("instruction", "")
            if not self.task:
                parts = self.state["observation"].split(" [SEP] ")
                if len(parts) < 3 or parts[1] != "Instruction:":
                    raise ValueError("WebShop reset omitted its task instruction")
                self.task = parts[2]

    @property
    def done(self):
        return bool(self.state["done"])

    @property
    def anchor(self):
        observation = self.state["observation"]
        if self.environment == "webshop":
            parts = observation.split(" [SEP] ")
            if self.task in parts:
                return " [SEP] ".join(f"'{part}'" for part in parts[parts.index(self.task) + 1:])
        return observation

    @property
    def action_tools(self):
        return None

    def messages(self, history_length):
        if self.environment == "codegym":
            from agent_system.environments.prompts.codegym import build_codegym_step_messages

            # The live observation replaces any cached dataset observation.
            return build_codegym_step_messages(
                self.initial_messages, self.anchor, self.history, history_length)
        actions = self.state.get("available_actions")
        if self.environment == "webshop" and isinstance(actions, dict):
            if set(actions) - {"has_search_bar", "clickables"}:
                raise ValueError("Unknown WebShop available-action field")
            actions = (["search[<your query>]"] if actions.get("has_search_bar") else []) + [
                f"click[{target}]" for target in actions.get("clickables", [])]
        if not isinstance(actions, list) or not all(isinstance(action, str) for action in actions):
            raise ValueError("Environment omitted its available action list")
        build = build_alfworld_prompt if self.environment == "alfworld" else build_webshop_prompt
        history = self.history if history_length else []
        return [{"role": "user", "content": build(self.task, self.anchor, actions, history,
                                                   history_length)}]

    async def execute(self, raw_text, response_ids, tokenizer, selected_action_names):
        before = self.anchor
        if self.environment == "codegym":
            from agent_system.parsers.codegym_action import prepare_codegym_action
            action, error = prepare_codegym_action(raw_text)
            reward = 0.0
            actions = [] if error else [action]
            valid = error is None
            if error:
                # Match the official client's JSON error, without touching the
                # environment or manufacturing a generic retry instruction.
                self.state = {**self.state, "observation": error}
            else:
                # Preserve missing fields, null parameters and multi-call lists:
                # env.step must reject them itself, not execute a repaired call.
                self.state = await self.tool.pool.step_session(self.lease_id, action)
                check_state(self.state)
                reward = float(self.state["reward"])
                try:
                    call = json.loads(action)
                    valid = (isinstance(call, dict) and isinstance(call.get("name"), str)
                             and isinstance(call.get("parameters"), dict))
                except (ValueError, TypeError):
                    valid = False
            turn = [{"role": "user", "content": before}, {"role": "assistant", "content": raw_text}]
            turn.append({"role": "user", "content": self.anchor})
            self.turns.append(turn)
            self.history.append((before, raw_text))
            return Transition(reward, valid, actions, executed=bool(actions))
        projection = project_alfworld_action if self.environment == "alfworld" else project_webshop_action
        action, valid = projection(raw_text)
        self.state = await self.tool.pool.step_session(self.lease_id, action)
        check_state(self.state)
        if self.environment == "alfworld":
            reward = float(self.state["reward"]) * 10.0
        else:
            score = float(self.state.get("task_score", self.state["reward"]))
            if not 0 <= score <= 1:
                raise ValueError("WebShop task score must be in [0, 1]")
            reward = 10.0 if self.done and score == 1 else 0.0
        self.history.append((before, action))
        return Transition(reward, valid, action)

    async def finalize(self, reason):
        return 0.0, {}

    def evidence(self):
        score = float(self.state.get("task_score", self.state.get("won", False)))
        won = self.done and score == 1 if self.environment == "webshop" else bool(self.state.get("won", False))
        create = self.settings.get("create_kwargs", {})
        task_id = create.get("task_id", self.reset_spec.get("task_id", self.reset_spec.get("game_idx", self.reset_spec.get("env_str", ""))))
        return {"benchmark": self.environment, "task_id": str(task_id), "metric_valid": True,
                "service_error": False, "won": won, "task_score": score}

    async def close(self):
        await self.tool.pool.close_session(self.lease_id)


class Gsm8kStepSession(PoolStepSession):
    """Evaluation-only calculator decisions with the answer confined to reset state."""
    def __init__(self, tool, lease_id, settings, initial_messages):
        super().__init__(tool, lease_id, settings, initial_messages, "gsm8k")
        self.task = settings.get("task_description")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("GSM8K evaluation requires the public dataset question")

    def messages(self, history_length):
        from agent_system.environments.prompts.gsm8k import build_gsm8k_messages
        return build_gsm8k_messages(self.task, self.anchor, self.history, history_length)

    async def execute(self, raw_text, response_ids, tokenizer, selected_action_names):
        from agent_system.parsers.react import ReactToolParser
        before = self.anchor
        _, calls = await ReactToolParser(tokenizer).extract_tool_calls(response_ids)
        valid = len(calls) == 1
        action = self.tool.adapter.build_action(json.loads(calls[0].arguments)) if valid else ""
        if valid:
            self.state = await self.tool.pool.step_session(self.lease_id, action)
            check_state(self.state)
            reward = float(self.state["reward"])
        else:
            reward = 0.0
            self.state = {**self.state, "observation": before.removesuffix("\nFormat error.") + "\nFormat error."}
        self.history.append((before, raw_text))
        return Transition(reward, valid, action, executed=valid)


class NativeToolStepSession:
    environment = "native_tools"

    def __init__(self, tool, lease_id, settings, initial_messages):
        self.environment = getattr(tool, "environment", self.environment)
        self.tool, self.lease_id, self.settings = tool, lease_id, copy.deepcopy(settings)
        self.turns = []
        self.context = {}
        self.result = None
        session_config = self.settings.get("create_kwargs", {}).get("create_payload", {}).get("session_config", {})
        self.max_tokens = session_config.get("max_tokens")
        if self.max_tokens is not None and (type(self.max_tokens) is not int or self.max_tokens < 1):
            raise ValueError("Native-tool per-decision max_tokens must be a positive integer")

    async def reset(self):
        await self.tool.create(instance_id=self.lease_id, **self.settings)
        self.context = self.tool.get_session_context(self.lease_id)
        check_state(self.context)
        self.initial_messages = copy.deepcopy(self.context["initial_messages"])

    @property
    def anchor(self):
        return self.context["observation"]

    @property
    def done(self):
        return bool(self.context["done"])

    @property
    def action_tools(self):
        return self.context["action_tools"]

    def messages(self, history_length):
        messages = copy.deepcopy(self.initial_messages)
        recent = self.turns[-history_length:] if history_length else []
        for turn in recent:
            messages.extend(copy.deepcopy(turn))
        # Tool replies stay paired with their sampled assistant call, never orphaned.
        if self.turns and not recent:
            messages.append({"role": "user", "content": self.anchor})
        return messages

    async def execute(self, raw_text, response_ids, tokenizer, selected_action_names):
        from agent_system.parsers.native_tools import NativeToolFormatError, decode_response, native_protocol, parse_response
        raw_text = decode_response(tokenizer, response_ids)
        try:
            action = parse_response(raw_text, self.action_tools, native_protocol(tokenizer),
                                    call_prefix=f"{self.lease_id}_{len(self.turns)}")
            if selected_action_names is not None:
                names = list(selected_action_names)
                if names != [call["name"] for call in action["tool_calls"]]:
                    raise NativeToolFormatError("Tool calls do not match verified policy selections")
                action["selected_actions"] = names
        except NativeToolFormatError as exc:
            action = {"raw_text": raw_text, "content": "", "tool_calls": [], "format_error": str(exc)}
        await self.tool.execute(self.lease_id, action)
        context = self.tool.get_session_context(self.lease_id)
        check_state(context)
        if context["schema_hash"] != self.context["schema_hash"]:
            raise ValueError("Native-tool environment changed action schema within one trajectory")
        self.context = context
        delta = copy.deepcopy(context.get("messages_delta", []))
        if not self.done and not delta:
            raise ValueError("Nonterminal native-tool decision returned no tool feedback")
        self.turns.append([{"role": "assistant", "content": raw_text}, *delta])
        valid = not bool(action.get("format_error") or context.get("format_error"))
        return Transition(0.0, valid, action, executed=valid)

    async def finalize(self, reason):
        result = await self.tool.finalize(self.lease_id, reason)
        if not result or not result.get("metric_valid"):
            raise RuntimeError("Native-tool environment returned no valid terminal result")
        reward = result.get("official_reward") if result.get("official_scored") else result.get("attempt_reward")
        if reward is None or not math.isfinite(float(reward)):
            raise ValueError("Native-tool environment returned no finite terminal reward")
        self.result = result
        return float(reward), {"episode_result": copy.deepcopy(result), "runtime_protocol": self.context.get("protocol", "native_tools")}

    def evidence(self):
        return {"benchmark": self.environment, **copy.deepcopy(self.result or {})}

    async def close(self):
        await self.tool.release(self.lease_id)


class DiveStepSession(NativeToolStepSession):
    environment = "dive"

    async def reset(self):
        await super().reset()
        self.prompt_history = []

    def messages(self, history_length):
        if self.context.get("custom_initial_messages", False):
            return super().messages(history_length)
        return build_dive_messages(
            self.context["task_description"], self.action_tools,
            observation=self.context["current_observation"],
            history=self.prompt_history, history_length=history_length)

    async def execute(self, raw_text, response_ids, tokenizer, selected_action_names):
        before = self.context["current_observation"]
        transition = await super().execute(raw_text, response_ids, tokenizer, selected_action_names)
        self.prompt_history.append((before, self.turns[-1][0]["content"]))
        return transition


class SwebenchStepSession(NativeToolStepSession):
    environment = "swebench_verified"

    async def reset(self):
        await super().reset()
        self.prompt_history = []

    def messages(self, history_length):
        from agent_system.environments.prompts.swebench import build_swebench_messages
        return build_swebench_messages(
            self.context["task_description"], self.action_tools,
            observation=self.context["current_observation"],
            history=self.prompt_history, history_length=history_length,
            system_prompt=self.initial_messages[0]["content"])

    async def execute(self, raw_text, response_ids, tokenizer, selected_action_names):
        before = self.context["current_observation"]
        transition = await super().execute(raw_text, response_ids, tokenizer, selected_action_names)
        self.prompt_history.append((before, self.turns[-1][0]["content"]))
        return transition


class TauStepSession(NativeToolStepSession):
    """Evaluation-only text actions, with the official customer/tool transcript retained."""
    environment = "t2bench"

    async def reset(self):
        await self.tool.create(instance_id=self.lease_id, **self.settings)
        self.context = self.tool.get_session_context(self.lease_id)
        self.initial_messages = copy.deepcopy(self.context["initial_messages"])
        self.step_count = 0

    @property
    def anchor(self):
        return json.dumps(self.context["agent_messages"], ensure_ascii=False, sort_keys=True)

    @property
    def action_tools(self):
        # Schemas live in the text system prompt, not the native chat tool interface.
        return None

    def messages(self, history_length):
        from agent_system.environments.prompts.t2bench import build_t2bench_messages
        tools = [tool for tool in self.context["action_tools"] if tool["function"]["name"] != "respond"]
        return build_t2bench_messages(
            self.context["domain_policy"], tools,
            [m for m in self.context["agent_messages"] if m["role"] != "system"],
            history_length, task_description=self.context["task_description"])

    async def execute(self, raw_text, response_ids, tokenizer, selected_action_names):
        action = {"raw_text": raw_text}
        if selected_action_names is not None:
            if len(selected_action_names) != 1:
                action["missing_expanded_head"] = True
            else:
                action["selected_action"] = selected_action_names[0]
        await self.tool.execute(self.lease_id, action)
        context = self.tool.get_session_context(self.lease_id)
        if context["schema_hash"] != self.context["schema_hash"]:
            raise ValueError("t2bench changed action schema within one trajectory")
        self.context = context
        self.step_count += 1
        if not self.done and not context.get("messages_delta"):
            raise ValueError("Nonterminal t2bench decision returned no feedback")
        valid = not bool(context.get("format_error"))
        return Transition(0.0, valid, action, executed=valid)

    async def finalize(self, reason):
        # The shared collector counts policy decisions, unlike official orchestration steps.
        result = await self.tool.finalize(self.lease_id, "max_assistant_turns" if reason == "max_steps" else reason)
        if not isinstance(result, dict) or "metric_valid" not in result:
            raise ValueError("t2bench returned no terminal evaluation result")
        reward = result.get("official_reward") if result.get("official_scored") else result.get("attempt_reward")
        if result["metric_valid"] and (reward is None or not math.isfinite(float(reward))):
            raise ValueError("t2bench returned no finite reward for a scored attempt")
        self.result = result
        # Unscored failures keep null rewards in episode_result; the tensor value is
        # only transport padding and is never used by the official tau metrics.
        return float(reward) if result["metric_valid"] else 0.0, {
            "episode_result": copy.deepcopy(result), "runtime_protocol": "t2bench"}


def make_step_session(tool, lease_id, settings, initial_messages):
    environment = getattr(tool, "protocol", None) or getattr(tool, "env_type", None) or getattr(tool, "DEFAULT_ENV_TYPE", None)
    if environment == "gsm8k_calc":
        return Gsm8kStepSession(tool, lease_id, settings, initial_messages)
    if environment == "t2bench":
        return TauStepSession(tool, lease_id, settings, initial_messages)
    if environment == "native_tools":
        if getattr(tool, "environment", None) == "swebench_verified":
            return SwebenchStepSession(tool, lease_id, settings, initial_messages)
        return NativeToolStepSession(tool, lease_id, settings, initial_messages)
    if environment == "dive":
        return DiveStepSession(tool, lease_id, settings, initial_messages)
    if environment in {"alfworld", "webshop", "codegym"}:
        return PoolStepSession(tool, lease_id, settings, initial_messages, environment)
    raise ValueError(f"Unsupported training step environment: {environment!r}")
