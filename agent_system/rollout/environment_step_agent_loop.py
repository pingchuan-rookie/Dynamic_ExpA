"""Shared text/Dyad environment decisions for GRPO and GiGPO.

This loop deliberately has no advantage-estimator branch. Each output contains
only the tokens sampled at one decision, not a slice of an accumulated rollout.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Shared environment decisions for the text baseline and Dyad policies.
# Extension point: AgentLoopManagerTQ -> AgentLoopWorkerTQ -> registered AgentLoopBase implementation
from __future__ import annotations

import asyncio
import copy
import math
import os
from typing import Any
from types import SimpleNamespace
from uuid import uuid4

from agent_system.rollout.prompt import strip_thinking_prefill
from agent_system.environments.prompts.protocol import task_prompt_protocol
from agent_system.environments.step_session import make_step_session
from agent_system.policies.registry import make_step_policy
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op


@register("environment_step_agent")
class EnvironmentStepAgentLoop(AgentLoopBase):
    def __init__(self, *args, tools=None, policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.get("algorithm", {}).get("step_rollout", {})
        evaluation_only = cfg.get("evaluation_only", False)
        if evaluation_only and not self.config.trainer.get("val_only", False):
            raise ValueError("Evaluation-only step prompts require trainer.val_only=true")
        if not cfg.get("enabled", False) and not evaluation_only:
            raise ValueError("environment_step_agent requires step rollout or evaluation-only decisions")
        self.profile = cfg.get("profile", "reference_v1")
        self.protocol_version = cfg.get("protocol_version", 1)
        if self.protocol_version not in {1, 2}:
            raise ValueError("Unsupported environment step protocol version")
        self.history_length = cfg.get("history_length", 2)
        self.max_steps = cfg.get("max_steps", 50)
        if type(self.history_length) is not int or self.history_length < 0:
            raise ValueError("step_rollout.history_length must be a nonnegative integer")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("step_rollout.max_steps must be a positive integer")
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        if self.response_length < 1 or self.prompt_length < 1:
            raise ValueError("Step prompt and response budgets must be positive")
        if self.enable_continuous_token:
            raise ValueError("Independent decisions do not use continuous-token trajectory assembly")
        candidates = tools.tools if tools is not None else []
        if len(candidates) != 1:
            raise ValueError("Step rollout requires exactly one environment tool")
        self.env_tool = candidates[0]
        self.action_interface = cfg.get("action_interface", "text")
        if self.action_interface not in {"text", "dyad"}:
            raise ValueError("step_rollout.action_interface must be text or dyad")
        self.policy = policy if policy is not None else make_step_policy(self.action_interface)

    def _cap_text_prompt_length(self, prompt_ids):
        # Decisions never truncate. run() checks the budget on the sent prompt,
        # after the thinking prefill is removed and SWE may end on context exhaustion.
        return prompt_ids

    def _require_prompt_budget(self, prompt_ids):
        if len(prompt_ids) > self.prompt_length:
            raise ValueError(f"Environment step prompt has {len(prompt_ids)} tokens; limit={self.prompt_length}")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> list[AgentLoopOutput] | AgentLoopOutput:
        group_id = str(kwargs["uid"])
        trajectory_id = f"{group_id}_{kwargs['session_id']}"
        lease_id = uuid4().hex
        tool = self.env_tool
        if kwargs.get("validate") and os.environ.get("DYAD_VAL_TOOL_CONFIG"):
            from verl.tools.tool_registry import initialize_tools_from_config
            tools = initialize_tools_from_config(os.environ["DYAD_VAL_TOOL_CONFIG"])
            if len(tools) != 1:
                raise ValueError("Step validation requires exactly one environment tool")
            tool = tools[0]
        if kwargs.get("tau_padding"):
            # Evaluation padding must not initialize a duplicate official episode.
            return AgentLoopOutput(prompt_ids=[self.tokenizer.eos_token_id], response_ids=[],
                                   response_mask=[], reward_score=0.0, num_turns=0, metrics={},
                                   extra_fields={"tau_padding": True, "episode_result": {}, "raw_prompt": []})
        tool_kwargs = kwargs.get("tools_kwargs") or (kwargs.get("extra_info") or {}).get("tools_kwargs", {})
        settings = copy.deepcopy(tool_kwargs.get(tool.name, {}))
        if getattr(tool, "env_type", None) == "gsm8k_calc" or getattr(tool, "DEFAULT_ENV_TYPE", None) == "gsm8k_calc":
            settings["task_description"] = (kwargs.get("extra_info") or {}).get("question")
        session = make_step_session(tool, lease_id, settings, kwargs.get("raw_prompt", []))
        if session.environment in {"t2bench", "gsm8k"} and not kwargs.get("validate"):
            raise ValueError(f"{session.environment} is evaluation-only")
        reference_environment = session.environment in {"alfworld", "webshop"}
        if self.protocol_version == 2:
            kind = "official" if reference_environment else "native"
            expected_profile = f"{session.environment}_{kind}_v2"
            allowed_profiles = {expected_profile}
        else:
            kind = "official" if reference_environment else "project"
            expected_profile = f"{session.environment}_{kind}_v1"
            allowed_profiles = {"reference_v1", expected_profile}
        # Cross-environment validation deliberately uses the active validation
        # environment's profile, while training retains strict environment identity.
        cross_environment_validation = kwargs.get("validate") and os.environ.get("DYAD_VAL_TOOL_CONFIG")
        if self.profile not in allowed_profiles and not cross_environment_validation:
            raise ValueError(f"Step profile {self.profile!r} does not match {session.environment}")
        params = dict(sampling_params)
        params.pop("max_new_tokens", None)
        params["max_tokens"] = self.response_length
        rows = []
        policy = copy.copy(self.policy)
        try:
            await session.reset()
            if session.done:
                if session.environment == "t2bench":
                    reward, fields = await session.finalize("env_done")
                    return AgentLoopOutput(
                        prompt_ids=[self.tokenizer.eos_token_id], response_ids=[], response_mask=[],
                        reward_score=reward, num_turns=0, metrics={},
                        extra_fields={**fields, "terminal_official": True, "raw_prompt": [],
                                      "task_prompt_protocol": task_prompt_protocol(session.environment)})
                raise ValueError("Environment reset returned a terminal state without a policy decision")
            await policy.prepare(self, session, kwargs)
            # Evaluation trials carry a deterministic policy sampling seed, not just a task label.
            if getattr(session, "context", {}).get("generation_seed") is not None:
                seed = session.context["generation_seed"]
                if type(seed) is not int or not 0 <= seed < 2**31:
                    raise ValueError("Invalid environment policy generation seed")
                params["seed"] = seed
            if getattr(session, "max_tokens", None) is not None:
                params["max_tokens"] = min(params["max_tokens"], session.max_tokens)
            for step_index in range(self.max_steps):
                before = session.anchor
                messages = session.messages(self.history_length)
                prompt_ids = list(await self.apply_chat_template(messages, tools=session.action_tools))
                # Native model thinking mode does not select the environment's ReAct protocol.
                if session.environment in {"alfworld", "webshop", "codegym", "dive", "t2bench", "swebench_verified", "gsm8k"}:
                    prompt_ids = strip_thinking_prefill(prompt_ids, self.tokenizer)
                if session.environment == "swebench_verified" and rows and len(prompt_ids) > self.prompt_length:
                    # A growing repository/tool history can exhaust the decision budget.
                    # Submit the existing workspace, without truncating history or inventing a policy row.
                    reason = "context_budget_exhausted"
                    rows[-1].extra_fields["environment_step"].update(done=True, termination_reason=reason)
                    rows[-1].extra_fields["context_budget"] = {
                        "prompt_tokens": len(prompt_ids), "prompt_limit": self.prompt_length}
                    break
                self._require_prompt_budget(prompt_ids)
                metrics = {}
                with simple_timer("generate_sequences", metrics):
                    generated = await self.server_manager.generate(
                        request_id=f"{lease_id}_{step_index}", prompt_ids=prompt_ids,
                        sampling_params=policy.sampling_params(params, step_index, self.max_steps),
                        priority=int(priority),
                    )
                if generated.stop_reason == "aborted":
                    raise RuntimeError("Environment policy generation was aborted")
                response_ids = list(generated.token_ids)
                if len(response_ids) > self.response_length:
                    raise ValueError("Policy exceeded the per-decision response budget")
                logprobs = None if generated.log_probs is None else list(generated.log_probs)
                if logprobs is not None and len(logprobs) != len(response_ids):
                    raise ValueError("Response logprobs do not align with sampled token IDs")
                if self.rollout_config.get("calculate_log_probs", False):
                    if not response_ids and logprobs is None:
                        logprobs = []
                    if logprobs is None:
                        raise ValueError("Policy omitted requested rollout logprobs")
                else:
                    logprobs = None
                trace_fields, payload = policy.trace(generated)
                selected_action_names = policy.selected_action_names(payload, session.environment)
                raw_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                audit = None
                if getattr(session, "context", {}).get("protocol") in {"dive", "native_tools", "t2bench"}:
                    from agent_system.rollout.env_session import record_action_audit
                    audit_data = SimpleNamespace(
                        runtime_env_tool=tool, runtime_env_context=session.context,
                        assistant_turns=step_index + 1, runtime_generation_max_tokens=params["max_tokens"],
                        extra_fields={},
                    )
                    record_action_audit(self, audit_data, generated, payload, trace_fields or None)
                    audit = audit_data.extra_fields["action_audit"][0]
                    if session.environment == "t2bench":
                        audit.update(prompt_ids=list(prompt_ids), messages=copy.deepcopy(messages))
                # Empty/malformed/full-budget generations are still decisions.
                # The last allowed action is executed before imposing max_steps.
                with simple_timer("tool_calls", metrics):
                    transition = await session.execute(raw_text, response_ids, self.tokenizer,
                                                       selected_action_names)
                if not math.isfinite(transition.reward):
                    raise ValueError("Environment returned a nonfinite transition reward")
                reason = "env_done" if session.done else (
                    "max_steps" if step_index + 1 == self.max_steps else "continue")
                extra = copy.deepcopy(generated.extra_fields or {})
                extra.update({
                    "raw_prompt": messages, "prompt_reward_profile": expected_profile,
                    "task_prompt_protocol": task_prompt_protocol(session.environment),
                    "environment_step": {
                        "schema_version": 1, "protocol_version": self.protocol_version,
                        "group_id": group_id, "trajectory_id": trajectory_id,
                        "step_index": step_index, "anchor": before, "env_reward": transition.reward,
                        "action_valid": transition.valid, "executed": transition.executed,
                        "done": session.done, "termination_reason": reason,
                        # Sampled tokens against this decision's effective budget; reaching it
                        # counts as clipped, as in response_length/clip_ratio.
                        "response_length": len(response_ids), "response_budget": params["max_tokens"],
                        "response_clipped": len(response_ids) >= params["max_tokens"],
                    },
                    "projected_action": transition.action, "generation_stop_reason": generated.stop_reason,
                    **policy.extra_fields,
                })
                if session.environment == "gsm8k" and payload is not None:
                    extra["sampled_action_trace"] = copy.deepcopy(payload)
                if audit is not None:
                    audit.update(submitted=True, submitted_raw_text=transition.action["raw_text"],
                                 selections=transition.action.get("selected_actions"))
                    extra["action_audit"] = [audit]
                metrics["num_preempted"] = generated.num_preempted if generated.num_preempted is not None else -1
                rows.append(AgentLoopOutput(
                    prompt_ids=prompt_ids, response_ids=response_ids, response_mask=[1] * len(response_ids),
                    response_logprobs=logprobs, routed_experts=generated.routed_experts,
                    num_turns=2, metrics=metrics, extra_fields=extra, **trace_fields,
                ))
                if session.done:
                    break
            final_reward, final_fields = await session.finalize(reason)
            if "episode_result" in final_fields:
                final_fields["episode_result"]["action_audit"] = [
                    audit for row in rows for audit in row.extra_fields.get("action_audit", [])]
            rows[-1].extra_fields["environment_step"]["env_reward"] += final_reward
            episode_reward = math.fsum(row.extra_fields["environment_step"]["env_reward"] for row in rows)
            evidence = session.evidence()
            won = bool(evidence.get("won", episode_reward > 0))
            reward_extra_info = {"score": episode_reward, "won": float(won)}
            if session.environment == "webshop":
                # Evaluation retains partial task credit; training uses sparse 10/0 rewards.
                reward_extra_info.update(score=evidence["task_score"], task_score=evidence["task_score"])
            for row in rows:
                row.num_turns = 2 * len(rows)
                row.reward_score = episode_reward
                row.extra_fields["environment_step"].update(episode_reward=episode_reward, episode_length=len(rows))
                row.extra_fields.update(
                    won=won, termination_reason=reason, **final_fields,
                    reward_extra_info=dict(reward_extra_info),
                )
                if kwargs.get("validate"):
                    row.extra_fields["environment_evidence"] = copy.deepcopy(evidence)
            if session.environment == "gsm8k" and kwargs.get("validate"):
                rows[-1].extra_fields["decision_history"] = [
                    {"messages": row.extra_fields["raw_prompt"], "prompt_ids": row.prompt_ids,
                     "response_ids": row.response_ids, "action": row.extra_fields["projected_action"],
                     "sampled_action_trace": row.extra_fields.get("sampled_action_trace")}
                    for row in rows]
                return rows[-1]
            if session.environment == "t2bench":
                # Tau validation consumes one official result per planned episode.
                # All sampled decisions remain in action_audit, not a fictitious
                # concatenated trajectory with prompts the policy never received.
                rows[-1].extra_fields["terminal_official"] = True
                return rows[-1]
            return rows
        finally:
            # Cancellation must not strand a lease. Pool reset failures already
            # discard unknown workers; close remains idempotent for this unique ID.
            await asyncio.shield(session.close())
