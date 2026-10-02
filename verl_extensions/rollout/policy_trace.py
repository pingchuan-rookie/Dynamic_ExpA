# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


"""Validate and serialize the sampled action distribution without changing candidate identities.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def serialize_policy_trace(self, output: dict):
    # Serialize exact action masks and labels; keep empty decisions valid without inventing token rewards.
    from verl.experimental.agent_loop.agent_loop import (
        torch,
    )

    trace_fields = ("response_dyad", "seq_mask", "tool_mask", "dyad_allowed_action_ids")
    if any(output.get(key) is not None for key in trace_fields):
        if any(output.get(key) is None for key in trace_fields):
            raise ValueError("Incomplete Dyad policy trace in AgentLoopOutput")
        length = len(self.response_ids)
        if len(self.response_mask) != length or any(len(output[key]) != length for key in trace_fields):
            raise ValueError("Dyad policy trace lengths must match the response")
        capacity = self.dyad_action_size
        if capacity <= 0:
            raise ValueError("Dyad action capacity must be positive")
        for key in ("seq_mask", "tool_mask"):
            if any(value not in (0, 1) for value in output[key]):
                raise ValueError(f"Dyad {key} must contain only boolean values")
        allowed = output.pop("dyad_allowed_action_ids")
        action_mask = torch.zeros((length, capacity), dtype=torch.bool)
        for position, ids in enumerate(allowed):
            if any(index < 0 or index >= capacity for index in ids):
                raise ValueError("Dyad candidate index exceeds the sampling action capacity")
            if bool(ids) != bool(output["tool_mask"][position]):
                raise ValueError("Dyad candidates must exist exactly at action positions")
            action_mask[position, ids] = True
        output["response_dyad"] = torch.tensor(output["response_dyad"], dtype=torch.int64)
        output["seq_mask"] = torch.tensor(output["seq_mask"], dtype=torch.bool)
        output["tool_mask"] = torch.tensor(output["tool_mask"], dtype=torch.bool)
        output["dyad_action_mask"] = action_mask
    else:
        for key in trace_fields:
            output.pop(key, None)
    # Contexts stay in diagnostics too, but replay consumes per-row top-level fields.
    extras = output.setdefault("extra_fields", {})
    for key in ("dyad_action_context", "codegym_context", "dyad_action_context_identity"):
        if key in extras:
            output[key] = extras[key]
    if (
        output.get("dyad_action_context") is not None
        and output.get("codegym_context") is not None
        and output["dyad_action_context"] != output["codegym_context"]
    ):
        raise ValueError("Dyad sampling context aliases disagree")


def pad_policy_trace(self, output):
    # Keep authoritative sampled traces and environment metadata aligned with response tensors.
    from verl.experimental.agent_loop.agent_loop import (
        torch,
    )

    response_dyad = seq_mask = tool_mask = dyad_action_mask = None
    if output.response_dyad is not None:
        if output.seq_mask is None or output.tool_mask is None or output.dyad_allowed_action_ids is None:
            raise ValueError("Incomplete Dyad policy trace in AgentLoopOutput.")
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        actual_length = min(len(output.response_dyad), max_response_length)
        pad_length = max_response_length - actual_length
        response_dyad = torch.tensor(
            output.response_dyad[:actual_length] + [self.tokenizer.pad_token_id] * pad_length,
            dtype=torch.long,
        ).unsqueeze(0)
        seq_mask = torch.tensor(output.seq_mask[:actual_length] + [False] * pad_length, dtype=torch.bool).unsqueeze(0)
        tool_mask = torch.tensor(output.tool_mask[:actual_length] + [False] * pad_length, dtype=torch.bool).unsqueeze(0)
        dyad_action_mask = torch.zeros(1, max_response_length, output.dyad_action_size, dtype=torch.bool)
        for position, allowed_ids in enumerate(output.dyad_allowed_action_ids[:actual_length]):
            if allowed_ids:
                dyad_action_mask[0, position, allowed_ids] = True
    return response_dyad, seq_mask, tool_mask, dyad_action_mask


def merge_policy_traces(inputs, optional_outputs: dict):
    # Merge complete action traces and retain their field inventory through shared batch assembly.
    from verl.experimental.agent_loop.agent_loop import (
        log_event,
        torch,
    )

    dyad_present = [inp.response_dyad is not None for inp in inputs]
    if any(dyad_present) and not all(dyad_present):
        raise ValueError("Mixed Dyad and non-Dyad samples in one rollout batch.")
    # Record missing Dyad fields at their source for downstream batch-error diagnosis.
    if not any(dyad_present):
        log_event(
            "agent_loop",
            "postprocess_no_dyad_trace",
            num_inputs=len(inputs),
            agent_names=sorted({type(inp).__name__ for inp in inputs}),
            hint="rollout 侧没有产出 Dyad 策略轨迹；检查 DyadToolAgentLoop.run 的返回路径",
        )
    if all(dyad_present) and dyad_present:
        optional_outputs.update(
            response_dyad=torch.cat([inp.response_dyad for inp in inputs], dim=0),
            seq_mask=torch.cat([inp.seq_mask for inp in inputs], dim=0),
            tool_mask=torch.cat([inp.tool_mask for inp in inputs], dim=0),
            dyad_action_mask=torch.cat([inp.dyad_action_mask for inp in inputs], dim=0),
        )


def pad_transport_fields(template_sample, prompts):
    # DYAD-STEP-TQ: the optimizer distinguishes transport dummies from real copies.
    from verl.trainer.ppo.padding_utils import (
        torch,
    )

    template_sample["environment_step_is_padding"] = torch.tensor(True, dtype=torch.bool)
    # DYAD-STEP-TQ: copied action metadata must not retain the source response shape.
    if "response_dyad" in template_sample:
        template_sample["response_dyad"] = prompts.clone()
        for key in ("seq_mask", "tool_mask"):
            template_sample[key] = torch.zeros(1, dtype=torch.bool)
        source_mask = template_sample.get("dyad_action_mask")
        if not isinstance(source_mask, torch.Tensor) or source_mask.ndim != 2:
            raise ValueError("Dyad padding requires a response-by-action candidate mask")
        template_sample["dyad_action_mask"] = source_mask.new_zeros((1, source_mask.shape[-1]))
        if "dyad_allowed_action_ids" in template_sample:
            template_sample["dyad_allowed_action_ids"] = [[]]
