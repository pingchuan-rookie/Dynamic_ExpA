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


"""Shared environment sessions, native tool protocols and task evidence.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl.experimental.agent_loop.tool_agent_loop import (
        AgentData,
        AgentLoopOutput,
        AgentState,
        Any,
        FunctionCall,
        ToolResponse,
    )


async def tool_run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
    from verl.experimental.agent_loop.tool_agent_loop import (
        AgentData,
        AgentLoopOutput,
        AgentState,
        _grpo_diag_enabled,
        _grpo_diag_full,
        log_event,
        log_run_meta,
        logger,
        uuid4,
    )

    messages = list(kwargs["raw_prompt"])

    # extract multimodal inputs from messages
    multi_modal_data = await self.process_multi_modal_info(messages)
    images = multi_modal_data.get("images")
    videos = multi_modal_data.get("videos")
    audios = multi_modal_data.get("audios")
    mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

    metrics = {}
    request_id = uuid4().hex
    tools_kwargs = kwargs.get("tools_kwargs", {})

    agent_data = AgentData(
        messages=messages,
        image_data=images,
        video_data=videos,
        audio_data=audios,
        mm_processor_kwargs=mm_processor_kwargs,
        metrics=metrics,
        request_id=request_id,
        tools_kwargs=tools_kwargs,
        # DYAD-ADDED: Initialize environment sessions with dataset metadata.
        extra_info=kwargs.get("extra_info", {}) or {},
    )
    # DYAD-ADDED(diag): Read the same validation flag as the Dyad loop.
    agent_data.is_validate = bool(kwargs.get("validate", False))
    # DYAD-EVAL: task evidence follows actual reset IDs and infrastructure outcomes.
    from agent_system.rollout.env_session import initialize_task_evidence

    initialize_task_evidence(agent_data)

    # Per-sample tool selection: filter global tools by extra_info.tool_selection
    extra_info = kwargs.get("extra_info", {}) or {}
    tool_selection = extra_info.get("tool_selection")
    if tool_selection and self.tools:
        selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
        agent_data._active_tools = selected
        agent_data._active_tool_schemas = [
            t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
        ]
    else:
        agent_data._active_tools = self.tools
        agent_data._active_tool_schemas = self.tool_schemas

    # DYAD-GIGPO: validation needs no groups or step metadata, including tau evaluation.
    config = getattr(self, "config", None)
    algorithm = getattr(config, "algorithm", None)
    agent_data.gigpo_enabled = getattr(algorithm, "adv_estimator", None) == "gigpo" and not agent_data.is_validate
    if agent_data.gigpo_enabled:
        active = agent_data._active_tools
        if self.enable_continuous_token:
            raise ValueError("GiGPO training does not support continuous-token rollout")
        if len(active) != 1:
            raise ValueError("GiGPO training requires exactly one local environment tool")
        tool = next(iter(active.values()))
        # DYAD-DIVE: only sessions with explicit state/reward metadata support GiGPO.
        if getattr(tool, "runtime_session", False) and not getattr(tool, "supports_gigpo", False):
            raise ValueError("GiGPO training does not support this runtime environment session")
        if not callable(getattr(tool, "get_observation", None)):
            raise ValueError("GiGPO training requires raw observation metadata from a local environment tool")
        if getattr(tool, "reward_mode", None) != "last":
            raise ValueError("GiGPO training requires reward_mode=last")

    # DYAD: release sessions even when bootstrap or generation fails.
    try:
        # DYAD: reset runtime environments before the first model decision.
        from agent_system.rollout.env_session import bootstrap_session, finalize_session

        await bootstrap_session(self, agent_data)
        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        if agent_data.extra_info.get("need_tools_kwargs") and _grpo_diag_enabled():
            from agent_system.utils.diagnostics import dump_limit as _dyad_dump_limit

            _dump_limit = _dyad_dump_limit("GRPO_FULL_DUMP_LIMIT")
            type(self)._grpo_full_dump_count = getattr(type(self), "_grpo_full_dump_count", 0)
            _full_token_ids = None
            _full_text = None
            _response_mask_str = None
            _dump_seq = None
            # Bound full token dumps with DIAG_DUMP_LIMIT.
            if _grpo_diag_full() and type(self)._grpo_full_dump_count < _dump_limit:
                log_run_meta(
                    "grpo_tool_agent",
                    self.tokenizer,
                    path="grpo_react",
                    tool_parser=self.tool_parser_name,
                    response_length_limit=self.response_length,
                )
                # A comma-separated string avoids diagnostic list truncation during token-level mask checks.
                _full_token_ids = ",".join(str(int(t)) for t in agent_data.prompt_ids)
                _response_mask_str = ",".join(str(int(m)) for m in agent_data.response_mask)
                try:
                    _full_text = self.tokenizer.decode(agent_data.prompt_ids, skip_special_tokens=False)
                except Exception:  # noqa: BLE001 - Diagnostic decoding must not abort rollout.
                    _full_text = None
                type(self)._grpo_full_dump_count += 1
                _dump_seq = type(self)._grpo_full_dump_count - 1

            _turn_scores = agent_data.turn_scores
            _resp_mask = agent_data.response_mask
            log_event(
                "grpo_tool_agent",
                "trajectory_summary",
                request_id=agent_data.request_id,
                assistant_turns=agent_data.assistant_turns,
                user_turns=agent_data.user_turns,
                num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                response_len=len(agent_data.response_mask),
                seq_len=len(agent_data.prompt_ids),
                # Record the prompt/generation boundary explicitly.
                prompt_len=len(agent_data.prompt_ids) - len(agent_data.response_mask),
                turn_spans=agent_data.span_records or None,
                response_length_limit=self.response_length,
                length_truncated=len(agent_data.response_mask) >= self.response_length,
                won=bool(agent_data.env_won),
                num_turn_scores=len(_turn_scores),
                num_nonzero_turn_scores=len([_s for _s in _turn_scores if _s]),
                sum_turn_scores=sum(_turn_scores) if _turn_scores else 0.0,
                turn_scores=list(_turn_scores),
                full_token_ids=_full_token_ids,
                full_text=_full_text,
                response_mask=_response_mask_str,
                dump_seq=_dump_seq,
                validate=bool(agent_data.is_validate),
                termination_reason=agent_data.termination_reason,
                # raw_token_ids and decision_mask have Dyad-specific semantics; leave them null for GRPO.
                raw_token_ids=None,
                decision_mask=None,
                turns=agent_data.turn_records
                or [
                    {
                        "turn": _i,
                        "env_action": None,
                        "env_obs": None,
                        "env_obs_truncated": False,
                        "turn_score": _s,
                    }
                    for _i, _s in enumerate(_turn_scores)
                ]
                or None,
                num_response_tokens=int(sum(_resp_mask)),
                num_masked_tokens=int(len(_resp_mask) - sum(_resp_mask)),
            )

        await finalize_session(agent_data)

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :] if agent_data.response_mask else []
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data

        # DYAD-GIGPO: retain every real decision, clipping spans with the final output.
        if agent_data.gigpo_enabled:
            limit = min(len(agent_data.response_mask), self.response_length)
            agent_data.extra_fields["gigpo_steps"] = [
                dict(step, start=min(step["start"], limit), end=min(step["end"], limit))
                for step in agent_data.gigpo_steps
            ]

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        # Pair with _call_tool, which reuses them across turns.

        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                # DYAD-ADDED: Use the environment success flag for the won metric.
                "won": bool(agent_data.env_won),
                # An empty turn_scores key also exists on single-turn outputs and is insufficient evidence.
                "multi_turn_scored": True,
            }
        )
        return output
    finally:
        await self._cleanup_tool_instances(agent_data)


async def tool_handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
    """Handle the pending state: prepare the prompt and start generation."""
    from verl.experimental.agent_loop.tool_agent_loop import (
        AgentState,
    )

    # DYAD: reset failures have no policy-visible prompt or generation.
    if getattr(agent_data, "runtime_env_done", False) and not agent_data.messages:
        token = self.tokenizer.pad_token_id
        if token is None:
            token = self.tokenizer.eos_token_id
        if token is None:
            raise ValueError("Terminal runtime transport requires a padding or EOS token")
        agent_data.prompt_ids = [int(token)]
        agent_data.termination_reason = "env_done"
        return AgentState.TERMINATED
    schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
    if self.tool_parser_name in ("codegym", "react", "react_fc") or getattr(self, "runtime_env_session", False):
        schemas = None
    # DYAD-DIVE: both policies see the model's same native tool template.
    if getattr(agent_data, "runtime_env_context", {}).get("protocol") == "dive":
        schemas = agent_data.runtime_env_context["action_tools"]
    if self.enable_continuous_token:
        prompt_ids = await self.ct_build_initial_tokens(agent_data.messages, tools=schemas)
    else:
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
        )
    agent_data.prompt_ids = prompt_ids
    if getattr(agent_data, "runtime_env_done", False):
        agent_data.termination_reason = "env_done"
        return AgentState.TERMINATED
    return AgentState.GENERATING


async def tool_handle_generating_state(
    self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
) -> AgentState:
    """Handle the generating state: generate model response and check for tool calls."""
    from verl.experimental.agent_loop.tool_agent_loop import (
        SPEC_DECODE_EXTRA_KEYS,
        AgentState,
        TokenOutput,
        _grpo_diag_full,
        log_event_full,
        os,
        record_span,
        simple_timer,
    )

    # Checking after generate discarded that turn before parsing its tool call (e.g. Done).
    # Both tool-processing paths return here, so this also prevents an extra assistant
    # generation with continuous tokens, without changing user-turn or length guards.
    if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
        agent_data.termination_reason = "max_assistant_turns"
        return AgentState.TERMINATED

    # DYAD-BEGIN: runtime sessions own a separate per-decision generation cap.
    from agent_system.rollout.env_session import session_sampling_params

    sampling_params = session_sampling_params(self, agent_data, sampling_params)
    if sampling_params is None:
        return AgentState.TERMINATED
    # DYAD-END

    # Inject tool parser stop tokens so generation halts after each tool call
    if self.tool_parser.stop_token_ids and getattr(agent_data, "runtime_env_tool", None) is None:
        stop_token_ids = list(set((sampling_params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
        sampling_params = {**sampling_params, "stop_token_ids": stop_token_ids}

    with simple_timer("generate_sequences", agent_data.metrics):
        output: TokenOutput = await self.server_manager.generate(
            request_id=agent_data.request_id,
            prompt_ids=agent_data.prompt_ids,
            sampling_params=sampling_params,
            image_data=agent_data.image_data,
            video_data=agent_data.video_data,
            audio_data=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
        )
    # first time to set num_preempted
    if agent_data.metrics.get("num_preempted") is None:
        agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
    # then add num_preempted to the metrics
    else:
        agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

    if not agent_data.extra_fields:
        agent_data.extra_fields.update(output.extra_fields)
    else:
        # Multi-round calls, only update the maximum max_global_steps.
        max_global_steps = output.extra_fields.get("max_global_steps", None)
        if max_global_steps:
            agent_data.extra_fields["max_global_steps"] = max_global_steps
        for key in SPEC_DECODE_EXTRA_KEYS:
            if key in output.extra_fields and key in agent_data.extra_fields:
                agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(output.extra_fields[key])

    agent_data.assistant_turns += 1
    # DYAD: an empty generation can still request logprobs for later recovery turns.
    agent_data.collect_response_logprobs = output.log_probs is not None
    agent_data.response_ids = output.token_ids
    # DYAD-GIGPO: record before termination/parser checks, including empty or unexecuted turns.
    if agent_data.gigpo_enabled:
        start = len(agent_data.response_mask)
        agent_data.gigpo_steps.append(
            {
                "start": start,
                "end": start + len(output.token_ids),
                "anchor": agent_data.gigpo_anchor,
                "initial": agent_data.gigpo_initial,
                "reward": 0.0,
                "executed": False,
            }
        )
    # DYAD: retain runtime policy evidence even when diagnostic logging is disabled.
    from agent_system.rollout.env_session import record_action_audit

    record_action_audit(self, agent_data, output)
    if self.enable_continuous_token:
        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            agent_data.prompt_ids,
            agent_data.response_ids,
            agent_data.response_mask,
            agent_data.response_logprobs
            if (agent_data.response_logprobs or agent_data.collect_response_logprobs)
            else None,
            assistant_logprobs=output.log_probs,
        )
        agent_data.prompt_ids = merge_result.token_ids
        agent_data.response_mask = response_mask
        if response_logprobs is not None:
            agent_data.response_logprobs = response_logprobs
    else:
        agent_data.prompt_ids += agent_data.response_ids
        # DYAD-ADDED(diag): Record generation spans when appended, independently of later mask reconstruction.
        record_span(agent_data.span_records, "GEN", len(agent_data.response_mask), len(agent_data.response_ids))
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

    if output.routed_experts is not None:
        agent_data.routed_experts = output.routed_experts

    # DYAD-MODIFIED(diag): Record the termination reason at each stopping condition.
    if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
        agent_data.termination_reason = "response_length"
        return AgentState.TERMINATED
    if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
        agent_data.termination_reason = "max_user_turns"
        return AgentState.TERMINATED

    # DYAD: native runtime sessions parse and execute raw ReAct in the env actor.
    if getattr(agent_data, "runtime_env_tool", None) is not None:
        return AgentState.PROCESSING_TOOLS

    # Extract tool calls (use per-sample tools if routed)
    active_tools = getattr(agent_data, "_active_tools", self.tools)
    tools = [tool.tool_schema for tool in active_tools.values()]
    # action_content is the sampled policy decision; display text must not replace it during replay.
    if getattr(output, "action_content", None):
        assistant_content, agent_data.tool_calls = await self.tool_parser.dyad_extract_tool_calls(
            output.action_content[0]
        )
    else:
        assistant_content, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            agent_data.response_ids, tools
        )

    # Per-turn action logging is controlled by GRPO_PRINT_STEP_ACTION.
    if os.getenv("GRPO_PRINT_STEP_ACTION", "1").strip().lower() not in ("0", "false", "no", ""):
        try:
            _resp_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
            _open, _close = {
                "react": ("<Action>", "</Action>"),
                "react_fc": ("<Action>", "</Action>"),
                "codegym": ("<|FunctionCallBegin|>", "<|FunctionCallEnd|>"),
            }.get(self.tool_parser_name, ("<tool_call>", "</tool_call>"))
            _i, _j = _resp_text.rfind(_open), _resp_text.rfind(_close)
            if 0 <= _i < _j:
                _sel = _resp_text[_i : _j + len(_close)].strip()
            elif agent_data.tool_calls:
                _sel = f"(no {_open} tag) parsed={agent_data.tool_calls}"
            else:
                _sel = f"<no tool call> tail={_resp_text[-160:]!r}"
            print(
                f"[ToolAgentLoop step] req={agent_data.request_id[:8]} turn={agent_data.assistant_turns} | {_sel}",
                flush=True,
            )
        except Exception:
            pass

    # Count a valid parse per generation; this metric does not change reward.
    if self.tool_parser_name == "codegym":
        agent_data.fmt_attempts += 1
        if agent_data.tool_calls:
            agent_data.fmt_valid += 1
    if self.enable_continuous_token:
        agent_data.messages.append(self._build_assistant_message(assistant_content, agent_data))

    if agent_data.extra_info.get("need_tools_kwargs") and _grpo_diag_full():
        log_event_full(
            "grpo_tool_agent",
            "generation_result",
            request_id=agent_data.request_id,
            assistant_turn=agent_data.assistant_turns,
            stop_reason=getattr(output, "stop_reason", None),
            token_count=len(agent_data.response_ids),
            num_tool_calls=len(agent_data.tool_calls),
            tool_names=[_tc.name for _tc in agent_data.tool_calls],
        )

    if agent_data.tool_calls:
        return AgentState.PROCESSING_TOOLS
    else:
        # DYAD: action-required protocols receive feedback rather than silently finishing.
        from agent_system.rollout.action_recovery import recover_missing_action

        recovered = await recover_missing_action(self, agent_data)
        if recovered is not None:
            return AgentState.GENERATING if recovered else AgentState.TERMINATED
        agent_data.termination_reason = "no_tool_call"
        return AgentState.TERMINATED


async def tool_handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
    """Handle the processing tools state: execute tool calls and prepare tool responses."""
    from verl.experimental.agent_loop.tool_agent_loop import (
        AgentState,
        _grpo_diag_enabled,
        _grpo_diag_full,
        asyncio,
        build_gpt_oss_tool_response_text,
        log_event,
        logger,
        obs_maxlen,
        record_span,
        simple_timer,
    )

    # DYAD: preserve official agent-visible user/API messages.
    if getattr(agent_data, "runtime_env_tool", None) is not None:
        from agent_system.rollout.env_session import advance_session

        done = await advance_session(self, agent_data)
        return AgentState.TERMINATED if done else AgentState.GENERATING
    add_messages: list[dict[str, Any]] = []
    new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute
    previous_messages = list(agent_data.messages)

    # DYAD-GIGPO: one decision must identify one environment transition, never a parallel bundle.
    if agent_data.gigpo_enabled and (len(agent_data.tool_calls) != 1 or self.max_parallel_calls < 1):
        raise ValueError("GiGPO training requires exactly one tool call per assistant decision")
    tasks = []
    tool_call_names = []
    for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
        tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
        tool_call_names.append(tool_call.name)

    with simple_timer("tool_calls", agent_data.metrics):
        responses = await asyncio.gather(*tasks)
    # DYAD-EVAL: retain failures before any response truncation or early return.
    from agent_system.rollout.env_session import record_task_response

    for _, _, tool_metrics in responses:
        record_task_response(agent_data, tool_metrics)

    env_done = False
    for tool_index, (tool_response, tool_reward, tool_metrics) in enumerate(responses):
        tool_call = agent_data.tool_calls[tool_index]

        if tool_reward is not None:
            _m = tool_metrics if isinstance(tool_metrics, dict) else {}
            # Infrastructure failures contribute zero environment reward to avoid contaminating advantages.
            turn_reward = 0.0 if (_m.get("http_failure") or _m.get("tool_failure")) else float(tool_reward)
            agent_data.turn_scores.append(turn_reward)
            # DYAD-GIGPO: precisely the same reward consumed by the baseline scorer.
            if agent_data.gigpo_enabled:
                agent_data.gigpo_steps[-1]["reward"] += turn_reward
            # Partial reward is not proof of success; use won/success from the environment.
            if _m.get("won") or _m.get("success"):
                agent_data.env_won = True
            if _m.get("done"):
                env_done = True

            # Record executed actions, rewards, and candidates for environment-call verification.
            if agent_data.extra_info.get("need_tools_kwargs") and _grpo_diag_enabled():
                _action_sent = _m.get("last_action_repr")
                log_event(
                    "grpo_tool_agent",
                    "env_step",
                    request_id=agent_data.request_id,
                    # Use one-based step numbering to match Dyad trajectory events.
                    turn_index=len(agent_data.turn_scores),
                    action_sent=_action_sent,
                    reward=turn_reward,
                    http_failure=bool(_m.get("http_failure")),
                    done=bool(_m.get("done", False)),
                    step_count=_m.get("step_count"),
                    invalid_action=bool(_m.get("invalid_action", False)),
                    max_turns_reached=bool(_m.get("max_turns_reached", False)),
                    num_available_actions=len(_m.get("available_actions", []) or []),
                    available_actions=_m.get("available_actions", []),
                    observation=(
                        (tool_response.text or "")[: obs_maxlen()]
                        if (tool_response is not None and _grpo_diag_full())
                        else None
                    ),
                    execution_time_ms=_m.get("execution_time_ms"),
                    timed_out=bool(_m.get("timed_out", False)),
                    tool_failure=bool(_m.get("tool_failure", False)),
                )
        # Create message from tool response
        if tool_response.image or tool_response.video:
            # Multi-modal content with structured format
            if not getattr(self.processor, "image_processor", None):
                raise ValueError(
                    "Multimedia data can only be processed by `processor`, but the processor is None. "
                    "This error is often caused if you are using a LLM model but your tool returns multimodal "
                    "data. Plase use a vlm as the base model."
                )
            content = []
            if tool_response.image:
                content.append({"type": "image"})
            if tool_response.video:
                content.append({"type": "video"})
            if tool_response.text:
                content.append({"type": "text", "text": tool_response.text})
            message = {"role": "tool", "content": content}
        else:
            # Text-only content
            message = {"role": "tool", "content": tool_response.text or ""}
        if tool_call.tool_call_id is not None:
            message["tool_call_id"] = tool_call.tool_call_id

        add_messages.append(message)

        # Handle image data
        if tool_response.image:
            # Add new image data
            if isinstance(tool_response.image, list):
                # Ensure all elements in the list are valid image objects
                for img in tool_response.image:
                    if img is not None:  # Add a check to ensure the image is not None
                        new_images_this_turn.append(img)  # Using local variable
            else:
                # Ensure the image is not None
                if tool_response.image is not None:
                    new_images_this_turn.append(tool_response.image)  # Using local variable

        # Handle video data
        if tool_response.video:
            # Currently not supported, raise informative error
            logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
            raise NotImplementedError("Multimedia type 'video' is not currently supported. Only 'image' is supported.")

        if tool_reward is not None:
            agent_data.tool_rewards.append(tool_reward)  # DYAD-NOTE: Environment turn_scores were collected above.

    if agent_data.extra_info.get("need_tools_kwargs") and _grpo_diag_enabled():
        _obs_max = obs_maxlen()
        for _tool_response, _tool_reward, _metrics in responses:
            _obs = getattr(_tool_response, "text", None) or ""
            _m = _metrics if isinstance(_metrics, dict) else {}
            agent_data.turn_records.append(
                {
                    "turn": len(agent_data.turn_records),
                    "env_action": _m.get("last_action_repr"),
                    "env_obs": (_obs[:_obs_max] if _grpo_diag_full() else None),
                    "env_obs_truncated": len(_obs) > _obs_max,
                    "turn_score": (0.0 if _m.get("http_failure") else _tool_reward),
                }
            )

    agent_data.messages.extend(add_messages)

    if self.enable_continuous_token and not new_images_this_turn:
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
            previous_messages,
            agent_data.messages,
            agent_data.prompt_ids,
            agent_data.response_mask,
            agent_data.response_logprobs if agent_data.response_logprobs else None,
            tools=schemas,
        )
        if len(response_mask) >= self.response_length:
            return AgentState.TERMINATED
        agent_data.prompt_ids = merge_result.token_ids
        agent_data.response_mask = response_mask
        if agent_data.response_logprobs:
            agent_data.response_logprobs = response_logprobs or []
        agent_data.user_turns += 1
        if env_done:
            agent_data.termination_reason = "env_done"
            return AgentState.TERMINATED
        return AgentState.GENERATING
    elif self.tool_parser_name == "gpt-oss":
        logger.info("manually format tool responses for gpt-oss")
        tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
        response_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
        )
    elif self.tool_parser_name == "gemma4":
        # Gemma4's chat template drops tool responses when passed without the preceding
        # assistant tool_call message. Manually format the response tokens.
        # Format: <|tool_response>response:func_name{value:<|"|>content<|"|>}<tool_response|>
        parts = []
        for msg, name in zip(add_messages, tool_call_names, strict=True):
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
            parts.append(f'<|tool_response>response:{name}{{value:<|"|>{content}<|"|>}}<tool_response|>')
        tool_response_text = "".join(parts)
        response_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
        )
    else:
        # Note that we have to pass None to the images and videos if there are no new images / videos
        # to stay compatible with downstream image processing logic!
        images = new_images_this_turn if new_images_this_turn else None
        videos = None
        response_ids = await self.apply_chat_template(
            add_messages,
            images=images,
            videos=videos,
            remove_system_prompt=True,
        )
        # The model stopped at the assistant close token and never emitted the template's
        # trailing turn separator (e.g. "\n" for Qwen); rendering this tool turn in isolation
        # also omits it. Restore it so the incremental sequence matches apply_chat_template of
        # the full conversation (see verl issue #6501 comment). ``turn_separator`` is [] for
        # templates without one, so this is a no-op there.
        response_ids = self.turn_separator + response_ids

    if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
        return AgentState.TERMINATED
    # Update prompt_ids and response_mask

    if new_images_this_turn:
        if agent_data.image_data is None:
            agent_data.image_data = []
        elif not isinstance(agent_data.image_data, list):
            agent_data.image_data = [agent_data.image_data]
        for img in new_images_this_turn:
            agent_data.image_data.append(img)

    agent_data.prompt_ids += response_ids
    # DYAD-ADDED(diag): Record observation spans when injected into the response.
    record_span(agent_data.span_records, "ENV", len(agent_data.response_mask), len(response_ids))
    agent_data.response_mask += [0] * len(response_ids)
    if agent_data.response_logprobs:
        agent_data.response_logprobs += [0.0] * len(response_ids)
    agent_data.user_turns += 1

    if env_done:
        agent_data.termination_reason = "env_done"
        return AgentState.TERMINATED

    return AgentState.GENERATING


async def tool_cleanup_tool_instances(self, agent_data: AgentData) -> None:
    """Release environment sessions created by _call_tool for this trajectory."""
    from verl.experimental.agent_loop.tool_agent_loop import (
        logger,
    )

    instances = agent_data.extra_fields.setdefault("_env_tool_instances", {})
    # DYAD: runtime bootstrap shares the Dyad loop's neutral lease registry.
    instances.update(agent_data.extra_fields.pop("tool_instances", {}))
    active_tools = getattr(agent_data, "_active_tools", self.tools)
    # DYAD: drain every lease, but never publish a successful trajectory after
    # a strict environment failed to close.
    cleanup_error = None
    for tool_name, instance_id in list(instances.items()):
        tool = active_tools.get(tool_name) or self.tools.get(tool_name)
        if tool is None:
            continue
        try:
            released = await tool.release(instance_id)
            # DYAD-EVAL: local ALFWorld logs close failures instead of raising.
            if released is False:
                from agent_system.rollout.env_session import invalidate_task_evidence

                invalidate_task_evidence(agent_data)
        except Exception as e:
            # DYAD-EVAL: even non-fatal cleanup errors invalidate this evaluation attempt.
            from agent_system.rollout.env_session import invalidate_task_evidence

            invalidate_task_evidence(agent_data)
            logger.warning(f"释放 env 工具实例失败 tool={tool_name} instance={instance_id}: {e}")
            if getattr(tool, "fail_fast_on_error", False) and cleanup_error is None:
                cleanup_error = e
    instances.clear()
    if cleanup_error is not None:
        raise RuntimeError("External env session cleanup failed") from cleanup_error


async def tool_call_tool(
    self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
) -> tuple[ToolResponse, float, dict]:
    """Call tool and return tool response.

    Dispatches between two contracts:
    - ``FunctionTool``: stateless function-based tool. Invoked directly with
      parsed arguments; no lifecycle.
    - ``BaseTool`` subclass: stateful tool with full lifecycle.
    """
    from verl.experimental.agent_loop.tool_agent_loop import (
        FunctionTool,
        ToolResponse,
        asyncio,
        json,
        logger,
        normalize_function_tool_return,
        os,
    )

    active_tools = getattr(agent_data, "_active_tools", self.tools)
    # DYAD-GIGPO: direct baseline tool callers need not carry algorithm metadata.
    gigpo_enabled = bool(getattr(agent_data, "gigpo_enabled", False))

    # Validate tool name
    tool_name = tool_call.name
    if tool_name not in active_tools:
        available = list(active_tools.keys())
        msg = f"Unknown function '{tool_name}'. Available tools: {available}"
        logger.warning(msg)
        return ToolResponse(text=msg), 0.0, {}

    # Validate tool arguments
    try:
        tool_args = json.loads(tool_call.arguments)
    except (json.JSONDecodeError, TypeError) as e:
        msg = f"Invalid JSON in arguments for '{tool_name}': {e}"
        logger.warning(msg)
        return ToolResponse(text=msg), 0.0, {}

    # Execute tool
    tool, instance_id = None, None
    try:
        tool = active_tools[tool_name]

        if isinstance(tool, FunctionTool):
            # Function-based tools have no lifecycle; call directly.
            # Note: tools_kwargs (create_kwargs / release_kwargs) is intentionally
            # ignored here. Function tools are stateless and per-trajectory state
            # injection is not supported by design; use a BaseTool subclass instead.
            raw = await tool.call(tool_args)
            tool_execution_response, tool_reward, res = normalize_function_tool_return(raw)
        else:
            # >>> BEGIN DYAD(env-session): Reuse stateful environment instances across trajectory turns.
            # Creating on each call would reset progress. Cache by tool name, use request_id for identity,
            # and release through _cleanup_tool_instances when the trajectory ends.
            kwargs = tools_kwargs.get(tool_name, {})
            env_instances = agent_data.extra_fields.setdefault("_env_tool_instances", {})
            if tool_name in env_instances:
                instance_id = env_instances[tool_name]
            else:
                instance_id, _ = await tool.create(
                    instance_id=agent_data.request_id,
                    create_kwargs=kwargs.get("create_kwargs", {}),
                )
                env_instances[tool_name] = instance_id
                # DYAD-GIGPO: lazy reset must not change the policy's initial prompt.
                if gigpo_enabled:
                    initial = tool.get_observation(instance_id)
                    if not isinstance(initial, str):
                        raise ValueError("GiGPO reset observation must be a string")
                    agent_data.gigpo_anchor = initial
                    for step in agent_data.gigpo_steps:
                        if step["initial"] and step["anchor"] is None:
                            step["anchor"] = initial

            # Bound unresponsive environment calls; a timeout of zero disables the bound.
            _timeout_ms = float(os.getenv("GRPO_TOOL_TIMEOUT_MS", "") or getattr(tool, "execution_timeout_ms", 0) or 0)
            _exec = tool.execute(instance_id, tool_args, agent_data=agent_data)
            if _timeout_ms > 0:
                tool_execution_response, tool_reward, res = await asyncio.wait_for(_exec, timeout=_timeout_ms / 1000.0)
            else:
                tool_execution_response, tool_reward, res = await _exec
    except asyncio.TimeoutError:
        logger.warning(f"Tool '{tool_name}' execution timed out")
        # A timed-out fail-fast session is untrustworthy: discard it and fail the trajectory.
        if tool is not None and (getattr(tool, "fail_fast_on_error", False) or gigpo_enabled):
            # DYAD: cancellation of a Ray await does not stop its actor.
            # Discard it before finally attempts to release the session.
            if instance_id is not None and hasattr(tool, "abort"):
                await tool.abort(instance_id, "tool execution failed or timed out")
            agent_data.extra_fields.get("_env_tool_instances", {}).pop(tool_name, None)
            # Preserve the pre-migration exception context for timeout diagnostics.
            raise RuntimeError(  # noqa: B904
                f"External env tool {tool_name!r} timed out for trajectory {agent_data.request_id}"
            )
        return (
            ToolResponse(text="Error: tool execution timed out"),
            0.0,
            {"timed_out": True, "tool_failure": True},
        )
    except Exception as e:
        logger.warning(f"Error executing tool '{tool_name}': {e}")
        if tool is not None and (getattr(tool, "fail_fast_on_error", False) or gigpo_enabled):
            # DYAD: cancellation of a Ray await does not stop its actor.
            # Discard it before finally attempts to release the session.
            if instance_id is not None and hasattr(tool, "abort"):
                await tool.abort(instance_id, "tool execution failed or timed out")
            agent_data.extra_fields.get("_env_tool_instances", {}).pop(tool_name, None)
            raise RuntimeError(
                f"External env tool {tool_name!r} failed for trajectory {agent_data.request_id}: "
                f"{type(e).__name__}: {e}"
            ) from e
        return (
            ToolResponse(text=f"Error executing tool '{tool_name}': {e}"),
            0.0,
            {"tool_failure": True, "error": str(e), "error_type": type(e).__name__},
        )
    finally:
        # Release non-environment instances here; trajectory cleanup owns environment sessions.
        if tool and instance_id and not isinstance(tool, FunctionTool):
            if instance_id not in agent_data.extra_fields.get("_env_tool_instances", {}).values():
                await tool.release(instance_id)
            # <<< END DYAD

    # DYAD-GIGPO: use raw state metadata before display truncation, not ToolResponse.text.
    if gigpo_enabled:
        if (
            not isinstance(res, dict)
            or res.get("env_actor_error")
            or res.get("tool_failure")
            or res.get("http_failure")
        ):
            raise ValueError("GiGPO cannot train on unknown environment state after a tool failure")
        before = res.get("observation_before")
        after = res.get("observation_after")
        advanced = res.get("environment_advanced")
        if not isinstance(before, str) or not isinstance(after, str) or not isinstance(advanced, bool):
            raise ValueError("GiGPO tool result is missing raw observation metadata")
        if before != agent_data.gigpo_anchor:
            raise ValueError("GiGPO pre-action observation differs from the recorded environment state")
        if not advanced and before != after:
            raise ValueError("GiGPO unexecuted action cannot change the environment observation")
        agent_data.gigpo_steps[-1]["executed"] = advanced
        agent_data.gigpo_anchor = after
        if advanced:
            agent_data.gigpo_initial = False

    tool_response_text = tool_execution_response.text
    if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
        if self.tool_response_truncate_side == "left":
            tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
        elif self.tool_response_truncate_side == "right":
            tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
        else:
            length = self.max_tool_response_length // 2
            tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

    # Create ToolResponse from tool execution result
    tool_response_kwargs = {"text": tool_response_text}

    # Add multimedia data if present
    for attr_name in ["image", "video"]:
        if hasattr(tool_execution_response, attr_name):
            attr_value = getattr(tool_execution_response, attr_name)
            if attr_value is not None:
                tool_response_kwargs[attr_name] = attr_value

    return ToolResponse(**tool_response_kwargs), tool_reward, res
