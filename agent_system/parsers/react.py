# Copyright 2025 dyad2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""ToolParser for the ReAct text protocol (used by the GRPO baseline).

Registered names:
- `react`    : `<think>...</think><action>...</action>`, extracts <action> and sends it to env as raw_action
- `react_fc` : the function-call variant of the above
"""

import json
import logging
import os

import regex

from agent_system.parsers.action_envelope import extract_action
from agent_system.parsers.native_tools import decode_response

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

from agent_system.utils.logging import get_dyad_logger

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
dyad_logger = get_dyad_logger()


@ToolParser.register("react")
class ReactToolParser(ToolParser):
    """Parser for the ALFWorld react (non function-call) format.

    The model emits the same tag style as Dyad: <think>...</think> then <action>cmd</action>,
    but the content of <action> is the [raw ALFWorld command] (e.g. "go to cabinet 1").
    This parser extracts the command inside <action> and wraps it into a call to the alfworld_action tool,
    arguments={"raw_action": cmd}, which AlfworldLocalEnvTool/AlfworldAdapter.build_action then dispatches
    as raw_action to the in-process env pool.

    Requires one <think> block followed by one <action> block; with missing or empty action tags it
    returns an empty tool call list, allowing the agent loop to request a corrected action
    without advancing the environment.
    Note: the react prompt documents <action> itself, so the training-side tool_agent_loop no longer injects the hermes <tools> block (see _handle_pending_state).
    """

    # Environment tool name for the react format (matches tool_name in env/configs/alfworld_tool.yaml).
    TOOL_NAME = "alfworld_action"
    missing_action_feedback = "Format error."

    def __init__(self, tokenizer) -> None:
        dyad_logger.debug("[ReactToolParser].init()")
        super().__init__(tokenizer)
        self.action_regex = regex.compile(r"<action>(.*?)</action>", regex.DOTALL)

    @rollout_trace_op
    # DYAD-NOTE(verl0.9): the base signature gained a `tools` parameter
    # (ToolParser.extract_tool_calls(self, responses_ids, tools=None)), and 0.9's
    # ToolAgentLoop._handle_generating_state passes the tool schema list as the second
    # *positional* argument. dyad's parsers speak a text protocol (<action>...</action> and
    # friends) and never look at the schema, so this is accepted and ignored. Not accepting it
    # raises `TypeError: takes 2 positional arguments but 3 were given` -- inside a Ray actor
    # during rollout, where the driver only ever sees "no materializable trajectories".
    async def extract_tool_calls(
        self,
        responses_ids: list[int],
        tools: "list | None" = None,  # noqa: ARG002  a text-protocol parser has no use for schemas
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        # decode is slow, run it in a thread pool so the event loop is not blocked
        text = await loop.run_in_executor(None, decode_response, self.tokenizer, responses_ids)

        try:
            _, action = extract_action(text)
        except ValueError:
            return text, []
        matches = [action]
        # The envelope contains exactly one action.
        command = matches[-1].strip()
        if not command:
            return text, []

        function_calls = [
            FunctionCall(
                name=self.TOOL_NAME,
                arguments=json.dumps({"raw_action": command}, ensure_ascii=False),
            )
        ]
        # remove <action>...</action> from the text; what is left (including <think>) is returned as content
        content = self.action_regex.sub("", text)
        return content, function_calls
@ToolParser.register("react_fc")
class ReactFcToolParser(ToolParser):
    """Parser for the GSM8K calculator function-call format (user-specified, not the standard hermes <tool_call>).

    The model emits <think>...</think> then <action>{json}</action>, where the JSON is a call to the calculator:
      - compute: <action>{"caculate": "48/2="}</action> (note the user's spelling `caculate`; calculate/expression are also accepted)
      - answer:  <action>{"answer": "72"}</action>

    The only difference from react is that the content of <action> is JSON instead of bare text. The parsed dict
    is handed directly as the tool-call arguments to CalcAdapter.build_action (which accepts
    caculate/calculate/expression/answer/raw_action).

    Requires the shared reasoning/action envelope; if JSON parsing fails it treats the payload as raw_action text
    (so the env reports an error rather than the action being dropped).
    The target tool name (TOOL_NAME) is pointed automatically by tool_agent_loop according to the config (GSM8K -> calculator).
    """

    TOOL_NAME = "calculator"
    missing_action_feedback = "Format error."

    def __init__(self, tokenizer) -> None:
        dyad_logger.debug("[ReactFcToolParser].init()")
        super().__init__(tokenizer)
        self.action_regex = regex.compile(r"<action>(.*?)</action>", regex.DOTALL)

    @rollout_trace_op
    # DYAD-NOTE(verl0.9): the base signature gained a `tools` parameter
    # (ToolParser.extract_tool_calls(self, responses_ids, tools=None)), and 0.9's
    # ToolAgentLoop._handle_generating_state passes the tool schema list as the second
    # *positional* argument. dyad's parsers speak a text protocol (<action>...</action> and
    # friends) and never look at the schema, so this is accepted and ignored. Not accepting it
    # raises `TypeError: takes 2 positional arguments but 3 were given` -- inside a Ray actor
    # during rollout, where the driver only ever sees "no materializable trajectories".
    async def extract_tool_calls(
        self,
        responses_ids: list[int],
        tools: "list | None" = None,  # noqa: ARG002  a text-protocol parser has no use for schemas
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, decode_response, self.tokenizer, responses_ids)

        try:
            _, action = extract_action(text)
        except ValueError:
            return text, []
        matches = [action]
        raw = matches[-1].strip()
        if not raw:
            return text, []

        # parse the JSON inside <action>; on failure fall back to treating it as raw_action text.
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                obj = {"raw_action": str(obj)}
        except Exception:
            obj = {"raw_action": raw}

        function_calls = [
            FunctionCall(
                name=self.TOOL_NAME,
                arguments=json.dumps(obj, ensure_ascii=False),
            )
        ]
        content = self.action_regex.sub("", text)
        return content, function_calls
