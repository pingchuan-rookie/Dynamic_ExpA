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
"""CodeGym function-call parser (faithful reproduction of the paper's B2 text protocol).

The paper (arXiv:2509.17325) has the model write tool calls as **plain text**:

    <|FunctionCallBegin|>[{"name":"CalculateGCD","parameters":{"array":[4,4,6]}}]<|FunctionCallEnd|>

Note that these markers are ordinary characters in the Qwen vocabulary (`< | Function Call Begin | >`
is 7 ordinary tokens), **not special tokens**; the model generates them freely and there is no
decoding constraint during training.

This parser extracts that text and **wraps** it into a single call to the veRL environment tool
(the in-process CodeGymLocalEnvTool, whose tool name is fixed to ``codegym_call``): the original
``{"name":..,"parameters":..}`` JSON string is passed through as the ``action`` argument, and is
finally handed to CodeGym ``env.step`` via the in-process CodeGymEnvPool.

This way veRL only has to register one generic tool ``codegym_call``, while the model still
"calls a CodeGym function name" directly in the paper's format, without having to register one
veRL tool per function name of each env.

Usage: set ``actor_rollout_ref.rollout.multi_turn.format=codegym`` in the training config.
"""

import json
import logging
import os

import regex

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Generic veRL-side tool name carrying CodeGym calls; must match tool_name in env/configs/codegym_tool.yaml.
CODEGYM_WRAPPER_TOOL_NAME = "codegym_call"


def codegym_block_to_function_calls(block: str) -> list[FunctionCall]:
    """Turn the body of one ``<|FunctionCallBegin|>...<|FunctionCallEnd|>`` block into tool calls.

    Shared by two callers that must stay byte-compatible:
      - grpo_react (this module's parser), which regex-extracts the block from the model's free text;
      - Dyad raw-command (parsers/dyad.py, env_serialize=codegym_raw), which rebuilds the identical block from
        `surface_form + free text`.
    The raw-command schema's premise is that the two produce the same block, so a second copy of this parsing here
    would be a silent way for the two conditions to drift apart.

    A block that does not parse yields an empty list and an error log -- same as grpo_react. Deliberately
    no lenient fallback: a malformed call must cost the trajectory its reward, otherwise a model that
    writes broken JSON gets scored as if it had acted.
    """
    block = block.strip()
    if not block:
        return []
    try:
        # The paper's format is a JSON list holding one dict; a bare dict is also accepted.
        parsed = json.loads(block)
    except Exception as e:  # noqa: BLE001
        logger.error("CodeGym tool-call JSON decode failed: %s | block=%r", e, block)
        return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        logger.error("CodeGym tool-call expected list/dict, got %r", type(parsed))
        return []

    function_calls: list[FunctionCall] = []
    for call in parsed:
        if not isinstance(call, dict) or "name" not in call:
            logger.error("CodeGym tool-call missing 'name': %r", call)
            continue
        # Rebuild the {"name":..,"parameters":..} JSON string expected by CodeGym env.step,
        # passed through as the action argument of the codegym_call tool.
        inner_call = json.dumps(
            {"name": call["name"], "parameters": call.get("parameters", {})},
            ensure_ascii=False,
        )
        function_calls.append(
            FunctionCall(
                name=CODEGYM_WRAPPER_TOOL_NAME,
                arguments=json.dumps({"action": inner_call}, ensure_ascii=False),
            )
        )
    return function_calls


@ToolParser.register("codegym")
class CodeGymToolParser(ToolParser):
    """Parses the ``<|FunctionCallBegin|>[...]<|FunctionCallEnd|>`` text protocol."""

    missing_action_feedback = "Format error."

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.begin_token = "<|FunctionCallBegin|>"
        self.end_token = "<|FunctionCallEnd|>"
        # DOTALL lets . match across lines; non-greedy so it grabs the content between the markers.
        self.call_regex = regex.compile(
            r"<\|FunctionCallBegin\|>(.*?)<\|FunctionCallEnd\|>", regex.DOTALL
        )

    @rollout_trace_op
    # DYAD-NOTE(verl0.9): the base signature gained a `tools` parameter
    # (ToolParser.extract_tool_calls(self, responses_ids, tools=None)), and 0.9's
    # ToolAgentLoop._handle_generating_state passes the tool schema list as the second
    # *positional* argument. dyad's parsers speak a text protocol (<Action>...</Action> and
    # friends) and never look at the schema, so this is accepted and ignored. Not accepting it
    # raises `TypeError: takes 2 positional arguments but 3 were given` -- inside a Ray actor
    # during rollout, where the driver only ever sees "no materializable trajectories".
    async def extract_tool_calls(
        self,
        responses_ids: list[int],
        tools: "list | None" = None,  # noqa: ARG002  a text-protocol parser has no use for schemas
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)

        # Fast prune: without both markers there cannot be any tool call.
        if self.begin_token not in text or self.end_token not in text:
            return text, []

        function_calls: list[FunctionCall] = []
        for block in self.call_regex.findall(text):
            function_calls += codegym_block_to_function_calls(block)

        # content = the plain text left after removing the tool-call fragments (thinking / natural language part).
        content = self.call_regex.sub("", text)
        return content, function_calls
