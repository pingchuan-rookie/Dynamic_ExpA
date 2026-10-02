# Copyright 2025 dyad2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""ToolParser for the Dyad unified action protocol (★ Dyad only).

Registered name `dyad`: parses the unified action produced by the action head (`{"name", "params"}`)
and restores it into an env-executable call according to the schema selected by DYAD_ACTION_YAML.

Env-serialize variants (selected by the schema's `env_serialize` field):
- `alfworld`     (fixed slots): `unified_action_to_alfworld_call` turns the slotted params into the
  `{"action name", "argument"}` the ALFWorld env expects (id_* merged into the previous param),
  after which the projector applies the template.
- `codegym`      (fixed slots): `decisions_to_codegym_actions` replays the decisions straight into
  `{"name", "parameters"}`.
- `alfworld_raw` / `calc_raw` (free arguments): `unified_action_to_raw_command` joins `surface_form + free string`
  into that env's raw command and passes it through the projector's `raw_action` verbatim (no longer
  going through action_templates). The join is env-agnostic, so the two share one branch.
- `webshop_raw` (free arguments): validate the emitted native command against the selected verb;
  preserve its query/target verbatim rather than space-joining replayed parameters.
- `codegym_raw`  (free arguments)    : same join, but the result is the paper's B2 block
  (`[{"name": .., "parameters": {..}}]`), handed to grpo_react's own `codegym_block_to_function_calls`.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import regex
import yaml

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

from agent_system.utils.logging import get_dyad_logger

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
dyad_logger = get_dyad_logger()

# env_serialize values whose action is rebuilt as a raw command string (the Dyad raw-command variants).
# Adding an env here is enough as long as its projector accepts `raw_action`; see
# agent_system/environments/{alfworld,calc}/adapter.py.
_RAW_ENV_SERIALIZE = ("alfworld_raw", "calc_raw")


def _strip_exit_marker(action_config: dict[str, Any], command: str) -> str:
    """Drop a trailing exit marker from a rebuilt raw command.

    Dyad raw-command's free-text segment ends *on* the exit marker, and `unified_action_to_raw_command`
    joins `surface_form + free text` verbatim -- so the rebuilt command carries the marker with it. The
    alfworld and calc raw paths do not care: their projectors take the whole string as `raw_action`,
    and the env strips or ignores the tail.

    codegym does care, because its rebuilt command is *parsed as JSON*:

        block='[{"name": "Observe", "parameters": {}}]<|FunctionCallEnd|>'
        -> json.loads: Extra data: line 1 column 40

    and `codegym_block_to_function_calls` returns an empty list on a parse failure (deliberately --
    a malformed call must cost the trajectory its reward). An empty list means the agent loop calls
    no tool at all, which it does **silently**: the trajectory just ends after one turn with the env
    never stepped. Measured before the fix: 16 tool-call steps, 0 env calls.
    """
    exit_str = str(action_config.get("exit_str") or "")
    if exit_str and command.endswith(exit_str):
        return command[: -len(exit_str)].rstrip()
    return command


# Instance-id slots are serialized with the preceding entity, e.g. cabinet + 1 becomes cabinet 1.
ID_PREFIX = "id_"


def unified_action_to_alfworld_call(
    action_config: dict[str, Any],
    action: dict[str, Any],
    *,
    merge_id_to_previous: bool = True,
    id_sep: str = " ",
) -> dict[str, Any]:
    """Turn the {"name","params":{pname:value}} produced by replay_unified_actions into the alfworld env
    call {"action name","argument"} (id_* merged into the previous param, aligned with the structure the
    ALFWorld env expects).

    Iterates in [the param signature order of the compiled config] (not the model's param selection order),
    so merging id_* into the previous param is stable for fixed slot order. Missing params
    (truncated generation) are skipped.
    """
    name = action["name"]
    values = action.get("params", {}) or {}
    action_cfg = action_config.get("actions", {}).get(name, {})
    sig = [p["name"] for p in action_cfg.get("params", [])]

    argument: dict[str, str] = {}
    previous_key: str | None = None
    for pname in sig:
        if pname not in values:
            continue
        val = str(values[pname]).strip()
        if not val:
            continue          # an open param value may be empty (truncated generation / terminator right after the fixed action prefix), skip instead of joining
        if merge_id_to_previous and pname.startswith(ID_PREFIX):
            if previous_key is None:
                argument[pname] = val
            else:
                argument[previous_key] = f"{argument[previous_key]}{id_sep}{val}"
            continue
        argument[pname] = val
        previous_key = pname

    return {"action name": name, "argument": argument}


def unified_action_to_raw_command(
    action_config: dict[str, Any],
    action: dict[str, Any],
) -> str:
    """Dyad raw-command (env_serialize=*_raw): join a unified action back into that env's raw command string.

    In the free-argument schema every action has only [one free-text segment] synthesized by the compiler, and
    `surface_form` is exactly the command's fixed prefix ('go to' / 'take' for alfworld, 'answer' for the
    calculator) -- so command = `surface_form + free string`, byte-for-byte identical to the text written
    inside <Action>. When the value is empty (truncated generation) only the prefix is returned, and the
    env reports the error itself ("Nothing happens" for alfworld).

    An empty surface_form is **no longer legal** and this code no longer has to tolerate one. It used to:
    calc_base.yaml set `surface_form: ""` so that its body matched grpo_react's `<Action>48/2=</Action>`
    byte for byte. That conflicts with the invariant that every decision writes at least one base
    vocabulary token, so Dyad raw-command could not run at all; calc_base now writes 'calc', which
    calc_session._CALC_PREFIXES strips before evaluating, and compile_action_schema rejects an empty
    surface_form outright. Only a name missing from the config entirely falls back to the action name.

    Leading whitespace of the value has already been stripped by _clean_open_value, so a single space is
    used uniformly for joining here.
    """
    name = action["name"]
    actions = action_config.get("actions", {})
    surface_form = str(actions[name].get("surface_form", "") or "") if name in actions else str(name)
    parts = [surface_form.strip()]
    parts += [
        str(v).strip()
        for v in (action.get("params", {}) or {}).values()
        if str(v).strip()
    ]
    return " ".join(p for p in parts if p)
@ToolParser.register("dyad")
class DyadToolParser(ToolParser):
    """
    Adapts the tool call output format of Dyad / vLLM.
    Modelled on vllm's Dyad_tool_parser implementation (link given in its comments).
    """

    missing_action_feedback = "Format error."

    def __init__(self, tokenizer) -> None:
        dyad_logger.debug("[DyadToolParser].init()")
        super().__init__(tokenizer)

        # Dyad style: tool calls are wrapped in <tool_call> ... </tool_call>
        self.tool_call_start_token: str = "<tool>"
        self.tool_call_end_token: str = "</tool>"

        # DOTALL: lets '.' match newlines so (.*?) can grab tool call content across lines
        # This regex finds the inner content of every <tool>...</tool>
        self.tool_call_regex = regex.compile(r"<tooL>(.*?)</tool>", regex.DOTALL)

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
        """
        Core logic:
          1) decode token ids -> text
          2) grab every tool_call block with a regex
          3) json.loads each block, extract name/arguments
          4) remove the tool_call blocks from the text and return the rest as content
        """
        dyad_logger.debug("[DyadToolParser].extract_tool_calls()")
        loop = get_event_loop()

        # tokenizer.decode is usually CPU-bound / possibly slow: run it in a thread-pool executor so the event loop is not blocked
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)

        # Fast prune: without the begin/end markers there cannot be any tool call
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        # find the inner string of every <tool_call>...</tool_call> (there may be several tool calls)
        matches = self.tool_call_regex.findall(text)

        function_calls = []
        for match in matches:
            try:
                # the Dyad format is usually JSON, e.g.:
                # {"name": "xxx", "arguments": {...}}
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]

                # arguments is json.dumps'd once more here:
                # - ensure_ascii=False: do not escape non-ASCII characters
                # - note: FunctionCall.arguments is typed str, hence it is carried as a string
                function_calls.append(
                    FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False))
                )
            except Exception as e:
                # log parse failures without aborting the parsing of the whole output
                logger.error(f"Failed to decode tool call: {e}")

        # remaining text exclude tool call tokens
        # sub("", text) blanks out every tool_call block, leaving the pure content
        content = self.tool_call_regex.sub("", text)

        return content, function_calls

    @rollout_trace_op
    async def dyad_extract_tool_calls(self, action_content) -> tuple[str, list[FunctionCall]]:
        """
        Parse raw token ids into structured tool/function calls.
        """

        dyad_logger.debug("[DyadToolParser.extract_tool_calls] Start parsing tool call")
        loop = get_event_loop()

        action_config = action_content["action_config"]
        raw_token_ids = action_content["raw_token_ids"]

        if action_config.get("env_serialize") == "webshop_raw":
            from agent_system.policies.dyad.actions.action_router import build_unified_policy_trace

            from agent_system.parsers.react import ReactToolParser

            if not action_content.get("action_complete", True):
                return [], []
            emitted = action_content.get("emitted_token_ids")
            if emitted is None:
                raise ValueError("webshop_raw requires actual emitted_token_ids; raw decisions cannot prove completion")
            # Validate the actual rollout text, not a reconstruction that drains
            # pending forced tokens and can make a truncated command executable.
            trace = await loop.run_in_executor(
                None, lambda: build_unified_policy_trace(action_config, raw_token_ids, emitted)
            )
            heads = [
                token for token, selected in zip(trace["response_dyad"], trace["tool_mask"])
                if selected
            ]
            if not heads:
                return [], []
            if len(heads) != 1:
                raise ValueError("webshop_raw requires exactly one action head selection")
            selected = next(
                (name for name, token in action_config["action_name_ids"].items() if token == heads[0]), None
            )
            parser = ReactToolParser(self.tokenizer)
            parser.TOOL_NAME = "webshop_action"
            _, function_calls = await parser.extract_tool_calls(emitted)
            if not function_calls:
                return [], []
            command = json.loads(function_calls[0].arguments)["raw_action"]
            native = regex.fullmatch(r"(search|click)\[(.+)\]", command, regex.DOTALL)
            if native is None:
                return [], []
            if native.group(1) != selected:
                raise ValueError("WebShop emitted command does not match the selected action head")
            return function_calls, function_calls

        # Dyad raw-command (env_serialize=<env>_raw): the action head only picks the action name, the params are
        # one whole free-text segment. What is joined together is that env's raw command, passed through the
        # projector's raw_action (bypassing action_templates, so the free string is not split back into
        # object/receptacle for alfworld, nor into expression/answer for the calculator).
        #
        # alfworld_raw and calc_raw share this branch because the join is env-agnostic (surface_form + free
        # string) and both projectors take the result under `raw_action`. Keeping them apart would be two
        # copies of the same code drifting independently.
        if action_config.get("router") == "unified" and action_config.get("env_serialize") in _RAW_ENV_SERIALIZE:
            from agent_system.policies.dyad.actions.action_router import replay_unified_actions

            actions = await loop.run_in_executor(
                None,
                lambda: replay_unified_actions(action_config, raw_token_ids, tokenizer=self.tokenizer),
            )
            function_calls: list[FunctionCall] = []
            if actions:
                command = unified_action_to_raw_command(action_config, actions[0])
                function_calls.append(
                    FunctionCall(
                        name=actions[0]["name"],
                        arguments=json.dumps({"raw_action": command}, ensure_ascii=False),
                    )
                )
            dyad_logger.debug(
                f"[Unified/{action_config.get('env_serialize')}] parsed {len(actions)} action(s), "
                f"first: {actions[:1]}"
            )
            return function_calls, function_calls

        # Dyad raw-command for CodeGym (env_serialize=codegym_raw): the action head force-writes
        # `[{"name": "<Fn>", "parameters": ` and the model writes the rest of the paper's B2 block
        # (`{args}}]`) as free text -- so the text between the markers is byte-for-byte the grpo_react
        # baseline's call. Reuse grpo_react's own parsing rather than re-deriving {name, parameters} from
        # the decisions: the raw-command schema requires the two conditions to emit the same block, so they must
        # not have two parsers that can disagree.
        #
        # Must sit before the generic codegym branch below: the raw-command schema has mode:codegym too, and that branch
        # would replay it through the slot-based serializer, which has no slots to fill here.
        if action_config.get("env_serialize") == "codegym_raw":
            from agent_system.policies.dyad.actions.action_router import replay_unified_actions

            from agent_system.parsers.codegym import codegym_block_to_function_calls

            actions = await loop.run_in_executor(
                None,
                lambda: replay_unified_actions(action_config, raw_token_ids, tokenizer=self.tokenizer),
            )
            function_calls = (
                codegym_block_to_function_calls(
                    _strip_exit_marker(action_config, unified_action_to_raw_command(action_config, actions[0]))
                )
                if actions
                else []
            )
            dyad_logger.debug(
                f"[Unified/codegym_raw] parsed {len(actions)} action(s), first: {actions[:1]}"
            )
            return function_calls, function_calls

        # Unified ActionRouter + alfworld env-serialize: replay the decision sequence back into alfworld env
        # commands (go to X / take Y from Z ...). Decoupled from codegym serialization (alfworld keeps
        # mode:codegym only to reuse head-init/compilation; env-serialize is selected explicitly by
        # env_serialize:alfworld).
        if action_config.get("router") == "unified" and action_config.get("env_serialize") == "alfworld":
            from agent_system.policies.dyad.actions.action_router import replay_unified_actions

            actions = await loop.run_in_executor(
                None,
                lambda: replay_unified_actions(action_config, raw_token_ids, tokenizer=self.tokenizer),
            )
            # Stateful multi-turn env: only the first action is executed per turn, the rest are left for the
            # next turn to re-decide on the new observation.
            function_calls: list[FunctionCall] = []
            if actions:
                call = unified_action_to_alfworld_call(action_config, actions[0])
                function_calls.append(
                    FunctionCall(
                        name=call["action name"],
                        arguments=json.dumps(call["argument"], ensure_ascii=False),
                    )
                )
            dyad_logger.debug(f"[Unified/alfworld] parsed {len(actions)} action(s), first: {actions[:1]}")
            return function_calls, function_calls

        # CodeGym-compatible serialization still uses the flat schema fields and
        # FlatActionRouter internally, even though normal rollout uses ActionRouter.
        if action_content.get("codegym") or action_config.get("mode") == "codegym":
            from agent_system.policies.dyad.actions.codegym_serialization import decisions_to_codegym_actions

            actions = await loop.run_in_executor(
                None,
                lambda: decisions_to_codegym_actions(
                    action_config, raw_token_ids, tokenizer=self.tokenizer
                ),
            )
            # CodeGym is a stateful multi-turn env: only [one] action is executed at a time (aligned with the
            # single-action semantics of ALFWorld). One rollout turn may write several actions, but the agent
            # loop executes up to max_parallel_calls tool calls in parallel -- running several actions
            # consecutively/in parallel against a stateful env would corrupt the env state. So only the first
            # one is taken and the rest are re-decided by later turns (based on the new observation).
            # decisions_to_codegym_actions (plural) is kept for debugging / future extension.
            function_calls: list[FunctionCall] = []
            if actions:
                act = actions[0]
                function_calls.append(
                    FunctionCall(
                        name=act["name"],
                        # the complete CodeGym action goes into a dedicated key, from which
                        # CodeGymAdapter.build_action deterministically builds the {"name","parameters"} JSON
                        # string the env expects.
                        arguments=json.dumps({"codegym_action": act}, ensure_ascii=False),
                    )
                )
            dyad_logger.debug(f"[CodeGym] parsed {len(actions)} action(s), executing first: {actions[:1]}")
            return function_calls, function_calls

        # Reaching here means action_config is neither unified(env_serialize:alfworld) nor codegym;
        # every env currently goes through one of the two unified branches above.
        raise NotImplementedError(
            "dyad_extract_tool_calls only supports action_config with router:unified(env_serialize:alfworld) "
            f"or mode:codegym; got router={action_config.get('router')!r} "
            f"mode={action_config.get('mode')!r} env_serialize={action_config.get('env_serialize')!r}"
        )
