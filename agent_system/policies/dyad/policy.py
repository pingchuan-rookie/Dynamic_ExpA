"""Expanded-action sampling context and exact training replay for each decision."""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Add expanded-action sampling context and exact replay to the shared environment loop.
# Extension point: EnvironmentStepAgentLoop -> make_step_policy("dyad")
from __future__ import annotations

from dataclasses import asdict
import os

from agent_system.policies.dyad.actions.policy_replay import build_policy_trace, validate_policy_trace
from agent_system.policies.text import TextStepPolicy


async def write_runtime_action_context(tokenizer, vocab_size, context):
    from agent_system.policies.dyad.actions.task_context import action_capacity, context_directory, context_path
    from agent_system.policies.dyad.models import llm_encoder_config_from_env
    from agent_system.policies.dyad.models.action_head_factory import encoder_prompts
    from agent_system.policies.dyad.rollout.encoder_worker import get_or_create_encoder_actor

    capacity = action_capacity()
    tools = context["action_tools"]
    if context.get("protocol") == "dive":
        from agent_system.policies.dyad.actions.dive_schema import compile_tools
        config = compile_tools(tokenizer, vocab_size, tools, capacity)
    elif context.get("protocol") == "native_tools":
        from agent_system.policies.dyad.actions.native_tools import compile_tools
        config = compile_tools(tokenizer, vocab_size, tools, capacity)
    else:
        from agent_system.policies.dyad.inference.schema import compile_tools
        virtual = [tool for tool in tools if tool["function"]["name"] == "respond"]
        if len(virtual) > 1:
            raise ValueError("Runtime action schema contains duplicate respond actions")
        config = compile_tools(
            tokenizer, vocab_size, [tool for tool in tools if tool["function"]["name"] != "respond"],
            capacity, respond_definition=virtual[0]["function"] if virtual else None,
        )
    cfg = llm_encoder_config_from_env()
    model_path = os.environ.get("MODEL_PATH") or os.environ["DYAD_MODEL_SOURCE_PATH"]
    identity = {"encoder": asdict(cfg), "model_path": model_path,
                "tokenizer": str(getattr(tokenizer, "name_or_path", "")), "vocab_size": vocab_size}
    prompts = encoder_prompts(config, description=cfg.description)
    path = context_path(context_directory(), config, prompts, identity)
    actor = get_or_create_encoder_actor(cfg, model_path)
    path = await actor.write_action_context.remote(str(path), config, prompts, identity)
    return config, path


async def write_codegym_action_context(tokenizer, vocab_size, messages, ability, *, step_protocol_version=None):
    from agent_system.policies.dyad.actions.codegym_tasks import task_schema, compile_task, context_path
    from agent_system.policies.dyad.models import llm_encoder_config_from_env
    from agent_system.policies.dyad.models.action_head_factory import encoder_prompts
    from agent_system.policies.dyad.rollout.encoder_worker import get_or_create_encoder_actor

    capacity = int(os.environ["DYAD_CODEGYM_ACTION_CAPACITY"])
    config = compile_task(tokenizer, vocab_size, task_schema(messages, ability), capacity,
                          step_protocol_version=step_protocol_version)
    cfg = llm_encoder_config_from_env()
    prompts = encoder_prompts(config, description=cfg.description)
    path = context_path(os.environ["DYAD_CODEGYM_CONTEXT_DIR"], config, prompts)
    actor = get_or_create_encoder_actor(cfg, os.environ.get("MODEL_PATH") or os.environ["DYAD_MODEL_SOURCE_PATH"])
    path = await actor.write_codegym_context.remote(str(path), config, prompts)
    return config, path


class DyadStepPolicy(TextStepPolicy):
    def __init__(self, action_config=None, context_path=None):
        self.action_config = action_config
        self.context_path = context_path

    async def prepare(self, loop, session, kwargs):
        if self.action_config is None:
            from transformers import AutoConfig
            from agent_system.policies.dyad.actions.schema_config import load_action_config_by_name, load_default_action_config
            from agent_system.utils.hf_config import text_vocab_size

            model = loop.config.actor_rollout_ref.model
            vocab_size = text_vocab_size(AutoConfig.from_pretrained(
                model.path, trust_remote_code=model.get("trust_remote_code", False)))
            validation_schema = os.environ.get("DYAD_VAL_ACTION_YAML") if kwargs.get("validate") else None
            self.action_config = (load_action_config_by_name(loop.tokenizer, vocab_size, validation_schema)
                                  if validation_schema else load_default_action_config(loop.tokenizer, vocab_size))
        vocab_size = self.action_config["num_embeddings_size"]
        if getattr(session, "context", {}).get("protocol") in {"dive", "native_tools", "t2bench"}:
            self.action_config, self.context_path = await write_runtime_action_context(
                loop.tokenizer, vocab_size, session.context)
        elif session.environment == "codegym":
            self.action_config, self.context_path = await write_codegym_action_context(
                loop.tokenizer, vocab_size, session.initial_messages, session.reset_spec["env_str"],
                step_protocol_version=2 if loop.protocol_version == 2 else None)
        if session.environment in {"alfworld", "webshop", "gsm8k"}:
            markers = self.action_config["markers"]
            expected_enter = loop.tokenizer.encode("<action>", add_special_tokens=False)
            expected_exit = loop.tokenizer.encode("</action>", add_special_tokens=False)
            if markers["enter_seq"] != expected_enter or markers["exit_seq"] != expected_exit:
                raise ValueError(
                    f"Shared {session.environment} step prompts require <action>...</action>; "
                    "select a lowercase action schema for both sampling and replay, "
                    "not the legacy <Action> schema"
                )

    def sampling_params(self, params, step_index, max_steps):
        result = dict(params)
        if self.context_path:
            result["extra_args"] = {**result.get("extra_args", {}),
                                    "dyad_action_context": self.context_path,
                                    "dyad_codegym_context": self.context_path}
        result.update(_dyad_rollout_turn=step_index + 1, _dyad_max_turns=max_steps)
        return result

    def trace(self, generated):
        content = getattr(generated, "action_content", None)
        if not isinstance(content, (list, tuple)) or len(content) != 1 or not isinstance(content[0], dict) or len(content[0]) != 1:
            raise ValueError("A step requires exactly one sampled Dyad trace, including vocabulary-only decisions")
        payload = next(iter(content[0].values()))
        if not isinstance(payload, dict) or not isinstance(payload.get("raw_token_ids"), list):
            raise ValueError("Dyad generation returned an invalid sampled action trace")
        if payload.get("action_config") != self.action_config:
            raise ValueError("Dyad sampling action context differs from replay context")
        trace = build_policy_trace(payload, list(generated.token_ids))
        validate_policy_trace(trace, list(generated.token_ids))
        if trace["action_size"] != self.action_config["total_size"]:
            raise ValueError("Dyad action size changed between sampling and replay")
        fields = {"response_dyad": trace["response_dyad"], "seq_mask": trace["seq_mask"],
                  "tool_mask": trace["tool_mask"], "dyad_allowed_action_ids": trace["allowed_action_ids"],
                  "dyad_action_size": trace["action_size"]}
        return fields, payload

    def selected_action_names(self, payload, environment):
        if environment == "t2bench":
            from agent_system.policies.dyad.inference.schema import MissingExpandedHeadSelection, verify_trace
            try:
                return (verify_trace(payload, self.action_config)["name"],)
            except MissingExpandedHeadSelection:
                return ()
        if environment not in {"dive", "swebench_verified"}:
            return None
        from agent_system.policies.dyad.actions.native_tools import verify_trace

        selections = verify_trace(payload, self.action_config)
        return tuple(selection["name"] for selection in selections)

    @property
    def extra_fields(self):
        fields = {"action_interface": "dyad"}
        if self.context_path:
            fields.update(dyad_action_context=self.context_path, codegym_context=self.context_path)
        return fields
