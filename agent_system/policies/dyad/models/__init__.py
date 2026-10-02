# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Shared action-model components and lazy public exports.

The carrier of Dyad's transferability (PRINCIPLES P2): the input is an element's natural-language
description, and the mapping itself is env-agnostic, so moving to a new env means feeding the new
descriptions through the same encoder. See ../dynamic-expa-design/algorithm/architecture.md.

LlmActionEncoder reads action prompts; DirectActionHead pools and projects its representations.
MeanProjector is the parameter-free encoder pooler.
"""

from agent_system.policies.dyad.models.encoder_config import LlmEncoderConfig

__all__ = [
    "LlmEncoderConfig",
    "build_action_head",
    "apply_training_schedule",
    "assert_encoder_lm_training_supported",
    "encoder_prompts",
    "EncoderSetting",
    "Backbone",
    "Projector",
    "Representation",
    "Description",
    "TrainingSchedule",
    "EncoderTraining",
    "legal_settings",
    "build_llm_encoder",
    "llm_encoder_config_from_env",
    # The names below are re-exported lazily by __getattr__: importing them eagerly would drag
    # transformers into every process that touches an action config, including the import-light
    # Ray env workers (AGENTS.md section 0.3 -- one stray heavy import there is a worker storm).
    "LlmActionEncoder",
    "DirectActionHead",
    "build_projector",
    "projector_names",
    "describe_action_config",
    "build_action_prompt",
    "build_catalogue",
    "known_forms",
    "RemoteLlmActionEncoder",
    "EncoderActorImpl",
    "get_or_create_encoder_actor",
    "shutdown_encoder_actor",
    "build_remote_encoder",
    "CachedHiddenSource",
]

_LAZY = {
    "build_action_head": "agent_system.policies.dyad.models.action_head_factory",
    "apply_training_schedule": "agent_system.policies.dyad.models.action_head_factory",
    "assert_encoder_lm_training_supported": "agent_system.policies.dyad.models.action_head_factory",
    "encoder_prompts": "agent_system.policies.dyad.models.action_head_factory",
    "EncoderSetting": "agent_system.policies.dyad.models.encoder_config",
    "Backbone": "agent_system.policies.dyad.models.encoder_config",
    "Projector": "agent_system.policies.dyad.models.encoder_config",
    "Representation": "agent_system.policies.dyad.models.encoder_config",
    "Description": "agent_system.policies.dyad.models.encoder_config",
    "TrainingSchedule": "agent_system.policies.dyad.models.encoder_config",
    "EncoderTraining": "agent_system.policies.dyad.models.encoder_config",
    "legal_settings": "agent_system.policies.dyad.models.encoder_config",
    "build_llm_encoder": "agent_system.policies.dyad.models.action_head_factory",
    "llm_encoder_config_from_env": "agent_system.policies.dyad.models.action_head_factory",
    "LlmActionEncoder": "agent_system.policies.dyad.models.action_encoder",
    "DirectActionHead": "agent_system.policies.dyad.models.action_head",
    "build_projector": "agent_system.policies.dyad.models.action_projector",
    "projector_names": "agent_system.policies.dyad.models.action_projector",
    "describe_action_config": "agent_system.policies.dyad.models.action_descriptions",
    # The description dimension. One module per form under descriptions/; these three are the whole
    # surface the rest of the codebase uses, so a new form changes nothing outside that package.
    "build_action_prompt": "agent_system.policies.dyad.models.action_descriptions",
    "build_catalogue": "agent_system.policies.dyad.models.action_descriptions",
    "known_forms": "agent_system.policies.dyad.models.action_descriptions",
    # remote pulls in ray as well as transformers; it must stay lazy for the same reason.
    "RemoteLlmActionEncoder": "agent_system.policies.dyad.rollout.encoder_worker",
    "EncoderActorImpl": "agent_system.policies.dyad.rollout.encoder_worker",
    "get_or_create_encoder_actor": "agent_system.policies.dyad.rollout.encoder_worker",
    "shutdown_encoder_actor": "agent_system.policies.dyad.rollout.encoder_worker",
    "build_remote_encoder": "agent_system.policies.dyad.rollout.encoder_worker",
    "CachedHiddenSource": "agent_system.policies.dyad.models.encoder_cache",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
