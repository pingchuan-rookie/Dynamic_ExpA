"""Environment-independent dynamic action context and legacy compatibility."""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Keep task schema and encoder identity attached to dynamic sampling/replay contexts.
# Extension point: DyadStepPolicy -> encoder actor -> Dyad engine replay
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def dynamic_actions_enabled():
    return os.environ.get("DYAD_DYNAMIC_ACTIONS") == "1" or os.environ.get("DYAD_CODEGYM_ALL") == "1"


def action_capacity():
    value = os.environ.get("DYAD_ACTION_CAPACITY") or os.environ.get("DYAD_CODEGYM_ACTION_CAPACITY")
    if value is None or int(value) <= 0:
        raise ValueError("Dynamic actions require a positive DYAD_ACTION_CAPACITY")
    return int(value)


def context_directory():
    value = os.environ.get("DYAD_ACTION_CONTEXT_DIR") or os.environ.get("DYAD_CODEGYM_CONTEXT_DIR")
    if not value:
        raise ValueError("Dynamic actions require DYAD_ACTION_CONTEXT_DIR")
    return value


def bootstrap_schema(capacity):
    from agent_system.policies.dyad.actions.codegym_tasks import bootstrap_schema as legacy_bootstrap
    raw = legacy_bootstrap(capacity)
    raw["action_capacity"] = int(capacity)
    return raw


def context_path(directory, cfg, prompts, identity):
    if not identity:
        raise ValueError("Action context requires encoder/tokenizer identity")
    key = hashlib.sha256(json.dumps([cfg, prompts, identity], sort_keys=True).encode()).hexdigest()
    return Path(directory) / (key + ".pt")


def load_context(path):
    from agent_system.policies.dyad.actions.codegym_tasks import load_context as legacy_load
    return legacy_load(path)


def context_head(*args, **kwargs):
    from agent_system.policies.dyad.actions.codegym_tasks import context_head as legacy_head
    return legacy_head(*args, **kwargs)


def packed_action_logits(*args, **kwargs):
    from agent_system.policies.dyad.actions.codegym_tasks import packed_action_logits as legacy_logits
    return legacy_logits(*args, **kwargs)
