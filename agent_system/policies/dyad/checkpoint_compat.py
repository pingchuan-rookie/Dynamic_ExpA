"""Read ExpA checkpoint names as Dyad without rewriting the saved artifacts."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path


def normalize_training_protocol(protocol):
    """Canonicalize method labels while retaining every optimizer/protocol setting."""
    if not isinstance(protocol, Mapping):
        return copy.deepcopy(protocol)
    result = {}
    for key, value in protocol.items():
        target = ("DYAD_" + key[5:] if isinstance(key, str) and key.startswith("EXPA_")
                  else "dyad_" + key[5:] if isinstance(key, str) and key.startswith("expa_") else key)
        if isinstance(value, Mapping):
            value = normalize_training_protocol(value)
        elif key in {"action_interface", "loss_mode", "strategy"} and value == "expa":
            value = "dyad"
        else:
            value = copy.deepcopy(value)
        if target in result and result[target] != value:
            raise ValueError(f"Conflicting legacy and Dyad protocol settings: {target}")
        result[target] = value
    return result


def normalize_saved_model_config(config):
    """Translate only method identities and model-setting keys, preserving paths."""
    if not isinstance(config, Mapping):
        return config
    result = copy.deepcopy(dict(config))
    aliases = {"expa": "dyad", "expa-grpo": "dyad-grpo", "expa-gigpo": "dyad-gigpo"}
    for field in ("algo", "action_interface", "public_alias"):
        if isinstance(result.get(field), str):
            result[field] = aliases.get(result[field], result[field])
    if isinstance(result.get("model"), Mapping):
        model = {}
        for key, value in result["model"].items():
            target = "DYAD_" + key[5:] if isinstance(key, str) and key.startswith("EXPA_") else key
            if target == "DYAD_TRAINING_SCHEDULE":
                value = {"joint": "joint_optimization", "encoder_only": "frozen_llm_adaptation"}.get(value, value)
            if target in model and model[target] != value:
                raise ValueError(f"Conflicting legacy and Dyad checkpoint settings: {target}")
            model[target] = value
        result["model"] = model
    if "training_protocol" in result:
        result["training_protocol"] = normalize_training_protocol(result["training_protocol"])
    return result


def read_saved_model_config(path):
    return normalize_saved_model_config(json.loads(Path(path).read_text()))


def normalize_parameter_state(state):
    """Translate registered-head names in model/optimizer state without copying tensors."""
    def name(value):
        if not isinstance(value, str):
            return value
        return ".".join("dyad_residual_head" if p == "expa_residual_head" else p
                        for p in value.split("."))

    def visit(value):
        if isinstance(value, Mapping):
            result = copy.copy(value)
            result.clear()
            for key, item in value.items():
                target = name(key)
                if target in result:
                    raise ValueError(f"Conflicting legacy and Dyad parameter names: {target}")
                result[target] = visit(item)
            if hasattr(value, "_metadata"):
                result._metadata = visit(value._metadata)
            return result
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, tuple):
            return tuple(visit(item) for item in value)
        return name(value)

    return visit(state)
