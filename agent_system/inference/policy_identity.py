"""Read safe policy weight headers and optionally verify a portable export manifest.

This module never imports Torch, deserializes pickle, or constructs model modules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _json(path):
    with path.open() as stream:
        return json.load(stream, object_pairs_hook=_object)


def _weight_name(name):
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
        raise ValueError("Policy index must reference local safetensors filenames without directories")
    return name


def _reject_non_policy(name):
    parts = name.lower().split(".")
    blocked = {
        "dyad_residual_head",
        "action_head",
        "action_encoder",
        "actionencoder",
        "encoder_lm",
        "projector",
        "projector_state",
        "dyad_projector",
    }
    if any(part in blocked or part.startswith("dyad_") for part in parts):
        raise ValueError(f"Policy-only evaluation rejects action encoder/projector tensor: {name}")


def _verify_export(path, manifest_path):
    manifest = _json(manifest_path)
    required_true = ("policy_only", "dtype_preserved", "policy_strict_load", "serialized_tensor_equality_verified")
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or any(manifest.get(key) is not True for key in required_true)
        or manifest.get("action_encoder_loaded") is not False
        or manifest.get("projector_loaded") is not False
    ):
        raise ValueError("Incomplete or unsupported policy export manifest")
    files = manifest.get("artifact_files_sha256")
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()} - {manifest_path.name}
    if not isinstance(files, dict) or set(files) != actual:
        raise ValueError("Policy export artifact inventory mismatch")
    for name, digest in files.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid policy export artifact filename or digest")
        if _digest(path / name) != digest:
            raise ValueError(f"Policy export artifact hash mismatch: {name}")
    return {
        "source_checkpoint": manifest.get("source_checkpoint"),
        "training_algorithm": manifest.get("training_algorithm"),
        "world_size": manifest.get("world_size"),
        "policy_tensor_count": manifest.get("policy_tensor_count"),
        "manifest_sha256": _digest(manifest_path),
        "artifact_hashes_verified": True,
        "source_checkpoint_reverified": False,
    }


def training_identity(config):
    """Resolve saved method names while rejecting contradictory optional axes."""
    algorithm = config.get("algo")
    model = config.get("model") or {}
    if not isinstance(model, dict):
        raise ValueError("Training model metadata must be an object")
    methods = {
        "grpo_react": ("react", "grpo"),
        "gigpo": ("react", "gigpo"),
        "dyad-grpo": ("dyad", "grpo"),
        "dyad-gigpo": ("dyad", "gigpo"),
    }
    method = algorithm
    if algorithm == "dyad":
        method = "dyad-" + str(config.get("adv_estimator", model.get("DYAD_ADV_ESTIMATOR", "grpo")))
    if method not in methods:
        raise ValueError("Unsupported saved training algorithm")
    interface, estimator = methods[method]
    axes = {"action_interface", "adv_estimator"}
    present = axes.intersection(config)
    if present and (present != axes or (config["action_interface"], config["adv_estimator"]) != (interface, estimator)):
        raise ValueError("Saved training axes conflict with algorithm identity")
    if interface == "dyad" and model.get("DYAD_ADV_ESTIMATOR", estimator) != estimator:
        raise ValueError("Saved Dyad estimator conflicts with algorithm identity")
    return {"training_method": method, "action_interface": interface, "adv_estimator": estimator}


def inspect_policy_weights(path):
    """Validate the complete safetensors inventory and reject non-policy modules."""
    from safetensors import safe_open

    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError("Policy weights must be an existing HF directory")
    if list(path.glob("*.bin")) or list(path.glob("*.bin.index.json")):
        raise ValueError(
            "Policy evaluation requires safetensors; export trusted native weights "
            "with export_policy.py instead of loading pickle .bin files"
        )
    if list(path.glob("*.pt")) or list(path.glob("*.pth")):
        raise ValueError("Policy-only HF directory must not contain native checkpoint or projector files")
    weights = sorted(path.glob("*.safetensors"))
    if not weights or any(not p.is_file() or p.stat().st_size == 0 for p in weights):
        raise ValueError("Policy safetensors weights are missing or empty")
    indexes = list(path.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError("Policy must have exactly one safetensors index, not ambiguous indexes")
    mapping = None
    if indexes:
        index = _json(indexes[0])
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Policy safetensors index must have a nonempty weight_map")
        filenames = {_weight_name(name) for name in mapping.values()}
        if filenames != {p.name for p in weights}:
            raise ValueError("Policy safetensors inventory differs from index; missing or unindexed shards")
    elif len(weights) != 1:
        raise ValueError("Multiple safetensors files require an explicit complete index")
    seen, inventory = set(), {}
    for weight in weights:
        try:
            with safe_open(weight, framework="numpy") as handle:
                keys = set(handle.keys())
                if not keys:
                    raise ValueError(f"Empty policy safetensors shard: {weight.name}")
                if keys & seen:
                    raise ValueError("Duplicate policy tensor keys across safetensors shards")
                if mapping is not None and keys != {name for name, shard in mapping.items() if shard == weight.name}:
                    raise ValueError(f"Policy safetensors header differs from index: {weight.name}")
                for name in keys:
                    _reject_non_policy(name)
                    handle.get_slice(name).get_shape()
                seen.update(keys)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"Invalid policy safetensors file: {weight.name}") from error
        stat = weight.stat()
        inventory[weight.name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "tensor_count": len(keys)}
    result = {
        "weight_inventory": inventory,
        "verification": (
            "complete safetensors header/index validation; not model-architecture or tensor-value equivalence"
        ),
    }
    manifest = path / "policy_export.json"
    if manifest.exists():
        provenance = _verify_export(path, manifest)
        if provenance.get("policy_tensor_count") != len(seen):
            raise ValueError("Policy export tensor count differs from safetensors headers")
        result["policy_export"] = provenance
        result["verification"] = (
            "complete safetensors header/index validation and export artifact hashes; native source not reverified"
        )
    return result
