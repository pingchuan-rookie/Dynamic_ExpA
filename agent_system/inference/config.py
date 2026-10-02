"""Model selection and vLLM resources shared by Python and HTTP inference."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypedDict


class ActionArguments(TypedDict, total=False):
    """Request-local action definitions; empty arguments leave tools optional."""

    schema: dict[str, Any]
    schema_name: str
    protocol: Literal["native_tools", "tau_json"]


@dataclass
class InferenceConfig:
    """Load exactly one policy source; token budgets are measured in tokens."""

    model_path: str | None = None
    checkpoint: str | None = None
    projector_init: str | None = None
    model_config: str | None = None
    model: str = "dyad-model"
    tensor_parallel_size: int = 1
    context_length: int = 40960
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.7
    action_capacity: int = 256
    encoder_device: str = "cpu"
    restore_dir: str | None = None
    safetensors_load_strategy: Literal["prefetch", "lazy", "eager"] = "prefetch"
    action_interface: Literal["text", "dyad"] = field(default="text", init=False)
    source_root: str = field(default="", init=False)
    source: dict[str, Any] | None = field(default=None, init=False)


def resolve_source(config: InferenceConfig) -> InferenceConfig:
    """Resolve saved weight identity without HTTP, GPUs or a training runtime."""
    args = replace(config)
    if sum(bool(value) for value in (args.model_path, args.checkpoint, args.projector_init)) != 1:
        raise ValueError("Choose exactly one of model_path, checkpoint or projector_init")
    for name in ("tensor_parallel_size", "context_length", "max_num_seqs", "action_capacity"):
        value = getattr(args, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    if args.safetensors_load_strategy not in {"prefetch", "lazy", "eager"}:
        raise ValueError("safetensors_load_strategy must be prefetch, lazy or eager")
    for name in ("model_path", "checkpoint", "model_config", "restore_dir"):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(Path(value).expanduser().resolve()))
    if args.model_path and (Path(args.model_path) / "actor").is_dir():
        args.checkpoint, args.model_path = args.model_path, None
    if args.model_config and not (args.checkpoint or args.projector_init):
        raise ValueError("model_config requires checkpoint or projector_init")
    if args.checkpoint:
        from agent_system.policies.dyad.checkpoint_compat import read_saved_model_config

        config_path = (
            Path(args.model_config) if args.model_config else Path(args.checkpoint).parent / "model_config.json"
        )
        saved = read_saved_model_config(config_path)
        args.model_config = str(config_path)
        args.action_interface = "dyad" if saved.get("algo") in {"dyad", "dyad-grpo", "dyad-gigpo"} else "text"
    elif args.projector_init:
        args.action_interface = "dyad"
    if args.action_interface == "dyad":
        from agent_system.policies.dyad.inference.source import source_metadata

        args.source = source_metadata(args)
        args.source_root = args.source["source"]
    else:
        path = args.checkpoint or args.model_path
        assert path is not None  # Exactly one source was validated above.
        args.source_root = path
        if not args.checkpoint:
            from agent_system.inference.policy_identity import inspect_policy_weights

            inspect_policy_weights(Path(args.source_root))
    if args.checkpoint and not args.restore_dir:
        from uuid import uuid4

        from agent_system.utils.artifact_paths import artifact_root

        args.restore_dir = str(artifact_root(Path(__file__).resolve().parents[2]) / "outputs/inference" / uuid4().hex)
    return args


def validate_actions(tools: list[dict[str, Any]], args: ActionArguments) -> None:
    """Reject ambiguous action requests before encoding or submitting to vLLM."""
    if not isinstance(tools, list) or any(
        not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict)
        for tool in tools
    ):
        raise ValueError("tools must be a list of function definitions")
    if not isinstance(args, dict) or set(args) - {"schema", "schema_name", "protocol"}:
        raise ValueError("Action args accepts schema, schema_name and protocol")
    if args.get("protocol", "native_tools") not in ("native_tools", "tau_json"):
        raise ValueError("args.protocol must be native_tools or tau_json")
    if "schema" in args and (not isinstance(args["schema"], dict) or not args["schema"]):
        raise ValueError("args.schema must be a nonempty action schema")
    if "schema_name" in args and (not isinstance(args["schema_name"], str) or not args["schema_name"]):
        raise ValueError("args.schema_name must be a nonempty bundled schema name")
    if args.get("schema") and args.get("schema_name"):
        raise ValueError("Choose schema or schema_name, not both")
