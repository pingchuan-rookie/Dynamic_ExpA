"""Load and compile action schemas without importing model or execution backends."""

from agent_system.compat import resolve_resource_path
from agent_system.policies.dyad.actions.task_context import dynamic_actions_enabled, action_capacity
from pathlib import Path
from typing import Any

import copy
import os


import yaml


SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"


def resolve_schema_path(yaml_name: str) -> Path:
    """schema file name -> absolute path. **The repo's only schemas/ locator.**

    config.py and rollout/dyad_gpu_worker.py each used to hard-code `__file__.parent/"schemas"`,
    which broke as soon as the directory moved (hit twice already). Every new schema consumer
    must call this function.
    """
    return SCHEMAS_DIR / Path(resolve_resource_path(yaml_name))


def load_action_config_from_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """
    Read the action config from a YAML file and run the most basic field validation.
    """
    yaml_path = Path(resolve_resource_path(yaml_path))

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"YAML file is empty: {yaml_path}")

    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML content must be a dict, got {type(data)}")

    # Unified new format (router=unified): shaped as markers/actions/head/argument_order/value_sets,
    # handed to compile_action_schema. Must sit before the codegym/alfworld validation (the new
    # format carries no actions_schema/enter_and_exit_str and would be wrongly rejected by the old checks).
    if data.get("router") == "unified":
        for key in ["env_name", "actions", "markers"]:
            if key not in data:
                raise KeyError(f"Missing required key in unified YAML: {key}")
        if not isinstance(data["actions"], dict):
            raise TypeError("actions must be a dict")
        return data

    # CodeGym mode: a different schema shape (routing + per-param argument_kind, no enter_and_exit_str),
    # so only the common fields are validated and compile_flat_schema does the compiling.
    # Fully isolated from the ALFWorld validation.
    if data.get("mode") == "codegym":
        for key in ["env_name", "actions_schema"]:
            if key not in data:
                raise KeyError(f"Missing required key in codegym YAML: {key}")
        if not isinstance(data["actions_schema"], dict):
            raise TypeError("actions_schema must be a dict")
        return data

    required_keys = [
        "env_name",
        "actions_schema",
        "enter_and_exit_str",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key in YAML: {key}")

    if not isinstance(data["actions_schema"], dict):
        raise TypeError("action_schemas must be a dict")

    if not isinstance(data["enter_and_exit_str"], dict):
        raise TypeError("enter_and_exit_str must be a dict")

    for key in ["enter_str", "exit_param_str"]:
        if key not in data["enter_and_exit_str"]:
            raise KeyError(f"Missing key in enter_and_exit_str: {key}")

    return data


# New runs follow the common lowercase ReAct prompt; explicit legacy paths stay unchanged.
DEFAULT_ACTION_CONFIG_PATH = SCHEMAS_DIR / "alfworld" / "shared_step_v2.yaml"


# In-process action_config compilation cache. compile_action_schema (tokenizer.decode + a full
# vocabulary scan) costs ~1.46s per call, while AgentLoopWorker hydra-instantiates one
# DyadToolAgentLoop per trajectory -> __init__ recompiles every time; 32 trajectories serialize to
# ~46.7s on the event-loop thread (holding the GIL) and starve every await in rollout (measured:
# the dyad gen 66s vs grpo 16.5s gap was exactly this). The compilation result is constant for a
# given (yaml, vocab_size) (the tokenizer is fixed within a process), so it is compiled once and
# later calls return a deep copy (~0.05ms), avoiding recompilation. Cross-env val uses a different
# yaml_name -> a different key.
_ACTION_CONFIG_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


# Sibling files separate MCP semantics, value inventories, and surface serialization.
MCP_FILENAME = "mcp.json"
VALUES_FILENAME = "values.yaml"


def resolve_mcp_path(yaml_path: Path | str) -> Path | None:
    """Locate the MCP definition associated with a surface-form YAML file.

    Try sibling mcp.json, then ../mcp/<stem>.json for per-task schemas. Return None
    for legacy single-file schemas that embed their own definitions.
    """
    yaml_path = Path(resolve_resource_path(yaml_path))
    same_dir = yaml_path.parent / MCP_FILENAME
    if same_dir.exists():
        return same_dir
    per_problem = yaml_path.parent.parent / "mcp" / f"{yaml_path.stem}.json"
    if per_problem.exists():
        return per_problem
    return None


def compile_schema_file(tokenizer, vocab_size: int, yaml_path: Path | str) -> dict[str, Any]:
    """Compile a schema file together with its MCP definition and value sets.

    All file-based callers use this entrypoint so semantic definitions cannot be
    silently omitted. In-memory schema generators may call the compiler directly.
    """
    yaml_path = Path(resolve_resource_path(yaml_path))
    raw = load_action_config_from_yaml(yaml_path)
    from agent_system.policies.dyad.actions.schema_compiler import compile_action_schema

    mcp_path = resolve_mcp_path(yaml_path)
    mcp = values = None
    values_path = None
    if mcp_path is not None:
        from agent_system.policies.dyad.actions.mcp_source import load_mcp, load_values

        mcp = load_mcp(mcp_path)
        # Missing inventories are valid for open slots; compilation checks referenced closed sets.
        values_path = mcp_path.parent / VALUES_FILENAME
        values = load_values(values_path)
    cfg = compile_action_schema(tokenizer, vocab_size, raw, mcp=mcp, values=values)
    cfg["_parse_yaml_path"] = str(yaml_path)
    cfg["_mcp_path"] = str(mcp_path) if mcp_path else ""
    cfg["_values_path"] = str(values_path) if values_path and values_path.exists() else ""
    return cfg


def load_action_config_by_name(tokenizer, vocab_size: int, yaml_name: str | None = None) -> dict[str, Any]:
    """Load and compile an environment schema, recording its source YAML path.

    Cache by (yaml_path, vocab_size) to avoid recompiling per trajectory. The source
    path lets parsers reload the correct schema for cross-environment validation.
    MCP supplies action semantics; surface YAML controls context serialization.
    """
    yaml_path = resolve_schema_path(yaml_name) if yaml_name else DEFAULT_ACTION_CONFIG_PATH
    _cache_key = (str(yaml_path), int(vocab_size))
    _cached = _ACTION_CONFIG_CACHE.get(_cache_key)
    if _cached is not None:
        # Return a deep copy so callers' potential in-place edits cannot pollute the cached object.
        return copy.deepcopy(_cached)
    raw = load_action_config_from_yaml(yaml_path)
    if not (isinstance(raw, dict) and (raw.get("router") == "unified" or raw.get("mode") == "codegym")):
        raise NotImplementedError(
            f"Only action yaml with router:unified or mode:codegym is supported; {getattr(yaml_path, 'name', yaml_path)} satisfies neither."
            " The old ALFWorld build_action_config path was removed along with the unified ActionRouter migration."
        )
    cfg = compile_schema_file(tokenizer, vocab_size, yaml_path)
    _ACTION_CONFIG_CACHE[_cache_key] = cfg
    return copy.deepcopy(cfg)


def load_default_action_config(tokenizer, vocab_size: int) -> dict[str, Any]:
    # Reads the same DYAD_ACTION_YAML as the rollout side (dyad_gpu_worker), so the policy LLM backbone head's
    # action_config matches rollout's action-space layout exactly (same yaml -> same extended-id order).
    # mode==codegym takes the CodeGym compiler; otherwise the original ALFWorld path, logic unchanged.
    if dynamic_actions_enabled():
        from agent_system.policies.dyad.actions.task_context import bootstrap_schema
        from agent_system.policies.dyad.actions.codegym_tasks import compile_task
        capacity = action_capacity()
        key = (f"dynamic_actions:{capacity}", int(vocab_size))
        if key not in _ACTION_CONFIG_CACHE:
            _ACTION_CONFIG_CACHE[key] = compile_task(tokenizer, vocab_size, bootstrap_schema(capacity), capacity)
        return copy.deepcopy(_ACTION_CONFIG_CACHE[key])
    return load_action_config_by_name(tokenizer, vocab_size, os.environ.get("DYAD_ACTION_YAML"))
