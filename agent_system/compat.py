"""Resolve saved runtime locations without rewriting checkpoint data or import aliases.

Only explicit runtime fields should pass through these helpers at loading boundaries.
Model, dataset and artifact locations are not runtime resource locations.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_MODULES = {
    "verl.trainer.ppo.agent_steps": "verl_extensions.agent_steps",
    "verl.utils.dataset.resume": "verl_extensions.dataset.resume",
    "verl.utils.vocab_statistics": "verl_extensions.vocab_statistics",
    "agent_system.policies.dyad.training.projector_training": "agent_system.policies.dyad.training.alignment",
    "agent_system.policies.expa": "agent_system.policies.dyad",
    "expa.trainer.step_advantage_adapter": "verl_extensions.agent_steps.advantage_adapter",
    "expa.trainer.step_advantage": "verl_extensions.agent_steps.advantages",
    "expa.trainer.step_batch": "verl_extensions.agent_steps.batch",
    "expa.trainer.step_minibatch": "verl_extensions.agent_steps.minibatch",
    "expa.trainer.step_replay_buffer": "verl_extensions.agent_steps.replay_buffer",
    "expa.trainer.training_protocol": "verl_extensions.agent_steps.protocol",
    "expa.trainer.loss_reduction": "verl_extensions.agent_steps.loss_reduction",
    "expa.trainer.batch_transport": "verl_extensions.agent_steps.batch_transport",
    "expa.trainer.batch_padding_config": "verl_extensions.agent_steps.batch_padding_config",
    "expa.utils.dataset_resume": "verl_extensions.dataset.resume",
    "expa.agent_loop.environment_step_agent_loop": "agent_system.rollout.environment_step_agent_loop",
    "expa.agent_loop.step_environment": "agent_system.environments.step_session",
    "expa.agent_loop.env_session": "agent_system.rollout.env_session",
    "expa.agent_loop.action_recovery": "agent_system.rollout.action_recovery",
    "expa.agent_loop.parsers.expa": "agent_system.policies.dyad.parser",
    "expa.agent_loop.parsers": "agent_system.parsers",
    "expa.environments": "agent_system.environments",
    "expa.rewards": "agent_system.rewards",
    "expa.utils": "agent_system.utils",
    "expa.actions": "agent_system.policies.dyad.actions",
    "expa.models": "agent_system.policies.dyad.models",
    "expa.algorithms": "agent_system.policies.dyad.algorithms",
    "expa.rollout": "agent_system.policies.dyad.rollout",
    "expa.inference": "agent_system.policies.dyad.inference",
    "expa.data": "agent_system.policies.dyad.data",
    "expa.trainer": "agent_system.policies.dyad.training",
}
# Explicit environment relocations resolve both generations of saved paths directly
# to their final modules, without installing old import aliases or re-export trees.
_ENVIRONMENT_MODULES = {
    "base_env_pool": "core.pool",
    "base_local_tool": "core.local_tool",
    "native_tool": "core.native_tool",
    "adapters.base": "core.adapter",
    "adapters.generic": "core.generic",
    "adapters.AlfworldAdapter": "alfworld.adapter.AlfworldAdapter",
    "adapters.BaseEnvAdapter": "core.adapter.BaseEnvAdapter",
    "adapters.GenericAgentGymAdapter": "core.generic.GenericAgentGymAdapter",
    "workers.base_session": "core.session",
    "workers.base_worker": "core.worker",
    "alfworld_env_pool": "alfworld.pool",
    "alfworld_local_tool": "alfworld.tool",
    "adapters.alfworld": "alfworld.adapter",
    "workers.alfworld_session": "alfworld.session",
    "workers.alfworld_worker": "alfworld.worker",
    "codegym_env_pool": "codegym.pool",
    "codegym_local_tool": "codegym.tool",
    "adapters.codegym": "codegym.adapter",
    "workers.codegym_session": "codegym.session",
    "workers.codegym_worker": "codegym.worker",
    "calc_env_pool": "calc.pool",
    "calc_local_tool": "calc.tool",
    "adapters.calc": "calc.adapter",
    "workers.calc_session": "calc.session",
    "workers.calc_worker": "calc.worker",
    "webshop_env_pool": "webshop.pool",
    "webshop_local_tool": "webshop.tool",
    "adapters.webshop": "webshop.adapter",
    "workers.webshop_worker": "webshop.worker",
    "webshop_config": "webshop.config",
    "workers.webshop_backend": "webshop.backend",
    "dive_env_pool": "dive.pool",
    "dive_local_tool": "dive.tool",
    "workers.dive_worker": "dive.worker",
    "dive_dataset": "dive.dataset",
    "tau_env_pool": "tau.pool",
    "tau_local_tool": "tau.tool",
    "adapters.tau": "tau.adapter",
    "workers.tau_worker": "tau.worker",
    "tau_dataset": "tau.dataset",
    "tau_metrics": "tau.metrics",
    "swebench_env_pool": "swebench.pool",
    "swebench_local_tool": "swebench.tool",
    "workers.swebench_session": "swebench.session",
    "workers.swebench_worker": "swebench.worker",
    "swebench_dataset": "swebench.dataset",
    "swebench_metrics": "swebench.metrics",
}
_BACKENDS = frozenset({"alfworld", "codegym", "calc", "webshop", "dive", "tau", "swebench"})
# Saved Python targets keep their original spelling; resolve them at the read boundary.
_ALIGNMENT_MODULES = {
    "agent_system.policies.dyad.models.action_selector": "agent_system.policies.dyad.models.actenc_alignment_model",
    "agent_system.policies.dyad.data.action_dataset": "agent_system.policies.dyad.data.actenc_alignment_dataset",
    "agent_system.policies.dyad.data.parquet_dataset": "agent_system.policies.dyad.data.actenc_alignment_parquet",
}
_ALIGNMENT_OLD = "agent_system.policies.dyad.training.alignment"
_ALIGNMENT_NEW = "agent_system.policies.dyad.training.action_encoder_alignment"
for _module in ("config", "train", "evaluate", "evaluation", "wandb_run"):
    _ALIGNMENT_MODULES[f"{_ALIGNMENT_OLD}.{_module}"] = f"{_ALIGNMENT_NEW}.actenc_alignment_{_module}"
_ALIGNMENT_MODULES[_ALIGNMENT_OLD] = _ALIGNMENT_NEW
_ALIGNMENT_MODULES = dict(sorted(_ALIGNMENT_MODULES.items(), key=lambda item: -len(item[0])))
_MODULES.update(_ALIGNMENT_MODULES)
for _old, _new in _ENVIRONMENT_MODULES.items():
    if _new.split(".", 1)[0] in _BACKENDS:
        _new = "backends." + _new
    for _prefix in ("expa.environments.", "agent_system.environments."):
        _MODULES[_prefix + _old] = "agent_system.environments." + _new
# The intermediate per-environment packages may also occur in saved configs.
# Resolve every supported generation directly, since resolution is one pass.
for _backend in _BACKENDS:
    for _prefix in ("expa.environments.", "agent_system.environments."):
        _MODULES[_prefix + _backend] = "agent_system.environments.backends." + _backend
_MODULES = dict(sorted(_MODULES.items(), key=lambda item: -len(item[0])))
_POLICY_SYMBOLS = {
    "TextStepPolicy": "agent_system.policies.text.TextStepPolicy",
    "ExpaStepPolicy": "agent_system.policies.dyad.policy.DyadStepPolicy",
    "write_runtime_action_context": "agent_system.policies.dyad.policy.write_runtime_action_context",
    "write_codegym_action_context": "agent_system.policies.dyad.policy.write_codegym_action_context",
}
_MODULE_FIELDS = frozenset({"_target_", "class_name", "module", "module_path"})
_RESOURCE_FIELDS = frozenset({
    "tool_config_path", "agent_loop_config_path", "action_yaml", "action_yaml_path",
    "EXPA_ACTION_YAML", "ACTION_YAML", "EXPA_VAL_ACTION_YAML", "TOOL_CONFIG_PATH",
    "EXPA_VAL_TOOL_CONFIG", "DIVE_TOOL_CONFIG_PATH",
    "DYAD_ACTION_YAML", "DYAD_VAL_ACTION_YAML", "DYAD_VAL_TOOL_CONFIG",
})


def _dyad_module_name(name: str) -> str:
    """Rename implementation symbols after resolving a historical module prefix."""
    name = re.sub(r"(?<![A-Za-z])expa(?![a-z])", "dyad", name)
    name = name.replace("projector_training", "alignment")
    name = re.sub(r"(?:ExpA|Expa)(?![a-z])", "Dyad", name)
    for old, new in _ALIGNMENT_MODULES.items():
        for separator in (".", "/"):
            source, target = old.replace(".", separator), new.replace(".", separator)
            if name == source or name.startswith(source + separator) or name == source + ".py":
                return target + name[len(source):]
    return name


def _check_module_target(name: str) -> None:
    parts = name.split(".")
    if (_ROOT.joinpath(*parts) / "__init__.py").is_file():
        return
    # A qualified class/function can follow an existing Python module.
    for end in range(len(parts), 0, -1):
        if _ROOT.joinpath(*parts[:end]).with_suffix(".py").is_file():
            return
    raise ModuleNotFoundError(f"Saved runtime module has no migrated implementation: {name}")


def resolve_module_path(value: str) -> str:
    """Map a known old module or qualified symbol; leave unrelated modules intact."""
    prefix = "pkg://" if value.startswith("pkg://") else ""
    name = value[len(prefix):]
    alignment_target = any(name == old or name.startswith(old + ".") for old in _ALIGNMENT_MODULES)
    legacy_prefixes = (
        "expa.", "agent_system.policies.expa.",
        "verl.trainer.ppo.agent_steps", "verl.utils.dataset.resume", "verl.utils.vocab_statistics",
        "agent_system.policies.dyad.training.projector_training", "agent_system.environments.",
    )
    if not alignment_target and not name.startswith(legacy_prefixes) and name != "agent_system.policies.expa":
        return value
    policy_prefix = "expa.agent_loop.step_policy."
    if name.startswith(policy_prefix):
        symbol = name[len(policy_prefix):]
        if symbol not in _POLICY_SYMBOLS:
            raise ModuleNotFoundError(f"Unknown saved policy symbol: {name}")
        target = _POLICY_SYMBOLS[symbol]
    else:
        target = next((new + name[len(old):] for old, new in _MODULES.items()
                       if name == old or name.startswith(old + ".")), None)
        if target is None:
            if not name.startswith("expa."):
                return value
            raise ModuleNotFoundError(f"No unambiguous runtime migration for {name}")
    target = _dyad_module_name(target)
    _check_module_target(target)
    return prefix + target


def resolve_resource_path(value: str | Path) -> str | Path:
    """Resolve old repository runtime resources, preserving existing external files.

    Foreign absolute locations must contain an explicit Dynamic_ExpA or ExpA_verl
    repository component. Arbitrary paths containing an 'expa' directory are not
    enough evidence to relocate a user's data.
    """
    path = Path(value).expanduser()
    if ".." in path.parts:
        return value
    if path.is_absolute():
        try:
            relative = path.relative_to(_ROOT)
        except ValueError:
            if path.exists():
                return value
            parts = path.parts
            indices = [i for i, part in enumerate(parts) if part in {"dynamic-expa", "Dynamic_ExpA", "ExpA_verl"}]
            if not indices:
                return value
            relative = Path(*parts[indices[-1] + 1:])
    else:
        if path.exists() and not path.resolve().is_relative_to(_ROOT):
            return value
        relative = path
    text = relative.as_posix()
    for old, new in _MODULES.items():
        source = old.replace(".", "/")
        if text == source or text.startswith(source + "/") or text in {source + ".py", source + ".yaml"}:
            target = _ROOT / _dyad_module_name(new.replace(".", "/") + text[len(source):])
            if not target.exists():
                raise FileNotFoundError(f"Saved runtime resource {value!s} has no migrated target: {target}")
            return target if isinstance(value, Path) else str(target)
    return value


def resolve_config_paths(config):
    """Return a detached configuration with only declared runtime fields relocated.

    This returns ordinary containers, also accepting mapping-like saved configs.
    Protocol identities, artifact paths and unrecognized fields remain unchanged.
    """
    def visit(value, field=None):
        if isinstance(value, Mapping):
            return {key: visit(item, key) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, field) for item in value]
        if isinstance(value, tuple):
            return tuple(visit(item, field) for item in value)
        if isinstance(value, str) and field in _MODULE_FIELDS:
            return resolve_module_path(value)
        if isinstance(value, (str, Path)) and field in _RESOURCE_FIELDS:
            return resolve_resource_path(value)
        return copy.deepcopy(value)

    return visit(config)
