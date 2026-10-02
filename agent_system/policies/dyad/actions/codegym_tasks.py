"""Per-task CodeGym schemas and shared frozen-encoder artifacts.

Action IDs are local to a task. All masks have the run's maximum width; unused
columns are never candidates. Sampling and replay read the identical artifact.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Build task-specific schemas and padded action heads from preserved sampling contexts.
# Extension point: task_context / DyadStepPolicy / DyadFSDPEngineWithLMHead._prepare_codegym_heads
from __future__ import annotations

import ast
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path


def _property(annotation):
    name = ast.unparse(annotation) if annotation is not None else ""
    kind = {"int": "integer", "float": "number", "bool": "boolean", "str": "string",
            "list": "array", "dict": "object"}.get(name.split("[")[0])
    return {"type": kind} if kind else {"description": name or "untyped argument"}


def task_action_names(messages):
    source = "\n".join(str(m.get("content", "")) for m in messages
                       if m.get("role") == "system")
    names = tuple(dict.fromkeys(re.findall(r"(?m)^\s*(?:async\s+)?def\s+(\w+)\s*\(", source)))
    if not names:
        raise ValueError("CodeGym task has no function declarations")
    return names


def task_schema(messages, ability=None, envs_dir=None):
    """Use the exact task's source signatures (some prompt signatures are truncated)."""
    names = task_action_names(messages)
    if ability:
        import os
        from agent_system.environments.backends.codegym.worker import parse_codegym_env_str
        filename, class_name, _ = parse_codegym_env_str(ability)
        directory = Path(envs_dir or os.environ.get("CODEGYM_ENVS_DIR") or
                         Path(__file__).resolve().parents[4] / "data/codegym/dataset/envs/codegym_v1")
        path = directory / filename
        if path.parent.resolve() != directory.resolve():
            raise ValueError("CodeGym environment filename must stay within its source directory")
        return _schema_from_source(str(path), class_name, names)
    # Complete declarations also support small standalone fixtures without environment files.
    source = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    cleaned = source.split("========================")[0].replace("Function:", "")
    return _schema_from_nodes(ast.parse(cleaned).body, names)


@lru_cache(maxsize=128)
def _schema_from_source(path, class_name, names):
    source = Path(path).read_text()
    if not re.search(r"(?m)^class " + re.escape(class_name) + r"[(:]", source):
        raise ValueError(f"CodeGym source {path} does not define {class_name}")
    # Read signatures and docstrings only. Unrelated implementation bodies may contain
    # syntax errors in upstream generated environments; schema extraction must not execute them.
    pattern = r"(?m)^[ \t]+(?P<header>def (?P<name>\w+)\((?:(?!\n[ \t]*def ).)*?\)(?:[ \t]*->[^:\n]+)?:)[ \t]*\n[ \t]+[rRuU]?(?P<quote>\"\"\"|''')(?P<doc>.*?)(?P=quote)"
    nodes = []
    for match in re.finditer(pattern, source, re.DOTALL):
        if match.group("name") not in names:
            continue
        node = ast.parse(match.group("header") + "\n    pass").body[0]
        node.body = [ast.Expr(value=ast.Constant(value=match.group("doc")))]
        nodes.append(node)
    return _schema_from_nodes(nodes, names)


def _schema_from_nodes(nodes, names):
    from agent_system.policies.dyad.actions.codegym_schema import build_codegym_schema
    functions = {n.name: n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    definitions = []
    for name in names:
        if name not in functions:
            raise ValueError(f"CodeGym declared action {name!r} is absent from the environment source")
        node = functions[name]
        args = [a for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                if a.arg not in {"self", "cls"}]
        properties = {arg.arg: _property(arg.annotation) for arg in args}
        positional = [*node.args.posonlyargs, *node.args.args]
        required = [a.arg for a in positional[:len(positional)-len(node.args.defaults)]
                    if a.arg not in {"self", "cls"}]
        required += [a.arg for a, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if default is None]
        definitions.append({"name": name, "description": ast.get_docstring(node) or name,
                            "inputSchema": {"type": "object", "properties": properties, "required": required}})
    raw = build_codegym_schema({"env_name": "codegym_task", "actions": names, "params": {}})
    for tool in definitions:
        raw["actions"][tool["name"]].update(description=tool["description"], mcp=tool)
    return raw


def bootstrap_schema(capacity):
    from agent_system.policies.dyad.actions.codegym_schema import build_codegym_schema
    raw = build_codegym_schema({"env_name": "codegym_all", "actions": ["Observe"], "params": {}})
    raw["codegym_action_capacity"] = int(capacity)
    return raw


def compile_task(tokenizer, vocab_size, raw, capacity, *, step_protocol_version=None):
    from agent_system.policies.dyad.actions.schema_compiler import compile_action_schema
    if step_protocol_version is not None:
        if step_protocol_version != 2:
            raise ValueError("Unsupported CodeGym step action protocol")
        from copy import deepcopy
        raw = deepcopy(raw)
        raw["markers"].update(preserve_exit_prefix=True, preserve_partial_exit_prefix=True)
    cfg = compile_action_schema(tokenizer, vocab_size, raw)
    if not 0 < len(cfg["action_ids"]) <= capacity:
        raise ValueError("CodeGym task action count exceeds the run's action capacity")
    cfg["total_size"] = int(capacity)
    return cfg


def context_path(directory, cfg, prompts):
    key = hashlib.sha256(json.dumps([cfg, prompts], sort_keys=True).encode()).hexdigest()
    return Path(directory) / (key + ".pt")


def load_context(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg, hidden, mask = payload["action_config"], payload["hidden"], payload["mask"]
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or hidden.shape[0] != len(cfg["action_ids"]):
        raise ValueError("CodeGym encoder artifact does not match its task schema")
    return payload


def context_head(payload, projector, reference, tokenizer=None):
    """One differentiable head, padded without making padding a legal action."""
    import torch
    from agent_system.policies.dyad.models.action_head_factory import _reference_rows
    cfg = payload["action_config"]
    if projector.scale == "uniform":
        reference = _reference_rows(cfg, reference, tokenizer)
    # The projector transfers independent action chunks; moving the entire padded
    # cache here defeats that memory bound for long native tool descriptions.
    weight = projector(payload["hidden"], payload["mask"], reference)
    return torch.nn.functional.pad(weight, (0, 0, 0, cfg["total_size"] - weight.shape[0]))


def packed_action_logits(hidden, heads, lengths, lines):
    """Apply each task's head to its own packed sequence, with split gradients."""
    import torch
    parts = [[] for _ in range(2 if lines == "both" else 1)]
    offset = 0
    for weight, length in zip(heads, lengths, strict=True):
        x = hidden[..., offset:offset + length, :]
        weight = weight.to(x.dtype)
        if lines == "both":
            values = (torch.nn.functional.linear(x, weight.detach()),
                      torch.nn.functional.linear(x.detach(), weight))
        elif lines == "encoder":
            values = (torch.nn.functional.linear(x.detach(), weight),)
        elif lines == "policy":
            values = (torch.nn.functional.linear(x, weight.detach()),)
        else:
            values = (torch.nn.functional.linear(x, weight),)
        for dest, value in zip(parts, values, strict=True):
            dest.append(value)
        offset += length
    if offset > hidden.shape[-2]:
        raise ValueError("CodeGym packed sequence bounds exceed hidden states")
    # Static-bucket padding is excluded from loss and from both gradient lines.
    if offset < hidden.shape[-2]:
        padding = hidden.new_zeros((*hidden.shape[:-2], hidden.shape[-2] - offset, heads[-1].shape[0]))
        for part in parts:
            part.append(padding)
    return tuple(torch.cat(part, dim=-2) for part in parts)


class FullSequentialSampler:
    """Deterministic full batches, repeating a prefix instead of dropping the last tasks."""
    def __init__(self, size, padded_size):
        self.size, self.padded_size = size, padded_size
    def __len__(self):
        return self.padded_size
    def __iter__(self):
        return (i % self.size for i in range(self.padded_size))


def complete_epoch_sampler(dataset, config):
    import torch
    from torchdata.stateful_dataloader.sampler import RandomSampler
    size = len(dataset)
    batch = int(config.get("gen_batch_size") or config["train_batch_size"])
    if size < 1 or batch < 1:
        raise ValueError("CodeGym needs nonempty training data and a positive batch size")
    padded = ((size + batch - 1) // batch) * batch
    if not config.get("shuffle", True):
        return FullSequentialSampler(size, padded)
    generator = torch.Generator()
    if config.get("seed") is not None:
        generator.manual_seed(int(config["seed"]))
    return RandomSampler(dataset, replacement=False, num_samples=padded, generator=generator)
