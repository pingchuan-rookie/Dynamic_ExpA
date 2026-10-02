"""DIVE task records for shared runtime-initialized training and evaluation."""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from torch.utils.data import Dataset

from agent_system.environments.backends.dive.selection import DEFAULT_DATA_DIR as DEFAULT_DATA_DIR
from agent_system.environments.backends.dive.selection import SELECTED_DOMAINS


DOMAINS = tuple(domain + suffix for domain in ("biological", "medical", "academic", "financial")
                for suffix in ("", "_general"))
SOURCE_FILES = {"train": "DIVE-RL-3K/DIVE-RL-3K.jsonl", "test": "DIVE-Eval/DIVE-Eval-800.jsonl"}
INVALID_PLACEHOLDERS = {"Failed to evolve question", "Failed to evolve answer",
                        "Failed to derive question", "Failed to derive answer"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def task_digest(task):
    return hashlib.sha256(canonical_json(task).encode()).hexdigest()


class InvalidPublishedTask(ValueError):
    """A known failed-generation placeholder, eligible for explicit exclusion."""


@lru_cache(maxsize=2048)
def _validate_schema(serialized):
    from jsonschema.validators import validator_for

    schema = json.loads(serialized)
    validator_for(schema).check_schema(schema)


def validate_task(task, *, source="DIVE task"):
    """Validate the published interface, without executing tools or changing schemas."""

    if not isinstance(task, dict):
        raise ValueError(f"{source}: expected a task object")
    for name in ("trace_id", "query", "answer"):
        value = task.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}: {name} must be nonempty text")
    metadata = task.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("domain") not in DOMAINS:
        raise ValueError(f"{source}: expected an official DIVE domain")
    tools = task.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{source}: expected a nonempty tools list")
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"{source}: expected OpenAI function tools")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"{source}: missing function definition")
        name = function.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"{source}: empty or duplicate tool name {name!r}")
        names.add(name)
        if not isinstance(function.get("description"), str):
            raise ValueError(f"{source}: missing description for {name}")
        schema = function.get("parameters")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"{source}: {name} parameters must be an object schema")
        try:
            _validate_schema(canonical_json(schema))
        except Exception as exc:
            raise ValueError(f"{source}: invalid parameter schema for {name}") from exc
    for name in ("trace_id", "query", "answer"):
        if task[name].strip() in INVALID_PLACEHOLDERS:
            raise InvalidPublishedTask(f"{source}: failed-generation placeholder in {name}")
    return task


def read_task_rows(paths):
    """Read and validate complete prepared sources before applying sample limits."""
    import pyarrow.parquet as pq

    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    rows, identities = [], set()
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                task = json.loads(row["task_json"])
                validate_task(task, source=f"{path}:{row.get('source_line', '?')}")
                if row.get("split") not in SOURCE_FILES:
                    raise ValueError(f"{path}: unsupported DIVE split")
                if row.get("trace_id") != task["trace_id"] or row.get("domain") != task["metadata"]["domain"]:
                    raise ValueError(f"{path}: task identity disagrees with its payload")
                if row.get("task_sha256") != task_digest(task):
                    raise ValueError(f"{path}: task payload hash mismatch")
                if not row.get("source_sha256") or not row.get("source_revision"):
                    raise ValueError(f"{path}: missing source provenance")
                if task["trace_id"] in identities:
                    raise ValueError(f"{path}: duplicate trace_id {task['trace_id']}")
                identities.add(task["trace_id"])
                rows.append(row)
    if not rows:
        raise ValueError("DIVE task selection is empty")
    if len({row["split"] for row in rows}) != 1:
        raise ValueError("Do not mix DIVE training and evaluation sources")
    return rows


class DiveDataset(Dataset):
    """Task-owned schemas; private answers are only transported to the env actor."""

    def __init__(self, data_files, tokenizer=None, config=None, processor=None, max_samples=-1):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        settings = (config or {}).get("dive", {})
        from omegaconf import OmegaConf

        if OmegaConf.is_config(settings):
            settings = OmegaConf.to_container(settings, resolve=True)
        self.settings = copy.deepcopy(dict(settings))
        rows = read_task_rows(data_files)
        domains = self.settings.get("domains", list(SELECTED_DOMAINS))
        if isinstance(domains, str):
            domains = domains.split(",")
        if not domains or len(domains) != len(set(domains)) or set(domains) - set(SELECTED_DOMAINS):
            raise ValueError("DIVE domains must be unique names from academic, biological, medical")
        rows = [row for row in rows if row["domain"] in domains]
        if int(max_samples) < -1 or int(max_samples) == 0:
            raise ValueError("DIVE max_samples must be -1 or positive")
        if int(max_samples) != -1:
            rows = rows[:int(max_samples)]
        if not rows:
            raise ValueError("DIVE task selection is empty")
        self.rows = rows
        self.planned_episodes = [{key: row[key] for key in ("trace_id", "domain", "split", "task_sha256")}
                                 for row in rows]

    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        if any(not isinstance(message.get("content", ""), (str, type(None))) for message in messages):
            raise ValueError("DIVE supports text and native tool messages, not vision inputs")
        return None, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        task = json.loads(row["task_json"])
        payload = {"task": task, "split": row["split"], "episode_id": row["trace_id"],
                   "session_config": copy.deepcopy(self.settings.get("session_config", {})),
                   "source_sha256": row["source_sha256"], "source_revision": row["source_revision"],
                   "source_line": row["source_line"], "task_sha256": row["task_sha256"]}
        return {"raw_prompt": [], "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
                "uid": row["trace_id"], "index": index, "data_source": "dive",
                "reward_model": {"style": "rule", "ground_truth": None},
                "tools_kwargs": {"dive_session": {"create_kwargs": {"create_payload": payload}}},
                "interaction_kwargs": {},
                "extra_info": {"index": index, "trace_id": row["trace_id"],
                               "domain": row["domain"], "split": row["split"]}}
