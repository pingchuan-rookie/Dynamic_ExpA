"""Evaluation task references; private task content stays inside env actors."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TauEvaluationDataset(Dataset):
    """Expand official snapshot identities into independently seeded trials."""

    episode_field = "tau_episode"

    def __init__(self, data_files, tokenizer=None, config=None, processor=None, max_samples=-1):
        import pyarrow.parquet as pq
        from omegaconf import OmegaConf

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        settings = (config or {}).get("tau", {})
        if OmegaConf.is_config(settings):
            settings = OmegaConf.to_container(settings, resolve=True)
        settings = dict(settings)
        benchmark = settings.get("benchmark")
        if benchmark != "t2bench":
            raise ValueError("data.tau.benchmark must be t2bench")
        paths = [data_files] if isinstance(data_files, (str, Path)) else list(data_files)
        identities = []
        for path in paths:
            # Column projection is intentional: task_json contains hidden instructions and grading data.
            identities.extend(pq.read_table(path, columns=[
                "benchmark", "domain", "split", "task_id", "source_commit",
                "source_path", "source_sha256", "resources_json",
            ]).to_pylist())
        domains = settings.get("domains") or ["retail", "airline", "telecom"]
        if isinstance(domains, str):
            domains = domains.split(",")
        allowed_domains = {"retail", "airline", "telecom"}
        if not domains or len(domains) != len(set(domains)) or set(domains) - allowed_domains:
            raise ValueError("Choose unique official domains for the selected benchmark")
        split = settings.get("split", "base")
        requested = settings.get("task_ids")
        if isinstance(requested, str):
            requested = requested.split(",")
        requested = None if requested is None else [str(x) for x in requested]
        if requested is not None and (not requested or len(requested) != len(set(requested))):
            raise ValueError("Task IDs must be nonempty and unique")
        limit = settings.get("limit")
        limit = (3 if settings.get("debug", False) else -1) if limit is None else int(limit)
        trials = int(settings.get("num_trials", 1))
        base_seed = int(settings.get("seed", 300))
        if not 0 <= base_seed < 2**31:
            raise ValueError("Evaluation seed must be in [0, 2^31)")
        if trials < 1 or limit < -1 or int(max_samples) < -1:
            raise ValueError("Invalid evaluation trial/sample limit")
        selected = []
        for domain in domains:
            rows = [r for r in identities if r["benchmark"] == benchmark and r["domain"] == domain and r["split"] == split]
            if not rows:
                raise ValueError(f"No official task references for {benchmark}/{domain}/{split}")
            if len({str(r['task_id']) for r in rows}) != len(rows):
                raise ValueError(f"Duplicate task references for {domain}")
            if requested is not None:
                by_id = {str(r["task_id"]): r for r in rows}
                missing = set(requested) - by_id.keys()
                if missing:
                    raise ValueError(f"Unknown task IDs for {domain}: {sorted(missing)}")
                rows = [by_id[task_id] for task_id in requested]
            else:
                if settings.get("shuffle", False):
                    random.Random(base_seed).shuffle(rows)
                if limit != -1:
                    rows = rows[:limit]
            selected.extend(rows)
        if max_samples != -1:
            selected = selected[:max_samples]
        if not selected:
            raise ValueError("Evaluation task selection is empty")
        self.episodes = []
        session_config = copy.deepcopy(settings.get("session_config", {}))
        for row in selected:
            snapshot_ref = {k: row[k] for k in ("source_commit", "source_path", "source_sha256")}
            snapshot_ref["resources"] = json.loads(row["resources_json"])
            if not isinstance(snapshot_ref["resources"], list) or not all(snapshot_ref[k] for k in ("source_commit", "source_path", "source_sha256")):
                raise ValueError("Snapshot provenance is incomplete")
            public_identity = {k: row[k] for k in ("benchmark", "domain", "split", "task_id")}
            for trial in range(trials):
                offset = int(hashlib.sha256((row["domain"] + "/" + str(row["task_id"])).encode()).hexdigest()[:8], 16)
                episode = {**public_identity, "task_id": str(row["task_id"]), "trial": trial,
                           "base_seed": base_seed, "seed": (base_seed + trial * 1000003 + offset) % 2**31,
                           "snapshot_ref": copy.deepcopy(snapshot_ref),
                           "session_config": copy.deepcopy(session_config)}
                identity = {k: episode[k] for k in ("benchmark", "domain", "split", "task_id", "trial", "seed")}
                episode["episode_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
                self.episodes.append(episode)
        self.planned_episodes = [{k: v for k, v in ep.items() if k != "session_config"} for ep in self.episodes]

    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        if any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("Tau runtime prompts must contain text messages only")
        return None, None

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        episode = copy.deepcopy(self.episodes[index])
        return {"raw_prompt": [], "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
                "uid": episode["episode_id"], "index": index, self.episode_field: episode,
                "tau_padding": False, "data_source": episode["benchmark"],
                "reward_model": {"style": "rule", "ground_truth": None},
                "tools_kwargs": {"tau_session": {"create_kwargs": {"create_payload": episode}}},
                "interaction_kwargs": {}, "extra_info": {"index": index}}
