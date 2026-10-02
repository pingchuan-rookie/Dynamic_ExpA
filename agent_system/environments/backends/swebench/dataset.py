"""Public-only SWE-bench task references, with independent deterministic trials."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

PUBLIC_FIELDS = {"instance_id", "repo", "base_commit", "problem_statement"}


class SwebenchEvaluationDataset(Dataset):
    episode_field = "swebench_episode"

    def __init__(self, data_files, tokenizer=None, config=None, processor=None, max_samples=-1):
        self.tokenizer, self.processor = tokenizer, processor
        settings = dict((config or {}).get("swebench", {}))
        if isinstance(data_files, dict):
            snapshot = deepcopy(data_files)
        else:
            paths = [data_files] if isinstance(data_files, (str, Path)) else list(data_files)
            if len(paths) != 1:
                raise ValueError("SWE-bench evaluation requires one pinned public snapshot")
            snapshot = json.loads(Path(paths[0]).read_text())
        if snapshot.get("benchmark") != "swebench_verified" or snapshot.get("format") != "swebench_public_v1":
            raise ValueError("Expected a pinned public SWE-bench snapshot")
        rows = snapshot.get("tasks")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Public snapshot has no tasks")
        if any(not isinstance(row, dict) or set(row) != PUBLIC_FIELDS for row in rows):
            raise ValueError("Public task fields differ from the policy allowlist")
        if len({row["instance_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate SWE-bench task references")
        source_identity = snapshot.get("source_identity")
        if not isinstance(source_identity, dict) or not source_identity.get("manifest_sha256"):
            raise ValueError("Public snapshot lacks asset identity")
        trials = int(settings.get("num_trials", 1))
        seed = int(settings.get("seed", 10))
        if trials < 1 or not 0 <= seed < 2**31 or max_samples != -1:
            raise ValueError("Invalid trials, seed, or implicit sample truncation")
        self.episodes = []
        for row in rows:
            for trial in range(trials):
                identity = {"benchmark": "swebench_verified", "instance_id": row["instance_id"],
                            "trial": trial, "seed": (seed + trial * 1000003) % 2**31}
                uid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
                self.episodes.append({**identity, "episode_id": uid, "source_identity": deepcopy(source_identity),
                                      "public_task": deepcopy(row)})
        self.planned_episodes = [{key: deepcopy(value) for key, value in row.items() if key != "public_task"}
                                 for row in self.episodes]

    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        if any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("SWE-bench prompts must be text-only")
        return None, None

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        episode = deepcopy(self.episodes[index])
        return {"raw_prompt": [], "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
                "uid": episode["episode_id"], "index": index, self.episode_field: episode,
                "data_source": "swebench_verified", "reward_model": {"style": "rule", "ground_truth": None},
                "tools_kwargs": {"swebench_session": {"create_kwargs": {"create_payload": episode}}},
                "interaction_kwargs": {}, "extra_info": {"index": index}}
