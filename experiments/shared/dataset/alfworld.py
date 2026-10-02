"""Build and validate offline ALFWorld source snapshots for GRPO-ReAct and Dyad."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from experiments.shared.dataset.utils import source_snapshot as snapshot


def validate_full_test(path: Path, mapping: Path, *, split="test") -> list[int]:
    import pyarrow.parquet as pq

    expected_rows = json.loads(mapping.read_text())
    expected = {int(row["item_id"]): row for row in expected_rows}
    if not expected or len(expected) != len(expected_rows):
        raise ValueError(f"Invalid ALFWorld test mapping: {mapping}")
    table = pq.read_table(path)
    if "agent_name" in table.column_names:
        raise ValueError(f"ALFWorld evaluation requires algorithm-neutral data: {path}")
    ids = []
    for row in table.to_pylist():
        extra = row.get("extra_info") or {}
        task = ((extra.get("tools_kwargs") or {}).get("alfworld_action") or {}).get("create_kwargs") or {}
        game = task.get("game")
        if extra.get("split") != split or game not in expected:
            raise ValueError(f"ALFWorld full evaluation requires {split} rows: {path}, game={game}")
        if split == "test_unseen" and extra.get("source_split", "valid_unseen") != "valid_unseen":
            raise ValueError(f"ALFWorld evaluation requires valid_unseen source identity: {path}")
        if task.get("task_id") != expected[game]["task_id"]:
            raise ValueError(f"ALFWorld task identity mismatch: {path}, game={game}")
        ids.append(game)
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f"Incomplete ALFWorld evaluation source: {path}: {len(ids)}/{len(expected)} rows; "
                         "build alfworld.py --full-eval, never use a pre-truncated test source")
    return ids


def prepare_evaluation(env, project: Path, overrides: list[str], *, check_only=False):
    """Apply evaluation-only source/seed settings; leave training selection untouched."""
    from agent_system.utils.evaluation_protocol import selection_identity
    project = Path(project).expanduser().resolve()
    code_root = Path(env.get("CODE_ROOT") or project).expanduser().resolve()
    data_root = Path(env.get("DATA_ROOT") or code_root).expanduser().resolve()
    storage = project if data_root == code_root else data_root
    canonical = storage / "data/alfworld/dataset/test_unseen.parquet"
    destination = Path(env.get("TEST_DATA") or canonical).expanduser().resolve()
    env["TEST_DATA"] = str(destination)
    values = {}
    for arg in overrides:
        key, sep, value = arg.lstrip("+").partition("=")
        if sep:
            values[key] = value
        if arg.lstrip("+~").split("=", 1)[0] in {"data.val_files", "data.filter_overlong_prompts", "data.truncation"}:
            raise ValueError("Full ALFWorld evaluation owns source and non-filtering policy; remove " + arg)
    # Pin evaluation sampling only. The shared training seed/shuffle defaults are unchanged.
    seed_text = values.get("data.seed", env.get("ALFWORLD_EVAL_SEED", "0"))
    seed = int(seed_text)
    if "data.seed" not in values:
        overrides.append(f"data.seed={seed}")
    overrides.extend(["data.filter_overlong_prompts=False", "data.truncation=error"])
    mapping_path = project / "agent_system/environments/configs/alfworld_mappings_unseen.json"
    ids = validate_full_test(destination, mapping_path, split="test_unseen")
    if len(ids) != 134:
        raise ValueError("ALFWorld formal evaluation requires all 134 valid_unseen tasks")
    import numpy as np
    cap = int(values.get("data.val_max_samples", env.get("VAL_MAX_SAMPLES", "-1")))
    if env.get("RUN_IS_DEBUG") != "1" and cap not in (-1, 134):
        raise ValueError("Formal ALFWorld evaluation requires all 134 valid_unseen tasks; use --debug for a smaller run")
    shuffle_text = values.get("data.shuffle", "true").lower()
    if shuffle_text not in {"true", "false"}:
        raise ValueError("ALFWorld evaluation data.shuffle must be true or false")
    selected = ids
    if 0 < cap < len(ids):
        indices = np.random.default_rng(seed).choice(len(ids), size=cap, replace=False) if shuffle_text == "true" else range(cap)
        selected = [ids[int(index)] for index in indices]
    import hashlib
    mapping = json.loads(mapping_path.read_text())
    tasks = {row['item_id']: row['task_id'] for row in mapping}
    metadata = {"source": str(destination), "source_rows": len(ids), "seed": seed,
                "split": "valid_unseen", "evaluation_selection": selection_identity("alfworld", list(tasks.values())),
                "source_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "shuffle": shuffle_text == "true", "sample_limit": cap, "selected_game_ids": selected,
                "selected_task_ids": [tasks[game] for game in selected]}
    env["ALFWORLD_EVAL_DATA"] = json.dumps(metadata)
    print("[prepare] ALFWorld full evaluation data: " + env["ALFWORLD_EVAL_DATA"])


SPLITS = {"train": 3553, "test": 140, "test_unseen": 134, "test_full": 140}
SPLIT_SOURCES = {"train": ("train", "train"), "test": ("valid_seen", "test"),
                 "test_unseen": ("valid_unseen", "unseen"), "test_full": ("valid_seen", "test")}
SOURCE = {
    "dataset": "ALFWorld json_2.1.1", "status": "local_release_assets_unverified",
    "selected_unique_tasks": 3827,
    "split_mapping": {key: value[0] for key, value in SPLIT_SOURCES.items()},
    "limitations": ["Selected task assets are copied byte-for-byte from the local ALFWorld release tree; no official release-byte digest has been authenticated.",
                    "Public initial observations/actions and private walkthroughs come from preserved local reset/expert caches, not original release records.",
                    "test_full reuses the same valid_seen task assets as test; runtime reset remains authoritative."],
}


def source_rows(root: Path, split: str):
    import copy
    source_split, mapping_name = SPLIT_SOURCES[split]
    mapping = json.loads((root / f"source/mappings/alfworld_mappings_{mapping_name}.json").read_text())
    by_id = {entry["item_id"]: entry for entry in mapping}
    if len(by_id) != len(mapping) or len(mapping) != SPLITS[split]:
        raise ValueError(f"Duplicate or incomplete ALFWorld mapping: {split}")
    seen = set()
    for previous in snapshot.iter_rows(root / f"source/local_selection/{split}.parquet"):
        row = copy.deepcopy(previous)
        extra = row["extra_info"]
        task = extra["tools_kwargs"]["alfworld_action"]["create_kwargs"]
        game = task["game"]
        if game in seen or game not in by_id:
            raise ValueError(f"Duplicate/unknown ALFWorld game: {split}/{game}")
        seen.add(game)
        entry = by_id[game]
        if task["task_id"] != entry["task_id"]:
            raise ValueError(f"ALFWorld mapping identity mismatch: {split}/{game}")
        directory = f"source/json_2.1.1/{source_split}/{entry['task_type']}/{entry['task_id']}"
        for name in ("traj_data.json", "game.tw-pddl", "initial_state.pddl"):
            if not (root / directory / name).is_file():
                raise ValueError(f"Missing complete ALFWorld task asset: {directory}/{name}")
        trajectory = json.loads((root / directory / "traj_data.json").read_text())
        if trajectory["task_id"] != task["task_id"] or trajectory["task_type"] != task["task_type"]:
            raise ValueError(f"ALFWorld source identity mismatch: {split}/{game}")
        cache_path = f"source/derived_cache/{source_split}/{entry['task_type']}/{entry['task_id']}.json"
        cache = json.loads((root / cache_path).read_text())
        if (cache.get("split") != source_split or cache.get("task_type") != entry["task_type"]
                or cache.get("task_id") != entry["task_id"]):
            raise ValueError(f"ALFWorld cache identity mismatch: {split}/{game}")
        observation = cache.get("observation")
        if not isinstance(observation, str) or not observation.strip():
            raise ValueError(f"Missing cached public reset: {split}/{game}; expert/reset execution is not allowed")
        if (cache.get("available_actions") != extra["initial_available_actions"]
                or cache.get("walkthrough") != row["reward_model"]["ground_truth"]):
            raise ValueError(f"ALFWorld cached private reference/actions differ: {split}/{game}")
        row["prompt"] = [{"role": "user", "content": observation}]
        identity = f"alfworld/json_2.1.1/{source_split}/{entry['task_type']}/{entry['task_id']}"
        extra["initial_observation"] = observation
        extra["reset_record"] = snapshot.source_ref(root, cache_path, task_id=identity)
        yield snapshot.mark_row(row, identity, snapshot.source_ref(
            root, directory + "/traj_data.json", task_id=entry["task_id"]), source_split)
    if seen != set(by_id):
        raise ValueError(f"ALFWorld selection differs from mapping: {split}")


def validate_source_dataset(root: Path) -> None:
    manifest = snapshot.validate_manifest(root, "alfworld")
    if {k: v["rows"] for k, v in manifest["splits"].items()} != SPLITS:
        raise ValueError("ALFWorld split membership contract changed")
    for name in ("alfred.pddl", "alfred.twl2"):
        if not (root / "source/logic" / name).is_file():
            raise ValueError(f"Missing ALFWorld logic asset: {name}")
    for split in SPLITS:
        snapshot.compare_rows(root, split, source_rows(root, split))
    if list(snapshot.iter_rows(root / "test.parquet")) != list(snapshot.iter_rows(root / "test_full.parquet")):
        raise ValueError("ALFWorld test_full must preserve complete test membership and order")


def rebuild_source_dataset(destination: Path, assets: Path, cache: Path, *, overwrite=False):
    def populate(stage, active):
        if not (stage / "source").exists():
            snapshot.preserve_selection(active, stage, SPLITS)
            for source_split, name in dict.fromkeys(SPLIT_SOURCES.values()):
                mapping = _ROOT / f"agent_system/environments/configs/alfworld_mappings_{name}.json"
                snapshot.copy_asset(mapping, stage, f"source/mappings/{mapping.name}")
                for entry in json.loads(mapping.read_text()):
                    relative = f"{source_split}/{entry['task_type']}/{entry['task_id']}"
                    task = assets / "json_2.1.1" / relative
                    for required in ("traj_data.json", "game.tw-pddl", "initial_state.pddl"):
                        if not (task / required).is_file():
                            raise ValueError(f"Missing ALFWorld source asset: {task / required}")
                    # Preserve every per-task source file, including optional receps and
                    # any unknown metadata, instead of a lossy field/file projection.
                    for path in sorted(task.rglob("*")):
                        if path.is_file():
                            snapshot.copy_asset(path, stage, f"source/json_2.1.1/{relative}/" + path.relative_to(task).as_posix())
                    cache_file = cache / source_split / entry["task_type"] / (entry["task_id"] + ".json")
                    snapshot.copy_asset(cache_file, stage, f"source/derived_cache/{relative}.json")
            for name in ("alfred.pddl", "alfred.twl2"):
                snapshot.copy_asset(assets / "logic" / name, stage, f"source/logic/{name}")
        for split in SPLITS:
            snapshot.write_rows(stage / f"{split}.parquet", source_rows(stage, split))
        snapshot.make_manifest(stage, "alfworld", SPLITS, SOURCE)
    return snapshot.rebuild(destination, overwrite=overwrite, populate=populate, validate=validate_source_dataset)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline ALFWorld complete source snapshot conversion; never runs experts")
    parser.add_argument("--out", type=Path, default=_ROOT / "data/alfworld/dataset")
    parser.add_argument("--assets", type=Path, default=Path(os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")).expanduser())
    parser.add_argument("--cache", type=Path, default=_ROOT / ".cache/agentic_rl/alfworld/walkthroughs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", "--validate-only", action="store_true")
    parser.add_argument("--full-eval", action="store_true", help="Validate all 134 valid_unseen evaluation tasks, without modifying other data")
    parser.add_argument("--output", type=Path, help="Existing full-eval parquet to validate")
    args = parser.parse_args()
    if args.output and not args.full_eval:
        parser.error("--output requires --full-eval")
    if args.full_eval:
        path = args.output or args.out / "test_unseen.parquet"
        ids = validate_full_test(path, _ROOT / "agent_system/environments/configs/alfworld_mappings_unseen.json",
                                 split="test_unseen")
        print(f"[alfworld] existing complete full evaluation validated: {len(ids)} games -> {path}")
    elif args.check_only:
        snapshot.check_only(args.out.resolve(), validate_source_dataset)
    else:
        rebuild_source_dataset(args.out.resolve(), args.assets.expanduser(), args.cache.expanduser(), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
