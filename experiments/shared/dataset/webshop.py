"""Build and validate shared WebShop full/human data from real fixed-task resets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[3]
COLUMNS = ("data_source", "index", "prompt", "ability", "reward_model", "extra_info")
SPLITS = ("train", "dev", "test")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROMPTS = _load("webshop_prompt_loader", Path(__file__).parent / "prompt/loader.py").load("webshop")
SYSTEM_PROMPT = _PROMPTS.text("system_react")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_ranges(total):
    if type(total) is not int or total <= 1500:
        raise ValueError("WebShop full requires all official human goals, including train")
    return {"test": [0, 500], "dev": [500, 1500], "train": [1500, total]}


def read_tasks(assets_dir):
    """Validate export identity before constructing any model-visible columns."""
    assets_dir = Path(assets_dir)
    manifest_path = assets_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format_version") != 1 or manifest.get("benchmark") != "webshop"
            or manifest.get("variant") != "full" or manifest.get("human_goals") is not True
            or manifest.get("seed") != 233 or manifest.get("observation_mode") != "text_rich"):
        raise ValueError(f"Incompatible WebShop full asset manifest: {manifest_path}")
    counts = manifest["counts"]
    ranges = split_ranges(counts["goals"])
    if counts["products"] < 1_000_000 or manifest.get("split_ranges") != ranges:
        raise ValueError("WebShop manifest does not describe the official full split")
    if any(counts[split] != high - low for split, (low, high) in ranges.items()):
        raise ValueError("WebShop manifest split counts disagree")
    if manifest["tasks"]["path"] != "tasks.jsonl":
        raise ValueError("WebShop task export must be assets/tasks.jsonl")
    path = assets_dir / "tasks.jsonl"
    if sha256_file(path) != manifest["tasks"]["sha256"]:
        raise ValueError("WebShop task export checksum mismatch")
    tasks = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            task = json.loads(line)
            split = "test" if index < 500 else "dev" if index < 1500 else "train"
            if type(task.get("task_id")) is not int or task["task_id"] != index or task.get("split") != split:
                raise ValueError(f"WebShop task export must preserve fixed official goal order: row {index}")
            instruction, observation = task.get("instruction"), task.get("initial_observation")
            if (not isinstance(instruction, str) or not instruction.strip() or
                    not isinstance(observation, str) or instruction not in observation):
                raise ValueError(f"WebShop task {index} has no authentic instruction/reset observation")
            tasks.append(task)
    if len(tasks) != counts["goals"]:
        raise ValueError(f"Incomplete WebShop task export: {len(tasks)}/{counts['goals']}")
    return manifest, tasks


def make_sample(task):
    """Whitelist only public reset data; metadata.goal is deliberately not copied."""
    task_id, split = task["task_id"], task["split"]
    start = {"test": 0, "dev": 500, "train": 1500}[split]
    observation = task["initial_observation"]
    return {
        "data_source": "webshop", "index": task_id - start,
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": observation}],
        "ability": "web_shopping", "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"split": split, "task_id": task_id, "need_tools_kwargs": True,
                       "tools_kwargs": {"webshop_action": {"create_kwargs": {
                           "task_id": task_id, "split": split, "initial_observation": observation}}}},
    }


def backend_health(assets_dir):
    """Use the actual isolated runtime; missing dependencies cannot pass validation."""
    # Import-light modules by path keep dataset construction independent of torch/Ray.
    config_module = _load("webshop_dataset_config", PROJECT / "agent_system/environments/backends/webshop/config.py")
    worker_module = _load("webshop_dataset_worker", PROJECT / "agent_system/environments/backends/webshop/worker.py")
    config = config_module.resolve_webshop_config({})
    # Explicit CLI/mount selection wins over inherited aliases during this check.
    config["backend_config"]["assets_dir"] = str(Path(assets_dir).resolve())
    worker = worker_module.WebShopEnvWorker(config)
    try:
        health = worker.health_check()
        if not health.get("ok") or health.get("variant") != "full" or health.get("human_goals") is not True:
            raise ValueError(f"WebShop full backend health failed: {health}")
        return health
    finally:
        worker.shutdown()


def dataset_identity(assets_dir, manifest):
    return {"format_version": 1, "benchmark": "webshop", "variant": "full", "human_goals": True,
            "assets_manifest_sha256": sha256_file(Path(assets_dir) / "manifest.json"),
            "tasks_sha256": manifest["tasks"]["sha256"], "goals_sha256": manifest["goals_sha256"],
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "split_ranges": manifest["split_ranges"], "columns": list(COLUMNS),
            "initial_state_source": "official_backend_fixed_task_reset",
            "reward": {"scale": [0, 1], "full_success": "terminal and reward == 1"}}


def validate_dataset(output_dir, assets_dir, *, check_backend=True):
    import pyarrow.parquet as pq

    output_dir, assets_dir = Path(output_dir), Path(assets_dir)
    source, tasks = read_tasks(assets_dir)
    saved = json.loads((output_dir / "manifest.json").read_text())
    identity = dataset_identity(assets_dir, source)
    if any(saved.get(key) != value for key, value in identity.items()):
        raise ValueError("WebShop dataset manifest/source/prompt mismatch; explicitly regenerate data")
    counts = {}
    for split in SPLITS:
        path = output_dir / f"{split}.parquet"
        record = saved["splits"][split]
        if record["path"] != path.name or record["sha256"] != sha256_file(path):
            raise ValueError(f"WebShop {split} parquet checksum mismatch")
        table = pq.read_table(path)
        if tuple(table.column_names) != COLUMNS:
            raise ValueError(f"WebShop requires the algorithm-neutral six-column protocol: {path}")
        low, high = source["split_ranges"][split]
        if len(table) != high - low or record["rows"] != len(table):
            raise ValueError(f"Incomplete WebShop {split} source: {len(table)}/{high - low}")
        for offset, row in enumerate(table.to_pylist()):
            if row != make_sample(tasks[low + offset]):
                raise ValueError(f"WebShop {split} row {offset} differs from its public fixed-task reset; no hidden fields allowed")
        counts[split] = len(table)
    if check_backend:
        health = backend_health(assets_dir)
        if health["counts"] != source["counts"] or health["goals_sha256"] != source["goals_sha256"]:
            raise ValueError("WebShop live backend identity differs from dataset source")
    return {"source": str(output_dir.resolve()), "assets": str(assets_dir.resolve()),
            "rows": counts, "split_ranges": source["split_ranges"],
            "assets_manifest_sha256": identity["assets_manifest_sha256"]}


def build_dataset(assets_dir, output_dir, *, overwrite=False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    assets_dir, output_dir = Path(assets_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        return validate_dataset(output_dir, assets_dir)
    manifest, tasks = read_tasks(assets_dir)
    health = backend_health(assets_dir)
    if health["counts"] != manifest["counts"] or health["goals_sha256"] != manifest["goals_sha256"]:
        raise ValueError("WebShop backend and exported task identity disagree")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Build every split before publishing any file. A failing row never disappears.
    with tempfile.TemporaryDirectory(prefix=".webshop-dataset-", dir=output_dir.parent) as temp:
        staging = Path(temp)
        result = dataset_identity(assets_dir, manifest)
        result["splits"] = {}
        for split in SPLITS:
            low, high = manifest["split_ranges"][split]
            rows = [make_sample(task) for task in tasks[low:high]]
            path = staging / f"{split}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            result["splits"][split] = {"path": path.name, "rows": len(rows), "sha256": sha256_file(path)}
        (staging / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        validate_dataset(staging, assets_dir, check_backend=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in [*(f"{split}.parquet" for split in SPLITS), "manifest.json"]:
            os.replace(staging / name, output_dir / name)
    return validate_dataset(output_dir, assets_dir, check_backend=False)


def evaluation_subset(directory):
    """Publish a separate, identity-checked selection without changing official splits."""
    import pyarrow.parquet as pq
    from agent_system.utils.evaluation_protocol import selection_identity

    directory = Path(directory)
    source = directory / "test.parquet"
    table = pq.read_table(source)
    ids = [row["extra_info"]["task_id"] for row in table.to_pylist()]
    identity = selection_identity("webshop", ids)
    indices = {task_id: index for index, task_id in enumerate(ids)}
    selected = table.take([indices[task_id] for task_id in identity["selected_task_ids"]])
    destination = directory / "test100_seed42.parquet"
    metadata_path = directory / "test100_seed42.selection.json"
    identity.update(source=source.name, source_sha256=sha256_file(source))
    if destination.exists():
        if not pq.read_table(destination).equals(selected):
            raise ValueError("Existing WebShop evaluation subset differs from fixed protocol; refusing overwrite")
    else:
        with tempfile.NamedTemporaryFile(prefix=".webshop-test100-", suffix=".parquet", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            pq.write_table(selected, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    identity["selected_source_sha256"] = sha256_file(destination)
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != identity:
            raise ValueError("Existing WebShop subset identity differs; refusing overwrite")
    else:
        metadata_path.write_text(json.dumps(identity, indent=2) + "\n")
    return destination.resolve(), identity


def prepare_data(env, project, overrides, *, check_only=False):
    """Own split selection in both public entrypoints, including --check."""
    project = Path(project).expanduser().resolve()
    code_root = Path(env.get("CODE_ROOT") or project).expanduser().resolve()
    data_root = Path(env["DATA_ROOT"]).expanduser().resolve() if env.get("DATA_ROOT") else code_root
    # Local/container same-tree data moved with the source; a separate mount keeps its existing namespace.
    storage = project if data_root == code_root else data_root
    directory = storage / "data/webshop/dataset"
    assets = Path(env.get("WEBSHOP_ASSETS_DIR") or env.get("WEBSHOP_ASSETS") or
                  env.get("WEBSHOP_DATA_DIR") or directory.parent / "assets").expanduser().resolve()
    # Runtime configuration uses this same absolute mount, never a baked-in host path.
    env["WEBSHOP_ASSETS"] = str(assets)
    from_config = {"WEBSHOP_MAX_STEPS": "environment steps", "MAX_ASSISTANT_TURNS": "assistant turns",
                   "MAX_TOOL_TURNS": "tool turns", "WEBSHOP_BACKEND_REPLICAS": "backend replicas",
                   "WEBSHOP_SESSIONS_PER_BACKEND": "session slots"}
    for key, label in from_config.items():
        if key in env and (not str(env[key]).isdigit() or int(env[key]) <= 0):
            raise ValueError(f"WebShop {label} requires a positive integer {key}")
    split = "test" if env.get("RUN_IS_EVAL") == "1" else "dev"
    # Validate the complete source before selecting a fixed evaluation subset.
    report = validate_dataset(directory, assets)
    evaluation_path, selection = evaluation_subset(directory) if split == "test" else (None, None)
    selected = {"TRAIN_DATA": "train", "TEST_DATA": split, "VAL_DATA": split}
    for key, target_split in selected.items():
        path = evaluation_path if target_split == "test" else (directory / f"{target_split}.parquet").resolve()
        if env.get(key) and Path(env[key]).expanduser().resolve() != path:
            raise ValueError(f"WebShop official split isolation requires {key}={path}; train validates on dev, evaluation uses test")
        env[key] = str(path)
    protected = {"data", "data.train_files", "data.val_files", "data.filter_overlong_prompts", "data.truncation"}
    for arg in overrides:
        if arg.lstrip("+~").partition("=")[0] in protected:
            raise ValueError("WebShop owns official split selection and complete sample coverage; remove " + arg)
    if selection is not None:
        cap = next((int(arg.split("=", 1)[1]) for arg in reversed(overrides)
                    if arg.lstrip("+").startswith("data.val_max_samples=")), int(env.get("VAL_MAX_SAMPLES", "-1")))
        if env.get("RUN_IS_DEBUG") != "1" and cap not in (-1, 100):
            raise ValueError("Formal WebShop evaluation requires the fixed 100 test tasks; use --debug for smaller runs")
        # Keep the dataset sidecar portable; saved run configuration records the resolved source.
        report["evaluation_selection"] = {**selection, "source": str((directory / selection["source"]).resolve())}
        report["evaluation_source"] = str(evaluation_path)
    report["validation_split"] = split
    env["WEBSHOP_DATA"] = json.dumps(report)
    print("[prepare] WebShop full data: " + env["WEBSHOP_DATA"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root = PROJECT / "data/webshop"
    parser.add_argument("--assets-dir", type=Path, default=os.environ.get("WEBSHOP_ASSETS_DIR") or os.environ.get("WEBSHOP_ASSETS") or root / "assets")
    parser.add_argument("--output-dir", type=Path, default=root / "dataset")
    parser.add_argument("--check", action="store_true", help="Validate full assets, real backend dependencies, and all three complete splits")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.check and args.overwrite:
        parser.error("--check cannot be combined with --overwrite")
    try:
        report = (validate_dataset(args.output_dir, args.assets_dir) if args.check else
                  build_dataset(args.assets_dir, args.output_dir, overwrite=args.overwrite))
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f"[webshop] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
