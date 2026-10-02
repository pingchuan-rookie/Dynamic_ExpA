"""Offline, lossless task snapshots for native t2bench evaluation.

These records are evaluation inputs, not ready-to-tokenize training prompts.
Only the environment, user simulator and evaluator may read ``task_json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[4]
DYAD_ROOT = Path(__file__).resolve().parents[4]
COLUMNS = (
    "benchmark", "domain", "split", "task_id", "sample_id", "task_json",
    "source_commit", "source_path", "source_sha256", "resources_json",
)
DOMAINS = {"t2bench": ("airline", "retail", "telecom")}
DEFAULT_SPLIT = {"t2bench": "base"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_reference(path, root):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Required source file missing or empty: {path}")
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number in {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object, parse_constant=reject_constant)


def validate_t2bench_task(task):
    """Validate the native task envelope without loading model/provider packages."""
    if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not task["id"]:
        raise ValueError("Expected a task with a nonempty native string ID")
    scenario = task.get("user_scenario")
    if not isinstance(scenario, dict):
        raise ValueError(f"Task {task['id']} requires user_scenario")
    instructions = scenario.get("instructions")
    if isinstance(instructions, dict):
        if not all(isinstance(instructions.get(key), str)
                   for key in ("domain", "reason_for_call", "task_instructions")):
            raise ValueError(f"Task {task['id']} has invalid structured user instructions")
    elif not isinstance(instructions, str):
        raise ValueError(f"Task {task['id']} requires user instructions")
    for key in ("description", "initial_state", "evaluation_criteria"):
        if task.get(key) is not None and not isinstance(task[key], dict):
            raise ValueError(f"Task {task['id']} has invalid {key}")


def _git_identity(root):
    if (root / "source_manifest.json").is_file():
        from agent_system.environments.env_package.source_bundle import source_identity
        return source_identity(root)
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], check=False, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"Cannot read source Git identity in {root}")
        return result.stdout.strip()
    # Do not accidentally record a parent repository as the source checkout.
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError(f"Source root is not a Git checkout root: {root}")
    return {"commit": git("rev-parse", "HEAD"), "tracked_changes": git("status", "--porcelain", "--untracked-files=no")}


def _resources(domain, root):
    base = root / "data/tau2/domains" / domain
    names = ("db.toml", "user_db.toml", "main_policy.md", "tech_support_manual.md") if domain == "telecom" else ("db.json", "policy.md")
    paths = [base / name for name in names]
    source = root / "src" / "tau2"
    paths += sorted((source / "domains" / domain).glob("*.py"))
    paths += [source / "data_model" / "tasks.py", source / "environment" / "environment.py", source / "orchestrator" / "orchestrator.py"]
    paths += sorted((source / "evaluator").glob("*.py"))
    return [file_reference(path, root) for path in sorted(set(paths))]


def build_snapshot(benchmark, source_root=None, domains=None, split=None):
    if benchmark not in DOMAINS:
        raise ValueError(f"Unsupported benchmark: {benchmark!r}; use t2bench for tau evaluation snapshots.")
    root = Path(source_root or REPO_ROOT / "data" / benchmark / "source").expanduser().resolve()
    split = split or DEFAULT_SPLIT[benchmark]
    domains = list(DOMAINS[benchmark] if domains is None else domains)
    if not domains or len(domains) != len(set(domains)) or set(domains) - set(DOMAINS[benchmark]):
        raise ValueError(f"Choose unique domains from {DOMAINS[benchmark]}")
    if split != DEFAULT_SPLIT[benchmark]:
        raise ValueError(f"This evaluation snapshot supports only the official {DEFAULT_SPLIT[benchmark]!r} split for {benchmark}")
    identity = _git_identity(root)
    rows, inputs, counts, resources_by_domain = [], [], {}, {}
    for domain in domains:
        base = root / "data/tau2/domains" / domain
        path, split_path = base / "tasks.json", base / "split_tasks.json"
        tasks, splits = read_json(path), read_json(split_path)
        if not isinstance(tasks, list) or not tasks or not all(isinstance(t, dict) and isinstance(t.get("id"), str) for t in tasks):
            raise ValueError(f"Expected nonempty tasks with native string IDs: {path}")
        for task in tasks:
            validate_t2bench_task(task)
        ids = [task["id"] for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate task IDs in {path}")
        if not isinstance(splits, dict) or split not in splits:
            raise ValueError(f"No official {split!r} split for {domain}")
        selected_ids = splits[split]
        if not isinstance(selected_ids, list) or not selected_ids or not all(isinstance(v, str) for v in selected_ids):
            raise ValueError(f"Expected nonempty string ID split in {split_path}")
        selected_set = set(selected_ids)
        if len(selected_ids) != len(selected_set) or selected_set - set(ids):
            raise ValueError(f"Duplicate or missing split task IDs in {split_path}")
        # Official loaders retain tasks.json ordering, not split manifest ordering.
        selected = [(task["id"], task) for task in tasks if task["id"] in selected_set]
        inputs.append(file_reference(split_path, root))
        task_source = file_reference(path, root)
        inputs.append(task_source)
        resources = _resources(domain, root)
        resources_by_domain[domain] = resources
        counts[domain] = len(selected)
        for task_id, task in selected:
            rows.append(dict(zip(COLUMNS, (
                benchmark, domain, split, task_id, f"{benchmark}/{domain}/{split}/{task_id}",
                canonical_json(task), identity["commit"], task_source["path"], task_source["sha256"], canonical_json(resources),
            ))))
    manifest = {
        "schema_version": 1, "record_type": "evaluation_task_snapshot", "ready_for_training": False,
        "benchmark": benchmark, "split": split, "domains": domains, "num_records": len(rows), "domain_counts": counts,
        "columns": list(COLUMNS), "source_root": str(root), "source": identity, "inputs": inputs,
        "resources": resources_by_domain,
        "task_json_visibility": "environment_user_simulator_evaluator_only; never render into agent prompts",
        "initial_observation": "runtime_user_simulator_required; no static task prompt or generated user message",
        "split_policy": "official membership and native task order; no random split, subsampling, or train/test fallback",
        "scope": "t2bench airline/retail/telecom text dialogue, telecom manual policy",
        "warnings": [
            "The source task JSON includes hidden user instructions, initialization and grading information.",
            "Snapshot generation is offline; dialogue and some native grading require separately configured models.",
            "t2bench base includes its train and test splits; it is not a held-out test set after training on train.",
        ],
    }
    return rows, manifest


def write_snapshot(rows, manifest, output_dir, overwrite=False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir = Path(output_dir).expanduser().resolve()
    source_root = Path(manifest["source_root"])
    if output_dir.is_relative_to(source_root):
        raise ValueError("Generated data must not be written into the native source checkout")
    name = manifest["split"]
    parquet_path, manifest_path = output_dir / f"{name}.parquet", output_dir / "manifest.json"
    if not overwrite and (parquet_path.exists() or manifest_path.exists()):
        raise ValueError(f"Snapshot already exists in {output_dir}; use --overwrite explicitly")
    schema = pa.schema([(name, pa.string()) for name in COLUMNS], metadata={
        b"record_type": b"evaluation_task_snapshot", b"ready_for_training": b"false",
        b"task_json_visibility": b"environment_user_simulator_evaluator_only",
    })
    table = pa.Table.from_pylist(rows, schema=schema)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tau_snapshot_", dir=output_dir) as temp:
        temp = Path(temp)
        staging = temp / parquet_path.name
        pq.write_table(table, staging, compression="zstd")
        metadata = dict(manifest)
        metadata["output"] = {"path": parquet_path.name, "sha256": hashlib.sha256(staging.read_bytes()).hexdigest(), "num_records": table.num_rows}
        staged_manifest = temp / manifest_path.name
        staged_manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(staging, parquet_path)
        os.replace(staged_manifest, manifest_path)
    return parquet_path, manifest_path


def main(benchmark, argv=None):
    parser = argparse.ArgumentParser(description=f"Export {benchmark} native task snapshots offline; not a training prompt dataset.")
    if benchmark not in DOMAINS:
        parser.error(f"Unsupported benchmark: {benchmark!r}; use t2bench for tau evaluation snapshots.")
    parser.add_argument("--source-root", "--source", dest="source_root", type=Path, default=REPO_ROOT / "data" / benchmark / "source")
    parser.add_argument("--output-dir", type=Path, default=DYAD_ROOT / "data" / benchmark / "dataset")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS[benchmark], default=list(DOMAINS[benchmark]))
    parser.add_argument("--split", choices=(DEFAULT_SPLIT[benchmark],), default=DEFAULT_SPLIT[benchmark])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        rows, manifest = build_snapshot(benchmark, args.source_root, args.domains, args.split)
        parquet_path, manifest_path = write_snapshot(rows, manifest, args.output_dir, args.overwrite)
    except (OSError, ValueError, SyntaxError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"benchmark": benchmark, "split": args.split, "num_records": len(rows), "domain_counts": manifest["domain_counts"], "parquet": str(parquet_path), "manifest": str(manifest_path)}))
    return 0
