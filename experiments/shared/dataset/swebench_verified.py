"""Prepare allowlisted, evaluation-only SWE-bench Verified agent records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_system.environments.env_package.swebench.assets import (
    DATASET_ID, DATASET_REVISION, FULL_INSTANCE_IDS_SHA256, FULL_TASK_COUNT,
    HARNESS_COMMIT, HARNESS_VERSION, PUBLIC_FIELDS, SOURCE_SHA256,
    canonical_json, file_digest, load_public_tasks, prepare_assets, read_json, verify_assets,
)

FORMAT = "swebench-verified-public-v1"


def _identity(source):
    return {
        "format": FORMAT, "benchmark": "swebench_verified", "split": "test",
        "ready_for_training": False, "columns": list(PUBLIC_FIELDS),
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "source_sha256": SOURCE_SHA256,
        "harness_version": HARNESS_VERSION, "harness_commit": HARNESS_COMMIT,
        "full_task_count": FULL_TASK_COUNT, "full_instance_ids_sha256": FULL_INSTANCE_IDS_SHA256,
        "instance_ids": source["instance_ids"], "num_records": source["num_records"],
        "is_full_verified": source["is_full_verified"], "scope": source["scope"],
    }


def verify(output_dir, *, source_dir, require_full=False):
    """Trusted preflight only: locally authenticate the public data against gold source.

    Invoke outside policy/actor processes, which must read only test.parquet.
    """
    import pyarrow.parquet as pq

    source = verify_assets(source_dir, require_images=False, require_full=require_full)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if file_digest(manifest_path) != (output_dir / "manifest.sha256").read_text().strip():
        raise ValueError("Public dataset manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    for key, value in _identity(source).items():
        if manifest.get(key) != value:
            raise ValueError(f"Public dataset identity mismatch: {key}")
    reference = manifest.get("output", {})
    path = output_dir / "test.parquet"
    if reference.get("path") != path.name or file_digest(path) != reference.get("sha256"):
        raise ValueError("Public dataset parquet SHA256 mismatch")
    table = pq.read_table(path)
    if table.column_names != list(PUBLIC_FIELDS):
        raise ValueError("Public dataset contains non-allowlisted columns")
    rows = table.to_pylist()
    if rows != load_public_tasks(source_dir):
        raise ValueError("Public dataset differs from official allowlisted fields")
    if reference.get("num_records") != len(rows):
        raise ValueError("Public dataset row count mismatch")
    return manifest


def prepare(source_dir, output_dir, *, check=False):
    """Export local source; existing output is verified, never overwritten."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir or source_dir.is_relative_to(output_dir):
        raise ValueError("Public dataset directory must not contain the evaluator asset directory")
    source = verify_assets(source_dir, require_images=False)
    if check or (output_dir.exists() and any(output_dir.iterdir())):
        return verify(output_dir, source_dir=source_dir)
    rows = load_public_tasks(source_dir)
    schema = pa.schema([(key, pa.string()) for key in PUBLIC_FIELDS], metadata={
        b"record_type": b"swebench_verified_public_evaluation_task",
        b"ready_for_training": b"false",
        b"visibility": b"agent_public_allowlist",
    })
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swebench-public-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), staging / "test.parquet", compression="zstd")
        manifest = _identity(source)
        manifest["output"] = {"path": "test.parquet", "sha256": file_digest(staging / "test.parquet"),
                              "num_records": len(rows)}
        (staging / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        (staging / "manifest.sha256").write_text(file_digest(staging / "manifest.json") + "\n", encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("test.parquet", "manifest.json", "manifest.sha256"):
            with (output_dir / name).open("xb") as stream:
                stream.write((staging / name).read_bytes())
    return manifest


def prepare_data(env, project_dir, overrides, *, check_only=False, manifest=None):
    """Shared launch hook: verification only, never data acquisition or training."""
    project_dir = Path(project_dir)
    source_dir = Path(env.get("SWE_BENCH_ASSET_DIR") or project_dir / "data/swebench_verified/assets").resolve()
    test = Path(env.get("TEST_DATA") or project_dir / "data/swebench_verified/dataset/test.parquet").resolve()
    if test.name != "test.parquet":
        raise ValueError("SWE-bench Verified requires audited test.parquet, never training input")
    manifest = verify(test.parent, source_dir=source_dir)
    report = {"manifest": str(test.parent / "manifest.json"),
              "manifest_sha256": file_digest(test.parent / "manifest.json"),
              "asset_manifest_sha256": file_digest(source_dir / "manifest.json"),
              "scope": manifest["scope"], "num_records": manifest["num_records"],
              "is_full_verified": manifest["is_full_verified"],
              "dataset_revision": DATASET_REVISION, "harness_commit": HARNESS_COMMIT,
              "prompt_lengths_checked": False}
    env.update(SWE_BENCH_ASSET_DIR=str(source_dir), TEST_DATA=str(test),
               SWE_BENCH_DATA=json.dumps(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "verify"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--source-dir", "--asset-dir", dest="source_dir", type=Path,
                                    default=ROOT / "data/swebench_verified/assets")
        command_parser.add_argument("--output-dir", type=Path, default=ROOT / "data/swebench_verified/dataset")
        if command == "prepare":
            command_parser.add_argument("--download", action="store_true", help="Explicitly acquire pinned official data")
            command_parser.add_argument("--source-file", type=Path, help="Import pinned parquet without networking")
            command_parser.add_argument("--instance-ids", nargs="+", help="Debug selection; default is all 500")
        else:
            command_parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.download or args.source_file is not None:
            prepare_assets(args.source_dir, instance_ids=args.instance_ids, download=args.download,
                           source_file=args.source_file)
        elif args.instance_ids is not None:
            parser.error("--instance-ids is used only when explicitly preparing assets with --download/--source-file")
        manifest = prepare(args.source_dir, args.output_dir)
    else:
        manifest = verify(args.output_dir, source_dir=args.source_dir, require_full=args.require_full)
    print(json.dumps({"scope": manifest["scope"], "num_records": manifest["num_records"],
                      "manifest_sha256": file_digest(args.output_dir / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
