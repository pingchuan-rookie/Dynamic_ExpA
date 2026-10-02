#!/usr/bin/env python3
"""Split the existing Alignment test in half, publishing only into a new data root."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import yaml

from agent_system.policies.dyad.data.actenc_alignment_parquet import LEGACY_SPLIT_POLICY, SPLIT_POLICY, read_dataset
from experiments.action_encoder_alignment_dataset import actenc_alignment_generate_dataset as build
from experiments.action_encoder_alignment_dataset import actenc_alignment_generation_config as cfg  # noqa: E402
from experiments.action_encoder_alignment_dataset.actenc_alignment_gen_sample_records import render_sample_records
from experiments.action_encoder_alignment_dataset.actenc_alignment_holdout_split import (
    DEFAULT_SPLIT_SEED,
    split_metadata,
    split_unseen_holdout,
)
from experiments.action_encoder_alignment_dataset.actenc_alignment_parquet_writer import write_dataset

HISTORICAL_USAGE = (
    "The whole source test previously participated in validation/checkpoint selection. "
    "This split isolates future use only, not historical exposure. Val/test share the same "
    "20 unseen-to-train actions; results are case holdouts, not val-unseen action generalization. "
    "Train fresh after fixing the split and select checkpoints using val only."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_shape(rows: list[dict]) -> None:
    counts = dict(Counter(row["split"] for row in rows))
    if counts != {"train": 2100, "test": 560}:
        raise ValueError(f"migration expects train2100/test560, got {counts}")
    strata = defaultdict(set)
    for row in rows:
        if row["split"] == "test":
            strata[row["domain"], row["label"], row["mcp_size"]].add(row["case_id"])
    expected = {(domain, label, size) for domain, labels in cfg.load_action_names(unseen=True).items()
                for label in labels for size in cfg.MCP_SIZES}
    if set(strata) != expected or any(len(cases) != 2 for cases in strata.values()):
        raise ValueError("migration requires two paired cases per configured unseen domain/label/size stratum")
    train = Counter((r["domain"], r["mcp_size"], r["action_set_form"]) for r in rows if r["split"] == "train")
    expected_train = {(domain, size, form): 30 for domain in cfg.DOMAIN_ORDER
                      for size in cfg.MCP_SIZES for form in cfg.FORMS}
    if train != expected_train:
        raise ValueError("migration requires the existing balanced train shape")


def validate_migration(root: Path, manifest: dict, rows: list[dict]) -> None:
    """Verify self-contained source snapshots and exact row-preserving split assignment."""
    migration = manifest["holdout_migration"]
    provenance = root / "intermediate" / "holdout_source"
    source_path = provenance / "dataset.parquet"
    source_manifest_path = provenance / "manifest.yaml"
    if sha256(source_path) != migration["source_dataset_sha256"]:
        raise ValueError("holdout source dataset hash mismatch")
    if sha256(source_manifest_path) != migration["source_manifest_sha256"]:
        raise ValueError("holdout source manifest hash mismatch")
    for name, digest in migration["source_intermediate_sha256"].items():
        if sha256(root / "intermediate" / name) != digest:
            raise ValueError(f"holdout source intermediate hash mismatch: {name}")
    source = read_dataset(source_path, expected_split_policy=LEGACY_SPLIT_POLICY)
    validate_source_shape(source)
    seed = manifest["holdout_split"]["seed"]
    if manifest["holdout_split"] != split_metadata(seed):
        raise ValueError("holdout split algorithm metadata mismatch")
    if rows != split_unseen_holdout(source, seed=seed):
        raise ValueError("holdout migration changed rows beyond the deterministic split field")
    if migration["teacher_calls"] != 0 or not manifest.get("historical_usage"):
        raise ValueError("holdout migration must record zero teacher calls and historical usage")


def migrate(source: Path, output: Path, *, seed: int = DEFAULT_SPLIT_SEED) -> Path:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("output must be an independent new data root, not inside/around the source")
    if output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
    source_dataset = source / "final" / "dataset.parquet"
    source_manifest_path = source / "intermediate" / "manifest.yaml"
    source_manifest = yaml.safe_load(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("split_policy") != LEGACY_SPLIT_POLICY:
        raise ValueError(f"source must use {LEGACY_SPLIT_POLICY}")
    if source_manifest.get("policy_prompt_version") != build.sf.POLICY_PROMPT_VERSION:
        raise ValueError("source has an unsupported policy prompt version; migration never rerenders prompts")
    rows = read_dataset(source_dataset, expected_split_policy=LEGACY_SPLIT_POLICY)
    validate_source_shape(rows)
    source_digest = sha256(source_dataset)
    manifest_digest = sha256(source_manifest_path)
    for name, digest in source_manifest["products"].items():
        if build._sha256(source / name) != digest:
            raise ValueError(f"source product hash mismatch: {name}")
    for name, digest in source_manifest["inputs"].items():
        if name.startswith("intermediate/") and build._sha256(source / name) != digest:
            raise ValueError(f"source input hash mismatch: {name}")
    result = split_unseen_holdout(rows, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    previous_root = os.environ.get("DYAD_ALIGNMENT_DATA")
    try:
        os.environ["DYAD_ALIGNMENT_DATA"] = str(staging)
        cfg.intermediate_dir().mkdir()
        names = ["mcp.yaml", "nl.yaml", "mcp_unseen.yaml", "nl_unseen.yaml",
                 "case_plan.jsonl", "case_plan_unseen.jsonl", "context_reasoning.jsonl",
                 "context_reasoning_unseen.jsonl"]
        for name in names:
            shutil.copy2(source / "intermediate" / name, cfg.intermediate_dir() / name)
        provenance = cfg.intermediate_dir() / "holdout_source"
        provenance.mkdir()
        shutil.copy2(source_dataset, provenance / "dataset.parquet")
        shutil.copy2(source_manifest_path, provenance / "manifest.yaml")
        write_dataset(cfg.dataset_path(), result)
        generation = cfg.load_generation()
        generation["split_seed"] = seed
        build.write_manifest(generation, dict(Counter(r["split"] for r in result)))
        manifest = yaml.safe_load(cfg.manifest_path().read_text(encoding="utf-8"))
        for key in ("prompt_revision", "split_revision"):
            if key in source_manifest:
                manifest[key] = source_manifest[key]
        manifest["historical_usage"] = HISTORICAL_USAGE
        manifest["holdout_migration"] = {
            "source_root": str(source), "source_split_policy": LEGACY_SPLIT_POLICY,
            "source_dataset_sha256": source_digest, "source_manifest_sha256": manifest_digest,
            "source_intermediate_sha256": {name: sha256(source / "intermediate" / name) for name in names},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "teacher_calls": 0, "changes": "split field only; train and physical row order unchanged",
        }
        cfg.manifest_path().write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
        cfg.dataset_path().with_name("SAMPLE_RECORDS.md").write_text(render_sample_records(), encoding="utf-8")
        validate_migration(staging, manifest, read_dataset(cfg.dataset_path()))
        subprocess.run(
            [sys.executable, str(cfg.DATASET_SCRIPT_DIR / "actenc_alignment_validate_dataset.py")], check=True,
            env={**os.environ, "DYAD_ALIGNMENT_DATA": str(staging)},
        )
        if sha256(source_dataset) != source_digest or sha256(source_manifest_path) != manifest_digest:
            raise ValueError("source changed during migration")
        if output.exists():
            raise FileExistsError(f"output appeared during migration: {output}")
        staging.chmod(0o755)
        staging.rename(output)
    finally:
        if previous_root is None:
            os.environ.pop("DYAD_ALIGNMENT_DATA", None)
        else:
            os.environ["DYAD_ALIGNMENT_DATA"] = previous_root
        if staging.exists():
            shutil.rmtree(staging)
    print(f"[holdout] published {SPLIT_POLICY} at {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args(argv)
    try:
        migrate(args.source_root, args.output_root, seed=args.seed)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"[holdout] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
