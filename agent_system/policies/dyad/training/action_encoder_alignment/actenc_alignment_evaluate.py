"""Explicit, complete Alignment test evaluation of one saved validation-selected checkpoint.

No training, checkpoint search, sample limit, or automatic online publication occurs here.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from agent_system.policies.dyad.data.actenc_alignment_dataset import load_split
from agent_system.policies.dyad.data.actenc_alignment_parquet import SPLIT_POLICY
from agent_system.policies.dyad.models.actenc_alignment_model import AlignmentActionSelector, AlignmentConfig
from agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_evaluation import (
    by_group,
    evaluate,
    file_identity,
)

INTERPRETATION = (
    "Validation and test share the same actions unseen in training, with disjoint cases. "
    "This measures held-out cases in that action pool, not actions unseen in validation. "
    "The complete former test set was used in historical development/model selection; "
    "this split isolates future use only and is not historically untouched."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="explicit projector.pt file")
    parser.add_argument("--dataset", required=True, help="the train/val/test Parquet used for this training")
    parser.add_argument("--out", help="new independent evaluation directory; defaults to ARTIFACT_ROOT/outputs/<site>/alignment_test/<run>_<time>")
    parser.add_argument("--policy-device", help="placement override only; architecture comes from config.yaml")
    parser.add_argument("--encoder-device", help="placement override only")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("independent test uses a single process; do not launch with torchrun")

    checkpoint = Path(args.checkpoint).resolve()
    source_config = checkpoint.parent / "config.yaml"
    record = yaml.safe_load(source_config.read_text())
    if (record.get("split_policy") != SPLIT_POLICY or
            record.get("evaluation_role") != "train-validation"):
        raise ValueError("Independent test requires a new train/val/test validation-selected checkpoint; "
                         "legacy test-selected checkpoints are not an independent final evaluation")
    from agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_config import evaluation_dir
    out = (Path(args.out).expanduser().resolve() if args.out else
           evaluation_dir() / f"{checkpoint.parent.name}_{time.time_ns()}")
    args.out = str(out)
    protected = [checkpoint.parent, Path(record["args"]["out"]).resolve()]
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
        raise ValueError("Evaluation output must be separate from training and checkpoint directories")
    if out.exists():
        raise ValueError(f"Evaluation output already exists: {out}; choose a new directory")
    dataset_identity = file_identity(args.dataset)
    if dataset_identity["sha256"] != record["dataset_identity"]["sha256"]:
        raise ValueError("Evaluation dataset differs from the version pinned by training config")
    rows = load_split(args.dataset, split="test")
    if not rows:
        raise ValueError("Empty independent test split")

    saved_model = record["model_config"]
    model_config = dict(saved_model)
    for name in ("policy_device", "encoder_device"):
        if getattr(args, name) is not None:
            model_config[name] = getattr(args, name)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {"split_policy": SPLIT_POLICY, "evaluation_role": "train-validation",
                "projector": saved_model["projector"], "scale": saved_model["scale"],
                "representation": saved_model["representation"],
                "policy_model": saved_model["policy_model"],
                "encoder_model": saved_model["encoder_model"] or saved_model["policy_model"],
                "policy_hidden": record["policy_hidden"], "encoder_hidden": record["encoder_hidden"],
                "dataset_identity": record["dataset_identity"]}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Checkpoint/config mismatch for {key}")
    model = AlignmentActionSelector(AlignmentConfig(**model_config))
    if model.policy_hidden != payload["policy_hidden"] or model.encoder.hidden_size != payload["encoder_hidden"]:
        raise ValueError("Loaded model hidden sizes differ from the checkpoint")
    model.head.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    protocol = {"split_policy": SPLIT_POLICY, "evaluation_role": "final-test"}
    started = time.time()
    summary = {
        **protocol,
        "checkpoint_identity": file_identity(checkpoint),
        "dataset_identity": dataset_identity,
        "source_config_identity": file_identity(source_config),
        "selected_step": payload["selected_step"],
        "interpretation": INTERPRETATION,
        "test": evaluate(model, rows, args.eval_batch_size),
    }
    for name, key in (("form", "action_set_form"), ("size", "mcp_size"),
                      ("domain", "domain"), ("action", "label")):
        summary[f"test_by_{name}"] = by_group(model, rows, key, args.eval_batch_size)
    summary["wall_seconds"] = time.time() - started
    # An evaluation has exactly one observation, not a checkpoint-selection curve.
    row = {**protocol, "step": 0, "epoch": 0, "test": summary["test"]}
    config = {**protocol, "args": vars(args), "model_config": asdict(model.config),
              "checkpoint_identity": summary["checkpoint_identity"],
              "dataset_identity": dataset_identity}
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (out / "metrics.jsonl").write_text(json.dumps(row, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["test"], sort_keys=True))
    print(f"[alignment-test] wrote {out}; online upload is a separate explicit command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
