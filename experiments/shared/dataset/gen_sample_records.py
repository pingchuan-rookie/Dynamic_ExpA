#!/usr/bin/env python3
# Copyright 2025 ExpA_verl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Render real records from shared Agentic RL datasets and external evaluation snapshots.

Reads data/<env>/dataset/*.parquet.
Use --check to verify a report without
writing it, and --env to limit the environments. No model inference is performed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

DYAD_ROOT = Path(__file__).resolve().parents[3]
if str(DYAD_ROOT) not in sys.path:
    sys.path.insert(0, str(DYAD_ROOT))
DATA_ROOT = DYAD_ROOT / "data"
SPEC_PATH = Path(__file__).resolve().parent / "sample_records_spec.yaml"
DEFAULT_MODEL_NAME = "Qwen3.5-2B"
EXTENSIONS = ("parquet", "jsonl")
SOURCE_REPORT_ENVS = frozenset({"alfworld", "codegym", "gsm8k"})
REDACTED = "<redacted>"
PUBLIC_IDENTITY_FIELDS = frozenset({
    "dataset_format", "index", "split", "source_split", "source_index", "source_id",
    "task_id", "task_type", "code_id", "env_name",
})
SOURCE_LOCATOR_FIELDS = frozenset({
    "path", "file", "row", "row_index", "line", "line_number", "task_id", "task_type",
    "split", "source_split", "sha256", "file_sha256", "record_sha256",
})


# --------------------------------------------------------------------------- inputs


def environment_dir(env: str) -> Path:
    """Resolve a retained dataset's directory, including evaluation-only datasets."""
    return DATA_ROOT / env


def display_dataset_path(env: str) -> str:
    return f"data/{env}/dataset/"


def load_spec(path: Path) -> tuple[dict, dict]:
    """Read the spec, keeping the configured environment order."""
    with path.open(encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    if not isinstance(spec, dict) or "envs" not in spec or "labels" not in spec:
        sys.exit(f"spec {path} must be a mapping with 'labels' and 'envs'")
    return spec["labels"], spec["envs"]


def resolve_model() -> str:
    """MODEL_PATH wins; otherwise resolve MODEL_NAME against the local HF cache.

    Several snapshots of one model can coexist in the cache, so the lowest revision
    hash is picked rather than whatever the filesystem lists first - the choice has
    to be stable across runs for --check to mean anything.
    """
    explicit = os.environ.get("MODEL_PATH")
    if explicit:
        return explicit.rstrip("/")
    name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME).split("/")[-1]
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    cands = sorted(c for c in glob.glob(os.path.join(hub, f"models--*--{name}", "snapshots", "*"))
                   if os.path.isdir(c))
    if not cands:
        sys.exit(f"model {name} not found under {hub}; set MODEL_PATH or MODEL_NAME")
    return cands[0]


def model_tag(model: str) -> str:
    """Display name of the tokenizer: the repo name when the path is a HF cache snapshot."""
    hit = next((p.split("--", 2)[-1] for p in Path(model).parts if p.startswith("models--")), None)
    return hit or os.path.basename(model.rstrip("/"))


def split_path(vdir: Path, split: str) -> Path | None:
    """Locate one shared split, preferring parquet over the jsonl mirror."""
    for ext in EXTENSIONS:
        fp = vdir / f"{split}.{ext}"
        if fp.exists():
            return fp
    return None


def load_split(fp: Path):
    import datasets
    if fp.suffix == ".parquet":
        return datasets.Dataset.from_parquet(str(fp))
    return datasets.Dataset.from_json(str(fp))


# --------------------------------------------------------------------------- stats

def to_messages(prompt) -> list[dict]:
    """Normalise the prompt column (list of {role, content}) into plain dicts."""
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt]


def prompt_token_lengths(tokenizer, ds, limit: int | None) -> np.ndarray:
    """Token length of stored public input after chat templating, not runtime steps."""
    from agent_system.utils.thinking import resolve_chat_template_kwargs
    n = len(ds) if limit is None else min(limit, len(ds))
    template_kwargs = resolve_chat_template_kwargs(tokenizer=tokenizer)

    # Format to text first (special tokens land as literal text), then tokenize the
    # batch with add_special_tokens=False since the template already emitted them.
    texts = [
        tokenizer.apply_chat_template(to_messages(ds[i]["prompt"]), add_generation_prompt=True,
                                      tokenize=False, **template_kwargs)
        for i in range(n)
    ]
    enc = tokenizer(texts, add_special_tokens=False)
    return np.array([len(x) for x in enc["input_ids"]], dtype=np.int64)


def stats_row(split: str, arr: np.ndarray, thresholds: list[int]) -> str:
    def pct(p: int) -> int:
        return int(np.percentile(arr, p))
    over = "".join(f" {int((arr > t).sum())} |" for t in thresholds)
    return (
        f"| {split} | {len(arr)} | {int(arr.min())} | {arr.mean():.1f} | {int(np.median(arr))} | "
        f"{pct(90)} | {pct(95)} | {pct(99)} | {int(arr.max())} |{over}"
    )


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _public_scalar(value):
    """Do not recursively expose metadata attached to an otherwise public key."""
    if isinstance(value, np.generic):
        value = value.item()
    return value if value is None or isinstance(value, (str, int, float, bool)) else REDACTED


def source_report_record(row: dict) -> dict:
    """Project transport rows onto an explicit report allowlist, without opening sources.

    Source snapshots, reset parameters, reward targets and abilities are private even
    when nested under metadata. Unknown fields fail closed rather than being dumped.
    """
    extra = row.get("extra_info")
    extra = extra if isinstance(extra, dict) else {}
    public_extra = {key: _public_scalar(value) for key, value in extra.items()
                    if key in PUBLIC_IDENTITY_FIELDS}
    source = extra.get("source_record")
    public_extra["source_record"] = (
        {key: _public_scalar(value) for key, value in source.items() if key in SOURCE_LOCATOR_FIELDS}
        if isinstance(source, dict) else REDACTED
    )
    public_extra["private_fields"] = REDACTED
    # Select only the declared public message columns, not message-level metadata.
    for message in row["prompt"]:
        if message["role"] not in {"system", "user"} or not isinstance(message["content"], str):
            raise ValueError("Source report requires public system/user text messages")
    return {
        "data_source": _public_scalar(row.get("data_source")),
        "index": _public_scalar(row.get("index")),
        "prompt": to_messages(row["prompt"]),
        "ability": REDACTED,
        "reward_model": REDACTED,
        "extra_info": public_extra,
    }


def sample_json(ds, idx: int = 0, *, env: str | None = None) -> str:
    row = dict(ds[idx])
    extra = row.get("extra_info")
    if env in SOURCE_REPORT_ENVS or (isinstance(extra, dict) and extra.get("dataset_format") == "source_tasks_v1"):
        row = source_report_record(row)
    return json.dumps(row, ensure_ascii=False, indent=2, default=_json_default)


# --------------------------------------------------------------------------- format

def format_env(env: str, cfg: dict, lb: dict, tokenizer, tag: str, limit: int | None) -> str:
    data_dir = environment_dir(env) / "dataset"
    if env in SOURCE_REPORT_ENVS:
        lb = {**lb, **lb["source_tasks"]}
    splits, thresholds = cfg["splits"], cfg["thresholds"]
    scope = lb["scope_limit"].format(limit=limit) if limit else lb["scope_all"]
    lines = [f"# {cfg['title']}", "", "> " + lb["header_note"].format(tag=tag, scope=scope)]
    for para in cfg["intro"]:
        lines += [">", "> " + para]
    lines += ["", lb["path_prefix"] + f"`{display_dataset_path(env)}`", "", lb["stat_intro"], "",
              lb["stat_columns"] + "".join(f" >{t} |" for t in thresholds),
              "|---|---|---|---|---|---|---|---|---|" + "---|" * len(thresholds)]
    sample = None
    for split in splits:
        path = split_path(data_dir, split)
        if path is None:
            if split == "test_unseen":
                continue
            raise ValueError(f"Missing required shared split: {data_dir}/{split}.parquet")
        print(f"[gen] {env}/{split}", flush=True)
        ds = load_split(path)
        lengths = prompt_token_lengths(tokenizer, ds, limit)
        lines.append(stats_row(split, lengths, thresholds))
        if split == "train":
            sample = (sample_json(ds, env=env), int(lengths[0]))
    if sample:
        lines += ["", lb["sample_line"].format(tag=lb["sample_tag"].format(n=sample[1])), "",
                  "```json", sample[0], "```", ""]
    return "\n".join(lines) + "\n"


def format_benchmark(env: str, cfg: dict) -> str:
    """Summarise interactive task snapshots without treating hidden tasks as prompts."""
    from collections import Counter

    import pyarrow.parquet as pq

    lines = [f"# {cfg['title']}", "",
             "> 由 `experiments/shared/dataset/gen_sample_records.py` 自动生成。",
             "> 使用 `--check` 校验，不要手改。", ""]
    lines.extend(cfg["intro"])
    lines += ["", f"路径：`{display_dataset_path(env)}`", "",
              "| split | domain | 条数 |", "|---|---|---:|"]
    samples = []
    for split in cfg["splits"]:
        path = environment_dir(env) / "dataset" / f"{split}.parquet"
        rows = pq.read_table(path).to_pylist()
        if not rows:
            raise ValueError(f"Empty evaluation split: {path}")
        counts = Counter(row["domain"] for row in rows)
        for domain, count in sorted(counts.items()):
            lines.append(f"| {split} | {domain} | {count} |")
            row = next(row for row in rows if row["domain"] == domain)
            samples.append((split, domain, row))
    for split, domain, row in samples:
        lines += ["", f"## {domain} / {split} 首条真实记录", "",
                  "```json", json.dumps(row, ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- cli

def main() -> int:
    labels, envs_spec = load_spec(SPEC_PATH)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", action="append", choices=sorted(envs_spec),
                    help="only build these environments (repeatable; default: all)")
    ap.add_argument("--out", default=None,
                    help="write here instead of the dataset directory’s SAMPLE_RECORDS.md "
                         "(needs exactly one --env)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only tokenize the first N rows of each split (faster; default: all)")
    ap.add_argument("--model", default=None,
                    help="tokenizer path (default: $MODEL_PATH, else $MODEL_NAME in the HF cache)")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 if any report differs from what would be generated")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")

    envs = args.env or sorted(envs_spec)
    if args.out and len(envs) != 1:
        ap.error("--out needs exactly one --env")

    tokenizer, tag = None, ""
    if any(envs_spec[env].get("kind") != "benchmark" for env in envs):
        model = args.model or resolve_model()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        tag = model_tag(model)

    stale: list[Path] = []
    for env in envs:
        env_dir = environment_dir(env)
        if not env_dir.is_dir():
            if args.env:
                raise ValueError(f"Missing requested dataset: {env_dir}")
            print(f"[gen] skip {env}: {env_dir} does not exist", flush=True)
            continue
        cfg = envs_spec[env]
        text = (format_benchmark(env, cfg) if cfg.get("kind") == "benchmark"
                else format_env(env, cfg, labels, tokenizer, tag, args.limit))
        out = Path(args.out) if args.out else env_dir / "SAMPLE_RECORDS.md"
        if args.check:
            current = out.read_text(encoding="utf-8") if out.exists() else None
            if current == text:
                print(f"[check] ok    {out}", flush=True)
            else:
                stale.append(out)
                print(f"[check] STALE {out}", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[gen] wrote {out} ({out.stat().st_size} bytes)", flush=True)

    if stale:
        print(f"\n{len(stale)} report(s) out of date; re-run without --check:", flush=True)
        for path in stale:
            print(f"  {path}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
