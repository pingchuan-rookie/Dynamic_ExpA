"""Prepare auditable official DIVE RL/Eval records without synthesizing tasks."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_system.environments.backends.dive.dataset import (
    DEFAULT_DATA_DIR, SELECTED_DOMAINS, InvalidPublishedTask, SOURCE_FILES, read_task_rows, task_digest, validate_task,
)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_rows(source_dir, *, drop_invalid=False):
    source_dir = Path(source_dir)
    download = json.loads((source_dir / "download_manifest.json").read_text())
    entries = {}
    for dataset in download["datasets"]:
        for item in dataset["files"]:
            if item["path"] in entries:
                raise ValueError(f"Duplicate source manifest entry: {item['path']}")
            entries[item["path"]] = (dataset, item)
    splits, provenance, identities = {}, {}, set()
    for split, relative in SOURCE_FILES.items():
        dataset, expected = entries[relative]
        path = source_dir / relative
        digest = file_digest(path)
        if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Source hash/size mismatch: {path}")
        if not dataset.get("revision"):
            raise ValueError(f"Missing pinned source revision: {path}")
        rows, exclusions, filtered, original_count = [], [], [], 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                original_count += 1
                task = json.loads(line)
                trace_id = task.get("trace_id") if isinstance(task, dict) else None
                if not isinstance(trace_id, str) or not trace_id or trace_id in identities:
                    raise ValueError(f"{path}:{line_number}: missing or duplicate trace_id")
                identities.add(trace_id)
                try:
                    validate_task(task, source=f"{relative}:{line_number}")
                except InvalidPublishedTask as exc:
                    if task["metadata"]["domain"] not in SELECTED_DOMAINS:
                        pass  # Outside this experiment, regardless of placeholder policy.
                    elif not drop_invalid:
                        raise ValueError(f"{exc}; use --drop-invalid to explicitly exclude known placeholders") from exc
                    else:
                        exclusions.append({"source_line": line_number, "trace_id": trace_id,
                                           "task_sha256": task_digest(task), "reason": str(exc)})
                        continue
                if task["metadata"]["domain"] not in SELECTED_DOMAINS:
                    filtered.append({"source_line": line_number, "trace_id": trace_id,
                                     "domain": task["metadata"]["domain"],
                                     "task_sha256": task_digest(task), "reason": "domain_not_selected"})
                    continue
                rows.append({"split": split, "trace_id": trace_id,
                             "domain": task["metadata"]["domain"], "source_line": line_number,
                             "source_sha256": digest, "source_revision": dataset["revision"],
                             "task_sha256": task_digest(task), "task_json": line.rstrip("\r\n")})
        if original_count != expected["rows"]:
            raise ValueError(f"Source row count mismatch: {path}")
        if not rows:
            raise ValueError(f"No valid DIVE tasks in {path}")
        splits[split] = rows
        provenance[split] = {"source": relative, "repo_id": dataset["repo_id"],
                             "source_revision": dataset["revision"], "source_sha256": digest,
                             "original_count": original_count, "effective_count": len(rows),
                             "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
                             "excluded": exclusions, "filtered_domains": filtered}
    return splits, provenance


def prepare(source_dir, output_dir, *, drop_invalid=False, check=False, overwrite=False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if check and overwrite:
        raise ValueError("--check and --overwrite are mutually exclusive")
    splits, provenance = source_rows(source_dir, drop_invalid=drop_invalid)
    expected = {"format": "dive-runtime-v1", "drop_invalid": drop_invalid,
                "selected_domains": list(SELECTED_DOMAINS), "splits": provenance}
    output_dir = Path(output_dir)
    names = ["train.parquet", "test.parquet", "manifest.json"]
    targets = [output_dir / name for name in names]
    existing = [path.exists() for path in targets]
    if any(existing):
        if not all(existing):
            raise ValueError("Partial DIVE output exists; preserve it and choose a new output directory")
        manifest = json.loads(targets[-1].read_text())
        if manifest.get("format") != expected["format"]:
            raise ValueError("Refusing to replace an unrecognized dataset")
        for split in SOURCE_FILES:
            path = output_dir / f"{split}.parquet"
            if file_digest(path) != manifest["outputs"][path.name]["sha256"]:
                raise ValueError(f"Existing output hash mismatch: {path}; preserve and inspect it")
        if not overwrite:
            if {key: manifest.get(key) for key in expected} != expected:
                raise ValueError("Existing DIVE output uses different sources/options; choose a new directory")
            for split, rows in splits.items():
                if read_task_rows(output_dir / f"{split}.parquet") != rows:
                    raise ValueError(f"Existing DIVE output differs from its source: {split}")
            return manifest
    elif check:
        raise FileNotFoundError(f"Missing prepared DIVE data: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dive-", dir=output_dir) as temporary:
        staging = Path(temporary)
        expected["outputs"] = {}
        for split, rows in splits.items():
            name = f"{split}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), staging / name, compression="zstd")
            if read_task_rows(staging / name) != rows:
                raise ValueError(f"DIVE serialization changed rows: {split}")
            expected["outputs"][name] = {"sha256": file_digest(staging / name), "rows": len(rows)}
        (staging / "manifest.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
        for name in names:
            os.replace(staging / name, output_dir / name)
    return expected


def prepare_data(env, project_dir, overrides, *, check_only=False, manifest=None):
    """Validate complete prepared sources before shared launch applies loader limits."""
    project_dir = Path(project_dir)
    source = Path(env.get("DIVE_SOURCE_DIR") or project_dir / "data/dive").resolve()
    default_dir = project_dir / DEFAULT_DATA_DIR
    train = Path(env.get("TRAIN_DATA") or default_dir / "train.parquet").resolve()
    test = Path(env.get("TEST_DATA") or default_dir / "test.parquet").resolve()
    if train.parent != test.parent or train.name != "train.parquet" or test.name != "test.parquet":
        raise ValueError("DIVE requires train.parquet/test.parquet from one audited dataset directory")
    manifest_path = train.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Prepare DIVE data explicitly with experiments/shared/dataset/dive.py; "
                                "use --drop-invalid only when excluding known placeholders is intended")
    declared = json.loads(manifest_path.read_text())
    if type(declared.get("drop_invalid")) is not bool:
        raise ValueError("DIVE manifest must explicitly declare its invalid-row policy")
    if manifest is None:
        manifest = prepare(source, train.parent, drop_invalid=declared["drop_invalid"], check=True)
    env.update(TRAIN_DATA=str(train), TEST_DATA=str(test))
    # Keep per-row exclusions in the manifest, not a potentially oversized env var.
    summaries = {}
    for split, audit in manifest["splits"].items():
        summaries[split] = {key: value for key, value in audit.items() if key != "filtered_domains"}
        summaries[split]["filtered_domain_counts"] = dict(sorted(Counter(
            row["domain"] for row in audit.get("filtered_domains", [])).items()))
    report = {"manifest": str(manifest_path), "manifest_sha256": file_digest(manifest_path),
              "selected_domains": manifest["selected_domains"],
              "drop_invalid": manifest["drop_invalid"], "splits": summaries,
              "prompt_lengths_checked": False}
    if not check_only:
        from transformers import AutoTokenizer
        from agent_system.environments.prompts.dive import build_dive_messages
        from agent_system.parsers.dive import native_protocol
        from agent_system.rollout.prompt import strip_thinking_prefill
        from agent_system.utils.thinking import resolve_chat_template_kwargs
        tokenizer = AutoTokenizer.from_pretrained(env["MODEL_PATH"], local_files_only=True)
        template_kwargs = resolve_chat_template_kwargs(model=env["MODEL_PATH"], tokenizer=tokenizer)
        native_protocol(tokenizer)
        budget = int(env["MAX_PROMPT_LENGTH"])
        lengths = {}
        for split, path in (("train", train), ("test", test)):
            maximum = 0
            for row in read_task_rows(path):
                task = json.loads(row["task_json"])
                prompt = build_dive_messages(task["query"], task["tools"])
                ids = tokenizer.apply_chat_template(prompt, tools=task["tools"], tokenize=True,
                                                     add_generation_prompt=True, return_dict=False, **template_kwargs)
                length = len(strip_thinking_prefill(list(ids), tokenizer))
                maximum = max(maximum, length)
                if length > budget:
                    raise ValueError(f"DIVE {split} task {row['trace_id']} needs {length} prompt tokens, "
                                     f"exceeding MAX_PROMPT_LENGTH={budget}; no task was dropped")
            lengths[split] = maximum
        report.update(prompt_lengths_checked=True, maximum_prompt_tokens=lengths)
    env["DIVE_DATA"] = json.dumps(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/dive")
    parser.add_argument("--output-dir", type=Path, default=ROOT / DEFAULT_DATA_DIR)
    parser.add_argument("--drop-invalid", action="store_true",
                        help="Explicitly exclude known failed-generation placeholders and record each exclusion")
    parser.add_argument("--check", action="store_true", help="Validate existing outputs against complete sources")
    parser.add_argument("--overwrite", action="store_true", help="Replace only recognized, unmodified DIVE outputs")
    args = parser.parse_args(argv)
    manifest = prepare(**vars(args))
    print(json.dumps({split: {key: value[key] for key in ("original_count", "effective_count", "domains")}
                      for split, value in manifest["splits"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
