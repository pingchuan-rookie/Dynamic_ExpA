#!/usr/bin/env python3
"""Render complete, real Alignment train MCP and NL records without model inference.

Use --check to reject missing or stale documentation without writing any files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from agent_system.policies.dyad.data.actenc_alignment_parquet import read_dataset  # noqa: E402
from experiments.action_encoder_alignment_dataset import actenc_alignment_generation_config as cfg  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_sample_records(dataset: Path | str | None = None) -> str:
    """Read Parquet and return deterministic Markdown, without writing anything.

    Select the first train record of each form in physical, zero-based file order
    and keep every field, including split and all candidate definitions/encoder prompts.
    The filename and SHA-256 identify the source without machine-specific paths.
    None resolves cfg.dataset_path() at call time, including DYAD_ALIGNMENT_DATA.
    """
    path = Path(dataset) if dataset is not None else cfg.dataset_path()
    digest = _sha256(path)
    rows = read_dataset(path)
    if _sha256(path) != digest:
        raise ValueError(f"Dataset changed while rendering sample records: {path}")
    lines = [
        "# Action Encoder Alignment 完整真实样本",
        "",
        "由 `experiments/action_encoder_alignment_dataset/actenc_alignment_gen_sample_records.py` "
        "从实际 Parquet 自动生成。",
        "不要手工修改，使用 `--check` 检查与数据的一致性。",
        "按文件物理行序分别选取第一条 train MCP 和 NL 记录，JSON 保留全部字段和值。",
        "",
        f"- 数据文件：`{path.name}`",
        f"- 数据 SHA-256：`{digest}`",
        "",
    ]
    for name, form in (("MCP", cfg.FORM_MCP), ("NL", cfg.FORM_NL)):
        selected = next(
            ((index, row) for index, row in enumerate(rows)
             if row["split"] == "train" and row["action_set_form"] == form),
            None,
        )
        if selected is None:
            raise ValueError(f"No train {name} ({form}) record in {path}")
        index, row = selected
        lines.extend([
            f"## {name} 样本",
            "",
            f"- split：`{row['split']}`",
            f"- 物理行号（0-based）：`{index}`",
            f"- case_id：`{row['case_id']}`",
            "",
            "```json",
            json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False),
            "```",
            "",
        ])
    return "\n".join(lines)


def _normalize_generator_reference(document: bytes) -> bytes:
    """Accept historical headings and generator paths, preserving all sample content."""
    for previous_title in ("Stage 1", "Alignment"):
        old_title = f"# {previous_title} 完整真实样本\n".encode()
        if document.startswith(old_title):
            document = "# Action Encoder Alignment 完整真实样本\n".encode() + document[len(old_title):]
    current = "experiments/action_encoder_alignment_dataset/actenc_alignment_gen_sample_records.py"
    for previous in ("experiments/stage1_dataset/gen_sample_records.py",
                     "experiments/alignment_dataset/gen_sample_records.py"):
        old_line = f"\n由 `{previous}` 从实际 Parquet 自动生成。\n".encode()
        new_line = f"\n由 `{current}` 从实际 Parquet 自动生成。\n".encode()
        document = document.replace(old_line, new_line, 1)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Parquet input (default: cfg.dataset_path())")
    parser.add_argument("--output", type=Path, help="Markdown output (default: SAMPLE_RECORDS.md beside input)")
    parser.add_argument("--check", action="store_true", help="Write nothing; exit 1 if documentation is missing or stale")
    args = parser.parse_args(argv)
    dataset = args.dataset if args.dataset is not None else cfg.dataset_path()
    output = args.output if args.output is not None else dataset.with_name("SAMPLE_RECORDS.md")
    try:
        if output.resolve() == dataset.resolve() or (output.exists() and output.samefile(dataset)):
            raise ValueError("Sample output must not overwrite the source dataset")
        expected = render_sample_records(dataset).encode("utf-8")
        if args.check:
            if not output.is_file() or _normalize_generator_reference(output.read_bytes()) != expected:
                print(f"[check] STALE {output}; re-run without --check", file=sys.stderr)
                return 1
            print(f"[check] ok {output}")
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(expected)
        print(f"[gen] wrote {output}")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"[sample-records] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
