#!/usr/bin/env python3
# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Export encoder input text for action names and enumerated values in supported schemas."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_DIR))

from agent_system.policies.dyad.models.action_head_factory import encoder_prompts  # noqa: E402
from agent_system.policies.dyad.models.action_descriptions import describe_action_config, known_forms
from agent_system.policies.dyad.actions.schema_config import compile_schema_file  # noqa: E402

from agent_system.utils.artifact_paths import artifact_root

SCHEMA_DIR = PROJECT_DIR / "agent_system/policies/dyad/actions/schemas"
OUT_ROOT = artifact_root(PROJECT_DIR) / "outputs/descriptions"

# Surface form, environment directory, and schema path.
# Separate open and closed value outputs to avoid overwriting them; use a representative CodeGym task.
TARGETS: list[tuple[str, str, str]] = [
    ("base", "gsm8k", "gsm8k/base.yaml"),
    ("base", "alfworld", "alfworld/base.yaml"),
    ("base", "alfworld_closed", "alfworld/base_closed.yaml"),
    ("base", "codegym", "codegym_generated/base/HeapSortEnv.yaml"),
]


def _tokenizer(model_path: str | None):
    """Compile the action configuration with the tokenizer that determines head-row order."""
    from transformers import AutoTokenizer

    if model_path:
        return AutoTokenizer.from_pretrained(model_path)
    hub = Path(
        os.environ.get("HF_HUB_DIR")
        or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    name = os.environ.get("SHOW_DESC_MODEL", "Qwen3.5-2B")
    matches = sorted(hub.glob(f"models--*--{name}/snapshots/*/"))
    if not matches:
        raise SystemExit(
            f"[dump-desc] no local snapshot for {name} under {hub}; pass --model <path>. "
            "A tokenizer is needed because the head's row order depends on it."
        )
    return AutoTokenizer.from_pretrained(str(matches[0]).rstrip("/"))


def _safe(name: str) -> str:
    """Convert an action or value name into a safe filename.

    Strip tokenizer-significant leading spaces only from filenames. Preserve the
    original string in file contents.
    """
    out = "".join(c if (c.isalnum() or c in "-._") else "_" for c in name.strip())
    return out or "_empty_"


def _owner_map(action_config: dict) -> dict:
    """Map each enumerated value to the action slots that reference it."""
    actions = action_config.get("actions", {}) or {}
    value_sets = action_config.get("value_sets", {}) or {}

    value_refs: dict[int, list[tuple[str, str]]] = {}
    for action_name, action in actions.items():
        for slot in action.get("params", []) or []:
            set_name = slot.get("value_set")
            if slot.get("value_kind") == "closed" and set_name:
                for vid in value_sets.get(set_name, []) or []:
                    value_refs.setdefault(int(vid), []).append((action_name, slot["name"]))
    return value_refs


def _write_raw(path: Path, prompt: str) -> None:
    """Write the exact encoder prompt without a header or trailing newline.

    Keep row metadata in the adjacent INDEX.md so byte comparisons and hashes
    measure the same text that encode_hidden receives.
    """
    path.write_text(prompt, encoding="utf-8")


def dump_schema(surface_form: str, env_dir: str, rel: str, tokenizer, forms: list[str]) -> dict:
    path = SCHEMA_DIR / rel
    if not path.exists():
        raise SystemExit(f"[dump-desc] schema not found: {path}")

    cfg = compile_schema_file(tokenizer, len(tokenizer), path)
    docs = describe_action_config(cfg)
    env_name = cfg.get("env_name", env_dir)
    vocab = int(cfg["num_embeddings_size"])
    action_ids = [int(i) for i in cfg["action_ids"]]
    value_refs = _owner_map(cfg)

    stats = {"tool": 0, "parameter": 0, "value": 0, "files": 0}

    for form in forms:
        # Use the training renderer and preserve action_ids/head-row order.
        prompts = encoder_prompts(cfg, description=form)
        if len(prompts) != len(docs):
            raise RuntimeError(
                f"{rel}: {len(prompts)} prompts for {len(docs)} docs -- "
                "prompt 列表与 head 行是同一个有序列表，长度不等说明它们没对齐"
            )

        base = OUT_ROOT / surface_form / env_dir / form
        flat = base / "_head_rows"
        flat.mkdir(parents=True, exist_ok=True)

        for row, (doc, prompt) in enumerate(zip(docs, prompts)):
            stats[doc.kind] = stats.get(doc.kind, 0) + 1

            flat_name = f"{row:03d}_{doc.kind}_{_safe(doc.name)}.txt"
            _write_raw(flat / flat_name, prompt)
            stats["files"] += 1

            if doc.kind == "tool":
                targets = [(base / _safe(doc.name), "_action.txt")]
            else:
                # Group shared closed values by value set, avoiding duplicate copies under every referencing slot.
                prefix, _, value = doc.name.partition(".")
                targets = [(base / "_values" / _safe(prefix), f"{_safe(value)}.txt")]

            for directory, filename in targets:
                directory.mkdir(parents=True, exist_ok=True)
                _write_raw(directory / filename, prompt)
                stats["files"] += 1

        _write_index(base, rel, env_name, form, surface_form, docs, action_ids,
                     value_refs, prompts, vocab, cfg)

    return stats


def _write_index(base: Path, rel: str, env_name: str, form: str, surface_form: str,
                 docs, action_ids, value_refs, prompts, vocab, cfg) -> None:
    head = cfg.get("head", {}) or {}
    lines = [
        f"# {env_name} · surface_form={surface_form} · description form=`{form}`",
        "",
        f"- schema: `agent_system/policies/dyad/actions/schemas/{rel}`",
        f"- head 行数: {len(docs)}",
        f"- argument_order: `{cfg.get('argument_order')}`",
        f"- head 开关: action_name={head.get('action_name')} "
        f"argument_key={head.get('argument_key')} closed_value={head.get('closed_value')}",
        f"- catalogue 长度: {len(prompts[0]) - len(docs[0].prompt(form=form))} 字符（每一行都相同）",
        "",
        "`_head_rows/` 是 head 的真实形状（扁平，一行一个文件）。",
        "其余目录是按 `<action>_<argument_key>/<argument_value>` 展开的读法；",
        "closed value 被多个槽共享，因此会在多个目录下重复出现，看 `head row` 判断是否同一行。",
        "",
        "| row | extended id | kind | name | 归属 | description |",
        "|---|---|---|---|---|---|",
    ]
    for row, doc in enumerate(docs):
        aid = action_ids[row]
        if doc.kind == "value":
            refs = value_refs.get(aid, [])
            owner = ", ".join(f"`{a}.{p}`" for a, p in refs) or "**无人引用**"
        else:
            owner = "-"
        desc = (doc.description or "").replace("|", "\\|").replace("\n", " ")
        if len(desc) > 70:
            desc = desc[:67] + "..."
        lines.append(
            f"| {row} | {aid} | {doc.kind} | `{doc.name.strip() or doc.name!r}` | {owner} | {desc} |"
        )
    lines.append("")
    (base / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="只跑 env 目录名匹配的那些（如 alfworld_closed）")
    ap.add_argument("--form", choices=known_forms(), default=None,
                    help="只生成一种描述形式；默认两种都生成")
    ap.add_argument("--model", default=None, help="tokenizer 路径（head 行序依赖它）")
    args = ap.parse_args()

    forms = [args.form] if args.form else known_forms()
    targets = [t for t in TARGETS if not args.only or args.only in t[1]]
    if not targets:
        raise SystemExit(f"[dump-desc] --only {args.only!r} 匹配不到任何 env；"
                         f"可选: {sorted({t[1] for t in TARGETS})}")

    tokenizer = _tokenizer(args.model)
    print(f"[dump-desc] tokenizer={tokenizer.name_or_path} vocab={len(tokenizer)}")
    print(f"[dump-desc] forms={forms}  out={OUT_ROOT}")
    print()

    total = 0
    skipped: list[tuple[str, str, str]] = []
    for surface_form, env_dir, rel in targets:
        try:
            stats = dump_schema(surface_form, env_dir, rel, tokenizer, forms)
        except Exception as exc:  # noqa: BLE001
            # Report individual schema failures while preserving outputs from successful schemas.
            skipped.append((surface_form, env_dir, f"{type(exc).__name__}: {exc}"))
            print(f"  {surface_form:5s} {env_dir:16s} {rel:44s} SKIPPED")
            continue
        total += stats["files"]
        print(f"  {surface_form:5s} {env_dir:16s} {rel:44s} "
              f"tool={stats['tool']:4d} param={stats['parameter']:4d} "
              f"value={stats['value']:4d}  files={stats['files']}")

    print()
    print(f"[dump-desc] {len(targets) - len(skipped)}/{len(targets)} 份 schema，{total} 个文件 -> "
          f"{OUT_ROOT}/")
    if skipped:
        print()
        print(f"[dump-desc] {len(skipped)} 份跳过（schema 编译不过，与本脚本无关）：")
        for surface_form, env_dir, reason in skipped:
            print(f"    {surface_form}/{env_dir}: {reason}")
        # Partial output must produce a failing exit status.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
