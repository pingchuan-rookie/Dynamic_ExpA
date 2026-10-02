#!/usr/bin/env python3
# Copyright 2025 ExpA_verl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Build the shared codegym dataset for GRPO-ReAct and Dyad.

Without a subcommand, build the dataset using the options below.
Use ``schema`` to export action schemas or ``select-test`` to select rows for offline analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import pyarrow.parquet as pq

# veRL tool name (aligned with tool_name in the tool config yaml and TOOL_NAME in the parser).
WRAPPER_TOOL_NAME = "codegym_call"

# Repository data/ root (codegym.py lives in experiments/shared/dataset/).
_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

# Task-source subdirectory; sibling envs_en definitions are also preserved and verified.
_HF_TASK_SUBDIR = "task_en_instruction_en_env"

# Keep direct script execution and import-by-path on the same shared prompt builder.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_system.environments.prompts.codegym import build_codegym_messages
from experiments.shared.dataset.utils import source_snapshot as snapshot


def public_messages(messages: list[dict]) -> list[dict]:
    """Only source declarations preceding the complete first user instruction.

    Oracle conversation turns remain preserved in the source shard, never in the
    active public messages. Known local suffix removal is exact, not heuristic.
    """
    public = []
    for message in build_codegym_messages(messages):
        if message.get("role") == "system":
            public.append(message)
        elif message.get("role") == "user":
            public.append(message)
            break
        else:
            raise ValueError("Unexpected private turn before CodeGym public task")
    if not public or public[-1].get("role") != "user" or not any(m["role"] == "system" for m in public):
        raise ValueError("CodeGym requires public function declarations and first user task")
    return public


def parse_env_name(ability: str) -> str | None:
    """Extract EnvName from the ability ``codegym_v1@<code_id>__<EnvName>@<json>``."""
    try:
        mid = ability.split("@", 2)[1]          # <code_id>__<EnvName>
        return mid.split("__", 1)[1]            # <EnvName>
    except (IndexError, AttributeError):
        return None


# ===========================================================================
# Building one shared row
# ===========================================================================

def build_row(lite: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    ability = lite["ability"]
    prompt = public_messages(lite["prompt"])

    tools_kwargs = {
        WRAPPER_TOOL_NAME: {
            "create_kwargs": {
                "create_payload": {"env_str": ability},
            }
        }
    }

    extra_info: dict[str, Any] = {
        "index": index,
        "split": split,
        "need_tools_kwargs": True,
        "env_name": parse_env_name(ability) or "unknown",
        "code_id": lite.get("code_id"),
        "solve_fc_round": lite.get("solve_fc_round"),
        # Preserve the private source reference; rewards come from env.step, not this text.
        "ref_answer": lite.get("answer"),
        "tools_kwargs": tools_kwargs,
    }

    return {
        "data_source": lite.get("data_source", "codegym_v1"),
        "index": index,
        "prompt": prompt,
        "ability": ability,
        "reward_model": dict(lite["reward_model"]),
        # The dataset loader reads tool creation kwargs from extra_info.
        "extra_info": extra_info,
    }


# ===========================================================================
# Selecting tasks for offline analysis from the existing global test split
# ===========================================================================

def select_lites(lites: list[dict[str, Any]], env_name: str | None,
                 code_id: str | None) -> list[dict[str, Any]]:
    """Filter by optional environment and code identity, preserving order and all matches."""
    out = lites
    if env_name:
        out = [x for x in out if parse_env_name(x["ability"]) == env_name]
        if not out:
            names = sorted({parse_env_name(x["ability"]) or "?" for x in lites})
            raise SystemExit(f"--env-name {env_name!r} matched no task; available: {names[:20]}"
                             f"{' ...' if len(names) > 20 else ''}")
    if code_id:
        out = [x for x in out if str(x.get("code_id")) == code_id]
        if not out:
            ids = sorted({str(x.get("code_id")) for x in lites
                          if not env_name or parse_env_name(x["ability"]) == env_name})
            raise SystemExit(f"--code-id {code_id!r} matched no task; available: {ids[:20]}"
                             f"{' ...' if len(ids) > 20 else ''}")
    return out


# ===========================================================================
# Action schema export
# ===========================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
JSONL = Path(os.environ.get("CODEGYM_ACTIONS_JSONL", REPO_ROOT / "data/codegym/metadata/codegym_env_actions.jsonl"))
OUTDIR = REPO_ROOT / "agent_system" / "policies" / "dyad" / "actions" / "schemas" / "codegym_generated"


def load_envs():
    envs = {}
    with open(JSONL) as f:
        for line in f:
            d = json.loads(line)
            envs[d["env_name"]] = d
    return envs


def gen_mcp(env: dict) -> dict:
    """Build semantic action definitions from upstream names and parameter types."""
    tools = []
    for action_name in env["actions"]:
        params = env["params"].get(action_name, [])
        properties = {}
        for pname, ptype in params:
            properties[pname] = {
                "type": _json_type(ptype),
                "description": f"{pname} ({ptype})",
            }
        tools.append({
            "name": action_name,
            "description": (f"call {action_name} with {len(params)} param(s)"
                            if params else f"call {action_name}"),
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        })
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "env": env.get("env_name", ""),
        "tools": tools,
        "$defs": {},
    }


#: Upstream type names map to JSON Schema types; unknown types stay strings.
_JSON_TYPES = {"int": "integer", "float": "number", "bool": "boolean", "str": "string"}


def _json_type(ptype: str) -> str:
    return _JSON_TYPES.get(str(ptype).strip().lower(), "string")


def schema_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog=f"{Path(__file__).name} schema",
        description="Generate native JSON action prefixes and MCP definitions for CodeGym.",
    )
    ap.add_argument("--envs_file", help="JSON array of environment names; required with upstream JSONL")
    ap.add_argument("--from_mcp", type=Path, metavar="DIR",
                    help="Generate native schemas from existing MCP definitions, preserving all parameter metadata")
    ap.add_argument("--outdir", default=OUTDIR)
    cfg = ap.parse_args(argv)

    # Dataset conversion, offline selection, help and import-by-path do not need
    # schema dependencies or an installed dyad package.
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from agent_system.policies.dyad.actions.codegym_schema import build_codegym_schema

    source_mcps = {}
    if cfg.from_mcp:
        all_envs = {}
        for path in sorted(cfg.from_mcp.glob("*.json")):
            document = json.loads(path.read_text())
            tools = document.get("tools") or []
            if not tools:
                ap.error(f"{path} contains no action definitions")
            name = path.stem
            source_mcps[name] = document
            all_envs[name] = {
                "env_name": name,
                "actions": [tool["name"] for tool in tools],
                "params": {tool["name"]: list((tool.get("inputSchema") or {}).get("properties", {}).items())
                           for tool in tools},
            }
        if not all_envs:
            ap.error(f"No MCP definitions found in {cfg.from_mcp}")
        names = json.loads(Path(cfg.envs_file).read_text()) if cfg.envs_file else sorted(all_envs)
    else:
        if not cfg.envs_file:
            ap.error("Use --envs_file with upstream JSONL, or --from_mcp DIR")
        if not JSONL.is_file():
            ap.error(f"Upstream definitions missing: {JSONL}; use --from_mcp DIR")
        all_envs = load_envs()
        names = json.loads(Path(cfg.envs_file).read_text())
    if not isinstance(names, list) or not names or any(name not in all_envs for name in names):
        ap.error("Environment selection must be a nonempty list of names present in the source")

    mcp_dir = os.path.join(cfg.outdir, "mcp")
    os.makedirs(mcp_dir, exist_ok=True)
    for name in names:
        with open(os.path.join(mcp_dir, f"{name}.json"), "w") as f:
            json.dump(source_mcps[name] if name in source_mcps else gen_mcp(all_envs[name]),
                      f, indent=2, ensure_ascii=False)
            f.write("\n")
    print(f"[done] mcp: {len(names)} json written to {mcp_dir}\n")

    sub = Path(cfg.outdir) / "base"
    sub.mkdir(parents=True, exist_ok=True)
    for name in names:
        out_path = sub / f"{name}.yaml"
        with out_path.open("w") as f:
            yaml.safe_dump(build_codegym_schema(all_envs[name]), f, sort_keys=False,
                           allow_unicode=True, width=200)
    print(f"[done] base: {len(names)} yaml written to {sub}")


# ===========================================================================
# Offline analysis selection from the already formatted global test split
# ===========================================================================

def select_test(source: Path, env_name: str | None,
                code_id: str | None, out_dir: Path) -> Path:
    if not source.is_file():
        raise ValueError(f"global test parquet is missing: {source}; generate it with dataset/codegym.py")
    table = pq.read_table(source)
    if not table.num_rows:
        raise ValueError(f"global test parquet is empty: {source}")
    lites = []
    for index, row in enumerate(table.to_pylist()):
        extra = row.get("extra_info") or {}
        if extra.get("split") != "test":
            raise ValueError(f"expected only test rows in {source}; row {index} has split={extra.get('split')!r}")
        lites.append({"ability": row.get("ability"), "code_id": extra.get("code_id"), "index": index})
    selected = select_lites(lites, env_name, code_id)
    if not selected:
        raise ValueError(f"no test rows for ENV_NAME={env_name!r} CODE_ID={code_id!r} in {source}")
    print(f"[benchmark] CodeGym test source={source} "
          f"ENV_NAME={env_name or '<all>'} CODE_ID={code_id or '<all>'} "
          f"rows={len(selected)}/{table.num_rows} (before VAL_MAX_SAMPLES)", file=sys.stderr)
    if not env_name and not code_id:
        return source.resolve()
    subset = table.take([row["index"] for row in selected])
    out_dir.mkdir(parents=True, exist_ok=True)
    # Each launch owns an immutable selection, including concurrent runs with
    # different TEST_DATA overrides. Keep it alongside datasets, not experiments.
    fd, name = tempfile.mkstemp(prefix="shared-test-", suffix=".parquet", dir=out_dir)
    os.close(fd)
    destination = Path(name)
    try:
        pq.write_table(subset, destination)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination.resolve()


def select_test_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{Path(__file__).name} select-test",
        description=(
            "Select rows for offline analysis from an existing formatted global CodeGym test split. "
            "Never rebuild or re-split HF data: filtering before splitting would change which "
            "examples are held out. Preserve every selected row, its schema and original order."
        ),
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--env-name")
    parser.add_argument("--code-id")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        path = select_test(args.source, args.env_name, args.code_id, args.out_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"[benchmark] {exc}\n")
    print(path)


# ===========================================================================
# CLI
# ===========================================================================

SPLITS = {"train": 79106, "test": 1024}
REVISION = "85286359a342f7a288aea74273772b69b9b784c2"
SOURCE = {
    "dataset": "VanishD/CodeGym", "revision": REVISION, "status": "reproduction_snapshot",
    "split_mapping": {"train": "source train rows 0:79106", "test": "source train rows 79106:80130"},
    "environment_records": 22366,
    "limitations": ["VanishD/CodeGym is a reproduction publication, not authenticated paper-official data.",
                    "The published task source has only train; local test is the already selected final 1024 rows, not an official test split.",
                    "Complete cached snapshot shards and dataset card are preserved byte-for-byte; revision label is not independently authenticated."],
}


def source_rows(root: Path):
    """Join the immutable original split sequence to source rows without re-splitting."""
    import itertools
    expected = itertools.chain.from_iterable(
        ((split, index, row) for index, row in enumerate(snapshot.iter_rows(root / f"source/local_selection/{split}.parquet")))
        for split in SPLITS
    )
    for path in sorted((root / "source/snapshot" / _HF_TASK_SUBDIR).glob("*.parquet")):
        relative = path.relative_to(root).as_posix()
        digest = snapshot.sha256(path)
        # Heavy and unknown fields are fully retained in the unmodified shard.
        for source_index, raw in enumerate(snapshot.iter_rows(path, columns=[
                "ability", "answer", "code_id", "data_source", "prompt", "reward_model", "extra_info.solve_fc_round"])):
            try:
                split, index, previous = next(expected)
            except StopIteration as exc:
                raise ValueError("CodeGym source contains tasks beyond original selection") from exc
            if (previous["index"] != index or previous["ability"] != raw["ability"]
                    or previous["extra_info"]["code_id"] != raw["code_id"]
                    or public_messages(previous["prompt"]) != public_messages(raw["prompt"])):
                raise ValueError(f"CodeGym source/order identity mismatch: {split}[{index}]")
            lite = {**raw, "solve_fc_round": raw["extra_info"]["solve_fc_round"]}
            row = build_row(lite, index, split)
            if row["reward_model"] != previous["reward_model"] or row["extra_info"] != previous["extra_info"]:
                raise ValueError(f"CodeGym existing private/runtime data changed: {split}[{index}]")
            identity = f"codegym/{REVISION}/{path.name}/{source_index}"
            yield split, snapshot.mark_row(row, identity, {"path": relative, "sha256": digest,
                                           "row": source_index, "task_id": identity}, "train")
    if next(expected, None) is not None:
        raise ValueError("CodeGym source missing selected tasks")


def validate_environment_sources(root: Path) -> None:
    import hashlib
    expected = set()
    for path in sorted((root / "source/snapshot/envs_en").glob("*.parquet")):
        for row in snapshot.iter_rows(path):
            name = f"{row['source']}__{row['env_name']}.py"
            if name in expected:
                raise ValueError(f"Duplicate CodeGym environment: {name}")
            expected.add(name)
            code = root / "envs/codegym_v1" / name
            if not code.is_file() or snapshot.sha256(code) != hashlib.sha256(row["env_code"].encode()).hexdigest():
                raise ValueError(f"CodeGym environment source mismatch: {name}")
    actual = {p.name for p in (root / "envs/codegym_v1").glob("*.py")}
    if len(expected) != 22366 or actual != expected | {"__init__.py"}:
        raise ValueError("CodeGym environment set differs from the complete 22366-record snapshot")
    for split in SPLITS:
        for row in snapshot.iter_rows(root / f"source/local_selection/{split}.parquet", columns=["ability"]):
            name = row["ability"].split("@", 2)[1] + ".py"
            if name not in expected:
                raise ValueError(f"Task references missing environment: {name}")


def validate_source_dataset(root: Path) -> None:
    import itertools
    manifest = snapshot.validate_manifest(root, "codegym")
    if {k: v["rows"] for k, v in manifest["splits"].items()} != SPLITS:
        raise ValueError("CodeGym split membership contract changed")
    validate_environment_sources(root)
    for split, group in itertools.groupby(source_rows(root), key=lambda pair: pair[0]):
        snapshot.compare_rows(root, split, (row for _, row in group))


def rebuild_source_dataset(destination: Path, source_dir: Path, *, overwrite=False):
    import itertools
    def populate(stage, active):
        if not (stage / "source").exists():
            snapshot.preserve_selection(active, stage, SPLITS)
            paths = [source_dir / "README.md", *sorted((source_dir / _HF_TASK_SUBDIR).glob("*.parquet")),
                     *sorted((source_dir / "envs_en").glob("*.parquet"))]
            if len(paths) != 6:
                raise ValueError("Expected dataset card, four task shards and one environment shard")
            for path in paths:
                snapshot.copy_asset(path, stage, "source/snapshot/" + path.relative_to(source_dir).as_posix())
        validate_environment_sources(stage)
        for split, group in itertools.groupby(source_rows(stage), key=lambda pair: pair[0]):
            snapshot.write_rows(stage / f"{split}.parquet", (row for _, row in group))
        snapshot.make_manifest(stage, "codegym", SPLITS, SOURCE)
    return snapshot.rebuild(destination, overwrite=overwrite, populate=populate, validate=validate_source_dataset)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Only a leading command selects an auxiliary operation. Existing flag-only
    # invocations, including option values named "schema", keep their meaning.
    if argv and argv[0] == "schema":
        schema_main(argv[1:])
        return
    if argv and argv[0] == "select-test":
        select_test_main(argv[1:])
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None,
                    help="Complete local snapshot directory, or its task_en_instruction_en_env subdirectory; "
                         "overrides --snapshot-dir for initial source packaging (not a parquet file or glob)")
    ap.add_argument("--out", default=str(_DATA_ROOT / "codegym" / "dataset"), help="output dataset directory (default data/codegym/dataset)")
    ap.add_argument("--overwrite", action="store_true", help="Authorize transactional replacement with recovery snapshot")
    ap.add_argument("--check-only", "--validate-only", action="store_true", help="Validate packaged source and derived rows without rebuilding")
    ap.add_argument("--snapshot-dir", type=Path, default=Path.home() / f".cache/huggingface/hub/datasets--VanishD--CodeGym/snapshots/{REVISION}",
                    help="Complete local snapshot directory including task shards, envs_en and README.md")
    args = ap.parse_args(argv)
    destination = Path(args.out).expanduser().resolve()
    if args.check_only:
        snapshot.check_only(destination, validate_source_dataset)
        return
    source_dir = Path(args.src).expanduser() if args.src else args.snapshot_dir.expanduser()
    if source_dir.name == _HF_TASK_SUBDIR:
        source_dir = source_dir.parent
    rebuild_source_dataset(destination, source_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
