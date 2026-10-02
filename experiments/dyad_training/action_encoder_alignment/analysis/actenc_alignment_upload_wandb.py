#!/usr/bin/env python3
"""Finish a Alignment run on Weights & Biases: the breakdown tables and the summary.

`agent_system/policies/dyad/training/action_encoder_alignment/actenc_alignment_train.py` streams the curve while it
trains, so by the time this script runs the
history is usually already on the server and there is nothing to replay. What is never there is
what does not exist until training is over: the per-slice tables, and the headline summary --
which has to be published by whichever process writes to the run *last*, because wandb refreshes
the summary from every `log`.

There is no figure. wandb draws the curve from the history it already has, interactively and in
both themes; a second, static rendering of the same numbers was one more thing to generate,
attach, and check, and it could disagree with the panel beside it.

So this reconciles first, then tops up:

    run absent on the server      replay metrics.jsonl in full, then top up
    history matches the disk      top up only
    history does not match        exit non-zero and say which steps differ

The last case does not repair itself. Republishing rewrites a run someone may be reading, and a
gap in a streamed history is worth a human look -- it usually means the training process was
killed, or wandb's background thread was. `--rebuild` is the explicit repair: reopen the run from
empty and replay `metrics.jsonl` in full. That is what keeps a streamed run re-publishable after a
network failure, which is the property the older replay-only design was protecting.

Nothing here deletes a run. wandb retires the id along with it, and the id is derived from the run
directory -- so a delete does not free the name, it burns it. See
`agent_system/policies/dyad/training/action_encoder_alignment/actenc_alignment_wandb_run.py`.

**`WANDB_MODE` is set here, not read from the environment.** An `offline` left over from an
earlier debug session makes this upload nothing at all, with exit code 0, no error in the log, and
no curve on the dashboard. The reverse -- an unwanted online run -- is merely noise. So the mode
is forced and announced.

Everything the dashboard's shape depends on -- project, run id, series names, init arguments --
lives in `agent_system/policies/dyad/training/action_encoder_alignment/actenc_alignment_wandb_run.py`, shared with the
trainer. Nothing here decides it twice.

Usage:
    python experiments/dyad_training/action_encoder_alignment/analysis/actenc_alignment_upload_wandb.py
    python experiments/dyad_training/action_encoder_alignment/analysis/actenc_alignment_upload_wandb.py --run base
    --project my-project
    python experiments/dyad_training/action_encoder_alignment/analysis/actenc_alignment_upload_wandb.py --run base
    --rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Four levels up: experiments/dyad_training/action_encoder_alignment/analysis/<this file>. Counting directories is exactly the
# thing that broke when this script moved one level deeper, so everything below asks
# `agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_config` for a path instead of
# rebuilding one -- that module is the single source of
# truth for where Alignment keeps its dataset and its runs.
PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

from agent_system.policies.dyad.training.action_encoder_alignment import (  # noqa: E402
    actenc_alignment_config as alignment_config,
)
from agent_system.policies.dyad.training.action_encoder_alignment import (  # noqa: E402
    actenc_alignment_wandb_run as wandb_run,
)

DEFAULT_PROJECT = wandb_run.DEFAULT_PROJECT

#: How long to keep asking before calling the server's history "short". `handle.finish()` in the
#: trainer blocks until the data is sent, so the server has it by the time this script starts --
#: but "sent" and "visible to the query API" are not guaranteed to be the same instant, and a false
#: mismatch costs a red job plus a needless full rebuild. Two extra questions are cheaper.
RECONCILE_ATTEMPTS = 3
RECONCILE_PAUSE_SECONDS = 5.0


def log(message: str) -> None:
    print(f"[wandb-upload] {message}", flush=True)


def read_run(run_dir: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in
            (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "name": run_dir.name,
        "dir": run_dir,
        "rows": rows,
        "summary": json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
        "config": yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")),
    }


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Publish explicit val/test unchanged; only historical test is mapped to validation."""
    wandb_run.selection_split(summary)
    out: dict[str, Any] = {}
    for key in ("steps", "selected_step", "wall_seconds", "trainable_parameters",
                "gate", "projector_l2_displacement", "train_loss", "split_policy", "evaluation_role",
                "dataset_identity", "checkpoint_identity"):
        if key in summary:
            out[key] = summary[key]
    for split in ("test", "val", "unseen"):
        for metric, value in (summary.get(split) or {}).items():
            out[f"{wandb_run.published_name(split, summary)}/{metric}"] = value
    for group in wandb_run.breakdown_groups(summary):
        for slice_name, metrics in (summary.get(group) or {}).items():
            safe = str(slice_name).replace(" ", "_")
            prefix = wandb_run.published_name(group, summary)
            for metric, value in metrics.items():
                out[f"{prefix}/{safe}/{metric}"] = value
    return out


def breakdown_table(wandb, summary: dict[str, Any], group: str):
    rows = summary.get(group) or {}
    columns = ["n", "top1_accuracy", "cross_entropy", "demonstrated_action_probability",
               "chance_accuracy", "chance_cross_entropy", "mean_candidates"]
    table = wandb.Table(columns=["slice", *columns])
    legacy = wandb_run.metric_protocol(summary) in (wandb_run.LEGACY_TEST, wandb_run.LEGACY_VAL)
    for slice_name, metrics in rows.items():
        # Old reports did not record every baseline/probability column. Leave absent
        # values null, never infer a baseline from mean candidate count.
        values = [metrics.get(key) if legacy else metrics[key] for key in columns]
        table.add_data(str(slice_name), *values)
    return table


class HistoryMismatch(Exception):
    """The server's curve is not the one on disk, and repairing it needs `--rebuild`."""


def reconcile(name: str, local_steps: list[int], project: str, *,
              record: dict[str, Any] | None = None) -> bool:
    """Has the trainer already streamed this run's curve? Raises `HistoryMismatch` if the server
    holds a different one.

    Returns False when the server has no such run at all. That is the pre-streaming path, and
    still the path taken by `--wandb off` runs and by anything trained on a machine with no
    credential -- so the full replay below is not dead code kept for history's sake.
    """
    for attempt in range(1, RECONCILE_ATTEMPTS + 1):
        run = wandb_run.find_run(name, project)
        if run is None:
            log(f"{wandb_run.run_id(name)} is not on the server; replaying all "
                f"{len(local_steps)} points")
            return False
        namespace = "val" if record is None else wandb_run.published_name(
            wandb_run.selection_split(record), record)
        if record is not None:
            config = getattr(run, "config", {})
            if any(config.get(key) != record.get(key) for key in ("split_policy", "evaluation_role")):
                raise HistoryMismatch("Server and local split semantics differ; use a distinct run name")
            if record.get("evaluation_role") == "final-test" and any(
                    config.get(key) != record[key] for key in ("checkpoint_identity", "dataset_identity")):
                raise HistoryMismatch("Server and local evaluated identities differ; use a distinct run name")
            wrong_splits = ["val", "unseen"] if namespace == "test" else ["test"]
            if record.get("evaluation_role") == "train-validation":
                wrong_splits.append("unseen")
            forbidden = tuple(prefix for split in wrong_splits
                              for prefix in (f"{split}/", f"{split}_by_", f"breakdown/{split}_by_"))
            if any(key.startswith(forbidden) for row in run.scan_history() for key in row):
                raise HistoryMismatch("Server contains incompatible split metric names")
        server_steps = wandb_run.history_steps(run, namespace)
        if server_steps == local_steps:
            log(f"{len(server_steps)} points already streamed by the trainer; nothing to replay")
            return True
        if attempt < RECONCILE_ATTEMPTS:
            log(f"server has {len(server_steps)} points against {len(local_steps)} on disk; "
                f"asking again in {RECONCILE_PAUSE_SECONDS:.0f}s "
                f"({attempt}/{RECONCILE_ATTEMPTS})")
            time.sleep(RECONCILE_PAUSE_SECONDS)
            continue

        missing = [s for s in local_steps if s not in set(server_steps)]
        extra = [s for s in server_steps if s not in set(local_steps)]
        raise HistoryMismatch(
            f"{wandb_run.run_id(name)} holds {len(server_steps)} points, metrics.jsonl has "
            f"{len(local_steps)}. Missing on the server: {missing[:6]}"
            f"{'...' if len(missing) > 6 else ''}; not on disk: {extra[:6]}"
            f"{'...' if len(extra) > 6 else ''}"
        )
    raise AssertionError("unreachable")


def upload(run: dict[str, Any], project: str, *, rebuild: bool) -> str:
    import wandb

    name = run["name"]
    summary = run["summary"]
    protocol = wandb_run.metric_protocol(summary)
    if any(wandb_run.metric_protocol(row) != protocol for row in run["rows"]):
        raise ValueError("Alignment history and summary disagree on the selection split")
    if summary.get("split_policy") and any(
            run["config"].get(key) != summary[key] for key in ("split_policy", "evaluation_role")):
        raise ValueError("Alignment config and metrics disagree on split semantics")
    local_steps = [row["step"] for row in run["rows"]]

    if rebuild:
        # Republish in place. Not a delete: wandb retires the id along with the run and no later
        # init could use the name again -- see the note above `find_run` in
        # agent_system/policies/dyad/training/action_encoder_alignment/actenc_alignment_wandb_run.py. Replaying onto a
        # run that still holds history does not work
        # either, because wandb requires `step` to be non-decreasing and would drop every replayed
        # point. So this is only possible while the server side is empty, and says so when it is
        # not rather than half-writing a curve.
        existing = wandb_run.find_run(name, project)
        held = -1 if existing is None else existing.lastHistoryStep
        if held >= 0:
            raise SystemExit(
                f"--rebuild cannot republish {wandb_run.run_id(name)}: the server still holds "
                f"{held + 1} history point(s), and wandb will not accept a step it has already "
                f"passed.\n"
                f"  The id cannot be freed either -- deleting the run retires the name.\n"
                f"  Publish under a new id instead: rename {run['dir']} and re-run. The run id "
                f"is derived from the directory name."
            )
        log(f"--rebuild: server side is empty, replaying all {len(local_steps)} points")
        resume, replay = "allow", True
    else:
        replay = not reconcile(name, local_steps, project, record=summary)
        resume = "allow"

    try:
        handle = wandb.init(**wandb_run.init_kwargs(
            name, wandb_run.run_config(run["config"]), project, resume=resume))
    except Exception as exc:                                        # noqa: BLE001
        raise SystemExit(
            f"cannot open {wandb_run.run_id(name)}: {type(exc).__name__}: {exc}\n"
            f"  If the id was retired by an earlier delete, wandb will not take it again. "
            f"Rename {run['dir']} and publish that instead -- the id comes from the "
            f"directory name."
        ) from exc
    wandb_run.define_metrics(handle, protocol)

    if replay:
        for row in run["rows"]:
            handle.log(wandb_run.history_payload(row), step=row["step"])

    # No `step=` on the tables. The SDK can merge them into a pending final history row
    # or commit a separate row after resuming. They add no val CE, so neither case adds
    # an evaluation point to the verifier's curve count.
    groups = wandb_run.breakdown_groups(summary)
    handle.log({f"breakdown/{wandb_run.published_name(group, summary)}":
                breakdown_table(wandb, run["summary"], group) for group in groups})

    # **Last, after every `log`.** Each `log` refreshes the summary from what it just wrote, so a
    # `summary.update` placed before the tables put the dashboard's headline numbers back to the
    # final step's -- and the final step is not what Alignment ships. `train.py` restores the
    # best-held-out-CE checkpoint before saving, so on 3B the shipped val top-1 is 0.9810 (step
    # 800) while step 1896 reads 0.9619. Uploading the latter means the dashboard disagrees with
    # `summary.json` and with the checkpoint on disk, by 1.9 points, with nothing saying so.
    # `actenc_alignment_verify_wandb.py` W4 is the assertion; it caught exactly this on 2026-08-31.
    #
    # It is also why the trainer publishes no summary of its own: it finishes at the final step,
    # and the dashboard believes whichever process wrote last.
    handle.summary.update(flatten_summary(run["summary"]))
    url = handle.url
    handle.finish()
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(alignment_config.records_dir()))
    parser.add_argument("--run", action="append", help="run name; repeatable. Default: all of them")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--rebuild", action="store_true",
                        help="reopen the run from empty and replay metrics.jsonl in full. "
                             "The repair for a streamed history with a gap in it")
    args = parser.parse_args()

    # Forced, never inherited. See the module docstring.
    previous = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "online"
    if previous and previous != "online":
        log(f"WANDB_MODE was {previous!r} in the environment; forcing 'online'")

    runs_root = Path(args.runs_dir)
    names = args.run or sorted(d.name for d in runs_root.iterdir()
                               if (d / "metrics.jsonl").exists())
    if not names:
        print(f"no runs with a metrics.jsonl under {runs_root}", file=sys.stderr)
        return 1

    urls = {}
    mismatched: list[str] = []
    for name in names:
        run = read_run(runs_root / name)
        log(f"publishing {name}: {len(run['rows'])} points on disk")
        try:
            urls[name] = upload(run, args.project, rebuild=args.rebuild)
        except HistoryMismatch as exc:
            # Recorded and carried on, rather than raised: with several runs named, one bad one
            # should not hide whether the others published. The non-zero exit below is what fails
            # the job.
            log(f"MISMATCH {exc}")
            mismatched.append(name)
            continue
        log(f"  -> {urls[name]}")

    print()
    for name, url in urls.items():
        print(f"{name}: {url}")
    if mismatched:
        print(f"\n[wandb-upload] {len(mismatched)} run(s) do not match the server. "
              f"Re-publish each with --rebuild:", file=sys.stderr)
        for name in mismatched:
            print(f"  python {Path(__file__).name} --run {name} --rebuild", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
