#!/usr/bin/env python3
"""Gate: what is on the wandb dashboard is what is on disk.

    .venvs/expa-verl/bin/python
    experiments/dyad_training/action_encoder_alignment/analysis/actenc_alignment_verify_wandb.py
    .venvs/expa-verl/bin/python
    experiments/dyad_training/action_encoder_alignment/analysis/actenc_alignment_verify_wandb.py --run
    qwen2.5-3b-instruct_20260831_145620

**Reads back from the wandb API, never from either publisher's own log.** Alignment writes to a run
from two processes -- `train.py` streams the curve while it trains, `actenc_alignment_upload_wandb.py`
adds the tables and summary afterwards -- and each of them printing a URL and a count proves it
ran, not that anything arrived: a dropped history point or a summary overwritten by a later `log`
leaves both logs looking perfect.

    W1  the run exists and reached state=finished
    W2  the server has as many history points as metrics.jsonl, on the same steps
    W3  every uploaded curve value matches the local one
    W4  the headline metrics match summary.json -- **not** the last step's.
        `train.py` restores the best selection-CE checkpoint before saving, so the shipped numbers
        and the final step's differ (1.9 points of top-1 on 3B). wandb refreshes the summary from
        each `log`, so an uploader that sets the summary before its last `log` silently publishes
        the wrong one. This check caught exactly that on 2026-08-31.
    W5  the trainable-parameter count matches, and is the projector's rather than a model's
    W7  counter-example: a run id that does not exist is reported as absent, not passed

W6 checked that a rendered figure was attached. Alignment no longer produces one -- wandb draws the
curve from the history, so W2 and W3 already assert what the figure used to stand for. The tag is
left unused rather than renumbered: W1-W5 and W7 appear in `lucia_job/RULES.md` and in run logs going
back, and shifting them would make those references point at different checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

from agent_system.policies.dyad.training.action_encoder_alignment import actenc_alignment_config as cfg  # noqa: E402
from agent_system.policies.dyad.training.action_encoder_alignment import (  # noqa: E402
    actenc_alignment_wandb_run as wandb_run,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

#: Where to look, taken from the shared module -- a gate that checks a different project than the
#: one written to passes vacuously. Only the *addressing* is shared. Reading the history back is
#: re-implemented below rather than imported, on purpose: this file's job is to disagree with the
#: publisher, and a gate that runs the publisher's own code cannot.
DEFAULT_PROJECT = wandb_run.DEFAULT_PROJECT

_failures: list[str] = []


def check(tag: str, ok: bool, detail: str) -> bool:
    print(f"[alignment-wandb] {tag} {detail} -> {PASS if ok else FAIL}")
    if not ok:
        _failures.append(f"{tag} {detail}")
    return ok


def local(run_dir: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line
            in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "rows": rows,
        "summary": json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(cfg.records_dir()))
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    args = parser.parse_args(argv)
    _failures.clear()

    try:
        import wandb
    except ImportError:
        print("wandb is not installed in this environment", file=sys.stderr)
        return 2

    runs_root = Path(args.runs_dir)
    names = args.run or sorted(p.name for p in runs_root.iterdir()
                               if (p / "summary.json").exists())
    if not names:
        print(f"[alignment-wandb] no runs under {runs_root}", file=sys.stderr)
        return 2

    api = wandb.Api()
    entity = api.default_entity
    print(f"--- {entity}/{args.project} ---")

    for name in names:
        disk = local(runs_root / name)
        summary = disk["summary"]
        protocol = wandb_run.metric_protocol(summary)
        split = wandb_run.selection_split(summary)
        namespace = "test" if protocol == "final-test" else "val"
        if any(wandb_run.metric_protocol(row) != protocol for row in disk["rows"]):
            raise ValueError("Alignment history and summary disagree on the selection split")
        run_id = wandb_run.run_id(name)
        try:
            run = api.run(f"{entity}/{args.project}/{run_id}")
        except Exception as exc:                                    # noqa: BLE001
            check("W1", False, f"[{name}] cannot fetch {run_id}: {type(exc).__name__}")
            continue

        check("W1", run.state == "finished", f"[{name}] state={run.state}")
        if summary.get("split_policy"):
            config = getattr(run, "config", {})
            check("W0", all(config.get(key) == summary[key]
                            for key in ("split_policy", "evaluation_role")),
                  f"[{name}] explicit split protocol and role match")

        # Fetched whole, then filtered here. `scan_history(keys=[...])` asks the server to project
        # the columns and fails on this project with "Step column '_step' not found in schema";
        # and the run carries more rows than the curve anyway -- the breakdown tables and the
        # figure are each their own `log`. Keeping only rows that have a selection metric is what makes
        # the count comparable with metrics.jsonl.
        all_history = list(run.scan_history())
        wrong_splits = ["val", "unseen"] if protocol == "final-test" else ["test"]
        if protocol == "train-validation":
            wrong_splits.append("unseen")
        forbidden = tuple(prefix for split_name in wrong_splits for prefix in
                          (f"{split_name}/", f"{split_name}_by_", f"breakdown/{split_name}_by_"))
        check("W0", not any(key.startswith(forbidden) for h in all_history + [dict(run.summary)]
                            for key in h), f"[{name}] no mixed evaluation namespaces")
        history = [h for h in all_history
                   if h.get(f"{namespace}/cross_entropy") is not None]
        server_steps = [h["step"] for h in history if h.get("step") is not None]
        local_steps = [r["step"] for r in disk["rows"]]
        check("W2", server_steps == local_steps,
              f"[{name}] {len(server_steps)} server-side against {len(local_steps)} on disk, "
              f"first four {server_steps[:4]} against {local_steps[:4]}")

        mismatched = []
        by_step = {h["step"]: h for h in history if h.get("step") is not None}
        for row in disk["rows"]:
            got = by_step.get(row["step"])
            if got is None:
                mismatched.append((row["step"], "absent"))
                continue
            # Independently assert the public namespace, not the publisher's mapper.
            for metric in ("cross_entropy", "top1_accuracy", "demonstrated_action_probability",
                           "chance_accuracy", "chance_cross_entropy"):
                key, value = f"{namespace}/{metric}", row[split][metric]
                if got.get(key) is None or abs(got[key] - value) > 1e-9:
                    mismatched.append((row["step"], key))
        check("W3", not mismatched,
              f"[{name}] {5 * len(disk['rows'])} curve values match"
              + (f"; {mismatched[:3]} do not" if mismatched else ""))

        # W4: against summary.json, which is the *shipped* checkpoint, not the final step.
        flat = {f"{namespace}/{metric}": value for metric, value in summary[split].items()}
        if protocol == wandb_run.LEGACY_VAL and "unseen" in summary:
            flat.update({f"unseen/{metric}": value for metric, value in summary["unseen"].items()})
        for group in wandb_run.breakdown_groups(summary):
            # Independent naming assertion: only the old local test groups were validation.
            prefix = group.replace("test_", "val_", 1) if protocol == wandb_run.LEGACY_TEST else group
            for name_, metrics in summary[group].items():
                safe = str(name_).replace(" ", "_")
                flat.update({f"{prefix}/{safe}/{metric}": value for metric, value in metrics.items()})
        wrong = {k: (run.summary.get(k), v) for k, v in flat.items()
                 if run.summary.get(k) is None or abs(run.summary[k] - v) > 1e-9}
        check("W4", not wrong, f"[{name}] {len(flat)} headline metrics match summary.json"
              + (f"; {', '.join(f'{k}: {a} != {b}' for k, (a, b) in wrong.items())}" if wrong else ""))

        if protocol == "final-test":
            check("W5", all(run.summary.get(key) == summary[key] for key in
                            ("checkpoint_identity", "dataset_identity")),
                  f"[{name}] evaluated checkpoint and dataset identities match")
        else:
            want = summary["trainable_parameters"]
            got = run.summary.get("trainable_parameters")
            check("W5", got == want, f"[{name}] trainable parameters {got} against {want}")

    # W7: without this, every check above passes vacuously if `api.run` started returning stubs.
    absent_reported = False
    try:
        api.run(f"{entity}/{args.project}/alignment-does-not-exist-{id(api)}")
    except Exception:                                               # noqa: BLE001
        absent_reported = True
    check("W7", absent_reported, "counter-example: a nonexistent run id raises rather than passing")

    print()
    if _failures:
        print(f"[alignment-wandb] FAILED {len(_failures)}:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"[alignment-wandb] {len(names)} run(s) on wandb match the local artefacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
