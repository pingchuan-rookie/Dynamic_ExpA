#!/usr/bin/env python3
"""Read an Dyad training run's diagnostics and answer the three questions asked of it.

Summarizes recorded loss and parameter diagnostics for one run. The verifiers
check recorded invariants separately; this summary is not a correctness verdict.

  1. loss    pg_loss / kl_loss / entropy / grad_norm per step and forward/backward counts.
  2. projector   the projector's gradient norm and L2 across optimizer steps.
  3. action head   signatures of DirectActionHead weights rebuilt from synchronized projector
             parameters and encoder features. A policy gradient alone does not imply that this
             head changes; signatures are observations, not proof of synchronization correctness.

Usage:  python experiments/shared/analysis/run/summarize_training.py <run_dir> [<trainer_log>]
"""

from __future__ import annotations

import collections
import glob
import json
import re
import argparse
from pathlib import Path


def events(run: Path, component: str, event: str):
    out = []
    for p in sorted(glob.glob(str(run / f"{component}_pid*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if d.get("event") == event:
                d["_pid"] = p
                out.append(d)
    return out


def scalar(value):
    """Diagnostics writes a tensor as a dict with `mean` and a Python scalar as itself."""
    if isinstance(value, dict):
        value = value.get("mean")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def head_signature(ev):
    w = (((ev.get("action_head") or {}).get("parameters") or {}).get("weight") or {}).get("value") or {}
    return w.get("mean"), w.get("min"), w.get("max")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize loss, projector gradients and action-head diagnostics")
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('trainer_log', nargs='?', type=Path)
    args = parser.parse_args()
    run = args.run_dir
    if not run.is_dir():
        parser.error(f'Run directory does not exist: {run}')
    log = args.trainer_log or next((run / name for name in ('trainer.log', 'dyad.log', 'main_dyad.log')
                                   if (run / name).is_file()), None)

    # ---------------------------------------------------------------- 1. loss
    print("=" * 78)
    print("1. loss")
    print("=" * 78)
    if log and log.exists():
        text = log.read_text(encoding="utf-8", errors="ignore")
        keys = ("actor/pg_loss", "actor/kl_loss", "actor/entropy", "actor/grad_norm",
                "critic/score/mean", "actor/ppo_kl", "actor/pg_clipfrac",
                "actor/dyad/action_positions", "actor/dyad/vocab_positions")
        rows = []
        for line in text.splitlines():
            if "training/global_step" not in line:
                continue
            got = {}
            for k in keys:
                m = re.search(rf"{re.escape(k)}:(?:np\.float64\()?(-?[\d.e+-]+)", line)
                if m:
                    got[k] = float(m.group(1))
            m = re.search(r"training/global_step:(\d+)", line)
            got["step"] = int(m.group(1)) if m else len(rows) + 1
            rows.append(got)
        print(f"{'step':>4} {'score':>7} {'pg_loss':>11} {'kl_loss':>9} {'entropy':>8} "
              f"{'grad_norm':>10} {'ppo_kl':>8} {'clipfrac':>9} {'act/vocab pos':>15}")
        for r in rows:
            print(f"{r['step']:>4} {r.get('critic/score/mean', float('nan')):>7.4f} "
                  f"{r.get('actor/pg_loss', float('nan')):>11.6f} "
                  f"{r.get('actor/kl_loss', float('nan')):>9.5f} "
                  f"{r.get('actor/entropy', float('nan')):>8.5f} "
                  f"{r.get('actor/grad_norm', float('nan')):>10.6f} "
                  f"{r.get('actor/ppo_kl', float('nan')):>8.5f} "
                  f"{r.get('actor/pg_clipfrac', float('nan')):>9.5f} "
                  f"{r.get('actor/dyad/action_positions', 0):>7.0f}/"
                  f"{r.get('actor/dyad/vocab_positions', 0):<7.0f}")
    fwd = len(events(run, "dyad_actor", "policy_micro_batch_forward"))
    bwd = len(events(run, "dyad_actor", "policy_micro_batch_backward"))
    print(f"\n  forward micro-batches = {fwd}, backward = {bwd} "
          f"-> {'MATCH' if fwd == bwd else 'MISMATCH (a micro-batch was skipped)'}")

    # ---------------------------------------------------------------- 2. projector
    print()
    print("=" * 78)
    print("2. projector")
    print("=" * 78)
    steps = events(run, "dyad_actor", "optimizer_step")
    by_pid = collections.defaultdict(list)
    for e in steps:
        by_pid[e["_pid"]].append(e)
    one = sorted(by_pid.items())[0][1] if by_pid else []
    print(f"{'step':>4} {'grad_norm':>12} {'projector grad':>15} {'projector L2':>22}")
    l2_values = []
    for i, e in enumerate(one, 1):
        values = [scalar(e.get(key)) for key in
                  ("grad_norm", "dyad/adapter_grad_norm", "dyad/adapter_l2")]
        l2_values.append(values[-1])
        fields = [f"{value:>{width}.8f}" if value is not None else f"{'n/a':>{width}}"
                  for value, width in zip(values, (12, 15, 22))]
        print(f"{i:>4} " + " ".join(fields))
    if l2_values and l2_values[0] and l2_values[-1] is not None:
        moved = abs(l2_values[-1] - l2_values[0]) / l2_values[0]
        print(f"\n  projector L2 relative movement over {len(one)} steps: {moved:.3e}")

    # ---------------------------------------------------------------- 3. action head
    print()
    print("=" * 78)
    print("3. action head re-derived after every weight sync")
    print("=" * 78)
    inits = events(run, "dyad_vllm_model_runner", "action_head_initialized")
    per_rank = collections.defaultdict(list)
    for e in inits:
        per_rank[e["_pid"]].append(e)
    checked = changed = unknown = 0
    for pid, group in sorted(per_rank.items()):
        group.sort(key=lambda e: e.get("time", 0))
        print(f"  {Path(pid).name}: {len(group)} init(s)")
        for i, e in enumerate(group, 1):
            print(f"     init {i}: mean={head_signature(e)[0]!r}")
        for a, b in zip(group, group[1:]):
            checked += 1
            before, after = head_signature(a), head_signature(b)
            if any(value is None for value in before + after):
                unknown += 1
            elif before != after:
                changed += 1
    print(f"\n  adjacent pairs={checked}  changed={changed}  "
          f"unchanged={checked - changed - unknown}  missing-signature={unknown}")
    print("  signatures alone cannot verify synchronization; an unchanged head can be correct "
          "when only the policy changes or when updates leave these summary statistics unchanged.")

    fallbacks = len(events(run, "dyad_tool_agent", "missing_action_content_fallback"))
    trajs = len(events(run, "dyad_tool_agent", "trajectory_summary"))
    print(f"\n  (for context: {trajs} trajectories, {fallbacks} missing_action_content fallbacks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
