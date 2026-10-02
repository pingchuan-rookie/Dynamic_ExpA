#!/usr/bin/env python3
# Copyright 2025 ExpA_verl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""#2 GRPO tool-use check (react external environment) -- tool hit / external env recognition / returned content / return timeout / single-step reward.

Applies to gsm8k-grpo_react (text `<Action>48/2=</Action>`): goes through tool_agent, writes `grpo_tool_agent_*.jsonl`.
Also reads ALFWorld, CodeGym and WebShop; reward is the raw environment outcome.
Pool lifecycle events use `<env>_env_pool_*.jsonl`.
WebShop permits concurrent sessions on one catalog backend, bounded by sessions_per_backend.

Reads the env_step / trajectory_summary dumped with DYAD_DIAG_ENABLED=1 and checks (each conclusion = invariant + criterion + data):

  U · TOOL USE / recognition -- the tool really was called, the external env recognises the action:
      - U1 env_step exists (otherwise the model produced no parseable tool_call -> 0 tool executions ->
        reward constantly 0, the most common broken link in react).
      - U2 every step has action_sent (the model emitted an action string).
      - U3 external env recognition rate = 1 − invalid_action share (invalid_action=True means calc/env could not parse that action).
  C · returned content -- every (non-timeout / non-failed) env_step recorded the tool observation it returned (new diagnostic field).
  TO · return timeout / latency -- timed_out count + execution_time_ms distribution
      (the soft timeout can only trigger when training ran with GRPO_TOOL_TIMEOUT_MS>0; otherwise only latency is reported and timed_out is constantly 0).
  R · REWARD -- single-step reward + trajectory aggregation:
      - R1 infra/tool_failure/timeout steps record reward 0 (no penalty).
      - R2 trajectory.sum_turn_scores == Σ env_step.reward of that trajectory.

  ⚠️ LOSS/MASK/ADVANTAGE: standard GRPO dp_actor.update_policy has no log_event, so this script does not check them.

Usage (cwd=Dynamic_ExpA):
  python experiments/shared/analysis/run/verify_grpo_react.py [debug_dir]
  Defaults to the newest run under outputs/<algo>/<env>/<run>/ that holds diagnostic jsonl.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

from _mask_checks import unretained_observation_requests
from _env_checks import action_evidence, is_probe, lease_violations

OK = "\033[92mPASS\033[0m"
BAD = "\033[91mFAIL\033[0m"
NA = "\033[93mN/A (not recorded)\033[0m"

def _resolve_dir(argv: list[str]) -> str:
    """Newest run under outputs/<algo>/<env>/<run>/; single implementation in _events.resolve_dir."""
    from decode_trajectories import resolve_dir

    return resolve_dir(argv[1] if len(argv) > 1 else None)


def _read_events(pattern: str, event: str) -> list[dict]:
    out: list[dict] = []
    for fn in sorted(glob.glob(pattern)):
        # Keep event provenance: worker_idx is unique only within one process's environment pool.
        source = os.path.basename(fn)
        with open(fn, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == event:
                    rec["_source"] = source
                    out.append(rec)
    return out


def _events(debug_dir: str, event: str, prefix: str = "grpo_tool_agent") -> list[dict]:
    return _read_events(f"{debug_dir}/{prefix}*.jsonl", event)


def _pool_events(debug_dir: str, event: str) -> list[dict]:
    """Read external env pool lifecycle events. The pool diagnostic files are named per env
    (calc_env_pool / alfworld_env_pool / codegym_env_pool) and share one schema; a single run
    only ever has one env kind, so matching all `*_env_pool*.jsonl` covers gsm8k-calc / alfworld / codegym."""
    return _read_events(f"{debug_dir}/*_env_pool*.jsonl", event)


# The env preflight (experiments/shared/train_eval/scripts/prepare.py) opens one session
# per env named "<env>-probe" before training starts, and it writes into the same DYAD_DIAG_DIR as
# the run. That session is a health check, not a trajectory: counting it makes "one reset per
# trajectory" off by exactly one (65 vs 64) and "one bind per env step" off by two (127 vs 125) --
# on runs where nothing is wrong. E1/RE-E1 already assert the probe ran and succeeded, which is
# where that evidence belongs.
PROBE_SUFFIX = "-probe"


def _is_probe(session_id) -> bool:
    return is_probe(session_id)


def _p(cond: bool, tag: str, msg: str) -> bool:
    print(f"  [{OK if cond else BAD}] {tag}: {msg}")
    return cond


def _is_failed(s: dict) -> bool:
    """infra / tool exception / timeout: these steps record reward 0 without penalty and are not required to carry a normal observation."""
    return bool(s.get("http_failure") or s.get("tool_failure") or s.get("timed_out"))


def verify(debug_dir: str) -> bool:
    print("=" * 72)
    print(f"GRPO tool-use (react) check: {debug_dir}")
    print("=" * 72)
    from _env_checks import step_run_data, verify_environment_steps

    data = step_run_data(debug_dir)
    if data["shared"]:
        return verify_environment_steps(debug_dir, data)
    ok = True

    steps = _events(debug_dir, "env_step")
    trajs = _events(debug_dir, "trajectory_summary")
    pool_inits = _pool_events(debug_dir, "pool_init")
    resets = _pool_events(debug_dir, "env_reset")
    binds = _pool_events(debug_dir, "env_step_bind")
    closes = _pool_events(debug_dir, "env_close")
    aborts = _pool_events(debug_dir, "env_abort")

    # ---------------- TOOL USE / recognition ----------------
    print("\n# TOOL USE / recognition (was the tool called + does the external env recognise the action)")
    ok &= _p(bool(steps), "U1 tool was called (env_step present)",
             f"env_step={len(steps)} (=0 means no tool_call was parsed / the tool never ran, reward will be constantly 0)")
    if not steps:
        print("  -> Investigate: does the data prompt describe the <Action> protocol; does multi_turn.format=react match the tool_parser;")
        print("         are agent_loop tools None; does the action match this benchmark's tool schema.")
        return False

    by_req_steps = collections.Counter(s.get("request_id") for s in steps)
    print(f"  trajectories={len(by_req_steps)}  total env_step={len(steps)}  average tool calls per trajectory={len(steps)/max(1,len(by_req_steps)):.1f} times")

    with_action, invalid_missing = action_evidence(steps)
    ok &= _p(with_action == len(steps), "U2 action evidence recorded for every attempt",
             f"with action {with_action}/{len(steps)}; rejected attempts missing evidence={invalid_missing} "
             "(diagnostic gap, not proof the model emitted no action)")

    invalid = sum(1 for s in steps if s.get("invalid_action"))
    recog = len(steps) - invalid
    ok &= _p(recog > 0, "U3 external env recognises the action",
             f"recognised {recog}/{len(steps)} (invalid_action={invalid}, recognition rate {100 * recog / len(steps):.1f}%)")
    if invalid:
        print("  invalid_action>0: attempts were rejected by the selected tool/environment. "
              "For WebShop use search[query] or click[target] with a currently visible target; "
              "for other benchmarks inspect their action schema. Invalid attempts are policy outcomes, "
              "not automatically infrastructure failures.")

    # ---------------- ENV SESSION LIFECYCLE ----------------
    print("\n# ENV SESSION LIFECYCLE (trajectory <-> session <-> worker)")
    ok &= _p(bool(pool_inits) and all(e.get("ok") for e in pool_inits), "E1 pool startup and health probe",
             f"pool_init={len(pool_inits)}, failed={sum(1 for e in pool_inits if not e.get('ok'))}")

    request_ids = {s.get("request_id") for s in steps if not _is_probe(s.get("request_id"))}
    reset_ids = {e.get("session_id") for e in resets if not _is_probe(e.get("session_id"))}
    close_ids = {e.get("session_id") for e in closes if not _is_probe(e.get("session_id"))}
    ok &= _p(request_ids == reset_ids, "E2 trajectory request_id == env session_id",
             f"tool trajectories={len(request_ids)} reset sessions={len(reset_ids)} difference={len(request_ids ^ reset_ids)}")

    reset_worker = {e.get("session_id"): e.get("worker_idx") for e in resets}
    drift = [e for e in binds if reset_worker.get(e.get("session_id")) != e.get("worker_idx")]
    ok &= _p(bool(binds) and not drift, "E3 one trajectory stays bound to one worker",
             f"worker drift={len(drift)}/{len(binds)} env step")

    close_counts = collections.Counter(e.get("session_id") for e in closes)
    close_bad = [sid for sid in reset_ids if close_counts[sid] != 1]
    close_failed = [e for e in closes if not e.get("ok")]
    ok &= _p(bool(reset_ids) and reset_ids == close_ids and not close_bad and not close_failed,
             "E4 every session is closed exactly once and successfully",
             f"reset={len(reset_ids)} close={len(close_ids)} missing/duplicate={len(close_bad)} close failed={len(close_failed)}")

    # Identify workers by (source pool, worker_idx) when merging events from multiple processes.
    reuse_before_close, active_sessions = lease_violations(pool_inits, resets, closes)

    ok &= _p(bool(resets) and reuse_before_close == 0 and active_sessions == 0, "E5 session leases respect capacity and close",
             f"early reuse={reuse_before_close} still active at end of run={active_sessions} "
             f"abort={len(aborts)} pools={len({e.get('_source') for e in resets})}")

    if resets:
        peak = max(int(e.get("peak_active", 0) or 0) for e in resets)
        max_wait = max(float(e.get("waited_ms", 0) or 0) for e in resets)
        pool_size = max(int(e.get("pool_size", 0) or 0) for e in resets)
        print(f"  concurrency stats: pool_size={pool_size} peak_active={peak} max_wait={max_wait:.3f}ms")

    # ---------------- returned content ----------------
    print("\n# returned content (was the tool observation recorded)")
    has_obs_field = any("observation" in s for s in steps)
    if has_obs_field:
        need = [s for s in steps if not _is_failed(s)]
        with_obs = sum(1 for s in need if s.get("observation") not in (None, ""))
        ok &= _p(with_obs == len(need), "C1 returned content recorded (observation)",
                 f"{with_obs}/{len(need)} non-failed steps recorded the content the tool returned")
        sample = next((s.get("observation") for s in need if s.get("observation")), None)
        if sample:
            print(f"  sample observation: {str(sample)[:140]!r}")
    else:
        print(f"  [{NA}] C1 observation field not recorded (old diagnostic data; only recorded when debug is re-run with the current tool_agent_loop).")

    # ---------------- return timeout / latency ----------------
    print("\n# return timeout / latency")
    times = [float(s["execution_time_ms"]) for s in steps if s.get("execution_time_ms") is not None]
    timed_out = sum(1 for s in steps if s.get("timed_out"))
    tool_fail = sum(1 for s in steps if s.get("tool_failure"))
    ok &= _p(timed_out == 0, "TO1 no tool return timeout",
             f"timed_out={timed_out}/{len(steps)} (the soft timeout only triggers when training ran with GRPO_TOOL_TIMEOUT_MS>0)")
    if times:
        print(f"  tool execution latency: mean={sum(times) / len(times):.1f}ms max={max(times):.1f}ms "
              f"min={min(times):.1f}ms ({len(times)}/{len(steps)} steps timed) tool_failure={tool_fail}")
    else:
        print(f"  [{NA}] execution_time_ms not recorded (old diagnostic data).")

    # ---------------- REWARD ----------------
    print("\n# REWARD (single step + trajectory aggregation)")
    r1_bad = sum(1 for s in steps if _is_failed(s) and abs(float(s.get("reward", 0) or 0)) > 1e-9)
    ok &= _p(r1_bad == 0, "R1 failed steps get reward 0 (no penalty)", f"violations {r1_bad}/{len(steps)}")

    by_req = collections.defaultdict(float)
    for s in steps:
        by_req[s.get("request_id")] += float(s.get("reward", 0) or 0)
    agg_bad = [t.get("request_id") for t in trajs
               if t.get("request_id") in by_req
               and abs(float(t.get("sum_turn_scores", 0) or 0) - by_req[t["request_id"]]) > 1e-6]
    ok &= _p(not agg_bad, "R2 trajectory sum_turn_scores==Σ env_step reward", f"violations {len(agg_bad)}/{len(trajs)}")
    won = sum(1 for t in trajs if t.get("won") or t.get("env_won"))
    nonzero = sum(1 for s in steps if float(s.get("reward", 0) or 0))
    print(f"  trajectories={len(trajs)} won(correct)={won}; env_step steps with non-zero reward={nonzero}/{len(steps)}")
    if trajs and won == 0 and nonzero == 0:
        print("  Note: all-zero reward = no in-group variance -> GRPO advantage constantly 0 (normal at cold start / on hard tasks, not a bug).")

    # ---------------- MASK (are the tool observation tokens correctly masked out of the loss) ----------------
    print("\n# MASK (are the tool observation tokens correctly masked out of the loss)")
    masked = [t for t in trajs if t.get("num_masked_tokens") is not None]
    if masked:
        req_with_steps = {s.get("request_id") for s in steps}
        no_llm = [t for t in masked if not (t.get("num_response_tokens") or 0)]
        tool_trajs = [t for t in masked
                      if (t.get("num_turn_scores") or 0) > 0 or t.get("request_id") in req_with_steps]
        unretained = unretained_observation_requests(tool_trajs, _events(debug_dir, "generation_result"))
        unmasked_tool = [t for t in tool_trajs if not (t.get("num_masked_tokens") or 0)
                         and t.get("request_id") not in unretained]
        ok &= _p(not no_llm, "M1 every trajectory has LLM generated tokens (mask=1)",
                 f"trajectories with num_response_tokens=0 {len(no_llm)}/{len(masked)}")
        ok &= _p(not unmasked_tool, "M2 retained tool observations are masked out of the loss (mask=0)",
                 f"unexplained zero-mask tool trajectories={len(unmasked_tool)}/{len(tool_trajs)}; "
                 f"observations not retained (generation/mask counts agree)={len(unretained)}")
        tot_llm = sum(int(t.get("num_response_tokens") or 0) for t in masked)
        tot_mask = sum(int(t.get("num_masked_tokens") or 0) for t in masked)
        print(f"  stats: {len(masked)}/{len(trajs)} recorded mask counts; LLM tokens total={tot_llm}, "
              f"masked (tool observation+padding) total={tot_mask}")
        print("  Note: response_mask is per token (LLM=1 / tool observation+padding=0) and stored as a comma string; token-level visual alignment is in decode_trajectories.py.")
    else:
        print(f"  [{NA}] mask: no response_mask dumped (check that tool_agent_loop trajectory_summary carries response_mask + DYAD_DIAG_ENABLED=1).")

    # ---------------- LOSS (not recorded) ----------------
    print("\n# LOSS")
    print(f"  [{NA}] loss: GRPO dp_actor has no log_event, the tool-use pg_loss/ratio is not dumped.")

    print("\n" + "=" * 72)
    print(f"Overall verdict (recorded dimensions): {'all PASS ✅' if ok else 'FAIL present ❌'}  |  loss/advantage=not recorded (mask dumped)")
    print("=" * 72)
    return ok


if __name__ == "__main__":
    from decode_trajectories import save_report

    _dir = _resolve_dir(sys.argv)
    with save_report("verify_grpo_react", _dir):
        _ok = verify(_dir)
    sys.exit(0 if _ok else 1)
