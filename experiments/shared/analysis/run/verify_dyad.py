#!/usr/bin/env python3
# Copyright 2025 ExpA_verl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""#3 Dyad training correctness check (full REACT check surface + the five Dyad dimensions + action head re-init).

Corresponds to the debug question "is Dyad correct". Reads the *.jsonl written with
DYAD_DIAG_ENABLED=1 under outputs/<algo>/<env>/<run>/ and reports PASS/FAIL + data per dimension,
matching the M*/L*/R*/E*/T* items of DYAD_RESULT_ANALYSIS_REQUIREMENTS.md.

Dyad passes only if all REACT external env pre-checks PASS and all five Dyad dimensions below PASS.

REACT pre-checks: tool/action/observation/timeout, trajectory-session-worker lifecycle,
reuse after close, reward aggregation, observation mask.

The five Dyad dimensions:
  1 · MASK   -- dual-head split, no leakage at inactive positions, rollout mask ≡ training mask.
  2 · LOSS   -- on-policy ratio, old==new logprob, numerical health, forward==backward.
  3 · ACTION HEAD -- **after every backbone update (weight sync), is the action head re-initialised
      from the latest lm_head** (data source: the dyad_vllm_model_runner.action_head_initialized event;
      compare whether the weight signature changes between adjacent inits).
      Plus a degraded gradient evidence (grad_norm>0).
  4 · REWARD -- single-step reward consistency, trajectory reward = sum over turns, GRPO advantage.
  5 · MAX TURN -- the cap always equals the configured value, no overrun, legal termination reasons.

For the full decode of raw token ids see experiments/shared/analysis/run/decode_trajectories.py.

Usage (cwd=Dynamic_ExpA):
  python experiments/shared/analysis/run/verify_dyad.py [debug_dir]
  Defaults to the newest run under outputs/<algo>/<env>/<run>/ that holds diagnostic jsonl.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys

from _mask_checks import unretained_observation_requests
from _env_checks import action_evidence, is_probe, lease_violations

OK = "\033[92mPASS\033[0m"
BAD = "\033[91mFAIL\033[0m"
# Neither a pass nor a failure: this run's data cannot answer the question arithmetically.
# Kept distinct from FAIL because reporting a defect that is not there sends the next person
# looking for it, and distinct from PASS because the check genuinely did not run.
NA_UNDECIDABLE = "\033[93mUNDECIDABLE\033[0m"


def _resolve_dir(argv: list[str]) -> str:
    """Newest run under outputs/<algo>/<env>/<run>/; single implementation in _events.resolve_dir."""
    from decode_trajectories import resolve_dir

    return resolve_dir(argv[1] if len(argv) > 1 else None)


def _env_pool_prefix(debug_dir: str) -> str:
    """Which env pool wrote this run: calc / alfworld / codegym all use '<env>_env_pool'.

    Hardcoding 'calc_env_pool' made the RE-* dimensions structurally unpassable for the other two
    envs -- their events were all present, just under a different component name, so the checks
    reported "the env never ran" for runs where it had run fine.
    """
    for fn in sorted(glob.glob(f"{debug_dir}/*_env_pool*.jsonl")):
        return os.path.basename(fn).split("_env_pool")[0] + "_env_pool"
    return "calc_env_pool"


def _events(debug_dir: str, prefix: str, event: str) -> list[dict]:
    out: list[dict] = []
    for fn in sorted(glob.glob(f"{debug_dir}/{prefix}*.jsonl")):
        # Keep event provenance: worker_idx is unique only within one process's environment pool.
        source = os.path.basename(fn)
        with open(fn, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == event:
                    rec["_source"] = source
                    out.append(rec)
    return out


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


def _recorded_p(evidence, cond: bool, tag: str, msg: str) -> bool | None:
    if not evidence:
        print(f"  [{NA_UNDECIDABLE}] {tag}: no required evidence; {msg}")
        return None
    return _p(cond, tag, msg)


class _Verdict:
    """FAIL dominates missing evidence; UNDECIDABLE must never become PASS or FAIL."""

    def __init__(self):
        self.result: bool | None = True

    def __iand__(self, result: bool | None):
        if self.result is False or result is False:
            self.result = False
        elif self.result is None or result is None:
            self.result = None
        return self

    @property
    def label(self):
        return {True: "all PASS", False: "FAIL present", None: "UNDECIDABLE (required evidence incomplete)"}[self.result]


def _tensor_stat(contract: dict, key: str) -> dict:
    v = (contract or {}).get(key)
    return v if isinstance(v, dict) else {}


def _grad_norm(event: dict) -> float:
    """The `grad_norm` an `optimizer_step` recorded, as a scalar and as a tensor summary.

    The diagnostics layer writes a tensor as `_tensor_summary` (a dict carrying `mean`) and a Python
    scalar as the number itself. verl 0.7's `optimizer_step()` returned a tensor, 0.9 returns a
    float -- and only the dict branch was written here, so after the port `.get("mean")` raised
    AttributeError and **the whole verify died two lines into its own report**: analysis/verify_dyad.md
    held nothing but a heading and post_analysis.status recorded `verify=... rc=1`.
    Degrading to 0 instead would be worse: it would silently make "gradients did flow" UNDECIDABLE
    for every run, forever.
    """
    value = (event or {}).get("grad_norm")
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _head_sig(ev: dict) -> tuple:
    """Summarize recorded weights without treating signature equality as a sync failure."""
    w = (((ev.get("action_head") or {}).get("parameters") or {}).get("weight") or {}).get("value") or {}
    return (
        tuple(round(float(x), 6) for x in (w.get("sample") or [])[:16]),
        w.get("mean"),
        w.get("min"),
        w.get("max"),
    )


def _events_by_source(debug_dir: str, prefix: str, event: str) -> dict[str, list[dict]]:
    """Same as `_events`, but keyed by source file so each rank's sequence stays separate.

    `_events` concatenates one file per pid in filename order, so with 2 ranks the flat list is
    [rank0 t1, rank0 t2, rank0 t3, rank1 t1, ...]. Comparing adjacent entries then compares rank0's
    last re-derive with rank1's *first* -- two records of different moments on different ranks,
    which are identical whenever the ranks agree. The check fails on a healthy run for a reason
    that has nothing to do with the head.
    """
    out: dict[str, list[dict]] = {}
    for fn in sorted(glob.glob(f"{debug_dir}/{prefix}*.jsonl")):
        recs = []
        with open(fn, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == event:
                    recs.append(rec)
        if recs:
            out[os.path.basename(fn)] = recs
    return out


def _is_env_failure(event: dict) -> bool:
    return bool(
        event.get("http_failure")
        or event.get("tool_failure")
        or event.get("timed_out")
        or event.get("env_actor_error")
    )


def _verify_react_surface(debug_dir: str, steps: list[dict], trajectories: list[dict]) -> bool:
    """Dyad must first pass the whole external env check surface of GRPO-REACT."""
    ok = True
    pool = _env_pool_prefix(debug_dir)
    pool_inits = _events(debug_dir, pool, "pool_init")
    resets = _events(debug_dir, pool, "env_reset")
    binds = _events(debug_dir, pool, "env_step_bind")
    closes = _events(debug_dir, pool, "env_close")
    aborts = _events(debug_dir, pool, "env_abort")

    print("\n# Pre-check · full REACT check surface (Dyad must include all of it and PASS)")

    # TOOL USE / ACTION
    ok &= _p(bool(steps), "RE-U1 tool was called", f"env_step={len(steps)}")
    with_action, invalid_missing = action_evidence(steps)
    ok &= _p(with_action == len(steps), "RE-U2 action evidence recorded for every attempt",
             f"with action={with_action}/{len(steps)} rejected attempts missing evidence={invalid_missing}")
    invalid = sum(1 for step in steps if step.get("invalid_action"))
    recognized = len(steps) - invalid
    ok &= _p(recognized > 0, "RE-U3 external env recognises the action",
             f"recognised={recognized}/{len(steps)} invalid_action={invalid}")

    # TRAJECTORY / SESSION / WORKER LIFECYCLE
    ok &= _p(bool(pool_inits) and all(event.get("ok") for event in pool_inits),
             "RE-E1 pool startup and health probe",
             f"pool_init={len(pool_inits)} failed={sum(1 for event in pool_inits if not event.get('ok'))}")

    request_ids = {step.get("request_id") for step in steps if not _is_probe(step.get("request_id"))}
    reset_ids = {event.get("session_id") for event in resets if not _is_probe(event.get("session_id"))}
    close_ids = {event.get("session_id") for event in closes if not _is_probe(event.get("session_id"))}
    ok &= _p(request_ids == reset_ids, "RE-E2 trajectory request_id == env session_id",
             f"tool trajectories={len(request_ids)} reset={len(reset_ids)} difference={len(request_ids ^ reset_ids)}")

    reset_worker = {event.get("session_id"): event.get("worker_idx") for event in resets}
    drift = [event for event in binds
             if reset_worker.get(event.get("session_id")) != event.get("worker_idx")]
    ok &= _p(bool(binds) and not drift, "RE-E3 one trajectory stays bound to one worker",
             f"worker drift={len(drift)}/{len(binds)} env step")

    close_counts = collections.Counter(event.get("session_id") for event in closes)
    close_bad = [session_id for session_id in reset_ids if close_counts[session_id] != 1]
    close_failed = [event for event in closes if not event.get("ok")]
    ok &= _p(bool(reset_ids) and reset_ids == close_ids and not close_bad and not close_failed,
             "RE-E4 every session is closed exactly once and successfully",
             f"reset={len(reset_ids)} close={len(close_ids)} missing/duplicate={len(close_bad)} close failed={len(close_failed)}")

    # Identify workers by (source pool, worker_idx) when merging events from multiple processes.
    reuse_before_close, active_sessions = lease_violations(pool_inits, resets, closes)

    ok &= _p(bool(resets) and reuse_before_close == 0 and active_sessions == 0,
             "RE-E5 session leases respect capacity and close",
             f"early reuse={reuse_before_close} still active at end of run={active_sessions} "
             f"abort={len(aborts)} pools={len({event.get('_source') for event in resets})}")

    # OBSERVATION / TIMEOUT / FAILURE
    successful_steps = [step for step in steps if not _is_env_failure(step)]
    # Same exclusion as above: the preflight's probe binds twice and never appears as an env_step.
    binds = [event for event in binds if not _is_probe(event.get("session_id"))]
    observations = [event for event in binds if event.get("observation_out") not in (None, "")]
    ok &= _p(len(binds) == len(successful_steps) and len(observations) == len(binds),
             "RE-C1 every successful env step returns an observation",
             f"successful steps={len(successful_steps)} bind={len(binds)} observation={len(observations)}")

    timed_out = sum(1 for step in steps if step.get("timed_out"))
    tool_failures = sum(1 for step in steps if step.get("tool_failure") or step.get("env_actor_error"))
    ok &= _p(timed_out == 0 and tool_failures == 0 and not aborts,
             "RE-TO1 no timeout / actor / tool failure",
             f"timeout={timed_out} tool_failure={tool_failures} abort={len(aborts)}")
    server_times = [float(event["server_ms"]) for event in binds if event.get("server_ms") is not None]
    if server_times:
        print(f"  env actor RPC: mean={sum(server_times) / len(server_times):.3f}ms "
              f"max={max(server_times):.3f}ms n={len(server_times)}")

    # REACT OBSERVATION MASK
    masked = [trajectory for trajectory in trajectories if trajectory.get("num_masked_tokens") is not None]
    no_llm = [trajectory for trajectory in masked if not (trajectory.get("num_response_tokens") or 0)]
    tool_trajectories = [trajectory for trajectory in masked if trajectory.get("request_id") in request_ids]
    unretained = unretained_observation_requests(
        tool_trajectories, _events(debug_dir, "dyad_tool_agent", "generation_result")
    )
    unmasked_tool = [trajectory for trajectory in tool_trajectories
                     if not (trajectory.get("num_masked_tokens") or 0)
                     and trajectory.get("request_id") not in unretained]
    ok &= _p(len(masked) == len(trajectories) and not no_llm,
             "RE-M1 every trajectory has recorded mask counts and LLM tokens",
             f"mask counts={len(masked)}/{len(trajectories)} without LLM token={len(no_llm)}")
    all_tool_trajectories_covered = len(tool_trajectories) == len(request_ids)
    ok &= _p(all_tool_trajectories_covered and not unmasked_tool,
             "RE-M2 retained tool observations are masked out of the policy loss",
             f"covered tool trajectories={len(tool_trajectories)}/{len(request_ids)} "
             f"unexplained masked_token=0={len(unmasked_tool)}; "
             f"observations not retained (generation/mask counts agree)={len(unretained)}")

    if resets:
        peak = max(int(event.get("peak_active", 0) or 0) for event in resets)
        max_wait = max(float(event.get("waited_ms", 0) or 0) for event in resets)
        pool_size = max(int(event.get("pool_size", 0) or 0) for event in resets)
        print(f"  env concurrency: pool_size={pool_size} peak_active={peak} max_wait={max_wait:.3f}ms")

    print("  RE-R1~R4 reward/advantage checks continue in the Dyad reward dimension below.")
    return ok


def verify(debug_dir: str) -> bool | None:
    print("=" * 70)
    print(f"Dyad run check: {debug_dir}")
    print("=" * 70)
    ok = _Verdict()

    gens = _events(debug_dir, "dyad_tool_agent", "generation_result")
    steps = _events(debug_dir, "dyad_tool_agent", "env_step")
    trs = _events(debug_dir, "dyad_tool_agent", "trajectory_summary")
    trj = _events(debug_dir, "agent_loop", "trajectory_reward_aggregated")
    advs = _events(debug_dir, "dyad_trainer", "advantage_computed")
    fwd = _events(debug_dir, "dyad_actor", "policy_micro_batch_forward")
    bwd = _events(debug_dir, "dyad_actor", "policy_micro_batch_backward")
    opt = _events(debug_dir, "dyad_actor", "optimizer_step")
    grad_norms = [_grad_norm(o) for o in opt]
    opt_counts = collections.Counter(o.get("_source") for o in opt)
    print(f"  optimizer_step event records by actor source={dict(opt_counts)}; "
          f"total rank records={len(opt)} (not global optimizer updates)")
    if opt_counts and len(set(opt_counts.values())) == 1:
        print(f"  synchronized-rank interpretation: {next(iter(opt_counts.values()))} updates "
              f"across {len(opt_counts)} actor sources, not {len(opt)}; "
              "global update IDs are not recorded, so rank pairing is not independently verified.")
    head_groups = _events_by_source(debug_dir, "dyad_vllm_model_runner", "action_head_initialized")
    heads = [h for group in head_groups.values() for h in group]

    from _env_checks import step_run_data, verify_environment_steps

    data = step_run_data(debug_dir)
    shared = data["shared"]
    # Native step evidence replaces only the legacy rollout checks, never actor evidence.
    ok &= (verify_environment_steps(debug_dir, data) if shared
           else _verify_react_surface(debug_dir, steps, trs))
    if not (fwd and bwd and opt):
        print(f"  [{NA_UNDECIDABLE}] complete training evidence unavailable: "
              f"forward={len(fwd)} backward={len(bwd)} optimizer_step={len(opt)}. "
              "Environment execution does not establish gradient or optimizer success.")
        ok &= None

    # ---------------- Dimension 1: MASK (M1-M3) ----------------
    print("\n# Dyad dimension 1 · dual-policy MASK")
    cand = collections.Counter()
    bad_allowed = 0
    n_vocab = 0
    for g in gens:
        n_vocab += g.get("num_vocab_decisions", 0) or 0
        for d in g.get("dyad_decisions", []) or []:
            cand[d["num_candidates"]] += 1
            if not d.get("chosen_in_allowed", False):
                bad_allowed += 1
    total_dec = sum(cand.values())
    print(f"  action decisions={total_dec}  vocab decisions={n_vocab}  admissible set size distribution={dict(sorted(cand.items()))}")
    # Readable decision samples: print a few actual choices "during dyad" (action name + admissible set + verbatim open-vocabulary parameter value).
    _samples = []
    for g in gens:
        for d in g.get("dyad_decisions", []) or []:
            if "chosen_str" in d:
                _samples.append(d)
        if len(_samples) >= 5:
            break
    if _samples:
        print("  sample decisions (chosen_str ∈ allowed_strs | surface_form | verbatim parameter value):")
        for d in _samples[:5]:
            _cs = d.get("chosen_str")
            # The dumped admissible list is capped at DYAD_DIAG_MAX_VALUES and ends in the
            # diagnostics layer's `{"__truncated_items__": N}` sentinel (see
            # agent_system/utils/diagnostics.py). Printing it raw puts a dict where an action belongs
            # and makes the list disagree with the num_candidates printed beside it.
            _raw = d.get("allowed_strs")
            _al = [x for x in _raw if not (isinstance(x, dict) and "__truncated_items__" in x)] \
                if isinstance(_raw, list) else _raw
            _dropped = sum(int(x["__truncated_items__"]) for x in _raw
                           if isinstance(x, dict) and "__truncated_items__" in x) \
                if isinstance(_raw, list) else 0
            _more = f" (+{_dropped} not dumped)" if _dropped else ""
            _rd = d.get("chosen_surface_form")
            _pv = d.get("argument_value_text")
            print(
                f"    pos={d.get('pos')} candidates{d.get('num_candidates')}={_al}{_more} chosen={_cs!r} "
                f"surface_form={_rd!r} param value={_pv!r}"
            )
    moi = mni = 0.0
    tool_active = vocab_active = 0
    split_bad = 0
    for f in fwd:
        al = f.get("alignment_stats", {}) or {}
        moi = max(moi, al.get("old_inactive_abs_max", 0) or 0)
        mni = max(mni, al.get("new_inactive_abs_max", 0) or 0)
        ta = al.get("tool_active_positions", 0) or 0
        va = al.get("vocab_active_positions", 0) or 0
        active = al.get("active_positions", 0) or 0
        tool_active += ta
        vocab_active += va
        if active != ta + va:
            split_bad += 1
    ok &= _recorded_p(fwd, moi == 0 and mni == 0, "M1 leakage at inactive positions", f"old_inactive_max={moi} new_inactive_max={mni}")
    ok &= _recorded_p(fwd, split_bad == 0, "M2 dual-head split without overlap", f"micro-batches with active!=tool+vocab={split_bad}/{len(fwd)}; tool_active={tool_active} vocab_active={vocab_active}")
    ok &= _recorded_p(total_dec, bad_allowed == 0, "M3 sampled action ∈ admissible set", f"violations={bad_allowed}/{total_dec}")

    # ---------------- Dimension 2: LOSS (L1-L4) ----------------
    print("\n# Dyad dimension 2 · LOSS")
    ratio_nonfinite = 0
    ratio_non1 = 0
    for f in fwd:
        rt = f.get("ratio", {}) or {}
        if rt.get("finite_fraction", 1.0) < 1.0:
            ratio_nonfinite += 1
        if rt.get("min") is not None and (abs((rt.get("min") or 1) - 1) > 1e-4 or abs((rt.get("max") or 1) - 1) > 1e-4):
            ratio_non1 += 1
    loss_nonfinite = sum(1 for b in bwd if (b.get("loss", {}) or {}).get("finite_fraction", 1.0) < 1.0)
    ok &= _recorded_p(fwd, ratio_nonfinite == 0, "L3 ratio all finite", f"non-finite micro-batches={ratio_nonfinite}/{len(fwd)}")
    ok &= _recorded_p(bwd, loss_nonfinite == 0, "L3 loss all finite", f"non-finite micro-batches={loss_nonfinite}/{len(bwd)}")
    fwd_counts = collections.Counter(f.get("_source") for f in fwd)
    bwd_counts = collections.Counter(b.get("_source") for b in bwd)
    ok &= _recorded_p(fwd and bwd and opt, fwd_counts == bwd_counts,
                      "L4 forward==backward count per actor source",
                      f"forward rank records={len(fwd)} backward rank records={len(bwd)} "
                      f"optimizer rank records={len(opt)}")
    print(f"  L1 policy ratio diagnostic: micro-batches with ratio≠1={ratio_non1}/{len(fwd)}; "
          "sequential optimizer updates can change ratios against fixed old-policy logprobs "
          "even when ppo_epochs=1. Ratio≈1 is expected only before policy updates with "
          "matching logprob computation.")
    print(f"  generic grad_norm per optimizer rank record={[round(x, 4) for x in grad_norms]}; "
          "generic/backbone gradients do not establish Dyad adapter gradients. "
          "Zero reward alone does not establish zero gradients (KL/entropy may contribute).")

    # ---------------- Dimension 3: ACTION HEAD re-init + gradient evidence ----------------
    print("\n# Dyad dimension 3 · ACTION HEAD (re-initialised from the latest lm_head after every backbone update)")
    n_init = len(heads)
    if heads:
        asz = heads[0].get("action_size")
        pcount = (heads[0].get("action_head") or {}).get("parameter_count")
        print(f"  action_head_initialized events={n_init}  action_size={asz}  parameter_count={pcount}")
    if n_init >= 2:
        pairs_checked = changed = 0
        for group in head_groups.values():
            ordered = sorted(group, key=lambda e: e.get("time", 0))
            for before, after in zip(ordered, ordered[1:]):
                pairs_checked += 1
                changed += _head_sig(before) != _head_sig(after)
        print(f"  [{NA_UNDECIDABLE}] E-reinit synchronization correctness: "
              f"ranks={len(head_groups)} adjacent pairs={pairs_checked} "
              f"changed signatures={changed}")
        print("      DirectActionHead is rebuilt from projector/cache state; policy gradients "
              "do not require its weights to change. Signatures alone cannot verify that "
              "every weight version was synchronized and rebuilt correctly.")
        ok &= None
    elif n_init == 1:
        print(f"  [{NA_UNDECIDABLE}] E-reinit: only 1 action_head_initialized (cold-start); "
              f"this run has too few training steps (optimizer_step={len(opt)}), a longer run is needed to observe the re-init after every sync.")
        ok &= None
    elif not glob.glob(f"{debug_dir}/*_pid*.jsonl"):
        # Diagnostics disabled (e.g. native t2bench evaluation): the head init is unobservable, not absent.
        print(f"  [{NA_UNDECIDABLE}] E-reinit: no diagnostic event streams (*_pid*.jsonl) were recorded; "
              "action head initialization is not observable in this run.")
        ok &= None
    else:
        ok &= _p(False, "E-reinit", "no action_head_initialized event (the rollout worker did not record the head init)")
    nonzero_grad_steps = sum(math.isfinite(x) and x > 0 for x in grad_norms)
    print(f"  Generic model gradient evidence: optimizer rank records with finite grad_norm>0="
          f"{nonzero_grad_steps}/{len(grad_norms)}; this is NOT Dyad action/adapter gradient validation.")
    adapter_norms = []
    for event in opt:
        value = event.get("dyad/adapter_grad_norm")
        if isinstance(value, dict):
            value = value.get("mean")
        if type(value) in (int, float):
            adapter_norms.append(float(value))
    adapter_nonzero = sum(math.isfinite(x) and x > 0 for x in adapter_norms)
    adapter_recorded = len(adapter_norms) == len(opt) and bool(opt)
    finite_adapter = all(math.isfinite(x) and x >= 0 for x in adapter_norms)
    if adapter_recorded and not finite_adapter:
        ok &= _p(False, "E-adapter gradients", "nonfinite/negative adapter_grad_norm recorded")
    elif not adapter_recorded or tool_active == 0 or adapter_nonzero == 0:
        print(f"  [{NA_UNDECIDABLE}] Dyad action/adapter gradient path NOT validated: "
              f"tool_active_positions={tool_active}, adapter norm records={len(adapter_norms)}/{len(opt)}, "
              f"positive adapter norm rank records={adapter_nonzero}. "
              "Vocabulary-only gradients and generic optimizer events are insufficient.")
        ok &= None
    else:
        ok &= _p(True, "E-adapter gradient evidence",
                 f"tool_active_positions={tool_active}, positive adapter norm rank records="
                 f"{adapter_nonzero}/{len(adapter_norms)}; not a checkpoint/update correctness proof")

    # ---------------- Dimension 4: REWARD (R1-R4) ----------------
    print("\n# Dyad dimension 4 · REWARD / ADVANTAGE (includes the REACT reward aggregation)")
    if shared:
        print("  Shared decision/native rewards checked above; legacy turn_scores do not apply.")
    else:
        infra = sum(1 for s in steps if s.get("http_failure"))
        nonzero_steps = sum(1 for s in steps if float(s.get("reward", 0) or 0))
        print(f"  env_step={len(steps)} steps with non-zero reward={nonzero_steps} infra_step={infra}")
        traj_bad = sum(1 for t in trj if abs(sum(t.get("turn_scores") or []) - (t.get("reward_score") or 0)) > 1e-6)
        nz = sum(1 for t in trj if t.get("reward_score"))
        won = sum(1 for t in trj if t.get("success"))
        ok &= _recorded_p(trj, traj_bad == 0, "R1 reward==sum(turn_scores)", f"violations={traj_bad}/{len(trj)}; non-zero={nz} won={won}")
    if not advs:
        print(f"  [{NA_UNDECIDABLE}] no advantage evidence recorded")
        ok &= None
    print("  R4 advantage/step:")
    for a in advs:
        adv = _tensor_stat(a.get("contract", {}), "advantages")
        print(f"    step{a.get('global_step')}: nonzero={adv.get('nonzero_fraction')} finite={adv.get('finite_fraction')} range=[{adv.get('min')},{adv.get('max')}]")

    # ---------------- Dimension 5: MAX TURN (T1-T3) ----------------
    print("\n# Dyad dimension 5 · MAX TURN")
    if shared:
        print(f"  Shared decision cap={data['cap']}; continuity and termination use environment_step "
              "metadata above, not legacy max_tool_turns or message num_turns.")
    else:
        mtt = collections.Counter(g.get("max_tool_turns") for g in gens)
        tool_turns = [t.get("tool_turns", 0) or 0 for t in trs]
        turn_idx = [s.get("turn_index", 0) or 0 for s in steps if s.get("turn_index") is not None]
        cap = next(iter(mtt)) if len(mtt) == 1 else None
        ok &= _recorded_p(gens, len(mtt) == 1, "T1 max_tool_turns single valued", f"value distribution={dict(mtt)}")
        max_tt = max(tool_turns) if tool_turns else 0
        max_ti = max(turn_idx) if turn_idx else 0
        over = sum(1 for x in tool_turns if cap is not None and x > cap)
        ok &= _recorded_p(trs, over == 0 and (cap is None or max_ti <= cap), "T2 no trajectory overruns", f"max tool_turns={max_tt} max turn_index={max_ti} cap={cap} overruns={over}")
        reasons = collections.Counter(t.get("termination_reason") for t in trs)
        print(f"  T3 termination_reason distribution={dict(reasons)} (None can mean an observation overflowing response_length)")

    print("\n" + "=" * 70)
    print(f"Overall verdict: {ok.label}")
    print("=" * 70)
    return ok.result


if __name__ == "__main__":
    from decode_trajectories import save_report

    _dir = _resolve_dir(sys.argv)
    with save_report("verify_dyad", _dir):
        _ok = verify(_dir)
    # Missing required evidence is non-success, but not an observed failed check.
    sys.exit(2 if _ok is None else (0 if _ok else 1))
