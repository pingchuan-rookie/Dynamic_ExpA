"""Shared diagnostics for exclusive workers and shared-catalog session leases."""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path


def _json_rows(path):
    # Corrupt evidence must not silently turn into a successful empty check.
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object in {path}")
                yield row


def step_metadata(row):
    extra = row.get("extra_fields") or {}
    value = row.get("environment_step", extra.get("environment_step"))
    if value is not None and not isinstance(value, dict):
        raise ValueError("environment_step must be a metadata object")
    return value


def _saved_value(value):
    # Evaluation writers save Hydra override syntax (including quoted strings),
    # whereas training writers may already have decoded these scalar values.
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def step_run_data(directory):
    """Read shared decision evidence without manufacturing legacy trajectories."""
    root = Path(directory)
    config = {}
    for name in ("resolved_config.json", "model_config.json"):
        path = root / name
        if path.is_file():
            config[name] = json.loads(path.read_text())
    resolved = config.get("resolved_config.json", {})
    model = config.get("model_config.json", {})
    hydra = {key: _saved_value(value) for key, value in resolved.get("hydra", {}).items()}
    protocol = model.get("training_protocol", {})
    settings = protocol.get("step_rollout", {})
    enabled = settings.get("enabled", resolved.get("parameters", {}).get("STEP_ROLLOUT_ENABLED"))
    shared = (str(enabled).lower() == "true" or
              hydra.get("actor_rollout_ref.rollout.agent.default_agent_loop") == "environment_step_agent")
    cap = settings.get("max_steps", hydra.get("algorithm.step_rollout.max_steps"))
    if cap is None and shared:
        cap = resolved.get("parameters", {}).get("MAX_ASSISTANT_TURNS")
    cap = int(cap) if cap is not None else None
    events, generations, records = [], [], []
    for path in sorted(root.glob("*_pid*.jsonl")):
        for row in _json_rows(path):
            row["_source"] = path.name
            events.append(row)
            if step_metadata(row) is not None:
                records.append(row)
    # GSM8K validation returns only the final decision, retaining its original
    # step_index/episode_length. Training and other full-decision dumps must
    # still satisfy completeness; never infer sampling from missing rows alone.
    benchmark = resolved.get("benchmark", model.get("benchmark"))
    profile = settings.get("profile", hydra.get("algorithm.step_rollout.profile"))
    terminal_validation = benchmark == "gsm8k" or profile in {"gsm8k_native_v2", "gsm8k_project_v1"}
    folders = [(root / "val_generations", True), (root / "train_generations", False),
               (root / "rollout_generations", False)]
    for key in ("trainer.validation_data_dir", "trainer.rollout_data_dir"):
        value = hydra.get(key)
        if value and str(value).lower() not in {"none", "null"}:
            path = Path(value).expanduser()
            folders.append((path if path.is_absolute() else root / path,
                            key == "trainer.validation_data_dir"))
    seen = set()
    summaries = []
    for folder, validation in folders:
        if validation and shared:
            for pattern in ("tau_eval_step_*/summary.json", "swebench_eval_step_*/summary.json"):
                for path in sorted(folder.glob(pattern)):
                    if path.resolve() in seen:
                        continue
                    seen.add(path.resolve())
                    summary = json.loads(path.read_text())
                    if not isinstance(summary, dict) or not isinstance(summary.get("episodes"), list):
                        raise ValueError(f"Invalid native evaluation summary: {path}")
                    if not all(isinstance(row, dict) for row in summary["episodes"]):
                        raise ValueError(f"Invalid native evaluation episode: {path}")
                    summary["_source"] = str(path)
                    summaries.append(summary)
        for path in sorted(folder.glob("*.jsonl")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            for row in _json_rows(path):
                row["_source"] = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
                row["_terminal_validation"] = validation and terminal_validation
                generations.append(row)
                if step_metadata(row) is not None:
                    records.append(row)
    return {"shared": shared or bool(records), "cap": cap, "events": events,
            "generations": generations, "records": records, "evaluation_summaries": summaries}


def verify_native_evaluations(summaries):
    """Check recorded terminal results; submitted model text is not execution proof."""
    ok, evidence = True, False
    for summary in summaries:
        episodes = summary["episodes"]
        identities = [e.get("episode_id") for e in episodes]
        valid = (type(summary.get("planned")) is int and summary["planned"] > 0
                 and len(episodes) == summary["planned"] and summary.get("complete") is True
                 and all(isinstance(i, str) and i for i in identities)
                 and len(set(identities)) == len(identities))
        responses, reported_calls, scored = 0, 0, 0
        for episode in episodes:
            official = episode.get("official_scored")
            valid &= episode.get("metric_valid") is True and type(official) is bool
            scored += official is True
            if official is True:
                valid &= _finite(episode.get("official_reward"))
            calls = episode.get("tool_calls_count", 0)
            valid &= type(calls) is int and calls >= 0
            if type(calls) is int and calls > 0:
                reported_calls += calls
            events = episode.get("tool_events", [])
            if not isinstance(events, list):
                valid = False
                continue
            for event in events:
                call = event.get("call", {}) if isinstance(event, dict) else {}
                response = event.get("response", {}) if isinstance(event, dict) else {}
                paired = (isinstance(call, dict) and isinstance(response, dict)
                          and isinstance(call.get("id"), str) and bool(call["id"])
                          and call["id"] == response.get("id")
                          and isinstance(call.get("name"), str) and bool(call["name"])
                          and isinstance(response.get("content"), str))
                valid &= paired
                responses += bool(paired)
        evidence |= responses > 0 or reported_calls > 0
        ok &= valid
        print(f"  [{'PASS' if valid else 'FAIL'}] native evaluation results: "
              f"episodes={len(episodes)} official_scored={scored} "
              f"unscored={len(episodes) - scored} recorded tool responses={responses} "
              f"reported tool calls={reported_calls}")
        print(f"  terminal statuses={dict(collections.Counter(e.get('status') for e in episodes))}")
        print("  [UNDECIDABLE] Missing pool lifecycle/decision metadata is not reconstructed "
              "from terminal summaries. Unscored attempts are not official zero rewards; "
              "tool errors are not successful task actions.")
    return ok, evidence


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_step_metadata(records, cap):
    """Check recorded decisions; require completeness only for full-decision sources."""
    if not records:
        print("  [UNDECIDABLE] environment_step metadata not persisted; decision continuity, "
              "executed/action_valid, reward aggregation and termination cannot be verified.")
        return True  # This is explicitly outside the recorded-evidence verdict.
    groups = collections.defaultdict(list)
    for row in records:
        meta = step_metadata(row)
        groups[(row["_source"], meta.get("trajectory_id"))].append(row)
    ok = True
    for (source, trajectory), rows in groups.items():
        metas = [step_metadata(row) for row in rows]
        length = metas[0].get("episode_length")
        group_id = metas[0].get("group_id")
        protocol = metas[0].get("protocol_version")
        terminal_only = len(rows) == 1 and rows[0].get("_terminal_validation") is True
        partial = terminal_only and type(length) is int and length > 1
        valid = (bool(trajectory) and type(length) is int and length > 0
                 and (cap is None or length <= cap)
                 and all(type(m.get("step_index")) is int for m in metas))
        if terminal_only:
            valid &= type(length) is int and metas[0].get("step_index") == length - 1
        else:
            valid &= (type(length) is int and len(rows) == length
                      and sorted(m.get("step_index") for m in metas if type(m.get("step_index")) is int)
                      == list(range(length)))
        rewards = [m.get("env_reward") for m in metas]
        valid &= all(_finite(r) for r in rewards)
        total = math.fsum(rewards) if all(_finite(r) for r in rewards) else float("nan")
        for row, meta in zip(rows, metas):
            valid &= (meta.get("schema_version") == 1 and meta.get("protocol_version") in (1, 2)
                      and bool(group_id) and meta.get("group_id") == group_id
                      and meta.get("protocol_version") == protocol and meta.get("episode_length") == length
                      and _finite(meta.get("episode_reward"))
                      and (partial or math.isclose(meta["episode_reward"], total, rel_tol=1e-6, abs_tol=1e-6))
                      and all(type(meta.get(k)) is bool for k in ("done", "executed", "action_valid")))
            last = meta.get("step_index") == length - 1 if type(length) is int else False
            reason = meta.get("termination_reason")
            valid &= (reason == ("env_done" if meta.get("done") else "max_steps") if last
                      else reason == "continue" and meta.get("done") is False)
            if last and reason == "max_steps" and cap is not None:
                valid &= length == cap
            turns = row.get("num_turns", row.get("__num_turns__"))
            if turns is not None:
                valid &= type(length) is int and turns == 2 * length
        label = "recorded final decision metadata/termination" if partial else "decision continuity/rewards/termination"
        print(f"  [{'PASS' if valid else 'FAIL'}] {label}: "
              f"source={source} trajectory={trajectory} rows={len(rows)} episode_length={length}")
        if partial:
            print("  [UNDECIDABLE] decision continuity and episode reward aggregation: "
                  "GSM8K validation persists only the final decision; earlier decision metadata/rewards "
                  "are not recorded. Final-row metadata is not a full trajectory.")
        ok &= valid
    print(f"  metadata episodes={len(groups)} decisions={len(records)}; "
          "num_turns counts messages (2 per decision), not decisions.")
    metas = [step_metadata(row) for row in records]
    any_valid = any(m.get("action_valid") is True for m in metas)
    print(f"  [{'PASS' if any_valid else 'FAIL'}] at least one valid-format decision; "
          f"recorded executed decisions={sum(m.get('executed') is True for m in metas)} "
          f"valid-format decisions={sum(m.get('action_valid') is True for m in metas)} "
          f"invalid-format decisions={sum(m.get('action_valid') is False for m in metas)}; "
          "format validity does not prove native task success.")
    ok &= any_valid
    evidence = [row.get("environment_evidence", (row.get("extra_fields") or {}).get("environment_evidence"))
                for row in records]
    evidence = [value for value in evidence if value is not None]
    if evidence:
        healthy = all(isinstance(value, dict) and value.get("metric_valid") is True
                      and not value.get("service_error") for value in evidence)
        print(f"  [{'PASS' if healthy else 'FAIL'}] native metric_valid and no recorded service_error: "
              f"{len(evidence)}/{len(records)} rows recorded")
        ok &= healthy
    return ok


def verify_environment_steps(directory, data=None):
    """Shared rollout checks use native environment evidence, never missing legacy logs."""
    data = data or step_run_data(directory)
    events = [e for e in data["events"] if "_env_pool" in e["_source"]]
    pools = [e for e in events if e.get("event") == "pool_init"]
    real = [e for e in events if not is_probe(e.get("session_id"))]
    resets = [e for e in real if e.get("event") == "env_reset"]
    steps = [e for e in real if e.get("event") == "env_step_bind"]
    closes = [e for e in real if e.get("event") == "env_close"]
    key = lambda e: (e["_source"], e.get("session_id"))
    reset_counts = collections.Counter(map(key, resets))
    close_counts = collections.Counter(map(key, closes))
    step_counts = collections.Counter(map(key, steps))
    workers = {key(e): e.get("worker_idx") for e in resets}
    violations, active = lease_violations(pools, resets, closes)
    cap = data["cap"]
    print("\n# SHARED ENVIRONMENT STEP rollout (native evidence; probes excluded)")
    print(f"  native episodes={len(resets)} executed environment steps={len(steps)} "
          f"steps per session={dict(collections.Counter(step_counts.values()))} configured decision cap={cap}")
    checks = {}
    if events:
        checks = {
            "healthy native pools": bool(pools) and all(e.get("ok") is True for e in pools),
            "real environment executions": bool(steps),
            "one reset and successful close per session": bool(resets) and reset_counts == close_counts
            and all(n == 1 for n in reset_counts.values()) and all(e.get("ok") is True for e in closes),
            "every execution belongs to a reset session": bool(steps) and set(step_counts) <= set(reset_counts),
            "worker binding preserved": all(key(e) in workers and workers[key(e)] == e.get("worker_idx")
                                            for e in steps + closes),
            "all leases returned within capacity": violations == 0 and active == 0,
            "native actions and returned observations recorded": bool(steps) and all(
                isinstance(e.get("action_in"), str) and isinstance(e.get("observation_out"), str)
                and bool(e["observation_out"]) for e in steps),
            "finite native rewards": bool(steps) and all(_finite(e.get("reward")) for e in steps),
            "no native aborts or recorded tool failures": not any(
                e.get("event") == "env_abort" or any(e.get(k) for k in
                ("timed_out", "tool_failure", "http_failure", "env_actor_error")) for e in real),
        }
        # Pool calls need not equal decisions (invalid decisions or multi-action CodeGym).
        # Only ALFWorld/WebShop execute at most one native call per decision.
        single_action = all(e["_source"].startswith(("alfworld_", "webshop_")) for e in steps)
        if single_action:
            checks["one or more native steps per single-action episode"] = set(step_counts) == set(reset_counts)
            if cap is not None:
                checks["native executions do not exceed decision cap"] = all(n <= cap for n in step_counts.values())
        reset_times = {key(e): e.get("time") for e in resets}
        close_times = {key(e): e.get("time") for e in closes}
        checks["native steps occur inside their lease"] = all(
            _finite(e.get("time")) and _finite(reset_times.get(key(e)))
            and _finite(close_times.get(key(e)))
            and reset_times[key(e)] <= e["time"] <= close_times[key(e)] for e in steps)
        ended = set()
        terminal_order = True
        for event in sorted(steps, key=lambda e: e.get("time", 0)):
            terminal_order &= key(event) not in ended and type(event.get("done")) is bool
            if event.get("done"):
                ended.add(key(event))
        checks["no native calls after terminal result"] = terminal_order
        for label, passed in checks.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        print(f"  terminal native steps={sum(e.get('done') is True for e in steps)} "
              f"nonzero native rewards={sum(bool(e.get('reward')) for e in steps)}")
        print("  Native rewards are unscaled environment outcomes, not necessarily policy rewards. "
              "Action execution does not prove recognition/validity; absent invalid_action is not success.")
        print("  [UNDECIDABLE] timeout coverage beyond recorded failures; pool logs do not record all policy/parser outcomes.")
    else:
        print("  [UNDECIDABLE] native pool evidence unavailable; reset/close/worker lifecycle not verified.")
    ok = verify_step_metadata(data["records"], cap)
    native_ok, native_evidence = verify_native_evaluations(data.get("evaluation_summaries", []))
    ok &= native_ok
    if steps and data["records"]:
        print("  [UNDECIDABLE] environment_step.trajectory_id and native session_id use different "
              "identities; per-decision/native action and reward correspondence is not verified.")
    # If neither source can prove interaction, do not pass an empty diagnostic run.
    evidence = (bool(steps) or native_evidence
                or any(step_metadata(r).get("executed") is True for r in data["records"]))
    ok &= all(checks.values()) and evidence
    print(f"  [{'PASS' if evidence else 'FAIL'}] recorded environment interaction evidence")
    by_file = collections.Counter(r["_source"] for r in data["generations"])
    for source, count in sorted(by_file.items()):
        print(f"  {source}: {count} independent generation rows (not full trajectories)")
    if data["generations"] and not data["records"]:
        print("  [UNDECIDABLE] generation rows omit environment_step/session IDs; "
              "cannot join validation rows to native leases or label all native calls as validation.")
    print("  Message turns are 2 * episode decisions in environment_step_agent; "
          "num_turns=100 means 50 decisions, not a 50-step cap overrun.")
    print("  [UNDECIDABLE] token masks, Dyad candidate/replay alignment, advantages, "
          "backward gradients and optimizer updates are NOT verified by environment/generation evidence.")
    status = Path(directory) / "run.status"
    if status.is_file():
        print(f"  Recorded run status: {status.read_text().strip()}; "
              "an environment-only PASS is not training/optimizer success.")
    print(f"  Environment verdict (recorded dimensions only): {'PASS' if ok else 'FAIL'}")
    return ok


def is_probe(session_id):
    return isinstance(session_id, str) and (
        session_id.endswith("-probe") or session_id in {"webshop-probe-a", "webshop-probe-b"}
    )


def lease_violations(pool_inits, resets, closes):
    """Count duplicate sessions/capacity overflow; legacy workers have capacity one."""
    capacities = {
        event.get("_source", ""): int(event.get("sessions_per_backend", 1))
        for event in pool_inits
    }
    active = {}
    violations = 0
    for event in sorted(resets + closes, key=lambda item: float(item.get("time", 0))):
        source = event.get("_source", "")
        worker = (source, event.get("worker_idx"))
        sessions = active.setdefault(worker, set())
        session = event.get("session_id")
        if event.get("event") == "env_reset":
            capacity = capacities.get(source, 1)
            if session in sessions or len(sessions) >= capacity:
                violations += 1
            sessions.add(session)
        else:
            if session not in sessions:
                violations += 1
            sessions.discard(session)
    return violations, sum(len(sessions) for sessions in active.values())


def action_evidence(steps):
    """Do not confuse rejected attempts with sent actions, or hide missing evidence."""
    missing = [step for step in steps if step.get("action_sent") in (None, "")]
    invalid_missing = sum(bool(step.get("invalid_action")) for step in missing)
    return len(steps) - len(missing), invalid_missing
