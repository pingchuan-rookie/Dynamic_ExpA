"""Complete-population accounting for independent SWE-bench attempts."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile


IDENTITY_FIELDS = ("benchmark", "instance_id", "trial", "seed", "episode_id")


def episode_key(record):
    if record.get("benchmark") != "swebench_verified":
        raise ValueError("Expected SWE-bench Verified episode")
    return tuple(record[key] for key in IDENTITY_FIELDS)


def summarize_episodes(planned, results):
    plan = {episode_key(row): dict(row) for row in planned}
    if not plan or len(plan) != len(planned):
        raise ValueError("Evaluation plan must be nonempty and unique")
    seen = {}
    for result in results:
        key = episode_key(result)
        if key not in plan or key in seen:
            raise ValueError("Unexpected or duplicate SWE-bench episode")
        valid = result.get("metric_valid") is True and result.get("official_scored") is True
        reward = result.get("official_reward")
        if valid and (type(reward) not in (float, int) or reward not in (0, 1)):
            raise ValueError("Official SWE-bench reward must be binary")
        if valid and type(result.get("resolved")) is not bool:
            raise ValueError("Scored SWE-bench result requires resolved boolean")
        if valid and result["resolved"] != bool(reward):
            raise ValueError("Official resolved and reward disagree")
        if not valid and reward is not None:
            raise ValueError("Unscored/infrastructure failures cannot carry an official reward")
        seen[key] = {**plan[key], **result}
    episodes = [seen.get(key, {**row, "status": "missing", "metric_valid": False,
                             "official_scored": False, "official_reward": None})
                for key, row in plan.items()]
    trials = {}
    for trial in sorted({row["trial"] for row in episodes}):
        rows = [row for row in episodes if row["trial"] == trial]
        scored = [row for row in rows if row.get("metric_valid") is True and row.get("official_scored") is True]
        complete = len(scored) == len(rows)
        resolved = sum(row["resolved"] for row in scored)
        trials[str(trial)] = {"planned": len(rows), "official_scored": len(scored),
                              "resolved": resolved, "complete": complete,
                              "resolved_percentage": 100.0 * resolved / len(rows) if complete else None}
    complete = all(row["complete"] for row in trials.values())
    return {"benchmark": "swebench_verified", "complete": complete, "planned": len(episodes),
            "official_scored": sum(row["official_scored"] for row in trials.values()),
            "status_counts": dict(Counter(row["status"] for row in episodes)), "trials": trials,
            "resolved_percentage": (sum(row["resolved_percentage"] for row in trials.values()) / len(trials)
                                    if complete else None),
            "metric": "mean_independent_trial_resolved_percentage", "episodes": episodes}


def finish_swebench_validation(trainer, results):
    summary = summarize_episodes(trainer.val_dataset.planned_episodes, results)
    output = trainer.config.trainer.get("validation_data_dir") or trainer.config.trainer.default_local_dir
    Path(output).mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=f"swebench_eval_step_{trainer.global_steps}_", dir=output))
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"[swebench-evaluation] results: {destination / 'summary.json'}")
    if not summary["complete"]:
        raise RuntimeError("Incomplete SWE-bench validation: official scoring did not cover every planned attempt")
    return {"val-swebench/resolved_percentage": summary["resolved_percentage"],
            "val-swebench/planned": summary["planned"], "val-swebench/official_scored": summary["official_scored"]}
