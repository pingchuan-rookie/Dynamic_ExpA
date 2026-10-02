"""Official episode accounting, separate from transport reward tensors."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def episode_key(record):
    if record['benchmark'] != 't2bench':
        raise ValueError('Tau evaluation metrics require benchmark=t2bench')
    return tuple(str(record[k]) for k in ("benchmark", "domain", "split", "task_id", "trial"))


def summarize_episodes(planned, results):
    """Reject unexpected/duplicate results and never hide unfinished trials."""
    plan = {episode_key(p): dict(p) for p in planned}
    if len(plan) != len(planned):
        raise ValueError("Duplicate planned evaluation episodes")
    seen = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Missing structured episode_result")
        key = episode_key(result)
        if key not in plan or key in seen:
            raise ValueError(f"Unexpected or duplicate episode result: {key}")
        result = dict(result)
        reward = result.get("official_reward")
        scored = result.get("official_scored", False)
        if scored and (not isinstance(reward, (float, int)) or not math.isfinite(reward)):
            raise ValueError(f"Invalid official reward: {key}")
        if not scored and reward is not None:
            raise ValueError(f"Unscored episode cannot carry official reward: {key}")
        seen[key] = {**plan[key], **result}
    episodes = [seen.get(key, {**p, "status": "missing", "official_scored": False,
                              "official_reward": None, "metric_valid": False}) for key, p in plan.items()]
    groups = defaultdict(list)
    for episode in episodes:
        groups[f"{episode['benchmark']}/{episode['domain']}/{episode['split']}"].append(episode)
    summaries = {}
    for name, rows in groups.items():
        scored = [r for r in rows if r.get("official_scored")]
        valid = [r for r in rows if r.get("metric_valid", r.get("official_scored", False))]
        values = [r.get("official_reward") if r.get("official_scored") else r.get("attempt_reward") for r in valid]
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
            raise ValueError(f"Invalid attempt reward in {name}")
        complete = len(valid) == len(rows)
        successes = sum(value == 1.0 for value in values)
        summaries[name] = {"planned": len(rows), "attempted": sum(r['status'] != 'missing' for r in rows),
                           "official_scored": len(scored), "unfinished": len(rows) - len(valid),
                           "status_counts": dict(Counter(r['status'] for r in rows)),
                           "success_rate": successes / len(rows) if complete else None,
                           "official_mean": sum(r['official_reward'] for r in scored) / len(rows) if len(scored) == len(rows) else None,
                           "diagnostic_valid_subset_mean": sum(values) / len(values) if values else None}
        summaries[name]['pass^1'] = summaries[name]['success_rate']
    return {"complete": all(s["unfinished"] == 0 for s in summaries.values()),
            "planned": len(episodes), "groups": summaries, "episodes": episodes}


def finish_tau_validation(trainer, results):
    """Write auditable results and return only meaningful numeric logger metrics."""
    summary = summarize_episodes(trainer.val_dataset.planned_episodes, results)
    output = trainer.config.trainer.get("validation_data_dir") or trainer.config.trainer.default_local_dir
    import tempfile
    Path(output).mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=f"tau_eval_step_{trainer.global_steps}_", dir=output))
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"[tau-evaluation] results: {destination / 'summary.json'}")
    metrics = {}
    for group, values in summary["groups"].items():
        for key in ("planned", "attempted", "official_scored", "unfinished", "success_rate", "official_mean"):
            if values[key] is not None:
                metrics[f"val-tau/{group}/{key}"] = values[key]
    return metrics


def mark_tau_padding(batch, pad_size):
    """Padding is transport only and must never initialize another environment."""
    if "tau_episode" in batch.non_tensor_batch:
        import numpy as np
        batch.non_tensor_batch["tau_padding"] = np.array(
            [False] * (len(batch) - pad_size) + [True] * pad_size, dtype=object)
    return batch
