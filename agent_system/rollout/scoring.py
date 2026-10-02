# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Transport environment outcomes, validity and evidence before the ordinary reward fallback.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def score_environment_output(final_output):
    # Use environment episode outcomes when present and retain the ordinary reward-manager path otherwise.
    from verl.experimental.agent_loop.agent_loop import (
        log_event,
    )

    _ef = final_output.extra_fields
    # DYAD: terminal official results bypass turn sums and reward-model fallback.
    if _ef.get("terminal_official"):
        result = _ef["episode_result"]
        valid = bool(result.get("metric_valid", False))
        # DYAD-DIVE: infrastructure failures must not become training rewards or evaluation failures.
        if _ef.get("runtime_protocol") == "dive" and not valid:
            raise RuntimeError("DIVE cannot score an invalid environment or judge result")
        score = result.get("official_reward") if result.get("official_scored") else result.get("attempt_reward")
        if valid and score is None:
            raise ValueError("Valid episode result is missing its score")
        # Tensor transport needs a number; validity and the nullable result remain explicit.
        final_output.reward_score = float(score) if valid else 0.0
        _ef["reward_extra_info"] = {
            "score": float(score) if valid else float("nan"),
            "metric_valid": float(valid),
        }
        return True
    if _ef is not None and (_ef.get("multi_turn_scored") or _ef.get("turn_scores")):
        turn_scores = final_output.extra_fields.get("turn_scores") or []
        final_output.reward_score = sum(float(score) for score in turn_scores) if turn_scores else 0.0
        # Pass raw score and environment-reported won; the trainer maps these to validation panels.
        final_output.extra_fields["reward_extra_info"] = {"score": final_output.reward_score}
        # DYAD-EVAL: all trainer backends already transport reward_extra_info
        # to generation dumps. These fields come from the environment lifecycle,
        # not from the reward value or the job's eventual exit code.
        evidence = final_output.extra_fields.get("environment_evidence")
        if evidence is not None:
            final_output.extra_fields["reward_extra_info"].update(
                {key: evidence[key] for key in ("task_id", "metric_valid", "service_error")}
            )
        if "won" in final_output.extra_fields:
            final_output.extra_fields["reward_extra_info"]["won"] = float(bool(final_output.extra_fields["won"]))
        # DYAD-EVAL: record native task outcomes even if tool feedback was
        # truncated before the loop's reward bookkeeping. Training is unchanged.
        if evidence is not None:
            final_output.extra_fields["reward_extra_info"]["won"] = float(evidence["won"])
            if evidence["benchmark"] == "webshop":
                final_output.extra_fields["reward_extra_info"]["score"] = evidence["task_score"]
        # Format validity = parsed tool-call turns / generated turns; it is diagnostic, not reward.
        if "fmt_attempts" in final_output.extra_fields:
            _fa = final_output.extra_fields["fmt_attempts"]
            _fv = final_output.extra_fields.get("fmt_valid", 0)
            final_output.extra_fields["reward_extra_info"]["fmt"] = (_fv / _fa) if _fa > 0 else 0.0
        _nonzero = [s for s in turn_scores if s] if turn_scores else []
        log_event(
            "agent_loop",
            "trajectory_reward_aggregated",
            turn_scores=turn_scores,
            reward_score=final_output.reward_score,
            num_turns=getattr(final_output, "num_turns", None),
            num_turn_scores=len(turn_scores) if turn_scores else 0,
            num_nonzero_turn_scores=len(_nonzero),
            max_turn_score=max(turn_scores) if turn_scores else 0.0,
            # Record train/validation identity and environment-reported success.
            validate=bool(final_output.extra_fields.get("validate", False)),
            success=bool(final_output.extra_fields.get("won", False)),
        )
    return False
