# Single-turn capability evaluation

This directory evaluates MMLU-Pro, HMMT February 2026 and LiveCodeBench v6 through an OpenAI-compatible inference API.
Evaluation scripts prepare data, send requests, score answers and write reports. Model serving is documented in the [standalone inference API](../../agent_system/inference/README.md).

## Files

| File | Purpose |
|---|---|
| `evaluate.py` | Benchmark entrypoint; validate arguments, model identity and output directory |
| `backend.py` | Data and scoring integration for pinned EvalScope |
| `serve.py` | vLLM-powered serving entrypoint |
| `export_policy.py`, `policy_identity.py` | Text-policy export and file-identity checks |
| `hmmt.py` | HMMT repeated sampling and avg@4 validation |
| `checker.py`, `process_checker.py`, `docker_checker.py` | LiveCodeBench grading configuration and execution backends |
| `token_metrics.py` | Output-token statistics for final answers |
| `summarize.py` | Paired capability-score comparison before and after training |
| `wandb_tracking.py` | Metric logging and upload of saved results |

## Protocol

| Benchmark | Data and prompts | Sampling | Default maximum output |
|---|---|---|---:|
| `mmlu_pro` | test; 5-shot within the same subject | One answer per question, temperature=0, seed=42 | 2048 |
| `hmmt26` | MathArena's official 33 problems, train split; 0-shot | Four answers per question, temperature=0.6, seed=42–45 | 8192 |
| `livecodebench_v6` | Cumulative `release_v6` set, Code Generation | One answer per question, temperature=0, seed=42 | 16384 |

All three benchmarks use single-turn, tool-free requests with `args={}`. Thinking is off by default and must remain off for Qwen3.5.
HMMT avg@4 is the mean accuracy across four answers for every problem. LiveCodeBench uses pass@1 without self-repair.
`release_v6` includes `test.jsonl` through `test6.jsonl`.

Pinned dataset versions are in `_TASKS` in [backend.py](backend.py); the EvalScope version is specified in
[backend_requirements.txt](backend_requirements.txt). Before/after-training comparisons must use identical data versions, prompts,
sample counts and generation budgets.

## Prepare dependencies and serving

Run from the repository root and install evaluation dependencies in a separate environment:

```bash
uv venv .venvs/capability-eval --python 3.12
uv pip install --python .venvs/capability-eval/bin/python \
  -r experiments/capability_eval/backend_requirements.txt

.venvs/expa-verl/bin/python -m agent_system.inference.server \
  --model-path /absolute/path/to/model --model evaluated-model
```

Use the service's `--checkpoint` argument for a full trained checkpoint. Dyad restores the policy, projector and saved encoder;
tool-free requests generate text with that same loaded model. The service exports the policy for ordinary text checkpoints.
See [serving](../../agent_system/inference/README.md) for restoration options and hardware settings.

## Run

After starting the service, run in another terminal:

```bash
.venvs/capability-eval/bin/python experiments/capability_eval/evaluate.py mmlu_pro \
  --api-url http://127.0.0.1:8000/v1 --model evaluated-model \
  --model-path /absolute/path/to/model --label evaluation \
  --eval-batch-size 1 --check
```

Remove `--check` to evaluate; change the benchmark name to select another task.
`--model-path` optionally verifies a local model/checkpoint against the served identity.
`--api-url` uses a loopback `/v1` endpoint; remote services can be reached through local forwarding.

- `--dry-run`: validate input selection and print the protocol without generating answers.
- `--check`: check evaluation dependencies, service identity and the selected grading backend.
- `--limit N`: run a small sample. EvalScope applies the limit per subject/subset, and the result is labeled smoke.
- `--max-tokens`: explicitly override the output budget.
- `--output-dir`: select a new output directory; existing directories are rejected.

LiveCodeBench requires an explicitly prepared grading backend, configured through `--sandbox-config`.
The image build entrypoint is [ops/build_capability_eval.sh](../../ops/build_capability_eval.sh).
The process backend enforces time and resource limits but does not provide complete filesystem or network isolation.
Missing dependencies, missing valid grading results or infrastructure failures cause evaluation to fail.

## Outputs and comparison

`config.json` records the execution protocol and source identities, `status.json` records completion, and `evalscope/` retains samples and raw reports.
Outputs default to `${ARTIFACT_ROOT}/outputs/capability_eval/`.

W&B records scores, sample counts, output budgets and the mean output-token count of the final cached answers.
Configure `WANDB_PROJECT`, `WANDB_ENTITY` and required authentication before running. `--check` and `--dry-run` do not create W&B runs.

```bash
.venvs/capability-eval/bin/python experiments/capability_eval/summarize.py \
  --initial-run /absolute/path/to/initial-evaluation \
  --trained-run /absolute/path/to/trained-evaluation \
  --checkpoint /absolute/path/to/run/global_step_100
```

Forgetting is the initial score minus the post-training score, measured in percentage points.
The summarizer verifies full sample denominators, model/checkpoint identities, data versions and generation protocols. Smoke runs, missing measurements and inconsistent runs are excluded from formal comparisons.
