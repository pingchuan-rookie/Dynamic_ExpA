# Main experiment entrypoints

This directory lists experiment matrices, binds weights and aggregates results. See [matrix.yaml](matrix.yaml).
Training and environment evaluation use shared entrypoints; capability evaluation uses [capability_eval](../../capability_eval/README.md).

## Matrix

| Suite | Models | Separate training sources | Evaluation per checkpoint |
|---|---|---|---|
| `transfer` | Qwen3.5-4B / 9B / 27B | DIVE、CodeGym | ALFWorld、t2bench、WebShop、SWE-bench Verified（D-OOD） |
| `retention` | Qwen3.5-4B / 9B | ALFWorld, WebShop | Same-domain tasks plus MMLU-Pro, HMMT February 2026 and LiveCodeBench v6 |

Each training seed has 40 Agentic RL combinations across `grpo`, `gigpo`, `dyad-grpo` and `dyad-gigpo`.
`grpo` maps to public method `grpo_react`; other names stay unchanged.
Both Dyad algorithms share the alignment projector binding for the same model.
Initial-model evaluation is independent of training seed/source, with 12 transfer slots and 6 retention capability slots.
Environment targets are all 134 ALFWorld valid_unseen tasks, a fixed 100 of 500 WebShop test tasks, t2bench, and a fixed 50 of 500 SWE-bench Verified test tasks.
WebShop/SWE subsets are sampled without replacement after sorting IDs, using independent `random.Random(42)` instances, then sorted again. All models/methods share them.
DIVE and CodeGym are training sources only in this matrix. Preserve original data and historical full-set results without relabeling them as current subset results.
The 3 initial SWE slots and all trained SWE slots remain `blocked` pending complete offline assets, integration and verifiable official grading.
Unblocking requires actual acceptance evidence; configuration files or a few passing tests are insufficient.

## List the plan

Run from `dynamic-expa/` with the project Python:

```bash
.venvs/expa-verl/bin/python experiments/shared/main_experiments/run.py --kind train
.venvs/expa-verl/bin/python experiments/shared/main_experiments/run.py --suite transfer --model Qwen3.5-9B --json
.venvs/expa-verl/bin/python experiments/shared/main_experiments/run.py --seeds 42 43 44 --kind train
```

By default, list only seed 42 without starting processes, Ray, GPUs or external services.
`planned` means required command bindings exist, not that data, dependencies, deployment or GPU capacity have passed validation.
Public configuration determines data, budgets and splits. This entrypoint does not silently adopt all hyperparameters suggested in notes.
Training seeds flow to task sampling, rollout and actor mini-batch configuration; this does not guarantee determinism of external environments or independent simulators.

## Bind exact inputs

Store the JSON binding file at a chosen workspace location; keep local-path run materials out of Git.
Use absolute paths and exact `global_step_N` checkpoints, without wildcards or latest aliases.

```json
{
  "projectors": {
    "Qwen3.5-4B": "/absolute/alignment/run/projector.pt"
  },
  "initial_models": {
    "Qwen3.5-4B": "/absolute/huggingface/snapshots/revision"
  },
  "checkpoints": {
    "transfer/Qwen3.5-4B/codegym/gigpo/seed-42/train": "/absolute/codegym/gigpo/run/global_step_300"
  }
}
```

`projectors` binds Dyad training inputs; `initial_models` binds initial HF policies for capability evaluation. Zero-shot environment evaluation uses public model resolution.
`checkpoints` keys are training job IDs. All target evaluations of a job share its weights, including retention task and capability measurements.
Bindings validate training source, saved method and model; ALFWorld checkpoints cannot stand in for CodeGym-trained weights.
Also check original training configuration to establish that different seed bindings actually came from different seeds.

## Check or execute one selection

```bash
.venvs/expa-verl/bin/python experiments/shared/main_experiments/run.py \
  --id transfer/Qwen3.5-4B/codegym/gigpo/seed-42/eval/alfworld \
  --bindings /absolute/bindings.json --check

.venvs/expa-verl/bin/python experiments/shared/main_experiments/run.py \
  --id transfer/Qwen3.5-4B/codegym/gigpo/seed-42/eval/alfworld \
  --bindings /absolute/bindings.json
```

Without `--check` or `--execute`, only display the command.
`--execute` runs one explicitly selected job, never the entire matrix by default. Confirm resource and service authorization before use.
`--check` delegates to public preflight. Capability preflight may inspect GPUs, dependencies and the Docker checker; it is not purely static.
See the [capability guide](../../capability_eval/README.md) for service and execution commands.
LiveCodeBench requires `--sandbox-config`; generated code is not executed on the host.
`--hardware` applies to environment entrypoints; capability services use `--tensor-parallel-size` separately.

## Aggregate tables from artifacts

Map evaluation job IDs to actual output directories in a separate JSON result index.
Do not provide manually copied scores, training/checkpoint directories or files selected automatically by timestamps.

```json
{
  "transfer/Qwen3.5-4B/initial/alfworld": "/absolute/outputs/initial-alfworld",
  "retention/Qwen3.5-4B/initial/mmlu_pro": "/absolute/outputs/initial-mmlu",
  "retention/Qwen3.5-4B/alfworld/grpo/seed-42/capability/mmlu_pro": "/absolute/outputs/trained-mmlu"
}
```

```bash
.venvs/expa-verl/bin/python experiments/shared/main_experiments/summarize.py \
  --runs /absolute/result-index.json --bindings /absolute/bindings.json
```

Default output is Markdown on stdout. `--format json` retains raw scores, derived forgetting, protocols, provenance and rejection reasons.
Missing measurements stay `-`. Training seeds are reported separately rather than averaged or merged silently.
Zero-shot is read once and displayed in both groups; forgetting reports percentage-point differences in ALF/WS order.
Invalid supplied results are explained and cause a nonzero exit. Unprovided measurements are not run failures.
Environment comparisons require matching model, method, exact checkpoint, data and protocol. Conflicting protocols reject the entire comparison group.
Historical outputs without complete task identity, protocol or verifiable source data are rejected. Exit code zero or manually added metadata does not establish comparability.
Capability pairing also verifies initial models, training sources and export provenance; preserve the required original metadata.

See the [public entrypoint guide](../train_eval/README.md) for environment protocols, checkpoint support and capacity limits.
See the [capability guide](../../capability_eval/README.md) for scoring and forgetting comparisons.

## SWE-bench Verified table acceptance

D-OOD uses `100 × resolved / 50` for one independent attempt per task, not pass@k or a full 500-task score.
Verify all 500 official source instance IDs, reconstruct the fixed 50-task subset and check complete unique-trial coverage. Row counts or self-reported planned counts are insufficient.
Each record requires `metric_valid=true`, `official_scored=true` and a traceable original harness report.
Official test failures count as unresolved. Missing tasks, debug runs, absent resources or incomplete grading invalidate the result; do not fill zeros or remove denominator entries.
Verify dataset revision, harness commit, architecture, task image identities, scaffold version, offline evidence and full checkpoint identity.
Dyad must restore policy, encoder/projector and the action mechanism. A plain HF policy export is insufficient for transfer evaluation.
Preparation may download resources; agent inference, interaction and grading prohibit public networking. Isolate task and grading containers separately and record driver/model isolation evidence.
Baseline and Dyad share prompts, Bash/Search/Editor/Finish tools, context/output/interaction budgets, temperature, seed and timeouts. Disable Qwen3.5 thinking.
SWE is evaluation-only, without target-based tuning or checkpoint selection. Small functionality checks are not table results.
See [matrix.yaml](matrix.yaml) and the environment guide for selection and scoring settings.
`blocked_targets` prevents both dispatch and table acceptance of SWE slots. Remove the block explicitly after acceptance; manually supplied scores do not bypass it.
The standalone reader verifies configuration against commands, fixed task identity, complete source manifests, 50 selected image identities, task digests/patches, official reports/logs/predictions, container inspections and driver/worker offline evidence.
The shared restoration verifier rechecks native Dyad policy, projector and frozen encoder files. Scoring requires all evidence, including every task report.
Each Dyad task needs an `action_audit` matching worker `step_count`, replaying sampled tokens/action contexts and checking `tool_mask`, allowed IDs and selections per turn.
Mechanism acceptance requires at least one real expanded-head selection across the run, not one in every task.
A task with no head usage due to refusal or malformed output may be a valid model failure. Keep it in the fixed 50-task denominator when officially graded.
Native restoration or official complete/unresolved status cannot satisfy mechanism acceptance if the head was never used anywhere in the run.
Worker preflight and per-task Ray/Python versions must match the producer's `expected_ray_runtime`. Version identity enters the protocol; optional audit fields do not replace required evidence.
Baseline/zero-shot remain blocked where producers lack complete base-HF/native-text weight identity. Model paths and exit codes alone cannot establish formal scores.
Synthetic fixed-50 archive regressions validate acceptance/rejection logic, not real task, model or image readiness.

## Scope

`run.py` lists plans by default; training and environment evaluation require one explicit job selection.
Use [evaluate.py](../../capability_eval/evaluate.py) for capability inference services, then add its output directory to the result index.
Inspect `blocked_targets` before execution; it constrains dispatch and formal result acceptance.
