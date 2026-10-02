# Supported experiments

Formal main experiments are specified in [main-experiments.md](../../../dynamic-expa-design/experiments/main-experiments.md), with ablations in [ablation.md](../../../dynamic-expa-design/experiments/ablation.md).
The legacy configuration notes below do not define the current formal experiment matrix. Current support is defined by the [public entrypoints](train_eval/README.md).
GSM8K remains under `data/gsm8k/` for evaluation only; old training examples are no longer supported. Training datasets are limited to DIVE, CodeGym, ALFWorld and WebShop.

## Main comparison

`bash experiments/shared/train_eval/train.sh <env> [algo] [--debug]`; omitting `algo` selects `dyad`.

| | `grpo_react` | `dyad` (main experiment) |
|---|---|---|
| **gsm8k** | ✅ | ✅ |
| **alfworld** | ✅ | ✅ |
| **codegym** | ✅ | ✅ |

The key difference is **whether actions are written or selected, and how the selection head is constructed.**
**

| | Action generation | Multi-turn / tools | action_head |
|---|---|---|---|
| `grpo_react` | Actions are **written as text** (`<Think>` / `<Action>`) and parsed into tool calls | Multi-turn + external environment | None |
| `dyad` | The head selects an action **name** from the candidate set; arguments remain text | Multi-turn + external environment | Each row is computed by the encoder and trained projector |

Both read the same `data/<env>/dataset/` (evaluation-only GSM8K uses `data/gsm8k/dataset/`). Run configuration selects the agent loop; data and prompts are shared across algorithms.

**The main `dyad` condition is `joint_optimization`: the action encoder and policy LLM backbone train together.**
**
Training only one side defines an ablation, not a second main-table condition:

| | Trainable components | Entrypoint |
|---|---|---|
| Main experiment | Both sides (`joint_optimization`) | `train.sh <env>` |
| Ablation 2.7 | Action encoder only; policy LLM backbone frozen | `train.sh <env> dyad --experiment 2.7_frozen_llm_adaptation` |
| Ablation 2.8 | Policy LLM backbone only; head frozen at Alignment weights | `train.sh <env> dyad --experiment 2.8_policy_lm_only` |

> Before 2026-09-01, the main comparison had two conditions (`frozen_llm_adaptation` and two-stage `encoder_then_policy_lm`).
> Trainable scope depended jointly on `DYAD_TRAINING_SCHEDULE` and `TRAIN_TARGET`.
> These settings could conflict, preventing training while metrics appeared normal.
> Trainable scope now has a single configuration dimension,
> enforced by the entrypoint's training-schedule checks.

## Run an experiment

```bash
cd Dynamic_ExpA

# Full profile.
bash experiments/shared/train_eval/train.sh gsm8k                      # Main experiment.
bash experiments/shared/train_eval/train.sh gsm8k grpo_react           # Baseline.

# Two-step smoke with automatic analysis, including token-id decoding.
bash experiments/shared/train_eval/train.sh alfworld grpo_react --debug
bash experiments/shared/train_eval/train.sh --debug gsm8k                # = train.sh gsm8k --debug
```

`--debug` selects a small configuration and detailed diagnostics; `prepare.py` sets W&B offline.
`run.py` automatically analyzes the results when the run ends.

## Ablations

Use `bash experiments/shared/train_eval/train.sh <env> dyad --experiment <name> [--debug]`.
See [`ablation/`](../dyad_training/agentic_rl/train_eval/ablation/README.md) for supported conditions and environments; each YAML contains that condition's changes.
Training-scope constraints are enforced by entrypoint configuration validation.

## Evaluate a model on benchmarks

Use `train_eval/evaluate.sh`, selecting the model source through its arguments:

```bash
bash experiments/shared/train_eval/evaluate.sh gsm8k grpo_react
bash experiments/shared/train_eval/evaluate.sh gsm8k dyad --projector-init PATH
bash experiments/shared/train_eval/evaluate.sh gsm8k dyad --checkpoint PATH
```

`PATH` identifies either a projector file/directory or a `global_step_N` directory.
Select the model through `MODEL_NAME` or
`MODEL_PATH`.
`--debug` selects the debug profile; `--check` validates choices and paths without starting Ray.

CodeGym Dyad requires `ENV_NAME` and its schema, selecting that environment's samples from the global test set.
CodeGym ReAct can evaluate the full test set or use `ENV_NAME`/`CODE_ID` to select a matching subset.
Algorithm comparisons must use the same sample range.

Sampling and batch settings are in `train_eval/scripts/prepare.py` and `train_eval/scripts/evaluation.sh`. Weight validation is centralized in
`train_eval/scripts/prepare.py`; evaluation does not delete source checkpoints.

## Model and hardware configuration

Scale and training parameters combine shared defaults, algorithm settings, model/hardware settings,
experiment overrides and debug overrides in `train_eval/config/<env>.yaml`; `scripts/prepare.py` resolves them.
Select with `--model` and `--hardware`; `--debug` reduces scale while retaining the model.
See [configuration](train_eval/config/README.md) for organization, precedence, supported combinations and validation commands.
Explicit environment exports on the cluster still take precedence over configuration files.

## Directory layout

See [train_eval/README.md](train_eval/README.md) for file organization and entrypoint flow.


`experiments/` does not store datasets or training artifacts.
Data is organized by purpose into eight top-level directories under `data/`; see [data responsibilities](dataset/README.md#data-directory-responsibilities). Artifacts use `outputs/<site>/<stage>/` and `ckpt/<site>/<stage>/`.
