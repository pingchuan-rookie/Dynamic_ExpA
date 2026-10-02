# Agentic RL training and evaluation

This directory provides shared configuration and evaluation entrypoints. Method-specific training entrypoints use the same environments, data and training extensions;
evaluation generates actions through the [standalone inference API](../../../agent_system/inference/README.md), then executes and scores them in environment sessions.
Run all commands below from the repository root.

## Methods and environments

| Method | Action interface | Advantage estimator | Training entrypoint |
|---|---|---|---|
| `grpo_react` | Text ReAct | GRPO | `experiments/grpo_training/train_eval/train.sh` |
| `gigpo` | Text ReAct | GiGPO | `experiments/gigpo_training/train_eval/train.sh` |
| `dyad-grpo` | Dyad | GRPO | `experiments/dyad_training/agentic_rl/train_eval/train.sh` |
| `dyad-gigpo` | Dyad | GiGPO | `experiments/dyad_training/agentic_rl/train_eval/train.sh` |

Training supports `alfworld`, `codegym`, `webshop` and `dive`.
Standalone evaluation additionally supports `gsm8k`, `t2bench` and `swebench_verified`.
Models, hardware and task budgets are defined in [config/](config/README.md).

Complete [data preparation](../dataset/README.md) and [environment installation](../../../agent_system/environments/README.md) first.
A new Dyad training run also needs a matching [Alignment projector](../../dyad_training/action_encoder_alignment/train_eval/README.md).

## Training

```bash
# Text GRPO
bash experiments/grpo_training/train_eval/train.sh codegym \
  --model Qwen3.5-4B --hardware h100 --check

# Text GiGPO
bash experiments/gigpo_training/train_eval/train.sh codegym \
  --model Qwen3.5-4B --hardware h100 --check

# Dyad-GRPO
bash experiments/dyad_training/agentic_rl/train_eval/train.sh codegym dyad-grpo \
  --model Qwen3.5-4B --hardware h100 \
  --projector-init /absolute/path/to/projector.pt --check
```

Replace `dyad-grpo` with `dyad-gigpo` to use GiGPO. After checks pass, remove `--check` to start training.
`MODEL_PATH` can specify a model directory; `--model` and `--hardware` select configuration entries.
`--dry-run` prepares required inputs and prints the final command; `--debug` uses the configured small-run settings.
`--debug` still permits only the four training environments and uses their configured budgets. The dedicated `--local-debug-smoke` entrypoint and fixed test budgets have been removed.
Hardware profiles are starting configurations; usable context lengths and batch sizes depend on the target device.

The main Dyad configuration uses `joint_optimization` and `projector_and_encoder_lm`, training the policy LM,
independent encoder backbone and projector. Alignment trains only the projector.
See [ablation configuration](../../dyad_training/agentic_rl/train_eval/ablation/README.md) for frozen-policy and other research conditions.

## Standalone evaluation

```bash
bash experiments/shared/train_eval/evaluate.sh alfworld \
  --checkpoint /absolute/path/to/run/global_step_100 --check

bash experiments/shared/train_eval/evaluate.sh alfworld \
  --checkpoint /absolute/path/to/run/global_step_100
```

Evaluation restores model and training-method identity from the checkpoint's `model_config.json`;
use `--model-config` when an explicit configuration file is needed. A specific checkpoint is required; evaluation does not infer or select the latest weights.

Reuse an existing inference service with:

```bash
bash experiments/shared/train_eval/evaluate.sh alfworld \
  --checkpoint /absolute/path/to/run/global_step_100 \
  --api-url http://127.0.0.1:8000/v1 --served-model evaluated-model
```

Without an explicit service, the evaluator starts and cleans up its own vLLM/Dyad service.
Full Dyad evaluation restores the policy, projector and encoder and builds action candidates from the target task definitions.
Inference TP is independent of training FSDP shard count. Environment sessions use Ray pools; the inference service manages the policy and encoder.
`EVAL_CONCURRENCY`, `EVAL_TENSOR_PARALLEL_SIZE` and `EVAL_ENCODER_DEVICE` control concurrency, TP and the encoding device.
Standalone evaluation accepts only generation/evaluation options and does not start a training optimizer.

See [capability_eval](../../capability_eval/README.md) for single-turn language-capability evaluation.

## Environment configuration

### ALFWorld and WebShop

Both use shared templates for tasks, observations, legal actions and bounded history.
ALFWorld evaluates `valid_unseen`; data preparation produces WebShop's fixed evaluation subset.
Compared methods must use the same task set. Task counts or configuration names cannot replace provenance checks.

Training uses shared reward profiles: ALFWorld environment rewards are multiplied by ten; WebShop gives 10 when the terminal task score is 1
and 0 otherwise. WebShop evaluation also retains the original partial score.

### CodeGym

CodeGym uses the global task source, deriving each task's legal actions from its environment source.
Fixed action capacity is used only for batch padding and does not add other tasks' actions as candidates.
Overlong prompts fail explicitly. Debug runs may explicitly limit samples, batch size or steps.

Actions execute against an environment snapshot; normal returns commit the new state. Exceptions or timeouts return the original state and error feedback.
Existing parsers and environments handle JSON/argument errors without repairing generated actions.
See [data documentation](../dataset/README.md) and the environment implementation for tool timeouts and data formats.

### DIVE

DIVE uses native tool calls and official task scoring. Installation and service settings are documented in
the [DIVE environment package](../../../agent_system/environments/env_package/dive/README.md).
Explicitly configure the judge provider, model and API URL; these settings do not replace the policy being trained or evaluated.
Checkpoints retain judge identity across restoration. Missing valid judge results are infrastructure failures.

### t2bench

`t2bench` is evaluation-only. Its user and judge use separate model services, configured explicitly:

```bash
bash experiments/shared/train_eval/evaluate.sh t2bench grpo_react \
  --model Qwen3.5-4B --domain retail --debug \
  --user-provider openai --user-model USER_MODEL \
  --user-base-url http://127.0.0.1:8001/v1 \
  --judge-provider openai --judge-model JUDGE_MODEL \
  --judge-base-url http://127.0.0.1:8002/v1 --check
```

`--history-length` controls decision history; `--max-tokens` and `--max-prompt-length` control per-request budgets.
`--max-assistant-turns` limits policy decisions; the official environment also has `--max-steps`.
The environment package owns native task initialization, simulation and scoring rules.
See [reference/t2bench](reference/t2bench/README.md) when a standalone reference workflow is needed.

### SWE-bench Verified

SWE-bench is evaluation-only. The model operates on a task workspace through shared tools, and the pinned official harness scores the final result.
Prepare task resources, Docker and isolation settings as described in
the [SWE-bench environment package](../../../agent_system/environments/env_package/swebench/README.md).
`--mini-swe-config` configures mini-swe-agent tools. The baseline and Dyad use identical tool definitions and budgets.

## Shared training semantics

Training follows the verl V1 sync loop, with shared extensions in [verl_extensions](../../../verl_extensions/README.md).
Each environment decision constructs its own prompt/history; GRPO and GiGPO differ only in advantage estimation.
Text policies and Dyad share environment state and rewards. Dyad additionally records action candidates, actual samples and forced-serialization positions for probability replay.

- `ppo_mini_batch_size_unit=step` counts training occurrences; one rollout iteration may produce multiple optimizer updates.
- Statistical copies from `reference_copy` participate in advantages and loss; transport padding participates in neither.
- `reference_microbatch` uses the nominal global-mini denominator; an incomplete final mini-batch is not renormalized to a full one.
- Micro-batching and data-parallel topology may affect statistical replication and mean-of-means partitions; keep them fixed in comparisons.
- Qwen3.5 native thinking is disabled; prompts and parsers still define the environment's task-level reasoning/action protocol.

See [training extensions](../../../verl_extensions/README.md)
and [prompt documentation](../../../agent_system/environments/prompts/README.md) for advantages, masks, loss normalization and protocol checks.

## Weight restoration and configuration overrides

Full training restoration checks data order, loader configuration, task and numerical protocols, and optimizer state together.
Missing identity metadata or incompatible checkpoints are rejected. Weight-only evaluation and full training resumption are distinct entrypoint behaviors.
A trained encoder requires `actor/encoder_backbone.pt` in the checkpoint.

`--projector-init` selects an Alignment projector; `--checkpoint` selects a full trained checkpoint for evaluation.
Use explicit file or run-directory paths for all inputs.
See [configuration](config/README.md) for precedence among environment, algorithm, model, hardware, debug and explicit overrides.

## Outputs and diagnostics

`ARTIFACT_ROOT` is an absolute artifact root, defaulting to the repository's `artifacts/` directory.
`ckpt/` stores weights and restoration state; `outputs/` stores run configuration, metrics, trajectories and analysis.
Each run uses a separate directory, and evaluation does not rewrite its source checkpoint.

Training saves `model_config.json` and `resolved_config.json`. Standalone evaluation saves `evaluation_config.json`,
service/source identities and step-level decisions and scoring evidence under `val_generations/`.
Check completion status, actual interaction counts, score-validity flags and original reports; missing measurements are not zero scores.

`--debug` uses a small configuration, detailed diagnostics and offline W&B; normal training uses online W&B.
`GRPO_PRINT_STEP_ACTION` and `GRPO_PRINT_ENV_STEP` control stepwise console output,
`DYAD_DIAG_ENABLED` controls diagnostic files, and `POST_ANALYSIS` controls post-training analysis.
See [analysis](../analysis/README.md) for result readers and checks.
