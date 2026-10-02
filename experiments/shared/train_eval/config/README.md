# Agentic RL experiment and training configuration

This directory maintains shared environment, model and hardware configuration.
Dyad main recipes live in `experiments/dyad_training/agentic_rl/train_eval/config/experiments.yaml`; ablations in `experiments/dyad_training/agentic_rl/train_eval/ablation/*.yaml`. Baselines do not read them.
Benchmark YAML files store defaults, algorithm differences, model/hardware overrides and debug settings. GSM8K configuration is retained for evaluation only.
Supported action spaces, schemas and data directories are listed below.
`../scripts/prepare.py` owns parsing and validation; this directory contains no execution scripts.
Public methods `dyad-grpo` and `dyad-gigpo` share the `dyad` family model, hardware and encoder configuration.
They select GRPO and GiGPO advantage estimation respectively. `dyad` remains a GRPO compatibility alias; arbitrary Hydra overrides cannot masquerade as another method.
`base` names an action schema, not an optimizer. Selection and artifact classification must match saved configuration.

## Selection

Run from `dynamic-expa/`:

```bash
bash experiments/shared/train_eval/train.sh alfworld dyad-grpo \
  --model Qwen3.5-4B --hardware h100 --check
bash experiments/shared/train_eval/train.sh alfworld dyad-grpo \
  --model Qwen3.5-2B --hardware h200 --debug
bash experiments/shared/train_eval/evaluate.sh gsm8k grpo_react \
  --model Qwen3.5-4B --hardware h100 --dry-run
```

`MODEL_NAME` and `HARDWARE_PROFILE` may also be environment variables; CLI options take priority.
Without an explicit hardware selection, GPU names distinguish A6000, H100 and H200. Custom profiles such as `h100_4gpu`
require `--hardware`.
Legacy `SCALE_PROFILE` remains supported; `debug` selects the A6000 validation profile.
Model names are case-insensitive and accept the `Qwen/` prefix.
`MODEL_PATH` selects a local weight directory.
Without it, the shared Python launcher searches snapshots under `HF_HUB_DIR`, `HF_HUB_CACHE`, then `${HF_HOME:-~/.cache/huggingface}/hub`. Offline containers can mount the hub read-only and set `HF_HUB_CACHE`.
`--check` validates models, combinations, parameters and output directories without loading models or starting Ray.
`--dry-run` also prepares data and displays the final verl command.

## File structure

Benchmark files use these top-level fields:

- `defaults`: parameters shared by all algorithms.
- `algorithms`: `dyad`, `grpo_react` and `gigpo` defaults; an empty `gigpo` block inherits GRPO-ReAct.
- `models`: model names, each with `defaults` and `hardware`.
- `debug`: optional `defaults`, `algorithms` and `experiments` overrides.
- `step_profile`: shared step semantics and [per-step budgets](#per-step-budgets).

Each `models.<model>.hardware.<profile>` can contain:

- `defaults`: shared overrides for that combination.
- `algorithms.<algorithm>`: algorithm-specific overrides.
- `experiments.<experiment>`: ablation runtime overrides.
- `note`: provenance or unverified limitations.

Unused combinations can be omitted. Missing model/hardware combinations fail before startup instead of falling back to another model.

ALFWorld/CodeGym Qwen3.5-2B GRPO-ReAct evaluation supports `--hardware a6000_eval`; this does not establish training capacity.
GSM8K Qwen3.5-2B `a6000` retains three algorithm resource profiles for evaluation and local regression only.
Full-context capacity requires separate validation.
H100/H200 values were migrated from the former shared profiles and can now be adjusted independently.
Qwen3.5 cluster settings were migrated
from existing templates; this does not establish that every model runs. No A6000 capacity settings were invented for new models.
YAML `&name` defines an anchor and `*name` reuses it. Expand the reference into an override when tuning independently.

## Override order

Later values override matching earlier keys; other keys are inherited:

```text
benchmark.defaults
  → benchmark.algorithms[algorithm]
  → models[model].defaults
  → models[model].hardware[profile].defaults
  → models[model].hardware[profile].algorithms[algorithm]
  → ablation/<experiment>.yaml parameters
  → models[model].hardware[profile].experiments[experiment]
  → debug.defaults → debug.algorithms[algorithm] → debug.experiments[experiment]
  → step_profile → debug.step_profile (debug only)
  → explicit environment variables
  → command-line Hydra overrides
```

For `gigpo`, each algorithm layer merges `grpo_react` first, then `gigpo`, including benchmark, hardware and debug layers.
No extra hardware block is needed without GiGPO differences; the shared training budget stays fixed.
GiGPO-specific defaults live in `prepare.py` as `GIGPO_DEFAULTS`, without duplication across YAML files.

| YAML parameter | Environment variable | Hydra parameter | Default |
|---|---|---|---|
| `gigpo_gamma` | `GIGPO_GAMMA` | `algorithm.gamma` | `0.95` |
| `gigpo_mode` | `GIGPO_MODE` | `algorithm.gigpo.mode` | `mean_std_norm` |
| `gigpo_step_advantage_w` | `GIGPO_STEP_ADVANTAGE_W` | `algorithm.gigpo.step_advantage_w` | `1.0` |

`gigpo_mode` supports only `mean_std_norm` and `mean_norm`; gamma is in `[0,1]`, and step weights are finite and nonnegative.
Select GiGPO through `train.sh <env> gigpo`, not by overriding GRPO's `algorithm.adv_estimator`.

Dyad `config/experiments.yaml` stores model/training conditions under `main.defaults`. Ablation `overrides`
contain only changed conditions, with `benchmarks` declaring applicability.
Use `parameters` for ablation runtime settings.
Parameter names are lowercase; most map to uppercase environment variables.
`learning_rate` maps to `LR`;
`trainer_gpus` to `N_GPUS_PER_NODE`, and `encoder_gpus` to `DYAD_ENCODER_NUM_GPUS`.
Values may be integers, booleans, floats or strings. Unknown parameters and duplicate YAML keys fail validation.
`prepare.py` lists supported runtime keys in `SCALE_KEYS`; new keys must affect execution, not merely appear in YAML.

CodeGym Dyad training/evaluation use `MAX_ASSISTANT_TURNS` as the generation limit, falling back to resolved `MAX_TOOL_TURNS` when absent or empty.
`MAX_TOOL_TURNS` still controls the original user-turn guard; independent environment execution budgets remain unchanged.
An explicit assistant limit may differ from the user limit. A looser assistant limit can still encounter the post-generation user guard before execution.

Debug reduces run size while retaining the selected model; explicit environment variables still take priority.
Evaluation enforces evaluation-only execution without training checkpoint saves; output paths remain constrained to project outputs/ckpt roots.
Each run saves selection, merged parameters and final Hydra arguments in `outputs/.../<run>/resolved_config.json`.
`command.json` stores the complete execution command.
The inherited environment is not dumped into artifacts.

## Validation

Use training/evaluation `--check` to validate configuration, data selection and argument conflicts without training.
Developer regressions are managed separately from distributed runtime source.

## Action spaces and data

`base` uses native environment command text.
`open` generates free-text argument values; ALFWorld `closed` uses fixed value slots selected by the action head.

| Benchmark | surface_form | values | Action schema | Data subdirectory | Extra settings |
|---|---|---|---|---|---|
| GSM8K | base | open | `gsm8k/base.yaml` | `dataset` | None; evaluation only |
| ALFWorld | base | open | `alfworld/base.yaml` | `dataset` | None |
| ALFWorld | base | closed | `alfworld/base_closed.yaml` | `dataset` | None |
| CodeGym | base | open | Generated per task | `dataset` | `TEMPLATE_STYLE=base`, all environments |
| WebShop | base | open | `webshop/base.yaml` | `dataset` | Official full human tasks; train/dev for training, test for evaluation |

Schemas live under `agent_system/policies/dyad/actions/schemas/`; data under `data/<benchmark>/<data_subdirectory>/`.
Evaluation-only GSM8K uses `data/gsm8k/<data_subdirectory>/`.
CodeGym uses shared data and per-task schemas.
Preparation code validates these combinations directly, without reading Markdown.

`WEBSHOP_MAX_STEPS` controls environment steps. `MAX_ASSISTANT_TURNS`/`MAX_TOOL_TURNS` control rollout generation/tool turns; token and feedback-character budgets are separate.
`WEBSHOP_BACKEND_REPLICAS` sets full backend replicas per agent worker. `WEBSHOP_SESSIONS_PER_BACKEND` sets session slots over shared products, without duplicating the catalog per batch × rollout.
Model/hardware profiles are initial settings, not verified full-context capacity. Qwen3.5-0.8B has only a local validation profile and is excluded from cluster launch.

See [data preparation](../../dataset/README.md) for provenance, rebuilding and integrity checks.
ALFWorld, CodeGym and GSM8K use provenance-preserving `source_tasks_v1`; generators do not render local step prompts.

Within a benchmark, `defaults.train_batch_size`, `defaults.rollout_n`, `defaults.ppo_mini_batch_size` and `defaults.ppo_epochs` define the formal budget shared across models, hardware, algorithms and ablations.
Each optimizer update uses `ppo_mini_batch_size × rollout_n` trajectories; benchmarks may choose different values.
ALFWorld/CodeGym `debug.defaults` defines separate small-batch budgets and train/validation sample limits. Dyad debug saves checkpoints for short-run evaluation.
Limits select from complete original splits without creating replacements.
Every mode validates its budget; environment/Hydra overrides cannot bypass it. Adjust capacity through micro-batches, offload and parallelism.
CodeGym Dyad pads the epoch's final batch; other training loaders drop incomplete tails. Evaluation is exempt from training-budget checks.


`ppo_mini_batch_size` is separate from `train_batch_size`; both are protected by formal/debug budget validation.
The verl trainer multiplies `ppo_mini_batch_size` by `rollout_n` for global trajectories per update. Training micro-batches affect splitting and accumulation only.
Formal CodeGym training follows the [paper's batch definition](https://arxiv.org/html/2509.17325#S11.SS1), with values maintained only in `codegym.yaml`.
Other hyperparameters retain project settings; this is not a full-paper reproduction, and small-batch debug results do not represent the formal budget.

## Per-step budgets

Each decision is a separate sample. `max_response_length` limits tokens per decision, not accumulated trajectory generation.
`max_assistant_turns` limits decisions per trajectory through `algorithm.step_rollout.max_steps`.
`step_profile` merges after algorithm, hardware and debug layers so all four methods and their debug runs share step budgets.

| Benchmark | Response tokens per step | Maximum decisions | Notes |
|---|---|---|---|
| ALFWorld | 512 | 50 | |
| CodeGym | 386 | 80 | Project setting, not the paper's 24576 response budget |
| DIVE | 1024 | 30 | Provisional; calibrate using the length records below |
| WebShop | 512 | 15 | |

ALFWorld, CodeGym and WebShop `max_tool_turns` equals the decision limit; DIVE does not use it.
Prompt budgets retain their per-file settings; DIVE uses 32768 for native templates.
Configure these values through experiment entrypoints; runs save their resolved values.
Resume/evaluation restores the checkpoint's saved step protocol and budgets rather than adopting this document's values.
Local developer checks retain their own fixed 1024-token, 8-turn budget.

Training `response_length/clip_ratio` uses the configured per-step budget, not the batch's padded maximum response width.
`prompt_length/clip_ratio` still uses the longest prompt in the batch.
Regular step validation also reports `val-aux/response_length/{mean,max,min,clip_ratio}`; official tau/SWE paths do not.
Per-decision `environment_step` records include `response_length`, `response_budget` and `response_clipped`, saved in rollout/validation dumps.
`response_budget` is the decision's actual limit; reaching it counts as clipping.
