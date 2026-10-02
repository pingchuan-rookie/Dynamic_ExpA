# Project extensions to verl

This directory implements training extensions shared by text baselines and Dyad. The upstream comparison is pinned to
[verl v0.9.0 / `483b8a009ba3a97563edee3a19887e4862b8094a`](https://github.com/verl-project/verl/tree/483b8a009ba3a97563edee3a19887e4862b8094a)。
This revision is the reference for source structure and interfaces.

Only runtime implementations and required configuration belong here. Test drivers, development checks and temporary debugging budgets are kept outside the package; runtime code does not import tests or experiment scripts.

`verl/` retains entrypoints, classes, registration interfaces, distributed decorators and the official GRPO loop. Explicit calls at the original integration points delegate to project implementations.
Replaced upstream code remains in `DYAD-REPLACE` comments; supported fallbacks remain executable.
This is a verl source distribution with project integration points, still installed as `verl`. The unmodified PyPI package cannot replace it.

## Following the upstream GRPO flow

| Upstream location | Project implementation | Responsibility |
|---|---|---|
| `main_ppo.run_ppo / TaskRunnerV1.run` | [`runtime/`](runtime/) | Ray environment forwarding, dataloader cleanup and tracking-service shutdown |
| `PPOTrainer.fit → step → _step_once` | Original loop retained in `verl/` | Preserve sampling, rewards, balancing, old/ref log-probabilities, advantages and updates in order |
| `_init_dataloader / _load_checkpoint` | [`trainer/stages.py`](trainer/stages.py), [`dataset/`](dataset/) | Separate evaluation from training data; validate actual data identity and protocol on restore |
| Before balancing in `_step_once` | [`agent_steps/batch.py`](agent_steps/batch.py) | Prepare real step occurrences; distinguish statistical replication from transport padding |
| `AgentLoopWorkerTQ` | [`rollout/`](rollout/) | Environment concurrency, sampling versions, action-trace transport and padding |
| `_compute_advantage` | [`trainer/stages.py`](trainer/stages.py), [`agent_steps/`](agent_steps/) | Shared GRPO/GiGPO grouping, masks and metrics |
| `_update_actor` | [`trainer/stages.py`](trainer/stages.py), [`agent_steps/minibatch.py`](agent_steps/minibatch.py) | Step mini-batch dispatch through the upstream actor update |
| FSDP `forward_backward_batch / prepare_model_outputs` | [`agent_steps/loss_reduction.py`](agent_steps/loss_reduction.py), [`vocab_statistics.py`](vocab_statistics.py) | Reference-micro-batch normalization and bounded vocabulary statistics |
| `_validate / _log_rollout_data` | [`trainer/stages.py`](trainer/stages.py) | Complete environment-task metrics and evaluation evidence; legacy integration is in `trainer/legacy.py` |
| `PPOTrainerSync.on_step_end → update_weights` | Upstream boundaries and Dyad worker/engine subclasses | Preserve synchronization order and the encoder-version protocol |

Environment sessions and policies remain in [`agent_system/`](../agent_system/README.md); action encoders, Dyad engines, workers and losses belong to `policies/dyad/`.
This package does not add a second environment loop or `fit` loop. Stage functions explicitly import upstream dependencies when called, preserving process-local registries, decorators and dependency-replacement boundaries while avoiding package-initialization cycles. It installs neither import hooks nor runtime monkey patches.

A shared call structure does not imply unchanged upstream numerical semantics. Text baselines and Dyad use the project step protocol, GRPO/GiGPO grouping and `reference_microbatch` normalization; they are not unmodified upstream GRPO.

## Configuration and source boundaries

`config/project/agentic_rl.yaml` composes through the upstream `ppo_trainer.yaml` Hydra defaults and search path.
Override order is upstream component defaults → shared project defaults → main configuration `_self_` → command-line overrides.
`config/defaults.py` supplies independent default factories for the thin extensions to upstream structured configurations.
Package initialization does not import Ray, models, environments or trainers. `pyproject.toml` declares the package and configuration resources; `setup_uv.sh` manages environment installation.

[`upstream.json`](upstream.json) retains baseline SHA-256 digests for 825 runtime-source and public-documentation files.
Its integration list contains 33 `verl/` files and the checkpoint converter whose embedded test branch was removed.
`comment_only_files` records the updated digests of 7 comment-cleaned files; `documentation_files` records English documentation updates. Original baseline hashes remain in `files`.
Development tests, smoke runs and diagnostic scripts are excluded from distributed source and the source inventory.
No project-specific modules are added under `verl/`.
Workflow coverage must be established from the test entrypoints actually supplied by the repository.

Project images are defined in [`ops/`](../ops/README.md); the DIVE sandbox is in `ops/dive_sandbox/`.
See [`env_package/`](../agent_system/environments/env_package/README.md) for environment integration and bundled dependency source.
The root README, packaging, dependencies and ignore configuration are project integration responsibilities. Original licenses and authorship are retained.

Retained framework fixes include response-budget metrics, empty-mask denominators, nested position IDs, scheduler advancement and worker-residency interfaces.
Tracking shutdown and TransferQueue field behavior remain owned by their respective implementations.

## Validation scope

See [contribution guidelines](../CONTRIBUTING.md) for public formatting checks. Local test suites manage development regressions, dependencies, resources and source-version correspondence; they are not distributed by Git clone or Python packages.
The main repository currently has no CI workflow executing private tests. Local success does not establish CI coverage.
GPU execution, model training, containers and cluster runs each require their own runtime evidence.
