# agent_system/environments - Local environment pools for training and evaluation

ALFWorld, CodeGym, GSM8K-calc and WebShop integrate through the tools and pools here; no HTTP environment service needs to be started beforehand.
ALFWorld, CodeGym and the calculator execute in Ray workers. WebShop workers manage JSONL subprocesses using an isolated interpreter.
The old HTTP implementations were removed to avoid batch-wide failures when an external environment service exits.

Single-environment execution lives in the Git-managed [`env_package/`](env_package/README.md), with `envs.py` and `build_env()` per environment.
`backends/` contains Ray pools, workers, tools, adapters, datasets and training-side integration; `backends/tau/` connects to `env_package/t2bench/`.
Dependency preparation is in `ops/env_deps/`, and standalone reference evaluation is in `experiments/shared/train_eval/reference/`.
GSM8K/calc remains available for quick debugging and evaluation even though it is outside the formal training matrix.

Environment-side model configuration, authentication and requests belong to the corresponding `env_package/` implementation:
DIVE uses [`env_package/dive/model_api/`](env_package/dive/model_api/README.md), while
t2bench uses [`env_package/t2bench/trapi.py`](env_package/t2bench/trapi.py) for user-simulator and judge requests.

| Component | Responsibility |
|---|---|
| `core/` | Shared pool, local/native tool, adapter, generic adapter, session and worker base classes |
| `env_package/` | Git-managed environment objects, tools, state and rewards |
| `backends/alfworld/` | TextWorld pools, tools, adapters and lightweight workers; the session module describes the external engine boundary |
| `backends/codegym/` | Function-calling environment pools, tools, adapters and dynamic task workers |
| `backends/calc/` | GSM8K calculator; the execution implementation is `backends/calc/session.py` |
| `backends/webshop/` | Bounded backend pools, tools, adapters, portable configuration, lightweight workers and an isolated interpreter backend |
| `backends/dive/` | CPU Ray workers, tools, pools and task datasets; official tools and terminal judging run in an isolated environment |
| `backends/tau/` / `backends/swebench/` | Evaluation pools, tools, workers, datasets and metrics, with their existing adapters/sessions |
| `registry.py` / `step_session.py` | Shared adapter registration and stepwise sessions |
| `prompts/` | Environment templates and shared response protocols used by text baselines and Dyad |
| `configs/` | YAML/JSON package resources at their established paths; tool configs point to each environment's `tool.py` |

Production pools use Ray actors; this organization does not add a `backend=inproc` mode.
CPU smoke tests can drive real workers directly, while distinguishing controlled transport from production Ray pools.
The shared ALFWorld worker explicitly constructs a TextWorld environment without `AlfredExpert`, matching the verl-agent policy environment.
The installed `alfworld==0.4.2` dagger/train path enables the expert by default; setting `use_expert=False` alone is insufficient.
Training, validation, text baselines and Dyad all use the same implementation to disable the expert while retaining observations, legal actions, success rewards, randomization and configured step limits.
Dyad's ALFWorld schema includes the native `use` action; lighting tasks require it and cannot substitute `toggle`.

ALFWorld defaults to `pool_size: null` and reserves `0.1` Ray CPU per actor.
Before trajectories begin, the TQ scheduler prepares enough environments for the current worker's shard and rollouts per question.
For a global `16 × 8` batch split between two workers, each prepares 64 environments; one worker handling 140 validation trajectories prepares 140.
Training and validation share pools with the same configuration. Capacity only grows, and existing sessions retain their leases during expansion.
`ALFWORLD_ENV_POOL_SIZE` explicitly sets capacity per worker and must cover its concurrent tasks.
Insufficient CPUs fail explicitly instead of silently reducing 128 environments to 92; startup contention remains bounded by health-check timeouts.
This preserves independent environment parallelism, asynchronous trajectory progression and training/validation reuse.

ALFWorld's default reset, step, close and lease-wait timeouts are 120, 60, 30 and 300 seconds.
Override them with `reset_timeout_s`, `step_timeout_s`, `close_timeout_s` and `lease_timeout_s` in the tool configuration.
Infrastructure failures invalidate the trajectory rather than producing a valid zero reward; actors with unknown state are not returned to the pool.
`BaseEnvPool` likewise rejects silent capacity reduction and supports cancellation of queued session creation, duplicate-ID checks and bounded lease waits.
CodeGym also sizes pools from the actual shard; `CODEGYM_ENV_POOL_SIZE` is an explicit per-worker capacity and must cover concurrency.
CodeGym reserves 0.25 Ray CPU per actor and isolates each action transactionally. Startup health checks must find a real task.
CodeGym, DIVE, SWE and t2bench default to a 3600-second lease-wait limit, configurable with `lease_timeout_s`.
CodeGym also accepts `CODEGYM_LEASE_TIMEOUT_S`. SWE and t2bench remain evaluation-only.
Native session tools clean up only sessions they successfully created, so a rejected duplicate ID cannot interrupt another caller.
t2bench infrastructure failures remain invalid evaluations and are not converted to zero scores.

WebShop uses a separate bounded Ray pool rather than copying the full catalog for every trajectory.
Each backend replica owns a full catalog and index. Its sessions share these read-only resources while retaining independent browsing state.
The pool is shared within each agent-loop process; total replica count also depends on the number of agent workers.
Shutdown cancels and awaits unfinished session creation. Queued RPCs recheck backend state after acquiring the lock to avoid using a closed catalog.

## Environment prompts

[prompts/](prompts/) follows [verl-agent](https://github.com/langfengQ/verl-agent) in separating templates from execution, action parsing and rollout scheduling.
[alfworld.py](prompts/alfworld.py) and [webshop.py](prompts/webshop.py) maintain templates with and without history, plus observation, legal-action and bounded-history formatting.
Local templates preserve the upstream ReAct structure but introduce generic household-interaction or online-shopping scenarios without announcing benchmark names. They are not claimed to match upstream text byte for byte.
Qwen3.5's `enable_thinking=False` controls only native template mode; task-level `<think>...</think>` requirements and validation remain unchanged.
Text-action environments require `<action>...</action>` after reasoning. DIVE and SWE-bench use native tool templates, followed directly by tool calls or a final answer (`environment_react_v4`).
Text baselines and Dyad use the same builders, without GRPO/GiGPO- or action-interface-specific prompts.
Importing templates does not load Ray or torch or register agent loops.

[Dataset preparation](../../experiments/shared/dataset/) and runtime execution share these prompts. ALFWorld/WebShop rebuild stepwise context after the actual reset.
DIVE/CodeGym also use parameterized templates with and without history, populated with public tasks, current observations, available tools and bounded action history while retaining native action formats.
DIVE's full public trajectory remains available for GiGPO state anchors; it does not bypass the prompt history window as the current observation.
This directory does not store hidden answers, judge prompts or Alignment data-synthesis prompts.

## DIVE

DIVE reuses GPU policy rollout; tools and judges execute in CPU Ray actors with dependencies isolated in `.venvs/dive`.
`planned_episodes` records dataset provenance and does not select an official evaluator. t2bench/SWE datasets declare their official result protocol through
`episode_field`; DIVE uses generic step validation and its own terminal reward.
Pinned execution source is in `agent_system/environments/env_package/dive/source/`, with compatibility integration in `agent_system/environments/env_package/dive/`. Runtime execution does not run task synthesis.
Each active trajectory holds an exclusive session lease. Tools execute in model-message order; timed-out actors or actors with unknown state must be destroyed rather than reused.
Ray actors are not security sandboxes. Code execution requires an explicitly provisioned SandboxFusion service; the adapter does not create or delete Docker containers.
Tool parameters retain the published data schemas. Dyad extends only tool-name selection and does not introduce a synthetic `respond` action.
Final natural-language answers use upstream verifier labels: `correct=1`, `partial/incorrect=0`. Infrastructure failures must not masquerade as valid zero rewards.
GiGPO state anchors contain the full public history and a tool-schema summary, never the reference answer. One assistant generation counts as one decision even if it contains multiple tool calls.
See [dataset preparation](../../experiments/shared/dataset/README.md#dive-official-tasks) for data selection/exclusions and the [training entrypoint](../../experiments/shared/train_eval/README.md#dive) for service configuration.

## WebShop

Official execution source is bundled in `agent_system/environments/env_package/webshop/source/`; integration preserves its execution and template semantics.
Training uses `.venvs/expa-verl`; environment dependencies use `.venvs/webshop`. Deployment paths are selected through `WEBSHOP_PYTHON_BIN`, `WEBSHOP_REPO`, `WEBSHOP_ASSETS` and `WEBSHOP_JAVA_HOME`.
See [dataset preparation](../../experiments/shared/dataset/README.md#webshop-full-human-tasks) for complete assets and data preparation.
Queries and clicks use native `search[query]` and `click[target]`. The model sees public page content, without hidden goals or reference queries.
Purchase scores remain in `0..1`; full success requires termination with score 1.
Environment exceptions fail rollout rather than producing valid zero-score purchases. Finalizing a session does not repeat a purchase, and releasing it does not destroy other browsing sessions.

Public class exports remain lazy; importing environment packages, workers or prompts does not load Ray, verl or torch.
`backends/webshop/backend.py` supports both direct execution by an isolated interpreter and `python -m agent_system.environments.backends.webshop.backend`.
[compat.py](../compat.py) resolves old `dyad.environments.*` and pre-migration `agent_system.environments.*` configuration paths at load time, without retaining an old re-export tree or rewriting historical checkpoints.

Environment argument builders in `experiments/shared/train_eval/scripts/training.sh` consume the shared configuration used by training and benchmark evaluation.
Public entrypoints are `train.sh` and `evaluate.sh`; training selects ablations with `--experiment`.
See [Agentic RL usage](../../experiments/shared/train_eval/README.md) for the full file flow.

Environment `adapter.py` modules normalize environment interfaces. They are separate from the action projector in `agent_system/policies/dyad/models/action_projector.py`.
