# Environment implementations

`env_package/` contains independently constructible environment implementations tracked by Git.
Following verl-agent's `env_package` organization, each environment exposes `envs.py` and `build_env()`; outer layers integrate it with training and evaluation.
Directories use the environment identifiers `alfworld`, `codegym`, `dive`, `webshop`, `t2bench` and `swebench`, rather than grouping implementations by source repository.

| Package | Runtime responsibilities |
|---|---|
| `alfworld/` | Reset, step and close a TextWorld game; expose legal actions and success rewards |
| `codegym/` | Load generated task environments and execute actions with transaction isolation, timeouts and known source compatibility fixes |
| `dive/` | Tool execution, public task sessions and terminal judging; environment-side user/judge models are owned here |
| `webshop/` | Product/index resources, independent shopping sessions, native actions and purchase scoring |
| `t2bench/` | Official user simulation, tasks, tools, session progression and scoring; policy actions arrive from the caller |
| `swebench/` | Isolated workspaces, file/command tools, patch extraction, mini-swe-agent tools and official scoring |

Each `build_env(**kwargs)` returns the existing single-environment interface without creating Ray pools or starting training.
Environments retain their native task/action protocols, rewards, step budgets and tool formats.

```python
from agent_system.environments.env_package.codegym import build_env

env = build_env(envs_dir="/absolute/data/codegym/envs/codegym_v1")
try:
    observation = env.reset(env_str=task_env_str)
    observation = env.step(action)
finally:
    env.close()
```

The outer `backends/` layer handles Ray workers/pools, tool registration and training-side protocols. Environment implementations do not import trainers or experiment scripts.
ALFWorld and CodeGym workers inherit the implementations here; WebShop's JSONL transport calls its environment object.
DIVE, t2bench and SWE-bench load their dependencies on demand in isolated interpreters.
Package initialization and importing `build_env` remain lightweight. Constructing an environment still requires its actual dependencies and data.

Dependency installation and source-version checks are in [`ops/env_deps/`](../../../ops/env_deps/README.md),
standalone reference evaluators are in [`experiments/shared/train_eval/reference/`](../../../experiments/shared/train_eval/reference/README.md),
and dataset preparation is in [`experiments/shared/dataset/`](../../../experiments/shared/dataset/README.md).
Required dependency source is bundled in each environment's `source/` directory. `source_manifest.json` records its origin revision and per-file SHA-256 digests.
Original modules and licenses are retained. Environments call the execution paths; outer entrypoints handle installation, task synthesis and experiments.
Runtime execution requires no external Git checkout; DIVE and t2bench read the manifest for version checks.
Task data remains separate: t2bench's pinned text tasks, databases and resource snapshots are in `data/t2bench/source/`,
WebShop products and indexes are in `data/webshop/`, and CodeGym tasks and generated environments are in `data/codegym/`.
Source provenance and data identity are checked at read boundaries without changing official environment semantics.
`repos/` is only for manual reference; runtime, builds and tests do not depend on it.
