# Standalone ALFWorld reference evaluation

This directory retains a standalone HTTP service, ReAct evaluators and mapping tools. Current training uses `agent_system/environments/env_package/alfworld/`; this directory does not depend on an external reference checkout.

## Directory layout

| Directory | Purpose |
|---|---|
| [`alfworld_server/`](alfworld_server) | Self-contained ALFWorld HTTP service on port 36001; depends on the pip `alfworld` package |
| [`alfworld_react_baseline/`](alfworld_react_baseline) | Online ReAct evaluation of the base model, defaulting to Qwen2.5-3B-Instruct |
| [`alfworld_official_eval/`](alfworld_official_eval) | API-model OOD evaluation on `valid_unseen`, following the official ReAct paper protocol |
| [`tools/`](tools) | Mapping generation: `export_game_files.py` → `CreateMappings.py` |
| Original data report | The earlier `ALFWORLD_DATA_REPORT.md` described raw and processed data; consult the mapping flow below for the retained source contract |

The service and both evaluator directories provide their own READMEs.

---

## Data flow

```
~/.cache/alfworld                        Original ALFWorld games
    |
    | tools/export_game_files.py         Scan train / valid_seen
    v
tools/{train,test}_file.json             3552 / 140 game paths
    |
    | tools/CreateMappings.py
    v
alfworld_server/configs/mappings_{train,test}.json      3553 / 140 entries
    +  alfworld_official_eval/make_unseen_mappings.py
    -> alfworld_server/configs/mappings_unseen.json      134 entries (indices start at 3693)
    |
    +--> alfworld_server (HTTP, evaluation)
    +--> agent_system/environments/configs/ (byte-identical copies, training)
```

Both consumers must read the same mappings; otherwise one `game` index identifies different episodes during training and evaluation.
Mappings preserve the task order in the current environment configuration.

Standalone ReAct scripts still read legacy JSONL, defaulting to `data/alfworld/dyad_stmt/{train,test}.jsonl` and using each sample's `game`, `task_id`, `task_type` and `ground_truth`.
The current workspace contains Parquet under `data/alfworld/dataset/`, not that default JSONL. Supply compatible JSONL through `--data`; relocation alone does not make this legacy entrypoint ready to run.

---

## Training-side independence

Current training code does not import these reference evaluators or an external reference checkout.
Training uses the in-process pool in `agent_system/environments/backends/alfworld/pool.py`, configured through `agent_system/environments/configs/`.
This directory contains evaluation and data tools, outside the main training path.

The shared contract is that its mappings must match the training copy.

---

## Upstream components outside this integration

Upstream agentenv includes environments such as webshop, webarena, sciworld and babyai;
only its ALFWorld component is used here.
Other upstream components remain reference material and are not modified by this integration.

The upstream `agentenv-alfworld/configs/mappings_*.json` files are different from the mappings here:
upstream train has 2420 entries and its test comes from `valid_train`, with zero task_id overlap with this dataset.
Do not substitute them. See [alfworld_server/README.md](alfworld_server/README.md#configs-why-this-service-ships-its-own-copy).
