# overlay/agentenv/alfworld_server/

HTTP environment server for ALFWorld.
Endpoints: `create`, `reset`, `step`, `close`, `stats`, `observation` and `available_actions`. Default port: 36001.

Requires `alfworld`,
plus fastapi, uvicorn, anyio and pyyaml.
Implementation and launch scripts are in this directory.

---

## Consumers

Used by the evaluators in `../alfworld_react_baseline/` and `../alfworld_official_eval/`.

Training uses the in-process environment pool.
`agent_system/environments/backends/alfworld/pool.py` runs ALFWorld directly:
alfworld is installed in the training venv, and each trajectory leases a Ray actor without an HTTP layer.
The training pool and HTTP server serve separate entrypoints with independent installation and runtime dependencies.

---

## Start the server

Game data defaults to `~/.cache/alfworld`. Download it first if absent (approximately 20 minutes):

```sh
# Run from the repository root.
ALFWORLD_DATA=~/.cache/alfworld .venvs/expa-verl/bin/alfworld-download
```

Then run:

```sh
# Run from the repository root.
bash experiments/shared/train_eval/reference/alfworld/alfworld_server/start.sh

# Include valid_unseen: historical snapshot of 134 games, starting at index 3693.
ALFWORLD_INCLUDE_UNSEEN=1 bash experiments/shared/train_eval/reference/alfworld/alfworld_server/start.sh

# Override port or interpreter.
PORT=36011 ALFWORLD_PYTHON=/opt/venv-alfworld/bin/python bash .../start.sh
```

`start.sh` selects Python in this order: `$ALFWORLD_PYTHON`, `/opt/venv-alfworld/bin/python` (container),
then `<repo>/.venvs/expa-verl/bin/python` (development environment with alfworld 0.4.2).
Run `setup_venv.sh` only if a separate venv is needed.

Check the game counts after startup:

```sh
curl -s localhost:36001/stats
#   total_games 3693 = 3553 train + 140 test
# Historical OOD snapshot: 3827 total, unseen_start_index 3693, num_unseen_games 134.
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ALFWORLD_DATA` | `~/.cache/alfworld` | Game data directory |
| `ALFWORLD_CONFIGS_DIR` | This package's `configs/` | Mappings and base_config directory |
| `ALFWORLD_INCLUDE_UNSEEN` | Off | Set to `1` to append valid_unseen games |
| `ALFWORLD_SERVER_DIAG_PATH` | `outputs/alfworld_server.jsonl` | Structured event log; empty disables it |
| `ALFWORLD_SERVER_THREADPOOL_LIMIT` | 256 | FastAPI thread pool limit |
| `ALFWORLD_PYTHON` | Auto-detected | Interpreter used by `start.sh` |

---

## Configs: why this service ships its own copy

`mappings_{train,test,unseen}.json` determines which game each `game` index selects.
The upstream agentenv mappings use a different task set:

| Split | Upstream | This service |
|---|---|---|
| train | 2420 entries | **3553** entries |
| test | 200 entries from `json_2.1.1/valid_train` | **140** entries from `valid_seen` |
| unseen | Absent | **134** valid_unseen games with `solvable=True` |

Only 2181 train task_ids overlap; the test task_id intersection is empty.
The same index can therefore refer to different games.

`agent_system/environments/configs/` contains a byte-identical copy for training.
Keep the copies identical so training and evaluation indices select the same games.
Check consistency with:

```sh
python3 experiments/tools/check_alfworld_mappings.py
```

The check also verifies that `mappings_unseen.json` starts at item_id = train count + test count.
Both the server and `run_react_eval.py` use that offset for the OOD index range.

To regenerate the three mappings:

```
ALFWORLD_DATA
  -> ../tools/export_game_files.py  -> train_file.json / test_file.json
  -> ../tools/CreateMappings.py     -> configs/mappings_{train,test}.json
  -> ../alfworld_official_eval/make_unseen_mappings.py -> configs/mappings_unseen.json
```

Synchronize the copy in `agent_system/environments/configs/` and rerun the check afterwards.

---

## Differences from the former submodule implementation

The move preserved execution logic while making the service self-contained:

| Component | Former submodule version (reverted) | This service |
|---|---|---|
| `SingleAlfredTWEnv` | `agentenv_alfworld/environment.py`, with an upstream default changed to `train_eval="train"` | `single_env.py`, with an explicit argument and no unused `gym` import |
| `load_config` / `process_ob` | `agentenv_alfworld/utils.py` | `utils.py` |
| Config path | `dirname(__file__)/../configs` | `configs_dir()`, with an environment override |
| Startup | `start.sh` required `SCRIPT_DIR/../.venv`, absent on the development machine | Interpreter priority supports development, containers and explicit overrides |

This removes the missing-interpreter and unused-gym requirements of the former local startup path.

## Extensions to upstream agentenv

The following behavior was added locally before the move:

- **Concurrency safety:** per-session locks and a global `_engine_lock`.
  TextWorld's Gym/parser state is process-global; concurrent engine entry can corrupt it.
- **Resource release:** `close()` closes each session and removes its env, info and lock, instead of relying on bulk cleanup in `__del__`.
- **Synchronous routes:** FastAPI runs blocking TextWorld calls in its thread pool rather than uvicorn's event loop.
- **`/stats` and `/close` endpoints.**
- **Explicit errors:** payloads containing `error` return HTTP 500 with `error_type`.
- **`SystemExit` handling:** malformed `game.tw-pddl` files can cause fast_downward to exit.
- **Environment-first `ALFWORLD_DATA`:** upstream unconditionally uses `~/.cache/alfworld`,
  which can override an entrypoint's Blob path and make `/create` fail.
- **Opt-in OOD loading:** append-only, preserving existing indices.
- **Structured diagnostic logs** in `diagnostics.py`.
