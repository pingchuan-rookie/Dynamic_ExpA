# ALFWorld ReAct reference evaluation through a model API

Evaluate a chat model, defaulting to **GPT-5.5**, using the ALFWorld evaluation structure
from **ReAct (Yao et al., 2022)**. The default supplies admissible actions; use `--no-admissible` for the candidate-free protocol.

Official reference: <https://github.com/ysymyth/ReAct>, `alfworld.py` and
`prompts/alfworld_3prompts.json`。

> This directory is the maintained copy of these evaluation scripts.
> The `repos/agentenv/` submodule has been restored to upstream.
> See [../README.md](../README.md).
>
> Mappings and the HTTP server are in [`../alfworld_server/`](../alfworld_server).
> Evaluation computes the OOD offset as `len(train)+len(test)`; the server concatenates games in that order.
> Both must read the same configs, so no separate copy is stored here.

---

## Shared elements with official ReAct

| Dimension | Official ReAct | This implementation |
|---|---|---|
| Split | `eval_out_of_distribution` = `valid_unseen` | Same: 134 `solvable=True` games in `configs/mappings_unseen.json` |
| 2-shot prompt | `react_{type}_1` + `react_{type}_0` | Same, using the original `prompts/alfworld_3prompts.json` |
| Task routing | Six gamefile directory prefixes matched with `startswith` | Same: put/clean/heat/cool/examine/puttwo |
| Actions | Pass the complete output line to `env.step([action])` without correction | Same: no candidate matching or regex correction |
| `think:` turns | Step normally, then replace the observation with `OK.` | Same |
| Success | `info['won']` (reward >= 1) | Same |
| Metrics | Success rates for six task categories | Same, plus overall success rate |

**API adaptation:** official ReAct uses a completion model (text-davinci-002, continuing with `stop=['\n']`).
This script puts examples, interaction history and the trailing `>` in one user message
and adds a short system instruction requesting only the next action line. Admissible-action injection is a separate protocol difference described below.

---

## Files

This directory is tracked in the main repository:

- [`run_react_eval.py`](run_react_eval.py): evaluation entrypoint.
- [`make_unseen_mappings.py`](make_unseen_mappings.py): generates
  `mappings_unseen.json` from `valid_unseen` under `../alfworld_server/configs/`.
- `prompts/alfworld_3prompts.json`: original official 2-shot prompts.
- `outputs/`: automatically created, Git-ignored evaluation JSON directory.

Related files in [`../alfworld_server/`](../alfworld_server):

- `configs/mappings_{train,test,unseen}.json`: server game indices.
- `start.sh`: starts the server, loading OOD games when `ALFWORLD_INCLUDE_UNSEEN=1`
  and generating `mappings_unseen.json` first if it is absent.

Override the configs directory with `ALFWORLD_CONFIGS_DIR`.

---

## Local pipeline validation

Developer checks are provisioned separately under ignored `dyad_test/` and are not included in a source clone:

```bash
python3 dyad_test/environments/alfworld_reference_smoke.py --evaluator official --limit 12
```

Run from the repository root. This uses scripted fixtures without services or GPUs and does not measure model performance.


## Run evaluation

### 1. Generate OOD mappings on first use or after data changes

`valid_unseen` contains 195 game files, but evaluation uses only games marked
`solvable=True`, matching the official environment's `collect_game_files` filter.
This selects **134** games, matching the task count in the ReAct paper.
The obsolete 150-game loadability-based selection included 16 games without expert solutions and must not be used.

```sh
# Run from the repository root.
# Run the unseen-mapping generator with the project interpreter.
ALFWORLD_DATA=~/.cache/alfworld python3 \
  experiments/shared/train_eval/reference/alfworld/alfworld_official_eval/make_unseen_mappings.py
#   -> kept (solvable=True): 134 ; skipped (unsolvable): 61 ; index range [3693, 3826]
# Output: experiments/shared/train_eval/reference/alfworld/alfworld_server/configs/mappings_unseen.json.
```

> System `python3` can read the `solvable` flag from `game.tw-pddl`.
> With alfworld installed (`.venvs/expa-verl/bin/python`), generation also checks loading and filters broken games.
> Run `python3 dyad_test/tools/check_alfworld_mappings.py` afterwards
> and synchronize `agent_system/environments/configs/alfworld_mappings_unseen.json`.

### 2. Start the ALFWorld server with valid_unseen

OOD loading is optional and off by default. Existing train/valid_seen indices remain unchanged:

```sh
# Run from the repository root.
ALFWORLD_INCLUDE_UNSEEN=1 bash experiments/shared/train_eval/reference/alfworld/alfworld_server/start.sh
```

`start.sh` generates missing `mappings_unseen.json`, so explicit generation can be skipped.

Optional check:

```sh
curl -s http://127.0.0.1:36001/stats
# Expect num_unseen_games > 0; this dataset has 134 solvable games.
```

### 3. Configure the model API and run

The OpenAI-compatible SDK reads `base_url` and `api_key` from environment variables,
defaulting to `GITHUB_MODELS_BASE_URL` and `GITHUB_TOKEN`. Use `--env-file` to load
a specific file; no default credentials file is configured.

```sh
# Run from the repository root.
python3 experiments/shared/train_eval/reference/alfworld/alfworld_official_eval/run_react_eval.py \
    --model openai/gpt-5.5 \
    --env-server http://127.0.0.1:36001 \
    --max-steps 50
```

> Requires `openai`, `requests`, and optionally `python-dotenv`.

- Alternate backend: `--base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY`.
- Evaluate only the first N tasks: `--limit 134`.

## Common arguments

| Argument | Default | Purpose |
|---|---|---|
| `--model` | `openai/gpt-5.5` | Model name: GitHub Models uses `openai/...`; copilot-api uses `gpt-5.5` |
| `--api` | `chat` | `chat` uses /chat/completions; `responses` uses /responses |
| `--env-server` | `http://127.0.0.1:36001` | ALFWorld HTTP server |
| `--base-url` | `GITHUB_MODELS_BASE_URL` | OpenAI-compatible base URL |
| `--api-key-env` | `GITHUB_TOKEN` | Environment variable containing the API key |
| `--env-file` | None | Optional .env file; otherwise use the environment only |
| `--limit` | All | Evaluate the first N games; official set size is 134 |
| `--max-steps` | `50` | Per-game step limit; official ReAct uses 49 |
| `--temperature` | `0.0` | Sampling temperature; ignored for Responses reasoning models |
| `--max-tokens` | `256` | Chat reply token limit, with fallback to `max_completion_tokens` |
| `--max-output-tokens` | `2048` | Responses output budget, including reasoning tokens |
| `--request-interval` | `0.0` | Minimum seconds between calls; set above zero for a rate-limited proxy |
| `--admissible` / `--no-admissible` | `--admissible` | Supply legal actions after each observation; disable for the official candidate-free protocol |
| `--record-raw` | Off | Record complete prompts, model replies and environment replies, plus readable `.txt` |
| `--verbose` | Off | Print interactions |
| `--no-verify-server` | Off | Skip the initial `/stats` check |
| `--output` | Auto | Output JSON path |

> **Admissible actions differ from the official protocol.** Official ReAct generates actions from
> 2-shot examples without candidate lists. This repository defaults to appending `available_actions`
> as `Admissible actions: [...]` after each observation, using reset candidates initially.
> `ADMISSIBLE_SYSTEM_PROMPT` asks for an exact candidate or `think:`. Use `--no-admissible`
> for the candidate-free protocol. Candidates are added only to the current task, not retrofitted into examples.

---

## Outputs

- The terminal reports overall and six-category success rates.
- `outputs/react_official_unseen_<timestamp>.json` stores `summary`, `config` and complete
  per-game `transcript` entries: action, is_think, observation, reward and done.

---

## Protocol fidelity

- **134 versus 195 OOD games:** filtering the 195 original files by `solvable=True`
  yields the default 134-game set, matching the official environment and paper's count.
  The former 150-game loadability-based set included 16 unsolvable games and is obsolete.
- **Broken games:** generation filters these. If reset still fails, the server catches errors,
  including fast_downward's `SystemExit`, without crashing. The evaluator counts load errors separately
  and excludes them from the loadable-game success-rate denominator: successes / loadable games.
- **Comparability:** base model, data version, seen/unseen split, examples, step budget and reward
  definition must match before comparing success rates.
- **Step limit:** official `range(1, 50)` permits 49 steps. Use `--max-steps 49` to match it.
- **OOD indices:** `ALFWORLD_INCLUDE_UNSEEN=1` appends valid_unseen to `self.games`,
  starting at global index 3693. Existing zero-based train/valid_seen indices remain unchanged,
  preserving the training data contract.
