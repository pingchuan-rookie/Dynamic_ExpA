# ALFWorld ReAct baseline with an untrained model

Evaluate an untrained model's ReAct behavior on ALFWorld. The default model is **Qwen2.5-3B-Instruct**.

See the [parent guide](../README.md) for the reference workflow and intended use.

## Evaluation behavior

- **Task identity:** read `data/alfworld/dyad_stmt/{train,test}.jsonl` by default, or provide compatible JSONL with `--data`.
  Reuse each record's `game` (the server reset index), `task_id`, `task_type` and `ground_truth`.
  This workspace currently has only the Parquet files from [data preparation](../../../../dataset/README.md). Supply `--data`; these commands cannot read Parquet directly.
- **Prompt construction:** build a conventional ALFWorld
  **ReAct** system prompt (`Thought:` + `Action:`),
  supplying the environment's **available actions** at each step instead of reusing KV-action prompts.
- **Online interaction:** communicate with the ALFWorld FastAPI server over HTTP.
  `create -> reset(game) -> step* -> close`。
- **Action extraction:** extract text after `Action:` and match it to available actions.
  - Default **strict matching:** exact, then case-insensitive, then `in/on`, whitespace, punctuation and Markdown normalization.
    Object indices are not corrected. Hallucinated targets such as a nonexistent `cabinet 3` remain invalid.
  - `--fuzzy-match` enables substring and difflib matching to repair missing indices or typos.
    It can map hallucinated indices to other actions and inflate scores, so it is disabled by default.

## Files

- [`react_baseline.py`](react_baseline.py): standalone evaluation entrypoint.
- `outputs/`: automatically created directory for evaluation JSON.

## Local pipeline validation

Developer checks are provisioned separately under ignored `dyad_test/` and are not included in a source clone:

```bash
python3 dyad_test/environments/alfworld_reference_smoke.py --evaluator baseline --limit 12
```

Run from the repository root. This uses scripted fixtures without services or GPUs and does not measure model performance.


## Evaluation with environment and model services

Start the required services before running evaluation.

### 1. Start the ALFWorld server

The server is in [`../alfworld_server/`](../alfworld_server):

```sh
# Run from the repository root.
bash experiments/shared/train_eval/reference/alfworld/alfworld_server/start.sh    # Default: 0.0.0.0:36001.
```

### 2. Start an OpenAI-compatible model service, such as vLLM

```sh
vllm serve Qwen/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --served-model-name Qwen/Qwen2.5-3B-Instruct
# Endpoint: http://127.0.0.1:8000/v1.
```

### 3. Run the baseline

```sh
# Run from the repository root.

python3 experiments/shared/train_eval/reference/alfworld/alfworld_react_baseline/react_baseline.py \
    --split test \
    --env-server http://127.0.0.1:36001 \
    --model-base-url http://127.0.0.1:8000/v1 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --max-steps 40 \
    --temperature 0.0
```

Evaluation requires `requests`; use an interpreter with dependencies, such as `.venvs/expa-verl/bin/python`.

> Expert trajectories can approach 98 steps. Untrained models may need a larger budget;
> consider `--max-steps 50` for long clean, heat, cool or two-object tasks.

## Common arguments

| Argument | Default | Purpose |
|---|---|---|
| `--split` | `test` | `test`(50) / `train`(1000) |
| `--data` | Auto | JSONL path, overriding `--split` |
| `--limit` | All | Evaluate the first N tasks |
| `--env-server` | `http://127.0.0.1:36001` | ALFWorld server URL |
| `--model-base-url` | `http://127.0.0.1:8000/v1` | OpenAI-compatible endpoint |
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | Model name |
| `--max-steps` | `40` | Maximum interactions per task |
| `--temperature` | `0.0` | Sampling temperature; greedy is recommended for this baseline |
| `--history-window` | `0` | History truncation window; 0 retains all history |
| `--no-fewshot` | Off | Omit in-context examples from the first turn |
| `--fuzzy-match` | Off | Enable substring/difflib action matching |
| `--verbose` | Off | Print each interaction |
| `--output` | Auto | Output JSON path |

## Outputs

- The terminal reports success rate, errors, mean successful-task steps and success rates by `task_type`.
- `outputs/react_<split>_<timestamp>.json` stores `summary` and each task's full `transcript`,
  including thought, parsed/matched action, observation and reward.

## Protocol notes

- A task succeeds when an environment step returns `reward >= 1.0` (server `info["won"]`).
- Unmatched action text is sent unchanged to the environment, which typically returns
  `Nothing happens.` The event is counted in `invalid_actions`.
- `game` indices match `../alfworld_server/configs/mappings_{train,test}.json`,
  preserving task identity with `alfworld/dyad_stmt`.
