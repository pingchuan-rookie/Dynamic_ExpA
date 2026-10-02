# GPT-5.5 evaluation on all 134 ALFWorld valid_unseen tasks

> **Historical report recorded before August 2026.**
> Paths and commands below reflect the original layout under `repos/agentenv/agentenv-alfworld/`,
> using `start_unseen_server.sh`. Scripts now live in `experiments/shared/train_eval/reference/alfworld/`
> and the server starts through `alfworld_server/start.sh`; see the adjacent README for current usage.
> The original findings and measurements are preserved.

- **Date:** 2026-07-14
- **Model:** `gpt-5.5` through a local copilot-api proxy using the **Responses API**
- **Dataset:** all **134** `solvable=True` ALFWorld `valid_unseen` games, with indices `[3693, 3826]`, matching the ReAct paper's task count
- **Result:** **134/134 successful, 100.00% overall**, with 100% in all six official categories; admissible actions were supplied

---

## 1. Summary

| Metric | Value |
|---|---|
| Total solvable OOD / valid_unseen tasks | 134 |
| Loadable games | 134; load errors: 0 |
| Successful games | **134** |
| Success rate over loadable games | **100.00%** |
| Success rate over all games | **100.00%** |
| Games with errors | 0 |
| Mean steps for successful games | 10.86 |
| Median steps for successful games | 9 |
| Step range | 4-40 |
| Total duration | 4998 seconds, approximately 83 minutes, including `--request-interval 3` |

### Six official task categories

| Task type | Successful / total | Success rate |
|---|---|---|
| put（pick_and_place） | 24 / 24 | 100.0% |
| clean（pick_clean_then_place） | 31 / 31 | 100.0% |
| heat（pick_heat_then_place） | 23 / 23 | 100.0% |
| cool（pick_cool_then_place） | 21 / 21 | 100.0% |
| examine（look_at_obj_in_light） | 18 / 18 | 100.0% |
| puttwo（pick_two_obj_and_place） | 17 / 17 | 100.0% |
| **Total** | **134 / 134** | **100.0%** |

---

## 2. Evaluation configuration

From the `config` section of `outputs/gpt55_full134.json`:

| Parameter | Value | Notes |
|---|---|---|
| `model` | `gpt-5.5` | Reasoning model |
| `api` | `responses` | This service required Responses; /chat/completions returned `unsupported_api_for_model` |
| `admissible` | `true` | **The LLM received the environment's legal action candidates at every step**; see section 4 |
| `env_server` | `http://127.0.0.1:36001` | ALFWorld HTTP server with `ALFWORLD_INCLUDE_UNSEEN=1` |
| `max_steps` | 50 | Maximum interactions per game |
| `temperature` | 0.0 | Greedy decoding setting |
| `request_interval` | 3.0 s | Rate limit used with the copilot-api proxy |
| `max_output_tokens` | 4000 | Includes Responses reasoning tokens; an insufficient budget can leave `output_text` empty |
| `prompts` | `prompts/alfworld_3prompts.json` | Original official 2-shot prompts |

**API access:** this run accessed `gpt-5.5` through a local **copilot-api** proxy rather than the GitHub Models catalog.
The OpenAI-compatible endpoint was `http://localhost:4141/v1`, with no authentication and a placeholder key.

---

## 3. Elements reproduced from official ReAct

The evaluation follows the ALFWorld structure from ReAct (Yao et al., 2022),
using [`ysymyth/ReAct`](https://github.com/ysymyth/ReAct), `alfworld.py` and
`prompts/alfworld_3prompts.json`）：

- **Split:** `eval_out_of_distribution` = `valid_unseen`, selecting 134 `solvable=True` games.
- **2-shot prompt:** `react_{type}_1` + `react_{type}_0` from the original prompt file.
- **Task routing:** six gamefile directory prefixes: put/clean/heat/cool/examine/puttwo.
- **`think:` turns:** call `env.step`, then replace the observation with `OK.`.
- **Success:** `info['won']` (reward >= 1).
- **Metrics:** success rates for all six official categories and overall.

**Chat adaptation:** official ReAct uses text-davinci-002 completions with `stop=['\n']`.
This run places the 2-shot examples, current history and trailing `>` in one user message
with a short system instruction to output only the next action. The action-candidate difference is described below.

---

## 4. Key protocol difference: admissible actions

This evaluation enabled **`--admissible`**. Each observation was followed by
the environment's `Admissible actions: [...]` list, with a system instruction
to **select a candidate verbatim**.

- This protocol change was explicitly requested for the experiment.
- Compared with official candidate-free ReAct, it makes action generation easier:
  the model need not construct legal action strings and avoids most invalid-format turns.
- Interpret the **100% success rate under the supplied-candidate condition**.
  It is not directly comparable with candidate-free official ReAct scores. Observed candidate lists typically had more than 20 actions per step.

> For a candidate-free comparison, rerun with `--no-admissible`, which restores
> `DEFAULT_SYSTEM_PROMPT` without candidate instructions.

---

## 5. Step statistics

- Mean **10.86**, median **9**, minimum **4**, maximum **40** steps for successful games.
- Distribution across all 134 successful games:

| Step range | Games | Share |
|---|---|---|
| ≤ 5 | 10 | 7.5% |
| 6 – 10 | 82 | 61.2% |
| 11 – 20 | 30 | 22.4% |
| 21 – 35 | 10 | 7.5% |
| > 35 | 2 | 1.5% |

Most tasks finished within 10 steps. The long tail mainly involved opening multiple
containers to find objects or greater exploration for puttwo tasks.

### Five longest games, all successful

| Index | Task | Steps |
|---|---|---|
| 3813 | pick_two_obj_and_place-KeyChain-None-Safe-219（puttwo） | 40 |
| 3784 | pick_cool_then_place_in_recep-Tomato-None-Microwave-10（cool） | 36 |
| 3730 | pick_and_place_simple-Vase-None-Safe-219（put） | 34 |
| 3797 | pick_heat_then_place_in_recep-Egg-None-GarbageCan-10（heat） | 34 |
| 3787 | pick_heat_then_place_in_recep-Apple-None-Fridge-10（heat） | 29 |

All finished within `max_steps=50`; none failed at the step limit.

---

## 6. Historical reproduction commands

Prerequisites: start the OOD-enabled ALFWorld server with `start_unseen_server.sh` on port 36001;
`/stats` should report `num_unseen_games=134`. The local copilot-api proxy must be available on port 4141.

```sh
cd agentenv/agentenv-alfworld/official_react_eval

COPILOT_PROXY_KEY=dummy ../../.venv/bin/python run_react_eval.py \
    --api responses \
    --model gpt-5.5 \
    --base-url http://localhost:4141/v1 \
    --api-key-env COPILOT_PROXY_KEY \
    --env-server http://127.0.0.1:36001 \
    --admissible \
    --max-steps 50 \
    --max-output-tokens 4000 \
    --request-interval 3 \
    --output outputs/gpt55_full134.json
```

Use `--no-admissible` for the candidate-free variant.

---

## 7. Artifacts

| File | Contents |
|---|---|
| `outputs/gpt55_full134.json` | Complete `summary`, `config`, per-game `results` and transcripts |
| `outputs/gpt55_full134_20260713_104536.log` | Progress log and final statistics |

Each `results[i].transcript` in `gpt55_full134.json` contains the game's raw
prompts, model responses and environment replies.

---

## 8. Interpretation

- With **admissible actions supplied**, gpt-5.5 achieved **100% success across all 134 games**,
  covering all six categories in approximately 11 steps on average, without errors or step-limit failures.
- This measures multi-step decision-making and planning with legal action candidates.
  It does not establish performance under candidate-free official ReAct; comparisons require matched protocols.
- A full `--no-admissible` evaluation would provide the candidate-free comparison.
