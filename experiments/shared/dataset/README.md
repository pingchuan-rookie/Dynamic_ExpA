# Agentic RL data preparation

## Script responsibilities

Each environment has one main script for its data generation, validation and offline operations.

| Script | Responsibility |
|---|---|
| `gsm8k.py` | Recover GSM8K questions/answers from local preprocessed records, preserve provenance and build evaluation transport data |
| `alfworld.py` | Generate ALFWorld data and prepare/validate the complete evaluation set |
| `alfworld_precompute_gt.py` | Precompute expert trajectories and cache initial observations, with resume support |
| `codegym.py` | Generate CodeGym data; provide offline `schema` and `select-test` commands |
| `webshop.py` | Generate initial observations with real resets for official full human tasks; validate splits and manifests |
| `dive.py` | Validate official DIVE-RL/DIVE-Eval, convert task snapshots and audit invalid records |
| `t2bench.py` | Export t2bench evaluation snapshots |
| `swebench_verified.py` | Preserve all 500 official task identities and public fields; select a fixed 50 for evaluation; keep hidden tests on the grading side |
| `gen_sample_records.py` | Generate and validate cross-environment sample documentation |

`prompt/` contains WebShop's current data templates and loader. Shared interaction templates live in `agent_system/environments/prompts/`.
`utils/source_snapshot.py` handles provenance, manifests and transactional publication; `utils/tau_snapshot.py` handles t2bench snapshots.
`alfworld_precompute_gt.py` owns expert solving and initial-state caching. Normal snapshot rebuilding does not invoke experts or reset environments.

## Data directory responsibilities

Top-level `data/` directories reflect data purpose; algorithms share the same data.
`data/swebench_verified/` is evaluation-only. See the [adapter guide](../../../agent_system/environments/env_package/swebench/README.md) for data/image preparation, checks and offline boundaries.

| Directory | Responsibility |
|---|---|
| `actenc_alignment/` | Alignment projector data: training inputs in `final/`, generation records and source snapshots in `intermediate/` |
| `capability/` | Evaluation-only snapshots, including MMLU-Pro, HMMT and LiveCodeBench |
| `gsm8k/` | Retained `dataset/` and sample documentation, for evaluation only |
| `dive/` | Original snapshots and `download_manifest.json`; audited shared tasks under `dataset/` |
| `codegym/` | Shared tasks and required `dataset/envs/codegym_v1/` environment source |
| `alfworld/` | Shared tasks; formal evaluation uses `dataset/test_unseen.parquet`; seen data is retained |
| `webshop/` | Complete products, tasks and search index in `assets/`; train/dev/test in `dataset/` |
| `t2bench/` | Official base-task snapshots in `dataset/`, for evaluation only; runtime still requires the official environment |

Directory organization does not change data protocols, splits or training eligibility.
Training is limited to DIVE, CodeGym, ALFWorld and WebShop. GSM8K and t2bench are evaluation-only.
`capability_eval/backend.py` currently downloads or reads pinned official data from the Hugging Face cache; it does not automatically read `data/capability/`.
Its per-run `dataset/` contains converted outputs, not shared source data.

## Fixed environment evaluation protocol

The `env_eval` suite contains four targets. DIVE/CodeGym retain training and historical assets but are not targets in this suite; GSM8K is not included by default.

| Target | Official source and formal evaluation membership |
|---|---|
| ALFWorld | All 134 `valid_unseen` tasks in `dataset/test_unseen.parquet`; the 140 `valid_seen` tasks remain for development or training validation |
| WebShop | Fixed 100 of 500 official test tasks in `dataset/test100_seed42.parquet`, identified by `dataset/test100_seed42.selection.json` |
| SWE-bench Verified | Fixed 50 of 500 official test tasks, one attempt each; resolved percentage uses denominator 50 |
| t2bench | Existing base protocol: Airline / Retail / Telecom = 50 / 114 / 114, totaling 278 |

Selection rules are defined in [evaluation_protocol.py](../evaluation_protocol.py).
Sort WebShop by numeric goal ID and SWE by lexicographic instance ID. Select 100/50 without replacement using separate `random.Random(42).sample` calls, then save sorted IDs.
Selections are independent of model scores and reused across models, methods and training sources. Arbitrary equal-sized selections and debug truncation are not the formal subsets.
Keep complete official sources and original train/dev/test files unchanged; derived subsets must not overwrite them.
Result identity requires unique complete task coverage, source digests and actual artifacts. Full-set results do not automatically establish subset results.

## Shared training data

ALFWorld, CodeGym and WebShop each provide one algorithm-independent dataset, shared by GRPO-ReAct and Dyad, including the ALFWorld closed-argument ablation.
DIVE also shares tasks while retaining each task's native tool protocol.
GSM8K retains split names and offline generation under `data/gsm8k/dataset/`; this does not authorize training.
Data contains no `agent_name` or algorithm-specific `surface_form`. Entrypoints choose the agent loop; schemas define Dyad action selection.

Shared prompts use the ReAct text protocol. Dyad adds no action-head instructions, automatic action-name completion, empty Action fields or other method-specific guidance.
Runtime and schema code implement argument-selection differences without changing task instructions.

```text
data/<env>/
├── dataset/
│   ├── train.parquet
│   ├── dev.parquet          # WebShop training validation
│   ├── test.parquet
│   ├── test_unseen.parquet  # ALFWorld formal unseen134 evaluation
│   ├── test_full.parquet    # Retained ALFWorld seen140 source
│   ├── test100_seed42.parquet       # WebShop fixed 100-task evaluation
│   ├── test100_seed42.selection.json # WebShop selection identity
│   ├── source/              # Complete ALFWorld/CodeGym/GSM8K snapshots
│   ├── manifest.json        # Provenance, file digests and ordered task IDs
│   └── envs/codegym_v1/     # Required CodeGym environment source
└── SAMPLE_RECORDS.md
```

### ALFWorld, CodeGym and GSM8K provenance and transport contract

These environments mark new data with `extra_info.dataset_format = source_tasks_v1` and retain six columns: `data_source`, `index`, `prompt`, `ability`, `reward_model`, `extra_info`.
A transport projection is not the complete source record; a complete source record is not model input.

| Layer | Contents and boundary |
|---|---|
| Complete source snapshot | `dataset/source/` preserves original files/assets, possibly including answers, trajectories, oracles, environment source and private initialization. These are not exposed automatically to the model, encoder or examples |
| Public transport input | Parquet `prompt` contains allowlisted public task input. `extra_info.source_record` stores relative location, row/task identity and SHA256; `extra_info.index` matches the transport row |
| Shared runtime template | `agent_system/environments/prompts/` and shared sessions handle roles, response protocol, history and live observations for both GRPO-ReAct and Dyad |

`ability`, `reward_model` and private reset arguments remain initialization/grading data even when stored in the same row.
Examples display public prompts, safe identities, source locations/digests and redacted private fields, excluding full sources, answers, oracles, env_str and ability.
Changing runtime roles, response format or history templates does not require rebuilding source snapshots or Parquet.
Rebuild explicitly when provenance, task membership, public-input extraction or the data contract changes.
Legacy local prompt templates and generation branches have been removed; historical snapshots remain unchanged.

| Environment | Source and verification boundary | Fixed membership and splits |
|---|---|---|
| ALFWorld | Mapping-selected local `json_2.1.1` assets: `traj_data.json`, `game.tw-pddl`, `initial_state.pddl`, optional `receps.json`, shared logic and mappings. Official release byte digests are not yet verified | train 3553, valid_seen 140, valid_unseen 134; local test/test_full reuse valid_seen |
| CodeGym | Reproduction from `VanishD/CodeGym` revision `85286359a342f7a288aea74273772b69b9b784c2`, preserving four task shards, linked environment shards and dataset card; not claimed as official paper data | Source has train only; local train 79106 and fixed final 1024 test records, with no reshuffle or repartition |
| GSM8K | Losslessly recovered `extra_info.question/answer` from `data/gsm8k/source/{train,test}.parquet`, retaining files/digests and marked `recovered_from_local_preprocessed`; official native artifact/revision unverified | Original train 7473 and test 1319, both evaluation-only |

CodeGym public input retains source system function declarations and the complete first user instruction, without trimming at `Question:` or exposing solution traces.
`dataset/envs/codegym_v1/` is a required runtime asset delivered with the full source snapshot.
GSM8K public input contains only the question; the full answer and calculator ground_truth stay private.
ALFWorld's stored public input is not a replay of live observations. Runtime task descriptions, observations and available actions come from real reset/step calls.
Existing walkthroughs remain private references; prompt changes do not require rerunning experts.

From `dynamic-expa/`, validate published snapshots and derived data without regenerating them:
`--validate-only` is an alias for `--check-only`.

```bash
.venvs/expa-verl/bin/python experiments/shared/dataset/gsm8k.py --check-only
.venvs/expa-verl/bin/python experiments/shared/dataset/codegym.py --check-only
.venvs/expa-verl/bin/python experiments/shared/dataset/alfworld.py --check-only
```

For an explicit overwrite/rebuild, use the commands below. Transactional publication preserves a digest-identified recovery snapshot.
Sources default to verified local assets. GSM8K explicitly selects its recovery directory; CodeGym accepts `--snapshot-dir`, and ALFWorld accepts `--assets` and `--cache` for assets and historical reset caches.

```bash
.venvs/expa-verl/bin/python experiments/shared/dataset/gsm8k.py --gsm8k_dir data/gsm8k/source --overwrite
.venvs/expa-verl/bin/python experiments/shared/dataset/codegym.py --overwrite
.venvs/expa-verl/bin/python experiments/shared/dataset/alfworld.py --overwrite
```

Source-preservation entrypoints do not offer online downloading, sample truncation or membership repartitioning.
Complete sources retain all original fields, including private references, without a selective-drop option.
ALFWorld `--full-eval` validates an existing 134-task `test_unseen.parquet`. `--output` may select another file containing the same membership; missing data is not generated implicitly.
Task-only data requires the shared step protocol and cannot use the legacy full-trajectory loop. Match old snapshots explicitly to their protocol.
Startup checks Parquet digests, task identity, manifests and source/asset presence and size. `--check-only` verifies full source digests and source-to-transport consistency.
Rebuilding changes the data fingerprint and does not imply exact full-state resume compatibility with old checkpoints.

Single-environment CodeGym filtering is for offline analysis; formal training rejects it.
Explicit `codegym.py schema ...` and `codegym.py select-test ...` commands export schemas and filter existing test data; normal data generation invokes neither.
See each subcommand's `--help` and the [action schema guide](../../../agent_system/policies/dyad/actions/schemas/README.md).

### Examples and length statistics

After rebuilding and validating the three environments, run these commands from `dynamic-expa/`:

```bash
MODEL_NAME=Qwen3.5-2B .venvs/expa-verl/bin/python experiments/shared/dataset/gen_sample_records.py --env gsm8k --env alfworld --env codegym
MODEL_NAME=Qwen3.5-2B .venvs/expa-verl/bin/python experiments/shared/dataset/gen_sample_records.py --env gsm8k --env alfworld --env codegym --check
```

Examples read and redact train row 0 from the actual Parquet file, without opening linked source files.
Lengths measure stored public input under the specified tokenizer's chat template; they are not live step-prompt lengths or runtime context budgets.
They exclude runtime templates, history and live observations. Generation neither calls a model nor resets an environment, and does not fabricate observations.
`--check` verifies that sample documentation matches the current redacted data view. It does not replace source, identity or environment checks.

## WebShop full human tasks

Complete resources and task exports live in `data/webshop/assets/`; shared splits in `dataset/{train,dev,test}.parquet`, audit metadata in `dataset/manifest.json`, and examples in `SAMPLE_RECORDS.md`.
First run `prepare_webshop_assets.py --install` to prepare full products, human instructions, the search index and isolated backend dependencies.
See its `--help` for asset options and checks. WebShop dependencies are not installed into the training interpreter.
Preparation and `--verify-only` check provenance, deterministic goals and task-export completeness without executing a fixed purchase trajectory.
New manifests label this scope `asset_integrity`; interaction checks run separately in local developer tests.

```bash
.venvs/expa-verl/bin/python experiments/shared/dataset/webshop.py
.venvs/expa-verl/bin/python experiments/shared/dataset/webshop.py --check
MODEL_NAME=Qwen3.5-2B .venvs/expa-verl/bin/python experiments/shared/dataset/gen_sample_records.py --env webshop
MODEL_NAME=Qwen3.5-2B .venvs/expa-verl/bin/python experiments/shared/dataset/gen_sample_records.py --env webshop --check
```

Use `--assets-dir` and `--output-dir` for deployed copies. Existing files are validated by default; regeneration requires explicit `--overwrite`.
Preserve official goal order: test `[0,500)`, dev `[500,1500)`, train `[1500,N)`. Do not reshuffle splits.
Do not label 1k-product subsets, synthetic goals or incomplete exports as full.
Generation obtains initial observations through real backend resets. Any failure stops generation rather than dropping rows or publishing incomplete splits.
The six columns are `data_source`, `index`, `prompt`, `ability`, `reward_model`, `extra_info`; `task_id` retains the official goal index and `index` is the split-local row.
`reward_model.ground_truth` is empty. The environment grades hidden goals; prompts, tool initialization and examples exclude target products, hidden attributes, price thresholds and reference traces.
Training uses train/dev. Independent evaluation validates the complete test source before deriving the protocol's fixed 100 tasks.
`test100_seed42.parquet` and `test100_seed42.selection.json` never overwrite the original 500-task `test.parquet`. Existing derived files must match fixed IDs and source digests.
Debug may restrict samples further but does not change the formal subset or establish a complete 100-task score.
Algorithms share `prompt/webshop.md` and the `webshop_action` tool. Record raw reward `0..1` separately from full success rate.

## DIVE official tasks

DIVE uses existing DIVE-RL-3K and DIVE-Eval data, without task synthesis, added SFT training or custom OOD splits.
Original data lives under `data/dive/`. Conversion checks SHA256, byte counts, row counts and pinned revisions against `download_manifest.json`.
Select only `academic`, `biological` and `medical` by default, excluding financial and all `_general` domains.
Write `data/dive/dataset_science3/{train,test}.parquet`, preserving complete task JSON, original row numbers, identities and source digests. Keep old `dataset/` unchanged.
Reference answers are visible only to environment actors and judges, not policy prompts or the action encoder.
Baseline and Dyad share tasks, native tool schemas and prompts.

```bash
.venvs/expa-verl/bin/python experiments/shared/dataset/dive.py --drop-invalid
.venvs/expa-verl/bin/python experiments/shared/dataset/dive.py --drop-invalid --check
```

Each selected domain originally has 400 training records, totaling 1200. Biological source row 3199 is a failed-generation placeholder.
Excluding it yields academic 400, biological 399, medical 400: **1199** training tasks. Validation has 100 per domain, totaling **300**.
The public training entrypoint defaults to `DIVE_DROP_INVALID=1`. Standalone conversion requires `--drop-invalid`; `DIVE_DROP_INVALID=0` strictly rejects the placeholder.
`manifest.json` records selected_domains, per-record domain filtering, invalid records, original/effective counts, source versions and output digests.
`--drop-invalid` does not hide schema damage, duplicate IDs or source-digest errors. Loading reapplies the domain filter before sample limits.
Existing output is validated by default. Overwrite requires `--overwrite` and rejects targets with mismatched digests or unknown provenance.
Training and validation share three domains; this is not unseen-domain OOD and must not reuse old eight-domain denominators.
This subset needs no SandboxFusion, Tushare, Serper, Jina or browse LLM. Public academic/medical queries still need networking, and grading needs an LLM judge.

## External benchmark evaluation data

`t2bench.py` exports official interactive-task snapshots for evaluation, without training data or Agentic RL training integration.
The native environment handles user simulation, tools and grading. Default Ray evaluation validates snapshots against official tasks; the reference evaluator reads official tasks directly.
The old tau-bench exporter, generated `tbench/` data and official `repos/tbench/` checkout have been removed.
Successful export does not establish model evaluation or Dyad inference compatibility.

```text
data/
└── t2bench/
    ├── dataset/
    │   ├── base.parquet
    │   └── manifest.json
    └── SAMPLE_RECORDS.md
```

Run from `dynamic-expa/` without a model, tokenizer, GPU or network:

```bash
.venvs/expa-verl/bin/python experiments/shared/dataset/t2bench.py
.venvs/expa-verl/bin/python experiments/shared/dataset/gen_sample_records.py --env t2bench
.venvs/expa-verl/bin/python experiments/shared/dataset/gen_sample_records.py --env t2bench --check
```

The default source is `repos/t2bench/`. Source/output paths are CLI-configurable; see `--help`.
Existing snapshots are protected; regeneration requires explicit `--overwrite`.

| Dataset | Official split | Domains | Expected count for this version |
|---|---|---|---:|
| t2bench | base | Retail / Airline / Telecom | 114 / 50 / 114 |

t2bench `base` follows official split membership, not all of `tasks.json` or the official `test` split.
Telecom full, Mock, Banking and Telecom-workflow are excluded by default.
This t2bench revision includes tau3 updates; it is not the original tau2 paper dataset.
Some tasks descend from old tau-bench domains. Do not sum historical counts as independent tasks.
Each `dataset/manifest.json` records the actual source revision, input digests and exported counts.

Records retain benchmark, domain, official split, original task ID, cross-domain unique ID and full task JSON.
`task_json` is evaluation-side data containing private simulator instructions, reference actions, scoring information and optional initial state; it is not an agent prompt.
Exports therefore contain no fabricated prompt or training-reward columns, and examples do not tokenize hidden tasks as prompts.
Evaluation still requires matching official policies, tools, databases and an independent user simulator. A snapshot is not a self-contained environment.
