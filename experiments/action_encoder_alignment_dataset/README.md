# Action Encoder Alignment dataset preparation

This directory implements generation, validation, complete sample display and explicit holdout-split migration for the current data protocol.
Runtime data loading is in `agent_system/policies/dyad/data/`; training entrypoints are in [train_eval/](../dyad_training/action_encoder_alignment/train_eval/README.md).

## Current splits and interpretation limits

The current protocol is `seen-train-unseen-val-test-v1`. Rows retain schema v5; the `split` column in one Parquet file contains `train/val/test`.
Train is used for updates, val for periodic validation and minimum-CE checkpoint selection, and test only for explicitly requested final standalone evaluation.

The old test split is divided in half without adding actions, requesting the teacher or regenerating candidates, context, labels or prompts.
Stratify by `(domain, label, mcp_size)` with fixed `split_seed: 42`. Within each stratum, sort by `SHA256(UTF-8(str(seed) + ':' + case_id))`, breaking hash ties by case ID; assign the first half to val and retain the second half for test.
Each original stratum has two cases. MCP/NL rows for one case stay paired in the same split. All rows retain physical order and every field except `split`.
New generation reuses the same function and requires a positive even case count for each unseen-action stratum.

Val/test share the same 20 actions unseen relative to train, with one case per action at each candidate count from 4 through 10.
Neither split's labels may appear among any train candidates. Val/test actions are not isolated from each other, and distractors may mix seen and unseen actions.
The resulting metric measures held-out cases from the same unseen-action pool, not generalization to actions unseen relative to val.
New formal experiments should fix the split, train from scratch and select weights using val only. Weights selected on the complete old test set cannot support an independent-test claim.

## Migrate existing data offline

Run the explicit commands below from `dynamic-expa/`.
The destination root must not exist, contain the source directory or lie inside the source directory.

```bash
.venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_migrate_holdout.py \
  --source-root /absolute/path/to/legacy-alignment \
  --output-root /absolute/path/to/new-alignment-holdout \
  --seed 42

DYAD_ALIGNMENT_DATA="/absolute/path/to/new-alignment-holdout" \
  .venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_validate_dataset.py
```

Migration accepts only legacy two-split schema-v5 data with the `mcp-json-name-v1` prompt. It strictly checks train2100/test560 and two cases in each of the 140 configured unseen-action strata.
Write and validate in a separate temporary root before publishing the new directory. Existing destinations are rejected; source datasets, manifests, historical artifacts and shared defaults remain untouched.
The manifest records the full source SHA256, algorithm version, seed, zero teacher calls and historical usage.
`intermediate/holdout_source/` retains exact source-dataset and manifest snapshots for offline checks of original hashes, row contents and physical order.
Existing `prompt_revision` and `split_revision` history is preserved.
Rerunning migration when the new version already exists refuses to overwrite it; run validation directly instead.

Generation configuration and `DYAD_ALIGNMENT_DATA` determine the data root. Migration requires an explicit path to the old data
and a new destination; the data manifest records provenance, the split algorithm and artifact hashes.

## Generation and validation entrypoints

Use a separate `DYAD_ALIGNMENT_DATA` root for newly generated data as well.

```bash
.venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_run_pipeline.py --all
.venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_run_pipeline.py --step dataset
.venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_validate_dataset.py
.venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_gen_sample_records.py --check
```

`--all` calls the teacher. Use the migration entrypoint when changing only the split of existing data.
`--step dataset` reads existing catalogues, case plans and context/reasoning, then creates Parquet, manifests and sample documents offline without teacher requests.
Generation refuses to overwrite data with an old split policy or policy-prompt version.
`actenc_alignment_validate_dataset.py` checks the data contract, paired cases, stratified balance, candidate-level isolation, complete input reconstruction and artifact hashes. It does not run model scoring.
Migrated data additionally verifies source snapshots and that all non-split fields are unchanged.
`actenc_alignment_gen_sample_records.py` selects the first train MCP and first train NL row from actual Parquet, displays every field and encoder prompt, and records provenance, physical row numbers and hashes.
`--check` compares existing documents and exits nonzero on differences.

## Module responsibilities

| File | Responsibility |
|---|---|
| `actenc_alignment_run_pipeline.py` | Public generation CLI |
| `actenc_alignment_generate_dataset.py` | Build catalogues, case plans, context/reasoning and final data |
| `actenc_alignment_generation_config.py` | Configuration, paths and intermediate-artifact I/O |
| `actenc_alignment_balanced_sampling.py` | Balanced candidate-set and target-action sampling |
| `actenc_alignment_holdout_split.py` | Deterministic case holdout shared by generation and migration |
| `actenc_alignment_migrate_holdout.py` | Split old test data into a new directory and verify provenance |
| `actenc_alignment_prompt_rendering.py` | Render policy and encoder inputs |
| `actenc_alignment_parquet_writer.py` | Validate runtime schema and atomically write Parquet |
| `actenc_alignment_teacher_client.py` | Teacher requests, response validation, retries and caching |
| `actenc_alignment_validate_dataset.py` | D0-D14 data gates and holdout-provenance checks |
| `actenc_alignment_gen_sample_records.py` | Complete real-sample documents and consistency checks |
| `config/`, `prompts/` | Action names, generation configuration and teacher templates |

## Data contract and artifacts

```text
<data-root>/
├── final/
│   ├── dataset.parquet
│   └── SAMPLE_RECORDS.md
└── intermediate/
    ├── mcp.yaml / nl.yaml / mcp_unseen.yaml / nl_unseen.yaml
    ├── case_plan.jsonl / case_plan_unseen.jsonl
    ├── context_reasoning.jsonl / context_reasoning_unseen.jsonl
    ├── manifest.yaml
    ├── llm_cache/                 # May exist for generation; not copied during migration
    └── holdout_source/            # Explicit migration snapshots only
        ├── dataset.parquet
        └── manifest.yaml
```

`action_set` is the single ordered candidate list; every candidate requires its definition and encoder input.
Parquet metadata and the manifest record the new `split_policy`; `policy_prompt_version: mcp-json-name-v1` identifies input semantics in the manifest.
`load_split` supports the new three-way split. Legacy two-split data without val is rejected rather than automatically renaming test to val.
`read_dataset(..., expected_split_policy=LEGACY_SPLIT_POLICY)` is reserved for explicit offline migration. Older fields and schemas remain unsupported.
Runtime checks enforce disjoint case IDs across all splits, complete MCP/NL pairing, and exclusion of val/test labels from train candidates, even when a sample limit is requested.

## Action-selection boundary

MCP uses the `params.name` semantics of `tools/call` without generating a JSON-RPC transport envelope.
Tool catalogues retain their original representation; the call prefix is independent of `inputSchema`.

```text
<tool>{"name": "
```

Each sample uses its own marker in the actual prefix.
Policy input ends at the opening quote of the action-name string. The action head selects from all candidates rather than generating names or arguments token by token.
The format example shows only the `name` field. Example actions are sampled deterministically and uniformly per case, without always showing or excluding the label.
The natural-language format ends with the marker; encoder prompts contain the complete catalogue, target definition and instruction.
Model inputs must remain complete; scoring must not continue after silent truncation.
