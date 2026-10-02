# Action Encoder Alignment training

`actenc_alignment_train.sh` trains only the projector and selects a checkpoint using val. `actenc_alignment_evaluate.sh` explicitly evaluates the full test split for one saved checkpoint.
The data protocol is `seen-train-unseen-val-test-v1`, with `train`, `val` and `test` splits.
The old train split is unchanged. The old test split is divided equally into stratified val/test cases, without adding actions or calling the teacher.
Legacy two-split data must first be migrated to a new version; test must never silently become val.
`config/actenc_alignment_training.yaml` defines the default dataset path; `DATASET` explicitly selects a data file.
`DATASET=/absolute/path/to/new-version/final/dataset.parquet` can still select another dataset path explicitly.

The data manifest defines candidate sets and case assignments for val/test; evaluation verifies this split identity.
Training selects checkpoints using val only. Standalone evaluation rejects checkpoints that do not match the selected data protocol.

`EVAL_EVERY` in `config/actenc_alignment_training.yaml` controls the evaluation interval, measured in optimizer updates.
Step 0 and periodic evaluation are retained. The final step is evaluated if it falls between evaluation boundaries; otherwise its existing result is reused.
Select the lowest val CE, keeping the earlier checkpoint on ties, including step 0. Strictly restore the selected head before saving.
Missing, NaN or infinite val CE fails immediately: that validation point is not published and no incorrectly selected checkpoint is saved. Saving is also rejected if no valid best checkpoint exists.
Models share the formal default interval; debug uses its own configuration, and explicit environment variables may override it.
New training `metrics.jsonl` records use `val`. In `summary.json`, `val`, `val_by_form` and `val_by_size` describe the selected weights, while `val_at_last_step` describes the final step.
Local and W&B records explicitly retain `split_policy` and `evaluation_role=train-validation`; curves, replay and summaries use `val/*`.
Standalone test evaluation uses the same data protocol with `evaluation_role=final-test` and publishes `test/*`, without remapping it to val.
Historical test records without a protocol marker retain the old test->val reading rule; historical val/unseen names remain unchanged. New val records do not require an unseen field.
Existing misnamed online curves are not migrated automatically. Do not mix new-protocol and old runs or rewrite historical files and online runs.

```text
actenc_alignment_train.sh                          Public entrypoint: --model / --hardware / --debug
   |
   v
scripts/actenc_alignment_prepare.py                Read config/actenc_alignment_training.yaml; prepare paths and check data, models, GPUs and W&B
   |
   v
scripts/actenc_alignment_run.py                    Save configuration/logs and manage the training subprocess
   |
   v
scripts/actenc_alignment_training.sh               Final Python / torchrun command
   |
   v
agent_system/policies/dyad/training/action_encoder_alignment/actenc_alignment_train.py              Freeze both LMs, train the projector, select and save the checkpoint
```

Launch from `dynamic-expa/`:

```bash
bash experiments/dyad_training/action_encoder_alignment/train_eval/actenc_alignment_train.sh --model qwen3.5_2b --hardware h100
bash experiments/dyad_training/action_encoder_alignment/train_eval/actenc_alignment_train.sh --model qwen3.5_2b --hardware a6000 --debug
CUDA_VISIBLE_DEVICES=0,1 LIMIT=192 bash experiments/dyad_training/action_encoder_alignment/train_eval/actenc_alignment_train.sh --model qwen3.5_2b --hardware a6000 --debug
bash experiments/dyad_training/action_encoder_alignment/train_eval/actenc_alignment_train.sh --model qwen3.5_4b --hardware h200 --dry-run
bash experiments/dyad_training/action_encoder_alignment/train_eval/actenc_alignment_train.sh --model qwen3.5_4b --hardware h200 --check
```

`--dry-run` prints configuration only; `--check` checks inputs and resources without starting training.
`--debug` retains the model while reducing micro-batch size, epochs and samples. It preserves the global batch per update and records a local offline W&B run.
With the current global-batch configuration, the Qwen3.5-2B example above completes three optimizer steps. `LIMIT` independently limits train and val; test is not loaded for training-time scoring. Select two GPUs available on the local machine before running.
`BATCH_SIZE` is global. Each replica uses two GPUs; the launcher derives per-replica batches and accumulates `MICRO_BATCH_SIZE` forward/backward passes into a full update. An incomplete global batch at epoch end is dropped.
`CUDA_VISIBLE_DEVICES` limits visible GPUs; `REPLICAS` limits replica count.

```text
${ARTIFACT_ROOT}/
  outputs/<site>/alignment/<run>/   Configuration, metrics, summaries, logs and W&B
  ckpt/<site>/alignment/<run>/      projector.pt、config.yaml、metrics.jsonl、summary.json
```

`ARTIFACT_ROOT` must be absolute and defaults to the repository's `artifacts/` directory.
Configuration, metrics and final summaries are saved with the checkpoint; logs and W&B records are in the output directory.
`MICRO_BATCH_SIZE`, `ENCODER_BATCH_SIZE` and `EVAL_BATCH_SIZE` control per-call training, candidate-encoding and evaluation sizes.
They do not change the required global batch. See [configuration](config/README.md) for other override precedence.

## Standalone test evaluation

Training, debug and upload workflows never score test automatically.
Fix the configuration and checkpoint before explicitly running the commands below. Do not repeatedly inspect the formal test split to tune settings.

```bash
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PWD/artifacts}"
bash experiments/dyad_training/action_encoder_alignment/train_eval/actenc_alignment_evaluate.sh \
  --checkpoint "$ARTIFACT_ROOT/ckpt/local/alignment/<train-run>/projector.pt" \
  --dataset /absolute/path/to/new-version/final/dataset.parquet \
  --out "$ARTIFACT_ROOT/outputs/local/alignment_test/<evaluation-run>"
```

Architecture, representations, normalization, dtype and length limits are restored from `config.yaml` beside the checkpoint, with strict weight loading.
The dataset SHA256 must match the version fixed during training. Paths may move, but the split cannot change.
Only `--policy-device`, `--encoder-device` and `--eval-batch-size` execution settings may be overridden; sample limits are not accepted.
Evaluation does not create an optimizer, select weights or overwrite training artifacts or an existing output directory.
`summary.json` records overall CE, top1, demonstrated-action probability, chance baseline, n, candidate counts, form/size/domain/action groups and dataset/checkpoint SHA256 digests.
It also writes `config.yaml` and a single-point `metrics.jsonl` compatible with the existing upload tools.
Without `--out`, standalone evaluation uses `outputs/<site>/alignment_test/<checkpoint-run>_<time>` under the selected artifact root. A historical checkpoint's location does not choose the new output root.

Use the result-recording tools as needed to upload and verify standalone evaluation metrics.
The commands below publish saved reports only; they neither rescore data nor rewrite source files.

```bash
python experiments/dyad_training/action_encoder_alignment/analysis/actenc_alignment_upload_wandb.py \
  --runs-dir "$ARTIFACT_ROOT/outputs/local/alignment_test" --run evaluation-run
python experiments/dyad_training/action_encoder_alignment/analysis/actenc_alignment_verify_wandb.py \
  --runs-dir "$ARTIFACT_ROOT/outputs/local/alignment_test" --run evaluation-run
```

Use a separate, unique evaluation-run name, without sharing the training run's W&B ID.
Upload scripts explicitly use online mode; these commands are not offline development checks.

Explicit environment variables still override YAML, for example `LR=2e-4 MICRO_BATCH_SIZE=8`. `BATCH_SIZE` accepts only the shared configured value.
`RUN_NAME` must be a single directory name. Existing nonempty directories are rejected unless explicitly reused with `FORCE=1`.

`actenc_alignment_training.yaml` defines memory settings per model and hardware profile.
`MICRO_BATCH_SIZE` controls training samples per replica; `ENCODER_BATCH_SIZE` limits action prompts per encoding call. Neither changes the global update budget. `EVAL_BATCH_SIZE` independently limits evaluation samples.
`--hardware a100` targets 80GB A100 devices. All four configured models have entries, initially using the corresponding H100 batch-capacity settings.
Four GPUs default to two replicas, each processing 32 samples per update, preserving global batch 64. Each LM still occupies one GPU; models are not sharded across all four.
Measure training/evaluation peak memory and throughput on the target GPU. The A100/H100/H200 profiles have not been capacity-tested and are not capacity guarantees.
