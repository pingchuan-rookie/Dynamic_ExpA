# Experiment entrypoints

Source is organized by method and shared responsibility. Datasets and run artifacts are stored separately.

| Directory | Responsibility |
|---|---|
| [action_encoder_alignment_dataset](action_encoder_alignment_dataset/README.md) | Action Encoder Alignment data preparation and validation |
| [dyad_training/action_encoder_alignment](dyad_training/action_encoder_alignment/README.md) | Action Encoder Alignment training and evaluation |
| `dyad_training/agentic_rl/` | Dyad entrypoints, main experiment recipes, ablations and model analysis |
| `grpo_training/train_eval/train.sh` | Text GRPO entrypoint |
| `gigpo_training/train_eval/train.sh` | Text GiGPO entrypoint |
| [shared/train_eval](shared/train_eval/README.md) | Shared environment configuration, argument parsing, runners and evaluation |
| [shared/dataset](shared/dataset/README.md) | Task data, native prompts and environment assets shared by all methods |
| [shared/analysis](shared/analysis/README.md) | Shared run diagnostics and trajectory analysis |
| [capability_eval](capability_eval/README.md) | Single-turn capability evaluation |

Run the method entrypoints from `dynamic-expa/`:

```bash
bash experiments/dyad_training/agentic_rl/train_eval/train.sh alfworld dyad-grpo --help
bash experiments/dyad_training/agentic_rl/train_eval/train.sh alfworld dyad-gigpo --help
bash experiments/grpo_training/train_eval/train.sh alfworld --help
bash experiments/gigpo_training/train_eval/train.sh alfworld --help
bash experiments/shared/train_eval/evaluate.sh --help
```

Each method entrypoint rejects algorithms outside its scope and reuses the shared defaults and configuration.
Alignment data uses `data/actenc_alignment`; task data is shared in `data/<env>`. Evaluation-only GSM8K data is in `data/gsm8k`.
Existing site, stage and algorithm identities under `outputs/` and `ckpt/` remain unchanged; historical records are preserved.
