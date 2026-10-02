# Ablation experiments

Each experiment has one YAML file, whose stem is the `--experiment` value.
Edit that file to change its conditions. Shared baseline settings are in `main.defaults` of `../config/experiments.yaml`; each ablation records only its differences.

| File | Main change |
|---|---|
| `2.1_description_natural_language.yaml` | MCP descriptions → natural-language descriptions |
| `2.2_projector_mean.yaml` | Attention → mean pooling, with encoder-backbone training retained |
| `2.3_frozen_encoder_llm.yaml` | Freeze the encoder backbone while training the policy and projector |
| `2.4a_policy_lm_backbone_attention.yaml` | Use policy-LM action representations with attention pooling |
| `2.4b_policy_lm_backbone_mean.yaml` | Use policy-LM action representations with mean pooling |
| `2.6_values_closed.yaml` | Select ALFWorld argument values from fixed sets |
| `2.7_frozen_llm_adaptation.yaml` | Train only the encoder side |
| `2.8_policy_lm_only.yaml` | Train only the policy LM |

```bash
# Run from the repository root.
bash experiments/shared/train_eval/train.sh alfworld --experiment 2.1_description_natural_language
bash experiments/shared/train_eval/train.sh alfworld --experiment 2.7_frozen_llm_adaptation --debug
bash experiments/shared/train_eval/train.sh alfworld --experiment 2.6_values_closed --check
```

Each YAML contains:

- `title`, `description`: experiment name, purpose and coupled changes to consider.
- `benchmarks`: supported environments; unsupported combinations fail explicitly.
- `overrides`: model or training conditions that differ from the main experiment.
- `parameters` (optional): batch sizes, learning rates and other run settings specific to this ablation.

Model/GPU-specific parameters remain under
`models.<model>.hardware.<hardware>.experiments.<experiment>` in `experiments/shared/train_eval/config/<benchmark>.yaml` and override this file's `parameters`.
Adding an ablation requires only a new YAML file: the loader discovers it without an extra registry entry or copied training script.

```text
ablation/<experiment>.yaml ── experiment differences ─┐
config/experiments.yaml ───── baseline defaults ───────┼── scripts/prepare.py → scripts/run.py → scripts/training.sh
train.sh --experiment <experiment> ──────────────────┘
```

`evaluate.sh` runs benchmark evaluation only and does not launch ablation training.
Historical ablation 2.5, the kv/stmt surface-form experiment, is outside the supported set and has not been restored as a runnable experiment. Supported action spaces are described in [shared configuration](../../../../shared/train_eval/config/README.md).
