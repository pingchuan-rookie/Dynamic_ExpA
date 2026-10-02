# Alignment configuration

A single training task is configured through `actenc_alignment_training.yaml`, without benchmark or algorithm branches.
Training updates on train, validates and selects checkpoints on val, and does not score test automatically.
The YAML `DATASET` field defines the default data file; an environment variable with the same name can explicitly select data that follows the split protocol.
Standalone `actenc_alignment_evaluate.sh` restores `model_config` from the selected checkpoint's `config.yaml` rather than reconstructing the model from training defaults here.

```text
Default model: default_model
Effective parameters: defaults
                      → hardware[hardware]
                      → models[model].defaults (optional)
                      → models[model].hardware[hardware]
                      → debug (--debug only)
                      → explicit environment variables
```

`models` defines Hugging Face paths and supported hardware combinations.
All models, hardware profiles and debug runs share `defaults.BATCH_SIZE`; environment overrides that change it are rejected.
Memory differences between models or hardware affect only `MICRO_BATCH_SIZE`, the per-replica sample limit for one forward/backward pass.

Each update divides the global batch equally across replicas, accumulates micro-batch gradients, then synchronizes, clips and updates once.
The last micro-batch may be smaller, with loss weighted by its actual sample count. Each epoch drops an incomplete global batch and reshuffles for the next epoch.
Data smaller than one complete update causes an error.
With the same data, seed and epochs, all models and replica counts use the same global sample groups and update count.

Unsupported model/hardware combinations fail explicitly; a configuration entry is not proof of measured memory capacity.
`resolved_config.json` records per-replica batch size, micro-batch size and accumulation count. Training `config.yaml` also records the global batch and samples dropped per epoch.

`--model` accepts a model key or configured HF ID; the GPU is detected if hardware is unspecified.
`POLICY_MODEL`, `SCALE_PROFILE` and `RUN_IS_DEBUG` remain supported.
Debug applies overrides without replacing the selected model.

`BATCH_SIZE` is the global training batch; `EVAL_BATCH_SIZE` is the per-replica evaluation batch.
Both LMs are independently frozen during training. Projector, scale and other experiment settings are stored in the same file.

Public training configuration supports Qwen3.5 2B, 4B, 9B and 27B, with 2B as the default.
Each model/hardware combination defines its training micro-batch, evaluation batch and `ENCODER_BATCH_SIZE`, the action-prompt limit per encoder forward pass.
Candidates are deduplicated within each micro-batch, then encoded and projected in bounded chunks while retaining every candidate and gradient.
These settings are starting points; throughput and peak memory must be measured on the target GPU.
