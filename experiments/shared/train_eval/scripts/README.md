# Agentic RL internal scripts

The public entrypoints are `train.sh` and `evaluate.sh` in the parent directory.
See the [parent README](../README.md) for the complete flow, usage and record locations.

| File | Responsibility |
|---|---|
| `training_config.sh` | Select a main experiment, baseline or ablation and prepare training configuration |
| `evaluation_config.sh` | Select evaluation weights and benchmarks without running a training experiment |
| `prepare.py` | Scale/debug settings, output paths, model configuration restoration and launch checks |
| `run.py` | Manage training Ray resources, model staging and analysis; delegate evaluation to its separate runner |
| `training.sh` | Assemble environment- and algorithm-specific verl arguments and launch training |
| `evaluation.sh` | Compatibility wrapper that does not assemble or launch verl |

Training commands are assembled by `training.sh --build`. Evaluation directly builds an `agent_system.evaluation.runner` configuration,
with generation provided by `agent_system.inference.server`; it neither sources `training.sh` nor passes `trainer.val_only`.

Launch checks are centralized in `prepare.py`. From the parent `train_eval/` directory, standalone diagnostics use:

```bash
python scripts/prepare.py run std --depth light
python scripts/prepare.py run calc dyad --depth full
python scripts/prepare.py projector --projector-init /path/to/projector.pt --model-name Qwen3.5-2B
```

`--check` and `--dry-run` do not start Ray. Normal runs select check depth from their preparation configuration.
The `std` checks do not import Dyad environments, action encoders or projector modules.

Ablation purposes and parameter changes are documented in [Dyad ablations](../../../dyad_training/agentic_rl/train_eval/ablation/README.md), with one YAML per condition. This directory contains only shared configuration preparation and execution code.
