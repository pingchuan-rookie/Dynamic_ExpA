# Agentic RL analysis

`run/` contains run summaries, trajectory decoding and diagnostics shared by the training methods.
Dyad-specific experiment reports and encoder/projector tools live in `experiments/dyad_training/agentic_rl/analysis/{experiment,model}/`.

```text
shared/analysis/run/
├── summarize_training.py
├── decode_trajectories.py
├── verify_dyad.py
├── verify_grpo_react.py
├── _env_checks.py
└── _mask_checks.py
```

## Inspect a run

Run from `dynamic-expa/` using `.venvs/expa-verl/bin/python`.
The examples use `dyad-grpo`; for GiGPO, use the corresponding `dyad-gigpo` path. Both use `verify_dyad.py`.
Replace `<run>` below with the actual directory:

```bash
.venvs/expa-verl/bin/python experiments/shared/analysis/run/summarize_training.py /absolute/path/to/run
.venvs/expa-verl/bin/python experiments/shared/analysis/run/decode_trajectories.py /absolute/path/to/run --show 3
.venvs/expa-verl/bin/python experiments/shared/analysis/run/verify_dyad.py /absolute/path/to/run
```

- **Summary** reads trainer logs and diagnostic events and prints training metrics; it does not establish complete correctness.
- **Decoding** defaults to a turn-based view. `--view aligned` shows token alignment, and `--out` selects all trajectories.
Views share `<run>/analysis/decode_trajectories.md`; rerunning overwrites that file.
- **Verification** selects `verify_*.py` for the run's algorithm and writes `<run>/analysis/verify_<algo>.md`.
Conclusions cover only recorded evidence. Missing evidence, nonzero gradients or a passing script do not establish full algorithm correctness.

Shared-step evaluation reads native t2bench/SWE `summary.json` files and action audits, including configurations saved as Hydra literals.
Reports distinguish officially scored and unscored terminal states, tool responses and reported call counts. Submitted action text is not proof of tool execution, and a native tool error is not task success.
A terminal summary alone cannot establish missing pool lifecycles, step rewards or gradients. Native audit views truncate long prompts and outputs by default; `--out` shows full records. This view does not validate token replay.
Native t2bench and SWE evaluation does not write diagnostic event streams. With no `*_pid*.jsonl` files, `verify_dyad.py` marks action-head initialization UNDECIDABLE.
If an event stream exists but lacks `action_head_initialized`, the verdict remains FAIL.

After a tool executes, the loop may stop if appending its response would exceed the response budget; that response then never enters the token sequence.
ReAct and Dyad observation-mask checks classify such trajectories as "observation not retained" only when response counts match an independent single-generation count and a complete all-one mask or single GEN span supports the conclusion. Proximity to the length limit alone is insufficient.
Missing or contradictory records cannot use this exception. Retained tool tokens must still have zero masks.
`run/_mask_checks.py` shares this criterion without changing training behavior.

Decoding and verification share path resolution and report writing from `run/decode_trajectories.py`.
If the run directory is omitted, the latest diagnostic run under the project's `outputs/<site>/agentic_rl/` is selected independently of the working directory and without filtering by algorithm.
Always pass a specific run when analyzing a particular algorithm.

`train_eval/scripts/run.py` invokes the appropriate verifier and decoder automatically after a debug run.
Analysis exit codes are recorded in `<run>/analysis/post_analysis.status` without replacing the training exit code. `POST_ANALYSIS=0` disables automatic analysis.

## Generate an experiment report

[experiment/generate_report.py](../../dyad_training/agentic_rl/analysis/experiment/generate_report.py) reads the [main experiment configuration](../../dyad_training/agentic_rl/train_eval/config/experiments.yaml), [ablation definitions](../../dyad_training/agentic_rl/train_eval/ablation/README.md) and selected runs to assemble evidence about encoder inputs, projectors and training scope.
It is separate from a single-run metric summary.

```bash
.venvs/expa-verl/bin/python experiments/dyad_training/agentic_rl/analysis/experiment/generate_report.py --experiment train --env gsm8k --run-dir /absolute/path/to/run
```

Without an explicit run, it finds the latest matching experiment name in `resolved_config.json`. Specify older runs explicitly when their metadata is missing.
Reports print by default; `--out-dir <run>/analysis` writes `<out-dir>/<experiment>/<env>.md`.

## Model tools

[model/dump_encoder_inputs.py](../../dyad_training/agentic_rl/analysis/model/dump_encoder_inputs.py) requires no training run. It generates input text for the schema collection declared in the script and writes it under the project's `outputs/descriptions/`.
`--only` selects targets, `--form` selects the description form, and `--model` selects the tokenizer.
This shows the chosen schemas, not necessarily the actual inputs of a particular run or every CodeGym task.

[model/extract_agentic_rl_projector.py](../../dyad_training/agentic_rl/analysis/model/extract_agentic_rl_projector.py) reads `--agentic_rl-ckpt`, takes compatibility metadata from `--from-alignment` and writes the projector to `--out`.
This exports initialization weights; it does not restore full training state.

```bash
.venvs/expa-verl/bin/python experiments/dyad_training/agentic_rl/analysis/model/dump_encoder_inputs.py --help
.venvs/expa-verl/bin/python experiments/dyad_training/agentic_rl/analysis/model/extract_agentic_rl_projector.py --help
```

Generated artifacts do not belong in source directories.
Cross-check original trajectories, task identity and run configuration, distinguishing valid scores from incomplete records.
