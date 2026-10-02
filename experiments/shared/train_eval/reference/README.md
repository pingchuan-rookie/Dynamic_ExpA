# Standalone reference evaluation

This directory contains explicitly selected ALFWorld and t2bench reference evaluators, policy clients and evaluation services.
Current training and unified environment evaluation use the public entrypoints in `experiments/shared/train_eval/`.

- `alfworld/` contains the HTTP environment service, ReAct baseline, evaluation using the official ReAct protocol and mapping tools. It is separate from the Ray pools used for current training.
- `t2bench/` contains standalone ReAct/Dyad clients, argument parsing and result summaries, selected explicitly with `--backend reference`.

Reference evaluators reuse environment semantics from `agent_system/environments/env_package/`; environment packages do not import these evaluators.
These entrypoints require their environment dependencies, task data and explicitly configured model services. Relocating the files does not establish that a full evaluation has been run.
