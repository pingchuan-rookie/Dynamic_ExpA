# Agent system

The shared interaction system serves both text policies and Dyad; training algorithms use the same environment loop.
Dyad is the extended-action policy in `policies/dyad/`; the surrounding runtime is shared.

## Responsibilities

| Directory | Contents |
|---|---|
| `environments/` | Environment pools, workers, sessions, prompts/history, execution and feedback |
| [`inference/`](inference/README.md) | vLLM-powered serving, model restoration and per-request action definitions |
| `evaluation/` | Shared environment sessions driven through inference APIs, evaluation records and summaries |
| `rollout/` | Shared environment-decision collection, session helpers and lazy collector selection |
| `parsers/` | Shared text-protocol parsers loaded on demand |
| `policies/` | Policy interfaces and assembly; `text.py` provides the text policy and `dyad/` the extended-action policy |
| `rewards/` | Environment scoring integration |
| `utils/` | Shared model configuration, generation constraints, diagnostics and artifact paths |

`policies/dyad/` contains action schemas, candidate encoding, action heads, routers, generation backends, decision traces, probability replay and policy-specific training integration.
Its `rollout/` directory contains model-generation backends; the environment interaction loop remains shared.
Alignment projector training is implemented in `policies/dyad/training/action_encoder_alignment/`; experiment recipes and dataset preparation remain in `experiments/`.

GRPO/GiGPO advantages, statistical occurrences, mini-batch dispatch and loss reduction live in `verl_extensions/agent_steps/`.
Training stages in [`verl_extensions/trainer/`](../verl_extensions/trainer/) are called explicitly from the original stage methods in `verl/`; `fit → step → _step_once` remains the shared training flow.
`rollout/tool_loop.py`, `tool_state.py` and `scoring.py` contain the environment-session, native-protocol and reward implementations extracted from the upstream tool loop.
Training-side step components consume real step records without reading action schemas or executing environments.
The shared collector obtains samples and traces through the policy interface. Policies are selected lazily, so the text path does not load Dyad models.
Environments report observations, rewards, validity and execution status independently of the advantage algorithm.
[Environment integration](environments/README.md) is grouped by environment in `environments/backends/`. Shared base classes are in `environments/core/`; registration and step sessions remain at the environment package root.
`environments/prompts/` and `environments/configs/` retain their shared resource paths. Package initialization does not eagerly load environment or model dependencies.
Each environment implements its model API needs in `environments/env_package/<environment>/`: DIVE owns judge and browse calls, while t2bench owns user simulation and judging.

## Correctness and compatibility

All four training combinations share environments, native prompts/history and rewards. Switching advantage estimators preserves the interaction flow.
Training replay must retain Dyad action selections, legal candidates and forced-token masks; rendered text alone is insufficient.

Current training uses shared step v2 and verl V1 sync. Legacy trajectory and GiGPO step v1 collectors, the old Dyad trainer and old advantage implementations have been removed.
Checkpoints from older interaction protocols cannot resume full training state or evaluate with the original protocol. Reusing their weights requires an explicit model-only import into a new step v2 run.
`compat.py` resolves legacy module and runtime-resource paths at configuration load time. It leaves historical checkpoints, data and logs intact and does not create dynamic import aliases for old packages.
Relocating implementations preserves the `dyad` policy identity, `dyadvllm` rollout name, saved advantage algorithm and training protocol.
See [training and evaluation](../experiments/shared/train_eval/README.md) for usage and [contribution guidelines](../CONTRIBUTING.md) for source checks.
