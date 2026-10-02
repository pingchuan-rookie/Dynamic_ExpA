# Dyad policy

Dyad implements extended-action representations, selection and probabilities. The environment loop and choice of GRPO or GiGPO remain outside the policy.
See [agent system](../../README.md) for shared responsibilities and training-side boundaries.

| Directory | Responsibility |
|---|---|
| `actions/` | Schemas, legal candidates, routing and decision replay shared by sampling and training |
| `models/` | Action encoders, projectors, action heads and policy forward passes |
| `algorithms/` | Action selection, split-policy probabilities and separate gradient paths |
| `data/` | Loading and validating prepared Alignment data; dataset creation lives outside this package |
| `rollout/` | verl/vLLM generation integration, constrained decoding, encoder workers and weight synchronization |
| `training/` | Dyad engines/workers, PPO loss integration and Alignment projector training |
| `inference/` | Native Dyad checkpoint restoration and action-head loading; shared APIs live in `agent_system/inference/` |

`policy.py` implements the shared step-policy interface, sampling context and strict trace validation.
An expanded action may render as multiple context tokens but counts as one sampled decision.
Training replay must verify consistency of legal candidates, sampled labels, text and masks; it must not substitute ordinary vocabulary probabilities.

## Action protocol

New ALFWorld/WebShop ReAct runs use `<action>...</action>` and allow reasoning in `<think>...</think>`, with the same prompts as the text baseline.
Qwen3.5 consistently disables thinking while retaining the template's empty, closed thinking prefill.
The default lowercase schema is `shared_step_v2.yaml`; ALFWorld's closed-set variant is `shared_step_v2_closed.yaml`.
`base.yaml` and `base_closed.yaml` support model-schema identity checks and explicit weight imports. They do not provide entrypoints for running the legacy interaction protocol.
DIVE and CodeGym retain their native tool-call protocols.
Multi-token text start markers use compiled, bounded character-prefix states, allowing different BPE sequences to represent the same marker. Sampling and replay share the state machine without retokenizing or rewriting sampled tokens.
A vocabulary token that already contains both the complete marker and an action name must cause an error; an action-head selection cannot be fabricated afterward. Native special-token tool protocols do not use this text matcher.

## Entrypoints and limitations

Public training entrypoints remain in [experiments/dyad_training/](../../../experiments/dyad_training/).
The low-level Alignment entrypoint is `python -m agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_train`.
Agentic RL's shared step mode delegates to verl V1 sync without duplicating the general training loop.
Migration of `rollout/async_policy/` is incomplete; its entrypoints continue to reject execution explicitly.
The legacy trajectory collector, trainer and reward compatibility wrappers have been removed.

Moving Python packages does not change model, action-field or checkpoint identities.
See the [repository README](../../../README.md) for the method overview and this package's modules for implementations and entrypoints.

Action heads for long descriptions project independent action rows in chunks and transfer frozen encoder-cache chunks to the device. Training recomputes projector intermediates while preserving parameter gradients. Every action and complete description participates; chunking does not shorten inputs or change legal candidates.
`verl_extensions.vocab_statistics` computes full-vocabulary normalization by token row. Packed baseline/reference paths and Dyad share bounded workspace; this PPO boundary supports first-order gradients only.
Before constructing action heads, the action forward hook releases unused per-layer hidden-state outputs, retaining the final layer and its autograd path. Long inputs can use `actor_rollout_ref.model.enable_activation_offload=True` to trade CPU transfers/recomputation for device memory. Component peak memory does not establish full-training capacity; full input lengths still require hardware-specific measurement.

The packed Dyad path can use model-provided fused vocabulary scoring with `actor_rollout_ref.model.use_fused_kernels=True` and `actor_rollout_ref.model.fused_kernel_options.impl_backend=torch`, avoiding full context-by-vocabulary logits. Actions retain separate FP32 softmax normalization, and mismatched legal candidates or replay/KL labels still fail. The encoder branch receives detached vocabulary statistics, preserving separate gradient paths. This mode inherits the upstream single-temperature restriction, does not support `sum_pi_squared`, and requires the model to return the needed log-probabilities/entropy and final-layer hidden states. The BF16 fused path uses upstream FP32 vocabulary normalization and entropy computation; bitwise equality with eager BF16 entropy is not claimed.
When entropy_coeff=0, entropy metrics are retained without constructing a zero-weight entropy backward path; nonzero entropy regularization is unchanged. Fused scoring may still require activation offload: removing the full logits allocation does not guarantee that all training intermediates fit in memory.
