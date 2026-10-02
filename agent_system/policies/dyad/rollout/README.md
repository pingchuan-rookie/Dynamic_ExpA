# Dyad rollout: constrained decoding and replay

This guide covers per-sequence state, logits masks and decision traces in the vLLM rollout backend. See [training](../training/) for probabilities and losses, and the [runtime guide](../../../README.md) for the call chain.

Reference: [Expanding the Action Space of LLMs to Reason Beyond Language](https://arxiv.org/pdf/2510.07581).

## Context tokens and policy decisions

The full action space combines vocabulary IDs `[0, V)` and expanded action IDs `[V, V+|E|)`. The action head assigns expanded IDs to structured actions and enumerated argument values.

Expanded IDs are policy labels, never embedding inputs. Each selection emits ordinary vocabulary tokens for its fixed surface form. The selection contributes one policy decision; subsequent deterministic tokens are forced writes and add no policy-loss terms. Generated argument values contribute their own vocabulary decisions.

Rollout therefore maintains two sequences:

- **Context:** ordinary tokens actually consumed by the model, including forced writes and observations.
- **Decisions:** sampled vocabulary or expanded IDs whose log-probabilities are trained.

Each decoding step obtains a candidate mask, emits context tokens through the router, and records sampled decisions in `action_content["raw_token_ids"]`.

## Action and argument structure

Schemas independently determine whether action names and arguments are selected by the head or generated as text. Supported combinations are listed in the [configuration guide](../../../../experiments/shared/train_eval/config/README.md).

The open-value `base` configuration selects an action name, then generates argument text. ALFWorld `base × closed` selects enumerated values in fixed slot order, so its arguments are ordered. CodeGym JSON templates include argument names while generating values from the vocabulary; visible argument names do not imply head selection.

Fully enumerated values concern arguments inside an action. Reasoning and action-entry markers can still contain vocabulary decisions.

## Modules

| Module | Responsibility |
|---|---|
| `../actions/schema_compiler.py` | Compile YAML into shared action configuration |
| `../actions/schema_config.py` | Load/cache schemas; `resolve_schema_path()` is the common schema locator |
| `../actions/schemas/` | Markers, actions, value sets, argument order and head settings |
| `../actions/action_router.py` | Shared `ActionRouter` for sampling and training replay |
| `vllm/dyad_gpu_model_runner.py` | Extend `GPUModelRunner.sample_tokens()` with router masks |
| `vllm/dyad_logits_processor.py` | Compute expanded logits from final hidden states and append them to base logits |
| `../models/action_head_factory.py` | Assemble materialized action heads |
| `../actions/policy_replay.py` | Dispatch training replay to `build_unified_policy_trace` |
| `vllm/dyad_core.py`, `dyad_outputs.py`, `dyad_core_client.py`, `dyad_core_engine.py`, `dyad_output_processor.py`, `dyad_async_llm.py` | Transport and assemble action traces across EngineCore |
| `agent_system/environments/` | Environment pools backed by Ray actors |

## Compiled configuration

`compile_action_schema(tokenizer, vocab_size, raw)` produces both legacy flat keys and the structured keys read by the router:

| Key | Meaning |
|---|---|
| `markers.enter_seq` | Action-entry token sequence |
| `markers.exit_seq` | Forced action-exit sequence |
| `markers.value_end_ids` | Tokens ending an open value, such as the next surface-form prefix |
| `markers.exit_value_end_ids` | Closing-prefix tokens such as `</` |
| `markers.turn_end_ids` | Assistant-turn closing tokens |
| `markers.max_value_tokens` | Open-value limit; zero disables the limit |
| `actions[name].surface_form_seq` | Fixed action prefix tokens |
| `actions[name].surface_form` | Literal prefix used in environment serialization |
| `actions[name].params[i]` | Slot value_kind, value_set, suffix tokens, argument_key_id and value_end_ids |
| `argument_order` | Fixed schema order or free argument text |
| `head` | action_name, argument_key, closed_value and init_from settings |
| `value_sets` | Expanded IDs for enumerated values |
| `id_to_str`, `id_to_seq`, `id_to_description` | Expanded-ID semantics, surface tokens and descriptions |
| `num_embeddings_size`, `total_size` | Base vocabulary size and expanded-space size |
| `router` | `"unified"`, selecting shared replay |

Action-name IDs are allocated before closed-value IDs. Their allocation order must match action-head rows, using identical compiled configuration in rollout and training.

`compile_multi_schema` offsets per-task schemas into disjoint global ID ranges. One head contains their concatenated rows; each sequence is masked to its own environment's range.

## Router and masks

Each sequence owns an `ActionRouter` in `dyad_gpu_model_runner.unified_routers[seq_key]`:

```python
router.decision() -> Decision
router.advance(raw_id) -> [int]
```

Entry moves from `NONE` to `ACTION_NAME`. Free argument text uses `FREE_TEXT`; fixed slots use `ARGUMENT_VALUE_OPEN` or `ARGUMENT_VALUE_CLOSED` until all slots are complete. The router then forces exit and EOS tokens and returns to `NONE`. Free text and open slots remain distinct to avoid unused slot state diverging between sampling and replay.

| Decision kind | Condition | Logits mask | Trained decision |
|---|---|---|---|
| `force` | Pending template, separator, exit or EOS token | Only the required token remains finite | No |
| `base_vocab` | Reasoning, open slot or free text | Mask `[V:]` | Vocabulary |
| `expanded` | Action name or closed value | Only current admissible IDs remain finite | Expanded action |

Masks are per decision: action names and successive slots can have different candidate sets.

### Multi-token markers

Markers depend on the tokenizer. For example:

```text
"<Action>"  -> [67424, 29]        '<Action' + '>'
"</Action>" -> [522, 2512, 29]    '</' + 'Action' + '>'
```

In `NONE`, `none_recent` tracks vocabulary decisions against `enter_seq`. A full match enters `ACTION_NAME`; when only the final marker token remains, it can be forced before entry to protect the boundary from BPE merging.

Sampled marker tokens are vocabulary policy decisions: `seq_mask=True`, `tool_mask=False`. Forced marker tails are excluded from decisions. After arguments finish, the router forces the schema's exit sequence and EOS.

### Open-value termination

Open values end when a slot's `value_end_ids` matches the next surface prefix, a closing prefix/newline appears, or `max_value_tokens` is reached. The length limit prevents indefinite generation if delimiters are missed.

Runtime guards reject a second head sample in a free-argument base call and require every decision to emit at least one ordinary token. Otherwise an expanded ID could enter `input_ids` and cause an out-of-range embedding lookup. Compilation also rejects empty action surface forms before GPU execution.

For example, selecting `goto` emits `go to`, then free text supplies `cabinet 1`; the environment receives `go to cabinet 1`. Closed-value ALFWorld instead selects container and index slots while producing the same native command format.

## Decision traces and transport

`_unified_advance` excludes forced writes and records sampled vocabulary/expanded IDs. Vocabulary decisions have `seq_mask=True, tool_mask=False`; expanded decisions have both masks true.

`add_action_content()` sends deltas on every valid decode step:

```python
output.action_content[str(seq_key)] = {
    "unified": True,
    "trace_delta": True,
    "raw_offset": offset,
    "raw_token_ids": delta,
    "action_complete": complete,
    # The first delta also carries action_config.
}
```

Forced-write deltas are empty. Discarded prefill samples neither advance the router nor publish traces. `raw_offset` locates each delta in the cumulative decision sequence.

Clear `unified_decisions[seq]` only after a completed action: the router is back in `NONE`, no pending tokens/recent marker prefix remain, and an expanded decision exists. A partial multi-token marker is not a completion boundary.

EOS or length limits may stop an action midway. Return its accumulated trace for training but never execute incomplete arguments. Only `action_complete=True` triggers tool parsing. Valid pure-vocabulary output without action_content can use `build_vocab_policy_trace`; missing expanded decisions or replay misalignment must fail rather than fall back.

EngineCore fields pass through msgspec and zmq. The core slices by client/request, output classes preserve fields, and the output processor accumulates deltas while rejecting missing/duplicate offsets. Final payloads attach to `RequestOutput` and return through async LLM/server code. Request completion cleans per-sequence state using `finished_req_ids`.

## Sampling and replay must agree

Sampling and log-probability computation must use identical candidate restrictions. Both call the same `ActionRouter.decision()`:

```text
rollout: dyad_gpu_model_runner.sample_tokens
replay:  router.build_unified_policy_trace
```

Replay scans `raw_token_ids`, obtains the current decision, records expanded IDs with `allowed_action_ids=d.allowed_ids` or aligns vocabulary decisions to context tokens, then advances the router. Forced tokens occupy context positions without decision steps.

The agent loop expands allowed IDs into `dyad_action_mask[B,R,|E|]`. Modify candidate rules in the shared router; runner-only exceptions can invalidate PPO ratios. Training guards in `split_policy.compute_split_policy_outputs` check these constraints, including `Dyad selected action label is not enabled by dyad_action_mask`.

## Action-head synchronization

Agentic RL uses the direct action encoder assembled through `models/action_head_factory.py`. Rollout holds a projector and synchronized encoder hidden/mask tensors without another encoder backbone. Its action head contains materialized sampling weights.

The `dyad_residual_head.*` namespace remains a persistence/transport contract, not a residual algorithm. Receive projector parameters and `encoder_hidden`/`encoder_mask` before rebuilding the head. Do not load streamed `action_head.*` directly, as those materialized weights can be stale. The retained `reinit_action_head_from_lm_head()` name now rebuilds from synchronized encoder/projector state.

A logits processor allocated before synchronization remains an unready placeholder. Fresh and resumed runs must complete synchronization/rebuilding before sampling, without a mean-pool fallback. Rollout and its corresponding log-probability pass use the same projector/cache version without intervening updates.

Cross-environment validation uses the same builder, with target-schema encoder data supplied by the caller, and restores the training schema/head afterwards.

## Diagnostics and limitations

`_build_logits_seq_keys()` maps logits rows to request IDs through `num_scheduled_tokens`. Unsupported speculative decoding, prompt-logprob layouts or shared request IDs across completions must fail when the mapping is not one-to-one. Use the phase enum belonging to each state machine.

Events under `log_event("dyad_vllm_model_runner", ...)` include:

- `action_head_initialized`: allocation/rebuild parameter summaries; allocation alone does not mean sampling is ready.
- `action_decision_or_forced_token`: raw/emitted token IDs and phase.

See the [analysis guide](../../../../experiments/shared/analysis/README.md), [training guide](../../../../experiments/shared/train_eval/README.md) and [ActionRouter](../actions/action_router.py) for diagnostics, artifacts and exact transition rules.
