# Action Encoder Alignment

Action Encoder Alignment trains the projector that maps action-encoder representations into
policy-space action embeddings. The policy LM and the independent encoder backbone remain frozen.
The objective is to select the demonstrated action from the complete displayed candidate set.

## Implementation

| Component | Location |
|---|---|
| Data generation, splitting and validation | [action_encoder_alignment_dataset](../../action_encoder_alignment_dataset/README.md) |
| Training and independent evaluation entrypoints | [train_eval](train_eval/README.md) |
| Model and hardware configuration | [train_eval/config](train_eval/config/README.md) |
| Training implementation | [action_encoder_alignment](../../../agent_system/policies/dyad/training/action_encoder_alignment/) |
| Encoder and projector | [Dyad models](../../../agent_system/policies/dyad/models/) |
| Prepared-data readers | [Dyad data](../../../agent_system/policies/dyad/data/) |

## Dataset contract

The schema uses `action_set` as the sole ordered candidate list. `action_definitions` and
`action_encoder_prompts` are mappings keyed by action name and must cover every candidate.
The scorer follows list order when constructing encoder inputs and logits; mapping-key order
does not affect the candidate order.

Every displayed action participates in the softmax. The scorer does not draw a second subset
or silently truncate candidate prompts. Paired MCP and natural-language records share their
case, label, candidate order and split.

The `mcp-json-name-v1` policy prompt ends at the opening quote of the JSON `name` value after
the action marker. Natural-language prompts end at the marker. The policy predicts the action
name through the expanded head; this objective does not generate tool arguments.

The `seen-train-unseen-val-test-v1` split policy separates training, validation and final test
roles. Validation selects the checkpoint with minimum cross-entropy, retaining the earlier
checkpoint on ties. Test scoring is an explicit independent operation. Data manifests and
checkpoint metadata identify the split and prompt versions used by each run.

## Prepare and train

Configure the generation settings and teacher connection according to the
[data preparation guide](../../action_encoder_alignment_dataset/README.md), then run:

```bash
.venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_run_pipeline.py --all
.venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_validate_dataset.py

bash experiments/dyad_training/action_encoder_alignment/train_eval/actenc_alignment_train.sh \
  --model qwen3.5_4b --hardware h200 --check
```

Remove `--check` after preparing the required inputs. Model and hardware choices must exist
in the training configuration. Each replica uses two independently frozen LMs; micro-batch
size and replica count preserve the configured global batch.

Generated data and its provenance manifest are written to the configured data directory.
`ARTIFACT_ROOT` selects the training output root: `ckpt/` stores the projector and restoration
metadata, while `outputs/` stores configuration, metrics, summaries and logs.

See [training and evaluation usage](train_eval/README.md) for checkpoint selection,
independent test evaluation and metric-recording options.
