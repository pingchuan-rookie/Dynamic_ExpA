#!/usr/bin/env bash
# Final training command; configuration and checks live in actenc_alignment_prepare.py.
set -euo pipefail
cd "${PROJECT}"
LAUNCH=("${PYTHON}" -m agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_train)
if [ "${REPLICAS}" -gt 1 ]; then
    # Each single-node job gets its own rendezvous and an available port.
    LAUNCH=("${PYTHON}" -m torch.distributed.run
            --standalone --nnodes=1 --nproc_per_node="${REPLICAS}"
            -m agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_train)
fi

exec "${LAUNCH[@]}" \
  --dataset "${DATASET}" \
  --out "${OUT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --policy-model "${POLICY_MODEL}" \
  --encoder-model "${ENCODER_MODEL}" \
  --projector "${PROJECTOR}" \
  --representation "${REPRESENTATION}" \
  --scale "${SCALE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${PER_REPLICA_BATCH}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --drop-last \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --lr "${LR}" \
  --lr-schedule "${LR_SCHEDULE}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --grad-clip "${GRAD_CLIP}" \
  --eval-every "${EVAL_EVERY}" \
  --max-length "${MAX_LENGTH}" \
  --encoder-batch-size "${ENCODER_BATCH_SIZE}" \
  --encoder-max-length "${ENCODER_MAX_LENGTH}" \
  --policy-device "${POLICY_DEVICE}" \
  --encoder-device "${ENCODER_DEVICE}" \
  --dtype "${DTYPE}" \
  --limit "${LIMIT}" \
  --seed "${SEED}" \
  --wandb "${ALIGNMENT_WANDB:-auto}"
