# DYAD-MEMORY: shared by eager baseline/reference and expanded-action scoring.
"""Bound vocabulary statistics workspace in both inference and first-order training."""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Bound vocabulary-statistics workspace for the text and Dyad log-prob paths.
# Extension point: FSDPEngineWithLMHead.prepare_model_outputs / compute_split_policy_outputs

import torch
from torch.autograd.function import once_differentiable

import verl.utils.torch_functional as verl_F

VOCAB_STATISTICS_CHUNK_SIZE = 32


def _chunk_statistics(logits, labels, positions, temperature, entropy, sum_pi_squared, kl_labels):
    # Preserve the original vocabulary dtype and CE dispatch, including BF16 rounding.
    scaled = logits / temperature
    selected = scaled[positions]
    outputs = [torch.zeros_like(labels, dtype=torch.float32) for _ in range(3)]
    if selected.shape[0]:
        outputs[0][positions] = verl_F.logprobs_from_logits(selected, labels[positions], inplace_backward=False).float()
        if entropy:
            outputs[1][positions] = verl_F.entropy_from_logits(selected).float()
        if sum_pi_squared:
            outputs[2][positions] = verl_F.calculate_sum_pi_squared_from_logits(selected).float()
    if kl_labels is not None:
        outputs.append(verl_F.logprobs_from_logits(scaled, kl_labels, inplace_backward=False))
    return tuple(outputs)


class _VocabularyStatistics(torch.autograd.Function):
    """Offload saved logits, then replay ordinary autograd one token chunk at a time.

    Checkpointing individual indexed chunks still creates full-logits scatter gradients
    for every chunk. This boundary instead allocates one logits gradient and fills its
    disjoint slices, combining policy, entropy and reference-KL gradients locally.
    CUDA inputs are snapshotted on the CPU so the LM-head output storage can be freed
    before its gradient is allocated. The original logits are never mutated or reused
    as a gradient buffer, and the LM-head's normal autograd edge remains intact.
    """

    @staticmethod
    def forward(ctx, logits, labels, positions, temperature, entropy, sum_pi_squared, kl_labels, chunk_size):
        # DYAD-MEMORY: keep one GPU-sized matrix alive at a time across the LM-head
        # boundary: forward logits, then backward logits gradient. A blocking CPU copy
        # completes before the caller can release output.logits; no stream race or
        # mutation of an aliased saved input is involved. Inference needs no snapshot.
        saved_logits = None
        if ctx.needs_input_grad[0]:
            saved_logits = logits.detach().to(device="cpu", copy=True) if logits.is_cuda else logits
        ctx.logits_device = logits.device
        ctx.save_for_backward(
            saved_logits, labels, positions, kl_labels, temperature if isinstance(temperature, torch.Tensor) else None
        )
        ctx.scalar_temperature = None if isinstance(temperature, torch.Tensor) else temperature
        ctx.entropy = entropy
        ctx.sum_pi_squared = sum_pi_squared
        ctx.chunk_size = chunk_size
        ctx.set_materialize_grads(False)
        outputs = None
        for start in range(0, labels.numel(), chunk_size):
            stop = start + chunk_size
            divisor = temperature[start:stop] if isinstance(temperature, torch.Tensor) else temperature
            chunk = _chunk_statistics(
                logits[start:stop],
                labels[start:stop],
                positions[start:stop],
                divisor,
                entropy,
                sum_pi_squared,
                None if kl_labels is None else kl_labels[start:stop],
            )
            if outputs is None:
                outputs = [value.new_empty(labels.shape) for value in chunk]
            for output, value in zip(outputs, chunk, strict=True):
                output[start:stop] = value
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *output_grads):
        logits, labels, positions, kl_labels, temperature = ctx.saved_tensors
        # Saved-tensor offload hooks may return CPU metadata as well as CPU logits.
        # Restore only these O(tokens) vectors eagerly; logits stay on the host and
        # are reloaded one chunk at a time. Never infer the execution device from
        # saved storage, whose device is deliberately independent of the LM head.
        labels = labels.to(device=ctx.logits_device)
        positions = positions.to(device=ctx.logits_device)
        if kl_labels is not None:
            kl_labels = kl_labels.to(device=ctx.logits_device)
        if temperature is None:
            temperature = ctx.scalar_temperature
        else:
            temperature = temperature.to(device=ctx.logits_device)
        logits_grad = torch.empty_like(logits, device=ctx.logits_device)
        for start in range(0, labels.numel(), ctx.chunk_size):
            stop = start + ctx.chunk_size
            divisor = temperature[start:stop] if isinstance(temperature, torch.Tensor) else temperature
            with torch.enable_grad():
                chunk_logits = logits[start:stop].to(device=ctx.logits_device).detach().requires_grad_(True)
                outputs = _chunk_statistics(
                    chunk_logits,
                    labels[start:stop],
                    positions[start:stop],
                    divisor,
                    ctx.entropy,
                    ctx.sum_pi_squared,
                    None if kl_labels is None else kl_labels[start:stop],
                )
                active = [
                    (value, grad[start:stop])
                    for value, grad in zip(outputs, output_grads, strict=True)
                    if grad is not None and value.requires_grad
                ]
                if active:
                    values, grads = zip(*active, strict=True)
                    (gradient,) = torch.autograd.grad(values, chunk_logits, grads)
                    logits_grad[start:stop].copy_(gradient)
                    del values, grads, gradient
                else:
                    logits_grad[start:stop].zero_()
            # Do not retain the previous chunk's graph while constructing the next one.
            del outputs, active, chunk_logits
        return logits_grad, None, None, None, None, None, None, None


def compute_vocab_statistics(
    logits,
    labels,
    positions,
    temperature,
    calculate_entropy=False,
    calculate_sum_pi_squared=False,
    kl_labels=None,
    chunk_size=VOCAB_STATISTICS_CHUNK_SIZE,
):
    """Score selected vocabulary positions and optional vocabulary-only KL labels.

    Temperature is a constant scalar or one divisor per token, not a learned input.
    All vocabulary candidates remain in every normalization; only token rows are chunked.
    Higher-order derivatives are intentionally unsupported, as this is a PPO boundary.
    """
    if chunk_size < 1:
        raise ValueError("Vocabulary statistics chunk_size must be positive.")
    if isinstance(temperature, torch.Tensor) and temperature.requires_grad:
        raise ValueError("Vocabulary statistics temperature must be constant.")
    # Scalar division uses PyTorch's wrapped-scalar semantics; vector temperatures
    # were explicitly cast to the vocabulary dtype by the engine before division.
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.to(device=logits.device, dtype=logits.dtype).reshape(-1, 1)
        if temperature.shape[0] == 1:
            temperature = temperature.expand(labels.numel(), 1)
        elif temperature.shape[0] != labels.numel():
            raise ValueError("Vocabulary temperature must have one value per token.")
    if labels.numel() == 0:
        empty = logits.sum(-1).float()
        return (empty, empty, empty) + ((empty.to(logits.dtype),) if kl_labels is not None else ())
    return _VocabularyStatistics.apply(
        logits if torch.is_grad_enabled() else logits.detach(),
        labels,
        positions,
        temperature,
        calculate_entropy,
        calculate_sum_pi_squared,
        kl_labels,
        chunk_size,
    )
