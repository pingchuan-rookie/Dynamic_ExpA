"""Alignment's loss: cross-entropy over the admissible set, and nothing else (section 2.2).

The formula is ordinary. What it is worth being careful about is the *support*: the softmax runs
over `C_t`, exactly the sample's visible `action_set`, not the full action library.
A denominator that quietly includes one extra action still produces a loss that decreases, an
accuracy that looks plausible, and a policy trained against a decision it will never face.

`admissible_log_probs` is separated from `alignment_loss` because the evaluation metrics need the same
distribution the loss was computed from. Two call sites re-deriving "the probability of the
demonstrated action" is two chances for the metric to describe a different distribution than the
objective did.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch.nn import functional as F


def admissible_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """`log softmax` over one sample's admissible action set. `logits` is `[|C_t|]`."""
    if logits.ndim != 1:
        raise ValueError(f"expected one sample's logits [|C_t|], got {tuple(logits.shape)}")
    if logits.numel() < 2:
        raise ValueError(
            f"an admissible set of {logits.numel()} leaves nothing to choose between; the loss "
            "would be exactly 0 and the sample would teach nothing"
        )
    return F.log_softmax(logits.float(), dim=0)


def alignment_loss(
    logits: Sequence[torch.Tensor],
    label_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(mean loss, per-sample negative log probability)`.

    The mean is over samples, so a batch whose samples have four and ten admissible actions weights
    them equally. That is what section 2.2's `L = (1/N) sum L_t` says, and it is not the same as
    concatenating the action axis: the latter would let a ten-action sample dominate.
    """
    if len(logits) != len(label_indices):
        raise ValueError(f"{len(logits)} logit vectors against {len(label_indices)} labels")
    if not logits:
        raise ValueError("empty batch")
    per_sample = []
    for vector, label in zip(logits, label_indices):
        if not 0 <= label < vector.numel():
            raise ValueError(
                f"label index {label} is outside the {vector.numel()} admissible actions; the "
                "demonstrated action must be in C_t (strategy document section 1.8)"
            )
        per_sample.append(-admissible_log_probs(vector)[label])
    stacked = torch.stack(per_sample)
    return stacked.mean(), stacked
