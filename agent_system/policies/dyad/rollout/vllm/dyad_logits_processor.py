# Standalone Dyad logits processor. Its responsibilities are:
# 1. own an extra linear head, action_head
# 2. read the Dyad meta information and weights
# 3. compute "expanded action logits" from hidden_states at forward time
# 4. concatenate those extra logits right after the base model logits

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file


class DyadLogitsProcessor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dyad_action_size: int,
        use_bias: bool,
        dtype: torch.dtype,
        device: torch.device,
        action_config: dict,
    ) -> None:
        super().__init__()

        # Size of the Dyad expanded action space: if the base vocab is V and Dyad adds
        # K extra actions, dyad_action_size is normally K. The full action space
        # contains both vocabulary tokens and expanded actions, with V + K choices.
        self.dyad_action_size = dyad_action_size

        self.action_head = nn.Linear(
            in_features=hidden_size,
            out_features=dyad_action_size,
            bias=use_bias,
            device=device,
            dtype=dtype,
        )


    def load_dyad_weights(self, ckpt_path: str | Path) -> None:
        # load_file returns a state-dict-like object; the checkpoint contract is:
        # {
        #   "action_head.weight": tensor(...),
        #   "action_head.bias": tensor(...),   # only when bias is enabled
        # }
        sd = load_file(str(ckpt_path))

        missing = []

        if "action_head.weight" not in sd:
            missing.append("action_head.weight")

        if self.action_head.bias is not None and "action_head.bias" not in sd:
            missing.append("action_head.bias")

        # Fail fast so a format mismatch in the training-side export is caught early.
        if missing:
            raise ValueError(f"Missing Dyad weights: {missing}")

        with torch.no_grad():
            # Cast to the module's device/dtype: the checkpoint may sit on CPU while the
            # module is on GPU, or be fp32 while the model currently runs bf16/fp16.
            self.action_head.weight.copy_(sd["action_head.weight"].to(
                device=self.action_head.weight.device,
                dtype=self.action_head.weight.dtype,
            ))

            if self.action_head.bias is not None:
                self.action_head.bias.copy_(sd["action_head.bias"].to(
                    device=self.action_head.bias.device,
                    dtype=self.action_head.bias.dtype,
                ))

    def forward(
        self,
        base_logits: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # base_logits:
        #   output of the main model's original lm_head, shape is usually:
        #   [..., vocab_size]
        #
        # hidden_states:
        #   hidden states fed to the Dyad head, shape is usually:
        #   [..., hidden_size]
        #
        # Return value:
        #   the concatenated logits, shape is usually:
        #   [..., vocab_size + dyad_action_size]

        # No base logits were supplied by the caller. By design Dyad does not replace the
        # base logits, it appends extra action logits after them, so without base_logits
        # there is nothing to compute here.
        if base_logits is None:
            return None
        extra_logits = self.action_head(hidden_states)

        # Unify the dtype for the torch.cat below, and so the downstream sampling logic
        # always sees logits at a single precision.
        if extra_logits.dtype != base_logits.dtype:
            extra_logits = extra_logits.to(base_logits.dtype)

        # Concatenating on the last dim means a downstream sampler/decoder that accepts
        # this tensor can treat the expanded actions as extra tokens/actions taking part
        # in the same decision.
        logits = torch.cat([base_logits, extra_logits], dim=-1)

        return logits