"""The Alignment scorer: a frozen policy LLM backbone, a frozen encoder LLM backbone, and the trainable head between them.

Section 2.1 of the strategy document fixes the parameter surface:

    policy LLM backbone       frozen
    encoder LLM backbone      frozen
    projector       trainable

and section 2.2 fixes the arithmetic: `s_t(a) = x_t' w_a`, softmax over `C_t` only.

`w_a` is `scale(proj(projector(encoder_LM(prompt(a)))))`: the encoder reads the action's prompt,
the projector pools its final-layer states into one vector, and that vector **is** the head row.

Every row is normalized to unit length by default.
With scale="uniform", the policy vocabulary matrix supplies the target mean row norm instead.
Attention and MLP pooling start from the masked mean of encoder representations.
The direct head has a differentiable projector path from the first training step.

Nothing here reimplements the head. `LlmActionEncoder` and `DirectActionHead` are the objects the
Dyad runtime builds, wired the same way, so what Alignment trains is a module the runtime can load --
not a differently-shaped stand-in that happens to have the same output width. A Alignment projector
trained against a reimplementation would transfer numerically and be wrong in exactly the places the
two implementations disagreed, and there is no metric that reports it.

Two details that are easy to get wrong and produce no error when you do.

**`x_t` is the final input token's state, after the complete decision prefix.** For MCP this
includes the marker and the fixed JSON name-value opening, not the action name being predicted.
The state is the one that would otherwise feed `lm_head`. Taking a mean over the prompt, or the
state at the old marker position, still produces the right width and a loss that goes down.

**Right padding, and the last *real* token is found by the mask.** Preserve the complete prompt:
truncating either end changes the decision, and an automatically appended EOS changes its position.
Alignment rejects these inputs rather than silently scoring a different decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
from torch import nn

from agent_system.policies.dyad.data.actenc_alignment_dataset import validate_sample
from agent_system.policies.dyad.models.action_encoder import LlmActionEncoder
from agent_system.policies.dyad.models.action_head import DirectActionHead
from agent_system.utils.hf_config import text_hidden_size

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class AlignmentConfig:
    policy_model: str
    encoder_model: str = ""            # "" = the same path as the policy LLM backbone, separate weights
    projector: str = "attention"
    representation: str = "final_layer_hidden_states"
    scale: str = "unit"
    max_length: int = 2048
    encoder_max_length: int = 1024
    encoder_batch_size: int = 8       # action prompts per encoder forward, not training samples
    policy_device: str = "cuda:0"
    encoder_device: str = "cuda:1"
    dtype: str = "bfloat16"
    projector_kwargs: Optional[dict] = field(default=None)

    def resolved_encoder_model(self) -> str:
        return self.encoder_model or self.policy_model


class AlignmentActionSelector(nn.Module):
    """Scores every action in `C_t` against the policy's state at the decision position."""

    def __init__(self, config: AlignmentConfig):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if config.encoder_batch_size < 1:
            raise ValueError("encoder_batch_size must be positive")
        if config.max_length < 1 or config.encoder_max_length < 1:
            raise ValueError("max_length and encoder_max_length must be positive")
        self.config = config
        dtype = DTYPES.get(config.dtype, torch.bfloat16)

        self.policy_tokenizer = AutoTokenizer.from_pretrained(config.policy_model)
        if self.policy_tokenizer.pad_token_id is None:
            self.policy_tokenizer.pad_token = self.policy_tokenizer.eos_token
        # Right padding: the mask is what locates the decision position, and left padding would
        # instead put pad tokens *before* every prompt, changing the state the policy computes.
        self.policy_tokenizer.padding_side = "right"
        self.policy = AutoModelForCausalLM.from_pretrained(
            config.policy_model, dtype=dtype
        ).to(config.policy_device).eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

        self.encoder = LlmActionEncoder(
            config.resolved_encoder_model(),
            device=config.encoder_device,
            dtype=dtype,
            max_length=config.encoder_max_length,
            representation=config.representation,
        )

        self.policy_hidden = text_hidden_size(self.policy.config)
        self.head = DirectActionHead(
            config.projector,
            self.encoder.hidden_size,
            self.policy_hidden,
            scale=config.scale,
            projector_kwargs=config.projector_kwargs,
        ).to(config.policy_device)
        # The head is where the optimizer lives, so it is the one module that must be float32:
        # a bf16 attention softmax over a few hundred positions loses enough precision that two
        # nearly-tied positions swap.
        self.head.float()

    # ---- what trains -------------------------------------------------------------------

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def expected_trainable_names(self) -> list[str]:
        """The names section 2.1 says should train: whatever the head exposes.

        Computed from the module rather than written down, so a projector that gains a parameter is
        covered without anyone remembering to update a list -- and so the check that asserts the two
        agree is asserting something.
        """
        return [f"head.{name}" for name, _ in self.head.named_parameters()]

    # ---- the policy side ---------------------------------------------------------------

    def policy_state(self, prompts: Sequence[str]) -> torch.Tensor:
        """`[B, H]`: the final-layer state at each complete prompt's last real token.

        Tokenize once without truncation, validate before allocating a padded tensor, and feed
        those exact IDs to the LM. Never repair prompts here: their format belongs to the dataset.
        """
        prompts = list(prompts)
        if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("Alignment policy_state needs non-empty prompts")
        if self.policy_tokenizer.padding_side != "right":
            raise ValueError("Alignment policy tokenizer requires right padding")
        encoded = self.policy_tokenizer(
            prompts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        for i, (ids, special) in enumerate(zip(
            encoded["input_ids"], encoded["special_tokens_mask"], strict=True,
        )):
            if not ids:
                raise ValueError(f"Alignment policy prompt {i} must produce non-empty token IDs")
            if len(ids) > self.config.max_length:
                raise ValueError(
                    f"Alignment policy prompt {i} needs {len(ids)} tokens, exceeding "
                    f"max_length={self.config.max_length}; refusing to truncate the complete "
                    "policy input. Increase max_length within the model's supported context."
                )
            # The mask marks tokenizer-added tokens, not literal markers in the source text.
            # Keep a model's leading BOS, but never read a post-processor EOS as the decision.
            if special[-1]:
                raise ValueError(
                    f"Alignment policy prompt {i} has an automatically appended special token "
                    "after its decision boundary; use a tokenizer without a trailing special token."
                )
        encoded.pop("special_tokens_mask")
        batch = self.policy_tokenizer.pad(encoded, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"].to(self.config.policy_device)
        mask = batch["attention_mask"].to(self.config.policy_device)
        with torch.no_grad():
            out = self.policy.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError(
                f"{type(self.policy.model).__name__} returned no hidden states. The policy is read "
                "through its inner transformer precisely so that the vocabulary projection is "
                "skipped; a causal-LM head would return logits instead."
            )
        last = mask.sum(dim=1) - 1                                     # [B]
        index = last.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        return hidden.gather(1, index).squeeze(1).float()              # [B, H]

    def vocab_head(self) -> torch.Tensor:
        """Policy vocabulary matrix used as the direct head's scale reference."""
        head = self.policy.get_output_embeddings()
        if head is None or not hasattr(head, "weight"):
            raise RuntimeError(
                f"{type(self.policy).__name__} exposes no output embedding for the head scale reference."
            )
        return head.weight

    # ---- the action side ---------------------------------------------------------------

    def action_rows(self, prompts: Sequence[str]) -> torch.Tensor:
        """`[A, H]` head rows for A admissible actions, one prompt each.

        The vocabulary matrix supplies the output device/dtype and the target norm for uniform scale.
        It is not added to the projected action rows.
        """
        prompts = list(prompts)
        if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("Alignment action_rows needs non-empty prompts")
        # Alignment requires the complete catalogue, target definition and instruction. Guard all
        # chunks before the first encoder forward without changing the shared AgenticRL encoder.
        for start in range(0, len(prompts), self.config.encoder_batch_size):
            encoded = self.encoder.tokenizer(
                prompts[start:start + self.config.encoder_batch_size],
                padding=False, truncation=False, add_special_tokens=True,
            )
            for offset, ids in enumerate(encoded["input_ids"]):
                if not ids:
                    raise ValueError(f"Alignment encoder prompt {start + offset} must produce non-empty token IDs")
                if len(ids) > self.encoder.max_length:
                    raise ValueError(
                        f"Alignment encoder prompt {start + offset} needs {len(ids)} tokens, exceeding "
                        f"encoder_max_length={self.encoder.max_length}; refusing to truncate the "
                        "catalogue, target definition or instruction. Increase encoder_max_length "
                        "within the model's supported context."
                    )
        # Unit/none scaling only needs the reference width/device/dtype. Avoid a full
        # fp32 vocabulary copy (several GiB for large models) on every micro-batch.
        reference = self.vocab_head()
        if self.config.scale != "uniform":
            reference = reference[:1]
        reference = reference.float()
        rows = []
        for start in range(0, len(prompts), self.config.encoder_batch_size):
            cached = self.encoder.encode_hidden(
                list(prompts[start:start + self.config.encoder_batch_size]), use_cache=False)
            # The head moves bf16 states to its device and then casts to fp32.
            rows.append(self.head(cached.hidden, cached.mask, reference))
        return torch.cat(rows, dim=0)

    # ---- one batch ----------------------------------------------------------------------

    def score_batch(self, samples: Sequence[dict[str, Any]]) -> list[torch.Tensor]:
        """One `[|C_t|]` logit vector per sample, in `action_set` order.

        Encoder prompts are deduplicated across the batch before the forward. Two samples in one
        batch often share a visible catalogue, and the encoder is the expensive half; the mapping
        back is by the prompt string itself, so a collision can only happen between prompts that
        are byte-identical and therefore have the same answer.
        """
        # Direct callers can bypass load_split; reject legacy fields before either LM runs.
        for sample in samples:
            validate_sample(sample)
        states = self.policy_state([s["policy_lm_prompt"] for s in samples])   # [B, H]

        unique: dict[str, int] = {}
        prompts: list[str] = []
        for sample in samples:
            for action in sample["action_set"]:
                prompt = sample["action_encoder_prompts"][action]
                if prompt in unique:
                    continue
                unique[prompt] = len(prompts)
                prompts.append(prompt)
        rows = self.action_rows(prompts)                                       # [U, H]

        out = []
        for i, sample in enumerate(samples):
            index = torch.tensor(
                [unique[sample["action_encoder_prompts"][a]] for a in sample["action_set"]],
                device=rows.device,
                dtype=torch.long,
            )
            admissible_rows = rows.index_select(0, index)                       # [|C_t|, H]
            out.append(admissible_rows @ states[i].to(admissible_rows.device))
        return out
