"""Dyad's forward on verl 0.9's engine abstraction.

verl 0.9 splits what 0.7 kept inside `DataParallelPPOActor` into two injectable pieces:

    engine.prepare_model_inputs / prepare_model_outputs   how model_output is produced
    loss_fn(config, model_output, data, dp_group)         what is optimized

Dyad needs both. The action head means `log_probs` is not "log-softmax over the vocabulary" at
every position, so it cannot be expressed as a loss on top of the stock model_output -- the split
has to happen where the logits are, which is `prepare_model_outputs`. The loss part lives in
`ppo_loss.dyad_ppo_loss`.

Layout note. 0.9 runs on `DatasetPadMode.NO_PADDING`: `input_ids` is a jagged nested tensor over
prompt+response and the packed values are what the model sees. Dyad's per-decision fields
(`response_dyad` / `tool_mask` / `seq_mask` / `dyad_action_mask`) arrive as `[B, R]` padded tensors
covering the response only, which is also the form `dyad_ppo_loss` wants (`response_mask` and
`advantages` are padded there too). So the packing to full-sequence packed form happens here, in
`prepare_model_inputs`, and nothing outside this file has to know about it.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Add action logits and masks while inheriting the official FSDP engine lifecycle.
# Extension point: EngineRegistry(model_type="dyad_language_model") -> FSDPEngineWithLMHead subclass

from agent_system.policies.dyad.actions.task_context import dynamic_actions_enabled
from contextlib import nullcontext
from agent_system.policies.dyad.training.encoder_training import EncoderTrainingMixin, replay_context

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_id
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead

from agent_system.utils.diagnostics import log_event
from agent_system.policies.dyad.algorithms.split_policy import compute_split_policy_outputs
from agent_system.policies.dyad.training.model_setup import freeze_dyad_action_head, install_dyad_action_head, install_dyad_encoder
from verl_extensions.agent_steps.batch_transport import exact_batch_padding_enabled

# Fields that must all be present for a batch to take the Dyad path. `seq_mask` is deliberately not
# in here: it has a well-defined default (all-True) and a reference actor never carries it, so
# requiring it would turn "no seq_mask" into "silently train the base-token policy", which is the
# exact failure the Dyad path refuses to make elsewhere.
DYAD_REQUIRED_FIELDS = ("response_dyad", "tool_mask", "dyad_action_mask")


def has_dyad_fields(data: TensorDict) -> bool:
    """Whether this batch carries Dyad policy labels.

    False for the reference actor, which is a plain vocabulary model by design (there is no
    reference distribution over the action head, see ppo_loss for what that means for KL).
    """
    keys = set(data.keys())
    return all(field in keys for field in DYAD_REQUIRED_FIELDS)


def _response_bounds(micro_batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence (response_start, response_end) as offsets into the packed sequence.

    Derived from `input_ids`' jagged offsets and the response lengths, so the bounds are the same
    ones `no_padding_2_padding` uses to slice model output back to `[B, R]`. Deriving them any
    other way would let the two disagree, and a one-token disagreement here is exactly the kind of
    drift that shows up as "Dyad tool position has no allowed actions" several layers down.
    """
    cu_seqlens = micro_batch["input_ids"].offsets()
    responses = micro_batch["responses"]
    if responses.is_nested:
        response_lens = responses.offsets().diff()
    else:
        # Padded responses: the response length is what the attention mask says is real.
        attention_mask = micro_batch["attention_mask"]
        prompt_len = micro_batch["prompts"].shape[1]
        response_lens = attention_mask[:, prompt_len:].sum(dim=1)
    response_lens = response_lens.to(device=cu_seqlens.device)
    ends = cu_seqlens[1:]
    return ends - response_lens, ends


def _scatter_response_field(
    field: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    packed: torch.Tensor,
) -> torch.Tensor:
    """Write a `[B, R, ...]` padded response field into its slots in a packed `(total_nnz, ...)`.

    `packed` supplies the prompt-position values, which differ per field: `response_dyad` keeps the
    real token ids there (a vocabulary label outside `[0, V)` trips a guard in
    `compute_split_policy_outputs` even at positions that `seq_mask` later zeroes), while the masks
    are all-False.
    """
    if field.shape[0] != len(starts):
        raise ValueError("Dyad response field has the wrong batch size")
    for i in range(field.shape[0]):
        length = int(ends[i] - starts[i])
        row = field[i]
        if row.shape[0] < length or (field.is_nested and row.shape[0] != length):
            raise ValueError("Dyad response field length disagrees with sampled response")
        if row.shape[1:] != packed.shape[1:]:
            raise ValueError("Dyad response field width disagrees with replay buffer")
        if length:
            packed[starts[i] : ends[i]] = row[:length]
    return packed


@EngineRegistry.register(model_type="dyad_language_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class DyadFSDPEngineWithLMHead(EncoderTrainingMixin, FSDPEngineWithLMHead):
    """FSDP engine whose model_output carries Dyad's split-head log-probs.

    Batching, offload, checkpointing and Ulysses padding are inherited. Forward
    processing delegates to the parent when the batch has no Dyad fields.
    FSDP1 NO_SHARD additionally finalizes gradients after each micro-batch
    to support mixed-precision accumulation with original tied parameters.
    """

    def forward_backward_batch(self, data: TensorDict, loss_function, forward_only=False):
        if exact_batch_padding_enabled() and not forward_only:
            # This boundary receives one rank-local optimizer minibatch, before
            # splitting. Reduce diagnostic counts separately from the parent's
            # response-span objective denominator so padding changes neither.
            fields = [key for key in ("response_mask", "seq_mask", "tool_mask") if key in data.keys()]
            masks = data.select(*fields).to_padded_tensor()
            decisions = masks["response_mask"].bool()
            if "seq_mask" in masks.keys():
                decisions = decisions & masks["seq_mask"].bool()
            action_count = decisions.new_zeros((), dtype=torch.long)
            if "tool_mask" in masks.keys():
                action_count = (decisions & masks["tool_mask"].bool()).sum()
            counts = torch.stack((decisions.sum(), action_count))
            counts = counts.to(get_device_id())
            torch.distributed.all_reduce(
                counts, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
            )
            tu.assign_non_tensor(
                data,
                dyad_metric_num_tokens=int(counts[0].item()),
                dyad_metric_num_action_tokens=int(counts[1].item()),
            )
        return super().forward_backward_batch(data, loss_function, forward_only=forward_only)

    def _gradient_sync_context(self, *, is_last_micro_batch: bool):
        """Finalize NO_SHARD gradients before the next forward.

        FSDP1 with original tied parameters and mixed precision can fail when
        no_sync leaves low-precision gradients attached to full-precision
        parameters (PyTorch 2.11), including replicated multi-rank NO_SHARD.
        Normal backward restores optimizer-precision gradients; accumulation,
        loss scaling and the single optimizer step remain owned by the parent.
        NO_SHARD synchronizes each micro-batch when replicated across ranks.
        Sharded FSDP1 and FSDP2 retain upstream deferred synchronization.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

        if (
            isinstance(self.module, FullyShardedDataParallel)
            and self.module.sharding_strategy == ShardingStrategy.NO_SHARD
        ):
            return nullcontext()
        return super()._gradient_sync_context(is_last_micro_batch=is_last_micro_batch)

    def _build_module(self):
        """Build the base model, then install Dyad's heads onto it before the FSDP wrap.

        Upstream's `_build_module` dispatches on `model_config.model_type`, which is set to
        `dyad_language_model` here so `EngineRegistry` resolves to this class. It is swapped back
        for the duration of the super() call rather than teaching upstream a third value, so no
        verl file needs an Dyad branch.

        The reference actor (`forward_only`) is left as a plain vocabulary model: there is no
        reference distribution over the action head, and `dyad_losses.dyad_ppo_loss` is written
        around that fact.
        """
        original_model_type = self.model_config.model_type
        self.model_config.model_type = "language_model"
        try:
            module = super()._build_module()
        finally:
            self.model_config.model_type = original_model_type

        if self.engine_config.forward_only:
            self.dyad_state = None
            return module

        from verl.utils.torch_dtypes import PrecisionType

        torch_dtype = self.engine_config.model_dtype
        if torch_dtype is None:
            torch_dtype = torch.float32
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        state = install_dyad_action_head(module, self.model_config.tokenizer, torch_dtype)
        install_dyad_encoder(
            module,
            state,
            tokenizer=self.model_config.tokenizer,
            local_path=self.model_config.local_path,
            total_training_steps=getattr(self.optimizer_config, "total_training_steps", 0),
        )
        self.dyad_state = state
        self.dyad_vocab_size = int(module.get_output_embeddings().weight.shape[0])

        # Freezing the materialized action rows leaves the module with mixed `requires_grad` over
        # its original parameters. FSDP1 only handles that with use_orig_params=True; without it
        # the flat parameter takes a single requires_grad for the whole shard and the frozen head
        # is trained after all. Forced here rather than required of the training scripts, because
        # it follows from the frozen materialization rather than a per-run configuration choice.
        if freeze_dyad_action_head(module) or state.use_orig_params:
            self._force_orig_params()

        return module

    def _force_orig_params(self) -> None:
        if getattr(self.engine_config, "use_orig_params", False):
            return
        try:
            self.engine_config.use_orig_params = True
        except Exception:  # frozen dataclass
            object.__setattr__(self.engine_config, "use_orig_params", True)
        print("[Dyad] use_orig_params forced to True (the materialized action head is frozen)", flush=True)

    def _dyad_projector(self, module):
        """The direct action head (projector + width projection), through the FSDP wrapper if present.

        Takes the module rather than reading `self.module`: upstream assigns `self.module` *after*
        `_build_optimizer` returns (transformer_impl.py, `_build_fsdp_module` -> `_build_optimizer`
        -> `self.module = module`), so reading it here finds None and the projector group is silently
        never created -- the exact failure this method exists to prevent.
        """
        for candidate in (module, getattr(module, "_fsdp_wrapped_module", None)):
            if candidate is None:
                continue
            projector = getattr(candidate, "dyad_residual_head", None)
            if projector is not None:
                return projector
        return None

    def _build_policy_optimizer(self, module):
        """Upstream's optimizer, plus a second parameter group for the projector at its own rate.

        The projector has its own parameter group and may use a different learning rate.
        The dyad_group tag lets the training schedule identify that group without changing
        FSDP-managed requires_grad flags after wrapping.

        Falls through to upstream when there is no projector or no rate was configured, so a run that
        does not set DYAD_PROJECTOR_LR keeps exactly the previous single-group optimizer.
        """
        projector = self._dyad_projector(module)
        config = getattr(getattr(self, "dyad_state", None), "encoder_config", None)
        projector_lr = getattr(config, "projector_lr", None) if config is not None else None
        if projector is None or projector_lr is None:
            return FSDPEngineWithLMHead._build_optimizer(self, module)

        from verl.workers.config.optimizer import build_optimizer

        projector_params = [p for p in projector.parameters() if p.requires_grad]
        if not projector_params:
            return FSDPEngineWithLMHead._build_optimizer(self, module)
        projector_ids = {id(p) for p in projector_params}
        rest = [p for p in module.parameters() if id(p) not in projector_ids]

        optimizer = build_optimizer(rest, self.optimizer_config)
        # add_param_group inherits betas / weight_decay from the group above, so only lr differs.
        # It has to happen before _build_lr_scheduler: LambdaLR reads `initial_lr` off every group at
        # construction, and a group added afterwards would never be scheduled (no warmup, no decay).
        optimizer.add_param_group(
            {"params": projector_params, "lr": float(projector_lr), "dyad_group": "projector"}
        )
        # Per rank, and 0 is a normal number to see here. FSDP shards the flat parameter by byte
        # range, and the projector is one contiguous region of it, so on two ranks it usually lands
        # entirely inside one rank's half: measured 12,584,961 on one and 0 on the other for
        # Qwen2.5-3B. The rank that owns it is also the rank FSDP reduce-scatters its gradient to, so
        # the rate is applied exactly once. Printing the local number rather than an all-reduced one
        # keeps this visible instead of hiding it behind a total that always looks right.
        print(
            f"[Dyad] projector optimizer group: {sum(p.numel() for p in projector_params):,} "
            f"parameters on this rank at lr={projector_lr:g} "
            f"(policy LM at lr={self.optimizer_config.lr:g}); 0 here means this rank's shard holds "
            f"none of the projector",
            flush=True,
        )
        return optimizer

    def _takes_dyad_path(self, micro_batch: TensorDict) -> bool:
        """Whether this engine should compute split-head log-probs for this batch.

        Two conditions, and the `forward_only` one is not redundant. The trainer hands the
        reference actor the same batch it hands the policy LLM backbone, Dyad fields included, so keying on the
        fields alone would put the reference model on the action-head path -- and it has no action
        head, only the base module. The KL would then be taken against a distribution the reference
        model never produced. verl 0.7 made the same distinction through
        `DyadActor.is_ref_actor = (actor_optimizer is None)`; `forward_only` is 0.9's spelling of
        exactly that.
        """
        if self.engine_config.forward_only:
            return False
        present = set(DYAD_REQUIRED_FIELDS).intersection(micro_batch.keys())
        if present and len(present) != len(DYAD_REQUIRED_FIELDS):
            raise ValueError("Incomplete Dyad policy trace in actor replay")
        return has_dyad_fields(micro_batch)

    def optimizer_step(self):
        """Apply the training_schedule's phase scale, then hand off to the base implementation.

        `dyad_phase_scale` is set on each param group by the worker when a phase begins. It is a
        multiplier rather than a replacement because the LR scheduler rewrites `group["lr"]` on
        every `scheduler.step()` -- overwriting the value directly would hold for exactly one step
        and then silently stop, leaving a run that says it is in phase 1 and behaves like phase 2.
        """
        scaled = []
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            for group in optimizer.param_groups:
                scale = group.get("dyad_phase_scale")
                if scale is not None and scale != 1.0:
                    scaled.append((group, group["lr"]))
                    group["lr"] = group["lr"] * scale
        try:
            grad_norm = super().optimizer_step()
        finally:
            for group, original in scaled:
                group["lr"] = original

        log_event(
            "dyad_actor",
            "optimizer_step",
            grad_norm=grad_norm.detach() if torch.is_tensor(grad_norm) else grad_norm,
            **self._dyad_adapter_metrics(),
        )
        return grad_norm

    def _dyad_adapter_metrics(self) -> dict:
        """Projector-side counterparts of `grad_norm`. Empty when there is no projector.

        `grad_norm` alone cannot tell "the policy LLM backbone is frozen and the projector is learning" from
        "nothing is being trained at all": under `frozen_llm_adaptation` the policy LLM backbone's grad_norm is 0 in
        both cases. The projector gradient norm and parameter L2 norm are reduced across ranks
        so every rank reports the same values even when its local projector shard is empty.
        """
        module = getattr(self, "module", None)
        projector = None
        for candidate in (module, getattr(module, "_fsdp_wrapped_module", None)):
            if candidate is None:
                continue
            projector = getattr(candidate, "dyad_residual_head", None)
            if projector is not None:
                break
        if projector is None:
            return {}

        with torch.no_grad():
            grad_sq = sum(float(p.grad.float().pow(2).sum()) for p in projector.parameters() if p.grad is not None)
            argument_sq = sum(float(p.float().pow(2).sum()) for p in projector.parameters())

            stats = torch.tensor([grad_sq, argument_sq], dtype=torch.float64, device=get_device_id())
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            grad_sq, argument_sq = (float(x) for x in stats.tolist())

        metrics = {
            "dyad/adapter_grad_norm": grad_sq**0.5,
            "dyad/adapter_l2": argument_sq**0.5,
        }
        return metrics

    def prepare_model_inputs(self, micro_batch: TensorDict):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch=micro_batch)

        if not self._takes_dyad_path(micro_batch):
            return model_inputs, output_args

        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)

        if not use_remove_padding:
            raise NotImplementedError(
                "The Dyad action-head path requires use_remove_padding=True on verl 0.9: the packed "
                "layout is what aligns the Dyad fields with input_ids_rmpad_rolled."
            )
        if self.use_ulysses_sp:
            raise NotImplementedError("The Dyad action-head path does not support Ulysses SP yet.")
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        # The action logits are produced by the forward hook installed by
        # `agent_system.policies.dyad.models.policy_forward.attach_dyad_forward_hook`, which reads `output.hidden_states[-1]` and
        # writes the action logits back into that same field. Without this flag the hook sees
        # hidden_states=None and returns the output untouched, and the failure downstream is a
        # missing key rather than anything naming Dyad.
        model_inputs["output_hidden_states"] = True

        import os
        if dynamic_actions_enabled():
            self._prepare_codegym_heads(micro_batch)
        output_args["dyad"] = self._pack_dyad_fields(micro_batch, output_args)
        return model_inputs, output_args

    def _prepare_codegym_heads(self, micro_batch):
        from agent_system.policies.dyad.actions.task_context import load_context, context_head, packed_action_logits

        context_key = "dyad_action_context" if "dyad_action_context" in micro_batch else "codegym_context"
        if context_key not in micro_batch:
            raise ValueError("Dynamic action replay is missing the sampling task context")
        contexts = list(micro_batch[context_key])
        if "dyad_action_context" in micro_batch and "codegym_context" in micro_batch:
            if contexts != list(micro_batch["codegym_context"]):
                raise ValueError("Dyad sampling/replay context identities disagree")
        payloads = [load_context(path) for path in contexts]
        if len(payloads) != micro_batch.batch_size[0]:
            raise ValueError("Dyad sampling/replay context count disagrees")
        if "dyad_action_context_identity" in micro_batch:
            identities = list(micro_batch["dyad_action_context_identity"])
            if any(p.get("identity") != identity for p, identity in zip(payloads, identities, strict=True)):
                raise ValueError("Dyad sampling/replay encoder identities disagree")
        lengths = micro_batch["input_ids"].offsets().diff().tolist()
        inner = getattr(self.module, "_fsdp_wrapped_module", self.module)
        capacity = micro_batch["dyad_action_mask"].shape[-1]
        if any(p["action_config"]["total_size"] != capacity for p in payloads):
            raise ValueError("CodeGym sampling/replay action widths disagree")
        for row, payload in enumerate(payloads):
            action_ids = payload["action_config"]["action_ids"]
            candidates = micro_batch["dyad_action_mask"][row]
            if candidates[..., len(action_ids):].any():
                raise ValueError("Dyad replay candidates include undefined task actions")
            labels = micro_batch["response_dyad"][row]
            selected = labels[micro_batch["tool_mask"][row].bool()]
            if selected.numel() and not torch.isin(selected, selected.new_tensor(action_ids)).all():
                raise ValueError("Dyad replay action label does not belong to the sampling task")

        def logits_fn(hidden):
            # This executes within the model forward, while FSDP parameters are gathered.
            reference = inner.get_output_embeddings().weight.detach()
            encoder = self._trainable_encoder()
            current = [replay_context(p, encoder) for p in payloads] if encoder is not None else payloads
            heads = [context_head(p, inner.dyad_residual_head, reference,
                                  self.model_config.tokenizer) for p in current]
            return packed_action_logits(hidden, heads, lengths,
                                        getattr(inner, "dyad_gradient_lines", None))
        inner.dyad_task_logits_fn = logits_fn

    def _pack_dyad_fields(self, micro_batch: TensorDict, output_args: dict) -> dict:
        """Bring the `[B, R]` Dyad fields into the same packed, rolled, padded frame as the logits.

        Three transforms, in the order `prepare_model_inputs` applies them to `input_ids`:

          1. scatter the response-only field into a full-sequence packed buffer;
          2. `roll(-1)`, so index t holds the label for the token predicted at t;
          3. right-pad by `output_args["pad_size"]`, matching the static bucket pad.

        Doing (2) before (3) matters for the same reason it does upstream: the roll has to see the
        true end of the global packed sequence, not a padded one.
        """
        input_ids = micro_batch["input_ids"]
        packed_values = input_ids.values()
        total_nnz = packed_values.shape[0]
        device = packed_values.device

        starts, ends = _response_bounds(micro_batch)

        response_dyad = micro_batch["response_dyad"]
        tool_mask = micro_batch["tool_mask"].bool()
        dyad_action_mask = micro_batch["dyad_action_mask"].bool()
        if "seq_mask" in micro_batch.keys():
            seq_mask = micro_batch["seq_mask"].bool()
        else:
            seq_mask = torch.ones_like(response_dyad, dtype=torch.bool)

        action_size = dyad_action_mask.shape[-1]

        labels = _scatter_response_field(
            response_dyad.to(device=device, dtype=packed_values.dtype),
            starts,
            ends,
            packed_values.clone(),
        )
        tool_packed = _scatter_response_field(
            tool_mask.to(device=device),
            starts,
            ends,
            torch.zeros(total_nnz, dtype=torch.bool, device=device),
        )
        seq_packed = _scatter_response_field(
            seq_mask.to(device=device),
            starts,
            ends,
            torch.zeros(total_nnz, dtype=torch.bool, device=device),
        )
        action_packed = _scatter_response_field(
            dyad_action_mask.to(device=device),
            starts,
            ends,
            torch.zeros(total_nnz, action_size, dtype=torch.bool, device=device),
        )

        labels = torch.roll(labels, shifts=-1, dims=0)
        tool_packed = torch.roll(tool_packed, shifts=-1, dims=0)
        seq_packed = torch.roll(seq_packed, shifts=-1, dims=0)
        action_packed = torch.roll(action_packed, shifts=-1, dims=0)

        pad_size = output_args.get("pad_size", 0)
        if pad_size:
            labels = torch.nn.functional.pad(labels, (0, pad_size))
            tool_packed = torch.nn.functional.pad(tool_packed, (0, pad_size))
            seq_packed = torch.nn.functional.pad(seq_packed, (0, pad_size))
            action_packed = torch.nn.functional.pad(action_packed, (0, 0, 0, pad_size))

        return {
            "labels": labels,
            "tool_mask": tool_packed,
            "seq_mask": seq_packed,
            "dyad_action_mask": action_packed,
        }

    @staticmethod
    def _extract_action_logits(output) -> tuple[torch.Tensor, ...]:
        """The action logits the forward hook left behind: one tensor per gradient line.

        One entry is the ordinary case (a single line, or the head frozen). Two entries means the
        hook emitted the policy line and the encoder line separately, in that order; see
        `agent_system.policies.dyad.models.policy_forward.attach_dyad_forward_hook`.
        """
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None or len(hidden_states) not in (1, 2):
            raise RuntimeError(
                "The Dyad training path needs action logits in output.hidden_states, written there "
                "by attach_dyad_forward_hook(actor_module) before the model is wrapped with FSDP. "
                f"Got hidden_states={type(hidden_states).__name__} with "
                f"{0 if hidden_states is None else len(hidden_states)} entries; 1 or 2 expected."
            )
        return tuple(hidden_states)

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        dyad = output_args.get("dyad")
        if dyad is None:
            return super().prepare_model_outputs(
                output=output,
                output_args=output_args,
                micro_batch=micro_batch,
                logits_processor_func=logits_processor_func,
            )

        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        calculate_sum_pi_squared = tu.get_non_tensor_data(
            data=micro_batch, key="calculate_sum_pi_squared", default=False
        )

        input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
        temperature_rmpad = output_args["temperature_rmpad"]
        pad_size = output_args["pad_size"]

        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        precomputed_vocab = None
        if use_fused_kernels:
            if calculate_sum_pi_squared:
                raise NotImplementedError("Dyad fused LM kernels do not provide sum_pi_squared.")
            if torch.any(dyad["labels"][~dyad["tool_mask"]] != input_ids_rmpad_rolled[~dyad["tool_mask"]]):
                raise ValueError("Fused vocabulary labels disagree with Dyad replay labels.")
            if getattr(output, "log_probs", None) is None:
                raise RuntimeError("Selected model/backend did not return fused vocabulary log-probabilities.")
            precomputed_vocab = {
                "log_probs": output.log_probs.squeeze(0), "labels": input_ids_rmpad_rolled,
            }
            if calculate_entropy:
                if getattr(output, "entropy", None) is None:
                    raise RuntimeError("Selected fused model/backend did not return vocabulary entropy.")
                precomputed_vocab["entropy"] = output.entropy.squeeze(0)
            base_logits = None
        else:
            base_logits = output.logits.squeeze(0)  # (total_nnz + pad, V)
        action_lines = tuple(t.squeeze(0) for t in self._extract_action_logits(output))  # each (total_nnz + pad, A)

        # DYAD-MEMORY: scale within token chunks so neither temperature nor entropy
        # materializes another full-context vocabulary matrix. Action scores still
        # promote to FP32 before division, while vocabulary scores retain their dtype.
        temperature = temperature_rmpad.clamp(min=1e-8)

        seq_mask = dyad["seq_mask"]

        policy_outputs = compute_split_policy_outputs(
            base_logits=base_logits,
            action_logits=action_lines[0],
            labels=dyad["labels"],
            tool_mask=dyad["tool_mask"],
            dyad_action_mask=dyad["dyad_action_mask"],
            temperature=temperature,
            calculate_entropy=calculate_entropy,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            vocab_kl_labels=input_ids_rmpad_rolled,
            precomputed_vocab=precomputed_vocab,
            vocab_size=self.dyad_vocab_size if use_fused_kernels else None,
        )

        log_probs = policy_outputs["log_probs"].masked_fill(~seq_mask, 0.0)

        # The encoder's own line, when the hook emitted one. Same distributions and the same
        # guards; the only difference is which factor of `x_t . w_a` was detached upstream, so the
        # values are bit-identical and only the graph differs. Recomputing rather than reusing is
        # what keeps the two gradients from meeting: one tensor cannot carry two objectives whose
        # normalisations differ.
        #
        # `base_logits` is passed detached here. The vocabulary half belongs entirely to the policy
        # line -- it has no `w_a` in it at all -- and letting it back into the encoder line would
        # give the projector's loss a gradient path into the policy LLM backbone, which is the thing the two
        # lines exist to prevent.
        encoder_log_probs = None
        if len(action_lines) == 2:
            encoder_outputs = compute_split_policy_outputs(
                base_logits=None if base_logits is None else base_logits.detach(),
                action_logits=action_lines[1],
                labels=dyad["labels"],
                tool_mask=dyad["tool_mask"],
                dyad_action_mask=dyad["dyad_action_mask"],
                temperature=temperature,
                calculate_entropy=False,
                calculate_sum_pi_squared=False,
                precomputed_vocab=None if precomputed_vocab is None else {
                    key: value.detach() for key, value in precomputed_vocab.items()
                },
                vocab_size=self.dyad_vocab_size if use_fused_kernels else None,
            )
            encoder_log_probs = encoder_outputs["log_probs"].masked_fill(~seq_mask, 0.0)

        # Vocabulary-only log-prob, for the KL against the reference model. The reference actor has
        # no action head, so a KL between the Dyad policy log-prob and a vocab-only reference would
        # be comparing two different sample spaces. `dyad_ppo_loss` masks the action positions out
        # on top of this.
        vocab_log_probs = policy_outputs["vocab_log_probs"].masked_fill(~seq_mask, 0.0)

        model_output = {}
        cu_seqlens = micro_batch["input_ids"].offsets()

        def _to_nested(packed: torch.Tensor) -> torch.Tensor:
            return torch.nested.nested_tensor_from_jagged(
                self._gather_and_unpad_packed(packed, pad_size), cu_seqlens
            )

        model_output["log_probs"] = _to_nested(log_probs)
        model_output["vocab_log_probs"] = _to_nested(vocab_log_probs)
        if encoder_log_probs is not None:
            model_output["encoder_log_probs"] = _to_nested(encoder_log_probs)

        if calculate_entropy:
            model_output["entropy"] = _to_nested(policy_outputs["entropys"].masked_fill(~seq_mask, 0.0))
        if calculate_sum_pi_squared:
            model_output["sum_pi_squared"] = _to_nested(
                policy_outputs["sum_pi_squared"].masked_fill(~seq_mask, 0.0)
            )

        return model_output
