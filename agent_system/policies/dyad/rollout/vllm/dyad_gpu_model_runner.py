from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch
import torch.distributed

from agent_system.utils.diagnostics import log_event, module_parameter_summary
from agent_system.utils.hf_config import text_hidden_size, text_vocab_size
from agent_system.policies.dyad import models as action_encoder

# DYAD-NOTE(vllm0.24): the whole vllm.attention package moved to vllm.v1.attention, and the old
# backends/abstract.py was folded into backend.py.
from vllm.v1.attention.backend import (
    AttentionMetadata,
)
from vllm.config import (
    CUDAGraphMode,
)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.utils import is_residual_scattered_for_sp
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

# >>> DYAD-BEGIN(vllm0.24): dependencies introduced by syncing 0.24's verbatim methods
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from dataclasses import replace
# <<< DYAD-END

# >>> DYAD-BEGIN(vllm0.24): dependencies introduced by syncing 0.24's verbatim methods
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU
from vllm.v1.outputs import RoutedExpertsLists
from vllm.v1.outputs import RoutedExpertsTensors
# <<< DYAD-END

logger = init_logger(__name__)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict

from vllm.v1.worker.gpu_model_runner import GPUModelRunner,ExecuteModelState,AsyncGPUModelRunnerOutput
from transformers import AutoTokenizer




def _lm_head_of(model) -> torch.nn.Module:
    """The output projection, whether the model is a plain causal LM or a vision+text wrapper.

    vLLM builds Qwen3.5 as `Qwen3_5ForConditionalGeneration`, which owns a vision tower and keeps
    the whole text stack -- lm_head included -- under `language_model`. Dyad reads this head for
    three things: the hidden size it sizes its action head to, and the device/dtype it aligns
    action embeddings to. Reaching for `model.lm_head` alone stops the engine before it serves a
    single request, which is the good case; the bad case would be silently finding some other
    head.
    """
    head = getattr(model, "lm_head", None)
    if head is None:
        head = getattr(getattr(model, "language_model", None), "lm_head", None)
    if head is None:
        raise ValueError(
            f"{type(model).__name__} exposes no lm_head, directly or under language_model; "
            "the generic Dyad path cannot infer hidden size."
        )
    if not hasattr(head, "weight"):
        raise ValueError("the resolved lm_head has no weight; cannot infer hidden size for Dyad head.")
    return head


def _apply_only_allowed_ids(logits: torch.Tensor, allowed_ids: list[int]) -> None:
    """
    Keep only the logits at the positions in allowed_ids, set every other position to -inf.
    Assumes logits.shape == [1, vocab_size].
    """
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"expected logits shape [1, vocab], got {tuple(logits.shape)}")

    vocab_size = logits.shape[1]

    if not allowed_ids:
        raise ValueError("allowed_ids is empty, no valid token can be selected.")

    for tid in allowed_ids:
        if tid < 0 or tid >= vocab_size:
            raise ValueError(f"token id {tid} out of range [0, {vocab_size})")

    allowed_ids_t = torch.tensor(allowed_ids, device=logits.device, dtype=torch.long)

    # Mask everything first, then restore the allowed positions.
    mask = torch.ones(vocab_size, device=logits.device, dtype=torch.bool)
    mask[allowed_ids_t] = False
    logits[0, mask] = float("-inf")

class DyadGPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config, device, raw_action_config, raw_val_action_config=None):
        super().__init__(vllm_config, device)
        self.num_embeddings=text_vocab_size(self.model_config.hf_config)
        self.raw_action_config=raw_action_config
        self._codegym_contexts = {}
        self._codegym_heads = {}
        # Cross-env validation compiles a second schema and uses the same direct-head
        # builder. With no val schema configured, schema switching is a no-op.
        self.raw_val_action_config=raw_val_action_config
        print(self.raw_action_config)
        self._get_dyad_tokenizer()
        self._init_action_config()
        self._init_dyad_encoder()
        # cross-env val state
        self._dyad_val_mode = False
        self._val_action_config = None
        self._saved_train_action_config = None
        self._saved_train_head = None
        self._saved_train_action_size = None
        if raw_val_action_config is not None:
            self._val_action_config = self._build_action_config_from_raw(
                raw_val_action_config
            )
        self._dyad_logits_seq_keys: list[Any] | None = None
        # Unified router (router==unified) only: one ActionRouter per sequence plus its decision record.
        self.unified_routers: dict[Any, Any] = {}
        self.unified_decisions: dict[Any, list[int]] = {}
        self._unified_last_decision: dict[Any, Any] = {}

    def _build_logits_seq_keys(
            self,
            req_ids,
            num_scheduled_tokens_np,
            logits_indices,
    ) -> list[Any]:
        """
        Build the seq key corresponding to each row of logits.

        Key relation:
            sample_hidden_states = hidden_states[logits_indices]
            logits = compute_logits(sample_hidden_states)

        Therefore:
            logits[row] corresponds to logits_indices[row]
        and logits_indices[row] is then looked up to find which req_id it belongs to.
        """
        req_ids = list(req_ids)

        if hasattr(num_scheduled_tokens_np, "tolist"):
            num_scheduled_tokens = [
                int(x) for x in num_scheduled_tokens_np.tolist()
            ]
        else:
            num_scheduled_tokens = [int(x) for x in num_scheduled_tokens_np]

        if torch.is_tensor(logits_indices):
            logits_indices_list = [
                int(x) for x in logits_indices.detach().cpu().tolist()
            ]
        else:
            logits_indices_list = [int(x) for x in logits_indices]

        if len(req_ids) != len(num_scheduled_tokens):
            raise ValueError(
                "[Dyad] req_ids and num_scheduled_tokens length mismatch. "
                f"len(req_ids)={len(req_ids)}, "
                f"len(num_scheduled_tokens)={len(num_scheduled_tokens)}"
            )

        # hidden_states is flattened in the req order of the current batch:
        #
        # req_ids[0] -> hidden_states[start0:end0]
        # req_ids[1] -> hidden_states[start1:end1]
        # ...
        intervals = []

        cursor = 0
        for req_id, num_tokens in zip(req_ids, num_scheduled_tokens):
            start = cursor
            end = cursor + int(num_tokens)

            if end > start:
                intervals.append((start, end, req_id))

            cursor = end

        seq_keys = []

        for row_idx, logits_pos in enumerate(logits_indices_list):
            matched_req_id = None

            for start, end, req_id in intervals:
                if start <= logits_pos < end:
                    matched_req_id = req_id
                    break

            if matched_req_id is None:
                raise ValueError(
                    "[Dyad] Cannot map logits row to req_id. "
                    f"row_idx={row_idx}, logits_pos={logits_pos}, "
                    f"intervals={intervals}, "
                    f"req_ids={req_ids}, "
                    f"num_scheduled_tokens={num_scheduled_tokens}, "
                    f"logits_indices={logits_indices_list}"
                )

            seq_keys.append(matched_req_id)

        # This simplified implementation requires each logits row to map to a distinct seq state.
        # A duplicate here means req_id cannot distinguish several completions,
        # or logits_indices contains extra positions such as prompt logprobs.
        if len(set(seq_keys)) != len(seq_keys):
            raise ValueError(
                "[Dyad] Duplicate seq_keys detected. "
                "Current per-seq Dyad state requires one unique key per logits row. "
                f"seq_keys={seq_keys}. "
                "If this happens, check whether rollout.n shares one req_id, "
                "or whether prompt_logprobs/speculative path introduced extra logits rows."
            )

        return seq_keys

    def _init_dyad(self) -> None:
        if getattr(self, "dyad_logits_processor", None) is not None:
            return
        extra_cfg = getattr(self.vllm_config.load_config, "model_loader_extra_config", {}) or {}
        dyad_cfg = extra_cfg.get("dyad", None)

        # If no dyad config was supplied explicitly, start from an empty dict and keep going
        # with the "Dyad enabled by default" logic below.
        dyad_cfg = extra_cfg.get("dyad", {}) or {}

        # Dyad is on by default; it is only disabled when the config explicitly sets enabled=False.
        if dyad_cfg.get("enabled", True) is False:
            self.dyad_enabled = False
            self.dyad_logits_processor = None
            return

        self.dyad_enabled = True

        from .dyad_logits_processor import DyadLogitsProcessor

        # phase1 assumes every common generative model exposes lm_head.
        lm_head = _lm_head_of(self.model)

        meta_path = dyad_cfg.get("meta_path", None)

        if meta_path:
            # Loading meta from meta_path would go here, e.g.:
            # meta = DyadLogitsProcessor.load_meta(meta_path)
            raise NotImplementedError(
                "TODO: loading Dyad meta from path is not implemented in phase1."
            )
        else:
            use_bias = bool(dyad_cfg.get("use_bias", False))

            meta = {
                "hidden_size": lm_head.weight.shape[1],
                "dyad_action_size": self.action_config["total_size"],
                "use_bias": use_bias,
            }

        # Keep a copy of meta around for later debugging or export.
        self.dyad_meta = meta
        print("[Dyad] dyad_meta =", self.dyad_meta)
        self.dyad_logits_processor = DyadLogitsProcessor(
            hidden_size=meta["hidden_size"],
            dyad_action_size=meta["dyad_action_size"],
            use_bias=meta["use_bias"],
            dtype=lm_head.weight.dtype,
            device=lm_head.weight.device,
            action_config=self.action_config
        )

        # Allocation may precede the first sync. A placeholder is never a sampling head.
        self._dyad_head_ready = False
        ckpt_path = dyad_cfg.get("ckpt_path", None)
        if ckpt_path:
            self.dyad_logits_processor.load_dyad_weights(ckpt_path)
        if getattr(self, "_dyad_adapter_synced", False):
            self._init_dyad_weights_from_vocab()
        self.dyad_logits_processor.eval()
        log_event(
            "dyad_vllm_model_runner",
            "action_head_initialized",
            action_size=self.action_config["total_size"],
            action_head=module_parameter_summary(self.dyad_logits_processor.action_head),
        )

    @torch.inference_mode()
    def reinit_action_head_from_lm_head(self) -> None:
        """Materialize the head from the synchronized direct encoder state.

        The existing name is retained for the rollout weight-sync contract.
        Before allocation, the first execute_model will build it after synchronization.
        """
        if getattr(self, "dyad_logits_processor", None) is None:
            return
        # In val mode the active head is the val head that enter_val_mode just built from the
        # current lm_head, so there is nothing to do here; besides, the weight sync happens
        # before enter_val_mode, so normally we are not in val mode here (defensive skip).
        if getattr(self, "_dyad_val_mode", False):
            return
        self._init_dyad_weights_from_vocab()
        # Record the rebuilt head alongside the synchronized projector version.
        log_event(
            "dyad_vllm_model_runner",
            "action_head_initialized",
            action_size=self.action_config["total_size"],
            action_head=module_parameter_summary(self.dyad_logits_processor.action_head),
        )

    # ------------------------------------------------------------------
    # Cross-env validation switches schemas and rebuilds through the direct encoder.
    # The caller must synchronize encoder data suitable for the target schema.
    # ------------------------------------------------------------------
    def _dyad_reset_seq_state(self) -> None:
        """Clear all per-seq state so nothing leaks across schemas when switching train/val."""
        self.unified_routers.clear()
        self.unified_decisions.clear()
        self._unified_last_decision.clear()

    @torch.inference_mode()
    def _build_head_init_weight_for(self, action_config: dict) -> torch.Tensor:
        """Build a schema head from the synchronized encoder and projector."""
        lm_head = _lm_head_of(self.model)
        # Same entry point as the policy LLM backbone side (agent_system/policies/dyad/models/action_head_factory.py): the two
        # heads must be numerically identical, and one function is how that stays true.
        encoder = self._action_head_encoder()
        return action_encoder.build_action_head(
            action_config,
            lm_head.weight,
            self._dyad_tokenizer,
            llm_cfg=self._dyad_llm_encoder_config,
            encoder=encoder,
            residual_head=self._dyad_residual_head,
        )

    @torch.inference_mode()
    def enter_val_mode(self) -> None:
        """Switch to the val schema and materialize its direct encoder head."""
        if self._val_action_config is None or self._dyad_val_mode:
            return
        if getattr(self, "dyad_logits_processor", None) is None:
            # The logits processor is not initialized yet (built on the first execute_model);
            # mark the switch as deferred until then.
            return
        # Snapshot the training state so it can be restored exactly.
        self._saved_train_action_config = self.action_config
        self._saved_train_head = self.dyad_logits_processor.action_head
        self._saved_train_action_size = self.dyad_logits_processor.dyad_action_size
        # Switch the active config to val (the mask state machine reads self.action_config, so
        # switching this switches the mask).
        self.action_config = self._val_action_config
        # Rebuild through the shared direct encoder entry point.
        init_weight = self._build_head_init_weight_for(self.action_config)
        ref_head = self._saved_train_head
        hidden = ref_head.weight.shape[1]
        val_size = int(self.action_config["total_size"])
        new_head = torch.nn.Linear(
            hidden,
            val_size,
            bias=ref_head.bias is not None,
            device=ref_head.weight.device,
            dtype=ref_head.weight.dtype,
        )
        with torch.no_grad():
            new_head.weight.copy_(
                init_weight.to(device=new_head.weight.device, dtype=new_head.weight.dtype)
            )
            if new_head.bias is not None:
                new_head.bias.zero_()
        self.dyad_logits_processor.action_head = new_head
        self.dyad_logits_processor.dyad_action_size = val_size
        if hasattr(self.dyad_logits_processor, "action_config"):
            self.dyad_logits_processor.action_config = self.action_config
        self._dyad_reset_seq_state()
        self._dyad_val_mode = True
        log_event(
            "dyad_vllm_model_runner",
            "enter_val_mode",
            val_action_size=val_size,
            train_action_size=int(self._saved_train_action_size or 0),
        )

    @torch.inference_mode()
    def exit_val_mode(self) -> None:
        """Restore the training schema and the training head (same module object, so later weight syncs still write to the right place)."""
        if not self._dyad_val_mode:
            return
        self.action_config = self._saved_train_action_config
        if self._saved_train_head is not None:
            self.dyad_logits_processor.action_head = self._saved_train_head
            self.dyad_logits_processor.dyad_action_size = int(self._saved_train_action_size)
        if hasattr(self.dyad_logits_processor, "action_config"):
            self.dyad_logits_processor.action_config = self.action_config
        self._dyad_reset_seq_state()
        self._dyad_val_mode = False
        self._saved_train_head = None
        log_event("dyad_vllm_model_runner", "exit_val_mode")

    @torch.inference_mode()
        # >>> DYAD-BEGIN(vllm0.24) DYAD-REBASED: taken from 0.24 verbatim, with four Dyad hooks
    # inserted. Those four are Dyad's entire delta from upstream (every other difference is
    # indentation):
    #   1. _init_dyad()                 make sure action_head is ready before the forward (idempotent)
    #   2. _dyad_cleanup_finished()     drop per-sequence router state for finished requests
    #   3. _build_logits_seq_keys()     build the "logits row <-> sequence" mapping
    #   4. dyad_logits_processor()      append the extended-action logits after the base logits
    # 0.24 split execute_model and sample_tokens: the former only computes logits and stores
    # ExecuteModelState, sampling happens in the latter. So hook 4 lands in execute_model and the
    # masking lands in sample_tokens.
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        # >>> DYAD-BEGIN(dyad): make sure action_head and action_config are ready before each forward
        # _init_dyad is idempotent (it returns immediately when dyad_logits_processor is not None).
        # It lives here rather than in __init__ because dyad's weights can only be initialised from
        # lm_head once the model has finished loading.
        self._init_dyad()
        # Drop per-sequence router state for finished requests, or a reused request_id reads the
        # previous trajectory's state.
        self._dyad_cleanup_finished(scheduler_output)
        # <<< DYAD-END
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if self.routed_experts_initialized:
            self.routed_experts_capturer.clear_buffer()

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            deferred_state_corrections_fn = self._update_states(scheduler_output)

            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            # >>> DYAD-BEGIN(dyad): record which sequence each row of this batch's logits belongs to
            # dyad_logits_processor and the router both keep state **per sequence**, while logits are
            # flattened by row. This step pairs "logits row i <-> which request", and without it the
            # admissible action set cannot be applied to the right row.
            self._dyad_logits_seq_keys = self._build_logits_seq_keys(
                req_ids=list(req_ids[:num_reqs]),
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                logits_indices=logits_indices,
            )
            # <<< DYAD-END

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            has_separate_kv_update = not all(
                all(
                    g.backend.forward_includes_kv_cache_update
                    for g in self.attn_groups[id]
                )
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            if self.cache_config.mamba_cache_mode == "align":
                # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                # to decide copy operations, so we must apply deferred
                # corrections before it runs.
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                mamba_bufs = self._get_mamba_bufs()
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    mamba_bufs.preprocess,
                )
                # preprocess_mamba resets num_accepted_tokens_cpu to 1
                # for requests whose state was copied to a new block.
                # Re-sync to GPU so the mamba kernel reads from the
                # correct initial state slot (init_token_idx = 0).
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.copy_to_gpu(num_reqs)

                # Stage per-request inputs for the fused postprocess kernel
                # only when that kernel will actually run. The kernel is
                # gated on spec-decode + hybrid (see MambaBuffers.create);
                # without it, ``mamba_bufs.postprocess_align`` is None and
                # the staging buffers don't exist.
                if mamba_bufs.postprocess_align is not None:
                    mamba_utils.stage_postprocess_inputs_to_gpu(
                        mamba_bufs.postprocess_align,
                        scheduler_output,
                        self.input_batch.req_ids,
                        num_reqs,
                        self.requests,
                        self.mamba_state_idx,
                    )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded,
                num_reqs_padded=(
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
                ),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )

            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    slot_mappings=slot_mappings_by_group,
                )
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        defer_kv_connector_finalize = self.speculative_config is not None
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
                skip_compiled=has_encoder_input,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                defer_finalize=defer_kv_connector_finalize,
            ) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)

                # >>> DYAD-BEGIN(dyad): append action_head's extended-action logits after the base ones
                # This is the single point where Dyad diverges from an ordinary rollout: the
                # vocabulary widens from V to V+K, and the last K columns come from
                # action_head(hidden_states).
                # It must use sample_hidden_states (already row-selected by logits_indices), not the
                # full hidden_states -- a row-count mismatch is an outright illegal memory access.
                if logits is not None and self.dyad_logits_processor is not None:
                    logits = self._append_task_logits(logits, sample_hidden_states)
                # <<< DYAD-END
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

                # >>> DYAD-BEGIN(dyad): append action_head's extended-action logits after the base ones
                # This is the single point where Dyad diverges from an ordinary rollout: the
                # vocabulary widens from V to V+K, and the last K columns come from
                # action_head(hidden_states).
                # It must use sample_hidden_states (already row-selected by logits_indices), not the
                # full hidden_states -- a row-count mismatch is an outright illegal memory access.
                if logits is not None and self.dyad_logits_processor is not None:
                    logits = self._append_task_logits(logits, sample_hidden_states)
                # <<< DYAD-END

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        return None

    @torch.inference_mode
        # >>> DYAD-BEGIN(vllm0.24) DYAD-REBASED: taken from 0.24 verbatim, with three Dyad blocks
    #   A before sampling: take router.decision() per sequence and mask to the admissible action set
    #                      (the sampling side of the invariant)
    #   B after sampling:  advance the router with the token actually sampled, writing back the
    #                      written token where needed
    #   C on output:       add_action_content fills in the action chosen this step
    # All four runtime checks (empty seq_keys / length mismatch / ndim != 2 / first-dim mismatch) are
    # kept verbatim and every one of them raises outright. They and the four guards in
    # split_policy.compute_split_policy_outputs are two ends of one defence: this end guarantees
    # the admissible set used while sampling is right, that end guarantees training uses the same one.
    # Turn either end into a silent fallback and a disagreement between them becomes undetectable.
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if self.execute_model_state is None:
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and not get_pp_group().is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        # >>> DYAD-BEGIN(dyad): apply the admissible action set restriction before sampling
        # This is the **sampling side** of the non-negotiable requirement in AGENTS.md section 1:
        # whichever router's decision restricts the admissible set here, the training-side log-prob
        # replay must use the same one. The two sides stay in sync by sharing the ActionRouter class;
        # this code only applies the decision to the correct logits rows.
        num_seqs = int(logits.shape[0])
        seq_keys = self._dyad_logits_seq_keys
        # Clear it immediately: execute_model rebuilds it on every forward, and without clearing the
        # next batch reads the previous batch's mapping.
        self._dyad_logits_seq_keys = None

        if seq_keys is None:
            raise RuntimeError(
                "[Dyad] _dyad_logits_seq_keys is None. "
                "It should be built in execute_model() right after _prepare_inputs()."
            )
        if len(seq_keys) != num_seqs:
            raise ValueError(
                "[Dyad] seq_keys length does not match logits batch size. "
                f"len(seq_keys)={len(seq_keys)}, num_seqs={num_seqs}, "
                f"logits.shape={tuple(logits.shape)}, seq_keys={seq_keys}"
            )

        V_BASE = self.num_embeddings
        if self.action_config.get("router") != "unified":
            raise RuntimeError("Dyad rollout requires a compiled unified action schema")
        self._require_dyad_head_ready()

        # Collect every sequence's decision on CPU first (a pure state machine, very cheap), then
        # apply the GPU mask in one go. "the whole batch is free text" is the common case and takes a
        # vectorised fast path, avoiding N tiny per-row kernels.
        _decisions: list[Any] = []
        _all_free_text = True
        for seq_idx in range(num_seqs):
            seq_key = seq_keys[seq_idx]
            d = self._unified_get_router(seq_key).decision()
            self._unified_last_decision[seq_key] = d
            free = d.kind == "base_vocab"
            _decisions.append(d)
            _all_free_text = _all_free_text and free

        if _all_free_text:
            # Vectorised equivalent of the per-row row[0, V:] = -inf
            logits[:, V_BASE:] = float("-inf")
        else:
            for seq_idx in range(num_seqs):
                row_logits = logits[seq_idx : seq_idx + 1]
                d = _decisions[seq_idx]
                if d.kind == "force":
                    row_logits.fill_(float("-inf"))
                    row_logits[0, int(d.forced_token)] = 0.0
                elif d.kind == "base_vocab":
                    row_logits[0, V_BASE:] = float("-inf")
                else:
                    _apply_only_allowed_ids(row_logits, list(d.allowed_ids))
        # <<< DYAD-END

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        # >>> DYAD-BEGIN(dyad): advance the router with the sampled token, writing back the
        # written one. This must happen before _update_states_after_model_execute, which consumes
        # sampled_token_ids -- and the router may rewrite an "action id" into the token that actually
        # lands in the sequence.
        _sampled = sampler_output.sampled_token_ids
        if _sampled.ndim != 2:
            raise ValueError(
                f"[Dyad] expected sampled_token_ids ndim=2, got shape={tuple(_sampled.shape)}"
            )
        if _sampled.shape[0] != num_seqs:
            raise ValueError(
                "[Dyad] sampled_token_ids first dim does not match logits num_seqs. "
                f"sampled_token_ids.shape={tuple(_sampled.shape)}, num_seqs={num_seqs}"
            )

        # Fetch to CPU once, instead of N .item() calls and their N GPU synchronisations
        _sampled_col0 = _sampled[:, 0].tolist()
        _parse_ids = [0] * num_seqs
        for seq_idx in range(num_seqs):
            seq_key = seq_keys[seq_idx]
            raw_token_id = int(_sampled_col0[seq_idx])
            if self._dyad_sample_is_discarded(seq_key):
                # Chunked prefill samples are discarded by upstream bookkeeping.
                # They are neither emitted context nor real router decisions.
                _parse_ids[seq_idx] = raw_token_id
                continue
            _parse_ids[seq_idx] = int(self._unified_advance(seq_key, raw_token_id))
        # Write back in bulk only when the router really rewrote a token (an action step); on a
        # free-text step parse == raw and there is nothing to write.
        if _parse_ids != _sampled_col0:
            _sampled[:, 0] = torch.tensor(
                _parse_ids, device=_sampled.device, dtype=_sampled.dtype
            )
        # <<< DYAD-END

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(
                    sampler_output.sampled_token_ids
                )

        self._draft_token_ids = None
        self._draft_probs = None
        self._draft_prob_req_ids = None
        self._draft_token_req_ids = None
        self.valid_sampled_token_count_gpu = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            # Decide whether to run the drafter or zero out draft tokens.
            input_fits_in_drafter = self._input_fits_in_drafter(
                spec_decode_common_attn_metadata
            )
            use_gpu_toks = (
                spec_config.use_eagle()
                or spec_config.uses_draft_model()
                or spec_config.uses_extract_hidden_states()
            ) and not spec_config.disable_padded_drafter_batch
            if use_gpu_toks:
                # EAGLE/DraftModel speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DFlashProposer
                    | DraftModelProposer
                    | ExtractHiddenStatesProposer
                    | Gemma4Proposer,
                )
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = (
                        self.drafter.prepare_next_token_ids_padded(
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
            elif (
                spec_config.use_ngram_gpu()
                and not spec_config.disable_padded_drafter_batch
            ):
                assert isinstance(self.drafter, NgramProposerGPU)
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count, _ = (
                        self.drafter.update_token_ids_ngram(
                            sampled_token_ids,
                            self.input_batch,
                            self.token_ids_gpu_tensor,
                            self.num_tokens_no_spec_gpu,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

            if not input_fits_in_drafter:
                # Zero out draft tokens so the scheduler doesn't schedule
                # stale drafts from the previous step.
                # For Nemotron-H: it is necessary to zero out the draft tokens,
                # otherwise the stale tokens will corrupt Mamba recurrent
                # state and logprobs for sequences near max_model_len.
                self._draft_token_ids = torch.zeros(
                    1, device=self.device, dtype=torch.int32
                ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                self._draft_probs = None
                self._draft_prob_req_ids = None
                self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
            )

        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        # Finalize KV connector (wait_for_save + clear metadata) after
        # draft model runs. Deferred from target model forward to allow
        # draft model to also save its KV cache.
        if spec_config is not None:
            self.finalize_kv_connector()

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        # self.kv_connector_output may be modified during drafting
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                routed_experts=None,
            )

        # >>> DYAD-BEGIN(dyad): fill the action chosen this step into the output
        # Consumed by the agent loop's dyad_extract_tool_calls and by the training-side replay: the
        # text is a human-readable presentation, while action_content is what the policy actually
        # chose.
        self.add_action_content(output, seq_keys)
        # <<< DYAD-END

        if not self.use_async_scheduling:
            if self.routed_experts_initialized:
                # Sync path: D2H was issued in ``_bookkeeping_sync`` and
                # synchronized by ``_to_list``'s event.synchronize(), so
                # the pinned buffers are ready to be wrapped as numpy.
                total = scheduler_output.total_num_scheduled_tokens
                output.routed_experts = RoutedExpertsLists(
                    routing_data=self.routed_experts_cpu[:total].numpy(),
                    slot_mapping=self.routed_experts_slot_mapping_cpu[:total].numpy(),
                )
            return output

        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            # Async path: produce a device-side snapshot that the async
            # copy stream can D2H later. Both tensors must be private
            # clones because:
            #   - ``routing_data`` source is the shared capturer buffer,
            #     which is ``clear_buffer()``-ed at the start of the
            #     next step on the default stream.
            #   - ``slot_mapping`` source is our own
            #     ``routed_experts_slot_mapping_device``, which the
            #     next ``_prepare_inputs`` overwrites on the default
            #     stream while the D2H is still pending on the copy
            #     stream.
            # Without clones, the copy stream would read torn data.
            routed_experts_snapshot = None
            if self.routed_experts_initialized:
                buf = self.routed_experts_capturer.get_device_buffer()
                total = scheduler_output.total_num_scheduled_tokens
                routed_experts_snapshot = RoutedExpertsTensors(
                    routing_data=buf[:total].clone(),
                    slot_mapping=self.routed_experts_slot_mapping_device[
                        :total
                    ].clone(),
                )

            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
                vocab_size=self.input_batch.vocab_size,
                routed_experts=routed_experts_snapshot,
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def _get_dyad_tokenizer(self):
        tokenizer_name = getattr(self.model_config, "tokenizer", None)
        if tokenizer_name is None:
            tokenizer_name = self.model_config.model

        revision = getattr(self.model_config, "tokenizer_revision", None)
        if revision is None:
            revision = getattr(self.model_config, "revision", None)

        self._dyad_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=revision,
        )

    def _encode_text(self, text: str) -> list[int]:
        return list(self._dyad_tokenizer.encode(text, add_special_tokens=False))

    def _build_action_config_from_raw(self, raw):
        """Compile unified or CodeGym source schemas into the unified router contract.

        Preserve the source path for MCP definitions and parser serialization.
        """
        if not (isinstance(raw, dict) and (raw.get("router") == "unified" or raw.get("mode") == "codegym")):
            raise NotImplementedError(
                "_build_action_config_from_raw only supports router:unified or mode:codegym (every env satisfies it under the unified ActionRouter);"
                " the old ALFWorld build_action_config path has been removed."
            )
        yaml_path = raw.get("_parse_yaml_path")
        if yaml_path:
            # Share MCP, closed-value domains and provenance with policy replay.
            # Rebuilding just the YAML/MCP pair changes both metadata and candidates.
            from agent_system.policies.dyad.actions.schema_config import compile_schema_file

            cfg = compile_schema_file(self._dyad_tokenizer, self.num_embeddings, yaml_path)
        else:
            # Dynamic task/bootstrap schemas are constructed in memory, not files.
            from agent_system.policies.dyad.actions.schema_compiler import compile_action_schema

            cfg = compile_action_schema(self._dyad_tokenizer, self.num_embeddings, raw)
        if raw.get("codegym_action_capacity"):
            cfg["total_size"] = int(raw["codegym_action_capacity"])
        if cfg.get("router") != "unified":
            raise RuntimeError("Dyad schema compiler must produce a unified router config")
        return cfg

    def _init_action_config(self) -> None:
        self.action_config = self._build_action_config_from_raw(
            self.raw_action_config
        )

    def _init_dyad_encoder(self) -> None:
        """Allocate the direct projector without loading an encoder backbone.

        Encoder hidden states and masks arrive as buffers in the same weight stream
        as the projector parameters. Loading a second encoder in this worker would
        duplicate GPU memory and can deadlock Ray placement-group scheduling.
        The historical dyad_residual_head attribute is a persistent weight-key contract.
        """
        self._dyad_llm_encoder_config = None
        self._dyad_llm_encoder = None
        self._dyad_residual_head = None

        from agent_system.policies.dyad.models.action_head_factory import llm_encoder_config_from_env

        cfg = llm_encoder_config_from_env()
        if not cfg.enabled:
            raise RuntimeError("Dyad rollout requires the direct action encoder")

        from agent_system.policies.dyad.models.action_head import DirectActionHead
        from agent_system.policies.dyad.models.encoder_cache import CachedHiddenSource

        hidden = text_hidden_size(self.model_config.hf_config)
        # Projector dimensions must agree with the policy side; state loading validates them.
        direct_head = DirectActionHead(
            cfg.projector,
            hidden,
            hidden,
            scale=cfg.scale,
            projector_kwargs=cfg.projector_kwargs,
        )
        # eval(): the engine never back-propagates, and a projector left in train mode would apply
        # dropout to the head at sampling time while the trainer's copy did not.
        direct_head = direct_head.eval().to(self.device)
        self._dyad_llm_encoder_config = cfg
        self._dyad_residual_head = direct_head
        self._dyad_llm_encoder = CachedHiddenSource(direct_head)
        self._dyad_adapter_synced = False
        print(
            f"[Dyad] rollout encoder projector built: projector={cfg.projector} hidden={hidden} "
            "(no backbone here; the frozen encoder output arrives with the weight sync)",
            flush=True,
        )

    def _dyad_adapter_state(self) -> str:
        """`none` / `cold` / `synced`, printed next to every head summary.

        Which of the three it is decides whether the head just built can be compared with the
        policy LLM backbone's at all, so reading a head summary without it invites the wrong conclusion.
        """
        if self._dyad_residual_head is None:
            return "none"
        return "synced" if getattr(self, "_dyad_adapter_synced", False) else "cold"

    def _action_head_encoder(self):
        """Return the synced hidden source; a direct head has no base-only fallback."""
        if self._dyad_residual_head is None:
            raise RuntimeError("Dyad rollout requires the direct action encoder")
        if not getattr(self, "_dyad_adapter_synced", False):
            raise RuntimeError("Dyad action encoder/cache must be synced before building the head")
        return self._dyad_llm_encoder

    def _require_dyad_head_ready(self) -> None:
        """Never sample with an allocation-only or stale materialized action head."""
        self._action_head_encoder()
        if not getattr(self, "_dyad_head_ready", False):
            raise RuntimeError("Dyad action head must be rebuilt after encoder/cache synchronization")

    def load_dyad_residual_head(self, weights) -> int:
        """Load `dyad_residual_head.*` tensors from the policy LLM backbone's weight stream. Returns the count.

        Called by `dyad_vllm_rollout.update_weights` before the head is re-derived, so the projector
        and `lm_head` reach the head builder at the same version.

        Strict on purpose. A silently ignored tensor here means the engine keeps sampling with the
        projector from step 0 while the trainer moves on, and nothing in any metric would show it.
        """
        if self._dyad_residual_head is None:
            raise RuntimeError(
                "the policy LM's weight stream carries dyad_residual_head.* but this engine has no "
                "direct head. DYAD_ENCODER_ENABLED must be set the same on both sides; with it "
                "set only on the trainer the engine would sample with a different head than the "
                "one training computes log-probs from."
            )
        self._dyad_adapter_synced = False
        self._dyad_head_ready = False
        state = {name.split("dyad_residual_head.", 1)[1]: tensor for name, tensor in weights}

        # The encoder cache is handled apart from the parameters. Its shape is not known here until
        # the first sync arrives (it depends on the schema's prompt lengths), so it is assigned
        # rather than copied into a pre-sized buffer.
        cache_keys = ("encoder_hidden", "encoder_mask")
        cache = {k: state.pop(k) for k in cache_keys if k in state}
        if set(cache) != set(cache_keys):
            raise RuntimeError(
                f"the weight stream carried {sorted(cache)} of the encoder cache, expected "
                f"{sorted(cache_keys)}. Without it the engine has an projector and nothing to feed "
                "it, so the head cannot be built at all -- and the trainer's would still build "
                "fine, which is how this becomes a one-sided divergence."
            )
        self._dyad_residual_head.set_encoder_cache(
            cache["encoder_hidden"].to(self.device), cache["encoder_mask"].to(self.device)
        )

        target = self._dyad_residual_head.state_dict()
        target = {k: v for k, v in target.items() if k not in cache_keys}
        missing = set(target) - set(state)
        unexpected = set(state) - set(target)
        if missing or unexpected:
            raise RuntimeError(
                f"projector state mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}. "
                "The trainer and the engine built different projectors (projector kind or widths)."
            )
        with torch.no_grad():
            for name, tensor in state.items():
                param = target[name]
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))
        self._dyad_adapter_synced = True
        return len(state) + len(cache)

    # ------------------------------------------------------------------
    # Unified router (router==unified): every sequence drives its mask and state advance with an
    # ActionRouter. Rollout and training (router.build_unified_policy_trace) share the same
    # ActionRouter rules, which keeps the mask identical decision by decision (the top invariant).
    # ------------------------------------------------------------------
    def _text_only_request(self, seq_key):
        """Empty serving arguments preserve exactly the base-vocabulary distribution."""
        extra = self.requests[seq_key].sampling_params.extra_args or {}
        return extra.get("dyad_text_only") is True

    def _codegym_payload(self, seq_key):
        from agent_system.policies.dyad.actions.task_context import load_context
        request = self.requests[seq_key]
        extra_args = request.sampling_params.extra_args or {}
        path = extra_args.get("dyad_action_context") or extra_args.get("dyad_codegym_context")
        if not path:
            raise ValueError("Full CodeGym rollout requires a per-task encoder context")
        if seq_key not in self._codegym_contexts:
            payload = load_context(path)
            if payload["action_config"]["total_size"] != self.action_config["total_size"]:
                raise ValueError("CodeGym context action capacity disagrees with rollout")
            self._codegym_contexts[seq_key] = payload
        return self._codegym_contexts[seq_key]

    def _append_task_logits(self, logits, hidden):
        if not getattr(self, "raw_action_config", {}).get("codegym_action_capacity"):
            return self.dyad_logits_processor(base_logits=logits, hidden_states=hidden)
        from agent_system.policies.dyad.actions.codegym_tasks import context_head
        self._require_dyad_head_ready()
        keys = self._dyad_logits_seq_keys
        if keys is None or len(keys) != hidden.shape[0]:
            raise ValueError("CodeGym logits lost their per-request task mapping")
        rows = []
        for index, key in enumerate(keys):
            if self._text_only_request(key):
                # No encoder context, projector evaluation, or marker routing for this request.
                rows.append(logits.new_full((1, self.action_config["total_size"]), float("-inf")))
                continue
            if key not in self._codegym_heads:
                reference = _lm_head_of(self.model).weight.detach()
                self._codegym_heads[key] = context_head(self._codegym_payload(key),
                    self._dyad_residual_head, reference, self._dyad_tokenizer)
            rows.append(torch.nn.functional.linear(hidden[index:index+1],
                        self._codegym_heads[key].to(hidden.dtype)))
        return torch.cat([logits, torch.cat(rows).to(logits.dtype)], dim=-1)

    def _unified_get_router(self, seq_key):
        r = self.unified_routers.get(seq_key)
        if r is None:
            from agent_system.policies.dyad.actions.action_router import ActionRouter

            cfg = self.action_config
            if self._text_only_request(seq_key):
                cfg = {"router": "unified", "num_embeddings_size": self.num_embeddings,
                       "total_size": self.action_config["total_size"], "markers": {},
                       "actions": {}, "action_name_ids": {}}
            elif getattr(self, "raw_action_config", {}).get("codegym_action_capacity"):
                cfg = self._codegym_payload(seq_key)["action_config"]
            r = ActionRouter(cfg)
            self.unified_routers[seq_key] = r
            self.unified_decisions[seq_key] = []
        return r


    def _unified_advance(self, seq_key, raw_token_id: int) -> int:
        """Advance the ActionRouter and return the plain token to write into the context; record the real decision (used by trace)."""
        router = self.unified_routers[seq_key]
        d = self._unified_last_decision.get(seq_key)
        if d is not None and d.kind != "force":
            self.unified_decisions[seq_key].append(int(raw_token_id))
        out = router.advance(int(raw_token_id))
        if out:
            return int(out[0])
        # Fallback write-back of raw_token_id: only safe when it is a [base vocabulary] token.
        # ⚠️ If some phase consumed an [expanded id] yet produced no token at all (such
        # as ARGUMENT_KEY whose id_to_seq is empty when head.argument_key=true), an id >=vocab_size
        # would be written into input_ids, and the embedding lookup of the next forward goes out
        # of range -> CUDA device-side assert (with the stack inside _model_forward, which gives
        # no hint at all that Dyad is the cause). Catch it here as a readable error instead.
        # Architectural invariant: **every decision must write at least one base vocabulary
        # token** (one decode step writes one token).
        _V = int(self.action_config["num_embeddings_size"])
        if int(raw_token_id) >= _V:
            raise RuntimeError(
                f"Dyad unified router: phase {getattr(d, 'phase', '?')} consumed expanded id "
                f"{int(raw_token_id)}(>=vocab_size {_V}) but produced no token."
                f" That phase of schema={self.action_config.get('env_name')} must have a non-empty"
                f" id_to_seq (every decision writes at least one base vocabulary token),"
                f" otherwise the expanded id is written into input_ids and the embedding lookup"
                f" goes out of range."
            )
        return int(raw_token_id)

    def _dyad_cleanup_finished(self, scheduler_output) -> None:
        for key in scheduler_output.finished_req_ids:
            getattr(self, "_codegym_contexts", {}).pop(key, None)
            getattr(self, "_codegym_heads", {}).pop(key, None)
        """Clean up the Dyad per-seq state of finished requests (a reliable turn boundary).

        vLLM puts the requests that finished this step into scheduler_output.finished_req_ids,
        which covers both eos and length termination. seq_key is exactly req_id, so the matching
        Dyad state is dropped here to prevent leftovers from the previous turn polluting the next
        one when the same trajectory reuses a req_id across turns, and to reclaim memory.
        """
        finished = getattr(scheduler_output, "finished_req_ids", None)
        if not finished:
            return
        for req_id in finished:
            # The unified router path is reclaimed the same way.
            self.unified_routers.pop(req_id, None)
            self.unified_decisions.pop(req_id, None)
            self._unified_last_decision.pop(req_id, None)

    def _dyad_sample_is_discarded(self, seq_key) -> bool:
        """Use the same per-request validity mask as upstream token bookkeeping."""
        index = self.input_batch.req_id_to_index[seq_key]
        return bool(self.discard_request_mask.np[index])

    def add_action_content(
            self,
            output,
            seq_keys: list[Any] | None = None,
    ):
        """Publish ordered decision deltas, including terminal partial actions."""
        if self.action_config.get("router") != "unified":
            raise RuntimeError("Dyad rollout requires a compiled unified action schema")
        action_content_by_seq = {}
        for seq_key in (seq_keys or []):
            if self._dyad_sample_is_discarded(seq_key):
                continue
            router = self.unified_routers.get(seq_key)
            decs = self.unified_decisions.get(seq_key)
            decision = self._unified_last_decision.get(seq_key)
            if router is None or not decs or decision is None:
                continue
            # Transport every decode step, not only completed actions: the scheduler
            # and detokenizer discover EOS/length stops after this method returns.
            # A terminal partial action must still carry its sampled head decision.
            delta = [decs[-1]] if decision.kind != "force" else []
            complete = (
                router.phase.name == "NONE" and not router.pending
                and not router.none_recent and not router.in_enter_prefix
                and any(t >= self.num_embeddings for t in decs)
                # Native DIVE permits several tool calls in one assistant turn.
                and not router.cfg.get("native_tool_protocol")
            )
            payload = {
                "unified": True,
                "trace_delta": True,
                "raw_offset": len(decs) - len(delta),
                "raw_token_ids": delta,
                "action_complete": complete,
            }
            if len(decs) == len(delta):
                payload["action_config"] = router.cfg
            action_content_by_seq[str(seq_key)] = payload
            if complete:
                self.unified_decisions[seq_key] = []
        output.action_content = action_content_by_seq
        return output

    @torch.inference_mode()
    def _init_dyad_weights_from_vocab(self) -> None:
        if getattr(self, "raw_action_config", {}).get("codegym_action_capacity"):
            self._action_head_encoder()  # synchronization guard
            self._codegym_heads.clear()
            self._dyad_head_ready = True
            return
        if self.dyad_logits_processor is None:
            raise RuntimeError("dyad_logits_processor is not initialized.")

        target_weight = self.dyad_logits_processor.action_head.weight
        lm_head = _lm_head_of(self.model)

        # All initial, per-sync and validation materializations share this head builder.
        encoder = self._action_head_encoder()
        init_weight = action_encoder.build_action_head(
            self.action_config,
            lm_head.weight,
            self._dyad_tokenizer,
            llm_cfg=self._dyad_llm_encoder_config,
            encoder=encoder,
            residual_head=self._dyad_residual_head,
        )
        init_weight = init_weight.to(
            device=target_weight.device,
            dtype=target_weight.dtype,
        )

        with torch.no_grad():
            target_weight.copy_(init_weight)
            if self.dyad_logits_processor.action_head.bias is not None:
                self.dyad_logits_processor.action_head.bias.zero_()
        self._dyad_head_ready = True

        # The rollout half of the AGENTS.md section 1 pairing. The actor prints the same line with
        # side=actor after its own re-derive; the two sequences are supposed to agree, and with a
        # trainable projector nothing enforces that automatically any more.
        from agent_system.utils.diagnostics import action_head_summary

        self._action_head_rederive_count = getattr(self, "_action_head_rederive_count", 0) + 1
        print(
            f"[Dyad-HEAD] side=rollout n={self._action_head_rederive_count} "
            f"{action_head_summary(target_weight)} projector={self._dyad_adapter_state()}",
            flush=True,
        )