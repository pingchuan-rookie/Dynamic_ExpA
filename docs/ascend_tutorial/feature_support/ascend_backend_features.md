# Ascend Backend Features Guide
==================================================================================

Last updated: 03/03/2026.

This reference describes Ascend integration and backend options in verl. Availability and defaults depend on the pinned backend versions.

---

## Inference backends

verl supports vLLM and SGLang on Ascend NPUs.

### 1. vllm:

vllm-ascend provides a pluggable Ascend backend following the [vLLM RFC](https://github.com/vllm-project/vllm/issues/11162).

#### Parameters

| vLLM parameter | verl parameter | Purpose |
| --- | --- | --- |
| `model_path` | `actor_rollout_ref.model.path` | Model weight path |
| `gpu_memory_utilization` | `actor_rollout_ref.rollout.gpu_memory_utilization` | Device-memory fraction, from 0 to 1. A value of 1 leaves no headroom |
| `enforce_eager`| `actor_rollout_ref.rollout.enforce_eager` | Disable graph mode; default False |
| `enable_chunked_prefill`| `actor_rollout_ref.rollout.enable_chunked_prefill` | Split large prefills and batch them with decode requests |
| `free_cache_engine`| `actor_rollout_ref.rollout.free_cache_engine`  | Release KV cache after generation; default True |
| `max_model_len` | `actor_rollout_ref.rollout.max_model_len` | Maximum model sequence length |
| `tp_size`|  `actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size`| Tensor parallel degree |
| `dp_size`| `actor_rollout_ref.rollout.data_parallel_size`| Data parallel degree |
| `ep_size`| `actor_rollout_ref.rollout.expert_parallel_size`| Expert parallel degree |
| `node_rank`| Computed from instance/device topology | Node rank |
| `load_format`|  `actor_rollout_ref.rollout.load_format` | Weight loading format |
| `disable_log_stats`|  `actor_rollout_ref.rollout.disable_log_stats`| Disable rollout statistics |
| `nnodes`| Computed from instance/device topology | Nodes per instance |
| `trust_remote_code`| `actor_rollout_ref.model.trust_remote_code`| Allow repository-defined model code |
| `max_num_seqs` | `actor_rollout_ref.rollout.max_num_seqs` | Maximum concurrent requests |
| `max_num_batched_tokens`| `actor_rollout_ref.rollout.max_num_batched_tokens` | Maximum total tokens per batch |
| `skip_tokenizer_init`| `actor_rollout_ref.rollout.skip_tokenizer_init` | Skip tokenizer initialization and send input_ids |
| `enable_prefix_caching` | `actor_rollout_ref.rollout.enable_prefix_caching` | Enable automatic prefix caching |
| `quantization`| actor_rollout_ref.rollout.quantization; default None | Quantization method |

### 2. sglang:

SGLang's Ascend support is maintained in its community.
Related components include:

| Component | Description |
| --- | --- |
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | Ascend inference kernels for attention, normalization, activations and LoRA |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) | Ascend DeepEP communication kernels for MoE expert parallelism |

#### Parameters

Rollout configuration supplies common options and backend-specific engine_kwargs.
Common SGLang options are below; see [NPU feature support](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features) for more.

| SGLang parameter | verl parameter | Purpose |
| --- | --- | --- |
| model_path | actor_rollout_ref.model.path| Model weight path |
| mem_fraction_static| actor_rollout_ref.rollout.gpu_memory_utilization | Static memory fraction for weights and KV cache |
| disable_cuda_graph| actor_rollout_ref.rollout.enforce_eager| Disable graph mode; default False |
| enable_memory_saver| Default True in verl | Enable release_memory_occupation/resume_memory_occupation |
| base_gpu_id| Computed from topology | Initial device ID per instance |
| gpu_id_step| Default 1 | Stride between device IDs |
| tp_size|  actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size| Tensor parallel degree |
| dp_size| actor_rollout_ref.rollout.data_parallel_size| Data parallel degree |
| ep_size| actor_rollout_ref.rollout.expert_parallel_size| Expert parallel degree |
| node_rank| Computed from topology | Node rank |
| load_format|  actor_rollout_ref.rollout.load_format| Weight format |
| dist_init_addr| Computed automatically | Distributed initialization address |
| nnodes| Computed from topology | Nodes per instance |
| trust_remote_code| actor_rollout_ref.model.trust_remote_code| Allow repository-defined model code |
| max_running_requests| actor_rollout_ref.rollout.max_num_seqs | Maximum concurrent requests |
| log_level| Default error | Log level |
| skip_tokenizer_init| actor_rollout_ref.rollout.skip_tokenizer_init | Skip tokenizer initialization and send input_ids |
| skip_server_warmup| Default True | Skip warmup |
| quantization| actor_rollout_ref.rollout.quantization; default None | Quantization method |
| attention_backend|actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend| Use ascend for NPU attention |

---

## Training backends

### 1. FSDP

torch_npu provides FSDP support. See [Ascend API compatibility](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/torch-distributed-fsdp.md).

#### FSDP1
##### Parameters
| verl parameter | Purpose |
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | CPU parameter offload; default False |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | CPU optimizer offload; default False |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | Reshard parameters after forward and all-gather again for backward; default True |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | NPUs per FSDP group; -1 selects automatically |

| `actor_rollout_ref.actor.fsdp_config.forward_prefetch`  | Prefetch the next forward all-gather; FSDP1 only, default False |
| `actor_rollout_ref.actor.fsdp_config.use_orig_params` | Use original module parameters; FSDP1 only, default False |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`| Ulysses sequence parallel degree |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`| Chunk entropy computation; default False |
| `actor_rollout_ref.actor.entropy_from_logits_chunk_size`| Entropy chunk size; default 2048 |
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`| Recompute entropy intermediates; default False |
| `actor_rollout_ref.actor.fsdp_config.forward_only` | Forward-only execution; default False |

#### FSDP2
##### Parameters
| verl parameter | Purpose |
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | CPU parameter offload; default False |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | CPU optimizer offload; default False |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | Reshard after forward, all-gather for backward; default True |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | NPUs per FSDP group; -1 selects automatically |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`| Ulysses sequence parallel degree |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`| Chunk entropy computation; default False |
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`| Recompute entropy intermediates; default False |
| `actor_rollout_ref.actor.fsdp_config.forward_only` | Forward-only execution; default False |



### 2. Megatron

Megatron provides model-parallel training. MindSpeed supplies the adaptations needed to run it on NPUs.

MindSpeed replaces components through monkey patches.

* MindSpeed patch framework

verl triggers patching through `from mindspeed.megatron_adaptor import repatch`:

~~~
from mindspeed.megatron_adaptor import repatch
├── Import megatron_adaptor.py
├── Import features_manager
├── Execute mindspeed/features_manager/__init__.py
├── Trigger @AutoExecuteFunction
├── Run patch_features()
└── Run apply_features_pre_patches and apply_features_patches
~~~

Patch implements dynamic function/class replacement.

~~~python
class Patch:
~~~

parse_path imports and creates modules dynamically.

~~~python
def parse_path(module_path, function_name, create_dummy):
~~~

The patch system supports stacked decorators.

~~~python
def apply_patch(self):
    final_patch_func = self.orig_func
    if self.patch_func is not None:
        final_patch_func = self.patch_func

    # Apply registered decorators.
    for wrapper in self.wrappers:
        final_patch_func = wrapper(final_patch_func)
~~~

* MindSpeedPatchesManager

MindSpeedPatchesManager is the global patch registry.

~~~python
class MindSpeedPatchesManager:
    patches_info: Dict[str, Patch] = {}
~~~

* Feature integration

Features inherit MindSpeedFeature to integrate with patching.

~~~python
class MindSpeedFeature:
    """Base class for mindspeed features."""

    def __init__(self, feature_name: str, optimization_level: int = 2):
        self.feature_name = feature_name.lower().strip().replace('-', '_')
        self.optimization_level = optimization_level
        self.default_patches = self.optimization_level == 0

    def is_need_apply(self, args):
        """Check the feature is need to apply."""
        return (self.optimization_level <= args.optimization_level and getattr(args, self.feature_name, None)) \
            or self.default_patches

    def register_args(self, parser: ArgumentParser):
        """Register cli arguments to enable the feature."""
        pass

    def pre_validate_args(self, args: Namespace):
        """Validate the arguments of mindspeed before megatron args validation
        and store some arguments of the mindspeed temporarily,
        in case that megatron validate fails.
        for example:
            ```python
            origin_context_parallel_size = args.context_parallel_size
            args.context_parallel_size = 1
            ```
        """
        pass

    def validate_args(self, args: Namespace):
        """Restore the arguments of the mindspeed.

        for example:
        ```python
        args.context_parallel_size = origin_context_parallel_size
        ```
        """
        pass

    def post_validate_args(self, args: Namespace):
        """validate mindspeed arguments after megatron arguments validation."""
        pass

    def pre_register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace):
        """Register all patch functions before import megatron"""
        pass

    def register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace):
        """Register all patch functions the feature is related."""
        pass

    def incompatible_check(self, global_args, check_args):
        """Register all incompatible functions the feature is related."""
        if getattr(global_args, self.feature_name, None) and getattr(global_args, check_args, None):
            raise AssertionError('{} and {} are incompatible.'.format(self.feature_name, check_args))

    def dependency_check(self, global_args, check_args):
        """Register all dependency functions the feature is related."""
        if getattr(global_args, self.feature_name, None) and not getattr(global_args, check_args, None):
            raise AssertionError('{} requires {}.'.format(self.feature_name, check_args))

    @staticmethod
    def add_parser_argument_choices_value(parser, argument_name, new_choice):
        """Add a new choice value to the existing choices of a parser argument."""
        for action in parser._actions:
            exist_arg = isinstance(action, argparse.Action) and argument_name in action.option_strings
            if exist_arg and action.choices is not None and new_choice not in action.choices:
                action.choices.append(new_choice)
~~~

#### Parameters
| verl parameter | Purpose |
| --- | --- |
| `actor_rollout_ref.actor.megatron.optimizer_offload` | CPU optimizer offload; default False |
| `actor_rollout_ref.actor.megatron.use_mbridge` | Enable mbridge, default True. The engine passes a bridge to checkpoint management for model/huggingface/. hf_model save/load contents require a bridge. It can coexist with use_dist_checkpointing to retain HF exports and model/dist_ckpt/ shards. Without mbridge, model-only sharded checkpoints require use_dist_checkpointing=True |
| `actor_rollout_ref.actor.megatron.param_offload` | CPU parameter offload; default False |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | Tensor parallel degree; default 1 |
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size`  | Pipeline parallel degree; default 1 |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert parallel degree; default 1 |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size`| Expert tensor parallel degree; default null |
| `actor_rollout_ref.actor.context_parallel_size`| Context parallel degree; default 1 |
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs`| Release outputs after sending to the next PP stage; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` | Persistent LayerNorm; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` | Grouped GEMM; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` | Routing/weighted-expert dtype, fp32 or fp64; default fp32 |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` | Count loss as a layer when splitting pipelines; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` | Count input embeddings as a layer; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` | Recompute full layers, selected core attention or none; default none |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` | Full-recompute method: uniform or block; default None |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` | Full-recompute group layer count; default None. Uniform groups must divide local layers. For example: full/uniform/4 |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | Use Megatron shards under model/dist_ckpt/ for model contents. Independent of mbridge and compatible with simultaneous HF export; default False |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | Distributed checkpoint source path; default null |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` | Flash Attention; default true |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` | Fused RoPE; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` | Fused SwiGLU; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` | First pipeline-stage layers; default none |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` | Last pipeline-stage layers; default none |

This documented stack does not support use_mbridge with virtual_pipeline_model_parallel_size. Disable default mbridge when selecting VPP.

### 3. VeOmni

VeOmni provides an FSDP-based training backend with parallelism and optimization options for large models and MoE.

#### Parameters

| verl parameter | Purpose |
| --- | --- |
| `actor_rollout_ref.actor.veomni.param_offload` | CPU parameter offload; default False |
| `actor_rollout_ref.actor.veomni.optimizer_offload` | CPU optimizer offload; default False |
| `actor_rollout_ref.actor.veomni.fsdp_size` | FSDP group size; -1 selects automatically |
| `actor_rollout_ref.actor.veomni.ulysses_parallel_size` | Ulysses degree; default 1 |
| `actor_rollout_ref.actor.veomni.expert_parallel_size` | Expert parallel degree; default 1 |
| `actor_rollout_ref.actor.veomni.mixed_precision` | Mixed precision; default true |
| `actor_rollout_ref.actor.veomni.enable_full_shard` | Full sharding/ZeRO-3; default true |
| `actor_rollout_ref.actor.veomni.forward_prefetch` | Forward all-gather prefetch; default true |
| `actor_rollout_ref.actor.veomni.attn_implementation` | Attention implementation: eager, sdpa, flash_attention_2/3, veomni_flash_attention_2/3_with_sp, native-sparse |
| `actor_rollout_ref.actor.veomni.moe_implementation` | MoE implementation: eager or fused; default fused |
| `actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation` | Cross-entropy implementation; default eager |
| `actor_rollout_ref.actor.veomni.rms_norm_implementation` | RMSNorm implementation; default eager |
| `actor_rollout_ref.actor.veomni.swiglu_mlp_implementation` | SwiGLU MLP implementation; default eager |
| `actor_rollout_ref.actor.veomni.rotary_pos_emb_implementation` | RoPE implementation; default eager |
| `actor_rollout_ref.actor.veomni.load_balancing_loss_implementation` | MoE load-balancing loss implementation; default eager |
| `actor_rollout_ref.actor.veomni.use_torch_compile` | torch.compile; default false |
| `actor_rollout_ref.actor.veomni.forward_only` | Forward-only; default false |
| `actor_rollout_ref.actor.veomni.enable_fsdp_offload` | FSDP CPU offload; default false |
| `actor_rollout_ref.actor.veomni.enable_reentrant` | Reentrant checkpointing; default false |
| `actor_rollout_ref.actor.veomni.ckpt_manager` | Checkpoint manager; default dcp |
| `actor_rollout_ref.actor.veomni.init_device` | Weight initialization device: cpu/cuda/meta/npu; default meta |
| `actor_rollout_ref.actor.veomni.activation_gpu_limit` | GPU activation limit during offload, in GB; default 0.0 |
| `actor_rollout_ref.rollout.moe_load_balance_metrics_interval` | Rollout MoE load-metric interval; default 0 disables it. Requires enable_rollout_routing_replay |

#### Router replay

Configure VeOmni MoE replay through actor_rollout_ref.actor.veomni.router_replay:

| Parameter | Purpose |
| --- | --- |
| `mode` | disabled, R2 recording/replay, or R3 rollout recording/replay |
| `record_file` | Route-recording path, required for R2/R3 |
| `replay_file` | Route-loading path, required for replay |

#### Example

Example VeOmni MoE/GRPO configuration:

```bash
# Select the VeOmni backend.
model_engine=veomni

# Parallelism.
actor_rollout_ref.actor.veomni.fsdp_size=16
actor_rollout_ref.actor.veomni.ulysses_parallel_size=1
actor_rollout_ref.actor.veomni.expert_parallel_size=1

# Memory settings.
actor_rollout_ref.actor.veomni.param_offload=True
actor_rollout_ref.actor.veomni.optimizer_offload=True

# Operator implementations.
actor_rollout_ref.actor.veomni.attn_implementation=veomni_flash_attention_2_with_sp
actor_rollout_ref.actor.veomni.moe_implementation=fused
```

#### Features

- Combine data, Ulysses and expert parallelism.
- Offload parameters, optimizer state and activations.
- Use fused MoE and router replay.
- Select attention/MLP implementations for the hardware.
- Support NVIDIA GPUs and Huawei Ascend NPUs within the compatible stack.
