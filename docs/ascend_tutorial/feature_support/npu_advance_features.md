# Advanced NPU features

> Ascend NPU features and optimization options in the verl ecosystem.
>
Last updated: 05/13/2026.

---

## Contents

- [Advanced NPU features](#advanced-npu-features)
  - [Contents](#contents)
  - [1. Inference](#1-inference)
    - [1.1 vLLM](#11-vllm)
    - [1.2 SGLang](#12-sglang)
      - [SGLang configuration](#sglang-configuration)
  - [2. Training](#2-training)
    - [2.1 FSDP](#21-fsdp)
    - [2.2 Megatron](#22-megatron)
      - [MindSpeed patching](#mindspeed-patching)
      - [Megatron configuration](#megatron-configuration)
        - [Memory and compute](#memory-and-compute)
        - [Fused operators](#fused-operators)
        - [Pipeline parallelism](#pipeline-parallelism)
        - [Weights](#weights)
  - [3. Performance](#3-performance)
    - [3.1 Memory](#31-memory)
    - [3.2 Compute](#32-compute)
    - [3.3 Parallelism](#33-parallelism)
  - [4. Mixture of experts](#4-mixture-of-experts)
    - [Inference MoE](#inference-moe)
    - [Training MoE](#training-moe)
  - [5. Limitations](#5-limitations)
  - [Parameter reference](#parameter-reference)
    - [Inference parameters](#inference-parameters)
    - [Training parameters](#training-parameters)

---

## 1. Inference

verl supports vLLM and SGLang on Ascend NPUs.

### 1.1 vLLM

The vllm-ascend plugin follows the [pluggable backend RFC](https://github.com/vllm-project/vllm/issues/11162), separating Ascend integration from vLLM.

---

### 1.2 SGLang

Ascend support includes:

| Component | Description |
|:---|:---|
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | Ascend inference kernels for attention, normalization, activations and LoRA |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) | Ascend DeepEP kernels for MoE expert-parallel communication |

#### SGLang configuration

| SGLang parameter | verl parameter | Purpose |
|:---|:---|:---|
| `attention_backend` | `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | Set ascend for optimized NPU attention kernels |
| `quantization` | `actor_rollout_ref.rollout.quantization` | Quantized model loading and inference |


> See [SGLang NPU features](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features).

---

## 2. Training

### 2.1 FSDP

torch_npu provides NPU support for FSDP.

### 2.2 Megatron

MindSpeed adapts Megatron to NPUs by patching key components.

#### MindSpeed patching

**Entry:**
```python
from mindspeed.megatron_adaptor import repatch
```

**Call chain:**
```
repatch
├── Import megatron_adaptor.py
├── Import features_manager
├── Execute mindspeed/features_manager/__init__.py
├── Trigger @AutoExecuteFunction
├── Run patch_features()
└── Apply feature pre-patches and patches
```

**Components:**

| Component | Responsibility |
|:---|:---|
| Patch | Dynamically replace functions/classes, supporting stacked decorators |
| `parse_path()` | Import/create modules dynamically |
| `MindSpeedPatchesManager` | Global patch registry |
| `MindSpeedFeature` | Base feature class integrating patches |

#### Megatron configuration

##### Memory and compute

| verl parameter | Purpose |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs` | Release outputs after sending to the next PP stage; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` | Recompute full layers, selected attention components or none; default none |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` | Full-recompute method: uniform or block; default None |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` | Full-recompute layer count; larger groups trade compute for memory. Uniform groups must divide local layer count |

##### Fused operators

| verl parameter | Purpose |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` | Flash Attention; default true |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` | Fused RoPE; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` | Fused SwiGLU; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` | Persistent LayerNorm; default False |

##### Pipeline parallelism

| verl parameter | Purpose |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` | Count loss as a layer when splitting pipelines; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` | Count input embeddings as a layer; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` | First pipeline-stage layer count; default none |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` | Last pipeline-stage layer count; default none |

##### Weights

| verl parameter | Purpose |
|:---|:---|
| `actor_rollout_ref.actor.megatron.use_mbridge` | Use mbridge for weight conversion |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | Distributed checkpoint save/load; default False |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | Distributed checkpoint source path; default null |

---

## 3. Performance

### 3.1 Memory

| Feature | Stage/backend | Purpose |
|:---|:---|:---|
| KV cache release (free_cache_engine) | vLLM inference | Release cache after generation; enabled by default |
| Memory saver (enable_memory_saver) | SGLang inference | Release/restore device allocations; default True |
| Parameter offload (param_offload) | FSDP/Megatron training | Offload weights to CPU |
| Optimizer offload (optimizer_offload) | FSDP/Megatron training | Offload optimizer state to CPU |
| Chunked entropy (entropy_from_logits_with_chunking) | FSDP training | Reduce entropy-computation memory |
| Entropy chunk size (entropy_from_logits_chunk_size) | FSDP training | Set chunk size |
| Entropy checkpointing (entropy_checkpointing) | FSDP training | Recompute entropy intermediates |
| Pipeline output release (deallocate_pipeline_outputs) | Megatron training | Release sent tensors |
| Activation recomputation (recompute_granularity) | Megatron training | Full/selective/none |

### 3.2 Compute

| Feature | Stage/backend | Purpose |
|:---|:---|:---|
| Chunked prefill (enable_chunked_prefill) | vLLM inference | Batch prefill chunks with decode work |
| Prefix caching (enable_prefix_caching) | vLLM inference | Reuse shared-prefix computation |
| Flash Attention | Megatron training | Accelerate attention; enabled by default |
| Fused RoPE (use_fused_rotary_pos_emb) | Megatron training | Fuse rotary embedding operations |
| Fused SwiGLU (use_fused_swiglu) | Megatron training | Fuse activation operations |
| Persistent LayerNorm (persist_layer_norm) | Megatron training | Optimize normalization execution |
| Group GEMM (`moe_grouped_gemm`) | Megatron training | Grouped expert GEMM |

### 3.3 Parallelism

| Type | vLLM | SGLang | FSDP | Megatron | Purpose |
|:---|:---|:---|:---|:---|:---|
| Data (DP) | ✅ | ✅ | ✅ | ✅ | Partition data |
| Tensor (TP) | ✅ | ✅ | — | ✅ | Partition within layers |
| Pipeline (PP) | — | — | — | ✅ | Partition layers |
| Expert (EP) | ✅ | ✅ | — | ✅ | Partition experts |
| Sequence (SP/Ulysses) | ✅ | ✅ | ✅ | ✅ | Partition sequence dimensions |
| Context (CP) | ✅ | — | — | ✅ | Partition context computation |

---

## 4. Mixture of experts

### Inference MoE

- Configure expert parallelism through ep_size to distribute experts across NPUs.
- SGLang provides optimized communication through [Ascend DeepEP](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md).

### Training MoE

| verl parameter | Purpose |
|:---|:---|
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert parallel size; default 1 |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | Expert tensor-parallel size; default null |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` | Grouped expert GEMM; default False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` | Router/weighted-expert dtype, fp32 or fp64; default fp32 for numerical stability |

---

## 5. Limitations

1. **mbridge and VPP**
   - This documented stack does not support use_mbridge with virtual_pipeline_model_parallel_size.
   - Disable the default use_mbridge when selecting VPP.

2. **FSDP1/FSDP2**
   - forward_prefetch and use_orig_params apply only to FSDP1.
   - FSDP2 is recommended; check [Ascend API support](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/torch-distributed-fsdp.md).

3. **Recomputation dependencies**
   - recompute_method requires recompute_granularity=full.
   - recompute_num_layers also requires full.
   - For uniform recomputation, group size must divide the process's layer count.

4. **SGLang NPU settings**
   - Set attention_backend=ascend.
   - verl enables enable_memory_saver by default.

---

## Parameter reference

### Inference parameters

| Category | vLLM | SGLang | verl |
|:---|:---|:---|:---|
| Model path | `model_path` | `model_path` | `actor_rollout_ref.model.path` |
| Device memory | `gpu_memory_utilization` | `mem_fraction_static` | `actor_rollout_ref.rollout.gpu_memory_utilization` |
| Graph mode | `enforce_eager` | `disable_cuda_graph` | `actor_rollout_ref.rollout.enforce_eager` |
| Quantization | `quantization` | `quantization` | `actor_rollout_ref.rollout.quantization` |
| Maximum sequence length | `max_model_len` | — | `actor_rollout_ref.rollout.max_model_len` |
| Concurrency | `max_num_seqs` | `max_running_requests` | `actor_rollout_ref.rollout.max_num_seqs` |
| Tokenizer | `skip_tokenizer_init` | `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` |
| Remote code | `trust_remote_code` | `trust_remote_code` | `actor_rollout_ref.model.trust_remote_code` |
| TP | `tp_size` | `tp_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size` |
| DP | `dp_size` | `dp_size` | `actor_rollout_ref.rollout.data_parallel_size` |
| EP | `ep_size` | `ep_size` | `actor_rollout_ref.rollout.expert_parallel_size` |

### Training parameters

| Category | FSDP | Megatron |
|:---|:---|:---|
| Parameter offload | `fsdp_config.param_offload` | `megatron.param_offload` |
| Optimizer offload | `fsdp_config.optimizer_offload` | `megatron.optimizer_offload` |
| Sequence parallelism | `ulysses_sequence_parallel_size` | `context_parallel_size` |
| Flash Attention | — | `override_transformer_config.use_flash_attn` |
| Recomputation | — | `override_transformer_config.recompute_granularity` |
| Distributed checkpoint | — | `use_dist_checkpointing` |
