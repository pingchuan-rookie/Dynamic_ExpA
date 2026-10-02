# Transfer to NPU guide

Last updated: 05/14/2026

This upstream guide covers GPU-to-NPU migration and native NPU model adaptation: setup, component integration, numerical alignment, tuning and evaluation.

## 1. Preparation

Prepare a runtime that loads the model and reads data before integration debugging.


### 1.1 Hardware and dependencies

Follow the [installation guide](../../get_start/install_guidance.rst). If the model needs different vLLM/vllm-ascend, Megatron/MindSpeed or transformers versions, use the versions required by its actual adapter.

### 1.2 Weights

BF16 is the mixed-precision baseline for this FSDP/Megatron stack. Convert quantized weights to the intended BF16 reference before comparison. This guide's A2/A3 stack does not support FP8 training; check release-specific Ascend 950 support separately.

### 1.3 Data

Follow [post-training data preparation](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) to produce Parquet with required reward fields.

## 2. Component integration

verl separates inference, training and checkpoint/weight synchronization. Validate inference engines, training engines and Megatron-Bridge individually before integrating the full loop. See [backend support](../../feature_support/ascend_backend_features.md).

### 2.1 Inference

Abstract interfaces support multiple engines, including vLLM and SGLang.

First follow [vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/main/docs/source/tutorials/models) or [SGLang](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/basic_usage) model guides to verify one instance: model/tokenizer initialization, single/batched generation, stop conditions and long context.

### 2.2 Training

The Engine interface separates scheduling from FSDP, Megatron and MindSpeed-LLM implementations.

NPU detection through is_npu_available applies device patches. Select the supported training backend and its configuration; see [MindSpeed/verl features](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md).

### 2.3 Megatron-Bridge

Megatron-Bridge converts between HF weights used in inference and mcore weights used in training. Enable it with:

```
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.vanilla_mbridge=False
```

Check [supported models](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/README.md). Special structures may still need custom NPU conversion.

For DeepSeek Sparse Attention, MindSpeed's absorbed-matrix path splits linear_kv_up_proj into linear_k_up_proj and linear_v_up_proj. Convert these from HF self_attn.kv_b_proj.weight; the original integration did not implement that split.

Correct conversion enables the absorbed matrices required by [sparse_flash_attention](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention) and [lightning_indexer](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer). These fused operators can reduce memory traffic and computation overhead.

### 2.4 Full integration

After component validation, configure the full loop using the [parameter guide](parameter_and_metrics.md) and verify end-to-end behavior.

## 3. Numerical alignment

Numerical problems can originate in training, inference or their interaction. Alignment is essential for reproducible diagnosis.

See [precision alignment](../precision_analysis/precision_alignment_zh.md) for stage-specific checks. The following focuses on training/rollout agreement using msprobe.

### 3.1 Monitoring

Enable actor_rollout_ref.rollout.calculate_log_probs=True and inspect:

* **Training/rollout agreement:**
  * training/rollout_probs_diff_mean. The source uses 0.01 as a diagnostic threshold; compare with the matched GPU baseline rather than treating it as universal.
  * training/rollout_probs_diff_max.
  * training/rollout_actor_probs_pearson_corr.
* **Training stability:**
  * actor/grad_norm. Inspect unexpected changes alongside rewards/losses; a decreasing norm alone does not establish convergence.

Set trainer.rollout_data_dir=./rollout_dump/ and inspect saved responses for malformed or repeated text.

### 3.2 Capture

If discrepancies exceed the expected baseline, use [msprobe](../precision_analysis/precision_debugger_zh.md) to capture tensors.

### 3.3 Locate discrepancies

Read construction.json and compare modules. First align layer.0.input_layernorm inputs, then find the first divergent output.

Small differences can accumulate across large models and produce large probability changes. Match semantics, dtypes and operation order before choosing numerical tolerances.

Use authoritative source and model reports to distinguish valid numerical differences from implementation errors.

#### 3.3.1 Common causes

Typical causes include:

1. **Implementation:** mathematically equivalent decompositions versus actual missing scales or extra operations.
2. **Dtypes:** BF16 paths versus normalization that promotes to FP32 and casts back.
3. **Hyperparameters:** inconsistent hardcoded LayerNorm epsilon.
4. **Parallelism:** different partitioning/batching changes floating-point accumulation order.
5. **Randomness:** differing dropout or sampling behavior.

The following GLM-5 migration examples illustrate these cases.

#### 3.3.2 FFN activation mismatch

Comparison found the first discrepancy in the first layer's MLP activation.

Inference used fused npu_swiglu while training used the original decomposed GLU.

* **Cause:** Megatron-Bridge had not set provider.bias_activation_fusion=True despite the verl SwiGLU setting.
* **Correction:** enable the corresponding bridge option:
  ```
  +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
  ```

#### 3.3.3 indexer_k_norm dtype and epsilon

Two differences appeared in indexer_k_norm:

* Inference used F.layer_norm(x.float(), ...).type_as(x), while the Megatron training path used BF16.
* Align the training promotion/cast-back behavior.
* GLM5 inference inherited a DeepSeek-V3.2 epsilon of 1e-6; the training engine and model report used 1e-5.
* Set inference epsilon to 1e-5.

```
self.k_norm=LayerNorm(self.head_dim,eps=1e-6 -> 1e-5)
```

#### 3.3.4 lightning_indexer logic

Comparison found missing and extra operations:

* Inference omitted weights scaling present in Megatron, slime and transformers. Restore the scale:

```
weights, _ = self.weights_proj(x)
+weights = weights * (self.n_head**-0.5) * (self.head_dim**-0.5)
```

* Training included rotate_activation, a Hadamard transform intended for quantization. Remove it for this BF16 path, following [transformers PR #45017](https://github.com/huggingface/transformers/pull/45017).

```
class DSAIndexer(MegatronModule):
    def forward_with_scores(
-		q = rotate_activation(q)
-		k = rotate_activation(k)
```

### 3.4 MoE routing stability

RL commonly samples with vLLM and trains those samples with Megatron.

In MoE models, small numerical differences can change expert selection and thus the entire computation path.

This changes the optimization signal between generation and training and can destabilize learning.

Routing replay fixes selected expert paths against such perturbations. Common variants are:

* **R2:** actor_rollout_ref.actor.router_replay.mode="R2".

  * Record routes in the training engine's old-policy forward pass and replay them during updates.
  * This reduces route changes caused by policy updates.
* **R3:** actor_rollout_ref.actor.megatron.router_replay.mode="R3".

  * Capture inference routes during generation and replay them in training.
  * This aligns selected paths across engines as well as updates; other numerical differences still require validation.

The source describes R3 use for DeepSeek-V3.2, GLM-5 and MiMo-V2. Support depends on the selected backend and model.

For supported large MoE models, configure R3 as follows:

```
actor_rollout_ref.actor.router_replay.mode="R3" \
actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
```

## 4. Performance

Start with [verl tuning](https://github.com/verl-project/verl/blob/04833f01/docs/perf/perf_tuning.rst). Capture data, locate bottlenecks, adjust configuration and remeasure with matched semantics.

1. [Ascend Performance Analysis Guide](../performance/ascend_performance_analysis_guide.md)
2. [Profiling configuration](../performance/ascend_profiling_en.rst)

### 4.1 Inference

Rollout can dominate runtime. Common options include:

1. **Graph mode:** reduce repeated dispatch overhead through captured execution.
2. **CPU affinity:** improve launch efficiency; vllm-ascend enabled it by default on Arm servers from v0.18.0rc1.
3. **AIV communication:** set HCCL_OP_EXPANSION_MODE=AIV for Vector Core expansion.
4. **Asynchronous scheduling:** overlap scheduler work with worker execution.

Example settings:

```
# Graph mode.
actor_rollout_ref.rollout.enforce_eager=False
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[2, 4, 8, 16, 24, 32]"
# CPU affinity.
++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
# Asynchronous scheduling.
++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True
```

### 4.2 Training

Variable response lengths and memory pressure affect updates. See [MindSpeed/verl](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md) for supported options:

1. Fuse RoPE, SwiGLU, RMSNorm and DSA where supported and numerically validated.
2. Remove padding to avoid unnecessary computation over variable-length responses.

## 5. Evaluation

After training, evaluate the migrated model under a fixed protocol. This upstream example uses GLM-5 and AISBenchmark with vLLM/SGLang.

The reference evaluates AIME2025 mathematical reasoning and GPQA science knowledge to assess target improvement and retention.

### 5.1 Install AISBench

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
```

### 5.2 Evaluation data

```shell
# Run from the tool root on Linux.
cd path/to/benchmark/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip
unzip aime2025.zip
rm aime2025.zip
```

### 5.3 AISBench service configuration

Edit benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py. Keep max_out_len consistent with the intended training/evaluation budget.

```shell
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/GLM-5",
        model="GLM-5",
	    stream=True,
        request_rate = 0,
	    use_timestamp=False,
        max_seq_len=2048,
        retry = 2,
	    api_key="",
        host_ip = "localhost",
        host_port = 12890,
        max_out_len = 8192,
        batch_size=48,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1234,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

### 5.4 Multi-node inference service

Follow [GLM5 deployment](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md#multi-node-deployment) for two A3 nodes. Match host_port and set max_model_len to the intended prompt-plus-response budget.

### 5.5 Run evaluation

Invoke the deployed service with the matching model configuration:

```
ais_bench --models vllm_api_stream_chat --datasets aime2025_gen_0_shot_chat_prompt
```

The upstream example reports improvements on both AIME2025 and GPQA at the checkpoints below. These measurements do not establish retention outside the evaluated tasks.

| Dataset | GLM5-base | 10step | 15step | 40step | 50step |
| ---------- | --------- | ------ | ------ | ------ | ------ |
| aime2025   | 47.5      | 49.17  | 49.17  | 48.33  | 52.5   |
| gpqa       | 64.65     | 68.81  | 68.43  | 69.07  | 71.21  |

## 6. Migration workflow

The workflow covers environment setup, individual components, numerical alignment, performance and evaluation.

Verify dependencies, weight precision and data first; integrate engines and conversion next. Localize numerical discrepancies before tuning. For MoE, validate routing replay. Measure optimized execution with fixed protocols and evaluate target performance and retention separately.

Successful migration requires evidence from the actual model, hardware and workload at each stage.
