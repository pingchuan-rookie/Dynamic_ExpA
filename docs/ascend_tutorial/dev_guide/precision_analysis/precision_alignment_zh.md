# Precision Alignment

Numerical alignment helps make verl reinforcement learning reproducible and debuggable.

This guide compares NPU and GPU execution.

Last updated: 05/09/2026.

## 1. Environment and weight alignment

### 1.1 Dependency versions

Match verl and transformers versions exactly; differences can affect numerical results.

Match other key dependencies, including torch, Megatron and vLLM, as closely as supported.

### 1.2 Model weights

Verify identical weights and config.json.


## 2. Input alignment

Add these settings to the verl launch script:

```bash
data.shuffle=False
data.validation_shuffle=False
```


## 3. Configuration alignment

Compare complete NPU/GPU configuration:
1. Compare explicit script settings.
2. Capture resolved runtime configuration, including defaults, and verify all critical parameters.


## 4. Determinism

### 4.1 Seeds

Install msprobe:

```bash
pip install mindstudio-probe
```

Initialize determinism at the start of worker files:

```python
from msprobe.pytorch import seed_all
seed_all(mode=True)
```

### 4.2 Communication variables

For multi-device communication:

- HCCL, the default:

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export HCCL_DETERMINISTIC="true"
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

- LCCL, enabled with export HCCL_OP_EXPANSION_MODE="AIV":

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export LCCL_DETERMINISTIC=1
  -  export ATB_LLM_LCOC_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

For one device without communication:

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0



## 5. Training precision

### 5.1 Capture and replay

Capture stage inputs/outputs to localize numerical differences. A common approach saves rollout data and replays it for training comparison.

**Step 1: Generate GPU reference data**

Run once on GPU with:

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```
This saves each step's generation output as JSONL.

**Step 2: Replay on NPU**

Reuse the saved sequences for an end-to-end NPU run:

```bash
skip.rollout.enable=True \
skip.rollout.dump_dir=/path/to/rollout_dump \
```

**Step 3: Compare metrics**

With identical rollout inputs, configuration and deterministic settings, compare rewards, pg_loss and grad_norm.


## 6. Inference precision

### 6.1 resharding

vLLM runs initialization/profiling passes to estimate memory. actor_rollout_ref.rollout.load_format selects dummy or safetensors weight loading. Check whether the relevant initialization uses random or real weights.

Garbled generation after dummy initialization can indicate sharding/synchronization problems. Loading safetensors successfully does not by itself clear that diagnosis; compare forward passes.


### 6.2 Compare inference outputs

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```

Inspect saved per-step JSONL for garbled generation to localize inference failures.


If reproduction is costly, try fewer batches or shorter sequences and verify that the same failure remains. This diagnostic reduction is not capacity or formal-protocol validation.


## 7. Tensor dumps

After locating the affected stage, use [precision debugging tools](./precision_debugger_zh.md) and msprobe for detailed dumps.

For unexpected output or NaN/Inf, capture intermediate features, weights, activations and layer inputs/outputs, along with prompts, dtypes and hardware configuration to trace the numerical error.




