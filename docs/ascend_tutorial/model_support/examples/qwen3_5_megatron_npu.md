# Qwen3.5 with Megatron on NPUs

Last updated: 09/07/2026.

This guide runs the Qwen3.5-35B-A3B and Qwen3.5-122B-A10B GRPO examples on Ascend NPUs with verl, Megatron and vLLM.

## Version requirements

| software | version                                                       |
| --- |---------------------------------------------------------------|
| Docker image | `quay.io/ascend/verl:v0.8.0-cann9.0.0-torch2.9.0post2-a3-ubuntu22.04-py3.11-vllm` |
| verl | 0.8.0                                                         |
| Python | 3.11                                                          |
| CANN | 9.0.0                                                         |
| Megatron-LM | 0.16.0                                                        |
| MindSpeed | 0.16.0                                                        |
| Megatron-Bridge | `de93536e`                                                    |

Use the image listed in the table above:

```bash
docker pull quay.io/ascend/verl:v0.8.0-cann9.0.0-torch2.9.0post2-a3-ubuntu22.04-py3.11-vllm
```

## Models and scripts

| model             | HF model | script |
|-------------------| --- | --- |
| Qwen3.5-35B-A3B   | `Qwen/Qwen3.5-35B-A3B` | `examples/grpo_trainer/run_qwen3_5_35b_megatron.sh` |
| Qwen3.5-122B-A10B | `Qwen/Qwen3.5-122B-A10B` | `examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh` |
| Qwen3.5-397B-A17B | `Qwen/Qwen3.5-397B-A17B` | `examples/grpo_trainer/run_qwen3_5_397b_megatron.sh` |

## Hardware and parallelism

The examples default to the following NPU configuration; environment variables with the same names can override it:

| model | nnodes | devices per node | TP | PP | CP | EP | ETP | GEN_DP | GEN_TP | GEN_EP |
| --- |--------| --- | --- | --- | --- |----| --- |---|----|----|
| Qwen3.5-35B-A3B | 1 | 16 | 2 | 2 | 1 | 8  | 1 | 1 | 8 | 1 |
| Qwen3.5-122B-A10B | 4 | 16 | 2 | 4 | 1 | 16 | 1 | 1 | 16 | 1 |
| Qwen3.5-397B-A17B | 16 | 16 | 2 | 4 | 1 | 64 | 1 | 16 | 16 | 256 |

## Prepare data and models

The scripts default to Geo3K and download it under `$HOME/data/geo3k`:

```bash
hf download tyzhu/geo3k --repo-type dataset --local-dir $HOME/data/geo3k
```

Model weights can be specified by a Hugging Face model ID or downloaded to a local path:

```bash
hf download Qwen/Qwen3.5-35B-A3B --local-dir /path/to/Qwen3.5-35B-A3B
hf download Qwen/Qwen3.5-122B-A10B --local-dir /path/to/Qwen3.5-122B-A10B
hf download Qwen/Qwen3.5-397B-A17B --local-dir /path/to/Qwen3.5-397B-A17B
```

## Start training

Start the Ray cluster before training. See [Multinode Training](../../../start/multinode.rst) for general guidance and [Ascend SGLang Best Practices](ascend_sglang_best_practices.rst) for Ascend multi-node scripts.

A minimal launch is shown below. Single-node runs need only the head-node command; multi-node runs also execute the worker command on other nodes. Use the same `MASTER_ADDR` everywhere and set `CURRENT_IP` to each node's own IP.

```bash
MASTER_ADDR=<head-node-ip>
CURRENT_IP=<current-node-ip>
NPUS_PER_NODE=16

# head node
ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

# worker nodes, only needed for multi-node jobs
ray start --address="$MASTER_ADDR:6766" --node-ip-address=$CURRENT_IP --resources='{"NPU": '$NPUS_PER_NODE'}'

ray status
```

After `ray status` reports the expected NPU count, run the training script on the head node. Qwen3.5-35B-A3B defaults to 16 NPUs; Qwen3.5-122B-A10B defaults to 64.

### Qwen3.5-35B-A3B

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-35B-A3B

bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
```

To override data paths:

```bash
DEVICE=npu \
HF_MODEL_PATH=/path/to/Qwen3.5-35B-A3B \
train_path=/path/to/train.parquet \
test_path=/path/to/test.parquet \
bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
```

### Qwen3.5-122B-A10B

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-122B-A10B

bash examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh
```

To override data, output paths or parallelism:

```bash
DEVICE=npu \
HF_MODEL_PATH=/path/to/Qwen3.5-122B-A10B \
train_files=/path/to/train.parquet \
test_files=/path/to/test.parquet \
save_path=/path/to/checkpoints \
NDEVICES_PER_NODE=16 \
nnodes=4 \
bash examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh
```

### Qwen3.5-397B-A17B

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-397B-A17B

bash examples/grpo_trainer/run_qwen3_5_397b_megatron.sh
```

To override data, output paths or parallelism:

```bash
DEVICE=npu \
HF_MODEL_PATH=/path/to/Qwen3.5-397B-A17B \
train_files=/path/to/train.parquet \
test_files=/path/to/test.parquet \
save_path=/path/to/checkpoints \
NDEVICES_PER_NODE=16 \
nnodes=16 \
bash examples/grpo_trainer/run_qwen3_5_397b_megatron.sh
```

## Notes

- Scripts detect NPUs through `torch_npu`; set `DEVICE=npu` to select the device explicitly.
- Qwen3.5 Gated Delta Net currently does not use packed sequences, so keep `use_remove_padding=False` and `use_dynamic_bsz=False`.
- The NPU branch sets Ascend options including `vanilla_mbridge=False`, `use_flash_attn=True` and `moe_token_dispatcher_type=alltoall`.
