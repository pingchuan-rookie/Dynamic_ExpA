# DAPO multi model optimization practice

## DAPO overview

Last updated: 07/03/2026.

The [DAPO paper](https://arxiv.org/pdf/2503.14476) introduces these techniques. This is an upstream reference recipe, not a Dynamic-ExpA training protocol.

* **Clip-Higher:** a higher importance-ratio clipping bound encourages diversity and mitigates entropy collapse.
* **Dynamic sampling:** filter groups with accuracy 0 or 1 to retain a consistent number of groups with useful gradients.
* **Token-level policy-gradient loss:** aggregate tokens across long-CoT sequences.
* **Overlong reward shaping:** reduce reward noise from excessive response length.

Configure DAPO in verl as follows:

- **DAPO reward management**
  Select the DAPO reward manager for this recipe.

```bash
reward_model.reward_manager.name=dapo
```

- **Clip-Higher**
  clip_ratio_low/high specify the lower/upper epsilon values in the objective.

```bash
clip_ratio_low=0.2
clip_ratio_high=0.28
```

- **Dynamic sampling**
  filter_groups.enable=True removes groups with identical metric values, such as all-correct or all-incorrect accuracy groups.
  The trainer repeatedly samples gen_batch_size until enough eligible groups exist or max_num_gen_batches is reached.

```bash
data.gen_batch_size=${gen_prompt_bsz}
algorithm.filter_groups.enable=${enable_filter_groups}
algorithm.filter_groups.metric=${filter_groups_metric}
algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}
```

- **Token-level Loss**
  loss_agg_mode=token-mean averages policy-gradient loss over all tokens in the batch.

```bash
actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
# token-mean is the default.
```

- **Overlong penalties**
  overlong_buffer.enable=True penalizes responses past max_response_length - overlong_buffer.len. As excess grows from 0 to the buffer length, the penalty grows linearly from 0 to penalty_factor.

```bash
reward_model.overlong_buffer.enable=${enable_overlong_buffer}
reward_model.overlong_buffer.len=${overlong_buffer_len}
reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}
```

See the [DAPO recipe](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md) for implementation.

## Hardware


The source recipe supports Atlas 800T A3 and Atlas 900 A3 SuperPoD and uses one SuperPoD. See [Ascend quick start](../../get_start/quick_start.rst) for software.


## Environment

| software      | version                                                    |
| ------------- | ---------------------------------------------------------- |
| Python        | 3.11                                                       |
| CANN          | ==9.0.0.B160 (CANN900B160)                                 |
| torch         | ==2.9.0                                                    |
| torch_npu     | ==2.9.0                                                    |
| triton_ascend | ==3.2.1                                                    |
| verl          | main                                                       |
| vllm          | v0.18.0                                                    |
| vllm-ascend   | v0.18.0                                                    |
| transformers  | 5.3.0                                                      |

This recipe pins verl. Later main revisions may invalidate patches; the upstream release/v0.8.0 branch is the documented stable alternative.

```bash
cd verl
git checkout main
# Select the matching recipe revision.
git submodule update --init --recursive recipe
cd recipe
git checkout main
```

## Training

### Data

[Geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k), developed by UCLA and Zhejiang University, contains 3002 multimodal geometry samples with text questions and diagrams describing shapes and spatial relationships.

```shell
# Download and preprocess source data.
python ./examples/data_preprocess/geo3k.py --local_dir=./data/geo3k
```

### Weights

Download [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main).

### jemalloc

Install and enable jemalloc to improve Ray process memory reclamation.

#### Ubuntu

Install from system repositories on Ubuntu >=20.04:

```shell
sudo apt install libjemalloc2
```

Before launch, locate libjemalloc.so.2 under /usr and set LD_PRELOAD:

```shell
# Arm64
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
# x86_64
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

#### OpenEuler

Install from system repositories:

```shell
yum install jemalloc
```

If unavailable, build a stable [jemalloc release](https://github.com/jemalloc/jemalloc/releases/) from source.

```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```

### Environment variables

- Enable jemalloc to improve memory reclamation during long Ray runs.

```bash
# Set the actual installed jemalloc path.
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
```

- Set the optimized-model switch to 0 if a model is incompatible with the vllm-ascend replacement.

```bash
export USE_OPTIMIZED_MODEL=0
```

- Enable vLLM V1.

```bash
export VLLM_USE_V1=1
```

- Increase Ascend communication connection timeout for slow cluster startup.

```bash
export HCCL_CONNECT_TIMEOUT=5400
```

- Control Ascend NZ optimization in vLLM.

```bash
export VLLM_ASCEND_ENABLE_NZ=0
```

### Launch
```bash
# Model Weights Paths
MODEL_PATH=hf_weights/Qwen3-VL-30B-A3B-Instruct
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# File System Paths
TRAIN_FILE=$RAY_DATA_HOME/datasets/geo3k/train.parquet
TEST_FILE=$RAY_DATA_HOME/datasets/geo3k/test.parquet

# Save frequency; -1 disables checkpoint saves.
trainer.save_freq=-1
```

- Single-node Qwen3-VL-30B launch:

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=5400
export VLLM_ASCEND_ENABLE_NZ=0
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training,
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0
export CPU_AFFINITY_CONF=2
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_VERSION="0.18.0"

# Head-node IP.
MASTER_ADDR="IP FOR MASTER NODE"
# NPUs per node.
NPUS_PER_NODE=16
ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

bash recipe/dapo/run_dapo_qwen3_vl_30b_fsdp2_npu.sh
```
- For multi-node Qwen3-VL-30B, adjust NNODES/NPUS_PER_NODE and matching trainer.nnodes/trainer.n_gpus_per_node:

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=5400
export VLLM_ASCEND_ENABLE_NZ=0
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training,
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0
export CPU_AFFINITY_CONF=2
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_VERSION="0.18.0"

# Training script.
DEFAULT_SH="./run_*.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir logs

# Number of nodes.
NNODES=2
# NPUs per node.
NPUS_PER_NODE=8
# Head-node IP.
MASTER_ADDR="IP FOR MASTER NODE"
# Local communication interface.
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
# Resolve local IP.
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # Head node.
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # Wait for all nodes.
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash $DEFAULT_SH
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # Retry worker registration.
  while true; do
      # Join Ray.
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # Check connection.
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600
```
DEFAULT_SH selects [the Qwen3-VL-30B recipe](https://github.com/verl-project/verl-recipe/blob/main/dapo/run_dapo_qwen3_vl_30b_fsdp2_npu.sh).

NNODES/NPUS_PER_NODE select nodes and NPUs per node; the illustrated values are 2 and 8.

MASTER_ADDR is identical on all nodes.

Set SOCKET_IFNAME, HCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME to the communication interface:

```bash
ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1
```

## Tuning

- **Dynamic batches**
  Adjust micro-batches using the per-device ppo_max_token_len_per_gpu budget.

```bash
actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
```

- **Tokens per device**
  With use_dynamic_bsz=True, this limits tokens per micro-batch.

```bash
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
```

- **Micro-batch size**
  Check the selected backend's batching behavior; dynamic batching uses token limits while fixed batching uses micro-batch sample counts.

```bash
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
```

- **FSDP2**
  Shard parameters, gradients and optimizer state across devices.

```bash
# Enable FSDP2.
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.ref.strategy=fsdp2
critic.strategy=fsdp2

# FSDP2: reshard after forward to reduce memory.
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
# FSDP2 forward resharding.
actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
```

- **Expert parallelism**
  Select how many devices distribute expert computation.

```bash
# MoE actor expert parallelism.
actor_rollout_ref.rollout.expert_parallel_size=8
```


