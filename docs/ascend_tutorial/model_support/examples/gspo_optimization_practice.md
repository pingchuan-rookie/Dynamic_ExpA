# NPU Qwen3-32B GSPO Optimization Practice

Last updated: 07/03/2026.

This upstream recipe pins verl main commit 9d05508f5e3bd8ecb70cf94ab10dc087b57a716d. Later refactors may invalidate patches; release/v0.8.0 is the documented stable alternative. It is not a Dynamic-ExpA training protocol.

Example: [run_qwen3_32b_fsdp.sh](../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh).

## Algorithm configuration

GSPO moves importance weighting from tokens to sequences to reduce the high-variance instability observed with GRPO in its motivating experiments.

Required verl settings:

```bash
# GSPO configuration.
algorithm.adv_estimator=grpo \
algorithm.use_kl_in_reward=False \

actor_rollout_ref.actor.policy_loss.loss_mode=gspo \

actor_rollout_ref.actor.clip_ratio_low=0.0003 \
actor_rollout_ref.actor.clip_ratio_high=0.0004 \

actor_rollout_ref.actor.use_kl_loss=False \
actor_rollout_ref.actor.kl_loss_coef=0.0 \

actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \

actor_rollout_ref.rollout.n=16
```

Use verl.trainer.main_ppo. See the [complete example](../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh).

## Environment

The recipe supports Atlas 800T A3 and Atlas 900 A3 SuperPoD and uses four Atlas 800T A3 nodes.

### Installation

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


```bash
cd verl
git checkout main
# Select the matching recipe revision.
git submodule update --init --recursive recipe
```

### Weights

Download [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B).

### Data

```bash
# Download math-17k.
git clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

# Download AIME_2024 evaluation data.
git clone https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
```

### jemalloc

Install and enable jemalloc to improve Ray memory reclamation.

#### Ubuntu

Install from system repositories on Ubuntu >=20.04:

```shell
sudo apt install libjemalloc2
```

Locate libjemalloc.so.2 under /usr before setting LD_PRELOAD:

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

Alternatively, build a stable [jemalloc release](https://github.com/jemalloc/jemalloc/releases/).

```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```

Set the preload variable before starting jobs:

```shell
# Use the actual installed jemalloc path.
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
```

### Multi-node startup

Use this launcher for the example:

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=1
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=0
export CPU_AFFINITY_CONF=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2

# Select the training script.
DEFAULT_SH="./run_*.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir logs

NNODES=4
NPUS_PER_NODE=16
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

DEFAULT_SH selects [run_qwen3_32b_fsdp.sh](../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh).

NNODES=4 and NPUS_PER_NODE=16 for this example.

MASTER_ADDR must match on all nodes.

Select SOCKET_IFNAME, HCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME using:

```bash
ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1
```

## Performance tuning

Consider training, inference, scheduling and shared settings separately.

### Training

#### Dynamic batches

```bash
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
```

Increasing token limits may improve throughput but can cause OOM.

The source experiment found gains from actor_ppo_max_token_len but no clear benefit from changing infer_ppo_max_token_len.

Parameter meanings:

Both limit tokens per device under dynamic batching.

- actor_ppo_max_token_len: actor forward/backward tokens during PPO updates.
- infer_ppo_max_token_len: tokens for reference/rollout log-probability computation.

### Inference

#### ACLgraph+FULL_DECODE_ONLY

The upstream example reports approximately 15%-20% gains from dispatch optimization; this is workload-specific.

Enable ACL Graph with:

```bash
# Graph mode requires TASK_QUEUE_ENABLE=1.
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[8,16,32,64,128]' \
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode='FULL_DECODE_ONLY'
```

Successful FULL_DECODE_ONLY initialization prints:

![FULL_DECODE_ONLY result](https://github.com/wucong25/verl-data/blob/main/ascend_acl_graph.png)

**Capture sizes**

cudagraph_capture_sizes describes vLLM capture batch sizes in scheduled tokens, not the training DP batch.

The default generation rule is shown below:

![cudagraph_capture_sizes](https://github.com/wucong25/verl-data/blob/main/ascend_set_cudagraph_sizes.png)

##### Attention backend

Set export VLLM_ATTENTION_BACKEND=XFORMERS.

![VLLM_ATTENTION_BACKEND](https://github.com/wucong25/verl-data/blob/main/ascend_vllm_attn_backend.png)

Older vllm-ascend versions may not support every backend.

##### vLLM V1

Set export VLLM_USE_V1=1.

Measure the effect on the selected version and workload.

### Scheduling

#### AIV

Set export HCCL_OP_EXPANSION_MODE="AIV".

Available expansion locations:

- AI_CPU: device AI CPU.
- AIV: device Vector Core.
- HOST: host CPU with a hardware-selected device scheduler.
- HOST_TS: host CPU submits to the device Task Scheduler.

Two mechanisms are illustrated below.

##### HOST expansion

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_task_queue1.png" alt="image-20260113194257095" style="zoom:50%;" />

- Host software expands communication into tasks.
- Each task enters the device runtime queue.
- STARS consumes tasks in order.
- SDMA/RDMA engines execute the corresponding work.
  In the illustrated host-bound case, each submission costs 2-5 microseconds and a communication operator may have hundreds of tasks without device-side caching.

##### AI CPU expansion

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_task_queue3.png" alt="image-20260113194333218" style="zoom:50%;" />

- The host submits a communication kernel rather than individual tasks.
- STARS schedules it on the AI CPU.
- An AI CPU thread expands tasks into the runtime queue for STARS.
- Host/AI CPU interactions decrease from hundreds to one.
- Some task submission is combined on the AI CPU.

#### TASK_QUEUE_ENABLE

Set export TASK_QUEUE_ENABLE=2.

Use level 1 with graph mode and level 2 with eager mode.

Diagram:

![ascend task queue](https://github.com/wucong25/verl-data/blob/main/ascend_task_queue2.png)

##### CPU affinity

Set export CPU_AFFINITY_CONF=1.

See https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0059.html.

### Shared settings

The following settings may affect both training and inference. The source does not provide sufficiently isolated ablations to attribute each contribution; monitor them separately.

#### jemalloc

After installation, set export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 using the actual path.

[Installation guide](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/install_guide.md#%E9%AB%98%E6%80%A7%E8%83%BD%E5%86%85%E5%AD%98%E5%BA%93-jemalloc-%E5%AE%89%E8%A3%85).

#### Multi-stream memory reuse

Reuse memory across streams where supported.

Set export MULTI_STREAM_MEMORY_REUSE=1.

See https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0040.html.

#### VLLM_ASCEND_ENABLE_FLASHCOMM

Set export VLLM_ASCEND_ENABLE_FLASHCOMM=1.

Enable Ascend FLASHCOMM communication optimization.

See https://vllm-ascend.readthedocs.io/zh-cn/latest/user_guide/release_notes.html.

#### VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE

Set export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1.

Enable Ascend dense-inference optimization.

See https://vllm-ascend.readthedocs.io/zh-cn/latest/user_guide/release_notes.html.

#### VLLM_ASCEND_ENABLE_PREFETCH_MLP

Set export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1.

Enable MLP weight prefetching.

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_prefetch.png" alt="image-20251124173132677" style="zoom:50%;" />

### verl memory settings

These options trade memory against recomputation, transfer or cache overhead; measure throughput and capacity together.

```bash
# Activation checkpointing.
# Recompute activations during backward instead of retaining them.
actor_rollout_ref.model.enable_gradient_checkpointing=True \

# Parameter offload.
# Transfer parameters between CPU and accelerator memory.
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.ref.fsdp_config.param_offload=True \

# Optimizer offload.
# Store optimizer state, such as Adam moments, on CPU.
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \

# Release inference cache.
# Free rollout resources while training uses the device.
actor_rollout_ref.rollout.free_cache_engine=True \

# Entropy memory controls.
# Recompute entropy intermediates.
# Process logits in chunks instead of one full batch/sequence/vocabulary tensor.
actor_rollout_ref.actor.entropy_checkpointing=True \
actor_rollout_ref.ref.entropy_checkpointing=True \
actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \

# Inference memory.
# gpu_memory_utilization is a device-memory fraction.
# Graph capture can accelerate inference but needs extra memory.
actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
actor_rollout_ref.rollout.enforce_eager=False \
```

## References

[Ascend environment variables](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/Envvariables/Envir_001.html).

[Ascend performance workflow](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0001.html).
