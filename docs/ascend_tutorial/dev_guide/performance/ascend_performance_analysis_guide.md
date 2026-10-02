# Ascend Performance Analysis Guide

Last updated: 02/24/2026.

## Background

This guide describes profiling and performance analysis for verl reinforcement learning on Ascend NPUs using MindStudio.

### RL computation stages

1. **Rollout:** the actor generates responses from prompts.
2. **Reference log-probability:** the reference model scores prompt/response pairs for KL computation.
3. **Actor log-probability:** the actor scores the same pairs for importance sampling.
4. **Reward:** the reward model or task grader produces rewards.
5. **Update:** rewards and log-probabilities determine the objective and actor gradients.

![rl_data_stream](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/rl_data_stream.png)

## Enable profiling

### Configuration

See [Ascend profiling](./ascend_profiling_en.rst).

## Analysis workflow

### Overall performance

#### 1. Long stages and idle resources

- Load profiling data in MindStudio Insight and inspect the RL timeline for long stages and NPU gaps.
- Compare time spent in each stage.
- Example:

![Bubble_analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Bubble_analysis.png)

#### 2. Load balance

- Inspect MSTX events across rollout DP ranks.
- Identify uneven work distribution.
- Example:

![Load_Balancing_Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Load_Balancing_Analysis.gif)

#### 3. Cluster overview

- Use MSTT rl_analysis to produce a cluster timeline overview.
- Compare stage durations and cluster bottlenecks.
- [rl_analysis guide](https://gitcode.com/Ascend/mstt/raw/pre-research/profiler/msprof_analyze/docs/features/rl_analysis.md)
- Example:

![Cluster%20Performance%20Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Cluster%20Performance%20Analysis.png)

### Detailed analysis

#### Execution performance

- Load traces in the Windows or Linux version of MindStudio Insight.
- Inspect scheduling, operators, compute utilization and collectives, including task decomposition and overlap in the interactive timeline.
- Example:

![performance%20analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/performance%20analysis.png)

#### Memory

##### Allocation changes and call stacks

- Enable call-stack and memory capture.
- Trace framework/CANN allocations and releases back to Python.
- Example:

![in-memory%20analytics](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/in-memory%20analytics.gif)

##### Detailed memory analysis with msleaks

- Follow the [msleaks guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/msleaks/atlas_msleaks_0001.html).
- Inspect allocation trends and memory blocks with matching call stacks.
- Example:

![msleaks](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/msleaks.gif)

## Analysis examples

Enable profiling level1 to capture the operator details needed below.

### 1. Host bottlenecks

Host-bound execution leaves the NPU waiting for CPU dispatch. Inspect Host2Device synchronization: a set signal already available before the device waits can instead indicate device-bound work.

![host_bound_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_1.png)

For confirmed host bottlenecks, sum CPU dispatch costs across calls rather than examining only first-call initialization. In this example, GmmSwigluQuant takes 1 ms initially and 200 microseconds thereafter.

![host_bound_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_2.png)

Prioritize operators whose cumulative host cost exceeds device execution cost.

### 2. Computation structure

Profiles can reveal inefficient operator composition.

Attention and FFN matrix multiplications often dominate LLM compute; the source example gives more than 70% of prefill and 50% of decode as rough expectations. Unexpected operators or excessive concatenation/conversion warrant inspection rather than assuming these percentages are universal.

slice/split/concat and transpose/cast can arise from incompatible adjacent layouts. Producer-side output handling may avoid launches and redundant memory traffic, subject to operator semantics.

For a matmul output [m,n0+n1] followed by two slices, one split may reduce slicing cost and release the shared buffer sooner. Alternatively split weights [k,n0+n1] into [k,n0] and [k,n1] and use two matmuls, if partitioning and combined runtime remain acceptable.

![network_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_1.png)

For RmsNorm(fp16) → cast(fp32) → matmul(fp32), fusing the cast could avoid the normalization's fp32-to-fp16 conversion and later promotion. However, mixed input/output dtypes alter the operator contract; numerical and performance benefits alone do not justify an incompatible interface.

![network_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_2.png)

### 3. Initial operator diagnosis

Inspect ./ASCEND_PROFILER_OUTPUT/operator_details.csv.

Pipeline ratios divide mean busy time across cores by the slowest core's kernel duration. Pipelines can overlap despite dependencies and shared bandwidth. As a heuristic, operators lasting at least 50 microseconds may be expected to exceed 80% utilization on a limiting pipeline; validate this against the workload.

The example shows Flash Attention at 88.1% vector utilization and matmul at 89.8% MAC utilization.

![Operator%20performance](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Operator%20performance.png)

### 4. Shapes suited to hardware

Concurrency, weight layout and partitioning can improve transfer efficiency and load balance without changing model hyperparameters. Measure the effects before adopting changes.

#### 4.1 Transfer efficiency

MTE2 efficiency depends strongly on shape. The source recommends either:

1. NZ matrix format, or
2. A trailing dimension aligned to 512 bytes but not an integer multiple of 16 KiB.

Inference, particularly decode, often uses NZ weights; training commonly uses aligned layouts. Where NZ is unavailable:

1. Consider transposing when the leading axis is well aligned but the trailing axis is not.
2. Adjust TP partitioning to avoid inefficient trailing dimensions.

#### 4.2 Load balance

Small decode shapes can leave cores idle or distribute work unevenly.

Check the target device's core count. In the illustrated hardware, 20/24 cube groups each pair one cube with two vector cores; pure-vector operators can schedule 40/48 vector cores independently.

Vector operators may partition by batch for trailing-axis reductions or flatten elementwise work. A batch of 48 on 40 vector cores needs a partially occupied second wave. Batches such as 64/80 may improve utilization; benchmark rather than assuming unchanged latency. A 48-core device has a different balance.

Cube operators commonly tile M/N, with K as the accumulation axis; K partitioning can affect determinism. For illustrative baseM=128/baseN=256 decode kernels, M<=128 often uses one tile and weight traffic dominates. Crossing 128 may add another wave, so measure at tile boundaries.
DeepSeek-R1 MLA preprocessing multiplies [batch,7168] by [7168,1536] and [7168,576]. Small batches may underutilize cores even with baseN=128. Concatenating weights to [7168,2112] can improve utilization if output semantics and total runtime remain equivalent.

Attention commonly partitions q_seqlen, batch and KV heads. Decode query grouping by MTP/GQA may still fit within one query tile, leaving parallelism mainly batch_size × kv_headnum.

Use shapes and operator scheduling to form hypotheses about partitioning and batch sizes, then verify numerical behavior and performance.
