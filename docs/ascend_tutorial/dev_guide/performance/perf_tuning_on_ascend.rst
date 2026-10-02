Performance Tuning Guide on Ascend
==================================

Last updated:  01/29/2026.

Author:  `Xiaobo Hu <https://github.com/tardis-key>`_, `Haozhe Li <https://github.com/ZLiao097>`_

General `performance tuning <https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>`_ also applies to Ascend. This guide covers fused operators and NPU-specific settings.

Fused operators
---------------

Common operators
**********************************

Fusion combines mathematically equivalent operations to reduce redundant computation and launch overhead. npu_patch.py integrates supported replacements for Qwen2/Qwen3.

See `npu_patch.py <https://github.com/verl-project/verl/blob/main/verl/models/transformers/npu_patch.py>`_ for the complete set.

Matrix Computation-Communication operator fusion (MC2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CANN MC2 operators fuse computation and communication through internal partitioning and pipelining.

In vllm-ascend, set:

.. code-block:: sh

    export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1

This enables torch_npu.npu_mm_all_reduce_base in RowParallelLinear, fusing matmul and allreduce.

`RotaryMul&RotaryMulGrad <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0030.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interface: torch_npu.npu_rotary_mul(x, r1, r2).

Arguments:

- x: q/k tensor, four-dimensional, commonly [B,N,S,D], [B,S,N,D] or [S,B,N,D].

- r1: cosine tensor, commonly [1,1,S,D], [1,S,1,D] or [S,1,1,D].

- r2: sine tensor with the corresponding four-dimensional shape.

`RmsNorm&RmsNormGrad <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0031.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interface: torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor).
Arguments:

- self: tensor with 1-8 dimensions.

- gamma: weight tensor matching trailing self dimensions.

- epsilon: floating-point stabilization constant.

Returns:

- Normalized output y.

- Intermediate rstd for backward computation.

`Swiglu <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0035.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interface: torch_npu.npu_swiglu(Tensor self, int dim=-1) -> Tensor.

Arguments:

- self: tensor with 1-8 dimensions.

- dim: integer, default -1.

Returns:

- Output tensor y.

`GroupMatMul <https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_grouped_matmul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Signature:

.. code:: python

    npu_grouped_matmul(
        x,
        weight,
        *,
        bias=None,
        scale=None,
        offset=None,
        antiquant_scale=None,
        antiquant_offset=None,
        per_token_scale=None,
        group_list=None,
        activation_input=None,
        activation_quant_scale=None,
        activation_quant_offset=None,
        split_item=0, group_type=None,
        group_list_type=0,
        act_type=0,
        output_dtype=None,
        tuning_config=None
    ) -> List[Tensor]

See the linked operator documentation for details.

FSDP integration
**********************************

verl/models/transformers/npu_patch.py applies available fused-operator patches by default.

Megatron integration
**********************************

MindSpeed provides Megatron fusion, enabled through configuration:

1. **Flash Attention, required**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
       ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True

2. **RotaryMul**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True

3. **RMSNorm**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True

4. **GroupMatMul**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True

5. **Swiglu**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True

6. **Permute/Unpermute**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.fused_permute_unpermute=True

7. **MC2**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_mc2=True

General Ascend settings
-----------------------

`Operator dispatch <https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md>`_
************************************************************************************************************************************************************************************************************

TASK_QUEUE_ENABLE sets dispatch-queue optimization, default level 1, reducing host launch overhead and related idle time.

.. image :: https://github.com/verl-project/verl-data/blob/main/images/ascend/perf_tuning_task_queue.png
    :width: 500px

Level 0: disable dispatch pipelining.

Level 1: move tasks, mainly aclnn calls, to a second pipeline; a queue overlaps work between pipelines.

Level 2: rebalance the pipelines further by moving workspace tasks to the second stage. It applies to binary execution and can improve overlap.

`Communication scheduling <https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0096.html>`_
************************************************************************************************************************************************************************************************************
HCCL_OP_EXPANSION_MODE selects where communication algorithms are expanded:

- **AI_CPU:** device AI CPU; hardware selects the scheduler.

- **AIV:** device Vector Core for expansion and execution.

- **HOST:** host CPU; hardware selects the device scheduler.

- **HOST_TS:** host CPU submits work to the device Task Scheduler.

Inference tuning
----------------

Chunked Prefill in V1
***************************

Current vLLM defaults to V1. Enable chunked prefill with:

.. code-block:: sh

    actor_rollout_ref.rollout.enable_chunked_prefill=True

See the `vLLM performance guide <https://docs.vllm.ai/en/v0.4.2/models/performance.html>`_ for the principle.

Graph Mode
***************************

Enable ACL Graph with:

.. code-block:: sh

    actor_rollout_ref.rollout.enforce_eager=False

.. note::
    ACL Graph and task-queue level 2 cannot be enabled together in this configuration.

Training tuning
---------------

FSDP
**********************************

.. csv-table::
   :header: "FSDP", "Description"
   :widths: 30, 60

   "/","Optimizer-only sharding (ZeRO-1; unsupported by FSDP)"
   SHARD_GRAD_OP,"Gradient and optimizer sharding (ZeRO-2)"
   "HYBRID_SHARD","Parameter, gradient and optimizer sharding within groups (ZeRO-3)"
   "2D device_mesh+HYBRID_SHARD","HSDP: for mesh [2,8], two FSDP groups of eight ranks synchronize gradients using DDP allreduce"
   "2D device_mesh+HYBRID_SHARD_ZERO2","HSDP with ZeRO-2-style sharding"
   NO_SHARD,DDP

FSDP does not support ZeRO-1. verl derives the mesh from device count and actor_rollout_ref.actor.fsdp_config.fsdp_size, defaulting to full sharding. For smaller models, evaluate reduced resharding overhead through the backend's reshard_after_forward setting; verify its FSDP/FSDP2 semantics before changing it.

Megatron
**********************************

Megatron offers more parallelism controls for large models.

Use TP when weights exceed DP capacity, then PP if needed. CP/SP can reduce long-sequence activation pressure. For MoE, EP distributes experts and expert tensor parallelism controls whether expert weights are further split.

TP, PP, EP and ETP follow Megatron configuration. NPU CP/SP use MindSpeed:

- SP partitions sequence dimensions in conjunction with TP:
  ::

      actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True

- CP partitions activation context across devices; both parameters below are required:
  ::

      actor_rollout_ref.actor.megatron.context_parallel_size
      actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size

Megatron-distributed optimizer
**********************************

For large models, shard optimizer state across the DP group. Enable the NPU Megatron distributed optimizer with:

::

    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
