NPU frequently asked questions
==============================

Last updated: 05/13/2026.

Common issues in verl NPU training and inference.

Environment configuration
-------------------------

Q1: NPU devices are not visible

**Symptom:** torch_npu.npu.is_available() returns False.

**Checks:**

.. code-block:: bash

   # Check visibility.
   echo $ASCEND_RT_VISIBLE_DEVICES

   # Select devices and disable automatic Ray selection.
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

   # Inspect drivers.
   npu-smi info

Debugging and diagnostics
-------------------------

Q1: Enable NPU profiling

Use verl's built-in profiler:

.. code-block:: shell

   actor_rollout_ref.actor.profiler.tool_config.npu.discrete=true \
   actor_rollout_ref.actor.profiler.tool_config.npu.contents=npu,cpu \
   actor_rollout_ref.actor.profiler.tool_config.npu.level=1 \
   actor_rollout_ref.actor.profiler.tool_config.npu.analysis=true

Q2: Diagnose training failures

**Steps:**

1. Check environment variables.
2. Verify device visibility.
3. Check CANN compatibility.
4. Read the specific log error.
5. Reproduce with a minimal example.

**Detailed logs:**

.. code-block:: bash

   # verl logs.
   export VERL_LOGGING_LEVEL=DEBUG

   # Ascend: 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR.
   export ASCEND_GLOBAL_LOG_LEVEL=0
   export ASCEND_SLOG_PRINT_TO_STDOUT=1

   # HCCL logs.
   export HCCL_DEBUG=INFO

Common errors
-------------

### Q1： "torch_npu detected, but NPU device is not available or visible"

**Cause:** missing drivers or invisible devices.

**Resolution:** check drivers and ASCEND_RT_VISIBLE_DEVICES.

### Q2： "KeyError: decoder.layers.0.self_attention.q_layernorm.weight"

**Cause:** an older MindSpeed version.

**Resolution:** use matching MindSpeed 2.3.0_core_r0.12.1 for this stack.

### Q3： "AssertionError: Weight ... is too large to fit in the bucket"

**Symptom:** weight synchronization fails with:

.. code-block:: text

   AssertionError: Weight model.embed_tokens.weight(torch.Size([151936, 4096]), torch.float32) is too large to fit in the bucket.
   Please increase rollout.update_weights_bucket_megabytes(2048 MB).

**Cause:** a weight tensor exceeds the default 2048 MiB transfer bucket; each tensor must fit in one bucket.

**Tensor size:**

Memory bytes = product of dimensions × bytes per element.

Element sizes:

- torch.float32: 4 bytes
- torch.float16 / torch.bfloat16: 2 bytes
- torch.int8: 1 byte

For model.embed_tokens.weight:

.. code-block:: text

   Shape: torch.Size([151936, 4096])
   Type: torch.float32 (4 bytes)
   Size = 151936 × 4096 × 4 = 2,489,319,424 bytes = 2374 MiB

   Default bucket = 2048 MiB < 2374 MiB, causing assertion failure.

**Resolution:** increase update_weights_bucket_megabytes above the largest weight tensor:

.. code-block:: bash

   actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096

**Choosing a value:**

1. Find the largest parameter nbytes and divide by 1024².

2. Round upward, for example to 4096 MiB for a 2374 MiB tensor.

3. Allow 1.2-1.5 times the tensor size for alignment/runtime headroom before rounding.

4. Avoid excessive allocation that could cause worker OOM.

**Example starting values:**

.. list-table::
   :header-rows: 1

   * - Model size
     - Typical largest shape
     - Suggested bucket
   * - 7B, such as Qwen2
     - [151936, 4096] float32
     - 4096 MB
   * - 14B
     - [152064, 5120] float32
     - 4096 MB
   * - 72B
     - [152064, 8192] float32
     - 8192 MB

Q4: Missing checkpoint metadata on non-shared storage

**Symptom:** multi-node verl/Megatron saves checkpoints but cannot reload common.pt, .metadata or metadata.json:

.. code-block:: text

   FileNotFoundError: common.pt
   FileNotFoundError: .metadata
   FileNotFoundError: metadata.json

**Cause:** incomplete non-shared checkpoint support:

- Weight shards are saved across nodes.
- Metadata may exist only on the saving node, usually rank 0.
- Every node needs metadata for restoration.

**Workaround:** copy metadata from the saving node to every other node:

.. code-block:: bash

   # Example checkpoint root on rank 0: /path/to/ckpt/.
   # Copy metadata to all other nodes.

   # Required files.
   /path/to/ckpt/common.pt
   /path/to/ckpt/.metadata
   /path/to/ckpt/metadata.json

   # Example transfer.
   scp /path/to/ckpt/common.pt node1:/path/to/ckpt/
   scp /path/to/ckpt/.metadata node1:/path/to/ckpt/
   scp /path/to/ckpt/metadata.json node1:/path/to/ckpt/

   # Repeat for each node.

**Notes:**

- Repeat after each save because metadata can change.
- For frequent saves, automate copying after successful checkpoint publication.
- Framework-level non-shared checkpoint support should synchronize metadata automatically.

References
----------

- `NPU performance tuning <../dev_guide/performance/perf_tuning_on_ascend.rst>`_
- `NPU quick start <../get_start/quick_start.rst>`_
- `NPU CI guide <../contribution_guide/ascend_ci_guide_zh.rst>`_
- Ascend documentation: https://www.hiascend.com/document
- CANN documentation: https://www.hiascend.com/software/cann

Further help
------------

If the issue persists:

1. Read complete error logs.
2. Search related GitHub issues.
3. Include errors and environment configuration.
4. Provide a minimal reproduction.
