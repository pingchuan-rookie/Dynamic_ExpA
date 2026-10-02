Ascend Quickstart
=================

**Last updated:** 2026/07/14.

Key updates
-----------

- 2026/06/30: Added four common training/rollout backend combinations to help users select a quick-start script.
- 2026/05/13: Split quick-start and installation guidance into separate documents.
- 2025/12/11: Existing verl scenarios can detect NPU devices automatically. GPU scripts generally no longer need an explicit ``trainer.device=npu`` override on Ascend. New features can still select ``trainer.device`` explicitly while automatic detection is being added.


Contents
--------

- `Hardware support <#hardware-support>`_
- `Qwen3-0.6B GSM8K GRPO Quick Start <#qwen3-06b-gsm8k-grpo-quick-start>`_
   - `Prepare model weights <#prepare-model-weights>`_
   - `Prepare data <#prepare-data>`_
   - `Run the examples <#run-the-examples>`_

- `Enable the SGLang backend <#enable-the-sglang-backend>`_
   - `Convert a vLLM script to SGLang <#convert-a-vllm-script-to-sglang>`_

Hardware support
----------------

- Atlas 200T A2 Box16
- Atlas 900 A2 PODc
- Atlas 800T A3



Qwen3-0.6B GSM8K GRPO Quick Start
---------------------------------

This upstream guide describes a minimal GRPO training-stack check on Ascend NPUs using GSM8K and Qwen3-0.6B.

It covers four common training/rollout backend combinations for selecting a quick-start script.

Install the verl Ascend environment before running these scripts.
See `installation guidance <./install_guidance.rst>`_.

All four scripts default to ``Qwen/Qwen3-0.6B`` and GSM8K for basic upstream pipeline verification.

They check whether:

- the verl entrypoint is available;
- data can be read;
- actor, rollout and reference workers can initialize;
- vLLM-Ascend/SGLang rollout can generate responses;
- the training pipeline can complete its first step.

Prepare model weights
~~~~~~~~~~~~~~~~~~~~~

Download model weights from Hugging Face.

The scripts default to ``~/models/Qwen/Qwen3-0.6B``.

Place weights there or set ``MODEL_PATH`` in the script to the local model directory.


Prepare data
~~~~~~~~~~~~

.. code-block:: bash

   python3 examples/data_preprocess/gsm8k.py --local_dataset_path /download/path/hf_data/gsm8k/

Download the original GSM8K dataset from Hugging Face.

Generated files:

.. code-block:: text

   ~/data/gsm8k/train.parquet
   ~/data/gsm8k/test.parquet

Run the examples
~~~~~~~~~~~~~~~~

The upstream scripts are in ``tests/special_npu/quick_start/``. Developer test scripts are not distributed with this runtime-only project; use the test suite matching the pinned upstream revision for this guide.

First enter the upstream verl checkout: ``cd /your/path/verl``.

Activate CANN, adjusting the following command if CANN is installed at a custom path.

.. code-block:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh

Quick Start provides four common training/rollout backend combinations. Select the script for the required pair.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 60

   * - Combination
     - Training backend
     - Rollout backend
     - Run the examples
   * - vLLM + FSDP2
     - FSDP2
     - vLLM-Ascend
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_vllm_ascend.sh
   * - vLLM + Megatron
     - Megatron
     - vLLM-Ascend
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_megatron_vllm_ascend.sh
   * - SGLang + FSDP2
     - FSDP2
     - SGLang
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_sglang_ascend.sh
   * - SGLang + Megatron
     - Megatron
     - SGLang
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_megatron_sglang_ascend.sh

See `training parameters and metrics <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dev_guide/model_dev/parameter_and_metrics.md>`_ for script parameters.

See `multi-node launch practices <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/model_support/examples/multi-machine_task_startup_practice.rst>`_ for multi-node runs.

Enable the SGLang backend
-------------------------

verl parses common inference parameters when constructing ``ServerArgs``; see `async_sglang_server.py <../../../verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_.

Other `SGLang parameters <https://github.com/sgl-project/sglang/blob/v0.5.10/docs/advanced_features/server_arguments.md>`_ can be supplied through ``engine_kwargs``.

Convert a vLLM script to SGLang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert a vLLM inference script to SGLang, add or update the following parameters.

.. code-block:: bash

   # Required
   actor_rollout_ref.rollout.name=sglang \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \

   # Optional
   # Enable inference EP; see:
   # https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README_CN.md
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" \
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="deepep" \

   # Required for MoE models with multiple DP replicas
   +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False \

   # chunked_prefill is disabled by default
   +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1

