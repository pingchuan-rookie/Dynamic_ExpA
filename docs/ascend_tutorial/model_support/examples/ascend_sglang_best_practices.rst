Ascend SGLang Best Practice
===========================

Last updated: 06/02/2026.

.. _Qwen3-30B: https://github.com/verl-project/verl/blob/main/examples/ascend_extras/grpo_trainer/run_qwen3_30b_a3b_megatron.sh

.. _doclink: https://github.com/verl-project/verl/blob/c98cb8cc/docs/ascend_tutorial/examples/ascend_sglang_best_practices.rst

Introduction
------------

SGLang is a high-performance open-source inference engine supported by Ascend in verl.
This upstream reference covers:

1. Environment setup
2. Model training and evaluation
3. Profiling

Example models and hardware:

- Use the documentation and scripts at commit c98cb8cc (`doclink`_) to preserve this historical example's paths.

.. list-table::
   :header-rows: 1

   * - Model
     - NPU model
     - Nodes
     - Training/inference
   * - `Qwen3-30B`_
     - Atlas 800T A3
     - 1
     - SGLang + Megatron

Environment setup
-----------------
The `installation guide <../../get_start/install_guidance.rst>`_ supports Dockerfile or custom Conda setup.

This example also pins verl to avoid unrelated changes:

.. code-block:: bash

    cd verl
    git checkout 772c224

Training and evaluation
-----------------------
1. Prepare model and data
^^^^^^^^^^^^^^^^^^^^^^^^^
`Qwen3-30B`_
^^^^^^^^^^^^
**Download weights**

--local-dir selects the destination.

.. code-block:: bash

  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download --resume-download Qwen/Qwen3-30B-A3B --local-dir /path/to/local_dir

**Download data**

.. code-block:: bash

  git clone https://www.modelscope.cn/datasets/AI-ModelScope/DAPO-Math-17k.git

**Optional Hugging Face to Megatron conversion**

.. code-block:: bash

  python scripts/converter_hf_to_mcore.py \
      --hf_model_path Qwen/Qwen3-30B-A3B \
      --output_path Qwen/Qwen3-30B-A3B-mcore \
      --use_cpu_initialization    # Only work for MoE models

mbridge can load HF weights directly with these settings:

.. code-block:: bash

    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.use_mbridge=True

2. Training
^^^^^^^^^^^
Update model/data paths in the training script:

.. code-block:: bash

    # Model Weights Paths
    MODEL_PATH=Qwen/Qwen3-30B-A3B
    MCORE_MODEL_PATH=Qwen/Qwen3-30B-A3B-mcore
    RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
    CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

    # File System Paths
    TRAIN_FILE=$RAY_DATA_HOME/dataset/dapo-math-17k.parquet
    TEST_FILE=$RAY_DATA_HOME/dataset/aime-2024.parquet

    # Save frequency; -1 disables saves. Enable saves for later evaluation.
    trainer.save_freq=-1

Run the upstream `Qwen3-30B`_ single-node example with bash:

.. code-block:: bash

  bash examples/grpo_trainer/run_qwen3moe-30b_sglang_megatron_npu.sh

For multiple nodes, adapt this launcher:

.. code-block:: bash

  pkill -9 python
  ray stop --force
  rm -rf /tmp/ray
  export RAY_DEDUP_LOGS=0
  export HYDRA_FULL_ERROR=1
  # Dispatch queue: graph mode 1, eager mode 2.
  export TASK_QUEUE_ENABLE=1
  export HCCL_ASYNC_ERROR_HANDLING=0
  export HCCL_EXEC_TIMEOUT=3600
  export HCCL_CONNECT_TIMEOUT=3600

  export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
  export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # Select the training script.
  DEFAULT_SH="./run_*.sh"
  echo "Use $DEFAULT_SH"

  ulimit -n 32768
  mkdir logs

  NNODES=2
  NPUS_PER_NODE=8
  # Set the head-node address.
  MASTER_ADDR="IP FOR MASTER NODE"
  # Select this node's communication interface.
  SOCKET_IFNAME="Your SOCKET IFNAME"
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  # Resolve the local IP.
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # Start the head node.
    ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))

        # Wait for the expected node count.
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

        # Check connection status.
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

DEFAULT_SH: path to the training configuration script.

NNODES/NPUS_PER_NODE: participating nodes and NPUs per node; this example uses 2 and 8.

MASTER_ADDR: head-node IP, identical on all nodes.

SOCKET_IFNAME, HCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME select communication interfaces. Inspect interfaces with:

.. code-block:: bash

  ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1

3. Evaluation
^^^^^^^^^^^^^

The example uses Qwen3-30B; other models follow the same sequence.

AISBenchmark supports evaluation through vLLM and SGLang.

**Installation**

.. code-block:: bash

  git clone https://gitee.com/aisbench/benchmark.git
  cd benchmark
  pip install -e .
  pip install math_verify latex2sympy2_extended

**Download evaluation data**

.. code-block:: bash

  cd path/to/benchmark/ais_bench/datasets
  wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
  unzip math.zip
  rm math.zip

**Configure AISBench for SGLang**

Edit benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py:

.. code-block:: bash

    from ais_bench.benchmark.models import VLLMCustomAPIChatStream
    from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content
    from ais_bench.benchmark.clients import OpenAIChatStreamClient, OpenAIChatStreamSglangClient

    models = [
        dict(
            attr="service",
            type=VLLMCustomAPIChatStream,
            abbr='sgl-api-stream-chat',
            path="/path/to/Qwen3-30B",
            model="qwen3-30b",
            request_rate = 0,
            max_seq_len=2048,
            retry = 2,
            host_ip = "localhost",
            host_port = 8005,
            max_out_len = 8192,
            batch_size=48,
            trust_remote_code=False,
            custom_client=dict(type=OpenAIChatStreamSglangClient),
            generation_kwargs = dict(
                temperature = 0,
                seed = 1234,
            ),
            pred_postprocessor=dict(type=extract_non_reasoning_content)
        )
    ]


**Start sglang_server**

.. code-block:: bash

    python -m sglang.launch_server --model-path "/path/to/Qwen3-30B"  --tp-size 4 --dp-size 1 --port 8005

**Run client evaluation**

.. code-block:: bash

    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt

**Upstream reported results**

The upstream example reports improved Math-500 scores after training:

.. list-table::
   :header-rows: 1

   * - iter
     - dataset
     - version
     - metric
     - mode
     - sgl-api-stream-chat
   * - 0
     - math_prm800k_500
     - c4b6f0
     - accuracy
     - gen
     - 84.4
   * - 150
     - math_prm800k_500
     - c4b6f0
     - accuracy
     - gen
     - 91.7

Profiling
---------
See the `NPU profiling guide <../../dev_guide/performance/ascend_profiling_en.rst>`_.

The Qwen3-30B script provides PROF_CONFIG. global_profiler.steps=null disables profiling; select steps as needed.

Analyze captures with `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_.

Full profiling can generate large, repetitive operator traces; limit captures to relevant stages.
