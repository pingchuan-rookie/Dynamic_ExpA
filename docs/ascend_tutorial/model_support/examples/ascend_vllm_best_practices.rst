Ascend vLLM Best Practice
=========================

Last updated: 06/06/2026.

.. _Qwen3-30B: https://github.com/verl-project/verl/blob/release/v0.7.1/examples/grpo_trainer/run_qwen3moe-30b_grpo_megatron_vllm_npu.sh

.. _doclink: https://github.com/verl-project/verl/blob/c98cb8cc/docs/ascend_tutorial/examples/ascend_vllm_best_pratice.rst

Introduction
------------

vLLM is a high-performance open-source inference engine supported by Ascend in verl.
These upstream examples cover:

1. Environment setup
2. Training and evaluation
3. Profiling

Models and hardware:

- Use commit c98cb8cc documentation (`doclink`_) and scripts to preserve the historical example paths.

.. list-table::
   :header-rows: 1

   * - Model
     - NPU model
     - Nodes
     - Training/inference
   * - `Qwen3-30B`_
     - Atlas 800T A3
     - 1
     - vLLM + Megatron


Environment setup
-----------------
The `installation guide <../../get_start/install_guidance.rst>`_ supports Dockerfile or custom Conda setup.

This example additionally pins verl:

.. code-block:: bash

    cd verl
    git checkout release/v0.7.1

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
  huggingface-cli download --resume-download Qwen/Qwen3-30B-A3B-Base --local-dir /path/to/local_dir

**Download data**

.. code-block:: bash

  git clone https://www.modelscope.cn/datasets/modelscope/gsm8k.git

**Optional Hugging Face to Megatron conversion**

.. code-block:: bash

  python scripts/converter_hf_to_mcore.py \
      --hf_model_path Qwen/Qwen3-30B-A3B-Base \
      --output_path Qwen/Qwen3-30B-A3B-Base-mcore \
      --use_cpu_initialization    # Only work for MoE models

mbridge supports direct HF loading with these parameters:

.. code-block:: bash

    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.use_mbridge=True

2. Training
^^^^^^^^^^^
Update paths in the training script:

.. code-block:: bash

    # Model Weights Paths
    MODEL_PATH=Qwen/Qwen3-30B-A3B-Base
    MCORE_MODEL_PATH=Qwen/Qwen3-30B-A3B-Base-mcore
    RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
    CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

    # File System Paths
    TRAIN_FILE=$RAY_DATA_HOME/dataset/gsm8k/test.parquet
    TEST_FILE=$RAY_DATA_HOME/dataset/gsm8k/test.parquet

    # Save frequency; -1 disables saves. Enable saves for later evaluation.
    trainer.save_freq=-1

Run the upstream single-node `Qwen3-30B`_ script with bash:

.. code-block:: bash

  bash examples/grpo_trainer/run_qwen3moe-30b_grpo_megatron_vllm_npu.sh

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
  # Select the local communication interface.
  SOCKET_IFNAME="Your SOCKET IFNAME"
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  # Resolve the local IP.
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # Start the head.
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

DEFAULT_SH selects the training configuration script.

NNODES/NPUS_PER_NODE select nodes and NPUs per node; this example uses 2 and 8.

MASTER_ADDR is the head-node IP and must match across nodes.

Set SOCKET_IFNAME, HCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME to communication interfaces. Inspect them with:

.. code-block:: bash

  ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1

3. Evaluation
^^^^^^^^^^^^^

Qwen3-30B is shown; other models follow the same sequence.

AISBenchmark supports vLLM/SGLang evaluation.

**Installation**

.. code-block:: bash

  git clone https://gitee.com/aisbench/benchmark.git
  cd benchmark
  pip install -e .
  pip install math_verify latex2sympy2_extended

**Download evaluation data**

.. code-block:: bash

  cd /examples/benchmark/ais_bench/datasets
  mkdir aime/
  cd aime/
  wget https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
  unzip aime.zip
  rm aime.zip

**Configure AISBench for vLLM**

.. code-block:: bash

   vim /examples/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py

Match host_port to the service. Adjust max_seq_len/max_out_len for the model; this example uses 2k input and 20k output:

.. code-block:: bash

  from ais_bench.benchmark.models import VLLMCustomAPI

  models = [
      dict(
          attr="service",
          type=VLLMCustomAPI,
          abbr='vllm-api-general',
          path="/path/to/Qwen3-30B",
          model="qwen3-30b",
          request_rate = 0,
          retry = 2,
          host_ip = "localhost",
          host_port = 6380,
          max_seq_len = 2048,
          max_out_len = 20480,
          batch_size=48,
          trust_remote_code=False,
          generation_kwargs = dict(
              temperature = 0.5,
              top_k = 10,
              top_p = 0.95,
              seed = None,
              repetition_penalty = 1.03,
          )
      )
  ]


**Start vllm_server**

Update model and tensor-parallel-size in the NPU service command:

/path/to/Qwen3-30B/ selects trained HF weights.
tensor-parallel-size selects TP; match the training rollout setting when comparing.
data-parallel-size selects DP, default 1; align it with the intended rollout configuration.
port selects an available service port.

.. code-block:: bash

  cd /path/to/vllm
  vllm serve /path/to/Qwen3-30B/ \
      --served-model-name auto \
      --gpu-memory-utilization 0.9 \
      --max-num-seqs 24 \
      --max-model-len 10240 \
      --max-num-batched-tokens 10240 \
      --enforce-eager \
      --trust-remote-code \
      --distributed_executor_backend=mp \
      --tensor-parallel-size 4 \
      --data-parallel-size 1 \
      --generation-config vllm \
      --port 6380


**Run client evaluation**

.. code-block:: bash

  cd /examples/benchmark
  ais_bench --models vllm_api_general --datasets aime2024_gen


**Upstream reported results**

The upstream example reports improved aime2024 scores after training:

.. list-table::
   :header-rows: 1

   * - iter
     - dataset
     - version
     - metric
     - mode
     - vllm-api-stream-chat
   * - 0
     - aime2024
     - a4b6f0
     - accuracy
     - gen
     - 85.4
   * - 150
     - aime2024
     - a4b6f0
     - accuracy
     - gen
     - 91.2

Profiling
---------
See the `NPU profiling guide <../../dev_guide/performance/ascend_profiling_en.rst>`_.

Analyze captures with `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_.

Full captures can produce large, repetitive traces; profile only relevant stages.
