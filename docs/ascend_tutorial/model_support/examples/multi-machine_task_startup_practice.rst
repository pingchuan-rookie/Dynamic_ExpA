Multi-node startup
==================

Last updated: 07/28/2026.

Introduction
------------

Large-model training can require multiple nodes. verl uses Ray for distributed scheduling;
configure Ray and Ascend environment variables consistently across nodes.

This guide covers:

1. Prerequisites
2. Multi-node startup

Prerequisites
-------------

1. Environment and networking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure that:

- All nodes follow the `installation guide <../../get_start/install_guidance.rst>`_ with matching verl, Ray, PyTorch, torch_npu and CANN versions.
- Training networks allow Ray, dashboard and configured HCCL TCP ports. Ping alone does not establish port reachability.
- Script/model/data/checkpoint paths match across nodes, preferably on a shared filesystem such as NFS.
- NPU drivers/CANN are installed and npu-smi info detects devices.
- Clocks are synchronized for meaningful logs.

2. Communication interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inspect IPv4 interfaces on every node:

.. code-block:: bash

  ip -o -4 addr show scope global | awk '{print $2, $4}'

Choose the training interface and record its name per node. To find the route to a known head-node IP:

.. code-block:: bash

  MASTER_ADDR="IP FOR MASTER NODE"
  ip route get "$MASTER_ADDR" | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}'

Use that interface for HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME and launcher SOCKET_IFNAME.

3. Node roles
^^^^^^^^^^^^^

The cluster has one head node and one or more workers:

- **Head:** starts Ray, schedules work and begins training after all workers join.
- **Workers:** register with the head and wait for assignments.

Select the head and record its IP.

Start the cluster
-----------------

1. Environment variables
^^^^^^^^^^^^^^^^^^^^^^^^

Configure every node:

.. code-block:: bash

  # Ray log deduplication and detailed errors.
  export RAY_DEDUP_LOGS=0
  export HYDRA_FULL_ERROR=1

  # Ascend dispatch: graph mode 1, eager mode 2.
  export TASK_QUEUE_ENABLE=1

  # HCCL timeouts in seconds; adjust for model scale.
  export HCCL_ASYNC_ERROR_HANDLING=0
  export HCCL_EXEC_TIMEOUT=3600
  export HCCL_CONNECT_TIMEOUT=3600

  # HCCL ports; avoid conflicts.
  export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
  export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

  # Visible NPUs.
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

  # Local communication interface.
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"

  # File descriptor limit.
  ulimit -n 32768

  # Optional:
  # Disable asynchronous HF loading if host memory peaks are excessive.
  export HF_DEACTIVATE_ASYNC_LOAD=1

2. Launcher script
^^^^^^^^^^^^^^^^^^

Run this example on each node; it selects head/worker behavior from the local IP:

.. code-block:: bash

  # Use only on dedicated nodes where stopping the prior Ray instance is authorized.
  pkill -9 python
  ray stop --force
  rm -rf /tmp/ray

  # Deployment configuration.
  # Training script.
  DEFAULT_SH="./run_*.sh"
  echo "Use $DEFAULT_SH"

  # Node and NPU counts.
  NNODES=2
  NPUS_PER_NODE=16

  # Head-node IP.
  MASTER_ADDR="IP FOR MASTER NODE"

  # Local communication interface.
  SOCKET_IFNAME="Your SOCKET IFNAME"
  # End configuration.

  # Resolve the local IP.
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # Head node.
    ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))

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
    # Worker node.
    while true; do
        ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

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

**Parameters:**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``DEFAULT_SH``
     - Training script, such as run_qwen3moe-30b_grpo_megatron_vllm_npu.sh
   * - ``NNODES``
     - Number of nodes
   * - ``NPUS_PER_NODE``
     - NPUs per node, commonly 16 on Atlas 800T A3
   * - ``MASTER_ADDR``
     - Head-node IP, identical on all nodes
   * - ``SOCKET_IFNAME``
     - Local communication interface, which may differ by node

3. Launch training
^^^^^^^^^^^^^^^^^^

Save as ray_start.sh and run on every node:

.. code-block:: bash

  bash ray_start.sh

Suggested order:

1. Start the head and wait for Ray readiness.
2. Start workers, which register automatically.
3. The head launches training once all nodes have joined.

4. Monitor training
^^^^^^^^^^^^^^^^^^^

Monitor through:

**Ray Dashboard**

Open http://<MASTER_ADDR>:8260 for Ray status, resources and tasks.

**Command line**

.. code-block:: bash

  ray status

**Training logs**

Log locations depend on DEFAULT_SH. If the script writes a log, inspect it with:

.. code-block:: bash

  tail -f <TRAINING_LOG_PATH>
