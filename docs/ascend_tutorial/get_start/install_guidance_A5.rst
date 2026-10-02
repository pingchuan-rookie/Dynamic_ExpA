Last updated: 08/03/2026.

Supported versions and dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - CANN
     - Link pending the Q2 commercial CANN release
     - Software for developing and running AI workloads on Ascend hardware
   * - Python
     - ``3.11``
     - Python version
   * - torch
     - ``2.10.0``
     - PyTorch deep-learning framework
   * - torch_npu
     - Link pending the Q2 commercial torch_npu release
     - PyTorch adapter for NPUs
   * - triton
     - ``3.5.0``
     - Triton for custom kernels
   * - triton-ascend
     - ``3.2.2``
     - Triton adapter for NPUs
   * - transformers
     - ``4.57.6``
     - Hugging Face architectures and pretrained weights
   * - vLLM
     - ``0.23.0``
     - High-performance LLM inference and serving
   * - vLLM-Ascend
     - ``0.23.0``
     - vLLM backend for NPUs
   * - Megatron-LM
     - ``core_r0.12.0``
     - Large-scale distributed training framework
   * - MindSpeed
     - ``0c6c0ceaa523a96032dee1539a52032155e6404e``
     - Megatron-LM adaptation and optimization for Ascend NPUs

Environment installation
^^^^^^^^^^^^^^^^^^^^^^^^

vLLM inference backend
~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    # Install vLLM
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout v0.23.0
    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..

    # Install vLLM-Ascend
    # Source CANN before installation: source /usr/local/Ascend/cann/set_env.sh
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout releases/v0.23.0
    pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    cd ..


Megatron training backend
~~~~~~~~~~~~~~~~~~~~~~~~~

Install MindSpeed, Megatron and their dependencies from source:

.. code:: bash

    # MindSpeed
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed
    git checkout 0c6c0ceaa523a96032dee1539a52032155e6404e
    pip install -e .
    cd ..

    # Megatron
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.12.0
    pip install -e .
    cd ..

    # Configure environment variables
    export PYTHONPATH=$PYTHONPATH:your_path/Megatron-LM
    export PYTHONPATH=$PYTHONPATH:your_path/MindSpeed

    # Install mbridge
    pip install mbridge

Install verl dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    git clone https://github.com/verl-project/verl.git
    cd verl
    pip install -r requirements-npu.txt
    pip install -e .

