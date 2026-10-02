Ascend Install Guidance
=======================

Last updated: 2026/08/03.

Updates
-------

-  2026/08/03: vLLM/vLLM-Ascend moved from 0.18.0 to 0.23.0,
   with torch 2.10.0 and torch_npu 2.10.0.post2.

-  2026/05/13: `PR
   #6291 <https://github.com/verl-project/verl/pull/6291>`__ updated vLLM/
   vLLM-Ascend from 0.13.0 to 0.18.0,
   with torch 2.9.0 and torch_npu
   ``2.9.0.post2``\ 。

-  2025/12/11: existing verl scenarios detect NPUs automatically.
   GPU scripts generally no longer need an explicit
   trainer.device=npu setting. New features can still
   select a device explicitly with trainer.device.

..

.. note::

   Automatic detection requires torch_npu. Without it, explicitly set trainer.device=npu.

   A3 cards have two dies, A2 one. For the eight-card A3 examples, set n_gpus_per_node=16.

Contents
--------

- `Hardware support <#hardware-support>`_
- `Backend support <#backend-support>`_
- `Deployment <#deployment>`_
  - `Docker images <#docker-images>`_
  - `vLLM with FSDP/Megatron <#vllm-with-fsdp-megatron>`_
  - `SGLang with FSDP/Megatron <#sglang-with-fsdp-megatron>`_
  - `Additional training backends <#additional-training-backends>`_

- `Appendix <#appendix>`_

Hardware support
----------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3

`Ascend 950 series <install_guidance_A5.rst>`_


Backend support
---------------

Use published images from the `image guide <dockerfile_build_guidance.rst>`__ or install the supported NPU backends below.

.. list-table::
   :header-rows: 1

   * - Inference engine
     - Training engine
   * - vLLM
     - FSDP/FSDP2/Megatron
   * - SGLang
     - FSDP/FSDP2/Megatron

Training backend extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

verl separates inference and training backends to support custom integrations.

MindSpeed-LLM is an Ascend distributed LLM training suite integrated into verl; see `MindSpeed-LLM backend <#mindspeed-llm-backend>`_.


Deployment
----------

Docker images
~~~~~~~~~~~~~

Obtain images from `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_ or build from Dockerfiles.
See the `image guide <dockerfile_build_guidance.rst>`__.


vLLM with FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~


Versions and dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - HDK
     - ``26.0.rc1``
     - NPU driver and firmware
   * - CANN
     - ``9.0.0``
     - Ascend compute toolkit
   * - Python
     - ``>=3.10, <3.12``
     - Recommended: 3.11
   * - torch
     - ``2.10.0``
     - PyTorch
   * - torch_npu
     - ``2.10.0.post2``
     - PyTorch NPU adapter
   * - torchvision
     - ``0.25.0``
     - Vision library
   * - torchaudio
     - ``2.10.0``
     - Audio library
   * - triton
     - ``3.5.0``
     - Custom operator compiler
   * - triton-ascend
     - ``3.2.1``
     - NPU adapter; see installation script
   * - transformers
     - ``5.10.4``
     - Model architectures and weights
   * - vLLM
     - ``0.23.0``
     - LLM inference and serving
   * - vLLM-Ascend
     - ``0.23.0``
     - NPU vLLM backend
   * - Megatron-LM
     - ``core_r0.16.0``
     - Distributed training
   * - MindSpeed
     - ``core_r0.16.0``
     - Megatron NPU adaptation


Prepare HDK and CANN
^^^^^^^^^^^^^^^^^^^^

The following commands target Arm A3 systems.
For other hardware, obtain matching packages from the `CANN community <https://www.hiascend.com/cann/download?versionId=723&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_.

.. code:: bash

   # Configure the user group.
   sudo groupadd HwHiAiUser
   sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   # Install dependencies and package sources.
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install NPU drivers.
   sudo yum install -y Atlas-A3-hdk-npu-driver-26.0.rc1
   # Install the toolkit; --install-path selects a custom location.
   sudo yum install -y Ascend-cann-toolkit-9.0.0
   sudo yum install -y Ascend-cann-A3-ops-9.0.0
   # Verify installation.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

Install from source
^^^^^^^^^^^^^^^^^^^

The Conda-based `installer <../../../scripts/install_vllm_mcore_npu.sh>`_ runs in stages. Diagnose errors at the failing stage or report an issue with details.

.. code:: bash

   # Extra package index for x86.
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # Adjust this path for a custom CANN installation.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-vllm-npu python=3.11 -y
   conda activate verl-vllm-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_vllm_mcore_npu.sh
   # FSDP-only installation.
   # USE_MEGATRON=0 bash verl/scripts/install_vllm_mcore_npu.sh

SGLang with FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~

Versions and dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - HDK
     - ``25.5.0``
     - NPU driver and firmware
   * - CANN
     - ``>=8.5.0``
     - Ascend compute toolkit
   * - Python
     - ``>=3.10, <3.12``
     - Recommended: 3.11
   * - torch
     - ``2.8.0``
     - PyTorch
   * - torch_npu
     - ``2.8.0.post2``
     - PyTorch NPU adapter
   * - SGLang
     - ``v0.5.10``
     - LLM inference
   * - triton
     - ``3.5.0``
     - Custom operator compiler
   * - triton-ascend
     - ``3.2.1``
     - NPU adapter; see installation script
   * - transformers
     - ``5.3.0``
     - Model architectures and weights
   * - Megatron-LM
     - ``core_r0.16.0``
     - Distributed training
   * - MindSpeed
     - ``core_r0.16.0``
     - Megatron NPU adaptation


Prepare HDK and CANN
^^^^^^^^^^^^^^^^^^^^

These commands target Arm A3 systems.
For other hardware, use the `CANN community <https://www.hiascend.com/cann/download?versionId=680&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_.

.. code:: bash

   # Configure the user group.
   sudo groupadd HwHiAiUser
   sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   # Install dependencies and package sources.
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install NPU drivers.
   sudo yum install -y Atlas-A3-hdk-npu-driver-25.5.0
   # Install the toolkit; --install-path selects a custom location.
   sudo yum install -y Ascend-cann-toolkit-8.5.0
   sudo yum install -y Ascend-cann-A3-ops-8.5.0
   # Verify installation.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

Install from source
^^^^^^^^^^^^^^^^^^^

Use the staged Conda-based `SGLang installer <../../../scripts/install_sglang_mcore_npu.sh>`_. Diagnose errors at the failing stage or report details in an issue.

.. code:: bash

   # Extra package index for x86.
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # Adjust this path for a custom CANN installation.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-sgl-npu python=3.11 -y
   conda activate verl-sgl-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_sglang_mcore_npu.sh
   # FSDP-only installation.
   # USE_MEGATRON=0 bash verl/scripts/install_sglang_mcore_npu.sh

SGLang requirements
^^^^^^^^^^^^^^^^^^^

Set these variables for the NPU SGLang backend:

.. code:: bash

   # Multiple processes on one NPU.
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

   # Make Ray worker device selection explicit.
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

   # Match visible devices to the hardware allocation.
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   # in A3
   # export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

   # Required for inference expert parallelism.
   export SGLANG_DEEPEP_BF16_DISPATCH=1

Additional training backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MindSpeed-LLM backend
^^^^^^^^^^^^^^^^^^^^^

Download MindSpeed-LLM for the Megatron/MindSpeed training backend.
This integration requires MindSpeed-LLM master, MindSpeed master
and Megatron-LM core_r0.16.0.


Install sources and dependencies:

.. code:: bash

   # Fetch MindSpeed-LLM, MindSpeed and Megatron-LM.
   git clone https://gitcode.com/Ascend/MindSpeed-LLM.git
   git clone https://gitcode.com/Ascend/MindSpeed.git
   git clone --depth 1 --branch core_r0.16.0 https://github.com/NVIDIA/Megatron-LM.git

   # Configure environment variables.
   export PYTHONPATH=$PYTHONPATH:/your/path/Megatron-LM
   export PYTHONPATH=$PYTHONPATH:/your/path/MindSpeed
   export PYTHONPATH=$PYTHONPATH:/your/path/MindSpeed-LLM

   # Install mbridge.
   pip install mbridge

To select MindSpeed-LLM:

1. Set the worker model strategy to mindspeed, for example
   ``actor_rollout_ref.actor.strategy=mindspeed``\ 。

2. Pass MindSpeed-LLM settings through llm_kwargs. For MoE
   GMM support, use
   ``+actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_grouped_gemm=True``\ 。

3. See the `MindSpeed-LLM
   feature documentation <https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/docs/zh/pytorch/features/mcore>`__.

Appendix
--------

Unsupported ecosystem libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ascend support currently excludes:

.. list-table::
   :header-rows: 1

   * - Software
     - Notes
   * - ``flash_attn``
     - The standalone flash_attn package is unsupported; use the supported transformers integration.


