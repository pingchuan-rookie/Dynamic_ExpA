Ascend Dockerfile Build Guidance
================================

Last updated: 08/10/2026.


Published images and image sources
----------------------------------

Ascend hosts daily A2/A3 images at `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_. They are built from the `Dockerfiles <../../../docker/ascend>`_; see :ref:`the Dockerfile inventory <ascend-dockerfile-list>`.

Daily image naming: latest-{inference-backend}-{product-information}-{operating-system}-{other-fields}

verl release image naming: {verl-release}-{CANN-version}-{TorchNPU-version}[-{product-information}-{operating-system}]-{Python-version}[-{inference-backend}-{other-fields}]



Supported hardware
------------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3


Component versions in the latest image
--------------------------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Version
   * - Base image
     - Ubuntu 22.04
   * - Python
     - 3.12
   * - CANN
     - 9.1.0
   * - torch
     - 2.10.0
   * - torch_npu
     - 2.10.0.post4
   * - torchvision
     - 0.25.0
   * - vLLM
     - 0.23.0
   * - vLLM-ascend
     - 0.23.0
   * - Megatron-LM
     - core_r0.16.0
   * - MindSpeed
     - core_r0.16.0
   * - triton-ascend
     - 3.2.2
   * - mbridge
     - 0.15.1
   * - SGLang
     - v0.5.10
   * - sgl-kernel-npu
     - 2026.02.01



.. _ascend-dockerfile-list:

Dockerfile inventory
--------------------

**General-purpose images**

.. list-table::
   :header-rows: 1

   * - Device type
     - CANN base-image version
     - Inference backend
     - Reference file
   * - A2
     - 9.1.0
     - vLLM
     - `Dockerfile.ascend_9.1.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a2>`_
   * - A3
     - 9.1.0
     - vLLM
     - `Dockerfile.ascend_9.1.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a3>`_
   * - A2
     - 8.5.0
     - vLLM
     - `Dockerfile.ascend_8.5.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2>`_
   * - A3
     - 8.5.0
     - vLLM
     - `Dockerfile.ascend_8.5.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3>`_
   * - A2
     - 8.5.0
     - SGLang
     - `Dockerfile.ascend.sglang_8.5.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a2>`_
   * - A3
     - 8.5.0
     - SGLang
     - `Dockerfile.ascend.sglang_8.5.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a3>`_
   * - A2
     - 8.3.RC1
     - vLLM
     - `Dockerfile.ascend_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a2>`_
   * - A3
     - 8.3.RC1
     - vLLM
     - `Dockerfile.ascend_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a3>`_
   * - A2
     - 8.3.RC1
     - SGLang
     - `Dockerfile.ascend.sglang_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a2>`_
   * - A3
     - 8.3.RC1
     - SGLang
     - `Dockerfile.ascend.sglang_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a3>`_
   * - A2
     - 8.2.RC1
     - vLLM
     - `Dockerfile.ascend_8.2.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a2>`_
   * - A3
     - 8.2.RC1
     - vLLM
     - `Dockerfile.ascend_8.2.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a3>`_


**verl release images**

.. list-table::
   :header-rows: 1

   * - Device type
     - CANN base-image version
     - Inference backend
     - verl version
     - Reference file
   * - A2
     - 9.0.0
     - vLLM
     - release/v0.8.0
     - `Dockerfile.ascend_9.0.0_a2_v0.8.0 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2_v0.8.0>`_
   * - A3
     - 9.0.0
     - vLLM
     - release/v0.8.0
     - `Dockerfile.ascend_9.0.0_a3_v0.8.0 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3_v0.8.0>`_
   * - A2
     - 8.5.0
     - vLLM
     - release/v0.7.1
     - `Dockerfile.ascend_8.5.0_a2_v0.7.1 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2_v0.7.1>`_
   * - A3
     - 8.5.0
     - vLLM
     - release/v0.7.1
     - `Dockerfile.ascend_8.5.0_a3_v0.7.1 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3_v0.7.1>`_


**Model-specific images**

.. list-table::
   :header-rows: 1

   * - Device type
     - CANN base-image version
     - Inference backend
     - Model
     - Reference file
   * - A2
     - 8.5.2
     - vLLM
     - Qwen3.5
     - `Dockerfile.ascend_8.5.2_a2_qwen3-5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.2_a2_qwen3-5>`_
   * - A3
     - 8.5.2
     - vLLM
     - Qwen3.5
     - `Dockerfile.ascend_8.5.2_a3_qwen3-5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.2_a3_qwen3-5>`_



**Notes:**

* Images using ``vLLM`` install vLLM, vLLM-Ascend, MindSpeed, Megatron-LM and verl from source, located under the image root ``/``.
* Images using ``SGLang`` install SGLang, MindSpeed and verl from source, located under the image root ``/``.


Example image build command
---------------------------

.. code:: bash

   # Navigate to the directory containing the Dockerfile
   cd {verl-root-path}/docker/ascend

   # Build the image
   # vLLM
   docker build -f Dockerfile.ascend_8.5.0_a2 -t verl-ascend:8.5.0-a2 .
   # SGLang
   docker build -f Dockerfile.ascend.sglang_8.5.0_a2 -t verl-ascend-sglang:8.5.0-a2 .

   # Query local images after build
   docker images

**Notes:**

* In the vLLM example, ``Dockerfile.ascend_8.5.0_a2`` is the Dockerfile. In ``verl-ascend:8.5.0-a2``, verl-ascend is the chosen image name and 8.5.0-a2 is the chosen tag.

Container launch template
-------------------------

.. code:: bash

   docker run -dit \
       --ipc=host \
       --network host \
       --name {your_docker_name} \
       --privileged \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
       -v /usr/local/sbin:/usr/local/sbin \
       -v /usr/sbin:/usr/sbin \
       -v /home:/home \
       -v /data:/data \
       {image_name}:{tag} \
       /bin/bash

**Notes:**

* Add ``-v <host-path>:<container-path>`` for additional local mounts.
* Replace ``{your_docker_name}`` with a meaningful container name.
* ``--privileged`` grants expanded container privileges; assess whether the deployment requires it.
* Replace ``{image_name}:{tag}`` with the image name and tag used during the build.

Start the container
-------------------

.. code:: bash

   docker start {your_docker_name}

Enter a running container
-------------------------

.. code:: bash

   docker exec -it {your_docker_name} bash


Scope
-----
Ascend Dockerfiles and images supplied by verl are reference examples for evaluation. Contact the official support channels before adopting them in production.
