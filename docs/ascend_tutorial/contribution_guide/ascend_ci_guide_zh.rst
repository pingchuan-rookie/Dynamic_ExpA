Adding NPU CI coverage
======================

Last updated: 02/02/2026.

This reference describes upstream verl CI on Huawei Ascend devices. In this checkout, developer tests are provisioned separately under ignored dyad_test/; upstream workflow examples do not establish active CI coverage.

Upstream verl uses GitHub Actions with layered tests for code quality and system behavior.
NPU workflows include:

* ``npu_unit_test.yml`` for unit tests.
* ``*_ascend.yml`` for Ascend end-to-end or feature-specific tests.

Adding test cases
-----------------

1. Data and weights
^^^^^^^^^^^^^^^^^^^
Models and absolute paths on the upstream runner:

.. list-table::
   :header-rows: 1

   * - Model
     - Absolute path
   * - Qwen2.5-0.5B
     - ``${HOME}/.cache/models/Qwen/Qwen2.5-0.5B``
   * - Qwen2.5-0.5B-Instruct
     - ``${HOME}/.cache/models/Qwen/Qwen2.5-0.5B-Instruct``
   * - Qwen2.5-1.5B-Instruct
     - ``${HOME}/.cache/models/Qwen/Qwen2.5-1.5B-Instruct``
   * - Qwen2.5-7B-Instruct
     - ``${HOME}/.cache/models/Qwen/Qwen2.5-7B-Instruct``
   * - Qwen2.5-VL-3B-Instruct
     - ``${HOME}/.cache/models/Qwen/Qwen2.5-VL-3B-Instruct``
   * - Qwen3-0.6B
     - ``${HOME}/.cache/models/Qwen/Qwen3-0.6B``
   * - Qwen3-8B
     - ``${HOME}/.cache/models/Qwen/Qwen3-8B``
   * - Qwen3-8B-Base
     - ``${HOME}/.cache/models/Qwen/Qwen3-8B-Base``
   * - Qwen3-30B-A3B-Instruct-2507
     - ``${HOME}/.cache/models/Qwen/Qwen3-30B-A3B-Instruct-2507``
   * - Qwen3-32B
     - ``${HOME}/.cache/models/Qwen/Qwen3-32B``
   * - Qwen3-VL-2B-Instruct
     - ``${HOME}/.cache/models/Qwen/Qwen3-VL-2B-Instruct``
   * - Qwen3-VL-4B-Instruct
     - ``${HOME}/.cache/models/Qwen/Qwen3-VL-4B-Instruct``
   * - Qwen3-4B-Instruct-2507
     - ``${HOME}/.cache/models/Qwen/Qwen3-4B-Instruct-2507``
   * - Qwen3-VL-8B-Instruct
     - ``${HOME}/.cache/models/Qwen/Qwen3-VL-8B-Instruct``
   * - Skywork-Reward-V2-Llama-3.2-1B
     - ``${HOME}/.cache/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B``
   * - Qwen3.5-2B
     - ``${HOME}/.cache/models/Qwen/Qwen3.5-2B``

Datasets and absolute paths:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Absolute path
   * - gsm8k
     - ``${HOME}/.cache/datasets/openai/gsm8k``
   * - geo3k
     - ``${HOME}/.cache/datasets/hiyouga/geometry3k``

**Note**

   The runner uses /root as HOME.

   GPU examples use ~/models. A compatibility link is ``ln -s /root/.cache/models ~/models``.

   These are raw datasets; preprocess them as needed, for example:

   ``python examples/data_preprocess/gsm8k_multiturn_sft.py --local_dataset_path ${HOME}/.cache/datasets/openai/gsm8k``


2. Workflow YAML template
^^^^^^^^^^^^^^^^^^^^^^^^^

For an upstream workflow, adapt this example as .github/workflows/your_yml_ascend.yml.

Customize:

* Workflow name (``name``)
* Triggers (``on``)
* Runner (``runs-on``)
* Image (``container.image``)
* Steps (``jobs.<job_id>.steps``)

.. code-block:: yaml
   :linenos:

   name: your_yml_ascend
   # Trigger configuration.
   on:
     push:
       branches:
         - main
         - v0.*
     pull_request:
       branches:
         - main
       paths:
         - ".github/workflows/your_yml_ascend.yml"
         - "path/to/affected_files"

   # Cancel obsolete runs outside main.
   concurrency:
     group: ${{ github.workflow }}-${{ github.ref }}
     cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}

   permissions:
     contents: read

   jobs:
     your_job_name:
       if: github.repository_owner == 'verl-project'
       runs-on: linux-aarch64-a2-4  # A2 with four NPUs.
       timeout-minutes: 60
       container:
         # vLLM image.
         image: swr.ap-southeast-1.myhuaweicloud.com/base_image/ascend-ci/verl/verl:latest-cann9.0.0-torch_npu2.9.0post2-910b-ubuntu22.04-py3.11-vllm
         options: >-
           --shm-size 16g
       env:
         HF_ENDPOINT: "https://hf-mirror.com"
         HF_HUB_ENABLE_HF_TRANSFER: "0"
       steps:
         - name: Check npu and CANN info
           run: |
             cat /usr/local/Ascend/ascend-toolkit/latest/"$(uname -i)"-linux/ascend_toolkit_install.info
             npu-smi info
         - name: Check initial pip list from image
           run: pip list
         - name: Checkout repository
           uses: actions/checkout@v4
           with:
             fetch-depth: 0
             clean: true
         - name: Install dependencies
           run: |
             pip install --no-deps -e .
         - name: Verify environment
           run: pip list
         # Add the required test steps.
         - name: Preprocess dataset
           run: python examples/data_preprocess/your_script.py --local_dataset_path ${HOME}/.cache/datasets/your_dataset
         - name: Execute NPU test
           run: |
             ray stop --force
             bash tests/special_npu/your_test_script.sh

**Note**


   Runner ${HOME}/.cache persists after containers are destroyed. Avoid adding unnecessary content.


3. Unit tests
^^^^^^^^^^^^^

Steps:

(1) Add or modify a test in the upstream tests/ tree, such as test_xxx.py.
(2) Files not excluded by npu_unit_test.yml --ignore-glob run through:

   .. code-block:: yaml

      pytest -s -x --ignore-glob="xxx" --ignore-glob="xxx" tests/

(3) Add an explicit workflow step for tests excluded by ignore-glob.
(4) A separate workflow can group a substantial related suite.

4. End-to-end scripts
^^^^^^^^^^^^^^^^^^^^^

Steps:

(1) Add the script under upstream tests/special_npu/.
(2) Invoke it from the closest matching ``*_ascend.yml`` workflow.
(3) Use a separate workflow for independent or complex scenarios.

5. Test strategy
^^^^^^^^^^^^^^^^

* **Unit tests:** cover core function/class behavior.
* **Integration/end-to-end tests:** cover representative training/inference pipelines and hardware integration.
* **Resources:** jobs within a workflow run concurrently. Set timeouts and target less than 40 minutes per job.

Verify that the configured workflow actually runs the intended tests before treating their presence as CI coverage.
