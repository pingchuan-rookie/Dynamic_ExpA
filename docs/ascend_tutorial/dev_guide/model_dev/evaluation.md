# Model evaluation

Last updated: 07/14/2026.

The steps are the same across models; Qwen3-30B is used here as an example.

Models are evaluated with AISBench, which supports inference backends including vLLM and SGLang.

## 1. Installation

~~~bash
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
~~~


## 2. Download evaluation datasets

~~~bash
cd path/to/benchmark/ais_bench/datasets
wget https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip math.zip
rm math.zip
~~~

## 3. Convert model weights

verl supports saving HF-format weights directly through mbridge; these weights can be used without conversion.

Weights in other formats must be converted to HF format before evaluation.

Follow the upstream verl [checkpoint conversion instructions](../../../advance/checkpoint.rst).

## 4. Evaluation with vLLM

**Start the vllm_server service**

Start the inference service with the following command, adjusting `model` and `tensor-parallel-size`.

model: path to the trained model after conversion to Hugging Face format;

tensor-parallel-size: tensor-parallel degree; use the same TP setting as inference during training where possible;

data-parallel-size: data-parallel degree, defaulting to 1; use the same DP setting as inference during training where possible;

port: any available port;

~~~bash
vllm serve /path/to/Qwen3-30B/ \
       --served-model-name auto \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 24 \
       --max-model-len 22528 \
       --max-num-batched-tokens 22528 \
       --enforce-eager \
       --trust-remote-code \
       --distributed_executor_backend=mp \
       --tensor-parallel-size 8 \
       --data-parallel-size 1 \
       --generation-config vllm \
       --port 8080
~~~

**Configure AISBench and start vllm_client evaluation**

Open the inference configuration file `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py`.

Set `host_port` to the server port, and adjust `max_seq_len` and `max_out_len` for the model.
~~~bash
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        max_out_len = 512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
~~~

Run evaluation in another terminal:
~~~bash
    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt
~~~
## 5. Evaluation with SGLang
Follow the evaluation section in [SGLang best practices](../../model_support/examples/ascend_sglang_best_practices.rst).
