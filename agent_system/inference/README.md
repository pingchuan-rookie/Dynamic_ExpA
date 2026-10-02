# vLLM-powered inference

`agent_system/inference` is the standalone vLLM-powered layer for loading models and generating responses for text baselines and Dyad.
Text generation uses vLLM's `AsyncLLM`; `DyadAsyncLLM` integrates action encoding and candidate selection into the vLLM generation path.
The HTTP layer exposes `/v1/chat/completions` for generation and `/v1/models` for model identity;
the evaluator executes environments and computes scores.

The Python API owns model restoration and generation. It accepts prompt token IDs,
vLLM `SamplingParams`, optional `tools` and action `args`, and returns the complete
vLLM `RequestOutput` with verified Dyad sampling evidence. HTTP rendering and response
serialization live in `backend.py`; the engine does not import the HTTP server.

```python
import asyncio
from vllm import SamplingParams
from agent_system.inference import InferenceConfig, VLLMInference

async def main():
    config = InferenceConfig(model_path="/absolute/model")
    async with VLLMInference(config) as inference:
        tokens = inference.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain recursion briefly."}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
            enable_thinking=False,
        )
        result = await inference.generate(
            tokens, SamplingParams(max_tokens=256, temperature=0), args={},
        )
        print(result.output.outputs[0].text)

asyncio.run(main())
```

Use `InferenceConfig(checkpoint=..., restore_dir=...)` for a native checkpoint,
or `projector_init=..., model_config=...` for Alignment initialization. The Python
and HTTP paths use the same restoration code. Python callers supply rendered tokens;
the HTTP adapter applies the model's chat template and thinking settings.

Run from the repository root:

```bash
# Initial text model
.venvs/expa-verl/bin/python -m agent_system.inference.server \
  --model-path /absolute/hf/model --model evaluated-model --tensor-parallel-size 1

# Full trained checkpoint: infer text/Dyad from saved model_config.json
.venvs/expa-verl/bin/python -m agent_system.inference.server \
  --checkpoint /absolute/run/global_step_100 --model evaluated-model \
  --restore-dir /absolute/artifacts/outputs/inference/run_100
```

A full Dyad checkpoint restores the policy, projector and any trained independent encoder. A frozen encoder uses its saved source identity.
Native shards are reconstructed strictly on CPU; inference TP is independent of the training FSDP rank count. Alignment initialization instead uses
`--projector-init /absolute/projector.pt --model-config /absolute/model_config.json` and does not represent a full checkpoint restore.
Full restoration currently supports an independent `encoder_lm`, initialized from the saved original policy base;
`projector_only` and `projector_and_encoder_lm` retain their respective weight semantics. Unsupported backbones fail explicitly.
Source weights are read-only, and restoration must use a separate artifact directory. `--check` validates the selection without loading models.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
answer = client.chat.completions.create(
    model="evaluated-model",
    messages=[{"role": "user", "content": "Reply with OK only."}],
    max_tokens=32,
    temperature=0,
    extra_body={"args": {}, "chat_template_kwargs": {"enable_thinking": False}},
)
print(answer.choices[0].message.content)
```

For text-only requests, omit `tools` or pass `[]`, and omit `args` or pass `{}`. The loaded model generates text directly,
without a synthetic `respond` action, action encoding or candidate selection triggered by action markers in text.
For tool requests, use standard `tools=[{"type":"function","function":...}]`. Dyad uses its action encoder
to construct the request's candidate head and returns standard `tool_calls` plus `dyad` sampling evidence.
The LM still generates tool-call `arguments`; request-level `args` contains additional action configuration.

Environment evaluation selects a built-in action schema with `args.schema_name` or supplies a per-task schema through `args.schema`.
`t2bench` uses `args.protocol="tau_json"`; other native tools use `native_tools`.
`template_tools`, `strip_thinking_prefill` and `prompt_limit` preserve training-time prompt and length semantics.
These options do not rewrite environment feedback or repair illegal actions. Environments handle blank, incomplete and malformed responses.

The service accepts explicit generation budgets, temperature, top_p, top_k, seed, stop and common repetition penalties.
Only non-streaming, single-response requests are supported (`stream=false`, `n=1`); evaluators issue independent requests for repeated sampling.
Unsupported controls fail explicitly. Qwen3.5 retains its no-thinking setting.
Outputs retain token IDs, usage, stop reasons and action evidence; the service does not execute tools or score results.

Environment evaluation uses `experiments/shared/train_eval/evaluate.sh`. Pass `--api-url` and `--served-model` to use an existing service;
otherwise the entrypoint starts and cleans up its own service. Environments reuse the existing Ray session pools; Ray does not manage the policy or encoder.
Encoding defaults to CPU and can be changed with `--encoder-device`. GPU/TP behavior, long-context capacity and throughput require validation on the actual hardware.
