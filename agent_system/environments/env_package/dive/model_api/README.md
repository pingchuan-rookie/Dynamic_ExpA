# DIVE model API clients

This package implements DIVE's judge clients. `DiveEnv` owns the worker-local
client and injects it into `DiveJudge`; the judge owns the upstream verifier prompt,
parser and scoring. The adapters handle SDKs, authentication, request parameter
compatibility and response normalization. DIVE's browse tool and judge configuration
also use the local `providers/trapi.py` helpers.

```python
from agent_system.environments.env_package.dive.model_api import (
    GenerateOptions, Message, ModelConfig, create_model_client,
)

config = ModelConfig(
    provider="openai_compatible",
    model="your-model",
    base_url="https://model-api.example.com/v1",
    api_key_env="EXTERNAL_LLM_API_KEY",
)
with create_model_client(config, timeout_s=120, max_retries=0) as client:
    result = client.generate(
        [Message("user", "Reply with OK only.")],
        options=GenerateOptions(max_output_tokens=128, temperature=0),
    )
    print(result.text)
```

Supply credentials in the named environment variable in the calling worker.
Configuration contains only a credential reference, never the key itself.
For compatibility with local OpenAI servers, missing/empty keys use `EMPTY`.
TRAPI instead obtains refreshing Azure CLI / Managed Identity credentials locally.
The factory does not read shared or environment-specific model defaults: the
application entry point resolves and records its configuration before calling it.

Supported providers: `openai_compatible`, `trapi`, and `anthropic`. Provider SDKs
are imported only on use. Create clients inside workers, not before serialization.
Each `generate` opens and closes its transport (and TRAPI credentials), including
on failure or deadline interruption. `close` prevents further calls. The default
retry count is zero; callers can explicitly configure transport retries. There is
no model fallback. Blank responses and SDK failures raise sanitized `ModelAPIError`
with error type and HTTP status, without SDK request/response objects.

The initial interface supports system/user/assistant text messages and normalized
text, finish reason, token usage, and effective generation parameters. Streaming,
async generation, tool calls, and multimodal inputs are outside this interface.
Output budgets include reasoning where the provider counts it; they do not promise
an exact visible-text token count. GPT-5 parameter adaptation and the existing
Qwen3.5 no-thinking policy are handled in the OpenAI adapter.

These clients belong to the DIVE environment. Other environments handle their own
API calls; t2bench's user and judge transport lives in `env_package/t2bench/trapi.py`.
DIVE settings, checkpoint identities, verification prompts, deadlines and reward
semantics are unchanged by this relocation.
