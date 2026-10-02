# DIVE CPU environment runtime

This environment uses published DIVE tasks and upstream tools, not task synthesis or upstream model rollout.
The bundled source records its upstream revision, file hashes and local comment cleanup in `source/source_manifest.json`.
The cleanup removes comments and docstrings without changing executable statements or published JSON tool definitions.
The training driver owns policy inference; synchronous CPU Ray actors own task state, tool execution and terminal verification.
Ray process isolation is not a security sandbox.

## Prepare

From the workspace project root:

```bash
bash ops/env_deps/dive/prepare_source.sh
bash ops/env_deps/dive/install.sh
PYTHONPATH="$PWD" .venvs/dive/bin/python -m ops.env_deps.dive.verify \
  --driver-python "$PWD/.venvs/expa-verl/bin/python" \
  --dataset data/dive/DIVE-RL-3K/DIVE-RL-3K.jsonl
```

Source preparation clones only a missing destination and refuses to alter an existing checkout.
The installer creates `.venvs/dive` from the driver interpreter, syncs the pinned CPU dependencies, aligns Ray with the driver and verifies the exact Python patch version and Ray version.
It refuses a target equal to the driver environment or its parent, an existing non-venv directory, or an existing interpreter with a different Python patch version before syncing dependencies.
It does not install the upstream package editable or modify the training environment.
An existing destination must carry the installer's `.dyad-dive-managed` marker; otherwise choose a new directory rather than synchronizing an unknown environment.
`DIVE_REPO`, `DIVE_VENV`, `DYAD_PYTHON_BIN` and `DIVE_PYTHON_BIN` allow explicit deployment paths.
On every node, deploy the unchanged packaged environment sources, runtime bridge and isolated environment at the same paths.
The actor needs the repository root and `repos` on `PYTHONPATH`, but imports neither `torch` nor `verl`.
The tools-only verification imports the 200 tools in academic, biological and medical and checks selected published schemas without calling remote APIs.
Training uses the audited science3 subset (1199 valid train / 300 test); financial and all `_general` domains are excluded.
It does not certify API availability, credentials, data freshness or judge agreement.

## Configuration and services

Use `agent_system/environments/configs/dive_tool.yaml` as bootstrap transport.
Each dataset task supplies its own original ordered function schemas; `dive_session` is not a policy action.
The pool uses 0.25 CPU per worker and four workers by default, configurable through `num_cpus_per_worker` and `pool_size` or `DIVE_ENV_POOL_SIZE`.
Set `session_config` for `max_steps`, `tool_timeout_s`, `episode_timeout_s`, `sandbox_url` and `judge_config`.

Configure the judge explicitly with `DIVE_JUDGE_PROVIDER`, `DIVE_JUDGE_MODEL` and
`DIVE_JUDGE_BASE_URL`, or supply `judge_config` in the session configuration.
For an OpenAI-compatible endpoint:

```bash
export DIVE_JUDGE_PROVIDER=openai_compatible
export DIVE_JUDGE_MODEL=your-judge-model
export DIVE_JUDGE_BASE_URL=https://model-api.example.com/v1
export DIVE_JUDGE_API_KEY_ENV=DIVE_JUDGE_API_KEY
```

Set the referenced credential variable in the worker environment. Configuration stores
credential references rather than secret values. Explicit judge settings take precedence
over shared model settings, and checkpoint restoration preserves the saved judge identity.

The judge reuses the upstream verification prompt and label parser with bounded requests.
`DiveEnv` injects a worker-local client from [`model_api/`](model_api/README.md); the judge
owns prompting and scoring, while provider adapters own request transport and authentication.
Session close releases clients owned by the environment.
Only `correct` receives reward 1; `partial` and `incorrect` receive 0.
Unknown labels or unavailable judges invalidate the attempt.

Browse uses `BROWSE_LLM_PROVIDER`, `BROWSE_LLM_MODEL` and `BROWSE_LLM_BASE_URL` for its
model transport. These environment-side models are independent of policy generation.
The adapter preserves the upstream prompt, webpage truncation and tool deadline.

Tool services depend on the task's selected tools:

- Financial tools require appropriate Tushare credentials and API permissions.
- Search uses `SERPER_API_KEY` or `JINA_API_KEY` according to the published arguments.
- Browse may use `JINA_API_KEY` and `BROWSE_LLM_API_KEY`, `BROWSE_LLM_BASE_URL`, `BROWSE_LLM_MODEL`.
- Optional NCBI and Semantic Scholar authentication uses `NCBI_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY`.
- Code execution requires an explicit, separately provisioned `SANDBOX_FUSION_URL`.

Actors never auto-start Docker or SandboxFusion.
Sandbox calls translate `cell_source` and clamp `timeout_seconds` to the tool budget, preserving upstream stateless notebook execution rather than replaying previous cells.
The notebook client adapts the pinned server's actual `cells` response, validates driver completion and exact cell counts, and retains structured display/error output with string clipping.
Missing outputs or failed drivers invalidate the attempt; Python cell errors remain ordinary tool feedback, not infrastructure failures.
Run untrusted code only in an appropriately secured external sandbox with its own resource and network policies.
A killed Ray actor does not roll back remote requests that have already reached a service.

## Protocol and lifecycle

`reset(task=...)` accepts official `trace_id`, `query`, private `answer`, ordered `tools` and optional `metadata`.
`initial_messages` can be supplied separately from the same algorithm-neutral dataset prompt.
The default is a user message with the shared GiGPO-style task, observation, history and reasoning instructions.
Available actions contain the complete public tool schemas, also passed to the native chat tool interface.
Responses require `<think>...</think>` followed directly by native tool calls or the final answer, without an outer `<action>` block (`environment_react_v4`).
The native tool template instructs the model to reply with bare calls, and an outer block contradicted it.
The upstream solver starts with the query and tool schemas; the decision wrapper is local.
Reset validates schemas and imports the selected tools.
Public context contains only initial messages, raw assistant history, tool results and a schema hash.
The reference answer is never added to public context.

`step` takes `{content, raw_text, tool_calls}`.
Calls accept native OpenAI `{id, function: {name, arguments}}` objects or flattened name/arguments objects.
Arguments are checked against the published task schema before compatibility mapping.
All calls in a turn are validated before any tool executes, then execute sequentially in original order.
A nonempty natural-language assistant response with no calls is the final answer; there is no synthetic `respond` tool.
`content` is the extracted final answer while `raw_text` preserves generated text for append-only auditing.
Malformed actions generate public format feedback without side effects.
Reaching the configured step budget without an answer is a valid zero-reward attempt and does not call the judge.

`finalize` is idempotent and returns a terminal `result` and `episode_result` with `metric_valid`, `official_scored`, `official_reward`, counters and judge provenance.
Only finalization supplies the training reward; `execute`'s per-call reward is transport-only zero.
A normal `close` clears private task state and returns the actor to the pool.
On interruption, timeout or execution infrastructure error, the local tool aborts the lease and the pool kills the actor.
Lost capacity fails subsequent leases rather than silently reducing a GRPO group.

Synchronous tool and judge calls use Linux main-thread signal deadlines that upstream broad `except Exception` handlers cannot swallow.
RPC watchdogs separately bound reset, step, finalize and close, with defaults of 120, 1800, 150 and 15 seconds.
These are configurable as `<method>_timeout_s` or `DIVE_<METHOD>_TIMEOUT_S`.
The episode deadline is checked between calls; an in-flight call remains bounded by its tool and RPC deadlines.
Upstream explicit infrastructure error envelopes and swallowed web-service error diagnostics are promoted to invalid attempts, with diagnostic output suppressed.
Heterogeneous external APIs still require service-specific operational validation before a full benchmark run.

## Compatibility

- All 17 published `semantic_scholar_*` names map to their unprefixed registered implementation IDs.
- Published `jupyter_execute_code_cell` maps to the code-execution service boundary.
- Four exact-hash protparam sources have a trailing dangling decorator removed only in memory.
- The exact-hash `bio_sequtils` package initializer is replaced by a namespace in memory because it imports two nonexistent modules; its concrete registered tool module remains upstream.
- `stock_stk_surv.fields` is forwarded unchanged; upstream controls its handling.
- Published `rxnorm_get_display_terms.rxcui` remains schema-required although upstream ignores it and queries all display names.

Unknown source revisions or compatibility hashes fail instead of applying speculative repairs.
Runtime source validation is cached per actor process; treat the deployed checkout as immutable while actors run.

## Deployment checks

```bash
.venvs/dive/bin/python -m ops.env_deps.dive.verify
```

This imports the pinned SDKs, verifies the packaged source hashes and imports the
200 tools in the selected academic, biological and medical domains. It does not
create mock model requests, acquire credentials or call remote services.
Development tests and interaction smoke runs are maintained separately from the
runtime package. Passing this deployment check does not establish service access
or complete benchmark execution.
