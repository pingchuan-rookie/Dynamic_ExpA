# t2bench evaluation for Agentic RL ReAct and Dyad

This adapter pins bundled official source commit `a2c024725189473d2d7cea3a5cfdbcc67478e41f`, which includes tau3 updates. Its results are not original tau2-paper scores.
The unified entrypoint generates policy actions through the shared inference API. `session.py` drives the official Orchestrator in a Ray actor using an isolated environment interpreter.
User simulation and required judges use independent services. There is no tau training or modification of official source/data.
See [Agentic RL evaluation](../../README.md) for default entrypoints, resources and artifacts.

Direct `evaluate.py` commands and standalone endpoints below describe the reference evaluator.
Select `--backend reference` explicitly when using these features through the unified entrypoint; there is no automatic fallback.
The reference path sends `grpo_react` to a text endpoint and `dyad` to a separately started native Dyad GPU service.
Official user simulation, tools, initialization, orchestration, termination and `EvaluationType.ALL` grading remain intact.

## Entrypoints and tasks

Run from `dynamic-expa/` using the dedicated interpreter without activating a venv.

```bash
.venvs/t2bench/bin/python experiments/shared/train_eval/reference/t2bench/evaluate.py grpo_react --domain all --debug --check
.venvs/t2bench/bin/python experiments/shared/train_eval/reference/t2bench/evaluate.py --help
```

Unified entrypoint: `experiments/shared/train_eval/evaluate.sh t2bench <grpo_react|gigpo|dyad-grpo|dyad-gigpo>`; `dyad` remains a compatibility alias.
Docker installs pinned dependencies through `ops/env_deps/t2bench/`; container startup rechecks official versions, tasks and tools.
Host validation does not establish Docker or cluster readiness.

| Domain | Official split | Full count | Default debug IDs |
|---|---|---:|---|
| retail | base | 114 | `0`, `1`, `2` |
| airline | base | 50 | `0`, `1`, `2` |
| telecom | base | 114 | First three complete string IDs in official base; see `--check` |

`--domain all` selects all three domains; `both` is a compatibility alias with the same behavior.
Telecom `full` (2285 tasks), mock, banking and telecom-workflow are unsupported.
Some tasks descend from old tau-bench; their counts cannot be summed as independent tasks.
`--check` loads official tasks/environments without networking or output directories, checking commit, counts, unique IDs, schemas, selection and output paths.
Offline checks need no service credentials and distinguish full counts, selected IDs, planned counts and zero completed tasks.

`--task-ids` accepts exact strings, not indices. Quote Telecom IDs.
`--limit` and `--task-ids` are mutually exclusive; `--shuffle --seed` selects deterministic subsets.
`--debug` permits at most three tasks per domain and one trial per task; conflicting settings fail rather than expanding the run.
Remove `--debug` for formal evaluation. Without selection arguments, evaluate the complete base split.
Execution order is domain, trial, selected task. Only `--max-concurrency 1` is supported.

## Independent services and weights

Agent, user and judge each accept `--ROLE-model`, `--ROLE-provider`, `--ROLE-base-url` and `--ROLE-api-key-env`.
`--model` also selects the agent model.
User/judge default to provider `trapi`; policy defaults to `openai`.
Explicit `--user-provider openai` / `--judge-provider openai` select local services. Other LiteLLM providers are supported only with non-Qwen3.5 models and `--thinking default`.
Qwen3.5 requires an OpenAI-compatible service with explicit no-thinking parameters.
Credentials come only from environment variables, never plaintext CLI keys; inherited environments and request headers are not saved.
Defaults are `T2BENCH_AGENT_API_KEY`, `T2BENCH_USER_API_KEY` and `T2BENCH_JUDGE_API_KEY`.
Except for `trapi`, non-loopback services require the corresponding static credential.
Set default model/provider/URL through `T2BENCH_ROLE_MODEL`, `T2BENCH_ROLE_PROVIDER` and `T2BENCH_ROLE_BASE_URL`.

```bash
# Deploy the three endpoints first; this command starts no services and uses no GPU.
ARGS=(grpo_react --domain all --debug
  --model Qwen3.5-2B --agent-base-url http://127.0.0.1:18000/v1
  --user-provider openai --user-model USER_MODEL --user-base-url http://127.0.0.1:18001/v1
  --judge-provider openai --judge-model JUDGE_MODEL --judge-base-url http://127.0.0.1:18002/v1)

.venvs/t2bench/bin/python experiments/shared/train_eval/reference/t2bench/evaluate.py "${ARGS[@]}" --check
.venvs/t2bench/bin/python experiments/shared/train_eval/reference/t2bench/evaluate.py "${ARGS[@]}" --preflight
.venvs/t2bench/bin/python experiments/shared/train_eval/reference/t2bench/evaluate.py "${ARGS[@]}"
```

Replace endpoints and user/judge models with the actual deployment. The evaluated model is not automatically reused as the simulator.
Default retail debug task `2` needs an NL judge, so `all --debug` requires independent judge configuration.
Judge services are checked only when selected tasks include `NL_ASSERTION` in the official reward basis with nonempty assertions.
Airline existence assertions are not automatically added to the reward basis; official grading scope is preserved.
Because the official NL evaluator exposes module-level defaults, judge configuration temporarily replaces them within a scope and restores them afterwards, retaining prompts and scoring.
This scoped state is why parallel episodes are currently disabled.

`--preflight` checks agent/user/required-judge services and makes small requests, without running benchmark tasks.
OpenAI-compatible services must list the matching model in `/models`; returned `max_model_len` is also checked against agent `--context-length`.
Preflight failure is not task failure or completed evaluation. Summary retains planned and missing counts.

The HF-weight and model-merger instructions below apply only to `grpo_react`.
Use `--model-path /absolute/HF/path` or `--checkpoint /absolute/exported_HF/path` to record weights and validate config, tokenizer, nonempty files and indexed shards.
Weight paths are independent of service model IDs; evaluation does not start or switch models.
Local provenance must match the loopback OpenAI-compatible `/models` `root`. Local files cannot establish a remote service's identity.
Endpoint-only mode explicitly records unverified weight files.
`--checkpoint-source` can record historical `ExpA_verl/ckpt/.../global_step_N` provenance but does not load it.
Export raw verl FSDP shards or config/tokenizer-only `actor/huggingface` directories with the existing model merger before evaluation.
Clear ambiguous `MODEL_PATH` settings before using `--checkpoint`.
`--tokenizer` and `--chat-template` record service settings without changing them.

`--temperature`, `--max-tokens` and `--thinking` configure the agent; user/judge variants use the corresponding role prefix.
Agent thinking defaults off. For Qwen3.5 in any role, `default` also resolves to off and sends `enable_thinking=false`; explicit on is rejected before requests.
Other models retain explicit on/off/default choices; default omits the extension.
`--timeout` limits each request. `--max-retries` retries transport only, not entire tasks.
`--episode-timeout` is the Orchestrator's between-step wall-clock limit; active synchronous requests remain bounded by request timeout.
`--max-steps` counts official Orchestrator steps, including user and tool steps, not only agent generations.
`--history-length` retains recent observation-action pairs, default 2; zero keeps only the task and latest observation.
Full official traces remain for execution/grading. Context is not fitted by token truncation or hidden reference answers.
Cross-model comparisons must fix tasks, simulator, judge, seed, trials and generation settings.

## Environment model configuration

Configure user/judge provider, model and URL separately, selecting credentials with each role's `--ROLE-api-key-env`.
See [environment configuration](../../../../../agent_system/environments/env_package/t2bench/config.py)
and the environment package's provider code. Configuration checks make no model requests; `--preflight` checks live services.

## Native Dyad

`evaluate.sh t2bench dyad --backend reference` requires exact `--projector-init` plus `--model-config`, or native Agentic RL `--checkpoint /absolute/run/global_step_N`, not a plain HF export or text service.
Alignment initialization does not restore a trained policy. Native Agentic RL uses strict DTensor CPU restoration, preserves training provenance and rebuilds the derived action head from target schemas.
Omit `--model-path` and clear `MODEL_PATH`. Client/service source paths and `--context-length` must match.
See the [standalone server](../../../../../agent_system/policies/dyad/inference/server.py) for restoration and service arguments.

Dyad sends text messages and complete official tool schemas to `/v1/dyad/act`; this is not an OpenAI function-calling request.
Responses must include valid expanded selections, provenance/schema fingerprints, target-head/projector fingerprints and `action_content`. The client validates these before text parsing for the official Orchestrator.
User simulation, Telecom user tools, task-selected NL judges and official scoring remain unchanged. Raw text and Dyad traces are stored in call records and message `raw_data`.
Dyad agent preflight checks restoration identity at `/v1/dyad/health`, not tool generation or task completion. User/required-judge checks remain separate.
CPU restoration, GPU generation, real external tasks and Docker/cluster validation are distinct scopes.

## Text protocol boundary

The `grpo_react` endpoint receives only system/user/assistant text messages, without tools, tool_choice, functions, function_call, tool-role messages or message-level tool_calls.
Tool schemas and official policy are rendered as system-prompt text.
Local prompts use GiGPO-style task, current-observation and limited-history templates, producing `<think>...</think>` followed by `<action>JSON</action>`.
The task comes from public customer messages, not private simulator instructions. Official policy/user/judge prompts remain unchanged.
Metadata records `task_prompt_protocol=environment_react_v3` and `text_protocol=tau_bench_action_v4` to distinguish earlier prompt runs.
The name records protocol provenance; it does not imply support for the removed tbench implementation.

```text
<think>Read the order before proceeding.</think>
<action>{"name":"get_order_details","arguments":{"order_id":"#W2378156"}}</action>
```

Parsing requires one closed think block and one closed action block containing one complete JSON object, with valid message structure, unique keys and finite numbers. These are adapter transport constraints.
Reject legacy Thought/Action syntax, uppercase tags, code fences and missing tags. Do not repair braces or ignore extra JSON. Unknown tools and invalid/missing/extra arguments go unchanged to the official executor for its error feedback and counts.
Malformed JSON or parsing failures end the attempt without fabricated customer replies, format-error retries or function-calling substitutes. Such failures record `official_scored=false`, `official_reward=null`. Missing/mismatched Dyad selection evidence is an identity failure, not a valid zero score.
Extract action content while retaining the full decision. If LiteLLM separates `reasoning_content`, reconstruct think only from that real field and retain both raw transport fields.
Valid tool actions become official `AssistantMessage(content=None, tool_calls=[ToolCall(...)])` objects for the Orchestrator.
Private thought is excluded from customer messages and communication grading.
`respond` is the adapter's text-sending action, not an official tool.

```text
<think>Ask for confirmation.</think>
<action>{"name":"respond","arguments":{"content":"Please confirm whether to exchange these two items."}}</action>
```

Only `content` reaches the official user simulator.
Raw tool content returns with the tau-bench `API output: ` prefix; call IDs and error flags remain in official traces and tool_events.
Keep initial failed XML runs. Rerunning the same task with a new protocol is a second attempt, not another independent task.
Source edits do not update already loaded processes or rewrite existing artifacts.
Telecom users retain the official user-tool mechanism; those tools are not transferred to the agent.

## Outputs, metrics and evidence

Default output: absolute `artifacts/outputs/local/agentic_rl/<algorithm>/t2bench/<unique_run>/`, with algorithm `grpo_react` or `dyad`.
`--output-dir` / `--save-to` require new absolute directories within the deployed outputs root; relative, existing or escaping paths/symlinks are rejected. Official `data/simulations/` is untouched.
The adapter uses official runner components and orchestration directly, avoiding the upstream batch CLI's default output behavior.

- `resolved_config.json`: actual command, redacted config, weight provenance, package versions, commit/dirty state, diff/task fingerprints and full/selected IDs.
- `endpoint_preflight.json`: independent service identities and validation status; failures produce `preflight_error.json`.
- `summary.json`: planned/missing counts for every domain from startup, atomically updated after each task.
- `run.log`: domain, official string ID, trial, status, reward and absolute trace path.
- `<domain>/task_<index>_<id_hash>_trial_<trial>.json`: raw agent requests/responses, official simulation, user messages, tools and state-hash changes.
- `<domain>/results.json` and `<domain>/summary.json`: domain results and progress.

Full task IDs live inside JSON, avoiding Telecom composite IDs and path traversal in filenames.
State-change observation wraps only live-environment `get_response`; official grading uses a separate replay environment and reference replay is not counted as agent state change.
Record tool errors, timeout, parse_error, service_error, task_failure, incomplete and internal_error separately.
Normal termination with zero official reward is task_failure. Step/tool-error limits use official early-termination zero scoring; agent parse failures are failed attempts with `official_scored=false`.
Service, judge or internal errors have null reward and suppress overall domain success rate; diagnostic rates report a valid-attempt denominator separately.
`completed_episodes` counts normal terminations; `official_scored_episodes` counts officially graded episodes. Neither is replaced by full split size.
Report mean reward, mean success rate and `pass^1` only, even with multiple trials; do not generate `pass^k` for k > 1.

Source/data changes reject execution. Non-runtime changes such as lockfiles record dirty state and diff SHA256 without automatically editing or restoring files.
Treat task contents and conversations as experiment data; never publish credentials.

## CPU regressions

```bash
PYTHONDONTWRITEBYTECODE=1 .venvs/t2bench/bin/python -m pytest agent_system/environments/env_package/t2bench/tests -q -p no:cacheprovider
```

Tests use real official tasks, tools, user simulation, orchestration and scoring; HTTP stubs control only model responses.
Coverage includes all three domains, observation feedback, retail writes, Telecom user-state changes, independent judges, JSON rejection, error classification, offline operation, debug limits and CLI outputs.
These checks validate interfaces and tools, not Qwen3.5-2B performance or Docker/cluster readiness.
