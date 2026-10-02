# SWE-bench Verified offline evaluation

SWE-bench Verified provides 500 official test tasks and their grading procedure; it does not define an agent tool interface.
This project uses it only for cross-benchmark evaluation and provides no SWE training entrypoint.
The tool interface and execution budget are part of this project's fixed agent scaffold. Project scores are not an exact reproduction of the DIVE paper.

## Separate preparation from execution

Preparation may access the network; formal evaluation must have no public network access.
Install the pinned harness with `bash ops/env_deps/swebench/install.sh`. Its isolated environment defaults to `.venvs/swebench`.
The isolated environment installs exactly the policy Python's Ray version for shared environment workers, without importing torch or training dependencies.
Build an evaluation image with `ops/build_env_eval.sh`. It adds `/opt/venv-swebench` to the pinned capability image and checks the harness commit and Ray version, without publishing images or submitting jobs. The `grader-runtime` target in `Dockerfile.swebench` does not enforce global offline mode or disable W&B; the default `offline` target retains those settings. A Python installation alone is insufficient: prepare Docker, cached task images, assets and worker network isolation separately.
All 73 installed harness source files are checked individually; a matching version number alone does not establish commit identity.
The pinned v4.1.0 upstream source is in `agent_system/environments/env_package/swebench/source`. Compatibility code lives outside that source directory.

```bash
.venvs/swebench/bin/python agent_system/environments/env_package/swebench/assets.py prepare \
  --asset-dir data/swebench_verified/assets --download
```

This command downloads only the pinned dataset revision, not the hundreds of task images.
Image preparation is a separate, explicit network operation. Afterwards, pin local image IDs and registry digests with `register-images --asset-dir ...`.
`verify --asset-dir ...` checks local data and images only. It does not download, rebuild or silently replace versions.
Having all 500 task records does not mean that all 500 task images are available.

Use a separate asset directory for a fixed validation subset. Keep formal data unchanged and do not select tasks by model performance.

```bash
.venvs/swebench/bin/python agent_system/environments/env_package/swebench/assets.py prepare \
  --asset-dir data/swebench_verified/smoke_assets \
  --source-file data/swebench_verified/assets/source.parquet \
  --instance-ids psf__requests-1142 pallets__flask-5014
# Run only during the explicitly network-enabled preparation stage.
docker pull swebench/sweb.eval.x86_64.psf_1776_requests-1142:latest
docker pull swebench/sweb.eval.x86_64.pallets_1776_flask-5014:latest
.venvs/swebench/bin/python agent_system/environments/env_package/swebench/assets.py register-images \
  --asset-dir data/swebench_verified/smoke_assets
```

Registry tags are used only during preparation; execution uses registered immutable image IDs.
The harness's generic `make_test_spec` also generates build scripts and fetches requirements, so it cannot be called directly during offline execution even with cached images.
This adapter constructs evaluation TestSpecs through the pinned harness's official functions without generating unused build scripts. It preserves official evaluation commands and report parsing.
Official images do not guarantee that every pre-grading installation step works offline.
For example, Flask's official `pip install -e .` attempts to download build dependencies.
`ops/Dockerfile.swebench-flask-offline` caches the image's existing setuptools/wheel versions in a local wheelhouse while retaining the official installation and test commands.
The grader archives the original official script and creates an execution script with `errexit` enabled outside the test phase. This prevents setup failures from being counted as unsolved tasks. Tests may exit nonzero; the official report still determines the grade.
Both scripts are archived. The execution difference is recorded as `setup-errexit-v1`; the scripts are not claimed to be byte-identical.
Register derived images with their preparation identity and upstream digest; do not overwrite official tags or represent derived images as unmodified.
Tool-visible inputs contain only allowlisted fields such as issue, repo, base_commit and instance_id.
Reference fixes, hidden test patches and grading criteria are visible only to the isolated grader.

## Strict offline boundary

`offline.py check` verifies that the current Linux network namespace has only loopback. `HF_HUB_OFFLINE=1` alone does not prove network isolation.
The local Docker driver container uses `--network=none`; tool and grading containers must each disable networking too.
The driver needs the local Docker Unix socket. Task containers must not mount that socket, host files, credentials or grading data.
The model and Ray communicate locally inside the driver container; no public model API is required.
Formal execution does not pull images, download weights or upload to W&B.

Add the isolated grading dependencies to an existing Dyad image during network-enabled preparation.
Replace `BASE_IMAGE` with an actual locally prepared image; runtime does not build it automatically.

```bash
docker build --build-arg BASE_IMAGE=<existing-dyad-image> \
  -f ops/Dockerfile.swebench -t dynamic-dyad-swebench:local ops
```

Run `offline.py launch --help` for the shared evaluation wrapper's arguments.
Provide an existing dedicated `--artifact-root`. Mount external model caches and checkpoints explicitly with `--mount-readonly`.
Source is mounted read-only and the artifact root is writable. Prepare resources beforehand; do not execute Python from a mounted host virtual environment.
`--dry-run` checks the local image and displays the command without starting evaluation.
The example below exercises one task through the public text-baseline entrypoint. The model path must contain a complete cached model; runtime downloads are not supported.
`smoke_offline_assets` is a separately registered asset directory with the cached Flask image. The original uncached `smoke_assets` directory is insufficient for offline execution.

```bash
.venvs/expa-verl/bin/python agent_system/environments/env_package/swebench/offline.py launch \
  --image dynamic-dyad-swebench:local \
  --artifact-root "${PWD}/artifacts" \
  --mount-readonly /absolute/path/to/model-cache \
  --gpus device=3 \
  --env RAY_NUM_CPUS=16 --env N_GPUS_PER_NODE=1 --env ROLLOUT_TP_SIZE=1 \
  --algorithm grpo_react -- \
  --model Qwen3.5-2B --model-path /absolute/path/to/model-cache/model-snapshot \
  --assets "${PWD}/data/swebench_verified/smoke_offline_assets" \
  --debug --task-ids pallets__flask-5014 \
  --max-steps 4 --episode-timeout 300 --grading-timeout 120
```

Append `--preflight` to check local models, tasks, grading dependencies and images without generation.
For full Dyad evaluation, use `--algorithm dyad-gigpo` or `dyad-grpo`, replace `--model-path` with `--checkpoint /absolute/path/to/global_step_N`, preserve the source model configuration and allocate a separate encoder GPU.
Responses use `<think>...</think>` followed directly by native tool calls, including finish, without an outer `<action>` block (`environment_react_v4`).
Qwen3.5's native tool template requires bare calls; an outer action block conflicts with parsing.
Native routing enters through the stable `<tool_call>` special token, emits the fixed protocol prefix, then uses the action head to select a tool. The LM still generates arguments.
This prevents BPE merging of Qwen XML's `<function=` prefix with tool names from bypassing the action head. Baseline and Dyad keep the same prompt.
For example, `--gpus device=2,3` exposes two GPUs. `N_GPUS_PER_NODE=1` counts only the policy GPU, excluding the encoder. Check resource availability before execution.
The example retains the default 32768 context / 24576 prompt / 4096 generation budgets, with four steps and short timeouts for interaction validation. It is not a formal full-budget or capacity run.
Each step uses the shared [local prompt](../../prompts/README.md#swe-bench-verified), populated with the public task, latest tool feedback and observation-action history.
`--history-length` controls the number of recent decisions. Zero still retains the task, latest feedback and absolute step number; full execution traces are stored separately.
Qwen3.5 native thinking stays disabled; task-level `<think>...</think>` reasoning is followed by native tool calls.
If the rendered prompt exceeds its budget, the current workspace is submitted with `context_budget_exhausted`. Inputs are not truncated further and executed traces are retained.

The evaluation entrypoint is `experiments/shared/train_eval/evaluate.sh swebench_verified`.
It installs no dependencies or data and fails if `/opt/venv-swebench`, isolated networking or a working local Docker service is missing.
SWE cluster jobs are not generated or submitted automatically. Validate network isolation and container execution on the target cluster first.

## Scoring protocol

The official source contains 500 test tasks. The public environment suite uses the fixed 50-task selection defined in [the shared protocol](../../../../experiments/shared/evaluation_protocol.py), with one independent attempt and resolved percentage over all 50 tasks.
Report the number of trials for repeated evaluations; mean success rate is not pass@k.
Other small debug subsets validate functionality and must not populate the formal benchmark table.
Record official test failures separately from infrastructure failures. Missing, duplicated, ungraded or unverified-offline runs do not constitute a complete score.
A grading container's gold-patch check validates the grader; it is not a model score.

## Using full mini-swe-agent as a tool

Append `--mini-swe-config /absolute/path/mini-swe.yaml` to let baseline and Dyad
share two tools: `mini_swe_agent(instruction)` and `finish`. Each delegation runs
the complete `DefaultAgent` loop from `mini-swe-agent==2.4.6`. A separately configured
submodel generates Bash calls, edits the task workspace and executes local tests.
The outer model can inspect the returned patch, delegate again, and call `finish` for official grading.
Without this option, the original four-tool protocol remains. The protocols have distinct scaffold versions and schema hashes.

The configuration contains only non-secret fields; replace the model name with the actual service name:

```yaml
model: your-served-model
base_url: http://127.0.0.1:8001/v1
# Set the credential environment-variable name only when authentication is required.
# api_key_env: MINI_SWE_API_KEY
max_steps: 30
max_tokens: 4096
timeout_s: 600
request_timeout_s: 120
temperature: 0
```

The submodel must support `bash` tool calls through OpenAI-compatible chat completions.
Its service must share the offline driver's network namespace and be reachable over loopback.
A host service on `127.0.0.1` is not automatically available inside the offline container.
Ray policy workers do not automatically serve the subagent API. The adapter rejects public addresses, does not start or download the submodel, and preserves SWE network isolation.
The submodel, subagent step limit and token budget are separate evaluation conditions and must match between baseline and Dyad.

During network-enabled preparation, rerun `ops/env_deps/swebench/install.sh` or build
with the updated `ops/Dockerfile.swebench` to install the pinned version. Older images
lack this dependency. Preflight checks the version without installing it during evaluation.
`--check` validates configuration and installation; service reachability and tool-call support require an actual run.

Full subagent traces, patches and call summaries are stored under
`mini_swe_agent/<episode_id>/call_NNNN/` in the run directory, including model calls,
commands, input/output tokens, termination status and file hashes. The returned patch has a length limit; the full saved patch is retained.
A subagent's `Submitted` status means delegation completed, not that the task was resolved.
Step and time exhaustion retain the current patch. API/container failures fail the run rather than count as official zero scores. Official grading reads the final workspace only.
Costs are not estimated. Submodel token usage and outer-model traces are recorded separately.

Configure the submodel service and check dependencies and grading assets through the selected entrypoint before use.
