#!/usr/bin/env bash
# Assemble the existing env/algo-specific verl arguments and launch training.
# evaluation.sh sources the argument builders only; it never calls the training entrypoint.
set -euo pipefail
NODE_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${NODE_SCRIPTS}/../../../.." && pwd)"

ray_effective_cpus() { printf '%s\n' "${EFFECTIVE_CPUS:?prepare.py must resolve CPU resources}"; }
dyad_variant_require() {
    local name="$2"
    [ -n "${!name:-}" ] || { echo "[config] Missing ${name} for $1" >&2; exit 2; }
}

append_resume_args() {
    if [ "${RESUME_MODE:-disable}" = resume_path ]; then
        OPTIONAL_HYDRA_ARGS+=("trainer.resume_from_path=${RESUME_CKPT:?checkpoint required}")
    fi
}

emit_command() {
    local module="$1"; shift
    local estimator="${RUN_ADV_ESTIMATOR:?prepare.py must resolve the advantage estimator}" i
    for i in "${!HYDRA_ARGS[@]}"; do
        case "${HYDRA_ARGS[$i]}" in
            algorithm.adv_estimator=*) HYDRA_ARGS[$i]="algorithm.adv_estimator=${estimator}" ;;
        esac
    done
    COMMAND=("${PYTHON_BIN:?}" -m "${module}" "${HYDRA_ARGS[@]}" "${OPTIONAL_HYDRA_ARGS[@]}")
    if [ "$estimator" = gigpo ]; then
        COMMAND+=("algorithm.gamma=${GIGPO_GAMMA:?prepare.py must resolve GiGPO defaults}"
            "algorithm.gigpo.mode=${GIGPO_MODE:?}"
            "algorithm.gigpo.step_advantage_w=${GIGPO_STEP_ADVANTAGE_W:?}")
    fi
    if [ "${STEP_ROLLOUT_ENABLED:-False}" = True ]; then
        COMMAND+=("algorithm.step_rollout.enabled=True"
            "algorithm.step_rollout.protocol_version=2"
            "algorithm.step_rollout.profile=${STEP_PROFILE:?}"
            "algorithm.step_rollout.action_interface=${STEP_ACTION_INTERFACE:?}"
            "algorithm.step_rollout.history_length=${STEP_HISTORY_LENGTH:?}"
            "algorithm.step_rollout.max_steps=${MAX_ASSISTANT_TURNS:?}"
            "algorithm.step_rollout.invalid_action_penalty=${STEP_INVALID_ACTION_PENALTY:?}"
            "algorithm.step_rollout.resampling=reference_copy"
            "algorithm.step_rollout.loss_reduction=reference_microbatch"
            "algorithm.step_rollout.compute_mean_std_cross_steps=True"
            "trainer.use_v1=True" "trainer.v1.trainer_mode=sync"
            "trainer.v1.sync.parameter_sync_step=1"
            "actor_rollout_ref.actor.ppo_mini_batch_size_unit=step"
            "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF:?}"
            "actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO:?}"
            "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO}"
            "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO}"
            "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE:?}"
            "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:?}"
            "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:?}"
            "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:?}"
            "data.tool_config_path=null" "data.function_tool_path=null"
            "data.filter_overlong_prompts=False")
    fi
    if [ "${RUN_IS_EVAL:-0}" = 1 ]; then
        COMMAND+=("actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-1}"
            "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.6}"
            "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-1.0}"
            "data.val_max_samples=${VAL_MAX_SAMPLES:--1}")
        [ -z "${VAL_BATCH_SIZE:-}" ] || COMMAND+=("data.val_batch_size=${VAL_BATCH_SIZE}")
    fi
    if [ "${benchmark:-}" = webshop ]; then
        COMMAND+=("actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TOOL_TURNS:?}")
    fi
    COMMAND+=("$@")
    if [ "${RUN_IS_EVAL:-0}" != 1 ]; then
        COMMAND+=("trainer.val_before_train=${VAL_BEFORE_TRAIN:?prepare.py must require initial training validation}")
    fi
    # Calculator evaluation uses fresh decisions without changing checkpoint loaders.
    local calculator_eval=0
    if [ "${benchmark:-}" = gsm8k ] && [ "${RUN_IS_EVAL:-0}" = 1 ]; then
        calculator_eval=1
        local interface=text
        [ "$module" != agent_system.policies.dyad.training.main_dyad ] || interface=dyad
        COMMAND+=("algorithm.step_rollout.enabled=False" "++algorithm.step_rollout.evaluation_only=True"
            "algorithm.step_rollout.protocol_version=2" "algorithm.step_rollout.profile=gsm8k_native_v2"
            "algorithm.step_rollout.action_interface=${interface}"
            "algorithm.step_rollout.history_length=${STEP_HISTORY_LENGTH:-2}"
            "algorithm.step_rollout.max_steps=${MAX_ASSISTANT_TURNS:?}"
            "algorithm.step_rollout.invalid_action_penalty=0.0"
            "data.filter_overlong_prompts=False" "data.tool_config_path=null" "data.function_tool_path=null")
    fi
    # Routing belongs to the run, never to the shared dataset.
    if [ "${STEP_ROLLOUT_ENABLED:-False}" != True ] && [ "$calculator_eval" != 1 ]; then
        echo "[config] Old interaction protocols were removed; use shared step v2" >&2
        exit 2
    fi
    COMMAND+=("trainer.use_v1=True" "trainer.v1.trainer_mode=sync"
        "actor_rollout_ref.rollout.agent.default_agent_loop=environment_step_agent"
        "actor_rollout_ref.rollout.agent.agent_loop_config_path=${PROJECT_DIR}/agent_system/rollout/environment_step_agent_loop.yaml")
    if [ "${benchmark:-}" = webshop ] || [ "${benchmark:-}" = dive ]; then
        COMMAND+=("data.filter_overlong_prompts=False" "data.truncation=error")
    fi
    if [ "${benchmark:-}" = codegym ] || [ "${DYAD_CODEGYM_ALL:-0}" = 1 ]; then
        COMMAND+=("data.filter_overlong_prompts=False" "data.truncation=error")
    fi
    if [ "${RUN_IS_EVAL:-0}" = 1 ]; then
        # Only mode/weight-safety invariants override the caller's sampling choices.
        COMMAND+=("trainer.val_only=True" "trainer.val_before_train=True"
            "trainer.save_freq=-1" "trainer.del_local_ckpt_after_load=False"
            "actor_rollout_ref.actor.checkpoint.load_contents=[model]"
            "trainer.validation_data_dir=${VALIDATION_DATA_DIR:-${RUN_DIR}/val_generations}")
    fi
    local variable
    # write-command enforces the Qwen3.5 no-thinking leaf after every caller
    # override, using both source identity and the resolved local model config.
    for variable in MODEL_NAME MODEL_PATH DYAD_MODEL_SOURCE_PATH MODEL_TAG ENV_NAME CODE_ID DYAD_ACTION_YAML TEMPLATE_STYLE; do
        [ -z "${!variable:-}" ] || export "${variable}"
    done
    "${PYTHON_BIN}" "${NODE_SCRIPTS}/prepare.py" write-command "${COMMAND[@]}"
}

# Shared text-environment builders. Environment wrappers select only their bindings.
build_alfworld_dyad() ( build_text_environment_dyad alfworld "$@"; )
build_webshop_dyad() ( build_text_environment_dyad webshop "$@"; )
build_alfworld_grpo_react() ( build_text_environment_grpo_react alfworld "$@"; )
build_webshop_grpo_react() ( build_text_environment_grpo_react webshop "$@"; )

configure_text_environment() {
    local env_name="$1" pool_key
    export REACT_TOOL_NAME="${REACT_TOOL_NAME:-${env_name}_action}"
    [ "$env_name" != webshop ] || export REACT_TOOL_NAME=webshop_action
    # WebShop bounds full-catalog replicas independently of concurrent session leases.
    [ "$env_name" != webshop ] || return 0
    # ALFWorld reserves the actual per-worker train/validation demand at dispatch.
    # Keep explicit user overrides, but never duplicate the whole batch per worker.
    [ "$env_name" != alfworld ] || return 0
    pool_key="${env_name^^}_ENV_POOL_SIZE"
    export "${pool_key}=${!pool_key:-$(( TRAIN_BATCH_SIZE * ROLLOUT_N ))}"
}

build_text_environment_dyad() (
text_environment="$1"; shift
configure_text_environment "$text_environment"
cd "${PROJECT_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venvs/expa-verl/bin/python}"

export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=0

export DYAD_ENCODER_ENABLED="${DYAD_ENCODER_ENABLED:-1}"
export DYAD_ENCODER_SCALE="${DYAD_ENCODER_SCALE:-unit}"
export DYAD_ENCODER_GATE_INIT="${DYAD_ENCODER_GATE_INIT:-0.0}"
export DYAD_PROJECTOR_LR="${DYAD_PROJECTOR_LR:-1e-4}"
export DYAD_ENCODER_PROJECTOR_INIT="${DYAD_ENCODER_PROJECTOR_INIT:-}"
export DYAD_ENCODER_MODEL_PATH="${DYAD_ENCODER_MODEL_PATH:-}"  # Empty selects the policy model specification with independent weights.
export DYAD_ENCODER_GPU_IDS="${DYAD_ENCODER_GPU_IDS:-}"
export DYAD_ENCODER_REMOTE="${DYAD_ENCODER_REMOTE:-1}"
export DYAD_ENCODER_ACTOR_NAME="${DYAD_ENCODER_ACTOR_NAME:-dyad_action_encoder}"
export DYAD_ENCODER_MAX_LENGTH="${DYAD_ENCODER_MAX_LENGTH:-1024}"
export DYAD_ENCODER_BACKBONE="${DYAD_ENCODER_BACKBONE:-encoder_lm}"
export DYAD_ENCODER_PROJECTOR="${DYAD_ENCODER_PROJECTOR:-attention}"
export DYAD_ENCODER_REPRESENTATION="${DYAD_ENCODER_REPRESENTATION:-final_layer_hidden_states}"
export DYAD_ENCODER_DESCRIPTION="${DYAD_ENCODER_DESCRIPTION:-mcp}"
export DYAD_TRAINING_SCHEDULE="${DYAD_TRAINING_SCHEDULE:-frozen_llm_adaptation}"
export DYAD_ENCODER_TRAINING="${DYAD_ENCODER_TRAINING:-projector_and_encoder_lm}"
export RAY_DEBUG_POST_MORTEM=1

TIME_SUFFIX="${TIME_SUFFIX:-$(date +"%Y%m%d_%H%M%S")}"

CKPT_ROOT="${CKPT_ROOT:-${PROJECT_DIR}/ckpt/local/agentic_rl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/local/agentic_rl}"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-0.5B-Instruct}"
HF_HUB_DIR="${HF_HUB_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH="$(ls -d "${HF_HUB_DIR}"/models--*--"${MODEL_NAME##*/}"/snapshots/*/ 2>/dev/null | head -1 || true)"
    MODEL_PATH="${MODEL_PATH%/}"
fi
[ -n "${MODEL_PATH:-}" ] || { echo "[dyad_alfworld] 无法解析模型路径 (MODEL_NAME=${MODEL_NAME}, HF_HUB_DIR=${HF_HUB_DIR})；请设 MODEL_PATH 或 MODEL_NAME"; exit 1; }
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL_NAME##*/}" | tr 'A-Z' 'a-z')}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${MODEL_TAG}}"
EXP_NAME="${EXP_NAME:-dyad_${TIME_SUFFIX}}"

model_path="${MODEL_PATH}"

TOOL_PARSER_FORMAT="dyad"
dyad_variant_require "$text_environment" DYAD_ACTION_YAML
export DYAD_ACTION_YAML

train_data="${TRAIN_DATA:?[dyad] TRAIN_DATA 未设置；走 experiments/shared/train_eval/train.sh alfworld}"
test_data="${TEST_DATA:?[dyad] TEST_DATA 未设置；走 experiments/shared/train_eval/train.sh alfworld}"

yaml_path="${TOOL_CONFIG_PATH:-${PROJECT_DIR}/agent_system/environments/configs/${text_environment}_tool.yaml}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))}"

if [ "${DYAD_ENCODER_ENABLED:-0}" = "1" ] && [ "${DYAD_ENCODER_REMOTE:-0}" = "1" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        _visible_gpus="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' ' ' | wc -w)"
    else
        _visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
    fi
    _need=$(( N_GPUS_PER_NODE + ${DYAD_ENCODER_NUM_GPUS:-1} ))
    if [ "${_visible_gpus}" -lt "${_need}" ]; then
        echo "[dyad] DYAD_ENCODER_REMOTE=1 需要 ${_need} 张可见卡" >&2
        echo "       trainer ${N_GPUS_PER_NODE} + encoder ${DYAD_ENCODER_NUM_GPUS:-1}，实际可见 ${_visible_gpus}。" >&2
        echo "       verl 的 placement group 会占满 Ray 可见的所有卡，encoder policy LM 会永远排队，" >&2
        echo "       表现为训练在 worker init 处静默挂死（不报错、不超时）。" >&2
        echo "       解法：扩大 CUDA_VISIBLE_DEVICES，或调小 N_GPUS_PER_NODE / DYAD_ENCODER_NUM_GPUS，" >&2
        echo "       或设 DYAD_ENCODER_REMOTE=0 让 encoder 与 policy LM 共卡（就不是 2+2 隔离了）。" >&2
        exit 1
    fi
    echo "[dyad] encoder 独占 ${DYAD_ENCODER_NUM_GPUS:-1} 卡，trainer ${N_GPUS_PER_NODE} 卡，可见 ${_visible_gpus} 卡"
fi


RUN_ALGO_BASE=dyad

export DYAD_DIAG_ENABLED="${DYAD_DIAG_ENABLED:-0}"
export DYAD_DIAG_MAX_VALUES="${DYAD_DIAG_MAX_VALUES:-64}"

RESUME_MODE="${RESUME_MODE:-disable}"
OPTIONAL_HYDRA_ARGS=()
append_resume_args

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    OPTIONAL_HYDRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi
if [ -n "${ATTN_IMPL:-}" ]; then
    OPTIONAL_HYDRA_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}")
fi

RAY_RUNTIME_LOG_PREFIX="[dyad-alfworld]"

HYDRA_ARGS=(
    "algorithm.adv_estimator=grpo"
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=${yaml_path}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=0.9"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.6}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-1.0}"
    "actor_rollout_ref.actor.strategy=dyad"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.return_raw_chat=True"
    "data.val_max_samples=${VAL_MAX_SAMPLES:--1}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}"
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
    "actor_rollout_ref.rollout.name=dyadvllm"
    "actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER_FORMAT}"
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS:-1}"
    "actor_rollout_ref.model.path=${model_path}"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left"
    "algorithm.use_kl_in_reward=False"
    "trainer.critic_warmup=0"
    "trainer.project_name=${text_environment}"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=1"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.max_critic_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.resume_mode=${RESUME_MODE}"
    'trainer.logger=["console","wandb"]'
    "data.train_files=${train_data}"
    "data.val_files=${test_data}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "hydra.run.dir=${HYDRA_RUN_DIR}"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512"
    "actor_rollout_ref.rollout.trace.token2text=False"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.multi_turn.enable=true"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}"
)

emit_command agent_system.policies.dyad.training.main_dyad "$@"
)

build_text_environment_grpo_react() (
text_environment="$1"; shift
configure_text_environment "$text_environment"
cd "${PROJECT_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venvs/expa-verl/bin/python}"

export VLLM_USE_V1=1
export RAY_DEBUG_POST_MORTEM=1

TIME_SUFFIX="${TIME_SUFFIX:-$(date +"%Y%m%d_%H%M%S")}"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-0.5B-Instruct}"
HF_HUB_DIR="${HF_HUB_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH="$(ls -d "${HF_HUB_DIR}"/models--*--"${MODEL_NAME##*/}"/snapshots/*/ 2>/dev/null | head -1 || true)"
    MODEL_PATH="${MODEL_PATH%/}"
fi
[ -n "${MODEL_PATH:-}" ] || { echo "[grpo_alfworld] 无法解析模型路径 (MODEL_NAME=${MODEL_NAME}, HF_HUB_DIR=${HF_HUB_DIR})；请设 MODEL_PATH 或 MODEL_NAME"; exit 1; }
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL_NAME##*/}" | tr 'A-Z' 'a-z')}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${MODEL_TAG}}"
EXP_NAME="${EXP_NAME:-${RUN_ALGO_BASE:-grpo_react}-${TIME_SUFFIX}}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))}"

RUN_ALGO_BASE="${RUN_ALGO_BASE:-grpo_react}"

export DYAD_DIAG_ENABLED="${DYAD_DIAG_ENABLED:-0}"
export DYAD_DIAG_MAX_VALUES="${DYAD_DIAG_MAX_VALUES:-64}"
export GRPO_FULL_DUMP_LIMIT="${GRPO_FULL_DUMP_LIMIT:-64}"

export STRESS_SYNTH_ENABLED="${STRESS_SYNTH_ENABLED:-0}"
export STRESS_ACTION_TOKENS="${STRESS_ACTION_TOKENS:-128}"
export STRESS_OBS_TOKENS="${STRESS_OBS_TOKENS:-512}"
export STRESS_TURNS="${STRESS_TURNS:-30}"

MULTI_TURN_FORMAT="react"
train_data="${TRAIN_DATA:?prepare.py must select the shared dataset}"
test_data="${TEST_DATA:?prepare.py must select the shared dataset}"

yaml_path="${TOOL_CONFIG_PATH:-${PROJECT_DIR}/agent_system/environments/configs/${text_environment}_tool.yaml}"

model_path="${MODEL_PATH}"

RESUME_MODE="${RESUME_MODE:-disable}"
OPTIONAL_HYDRA_ARGS=()
append_resume_args

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    OPTIONAL_HYDRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

if [ -n "${ATTN_IMPL:-}" ]; then
    OPTIONAL_HYDRA_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}")
fi

RAY_RUNTIME_LOG_PREFIX="[grpo-alfworld]"

HYDRA_ARGS=(
    "algorithm.adv_estimator=grpo"

    "data.train_files=${train_data}"
    "data.val_files=${test_data}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.return_raw_chat=True"
    "data.val_max_samples=${VAL_MAX_SAMPLES:--1}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}"

    "actor_rollout_ref.model.path=${model_path}"
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
    "actor_rollout_ref.actor.strategy=fsdp"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD}"

    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=0.9"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.6}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-1.0}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.multi_turn.enable=true"
    "actor_rollout_ref.rollout.multi_turn.format=${MULTI_TURN_FORMAT}"
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=${yaml_path}"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}"
    "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left"
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS:-1}"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512"
    "actor_rollout_ref.rollout.trace.token2text=False"

    "algorithm.use_kl_in_reward=False"
    "trainer.critic_warmup=0"
    "trainer.project_name=${text_environment}"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=1"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.max_critic_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.resume_mode=${RESUME_MODE}"
    'trainer.logger=["console","wandb"]'
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "hydra.run.dir=${HYDRA_RUN_DIR}"
)

emit_command verl.trainer.main_ppo "$@"
)

# codegym / dyad
build_codegym_dyad() (
cd "${PROJECT_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venvs/expa-verl/bin/python}"

export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_USE_V2_MODEL_RUNNER=0

export DYAD_ENCODER_ENABLED="${DYAD_ENCODER_ENABLED:-1}"
export DYAD_ENCODER_SCALE="${DYAD_ENCODER_SCALE:-unit}"
export DYAD_ENCODER_GATE_INIT="${DYAD_ENCODER_GATE_INIT:-0.0}"
export DYAD_PROJECTOR_LR="${DYAD_PROJECTOR_LR:-1e-4}"
export DYAD_ENCODER_PROJECTOR_INIT="${DYAD_ENCODER_PROJECTOR_INIT:-}"
export DYAD_ENCODER_MODEL_PATH="${DYAD_ENCODER_MODEL_PATH:-}"  # Empty selects the policy model specification with independent weights.
export DYAD_ENCODER_GPU_IDS="${DYAD_ENCODER_GPU_IDS:-}"
export DYAD_ENCODER_REMOTE="${DYAD_ENCODER_REMOTE:-1}"
export DYAD_ENCODER_ACTOR_NAME="${DYAD_ENCODER_ACTOR_NAME:-dyad_action_encoder}"
export DYAD_ENCODER_MAX_LENGTH="${DYAD_ENCODER_MAX_LENGTH:-1024}"
export DYAD_ENCODER_BACKBONE="${DYAD_ENCODER_BACKBONE:-encoder_lm}"
export DYAD_ENCODER_PROJECTOR="${DYAD_ENCODER_PROJECTOR:-attention}"
export DYAD_ENCODER_REPRESENTATION="${DYAD_ENCODER_REPRESENTATION:-final_layer_hidden_states}"
export DYAD_ENCODER_DESCRIPTION="${DYAD_ENCODER_DESCRIPTION:-mcp}"
export DYAD_TRAINING_SCHEDULE="${DYAD_TRAINING_SCHEDULE:-frozen_llm_adaptation}"
export DYAD_ENCODER_TRAINING="${DYAD_ENCODER_TRAINING:-projector_and_encoder_lm}"
export RAY_DEBUG_POST_MORTEM=${RAY_DEBUG_POST_MORTEM:-1}
export RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-0}

TIME_SUFFIX="${TIME_SUFFIX:-$(date +"%Y%m%d_%H%M%S")}"

dyad_variant_require codegym TEMPLATE_STYLE
EXP_NAME=${EXP_NAME:-dyad_${TIME_SUFFIX}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/local/agentic_rl}

MODEL_NAME="${MODEL_NAME:-Qwen2.5-0.5B-Instruct}"
HF_HUB_DIR="${HF_HUB_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
if [ -z "${MODEL_PATH:-}" ]; then
  MODEL_PATH="$(ls -d "${HF_HUB_DIR}"/models--*--"${MODEL_NAME##*/}"/snapshots/*/ 2>/dev/null | head -1 || true)"
  MODEL_PATH="${MODEL_PATH%/}"
fi
[ -n "${MODEL_PATH:-}" ] || { echo "[dyad_codegym] 无法解析模型路径 (MODEL_NAME=${MODEL_NAME}, HF_HUB_DIR=${HF_HUB_DIR})；请设 MODEL_PATH 或 MODEL_NAME"; exit 1; }
model_path="${MODEL_PATH}"

MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL_NAME##*/}" | tr 'A-Z' 'a-z')}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${MODEL_TAG}}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[dyad_codegym] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未限制（继承调度器分配的全部卡）}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
else
  echo "[dyad_codegym] WARNING: nvidia-smi not found; cannot verify GPU state." >&2
fi

[ "${DYAD_CODEGYM_ALL:-0}" = 1 ] || { echo 'Use train.sh/evaluate.sh: CodeGym requires all environments' >&2; exit 2; }
train_data="${TRAIN_DATA:?Full CodeGym train split must be prepared}"
val_data="${TEST_DATA:?Full CodeGym test split must be prepared}"
for data_path in "${train_data}" "${val_data}"; do
    [[ -f "${data_path}" ]] || { echo "[dyad_codegym] missing global parquet: ${data_path}" >&2; exit 2; }
done

yaml_path=${TOOL_CONFIG_PATH:-"${PROJECT_DIR}/agent_system/environments/configs/codegym_tool.yaml"}

if [ "${DYAD_ENCODER_ENABLED:-0}" = "1" ] && [ "${DYAD_ENCODER_REMOTE:-0}" = "1" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        _visible_gpus="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' ' ' | wc -w)"
    else
        _visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
    fi
    _need=$(( N_GPUS_PER_NODE + ${DYAD_ENCODER_NUM_GPUS:-1} ))
    if [ "${_visible_gpus}" -lt "${_need}" ]; then
        echo "[dyad] DYAD_ENCODER_REMOTE=1 需要 ${_need} 张可见卡" >&2
        echo "       trainer ${N_GPUS_PER_NODE} + encoder ${DYAD_ENCODER_NUM_GPUS:-1}，实际可见 ${_visible_gpus}。" >&2
        echo "       verl 的 placement group 会占满 Ray 可见的所有卡，encoder policy LM 会永远排队，" >&2
        echo "       表现为训练在 worker init 处静默挂死（不报错、不超时）。" >&2
        echo "       解法：扩大 CUDA_VISIBLE_DEVICES，或调小 N_GPUS_PER_NODE / DYAD_ENCODER_NUM_GPUS，" >&2
        echo "       或设 DYAD_ENCODER_REMOTE=0 让 encoder 与 policy LM 共卡（就不是 2+2 隔离了）。" >&2
        exit 1
    fi
    echo "[dyad] encoder 独占 ${DYAD_ENCODER_NUM_GPUS:-1} 卡，trainer ${N_GPUS_PER_NODE} 卡，可见 ${_visible_gpus} 卡"
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))}"

RUN_ALGO_BASE=dyad
RUN_SLUG="${RUN_SLUG:-all}"

export DYAD_DIAG_ENABLED=${DYAD_DIAG_ENABLED:-1}
export DYAD_DIAG_MAX_VALUES=${DYAD_DIAG_MAX_VALUES:-64}

TRAINER_LOGGER=${TRAINER_LOGGER:-[\"console\",\"wandb\"]}

OPTIONAL_HYDRA_ARGS=()
append_resume_args
if [ -n "${ATTN_IMPL:-}" ]; then
    OPTIONAL_HYDRA_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}")
fi
if [ -n "${CKPT_DIR:-}" ]; then
    OPTIONAL_HYDRA_ARGS+=("trainer.default_local_dir=${CKPT_DIR}")
fi
if [ -n "${TOTAL_TRAINING_STEPS:-}" ]; then
    OPTIONAL_HYDRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

RAY_RUNTIME_LOG_PREFIX="[dyad-codegym-ray]"

echo "[dyad_codegym] EXP_NAME=${EXP_NAME}"
echo "[dyad_codegym] all environments, per-task action schemas"
echo "[dyad_codegym] train=${train_data} val=${val_data}"
echo "[dyad_codegym] model=${model_path} gpu_mem_util=${GPU_MEM_UTIL}"

HYDRA_ARGS=(
    "algorithm.adv_estimator=grpo"
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=${yaml_path}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-0.65}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.95}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.65}"
    "actor_rollout_ref.actor.strategy=dyad"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS:-true}"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF:-0}"
    "actor_rollout_ref.actor.grad_clip=${GRAD_CLIP:-1.0}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.name=dyadvllm"
    "actor_rollout_ref.rollout.multi_turn.format=dyad"
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS:-1}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left"
    "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}"
    # Bound generation as well as observations; the user guard runs after generation.
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS:-${MAX_TOOL_TURNS}}"
    "actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TOOL_TURNS}"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.multi_turn.enable=true"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512"
    "actor_rollout_ref.rollout.trace.token2text=False"
    "actor_rollout_ref.model.path=${model_path}"
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "algorithm.use_kl_in_reward=False"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.return_raw_chat=True"
    "data.val_max_samples=${VAL_MAX_SAMPLES:--1}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}"
    "data.train_files=${train_data}"
    "data.val_files=${val_data}"
    "trainer.critic_warmup=0"
    "trainer.project_name=codegym"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=1"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.max_critic_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.resume_mode=${RESUME_MODE:-disable}"
    "trainer.logger=${TRAINER_LOGGER}"
    "trainer.rollout_data_dir=${RUN_DIR}/rollout"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "hydra.run.dir=${HYDRA_RUN_DIR}"
)

emit_command agent_system.policies.dyad.training.main_dyad "$@"
)

# codegym / grpo_react
build_codegym_grpo_react() (
cd "${PROJECT_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venvs/expa-verl/bin/python}"

export VLLM_USE_V1=1
export RAY_DEBUG_POST_MORTEM=1
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

TIME_SUFFIX="${TIME_SUFFIX:-$(date +"%Y%m%d_%H%M%S")}"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-0.5B-Instruct}"
HF_HUB_DIR="${HF_HUB_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH="$(ls -d "${HF_HUB_DIR}"/models--*--"${MODEL_NAME##*/}"/snapshots/*/ 2>/dev/null | head -1 || true)"
    MODEL_PATH="${MODEL_PATH%/}"
fi
[ -n "${MODEL_PATH:-}" ] || { echo "[codegym] 无法解析模型路径 (MODEL_NAME=${MODEL_NAME}, HF_HUB_DIR=${HF_HUB_DIR})；请设 MODEL_PATH 或 MODEL_NAME"; exit 1; }
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL_NAME##*/}" | tr 'A-Z' 'a-z')}"
EXP_NAME="${EXP_NAME:-${RUN_ALGO_BASE:-grpo_react}-${TIME_SUFFIX}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${MODEL_TAG}}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES
fi

RUN_ALGO_BASE="${RUN_ALGO_BASE:-grpo_react}"

export DYAD_DIAG_ENABLED="${DYAD_DIAG_ENABLED:-0}"
export DYAD_DIAG_MAX_VALUES="${DYAD_DIAG_MAX_VALUES:-64}"
export GRPO_FULL_DUMP_LIMIT="${GRPO_FULL_DUMP_LIMIT:-64}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))}"

train_data="${TRAIN_DATA:?prepare.py must select the shared dataset}"
test_data="${TEST_DATA:?prepare.py must select the shared dataset}"

yaml_path="${TOOL_CONFIG_PATH:-${PROJECT_DIR}/agent_system/environments/configs/codegym_tool.yaml}"

export CODEGYM_ENV_BACKEND="${CODEGYM_ENV_BACKEND:-ray}"
echo "[codegym] in-process Ray env pool (CodeGymLocalEnvTool) backend=${CODEGYM_ENV_BACKEND} pool_size=${CODEGYM_ENV_POOL_SIZE:-auto-per-worker}; no external server."

model_path="${MODEL_PATH}"

RESUME_MODE="${RESUME_MODE:-disable}"
OPTIONAL_HYDRA_ARGS=()
append_resume_args

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    OPTIONAL_HYDRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

if [ -n "${ATTN_IMPL:-}" ]; then
    OPTIONAL_HYDRA_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}")
fi

RAY_RUNTIME_LOG_PREFIX="[grpo-codegym-ray]"

TOOL_PARSER_FORMAT="codegym"

HYDRA_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=False"

    "data.train_files=${train_data}"
    "data.val_files=${test_data}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.return_raw_chat=True"
    "data.val_max_samples=${VAL_MAX_SAMPLES:--1}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}"

    "actor_rollout_ref.model.path=${model_path}"
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
    "actor_rollout_ref.actor.strategy=fsdp"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
    "actor_rollout_ref.actor.optim.weight_decay=0.1"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.actor.clip_ratio=0.2"
    "actor_rollout_ref.actor.grad_clip=1.0"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD}"

    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=1.0"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-1.0}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.7}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.multi_turn.enable=true"
    "actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER_FORMAT}"
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=${yaml_path}"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}"
    "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left"
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS}"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512"
    "actor_rollout_ref.rollout.trace.token2text=False"

    "trainer.critic_warmup=0"
    "trainer.project_name=codegym"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=1"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.max_critic_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.resume_mode=${RESUME_MODE}"
    'trainer.logger=["console","wandb"]'
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "hydra.run.dir=${HYDRA_RUN_DIR}"
)

emit_command verl.trainer.main_ppo "$@"
)

# gsm8k / dyad
build_gsm8k_dyad() (
cd "${PROJECT_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venvs/expa-verl/bin/python}"

export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=0

export DYAD_ENCODER_ENABLED="${DYAD_ENCODER_ENABLED:-1}"
export DYAD_ENCODER_SCALE="${DYAD_ENCODER_SCALE:-unit}"
export DYAD_ENCODER_GATE_INIT="${DYAD_ENCODER_GATE_INIT:-0.0}"
export DYAD_PROJECTOR_LR="${DYAD_PROJECTOR_LR:-1e-4}"
export DYAD_ENCODER_PROJECTOR_INIT="${DYAD_ENCODER_PROJECTOR_INIT:-}"
export DYAD_ENCODER_MODEL_PATH="${DYAD_ENCODER_MODEL_PATH:-}"  # Empty selects the policy model specification with independent weights.
export DYAD_ENCODER_GPU_IDS="${DYAD_ENCODER_GPU_IDS:-}"
export DYAD_ENCODER_REMOTE="${DYAD_ENCODER_REMOTE:-1}"
export DYAD_ENCODER_ACTOR_NAME="${DYAD_ENCODER_ACTOR_NAME:-dyad_action_encoder}"
export DYAD_ENCODER_MAX_LENGTH="${DYAD_ENCODER_MAX_LENGTH:-1024}"
export DYAD_ENCODER_BACKBONE="${DYAD_ENCODER_BACKBONE:-encoder_lm}"
export DYAD_ENCODER_PROJECTOR="${DYAD_ENCODER_PROJECTOR:-attention}"
export DYAD_ENCODER_REPRESENTATION="${DYAD_ENCODER_REPRESENTATION:-final_layer_hidden_states}"
export DYAD_ENCODER_DESCRIPTION="${DYAD_ENCODER_DESCRIPTION:-mcp}"
export DYAD_TRAINING_SCHEDULE="${DYAD_TRAINING_SCHEDULE:-frozen_llm_adaptation}"
export DYAD_ENCODER_TRAINING="${DYAD_ENCODER_TRAINING:-projector_and_encoder_lm}"
export RAY_DEBUG_POST_MORTEM=1

TIME_SUFFIX="${TIME_SUFFIX:-$(date +"%Y%m%d_%H%M%S")}"

CKPT_ROOT="${CKPT_ROOT:-${PROJECT_DIR}/ckpt/local/agentic_rl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/local/agentic_rl}"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-0.5B-Instruct}"
HF_HUB_DIR="${HF_HUB_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH="$(ls -d "${HF_HUB_DIR}"/models--*--"${MODEL_NAME##*/}"/snapshots/*/ 2>/dev/null | head -1 || true)"
    MODEL_PATH="${MODEL_PATH%/}"
fi
[ -n "${MODEL_PATH:-}" ] || { echo "[dyad_math] 无法解析模型路径 (MODEL_NAME=${MODEL_NAME}, HF_HUB_DIR=${HF_HUB_DIR})；请设 MODEL_PATH 或 MODEL_NAME"; exit 1; }
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL_NAME##*/}" | tr 'A-Z' 'a-z')}"
EXP_NAME="${EXP_NAME:-dyad_${TIME_SUFFIX}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${MODEL_TAG}}"

TOOL_PARSER_FORMAT="dyad"
dyad_variant_require gsm8k DYAD_ACTION_YAML
export DYAD_ACTION_YAML

TRAIN_DATA="${TRAIN_DATA:?[dyad] TRAIN_DATA 未设置；走 experiments/shared/train_eval/train.sh gsm8k}"
TEST_DATA="${TEST_DATA:?[dyad] TEST_DATA 未设置；走 experiments/shared/train_eval/train.sh gsm8k}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-${PROJECT_DIR}/agent_system/environments/configs/calc_tool.yaml}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))}"

if [ "${DYAD_ENCODER_ENABLED:-0}" = "1" ] && [ "${DYAD_ENCODER_REMOTE:-0}" = "1" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        _visible_gpus="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' ' ' | wc -w)"
    else
        _visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
    fi
    _need=$(( N_GPUS_PER_NODE + ${DYAD_ENCODER_NUM_GPUS:-1} ))
    if [ "${_visible_gpus}" -lt "${_need}" ]; then
        echo "[dyad] DYAD_ENCODER_REMOTE=1 需要 ${_need} 张可见卡" >&2
        echo "       trainer ${N_GPUS_PER_NODE} + encoder ${DYAD_ENCODER_NUM_GPUS:-1}，实际可见 ${_visible_gpus}。" >&2
        echo "       verl 的 placement group 会占满 Ray 可见的所有卡，encoder policy LM 会永远排队，" >&2
        echo "       表现为训练在 worker init 处静默挂死（不报错、不超时）。" >&2
        echo "       解法：扩大 CUDA_VISIBLE_DEVICES，或调小 N_GPUS_PER_NODE / DYAD_ENCODER_NUM_GPUS，" >&2
        echo "       或设 DYAD_ENCODER_REMOTE=0 让 encoder 与 policy LM 共卡（就不是 2+2 隔离了）。" >&2
        exit 1
    fi
    echo "[dyad] encoder 独占 ${DYAD_ENCODER_NUM_GPUS:-1} 卡，trainer ${N_GPUS_PER_NODE} 卡，可见 ${_visible_gpus} 卡"
fi

_calc_traj=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
_calc_cpu_cap=$(( $(ray_effective_cpus) * 10 ))
export CALC_ENV_POOL_SIZE="${CALC_ENV_POOL_SIZE:-$(( _calc_traj < _calc_cpu_cap ? _calc_traj : _calc_cpu_cap ))}"
echo "[dyad-math] calc env pool → CALC_ENV_POOL_SIZE=${CALC_ENV_POOL_SIZE} = min(cpu_cap=${_calc_cpu_cap}, batch*n=${_calc_traj})（超出 pool 的 session 排队复用 policy LM）"

RESUME_MODE="${RESUME_MODE:-disable}"

RUN_ALGO_BASE=dyad

export DYAD_DIAG_ENABLED="${DYAD_DIAG_ENABLED:-0}"
export DYAD_DIAG_MAX_VALUES="${DYAD_DIAG_MAX_VALUES:-64}"

OPTIONAL_HYDRA_ARGS=()
append_resume_args

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    OPTIONAL_HYDRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi
if [ -n "${ATTN_IMPL:-}" ]; then
    OPTIONAL_HYDRA_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}")
fi

echo "[dyad-math] model=${MODEL_PATH}"
echo "[dyad-math] prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} max_turns=${MAX_ASSISTANT_TURNS} tool_response=${MAX_TOOL_RESPONSE_LENGTH}"
echo "[dyad-math] batch=${TRAIN_BATCH_SIZE} micro=${MICRO_BATCH_SIZE} n=${ROLLOUT_N} gpu_mem=${GPU_MEM_UTIL} attn=${ATTN_IMPL:-model-default} remove_padding=${USE_REMOVE_PADDING}"

RAY_RUNTIME_LOG_PREFIX="[dyad-math]"

HYDRA_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=False"

    "data.train_files=${TRAIN_DATA}"
    "data.val_files=${TEST_DATA}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.return_raw_chat=True"
    "data.val_max_samples=${VAL_MAX_SAMPLES:--1}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}"

    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
    "actor_rollout_ref.actor.strategy=dyad"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD}"

    "actor_rollout_ref.rollout.name=dyadvllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=0.9"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.6}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-1.0}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.multi_turn.enable=true"
    "actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER_FORMAT}"
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH}"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}"
    "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left"
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS:-1}"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512"
    "actor_rollout_ref.rollout.trace.token2text=False"

    "trainer.critic_warmup=0"
    "trainer.project_name=gsm8k"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=1"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.max_critic_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.resume_mode=${RESUME_MODE}"
    'trainer.logger=["console","wandb"]'
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "hydra.run.dir=${HYDRA_RUN_DIR}"
)

emit_command agent_system.policies.dyad.training.main_dyad "$@"
)

# gsm8k / grpo_react
build_gsm8k_grpo_react() (
cd "${PROJECT_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venvs/expa-verl/bin/python}"

export VLLM_USE_V1=1
export RAY_DEBUG_POST_MORTEM=1

TIME_SUFFIX="${TIME_SUFFIX:-$(date +"%Y%m%d_%H%M%S")}"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-0.5B-Instruct}"
HF_HUB_DIR="${HF_HUB_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH="$(ls -d "${HF_HUB_DIR}"/models--*--"${MODEL_NAME##*/}"/snapshots/*/ 2>/dev/null | head -1 || true)"
    MODEL_PATH="${MODEL_PATH%/}"
fi
[ -n "${MODEL_PATH:-}" ] || { echo "[grpo_math] 无法解析模型路径 (MODEL_NAME=${MODEL_NAME}, HF_HUB_DIR=${HF_HUB_DIR})；请设 MODEL_PATH 或 MODEL_NAME"; exit 1; }
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL_NAME##*/}" | tr 'A-Z' 'a-z')}"
EXP_NAME="${EXP_NAME:-${RUN_ALGO_BASE:-grpo_react}-${TIME_SUFFIX}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${MODEL_TAG}}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))}"
MODEL_STAGE_TO_LOCAL="${MODEL_STAGE_TO_LOCAL:-$((N_GPUS_PER_NODE >= 4 ? 1 : 0))}"
MODEL_STAGE_CREATED=0
MODEL_STAGE_DIR=""

RUN_ALGO_BASE="${RUN_ALGO_BASE:-grpo_react}"

export DYAD_DIAG_ENABLED="${DYAD_DIAG_ENABLED:-0}"
export DYAD_DIAG_MAX_VALUES="${DYAD_DIAG_MAX_VALUES:-64}"
export GRPO_FULL_DUMP_LIMIT="${GRPO_FULL_DUMP_LIMIT:-64}"

MULTI_TURN_FORMAT="react"

export REACT_TOOL_NAME="${REACT_TOOL_NAME:-calculator}"

_calc_traj=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
_calc_cpu_cap=$(( $(ray_effective_cpus) * 10 ))
export CALC_ENV_POOL_SIZE="${CALC_ENV_POOL_SIZE:-$(( _calc_traj < _calc_cpu_cap ? _calc_traj : _calc_cpu_cap ))}"
echo "[grpo_math] calc env pool → CALC_ENV_POOL_SIZE=${CALC_ENV_POOL_SIZE} = min(cpu_cap=${_calc_cpu_cap}, batch*n=${_calc_traj})（超出 pool 的 session 排队复用 policy LM）"

train_data="${TRAIN_DATA:?prepare.py must select the shared dataset}"
test_data="${TEST_DATA:?prepare.py must select the shared dataset}"

yaml_path="${TOOL_CONFIG_PATH:-${PROJECT_DIR}/agent_system/environments/configs/calc_tool.yaml}"
echo "[grpo_math] using in-process Ray calculator env pool (CalcLocalEnvTool); no external server needed."

export DYAD_MODEL_SOURCE_PATH="${MODEL_PATH}"

model_path="${MODEL_PATH}"
echo "[grpo_math] model_stage=${MODEL_STAGE_TO_LOCAL}"

RESUME_MODE="${RESUME_MODE:-disable}"
OPTIONAL_HYDRA_ARGS=()
append_resume_args

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    OPTIONAL_HYDRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

if [ -n "${ATTN_IMPL:-}" ]; then
    OPTIONAL_HYDRA_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}")
fi

RAY_RUNTIME_LOG_PREFIX="[grpo_math]"

TOOL_PARSER_FORMAT="react"

HYDRA_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=False"

    "data.train_files=${train_data}"
    "data.val_files=${test_data}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.return_raw_chat=True"
    "data.val_max_samples=${VAL_MAX_SAMPLES:--1}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}"

    "actor_rollout_ref.model.path=${model_path}"
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
    "actor_rollout_ref.actor.strategy=fsdp"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD}"

    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=0.9"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.6}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-1.0}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.multi_turn.enable=true"
    "actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER_FORMAT}"
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=${yaml_path}"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}"
    "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left"
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS:-1}"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512"
    "actor_rollout_ref.rollout.trace.token2text=False"

    "trainer.critic_warmup=0"
    "trainer.project_name=gsm8k"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=1"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.max_critic_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.resume_mode=${RESUME_MODE}"
    'trainer.logger=["console","wandb"]'
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "hydra.run.dir=${HYDRA_RUN_DIR}"
)

emit_command verl.trainer.main_ppo "$@"
)

# Native DIVE sessions share all sampling, reward and optimization settings.
build_dive_dyad() ( build_dive dyad "$@"; )
build_dive_grpo_react() ( build_dive grpo_react "$@"; )
build_dive() (
    local interface="$1"; shift
    local module=verl.trainer.main_ppo strategy=fsdp rollout=vllm format=dive
    if [ "$interface" = dyad ]; then
        module=agent_system.policies.dyad.training.main_dyad strategy=dyad rollout=dyadvllm format=dyad
    fi
    export VLLM_USE_V1=1
    local session
    session="$("${PYTHON_BIN}" - <<'PY'
import json, os
def encode(value):
    if isinstance(value, dict):
        return '{' + ','.join(key + ':' + encode(item) for key, item in value.items()) + '}'
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
print(encode(json.loads(os.environ['DIVE_SESSION_CONFIG'])))
PY
)"
    OPTIONAL_HYDRA_ARGS=()
    append_resume_args
    [ -z "${TOTAL_TRAINING_STEPS:-}" ] || OPTIONAL_HYDRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
    [ -z "${ATTN_IMPL:-}" ] || OPTIONAL_HYDRA_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}")
    HYDRA_ARGS=(
        algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False
        "data.train_files=${TRAIN_DATA:?}" "data.val_files=${TEST_DATA:?}"
        data.custom_cls.path=pkg://agent_system.environments.backends.dive.dataset data.custom_cls.name=DiveDataset
        "+data.dive.session_config=${session}"
        "data.train_batch_size=${TRAIN_BATCH_SIZE}" "data.val_max_samples=${VAL_MAX_SAMPLES:--1}"
        "data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}"
        "data.max_prompt_length=${MAX_PROMPT_LENGTH}" "data.max_response_length=${MAX_RESPONSE_LENGTH}"
        data.filter_overlong_prompts=False data.truncation=error data.return_raw_chat=True
        "actor_rollout_ref.model.path=${MODEL_PATH:?}"
        "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
        "actor_rollout_ref.actor.strategy=${strategy}"
        "actor_rollout_ref.actor.optim.lr=${LR}" "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
        actor_rollout_ref.actor.optim.weight_decay=0.1
        "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
        "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
        actor_rollout_ref.actor.use_kl_loss=True
        "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}" "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
        actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.use_torch_compile=False
        "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}"
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}"
        "actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD}"
        "actor_rollout_ref.rollout.name=${rollout}" actor_rollout_ref.rollout.mode=async
        "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}" actor_rollout_ref.rollout.top_p=1.0
        "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.6}"
        "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-1.0}"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
        "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
        "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
        "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN:-$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))}"
        actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
        actor_rollout_ref.rollout.multi_turn.enable=True
        "actor_rollout_ref.rollout.multi_turn.format=${format}"
        "actor_rollout_ref.rollout.multi_turn.tool_config_path=${DIVE_TOOL_CONFIG_PATH:?}"
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}"
        "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}"
        "actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS}"
        actor_rollout_ref.rollout.trace.token2text=False
        trainer.critic_warmup=0 trainer.project_name=dive
        "trainer.experiment_name=${EXP_NAME}" "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}" trainer.nnodes=1
        "trainer.default_local_dir=${CKPT_DIR}" "trainer.save_freq=${SAVE_FREQ}" "trainer.test_freq=${TEST_FREQ}"
        "trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-null}" "trainer.max_critic_ckpt_to_keep=${MAX_CKPT_KEEP:-null}"
        "trainer.resume_mode=${RESUME_MODE:-disable}" 'trainer.logger=["console","wandb"]'
        "trainer.total_epochs=${TOTAL_EPOCHS}" "hydra.run.dir=${HYDRA_RUN_DIR}"
    )
    emit_command "$module" "$@"
)

assemble_command() {
    local benchmark="$1" algo="$2"; shift 2
    if [ "$benchmark" = tbench ]; then
        echo 'Unsupported benchmark: tbench has been removed.' >&2
        return 2
    fi
    if [ "${RUN_IS_EVAL:-0}" != 1 ]; then
        case "$benchmark" in
            dive|codegym|alfworld|webshop) ;;
            *) echo "$benchmark is evaluation-only; use evaluate.sh. Training datasets: dive, codegym, alfworld, webshop" >&2; return 2 ;;
        esac
    fi
    if [ "$algo" = gigpo ]; then
        export RUN_ALGO_BASE=gigpo
        algo=grpo_react
    fi
    case "$benchmark/$algo" in
        gsm8k/dyad|gsm8k/grpo_react|alfworld/dyad|alfworld/grpo_react|codegym/dyad|codegym/grpo_react|webshop/dyad|webshop/grpo_react|dive/dyad|dive/grpo_react)
            "build_${benchmark}_${algo}" "$@" ;;
        *) echo "Unsupported benchmark/algorithm: $benchmark/$algo" >&2; return 2 ;;
    esac
}

training_main() {
    case "${1:-}" in
        --build) shift; assemble_command "$@" ;;
        --execute)
            shift
            [ "${RUN_IS_EVAL:-0}" = 0 ] || { echo 'Use evaluation.sh for evaluation' >&2; exit 2; }
            exec "$@" ;;
        *) echo 'Internal execution script; use ../train.sh' >&2; exit 2 ;;
    esac
}
if [[ "${BASH_SOURCE[0]}" = "$0" ]]; then training_main "$@"; fi
