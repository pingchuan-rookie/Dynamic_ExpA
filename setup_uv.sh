#!/usr/bin/env bash
# The local installation entrypoint: package metadata comes from pyproject.toml,
# while CUDA runtime pins come from ops/Dockerfile.dyad-verl.
# Use bash setup_uv.sh for layered installation or --lock for requirements-lock.txt.
# VENV_DIR overrides the default .venvs/expa-verl location. Requires uv and a compatible CUDA runtime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
DOCKERFILE="${REPO_ROOT}/ops/Dockerfile.dyad-verl"
LOCK_FILE="${SCRIPT_DIR}/requirements-lock.txt"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venvs/expa-verl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

USE_LOCK=0
[ "${1:-}" = "--lock" ] && USE_LOCK=1

log()  { echo "[setup_uv] $*"; }
die()  { echo "[setup_uv][ERROR] $*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || die "未找到 uv，先装：curl -LsSf https://astral.sh/uv/install.sh | sh"
[ -f "${DOCKERFILE}" ] || die "找不到 ${DOCKERFILE}（版本号要从这里解析）"

# Extract package pins from executable Dockerfile lines; comments must not influence version selection.
pin() {
    local pkg="$1" ver
    ver="$(grep -v '^[[:space:]]*#' "${DOCKERFILE}" \
           | grep -oE "\"?${pkg}==[A-Za-z0-9.+]+\"?" | head -1 | tr -d '"')" || true
    [ -n "${ver}" ] || die "无法从 ops/Dockerfile.dyad-verl 解析 ${pkg} 的版本（Dockerfile 格式变了？）"
    echo "${ver}"
}

TORCH_PIN="$(pin torch)"
TORCHVISION_PIN="$(pin torchvision)"
TORCHAUDIO_PIN="$(pin torchaudio)"
VLLM_PIN="$(pin vllm)"
FLASH_PIN="$(pin flash-attn)"
FLA_PIN="$(pin flash-linear-attention)"
CAUSAL_PIN="$(pin causal-conv1d)"
ALFWORLD_PIN="$(pin alfworld)"
TQ_PIN="$(pin TransferQueue)"

log "解析到的版本（来自 ops/Dockerfile.dyad-verl）："
log "  ${TORCH_PIN} / ${TORCHVISION_PIN} / ${TORCHAUDIO_PIN}"
log "  ${VLLM_PIN} / ${FLASH_PIN} / ${FLA_PIN} / ${ALFWORLD_PIN} / ${TQ_PIN}"

# Detect local GPU architecture before compiling extensions.
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
    [ -n "${DETECTED_ARCH}" ] && log "  探测到本机 GPU 架构 sm_${DETECTED_ARCH//./}"
fi
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DETECTED_ARCH:-9.0}}"

# Reuse the relocatable environment; RECREATE_VENV=1 explicitly rebuilds it.
if [ -d "${VENV_DIR}" ] && [ "${RECREATE_VENV:-0}" != "1" ]; then
    log "venv 已存在，复用：${VENV_DIR}（要重建加 RECREATE_VENV=1）"
    uv venv --relocatable --python "${PYTHON_VERSION}" --allow-existing "${VENV_DIR}"
else
    log "创建 venv：${VENV_DIR}（python ${PYTHON_VERSION}）"
    uv venv --relocatable --python "${PYTHON_VERSION}" --clear "${VENV_DIR}"
fi
PY="${VENV_DIR}/bin/python"

if [ "${USE_LOCK}" = "1" ]; then
    [ -f "${LOCK_FILE}" ] || die "找不到 ${LOCK_FILE}（首次安装请不要加 --lock，装完会生成它）"
    log "按 requirements-lock.txt 精确复现"
    uv pip install --python "${PY}" -r "${LOCK_FILE}" \
        --index-url "${TORCH_INDEX}" \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match
    log "lock 安装完成，进入自检"
else

# Install the pinned CUDA PyTorch build before extensions that depend on its ABI.
log "[1/6] PyTorch ${TORCH_PIN}"
uv pip install --python "${PY}" \
    "${TORCH_PIN}" "${TORCHVISION_PIN}" "${TORCHAUDIO_PIN}" \
    --index-url "${TORCH_INDEX}"

log "[2/6] 构建工具"
uv pip install --python "${PY}" setuptools wheel packaging ninja

log "[3/6] ${VLLM_PIN} + ray"
uv pip install --python "${PY}" \
    "${VLLM_PIN}" "ray[default]>=2.41.0" \
    --extra-index-url "${TORCH_INDEX}" \
    --index-strategy unsafe-best-match

# Try a compatible locked wheel, package resolution, source build, then matching image artifacts.
# Extension compatibility requires matching torch, CUDA, Python, and C++ ABI.
# CPU-only checks may skip flash-attn; use VERL_DISABLE_FLASH_ATTN_CE=1 for CPU loss tensors.
log "[4/6] ${FLASH_PIN}"
FLASH_WHEEL="$(grep -oE 'https://github.com/Dao-AILab/flash-attention/[^ ]+\.whl' "${LOCK_FILE}" 2>/dev/null | head -1 || true)"
LOCAL_ABI="$("${PY}" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"

if [ "${SKIP_FLASH_ATTN:-0}" = "1" ]; then
    log "      SKIP_FLASH_ATTN=1，跳过（该环境只能跑 CPU 侧测试，不能跑训练）"
elif [ -n "${FLASH_WHEEL}" ] && [[ "${FLASH_WHEEL}" == *"cxx11abi${LOCAL_ABI}"* ]] \
   && uv pip install --python "${PY}" "${FLASH_WHEEL}"; then
    log "      lock 里的预编译 wheel 安装成功"
elif uv pip install --python "${PY}" "${FLASH_PIN}" --only-binary=:all: 2>/dev/null; then
    log "      从索引拿到了匹配当前 torch 的预编译 wheel"
elif command -v nvcc >/dev/null 2>&1; then
    log "      无可用 wheel，回落源码编译（arch=${TORCH_CUDA_ARCH_LIST}，很慢）……"
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS:-4}" \
        uv pip install --python "${PY}" "${FLASH_PIN}" --no-build-isolation
elif command -v docker >/dev/null 2>&1 && docker image inspect "${DYAD_VERL_IMAGE:-pingchuan03/dynamic-expa-verl:latest}" >/dev/null 2>&1; then
    # Reuse image-built extensions only with matching torch, CUDA, Python, and C++ ABI.
    IMG="${DYAD_VERL_IMAGE:-pingchuan03/dynamic-expa-verl:latest}"
    log "      本机无 nvcc，改从镜像 ${IMG} 取编译好的 flash-attn"
    SP_LOCAL="$("${PY}" -c 'import site; print(site.getsitepackages()[0])')"
    SP_IMG=/opt/venv-expa-verl/lib/python3.12/site-packages
    TMP_FA="$(mktemp -d)"
    CID="$(docker create "${IMG}")"
    for item in flash_attn "flash_attn-${FLASH_PIN#flash-attn==}.dist-info" flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so; do
        docker cp "${CID}:${SP_IMG}/${item}" "${TMP_FA}/" 2>/dev/null || log "      （镜像内无 ${item}，跳过）"
    done
    docker rm -f "${CID}" >/dev/null
    cp -r "${TMP_FA}"/* "${SP_LOCAL}/" 2>/dev/null || true
    rm -rf "${TMP_FA}"
    log "      已从镜像取出 flash-attn"
else
    die "flash-attn 装不上：没有匹配 torch ${TORCH_PIN#torch==} / cxx11abi=${LOCAL_ABI} 的预编译 wheel，
     本机又没有 nvcc（CUDA toolkit）无法源码编译，也没有可拷贝的镜像。三选一：
       - docker pull pingchuan03/dynamic-expa-verl:latest 后重跑（会自动从镜像里取编译产物）；
       - 装 CUDA toolkit 后重跑（源码编译，30 分钟起）；
       - SKIP_FLASH_ATTN=1 bash setup_uv.sh   # CPU-only developer checks"
fi

# Install FLA separately for Qwen3.5 linear attention; flash-attn does not supply these kernels.
log "      + ${FLA_PIN}"
uv pip install --python "${PY}" "${FLA_PIN}"

# CUDA extensions must match the existing torch/CUDA/Python/C++ ABI.
log "      + ${CAUSAL_PIN}"
if "${PY}" -c "import causal_conv1d, causal_conv1d_cuda; assert causal_conv1d.__version__ == '${CAUSAL_PIN#causal-conv1d==}'" 2>/dev/null; then
    log "      causal-conv1d 已安装且能 import"
elif command -v nvcc >/dev/null 2>&1; then
    CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --python "${PY}" "${CAUSAL_PIN}" --no-build-isolation --no-deps
elif command -v docker >/dev/null 2>&1; then
    IMG="${DYAD_VERL_IMAGE:-pingchuan03/dynamic-expa-verl:latest}"
    ABI_CHECK='import torch,sys,json; print(json.dumps([torch.__version__,torch.version.cuda,torch._C._GLIBCXX_USE_CXX11_ABI,list(sys.version_info[:2])]))'
    LOCAL_CONV_ABI="$("${PY}" -c "${ABI_CHECK}")"
    IMAGE_CONV_ABI="$(docker run --rm --entrypoint /opt/venv-expa-verl/bin/python "${IMG}" -c "${ABI_CHECK}")"
    [ "${LOCAL_CONV_ABI}" = "${IMAGE_CONV_ABI}" ] || die "causal-conv1d 镜像与本地 ABI 不一致"
    docker run --rm --entrypoint /opt/venv-expa-verl/bin/python "${IMG}" -c \
        "import causal_conv1d, causal_conv1d_cuda; assert causal_conv1d.__version__ == '${CAUSAL_PIN#causal-conv1d==}'"
    SP_LOCAL="$("${PY}" -c 'import site; print(site.getsitepackages()[0])')"
    SP_IMG=/opt/venv-expa-verl/lib/python3.12/site-packages
    CID="$(docker create "${IMG}")"
    for item in causal_conv1d "causal_conv1d-${CAUSAL_PIN#causal-conv1d==}.dist-info" causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so; do
        if ! docker cp "${CID}:${SP_IMG}/${item}" "${SP_LOCAL}/"; then
            docker rm "${CID}" >/dev/null
            die "无法从镜像提取 ${item}"
        fi
    done
    docker rm "${CID}" >/dev/null
else
    die "causal-conv1d 需要 nvcc 编译，或通过 DYAD_VERL_IMAGE 指定含匹配扩展的 Docker 镜像"
fi
"${PY}" -c "import causal_conv1d, causal_conv1d_cuda; assert causal_conv1d.__version__ == '${CAUSAL_PIN#causal-conv1d==}'"

# Read project dependencies from pyproject.toml, then remove the editable package
# so PYTHONPATH selects current source.
log "[5/6] verl 0.9 依赖（装完卸掉包本体）"
( cd "${SCRIPT_DIR}" && uv pip install --python "${PY}" -e ".[vllm]" \
    --extra-index-url "${TORCH_INDEX}" \
    --index-strategy unsafe-best-match )
uv pip uninstall --python "${PY}" verl || true

# Install TransferQueue explicitly because upstream install_requires omits it.
uv pip install --python "${PY}" "${TQ_PIN}"

# Environment actors use this interpreter; retain only the headless cv2 provider.
log "[6/6] 进程内 env 依赖（${ALFWORLD_PIN} + gymnasium + dill）"
uv pip install --python "${PY}" "${ALFWORLD_PIN}" gymnasium dill --index-strategy unsafe-best-match
uv pip uninstall --python "${PY}" opencv-python || true

fi

log "自检……"
FAIL=0
check() {  # check <description> <Python expression>
    if "${PY}" -c "$2" >/dev/null 2>&1; then
        echo "  [ok]   $1"
    else
        echo "  [FAIL] $1"
        FAIL=1
    fi
}

check "torch 能 import 且 CUDA 可用" \
      "import torch; assert torch.cuda.is_available()"
check "torch 是 CUDA 版（非 CPU 版）" \
      "import torch; assert torch.version.cuda"
check "torch 版本与 Dockerfile 的 pin 一致" \
      "import torch; assert torch.__version__.startswith('${TORCH_PIN#torch==}'.split('+')[0]), torch.__version__"
if [ "${SKIP_FLASH_ATTN:-0}" != "1" ]; then
check "flash_attn 能 import（ABI 匹配）" \
      "import flash_attn"
else
    echo "  [skip] flash_attn（SKIP_FLASH_ATTN=1；此环境不可用于训练）"
fi
check "vllm 能 import" \
      "import vllm"
check "vllm 版本与 Dockerfile 的 pin 一致" \
      "import vllm; assert vllm.__version__ == '${VLLM_PIN#vllm==}', vllm.__version__"
check "ray 能 import" \
      "import ray"
# Reject dependency downgrades below the NumPy version required by verl.
check "numpy 主版本 >= 2（verl 0.9 的硬要求）" \
      "import numpy; assert int(numpy.__version__.split('.')[0]) >= 2, numpy.__version__"
check "alfworld / gymnasium / dill 就位（进程内 env 需要）" \
      "import alfworld, gymnasium, dill"
check "opencv-python 已卸载（只留 headless）" \
      "import importlib.metadata as m, sys
try: m.version('opencv-python'); sys.exit(1)
except m.PackageNotFoundError: pass"
check "verl 未被装成包（应从仓库源码导入）" \
      "import importlib.metadata as m, sys
try: m.version('verl'); sys.exit(1)
except m.PackageNotFoundError: pass"

if [ "${FAIL}" != "0" ]; then
    die "自检未通过 —— 上面标 [FAIL] 的项需要处理，先别拿它跑训练"
fi

log "全部通过。venv：${VENV_DIR}"
log "生成 lock 以便复现：uv pip freeze --python ${PY} > ${LOCK_FILE}"
log "源码入口与配置用法见 experiments/shared/train_eval/README.md。"
