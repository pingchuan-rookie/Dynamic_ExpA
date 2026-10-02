# syntax=docker/dockerfile:1

# Build isolated dependency environments; provide project source through runtime mounts.

# Builder stage: install dependencies and retain environments plus managed interpreters.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=120

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=4

ENV HF_HOME=/workspace/.cache/huggingface
ENV RAY_TMPDIR=/workspace/.runtime/ray_tmp
ENV TMPDIR=/workspace/.runtime/tmp
ENV TOKENIZERS_PARALLELISM=false
ENV VLLM_USE_V1=1
ENV WANDB_MODE=offline

ENV ALFWORLD_DATA=/workspace/.cache/alfworld

ENV VENV_DYAD=/opt/venv-dyad
ENV VENV_ALF=/opt/venv-alfworld
ENV VENV_CG=/opt/venv-codegym

# Create TMPDIR before package installation: certificate post-install scripts use mktemp.
RUN mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" \
    && apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential ninja-build cmake \
    lsof \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN uv python install 3.11 3.12

RUN uv venv --python 3.12 ${VENV_DYAD} \
 && uv venv --python 3.11 ${VENV_ALF} \
 && uv venv --python 3.11 ${VENV_CG}

# =============================================================================
# 1) ExpA_sys（GPU：torch cu129 + vllm + flash-attn + verl）
# =============================================================================
WORKDIR /workspace/Dynamic_ExpA/ExpA_sys

# Copy dependency metadata first to preserve reusable build layers.
COPY pyproject.toml README.md README.verl.md LICENSE Notice.txt ./
COPY verl ./verl

# PyTorch cu129
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_DYAD}/bin/python \
    "torch==2.9.0+cu129" "torchvision==0.24.0+cu129" "torchaudio==2.9.0+cu129" \
    --index-url https://download.pytorch.org/whl/cu129

# Build tools are required when flash-attn compiles without build isolation.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_DYAD}/bin/python setuptools wheel packaging ninja

# vLLM + Ray
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_DYAD}/bin/python \
    "vllm==0.12.0" "ray[default]>=2.41.0" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_DYAD}/bin/python \
    "flash-attn==2.8.3" --no-build-isolation \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_DYAD}/bin/python -e ".[vllm]" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_DYAD}/bin/python alfworld==0.4.2 gymnasium dill \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --index-strategy unsafe-best-match \
    && (uv pip uninstall --python ${VENV_DYAD}/bin/python opencv-python || true)

# =============================================================================
# 2) agentenv-alfworld（CPU：alfworld + agentenv_alfworld）
# =============================================================================
WORKDIR /build/alfworld

# Keep opencv-python-headless as the sole cv2 provider.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_ALF}/bin/python alfworld==0.3.3
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip uninstall --python ${VENV_ALF}/bin/python opencv-python || true

COPY ops/env_deps/alfworld/requirements.txt /tmp/alfworld-requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_ALF}/bin/python -r /tmp/alfworld-requirements.txt

WORKDIR /build/codegym

COPY ops/env_deps/codegym/requirements.txt ./online_server/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_CG}/bin/python \
    -r online_server/requirements.txt \
    gymnasium openai pandas pyarrow numpy tqdm "huggingface_hub[cli]"

RUN set -e; \
    echo "== builder: verl =="; ${VENV_DYAD}/bin/python -c "import verl; print('verl', verl.__file__)"; \
    echo "== builder: agentenv =="; ${VENV_ALF}/bin/python -c "import alfworld, fastapi; print('alfworld', alfworld.__file__)"

# Remove editable project installation so PYTHONPATH selects the mounted source.
RUN uv pip uninstall --python ${VENV_DYAD}/bin/python verl || true

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=120

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=4

ENV HF_HOME=/workspace/.cache/huggingface
ENV RAY_TMPDIR=/workspace/.runtime/ray_tmp
ENV TMPDIR=/workspace/.runtime/tmp
ENV TOKENIZERS_PARALLELISM=false
ENV VLLM_USE_V1=1
ENV WANDB_MODE=offline
ENV ALFWORLD_DATA=/workspace/.cache/alfworld

ENV VENV_DYAD=/opt/venv-dyad
ENV VENV_ALF=/opt/venv-alfworld
ENV VENV_CG=/opt/venv-codegym

RUN mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" \
    && apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential ninja-build cmake \
    lsof \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy environments and managed interpreters to identical paths to preserve interpreter links.
COPY --from=builder /root/.local/share/uv/python /root/.local/share/uv/python
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/
COPY --from=builder /opt/venv-dyad     /opt/venv-dyad
COPY --from=builder /opt/venv-alfworld /opt/venv-alfworld
COPY --from=builder /opt/venv-codegym  /opt/venv-codegym

RUN mkdir -p "${HF_HOME}" "${RAY_TMPDIR}" "${TMPDIR}" "${ALFWORLD_DATA}"

# Check runtime dependencies that do not require mounted project source.
RUN set -e; \
    echo "== ExpA_sys =="; ${VENV_DYAD}/bin/python -c "import torch, vllm, ray, flash_attn; print('torch', torch.__version__, torch.version.cuda); print('vllm', vllm.__version__); print('flash_attn', flash_attn.__version__)"; \
    echo "== ExpA_sys/alfworld（进程内 env，必须在 venv-dyad 里）=="; ${VENV_DYAD}/bin/python -c "import alfworld, textworld; print('alfworld', alfworld.__version__ if hasattr(alfworld,'__version__') else 'ok', '| textworld ok')"; \
    echo "== ExpA_sys/codegym（进程内 env，必须在 venv-dyad 里）=="; ${VENV_DYAD}/bin/python -c "import gymnasium, dill; print('gymnasium', gymnasium.__version__, '| dill ok')"; \
    echo "== alfworld =="; ${VENV_ALF}/bin/python -c "import alfworld; print('alfworld ok')"; \
    echo "== codegym  =="; ${VENV_CG}/bin/python -c "import fastapi, uvicorn, gymnasium, dill, pydantic; print('codegym ok')"; \
    echo "== 确认镜像内无源码（必须靠 -v 挂载）=="; \
    for d in /workspace/Dynamic_ExpA/ExpA_sys; do \
      [ -e "$d" ] && { echo "[FAIL] $d 不应存在于镜像内"; exit 1; } || echo "  ok: 无 $d"; \
    done; \
    ${VENV_DYAD}/bin/python -c "import importlib.util as u; assert u.find_spec('verl') is None, 'verl 不应装在镜像里'; print('  ok: venv-dyad 无烤入的 verl')"

WORKDIR /workspace/Dynamic_ExpA
CMD ["/bin/bash"]
