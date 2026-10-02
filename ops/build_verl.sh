#!/usr/bin/env bash
# Build the training dependency image; PUSH=1 additionally publishes it.
# Use an existing Docker login or environment-supplied credentials.
# Runtime source comes from a mount or a checkout; dependency changes require a new image.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE_TAG="${IMAGE_TAG:-pingchuan03/dynamic-expa-verl:latest}"

# Build for the configured GPU architectures; override explicitly for a narrower target.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;9.0}"

# Bound compilation parallelism to leave CPU capacity for the host.
MAX_JOBS="${MAX_JOBS:-$(( $(nproc) > 8 ? $(nproc) - 4 : 4 ))}"

export DOCKER_BUILDKIT=1

echo "[INFO] 镜像 tag           ：${IMAGE_TAG}"
echo "[INFO] 上下文目录         ：$(pwd)"
echo "[INFO] TORCH_CUDA_ARCH_LIST：${TORCH_CUDA_ARCH_LIST}"
echo "[INFO] MAX_JOBS           ：${MAX_JOBS}"

docker build \
    -f ops/Dockerfile.dyad-verl \
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
    --build-arg "MAX_JOBS=${MAX_JOBS}" \
    -t "${IMAGE_TAG}" \
    .

echo "[INFO] 构建完成：${IMAGE_TAG}"

echo "[INFO] 版本核对……"
docker run --rm "${IMAGE_TAG}" /opt/venv-expa-verl/bin/python -c "
import torch, vllm, numpy, sys
expected = {'torch': '2.11.0', 'vllm': '0.24.0'}
actual = {'torch': torch.__version__.split('+')[0], 'vllm': vllm.__version__}
print('  torch', torch.__version__, '| cuda', torch.version.cuda)
print('  vllm ', vllm.__version__)
print('  numpy', numpy.__version__)
bad = [k for k in expected if actual[k] != expected[k]]
if bad:
    print('  [FAIL] 版本不符：', {k: (expected[k], actual[k]) for k in bad}); sys.exit(1)
if int(numpy.__version__.split('.')[0]) < 2:
    print('  [FAIL] numpy 被降到 1.x（verl 0.9 要求 >= 2）'); sys.exit(1)
print('  [ok] 版本核对通过')
"

if [ "${PUSH:-0}" != "1" ]; then
    echo "[INFO] 未推送（要推送：PUSH=1 bash ops/build_verl.sh）"
    echo "[INFO] 启动容器（需要 GPU）："
    echo "       docker run --rm -it --gpus all -v \$(pwd):/workspace/dynamic-expa ${IMAGE_TAG}"
    exit 0
fi

if [ -n "${DOCKERHUB_USER:-}" ] && [ -n "${DOCKERHUB_TOKEN:-}" ]; then
    echo "[INFO] 用 DOCKERHUB_USER/DOCKERHUB_TOKEN 登录"
    # Use password-stdin to keep credentials out of process arguments.
    printf '%s' "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USER}" --password-stdin
else
    echo "[INFO] 未设置 DOCKERHUB_USER/DOCKERHUB_TOKEN，沿用已有的 docker login 会话"
fi

docker push "${IMAGE_TAG}"
echo "[INFO] 已推送：${IMAGE_TAG}"

# Derive the immutable tag from image creation time so repeated publication keeps its identity.
IMMUTABLE_SUFFIX="${IMMUTABLE_SUFFIX:-$(docker image inspect "${IMAGE_TAG}" --format '{{.Created}}' | cut -c1-10 | tr -d -)}"
IMMUTABLE_TAG="${IMMUTABLE_TAG:-${IMAGE_TAG%%:*}:verl090-vllm024-cu130-${IMMUTABLE_SUFFIX}}"
docker tag "${IMAGE_TAG}" "${IMMUTABLE_TAG}"
docker push "${IMMUTABLE_TAG}"
echo "[INFO] 已推送不可变标签：${IMMUTABLE_TAG}"
echo "[INFO] 更新 lucia_job/ 活跃 YAML、模板和 Docker 默认镜像，并检查 YAML 与入口一致性。"
