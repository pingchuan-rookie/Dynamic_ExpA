#!/usr/bin/env bash
# Build the dependency image from the repository root; IMAGE_TAG overrides its tag.
set -euo pipefail

# Docker COPY paths are relative to the repository root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE_TAG="${IMAGE_TAG:-dynamic-dyad:latest}"

export DOCKER_BUILDKIT=1

echo "[INFO] 构建镜像：${IMAGE_TAG}"
echo "[INFO] 上下文目录：$(pwd)"

docker build -t "${IMAGE_TAG}" -f ops/Dockerfile .

echo "[INFO] 构建完成：${IMAGE_TAG}"
echo "[INFO] 启动容器（需要 GPU）："
echo "       docker run --rm -it --gpus all ${IMAGE_TAG}"
