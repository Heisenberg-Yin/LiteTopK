#!/usr/bin/env bash
# Launch (or reuse) the `simtopk` B200 container and drop into it, or run a command.
#
# The container is CUDA 12.8 (nvcc supports sm_100a), GPU-passthrough (--gpus all),
# with torch/ninja/safetensors preinstalled and the internal pip/HF mirrors set.
# Data disks /data00 /data01 /data02 /data07 are bind-mounted so CUTLASS 4.2.1,
# the SM100 deep_gemm, and this source tree are all visible inside.
#
# Usage:
#   ./run_in_docker.sh                 # interactive shell in the container
#   ./run_in_docker.sh python3 test_dsa.py 256k
set -euo pipefail

IMAGE="simtopk:latest"
NAME="simtopk"
WORKDIR="/opt/simtopk_src/b200/dsa"

# Create the container if it is not running.
if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  # Remove a stopped one with the same name, if any.
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=16g \
    -v /data01:/data01 -v /data00:/data00 -v /data02:/data02 -v /data07:/data07 \
    -e PIP_INDEX_URL=https://pypi.org/simple \
    -e PIP_TRUSTED_HOST=pypi.org \
    -e HF_ENDPOINT=https://huggingface.co \
    -w "$WORKDIR" \
    "$IMAGE" sleep infinity >/dev/null
fi

if [ "$#" -eq 0 ]; then
  exec docker exec -it -w "$WORKDIR" "$NAME" bash
else
  exec docker exec -w "$WORKDIR" "$NAME" "$@"
fi
