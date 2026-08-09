#!/bin/sh
# Build-time shared-memory budget probe for the DSv4 bf16 kernel.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/data01/home/ziqi.yin/vllm026-venv/bin/python}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
FLASHINFER_DATA=$(
  "$PYTHON_BIN" -c \
    'from pathlib import Path; import flashinfer; print(Path(flashinfer.__file__).parent / "data")'
)

"$CUDA_HOME/bin/nvcc" \
  -I"$SCRIPT_DIR" \
  -I"$FLASHINFER_DATA/cccl/cub" \
  -I"$FLASHINFER_DATA/cccl/libcudacxx/include" \
  -I"$FLASHINFER_DATA/cccl/thrust" \
  -I"$FLASHINFER_DATA/cutlass/include" \
  -I"$FLASHINFER_DATA/cutlass/tools/util/include" \
  -I"$FLASHINFER_DATA/include" \
  -I"$FLASHINFER_DATA/csrc" \
  -I"$FLASHINFER_DATA/spdlog/include" \
  --expt-relaxed-constexpr --expt-extended-lambda -std=c++17 -O1 \
  -gencode=arch=compute_100a,code=sm_100a \
  -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 \
  "$SCRIPT_DIR/probe_dsv4_smem.cu" -o /tmp/probe_dsv4_smem
/tmp/probe_dsv4_smem
