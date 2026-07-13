#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared B200 (SM100 / sm_100a) build + path configuration for the DSA port.

All DSA driver scripts (test_dsa.py, bench_3way.py, sweep_config.py) import this
module so the CUTLASS / deep_gemm include paths and nvcc arch flags live in one
place. Override any path via the matching environment variable.

Defaults were discovered on this host:
  * CUTLASS 4.2.1 (full SM100 / tcgen05 UMMA support):
        /opt/cutlass/include
  * deep_gemm with SM100 kernels (sm100_fp8_mqa_logits etc.):
        /opt/venvs/deepgemm/lib/python3.12/site-packages/deep_gemm
"""
from __future__ import annotations

import os
import sys

# --- deep_gemm (provides fp8_mqa_logits reference + SM100 headers) ------------
DEEP_GEMM_ROOT = os.environ.get(
    "DSA_DEEP_GEMM_ROOT",
    "/opt/venvs/deepgemm/lib/python3.12/site-packages/deep_gemm",
)
# deep_gemm is a site-packages module; make sure its parent is importable.
_DG_PARENT = os.path.dirname(os.path.dirname(DEEP_GEMM_ROOT))
if os.path.isdir(_DG_PARENT) and _DG_PARENT not in sys.path:
    sys.path.insert(0, _DG_PARENT)

DEEP_GEMM_INCLUDE = os.environ.get(
    "DSA_DEEP_GEMM_INCLUDE", os.path.join(DEEP_GEMM_ROOT, "include")
)

# --- CUTLASS 4.2.1 (SM100 tcgen05 UMMA) --------------------------------------
CUTLASS_INCLUDE = os.environ.get(
    "DSA_CUTLASS_INCLUDE", "/opt/cutlass/include"
)

# --- nvcc arch --------------------------------------------------------------
# B200 is compute capability 10.0; sm_100a enables the tcgen05 / UMMA / TMEM
# instructions the kernel uses.
TORCH_CUDA_ARCH = os.environ.get("DSA_TORCH_CUDA_ARCH", "10.0a")
GENCODE = os.environ.get("DSA_GENCODE", "arch=compute_100a,code=sm_100a")

DSA_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(DSA_DIR, "dsa_marsco.cu")

INC = [DSA_DIR, DEEP_GEMM_INCLUDE, CUTLASS_INCLUDE]
FLAGS = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    f"-gencode={GENCODE}",
]


def prepare_env():
    """Set TORCH_CUDA_ARCH_LIST for the JIT build if unset."""
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", TORCH_CUDA_ARCH)


def cache_path(tag: str) -> str:
    """DSA cache location. Override via DSA_CACHE_DIR / DSA_CACHE_GLOB."""
    cache_dir = os.environ.get("DSA_CACHE_DIR", "/workspace/project/dsa_caches")
    chunk = int(os.environ.get("CHUNK", "4096"))
    suffix = f"_chunk{chunk}" if chunk > 0 else ""
    return os.path.join(cache_dir, f"glm5_{tag}_realtext{suffix}.safetensors")
