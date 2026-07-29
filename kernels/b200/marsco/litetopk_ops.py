"""JIT loader and Python API for the B200 MS MARCO inner-product kernel.

The first call builds a PyTorch extension from ``litetopk_select.cu`` and
``litetopk_sm100_torch.cu``. Build-time kernel constants are fixed below so
the same source tree produces the same specialization on every invocation.

Required include paths can be supplied with ``LITETOPK_CUTLASS_INCLUDE`` and
``DSA_DEEP_GEMM_INCLUDE``. See ``README.md`` for a complete command.
"""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

_LOADED = False

_DEFAULT_CUTLASS_INCLUDE = "/opt/cutlass/include"
_CONTAINER_CUTLASS_INCLUDE = "/opt/conda/lib/python3.11/site-packages/flashinfer/data/cutlass/include"


def _ensure_loaded():
    global _LOADED
    if _LOADED:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    cutlass_include = os.environ.get("LITETOPK_CUTLASS_INCLUDE")
    if cutlass_include is None:
        cutlass_include = (
            _DEFAULT_CUTLASS_INCLUDE
            if os.path.exists(os.path.join(_DEFAULT_CUTLASS_INCLUDE, "cute/arch/mma_sm100_umma.hpp"))
            else _CONTAINER_CUTLASS_INCLUDE
        )
    cutlass_marker = os.path.join(
        cutlass_include, "cute/arch/mma_sm100_umma.hpp"
    )
    if not os.path.isfile(cutlass_marker):
        raise FileNotFoundError(
            "CUTLASS SM100 headers not found; set LITETOPK_CUTLASS_INCLUDE"
        )
    deep_gemm_include = os.environ.get("DSA_DEEP_GEMM_INCLUDE")
    if not deep_gemm_include:
        raise RuntimeError(
            "set DSA_DEEP_GEMM_INCLUDE to the compatible DeepGEMM include directory"
        )
    deep_gemm_marker = os.path.join(
        deep_gemm_include, "deep_gemm/common/sm100_utils.cuh"
    )
    if not os.path.isfile(deep_gemm_marker):
        raise FileNotFoundError(
            f"DeepGEMM SM100 headers not found under {deep_gemm_include}"
        )
    load(
        name="litetopk_marsco_ext",
        sources=[
            os.path.join(here, "litetopk_select.cu"),
            os.path.join(here, "litetopk_sm100_torch.cu"),
        ],
        extra_include_paths=[here, cutlass_include, deep_gemm_include],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DLITETOPK_KV_STAGES=6",
            "-DLITETOPK_WARP_QUEUE_CAP=32",
        ],
        extra_ldflags=["-lcuda"],
        is_python_module=False,
        verbose=os.environ.get("FLASHTOPK_BUILD_VERBOSE") == "1",
    )
    _LOADED = True


@torch.no_grad()
def fused_ip_sparse_b200(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    k: int,
    num_buckets: int = 64,
    buf_cap: int | None = None,
    sample_size: int | None = None,
    refresh_every: int = 0,
    num_ctas_x: int = 0,
    sample_mode: int = 1,
    qn: int = 0,
    bm: int = 0,
    out_fp32: bool = False,
):
    """Run B200 inner-product selection over a contiguous KV tensor.

    ``q`` must be ``[B, D]`` fp16 and ``k_cache`` must be ``[1, M, D]``
    fp16. ``B`` must be greater than 8 and compatible with the selected query
    tile; ``M`` must be a multiple of 64. The public contract supports only
    ``sample_mode=1`` (strided corpus sampling). The returned tuple contains
    values followed by int32 corpus indices.
    """
    if sample_mode != 1:
        raise ValueError("sample_mode must be 1 (strided corpus sampling)")
    if q.dim() == 4:
        q = q[0, :, 0, :]
    elif q.dim() == 3:
        q = q[0]
    if q.dim() != 2 or k_cache.dim() != 3:
        raise ValueError("q must be [Hq,D], k_cache [Hkv,M,D]")
    hq, d = q.shape
    hkv, m, db = k_cache.shape
    if hkv < 1 or d != db or hq % hkv != 0:
        raise ValueError(f"bad shapes: q {tuple(q.shape)}, k_cache {tuple(k_cache.shape)}")
    if hkv != 1 or hq <= 8:
        raise ValueError(
            "MS MARCO flat-batch mode requires "
            "k_cache.shape[0] == 1 and q.shape[0] > 8"
        )
    if m % 64 != 0:
        raise ValueError("k_cache.shape[1] must be a multiple of 64")
    if not q.is_cuda or not k_cache.is_cuda or q.device != k_cache.device:
        raise ValueError("q and k_cache must be CUDA tensors on the same device")
    if q.dtype != torch.float16 or k_cache.dtype != torch.float16:
        raise ValueError("q/k_cache must be float16")
    if qn not in (0, 8, 64):
        raise ValueError("qn must be 0, 8, or 64")
    if bm not in (0, 128, 256):
        raise ValueError("bm must be 0, 128, or 256")
    if sample_size is None:
        sample_size = min(m, max(k, 65536))
    if buf_cap is None:
        buf_cap = m
    _ensure_loaded()
    return torch.ops.litetopk_sm100.fused_ip_sparse_b200(
        q.contiguous(),
        k_cache.contiguous(),
        k,
        num_buckets,
        buf_cap,
        sample_size,
        refresh_every,
        num_ctas_x,
        sample_mode,
        qn,
        bm,
        out_fp32,
    )
