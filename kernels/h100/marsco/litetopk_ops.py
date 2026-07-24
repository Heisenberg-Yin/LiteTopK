"""JIT loader + Python API for the H100 (SM90) LiteTopK MS MARCO IP top-k kernel.

Builds one torch extension from `litetopk_select.cu` (FlashTopk select engine) +
`litetopk_sm90_torch.cu` (the SM90 WGMMA sparse-scan host binding, H100 port) and exposes
the main entry `fused_ip_sparse_h100(...)` = `torch.ops.litetopk_sm90.fused_ip_sparse_h100`.

Build env (project build container):
  * sm_90a (H100); nvcc.
  * CUTLASS headers via LITETOPK_CUTLASS_INCLUDE; deep_gemm headers via
    DSA_DEEP_GEMM_INCLUDE.
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
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    cutlass_include = os.environ.get("LITETOPK_CUTLASS_INCLUDE")
    if cutlass_include is None:
        cutlass_include = (
            _DEFAULT_CUTLASS_INCLUDE
            if os.path.exists(os.path.join(_DEFAULT_CUTLASS_INCLUDE, "cute/arch/mma_sm100_umma.hpp"))
            else _CONTAINER_CUTLASS_INCLUDE
        )
    deep_gemm_include = os.environ.get(
        "DSA_DEEP_GEMM_INCLUDE",
        "/opt/venvs/deepgemm/lib/python3.12/site-packages/deep_gemm/include",
    )
    load(
        name="litetopk_marsco_sm90_ext",
        sources=[
            os.path.join(here, "litetopk_select.cu"),
            os.path.join(here, "litetopk_sm90_torch.cu"),
        ],
        extra_include_paths=[here, cutlass_include, deep_gemm_include],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-gencode=arch=compute_90a,code=sm_90a",
            # Kernel tuning knobs (override via env). SPEC_THREADS is now
            # hardcoded to 64 in the kernel; the dead THR_REFRESH_GROUP /
            # M8_REG_ROW_QUEUE toggles were removed.
            f"-DLITETOPK_KV_STAGES={int(os.environ.get('LITETOPK_KV_STAGES', '6'))}",
            f"-DLITETOPK_WARP_QUEUE_CAP={int(os.environ.get('LITETOPK_WARP_QUEUE_CAP', '32'))}",
        ],
        extra_ldflags=["-lcuda"],
        is_python_module=False,
        verbose=os.environ.get("FLASHTOPK_BUILD_VERBOSE") == "1",
    )
    _LOADED = True

@torch.no_grad()
def fused_ip_sparse_h100(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    k: int,
    num_buckets: int = 64,
    buf_cap: int | None = None,
    sample_size: int | None = None,
    refresh_every: int = 0,
    num_ctas_x: int = 0,
    sample_mode: int = 0,
    qn: int = 0,
    bm: int = 0,
    out_fp32: bool = False,
):
    """B200 (SM100 UMMA/TMEM) IP top-k over a contiguous per-head KV cache.

    q: [Hq, D] fp16;  k_cache: [Hkv, M, D] fp16 (contiguous per KV head).
    Pipeline: sample -> gate (bucket threshold) -> SM90 warp-specialized
    sparse scan -> boundary select. Recall exact by construction.
    """
    _ensure_loaded()
    if q.dim() == 4:
        q = q[0, :, 0, :]
    elif q.dim() == 3:
        q = q[0]
    if q.dim() != 2 or k_cache.dim() != 3:
        raise ValueError("q must be [Hq,D], k_cache [Hkv,M,D]")
    hq, d = q.shape
    hkv, m, db = k_cache.shape
    if d != db or hq % hkv != 0:
        raise ValueError(f"bad shapes: q {tuple(q.shape)}, k_cache {tuple(k_cache.shape)}")
    if q.dtype != torch.float16 or k_cache.dtype != torch.float16:
        raise ValueError("q/k_cache must be float16")
    if sample_size is None:
        sample_size = min(m, max(k, 131072 if hq <= 8 and k <= 128 else 65536))
    if buf_cap is None:
        buf_cap = m
    return torch.ops.litetopk_sm90.fused_ip_sparse_h100(
        q.contiguous(), k_cache.contiguous(), k, num_buckets, buf_cap,
        sample_size, refresh_every, num_ctas_x, sample_mode, qn, bm, out_fp32)
