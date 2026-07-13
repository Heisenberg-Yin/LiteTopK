"""Hopper D512 torch op 加载器：把 hopper_src 的 C++/CUDA 文件编成 torch 扩展。

主要入口：
  * fused_ip_gqa_sparse_paged_group_indexed：TidalDecode 实际运行的 GQA
    group-indexed sparse 路径。

已退役入口：
  * fused_ip_dense：物化 dense [N,M] score，慢于 sparse 路径；wrapper 保持注释。
  * fused_ip_sparse / fused_ip_smalln_* / non-indexed GQA wrappers：不是
    TidalDecode 最优运行路径，C++ schema/impl 已注释，仅保留 Python stub 报错。

环境要求（在项目构建容器内）：
  * sm_90a（Hopper / H100）；nvcc 编译，gcc>=9 满足 torch C++ 头要求；
  * CUTLASS 头路径由 HOPPER_CUTLASS_INCLUDE 指定（默认 /opt/cutlass/include）。
"""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

_LOADED = False

_DEFAULT_CUTLASS_INCLUDE = "/opt/cutlass/include"
_CONTAINER_CUTLASS_INCLUDE = "/opt/conda/lib/python3.11/site-packages/flashinfer/data/cutlass/include"


def _retired_path(name: str):
    raise NotImplementedError(
        f"{name} is retired in this TidalDecode extract; use "
        "fused_ip_gqa_sparse_paged_group_indexed instead."
    )


@torch.no_grad()
def _smalln_ip_exact(x: torch.Tensor, base: torch.Tensor, k: int):
    scores = x @ base.t()
    return torch.topk(scores, k, dim=1, largest=True, sorted=True)


def _ensure_loaded():
    global _LOADED
    if _LOADED:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    cutlass_include = os.environ.get("HOPPER_CUTLASS_INCLUDE")
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
        name="tidal_hopper_topk_ext",
        sources=[
            os.path.join(here, "tidal_hopper_topk_torch.cu"),
            os.path.join(here, "tidal_hopper_topk_select.cu"),
            os.path.join(here, "tidal_hopper_topk_select_mb.cu"),
            os.path.join(here, "tidal_sm100_torch.cu"),
        ],
        extra_include_paths=[here, cutlass_include, deep_gemm_include],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-gencode=arch=compute_100a,code=sm_100a",
            # Default to the direct atomicAdd writeback path.  The register-row
            # queue variant is kept for extremely large k only, where candidate
            # pressure can make batched per-row flushes worthwhile.
            f"-DHOPPER_THR_REFRESH_GROUP={int(os.environ.get('HOPPER_THR_REFRESH_GROUP', '16'))}",
            f"-DHOPPER_M8_REG_ROW_QUEUE={int(os.environ.get('HOPPER_M8_REG_ROW_QUEUE', '0'))}",
            f"-DTIDAL_KV_STAGES={int(os.environ.get('TIDAL_KV_STAGES', '6'))}",
            f"-DTIDAL_WARP_QUEUE_CAP={int(os.environ.get('TIDAL_WARP_QUEUE_CAP', '32'))}",
            # 64 (default): TMA+UMMA only, refresh inlined into the math warps
            # (320-thread launch => 204-reg budget vs 168; +4-11% on every cell).
            # 128: legacy TMA+UMMA+2 spare refresh warps.
            f"-DTIDAL_SPEC_THREADS={int(os.environ.get('TIDAL_SPEC_THREADS', '64'))}",
        ],
        extra_ldflags=["-lcuda"],
        is_python_module=False,
        verbose=os.environ.get("FLASHTOPK_BUILD_VERBOSE") == "1",
    )
    _LOADED = True

# RETIRED: fused_ip_dense op is no longer registered (dense path is slower than
# sparse). The Python wrapper is kept commented for reference.
# @torch.no_grad()
# def fused_ip_dense(x: torch.Tensor, base: torch.Tensor, k: int,
#                    num_buckets: int = 64, cap: int | None = None):
#     _ensure_loaded()
#     if cap is None:
#         cap = min(max(4 * k, 8192), base.shape[0])
#     return torch.ops.tidal_hopper_topk.fused_ip_dense(x, base, k, num_buckets, cap)


@torch.no_grad()
def fused_ip_sparse(x: torch.Tensor, base: torch.Tensor, k: int,
                    num_buckets: int = 64,
                    buf_cap: int | None = None,
                    select_cap: int | None = None,
                    sample_size: int | None = None,
                    output_dtype: str | torch.dtype = "fp16"):
    """RETIRED: generic 2D/smalln sparse path is not the TidalDecode best path."""
    _retired_path("fused_ip_sparse")
@torch.no_grad()
def fused_ip_gqa_sparse(q: torch.Tensor, base: torch.Tensor, k: int,
                        num_buckets: int = 64,
                        buf_cap: int | None = None,
                        select_cap: int | None = None,
                        sample_size: int | None = None):
    """RETIRED: non-paged GQA wrapper is not registered in the best-path extract."""
    _retired_path("fused_ip_gqa_sparse")
@torch.no_grad()
def fused_ip_gqa_sparse_paged(q: torch.Tensor, kv_data: torch.Tensor, k: int,
                              num_buckets: int = 64,
                              buf_cap: int | None = None,
                              select_cap: int | None = None,
                              sample_size: int | None = None):
    """RETIRED: paged GQA without group_indices is not the TidalDecode best path."""
    _retired_path("fused_ip_gqa_sparse_paged")
@torch.no_grad()
def fused_ip_gqa_sparse_paged_group_indexed(
    q: torch.Tensor,
    kv_data: torch.Tensor,
    group_indices: torch.Tensor,
    k: int,
    num_buckets: int = 64,
    buf_cap: int | None = None,
    sample_size: int | None = None,
):
    """GQA/Tidal IP top-k from paged KV with one shared page list per KV head.

    group_indices: [Hkv, M] int32.  Query heads in the same GQA group share
    this list, so WGMMA can reuse one K tile across all query columns.
    """
    _ensure_loaded()
    if q.dim() == 4:
        if q.shape[0] != 1 or q.shape[2] != 1:
            raise ValueError(f"expected q [1,Hq,1,D], got {tuple(q.shape)}")
        q = q[0, :, 0, :]
    elif q.dim() == 3:
        if q.shape[0] != 1:
            raise ValueError(f"expected q [1,Hq,D], got {tuple(q.shape)}")
        q = q[0]
    if q.dim() != 2 or kv_data.dim() != 5 or group_indices.dim() != 2:
        raise ValueError("q must be [Hq,D], kv_data [P,2,1,Hkv,D], group_indices [Hkv,M]")
    if kv_data.shape[1] != 2 or kv_data.shape[2] != 1:
        raise ValueError("group-indexed path currently supports Tidal NHD page_size=1 only")
    hq, d = q.shape
    physical_pages, _, _, hkv, db = kv_data.shape
    gh, m = group_indices.shape
    if gh != hkv:
        raise ValueError(f"group_indices rows must match Hkv={hkv}, got {gh}")
    if d != db:
        raise ValueError(f"head_dim mismatch: q D={d}, kv_data D={db}")
    if hq % hkv != 0:
        raise ValueError(f"Hq={hq} must be divisible by Hkv={hkv}")
    group = hq // hkv
    if group < 1 or group > 8:
        raise ValueError(f"GQA group size must be in [1,8], got {group}")
    if m % 64 != 0:
        raise ValueError("logical page count M must be a multiple of 64 for TMA")
    if d % 64 != 0:
        raise ValueError("group-indexed path currently requires D to be a multiple of 64")
    if q.dtype != torch.float16 or kv_data.dtype != torch.float16:
        raise ValueError("q/kv_data must be float16")
    if group_indices.dtype != torch.int32:
        raise ValueError("group_indices must be int32")
    if not q.is_cuda or not kv_data.is_cuda or not group_indices.is_cuda:
        raise ValueError("q/kv_data/group_indices must be CUDA tensors")
    q = q.contiguous()
    kv_data = kv_data.contiguous()
    group_indices = group_indices.contiguous()

    if sample_size is None:
        sample_size = min(m, max(k, 131072 if hq <= 8 and k <= 128 else 65536))
    if buf_cap is None:
        # The indexed sparse path stores a compact candidate list and the final
        # selector scans the actual candidate count (qcount).  Keep the physical
        # candidate capacity at M by default so small samples do not lose recall
        # from pos < BUF / boundary CAP truncation.
        buf_cap = m
    return torch.ops.tidal_hopper_topk.fused_ip_gqa_sparse_paged_group_indexed(
        q, kv_data, group_indices, k, num_buckets, buf_cap, sample_size)


@torch.no_grad()
def fused_ip_gqa_sparse_b200(
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
    """B200 (SM100 UMMA/TMEM) GQA IP top-k over a contiguous per-head KV cache.

    q:       [Hq, D] fp16;  k_cache: [Hkv, M, D] fp16 (contiguous per KV head,
    i.e. the paged NHD page_size=1 layout with identity group_indices).
    Mirrors the Hopper fused_ip_gqa_sparse_paged_group_indexed pipeline with the
    scan replaced by the Blackwell warp-specialized kernel.
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
    return torch.ops.tidal_sm100.fused_ip_gqa_sparse_b200(
        q.contiguous(), k_cache.contiguous(), k, num_buckets, buf_cap,
        sample_size, refresh_every, num_ctas_x, sample_mode, qn, bm, out_fp32)


@torch.no_grad()
def fused_ip_anydim(
    x: torch.Tensor,
    base: torch.Tensor,
    k: int,
    *,
    use_hopper_d512: bool = True,
    chunk_size: int = 131_072,
    **hopper_kwargs,
):
    """Inner-product top-k for arbitrary feature dimensions.

    D<=512 can use the Hopper sparse fast path. Other dimensions use an exact
    chunked CUDA matmul+topk fallback that avoids materializing the full [N, M]
    score matrix.
    """
    if x.dim() != 2 or base.dim() != 2:
        raise ValueError("x/base must be 2-D")
    if x.shape[1] != base.shape[1]:
        raise ValueError(f"dimension mismatch: x D={x.shape[1]} base D={base.shape[1]}")
    if not x.is_cuda or not base.is_cuda:
        raise ValueError("x/base must be CUDA tensors")
    if not x.is_contiguous() or not base.is_contiguous():
        x = x.contiguous()
        base = base.contiguous()
    if x.dtype != torch.float16 or base.dtype != torch.float16:
        raise ValueError("x/base must be float16")
    if k < 1 or k > base.shape[0]:
        raise ValueError(f"require 1 <= k <= base rows, got k={k}")

    if use_hopper_d512 and x.shape[1] <= 768 and base.shape[0] % 64 == 0:
        _retired_path("fused_ip_anydim(use_hopper_d512=True)")

    n, m = x.shape[0], base.shape[0]
    chunk_size = max(k, min(int(chunk_size), m))
    top_vals = None
    top_idx = None

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        scores = x @ base[start:end].t()
        vals, idx = torch.topk(scores, k=min(k, end - start), dim=1, largest=True, sorted=False)
        idx = idx + start

        if top_vals is None:
            top_vals = vals
            top_idx = idx
        else:
            merged_vals = torch.cat([top_vals, vals], dim=1)
            merged_idx = torch.cat([top_idx, idx], dim=1)
            top_vals, pos = torch.topk(merged_vals, k=k, dim=1, largest=True, sorted=False)
            top_idx = torch.gather(merged_idx, 1, pos)

    top_vals, order = torch.sort(top_vals, dim=1, descending=True)
    top_idx = torch.gather(top_idx, 1, order)
    return top_vals, top_idx
