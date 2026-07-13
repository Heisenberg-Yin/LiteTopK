"""Hopper D512 torch op 加载器：把 hopper_src 的 C++/CUDA 文件编成 torch 扩展。

包含两类入口：
  * fused_ip_dense：Hopper WGMMA score-to-dense(-IP) + C++ dense threshold
    + C++ multi-CTA topk_select_thr_mb_idx，返回 IP topv/topi。
  * fused_ip_sparse：sample 前缀定桶，WGMMA tail 稀疏 epilogue 直接写 bv/bi/qc，
    再走 C++ multi-CTA select。

环境要求（simtopk 容器）：
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
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    cutlass_include = os.environ.get("HOPPER_CUTLASS_INCLUDE")
    if cutlass_include is None:
        cutlass_include = (
            _DEFAULT_CUTLASS_INCLUDE
            if os.path.exists(os.path.join(_DEFAULT_CUTLASS_INCLUDE, "cute/arch/mma_sm90_gmma.hpp"))
            else _CONTAINER_CUTLASS_INCLUDE
        )
    load(
        name="hopper_topk",
        sources=[
            os.path.join(here, "hopper_topk_torch.cu"),
            os.path.join(here, "hopper_topk_select.cu"),
            os.path.join(here, "hopper_topk_select_mb.cu"),
        ],
        extra_include_paths=[here, cutlass_include],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-gencode=arch=compute_90a,code=sm_90a",
            f"-DHOPPER_THR_REFRESH_GROUP={int(os.environ.get('HOPPER_THR_REFRESH_GROUP', '8'))}",
            f"-DHOPPER_SPARSE_NO_HIST={int(os.environ.get('HOPPER_SPARSE_NO_HIST', '0'))}",
            f"-DHOPPER_SPARSE_NOSTORE_HIST={int(os.environ.get('HOPPER_SPARSE_NOSTORE_HIST', '0'))}",
            f"-DHOPPER_SPARSE_FAKE_HIST_STORE={int(os.environ.get('HOPPER_SPARSE_FAKE_HIST_STORE', '0'))}",
            f"-DHOPPER_SPARSE_MERGE_HIST={int(os.environ.get('HOPPER_SPARSE_MERGE_HIST', '0'))}",
            f"-DHOPPER_SPARSE_NO_WRITE={int(os.environ.get('HOPPER_SPARSE_NO_WRITE', '0'))}",
            f"-DHOPPER_SPARSE_WARP_BATCH_WRITE={int(os.environ.get('HOPPER_SPARSE_WARP_BATCH_WRITE', '0'))}",
            f"-DHOPPER_GQA_WARP_COMPACT_WRITE={int(os.environ.get('HOPPER_GQA_WARP_COMPACT_WRITE', '1'))}",
            f"-DHOPPER_SPARSE_BUCKET_WRITE={int(os.environ.get('HOPPER_SPARSE_BUCKET_WRITE', '0'))}",
            f"-DHOPPER_SPARSE_BCOUNT_SHARDS={int(os.environ.get('HOPPER_SPARSE_BCOUNT_SHARDS', '1'))}",
            f"-DHOPPER_SPARSE_DENSE_WRITE={int(os.environ.get('HOPPER_SPARSE_DENSE_WRITE', '0'))}",
            f"-DHOPPER_SPARSE_REFRESH_FROM_BUF={int(os.environ.get('HOPPER_SPARSE_REFRESH_FROM_BUF', '0'))}",
            f"-DHOPPER_SPARSE_REFRESH_ROWS={int(os.environ.get('HOPPER_SPARSE_REFRESH_ROWS', '4'))}",
            f"-DHOPPER_RFB_FLUSH_SKIP_ABOVE_GATE={int(os.environ.get('HOPPER_RFB_FLUSH_SKIP_ABOVE_GATE', '1'))}",
            f"-DHOPPER_RFB_HIST_MATRIX_ADD={int(os.environ.get('HOPPER_RFB_HIST_MATRIX_ADD', '0'))}",
            f"-DHOPPER_RFB_GLOBAL_BATCH_ADD={int(os.environ.get('HOPPER_RFB_GLOBAL_BATCH_ADD', '1'))}",
            f"-DHOPPER_RFB_CANDIDATE_PARALLEL={int(os.environ.get('HOPPER_RFB_CANDIDATE_PARALLEL', '0'))}",
            f"-DHOPPER_RFB_EARLY_UNLOCK={int(os.environ.get('HOPPER_RFB_EARLY_UNLOCK', '1'))}",
            f"-DHOPPER_RFB_PRODUCER_ONLY={int(os.environ.get('HOPPER_RFB_PRODUCER_ONLY', '0'))}",
            f"-DHOPPER_RFB_FLATTEN_ONE_SYNC={int(os.environ.get('HOPPER_RFB_FLATTEN_ONE_SYNC', '0'))}",
            f"-DHOPPER_RFB_FLATTEN_PARALLEL_FLUSH={int(os.environ.get('HOPPER_RFB_FLATTEN_PARALLEL_FLUSH', '0'))}",
            f"-DHOPPER_SPARSE_SETMAXNREG={int(os.environ.get('HOPPER_SPARSE_SETMAXNREG', '0'))}",
            f"-DHOPPER_SPARSE_PRODUCER_REGS={int(os.environ.get('HOPPER_SPARSE_PRODUCER_REGS', '32'))}",
            f"-DHOPPER_SPARSE_CONSUMER_REGS={int(os.environ.get('HOPPER_SPARSE_CONSUMER_REGS', '232'))}",
            f"-DHOPPER_KCHUNK_RING={int(os.environ.get('HOPPER_KCHUNK_RING', '12'))}",
            f"-DHOPPER_KCHUNK_PIPE={int(os.environ.get('HOPPER_KCHUNK_PIPE', '1'))}",
            f"-DHOPPER_KCHUNK_FORCE={int(os.environ.get('HOPPER_KCHUNK_FORCE', '0'))}",
            f"-DHOPPER_KCHUNK_MATH_WG={int(os.environ.get('HOPPER_KCHUNK_MATH_WG', '3'))}",
            f"-DHOPPER_M8_REG_ROW_QUEUE={int(os.environ.get('HOPPER_M8_REG_ROW_QUEUE', '0'))}",
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
#     return torch.ops.hopper_topk.fused_ip_dense(x, base, k, num_buckets, cap)


@torch.no_grad()
def fused_ip_sparse(x: torch.Tensor, base: torch.Tensor, k: int,
                    num_buckets: int = 64,
                    buf_cap: int | None = None,
                    select_cap: int | None = None,
                    sample_size: int | None = None,
                    output_dtype: str | torch.dtype = "fp16"):
    """端到端 C++/CUDA IP top-k 稀疏 buffer 版。

    sample 前缀只用于定桶并 seed `bv/bi/qc`；主扫描 tail 的 WGMMA epilogue
    直接做 bucket/hist/threshold/update/write，不物化全量 dense score。D<=512
    会在 C++ wrapper 中补到最近的 64 倍数并只运行对应数量的 K stage。
    """
    _ensure_loaded()
    m = base.shape[0]
    n = x.shape[0]
    want_fp32 = output_dtype in ("fp32", "float32", torch.float32)
    if sample_size is None:
        if n == 1 and k <= 128:
            sample_size = m
        elif n <= 8 and k <= 128:
            sample_size = min(m, max(k, 131072))
        else:
            sample_size = min(m, max(k, 65536))
    if buf_cap is None:
        buf_cap = min(max(64 * k, 12800), m)
    if select_cap is None:
        select_cap = min(max(4 * k, 8192), buf_cap)
    d_ok = (x.shape[1] == base.shape[1] and 1 <= x.shape[1] <= 512)
    # smalln (N<=32) m64n32 path supports up to D=768 via the K-chunk kernel.
    d_ok_smalln = (x.shape[1] == base.shape[1] and 1 <= x.shape[1] <= 768)
    if n <= 32:
        # Default batch<=32 to the m64n32 fully-fused sparse path (smaller 64x32
        # tile, streams base once). Padding rows are filled with tiled real
        # queries in the C++ wrapper, so 8/16/32 all run at full speed.
        if (os.environ.get("HOPPER_SMALLN_SPARSE", "1") == "1"
                and d_ok_smalln and base.shape[0] % 64 == 0):
            _ensure_loaded()
            # Keep sample sizing identical to the batch>=64 path; the smalln
            # tail now refreshes thresholds from bcount during the scan.
            sn_sample = sample_size
            sn_buf = min(m, max(buf_cap, 64 * k))
            if want_fp32:
                return torch.ops.hopper_topk.fused_ip_smalln_sparse_fp32buf(
                    x, base, k, num_buckets, sn_buf, select_cap, sn_sample)
            return torch.ops.hopper_topk.fused_ip_smalln_sparse(
                x, base, k, num_buckets, sn_buf, select_cap, sn_sample)
        return _smalln_ip_exact(x, base, k)
    if n % 64 == 0:
        if want_fp32:
            return torch.ops.hopper_topk.fused_ip_sparse_fp32buf(
                x, base, k, num_buckets, buf_cap, select_cap, sample_size)
        return torch.ops.hopper_topk.fused_ip_sparse(
            x, base, k, num_buckets, buf_cap, select_cap, sample_size)
    if want_fp32:
        raise NotImplementedError("output_dtype='fp32' requires batch <= 32 or a multiple of 64")

    padded_n = ((n + 63) // 64) * 64
    x_padded = torch.empty((padded_n, x.shape[1]), device=x.device, dtype=x.dtype)
    x_padded[:n].copy_(x)
    x_padded[n:].zero_()
    val, idx = torch.ops.hopper_topk.fused_ip_sparse(
        x_padded, base, k, num_buckets, buf_cap, select_cap, sample_size)
    return val[:n], idx[:n]


@torch.no_grad()
def fused_ip_gqa_sparse(q: torch.Tensor, base: torch.Tensor, k: int,
                        num_buckets: int = 64,
                        buf_cap: int | None = None,
                        select_cap: int | None = None,
                        sample_size: int | None = None):
    """GQA/Tidal-style per-head-base IP top-k.

    q: [Hq, D], [1, Hq, D], or [1, Hq, 1, D];
    base: [Hkv, M, D] or [1, Hkv, M, D].
    A single CUDA launch per GEMM stage covers all Hkv base matrices; each
    blockIdx.y binds one KV head and uses the same value-gate-first sparse
    epilogue as fused_ip_sparse.
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
    if base.dim() == 4:
        if base.shape[0] != 1:
            raise ValueError(f"expected base [1,Hkv,M,D], got {tuple(base.shape)}")
        base = base[0]
    if q.dim() != 2 or base.dim() != 3:
        raise ValueError("q must be [Hq,D] and base must be [Hkv,M,D]")
    hq, d = q.shape
    hkv, m, db = base.shape
    if d != db:
        raise ValueError(f"head_dim mismatch: q D={d}, base D={db}")
    if hq % hkv != 0:
        raise ValueError(f"Hq={hq} must be divisible by Hkv={hkv}")
    group = hq // hkv
    if group < 1 or group > 8:
        raise ValueError(f"GQA group size must be in [1,8], got {group}")
    if m % 64 != 0:
        raise ValueError("base rows M must be a multiple of 64 for TMA")
    if q.dtype != torch.float16 or base.dtype != torch.float16:
        raise ValueError("q/base must be float16")
    if not q.is_cuda or not base.is_cuda:
        raise ValueError("q/base must be CUDA tensors")
    q = q.contiguous()
    base = base.contiguous()

    if sample_size is None:
        sample_size = min(m, max(k, 131072 if hq <= 8 and k <= 128 else 65536))
    if buf_cap is None:
        buf_cap = min(max(64 * k, 12800), m)
    if select_cap is None:
        select_cap = min(max(4 * k, 8192), buf_cap)
    return torch.ops.hopper_topk.fused_ip_gqa_sparse(
        q, base, k, num_buckets, buf_cap, select_cap, sample_size)


@torch.no_grad()
def fused_ip_gqa_sparse_paged(q: torch.Tensor, kv_data: torch.Tensor, k: int,
                              num_buckets: int = 64,
                              buf_cap: int | None = None,
                              select_cap: int | None = None,
                              sample_size: int | None = None):
    """GQA/Tidal-style IP top-k directly from Tidal NHD page storage.

    q: [Hq, D], [1, Hq, D], or [1, Hq, 1, D].
    kv_data: [M, 2, page_size=1, Hkv, D].

    This aligns the producer's source layout with Tidal's physical page storage.
    It does not apply an arbitrary page-indices gather; logical token order is
    assumed to match physical page order.
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
    if q.dim() != 2 or kv_data.dim() != 5:
        raise ValueError("q must be [Hq,D] and kv_data must be [M,2,1,Hkv,D]")
    if kv_data.shape[1] != 2 or kv_data.shape[2] != 1:
        raise ValueError("paged path currently supports Tidal NHD page_size=1 only")
    hq, d = q.shape
    m, _, _, hkv, db = kv_data.shape
    if d != db:
        raise ValueError(f"head_dim mismatch: q D={d}, kv_data D={db}")
    if hq % hkv != 0:
        raise ValueError(f"Hq={hq} must be divisible by Hkv={hkv}")
    group = hq // hkv
    if group < 1 or group > 8:
        raise ValueError(f"GQA group size must be in [1,8], got {group}")
    if m % 64 != 0:
        raise ValueError("page count M must be a multiple of 64 for TMA")
    if d % 64 != 0:
        raise ValueError("paged path currently requires D to be a multiple of 64")
    if q.dtype != torch.float16 or kv_data.dtype != torch.float16:
        raise ValueError("q/kv_data must be float16")
    if not q.is_cuda or not kv_data.is_cuda:
        raise ValueError("q/kv_data must be CUDA tensors")
    q = q.contiguous()
    kv_data = kv_data.contiguous()

    if sample_size is None:
        sample_size = min(m, max(k, 131072 if hq <= 8 and k <= 128 else 65536))
    if buf_cap is None:
        buf_cap = min(max(64 * k, 12800), m)
    if select_cap is None:
        select_cap = min(max(4 * k, 8192), buf_cap)
    return torch.ops.hopper_topk.fused_ip_gqa_sparse_paged(
        q, kv_data, k, num_buckets, buf_cap, select_cap, sample_size)


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
    return torch.ops.hopper_topk.fused_ip_gqa_sparse_paged_group_indexed(
        q, kv_data, group_indices, k, num_buckets, buf_cap, sample_size)


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
        return fused_ip_sparse(x, base, k, **hopper_kwargs)

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
