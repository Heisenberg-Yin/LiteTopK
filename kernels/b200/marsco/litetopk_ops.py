"""JIT loader and Python API for the B200 MS MARCO inner-product kernel.

The first call builds a PyTorch extension from ``litetopk_select.cu`` and
``litetopk_sm100_torch.cu``. Build-time kernel constants are fixed below so
the same source tree produces the same specialization on every invocation.

Required include paths can be supplied with ``LITETOPK_CUTLASS_INCLUDE`` and
``DSA_DEEP_GEMM_INCLUDE``. See ``../../../README.md`` for a complete command.
"""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

_LOADED = False
_LOADED_BUILD_CONFIG: tuple[str, str] | None = None
_LAST_CAP_STATS: dict[str, object] = {}
_DEFAULT_BOUNDED_CAP = 1 << 18

_DEFAULT_CUTLASS_INCLUDE = "/opt/cutlass/include"
_CONTAINER_CUTLASS_INCLUDE = "/opt/conda/lib/python3.11/site-packages/flashinfer/data/cutlass/include"


def _ensure_loaded():
    global _LOADED, _LOADED_BUILD_CONFIG
    caller_direct_workspace = os.environ.get(
        "LITETOPK_CALLER_DIRECTLT_WORKSPACE", "1"
    )
    if caller_direct_workspace not in ("0", "1"):
        raise ValueError(
            "LITETOPK_CALLER_DIRECTLT_WORKSPACE must be 0 or 1"
        )
    fp8_scan = os.environ.get("LITETOPK_MARSCO_FP8_SCAN", "0")
    if fp8_scan not in ("0", "1"):
        raise ValueError("LITETOPK_MARSCO_FP8_SCAN must be 0 or 1")
    build_config = (caller_direct_workspace, fp8_scan)
    if _LOADED:
        if _LOADED_BUILD_CONFIG != build_config:
            raise RuntimeError(
                "MARSCO compile-time environment changed after extension "
                f"load: loaded={_LOADED_BUILD_CONFIG}, requested={build_config}"
            )
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
    variant = (
        f"direct{caller_direct_workspace}_"
        f"fp8_{fp8_scan}"
    )
    load(
        name=f"litetopk_marsco_ext_{variant}",
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
            (
                "-DLITETOPK_CALLER_DIRECTLT_WORKSPACE="
                f"{caller_direct_workspace}"
            ),
            f"-DLITETOPK_MARSCO_FP8_SCAN={fp8_scan}",
        ],
        extra_ldflags=["-lcuda"],
        is_python_module=False,
        verbose=os.environ.get("FLASHTOPK_BUILD_VERBOSE") == "1",
    )
    _LOADED = True
    _LOADED_BUILD_CONFIG = build_config


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
    bounded_cap: bool | None = None,
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
        # Keep the public API on the same calibrated policy as
        # bench_marsco_b200.py. For the common K=128/512 cases this selects
        # 16K rows rather than the old 64K default, which otherwise paid four
        # times the sample GEMM/prep work and did not represent reported
        # benchmark latency. Explicit sample_size values remain unchanged.
        sample_size = min(m, max(16384, min(8 * k, 262144)))
    if bounded_cap is None:
        bounded_env = os.environ.get("LITETOPK_MARSCO_BOUNDED_CAP", "0")
        if bounded_env not in ("0", "1"):
            raise ValueError("LITETOPK_MARSCO_BOUNDED_CAP must be 0 or 1")
        bounded_cap = bounded_env == "1"
    if buf_cap is None:
        buf_cap = (
            min(m, _DEFAULT_BOUNDED_CAP)
            if bounded_cap
            else m
        )
    buf_cap = max(k, min(m, int(buf_cap)))
    _ensure_loaded()
    q_contiguous = q.contiguous()
    k_cache_contiguous = k_cache.contiguous()

    def call(capacity: int, with_counts: bool):
        op = (
            torch.ops.litetopk_sm100.fused_ip_sparse_b200_with_counts
            if with_counts
            else torch.ops.litetopk_sm100.fused_ip_sparse_b200
        )
        return op(
            q_contiguous,
            k_cache_contiguous,
            k,
            num_buckets,
            capacity,
            sample_size,
            refresh_every,
            num_ctas_x,
            sample_mode,
            qn,
            bm,
            out_fp32,
        )

    if not bounded_cap:
        _LAST_CAP_STATS.clear()
        _LAST_CAP_STATS.update(
            {
                "enabled": False,
                "initial_cap": buf_cap,
                "final_cap": buf_cap,
                "attempts": 1,
                "synchronized_for_overflow_check": False,
            }
        )
        return call(buf_cap, False)

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "bounded MARSCO candidate-cap retry is not CUDA Graph safe; "
            "warm and capture the fixed-cap operator instead"
        )

    initial_cap = buf_cap
    attempts: list[dict[str, int]] = []
    while True:
        values, indices, qcount = call(buf_cap, True)
        # Correctness requires knowing that every passing candidate was stored.
        # This is the one intentional host synchronization in bounded mode.
        max_count = int(qcount.max().item())
        attempts.append({"cap": buf_cap, "max_qcount": max_count})
        if max_count <= buf_cap:
            _LAST_CAP_STATS.clear()
            _LAST_CAP_STATS.update(
                {
                    "enabled": True,
                    "initial_cap": initial_cap,
                    "final_cap": buf_cap,
                    "attempts": len(attempts),
                    "attempt_log": attempts,
                    "synchronized_for_overflow_check": True,
                }
            )
            return values, indices
        if buf_cap == m:
            raise RuntimeError(
                f"MARSCO qcount={max_count} exceeds corpus size M={m}"
            )
        del values, indices, qcount
        required = 1 << (max_count - 1).bit_length()
        buf_cap = min(m, max(buf_cap * 2, required))


@torch.no_grad()
def quantize_e4m3_rowwise(
    x: torch.Tensor,
    *,
    chunk_rows: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the final dimension to E4M3 with one fp32 scale per row.

    The returned contract is ``x ~= q.float() * scale[..., None]``. Chunking
    bounds the temporary fp32 conversion when building a multi-million-row
    corpus cache; callers should build this cache once, outside timed calls.
    """
    if not x.is_cuda or x.dtype not in (torch.float16, torch.float32):
        raise ValueError("x must be a CUDA float16/float32 tensor")
    if x.dim() < 2 or not x.is_contiguous():
        raise ValueError("x must be contiguous with at least two dimensions")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    rows = x.reshape(-1, x.shape[-1])
    q_rows = torch.empty(
        rows.shape, dtype=torch.float8_e4m3fn, device=x.device
    )
    scales = torch.empty(
        rows.shape[0], dtype=torch.float32, device=x.device
    )
    for start in range(0, rows.shape[0], chunk_rows):
        end = min(start + chunk_rows, rows.shape[0])
        xf = rows[start:end].float()
        scale = (
            xf.abs().amax(dim=1).clamp_min_(1.0e-12).mul_(1.0 / 448.0)
        )
        q_rows[start:end].copy_(
            xf.div(scale.unsqueeze(1))
            .clamp_(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )
        scales[start:end].copy_(scale)
    return (
        q_rows.view(x.shape),
        scales.view(x.shape[:-1]).contiguous(),
    )


@torch.no_grad()
def fused_ip_sparse_fp8_b200(
    q: torch.Tensor,
    k_cache_ref: torch.Tensor,
    k_cache_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    k: int,
    *,
    coarse_factor: int = 4,
    coarse_k: int | None = None,
    q_fp8: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    num_buckets: int = 64,
    buf_cap: int | None = None,
    sample_size: int | None = None,
    refresh_every: int = 8,
    num_ctas_x: int = 296,
    rerank: bool = True,
    return_coarse: bool = False,
):
    """Default-off E4M3 coarse scan followed by authoritative fp16 re-score.

    The corpus FP8 cache and ``k_scale`` must be produced once with
    :func:`quantize_e4m3_rowwise`; the original fp16 corpus stays resident for
    winner re-scoring. The coarse width must be strictly larger than ``k`` so
    re-ranking can recover order changes introduced by quantization.
    ``rerank=False`` is a benchmark/debug mode that returns all coarse
    candidates and skips the fp16 gather+BMM+topk tail.
    """
    if os.environ.get("LITETOPK_MARSCO_FP8_SCAN", "0") != "1":
        raise RuntimeError(
            "set LITETOPK_MARSCO_FP8_SCAN=1 before the first extension load"
        )
    if q.dim() == 4:
        q = q[0, :, 0, :]
    elif q.dim() == 3:
        q = q[0]
    if q.dim() != 2 or k_cache_ref.dim() != 3:
        raise ValueError("q must be [B,D], k_cache_ref [1,M,D]")
    if q.shape != (64, 768):
        raise ValueError("experimental fp8 path requires q.shape == (64,768)")
    if (
        k_cache_ref.shape[0] != 1
        or k_cache_ref.shape[2] != 768
        or k_cache_ref.shape != k_cache_fp8.shape
    ):
        raise ValueError("corpus tensors must match with shape [1,M,768]")
    m = k_cache_ref.shape[1]
    if m % 64 != 0:
        raise ValueError("M must be a multiple of 64")
    if not 1 <= k < m:
        raise ValueError("experimental fp8 path requires 1 <= k < M")
    if coarse_k is None:
        if coarse_factor < 2:
            raise ValueError("coarse_factor must be at least 2")
        coarse_k = min(m, coarse_factor * k)
    if not k < coarse_k <= m:
        raise ValueError("coarse_k must satisfy k < coarse_k <= M")
    tensors = (q, k_cache_ref, k_cache_fp8, k_scale)
    if any(not t.is_cuda or t.device != q.device for t in tensors):
        raise ValueError("all inputs must be CUDA tensors on one device")
    if q.dtype != torch.float16 or k_cache_ref.dtype != torch.float16:
        raise ValueError("q and k_cache_ref must be float16")
    if k_cache_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("k_cache_fp8 must be float8_e4m3fn")
    if k_scale.dtype != torch.float32 or k_scale.shape != (1, m):
        raise ValueError("k_scale must be contiguous float32 [1,M]")
    if q_fp8 is None or q_scale is None:
        q_fp8, q_scale = quantize_e4m3_rowwise(q.contiguous())
    if (
        q_fp8.dtype != torch.float8_e4m3fn
        or q_fp8.shape != q.shape
        or q_scale.dtype != torch.float32
        or q_scale.shape != (64,)
    ):
        raise ValueError("q_fp8/q_scale must be E4M3 [64,768]/fp32 [64]")
    if q_fp8.device != q.device or q_scale.device != q.device:
        raise ValueError("q_fp8/q_scale must be on the query device")
    if sample_size is None:
        sample_size = min(m, max(16384, min(8 * coarse_k, 262144)))
    if buf_cap is None:
        buf_cap = m
    _ensure_loaded()
    values, indices, coarse_indices = (
        torch.ops.litetopk_sm100.fused_ip_sparse_fp8_b200(
            q.contiguous(),
            q_fp8.contiguous(),
            q_scale.contiguous(),
            k_cache_ref.contiguous(),
            k_cache_fp8.contiguous(),
            k_scale.contiguous(),
            k,
            coarse_k,
            num_buckets,
            buf_cap,
            sample_size,
            refresh_every,
            num_ctas_x,
            1,
            rerank,
        )
    )
    if return_coarse:
        return values, indices, coarse_indices
    return values, indices
