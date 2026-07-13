"""CuTeDSL Hopper INT8 GEMM with INT32 accumulator.

Wraps the upstream ``HopperWgmmaGemmPersistentKernel`` (vendored at
``fast_gemm/cute_lib/cute_hopper_gemm.py``) for INT8 ``A`` (M, K) and
INT8 ``B`` (N, K), producing INT32 ``C`` (M, N).  This is the
performance path for the ``ozaki2_triton``/``ozaki2_cute`` mode family —
the Triton ``_gemm_int8_kernel`` peaks at ~1290 TOPS / 65% of the 1979
TOPS H200 INT8 dense ceiling, while the CuTeDSL kernel uses TMA + WGMMA
+ TMA-multicast clusters and gets meaningfully closer.

API mirrors ``ozaki_int8._triton_int_mm`` exactly so the two are
swap-in compatible::

    cute_int8_mm(A_int8, B_int8, out_int32)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from flashlib.linalg.gemm.cutedsl.lib.hopper_gemm import HopperWgmmaGemmPersistentKernel


_TileShape = Tuple[int, int]
_ClusterShape = Tuple[int, int]


# Tuned defaults from bench/agent_loop.py cute_int8 sweep on H200 (8192³):
# (128, 256) tile, cluster (2, 1) along M, swizzle=8, raster along N → 1546
# TOPS / 78.1% of the 1979 TOPS peak.  Beats Triton _gemm_int8_kernel at
# 1290 TOPS / 65%.  Push further by editing the upstream HopperWgmma kernel
# (e.g. larger ab_stage), tracked in iter 14 backlog.
_DEFAULT_TILE_MN = (128, 256)
_DEFAULT_CLUSTER_MN = (2, 1)
_DEFAULT_SWIZZLE = 8
_DEFAULT_RASTER_ALONG_M = False


@lru_cache(maxsize=64)
def _compile_for(
    M: int, N: int, K: int,
    tile_shape_mn: _TileShape,
    cluster_shape_mn: _ClusterShape,
    swizzle_size: int,
    raster_along_m: bool,
):
    """JIT-compile the persistent INT8 GEMM kernel for this shape/config.

    Compile cost is multi-hundred ms, so this is cached aggressively. The
    cache survives across all callers (process-wide).
    """
    L = 1
    a_dtype = cutlass.Int8
    b_dtype = cutlass.Int8
    c_dtype = cutlass.Int32
    acc_dtype = cutlass.Int32
    # 8-bit kinds only support k-major; that matches our (M, K) and (N, K)
    # row-major layout (the K axis is the contiguous one).
    a_cpu = cutlass_torch.matrix(L, M, K, False, a_dtype)
    b_cpu = cutlass_torch.matrix(L, N, K, False, b_dtype)
    c_cpu = cutlass_torch.matrix(L, M, N, False, c_dtype)
    a_t, _ = cutlass_torch.cute_tensor_like(
        a_cpu, a_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_t, _ = cutlass_torch.cute_tensor_like(
        b_cpu, b_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_t, _ = cutlass_torch.cute_tensor_like(
        c_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
    )
    gemm = HopperWgmmaGemmPersistentKernel(
        acc_dtype, tile_shape_mn, cluster_shape_mn,
        swizzle_size, raster_along_m,
    )
    hw = cutlass.utils.HardwareInfo()
    max_active_clusters = hw.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(gemm, a_t, b_t, c_t, max_active_clusters, stream)


def _wrap(t: torch.Tensor, dtype):
    """Zero-copy view of a torch tensor as a CuTe tensor with explicit dtype."""
    ct = from_dlpack(t, assumed_align=16)
    ct.element_type = dtype
    leading_dim = cutlass_torch.get_leading_dim(t)
    return ct.mark_layout_dynamic(leading_dim=leading_dim)


def _ensure_3d(t: torch.Tensor) -> torch.Tensor:
    """Return ``t`` as a 3-D (M, K, 1) tensor.  The unsqueeze + contiguous
    is amortizable but small (it's a view on the same memory when possible),
    and importantly does NOT hold a reference to the input across calls —
    the previous data_ptr-keyed cache caused leaks across many calls."""
    if t.dim() == 3:
        return t
    v = t.unsqueeze(-1)
    if not v.is_contiguous():
        v = v.contiguous()
    return v


def cute_int8_mm(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor,
    tile_shape_mn: _TileShape = _DEFAULT_TILE_MN,
    cluster_shape_mn: _ClusterShape = _DEFAULT_CLUSTER_MN,
    swizzle_size: int = _DEFAULT_SWIZZLE,
    raster_along_m: bool = _DEFAULT_RASTER_ALONG_M,
) -> None:
    """``out = A @ B.T`` with A (M, K) INT8, B (N, K) INT8, out (M, N) INT32.

    All tensors must be CUDA, contiguous, and K-major. The caller owns ``out``.
    """
    assert A.is_cuda and B.is_cuda and out.is_cuda
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    assert out.dtype == torch.int32
    assert A.dim() == 2 and B.dim() == 2 and out.dim() == 2
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, f"K mismatch: A.K={K}, B.K={K2}"
    assert out.shape == (M, N), f"out shape {out.shape} != ({M}, {N})"

    # The CuTeDSL kernel works on (M, K, L) layouts. We unsqueeze to add the
    # batch dim.  These views are zero-copy when the input is contiguous.
    A3 = _ensure_3d(A)
    B3 = _ensure_3d(B)
    C3 = _ensure_3d(out)

    compiled = _compile_for(
        M, N, K, tile_shape_mn, cluster_shape_mn, swizzle_size, raster_along_m,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(_wrap(A3, cutlass.Int8), _wrap(B3, cutlass.Int8),
             _wrap(C3, cutlass.Int32), stream)


# ---- self-test --------------------------------------------------------------

def _selftest():
    """Sanity check: CuTeDSL INT8 GEMM matches torch._int_mm exactly."""
    torch.manual_seed(0)
    M, N, K = 1024, 1024, 1024
    A = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    B = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
    out = torch.empty((M, N), device="cuda", dtype=torch.int32)
    cute_int8_mm(A, B, out)
    ref = torch._int_mm(A, B.t())
    diff = (out - ref).abs().max().item()
    assert diff == 0, f"CuTeDSL INT8 GEMM mismatch: max diff {diff}"
    print("cute_int8_mm self-test: PASS (exact match vs torch._int_mm)")


if __name__ == "__main__":
    _selftest()
