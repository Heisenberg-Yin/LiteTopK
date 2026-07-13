"""3xbf16 GEMM — Ozaki/Markidis 3-product fp32 emulation via BF16 TC.

A = A_hi + A_lo  (each cast to bf16)
B = B_hi + B_lo
out = (A_hi @ B_hi) + (A_hi @ B_lo) + (A_lo @ B_hi)   [LoLo dropped]

Two implementations are exposed by :func:`gemm`. The dispatcher picks the
fastest available; the user never has to choose.

* ``cute_fused`` (CuTeDSL, single launch). Pareto-best in BOTH precision
  AND throughput: keeps the FP32 WGMMA accumulator across all 3 dots so
  the BF16 round-off floor is never re-quantised. Measured on H200 8192³:
  **4.2 ms / RMS 2.9e-5** — 1.5× faster AND 60× tighter than the legacy
  Python wrapper (which truncated each of the 3 GEMM outputs to BF16
  before summing in FP32, costing ~6 effective bits).

* ``python_3call`` (3 ``torch.matmul`` calls + FP32 sum). Pure-PyTorch
  fallback used when CUTLASS DSL is unavailable. ~1.7e-3 RMS, ~6 ms
  at 8192³.

Reference: Ootomo & Yokota 2022; Markidis et al. 2018; STATUS.md iter 12
in fast-gemm (single-launch fused path closes the precision gap).
"""
from __future__ import annotations

import torch

from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.gemm.fp32 import _shape_mkn


_CUTE_AVAILABLE: bool | None = None


def _cute_available() -> bool:
    global _CUTE_AVAILABLE
    if _CUTE_AVAILABLE is None:
        try:
            import cutlass  # noqa: F401
            import cutlass.cute  # noqa: F401
            _CUTE_AVAILABLE = (
                torch.cuda.is_available()
                and torch.cuda.get_device_properties(0).major >= 9
            )
        except Exception:
            _CUTE_AVAILABLE = False
    return _CUTE_AVAILABLE


def _python_3call(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Legacy 3-matmul Python path. Kept as a fallback when CUTLASS DSL
    is unavailable. Note: ``torch.matmul(bf16, bf16)`` returns BF16, so
    each partial is BF16-truncated before the FP32 sum."""
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    A_hi = A.to(torch.bfloat16)
    A_lo = (A - A_hi.to(torch.float32)).to(torch.bfloat16)
    B_hi = B.to(torch.bfloat16)
    B_lo = (B - B_hi.to(torch.float32)).to(torch.bfloat16)
    p1 = torch.matmul(A_hi, B_hi).to(torch.float32)
    p2 = torch.matmul(A_hi, B_lo).to(torch.float32)
    p3 = torch.matmul(A_lo, B_hi).to(torch.float32)
    return p1 + p2 + p3


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """3xbf16 GEMM. Auto-routes to the CuTeDSL fused single-launch kernel
    when available (Pareto win in both precision and throughput); falls
    back to a 3-matmul Python path otherwise. Standard PyTorch convention:
    ``A`` is ``(M, K)``, ``B`` is ``(K, N)`` — computes ``A @ B``.
    """
    if _cute_available():
        try:
            from flashlib.linalg.gemm.cutedsl.bf16x3_fused import matmul_bf16x3_cute_fused
            A32 = A.to(torch.float32).contiguous()
            B32 = B.to(torch.float32).t().contiguous()  # kernel takes (N, K)
            return matmul_bf16x3_cute_fused(A32, B32)
        except Exception:
            pass  # fall through to python
    return _python_3call(A, B)


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    M, K, N = _shape_mkn(shape)
    flops = 3 * 2 * M * K * N
    bytes_moved = (M * K + K * N) * 2 + M * N * 4 + (M * K + K * N) * 4
    rt, bound = roofline(flops, bytes_moved, "bf16", device, op_type="gemm")
    # cute_fused achieves ~74% of bf16/3 ceiling (228 TF on H200 at 8192^3).
    rt = rt * 1.35  # = 1/0.74
    return Estimate(
        op_name='gemm_3xbf16',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(M * K + K * N) * 4 / 1e9 + M * N * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        suggested_config={"backend": "cute_fused"}, subops=[],
        notes=[
            f"M={M} K={K} N={N}",
            "CuTeDSL single-launch BF16x3 (228 TF on H200), measured RMS rel-err ~3e-5.",
            "Falls back to 3-matmul Python (~1.7e-3 RMS, slower) if CUTLASS DSL absent.",
        ],
        expected_residual=3e-5, precision_tier="mixed", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {"backend": "cute_fused" if _cute_available() else "python_3call"}
