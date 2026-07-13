"""3xfp16 GEMM — Ozaki 3-product fp32 emulation via fp16 TC.

Same structure as 3xbf16 but uses fp16 splits. fp16's 10-bit mantissa gives
**measured RMS rel-err ~2.1e-4** on H200 (vs ~1.7e-3 for 3xbf16, ~3.6e-4
for single fp16). Constraint: inputs must fit in fp16 range (|x| < 65504).
"""
import torch

from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.gemm.fp32 import _shape_mkn


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    A_hi = A.to(torch.float16)
    A_lo = (A - A_hi.to(torch.float32)).to(torch.float16)
    B_hi = B.to(torch.float16)
    B_lo = (B - B_hi.to(torch.float32)).to(torch.float16)
    p1 = torch.matmul(A_hi, B_hi).to(torch.float32)
    p2 = torch.matmul(A_hi, B_lo).to(torch.float32)
    p3 = torch.matmul(A_lo, B_hi).to(torch.float32)
    return p1 + p2 + p3


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    M, K, N = _shape_mkn(shape)
    flops = 3 * 2 * M * K * N
    bytes_moved = 3 * ((M * K + K * N) * 2 + M * N * 4) + (M * K + K * N) * 4
    rt, bound = roofline(flops, bytes_moved, "fp16", device, op_type="gemm")
    return Estimate(
        op_name='gemm_3xfp16',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(M * K + K * N) * 4 / 1e9 + M * N * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=3,
        suggested_config={}, subops=[],
        notes=[
            f"Ozaki 3xfp16 emulation of fp32, M={M} K={K} N={N}",
            "3 fp16 TC GEMMs + accumulation; measured RMS rel-err ~2.1e-4 on H200.",
        ],
        expected_residual=2.1e-4, precision_tier="mixed", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {}
