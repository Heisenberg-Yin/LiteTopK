"""fp16 GEMM with fp32 accumulation."""
import torch

from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.gemm.fp32 import _shape_mkn


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """fp16 inputs, fp32 output. ~1e-3 rel err but tighter mantissa than bf16.

    fp16 has narrower range (max 65504); caller must ensure |A|, |B| fit.
    """
    return torch.matmul(A.to(torch.float16), B.to(torch.float16)).to(torch.float32)


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    M, K, N = _shape_mkn(shape)
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N) * 2 + M * N * 4
    rt, bound = roofline(flops, bytes_moved, "fp16", device, op_type="gemm")
    return Estimate(
        op_name='gemm_fp16',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(M * K + K * N) * 2 / 1e9 + M * N * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        suggested_config={}, subops=[],
        notes=[f"fp16 inputs (10-bit mantissa, narrow range), M={M} K={K} N={N}"],
        expected_residual=8e-4, precision_tier="fast", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {}
