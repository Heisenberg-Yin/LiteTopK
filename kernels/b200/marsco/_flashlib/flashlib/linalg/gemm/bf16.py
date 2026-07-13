"""bf16 GEMM with fp32 accumulation — single-precision tensor-core matmul."""
import torch

from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.gemm.fp32 import _shape_mkn


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """bf16 inputs, fp32 output. ~1e-3 relative error.

    Cast inputs to bf16, run torch.matmul (TC fp32 accumulation, bf16 output),
    cast result to fp32. Loses precision on output truncation; for tighter
    error use gemm_3xbf16 (Ozaki emulation).
    """
    return torch.matmul(A.to(torch.bfloat16), B.to(torch.bfloat16)).to(torch.float32)


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    M, K, N = _shape_mkn(shape)
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N) * 2 + M * N * 4
    rt, bound = roofline(flops, bytes_moved, "bf16", device, op_type="gemm")
    return Estimate(
        op_name='gemm_bf16',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(M * K + K * N) * 2 / 1e9 + M * N * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        suggested_config={}, subops=[],
        notes=[f"bf16 inputs / fp32 output (7-bit mantissa), M={M} K={K} N={N}"],
        expected_residual=1e-3, precision_tier="fast", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {}
