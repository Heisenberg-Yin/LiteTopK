"""fp32 GEMM — native torch.matmul on fp32 (no tensor cores)."""
import torch

from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Native fp32 matmul. ~1e-7 relative error. Slowest variant."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.matmul(A.to(torch.float32), B.to(torch.float32))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    M, K, N = _shape_mkn(shape)
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N + M * N) * 4
    rt, bound = roofline(flops, bytes_moved, "fp32", device, op_type="solver")
    return Estimate(
        op_name='gemm_fp32',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(M * K + K * N + M * N) * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        suggested_config={}, subops=[],
        notes=[f"native fp32 (CUDA cores, no TC), M={M} K={K} N={N}"],
        expected_residual=1e-7, precision_tier="exact", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {}


def _shape_mkn(shape):
    if len(shape) == 3:
        return shape  # (M, K, N)
    if len(shape) == 2:
        return shape[0], shape[1], shape[1]  # square fallback
    raise ValueError("gemm shape must be (M, K, N)")
