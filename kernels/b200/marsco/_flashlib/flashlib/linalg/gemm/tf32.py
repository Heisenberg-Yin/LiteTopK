"""TF32 GEMM — torch.matmul with TF32 enabled. Hopper tensor cores."""
import torch

from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.gemm.fp32 import _shape_mkn


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """fp32-input matmul routed through TF32 tensor cores. ~1e-5 rel err."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return torch.matmul(A.to(torch.float32), B.to(torch.float32))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    M, K, N = _shape_mkn(shape)
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N + M * N) * 4
    rt, bound = roofline(flops, bytes_moved, "tf32", device, op_type="gemm")
    return Estimate(
        op_name='gemm_tf32',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(M * K + K * N + M * N) * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        suggested_config={}, subops=[],
        notes=[
            f"TF32 tensor cores (10-bit mantissa), M={M} K={K} N={N}",
            "Measured RMS rel-err ~2.9e-4 on H200, K-independent.",
        ],
        expected_residual=3e-4, precision_tier="mixed", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {}
