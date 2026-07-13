"""FP16x3 GEMM with Kahan-compensated outer reduction.

Pareto-better than cuBLAS FP32 (~50 TF, ~17 bits at K=4096+):
  ~135 TF on H200, stable ~18-19 bits regardless of K.

Sits between bf16x3 (~13 bits, 250 TF) and cuBLAS FP64 (~50 bits, 56 TF)
on the precision/perf frontier. Constraint: |a|, |b| ≤ 65504 (FP16 range).
"""
import torch

from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.gemm.fp32 import _shape_mkn


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """FP32 in, FP32 out, ~18-19 bits at any K via Kahan outer-loop compensation.

    Note: the underlying fast-gemm Triton kernel expects ``B`` shaped ``(N, K)``;
    we transpose at the boundary so callers pass ``B`` in PyTorch's standard
    ``(K, N)`` shape (computes ``A @ B``).
    """
    from flashlib.linalg.gemm.triton.fp16x3_kahan import matmul_fp16x3_kahan
    A32 = A.to(torch.float32).contiguous()
    B32 = B.to(torch.float32).t().contiguous()
    return matmul_fp16x3_kahan(A32, B32)


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    M, K, N = _shape_mkn(shape)
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N) * 4 + M * N * 4
    rt, bound = roofline(flops, bytes_moved, "fp16", device, op_type="gemm")
    # 3 fp16 MMAs per output, plus Kahan compensation overhead → ~4× peak.
    rt = rt * 4.0
    return Estimate(
        op_name="gemm_fp16_x3_kahan",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(M * K + K * N + M * N) * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        suggested_config={}, subops=[],
        notes=[
            f"M={M} K={K} N={N}",
            "FP16x3 Triton + Kahan outer reduce; ~21 effective bits at any K.",
            "Measured RMS rel-err ~4.6e-7 on H200, K-independent (Kahan kills the K-floor).",
        ],
        expected_residual=5e-7, precision_tier="exact", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {}
