"""Cost model for CholQR2 / split_basis."""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    if isinstance(shape, (tuple, list)):
        N = shape[0]
        K = shape[1] if len(shape) > 1 else N // 2
    else:
        N = shape
        K = N // 2
    # 2 SYRKs (N x K) + 2 trsms + 1 GS pass = ~6 N x K x K matmuls
    flops = 6 * N * K * K
    bytes_moved = 8 * (N * K + K * K) * 4
    rt, bound = roofline(flops, bytes_moved, dtype, device, op_type="gemm")
    return Estimate(
        op_name='orthonormalize',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * K * 4 * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=6,
        suggested_config={}, subops=[],
        notes=[
            f"N={N}, K={K}; CholQR2 (SYRK + Cholesky + trsm) + cross-GS + refinement.",
            "Used by qdwh_eig spectral D&C to split into +/- eigenspaces.",
        ],
        expected_residual=1e-7, precision_tier="exact", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {"backend": "cholqr2"}
