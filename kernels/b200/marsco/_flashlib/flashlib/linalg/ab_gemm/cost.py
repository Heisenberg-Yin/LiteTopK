"""Cost model for ab_gemm: A (N, P) and B (N, D) -> A.T @ B (P, D)."""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    if len(shape) != 3:
        raise ValueError("ab_gemm shape must be (N, P, D)")
    N, P, D = shape
    if tol is None or tol <= 0:
        op_dtype = "fp32"; tier = "exact"; res = 1e-7
    elif tol >= 1e-3:
        op_dtype = "bf16"; tier = "fast"; res = 1e-3
    else:
        op_dtype = "bf16"; tier = "mixed"; res = 1e-5
    dtype_bytes = 4 if op_dtype == "fp32" else 2
    flops = 2 * N * P * D
    bytes_moved = (N * P + N * D + P * D) * dtype_bytes
    rt, bound = roofline(flops, bytes_moved, op_dtype, device, op_type="gemm")
    return Estimate(
        op_name="ab_gemm",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(N * P + N * D) * dtype_bytes / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        notes=[f"A=(N={N},P={P}), B=(N={N},D={D}); ({op_dtype})"],
        expected_residual=res, precision_tier=tier, tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {}
