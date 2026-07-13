"""Cost model for gram_gemm: X (N, D) -> X @ X.T (N, N)."""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    N, D = shape
    if tol is None or tol <= 0:
        op_dtype = "fp32"; tier = "exact"; res = 1e-7
    elif tol >= 1e-3:
        op_dtype = "bf16"; tier = "fast"; res = 1e-3
    else:
        op_dtype = "bf16"; tier = "mixed"; res = 1e-5  # 3xbf16 emulation
    dtype_bytes = 4 if op_dtype == "fp32" else 2
    flops = N * N * D
    bytes_moved = (N * D + N * N) * dtype_bytes
    rt, bound = roofline(flops, bytes_moved, op_dtype, device, op_type="gemm")
    return Estimate(
        op_name="gram_gemm",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(N * D + N * N) * dtype_bytes / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        notes=[f"N={N}, D={D}; X @ X.T ({op_dtype})"],
        expected_residual=res, precision_tier=tier, tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {}
