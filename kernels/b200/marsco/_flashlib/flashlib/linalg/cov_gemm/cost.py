"""Cost model for cov_gemm: X (N, D) -> X.T @ X (D, D)."""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    N, D = shape
    # tol routes the inner GEMM precision: tol=None -> fp32; >=1e-5 -> 3xbf16; etc.
    if tol is None or tol <= 0:
        op_dtype = "fp32"; tier = "exact"; res = 1e-7
    elif tol >= 1e-3:
        op_dtype = "bf16"; tier = "fast"; res = 1e-3
    elif tol >= 1e-5:
        op_dtype = "bf16"; tier = "mixed"; res = 1e-5  # via 3xbf16
    else:
        op_dtype = "tf32"; tier = "mixed"; res = 1e-5
    dtype_bytes = 4 if op_dtype in ("fp32", "tf32") else 2
    flops = 2 * N * D * D
    bytes_moved = (N * D + D * D) * dtype_bytes
    rt, bound = roofline(flops, bytes_moved, op_dtype, device, op_type="gemm")
    return Estimate(
        op_name="cov_gemm",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(N * D * dtype_bytes + D * D * 4) / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        notes=[f"N={N}, D={D}; X.T @ X tall-skinny GEMM ({op_dtype})"],
        expected_residual=res, precision_tier=tier, tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {}
