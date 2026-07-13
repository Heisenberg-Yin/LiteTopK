"""Cost model for connected_components on edge list."""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def estimate(shape, params=None, tol=None, dtype="int32", device="H100", **_):
    if len(shape) != 2:
        raise ValueError("connected_components shape must be (N, E)")
    N, E = shape
    flops = 4 * E * 16
    bytes_moved = E * 8 + N * 4 * 4
    rt, bound = roofline(flops, bytes_moved, "fp32", device, op_type="elementwise")
    return Estimate(
        op_name="connected_components",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * 4 / 1e9,
        bound=bound, confidence="heuristic", n_kernel_launches=2,
        suggested_config={"max_find": 16, "n_passes": 2},
        notes=[f"N={N} vertices, E={E} edges"],
        expected_residual=None, precision_tier=None, tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="int32", device="H100", **_):
    return {"max_find": 16, "n_passes": 2}
