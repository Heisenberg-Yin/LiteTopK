"""Cost model for flash_mst — GPU-resident dense Boruvka MST."""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Estimate cost of dense flash_mst on an (N, N) MRD matrix.

    The dominant cost is the per-Boruvka-iter argmin sweep of the (N, N)
    MRD matrix. Boruvka takes ~log2(N) outer rounds; per round we touch
    every cell once. So total bytes ≈ N² · 4 · log2(N).
    """
    if len(shape) == 2 and shape[0] == shape[1]:
        N = shape[0]
    elif len(shape) == 1:
        N = int(shape[0])
    else:
        raise ValueError(
            f"flash_mst.estimate expects an (N, N) MRD shape; got {shape}"
        )

    import math
    rounds = max(1, int(math.ceil(math.log2(max(N, 2)))))
    bytes_moved = N * N * 4 * rounds + N * 4 * 6 * rounds
    flops = N * N * 4 * rounds
    rt, bound = roofline(flops, bytes_moved, "fp32", device, op_type="elementwise")
    return Estimate(
        op_name="flash_mst",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * N * 4 / 1e9,
        bound=bound, confidence="heuristic",
        n_kernel_launches=4 * rounds,
        suggested_config={"rounds": rounds},
        notes=[
            f"N={N}, ~{rounds} Boruvka rounds",
            "Packed-int64 atomic_min argmin + concurrent UF + pointer-jump.",
        ],
        expected_residual=None, precision_tier=None, tol=tol,
    )


def estimate_cc(shape, params=None, tol=None, dtype="int32", device="H100", **_):
    """Cost for flash_cc_from_edges (sparse CC on edge list).

    `shape` should be (N, E) — number of vertices, number of edges.
    """
    if len(shape) != 2:
        raise ValueError(
            f"flash_cc_from_edges shape must be (N, E); got {shape}"
        )
    N, E = shape
    bytes_moved = E * 8 + N * 4 * 4
    flops = 4 * E * 16
    rt, bound = roofline(flops, bytes_moved, "int32", device, op_type="elementwise")
    return Estimate(
        op_name="flash_cc_from_edges",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(E * 8 + N * 4) / 1e9,
        bound=bound, confidence="heuristic",
        n_kernel_launches=2,
        suggested_config={"max_find": 8, "max_passes": 16},
        notes=[f"N={N} vertices, E={E} edges, BLOCK=128 edges/program"],
        expected_residual=None, precision_tier=None, tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {"backend": "triton", "kernel": "flash_mst_boruvka"}
