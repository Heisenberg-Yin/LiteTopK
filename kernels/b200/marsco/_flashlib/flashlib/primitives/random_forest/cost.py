"""Cost model for Random Forest (BFS-built histogram trees).

Dominant cost is the per-level histogram aggregation; a level processes
all ``N`` rows once, computing for each candidate split a sum/count over
``D`` features into a ``(L, D, n_bins)`` histogram.

A single tree of depth ``max_depth`` processes ``N`` rows
``max_depth`` times (BFS, each row visits one node per level). With
``n_trees`` parallel trees and a histogram of ``n_bins`` per feature
the per-tree work is::

    flops_per_tree = max_depth * N * D * n_bins * 4  // hist + best-split scan
    bytes_per_tree = max_depth * N * D * 4           // X read per level

Bytes dominate at our shapes (N=100K-500K, D=200-500, n_bins=128).
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    N, D = shape
    n_trees = params.get("n_estimators", params.get("n_trees", 100))
    max_depth = params.get("max_depth", 10)
    n_bins = params.get("n_bins", 128)

    flops_per_tree = max_depth * N * D * n_bins * 4
    bytes_per_tree = max_depth * N * D * 4
    flops = n_trees * flops_per_tree
    bytes_moved = n_trees * bytes_per_tree

    n_launches = n_trees * max_depth * 3  # hist + scan + scatter per level
    rt, bound = roofline(flops, bytes_moved, dtype, device,
                          op_type="elementwise", n_launches=n_launches)
    return Estimate(
        op_name="random_forest",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * D * 4 / 1e9,
        bound=bound, confidence="heuristic", n_kernel_launches=n_launches,
        suggested_config={"n_trees": n_trees, "max_depth": max_depth,
                           "n_bins": n_bins, "max_features": None},
        subops=[],
        notes=[
            f"N={N}, D={D}, n_trees={n_trees}, max_depth={max_depth}, "
            f"n_bins={n_bins}",
            "BFS histogram split; max_features=None for accuracy parity "
            "with cuML (default 'sqrt' under-trains on these shapes).",
        ],
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {"n_trees": 100, "max_depth": 10, "n_bins": 128,
             "max_features": None}


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_random_forest_triton(shape, params=None, tol=None,
                                    dtype="float32", device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "random_forest_triton"
    est.tol = tol
    return est


def estimate_random_forest_cutedsl(shape, params=None, tol=None,
                                     dtype="float32", device="H100", **_):
    """CuteDSL backend -- no architectural win (see cutedsl_impl.py).

    The histogram aggregation is fundamentally launch-dominated; the
    CuteDSL alternative ports the kernel but doesn't change the algorithmic
    structure. Reported parity end-to-end.
    """
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "random_forest_cutedsl"
    est.notes = list(est.notes) + [
        "cutedsl backend: ported histogram kernel; no architectural win "
        "(launch-dominated). End-to-end parity with Triton.",
    ]
    est.tol = tol
    return est
