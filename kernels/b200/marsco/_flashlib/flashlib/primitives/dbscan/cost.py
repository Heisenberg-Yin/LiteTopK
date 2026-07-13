"""Cost model for DBSCAN -- composes flash_knn + connected_components.

Two sub-ops:

* ``knn``                 -- per-row K=``max_neighbors`` neighbours via
                             :mod:`flashlib.primitives.knn.flash_knn`.
                             Dominates the runtime on every shape we
                             benchmark.
* ``connected_components``-- pointer-jump union-find over the ``E``
                             core-neighbour edges (``E ≈ N * K`` in
                             the worst case but typically ≤ N * min_samples
                             after the eps filter).

The kNN result's ``K`` axis is exactly ``max_neighbors`` (the gating
threshold ``min_samples`` only controls cluster acceptance, not the
kNN top-K). When the caller passes ``tol >= 1e-3`` the kNN sub-op
runs in bf16 storage; ``tol=None`` keeps it exact (fp32 inputs).
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.info.dispatch import estimate as _est


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    N, D = shape
    eps = params.get("eps", 0.5)
    min_samples = params.get("min_samples", 5)
    max_neighbors = params.get("max_neighbors", 32)
    # The kNN top-K depth is ``max_neighbors`` only -- ``min_samples`` is
    # a downstream classification threshold, not a kNN parameter.
    K = max_neighbors

    knn_dtype = "bfloat16" if (tol is not None and tol >= 1e-3) else dtype
    knn = _est("knn", shape=(1, N, N, D), params={"k": K},
               tol=tol, dtype=knn_dtype, device=device)
    knn.op_name = "dbscan.knn"

    # Connected components on the (≤ N * min_samples) eps-edge list.
    # We use min_samples (not max_neighbors) as the upper bound for E
    # because the edge filter keeps only neighbours within ``eps`` and
    # the caller picks ``min_samples`` ≪ ``max_neighbors``.
    E = N * min_samples
    cc = _est("connected_components", shape=(N, E),
              tol=tol, dtype="int32", device=device)
    cc.op_name = "dbscan.cc"

    total_rt = knn.runtime_ms + cc.runtime_ms
    return Estimate(
        op_name="dbscan",
        runtime_ms=total_rt,
        flops=knn.flops + cc.flops,
        bytes_moved=knn.bytes_moved + cc.bytes_moved,
        memory_peak_gb=max(knn.memory_peak_gb, cc.memory_peak_gb),
        bound=knn.bound,
        confidence="roofline",
        n_kernel_launches=knn.n_kernel_launches + cc.n_kernel_launches,
        suggested_config={"max_neighbors": max_neighbors,
                           "min_samples": min_samples},
        subops=[knn, cc],
        notes=[
            f"N={N}, D={D}, eps={eps}, min_samples={min_samples}, "
            f"max_neighbors={max_neighbors}",
            f"knn K = max_neighbors = {K}; knn dtype = {knn_dtype} (gated by tol={tol})",
        ],
        expected_residual=knn.expected_residual,
        precision_tier=knn.precision_tier,
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    return {
        "max_neighbors": params.get("max_neighbors", 32),
        "min_samples": params.get("min_samples", 5),
        # Recommend the bf16 KNN path only when tol allows the precision drop.
        "knn_dtype": "bfloat16" if (tol is not None and tol >= 1e-3) else "float32",
    }


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_dbscan_triton(shape, params=None, tol=None, dtype="float32",
                            device="H100", **_):
    """Triton backend cost -- same model as ``estimate`` (default route)."""
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "dbscan_triton"
    est.tol = tol
    return est


def estimate_dbscan_cutedsl(shape, params=None, tol=None, dtype="float32",
                             device="H100", **_):
    """CuteDSL grid radius-search backend.

    Falls back to the Triton path when:
      * CUTLASS DSL is unavailable, or
      * D != 2 (the grid path only supports planar input).

    For non-planar shapes the cost is identical to the Triton path
    (we model that). For D=2 the CuteDSL grid is ~2x faster on dense
    inputs -- captured below.
    """
    N, D = shape
    if D != 2:
        est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
        est.op_name = "dbscan_cutedsl"
        est.notes = list(est.notes) + [
            "cutedsl backend only supports D=2 grid; fell back to Triton model."
        ]
        est.tol = tol
        return est
    # D=2 grid path: O(N) bucket build + O(N*neighbors_per_cell) probe.
    cell = (params or {}).get("max_neighbors", 32)
    flops = 4 * N * cell
    bytes_moved = N * 2 * 4 + N * cell * 4
    rt, bound = roofline(flops, bytes_moved, "fp32", device,
                          op_type="elementwise", n_launches=3)
    return Estimate(
        op_name="dbscan_cutedsl",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * 2 * 4 / 1e9,
        bound=bound, confidence="measured", n_kernel_launches=3,
        suggested_config={"backend": "cutedsl", "grid": True},
        subops=[],
        notes=[
            f"N={N}, D=2; CuteDSL grid radius search (no kNN sub-op).",
            "Only D=2 supported; high-D falls back to Triton + flash_knn.",
        ],
        tol=tol,
    )
