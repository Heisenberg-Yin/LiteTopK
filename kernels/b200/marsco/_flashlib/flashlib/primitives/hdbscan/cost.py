"""Cost model for HDBSCAN -- composes flash_knn + sparse MST + condensation.

The default ``flash_hdbscan`` path (``approximate=True``) builds a
sparse kNN graph and runs Boruvka MST on the mutual-reachability
edges, *not* a dense ``N×N`` MRD matrix. So the cost decomposes as:

  knn      :  pairwise top-k via flash_knn  (≈ N × N work, k cols)
  mrd_edges:  per-edge max(d, core_a, core_b)  (≈ N · k bytes)
  mst      :  sparse MST on ≈ N · k edges
  condense :  small-N tree post-processing

The legacy dense path (``prefer='dense'``) materialises the full
``N×N`` MRD matrix and runs Boruvka on it -- exposed as
:func:`estimate_hdbscan_dense` for reference.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.info.dispatch import estimate as _est


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Default route: sparse kNN-MRD + Boruvka on the kNN edge list."""
    params = params or {}
    N, D = shape
    k = params.get("k", 32)
    min_samples = params.get("min_samples", 5)

    # KNN dtype gated by tol (mirrors the runtime path).
    knn_dtype = "bfloat16" if (tol is not None and tol >= 1e-3) else dtype
    knn = _est("knn", shape=(1, N, N, D), params={"k": k},
               tol=tol, dtype=knn_dtype, device=device)
    knn.op_name = "hdbscan.knn"

    # Mutual reachability edges -- one max-of-3 per edge, bw-bound.
    E = N * k
    mrd_flops = 3 * E      # max(d, core_a, core_b)
    mrd_bytes = E * 4 * 4  # read d + 2 cores + write
    mrd_rt, mrd_bound = roofline(mrd_flops, mrd_bytes, dtype, device,
                                  op_type="elementwise", n_launches=1)
    mrd = Estimate(
        op_name="hdbscan.mrd_edges", runtime_ms=mrd_rt,
        flops=mrd_flops, bytes_moved=mrd_bytes,
        memory_peak_gb=E * 4 / 1e9,
        bound=mrd_bound, confidence="roofline", n_kernel_launches=1,
        suggested_config={}, subops=[],
        notes=[f"E={E}; max(d_ij, core_i, core_j) on the kNN edges."],
        tol=tol,
    )

    # Sparse MST on the kNN edges. flashlib runs Boruvka via
    # ``flash_cc_from_edges`` on the kNN edge list (the connected-components
    # cost is the right shape match for this -- E edges, N vertices).
    mst = _est("connected_components", shape=(N, E),
                tol=tol, dtype="int32", device=device)
    mst.op_name = "hdbscan.mst"

    # Tree condensation + cluster stability: O(N) Python-ish driver +
    # one small reduction; small relative to KNN.
    cond_flops = 20 * N
    cond_bytes = N * 4 * 4
    cond_rt, cond_bound = roofline(cond_flops, cond_bytes, dtype, device,
                                    op_type="elementwise", n_launches=3)
    cond = Estimate(
        op_name="hdbscan.condense", runtime_ms=cond_rt,
        flops=cond_flops, bytes_moved=cond_bytes,
        memory_peak_gb=N * 4 / 1e9,
        bound=cond_bound, confidence="heuristic", n_kernel_launches=3,
        suggested_config={}, subops=[],
        notes=[f"N={N}; tree condense + min_cluster_size prune + stability."],
        tol=tol,
    )

    total_rt = knn.runtime_ms + mrd.runtime_ms + mst.runtime_ms + cond.runtime_ms
    total_flops = knn.flops + mrd.flops + mst.flops + cond.flops
    total_bytes = knn.bytes_moved + mrd.bytes_moved + mst.bytes_moved + cond.bytes_moved
    return Estimate(
        op_name="hdbscan",
        runtime_ms=total_rt, flops=total_flops, bytes_moved=total_bytes,
        memory_peak_gb=max(knn.memory_peak_gb, mst.memory_peak_gb),
        bound=knn.bound, confidence="roofline",
        n_kernel_launches=knn.n_kernel_launches + 1 + mst.n_kernel_launches + 3,
        suggested_config={"k": k, "min_samples": min_samples},
        subops=[knn, mrd, mst, cond],
        notes=[
            f"N={N}, D={D}, k={k}, min_samples={min_samples}",
            f"knn dtype: {knn_dtype} (gated by tol={tol})",
            "Sparse path: knn (E=N·k) -> MRD edges -> Boruvka MST -> condense.",
        ],
        expected_residual=knn.expected_residual,
        precision_tier=knn.precision_tier,
        tol=tol,
    )


def estimate_hdbscan_dense(shape, params=None, tol=None, dtype="float32",
                            device="H100", **_):
    """Legacy dense path: materialises the full (N, N) MRD matrix.

    Use when ``prefer='dense'``; not the default route. Cost dominated
    by the dense MRD computation + dense Boruvka.
    """
    N, D = shape
    # Dense MRD: 2*N*N*D pairwise distance + 3*N*N max ops.
    dist_flops = 2 * N * N * D
    mrd_flops  = 3 * N * N
    flops = dist_flops + mrd_flops
    bytes_moved = N * D * 4 + N * N * 4 * 2  # X + read/write dense MRD
    dist_rt, dist_bound = roofline(flops, bytes_moved, dtype, device,
                                     op_type="gemm", n_launches=2)
    mst = _est("flash_mst", shape=(N, N), tol=tol, dtype="fp32", device=device)
    mst.op_name = "hdbscan_dense.mst"
    return Estimate(
        op_name="hdbscan_dense",
        runtime_ms=dist_rt + mst.runtime_ms,
        flops=flops + mst.flops, bytes_moved=bytes_moved + mst.bytes_moved,
        memory_peak_gb=N * N * 4 / 1e9,
        bound="memory" if N >= 10_000 else dist_bound,
        confidence="roofline",
        n_kernel_launches=2 + mst.n_kernel_launches,
        suggested_config={}, subops=[mst],
        notes=[
            f"N={N}, D={D}; dense N×N MRD + dense Boruvka MST.",
            "Legacy path (prefer='dense'); the default route uses sparse kNN.",
        ],
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    return {
        "approximate": True,
        "k": params.get("k", 32),
        "min_samples": params.get("min_samples", 5),
        "min_cluster_size": params.get("min_cluster_size", 25),
    }


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_hdbscan_triton(shape, params=None, tol=None, dtype="float32",
                             device="H100", **_):
    """Triton backend cost -- same model as ``estimate`` (default route)."""
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "hdbscan_triton"
    est.tol = tol
    return est


def estimate_hdbscan_cutedsl(shape, params=None, tol=None, dtype="float32",
                              device="H100", **_):
    """CuteDSL fused MRD-edge kernel on the sparse path.

    Replaces the ``mrd_edges`` sub-op with a Hopper-fused
    TMA + WGMMA + max kernel. On the cuML-comparable shapes wall-clock
    is within ±5 % of the Triton path because the kNN sub-op dominates
    -- so we model parity.
    """
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "hdbscan_cutedsl"
    # Replace the mrd_edges sub-op note rather than tweaking runtime.
    for s in est.subops:
        if s.op_name == "hdbscan.mrd_edges":
            s.op_name = "hdbscan.mrd_edges_cutedsl"
            s.notes = list(s.notes) + ["CuteDSL TMA+WGMMA fused MRD-edges."]
    est.notes = list(est.notes) + [
        "cutedsl backend swaps in fused MRD-edges; total wall-clock ~ Triton "
        "(kNN sub-op dominates).",
    ]
    est.tol = tol
    return est
