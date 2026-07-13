"""Cost model for Spectral Clustering -- composes knn + Laplacian + eigh + kmeans.

The flashlib path:

1. ``knn``       -- ``n_neighbors`` per row (sparse affinity), via flash_knn.
2. ``laplacian`` -- ``L = D - W`` build + row normalise; bw-bound on
                    ``E ≈ N * n_neighbors`` edges.
3. ``eigh``      -- top-K eigenvectors of the ``(N, N)`` normalised
                    Laplacian. Dominant cost at our shapes (N <= 50K).
4. ``kmeans``    -- final K-means on the ``(N, K)`` embedding.

vs scikit-learn (which lacks a GPU peer) we see ~40x at N=8K.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.info.dispatch import estimate as _est


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    N, D = shape
    K = params.get("n_clusters", 10)
    n_neighbors = params.get("n_neighbors", 10)

    knn = _est("knn", shape=(1, N, N, D), params={"k": n_neighbors},
               tol=tol, dtype=dtype, device=device)
    knn.op_name = "spectral.knn"

    # Laplacian build: read W edges, compute degree, scatter.
    E = N * n_neighbors
    lap_flops = 4 * E
    lap_bytes = E * 4 * 4
    lap_rt, lap_bound = roofline(lap_flops, lap_bytes, dtype, device,
                                   op_type="elementwise", n_launches=2)
    lap = Estimate(
        op_name="spectral.laplacian", runtime_ms=lap_rt,
        flops=lap_flops, bytes_moved=lap_bytes,
        memory_peak_gb=E * 4 / 1e9,
        bound=lap_bound, confidence="roofline", n_kernel_launches=2,
        suggested_config={}, subops=[],
        notes=[f"E={E}; row-normalised symmetric Laplacian."],
        tol=tol,
    )

    # eigh top-K on the (N, N) Laplacian. The dispatcher picks Halko
    # for large N (K <= N/4), exact for small N.
    eig = _est("eigh", shape=(N, N), params={"K": K},
               tol=tol, dtype=dtype, device=device)
    eig.op_name = "spectral.eigh"

    # Final KMeans on the (N, K) embedding.
    km = _est("kmeans", shape=(N, K), params={"K": K},
              tol=tol, dtype=dtype, device=device)
    km.op_name = "spectral.kmeans"

    total = knn.runtime_ms + lap.runtime_ms + eig.runtime_ms + km.runtime_ms
    return Estimate(
        op_name="spectral_clustering",
        runtime_ms=total,
        flops=knn.flops + lap.flops + eig.flops + km.flops,
        bytes_moved=knn.bytes_moved + lap.bytes_moved + eig.bytes_moved
                    + km.bytes_moved,
        memory_peak_gb=max(knn.memory_peak_gb, eig.memory_peak_gb,
                            km.memory_peak_gb),
        bound=eig.bound, confidence="roofline",
        n_kernel_launches=(knn.n_kernel_launches + lap.n_kernel_launches
                            + eig.n_kernel_launches + km.n_kernel_launches),
        suggested_config={"n_clusters": K, "n_neighbors": n_neighbors},
        subops=[knn, lap, eig, km],
        notes=[
            f"N={N}, D={D}, K={K}, n_neighbors={n_neighbors}",
            "knn -> Laplacian -> eigh(top-K) -> kmeans on embedding.",
        ],
        expected_residual=eig.expected_residual,
        precision_tier=eig.precision_tier,
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {
        "n_clusters": (params or {}).get("n_clusters", 10),
        "n_neighbors": (params or {}).get("n_neighbors", 10),
        "assign_labels": "kmeans",
    }


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_spectral_clustering_triton(shape, params=None, tol=None,
                                          dtype="float32", device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "spectral_clustering_triton"
    est.tol = tol
    return est


def estimate_spectral_clustering_cutedsl(shape, params=None, tol=None,
                                           dtype="float32", device="H100", **_):
    """CuteDSL backend -- swaps in CUTLASS fused-Laplacian sub-kernel.

    Eigh dominates at every shape we benchmark, so the cutedsl swap
    is essentially invisible end-to-end; reported parity.
    """
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "spectral_clustering_cutedsl"
    est.notes = list(est.notes) + [
        "cutedsl backend: fused-Laplacian swapped; eigh dominates, total ~Triton."
    ]
    est.tol = tol
    return est
