"""Cost model for UMAP -- composes flash_knn + fuzzy-simplicial-set + SGD.

Stages:

* ``knn``                 -- ``n_neighbors`` per row via flash_knn.
* ``umap.fuzzy_simp_set`` -- per-edge ``ρ_i / σ_i`` solve + symmetrise;
                              ``E = N * n_neighbors`` edges, bw-bound.
* ``umap.sgd``            -- ``n_epochs`` × ``E`` negative-sampled edge
                              updates; bw-bound on the embedding.

Calibrated against ``vs_cuml_full.md``: 3.7-5.6x vs cuML at
N=10K..100K. The bf16-KNN opt-in (``tol=1e-3``) adds another ~10 %.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.info.dispatch import estimate as _est


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    N, D = shape
    n_neighbors = params.get("n_neighbors", 15)
    n_epochs = params.get("n_epochs", 200)

    knn_dtype = "bfloat16" if (tol is not None and tol >= 1e-3) else dtype
    knn = _est("knn", shape=(1, N, N, D), params={"k": n_neighbors},
               tol=tol, dtype=knn_dtype, device=device)
    knn.op_name = "umap.knn"

    # Fuzzy simplicial set: per-edge ρ + σ bisect (~20 FLOPs each).
    E = N * n_neighbors
    fss_flops = 20 * E
    fss_bytes = E * 4 * 3
    fss_rt, fss_bound = roofline(fss_flops, fss_bytes, dtype, device,
                                   op_type="elementwise", n_launches=2)
    fss = Estimate(
        op_name="umap.fuzzy_simp_set", runtime_ms=fss_rt,
        flops=fss_flops, bytes_moved=fss_bytes,
        memory_peak_gb=E * 4 / 1e9,
        bound=fss_bound, confidence="roofline", n_kernel_launches=2,
        suggested_config={}, subops=[],
        notes=[f"E={E}; ρ/σ bisect + symmetrise."],
        tol=tol,
    )

    # SGD: each epoch processes E edges; each edge ~20 FLOPs over the
    # 2-D embedding plus negative samples (~5 per pos edge default).
    n_neg = 5
    sgd_flops_iter = E * (n_neg + 1) * 20
    sgd_bytes_iter = E * (n_neg + 1) * 8 * 2
    sgd_flops = n_epochs * sgd_flops_iter
    sgd_bytes = n_epochs * sgd_bytes_iter
    sgd_rt, sgd_bound = roofline(sgd_flops, sgd_bytes, dtype, device,
                                   op_type="elementwise",
                                   n_launches=n_epochs)
    sgd = Estimate(
        op_name="umap.sgd", runtime_ms=sgd_rt,
        flops=sgd_flops, bytes_moved=sgd_bytes,
        memory_peak_gb=N * 2 * 4 / 1e9,
        bound=sgd_bound, confidence="heuristic", n_kernel_launches=n_epochs,
        suggested_config={"n_epochs": n_epochs, "n_negative": n_neg},
        subops=[],
        notes=[f"E={E}, n_epochs={n_epochs}, n_negative={n_neg}"],
        tol=tol,
    )

    total = knn.runtime_ms + fss.runtime_ms + sgd.runtime_ms
    return Estimate(
        op_name="umap",
        runtime_ms=total,
        flops=knn.flops + fss.flops + sgd.flops,
        bytes_moved=knn.bytes_moved + fss.bytes_moved + sgd.bytes_moved,
        memory_peak_gb=max(knn.memory_peak_gb, sgd.memory_peak_gb),
        bound=knn.bound, confidence="roofline",
        n_kernel_launches=(knn.n_kernel_launches + fss.n_kernel_launches
                            + sgd.n_kernel_launches),
        suggested_config={"n_neighbors": n_neighbors, "n_epochs": n_epochs},
        subops=[knn, fss, sgd],
        notes=[f"N={N}, D={D}, k={n_neighbors}, n_epochs={n_epochs}",
               f"knn dtype: {knn_dtype} (gated by tol={tol})"],
        expected_residual=knn.expected_residual,
        precision_tier=knn.precision_tier,
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {
        "n_neighbors": (params or {}).get("n_neighbors", 15),
        "n_epochs": (params or {}).get("n_epochs", 200),
        "n_negative": 5,
    }


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_umap_triton(shape, params=None, tol=None,
                           dtype="float32", device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "umap_triton"
    est.tol = tol
    return est


def estimate_umap_cutedsl(shape, params=None, tol=None,
                            dtype="float32", device="H100", **_):
    """CuteDSL backend -- swaps in fused fuzzy_simplicial_set kernel.

    KNN + SGD dominate end-to-end; the fuzzy-set swap is ~5 % of the
    total. Reported parity.
    """
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "umap_cutedsl"
    for s in est.subops:
        if s.op_name == "umap.fuzzy_simp_set":
            s.op_name = "umap.fuzzy_simp_set_cutedsl"
            s.notes = list(s.notes) + ["CuteDSL fused ρ/σ; parity with Triton."]
    est.notes = list(est.notes) + [
        "cutedsl backend: fuzzy-set swapped; KNN+SGD dominate, total ~Triton."
    ]
    est.tol = tol
    return est
