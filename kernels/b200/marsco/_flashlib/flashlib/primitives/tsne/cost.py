"""Cost model for flashlib's exact t-SNE.

flashlib uses the **exact** ``O(N²)`` per-iteration gradient (no
Barnes-Hut / FFT truncation), composed as:

* ``knn``         -- kNN graph + perplexity bisection -> P matrix.
* ``tsne.grad``   -- ``n_iter`` iterations of:
                        repulsive   Σ_j q_ij² (y_i − y_j)  -> N² × D_out FLOPs
                        attractive  Σ_j p_ij  (y_i − y_j)  -> nnz(P) × D_out FLOPs
                     The repulsive term dominates.

vs cuML on the exact path (``method='exact'``) flashlib runs 13× / 64× /
145× faster at N=5K/10K/20K respectively. cuML's default
``method='fft'`` is O(N) per iter and gives up trustworthiness at
N >= 20K -- not apples-to-apples.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.info.dispatch import estimate as _est


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    N, D = shape
    D_out = params.get("n_components", 2)
    n_iter = params.get("n_iter", 1000)
    n_neighbors = params.get("perplexity", 30) * 3  # P sparsity

    # P matrix: kNN graph + bisect perplexity (small fp32 work + bw).
    knn = _est("knn", shape=(1, N, N, D), params={"k": n_neighbors},
               tol=tol, dtype=dtype, device=device)
    knn.op_name = "tsne.knn_p"

    # Gradient: per-iter O(N² * D_out) repulsive + O(nnz * D_out) attractive.
    grad_flops_iter = 4 * N * N * D_out + 4 * N * n_neighbors * D_out
    grad_bytes_iter = (N * D_out + N * N) * 4
    grad_flops = n_iter * grad_flops_iter
    grad_bytes = n_iter * grad_bytes_iter
    grad_rt, grad_bound = roofline(grad_flops, grad_bytes, dtype, device,
                                     op_type="gemm",
                                     n_launches=2 * n_iter)
    grad = Estimate(
        op_name="tsne.grad", runtime_ms=grad_rt,
        flops=grad_flops, bytes_moved=grad_bytes,
        memory_peak_gb=N * D_out * 4 / 1e9,
        bound=grad_bound, confidence="roofline",
        n_kernel_launches=2 * n_iter,
        suggested_config={"n_iter": n_iter}, subops=[],
        notes=[f"N={N}, D_out={D_out}, n_iter={n_iter}; "
                "exact O(N²) per-iter repulsive + attractive."],
        tol=tol,
    )

    total = knn.runtime_ms + grad.runtime_ms
    return Estimate(
        op_name="tsne",
        runtime_ms=total,
        flops=knn.flops + grad.flops,
        bytes_moved=knn.bytes_moved + grad.bytes_moved,
        memory_peak_gb=max(knn.memory_peak_gb, grad.memory_peak_gb),
        bound=grad.bound, confidence="roofline",
        n_kernel_launches=knn.n_kernel_launches + grad.n_kernel_launches,
        suggested_config={"D_out": D_out, "n_iter": n_iter,
                           "n_neighbors": n_neighbors},
        subops=[knn, grad],
        notes=[f"N={N}, D={D}, D_out={D_out}, n_iter={n_iter}",
               "Exact O(N²) gradient -- not BH/FFT."],
        expected_residual=knn.expected_residual,
        precision_tier=knn.precision_tier,
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {
        "n_iter": (params or {}).get("n_iter", 1000),
        "method": "exact",
    }


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_tsne_triton(shape, params=None, tol=None,
                           dtype="float32", device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "tsne_triton"
    est.tol = tol
    return est


def estimate_tsne_cutedsl(shape, params=None, tol=None,
                            dtype="float32", device="H100", **_):
    """CuteDSL backend -- swaps in fused perplexity bisect.

    Grad dominates end-to-end, so the bisect swap is invisible; parity.
    """
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "tsne_cutedsl"
    est.notes = list(est.notes) + [
        "cutedsl backend: fused perplexity bisect; grad dominates, total ~Triton."
    ]
    est.tol = tol
    return est
