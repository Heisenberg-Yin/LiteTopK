"""Cost model for ``truncated_svd(X)`` -> top-K singular triplets.

The flashlib path mirrors PCA: a covariance-style preconditioning GEMM,
a small eigh, then the SVD reconstruction. We compose the sub-ops via
the info dispatcher so each child Estimate carries its own subtree.

For Halko (randomised SVD, tol-gated at runtime) the model is in
:func:`estimate_truncated_svd_halko`.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.dispatch import estimate as _est


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Exact SVD via Gram (N >> D) -- decompose into cov + eigh + project."""
    params = params or {}
    N, D = shape
    K = params.get("K", params.get("n_components", 50))

    # Tall-skinny path: G = XᵀX, eigh(G), then project X V to get U S.
    cov = _est("cov_gemm", shape=(N, D), tol=tol, dtype=dtype, device=device)
    cov.op_name = "tsvd.gram_gemm"
    eig = _est("eigh", shape=(D, D), tol=tol, dtype=dtype, device=device)
    eig.op_name = "tsvd.eigh"
    proj = _est("gemm", shape=(N, D, K), tol=tol, dtype=dtype, device=device)
    proj.op_name = "tsvd.project"

    total_rt = cov.runtime_ms + eig.runtime_ms + proj.runtime_ms
    return Estimate(
        op_name="truncated_svd",
        runtime_ms=total_rt,
        flops=cov.flops + eig.flops + proj.flops,
        bytes_moved=cov.bytes_moved + eig.bytes_moved + proj.bytes_moved,
        memory_peak_gb=max(cov.memory_peak_gb, eig.memory_peak_gb,
                            proj.memory_peak_gb),
        bound="compute" if eig.bound == "compute" else "memory",
        confidence="roofline",
        n_kernel_launches=(cov.n_kernel_launches + eig.n_kernel_launches
                            + proj.n_kernel_launches),
        suggested_config={"K": K, "method": "exact"},
        subops=[cov, eig, proj],
        notes=[f"N={N}, D={D}, K={K}",
               "Exact path: gram_gemm -> eigh(top-K) -> project X V."],
        expected_residual=eig.expected_residual,
        precision_tier=eig.precision_tier,
        tol=tol,
    )


def estimate_truncated_svd_halko(shape, params=None, tol=None, dtype="float32",
                                   device="H100", **_):
    """Halko randomised SVD path (opt-in via ``tol >= 1e-2`` at runtime)."""
    params = params or {}
    N, D = shape
    K = params.get("K", params.get("n_components", 50))
    p = K + params.get("oversampling", 10)
    n_iter = params.get("n_iter", 2)

    proj = _est("gemm", shape=(N, D, p), tol=tol, dtype=dtype, device=device)
    proj.op_name = "tsvd_halko.proj"
    power = _est("gemm", shape=(N, D, p), tol=tol, dtype=dtype, device=device)
    power.op_name = "tsvd_halko.power"
    power.runtime_ms *= 2 * n_iter
    power.flops *= 2 * n_iter
    small = _est("gemm", shape=(p, D, p), tol=tol, dtype=dtype, device=device)
    small.op_name = "tsvd_halko.small_svd"

    total = proj.runtime_ms + power.runtime_ms + small.runtime_ms
    return Estimate(
        op_name="truncated_svd_halko",
        runtime_ms=total,
        flops=proj.flops + power.flops + small.flops,
        bytes_moved=proj.bytes_moved + power.bytes_moved + small.bytes_moved,
        memory_peak_gb=max(proj.memory_peak_gb, power.memory_peak_gb,
                            small.memory_peak_gb),
        bound=proj.bound, confidence="roofline",
        n_kernel_launches=proj.n_kernel_launches + 4 * n_iter + 3,
        suggested_config={"K": K, "oversampling": p - K, "n_iter": n_iter,
                           "method": "halko"},
        subops=[proj, power, small],
        notes=[f"N={N}, D={D}, K={K}, p={p}, n_iter={n_iter}",
               "Halko randomised SVD (opt-in via tol >= 1e-2)."],
        expected_residual=2.5e-2,
        precision_tier="fast",
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    return {
        "method": "halko" if (tol is not None and tol >= 1e-2) else "exact",
        "K": params.get("K", params.get("n_components", 50)),
    }


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_truncated_svd_triton(shape, params=None, tol=None,
                                    dtype="float32", device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "truncated_svd_triton"
    est.tol = tol
    return est


def estimate_truncated_svd_cutedsl(shape, params=None, tol=None,
                                     dtype="float32", device="H100", **_):
    """CuteDSL alternative -- swaps in CUTLASS gram_gemm.

    The Gram GEMM is bandwidth-bound at the cuML-comparable shapes and
    matches Triton wall-clock within ±2 %; net ~parity end-to-end.
    """
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "truncated_svd_cutedsl"
    for s in est.subops:
        if s.op_name == "tsvd.gram_gemm":
            s.op_name = "tsvd.gram_gemm_cutedsl"
            s.notes = list(s.notes) + ["CuteDSL CUTLASS gram_gemm; parity with Triton."]
    est.notes = list(est.notes) + ["cutedsl backend: gram_gemm swapped; total ~Triton."]
    est.tol = tol
    return est
