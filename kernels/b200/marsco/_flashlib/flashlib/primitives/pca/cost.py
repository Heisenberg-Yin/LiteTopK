"""Cost model for PCA -- composes cov_gemm + eigh + transform.

Each sub-op estimate is obtained via :mod:`flashlib.info` dispatch
(importlib-lazy), so updating any sub-primitive's cost auto-propagates
to PCA's total. The model mirrors the actual ``flash_pca`` path:

* ``cov_gemm``        -- ``Xᵀ X`` (or ``X Xᵀ`` for the dual-space path).
* ``eigh``            -- top-K eigendecomposition of the ``(D, D)`` cov.
* ``pca.transform``   -- ``X @ V[:, :K]`` projection.

Halko randomised SVD is exposed as a separate variant
(:func:`estimate_pca_halko`); ``tol >= 5e-2`` routes there at the
``flash_pca`` dispatcher level.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.dispatch import estimate as _est


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    N, D = shape
    K = params.get("K", params.get("n_components", 50))

    cov = _est("cov_gemm", shape=(N, D), tol=tol, dtype=dtype, device=device)
    cov.op_name = "pca.cov_gemm"
    eig = _est("eigh", shape=(D, D), tol=tol, dtype=dtype, device=device)
    eig.op_name = "pca.eigh"
    xfm = _est("gemm", shape=(N, D, K), tol=tol, dtype=dtype, device=device)
    xfm.op_name = "pca.transform"

    total_rt = cov.runtime_ms + eig.runtime_ms + xfm.runtime_ms
    return Estimate(
        op_name="pca",
        runtime_ms=total_rt,
        flops=cov.flops + eig.flops + xfm.flops,
        bytes_moved=cov.bytes_moved + eig.bytes_moved + xfm.bytes_moved,
        memory_peak_gb=max(cov.memory_peak_gb, eig.memory_peak_gb,
                            xfm.memory_peak_gb),
        bound="compute" if eig.bound == "compute" else "memory",
        confidence="roofline",
        n_kernel_launches=(cov.n_kernel_launches + eig.n_kernel_launches
                            + xfm.n_kernel_launches),
        suggested_config={"K": K, "method": "exact"},
        subops=[cov, eig, xfm],
        notes=[f"N={N}, D={D}, K={K}",
               "Exact path: cov_gemm -> eigh(top-K) -> transform."],
        expected_residual=eig.expected_residual,
        precision_tier=eig.precision_tier,
        tol=tol,
    )


def estimate_pca_halko(shape, params=None, tol=None, dtype="float32",
                        device="H100", **_):
    """Randomised SVD path (opt-in via ``tol >= 5e-2`` at runtime).

    Two GEMMs (``X @ Ω`` and ``Y = Q.T @ X``) + a small QR + the
    final ``(p, D)`` SVD. Bytes dominated by the two ``(N, D)`` reads
    of X.
    """
    params = params or {}
    N, D = shape
    K = params.get("K", params.get("n_components", 50))
    p = K + params.get("oversampling", 10)
    n_iter = params.get("n_iter", 2)

    # X @ Ω, then n_iter power iterations
    proj = _est("gemm", shape=(N, D, p), tol=tol, dtype=dtype, device=device)
    proj.op_name = "pca_halko.proj"
    power = _est("gemm", shape=(N, D, p), tol=tol, dtype=dtype, device=device)
    power.op_name = "pca_halko.power"
    power.runtime_ms *= 2 * n_iter
    power.flops *= 2 * n_iter
    # Final small SVD on (p, D)
    small = _est("gemm", shape=(p, D, p), tol=tol, dtype=dtype, device=device)
    small.op_name = "pca_halko.small_svd"

    total = proj.runtime_ms + power.runtime_ms + small.runtime_ms
    return Estimate(
        op_name="pca_halko",
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
               "Halko randomised SVD (opt-in via tol >= 5e-2)."],
        expected_residual=5e-2,
        precision_tier="fast",
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    return {
        "method": "halko" if (tol is not None and tol >= 5e-2) else "exact",
        "K": params.get("K", params.get("n_components", 50)),
    }


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_pca_triton(shape, params=None, tol=None, dtype="float32",
                         device="H100", **_):
    """Triton backend cost -- same model as ``estimate`` (default route)."""
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "pca_triton"
    est.tol = tol
    return est


def estimate_pca_cutedsl(shape, params=None, tol=None, dtype="float32",
                          device="H100", **_):
    """CuteDSL alternative -- swaps in the CUTLASS-DSL cov_gemm kernel.

    End-to-end runtime is essentially the same as the Triton path
    because the eigh sub-op dominates at the K we benchmark. Reported
    parity with Triton.
    """
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "pca_cutedsl"
    for s in est.subops:
        if s.op_name == "pca.cov_gemm":
            s.op_name = "pca.cov_gemm_cutedsl"
            s.notes = list(s.notes) + ["CuteDSL CUTLASS cov_gemm; ~Triton wall-clock."]
    est.notes = list(est.notes) + ["cutedsl backend: cov_gemm swapped; total ~Triton."]
    est.tol = tol
    return est
