"""Cost model for linear regression via the normal equations.

Composes:

* ``cov_gemm``  -- ``Xᵀ X`` (``(N, D)`` -> ``(D, D)``); dominates at
                   ``N >> D`` shapes.
* ``xty_gemv``  -- ``Xᵀ y``  (``(N, D) × (N,)`` -> ``(D,)``).
* ``solve``     -- Cholesky factorisation + triangular solve of
                   ``Xᵀ X w = Xᵀ y``; small ``O(D³ / 3)`` cubic.

Calibrated against ``benchmarks/results/full_speedup_report.md`` rows:

  shape           pred    measured  ratio
  (50K, 64)       0.30 ms 0.36 ms   0.83x
  (500K, 256)     0.85 ms 0.77 ms   1.10x
  (1M, 512)       2.4  ms 2.24 ms   1.07x

The bf16 storage variant is deliberately not exposed -- on the
synthetic data used in our benches it pushes ``Xᵀ X`` out of PSD.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.info.dispatch import estimate as _est


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    N, D = shape

    cov = _est("cov_gemm", shape=(N, D), tol=tol, dtype=dtype, device=device)
    cov.op_name = "linreg.cov_gemm"
    # Xᵀy : a GEMV; cheap (~ N*D fp32 ops, ~N*D bw bytes).
    xty_flops = 2 * N * D
    xty_bytes = (N * D + N + D) * 4
    xty_rt, xty_bound = roofline(xty_flops, xty_bytes, dtype, device,
                                   op_type="elementwise", n_launches=1)
    xty = Estimate(
        op_name="linreg.xty_gemv", runtime_ms=xty_rt,
        flops=xty_flops, bytes_moved=xty_bytes,
        memory_peak_gb=N * D * 4 / 1e9,
        bound=xty_bound, confidence="roofline", n_kernel_launches=1,
        suggested_config={}, subops=[], tol=tol,
        notes=[f"GEMV: Xᵀ y, ({N}, {D}) × ({N},)"],
    )
    # Cholesky factorisation + triangular solve: O(D³/3) FLOPs, tiny.
    chol_flops = int(D ** 3 / 3) + D * D
    chol_bytes = D * D * 4 * 4
    chol_rt, chol_bound = roofline(chol_flops, chol_bytes, "fp32", device,
                                     op_type="solver", n_launches=2)
    chol = Estimate(
        op_name="linreg.chol_solve", runtime_ms=chol_rt,
        flops=chol_flops, bytes_moved=chol_bytes,
        memory_peak_gb=D * D * 4 / 1e9,
        bound=chol_bound, confidence="roofline", n_kernel_launches=2,
        suggested_config={}, subops=[], tol=tol,
        notes=[f"Cholesky({D}) + 2 triangular solves."],
    )

    total = cov.runtime_ms + xty.runtime_ms + chol.runtime_ms
    return Estimate(
        op_name="linear_regression",
        runtime_ms=total,
        flops=cov.flops + xty.flops + chol.flops,
        bytes_moved=cov.bytes_moved + xty.bytes_moved + chol.bytes_moved,
        memory_peak_gb=max(cov.memory_peak_gb, xty.memory_peak_gb,
                            chol.memory_peak_gb),
        bound=cov.bound, confidence="roofline",
        n_kernel_launches=cov.n_kernel_launches + 3,
        suggested_config={"backend": "cholesky"}, subops=[cov, xty, chol],
        notes=[f"N={N}, D={D}; normal equations via cov_gemm + Cholesky."],
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    N, D = shape
    return {"backend": "cholesky", "dtype": "fp32"}


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_linear_regression_triton(shape, params=None, tol=None,
                                       dtype="float32", device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "linear_regression_triton"
    est.tol = tol
    return est


def estimate_linear_regression_cutedsl(shape, params=None, tol=None,
                                         dtype="float32", device="H100", **_):
    """CuteDSL backend -- swaps in CUTLASS cov_gemm only."""
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "linear_regression_cutedsl"
    for s in est.subops:
        if s.op_name == "linreg.cov_gemm":
            s.op_name = "linreg.cov_gemm_cutedsl"
            s.notes = list(s.notes) + ["CuteDSL CUTLASS cov_gemm; parity with Triton."]
    est.notes = list(est.notes) + ["cutedsl backend: cov_gemm swapped; total ~Triton."]
    est.tol = tol
    return est
