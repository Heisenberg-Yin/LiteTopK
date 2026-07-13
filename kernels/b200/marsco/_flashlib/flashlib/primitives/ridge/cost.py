"""Cost model for Ridge regression.

Identical structure to :mod:`flashlib.primitives.linear_regression.cost`
plus a single fp32 add of ``alpha * I`` to the Gram matrix (negligible
relative to the cov GEMM). Composes the same sub-ops via the info
dispatcher.

Calibrated against ``benchmarks/results/full_speedup_report.md`` rows:

  shape           pred    measured  ratio
  (50K, 64)       0.30 ms 0.34 ms   0.88x
  (500K, 256)     0.85 ms 0.76 ms   1.12x
  (1M, 512)       2.4  ms 2.23 ms   1.07x
"""
from flashlib.info.estimate import Estimate
from flashlib.info.dispatch import estimate as _est


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    # Same call-graph as linear_regression -- delegate so any change to
    # the linreg model auto-propagates to ridge.
    est = _est("linear_regression", shape=shape, params=params, tol=tol,
               dtype=dtype, device=device)
    est.op_name = "ridge"
    # Rename sub-ops to ridge.* for clarity in the tree.
    for s in est.subops:
        s.op_name = s.op_name.replace("linreg.", "ridge.")
    est.notes = [
        f"N={shape[0]}, D={shape[1]}; normal equations + alpha*I "
        f"(alpha add is O(D), negligible vs cov_gemm)."
    ]
    est.suggested_config = {"backend": "cholesky", "alpha": (params or {}).get("alpha", 1.0)}
    return est


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {
        "backend": "cholesky",
        "alpha": (params or {}).get("alpha", 1.0),
        "dtype": "fp32",
    }


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_ridge_triton(shape, params=None, tol=None, dtype="float32",
                           device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "ridge_triton"
    est.tol = tol
    return est


def estimate_ridge_cutedsl(shape, params=None, tol=None, dtype="float32",
                            device="H100", **_):
    """CuteDSL backend -- swaps in CUTLASS XᵀX kernel."""
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "ridge_cutedsl"
    for s in est.subops:
        if s.op_name == "ridge.cov_gemm":
            s.op_name = "ridge.cov_gemm_cutedsl"
            s.notes = list(s.notes) + ["CuteDSL CUTLASS XᵀX; parity with Triton."]
    est.notes = list(est.notes) + ["cutedsl backend: XᵀX swapped; total ~Triton."]
    est.tol = tol
    return est
