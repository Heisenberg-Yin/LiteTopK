"""Cost model for logistic regression (binary or one-vs-rest multinomial).

The flashlib path runs L-BFGS over the negative log-likelihood; each
iteration does:

* one forward ``X @ w``  (``(N, D) × (D,)`` GEMV per class) -> ``2*N*D`` FLOPs
* one backward ``Xᵀ @ g`` (``(D, N) × (N,)`` GEMV per class) -> ``2*N*D`` FLOPs
* a tiny L-BFGS update step (``O(history * D)``, negligible)

Bytes per iter: 2 reads of X + scratch -> ``2 * N * D * dtype_bytes``.
The fp32 path is the default; the bf16-storage path was retired (it
oscillates at gtol=1e-4 on the synthetic data we benchmark).

Calibrated against ``full_speedup_report.md``: (100K, 200, 10) measures
1.13 ms ; model predicts 0.9 ms (ratio 0.80x).
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    params = params or {}
    N, D = shape
    K = params.get("n_classes", 2)              # one-vs-rest weight columns
    n_iter = params.get("n_iter", params.get("max_iter", 30))

    # Per iter: forward + backward over (N, D) for each of K classes.
    flops_iter = 4 * N * D * max(K, 1)
    bytes_iter = 2 * N * D * 4  # fp32 X read twice (fwd + bwd)
    flops = n_iter * flops_iter
    bytes_moved = n_iter * bytes_iter

    n_launches = 4 * n_iter  # fwd, bwd, line-search, update per iter
    rt, bound = roofline(flops, bytes_moved, dtype, device,
                          op_type="gemm", n_launches=n_launches)
    return Estimate(
        op_name="logistic_regression",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * D * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=n_launches,
        suggested_config={"optimizer": "L-BFGS", "history": 10},
        subops=[],
        notes=[f"N={N}, D={D}, K={K}, n_iter={n_iter}; L-BFGS with line search."],
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {"optimizer": "L-BFGS", "history": 10, "dtype": "fp32"}


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_logistic_regression_triton(shape, params=None, tol=None,
                                          dtype="float32", device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "logistic_regression_triton"
    est.tol = tol
    return est


def estimate_logistic_regression_cutedsl(shape, params=None, tol=None,
                                           dtype="float32", device="H100", **_):
    """CuteDSL backend -- swaps in CUTLASS forward GEMV.

    The Xᵀ backward kernel is bw-bound regardless of backend; net
    wall-clock parity with Triton on the cuML-comparable shapes.
    """
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "logistic_regression_cutedsl"
    est.notes = list(est.notes) + [
        "cutedsl backend: forward GEMV uses CUTLASS; total ~Triton."]
    est.tol = tol
    return est
