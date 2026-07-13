"""Cost model for Multinomial Naive Bayes (fit + predict).

Two distinct workloads:

* **fit**     -- one ``(K, N) × (N, D) -> (K, D)`` count GEMM over the
                 one-hot class indicator; ``2*N*D*K`` FLOPs.
                 Bandwidth-bound at every shape we benchmark
                 because K is tiny (10-20).
* **predict** -- one ``(N, D) × (D, K) -> (N, K)`` GEMM of
                 log-probabilities; ``2*N*D*K`` FLOPs.
                 Same shape class as fit, dominated by HBM traffic.

Calibrated against ``full_speedup_report.md`` rows -- (200K, 500, 20)
fit measures 0.30 ms ; predict measures 0.26 ms.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Combined fit + predict cost."""
    params = params or {}
    N, D = shape
    K = params.get("n_classes", params.get("K", 10))

    # fit: one (K, N) @ (N, D) GEMM equivalent (count + lap-smoothing).
    fit_flops = 2 * N * D * K
    fit_bytes = (N * D + N * K + K * D) * 4
    fit_rt, fit_bound = roofline(fit_flops, fit_bytes, dtype, device,
                                   op_type="elementwise", n_launches=2)
    fit = Estimate(
        op_name="mnb.fit", runtime_ms=fit_rt,
        flops=fit_flops, bytes_moved=fit_bytes,
        memory_peak_gb=N * D * 4 / 1e9,
        bound=fit_bound, confidence="calibrated", n_kernel_launches=2,
        suggested_config={}, subops=[],
        notes=[f"N={N}, D={D}, K={K}; per-class count + Laplace smoothing."],
        tol=tol,
    )

    # predict: one (N, D) @ (D, K) GEMM in log-space + argmax.
    pred_flops = 2 * N * D * K
    pred_bytes = (N * D + D * K + N * K) * 4
    pred_rt, pred_bound = roofline(pred_flops, pred_bytes, dtype, device,
                                     op_type="elementwise", n_launches=2)
    pred = Estimate(
        op_name="mnb.predict", runtime_ms=pred_rt,
        flops=pred_flops, bytes_moved=pred_bytes,
        memory_peak_gb=N * D * 4 / 1e9,
        bound=pred_bound, confidence="calibrated", n_kernel_launches=2,
        suggested_config={}, subops=[],
        notes=[f"N={N}, D={D}, K={K}; log-prob GEMM + argmax."],
        tol=tol,
    )

    total = fit.runtime_ms + pred.runtime_ms
    return Estimate(
        op_name="multinomial_nb",
        runtime_ms=total,
        flops=fit.flops + pred.flops,
        bytes_moved=fit.bytes_moved + pred.bytes_moved,
        memory_peak_gb=N * D * 4 / 1e9,
        bound="memory", confidence="calibrated",
        n_kernel_launches=4,
        suggested_config={"K": K}, subops=[fit, pred],
        notes=[f"N={N}, D={D}, K={K}; fit + predict end-to-end."],
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {"alpha": 1.0, "dtype": "fp32"}


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_multinomial_nb_triton(shape, params=None, tol=None,
                                     dtype="float32", device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "multinomial_nb_triton"
    est.tol = tol
    return est


def estimate_multinomial_nb_cutedsl(shape, params=None, tol=None,
                                      dtype="float32", device="H100", **_):
    """CuteDSL backend -- swaps in CUTLASS predict-GEMM."""
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "multinomial_nb_cutedsl"
    for s in est.subops:
        if s.op_name == "mnb.predict":
            s.op_name = "mnb.predict_cutedsl"
            s.notes = list(s.notes) + ["CuteDSL CUTLASS predict GEMM; parity w/ Triton."]
    est.notes = list(est.notes) + ["cutedsl backend: predict GEMM swapped; total ~Triton."]
    est.tol = tol
    return est
