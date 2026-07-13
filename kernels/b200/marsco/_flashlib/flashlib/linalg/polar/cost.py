"""Cost models for polar / matrix-sign backends.

All four backends share the input shape (N, N symmetric A) and output U = sign(A).
What differs is the runtime / residual / orth profile.

Numbers anchored to flash-diag's H100 SXM5 measurements (random Gaussian
symmetric fp32 inputs, single-iter):

    backend             time/N=8192   orth_err   residual    notes
    qdwh_hybrid          ~150 ms       ~1e-4      1e-3-5e-3   diag default
    polar_express        ~256 ms       ~5e-5      ~6e-4       no Cholesky
    polar_express_warm   ~125 ms       ~5e-5      ~5e-3       occasional FAIL
    zolo                 ~600 ms       ~1e-7      ~1e-6       tightest
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def _N(shape) -> int:
    if isinstance(shape, (tuple, list)):
        return shape[0]
    return shape


def _gemm_flops(N: int, n_gemms: int) -> tuple[float, float]:
    """Cost of `n_gemms` (N x N x N) matmuls."""
    return n_gemms * 2 * N ** 3, n_gemms * 3 * N * N * 4


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Smart dispatcher: defaults to qdwh_hybrid (current diag default)."""
    return qdwh_hybrid(shape, params=params, dtype=dtype, device=device)


def qdwh_hybrid(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """2 QDWH-Cholesky iters + 2 Kenney-Laub NS iters; mixed precision (TF32/3xbf16)."""
    N = _N(shape)
    # ~10-12 N x N matmuls equivalent at mixed precision (TF32 / 3xbf16)
    flops, bytes_moved = _gemm_flops(N, 12)
    rt, bound = roofline(flops, bytes_moved, "tf32", device, op_type="gemm")
    return Estimate(
        op_name='polar',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * N * 4 * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=14,
        suggested_config={}, subops=[],
        notes=[
            f"N={N}; QDWH-hybrid (Nakatsukasa-Higham 2013).",
            "S2 fp32 SYRK + Cholesky + chol_solve, S3 TF32, S4-S5 3xbf16 KL.",
        ],
        expected_residual=3e-3, precision_tier="fast", tol=tol,
    )


def polar_express(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Pure Newton-Schulz: 12 minimax-optimal odd-quintic iters (Amsel 2025)."""
    N = _N(shape)
    # 3 GEMMs per iter * 12 iters = 36 GEMMs; can run in 3xbf16
    flops, bytes_moved = _gemm_flops(N, 36)
    rt, bound = roofline(flops, bytes_moved, "bf16", device, op_type="gemm")
    return Estimate(
        op_name='polar',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * N * 4 * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=36,
        suggested_config={"n_iter": 12}, subops=[],
        notes=[
            f"N={N}; pure NS quintic (Polar Express, arXiv:2505.16932).",
            "All matmul, no Cholesky/trsm — highest TC utilization but slowest end-to-end.",
        ],
        expected_residual=6e-4, precision_tier="mixed", tol=tol,
    )


def polar_express_warm(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """QDWH-Cholesky warm-start (n_qdwh iters) + short Polar Express tail."""
    N = _N(shape)
    flops, bytes_moved = _gemm_flops(N, 18)
    rt, bound = roofline(flops, bytes_moved, "bf16", device, op_type="gemm")
    return Estimate(
        op_name='polar',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * N * 4 * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=20,
        suggested_config={"n_qdwh": 1}, subops=[],
        notes=[
            f"N={N}; QDWH-Chol warm start + Polar Express tail.",
            "Faster than pure PE but brings Cholesky cond(Z)≈2.3e6 back; rare FAIL.",
        ],
        expected_residual=5e-3, precision_tier="fast", tol=tol,
    )


def zolo(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """ZOLO-PD type-3 Zolotarev rational approx; 2 iters to fp64-grade precision."""
    N = _N(shape)
    flops, bytes_moved = _gemm_flops(N, 24)
    rt, bound = roofline(flops, bytes_moved, "fp32", device, op_type="solver")
    return Estimate(
        op_name='polar',
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * N * 4 * 8 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=24,
        suggested_config={"n_iter": 2}, subops=[],
        notes=[
            f"N={N}; ZOLO-PD type-3 (Nakatsukasa-Freund 2016).",
            "~3-4x slower than QDWH but ~10x tighter orth — pick when precision matters.",
        ],
        expected_residual=1e-6, precision_tier="exact", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {"backend": "qdwh_hybrid"}


def recommend_qdwh_hybrid(shape, **_): return {}
def recommend_polar_express(shape, **_): return {"n_iter": 12}
def recommend_polar_express_warm(shape, **_): return {"n_qdwh": 1, "n_pe": 8}
def recommend_zolo(shape, **_): return {"n_iter": 2}
