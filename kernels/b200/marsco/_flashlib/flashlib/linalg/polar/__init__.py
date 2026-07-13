"""Matrix sign / polar decomposition for real symmetric A — multi-variant.

For symmetric A, the polar factor U (where A = U|A|) equals the matrix sign
function sign(A). Four backends with different precision/performance profiles:

    backend             time/N=8192   orth_err   residual    notes
    qdwh_hybrid          ~150 ms       ~1e-4      1e-3-5e-3   default; mixed prec
    polar_express        ~256 ms       ~5e-5      ~6e-4       all-matmul (no Chol)
    polar_express_warm   ~125 ms       ~5e-5      ~5e-3       fastest, occasional FAIL
    zolo                 ~600 ms       ~1e-7      ~1e-6       tightest, ZOLO-PD

Public API:

    polar(A, tol=None, backend=None)
        Smart dispatcher; tol=None defaults to zolo (most accurate).
        tol >= 1e-3 picks qdwh_hybrid (fastest in budget).
    msign(A, backend="qdwh_hybrid"|"polar_express"|"polar_express_warm"|"zolo")
        flash-diag-compatible explicit-backend dispatcher.
    polar_qdwh_hybrid(A), polar_express(A), polar_express_warm(A), polar_zolo(A)
        Direct backend functions — also top-level via flashlib.polar_<backend>.

Pareto frontier on (runtime, residual): qdwh_hybrid, polar_express, zolo.
polar_express_warm is dominated when it FAILs but otherwise wins on speed —
keep it for runs where residual budget is loose.
"""
from flashlib.linalg.polar.qdwh_hybrid import polar_qdwh_hybrid
from flashlib.linalg.polar.polar_express import (
    polar_polar_express as polar_express,
    polar_polar_express_warm as polar_express_warm,
)
from flashlib.linalg.polar.zolo import polar_zolo
from flashlib.linalg.polar import cost

# Back-compat re-exports (used by linalg/eigh/qdwh.py + qdwh_ns.py).
from flashlib.linalg.polar.qdwh_hybrid import (  # noqa: F401
    _qdwh_polar, _qdwh_chol_step, _kenney_laub_step,
    _spectral_norm_estimate, _mm_tf32, _mm_3xtf32, _mm_3xbf16_smart,
)
from flashlib.linalg.polar.polar_express import _polar_ns, _polar_warmstart_pe  # noqa: F401
from flashlib.linalg.polar.zolo import zolo_polar  # noqa: F401


_BACKENDS = {
    "qdwh_hybrid":         polar_qdwh_hybrid,
    "polar_express":       polar_express,
    "polar_express_warm":  polar_express_warm,
    "zolo":                polar_zolo,
}


# Tighter to looser; first variant whose residual ≤ tol wins.
_RESIDUAL_PREFERENCE = [
    ("zolo",                1e-6),
    ("polar_express",       6e-4),
    ("qdwh_hybrid",         3e-3),
    ("polar_express_warm",  5e-3),
]


def _pick_by_tol(tol: float | None) -> str:
    """tol=None -> zolo (most accurate). Otherwise fastest qualifying."""
    if tol is None or tol <= 0:
        return "zolo"
    for name, res in reversed(_RESIDUAL_PREFERENCE):
        if res <= tol:
            return name
    return "zolo"


def msign(A, backend: str = "qdwh_hybrid", **kwargs):
    """flash-diag-style: compute U = sign(A), backend selected by name."""
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown msign backend {backend!r}. available: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[backend](A, **kwargs)


def polar(A, *, tol: float | None = None, backend: str | None = None, **kwargs):
    """Multi-variant polar / matrix-sign dispatcher.

    Args:
        A: symmetric (N, N).
        tol: residual tolerance (relative). None -> zolo (~1e-6, exact).
            tol=1e-3 -> qdwh_hybrid (default 'fast' option).
            tol=5e-3 -> polar_express_warm (fastest).
        backend: explicit ('qdwh_hybrid'|'polar_express'|'polar_express_warm'|'zolo').
    """
    if backend is None:
        backend = _pick_by_tol(tol)
    return msign(A, backend=backend, **kwargs)


__all__ = [
    "polar", "msign",
    "polar_qdwh_hybrid",
    "polar_express", "polar_express_warm",
    "polar_zolo",
    "cost",
]
