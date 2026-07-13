"""Cost models for eigh -- smart dispatcher + per-variant.

Each function takes ``tol`` directly. Returned Estimates have ``op_name``
set to the routed variant (e.g. ``eigh_qdwh``) so the call-stack tree is
clear.

The dispatcher in :mod:`flashlib.linalg.eigh.impl` decides which
variant runs at runtime; this module mirrors that decision via
:func:`route_op_name`.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.eigh import cusolver as _cusolver
from flashlib.linalg.eigh import jacobi as _jacobi
from flashlib.linalg.eigh import halko as _halko
from flashlib.linalg.eigh.impl import route_op_name as _route_op_name


def _N(shape):
    return shape[0] if isinstance(shape, (tuple, list)) else shape


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Cost of the smart eigh() dispatcher — picks the actually-routed variant."""
    N = _N(shape)
    K = (params or {}).get("K")
    chosen = _route_op_name(N=N, K=K, tol=tol)
    if chosen == "eigh_jacobi":
        est = _jacobi.estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    elif chosen == "eigh_qdwh":
        est = qdwh(shape, params=params, tol=tol, dtype=dtype, device=device)
    elif chosen == "eigh_qdwh_ns":
        est = qdwh_ns(shape, params=params, tol=tol, dtype=dtype, device=device)
    elif chosen == "eigh_halko":
        est = _halko.estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    else:
        est = _cusolver.estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    # Make the routed variant's name show up at this level of the call tree.
    est.op_name = chosen
    est.tol = tol
    return est


def qdwh(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """QDWH-eig — Nakatsukasa-Higham spectral D&C, ~1e-3 residual."""
    N = _N(shape)
    rt = 370.0 * (N / 8192) ** 3
    flops = int((20 / 3) * N ** 3)
    bytes_moved = N * N * 4 * 16
    return Estimate(
        op_name="eigh_qdwh",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * N * 4 * 6 / 1e9,
        bound="compute", confidence="measured", n_kernel_launches=30,
        suggested_config={"base_case": 1024, "max_depth": 1 if N < 10240 else 2},
        notes=[f"N={N}; QDWH spectral D&C, recursive split via polar factor."],
        expected_residual=3e-3, precision_tier="fast", tol=tol,
    )


def qdwh_ns(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """QDWH-eig-NS — pure Newton-Schulz polar (matmul-only critical path)."""
    N = _N(shape)
    rt = 700.0 * (N / 8192) ** 3
    flops = int((30 / 3) * N ** 3)
    bytes_moved = N * N * 4 * 24
    return Estimate(
        op_name="eigh_qdwh_ns",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * N * 4 * 6 / 1e9,
        bound="compute", confidence="measured", n_kernel_launches=50,
        suggested_config={},
        notes=[f"N={N}; QDWH spectral D&C with pure-NS polar (Polar Express)."],
        expected_residual=8e-4, precision_tier="mixed", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Picks the dispatcher's choice; mirrors the routing in eigh/__init__.py."""
    N = _N(shape)
    K = (params or {}).get("K")
    chosen = _route_op_name(N=N, K=K, tol=tol)
    out = {"variant": chosen}
    if chosen == "eigh_qdwh":
        out["base_case"] = 1024
        out["max_depth"] = 1 if N < 10240 else 2
    elif chosen == "eigh_halko":
        out["K"] = K
        out["n_iter"] = 5
        out["p"] = 30
    return out


def recommend_qdwh(shape, **_):
    N = _N(shape)
    return {"base_case": 1024, "max_depth": 1 if N < 10240 else 2}


def recommend_qdwh_ns(shape, **_):
    return {"base_case": 1024}
