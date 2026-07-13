"""Symmetric eigendecomposition with multiple precision/performance variants.

Public API
----------
Single dispatcher (recommended):

    eigh(A, K=None, *, tol=None, backend=None)

By default ``eigh(A)`` is **exact** in the input dtype (cuSOLVER /
single-kernel Jacobi for small ``N``). ``tol`` opts into the
approximation tier:

    eigh(A, tol=8e-4)         -> QDWH-NS at N >= 5120
    eigh(A, tol=3e-3)         -> QDWH    at N >= 5120
    eigh(A, K=K, tol=1e-4)    -> Halko subspace iteration when K << N

Backend-explicit (power users):

    eigh_cusolver(A)        torch.linalg.eigh -- ~1e-7 residual
    eigh_jacobi(A, ...)     single-block Jacobi for small N
    eigh_qdwh(A, ...)       Nakatsukasa-Higham spectral D&C, ~3e-3
    eigh_qdwh_ns(A, ...)    QDWH with pure-NS polar, ~8e-4
    eigh_halko(A, K, ...)   Halko randomized truncated eigh
"""
from flashlib._lazy import lazy_attr
from flashlib.linalg.eigh import cost
from flashlib.linalg.eigh.impl import eigh, route_op_name
from flashlib.linalg.eigh.cusolver import eigh as eigh_cusolver
from flashlib.linalg.eigh.jacobi import eigh as eigh_jacobi
from flashlib.linalg.eigh.halko import halko_eigh as eigh_halko


eigh_qdwh = lazy_attr("flashlib.linalg.eigh.qdwh", "qdwh_eig")
eigh_qdwh_ns = lazy_attr("flashlib.linalg.eigh.qdwh_ns", "qdwh_eig_ns")


__all__ = [
    "eigh",
    "eigh_cusolver",
    "eigh_jacobi",
    "eigh_halko",
    "eigh_qdwh",
    "eigh_qdwh_ns",
    "route_op_name",
    "cost",
]
