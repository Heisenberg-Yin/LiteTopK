"""Invariant subspace splitting from a polar / matrix-sign factor U_p.

Given U_p ≈ sign(A_s), build orthonormal Q_+, Q_- spanning the +1/-1
eigenspaces. Used by `diag.qdwh.qdwh_eig` after the polar step.

Public:
  split_basis(U_p, backend='cholqr2') -> (Q_plus, Q_minus, k)

Backends:
  cholqr2 (default) — pure BLAS-3: thin SYRK + Cholesky + trsm, then one
                      cross-Gram-Schmidt pass + one refinement CholQR.
                      Source: `cholqr2.split_basis_cholqr2`.

  ns       — pure Newton-Schulz orthogonalization (no Cholesky, no trsm),
             one matmul-only GS pass + one NS refinement. Source:
             `diag.qdwh_ns._split_basis_ns`. Slower per pass but
             matmul-only — used by `qdwh_eig_ns`, not the main path.
"""
from flashlib.linalg.orthonormalize.cholqr2 import (
    split_basis_cholqr2,
    _chol_qr_once,
    _split_basis,
)


_BACKENDS = {
    'cholqr2': split_basis_cholqr2,
}


def split_basis(U_p, backend='cholqr2', **kwargs):
    """Build (Q_plus, Q_minus, k) for the +1/-1 eigenspaces of sign(A) ≈ U_p.

    Parameters
    ----------
    U_p : (n, n) symmetric fp32 CUDA tensor, ≈ sign(A_s)
    backend : 'cholqr2' (default)

    Returns
    -------
    Q_plus  : (n, k)
    Q_minus : (n, n - k)
    k       : int = round(tr((I + U_p) / 2))
    """
    try:
        fn = _BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"unknown split_basis backend: {backend!r}. "
            f"available: {sorted(_BACKENDS)}"
        )
    return fn(U_p, **kwargs)


cholqr2 = split_basis_cholqr2  # primitive alias

from flashlib.linalg.orthonormalize import cost  # noqa: E402

__all__ = [
    'cholqr2', 'split_basis', 'split_basis_cholqr2',
    '_chol_qr_once', '_split_basis',
    'cost',
]
