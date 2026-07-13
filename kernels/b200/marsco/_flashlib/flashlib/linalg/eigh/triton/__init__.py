"""eigh triton backend.

Re-exports the top-level functions / classes / constants from each
component file. ``@triton.jit`` kernels stay private to their file
(call them via the Python wrapper that lives next to them).

Component files:

* :mod:`householder` -- Householder tridiagonalisation + unrolled
  QR finaliser. Powers :func:`flashlib.linalg.eigh.triton_eigh`,
  the small-D (D <= ~256) dense path used by PCA / TruncSVD /
  Halko subspace iteration.
* :mod:`jacobi`      -- Row-cyclic Givens-Jacobi for N <= 128.
  Powers :func:`flashlib.linalg.eigh.eigh_jacobi`, an opt-in
  backend that beats cuSOLVER at very small N (N <= 16) without
  the ``load_inline`` CUDA / C++ compile step the old
  implementation needed.
"""
from flashlib.linalg.eigh.triton.householder import (
    _eigh_cpu_initialized,
    triton_eigh,
)
from flashlib.linalg.eigh.triton.jacobi import triton_jacobi_eigh

__all__ = [
    "triton_eigh",
    "triton_jacobi_eigh",
]
