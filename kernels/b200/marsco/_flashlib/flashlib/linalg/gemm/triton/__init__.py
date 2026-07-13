"""Triton kernel implementations for the GEMM variants.

Files in this package are *raw kernels* — they have kernel-specific
calling conventions (e.g., ``B`` as ``(N, K)`` rather than ``(K, N)``,
required dtype/contiguity). Public per-variant API lives one level up
in ``flashlib/linalg/gemm/<variant>.py`` (clean ``gemm(A, B)`` + cost
model + capability gating).
"""
