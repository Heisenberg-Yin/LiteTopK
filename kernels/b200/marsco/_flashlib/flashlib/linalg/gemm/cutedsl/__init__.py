"""CuTeDSL kernel implementations for the GEMM variants.

Files in this package are *raw kernels* — they have kernel-specific
calling conventions. Public per-variant API lives one level up in
``flashlib/linalg/gemm/<variant>.py`` (clean ``gemm(A, B)`` + cost
model + capability gating).

The ``lib/`` subpackage holds the lower-level CuTe DSL building blocks
(generic Hopper WGMMA mainloops) that several variants share.
"""
