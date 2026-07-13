"""Native (cuBLAS / .so shim) GEMM kernels.

Files in this package wrap vendor libraries — cuBLAS (via PyTorch),
GEMMul8 (Ozaki-II INT8 native shim). Public per-variant API lives
one level up in ``flashlib/linalg/gemm/<variant>.py``.
"""
