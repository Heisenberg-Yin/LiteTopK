"""Ozaki Scheme II — INT8 tensor-cores with linear precision-vs-num_moduli.

Uchino/Imamura/Ootomo 2025 (arXiv 2504.08009): CRT over INT8 GEMMs with
INT32-exact accumulator. Precision scales LINEARLY with ``num_moduli``
(~7 bits per modulus), unlike bf16x3 / tf32x6 / fp16x9 which are bounded
by the FP32-WGMMA accumulator wall (~14 bits at K=8192 regardless of
split count).

Modes (selected via ``mode`` arg):
    "ozaki2_int8"          — num_moduli=14, fastmode=False  (~full FP64)
    "ozaki2_int8_fast"     — num_moduli=7,  fastmode=True   (~SGEMM)
    "ozaki2_int8_<k>"      — num_moduli=k,  fastmode=False
    "ozaki2_int8_<k>_fast" — num_moduli=k,  fastmode=True

Requires the ``libgemmul8_shim.so`` native build to be present alongside the
``fast_gemm`` package; otherwise the function raises ``GEMMul8NotBuilt``.
"""
import torch

from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.gemm.fp32 import _shape_mkn


def gemm(A: torch.Tensor, B: torch.Tensor, *,
         num_moduli: int = 14, fastmode: bool = False) -> torch.Tensor:
    """INT8 CRT (Ozaki Scheme II) GEMM. Standard PyTorch ``A @ B`` convention.

    Note: the underlying gemmul8 wrapper expects ``B`` shaped ``(N, K)``;
    we transpose at the boundary so callers pass ``B`` in PyTorch's standard
    ``(K, N)`` shape.
    """
    from flashlib.linalg.gemm.native.gemmul8 import matmul_ozaki2
    A_ = A.contiguous()
    B_ = B.t().contiguous()  # (K, N) -> kernel-native (N, K)
    return matmul_ozaki2(A_, B_, num_moduli=num_moduli, fastmode=fastmode)


def estimate(shape, params=None, tol=None, dtype="float64", device="H100", **_):
    M, K, N = _shape_mkn(shape)
    params = params or {}
    num_moduli = params.get("num_moduli", 14 if tol is None else 7)
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N) * 8 + M * N * 8
    # H100/H200 INT8 peak ~1979 TFLOPS; CRT overhead ~ num_moduli linear.
    rt, bound = roofline(flops, bytes_moved, "fp16", device, op_type="gemm")
    rt = rt * (num_moduli * 0.55)  # observed ~78% of peak per modulus
    bits_per_modulus = 7
    res = 2.0 ** (-(num_moduli * bits_per_modulus))
    return Estimate(
        op_name="gemm_ozaki2_int8",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(M * K + K * N + M * N) * 8 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=num_moduli,
        suggested_config={"num_moduli": num_moduli, "fastmode": False},
        subops=[],
        notes=[
            f"M={M} K={K} N={N}; num_moduli={num_moduli}",
            "INT8 CRT (Ozaki Scheme II); precision scales linearly with num_moduli.",
            "Requires libgemmul8_shim.so — raises GEMMul8NotBuilt otherwise.",
        ],
        expected_residual=res, precision_tier="exact", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float64", device="H100", **_):
    if tol is None or tol <= 1e-15:
        return {"num_moduli": 14, "fastmode": False}
    if tol <= 1e-7:
        return {"num_moduli": 9, "fastmode": False}
    return {"num_moduli": 7, "fastmode": True}
