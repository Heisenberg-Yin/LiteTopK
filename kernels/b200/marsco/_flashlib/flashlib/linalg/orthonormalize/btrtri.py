"""Block-recursive triangular inverse.

Extracted from flashlib.linalg.eigh.qdwh so it can be shared with diag.kernels.cute.* without
creating an import cycle (qdwh -> kernels.cute -> qdwh).
"""
import torch


def btrtri(L, base_size=1024, gemm=None):
    """Block-recursive inverse of lower-triangular fp32 matrix.

    Formula for L = [[L11, 0], [L21, L22]]:
        L⁻¹ = [[L11⁻¹, 0], [-L22⁻¹ L21 L11⁻¹, L22⁻¹]]

    Recurse on L11 and L22; the off-diagonal block becomes two GEMMs routed
    through `gemm` (default plain `@`). With `gemm=mm_tf32_lt` this turns
    the whole inverse into a tree of tensor-core GEMMs + small base-case
    BLAS-2 solves.

    On H100 at N=16384 this is 5.8× faster than `torch.linalg.solve_triangular`
    (110 ms → 19 ms) with relative error ~1.3e-4 — solve_triangular is BLAS-2
    fp32 with no tensor-core path, BRtrtri moves 95%+ of the flops into
    TF32 GEMMs.

    Why 1024 base size: empirically optimal on H100. Smaller base overfills
    the recursion tree with many tiny GEMMs; larger base has too much BLAS-2
    work in the leaves. 1024 is where the crossover sits.
    """
    k = L.size(0)
    if k <= base_size:
        I = torch.eye(k, device=L.device, dtype=L.dtype)
        return torch.linalg.solve_triangular(L, I, upper=False)
    m = k // 2
    m = ((m + 63) // 64) * 64
    if m >= k:
        m = k // 2
    L11 = L[:m, :m].contiguous()
    L22 = L[m:, m:].contiguous()
    L21 = L[m:, :m].contiguous()
    L11_inv = btrtri(L11, base_size, gemm)
    L22_inv = btrtri(L22, base_size, gemm)
    if gemm is None:
        tmp = L21 @ L11_inv
        D = L22_inv @ tmp
    else:
        tmp = gemm(L21, L11_inv)
        D = gemm(L22_inv, tmp)
    out = torch.zeros_like(L)
    out[:m, :m] = L11_inv
    out[m:, m:] = L22_inv
    out[m:, :m] = -D
    return out
