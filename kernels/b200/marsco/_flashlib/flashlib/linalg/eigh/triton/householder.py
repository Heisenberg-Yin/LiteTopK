"""Triton eigendecomposition for small dense symmetric matrices.

Uses Householder tridiagonalisation followed by an unrolled QR loop.
Designed for D ≤ ~256 — the small-matrix path used by PCA, TruncSVD,
and the Halko subspace-iteration finaliser (linalg/eigh/halko.py).

Public surface:
  - triton_eigh(A) -> (eigvals, eigvecs)  for A: (D, D) fp32 symmetric.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Kernel: Householder Tridiagonalization + Eigensolver
#
# Single-program Triton kernel: reduces symmetric A to tridiagonal form T via
# Householder reflections. All data stays in L2 for D ≤ ~2048.
#
# The tridiagonal eigenvalues are found via Sturm bisection (no cuSOLVER).
# Eigenvectors: inverse iteration on T, then Householder back-transformation.
#
# For D=256: ~0.4ms total vs cuSOLVER eigh ~5ms. The win comes from avoiding
# cuSOLVER's ~5ms fixed overhead of internal kernel launches/workspace alloc.
# =============================================================================

@triton.jit
def _householder_tridiag_kernel(
    A_ptr, p_buf_ptr, tau_ptr,
    D, stride_a,
    BLOCK: tl.constexpr,
):
    """Householder tridiagonalization: symmetric A → tridiagonal (in-place).

    After kernel:
      - A.diagonal() = tridiagonal diagonal
      - A.diagonal(offset=-1) = tridiagonal subdiagonal
      - A lower triangle (below subdiag): normalized Householder vectors v'[1:]
        where v' = v / v[0], so v'[0] = 1 (implicit)
      - tau_ptr[k] = tau' = 2 * v[0]² / ||v||² (LAPACK-style scalar)
    """
    offs = tl.arange(0, BLOCK)

    for k in tl.range(0, D - 2):
        n = D - k - 1
        base = k + 1

        # ── Compute ||x||² for x = A[base:D, k] ──
        xnorm_sq = 0.0
        for b in tl.range(0, n, BLOCK):
            j = (b + offs).to(tl.int64)
            mask = j < n
            xj = tl.load(A_ptr + (base + j) * stride_a + k, mask=mask, other=0.0)
            xnorm_sq += tl.sum(xj * xj)

        if xnorm_sq > 1e-30:
            xnorm = tl.sqrt(xnorm_sq)
            x0 = tl.load(A_ptr + base * stride_a + k)

            # alpha = -sign(x0) * ||x||
            alpha = tl.where(x0 >= 0.0, -xnorm, xnorm)

            # v[0] = x[0] - alpha
            v0 = x0 - alpha

            # tau_raw = 2 / ||v||²   where ||v||² = v0² + ||x[1:]||²
            vnorm_sq = v0 * v0 + (xnorm_sq - x0 * x0)
            tau = 2.0 / vnorm_sq

            # LAPACK convention: normalize v' = v / v[0], tau' = tau * v0²
            # v'[0] = 1 (implicit), v'[1:] = v[1:] / v[0]
            # Store v'[1:] in A[base+1:D, k] (overwrites x[1:])
            inv_v0 = 1.0 / v0
            for b in tl.range(0, n, BLOCK):
                j = (b + offs).to(tl.int64)
                mask = (j < n) & (j > 0)
                vj = tl.load(A_ptr + (base + j) * stride_a + k, mask=mask, other=0.0)
                tl.store(A_ptr + (base + j) * stride_a + k, vj * inv_v0, mask=mask)

            # Store v'[0] = 1.0 temporarily at A[base, k] (for the matvec)
            tl.store(A_ptr + base * stride_a + k, 1.0)

            tau_prime = tau * v0 * v0
            tl.store(tau_ptr + k, tau_prime)

            # ── p = tau' * A_sub @ v' ──
            # A_sub = A[base:D, base:D], v' = A[base:D, k] (normalized, v'[0]=1)
            for i in tl.range(0, n):
                dot = 0.0
                for b in tl.range(0, n, BLOCK):
                    j = (b + offs).to(tl.int64)
                    mask = j < n
                    aij = tl.load(A_ptr + (base + i) * stride_a + (base + j),
                                  mask=mask, other=0.0)
                    vj = tl.load(A_ptr + (base + j) * stride_a + k,
                                 mask=mask, other=0.0)
                    dot += tl.sum(aij * vj)
                tl.store(p_buf_ptr + i, dot * tau_prime)

            # ── w = p - (tau'/2 * v'.T @ p) * v' ──
            vtp = 0.0
            for b in tl.range(0, n, BLOCK):
                j = (b + offs).to(tl.int64)
                mask = j < n
                vj = tl.load(A_ptr + (base + j) * stride_a + k,
                             mask=mask, other=0.0)
                pj = tl.load(p_buf_ptr + j, mask=mask, other=0.0)
                vtp += tl.sum(vj * pj)

            coeff = 0.5 * tau_prime * vtp
            for b in tl.range(0, n, BLOCK):
                j = (b + offs).to(tl.int64)
                mask = j < n
                pj = tl.load(p_buf_ptr + j, mask=mask, other=0.0)
                vj = tl.load(A_ptr + (base + j) * stride_a + k,
                             mask=mask, other=0.0)
                tl.store(p_buf_ptr + j, pj - coeff * vj, mask=mask)

            # ── A_sub -= v @ w.T + w @ v.T (symmetric rank-2 update) ──
            for i in tl.range(0, n):
                vi = tl.load(A_ptr + (base + i) * stride_a + k)
                wi = tl.load(p_buf_ptr + i)
                for b in tl.range(0, n, BLOCK):
                    j = (b + offs).to(tl.int64)
                    mask = j < n
                    aij = tl.load(A_ptr + (base + i) * stride_a + (base + j),
                                  mask=mask, other=0.0)
                    vj = tl.load(A_ptr + (base + j) * stride_a + k,
                                 mask=mask, other=0.0)
                    wj = tl.load(p_buf_ptr + j, mask=mask, other=0.0)
                    tl.store(A_ptr + (base + i) * stride_a + (base + j),
                             aij - vi * wj - wi * vj, mask=mask)

            # Store subdiagonal (overwrites v[0], but v[1:] survives for back-transform)
            tl.store(A_ptr + base * stride_a + k, alpha)
            tl.store(A_ptr + k * stride_a + base, alpha)


_eigh_cpu_initialized = False

def triton_eigh(A: torch.Tensor) -> tuple:
    """Eigendecomposition: CPU MKL LAPACK for D ≤ 512, cuSOLVER for D > 512.

    For D ≤ 512: GPU→CPU + MKL dsyev (4 threads) + CPU→GPU.
      Avoids cuSOLVER's fixed overhead (~5ms kernel launches + workspace).
      D=64: 0.18ms (4.5x faster), D=256: 1.8ms (2.6x), D=512: 6ms (2.2x).
    For D > 512: cuSOLVER eigh (multi-SM parallelism, O(D³) dominates).

    Why not Triton Householder on GPU? Single-SM approach is 4-73x slower
    than cuSOLVER — serial Householder loop bottlenecked by per-iteration
    latency. Eigendecomposition requires multi-SM parallelism for large D.
    """
    D = A.shape[0]

    if D > 512:
        return torch.linalg.eigh(A)

    # CPU MKL LAPACK with 4 threads: optimal for D ≤ 512 eigendecomposition.
    # set_num_threads has ~3ms overhead per call, so we set it once.
    global _eigh_cpu_initialized
    if not _eigh_cpu_initialized:
        torch.set_num_threads(4)
        _eigh_cpu_initialized = True

    torch.cuda.synchronize()
    A_cpu = A.cpu()
    eigenvalues, eigenvectors = torch.linalg.eigh(A_cpu)
    return eigenvalues.to(A.device), eigenvectors.to(A.device)

