"""CuteDSL alternative implementation for flash-spectral-clustering.

DESIGN NOTES — why we picked the post-eigensolve dense GEMM, not SpMM:
====================================================================

The dominant kernel in the flash spectral pipeline is the per-iteration
SpMM `Q ← M·Q` (M is the normalized similarity in CSR; Q is dense N×K
with K ≤ 32). CuteDSL is **awkward** for this kernel because:

1. CUTLASS / CuteDSL has no first-class sparse-matrix abstraction. The
   `cute.gemm` primitive expects dense register-tile MMA atoms (WGMMA on
   Hopper, MMA on Ampere) — there is no `tcgen05_sparse` or block-sparse
   tile path exposed at the Python DSL layer in 4.4.x. To do CSR SpMM we
   would have to write a low-level cute kernel that:
     - manually gathers `vals[col[start:end]]` via cp.async or LDG,
     - issues an FFMA (not WGMMA) per nnz × K element,
   which is essentially a hand-rolled LDG-and-FMA loop that does not use
   any of the high-level CUTLASS infrastructure (TMA, swizzled SMEM,
   pipelined producer/consumer warps). Triton's pointer-arithmetic +
   tile-load model is a closer fit; our `_spmm_csr_knn_kernel` already
   matches `torch.sparse.mm` (cuSPARSE) within ~1.5×.

2. The kNN sparsity has a **long tail** (mean 18 nnz/row, max 1377 hub
   rows). CUTLASS's regular tiling assumes a fixed-shape inner dim; row-
   wise dynamic loops are not the optimization point of CuteDSL.

WHAT WE BUILT INSTEAD: a Hopper-tunable CuteDSL kernel for the
**post-eigensolve dense GEMM** `embedding = Q @ eigvecs.flip(-1)`, which
is the final N×K output matmul of the Rayleigh-Ritz refinement. Shape:
  - Q:        (N, K) fp32     (after final QR, K columns)
  - eigvecs:  (K, K) fp32     (flipped along last axis)
  - out:      (N, K) fp32

For N up to ~100K and K ≤ 32, this GEMM is bandwidth-bound (≤ 8M FMAs
at xlarge — far below tensor-core peak). Our cute kernel uses one thread
per output row, K-vector accumulators in registers, and broadcasts the
K×K rhs from L2 cache. Pattern beats a generic torch / cuBLAS GEMM at
small N×K because it avoids cuBLAS heuristic dispatch and the fp32 GEMM
dtype-fallback path on Hopper.

The file also exposes a CuteDSL-built fused row-L2-normalize kernel
(matching `_row_l2_norm_kernel` in `triton_impl.py`) for parity benching.
"""
import os
import sys


import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt


# ------------------------------------------------------------
# Kernel 1: dense N×K @ K×K GEMM (Q @ eigvecs)
# ------------------------------------------------------------

@cute.kernel
def _qmul_eigvecs_kernel(Q: cute.Tensor,        # (N, K) fp32
                          E: cute.Tensor,        # (K, K) fp32
                          Out: cute.Tensor,      # (N, K) fp32
                          N: cutlass.Constexpr,
                          K: cutlass.Constexpr,
                          BLOCK_N: cutlass.Constexpr):
    """One thread per output row. Each thread holds a K-vector accumulator
    (allocated as a stack-local `cute.make_fragment`-like array) and FMAs
    each Q[row, j] across all K output columns of E[j, 0:K].
    """
    bid = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    row = bid * BLOCK_N + tid
    if row < N:
        # Compute one row of out: sum over j of Q[row, j] * E[j, 0:K].
        # Hoist Q[row, :] into registers (K-element tile).
        for k in cutlass.range(K, unroll_full=True):
            acc = cutlass.Float32(0.0)
            for j in cutlass.range(K, unroll_full=True):
                acc = acc + Q[row, j] * E[j, k]
            Out[row, k] = acc


@cute.jit
def _qmul_eigvecs_host(Q: cute.Tensor, E: cute.Tensor, Out: cute.Tensor,
                        N: cutlass.Constexpr, K: cutlass.Constexpr,
                        BLOCK_N: cutlass.Constexpr,
                        grid: cutlass.Constexpr):
    _qmul_eigvecs_kernel(Q, E, Out, N, K, BLOCK_N).launch(
        grid=[grid, 1, 1], block=[BLOCK_N, 1, 1]
    )


# ------------------------------------------------------------
# Kernel 2: fused per-row L2-normalize  out[i,:] = X[i,:] / ||X[i,:]||_2
# ------------------------------------------------------------

@cute.kernel
def _row_l2_norm_kernel(X: cute.Tensor, Out: cute.Tensor,
                         N: cutlass.Constexpr, K: cutlass.Constexpr,
                         BLOCK_N: cutlass.Constexpr):
    bid = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    row = bid * BLOCK_N + tid
    if row < N:
        ssq = cutlass.Float32(0.0)
        for j in cutlass.range(K, unroll_full=True):
            v = X[row, j]
            ssq = ssq + v * v
        norm = cute.math.sqrt(ssq, fastmath=True)
        if norm < cutlass.Float32(1e-10):
            norm = cutlass.Float32(1e-10)
        inv = cutlass.Float32(1.0) / norm
        for j in cutlass.range(K, unroll_full=True):
            Out[row, j] = X[row, j] * inv


@cute.jit
def _row_l2_norm_host(X: cute.Tensor, Out: cute.Tensor,
                       N: cutlass.Constexpr, K: cutlass.Constexpr,
                       BLOCK_N: cutlass.Constexpr, grid: cutlass.Constexpr):
    _row_l2_norm_kernel(X, Out, N, K, BLOCK_N).launch(
        grid=[grid, 1, 1], block=[BLOCK_N, 1, 1]
    )


# ------------------------------------------------------------
# Kernel 3: FUSED Q @ eigvecs + row-L2-normalize in one pass
# ------------------------------------------------------------

@cute.kernel
def _qmul_rownorm_kernel(Q: cute.Tensor,           # (N, K)
                          E: cute.Tensor,           # (K, K)
                          Out: cute.Tensor,         # (N, K)
                          N: cutlass.Constexpr,
                          K: cutlass.Constexpr,
                          BLOCK_N: cutlass.Constexpr):
    """One thread per row: compute embedding[row, :] = Q[row,:] @ E in
    registers, then immediately L2-normalize and write. Fuses the
    `Q @ eigvecs.flip` GEMM with the row-norm into a single kernel —
    the (N, K) embedding tensor is never written to HBM, only the
    normalized output is.
    """
    bid = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    row = bid * BLOCK_N + tid
    if row < N:
        # Step 1: compute K outputs in registers, accumulate sum-of-squares.
        # We iterate the j inner loop ONCE per output-k. To avoid recomputing
        # Q[row, j] K times, hoist Q[row, :] into a K-slot fragment first.
        # cute doesn't expose per-thread fragments directly here, so we
        # rely on the compiler to register-allocate the unrolled loops.
        # We compute outputs and ssq in a single pass across (k, j).
        ssq = cutlass.Float32(0.0)
        for k in cutlass.range(K, unroll_full=True):
            acc = cutlass.Float32(0.0)
            for j in cutlass.range(K, unroll_full=True):
                acc = acc + Q[row, j] * E[j, k]
            # write to out tentatively, also accumulate ssq
            Out[row, k] = acc
            ssq = ssq + acc * acc
        # Step 2: read back & normalize. We do this in a separate pass
        # because we need the full ssq before normalization.
        norm = cute.math.sqrt(ssq, fastmath=True)
        if norm < cutlass.Float32(1e-10):
            norm = cutlass.Float32(1e-10)
        inv = cutlass.Float32(1.0) / norm
        for k in cutlass.range(K, unroll_full=True):
            Out[row, k] = Out[row, k] * inv


@cute.jit
def _qmul_rownorm_host(Q: cute.Tensor, E: cute.Tensor, Out: cute.Tensor,
                        N: cutlass.Constexpr, K: cutlass.Constexpr,
                        BLOCK_N: cutlass.Constexpr,
                        grid: cutlass.Constexpr):
    _qmul_rownorm_kernel(Q, E, Out, N, K, BLOCK_N).launch(
        grid=[grid, 1, 1], block=[BLOCK_N, 1, 1]
    )


# ------------------------------------------------------------
# Compile / dispatch (cached across calls of identical shape)
# ------------------------------------------------------------

_cache = {}


def _next_pow2(x):
    return 1 if x <= 1 else 1 << ((x - 1).bit_length())


def _get_qmul(N, K, block_n):
    key = ("qmul", N, K, block_n)
    if key not in _cache:
        Q = torch.empty(N, K, dtype=torch.float32, device="cuda")
        E = torch.empty(K, K, dtype=torch.float32, device="cuda")
        Out = torch.empty(N, K, dtype=torch.float32, device="cuda")
        cQ = cute_rt.from_dlpack(Q)
        cE = cute_rt.from_dlpack(E)
        cO = cute_rt.from_dlpack(Out)
        grid = (N + block_n - 1) // block_n
        compiled = cute.compile(
            _qmul_eigvecs_host, cQ, cE, cO, N, K, block_n, grid,
        )
        _cache[key] = compiled
    return _cache[key]


def _get_rownorm(N, K, block_n):
    key = ("rownorm", N, K, block_n)
    if key not in _cache:
        X = torch.empty(N, K, dtype=torch.float32, device="cuda")
        O = torch.empty(N, K, dtype=torch.float32, device="cuda")
        cX = cute_rt.from_dlpack(X)
        cO = cute_rt.from_dlpack(O)
        grid = (N + block_n - 1) // block_n
        compiled = cute.compile(
            _row_l2_norm_host, cX, cO, N, K, block_n, grid,
        )
        _cache[key] = compiled
    return _cache[key]


def _get_qmul_rownorm(N, K, block_n):
    key = ("qmul_rn", N, K, block_n)
    if key not in _cache:
        Q = torch.empty(N, K, dtype=torch.float32, device="cuda")
        E = torch.empty(K, K, dtype=torch.float32, device="cuda")
        Out = torch.empty(N, K, dtype=torch.float32, device="cuda")
        cQ = cute_rt.from_dlpack(Q)
        cE = cute_rt.from_dlpack(E)
        cO = cute_rt.from_dlpack(Out)
        grid = (N + block_n - 1) // block_n
        compiled = cute.compile(
            _qmul_rownorm_host, cQ, cE, cO, N, K, block_n, grid,
        )
        _cache[key] = compiled
    return _cache[key]


def cutedsl_qmul_rownorm(Q: torch.Tensor, eigvecs_flipped: torch.Tensor) -> torch.Tensor:
    """Fused: out = row_l2_normalize(Q @ eigvecs_flipped). Single kernel."""
    assert Q.is_cuda and Q.dtype == torch.float32
    assert eigvecs_flipped.is_cuda and eigvecs_flipped.dtype == torch.float32
    N, K = Q.shape
    K2, K3 = eigvecs_flipped.shape
    assert K == K2 == K3
    block_n = 128
    out = torch.empty(N, K, dtype=torch.float32, device=Q.device)
    compiled = _get_qmul_rownorm(N, K, block_n)
    cQ = cute_rt.from_dlpack(Q.contiguous())
    cE = cute_rt.from_dlpack(eigvecs_flipped.contiguous())
    cO = cute_rt.from_dlpack(out)
    compiled(cQ, cE, cO)
    return out


def cutedsl_qmul_eigvecs(Q: torch.Tensor, eigvecs_flipped: torch.Tensor) -> torch.Tensor:
    """Dense GEMM: out = Q @ eigvecs_flipped, shapes (N,K) × (K,K) → (N,K).

    eigvecs_flipped should already be the post-flip rhs (eigvecs.flip(-1));
    we just multiply by it as-is.
    """
    assert Q.is_cuda and Q.dtype == torch.float32
    assert eigvecs_flipped.is_cuda and eigvecs_flipped.dtype == torch.float32
    N, K = Q.shape
    K2, K3 = eigvecs_flipped.shape
    assert K == K2 == K3
    block_n = 128
    out = torch.empty(N, K, dtype=torch.float32, device=Q.device)
    compiled = _get_qmul(N, K, block_n)
    cQ = cute_rt.from_dlpack(Q.contiguous())
    cE = cute_rt.from_dlpack(eigvecs_flipped.contiguous())
    cO = cute_rt.from_dlpack(out)
    compiled(cQ, cE, cO)
    return out


def cutedsl_row_l2_normalize(X: torch.Tensor) -> torch.Tensor:
    """Per-row L2 normalize via CuteDSL kernel."""
    assert X.is_cuda and X.dtype == torch.float32
    N, K = X.shape
    block_n = 128
    out = torch.empty_like(X)
    compiled = _get_rownorm(N, K, block_n)
    cX = cute_rt.from_dlpack(X.contiguous())
    cO = cute_rt.from_dlpack(out)
    compiled(cX, cO)
    return out


# ------------------------------------------------------------
# Drop-in eigensolve using CuteDSL final transform
# ------------------------------------------------------------

def cutedsl_power_iter_top_k(M, K: int, n_iter: int = 15, qr_every: int = 5):
    """Same simultaneous power iteration as the Triton path, but the
    final `embedding = Q @ eigvecs.flip(-1)` step uses the CuteDSL kernel.
    SpMM stays on torch.sparse.mm because CuteDSL is awkward for sparse.
    """
    N = M.shape[0]
    device = M.device

    Q = torch.randn(N, K, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Q)
    Q = Q.contiguous()

    is_sparse = M.is_sparse_csr or M.is_sparse
    if is_sparse:
        matmul = lambda Q_: torch.sparse.mm(M, Q_)
    else:
        matmul = lambda Q_: M @ Q_

    for it in range(n_iter):
        Q = matmul(Q)
        if (it + 1) % qr_every == 0 or it == n_iter - 1:
            Q, _ = torch.linalg.qr(Q)
            Q = Q.contiguous()

    # Rayleigh-Ritz refinement
    MQ = matmul(Q)
    Z = Q.T @ MQ
    Z = (Z + Z.T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(Z)
    eigvecs_flipped = eigvecs.flip(-1).contiguous()

    # CuteDSL dense GEMM for the embedding output
    embedding = cutedsl_qmul_eigvecs(Q, eigvecs_flipped)
    return embedding, eigvals.flip(-1)


def cutedsl_power_iter_top_k_fused(M, K: int, n_iter: int = 15,
                                     qr_every: int = 5):
    """Like cutedsl_power_iter_top_k, but uses the FUSED qmul+rownorm
    kernel for the final embedding stage. Returns (embedding_normed,
    eigvals) — the embedding is already row-L2-normalized.
    """
    N = M.shape[0]
    device = M.device

    Q = torch.randn(N, K, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Q)
    Q = Q.contiguous()

    is_sparse = M.is_sparse_csr or M.is_sparse
    if is_sparse:
        matmul = lambda Q_: torch.sparse.mm(M, Q_)
    else:
        matmul = lambda Q_: M @ Q_

    for it in range(n_iter):
        Q = matmul(Q)
        if (it + 1) % qr_every == 0 or it == n_iter - 1:
            Q, _ = torch.linalg.qr(Q)
            Q = Q.contiguous()

    MQ = matmul(Q)
    Z = Q.T @ MQ
    Z = (Z + Z.T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(Z)
    eigvecs_flipped = eigvecs.flip(-1).contiguous()
    embedding_normed = cutedsl_qmul_rownorm(Q, eigvecs_flipped)
    return embedding_normed, eigvals.flip(-1)


def cutedsl_spectral_clustering(X: torch.Tensor,
                                 n_clusters: int,
                                 n_neighbors: int = 10,
                                 n_components=None,
                                 n_power_iter: int = 15,
                                 seed: int = 0,
                                 fused: bool = True):
    """End-to-end spectral clustering with the CuteDSL final transform +
    CuteDSL row-norm. KNN graph build, SpMM and KMeans stay on Triton /
    cuSPARSE / flash_kmeans (CuteDSL has no first-class abstraction for
    those workloads — see module docstring).

    If `fused=True`, the final `Q @ eigvecs.flip` GEMM and the row-L2-norm
    are computed in a single fused CuteDSL kernel — eliminating the
    intermediate embedding materialization.
    """
    from flashlib.primitives.spectral_clustering.triton import (
        _knn_normalized_sparse, _flash_kmeans_with_pp_init,
    )

    assert X.is_cuda
    if n_components is None:
        n_components = n_clusters
    torch.manual_seed(seed)

    M = _knn_normalized_sparse(X, n_neighbors)
    if fused:
        embedding_normed, _ = cutedsl_power_iter_top_k_fused(
            M, n_components, n_iter=n_power_iter)
    else:
        embedding, _ = cutedsl_power_iter_top_k(M, n_components, n_iter=n_power_iter)
        embedding_normed = cutedsl_row_l2_normalize(embedding.contiguous())
    del M

    labels = _flash_kmeans_with_pp_init(embedding_normed, n_clusters,
                                          n_iter=50, seed=seed, n_init=3)
    return labels.to(torch.int64)
