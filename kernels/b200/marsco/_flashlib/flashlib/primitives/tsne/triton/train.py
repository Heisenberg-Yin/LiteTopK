"""flash-tsne: end-to-end Triton t-SNE.

Pipeline:
  1. Pairwise squared distances (dense N×N)
  2. Vectorized binary search for β_i (perplexity calibration), all rows in parallel
  3. Symmetric P matrix
  4. SGD with early-exaggeration + final phase, using existing Triton kernels
     (triton_tsne_qsum + triton_tsne_grad)
"""

import math

import torch
import triton
import triton.language as tl

from flashlib.primitives.tsne.triton.grad import (
    triton_tsne_qsum, triton_tsne_grad,
)
from flashlib.primitives.tsne.triton.grad_blocked import (
    triton_tsne_grad_blocked, block_p_matrix,
)



# =============================================================================
# Fused P-matrix kernel: per-row Triton bisection in registers. Streams
# d_centered from L2 across the 50 bisect iterations so HBM traffic is
# ~1x N^2 (vs. the 50x N^2 that a naive per-iter exp() would require).
# =============================================================================

@triton.jit(do_not_specialize=["N", "LOG_PERPLEXITY"])
def _tsne_bisect_kernel(
    DISTS_SQ_ptr,        # (N, N) float32 — raw squared pairwise distances
    BETA_ptr,            # (N,) float32 — output: β_i
    DMIN_ptr,            # (N,) float32 — output: per-row off-diag d_min
    N,
    LOG_PERPLEXITY,
    N_BISECT: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Per-row 50-iter Triton bisection for β_i.

    Each program handles BLOCK_I rows. Pass 0 finds d_min[i] = min_{j≠i}
    dists_sq[i, j]. Then 50 iters of bisection: each iter streams j, computes
    sum_P + sum_Pd in registers, updates β via H = log(sum_P) + β·sum_Pd/sum_P
    until H matches log(perplexity).

    HBM per row: one read of dists_sq[i, :]; subsequent iters hit L2 (the row
    is ~120 KB at N=30K, well within L2's 60 MB).

    Output: β_i and d_min[i] — used by the second kernel to emit P[i, :].
    """
    pid_i = tl.program_id(0)
    i_offs = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_offs < N

    n_i64 = N.to(tl.int64)
    i_addr = i_offs.to(tl.int64) * n_i64

    # Pass 0: per-row min over off-diagonal entries.
    d_min = tl.full((BLOCK_I,), 1e30, tl.float32)
    for j_start in tl.range(0, N, BLOCK_J):
        j_offs = j_start + tl.arange(0, BLOCK_J)
        j_mask = j_offs < N
        not_diag = i_offs[:, None] != j_offs[None, :]
        valid = i_mask[:, None] & j_mask[None, :] & not_diag
        addr = i_addr[:, None] + j_offs[None, :].to(tl.int64)
        d = tl.load(DISTS_SQ_ptr + addr, mask=valid, other=1e30)
        d_min = tl.minimum(d_min, tl.min(d, axis=1))

    lo = tl.zeros((BLOCK_I,), tl.float32)
    hi = tl.full((BLOCK_I,), 1e10, tl.float32)
    beta = tl.full((BLOCK_I,), 1.0, tl.float32)

    for _ in tl.range(0, N_BISECT):
        sum_P = tl.zeros((BLOCK_I,), tl.float32)
        sum_Pd = tl.zeros((BLOCK_I,), tl.float32)
        for j_start in tl.range(0, N, BLOCK_J):
            j_offs = j_start + tl.arange(0, BLOCK_J)
            j_mask = j_offs < N
            not_diag = i_offs[:, None] != j_offs[None, :]
            valid = i_mask[:, None] & j_mask[None, :] & not_diag
            addr = i_addr[:, None] + j_offs[None, :].to(tl.int64)
            d = tl.load(DISTS_SQ_ptr + addr, mask=valid, other=0.0)
            dc = d - d_min[:, None]
            p = tl.exp(-beta[:, None] * dc)
            p = tl.where(valid, p, 0.0)
            sum_P += tl.sum(p, axis=1)
            sum_Pd += tl.sum(p * dc, axis=1)
        sum_P_safe = sum_P + 1e-12
        H = tl.log(sum_P_safe) + beta * sum_Pd / sum_P_safe
        too_high = H > LOG_PERPLEXITY
        lo = tl.where(too_high, beta, lo)
        hi = tl.where(too_high, hi, beta)
        beta = (lo + hi) * 0.5

    tl.store(BETA_ptr + i_offs, beta, mask=i_mask)
    tl.store(DMIN_ptr + i_offs, d_min, mask=i_mask)


@triton.jit(do_not_specialize=["N"])
def _tsne_pmat_emit_kernel(
    DISTS_SQ_ptr,        # (N, N) float32
    BETA_ptr,            # (N,) float32
    DMIN_ptr,            # (N,) float32
    P_OUT_ptr,           # (N, N) float32 — output: P_unnorm (off_diag · exp(-β(d-d_min)))
    N,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Emit P_unnorm[i, j] = (j ≠ i) · exp(-β_i · (dists_sq[i, j] - d_min[i])).

    Normalization (per-row sum to 1) and symmetrize are done in torch as a
    one-pass N² op, avoiding a second streaming pass over the j axis.
    """
    pid_i = tl.program_id(0)
    i_offs = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_offs < N
    n_i64 = N.to(tl.int64)
    i_addr = i_offs.to(tl.int64) * n_i64

    beta = tl.load(BETA_ptr + i_offs, mask=i_mask, other=1.0)
    d_min = tl.load(DMIN_ptr + i_offs, mask=i_mask, other=0.0)

    for j_start in tl.range(0, N, BLOCK_J):
        j_offs = j_start + tl.arange(0, BLOCK_J)
        j_mask = j_offs < N
        not_diag = i_offs[:, None] != j_offs[None, :]
        valid = i_mask[:, None] & j_mask[None, :] & not_diag
        addr = i_addr[:, None] + j_offs[None, :].to(tl.int64)
        d = tl.load(DISTS_SQ_ptr + addr, mask=valid, other=0.0)
        dc = d - d_min[:, None]
        p = tl.exp(-beta[:, None] * dc)
        p = tl.where(valid, p, 0.0)
        tl.store(P_OUT_ptr + addr, p, mask=valid)


def _pick_pmat_tile(N: int):
    """Choose (BLOCK_I, BLOCK_J, num_warps) for the fused P-matrix kernel.

    Tuned on H200 with BLOCK_I/BLOCK_J/num_warps sweep at large + huge:
      - BLOCK_I=4 wins over 8/16 at every tested N (smaller register tile
        → higher occupancy; 4 rows per CTA is enough to amortise the j
        stream, more rows just bloat registers without any throughput gain).
      - BLOCK_J=128 is forced because BLOCK_J ≥ 256 trips a Triton 3.6
        `TritonGPUOptimizeThreadLocality` MLIR-pass bug on this deeply-
        nested loop. L2 absorbs the streaming cost regardless.
      - num_warps=4 wins over 8 (lower instr issue contention on the
        scalar inner reductions).
    """
    return 4, 128, 4


def _compute_p_matrix(X, perplexity=30.0, n_bisect=50):
    """Compute symmetric P matrix.

    Pairwise distances via `torch.cdist` (GEMM-bound, leave to torch),
    then a single fused Triton kernel does the per-row 50-iter bisection
    AND the final exp + normalise. The bisection streams `dists_sq` from
    L2 across iterations (~1x N^2 HBM traffic instead of 50x N^2 for the
    naive per-iter exp materialisation).

    Symmetrize + clamp stay in torch (one extra N² read/write).

    Memory: dists_sq is N×N fp32 (3.6 GB at N=30K).
    """
    N = X.shape[0]
    device = X.device
    target = math.log(perplexity)

    dists_sq = torch.cdist(X, X, p=2).pow(2).contiguous()
    beta = torch.empty(N, device=device, dtype=torch.float32)
    d_min = torch.empty(N, device=device, dtype=torch.float32)
    # zeros, not empty: emit kernel masks off-diagonal stores; diag must be 0.
    P_unnorm = torch.zeros(N, N, device=device, dtype=torch.float32)

    BLOCK_I, BLOCK_J, num_warps = _pick_pmat_tile(N)
    grid = (triton.cdiv(N, BLOCK_I),)
    _tsne_bisect_kernel[grid](
        dists_sq, beta, d_min, int(N), float(target),
        N_BISECT=n_bisect,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
        num_warps=num_warps,
        num_stages=2,
    )
    _tsne_pmat_emit_kernel[grid](
        dists_sq, beta, d_min, P_unnorm, int(N),
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
        num_warps=num_warps,
    )

    # Per-row normalization (torch fused softmax-style: 1 N² read + write).
    P = P_unnorm / (P_unnorm.sum(dim=1, keepdim=True) + 1e-12)
    # Symmetrize + Σ_ij P_ij = 1 normalisation + clamp.
    P = (P + P.T) / (2.0 * N)
    return torch.clamp(P, min=1e-12)


def _compute_p_matrix_torch_ref(X, perplexity=30.0, n_bisect=50):
    """Reference (slow) torch impl — kept for correctness checks only."""
    N = X.shape[0]
    device = X.device
    target = math.log(perplexity)

    dists_sq = torch.cdist(X, X, p=2).pow(2)
    diag_mask = torch.eye(N, device=device, dtype=torch.bool)
    off_diag = (~diag_mask).to(torch.float32)
    d_for_min = dists_sq.masked_fill(diag_mask, float('inf'))
    d_min = d_for_min.min(dim=1, keepdim=True).values
    d_centered = dists_sq - d_min

    lo = torch.zeros(N, device=device, dtype=torch.float32)
    hi = torch.full((N,), 1e10, device=device, dtype=torch.float32)
    beta = torch.ones(N, device=device, dtype=torch.float32)

    for _ in range(n_bisect):
        P_unnorm = torch.exp(-beta[:, None] * d_centered) * off_diag
        sum_P = P_unnorm.sum(dim=1, keepdim=True) + 1e-12
        Hsum = (P_unnorm * d_centered).sum(dim=1, keepdim=True) / sum_P
        H = torch.log(sum_P).squeeze(-1) + beta * Hsum.squeeze(-1)
        too_high = H > target
        lo = torch.where(too_high, beta, lo)
        hi = torch.where(too_high, hi, beta)
        beta = (lo + hi) * 0.5

    P_unnorm = torch.exp(-beta[:, None] * d_centered) * off_diag
    P = P_unnorm / (P_unnorm.sum(dim=1, keepdim=True) + 1e-12)
    P = (P + P.T) / (2.0 * N)
    return torch.clamp(P, min=1e-12)


def triton_tsne(X, n_iter=1000, lr=200.0, perplexity=30.0,
                early_exag_iters=None, ee_factor=12.0, seed=0):
    """flash-tsne: end-to-end Triton t-SNE with early exaggeration + momentum SGD.

    Schedule (matches sklearn defaults at n_iter=1000):
      - First `early_exag_iters` iters: P × 12 (early exaggeration), momentum 0.5
      - Remaining iters:                 P × 1, momentum 0.8

    `early_exag_iters` defaults to `min(250, n_iter // 3)` so short runs still
    leave room for the main phase to converge.

    Per-iter cost: 2 Triton kernel launches (qsum + grad). No N×N Q matrix.
    """
    N = X.shape[0]
    device = X.device
    if early_exag_iters is None:
        early_exag_iters = min(250, max(50, n_iter // 3))

    P = _compute_p_matrix(X, perplexity)
    # Precompute the exaggerated P once instead of doing P * ee_factor each
    # iter (was an N²-sized fresh allocation per step). At ee_factor=12 and
    # early_exag_iters=100, this saved 100 N² mul + 100 N² alloc — at huge
    # ~150 ms across the run.
    P_exag = P * ee_factor

    torch.manual_seed(seed)
    Y = torch.randn(N, 2, device=device, dtype=torch.float32) * 1e-4
    velocity = torch.zeros_like(Y)

    for i in range(n_iter):
        in_ee = i < early_exag_iters
        P_use = P_exag if in_ee else P
        momentum = 0.5 if in_ee else 0.8

        qsum = triton_tsne_qsum(Y)
        grad = triton_tsne_grad(Y, P_use, qsum)
        velocity = momentum * velocity - lr * grad
        Y = Y + velocity

    return Y


def triton_tsne_gradient_only(Y, P, n_iter=300, lr=200.0):
    """Run only the gradient loop (P precomputed). For benchmarking."""
    N = Y.shape[0]
    velocity = torch.zeros_like(Y)
    P_exag = P * 12.0

    for i in range(n_iter):
        P_use = P_exag if i < 100 else P
        qsum = triton_tsne_qsum(Y)
        grad = triton_tsne_grad(Y, P_use, qsum)
        momentum = 0.5 if i < 250 else 0.8
        velocity = momentum * velocity - lr * grad
        Y = Y + velocity

    return Y
