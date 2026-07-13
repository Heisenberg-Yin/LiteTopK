"""Shared core kernel for MultinomialNB and BernoulliNB flash implementations.

Both algorithms have the same fit shape:
    feature_count_[c, d] = Σ_{i: y_i = c} X_proc[i, d]

where X_proc is:
    - MultinomialNB: X (non-negative counts; we pre-cast to fp32 for the matmul)
    - BernoulliNB:   1 if X[i,d] > binarize else 0   (binarized to fp32)

Both reduce to: feature_count = one_hot.T @ X_proc, where one_hot[n, c] = 1[y_n=c].
We compute it with the same `tl.dot(one_hot[C×N], X_proc[N×D])` pattern that
gaussian_nb's fit pass-1 uses (atomic-free, BW-bound).

This shared kernel returns:
    feature_count: (C, D) fp32  — Σ X_proc by class
    class_count:   (C,) fp32   — count of samples per class

Predict is *not* shared — multinomial and bernoulli predict have different
algebra (multinomial = single GEMM; bernoulli = single GEMM + scalar correction).
"""

import torch
import triton
import triton.language as tl


_FIT_CONFIGS = [
    triton.Config({"BLOCK_D": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=3),
]


def _select_block_n(N, D, C):
    """Choose BLOCK_N (rows per block).

    Trade-off:
      - Larger BLOCK_N → fewer blocks → less reduction work in epilogue,
        but more registers/SMEM for the (BLOCK_N × BLOCK_D) X tile and
        (C_PAD × BLOCK_N) one_hot tile.
      - Smaller BLOCK_N → more blocks → more parallelism (good when N small).
    H200 has 132 SMs — we want at least ~528 blocks for good occupancy.
    """
    if C <= 16:
        bn = 256
    elif C <= 32:
        bn = 128
    else:
        bn = 64
    target_blocks = 4 * 132
    while bn > 64 and (N // bn) < target_blocks:
        bn //= 2
    return bn


@triton.autotune(configs=_FIT_CONFIGS, key=["N", "D", "C", "BLOCK_N", "C_PAD", "BINARIZE_MODE"])
@triton.jit
def _nb_count_kernel(
    X_ptr, Y_ptr, PARTIAL_ptr, COUNT_partial_ptr,
    N, D, C,
    stride_xn, stride_xd,
    stride_pb, stride_pc, stride_pd,
    BINARIZE_THRESH: tl.constexpr,  # ignored unless BINARIZE_MODE=1; kept constexpr=0 default
    BINARIZE_MODE: tl.constexpr,    # 0 = use X as-is (multinomial); 1 = binarize (bernoulli)
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    C_PAD: tl.constexpr,
):
    """Per-block partial sums via tensor-core matmul.

    For each (row-tile, feature-tile) program:
        one_hot ∈ R^{C_PAD × BLOCK_N}   one_hot[c,n] = (y_n == c)
        X_blk   ∈ R^{BLOCK_N × BLOCK_D} (binarized if BINARIZE_MODE)
        partial = one_hot @ X_blk → (C_PAD, BLOCK_D)
    One tensor-core GEMM per block. Output to (n_blocks, C_PAD, D); reduced
    by torch.sum(dim=0) on the host.

    Counts emitted only by pid_d == 0 programs.
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n_i32 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = offs_n_i32.to(tl.int64)
    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)
    offs_c = tl.arange(0, C_PAD)
    mask_n = offs_n_i32 < N
    mask_d = offs_d < D
    mask_c = offs_c < C

    x_ptrs = X_ptr + offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd
    x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    if BINARIZE_MODE == 1:
        x = (x > BINARIZE_THRESH).to(tl.float32)

    y = tl.load(Y_ptr + offs_n, mask=mask_n, other=-1)
    one_hot = (y[None, :] == offs_c[:, None]).to(tl.float32)
    one_hot = tl.where(mask_n[None, :] & mask_c[:, None], one_hot, 0.0)

    # one_hot is exact 0/1; TF32 input precision is lossless on it.
    # x is fp32 (or 0/1 if binarized) — TF32 truncation costs ≤2^-10 rel
    # error per multiply; summed over class membership this is negligible
    # vs Laplace smoothing α=1 in the log later.
    partial = tl.dot(one_hot, x, out_dtype=tl.float32, input_precision="tf32")

    p_ptrs = (PARTIAL_ptr + pid_n.to(tl.int64) * stride_pb
              + offs_c[:, None] * stride_pc
              + offs_d[None, :] * stride_pd)
    tl.store(p_ptrs, partial, mask=mask_c[:, None] & mask_d[None, :])

    if pid_d == 0:
        cnt = tl.sum(one_hot, axis=1)
        tl.store(COUNT_partial_ptr + pid_n * C_PAD + offs_c, cnt, mask=mask_c)


def _round_up_c_pad(C):
    """Triton's tl.dot requires the contraction (here, C_PAD) be ≥ 16 on Hopper.
    Round up to next power of 2 ≥ 16."""
    cp = 16
    while cp < C:
        cp *= 2
    return cp


# =============================================================================
# Sort-then-segment kernel (analog of kmeans `_centroid_update_chunk_kernel`).
#
# Motivation
# ----------
# The atomic-free one-hot GEMM kernel above materialises two intermediates
# that scale with C:
#   (a) a per-CTA one_hot tile of shape (C_PAD, BLOCK_N) fp32
#       — costs (C_PAD * BLOCK_N * 4) bytes of SMEM/registers,
#   (b) a (n_blocks, C_PAD, D) partial-sum tensor that gets reduced
#       by torch.sum(dim=0) on the host — costs (N/BLOCK_N * C_PAD * D * 4)
#       bytes of HBM.
# With C_PAD rounded up to the next power-of-2 (>=16), both blow up the
# moment C goes past a few hundred. Concrete numbers on H200:
#   N=1M,  D=128, C=1_000  -> partial_sum = 8.2 GB (autotune trips
#                              repeated 8 GB allocs and stalls)
#   N=1M,  D=128, C=10_000 -> partial_sum = 131 GB (OOM)
#
# Sort-then-segment removes both: sort y once, then each CTA owns BLOCK_N
# *contiguous* sorted rows, sums them per (locally distinct) class, and
# issues ONE atomic_add per (class, d_tile) — no per-CTA one-hot tile,
# no partial_sum reduction tensor. Same algorithm as kmeans' centroid
# update sorted path, only the output shape changes from (B,K,D) to (C,D).
# =============================================================================


@triton.jit
def _nb_count_sorted_kernel(
    X_ptr,                # *T   [N, D] — original X, unsorted
    SORTED_Y_ptr,         # *i32 [N]    — y after torch.sort
    SORT_IDX_ptr,         # *i32 [N]    — gather indices from torch.sort
    SUM_ptr,              # *f32 [C, D] — output: per-class feature sum
    COUNT_ptr,            # *f32 [C]    — output: per-class sample count
    N, D, C,
    stride_xn, stride_xd,
    stride_sumc, stride_sumd,
    BINARIZE_THRESH: tl.constexpr,
    BINARIZE_MODE: tl.constexpr,   # 0 = multinomial, 1 = bernoulli
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Each program processes BLOCK_N consecutive sorted rows × BLOCK_D dims.

    Because the rows are sorted by class id, identical ids appear in
    contiguous runs. The kernel scans [first_id, last_id] for this chunk
    (typically 1–few iterations when N >> C * BLOCK_N) and performs ONE
    atomic_add to SUM per (class, d_tile) — independent of how many rows
    the class owns inside the chunk.
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    chunk_start = pid_n * BLOCK_N
    d_start = pid_d * BLOCK_D
    if chunk_start >= N:
        return

    offs_n_i32 = chunk_start + tl.arange(0, BLOCK_N)
    offs_n = offs_n_i32.to(tl.int64)
    offs_d = (d_start + tl.arange(0, BLOCK_D)).to(tl.int64)
    mask_n = offs_n_i32 < N
    mask_d = offs_d < D

    # Sorted class ids + gather indices for this chunk
    sorted_y = tl.load(SORTED_Y_ptr + offs_n, mask=mask_n, other=-1)
    sort_idx = tl.load(SORT_IDX_ptr + offs_n, mask=mask_n, other=0).to(tl.int64)

    # Range of class ids present in this chunk (sorted -> monotone)
    first_id = tl.load(SORTED_Y_ptr + chunk_start)
    last_pos = tl.minimum(chunk_start + BLOCK_N, N) - 1
    last_id = tl.load(SORTED_Y_ptr + last_pos)

    # Single gather load for the whole chunk × BLOCK_D
    x_row_ptrs = X_ptr + sort_idx[:, None] * stride_xn + offs_d[None, :] * stride_xd
    x = tl.load(x_row_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    if BINARIZE_MODE == 1:
        x = (x > BINARIZE_THRESH).to(tl.float32)

    # Walk distinct class ids in [first_id, last_id]. Typical occupancy
    # is 1–few iters: a uniform-random sorted chunk of BLOCK_N=256 rows
    # spans ~BLOCK_N / (N/C) distinct ids (~2-3 at N=1M, C=10K).
    for cid in range(first_id, last_id + 1):
        cluster_mask = (sorted_y == cid) & mask_n
        # masked sum along the chunk row axis -> (BLOCK_D,)
        sum_feats = tl.sum(tl.where(cluster_mask[:, None], x, 0.0), axis=0)
        dest = SUM_ptr + cid.to(tl.int64) * stride_sumc + offs_d * stride_sumd
        tl.atomic_add(dest, sum_feats, mask=mask_d)
        if pid_d == 0:
            cluster_size = tl.sum(cluster_mask.to(tl.int32))
            tl.atomic_add(COUNT_ptr + cid, cluster_size.to(tl.float32))


def _pick_sorted_block(N: int, D: int, C: int):
    """Hand-tuned BLOCK_N/BLOCK_D + num_warps for the sorted kernel.

    Autotune is unsafe for kernels that issue ``tl.atomic_add`` (each profile
    run accumulates into the output instead of overwriting), so we use a
    static heuristic here. The pattern is the same one kmeans'
    ``_centroid_update_chunk_kernel`` uses (fixed BLOCK_N passed at launch).

    BLOCK_N controls (a) HBM-load coalescing on the X gather, (b) the
    inner ``for cid in range(first_id, last_id + 1)`` loop bound. With
    sorted uniform-random data and N >> C, ``last_id - first_id`` is
    ~``BLOCK_N * C / N``, so larger BLOCK_N is fine until the gather
    register pressure dominates. 128–256 is the sweet spot on H200.
    """
    if D >= 256:
        block_n, block_d, nw = 128, 128, 4
    elif D >= 128:
        block_n, block_d, nw = 256, 128, 4
    elif D >= 64:
        block_n, block_d, nw = 256, 64, 4
    else:
        block_n, block_d, nw = 512, 32, 4
    block_d = min(block_d, max(triton.next_power_of_2(D), 16))
    return block_n, block_d, nw


def _nb_count_features_sorted(X: torch.Tensor, y: torch.Tensor,
                              n_classes: int, binarize: float = None):
    """Sort-then-segment variant for the large-C regime.

    Steps:
      1. ``sorted_y, sort_idx = torch.sort(y)`` — O(N log N) device sort
      2. Launch ``_nb_count_sorted_kernel_autotuned`` over
         ``(ceil(N/BLOCK_N), ceil(D/BLOCK_D))``
      3. Each CTA gathers BLOCK_N contiguous sorted rows × BLOCK_D dims,
         does one ``tl.atomic_add`` per (locally present class) into
         the (C, D) feature_count buffer.

    Memory footprint is O(N + C*D) — no per-block partial buffer.
    """
    N, D = X.shape
    C = n_classes
    device = X.device

    sorted_y, sort_idx = torch.sort(y)
    sorted_y = sorted_y.to(torch.int32).contiguous()
    sort_idx = sort_idx.to(torch.int32).contiguous()

    feature_count = torch.zeros((C, D), device=device, dtype=torch.float32)
    class_count = torch.zeros((C,), device=device, dtype=torch.float32)

    binarize_mode = 0 if binarize is None else 1
    binarize_thresh = float(binarize) if binarize is not None else 0.0

    BLOCK_N, BLOCK_D, nw = _pick_sorted_block(N, D, C)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))
    _nb_count_sorted_kernel[grid](
        X, sorted_y, sort_idx, feature_count, class_count,
        N, D, C,
        X.stride(0), X.stride(1),
        feature_count.stride(0), feature_count.stride(1),
        BINARIZE_THRESH=binarize_thresh,
        BINARIZE_MODE=binarize_mode,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=nw,
    )
    return feature_count, class_count


def _should_use_sorted(N: int, D: int, C: int) -> bool:
    """Dispatch heuristic.

    The one-hot GEMM kernel costs O(N/BN * C_PAD * D * 4) bytes for the
    partial_sum tensor (C_PAD rounded up to next pow2). Sorted-segment
    costs O(C*D + N) and is always memory-frugal.

    Measured H200 crossover (`benchmarks/results/micro_nb_fit_scaling.md`):
      C ≤ 32   one-hot wins by ~15-20% (tensor-core GEMM dominates;
               partial buffer fits in a few hundred MB)
      C ≥ 128  sorted wins by 1.5-3× (one-hot's partial buffer crosses
               the 1 GB cliff; tensor-core advantage gone)
      C ≥ 1K   one-hot's partial buffer exceeds free HBM — sorted is
               the only path that runs at all

    Rule: use sorted whenever C > 64 OR whenever the one-hot partial
    buffer would exceed 2 GB (the autotune-stall threshold).
    """
    # Estimate the partial buffer the one-hot path would allocate.
    if C <= 16:
        bn = 256
    elif C <= 32:
        bn = 128
    else:
        bn = 64
    target_blocks = 4 * 132
    while bn > 64 and (N // bn) < target_blocks:
        bn //= 2
    n_blocks = (N + bn - 1) // bn
    c_pad = _round_up_c_pad(C)
    partial_bytes = n_blocks * c_pad * D * 4
    return C > 64 or partial_bytes > 2 * 1024 * 1024 * 1024


def nb_count_features(X: torch.Tensor, y: torch.Tensor, n_classes: int,
                      binarize: float = None, *, force_path: str = None):
    """Compute per-class feature counts (and class counts).

    Args:
        X: (N, D) tensor on cuda. fp32 / fp16 / bf16 / int — internally upcast
           to fp32 for the matmul. For multinomial, must be ≥ 0 (we don't check).
        y: (N,) class labels in [0, n_classes).
        n_classes: C.
        binarize: if not None, cast to (X > binarize).float() inside the kernel.
                  Use None for multinomial; 0.0 (sklearn default) for bernoulli.
        force_path: optional override for testing / benching:
                  - ``"onehot"`` — original atomic-free one_hot GEMM kernel
                    (fast for small C, OOMs / autotune-stalls for large C);
                  - ``"sorted"`` — sort-then-segment kernel (memory-frugal,
                    no partial_sum tensor, scales to arbitrary C);
                  - ``None`` (default) — automatic dispatch on (N, D, C).

    Returns:
        feature_count: (C, D) fp32 — Σ_{i: y_i=c} X_proc[i, d]
        class_count:   (C,)  fp32 — Σ_i [y_i = c]
    """
    assert X.is_cuda and X.ndim == 2 and y.ndim == 1
    N, D = X.shape
    C = n_classes
    device = X.device

    if not X.is_contiguous():
        X = X.contiguous()
    if y.dtype != torch.int64:
        y = y.to(torch.int64)
    y = y.contiguous()

    if force_path == "sorted" or (force_path is None and _should_use_sorted(N, D, C)):
        return _nb_count_features_sorted(X, y, C, binarize=binarize)

    BLOCK_N = _select_block_n(N, D, C)
    n_blocks = triton.cdiv(N, BLOCK_N)
    C_PAD = _round_up_c_pad(C)

    partial_sum = torch.empty((n_blocks, C_PAD, D), device=device, dtype=torch.float32)
    count_partial = torch.zeros((n_blocks, C_PAD), device=device, dtype=torch.float32)

    binarize_mode = 0 if binarize is None else 1
    binarize_thresh = float(binarize) if binarize is not None else 0.0

    grid = lambda META: (n_blocks, triton.cdiv(D, META["BLOCK_D"]))
    _nb_count_kernel[grid](
        X, y, partial_sum, count_partial,
        N, D, C,
        X.stride(0), X.stride(1),
        partial_sum.stride(0), partial_sum.stride(1), partial_sum.stride(2),
        BINARIZE_THRESH=binarize_thresh,
        BINARIZE_MODE=binarize_mode,
        BLOCK_N=BLOCK_N, C_PAD=C_PAD,
    )
    feature_count = partial_sum.sum(dim=0)[:C].contiguous()
    class_count = count_partial.sum(dim=0)[:C].contiguous()
    return feature_count, class_count
