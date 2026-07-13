"""flash-rf Triton kernels: best-split, scatter / ranged histograms,
fused split-counts, hist subtraction, fused partition.

The python wrappers that drive each kernel live alongside the kernel
itself; the Tree-build orchestration that calls them lives in
``flashlib.primitives.random_forest.impl``.
"""
import math
import numpy as np
import torch
import triton
import triton.language as tl


# =============================================================================
# Triton best-split kernel: per (node, feat) program scans bins in registers,
# avoiding the (n_active, D, n_bins, K) cumsum temp tensor (~350MB at deep levels).
# Outputs (gain, bin) per (node, feat); caller picks argmax-feat per node.
# =============================================================================

@triton.jit
def _rf_best_split_per_feat_kernel(
    HIST_ptr,           # (n_active, D, N_BINS, K) float32
    PARENT_STAT_ptr,    # (n_active, 2) float32 — col0=n_total, col1=parent_gini
    FEAT_MASK_ptr,      # (n_active, D) uint8 (or dummy if HAS_MASK=False)
    HAS_MASK: tl.constexpr,
    OUT_GAIN_ptr,       # (n_active, D) float32
    OUT_BIN_ptr,        # (n_active, D) int32
    D,
    N_BINS: tl.constexpr,    # power-of-2; treated as parallel tile
    K: tl.constexpr,
    K_PAD: tl.constexpr,
):
    """One program per (node, feature). All N_BINS processed in parallel via
    `tl.cumsum` (no serial bin loop). The inner work is a single (N_BINS,K_PAD)
    tile loaded, scanned, and reduced — Triton parallelizes lanes across warps.
    """
    pid_node = tl.program_id(0)
    pid_feat = tl.program_id(1)

    if HAS_MASK:
        m = tl.load(FEAT_MASK_ptr + pid_node * D + pid_feat)
        if m == 0:
            tl.store(OUT_GAIN_ptr + pid_node * D + pid_feat, -1.0)
            tl.store(OUT_BIN_ptr + pid_node * D + pid_feat, 0)
            return

    n_total = tl.load(PARENT_STAT_ptr + pid_node * 2 + 0)
    parent_gini = tl.load(PARENT_STAT_ptr + pid_node * 2 + 1)

    if n_total < 2.0:
        tl.store(OUT_GAIN_ptr + pid_node * D + pid_feat, -1.0)
        tl.store(OUT_BIN_ptr + pid_node * D + pid_feat, 0)
        return

    base = (HIST_ptr
            + pid_node.to(tl.int64) * D * N_BINS * K
            + pid_feat.to(tl.int64) * N_BINS * K)

    bin_offs = tl.arange(0, N_BINS)              # (N_BINS,)
    k_offs = tl.arange(0, K_PAD)                 # (K_PAD,)
    k_mask = k_offs < K
    addr = base + bin_offs[:, None] * K + k_offs[None, :]
    counts = tl.load(addr, mask=k_mask[None, :], other=0.0)  # (N_BINS, K_PAD)

    cum_left = tl.cumsum(counts, axis=0)         # (N_BINS, K_PAD)
    total_vec = tl.sum(counts, axis=0)           # (K_PAD,) — feature-local total
    cum_right = total_vec[None, :] - cum_left    # (N_BINS, K_PAD)

    n_left_vec = tl.sum(cum_left, axis=1)        # (N_BINS,)
    n_right_vec = n_total - n_left_vec
    sq_l = tl.sum(cum_left * cum_left, axis=1)
    sq_r = tl.sum(cum_right * cum_right, axis=1)

    inv_ntot = 1.0 / n_total
    base_const = parent_gini - 1.0
    gain_vec = base_const + (sq_l / tl.maximum(n_left_vec, 1e-30)
                             + sq_r / tl.maximum(n_right_vec, 1e-30)) * inv_ntot
    valid = (n_left_vec > 0.0) & (n_right_vec > 0.0)
    # Last bin always invalid (n_right=0)
    is_last = bin_offs == (N_BINS - 1)
    valid = valid & (~is_last)
    gain_vec = tl.where(valid, gain_vec, -1.0)

    best_gain = tl.max(gain_vec, axis=0)
    best_bin = tl.argmax(gain_vec, axis=0).to(tl.int32)

    tl.store(OUT_GAIN_ptr + pid_node * D + pid_feat, best_gain)
    tl.store(OUT_BIN_ptr + pid_node * D + pid_feat, best_bin)


def _find_best_splits_subfeat(hist, n_classes, feat_idx):
    """Sub-feature wrapper: best_split over (n_active, n_feat_per_split, n_bins, K)
    hist, then maps best_k_sub back to actual feat index via feat_idx[node, k_sub].

    Also returns best_subfeat (the k_sub index, not the actual feat) — needed
    to compute per-node left/right child sample counts from this node's hist
    column without a separate bincount over all B·N samples.
    """
    best_subfeat, best_bin, best_gain, leaf_class = _find_best_splits_triton(
        hist, n_classes, feat_mask=None)
    n_active = hist.shape[0]
    rows = torch.arange(n_active, device=hist.device)
    best_feat_actual = feat_idx[rows, best_subfeat.to(torch.int64)].to(torch.int32)
    return best_feat_actual, best_bin, best_gain, leaf_class, best_subfeat


def _find_best_splits_triton(hist, n_classes, feat_mask=None):
    """Triton-backed best split. Replaces _find_best_splits_torch's huge cumsum."""
    n_active, D, n_bins, K = hist.shape
    device = hist.device
    K_PAD = max(2, 1 << (K - 1).bit_length())

    # Pre-compute per-node totals + parent gini in torch (cheap: O(n_active * K))
    total_full = hist[:, 0, :, :].sum(dim=1)                     # (n_active, K) fp32
    n_total = total_full.sum(dim=1)                              # (n_active,)
    n_total_safe = n_total.clamp(min=1.0)
    p = total_full / n_total_safe[:, None]
    parent_gini = 1.0 - (p * p).sum(dim=1)                       # (n_active,)
    leaf_class = total_full.argmax(dim=1).to(torch.int32)
    parent_stat = torch.stack([n_total, parent_gini], dim=1).contiguous()  # (n_active, 2)

    out_gain = torch.full((n_active, D), -1.0, dtype=torch.float32, device=device)
    out_bin = torch.zeros((n_active, D), dtype=torch.int32, device=device)
    if feat_mask is not None:
        fm = feat_mask.to(torch.uint8).contiguous()
        has_mask = True
    else:
        fm = out_gain  # dummy; HAS_MASK=False so unused
        has_mask = False

    # Pick num_warps based on N_BINS · K_PAD tile size (heuristic)
    tile_lanes = n_bins * K_PAD
    num_warps = 8 if tile_lanes >= 1024 else (4 if tile_lanes >= 256 else 2)

    grid = (n_active, D)
    _rf_best_split_per_feat_kernel[grid](
        hist.contiguous(), parent_stat,
        fm, has_mask, out_gain, out_bin,
        D=D, N_BINS=n_bins, K=K, K_PAD=K_PAD,
        num_warps=num_warps,
    )
    best_idx = out_gain.argmax(dim=1)
    rows = torch.arange(n_active, device=device)
    best_feat = best_idx.to(torch.int32)
    best_gain = out_gain[rows, best_idx]
    best_bin = out_bin[rows, best_idx]
    return best_feat, best_bin, best_gain, leaf_class


# =============================================================================
# Histogram kernel: per-(node, feature) class-count histogram
# Each (node_id, feature_id) gets ONE program; iterates over its samples and
# atomically accumulates label counts into hist[node, feat, bin, class].
# =============================================================================

@triton.jit
def _scatter_histogram_kernel(
    X_BIN_ptr,          # (N, D) uint8
    Y_ptr,              # (N,) int32 class labels
    SAMPLE_NODE_ptr,    # (N,) int32 — compact node id per sample, -1 = inactive
    HIST_ptr,           # (n_active, D, N_BINS, K) float32 atomic accumulator
    N, D,
    N_BINS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sample-tile × feature scatter histogram.

    Total work: O(N · D) — independent of n_active. Replaces the previous
    O(n_active · D · N) gather kernel that re-scanned all N for every active
    node and exploded at deep tree levels.
    """
    pid_n = tl.program_id(0)   # sample tile id
    pid_d = tl.program_id(1)   # feature id

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    node = tl.load(SAMPLE_NODE_ptr + n_offs, mask=n_mask, other=-1)
    valid = n_mask & (node >= 0)

    addr_x = n_offs.to(tl.int64) * D + pid_d
    bins = tl.load(X_BIN_ptr + addr_x, mask=valid, other=0).to(tl.int32)
    classes = tl.load(Y_ptr + n_offs, mask=valid, other=0).to(tl.int32)

    addr_h = (node.to(tl.int64) * (D * N_BINS * K)
              + pid_d * (N_BINS * K)
              + bins.to(tl.int64) * K
              + classes.to(tl.int64))
    tl.atomic_add(HIST_ptr + addr_h, 1.0, mask=valid)


def _build_node_histograms(X_bin, y, sample_node, n_active, n_bins, n_classes):
    """Build per-(node, feature) histograms.

    Uses scatter design: each program processes a sample tile × one feature
    and atomic-adds into hist[sample_node[i], feat, bin, class].
    Total work O(N · D), independent of n_active.

    Returns (n_active, D, n_bins, n_classes) float32.
    """
    N, D = X_bin.shape
    hist = torch.zeros(n_active, D, n_bins, n_classes,
                        device=X_bin.device, dtype=torch.float32)
    BLOCK_N = 1024
    grid = ((N + BLOCK_N - 1) // BLOCK_N, D)
    _scatter_histogram_kernel[grid](
        X_bin, y, sample_node, hist,
        N, D,
        N_BINS=n_bins, K=n_classes,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return hist


# =============================================================================
# Per-node sub-feature histogram kernel: each program tile-of-samples × ONE
# k_sub. Per sample, looks up the actual feature index = FEAT_IDX[node, k_sub]
# rather than using pid as the feature directly. Hist is (n_active,
# n_feat_per_split, n_bins, K) — much smaller than (n_active, D, ...) when
# max_features < D.
# =============================================================================

@triton.jit
def _subfeat_scatter_hist_kernel(
    X_BIN_ptr,          # (N, D) uint8
    Y_ptr,              # (N,) int32
    SAMPLE_NODE_ptr,    # (N,) int32 — compact node id, -1 = inactive
    FEAT_IDX_ptr,       # (n_active, n_feat_per_split) int32 — actual feat indices
    HIST_ptr,           # (n_active, n_feat_per_split, N_BINS, K) float32
    N, D, n_feat_per_split,
    N_BINS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)  # which sub-feature (0..n_feat_per_split-1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    node = tl.load(SAMPLE_NODE_ptr + n_offs, mask=n_mask, other=-1)
    valid = n_mask & (node >= 0)

    # Look up actual feat index: FEAT_IDX[node, pid_k]
    feat = tl.load(FEAT_IDX_ptr + node.to(tl.int64) * n_feat_per_split + pid_k,
                   mask=valid, other=0).to(tl.int64)
    addr_x = n_offs.to(tl.int64) * D + feat
    bins = tl.load(X_BIN_ptr + addr_x, mask=valid, other=0).to(tl.int32)
    classes = tl.load(Y_ptr + n_offs, mask=valid, other=0).to(tl.int32)

    # Hist layout: (n_active, n_feat_per_split, N_BINS, K)
    addr_h = (node.to(tl.int64) * (n_feat_per_split * N_BINS * K)
              + pid_k * (N_BINS * K)
              + bins.to(tl.int64) * K
              + classes.to(tl.int64))
    tl.atomic_add(HIST_ptr + addr_h, 1.0, mask=valid)


def _build_node_histograms_subfeat(X_bin, y, sample_node, feat_idx,
                                    n_active, n_bins, n_classes):
    """Sub-feature variant: hist is (n_active, n_feat_per_split, n_bins, K).
    feat_idx[node, k] gives the actual feature index in [0, D)."""
    N, D = X_bin.shape
    n_feat_per_split = feat_idx.shape[1]
    hist = torch.zeros(n_active, n_feat_per_split, n_bins, n_classes,
                       device=X_bin.device, dtype=torch.float32)
    BLOCK_N = 1024
    grid = ((N + BLOCK_N - 1) // BLOCK_N, n_feat_per_split)
    _subfeat_scatter_hist_kernel[grid](
        X_bin, y, sample_node, feat_idx, hist,
        N, D, n_feat_per_split,
        N_BINS=n_bins, K=n_classes,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return hist


# =============================================================================
# Ranged sub-feature hist kernel: CTA-per-(node, k_sub). Processes all of
# its node's contiguous samples in row_perm. Best when n_active is small and
# nodes are big (shallow levels of large trees) — each CTA writes to a tight
# (n_bins · K) hist region with no inter-CTA atomic conflicts.
# =============================================================================

@triton.jit
def _ranged_subfeat_hist_kernel(
    ROW_PERM_ptr,       # (total_active,) int32 — sample idx in original Xb
    NODE_OFFSETS_ptr,   # (n_active+1,) int32 — start of each node in row_perm
    X_BIN_ptr,          # (orig_N, D) uint8
    Y_ptr,              # (orig_N,) int32
    FEAT_IDX_ptr,       # (n_active, n_feat_per_split) int32
    HIST_ptr,           # (n_active, n_feat_per_split, N_BINS, K) float32
    D, n_feat_per_split,
    N_BINS: tl.constexpr,
    K: tl.constexpr,
    NBK: tl.constexpr,         # = N_BINS * K — real (bin, class) cell count
    NBK_PADDED: tl.constexpr,  # next pow-of-2 ≥ NBK + 1 (for sentinel)
    BLOCK_S: tl.constexpr,
):
    """Per-CTA shared-memory histogram: each (node, k_sub) CTA accumulates a
    private (N_BINS · K) tile via `tl.histogram` (parallel reduce, no atomics
    inside the loop), then performs a single atomic flush to global HIST at
    the end. Reduces atomic count by ~100× vs the per-sample atomic_add design.
    """
    pid_node = tl.program_id(0)
    pid_k = tl.program_id(1)

    start = tl.load(NODE_OFFSETS_ptr + pid_node)
    end = tl.load(NODE_OFFSETS_ptr + pid_node + 1)
    if start >= end:
        return

    feat = tl.load(FEAT_IDX_ptr + pid_node * n_feat_per_split + pid_k).to(tl.int64)

    # Per-CTA accumulator. Padded power-of-2 size; last bin (NBK) is sentinel
    # for masked-out samples and is not flushed.
    running = tl.zeros((NBK_PADDED,), dtype=tl.int32)

    for s in range(start, end, BLOCK_S):
        s_offs = s + tl.arange(0, BLOCK_S)
        mask = s_offs < end
        row = tl.load(ROW_PERM_ptr + s_offs, mask=mask, other=0).to(tl.int64)
        addr_x = row * D + feat
        bins = tl.load(X_BIN_ptr + addr_x, mask=mask, other=0).to(tl.int32)
        classes = tl.load(Y_ptr + row, mask=mask, other=0).to(tl.int32)
        # Encode (bin, class) → single tag in [0, NBK). Invalid samples get
        # tag = NBK (the sentinel real bin we mask off at flush).
        tag = bins * K + classes
        tag = tl.where(mask, tag, NBK)
        running = running + tl.histogram(tag, NBK_PADDED)

    bin_offs = tl.arange(0, NBK_PADDED)
    real = bin_offs < NBK
    base = (pid_node.to(tl.int64) * (n_feat_per_split * NBK)
            + pid_k.to(tl.int64) * NBK)
    tl.atomic_add(HIST_ptr + base + bin_offs, running.to(tl.float32), mask=real)


# =============================================================================
# Fused split-counts kernel: replaces 5 torch ops (indexed gather of prev_hist
# at parent's split column, sum over classes, cumsum over bins, indexed gather
# at split bin, total = cum[-1]) with a single Triton kernel. Avoids the big
# (n_internal_prev, n_bins, K) intermediate copy.
# =============================================================================

@triton.jit
def _rf_split_counts_kernel(
    PREV_HIST_ptr,       # (n_active_prev, n_feat, n_bins, K) float32
    PINTERNALS_ptr,      # (n_internal,) int64 — prev_internal_global_idx
    PSUBFEAT_ptr,        # (n_internal,) int64 — parent's best_subfeat
    PBIN_ptr,            # (n_internal,) int32 — parent's best_bin
    OUT_LEFT_ptr,        # (n_internal,) int64 — sum of cells with bin ≤ pbin
    OUT_TOTAL_ptr,       # (n_internal,) int64 — sum of all cells
    n_feat,
    N_BINS: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    pi = tl.load(PINTERNALS_ptr + pid)
    sf = tl.load(PSUBFEAT_ptr + pid)
    pbin = tl.load(PBIN_ptr + pid)

    base = (PREV_HIST_ptr
            + pi * (n_feat * N_BINS * K)
            + sf * (N_BINS * K))

    bin_offs = tl.arange(0, N_BINS)
    k_offs = tl.arange(0, K_PAD)
    k_mask = k_offs < K

    addr = base + bin_offs[:, None] * K + k_offs[None, :]
    cells = tl.load(addr, mask=k_mask[None, :], other=0.0)  # (N_BINS, K_PAD)

    bin_le = bin_offs <= pbin
    left = tl.sum(tl.where(bin_le[:, None], cells, 0.0))
    total = tl.sum(cells)

    tl.store(OUT_LEFT_ptr + pid, left.to(tl.int64))
    tl.store(OUT_TOTAL_ptr + pid, total.to(tl.int64))


def _split_counts_fused(prev_hist, prev_internal_idx, prev_subfeat, prev_bin):
    """Per-parent left/total sample counts, derived from prev_hist's split
    column. Returns (left_counts, total_counts) int64."""
    n_internal = prev_internal_idx.numel()
    n_active_prev, n_feat, n_bins, K = prev_hist.shape
    device = prev_hist.device
    K_PAD = max(2, 1 << (K - 1).bit_length())
    out_left = torch.empty(n_internal, dtype=torch.int64, device=device)
    out_total = torch.empty(n_internal, dtype=torch.int64, device=device)
    grid = (n_internal,)
    _rf_split_counts_kernel[grid](
        prev_hist.contiguous(),
        prev_internal_idx.contiguous(),
        prev_subfeat.contiguous(),
        prev_bin.contiguous(),
        out_left, out_total,
        n_feat,
        N_BINS=n_bins, K=K, K_PAD=K_PAD,
        num_warps=1,
    )
    return out_left, out_total


# =============================================================================
# Fused hist subtraction kernel: replaces three torch indexed ops
# (prev_hist[idx] gather, half_hist[smaller] gather, half_hist[bigger] write)
# plus the elementwise subtract with a single Triton kernel.
# =============================================================================

@triton.jit
def _rf_hist_sub_kernel(
    PREV_HIST_ptr,         # (n_active_prev, hist_stride) float32
    HALF_HIST_ptr,         # (n_active_curr, hist_stride) float32 — RW
    PREV_INTERNAL_IDX_ptr, # (n_internal,) int64
    SMALLER_GLOBAL_ptr,    # (n_internal,) int64
    BIGGER_GLOBAL_ptr,     # (n_internal,) int64
    HIST_STRIDE,
    BLOCK_HIST: tl.constexpr,
):
    pid_ip = tl.program_id(0)
    pid_b = tl.program_id(1)

    parent_idx = tl.load(PREV_INTERNAL_IDX_ptr + pid_ip)
    smaller_idx = tl.load(SMALLER_GLOBAL_ptr + pid_ip)
    bigger_idx = tl.load(BIGGER_GLOBAL_ptr + pid_ip)

    offs = pid_b * BLOCK_HIST + tl.arange(0, BLOCK_HIST)
    mask = offs < HIST_STRIDE

    parent = tl.load(PREV_HIST_ptr + parent_idx * HIST_STRIDE + offs,
                     mask=mask, other=0.0)
    smaller = tl.load(HALF_HIST_ptr + smaller_idx * HIST_STRIDE + offs,
                      mask=mask, other=0.0)
    bigger = parent - smaller
    tl.store(HALF_HIST_ptr + bigger_idx * HIST_STRIDE + offs, bigger, mask=mask)


def _hist_subtract_fused(prev_hist, half_hist, prev_internal_idx,
                          smaller_global, bigger_global):
    """Single-kernel fused hist subtraction. Avoids 3 torch indexed ops + a
    separate elementwise subtract (~500 ms / 80 calls at xlarge)."""
    n_internal = prev_internal_idx.numel()
    if n_internal == 0:
        return
    # Flatten per-node hist
    hist_stride = prev_hist[0].numel()
    BLOCK_HIST = 256
    grid = (n_internal, (hist_stride + BLOCK_HIST - 1) // BLOCK_HIST)
    _rf_hist_sub_kernel[grid](
        prev_hist.view(prev_hist.shape[0], -1),
        half_hist.view(half_hist.shape[0], -1),
        prev_internal_idx.contiguous(),
        smaller_global.contiguous(),
        bigger_global.contiguous(),
        hist_stride,
        BLOCK_HIST=BLOCK_HIST,
        num_warps=4,
    )


def _build_node_histograms_subfeat_hybrid(X_bin, y, sample_node, feat_idx,
                                            n_active, n_bins, n_classes):
    """Subfeat hist: ranged when n_active is small + big nodes, else scatter.

    Ranged design: grid (n_active, n_feat_per_split), one CTA per (node, k_sub).
    At shallow levels of large trees (e.g., xlarge depth 0 with B=42 trees,
    n_active=42, ~1M samples per node) the scatter kernel suffers heavy
    atomic-add contention because all 42M samples write to a small shared hist
    region. The ranged kernel partitions samples per-node first, so each CTA's
    atomics hit only its own (n_bins · K) cells — no inter-CTA conflicts.
    """
    N, D = X_bin.shape
    n_feat_per_split = feat_idx.shape[1]
    device = X_bin.device
    avg = N // max(n_active, 1)
    use_ranged = (n_active <= 4096) and (avg >= 64) and (N >= 200_000)
    if use_ranged:
        sn_safe = torch.where(sample_node >= 0, sample_node,
                               torch.full_like(sample_node, n_active))
        # sort() returns (sorted, perm) — we use both. Avoids the separate
        # bincount over B·N samples (~150 ms / 4 ms·call at xlarge): node
        # offsets come from searchsorted on the already-sorted sample_node.
        sorted_sn, row_perm = sn_safe.sort(stable=True)
        row_perm = row_perm.to(torch.int32).contiguous()
        node_offsets = torch.searchsorted(
            sorted_sn,
            torch.arange(n_active + 1, device=device, dtype=sorted_sn.dtype)
        ).to(torch.int32)
        hist = torch.zeros(n_active, n_feat_per_split, n_bins, n_classes,
                           device=device, dtype=torch.float32)
        BLOCK_S = 512
        NBK = n_bins * n_classes
        NBK_PADDED = max(2, 1 << ((NBK + 1 - 1).bit_length()))  # next pow-2 ≥ NBK+1
        grid = (n_active, n_feat_per_split)
        _ranged_subfeat_hist_kernel[grid](
            row_perm, node_offsets, X_bin, y, feat_idx, hist,
            D, n_feat_per_split,
            N_BINS=n_bins, K=n_classes,
            NBK=NBK, NBK_PADDED=NBK_PADDED,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )
        return hist
    return _build_node_histograms_subfeat(X_bin, y, sample_node, feat_idx,
                                           n_active, n_bins, n_classes)


# =============================================================================
# Ranged histogram kernel (cuML-style): grid (n_active, D). Each CTA processes
# ITS node's contiguous samples in row_perm. All atomics from one CTA hit a
# single (n_bins · K) hist region — no inter-CTA cache-line conflicts.
# =============================================================================

@triton.jit
def _ranged_histogram_kernel(
    ROW_PERM_ptr,       # (total_active,) int32 — original sample indices, sorted by node
    NODE_OFFSETS_ptr,   # (n_active+1,) int32 — start of each node in row_perm
    X_BIN_ptr,          # (orig_N, D) uint8 — original layout
    Y_ptr,              # (orig_N,) int32
    HIST_ptr,           # (n_active, D, N_BINS, K) float32
    D,
    N_BINS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_node = tl.program_id(0)
    pid_feat = tl.program_id(1)

    start = tl.load(NODE_OFFSETS_ptr + pid_node)
    end = tl.load(NODE_OFFSETS_ptr + pid_node + 1)
    if start >= end:
        return

    hist_base = (HIST_ptr
                 + pid_node.to(tl.int64) * (D * N_BINS * K)
                 + pid_feat.to(tl.int64) * (N_BINS * K))

    for s in range(start, end, BLOCK_S):
        s_offs = s + tl.arange(0, BLOCK_S)
        mask = s_offs < end
        row = tl.load(ROW_PERM_ptr + s_offs, mask=mask, other=0).to(tl.int64)
        addr_x = row * D + pid_feat
        bins = tl.load(X_BIN_ptr + addr_x, mask=mask, other=0).to(tl.int32)
        classes = tl.load(Y_ptr + row, mask=mask, other=0).to(tl.int32)
        addr_h = bins.to(tl.int64) * K + classes.to(tl.int64)
        tl.atomic_add(hist_base + addr_h, 1.0, mask=mask)


def _build_node_histograms_ranged(X_bin, y, row_perm, node_offsets,
                                   n_active, n_bins, n_classes):
    """Ranged-hist variant: samples for each active node are contiguous in
    row_perm[node_offsets[i] : node_offsets[i+1]]. One CTA per (node, feat).
    All hist writes from one CTA hit a tight n_bins·K region — better L2 reuse
    than the scatter design and no inter-CTA atomic conflicts.
    """
    _, D = X_bin.shape
    device = X_bin.device
    hist = torch.zeros(n_active, D, n_bins, n_classes,
                       device=device, dtype=torch.float32)
    BLOCK_S = 512
    grid = (n_active, D)
    _ranged_histogram_kernel[grid](
        row_perm, node_offsets, X_bin, y, hist,
        D,
        N_BINS=n_bins, K=n_classes,
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )
    return hist


# =============================================================================
# Fused partition kernel: replaces ~14 torch ops per level (gather, where,
# clamp, indexing, view) with a single Triton kernel. Saves ~13 launches/level
# at small (~13ms saved per fit @ 200 levels with launch overhead 5μs each).
# =============================================================================

@triton.jit
def _rf_partition_kernel(
    SAMPLE_NODE_IN_ptr,    # (B*N,) int32 — current per-tree compact ids (sample's node)
    BEST_FEAT_ptr,         # (n_active_total,) int32 — best feature per active node
    BEST_BIN_ptr,          # (n_active_total,) int32
    IS_LEAF_ptr,           # (n_active_total,) uint8 (0=internal, 1=leaf)
    INTERNAL_POS_ptr,      # (n_active_total,) int32 — exclusive seg-cumsum
    OFFSETS_ptr,           # (B,) int64 — per-tree start in active list
    XB_ptr,                # (B*N, D) uint8 — bootstrapped binned features
    SAMPLE_NODE_OUT_ptr,   # (B*N,) int32 — written: new per-tree compact id, -1 if leaf/inactive
    B, N, D,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offs = pid * BLOCK + tl.arange(0, BLOCK)
    BN = B * N
    mask = n_offs < BN

    sn = tl.load(SAMPLE_NODE_IN_ptr + n_offs, mask=mask, other=-1)
    valid = mask & (sn >= 0)

    tree_id = n_offs // N                                  # int32 tree id per slot
    per_tree_off = tl.load(OFFSETS_ptr + tree_id.to(tl.int64), mask=valid, other=0)
    global_id = sn.to(tl.int64) + per_tree_off             # safe; per_tree_off=0 when invalid

    feat = tl.load(BEST_FEAT_ptr + global_id, mask=valid, other=0).to(tl.int64)
    bin_thresh = tl.load(BEST_BIN_ptr + global_id, mask=valid, other=0)
    # X_bin[sample_idx_in_Xb, feat]; sample_idx_in_Xb = n_offs (Xb laid out (B*N, D))
    addr_x = n_offs.to(tl.int64) * D + feat
    bin_val = tl.load(XB_ptr + addr_x, mask=valid, other=0).to(tl.int32)
    go_left = bin_val <= bin_thresh

    ip = tl.load(INTERNAL_POS_ptr + global_id, mask=valid, other=0).to(tl.int32)
    new_compact_left = 2 * ip
    new_compact = tl.where(go_left, new_compact_left, new_compact_left + 1)

    leaf = tl.load(IS_LEAF_ptr + global_id, mask=valid, other=1).to(tl.int1)
    drop = leaf | (~valid)
    new_node = tl.where(drop, tl.full([BLOCK], -1, tl.int32), new_compact)
    tl.store(SAMPLE_NODE_OUT_ptr + n_offs, new_node, mask=mask)


def _partition_samples_fused(sample_node, best_feat, best_bin, is_leaf,
                              internal_pos_full, offsets_b, Xb):
    """One-kernel partition: each sample reads its split, decides left/right,
    handles leaves and inactive samples. Writes new sample_node tensor.
    """
    B, N = sample_node.shape
    _, D = Xb.shape
    device = sample_node.device
    out = torch.empty_like(sample_node)
    is_leaf_u8 = is_leaf.to(torch.uint8).contiguous()
    BLOCK = 1024
    grid = ((B * N + BLOCK - 1) // BLOCK,)
    _rf_partition_kernel[grid](
        sample_node.contiguous().view(-1),
        best_feat.contiguous(), best_bin.contiguous(),
        is_leaf_u8, internal_pos_full.contiguous(),
        offsets_b.contiguous(),
        Xb,
        out.view(-1),
        B, N, D,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


def _build_node_histograms_hybrid(X_bin, y, sample_node, n_active, n_bins, n_classes):
    """Hybrid hist build: pick scatter vs ranged based on n_active.

    Scatter (grid scaled by N) wins when n_active is large (many tiny nodes
    overpay CTA launch overhead in the ranged design). Ranged (grid scaled by
    n_active) wins when nodes are big enough to amortize per-CTA setup.

    Crossover ~n_active=4096 in our measurements.
    """
    N, D = X_bin.shape
    # Use ranged ONLY when avg samples-per-node is large enough that the ranged
    # kernel's better L2 reuse beats scatter PLUS amortizes the argsort+bincount
    # setup cost. Below ~32 samples/node, scatter wins.
    avg = N // max(n_active, 1)
    use_ranged = (n_active <= 4096) and (avg >= 64) and (N >= 200_000)
    if use_ranged:
        sn_safe = torch.where(sample_node >= 0, sample_node,
                               torch.full_like(sample_node, n_active))
        row_perm = sn_safe.argsort(stable=True).to(torch.int32).contiguous()
        counts_full = torch.bincount(sn_safe, minlength=n_active + 1).to(torch.int32)
        node_offsets = torch.zeros(n_active + 1, dtype=torch.int32, device=X_bin.device)
        node_offsets[1:] = counts_full[:n_active].cumsum(0).to(torch.int32)
        return _build_node_histograms_ranged(X_bin, y, row_perm, node_offsets,
                                              n_active, n_bins, n_classes)
    return _build_node_histograms(X_bin, y, sample_node, n_active, n_bins, n_classes)

