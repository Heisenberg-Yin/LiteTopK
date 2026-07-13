"""flash-rf orchestration: quantile binning, level-wise BFS tree growth
and the public :class:`FlashRandomForestClassifier`.

All Triton kernels (and their direct python wrappers) live under
``flashlib.primitives.random_forest.triton.rf_kernels``; this module
imports them and composes the high-level training / inference loops.
"""
import math
import numpy as np
import torch

from flashlib.primitives.random_forest.triton.rf_kernels import (
    _build_node_histograms,
    _build_node_histograms_subfeat,
    _build_node_histograms_subfeat_hybrid,
    _build_node_histograms_hybrid,
    _build_node_histograms_ranged,
    _find_best_splits_triton,
    _find_best_splits_subfeat,
    _split_counts_fused,
    _hist_subtract_fused,
    _partition_samples_fused,
)


# =============================================================================
# Helpers in pure torch (small ops, not the bottleneck)
# =============================================================================

def _quantile_bin(X, n_bins):
    """Bin features into uint8 [0, n_bins-1] using quantiles.
    Returns (X_bin uint8, bin_edges (D, n_bins-1) fp32).
    """
    N, D = X.shape
    # Subsample for quantile computation if N is huge
    n_sample = min(N, 65536)
    if N > n_sample:
        idx = torch.randperm(N, device=X.device)[:n_sample]
        Xs = X[idx]
    else:
        Xs = X
    # Compute quantiles per feature
    qs = torch.linspace(0, 1, n_bins + 1, device=X.device)[1:-1]  # (n_bins-1,)
    bin_edges = torch.quantile(Xs.T.contiguous(), qs, dim=1).T.contiguous()  # (D, n_bins-1)
    return _quantile_bin_apply(X, bin_edges), bin_edges


def _quantile_bin_apply(X, bin_edges):
    """Apply bin edges to data — fully vectorized via batched searchsorted.

    `torch.searchsorted` is batched on the boundaries tensor: passing
    bin_edges shape (D, n_bins-1) and X.T shape (D, N) produces (D, N) bin
    indices in one launch — replaces the Python `for d in range(D)` loop
    that ran D separate `torch.bucketize` calls (29 ms → ~1 ms at D=50).
    """
    N, D = X.shape
    # bin_edges: (D, n_bins-1), X: (N, D) → transpose to (D, N) for batched search
    X_T = X.T.contiguous()
    bin_idx = torch.searchsorted(bin_edges, X_T)  # (D, N) int64
    X_bin = bin_idx.T.contiguous().to(torch.uint8)  # (N, D)
    return X_bin


def _find_best_splits_torch(hist, n_classes, feat_mask=None):
    """[Legacy] Pure-torch best split via cumsum (slow for deep trees, kept for fallback).

    hist: (n_active, D, n_bins, K) — raw class counts at each (feat, bin).
    feat_mask: optional (n_active, D) bool. If provided, only features with
               mask=True are considered for the split (RF feature subsampling).

    Returns (best_feat, best_bin, best_gain, leaf_class) all (n_active,).
    """
    n_active, D, n_bins, K = hist.shape
    total = hist[:, 0, :, :].sum(dim=1)              # (n_active, K)
    n_total = total.sum(dim=1).clamp(min=1).float()  # (n_active,)
    gini_parent = 1.0 - ((total.float() / n_total[:, None]) ** 2).sum(dim=1)

    cum_left = hist.cumsum(dim=2)                    # (n_active, D, n_bins, K)
    cum_total = cum_left[:, :, -1:, :]               # (n_active, D, 1, K)
    cum_right = cum_total - cum_left

    n_left = cum_left.sum(dim=3)                      # (n_active, D, n_bins)
    n_right = cum_right.sum(dim=3)
    n_left_safe = n_left.clamp(min=1).float()
    n_right_safe = n_right.clamp(min=1).float()
    gini_left = 1.0 - ((cum_left.float() / n_left_safe[..., None]) ** 2).sum(dim=3)
    gini_right = 1.0 - ((cum_right.float() / n_right_safe[..., None]) ** 2).sum(dim=3)
    weighted = (n_left.float() * gini_left + n_right.float() * gini_right) / n_total[:, None, None].float()
    gain = gini_parent[:, None, None] - weighted

    invalid = (n_left == 0) | (n_right == 0)
    gain = torch.where(invalid, torch.full_like(gain, -1.0), gain)
    gain[..., -1] = -1.0  # right side empty for last bin

    # Apply feature subsampling mask: features not selected get gain = -inf
    if feat_mask is not None:
        # feat_mask: (n_active, D) — broadcast to (n_active, D, n_bins)
        masked_gain_value = torch.full_like(gain, -1e30)
        gain = torch.where(feat_mask[:, :, None], gain, masked_gain_value)

    flat = gain.view(n_active, -1)
    best_idx = flat.argmax(dim=1)
    best_gain = flat.gather(1, best_idx[:, None]).squeeze(1)
    best_feat = (best_idx // n_bins).to(torch.int32)
    best_bin = (best_idx % n_bins).to(torch.int32)

    leaf_class = total.argmax(dim=1).to(torch.int32)
    return best_feat, best_bin, best_gain, leaf_class


def _partition_samples(X_bin, sample_node, splits, n_active):
    """Update sample_node based on splits.

    splits: dict with 'feat', 'bin', 'is_leaf', 'left_id', 'right_id' tensors of
    shape (n_active,). For non-leaf nodes, samples in node n move to left_id[n]
    or right_id[n] based on whether their bin <= bin[n]. For leaves, move to -1
    (inactive).

    Returns new sample_node (N,) int32.
    """
    N = sample_node.shape[0]
    feat = splits['feat']           # (n_active,)
    bin_ = splits['bin']            # (n_active,)
    is_leaf = splits['is_leaf']     # (n_active,) bool
    left_id = splits['left_id']     # (n_active,)
    right_id = splits['right_id']   # (n_active,)

    new_sample_node = torch.full_like(sample_node, -1)
    active_mask = (sample_node >= 0) & (sample_node < n_active)
    active_idx = sample_node.clamp(min=0)
    sample_feat = feat[active_idx]
    sample_bin_thresh = bin_[active_idx]
    # Read X_bin at (sample, feat) — this is a gather
    rows = torch.arange(N, device=X_bin.device)
    sample_bin = X_bin[rows, sample_feat.long()].to(torch.int32)
    go_left = sample_bin <= sample_bin_thresh
    target = torch.where(go_left, left_id[active_idx], right_id[active_idx])
    # Leaves stay inactive (target = -1)
    sample_is_leaf = is_leaf[active_idx]
    target = torch.where(sample_is_leaf, torch.full_like(target, -1), target)
    new_sample_node = torch.where(active_mask, target, new_sample_node)
    return new_sample_node


# =============================================================================
# Tree storage: arrays of (left, right, feat, bin, leaf_class)
# Internal nodes: left/right >= 0, feat/bin set
# Leaves:        left = right = -1, leaf_class set
# =============================================================================

class _Tree:
    __slots__ = ('left', 'right', 'feat', 'bin', 'leaf_class')

    def __init__(self, max_nodes, device):
        self.left = torch.full((max_nodes,), -1, dtype=torch.int32, device=device)
        self.right = torch.full((max_nodes,), -1, dtype=torch.int32, device=device)
        self.feat = torch.zeros(max_nodes, dtype=torch.int32, device=device)
        self.bin = torch.zeros(max_nodes, dtype=torch.int32, device=device)
        self.leaf_class = torch.zeros(max_nodes, dtype=torch.int32, device=device)


def _build_tree(X_bin, y, sample_idx, max_depth, n_bins, n_classes,
                min_samples_split=2, min_gain=1e-7,
                max_features='sqrt', feat_seed=0):
    """Build a single tree from bootstrap sample indices.

    Vectorized active-node tracking: `active_node_ids` is a torch tensor (not
    a Python list), all per-node loops are eliminated via tensor scatter/gather.

    max_features: 'sqrt' (default), 'log2', None (all), or int. Per-split
        random feature subsampling — matches sklearn RF default.
    """
    N, D = X_bin.shape
    device = X_bin.device

    if max_features is None or max_features == 'all':
        n_feat_per_split = D
    elif max_features == 'sqrt':
        n_feat_per_split = max(1, int(D ** 0.5))
    elif max_features == 'log2':
        import math as _m
        n_feat_per_split = max(1, int(_m.log2(D)))
    elif isinstance(max_features, int):
        n_feat_per_split = min(D, max_features)
    else:
        raise ValueError(f"max_features={max_features!r}")
    feat_gen = torch.Generator(device=device).manual_seed(feat_seed)

    Xb = X_bin[sample_idx]
    yb = y[sample_idx]

    max_nodes = 2 ** (max_depth + 1)
    tree = _Tree(max_nodes, device)

    # ── Vectorized state ──
    # sample_node[i] = compact id of active node sample i is in (0..n_active-1),
    # OR -1 if the sample is in an inactive (leaf) node.
    sample_node = torch.zeros(N, dtype=torch.int32, device=device)
    # active_node_ids: int32 tensor of original tree-node ids for the current level
    active_node_ids = torch.zeros(1, dtype=torch.int32, device=device)
    next_node_id = 1  # int32 scalar tracked outside GPU

    for depth in range(max_depth):
        n_active = active_node_ids.shape[0]
        if n_active == 0:
            break

        # Build histograms for all active nodes (sample_node already in compact 0..n_active-1)
        hist = _build_node_histograms(Xb, yb, sample_node,
                                       n_active, n_bins, n_classes)

        # Per-node feature subsampling
        feat_mask = None
        if n_feat_per_split < D:
            scores = torch.rand((n_active, D), device=device,
                                  generator=feat_gen, dtype=torch.float32)
            ranks = scores.argsort(dim=1)
            feat_mask = torch.zeros((n_active, D), dtype=torch.bool, device=device)
            feat_mask.scatter_(1, ranks[:, :n_feat_per_split], True)

        best_feat, best_bin, best_gain, leaf_class = _find_best_splits_triton(
            hist, n_classes, feat_mask=feat_mask)

        is_last_depth = (depth == max_depth - 1)
        is_leaf = (best_gain < min_gain) | is_last_depth

        # Allocate child ids vectorized: each non-leaf gets a (left, right) pair.
        is_internal = ~is_leaf
        is_internal_i32 = is_internal.to(torch.int32)
        # internal_pos[i] = number of internal nodes BEFORE i (exclusive)
        internal_pos = (is_internal_i32.cumsum(0).to(torch.int32) - is_internal_i32)
        left_ids = torch.where(
            is_internal,
            (next_node_id + internal_pos * 2).to(torch.int32),
            torch.full((n_active,), -1, dtype=torch.int32, device=device))
        right_ids = torch.where(is_internal,
                                 (left_ids + 1).to(torch.int32),
                                 torch.full((n_active,), -1, dtype=torch.int32, device=device))
        n_internal = int(is_internal.sum().item())  # one .item() per level — OK
        next_node_id += 2 * n_internal

        # Write into tree storage at active_node_ids (vectorized scatter)
        ids_long = active_node_ids.to(torch.int64)
        tree.feat[ids_long] = best_feat
        tree.bin[ids_long] = best_bin
        tree.left[ids_long] = left_ids
        tree.right[ids_long] = right_ids
        tree.leaf_class[ids_long] = leaf_class

        # ── Vectorized partition ──
        # For each sample i in compact node c = sample_node[i]:
        #   - If is_leaf[c]: new sample_node[i] = -1
        #   - Else: read x = Xb[i, best_feat[c]], compare to best_bin[c], pick
        #           left_compact_id or right_compact_id as the NEW compact id.
        # We need NEW compact ids. They will be 0..2*n_internal-1 in the next
        # level. Specifically, internal node compact-i → children compact ids
        # (2*internal_pos[i], 2*internal_pos[i]+1).
        if n_internal > 0:
            new_compact_left = (2 * internal_pos).to(torch.int32)
            new_compact_right = (new_compact_left + 1).to(torch.int32)
            # Per-sample lookups via gather
            valid_sample = sample_node >= 0
            cidx_safe = sample_node.clamp(min=0).to(torch.int64)
            sample_feat = best_feat[cidx_safe].to(torch.int64)
            sample_thresh = best_bin[cidx_safe]
            sample_x = Xb.gather(1, sample_feat.unsqueeze(1)).squeeze(1).to(torch.int32)
            go_left = sample_x <= sample_thresh
            new_node = torch.where(go_left,
                                   new_compact_left[cidx_safe],
                                   new_compact_right[cidx_safe])
            sample_is_leaf = is_leaf[cidx_safe]
            sample_node = torch.where(valid_sample & ~sample_is_leaf, new_node,
                                       torch.full_like(sample_node, -1))
            # Build next level's active_node_ids: for each internal node, append (left_id, right_id)
            internal_mask = is_internal
            internal_idx = internal_mask.nonzero(as_tuple=True)[0]
            l_ids = left_ids[internal_idx]
            r_ids = right_ids[internal_idx]
            active_node_ids = torch.stack([l_ids, r_ids], dim=1).reshape(-1)
        else:
            sample_node = torch.full_like(sample_node, -1)
            active_node_ids = torch.zeros(0, dtype=torch.int32, device=device)

    # Any leftover active nodes (shouldn't happen due to is_last_depth above) → leaves
    if active_node_ids.numel() > 0:
        ids_long = active_node_ids.to(torch.int64)
        tree.left[ids_long] = -1
        tree.right[ids_long] = -1

    return tree


def _build_trees_batched(X_bin, y, sample_idx_batch, max_depth, n_bins, n_classes,
                          min_samples_split=2, min_gain=1e-7,
                          max_features='sqrt', feat_seed=0):
    """Build B trees in parallel — single kernel launches over B*n_active_per_tree nodes.

    sample_idx_batch: (B, N) — bootstrap indices for each tree.
    Returns: list of B _Tree instances.
    """
    B, N = sample_idx_batch.shape
    _, D = X_bin.shape
    device = X_bin.device

    if max_features is None or max_features == 'all':
        n_feat_per_split = D
    elif max_features == 'sqrt':
        n_feat_per_split = max(1, int(D ** 0.5))
    elif max_features == 'log2':
        import math as _m
        n_feat_per_split = max(1, int(_m.log2(D)))
    elif isinstance(max_features, int):
        n_feat_per_split = min(D, max_features)
    else:
        raise ValueError(f"max_features={max_features!r}")
    feat_gen = torch.Generator(device=device).manual_seed(feat_seed)

    # Bootstrap-gathered data, flat layout (B*N, D)
    Xb = X_bin[sample_idx_batch.view(-1)]      # (B*N, D)
    yb = y[sample_idx_batch.view(-1)]          # (B*N,)

    max_nodes = 2 ** (max_depth + 1)

    # ── Batched tree storage (B, max_nodes) — single tensor per field, scattered in
    # one op per level. At end, split into per-tree _Tree instances. Eliminates
    # the B-iteration Python loop with 2*B `.item()` syncs per level (≈3000+ syncs
    # avoided per fit at small).
    feat_b = torch.zeros((B, max_nodes), dtype=torch.int32, device=device)
    bin_b = torch.zeros((B, max_nodes), dtype=torch.int32, device=device)
    left_b = torch.full((B, max_nodes), -1, dtype=torch.int32, device=device)
    right_b = torch.full((B, max_nodes), -1, dtype=torch.int32, device=device)
    leaf_class_b = torch.zeros((B, max_nodes), dtype=torch.int32, device=device)

    # GPU-side per-tree next-node counter (avoids .cpu()/.tolist() per level)
    next_node_t = torch.ones(B, dtype=torch.int32, device=device)

    # Per-(tree, sample) state. sample_node[t, n] is the COMPACT id within tree t.
    sample_node = torch.zeros((B, N), dtype=torch.int32, device=device)

    # Per-tree active node ids (in the tree's own id space). Initially all 1 each (root=0).
    n_active_per_tree = torch.ones(B, dtype=torch.int64, device=device)
    # Flat list of active ids: starts as [0,0,...,0] (B zeros, root for each tree).
    active_ids_flat = torch.zeros(B, dtype=torch.int32, device=device)

    # ── Pre-allocated fixed-size scratch buffers (size B or B+1).
    # Avoids ~3 small `torch.zeros` allocations per level × max_depth levels × n_batches.
    # At small (depth=16, n_trees/B≈12 batches), saves ~600 torch.zeros calls per fit.
    offsets_buf = torch.zeros(B + 1, dtype=torch.int64, device=device)
    n_internal_per_tree_buf = torch.zeros(B, dtype=torch.int32, device=device)
    cum_before_start_buf = torch.zeros(B, dtype=torch.int32, device=device)
    arange_B = torch.arange(B, device=device, dtype=torch.int64)

    # Sub-feature hist (cuML-style): compute hist only over n_feat_per_split
    # features per node instead of all D. Memory & compute scale by
    # n_feat_per_split/D — 23× reduction at xlarge.
    #
    # Per-tree sampling (random-subspace variant of RF): all nodes in tree t
    # share the SAME n_feat_per_split features. This deviates from textbook RF
    # (per-node sampling) but enables histogram subtraction — siblings have
    # the same features so `bigger = parent − smaller` is valid. The ~2×
    # hist-work reduction at non-root levels outweighs the slight quality cost
    # in practice (well-known trick used by LightGBM/XGBoost subspace mode).
    import os as _os
    use_subfeat = (n_feat_per_split < D
                   and _os.environ.get("FLASH_RF_SUBFEAT", "1") != "0")
    # Per-tree feature sampling enables hist subtraction even with sub-feat,
    # but reduces split quality (fewer features considered at each node) which
    # can grow trees deeper. Net win only when hist work dominates — gate to
    # N≥50K, the same threshold as hist subtraction itself.
    use_per_tree_feat = (use_subfeat
                         and N >= 50_000
                         and _os.environ.get("FLASH_RF_PER_TREE_FEAT", "1") != "0")

    if use_per_tree_feat:
        # tree_feat_idx: (B, n_feat_per_split) int32 — features for each tree
        tree_scores = torch.rand((B, D), device=device, generator=feat_gen,
                                  dtype=torch.float32)
        tree_feat_idx = (tree_scores.argsort(dim=1)[:, :n_feat_per_split]
                         .to(torch.int32).contiguous())
    else:
        tree_feat_idx = None

    # Hist subtraction is enabled for: (a) full-D path with N≥50K, OR
    # (b) per-tree-feat subfeat path (siblings share features).
    use_hist_sub_full = (not use_subfeat
                         and N >= 50_000
                         and _os.environ.get("FLASH_RF_HIST_SUB", "1") != "0")
    use_hist_sub_subfeat = (use_subfeat and use_per_tree_feat
                             and N >= 50_000
                             and _os.environ.get("FLASH_RF_HIST_SUB", "1") != "0")
    use_hist_sub = use_hist_sub_full or use_hist_sub_subfeat
    prev_hist = None
    prev_internal_global_idx = None
    prev_internal_pos_full = None
    prev_tree_id_per_node = None
    prev_best_subfeat = None
    prev_best_bin = None

    for depth in range(max_depth):
        # offsets[t] = start index of tree t in the flat active list (reuse buffer)
        offsets = offsets_buf
        offsets[0] = 0
        offsets[1:] = n_active_per_tree.cumsum(0)
        n_active_total = int(offsets[-1].item())  # one sync per level (unavoidable for grid sizing)
        if n_active_total == 0:
            break

        # Compute per-sample global compact id
        per_tree_offset = offsets[:B][:, None]                             # (B, 1)
        global_node = sample_node.to(torch.int64) + per_tree_offset        # (B, N)
        global_node = torch.where(sample_node >= 0, global_node,
                                   torch.full_like(global_node, -1))
        global_node_flat = global_node.view(-1).to(torch.int32)

        # tree_id_per_node[i] = which tree node i belongs to (computed early so
        # the subfeat hist build can broadcast per-tree feat_idx → per-node).
        # searchsorted is ~3× faster than repeat_interleave at our sizes.
        arange_total = torch.arange(n_active_total, device=device, dtype=torch.int64)
        tree_id_per_node = torch.searchsorted(offsets, arange_total, right=True) - 1

        # ── Build histograms for current level ──
        # Path A: sub-feat (hist over only n_feat_per_split features per node).
        #   With per-tree sampling: tree_feat_idx (B, n_feat) → broadcast to per
        #   node via tree_id_per_node[i] lookup. Enables hist subtraction since
        #   siblings share parent's tree's features.
        #   With per-node sampling: each node samples its own random features —
        #   no hist subtraction.
        # Path B/C: full-D hist with optional hist subtraction (at depth>0).
        if use_subfeat:
            if use_per_tree_feat:
                feat_idx_per_node = tree_feat_idx[tree_id_per_node].contiguous()
            else:
                scores = torch.rand((n_active_total, D), device=device,
                                    generator=feat_gen, dtype=torch.float32)
                feat_idx_per_node = (scores.argsort(dim=1)[:, :n_feat_per_split]
                                     .to(torch.int32).contiguous())
            n_active_prev = prev_hist.shape[0] if prev_hist is not None else 0
            hist_sub_mem_bytes = (n_active_prev * n_feat_per_split
                                  * n_bins * n_classes * 4)
            do_hist_sub = (prev_hist is not None and use_hist_sub_subfeat
                           and hist_sub_mem_bytes < 4 * 1024**3)
            if not do_hist_sub:
                hist = _build_node_histograms_subfeat_hybrid(
                    Xb, yb, global_node_flat, feat_idx_per_node,
                    n_active_total, n_bins, n_classes)
                if prev_hist is not None:
                    del prev_hist
                    prev_hist = None
            else:
                # Hist subtraction in subfeat space (per-tree features only).
                ip_within_t = prev_internal_pos_full[prev_internal_global_idx]
                tree_per_internal = prev_tree_id_per_node[prev_internal_global_idx]
                cur_per_tree_offset = offsets[:B]
                left_global = (cur_per_tree_offset[tree_per_internal]
                               + 2 * ip_within_t.to(torch.int64))
                right_global = left_global + 1
                # Derive left/right child sample counts via fused Triton kernel
                # that reads prev_hist[parent, split_subfeat, :, :] inline and
                # computes (left_count, total) per parent. Avoids the big
                # intermediate (n_internal_prev, n_bins, K) copy from torch's
                # fancy indexing + sum + cumsum chain.
                pinternals = prev_internal_global_idx.to(torch.int64)
                psubfeat = prev_best_subfeat[pinternals].to(torch.int64)
                pbin = prev_best_bin[pinternals].to(torch.int32)
                left_counts, total_per_parent = _split_counts_fused(
                    prev_hist, pinternals, psubfeat, pbin)
                right_counts = total_per_parent - left_counts
                left_is_smaller = left_counts <= right_counts
                smaller_global = torch.where(left_is_smaller, left_global, right_global)
                bigger_global = torch.where(left_is_smaller, right_global, left_global)
                is_smaller = torch.zeros(n_active_total, dtype=torch.bool, device=device)
                is_smaller[smaller_global] = True
                sample_global_safe = global_node_flat.clamp(min=0).to(torch.int64)
                keep_per_sample = (global_node_flat >= 0) & is_smaller[sample_global_safe]
                sample_global_filtered = torch.where(
                    keep_per_sample, global_node_flat,
                    torch.full_like(global_node_flat, -1))
                half_hist = _build_node_histograms_subfeat_hybrid(
                    Xb, yb, sample_global_filtered, feat_idx_per_node,
                    n_active_total, n_bins, n_classes)
                _hist_subtract_fused(prev_hist, half_hist,
                                      prev_internal_global_idx.to(torch.int64),
                                      smaller_global, bigger_global)
                hist = half_hist
                del prev_hist
        else:
            n_active_prev = prev_hist.shape[0] if prev_hist is not None else 0
            hist_sub_mem_bytes = n_active_prev * D * n_bins * n_classes * 4
            do_hist_sub = (prev_hist is not None and use_hist_sub
                           and hist_sub_mem_bytes < 4 * 1024**3)
            if not do_hist_sub:
                hist = _build_node_histograms_hybrid(Xb, yb, global_node_flat,
                                                      n_active_total, n_bins, n_classes)
                if prev_hist is not None:
                    del prev_hist
                    prev_hist = None
            else:
                n_internal_prev = prev_internal_global_idx.numel()
                # Map each prev internal → its two children's GLOBAL compact ids in current level
                ip_within_t = prev_internal_pos_full[prev_internal_global_idx]
                tree_per_internal = prev_tree_id_per_node[prev_internal_global_idx]
                cur_per_tree_offset = offsets[:B]
                left_global = (cur_per_tree_offset[tree_per_internal]
                               + 2 * ip_within_t.to(torch.int64))
                right_global = left_global + 1
                counts = torch.bincount(
                    global_node_flat[global_node_flat >= 0].to(torch.int64),
                    minlength=n_active_total)
                left_counts = counts[left_global]
                right_counts = counts[right_global]
                left_is_smaller = left_counts <= right_counts
                smaller_global = torch.where(left_is_smaller, left_global, right_global)
                bigger_global = torch.where(left_is_smaller, right_global, left_global)
                is_smaller = torch.zeros(n_active_total, dtype=torch.bool, device=device)
                is_smaller[smaller_global] = True
                sample_global_safe = global_node_flat.clamp(min=0).to(torch.int64)
                keep_per_sample = (global_node_flat >= 0) & is_smaller[sample_global_safe]
                sample_global_filtered = torch.where(
                    keep_per_sample, global_node_flat,
                    torch.full_like(global_node_flat, -1))
                half_hist = _build_node_histograms_hybrid(
                    Xb, yb, sample_global_filtered, n_active_total, n_bins, n_classes)
                _hist_subtract_fused(prev_hist, half_hist,
                                      prev_internal_global_idx.to(torch.int64),
                                      smaller_global, bigger_global)
                hist = half_hist
                del prev_hist

        # Best-split: in subfeat path, hist is (n_active, n_feat_per_split, n_bins, K)
        # so we use the wrapper that maps best_k_sub → actual feat. In the full-D
        # path we apply feat_mask post-hoc.
        if use_subfeat:
            (best_feat, best_bin, best_gain, leaf_class,
             cur_best_subfeat) = _find_best_splits_subfeat(
                hist, n_classes, feat_idx_per_node)
        else:
            feat_mask = None
            if n_feat_per_split < D:
                scores = torch.rand((n_active_total, D), device=device,
                                      generator=feat_gen, dtype=torch.float32)
                ranks = scores.argsort(dim=1)
                feat_mask = torch.zeros((n_active_total, D), dtype=torch.bool, device=device)
                feat_mask.scatter_(1, ranks[:, :n_feat_per_split], True)
            best_feat, best_bin, best_gain, leaf_class = _find_best_splits_triton(
                hist, n_classes, feat_mask=feat_mask)
            cur_best_subfeat = None  # only used in subfeat hist-sub path

        is_last_depth = (depth == max_depth - 1)
        is_leaf = (best_gain < min_gain) | is_last_depth
        is_internal = ~is_leaf

        is_int_i32 = is_internal.to(torch.int32)

        # tree_id_per_node already computed above (before hist build).

        # Per-tree internal counts via scatter_add (reuse buffer)
        n_internal_per_tree = n_internal_per_tree_buf
        n_internal_per_tree.zero_()
        n_internal_per_tree.scatter_add_(0, tree_id_per_node, is_int_i32)

        # Segmented cumsum: internal_pos[i] = #internals before i WITHIN its tree
        global_cumsum = is_int_i32.cumsum(0).to(torch.int32)
        starts = offsets[:B]
        cum_before_start = cum_before_start_buf
        cum_before_start.zero_()
        cum_before_start[1:] = global_cumsum[starts[1:] - 1]
        base_per_node = cum_before_start[tree_id_per_node]
        internal_pos_full = global_cumsum - base_per_node - is_int_i32  # exclusive

        # Per-node next-node assignment: read each tree's running counter, then bump it
        next_per_node = next_node_t[tree_id_per_node]
        left_ids = torch.where(is_internal,
                                next_per_node + internal_pos_full * 2,
                                torch.full((n_active_total,), -1, dtype=torch.int32, device=device))
        right_ids = torch.where(is_internal,
                                 (left_ids + 1).to(torch.int32),
                                 torch.full((n_active_total,), -1, dtype=torch.int32, device=device))
        # Vectorized counter bump (GPU, no sync)
        next_node_t = next_node_t + 2 * n_internal_per_tree

        # ── Single vectorized scatter into the batched tree storage (B, max_nodes)
        # flat_idx[i] = tree_id[i] * max_nodes + active_ids_flat[i]
        flat_idx = (tree_id_per_node * max_nodes
                    + active_ids_flat.to(torch.int64))                # (n_active_total,)
        feat_b.view(-1)[flat_idx] = best_feat
        bin_b.view(-1)[flat_idx] = best_bin
        left_b.view(-1)[flat_idx] = left_ids
        right_b.view(-1)[flat_idx] = right_ids
        leaf_class_b.view(-1)[flat_idx] = leaf_class

        # Fused partition kernel — replaces ~14 torch ops with one Triton launch.
        sample_node = _partition_samples_fused(
            sample_node, best_feat, best_bin, is_leaf,
            internal_pos_full, offsets[:B], Xb)

        # Vectorized: next-level active list = interleaved (left, right) for internal nodes
        internal_idx = is_internal.nonzero(as_tuple=True)[0]
        new_active_ids = torch.stack([left_ids[internal_idx], right_ids[internal_idx]],
                                       dim=1).reshape(-1)
        # New active count per tree = 2 * n_internal_per_tree
        n_active_per_tree = 2 * n_internal_per_tree.to(torch.int64)
        active_ids_flat = new_active_ids

        # Save state for hist subtraction at next level (only if enabled)
        if use_hist_sub:
            prev_hist = hist
            prev_internal_global_idx = internal_idx
            prev_internal_pos_full = internal_pos_full
            prev_tree_id_per_node = tree_id_per_node
            prev_best_subfeat = cur_best_subfeat
            prev_best_bin = best_bin

    # Convert batched storage to per-tree _Tree instances (one-shot at end)
    trees = []
    for t in range(B):
        tree = _Tree(max_nodes, device)
        tree.feat = feat_b[t]
        tree.bin = bin_b[t]
        tree.left = left_b[t]
        tree.right = right_b[t]
        tree.leaf_class = leaf_class_b[t]
        trees.append(tree)
    return trees


def _partition_samples_orig(X_bin, sample_node, active_node_ids,
                             best_feat, best_bin, is_leaf,
                             left_ids, right_ids):
    """Partition samples from active nodes (original ids) into children.
    Returns new sample_node in original-id space."""
    N = sample_node.shape[0]
    device = sample_node.device

    # active_to_compact map (original id -> compact index in active list)
    max_id = int(max(active_node_ids)) + 1 if active_node_ids else 1
    a2c = torch.full((max_id,), -1, dtype=torch.int32, device=device)
    for ci, oid in enumerate(active_node_ids):
        a2c[oid] = ci
    # Per sample: compact id of its current node (or -1 if not active)
    cur_id = sample_node
    in_range = (cur_id >= 0) & (cur_id < max_id)
    cidx = torch.where(in_range, a2c[cur_id.clamp(min=0).long()], torch.full_like(cur_id, -1))
    active_mask = cidx >= 0

    # For each sample in an active node, look up its split decision
    cidx_safe = cidx.clamp(min=0).long()
    sample_feat = best_feat[cidx_safe]            # (N,)
    sample_thresh = best_bin[cidx_safe]
    rows = torch.arange(N, device=device)
    sample_bin = X_bin[rows, sample_feat.long()].to(torch.int32)

    go_left = sample_bin <= sample_thresh
    target = torch.where(go_left, left_ids[cidx_safe], right_ids[cidx_safe])
    leaf_mask = is_leaf[cidx_safe]
    new_node = torch.where(leaf_mask, torch.full_like(target, -1), target)
    new_sample_node = torch.where(active_mask, new_node, torch.full_like(sample_node, -1))
    return new_sample_node


# =============================================================================
# Public API
# =============================================================================

def _auto_batch_size(N, D, n_trees, target_xb_gb=25):
    """Pick a batch_size that keeps Xb tensor (B*N*D bytes) under target.

    Auto-tuning is critical: per-batch CPU loop overhead is ≈100ms regardless
    of B. At small (10K × 50, 100 trees), B=8 → 12 batches = 1200ms overhead;
    B=100 → 1 batch = 100ms. Measured 8.75× speedup at small from B=8 → B=100.
    """
    target_bytes = int(target_xb_gb * 1024**3)
    bytes_per_tree = max(1, N * D)  # uint8 X_bin
    B_max = max(1, target_bytes // bytes_per_tree)
    return min(n_trees, B_max)


class FlashRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=16, n_bins=16,
                 max_features='sqrt', batch_size=None, seed=0):
        """flash-rf classifier.

        max_features: 'sqrt', 'log2', None, or int — features per split.
        batch_size: number of trees built in parallel per CPU iteration. If
                    None, auto-tuned via `_auto_batch_size` to keep Xb under
                    ~20 GB.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.max_features = max_features
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        device = 'cuda'
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32, device=device)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.int32, device=device)
        X = X.to(device).contiguous()
        y = y.to(device).to(torch.int32).contiguous()
        N, D = X.shape
        self.n_classes_ = int(y.max().item()) + 1

        self.X_bin_, self.bin_edges_ = _quantile_bin(X, self.n_bins)

        self.trees_ = []
        gen = torch.Generator(device=device).manual_seed(self.seed)
        B = self.batch_size if self.batch_size is not None else _auto_batch_size(
            N, D, self.n_estimators)
        for batch_start in range(0, self.n_estimators, B):
            batch_end = min(batch_start + B, self.n_estimators)
            cur_B = batch_end - batch_start
            sample_idx_batch = torch.randint(
                0, N, (cur_B, N), device=device, generator=gen)
            new_trees = _build_trees_batched(
                self.X_bin_, y, sample_idx_batch,
                self.max_depth, self.n_bins, self.n_classes_,
                max_features=self.max_features,
                feat_seed=self.seed * 100003 + batch_start,
            )
            self.trees_.extend(new_trees)
        return self

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32, device='cuda')
        X = X.to('cuda').contiguous()
        X_bin = _quantile_bin_apply(X, self.bin_edges_)
        N = X_bin.shape[0]
        votes = torch.zeros(N, self.n_classes_, device='cuda', dtype=torch.float32)
        for tree in self.trees_:
            preds = self._predict_tree(tree, X_bin)
            votes.scatter_add_(1, preds[:, None].long(),
                                torch.ones(N, 1, device='cuda', dtype=torch.float32))
        return votes.argmax(dim=1).to(torch.int32)

    def _predict_tree(self, tree, X_bin):
        N, D = X_bin.shape
        cur = torch.zeros(N, dtype=torch.int32, device=X_bin.device)
        # iterate up to max_depth levels (forward traversal)
        for _ in range(self.max_depth + 1):
            # If left[cur] == -1, we've reached a leaf
            l = tree.left[cur.long()]
            r = tree.right[cur.long()]
            f = tree.feat[cur.long()].long()
            b = tree.bin[cur.long()]
            rows = torch.arange(N, device=X_bin.device)
            sample_bin = X_bin[rows, f].to(torch.int32)
            go_left = sample_bin <= b
            next_cur = torch.where(go_left, l, r)
            # If at leaf, stay
            at_leaf = (l < 0)
            cur = torch.where(at_leaf, cur, next_cur)
        # cur now contains leaf node ids
        return tree.leaf_class[cur.long()]
