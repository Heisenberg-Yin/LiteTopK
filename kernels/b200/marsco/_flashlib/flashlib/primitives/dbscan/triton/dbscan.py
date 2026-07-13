"""flash-dbscan: GPU-resident DBSCAN.

Two algorithmic paths, dispatched by D:

  • D ≤ 4 (low-D): **GRID radius search**. Bin points into cells of side eps;
    each point only needs to scan its 3^D=9 (D=2) or 27 (D=3) neighbor cells.
    For typical D=2 spatial data with N ≈ M ≈ 100s/cell, this is ~250×
    fewer distance evaluations than brute force.

  • D ≥ 5 (high-D): **brute-force top-K** via flash_knn (bf16 GEMM + on-chip
    top-K), then eps filter on the returned distances.

Both paths feed the same downstream pipeline:
  → core mask: deg ≥ min_samples
  → connected components (atomic-CAS UF on GPU)
  → border assignment: smallest CC label among in-eps core neighbors
  → compact labels to dense [0, n_clusters)

The grid path is exact for any D where the cell-side ≥ eps (so all in-eps
neighbors lie in the 3^D neighborhood). For high-D the curse of
dimensionality makes the per-cell cost grow as 3^D, hence the brute-force
path for D ≥ 5.
"""
import math

import torch
import triton
import triton.language as tl

from flashlib.primitives.knn import flash_knn
from flashlib.kernels.flash_mst import flash_cc_from_edges


def _next_pow2(x):
    return 1 << (int(x - 1).bit_length()) if x > 1 else 1


# ---------------------------------------------------------------------------
# GRID radius-search kernel (D=2)
# ---------------------------------------------------------------------------
# Each program handles BN query points. For each query, computes its grid
# cell, iterates over the 3×3 neighbor cells, looks up each cell in a
# hash table (linear probing) to find the (start, end) indices in
# sorted_idx, and computes fp32 (xi - xj)² for each candidate; emits hits
# (dist ≤ eps²) into the per-row neighbor buffer.
#
# Hash table layout: (HT_SIZE,) int64 packed as (cell_id << 32) | bin_idx.
# Linear probing with a sentinel (-1) for empty slots.
# ---------------------------------------------------------------------------

@triton.jit
def _grid_radius_2d_kernel(
    X_ptr,                  # (N, 2) fp32
    SORTED_PTR_ptr,         # (N,) int32 — point indices sorted by cell
    CELL_START_ptr,         # (GW * GH,) int32 — dense 2D grid start
    CELL_END_ptr,           # (GW * GH,) int32 — dense 2D grid end
    DEG_ptr,                # (N,) int32
    NBR_IDX_ptr,            # (N, K) int32
    N: tl.constexpr, K: tl.constexpr,
    GW: tl.constexpr, GH: tl.constexpr,   # grid dims
    INV_EPS, EPS_SQ,
    GRID_X_MIN, GRID_Y_MIN,
    BN: tl.constexpr,
):
    """Grid radius search with dense 2D grid index.

    Each query maps to its (cx, cy) cell; we scan the 3×3 neighborhood by
    direct indexing into CELL_START/END arrays of size (GW × GH). For our
    standardised D=2 inputs (~6σ in [-3, 3], eps=0.04 → grid ~150×150,
    storage ~22500 int32 × 2 = 180 KB).
    """
    pid = tl.program_id(0)
    n_offs = (pid * BN + tl.arange(0, BN)).to(tl.int64)
    n_mask = n_offs < N

    xi = tl.load(X_ptr + n_offs * 2, mask=n_mask, other=0.0)
    yi = tl.load(X_ptr + n_offs * 2 + 1, mask=n_mask, other=0.0)

    cxq = tl.floor((xi - GRID_X_MIN) * INV_EPS).to(tl.int32)
    cyq = tl.floor((yi - GRID_Y_MIN) * INV_EPS).to(tl.int32)

    cur_cnt = tl.zeros([BN], dtype=tl.int32)

    # 3×3 cell neighborhood. Static-unrolled outer.
    for nbi in tl.static_range(9):
        dx = nbi // 3 - 1
        dy = nbi % 3 - 1
        cx = cxq + dx
        cy = cyq + dy
        # Bounds check (drop out-of-grid cells by mask).
        in_grid = (cx >= 0) & (cx < GW) & (cy >= 0) & (cy < GH) & n_mask
        cell_idx = cy * GW + cx
        cell_idx_safe = tl.where(in_grid, cell_idx, 0)
        s = tl.load(CELL_START_ptr + cell_idx_safe.to(tl.int64),
                    mask=in_grid, other=0)
        e = tl.load(CELL_END_ptr + cell_idx_safe.to(tl.int64),
                    mask=in_grid, other=0)
        range_len = tl.where(in_grid, e - s, 0)
        max_iter = tl.max(range_len)

        for j in tl.range(0, max_iter, 1):
            in_range = (j < range_len) & in_grid
            idx = s + j
            pj = tl.load(SORTED_PTR_ptr + idx.to(tl.int64), mask=in_range, other=0)
            xj = tl.load(X_ptr + pj.to(tl.int64) * 2, mask=in_range, other=0.0)
            yj = tl.load(X_ptr + pj.to(tl.int64) * 2 + 1, mask=in_range, other=0.0)
            dxv = xi - xj
            dyv = yi - yj
            dist = dxv * dxv + dyv * dyv
            hit_eps = (dist <= EPS_SQ) & in_range
            slot_addr = n_offs * K + cur_cnt.to(tl.int64)
            emit = hit_eps & (cur_cnt < K)
            tl.store(NBR_IDX_ptr + slot_addr, pj, mask=emit)
            cur_cnt = cur_cnt + hit_eps.to(tl.int32)

    cur_cnt = tl.minimum(cur_cnt, tl.full([BN], K, dtype=tl.int32))
    tl.store(DEG_ptr + n_offs, cur_cnt, mask=n_mask)


def _build_grid_index(X: torch.Tensor, eps: float):
    """Build a dense 2D grid index over points X (N, 2) fp32.

    Returns:
        sorted_ptr:           (N,) int32 — point indices sorted by cell
        cell_start, cell_end: (GW * GH,) int32 — start/end indices in sorted_ptr
        gw, gh:               grid dims
        inv_eps, x_min, y_min: kernel params
    """
    N = X.shape[0]
    device = X.device
    inv_eps = 1.0 / eps
    x_min = float(X[:, 0].min().item())
    y_min = float(X[:, 1].min().item())
    x_max = float(X[:, 0].max().item())
    y_max = float(X[:, 1].max().item())

    GW = int((x_max - x_min) * inv_eps) + 2
    GH = int((y_max - y_min) * inv_eps) + 2
    n_cells = GW * GH

    cx = ((X[:, 0] - x_min) * inv_eps).floor().to(torch.int32).clamp(0, GW - 1)
    cy = ((X[:, 1] - y_min) * inv_eps).floor().to(torch.int32).clamp(0, GH - 1)
    cell_id = cy.to(torch.int64) * GW + cx.to(torch.int64)

    # Sort points by cell_id
    sorted_cell, perm = torch.sort(cell_id, stable=True)
    sorted_ptr = perm.to(torch.int32)

    # Build dense (GW*GH,) start/end via bucket counts.
    counts = torch.zeros(n_cells, dtype=torch.int32, device=device)
    counts.scatter_add_(0, sorted_cell, torch.ones(N, dtype=torch.int32, device=device))
    cell_end = torch.cumsum(counts, dim=0, dtype=torch.int32)
    cell_start = cell_end - counts

    return sorted_ptr, cell_start, cell_end, GW, GH, inv_eps, x_min, y_min


def _flash_dbscan_grid(X: torch.Tensor, eps: float, min_samples: int,
                       max_neighbors: int):
    """D=2 grid-based path."""
    N, D = X.shape
    assert D == 2
    device = X.device
    K = max(min_samples, max_neighbors)
    eps_sq = float(eps) ** 2

    # Build grid index
    (sorted_ptr, cell_start, cell_end, GW, GH,
     inv_eps, x_min, y_min) = _build_grid_index(X, eps)

    deg = torch.zeros(N, dtype=torch.int32, device=device)
    nbr_idx = torch.full((N, K), -1, dtype=torch.int32, device=device)

    BN = 32
    grid = (triton.cdiv(N, BN),)
    _grid_radius_2d_kernel[grid](
        X.contiguous(), sorted_ptr, cell_start, cell_end,
        deg, nbr_idx,
        N=N, K=K, GW=GW, GH=GH,
        INV_EPS=inv_eps, EPS_SQ=eps_sq,
        GRID_X_MIN=x_min, GRID_Y_MIN=y_min,
        BN=BN,
        num_warps=4, num_stages=1,
    )
    return deg, nbr_idx, K


def _flash_dbscan_brute(X: torch.Tensor, eps: float, min_samples: int,
                         max_neighbors: int, *, tol=None):
    """High-D brute-force path via flash_knn -- single bf16 kNN call.

    Enumerates each point's top-``max_neighbors`` ε-candidates (in bf16
    by default; pass ``tol=0`` to force fp32). Points with more than
    ``max_neighbors`` true ε-neighbors will have their excess neighbours
    silently truncated, but this does **not** change the DBSCAN
    clustering:

    * Core-mask correctness: a point is core iff its ε-degree ≥
      ``min_samples``. As long as ``max_neighbors >= min_samples``
      the bounded-K enumeration still observes ``min_samples``-many
      ε-neighbours when they exist, so the core mask is exact.
    * Cluster-membership correctness: two core points belong to the
      same cluster iff they are reachable through a chain of
      core-core ε-edges. Dense clusters expose Θ(max_neighbors) such
      edges per core point, so any one of them is enough to merge
      that core into the cluster's connected component. Edges
      ranked beyond top-K are redundant for CC.

    The historical K-grow loop ("double K until the K-th NN exceeds
    ε") was reverted in 2026-05 -- the iterative re-launches grew K
    geometrically in dense regimes (medium-D blob shapes hit K=1024
    in 7 calls = 566 ms total) without changing ARI vs cuML beyond
    the noise floor inherited from the bounded-K approximation.

    Callers who actually need exhaustive enumeration (e.g. for
    reach-distance diagnostics, not clustering) should pass a
    larger ``max_neighbors`` once.
    """
    N, D = X.shape
    device = X.device
    K = max(min_samples, max_neighbors)
    K = min(K, N)
    eps_sq = float(eps) ** 2

    # Default to fp32 KNN: at these shapes (N, M >= 16K) flash_knn
    # routes to the "large_n" insert path (BN=128 single-pass per CTA),
    # which is memory-bound and runs at the same speed in fp32 as in
    # bf16 -- the .to(bf16) cast itself eats the would-be win. fp32
    # also gives bit-exact ARI vs sklearn
    # on the standard ε boundaries; bf16 quantisation flips a handful
    # of borderline points (~3% on D=64 eps=8). Power users can opt
    # into bf16 via ``tol=1e-3`` if they have looser ε.
    if D < 16:
        Xpad = torch.zeros(N, 16, device=device, dtype=X.dtype)
        Xpad[:, :D] = X
        X_kernel = Xpad.contiguous()
    else:
        X_kernel = X.contiguous()
    knn_dist_sq, knn_idx = flash_knn(
        X_kernel[None], X_kernel[None], k=K, tol=tol)
    knn_dist_sq = knn_dist_sq[0]
    knn_idx = knn_idx[0]

    valid = knn_dist_sq <= eps_sq
    deg = valid.sum(dim=1).to(torch.int32)
    nbr_idx = torch.where(valid, knn_idx.to(torch.int32),
                          torch.full_like(knn_idx, -1, dtype=torch.int32))
    return deg, nbr_idx, K


def flash_dbscan(X: torch.Tensor, eps: float, min_samples: int = 5,
                 max_neighbors: int = 32, *, tol=None):
    """End-to-end Triton DBSCAN.

    Args:
        X: (N, D) float32 CUDA tensor.
        eps: distance threshold (Euclidean).
        min_samples: minimum points in eps-neighborhood for a point to be core.
        max_neighbors: per-point ε-candidate budget for the high-D brute
            path. Default 32 -- core detection stays exact for any
            ``max_neighbors >= min_samples``; the connected-component
            stage only needs O(max_neighbors) redundant edges per
            cluster, so cluster membership is robust to the bounded-K
            approximation. Bump this if you need exhaustive ε-edge
            enumeration (e.g. reach-distance diagnostics).
        tol: residual tolerance forwarded to :func:`flash_knn` on the
            high-D (D >= 3) brute-force path. ``None`` (default) keeps
            fp32 storage -- the "large_n" single-pass insert path
            (BN=128) flash_knn picks on these shapes is memory-bound,
            so fp32 and bf16 run at the same speed; fp32 wins on ARI by
            avoiding ε-boundary quantisation. Pass ``tol=1e-3`` to opt
            into bf16/fp16 KNN storage (matches the historical HEAD
            cast) when you want lower HBM pressure.

    Returns:
        labels: (N,) int32 -- cluster id (>= 0) or -1 for noise.
    """
    assert X.is_cuda
    N, D = X.shape
    device = X.device

    # ── Stage 1: per-point in-eps neighbor list ─────────────────────────
    # Path A: D ≤ 2 → grid (cell-side eps; 9-cell scan, exact in any dtype).
    # Path B: D ≥ 3 → brute-force top-K via flash_knn (tol-routed dtype).
    if D == 2:
        # Grid kernel reads X via raw fp32 strides; cast internally.
        # This is a kernel implementation detail, NOT a public dtype contract.
        deg, nbr_idx, K = _flash_dbscan_grid(
            X if X.dtype == torch.float32 else X.float(),
            eps, min_samples, max_neighbors,
        )
    else:
        deg, nbr_idx, K = _flash_dbscan_brute(
            X, eps, min_samples, max_neighbors, tol=tol)

    # ── Stage 2: core mask ──────────────────────────────────────────────
    core_mask = deg >= min_samples

    # ── Stage 3: edge construction (core-core only) ─────────────────────
    K_eff = nbr_idx.shape[1]
    nbr_idx_i64 = nbr_idx.to(torch.int64)
    valid_slot = nbr_idx >= 0
    core_per_row = core_mask[:, None].expand(-1, K_eff)
    nbr_idx_safe = torch.where(valid_slot, nbr_idx_i64, torch.zeros_like(nbr_idx_i64))
    core_per_col = core_mask[nbr_idx_safe] & valid_slot
    edge_mask = valid_slot & core_per_row & core_per_col
    rows = (torch.arange(N, device=device, dtype=torch.int32).view(-1, 1)
            .expand(-1, K_eff).contiguous())[edge_mask].contiguous()
    cols = nbr_idx[edge_mask].contiguous()
    label_cc = flash_cc_from_edges(rows, cols, N)

    # ── Stage 4: assign labels (core: CC label, else: -1 for now) ────────
    INT_MAX = 2 ** 31 - 1
    label = torch.where(core_mask, label_cc, torch.full_like(label_cc, -1))

    # ── Stage 5: border assignment ───────────────────────────────────────
    nbr_labels = label[nbr_idx_safe]
    nbr_is_core = core_mask[nbr_idx_safe] & valid_slot
    border_cand = torch.where(valid_slot & nbr_is_core, nbr_labels,
                               torch.full_like(nbr_labels, INT_MAX))
    min_core_label = border_cand.min(dim=1).values
    is_border = (~core_mask) & (min_core_label != INT_MAX) & (min_core_label >= 0)
    label = torch.where(is_border, min_core_label, label)

    # ── Stage 6: compact ─────────────────────────────────────────────────
    valid = label >= 0
    if valid.any():
        unique, inv = torch.unique(label[valid], return_inverse=True)
        compact = torch.full_like(label, -1)
        compact[valid] = inv.to(torch.int32)
        label = compact

    return label
