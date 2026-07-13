"""flash-dbscan: GPU-resident DBSCAN via flash-knn radius filter (legacy).

This is the pre-enhanced implementation, kept for parity testing. The
production path lives in :mod:`flashlib.primitives.dbscan.triton.dbscan`
and uses the faster ``flash_cc_from_edges`` from
``flashlib.kernels.flash_mst`` (BLOCK=128 edges per program with 4
warps; iterative converging Boruvka with merge counter). This legacy
path uses the slower 1-edge-per-program union-find from
``flashlib.kernels.connected_components``.

Pipeline:
  1. flash_knn (tol-routed dtype) — top-K neighbors per point, K = max(min_samples, 20)
  2. Filter to distances ≤ eps → sparse eps-adjacency
  3. core_mask: |eps-neighbors(i)| >= min_samples (including self)
  4. Connected components on core-induced subgraph (Triton union-find)
  5. Border points: any non-core adjacent to a core gets that core's cluster

The K cap means we miss neighbors past the K-th nearest. For typical DBSCAN
parameters (eps chosen so each point has < ~min_samples · 4 neighbors), K=20
captures the full eps-neighborhood for nearly all points.
"""
import torch

from flashlib.primitives.knn import flash_knn
from flashlib.kernels.connected_components import flash_cc_from_edges


def flash_dbscan_legacy(X: torch.Tensor, eps: float, min_samples: int = 5,
                         max_neighbors: int = 32, *, tol=None):
    """Legacy end-to-end Triton DBSCAN.

    Args:
        X: (N, D) float32 CUDA tensor.
        eps: distance threshold (Euclidean).
        min_samples: minimum points in eps-neighborhood for a point to be core.
        max_neighbors: per-point ε-candidate budget (default 32). See
            :mod:`flashlib.primitives.dbscan.triton.dbscan` for why the
            bounded-K enumeration is correctness-preserving for the
            core-mask + CC pipeline.
        tol: residual tolerance forwarded to :func:`flash_knn`. ``None``
            (default) keeps fp32 -- bit-exact at the ε boundary and
            equally fast at these K (memory-bound v1 path). Pass
            ``tol=1e-3`` to opt into bf16/fp16 KNN storage if you
            want lower HBM pressure on very wide D.

    Returns:
        labels: (N,) int32 -- cluster id (>= 0) or -1 for noise.
    """
    assert X.is_cuda
    N, D = X.shape
    device = X.device
    eps_sq = float(eps) ** 2

    K = max(min_samples, max_neighbors)
    K = min(K, N)
    # Single fp32 kNN call by default. See
    # :mod:`flashlib.primitives.dbscan.triton.dbscan` for the rationale
    # (bounded-K is correctness-preserving for the core+CC pipeline,
    # and fp32 is free at these K because v1's build kernel is
    # memory-bound).
    X_in = X.contiguous()
    knn_dist_sq, knn_idx = flash_knn(X_in[None], X_in[None], k=K, tol=tol)
    knn_dist_sq = knn_dist_sq[0]
    knn_idx = knn_idx[0].to(torch.int64)

    valid = knn_dist_sq <= eps_sq
    deg = valid.sum(dim=1)
    core_mask = deg >= min_samples

    core_per_row = core_mask[:, None].expand(-1, K)
    core_per_col = core_mask[knn_idx]
    edge_mask = valid & core_per_row & core_per_col
    rows = (torch.arange(N, device=device, dtype=torch.int32).view(-1, 1)
            .expand(-1, K).contiguous())[edge_mask].contiguous()
    cols = knn_idx.to(torch.int32)[edge_mask].contiguous()
    label_cc = flash_cc_from_edges(rows, cols, N)

    INT_MAX = 2 ** 31 - 1
    label = torch.where(core_mask, label_cc, torch.full_like(label_cc, -1))

    nbr_labels = label[knn_idx]
    nbr_is_core = core_mask[knn_idx]
    border_cand = torch.where(valid & nbr_is_core, nbr_labels,
                              torch.full_like(nbr_labels, INT_MAX))
    min_core_label = border_cand.min(dim=1).values
    is_border = (~core_mask) & (min_core_label != INT_MAX) & (min_core_label >= 0)
    label = torch.where(is_border, min_core_label, label)

    valid = label >= 0
    if valid.any():
        unique, inv = torch.unique(label[valid], return_inverse=True)
        compact = torch.full_like(label, -1)
        compact[valid] = inv.to(torch.int32)
        label = compact

    return label
