"""flash-umap: end-to-end UMAP using flash_knn + Triton kernels.

Pipeline:
  1. KNN graph via flash_knn
  2. smooth_knn_dist (Triton) -> sigma, rho per row
  3. Membership weights p_ij = exp(-(d_ij - rho_i) / sigma_i)
  4. Symmetrize: q = p + p^T - p * p^T   (cupy CSR)
  5. Random init embedding
  6. SGD layout optimization (Triton)
"""

import math

import torch

from flashlib.primitives.knn import flash_knn
from flashlib.primitives.umap.triton.smooth_knn_dist import triton_smooth_knn_dist
from flashlib.primitives.umap.triton.sgd_step import (
    triton_flash_umap_sgd_step,
)
from flashlib.primitives.umap.triton.fuzzy_simplicial_set import (
    triton_umap_fuzzy_simplicial_set,
)



# Default curve params (a, b) for spread=1.0, min_dist=0.1.
# Computed once via umap.umap_.find_ab_params(1.0, 0.1).
_DEFAULT_A = 1.5769434602697652
_DEFAULT_B = 0.8950608778515733


def _knn_graph(X: torch.Tensor, n_neighbors: int, *, tol=None):
    """Stage 1: build symmetric KNN graph using :func:`flash_knn`.

    ``tol`` is forwarded to ``flash_knn`` as-is; ``tol=None`` (default)
    keeps the input dtype intact (exact). UMAP's smooth-knn bisect
    absorbs ~1e-3 distance noise, so passing ``tol=1e-3`` is a safe
    speed lever for the user but never the library's default.

    Returns:
        nbr_indices: (N, k) int64 -- neighbor indices excluding self.
        nbr_dists:   (N, k) float32 -- Euclidean distances (sqrt of squared).
    """
    N, D = X.shape
    k = n_neighbors + 1  # +1 to drop self
    dists_sq, indices = flash_knn(X[None], X[None], k=k, tol=tol)
    # shapes: (1, N, k)
    dists_sq = dists_sq[0]
    indices = indices[0]
    # Drop self (column 0): self-distance is 0; flash_knn returns it first
    dists_sq = dists_sq[:, 1:].contiguous()
    indices = indices[:, 1:].contiguous()
    dists = torch.sqrt(dists_sq.clamp(min=0.0))
    return indices.to(torch.int64), dists


def _fuzzy_simplicial_set(nbr_indices: torch.Tensor, nbr_dists: torch.Tensor,
                          n_neighbors: int, fused: bool = True):
    """Stages 2-4: smooth_knn_dist -> per-row membership -> symmetrize.

    By default uses ``triton_umap_fuzzy_simplicial_set`` — single Triton
    kernel that fuses the smooth-knn bisect, membership, and symmetrize.
    Replaces the prior cupy CSR path (smooth_knn launch + 4-6 cupy COO/CSR
    launches) which was 3-9 ms / fixed launch overhead. The fused kernel
    runs in 0.5-5 ms over our N range — 1.7-8x faster on the fuzzy stage,
    which translates to 1.3-1.5x E2E flash_umap speedup.

    Pass ``fused=False`` to fall back to the cupy-sparse path (kept for
    reference/debug).

    Returns:
        head, tail: (E,) int64 — directed edges (each unordered pair appears
            once or twice depending on whether it is in the kNN graph in
            one or both directions). The downstream SGD updates BOTH head
            and tail under attractive force, so this is mathematically
            equivalent to the cuML symmetric COO graph (which emits (i,j)
            AND (j,i) separately).
        weights: (E,) float32 — symmetric edge weights in (0, 1].
    """
    if fused:
        return triton_umap_fuzzy_simplicial_set(
            nbr_indices, nbr_dists, n_iter=64, bandwidth=1.0, tol=1e-5,
            filter_eps=1e-9,
        )

    N, k = nbr_dists.shape
    # Stage 2: sigma, rho per row (smooth_knn includes self at column 0; we
    # already dropped self, so prepend a zero column to mimic upstream input)
    dists_with_self = torch.cat(
        [torch.zeros(N, 1, device=nbr_dists.device, dtype=nbr_dists.dtype),
         nbr_dists], dim=1
    ).contiguous()
    sigma, rho = triton_smooth_knn_dist(dists_with_self)

    # Stage 3: membership p_ij = exp(-(d_ij - rho_i) / sigma_i), clamp at 1
    diff = nbr_dists - rho[:, None]
    p = torch.where(diff > 0, torch.exp(-diff / sigma[:, None]),
                    torch.ones_like(diff))

    # Stage 4: symmetrize via cupy sparse (q = p + p^T - p * p^T)
    import cupy as cp
    import cupyx.scipy.sparse as csp

    rows = (
        torch.arange(N, device=nbr_indices.device)
        .view(-1, 1).expand(-1, k).contiguous().view(-1)
    )
    cols = nbr_indices.view(-1)
    vals = p.view(-1)

    rows_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(rows.to(torch.int32)))
    cols_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(cols.to(torch.int32)))
    vals_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(vals.contiguous()))

    P = csp.coo_matrix((vals_cp, (rows_cp, cols_cp)), shape=(N, N)).tocsr()
    Pt = P.T.tocsr()
    prod = P.multiply(Pt)
    Q = (P + Pt - prod).tocoo()

    # Filter out edges with weight < 1/n_epochs equiv: keep edges where weight > 0
    head_cp = Q.row.astype(cp.int64)
    tail_cp = Q.col.astype(cp.int64)
    w_cp = Q.data.astype(cp.float32)

    head = torch.from_dlpack(head_cp)
    tail = torch.from_dlpack(tail_cp)
    weights = torch.from_dlpack(w_cp)
    return head, tail, weights


def _make_epochs_per_sample(weights: torch.Tensor, n_epochs: int):
    """Match upstream UMAP make_epochs_per_sample exactly.

    Returns float32 epochs_per_sample[i]. Edges that would fire <1 times across
    all n_epochs are marked with a sentinel (n_epochs + 1) so they never fire.
    """
    w_max = weights.max()
    n_samples = n_epochs * weights / w_max
    # Upstream: result[n_samples > 0] = n_epochs / n_samples; rest = -1 sentinel
    eps_float = n_epochs / n_samples.clamp(min=1e-6)
    sentinel = float(n_epochs + 1)
    eps_float = torch.where(n_samples >= 1.0, eps_float,
                            torch.full_like(eps_float, sentinel))
    return eps_float.to(torch.float32)


def flash_umap(X: torch.Tensor, n_neighbors: int = 15, n_components: int = 2,
               n_epochs: int = 200, learning_rate: float = 1.0,
               spread: float = 1.0, min_dist: float = 0.1,
               n_neg_samples: int = 5, seed: int = 42,
               return_graph: bool = False, *, tol=None):
    """End-to-end Triton UMAP -- exact in input dtype by default.

    Returns ``Y`` (or ``(Y, head, tail, weights)`` with
    ``return_graph=True``) of shape (N, n_components).

    Args:
        X: (N, D) float32 CUDA tensor.
        n_neighbors: target k for KNN graph (default 15).
        n_components: embedding dim (default 2).
        n_epochs: SGD epochs (default 200).
        learning_rate: initial learning rate, decays linearly to 0.
        spread, min_dist: passed to find_ab_params (default a=1.577, b=0.895).
        n_neg_samples: negatives per positive (default 5).
        seed: RNG seed.
        return_graph: if True, also return ``(head, tail, weights)``.
        tol: residual tolerance forwarded to :func:`flash_knn`. ``None``
            (default) keeps ``X`` in its input dtype (exact). Pass
            ``tol=1e-3`` to opt into bf16 KNN storage internally.
    """
    assert X.is_cuda
    N, D = X.shape

    # 1. KNN graph
    nbr_idx, nbr_d = _knn_graph(X, n_neighbors, tol=tol)

    # 2-4. Fuzzy simplicial set
    head, tail, weights = _fuzzy_simplicial_set(nbr_idx, nbr_d, n_neighbors)

    # 5. Random init: uniform [-10, 10] matches upstream UMAP default for random init
    torch.manual_seed(seed)
    emb = (torch.rand(N, n_components, device=X.device, dtype=torch.float32) - 0.5) * 20.0

    # 6. SGD with real edges
    if spread == 1.0 and min_dist == 0.1:
        a, b = _DEFAULT_A, _DEFAULT_B
    else:
        from umap.umap_ import find_ab_params
        a, b = find_ab_params(spread, min_dist)

    eps_per = _make_epochs_per_sample(weights, n_epochs)
    eps_per_neg = eps_per / float(n_neg_samples)
    # Upstream init: epoch_of_next_sample = epochs_per_sample (clone), same for neg
    epoch_next = eps_per.clone()
    epoch_next_neg = eps_per_neg.clone()

    for epoch in range(n_epochs):
        lr = learning_rate * (1.0 - epoch / n_epochs)
        triton_flash_umap_sgd_step(
            emb, head, tail,
            eps_per, eps_per_neg,
            epoch_next, epoch_next_neg,
            epoch=float(epoch), lr=lr,
            a=a, b=b, gamma=1.0,
            n_neg_max=max(8, n_neg_samples + 3),
            seed=seed,
        )

    if return_graph:
        return emb, (head, tail, weights)
    return emb
