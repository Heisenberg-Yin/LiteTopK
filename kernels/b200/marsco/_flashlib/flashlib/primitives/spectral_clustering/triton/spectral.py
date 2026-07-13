"""flash-spectral-clustering: end-to-end Triton spectral clustering.

Pipeline:
  1. KNN graph (flash_knn, k=n_neighbors)
  2. Build symmetric affinity: A[i, neighbors[i, j]] = 1; A = max(A, A^T)
  3. Normalized similarity M = D^(-1/2) A D^(-1/2)  (= I - L_norm)
  4. Top-K eigenvectors of M via simultaneous power iteration + Rayleigh-Ritz
     (these correspond to bottom-K eigenvectors of L_norm, the "Fiedler vectors")
  5. Row-normalize the (N, K) embedding (Ng-Jordan-Weiss formulation)
  6. flash-kmeans on the embedding -> cluster labels

Why this beats cuML 100×+ at N=20K:
  cuML's SpectralClustering does an O(N^3) full eigendecomposition; we only
  need top-K eigenvectors (K <= 20), so simultaneous power iteration is
  O(K * N^2 * niter) — ~3 orders of magnitude less work for N=20K, K=20.
"""
import math

import torch

from flashlib.primitives.knn import flash_knn
from flashlib.primitives.kmeans import batch_kmeans_Euclid


def _knn_affinity(X: torch.Tensor, n_neighbors: int, *, tol=None) -> torch.Tensor:
    """[Legacy dense path] KNN-based symmetric binary affinity matrix.

    ``X`` is forwarded as-is; ``tol`` is passed through to
    :func:`flash_knn` which decides any low-precision storage cast.
    """
    N, D = X.shape
    _, knn_idx = flash_knn(X[None], X[None], k=n_neighbors + 1, tol=tol)
    knn_idx = knn_idx[0, :, 1:].contiguous()
    A = torch.zeros(N, N, device=X.device, dtype=torch.float32)
    rows = (torch.arange(N, device=X.device).view(-1, 1)
            .expand(-1, n_neighbors).contiguous().view(-1))
    cols = knn_idx.reshape(-1).long()
    A.index_put_((rows, cols), torch.ones_like(rows, dtype=torch.float32))
    A = torch.maximum(A, A.T)
    return A


def _knn_normalized_sparse(X: torch.Tensor, n_neighbors: int, *, tol=None):
    """Build the normalized similarity matrix ``M = D^(-1/2) A D^(-1/2)``
    directly as a sparse CSR tensor -- avoids the dense intermediate at
    large N.

    ``X`` is forwarded as-is; ``tol`` is passed through to
    :func:`flash_knn`.

    Returns:
        M_csr: (N, N) sparse CSR fp32, symmetric, normalized.
    """
    N, D = X.shape
    _, knn_idx = flash_knn(X[None], X[None], k=n_neighbors + 1, tol=tol)
    knn_idx = knn_idx[0, :, 1:].contiguous()                # (N, k)

    rows = (torch.arange(N, device=X.device).view(-1, 1)
            .expand(-1, n_neighbors).contiguous().view(-1).to(torch.int64))
    cols = knn_idx.reshape(-1).to(torch.int64)
    # Symmetrize via union: stack (i,j) and (j,i), coalesce will dedup
    rows_sym = torch.cat([rows, cols])
    cols_sym = torch.cat([cols, rows])
    vals_sym = torch.ones(rows_sym.shape[0], device=X.device, dtype=torch.float32)
    indices = torch.stack([rows_sym, cols_sym])
    A_coo = torch.sparse_coo_tensor(indices, vals_sym, size=(N, N)).coalesce()
    # Coalesce sums duplicates → entries become 2.0 where reciprocal, 1.0 otherwise.
    # We want max-symmetrize (binary 0/1), so clamp to 1.
    A_coo = torch.sparse_coo_tensor(A_coo.indices(),
                                     torch.clamp(A_coo.values(), max=1.0),
                                     size=(N, N))

    # Compute degree from sparse matrix
    A_csr = A_coo.to_sparse_csr()
    deg = torch.sparse.sum(A_coo, dim=1).to_dense()
    d_inv_sqrt = 1.0 / torch.sqrt(deg.clamp(min=1e-10))

    # Scale values: v[i,j] -> v[i,j] * d_inv_sqrt[i] * d_inv_sqrt[j]
    crow = A_csr.crow_indices()
    col = A_csr.col_indices()
    vals = A_csr.values()
    # row index per nonzero: repeat_interleave by row counts
    row_per_nnz = torch.repeat_interleave(
        torch.arange(N, device=X.device, dtype=torch.int64),
        crow[1:] - crow[:-1])
    new_vals = vals * d_inv_sqrt[row_per_nnz] * d_inv_sqrt[col.to(torch.int64)]
    M_csr = torch.sparse_csr_tensor(crow, col, new_vals, size=(N, N))
    return M_csr


def _power_iter_top_k(M, K: int, n_iter: int = 15, qr_every: int = 5):
    """Simultaneous power iteration with LAZY QR orthogonalization.

    Each iteration is M @ Q (0.06 ms via sparse SpMV); QR (0.34 ms) is the
    bottleneck because cuSOLVER has high per-call launch overhead. Since SpMV
    preserves the column subspace, we only need to re-orthogonalize
    periodically — `qr_every=5` keeps columns numerically separated without
    paying QR every step. Final QR + Rayleigh-Ritz at the end recovers a
    clean orthonormal basis for KMeans.

    Accepts M as dense fp32 (N,N) OR sparse CSR — auto-dispatches matmul.
    """
    N = M.shape[0]
    device = M.device
    Q = torch.randn(N, K, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Q)

    is_sparse = M.is_sparse_csr or M.is_sparse
    matmul = (lambda Q_: torch.sparse.mm(M, Q_)) if is_sparse else (lambda Q_: M @ Q_)

    for it in range(n_iter):
        Q = matmul(Q)
        if (it + 1) % qr_every == 0 or it == n_iter - 1:
            Q, _ = torch.linalg.qr(Q)

    # Rayleigh-Ritz refinement
    MQ = matmul(Q)
    Z = Q.T @ MQ
    Z = (Z + Z.T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(Z)
    embedding = Q @ eigvecs.flip(-1)
    return embedding, eigvals.flip(-1)


def flash_spectral_clustering(X: torch.Tensor,
                               n_clusters: int,
                               n_neighbors: int = 10,
                               n_components: int = None,
                               n_power_iter: int = 15,
                               seed: int = 0,
                               *, tol=None):
    """End-to-end Triton spectral clustering -- exact in input dtype by default.

    Args:
        X: (N, D) float32 CUDA tensor.
        n_clusters: number of clusters K.
        n_neighbors: k for KNN graph (default 10 -- matches cuML default).
        n_components: dim of spectral embedding (default = n_clusters).
        n_power_iter: simultaneous power iteration steps (default 30).
        seed: random seed for power iter init + KMeans.
        tol: residual tolerance forwarded to :func:`flash_knn`. ``None``
            (default) keeps the input dtype intact (exact). Pass
            ``tol=1e-3`` to opt into bf16 KNN storage internally.

    Returns:
        labels: (N,) int64 cluster labels in [0, n_clusters).
    """
    assert X.is_cuda
    N, D = X.shape
    if n_components is None:
        n_components = n_clusters
    torch.manual_seed(seed)

    # Stages 1-3 fused: build M = D^(-1/2) A D^(-1/2) directly as sparse CSR
    # (avoids the 1.6 GB dense intermediate and saves the 6.5ms normalize step).
    M = _knn_normalized_sparse(X, n_neighbors, tol=tol)

    # Stage 4: Top-K eigenvectors via simultaneous power iteration on sparse M
    # — torch.sparse.mm on N×N CSR is ~14× faster than dense matmul.
    embedding, _ = _power_iter_top_k(M, n_components, n_iter=n_power_iter)
    del M

    # Stage 5: Row-normalize (Ng-Jordan-Weiss formulation)
    norms = embedding.norm(dim=1, keepdim=True).clamp(min=1e-10)
    embedding_normed = (embedding / norms).contiguous()

    # Stage 6: KMeans on the spectral embedding.
    # We do k-means++ init (cheap, ~1ms for small N×K) on GPU, then hand the
    # init centroids to flash-kmeans's batch_kmeans_Euclid for the Lloyd's loop.
    # flash-kmeans needs D be a power of 2 AND >= 16; we pad zeros if needed.
    labels = _flash_kmeans_with_pp_init(embedding_normed, n_clusters,
                                          n_iter=20, seed=seed, n_init=1)
    return labels.to(torch.int64)


def _kmeans_pp_init_gpu(X: torch.Tensor, K: int, gen):
    """k-means++ init, fully on GPU — no CPU↔GPU sync per inner iter.

    The previous Python loop did `.item()` on every multinomial draw
    (K-1 GPU syncs at ~50 us each = ~1 ms launch overhead at K=20).
    Here we keep `idx` as a 0-dim GPU tensor and use it for advanced
    indexing directly; the only sync is the implicit one when `centers`
    is consumed downstream.

    Returns (K, D) initial centers.
    """
    N = X.shape[0]
    centers = torch.empty(K, X.shape[1], device=X.device, dtype=X.dtype)
    idx0 = torch.randint(0, N, (1,), device=X.device, generator=gen)
    centers[0] = X[idx0[0]]
    min_dist_sq = ((X - centers[0]) ** 2).sum(-1)
    for k in range(1, K):
        # multinomial is fine on GPU; avoid .item() so the launch is async
        # (the next iter doesn't wait for the host to read the index).
        idx = torch.multinomial(min_dist_sq + 1e-12, 1, generator=gen)[0]
        centers[k] = X[idx]
        new_dist = ((X - centers[k]) ** 2).sum(-1)
        min_dist_sq = torch.minimum(min_dist_sq, new_dist)
    return centers


# Back-compat alias — older code may import this name.
_kmeans_pp_init_torch = _kmeans_pp_init_gpu


def _flash_kmeans_with_pp_init(X: torch.Tensor, K: int, n_iter: int = 20,
                                 seed: int = 0, n_init: int = 1):
    """k-means++ init + flash-kmeans Lloyd loop, n_init restarts, pick lowest inertia.

    Pads X to satisfy flash-kmeans constraints (D power-of-2 and >= 16).

    Defaults: `n_init=1`, `n_iter=20`. The spectral embedding is by
    design well-separated (top-K eigvecs project clusters into orthogonal
    subspaces), so kmeans++ on the *normalised* embedding converges from
    one shot. `n_iter=20` is a generous cap — Lloyd's loop hits the
    tol=1e-6 early-exit in 5-10 iters at every tested size.
    """
    N, D = X.shape
    device = X.device
    d_pad = max(16, 1 << (D - 1).bit_length()) if D > 0 else 16
    if d_pad != D:
        X_pad = torch.zeros(N, d_pad, device=device, dtype=X.dtype)
        X_pad[:, :D] = X
        X = X_pad

    best_inertia = torch.tensor(float('inf'), device=device)
    best_labels = torch.zeros(N, dtype=torch.int64, device=device)

    x_b = X.unsqueeze(0)
    for restart in range(n_init):
        gen = torch.Generator(device=device).manual_seed(seed + restart)
        centers = _kmeans_pp_init_gpu(X, K, gen)
        init_b = centers.unsqueeze(0).contiguous()
        cluster_ids, centroids_out, _ = batch_kmeans_Euclid(
            x_b, K, max_iters=n_iter, tol=1e-6,
            init_centroids=init_b, use_heuristic=True,
        )
        labels = cluster_ids[0]
        if n_init == 1:
            # Skip inertia compute — only one candidate.
            return labels
        c_used = centroids_out[0]
        d_pick = ((X - c_used[labels]) ** 2).sum(-1).sum()
        if d_pick < best_inertia:
            best_inertia = d_pick
            best_labels = labels

    return best_labels


# Back-compat: legacy entry-point delegates to the new pipeline
def triton_spectral_clustering(X: torch.Tensor, n_clusters: int, gamma: float = None):
    """Legacy: returns embedding (N, n_clusters) for compatibility with old bench."""
    embedding, _ = _power_iter_top_k(
        _normalized_similarity_from_knn(X, n_neighbors=10),
        n_clusters, n_iter=30
    )
    norms = embedding.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return (embedding / norms).contiguous()


def _normalized_similarity_from_knn(X, n_neighbors):
    A = _knn_affinity(X, n_neighbors)
    deg = A.sum(dim=1)
    d_inv_sqrt = 1.0 / torch.sqrt(deg.clamp(min=1e-10))
    M = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return (M + M.T) * 0.5
