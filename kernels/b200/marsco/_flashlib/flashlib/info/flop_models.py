"""FLOP and byte-transfer models for each algorithm.

Each function returns (flop_count, bytes_moved) where either can be None
if hard to model accurately.

bytes_moved assumes float32 (4 bytes) unless specified otherwise.
"""


def pca_flops(N, D, K):
    """PCA: covariance X.T @ X + eigh + transform X @ V_k"""
    cov_flops = 2 * N * D * D          # X.T @ X
    eigh_flops = int((8 / 3) * D ** 3) # eigendecomposition
    transform_flops = 2 * N * D * K     # X @ V[:, :K]
    total = cov_flops + eigh_flops + transform_flops

    # Bytes: read X (N*D*4), write cov (D*D*4), read/write eigh (D*D*4 * few),
    # read X again + V (N*D*4 + D*K*4), write result (N*K*4)
    bytes_moved = (N * D + D * D + N * D + D * K + N * K) * 4
    return total, bytes_moved


def pca_cov_flops(N, D):
    """PCA covariance step only: X.T @ X"""
    flops = 2 * N * D * D
    bytes_moved = (N * D + D * D) * 4
    return flops, bytes_moved


def pca_eigh_flops(D):
    """PCA eigendecomposition step only."""
    flops = int((8 / 3) * D ** 3)
    bytes_moved = D * D * 4 * 10  # iterative reads/writes
    return flops, bytes_moved


def pca_lanczos_flops(N, D, K, block_size=None, m_steps=None):
    """Block Lanczos PCA: gram GEMM + Lanczos iterations + projection.

    Used when D >> N (dual-space approach).
    """
    B = block_size or min(2 * K, N, 64)
    m = m_steps or (K // B + 5)

    # Gram: X @ X.T, symmetric → ~N*N*D FLOPs (half of 2*N*N*D)
    gram_flops = N * N * D

    # Lanczos: m iterations of G @ Q_j (2*N*N*B) + reorth O(m^2*N*B^2)
    lanczos_flops = m * 2 * N * N * B
    reorth_flops = sum(2 * j * N * B * B for j in range(1, m + 1))

    # Tridiag eigh: (m*B)^3 * 8/3
    tridiag_flops = int((8 / 3) * (m * B) ** 3)

    # Projection: X.T @ U_sel = 2*N*D*K
    proj_flops = 2 * N * D * K

    total = gram_flops + lanczos_flops + reorth_flops + tridiag_flops + proj_flops

    # Bytes: X read twice (gram + projection) + G (N*N*4) + Q_basis + output
    x_bytes = 2 * N * D * 4
    gram_bytes = N * N * 4
    q_bytes = m * N * B * 4
    output_bytes = (D * K + K) * 4
    total_bytes = x_bytes + gram_bytes + q_bytes + output_bytes

    return total, total_bytes


def pca_flops_auto(N, D, K, **kwargs):
    """Auto-select PCA FLOP model based on N/D ratio."""
    if N >= 4 * D:
        return pca_flops(N, D, K)
    else:
        return pca_lanczos_flops(N, D, K, **kwargs)


def truncated_svd_flops(N, D, K, oversampling=10, n_power_iter=2):
    """Randomized SVD: X @ Omega, QR, B = Q.T @ X, SVD(B)"""
    p = K + oversampling
    proj_flops = 2 * N * D * p                     # X @ Omega
    qr_flops = 2 * N * p * p                        # QR of (N, p)
    power_flops = n_power_iter * 2 * (2 * N * D * p) # power iteration
    bt_flops = 2 * p * N * D                         # Q.T @ X -> (p, D)
    small_svd_flops = 2 * p * p * D                  # SVD of (p, D)
    total = proj_flops + qr_flops + power_flops + bt_flops + small_svd_flops

    bytes_moved = (N * D + D * p + N * p + p * D + N * K) * 4
    return total, bytes_moved


def svd_cov_flops(N, D, K):
    """Exact SVD via covariance: X.T @ X + eigh (N >> D path)."""
    gram_flops = 2 * N * D * D
    eigh_flops = int((8 / 3) * D ** 3)
    total = gram_flops + eigh_flops
    bytes_moved = (N * D + D * D) * 4
    return total, bytes_moved


def svd_dual_flops(N, D, K):
    """Exact SVD via dual Gram: X @ X.T + eigh + projection (D >> N path)."""
    gram_flops = N * N * D       # symmetric: ~half of 2*N*N*D
    eigh_flops = int((8 / 3) * N ** 3)
    proj_flops = 2 * N * D * K   # X.T @ U_K
    total = gram_flops + eigh_flops + proj_flops
    bytes_moved = (2 * N * D + N * N + D * K) * 4
    return total, bytes_moved


def svd_flops_auto(N, D, K, **kwargs):
    """Auto-select SVD FLOP model based on N/D ratio."""
    if N >= 4 * D:
        return svd_cov_flops(N, D, K)
    else:
        return svd_dual_flops(N, D, K)


def linear_regression_flops(N, D):
    """Normal equations: X.T @ X + X.T @ y + Cholesky solve"""
    xtx_flops = 2 * N * D * D          # X.T @ X (GEMM)
    xty_flops = 2 * N * D              # X.T @ y (GEMV)
    chol_flops = int(D ** 3 / 3)       # Cholesky factorization
    solve_flops = D * D                 # triangular solve
    total = xtx_flops + xty_flops + chol_flops + solve_flops

    bytes_moved = (N * D + D * D + D + D) * 4
    return total, bytes_moved


def ridge_regression_flops(N, D):
    """Same as linear regression (regularization adds O(D))."""
    return linear_regression_flops(N, D)


def logistic_regression_flops(N, D, n_iter):
    """Per-iteration: forward X @ w (N*D) + backward X.T @ g (N*D) = 4*N*D per iter"""
    flops_per_iter = 4 * N * D
    total = n_iter * flops_per_iter
    bytes_moved = n_iter * (N * D + D + N) * 4
    return total, bytes_moved


def pairwise_distance_flops(N, D, dtype_bytes=4):
    """Pairwise Euclidean distance: cdist(X, X)"""
    flops = 2 * N * N * D + N * N  # matmul + sqrt
    bytes_moved = N * D * dtype_bytes + N * N * dtype_bytes  # read X, write dists
    return flops, bytes_moved


def dbscan_flops(N, D):
    """DBSCAN dominated by pairwise distances."""
    return pairwise_distance_flops(N, D)


def hdbscan_flops(N, D):
    """HDBSCAN: pairwise distances + mutual reachability."""
    dist_flops, dist_bytes = pairwise_distance_flops(N, D)
    mrd_flops = 3 * N * N  # max operations for mutual reachability
    mrd_bytes = N * N * 4 * 2  # read dists + write mrd
    return dist_flops + mrd_flops, dist_bytes + mrd_bytes


def svc_kernel_flops(N, D):
    """SVC RBF kernel computation."""
    dist_flops = 2 * N * N * D  # pairwise distances
    kernel_flops = 3 * N * N    # exp(-gamma * d^2): sub, mul, exp
    total = dist_flops + kernel_flops
    bytes_moved = (N * D + N * N) * 4
    return total, bytes_moved


def tsne_flops(N, D_out, n_iter):
    """TSNE exact: per iteration, pairwise distances in embedding space + gradient."""
    per_iter = 2 * N * N * D_out + N * N * 10  # distances + gradient ops
    total = n_iter * per_iter
    bytes_moved = n_iter * (N * D_out + N * N) * 4
    return total, bytes_moved


def spectral_clustering_flops(N, D, K):
    """Spectral: pairwise distances + affinity + eigh."""
    dist_flops = 2 * N * N * D
    affinity_flops = 3 * N * N  # exp(-d^2/(2*sigma^2))
    laplacian_flops = 3 * N * N  # normalization
    eigh_flops = int((8 / 3) * N ** 3)
    total = dist_flops + affinity_flops + laplacian_flops + eigh_flops
    bytes_moved = (N * D + N * N * 3 + N * K) * 4
    return total, bytes_moved


def kernel_density_flops(N_train, N_query, D, dtype_bytes=4):
    """KDE score_samples: pairwise (N_query × N_train) distances + log-sum-exp.

    FLOPs:
      distances: 2 * N_query * N_train * D (Gram-style expansion)
      reductions / exp / log: 5 * N_query * N_train (exp/max/sub/add)

    Bytes (algorithmic lower bound for a fully fused streaming kernel):
      X_train (N_train × D fp32) read once: N_train * D * 4
      X_query (N_query × D fp32) read once: N_query * D * 4
      output (N_query fp32) written once:    N_query * 4
    BW% computed against this lower bound is the honest "fraction of
    theoretical-minimum bytes that the achieved time corresponds to."
    Real HBM traffic exceeds this by a factor of N_query/BLOCK_Q on the
    X_train term when the train tile is too large for L2 — this is
    captured as <100% BW achievement in the bench.
    """
    dist_flops = 2 * N_query * N_train * D
    reduce_flops = 5 * N_query * N_train  # exp/sub/max/add per pair
    flops = dist_flops + reduce_flops
    bytes_moved = (N_train * D + N_query * D) * dtype_bytes + N_query * 4
    return flops, bytes_moved


def random_forest_flops(N, D, n_trees, max_depth):
    """Approximate: n_trees * N * log(N) * D * 5 (histogram + split finding)."""
    import math
    flops = int(n_trees * N * math.log2(max(N, 2)) * D * 5)
    bytes_moved = int(n_trees * N * D * 4)  # rough: read data per tree
    return flops, bytes_moved


def gaussian_random_projection_flops(N, D, K):
    """Gaussian RP: X (N,D) @ R (D,K) = X_proj (N,K). Compute-bound dense GEMM."""
    flops = 2 * N * D * K
    # Read X once (N*D*4), read R (D*K*4), write Y (N*K*4)
    bytes_moved = (N * D + D * K + N * K) * 4
    return flops, bytes_moved


def sparse_random_projection_flops(N, D, K, density=None):
    """Sparse RP: X @ R where R has density `density` nonzeros (default 1/sqrt(D)).

    Effective FLOPs scaled by density (only nonzero elements contribute).
    Uses dense FLOP count (2*N*D*K) for fairness in cross-impl comparisons —
    the speedup from sparsity is apparent in time, not flop fraction.
    """
    import math
    if density is None:
        density = 1.0 / math.sqrt(D)
    # Dense-equivalent FLOPs (the work *would* be 2NDK if R were dense)
    dense_flops = 2 * N * D * K
    # Effective FLOPs: only nonzero entries do useful work
    effective_flops = int(2 * N * D * K * density)
    # Bytes: X read once + R sparse (nnz of R is D*K*density, each entry = 8 bytes idx+val)
    nnz = int(D * K * density)
    bytes_moved = N * D * 4 + nnz * 8 + N * K * 4
    return dense_flops, bytes_moved, effective_flops
