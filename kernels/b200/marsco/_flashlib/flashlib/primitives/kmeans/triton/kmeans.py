import torch
import torch.nn.functional as F
from torch.cuda import nvtx
from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton, cosine_assign_triton
from flashlib.primitives.kmeans.triton.update import (
    triton_centroid_update_cosine,
    triton_centroid_update_euclid,
    triton_centroid_update_sorted_euclid,
    triton_centroid_update_sorted_cosine,
    triton_lloyd_centroid_step_euclid,
)
from tqdm import trange

# -------------------- Compiled single-iteration kernels --------------------

# 1. Euclidean
def _euclid_iter(x, centroids, use_heuristic=True):

    cluster_ids = euclid_assign_triton(x, centroids, use_heuristic=use_heuristic)
    centroids_new = triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)

    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

# 2. Cosine
def _cosine_iter(x_norm, centroids):
    # cos_sim = torch.einsum('bnd,bkd->bnk', x_norm, centroids)
    # cluster_ids = cos_sim.argmax(dim=-1)
    cluster_ids = cosine_assign_triton(x_norm, centroids)
    centroids_new = triton_centroid_update_sorted_cosine(x_norm, cluster_ids, centroids)
    # centroids_new = centroids_new.clone()
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

# 3. Dot-product
def _dot_iter(x, centroids):
    # sim = torch.einsum('bnd,bkd->bnk', x, centroids)
    # cluster_ids = sim.argmax(dim=-1)
    cluster_ids = cosine_assign_triton(x, centroids)
    centroids_new = triton_centroid_update_sorted_cosine(x, cluster_ids, centroids)
    # centroids_new = centroids_new.clone()
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

COMPILE_FLAG = False

try:
    if COMPILE_FLAG:
        _euclid_iter_compiled = torch.compile(_euclid_iter, dynamic=True, mode="reduce-overhead")
        _cosine_iter_compiled = torch.compile(_cosine_iter, dynamic=True, mode="reduce-overhead")
        _dot_iter_compiled    = torch.compile(_dot_iter,    dynamic=True, mode="reduce-overhead")
    else:
        _euclid_iter_compiled = _euclid_iter
        _cosine_iter_compiled = _cosine_iter
        _dot_iter_compiled    = _dot_iter
except Exception:  # pragma: no cover
    _euclid_iter_compiled = _euclid_iter
    _cosine_iter_compiled = _cosine_iter
    _dot_iter_compiled    = _dot_iter

def batch_kmeans_Euclid(
    x,
    n_clusters,
    max_iters=100,
    tol=0.0,
    init_centroids=None,
    verbose=False,
    *,
    use_heuristic=True,
    fused=True,
):
    """
    Batched KMeans clustering in PyTorch using Euclidean distance.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement.
        verbose: Print loss for each iter.
        use_heuristic: Use heuristic Triton config (skip autotune).
        fused: If True (default), use the fused Lloyd path with preallocated
               sums/cnts/new/shift buffers and ping-pong centroid swap (no
               .clone() per iter). Falls back to per-iter alloc when False.
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
    """
    B, N, D = x.shape
    K = n_clusters

    if init_centroids is None:
        # Randomly select initial centers from x
        indices = torch.randint(0, N, (B, K), device=x.device)
        centroids = torch.gather(
            x,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        )  # (B, K, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, K, D).contiguous()

    if not fused:
        # ----- per-iter alloc + .clone() path -----
        for it in range(max_iters):
            centroids_new, center_shift, cluster_ids = _euclid_iter_compiled(
                x, centroids, use_heuristic
            )
            if verbose:
                print(f"Iter {it}, center shift: {center_shift.item():.6f}")
            if center_shift < tol:
                break
            centroids = centroids_new.clone()
        return cluster_ids, centroids, it + 1

    # ----- fused path: preallocated buffers + ping-pong centroid swap -----
    # Two centroid buffers swapped each iter so we never .clone().
    cent_a = centroids
    cent_b = torch.empty_like(centroids)

    sums_buf = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    cnts_buf = torch.zeros((B, K), device=x.device, dtype=torch.int32)
    shift_buf = torch.empty((B, K), device=x.device, dtype=torch.float32)

    cur, nxt = cent_a, cent_b
    cluster_ids = None
    it = 0
    for it in range(max_iters):
        cluster_ids = euclid_assign_triton(x, cur, use_heuristic=use_heuristic)
        # writes new centroids into `nxt`, returns scalar GPU tensor for shift
        new_cent, _, max_shift = triton_lloyd_centroid_step_euclid(
            x, cluster_ids, cur,
            sums_buf=sums_buf,
            cnts_buf=cnts_buf,
            new_buf=nxt,
            shift_buf=shift_buf,
        )
        if verbose:
            print(f"Iter {it}, center shift: {max_shift.item():.6f}")
        # swap before convergence check so `cur` always points to the latest
        cur, nxt = nxt, cur
        # Convergence check: `max_shift` is a 0-D GPU tensor, so
        # `if max_shift < tol` triggers `tensor.__bool__()` which forces
        # a per-iter cuda sync (~2.4 ms/iter on H200, drains the kernel
        # pipeline). The short-circuit on `tol > 0.0` keeps the default
        # `tol=0.0` path sync-free; users opting into early-exit accept
        # the sync as part of that contract.
        if tol > 0.0 and max_shift < tol:
            break

    return cluster_ids, cur, it + 1


def batch_kmeans_Cosine(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using Cosine similarity.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement.
        verbose: Print loss for each iter.
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
    """
    B, N, D = x.shape

    # Normalize input vectors for cosine similarity
    x_norm = F.normalize(x, p=2, dim=-1)  # (B, N, D)

    if init_centroids is None:
        # Randomly select initial centers from x_norm
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x_norm,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        ) # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)
    centroids = F.normalize(centroids, p=2, dim=-1)  # Ensure centroids are normalized

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _cosine_iter_compiled(x_norm, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1


def batch_kmeans_Dot(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using raw dot-product as similarity.

    """
    B, N, D = x.shape

    if init_centroids is None:
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        )
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        centroids_new, center_shift, cluster_ids = _dot_iter_compiled(x, centroids)

        if verbose:
            print(f"Iter {it} (dot), center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1
