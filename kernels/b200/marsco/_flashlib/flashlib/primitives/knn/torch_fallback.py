"""PyTorch reference implementations for KNN (correctness baselines)."""

import torch


def knn_torch_naive(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Brute-force KNN via full distance matrix materialization.

    Args:
        x: (B, N, D) query points
        c: (B, M, D) database points
        k: number of nearest neighbors

    Returns:
        (vals, idxs): each (B, N, K) — squared L2 distances and indices
    """
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x, c>
    x_f = x.float()
    c_f = c.float()
    x_sq = (x_f ** 2).sum(-1, keepdim=True)       # (B, N, 1)
    c_sq = (c_f ** 2).sum(-1, keepdim=True)       # (B, M, 1)
    cross = torch.bmm(x_f, c_f.transpose(1, 2))   # (B, N, M)
    dist = x_sq + c_sq.transpose(1, 2) - 2.0 * cross  # (B, N, M)
    dist.clamp_min_(0.0)
    vals, idxs = dist.topk(k, dim=-1, largest=False, sorted=True)
    return vals, idxs


def knn_torch_chunked(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked KNN — iterates over database in chunks, maintains online topK.

    Args:
        x: (B, N, D) query points
        c: (B, M, D) database points
        k: number of nearest neighbors
        chunk_size: number of database points per chunk

    Returns:
        (vals, idxs): each (B, N, K) — squared L2 distances and indices
    """
    B, N, D = x.shape
    M = c.shape[1]
    x_f = x.float()
    c_f = c.float()
    x_sq = (x_f ** 2).sum(-1, keepdim=True)  # (B, N, 1)

    topk_vals = torch.full((B, N, k), float('inf'), device=x.device, dtype=torch.float32)
    topk_idxs = torch.full((B, N, k), -1, device=x.device, dtype=torch.int64)

    for m_start in range(0, M, chunk_size):
        m_end = min(m_start + chunk_size, M)
        c_chunk = c_f[:, m_start:m_end, :]             # (B, chunk, D)
        c_sq_chunk = (c_chunk ** 2).sum(-1, keepdim=True)  # (B, chunk, 1)
        cross = torch.bmm(x_f, c_chunk.transpose(1, 2))   # (B, N, chunk)
        dist = x_sq + c_sq_chunk.transpose(1, 2) - 2.0 * cross
        dist.clamp_min_(0.0)

        # Combine with current topK
        combined_vals = torch.cat([topk_vals, dist], dim=-1)     # (B, N, K + chunk)
        combined_idxs = torch.cat([
            topk_idxs,
            torch.arange(m_start, m_end, device=x.device).view(1, 1, -1).expand(B, N, -1),
        ], dim=-1)

        # Take top-K
        _, sel = combined_vals.topk(k, dim=-1, largest=False, sorted=True)
        topk_vals = combined_vals.gather(-1, sel)
        topk_idxs = combined_idxs.gather(-1, sel)

    return topk_vals, topk_idxs
