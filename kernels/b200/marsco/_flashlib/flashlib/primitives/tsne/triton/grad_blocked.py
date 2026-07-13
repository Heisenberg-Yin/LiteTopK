"""t-SNE gradient with blocked-P layout (coalesced HBM reads).

Reorders P into BLK×BLK tiles in row-major; the gradient kernel
then reads P with one big coalesced load per tile.
"""
import torch
import triton
import triton.language as tl



# =============================================================================
# Kernel 4b: t-SNE gradient with blocked P layout — coalesced HBM reads
# =============================================================================

def block_p_matrix(P: torch.Tensor, BLK: int = 64) -> torch.Tensor:
    """Convert P from (N, N) to tile-blocked layout (NB, NB, BLK, BLK).

    Each (BLK, BLK) tile is contiguous in memory, eliminating strided reads
    that limit bandwidth to ~18% on standard layout.
    """
    N = P.shape[0]
    NB = math.ceil(N / BLK)
    N_padded = NB * BLK
    if N_padded != N:
        P_padded = torch.zeros(N_padded, N_padded, device=P.device, dtype=P.dtype)
        P_padded[:N, :N] = P
    else:
        P_padded = P
    # Reshape: (NB, BLK, NB, BLK) -> permute to (NB, NB, BLK, BLK) -> contiguous
    P_blocked = P_padded.view(NB, BLK, NB, BLK).permute(0, 2, 1, 3).contiguous()
    return P_blocked


@triton.jit
def _tsne_grad_blocked_kernel(
    Y_ptr,          # (N, 2) embedding
    P_BLK_ptr,      # (NB, NB, BLK, BLK) blocked P matrix
    QSUM_ptr,       # (1,) total Q-sum
    GRAD_ptr,       # (N, 2) output gradient
    N,
    stride_yn,
    stride_gn,
    NB_KEY: tl.constexpr,
    BLK: tl.constexpr,
):
    """t-SNE gradient with blocked P layout for coalesced memory access."""
    pid_i = tl.program_id(0)
    i_offs = pid_i * BLK + tl.arange(0, BLK)
    i_mask = i_offs < N

    yi_0 = tl.load(Y_ptr + i_offs * stride_yn + 0, mask=i_mask, other=0.0)
    yi_1 = tl.load(Y_ptr + i_offs * stride_yn + 1, mask=i_mask, other=0.0)

    q_sum = tl.load(QSUM_ptr)

    grad_0 = tl.zeros((BLK,), dtype=tl.float32)
    grad_1 = tl.zeros((BLK,), dtype=tl.float32)

    # Base pointer for P_blocked[pid_i, :, :, :] — all j-blocks for this i-block
    tile_size: tl.constexpr = BLK * BLK
    p_row_base = P_BLK_ptr + pid_i * NB_KEY * tile_size

    # Local offsets within a BLK×BLK tile (contiguous!)
    local_offs = tl.arange(0, BLK)[:, None] * BLK + tl.arange(0, BLK)[None, :]

    for j_block in tl.range(0, NB_KEY):
        j_offs = j_block * BLK + tl.arange(0, BLK)
        j_mask = j_offs < N

        yj_0 = tl.load(Y_ptr + j_offs * stride_yn + 0, mask=j_mask, other=0.0)
        yj_1 = tl.load(Y_ptr + j_offs * stride_yn + 1, mask=j_mask, other=0.0)

        d0 = yi_0[:, None] - yj_0[None, :]
        d1 = yi_1[:, None] - yj_1[None, :]
        dist_sq = d0 * d0 + d1 * d1

        q_num = 1.0 / (1.0 + dist_sq)

        # Load P tile — CONTIGUOUS 16KB read (vs 64 strided 256B reads before)
        p_tile_ptr = p_row_base + j_block * tile_size
        p_ij = tl.load(p_tile_ptr + local_offs, mask=i_mask[:, None] & j_mask[None, :], other=0.0)

        q_ij = q_num / q_sum

        same = (i_offs[:, None] == j_offs[None, :])
        pq_diff = tl.where(same, 0.0, p_ij - q_ij)

        coeff = 4.0 * pq_diff * q_num
        coeff = tl.where(i_mask[:, None] & j_mask[None, :], coeff, 0.0)

        grad_0 += tl.sum(coeff * d0, axis=1)
        grad_1 += tl.sum(coeff * d1, axis=1)

    tl.store(GRAD_ptr + i_offs * stride_gn + 0, grad_0, mask=i_mask)
    tl.store(GRAD_ptr + i_offs * stride_gn + 1, grad_1, mask=i_mask)


def triton_tsne_grad_blocked(Y: torch.Tensor, P_blocked: torch.Tensor,
                              qsum: torch.Tensor, N: int, BLK: int = 64) -> torch.Tensor:
    """Compute t-SNE gradient using blocked P layout for coalesced reads."""
    Y = Y.contiguous()
    NB = P_blocked.shape[0]
    grad = torch.zeros(N, 2, device=Y.device, dtype=torch.float32)
    grid = (NB,)
    _tsne_grad_blocked_kernel[grid](
        Y, P_blocked, qsum, grad, N,
        Y.stride(0), grad.stride(0),
        NB_KEY=NB, BLK=BLK,
    )
    return grad

