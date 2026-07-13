"""Flash t-SNE Q-sum + gradient kernels (no P blocking).

Pass 1: Q-sum reduction — sum of all Q_num = 1/(1+||yi-yj||^2)
Pass 2: Gradient accumulation — reads P, computes grad in registers
"""
import torch
import triton
import triton.language as tl

from flashlib.kernels.distance.triton._common import _round_to_bucket


# ============================================================================
# t-SNE Q-sum / gradient / blocked-gradient kernels migrated from kernels/common/.
# ============================================================================

# =============================================================================
# Kernel 4: Flash t-SNE gradient (never materialize N^2 Q matrix)
#
# Pass 1: Q-sum reduction — sum of all Q_num = 1/(1+||yi-yj||^2)
# Pass 2: Gradient accumulation — reads P, computes grad in registers
#
# Autotuned: N is regular param, N_KEY is constexpr for autotuner + loop bound.
# Uses tl.range with num_stages for software pipelining of Y/P loads.
# =============================================================================

_TSNE_QSUM_CONFIGS = [
    triton.Config({"BLOCK_I": 32, "BLOCK_J": 32}, num_warps=4),
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 64}, num_warps=4),
    triton.Config({"BLOCK_I": 32, "BLOCK_J": 64}, num_warps=4),
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 32}, num_warps=4),
    triton.Config({"BLOCK_I": 128, "BLOCK_J": 32}, num_warps=4),
]

_TSNE_GRAD_CONFIGS = [
    triton.Config({"BLOCK_I": 32, "BLOCK_J": 32}, num_warps=4),
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 64}, num_warps=4),
    triton.Config({"BLOCK_I": 32, "BLOCK_J": 64}, num_warps=4),
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 32}, num_warps=4),
    triton.Config({"BLOCK_I": 128, "BLOCK_J": 32}, num_warps=4),
]


@triton.autotune(configs=_TSNE_QSUM_CONFIGS, key=["N_KEY"], reset_to_zero=["QSUM_ptr"])
@triton.jit
def _tsne_qsum_kernel(
    Y_ptr,          # (N, 2) embedding
    QSUM_ptr,       # (1,) output: sum of all q_ij numerators
    N,              # regular param — actual count
    stride_yn,
    N_KEY: tl.constexpr,   # power-of-2 bucket for autotuner key stability
    N_LOOP: tl.constexpr,  # tight upper bound for inner loop (ceil_to_BLOCK_J)
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Compute sum of Q numerators exploiting symmetry: Q[i,j] = Q[j,i].

    Only computes upper triangle (i < j), then doubles the result.
    J-blocks entirely below the diagonal are skipped via conditional.
    """
    pid_i = tl.program_id(0)
    i_offs = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_offs < N
    i_min = pid_i * BLOCK_I  # smallest i index in this block

    # Load Y[i] - 2D embedding
    yi_0 = tl.load(Y_ptr + i_offs * stride_yn + 0, mask=i_mask, other=0.0)
    yi_1 = tl.load(Y_ptr + i_offs * stride_yn + 1, mask=i_mask, other=0.0)

    local_sum = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for j_start in tl.range(0, N_LOOP, BLOCK_J):
        # Skip j-blocks entirely below diagonal: max(j_offs) < min(i_offs)
        # j_start + BLOCK_J - 1 < i_min → entire j-block is below diagonal
        if j_start + BLOCK_J > i_min:
            j_offs = j_start + tl.arange(0, BLOCK_J)
            j_mask = j_offs < N

            yj_0 = tl.load(Y_ptr + j_offs * stride_yn + 0, mask=j_mask, other=0.0)
            yj_1 = tl.load(Y_ptr + j_offs * stride_yn + 1, mask=j_mask, other=0.0)

            # ||yi - yj||^2
            d0 = yi_0[:, None] - yj_0[None, :]
            d1 = yi_1[:, None] - yj_1[None, :]
            dist_sq = d0 * d0 + d1 * d1

            q_num = 1.0 / (1.0 + dist_sq)

            # Upper triangle only: i < j (no diagonal)
            upper = i_offs[:, None] < j_offs[None, :]
            q_num = tl.where(i_mask[:, None] & j_mask[None, :] & upper, q_num, 0.0)

            local_sum += tl.sum(q_num, axis=1)

    total = tl.sum(local_sum)
    # Multiply by 2 for symmetry: Q[i,j] = Q[j,i]
    tl.atomic_add(QSUM_ptr, 2.0 * total)


@triton.autotune(configs=_TSNE_GRAD_CONFIGS, key=["N_KEY"])
@triton.jit
def _tsne_grad_kernel(
    Y_ptr,          # (N, 2) embedding
    P_ptr,          # (N, N) P matrix (symmetric, precomputed)
    QSUM_ptr,       # (1,) total Q-sum
    GRAD_ptr,       # (N, 2) output gradient
    N,              # regular param — actual count
    stride_yn,
    stride_pn, stride_pm,
    stride_gn,
    N_KEY: tl.constexpr,   # power-of-2 bucket for autotuner key stability
    N_LOOP: tl.constexpr,  # tight upper bound for inner loop (ceil_to_BLOCK_J)
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """Compute t-SNE gradient: grad_i = 4 * sum_j (p_ij - q_ij) * q_num_ij * (y_i - y_j)."""
    pid_i = tl.program_id(0)
    i_offs = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_offs < N

    yi_0 = tl.load(Y_ptr + i_offs * stride_yn + 0, mask=i_mask, other=0.0)
    yi_1 = tl.load(Y_ptr + i_offs * stride_yn + 1, mask=i_mask, other=0.0)

    q_sum = tl.load(QSUM_ptr)

    grad_0 = tl.zeros((BLOCK_I,), dtype=tl.float32)
    grad_1 = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for j_start in tl.range(0, N_LOOP, BLOCK_J):
        j_offs = j_start + tl.arange(0, BLOCK_J)
        j_mask = j_offs < N

        yj_0 = tl.load(Y_ptr + j_offs * stride_yn + 0, mask=j_mask, other=0.0)
        yj_1 = tl.load(Y_ptr + j_offs * stride_yn + 1, mask=j_mask, other=0.0)

        d0 = yi_0[:, None] - yj_0[None, :]
        d1 = yi_1[:, None] - yj_1[None, :]
        dist_sq = d0 * d0 + d1 * d1

        q_num = 1.0 / (1.0 + dist_sq)

        # Load P[i, j]
        p_ptrs = P_ptr + i_offs[:, None] * stride_pn + j_offs[None, :] * stride_pm
        p_ij = tl.load(p_ptrs, mask=i_mask[:, None] & j_mask[None, :], other=0.0)

        # q_ij = q_num / q_sum
        q_ij = q_num / q_sum

        # Mask diagonal
        same = (i_offs[:, None] == j_offs[None, :])
        pq_diff = tl.where(same, 0.0, p_ij - q_ij)

        # grad contribution: 4 * (p - q) * q_num * (yi - yj)
        coeff = 4.0 * pq_diff * q_num
        coeff = tl.where(i_mask[:, None] & j_mask[None, :], coeff, 0.0)

        grad_0 += tl.sum(coeff * d0, axis=1)
        grad_1 += tl.sum(coeff * d1, axis=1)

    tl.store(GRAD_ptr + i_offs * stride_gn + 0, grad_0, mask=i_mask)
    tl.store(GRAD_ptr + i_offs * stride_gn + 1, grad_1, mask=i_mask)


def _ceil_to_block(N, max_block=128):
    """Round N up to nearest multiple of max_block for tight loop bound."""
    return ((N + max_block - 1) // max_block) * max_block


def triton_tsne_qsum(Y: torch.Tensor) -> torch.Tensor:
    """Compute sum of Q numerators without materializing N^2 matrix."""
    N = Y.shape[0]
    Y = Y.contiguous()
    qsum = torch.zeros(1, device=Y.device, dtype=torch.float32)
    N_KEY = _round_to_bucket(N)
    N_LOOP = _ceil_to_block(N, 128)  # tight bound: max BLOCK_J is 64 or 128
    grid = lambda META: (triton.cdiv(N, META["BLOCK_I"]),)
    _tsne_qsum_kernel[grid](
        Y, qsum, N, Y.stride(0),
        N_KEY=N_KEY, N_LOOP=N_LOOP,
    )
    return qsum


def triton_tsne_grad(Y: torch.Tensor, P: torch.Tensor, qsum: torch.Tensor) -> torch.Tensor:
    """Compute t-SNE gradient without materializing N^2 Q matrix."""
    N = Y.shape[0]
    Y = Y.contiguous()
    P = P.contiguous()
    grad = torch.zeros(N, 2, device=Y.device, dtype=torch.float32)
    N_KEY = _round_to_bucket(N)
    N_LOOP = _ceil_to_block(N, 128)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_I"]),)
    _tsne_grad_kernel[grid](
        Y, P, qsum, grad, N,
        Y.stride(0), P.stride(0), P.stride(1), grad.stride(0),
        N_KEY=N_KEY, N_LOOP=N_LOOP,
    )
    return grad
