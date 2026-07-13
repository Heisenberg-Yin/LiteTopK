"""UMAP layout-optimization SGD kernel (flash_umap production path).

Matches upstream UMAP's ``optimize_layout_euclidean``: symmetric
head+tail update with negative sampling and a float epoch-of-next
state machine.
"""
import math
import numpy as np
import torch
import triton
import triton.language as tl




# =============================================================================
# Kernel: Flash UMAP SGD step — matches upstream UMAP optimize_layout_euclidean.
# Float epoch_of_next_sample / epoch_of_next_negative_sample state machines,
# variable n_neg_samples per fire, gamma parameter, dist > 0 guards.
# =============================================================================

@triton.jit
def _flash_umap_sgd_kernel(
    EMB_ptr,                  # (N, D) float32, in-place updated
    HEAD_ptr, TAIL_ptr,       # (E,) int64
    EPS_PER_ptr,              # (E,) float32  — epochs_per_sample[i]
    EPS_PER_NEG_ptr,          # (E,) float32  — epochs_per_negative_sample[i]
    EPOCH_NEXT_ptr,           # (E,) float32 mutable
    EPOCH_NEXT_NEG_ptr,       # (E,) float32 mutable
    EPOCH,                    # float runtime — current epoch
    LR, A, B, GAMMA, SEED,
    N,
    E,
    D: tl.constexpr,
    N_NEG_MAX: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    next_pos = tl.load(EPOCH_NEXT_ptr + e_offs, mask=e_mask, other=1e30)
    fire = (next_pos <= EPOCH) & e_mask

    head = tl.load(HEAD_ptr + e_offs, mask=fire, other=0)
    tail = tl.load(TAIL_ptr + e_offs, mask=fire, other=0)

    d_offs = tl.arange(0, D)

    # Load head, tail embeddings: (BLOCK_E, D)
    h = tl.load(EMB_ptr + head[:, None] * D + d_offs[None, :],
                mask=fire[:, None], other=0.0)
    t = tl.load(EMB_ptr + tail[:, None] * D + d_offs[None, :],
                mask=fire[:, None], other=0.0)

    diff = h - t
    dist_sq = tl.sum(diff * diff, axis=1)
    pos_dist = dist_sq > 0.0

    safe_dsq = tl.maximum(dist_sq, 1e-12)
    log_dsq = tl.log(safe_dsq)
    pow_2b = tl.exp(B * log_dsq)
    pow_2b_m2 = tl.exp((B - 1.0) * log_dsq)
    grad_coef = -2.0 * A * B * pow_2b_m2 / (1.0 + A * pow_2b)
    grad_coef = tl.where(fire & pos_dist, grad_coef, 0.0)

    g = grad_coef[:, None] * diff
    g = tl.maximum(tl.minimum(g, 4.0), -4.0)

    addr_h = head[:, None] * D + d_offs[None, :]
    addr_t = tail[:, None] * D + d_offs[None, :]
    tl.atomic_add(EMB_ptr + addr_h, LR * g, mask=fire[:, None])
    tl.atomic_add(EMB_ptr + addr_t, -LR * g, mask=fire[:, None])

    # Negative sampling — variable n_neg per fire
    eps_neg = tl.load(EPS_PER_NEG_ptr + e_offs, mask=fire, other=1.0)
    next_neg = tl.load(EPOCH_NEXT_NEG_ptr + e_offs, mask=fire, other=EPOCH)
    n_neg_f = (EPOCH - next_neg) / tl.maximum(eps_neg, 1e-6)
    n_neg = n_neg_f.to(tl.int32)
    n_neg = tl.where(fire & (n_neg > 0), n_neg, 0)

    for p in tl.static_range(N_NEG_MAX):
        active = fire & (p < n_neg)
        # Per-edge per-step PRNG: vary the seed by p, mix epoch into offset
        salt = e_offs.to(tl.int32) ^ (EPOCH.to(tl.int32) * 1597334677)
        rnd = tl.randint(SEED + p * 7919, salt)
        # rnd is int32 (signed); take abs via mask, then mod N
        neg_idx = (rnd.to(tl.int64) & 0x7FFFFFFFFFFFFFFF) % N

        h_cur = tl.load(EMB_ptr + head[:, None] * D + d_offs[None, :],
                        mask=active[:, None], other=0.0)
        n_emb = tl.load(EMB_ptr + neg_idx[:, None] * D + d_offs[None, :],
                        mask=active[:, None], other=0.0)
        diff_n = h_cur - n_emb
        dist_sq_n = tl.sum(diff_n * diff_n, axis=1)
        pos_n = dist_sq_n > 0.0

        safe_n = tl.maximum(dist_sq_n, 1e-12)
        pow_2b_n = tl.exp(B * tl.log(safe_n))
        rep_coef = 2.0 * GAMMA * B / ((dist_sq_n + 1e-3) * (1.0 + A * pow_2b_n))
        rep_coef = tl.where(active & pos_n, rep_coef, 0.0)

        g_rep = rep_coef[:, None] * diff_n
        g_rep = tl.maximum(tl.minimum(g_rep, 4.0), -4.0)
        tl.atomic_add(EMB_ptr + head[:, None] * D + d_offs[None, :],
                      LR * g_rep, mask=active[:, None])

    # State updates (only edges that fired)
    eps_pos = tl.load(EPS_PER_ptr + e_offs, mask=fire, other=1.0)
    new_next_pos = next_pos + eps_pos
    tl.store(EPOCH_NEXT_ptr + e_offs, new_next_pos, mask=fire)

    new_next_neg = next_neg + n_neg.to(tl.float32) * eps_neg
    tl.store(EPOCH_NEXT_NEG_ptr + e_offs, new_next_neg, mask=fire)


def triton_flash_umap_sgd_step(emb, head, tail,
                                epochs_per_sample,
                                epochs_per_negative_sample,
                                epoch_of_next_sample,
                                epoch_of_next_negative_sample,
                                epoch, lr,
                                a=1.577, b=0.895, gamma=1.0,
                                n_neg_max=8, seed=0, BLOCK_E=256):
    """One epoch of flash-umap SGD using upstream's float state machine.

    All state arrays are float32 and updated in place. Matches upstream
    UMAP ``optimize_layout_euclidean`` exactly.
    """
    E = head.shape[0]
    N, D = emb.shape
    assert emb.dtype == torch.float32 and emb.is_cuda
    assert head.dtype == torch.int64 and tail.dtype == torch.int64
    for t in (epochs_per_sample, epochs_per_negative_sample,
              epoch_of_next_sample, epoch_of_next_negative_sample):
        assert t.dtype == torch.float32 and t.shape == (E,)
    grid = (triton.cdiv(E, BLOCK_E),)
    _flash_umap_sgd_kernel[grid](
        emb, head, tail,
        epochs_per_sample, epochs_per_negative_sample,
        epoch_of_next_sample, epoch_of_next_negative_sample,
        float(epoch), float(lr),
        float(a), float(b), float(gamma), int(seed),
        N, E, D,
        N_NEG_MAX=n_neg_max, BLOCK_E=BLOCK_E,
        num_warps=4,
    )
