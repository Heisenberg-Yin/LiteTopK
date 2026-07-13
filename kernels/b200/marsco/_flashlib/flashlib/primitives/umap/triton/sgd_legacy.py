"""UMAP legacy random-edge SGD kernel (kept for back-compat).

Used by the deprecated ``triton_umap_sgd_step`` API; the production
path uses ``sgd_step.py``'s flash_umap_sgd_step / _v2.
"""
import math
import numpy as np
import torch
import triton
import triton.language as tl



# =============================================================================
# Kernel 5: UMAP SGD kernel — sorted-edge SGD with fused attractive/repulsive
# =============================================================================

@triton.jit
def _umap_sgd_kernel(
    EMB_ptr,        # (N, 2) embedding
    HEAD_ptr,       # (E,) edge head indices
    TAIL_ptr,       # (E,) edge tail indices
    NEG_ptr,        # (E,) negative sample indices
    EPOCHS_PER_ptr, # (E,) epochs_per_sample
    EPOCH: tl.constexpr,
    LR,             # scalar learning rate
    N: tl.constexpr,
    E: tl.constexpr,
    stride_en,
    A: tl.constexpr,  # a parameter (default 1.0)
    B: tl.constexpr,  # b parameter (default 1.0)
    BLOCK_E: tl.constexpr,
):
    """Process BLOCK_E edges: attractive + repulsive forces."""
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    # Load edge indices
    head = tl.load(HEAD_ptr + e_offs, mask=e_mask, other=0)
    tail = tl.load(TAIL_ptr + e_offs, mask=e_mask, other=0)
    neg = tl.load(NEG_ptr + e_offs, mask=e_mask, other=0)

    # Check epoch sampling
    eps = tl.load(EPOCHS_PER_ptr + e_offs, mask=e_mask, other=1e10)
    should_run = (EPOCH % tl.maximum(eps.to(tl.int32), 1) == 0)
    e_mask = e_mask & should_run

    # Load embeddings
    h0 = tl.load(EMB_ptr + head * stride_en + 0, mask=e_mask, other=0.0)
    h1 = tl.load(EMB_ptr + head * stride_en + 1, mask=e_mask, other=0.0)
    t0 = tl.load(EMB_ptr + tail * stride_en + 0, mask=e_mask, other=0.0)
    t1 = tl.load(EMB_ptr + tail * stride_en + 1, mask=e_mask, other=0.0)
    n0 = tl.load(EMB_ptr + neg * stride_en + 0, mask=e_mask, other=0.0)
    n1 = tl.load(EMB_ptr + neg * stride_en + 1, mask=e_mask, other=0.0)

    # Attractive force: pull head toward tail
    d0_attr = h0 - t0
    d1_attr = h1 - t1
    dist_sq_attr = d0_attr * d0_attr + d1_attr * d1_attr + 1e-6
    # grad_coeff = -2ab * dist^(2b-2) / (1 + a*dist^2b)
    w_attr = -2.0 * A * B / (dist_sq_attr * (A * dist_sq_attr + 1.0))
    grad_h0_attr = w_attr * d0_attr
    grad_h1_attr = w_attr * d1_attr

    # Repulsive force: push head away from neg
    d0_rep = h0 - n0
    d1_rep = h1 - n1
    dist_sq_rep = d0_rep * d0_rep + d1_rep * d1_rep + 1e-6
    # grad_coeff = 2b / ((0.001 + dist_sq) * (1 + a*dist^2b))
    w_rep = 2.0 * B / (dist_sq_rep * (A * dist_sq_rep + 1.0))
    grad_h0_rep = w_rep * d0_rep
    grad_h1_rep = w_rep * d1_rep

    # Combined gradient: attractive + repulsive
    total_g0 = grad_h0_attr + grad_h0_rep
    total_g1 = grad_h1_attr + grad_h1_rep
    total_g0 = tl.maximum(tl.minimum(total_g0, 4.0), -4.0)
    total_g1 = tl.maximum(tl.minimum(total_g1, 4.0), -4.0)

    # Update head embedding
    new_h0 = h0 + LR * total_g0
    new_h1 = h1 + LR * total_g1
    tl.atomic_add(EMB_ptr + head * stride_en + 0, LR * total_g0, mask=e_mask)
    tl.atomic_add(EMB_ptr + head * stride_en + 1, LR * total_g1, mask=e_mask)


def triton_umap_sgd_step(emb, head, tail, neg, epochs_per_sample, epoch, lr,
                         a=1.0, b=1.0, BLOCK_E=256):
    """One epoch of UMAP SGD update."""
    E = head.shape[0]
    N = emb.shape[0]
    grid = (triton.cdiv(E, BLOCK_E),)
    _umap_sgd_kernel[grid](
        emb, head, tail, neg, epochs_per_sample,
        epoch, lr,
        N, E,
        emb.stride(0),
        a, b,
        BLOCK_E=BLOCK_E,
    )
