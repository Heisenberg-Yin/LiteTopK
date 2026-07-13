"""Forward LayerNorm / RMSNorm tuned for the small normalized-dim regime.

PyTorch's native (eager) LayerNorm/RMSNorm assign **one CTA per normalized
row**. When the normalized dimension is small -- e.g. per-head QK-norm over
``head_dim`` (64/128) with a very large number of rows -- each CTA does a
tiny reduction and the SMs are badly under-utilized (measured at ~8-17% of
HBM peak on H200 for ``head_dim`` in {64, 128}).

The fix is to process **multiple rows per CTA** (a ``(BLOCK_M, N)`` tile):
the contiguous tile load is coalesced, each CTA carries enough work to
saturate memory bandwidth, and launch/setup is amortized. On the same
shapes this recovers ~84-88% of HBM peak -- a 5-10x forward speedup over
eager, and on par with a hand-tuned single-row kernel at large N.

Provenance
----------
This multi-row-per-CTA normalization kernel was first introduced in
**Sparse VideoGen** (SVG), our ICML 2025 work on accelerating video
diffusion transformers, where per-head normalization over a small
``head_dim`` sits on the critical path::

    Sparse VideoGen: Accelerating Video Diffusion Transformers with
    Spatial-Temporal Sparsity. ICML 2025. arXiv:2502.01776.
    https://github.com/svg-project/Sparse-VideoGen
    (svg/kernels/triton/{rmsnorm,layernorm}.py)

The SVG implementation (first public early 2025) predates the equivalent
small-N normalization codegen later added to ``torch.compile`` (PyTorch
2.11, 2026); this module ports that original forward kernel into flashlib.

Forward-only by design (the SVG inference path): mean/rstd are not saved
for a backward pass. For training-grade fused norms (backward included),
``torch.compile`` now generates near-SOTA kernels.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X, Y, W,
    x_stride, y_stride,
    M,
    N: tl.constexpr, N2: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
):
    """One program per ``BLOCK_M`` rows; full row (padded to ``N2``) per tile."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(X + rows[:, None] * x_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = (x * rstd[:, None]) * w[None, :]
    tl.store(Y + rows[:, None] * y_stride + cols[None, :],
             y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _layernorm_fwd_kernel(
    X, Y, W, B,
    x_stride, y_stride,
    M,
    N: tl.constexpr, N2: tl.constexpr,
    eps,
    HAS_W: tl.constexpr, HAS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Mean-centered variant; ``W``/``B`` optional (``elementwise_affine``)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(X + rows[:, None] * x_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    # Zero the padded columns so they add nothing to the variance.
    xc = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xc * rstd[:, None]
    if HAS_W:
        w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = y * w[None, :]
    if HAS_B:
        b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = y + b[None, :]
    tl.store(Y + rows[:, None] * y_stride + cols[None, :],
             y.to(Y.dtype.element_ty), mask=mask)


def _launch_cfg(n2: int) -> tuple[int, int]:
    """``(BLOCK_M, num_warps)`` heuristic -- pack more rows the smaller ``N``.

    Calibrated on H200/bf16: small normalized dims want many rows per CTA to
    saturate HBM; once a single row already fills the warps (large ``N``),
    one row per CTA is best.
    """
    if n2 <= 512:
        return 16, 4
    if n2 <= 2048:
        return 4, 8
    return 1, 8


def _flatten(x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
    if x.ndim < 1:
        raise ValueError("norm input must have at least 1 dimension")
    D = x.shape[-1]
    return x.contiguous().reshape(-1, D), x.shape


def flash_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dim: ``y = x / sqrt(mean(x^2) + eps) * weight``.

    Args:
        x: ``(..., D)`` CUDA tensor (any floating dtype; reduced in fp32).
        weight: ``(D,)`` scale, broadcast over the leading dims.
        eps: numerical floor inside the rsqrt.

    Returns:
        ``y`` with the same shape and dtype as ``x``.
    """
    if not x.is_cuda:
        raise ValueError("flash_rmsnorm requires a CUDA tensor")
    x2, shape = _flatten(x)
    M, D = x2.shape
    if weight.shape[-1] != D:
        raise ValueError(f"weight dim {weight.shape[-1]} != x last dim {D}")
    w = weight.contiguous()
    y = torch.empty_like(x2)
    N2 = triton.next_power_of_2(D)
    BLOCK_M, num_warps = _launch_cfg(N2)
    _rmsnorm_fwd_kernel[(triton.cdiv(M, BLOCK_M),)](
        x2, y, w, x2.stride(0), y.stride(0), M, D, N2, eps,
        BLOCK_M=BLOCK_M, num_warps=num_warps,
    )
    return y.reshape(shape)


def flash_layernorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm over the last dim with optional affine ``weight``/``bias``.

    Args:
        x: ``(..., D)`` CUDA tensor (any floating dtype; reduced in fp32).
        weight: optional ``(D,)`` scale (``elementwise_affine``).
        bias: optional ``(D,)`` shift (LayerNorm only).
        eps: numerical floor inside the rsqrt.

    Returns:
        ``y`` with the same shape and dtype as ``x``.
    """
    if not x.is_cuda:
        raise ValueError("flash_layernorm requires a CUDA tensor")
    x2, shape = _flatten(x)
    M, D = x2.shape
    w = weight.contiguous() if weight is not None else None
    b = bias.contiguous() if bias is not None else None
    if w is not None and w.shape[-1] != D:
        raise ValueError(f"weight dim {w.shape[-1]} != x last dim {D}")
    if b is not None and b.shape[-1] != D:
        raise ValueError(f"bias dim {b.shape[-1]} != x last dim {D}")
    y = torch.empty_like(x2)
    N2 = triton.next_power_of_2(D)
    BLOCK_M, num_warps = _launch_cfg(N2)
    _layernorm_fwd_kernel[(triton.cdiv(M, BLOCK_M),)](
        x2, y, w, b, x2.stride(0), y.stride(0), M, D, N2, eps,
        HAS_W=w is not None, HAS_B=b is not None,
        BLOCK_M=BLOCK_M, num_warps=num_warps,
    )
    return y.reshape(shape)


__all__ = ["flash_rmsnorm", "flash_layernorm"]
