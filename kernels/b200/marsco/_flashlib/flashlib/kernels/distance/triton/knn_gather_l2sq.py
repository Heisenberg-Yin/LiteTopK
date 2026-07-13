"""Tiled gather kernel: true squared L2 to selected kNN neighbours.

After a fused kNN pass that ranks candidates with the shift-invariant
score ``c_sq - 2 * <x, c>``, this kernel writes the **true**
``||x[b, n] - c[b, idx[b, n, k]]||^2`` for every neighbour. Computing
the difference directly (``(x - y)^2`` per d) avoids the cancellation
inherent to the ``x^2 + y^2 - 2*x*y`` expansion, so the result is
accurate even for fp32 inputs where the fused kNN kernel may have used
TF32 in the cross-term.

Complexity: ``O(B * N * K * D)`` -- cheap vs the ``O(B * N * M * D)``
GEMM that found the candidates.

Design
------
One program owns a ``(BN, K_BLOCK)`` output tile. The query row tile
``x[b, n_block, :]`` is loaded once and reused across the ``K_BLOCK``
neighbours, then streamed through ``D`` in ``BLOCK_D`` chunks. The
corpus rows ``c[b, idx, :]`` are gather-loaded fresh per d-chunk into a
3-D tile ``(BN, K_BLOCK, BLOCK_D)`` and subtracted directly from the
broadcast x tile in fp32. The fp32 squared difference is summed along
``D`` into the accumulator.

Three regime presets pick ``BN`` and ``K_BLOCK`` so the c-tile fits in
SMEM/registers and amortises one x-load over many neighbours:

  * K <= 32:        ``K_BLOCK = next_pow2(K)``, ``BN = 16``.
  * 32 < K <= 512:  ``K_BLOCK = 32``, ``BN = 8``; multiple K-tiles per row.
  * K > 512:        ``K_BLOCK = 32``, ``BN = 4``.

``BLOCK_D`` is chosen so the per-program SMEM footprint
``BN * K_BLOCK * BLOCK_D * sizeof(dtype)`` stays under ~32 KB. The
unmodified ``D`` (not padded) is passed as a constexpr so the Triton
compiler unrolls the d-loop exactly.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def _knn_gather_l2sq_kernel(
    x_ptr, c_ptr, idx_ptr, out_ptr,
    M,
    stride_x_b, stride_x_n, stride_x_d,
    stride_c_b, stride_c_m, stride_c_d,
    stride_i_b, stride_i_n, stride_i_k,
    stride_o_b, stride_o_n, stride_o_k,
    N: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
    BN: tl.constexpr, K_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SINGLE_D_TILE: tl.constexpr,
):
    """Tiled (BN, K_BLOCK, BLOCK_D) gather + sq-distance accumulate.

    Grid: ``(ceil(N / BN), ceil(K / K_BLOCK), B)``.

    Each program owns one ``(BN, K_BLOCK)`` output tile. The fp32
    accumulator stays in registers; the c-row tile ``(BN, K_BLOCK,
    BLOCK_D)`` is reloaded per d-chunk to keep SMEM bounded.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2).to(tl.int64)

    n_start = pid_n * BN
    n_offs = (n_start + tl.arange(0, BN)).to(tl.int64)
    n_mask = n_offs < N

    k_start = pid_k * K_BLOCK
    k_offs = (k_start + tl.arange(0, K_BLOCK)).to(tl.int64)
    k_mask = k_offs < K

    idx_tile = tl.load(
        idx_ptr + pid_b * stride_i_b
        + n_offs[:, None] * stride_i_n
        + k_offs[None, :] * stride_i_k,
        mask=n_mask[:, None] & k_mask[None, :], other=0,
    ).to(tl.int64)
    idx_tile = tl.maximum(idx_tile, 0)
    idx_tile = tl.minimum(idx_tile, M - 1)

    acc = tl.zeros((BN, K_BLOCK), dtype=tl.float32)

    if SINGLE_D_TILE:
        d_offs = tl.arange(0, BLOCK_D).to(tl.int64)
        d_mask = d_offs < D
        x_tile = tl.load(
            x_ptr + pid_b * stride_x_b
            + n_offs[:, None] * stride_x_n
            + d_offs[None, :] * stride_x_d,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        c_tile = tl.load(
            c_ptr + pid_b * stride_c_b
            + idx_tile[:, :, None] * stride_c_m
            + d_offs[None, None, :] * stride_c_d,
            mask=(n_mask[:, None] & k_mask[None, :])[:, :, None]
            & d_mask[None, None, :], other=0.0,
        ).to(tl.float32)
        diff = x_tile[:, None, :] - c_tile
        acc = tl.sum(diff * diff, axis=2)
    else:
        for d_start in tl.range(0, D, BLOCK_D):
            d_offs = (d_start + tl.arange(0, BLOCK_D)).to(tl.int64)
            d_mask = d_offs < D
            x_tile = tl.load(
                x_ptr + pid_b * stride_x_b
                + n_offs[:, None] * stride_x_n
                + d_offs[None, :] * stride_x_d,
                mask=n_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            c_tile = tl.load(
                c_ptr + pid_b * stride_c_b
                + idx_tile[:, :, None] * stride_c_m
                + d_offs[None, None, :] * stride_c_d,
                mask=(n_mask[:, None] & k_mask[None, :])[:, :, None]
                & d_mask[None, None, :], other=0.0,
            ).to(tl.float32)
            diff = x_tile[:, None, :] - c_tile
            acc += tl.sum(diff * diff, axis=2)

    tl.store(
        out_ptr + pid_b * stride_o_b
        + n_offs[:, None] * stride_o_n
        + k_offs[None, :] * stride_o_k,
        acc, mask=n_mask[:, None] & k_mask[None, :],
    )


def _pick_tile(K: int, D: int, dtype_bytes: int) -> tuple[int, int, int, bool]:
    """Pick ``(BN, K_BLOCK, BLOCK_D, single_d_tile)`` for the kernel.

    Regimes mirror the file docstring -- small ``K_BLOCK`` for large K,
    larger ``BN`` for small K. ``BLOCK_D`` is sized so the 3-D c-tile
    SMEM footprint stays around ~32 KB.
    """
    K_pad = _next_pow2(K)

    if K_pad <= 32:
        K_BLOCK = max(1, K_pad)
        BN = 16
    elif K_pad <= 512:
        K_BLOCK = 32
        BN = 8
    else:
        K_BLOCK = 32
        BN = 4

    target_bytes = 32 * 1024
    per_d = BN * K_BLOCK * dtype_bytes
    block_d_cap = max(16, target_bytes // max(1, per_d))
    block_d_cap = min(block_d_cap, 256)

    if D <= block_d_cap:
        BLOCK_D = max(16, _next_pow2(D))
        single_d_tile = True
    else:
        BLOCK_D = 64 if block_d_cap >= 64 else 32
        single_d_tile = False

    return BN, K_BLOCK, BLOCK_D, single_d_tile


def triton_knn_gather_sqdist(
    x: torch.Tensor,
    c: torch.Tensor,
    idx: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out[b, n, k] = || x[b, n, :] - c[b, idx[b, n, k], :] ||^2`` (fp32).

    Args:
        x:    (B, N, D) query tensor, any float cuda dtype (bf16 / fp16 / fp32).
        c:    (B, M, D) corpus, same dtype and same batch as ``x``.
        idx:  (B, N, K) int32 / int64 column indices into ``c``.
        out:  optional pre-allocated (B, N, K) fp32 output buffer.

    Returns:
        (B, N, K) fp32. The computation runs in fp32 throughout
        (``diff = x.to(fp32) - c.to(fp32)``, ``sum(diff*diff)``) so the
        result is bit-equivalent to the naive torch reference up to
        accumulation order on the d-axis.
    """
    assert x.is_cuda and c.is_cuda and idx.is_cuda
    assert x.dim() == 3 and c.dim() == 3 and idx.dim() == 3
    B, N, D = x.shape
    Bc, M, Dc = c.shape
    Bi, Ni, K = idx.shape
    assert B == Bc == Bi and N == Ni and D == Dc

    if not x.is_contiguous():
        x = x.contiguous()
    if not c.is_contiguous():
        c = c.contiguous()
    if not idx.is_contiguous():
        idx = idx.contiguous()

    if out is None:
        out = torch.empty((B, N, K), device=x.device, dtype=torch.float32)
    else:
        assert out.shape == (B, N, K) and out.dtype == torch.float32

    dtype_bytes = x.element_size()
    BN, K_BLOCK, BLOCK_D, single_d_tile = _pick_tile(K, D, dtype_bytes)

    grid = (triton.cdiv(N, BN), triton.cdiv(K, K_BLOCK), B)
    _knn_gather_l2sq_kernel[grid](
        x, c, idx, out,
        M,
        x.stride(0), x.stride(1), x.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N=N, D=D, K=K,
        BN=BN, K_BLOCK=K_BLOCK, BLOCK_D=BLOCK_D,
        SINGLE_D_TILE=single_d_tile,
        num_warps=4,
    )
    return out


__all__ = ["triton_knn_gather_sqdist"]
