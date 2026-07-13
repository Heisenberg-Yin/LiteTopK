"""Fused per-row Ozaki-II split: FP32/FP64 (M, K) -> (S, M, K) INT8 + scale.

Algorithm (per row i):
  1. row_max = max_j |a_ij|
  2. exp_a   = floor(log2(row_max))                 # power-of-2 scale
  3. scale   = 2^(kA - exp_a - 1)                   # |a_ij * scale| <= 2^(kA-1)
  4. int_row = round(a_ij * scale)                  # INT32 in [-2^kA, 2^kA]
                                                      (kA <= 27 fits in int32)
  5. for each modulus m_t:
        a_t  = symmetric_mod(int_row, m_t)          # INT8 in [-127, 127]
        store at out_int[t, i, j]
  6. emit per-row "inverse scale" = 2^(exp_a - kA + 1)
        the FP factor needed to undo scale during reconstruction.

Implementation:
  Two passes — splitting the row-reduce from the modular split lifts the
  artificial "one CTA per row, BLOCK_K==K" constraint of the original
  fused kernel. Pass 1 uses ``torch.amax`` (a tuned cuBLAS-quality
  reduction). Pass 2 is tile-parallel over (M_blocks, K_blocks) so it
  spawns 100s-1000s of CTAs and pegs HBM bandwidth.

  Pass 2 also uses INT32 throughout the modular reduction (kA <= 27 fits
  in INT32; INT64 mod is software-emulated and ~5-10x slower).

Supported ``num_moduli`` range is 2..9 here. For s >= 10 use the GEMMul8
wrapper (``matmul_ozaki2``).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from flashlib.linalg.gemm.ozaki_constants import choose_kA_kB, crt_constants_tensor


@triton.jit
def _ozaki2_compute_scale_kernel(
    row_max_ptr,    # (M,) FP64 — per-row max |a|
    inv_scale_ptr,  # (M,) FP64  out: inverse scale
    scale_ptr,      # (M,) FP64  out: forward scale (passed to pass 2)
    M,
    KA: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M
    row_max = tl.load(row_max_ptr + offs, mask=mask, other=1.0)
    safe_max = tl.where(row_max > 0.0, row_max, 1.0)
    log2_max = tl.log2(safe_max)
    exp_a = tl.floor(log2_max)
    KA_F: tl.constexpr = float(KA - 1)
    scale_exp = KA_F - exp_a
    scale = tl.exp2(scale_exp)
    inv_scale = tl.exp2(-scale_exp)
    tl.store(scale_ptr + offs, scale, mask=mask)
    tl.store(inv_scale_ptr + offs, inv_scale, mask=mask)


@triton.jit
def _ozaki2_split_tile_kernel(
    in_ptr,         # (M, K) FP64 or FP32
    scale_ptr,      # (M,) FP64
    out_int_ptr,    # (S, M, K) INT8
    moduli_ptr,     # (S,) INT32
    M, K,
    stride_im, stride_ik,
    stride_o_s, stride_o_m, stride_o_k,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IN_FP64: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mm = offs_m[:, None] < M
    mk = offs_k[None, :] < K
    mask = mm & mk

    # Use INT64 for byte offsets so very large tensors (S * M * K > 2^31, e.g.
    # M=K=16384, S>=8) do not silently wrap.  Cheap on Hopper.
    offs_m64 = offs_m.to(tl.int64)
    offs_k64 = offs_k.to(tl.int64)
    in_off = offs_m64[:, None] * stride_im + offs_k64[None, :] * stride_ik
    if IN_FP64:
        row = tl.load(in_ptr + in_off, mask=mask, other=0.0)
    else:
        row = tl.load(in_ptr + in_off, mask=mask, other=0.0).to(tl.float64)

    scale = tl.load(scale_ptr + offs_m, mask=offs_m < M, other=1.0)
    scaled = row * scale[:, None]
    # kA <= 27 by construction (choose_kA_kB), so |int_row| < 2^27 fits in
    # INT32.  INT32 modulo is hardware-fast; INT64 modulo is software emul.
    int_row = tl.extra.cuda.libdevice.rint(scaled).to(tl.int32)

    base_mn = offs_m64[:, None] * stride_o_m + offs_k64[None, :] * stride_o_k
    stride_o_s64 = tl.cast(stride_o_s, tl.int64)
    for s in tl.static_range(0, S):
        m = tl.load(moduli_ptr + s)
        rem = int_row % m
        half = m // 2
        rem = tl.where(rem > half, rem - m, rem)
        rem = tl.where(rem < -half, rem + m, rem)
        out_off = s * stride_o_s64 + base_mn
        tl.store(out_int_ptr + out_off, rem.to(tl.int8), mask=mask)


def _next_pow2(x: int) -> int:
    return 1 << ((x - 1).bit_length()) if x > 1 else 1


# Tuned defaults from bench/agent_loop.py split sweep on H200 (8192³, s=8):
# Two-pass at (BLOCK_M=4, BLOCK_K=256, num_warps=2) → 0.39 ms (s=8) vs old
# fused 2.7 ms = 6.9x. Pass 1 reduction adds <50 us.
_DEFAULT_BLOCK_M = 4
_DEFAULT_BLOCK_K = 256
_DEFAULT_NUM_WARPS = 2


def split_fp_to_int8_moduli(
    x: torch.Tensor,
    num_moduli: int,
    kA: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (int_stack: (S, M, K) INT8, inv_scale: (M,) FP64)."""
    assert x.dim() == 2 and x.is_cuda
    assert x.dtype in (torch.float32, torch.float64)
    M, K = x.shape
    moduli_t, _, _, _, _ = crt_constants_tensor(num_moduli, x.device)
    assert num_moduli == moduli_t.numel()
    if kA > 30:
        raise ValueError(
            f"split kernel uses INT32 path; needs kA <= 30 but got kA={kA}. "
            "For larger kA increase num_moduli or reduce K."
        )

    x_c = x.contiguous()

    # Pass 1: per-row max via the highly-tuned cuDNN/cuBLAS reduction.
    row_max = x_c.abs().amax(dim=1).to(torch.float64)

    # Compute per-row scale + inverse scale.
    inv_scale = torch.empty((M,), device=x.device, dtype=torch.float64)
    scale = torch.empty((M,), device=x.device, dtype=torch.float64)
    BLOCK_M_S = 256
    _ozaki2_compute_scale_kernel[(triton.cdiv(M, BLOCK_M_S),)](
        row_max, inv_scale, scale, M,
        KA=kA, BLOCK_M=BLOCK_M_S, num_warps=2,
    )

    # Pass 2: tile-parallel modular split.
    out_int = torch.empty((num_moduli, M, K), device=x.device, dtype=torch.int8)
    BLOCK_M = _DEFAULT_BLOCK_M
    BLOCK_K = _DEFAULT_BLOCK_K
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    _ozaki2_split_tile_kernel[grid](
        x_c, scale, out_int, moduli_t,
        M, K,
        x_c.stride(0), x_c.stride(1),
        out_int.stride(0), out_int.stride(1), out_int.stride(2),
        S=num_moduli, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        IN_FP64=(x.dtype == torch.float64),
        num_warps=_DEFAULT_NUM_WARPS,
    )
    return out_int, inv_scale


# ---- self-test entry --------------------------------------------------------

def _selftest():
    """Sanity: reconstruct A from its modular slices and per-row scale."""
    torch.manual_seed(0)
    M, K = 64, 256
    s = 7
    a = torch.randn(M, K, device="cuda", dtype=torch.float64)
    kA, _ = choose_kA_kB(s, K)
    a_int, inv_scale = split_fp_to_int8_moduli(a, s, kA)
    from flashlib.linalg.gemm.ozaki_constants import crt_constants
    moduli, M_int, alphas, _ys = crt_constants(s)
    # Reconstruct integer per element using CRT, then scale back.
    # Convert int8 -> int64 host-side (small tensor, easy).
    a_int_h = a_int.cpu().to(torch.int64).numpy()
    inv_h = inv_scale.cpu().numpy()
    import numpy as np
    out = np.zeros((M, K), dtype=np.float64)
    M_py = M_int
    coeffs = []
    for t, m in enumerate(moduli):
        Mt = M_py // m
        y = pow(Mt % m, -1, m)
        coeffs.append(Mt * y)
    for i in range(M):
        for j in range(K):
            x_int = 0
            for t in range(s):
                x_int += int(a_int_h[t, i, j]) * coeffs[t]
            x_int %= M_py
            if x_int > M_py // 2:
                x_int -= M_py
            out[i, j] = x_int * inv_h[i]
    err = float(np.abs(out - a.cpu().numpy()).max())
    print(f"split selftest: max abs err {err:.3e} (kA={kA}, s={s})")


if __name__ == "__main__":
    _selftest()
