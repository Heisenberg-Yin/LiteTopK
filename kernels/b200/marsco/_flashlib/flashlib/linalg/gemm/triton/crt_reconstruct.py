"""CRT reconstruction kernel for Ozaki Scheme II.

Each fractional contribution per modulus is computed in EXACT integer
arithmetic before converting to FP64, so the well-known subtractive
cancellation issue of ``frac = x - round(x)`` (which loses log2(|x|) bits)
is avoided entirely.

Math: for each t the term ``P_t * y_t / m_t mod 1`` equals
    ((P_t mod m_t) * y_t mod m_t) / m_t
Both inner ops fit in INT32 since m_t <= 254 and y_t < m_t.

Inputs:
  P_stack       : (S, M, N) INT32  — modular partial products
  inv_scale_a   : (M,)      FP64   — per-row inverse scale of A
  inv_scale_b   : (N,)      FP64   — per-row inverse scale of B (B is (N,K))
  alphas        : (S,)      FP64   — y_t / m_t  (CRT coefficients, FP64)
  moduli        : (S,)      INT32  — m_t          (used for the integer reduce)
  ys            : (S,)      INT32  — y_t          (used for the integer reduce)
  M_hi, M_lo    : FP64 scalars     — M = M_hi + M_lo (two-word, exact)

For each (i, j):
  frac_int  = sum_t (((P_t[i,j] mod m_t) * y_t) mod m_t) / m_t   (FP64, no cancellation)
  frac      = frac_int - round(frac_int)                         (FP64, in [-0.5, 0.5])
  C_int     = frac * (M_hi + M_lo)                               (two-word)
  out[i, j] = C_int * inv_scale_a[i] * inv_scale_b[j]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _crt_reconstruct_kernel(
    P_ptr,            # (S, M, N) INT32
    out_ptr,          # (M, N) FP64 or FP32
    inv_a_ptr,        # (M,)  FP64
    inv_b_ptr,        # (N,)  FP64
    moduli_ptr,       # (S,)  INT32   m_t
    ys_ptr,           # (S,)  INT32   y_t = (M/m_t)^{-1} mod m_t
    inv_m_ptr,        # (S,)  FP64    1.0 / m_t  (precomputed reciprocal)
    M, N,
    M_hi, M_lo,
    stride_p_s, stride_p_m, stride_p_n,
    stride_o_m, stride_o_n,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    OUT_FP64: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # INT64 byte-offset arithmetic so that S * M * N > 2^31 (e.g. M=N=16384,
    # S>=8) does not silently wrap.  Cheap on Hopper.
    offs_m64 = offs_m.to(tl.int64)
    offs_n64 = offs_n.to(tl.int64)
    base_pmn = offs_m64[:, None] * stride_p_m + offs_n64[None, :] * stride_p_n
    stride_p_s64 = tl.cast(stride_p_s, tl.int64)

    # Per-modulus exact-integer fractional contribution:
    #   r_t = ((P_t mod m_t) * y_t) mod m_t        (INT32, range [0, m_t))
    #   frac_t = r_t * (1/m_t)                     (FP64 mul, NOT div)
    # FP64 division is ~16-20x slower than FP64 multiply on Hopper (slow path
    # software emulation), so passing 1/m_t as a precomputed FP64 array gives
    # a clean win in the inner loop.
    frac = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for s in tl.static_range(0, S):
        m_t = tl.load(moduli_ptr + s)               # INT32
        y_t = tl.load(ys_ptr + s)                   # INT32
        inv_m = tl.load(inv_m_ptr + s)              # FP64 ~ 1/m_t
        p_off = s * stride_p_s64 + base_pmn
        p = tl.load(P_ptr + p_off, mask=mask, other=0)   # INT32
        # r = ((p % m_t) * y_t) % m_t — all INT32, no overflow:
        # |p mod m_t| < 254, * y_t < 254 -> < 64516 < 2^17.
        # Triton % follows C semantics (sign-of-dividend); fold to non-negative:
        r1 = p % m_t
        r1 = tl.where(r1 < 0, r1 + m_t, r1)
        r2 = (r1 * y_t) % m_t
        frac = frac + r2.to(tl.float64) * inv_m

    # Now frac in [0, S]; reduce to [-0.5, 0.5] via single round-to-nearest.
    rx = tl.extra.cuda.libdevice.rint(frac)
    frac = frac - rx

    # frac * M = frac * (M_hi + M_lo) — two-word FP64.
    val = frac * M_hi + frac * M_lo

    inv_a = tl.load(inv_a_ptr + offs_m, mask=offs_m < M, other=0.0)
    inv_b = tl.load(inv_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    val = val * inv_a[:, None] * inv_b[None, :]

    out_off = offs_m64[:, None] * stride_o_m + offs_n64[None, :] * stride_o_n
    if OUT_FP64:
        tl.store(out_ptr + out_off, val, mask=mask)
    else:
        tl.store(out_ptr + out_off, val.to(tl.float32), mask=mask)


def _get_inv_m(moduli: torch.Tensor) -> torch.Tensor:
    """Return a FP64 (S,) tensor of 1/m_t, on the same device as moduli.

    Recomputed every call (S * 8 bytes alloc + S divisions) — that overhead
    is negligible vs the GEMM but a data_ptr cache is dangerous because
    PyTorch's caching allocator reuses freed CUDA addresses, causing stale
    cache hits when the moduli tensor for one S is GC'd and the next S call
    happens to land on the same address.
    """
    return (1.0 / moduli.to(torch.float64)).contiguous()


def crt_reconstruct(
    P_stack: torch.Tensor,    # (S, M, N) INT32
    inv_a: torch.Tensor,      # (M,) FP64
    inv_b: torch.Tensor,      # (N,) FP64
    moduli: torch.Tensor,     # (S,) INT32
    ys: torch.Tensor,         # (S,) INT32
    M_hi: float, M_lo: float,
    out_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    S, M, N = P_stack.shape
    assert P_stack.dtype == torch.int32
    assert moduli.dtype == torch.int32 and ys.dtype == torch.int32
    out = torch.empty((M, N), device=P_stack.device, dtype=out_dtype)
    inv_m = _get_inv_m(moduli)
    # Tuned config from bench/agent_loop.py recon sweep on H200 (8192³, s=8):
    # (32, 128, num_warps=8) → 2.50 ms / 967 GB/s, vs old (64, 128, nw=4) at
    # 8.0 ms / 300 GB/s = 3.2x speedup; reciprocal-mul is another ~30% on top.
    BLOCK_M, BLOCK_N = 32, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _crt_reconstruct_kernel[grid](
        P_stack, out, inv_a, inv_b, moduli, ys, inv_m,
        M, N,
        float(M_hi), float(M_lo),
        P_stack.stride(0), P_stack.stride(1), P_stack.stride(2),
        out.stride(0), out.stride(1),
        S=S, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        OUT_FP64=(out_dtype == torch.float64),
        num_warps=8, num_stages=2,
    )
    return out
