"""Fused kernels for the QDWH polar-factor stages.

Currently implemented:
  * `fused_ozaki_matmul`  — single-launch Ozaki 3xbf16 (or single-bf16) GEMM
                            on fp32 inputs/outputs. Replaces nvmath's 3-launch
                            accumulator-chaining + 4 HBM cast passes.
  * `kenney_laub_step`    — one Kenney-Laub cubic Newton-Schulz iteration
                            `X' = X (15 I - 10 M + 3 M²) / 8` built from
                            three fused-Ozaki GEMMs.

Planned (not yet implemented in this file):
  * K1 `syrk_symm_fp32` — fp32 SYRK + shift + symmetrize fused.
  * K2 `chol_trtri_fp32` — chol(Z) + trtri(L) fused.
  * K3 `chained_trsm_3xtf32` — (X @ L_inv.T) @ L_inv single kernel.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Ozaki-split 3-product bf16 GEMM, fused into a single Triton kernel
# ---------------------------------------------------------------------------
# The unfused path (diag/bf16_mm.py via nvmath) costs 6.85 ms per 3xbf16 at
# N=8192, ~49% of bf16 peak. The 51-point gap is ~11 pts of L2 traffic from
# writing four bf16 workspace buffers (A_hi, A_lo, B_hi, B_lo) plus the
# kernel-launch overhead of three chained cuBLAS LT calls.
#
# In this kernel, each CTA loads fp32 tiles of A and B, splits them into
# (hi, lo) bf16 in registers, runs up to three `tl.dot` calls against a
# shared fp32 accumulator, and writes a single fp32 output. No bf16 scratch
# is ever stored to HBM.

# Hopper-tuned configs. WGMMA bf16 uses m64n{8,16..256}k16 shapes natively,
# so BLOCK_M multiples of 64 + BLOCK_N multiples of 8 are preferred. For
# large K (N=8192 inner), num_stages≥3 with async pipelining (TMA) on SM90
# matters more than larger BLOCK_K. `num_warps=8` lets one CTA drive two
# WGMMA instructions per cycle on Hopper.
# Two-accumulator MODE=1 doubles register pressure; only configs whose acc
# footprint (2 × BLOCK_M × BLOCK_N × 4B / (num_warps × 32)) stays under ~120
# regs/thread avoid spilling. On Hopper with the 255-reg limit that means
# BLOCK_M × BLOCK_N ≤ 128 × 128 at num_warps=8, or ≤ 128 × 64 at num_warps=4.
_OZAKI_CONFIGS = [
    # Primary tile for the N ≥ 4096 regime — 128×128 at warps=8 fits 2-acc
    # budget and drives 2 WGMMA groups per CTA.
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=5, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8},
                  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8},
                  num_stages=4, num_warps=8),
    # Rectangular tiles that still fit — widen one dim, narrow the other.
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=4, num_warps=8),
    # Smaller variants for thin-shape / small-N fallbacks.
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=5, num_warps=4),
]


@triton.autotune(configs=_OZAKI_CONFIGS, key=['M', 'N', 'K', 'MODE'])
@triton.jit
def _ozaki_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    MODE: tl.constexpr,  # 0 = single bf16, 1 = 3xbf16 Ozaki
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Two accumulators in MODE=1: the dominant hi*hi product in `acc_hh` and
    # the two small-magnitude corrections summed together in `acc_cc`. We
    # separate them because a single interleaved accumulator, once it grows
    # to the magnitude of sum(hi*hi), rounds off bits of each subsequent
    # small correction tile, systematically biasing the result vs nvmath's
    # separate-matmul-then-add semantics. Keeping the corrections in their
    # own O(K * 2^-8)-magnitude register and adding them only at store time
    # loses one rounding rather than O(K) of them.
    # (3 separate accumulators would match nvmath bit-for-bit but blow the
    # register budget at the tile sizes we need for throughput.)
    acc_hh = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_cc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a_fp32 = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b_fp32 = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)

        a_hi = a_fp32.to(tl.bfloat16)
        b_hi = b_fp32.to(tl.bfloat16)

        if MODE == 0:
            acc_hh = tl.dot(a_hi, b_hi, acc_hh)
        else:
            a_lo = (a_fp32 - a_hi.to(tl.float32)).to(tl.bfloat16)
            b_lo = (b_fp32 - b_hi.to(tl.float32)).to(tl.bfloat16)
            acc_hh = tl.dot(a_hi, b_hi, acc_hh)
            acc_cc = tl.dot(a_hi, b_lo, acc_cc)
            acc_cc = tl.dot(a_lo, b_hi, acc_cc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if MODE == 0:
        acc = acc_hh
    else:
        acc = acc_hh + acc_cc

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def fused_ozaki_matmul(A, B, mode='3xbf16', out=None):
    """Fused-launch Ozaki matmul on H100 bf16 tensor cores.

    Args:
        A: fp32 tensor shape (M, K). Any stride layout (handles .T views).
        B: fp32 tensor shape (K, N). Any stride layout.
        mode: '3xbf16' (Ozaki 3-product, ~1e-5 rel err, matches fp32) or
              'bf16' (single-bf16, ~2.3e-3 rel err — use only where a
              later stage contracts the noise).
        out: optional preallocated fp32 (M, N).

    Returns:
        C (M, N) fp32.

    Single kernel launch. No HBM staging buffers — the bf16 split of A and B
    is computed in registers per-tile.
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    assert A.is_cuda and B.is_cuda
    assert A.ndim == 2 and B.ndim == 2 and A.size(1) == B.size(0)
    M, K = A.shape
    _, N = B.shape
    if out is None:
        out = torch.empty(M, N, dtype=torch.float32, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.float32

    mode_code = 1 if mode == '3xbf16' else 0

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _ozaki_matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        MODE=mode_code,
    )
    return out


# ---------------------------------------------------------------------------
# TF32-path matmul (K3 building block)
# ---------------------------------------------------------------------------
# Single kernel, fp32 in / fp32 out, with the three products fused inside
# `tl.dot(..., input_precision='tf32x3')`. Triton's tf32x3 lowers to three
# TF32 WGMMA instructions with shared-fp32 accumulator, matching the
# Python `_mm_3xtf32` (`_tf32_round(A)@B_hi + ...`) reference at ~1e-7
# rel error.

_TF32_CONFIGS = [
    # For 3xTF32 the kernel holds 2 fp32 accumulators (hi, corr) like the
    # bf16 Ozaki path. Same register budget: BLOCK_M × BLOCK_N ≤ 128×128 at
    # num_warps=8.
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                  num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                  num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8},
                  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                  num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8},
                  num_stages=4, num_warps=4),
]


@triton.jit
def _tf32_round(x):
    """Round fp32 → TF32 (zero low 13 mantissa bits).

    Keeps dtype fp32 so tl.dot picks the TF32 WGMMA path under
    input_precision='tf32'.
    """
    mask = 0xFFFFE000
    return (x.to(tl.int32, bitcast=True) & mask).to(tl.float32, bitcast=True)


@triton.autotune(configs=_TF32_CONFIGS, key=['M', 'N', 'K', 'PREC'])
@triton.jit
def _tf32_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    PREC: tl.constexpr,  # 0 = tf32 (single WGMMA), 1 = tf32x3 (manual 3-split)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 2-accumulator split matches nvmath's chained-beta semantics (see the
    # Ozaki kernel above for the rounding-bias argument). For PREC=0 (tf32)
    # the corr accumulator is unused; the compiler elides it.
    acc_hh = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_cc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        if PREC == 0:
            acc_hh = tl.dot(a, b, acc_hh, input_precision='tf32')
        else:
            a_hi = _tf32_round(a)
            b_hi = _tf32_round(b)
            a_lo = a - a_hi
            b_lo = b - b_hi
            acc_hh = tl.dot(a_hi, b_hi, acc_hh, input_precision='tf32')
            acc_cc = tl.dot(a_hi, b_lo, acc_cc, input_precision='tf32')
            acc_cc = tl.dot(a_lo, b_hi, acc_cc, input_precision='tf32')
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if PREC == 0:
        acc = acc_hh
    else:
        acc = acc_hh + acc_cc

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def fused_tf32_matmul(A, B, mode='tf32x3', out=None):
    """fp32-in/fp32-out matmul on TF32 tensor cores. Single kernel launch.

    mode='tf32x3' is the 3-split Ozaki-TF32 variant (matches fp32 to ~1e-7
    rel err; use when a later stage amplifies the matmul error by >1e2).
    mode='tf32' is one TF32 WGMMA (~1e-3 rel err; use only where a later
    stage contracts the noise, e.g. on a SYRK that feeds a Cholesky + solve).
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    assert A.is_cuda and B.is_cuda
    assert A.ndim == 2 and B.ndim == 2 and A.size(1) == B.size(0)
    M, K = A.shape
    _, N = B.shape
    if out is None:
        out = torch.empty(M, N, dtype=torch.float32, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.float32
    prec_code = 1 if mode == 'tf32x3' else 0
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _tf32_matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        PREC=prec_code,
    )
    return out


# ---------------------------------------------------------------------------
# Kenney-Laub polar-factor iteration (K4)
# ---------------------------------------------------------------------------
#     M  = X^T X
#     M2 = M @ M
#     X' = X (15 I - 10 M + 3 M²) / 8
#
# All three matmuls route through the fused Ozaki GEMM above. Asymmetric
# precision schedule for QDWH:
#   iter 1: inner matmuls (X^T X, M@M) are single-bf16; output matmul is
#           3xbf16 (its error goes straight to X' with no contraction).
#   iter 2: all three matmuls are 3xbf16 (output error IS final polar error).
# The iter-1 inner drop is safe because the cubic iter-2 contracts any
# residue introduced by the bf16 inners.

def kenney_laub_step(X, *, inner_mode='3xbf16', output_mode='3xbf16',
                     I_n=None, out=None):
    """One Kenney-Laub cubic Newton-Schulz iteration of the polar factor.

    Args:
        X: fp32 (N, N). Should be `||X^T X - I|| <~ 0.3` (inside the NS basin).
        inner_mode: precision for the two inner matmuls (X^T X, M@M).
                    Use 'bf16' on iter 1 (error contracted by iter 2); use
                    '3xbf16' on iter 2 (final iter, no contraction to rely on).
        output_mode: precision for the final X @ poly matmul. Stays '3xbf16'
                     in both iters (no further contraction).
        I_n: optional identity (N, N) fp32 to reuse across calls. If None,
             constructed on the fly.
        out: optional output buffer fp32 (N, N).

    Returns:
        X_new (N, N) fp32. Orthogonality error contracts cubically.
    """
    N = X.size(0)
    device, dtype = X.device, X.dtype
    assert dtype == torch.float32
    if I_n is None:
        I_n = torch.eye(N, device=device, dtype=dtype)

    # M = X^T X  (pass X.T as A — stride-swap view; no copy)
    M = fused_ozaki_matmul(X.T, X, mode=inner_mode)
    # M² = M @ M
    M2 = fused_ozaki_matmul(M, M, mode=inner_mode)
    # poly = (15·I - 10·M + 3·M²) / 8  (pointwise; negligible cost)
    # Overwrite M2 in place to save one HBM allocation.
    poly = M2.mul_(3.0 / 8.0).add_(M, alpha=-10.0 / 8.0).add_(I_n, alpha=15.0 / 8.0)
    # X' = X @ poly
    return fused_ozaki_matmul(X, poly, mode=output_mode, out=out)


# ---------------------------------------------------------------------------
# Elementwise glue kernels for the QDWH / Kenney-Laub fast path.
# ---------------------------------------------------------------------------
# These replace torch arithmetic chains that otherwise launch 3-4 separate
# elementwise kernels (one per temp) and pay 2-3 HBM round-trips on N×N fp32
# tensors. At N=8192 each HBM pass on 256MB is ~180μs at 3 TB/s, and torch's
# multi-op chain costs 0.6-1.2 ms that a single fused kernel does in ~0.2 ms.

_GLUE_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
]


@triton.autotune(configs=_GLUE_CONFIGS, key=['N'])
@triton.jit
def _axpby_kernel(
    X_ptr, Y_ptr, O_ptr,
    N,
    a, b,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
    x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, mask=mask, other=0.0)
    o = a * x + b * y
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, o, mask=mask)


def fused_axpby(X, Y, a, b, out=None):
    """`out = a*X + b*Y` in a single fused Triton kernel.

    All inputs fp32 (N,N). `out` can be one of the inputs (in-place). Avoids
    the 3-pass torch chain `a*X + b*Y` (alloc temp, alloc temp, add).
    """
    assert X.shape == Y.shape and X.dtype == Y.dtype == torch.float32
    N = X.size(0)
    if out is None:
        out = torch.empty_like(X)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _axpby_kernel[grid](
        X, Y, out,
        N, float(a), float(b),
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.autotune(configs=_GLUE_CONFIGS, key=['N'])
@triton.jit
def _shift_sym_kernel(
    M_ptr, O_ptr,
    N,
    c,
    stride_mm, stride_mn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Compute O[i,j] = 0.5 * c * (M[i,j] + M[j,i]) + (i == j ? 1.0 : 0.0)
    # Reads M twice — once at (i,j), once at (j,i). No separate I_n tensor needed.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
    m_ij = tl.load(M_ptr + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn, mask=mask, other=0.0)
    m_ji = tl.load(M_ptr + offs_n[None, :] * stride_mm + offs_m[:, None] * stride_mn, mask=mask, other=0.0)
    diag = tl.where(offs_m[:, None] == offs_n[None, :], 1.0, 0.0)
    o = 0.5 * c * (m_ij + m_ji) + diag
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, o, mask=mask)


def fused_shift_sym(M, c, out=None):
    """`out = 0.5 * c * (M + M.T) + I` in one kernel, no I_n tensor needed.

    Replaces the 3-pass sequence `Z = c*M + I_n; Z = 0.5*(Z + Z.T)` in the
    QDWH Cholesky inner loop.
    """
    assert M.dtype == torch.float32
    N = M.size(0)
    if out is None:
        out = torch.empty_like(M)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _shift_sym_kernel[grid](
        M, out,
        N, float(c),
        M.stride(0), M.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.autotune(configs=_GLUE_CONFIGS, key=['N'])
@triton.jit
def _kl_poly_kernel(
    M_ptr, M2_ptr, O_ptr,
    N,
    stride_mm, stride_mn,
    stride_m2m, stride_m2n,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # poly[i,j] = (15*(i==j) - 10*M[i,j] + 3*M2[i,j]) / 8
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
    m = tl.load(M_ptr + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn, mask=mask, other=0.0)
    m2 = tl.load(M2_ptr + offs_m[:, None] * stride_m2m + offs_n[None, :] * stride_m2n, mask=mask, other=0.0)
    diag = tl.where(offs_m[:, None] == offs_n[None, :], 15.0 / 8.0, 0.0)
    o = diag + (-10.0 / 8.0) * m + (3.0 / 8.0) * m2
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, o, mask=mask)


def fused_kl_poly(M, M2, out=None):
    """`out = (15*I - 10*M + 3*M2) / 8` in one kernel, no I_n tensor needed.

    Replaces the 4-pass sequence `(15*I_n - 10*M + 3*M2) * 0.125` or the
    3-pass in-place `M2.mul_(3/8).add_(M, alpha=-10/8).add_(I_n, alpha=15/8)`.
    """
    assert M.shape == M2.shape and M.dtype == M2.dtype == torch.float32
    N = M.size(0)
    if out is None:
        out = torch.empty_like(M)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _kl_poly_kernel[grid](
        M, M2, out,
        N,
        M.stride(0), M.stride(1),
        M2.stride(0), M2.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.autotune(configs=_GLUE_CONFIGS, key=['N'])
@triton.jit
def _sym_kernel(
    A_ptr, O_ptr,
    N,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # O[i,j] = 0.5 * (A[i,j] + A[j,i])
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
    a_ij = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask, other=0.0)
    a_ji = tl.load(A_ptr + offs_n[None, :] * stride_am + offs_m[:, None] * stride_an, mask=mask, other=0.0)
    o = 0.5 * (a_ij + a_ji)
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, o, mask=mask)


def fused_sym(A, out=None):
    """`out = 0.5*(A + A.T)` in one kernel. Replaces torch's 2-pass chain."""
    assert A.dtype == torch.float32
    N = A.size(0)
    if out is None:
        out = torch.empty_like(A)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _sym_kernel[grid](
        A, out, N,
        A.stride(0), A.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.autotune(configs=_GLUE_CONFIGS, key=['N'])
@triton.jit
def _diag_shift_kernel(
    A_ptr, O_ptr,
    N,
    sigma,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # O[i,j] = A[i,j] - (i==j ? sigma : 0)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask, other=0.0)
    shift = tl.where(offs_m[:, None] == offs_n[None, :], sigma, 0.0)
    o = a - shift
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, o, mask=mask)


def fused_diag_shift(A, sigma, out=None):
    """`out = A - sigma*I` in one kernel, no I_n tensor."""
    assert A.dtype == torch.float32
    N = A.size(0)
    if out is None:
        out = torch.empty_like(A)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _diag_shift_kernel[grid](
        A, out, N, float(sigma),
        A.stride(0), A.stride(1),
        out.stride(0), out.stride(1),
    )
    return out
