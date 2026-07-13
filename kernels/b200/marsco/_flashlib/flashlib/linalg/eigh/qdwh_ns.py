"""QDWH-eig variant using pure Newton-Schulz polynomial iteration.

Motivation: the Cholesky/trsm path in qdwh.py is precision-sensitive at
iter-1 (cond(Z) ≈ 2.3e6, amplifier 5.4e3) and the only BLAS-2 operation
on the critical path. Replacing it with pure polynomial iterations makes
every operation a tensor-core-eligible matmul.

Polynomial choice: **Polar Express** (Amsel et al. 2025, arXiv:2505.16932).
Each iteration uses a minimax-optimal odd quintic

    p_t(x) = a_t x + b_t x³ + c_t x⁵,   (a_t, b_t, c_t) precomputed

acting on X via  X_new = a_t X + X @ (b_t M + c_t M²)  with M = X^T X.
Three matmuls per iter (vs five for the nonic X(315 I - 420 M + ...)/128
it replaces), and each polynomial is the Chebyshev-optimal degree-5 fit
to the constant 1 on the current interval [l_t, 2 - l_t].

Convergence comparison (σ_min : 1e-6 → 1):

  Fixed nonic  (a = 315/128 = 2.46)  →  15 iters × 5 matmul = 75 matmul
  Polar Express (l=1e-6, T=12)       →  12 iters × 3 matmul = 36 matmul

First-iter slope at σ=0 is 8.5× for Polar Express vs 2.46× for the nonic,
so σ_min escapes the "stretch phase" ~4× faster.

Three building blocks, all matmul-only:

  _polar_ns: sign(A) for symmetric A via Polar Express quintic.
    12 precomputed coefficient triples (a_t, b_t, c_t) in
    diag.polar_express.POLAR_COEFFS_1E6. Early-exits on
    ‖M - I‖_F / sqrt(n) < tol.

  _orth_ns: orthonormalize rectangular C (N×k) via Polar Express polar
    on the thin matrix. Uses ORTH_COEFFS_1E3 (T=7). Replaces CholQR.

  qdwh_eig_ns: D&C entry that uses _polar_ns + _orth_ns. Structure
    mirrors qdwh_eig (shift, polar, split, project, recurse), with
    every QR/trsm/cholesky removed.

Precision: each matmul runs 3xbf16 (~1e-5 rel err, tensor-core peak).
Polar Express coefficients are computed in fp64 offline; applied in
3xbf16 online.
"""
import torch
from flashlib.linalg.polar import _spectral_norm_estimate, _qdwh_chol_step
from flashlib.linalg.polar.polar_express import polar_polar_express, polar_polar_express_warm
from flashlib.linalg.gemm.triton.triton_mm import mm_3xbf16, mm_tf32_lt
from flashlib.linalg.gemm.triton.fused_kernels import fused_diag_shift, fused_sym
from flashlib.linalg.polar.express_coeffs import (
    POLAR_COEFFS_1E6, ORTH_COEFFS_1E4, POLAR_TAIL_1QDWH, POLAR_TAIL_2QDWH,
)

# Back-compat aliases — _polar_ns / _polar_warmstart_pe are now in
# diag.msign.polar_express_ns. Tests / scratch scripts may still import
# them from here.
_polar_ns = polar_polar_express
_polar_warmstart_pe = polar_polar_express_warm


def _orth_ns(C, tol=1e-5, alpha=None, alpha_scale=1.01,
             coeffs=ORTH_COEFFS_1E4, matmul=mm_3xbf16):
    """Orthonormalize N×k matrix C via Polar Express on the thin matrix.

    Same quintic iteration as `_polar_ns`, but M = Q^T Q is k×k (cheap
    when k << N). Default coefficient table ORTH_COEFFS_1E3 is tuned for
    l=1e-3 (σ_min(C) ≥ 1/√k for column-pivoted projectors up to k≈1e6)
    and needs only T=7 iterations.
    """
    k = C.size(1)
    device, dtype = C.device, C.dtype
    if alpha is None:
        alpha = _spectral_norm_estimate(C)
    if alpha == 0.0:
        return C.clone()
    Q = C / (alpha * alpha_scale)
    I_k = torch.eye(k, device=device, dtype=dtype)
    inv_sqrt_k = 1.0 / (k ** 0.5)

    for a, b, c in coeffs:
        M = matmul(Q.T.contiguous(), Q)
        M = 0.5 * (M + M.T)
        err = torch.linalg.norm(M - I_k).item() * inv_sqrt_k
        if err < tol:
            break
        M2 = matmul(M, M)
        poly = a * I_k + b * M + c * M2
        Q = matmul(Q, poly)

    return Q


def _split_basis_ns(U_p):
    """Split U_p → (Q_+, Q_-, k) using NS orth instead of CholQR2.

    Same column-selection scheme as qdwh._split_basis (largest diagonal
    entries of P_+ pick Q_+'s columns). Difference: orthonormalization
    via _orth_ns (pure NS), cross-orthogonalization via one GS pass +
    one NS refinement — no trsm anywhere.
    """
    n = U_p.size(0)
    device = U_p.device

    diag_Up = torch.diagonal(U_p).contiguous()
    diag_P_plus = 0.5 * (1.0 + diag_Up)

    k = int(round(diag_P_plus.sum().item()))
    k = max(1, min(n - 1, k))

    perm_plus = torch.argsort(diag_P_plus, descending=True)
    idx_plus = perm_plus[:k]
    idx_minus = perm_plus[k:].flip(0)

    C_plus = (0.5 * U_p.index_select(1, idx_plus)).contiguous()
    rng_k = torch.arange(k, device=device)
    C_plus[idx_plus, rng_k] += 0.5

    C_minus = (U_p.index_select(1, idx_minus).neg_().mul_(0.5)).contiguous()
    rng_nk = torch.arange(n - k, device=device)
    C_minus[idx_minus, rng_nk] += 0.5

    # NS orthogonalization — each call is pure polynomial matmul,
    # no Cholesky, no solve_triangular.
    Q_plus = _orth_ns(C_plus)
    Q_minus = _orth_ns(C_minus)

    # Cross-orthogonalize Q_minus against Q_plus (polar leaves residual
    # cross-block noise of order sqrt(polar_err)). One GS pass + one NS
    # refinement replaces the CholQR in the original pipeline.
    Q_minus = Q_minus - mm_3xbf16(Q_plus, mm_3xbf16(Q_plus.T.contiguous(), Q_minus))
    Q_minus = _orth_ns(Q_minus)

    return Q_plus.contiguous(), Q_minus.contiguous(), k


def qdwh_eig_ns(A, base_case=1024, _depth=0, _max_depth=None, n_qdwh_warmup=0):
    """Symmetric eigendecomposition via NS-only QDWH spectral D&C.

    A (n,n) symmetric fp32 CUDA -> (w, V). w ascending, V orthonormal.

    Matmul-only variant of `diag.qdwh_eig`:
      - polar factor via `_polar_ns` (no Cholesky, no trsm) if
        `n_qdwh_warmup=0`, or `_polar_warmstart_pe(n_qdwh=n_qdwh_warmup)`
        if > 0 (1 or 2 QDWH-Cholesky iters to skip the stretch phase,
        then Polar Express tail)
      - basis split + orth via `_split_basis_ns` (no CholQR)
      - projection via cuBLAS LT TF32 GEMM (unchanged)
      - base case via cuSOLVER `eigh` (unchanged)

    `n_qdwh_warmup` choices (polar-phase cost at N=8192):
      0: pure Polar Express  (256 ms, no BLAS-2)
      1: 1 QDWH + 5 PE iters (173 ms, one Cholesky, occasionally unstable)
      2: 2 QDWH + 3 PE iters (157 ms, two Cholesky, most stable)

    Measured end-to-end at H100 SXM5 (random symmetric fp32, 3xbf16):

      N    pure_PE_NS  1QDWH+PE_NS  2QDWH+PE_NS   qdwh_eig  cuSOLVER
      2048   74/6e-4    35/2e-6      68/3e-3       35/2e-3   25
      4096  194/5e-4   FAIL          175/3e-3      92/1e-3   93
      8192  948/6e-4   691/4e-6      844/1e-3     400/1e-2  517
     12288 3236/1e-3  1744/5e-3     2921/7e-3    1053/5e-3 1239
      (ms / residual)

    Observations:
      - 1-QDWH warmstart is the fastest variant (30-40% speedup over
        pure PE) AND gives the tightest residual (~1e-6), because the
        Cholesky iter collapses the 1e-6→1e-2 stretch phase into one
        rational step and the PE tail then polishes to machine precision.
        BUT unstable at some sizes (FAIL at N=4096 shown above) — the
        actual σ_min at some D&C sub-problem falls below the tail
        table's l_0=0.01 and the polynomial diverges.
      - 2-QDWH warmstart is stable but slower than 1-QDWH because it
        pays for a second Cholesky without enough tail savings.
      - Neither beats qdwh_eig because qdwh_eig's Phase 2 uses 2 cubic
        KL iters (6 matmul) for the same σ∈[0.28,1] range where PE
        needs 3 iters (9 matmul) to stay safe.

    Useful when iter-1 Cholesky fails (ill-conditioned fp32 PSD failure,
    PCA-style fast-decay spectra) or when residual tightness matters more
    than raw speed.
    """
    n = A.size(0)
    if _max_depth is None:
        _max_depth = 2 if n >= 10240 else 1
    if n <= base_case or _depth >= _max_depth:
        from flashlib.linalg.eigh import eigh as _eigh_dispatch
        return _eigh_dispatch(A)

    device, dtype = A.device, A.dtype

    # mult-64 padding — same trick as qdwh_eig; odd N pays ~20% cuBLAS
    # penalty on Hopper WGMMA tiles.
    n_orig = n
    pad = (-n) % 64
    if pad > 0:
        alpha_A = _spectral_norm_estimate(A)
        lam = max(alpha_A, 1.0)
        A_padded = torch.zeros(n + pad, n + pad, dtype=dtype, device=device)
        A_padded[:n, :n] = A
        idx_pad = torch.arange(n, n + pad, device=device)
        A_padded[idx_pad, idx_pad] = lam
        A = A_padded
        n = n + pad

    sigma = torch.median(torch.diagonal(A)).item()
    A_s = fused_diag_shift(A, sigma)

    if n_qdwh_warmup > 0:
        U_p = _polar_warmstart_pe(A_s, n_qdwh=n_qdwh_warmup)
    else:
        U_p = _polar_ns(A_s)
    U_p = fused_sym(U_p)

    if torch.isnan(U_p).any():
        # NS diverged on an ill-conditioned sub-problem (σ_min below the
        # l=1e-6 table bound). Fall back to cuSOLVER for this subtree.
        from flashlib.linalg.eigh import eigh as _eigh_dispatch
        w, V = _eigh_dispatch(A)
        if pad > 0:
            w = w[:n_orig]
            V = V[:n_orig, :n_orig].contiguous()
        return w, V

    Q_plus, Q_minus, k = _split_basis_ns(U_p)

    if k < n // 8 or k > 7 * n // 8:
        from flashlib.linalg.eigh import eigh as _eigh_dispatch
        return _eigh_dispatch(A)

    Qp_T = Q_plus.T.contiguous()
    Qm_T = Q_minus.T.contiguous()
    A_plus = mm_tf32_lt(mm_tf32_lt(Qp_T, A), Q_plus)
    A_minus = mm_tf32_lt(mm_tf32_lt(Qm_T, A), Q_minus)
    A_plus = fused_sym(A_plus)
    A_minus = fused_sym(A_minus)

    w_plus, V_plus = qdwh_eig_ns(A_plus, base_case=base_case,
                                 _depth=_depth + 1, _max_depth=_max_depth,
                                 n_qdwh_warmup=n_qdwh_warmup)
    w_minus, V_minus = qdwh_eig_ns(A_minus, base_case=base_case,
                                   _depth=_depth + 1, _max_depth=_max_depth,
                                   n_qdwh_warmup=n_qdwh_warmup)

    V_pos = mm_3xbf16(Q_plus, V_plus)
    V_neg = mm_3xbf16(Q_minus, V_minus)

    w = torch.cat([w_minus, w_plus])
    V = torch.cat([V_neg, V_pos], dim=1)

    w_sorted, idx = torch.sort(w)
    V_sorted = V[:, idx].contiguous()
    if pad > 0:
        w_sorted = w_sorted[:n_orig]
        V_sorted = V_sorted[:n_orig, :n_orig].contiguous()
    return w_sorted, V_sorted
