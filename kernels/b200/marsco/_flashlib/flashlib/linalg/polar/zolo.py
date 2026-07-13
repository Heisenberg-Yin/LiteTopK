"""ZOLO-PD: Zolotarev-based polar / matrix-sign for symmetric A.

Rate-(2m+1) rational approximation to the matrix sign function. With
m chosen from the initial condition number, 2 iterations suffice to
reach machine precision for any κ₂(A) ≤ 10¹⁶.

Per iteration: 1 SYRK + m Cholesky + m triangular solves + 1 weighted
sum. The m sub-solves are INDEPENDENT (different shifts c_j applied
to the same X) — exploited via batching on GPU.

Reference: Nakatsukasa & Freund, "Computing Fundamental Matrix
Decompositions Accurately via the Matrix Sign Function in Two
Iterations", SIAM Review 58:3 (2016). Implementation pattern adapted
from https://github.com/timlautk/polargrad/blob/main/zolopd.py
(Apache 2.0), itself a port of Nakatsukasa's MATLAB code.

Public:
  polar_zolo(A, ...) -> U where A = U |A|
"""
import math
import torch


def _mellipke(alpha, tol=None):
    """Complete elliptic integrals K, E via AGM. alpha in radians."""
    if tol is None:
        tol = torch.finfo(torch.float64).eps
    m_val = math.sin(alpha) ** 2
    a0, b0, s0 = 1.0, math.cos(alpha), m_val
    i1, mm = 0, 1.0
    while mm > tol:
        a1 = 0.5 * (a0 + b0)
        b1 = math.sqrt(a0 * b0)
        c1 = 0.5 * (a0 - b0)
        i1 += 1
        mm = (2 ** i1) * (c1 ** 2)
        s0 += mm
        a0, b0 = a1, b1
    K = math.pi / (2 * a1)
    E = K * (1 - s0 / 2)
    return K, E


def _mellipj(u, alpha, tol=None):
    """Jacobi elliptic sn, cn, dn (scalar u, alpha in radians)."""
    if tol is None:
        tol = torch.finfo(torch.float64).eps
    m_val = math.sin(alpha) ** 2
    a_vals = [1.0]
    b_vals = [math.cos(alpha)]
    c_vals = [math.sin(alpha)]
    i = 0
    while abs(c_vals[i]) > tol and i < 1000:
        a_vals.append(0.5 * (a_vals[i] + b_vals[i]))
        b_vals.append(math.sqrt(a_vals[i] * b_vals[i]))
        c_vals.append(0.5 * (a_vals[i] - b_vals[i]))
        i += 1
    n = i
    phi = (2 ** n) * a_vals[-1] * u
    for j in range(n - 1, -1, -1):
        temp = c_vals[j + 1] * math.sin(phi) / a_vals[j + 1]
        temp = max(-1.0, min(1.0, temp))
        phi = 0.5 * (math.asin(temp) + phi)
    sn = math.sin(phi)
    cn = math.cos(phi)
    dn = math.sqrt(1 - m_val * (sn ** 2))
    return sn, cn, dn


def _choose_m(con):
    """Zolotarev degree m from condition estimate."""
    if con < 1.001: return 2
    if con <= 1.01: return 3
    if con <= 1.1:  return 4
    if con <= 1.2:  return 5
    if con <= 1.5:  return 6
    if con <= 2:    return 8
    if con < 6.5:   return 2
    if con < 180:   return 3
    if con < 1.5e4: return 4
    if con < 2e6:   return 5
    if con < 1e9:   return 6
    if con < 3e12:  return 7
    return 8


def _zolo_coeffs(con):
    """Zolotarev partial-fraction coefficients."""
    m = _choose_m(con)
    kp = 1.0 / con
    alpha_angle = math.acos(kp)
    K, _ = _mellipke(alpha_angle)
    c = torch.zeros(2 * m, dtype=torch.float64)
    for ii in range(2 * m):
        u = (ii + 1) * K / (2 * m + 1)
        sn, cn, _ = _mellipj(u, alpha_angle)
        c[ii] = (sn ** 2) / (cn ** 2)
    return c, m


def _eval_f(x, c):
    x2 = x * x
    val = x
    m = len(c) // 2
    for j in range(m):
        val = val * (x2 + c[2*j+1].item()) / (x2 + c[2*j].item())
    return val


def _partial_fraction_weights(c):
    m = len(c) // 2
    weights = torch.zeros(m, dtype=torch.float64)
    for ii in range(m):
        enu = 1.0
        for jj in range(m):
            enu *= (c[2*ii] - c[2*jj+1]).item()
        den = 1.0
        for jj in range(m):
            if ii != jj:
                den *= (c[2*ii] - c[2*jj]).item()
        weights[ii] = enu / den
    return weights


def _zolo_step_chol(X, c, weights, syrk_fp64=True):
    """One Zolotarev iteration via Cholesky path.

    X: (N,N) fp32, ≈symmetric, spectrum in [−1, 1].
    Returns X_new (N,N) fp32. Per iter: 1 SYRK + m Chol + m solves.
    """
    N = X.size(0)
    device = X.device
    dtype = X.dtype
    if syrk_fp64:
        X64 = X.to(torch.float64)
        M = (X64.T @ X64).to(dtype)
        del X64
    else:
        M = X.T @ X
    M = 0.5 * (M + M.T)

    I_n = torch.eye(N, device=device, dtype=dtype)
    m = len(c) // 2
    out = X.clone()
    for ii in range(m):
        c_val = c[2*ii].item()
        w_val = weights[ii].item()
        Z = M + c_val * I_n
        Z = 0.5 * (Z + Z.T)
        L = torch.linalg.cholesky(Z)
        Y = torch.cholesky_solve(X.T, L).T
        out = out - w_val * Y

    return out


def polar_zolo(A, alpha=None, L0=None, syrk_fp64=True, post_ns=True):
    """Polar factor of symmetric A via ZOLO-PD.

    A (n,n) symmetric fp32 on CUDA. Returns U (n,n) fp32 with U.T @ U ≈ I
    and U ≈ A (A²)^(-1/2).

    Scaling convention: pre-scale X = A/(alpha · L0) so the spectrum of
    X lies in [1, con] with con = 1/L0. Mis-scaling to [1/con, 1] gives
    f(x)/f(1) far from 1 and ~0.15 const rel-diff (memory: zolo_pd_scaling).
    """
    if alpha is None:
        from flashlib.linalg.polar.qdwh_hybrid import _spectral_norm_estimate
        alpha = _spectral_norm_estimate(A)
    if alpha == 0.0:
        return A.clone()

    X = A / alpha
    if L0 is None:
        L0 = 1e-5
    X = X / L0
    con = 1.0 / L0

    itmax = 1 if con < 2 else 2

    for it in range(itmax):
        c, m = _zolo_coeffs(con)
        weights = _partial_fraction_weights(c)
        f1 = _eval_f(1.0, c)
        X = _zolo_step_chol(X, c, weights, syrk_fp64=syrk_fp64)
        X = X / f1
        f_con = _eval_f(con, c)
        con = max(f_con / f1, 1.0)
        X = 0.5 * (X + X.T)
        if con < 1.001:
            break

    if post_ns:
        M = X.T @ X
        X = 1.5 * X - 0.5 * X @ M

    return X


# Back-compat alias used by tests/test_zolo.py.
zolo_polar = polar_zolo
