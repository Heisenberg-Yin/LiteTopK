"""Polar Express coefficients for Newton-Schulz polar iteration.

Reference: Amsel, Pouya, Freund, Nakatsukasa, Tropp. "The Polar Express:
Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"
(arXiv:2505.16932, 2025).

Idea: instead of fixed cubic Kenney-Laub (15/8, -10/8, 3/8) or the
classic nonic (315, -420, 378, -180, 35)/128, use a **quintic** odd
polynomial p_t(x) = a_t x + b_t x³ + c_t x⁵ whose coefficients change
every iteration. At each iteration t, (a_t, b_t, c_t) is the minimax
approximation to 1 on the interval [l_t, u_t = 2 - l_t] — the image
of the previous interval under the previous polynomial.

Per-iter cost vs our previous "nonic quintic":

  true quintic p(x) = a x + b x³ + c x⁵  →  3 matmuls:
      M = X^T X;  M2 = M @ M;  X_new = a X + X @ (b M + c M²)
  nonic (prev, 315,-420,378,-180,35)     →  5 matmuls:
      M, M², M³, M⁴, X @ poly

Initial-iter growth rate on σ_min near 0:

  Polar Express t=0 (l=1e-6): a_0 ≈ 8.28   (5× our 2.46)
  our fixed nonic:            a   = 315/128 = 2.46
  Muon fixed quintic:         a   = 3.4445

Result: converges in ~half the iterations with ~60% compute per iter.

Numerics — `optimal_quintic` uses the simplified Remez algorithm (2-4
iterations of linear solve + root finding) to find the optimal odd
quintic. Called offline once per (l, num_iters) pair and cached; online
cost is zero.
"""
from math import inf, sqrt
import numpy as np


def optimal_cubic(l, u):
    """Minimax odd-cubic approximation a*x + b*x³ to 1 on [l, u]."""
    alpha = sqrt(3.0 / (u * u + l * u + l * l))
    beta = 4.0 / (2.0 + l * u * (l + u) * (alpha ** 3))
    return (1.5 * alpha * beta, -0.5 * (alpha ** 3) * beta)


def optimal_quintic(l, u):
    """Minimax odd-quintic approximation a*x + b*x³ + c*x⁵ to 1 on [l, u].

    Returns (a, b, c). Uses simplified Remez: at each step, solve the
    4×4 equioscillation system at current node guesses (l, q, r, u),
    then update q, r to the roots of p'(x) = 0.
    """
    assert 0 <= l <= u
    if 1.0 - 5e-6 <= l / u:
        return ((15.0 / 8.0) / u,
                (-10.0 / 8.0) / (u ** 3),
                (3.0 / 8.0) / (u ** 5))

    q = (3.0 * l + u) / 4.0
    r = (l + 3.0 * u) / 4.0
    E, old_E = inf, None
    while old_E is None or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l ** 3, l ** 5, 1.0],
            [q, q ** 3, q ** 5, -1.0],
            [r, r ** 3, r ** 5, 1.0],
            [u, u ** 3, u ** 5, -1.0],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        disc = 9.0 * b * b - 20.0 * a * c
        if disc < 0:
            break
        q_new = np.sqrt((-3.0 * b - sqrt(disc)) / (10.0 * c))
        r_new = np.sqrt((-3.0 * b + sqrt(disc)) / (10.0 * c))
        q, r = q_new, r_new
    return float(a), float(b), float(c)


def optimal_composition(l, num_iters, safety_factor_eps=0.0, cushion=0.0):
    """Precompute coefficient table for `num_iters` Polar Express steps.

    After calling this, iteration t of polar NS uses coefficients
    `coeffs[t]` applied to `p_t(x) = a x + b x³ + c x⁵`.
    """
    u = 1.0
    assert 0 <= l <= u
    safety_factor = 1.0 + safety_factor_eps
    coefficients = []
    for t in range(num_iters):
        lo = max(l, cushion * u)
        a, b, c = optimal_quintic(lo, u)
        if cushion * u > l:
            pl = a * l + b * l ** 3 + c * l ** 5
            pu = a * u + b * u ** 3 + c * u ** 5
            rescaler = 2.0 / (pl + pu)
            a *= rescaler
            b *= rescaler
            c *= rescaler
        if t < num_iters - 1:
            a /= safety_factor
            b /= safety_factor ** 3
            c /= safety_factor ** 5
        coefficients.append((a, b, c))
        l = a * l + b * l ** 3 + c * l ** 5
        u = 2.0 - l
    return coefficients


# Precomputed default tables (computed once at import).
POLAR_COEFFS_1E6 = optimal_composition(
    l=1e-6, num_iters=12, safety_factor_eps=1e-3, cushion=1e-3,
)
ORTH_COEFFS_1E4 = optimal_composition(
    l=1e-4, num_iters=9, safety_factor_eps=1e-3, cushion=1e-3,
)
ORTH_COEFFS_1E3 = ORTH_COEFFS_1E4

POLAR_TAIL_1QDWH = optimal_composition(
    l=0.01, num_iters=5, safety_factor_eps=1e-3, cushion=0.0,
)
POLAR_TAIL_2QDWH = optimal_composition(
    l=0.2, num_iters=3, safety_factor_eps=1e-3, cushion=0.0,
)
