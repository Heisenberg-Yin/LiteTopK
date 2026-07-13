"""Precomputed CRT constants for Ozaki Scheme II.

Picks the largest s coprime odd integers <= 254 and computes the CRT
reconstruction coefficients y_t / m_t (as FP64) per modulus, plus log2(M).

CRT identity used for reconstruction (per output element):
    Let M = prod_t m_t, M_t = M / m_t, y_t = M_t^{-1} mod m_t.
    Given C_t = (true_C mod m_t) for each t, the unique integer in
    [-M/2, M/2] congruent to true_C mod M is reconstructed as:

        x      = sum_t C_t * (y_t / m_t)             # FP64, no mod yet
        frac   = x - round(x)                        # in [-1/2, 1/2]
        true_C = frac * M                            # integer in [-M/2, M/2]

    The y_t / m_t coefficients are exactly representable in FP64 (small
    rationals). The only FP-imprecise step is `frac * M` when M > 2^53,
    so we expose log2(M) and a high/low split so callers can apply the
    inverse scaling losslessly. For s <= 7 (M < 2^55) this is harmless.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Tuple

import torch


# Largest 20 odd primes <= 254 (sorted descending). Pairwise coprime.
# Chosen to maximize log2(M) per modulus while staying <= 254 so that
# A_t = A' mod m_t fits in INT8 (|A_t| <= 127).
_MODULI_TABLE: List[int] = [
    251, 241, 239, 233, 229, 227, 223, 211,
    199, 197, 193, 191, 181, 179, 173, 167,
    163, 157, 151, 149,
]


def get_moduli(num_moduli: int) -> List[int]:
    if num_moduli < 2 or num_moduli > len(_MODULI_TABLE):
        raise ValueError(
            f"num_moduli {num_moduli} out of [2, {len(_MODULI_TABLE)}]"
        )
    return _MODULI_TABLE[:num_moduli]


@lru_cache(maxsize=32)
def crt_constants(num_moduli: int) -> Tuple[List[int], int, List[float], List[int]]:
    """Return (moduli, M, alphas, ys).

    - moduli[t] : the t-th modulus
    - M         : product of all moduli (Python big-int, may exceed 2^63)
    - alphas[t] : y_t / m_t in FP64 (exact small fraction)
    - ys[t]     : the integer y_t = (M / m_t)^{-1} mod m_t, in [0, m_t)
    """
    moduli = get_moduli(num_moduli)
    M = 1
    for m in moduli:
        M *= m
    alphas: List[float] = []
    ys: List[int] = []
    for m in moduli:
        Mt = M // m  # Python big-int
        y = pow(Mt % m, -1, m)  # in [1, m)
        ys.append(y)
        alphas.append(y / m)
    return moduli, M, alphas, ys


def crt_constants_tensor(num_moduli: int, device, dtype=torch.float64
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                     float, float]:
    """Triton-friendly form.

    Returns:
        moduli_t : (s,) int32   — the moduli themselves
        ys_t     : (s,) int32   — y_t (modular inverses, used for exact recon)
        alphas_t : (s,) FP64    — y_t / m_t (kept for completeness / legacy)
        M_hi     : Python float — high part of M (top 53 bits, may equal M if small)
        M_lo     : Python float — residual (M - M_hi), so M = M_hi + M_lo exactly
    """
    moduli, M, alphas, ys = crt_constants(num_moduli)
    moduli_t = torch.tensor(moduli, dtype=torch.int32, device=device)
    ys_t = torch.tensor(ys, dtype=torch.int32, device=device)
    alphas_t = torch.tensor(alphas, dtype=dtype, device=device)
    M_hi = float(M)              # nearest FP64 to M
    M_lo = float(M - int(M_hi))  # residual; for M < 2^53 this is 0
    return moduli_t, ys_t, alphas_t, M_hi, M_lo


# ---- Scaling parameters ----------------------------------------------------

def choose_kA_kB(num_moduli: int, K: int) -> Tuple[int, int]:
    """Pick fixed-point bit budgets so the integer C = A_int @ B_int.T
    satisfies the CRT recovery condition |C| < M/2.

    Bound: K * 2^(kA + kB) < M/2  =>  kA + kB < log2(M/(2K)).
    We split evenly between A and B and back off by 2 bits as a safety
    margin (rounding + per-row scaling slack).
    """
    moduli, M, _, _ = crt_constants(num_moduli)
    log2M = math.log2(M)
    budget = log2M - math.log2(K) - 1.0 - 2.0  # 2-bit slack
    kA = max(8, int(budget // 2))
    kB = max(8, int(budget - kA))
    return kA, kB
