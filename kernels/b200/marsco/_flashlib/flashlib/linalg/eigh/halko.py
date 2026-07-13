"""Halko subspace iteration — randomized truncated eigendecomposition.

Top-K eigenpairs of a symmetric PSD matrix via power iteration on a random
sketch (Halko, Martinsson & Tropp 2011, "Finding structure with randomness").

This is a **TRUNCATED eigh** — a different operating point from the full
exact eigh paths (cuSOLVER, Jacobi, QDWH):

* Full eigh   : O(N³) work, returns ALL N eigenpairs, residual ~1e-7.
* Halko (this): O(N² K) work, returns top K eigenpairs, residual depends
                on n_iter (~1e-4 at n_iter=5, ~1e-6 at n_iter=10).

The cost is dominated by ``n_iter + 1`` GEMMs of shape (N, N) × (N, q)
where ``q = K + p`` (oversample) and a Rayleigh-Ritz on the q × q sketch.
For ``q ≪ N`` this is **orders of magnitude faster** than the direct eigh:

    M=5000,  K=64:  cuSOLVER 157 ms vs Halko ~6 ms (~26×)
    M=10000, K=64:  cuSOLVER 737 ms vs Halko ~9 ms (~80×)
    M=1024,  K=256: cuSOLVER  10 ms vs Halko ~10 ms (≈ 1×, no win)

The dispatcher in ``flashlib.linalg.eigh.eigh(A, K=K)`` automatically
routes to this path whenever ``should_use_halko(N, K)`` returns True.

Algorithm (q = K + p oversample, n_iter ≥ 4):

    Q = randn(N, q)             # Gaussian sketch
    Q = G @ Q                   # cuBLAS TF32 GEMM, dominant cost
    Q, _ = qr(Q)                # tall-skinny QR
    for _ in range(n_iter):
        Q = G @ Q
        Q, _ = qr(Q)            # power iter + re-orthonormalize
    H = Qᵀ G Q                  # Rayleigh-Ritz (q × q)
    eigvals, V = eigh(H)        # exact eigh on the small projection
    eigvecs_full = Q V[:, :K]
"""
from __future__ import annotations

import torch


def halko_eigh(
    G: torch.Tensor,
    K: int,
    *,
    n_iter: int = 5,
    p: int = 30,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-K eigenpairs of a symmetric PSD matrix ``G`` via subspace iteration.

    Args:
        G: (M, M) symmetric PSD CUDA tensor (e.g. covariance or Gram matrix).
        K: number of top eigenpairs to return.
        n_iter: power iterations after the initial GQ multiply (default 5).
            Higher → tighter residual at the cost of more GEMMs.
        p: oversample. Higher → more accurate top-K (default 30, ample
            for typical PSD spectra).
        seed: RNG seed for the Gaussian sketch (deterministic across runs).

    Returns:
        eigvals: (K,) ascending top-K eigenvalues — matches the
            ``torch.linalg.eigh`` convention used elsewhere in flashlib.
        eigvecs: (M, K) corresponding eigenvectors (columns, ascending).
    """
    M = G.shape[0]
    q = min(K + p, M)

    gen = torch.Generator(device=G.device).manual_seed(seed)
    Q = torch.randn(M, q, device=G.device, dtype=G.dtype, generator=gen)

    Q = G @ Q
    Q, _ = torch.linalg.qr(Q)
    for _ in range(n_iter):
        Q = G @ Q
        Q, _ = torch.linalg.qr(Q)

    H = Q.T @ (G @ Q)
    H = 0.5 * (H + H.T)
    eigvals, V = torch.linalg.eigh(H)

    top_eigvals = eigvals[-K:]
    eigvecs = Q @ V[:, -K:]
    return top_eigvals, eigvecs


def should_use_halko(N: int, K: int, *, min_N: int = 256) -> bool:
    """Internal heuristic -- Halko helps when ``K`` is much smaller than
    ``N`` and ``N`` is large enough that the direct eigh has measurable
    cost. Library-internal: callers should let :func:`flashlib.linalg.eigh.eigh`
    decide via ``tol`` instead of reading this directly.

    Two conditions to apply Halko:

    1. ``K * 4 < N`` — accuracy regime (q = K + 30 stays well below N).
    2. ``N >= min_N`` (default 256) — speedup regime (direct eigh is
       visible on the wallclock at all).

    Empirical (H200, GPU 4):

        N=5000,  K=64:    cuSOLVER 157 ms vs Halko 6 ms → 26×
        N=10000, K=64:    cuSOLVER 737 ms vs Halko 9 ms → 80×
        N=1024,  K=256:   cuSOLVER 10 ms vs Halko ~10 ms → ≈ 1×
        N ≤ 256:          cuSOLVER eigh < 1 ms regardless — Halko's
                          ~1e-2 K-th eigvec approximation noise is not
                          worth it without a wall-time win.
    """
    return K * 4 < N and N >= min_N


# ── Cost model ─────────────────────────────────────────────────────────


def estimate(shape, params=None, tol=None, dtype="float32",
             device="H100", **_):
    """Cost of Halko on shape ``(N, N)`` with ``params['K']`` truncation.

    Dominant term: (n_iter + 1) GEMMs of (N, N) × (N, q) where q = K + p.
    """
    from flashlib.info.estimate import Estimate
    from flashlib.info.roofline import roofline

    N = shape[0] if isinstance(shape, (tuple, list)) else shape
    params = params or {}
    K = params.get("K", min(32, N // 4))
    n_iter = params.get("n_iter", 5)
    p = params.get("p", 30)
    q = min(K + p, N)

    # (n_iter + 1) GEMMs of (N, N) × (N, q) → 2·N²·q FLOPs each.
    gemm_flops = (n_iter + 1) * 2 * N * N * q
    # Plus QR(N, q) per iter → ~4 N q² FLOPs each, negligible for q ≪ N.
    qr_flops = (n_iter + 1) * 4 * N * q * q
    # Rayleigh-Ritz: Qᵀ G Q (2 GEMMs of small q × N × N → 2·N²·q each) + eigh(q).
    rr_flops = 2 * (2 * N * N * q) + 9 * q ** 3
    flops = gemm_flops + qr_flops + rr_flops

    bytes_moved = (n_iter + 1) * (N * N + 2 * N * q) * 4
    rt, bound = roofline(flops, bytes_moved, "fp32", device, op_type="gemm")

    # Halko residual scales as σ_{K+1}/σ_K and (K/(K+p))^(2 n_iter+1).
    # Empirical default at n_iter=5, p=30 lands at ~1e-4 to 1e-6 depending
    # on spectrum decay. Report the conservative end of that range.
    residual = max(1e-7, 10 ** -(0.5 * n_iter + 1))

    return Estimate(
        op_name="eigh_halko",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(N * N + N * q) * 4 / 1e9,
        bound=bound, confidence="measured",
        n_kernel_launches=2 * (n_iter + 1) + 4,
        suggested_config={"K": K, "n_iter": n_iter, "p": p},
        notes=[
            f"N={N}, K={K}, n_iter={n_iter}, p={p}, q={q}",
            f"Halko subspace iteration; (n_iter+1)={n_iter+1} GEMMs of (N,N)·(N,q).",
            "Wins vs cuSOLVER when K*4 < N and N >= 256; "
            "~80× at N=10K K=64, ~26× at N=5K K=64.",
        ],
        expected_residual=residual, precision_tier="mixed", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32",
              device="H100", **_):
    N = shape[0] if isinstance(shape, (tuple, list)) else shape
    params = params or {}
    K = params.get("K", min(32, N // 4))
    return {"variant": "eigh_halko", "K": K, "n_iter": 5, "p": 30}
