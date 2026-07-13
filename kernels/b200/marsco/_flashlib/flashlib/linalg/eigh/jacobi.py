"""eigh_jacobi -- single-block row-cyclic Jacobi for small N (<= 128).

Thin user-facing wrapper over the Triton kernel in
:mod:`flashlib.linalg.eigh.triton.jacobi`. No CUDA / C++ compile step
on first call (the old ``jacobi_impl.py`` ``load_inline`` extension
was retired -- ninja is no longer required to import this backend).

Achieved residual is at the fp32 noise floor (``~1e-4`` for the
N=46 reference shape, identical to ``torch.linalg.eigh`` on fp32
input), and the kernel beats cuSOLVER ``syevd`` at N <= 16 where the
syevd launch fixed-cost dominates.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.eigh.triton.jacobi import triton_jacobi_eigh as _triton_jacobi


def eigh(A, num_sweeps: int = 6):
    """Single-kernel cyclic Jacobi eigensolver.

    Args:
        A: ``(N, N)`` symmetric float32 CUDA tensor; not modified.
        num_sweeps: full row-cyclic sweeps; each sweep performs
            ``N*(N-1)/2`` Givens rotations. ``6`` is enough to reach
            fp32 noise (~1e-4) for N <= 64; bump to ``8-12`` for
            tighter convergence at very large N.

    Returns:
        ``(w, V)`` -- ``(N,)`` eigenvalues (ascending) and ``(N, N)``
        eigenvectors as columns. ``A @ V[:, i] ≈ w[i] * V[:, i]``.
    """
    return _triton_jacobi(A, num_sweeps=num_sweeps)


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    N = shape[0] if isinstance(shape, (tuple, list)) else shape
    params = params or {}
    n_sweeps = params.get("num_sweeps", 6)
    # Sequential cyclic Jacobi: N*(N-1)/2 rotations per sweep, each
    # ~10 vector ops on N-length rows -> ~5 * N^2 fp32 ops per
    # rotation -> ~2.5 * n_sweeps * N^4 total.
    flops = int(n_sweeps * 2.5 * N ** 4)
    bytes_moved = N * N * 4 * 2  # A in + V out
    rt, bound = roofline(flops, bytes_moved, dtype, device, op_type="solver")
    return Estimate(
        op_name="eigh_jacobi",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * N * 4 * 2 / 1e9,
        bound=bound, confidence="heuristic", n_kernel_launches=1,
        suggested_config={"num_sweeps": n_sweeps},
        notes=[f"N={N}; single-CTA Triton cyclic Jacobi, {n_sweeps} sweeps."],
        expected_residual=1e-4, precision_tier="fast", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {"num_sweeps": 6}
