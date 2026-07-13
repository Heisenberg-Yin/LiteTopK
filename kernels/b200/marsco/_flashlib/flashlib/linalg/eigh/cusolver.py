"""eigh_cusolver — torch.linalg.eigh (cuSOLVER syevd). Reference precision."""
import torch

from flashlib.info.estimate import Estimate


def eigh(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh — cuSOLVER syevd. ~1e-7 residual."""
    return torch.linalg.eigh(A)


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Empirical model — cuSOLVER syevd is BLAS-2 bound; 512ms@8192."""
    N = shape[0] if isinstance(shape, (tuple, list)) else shape
    rt = 512.0 * (N / 8192) ** 3
    flops = int((8 / 3) * N ** 3)
    bytes_moved = N * N * 4 * 10
    return Estimate(
        op_name="eigh_cusolver",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=N * N * 4 * 4 / 1e9,
        bound="compute", confidence="measured", n_kernel_launches=1,
        notes=[f"N={N}; cuSOLVER syevd, fp32 BLAS-2 bound."],
        expected_residual=1e-7, precision_tier="exact", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {}
