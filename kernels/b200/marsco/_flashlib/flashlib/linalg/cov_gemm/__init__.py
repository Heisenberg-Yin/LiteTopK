"""cov_gemm(X) -> X.T @ X, tall-skinny GEMM optimized for N >> D."""
from flashlib.linalg.cov_gemm.triton import cov_gemm, full_gemm
from flashlib.linalg.cov_gemm import cost

__all__ = ["cov_gemm", "full_gemm", "cost"]
