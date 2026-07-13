"""ab_gemm(A, B) -> A.T @ B for tall-skinny inputs sharing N dim."""
from flashlib.linalg.ab_gemm.triton import ab_gemm
from flashlib.linalg.ab_gemm import cost

__all__ = ["ab_gemm", "cost"]
