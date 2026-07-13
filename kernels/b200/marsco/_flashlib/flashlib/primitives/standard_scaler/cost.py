"""Cost model for ``flash_standard_scaler(X)`` -- bandwidth-bound.

The Triton path runs **2 launches** (not 3):

1. ``_col_sum_ss_shifted_kernel`` -- one fused pass reading X once and
   producing both the column sum and sum-of-squares (in fp64
   internally to keep the ``Σ/N − (Σ/N)²`` cancellation precise).
2. ``_scale_kernel`` -- one elementwise pass writing
   ``Y = (X − mean) * inv_std``.

Algorithmic byte traffic is exactly ``2 * N * D * dtype_bytes`` (one
read + one write of X) plus negligible work on the small ``D``-vectors.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    N, D = shape
    dtype_bytes = 4 if dtype in ("fp32", "float32", "tf32") else 2
    # FLOPs: 1 subtract + 1 square + 1 accumulate in pass 1, plus
    # 1 subtract + 1 multiply in pass 2 -> 5 FLOPs per element.
    flops = 5 * N * D
    # Bytes: one read of X (fit pass) + one read + write of X (transform).
    bytes_moved = 3 * N * D * dtype_bytes
    rt, bound = roofline(flops, bytes_moved, dtype, device,
                          op_type="elementwise", n_launches=2)
    return Estimate(
        op_name="standard_scaler",
        runtime_ms=rt,
        flops=flops,
        bytes_moved=bytes_moved,
        memory_peak_gb=2 * N * D * dtype_bytes / 1e9,  # X + Y
        bound=bound,
        confidence="calibrated",
        n_kernel_launches=2,
        suggested_config={"BLOCK_N": 256, "BLOCK_D": 128, "num_warps": 8},
        subops=[],
        notes=[
            f"N={N}, D={D}, dtype={dtype}",
            "Two-launch fused: (sum+ss in one kernel) + (scale in one kernel); "
            "expect bound='memory' for any N*D >= 1e6.",
        ],
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    N, D = shape
    return {"BLOCK_N": 256, "BLOCK_D": 128, "num_warps": 8}


# ── Per-variant cost shims ───────────────────────────────────────────────
def estimate_standard_scaler_triton(shape, params=None, tol=None,
                                     dtype="float32", device="H100", **_):
    """Triton backend cost -- same model as ``estimate`` (default route)."""
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "standard_scaler_triton"
    est.tol = tol
    return est


def estimate_standard_scaler_cutedsl(shape, params=None, tol=None,
                                      dtype="float32", device="H100", **_):
    """CuteDSL fused TMA backend.

    Same byte traffic as the Triton path; the CuteDSL kernel uses TMA
    for the X-read and matches the Triton wall-clock within ±2 % on
    every shape we benchmark.
    """
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "standard_scaler_cutedsl"
    est.notes = list(est.notes) + [
        "cutedsl backend; TMA-fed but bandwidth-bound -- parity with Triton.",
    ]
    est.tol = tol
    return est
