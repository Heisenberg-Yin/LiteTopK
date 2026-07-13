"""ctypes wrapper for the GEMMul8 (Ozaki Scheme II) CUDA library.

GEMMul8 emulates SGEMM/DGEMM on INT8 (or FP8) tensor cores via the Chinese
Remainder Theorem. Each modular GEMM uses an INT32 accumulator (exact for
K * 127^2 < 2^31, i.e. K < ~130k), so precision scales linearly with the
``num_moduli`` parameter rather than being capped by the FP32 WGMMA
accumulator wall that limits ``bf16x3`` / ``tf32x6``.

Public entry: ``matmul_ozaki2(a, b, num_moduli=..., fastmode=...)``.

This module:
- dlopens ``fast_gemm/libgemmul8_shim.so`` (built by scripts/build_gemmul8.sh)
- caches a single cuBLAS handle per process
- grow-only workspace per (M_max, N_max, K_max, num_moduli) like the README
- exposes a presplit path so repeat calls with the same A or B reuse the
  internal INT8 representation (skip_scal{A,B}=True)

Convention note: cuBLAS is column-major. Our other paths follow PyTorch's
row-major convention with ``b`` shaped (N, K). To call cuBLAS DGEMM/SGEMM
without copies, we treat the row-major ``A: (M, K)`` as col-major ``A^T``
of shape (K, M), and similarly for B. The standard trick:

    C_row(M, N) = A_row(M, K) @ B_row(K, N)
                = (B_col(N, K) @ A_col(K, M))^T  in col-major

so we issue ``cublasGemm(op_A=N, op_B=N, m=N, n=M, k=K, A=B, B=A, C=C)``
treating the row-major buffers as their col-major equivalents. Result lands
in row-major C: (M, N).

Our existing fast_gemm convention has ``b`` as (N, K) (i.e. you're computing
``A @ B.T``). We accept that here too and transpose accordingly.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from threading import Lock
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_SHIM_PATH = _HERE / "libgemmul8_shim.so"


class GEMMul8NotBuilt(RuntimeError):
    pass


_lock = Lock()
_lib: Optional[ctypes.CDLL] = None


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is not None:
            return _lib
        if not _SHIM_PATH.exists():
            raise GEMMul8NotBuilt(
                f"libgemmul8_shim.so not found at {_SHIM_PATH}. "
                f"Build it with scripts/build_gemmul8.sh"
            )
        # cuBLAS / cuBLASLt / cudart need to be visible at dlopen time.
        # The shim was linked against the wheel libs; preload them so we
        # don't depend on ldconfig.
        nvidia_lib = Path("/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/lib")
        for soname in ("libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13"):
            so = nvidia_lib / soname
            if so.exists():
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        lib = ctypes.CDLL(str(_SHIM_PATH))
        _bind(lib)
        _lib = lib
        return lib


def _bind(lib: ctypes.CDLL) -> None:
    # workSize
    lib.gemmul8_shim_workSize.restype = ctypes.c_size_t
    lib.gemmul8_shim_workSize.argtypes = [
        ctypes.c_int,                                # is_complex
        ctypes.c_int,                                # backend (0=INT8, 1=FP8)
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,  # m, n, k
        ctypes.c_uint,                               # num_moduli
        ctypes.c_int, ctypes.c_int,                  # enable_skip_scalA, B
        ctypes.POINTER(ctypes.c_size_t),             # workSizeA out
        ctypes.POINTER(ctypes.c_size_t),             # workSizeB out
    ]

    # cublas handle helpers
    lib.gemmul8_shim_cublas_create.restype = ctypes.c_void_p
    lib.gemmul8_shim_cublas_create.argtypes = []
    lib.gemmul8_shim_cublas_destroy.restype = None
    lib.gemmul8_shim_cublas_destroy.argtypes = [ctypes.c_void_p]
    lib.gemmul8_shim_cublas_set_stream.restype = ctypes.c_int
    lib.gemmul8_shim_cublas_set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    # gemm
    lib.gemmul8_shim_gemm.restype = ctypes.c_int
    lib.gemmul8_shim_gemm.argtypes = [
        ctypes.c_void_p,                             # handle
        ctypes.c_int, ctypes.c_int,                  # dtype, backend
        ctypes.c_int, ctypes.c_int,                  # op_A, op_B
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,  # m, n, k
        ctypes.c_void_p,                             # alpha
        ctypes.c_void_p, ctypes.c_size_t,            # A, lda
        ctypes.c_void_p, ctypes.c_size_t,            # B, ldb
        ctypes.c_void_p,                             # beta
        ctypes.c_void_p, ctypes.c_size_t,            # C, ldc
        ctypes.c_uint, ctypes.c_int,                 # num_moduli, fastmode
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # work, workA, workB
        ctypes.c_int, ctypes.c_int,                  # enable_skip_scal A, B
        ctypes.c_int, ctypes.c_int,                  # skip_scal A, B
    ]


# ---- handle / workspace caches -----------------------------------------------

class _PerDeviceState:
    __slots__ = ("handle", "stream_id", "work", "workA", "workB",
                 "work_size", "workA_size", "workB_size")

    def __init__(self) -> None:
        self.handle: int = 0
        self.stream_id: Optional[int] = None
        self.work: Optional[torch.Tensor] = None
        self.workA: Optional[torch.Tensor] = None
        self.workB: Optional[torch.Tensor] = None
        self.work_size = 0
        self.workA_size = 0
        self.workB_size = 0


_state: dict[int, _PerDeviceState] = {}


def _get_state(device_index: int) -> _PerDeviceState:
    s = _state.get(device_index)
    if s is None:
        s = _PerDeviceState()
        lib = _load_lib()
        prev = torch.cuda.current_device()
        try:
            torch.cuda.set_device(device_index)
            s.handle = int(lib.gemmul8_shim_cublas_create())
            if s.handle == 0:
                raise RuntimeError("cublasCreate failed")
        finally:
            torch.cuda.set_device(prev)
        _state[device_index] = s
    return s


def _ensure_workspace(s: _PerDeviceState, device, need: int, needA: int, needB: int,
                      enable_skip_A: bool, enable_skip_B: bool) -> None:
    if need > s.work_size:
        s.work = torch.empty(need, dtype=torch.uint8, device=device)
        s.work_size = need
    if enable_skip_A and needA > s.workA_size:
        s.workA = torch.empty(needA, dtype=torch.uint8, device=device)
        s.workA_size = needA
    if enable_skip_B and needB > s.workB_size:
        s.workB = torch.empty(needB, dtype=torch.uint8, device=device)
        s.workB_size = needB


# ---- public entry ------------------------------------------------------------

# Limits per the GEMMul8 header.
_MAX_K = 1 << 17       # k <= 2^17
_MAX_S_DGEMM = 20
_MAX_S_SGEMM = 13


def matmul_ozaki2(
    a: torch.Tensor,
    b: torch.Tensor,
    num_moduli: int = 14,
    fastmode: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """C = A @ B^T via Ozaki Scheme II (INT8 backend, cuBLAS handle).

    Mirrors our existing fast_gemm convention:
      A: (M, K) float32 or float64
      B: (N, K) same dtype as A
      C: (M, N) same dtype as A

    For float64, ``num_moduli`` may be 2..20. For float32, 2..13.
    Larger ``num_moduli`` -> more bits of precision (~7 bits per modulus).
    ``fastmode=True`` drops the slowest modular reductions for a small
    precision hit (recommended for SGEMM-equivalent precision).
    """
    if a.dtype != b.dtype:
        raise TypeError(f"dtype mismatch: a={a.dtype} b={b.dtype}")
    if a.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"ozaki2 supports float32/float64, got {a.dtype}")
    if a.device != b.device or not a.is_cuda:
        raise ValueError("inputs must be on the same CUDA device")
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"shape mismatch a={tuple(a.shape)} b={tuple(b.shape)}")
    M, K = a.shape
    N = b.shape[0]
    if K > _MAX_K:
        raise ValueError(f"K={K} exceeds GEMMul8 limit 2^17={_MAX_K}")
    smax = _MAX_S_DGEMM if a.dtype == torch.float64 else _MAX_S_SGEMM
    if not (2 <= num_moduli <= smax):
        raise ValueError(f"num_moduli {num_moduli} out of [2, {smax}] for {a.dtype}")

    a = a.contiguous()
    b = b.contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    elif out.shape != (M, N) or out.dtype != a.dtype or out.device != a.device:
        raise ValueError("out tensor mismatched")

    lib = _load_lib()
    dev = a.device.index
    s = _get_state(dev)

    # Bind handle to the *current* torch stream for this device.
    stream_id = torch.cuda.current_stream(a.device).cuda_stream
    if stream_id != s.stream_id:
        rc = lib.gemmul8_shim_cublas_set_stream(s.handle, ctypes.c_void_p(stream_id))
        if rc != 0:
            raise RuntimeError(f"cublasSetStream failed: {rc}")
        s.stream_id = stream_id

    # Workspace.
    needA = ctypes.c_size_t(0)
    needB = ctypes.c_size_t(0)
    need = lib.gemmul8_shim_workSize(
        0, 0,                        # is_complex=0, backend=INT8
        M, N, K, num_moduli,
        0, 0,                        # enable_skip_scal A/B = 0 here
        ctypes.byref(needA), ctypes.byref(needB),
    )
    _ensure_workspace(s, a.device, int(need), int(needA.value), int(needB.value),
                      enable_skip_A=False, enable_skip_B=False)

    # cuBLAS column-major dispatch:
    # Row-major C(M,N) = A_row(M,K) @ B_row(N,K)^T
    # Treat row-major buffers as col-major:
    #   A_row(M,K)  is col-major (K,M) with ld=K
    #   B_row(N,K)  is col-major (K,N) with ld=K, but we need B^T so op_B = T
    #   C_row(M,N)  is col-major (N,M) with ld=N
    # Compute C^T_col(N,M) = B^T_col(N,K) @ A_col(K,M):
    #   m_cu = N, n_cu = M, k_cu = K
    #   op_A passed to shim corresponds to FIRST cuBLAS operand (B).
    #   op_B passed to shim corresponds to SECOND cuBLAS operand (A).
    # Row-major B(N,K) viewed col-major is (K,N); to get (N,K) for the cuBLAS
    # call we apply op = T -> (N,K). For A_row(M,K) viewed col-major is (K,M)
    # which is exactly what the second operand needs, so op = N.

    dtype = 0 if a.dtype == torch.float64 else 1
    if a.dtype == torch.float64:
        alpha = ctypes.c_double(1.0); beta = ctypes.c_double(0.0)
    else:
        alpha = ctypes.c_float(1.0); beta = ctypes.c_float(0.0)

    rc = lib.gemmul8_shim_gemm(
        ctypes.c_void_p(s.handle),
        dtype, 0,                             # backend INT8
        1, 0,                                 # op_A=T (B), op_B=N (A)
        N, M, K,                              # m_cu, n_cu, k_cu
        ctypes.byref(alpha),
        ctypes.c_void_p(b.data_ptr()), K,     # A_cu = B_row, lda = K
        ctypes.c_void_p(a.data_ptr()), K,     # B_cu = A_row, ldb = K
        ctypes.byref(beta),
        ctypes.c_void_p(out.data_ptr()), N,   # C_cu, ldc = N
        num_moduli, int(bool(fastmode)),
        ctypes.c_void_p(s.work.data_ptr()), 0, 0,
        0, 0, 0, 0,
    )
    if rc != 0:
        raise RuntimeError(f"gemmul8_shim_gemm returned {rc}")
    return out


# ---------- presplit / weight-cached path ------------------------------------

class Ozaki2Weight:
    """Cached preprocessed B for repeated A @ B.T calls.

    Holds the device-side INT8 representation of B inside the shim's workB
    buffer; subsequent calls skip the per-call scaling/splitting of B.
    Only valid while B's device buffer is unchanged.
    """

    def __init__(self, b: torch.Tensor, num_moduli: int, fastmode: bool):
        if b.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"unsupported dtype {b.dtype}")
        if not b.is_cuda or b.dim() != 2:
            raise ValueError("b must be 2D CUDA tensor")
        self.b = b.contiguous()
        self.num_moduli = num_moduli
        self.fastmode = fastmode
        self._first_call = True

    def matmul(self, a: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = self.b
        if a.dtype != b.dtype:
            raise TypeError("dtype mismatch")
        if a.shape[1] != b.shape[1]:
            raise ValueError("K mismatch")
        M, K = a.shape
        N = b.shape[0]
        if out is None:
            out = torch.empty((M, N), dtype=a.dtype, device=a.device)
        a = a.contiguous()

        lib = _load_lib()
        dev = a.device.index
        s = _get_state(dev)
        stream_id = torch.cuda.current_stream(a.device).cuda_stream
        if stream_id != s.stream_id:
            lib.gemmul8_shim_cublas_set_stream(s.handle, ctypes.c_void_p(stream_id))
            s.stream_id = stream_id

        needA = ctypes.c_size_t(0); needB = ctypes.c_size_t(0)
        need = lib.gemmul8_shim_workSize(
            0, 0, M, N, K, self.num_moduli,
            0, 1,
            ctypes.byref(needA), ctypes.byref(needB),
        )
        _ensure_workspace(s, a.device, int(need), int(needA.value), int(needB.value),
                          enable_skip_A=False, enable_skip_B=True)

        dtype = 0 if a.dtype == torch.float64 else 1
        if a.dtype == torch.float64:
            alpha = ctypes.c_double(1.0); beta = ctypes.c_double(0.0)
        else:
            alpha = ctypes.c_float(1.0); beta = ctypes.c_float(0.0)

        # Same column-major mapping as in matmul_ozaki2; but here B is the
        # weight that we want to skip-preprocess. In our cuBLAS call the
        # FIRST operand is B (op_A=T). We map enable_skip_scalA=True to
        # enable caching for B (the cuBLAS first operand) and pass the
        # workB tensor as workA in the shim.
        skip_now = 0 if self._first_call else 1
        rc = lib.gemmul8_shim_gemm(
            ctypes.c_void_p(s.handle),
            dtype, 0,
            1, 0,
            N, M, K,
            ctypes.byref(alpha),
            ctypes.c_void_p(b.data_ptr()), K,
            ctypes.c_void_p(a.data_ptr()), K,
            ctypes.byref(beta),
            ctypes.c_void_p(out.data_ptr()), N,
            self.num_moduli, int(bool(self.fastmode)),
            ctypes.c_void_p(s.work.data_ptr()),
            ctypes.c_void_p(s.workB.data_ptr()),  # workA arg <- workB tensor (B is cuBLAS first operand)
            0,
            1, 0,                                  # enable_skip_scalA=True, B=False
            skip_now, 0,                           # skip_scalA after first call
        )
        if rc != 0:
            raise RuntimeError(f"gemmul8_shim_gemm returned {rc}")
        self._first_call = False
        return out
