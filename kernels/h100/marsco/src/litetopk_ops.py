"""LiteTopk torch op 加载器：把 litetopk_select.cu + litetopk_torch.cu 编成 torch 扩展，
注册 torch.ops.litetopk.{topk_min,topk_max,sort_desc_}。

相比旧 ctypes 路线（libftk_fused.so + 手传 data_ptr/stream）：
  * op 直接吃 torch tensor，stream 由 dispatcher 自动取当前 CUDA stream；
  * 由 torch JIT 扩展编译，gcc>=9 满足 torch C++ 头要求（容器 gcc-12）。

环境要求（lagrange 容器）：CC=gcc-12 CXX=g++-12（镜像默认 clang20，nvcc 不支持）。
TORCH_CUDA_ARCH_LIST 缺省时按当前设备 capability 自动推断。
"""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

_LOADED = False


def _ensure_loaded():
    global _LOADED
    if _LOADED:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    load(
        name="litetopk",
        sources=[
            os.path.join(here, "litetopk_torch.cu"),
            os.path.join(here, "litetopk_select.cu"),
        ],
        extra_include_paths=[here],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr"],
        is_python_module=False,
        verbose=os.environ.get("FLASHTOPK_BUILD_VERBOSE") == "1",
    )
    _LOADED = True


@torch.no_grad()
def topk_min(scores: torch.Tensor, k: int):
    """对 [B, N] 每行精确取 k 个最小值及列索引。返回 (val[B,k], idx[B,k] int32)。"""
    _ensure_loaded()
    return torch.ops.litetopk.topk_min(scores, k)


@torch.no_grad()
def topk_max(scores: torch.Tensor, k: int):
    """对 [B, N] 每行精确取 k 个最大值及列索引。返回 (val[B,k], idx[B,k] int32)。"""
    _ensure_loaded()
    return torch.ops.litetopk.topk_max(scores, k)


@torch.no_grad()
def topk_select_thr(buf: torch.Tensor, origin: torch.Tensor, inv_delta: torch.Tensor,
                    th: torch.Tensor, qcount: torch.Tensor, k: int, num_buckets: int):
    """阈值复用选择：配合融合 _flat_kernel，跳过 sample/hist/threshold，只跑 gather+boundary。

    buf [R,BUF]（前 qcount[row] 为有效候选）；origin/inv_delta [R] float32；
    th/qcount [R] int32（收敛阈值桶 / 每行有效候选数）。
    返回 (val[R,k] 同 buf dtype, idx[R,k] int32，idx 为 buffer 内列位置 [0,BUF))。
    """
    _ensure_loaded()
    return torch.ops.litetopk.topk_select_thr(buf, origin, inv_delta, th, qcount, k, num_buckets)


@torch.no_grad()
def topk_select_thr_idx(buf: torch.Tensor, buf_idx: torch.Tensor,
                        origin: torch.Tensor, inv_delta: torch.Tensor,
                        th: torch.Tensor, qcount: torch.Tensor, k: int, num_buckets: int):
    """阈值复用选择，并在 CUDA kernel 内把 buffer 位置转换成最终 corpus id。
    buf_idx[:, :k] 已含 sample 段全局行号，其余为扫描段全局行号，统一从 buf_idx 取。"""
    _ensure_loaded()
    return torch.ops.litetopk.topk_select_thr_idx(
        buf, buf_idx, origin, inv_delta, th, qcount, k, num_buckets)


@torch.no_grad()
def topk_select_thr_mb_idx(buf: torch.Tensor, buf_idx: torch.Tensor,
                           origin: torch.Tensor, inv_delta: torch.Tensor,
                           th: torch.Tensor, qcount: torch.Tensor,
                           k: int, num_buckets: int, cap: int):
    """multi-block 版阈值复用选择：多 block 并行分桶（b<th→lt、b==th→eq）+ 单 block
    finalize（lt 全 copy + eq 上小 radix）。大 BUF 下显著缩短单 block 串行扫描。
    cap 为边界候选缓冲容量（>=k）。返回 (val[R,k], idx[R,k] int32 corpus id)。"""
    _ensure_loaded()
    return torch.ops.litetopk.topk_select_thr_mb_idx(
        buf, buf_idx, origin, inv_delta, th, qcount, k, num_buckets, cap)


@torch.no_grad()
def topk_select_thr_mb_dense(buf: torch.Tensor,
                             origin: torch.Tensor, inv_delta: torch.Tensor,
                             th: torch.Tensor, k: int, num_buckets: int, cap: int):
    """DENSE 变体：buf 为 [R,M] 稠密布局（buf[row,col]=score，col 即 corpus id），未通过
    gate 的位置为 sentinel(NaN)。无 buf_idx/qcount（id=col、扫满 M、isfinite 跳过 NaN）。
    复用 mb 的边界桶拆分 finalize。返回 (val[R,k], idx[R,k] int32 corpus id)。"""
    _ensure_loaded()
    return torch.ops.litetopk.topk_select_thr_mb_dense(
        buf, origin, inv_delta, th, k, num_buckets, cap)


@torch.no_grad()
def dense_compute_threshold(buf: torch.Tensor,
                            origin: torch.Tensor, inv_delta: torch.Tensor,
                            k: int, num_buckets: int):
    """DENSE 路径专用：对 [R,M] 稠密 buf（buf[row,col]=score）在 CUDA 内做
    SMEM 直方图 + per-row cumsum 求阈值桶，返回 th[R] (int32)。替代 Triton 端
    histogram/threshold，使 _flat 可写全量、零原子、无 sentinel memset。"""
    _ensure_loaded()
    return torch.ops.litetopk.dense_compute_threshold(
        buf, origin, inv_delta, k, num_buckets)


@torch.no_grad()
def topk_select_packed(buf_pack: torch.Tensor, segcnt: torch.Tensor,
                       cap: int, blk: int, m: int, k: int):
    """packed 分段 K-min select（fp16 专用）：直接读 4B packed buffer + segcnt，
    kernel 内 radix K-min + 解码 score/id。返回 (val[R,k] half, idx[R,k] int32 corpus id)。"""
    _ensure_loaded()
    return torch.ops.litetopk.topk_select_packed(buf_pack, segcnt, cap, blk, m, k)


@torch.no_grad()
def sort_desc_(val: torch.Tensor, idx: torch.Tensor):
    """就地把 [B, K] 的 (value, index) 按 value 降序排序。"""
    _ensure_loaded()
    torch.ops.litetopk.sort_desc_(val, idx)
