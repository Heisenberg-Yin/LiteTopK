"""Fused KNN top-k — single-pass fused operator (no score writeback).

融合算子 = 两份参考实现的结合：
  * 打分 + 单遍分桶 + 流式阈值 + flat buffer  ←  flashlargek.py（Triton）
  * 从有界 buffer 里精确选 top-k             ←  litetopk_select.cu（CUDA）

全融合：**不写回 score 矩阵**。一遍扫语料，在寄存器里算 score=‖c‖²−2⟨x,c⟩，
立刻分桶 + 流式阈值（atomic_min gate，只会收紧），只把通过 gate 的候选 append 到
每个 query 的有界 buffer [CHUNK, BUF=10·k]（和 flashlib 一样不物化 N×M）。最终用
CUDA litetopk_select 在小 buffer [R, w] 上精确取 k 个最小。

正确性：当 count_at_T ≲ BUF（buffer 不溢出）时精确；BUF=10·k 在真实分布下足够
（flashlargek 注释：真实 IP ~1.8×k、randn ~3.3×k）。溢出时退化为近似（丢高桶候选）。

入口：fused_knn_topk_l2 / fused_knn_topk_ip。
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from _common import _next_pow2
import litetopk_ops

_NUM_BUCKETS = int(os.environ.get("FLASHTOPK_NUM_BUCKETS", "64"))
_BUCKET_EPS = 1.0e-20

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _tl_dtype_of(dt: torch.dtype):
    """torch dtype -> triton 标量类型（仅支持 fp32 / fp16 / bf16）。"""
    if dt == torch.float16:
        return tl.float16
    if dt == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


# ─────────────────────────── CUDA 扩展（torch op，懒加载）───────────────────────────
#
# litetopk_select.cu 通过 litetopk_torch.cu 注册为 torch.ops.litetopk.*（gcc-12 满足 torch C++ 头要求）。
# op 直接吃 torch tensor，stream 由 dispatcher 自动取当前 CUDA stream。


def _load_lib():
    """编译/加载 litetopk torch 扩展（懒加载，幂等）。"""
    litetopk_ops._ensure_loaded()


@torch.no_grad()
def _cuda_flash_topk_min(scores: torch.Tensor, k: int):
    """对 [B, N] 的 score 矩阵每行精确取 k 个最小值及列索引（torch.ops.litetopk.topk_min）。"""
    return litetopk_ops.topk_min(scores, k)


@torch.no_grad()
def _cuda_sort_desc(val: torch.Tensor, idx: torch.Tensor):
    """就地把 [B, K] 的 (value, index) 按 value 降序排序。"""
    litetopk_ops.sort_desc_(val, idx)


def _sample_size_for(M: int, k: int) -> int:
    if os.environ.get("FLASHTOPK_SAMPLE_EQ_K", "0") == "1":
        return min(M, k)
    floor = int(os.environ.get("FLASHTOPK_SAMPLE_FLOOR", "32768"))
    return min(M, max(k, floor))


# SMEM 设备参数（默认按 L40S 标定，可经环境变量覆盖以适配 H200 等）。
#   optin   = sharedMemPerBlockOptin（单 block 能 opt-in 的上限）
#   per_sm  = sharedMemPerMultiprocessor（单 SM 总量，决定按 SMEM 能常驻几个 block）
_SMEM_OPTIN = int(os.environ.get("FLASHTOPK_SMEM_OPTIN", "101376"))
_SMEM_PER_SM = int(os.environ.get("FLASHTOPK_SMEM_PER_SM", "102400"))
# 带宽随常驻 CTA 数饱和：~2 个常驻块即可基本掩盖访存延迟，再多收益递减。
_BW_SAT = int(os.environ.get("FLASHTOPK_BW_SAT", "2"))


def _heuristic_config(*, N: int, M: int, D: int, bytes_per: int = 4) -> dict:
    # 语料 c 每个 query-chunk 完整重读一遍，重复读次数 = ceil(N/BN)，是带宽瓶颈下的主杠杆。
    # 但 BN 也是 matmul tile 高度，撑大 _flat_kernel 的 SMEM 占用，BN 越大 → 每 SM 能常驻
    # 的 block 数（ctas）越少 → 访存延迟掩盖变差 → 有效带宽下降。两股力拉锯，用代价模型选 BN：
    #   est_smem(BN) = (BN + BM)·D_INNER·bytes·2     （x_tile + c_tile，double-buffer）
    #   ctas(BN)     = per_sm // est_smem            （按 SMEM 能常驻的 block 数）
    #   cost(BN)     = ceil(N/BN) / min(ctas, BW_SAT)（重复读量 / 有效带宽）
    # 取 cost 最小者；并列时取【较大】BN —— 实测 fp16 下大 tile 的 tensor-core MMA 效率收益
    # 远超它牺牲的 occupancy（BN=256 vs 128 在 bandwidth cost 平局时实测快 ~1.5×、追平 cuBLAS）。
    # fp32 因 256 放不下 SMEM 被过滤，自然落 128；fp16 SMEM 减半放得下 256，tie-break 取大者
    # 即落到 256，这是 fp16 拿回算力红利的关键。SMEM 更大的卡（H200）会自动用到更大 tile。
    BM = 64 if D >= 256 else 128
    D_INNER = _next_pow2(D) if D <= 64 else 64
    _di_env = os.environ.get("FLASHTOPK_D_INNER")
    if _di_env:
        D_INNER = int(_di_env)

    def est_smem(bn: int) -> int:
        return (bn + BM) * D_INNER * bytes_per * 2

    cap = max(16, _next_pow2(N))  # BN 再大只是空算 masked 行
    cands = [bn for bn in (16, 32, 64, 128, 256, 512)
             if bn <= cap and est_smem(bn) <= _SMEM_OPTIN]
    if not cands:
        cands = [16]

    def cost(bn: int) -> float:
        ctas = max(1, _SMEM_PER_SM // est_smem(bn))
        return math.ceil(N / bn) / min(ctas, _BW_SAT)

    BN = min(cands, key=lambda bn: (cost(bn), -bn))

    # 大 tile 需要更多 warp 喂满 tensor-core / 掩盖访存；BN>=256 用 16 warp（实测最优），否则 8。
    default_warps = 16 if BN >= 256 else 8
    num_warps = int(os.environ.get("FLASHTOPK_NUM_WARPS", str(default_warps)))
    BM = int(os.environ.get("FLASHTOPK_BM", str(BM)))
    BN = int(os.environ.get("FLASHTOPK_BN", str(BN)))
    return dict(BN=BN, BM=BM, D_INNER=D_INNER, num_warps=num_warps)


# ─────────────────────────── sample 打分 kernel（L2 / IP）───────────────────────────


@triton.jit
def _sample_scores_kernel(
    x_ptr, c_ptr, score_ptr,
    stride_x_n, stride_x_d, stride_c_m, stride_c_d, stride_s_n, stride_s_m,
    N: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr, D_INNER: tl.constexpr,
    IS_L2: tl.constexpr, USE_TF32: tl.constexpr, OUT_DT: tl.constexpr,
    ACC_DT: tl.constexpr = tl.float32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_offs = pid_n * BN + tl.arange(0, BN)
    n_mask = n_offs < N
    m_offs = pid_m * BM + tl.arange(0, BM)
    m_mask = m_offs < M

    if D_INNER >= D:
        d_offs = tl.arange(0, D_INNER)
        d_mask = d_offs < D
        x_tile = tl.load(x_ptr + n_offs[:, None] * stride_x_n + d_offs[None, :] * stride_x_d,
                         mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        c_tile = tl.load(c_ptr + m_offs[:, None] * stride_c_m + d_offs[None, :] * stride_c_d,
                         mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        cross = tl.dot(x_tile, tl.trans(c_tile), allow_tf32=USE_TF32, out_dtype=ACC_DT)
        if IS_L2:
            c_f = c_tile.to(ACC_DT)
            csq = tl.sum(c_f * c_f, axis=1)
    else:
        cross = tl.zeros([BN, BM], dtype=ACC_DT)
        if IS_L2:
            csq = tl.zeros([BM], dtype=ACC_DT)
        for d_start in range(0, D, D_INNER):
            d_offs = d_start + tl.arange(0, D_INNER)
            d_mask = d_offs < D
            x_sub = tl.load(x_ptr + n_offs[:, None] * stride_x_n + d_offs[None, :] * stride_x_d,
                            mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            c_sub = tl.load(c_ptr + m_offs[:, None] * stride_c_m + d_offs[None, :] * stride_c_d,
                            mask=m_mask[:, None] & d_mask[None, :], other=0.0)
            cross += tl.dot(x_sub, tl.trans(c_sub), allow_tf32=USE_TF32, out_dtype=ACC_DT)
            if IS_L2:
                c_f = c_sub.to(ACC_DT)
                csq += tl.sum(c_f * c_f, axis=1)

    # 默认 ACC_DT=fp32 与 torch 对齐（cuBLAS half GEMM 也 fp32 累加）；ACC_DT=fp16 时启用
    # tensor-core fp16 累加（更快但精度更低）。score 算出即 cast 到 OUT_DT。
    score = (csq[None, :] - 2.0 * cross) if IS_L2 else (-cross)
    score = score.to(OUT_DT)
    tl.store(score_ptr + n_offs[:, None] * stride_s_n + m_offs[None, :] * stride_s_m,
             score, mask=n_mask[:, None] & m_mask[None, :])


@torch.no_grad()
def _sample_range_topk(
    xb, c, k, *, is_l2, use_tf32, sample_size,
    sample_val_out, sample_count_out=None, sample_idx_out=None,
    bufidx_out=None, acc_dt=tl.float32,
):
    """Sample 一段语料的 score，取样本 top-k 作为 [origin, k-th] 桶范围，并把样本
    top-k 候选写进 flat buffer（后续主 kernel 只需扫 c[sample_size:] 的尾部）。"""
    M = c.shape[0]
    S = sample_size
    R, D = xb.shape
    cfg = _heuristic_config(N=R, M=S, D=D)
    BN, BM, D_INNER = cfg["BN"], cfg["BM"], cfg["D_INNER"]
    grid = (math.ceil(S / BM), math.ceil(R / BN))

    sample_score = torch.empty((R, S), device=xb.device, dtype=xb.dtype)
    out_dt = _tl_dtype_of(xb.dtype)
    _sample_scores_kernel[grid](
        xb, c, sample_score,
        xb.stride(0), xb.stride(1), c.stride(0), c.stride(1),
        sample_score.stride(0), sample_score.stride(1),
        N=R, M=S, D=D, BN=BN, BM=BM, D_INNER=D_INNER,
        IS_L2=is_l2, USE_TF32=use_tf32, OUT_DT=out_dt, ACC_DT=acc_dt,
        num_warps=cfg["num_warps"],
    )
    if k < S:
        # litetopk_select 取每行 k 个最小，替代 torch.topk：免排序、单遍分桶。
        top_vals, top_idxs = _cuda_flash_topk_min(sample_score, k)
    else:
        top_vals = sample_score
        top_idxs = torch.arange(S, device=xb.device, dtype=torch.int32)[None, :].expand(R, S)

    # 桶的 origin/span 是整数桶号的换算系数（非返回 score），用 fp32 计算以免 fp16 span 下溢/
    # inv_delta 溢出 inf；top_vals/buffer 仍是 OUT_DT（与 torch 一致的 fp16 舍入）。
    # FLASHTOPK_FP16_BUCKET=1 时改用输入 dtype（fp16）算桶坐标，用于实测精度/延迟代价对照。
    _fp16_bucket = os.environ.get("FLASHTOPK_FP16_BUCKET", "0") == "1"
    tv32 = top_vals if _fp16_bucket else top_vals.float()
    bucket_origin = tv32.amin(dim=-1).contiguous()
    sample_hi = tv32.amax(dim=-1)
    sample_val_out[:, :k].copy_(top_vals)
    if sample_idx_out is not None:
        sample_idx_out[:, :k].copy_(top_idxs)
    # sample 段取 c[:S]，故 top_idxs（样本内局部位置 [0,S)）本身即全局 corpus 行号。
    # 把它直接写进 buffer 前 k 槽的 buf_idx，select 端即可对所有槽统一用 buf_idx，
    # 无需 sample_idx 旁路 + "前 K 槽即 sample 候选" 的脆弱假设。
    if bufidx_out is not None:
        bufidx_out[:, :k].copy_(top_idxs)
    sample_count_out.fill_(k)
    eps = 6e-5 if _fp16_bucket else _BUCKET_EPS  # fp16 最小正规数量级，避免下溢成 0
    span = torch.clamp(sample_hi - bucket_origin, min=eps)
    bucket_inv_delta = ((_NUM_BUCKETS - 1) / span).contiguous()
    if sample_idx_out is not None:
        return bucket_origin, bucket_inv_delta
    return bucket_origin, bucket_inv_delta, top_idxs


# ─────────────────────────── 全融合主 kernel（单遍，不写回 score）───────────────────────────


@triton.jit
def _flat_kernel(
    x_ptr, c_ptr, origin_ptr, inv_delta_ptr,
    bcount_ptr, threshold_ptr, wcount_ptr, qcount_ptr, buf_val_ptr, buf_idx_ptr,
    stride_x_n, stride_x_d, stride_c_m, stride_c_d, stride_o_n,
    stride_bc_n, stride_bc_k, stride_t_n, stride_q_n,
    stride_bv_n, stride_bv_p, stride_bi_n, stride_bi_p,
    N: tl.constexpr, M: tl.constexpr, D: tl.constexpr, K: tl.constexpr, BUF: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr, D_INNER: tl.constexpr, NUM_BUCKETS: tl.constexpr,
    IS_L2: tl.constexpr, USE_TF32: tl.constexpr, OUT_DT: tl.constexpr,
    CNT_EVERY: tl.constexpr = 1, M_START: tl.constexpr = 0,
    ACC_DT: tl.constexpr = tl.float32, ABLATE: tl.constexpr = 0,
    DENSE: tl.constexpr = 0, SKIP_HIST: tl.constexpr = 0,
    SKIP_THR: tl.constexpr = 0, SKIP_WRITE: tl.constexpr = 0,
    GEMM_ONLY: tl.constexpr = 0, THR_PER_CTA: tl.constexpr = 0,
):
    """一遍 [BN×BM]：寄存器算 score → 分桶 → 直方图 + 流式阈值 → append 通过项到 flat buffer。

    score 永不写回 HBM。仅通过 gate 的候选落入 buffer[row, qpos]。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_offs = pid_n * BN + tl.arange(0, BN)
    n_mask = n_offs < N
    m_offs = M_START + pid_m * BM + tl.arange(0, BM)
    m_mask = m_offs < M

    # block_ptr 版 load：移除 per-element 指针算术与 mask，让 Triton 在 SM90 上有机会
    # lower 成 TMA（异步批量拷贝），并配合 WGMMA。boundary_check + padding "zero" 等价
    # 于原来的 other=0.0。order=(1,0) 表示 D 维连续（行主序 [行, D]）。
    x_bp = tl.make_block_ptr(
        base=x_ptr, shape=(N, D), strides=(stride_x_n, stride_x_d),
        offsets=(pid_n * BN, 0), block_shape=(BN, D_INNER), order=(1, 0))
    c_bp = tl.make_block_ptr(
        base=c_ptr, shape=(M, D), strides=(stride_c_m, stride_c_d),
        offsets=(M_START + pid_m * BM, 0), block_shape=(BM, D_INNER), order=(1, 0))

    if D_INNER >= D:
        x_tile = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
        c_tile = tl.load(c_bp, boundary_check=(0, 1), padding_option="zero")
        cross = tl.dot(x_tile, tl.trans(c_tile), allow_tf32=USE_TF32, out_dtype=ACC_DT)
        if IS_L2:
            c_f = c_tile.to(ACC_DT)
            csq = tl.sum(c_f * c_f, axis=1)
    else:
        cross = tl.zeros([BN, BM], dtype=ACC_DT)
        if IS_L2:
            csq = tl.zeros([BM], dtype=ACC_DT)
        for d_start in range(0, D, D_INNER):
            x_sub = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
            c_sub = tl.load(c_bp, boundary_check=(0, 1), padding_option="zero")
            cross += tl.dot(x_sub, tl.trans(c_sub), allow_tf32=USE_TF32, out_dtype=ACC_DT)
            if IS_L2:
                c_f = c_sub.to(ACC_DT)
                csq += tl.sum(c_f * c_f, axis=1)
            x_bp = tl.advance(x_bp, (0, D_INNER))
            c_bp = tl.advance(c_bp, (0, D_INNER))

    # 默认 ACC_DT=fp32 与 torch 对齐；ACC_DT=fp16 启用 tensor-core fp16 累加（更快、精度更低）。
    score = (csq[None, :] - 2.0 * cross) if IS_L2 else (-cross)
    score = score.to(OUT_DT)
    if GEMM_ONLY:
        # 仅锚定 GEMM 存活：把每行 score 之和写 buffer 首列，跳过分桶/直方图/阈值/全量写。
        # 用于隔离 GEMM 真实耗时（避免下游全删被编译器 DCE）。
        rsum = tl.sum(score.to(tl.float32), axis=1)
        tl.store(buf_val_ptr + n_offs * stride_bv_n, rsum.to(score.dtype), mask=n_mask)
        return
    origin = tl.load(origin_ptr + n_offs * stride_o_n, mask=n_mask, other=0.0)
    inv_delta = tl.load(inv_delta_ptr + n_offs * stride_o_n, mask=n_mask, other=0.0)
    # 桶号是整数索引（非 score），用 fp32 算以免 fp16 溢出 / inf；输入是已 fp16 舍入的 score。
    bucket_f = (score.to(tl.float32) - origin[:, None].to(tl.float32)) * inv_delta[:, None].to(tl.float32)
    bucket_raw = bucket_f.to(tl.int32)
    in_range = bucket_raw < NUM_BUCKETS
    bucket_i = tl.minimum(tl.maximum(bucket_raw, 0), NUM_BUCKETS - 1)
    valid = n_mask[:, None] & m_mask[None, :] & in_range

    bc_off = n_offs[:, None] * stride_bc_n + bucket_i * stride_bc_k
    if ABLATE < 2 and SKIP_HIST == 0:
        tl.atomic_add(bcount_ptr + bc_off, 1, sem="relaxed", mask=valid)

    th_off = n_offs * stride_t_n
    gate = tl.load(threshold_ptr + th_off, mask=n_mask, other=NUM_BUCKETS - 1).to(tl.int32)
    if ABLATE < 2 and SKIP_THR == 0:
        # Fast path for the multi-CTA select pipeline: each score CTA updates the
        # row threshold once from the current global histogram, then uses that
        # CTA-local gate for its own buffer write.  The threshold only tightens
        # via atomic_min.  A partial histogram can make the gate loose or prove
        # that K items already exist in lower buckets; it cannot exclude a true
        # top-K item because buckets are monotone in score.
        if THR_PER_CTA:
            bkt = tl.arange(0, NUM_BUCKETS)
            hrow = tl.load(bcount_ptr + n_offs[:, None] * stride_bc_n + bkt[None, :] * stride_bc_k,
                           mask=n_mask[:, None], other=0).to(tl.int32)
            cum = tl.cumsum(hrow, axis=1)
            cand_b = tl.where(cum >= K, bkt[None, :], NUM_BUCKETS)
            my_thr = tl.minimum(tl.min(cand_b, axis=1), NUM_BUCKETS - 1).to(tl.int32)
            tl.atomic_min(threshold_ptr + th_off, my_thr, sem="relaxed", mask=n_mask)
            gate = tl.minimum(gate, my_thr)
        else:
            pass_mask0 = valid & (bucket_i <= gate[:, None])
            my_wr = tl.sum(pass_mask0.to(tl.int32))
            prev = tl.atomic_add(wcount_ptr, my_wr)
            if (prev // CNT_EVERY) != ((prev + my_wr) // CNT_EVERY):
                bkt = tl.arange(0, NUM_BUCKETS)
                hrow = tl.load(bcount_ptr + n_offs[:, None] * stride_bc_n + bkt[None, :] * stride_bc_k,
                               mask=n_mask[:, None], other=0).to(tl.int32)
                cum = tl.cumsum(hrow, axis=1)
                cand_b = tl.where(cum >= K, bkt[None, :], NUM_BUCKETS)
                my_thr = tl.minimum(tl.min(cand_b, axis=1), NUM_BUCKETS - 1).to(tl.int32)
                tl.atomic_min(threshold_ptr + th_off, my_thr, sem="relaxed", mask=n_mask)
                gate = tl.minimum(gate, my_thr)

    store_mask = valid & (bucket_i <= gate[:, None])
    if DENSE:
        # 确定性 col 地址写：score 落到 dense buffer[row, m_off]，地址唯一 → 零 qcount 原子。
        # 未通过 gate 的位置不写，保留预置 sentinel(+inf)，select 端 isfinite 自动跳过。
        # col(m_off) 即 corpus id，无需 buf_idx。buffer 形状 [BN, M]，stride_bv_p=1。
        if ABLATE < 1 and SKIP_WRITE == 0:
            dv_off = n_offs[:, None] * stride_bv_n + m_offs[None, :] * stride_bv_p
            tl.store(buf_val_ptr + dv_off, score, mask=store_mask)
    else:
        # CTA/block 级 reservation：每个 CTA 对每行只做一次 atomic_add(pass_count)，
        # 再用行内 prefix sum 得到候选在本 CTA 内的 local rank。相比逐候选 atomic_add，
        # 大 k 时把 qcount[row] 的原子竞争从 O(pass items) 降到 O(CTA rows)。
        pass_i = store_mask.to(tl.int32)
        pass_count = tl.sum(pass_i, axis=1)
        local_rank = tl.cumsum(pass_i, axis=1) - 1
        if ABLATE < 1:
            q_base = tl.atomic_add(
                qcount_ptr + n_offs * stride_q_n, pass_count,
                sem="relaxed", mask=n_mask & (pass_count > 0),
            )
            qpos = q_base[:, None] + local_rank
            wmask = store_mask & (qpos < BUF)
            bv_off = n_offs[:, None] * stride_bv_n + qpos * stride_bv_p
            tl.store(buf_val_ptr + bv_off, score, mask=wmask)
            bi_off = n_offs[:, None] * stride_bi_n + qpos * stride_bi_p
            tl.store(buf_idx_ptr + bi_off, m_offs[None, :].to(tl.int32), mask=wmask)


# ─────────────────────────── 固定分段 packed 主 kernel ───────────────────────────


@triton.jit
def _seg_pack_kernel(
    x_ptr, c_ptr, origin_ptr, inv_delta_ptr,
    bcount_ptr, threshold_ptr, wcount_ptr, segcnt_ptr, buf_pack_ptr,
    stride_x_n, stride_x_d, stride_c_m, stride_c_d, stride_o_n,
    stride_bc_n, stride_bc_k, stride_t_n,
    stride_sc_n, stride_sc_s, stride_bp_n, stride_bp_p,
    N: tl.constexpr, M: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
    NSEG: tl.constexpr, CAP: tl.constexpr, SEG_PER: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr, D_INNER: tl.constexpr, NUM_BUCKETS: tl.constexpr,
    IS_L2: tl.constexpr, USE_TF32: tl.constexpr, OUT_DT: tl.constexpr,
    CNT_EVERY: tl.constexpr = 1, ACC_DT: tl.constexpr = tl.float32,
):
    """固定分段 packed 写：每个 tile 完整落在一个 65536-段内（SEG_PER=BLK/BM tiles/段）。
    候选写 packed int32 = (fp16_score_bits<<16) | off16，off16 = (pid_m%SEG_PER)*BM+col。
    段内 local rank 由 per-(row,seg) 原子计数 segcnt 给出；段基址隐式 → id=seg*65536+off16。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_offs = pid_n * BN + tl.arange(0, BN)
    n_mask = n_offs < N
    m_offs = pid_m * BM + tl.arange(0, BM)
    m_mask = m_offs < M

    if D_INNER >= D:
        d_offs = tl.arange(0, D_INNER)
        d_mask = d_offs < D
        x_tile = tl.load(x_ptr + n_offs[:, None] * stride_x_n + d_offs[None, :] * stride_x_d,
                         mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        c_tile = tl.load(c_ptr + m_offs[:, None] * stride_c_m + d_offs[None, :] * stride_c_d,
                         mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        cross = tl.dot(x_tile, tl.trans(c_tile), allow_tf32=USE_TF32, out_dtype=ACC_DT)
        if IS_L2:
            c_f = c_tile.to(ACC_DT)
            csq = tl.sum(c_f * c_f, axis=1)
    else:
        cross = tl.zeros([BN, BM], dtype=ACC_DT)
        if IS_L2:
            csq = tl.zeros([BM], dtype=ACC_DT)
        for d_start in range(0, D, D_INNER):
            d_offs = d_start + tl.arange(0, D_INNER)
            d_mask = d_offs < D
            x_sub = tl.load(x_ptr + n_offs[:, None] * stride_x_n + d_offs[None, :] * stride_x_d,
                            mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            c_sub = tl.load(c_ptr + m_offs[:, None] * stride_c_m + d_offs[None, :] * stride_c_d,
                            mask=m_mask[:, None] & d_mask[None, :], other=0.0)
            cross += tl.dot(x_sub, tl.trans(c_sub), allow_tf32=USE_TF32, out_dtype=ACC_DT)
            if IS_L2:
                c_f = c_sub.to(ACC_DT)
                csq += tl.sum(c_f * c_f, axis=1)

    score = (csq[None, :] - 2.0 * cross) if IS_L2 else (-cross)
    score = score.to(OUT_DT)
    origin = tl.load(origin_ptr + n_offs * stride_o_n, mask=n_mask, other=0.0)
    inv_delta = tl.load(inv_delta_ptr + n_offs * stride_o_n, mask=n_mask, other=0.0)
    bucket_f = (score.to(tl.float32) - origin[:, None].to(tl.float32)) * inv_delta[:, None].to(tl.float32)
    bucket_raw = bucket_f.to(tl.int32)
    in_range = bucket_raw < NUM_BUCKETS
    bucket_i = tl.minimum(tl.maximum(bucket_raw, 0), NUM_BUCKETS - 1)
    valid = n_mask[:, None] & m_mask[None, :] & in_range

    bc_off = n_offs[:, None] * stride_bc_n + bucket_i * stride_bc_k
    tl.atomic_add(bcount_ptr + bc_off, 1, sem="relaxed", mask=valid)

    th_off = n_offs * stride_t_n
    gate = tl.load(threshold_ptr + th_off, mask=n_mask, other=NUM_BUCKETS - 1).to(tl.int32)
    pass_mask = valid & (bucket_i <= gate[:, None])
    my_wr = tl.sum(pass_mask.to(tl.int32))
    prev = tl.atomic_add(wcount_ptr, my_wr)
    if (prev // CNT_EVERY) != ((prev + my_wr) // CNT_EVERY):
        bkt = tl.arange(0, NUM_BUCKETS)
        hrow = tl.load(bcount_ptr + n_offs[:, None] * stride_bc_n + bkt[None, :] * stride_bc_k,
                       mask=n_mask[:, None], other=0).to(tl.int32)
        cum = tl.cumsum(hrow, axis=1)
        cand_b = tl.where(cum >= K, bkt[None, :], NUM_BUCKETS)
        my_thr = tl.minimum(tl.min(cand_b, axis=1), NUM_BUCKETS - 1).to(tl.int32)
        tl.atomic_min(threshold_ptr + th_off, my_thr, sem="relaxed", mask=n_mask)
        gate = tl.minimum(gate, my_thr)

    store_mask = valid & (bucket_i <= gate[:, None])
    seg = pid_m // SEG_PER  # tile 完整落在一段内（BLK % BM == 0）
    pass_i = store_mask.to(tl.int32)
    pass_count = tl.sum(pass_i, axis=1)
    local_rank = tl.cumsum(pass_i, axis=1) - 1
    # per-(row,seg) 原子计数：段内 append 位置
    seg_base = tl.atomic_add(
        segcnt_ptr + n_offs * stride_sc_n + seg * stride_sc_s, pass_count,
        sem="relaxed", mask=n_mask & (pass_count > 0),
    )
    slot = seg_base[:, None] + local_rank          # 段内槽位 [0,CAP)
    qpos = seg * CAP + slot                          # buffer 内绝对槽位
    wmask = store_mask & (slot < CAP)
    off16 = ((pid_m % SEG_PER) * BM + tl.arange(0, BM))[None, :].to(tl.int32)
    key16 = score.to(tl.float16, bitcast=False).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    packed = (key16 << 16) | off16
    bp_off = n_offs[:, None] * stride_bp_n + qpos * stride_bp_p
    tl.store(buf_pack_ptr + bp_off, packed, mask=wmask)


# ─────────────────────────── 融合驱动 ───────────────────────────


@torch.no_grad()
def _fused(x, c, k, *, is_l2, use_tf32, return_distances, sorted, buf_mult,
              fp16_acc=False, _force_bn=None):
    assert x.is_cuda and c.is_cuda and x.ndim == 2 and c.ndim == 2
    assert x.dtype == c.dtype, f"query/corpus dtype mismatch: {x.dtype} vs {c.dtype}"
    assert x.dtype in _SUPPORTED_DTYPES, f"unsupported dtype {x.dtype}"
    N, D = x.shape
    M, Dc = c.shape
    assert D == Dc and 1 <= k <= M
    device = x.device
    _load_lib()

    # 扫描精度严格跟随输入 dtype：fp32 输入 → fp32 读；fp16/bf16 输入 → 16-bit 读（带宽减半）。
    # 不做任何与输入 dtype 不一致的隐式 cast——保证输入/输出精度一致、与其它方法公平对比。
    scan_dtype = x.dtype
    is_16bit = scan_dtype in (torch.float16, torch.bfloat16)
    is_fp16 = scan_dtype == torch.float16
    cfg = _heuristic_config(N=N, M=M, D=D, bytes_per=2 if is_16bit else 4)
    BN, BM, D_INNER = cfg["BN"], cfg["BM"], cfg["D_INNER"]
    if _force_bn is not None:
        BN = _force_bn
    CHUNK = BN
    # buffer 下限 12800：小 k 时 buf_mult*k（如 10*100=1000）会因候选溢出截断而丢真·近邻
    # （实测 k=100 recall 0.82→0.99 的拐点在 ~50*k）。floor 到 12800 覆盖 k<=~1280 的小 k；
    # 大 k 仍按 buf_mult*k 走。最终 clamp 到 M（buffer 不可能超过语料数）。
    BUF = min(max(buf_mult * k, 12800), M)
    assert M * D < 2**31 and BN * BUF < 2**31

    # 中间量全部跟随输入 dtype（fp16/bf16 输入 → score/buffer/select 全 16-bit，与 torch 对齐）。
    # 累加器：默认 fp32（与 cuBLAS half GEMM 一致）；fp16_acc=True 且输入为 fp16 时启用
    # tensor-core fp16 累加（更快、精度更低，仅在 fp16 下生效）。
    c = c.contiguous()
    out_dt = _tl_dtype_of(scan_dtype)
    acc_dt = tl.float16 if (fp16_acc and is_fp16) else tl.float32
    sample_size = _sample_size_for(M, k)
    cnt_every = int(os.environ.get("FLASHTOPK_CNT_EVERY", str(max(1, k))))
    # IP 路径(score=-cross)指令调度更松，更深的软件流水(num_stages)能更好重叠访存与 WGMMA，
    # 但会抬高活跃寄存器。在寄存器/SMEM 受限的老卡(如 L40S SM89)上 num_stages=3 会把寄存器顶过
    # 128 触发大量 spill(实测 spill 132、wall 8.4 vs 6.1ms)，故老卡 IP 保守取 2。Hopper(SM90+)
    # 寄存器与 SMEM 更充裕，实测 IP stages=4 全面最快且 spill=0(如 N=128/M=1M:2.19→1.60ms,1.37x)，
    # 故 SM90+ 默认 4。L2 调度紧凑、对深度不敏感，统一保持 3。可经 env 覆盖。
    _sm_major = torch.cuda.get_device_capability(device)[0]
    if is_l2:
        default_stages = 3
    else:
        default_stages = 4 if _sm_major >= 9 else 2
    num_stages = int(os.environ.get("FLASHTOPK_NUM_STAGES", str(default_stages)))

    out_idxs = torch.empty((N, k), device=device, dtype=torch.int32)
    out_vals = torch.empty((N, k), device=device, dtype=scan_dtype) if return_distances else None

    # DENSE 路径：scatter 改确定性 col 写到 [CHUNK, M] 稠密 buffer（buf[row,col]=score），
    # 彻底消除 _flat_kernel 的 qcount 原子竞争（实测 scatter 是大 batch 主瓶颈）。col 即 corpus
    # id 无需 buf_idx；未通过 gate 的位置保留 sentinel(NaN)，select 端 isfinite 自动跳过。
    # 代价：buffer [CHUNK,M] 显存大（如 256*1M*2B=512MB），且 select 要扫满 M（非 qcount）。
    _dense = int(os.environ.get("FLASHTOPK_DENSE", "0"))
    # CUDA dense threshold：DENSE 模式下用 CUDA(SMEM 直方图+cumsum) 算 th，取代 _flat 的
    # Triton global 原子直方图/阈值流。开启时 _flat 设 SKIP_HIST/SKIP_THR（写全量、零原子），
    # 阈值改由 dense_compute_threshold 重扫 buffer 一次算出。sentinel memset 仍需保留
    # （dense_compute_threshold 靠 isfinite 跳过未写位置，未写处必须是 NaN）。
    _cuda_thr = _dense and int(os.environ.get("FLASHTOPK_DENSE_CUDA_THR", "0"))
    if _dense:
        dense_buf = torch.empty((CHUNK, M), device=device, dtype=scan_dtype)

    buf_val = torch.empty((CHUNK, BUF), device=device, dtype=scan_dtype)
    buf_idx = torch.empty((CHUNK, BUF), device=device, dtype=torch.int32)
    qcount = torch.empty((CHUNK,), device=device, dtype=torch.int32)
    bcount = torch.empty((CHUNK, _NUM_BUCKETS), device=device, dtype=torch.int32)
    threshold = torch.empty((CHUNK,), device=device, dtype=torch.int32)
    wcount = torch.zeros((1,), device=device, dtype=torch.int32)

    for q0 in range(0, N, CHUNK):
        q1 = min(q0 + CHUNK, N)
        R = q1 - q0
        xb = x[q0:q1].contiguous()
        bv, bi, qc = buf_val[:R], buf_idx[:R], qcount[:R]
        qc.zero_()

        if _dense:
            # 预置 sentinel：0xFF 字节填充 → fp16/bf16/fp32 均为 NaN（指数全1+尾数非0），
            # select 端 isfinite 自动跳过未写位置。view(uint8).fill_(0xFF) 退化为单次 memset
            # （远快于 float fill_(nan) 的逐元素写）。仅 [R, M] 子视图，避免触碰其它行。
            db = dense_buf[:R]
            db.view(torch.uint8).fill_(0xFF)

        ob, ib, sample_idx = _sample_range_topk(
            xb, c, k, is_l2=is_l2, use_tf32=use_tf32, sample_size=sample_size,
            sample_val_out=bv, sample_count_out=qc, bufidx_out=bi, acc_dt=acc_dt,
        )

        if _dense:
            # sample top-k 的局部位置 sample_idx∈[0,S) 即全局 corpus id；按 col 确定性 scatter
            # 入 dense buffer[row, id]=score。col 唯一 → 无原子竞争。
            db.scatter_(1, sample_idx.long(), bv[:, :k])

        scan_start = sample_size
        grid_m = math.ceil((M - scan_start) / BM)
        th = None
        if grid_m > 0:
            bc, th = bcount[:R], threshold[:R]
            bc.zero_(); th.fill_(_NUM_BUCKETS - 1); wcount.zero_()
            # dense 模式：buffer 改 [R, M] 稠密视图，score 按 col(=corpus id) 确定性写，
            # BUF=M、stride_bv_p=1，无 qcount 原子。非 dense 走原 flat buffer + qcount。
            kbv = db if _dense else bv
            k_buf = M if _dense else BUF
            try:
                _flat_kernel[(grid_m, math.ceil(R / BN))](
                    xb, c, ob, ib, bc, th, wcount, qc, kbv, bi,
                    xb.stride(0), xb.stride(1), c.stride(0), c.stride(1), ob.stride(0),
                    bc.stride(0), bc.stride(1), th.stride(0), qc.stride(0),
                    kbv.stride(0), kbv.stride(1), bi.stride(0), bi.stride(1),
                    N=R, M=M, D=D, K=k, BUF=k_buf, BN=BN, BM=BM, D_INNER=D_INNER,
                    NUM_BUCKETS=_NUM_BUCKETS, IS_L2=is_l2, USE_TF32=use_tf32, OUT_DT=out_dt,
                    CNT_EVERY=cnt_every, M_START=scan_start, ACC_DT=acc_dt,
                    ABLATE=int(os.environ.get("FLASHTOPK_ABLATE", "0")),
                    DENSE=int(_dense),
                    SKIP_HIST=int(os.environ.get("FLASHTOPK_SKIP_HIST", "0")) or int(_cuda_thr),
                    SKIP_THR=int(os.environ.get("FLASHTOPK_SKIP_THR", "0")) or int(_cuda_thr),
                    SKIP_WRITE=int(os.environ.get("FLASHTOPK_SKIP_WRITE", "0")),
                    GEMM_ONLY=int(os.environ.get("FLASHTOPK_GEMM_ONLY", "0")),
                    THR_PER_CTA=int(os.environ.get("FLASHTOPK_THR_PER_CTA", "0")),
                    num_warps=cfg["num_warps"], num_stages=num_stages,
                )
            except OutOfResources:
                # 估算偏乐观导致实际 SMEM 超限：缩小 BN 重跑整次调用（buffer 按 BN 重分配）。
                if BN <= 16:
                    raise
                return _fused(x, c, k, is_l2=is_l2, use_tf32=use_tf32,
                                 return_distances=return_distances, sorted=sorted,
                                 buf_mult=buf_mult, fp16_acc=fp16_acc, _force_bn=BN // 2)

            if _cuda_thr:
                # _flat 已写全量 score 到 dense buffer（无 hist/thr）。重扫 buffer 一遍，
                # 用 CUDA SMEM 直方图 + per-row cumsum 算阈值桶 th，替代 Triton 端的
                # global 原子直方图/流式阈值。th 形状/语义与原 threshold[:R] 一致。
                th = litetopk_ops.dense_compute_threshold(db, ob, ib, k, _NUM_BUCKETS)

        # 选择算法：复用 _flat_kernel 留下的收敛阈值桶 th 与分桶坐标 (ob, ib)，
        # 直接在有界 buffer [R, BUF] 上只跑 gather+boundary，跳过 sample/hist/threshold
        # 三个算子（这些信息融合阶段已算出）。buffer 尾部预填 +inf 经分桶自然剔除。
        # ob/ib 各自维护精度（算子内部算桶号时统一升 fp32，与 _flat_kernel 逐位一致）。
        # 退化分支（grid_m==0，整批落在 sample 段内）无 th，退回通用 topk_min。
        if th is not None:
            if _dense:
                # dense：buffer [R,M] sentinel NaN，col=corpus id（无 buf_idx），扫满 M（无 qcount）。
                # 复用 mb 的边界桶拆分 finalize，仅去掉 buf_idx/qcount 旁路。
                _cap_mult = int(os.environ.get("FLASHTOPK_SELECT_CAP_MULT", "4"))
                _cap = min(max(_cap_mult * k, 8192), M)
                topv, topi = litetopk_ops.topk_select_thr_mb_dense(
                    db, ob, ib, th, k, _NUM_BUCKETS, _cap)
            else:
                # 大 BUF 下单 block 串行扫满 BUF 成为 select 瓶颈；改用 multi-block 版（边界桶拆分）。
                _mb_default = 1 if BUF >= 65536 else 0
                _use_mb = int(os.environ.get("FLASHTOPK_SELECT_MB", str(_mb_default)))
                if _use_mb:
                    # CAP 给足边界桶候选余量；默认 4*k（过小会截断边界桶候选导致大 k recall 退化）。
                    _cap_mult = int(os.environ.get("FLASHTOPK_SELECT_CAP_MULT", "4"))
                    _cap = min(max(_cap_mult * k, 8192), BUF)
                    topv, topi = litetopk_ops.topk_select_thr_mb_idx(
                        bv, bi, ob, ib, th, qc, k, _NUM_BUCKETS, _cap)
                else:
                    topv, topi = litetopk_ops.topk_select_thr_idx(
                        bv, bi, ob, ib, th, qc, k, _NUM_BUCKETS)
        else:
            topv = bv[:, :k]
            topi = torch.arange(k, device=device, dtype=torch.int32)[None, :].expand(R, k)
        if sorted:
            neg = topv.neg().contiguous()
            topi = topi.contiguous()
            _cuda_sort_desc(neg, topi)
            topv = neg.neg()
        # bi[:, :k] 已是 sample 段全局行号；退化分支按 buffer 位置 gather 回全局 id。
        out_idxs[q0:q1] = topi if th is not None else torch.gather(bi, 1, topi.long())
        if return_distances:
            out_vals[q0:q1] = topv

    if not return_distances:
        return out_idxs
    if is_l2:
        # 距离重建精度跟随输入 dtype：fp16 输入 → x_sq/加法全 fp16（与 torch fp16 对齐）。
        x_sq = (x * x).sum(dim=-1)
        dist = (out_vals + x_sq.unsqueeze(-1)).clamp_min_(0.0)
        return dist, out_idxs
    return out_vals.neg(), out_idxs


# 分段 4B packed 路径在大 k 上靠 packing 省写带宽胜过 base 6B flat，小 k 上 base 因
# 跳过 sample 段而略胜。实测交叉点 ~k=6000（300k/1M 一致）。分段依赖 16-bit packed
# score，仅 fp16 输入可用。FLASHTOPK_SEG 可强制覆盖：'1' 恒走分段、'0' 恒走 base、
# 未设则按 (dtype==fp16 且 k>=阈值) 自动选路。
_SEG_K_THRESH = int(os.environ.get("FLASHTOPK_SEG_K_THRESH", "6000"))


def _use_seg(k, dtype):
    forced = os.environ.get("FLASHTOPK_SEG")
    if forced is not None:
        return forced == "1"
    return dtype == torch.float16 and k >= _SEG_K_THRESH


@torch.no_grad()
def _fused_seg(x, c, k, *, is_l2, use_tf32, return_distances, sorted, buf_mult):
    """固定分段 packed 路径（方案A）：corpus 每 65536 向量为一段，候选打包成 int32
    = (fp16_score_bits<<16) | off16，按段原子 append（段内槽位纯算术 seg*CAP+slot）。
    段基址隐式 → id = seg*65536 + off16，select 端只需 topk_min(score) + 解码，无需写真实 idx。
    每段容量 CAP=2048（用户指定 2k），固定分配不抢页。仅支持 fp16（packed 依赖 16-bit score）。"""
    assert x.is_cuda and c.is_cuda and x.ndim == 2 and c.ndim == 2
    assert x.dtype == c.dtype, f"query/corpus dtype mismatch: {x.dtype} vs {c.dtype}"
    assert x.dtype == torch.float16, "seg path 仅支持 fp16（packed 依赖 16-bit score）"
    N, D = x.shape
    M, Dc = c.shape
    assert D == Dc and 1 <= k <= M
    device = x.device
    _load_lib()

    scan_dtype = x.dtype
    cfg = _heuristic_config(N=N, M=M, D=D, bytes_per=2)
    BN, BM, D_INNER = cfg["BN"], cfg["BM"], cfg["D_INNER"]
    CHUNK = BN
    BLK = 65536
    assert BLK % BM == 0, f"BLK={BLK} 必须整除 BM={BM}"
    SEG_PER = BLK // BM
    NSEG = math.ceil(M / BLK)
    # 每段容量随 k 自适应。CUDA packed select 用段内 prefix-sum 紧凑遍历，迭代量=实际
    # 候选数，**与 CAP 解耦**——CAP 只需覆盖 maxseg 防溢出（溢出即丢候选、掉 recall），
    # 超配仅多占内存、不拖慢 select。真实语料 top 候选聚集，实测 maxseg≈3.2x 均值
    # （BUF/NSEG），取 4x 留余量；clamp 到 BLK（单段物理上限）。floor 2048 保小 k。
    BUF = min(buf_mult * k, M)
    CAP = int(os.environ.get("FLASHTOPK_SEG_CAP",
                             str(min(BLK, max(2048, math.ceil(BUF / NSEG) * 4)))))
    assert CHUNK * NSEG * CAP < 2**31

    c = c.contiguous()
    out_dt = _tl_dtype_of(scan_dtype)
    acc_dt = tl.float32
    sample_size = _sample_size_for(M, k)
    cnt_every = max(1, k)
    num_stages = 3 if is_l2 else 2

    out_idxs = torch.empty((N, k), device=device, dtype=torch.int32)
    out_vals = torch.empty((N, k), device=device, dtype=scan_dtype) if return_distances else None

    buf_pack = torch.empty((CHUNK, NSEG * CAP), device=device, dtype=torch.int32)
    segcnt = torch.empty((CHUNK, NSEG), device=device, dtype=torch.int32)
    bcount = torch.empty((CHUNK, _NUM_BUCKETS), device=device, dtype=torch.int32)
    threshold = torch.empty((CHUNK,), device=device, dtype=torch.int32)
    wcount = torch.zeros((1,), device=device, dtype=torch.int32)
    dummy_val = torch.empty((CHUNK, k), device=device, dtype=scan_dtype)
    dummy_cnt = torch.empty((CHUNK,), device=device, dtype=torch.int32)
    dummy_idx = torch.empty((CHUNK, k), device=device, dtype=torch.int32)

    for q0 in range(0, N, CHUNK):
        q1 = min(q0 + CHUNK, N)
        R = q1 - q0
        xb = x[q0:q1].contiguous()
        bp = buf_pack[:R]; sc = segcnt[:R]; bc = bcount[:R]; th = threshold[:R]
        # 不 fill buf_pack：segcnt 界定每段有效槽，select 只扫 [seg*CAP, seg*CAP+segcnt)。
        sc.zero_(); bc.zero_(); th.fill_(_NUM_BUCKETS - 1); wcount.zero_()

        # sample 只为算桶坐标（origin/inv_delta）；seg kernel 全量扫 [0,M)，不依赖预填段。
        ob, ib = _sample_range_topk(
            xb, c, k, is_l2=is_l2, use_tf32=use_tf32, sample_size=sample_size,
            sample_val_out=dummy_val[:R], sample_count_out=dummy_cnt[:R],
            sample_idx_out=dummy_idx[:R], acc_dt=acc_dt,
        )

        grid = (math.ceil(M / BM), math.ceil(R / BN))
        _seg_pack_kernel[grid](
            xb, c, ob, ib, bc, th, wcount, sc, bp,
            xb.stride(0), xb.stride(1), c.stride(0), c.stride(1), ob.stride(0),
            bc.stride(0), bc.stride(1), th.stride(0),
            sc.stride(0), sc.stride(1), bp.stride(0), bp.stride(1),
            N=R, M=M, D=D, K=k, NSEG=NSEG, CAP=CAP, SEG_PER=SEG_PER,
            BN=BN, BM=BM, D_INNER=D_INNER, NUM_BUCKETS=_NUM_BUCKETS,
            IS_L2=is_l2, USE_TF32=use_tf32, OUT_DT=out_dt,
            CNT_EVERY=cnt_every, ACC_DT=acc_dt,
            num_warps=cfg["num_warps"], num_stages=num_stages,
        )

        # CUDA packed select：kernel 内对有效槽 radix K-min + 寄存器解码 score/id，
        # 取代 torch 侧 (>>/&/view/gather/topk_min) 那串 elementwise。
        topv, pred_id = litetopk_ops.topk_select_packed(bp, sc, CAP, BLK, M, k)
        if sorted:
            neg = topv.neg().contiguous()
            pred_id = pred_id.contiguous()
            _cuda_sort_desc(neg, pred_id)
            topv = neg.neg()
        out_idxs[q0:q1] = pred_id
        if return_distances:
            out_vals[q0:q1] = topv

    if not return_distances:
        return out_idxs
    if is_l2:
        x_sq = (x * x).sum(dim=-1)
        dist = (out_vals + x_sq.unsqueeze(-1)).clamp_min_(0.0)
        return dist, out_idxs
    return out_vals.neg(), out_idxs


@torch.no_grad()
def fused_knn_topk_l2(x, c, k, *, use_tf32=True, return_distances=True,
                         sorted=False, buf_mult=10, fp16_acc=False):
    """全融合 squared-L2 KNN top-k（不写回 score 矩阵）。返回 (squared_l2, idx) 或 idx。

    扫描精度严格跟随输入 dtype：fp16 输入则语料/查询以 fp16 读（corpus 读带宽减半），
    fp32 输入则全程 fp32。matmul 累加器默认 fp32（与 cuBLAS half GEMM / torch 对齐）；
    fp16_acc=True 且输入为 fp16 时启用 tensor-core fp16 累加（更快、精度更低）。
    """
    if _use_seg(k, x.dtype):
        return _fused_seg(x, c, k, is_l2=True, use_tf32=use_tf32,
                          return_distances=return_distances, sorted=sorted, buf_mult=buf_mult)
    return _fused(x, c, k, is_l2=True, use_tf32=use_tf32,
                     return_distances=return_distances, sorted=sorted,
                     buf_mult=buf_mult, fp16_acc=fp16_acc)


@torch.no_grad()
def fused_knn_topk_ip(x, c, k, *, use_tf32=False, return_distances=True,
                         sorted=False, buf_mult=10, fp16_acc=False):
    """全融合 inner-product KNN top-k（不写回 score 矩阵）。返回 (ip, idx) 或 idx。"""
    if _use_seg(k, x.dtype):
        return _fused_seg(x, c, k, is_l2=False, use_tf32=use_tf32,
                          return_distances=return_distances, sorted=sorted, buf_mult=buf_mult)
    return _fused(x, c, k, is_l2=False, use_tf32=use_tf32,
                     return_distances=return_distances, sorted=sorted,
                     buf_mult=buf_mult, fp16_acc=fp16_acc)
