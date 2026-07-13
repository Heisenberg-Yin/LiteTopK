"""Unified KNN top-k benchmark — 一个脚本对比所有方法。

把原先分散在 bench_torch_topk / bench_raft_topk / bench_faiss / bench_flashlib /
bench_onekernel / bench_vs_flashlib 里的 baseline 合并到这里，统一数据、统一计时、
统一 recall（torch 暴力 top-k 为 ground truth）。

被测方法（--methods 逗号分隔，缺省 ours,torch）：
  * torch           —— torch.matmul + torch.topk（精确，recall ground truth）
  * raft            —— torch.matmul + pylibraft.matrix.select_k（radix select）
  * faiss           —— faiss.knn_gpu（自带 matmul + 选择，k<=2048）
  * flashlib        —— pristine flashlib.primitives.knn.flash_knn（融合精确）
  * flashlib_triton —— flashlib...triton.dispatch.flash_knn_triton（融合精确）
  * ours            —— src/litetopk_fused.fused_knn_topk_l2（融合近似，CUDA litetopk_select 收尾）

依赖互斥说明（重要）：
  * flashlib/flashlib_triton 需 Triton 3.3（旧容器 ziqi_qrita_triton），在 Triton 3.1 下
    import 即 native segfault（无法被 try/except 捕获）——故用 --methods / BENCH_METHODS
    从源头控制 import 哪些方法。
  * ours 的 litetopk_select op 需 gcc>=9（新容器 torch_efficient_topk）。
  * faiss / pylibraft 仅在装了对应包的 conda env 可用，未装则该方法自动报错跳过。
  * torch 暴力 top-k 始终作为跨容器/跨环境对齐的公共锚点（同卡、同数据、同实现）。

容器内（新容器，ours + torch；扫 k 与 dtype）：
    docker exec -e CUDA_VISIBLE_DEVICES=7 -e CC=/usr/bin/gcc-12 -e CXX=/usr/bin/g++-12 \
      torch_efficient_topk bash -c 'cd /home/user/test/BBC-GPU/EfficientTopK && \
      python bench_all.py --base .../base.fvecs --query .../query.fvecs \
      --num-base 1000000 --num-queries 1024 --k 100,500,1000 --dtype fp32,fp16 --methods ours,torch'

容器内（旧容器，flashlib + torch；不要设 CUDA_VISIBLE_DEVICES，容器已锁单卡）：
    docker exec ziqi_qrita_triton bash -c 'cd .../EfficientTopK && \
      python bench_all.py ... --k 100,1000 --dtype fp32,fp16 --methods flashlib_triton,torch'

  --k     逗号分隔的 k 列表（如 100,500,1000）
  --dtype 逗号分隔的精度（fp32,fp16）：所有方法在 fp16 下都把 Q/X cast 成 half，
          输入/输出 dtype 一致（matmul 走 half tensor core，按 fp32 累加）；faiss 无 half 路径仍 fp32。
  --autotune  对 ours/ours_acc16，每个 (k,dtype) 先扫 num_warps×num_stages×BN×BM 这组
          _flat_kernel 超参（经 FLASHTOPK_* env 覆盖），按 recall>=--autotune-rec-min 中
          median wall 最小选最优 config，再用该 config 计时；summary 末列打印选中的超参。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", "marsco")
_DEFAULT_BASE = os.path.join(_DATA_DIR, "base.fvecs")
_DEFAULT_QUERY = os.path.join(_DATA_DIR, "query.fvecs")
sys.path.insert(0, os.path.join(_HERE, "src"))          # litetopk_fused / litetopk_ops
sys.path.insert(0, _HERE)                                # _common
sys.path.insert(0, os.path.dirname(_HERE))               # BBC-GPU (flashlib)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "flashlib"))  # pristine flashlib

import numpy as np
import torch

# CUDA context before any Triton-backed import (driver probe).
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()


_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_DTYPE_NAME = {v: k for k, v in _DTYPES.items()}

# 距离度量：l2=squared-L2（取最小），ip=inner product（取最大）。
# 经 --metric 设置，影响 ours 走 l2/ip 路径、torch 参考算法、ground-truth 排序方向。
_METRIC = "l2"


def _cast(t, dt):
    return t if dt == torch.float32 else t.to(dt)


def read_fvecs(filename: str, limit=None, dim: int | None = None) -> np.ndarray:
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    src_dim = int(data[0])
    if dim is None:
        dim = src_dim
    if dim < 1 or dim > src_dim:
        raise ValueError(f"dim={dim} must be in [1, {src_dim}] for {filename}")
    rec = src_dim + 1
    n = data.size // rec
    arr = data.reshape(n, rec)[:, 1:1 + dim].view(np.float32)
    if limit is not None:
        arr = arr[:limit]
    return np.ascontiguousarray(arr)


def _time_cuda(fn, warmup: int, iters: int, label: str):
    """CUDA-event 计时，返回 (最后一次输出, 每次耗时数组 ms)。供 test_fused 复用。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    last = None
    for i in range(iters):
        starts[i].record()
        last = fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    print(
        f"[{label}] iters={iters}  mean={times.mean():.3f} ms  "
        f"median={np.median(times):.3f} ms  min={times.min():.3f} ms  max={times.max():.3f} ms"
    )
    return last, times


# ─────────────────────────── 方法实现（每个返回 idx[B,k] int64）───────────────────────────
# 闭包工厂 make_*(Q32, X32, k, fp16) -> fn()：Q32/X32 是 fp32 主拷贝，工厂内部按 fp16 标志
# 自行 cast。所有方法（含 ours）在 fp16 模式都真正把 Q/X cast 成 half，输入/输出 dtype 一致，
# matmul 走 half tensor core（按 fp32 累加）。faiss 无 half 路径，fp16 模式仍喂 fp32。
# 持久句柄（raft handle / faiss res）在工厂里建一次，避免计入逐次延迟。


def make_torch(Q32, X32, k, dt):
    Q = _cast(Q32, dt)
    X = _cast(X32, dt)

    @torch.no_grad()
    def fn():
        if _METRIC == "ip":
            score = Q @ X.t()
            return torch.topk(score, k, dim=-1, largest=True, sorted=False).indices
        csq = (X * X).sum(-1)
        score = csq[None, :] - 2.0 * (Q @ X.t())
        return torch.topk(score, k, dim=-1, largest=False, sorted=False).indices
    return fn


def make_raft(Q32, X32, k, dt):
    from pylibraft.common import DeviceResources
    from pylibraft.matrix import select_k
    Q = _cast(Q32, dt)
    X = _cast(X32, dt)
    handle = DeviceResources()
    csq = (X * X).sum(-1)

    @torch.no_grad()
    def fn():
        score = (csq[None, :] - 2.0 * (Q @ X.t())).contiguous()
        out_d = torch.empty(Q.shape[0], k, device=Q.device, dtype=score.dtype)
        out_i = torch.empty(Q.shape[0], k, device=Q.device, dtype=torch.int64)
        select_k(score, k=k, distances=out_d, indices=out_i, select_min=True, handle=handle)
        handle.sync()
        return out_i
    return fn


def make_faiss(Q32, X32, k, dt):
    import faiss
    import faiss.contrib.torch_utils  # noqa: F401
    res = faiss.StandardGpuResources()
    # faiss.knn_gpu 只接受 fp32；fp16/bf16 模式下仍用 fp32（faiss 无 half 路径），标注于汇总。
    x = Q32.contiguous()
    c = X32.contiguous()

    @torch.no_grad()
    def fn():
        _, Iout = faiss.knn_gpu(res, x, c, k, metric=faiss.METRIC_L2)
        return Iout.long()
    return fn


def make_flashlib(Q32, X32, k, dt):
    from flashlib.primitives.knn import flash_knn
    Q = _cast(Q32, dt)
    X = _cast(X32, dt)

    @torch.no_grad()
    def fn():
        _, idxs = flash_knn(Q, X, k, return_distances=True)
        return idxs.long()
    return fn


def make_flashlib_triton(Q32, X32, k, dt):
    from flashlib.primitives.knn.triton.dispatch import flash_knn_triton
    Q = _cast(Q32, dt)
    X = _cast(X32, dt)

    @torch.no_grad()
    def fn():
        return flash_knn_triton(Q.unsqueeze(0), X.unsqueeze(0), k)[0].long()
    return fn


def make_ours(Q32, X32, k, dt):
    from litetopk_fused import fused_knn_topk_l2, fused_knn_topk_ip
    # 与其它方法一致：fp16/bf16 模式真正传 16-bit 张量（读 16-bit、tl.dot fp32 累加），
    # 输入/输出 dtype 一致，不再用解耦旋钮偷偷只降语料读精度。
    Q = _cast(Q32, dt)
    X = _cast(X32, dt)

    @torch.no_grad()
    def fn():
        if _METRIC == "ip":
            return fused_knn_topk_ip(Q, X, k, return_distances=False).long()
        return fused_knn_topk_l2(Q, X, k, return_distances=False).long()
    return fn


def make_ours_acc16(Q32, X32, k, dt):
    # 可选 tensor-core fp16 累加路径：仅 fp16 输入下生效（fp32/bf16 输入回退到 fp32 累加，
    # 与 ours 等价）。更快但精度更低，用于对照 torch 的 cuBLAS half GEMM 加速。
    from litetopk_fused import fused_knn_topk_l2
    Q = _cast(Q32, dt)
    X = _cast(X32, dt)

    @torch.no_grad()
    def fn():
        return fused_knn_topk_l2(Q, X, k, return_distances=False, fp16_acc=True).long()
    return fn


_FACTORIES = {
    "torch": make_torch,
    "raft": make_raft,
    "faiss": make_faiss,
    "flashlib": make_flashlib,
    "flashlib_triton": make_flashlib_triton,
    "ours": make_ours,
    "ours_acc16": make_ours_acc16,
}

# ─────────────────────────── autotune（ours 的 _flat_kernel 超参）───────────────────────────
# 搜索空间：num_warps × num_stages × BN × BM，全部通过 FLASHTOPK_* env 覆盖（litetopk_fused
# 在每次调用时读取）。实测发现最优点随 metric/形状漂移（如 BN=256+num_warps=8 常胜过启发式
# 默认的 num_warps=16），且最低 spill 未必最快（小 tile 牺牲 tensor-core 效率），故用真实计时
# 选最优、recall 兜底过滤近似解。
_AUTOTUNE_ENV = ("FLASHTOPK_NUM_WARPS", "FLASHTOPK_NUM_STAGES", "FLASHTOPK_BN", "FLASHTOPK_BM")
_AUTOTUNE_SPACE = [
    {"FLASHTOPK_NUM_WARPS": str(nw), "FLASHTOPK_NUM_STAGES": str(ns),
     "FLASHTOPK_BN": str(bn), "FLASHTOPK_BM": str(bm)}
    for nw in (8, 16) for ns in (2, 3, 4) for bn in (16, 32, 64, 128, 256) for bm in (64, 128)
]


def _autotune_space_for(N):
    """按 query batch N 裁剪搜索空间：BN 一次铺 BN 行做 tensor-core MMA，BN>N 的部分是
    padding 空算且仍要扫完整个语料 → 对小 N 是纯浪费。保留最多一个 >=N 的 BN 档（向上取整到
    覆盖 N 的最小候选），更大的 BN 直接剔除，避免 autotune 把小 N 选到 3ms padding 地板。"""
    bns = sorted({16, 32, 64, 128, 256})
    keep = [b for b in bns if b < N]
    cover = [b for b in bns if b >= N]
    if cover:
        keep.append(cover[0])  # 覆盖 N 的最小 BN（如 N=16→16, N=48→64）
    keep = set(keep) or {min(bns)}
    return [c for c in _AUTOTUNE_SPACE if int(c["FLASHTOPK_BN"]) in keep]


def _clear_autotune_env():
    for kk in _AUTOTUNE_ENV:
        os.environ.pop(kk, None)


def _max_spills():
    try:
        from litetopk_fused import _flat_kernel
    except Exception:
        return None
    best = 0
    # triton<3.5: JITFunction.cache = {device: {key: compiled}};
    # triton>=3.5: JITFunction.device_caches = {device: (cache_dict, binder)}.
    caches = getattr(_flat_kernel, "cache", None)
    if caches is not None:
        per_device = list(caches.values())
    else:
        dc = getattr(_flat_kernel, "device_caches", {})
        per_device = [v[0] if isinstance(v, tuple) else v for v in dc.values()]
    for kerns in per_device:
        for ck in kerns.values():
            best = max(best, getattr(ck, "n_spills", 0) or 0)
    return best


def _autotune_ours(factory, Q, X, k, dt, ref, S, *, warmup, iters, rec_min):
    """扫 _AUTOTUNE_SPACE，按 recall>=rec_min 中 median wall 最小选最优 env 配置。
    返回 (best_env, best_ms, best_recall, best_spills)。副作用：把 best_env 写入 os.environ。"""
    try:
        from litetopk_fused import _flat_kernel
    except Exception:
        _flat_kernel = None
    dtn = _DTYPE_NAME[dt]
    space = _autotune_space_for(Q.shape[0])
    print(f"[autotune] ours k={k} {dtn} N={Q.shape[0]}: "
          f"sweeping {len(space)} configs ...")
    results = []
    for cfg in space:
        os.environ.update(cfg)
        if _flat_kernel is not None:
            # triton<3.5: JITFunction.cache (dict per device);
            # triton>=3.5: JITFunction.device_caches (dict per device).
            _cache = getattr(_flat_kernel, "cache", None)
            if _cache is not None:
                _cache.clear()
            _dcache = getattr(_flat_kernel, "device_caches", None)
            if _dcache is not None:
                _dcache.clear()
        try:
            fn = factory(Q, X, k, dt)
            idx, ms = _time(fn, warmup=warmup, iters=iters,
                            label=f"autotune nw={cfg['FLASHTOPK_NUM_WARPS']} "
                                  f"ns={cfg['FLASHTOPK_NUM_STAGES']} "
                                  f"BN={cfg['FLASHTOPK_BN']} BM={cfg['FLASHTOPK_BM']}")
        except Exception as e:  # noqa: BLE001  (OutOfResources 等 → 跳过该 config)
            print(f"  [autotune] cfg {cfg} FAIL {type(e).__name__}")
            continue
        rec = _recall(idx[:S], ref, k, S)
        sp = _max_spills()
        results.append((ms, rec, sp, cfg))
    _clear_autotune_env()
    if not results:
        return None, None, None, None
    valid = [r for r in results if r[1] >= rec_min] or results
    best_ms, best_rec, best_sp, best_cfg = min(valid, key=lambda r: r[0])
    os.environ.update(best_cfg)
    print(f"[autotune] best ours k={k} {dtn}: "
          f"{best_ms:.3f}ms recall={best_rec:.5f} spills={best_sp} cfg={best_cfg}")
    return best_cfg, best_ms, best_rec, best_sp


def _time(fn, warmup, iters, label):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    out = None
    for i in range(iters):
        ev0[i].record()
        out = fn()
        ev1[i].record()
    torch.cuda.synchronize()
    t = np.array([s.elapsed_time(e) for s, e in zip(ev0, ev1)])
    print(f"[{label}] iters={iters}  mean={t.mean():.3f} ms  median={np.median(t):.3f} ms  "
          f"min={t.min():.3f} ms  max={t.max():.3f} ms")
    return out, float(np.median(t))


@torch.no_grad()
def _torch_ref(Q, X, k, dt=torch.float32, q_chunk=512):
    """ground-truth 与被测算子走同一精度链路：输入 cast 到 dt，score 用 fp32 累加，
    再 cast 回 dt 后 topk。这样 recall 分母衡量的是'算法相对同精度暴力解'的损失，
    不混入 dt 输入相对 fp32 的量化误差。"""
    Xd = X.to(dt)
    idxs = []
    if _METRIC == "ip":
        for s in range(0, Q.shape[0], q_chunk):
            q = Q[s:s + q_chunk].to(dt)
            score = (q.float() @ Xd.float().T).to(dt)
            idxs.append(torch.topk(score, k, dim=-1, largest=True).indices)
        return torch.cat(idxs, 0)
    x_sq = (Xd.float() * Xd.float()).sum(-1)
    for s in range(0, Q.shape[0], q_chunk):
        q = Q[s:s + q_chunk].to(dt).float()
        d = (q * q).sum(-1, keepdim=True) + x_sq[None, :] - 2.0 * (q @ Xd.float().T)
        d = d.to(dt)
        idxs.append(torch.topk(d, k, dim=-1, largest=False).indices)
    return torch.cat(idxs, 0)


def _recall(a, b, k, n):
    a = a.to(torch.int64); b = b.to(torch.int64)
    r = 0.0
    for i in range(n):
        r += len(set(a[i].tolist()) & set(b[i].tolist())) / k
    return r / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--k", default="100", help="逗号分隔的 k 列表，如 100,500,1000")
    p.add_argument("--dtype", default="fp32", help="逗号分隔：fp32,fp16")
    p.add_argument("--metric", default="l2", choices=["l2", "ip"],
                   help="距离度量：l2=squared-L2(取最小)；ip=inner product(取最大)")
    p.add_argument("--num-queries", default="1024",
                   help="comma-separated query batch (N), e.g. 256,1024,4096")
    p.add_argument("--num-base", default="",
                   help="comma-separated corpus (M), e.g. 200000,1000000; empty=full")
    p.add_argument("--dim", type=int, default=0,
                   help="optional feature dimension prefix to use; 0 keeps full fvecs dim")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--methods", default=os.environ.get("BENCH_METHODS", "ours,torch"),
                   help="逗号分隔：torch,raft,faiss,flashlib,flashlib_triton,ours")
    p.add_argument("--autotune", action="store_true",
                   help="对 ours/ours_acc16 扫 num_warps×num_stages×BN×BM 选最优超参后再计时")
    p.add_argument("--autotune-rec-min", type=float, default=0.99,
                   help="autotune 选优时的 recall 下限（低于此的近似配置不参与选最优）")
    args = p.parse_args()

    global _METRIC
    _METRIC = args.metric

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "torch" not in methods:
        methods.append("torch")  # 始终作为 recall 锚点
    for m in methods:
        if m not in _FACTORIES:
            raise SystemExit(f"unknown method '{m}'; choose from {list(_FACTORIES)}")
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    dtypes = [d.strip() for d in args.dtype.split(",") if d.strip()]
    for d in dtypes:
        if d not in _DTYPES:
            raise SystemExit(f"unknown dtype '{d}'; choose from {list(_DTYPES)}")
    ns = [int(x) for x in args.num_queries.split(",") if x.strip()]
    ms_list = [int(x) for x in args.num_base.split(",") if x.strip()] or [None]

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  methods={methods}  metric={_METRIC}  "
          f"ks={ks}  dtypes={dtypes}  N={ns}  M={ms_list}")
    max_n = max(ns)
    max_m = max([m for m in ms_list if m is not None], default=None)
    t0 = time.time()
    dim = args.dim or None
    X_full = torch.from_numpy(read_fvecs(args.base, limit=max_m, dim=dim)).to(device).float()
    Q_full = torch.from_numpy(read_fvecs(args.query, limit=max_n, dim=dim)).to(device).float()
    if Q_full.shape[0] < max_n:
        reps = (max_n + Q_full.shape[0] - 1) // Q_full.shape[0]
        Q_full = Q_full.repeat(reps, 1)[:max_n].contiguous()
        print(f"[tile] query file too small; tiled queries up to N={max_n}")
    print(f"loaded X={tuple(X_full.shape)} Q={tuple(Q_full.shape)} in {time.time()-t0:.1f}s")

    # summary 行：(N, M, k, dtype_name, method, ms, recall, cfg_note)
    summary = []

    for M in ms_list:
        X = X_full if M is None else X_full[:M].contiguous()
        Meff = X.shape[0]
        for N in ns:
            Q = Q_full[:N].contiguous()
            Neff = Q.shape[0]
            S = min(128, Neff)
            ref_cache = {}  # (k, dn) -> same-precision ground-truth idx[S,k]
            for k in ks:
                if k > Meff:
                    print(f"[skip] k={k} > M={Meff}")
                    continue
                for dn in dtypes:
                    dt = _DTYPES[dn]
                    if (k, dn) not in ref_cache:
                        ref_cache[(k, dn)] = _torch_ref(Q[:S], X, k, dt)
                    ref = ref_cache[(k, dn)]
                    print(f"\n########## N={Neff} M={Meff} k={k} dtype={dn} ##########")
                    for m in methods:
                        cfg_note = ""
                        if args.autotune and m in ("ours", "ours_acc16"):
                            best_cfg, *_ = _autotune_ours(
                                _FACTORIES[m], Q, X, k, dt, ref, S,
                                warmup=args.warmup, iters=args.iters,
                                rec_min=args.autotune_rec_min)
                            if best_cfg:
                                cfg_note = (f"nw={best_cfg['FLASHTOPK_NUM_WARPS']},"
                                            f"ns={best_cfg['FLASHTOPK_NUM_STAGES']},"
                                            f"BN={best_cfg['FLASHTOPK_BN']},"
                                            f"BM={best_cfg['FLASHTOPK_BM']}")
                        try:
                            fn = _FACTORIES[m](Q, X, k, dt)
                        except Exception as e:  # noqa: BLE001 (faiss/raft 未装 → 跳过)
                            print(f"[skip] {m}: {e}")
                            _clear_autotune_env()
                            continue
                        try:
                            idx, ms = _time(fn, warmup=args.warmup, iters=args.iters,
                                            label=f"{m}|N={Neff}|M={Meff}|k={k}|{dn}")
                        except Exception as e:  # noqa: BLE001 (运行期 OOM/不支持 → 跳过)
                            print(f"[fail] {m}: {type(e).__name__}: {str(e)[:200]}")
                            _clear_autotune_env()
                            continue
                        rec = _recall(idx[:S], ref, k, S)
                        summary.append((Neff, Meff, k, dn, m, ms, rec, cfg_note))
                        if args.autotune and m in ("ours", "ours_acc16"):
                            _clear_autotune_env()  # 不让最优 env 泄漏到后续方法/组合

    print("\n=== summary ===")
    print(f"{'N':>7} {'M':>9} {'k':>6} {'dtype':>6} {'method':<18} "
          f"{'median_ms':>10} {'recall':>9}  {'autotune_cfg'}")
    for N, M, k, dn, m, ms, rec, cfg_note in summary:
        print(f"{N:>7} {M:>9} {k:>6} {dn:>6} {m:<18} {ms:>10.3f} {rec:>9.5f}  {cfg_note}")
    # 每个 (N,M,k,dtype) 组合内 ours vs torch 加速比
    print("\n--- ours vs torch speedup ---")
    for row in summary:
        N, M, k, dn, m = row[0], row[1], row[2], row[3], row[4]
        if m != "ours":
            continue
        t = next((s[5] for s in summary
                  if (s[0], s[1], s[2], s[3], s[4]) == (N, M, k, dn, "torch")), None)
        if t:
            o = row[5]
            print(f"  N={N:<6} M={M:<9} k={k:<6} {dn}: {t / o:.2f}x  "
                  f"(ours {o:.2f} / torch {t:.2f} ms)")


if __name__ == "__main__":
    main()
