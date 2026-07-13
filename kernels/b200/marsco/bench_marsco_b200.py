"""MS MARCO IP top-k benchmark on B200 (SM100 LiteTopK vs torch dense).

Defaults: batch = 128 concurrent queries, fp32 output values (fp16 inputs).

Task (paper spec): batch = 64 concurrent queries over a shared corpus of
768-d embeddings, corpus size M in {1M, 2M, 4M, 5M}, k in {128, 1024, 8192,
32768}. LiteTopK runs the flat-batch mode of tidal_sm100::fused_ip_gqa_sparse_b200
(64 rows = 8 groups x QN=8, all mapped to the single corpus via q_group_size);
the D=768 scan uses the kernel's 256-wide D-chunk pipeline.

Data: ../../data/marsco/base_5m.fvecs (5M x 768 fp32) + query.fvecs (1000 x 768).
First run converts the corpus to an fp16 .bin cache next to the fvecs.

Usage:
    python3 bench_marsco_b200.py --m 1000000 --k 128
Env: BS(=64) SAMPLE_SIZE NUM_BUCKETS(=256) REFRESH(=0) WARMUP ITERS BASELINE(torch|oursdense)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "litetopk_engine"))
from tidal_hopper_ops import fused_ip_gqa_sparse_b200  # noqa: E402

DATA = os.environ.get("MARSCO_DATA", "/data/marsco")


def read_fvecs(path, max_rows=None):
    a = np.memmap(path, dtype="int32", mode="r")
    d = int(a[0])
    n = a.size // (d + 1)
    if max_rows is not None:
        n = min(n, max_rows)
    fv = np.asarray(a.reshape(-1, d + 1)[:n, 1:]).view("float32")
    return fv, d


def load_base_fp16(m_rows):
    cache = os.path.join(DATA, "base_5m_fp16_768.bin")
    if not os.path.exists(cache):
        print("converting base_5m.fvecs -> fp16 cache (one-time)...", flush=True)
        fv, d = read_fvecs(os.path.join(DATA, "base_5m.fvecs"))
        assert d == 768, d
        fv.astype(np.float16).tofile(cache)
        del fv
    base = np.memmap(cache, dtype="float16", mode="r").reshape(-1, 768)
    assert m_rows <= base.shape[0], f"corpus has {base.shape[0]} rows"
    t = torch.from_numpy(np.ascontiguousarray(base[:m_rows])).cuda()
    return t


def cuda_time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def recall(got, ref):
    g = got.long().cpu().tolist()
    r = ref.cpu().tolist()
    return sum(len(set(a) & set(b)) for a, b in zip(g, r)) / (len(r) * len(r[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--k", type=int, default=int(os.environ.get("K", 128)))
    p.add_argument("--bs", type=int, default=int(os.environ.get("BS", 64)))
    p.add_argument("--sample-size", type=int, default=int(os.environ.get("SAMPLE_SIZE", 0)))
    p.add_argument("--num-buckets", type=int, default=int(os.environ.get("NUM_BUCKETS", 64)))  # NB=64: MARCO scores are O(70) so fp16 ULP≈0.06;
    # finer buckets make the select's fp16 bucket recompute disagree with the scan's fp32 bcount
    # (bucket width must stay >= several fp16 ULP) — same reason the h100 marsco bundle uses 64
    p.add_argument("--refresh", type=int, default=int(os.environ.get("REFRESH", -1)))  # -1 = auto
    p.add_argument("--num-ctas-x", type=int, default=int(os.environ.get("NUM_CTAS_X", 0)))
    p.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", 10)))
    p.add_argument("--iters", type=int, default=int(os.environ.get("ITERS", 20)))
    args = p.parse_args()

    assert args.m % 64 == 0, "M must be a multiple of 64"
    dev = "cuda"
    base = load_base_fp16(args.m)                      # [M, 768] fp16
    qv, d = read_fvecs(os.path.join(DATA, "query.fvecs"), max_rows=args.bs)
    q = torch.from_numpy(qv.astype(np.float16)).to(dev).contiguous()
    M, D = base.shape
    k = args.k
    sample = args.sample_size or min(M, max(16384, min(8 * k, 262144)))
    if args.refresh < 0:
        args.refresh = 8   # full-grid sweep: REFRESH=8 won every (M, k) cell
    # Engine-shape auto policy (from sweep_marsco_b200.py's full grid):
    #   QN64+BM256 (single-pass corpus, 2 math warpgroups, grid 296) wins
    #   everywhere except small-M large-k, where the QN8 warp-queue path with
    #   BM=256 wins (emission-bound: queue batching + 8-way group parallelism).
    if "TIDAL_FLAT_QN" not in os.environ:
        # Forced single engine shape: QN64+BM256 everywhere (no QN8 fallback).
        os.environ["TIDAL_FLAT_QN"] = "64"
        os.environ["TIDAL_BM"] = "256"
        if args.num_ctas_x == 0:
            args.num_ctas_x = 296
    print(f"device={torch.cuda.get_device_name(0)} M={M} D={D} bs={q.shape[0]} k={k} "
          f"sample={sample} NB={args.num_buckets} refresh={args.refresh}", flush=True)

    kv3 = base.unsqueeze(0)                            # [1, M, D] flat-batch layout

    if os.environ.get("BASELINE", "torch") == "oursdense":
        import tidal_hopper_ops as _ops
        _ops._ensure_loaded()
        def baseline():
            return torch.ops.tidal_sm100.dense_scores(q, kv3, 0).topk(k, dim=-1).indices
        base_name = "ours-dense qk+topk"
    else:
        def baseline():
            return (q @ base.t()).topk(k, dim=-1).indices
        base_name = "dense matmul+topk (torch)"
    ref = baseline()

    def ours():
        return fused_ip_gqa_sparse_b200(
            q, kv3, k, num_buckets=args.num_buckets, sample_size=sample,
            refresh_every=args.refresh, num_ctas_x=args.num_ctas_x,
            sample_mode=1, out_fp32=os.environ.get("OUT_FP32", "1") == "1")[1]  # strided sample: MARCO corpus ordering makes tail windows unrepresentative

    got = ours()
    torch.cuda.synchronize()
    rec = recall(got, ref)
    print(f"recall vs {base_name} @k={k}: {rec:.4f}", flush=True)

    t_base = cuda_time(baseline, args.warmup, args.iters)
    t_ours = cuda_time(ours, args.warmup, args.iters)
    print(f"\n  {base_name:<26}: {t_base:.4f} ms", flush=True)
    print(f"  LiteTopK SM100 fused       : {t_ours:.4f} ms   speedup {t_base / t_ours:.2f}x", flush=True)
    print("RESULT:", "PASS" if rec >= 0.99 else "RECALL_LOW", flush=True)


if __name__ == "__main__":
    main()
