"""Full hyper-parameter sweep for the MARSCO B200 op: engine shape x grid x
refresh x sample x buckets, per (M, k), in-process (corpus + dense reference
loaded once per M).

Grid (pruned to the knobs that measurably matter):
  engine  : (qn=8, bm=128) warp-queue path | (qn=8, bm=256) | (qn=64, bm=256)
  ctas    : 0 (auto) | 296
  refresh : 0 | 8
  sample  : 8k-clamped | 16k-clamped   (clamp [16384, 262144])
  NB      : 64  (marsco fp16 bucket-width constraint)

Usage:
    python3 sweep_marsco_b200.py [--m 1000000 ...] [--k 128 1024 4096 8192]
Prints RESULT lines + per-(M, k) BEST (recall >= 0.99).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "litetopk_engine"))
from bench_marsco_b200 import read_fvecs, load_base_fp16  # noqa: E402
from tidal_hopper_ops import fused_ip_gqa_sparse_b200  # noqa: E402


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, nargs="+",
                   default=[1000000, 2000000, 4000000, 5000000])
    p.add_argument("--k", type=int, nargs="+", default=[128, 1024, 4096, 8192])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--min-recall", type=float, default=0.99)
    args = p.parse_args()

    dev = "cuda"
    qv, _ = read_fvecs(os.path.join(
        "/data/marsco", "query.fvecs"), max_rows=64)
    q = torch.from_numpy(qv.astype(np.float16)).to(dev).contiguous()
    best = {}
    for M in args.m:
        base = load_base_fp16(M)
        kv3 = base.unsqueeze(0)
        scores = q @ base.t()
        for k in args.k:
            ref_sets = [set(r) for r in scores.topk(k, dim=-1).indices.cpu().tolist()]
            dense_ms = cuda_time(lambda: (q @ base.t()).topk(k, dim=-1).indices,
                                 args.warmup, args.iters)
            for qn, bm in ((8, 128), (8, 256), (64, 256)):
                for ctas in (0, 296):
                    for rf in (0, 8):
                        for smult in (8, 16):
                            sample = min(M, max(16384, min(smult * k, 262144)))
                            def run():
                                return fused_ip_gqa_sparse_b200(
                                    q, kv3, k, num_buckets=64, sample_size=sample,
                                    refresh_every=rf, num_ctas_x=ctas,
                                    sample_mode=1, qn=qn, bm=bm)[1]
                            try:
                                got = run()
                                torch.cuda.synchronize()
                                g = got.long().cpu().tolist()
                                rec = sum(len(set(a) & b) for a, b in zip(g, ref_sets)) / (64 * k)
                                ms = cuda_time(run, args.warmup, args.iters)
                            except Exception as ex:  # noqa: BLE001
                                print(f"RESULT M={M} k={k} qn={qn} bm={bm} ctas={ctas} "
                                      f"rf={rf} s={sample} ERROR {str(ex)[:60]}", flush=True)
                                continue
                            print(f"RESULT M={M} k={k} qn={qn} bm={bm} ctas={ctas} rf={rf} "
                                  f"s={sample} ms={ms:.4f} recall={rec:.4f}", flush=True)
                            key = (M, k)
                            if rec >= args.min_recall and (key not in best or ms < best[key][0]):
                                best[key] = (ms, dense_ms, qn, bm, ctas, rf, sample, rec)
        del base, kv3, scores
        torch.cuda.empty_cache()

    print("\n==== per-(M, k) best ====", flush=True)
    for (M, k), (ms, dms, qn, bm, ctas, rf, sample, rec) in sorted(best.items()):
        print(f"BEST M={M} k={k}: {ms:.4f} ms ({dms/ms:.2f}x vs dense {dms:.3f})  "
              f"qn={qn} bm={bm} ctas={ctas} rf={rf} s={sample} recall={rec:.4f}", flush=True)


if __name__ == "__main__":
    main()
