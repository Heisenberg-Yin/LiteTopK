"""Compare the B200 LiteTopK MS MARCO operator with a dense reference.

The input corpus and queries are fp16. ``--m`` selects a prefix of the
corpus, while ``--k`` and ``--bs`` select the output width and query count.
The script reports set-overlap recall and CUDA-event latency for both paths.

Required files under ``MARSCO_DATA``:

* ``query.fvecs``
* ``base_5m_fp16_768.bin``; if absent, ``base_5m.fvecs`` is converted and
  the cache is written to the same directory.

Usage:
    python3 bench_marsco_b200.py --m 1000000 --k 128

See ``README.md`` for the CUDA include paths required by the JIT build.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from litetopk_ops import fused_ip_sparse_b200  # noqa: E402

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
    # Copy-on-write keeps the mapping writable for torch.from_numpy without
    # copying the full corpus or modifying the cache file.
    base = np.memmap(cache, dtype="float16", mode="c").reshape(-1, 768)
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
    p.add_argument("--num-buckets", type=int, default=int(os.environ.get("NUM_BUCKETS", 64)))
    p.add_argument("--refresh", type=int, default=int(os.environ.get("REFRESH", -1)))
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
        args.refresh = 8
    # Fix the benchmark launch shape unless the caller explicitly supplies
    # another supported value.
    if "LITETOPK_FLAT_QN" not in os.environ:
        os.environ["LITETOPK_FLAT_QN"] = "64"
        os.environ["LITETOPK_BM"] = "256"
        if args.num_ctas_x == 0:
            args.num_ctas_x = 296
    print(f"device={torch.cuda.get_device_name(0)} M={M} D={D} bs={q.shape[0]} k={k} "
          f"sample={sample} NB={args.num_buckets} refresh={args.refresh}", flush=True)

    kv3 = base.unsqueeze(0)                            # [1, M, D] flat-batch layout

    baseline_kind = os.environ.get("BASELINE", "torch")
    if baseline_kind not in ("torch", "engine_dense"):
        raise ValueError("BASELINE must be 'torch' or 'engine_dense'")
    if baseline_kind == "engine_dense":
        import litetopk_ops as _ops
        _ops._ensure_loaded()
        def baseline():
            return torch.ops.litetopk_sm100.dense_scores(q, kv3, 0).topk(k, dim=-1).indices
        base_name = "engine dense scores + torch.topk"
    else:
        def baseline():
            return (q @ base.t()).topk(k, dim=-1).indices
        base_name = "dense matmul+topk (torch)"
    ref = baseline()

    def litetopk():
        return fused_ip_sparse_b200(
            q, kv3, k, num_buckets=args.num_buckets, sample_size=sample,
            refresh_every=args.refresh, num_ctas_x=args.num_ctas_x,
            sample_mode=1,
            out_fp32=os.environ.get("OUT_FP32", "1") == "1",
        )[1]

    got = litetopk()
    torch.cuda.synchronize()
    rec = recall(got, ref)
    print(f"recall vs {base_name} @k={k}: {rec:.4f}", flush=True)

    t_base = cuda_time(baseline, args.warmup, args.iters)
    t_litetopk = cuda_time(litetopk, args.warmup, args.iters)
    print(f"\n  {base_name:<26}: {t_base:.4f} ms", flush=True)
    print(f"  LiteTopK SM100 fused       : {t_litetopk:.4f} ms   speedup {t_base / t_litetopk:.2f}x", flush=True)
    print("RESULT:", "PASS" if rec >= 0.99 else "RECALL_LOW", flush=True)


if __name__ == "__main__":
    main()
