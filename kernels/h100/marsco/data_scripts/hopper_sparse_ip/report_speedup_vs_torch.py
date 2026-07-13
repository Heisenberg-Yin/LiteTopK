"""Speedup report: Hopper sparse IP top-k vs torch.topk over (batch, N, k).

Baseline: torch.topk(q @ base.T, k)  -- the dense matmul + full sort that a
naive IP top-k does.  Under test: the fused sparse path (bs_common.sparse_topk_ip),
which routes n<=32 to fused_ip_smalln_sparse (m64n8/m64n32) and n%64==0 to
fused_ip_sparse.  Data is real MSMARCO embeddings (D=768) so gate selectivity
matches production, not random.

Both are timed with CUDA events (best of `reps` windows of `iters` calls each,
after `warmup`).  We also report index recall@k of the sparse result vs the
torch top-k so the speedup is only credited when the answer is correct.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_DATA_DIR = os.path.abspath(os.path.join(_HERE, "../../../data/marsco"))

from bs_common import sparse_topk_ip  # noqa: E402

_SAMPLE = 131072


def fvec_shape(path: str) -> tuple[int, int]:
    dim = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    rows = os.path.getsize(path) // ((dim + 1) * 4)
    return rows, dim


def read_fvecs(path: str, limit: int, dim: int | None = None) -> np.ndarray:
    rows, src_dim = fvec_shape(path)
    if dim is None:
        dim = src_dim
    if dim > src_dim:
        raise ValueError(f"dim={dim} exceeds source dim={src_dim}")
    n = min(limit, rows)
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(rows, src_dim + 1))
    return np.array(mm[:n, 1:1 + dim], dtype=np.float32, copy=True, order="C")


def recall_at_k(got: torch.Tensor, ref: torch.Tensor) -> float:
    g = got.detach().long().cpu().tolist()
    r = ref.detach().long().cpu().tolist()
    return sum(len(set(a) & set(b)) for a, b in zip(g, r)) / (len(r) * len(r[0]))


def cuda_best_ms(fn, warmup: int, iters: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best


@torch.no_grad()
def torch_topk(q: torch.Tensor, base: torch.Tensor, k: int):
    return torch.topk(q @ base.t(), k, dim=1, largest=True, sorted=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=os.path.join(_DATA_DIR, "base_5m.fvecs"))
    p.add_argument("--query", default=os.path.join(_DATA_DIR, "query.fvecs"))
    p.add_argument("--bs-list", default="1,8,32,64,128")
    p.add_argument("--m-list", default="262144,1000000,4000000")
    p.add_argument("--k-list", default="10,100,1000")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--reps", type=int, default=3)
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    args = parse_args()
    bs_list = [int(x) for x in args.bs_list.split(",") if x.strip()]
    m_list = [(int(x) // 64) * 64 for x in args.m_list.split(",") if x.strip()]
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    max_m = max(m_list)
    max_bs = max(bs_list)
    for bs in bs_list:
        if not (bs <= 32 or bs % 64 == 0):
            raise ValueError(f"bs={bs} unsupported (need <=32 or a multiple of 64)")

    rows, src_dim = fvec_shape(args.base)
    qrows, qdim = fvec_shape(args.query)
    if max_m > rows:
        raise ValueError(f"max M={max_m} exceeds base rows={rows}")
    if max_bs > qrows:
        raise ValueError(f"max bs={max_bs} exceeds query rows={qrows}")
    if args.dim > src_dim or args.dim > qdim:
        raise ValueError(f"dim={args.dim} exceeds base/query dim {src_dim}/{qdim}")

    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)
    print(f"base={args.base} D={args.dim} S={_SAMPLE} "
          f"warmup={args.warmup} iters={args.iters} reps={args.reps}", flush=True)
    t0 = time.time()
    base_full = torch.from_numpy(read_fvecs(args.base, max_m, args.dim)).to(
        "cuda", dtype=torch.float16).contiguous()
    q_full = torch.from_numpy(read_fvecs(args.query, max_bs, args.dim)).to(
        "cuda", dtype=torch.float16).contiguous()
    print(f"loaded base={tuple(base_full.shape)} q={tuple(q_full.shape)} "
          f"in {time.time() - t0:.1f}s\n", flush=True)

    hdr = "{:>5} {:>9} {:>6} {:>11} {:>11} {:>9} {:>8}".format(
        "bs", "N", "k", "torch_ms", "sparse_ms", "speedup", "recall")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for bs in bs_list:
        q = q_full[:bs].contiguous()
        for m in m_list:
            base = base_full[:m].contiguous()
            for k in k_list:
                if k > m:
                    continue
                ref_v, ref_i = torch_topk(q, base, k)
                sp_v, sp_i = sparse_topk_ip(q, base, k, sample_size=_SAMPLE)
                rec = recall_at_k(sp_i, ref_i)
                t_torch = cuda_best_ms(lambda: torch_topk(q, base, k),
                                       args.warmup, args.iters, args.reps)
                t_sparse = cuda_best_ms(lambda: sparse_topk_ip(q, base, k, sample_size=_SAMPLE),
                                        args.warmup, args.iters, args.reps)
                spd = t_torch / t_sparse if t_sparse > 0 else float("nan")
                print("{:>5} {:>9} {:>6} {:>11.4f} {:>11.4f} {:>8.2f}x {:>8.4f}".format(
                    bs, m, k, t_torch, t_sparse, spd, rec), flush=True)
                del ref_v, ref_i, sp_v, sp_i
            del base
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
