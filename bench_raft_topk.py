"""Baseline 2 — ``torch.matmul`` + RAFT ``select_k`` large-K KNN.

Same scoring as the torch baseline (``s = ‖c‖² − 2⟨x,c⟩`` via one matmul), but
the top-k is RAFT's ``pylibraft.matrix.select_k`` (radix select) instead of
``torch.topk``. Times the whole call (matmul + select_k + handle.sync) over
``--iters`` runs (default 100) and reports the mean.

Run with the python10 env (has pylibraft):
    /mnt/hdd/yinziqi/miniconda3/envs/python10/bin/python \\
        efficientlargek/bench_raft_topk.py --batch-size 512 --k 10000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pylibraft.common import DeviceResources
from pylibraft.matrix import select_k

from benchmark import _DEFAULT_BASE, _DEFAULT_QUERY, _time_cuda, read_fvecs


@torch.no_grad()
def matmul_raft_topk(x, c, k, handle, out_dist, out_idx):
    """``(B,D)``×``(M,D)`` → top-k smallest squared-L2 per query via RAFT.

    ``csq = ‖c‖²`` is computed here (inside the timed region) so the
    corpus-side preprocessing is counted, matching the other two baselines.
    ``out_dist`` is the *true* squared-L2 distance ``‖x−c‖² = s + ‖x‖²``,
    recovered from the shifted score ``s = ‖c‖² − 2⟨x,c⟩`` (same as flashlargek)."""
    csq = (c * c).sum(dim=-1)                       # ‖c‖²
    cross = torch.matmul(x, c.t())                  # (B, M)
    score = (csq.unsqueeze(0) - 2.0 * cross).contiguous()
    select_k(score, k=k, distances=out_dist, indices=out_idx,
             select_min=True, handle=handle)
    handle.sync()
    xsq = (x * x).sum(dim=-1)                        # ‖x‖²
    out_dist.add_(xsq.unsqueeze(1)).clamp_min_(0.0)  # true ‖x−c‖²
    return out_dist, out_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=_DEFAULT_BASE)
    p.add_argument("--query", default=_DEFAULT_QUERY)
    p.add_argument("--num-base", type=int, default=1_000_000, help="corpus rows M")
    p.add_argument("--batch-size", type=int, default=512, help="queries per call")
    p.add_argument("--k", type=int, default=10_000)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    X = torch.from_numpy(read_fvecs(args.base, limit=args.num_base)).to(device).float()
    Q = torch.from_numpy(read_fvecs(args.query)).to(device).float()
    M, D = X.shape
    B = min(args.batch_size, Q.shape[0])
    k = min(args.k, M)
    x = Q[:B].contiguous()                    # queries from query.fvecs
    c = X                                     # corpus
    assert x.shape[1] == D, f"query dim {x.shape[1]} != corpus dim {D}"
    print(f"M={M} D={D} batch_size={B} (of {Q.shape[0]} queries) k={k}")

    handle = DeviceResources()
    out_dist = torch.empty(B, k, device=device, dtype=torch.float32)
    out_idx = torch.empty(B, k, device=device, dtype=torch.int64)

    fn = lambda: matmul_raft_topk(x, c, k, handle, out_dist, out_idx)

    torch.cuda.reset_peak_memory_stats()
    _time_cuda(fn, warmup=args.warmup, iters=args.iters,
               label=f"torch.matmul+raft.select_k B={B} M={M} k={k}")
    print(f"peak GPU mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
