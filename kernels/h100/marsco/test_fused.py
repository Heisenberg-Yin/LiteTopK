"""Correctness + speed test for the fused (Triton matmul + CUDA FlashTopk) KNN.

校验 fused_knn_topk_l2 / fused_knn_topk_ip 与 torch 暴力 top-k 是否一致，并计时。

容器内运行（Triton 3.3 + nvcc 12.4）：
    docker exec ziqi_qrita_triton bash -lc \
      'cd /home/user/test/BBC-GPU/EfficientTopK && \
       python test_fused.py --num-base 200000 --num-queries 256 --k 100'
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))   # litetopk_fused / litetopk_ops
sys.path.insert(0, _HERE)                          # benchmark / _common

import numpy as np
import torch

from bench_all import read_fvecs, _DEFAULT_BASE, _time_cuda
from litetopk_fused import (
    fused_knn_topk_l2 as fused_knn_topk_l2,
    fused_knn_topk_ip as fused_knn_topk_ip,
)


@torch.no_grad()
def _ref_l2(x, c, k):
    d2 = torch.cdist(x.float(), c.float()).pow(2)  # (S, M)
    v, i = torch.topk(d2, k, dim=-1, largest=False, sorted=True)
    return v, i


@torch.no_grad()
def _ref_ip(x, c, k):
    ip = x.float() @ c.float().t()  # (S, M)
    v, i = torch.topk(ip, k, dim=-1, largest=True, sorted=True)
    return v, i


def _recall(got_idx, ref_idx, k):
    got = got_idx.long()
    ref = ref_idx.long()
    r = 0.0
    for a in range(got.shape[0]):
        r += len(set(got[a].tolist()) & set(ref[a].tolist())) / k
    return r / got.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=_DEFAULT_BASE)
    p.add_argument("--num-base", type=int, default=200_000)
    p.add_argument("--num-queries", type=int, default=256)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--recall-queries", type=int, default=64)
    args = p.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    base_np = read_fvecs(args.base, limit=args.num_base)
    print(f"read {base_np.shape} from {args.base} in {time.time()-t0:.1f}s")
    X = torch.from_numpy(base_np).to(device).float()
    M, D = X.shape
    N = min(args.num_queries, M)
    k = min(args.k, M)
    x = X[:N].contiguous()
    c = X
    print(f"M={M} N={N} D={D} k={k}")

    # ── L2 正确性 ──
    print("\n=== squared-L2 ===")
    v, i = fused_knn_topk_l2(x, c, k, return_distances=True, sorted=True)
    S = min(args.recall_queries, N)
    rv, ri = _ref_l2(x[:S], c, k)
    rec = _recall(i[:S], ri, k)
    dist_err = (v[:S].float() - rv.float()).abs().max().item()
    print(f"recall@{k} = {rec:.5f}   max|dist-ref| = {dist_err:.4g}")

    fn_l2 = lambda: fused_knn_topk_l2(x, c, k, return_distances=True)
    for _ in range(args.warmup):
        fn_l2()
    torch.cuda.synchronize()
    _time_cuda(fn_l2, warmup=0, iters=args.iters, label=f"fused-L2 N={N} M={M} k={k}")

    # ── IP 正确性 ──
    print("\n=== inner-product ===")
    v2, i2 = fused_knn_topk_ip(x, c, k, return_distances=True, sorted=True)
    rv2, ri2 = _ref_ip(x[:S], c, k)
    rec2 = _recall(i2[:S], ri2, k)
    ip_err = (v2[:S].float() - rv2.float()).abs().max().item()
    print(f"recall@{k} = {rec2:.5f}   max|ip-ref| = {ip_err:.4g}")

    fn_ip = lambda: fused_knn_topk_ip(x, c, k, return_distances=True)
    for _ in range(args.warmup):
        fn_ip()
    torch.cuda.synchronize()
    _time_cuda(fn_ip, warmup=0, iters=args.iters, label=f"fused-IP N={N} M={M} k={k}")


if __name__ == "__main__":
    main()
