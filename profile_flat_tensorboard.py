"""Disabled profiling entry point.

Use ``benchmark.py`` as the single active benchmark/test script. The previous
TensorBoard profiling helper is commented below for reference.
"""

'''
"""TensorBoard/profiler breakdown for flashlargek flat path."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.profiler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import _DEFAULT_BASE, _DEFAULT_QUERY, read_fvecs
from flashlargek import (
    _NUM_BUCKETS,
    _flashlargek_flat_kernel,
    _heuristic_flashlargek_config,
    _sample_range_per_row,
)


def _event_time(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


@torch.no_grad()
def flashlargek_flat_profiled(x, c, k, *, return_distances=True):
    import math

    N, D = x.shape
    M, Dc = c.shape
    assert D == Dc and 1 <= k <= M
    device = x.device

    timings = defaultdict(float)

    cfg = _heuristic_flashlargek_config(N=N, M=M, D=D)
    BN, BM, D_INNER = cfg["BN"], cfg["BM"], cfg["D_INNER"]
    CHUNK = BN
    BUF = min(10 * k, M)

    c = c.contiguous()
    with torch.profiler.record_function("flat/sample_range"):
        ms, sample_out = _event_time(lambda: _sample_range_per_row(x, c, k))
    timings["sample_range_ms"] += ms
    bucket_origin, bucket_inv_delta = sample_out

    out_idxs = torch.empty((N, k), device=device, dtype=torch.int32)
    out_shift = (
        torch.empty((N, k), device=device, dtype=torch.float32)
        if return_distances
        else None
    )

    bcount = torch.empty((CHUNK, _NUM_BUCKETS), device=device, dtype=torch.int32)
    threshold = torch.empty((CHUNK,), device=device, dtype=torch.int32)
    wcount = torch.zeros((1,), device=device, dtype=torch.int32)
    qcount = torch.empty((CHUNK,), device=device, dtype=torch.int32)
    buf_val = torch.empty((CHUNK, BUF), device=device, dtype=torch.float32)
    buf_idx = torch.empty((CHUNK, BUF), device=device, dtype=torch.int32)

    cnt_every = max(1, k // 2)
    grid_m = math.ceil(M / BM)

    for q0 in range(0, N, CHUNK):
        q1 = min(q0 + CHUNK, N)
        R = q1 - q0
        xb = x[q0:q1].contiguous()
        ob = bucket_origin[q0:q1]
        ib = bucket_inv_delta[q0:q1]
        bc = bcount[:R]
        th = threshold[:R]
        qc = qcount[:R]
        bv = buf_val[:R]
        bi = buf_idx[:R]

        with torch.profiler.record_function("flat/reset_buffers"):
            ms, _ = _event_time(
                lambda: (
                    bc.zero_(),
                    th.fill_(_NUM_BUCKETS - 1),
                    qc.zero_(),
                    wcount.zero_(),
                    bv.fill_(float("inf")),
                )
            )
        timings["reset_buffers_ms"] += ms

        with torch.profiler.record_function("flat/fused_kernel"):
            ms, _ = _event_time(
                lambda: _flashlargek_flat_kernel[(grid_m, math.ceil(R / BN))](
                    xb,
                    c,
                    ob,
                    ib,
                    bc,
                    th,
                    wcount,
                    qc,
                    bv,
                    bi,
                    xb.stride(0),
                    xb.stride(1),
                    c.stride(0),
                    c.stride(1),
                    ob.stride(0),
                    bc.stride(0),
                    bc.stride(1),
                    th.stride(0),
                    qc.stride(0),
                    bv.stride(0),
                    bv.stride(1),
                    bi.stride(0),
                    bi.stride(1),
                    N=R,
                    M=M,
                    D=D,
                    K=k,
                    BUF=BUF,
                    BN=BN,
                    BM=BM,
                    D_INNER=D_INNER,
                    NUM_BUCKETS=_NUM_BUCKETS,
                    NUM_STAGES_PIPE=cfg["num_stages_pipe"],
                    CNT_EVERY=cnt_every,
                    num_warps=cfg["num_warps"],
                )
            )
        timings["fused_kernel_ms"] += ms

        with torch.profiler.record_function("flat/qcount_max_item"):
            ms, w = _event_time(lambda: max(int(qc.max().item()), k))
        timings["qcount_max_item_ms"] += ms

        with torch.profiler.record_function("flat/final_topk"):
            ms, topk_out = _event_time(
                lambda: torch.topk(
                    bv[:, :w], k=k, dim=-1, largest=False, sorted=False
                )
            )
        timings["final_topk_ms"] += ms
        topv, topp = topk_out

        with torch.profiler.record_function("flat/gather_indices"):
            ms, gathered = _event_time(lambda: torch.gather(bi[:, :w], 1, topp))
        timings["gather_indices_ms"] += ms
        out_idxs[q0:q1] = gathered
        if return_distances:
            out_shift[q0:q1] = topv

    if not return_distances:
        return out_idxs, timings

    with torch.profiler.record_function("flat/final_distance"):
        ms, out_vals = _event_time(
            lambda: (out_shift + (x.float() * x.float()).sum(dim=-1).unsqueeze(-1))
            .clamp_min_(0.0)
            .float()
        )
    timings["final_distance_ms"] += ms
    return (out_vals, out_idxs), timings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=_DEFAULT_BASE)
    p.add_argument("--query", default=_DEFAULT_QUERY)
    p.add_argument("--num-base", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--k", type=int, default=10_000)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--logdir", default="efficientlargek/tb_flat_profile")
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    X = torch.from_numpy(read_fvecs(args.base, limit=args.num_base)).to(device).float()
    Q = torch.from_numpy(read_fvecs(args.query)).to(device).float()
    M, D = X.shape
    B = args.batch_size
    k = min(args.k, M)
    x = Q[torch.arange(B, device=device) % Q.shape[0]].contiguous()
    c = X
    print(f"M={M} D={D} B={B} k={k} iters={args.iters}")

    for _ in range(args.warmup):
        flashlargek_flat_profiled(x, c, k, return_distances=True)
    torch.cuda.synchronize()

    totals = defaultdict(float)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.logdir),
    ) as prof:
        for _ in range(args.iters):
            _, timings = flashlargek_flat_profiled(x, c, k, return_distances=True)
            for name, ms in timings.items():
                totals[name] += ms
            prof.step()

    total_ms = sum(totals.values())
    print(f"tensorboard_logdir={os.path.abspath(args.logdir)}")
    print(f"component_total_ms_per_call={total_ms / args.iters:.4f}")
    for name, ms in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        mean_ms = ms / args.iters
        pct = 100.0 * ms / total_ms if total_ms else 0.0
        print(f"{name},{mean_ms:.4f},{pct:.2f}")

    print("\nProfiler CUDA top ops:")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=15
        )
    )


if __name__ == "__main__":
    main()
'''
