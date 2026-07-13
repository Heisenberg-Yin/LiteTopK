"""Probe: does Triton 3.5.1 emit tcgen05 (MMAv5) on sm_100a, and under what dot shapes?"""
import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def dot_kernel(x_ptr, c_ptr, o_ptr,
               BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    mo = tl.arange(0, BM)
    no = tl.arange(0, BN)
    do = tl.arange(0, BD)
    c = tl.load(c_ptr + mo[:, None] * BD + do[None, :])           # [BM, BD]
    x = tl.load(x_ptr + no[:, None] * BD + do[None, :])           # [BN, BD]
    acc = tl.dot(c, tl.trans(x))                                   # [BM, BN]
    tl.store(o_ptr + mo[:, None] * BN + no[None, :], acc)


def probe(bm, bn, bd, nw):
    c = torch.randn(bm, bd, device="cuda", dtype=torch.float16)
    x = torch.randn(bn, bd, device="cuda", dtype=torch.float16)
    o = torch.empty(bm, bn, device="cuda", dtype=torch.float32)
    h = dot_kernel[(1,)](x, c, o, BM=bm, BN=bn, BD=bd, num_warps=nw)
    ptx = h.asm["ptx"]
    tc = ptx.count("tcgen05")
    mma = ptx.count("mma.sync")
    print(f"BM={bm:4d} BN={bn:4d} BD={bd:3d} nw={nw}: tcgen05={tc:3d} mma.sync={mma:3d}")


for bm, bn, nw in [(128, 8, 4), (128, 16, 4), (128, 64, 4), (256, 64, 4),
                   (64, 64, 4), (128, 64, 8), (8, 128, 4), (16, 128, 4)]:
    try:
        probe(bm, bn, 64, nw)
    except Exception as e:
        print(f"BM={bm} BN={bn} nw={nw}: FAIL {type(e).__name__} {str(e)[:80]}")
