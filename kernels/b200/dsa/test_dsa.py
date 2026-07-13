#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-run correctness/latency check for the B200 (SM100) DSA fused kernel.

Mirrors the Hopper test_dsa.py but builds the SM100 UMMA/TMEM extension via
b200_config (arch=compute_100a, CUTLASS 4.2.1 + SM100 deep_gemm includes).

Usage:
    python3 test_dsa.py [256k 512k 768k 1m]
Env: CHUNK(=4096) K(=2048) SAMPLE(=65536) NB(=256) REFRESH(=0) V(=0)
     DSA_CACHE_DIR / DSA_CUTLASS_INCLUDE / DSA_DEEP_GEMM_ROOT to override paths.
"""
import os, sys, torch
from torch.utils.cpp_extension import load

import b200_config as cfg
cfg.prepare_env()
import deep_gemm
from safetensors import safe_open

mod = load(
    name="github_dsa_b200",
    sources=[cfg.SRC],
    extra_include_paths=cfg.INC,
    extra_cuda_cflags=cfg.FLAGS,
    extra_ldflags=["-lcuda"],
    verbose=os.environ.get("V", "0") == "1",
)


def evt(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters


def recall(idx, ref, Q, K):
    return 100 * sum(torch.isin(idx[i], ref[i]).sum().item() for i in range(Q)) / (Q * K)


def run(tag="1m", K=2048, Ssample=65536, NB=256, refresh=0):
    T = {}
    with safe_open(cfg.cache_path(tag), "pt", device="cuda") as f:
        for k in f.keys(): T[k] = f.get_tensor(k)
    q = T["q_index"].contiguous(); qs = T["q_index_scale"].squeeze(-1)
    kf = T["idx_k_cache"][0].contiguous(); ksc = T["idx_k_scale"][0, :, 0].contiguous()
    gw = T["gate_w"]; Q, H, D = q.shape; S = kf.shape[0]
    w = (gw * qs * (D ** -0.5)).contiguous().float()
    ks = torch.zeros(Q, dtype=torch.int32, device="cuda")
    ke_full = torch.full((Q,), S, dtype=torch.int32, device="cuda")
    ref_logits = deep_gemm.fp8_mqa_logits(q, (kf, ksc), w, ks, ke_full, clean_logits=True, max_seqlen_k=0).contiguous()
    ref = ref_logits.topk(K, dim=-1).indices.long()

    Ssample = min(Ssample, S); head_end = Ssample  # HEAD sample: [0, Ssample)
    pos = torch.arange(0, head_end, device="cuda", dtype=torch.long)
    ksamp = kf[pos].contiguous(); kss = ksc[pos].contiguous(); Ss = ksamp.shape[0]
    ke_s = torch.full((Q,), Ss, dtype=torch.int32, device="cuda")
    sl = deep_gemm.fp8_mqa_logits(q, (ksamp, kss), w, ks, ke_s, clean_logits=True, max_seqlen_k=0).contiguous()
    xs = -sl
    origin = xs.min(dim=1).values.contiguous(); hi = xs.max(dim=1).values
    inv = ((NB - 1) / (hi - origin).clamp_min(1e-20)).contiguous()
    seed_val, seed_pos = xs.topk(K, dim=-1, largest=False)
    seed_idx = pos[seed_pos].to(torch.int32).contiguous(); seed_val = seed_val.contiguous()
    th = ((seed_val[:, -1] - origin) * inv).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
    # Main kernel scans [Ssample, S) to avoid re-scanning the seeded head window.
    ks_scan = torch.full((Q,), head_end, dtype=torch.int32, device="cuda")
    ke_scan = torch.full((Q,), S, dtype=torch.int32, device="cuda")
    cv, ci, cc = mod.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_scan, origin, inv, th, seed_val, seed_idx, NB, 1, K, refresh)
    outv, outi = mod.compact_topk_min_idx_marsco(cv, ci, cc, K); torch.cuda.synchronize()
    r = recall(outi.long(), ref, Q, K)
    print(f"tag={tag} atomic head_seed={Ss:,} NB={NB} refresh={refresh} actual_cap=M")
    print(f"recall={r:.3f}% cnt_mean={cc.float().mean().item():.1f} max={cc.max().item()} overflow={(cc > S).sum().item()}")
    def run_bucket(): return mod.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_scan, origin, inv, th, seed_val, seed_idx, NB, 1, K, refresh)
    def run_select(): return mod.compact_topk_min_idx_marsco(cv, ci, cc, K)
    print(f"time spbucket_atomic={evt(run_bucket):.3f} ms compact_select={evt(run_select):.3f} ms")


if __name__ == "__main__":
    refresh = int(os.environ.get("REFRESH", "0"))
    K = int(os.environ.get("K", "2048"))
    Ssample = int(os.environ.get("SAMPLE", "65536"))
    NB = int(os.environ.get("NB", "256"))
    for tag in sys.argv[1:] or ["1m"]:
        run(tag, K=K, Ssample=Ssample, NB=NB, refresh=refresh)
