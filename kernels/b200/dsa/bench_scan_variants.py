#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Microbenchmark the fused sparse SCAN kernel in isolation under different
configs, to locate the bottleneck before optimizing:

  dense       our KV-split dense kernel (writes full [Q,S])
  sparse      default fused sparse scan (REFRESH as configured)
  sparse-nr   sparse scan with refresh_every=0 (no bcount atomics, static gate)
  null-epi    sparse build with -DDSA_NULL_EPILOGUE (score + reduce, no gate/store)
              -> the pure TMA/UMMA/TMEM+reduction floor

Run inside the `simtopk` container:
  PYTHONPATH=/opt/venvs/deepgemm/lib/python3.12/site-packages \
  DSA_CACHE_DIR=/data/dsa_caches \
  /usr/bin/python3.12 bench_scan_variants.py 256k:4096 1m:8192
"""
import os, sys, torch
from torch.utils.cpp_extension import load

import b200_config as cfg
cfg.prepare_env()
from safetensors import safe_open

K = int(os.environ.get("K", "2048")); NB = int(os.environ.get("NB", "256"))
REFRESH = int(os.environ.get("REFRESH", "64")); SAMPLE = int(os.environ.get("SAMPLE", "65536"))
WARMUP = int(os.environ.get("WARMUP", "5")); ITERS = int(os.environ.get("ITERS", "20"))
NULL_EPI = os.environ.get("NULL_EPI", "1") == "1"

CELLS = sys.argv[1:] or ["256k:4096", "1m:8192"]


FLAGS_EXTRA = os.environ.get("FLAGS_EXTRA", "").split()
BUILD_TAG = os.environ.get("BUILD_TAG", "")


def build_one(name, extra):
    name = name + BUILD_TAG
    d = f"/tmp/build_{name}"
    os.makedirs(d, exist_ok=True)
    return load(name=f"github_dsa_b200_{name}", sources=[cfg.SRC], extra_include_paths=cfg.INC,
                extra_cuda_cflags=cfg.FLAGS + FLAGS_EXTRA + extra, build_directory=d, extra_ldflags=["-lcuda"], verbose=False)


def evt(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(ITERS): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / ITERS


def load_cache(tag, chunk):
    d = os.environ.get("DSA_CACHE_DIR", "/data/dsa_caches")
    p = os.path.join(d, f"glm5_{tag}_realtext_chunk{chunk}.safetensors")
    T = {}
    with safe_open(p, "pt", device="cuda") as f:
        for k in f.keys(): T[k] = f.get_tensor(k)
    q = T["q_index"].contiguous(); qs = T["q_index_scale"].squeeze(-1)
    kf = T["idx_k_cache"][0].contiguous(); ksc = T["idx_k_scale"][0, :, 0].contiguous()
    gw = T["gate_w"]; Q, H, D = q.shape
    w = (gw * qs * (D ** -0.5)).contiguous().float()
    return q, kf, ksc, w


def dense_scores(dn, q, kf, ksc, w, S):
    Q = q.shape[0]
    ks0 = torch.zeros(Q, dtype=torch.int32, device="cuda"); ke = torch.full((Q,), kf.shape[0], dtype=torch.int32, device="cuda")
    o0 = torch.zeros(Q, device="cuda"); i1 = torch.ones(Q, device="cuda"); th0 = torch.zeros(Q, dtype=torch.int32, device="cuda")
    sv0 = torch.zeros(Q, 1, device="cuda"); si0 = torch.zeros(Q, 1, dtype=torch.int32, device="cuda")
    cv, _, _ = dn.mqa_logits_dsa_marsco(q, kf, ksc, w, ks0, ke, o0, i1, th0, sv0, si0, NB, S, K, 0, -1)
    return cv


def main():
    sp = build_one("sparse_b200", [])
    dn = build_one("dense_b200", ["-DDENSE_WRITE"])
    ne = build_one("nullepi_b200", ["-DDSA_NULL_EPILOGUE"]) if NULL_EPI else None

    for cell in CELLS:
        tag, chunk = cell.split(":"); chunk = int(chunk)
        q, kf, ksc, w = load_cache(tag, chunk)
        Q = q.shape[0]; Smax = kf.shape[0]
        ke_full = torch.full((Q,), Smax, dtype=torch.int32, device="cuda")

        head = min(SAMPLE, Smax); pos = torch.arange(0, head, device="cuda", dtype=torch.long)
        ksamp = kf[pos].contiguous(); kss = ksc[pos].contiguous()
        ks_scan = torch.full((Q,), head, dtype=torch.int32, device="cuda")
        col_full = pos.to(torch.int32).view(1, -1).expand(Q, -1).contiguous()
        cnt_full = torch.full((Q,), head, dtype=torch.int32, device="cuda")

        sl = dense_scores(dn, q, ksamp, kss, w, head)
        xs = -sl; o2 = xs.min(dim=1).values.contiguous(); hi = xs.max(dim=1).values
        inv2 = ((NB - 1) / (hi - o2).clamp_min(1e-20)).contiguous()
        sv, si = sp.compact_topk_min_idx_marsco(xs, col_full, cnt_full, K)
        sv = sv.contiguous(); si = si.contiguous()
        th2 = ((sv.max(dim=1).values - o2) * inv2).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
        del sl, xs, hi
        torch.cuda.empty_cache()

        def t_dense():
            cv = dense_scores(dn, q, kf, ksc, w, Smax); del cv
        def mk_sparse(mod, refresh):
            def fn():
                th = th2.clone()
                mod.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_full, o2, inv2, th, sv, si, NB, 1, K, refresh, -1)
            return fn

        r = {}
        r["dense"] = evt(t_dense)
        r["sparse"] = evt(mk_sparse(sp, REFRESH))
        r["sparse-nr"] = evt(mk_sparse(sp, 0))
        if ne is not None:
            r["null-epi"] = evt(mk_sparse(ne, REFRESH))

        # candidate count under default config
        th = th2.clone()
        _, _, cc = sp.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_full, o2, inv2, th, sv, si, NB, 1, K, REFRESH, -1)
        emitted = (cc - K).float()
        print(f"[{tag} chunk={chunk}] Q={Q} S={Smax}  emitted/row mean={emitted.mean().item():.0f} max={emitted.max().item():.0f}")
        for k, v in r.items():
            print(f"  {k:>9} = {v:8.3f} ms")
        sys.stdout.flush()
        del q, kf, ksc, w
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
