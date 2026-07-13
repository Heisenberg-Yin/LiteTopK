#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-level timing breakdown of the LiteTopK E2E path on the REAL chunked
GLM-5 caches, to direct kernel optimization:

  O_prep      = dense scoring on the SAMPLE prefix + seed radix select + thr math
  scan        = the fused sparse scoring kernel (mqa_logits_dsa_marsco)
  select      = compact_topk_min_thr_marsco (boundary radix select)

Also reports per-row emitted-candidate counts (cand_cnt - seed) which size the
atomic/HBM write traffic of the sparse epilogue.

Run inside the `simtopk` container:
  PYTHONPATH=/opt/venvs/deepgemm/lib/python3.12/site-packages \
  DSA_CACHE_DIR=/data/dsa_caches \
  /usr/bin/python3.12 profile_breakdown.py 256k:4096 1m:8192
"""
import os, sys, torch
from torch.utils.cpp_extension import load

import b200_config as cfg
cfg.prepare_env()
from safetensors import safe_open

K = int(os.environ.get("K", "2048")); NB = int(os.environ.get("NB", "256"))
REFRESH = int(os.environ.get("REFRESH", "64")); SAMPLE = int(os.environ.get("SAMPLE", "65536"))
WARMUP = int(os.environ.get("WARMUP", "5")); ITERS = int(os.environ.get("ITERS", "20"))

CELLS = sys.argv[1:] or ["256k:4096", "1m:8192"]


def build():
    os.makedirs("/tmp/build_sparse_b200", exist_ok=True); os.makedirs("/tmp/build_dense_b200", exist_ok=True)
    sp = load(name="github_dsa_b200_sparse", sources=[cfg.SRC], extra_include_paths=cfg.INC, extra_cuda_cflags=cfg.FLAGS,
              build_directory="/tmp/build_sparse_b200", extra_ldflags=["-lcuda"], verbose=False)
    dn = load(name="github_dsa_b200_dense", sources=[cfg.SRC], extra_include_paths=cfg.INC, extra_cuda_cflags=cfg.FLAGS + ["-DDENSE_WRITE"],
              build_directory="/tmp/build_dense_b200", extra_ldflags=["-lcuda"], verbose=False)
    return sp, dn


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
    sp, dn = build()
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

        def O_prep():
            sl = dense_scores(dn, q, ksamp, kss, w, head)
            xs = -sl; o2 = xs.min(dim=1).values.contiguous(); hi = xs.max(dim=1).values
            inv2 = ((NB - 1) / (hi - o2).clamp_min(1e-20)).contiguous()
            sv, si = sp.compact_topk_min_idx_marsco(xs, col_full, cnt_full, K)
            sv = sv.contiguous(); si = si.contiguous()
            th2 = ((sv.max(dim=1).values - o2) * inv2).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
            return o2, inv2, th2, sv, si

        o2, inv2, th2, sv, si = O_prep()

        def scan():
            th = th2.clone()  # scan mutates th_bucket; keep th2 pristine per-iter
            return sp.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_full, o2, inv2, th, sv, si, NB, 1, K, REFRESH, -1)

        cv2, ci2, cc2 = scan(); torch.cuda.synchronize()

        def select():
            return sp.compact_topk_min_thr_marsco(cv2, ci2, cc2, o2, inv2, th2, NB, K)

        t_prep = evt(O_prep)
        t_scan = evt(scan)
        t_sel = evt(select)

        # thr clone overhead is measured inside scan(); estimate it
        def clone_only(): th2.clone()
        t_clone = evt(clone_only)

        emitted = (cc2 - K).float()  # cand_cnt starts at seed_k == K
        tot = t_prep + t_scan + t_sel
        print(f"[{tag} chunk={chunk}] Q={Q} S={Smax}")
        print(f"  O_prep = {t_prep:8.3f} ms ({100*t_prep/tot:4.1f}%)")
        print(f"  scan   = {t_scan:8.3f} ms ({100*t_scan/tot:4.1f}%)  (incl. th.clone ~{t_clone:.3f} ms)")
        print(f"  select = {t_sel:8.3f} ms ({100*t_sel/tot:4.1f}%)")
        print(f"  total  = {tot:8.3f} ms")
        print(f"  emitted cand/row (excl. {K} seed): mean={emitted.mean().item():.0f} "
              f"min={emitted.min().item():.0f} max={emitted.max().item():.0f}  (K={K})", flush=True)
        del q, kf, ksc, w, cv2, ci2, cc2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
