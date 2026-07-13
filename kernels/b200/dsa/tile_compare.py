#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare tile configs for the SM100 DSA sparse kernel on real GLM-5 caches:
  cfg A = BLOCK_KV=128 / 1 math warpgroup (current default)
  cfg B = BLOCK_KV=256 / 2 math warpgroups (larger tile, uses more B200 smem/TMEM)

Baseline (DSA) is the shared dense kernel + torch.topk (same for both). Prints
LiteTopK latency for each tile config + speedup vs baseline, and recall.

Run in the `simtopk` container:
  PYTHONPATH=/opt/venvs/deepgemm/lib/python3.12/site-packages \
  DSA_CACHE_DIR=/data/dsa CHUNK=0 \
  /usr/bin/python3.12 tile_compare.py
"""
import os, sys, torch
from torch.utils.cpp_extension import load
import b200_config as cfg
cfg.prepare_env()
from safetensors import safe_open

K = int(os.environ.get("K", "2048")); NB = int(os.environ.get("NB", "256"))
REFRESH = int(os.environ.get("REFRESH", "64")); SAMPLE = int(os.environ.get("SAMPLE", "65536"))
WARMUP = int(os.environ.get("WARMUP", "5")); ITERS = int(os.environ.get("ITERS", "20"))
CELLS = [(t, int(c)) for t in os.environ.get("TAGS", "256k 1m").split()
         for c in os.environ.get("CHUNKS", "2048 4096").split()]


def build(name, extra, bd):
    os.makedirs(bd, exist_ok=True)
    return load(name=name, sources=[cfg.SRC], extra_include_paths=cfg.INC,
                extra_cuda_cflags=cfg.FLAGS + extra, extra_ldflags=["-lcuda"],
                build_directory=bd, verbose=False)


def evt(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(ITERS): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / ITERS


def dense_scores(dn, q, kf, ksc, w, S):
    Q = q.shape[0]
    ks0 = torch.zeros(Q, dtype=torch.int32, device="cuda"); ke = torch.full((Q,), kf.shape[0], dtype=torch.int32, device="cuda")
    o0 = torch.zeros(Q, device="cuda"); i1 = torch.ones(Q, device="cuda"); th0 = torch.zeros(Q, dtype=torch.int32, device="cuda")
    sv0 = torch.zeros(Q, 1, device="cuda"); si0 = torch.zeros(Q, 1, dtype=torch.int32, device="cuda")
    cv, _, _ = dn.mqa_logits_dsa_marsco(q, kf, ksc, w, ks0, ke, o0, i1, th0, sv0, si0, NB, S, K, 0, -1)
    return cv


def litetopk_e2e(sp, dn, q, kf, ksc, w, S):
    Q = q.shape[0]; Smax = kf.shape[0]
    ke_full = torch.full((Q,), Smax, dtype=torch.int32, device="cuda")
    head = min(SAMPLE, Smax); pos = torch.arange(0, head, device="cuda", dtype=torch.long)
    ksamp = kf[pos].contiguous(); kss = ksc[pos].contiguous(); Ss = ksamp.shape[0]
    ks_scan = torch.full((Q,), head, dtype=torch.int32, device="cuda")
    col_full = pos.to(torch.int32).view(1, -1).expand(Q, -1).contiguous()
    cnt_full = torch.full((Q,), head, dtype=torch.int32, device="cuda")
    def O():
        sl = dense_scores(dn, q, ksamp, kss, w, Ss)  # sample scores via the dense kernel
        xs = -sl; o2 = xs.min(1).values.contiguous(); hi = xs.max(1).values
        inv2 = ((NB - 1) / (hi - o2).clamp_min(1e-20)).contiguous()
        sv, si = sp.compact_topk_min_idx_marsco(xs, col_full, cnt_full, K); sv = sv.contiguous(); si = si.contiguous()
        th2 = ((sv.max(1).values - o2) * inv2).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
        cv2, ci2, cc2 = sp.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_full, o2, inv2, th2, sv, si, NB, 1, K, REFRESH, -1)
        return sp.compact_topk_min_thr_marsco(cv2, ci2, cc2, o2, inv2, th2, NB, K)
    return O


def tile_q(q, w, Q0, chunk):
    reps = (chunk + Q0 - 1) // Q0
    return q.repeat(reps, 1, 1)[:chunk].contiguous(), w.repeat(reps, 1)[:chunk].contiguous()


def main():
    dn = build("dsa_dense_cmp", ["-DDENSE_WRITE"], "/tmp/bd_dense_cmp")
    spA = build("dsa_sparse128", [], "/tmp/bd_sp128")
    spB = build("dsa_sparse256", ["-DDSA_BLOCK_KV=256", "-DDSA_MATH_THREADS=256"], "/tmp/bd_sp256")
    print(f"{'seq':>5} {'chunk':>6} {'DSA(ms)':>9} {'A_128/1(ms)':>12} {'A_spdup':>8} {'B_256/2(ms)':>12} {'B_spdup':>8} {'B/A':>6} {'recall':>8}")
    for tag, chunk in CELLS:
        T = {}
        with safe_open(cfg.cache_path(tag), "pt", device="cuda") as f:
            for k in f.keys(): T[k] = f.get_tensor(k)
        q0 = T["q_index"].contiguous(); qs = T["q_index_scale"].squeeze(-1)
        kf = T["idx_k_cache"][0].contiguous(); ksc = T["idx_k_scale"][0, :, 0].contiguous()
        gw = T["gate_w"]; Q0, H, D = q0.shape; S = kf.shape[0]
        w0 = (gw * qs * (D ** -0.5)).contiguous().float()
        q, w = tile_q(q0, w0, Q0, chunk)
        cvref = dense_scores(dn, q, kf, ksc, w, S); ref = cvref.topk(K, -1).indices.long(); refs, _ = ref.sort(-1); del cvref
        def recall(idx):
            p = torch.searchsorted(refs, idx).clamp(max=K - 1)
            return 100.0 * (torch.gather(refs, 1, p) == idx).sum().item() / (idx.shape[0] * K)
        def DSA():
            cv = dense_scores(dn, q, kf, ksc, w, S); cv.topk(K, -1); del cv
        tDSA = evt(DSA)
        OA = litetopk_e2e(spA, dn, q, kf, ksc, w, S); OB = litetopk_e2e(spB, dn, q, kf, ksc, w, S)
        _, oiA = OA(); _, oiB = OB(); torch.cuda.synchronize()
        rA = recall(oiA.long()); rB = recall(oiB.long())
        tA = evt(lambda: OA()); tB = evt(lambda: OB())
        print(f"{tag:>5} {chunk:>6} {tDSA:>9.3f} {tA:>12.3f} {tDSA/tA:>7.2f}x {tB:>12.3f} {tDSA/tB:>7.2f}x {tA/tB:>5.2f}x {min(rA,rB):>7.2f}%", flush=True)
        del T, q0, kf, ksc, w0, q, w; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
