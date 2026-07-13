#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-(chunk, seq) config sweep for the B200 (SM100) DSA fused E2E path.

Sample size, bucket count (NB) and refresh cadence trade off sample-stage cost
vs. main-scan cost vs. candidate density. This grid-searches them and, per
(chunk, seq), reports the FASTEST end-to-end config whose recall clears a
threshold, alongside the config-independent B_e2e (KV-split dense + torch.topk).

Env:
  CHUNKS   default "1024 2048 4096 8192"
  SAMPLES  default "8192 16384 32768 65536 131072"
  BUCKETS  default "128 256 512"
  REFRESHS default "16 64 128"
  K        default 2048
  RECMIN   default 99.0
  WARMUP   default 3      ITERS default 8
  plus the DSA_* path overrides from b200_config.
Usage: python3 sweep_config.py [256k 512k 768k 1m]
"""
import os, sys, itertools, torch
from torch.utils.cpp_extension import load

import b200_config as cfg
cfg.prepare_env()
import deep_gemm
from safetensors import safe_open

K = int(os.environ.get("K", "2048"))
RECMIN = float(os.environ.get("RECMIN", "99.0"))
WARMUP = int(os.environ.get("WARMUP", "3")); ITERS = int(os.environ.get("ITERS", "8"))
SAMPLES = [int(x) for x in os.environ.get("SAMPLES", "8192 16384 32768 65536 131072").split()]
BUCKETS = [int(x) for x in os.environ.get("BUCKETS", "128 256 512").split()]
REFRESHS = [int(x) for x in os.environ.get("REFRESHS", "16 64 128").split()]


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


def rowbatch(Q, S):
    raw = max(256, (1 << 30) // (S * 4))
    nb = (Q + raw - 1) // raw
    return min(Q, ((Q + nb - 1) // nb + 3) & ~3)


def run(tag, chunk, sp, dn):
    os.environ["CHUNK"] = str(chunk)
    T = {}
    with safe_open(cfg.cache_path(tag), "pt", device="cuda") as f:
        for k in f.keys(): T[k] = f.get_tensor(k)
    q = T["q_index"].contiguous(); qs = T["q_index_scale"].squeeze(-1)
    kf = T["idx_k_cache"][0].contiguous(); ksc = T["idx_k_scale"][0, :, 0].contiguous()
    gw = T["gate_w"]; Q, H, D = q.shape; S = kf.shape[0]
    w = (gw * qs * (D ** -0.5)).contiguous().float()
    ks0 = torch.zeros(Q, dtype=torch.int32, device="cuda")
    ke_full = torch.full((Q,), S, dtype=torch.int32, device="cuda")
    RB = rowbatch(Q, S)

    ref = torch.empty(Q, K, dtype=torch.long, device="cuda")
    for r0 in range(0, Q, RB):
        r1 = min(r0 + RB, Q)
        lg = deep_gemm.fp8_mqa_logits(q[r0:r1].contiguous(), (kf, ksc), w[r0:r1].contiguous(),
                                      ks0[r0:r1], ke_full[r0:r1], clean_logits=True, max_seqlen_k=0)
        ref[r0:r1] = lg.topk(K, dim=-1).indices.long(); del lg
    refs, _ = ref.sort(dim=-1)

    def recall(idx):
        p = torch.searchsorted(refs, idx).clamp(max=K - 1)
        return 100.0 * (torch.gather(refs, 1, p) == idx).sum().item() / (Q * K)

    o0 = torch.zeros(Q, device="cuda"); i1 = torch.ones(Q, device="cuda")
    th0 = torch.zeros(Q, dtype=torch.int32, device="cuda")
    sv0 = torch.zeros(Q, 1, device="cuda"); si0 = torch.zeros(Q, 1, dtype=torch.int32, device="cuda")
    def B_e2e():
        for r0 in range(0, Q, RB):
            r1 = min(r0 + RB, Q)
            cv, _, _ = dn.mqa_logits_dsa_marsco(q[r0:r1].contiguous(), kf, ksc, w[r0:r1].contiguous(),
                                                ks0[r0:r1], ke_full[r0:r1], o0[r0:r1], i1[r0:r1], th0[r0:r1],
                                                sv0[r0:r1], si0[r0:r1], 256, S, K, 0, -1)
            cv.topk(K, dim=-1); del cv
    tB = evt(B_e2e)

    best = None
    for sample, NB, refresh in itertools.product(SAMPLES, BUCKETS, REFRESHS):
        head = min(sample, S)
        pos = torch.arange(0, head, device="cuda", dtype=torch.long)
        ksamp = kf[pos].contiguous(); kss = ksc[pos].contiguous(); Ss = ksamp.shape[0]
        ks_scan = torch.full((Q,), head, dtype=torch.int32, device="cuda")
        col_full = pos.to(torch.int32).view(1, -1).expand(RB, -1).contiguous()
        cnt_full = torch.full((RB,), head, dtype=torch.int32, device="cuda")

        def O_prep_all():
            prep = {}
            for r0 in range(0, Q, RB):
                r1 = min(r0 + RB, Q)
                ke_s = torch.full((r1 - r0,), Ss, dtype=torch.int32, device="cuda")
                sl = deep_gemm.fp8_mqa_logits(q[r0:r1].contiguous(), (ksamp, kss), w[r0:r1].contiguous(),
                                              ks0[r0:r1], ke_s, clean_logits=True, max_seqlen_k=0).contiguous()
                xs = -sl; o2 = xs.min(dim=1).values.contiguous(); hi = xs.max(dim=1).values
                inv2 = ((NB - 1) / (hi - o2).clamp_min(1e-20)).contiguous()
                sv, si = sp.compact_topk_min_idx_marsco(xs, col_full[:r1 - r0], cnt_full[:r1 - r0], K)
                sv = sv.contiguous(); si = si.contiguous()
                th2 = ((sv.max(dim=1).values - o2) * inv2).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
                prep[r0] = (o2, inv2, th2, sv, si)
            return prep

        def O_e2e():
            prep = O_prep_all()
            oi_all = torch.empty(Q, K, dtype=torch.int32, device="cuda")
            for r0 in range(0, Q, RB):
                r1 = min(r0 + RB, Q); o2, inv2, th2, sv, si = prep[r0]
                cv, ci, cc = sp.mqa_logits_dsa_marsco(q[r0:r1].contiguous(), kf, ksc, w[r0:r1].contiguous(),
                                                      ks_scan[r0:r1], ke_full[r0:r1], o2, inv2, th2, sv, si,
                                                      NB, 1, K, refresh, -1)
                _, oi = sp.compact_topk_min_thr_marsco(cv, ci, cc, o2, inv2, th2, NB, K); oi_all[r0:r1] = oi; del cv, ci, cc
            return oi_all

        try:
            oi = O_e2e(); torch.cuda.synchronize(); rec = recall(oi.long())
        except Exception as ex:
            print(f"    [s{sample} nb{NB} r{refresh}] ERR {str(ex)[:50]}", flush=True); continue
        tO = evt(O_e2e) if rec >= RECMIN else None
        tag_ok = "OK " if rec >= RECMIN else "low"
        print(f"    [s{sample:>6} nb{NB:>3} r{refresh:>3}] recall={rec:6.2f}% "
              f"tO={('%.3f' % tO) if tO else '   -  '} ms  {tag_ok}", flush=True)
        if tO is not None and (best is None or tO < best["tO"]):
            best = dict(sample=sample, NB=NB, refresh=refresh, tO=tO, rec=rec)
        del oi
        torch.cuda.empty_cache()

    del T, q, kf, ksc, ref, refs; torch.cuda.empty_cache()
    return dict(Q=Q, S=S, tB=tB, best=best)


if __name__ == "__main__":
    sp, dn = build()
    chunks = [int(x) for x in os.environ.get("CHUNKS", "1024 2048 4096 8192").split()]
    tags = sys.argv[1:] or ["256k", "512k", "768k", "1m"]
    rows = []
    for chunk in chunks:
        for tag in tags:
            print(f"[chunk{chunk} {tag}] sweeping...", flush=True)
            r = run(tag, chunk, sp, dn); rows.append((chunk, tag, r))
            b = r["best"]
            if b:
                print(f"[chunk{chunk} {tag}] BEST sample={b['sample']} NB={b['NB']} refresh={b['refresh']} "
                      f"O={b['tO']:.3f} B={r['tB']:.3f} O/B={r['tB']/b['tO']:.2f}x recall={b['rec']:.2f}%", flush=True)
            else:
                print(f"[chunk{chunk} {tag}] no config reached recall>={RECMIN}%", flush=True)

    print("\n============== DSA E2E BEST-CONFIG SWEEP (B200 SM100, K=%d, recall>=%.1f%%) ==============" % (K, RECMIN))
    print(f"{'chunk':>6} {'seq':>5} {'Q':>6} {'sample':>7} {'NB':>4} {'refr':>5} {'B_e2e':>9} {'O_e2e':>9} {'O/B':>6} {'recall':>7}")
    for chunk, tag, r in rows:
        b = r["best"]
        if b:
            print(f"{chunk:>6} {tag:>5} {r['Q']:>6} {b['sample']:>7} {b['NB']:>4} {b['refresh']:>5} "
                  f"{r['tB']:>9.3f} {b['tO']:>9.3f} {r['tB']/b['tO']:>5.2f}x {b['rec']:>6.2f}%")
        else:
            print(f"{chunk:>6} {tag:>5} {r['Q']:>6} {'-':>7} {'-':>4} {'-':>5} {r['tB']:>9.3f} {'-':>9} {'-':>6} {'-':>7}")
