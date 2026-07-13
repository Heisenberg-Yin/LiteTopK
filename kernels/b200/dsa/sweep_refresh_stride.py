#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep the spare-warp refresh stride (DSA_REFRESH_STRIDE = refresh every N CTA
KV-blocks) on real GLM-5 caches. Builds one sparse variant per stride value and
times LiteTopK end-to-end vs the shared dense baseline. recall must stay 100%.

Run in the `simtopk` container:
  PYTHONPATH=/opt/venvs/deepgemm/lib/python3.12/site-packages \
  DSA_CACHE_DIR=/data/dsa CHUNK=0 \
  /usr/bin/python3.12 sweep_refresh_stride.py
"""
import os, sys, torch
sys.path.insert(0, "/opt/simtopk_src/b200/dsa")
import b200_config as cfg
cfg.prepare_env()
from torch.utils.cpp_extension import load
from safetensors import safe_open

K = 2048; NB = 256; SAMPLE = 65536
WARMUP = int(os.environ.get("WARMUP", "5")); ITERS = int(os.environ.get("ITERS", "20"))
STRIDES = [int(x) for x in os.environ.get("STRIDES", "2 4 8 16 32 64").split()]
CELLS = [(t, int(c)) for t in os.environ.get("TAGS", "768k 1m").split()
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


def dense(dn, q, kf, ksc, w, S):
    Q = q.shape[0]; ks0 = torch.zeros(Q, dtype=torch.int32, device="cuda"); ke = torch.full((Q,), kf.shape[0], dtype=torch.int32, device="cuda")
    o0 = torch.zeros(Q, device="cuda"); i1 = torch.ones(Q, device="cuda"); th0 = torch.zeros(Q, dtype=torch.int32, device="cuda")
    sv0 = torch.zeros(Q, 1, device="cuda"); si0 = torch.zeros(Q, 1, dtype=torch.int32, device="cuda")
    cv, _, _ = dn.mqa_logits_dsa_marsco(q, kf, ksc, w, ks0, ke, o0, i1, th0, sv0, si0, NB, S, K, 0, -1)
    return cv


def tile_q(q, w, Q0, chunk):
    reps = (chunk + Q0 - 1) // Q0
    return q.repeat(reps, 1, 1)[:chunk].contiguous(), w.repeat(reps, 1)[:chunk].contiguous()


def make_e2e(sp, dn, q, kf, ksc, w):
    Q = q.shape[0]; Smax = kf.shape[0]
    ke_full = torch.full((Q,), Smax, dtype=torch.int32, device="cuda")
    head = min(SAMPLE, Smax); pos = torch.arange(0, head, device="cuda", dtype=torch.long)
    ksamp = kf[pos].contiguous(); kss = ksc[pos].contiguous()
    ks_scan = torch.full((Q,), head, dtype=torch.int32, device="cuda")
    col = pos.to(torch.int32).view(1, -1).expand(Q, -1).contiguous(); cnt = torch.full((Q,), head, dtype=torch.int32, device="cuda")
    def O():
        sl = dense(dn, q, ksamp, kss, w, head); xs = -sl; o2 = xs.min(1).values.contiguous()
        inv2 = ((NB - 1) / (xs.max(1).values - o2).clamp_min(1e-9)).contiguous()
        sv, si = sp.compact_topk_min_idx_marsco(xs, col, cnt, K); sv = sv.contiguous(); si = si.contiguous()
        th2 = ((sv.max(1).values - o2) * inv2).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
        cv2, ci2, cc2 = sp.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_full, o2, inv2, th2, sv, si, NB, 1, K, 64, -1)
        return sp.compact_topk_min_thr_marsco(cv2, ci2, cc2, o2, inv2, th2, NB, K)
    return O


def main():
    dn = build("dsa_dn_rs", ["-DDENSE_WRITE"], "/tmp/bd_dn_rs")
    variants = {s: build(f"dsa_sp_rs{s}", [f"-DDSA_REFRESH_STRIDE={s}"], f"/tmp/bd_sp_rs{s}") for s in STRIDES}
    print(f"{'seq':>5} {'chunk':>6} {'DSA(ms)':>9} | " + " ".join(f"str{s:<5}" for s in STRIDES) + "   (LiteTopK ms / speedup)")
    for tag, chunk in CELLS:
        T = {}
        with safe_open(cfg.cache_path(tag), "pt", device="cuda") as f:
            for k in f.keys(): T[k] = f.get_tensor(k)
        q0 = T["q_index"].contiguous(); qs = T["q_index_scale"].squeeze(-1)
        kf = T["idx_k_cache"][0].contiguous(); ksc = T["idx_k_scale"][0, :, 0].contiguous()
        gw = T["gate_w"]; Q0, H, D = q0.shape; S = kf.shape[0]
        w0 = (gw * qs * (D ** -0.5)).contiguous().float()
        q, w = tile_q(q0, w0, Q0, chunk)
        cvref = dense(dn, q, kf, ksc, w, S); ref = cvref.topk(K, -1).indices.long(); refs, _ = ref.sort(-1); del cvref
        def recall(idx):
            p = torch.searchsorted(refs, idx).clamp(max=K - 1)
            return 100.0 * (torch.gather(refs, 1, p) == idx).sum().item() / (idx.shape[0] * K)
        tDSA = evt(lambda: dense(dn, q, kf, ksc, w, S).topk(K, -1))
        cells = []
        for s in STRIDES:
            O = make_e2e(variants[s], dn, q, kf, ksc, w)
            _, oi = O(); torch.cuda.synchronize(); rec = recall(oi.long())
            t = evt(lambda: O())
            cells.append((t, tDSA / t, rec))
        line = f"{tag:>5} {chunk:>6} {tDSA:>9.2f} | "
        line += " ".join(f"{t:6.2f}/{sp:.2f}x" for (t, sp, r) in cells)
        line += "  rec=" + "/".join(f"{r:.1f}" for (_, _, r) in cells)
        print(line, flush=True)
        del T, q0, kf, ksc, w0, q, w; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
