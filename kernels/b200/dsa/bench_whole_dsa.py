#!/usr/bin/env python3
"""Standalone WHOLE-DSA latency bench on off-shelf caches (no 78L model load).

Whole DSA = the fused indexer top-k + the sparse MLA attention that follows.
Three methods on the SAME off-shelf chunk8192 cache (real GLM-5.2 layer-0
indexer inputs + real topk_idx + real absorbed MLA q/kv):

  vLLM     : dense fp8 logits (deep_gemm) + top_k_per_row_prefill  ->  stock trtllm sparse MLA
  LiteTopK : this repo's fused indexer (dsa_litetopk.cu, backported tuning) -> stock trtllm sparse MLA
  LiteTopK+LiteDSA : same indexer -> litedsa grouped masked MLA (union over
             G=16 adjacent queries' top-k, per-query membership mask)

Everything here is self-contained to this repo: the indexer goes through
`litetopk_indexer.try_chunk` (the same call `bench_q8192.py` times), and
LiteDSA goes through `flashinfer_port/litedsa.so` directly (tvm_ffi), not
through any external vLLM fork -- no dependency on vllm-lt-venv.

Attention is fed the cache's REAL topk_idx for all three (isolates
stock-vs-grouped on identical input). LiteDSA's union is rebuilt every call,
so the per-call number includes the union build, matching a fresh chunk in
deployment.

Run in the glm5-prefill container, vLLM standard env (site-packages deep_gemm,
flashinfer, tvm_ffi all present there already).
"""
import os
import sys

import torch

sys.path.insert(0, os.environ.get("LITETOPK_MODULE_DIR", "/opt/litetopk_repro/glm5_prefill/litetopk_vllm"))
os.environ.setdefault("VLLM_LITETOPK_MIN_S", "0")
os.environ.setdefault("VLLM_LITETOPK_XFLAGS", "DSA_BUCKET_GATE4=1")
os.environ.setdefault("VLLM_LITETOPK_PREP_TILE", "0")
os.environ.setdefault("VLLM_LITETOPK_HOTONLY", "1")
import litetopk_indexer  # noqa: E402

import json
import math

import tvm_ffi  # noqa: E402
from safetensors import safe_open  # noqa: E402
import deep_gemm  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.v1.attention.backends.mla.sparse_utils import (  # noqa: E402
    triton_convert_req_index_to_global_index,
)
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla  # noqa: E402

dev = "cuda"
K, Q = 2048, 8192      # topk, prefill chunk
G = 16                 # litedsa pack (TP8: n_rank_heads=8, 8*16=128)
BLOCK_SIZE = 64         # MLA paged block size (FlashInferMLASparse default)
TAGS = os.environ.get("TAGS", "768k 1m").split()
DATA = "/data/dsa_caches/glm5_{}_realtext_chunk8192.safetensors"
QK_NOPE, KV_LORA, QK_ROPE = 192, 512, 64

_here = os.path.dirname(os.path.abspath(__file__))
_litedsa_so = os.environ.get("LITEDSA_SO", os.path.join(_here, "flashinfer_port", "litedsa.so"))
m = tvm_ffi.load_module(_litedsa_so)

WORKSPACE = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)


def evt(fn, w, n):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def recall(a, b):
    a = a.long().sort(1).values
    b = b.long().sort(1).values
    p = torch.searchsorted(b, a).clamp(max=b.shape[1] - 1)
    return 100.0 * (torch.gather(b, 1, p) == a).float().mean().item()


vllm_ms, litetopk_ms, stock_ms, litedsa_ms = {}, {}, {}, {}
idx_recall, attn_maxdiff, slen = {}, {}, {}

print(f"{'S':>6} {'idx(vLLM/LT)':>15} {'attn(stock/LD)':>16} "
      f"{'WHOLE(vLLM/LT/LD)':>20} {'LD-vs-vLLM':>10} {'idx%':>6} {'attn_maxdiff':>12}", flush=True)

for tag in TAGS:
    T = {}
    with safe_open(DATA.format(tag), "pt", device="cuda") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            T[k] = f.get_tensor(k)
    mla = json.loads(meta["mla"])
    n_rank_heads = mla["n_rank_heads"]
    bmm1, bmm2 = mla["bmm1_scale"], mla["bmm2_scale"]

    q = T["q_index"][:Q].contiguous()
    qs = T["q_index_scale"].squeeze(-1)[:Q]
    kf = T["idx_k_cache"][0].contiguous()
    ksc = T["idx_k_scale"][0, :, 0].contiguous()
    w = (T["gate_w"][:Q] * qs * (q.shape[2] ** -0.5)).contiguous().float()
    S = kf.shape[0]
    slen[tag] = S
    # Full window (every row sees all S), not bench_q8192.py's row-causal window:
    # the cache's stored topk_idx was generated this way, and attention below is
    # fed that same topk_idx, so indexer/attention/reference must all agree on it.
    ks = torch.zeros(Q, dtype=torch.int32, device=dev)
    ke = torch.full((Q,), S, dtype=torch.int32, device=dev)
    topk_ref = T["topk_idx"][:Q].contiguous().int()  # real reference top-k, for recall

    # ---------- INDEXER: official dense (chunk=8192, its memory-unconstrained best) ----------
    outo = torch.full((Q, K), -1, dtype=torch.int32, device=dev)

    def official():
        lg = deep_gemm.fp8_fp4_mqa_logits((q, None), (kf, ksc), w, ks, ke, clean_logits=False)
        ops.top_k_per_row_prefill(lg, ks, ke, outo, Q, lg.stride(0), lg.stride(1), K)

    official()
    torch.cuda.synchronize()
    vllm_ms[tag] = round(evt(official, 5, 8), 3)

    # ---------- INDEXER: this repo's fused LiteTopK (backported kernel), hot-started ----------
    hot_key = f"whole_{tag}"
    litetopk_indexer.stash_carry(hot_key, outo, S)
    torch.cuda.synchronize()
    carry = litetopk_indexer._HOT_CARRY[(str(q.device), hot_key)][0]
    outi = torch.full((Q, K), -1, dtype=torch.int32, device=dev)

    def fused_ind():
        assert litetopk_indexer.try_chunk(q, kf, ksc, w, ks, ke, outi, K,
                                          hot_prev=carry, _carry_io=False)

    fused_ind()
    torch.cuda.synchronize()
    litetopk_ms[tag] = round(evt(fused_ind, 6, 8), 3)
    idx_recall[tag] = round(recall(outi, topk_ref), 2)

    # ---------- ATTENTION setup (real absorbed MLA q/kv from the cache) ----------
    mla_q = T["mla_q"].contiguous()                                # [8192, n_rank_heads, 576] fp8
    mla_kv = T["mla_kv"].contiguous()                              # [S, 576] fp8
    nblk = math.ceil(S / BLOCK_SIZE)
    kv_paged = torch.zeros(nblk * BLOCK_SIZE, mla_kv.shape[-1], dtype=mla_kv.dtype, device=dev)
    kv_paged[:S] = mla_kv
    kv_paged = kv_paged.view(nblk, BLOCK_SIZE, -1)                 # [nblk, 64, 576]
    blk_tbl = torch.arange(nblk, device=dev, dtype=torch.int32)[None, :]  # [1, nblk]
    req_id = torch.zeros(Q, dtype=torch.int32, device=dev)

    # ---------- ATTENTION: stock trtllm sparse MLA, fed the real topk_idx ----------
    def stock_attn():
        phys, seq_lens = triton_convert_req_index_to_global_index(
            req_id, blk_tbl, topk_ref, BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=K,
            return_valid_counts=True)
        return trtllm_batch_decode_with_kv_cache_mla(
            query=mla_q.unsqueeze(1),
            kv_cache=kv_paged.unsqueeze(1),
            workspace_buffer=WORKSPACE,
            qk_nope_head_dim=QK_NOPE, kv_lora_rank=KV_LORA, qk_rope_head_dim=QK_ROPE,
            block_tables=phys.unsqueeze(1), seq_lens=seq_lens, max_seq_len=K,
            bmm1_scale=bmm1, bmm2_scale=bmm2, sparse_mla_top_k=K, return_lse=False)

    stock_out = stock_attn()
    torch.cuda.synchronize()
    stock_ms[tag] = round(evt(stock_attn, 5, 8), 3)

    # ---------- ATTENTION: LiteDSA grouped masked MLA (union rebuilt each call) ----------
    ng = Q // G
    cap = G * K  # constructive union bound, 128-aligned by construction (K=2048 -> cap=32768)
    u_phys = torch.empty(ng, cap, dtype=torch.int32, device=dev)
    counts = torch.empty(ng, dtype=torch.int32, device=dev)
    memb_qm = torch.empty(ng, G, cap // 32, dtype=torch.int32, device=dev)
    req_pg = req_id[::G].contiguous()  # [ng], one physical request, all zero

    qp = mla_q.view(ng, 128, mla_q.shape[2]).contiguous()
    kv_flat = kv_paged.view(-1, 1, kv_paged.shape[-1])
    out = torch.empty(ng, 128, 512, dtype=torch.bfloat16, device=dev)
    max_logits = torch.empty(ng, 128, dtype=torch.float32, device=dev)
    lse = torch.empty(ng, 128, dtype=torch.float32, device=dev)

    def litedsa_attn():
        m.union_qm(topk_ref, u_phys, counts, memb_qm, req_pg, blk_tbl, BLOCK_SIZE, S)
        m.masked_mla_fp8(qp, kv_flat, u_phys.view(ng, 1, cap), bmm1, bmm2, counts, memb_qm,
                         n_rank_heads, out, max_logits, lse)
        return out

    litedsa_out = litedsa_attn()
    torch.cuda.synchronize()
    litedsa_ms[tag] = round(evt(litedsa_attn, 5, 8), 3)

    attn_maxdiff[tag] = round((stock_out.view(ng, 128, 512).float() - litedsa_out.float()).abs().max().item(), 4)

    wv = vllm_ms[tag] + stock_ms[tag]
    wt = litetopk_ms[tag] + stock_ms[tag]
    wd = litetopk_ms[tag] + litedsa_ms[tag]
    print(f"{tag:>6} {vllm_ms[tag]:>7.2f}/{litetopk_ms[tag]:<6.2f} "
          f"{stock_ms[tag]:>7.2f}/{litedsa_ms[tag]:<7.2f} "
          f"{wv:>6.2f}/{wt:<6.2f}/{wd:<6.2f} {wv / wd:>9.3f}x "
          f"{idx_recall[tag]:>5.2f}% {attn_maxdiff[tag]:>11.4f}", flush=True)
    del T, q, kf, ksc, w, outo, outi, mla_q, mla_kv, kv_paged, u_phys, counts, memb_qm
    torch.cuda.empty_cache()

print("\n============ WHOLE-DSA ms (indexer + attention) ============")
print(f"{'S':>6} {'vLLM@8192':>10} {'LiteTopK':>9} {'LiteTopK+LiteDSA':>17} {'speedup':>8}")
for tag in TAGS:
    wv = vllm_ms[tag] + stock_ms[tag]
    wt = litetopk_ms[tag] + stock_ms[tag]
    wd = litetopk_ms[tag] + litedsa_ms[tag]
    print(f"{tag:>6} {wv:>10.2f} {wt:>9.2f} {wd:>17.2f} {wv / wd:>7.3f}x")
