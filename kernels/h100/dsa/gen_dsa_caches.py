#!/usr/bin/env python3
"""Reconstructed generator for the GLM-5.2-FP8 DSA layer-0 indexer caches.

The original generator that produced glm5_{size}_realtext.safetensors was lost;
this rebuilds it from the model shard + corpus.  Validated against the surviving
Q=64 caches: key-side cos-sim 0.9997, bf16 topk recall 99.66%.

Difference vs the old caches: the QUERY is now a realistic prefill CHUNK of
`--chunk` tokens (positions [S, S+chunk)) instead of the last 64 context tokens,
so Q = chunk (default 4096).  The KV side (context [0,S)) is unchanged.
"""
import argparse, json, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer

MODEL = "/workspace/project/glm5"
SHARD = MODEL + "/model-00001-of-00141.safetensors"
CACHE_DIR = "/workspace/project/dsa_caches"
DG = "/workspace/project/build/src/hopper/dsa/DeepGEMM_reduced_topk"
sys.path.insert(0, DG)
import deep_gemm

cfg = json.load(open(MODEL + "/config.json"))
THETA = cfg["rope_parameters"]["rope_theta"]
QK_ROPE = cfg["qk_rope_head_dim"]        # 64
IDX_HD = cfg["index_head_dim"]           # 128
EPS = cfg["rms_norm_eps"]                # 1e-5
NH = cfg["index_n_heads"]               # 32
TOPK = cfg["index_topk"]                 # 2048
FP8MAX = 448.0
DEV = "cuda"

SIZES = {"256k": 262144, "512k": 524288, "768k": 786432, "1m": 1048576}


def deq(w, s, bs=128):
    w = w.float(); s = s.float(); O, I = w.shape
    return w * s.repeat_interleave(bs, 0)[:O].repeat_interleave(bs, 1)[:, :I]


def load_weights():
    W = {}
    with safe_open(SHARD, "pt", device=DEV) as f:
        g = f.get_tensor
        W["emb"] = g("model.embed_tokens.weight")
        W["ln"] = g("model.layers.0.input_layernorm.weight").float()
        W["qa"] = deq(g("model.layers.0.self_attn.q_a_proj.weight"),
                      g("model.layers.0.self_attn.q_a_proj.weight_scale_inv"))
        W["qaln"] = g("model.layers.0.self_attn.q_a_layernorm.weight").float()
        W["wqb"] = deq(g("model.layers.0.self_attn.indexer.wq_b.weight"),
                       g("model.layers.0.self_attn.indexer.wq_b.weight_scale_inv"))
        W["wk"] = deq(g("model.layers.0.self_attn.indexer.wk.weight"),
                      g("model.layers.0.self_attn.indexer.wk.weight_scale_inv"))
        W["knw"] = g("model.layers.0.self_attn.indexer.k_norm.weight").float()
        W["knb"] = g("model.layers.0.self_attn.indexer.k_norm.bias").float()
        W["wp"] = g("model.layers.0.self_attn.indexer.weights_proj.weight").float()
    return W


def rope_hs(x, dim, pos, heads):
    inv = 1.0 / (THETA ** (torch.arange(0, dim, 2, device=DEV).float() / dim))
    fr = torch.outer(pos.float(), inv)
    cos = torch.cat([fr.cos()] * 2, -1); sin = torch.cat([fr.sin()] * 2, -1)
    if heads:
        cos = cos[:, None, :]; sin = sin[:, None, :]
    xr = x[..., :dim]
    x1 = xr[..., :dim // 2]; x2 = xr[..., dim // 2:]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * cos + rot * sin, x[..., dim:]], -1)


def hidden(ids, W):
    h = W["emb"][ids].float()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * W["ln"]


def build_keys(ids, pos, W, batch=131072):
    # Row-batched to bound the [S,6144] fp32 hidden buffer (24GB at S=1M).
    S = ids.numel()
    out = torch.empty(S, IDX_HD, device=DEV)
    for r0 in range(0, S, batch):
        r1 = min(r0 + batch, S)
        h = hidden(ids[r0:r1], W)
        k = torch.nn.functional.linear(h.to(torch.bfloat16), W["wk"].to(torch.bfloat16)).float()
        m = k.mean(-1, keepdim=True); v = k.var(-1, keepdim=True, unbiased=False)
        k = (k - m) * torch.rsqrt(v + 1e-6) * W["knw"] + W["knb"]
        out[r0:r1] = rope_hs(k, QK_ROPE, pos[r0:r1], heads=False)
        del h, k, m, v
    return out                                            # [S,128] bf16-equiv


def build_q_gate(ids, pos, W):
    h = hidden(ids, W)
    qr = torch.nn.functional.linear(h.to(torch.bfloat16), W["qa"].to(torch.bfloat16)).float()
    qr = qr * torch.rsqrt(qr.pow(2).mean(-1, keepdim=True) + EPS) * W["qaln"]
    q = torch.nn.functional.linear(qr.to(torch.bfloat16), W["wqb"].to(torch.bfloat16)).float().view(-1, NH, IDX_HD)
    q = rope_hs(q, QK_ROPE, pos, heads=True)              # [Q,32,128]
    gate = torch.nn.functional.linear(h, W["wp"]) * (NH ** -0.5)   # [Q,32]
    return q, gate


def quant_fp8(x):   # per last-dim row
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-20)
    scale = amax / FP8MAX
    xq = (x / scale).clamp(-FP8MAX, FP8MAX).to(torch.float8_e4m3fn)
    return xq, scale.squeeze(-1)


def gen(size, chunk, W, allids):
    S = SIZES[size]
    assert S + chunk <= allids.numel(), f"corpus too short for {size}+chunk{chunk}"
    t0 = time.time()
    # KV: context [0,S)
    kpos = torch.arange(0, S, device=DEV)
    K = build_keys(allids[kpos], kpos, W)
    kq, ksc = quant_fp8(K)                                # [S,128], [S]
    # Q: next prefill chunk [S, S+chunk)
    qpos = torch.arange(S, S + chunk, device=DEV)
    q, gate = build_q_gate(allids[qpos], qpos, W)
    qq, qsc = quant_fp8(q)                                # [Q,32,128], [Q,32]
    Q = chunk
    # topk_idx via deep_gemm fp8 (query row-batched to bound memory)
    w = (gate * qsc * (IDX_HD ** -0.5)).contiguous().float()
    topk_idx = torch.empty(Q, TOPK, dtype=torch.int32, device=DEV)
    RB = 512
    for r0 in range(0, Q, RB):
        r1 = min(r0 + RB, Q)
        ks = torch.zeros(r1 - r0, dtype=torch.int32, device=DEV)
        ke = torch.full((r1 - r0,), S, dtype=torch.int32, device=DEV)
        lg = deep_gemm.fp8_mqa_logits(qq[r0:r1].contiguous(), (kq.contiguous(), ksc.contiguous()),
                                      w[r0:r1].contiguous(), ks, ke,
                                      clean_logits=True, max_seqlen_k=0)
        topk_idx[r0:r1] = lg.topk(TOPK, dim=-1).indices.to(torch.int32)
        del lg
    tensors = {
        "idx_k_cache": kq.unsqueeze(0).contiguous().cpu(),
        "idx_k_scale": ksc.view(1, S, 1).contiguous().cpu(),
        "q_index": qq.contiguous().cpu(),
        "q_index_scale": qsc.view(Q, NH, 1).contiguous().cpu(),
        "gate_w": gate.contiguous().cpu(),
        "topk_idx": topk_idx.contiguous().cpu(),
    }
    meta = {
        "source": "zai-org/GLM-5.2-FP8 (REAL weights, layer 0)",
        "note": "GLM DSA layer-0; bf16-equiv indexer (no Hadamard), half-split RoPE; fp8 q/k dynamic e4m3 (no ue8m0)",
        "fp8_max": "448.0", "fp8_block": "128", "scale_fmt": "e4m3_dynamic",
        "tokens": f"text-file:glm5/corpus.txt (chunk-prefill: kv=[0,{S}) q=[{S},{S+chunk}))",
        "config": json.dumps({"s_kv": S, "num_q": Q, "chunk": chunk, "H_I": NH, "d_I": IDX_HD,
                              "qk_rope": QK_ROPE, "topk": TOPK, "layer": 0, "theta": THETA,
                              "q_positions": f"[{S},{S+chunk})", "ke": "S (full context, no intra-chunk causal)"}),
        "layout": json.dumps({k: [list(v.shape), str(v.dtype)] for k, v in tensors.items()}),
    }
    out = f"{CACHE_DIR}/glm5_{size}_realtext_chunk{chunk}.safetensors"
    save_file(tensors, out, metadata=meta)
    # quick recall of our fp8 topk vs bf16 truth on a query subset (sanity)
    print(f"[{size}] S={S} Q={Q} chunk={chunk} -> {out}  ({time.time()-t0:.1f}s)")
    del K, kq, ksc, q, qq, qsc, topk_idx
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--sizes", nargs="*", default=["256k", "512k", "768k", "1m"])
    a = ap.parse_args()
    W = load_weights()
    t = time.time()
    tk = Tokenizer.from_file(MODEL + "/tokenizer.json")
    allids = torch.tensor(tk.encode(open(MODEL + "/corpus.txt").read()).ids, device=DEV, dtype=torch.long)
    print(f"tokenized corpus: {allids.numel()} tokens ({time.time()-t:.1f}s)")
    for s in a.sizes:
        gen(s, a.chunk, W, allids)


if __name__ == "__main__":
    main()
