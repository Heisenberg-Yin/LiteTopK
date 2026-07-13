#!/usr/bin/env python3
"""Build N-layer truncated models from the OFFICIAL FP8 checkpoint.

Usage (container python, needs safetensors):
    python build_fp8_truncated.py 16 /models/glm5-prefill-model-fp8
    python build_fp8_truncated.py 1  /models/glm5-prefill-model-fp8-1l

Steps:
  1. shard filtering: pure shards symlinked, mixed shards rewritten;
  2. shared-indexer patch: layers whose indexer_type == 'shared' get the FULL
     layer's indexer tensor group copied under their own names (vLLM's strict
     loader wants every param). Copies by name prefix, so fp8 weight +
     weight_scale_inv pairs come along automatically;
  3. config: num_hidden_layers=N, every per-layer list truncated to N;
  4. tokenizer/aux json files copied from /models/glm5.
"""
import json
import os
import re
import shutil
import sys

from safetensors.torch import load_file, save_file

SRC = "/models/glm5-fp8-official"
AUX = "/models/glm5"  # tokenizer etc.

LAYERS = int(sys.argv[1])
DST = sys.argv[2]
os.makedirs(DST, exist_ok=True)

cfg = json.load(open(os.path.join(SRC, "config.json")))
indexer_types = cfg.get("indexer_types", ["full"] * LAYERS)[:LAYERS]
# map each shared layer to its governing full layer (nearest full above it)
share_map = {}
last_full = None
for i, t in enumerate(indexer_types):
    if t == "full":
        last_full = i
    elif t == "shared":
        assert last_full is not None, "shared layer before any full layer"
        share_map[i] = last_full
print(f"layers={LAYERS} shared->full map: {share_map}")


def needed(name):
    m = re.match(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) < LAYERS if m else True


idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wm = idx["weight_map"]
by_file = {}
for name, f in wm.items():
    by_file.setdefault(f, []).append(name)

new_map = {}
for f, names in sorted(by_file.items()):
    keep = [n for n in names if needed(n)]
    dst_f = os.path.join(DST, f)
    if os.path.islink(dst_f) or os.path.exists(dst_f):
        os.remove(dst_f)
    if not keep:
        continue
    if len(keep) == len(names):
        os.symlink(os.path.join(SRC, f), dst_f)
    else:
        print(f"[rewrite] {f}: {len(keep)}/{len(names)} tensors", flush=True)
        t = load_file(os.path.join(SRC, f))
        save_file({n: t[n] for n in keep}, dst_f, metadata={"format": "pt"})
    for n in keep:
        new_map[n] = f

# shared-indexer patch: whole tensor group by prefix (fp8 scales included)
if share_map:
    by_src_file = {}
    for shared, full in share_map.items():
        pref = f"model.layers.{full}.self_attn.indexer."
        for n, f in wm.items():
            if n.startswith(pref):
                by_src_file.setdefault(f, []).append(
                    (n, n.replace(f"model.layers.{full}.", f"model.layers.{shared}.")))
    patch = {}
    for f, pairs in by_src_file.items():
        t = load_file(os.path.join(SRC, f))
        for src_n, dst_n in pairs:
            patch[dst_n] = t[src_n].clone()  # distinct storage: safetensors
                                             # rejects shared-memory tensors
    pf = "model-shared-indexer-patch.safetensors"
    save_file(patch, os.path.join(DST, pf), metadata={"format": "pt"})
    for n in patch:
        new_map[n] = pf
    print(f"[patch] {len(patch)} shared-indexer tensors -> {pf}")

json.dump({"metadata": {"total_size": 0}, "weight_map": new_map},
          open(os.path.join(DST, "model.safetensors.index.json"), "w"))

cfg["num_hidden_layers"] = LAYERS
for k, v in list(cfg.items()):
    if isinstance(v, list) and len(v) == 78:
        cfg[k] = v[:LAYERS]
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)

for f in os.listdir(AUX):
    if f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
             "chat_template.jinja"):
        shutil.copy(os.path.join(AUX, f), os.path.join(DST, f))
print(f"done: {len(new_map)} tensors mapped -> {DST}")
