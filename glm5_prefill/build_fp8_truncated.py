#!/usr/bin/env python3
"""Build an N-layer model from a sharded GLM FP8 checkpoint.

Shards containing only retained tensors are symlinked. Mixed shards are
rewritten, shared-indexer tensors are copied to the names required by vLLM,
and per-layer configuration lists are truncated. ``MODEL_SRC`` selects the
source checkpoint and ``MODEL_AUX`` selects tokenizer/configuration assets.

Usage:
    MODEL_SRC=/models/glm5-fp8 MODEL_AUX=/models/glm5 \
      python build_fp8_truncated.py 16 /models/glm5-fp8-16l
"""
import json
import os
import re
import shutil
import sys

from safetensors.torch import load_file, save_file

if len(sys.argv) != 3:
    raise SystemExit(
        f"usage: {os.path.basename(sys.argv[0])} NUM_LAYERS DESTINATION"
    )

SRC = os.path.abspath(
    os.environ.get("MODEL_SRC", "/models/glm5-fp8-official")
)
AUX = os.path.abspath(os.environ.get("MODEL_AUX", "/models/glm5"))
LAYERS = int(sys.argv[1])
DST = os.path.abspath(sys.argv[2])

if LAYERS < 1:
    raise ValueError("NUM_LAYERS must be at least 1")
if DST == SRC:
    raise ValueError("DESTINATION must differ from MODEL_SRC")
if not os.path.isfile(os.path.join(SRC, "model.safetensors.index.json")):
    raise FileNotFoundError(f"source checkpoint index not found under {SRC}")
if not os.path.isfile(os.path.join(SRC, "config.json")):
    raise FileNotFoundError(f"source config not found under {SRC}")
if not os.path.isdir(AUX):
    raise FileNotFoundError(f"MODEL_AUX directory does not exist: {AUX}")
os.makedirs(DST, exist_ok=True)

with open(os.path.join(SRC, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
indexer_types = cfg.get("indexer_types", ["full"] * LAYERS)[:LAYERS]
# map each shared layer to its governing full layer (nearest full above it)
share_map = {}
last_full = None
for i, t in enumerate(indexer_types):
    if t == "full":
        last_full = i
    elif t == "shared":
        if last_full is None:
            raise ValueError("shared layer appears before any full layer")
        share_map[i] = last_full
print(f"layers={LAYERS} shared->full map: {share_map}")


def needed(name):
    m = re.match(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) < LAYERS if m else True


with open(
    os.path.join(SRC, "model.safetensors.index.json"), encoding="utf-8"
) as f:
    idx = json.load(f)
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

with open(
    os.path.join(DST, "model.safetensors.index.json"), "w", encoding="utf-8"
) as f:
    json.dump({"metadata": {"total_size": 0}, "weight_map": new_map}, f)

cfg["num_hidden_layers"] = LAYERS
for k, v in list(cfg.items()):
    if isinstance(v, list) and len(v) == 78:
        cfg[k] = v[:LAYERS]
with open(os.path.join(DST, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

for f in os.listdir(AUX):
    if f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
             "chat_template.jinja"):
        shutil.copy(os.path.join(AUX, f), os.path.join(DST, f))
print(f"done: {len(new_map)} tensors mapped -> {DST}")
