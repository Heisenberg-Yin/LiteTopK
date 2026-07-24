# vLLM modifications (v0.23.0)

Unified diffs of the two vLLM v0.23.0 files this repo modifies. No vendored
copy of the modified files lives elsewhere in this repo (that used to be at
`../vllm/` — removed 2026-07-24 as a redundant, derivable duplicate of these
diffs: either format regenerates the other against the pinned v0.23.0 base).

Verified 2026-07-24 by a full-tree diff against a clean
`git clone https://github.com/vllm-project/vllm.git` checked out at `0fc695f`
(`vllm-v0.23.0/` on the build host): **exactly two files are modified**,
nothing else. Regenerated from that same clean checkout, so both diffs below
apply with `patch -p0` (or `git apply -p0`) straight from a fresh v0.23.0 tree
— confirmed by a round-trip: patch applied to a pristine copy reproduced the
pre-removal `../vllm/vllm/...` byte-for-byte.

## Files

- `vllm/model_executor/layers/sparse_attn_indexer.py` (`sparse_attn_indexer.diff`, ~70 added lines)
  - LiteTopK hook, enabled by `VLLM_LITETOPK=1`: lazily imports
    `litetopk_indexer` from the `litetopk_vllm` module directory and routes
    indexer chunks with S >= `MIN_S` to our merged kernel
    (`_litetopk_try` / `_litetopk_merge_wanted`); everything else falls
    through to the official path unchanged.
  - **Module path**: the hook inserts `LITETOPK_MODULE_DIR` (env var,
    defaults to `/opt/simtopk_repro/glm5_prefill/litetopk_vllm`) into
    `sys.path`. When reproducing elsewhere, set that env var to point at
    `../glm5_prefill/litetopk_vllm` — no source edit needed.
- `vllm/v1/attention/backends/mla/indexer.py` (`mla_indexer.diff`, 8 added lines)
  - New env `VLLM_LITETOPK_RUNTIME_LOGITS_MB`: decouples the init-time logits
    pool carve (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`) from the runtime chunk
    budget, so the hybrid deployment can reserve headroom.

## Deploy

Apply the diffs here with `patch -p0` against your own v0.23.0 checkout or an
unpacked wheel's `site-packages/vllm/` (pure-Python changes, no C++ to
recompile — no rebuild needed either way). Then set `LITETOPK_MODULE_DIR` in
the environment (see above) rather than editing the patched file.
