# vLLM modifications (v0.23.0)

The **full modified source tree lives at `../vllm/`** (v0.23.0 @0fc695f,
.git stripped, our changes applied). This directory holds only the unified
diffs vs official v0.23.0, for quick review or for patching an installed
wheel directly.

Verified by full-tree diff of the deployed container venv
(`vllm-prefill:/opt/vllm-venv/.../vllm`) against a pristine v0.23.0 checkout:
**exactly two files are modified**, nothing else.

## Files

- `vllm/model_executor/layers/sparse_attn_indexer.py` (`sparse_attn_indexer.diff`, ~70 added lines)
  - LiteTopK hook, enabled by `VLLM_LITETOPK=1`: lazily imports
    `litetopk_indexer` from the `litetopk_vllm` module directory and routes
    indexer chunks with S >= `MIN_S` to our merged kernel
    (`_litetopk_try` / `_litetopk_merge_wanted`); everything else falls
    through to the official path unchanged.
  - **Hardcoded module path**: the hook inserts
    `/opt/litetopk_repro/glm5_prefill/litetopk_vllm` into `sys.path`.
    When reproducing elsewhere, point it at `../glm5_prefill/litetopk_vllm`.
- `vllm/v1/attention/backends/mla/indexer.py` (`mla_indexer.diff`, 8 added lines)
  - New env `VLLM_LITETOPK_RUNTIME_LOGITS_MB`: decouples the init-time logits
    pool carve (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`) from the runtime chunk
    budget, so the hybrid deployment can reserve headroom.

## Deploy

Fastest (what we actually ran): install the official vllm 0.23.0 wheel, then
overwrite the two files in `site-packages/vllm/...` with the copies from
`../vllm/vllm/...` — no rebuild needed (pure-Python changes).
Alternatively apply the diffs with `patch -p0`, or build `../vllm/` from
source.
