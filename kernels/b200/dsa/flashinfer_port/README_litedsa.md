# litedsa

Grouped (packed) sparse prefill attention for DSA models on SM100: G=16
adjacent queries' heads are packed into one 128-row attention call over the
UNION of their top-k lists (6-7x KV dedup from causal locality), with a
per-query membership bitmask restoring EXACT per-query attend-sets (LSE-equal
to the per-query/stock path).

Ported from `vllm-litetopk-longcat/csrc/libtorch_stable/attention/dsa/litedsa.cu`
(torch::stable::Tensor / STD_TORCH_CHECK) to this repro's tvm_ffi TensorView
binding style, matching `dsa_indexer.cu`'s pattern -- same approach as the
indexer half already living here. Kernel logic byte-for-byte unchanged.

**Verified bit-exact against the wheel** (`verify_litedsa_vs_wheel.py`,
S=1M, G=16): max abs diff **0.000000**. Same kernel, different binding, same
output.

## Files

| file | what |
|---|---|
| `litedsa.so` | prebuilt compiled module (2 ops: `union_qm`, `masked_mla_fp8`) |
| `csrc/litedsa.cu` | tvm_ffi launcher (adapted from the wheel's torch-stable-ABI version) |
| `csrc/litedsa_jit_binding.cu` | TVM-FFI export |
| `csrc/vendor_fmla/` | vendored FlashMLA fp8 subtree (the masked attention kernel itself, framework-agnostic, unmodified) |
| `include/flashinfer/litedsa/litedsa_union.cuh` | union+membership builder kernel (framework-agnostic, unmodified) |
| `verify_litedsa_vs_wheel.py` | correctness check vs the wheel |
| `build_litedsa.ninja.repro` | the exact build recipe (CUTLASS + kerutils + tvm_ffi include paths) |

## Build

```bash
# copy build_litedsa.ninja.repro to build.ninja in a scratch dir (it has
# absolute paths baked in matching this host), or reuse an existing
# flashinfer JIT cache dir's build.ninja as a template and retarget:
#   -I <this dir>/csrc -I <this dir>/csrc/vendor_fmla
#   -I <this dir>/csrc/vendor_fmla/kerutils/include -I <this dir>/include
#   -isystem <cutlass-src>/include -isystem <cutlass-src>/tools/util/include
# source: csrc/litedsa_jit_binding.cu
ninja
```

## Use

```python
import tvm_ffi
m = tvm_ffi.load_module("litedsa.so")
m.union_qm(topk_indices, u_phys, counts, memb_qm, req_id_per_group, block_table,
           block_size, seq_space)  # builds group union + membership bitmask
m.masked_mla_fp8(q, kv, u_phys, bmm1_scale, bmm2_scale, counts, memb_qm,
                 num_heads, out, max_logits, lse)  # grouped masked attention
```

`q`: `[T, num_heads, 576]` fp8_e4m3 (T % G == 0, num_heads * G == 128).
Returns bf16 `[T, num_heads, 512]`. See
`vllm/model_executor/layers/litedsa.py` (in vllm-litetopk-longcat) for the
full orchestration (version cache, persistent buffers) this wraps in
production -- not ported here, since this repro package is kernel-only.

## Speedup (measured, standalone attention-only, vs stock trtllm sparse MLA)

96K-1M sweep (see the parent session's `glm5_prefill_test/whole_dsa_*.json`):
**1.46x-1.83x across the entire range, no crossover** (unlike the indexer,
which loses below ~170K). Attention cost is ~S-independent (fixed topk=2048);
the win is the union dedup, roughly flat.
