# LiteDSA grouped sparse attention

This directory contains the B200 (`sm_100a`) LiteDSA attention module used
after the LiteTopK DSA indexer. Sixteen adjacent query tokens are packed into
128 query-head rows. `union_qm` builds a union of their selected KV positions
and a per-query membership mask; `masked_mla_fp8` applies sparse MLA attention
over that representation.

The DSA indexer is separate and is built from the parent directory. Use
`../bench_q8192.py` to exercise the indexer.

## Files

| File | Purpose |
|---|---|
| `litedsa.so` | Prebuilt TVM-FFI module exporting `union_qm` and `masked_mla_fp8`. |
| `csrc/litedsa.cu` | Tensor validation and CUDA launch wrapper. |
| `csrc/litedsa_jit_binding.cu` | TVM-FFI registration unit. |
| `include/flashinfer/litedsa/litedsa_union.cuh` | Union and membership-mask kernel. |
| `csrc/vendor_fmla/` | Vendored FlashMLA-derived fp8 attention headers; see its README and license. |
| `build_litedsa.ninja.repro` | Host-specific Ninja template used to document compiler and linker inputs. |
| `verify_litedsa_vs_wheel.py` | Compares this module with an external reference implementation. |

## Runtime requirements

- An NVIDIA B200 GPU.
- A CUDA 12-compatible runtime.
- Python with PyTorch and `tvm_ffi`.
- fp8 MLA query/KV tensors and int32 top-k indices in the shapes described
  below.

`litedsa.so` is an architecture- and ABI-specific build artifact. Rebuild it
when the CUDA toolchain, TVM-FFI ABI, C++ ABI, or target architecture differs
from the environment that produced the bundled file.

Check that the module can be loaded:

```bash
LITEDSA_SO=/path/to/litetopk_github_clone/kernels/b200/dsa/flashinfer_port/litedsa.so \
python3 -c 'import os, tvm_ffi; tvm_ffi.load_module(os.environ["LITEDSA_SO"]); print("load OK")'
```

## Operator contract

```python
import tvm_ffi

module = tvm_ffi.load_module("/path/to/litedsa.so")
module.union_qm(
    topk_indices,
    union_indices,
    counts,
    membership,
    request_id_per_group,
    block_table,
    block_size,
    seq_space,
)
module.masked_mla_fp8(
    packed_q,
    kv_cache,
    union_indices,
    bmm1_scale,
    bmm2_scale,
    counts,
    membership,
    num_heads,
    output,
    max_logits,
    lse,
)
```

The wrapper in
`glm5_prefill/litetopk_vllm/litedsa_attn.py` prepares these buffers. Its
public input contract is:

- query: `[T, num_heads, 576]`, fp8 e4m3;
- KV cache: paged fp8 cache with width 576;
- top-k indices: `[T, K]`, int32;
- `T % group_size == 0`;
- `num_heads * group_size == 128`;
- output: `[T, num_heads, 512]`, bf16.

To route vLLM sparse MLA calls through this wrapper, apply
`vllm_patches/flashinfer_mla_sparse.diff`, set `LITETOPK_MODULE_DIR` to
`glm5_prefill/litetopk_vllm`, and set `VLLM_LITETOPK_LITEDSA=1`. The patch
requirements are documented in
[`vllm_patches/README.md`](../../../../vllm_patches/README.md).

## Build from source

`build_litedsa.ninja.repro` is a path-specific template, not a portable build
system. Before using it, update every absolute include, source, Python, CUDA,
and output path for the target environment. Required development inputs are:

- `nvcc` and a C++17 host compiler;
- Python development headers;
- TVM-FFI headers matching the runtime package;
- CUTLASS/CuTe headers with SM100 support;
- CCCL/CUB and libcu++ headers;
- the local `csrc`, `csrc/vendor_fmla`, and `include` directories.

After adapting the template:

```bash
mkdir -p /tmp/litedsa_build
ninja -f /path/to/adapted-build.ninja
cp /tmp/litedsa_build/dsa_indexer.so /path/to/flashinfer_port/litedsa.so
```

In the current `glm5-prefill` container, use
`/opt/vllm-venv/bin/ninja` instead of a bare `ninja` command.

The output name in the supplied template is `dsa_indexer.so`; rename or copy
it to the path passed through `LITEDSA_SO`. Loading the resulting module is
the minimum build check.

## Verify against the external reference

`verify_litedsa_vs_wheel.py` requires dependencies that are not shipped here:

- a compatible vLLM wheel or checkout providing
  `vllm.model_executor.layers.litedsa.litedsa_masked_mqa`;
- a safetensors cache containing `mla_q`, `mla_kv`, `topk_idx`, and `mla`
  metadata;
- a B200 GPU.

Stock vLLM 0.23 does not provide that reference module. In an environment
that does, run:

```bash
CUDA_VISIBLE_DEVICES=0 \
LITEDSA_SO=/path/to/flashinfer_port/litedsa.so \
LITETOPK_DSA_CACHE=/path/to/glm5_1m_realtext_chunk8192.safetensors \
python3 verify_litedsa_vs_wheel.py
```

The script prints maximum and mean absolute differences and exits nonzero
when the maximum exceeds `LITEDSA_ATOL` (default `0.01`).
