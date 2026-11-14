# B200 DSA kernels

This directory retains the SM100 LiteDSA attention kernels and their
standalone build scripts. The local LiteTopK indexer has been removed.

For the fastest recommended LiteTopK prefill indexer implementations, use
[vLLM #65920](https://github.com/vllm-project/vllm/pull/65920) or
[SGLang #47469](https://github.com/sgl-project/sglang/pull/47469).
See the [repository README](../../../README.md) for integration status,
supported workloads, and the accuracy trade-off.

The vendored-copy mappings below refer to the historical local
`vllm-v026` integration, not upstream vLLM main.

## Generic FP8 LiteDSA

- `litedsa_jit_binding.cu`
- `litedsa.cu`
- `litedsa_attention_sm100.cuh`
- `litedsa_union.cuh`
- `build_litedsa.sh`

The attention and union headers must be byte-identical to vLLM's
self-contained `phase1.cuh` and `litedsa_union.cuh`, respectively.  The
translation unit is intentionally different: this tree exposes TVM-FFI,
whereas vLLM compiles a Torch stable native binding.

Run `build_litedsa.sh` to rebuild `litedsa.so`.  `PYTHON_BIN`, `CUDA_HOME`,
`LITEDSA_BUILD_DIR`, and `LITEDSA_SO` may be overridden; the script discovers
all Python-package include paths from the selected interpreter.

The locally qualified CUDA 13 artifact is retained (but intentionally
gitignored) at
`.codex_variants/litedsa_cuda13_b264c21c9ce1_cuda13.so`; its SHA256 is
`320c29a13df2e2c65c8500587a72c44a02c28649b38f53ce26367f6e4053addb`.
The four-source fingerprint recorded by the build is `b264c21c9ce1`.

## DeepSeek-V4 packed LiteDSA

- `litedsa_dsv4_binding.cu`
- `litedsa_dsv4.cu`
- `litedsa_attention_sm100_dsv4.cuh`
- `litedsa_dsv4_atoms.cuh`
- `build_dsv4.sh`
- `probe_dsv4_smem.cu`
- `build_probe_dsv4_smem.sh`

The four kernel sources must be byte-identical to vLLM's `dsv4_packed/`
copies.  Run the shared-memory probe after every layout change; the kernel is
close enough to the SM100 dynamic shared-memory limit that successful CUDA
compilation alone is not a sufficient check.

`LICENSE.deepseek-flashmla` covers the FlashMLA-derived attention sources.
`DSV4_ZEROCOPY_PLAN.md` records a future optimization and is not production
code.
