# LiteTopK / LiteDSA

This repository retains the **LiteDSA attention kernels for NVIDIA SM100
(B200)**. The repository-local LiteTopK indexer and standalone MS MARCO
top-k kernels have been removed. For LiteTopK prefill indexer + top-k,
use the framework implementations below.

## Fastest recommended LiteTopK implementations

These are our two recommended fastest implementation paths for their
respective frameworks and supported workloads. Their PRs report benchmarks
on different GPUs and models; the numbers are not a head-to-head comparison
or a claim that the fused path wins on every shape.

| Framework | Implementation | Integration status (2026-11-14) | Accuracy |
|---|---|---|---|
| vLLM | [vLLM #65920](https://github.com/vllm-project/vllm/pull/65920) | Proposed vLLM integration; PR open, not yet merged | No accuracy loss observed in the reported evaluations; GPU-only recovery handles candidate underflow/overflow. |
| SGLang | [SGLang #47469](https://github.com/sgl-project/sglang/pull/47469) | Merged into SGLang on 2026-11-11; opt-in | Slight accuracy loss is possible; see the reported results below. |

**vLLM:** FP8 indexer, 32/64 index heads, top-k 512/1024/2048 on SM100.
Use the PR branch until it is merged. It keeps the
`VLLM_USE_FUSED_MQA_TOPK` integration and provides GPU-only recovery.
The reported GLM-5.2 MRCR aggregate score is 0.925 versus 0.924 for
`main`, with no observed accuracy degradation. This describes the reported
evaluation, not bitwise identity for every workload.
See [the PR's benchmarks and test plan](https://github.com/vllm-project/vllm/pull/65920)
for supported configurations and routing thresholds.

**SGLang:** FP8/FP4 indexer, 32/64 index heads, top-k 512/1024/2048 on
SM100. Enable it with `SGLANG_OPT_USE_PREFILL_LITE_TOPK=1`; the JIT kernels
require nvcc >= 13.4. The reported DeepSeek-V4.1-Flash RULER2 128K score
decreases from 82.17 to 79.68 (the paired difference is not statistically
significant, p = 0.39). Approximately 1e-6 to 1e-5 of rows can return
slightly fewer than k indices when the sampled threshold is too high.
See [the merged PR](https://github.com/sgl-project/sglang/pull/47469)
for the complete accuracy results, benchmarks, and routing thresholds.

## Retained DSA attention kernels

| Path | Purpose |
|---|---|
| [`kernels/b200/dsa/`](kernels/b200/dsa/README.md) | Generic FP8 grouped LiteDSA and DeepSeek-V4 BF16 packed attention, bindings, and build scripts |
| [`e2e/`](e2e/README.md) | Historical attention-only benchmark results and migration notes |
| `rebuttal/score_concentration.pdf` | Research material |

These attention kernels are separate from the LiteTopK indexer implementations
linked above. The upstream LiteTopK PRs do not integrate these LiteDSA
attention kernels.

### Generic FP8 LiteDSA

Adjacent query tokens share a 128-row SM100 attention tile. A union kernel
deduplicates their top-k KV indices, and a query-major membership mask
preserves each token's attention set. The standalone TVM-FFI module exports
`union_qm` and `masked_mla_fp8`.

```bash
cd kernels/b200/dsa
PYTHON_BIN=python3 CUDA_HOME=/usr/local/cuda ./build_litedsa.sh
```

The selected Python environment must provide PyTorch, TVM-FFI, and
FlashInfer (including its CUTLASS headers). The script targets `sm_100a`.
Override `LITEDSA_BUILD_DIR` and `LITEDSA_SO` to select build and output paths.

### DeepSeek-V4 packed LiteDSA

The BF16 C128A attention kernel packs adjacent query tokens into one
128-row tile, representing each query's compressed prefix and sliding
window with two exact ranges.

```bash
cd kernels/b200/dsa
./build_dsv4.sh
./build_probe_dsv4_smem.sh
```

Run the shared-memory probe after layout changes; compilation alone does
not establish that the SM100 shared-memory limit is respected.
See the [kernel README](kernels/b200/dsa/README.md) for details.

## Migration and licensing

The old local indexer, MS MARCO top-k implementation, and four-arm E2E
launcher are no longer maintained here. Use the linked framework PRs for
LiteTopK integration, testing, and performance measurement. Older source
snapshots remain available in Git history.

Model weights, datasets, generated shared objects, and benchmark logs are
machine-local artifacts. The FlashMLA-derived attention sources retain
their license in
[`LICENSE.deepseek-flashmla`](kernels/b200/dsa/LICENSE.deepseek-flashmla).
