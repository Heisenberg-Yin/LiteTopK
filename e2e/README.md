# Historical LiteDSA attention benchmarks

The former `run_e2e.sh` four-arm launcher has been removed together with
the repository-local LiteTopK kernels. It depended on the old indexer
source fingerprints, extension ABI, and a separately patched
`vllm-v026` checkout; it is not compatible with the current framework
implementations.

For current LiteTopK integration and benchmarks, use:

- [vLLM #65920](https://github.com/vllm-project/vllm/pull/65920):
  proposed integration, no observed accuracy loss in its reported evaluation.
- [SGLang #47469](https://github.com/sgl-project/sglang/pull/47469):
  merged integration, with a possible slight accuracy loss.

See the [main README](../README.md) for details and enablement instructions.

## Historical attention-only results (2026-08-09)

These results used the old local vLLM integration and are retained only as
historical measurements of the attention kernels. LiteTopK was disabled in
both arms. They are not measurements of the linked upstream PRs.

| Model | Baseline | LiteDSA attention only | Speedup |
|---|---:|---:|---:|
| GLM-5.2 TP8+EP8 | 152.331 s | 143.955 s | 1.058x |
| DeepSeek-V4-Flash TP4+EP4 | 60.007 s | 52.397 s | 1.145x |

Median of three trials after one warmup; 1,048,512 input tokens, chunk
8192, FP8 attention KV, MTP, asynchronous scheduling, and expert parallel.
GLM used grouped FP8 attention; DeepSeek-V4 used BF16 packed C128A attention.

The original runner, full four-arm results, and exact artifact
qualification notes remain available in Git history. Build the retained
attention kernels using the [DSA build instructions](../kernels/b200/dsa/README.md).
