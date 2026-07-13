# DSA LiteTopK — B200 (Blackwell / SM100) Port

This directory is the **Blackwell (B200, `sm_100a`) migration** of the Hopper
(H100, `sm_90a`) DSA top-k operator in `../../h100/dsa`.

It reproduces the same task: GLM-5.2-FP8 DSA ReLU-MQA scoring with KV-split
scheduling and a fused sparse sample/write selection, but the scoring inner loop
is rewritten from **Hopper WGMMA + register accumulators** to the **Blackwell
tcgen05 UMMA + Tensor Memory (TMEM)** path.

## What changed vs. the H100 version

| aspect | H100 (`sm_90a`) | B200 (`sm_100a`) |
|---|---|---|
| MMA | `mma::sm90::FP8MMASelector` WGMMA, issued by the math warpgroups | `cute::SM100_MMA_F8F6F4_SS` UMMA (`tcgen05.mma`), issued by a dedicated **UMMA warp** |
| Accumulator | WGMMA registers in the math warpgroups | **Tensor Memory**, read back with `SM100_TMEM_LOAD_32dp32b*` |
| Producer | one producer warpgroup (TMA + periodic refresh) | split into a **TMA-load warp** + **UMMA warp** (mirrors deep_gemm `sm100_fp8_mqa_logits`) |
| Score layout | `m64n8`: a head spans 4 lanes, 2 columns/lane, `__shfl_xor` reduce, elect 1-of-4 lanes to emit | 1 score column per lane (`v_offset = lane_idx`), in-register head reduction, **every lane emits a distinct candidate** |
| `NUM_SMS` | 132 | 148 |
| gencode | `compute_90a` | `compute_100a` |

The DSA-specific parts are **unchanged in spirit**: the bucket-gated sparse
epilogue, the warp-aggregated candidate queue (`DSA_WARP_QUEUE`), the `bcount`
histogram, and the two radix-select post-kernels
(`compact_topk_min_idx_marsco`, `compact_topk_min_thr_marsco`) are ported
verbatim (the selects are architecture-agnostic).

### Refresh behaviour note
The Hopper kernel kept the whole producer warpgroup alive to run a mid-scan
warp-parallel threshold refresh. The SM100 producer is only a TMA warp + a UMMA
warp, so the **math warps run the refresh themselves**: math warp `w` owns query
row `block_q_idx*BLOCK_Q + w`, mapping the `BLOCK_Q` rows 1:1 onto the `BLOCK_Q`
math warps (same "one warp per query row" layout as the Hopper producer). Every
`refresh_every` KV blocks, each warp does a 32-lane prefix-sum over its row's
`bcount` histogram and monotonically tightens `th_bucket` (the next KV block
re-reads it into the gate, so tightening takes effect immediately). Reading a
possibly-stale `bcount` only makes the gate *more* conservative (never fewer
candidates), so recall is preserved. This keeps the H100 dynamic-threshold
behaviour that the operator relies on. (`refresh_every = -1` still runs the
host-side external refresh after the scan, unchanged.)

Requirement: `BLOCK_Q <= #math warps` (`= MATH_THREADS/32`), enforced by a
static assert. With the defaults (`BLOCK_Q=4`, `MATH_THREADS=128`, i.e. 4 math
warps) this is exact.

## Files

| file | purpose |
|---|---|
| `sm100_dsa_marsco.cuh` | SM100 fp8 UMMA/TMEM scoring kernel with KV-split + sparse candidate epilogue |
| `dsa_marsco.cu` | PyTorch/CUDA binding, launch wrapper (TMEM smem sizing, `NUM_SMS=148`), radix top-k select |
| `b200_config.py` | shared build config: CUTLASS 4.2.1 + SM100 deep_gemm include paths, `sm_100a` gencode |
| `test_dsa.py` | single-run correctness/latency check |
| `bench_3way.py` | A/B/O benchmark: stock deep_gemm, ideal KV-split dense, fused sparse |
| `sweep_chunks.py` | benchmark over chunk sizes 1024/2048/4096/8192 and seq lengths 256k/512k/768k/1m |
| `sweep_config.py` | per-(chunk, seq) sample/bucket/refresh config sweep |
| `profile_breakdown.py` | LiteTopK E2E stage breakdown (O_prep / fused scan / select) + emitted-candidate stats |
| `bench_scan_variants.py` | scan-kernel microbench: dense vs sparse vs no-refresh vs `-DDSA_NULL_EPILOGUE` floor; `FLAGS_EXTRA`/`BUILD_TAG` env for config sweeps |

## Environment

- NVIDIA **Blackwell (B200, `sm_100a`)** GPU.
- **CUDA 13.x** toolkit (`nvcc`), gcc with the toolkit's host-compiler support, PyTorch with CUDA.
- **CUTLASS 4.2.1** (full SM100 tcgen05/UMMA/TMEM support) — default
  `/opt/cutlass/include`, override with `DSA_CUTLASS_INCLUDE`.
- **deep_gemm with SM100 kernels** (provides the `fp8_mqa_logits` reference and
  the `deep_gemm/impls/sm100_fp8_mqa_logits.cuh` + `common/sm100_utils.cuh` the
  kernel is based on) — default
  `/opt/venvs/deepgemm/lib/python3.12/site-packages/deep_gemm`,
  override with `DSA_DEEP_GEMM_ROOT`.

The CUDA extension is JIT-compiled by PyTorch on first run into
`$TORCH_EXTENSIONS_DIR` (first invocation `nvcc`s the kernel; later runs are
cached). All paths and the `sm_100a` gencode live in `b200_config.py` and can be
overridden via `DSA_*` environment variables.

## Docker environment (`litetopk`)

A ready-to-use GPU container is provided. It is based on
`nvidia/cuda:12.8.0-devel-ubuntu22.04` (nvcc 12.8 supports `sm_100a`), runs with
`--gpus all` passthrough, has `torch` / `ninja` / `safetensors` preinstalled, and
sets the internal pip + HF mirrors. The data disks (`/data00 /data01 /data02
/data07`) are bind-mounted so CUTLASS 4.2.1, the SM100 deep_gemm and this source
tree are all visible inside. The prepared image is committed as `litetopk:latest`.

```bash
cd /opt/simtopk_src/b200/dsa

# open an interactive shell in the container (creates it if needed)
./run_in_docker.sh

# or run a command directly
./run_in_docker.sh python3 test_dsa.py 256k
./run_in_docker.sh python3 bench_3way.py 256k 512k 768k 1m
```

Validated on this host inside the container:
- 8x NVIDIA B200 visible (`cc 10.0`), `/dev/nvidia-uvm` auto-mounted;
- the extension JIT-builds for `sm_100a` and the radix-select smoke test returns
  recall `1.0000`;
- the SM100 UMMA/TMEM scoring kernel matches a torch ReLU-MQA reference
  bit-for-bit (`max|err| = 0.0`) and the dynamic-threshold (`refresh`) path runs
  at `recall = 100%`.

Manual container launch (equivalent to `run_in_docker.sh`):

```bash
docker run -d --name litetopk --gpus all --ipc=host --shm-size=16g \
  -v /data01:/data01 -v /data00:/data00 -v /data02:/data02 -v /data07:/data07 \
  -e PIP_INDEX_URL=https://pypi.org/simple \
  -e PIP_TRUSTED_HOST=pypi.org \
  -e HF_ENDPOINT=https://huggingface.co \
  -w /opt/simtopk_src/b200/dsa \
  litetopk:latest sleep infinity
```

## Quick run

```bash
cd /opt/simtopk_src/b200/dsa

# optional overrides:
#   export DSA_CUTLASS_INCLUDE=/opt/cutlass/include
#   export DSA_DEEP_GEMM_ROOT=/opt/venvs/deepgemm/lib/python3.12/site-packages/deep_gemm
#   export DSA_CACHE_DIR=/path/to/dsa_caches

# Single cache, default CHUNK=4096
python3 test_dsa.py 256k

# 3-way benchmark
CHUNK=4096 python3 bench_3way.py 256k 512k 768k 1m

# Full chunk sweep
python3 sweep_chunks.py

# Config sweep (sample x buckets x refresh)
python3 sweep_config.py 256k 512k 768k 1m
```

The scripts expect the DSA caches under `$DSA_CACHE_DIR`
(default `/workspace/project/dsa_caches/`). Cache generation is
architecture-independent and is shared with the Hopper bundle — regenerate with
`../../h100/dsa/gen_dsa_caches.py` (it only needs the GLM model shard + corpus +
deep_gemm, not the `sm_90a` kernel).

## Measured B200 results (real GLM-5 DSA caches)

Measured on this host inside the `simtopk` container, on the real GLM-5 DSA
caches under `/data/dsa/`, `K=2048`,
**recall = 100%** for every cell. Kernel config: large tile **BLOCK_KV=256 + 2
math warpgroups**, **spare-warp background threshold refresh**, plus the SM100
pipeline/epilogue optimizations below (all are the current B200 defaults).
Baseline and ours share the exact same kernel/tile/config
(baseline = same kernel built with `-DDENSE_WRITE`), so the comparison is fair —
the larger tile and deeper pipeline speed up *both*, but the fused sparse path
benefits more because its epilogue / candidate write-out is better amortized,
and the spare-warp refresh tightens the gate without stealing math-warp cycles.

Optimizations added on top of the initial 256/2-WG port (stage-level profiling
showed the fused scan is 61-88% of LiteTopK E2E, and its sparse epilogue's
*always-on* per-block cost — not the candidate writes — was 2.5x the dense
epilogue's):

1. **Batched-vote emit**: the per-row gate predicates of a KV block are packed
   into per-lane bits and the divergent emit machinery (per-row
   `__ballot_sync` + warp-queue push, with their branch-reconvergence cost)
   only runs after a single warp-uniform `__any_sync` says some lane passed
   some row. On ~90% of KV blocks the whole warp has no candidate, and the
   epilogue reduces to 4 predicate evaluations + 1 vote.
2. **Strided gate reload** (`DSA_GATE_STRIDE`, default 8): `th_bucket` is
   re-read from global every 8 KV blocks instead of every block, and the
   `(gate+1)/inv` boundary (an fdiv) is recomputed only when the loaded gate
   actually changed. The refresh daemon only publishes every
   `DSA_REFRESH_STRIDE=16` blocks anyway, and a stale gate is merely
   conservative, so recall is unaffected (measured: emitted candidates +<1%).
3. **Vectorized weight loads**: the ReLU-MQA reduction loads the per-head
   weights as `float4` (`LDS.128`) instead of scalars — 32 instead of 128
   shared loads per math thread per KV block, with an unchanged FFMA
   accumulation order (bit-identical scores).
4. **Deeper KV pipeline** (`DSA_Q_STAGES=1`, `DSA_KV_STAGES=6`): each CTA
   processes exactly one q-block, so 2 of the 3 Q stages inherited from
   deep_gemm's persistent scheduler were dead smem; the freed smem plus the
   remaining headroom deepen the KV TMA pipeline 3 -> 6 stages (~223KB of the
   227KB budget).
5. **TMEM accumulator double buffering** (`DSA_TMEM_BUFS=2`): two accumulator
   tiles per math WG (512 of 512 tmem columns) let the UMMA warp compute KV
   block n+1 while the math warps drain block n.

Net effect at 1M/8192: fused scan 60.0 -> 45.3 ms (the dense scan improves
too, 48.8 -> 47.3 ms), LiteTopK E2E 68.0 -> 54.0 ms.

Note on the baseline: the shipped `deep_gemm.fp8_mqa_logits` SM100 kernel cannot
run the GLM-5 DSA shape (`head_dim=128`, `num_heads=32`). Its head_dim=128 path is
real and used by DeepSeek V3.2 (`index_head_dim=128`, `index_n_heads=64`), but the
kernel hard-codes `kNumWeightsInReg=52` and static-asserts `kNumWeightsInReg <=
num_heads`, i.e. it requires `num_heads >= 52`; GLM-5 uses `index_n_heads=32`, so
it fails to compile (SGLang issue #19529, repro model `zai-org/GLM-5-FP8`). No
official DeepSeek Blackwell kernel currently supports this shape out of the box.
So the "DSA" baseline here is our own KV-split **dense** fp8 scoring kernel
(`-DDENSE_WRITE`) + `torch.topk`, i.e. the same "ideal dense" baseline as the
H100 figures.

Data: the **real chunked** GLM-5 caches
(`glm5_{tag}_realtext_chunk{N}.safetensors` under
`/data/dsa_caches/`) where the query batch `Q == chunk` is
genuine prefill-chunk data (not tiled). KV/numerics/score-distribution are all
real.

End-to-end latency (ms), `DSA` = dense + topk, `LiteTopK` = fused sparse (ours):

Measurement protocol: benchmarks run **one at a time** on an otherwise-idle
GPU (no concurrent jobs on the card or its peers), each cell = 5 warmup
iterations + 20 timed iterations averaged with CUDA events.

Config note: after the optimizations below, the (SAMPLE, NB, REFRESH) hyper
params were re-swept (`sweep_config.py` grid + full figure-harness validation
at SAMPLE 8192/16384/32768/65536). No single sample size dominates: the
optimum is data/shape-dependent and the total spread is only ~1.5% (16384/32768
win most cells; 65536 is clearly best on the 512k cache, whose prefix is less
representative — small samples mis-set the initial gate there). NB 128 vs 256
and REFRESH 16/64/128 are flat. The published default stays **SAMPLE=65536,
NB=256, REFRESH=64** (most robust); tune `SAMPLE` per shape via env if you
want the last ~1-2%.

| seq | chunk | DSA (ms) | LiteTopK (ms) | speedup |
|---:|---:|---:|---:|---:|
| 256k | 1024 | 4.66 | 2.96 | 1.57x |
| 256k | 2048 | 8.75 | 5.49 | 1.59x |
| 256k | 4096 | 17.30 | 10.13 | 1.71x |
| 256k | 8192 | 36.13 | 19.08 | 1.89x |
| 512k | 1024 | 9.07 | 5.70 | 1.59x |
| 512k | 2048 | 17.34 | 10.61 | 1.63x |
| 512k | 4096 | 36.18 | 21.12 | 1.71x |
| 512k | 8192 | 72.21 | 40.39 | 1.79x |
| 768k | 1024 | 13.51 | 6.55 | 2.06x |
| 768k | 2048 | 25.73 | 12.08 | 2.13x |
| 768k | 4096 | 54.16 | 23.86 | 2.27x |
| 768k | 8192 | 109.44 | 46.83 | 2.34x |
| 1M | 1024 | 18.10 | 7.41 | 2.44x |
| 1M | 2048 | 36.28 | 13.36 | 2.72x |
| 1M | 4096 | 72.44 | 26.75 | 2.71x |
| 1M | 8192 | 146.55 | 53.65 | 2.73x |

Pattern (same as H100): speedup grows with sequence length and chunk size,
peaking at **2.73x @ 1M / chunk 8192**. Optimization history on real caches:
128/1-WG tile (~1.1-1.44x) → large 256/2-WG tile (~1.4-2.05x) → + spare-warp
background refresh (1.45-2.17x) → + batched-vote emit / strided gate /
float4 weights / Q1+KV6 pipeline / TMEM double buffering (**1.57-2.73x** on
real Q=chunk data; see `OPTIMIZATION_NOTES.md` for the full breakdown).
Reproduce with:

```bash
# inside the container (REAL chunked caches, Q=chunk):
PYTHONPATH=/opt/venvs/deepgemm/lib/python3.12/site-packages \
DSA_CACHE_DIR=/data/dsa_caches \
/usr/bin/python3.12 sweep_b200_real_chunk.py       # -> figure_b200_realchunk.json
python3 figures/dsa_bench.py                        # -> figures/dsa.pdf / .png
```

The grouped-bar figure mirroring `../../h100/figures/dsa.pdf` is at
[`figures/dsa.pdf`](figures/dsa.pdf) (and `figures/dsa.png`).

### Three-way comparison vs the official (patched) DeepGEMM kernel

To include the official DeepSeek kernel as a reference, the official DeepGEMM
`sm100_fp8_mqa_logits.cuh` was copied to a writable tree and given the one-line
SGLang #19529 patch so it compiles for `num_heads=32`:

```cpp
// constexpr uint32_t kNumWeightsInReg = 52;                 // original (requires >=52 heads)
constexpr uint32_t kNumWeightsInReg = (kNumHeads < 52) ? (kNumHeads / 4 * 4) : 52;  // patched
```

(patched copy: `/home/user/deepgemm_patched/deep_gemm`; it produces
bit-identical logits to our dense kernel, top-k recall 100%.)

Three paths, all recall = 100% on the real chunked caches, `K=2048`:
- **official** = patched `deep_gemm.fp8_mqa_logits` (full `[Q,S]` logits) + `torch.topk`
- **dense** = our KV-split dense fp8 kernel + `torch.topk` (a *stronger* baseline)
- **LiteTopK** = our fused sparse scan + radix select (ours)

| seq | chunk | official (ms) | dense (ms) | LiteTopK (ms) | LiteTopK vs official | vs dense |
|---:|---:|---:|---:|---:|---:|---:|
| 256k | 8192 | 43.45 | 36.12 | 19.10 | 2.27x | 1.89x |
| 512k | 8192 | 88.87 | 72.47 | 40.44 | 2.20x | 1.79x |
| 768k | 8192 | 135.51 | 109.42 | 47.23 | 2.87x | 2.32x |
| 1M | 2048 | 45.31 | 36.30 | 13.46 | 3.37x | 2.70x |
| 1M | 4096 | 88.17 | 72.56 | 27.19 | 3.24x | 2.67x |
| 1M | 8192 | 183.15 | 147.05 | 54.28 | **3.37x** | 2.71x |

Two takeaways:
- The **official `fp8_mqa_logits` (non-paged) kernel is the slowest** of the
  three. The dominant reason is that it **materializes the full `[Q,S]` logits to
  HBM** (e.g. 32 GB at 1M / chunk 8192) and then `torch.topk` reads that 32 GB
  back — a full extra HBM round-trip. LiteTopK never materializes the full score
  matrix (the sparse gate only writes candidates), and is up to **3.37x** faster
  than this official (patched) kernel at 1M / chunk 8192.
- Scheduling note: DeepGEMM DOES implement **KV-split** — in the *paged* kernel
  `fp8_paged_mqa_logits` (template param `SPLIT_KV`, a metadata-driven
  `PagedMQALogitsScheduler` + `get_paged_mqa_logits_metadata`); our KV-split is
  modeled on it. That paged kernel is designed for **decode** (small `kNextN`,
  paged KV cache with a `block_table`), not the **prefill-chunk** case measured
  here (`Q = chunk` up to 8192, contiguous KV), so the non-paged `fp8_mqa_logits`
  is the appropriate official prefill baseline. (At these chunk sizes the q-blocks
  already exceed the SM count, so KV-split would not change the picture anyway —
  the logits materialization is what dominates.) Our dense baseline also
  materializes the logits but with a better tile/epilogue, so it is a *harder*
  opponent than the official non-paged kernel.
- Full 4x4 numbers are in `threeway_b200.json`; the figure is at
  [`figures/dsa_3way.pdf`](figures/dsa_3way.pdf) (green=official, blue=dense,
  orange=LiteTopK). Reproduce with:

```bash
PYTHONPATH=/home/user/deepgemm_patched:/opt/venvs/deepgemm/lib/python3.12/site-packages \
DSA_CACHE_DIR=/data/dsa_caches \
/usr/bin/python3.12 threeway_b200.py           # -> threeway_b200.json
python3 figures/dsa_3way_bench.py               # -> figures/dsa_3way.pdf / .png
```

## Tuning knobs (compile-time `-D…`)

Same set as the Hopper kernel; pass via `extra_cuda_cflags` (see `b200_config.py`):

- `DSA_BLOCK_Q` (default 4), `DSA_BLOCK_KV` (default 256 — must equal the math
  thread count on SM100), `DSA_MATH_THREADS` (default `= DSA_BLOCK_KV`).
- `DSA_Q_STAGES` (default 1) / `DSA_KV_STAGES` (default 6) — TMA pipeline
  depths. One Q stage suffices (one q-block per CTA); KV 6 is the deepest that
  fits the 227KB smem budget with the warp queue.
- `DSA_TMEM_BUFS` (default 2) — tensor-memory accumulator tiles per math WG
  (2 = double buffered UMMA/TMEM-readback overlap; 1 = the old serialized
  behaviour).
- `DSA_GATE_STRIDE` (default 16) — re-read `th_bucket` (and recompute the fdiv
  boundary on change) every N KV blocks. Stale gates are conservative-safe.
  Swept {4,8,16,32}: monotonically faster with larger stride but all within
  ~1%; 16 matches the refresh publish cadence.
- `DSA_WARP_QUEUE` (default 1) / `DSA_WARP_QUEUE_CAP` (default 32) — warp-local
  candidate queue before flushing to the compact buffer.
- `DSA_MERGE_BCOUNT` (default 0) — merge same-bucket `bcount` atomics per warp.
- `-DDENSE_WRITE` — dense-store baseline (the `B` path in the benchmarks).
- `-DDSA_NULL_EPILOGUE` — measurement-only build: score + reduce, no gate/store
  (the pure TMA/UMMA/TMEM floor used by `bench_scan_variants.py`).
