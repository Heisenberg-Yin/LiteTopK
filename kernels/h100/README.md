# LiteTopK — H100 (SM90) port of the B200 kernels

Status **2026-07-17**: both kernels (DSA + marsco) code-complete and
**compile-verified** (`nvcc 12.8, -gencode=arch=compute_90a,code=sm_90a`;
DSA: 168 regs 0 spills both gates; marsco: all instantiations 0 spills).
**Not yet run** — this host has only B200s; recall + perf must be validated
on an H100 before trusting it.

## DSA — `dsa/`

| file | what it is |
|---|---|
| `dsa/sm90_dsa_litetopk.cuh` | SM90 port of the V3 DSA LiteTopK scan: deep_gemm 2.5 `sm90_fp8_mqa_logits` WGMMA scoring loop (verbatim) + the B200 V3 sparse epilogue and V1 KV-split scheduling. |
| `dsa/dsa_litetopk.cu` | Host wrapper — identical to `kernels/b200/dsa/dsa_litetopk.cu` except: includes the sm90 header, `BLOCK_KV=128`, `NUM_SMS=132`, no UMMA-barrier smem, kernel symbol. The fused seed/prep + radix-select kernels are byte-identical (arch-agnostic). |
| `dsa/compile_check_sm90.cu` | Torch-free TU that instantiates the production shape (H=32, D=128, BLOCK_Q=4, BLOCK_KV=128, 4 KV stages, 128+256 threads). |

Build (same DeepGEMM 2.5 tree the B200 kernel uses):

```bash
DG=/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM   # or $DEEPGEMM_DIR
nvcc -O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda \
     -gencode=arch=compute_90a,code=sm_90a -DDSA_BUCKET_GATE4 \
     -I$DG/deep_gemm/include -I$DG/third-party/cutlass/include \
     -c dsa/compile_check_sm90.cu
```

To drive it through `glm5_prefill/litetopk_vllm/litetopk_indexer.py` on an
H100 box, point `_DSA_DIR` at this folder and switch the build flags:
`TORCH_CUDA_ARCH_LIST=9.0a`, gencode `compute_90a,code=sm_90a`.

## Support matrix — B200 attempts vs H100

### Ports unchanged (arch-agnostic)

| B200 attempt | note |
|---|---|
| V1 non-persistent KV-split grid (`blockIdx.x`=q-block, `blockIdx.y`=split) | pure scheduling; H100 has 132 SMs (vs 148), host `NUM_SMS` adjusted |
| Ragged-Q padded rows → empty KV range | pure index math |
| GATE_STRIDE gate prefetch (`__ldcg` one window early) | pure ALU/L2 |
| GATE4 bucket-space float gate (shipped default) | folded `-inv` weights + bits-compare vs `float(g+1)` edge; cand_val stores bq, Python-side select rebase (o'=0, inv'=1) contract unchanged |
| Register-count warp queues (`qn_reg`, no smem bookkeeping) | warp-level |
| `__reduce_or_sync` row-union emit pruning | redux.sync is sm80+ |
| Interior-block range-check elision (unsigned `kstart/kspan` trick) | pure ALU |
| Spare-warp threshold-refresh daemon (`__nanosleep` poll + progress counter) | producer warpgroup has 3 idle warps on SM90 too (2 used, 1 parked) |
| Per-CTA smem histogram (`num_kv_splits==1`) | smem + atomics |
| Probe compaction (magic mul-shift), remap deferred to drain | pure ALU |
| `__stcs` streaming candidate stores | sm70+ |
| `setmaxnreg` register redistribution | **Hopper-native** feature; budget re-derived (see below) |
| TMA tensor maps, mbarrier tx-count pipeline, `elect_one_sync` | all born on SM90 |
| Fused seed/prep kernel, radix selects, external refresh kernel | plain CUDA, copied verbatim |
| deep_gemm 32-head shape (GLM-5) | **no patch needed on H100**: DG 2.5.0's smxx dispatcher asserts `num_heads == 32 or 64` and the sm90 impl takes BLOCK_Q=4/N=128 natively (`deepgemm_patch/` is a B200/legacy-DG artifact) |

### Replaced (Blackwell feature, no SM90 equivalent)

| B200 mechanism | SM90 replacement |
|---|---|
| `tcgen05.mma` (UMMA) issued by a dedicated warp, `umma_arrive` handoff | WGMMA (m64) issued by the math warpgroups themselves; `full_umma/empty_umma` barrier pair deleted |
| TMEM accumulators + per-row-pair `32dp32b` loads + tcgen05 fences | register accumulators (`WGMMA::kNumAccum=64/thread`) + `warpgroup_commit/wait<0>` |
| "1 math thread = 1 KV row" (`BLOCK_KV == kNumMathThreads`) | deep_gemm SM90 fragment layout: thread holds rows `lane/4` and `lane/4+8` of its warp's 16 rows; quad `shfl_xor` head reduction; emit elects `lane%4==0` (8+8-lane ballots instead of 32) |
| Early UMMA release at the last row-pair | not expressible: `warpgroup_wait<0>` releases all at once |
| UMMA_M=128 per warpgroup, BLOCK_KV=256 | WGMMA M=64 per warpgroup, BLOCK_KV=128 (`== kNumMathThreads/2`, deep_gemm's sm90 invariant) |
| setmaxnreg 232/40 (TMEM frees math registers) | 184/40 default (`DSA_REGS_224` optional): accumulator now lives in registers, so the "spend everything on math" ceiling is lower; measured 168 regs, 0 spills |

### Not ported (deliberately)

`DSA_BULK_DRAIN`, `DSA_LANE_EMIT`, `DSA_DIRECT_EMIT`, `DSA_OLD_GATE`, and
`DSA_BUCKET_GATE3` were measured-rejected ablation variants; they have since
been deleted from the B200 kernel too (not just skipped here), so the table
below is a historical record, not a live B200-vs-SM90 diff.

| B200 attempt | why |
|---|---|
| `DSA_BULK_DRAIN` (cp.async.bulk smem→gmem drain) | the instruction **is** SM90-supported; skipped only to keep the first port minimal — the 4-align/sentinel machinery can be lifted verbatim if NCU shows drain stalls |
| `DSA_PERSIST`, `DSA_LANE_EMIT`, `DSA_DIRECT_EMIT`, `*_NULL` diagnostics, `BUCKET_GATE1-3`, `DSA_HIST_DEFER`, `DSA_REGS_240` | measured-rejected or diagnostic-only variants; `#error`-guarded in the sm90 header so a stale xflag fails loudly instead of silently diverging |
| deep_gemm relaxed-UMMA `elect_one` pitfall workaround | N/A — WGMMA has no single-lane-issue requirement |

### Semantics deltas to re-verify on real H100

- **Default gate NaN**: B200's FFMA sign gate *admits* NaN; the SM90 default is
  a plain `v > vth` FSETP which *drops* NaN (old-gate semantics). GATE4 drops
  NaN on both arches. Recall check is the arbiter, as on B200.
- **GATE4 reduction order**: bq is produced by quad `shfl_xor` reduction + one
  FADD instead of B200's FFMA chain — 1-ULP differences vs B200 outputs are
  expected, but gate/hist-feed/select all consume the SAME bq value, so the
  fp16-bucket self-consistency invariant holds by construction.
- **Emit throughput**: ballots carry ≤16 candidates (8 v0 + 8 v1) instead of
  32; queue drains fire ~2× as often per warp. `DSA_WARP_QUEUE_CAP=64` kept.
- **Perf expectations do NOT transfer**: H100 SXM has ~3.35 TB/s HBM3 (vs ~8
  TB/s) and 132 SMs; the 1M-scan wall time will scale roughly with bandwidth.
  KV pipeline depth may want `DSA_V3_KV_STAGES=6..8` (smem headroom: ~103 KB
  used of 227 KB at 4 stages).

## MS MARCO (marsco) on H100 — `marsco/`

Full port of the B200 fused pipeline, same compile-verified status (all 20
template instantiations, 0 spills; QN=8 fp16: 116 regs, QN=64: 130 regs +
64 B local from the `m0a[lane_idx]` dynamic index — same construct as B200's
`m8[lane_idx]`).

| file | what it is |
|---|---|
| `marsco/sm90_litetopk_marsco.cuh` | WGMMA (m64nQNk16 F32F16F16) port of the B200 scan: chunked-D TMA pipeline, bucket-space storage + fp16 rounding invariant, register-count warp queues (QN≤8), parallel-reservation direct emit (QN=64), gate-stride reload, last-CTA in-kernel final refresh — all unchanged. |
| `marsco/litetopk_sm90_torch.cu` | Host binding = B200 file with: `MATH_THREADS=256` (2 warpgroups per BM=128 tile), `SPEC_THREADS=64` (TMA warp + 1 refresh-poller — the B200 layout's second spec warp was the UMMA issuer), `NUM_SMS=132`, no UMMA-barrier smem, `bm=256` variant not ported. Ops renamed `litetopk_sm90::fused_ip_sparse_h100`. |
| `marsco/litetopk_select.cu` + `.h` + `litetopk_topk.h` | **Byte-identical copies** of the B200 select-extract engine (md5-verified) — pure CUDA+CUB, compiles for sm_90a unchanged. |
| `marsco/litetopk_ops.py`, `marsco/bench_marsco_h100.py` | JIT loader + bench, renamed for `sm_90a` / `litetopk_sm90` (`TORCH_CUDA_ARCH_LIST=9.0a`). |

marsco-specific mapping notes (vs the DSA port):

- fp16 IP has **no head reduction** — the WGMMA accumulator element IS the
  score, and each thread's fragment values are UNIQUE (kv row, q col)
  candidates: rows `lane/4`, `lane/4+8` × column pair `(lane%4)*2, +1` per
  8-column group. Emit therefore needs **no lane election** (unlike DSA's
  quad redundancy); ballots per q column have 8 participating lanes carrying
  up to 2 candidates.
- deep_gemm ships no fp16 WGMMA selector (FP8/BF16 only) — the header defines
  `FP16MMASelector` on cute's `MMA_64xNx16_F32F16F16_SS` in the same wrapper
  style.
- fp16 128B-swizzle atoms (64 halfs) are byte-identical in geometry to the
  fp8 D=128 case, so the descriptor constants carry over (SBO=1024, LBO=0).
- The B200 TMEM double-buffered accumulator + early `umma_arrive` release has
  no SM90 equivalent; the tensor core idles during the emit epilogue. If NCU
  shows that gap on real hardware, the known fix is warpgroup ping-pong over
  alternating tiles (see `github_simtopk/h100/dsa/sm90_dsa_marsco.cuh`).
- Ancestry note: the B200 kernel was itself ported FROM the Hopper kernel
  `hopper_gqa_smalln_score_to_sparse_m64n8_full_kernel`
  (`github_simtopk/h100/marsco/data_scripts/hopper_sparse_ip/hopper_topk_select.cu:3945`);
  this port brings the B200-era improvements (select-extract engine,
  bucket-space storage, register queues, chunked-D, in-kernel final refresh)
  back onto SM90 in one bundle.

## GLM-5.2 vLLM E2E on H100

Plausible but unverified (needs an H100 machine): vLLM's DSA prefill path uses
deep_gemm on SM90 (`sm90_fp8_mqa_logits` / `sm90_fp8_paged_mqa_logits` both
exist in DG 2.5.0), the two vLLM patch files are Python-only, and the LiteTopK
hook JIT-builds this folder's `dsa_litetopk.cu`. The only code change needed
is the arch switch in `litetopk_indexer.py` (`10.0a` → `9.0a`, gencode
`sm_100a` → `sm_90a`).
