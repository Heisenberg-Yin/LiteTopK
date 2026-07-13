# Reproduction log — LiteTopK (2026-07-13)

End-to-end reproduction of the three figures, run from THIS package with the
**renamed LiteTopK code** (renamed engine, renamed module, renamed
`VLLM_LITETOPK*` env vars + deployed vLLM hook). All numbers are fresh runs
on the 8×B200 host; the "README" columns are the previously-recorded tables.

## 1. MARSCO figure — `kernels/b200/marsco/bench_marsco_b200.py`

Container `simtopk`, GPU 0, full 16-cell grid, batch 64, fp32-out, defaults.
Cell = LiteTopK ms / speedup vs cuBLAS dense+topk (README value in parens).
Recall 0.9958–0.9990 on every cell (red line ≥0.9958) — all PASS.

| M | k=128 | k=1024 | k=4096 | k=8192 |
|---:|---|---|---|---|
| 1M | 0.524 / 1.63x (0.497/1.72) | 0.620 / 1.39x (0.577/1.50) | 0.792 / 1.09x (0.756/1.14) | 0.912 / 1.06x (0.864/1.13) |
| 2M | 0.697 / 2.31x (0.683/2.37) | 0.843 / 1.93x (0.800/2.03) | 1.133 / 1.44x (1.064/1.53) | 1.327 / 1.30x (1.249/1.38) |
| 4M | 1.132 / 2.93x (1.106/3.00) | 1.301 / 2.56x (1.243/2.69) | 1.684 / 1.99x (1.566/2.14) | 2.009 / 1.71x (1.868/1.84) |
| 5M | 1.349 / 3.13x (1.319/3.20) | 1.519 / 2.80x (1.466/2.90) | 1.931 / 2.21x (1.795/2.38) | 2.301 / 1.89x (2.146/2.03) |

Reproduces within ~3–8% (fresh run vs the idle-GPU best-config table); the
speedup pattern — grows with M, shrinks with k — matches exactly. The renamed
`litetopk_engine` compiled and ran (op label prints "LiteTopK SM100 fused").

## 2. DSA figure — `kernels/b200/dsa/kda_bench/`

Container `vllm-prefill`, real GLM-5 chunk caches, Q=8192, recall 100%.

**Ours** (`bench_q8192.py` with `VLLM_LITETOPK_PREP_TILE=0` = the record
mode; drives the renamed `litetopk_indexer` module, `[litetopk] using v3
kernel`; recall 100% on every cell):

| S | ours ms | speedup | record |
|---:|---|---|---|
| 256k | 12.69 | 1.05x | 12.61–12.70 / 1.05x ✓ |
| 512k | 23.86 | 1.13x | 23.64–23.66 / 1.13x ✓ |
| 768k | 34.39 | 1.18x | 33.93–34.06 / 1.18–1.20x ✓ |
| 1m | 44.4 (forced-prefix) / 45.4 (auto) | 1.21x / 1.18x | 43.5 / 1.23–1.24x |

256k/512k/768k reproduce the record. The **1m 1.24x record is fully
reproduced** once the two config knobs the figure/deployment actually use are
applied — it is NOT a power-cap effect (that earlier hypothesis was wrong):

| 1m config | ours ms | speedup |
|---|---|---|
| default gate + auto path (shipped `bench_q8192`) | 45.4 | 1.18x |
| default gate + forced-prefix (`strided_plan=False`) | 44.4 | 1.21x |
| **GATE4** kernel + auto | 44.5 | 1.21x |
| **GATE4 + forced-prefix** | **43.6** | **1.24x ✓** |

- **GATE4 bucket-gate kernel** (`VLLM_LITETOPK_XFLAGS=DSA_BUCKET_GATE4=1` —
  the variant the hot-start deployment builds) saves ~0.9ms at 1M, and also
  lifts 256k→1.07x / 512k→1.14x (matches/beats record).
- **Forced-prefix plan** (the sweep/figure methodology; the auto path leaks
  ~1ms at 1M on its deferred-probe/exploration machinery) saves ~1.0ms.
- The two are additive → 43.6ms / 1.24x, reproducing the record exactly on
  this same GPU. recall 100%.

To reproduce the 1m figure number directly:
`VLLM_LITETOPK_PREP_TILE=0 VLLM_LITETOPK_XFLAGS=DSA_BUCKET_GATE4=1` and force
the prefix plan (`try_chunk(..., strided_plan=False)`).

### Kernel-level GATE4 + hot-start (deployed default: prev top-k as sample)

Measured with HOTONLY + HS=8192 + QSPLIT=4096, a fixed `hot_key` so the carry
engages across the timed loop (`hot?=True` confirmed on every row), recall
100% (best-case: same-chunk carry):

| S | official | ours-hot | speedup | vs GATE4-cold |
|---|---|---|---|---|
| 256k | 13.35 | 13.16 | 1.01x | 12.59 / 1.07x → worse |
| 512k | 26.72 | 23.55 | 1.13x | 23.57 / 1.14x → same |
| 768k | 40.32 | 34.40 | 1.17x | 33.89 / 1.18x → same |
| 1m | 53.69 | 44.57 | 1.20x | 44.5 / 1.21x → same |

**Finding: hot-start is not a kernel-speed win.** It only shrinks the sample
GEMM (64K→8K cols), a small fraction of the total; the scan (all S cols) and
select dominate and are unchanged, and QSPLIT=4096's split overhead makes
256k slightly *worse*. Hot-start's value is memory (~1.07GB transient vs
cert-strided 3.2GB) + constructive recall (prev-layer top-k subset bound),
which is why it's the deployed default despite the E2E being ~3% slower than
cert-strided (128.3s vs 124.4s @1M). `bench_q8192.py` therefore stays on the
cold-sample path for the kernel figure.

**Official baseline line** (`bench_vllm_cells.py`, deep_gemm fp8 logits +
top_k_per_row, chunk8192): 256k 12.31, 512k 25.62, 768k 38.54, 1m 52.47 ms.

Both lines reproduce; the renamed module builds the DSA v3 kernel from the
repro copy (`kernels/b200/dsa`) and runs at 100% recall.

## Module cleanup — GATE4 + hot-start as the only path (2026-07-13)

Per request, `litetopk_indexer.py` was stripped to OUR METHOD and the
comparison/ablation paths deleted (backup: `litetopk_indexer_full.py.bak`):

- **Deleted**: the QSPLIT split branch (method uses full 8192, no split),
  the strided / page-probe / two-step / column-strided / PROBE *sampling*
  elif branches, and the cert-strided `use_strided` setter. `use_strided` is
  now True **only** for the hot path. 1227 → 995 lines.
- **Defaults now = the method**: GATE4 kernel
  (`os.environ.setdefault(XFLAGS, DSA_BUCKET_GATE4=1)` — build + select
  consistent), `HOTONLY=1`, `HOTSAMPLE=8192`, `CERT=0`, `COMPACT=0`.
- **Kept**: hot sampling (prev-chunk top-k carry), the exact **cold-prefix
  seed** path (first chunk before a carry exists), scan, select, merge, and
  the unsplit carry read/write (lines ~649 / ~1215 — the E2E hot-start does
  NOT need QSPLIT). A little now-dead cert/compact code remains inside the
  shared scan block (it never executes under CERT=0/COMPACT=0; removing it
  risks the hot path, which reuses the same `use_strided` scan machinery).

Validated on the stripped module, recall **100%** on every size:

| S | cold GATE4 (`bench_q8192`) | hot-start (method) |
|---|---|---|
| 256k | 12.53 / 1.08x | 12.67 / 1.06x |
| 512k | 26.88 / 1.00x | 23.28 / 1.15x |
| 768k | 34.90 / 1.15x | 34.00 / 1.19x |
| 1m | 43.13 / 1.25x | 44.24 / 1.22x |

`bench_q8192` opts out of hot (`HOTONLY=0`) to stay a deterministic cold
figure. Note 512k cold = 1.00x: deleting strided-cert means that band now
uses cold prefix (slower there) — but the **method is hot-start**, where
512k is 1.15x. The hook already skipped compaction when HOTONLY is on
(`if _strided and not _hot`), so the stripped module needs no hook change.

## 3. Prefill figure — `glm5_prefill/run_e2e_best.sh`

Full 78L GLM-5.2-FP8, TP8+EP, 8×B200, renamed vLLM hook + module deployed
(`VLLM_LITETOPK=1 MERGE=1 MIN_S=262144 MERGE_CAP=49152`, LOGITS_MB=4096).
All 8 workers logged `LiteTopK indexer hook enabled (min_s=262144,
sample=65536)` + `using v3 kernel` + `CPU hints validated; sync-free path
active` — the renamed hook/module handled every S≥262144 chunk (not a
fall-through to the official path); zero overflow warnings.

| S | this run | recorded | Δ |
|---:|---|---|---|
| 512K | 47.07s (11139 tok/s) | 46.66s | +0.9% |
| 768K | 82.52s (9531 tok/s) | 81.82s | +0.9% |
| 1M | 125.55s (8351 tok/s) | 124.41s | +0.9% |

Reproduces within ~1%. (`PREFILL-EXIT=0`; the harness marked the task
"killed" only because it caught vLLM's normal SIGTERM engine teardown after
the results printed.)

## Notes

- Container `vllm-prefill`'s deployed vLLM files were temporarily replaced
  with the repro's renamed versions for this run, then restored to their
  prior SimTopK state — the container is back as it was; the renamed code
  lives only in this package (`vllm/` + `glm5_prefill/litetopk_vllm/`).
- Small deltas vs the recorded tables are fresh-run vs idle-GPU-best
  variance (GPU 0 carried a ~1.1GB neighbor allocation throughout); recall
  and the qualitative pattern reproduce exactly.
