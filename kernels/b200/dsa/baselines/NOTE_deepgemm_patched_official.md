# deepgemm_patched_official/ (excluded from this package)

The original tree has `baselines/deepgemm_patched_official/deep_gemm/` — a
full copy of the sglang-venv deep_gemm package whose ONLY difference from
stock is one header. Rebuild it from `../../../../deepgemm_patch/`
(patched `sm100_fp8_mqa_logits.cuh` + diff + instructions).
