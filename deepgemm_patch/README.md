# deep_gemm patch — GLM-5 32-head MQA logits shape

Stock deep_gemm (the SGLang-era build in
`/opt/venvs/deepgemm/.../deep_gemm`) cannot compile GLM-5's
32-head `fp8_mqa_logits` shape (same class as SGLang issue #19529, fixed
upstream in DeepGEMM 2.5.0). This is a **single-header patch**:

- `sm100_fp8_mqa_logits.cuh` — patched copy of
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh`
- `sm100_fp8_mqa_logits.diff` — the delta vs stock

## Rebuild `deepgemm_patched/` (used via PYTHONPATH by test_dsa.py, threeway_b200.py)

```bash
cp -r <venv>/site-packages/deep_gemm deepgemm_patched/deep_gemm
cp sm100_fp8_mqa_logits.cuh \
   deepgemm_patched/deep_gemm/include/deep_gemm/impls/
# then: PYTHONPATH=/path/to/deepgemm_patched:$PYTHONPATH python test_dsa.py
```

`kernels/b200/dsa/baselines/deepgemm_patched_official/` in the original tree
was exactly this (excluded here to avoid shipping a package copy + .so).

Note: the vLLM E2E container does NOT need this — DeepGEMM 2.5.0 @891d57b
(built from source, unmodified) supports the 32-head shape natively.
