# FlashInfer port — litedsa (whole-DSA attention)

This directory used to also hold a standalone flashinfer-JIT port of the DSA
**indexer** top-k kernel. That build, and the separate standalone repro it was
compared against, are both gone now: the E2E kernel (`../dsa_litetopk.cu` +
`../sm100_dsa_litetopk.cuh`) got backported to the same tuning and is now the
single fastest copy, 1.27–1.28x @1M. **For the indexer, use `../bench_q8192.py`**
(see the parent `README.md`).

What's left here is **litedsa** — see `README_litedsa.md`. It has no
duplicate elsewhere in this repo: it's the standalone tvm_ffi port of the
grouped sparse MLA attention kernel that follows the indexer (union-dedup
masked attention over G=16 packed queries), verified bit-exact against the
vLLM wheel.
