#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLM-5.2 fp8 end-to-end PREFILL test via vLLM.

Model: 16-layer truncated shadow of /models/glm5-code (weights of
the first 16 layers are the real GLM-5.2 fp8 weights; full model = 1.51TB and
does not fit on one 8xB200 node). Exercises the real vLLM prefill path:
fp8 block GEMMs, MLA, MoE (EP), and the DSA sparse-attention indexer
(deep_gemm.fp8_mqa_logits, patched for the 32-head GLM shape).

Prompts are built from the wikitext-style parquet by concatenating rows and
tokenizing with the GLM tokenizer; max_tokens=1 makes each generate() call
prefill + a single decode step.

Run (inside the `vllm-prefill` container):
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  PYTHONPATH=/opt/simtopk_src/b200/dsa/baselines/deepgemm_patched_official \
  VLLM_USE_DEEP_GEMM=1 \
  /opt/vllm-venv/bin/python run_prefill.py
"""
import json
import os
import time

import pyarrow.parquet as pq

PARQUET = "/models/glm5/train-00000-of-00002.parquet"
MODEL = os.environ.get("MODEL", "/models/glm5-prefill-model")
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "4096 32768 131072 262144").split()]
TP = int(os.environ.get("TP", "4"))
# cap at the model's max_position_embeddings (1M)
MAX_LEN = min(max(LENGTHS) + 128, 1048576)
CHUNK = int(os.environ.get("CHUNK", "8192"))  # chunked-prefill chunk size
EAGER = os.environ.get("EAGER", "0") == "1"          # enforce_eager: no compile/cudagraphs
ASYNC_SCHED = os.environ.get("ASYNC_SCHED", "0") == "1"  # overlap scheduler CPU with GPU

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        enable_expert_parallel=True,
        max_model_len=MAX_LEN,
        max_num_batched_tokens=CHUNK,
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.90")),
        trust_remote_code=True,
        # honest prefill timing: identical repeated prompts must NOT hit the
        # prefix cache (vLLM v1 enables it by default)
        enable_prefix_caching=False,  # per official docs: disable for benchmarking
        kv_cache_dtype=os.environ.get("KVDTYPE", "auto"),
        speculative_config=({"method": "mtp", "num_speculative_tokens": 5}
                            if os.environ.get("MTP", "0") == "1" else None),
        enforce_eager=EAGER,
        async_scheduling=ASYNC_SCHED,
    )
    print(f"[load] model ready in {time.time() - t0:.1f}s", flush=True)

    tok = llm.get_tokenizer()
    buf, nchar = [], 0
    for r in pq.read_table(PARQUET, columns=["text"])["text"].to_pylist():
        if r and r.strip():
            buf.append(r)
            nchar += len(r)
        if nchar > MAX_LEN * 8:  # enough chars to cover MAX_LEN tokens
            break
    ids_full = tok.encode("\n".join(buf))
    print(f"[data] corpus tokens available: {len(ids_full)}", flush=True)

    sp = SamplingParams(max_tokens=1, temperature=0)
    results = {}
    for L in LENGTHS:
        L = min(L, MAX_LEN - 64)  # leave room for the decode token
        ids = ids_full[:L]
        # warmup once, then 3 timed runs
        llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
        ts = []
        for _ in range(int(os.environ.get("REPEATS", "3"))):
            t = time.time()
            llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
            ts.append(time.time() - t)
        best = min(ts)
        results[L] = best
        print(f"[prefill] tokens={L:>7}  wall(best of 3)={best:7.3f}s  "
              f"throughput={L / best:9.0f} tok/s", flush=True)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefill_results.json"), "w") as f:
        json.dump({"model": MODEL, "tp": TP, "chunk": CHUNK, "results": results}, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
