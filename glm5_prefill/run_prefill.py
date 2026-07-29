#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run GLM FP8 prefill through vLLM.

The input prompt is assembled from the ``text`` column of a Parquet file.
Each request performs prefill followed by one deterministic decode token.
Configuration is supplied through environment variables; see
``glm5_prefill/README.md`` for stock, LiteTopK, and LiteDSA commands.
"""
import json
import os
import time

import pyarrow.parquet as pq

PARQUET = os.environ.get(
    "PARQUET", "/models/glm5/train-00000-of-00002.parquet"
)
MODEL = os.environ.get("MODEL", "/models/glm5-prefill-model")
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "4096 32768 131072 262144").split()]
TP = int(os.environ.get("TP", "4"))
if not LENGTHS or any(length < 1 for length in LENGTHS):
    raise ValueError("LENGTHS must contain positive integers")
if TP < 1:
    raise ValueError("TP must be at least 1")
# Keep space for the decode token.
MAX_LEN = min(max(LENGTHS) + 128, 1048576)
CHUNK = int(os.environ.get("CHUNK", "8192"))
if CHUNK < 1:
    raise ValueError("CHUNK must be at least 1")
REPEATS = int(os.environ.get("REPEATS", "3"))
EAGER = os.environ.get("EAGER", "0") == "1"
ASYNC_SCHED = os.environ.get("ASYNC_SCHED", "0") == "1"
OUTPUT = os.environ.get(
    "OUTPUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefill_results.json"),
)

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    if not os.path.isdir(MODEL):
        raise FileNotFoundError(f"MODEL directory does not exist: {MODEL}")
    if not os.path.isfile(PARQUET):
        raise FileNotFoundError(f"PARQUET file does not exist: {PARQUET}")
    if REPEATS < 1:
        raise ValueError("REPEATS must be at least 1")

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        enable_expert_parallel=True,
        max_model_len=MAX_LEN,
        max_num_batched_tokens=CHUNK,
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.90")),
        trust_remote_code=True,
        # Repeated prompts must execute prefill on every invocation.
        enable_prefix_caching=False,
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
    required_tokens = max(min(length, MAX_LEN - 64) for length in LENGTHS)
    if len(ids_full) < required_tokens:
        raise RuntimeError(
            f"input corpus has {len(ids_full)} tokens, but "
            f"{required_tokens} are required"
        )

    sp = SamplingParams(max_tokens=1, temperature=0)
    results = {}
    for L in LENGTHS:
        L = min(L, MAX_LEN - 64)  # leave room for the decode token
        ids = ids_full[:L]
        # Warm up once, then keep the minimum wall time.
        llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
        ts = []
        for _ in range(REPEATS):
            t = time.time()
            llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
            ts.append(time.time() - t)
        best = min(ts)
        results[L] = best
        print(f"[prefill] tokens={L:>7}  wall(best of {len(ts)})={best:7.3f}s  "
              f"throughput={L / best:9.0f} tok/s", flush=True)

    output_dir = os.path.dirname(os.path.abspath(OUTPUT))
    os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "tp": TP, "chunk": CHUNK, "results": results}, f, indent=2)
    print(f"done: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
