#!/bin/bash
# HOT-START (HOTONLY) config — the memory-lean fused path: the previous
# layer's top-K is carried forward as the sample (VLLM_LITETOPK_HOTONLY=1 +
# HOTSAMPLE), no cold uniform/strided probe, zero cert/repair/compaction,
# constructive recall by construction. Transient ~1.07GB (vs cert-strided
# 3.2GB), which frees headroom for MTP. GLM-5.2-FP8 78L, TP8+EP, 8xB200.
# Recorded 1M: 128.41s @ HS=8192/QSPLIT=0/cap24576/MTP/util0.92.
# (The cert-strided path is slightly faster on raw wall — 124.41s @1M — but
#  uses more transient memory and no MTP; hot-start is the deployed default.)
# NOTE: prefix caching must stay disabled (run_prefill.py does this).
LOG=${LOG:-e2e_best.log}
docker exec -w /opt/simtopk_repro/glm5_prefill vllm-prefill bash -c '
export PATH=/opt/vllm-venv/bin:$PATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP=8 \
MODEL=/models/glm5-fp8-official \
VLLM_USE_DEEP_GEMM=1 REPEATS=1 KVDTYPE=fp8 MTP=1 GPU_UTIL=0.92 \
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1024 \
VLLM_LITETOPK=1 VLLM_LITETOPK_MERGE=1 VLLM_LITETOPK_MIN_S=262144 \
VLLM_LITETOPK_MERGE_CAP=24576 VLLM_LITETOPK_CAP_CEIL=24576 \
VLLM_LITETOPK_QSPLIT=0 \
VLLM_LITETOPK_HOTSAMPLE=8192 VLLM_LITETOPK_HOTONLY=1 \
VLLM_LITETOPK_COMPACT=0 VLLM_LITETOPK_CERT=0 \
VLLM_LITETOPK_XFLAGS=DSA_BUCKET_GATE4=1 \
CHUNK=8192 LENGTHS="524288 786432 1048512" \
/opt/vllm-venv/bin/python run_prefill.py' 2>&1 | tee -a "$LOG"
# vLLM workers retitle to "VLLM::Worker_TP*" — plain python pkill misses them
docker exec vllm-prefill bash -c 'for p in $(pgrep -f "VLLM::"); do kill -9 $p 2>/dev/null; done; true'
