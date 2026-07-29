#!/usr/bin/env bash
set -euo pipefail

# Host-side launcher for the patched vLLM prefill path. The repository must be
# visible at the same path inside the container, or REPO_DIR must be set to its
# container path.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CONTAINER="${CONTAINER:-glm5-prefill}"
REPO_DIR="${REPO_DIR:-$(dirname "$SCRIPT_DIR")}"
MODEL="${MODEL:?set MODEL to a checkpoint directory visible in the container}"
PARQUET="${PARQUET:-/models/glm5/train-00000-of-00002.parquet}"
DEEPGEMM_DIR="${DEEPGEMM_DIR:-/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
LOG="${LOG:-e2e.log}"

docker exec \
  -w "$REPO_DIR/glm5_prefill" \
  -e PATH=/opt/vllm-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  -e CUDA_VISIBLE_DEVICES="$DEVICES" \
  -e DEEPGEMM_DIR="$DEEPGEMM_DIR" \
  -e MODEL="$MODEL" \
  -e PARQUET="$PARQUET" \
  -e TP="$TP" \
  -e VLLM_USE_DEEP_GEMM=1 \
  -e REPEATS="${REPEATS:-1}" \
  -e KVDTYPE="${KVDTYPE:-fp8}" \
  -e MTP="${MTP:-0}" \
  -e EAGER="${EAGER:-0}" \
  -e ASYNC_SCHED="${ASYNC_SCHED:-0}" \
  -e GPU_UTIL="${GPU_UTIL:-0.90}" \
  -e CHUNK="${CHUNK:-8192}" \
  -e LENGTHS="${LENGTHS:-524288 786432 1048512}" \
  -e OUTPUT="${OUTPUT:-$REPO_DIR/glm5_prefill/prefill_results.json}" \
  -e LITETOPK_MODULE_DIR="$REPO_DIR/glm5_prefill/litetopk_vllm" \
  -e LITETOPK_DSA_DIR="$REPO_DIR/kernels/b200/dsa" \
  -e LITEDSA_SO="${LITEDSA_SO:-$REPO_DIR/kernels/b200/dsa/flashinfer_port/litedsa.so}" \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-1024}" \
  -e VLLM_LITETOPK=1 \
  -e VLLM_LITETOPK_MERGE="${VLLM_LITETOPK_MERGE:-1}" \
  -e VLLM_LITETOPK_MIN_S="${VLLM_LITETOPK_MIN_S:-262144}" \
  -e VLLM_LITETOPK_MERGE_CAP="${VLLM_LITETOPK_MERGE_CAP:-24576}" \
  -e VLLM_LITETOPK_HOTSAMPLE="${VLLM_LITETOPK_HOTSAMPLE:-8192}" \
  -e VLLM_LITETOPK_HOTONLY="${VLLM_LITETOPK_HOTONLY:-1}" \
  -e VLLM_LITETOPK_COMPACT="${VLLM_LITETOPK_COMPACT:-0}" \
  -e VLLM_LITETOPK_CHECK="${VLLM_LITETOPK_CHECK:-0}" \
  -e VLLM_LITETOPK_OVF_LOG="${VLLM_LITETOPK_OVF_LOG:-0}" \
  -e VLLM_LITETOPK_LITEDSA="${VLLM_LITETOPK_LITEDSA:-0}" \
  "$CONTAINER" /opt/vllm-venv/bin/python run_prefill.py \
  2>&1 | tee -a "$LOG"
