#!/usr/bin/env bash
set -euo pipefail

HOST_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# REPO_DIR is the repository path visible inside the container. By default the
# host and container use the same absolute mount.
REPO_DIR="${REPO_DIR:-$(dirname "$HOST_SCRIPT_DIR")}"
CONTAINER_SCRIPT_DIR="$REPO_DIR/glm5_prefill"
CONTAINER="${CONTAINER:-glm5-prefill}"
LITETOPK_VLLM_SRC="${LITETOPK_VLLM_SRC:-${VLLM_SRC:-/data01/home/ziqi.yin/vllm-litetopk-longcat}}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MODEL_FAMILY="${MODEL_FAMILY:?set MODEL_FAMILY to glm5.2 or longcat}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$HOST_SCRIPT_DIR/results/$STAMP-$MODEL_FAMILY}"

case "$MODEL_FAMILY" in
  glm5.2)
    MODEL="${MODEL:-/data07/glm5-fp8-official}"
    LENGTHS="${LENGTHS:-1048512}"
    TP="${TP:-8}"
    GPU_UTIL="${GPU_UTIL:-0.90}"
    MTP="${MTP:-1}"
    MTP_K="${MTP_K:-5}"
    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-2048}"
    MAX_SEQS="${MAX_SEQS:-32}"
    KV_BLOCKS="${KV_BLOCKS:-17000}"
    VLLM_LITETOPK_HEADROOM="${VLLM_LITETOPK_HEADROOM:-0.0}"
    ;;
  longcat)
    MODEL="${MODEL:-/data07/longcat-19l-mtp}"
    LENGTHS="${LENGTHS:-974848}"
    TP="${TP:-8}"
    GPU_UTIL="${GPU_UTIL:-0.90}"
    MTP="${MTP:-1}"
    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-2048}"
    MAX_SEQS="${MAX_SEQS:-4}"
    KV_BLOCKS="${KV_BLOCKS:-15300}"
    MTP_K="${MTP_K:-3}"
    VLLM_LITETOPK_HEADROOM="${VLLM_LITETOPK_HEADROOM:-0.5}"
    ;;
  *)
    echo "MODEL_FAMILY must be glm5.2 or longcat" >&2
    exit 2
    ;;
esac

run_arm() {
  local mode="$1"
  local output="$2"
  local litetopk_enabled=0
  if [[ "$mode" == litetopk ]]; then
    litetopk_enabled=1
  fi

  docker exec \
    -w "$CONTAINER_SCRIPT_DIR" \
    -e PATH=/opt/vllm-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    -e PYTHONPATH="$LITETOPK_VLLM_SRC" \
    -e LITETOPK_VLLM_SRC="$LITETOPK_VLLM_SRC" \
    -e CUDA_VISIBLE_DEVICES="$DEVICES" \
    -e MODEL_FAMILY="$MODEL_FAMILY" \
    -e MODEL="$MODEL" \
    -e PARQUET="${PARQUET:-/data01/home/ziqi.yin/glm5/train-00000-of-00002.parquet}" \
    -e OUTPUT="$output" \
    -e VLLM_DSA_MODE="$mode" \
    -e VLLM_LITETOPK="$litetopk_enabled" \
    -e LITETOPK_DSA_DIR="${LITETOPK_DSA_DIR:-$REPO_DIR/kernels/b200/dsa}" \
    -e DEEPGEMM_DIR="${DEEPGEMM_DIR:-/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM}" \
    -e VLLM_LITETOPK_BUILD="${VLLM_LITETOPK_BUILD:-/root/.cache/vllm/litetopk_build}" \
    -e VLLM_LITETOPK_MIN_S="${VLLM_LITETOPK_MIN_S:-196608}" \
    -e VLLM_LITETOPK_DENSE_SELECT="${VLLM_LITETOPK_DENSE_SELECT:-1}" \
    -e VLLM_LITETOPK_DENSE_SELECT_MIN_S="${VLLM_LITETOPK_DENSE_SELECT_MIN_S:-40960}" \
    -e VLLM_LITETOPK_DENSE_SELECT_MAX_S="${VLLM_LITETOPK_DENSE_SELECT_MAX_S:-262144}" \
    -e VLLM_LITETOPK_DENSE_SELECT_BINS="${VLLM_LITETOPK_DENSE_SELECT_BINS:-4096}" \
    -e VLLM_LITETOPK_DENSE_SELECT_MIN_LOGITS_MB="${VLLM_LITETOPK_DENSE_SELECT_MIN_LOGITS_MB:-0}" \
    -e VLLM_LITETOPK_MERGE_CAP="${VLLM_LITETOPK_MERGE_CAP:-24576}" \
    -e VLLM_LITETOPK_HOTSAMPLE="${VLLM_LITETOPK_HOTSAMPLE:-8192}" \
    -e VLLM_LITETOPK_HOTONLY="${VLLM_LITETOPK_HOTONLY:-1}" \
    -e VLLM_LITETOPK_HEADROOM="$VLLM_LITETOPK_HEADROOM" \
    -e VLLM_LITETOPK_PREP_TILE="${VLLM_LITETOPK_PREP_TILE:-2048}" \
    -e VLLM_LITETOPK_PREP_UNTILED_MAX_MB="${VLLM_LITETOPK_PREP_UNTILED_MAX_MB:-512}" \
    -e VLLM_LITETOPK_CARRY_ROW_STRIDE="${VLLM_LITETOPK_CARRY_ROW_STRIDE:-8}" \
    -e VLLM_LITETOPK_CARRY_STRIDE16_MAX_NV="${VLLM_LITETOPK_CARRY_STRIDE16_MAX_NV:-131072}" \
    -e VLLM_LITETOPK_CARRY_CUSTOM_TOPK="${VLLM_LITETOPK_CARRY_CUSTOM_TOPK:-1}" \
    -e VLLM_LITETOPK_CACHE_MERGED_KE="${VLLM_LITETOPK_CACHE_MERGED_KE:-1}" \
    -e VLLM_LITETOPK_DEDUP_CARRY_WAIT="${VLLM_LITETOPK_DEDUP_CARRY_WAIT:-1}" \
    -e VLLM_LITETOPK_FUSED_HOT_GATHER="${VLLM_LITETOPK_FUSED_HOT_GATHER:-1}" \
    -e VLLM_LITEDSA_REUSE_OUTPUT_BUFS="${VLLM_LITEDSA_REUSE_OUTPUT_BUFS:-0}" \
    -e VLLM_LITEDSA_DYNAMIC_SPAN="${VLLM_LITEDSA_DYNAMIC_SPAN:-1}" \
    -e VLLM_LITEDSA_UNION_SO="${VLLM_LITEDSA_UNION_SO:-}" \
    -e VLLM_LITETOPK_PROBE_EVERY="${VLLM_LITETOPK_PROBE_EVERY:-8}" \
    -e VLLM_LITETOPK_CHECK="${VLLM_LITETOPK_CHECK:-0}" \
    -e VLLM_LITETOPK_OVF_LOG="${VLLM_LITETOPK_OVF_LOG:-1}" \
    -e VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}" \
    -e VLLM_FLASHINFER_ALLREDUCE_BACKEND="${VLLM_FLASHINFER_ALLREDUCE_BACKEND:-trtllm}" \
    -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" \
    -e TP="$TP" \
    -e CHUNK="${CHUNK:-8192}" \
    -e REPEATS="${REPEATS:-2}" \
    -e GPU_UTIL="$GPU_UTIL" \
    -e KVDTYPE="${KVDTYPE:-fp8}" \
    -e MAX_SEQS="$MAX_SEQS" \
    -e LENGTHS="$LENGTHS" \
    -e MTP="$MTP" \
    -e MTP_K="$MTP_K" \
    -e KV_BLOCKS="$KV_BLOCKS" \
    -e EAGER="${EAGER:-0}" \
    -e ASYNC_SCHED="${ASYNC_SCHED:-0}" \
    -e ATTN_BACKEND="${ATTN_BACKEND:-}" \
    -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
    -e PROF_DIR="${PROF_DIR:-}" \
    -e PROFILE_TRIAL="${PROFILE_TRIAL:-1}" \
    "$CONTAINER" /opt/vllm-venv/bin/python \
    "$CONTAINER_SCRIPT_DIR/run_prefill.py"
}

mkdir -p "$RESULT_DIR"
RAW="$RESULT_DIR/raw.json"
LITE="$RESULT_DIR/litetopk.json"
SUMMARY="$RESULT_DIR/summary.json"

run_arm raw "$RAW" 2>&1 | tee "$RESULT_DIR/raw.log"
sleep "${COOLDOWN_SECONDS:-10}"
run_arm litetopk "$LITE" 2>&1 | tee "$RESULT_DIR/litetopk.log"

"${PYTHON:-python3}" "$HOST_SCRIPT_DIR/compare_results.py" \
  "$RAW" "$LITE" "$SUMMARY" | tee "$RESULT_DIR/summary.txt"

echo "results: $RESULT_DIR"
