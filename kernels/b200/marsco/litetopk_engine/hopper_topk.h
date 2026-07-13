#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---- multi-block threshold-reuse select, migrated from src/litetopk_select.cu ----
// These launchers consume score buffers and threshold metadata produced by a
// score-to-buffer kernel:
//   buf/buf_idx : [R, BUF] candidates and corpus ids
//   origin/inv_delta/th/qcount : bucket coordinates, threshold bucket, valid count
//   cand_*      : eq bucket workspace [R, CAP] + [R]
//   lt_*        : lt bucket workspace [R, CAP] + [R]
//   out_*       : selected top-k values/ids [R, K]
void launch_flash_topk_select_thr_mb_idx_fp16(
    const __half* buf, const int32_t* buf_idx, const int32_t* sample_idx,
    int R, int BUF, int K, int CAP,
    const void* origin, const void* inv_delta,
    const int32_t* th, const int32_t* qcount, int NB,
    __half* cand_val, int32_t* cand_idx, int32_t* cand_cnt,
    __half* lt_val, int32_t* lt_idx, int32_t* lt_cnt,
    __half* out_val, int* out_idx, bool coords_fp16, cudaStream_t stream,
    int skip_zero = 0, bool bucket_space = true);

void launch_flash_topk_select_thr_mb_idx_fp32(
    const float* buf, const int32_t* buf_idx, const int32_t* sample_idx,
    int R, int BUF, int K, int CAP,
    const float* origin, const float* inv_delta,
    const int32_t* th, const int32_t* qcount, int NB,
    float* cand_val, int32_t* cand_idx, int32_t* cand_cnt,
    float* lt_val, int32_t* lt_idx, int32_t* lt_cnt,
    float* out_val, int* out_idx, cudaStream_t stream);

void launch_flash_topk_select_bucket_idx_fp16(
    const __half* bucket_val, const int32_t* bucket_idx, const int32_t* bcount,
    int R, int NB, int BUCKET_CAP, int K, int CAP,
    const int32_t* th,
    __half* cand_val, int32_t* cand_idx, int32_t* cand_cnt,
    __half* lt_val, int32_t* lt_idx, int32_t* lt_cnt,
    __half* out_val, int* out_idx, cudaStream_t stream);

void launch_flash_topk_min_fp16(const __half* in,
                                int B, int N, int K,
                                __half* out_val, int* out_idx,
                                cudaStream_t stream);

void launch_flash_topk_dense_threshold_fp16(
    const __half* buf, int R, int M, int K,
    const void* origin, const void* inv_delta, int NB,
    int32_t* bcount, int32_t* th, bool coords_fp16, cudaStream_t stream,
    bool skip_bcount_zero = false);

void launch_flash_topk_threshold_from_bcount(
    const int32_t* bcount, int R, int NB, int K, int32_t* th, cudaStream_t stream);

void launch_hopper_refresh_threshold_from_bcount(
    int32_t* th, const int32_t* bcount, int R, int NB, int K, cudaStream_t stream);

// RETIRED: dense full-score launcher for fused_ip_dense.  The live path uses
// launch_hopper_sample_score_to_dense_ip_fp16 for the sample window only.
// void launch_hopper_score_to_dense_ip_fp16(
//     const __half* x, const __half* base, int R, int M, int dim,
//     __half* dense_scores, cudaStream_t stream);

void launch_hopper_sample_score_to_dense_ip_fp16(
    const __half* x, const __half* base, int R, int M, int dim,
    __half* dense_scores, cudaStream_t stream);

void launch_hopper_seed_from_sample_fp16(
    const __half* sample_val, const int32_t* sample_idx,
    int R, int K, int BUF, int NB,
    float* origin, float* inv_delta, int32_t* th, int32_t* qcount, int32_t* bcount,
    int32_t* qprev, int32_t* refresh_lock,
    __half* buf_val, int32_t* buf_idx, uint8_t* buf_bucket,
    __half* bucket_val, int32_t* bucket_idx, int bucket_cap, cudaStream_t stream);

void launch_hopper_add_sample_idx_offset(
    int32_t* sample_idx, int total_items, int offset, cudaStream_t stream);

void launch_hopper_seed_from_sample_fp32(
    const __half* sample_val, const int32_t* sample_idx,
    int R, int K, int BUF, int NB,
    float* origin, float* inv_delta, int32_t* th, int32_t* qcount, int32_t* bcount,
    int32_t* qprev, int32_t* refresh_lock,
    float* buf_val, int32_t* buf_idx, uint8_t* buf_bucket,
    __half* bucket_val, int32_t* bucket_idx, int bucket_cap, cudaStream_t stream);

void launch_hopper_score_to_sparse_ip_fp16(
    const __half* x, const __half* base, int R, int M, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    int32_t* qprev, int32_t* refresh_lock,
    __half* buf_val, int32_t* buf_idx, uint8_t* buf_bucket,
    __half* bucket_val, int32_t* bucket_idx, int bucket_cap, int BUF, int NB, int K,
    cudaStream_t stream);

void launch_hopper_score_to_sparse_ip_fp32(
    const __half* x, const __half* base, int R, int M, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    float* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream);

void launch_hopper_smalln_score_to_sparse_ip_gmma_m64n32_fp16(
    const __half* x, const __half* base, int Npad, int logical_rows, int M, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    __half* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream);

void launch_hopper_smalln_score_to_sparse_ip_gmma_m64n32_fp32(
    const __half* x, const __half* base, int Npad, int logical_rows, int M, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    float* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream);

// RETIRED: dense materialization bucket-coordinate helper.
// void launch_hopper_dense_bucket_coords_fp16(
//     const __half* dense_scores, int R, int M,
//     float* origin, float* inv_delta, int NB, cudaStream_t stream);

void launch_hopper_negate_fp16(__half* values, int n, cudaStream_t stream);

// RETIRED: exact small-N and WMMA dense fallbacks, slower than sparse.
// void launch_hopper_smalln_ip_topk_fp16(
//     const __half* x, const __half* base, int N, int M, int dim, int K,
//     __half* partial_val, int32_t* partial_idx, int num_chunks,
//     __half* out_val, int32_t* out_idx, cudaStream_t stream);
//
// void launch_hopper_smalln_score_dense_ip_fp16(
//     const __half* x, const __half* base, int Npad, int M,
//     __half* dense_scores, cudaStream_t stream);

void launch_hopper_smalln_score_dense_ip_gmma_m64n32_fp16(
    const __half* x, const __half* base, int Npad, int logical_rows, int M, int dim,
    __half* dense_scores, cudaStream_t stream);

void launch_hopper_gqa_smalln_score_dense_ip_gmma_m64n8_fp16(
    const __half* x, const __half* base3d, int hkv, int npad_per_group, int logical_rows,
    int M, int base_head_stride, int dim, __half* dense_scores, cudaStream_t stream);

void launch_hopper_gqa_smalln_score_to_sparse_ip_gmma_m64n8_fp16(
    const __half* x, const __half* base3d, int hkv, int npad_per_group, int logical_rows,
    int M, int base_head_stride, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    __half* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream);

void launch_hopper_gqa_paged_score_dense_ip_gmma_m64n8_fp16(
    const __half* x, const __half* kv_data, int hkv, int npad_per_group, int logical_rows,
    int M, int dim, __half* dense_scores, cudaStream_t stream);

void launch_hopper_gqa_paged_score_to_sparse_ip_gmma_m64n8_fp16(
    const __half* x, const __half* kv_data, int hkv, int npad_per_group, int logical_rows,
    int M, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    __half* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream);

void launch_hopper_gqa_group_indexed_score_dense_ip_gmma_m64n8_fp16(
    const __half* x, const __half* kv_data, const int32_t* group_indices,
    int hkv, int logical_rows, int index_stride, int M, int physical_pages, int dim,
    __half* dense_scores, cudaStream_t stream);

void launch_hopper_gqa_group_indexed_score_to_sparse_ip_gmma_m64n8_fp16(
    const __half* x, const __half* kv_data, const int32_t* group_indices,
    int hkv, int logical_rows, int index_stride, int M, int physical_pages, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    __half* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream);

// RETIRED: dense_bucket_fused histogram launcher.
// void launch_hopper_smalln_score_dense_hist_ip_gmma_m64n8_fp16(
//     const __half* x, const __half* base, int Npad, int logical_rows, int M, int dim, int NB, int K,
//     const float* origin, const float* inv_delta, int32_t* gate, int32_t* bcount,
//     __half* dense_scores, cudaStream_t stream, int start_col = 0);
