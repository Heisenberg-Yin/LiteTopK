#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// FlashTopk-v1：对 [B, N] 的每一行求前 K 个值及其列索引。
//   算法（区别于 radixselect 的 4-pass radix）：采样定阈 + 单遍直方图 + 边界桶精确选。
//     1. sample：每行单 block 在前缀采样子集(S=max(k,16384))上 in-block radix 求采样第 K
//        小 enc 作分桶上界 hi（采样第 K 小 >= 全集第 K 小，安全），min 作下界 lo。
//     2. hist（全量 1 遍）：每元素分桶，per-row 直方图（shared 累加 + 全局 atomic 归约）。
//     3. threshold：cumsum 求精确阈值桶 T 与 count_below。
//     4. gather（全量 1 遍）：bucket<T 直接写出（确定入选），bucket==T 收集进候选 buffer。
//     5. boundary：候选 buffer 内 in-block 精确选 need=K-count_below 个补齐（含 tie / overflow 回退）。
//   把 radix-select 的「5 遍全量扫描」压成「采样(~1.6%) + 2 遍全量」。fp16 原生 16-bit 编码。
//   行首 16B 对齐(N % VEC == 0)时 hist/gather 走 float4 向量化读。
//   输出未排序（与 torch.topk(sorted=False) 等价），ties 任意补足。
//
//   in       : [B, N]，行优先连续
//   out_val  : [B, K]，同 in 的 dtype
//   out_idx  : [B, K]，int32，列索引（落在 [0, N)）
//   K 必须满足 1 <= K <= N。
//   workspace 由实现内部以 stream-ordered 方式分配/释放，调用方无需准备。

void launch_flash_topk_min_fp32(const float* in,
                                int B, int N, int K,
                                float* out_val, int* out_idx,
                                cudaStream_t stream);

void launch_flash_topk_min_fp16(const __half* in,
                                int B, int N, int K,
                                __half* out_val, int* out_idx,
                                cudaStream_t stream);

void launch_flash_topk_min_bf16(const __nv_bfloat16* in,
                                int B, int N, int K,
                                __nv_bfloat16* out_val, int* out_idx,
                                cudaStream_t stream);

// 取每行 K 个【最大】值与索引（与 tf.math.top_k 等价语义，输出未排序）。
//   实现方式：先把输入取负 -> 求 K 个最小 -> 再把输出 values 取负回来。
//   workspace 全部由内部以 stream-ordered 方式分配/释放。
void launch_flash_topk_max_fp32(const float* in,
                                int B, int N, int K,
                                float* out_val, int* out_idx,
                                cudaStream_t stream);

void launch_flash_topk_max_fp16(const __half* in,
                                int B, int N, int K,
                                __half* out_val, int* out_idx,
                                cudaStream_t stream);

void launch_flash_topk_max_bf16(const __nv_bfloat16* in,
                                int B, int N, int K,
                                __nv_bfloat16* out_val, int* out_idx,
                                cudaStream_t stream);

// ---- 调用方预分配 workspace 版本（消除内部 cudaMallocAsync 开销）----------------
// 用法：先用 *_workspace_bytes(...) 查询所需字节数，分配单块 >= 该大小的设备 blob，
//   再传给对应 *_ws(...)。blob 由调用方（如 TF op 的 allocate_temp）管理生命周期。
//   blob 需按 256B 对齐（TF/CUDA 设备分配天然满足）。
size_t flash_topk_max_workspace_bytes_fp32(int B, int N, int K);
size_t flash_topk_max_workspace_bytes_fp16(int B, int N, int K);
size_t flash_topk_max_workspace_bytes_bf16(int B, int N, int K);

void launch_flash_topk_max_fp32_ws(const float* in,
                                   int B, int N, int K,
                                   float* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream);

void launch_flash_topk_max_fp16_ws(const __half* in,
                                   int B, int N, int K,
                                   __half* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream);

void launch_flash_topk_max_bf16_ws(const __nv_bfloat16* in,
                                   int B, int N, int K,
                                   __nv_bfloat16* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream);

size_t flash_topk_min_workspace_bytes_fp32(int B, int N, int K);
size_t flash_topk_min_workspace_bytes_fp16(int B, int N, int K);
size_t flash_topk_min_workspace_bytes_bf16(int B, int N, int K);

void launch_flash_topk_min_fp32_ws(const float* in,
                                   int B, int N, int K,
                                   float* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream);

void launch_flash_topk_min_fp16_ws(const __half* in,
                                   int B, int N, int K,
                                   __half* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream);

void launch_flash_topk_min_bf16_ws(const __nv_bfloat16* in,
                                   int B, int N, int K,
                                   __nv_bfloat16* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream);

// ---- 阈值复用选择（配合融合 _flat_kernel）-------------------------------------
// 调用方已持有每行收敛阈值桶 th 与分桶坐标 (origin, inv_delta)，故跳过 sample/hist/
// threshold，仅做 gather + boundary：bucket=(int)((v-origin)*inv_delta) 钳到 [0,NB-1]，
// bucket<th 确定入选、bucket==th 边界桶精确取 need 个、bucket>th 或非有限丢弃。
//   buf      : [R, BUF]，行优先连续（尾部空位须预填 +inf，会被自然剔除）
//   origin   : [R] float；inv_delta : [R] float；th : [R] int32（桶号，<NB）
//   out_val  : [R, K] 同 buf dtype；out_idx : [R, K] int32（buffer 内列位置 [0,BUF)）
//   workspace 内部不需要；out_idx 启动时清零兜底 BUF 溢出欠填。
void launch_flash_topk_select_thr_fp32(const float* buf, int R, int BUF, int K,
                                       const float* origin, const float* inv_delta,
                                       const int32_t* th, const int32_t* qcount, int NB,
                                       float* out_val, int* out_idx, cudaStream_t stream);
void launch_flash_topk_select_thr_idx_fp32(const float* buf, const int32_t* buf_idx,
                                           const int32_t* sample_idx,
                                           int R, int BUF, int K,
                                           const float* origin, const float* inv_delta,
                                           const int32_t* th, const int32_t* qcount, int NB,
                                           float* out_val, int* out_idx, cudaStream_t stream);

void launch_flash_topk_select_thr_fp16(const __half* buf, int R, int BUF, int K,
                                       const void* origin, const void* inv_delta,
                                       const int32_t* th, const int32_t* qcount, int NB,
                                       __half* out_val, int* out_idx,
                                       bool coords_fp16, cudaStream_t stream);
void launch_flash_topk_select_thr_idx_fp16(const __half* buf, const int32_t* buf_idx,
                                           const int32_t* sample_idx,
                                           int R, int BUF, int K,
                                           const void* origin, const void* inv_delta,
                                           const int32_t* th, const int32_t* qcount, int NB,
                                           __half* out_val, int* out_idx,
                                           bool coords_fp16, cudaStream_t stream);

void launch_flash_topk_select_thr_bf16(const __nv_bfloat16* buf, int R, int BUF, int K,
                                       const void* origin, const void* inv_delta,
                                       const int32_t* th, const int32_t* qcount, int NB,
                                       __nv_bfloat16* out_val, int* out_idx,
                                       bool coords_bf16, cudaStream_t stream);
void launch_flash_topk_select_thr_idx_bf16(const __nv_bfloat16* buf, const int32_t* buf_idx,
                                           const int32_t* sample_idx,
                                           int R, int BUF, int K,
                                           const void* origin, const void* inv_delta,
                                           const int32_t* th, const int32_t* qcount, int NB,
                                           __nv_bfloat16* out_val, int* out_idx,
                                           bool coords_bf16, cudaStream_t stream);

// ---- multi-block 阈值复用选择（mb）-------------------------------------------
// 把整 BUF 扫描拆到多 block 并行分桶：b<th→lt 缓冲、b==th→eq(cand) 缓冲，b>th 丢弃；再单
// block finalize（lt 全 copy + eq 上小 radix 补齐）。大 BUF（阈值解耦放大 buffer）下单 block
// 串行扫 BUF 是瓶颈，mb 显著缩短。cand_*（eq）与 lt_* 缓冲均 [R,CAP] val/idx + [R] cnt，
// 由调用方分配（CAP>=K）。
void launch_flash_topk_select_thr_mb_idx_fp32(
    const float* buf, const int32_t* buf_idx, const int32_t* sample_idx,
    int R, int BUF, int K, int CAP,
    const float* origin, const float* inv_delta,
    const int32_t* th, const int32_t* qcount, int NB,
    float* cand_val, int32_t* cand_idx, int32_t* cand_cnt,
    float* lt_val, int32_t* lt_idx, int32_t* lt_cnt,
    float* out_val, int* out_idx, cudaStream_t stream);
void launch_flash_topk_select_thr_mb_idx_fp16(
    const __half* buf, const int32_t* buf_idx, const int32_t* sample_idx,
    int R, int BUF, int K, int CAP,
    const void* origin, const void* inv_delta,
    const int32_t* th, const int32_t* qcount, int NB,
    __half* cand_val, int32_t* cand_idx, int32_t* cand_cnt,
    __half* lt_val, int32_t* lt_idx, int32_t* lt_cnt,
    __half* out_val, int* out_idx, bool coords_fp16, cudaStream_t stream);
void launch_flash_topk_select_thr_mb_idx_bf16(
    const __nv_bfloat16* buf, const int32_t* buf_idx, const int32_t* sample_idx,
    int R, int BUF, int K, int CAP,
    const void* origin, const void* inv_delta,
    const int32_t* th, const int32_t* qcount, int NB,
    __nv_bfloat16* cand_val, int32_t* cand_idx, int32_t* cand_cnt,
    __nv_bfloat16* lt_val, int32_t* lt_idx, int32_t* lt_cnt,
    __nv_bfloat16* out_val, int* out_idx, bool coords_bf16, cudaStream_t stream);

// ---- packed 分段 K-min select（fp16 专用）-----------------------------------
// buf_pack[R, NSEG*CAP] int32：packed = (fp16_score_bits<<16) | off16；
// segcnt[R, NSEG] int32：每段实际候选数（kernel 内 clamp 到 CAP）；空槽不读、无需 fill。
// kernel 内对有效槽做 in-block radix K-min，并解码 score=packed>>16、id=seg*BLK+off16。
//   out_val[R,K] __half；out_idx[R,K] int32（最终 corpus id）。
void launch_flash_topk_select_packed_fp16(
    const int32_t* buf_pack, int R, int CAP, int NSEG, int BLK, int M, int K,
    const int32_t* segcnt, __half* out_val, int* out_idx, cudaStream_t stream);

// 直通拷贝：当 K >= N 时无需做 topk，直接把每行 N 个元素原样写入 out_val，
//   out_idx 填 0..N-1（每行）。out 形状视为 [B, N]，调用方需保证容量 >= B*N。
void launch_topk_identity_fp32(const float* in, int B, int N,
                               float* out_val, int* out_idx, cudaStream_t stream);

void launch_topk_identity_fp16(const __half* in, int B, int N,
                               __half* out_val, int* out_idx, cudaStream_t stream);

void launch_topk_identity_bf16(const __nv_bfloat16* in, int B, int N,
                               __nv_bfloat16* out_val, int* out_idx, cudaStream_t stream);

// 行内降序排序（就地）：对已选出的 [B, K] 个 (value, index) 按 value 从大到小排序，
//   index 随之重排。用于支持 sorted=true（tf.math.top_k 语义：返回值降序）。
//   实现：value -> 单调 uint 编码 -> cub 分段 radix 降序排序（payload 携带原位置）-> gather 重排。
//   workspace 由内部以 stream-ordered 方式分配/释放。
void launch_topk_sort_desc_fp32(float* out_val, int* out_idx,
                                int B, int K, cudaStream_t stream);

void launch_topk_sort_desc_fp16(__half* out_val, int* out_idx,
                                int B, int K, cudaStream_t stream);

void launch_topk_sort_desc_bf16(__nv_bfloat16* out_val, int* out_idx,
                                int B, int K, cudaStream_t stream);

// sort 的调用方预分配 workspace 版本。
size_t topk_sort_desc_workspace_bytes_fp32(int B, int K);
size_t topk_sort_desc_workspace_bytes_fp16(int B, int K);
size_t topk_sort_desc_workspace_bytes_bf16(int B, int K);

void launch_topk_sort_desc_fp32_ws(float* out_val, int* out_idx,
                                   int B, int K, void* ws, cudaStream_t stream);
void launch_topk_sort_desc_fp16_ws(__half* out_val, int* out_idx,
                                   int B, int K, void* ws, cudaStream_t stream);
void launch_topk_sort_desc_bf16_ws(__nv_bfloat16* out_val, int* out_idx,
                                   int B, int K, void* ws, cudaStream_t stream);
