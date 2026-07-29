// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI binding for grouped packed sparse prefill attention on SM100.
// litedsa_union_qm builds the union, membership bitmap, and physical indices.
// litedsa_masked_mla_fp8 invokes the vendored FlashMLA-derived FP8 kernel
// under csrc/vendor_fmla/.

#include <cuda_runtime.h>

#include <flashinfer/litedsa/litedsa_union.cuh>

#include "sm100/prefill/sparse/fwd/head128_fp8/phase1.cuh"

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {
constexpr float kLog2e = 1.4426950408889634f;
inline int num_sms_of(TensorView t) {
  int dev = t.device().device_id;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  return sms;
}
}  // namespace

void litedsa_union_qm(TensorView topk_indices, TensorView u_phys, TensorView counts,
                      TensorView memb_qm, TensorView req_pg, TensorView block_table,
                      int64_t block_size, int64_t seq_space) {
  using namespace flashinfer::litedsa;
  CHECK_DIM(2, topk_indices);
  TVM_FFI_ICHECK(u_phys.ndim() == 2 && memb_qm.ndim() == 3)
      << "u_phys [ng, cap], memb_qm [ng, G, cap/32]";
  const int T = static_cast<int>(topk_indices.size(0));
  const int k = static_cast<int>(topk_indices.size(1));
  const int ng = static_cast<int>(u_phys.size(0));
  const int cap = static_cast<int>(u_phys.size(1));
  const int G = static_cast<int>(memb_qm.size(1));
  TVM_FFI_ICHECK(ng > 0 && T == ng * G) << "T must equal ng * G";
  TVM_FFI_ICHECK(cap == G * k && cap % 128 == 0)
      << "cap must be G*k (constructive union bound), 128-aligned";
  TVM_FFI_ICHECK(G <= 16 && memb_qm.size(2) == cap / 32)
      << "memb_qm last dim must be cap/32, G <= 16";
  TVM_FFI_ICHECK((int64_t)G * (cap / 32) <= 16 * 1024)
      << "qm smem accumulator overflow (G*cap/32 > 16K words)";
  TVM_FFI_ICHECK(counts.size(0) == ng && req_pg.size(0) == ng)
      << "counts/req_pg must be [ng]";
  CHECK_DIM(2, block_table);
  const int s_words = static_cast<int>((seq_space + 31) / 32);
  TVM_FFI_ICHECK(s_words % (kLiteDSAUnionThreads * 4) == 0)
      << "seq_space must give 4-aligned word chunks";
  const int smem = litedsa_union_qm_smem_bytes(s_words);
  TVM_FFI_ICHECK(smem <= 224 * 1024) << "seq_space too large for smem bitmap";

  cudaSetDevice(topk_indices.device().device_id);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute((void*)litedsa_union_qm_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 224 * 1024);
    attr_set = true;
  }
  cudaStream_t stream = get_stream(topk_indices.device());
  litedsa_union_qm_kernel<<<ng, kLiteDSAUnionThreads, smem, stream>>>(
      static_cast<int32_t*>(topk_indices.data_ptr()), static_cast<int32_t*>(u_phys.data_ptr()),
      static_cast<int32_t*>(counts.data_ptr()), static_cast<uint32_t*>(memb_qm.data_ptr()),
      static_cast<int32_t*>(req_pg.data_ptr()), static_cast<int32_t*>(block_table.data_ptr()),
      static_cast<int>(block_table.size(1)), static_cast<int>(block_table.size(1)),
      static_cast<int>(block_size), k, G, cap, s_words);
}

void litedsa_masked_mla_fp8(TensorView q, TensorView kv, TensorView indices, double sm_scale,
                            double out_scale, TensorView topk_length, TensorView memb_qm,
                            int64_t h_per_q, TensorView out, TensorView max_logits,
                            TensorView lse) {
  using bf16 = cutlass::bfloat16_t;
  TVM_FFI_ICHECK(q.ndim() == 3 && kv.ndim() == 3 && indices.ndim() == 3)
      << "q [rows, 128, 576], kv [s_kv, 1, 576], indices [rows, 1, topk]";
  const int s_q = static_cast<int>(indices.size(0));
  const int s_kv = static_cast<int>(kv.size(0));
  const int h_q = static_cast<int>(q.size(1));
  const int h_kv = static_cast<int>(kv.size(1));
  const int d_qk = static_cast<int>(q.size(2));
  const int topk = static_cast<int>(indices.size(2));
  TVM_FFI_ICHECK(q.size(0) == s_q) << "q rows must match indices rows";
  TVM_FFI_ICHECK(h_q == 128) << "packed path requires h_q == 128";
  TVM_FFI_ICHECK(d_qk == 576 && kv.size(2) == 576) << "d_qk must be 576";
  TVM_FFI_ICHECK(topk % 128 == 0) << "topk must be a multiple of 128";
  TVM_FFI_ICHECK(h_per_q > 0 && 128 % h_per_q == 0 && 128 / h_per_q <= 16)
      << "h_per_q must divide 128 with <= 16 queries per row";
  TVM_FFI_ICHECK(topk_length.size(0) == s_q) << "topk_length must be [rows]";
  TVM_FFI_ICHECK(memb_qm.numel() == (int64_t)s_q * (128 / h_per_q) * (topk / 32))
      << "membership_qm must be [rows, G, topk/32]";

  cudaSetDevice(q.device().device_id);
  SparseAttnFwdParams params = {
      s_q,
      s_kv,
      h_q,
      h_kv,
      d_qk,
      512,
      topk,
      (float)sm_scale,
      (float)sm_scale * kLog2e,
      static_cast<bf16*>(q.data_ptr()),
      static_cast<bf16*>(kv.data_ptr()),
      static_cast<int32_t*>(indices.data_ptr()),
      nullptr,  // attn_sink
      static_cast<int32_t*>(topk_length.data_ptr()),
      static_cast<int>(q.stride(0)),
      static_cast<int>(q.stride(1)),
      static_cast<int>(kv.stride(0)),
      static_cast<int>(kv.stride(1)),
      static_cast<int>(indices.stride(0)),
      static_cast<int>(indices.stride(1)),
      static_cast<bf16*>(out.data_ptr()),
      static_cast<float*>(max_logits.data_ptr()),
      static_cast<float*>(lse.data_ptr()),
      num_sms_of(q),
      get_stream(q.device())};
  params.membership = nullptr;
  params.h_per_q = static_cast<int>(h_per_q);
  params.out_scale = (float)out_scale;
  params.topk_length_per_q = nullptr;
  params.q_group_div = 1;
  params.membership_qm = static_cast<uint32_t*>(memb_qm.data_ptr());

  sm100::fwd::head128_fp8::run_fwd_phase1_kernel<576>(params);
}
