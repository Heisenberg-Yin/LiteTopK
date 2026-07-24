// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

#include "litedsa.cu"
#include "tvm_ffi_utils.h"

void litedsa_union_qm(TensorView topk_indices, TensorView u_phys, TensorView counts,
                      TensorView memb_qm, TensorView req_pg, TensorView block_table,
                      int64_t block_size, int64_t seq_space);

void litedsa_masked_mla_fp8(TensorView q, TensorView kv, TensorView indices, double sm_scale,
                            double out_scale, TensorView topk_length, TensorView memb_qm,
                            int64_t h_per_q, TensorView out, TensorView max_logits,
                            TensorView lse);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(union_qm, litedsa_union_qm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(masked_mla_fp8, litedsa_masked_mla_fp8);
