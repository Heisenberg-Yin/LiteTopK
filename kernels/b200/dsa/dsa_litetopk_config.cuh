#pragma once

// Fixed B200 LiteTopK production configuration.
//
// Keep the CUDA implementation self-contained: command-line -D flags and the
// Python JIT loader are not required to select the supported kernel path.
// Undefining first also prevents an external build from silently selecting a
// different experimental variant.

#undef DSA_BUCKET_GATE4
#define DSA_BUCKET_GATE4 1

#undef DSA_CHUNKED_EMIT
#define DSA_CHUNKED_EMIT 1

#undef DSA_SPARSE_REFRESH
#define DSA_SPARSE_REFRESH 1

#undef DSA_CHUNKED_FLUSH_DIRECT
#define DSA_CHUNKED_FLUSH_DIRECT 1

#undef DSA_CHUNKED_FLUSH_DIRECT_PACKEDPREFIX
#define DSA_CHUNKED_FLUSH_DIRECT_PACKEDPREFIX 1

#undef DSA_INPLACE_BOUNDARY_SELECT
#define DSA_INPLACE_BOUNDARY_SELECT 1

#undef DSA_SPARSE_REFRESH_REDUX_SKIP
#define DSA_SPARSE_REFRESH_REDUX_SKIP 1

#undef DSA_SPARSE_REFRESH_ADAPTIVE_SLEEP
#define DSA_SPARSE_REFRESH_ADAPTIVE_SLEEP 1

#undef DSA_SPARSE_REFRESH_ZERO_BASE
#define DSA_SPARSE_REFRESH_ZERO_BASE 1

#undef DSA_CANDIDATE_U16
#define DSA_CANDIDATE_U16 1

#undef DSA_CANDIDATE_FP16_NUMERIC
#define DSA_CANDIDATE_FP16_NUMERIC 1

#undef DSA_CANDIDATE_FP16_LOCAL32
#define DSA_CANDIDATE_FP16_LOCAL32 1

#undef DSA_EMIT_CHUNK_BLOCKS
#define DSA_EMIT_CHUNK_BLOCKS 256

#undef DSA_EMIT_LANE_SLOTS
#define DSA_EMIT_LANE_SLOTS 18

// Fixed refresh cadence for the supported sparse-threshold path.
#undef DSA_REFRESH_STRIDE
#define DSA_REFRESH_STRIDE 2

#undef DSA_GATE_STRIDE
#define DSA_GATE_STRIDE 64
