# DSv4 packed attention: removing the two host-side copies

Status: **not implemented**. The working kernel remains in the vendored
source files; use version control rather than adjacent `.works*` backup files
when comparing with the pre-zero-copy implementation.

## What we are removing

`vllm-v026/vllm/model_executor/layers/dsv4_packed_attn.py` currently does

```python
qp.view(ng, G, h, 512).copy_(q[:, :h].view(ng, G, h, 512))   # ~0.086 ms/layer at TP8
...
out[:, :h].copy_(o_packed.view(num_tokens, h, 512))          # ~0.03 ms/layer at TP8
```

because the model hands us q as `[s_q, 64, 512]` with only the first `h` head
slots populated (`h` = 64 / TP), while a packed tile needs 128 rows whose
r-th row is `(token g*G + r//h, head r%h)`. That address needs **two** strides
(token 64*512, head 512), and the Q tensor map has one stride per mode.

Together ~12% of the packed path at TP8, and proportionally more at lower TP
(the copies scale with `h`, so TP2 moves 4x what TP8 does).

## Correction to an earlier assumption

`shape_Q` is **already 3-mode** — `(h_q, d_qk, s_q/q_group_div)` with stride
`(stride_q_h_q, 1, stride_q_s_q)` (litedsa_attention_sm100_dsv4.cuh:2421).
"Make it 3D" was never the fix. Reading packed rows out of the padded tensor
needs **four** modes:

```text
(head = h, token_in_group = G, dim = 512, group = ng)
stride ( 512,   64*512,           1,        G*64*512 )
```

## The decisive constraint: `h` is a runtime value

`h` is 8 / 16 / 32 for TP8 / TP4 / TP2. Both candidate fixes need the Q smem
tile shape to depend on `h`, and a `cute::make_tma_copy` object is
compile-time-typed on its smem layout. So **either fix requires templating the
kernel (or at least its Q prologue) on `h`**, with host-side dispatch over
{8, 16, 32}.

That kills the reason to prefer B (the per-token copy loop) over A: B was
attractive only as "the small edit", and once templating is required it is no
smaller than A, while being strictly worse at runtime (G TMA issues instead of
1, and a row-offset slice into a swizzled smem tile). **Go straight to A.**

## Plan A (single 4-mode map), in order

1. `litedsa_attention_sm100_dsv4.cuh` — template `KernelTemplate` (or just
   `run_fwd_phase1_kernel`) on `int H_REAL`; assert `H_REAL * G == B_H`.
2. `SparseAttnFwdParams` (same file, ~line 1305): add
   `int q_real_heads = 0;` and output strides
   `int stride_out_token = 0, stride_out_head = 0;`
   (0 = legacy contiguous packed layout, i.e. today's behaviour).
3. `shape_Q` (~line 2421): when `q_real_heads != 0`, build the 4-mode tensor
   `(H_REAL, G, d_qk, ng)` with stride `(stride_q_h_q, stride_q_s_q, 1,
   G*stride_q_s_q)`; keep the 3-mode form otherwise.
4. Prologue (~line 1875): `flat_divide(..., Tile<Int<B_H/2>>{})(_, cta_idx, _)`
   splits the *head* mode across the two CTAs. With 4 modes the CTA split must
   happen on the **combined (head, token) extent**: CTA c takes tokens
   `[c*G/2, (c+1)*G/2)` with all `H_REAL` heads. Coalesce those two modes into
   one 64-row mode before the divide, or slice the token mode by `cta_idx`
   first and then take all heads.
5. Epilogue store: destination offset becomes
   `(g*G + r/H_REAL) * stride_out_token + (r % H_REAL) * stride_out_head`
   when `stride_out_token != 0`.
6. `litedsa_dsv4.cu`: relax `TVM_FFI_ICHECK(h_q == 128)` to accept a padded q
   (`q.size(1) == 64` with `h_per_q` real heads), pass `q_real_heads` and the
   two output strides, and dispatch on `h_per_q` to the templated kernel.
7. `dsv4_packed_attn.py`: drop both `copy_` calls; pass `q` and
   `out[: , :h]`'s base tensor straight through. Keep the copies behind an env
   flag for one release so a regression can be bisected.

## Validation order (do not skip)

1. Compile (`build_dsv4.sh`), and re-run `build_probe_dsv4_smem.sh` — smem is
   215024 B of a 232448 B cap, only 17 KB of headroom.
2. `bench_dsv4_kernel.py` against the dumps: must hold **lse max abs 3e-6 and
   out rel 0.0025**, and the union blow-up line must stay 1.002x.
3. Only then E2E: `VLLM_DSV4_PACKED_CHECK=1` at 256K (expect worst rel
   ~0.0075), then the controlled TP sweep.

## Expected gain

TP8 ~12% of the packed path; TP4/TP2 more. On the measured controlled sweep
(1M, official flags, token-identical) the packed path currently gives
1.176x / 1.126x / 1.087x at TP8 / TP4 / TP2.
