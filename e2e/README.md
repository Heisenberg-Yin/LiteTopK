# GLM-5.2 / DSV4 四臂端到端测试

本目录在同一个 `/data01/home/ziqi.yin/vllm-v026` 源码树上测试四种配置，
分别隔离 LiteTopK indexer、LiteDSA attention 及两者组合的收益。唯一入口是
`run_e2e.sh`。

## 四臂定义

| 模型 | arm | `VLLM_DSA_MODE` | `VLLM_LITETOPK` | packed attention |
|---|---|---|---:|---:|
| GLM-5.2 | `raw` | `raw` | 0 | 0 |
| GLM-5.2 | `litetopk` | `litetopk` | 1 | 0 |
| GLM-5.2 | `litedsa` | `litedsa` | 0 | 0 |
| GLM-5.2 | `combo` | `litedsa` | 1 | 0 |
| DSV4 | `raw` | `raw` | 0 | 0 |
| DSV4 | `litetopk` | `litetopk` | 1 | 0 |
| DSV4 | `litedsa` | `raw` | 0 | 1 |
| DSV4 | `combo` | `litetopk` | 1 | 1 |

launcher 根据模型和 arm 强制派生这三个开关，外部同名环境变量不能污染基线。
GLM 的 attention-only 臂使用 FP8 grouped LiteDSA；DSV4 的 attention-only 臂
使用 `VLLM_DSV4_PACKED_ATTN=1` 的 BF16 packed C128A attention。

## 文件

| 文件 | 用途 |
|---|---|
| `run_e2e.sh` | 唯一入口；内嵌 prefill runner 和四臂 comparator，检查实际 kernel 命中并生成 summary |

## 默认资格口径

两种模型均默认 `async_scheduling=true`、1,048,512 tokens、chunk 8192、
FP8 attention KV、GPU util 0.90。

- GLM-5.2：TP8+EP8、MTP=5、`FLASHINFER_MLA_SPARSE`；production gate
  65,536，dense-select 40,960..65,536，固定的 warm-started ring
  refresh/daemon（编译期 2048 ns pacing），merge cap 65,536，probe every 1，
  overflow watermark 49,152。ring policy 不再提供运行时开关。
- DSV4 Flash：TP4+EP4、MTP=2、`deep_gemm_mega_moe`、FP4 indexer cache、
  `FLASHMLA_SPARSE_DSV4`；FP4 production gate 65,536，dense-select 上限
  262,144，merge cap 65,536，probe every 8，overflow watermark 65,536。

## 2026-08-09 正式 1M 结果

主口径为一次 warmup 后 3 次 trial 的 median；两种模型均开启
`async_scheduling` 和 expert parallel。

| 模型 | raw (s) | LiteTopK (s) | LiteDSA (s) | combo (s) | raw/TopK | raw/DSA | raw/combo | TopK/combo | DSA/combo | interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-5.2 TP8+EP8 | 152.331 | 118.668 | 143.955 | 109.343 | 1.284x | 1.058x | **1.393x** | 1.085x | 1.317x | 1.026 |
| DSV4 TP4+EP4 | 60.007 | 55.779 | 52.397 | 48.185 | 1.076x | 1.145x | **1.245x** | 1.158x | 1.087x | 1.011 |

GLM 结果位于
`results/formal-20260809-glm52-1m-ring64-four-arm/`。该次运行使用现已固化的
warm-started ring daemon、编译期 2048 ns pacing、merge cap 65,536 和
watermark 49,152；LiteTopK-only
与 combo 的最大候选数分别为 38,914 和 39,580。8 个 rank 的 HOT/LiteDSA
dispatch marker 均符合 arm 定义，所有 warmup/trial 输出 token 都是 279，且无
overflow 或非零 status。该次运行的 `13ae986f6884` 源码在该口径下复现了历史
“约 1.395x”；109.343 秒是 LiteTopK+LiteDSA combo，不是 LiteTopK-only。

移除运行时 ring 参数并将策略固化到 kernel 后，使用 source ID
`f75dfed60674` 直接复线 combo，三次 trial 为 109.738、109.763、109.769 秒，
median 为 **109.763 秒**。结果位于
`results/formal-20260809-glm52-1m-fixed-ring-combo/`；8 个 rank 均命中
LiteTopK 和 LiteDSA，最大候选数 36,377，所有 watermark/status/overflow
门禁通过，warmup 和 trial token 均为 279。

DSV4 结果位于 `results/formal-20260809-dsv4-1m-four-arm/`。四臂输出 token
一致，FP4 LiteTopK 与 BF16 packed attention 的实际 dispatch marker、候选上限
和 overflow 门禁均通过。该结果是精确 artifact-qualified 的 mixed-toolchain
运行：Torch/LiteTopK 使用 CUDA 13，而 packed attention SO（SHA256
`b27ece4210d33392ae0b47eb05a7c7bdabc30f0a397b8b8c960f986d5673936c`）
链接 `libcudart.so.12`；重编 packed SO 后必须重新资格测试。

固化前曾误关 ring：候选数升到约 161K，64K cap 会正确 fail closed；把 cap
临时扩大到 196K 得到的 combo 110.087 秒不是现役配方。相关运行时开关及
非现役 kernel 分支现已删除。

## 运行

```bash
cd /data01/home/ziqi.yin/litetopk/e2e

MODEL_FAMILY=glm5.2 REPEATS=3 WARMUPS=1 \
VLLM_LITEDSA_SO=/data01/home/ziqi.yin/litetopk/kernels/b200/dsa/.codex_variants/litedsa_cuda13_b264c21c9ce1_cuda13.so \
./run_e2e.sh
MODEL_FAMILY=dsv4 REPEATS=3 WARMUPS=1 ./run_e2e.sh
```

只跑部分臂可设置空格或逗号分隔的 `ARMS`：

```bash
MODEL_FAMILY=glm5.2 ARMS="raw litedsa" ./run_e2e.sh
MODEL_FAMILY=dsv4 ARMS=litedsa VLLM_DSV4_PACKED_CHECK=1 ./run_e2e.sh
```

四臂完整运行会写入：

```text
results/<UTC时间>-<family>/
├── raw.{json,log}
├── litetopk.{json,log}
├── litedsa.{json,log}
├── combo.{json,log}
├── summary.json
└── summary.txt
```

主口径是 median。summary 同时给出 `raw/litetopk`、`raw/litedsa`、
`raw/combo`、`litetopk/combo`、`litedsa/combo`，以及衡量两项优化是否独立
叠加的 interaction factor。

## Fail-closed 验证

`run_e2e.sh` 内嵌的 runner 会校验 vLLM checkout、CUDA source ID、integration
SHA、加载的 LiteTopK extension 和 attention SO。attention 臂还必须在 worker
日志中出现实际成功 dispatch 后生成的一次性 marker：

- GLM：`LITEDSA_KERNEL_EXECUTED`
- DSV4：`DSV4_PACKED_KERNEL_EXECUTED`

marker 缺失、出现在错误臂、四臂配置或源码不同、candidate ABI 不符、生成 token
不同都会使运行失败，不会静默接受 fallback。DSV4 的 correctness smoke 可单独设置
`VLLM_DSV4_PACKED_CHECK=1`；正式性能四臂必须保持该值为 0。

可用 `MODEL`、`PARQUET`、`LENGTHS`、`TP`、`DEVICES`、`REPEATS`、
`GPU_UTIL`、`RESULT_DIR` 等覆盖机器相关参数。需要固定预编译 LiteTopK extension
时，必须同时设置 `VLLM_LITETOPK_SO` 和 `VLLM_LITETOPK_SO_SHA256`；DSV4 packed
attention 可用 `VLLM_DSV4_PACKED_SO` 指定精确构建。

上例 GLM CUDA 13 LiteDSA artifact 的 SHA256 是
`320c29a13df2e2c65c8500587a72c44a02c28649b38f53ce26367f6e4053addb`。
固定策略 combo 复线实际加载的 LiteTopK artifact SHA256 是
`5a54b3c10f749d32c4a123bf4bfd948ad305964aca5a9280cadf65f141abd7ec`。
当前 installed `_C_stable_libtorch` 不包含 LiteDSA ops，因此完整 GLM 四臂运行
必须显式提供 `VLLM_LITEDSA_SO`；runner 会记录并核验实际加载文件的 SHA。
