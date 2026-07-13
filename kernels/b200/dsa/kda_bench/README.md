# LiteTopK B200 DSA — 测试与扫参方法论

本目录是 GLM-5.2 vLLM prefill 稀疏 top-k indexer(LiteTopK v3 内核)的
基准与实验工作区。**所有实验按本文档执行**;结论(含被否决的)记入
`candidates.jsonl`,状态记入 agent memory(litetopk-b200-dsa-setup.md)。

## 红线与纪律

1. **Recall = 100.00% 是红线**。99.99% 即失败。每个实验 cell 必须带全量
   recall 校验(对照官方 dense+topk 的精确结果)。
2. **recall 必须由构造保证,不允许"概率安全"**。两种合法形态:
   (a) 精确子集界(样本/探针的第 K 大);(b) **预测门限 + 每行精确性证书 +
   选择性修复**(CERT 模式,2026-07-12 起默认)——行的精确性由"预测门限下
   的真实计数 ≥ K 且未溢出"构造性证明,未获证行以精确界重扫,一轮收敛。
   裸预测门限(无证书)的性能数字不可作为结论。
3. **串行基准,空闲 GPU**。跑前 `nvidia-smi` 确认无邻居进程(历史事故:
   17GB 邻居使基线膨胀 20%)。默认 GPU 3。CUDA events 计时,≥4 warmup +
   12-20 次取均值;首次调用有懒加载 cubin(10ms+),必须落在 warmup。
4. **负收益即回退**。改动前备份到 scratchpad(`pre_*.{cuh,cu}.bak`);
   平(噪声内)的改动只有在"严格更少指令/消除隐患"时才保留。
5. 亚毫秒差异(<0.5ms)单次运行不可信,需交错 A/B 取中位数
   (7 轮交错模式)或复跑两次一致。
6. **拷贝核算:不许假设输入零拷贝**。生产输入一律是 paged KV cache,
   工作区必须 gather。bench 的失真是双向的:prefix 的样本切片零拷贝
   (生产=已 gather 工作区的切片,同样免费,但整体 gather 是共同成本)、
   strided 在 bench 里付一笔生产不付的压实拷贝(容器按保留页表直接
   gather 出压实工作区)。实测各类 gather 均为 0.02-0.1ms 量级
   (`tgather.py`),比模式间差异低两个数量级——结论不受影响,但涉及
   分段边界(400K/640K/900K)的决策以容器 E2E 为最终仲裁。

## 环境

- 容器 `vllm-prefill`,Python `/opt/vllm-venv/bin/python`。
- 数据:`/data/dsa_caches/glm5_{256k,512k,768k,1m}_realtext_chunk8192.safetensors`
  (真实文本,K=2048,Q 最大 8192)。
- 内核源:`/opt/simtopk_src/b200/dsa/{dsa_marsco_v3.cu, sm100_dsa_marsco_v3.cuh}`;
  模块:`/opt/simtopk_repro/glm5_prefill/litetopk_vllm/litetopk_indexer.py`。
- 构建缓存 `/root/.cache/litetopk_v3_build*`(.cuh 改动会被 ninja depfile 捕获自动重编)。

## 本目录的基准:bench_q8192.py

主记分板(唯一保留的工具):Q=8192 合并块形状,四规模(256K/512K/768K/1M)
对照官方 mqa_logits+top_k_per_row,真实 KV,auto 全流程 + 全量 recall 校验。

```bash
# 容器 vllm-prefill 内(模块路径见脚本顶部 sys.path)
CUDA_VISIBLE_DEVICES=3 python bench_q8192.py
```

实现要点:
- **参考 topk 落盘复用**:`refcache_{tag}_q{Q}k{K}.pt`,首次算一次
  (分块官方 dense+topk),后续直接加载。
- 计时用 CUDA events,≥4 warmup + 12-20 次取均值;首次调用有懒加载 cubin
  (10ms+),必须落在 warmup。
- 首过 recall 用模块目录的 `verify_module_realshape.py`(小 Q 四形状)。

> 本记分板背后曾有一套完整的分解/扫参/NCU 工具链(stage 分解、gate-vs-emit
> 密度分解、oracle 门限上限、交错 A/B 中位数、stride 扫参、NCU 最小复现),
> 用于得出下方结论;复线只需 bench_q8192.py,那些诊断工具不在本包内。

## 诚实基线(2026-07-12,证书制度 + 分段采样策略)

Q=8192,auto 全流程(`bench_q8192.py`,复跑×2 一致):

| | 256K | 512K | 768K | 1M |
|---|---|---|---|---|
| ours (ms) | 12.61-12.70 | 23.64-23.66 | 33.93-34.06 | 43.53-43.75 |
| vs official | 1.05x | 1.13x | 1.18-1.20x | 1.23x |

复现要点(2026-07-13 核实):此表需**两个旋钮**才能在 1M 拿到 1.23-1.24x
——(1) **GATE4 bucket-gate 内核**(`VLLM_LITETOPK_XFLAGS=DSA_BUCKET_GATE4=1`,
hot-start 部署用的那份;bench_q8192 已设为默认),(2) **强制 prefix**
(`try_chunk(..., strided_plan=False)`,即图表/sweep 口径,绕开 auto 的
deferred-probe/exploration ~1ms 开销)。各省 ~0.9-1.0ms,叠加把 1M 从
默认 gate+auto 的 45.4ms/1.18x 拉到 GATE4+prefix 的 43.6ms/**1.24x**。
256K/512K/768K 在 GATE4 下 auto 即达标(12.59/1.07x、23.57/1.14x、33.89/1.18x)。

门限制度升级(2026-07-12):**预测门限 + 每行精确性证书 + 选择性修复**
——扫描跑预测轨迹;行的精确性由"预测门限下的真实计数 ≥ K"构造性证明,
未获证行由精确子集界重扫(一轮收敛)。recall 恒 100%(margin 1.05 强制
修复测试通过),预测速度照拿。策略:400-900K 走证书 strided(64K 探针),
其余 prefix-64K。

AUTO_XK=12(auto 降级为纯 CAP 溢出保护,不做性能翻转)。

瘦身更新(2026-07-12 深夜):默认 `PREP_TILE=2048`(sample GEMM+seed_prep
行分块,slog 2.1GB→512MB,代价 +0.2-0.3ms/调用);当前默认态复跑×2 =
12.88/23.91/34.25/45.0-45.3(1.03-1.04/1.11-1.12/1.17-1.18/1.18-1.20x),
`PREP_TILE=0` 复现记录数字。`COMPACT_REPAIR` 默认 **0**:cand 3.2GB→0.6GB
的软 cap+紧凑修复方案 kernel-bench 持平、margin-1.05 压测 recall 100%,
但每调用一次 host 同步击穿 vLLM CPU 前瞻,E2E 1M +21%(129.1→155.8s)
——bench 与生产动态分歧的第 3 例;内存优先部署可 env 开启。

历史参考:强制单模式(同为精确门限,绕开 auto 状态机):

| | 256K | 512K | 768K | 1M |
|---|---|---|---|---|
| prefix | 12.59 | 28.24 | 35.78 | 43.28 |
| strided(exact) | 13.95 | 27.46 | 36.70 | 45.17 |

注:上表为固定 SAMPLE=64K 的旧强制对比;分段样本策略(128K@512-768K)
已让 prefix 四规模全胜并消化了 auto 损耗(AUTO_XK=12 后探索翻转不再
发生)。历史"预测门限"数字(如 512K 22.96)一律作废,不做比较基准。

## 全模型 E2E(2026-07-12,78L GLM-5.2-FP8,TP8+EP,8×B200,chunk 8192)

prefix caching 关(官方配方自己的基准指引);best-of-3;
run_prefill.py 环境钩子:MTP/GPU_UTIL/KVDTYPE/LOGITS_MB(via env)。

无 MTP,util 0.90(prefill wall 秒):

| | 512K | 768K | 1M |
|---|---|---|---|
| 官方默认(B=512MB) | 55.2 | 122.3 | 236.3 |
| 官方调优(B=4GB) | 48.79 | 88.89 | 140.28 |
| ours 混合(MIN_S=262144, MERGE_CAP=49152) | **46.66** | **81.82** | **124.41** |

官方完整配方(MTP 开)与 1M 补救阶梯:

| 配置 | 512K | 768K | 1M |
|---|---|---|---|
| 官方全默认(util 0.90, B=512MB) | 56.88 | 126.73 | **init 拒绝**(需 54.62 > 有 52.96 GB) |
| 官方 util 0.95 | — | — | 245.57 |
| 官方 util 0.95 + B=1GB | — | — | 167.29 |
| 官方 util 0.95 + B=2GB | — | — | 153.30 |
| 官方 util 0.95 + B=3GB(0.95 可行域顶点) | — | — | 147.46 |
| 官方 util 0.95 + B=4GB | — | — | 运行期 OOM(4GB vs 3.63GB 空闲) |
| 官方 + expandable_segments | — | — | 崩溃(VMM 与 TP8 custom allreduce IPC 不兼容) |
| 官方 util 0.94 + B=4GB(甜点区) | — | — | **145.17(官方 1M 天花板)** |
| ours 混合 + MTP(util 0.92, B=1GB) | — | — | **129.13**(+12.4% vs 官方天花板) |

ours 内存-时间 Pareto(1M+MTP,免同步紧凑修复,召回恒 100%):

| SCAN_CAP | 瞬态 | 时间 | util 上限(池) |
|---|---|---|---|
| 49152 | 3.2GB | 129.13s | 0.92(56.5GB) |
| 20480 | 1.25GB | 133.2-133.3s | **0.95(57.9GB,刻池法)** |
| 8192 | 0.5GB | ~157s | 0.95 |

HOTONLY 融合路径(2026-07-12,QSPLIT=4096 + 热样本自携带,零证书/修复/压实,
按层 carry,cap 32768):**131.8-132.5s / ~1.07GB**,零溢出=构造性召回。
HOTSAMPLE 扫参(4096/6144/8192):时间 132.54/132.06/131.76(更大样本微赚),
候选峰值 19.3K/18.2K/20.0K(结构性,不随样本降 → cap 地板 ~30K;
再压内存需"仅溢出行修复"路线)。

刻池法:申报 `LOGITS_MB=4096` 刻出池外空闲 + 容器补丁
`VLLM_LITETOPK_RUNTIME_LOGITS_MB=1024` 钳住官方路径实际用量。
曲线上每点都支配官方最优(56.1GB/145.17s)。

要点:MTP 固定占用 ~5.2GB/卡 + 0.69GB 草稿 KV/1M 序列,把官方全默认
挤出 1M 可行域(vLLM 报最大可服务 1,016,640);补救阶梯每一步都出自
官方文档/报错原文;MTP prefill 税 +3.6-3.9% 方法无关;预算伤害
Q_sub=B/(4S) 专门打击最长请求。生产 prefix caching 默认开 = 池内
每 GB 是有市价的缓存容量(DeepSeek 公开生产命中率 56.3% 可引用)。
