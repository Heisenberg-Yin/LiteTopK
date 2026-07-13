# B200 DSA LiteTopK Kernel 优化详解（2026-07-09）

本文详细记录对 `sm100_dsa_marsco.cuh` / `dsa_marsco.cu` 的一轮性能优化：动机、
实现、正确性论证与逐项实测收益。测量协议：本机 `simtopk` 容器内、真实 GLM-5
chunked caches（`Q=chunk`、`K=2048`、`NB=256`、`SAMPLE=65536`、`REFRESH=64`）；
**benchmark 逐个串行执行**（跑前用 `nvidia-smi` 确认整卡空闲、无并发任务），
每个 cell = warmup 5 次 + CUDA event 计时 20 次取平均。串行与双卡并行对照的
最大偏差 0.41%（这些 kernel 为纯单卡负载，跨卡干扰可忽略，但正式数据一律
取串行值）。

---

## 背景：这个 kernel 在干什么

GLM-5 的 DSA（DeepSeek-Sparse-Attention 式稀疏注意力）在做真正的 attention
之前，先用一个轻量 **indexer** 给每个 query 从整条 KV cache（最长 1M token）
里挑出得分最高的 **top-K=2048** 个位置，之后的 attention 只算这 2048 个。
挑选的依据是一个 fp8 的 **ReLU-MQA 打分**：

```
score(q, k) = Σ_h  w[q,h] · max(0, dot(q[h], k)) · kv_scale[k]     （h = 32 个头）
```

即每个 (query, kv) 对都要做 32 头 × 128 维的 fp8 点积、过 ReLU、按头加权求和。
prefill 一个 chunk（Q = 1024…8192 行）× 1M KV，就是一个 [Q, S] 的巨型打分
矩阵 + 每行 top-K。

**痛点**：标准做法（官方 DeepGEMM `fp8_mqa_logits` 或我们的 dense 基线）要把
整个 [Q, S] 分数矩阵写到 HBM（1M/8192 时 = 34 GB），再让 `torch.topk` 把这
34 GB 读回来——一整个多余的 HBM 往返，而其中 99.8% 的分数注定被丢弃。

**LiteTopK 的思路**：top-K 只关心"够不够大"，不需要完整矩阵。
1. 先在一个 64K 的采样前缀上算分，得到每行第 K 大分数的近似 → 换算成一个
   per-row 阈值（离散化到 256 个桶，`th_bucket`）；
2. 主 scan 时每算出一个分数就地与阈值比较，**只把过线的候选**（每行仅几千个）
   写进紧凑缓冲区，顺带维护每行的桶直方图 `bcount`；
3. 空闲的 spare warp 作为后台守护线程，边扫边用直方图**单调收紧**阈值
   （阈值只会变严不会放松，所以读到旧值最多多收几个候选、绝不漏真 top-K
   —— recall 恒为 100%）；
4. 扫完后一个小的 radix-select kernel 从候选缓冲里精确选出 top-K。

这样全量 [Q,S] 矩阵从头到尾不落地，省掉 34 GB 写 + 34 GB 读。

**硬件路径（B200 / SM100）**：kernel 采用 warp 特化流水——1 个 TMA warp 负责
把 KV 分块异步搬进 shared memory，1 个 UMMA warp 负责发射 `tcgen05.mma`
（Blackwell 的新张量核指令，累加结果存在片上 **Tensor Memory / TMEM** 里，
不占寄存器），8 个 math warp 把 TMEM 累加器读回寄存器、做 ReLU 加权归约，
然后执行上面的"门控 + 稀疏写出" epilogue；2 个 spare warp 跑阈值守护。
KV 按 256 行分块流过这条流水线，一个 CTA 负责 4 个 query 行 × 一段 KV 区间
（KV-split 让所有 SM 并行扫同一批 query）。

**benchmark 口径**（`figure_b200_realchunk.json` / `figures/dsa.pdf`）：
`DSA` = dense 打分 kernel + `torch.topk`（理想化基线，与我们共享同一 kernel、
仅编译成全量直写）；`LiteTopK` = 采样定阈 + 融合稀疏 scan + radix select 的
端到端时间。本轮优化前 LiteTopK 比基线快 1.45–2.17x，优化后 1.57–2.71x。

---

## 0. 方法论：先定位瓶颈，再动手

### 0.1 端到端阶段分解（`profile_breakdown.py`）

LiteTopK E2E = `O_prep`（64K 采样前缀 dense 打分 + seed radix select + 阈值
计算）→ `scan`（融合稀疏打分 kernel `mqa_logits_dsa_marsco`）→ `select`
（`compact_topk_min_thr_marsco` 边界 radix）。优化前：

| cell | O_prep | scan | select |
|---|---:|---:|---:|
| 256k/4096 | 4.16 ms (37%) | 6.97 ms (61%) | 0.19 ms (2%) |
| 512k/8192 | 7.85 ms (18%) | 35.16 ms (80%) | 0.96 ms (2%) |
| 1m/2048   | 2.20 ms (13%) | 15.30 ms (87%) | 0.09 ms (0.5%) |
| 1m/8192   | 7.84 ms (11%) | 60.05 ms (88%) | 0.28 ms (0.4%) |

**结论：融合 scan kernel 主导（61–88%），select 可忽略。** 优化对象 = scan。

### 0.2 scan 变体对照（`bench_scan_variants.py`）

四个同源构建隔离出各部分成本（1 CTA = 1 q-block + KV-split，完全相同的
tile/调度）：

- `dense`：`-DDENSE_WRITE`，同一打分循环 + 全量 [Q,S] 直写（基线 kernel）；
- `sparse`：默认融合稀疏路径；
- `sparse-nr`：`refresh_every=0`，无 bcount 原子、门控不收紧；
- `null-epi`：`-DDSA_NULL_EPILOGUE`，打分 + 归约后直接丢弃（防 DCE 的假消费），
  即 **TMA/UMMA/TMEM+归约的纯下限**。

优化前基线（ms）：

| 变体 | 256k/4096 | 1m/2048 | 1m/8192 |
|---|---:|---:|---:|
| null-epi（下限） | 3.95 | 10.20 | **41.55** |
| dense | 5.83 | 11.95 | 48.82 |
| sparse | 7.09 | 15.33 | **60.03** |
| sparse-nr | 6.67 | 15.18 | 59.84 |

三个关键判断（以 1m/8192 为例）：

1. **sparse epilogue 的"常驻"开销 = 60.0 − 41.6 = 18.5 ms**，是 dense 直写
   开销（48.8 − 41.6 = 7.3 ms，其中 ~4 ms 是 34GB 写带宽）的 **2.5 倍**——
   尽管每行实际只发射约 700–2000 个候选。说明贵的不是候选写出本身，而是
   **每个 KV block 都要执行的门控/表决/分支代码**。
2. **bcount 原子与后台 refresh 不是瓶颈**：关掉 refresh 只省 0.2 ms。
3. ptxas 确认 **168 寄存器、0 spill**——排除寄存器溢出假设。

由此确定优化顺序：先砍 epilogue 常驻开销（§1–§2），再砍归约指令数（§3），
最后抬打分下限（§4–§5）。

---

## 1. 批量 vote emit（收益最大：−7.4 ms @1m/8192）

### 原实现的问题

每个 KV block 的每行 `i ∈ [0, BLOCK_Q)`，warp 立即做一次 ballot 并进入
分支：

```cpp
const unsigned m = __ballot_sync(FULL, g);
if (m != 0) {
    // 读 warpq_count → shfl 广播 → 队列满则 flush(atomicAdd+STG.CS)
    // → 通过者按 ballot 前缀写 warp 队列 → bcount atomicAdd → __syncwarp
}
```

即使整个 warp 一个候选都没有（大多数情况），每 block 也要付出
**4 次 `VOTE.SYNC` + 4 次分支重汇聚（SASS 层的 BSSY/BRA/BSYNC 序列）**。
在 1M 序列上每个 CTA 要扫约 3840 个 KV block，这套固定开销被放大了几千倍。

通过率数学（1m/8192）：每行发射 ≈744 / 983K 列 → 单列通过率 p ≈ 0.076%；
warp（32 列）单行命中率 1−(1−p)³² ≈ 2.4%；4 行任一命中 ≈ **9.2%**。
即 **~91% 的 KV block 整个 warp 无任何候选**，却在白付 emit 机器的钱。
（256k 时门控相对更松，快路径占比降到 ~26–70%，收益相应变小但仍为正。）

### 新实现（`sm100_dsa_marsco.cuh:548` 起）

归约循环里只做谓词求值并打包到每 lane 的位掩码，不做任何 warp 集体操作：

```cpp
uint32_t pass_bits = 0;           // bit i = 本 lane 在行 i 是否通过
float    x_row[BLOCK_Q];          // 暂存 x = -score，emit 阶段复用
...（每行归约得到 v 后）
const uint32_t col = kv_offset + v_offset;
const float x = -v;
x_row[i] = x;
const bool g = seq_k_start[i] <= col and col < seq_k_end[i] and x < vth_reg[i];
pass_bits |= g ? (1u << i) : 0u;
```

循环结束后，**一次 warp 级表决**决定是否进入 emit 机器
（`sm100_dsa_marsco.cuh:607`）：

```cpp
if (__any_sync(0xffffffffu, pass_bits)) {      // warp 一致的谓词
    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
        const bool g = (pass_bits >> i) & 1u;
        const unsigned m = __ballot_sync(FULL, g);
        if (m != 0) { /* 原 emit 逻辑逐字保留：队列/flush/bcount */ }
    }
}
```

快路径成本从「4×(VOTE+分支重汇聚)」降为「4 次谓词求值 + 1 次 VOTE」。

### 正确性

- `__any_sync` 的返回值对全 warp 一致 → 整个 warp 同进同出外层分支，内部的
  `__ballot_sync`/`__shfl_sync`/`__syncwarp` участ全员参与，语义与原版完全一致；
- `m = ballot(pass_bits>>i & 1)` 与原来的 `m = ballot(g)` 等价；
- emit 体内代码逐字未动，候选值/索引/bcount 更新路径不变。

### 实测

固定其余配置（原 Q3/KV3 stage），1m/8192 sparse scan：54.13 → **46.67 ms**
（−13.8%）。emitted 候选数不变，recall 100%。

---

## 2. 门控 stride 重读 + fdiv 惰性重算（与 §3 合计 −5.9 ms @1m/8192）

### 原实现的问题

每个 KV block 开头，每个 math 线程都执行：

```cpp
#pragma unroll
for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
    gate_reg[i] = th_bucket[block_q_idx * BLOCK_Q + i];          // 4×LDG
    vth_reg[i]  = o_reg[i] + float(gate_reg[i] + 1) / inv_reg[i]; // 4×I2F+4×fdiv
}
```

- 4 次全局读（同地址，L2 命中，但仍占 LSU 发射槽并形成依赖链）；
- 4 次 `fdiv.rn.f32`——SASS 里展开为 `MUFU.RCP` + 2 轮 Newton `FFMA` + 修正，
  每次约 8–10 条指令，且 **每 block 重复计算一个几乎从不变化的值**。

而门控值本身的更新途径是 spare-warp refresh 守护线程，它 **每
`DSA_REFRESH_STRIDE=16` 个 KV block 才发布一次**（math 侧按 stride 公布
`kv_progress`，守护线程据此触发）。逐块重读纯属浪费。

### 新实现（`sm100_dsa_marsco.cuh:501` 起）

```cpp
// 初始化：gate_reg[i] = INT_MAX（保证首块必然重算 vth）
if ((kv_block_idx % DSA_GATE_STRIDE) == 0) {        // 默认 8
    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
        const int g = th_bucket[block_q_idx * BLOCK_Q + i];
        if (g != gate_reg[i]) {                     // 门控真变了才重算
            gate_reg[i] = g;
            vth_reg[i] = o_reg[i] + float(g + 1) / inv_reg[i];
        }
    }
}
```

全局读摊薄 8 倍；fdiv 只在门控值实际变化时执行（整个 scan 一行只收紧几次），
摊薄后 ≈ 0。

### 两个刻意的设计取舍

1. **为什么不用「预计算 1/inv，除法换乘法」？** `(g+1)*rcp(inv)` 与
   `(g+1)/inv` 的舍入方向可能不同；若 vth 被舍小，边界桶（bucket == th）的
   候选可能在 scan 侧被漏掉，而 select 侧仍按精确桶号索取它们 → recall 有
   跌破 100% 的风险。改成「变化才重算」保留了 **与原版 bit 级相同的除法
   表达式**，零数值风险。
2. **为什么读旧门控是安全的？** `th_bucket` 只会单调收紧（守护线程
   `if (found < th_bucket[row])` 才写，math 侧从不放松）。读到旧值 = 门控
   更宽 = 只会**多**发候选，绝不会漏掉真 top-K —— select 阶段照常裁剪。
   实测滞后代价：emitted/row 739 → 744（1m/8192，+0.7%），可忽略。

---

## 3. 权重 shared 读 float4 向量化

### 原实现的问题

ReLU-MQA 归约对每行 32 个 head 权重逐标量 `ld_shared`：

```cpp
auto b = make_float2(ld_shared(w + j), ld_shared(w + j + 1));  // 2×LDS.32
```

每线程每 KV block：4 行 × 32 头 = **128 次 `LDS.32`**。权重在整个 KV 循环内
不变（只依赖 q-block），却被每块重读一遍；放寄存器又放不下（128 个 float，
预算 208 已被 TMEM 回读的 128 个累加器占满）。

### 新实现（`sm100_dsa_marsco.cuh:565` 起）

```cpp
const float* wrow = smem_weights[q_stage_idx] + i * kNumHeads;
#pragma unroll
for (uint32_t j = 0; j < kNumHeads; j += 4) {
    const float4 w4 = ld_shared(reinterpret_cast<const float4*>(wrow + j));  // 1×LDS.128
    sum_0 = __ffma2_rn({fmaxf(accum[j],0),   fmaxf(accum[j+1],0)}, {w4.x, w4.y}, sum_0);
    sum_1 = __ffma2_rn({fmaxf(accum[j+2],0), fmaxf(accum[j+3],0)}, {w4.z, w4.w}, sum_1);
}
```

128 次 `LDS.32` → **32 次 `LDS.128`**（deep_gemm utils 自带 float4 重载）。

### 对齐与 bit 一致性

- 每行权重 32×4B = 128B，行基址 512B 对齐（`SMEM_WEIGHT_SIZE_PER_STAGE`
  静态断言）→ float4 访问合法；warp 内 32 lane 读同一地址 → smem 广播，无
  bank conflict；
- **累加顺序完全不变**（j%4∈{0,1}→sum_0，{2,3}→sum_1，j 升序；尾部
  `(sum_0.x+sum_0.y+sum_1.x+sum_1.y)*scale_kv` 原样）→ 分数与原版
  **bit 级一致**，任何依赖分数数值的门控/select 行为不受影响。
- 该循环 dense/sparse 两个构建共享 → **基线同样受益，对比依然公平**。

§2+§3 合计实测：1m/8192 sparse scan 60.03 → **54.13 ms**（−9.8%）；
1m/2048 15.33 → 13.83；256k/4096 7.09 → 6.68。

---

## 4. 流水线整理：Q stage 3→1，KV stage 3→6（−1.4 ms @1m/8192，dense −2.1 ms）

### 发现：3 个 Q stage 中 2 个是死的

本 kernel **没有持久化调度**：`blockIdx.x` 直接就是 q-block，每个 CTA 只在
启动时调用一次 `issue_tma_q(0, block_q_idx)`，`q_stage_idx` 恒为 0。三级
Q stage 是从 deep_gemm 持久化调度版本抄来的残留，白占
2 × (16KB q + 0.5KB weights) = **33KB shared memory**。

### 用省下的 smem 加深 KV 流水

smem 预算（SM100 动态上限 ≈227KB；每级 KV = 32KB 数据 + 1KB scale）：

| 配置 | Q stage | KV stage | smem 估算 |
|---|---:|---:|---:|
| 旧默认 | 3 | 3 | ≈157 KB |
| 新默认 | 1 | **6** | ≈223 KB |
| （放不下） | 1 | 7 | ≈256 KB ✗ |

KV 深度从 3 提到 6，TMA 预取窗口翻倍，更好地吞掉 HBM/L2 供数抖动——这是
打分**下限**的主要来源（对 dense 基线同样生效）。

### 配置矩阵实测（1m/8192 sparse scan，均含 §1–§3）

| 配置 | sparse | dense |
|---|---:|---:|
| bufs1 / Q3 / KV3（旧 stage） | 46.67 | 49.42 |
| bufs1 / Q1 / KV4 | 47.08 | 49.03 |
| bufs2 / Q3 / KV3 | 46.27 | 49.12 |
| bufs2 / Q1 / KV5 | 46.24 | 48.37 |
| bufs1 / Q1 / KV6 | 45.30 | 48.18 |
| **bufs2 / Q1 / KV6（新默认）** | **45.28** | **47.33** |

注意 KV4 甚至略差于 KV3（Q stage 腾出的 smem 不够形成有效深度差），
**KV6 才是拐点**——这就是为什么单独砍 Q stage 不够、必须配合加深 KV。

---

## 5. TMEM 累加器双缓冲（附带，≈中性，保留）

动机：原来每 WG 只有一个 TMEM 累加 tile（共用 256/512 列），UMMA(n+1) 必须
等 math warp 完成 block n 的 TMEM 回读，二者串行。改为按 KV block 奇偶双缓冲
（512/512 列，`DSA_TMEM_BUFS=2`，UMMA/empty barrier 均按 `wg*2+parity` 索引，
相位 = `(idx/2)&1`）。

实测基本中性（45.30 vs 45.28 @KV6；Q3/KV3 下 +0.4 ms）——说明 UMMA↔TMEM
串行化并非当前瓶颈（UMMA 本身只占理论峰值的 ~1/3 负载），但该改动无回退、
且在未来 epilogue 更快之后可能显现价值，保留为默认。

---

## 6. 汇总

### scan kernel 演进（1m/8192）

| 阶段 | sparse scan | Δ |
|---|---:|---:|
| 优化前（256/2WG + spare-warp refresh） | 60.03 ms | — |
| + §2 门控 stride/fdiv 惰性 + §3 float4 权重 | 54.13 ms | −9.8% |
| + §1 批量 vote emit | 46.67 ms | −13.8% |
| + §4 Q1/KV6（+§5 双缓冲） | **45.28 ms** | −3.0% |

累计 **−24.6%**；epilogue 常驻开销从 18.5 ms 压到 ~3 ms 量级
（45.28 − 42.4 null-epi）。sparse scan 现已**快于 dense 直写**（47.33）。

### E2E 全量 sweep（16 cell，K=2048，recall 全部 100%）

| cell | LiteTopK 前 | LiteTopK 后 | 提升 | speedup 前 | speedup 后 |
|---|---:|---:|---:|---:|---:|
| 256k/1024 | 3.27 | 2.96 | +9.5% | 1.45x | 1.57x |
| 256k/2048 | 6.04 | 5.49 | +9.1% | 1.47x | 1.59x |
| 256k/4096 | 11.27 | 10.15 | +9.9% | 1.56x | 1.70x |
| 256k/8192 | 21.63 | 19.12 | +11.6% | 1.69x | 1.89x |
| 512k/1024 | 6.33 | 5.72 | +9.6% | 1.45x | 1.59x |
| 512k/2048 | 11.40 | 10.64 | +6.7% | 1.54x | 1.63x |
| 512k/4096 | 22.46 | 21.17 | +5.7% | 1.63x | 1.71x |
| 512k/8192 | 43.81 | 40.46 | +7.6% | 1.66x | 1.79x |
| 768k/1024 | 8.14 | 6.60 | +18.9% | 1.68x | 2.05x |
| 768k/2048 | 14.48 | 12.15 | +16.1% | 1.80x | 2.12x |
| 768k/4096 | 28.55 | 24.00 | +15.9% | 1.92x | 2.26x |
| 768k/8192 | 56.31 | 47.03 | +16.5% | 1.96x | 2.33x |
| 1m/1024 | 9.84 | 7.48 | +24.0% | 1.87x | 2.42x |
| 1m/2048 | 17.53 | 13.41 | +23.5% | 2.09x | 2.71x |
| 1m/4096 | 34.44 | 27.02 | +21.5% | 2.12x | 2.68x |
| 1m/8192 | 67.95 | 54.00 | +20.5% | 2.17x | **2.72x** |

对官方（patched）DeepGEMM `fp8_mqa_logits` 的 3-way 峰值：2.70x → **3.37x**
@1M/8192（`threeway_b200.json`）。

序列越长提升越大，两个原因：scan 在 E2E 中占比更高（88% vs 61%）、门控更紧
使批量 vote 的快路径占比更高（~91% vs ~26%）。

### 正确性验证清单

- `test_dsa.py 256k`（refresh=0）与 `512k`（refresh=64 动态阈值路径）：
  recall = 100.000%，候选统计（cnt_mean=5182.4）与优化前**完全一致**；
- 全量 16-cell sweep：recall 全部 100%（两处 99.99999% 级别的历史浮动也消失）；
- 3-way 全 16 cell：官方/dense/LiteTopK 三方 recall 均 100%；
- float4 权重：FP 累加 DAG 不变 → 分数 bit 级一致；
- 门控 stride：emitted 候选数 +<1%，方向保守（只多不漏）。

### 新增调参项（`-D...`，均已有实测依据的默认值）

| 宏 | 默认 | 含义 |
|---|---|---|
| `DSA_GATE_STRIDE` | 8 | 每 N 个 KV block 重读一次 th_bucket；旧值保守安全 |
| `DSA_Q_STAGES` | 1 | Q TMA 流水级数（每 CTA 一个 q-block，1 级即可） |
| `DSA_KV_STAGES` | 6 | KV TMA 流水级数（227KB smem 预算内的最大值） |
| `DSA_TMEM_BUFS` | 2 | 每 math WG 的 TMEM 累加 tile 数（2=双缓冲） |
| `DSA_NULL_EPILOGUE` | off | 仅测量用：打分+归约、不门控不写出（性能下限） |

### 复现命令（容器内）

```bash
# 阶段分解 + 变体对照
PYTHONPATH=/opt/venvs/deepgemm/lib/python3.12/site-packages \
DSA_CACHE_DIR=/data/dsa_caches \
/usr/bin/python3.12 profile_breakdown.py 256k:4096 1m:8192

PYTHONPATH=... DSA_CACHE_DIR=... /usr/bin/python3.12 bench_scan_variants.py 1m:8192
# 配置扫描示例：
BUILD_TAG=_b1q3k3 FLAGS_EXTRA="-DDSA_TMEM_BUFS=1 -DDSA_Q_STAGES=3 -DDSA_KV_STAGES=3" \
  /usr/bin/python3.12 bench_scan_variants.py 1m:8192

# E2E 全量 + figures
PYTHONPATH=... DSA_CACHE_DIR=... /usr/bin/python3.12 sweep_b200_real_chunk.py
python3 figures/dsa_bench.py
```

### 未做/未生效的方向（记录以免重走）

- **寄存器溢出假设**：ptxas 实测 168 reg / 0 spill，排除；
- **倒数乘法替代 fdiv**：有舍入方向导致边界候选丢失的 recall 风险，放弃，
  改用惰性重算（收益等价、零风险）；
- **TMEM 双缓冲**：单独收益 ≈0（UMMA 非瓶颈），保留因无回退；
- **bcount 原子合并（`DSA_MERGE_BCOUNT`）/关 refresh**：原子量太小，无意义；
- **下一步候选**：小序列时 O_prep 占 37%（其中 ~3.5ms 是 torch 端
  neg/min/max/clone 等张量操作），可融合成一个小 kernel；scan 下限 42 ms
  距 KV 数据 HBM 往返理论值仍有距离，可探 KV 复用（同 wave 内 q-block 对
  KV 的 L2 共享调度）。
