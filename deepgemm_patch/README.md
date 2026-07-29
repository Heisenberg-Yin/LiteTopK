# 旧版 DeepGEMM 的 32-head MQA 补丁

这个 header overlay 只用于无法编译 `num_heads=32` 的旧版 DeepGEMM。
DeepGEMM 2.5 已支持该 shape，B200 DSA 和 vLLM 复现应优先直接使用 2.5，
无需应用本目录补丁。

## 文件

- `sm100_fp8_mqa_logits.cuh`：可覆盖到旧 DeepGEMM package 的完整 header。
- `sm100_fp8_mqa_logits.diff`：同一处修改的可移植 patch。

修改将寄存器预加载的权重数限制在实际 head 数以内。本目录不生成 `.so`；
覆盖 header 后，DeepGEMM 会在下一次对应 JIT build 时重新编译。

## Header overlay

先复制整个 Python package，避免修改共享环境：

```bash
SRC_PACKAGE=/path/to/site-packages/deep_gemm
PATCH_DIR=/data01/home/ziqi.yin/litetopk_github_clone/deepgemm_patch
DEST_ROOT=/tmp/deepgemm_32h

cp -a "$SRC_PACKAGE" "$DEST_ROOT"
cp "$PATCH_DIR/sm100_fp8_mqa_logits.cuh" \
  "$DEST_ROOT/deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh"

PYTHONPATH="$DEST_ROOT" python your_program.py
```

`DEST_ROOT` 中应直接包含 `deep_gemm/`。若目标 package 的目录布局或 header
内容不同，应停止并换用匹配版本。

## Patch 方式

在包含 `deep_gemm/` 的 package 根目录执行：

```bash
PATCH_DIR=/data01/home/ziqi.yin/litetopk_github_clone/deepgemm_patch
patch --dry-run -p1 < "$PATCH_DIR/sm100_fp8_mqa_logits.diff"
patch -p1 < "$PATCH_DIR/sm100_fp8_mqa_logits.diff"
```

应用后用所需 `num_heads=32` 的调用触发 JIT 编译；仅检查 patch 命令成功
不能替代 CUDA 编译检查。
