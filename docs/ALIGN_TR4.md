# TR4 框架内存对齐规范（最终定稿）

**版本**: V4.1 Final
**日期**: 2026-05-06
**作者**: 技术觉醒团队（S/K/D 三方共识 + KM/DS/GL 三位专家评审）
**状态**: ✅ 定稿
**设计原则**: 统一、兼容、极简

---

## 📋 文档概述

本文档是 TR4 框架**张量内存对齐的唯一权威规范**，规定所有张量在 cuDNN（GPU）和 XNNPACK（CPU）两大后端、FP32 / FP16 / INT32 / INT8 四种精度下的内存布局策略。该方案经过四轮深入讨论和三位专家独立评审，是 TR4 追求极致性能的基石。

### 核心目标

- ✅ **统一**：一套规则服务所有后端、所有精度、所有张量类型
- ✅ **兼容**：同时满足 cuDNN Tensor Core 和 XNNPACK 的所有要求
- ✅ **极简**：概念清晰，实现简单，无过度对齐
- ✅ **高效**：以最小显存开销换取最大带宽利用率
- ✅ **安全**：避免 cuDNN 内部 layout conversion 风险，性能可预测

### 适用范围

| 维度 | 覆盖范围 |
|------|----------|
| **后端** | cuDNN / cuDNN Frontend（GPU）、XNNPACK（CPU） |
| **精度** | FP32、FP16、INT32、INT8 |
| **张量类型** | 特征图（activation）、特征图梯度、权重（filter）、BN 参数（scale/bias/running_mean/running_var）、mask、标量、EMA 权重 |

---

## 🎯 核心结论（一句话定案）

> **特征图 / 特征图梯度区：FP32 / FP16 统一 C 补齐到 8 的倍数。输入缓冲区：FP32 不做 C 补齐（保持原始 C=3 或 C=1），FP16 固定 C 补齐到 4。INT32 / INT8 / 权重 / 其他一切张量：永远紧凑。所有张量末尾统一 +16 字节，首地址 256 字节对齐由 MemoryPlan 布局算法保证。**

---

## 📐 一、术语定义

为避免歧义，本文档使用以下精确术语：

| 术语 | 符号 | 定义 | 单位 |
|------|------|------|------|
| **pixel_stride** | — | NHWC 布局下相邻两个像素之间的字节距离，即 `C_aligned × elem_size` | 字节 |
| **row_stride** | — | NHWC 布局下相邻两行之间的字节距离，即 `W × pixel_stride`（packed 时） | 字节 |
| **elem_size** | — | 单个元素的字节数：FP32=4，FP16=2，INT32=4，INT8=1 | 字节 |
| **C_aligned** | — | 对 C 通道补齐后的逻辑通道数 | 元素个数 |
| **packed** | — | stride 严格等于各维度大小的乘积，无行间或通道间 padding | — |
| **行尾 padding** | — | 让 `row_stride > W × pixel_stride`，在每行末尾留下空洞 | — |
| **XNN_EXTRA_BYTES** | — | XNNPACK 要求的张量末尾预留字节数，x86/ARM 平台为 16 | 字节 |

---

## 📐 二、设计原则

### 2.1 张量分类与布局路径

TR4 框架中，不同 Region 的张量采用差异化的对齐策略。这种差异化不是理论推导的结果，而是基于实测数据：输入缓冲区 FP32 不补齐最快（补齐反而增加显存带宽开销），输入缓冲区 FP16 补齐到 4 最快（补齐到 8 反而更慢）。

| 张量类别 | Region | 允许的 dtype | 对齐方式 | 理由 |
|----------|--------|:------------:|----------|------|
| **默认区（手动绘图）** | `DEFAULT` | **所有四种** | **永远紧凑**，零 padding | `alloc()` 不指定区域时的默认目标，适用于逐元素算子、AXPY 等不需要行步幅对齐的场景 |
| **输入缓冲区 — 图像张量** | `INPUT_BUF_A`, `INPUT_BUF_B` | FP32、FP16 | **FP32：不补齐（保持原始 C）**；**FP16：C 补齐到 4** | H2D 传输直接搬运 CPU 端数据，padding 会增大传输字节数成倍增长；实测 FP32 不补齐运算最快，FP16 补齐到 4 运算最快 |
| **输入缓冲区 — 标签张量** | `INPUT_BUF_A`, `INPUT_BUF_B` | INT32 | **强制紧凑** | 标签张量位于低地址，非浮点计算，语义不兼容 padding |
| **特征图 / 特征图梯度** | `FEATURE`, `GRAD_SLOT` | **仅 FP32、FP16** | **C 补齐到 8 的倍数**，保持 packed | 网络内部逐层流动，大体积全量全局内存读写，行步幅对齐决定带宽利用率 |
| **权重 / filter / BN 参数** | `PARAM_FP32_W`, `PARAM_FP32_G`, `PARAM_FP16_W`, `PARAM_FP16_G` | FP32、FP16 | **完全紧凑**，零 padding | 被 cuDNN 内部 repack，原始布局被屏蔽 |
| **BN 统计量 / mask / 标量 / EMA / INT32 / INT8 张量** | `BN_STATS`, `MASK`, `SCALAR`, `EMA_FP32_W`, `EMA_FP16_W` 等 | FP32、FP16、INT32、INT8 | **完全紧凑**，零 padding | 数据量极小，或非浮点计算 |

### 2.2 八项铁律

1. **特征图对齐只能通过 C 通道补齐实现**。`row_stride` 必须等于 `W × aligned_pixel_stride`，保持 NHWC packed 关系。严禁在行尾单独添加 padding。

2. **特征图区 FP32 和 FP16 统一 C → 8**。TR4 的性能敏感路径是 AMP 下的 FP16 训练。

3. **输入缓冲区 FP32 不做 C 补齐**。H2D 传输直接搬运 CPU 端数据，padding 会导致传输字节数成倍增长；实测不补齐运算速度最快。

4. **输入缓冲区 FP16 C 补齐到 4**（不是 8）。cudnn TensorCore 要求 C 为 4 的倍数（实测 C=3 报找不到 engine）；实测 C=4 比 C=8 更快（少搬 4 个死通道的带宽）。

5. **只有 FP32 和 FP16 能使用 C 通道补齐**。INT32 和 INT8 永远紧凑，不会出现在 `FEATURE` / `GRAD_SLOT` 中。

6. **INT32 和 INT8 必须永远紧凑**。`row_stride = W × C × elem_size`。仍需末尾 +16 字节和首地址 256 字节对齐。

7. **权重 / filter 始终紧凑**。stride 严格等于各维度大小的乘积。cudnn 和 XNNPACK 对此要求一致。

8. **所有张量末尾预留 16 字节且首地址 256 字节对齐**。XNNPACK 硬性要求尾部预留（cudnn 侧无害），首地址对齐由 `MemoryPlan::finalize()` 布局算法保证。

---

## 📐 三、输入缓冲区对齐方案

**适用范围**：`INPUT_BUF_A`、`INPUT_BUF_B` 中的图像张量。这两个缓冲区各包含一个标签张量（INT32，低地址，紧凑）和一个数据张量（FP32 或 FP16，高地址）。

输入缓冲区的对齐策略不同于特征图区。原因有两个：

1. **传输性能**：输入数据由 Preprocessor 在 CPU 端产生，通过 H2D 传输流搬运到 GPU 显存。如果对 C 通道做了不必要的 padding，H2D 传输就要搬运额外的死字节，传输时间可能成倍增长。
2. **运算性能（实测）**：FP32 输入不补齐最快（补齐后增加显存带宽开销）；FP16 输入补齐到 4 最快（补齐到 8 反而更慢）。

### 3.1 FP32 输入图像张量

**规则：不做 C 通道补齐，保持原始通道数。** 例如 ImageNet 的 C=3，MNIST 的 C=1。

```cpp
// FP32 输入图像张量（位于 INPUT_BUF_A/B）
uint64_t pixel_stride   = static_cast<uint64_t>(C) * 4;           // C × sizeof(float)
uint64_t row_stride     = static_cast<uint64_t>(W) * pixel_stride; // 完全紧凑
uint64_t logical_bytes  = static_cast<uint64_t>(N) * H * row_stride;
uint64_t total_bytes    = logical_bytes + 16;                      // XNN_EXTRA_BYTES
```

**设计理由**：
- **运算性能**：实测 FP32 输入张量不 padding 运算速度最快。padding 后显存带宽开销增加，反而更慢。
- **传输性能**：H2D 搬运 `N × H × W × C × 4` 字节。若 C=3→8，传输量膨胀 2.7 倍，传输时间可能无法被计算完全掩盖，破坏计算-传输重叠设计。
- **cudnn 兼容性**：FP32 下 cudnn 对输入张量的 C 通道没有硬性倍数要求，不补齐也能正常运算。
- **XNNPACK 兼容性**：XNNPACK 允许自定义 C stride，C=1 或 C=3 均可正常运行。

### 3.2 FP16 输入图像张量

**规则：C 通道补齐到 4（恰好 4，不是 8 的倍数）。**

```cpp
// FP16 输入图像张量（位于 INPUT_BUF_A/B）
uint64_t C_aligned      = align_up(static_cast<uint64_t>(C), 4);   // C 补齐到 4 的倍数
uint64_t aligned_pixel  = C_aligned * 2;                           // C_aligned × sizeof(half)
uint64_t row_stride     = static_cast<uint64_t>(W) * aligned_pixel; // 保持 packed
uint64_t logical_bytes  = static_cast<uint64_t>(N) * H * row_stride;
uint64_t total_bytes    = logical_bytes + 16;                      // XNN_EXTRA_BYTES
```

**为什么 C 补到 4（恰好到 4），不是 8？**

这是实测结果（见 `docs/NEW_VIEW.md`）：

> "输入通道数必须 pad 到 4 的倍数，否则会报找不到引擎的错误，因为 TensorCore 不支持。实测输入通道数 4 比输入通道数 8 要快。"

- **C 必须 ≥ 4**：FP16 下 cudnn TensorCore 要求输入通道数为 4 的倍数。C=3 时报找不到 engine 的错误。
- **C=4 比 C=8 快**：C=4 意味着 H2D 只搬运 4 个通道的数据；C=8 要多搬运 4 个死通道的带宽，且显存中的 dead channels 增加计算开销。实测 C=4 更快。

### 3.3 INT32 标签张量

**规则：完全紧凑，不做任何 padding。** 标签张量位于输入缓冲区的低地址位置。

```cpp
// INT32 标签张量（位于 INPUT_BUF_A/B 的低地址位置）
uint64_t row_stride    = static_cast<uint64_t>(W) * C * 4;        // 完全紧凑
uint64_t logical_bytes = static_cast<uint64_t>(N) * H * row_stride;
uint64_t total_bytes   = logical_bytes + 16;                      // XNN_EXTRA_BYTES
```

---

## 📊 四、特征图 / 特征图梯度对齐方案

**适用范围**：`FEATURE`、`GRAD_SLOT` 中的 FP32 / FP16 张量。

**统一规则**：FP32 和 FP16 采用完全相同的对齐策略——C 补齐到 8 的倍数。

### 4.1 对齐参数

| 参数 | FP32 | FP16 | 说明 |
|------|:----:|:----:|------|
| `elem_size` | 4 | 2 | `sizeof(float)` / `sizeof(half)` |
| C 补齐目标 | **8 的倍数** | **8 的倍数** | `C_aligned = align_up(C, 8)` |
| `aligned_pixel_stride` | `C_aligned × 4` | `C_aligned × 2` | FP32 最小 32 B，FP16 最小 16 B |
| `row_stride` | `W × C_aligned × 4` | `W × C_aligned × 2` | 保持 packed |

### 4.2 计算公式

```cpp
// FP32 / FP16 特征图 / 特征图梯度（统一规则：C 补齐到 8 的倍数）
uint64_t C_aligned     = align_up(static_cast<uint64_t>(C), 8);   // C 统一补齐到 8 的倍数
uint64_t aligned_pixel = C_aligned * elem_size;                   // FP32: 最小 32 B, FP16: 最小 16 B
uint64_t row_stride    = static_cast<uint64_t>(W) * aligned_pixel; // 保持 packed
uint64_t logical_bytes = static_cast<uint64_t>(N) * H * row_stride;
uint64_t total_bytes   = logical_bytes + 16;                       // XNN_EXTRA_BYTES
```

**等价的 C 补齐表示**：

```cpp
uint64_t C_aligned = align_up(C, 8);       // FP32/FP16 统一 C 补齐到 8 的倍数
// FP32: pixel_stride = C_aligned × 4（最小 32 B，C=1~8 时均为 32 B）
// FP16: pixel_stride = C_aligned × 2（最小 16 B，C=1~8 时均为 16 B）
// row_stride = W × aligned_pixel_stride
```

### 4.3 设计理由

#### 为什么 FP32 和 FP16 在特征图区统一？

1. **TR4 的性能敏感路径是 AMP 下的 FP16 训练**。冲击 ResNet-50 世界纪录依赖 FP16 特征图。FP32 特征图仅在非 AMP 模式或 CPU 推理中使用，不是性能瓶颈。
2. **FP32 在 CPU 上（XNNPACK）**：`pixel_stride = 32` 字节（C=8×4）已满足 AVX2 的 32B SIMD 宽度。
3. **FP32 在 GPU 上**：cudnn 内部 tiling 和 shared memory buffering 掩盖跨行不对齐的惩罚。
4. **极简压倒一切**：特征图区只有一条规则（C → 8），无 dtype 分支。

#### 为什么和输入缓冲区规则不同？

特征图在网络内部由算子逐层产生，各层之间的 C 值在设计模型时已可预见。C=8 补齐在 ResNet-50 中间层（C=64/128/256...）**零额外开销**。输入缓冲区数据来自 CPU 端，C 值由数据集决定（通常 C=1 或 3），对其进行不必要的 padding 会直接增大 H2D 传输量，破坏计算-传输重叠设计。

#### 为什么选 C→8 而不是更高？

**方案对比（首层 C=4 → C_aligned）**：

| 方案 | C 补齐值 | pixel_stride（FP16） | 首层显存膨胀 | 性能收益 | 状态 |
|------|:--------:|:--------------------:|:------------:|:--------:|:----:|
| **C→8（定案）** | **8** | **16 B** | **×2.0** | **基准（满足 Tensor Core 门槛）** | **✅ 采用** |
| C→16 | 16 | 32 B | ×4.0 | < 3% | ❌ 否决 |
| C→32 | 32 | 64 B | ×8.0 | < 5% | ❌ 否决 |
| C→64 | 64 | 128 B | ×16.0 | < 7% | ❌ 三位专家一致否决 |

**选 C→8 的五大理由**：

1. **cudnn Tensor Core 准入门槛**（KM 专家）：NVIDIA 官方要求 FP16 下 `"C and K are divisible by 8"`。C=8 是启用 Tensor Core 快速路径的最低要求。
2. **显存开销可控**：C→8 首层仅膨胀 2 倍。C→64 首层膨胀 16 倍，显存惩罚远大于带宽收益。
3. **边际收益递减**（KM 专家实测）：从 C=8 提升到 C=64，整体网络性能提升 < 5%，显存峰值可能翻倍。
4. **ResNet-50 实测验证**：FP16 下 C 补到 8 后，除首层外的所有层 row_stride 天然 128 倍数。
5. **极简原则**：统一 C → 8 规则，无分层条件分支。

#### 为什么不做行尾补齐？

GL 专家提出"C 补到 8 + 行末尾补到 128 字节"的方案。该方案被否决：KM 专家警告行尾 padding 让 `row_stride > W × pixel_stride`，cudnn 可能触发内部 layout conversion，性能暴跌。且 TR4 核心平台是 A100×8，核心后端是 cudnn，必须以 cudnn 行为准则为最高优先级。

---

## 📊 五、权重 / Filter / 其他一切张量

**适用范围**：`PARAM_*`、`EMA_*`、`BN_STATS`、`MASK`、`SCALAR` 中的所有张量，以及 `INPUT_BUF_A` / `INPUT_BUF_B` 中的 INT32 标签张量。**适用于所有精度（FP32、FP16、INT32、INT8）**。

```cpp
// 权重 / Filter / BN 参数 / mask / 标量 / EMA / INT32 / INT8 标签
uint64_t row_stride    = static_cast<uint64_t>(W) * C * elem_size;  // 完全紧凑
uint64_t logical_bytes = static_cast<uint64_t>(N) * H * row_stride;
uint64_t total_bytes   = logical_bytes + 16;                        // XNN_EXTRA_BYTES
```

**为什么权重必须紧凑？**

- **cudnn**: filter 必须为 packed layout（如 KRSC）。添加 stride padding 会导致 non-packed，cudnn 可能无法走 Tensor Core 快速路径。
- **XNNPACK**: filter 接口不提供 `pixel_stride` 参数，要求内存连续。
- **INT32/INT8**: 非浮点计算类型，语义不兼容 padding，强制紧凑。

---

## 📋 六、关键技术决策汇总

### 6.1 被否决的错误做法（严禁使用）

| 错误做法 | 来源 | 问题 | 正确处理 |
|----------|------|------|----------|
| `align_up_256(c)` — 把 C 对齐到 256 | 原版 NEW_PLAN | C=3→256，显存膨胀数十倍 | 按 Region 和 dtype 差异化处理 |
| `align_up(row_bytes, 128/256)` — 行尾 padding | 当前 `memory_plan.cpp` | `row_stride > W × pixel_stride`，非 packed | 保持 packed |
| FP32 特征图 C 补到 32 | 旧版 ALIGN_TR4 | 过度对齐，FP32 非性能敏感路径 | FP32/FP16 统一 C → 8（仅特征图区） |
| 输入缓冲区 FP32 做 C 补齐 | 旧版 ALIGN_TR4 | H2D 传输膨胀，实测反而更慢 | 输入缓冲区 FP32 不补齐 |
| 输入缓冲区 FP16 C 补到 8 | 旧版 ALIGN_TR4 | 实测 C=4 最快，C=8 多搬运死通道 | 输入缓冲区 FP16 C → 4 |
| INT32 / INT8 走 C 补齐路径 | 旧版假设 | 破坏整数语义 | 强制紧凑 |
| FP16 C 补到 64 | D 第二轮 | 首层膨胀 16 倍 | C 补到 8 |
| FP16 行末尾补齐到 128 B | GL 第三轮 | 与 KM 的 layout conversion 警告冲突 | 不做行尾补 |
| `nbytes` 外层包 `align_up_256` | 原版 NEW_PLAN | 布局算法已做 `align_up(cursor, 256)` | 不包 |

### 6.2 统一规则表

| | 输入缓冲区 FP32 | 输入缓冲区 FP16 | 特征图 FP32/FP16 | INT32 / INT8 | 权重（所有精度） |
|---|---|---|---|---|---|
| `elem_size` | 4 | 2 | 4 / 2 | 4 / 1 | 4 / 2 / 1 |
| C 补齐 | **无** | **C → 4** | **C → 8** | 无 | 无 |
| `pixel_stride` | `C × 4` | `C_aligned × 2` | `C_aligned × elem_size` | `C × elem_size` | `C × elem_size` |
| `row_stride` | `W × C × 4` | `W × pixel_stride` | `W × pixel_stride` | `W × C × elem_size` | `W × C × elem_size` |
| 行间空洞 | 无 | 无 | 无 | 无 | 无 |
| 末尾预留 | +16 B | +16 B | +16 B | +16 B | +16 B |
| 首地址对齐 | 256 B | 256 B | 256 B | 256 B | 256 B |

---

## 💻 七、统一实现公式（最终版）

```cpp
/**
 * @brief 计算行步幅（NHWC 布局下 H 维度 stride，单位：字节）
 * @param s            张量形状 (N, H, W, C)
 * @param dt           数据类型
 * @param region_kind  语义分类，控制补齐策略：
 *                     FEATURE_MAP: 特征图/特征图梯度，FP32/FP16 → C 补齐到 8
 *                     INPUT_BUF:   输入缓冲区，FP32 不补齐，FP16 → C 补齐到 4
 *                     COMPACT:     权重/其他，一律紧凑
 * @return 行步幅（字节数）
 */
enum class RegionKind { FEATURE_MAP, INPUT_BUF, COMPACT };

uint64_t compute_row_stride(const Shape& s, DType dt, RegionKind kind) {
    uint64_t elem_size = (dt == DType::FP32 || dt == DType::INT32) ? 4 :
                         (dt == DType::FP16)                     ? 2 : 1;

    // ===== 路径A：特征图 / 特征图梯度 — C 补齐到 8 =====
    if (kind == RegionKind::FEATURE_MAP && (dt == DType::FP32 || dt == DType::FP16)) {
        uint64_t C_aligned     = align_up(static_cast<uint64_t>(s.c()), 8);
        uint64_t aligned_pixel = C_aligned * elem_size;
        return static_cast<uint64_t>(s.w()) * aligned_pixel;
    }

    // ===== 路径B：输入缓冲区 — FP16 C→4，FP32 不补齐 =====
    if (kind == RegionKind::INPUT_BUF && dt == DType::FP16) {
        uint64_t C_aligned     = align_up(static_cast<uint64_t>(s.c()), 4);
        uint64_t aligned_pixel = C_aligned * 2;
        return static_cast<uint64_t>(s.w()) * aligned_pixel;
    }

    // ===== 路径C：权重 / INT32 / INT8 / 输入缓冲区 FP32 — 完全紧凑 =====
    return static_cast<uint64_t>(s.w()) * s.c() * elem_size;
}

/**
 * @brief 计算张量总字节数（含 XNN_EXTRA_BYTES 尾部预留）
 */
uint64_t compute_nbytes(const Shape& s, DType dt, RegionKind kind) {
    uint64_t rs = compute_row_stride(s, dt, kind);
    return static_cast<uint64_t>(s.n()) * s.h() * rs + 16;
}
```

**使用方式** — 在 `MemoryPlan::alloc_impl` 中按 Region 分发：

```cpp
DTensor MemoryPlan::alloc_impl(const Shape& shape, DType dtype,
                               Region region, /*...*/) {
    RegionKind kind;
    if (region == Region::FEATURE || region == Region::GRAD_SLOT) {
        kind = RegionKind::FEATURE_MAP;
    } else if (region == Region::INPUT_BUF_A || region == Region::INPUT_BUF_B) {
        kind = RegionKind::INPUT_BUF;
    } else {
        kind = RegionKind::COMPACT;
    }

    uint64_t row_stride  = compute_row_stride(shape, dtype, kind);
    uint64_t total_bytes = compute_nbytes(shape, dtype, kind);
    // ... 后续逻辑
}
```

---

## 🔌 八、与后端 API 的衔接

### 8.1 cudnn Frontend API

```cpp
// FP32 特征图（C 补齐到 8，pixel_stride = 32）
std::vector<int64_t> strides = {
    H * row_stride,      // N stride
    row_stride,          // H stride
    aligned_pixel_stride,// W stride = pixel_stride
    elem_size            // C stride
};

auto tensor = cudnn_frontend::Tensor_builder()
    .set_dim({N, H, W, C_aligned})  // C 是补齐后的值！
    .set_stride(strides)
    .set_alignment(256)
    .build();
```

### 8.2 XNNPACK API

```cpp
// FP16 特征图（C 补齐到 8，pixel_stride = 16）
status = xnn_create_convolution2d_nhwc_f16(
    ..., aligned_pixel_stride, aligned_pixel_stride, ...);
```

---

## 🔟 九、完整代码示例

### 示例 1：输入缓冲区 FP16（ImageNet 首层）

**输入**：N=256, H=224, W=224, C=3（RGB 图像，AMP 开启 → FP16）

```cpp
Shape shape(256, 224, 224, 3);  // NHWC, FP16

uint64_t C_aligned     = align_up(3, 4);        // C 补齐到 4
uint64_t aligned_pixel = 4 * 2;                 // = 8 B
uint64_t row_stride    = 224 * 8;               // = 1792
uint64_t logical_bytes = 256 * 224 * 1792;      // = 102,760,448
uint64_t total_bytes   = 102,760,448 + 16;

assert(C_aligned == 4);             // C=3→4 ✅
assert(aligned_pixel == 8);         // pixel_stride = 8 B ✅
assert(row_stride == 224 * 8);      // packed ✅
```

**对比 C→8（被否决）**：C_aligned = 8, pixel_stride=16B, row_stride=3584, 总字节膨胀 2 倍，H2D 传输量也膨胀 2 倍。

### 示例 2：输入缓冲区 FP32（ImageNet 首层）

**输入**：N=256, H=224, W=224, C=3（RGB 图像，AMP 关闭 → FP32）

```cpp
Shape shape(256, 224, 224, 3);  // NHWC, FP32

uint64_t pixel_stride  = 3 * 4;                 // = 12 B（不补齐！）
uint64_t row_stride    = 224 * 12;              // = 2688
uint64_t logical_bytes = 256 * 224 * 2688;      // = 154,140,672
uint64_t total_bytes   = 154,140,672 + 16;

assert(row_stride == 224 * 3 * 4);  // 完全紧凑，无 C 补齐 ✅
```

### 示例 3：特征图区 FP16（ResNet-50 conv2_x）

**输入**：N=256, H=56, W=56, C=64

```cpp
Shape shape(256, 56, 56, 64);  // NHWC, FP16

uint64_t C_aligned     = align_up(64, 8);       // C 已是 8 的倍数
uint64_t aligned_pixel = 64 * 2;                // = 128 B
uint64_t row_stride    = 56 * 128;              // = 7168
uint64_t logical_bytes = 256 * 56 * 7168;       // = 102,502,528
uint64_t total_bytes   = 102,502,528 + 16;

assert(C_aligned == 64);            // 无膨胀 ✅
assert(row_stride % 128 == 0);      // 天然 128 对齐 ✅
```

### 示例 4：特征图区 FP32（ResNet-50 conv2_x）

**输入**：N=256, H=56, W=56, C=64

```cpp
Shape shape(256, 56, 56, 64);  // NHWC, FP32

uint64_t C_aligned     = align_up(64, 8);       // C 已是 8 的倍数
uint64_t aligned_pixel = 64 * 4;                // = 256 B
uint64_t row_stride    = 56 * 256;              // = 14336
uint64_t logical_bytes = 256 * 56 * 14336;      // = 205,783,456
uint64_t total_bytes   = 205,783,456 + 16;

assert(C_aligned == 64);            // 无膨胀 ✅
assert(aligned_pixel == 256);       // pixel_stride = 256 B ✅
```

### 示例 5：ResNet-50 首层卷积权重（FP32）

**输入**：K=64, R=7, S=7, C=3（7×7 Conv）

```cpp
Shape shape(64, 7, 7, 3);  // NHWC filter 布局：[K, R, S, C]

uint64_t row_stride    = 7 * 3 * 4;             // = 84（完全紧凑）
uint64_t logical_bytes = 64 * 7 * 84;           // = 37,632
uint64_t total_bytes   = 37,632 + 16;

assert(row_stride == 7 * 3 * 4);    // 完全紧凑 ✅
```

### 示例 6：INT32 标签张量（输入缓冲区低地址）

**输入**：N=256，Shape = [1, 1, 1, 256]

```cpp
Shape shape(1, 1, 1, 256);  // NHWC, INT32

uint64_t row_stride    = 1 * 256 * 4;           // = 1024（完全紧凑）
uint64_t logical_bytes = 1 * 1 * 1024;          // = 1024
uint64_t total_bytes   = 1024 + 16;

assert(row_stride == 1024);         // 完全紧凑，无 C 补齐 ✅
```

---

## 📖 十、快速参考速查表

### 10.1 对齐参数速查

| 张量类型 | Region | dtype | C 补齐 | `pixel_stride` | `row_stride` | 末尾 | 首地址 |
|---------|--------|-------|:------:|--------|-------------|:----:|:------:|
| 输入缓冲区（图像） | `INPUT_BUF_A/B` | FP32 | **无** | `C × 4` | `W × C × 4` | +16 B | 256 B |
| 输入缓冲区（图像） | `INPUT_BUF_A/B` | FP16 | **C → 4** | `C_aligned × 2` | `W × pixel_stride` | +16 B | 256 B |
| 输入缓冲区（标签） | `INPUT_BUF_A/B` | INT32 | 无 | `C × 4` | `W × C × 4` | +16 B | 256 B |
| 特征图 | `FEATURE`, `GRAD_SLOT` | FP32 / FP16 | **C → 8** | `C_aligned × elem_size` | `W × pixel_stride` | +16 B | 256 B |
| 权重 | `PARAM_*`, `EMA_*` 等 | 任意 | 无 | `C × elem_size` | `W × C × elem_size` | +16 B | 256 B |
| INT32 / INT8 张量 | 其他 | INT32 / INT8 | 无 | `C × elem_size` | `W × C × elem_size` | +16 B | 256 B |

### 10.2 常见错误速查

| ❌ 错误做法 | ✅ 正确做法 | 原因 |
|------------|-----------|------|
| `align_up_256(c)` | 按 Region/dtype 差异化补齐 | 对齐对象不是 C 本身 |
| 行尾 padding | 保持 packed | 避免 cudnn layout conversion |
| 输入缓冲区 FP32 做 C 补齐 | 输入 FP32 不补齐 | H2D 传输膨胀，实测反而慢 |
| 输入缓冲区 FP16 C→8 | 输入 FP16 C→4 | 实测 C=4 最快 |
| FP16 C→64 | FP16 C→8 | 显存膨胀 16 倍，收益 < 5% |
| `nbytes` 外包 `align_up_256` | 仅末尾 +16 | `finalize()` 已处理首地址对齐 |
| 权重做 C 补齐 | 权重始终紧凑 | cudnn / XNNPACK 均要求 filter packed |

---

## 🔧 十一、MemoryPlan 修改清单

### 11.1 当前代码病灶

`src/graph/memory_plan.cpp` 中的 `compute_row_stride` 对所有张量一视同仁地做**行尾 padding**：

```cpp
// ❌ 当前代码 — 必须废除
uint64_t MemoryPlan::compute_row_stride(const Shape& s, DType dt) {
    uint64_t row_bytes = static_cast<uint64_t>(s.w()) * s.c() * elem_size;
    uint64_t align = (dt == DType::FP16 || dt == DType::INT8) ? 128 : 256;
    return align_up(row_bytes, align);  // 行尾 padding！
}
```

### 11.2 精确修改点

| # | 修改项 | 位置 | 说明 |
|---|--------|------|------|
| 1 | **引入 `RegionKind` 枚举** | `memory_plan.h` | 三类语义：`FEATURE_MAP`（C→8）、`INPUT_BUF`（FP16 C→4 / FP32 紧凑）、`COMPACT`（紧凑） |
| 2 | **重写 `compute_row_stride`** | `memory_plan.cpp` | 按三条路径分发，废除行尾 padding |
| 3 | **`compute_nbytes` 加 +16** | `memory_plan.cpp` | 所有张量统一：`N × H × row_stride + 16` |
| 4 | **修正 `alloc_impl`** | `memory_plan.cpp` | 根据 `Region` 映射到 `RegionKind` |
| 5 | **更新 `Entry` 的 `row_stride`** | `memory_plan.cpp` | 反映对齐后的值 |
| 6 | **验证 `validate_alignment_iron_law`** | `memory_plan.cpp` | 新增：验证 `row_stride % pixel_stride == 0` |
| 7 | **声明同步** | `memory_plan.h` | 更新签名 |

### 11.3 不修改的范围

以下逻辑**保持不变**：
- `finalize()` 两遍布局算法（预计算 → 预留 → 布局 FP16 → 回填 FP32）
- FP16/FP32 配对公式 `O32 = 2×O16 - C`
- `align_up(cursor, 256)` 首地址对齐
- 通信分桶锚点计算
- `dtensors_cache_` 构建逻辑

---

## ✅ 十二、兼容性论证

### 12.1 cudnn / cudnn Frontend（GPU）

| 要求 | 本规范满足情况 |
|------|----------------|
| Tensor Core 准入（FP16 C 为 4 的倍数，输入） | ✅ 输入 FP16 C → 4 |
| Tensor Core 准入（FP16 C 为 8 的倍数，特征图） | ✅ 特征图 FP16 C → 8 |
| Filter 必须 packed | ✅ 权重零 padding |
| 指针首地址 128 B 对齐 | ✅ 256 B 对齐超过要求 |
| NHWC stride 描述符正确 | ✅ 标准 packed |
| 禁止 filter 行间 padding | ✅ 权重无行间 padding |

### 12.2 XNNPACK（CPU）

| 要求 | 本规范满足情况 |
|------|----------------|
| `XNN_EXTRA_BYTES` 尾部预留 16 B | ✅ 所有张量统一 +16 |
| pixel_stride SIMD 对齐（C 为 8 的倍数，特征图） | ✅ 特征图 FP32/FP16 C → 8 |
| filter 紧凑 | ✅ 权重完全 packed |
| C=1 或 C=3 小通道正常运行 | ✅ 输入 FP32 不补齐，实测 C=1 可运行 |
| 自定义 `input_pixel_stride` | ✅ 可直接传入 |
| 数据指针 SIMD 对齐（16 B 底线） | ✅ 256 B 对齐远超要求 |

---

## 📚 十三、参考依据

### 13.1 实测数据

- **NEW_VIEW.md**：输入缓冲区对齐实测
  - FP32 输入不补齐运算最快，padding 后更慢
  - FP16 输入 C=4 最快，C=3 报找不到 engine，C=8 慢于 C=4
  - XNNPACK C=1 也能正常运行

### 13.2 专家评审记录

- **KM 专家**：cudnn / XNNPACK 行步幅对齐专题分析（`memory/KM.md`）
- **DS 专家**：cudnn layout 与 Tensor Core 要求（`memory/DS.md`）
- **GL 专家**：两级 Padding 策略与 XNNPACK API 分析（`memory/GL.md`）

### 13.3 内部讨论记录

- **NEW_PLAN.md**：内存规划新提案及四轮评审记录
- **NEW_VIEW.md**：输入缓冲区对齐实测与策略分化

### 13.4 官方文档

- **NVIDIA cudnn 官方文档**：Tensor Core 要求 "C and K are divisible by 8 (FP16)"
- **XNNPACK 官方文档**：`xnnpack.h` `XNN_EXTRA_BYTES` 宏定义，自定义 C stride 支持

---

## 🎯 十四、最终定案声明

> **本规范为 TR4 框架张量内存对齐的最终权威标准，所有后续开发严格以此为准。**

**核心结论**：

1. **输入缓冲区 FP32**：不做 C 补齐，保持原始 C（如 ImageNet C=3，MNIST C=1）
2. **输入缓冲区 FP16**：C 补齐到 4（恰好到 4，不是 8）
3. **特征图 / 特征图梯度 FP32 / FP16**：统一 C 补齐到 8 的倍数
4. **INT32 / INT8 / 权重（所有精度）**：永远紧凑
5. **所有张量**：末尾统一 +16 字节，首地址 256 字节对齐

**满足目标**：

- ✅ **统一**：语义分类清晰（三类 RegionKind），仅对齐值按 Region 区分
- ✅ **兼容**：同时满足 cudnn Tensor Core 和 XNNPACK 所有要求
- ✅ **极简**：三条路径覆盖全场景，无不必要的分支
- ✅ **高效**：基于实测数据，不对齐的不补齐、该补齐的恰好补齐
- ✅ **安全**：避免 layout conversion 风险，H2D 传输量最优

---

**文档状态**：✅ 最终定稿
**最后更新**：2026-05-06
**下次审查**：无重大问题无需修订

*本文档为 TR4 张量内存对齐的最终定稿方案，所有后续开发严格以此为准。*
