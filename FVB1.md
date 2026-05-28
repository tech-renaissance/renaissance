# FVB1：FusedNormalization C=1 AMP Bug 最终验证与修复方案

**版本**: Final
**日期**: 2026-05-25
**基于**: FND1.md、FND2.md、FND3.md + 源码逐行验证

---

## 一、背景与现象

MNIST AMP 测试 raw hex dump：

```
first 8 u16:  b6ca  0  0  0  0  0  0  0
```

NHWC C_padded=4 解读：
- Pixel 0 = [b6ca, 0, 0, 0] ✅ 正确（b6ca ≈ -0.4243 = 黑背景经 normalize 的 FP16 值）
- Pixel 1 = [0, 0, 0, 0] ❌ 应为 [b6ca, 0, 0, 0]

CIFAR10 AMP ✅ / FP32 ✅ 均正常。

---

## 二、源码验证：三个问题逐一确认

### Bug 1（致命）：mul_v / sub_v 仅填充 lane[0]，lane[1..3] 为 0

**位置**: `src/data/fused_normalization.cpp` 第 536-537 行（no-flip）、542-543 行（flip）

**源码**：

```cpp
// Line 536-537 (C == 1 && !do_flip)
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);

// Line 542-543 (C == 1 && do_flip) — 同样错误
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);
```

**根因分析**：

`_mm_set_ps(a,b,c,d)` 的 lane 映射为：
- lane[3] = a, lane[2] = b, lane[1] = c, lane[0] = d

故 `_mm_set_ps(0,0,0,mul[0])` → `[0, 0, 0, mul[0]]`，仅 lane[0] 有值。

`simd_process_4pixels_c1`（line 106-124）一次处理 **4 个独立像素**，每个 float lane 对应一个不同像素：

```cpp
__m128 f = _mm_cvtepi32_ps(i32);   // [pixel0, pixel1, pixel2, pixel3]  各像素灰度值
f = _mm_mul_ps(f, mul_v);          // [p0*mul, p1*0,  p2*0,  p3*0]      ← BUG!
f = _mm_sub_ps(f, sub_v);          // [p0*mul-sub, 0, 0, 0]              ← BUG!
```

四个像素乘以/减去的是**同一个** mul[0]/sub[0]（单通道归一化因子），需要广播到全部 4 个 lane。当前写法导致 pixel 1-3 被乘 0 后强制清零。

**对比 C=3 正确路径**（line 524-525）：

```cpp
// C=3: simd_process_2pixels_c3 中每个 lane 对应同一像素的不同通道
__m128 mul_v = _mm_set_ps(0.0f, mul[2], mul[1], mul[0]);  // lane[0]=R, lane[1]=G, lane[2]=B, lane[3]=0(unused)
```

C=3 中每个 lane 语义为"同像素不同通道"，逐 lane 赋值是正确的。C=1 路径**错误复用了**此模式，但 C=1 语义为"不同像素同一通道"，需要广播。

**SIMD 输出追踪验证**：

以 `h = _mm_cvtps_ph(f, 0)` 为起点，验证 `simd_process_4pixels_c1` 的数据布局（假设 mul_v 正确广播后，fp0~fp3 均有效）：

```
h       = [fp0, fp1, fp2, fp3, 0, 0, 0, 0]   (4 FP16 值在低 64 位)
t0      = _mm_unpacklo_epi16(h, zero)
        = [0, fp0, 0, fp1, 0, fp2, 0, fp3]   (int16_t 级插零)
lo      = _mm_unpacklo_epi32(t0, zero)
        = [fp0|0, 0, fp1|0, 0]               (int32_t 级插零)
        = uint16_t[8]: [fp0,0,0,0, fp1,0,0,0] ← pixel 0,1 ✅
hi      = _mm_unpackhi_epi32(t0, zero)
        = [fp2|0, 0, fp3|0, 0]
        = uint16_t[8]: [fp2,0,0,0, fp3,0,0,0] ← pixel 2,3 ✅
```

**结论**：前 2 个 store（`dst`、`dst+8`）的布局**完全正确**——NHWC C_padded=4，每个像素 `[fp16_val, 0, 0, 0]`。问题不在布局，而在 pixel 1-3 的值为 0（由 Bug 1 导致）。

**验证结论**：✅ 确认。FND1/FND2/FND3 三份报告对此判断一致，源码逐行验证通过。

---

### Bug 2（次要）：simd_process_4pixels_c1 写入量冗余

**位置**: `src/data/fused_normalization.cpp` 第 116-123 行

**源码**：

```cpp
__m128i t0 = _mm_unpacklo_epi16(h, zero);
__m128i t1 = _mm_unpackhi_epi16(h, zero);        // ← t1 全是 0

_mm_storeu_si128(reinterpret_cast<__m128i*>(dst),      _mm_unpacklo_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),  _mm_unpackhi_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), _mm_unpacklo_epi32(t1, zero));  // ← 多余
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 24), _mm_unpackhi_epi32(t1, zero));  // ← 多余
```

**根因**：

- `_mm_cvtps_ph(f, 0)` 只填充寄存器低 64 位（4 个 FP16），高 64 位为 0
- `t1 = _mm_unpackhi_epi16(h, zero)` 处理的是高 4 个 uint16_t（全 0）
- 因此 `dst+16` 和 `dst+24` 两条 store 写入的 16 个 uint16_t **全是 0x0000**
- 函数实际写入 **32 个 uint16_t**，但 4 像素 × 4 通道仅需 **16 个 uint16_t**

**步进不匹配**：

```cpp
// simd_row_c1_noflip (line 167-177)
for (; w + 3 < W; w += 4) {
    simd_process_4pixels_c1(src + w, dst + w * 4, mul_v, sub_v);
    // dst + w*4 按 16 uint16_t 步进，但函数写入 32 uint16_t
}
```

写入量 / 步进对比：

```
w=0:  写入 dst[0..31]   步进目标 dst[0]
w=4:  写入 dst[16..47]  步进目标 dst[16]  ← [16..31] 与 w=0 重叠
w=24: 写入 dst[96..127] 步进目标 dst[96]  ← [112..127] 溢出到下一行开头
```

**实际影响评估**（MNIST W=28）：

- 前 7 组（w=0,4,8,12,16,20,24）的溢出都会被下一组覆盖，**如果 mul_v 已修复**，不会造成数据错误
- 最后一组（w=24）的 `dst[112..127]` 溢出到下一行首 4 像素位置 → **但会被下一行的处理覆盖**
- 如果 W 不是 4 的倍数，`scalar_process_pixel_c1` 补尾像素的数据可能被覆盖 → **潜在风险**

**验证结论**：✅ 确认。多余 store 存在，当前 W=28 不触发数据损坏（会被覆盖），但属于代码缺陷和缓冲区溢出隐患。

---

### 问题 3（调试误导）：DIAG-WORKER 以 float* 读取 FP16 数据

**位置**: `src/data/preprocess_worker.cpp` 第 1221 行

**源码**：

```cpp
float* fout = reinterpret_cast<float*>(final_output_ptr);  // ← AMP下这是FP16!
for (int i = 0; i < 784; ++i) {
    float v = fout[i];                                      // ← Type punning UB
    if (v > -0.424f && v < -0.423f) continue;               // ← 阈值对FP32有意义
    ...
}
```

**根因**：
- `final_output_ptr` 在 AMP 模式下指向 `uint16_t*`（FP16）缓冲区
- 强制按 `float*` 读取属于 C++ 严格别名规则违规（undefined behavior）
- 碰巧 FP16 的 bit pattern 被 reinterpret 为 FP32 时指数未溢出（落在 subnormal 区间），数值虽乱但未 crash——纯属巧合

**实际影响**：
- **不影响生产数据**（仅诊断日志），但会严重误导调试
- 当前输出的 `nz=784` 和数值 `[0, 6.55724e-41]` 都是 FP16→FP32 reinterpret 的伪影，不代表真实归一化结果

**验证结论**：✅ 确认。调试代码 UB，但不影响数据正确性。

---

## 三、FND 三份报告的一致性总结

| 发现项 | FND1 | FND2 | FND3 | 源码验证 |
|--------|------|------|------|----------|
| mul_v/sub_v 未广播（致命） | ✅ | ✅ | ✅ | ✅ 确认 |
| 多余 store 写入（次要） | ✅ | ✅ | ✅ | ✅ 确认 |
| DIAG-WORKER FP16→FP32（调试） | - | ✅ | - | ✅ 确认 |
| SIMD 布局 (前 16 个 uint16_t) | ✅ 正确 | ✅ 正确 | ✅ 正确 | ✅ 验证通过 |

**三份报告对主因判断完全一致**：Bug 1 (`_mm_set_ps` → 未广播) 是 MNIST AMP 数据大面积归零的**唯一必要充分条件**。

---

## 四、修复方案（优先级排序）

### P0 — fix-1：mul_v/sub_v 广播（必须修复）

**文件**: `src/data/fused_normalization.cpp`

**行号**: 536-537（no-flip）

```cpp
// Before
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);

// After
__m128 mul_v = _mm_set1_ps(mul[0]);
__m128 sub_v = _mm_set1_ps(sub[0]);
```

**行号**: 542-543（flip）

注意：`simd_row_c1_flip`（line 179-187）并未使用 SIMD，而是回退到 `scalar_process_pixel_c1`（`(void)mul_v; (void)sub_v;`），所以这两行目前是死代码。但为了一致性（未来若 SIMD-化 flip 路径），应同步修改：

```cpp
// Before
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);

// After
__m128 mul_v = _mm_set1_ps(mul[0]);
__m128 sub_v = _mm_set1_ps(sub[0]);
```

### P1 — fix-2：删除 simd_process_4pixels_c1 中的多余 store（推荐修复）

**文件**: `src/data/fused_normalization.cpp`
**行号**: 116-123

```cpp
// Before
__m128i zero = _mm_setzero_si128();
__m128i t0 = _mm_unpacklo_epi16(h, zero);
__m128i t1 = _mm_unpackhi_epi16(h, zero);

_mm_storeu_si128(reinterpret_cast<__m128i*>(dst),      _mm_unpacklo_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),  _mm_unpackhi_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), _mm_unpacklo_epi32(t1, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 24), _mm_unpackhi_epi32(t1, zero));

// After
__m128i zero = _mm_setzero_si128();
__m128i t0 = _mm_unpacklo_epi16(h, zero);

_mm_storeu_si128(reinterpret_cast<__m128i*>(dst),      _mm_unpacklo_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),  _mm_unpackhi_epi32(t0, zero));
```

变化：删除 2 行（t1 计算 + 2 个多余 store），输出从 32 个 uint16_t 缩减为恰好 16 个，与 `dst + w * 4` 步进对齐。

### P2 — fix-3：DIAG-WORKER 增加 AMP 类型判断（建议修复）

**文件**: `src/data/preprocess_worker.cpp`
**行号**: 1221 附近

```cpp
// Before
float* fout = reinterpret_cast<float*>(final_output_ptr);
for (int i = 0; i < 784; ++i) {
    float v = fout[i];
    ...
}

// After (伪代码)
if (using_amp_) {
    const uint16_t* fout16 = reinterpret_cast<const uint16_t*>(final_output_ptr);
    for (int i = 0; i < 784; ++i) {
        float v = half_to_float(fout16[i * 4]);  // 只读每个像素的有效通道 0
        ...
    }
} else {
    float* fout = reinterpret_cast<float*>(final_output_ptr);
    for (int i = 0; i < 784; ++i) {
        float v = fout[i];
        ...
    }
}
```

---

## 五、影响范围对照表

| 场景 | 受影响 | 原因 |
|------|--------|------|
| **MNIST AMP** | ✅ Bug 1 | 每 4 像素仅 lane[0] 正确，其余 3 个为零 |
| **CIFAR10 AMP** | ❌ | C=3 → `simd_process_2pixels_c3`，mul_v 逐 lane 赋值正确 |
| **ImageNet AMP** | ❌ | C=3，同上 |
| **FP32 所有** | ❌ | 不走 `__AVX2__` FP16 路径 |
| **非 AVX2** | ❌ | `#else` 分支报 TR_NOT_IMPLEMENTED 或走 Eigen 回退 |

---

## 六、验证计划

修复后需运行：

1. `test_two_batch_correction --amp --dataset mnist` → raw hex 应为 `b6ca 0 0 0 b6ca 0 0 0 ...`
2. `test_two_batch_correction --amp --dataset cifar10` → 回归验证，应继续保持 PASS
3. `test_two_batch_correction --dataset mnist`（GPU/CPU，非 AMP） → 回归验证

---

## 七、总结

| 项目 | 内容 |
|------|------|
| **根本原因** | `execute()` C=1 AMP 路径中 `mul_v`/`sub_v` 使用 `_mm_set_ps` 仅填充 lane[0]，lane[1..3] 为 0，导致 `simd_process_4pixels_c1` 中 pixel 1-3 被乘 0 清零 |
| **为什么 CIFAR 不受影响** | C=3 路径每个 lane 对应同一像素的不同通道，逐 lane 赋值 mul[0..2] 是正确的 |
| **修复方案** | fix-1: `_mm_set1_ps` 替换 `_mm_set_ps`（必须）；fix-2: 删除冗余 store（推荐）；fix-3: DIAG-WORKER AMP 兼容（建议） |
| **只修 fix-1 是否足够** | ✅ 足够消除 MNIST AMP 的数据错误 |
| **风险评估** | 极小 — 改动仅影响 C=1 AMP AVX2 路径，不涉及 FP32、C=3、非 AVX2 代码路径 |