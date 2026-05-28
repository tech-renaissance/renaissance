# FVB_FINAL：FusedNormalization C=1 AMP Bug 最终裁定与修复方案

**版本**: FVB_FINAL
**日期**: 2026-05-25
**基于**: FND1/2/3.md + FVB1/2/3.md + 源码逐行交叉验证
**状态**: 终审定稿，待执行

---

## 一、现象

MNIST AMP 测试 raw hex dump：

```
first 8 u16:  b6ca  0  0  0  0  0  0  0
```

NHWC C_padded=4：Pixel 0 = [b6ca, 0, 0, 0] ✅，Pixel 1 = [0, 0, 0, 0] ❌
CIFAR10 AMP ✅ / FP32 ✅ 正常。

---

## 二、确定问题清单（三个问题，FVB1/2/3全体一致确认）

### P0 — 问题 1：mul_v / sub_v 未广播到全部 SIMD lane

| 项 | 内容 |
|---|------|
| **位置** | `src/data/fused_normalization.cpp` 第 536-537 行 (no-flip), 542-543 行 (flip) |
| **FND引用** | FND1 §二、FND2 §二问题1、FND3 §二§2 |
| **FVB引用** | FVB1 §二 Bug1、FVB2 §二问题1、FVB3 §问题一 |
| **判决** | ✅ 确认。四份验证报告一致裁定为致命级根因。 |

**源码 (line 536-537, no-flip)：**
```cpp
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);
```

**根因**：
- `_mm_set_ps(a,b,c,d)` → lane[3]=a, lane[2]=b, lane[1]=c, lane[0]=d
- 故 `_mm_set_ps(0,0,0,mul[0])` → `[0, 0, 0, mul[0]]`，仅 lane[0] 有值
- `simd_process_4pixels_c1` 中 4 个 float lane 代表 **4 个独立像素**（非同一像素的不同通道）
- 运算后果：`[p0*mul[0], p1*0, p2*0, p3*0]` → pixel 1-3 强制归零
- 这是破解 raw hex `b6ca 0 0 0 0 0 0 0` 之谜的**终极答案**

**为何 CIFAR 不受影响**：
C=3 路径 (line 524) 的 `_mm_set_ps(0, mul[2], mul[1], mul[0])` 语义正确：
每个 lane 对应同一像素的 R/G/B/padding 通道，逐 lane 赋值 mul[0..2] 是设计意图。

---

### P1 — 问题 2：simd_process_4pixels_c1 写入量 2x 冗余

| 项 | 内容 |
|---|------|
| **位置** | `src/data/fused_normalization.cpp` 第 116-123 行 |
| **FND引用** | FND1 §三、FND2 §二问题2、FND3 §二§3 |
| **FVB引用** | FVB1 §二 Bug2、FVB2 §二问题2、FVB3 §问题二 |
| **判决** | ✅ 确认。冗余 store 存在，W=28/32 暂不触发但缓冲区溢出隐患确实存在。 |

**源码 (line 116-123)：**
```cpp
__m128i t0 = _mm_unpacklo_epi16(h, zero);   // ← h 低 4 uint16，有实际数据
__m128i t1 = _mm_unpackhi_epi16(h, zero);   // ← h 高 4 uint16，全是 0

_mm_storeu_si128(dst + 0,  _mm_unpacklo_epi32(t0, zero));   // 8 uint16 ✅
_mm_storeu_si128(dst + 8,  _mm_unpackhi_epi32(t0, zero));   // 8 uint16 ✅
_mm_storeu_si128(dst + 16, _mm_unpacklo_epi32(t1, zero));   // 8 uint16 ❌ 全是 0
_mm_storeu_si128(dst + 24, _mm_unpackhi_epi32(t1, zero));   // 8 uint16 ❌ 全是 0
```

**根因**：
- `_mm_cvtps_ph(f, 0)` 只产出 4 个 FP16（寄存器低 64 位），高 64 位为 0
- `t1` 处理高 64 位 → 全 0 → 后两条 store 写入 16 个无意义的 0x0000
- 函数应当输出 16 个 uint16_t（4 像素 × 4 通道），实际输出 32 个

**步进冲突**：
`simd_row_c1_noflip` 的 `dst + w*4` 假定每轮 16 个 uint16_t，但函数写 32 个：
```
w=0:  写入 [0,31]，但预期只占用 [0,15]
w=4:  写入 [16,47]，前16个覆盖 w=0 的溢出席
w=24: 写入 [96,127]，[112,127] 溢出到下一行前4像素位置
```

**影响评估**：
- MNIST W=28 (4 的倍数)：溢出被下一组覆盖，**fix-1 后不会产生数据错误**
- 但 W 非 4 倍数时：`scalar_process_pixel_c1` 写入的尾部像素会被溢出零覆盖 → 静默损坏

---

### P2 — 问题 3：DIAG-WORKER 以 float* 误读 FP16

| 项 | 内容 |
|---|------|
| **位置** | `src/data/preprocess_worker.cpp` 第 1221 行 |
| **FND引用** | FND2 §二问题3 |
| **FVB引用** | FVB1 §二问题3、FVB2 §二问题3、FVB3 §问题三 |
| **判决** | ✅ 确认。调试代码 UB，不破坏生产数据但严重误导诊断。 |

**源码 (line 1221)：**
```cpp
float* fout = reinterpret_cast<float*>(final_output_ptr);  // AMP下这是 uint16_t*！
```

**根因**：AMP 模式 `final_output_ptr` → FP16 缓冲区 → 强转 `float*` → type punning UB。

---

## 三、FVB 报告交叉验证矩阵

| 判定项 | FVB1 | FVB2 | FVB3 | 最终裁决 |
|--------|------|------|------|----------|
| 问题1：mul_v/sub_v未广播 | P0 ✅ | P0 ✅ | P0 ✅ | **P0 确认** |
| 问题2：冗余 store 写入 | P1 ✅ | P1 ✅ | P1 ✅ | **P1 确认** |
| 问题3：DIAG-WORKER UB | P2 ✅ | P2 ✅ | P2 ✅ | **P2 确认** |
| SIMD前16个uint16布局正确 | ✅ | ✅ | ✅ | **确认，布局无误** |
| flip路径542-543为死代码 | ✅ | - | - | **确认(线182-183 void cast)** |
| 只修fix-1可消MNIST错 | ✅ | ✅ | ✅ | **确认** |

三份 FVB 报告对问题认定、优先级、修复方案**完全一致**，无分歧。

---

## 四、最终修复方案

### 改动文件清单

| 文件 | 行号 | 改动类型 | 优先级 |
|------|------|----------|--------|
| `src/data/fused_normalization.cpp` | 536-537, 542-543 | `_mm_set_ps` → `_mm_set1_ps` | P0 |
| `src/data/fused_normalization.cpp` | 116-123 | 删除 t1 计算 + 2 个冗余 store | P1 |
| `src/data/preprocess_worker.cpp` | 1221 附近 | DIAG 增加 AMP 分支 | P2 |

---

### Fix-1 (P0·必须)：mul_v/sub_v 改为广播

**文件**: `src/data/fused_normalization.cpp`

**行号 536-537**（`C == 1 && !do_flip`）：
```cpp
// Before
            __m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
            __m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);

// After
            __m128 mul_v = _mm_set1_ps(mul[0]);
            __m128 sub_v = _mm_set1_ps(sub[0]);
```

**行号 542-543**（`C == 1 && do_flip`）—— 当前为死代码（`simd_row_c1_flip` 走 scalar 回退），但为保持一致性：
```cpp
// Before
            __m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
            __m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);

// After
            __m128 mul_v = _mm_set1_ps(mul[0]);
            __m128 sub_v = _mm_set1_ps(sub[0]);
```

---

### Fix-2 (P1·推荐)：删除 simd_process_4pixels_c1 冗余 store

**文件**: `src/data/fused_normalization.cpp`，行号 116-123

**完整替换**（整个函数不变，仅改 store 部分）：
```cpp
// Before (lines 116-123)
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

**说明**：
- 删除 `t1` 计算行和 `dst+16`/`dst+24` 两条 store（删 3 行）
- 输出从 32 个 uint16_t 缩减为恰好 16 个，与 `dst + w*4` 步进对齐
- 前 2 个 store 的布局已经过 SIMD trace 验证为正确的 NHWC C_padded=4 格式

---

### Fix-3 (P2·建议)：DIAG-WORKER AMP 兼容

**文件**: `src/data/preprocess_worker.cpp`，行号 1219-1255

**注意**: `tensor.cpp:1540` 的 `half_to_float` 为 `static` 函数，跨翻译单元不可见。
Fix-3 使用**局部内联 lambda** 实现 FP16→FP32 转换，不依赖外部符号。

```cpp
// Before
            float* fout = reinterpret_cast<float*>(final_output_ptr);
            ...
            for (int i = 0; i < 784; ++i) {
                float v = fout[i];
                ...
            }

// After
            if (using_amp_) {
                auto fp16_to_float = [](uint16_t h) -> float {
                    uint32_t sign = (h >> 15) & 0x1;
                    uint32_t exp  = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    if (exp == 0) return sign ? -0.0f : 0.0f;
                    if (exp == 31) return (mant == 0)
                        ? (sign ? -std::numeric_limits<float>::infinity()
                                : std::numeric_limits<float>::infinity())
                        : std::numeric_limits<float>::quiet_NaN();
                    return (sign ? -1.0f : 1.0f)
                         * std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f,
                                      static_cast<int>(exp) - 15);
                };
                const std::uint16_t* fout16 = reinterpret_cast<const std::uint16_t*>(final_output_ptr);
                for (int i = 0; i < 784; ++i) {
                    float v = fp16_to_float(fout16[i * 4]);   // 只读通道0
                    // 统计逻辑不变
                }
            } else {
                float* fout = reinterpret_cast<float*>(final_output_ptr);
                for (int i = 0; i < 784; ++i) {
                    float v = fout[i];
                    // 统计逻辑不变
                }
            }
```

---

## 五、修复执行顺序

| Step | 动作 | 命令 |
|------|------|------|
| 1 | 应用 Fix-1 + Fix-2 | 编辑 `fused_normalization.cpp` |
| 2 | 编译 test_two_batch_correction | `ninja test_two_batch_correction` |
| 3 | 验证 MNIST AMP | `test_two_batch_correction --amp --dataset mnist` |
| 4 | 验证 CIFAR AMP | `test_two_batch_correction --amp --dataset cifar10` |
| 5 | 回归 MNIST GPU/CPU | `test_two_batch_correction --dataset mnist` |
| 6 | 回归 CIFAR GPU/CPU | `test_two_batch_correction --dataset cifar10` |
| 7 | (可选) 应用 Fix-3 | 编辑 `preprocess_worker.cpp` |

---

## 六、影响范围确认

| 场景 | 影响 | 说明 |
|------|------|------|
| MNIST AMP | ✅ 需修复 | Fix-1 后 raw hex 应为 `b6ca 0 0 0 b6ca 0 0 0 ...` |
| CIFAR10 AMP | ❌ 无影响 | C=3 路径 mul_v 语义正确 |
| ImageNet AMP | ❌ 无影响 | C=3，同上 |
| FP32 (全部) | ❌ 无影响 | 不走 `__AVX2__` FP16 分支 |
| 非 AVX2 | ❌ 无影响 | C=1 AMP 直接 `TR_NOT_IMPLEMENTED` |

---

## 七、总结

| 项 | 内容 |
|------|------|
| 经过的报告 | FND1/2/3 + FVB1/2/3 = **6 份独立分析** |
| 确认问题数 | **3 个**（1 致命 + 1 潜在 + 1 调试） |
| 判决一致性 | **100%** — 6 份报告对主因无分歧 |
| 最小修复成本 | **2 行** `_mm_set_ps` → `_mm_set1_ps` |
| 推荐修复范围 | **5 行改动**（fix-1 两处各 2 行 + fix-2 减 3 行） |
| 风险等级 | **极低** — 仅影响 C=1 AMP AVX2 路径，不波及 C=3 / FP32 |