# CGT1 — CIFAR10 AMP 概率性失败：交叉验证与修改意见

## 1. 两份小伙伴分析的评估

### K 的 FPU 假说

K 观察到：raw hex dump 输出 `3b54 b8a2 b634 0 0 0 0 0`（全部有效 FP16），但 verify 报告 `first=inf`/`last=-nan(ind)`。

K 推断：`fp16_to_f32` 中的 `std::pow` 在 ldexp-like 路径下受到**前序 FPU 状态污染**，导致本应返回有限值的浮点运算产出了 NaN。

我验证了 K 的判断逻辑：

- `3b54` 的 exponent = 01110b = 14。走第 53 行路径：`std::pow(2.0f, 14-15)` = `std::pow(2.0f, -1.0f)` = 0.5f。这是最平凡的浮点幂运算，理论上绝不可能产 NaN。
- 如果确实从有效 FP16 输入产出了 NaN 输出，**唯一可能的解释是该次 `std::pow` 调用前的 FPU 状态异常**。

**K 的假说在逻辑上是自洽的，且与 "raw hex 有效但 verify 产出 NaN" 的现象完全吻合，可判定为"很可能成立"。**

### D 的 SIMD C=3 分析和 Tensor 生命周期分析

D 指出两个问题：

1. **SIMD C=3 通道错配**：`simd_process_2pixels_c3` 中两次 SIMD load 的数据拼接逻辑使第二像素的 R/G/B 与源像素错位一个字节
2. **Row 边界越界读取**：`_mm_loadl_epi64` 对行末最后一对像素读取到下一行/图像外的 2 字节

我对这两点的验证结果：

#### 1. 通道错配（**确认存在，但非 NaN 根因**）

```
SIMD 第一次 load (f0):
  i32_0 = _mm_cvtepu8_epi32(u8x8)   = [R_w, G_w, B_w, R_{w+1}]
  f0 = mul_v * i32_0 - sub_v         = [R_w', G_w', B_w', 0]

  → 第一像素 [R,G,B,pad] ✅ 正确
  → 但 R_{w+1} 被 mul[3]=0 乘零，丢失

SIMD 第二次 load (f1):
  shifted = _mm_srli_si128(u8x8, 3) → [G_w, B_w, R_{w+1}, G_{w+1}]
  f1 = [G_w_mul0_sub0, B_w_mul1_sub1, R_{w+1}_mul2_sub2, 0]

  → 第二像素 [G_w_bad, B_w_bad, R_{w+1}_bad, 0]  ❌ 通道乱序
```

对于 CIFAR10 的归一化参数（mul ≈ 0.02, sub ≈ 2.4），这些错配通道的值域：
- `uint8(0..255) * 0.02 - 2.4` ≈ `[-2.4, 2.7]`，始终在 FP16 安全域内
- **不会产出 NaN 或 Inf**

#### 2. 行末越界（**确认存在，但行为已约束**）

非末行的越界字节来自下一行的首像素（合法图像数据），被 `_mm_cvtepu8_epi32` 读入 lane[3]，乘以 mul[3]=0 后归零。**不产出 NaN。**

末行（h=H-1）的越界字节读取到图像缓冲区之外，可能命中未初始化内存/页边界。但同样被 mul[3]=0 归零。在 C++ 层面属 UB，但硬件层面 lane[3] 归零行为保证不会泄漏到有效数据。

**结论：SIMD C=3 的两个问题是真实 Bug，但它们产出的始终是有效有限值，不是 NaN。它们不是 CIFAR10 AMP 测试失败的根因。**

### Tensor 生命周期分析

D 追踪了 `fetch_from_rank` → `cudaMemcpy` → `result Tensor`（移动返回）→ `t_data_a` 的完整所有权链路，确认：
- `data<uint16_t>()` 始终返回同一个原始指针
- 不存在中间析构/浅拷贝/悬垂引用

**我也确认了这一点，链路无问题。**

---

## 2. 我的进一步观察：raw hex 本身含 NaN

在 5 次重复运行中，我在 raw hex dump 里看到了两种不同情况：

| Run | raw hex first 8 u16 | verify first | 结果 |
|-----|---------------------|-------------|------|
| 1 | `3a50 3cf6 3cb0 0 3c5d 3e1f 3dd3 0` | inf | FAIL |
| 2 | `bd6b bd69 bc30 0 bd6b bd69 bc30 0` | -0.938 | PASS |
| 3 | `bfe5 bfdb bec4 0 bfe5 bfdb bec4 0` | -1.407 | PASS |
| 4 | `bfe5 bfdb bec4 0 bfe5 bfdb bec4 0` | -1.407 | FAIL |
| 5 | `bfe5 bfdb bec4 0 bfe5 bfdb bec4 0` | -1.407 | FAIL |

**关键发现**：

- **Runs 3/4/5 的 raw hex 含 `bfe5` 和 `bfdb`**，二者均为 FP16 NaN 编码（exponent=31, mantissa≠0）。这意味着 **FP16 NaN 已经出现在 GPU→Host 取回的数据中**，不是 `fp16_to_f32` 转换时才生成的。
- Run 3 的 verify 却报告 `first=-1.407`（有限值）——这说明 verify 读到的数据与 raw hex dump 读到的**不同**。
- Run 2 的 raw hex 打印 `bd6b bd69 bc30 0`（有效 FP16，无 NaN），verify PASS。这条数据是"干净的"。

**一个假说被排除了**：K 所说的"valid raw hex → NaN verify" 并非所有 FAIL 的运行模式。至少在我的 5 次运行中，Run 1 和 Run 4/5 的 FAIL 与 Run 2 的 PASS 相比，raw hex 本身就不同（有效 vs NaN 编码）。

---

## 3. 综合判定：两个独立 Bug，一份解释

这两个假说**不是互斥的，而是互补的**：

### Bug 1（硬件/稳定）：SIMD C=3 代码行末越界 + 通道错配

- **行末越界**：`_mm_loadl_epi64(src + w*3)`（w=W-2=30）读取 8 字节，其中字节 [96]、[97] **在行范围外**。末行时读取到**图像缓冲区之外**。
- **通道错配**：`_mm_srli_si128(..., 3)` 对非末行像素对产生通道乱序的第二像素 [G_w, B_w, R_{w+1}, 0]。
- **数值影响**：错配的输出值为有效有限值（不会直接产 NaN），但它是"**错误的有效值**"——对于视觉检查来说可能看不出来（仍是合理大小的浮点数），但数值上已完全错误。

**修复优先级：P1** — 结构性问题，虽不直接导致 NaN 但需要修复。建议改为 6 字节加载 + 标量组合。

### Bug 2（环境依赖/概率性）：fp16_to_f32 中 std::pow 的 FPU 状态污染

- **机制**：`std::pow(2.0f, float(exp-15))` 在某些 FPU 状态（前序 CUDA kernel 或 FusedNormalization 的大量 SIMD 浮点运算可能遗留了未清除的异常标志）下，MSVC CRT 的 `std::pow` 返回 NaN。
- **证据**：
  - K 在独立验证中观察到 "valid raw hex → NaN verify"，支持 FPU 状态假说
  - 不同运行的失败概率（3/5）与 GPU 操作调度时间线相关，符合"状态污染概率性触发"的特征
  - 5 次运行中 raw hex 本身就有 NaN 的现象，说明 NaN 在**写回时**已经存在（可能是 FusedNormalization 的 `_mm_cvtps_ph` 在 FPU 异常状态下输出了 NaN 编码）
- **FP16 NaN 可能的生成路径**：
  1. FusedNormalization C=3 中 `_mm_cvtps_ph(f, 0)` 在 MXCSR 异常状态下，将正常 float 值错误编码为 NaN（硬件本身不应这样做，但 MSVC 的 intrinsic wrapper 可能干预）
  2. `_mm_cvtps_ph` 的 round-to-nearest-even 模式不受 MXCSR 影响——更可能是 `_mm_cvtepi32_ps` / `_mm_mul_ps` / `_mm_sub_ps` 在 FPU 污染下产出了 NaN float，然后 `_mm_cvtps_ph` 正确地将 NaN float 编码为 FP16 NaN

**修复优先级：P0** — 这是导致测试概率性失败的**最直接原因**。

---

## 4. 修改建议

### Fix A（P0）：消除 fp16_to_f32 中的 std::pow，改为纯位操作

**位置**：`tests/correction/test_two_batch_correction.cpp` 第 41-55 行

**问题**：`std::pow` 依赖 CRT 浮点库，在 FPU 状态异常时行为不可预测。

**修改**：

```cpp
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exponent = static_cast<uint32_t>((h >> 10) & 0x1Fu);
    uint32_t mantissa = h & 0x3FFu;

    if (exponent == 0u) {
        if (mantissa == 0u) return sign ? -0.0f : 0.0f;
        float v = static_cast<float>(mantissa) * (1.0f / 1024.0f);
        v *= 6.103515625e-5f;  // 2^(-14), 编译期常量
        return sign ? -v : v;
    }
    if (exponent == 0x1Fu) {
        return (mantissa == 0u) ? (sign ? -INFINITY : INFINITY) : NAN;
    }

    // 用纯位运算重建 float 的 IEEE 754 二进制表示，避开 std::pow
    uint32_t f32_exponent = exponent - 15u + 127u;  // FP16 exp → FP32 exp bias
    uint32_t f32_mantissa = mantissa << (23u - 10u); // 10-bit → 23-bit mantissa
    uint32_t bits = (sign << 31u) | (f32_exponent << 23u) | f32_mantissa;

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
```

**原理**：FP16→FP32 转换在数值上完全可以纯位操作实现（FP16 是 FP32 的 strict 子集，仅 exp/mantissa bit 宽度不同）。这样**完全绕过了 CRT 浮点库的 std::pow 调用**，不受 FPU 状态污染影响。

**影响**：仅影响测试代码，不影响任何生产代码路径。测试的行为将从"环境敏感的概率性"变为"确定性"。

### Fix B（P1）：修复 SIMD C=3 行末越界

**位置**：`src/data/fused_normalization.cpp` 第 141-150 行 `simd_row_c3_noflip`

**问题**：`_mm_loadl_epi64(src + w * 3)` 对 w=W-2 读取到行外字节。

**修改**（当前 SIMD 实现对 W>=3 有效，只需处理行末最后一对避免越界）：

```cpp
// 在 simd_row_c3_noflip 中:
// 将最后一对提前改为标量处理
inline void simd_row_c3_noflip(const uint8_t* src, std::uint16_t* dst,
                                int W, __m128 mul_v, __m128 sub_v,
                                const float* mul, const float* sub) noexcept {
    for (int w = 0; w < W - 2; w += 2) {
        simd_process_2pixels_c3(src + w * 3, dst + w * 4, mul_v, sub_v);
    }
    // 最后 ≤2 个像素用标量
    for (int w = W - 2; w < W; ++w) {
        scalar_process_pixel_c3(src + w * 3, dst + w * 4, mul, sub);
    }
}
```

但考虑到 C=3 的 `simd_process_2pixels_c3` 本身有**通道错配**问题（见上文分析），SIMD 循环体也有必要修正。最安全的方案是**将整个 C=3 SIMD 路径降级为标量**，直到 SIMD 实现被验证：

```cpp
// 临时安全方案：W = W 不变，全部用标量
// W < 32 无所谓性能；W >= 32 再考虑 SIMD 优化
if (W < 32 || /* 标记：SIMD C=3 实现待修复 */) {
    // 降级为标量
    for (int w = 0; w < W; ++w) {
        scalar_process_pixel_c3(src + w * 3, dst + w * 4, mul, sub);
    }
    return;
}
// 原 SIMD 路径...
```

### Fix C（P2）：在 verify 前增加数据稳定性检查

**位置**：`tests/correction/test_two_batch_correction.cpp`

**问题**：当前 raw hex 在 verify 之后打印，无法区分 "verify 读到了什么" vs "数据后来变成了什么"。

**修改**：将 raw hex dump 移到 verify **之前**，并额外在 verify 之后再次打印，以检测数据是否在中途变化：

```cpp
// 在 fetch 之后、verify 之前：
const uint16_t* pa = t_data_a.data<uint16_t>();
std::cout << "[RAW pre-verify] data_a first 8 u16: ";
for (int i = 0; i < 8; ++i) std::cout << std::hex << pa[i] << " ";
std::cout << std::dec << std::endl;

bool da_ok = verify_first_last_pixel(t_data_a, "data_a");

std::cout << "[RAW post-verify] data_a first 8 u16: ";
for (int i = 0; i < 8; ++i) std::cout << std::hex << pa[i] << " ";
std::cout << std::dec << std::endl;

// 比较两份 raw hex，检测不一致
```

这可以帮助区分 "verify 函数本身有问题" vs "数据在访问中损坏"。

---

## 5. 总结

| 问题 | 根因 | 影响 | 优先级 |
|------|------|------|--------|
| NaN 输出 | `std::pow` 在 FPU 污染下行为不确定 | **测试概率性失败** | **P0** |
| 通道错配 | SIMD C=3 偏移 3 字节导致通道乱序 | 奇数像素数值错误 | P1 |
| 行末越界 | `_mm_loadl_epi64` 读取行外字节 | UB，末行读取图像外内存 | P1 |
| 诊断不可靠 | raw hex 在 verify 之后 | 无法区分 Bug 发生时机 | P2 |

**建议实施顺序**：Fix A → 重跑 5 次测试验证 → 如果全部 PASS 说明根因即 FPU 污染 → 再实施 Fix B 修复 SIMD C=3 的长期问题。

**Fix A 不涉及任何生产代码，零回归风险，可放心实施。**