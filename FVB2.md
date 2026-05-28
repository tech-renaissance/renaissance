# FVB2：FusedNormalization C=1 AMP Bug 最终验证报告

## 一、代码复查结论

对 `src/data/fused_normalization.cpp` 关键路径进行逐行追踪后，确认 **FND1.md / FND3.md 的分析完全成立**，未发现新的隐藏问题。

复查范围：
- `simd_process_4pixels_c1` (line 106-124)
- `simd_row_c1_noflip` (line 167-177)
- `execute()` C=1 AMP 分支 (line 535-546)
- `scalar_process_pixel_c1` (line 135-141)

---

## 二、真实存在的问题（已确认）

### 问题 1：mul_v / sub_v 未广播（致命，P0）

**位置**：`src/data/fused_normalization.cpp` 第 536-537 行、542-543 行

**根因**：
- C=1 AMP 的 `simd_process_4pixels_c1` 一次处理 **4 个独立像素**（lane[0]=pixel0, lane[1]=pixel1, lane[2]=pixel2, lane[3]=pixel3）。
- 当前代码 `_mm_set_ps(0,0,0,mul[0])` 仅向 lane[0] 填充因子，lane[1..3] 为 0。
- 导致 pixel 1-3 的乘减运算结果为 0，FP16 输出强制归零。

**数据验证**：
```
raw dump first 8 u16: b6ca 0 0 0 0 0 0 0
→ pixel 0 (lane[0]) = b6ca (-0.4243) ✓
→ pixel 1-3 (lane[1..3]) = 0x0000 (乘0归零) ✗
```

### 问题 2：simd_process_4pixels_c1 超量写入（P1，潜在风险）

**位置**：`src/data/fused_normalization.cpp` 第 116-123 行

**根因**：
- `_mm_cvtps_ph` 仅填充寄存器低 64 位（4 个 FP16），高 64 位为 0。
- `t1 = _mm_unpackhi_epi16(h, zero)` 因此为全 0。
- 后两条 `_mm_storeu_si128(dst+16, dst+24)` 写入 16 个全 0 `uint16_t`，完全冗余。
- 函数实际写入 **32** 个 `uint16_t`，但 4 像素×4 通道仅需 **16** 个。
- 外层 `simd_row_c1_noflip` 按 `dst + w*4`（16 个 `uint16_t`）步进，当前 MNIST(W=28)/CIFAR(W=32) 均为 4 的倍数，冗余零会被下一组覆盖，**暂不触发实际错误**。
- **但若图像宽度非 4 的倍数**（如 W=30），最后一个 SIMD 块的全 0 会覆盖后续 `scalar_process_pixel_c1` 回退写入的尾部像素，造成静默数据损坏。

### 问题 3：DIAG-WORKER 类型双关（P2，调试误导）

**位置**：`src/data/preprocess_worker.cpp` 第 1221 行

**根因**：
- AMP 模式下 `final_output_ptr` 指向 `uint16_t` 缓冲区。
- 调试代码 `float* fout = reinterpret_cast<float*>(final_output_ptr)` 属于未定义行为（type punning）。
- 当前结果碰巧落在合理区间，但可能在不同编译器/优化级别下产生完全错误的数值，误导调试方向。

---

## 三、修复方案（已验证正确性）

### Fix-1：mul_v / sub_v 广播（必须）

**文件**：`src/data/fused_normalization.cpp`  
**行号**：536-537、542-543

```cpp
// Before
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);

// After
__m128 mul_v = _mm_set1_ps(mul[0]);   // 广播到 4 个 lane
__m128 sub_v = _mm_set1_ps(sub[0]);
```

**验证**：
- `_mm_set1_ps(x)` → lane[0..3] 均为 x。
- 4 个独立像素均得到相同归一化因子，pixel 1-3 不再归零。

### Fix-2：删除冗余 store（推荐）

**文件**：`src/data/fused_normalization.cpp`  
**行号**：116-123

```cpp
// Before
__m128i zero = _mm_setzero_si128();
__m128i t0 = _mm_unpacklo_epi16(h, zero);
__m128i t1 = _mm_unpackhi_epi16(h, zero);   // ← t1 为全 0，多余

_mm_storeu_si128(reinterpret_cast<__m128i*>(dst),      _mm_unpacklo_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),  _mm_unpackhi_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), _mm_unpacklo_epi32(t1, zero));  // ← 全 0，多余
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 24), _mm_unpackhi_epi32(t1, zero));  // ← 全 0，多余

// After
__m128i zero = _mm_setzero_si128();
__m128i t0 = _mm_unpacklo_epi16(h, zero);   // [fp0,0, fp1,0, fp2,0, fp3,0] as 32-bit lanes

_mm_storeu_si128(reinterpret_cast<__m128i*>(dst),
                 _mm_unpacklo_epi32(t0, zero));   // dst[0..7]  = [fp0,0,0,0, fp1,0,0,0]
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),
                 _mm_unpackhi_epi32(t0, zero));   // dst[8..15] = [fp2,0,0,0, fp3,0,0,0]
```

**布局验证**：
- `_mm_unpacklo_epi16(h, zero)` 将 4 个 FP16 零扩展为 `[fp0,0, fp1,0, fp2,0, fp3,0]`（32-bit lanes）。
- `_mm_unpacklo_epi32(t0, zero)` 提取 lane[0], lane[1] 并各自拼接 0 → `[fp0,0,0,0, fp1,0,0,0]`。
- `_mm_unpackhi_epi32(t0, zero)` 提取 lane[2], lane[3] 并各自拼接 0 → `[fp2,0,0,0, fp3,0,0,0]`。
- 输出恰好 16 个 `uint16_t`，与 `dst + w*4` 步进严格对齐，消除尾部覆盖风险。

### Fix-3：DIAG-WORKER 增加 AMP 分支（建议）

**文件**：`src/data/preprocess_worker.cpp`  
**行号**：1219-1254 附近

```cpp
// Before
float* fout = reinterpret_cast<float*>(final_output_ptr);

// After
if (using_amp_) {
    const std::uint16_t* fout16 = reinterpret_cast<const std::uint16_t*>(final_output_ptr);
    for (int i = 0; i < 784; ++i) {
        float v = half_to_float(fout16[i * 4]);  // 只读通道 0
        // ... 统计逻辑不变
    }
} else {
    float* fout = reinterpret_cast<float*>(final_output_ptr);
    // ... 原有逻辑
}
```

---

## 四、影响范围

| 场景 | 影响 | 说明 |
|------|------|------|
| MNIST AMP | ✅ 受影响 | Bug #1 导致每 4 像素仅第 1 个正确；修复后应得到 `b6ca,0,0,0,b6ca,0,0,0,...` |
| CIFAR10 AMP | ❌ 不受影响 | C=3 路径 `simd_process_2pixels_c3` 正确，mul_v[3]=0 是设计意图（padding 通道） |
| ImageNet AMP | ❌ 不受影响 | C=3，同上 |
| 所有 FP32 | ❌ 不受影响 | 不走 AMP/SIMD 路径 |
| 非 AVX2 | ❌ 不受影响 | C=1 AMP 直接报 `TR_NOT_IMPLEMENTED` |
| W%4≠0 数据集 | ⚠️ 潜在风险 | Bug #2 会在非 4 倍数宽度下触发尾部覆盖 |

---

## 五、验证步骤

1. **编译**：修改 `fused_normalization.cpp` 后重新编译。
2. **MNIST AMP**：运行 `test_two_batch_correction --amp --dataset mnist`，确认 raw dump 变为 `b6ca 0 0 0 b6ca 0 0 0 ...`，且 `first`/`last` 均非零。
3. **CIFAR AMP**：运行 `test_two_batch_correction --amp --dataset cifar10`，确认 CIFAR 数据不受影响，继续保持 PASS。
4. **回归测试**：运行 CPU/GPU 模式下的 MNIST/CIFAR，确保 FP32 路径不受影响。
