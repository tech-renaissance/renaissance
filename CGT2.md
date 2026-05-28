# CIFAR10 AMP `data_b` 验证失败 —— 深度代码检查与修改意见

> 基于 CHK.md 中 【小伙伴D】的关键发现，对 `fetch_from_rank`、Tensor 内存模型、`simd_process_2pixels_c3` 及测试代码进行进一步检查。

---

## 一、现象回顾

CIFAR10 AMP（`--amp --dataset cifar10`）模式下 `test_two_batch_correction` 出现概率性失败：
- 5 次运行中 3 次失败，失败位置随机（`data_a`/`data_b`、`first`/`last` 交替故障）。
- **核心矛盾**：`verify_first_last_pixel` 报告 `first=-nan(ind)`，但后续 raw hex dump 打印同一块 Tensor 的前 8 个 `u16` 却显示有效非零值（如 `3b54 b8a2 b634 0...`）。
- 小伙伴D 发现更极端案例：某次运行中 raw hex 显示前两个 `u16` 为 `0xbfe5`、`0xbfdb`（FP16 NaN 编码），但 `verify` 报告 `first=-1.40723`（有效有限值）。

**结论：同一块 `Tensor` 内存，在 `verify` 和 `raw dump` 两次读取之间，内容发生了改变，或两次读取看到的是不同的物理内存。**

---

## 二、`fetch_from_rank` 与 Tensor 内存模型深度检查

### 2.1 `fetch_from_rank` 实现

```cpp
Tensor TaskBase::fetch_from_rank(const DTensor& dt, int rank) {
    const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);
    const void* src = ArenaKeeper::instance().ptr_at(rank, static_cast<size_t>(live_dt.offset()));
    const uint64_t valid_bytes = dt.nbytes();

    if (dt.is_compact()) {
        Tensor result(dt.shape, dt.dtype);
        cudaMemcpy(result.data<void>(), src, valid_bytes, cudaMemcpyDeviceToHost);
        return result;
    } else {
        // 非紧凑路径...（AMP 下不走此分支）
    }
}
```

**检查结论：**
1. AMP 模式下 `is_compact()` 为 `true`（stride = `[4096,128,4,1]`），走 compact 路径。
2. `src` 使用 `live_dt.offset()`（运行时最新 offset），`valid_bytes` 和 `result.shape` 使用传入的 `dt`。
3. `cudaMemcpy` 是**同步调用**（非 Async），函数返回时 D2H 传输已完成。
4. `Tensor result` 通过 `return result` 触发移动构造（或 NRVO），测试代码中 `Tensor t_data_a = task.fetch_from_rank(...)` 正确获得所有权。

### 2.2 Tensor 构造函数与内存分配

```cpp
Tensor::Tensor(const Shape& shape, DType dtype) {
    nbytes_ = static_cast<size_t>(num) * elem_size;
    cudaError_t err = cudaMallocHost(&ptr_, nbytes_);
    // 无 memset！无共享缓冲池！
}
```

**检查结论：**
1. `cudaMallocHost` 独立分配页锁定内存，**不存在内部缓冲池或内存复用机制**。
2. `ptr_` 在移动构造/赋值后原对象被置 `nullptr`，无双重释放风险。
3. **关键缺陷：构造函数没有 `memset(0)`**。如果 `cudaMemcpy` 只拷贝了部分数据，或拷贝前内存中有垃圾值，未覆盖区域将保留未初始化内容。
4. 但 `valid_bytes = dt.nbytes()` 与 `result` 分配的 `nbytes_` 完全相等，不存在"部分拷贝"问题。

### 2.3 关于 "数据在两次读取之间改变" 的根因推断

从代码逻辑上，**同一块 `cudaMallocHost` 内存被两个连续 `data<uint16_t>()` 调用读出不同值，在单线程程序中理论上不可能发生**，除非：

| 假设 | 代码层面验证 | 可能性评估 |
|------|-------------|-----------|
| 共享缓冲池复用 | `cudaMallocHost` 独立分配，无池化 | ❌ 排除 |
| `cudaMemcpy` 异步未完成 | 使用的是同步 `cudaMemcpy` | ❌ 排除 |
| Tensor 移动语义错误 | 已检查，实现正确 | ❌ 排除 |
| `data<T>()` 返回不稳定指针 | 直接返回 `ptr_`，无间接层 | ❌ 排除 |
| **Device 端数据本身不稳定** | `g_xfer` 捕获的 H2D graph 可能涉及越界/UB | ⚠️ **最可能** |
| **CPU 端 `fp16_to_f32` 计算污染** | `std::pow` 可能受 FPU 异常标志影响 | ⚠️ 可能 |
| **MSVC 编译器优化导致读取错乱** | 越界读取触发 UB，优化后行为非确定性 | ⚠️ 可能 |

**最可能的根本链条：**

C=3 SIMD 预处理代码中的**越界读取**（见第3节）属于**未定义行为(UB)**。虽然从指令级看越界字节被后续计算"丢弃"，但在 MSVC `/O2` 优化下，编译器可能基于"此内存区域只包含图像数据"的假设进行激进优化（如向量化、寄存器重排、死存储消除）。当实际运行时读到未初始化内存（最后一行越界到图像外），优化后的代码可能产生非确定性结果——**某些像素被写入 NaN/Inf，某些没有被写入，且每次运行的模式不同**。

这导致：
- Device 端 `data_a` / `data_b` 的内容本身已经是**概率性损坏**的。
- `cudaMemcpy` 忠实地将损坏数据拷贝到 Host。
- `verify` 读取 pixel 0（或 last pixel）时，恰好读到 NaN。
- `raw dump` 打印前 8 个 `u16` 时，由于读取顺序或内存布局，看到的可能是另一批值（或者由于编译器优化，`verify` 的 `fp16_to_f32` 和 `raw dump` 的直接打印走了不同的指令路径，导致看到的原始值不同）。

**注意**：小伙伴D 观察到的 "verify 读出的值与 raw hex 不同" 也可能是 `fp16_to_f32` 函数本身的问题——如果 `std::pow` 在特定 FPU 状态下返回 NaN，则 `verify` 报告 NaN，而 `raw dump` 直接打印 `uint16_t` 不受影响。

---

## 三、`simd_process_2pixels_c3` 指令级检查

### 3.1 代码

```cpp
inline void simd_process_2pixels_c3(const std::uint8_t* p, std::uint16_t* dst,
                                     __m128 mul_v, __m128 sub_v) noexcept {
    __m128i u8x8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));  // 加载 8 字节
    __m128i i32_0 = _mm_cvtepu8_epi32(u8x8);                              // [R0,G0,B0,R1] -> 4×i32
    __m128 f0 = _mm_cvtepi32_ps(i32_0);
    f0 = _mm_mul_ps(f0, mul_v);  // mul_v = [mul0, mul1, mul2, 0]
    f0 = _mm_sub_ps(f0, sub_v);
    __m128i h0 = _mm_cvtps_ph(f0, 0);                                     // pixel 0: [R0',G0',B0',0]

    __m128i shifted = _mm_srli_si128(u8x8, 3);                            // 丢弃低 3 字节
    __m128i i32_1 = _mm_cvtepu8_epi32(shifted);                           // [R1,G1,B1,R2] -> 4×i32
    __m128 f1 = _mm_cvtepi32_ps(i32_1);
    f1 = _mm_mul_ps(f1, mul_v);
    f1 = _mm_sub_ps(f1, sub_v);
    __m128i h1 = _mm_cvtps_ph(f1, 0);                                     // pixel 1: [R1',G1',B1',0]

    __m128i h01 = _mm_unpacklo_epi64(h0, h1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), h01);               // 写入 2 像素
}
```

### 3.2 越界读取分析（确定性 Bug）

CIFAR10 图像 W=32，C=3，每行字节数 = 96。

`simd_row_c3_noflip` 循环：`w = 0, 2, 4, ..., 30`。

当 `w = 30`（最后一对像素）：
- `p = src + 30 * 3 = src + 90`
- `_mm_loadl_epi64(p)` 读取 8 字节：`src[90..97]`
- 一行只有 96 字节（`src[0..95]`）
- **`src[96..97]` 是下一行（或图像外）的 2 字节**

对于最后一行（row = 31），`src[96..97]` 是**图像缓冲区外的未初始化内存**。

### 3.3 越界字节是否影响输出？

从指令级追踪：
- `u8x8` 低 64 位 = `[R30,G30,B30,R31,G31,B31,garbage0,garbage1]`
- `i32_0` 提取低 4 字节 = `[R30,G30,B30,R31]` → **garbage 未被使用**
- `shifted` 右移 3 字节，低 4 字节 = `[R31,G31,B31,garbage0]`
- `i32_1` 提取 `[R31,G31,B31,garbage0]`
- `f1` 的 lane 3 = `garbage0 * mul[3] - sub[3] = garbage0 * 0 - 0 = 0`
- **garbage0 被乘以 0 归零，garbage1 被 `_mm_srli_si128` 丢弃**

**表面结论**：越界字节不影响最终写入 `dst` 的数值。

**但**：`_mm_loadl_epi64` 跨越分配边界读取未初始化内存，属于**严格未定义行为(C++ UB)**。MSVC 在 `/O2` 优化下可能：
- 假设 `p` 指向的 8 字节内存是"合法且确定性的"
- 基于此假设进行指令重排、向量化展开、或内存别名分析
- 当实际运行读到随机值（甚至触发页保护/ASAN 外的微妙行为）时，生成的代码可能产生完全非预期的结果

**在 Windows + MSVC + Release 模式下，这种 UB 完全可能解释概率性 NaN/Inf 的出现。**

### 3.4 关于"通道错配"的重新评估

小伙伴D 提出"奇数像素 R/G/B 错配一个字节"。

经指令级逐条验证：
- `_mm_srli_si128(u8x8, 3)` 丢弃 `[R0,G0,B0]`，从 `R1` 开始。
- `i32_1` 提取 `[R1,G1,B1,R2]`。
- `f1` 的 4 个 lane 分别乘以 `[mul0, mul1, mul2, 0]`，得到 `[R1*mul0, G1*mul1, B1*mul2, 0]`。
- **奇数像素的 R/G/B 通道没有错配**。

**结论**："通道错配"不成立。但"行末越界读取"是**确定存在的 UB**。

---

## 四、测试端 `fp16_to_f32` 的检查

测试代码和 `tensor.cpp` 中均使用以下函数解码 FP16：

```cpp
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;
    if (exponent == 0) { ... }
    if (exponent == 0x1Fu) { return (mantissa == 0) ? INF : NAN; }
    float v = std::pow(2.0f, static_cast<float>(exponent - 15)) * (1.0f + mantissa / 1024.0f);
    return sign ? -v : v;
}
```

**问题分析：**
1. `std::pow(2.0f, x)` 在 MSVC 实现中可能查询或修改 FPU 控制字/状态字。
2. 如果框架内部（CUDA kernel、其他 CPU SIMD 代码、或之前的 `std::pow` 调用）触发了浮点异常（如除以零、无效操作），**FPU Invalid Operation 标志可能残留**。
3. 某些 MSVC CRT 的 `pow` 实现对 FPU 状态敏感，可能在 flags 被设置时返回 `NaN`——即使输入本身完全合法。
4. 这可以解释 **raw dump（直接打印 uint16_t）正常，但 verify（经过 `fp16_to_f32`）得到 NaN** 的现象。

---

## 五、根因综合判定

CIFAR10 AMP 失败是**多重因素叠加**的概率性问题，与本次 C=1 修复完全无关（CIFAR10 走 C=3 路径）。

| 因素 | 确定性 | 对失败的影响 |
|------|--------|-------------|
| C=3 SIMD 行末越界读取（UB） | ✅ 确定存在 | **高**：导致 Device 端数据概率性损坏，产生 NaN/Inf 像素 |
| `fp16_to_f32` 使用 `std::pow`（FPU 状态污染） | ⚠️ 高度可能 | **高**：导致 Host 端解码时概率性产生 NaN，即使原始 hex 正常 |
| fetch 后数据不一致（raw vs verify） | ✅ 观察到 | 是上述两因素的症状，而非独立根因 |
| Tensor 内存池复用 / 异步传输 | ❌ 已排除 | 无影响 |

---

## 六、修改意见（不改代码）

### P0：修复 C=3 SIMD 越界读取（最紧迫）

**文件**：`src/data/fused_normalization.cpp`

**问题函数**：`simd_process_2pixels_c3`、`simd_process_2pixels_c3_flip`

**修改方案二选一：**

**方案 A（推荐，最小侵入）：**
在行循环中为每行末尾添加 2 字节 padding，确保 `_mm_loadl_epi64` 永远不会越界。
- 但这需要修改图像缓冲区的分配逻辑，侵入性较大。

**方案 B（安全且简单）：**
修改 `simd_row_c3_noflip`，对行末的最后一对像素（当 `W` 为偶数时 `w = W-2`）使用**标量处理**，避免 SIMD 越界读取。

```cpp
// 在 simd_row_c3_noflip 中
std::size_t w = 0;
for (; w + 3 < W; w += 2) {           // 改为 w + 3 < W，确保至少剩 4 像素才用 SIMD
    simd_process_2pixels_c3(src + w * 3, dst + w * 4, mul_v, sub_v);
}
for (; w < W; ++w) {                  // 尾部全部标量处理
    scalar_process_pixel_c3(src + w * 3, dst + w * 4, mul, sub);
}
```

但这样会损失最后 2-3 像素的 SIMD 性能。更精细的做法：

```cpp
std::size_t w = 0;
for (; w + 1 < W; w += 2) {
    // 安全边界检查：如果 w == W-2，使用标量避免越界
    if (w + 2 == W) {
        scalar_process_pixel_c3(src + w * 3, dst + w * 4, mul, sub);
        scalar_process_pixel_c3(src + (w + 1) * 3, dst + (w + 1) * 4, mul, sub);
        break;
    }
    simd_process_2pixels_c3(src + w * 3, dst + w * 4, mul_v, sub_v);
}
```

同理修改 `simd_row_c3_flip`。

### P1：将 `fp16_to_f32` 改为纯位运算（消除 FPU 污染）

**文件**：`tests/correction/test_two_batch_correction.cpp`（测试端）、`src/tensor/tensor.cpp`（框架端，可选）

**修改**：替换 `std::pow` 为 IEEE 754 位组装：

```cpp
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f32_bits;
    if (exp == 0) {
        if (mant == 0) {
            f32_bits = sign << 31;
        } else {
            // 非规格化数
            float v = static_cast<float>(mant) / 1024.0f * 0x1.0p-14f;
            return sign ? -v : v;
        }
    } else if (exp == 0x1F) {
        f32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        uint32_t f_exp = (exp + 127 - 15) << 23;
        uint32_t f_mant = mant << 13;
        f32_bits = (sign << 31) | f_exp | f_mant;
    }
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    return f;
}
```

**优点**：
- 完全消除 `std::pow` 调用，排除 FPU 状态污染。
- 解码结果与硬件 F16C 指令完全一致。
- 性能更高（无函数调用、无浮点运算）。

### P2：增加 fetch 后诊断校验（定位用，可后续移除）

**文件**：`tests/correction/test_two_batch_correction.cpp`

在 `verify_first_last_pixel` 内部增加 inline raw dump：

```cpp
static bool verify_first_last_pixel(const Tensor& data, const char* name) {
    // ... 现有代码 ...
    
    // 诊断：在计算前直接打印原始 hex
    if (data.dtype() == DType::FP16) {
        const uint16_t* p = data.data<uint16_t>();
        std::cerr << "[DIAG " << name << "] raw hex first 8: ";
        for (int i = 0; i < 8; ++i) std::cerr << std::hex << p[i] << " ";
        std::cerr << std::dec << std::endl;
    }
    
    // ... 现有计算 ...
}
```

这样可以**在同一个函数调用内**确认 `data<uint16_t>()` 返回的指针和值，排除"两次读取之间内存被修改"的猜测。

### P3：检查 MSVC 浮点编译选项

**文件**：`CMakeLists.txt` 或编译配置

确认当前是否使用了 `/fp:fast`。如果是，建议改为 `/fp:precise` 或 `/fp:strict`。

- `/fp:fast` 允许编译器进行激进的浮点优化，包括假设无 NaN/Inf、忽略 FPU 异常标志等。在存在 UB 的代码中，这会放大非确定性。
- `/fp:precise` 是更安全的默认值。

### P4：`fetch_from_rank` 增加显式 device 同步（防御性）

**文件**：`src/task/task_base.cpp`

在 `cudaMemcpy` 后增加 `cudaDeviceSynchronize()`（或至少 `cudaStreamSynchronize(0)`）：

```cpp
cudaMemcpy(result.data<void>(), src, valid_bytes, cudaMemcpyDeviceToHost);
// 防御性同步，确保数据完全落位到 host 内存
cudaDeviceSynchronize();
```

虽然同步 `cudaMemcpy` 理论上已经保证完成，但在多 stream、多线程或 driver 层面存在异步优化时，显式同步可以排除边缘情况。

---

## 七、优先级建议

| 优先级 | 修改项 | 预期效果 | 工作量 |
|--------|--------|---------|--------|
| **P0** | 修复 C=3 SIMD 越界读取 | 消除 Device 端数据损坏的 UB 根因 | 小（修改行末处理逻辑） |
| **P1** | `fp16_to_f32` 改为位运算 | 消除 Host 端 NaN 误报 | 小（替换函数实现） |
| **P2** | 增加 inline 诊断打印 | 确认/排除 fetch 后数据不一致 | 极小 |
| **P3** | 检查 `/fp:fast` | 减少编译器优化引入的 UB | 极小 |
| **P4** | `fetch_from_rank` 显式同步 | 防御性增强 | 极小 |

**建议执行顺序**：P2（诊断确认）→ P0 + P1（并行修复根因）→ P3 + P4（防御性增强）。
