# PreprocessOperation (PO) 设计文档

**版本**: V1.0.0
**日期**: 2026-02-17
**作者**: 技术觉醒团队
**状态**: 已实现 (P0阶段)

---

## 【目录】

1. [概述](#概述)
2. [设计原则](#设计原则)
3. [核心接口详解](#核心接口详解)
4. [解码策略](#解码策略)
5. [实现规范](#实现规范)
6. [示例实现](#示例实现)
7. [性能优化](#性能优化)
8. [测试指南](#测试指南)

---

## 【概述】

PreprocessOperation (PO) 是 renAIssance 框架中图像预处理操作的抽象基类，定义了所有预处理操作必须遵循的接口规范。PO 设计遵循**轻量级、可克隆、无状态共享**的原则，作为 PreprocessWorker (PW) 的基本执行单元。

### 核心特性

- **统一接口**：所有预处理操作通过 `execute()` 方法执行
- **类型安全**：编译期类型检查，避免运行时错误
- **性能优化**：支持 Simd 加速库，提供缓存机制
- **JPEG 解码集成**：通过 `DecodeStrategy` 与 TurboJPEG 解码器协同工作

### 已实现的 PO 子类

| 类名 | 功能 | 随机性 | 解码策略 |
|------|------|--------|---------|
| `Resize` | 图像缩放（双线性插值） | 无 | 完整解码 |
| `CenterCrop` | 中心裁剪（PyTorch 兼容） | 无 | 局部解码 |
| `DoNothing` | 占位操作（直接复制） | 无 | 不解码 |

---

## 【设计原则】

### 1. 轻量级（Lightweight）

PO 仅持有**参数**，不持有大块内存：

```cpp
class Resize : public PreprocessOperation {
private:
    int output_size_;  // ✓ 仅持有参数
    // ✗ 不持有输入/输出 buffer
};
```

**原因**：
- 每个 PW 持有一组 PO 的副本（train + val），轻量级设计减少内存开销
- 输入/输出 buffer 由 PW 统一管理（D/A/B/T/S/C/E 区）

### 2. 可克隆（Clonable）

每个 PW 持有**独立的 PO 副本**：

```cpp
std::unique_ptr<PreprocessOperation> clone() const = 0;
```

**实现示例**：
```cpp
std::unique_ptr<PreprocessOperation> Resize::clone() const {
    return std::make_unique<Resize>(output_size_);
}
```

**原因**：
- 避免多线程共享状态导致的缓存冲突
- 每个 PW 可以独立缓存 Simd 上下文（如 `Resize::resizer_cache_`）

### 3. 无状态共享（Stateless）

同一 PO 多次调用 `execute()` 结果一致（给定相同 `rng` 状态）：

```cpp
// 确定性操作：相同输入 → 相同输出
po->execute(input, ...);  // 输出 A
po->execute(input, ...);  // 输出 A（相同）

// 随机性操作：通过 rng 控制随机性
po->execute(input, ..., rng1);  // 输出 A
po->execute(input, ..., rng2);  // 输出 B（不同 rng 状态）
```

**原因**：
- 保证训练可复现性
- 便于调试和单元测试

### 4. 性能优化（Performance Oriented）

支持缓存 Simd 上下文：

```cpp
class Resize : public PreprocessOperation {
private:
    mutable void* resizer_cache_ = nullptr;  // Simd Resizer 缓存
    mutable int cached_src_w_ = 0;
    mutable int cached_src_h_ = 0;
    // ...
};
```

**性能提升**：
- Resize 缓存命中时性能提升 **30%+**
- 224×224 输出，约 **0.5ms/image** (AVX2)

---

## 【核心接口详解】

### execute() - 核心执行接口

```cpp
virtual void execute(
    const uint8_t* input_ptr,      // 输入图像数据（RGB uint8，值域 0-255）
    int32_t input_width,            // 输入宽度
    int32_t input_height,           // 输入高度
    size_t input_stride,            // 输入行步长（字节）← 关键：Simd 必需
    uint8_t* output_ptr,            // 输出图像数据（预分配）
    int32_t& output_width,          // [输出] 输出宽度
    int32_t& output_height,         // [输出] 输出高度
    size_t output_stride,           // 输出行步长（字节）← 关键：Simd 必需
    Generator* rng = nullptr        // 随机数生成器（可选，仅随机操作使用）
) = 0;
```

### 关键设计点

#### 1. Stride 参数（行步长）

**定义**：一行像素数据的字节大小，包括填充。

**作用**：
- Simd 库要求 64 字节对齐，`stride` 通常 `> width * 3`
- 支持 SIMD 向量化操作，提升性能

**计算方式**：
```cpp
size_t stride = PreprocessOperation::calculate_stride(width, 3);
// 对于 width=224，channels=3：
// raw_stride = 224 * 3 = 672
// stride = align_up(672, 64) = 704
```

**内存布局示例**：
```
Width=224, Channels=3, Stride=704 (64-byte aligned)

|<- 224 * 3 = 672 bytes ->|<- 32 bytes padding ->|
+------------------------+------------------------+
|  Pixel Data (RGB)      |  Padding (Simd 对齐)  |
+------------------------+------------------------+
  ^input_ptr            ^input_ptr + 672        ^input_ptr + 704 (下一行)
```

#### 2. 预分配输出 Buffer

**设计**：`output_ptr` 由调用者预分配，PO 内部不分配内存。

**优点**：
- 避免 PO 内部内存分配，提升性能
- 统一内存管理（由 PW 负责）
- 支持 zero-copy 操作（A/B 区乒乓）

**实现要求**：
```cpp
// ✅ 正确：直接写入预分配的 buffer
std::memcpy(output_ptr + y * output_stride, src, row_bytes);

// ✗ 错误：PO 内部分配内存
uint8_t* temp = new uint8_t[size];  // 违反设计原则
```

#### 3. 输出尺寸通过引用返回

```cpp
int32_t& output_width;
int32_t& output_height;
```

**原因**：
- 输出尺寸可能不等于 `output_size_`（特殊情况）
- 例如：CenterCrop 在 PyTorch 兼容模式下，输出始终是 `output_size_ × output_size_`

### clone() - 深拷贝接口

```cpp
virtual std::unique_ptr<PreprocessOperation> clone() const = 0;
```

**作用**：
- 每个 PW 持有独立的 PO 副本
- 避免多线程共享状态

**实现模板**：
```cpp
std::unique_ptr<PreprocessOperation> YourPO::clone() const {
    return std::make_unique<YourPO>(param1_, param2_, ...);
}
```

### 元信息查询接口

```cpp
virtual std::string name() const = 0;
virtual bool introduce_randomness() const = 0;
virtual bool is_crop() const { return false; }
virtual bool is_resize() const { return false; }
virtual bool is_random_horizontal_flip() const { return false; }
virtual bool require_temp() const { return false; }
```

| 方法 | 作用 | 典型用途 |
|------|------|---------|
| `name()` | 返回操作名称 | 日志输出、调试信息 |
| `introduce_randomness()` | 是否引入随机性 | CPVS 优化判断 |
| `is_crop()` | 是否为裁剪操作 | 渐进式分辨率、解码策略 |
| `is_resize()` | 是否为缩放操作 | 渐进式分辨率、解码策略 |
| `is_random_horizontal_flip()` | 是否为随机翻转 | Preprocessor 排序优化 |
| `require_temp()` | 是否需要暂存区 | PW 内存分配（T 区） |

### 动态参数更新接口

```cpp
virtual void set_output_size(int size) { (void)size; }
virtual int get_output_size() const { return 0; }
```

**用途**：渐进式分辨率训练（Progressive Resolution Training）

**调用时机**：每个 phase（train/val）开始时，通过 `update_parameters()` 更新。

**实现示例**：
```cpp
void Resize::set_output_size(int size) {
    if (size != output_size_) {
        output_size_ = size;
        // 清空缓存，下次 execute 时会重建
        if (resizer_cache_) {
            SimdRelease(resizer_cache_);
            resizer_cache_ = nullptr;
        }
    }
}
```

---

## 【解码策略】

### DecodeStrategy 结构体

```cpp
struct DecodeStrategy {
    bool need_decode = false;       // 是否需要解码（非 ImageNet 为 false）
    bool use_partial = false;       // 局部解码 vs 完整解码

    // MCU 对齐的解码窗口（8 的倍数）
    int32_t decode_x = 0;           // 解码起始 X（MCU 对齐，向下取整）
    int32_t decode_y = 0;           // 解码起始 Y（MCU 对齐，向下取整）
    int32_t decode_w = 0;           // 解码宽度（MCU 对齐，向上取整）
    int32_t decode_h = 0;           // 解码高度（MCU 对齐，向上取整）

    // 精确裁剪窗口（相对于解码窗口的偏移）
    int32_t crop_x = 0;             // 裁剪起始 X（相对于 decode_x）
    int32_t crop_y = 0;             // 裁剪起始 Y（相对于 decode_y）
    int32_t crop_w = 0;             // 裁剪宽度
    int32_t crop_h = 0;             // 裁剪高度
};
```

### 解码策略接口

```cpp
virtual DecodeStrategy get_decode_strategy(
    int32_t image_width,
    int32_t image_height,
    int sdmp_factor,
    Generator* rng
) const {
    // 默认：不需要解码（非 ImageNet 或非首位）
    return DecodeStrategy{};
}
```

### 设计目的

**仅在 PO 作为首个操作时调用**，告诉 TurboJPEG 解码器：
1. **是否需要解码**（ImageNet 需要，MNIST/CIFAR 不需要）
2. **局部解码 vs 完整解码**
   - 完整解码：解码整张图（Resize 使用）
   - 局部解码：仅解码需要的区域（CenterCrop 使用）
3. **解码窗口**（MCU 对齐）和**精确裁剪窗口**

### 示例：CenterCrop 的解码策略

```cpp
DecodeStrategy CenterCrop::get_decode_strategy(
    int32_t image_width,
    int32_t image_height,
    int sdmp_factor,
    Generator* rng
) const {
    (void)sdmp_factor;
    (void)rng;

    DecodeStrategy strategy;
    strategy.need_decode = true;
    strategy.use_partial = true;  // 局部解码

    // 计算中心裁剪区域
    int crop_w = std::min(image_width, output_size_);
    int crop_h = std::min(image_height, output_size_);
    int crop_x = (image_width - crop_w) / 2;
    int crop_y = (image_height - crop_h) / 2;

    // MCU 对齐解码窗口
    int mcu_x = align_down_mcu(crop_x);
    int mcu_y = align_down_mcu(crop_y);
    int mcu_x_end = align_up_mcu(crop_x + crop_w);
    int mcu_y_end = align_up_mcu(crop_y + crop_h);

    int decode_w = mcu_x_end - mcu_x;
    int decode_h = mcu_y_end - mcu_y;

    // 边界检查
    if (mcu_x + decode_w > image_width) {
        decode_w = image_width - mcu_x;
    }
    if (mcu_y + decode_h > image_height) {
        decode_h = image_height - mcu_y;
    }

    strategy.decode_x = mcu_x;
    strategy.decode_y = mcu_y;
    strategy.decode_w = decode_w;
    strategy.decode_h = decode_h;

    // 精确裁剪窗口（相对偏移）
    strategy.crop_x = crop_x - mcu_x;
    strategy.crop_y = crop_y - mcu_y;
    strategy.crop_w = crop_w;
    strategy.crop_h = crop_h;

    return strategy;
}
```

### 性能对比

| 操作 | 解码策略 | 解码时间 (1822×1024→224×224) | 加速比 |
|------|---------|---------------------------|--------|
| Resize | 完整解码 | 41.95 ms | 1.0× |
| CenterCrop | 局部解码 | 9.08 ms | **4.6×** |

---

## 【实现规范】

### 1. 文件结构

```
include/renaissance/data/
├── preprocess_operation.h    # PO 抽象基类
├── decode_strategy.h          # 解码策略结构体
├── resize.h                   # Resize 类（示例）
└── center_crop.h              # CenterCrop 类（示例）

src/data/
├── preprocess_operation.cpp   # PO 基类实现（空实现）
├── resize.cpp                 # Resize 实现
└── center_crop.cpp            # CenterCrop 实现
```

### 2. 头文件模板

```cpp
/**
 * @file your_operation.h
 * @brief 操作简短描述
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include <algorithm>  // 根据需要添加其他头文件

namespace tr {

/**
 * @class YourOperation
 * @brief 详细描述
 *
 * 功能：
 * - 功能点 1
 * - 功能点 2
 *
 * 性能：
 * - 性能指标（如有）
 */
class YourOperation : public PreprocessOperation {
public:
    // 构造函数
    explicit YourOperation(/* 参数 */);

    // 核心接口
    void execute(...) override;
    std::unique_ptr<PreprocessOperation> clone() const override;

    // 元信息
    std::string name() const override;
    bool introduce_randomness() const override;
    bool is_crop() const override;  // 根据实际情况实现
    bool is_resize() const override;

    // 动态参数（如需要）
    void set_output_size(int size) override;
    int get_output_size() const override;

    // 解码策略（如作为首位操作）
    DecodeStrategy get_decode_strategy(...) const override;

private:
    // 成员变量（仅参数，不持有大块内存）
    int param1_;
    // ...
};

} // namespace tr
```

### 3. 实现文件模板

```cpp
/**
 * @file your_operation.cpp
 * @brief 操作实现
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 */

#include "renaissance/data/your_operation.h"
#include "renaissance/base/logger.h"  // 如需日志
#include <cstring>  // 如需 memcpy
#include <algorithm>  // 如需 std::min/max

namespace tr {

void YourOperation::execute(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    int32_t& output_width,
    int32_t& output_height,
    size_t output_stride,
    Generator* rng
) {
    (void)rng;  // 如不使用随机数

    // Step 1: 设置输出尺寸
    output_width = /* 计算输出宽度 */;
    output_height = /* 计算输出高度 */;

    // Step 2: 执行操作
    // ...

    // Step 3: 写入输出（注意 stride）
    // std::memcpy(output_ptr + y * output_stride, src, row_bytes);
}

std::unique_ptr<PreprocessOperation> YourOperation::clone() const {
    return std::make_unique<YourOperation>(param1_);
}

std::string YourOperation::name() const {
    return "YourOperation";
}

bool YourOperation::introduce_randomness() const {
    return /* true 或 false */;
}

// 其他方法实现...

} // namespace tr
```

### 4. 关键注意事项

#### ✅ 正确做法

1. **仅持有参数**：
```cpp
class YourPO {
private:
    int output_size_;  // ✓ 参数
};
```

2. **使用预分配 buffer**：
```cpp
void execute(..., uint8_t* output_ptr, ...) {
    // ✓ 直接写入预分配的 buffer
    std::memcpy(output_ptr, src, size);
}
```

3. **处理 stride**：
```cpp
for (int y = 0; y < height; ++y) {
    std::memcpy(output_ptr + y * output_stride,
               input_ptr + y * input_stride,
               width * 3);
}
```

#### ❌ 错误做法

1. **持有大块内存**：
```cpp
class YourPO {
private:
    std::vector<uint8_t> buffer_;  // ✗ 违反轻量级原则
};
```

2. **PO 内部分配内存**：
```cpp
void execute(...) {
    uint8_t* temp = new uint8_t[size];  // ✗ 应由调用者管理
}
```

3. **忽略 stride**：
```cpp
// ✗ 错误：假设紧凑存储
std::memcpy(output_ptr, input_ptr, width * height * 3);

// ✓ 正确：处理 stride
for (int y = 0; y < height; ++y) {
    std::memcpy(output_ptr + y * output_stride,
               input_ptr + y * input_stride,
               width * 3);
}
```

---

## 【示例实现】

### 示例 1: Resize（图像缩放）

**功能**：使用 Simd 库进行双线性插值缩放。

**核心特性**：
- 缓存 Simd Resizer 上下文（性能优化）
- 支持动态分辨率（渐进式训练）
- 必须完整解码

**实现要点**：
```cpp
void Resize::execute(...) {
    output_width = output_size_;
    output_height = output_size_;

    // 检查缓存
    if (!resizer_cache_ || /* 尺寸变化 */) {
        if (resizer_cache_) {
            SimdRelease(resizer_cache_);
        }
        // 创建新的 resizer
        resizer_cache_ = SimdResizerInit(
            input_width, input_height,
            output_size_, output_size_,
            3,  // RGB
            SimdResizeChannelByte,
            SimdResizeMethodBilinear
        );
    }

    // 执行缩放
    SimdResizerRun(resizer_cache_,
                  input_ptr, input_stride,
                  output_ptr, output_stride);
}
```

**性能**：224×224 输出，约 0.5ms/image (AVX2)

### 示例 2: CenterCrop（中心裁剪）

**功能**：从输入图像中心裁剪指定尺寸。

**核心特性**：
- PyTorch 兼容：输入 < 输出时自动填充黑色
- 使用局部解码（性能优化）
- 纯 memcpy 操作

**实现要点**：
```cpp
void CenterCrop::execute(...) {
    output_width = output_size_;
    output_height = output_size_;

    // Step 1: 填充黑色（PyTorch 兼容）
    std::memset(output_ptr, 0, output_stride * output_size_);

    // Step 2: 计算居中位置
    int copy_w = std::min(input_width, output_size_);
    int copy_h = std::min(input_height, output_size_);
    int paste_x = (output_size_ - copy_w) / 2;
    int paste_y = (output_size_ - copy_h) / 2;
    int src_x = (input_width - copy_w) / 2;
    int src_y = (input_height - copy_h) / 2;

    // Step 3: 复制到居中位置
    const uint8_t* src_base = input_ptr + src_y * input_stride + src_x * 3;
    size_t row_bytes = copy_w * 3;

    for (int y = 0; y < copy_h; ++y) {
        std::memcpy(output_ptr + (paste_y + y) * output_stride + paste_x * 3,
                   src_base + y * input_stride,
                   row_bytes);
    }
}
```

**性能**：224×224 输出，约 0.1ms/image（纯 memcpy）

---

## 【性能优化】

### 1. Simd 上下文缓存

**适用场景**：多次使用相同参数调用同一 PO。

**实现**：
```cpp
class Resize : public PreprocessOperation {
private:
    mutable void* resizer_cache_ = nullptr;
    mutable int cached_src_w_ = 0;
    mutable int cached_src_h_ = 0;
    // ...

    void execute(...) {
        if (!resizer_cache_ ||
            cached_src_w_ != input_width ||
            cached_src_h_ != input_height) {
            // 重建缓存
        }
        // 使用缓存
    }
};
```

**性能提升**：30%+

### 2. 局部解码（Partial Decode）

**适用场景**：CenterCrop、RandomCrop 等裁剪操作。

**原理**：仅解码需要的 MCU 对齐区域，而非整张图。

**实现**：
```cpp
DecodeStrategy get_decode_strategy(...) const {
    strategy.use_partial = true;
    // 计算 MCU 对齐的解码窗口
    // ...
}
```

**性能提升**：4.6×（1822×1024→224×224）

### 3. Stride 对齐

**目的**：满足 Simd 库的 SIMD 向量化要求。

**对齐规则**：64 字节对齐。

**计算方式**：
```cpp
size_t calculate_stride(int32_t width, int32_t channels) {
    constexpr size_t ALIGNMENT = 64;
    size_t raw_stride = static_cast<size_t>(width) * channels;
    return ((raw_stride + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
}
```

---

## 【测试指南】

### 测试程序

**位置**：`tests/pw/test_po.cpp`

**功能**：
- 使用 TurboJPEG 3.x API 读取 JPEG
- 调用 PO 的 `get_decode_strategy()` 获取解码策略
- 根据解码策略执行 JPEG 解码（完整/局部）
- 执行 PO 的 `execute()` 方法
- 保存结果为 JPEG

**命令行参数**：
```bash
--input <PATH>      # 输入图片路径（默认：input.jpg）
--output <PATH>     # 输出图片路径（默认：workspace/output.jpg）
--po <NAME>         # 要测试的 PO（Resize/CenterCrop/DoNothing）
--size <N>          # PO 的输出尺寸（默认：224）
--quality <N>       # JPEG 质量用于输出（默认：90）
```

**运行示例**：
```bash
# 测试 Resize
./build/bin/tests/pw/test_po --po Resize --size 224

# 测试 CenterCrop
./build/bin/tests/pw/test_po --po CenterCrop --size 224

# 自定义输入输出
./build/bin/tests/pw/test_po --input custom.jpg --output result.jpg --po Resize --size 128
```

### 验收标准

1. **编译通过**：无编译错误
2. **功能正确**：输出图片尺寸正确，图像质量良好
3. **性能达标**：
   - Resize < 1ms/image (224×224)
   - CenterCrop < 0.2ms/image
4. **输出正确**：
   - workspace/output.jpg 成功生成
   - 图片格式正确（可用图像查看器打开）

### 性能基准（Release 模式）

| 操作 | 输入尺寸 | 输出尺寸 | 解码时间 | PO 执行时间 | 总时间 |
|------|---------|---------|---------|------------|--------|
| Resize | 1822×1024 | 224×224 | 41.95 ms | 0.45 ms | ~42.40 ms |
| CenterCrop | 1822×1024 | 224×224 | 9.08 ms | ~0 ms | ~9.08 ms |

---

**文档结束**

*本文档随代码实现同步更新。如有疑问，请参考源码或联系技术觉醒团队。*
