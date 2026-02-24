# PO（PreprocessOperation）开发指南

## 文档说明

本文档为开发新的 `PreprocessOperation` 子类提供完整指南，确保新实现的 PO 能够正确集成到 renAIssance 框架的数据加载管道中。

**版本**: 1.0.0
**日期**: 2026-02-22
**作者**: 技术觉醒团队

---

## 核心原则

### 1. 继承 PreprocessOperation 基类

所有 PO 必须继承 `PreprocessOperation` 抽象基类，并实现纯虚函数。

```cpp
class MyPreprocessOperation : public PreprocessOperation {
public:
    // 必须实现的接口
    void execute(...) override;
    std::unique_ptr<PreprocessOperation> clone() const override;
    std::string name() const override;
    bool introduce_randomness() const override;
};
```

**基类关键成员变量**（继承后自动可用）：
```cpp
protected:
    int num_channels_ = -1;              // 颜色通道数（1=灰度, 3=RGB）
    int output_size_ = -1;               // 输出尺寸
    bool rank_first_in_the_po_chain_;    // 是否为PO链首位
    size_t output_alignment_;            // 输出对齐字节数
    size_t output_stride_;               // 缓存的对齐stride
    size_t compact_output_stride_;       // 缓存的紧凑stride
    static constexpr int MCU_SIZE = 16;  // JPEG MCU大小
```

---

## 核心接口实现

### 2. 实现 execute() 方法

`execute()` 是 PO 的核心执行方法，必须正确处理所有参数。

```cpp
void execute(
    const uint8_t* input_ptr,       // 输入图像数据（RGB uint8）
    int32_t input_width,            // 输入宽度
    int32_t input_height,           // 输入高度
    size_t input_stride,            // 输入行步长（字节）← Simd必需
    uint8_t* output_ptr,            // 输出图像数据（预分配）
    int32_t& output_width,          // [输出] 输出宽度
    int32_t& output_height,         // [输出] 输出高度
    size_t& output_stride,          // [输出] 输出行步长（字节）
    Generator* rng = nullptr,       // 随机数生成器（随机操作使用）
    bool execute_from_full = false, // 解码模式标志
    bool forced_compact_output = true // 紧凑布局标志
) override;
```

**关键实现要点**：

#### 2.1 自动计算 output_stride

```cpp
// 当 output_stride == 0 时，自动计算
if (output_stride == 0) {
    if (forced_compact_output) {
        output_stride = compact_output_stride_;  // 紧凑布局
    } else {
        output_stride = output_stride_;          // 对齐布局
    }
}
```

#### 2.2 设置输出尺寸

```cpp
output_width = output_size_;
output_height = output_size_;
```

#### 2.3 处理 execute_from_full 参数

```cpp
if (execute_from_full || !rank_first_in_the_po_chain_) {
    // 模式1：从完整解码的图像中处理（使用全局坐标）
    execute_from_full_decode(input_ptr, input_width, input_height, ...);
} else {
    // 模式2：从局部解码的R2区域中处理（使用相对偏移）
    execute_from_partial_decode(input_ptr, input_stride, ...);
}
```

**设计说明**：
- `execute_from_full=false`：输入是局部解码结果（如 300x300），PO 使用内部保存的 R1 相对偏移
- `execute_from_full=true`：输入是完整图像（如 2000x2000），PO 直接使用全局坐标

---

### 3. 实现 clone() 深拷贝方法

`clone()` 必须创建完全独立的副本，确保每个 PW 持有的 PO 互不干扰。

```cpp
std::unique_ptr<PreprocessOperation> clone() const override {
    auto cloned = std::make_unique<MyPreprocessOperation>(
        // 派生类构造函数的所有参数
        output_size_, scale_min_, scale_max_, ratio_min_, ratio_max_
    );

    // 复制基类成员变量（重要！）
    cloned->num_channels_ = num_channels_;
    cloned->output_size_ = output_size_;
    cloned->output_alignment_ = output_alignment_;
    cloned->use_compact_output_as_default_ = use_compact_output_as_default_;
    cloned->output_stride_ = output_stride_;
    cloned->compact_output_stride_ = compact_output_stride_;
    cloned->rank_first_in_the_po_chain_ = rank_first_in_the_po_chain_;

    // 复制派生类成员变量
    cloned->scale_min_ = scale_min_;
    cloned->scale_max_ = scale_max_;
    // ... 其他派生类成员

    return cloned;
}
```

**关键原则**：
- 必须复制基类的**全部**成员变量
- 必须复制派生类构造函数签名中的**所有**参数
- 确保新对象完全独立，无共享状态

---

### 4. 正确识别 PO 类型

通过虚函数正确标识 PO 类型，支持 Preprocessor 的类型检查逻辑。

```cpp
// 在派生类中重写
bool is_crop() const override { return true; }   // Crop类操作
bool is_resize() const override { return true; } // Resize类操作
bool is_random_horizontal_flip() const override { return true; }
```

**Preprocessor 使用方式**：
```cpp
// Preprocessor 判断 ImageNet 首位 PO 合法性
bool is_crop_or_resize_op(const PreprocessOperation* op) {
    return op->is_crop() || op->is_resize();  // 使用虚函数，不使用 dynamic_cast
}
```

**设计原则**：
- ✅ 使用虚函数 `is_crop()` / `is_resize()`
- ❌ 不使用 `dynamic_cast`（硬编码类型检查）
- 注意：`RandomResizedCrop` 和 `FastRandomResizedCrop` 是 Crop，不是 Resize

---

### 5. 实现 get_decode_strategy() 解码策略

只有首位 Crop/Resize 操作需要实现解码策略（决定使用局部解码还是完整解码）。

```cpp
DecodeStrategy get_decode_strategy(
    int32_t image_width,
    int32_t image_height,
    int sdmp_factor,
    Generator* rng
) const override;
```

**返回策略类型**：
```cpp
struct DecodeStrategy {
    bool need_decode = true;      // 是否需要解码
    bool use_partial = false;     // 是否使用局部解码
    int32_t decode_x = 0;         // 解码区域X坐标
    int32_t decode_y = 0;         // 解码区域Y坐标
    int32_t decode_w = 0;         // 解码区域宽度
    int32_t decode_h = 0;         // 解码区域高度
};
```

**决策示例**（RandomResizedCrop）：
```cpp
DecodeStrategy RandomResizedCrop::get_decode_strategy(
    int32_t image_width,
    int32_t image_height,
    int sdmp_factor,
    Generator* rng
) const {
    DecodeStrategy strategy;
    strategy.need_decode = true;

    if (sdmp_factor == 1) {
        // sdmp_factor=1：局部解码（性能优先）
        strategy.use_partial = true;

        // 步骤1：生成随机crop参数
        auto* self = const_cast<RandomResizedCrop*>(this);
        self->generate_crop_params(image_width, image_height, rng);

        // 步骤2：计算MCU对齐的解码窗口
        self->calculate_mcu_aligned_region(image_width, image_height);

        // 步骤3：设置解码区域
        strategy.decode_x = mcu_x_;
        strategy.decode_y = mcu_y_;
        strategy.decode_w = mcu_w_;
        strategy.decode_h = mcu_h_;
    } else {
        // sdmp_factor>1：完整解码（SDMP缓存优先）
        strategy.use_partial = false;
    }

    return strategy;
}
```

**关键原则**：
- 仅在作为首位操作时调用
- 根据随机性和缓存需求选择局部/完整解码
- 局部解码必须计算 MCU 对齐区域

---

## 性能优化

### 6. MCU 对齐的局部解码

对于支持局部解码的 Crop 操作，必须计算 MCU 对齐的解码窗口。

```cpp
void calculate_mcu_aligned_region(int32_t image_width, int32_t image_height) {
    // MCU对齐：向下对齐起始坐标
    mcu_x_ = (crop_x_ / MCU_SIZE) * MCU_SIZE;
    mcu_y_ = (crop_y_ / MCU_SIZE) * MCU_SIZE;

    // MCU对齐：向上对齐结束坐标
    int crop_x_end = crop_x_ + crop_w_;
    int crop_y_end = crop_y_ + crop_h_;
    int mcu_x_end = ((crop_x_end + MCU_SIZE - 1) / MCU_SIZE) * MCU_SIZE;
    int mcu_y_end = ((crop_y_end + MCU_SIZE - 1) / MCU_SIZE) * MCU_SIZE;

    // 计算MCU对齐的宽度和高度
    mcu_w_ = mcu_x_end - mcu_x_;
    mcu_h_ = mcu_y_end - mcu_y_;

    // 边界检查
    if (mcu_x_ + mcu_w_ > image_width) {
        mcu_w_ = image_width - mcu_x_;
    }
    if (mcu_y_ + mcu_h_ > image_height) {
        mcu_h_ = image_height - mcu_y_;
    }

    // 确保解码窗口非空
    mcu_w_ = std::max(8, mcu_w_);   // 至少1个MCU
    mcu_h_ = std::max(8, mcu_h_);
}
```

**性能收益**：局部解码可减少 70-90% 的解码数据量

---

### 7. 使用 Simd 库加速

所有图像处理操作应使用 Simd 库进行加速（Resize、Crop、Flip 等）。

#### 7.1 Simd Resizer 缓存优化

```cpp
// Resize 类的缓存示例
void Resize::execute(...) {
    // 检查缓存是否命中
    if (!resizer_cache_ ||
        cached_src_w_ != input_width ||
        cached_src_h_ != input_height ||
        cached_dst_w_ != output_size_ ||
        cached_dst_h_ != output_size_) {

        // 释放旧的resizer
        if (resizer_cache_) {
            SimdRelease(resizer_cache_);
        }

        // 创建新的resizer（预计算系数）
        resizer_cache_ = SimdResizerInit(
            input_width, input_height,
            output_size_, output_size_,
            num_channels_,
            SimdResizeChannelByte,
            SimdResizeMethodBilinear
        );

        // 更新缓存key
        cached_src_w_ = input_width;
        cached_src_h_ = input_height;
        cached_dst_w_ = output_size_;
        cached_dst_h_ = output_size_;
    }

    // 执行Resize
    SimdResizerRun(resizer_cache_,
                  input_ptr, input_stride,
                  output_ptr, output_stride);
}
```

#### 7.2 随机操作不缓存 Resizer

```cpp
// RandomResizedCrop 每次创建新 resizer（随机性导致缓存无效）
void RandomResizedCrop::execute_from_full_decode(...) {
    // 每次crop区域不同，创建新resizer
    void* resizer = SimdResizerInit(
        crop_w_, crop_h_,
        output_size_, output_size_,
        num_channels_,
        SimdResizeChannelByte,
        SimdResizeMethodBilinear
    );

    SimdResizerRun(resizer,
                  src_row, input_stride,
                  output_ptr, output_stride);

    // 立即释放resizer（不缓存）
    SimdRelease(resizer);
}
```

**性能收益**：Simd SIMD 加速可提升 3-5x 性能

---

## 随机操作特殊处理

### 8. 随机可复现性

所有随机操作必须使用框架提供的 RNG 机制，确保相同种子产生相同结果。

#### 8.1 使用 Philox RNG

```cpp
void generate_crop_params(int32_t image_width, int32_t image_height, Generator* rng) {
    // 生成随机面积（对数空间均匀采样）
    uint64_t scale_offset = rng->next_offset(1);
    float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
    const float target_area = area * std::exp(log_scale_min + scale_rand * (log_scale_max - log_scale_min));

    // 生成随机宽高比
    uint64_t ratio_offset = rng->next_offset(1);
    float ratio_rand = detail::philox_uniform_float(rng->seed(), ratio_offset);
    const float aspect_ratio = std::exp(log_ratio_min + ratio_rand * (log_ratio_max - log_ratio_min));

    // 生成随机位置
    uint64_t x_offset = rng->next_offset(1);
    float x_rand = detail::philox_uniform_float(rng->seed(), x_offset);
    crop_x_ = static_cast<int>(x_rand * max_offset_x);

    // ...
}
```

**关键原则**：
- 必须使用 `Generator* rng` 参数
- 每次随机采样前调用 `rng->next_offset(1)`
- 使用 `detail::philox_uniform_float()` 生成均匀分布

#### 8.2 introduce_randomness() 方法

```cpp
bool introduce_randomness() const override {
    return true;  // 随机操作返回 true
}
```

**Preprocessor 用途**：
- 检查 CPVS 与 val_ops 随机性是否冲突
- 验证 SDMP 缓存有效性

---

### 9. RandomHorizontalFlip 的特殊决策接口

`RandomHorizontalFlip` 实现 `should_flip()` 提前决策接口，优化执行路径。

```cpp
virtual bool should_flip(Generator* rng) {
    uint64_t offset = rng->next_offset(1);
    float rand = detail::philox_uniform_float(rng->seed(), offset);
    return rand < flip_prob_;
}
```

**Preprocessor 使用方式**：
```cpp
// PW 在执行 PO 链前调用
if (auto* flip = dynamic_cast<RandomHorizontalFlip*>(po2)) {
    bool need_flip = flip->should_flip(rng);
    if (!need_flip) {
        // 跳过翻转，直接返回 PO1 结果
        return;
    }
}
```

**其他操作**：基类默认实现抛出 `NotImplementedError`

---

## 线程安全性

### 10. 必须单线程执行

PO 的 `execute()` 方法必须保证单线程调用，避免线程爆炸。

**设计说明**：
- Preprocessor 使用 M 个持久化 PW 线程
- 每个 PW 独立处理样本 `i + k×M`
- PO 不使用静态变量、全局变量或共享缓存

**正确示例**：
```cpp
class Resize : public PreprocessOperation {
private:
    void* resizer_cache_ = nullptr;  // 实例成员（非静态）

    // 每个PW持有独立副本，通过clone()创建
    // 各PW的resizer_cache_互不干扰
};
```

**错误示例**：
```cpp
class Resize : public PreprocessOperation {
private:
    static void* global_resizer_cache_;  // ❌ 静态缓存导致线程竞争

    // 多个PW共享同一缓存，引发数据竞争和性能瓶颈
};
```

---

## ImageNet 首位 PO 要求

### 11. 首位 PO 必须提供解码策略

对于 ImageNet 数据集，第一个 PO 必须是 Crop 或 Resize 操作。

**原因**：
- ImageNet 使用 JPEG 格式，支持局部解码
- 首位 PO 决定解码策略（局部 vs 完整）
- 其他操作（Flip、DoNothing）无法提供解码区域

**Preprocessor 检查逻辑**：
```cpp
// src/data/preprocessor.cpp
bool is_crop_or_resize_op(const PreprocessOperation* op) {
    return op->is_crop() || op->is_resize();
}

// 配置验证
if (dataset_type_ == DatasetType::imagenet) {
    if (!is_crop_or_resize_op(train_ops1)) {
        TR_CHECK(false, ValueError,
                 "ImageNet requires first train transform to be Crop or Resize, "
                 "got: " << train_ops1->name());
    }
}
```

**支持的首位 PO**：
- ✅ `Resize`
- ✅ `CenterCrop`
- ✅ `RandomResizedCrop`
- ✅ `FastRandomResizedCrop`
- ❌ `RandomHorizontalFlip`
- ❌ `DoNothing`

---

## 灰度图和 RGB 兼容性

### 12. 支持灰度和 RGB 两种格式

PO 必须通过 `num_channels_` 动态适配通道数（1 或 3）。

#### 12.1 在 Simd 调用中使用动态通道数

```cpp
void* resizer = SimdResizerInit(
    crop_w_, crop_h_,
    output_size_, output_size_,
    num_channels_,  // 动态通道数（1=灰度, 3=RGB）
    SimdResizeChannelByte,
    SimdResizeMethodBilinear
);
```

#### 12.2 计算字节偏移时乘以通道数

```cpp
// 错误：硬编码3通道
const uint8_t* src_row = input_ptr + crop_y_ * input_stride + crop_x_ * 3;

// 正确：使用动态通道数
const uint8_t* src_row = input_ptr + crop_y_ * input_stride + crop_x_ * num_channels_;
```

#### 12.3 灰度图示例（MNIST）

```cpp
// MNIST 配置
prep.config_preprocessor(..., 1, ...);  // num_color_channels = 1

// PO 自动适配
void execute(...) {
    // num_channels_ = 1（MNIST）
    // num_channels_ = 3（ImageNet）
    const uint8_t* src_row = input_ptr + offset_y * input_stride + offset_x * num_channels_;
    // Simd 自动处理单通道/三通道
}
```

---

## 构造函数设计

### 13. 构造函数初始化基类

```cpp
explicit MyPreprocessOperation(
    int output_size = 224,
    float scale_min = 0.08f,
    float scale_max = 1.0f,
    float ratio_min = 3.0f / 4.0f,
    float ratio_max = 4.0f / 3.0f,
    size_t output_alignment = 0  // 0=紧凑布局，非0=对齐字节数
)
    : PreprocessOperation(output_alignment)  // 初始化基类
    , scale_min_(scale_min)
    , scale_max_(scale_max)
    , ratio_min_(ratio_min)
    , ratio_max_(ratio_max)
{
    output_size_ = output_size;  // 设置基类成员

    // 参数验证
    TR_CHECK(scale_min_ > 0.0f && scale_min_ <= scale_max_, ValueError,
             "scale_min must be in (0, scale_max], got: " << scale_min_ << ", " << scale_max_);
}
```

**关键原则**：
- 必须调用基类构造函数 `PreprocessOperation(output_alignment)`
- 设置 `output_size_`（基类成员）
- 使用 `TR_CHECK` 验证参数合法性

---

## 完整示例

### 示例1：简单 PO（Resize）

```cpp
// resize.h
class Resize : public PreprocessOperation {
public:
    explicit Resize(int output_size = 224, size_t output_alignment = 0);

    void execute(...) override;
    std::unique_ptr<PreprocessOperation> clone() const override;
    std::string name() const override { return "Resize"; }
    bool introduce_randomness() const override { return false; }
    bool is_resize() const override { return true; }
    DecodeStrategy get_decode_strategy(...) const override;

    ~Resize();

private:
    void* resizer_cache_ = nullptr;
    int cached_src_w_ = 0;
    int cached_src_h_ = 0;
    int cached_dst_w_ = 0;
    int cached_dst_h_ = 0;
};
```

```cpp
// resize.cpp
Resize::Resize(int output_size, size_t output_alignment)
    : PreprocessOperation(output_alignment) {
    output_size_ = output_size;
}

void Resize::execute(...) {
    output_width = output_size_;
    output_height = output_size_;

    if (output_stride == 0) {
        output_stride = forced_compact_output ? compact_output_stride_ : output_stride_;
    }

    // 缓存优化
    if (!resizer_cache_ || cached_src_w_ != input_width || ...) {
        if (resizer_cache_) SimdRelease(resizer_cache_);
        resizer_cache_ = SimdResizerInit(...);
        cached_src_w_ = input_width;
        // ...
    }

    SimdResizerRun(resizer_cache_, input_ptr, input_stride, output_ptr, output_stride);
}

std::unique_ptr<PreprocessOperation> Resize::clone() const {
    auto cloned = std::make_unique<Resize>(output_size_);
    cloned->num_channels_ = num_channels_;
    cloned->output_size_ = output_size_;
    cloned->output_alignment_ = output_alignment_;
    cloned->output_stride_ = output_stride_;
    cloned->compact_output_stride_ = compact_output_stride_;
    cloned->rank_first_in_the_po_chain_ = rank_first_in_the_po_chain_;
    return cloned;
}
```

---

### 示例2：复杂随机 PO（RandomResizedCrop）

```cpp
// random_resized_crop.h
class RandomResizedCrop : public PreprocessOperation {
public:
    RandomResizedCrop(int output_size, float scale_min, float scale_max,
                      float ratio_min, float ratio_max, size_t output_alignment);

    void execute(...) override;
    std::unique_ptr<PreprocessOperation> clone() const override;
    std::string name() const override { return "RandomResizedCrop"; }
    bool introduce_randomness() const override { return true; }
    bool is_crop() const override { return true; }
    DecodeStrategy get_decode_strategy(...) const override;

private:
    float scale_min_, scale_max_, ratio_min_, ratio_max_;
    int crop_x_, crop_y_, crop_w_, crop_h_;  // 当前样本的crop参数
    int mcu_x_, mcu_y_, mcu_w_, mcu_h_;      // MCU对齐的解码区域

    void generate_crop_params(int32_t image_width, int32_t image_height, Generator* rng);
    void calculate_mcu_aligned_region(int32_t image_width, int32_t image_height);
    void execute_from_full_decode(...);
    void execute_from_partial_decode(...);
};
```

```cpp
// random_resized_crop.cpp
DecodeStrategy RandomResizedCrop::get_decode_strategy(...) const {
    DecodeStrategy strategy;
    strategy.need_decode = true;

    if (sdmp_factor == 1) {
        strategy.use_partial = true;
        auto* self = const_cast<RandomResizedCrop*>(this);
        self->generate_crop_params(image_width, image_height, rng);
        self->calculate_mcu_aligned_region(image_width, image_height);
        strategy.decode_x = mcu_x_;
        strategy.decode_y = mcu_y_;
        strategy.decode_w = mcu_w_;
        strategy.decode_h = mcu_h_;
    } else {
        strategy.use_partial = false;
    }

    return strategy;
}

void RandomResizedCrop::execute(...) {
    if (execute_from_full || !rank_first_in_the_po_chain_) {
        generate_crop_params(input_width, input_height, rng);
    }

    output_width = output_size_;
    output_height = output_size_;

    if (output_stride == 0) {
        output_stride = forced_compact_output ? compact_output_stride_ : output_stride_;
    }

    if (execute_from_full || !rank_first_in_the_po_chain_) {
        execute_from_full_decode(input_ptr, input_width, input_height, ...);
    } else {
        execute_from_partial_decode(input_ptr, input_stride, ...);
    }
}
```

---

## 常见错误和最佳实践

### ❌ 错误示例

#### 错误1：硬编码通道数
```cpp
// 错误：硬编码3通道
const uint8_t* src_row = input_ptr + crop_y_ * input_stride + crop_x_ * 3;
```

#### 错误2：使用静态缓存
```cpp
// 错误：静态缓存导致线程竞争
static void* global_resizer_cache_ = nullptr;
```

#### 错误3：clone() 不完整
```cpp
// 错误：未复制基类成员
std::unique_ptr<PreprocessOperation> clone() const override {
    return std::make_unique<Resize>(output_size_);  // ❌ 缺少基类成员复制
}
```

#### 错误4：使用 dynamic_cast 判断类型
```cpp
// 错误：硬编码类型检查
bool is_crop_or_resize_op(const PreprocessOperation* op) {
    if (auto* resize = dynamic_cast<const Resize*>(op)) return true;
    if (auto* rrc = dynamic_cast<const RandomResizedCrop*>(op)) return true;
    return false;  // ❌ 无法识别新的 FastRandomResizedCrop
}
```

---

### ✅ 正确示例

#### 正确1：动态通道数
```cpp
// 正确：使用动态通道数
const uint8_t* src_row = input_ptr + crop_y_ * input_stride + crop_x_ * num_channels_;
```

#### 正确2：实例成员缓存
```cpp
// 正确：每个PW持有独立副本
class Resize : public PreprocessOperation {
private:
    void* resizer_cache_ = nullptr;  // 实例成员
};
```

#### 正确3：完整的 clone()
```cpp
// 正确：复制所有成员
std::unique_ptr<PreprocessOperation> clone() const override {
    auto cloned = std::make_unique<Resize>(output_size_);
    cloned->num_channels_ = num_channels_;
    cloned->output_size_ = output_size_;
    cloned->output_alignment_ = output_alignment_;
    cloned->output_stride_ = output_stride_;
    cloned->compact_output_stride_ = compact_output_stride_;
    cloned->rank_first_in_the_po_chain_ = rank_first_in_the_po_chain_;
    return cloned;
}
```

#### 正确4：使用虚函数判断类型
```cpp
// 正确：使用虚函数
bool is_crop_or_resize_op(const PreprocessOperation* op) {
    return op->is_crop() || op->is_resize();  // ✅ 自动支持新PO
}
```

---

## 检查清单

开发新 PO 时，请确保完成以下检查：

- [ ] 继承 `PreprocessOperation` 基类
- [ ] 实现纯虚函数：`execute()`, `clone()`, `name()`, `introduce_randomness()`
- [ ] 实现类型识别虚函数：`is_crop()` 或 `is_resize()`
- [ ] 实现解码策略：`get_decode_strategy()`（首位 Crop/Resize）
- [ ] 使用动态通道数 `num_channels_` 而非硬编码
- [ ] `clone()` 复制基类和派生类的**全部**成员变量
- [ ] 使用 Simd 库加速图像处理操作
- [ ] 随机操作使用 Philox RNG（`rng->next_offset()` + `philox_uniform_float()`）
- [ ] 不使用静态/全局缓存（避免线程竞争）
- [ ] 局部解码操作实现 MCU 对齐
- [ ] `execute()` 正确处理 `execute_from_full` 参数
- [ ] 构造函数使用 `TR_CHECK` 验证参数
- [ ] 支持 RGB 和灰度图（通过 `num_channels_` 动态适配）
- [ ] 添加到 `src/data/CMakeLists.txt`
- [ ] 添加到 `include/renaissance.h`
- [ ] 在 `test_po.cpp` 和 `test_pw_ultimate.cpp` 添加测试支持

---

## 参考文档

- **数据流架构**: `docs/data_flow_analysis.md`
- **PW开发指南**: `docs/BEST_PW.md`
- **Preprocessor配置**: `docs/BEST_PREPROCESSOR.md`
- **异常处理**: `docs/logger_exception_handbook.md`
- **构建系统**: `docs/alpha_build.md`
- **FRRC算法**: `docs/FRRC_VS_RRC.md`

---

**更新记录**：
- 2026-02-22: 初始版本，整合 PO 开发全部指南
