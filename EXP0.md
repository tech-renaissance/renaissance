### 【技术觉醒框架 - PreprocessWorker完整设计方案 V2.1】

#### 【版本信息】

- **方案版本**: V2.1（评审修正版）
- **编写日期**: 2026-02-16
- **修正依据**: 采纳评审意见【七】的8条关键修正
- **状态**: 可直接实施
- **修正重点**: 批次边界保护、S区独立索引、状态重置、NUMA亲和性

---

#### 【一、设计总览】

##### 1.1 核心设计原则

本方案完全遵循【三】的基本设计理念，并基于评审意见进行了关键修正：

| 设计原则               | 实现策略                           | 修正要点（V2.1）                       |
| ---------------------- | ---------------------------------- | -------------------------------------- |
| **静态分配，零竞争**   | 所有线程到资源的映射在初始化期确定 | ✅ 增加批次边界保护                     |
| **NUMA本地化**         | First-Touch + 线程绑核             | ✅ 延迟First-Touch到绑核后              |
| **一次分配，永久存在** | Workshop在构造时分配，析构时释放   | ✅ 分离分配与First-Touch                |
| **随机可复现**         | Philox RNG + worker_id派生seed     | ✅ S区洗牌seed修正、Generator延迟初始化 |
| **最小化接口**         | 类间接口简洁，避免过度抽象         | ✅ EngineBuffer增加batch_id参数         |

##### 1.2 架构图

```
Preprocessor（工厂）
    │
    ├─ 配置阶段
    │   ├─ 选择DataLoader（MNIST/CIFAR/ImageNet × DTS/RAW）
    │   ├─ 计算Workshop大小（D/A/B/T/S/C区）
    │   ├─ 计算CPU绑定策略（仅GPU_CLOUD + AUTO_CPU_BINDING）
    │   └─ 准备PO模板（train_ops/val_ops）
    │
    ├─ 运行阶段
    │   ├─ 启动持久线程池
    │   │   └─ 每个线程首次启动时
    │   │       ├─ 绑定CPU核心（触发NUMA亲和）
    │   │       ├─ 创建PW对象（分配Workshop）
    │   │       ├─ 执行First-Touch（确保本地内存）
    │   │       └─ 克隆PO列表
    │   │
    │   ├─ 每个Phase开始时
    │   │   ├─ 更新GlobalRegistry（current_resolution等）
    │   │   ├─ 准备PWParam
    │   │   ├─ 遍历所有PW更新参数（重置计数器）
    │   │   └─ Lazy epoch时：主线程调用shuffle_s_region()
    │   │
    │   └─ 工作循环
    │       ├─ loader.get_next_sample()
    │       ├─ pw->work() → 计算batch_id → 写入EngineBuffer
    │       └─ EngineBuffer批次保护 + 自动触发传输
    │
    └─ EngineBuffer[0..world_size-1]
        └─ 双缓冲 + 批次边界保护 + 零竞争写入
```

##### 1.3 V2.1关键修正点

| 修正项            | 原问题                   | 修正后                   | 来源     |
| ----------------- | ------------------------ | ------------------------ | -------- |
| 批次边界保护      | 快Worker覆盖慢Worker数据 | write_at增加batch_id参数 | B1+B2+B5 |
| S区独立索引       | 多个S区共享索引混乱      | s_shuffle_indices_[n][m] | B1+B3+B5 |
| total_samples重置 | Phase切换后计数器未清零  | update_parameters中重置  | B1+B5    |
| C区索引重置       | 第二次验证失败           | update_parameters中重置  | B3+B5    |
| First-Touch时机   | 可能在主线程触发         | 延迟到绑核后首次work()   | B5       |
| S区洗牌seed       | 破坏可复现性             | 移除s_region_idx         | B4+B5    |
| Generator初始化   | 可能用到默认seed=0       | 延迟到首次work()         | B3       |
| Flip的RNG调用澄清 | 文档不清晰               | 增强注释说明             | B2+B5    |

---

#### 【二、PreprocessWorkerParameter 设计】

##### 2.1 核心理念

**只包含运行时动态信息**，所有固定配置从GlobalRegistry读取，避免冗余传递。

##### 2.2 头文件定义

```cpp
/**
 * @file preprocess_worker_parameter.h
 * @brief PW运行时参数（每个phase之初更新）
 * @version 2.1.0（评审修正版）
 * @date 2026-02-16
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace tr {

/**
 * @struct PreprocessWorkerParameter
 * @brief 预处理工作器运行时参数
 * 
 * 设计原则：
 * - 只包含每个phase开始时需要更新的动态信息
 * - 固定配置（batch_size、sdmp_factor等）从GlobalRegistry读取
 * - 轻量级，方便传递和复制
 * 
 * 修正记录（V2.1）：
 * - 无需修改（此结构体设计评审通过）
 */
struct PreprocessWorkerParameter {
    // ==================== Phase标识 ====================
    bool is_train = true;              ///< true=训练阶段, false=验证阶段
    
    // ==================== SDMP状态 ====================
    bool is_busy_epoch = true;         ///< true=需解码预处理, false=从S区读取
    int active_s_region_idx = 0;       ///< Lazy epoch使用的S区索引(0~sdmp_factor-2)
    
    // ==================== CPVS状态 ====================
    bool is_first_val = true;          ///< true=首次验证需预处理, false=从C区读取
    
    // ==================== 当前分辨率 ====================
    int current_train_resolution = 224; ///< 训练集当前分辨率（渐进式训练）
    int current_val_resolution = 224;   ///< 验证集当前分辨率（通常固定224）
    
    // ==================== Epoch计数 ====================
    int epoch_id = 0;                  ///< 当前epoch ID（用于洗牌seed）
};

} // namespace tr
```

---

#### 【三、DecodeStrategy 设计】

##### 3.1 结构体定义

```cpp
/**
 * @file decode_strategy.h
 * @brief JPEG解码策略封装
 * @version 2.1.0（评审修正版）
 * @date 2026-02-16
 * @author 技术觉醒团队
 * @note 所属系列: transforms
 */

#pragma once

#include <cstdint>

namespace tr {
namespace transforms {

/**
 * @struct DecodeStrategy
 * @brief JPEG解码策略
 * 
 * 说明：
 * - need_decode=false: 非ImageNet，直接使用输入（已解码）
 * - use_partial=true: 局部解码到decode窗口，然后从crop位置开始处理
 * - use_partial=false: 完整解码整张图
 * 
 * MCU对齐规则：
 * - decode_x/y: 向下对齐到8的倍数
 * - decode_w/h: 向上对齐到8的倍数
 * - crop_x/y/w/h: 相对于decode窗口的偏移，可以不对齐
 * 
 * 修正记录（V2.1）：
 * - 无需修改（结构体设计评审通过）
 */
struct DecodeStrategy {
    bool need_decode = false;       ///< 是否需要解码（非ImageNet为false）
    bool use_partial = false;       ///< 局部解码vs完整解码
    
    // ==================== MCU对齐的解码窗口（8的倍数）====================
    int32_t decode_x = 0;           ///< 解码起始X（MCU对齐，向下取整）
    int32_t decode_y = 0;           ///< 解码起始Y（MCU对齐，向下取整）
    int32_t decode_w = 0;           ///< 解码宽度（MCU对齐，向上取整）
    int32_t decode_h = 0;           ///< 解码高度（MCU对齐，向上取整）
    
    // ==================== 精确裁剪窗口（相对于解码窗口的偏移）====================
    int32_t crop_x = 0;             ///< 裁剪起始X（相对于decode_x）
    int32_t crop_y = 0;             ///< 裁剪起始Y（相对于decode_y）
    int32_t crop_w = 0;             ///< 裁剪宽度
    int32_t crop_h = 0;             ///< 裁剪高度
};

} // namespace transforms
} // namespace tr
```

---

#### 【三.五、局部解码失败处理机制】

##### 3.5.1 问题背景

在 ImageNet RAW 数据集中，存在少量 JPEG 文件使用非标准格式（如 CMYK 色彩空间、特殊标记等），导致 **libjpeg-turbo 的局部解码 API（tj3Decompress8 + tj3SetCroppingRegion）解码失败**。

这些文件的特点：
- PIL（Pillow）可以解码（DTS 创建时自动转换为 RGB）
- libjpeg-turbo 完整解码也失败
- **STB Image 可以解码**（兼容性更强）

##### 3.5.2 处理策略

**核心原则**：TurboJPEG 局部解码失败 → STB 完整解码备用 → PO 适配两种输入模式

**PW 的工作流程**：

```cpp
// 1. 获取解码策略
DecodeStrategy strategy = current_op->get_decode_strategy(...);
current_op->set_decode_strategy(strategy);  // 存储到PO中（CenterCrop等需要）

bool successfully_partial_decoded = false;

// 2. 尝试 TurboJPEG 局部解码
if (strategy.use_partial) {
    if (tj3SetCroppingRegion(tj, {strategy.decode_x, strategy.decode_y,
                                   strategy.decode_w, strategy.decode_h}) == 0) {
        if (tj3Decompress8(...) == 0) {
            successfully_partial_decoded = true;  // ✅ 局部解码成功
        }
    }
}

// 3. TurboJPEG 失败，尝试 STB 完整解码
if (!successfully_partial_decoded) {
    #if TR_USE_STB
    if (decode_jpeg_with_stb(...)) {
        // ✅ STB 完整解码成功
        successfully_partial_decoded = false;  // 保持 false 标记
    } else {
        // ❌ 两者都失败
        LOG_ERROR << "Both TurboJPEG and STB failed";
        return;  // 跳过样本
    }
    #endif
}

// 4. 调用 PO::execute()
current_op->execute(
    decode_buffer,
    decoded_width,
    decoded_height,
    decoded_stride,
    output_buffer,
    ...,
    !successfully_partial_decoded  // 关键：局部成功=false, STB完整=true
);
```

##### 3.5.3 PreprocessOperation 的 execute() 参数扩展

**新增参数**：`bool execute_from_full = false`

```cpp
virtual void execute(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    int32_t& output_width,
    int32_t& output_height,
    size_t output_stride,
    Generator* rng = nullptr,
    /**
     * @brief 是否从完整解码的图像中执行（而非局部解码的图像）
     * @details
     * - false (默认): 从局部解码的图像中执行（TurboJPEG 局部解码成功）
     * - true: 从完整解码的图像中执行（TurboJPEG 失败，STB 完整解码）
     *
     * @note 对于 CenterCrop 等支持局部解码的操作：
     *       - false: input 是局部解码区域（如 300x300），使用 DecodeStrategy 中的相对偏移
     *       - true: input 是完整图像（如 2000x2000），直接计算全局居中位置
     */
    bool execute_from_full = false
) = 0;
```

##### 3.5.4 CenterCrop 的两套算法实现

**存储 DecodeStrategy**：

```cpp
class CenterCrop : public PreprocessOperation {
private:
    DecodeStrategy strategy_;  // 存储解码策略（用于 execute_from_full=false）

public:
    void set_decode_strategy(const DecodeStrategy& strategy) {
        strategy_ = strategy;
    }
};
```

**两种模式实现**：

```cpp
void CenterCrop::execute(..., bool execute_from_full) {
    int copy_w, copy_h, src_x, src_y;

    if (execute_from_full) {
        // ===================================================================
        // 模式 1：从完整解码的图像中 crop（STB 备用解码）
        // input 是完整图像（如 2000x2000），直接计算全局居中位置
        // ===================================================================
        copy_w = std::min(input_width, output_size_);
        copy_h = std::min(input_height, output_size_);
        src_x = (input_width - copy_w) / 2;   // 全局坐标
        src_y = (input_height - copy_h) / 2;  // 全局坐标
    } else {
        // ===================================================================
        // 模式 2：从局部解码的图像中 crop（TurboJPEG 局部解码成功）
        // input 是局部解码区域（如 300x300），使用 DecodeStrategy 中的相对偏移
        // ===================================================================
        copy_w = strategy_.crop_w;
        copy_h = strategy_.crop_h;
        src_x = strategy_.crop_x;  // 相对坐标（如 8，MCU 对齐偏移）
        src_y = strategy_.crop_y;
    }

    // 后续复制逻辑相同...
}
```

##### 3.5.5 RandomResizedCrop 的类似处理

**RandomResizedCrop 本身就需要两套算法**：

- **不使用 SDMP**：必定局部解码（需要随机 crop 不同区域）
- **使用 SDMP**：必定完整解码（一次预处理，多次复用）

因此，RandomResizedCrop 的实现模式：
1. `execute_from_full=false`：使用局部解码的算法（从局部数据随机 crop）
2. `execute_from_full=true`：使用完整解码的算法（从完整数据随机 crop）

##### 3.5.6 性能影响分析

**失败样本占比**：< 1%（ImageNet RAW 数据集）

**性能对比**：

| 场景 | TurboJPEG 局部解码 | STB 完整解码 | 性能损失 |
|------|-------------------|-------------|---------|
| CenterCrop（2000→224） | ~0.5ms | ~2.0ms | +1.5ms |
| RandomResizedCrop（2000→224） | ~1.0ms | ~3.0ms | +2.0ms |

**整体影响**：可忽略（< 1% 样本 × ~2ms 额外时间 << 总体训练时间）

##### 3.5.7 设计优势

1. **零额外复制**：D 区解码后直接 crop 到 A 区
2. **无临时内存分配**：使用预分配的 D 区缓冲区
3. **NUMA 友好**：所有操作在本地内存
4. **逻辑清晰**：PO 明确知道处理的是完整数据还是局部数据
5. **可扩展**：其他 PO 可按需实现

##### 3.5.8 实现清单

- [x] PreprocessOperation::execute() 添加 `execute_from_full` 参数
- [x] CenterCrop 实现 `set_decode_strategy()` + 两套算法
- [x] Resize、DoNothing 添加参数但忽略（总是完整解码）
- [ ] RandomResizedCrop 实现两套算法（SDMP vs 非SDMP）
- [ ] PW 实现局部解码失败 → STB 完整解码的回退逻辑

---

#### 【四、PreprocessOperation 设计】

##### 4.1 抽象基类

```cpp
/**
 * @file preprocess_operation.h
 * @brief 预处理操作抽象基类
 * @version 2.1.0（评审修正版）
 * @date 2026-02-16
 * @author 技术觉醒团队
 * @note 所属系列: transforms
 * 
 * 修正记录（V2.1）：
 * - 数据类型：uint8_t*（像素值0-255无符号）
 * - 移除input_bytes参数（冗余，可计算）
 * - 增加stride参数（Simd必需）
 */

#pragma once

#include "renaissance/transforms/decode_strategy.h"
#include "renaissance/base/rng.h"
#include <cstdint>
#include <string>
#include <memory>

namespace tr {
namespace transforms {

/**
 * @class PreprocessOperation
 * @brief 预处理操作抽象基类
 * 
 * 设计原则：
 * 1. 轻量级：仅持有参数，不持有大块内存
 * 2. 可克隆：通过clone()深拷贝给每个PW
 * 3. 无状态共享：同一PO多次调用execute()结果一致（给定相同rng状态）
 * 4. 性能优化：可缓存Simd上下文（如ResizerCache）
 */
class PreprocessOperation {
public:
    virtual ~PreprocessOperation() = default;
    
    // =========================================================================
    // 核心执行接口
    // =========================================================================
    
    /**
     * @brief 执行预处理操作
     * @param input_ptr 输入图像数据（RGB uint8，值域0-255）
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入行步长（字节）← 关键：Simd必需
     * @param output_ptr 输出图像数据（预分配）
     * @param output_width [输出] 输出宽度
     * @param output_height [输出] 输出高度
     * @param output_stride 输出行步长（字节）← 关键：Simd必需
     * @param rng 随机数生成器（可选，仅随机操作使用）
     * 
     * @note 所有Simd操作都需要stride，调用者负责计算对齐后的stride
     * @note 输出指针已预分配，操作内部不分配内存
     */
    virtual void execute(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        int32_t& output_width,
        int32_t& output_height,
        size_t output_stride,
        Generator* rng = nullptr
    ) = 0;
    
    // =========================================================================
    // 克隆接口（用于复制给PW）
    // =========================================================================
    
    /**
     * @brief 深拷贝当前对象
     * @return 新的独立副本（unique_ptr）
     * 
     * @note 每个PW持有独立副本，避免共享状态导致的缓存冲突
     */
    virtual std::unique_ptr<PreprocessOperation> clone() const = 0;
    
    // =========================================================================
    // 元信息查询
    // =========================================================================
    
    virtual std::string name() const = 0;
    virtual bool introduce_randomness() const = 0;
    virtual bool is_crop() const { return false; }
    virtual bool is_resize() const { return false; }
    virtual bool is_random_horizontal_flip() const { return false; }
    virtual bool require_temp() const { return false; }
    
    // =========================================================================
    // 动态参数更新（渐进式分辨率）
    // =========================================================================
    
    /**
     * @brief 设置输出尺寸
     * @note 仅Crop/Resize类操作需要实现
     */
    virtual void set_output_size(int size) { (void)size; }
    virtual int get_output_size() const { return 0; }
    
    // =========================================================================
    // 解码策略（仅首位Crop/Resize操作使用）
    // =========================================================================
    
    /**
     * @brief 获取解码策略
     * @param image_width 原始图像宽度（从JPEG头读取）
     * @param image_height 原始图像高度（从JPEG头读取）
     * @param sdmp_factor SDMP因子
     * @param rng 随机数生成器
     * @return 解码策略
     * 
     * @note 仅在作为首个操作时调用
     * @note 调用前必须已读取JPEG头获取真实尺寸
     */
    virtual DecodeStrategy get_decode_strategy(
        int32_t image_width,
        int32_t image_height,
        int sdmp_factor,
        Generator* rng
    ) const {
        // 默认：不需要解码（非ImageNet或非首位）
        return DecodeStrategy{};
    }

protected:
    // ==================== 工具方法 ====================
    static constexpr int MCU_SIZE = 8;
    
    /**
     * @brief MCU对齐（向下取整）
     */
    static int32_t align_down_mcu(int32_t value) {
        return (value / MCU_SIZE) * MCU_SIZE;
    }
    
    /**
     * @brief MCU对齐（向上取整）
     */
    static int32_t align_up_mcu(int32_t value) {
        return ((value + MCU_SIZE - 1) / MCU_SIZE) * MCU_SIZE;
    }
    
    /**
     * @brief 计算Simd对齐的stride（64字节对齐）
     */
    static size_t calculate_stride(int32_t width, int32_t channels) {
        constexpr size_t ALIGNMENT = 64;
        size_t raw_stride = static_cast<size_t>(width) * channels;
        return ((raw_stride + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }
};

} // namespace transforms
} // namespace tr
```

---

##### 4.2 Resize实现

```cpp
/**
 * @file resize.h
 * @brief Resize操作（双线性插值，Simd加速）
 * @version 2.1.0（评审修正版）
 * @date 2026-02-16
 * @author 技术觉醒团队
 */

#pragma once

#include "renaissance/transforms/preprocess_operation.h"
#include "renaissance/base/tr_exception.h"
#include <Simd/SimdLib.h>

namespace tr {
namespace transforms {

/**
 * @class Resize
 * @brief 图像缩放操作
 * 
 * 核心优化：
 * - 缓存Simd Resizer上下文，避免每次Init/Release
 * - 支持动态分辨率（渐进式训练）
 * 
 * 性能：
 * - 缓存命中时，性能提升30%+
 * - 224x224输出，约0.5ms/image（AVX2）
 */
class Resize : public PreprocessOperation {
public:
    explicit Resize(int output_size = 224) : output_size_(output_size) {}
    
    void execute(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        int32_t& output_width,
        int32_t& output_height,
        size_t output_stride,
        Generator* rng = nullptr
    ) override {
        (void)rng;  // Resize不使用随机数
        
        output_width = output_size_;
        output_height = output_size_;
        
        // ==================== Simd Resizer缓存优化 ====================
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
                3,  // RGB
                SimdResizeChannelByte,
                SimdResizeMethodBilinear
            );
            
            TR_CHECK(resizer_cache_ != nullptr, MemoryError,
                     "SimdResizerInit failed: " << input_width << "x" << input_height
                     << " -> " << output_size_ << "x" << output_size_);
            
            // 更新缓存key
            cached_src_w_ = input_width;
            cached_src_h_ = input_height;
            cached_dst_w_ = output_size_;
            cached_dst_h_ = output_size_;
        }
        
        // ==================== 执行Resize ====================
        SimdResizerRun(resizer_cache_, 
                      input_ptr, input_stride,
                      output_ptr, output_stride);
    }
    
    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<Resize>(output_size_);
    }
    
    std::string name() const override { return "Resize"; }
    bool introduce_randomness() const override { return false; }
    bool is_resize() const override { return true; }
    
    void set_output_size(int size) override { 
        if (size != output_size_) {
            output_size_ = size;
            // 清空缓存，下次execute时会重建
            if (resizer_cache_) {
                SimdRelease(resizer_cache_);
                resizer_cache_ = nullptr;
            }
        }
    }
    
    int get_output_size() const override { return output_size_; }
    
    /**
     * @brief 获取解码策略（Resize首位时必须完整解码）
     */
    DecodeStrategy get_decode_strategy(
        int32_t image_width,
        int32_t image_height,
        int sdmp_factor,
        Generator* rng
    ) const override {
        (void)sdmp_factor;
        (void)rng;
        
        DecodeStrategy strategy;
        strategy.need_decode = true;
        strategy.use_partial = false;  // Resize必须完整解码
        return strategy;
    }
    
    ~Resize() {
        if (resizer_cache_) {
            SimdRelease(resizer_cache_);
        }
    }

private:
    int output_size_;
    
    // ==================== Simd上下文缓存 ====================
    mutable void* resizer_cache_ = nullptr;
    mutable int cached_src_w_ = 0;
    mutable int cached_src_h_ = 0;
    mutable int cached_dst_w_ = 0;
    mutable int cached_dst_h_ = 0;
};

} // namespace transforms
} // namespace tr
```

---

##### 4.3 CenterCrop实现

```cpp
/**
 * @file center_crop.h
 * @brief 中心裁剪操作
 * @version 2.1.0（评审修正版）
 * @date 2026-02-16
 * @author 技术觉醒团队
 */

#pragma once

#include "renaissance/transforms/preprocess_operation.h"
#include "renaissance/base/tr_exception.h"
#include <algorithm>
#include <cstring>

namespace tr {
namespace transforms {

/**
 * @class CenterCrop
 * @brief 中心裁剪
 * 
 * 功能：
 * - 从输入图像中心裁剪指定尺寸
 * - 如果输入小于输出，返回整个输入（不放大）
 * 
 * 性能：
 * - 纯memcpy，约0.1ms/image（224x224）
 */
class CenterCrop : public PreprocessOperation {
public:
    explicit CenterCrop(int output_size = 224) : output_size_(output_size) {}
    
    void execute(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        int32_t& output_width,
        int32_t& output_height,
        size_t output_stride,
        Generator* rng = nullptr
    ) override {
        (void)rng;
        
        // 计算中心裁剪区域
        int crop_w = std::min(input_width, output_size_);
        int crop_h = std::min(input_height, output_size_);
        int start_x = (input_width - crop_w) / 2;
        int start_y = (input_height - crop_h) / 2;
        
        output_width = crop_w;
        output_height = crop_h;
        
        // 逐行复制（处理stride可能不同的情况）
        const uint8_t* src = input_ptr + start_y * input_stride + start_x * 3;
        size_t row_bytes = crop_w * 3;
        
        for (int y = 0; y < crop_h; ++y) {
            std::memcpy(output_ptr + y * output_stride,
                       src + y * input_stride,
                       row_bytes);
        }
    }
    
    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<CenterCrop>(output_size_);
    }
    
    std::string name() const override { return "CenterCrop"; }
    bool introduce_randomness() const override { return false; }
    bool is_crop() const override { return true; }
    
    void set_output_size(int size) override { output_size_ = size; }
    int get_output_size() const override { return output_size_; }
    
    /**
     * @brief 获取解码策略（CenterCrop首位时使用局部解码）
     */
    DecodeStrategy get_decode_strategy(
        int32_t image_width,
        int32_t image_height,
        int sdmp_factor,
        Generator* rng
    ) const override {
        (void)sdmp_factor;
        (void)rng;
        
        DecodeStrategy strategy;
        strategy.need_decode = true;
        strategy.use_partial = true;  // 局部解码
        
        // 计算裁剪区域
        int crop_w = std::min(image_width, output_size_);
        int crop_h = std::min(image_height, output_size_);
        int crop_x = (image_width - crop_w) / 2;
        int crop_y = (image_height - crop_h) / 2;
        
        // ==================== MCU对齐解码窗口 ====================
        strategy.decode_x = align_down_mcu(crop_x);
        strategy.decode_y = align_down_mcu(crop_y);
        
        int decode_x_end = crop_x + crop_w;
        int decode_y_end = crop_y + crop_h;
        strategy.decode_w = align_up_mcu(decode_x_end - strategy.decode_x);
        strategy.decode_h = align_up_mcu(decode_y_end - strategy.decode_y);
        
        // 边界检查（防止超出图像范围）
        if (strategy.decode_x + strategy.decode_w > image_width) {
            strategy.decode_w = image_width - strategy.decode_x;
        }
        if (strategy.decode_y + strategy.decode_h > image_height) {
            strategy.decode_h = image_height - strategy.decode_y;
        }
        
        // ==================== 精确裁剪窗口（相对偏移）====================
        strategy.crop_x = crop_x - strategy.decode_x;
        strategy.crop_y = crop_y - strategy.decode_y;
        strategy.crop_w = crop_w;
        strategy.crop_h = crop_h;
        
        return strategy;
    }

private:
    int output_size_;
};

} // namespace transforms
} // namespace tr
```

---

##### 4.4 RandomHorizontalFlip实现

```cpp
/**
 * @file random_horizontal_flip.h
 * @brief 随机水平翻转（50%概率，优化版）
 * @version 2.1.0（评审修正版）
 * @date 2026-02-16
 * @author 技术觉醒团队
 * 
 * 修正记录（V2.1）：
 * - 采纳B2+B5意见3：增强文档注释，说明SDMP下的RNG调用逻辑
 */

#pragma once

#include "renaissance/transforms/preprocess_operation.h"
#include "renaissance/base/philox.h"

namespace tr {
namespace transforms {

/**
 * @class RandomHorizontalFlip
 * @brief 随机水平翻转
 * 
 * ✅ 随机可复现性说明（V2.1澄清）：
 * - should_flip()每次调用都会消耗1个RNG offset
 * - 在SDMP模式下，对同一原始样本预处理N次，should_flip()会被调用N次
 * - 这是**正确设计**：每次预处理的翻转决策应独立
 * - PW在execute_po_chain的每次SDMP循环中都会调用should_flip()
 * - 例如：sdmp_factor=3时，同一原始样本会生成3个不同的预处理结果
 *   - 第1次：flip=true  → S1区存储翻转版本
 *   - 第2次：flip=false → S2区存储原版
 *   - 第3次：flip=true  → 输出到EngineBuffer（翻转版本）
 * 
 * 优化策略：
 * - PW在execute_po_chain前调用should_flip()预判
 * - 如果不需要翻转，倒数第二步直接输出到final位置
 * - 节省1次AB区复制，约1-2s/epoch
 * 
 * 性能：
 * - Flip操作：约0.2ms/image（224x224）
 * - 优化收益：50%概率节省0.2ms复制时间
 */
class RandomHorizontalFlip : public PreprocessOperation {
public:
    explicit RandomHorizontalFlip(float prob = 0.5f) : prob_(prob) {}
    
    /**
     * @brief 预判是否需要翻转（供PW优化路径使用）
     * @param rng 随机数生成器
     * @return true=需要翻转, false=不需要
     * 
     * @note ✅ 关键：此方法每次SDMP循环都会调用一次
     * @note 这保证了"多次独立预处理"的随机性
     */
    bool should_flip(Generator* rng) {
        if (!rng) return false;
        
        uint64_t offset = rng->next_offset(1);
        float rand_val = detail::philox_uniform_float(rng->seed(), offset);
        return rand_val < prob_;
    }
    
    void execute(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        int32_t& output_width,
        int32_t& output_height,
        size_t output_stride,
        Generator* rng = nullptr
    ) override {
        (void)rng;  // ✅ execute不使用RNG（已在should_flip中消耗）
        
        output_width = input_width;
        output_height = input_height;
        
        // ==================== 水平翻转：逐行复制并反转像素顺序 ====================
        for (int y = 0; y < input_height; ++y) {
            const uint8_t* src_row = input_ptr + y * input_stride;
            uint8_t* dst_row = output_ptr + y * output_stride;
            
            for (int x = 0; x < input_width; ++x) {
                int src_x = input_width - 1 - x;
                dst_row[x * 3 + 0] = src_row[src_x * 3 + 0];
                dst_row[x * 3 + 1] = src_row[src_x * 3 + 1];
                dst_row[x * 3 + 2] = src_row[src_x * 3 + 2];
            }
        }
    }
    
    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<RandomHorizontalFlip>(prob_);
    }
    
    std::string name() const override { return "RandomHorizontalFlip"; }
    bool introduce_randomness() const override { return true; }
    bool is_random_horizontal_flip() const override { return true; }

private:
    float prob_;
};

} // namespace transforms
} // namespace tr
```

---

##### 4.5 DoNothing实现

```cpp
/**
 * @file do_nothing.h
 * @brief 占位操作（直接复制）
 * @version 2.1.0
 * @date 2026-02-16
 * @author 技术觉醒团队
 */

#pragma once

#include "renaissance/transforms/preprocess_operation.h"
#include <cstring>

namespace tr {
namespace transforms {

/**
 * @class DoNothing
 * @brief 占位操作（直接复制输入到输出）
 * 
 * 用途：
 * - 测试框架流程
 * - 非ImageNet数据集的直接传递
 * 
 * 注意：
 * - 不能单独用于ImageNet（需要先Crop或Resize）
 */
class DoNothing : public PreprocessOperation {
public:
    void execute(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        int32_t& output_width,
        int32_t& output_height,
        size_t output_stride,
        Generator* rng = nullptr
    ) override {
        (void)rng;
        
        output_width = input_width;
        output_height = input_height;
        
        // 逐行复制（处理stride可能不同的情况）
        size_t row_bytes = input_width * 3;
        
        for (int y = 0; y < input_height; ++y) {
            std::memcpy(output_ptr + y * output_stride,
                       input_ptr + y * input_stride,
                       row_bytes);
        }
    }
    
    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<DoNothing>();
    }
    
    std::string name() const override { return "DoNothing"; }
    bool introduce_randomness() const override { return false; }
};

} // namespace transforms
} // namespace tr
```

---

#### 【五、PreprocessWorker 设计】

##### 5.1 头文件定义

```cpp
/**
 * @file preprocess_worker.h
 * @brief 预处理工作器（每个线程一个）
 * @version 2.1.0（评审修正版）
 * @date 2026-02-16
 * @author 技术觉醒团队
 * @note 所属系列: data
 * 
 * 修正记录（V2.1）：
 * - 采纳B1+B2+B5意见1：EngineBuffer增加batch_id参数
 * - 采纳B1+B3+B5意见2：S区改为独立索引管理
 * - 采纳B1+B5意见4：update_parameters重置total_samples_processed_
 * - 采纳B3+B5意见5：update_parameters重置c_read_idx_
 * - 采纳B5意见1：延迟First-Touch到绑核后
 * - 采纳B4+B5意见6、7：S区洗牌seed修正
 * - 采纳B3意见8：Generator延迟初始化
 */

#pragma once

#include "renaissance/data/preprocess_worker_parameter.h"
#include "renaissance/transforms/preprocess_operation.h"
#include "renaissance/base/rng.h"
#include "renaissance/base/global_config.h"
#include "renaissance/base/tr_exception.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <turbojpeg.h>

namespace tr {

// 前向声明
class EngineBuffer;

/**
 * @class PreprocessWorker
 * @brief 预处理工作器
 * 
 * 核心职责：
 * 1. 管理Workshop内存（D/A/B/T/S/C区）
 * 2. 执行JPEG解码（完整/局部）
 * 3. 调用PO链进行图像变换
 * 4. 管理SDMP缓存和CPVS缓存
 * 5. 零竞争写入EngineBuffer（带批次保护）
 * 
 * 内存布局（一次分配，永久存在）：
 * ┌──────┬──────┬──────┬──────┬─────────┬──────┐
 * │  D区  │  A区  │  B区  │  T区  │ S1~Sn区 │  C区  │
 * └──────┴──────┴──────┴──────┴─────────┴──────┘
 *  解码   Ping   Pong   临时   SDMP缓存  CPVS缓存
 * 
 * NUMA优化（V2.1修正）：
 * - 构造时分配内存（不立即memset）
 * - 首次work()时触发First-Touch（确保在绑核后）
 * - 线程绑核确保访问本地内存
 * 
 * 修正说明（V2.1）：
 * - S区管理改为完全独立：s_shuffle_indices_[num_region_s]
 * - S区标签改为一维：region_s_labels_（所有S区共享，标签相同）
 * - Generator延迟初始化：ensure_rng_initialized()
 * - First-Touch延迟执行：ensure_first_touch()
 */
class PreprocessWorker {
public:
    /**
     * @struct Config
     * @brief PW构造配置（一次性，不可变）
     */
    struct Config {
        int worker_id;              ///< 线程ID（全局唯一）
        int engine_id;              ///< 对应的Engine ID = worker_id % world_size
        int pid_in_engine;          ///< 在Engine内的编号
        
        // ==================== Workshop各区大小（字节）====================
        size_t region_d_size;       ///< D区大小
        size_t region_ab_size;      ///< A/B区大小（相等）
        size_t region_t_size;       ///< T区大小
        size_t region_s_size;       ///< 单个S区大小
        size_t region_c_size;       ///< C区大小
        int num_region_s;           ///< S区数量 = sdmp_factor - 1
        
        // ==================== 固定配置（从GlobalRegistry复制）====================
        int local_batch_size;
        int world_size;
        int sdmp_factor;
        bool using_cpvs;
        int num_workers_per_engine;
        
        // ==================== 数据集信息 ====================
        DatasetType dataset_type;   ///< 数据集类型（用于判断是否需要解码）
        int num_color_channels;     ///< 通常是3
        int raw_image_width;        ///< 非ImageNet数据集的原始宽度
        int raw_image_height;       ///< 非ImageNet数据集的原始高度
    };
    
    /**
     * @brief 构造函数
     * @param config 配置参数
     * @param train_ops 训练集PO列表（会被克隆）
     * @param val_ops 验证集PO列表（会被克隆）
     */
    PreprocessWorker(
        const Config& config,
        const std::vector<std::unique_ptr<transforms::PreprocessOperation>>& train_ops,
        const std::vector<std::unique_ptr<transforms::PreprocessOperation>>& val_ops
    );
    
    ~PreprocessWorker();
    
    // 禁止拷贝和移动
    PreprocessWorker(const PreprocessWorker&) = delete;
    PreprocessWorker& operator=(const PreprocessWorker&) = delete;
    
    // =========================================================================
    // 参数更新
    // =========================================================================
    
    /**
     * @brief 更新运行时参数（每个phase之初调用）
     * @param param 新参数
     * 
     * 修正（V2.1）：
     * - 采纳B1+B5意见4：重置total_samples_processed_为0
     * - 采纳B3+B5意见5：验证阶段重置c_read_idx_为0
     */
    void update_parameters(const PreprocessWorkerParameter& param);
    
    // =========================================================================
    // 洗牌方法
    // =========================================================================
    
    /**
     * @brief 洗牌S区（Lazy epoch的train phase开始时，主线程调用）
     * @param s_region_idx S区索引（0 ~ num_region_s-1）
     * @param epoch_id 当前epoch ID（用于seed）
     * 
     * 修正（V2.1）：
     * - 采纳B1+B3+B5意见2：使用独立的s_shuffle_indices_[s_region_idx]
     * - 采纳B4+B5意见7：seed只使用epoch_id，移除s_region_idx
     */
    void shuffle_s_region(int s_region_idx, int epoch_id);
    
    // =========================================================================
    // 核心工作方法
    // =========================================================================
    
    /**
     * @brief 执行预处理工作（处理单个样本）
     * @param label 样本标签
     * @param data_ptr 样本数据（JPEG或RAW）
     * @param data_size 样本数据大小
     * @param engine_buffer 目标EngineBuffer
     * @return true=成功, false=epoch结束或错误
     * 
     * 工作流程：
     * Busy epoch:
     *   解码 → PO链×sdmp_factor次 → 输出到S区×(sdmp_factor-1) + EngineBuffer×1
     * Lazy epoch:
     *   读S区 → 写EngineBuffer
     * CPVS首次:
     *   解码 → PO链 → 写C区 + EngineBuffer
     * CPVS非首次:
     *   读C区 → 写EngineBuffer
     * 
     * 修正（V2.1）：
     * - 采纳B5意见1：首次调用时执行ensure_first_touch()
     * - 采纳B3意见8：首次调用时执行ensure_rng_initialized()
     * - 采纳B1+B2+B5意见1：write_at传递batch_id
     */
    bool work(
        int32_t label,
        const uint8_t* data_ptr,
        size_t data_size,
        EngineBuffer& engine_buffer
    );
    
    // =========================================================================
    // 状态查询
    // =========================================================================
    
    int worker_id() const { return config_.worker_id; }
    int engine_id() const { return config_.engine_id; }
    int pid_in_engine() const { return config_.pid_in_engine; }
    size_t total_samples_processed() const { return total_samples_processed_; }
    size_t num_samples_in_s() const { return num_samples_in_s_; }
    size_t num_samples_in_c() const { return num_samples_in_c_; }

private:
    // =========================================================================
    // 配置和参数
    // =========================================================================
    
    Config config_;
    PreprocessWorkerParameter param_;
    
    // =========================================================================
    // Workshop内存
    // =========================================================================
    
    uint8_t* workshop_ = nullptr;      ///< 统一分配的内存块
    size_t workshop_size_ = 0;         ///< 总大小
    
    uint8_t* region_d_ = nullptr;      ///< D区指针
    uint8_t* region_a_ = nullptr;      ///< A区指针
    uint8_t* region_b_ = nullptr;      ///< B区指针
    uint8_t* region_t_ = nullptr;      ///< T区指针（可选）
    std::vector<uint8_t*> region_s_;   ///< S区指针数组
    uint8_t* region_c_ = nullptr;      ///< C区指针（可选）
    
    // =========================================================================
    // S区和C区管理
    // =========================================================================
    
    // ✅ 修正（V2.1）：S区完全独立管理
    std::vector<std::vector<size_t>> s_shuffle_indices_;  ///< 每个S区独立的洗牌索引
    std::vector<size_t> s_read_indices_;                  ///< 每个S区的独立读取索引
    std::vector<size_t> num_samples_per_s_;               ///< 每个S区的实际样本数
    
    std::vector<int32_t> region_s_labels_;  ///< S区标签（一维，所有S区共享）
    std::vector<int32_t> region_c_labels_;  ///< C区标签
    
    size_t num_samples_in_s_ = 0;      ///< S区实际样本数（所有S区应相同）
    size_t num_samples_in_c_ = 0;      ///< C区实际样本数
    size_t c_read_idx_ = 0;            ///< C区读取索引
    
    // =========================================================================
    // 预处理操作链
    // =========================================================================
    
    std::vector<std::unique_ptr<transforms::PreprocessOperation>> train_ops_;
    std::vector<std::unique_ptr<transforms::PreprocessOperation>> val_ops_;
    
    // =========================================================================
    // TurboJPEG解码器
    // =========================================================================
    
    tjhandle tj_handle_ = nullptr;     ///< 持久句柄（TurboJPEG 3.x API）
    
    // =========================================================================
    // 随机数生成器
    // =========================================================================
    
    Generator rng_;                    ///< 独立Generator（seed = base_seed ^ (worker_id << 16)）
    bool rng_initialized_ = false;     ///< ✅ 新增（V2.1）：RNG是否已初始化
    
    // =========================================================================
    // NUMA优化标志
    // =========================================================================
    
    bool workshop_touched_ = false;    ///< ✅ 新增（V2.1）：First-Touch是否已执行
    
    // =========================================================================
    // 统计
    // =========================================================================
    
    size_t total_samples_processed_ = 0;  ///< 全局样本计数（用于位置计算）
    
    // =========================================================================
    // 内部方法
    // =========================================================================
    
    /**
     * @brief 分配Workshop内存（64字节对齐，不立即First-Touch）
     * 
     * 修正（V2.1）：
     * - 采纳B5意见1：分离分配与First-Touch
     * - S区大小对齐到4KB页边界
     */
    void allocate_workshop();
    
    /**
     * @brief 释放Workshop内存
     */
    void free_workshop();
    
    /**
     * @brief 确保First-Touch已执行（在绑核后首次调用）
     * 
     * ✅ 新增（V2.1）：
     * - 延迟到worker线程绑核后
     * - 确保内存分配在本地NUMA节点
     */
    void ensure_first_touch();
    
    /**
     * @brief 确保Generator已初始化（首次work()时调用）
     * 
     * ✅ 新增（V2.1）：
     * - 延迟初始化，使用用户设置的seed
     * - 避免构造时使用默认seed=0
     */
    void ensure_rng_initialized();
    
    /**
     * @brief 完整解码JPEG到D区
     */
    bool decode_full(
        const uint8_t* jpeg_data,
        size_t jpeg_size,
        int32_t& width,
        int32_t& height,
        size_t& stride
    );
    
    /**
     * @brief 局部解码JPEG到D区
     */
    bool decode_partial(
        const uint8_t* jpeg_data,
        size_t jpeg_size,
        const transforms::DecodeStrategy& strategy,
        size_t& stride
    );
    
    /**
     * @brief 执行PO链（AB区乒乓 + RandomHorizontalFlip优化）
     * 
     * 修正（V2.1）：
     * - 采纳B2+B5意见3：增强Flip调用逻辑的文档说明
     */
    void execute_po_chain(
        const uint8_t* initial_ptr,
        int32_t initial_width,
        int32_t initial_height,
        size_t initial_stride,
        uint8_t* final_output_ptr,
        size_t final_stride,
        const std::vector<std::unique_ptr<transforms::PreprocessOperation>>& ops,
        int32_t& final_width,
        int32_t& final_height
    );
    
    /**
     * @brief 计算在EngineBuffer中的写入位置
     * @return {position, is_valid}
     * 
     * 核心公式：position = (n * M + j) % B
     * - n: 该PW的全局样本计数（total_samples_processed_）
     * - M: num_workers_per_engine
     * - j: pid_in_engine
     * - B: local_batch_size
     * 
     * 零竞争证明：
     * 不同PW的j值不同，且n是各自独立的计数器，
     * 所以(n₁*M + j₁) ≠ (n₂*M + j₂) (mod B)，
     * 即不同PW永远不会写入同一位置 ∎
     */
    std::pair<int, bool> calculate_write_position();
};

} // namespace tr
```

---

