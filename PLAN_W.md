# 【PreprocessWorker（PW）实施计划】

**版本**: V1.0
**日期**: 2026-02-17
**状态**: 准备实施
**预期工期**: P0阶段（约1周）

---

## 【一、设计原则确认】

### 1.1 目录结构
- **新增代码位置**: `src/data/` 目录（不新建transform目录）
- **测试代码位置**: `tests/pw/` 目录（新建）
- **命名空间**: `namespace tr`（与现有data模块一致）

### 1.2 第一阶段范围（P0）
**只实现以下内容**：
- ✅ `PreprocessOperation` 抽象基类
- ✅ `DecodeStrategy` 结构体
- ✅ `Resize` 类（双线性插值，Simd加速）
- ✅ `CenterCrop` 类（中心裁剪）
- ✅ `DoNothing` 占位类
- ✅ `test_po.cpp` 测试程序

**暂不实现**：
- ❌ `PreprocessWorker` 类
- ❌ `EngineBuffer` 类
- ❌ `RandomHorizontalFlip` 类
- ❌ SDMP、CPVS优化逻辑

---

## 【二、文件清单】

### 2.1 新增头文件（include/renaissance/data/）

| 文件名 | 说明 |
|--------|------|
| `preprocess_operation.h` | PO抽象基类 |
| `decode_strategy.h` | JPEG解码策略结构体 |

### 2.2 新增源文件（src/data/）

| 文件名 | 说明 |
|--------|------|
| `preprocess_operation.cpp` | PO基类实现（空实现，只有虚析构） |
| `resize.cpp` | Resize类实现 |
| `center_crop.cpp` | CenterCrop类实现 |
| `do_nothing.cpp` | DoNothing类实现 |

### 2.3 测试文件（tests/pw/）

| 文件名 | 说明 |
|--------|------|
| `test_po.cpp` | PO单元测试程序 |

### 2.4 修改文件

| 文件 | 修改内容 |
|------|----------|
| `src/data/CMakeLists.txt` | 添加新的源文件到DATA_SOURCES |
| `tests/CMakeLists.txt` | 添加tests/pw子目录 |
| `tests/pw/CMakeLists.txt` | 新建，定义test_po目标 |
| `include/renaissance/renaissance.h` | 添加新头文件引用 |

---

## 【三、头文件详细设计】

### 3.1 decode_strategy.h

```cpp
/**
 * @file decode_strategy.h
 * @brief JPEG解码策略封装
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include <cstdint>

namespace tr {

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
 */
struct DecodeStrategy {
    bool need_decode = false;       ///< 是否需要解码（非ImageNet为false）
    bool use_partial = false;       ///< 局部解码vs完整解码

    // MCU对齐的解码窗口（8的倍数）
    int32_t decode_x = 0;           ///< 解码起始X（MCU对齐，向下取整）
    int32_t decode_y = 0;           ///< 解码起始Y（MCU对齐，向下取整）
    int32_t decode_w = 0;           ///< 解码宽度（MCU对齐，向上取整）
    int32_t decode_h = 0;           ///< 解码高度（MCU对齐，向上取整）

    // 精确裁剪窗口（相对于解码窗口的偏移）
    int32_t crop_x = 0;             ///< 裁剪起始X（相对于decode_x）
    int32_t crop_y = 0;             ///< 裁剪起始Y（相对于decode_y）
    int32_t crop_w = 0;             ///< 裁剪宽度
    int32_t crop_h = 0;             ///< 裁剪高度
};

} // namespace tr
```

---

### 3.2 preprocess_operation.h

```cpp
/**
 * @file preprocess_operation.h
 * @brief 预处理操作抽象基类
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/decode_strategy.h"
#include "renaissance/base/rng.h"
#include <cstdint>
#include <string>
#include <memory>

namespace tr {

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

} // namespace tr
```

---

### 3.3 resize.h（内联在preprocess_operation.cpp之后）

```cpp
/**
 * @file resize.h
 * @brief Resize操作（双线性插值，Simd加速）
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/base/tr_exception.h"
#include <Simd/SimdLib.h>

namespace tr {

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
    ) override;

    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<Resize>(output_size_);
    }

    std::string name() const override { return "Resize"; }
    bool introduce_randomness() const override { return false; }
    bool is_resize() const override { return true; }

    void set_output_size(int size) override;
    int get_output_size() const override { return output_size_; }

    DecodeStrategy get_decode_strategy(
        int32_t image_width,
        int32_t image_height,
        int sdmp_factor,
        Generator* rng
    ) const override;

    ~Resize();

private:
    int output_size_;

    // Simd上下文缓存
    mutable void* resizer_cache_ = nullptr;
    mutable int cached_src_w_ = 0;
    mutable int cached_src_h_ = 0;
    mutable int cached_dst_w_ = 0;
    mutable int cached_dst_h_ = 0;
};

} // namespace tr
```

---

### 3.4 center_crop.h

```cpp
/**
 * @file center_crop.h
 * @brief 中心裁剪操作
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include <algorithm>
#include <cstring>

namespace tr {

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
    ) override;

    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<CenterCrop>(output_size_);
    }

    std::string name() const override { return "CenterCrop"; }
    bool introduce_randomness() const override { return false; }
    bool is_crop() const override { return true; }

    void set_output_size(int size) override { output_size_ = size; }
    int get_output_size() const override { return output_size_; }

    DecodeStrategy get_decode_strategy(
        int32_t image_width,
        int32_t image_height,
        int sdmp_factor,
        Generator* rng
    ) const override;

private:
    int output_size_;
};

} // namespace tr
```

---

### 3.5 do_nothing.h

```cpp
/**
 * @file do_nothing.h
 * @brief 占位操作（直接复制）
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include <cstring>

namespace tr {

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
    ) override;

    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<DoNothing>();
    }

    std::string name() const override { return "DoNothing"; }
    bool introduce_randomness() const override { return false; }
};

} // namespace tr
```

---

## 【四、源文件实现要点】

### 4.1 resize.cpp

**关键点**：
1. 使用`SimdResizerInit`创建resizer，缓存上下文
2. 检查缓存有效性：`cached_src_w_ != input_width || ...`
3. 缓存失效时调用`SimdRelease`旧resizer，创建新的
4. 使用`SimdResizerRun`执行resize
5. `set_output_size`时清空缓存
6. `get_decode_strategy`返回完整解码策略

### 4.2 center_crop.cpp

**关键点**：
1. 计算中心裁剪区域：`start_x = (width - crop_w) / 2`
2. 逐行memcpy处理stride可能不同的情况
3. `get_decode_strategy`返回局部解码策略：
   - decode窗口MCU对齐
   - crop窗口相对偏移
   - 边界检查防止越界

### 4.3 do_nothing.cpp

**关键点**：
1. 纯memcpy，处理stride
2. 返回原始尺寸

---

## 【五、测试程序设计】

### 5.1 test_po.cpp 功能需求

**命令行参数**：
```
--input <PATH>      输入图片路径（默认：input.jpg）
--output <PATH>     输出图片路径（默认：workspace/output.jpg）
--po <NAME>         要测试的PO（Resize/CenterCrop/DoNothing）
--size <N>          PO的输出尺寸（默认：224）
--help              显示帮助信息
```

**功能**：
1. 使用TurboJPEG 3.x API读取JPEG头获取图片尺寸
2. 调用PO的`get_decode_strategy()`获取解码策略（完整解码/局部解码）
3. 根据解码策略执行JPEG解码（使用`tj3Decompress8`或`tj3SetCroppingRegion`）
4. 创建输出缓冲区（64字节对齐stride）
5. 执行PO的`execute()`方法
6. 保存结果为JPEG格式（使用TurboJPEG编码）
7. 输出处理时间和尺寸信息

**设计原则**：
- 只测试单个PO，不测试串联
- 串联逻辑由PreprocessWorker负责，在P1阶段测试
- 测试程序验证PO本身的功能正确性

**示例命令**：
```bash
# 测试Resize（完整解码）
./test_po --po Resize --size 224

# 测试CenterCrop（局部解码）
./test_po --po CenterCrop --size 224

# 自定义输入输出
./test_po --input custom.jpg --output result.jpg --po Resize --size 128
```

### 5.2 test_po.cpp 实现要点

1. **TurboJPEG解码实现**：
   - 使用`tj3Init`初始化TurboJPEG句柄
   - 使用`tj3DecompressHeader`读取原始图片尺寸
   - 调用PO的`get_decode_strategy(image_w, image_h, 1, nullptr)`获取解码策略

2. **根据解码策略解码**：
   - 如果`strategy.need_decode = false`：直接使用输入数据（非ImageNet场景）
   - 如果`strategy.use_partial = true`：局部解码
     - 使用`tj3SetCroppingRegion`设置裁剪区域
     - 使用`tj3Decompress8`解码
   - 如果`strategy.use_partial = false`：完整解码
     - 直接使用`tj3Decompress8`解码整张图

3. **缓冲区管理**：
   - 计算对齐的stride：`PreprocessOperation::calculate_stride()`
   - 分配输出缓冲区（64字节对齐）

4. **PO创建和执行**：
   - 使用工厂模式：`if (name == "Resize") return std::make_unique<Resize>(size);`
   - 支持的PO：Resize、CenterCrop、DoNothing
   - 调用`po->execute(...)`执行预处理

5. **保存结果**：
   - 使用`tj3Compress8`压缩为JPEG
   - 写入到指定路径（确保workspace目录存在）

---

## 【六、CMakeLists.txt 修改】

### 6.1 src/data/CMakeLists.txt

```cmake
# 在DATA_SOURCES中添加：
set(DATA_SOURCES
    shape.cpp
    storage.cpp
    tensor.cpp
    preprocessor.cpp
    # ... 现有文件 ...
    # P0阶段新增：PO实现
    preprocess_operation.cpp
    resize.cpp
    center_crop.cpp
    do_nothing.cpp
)
```

### 6.2 tests/pw/CMakeLists.txt（新建）

```cmake
# ============================================================================
# PW测试模块
# ============================================================================

# 定义test_po可执行文件
add_executable(test_po test_po.cpp)

# 链接renaissance库
target_link_libraries(test_po PRIVATE renaissance)

# 设置输出目录
set_target_properties(test_po PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}/pw
)

# 安装目标
install(TARGETS test_po RUNTIME DESTINATION bin/tests/pw)
```

### 6.3 tests/CMakeLists.txt

```cmake
# 添加pw子目录
add_subdirectory(pw)
```

---

## 【七、实施步骤】

### 步骤1：创建头文件（第1天）
- [ ] 创建`include/renaissance/data/decode_strategy.h`
- [ ] 创建`include/renaissance/data/preprocess_operation.h`
- [ ] 创建`include/renaissance/data/resize.h`
- [ ] 创建`include/renaissance/data/center_crop.h`
- [ ] 创建`include/renaissance/data/do_nothing.h`
- [ ] 修改`include/renaissance/renaissance.h`添加新头文件引用

### 步骤2：实现源文件（第2-3天）
- [ ] 创建`src/data/preprocess_operation.cpp`（空实现）
- [ ] 创建`src/data/resize.cpp`
- [ ] 创建`src/data/center_crop.cpp`
- [ ] 创建`src/data/do_nothing.cpp`
- [ ] 修改`src/data/CMakeLists.txt`添加新源文件

### 步骤3：创建测试程序（第4-5天）
- [ ] 创建`tests/pw/`目录
- [ ] 创建`tests/pw/CMakeLists.txt`
- [ ] 创建`tests/pw/test_po.cpp`
- [ ] 修改`tests/CMakeLists.txt`添加pw子目录

### 步骤4：编译和测试（第6-7天）
- [ ] 编译项目
- [ ] 运行`test_po --po1 Resize --size1 224`
- [ ] 运行`test_po --po1 CenterCrop --size1 224`
- [ ] 运行`test_po --po1 CenterCrop --size1 256 --po2 Resize --size2 224`
- [ ] 验证输出图片正确性
- [ ] 测试性能（处理时间）

---

## 【八、验收标准】

### 8.1 编译通过
- 无编译错误
- 无警告（或仅有Simd库的警告）

### 8.2 功能正确
- Resize：输出图片尺寸正确，图像质量良好
- CenterCrop：输出居中裁剪，尺寸正确
- DoNothing：输出与输入一致（仅用于测试框架）

### 8.3 性能达标
- Resize处理时间 < 1ms/image（224x224）
- CenterCrop处理时间 < 0.2ms/image

### 8.4 输出正确
- workspace/output.jpg 成功生成
- 图片格式正确（可用图像查看器打开）

---

## 【九、后续工作（P1阶段准备）**

完成P0阶段后，下一步将实现：
- PreprocessWorker类框架
- Workshop内存管理
- TurboJPEG局部解码集成
- EngineBuffer基础设计

---

**施工计划完毕，准备实施** 🚀
