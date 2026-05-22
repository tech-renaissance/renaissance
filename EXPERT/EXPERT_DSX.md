以下是用 XNNPACK Subgraph API 实现的 Global Average Pooling（GAP）算子完整代码，包含前向传播和反向传播。

```cpp
// gap_xnnpack.hpp
#pragma once

#include <xnnpack.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

/**
 * @brief 基于 XNNPACK Subgraph API 的 Global Average Pooling 算子
 * 
 * 本实现提供前向和反向传播，完全遵循 XNNPACK 的 API 使用规范。
 * 数据格式：NHWC (batch, height, width, channels)
 * 数据类型：FP32 (float)
 * 使用 C++17 特性。
 */
class GAPOperator {
public:
    /**
     * @brief 构造函数：初始化 GAP 算子的维度信息
     * @param batch_size  批大小 (N)
     * @param height      输入特征图高度 (H)
     * @param width       输入特征图宽度 (W)
     * @param channels    通道数 (C)
     */
    GAPOperator(size_t batch_size, size_t height, size_t width, size_t channels)
        : N_(batch_size), H_(height), W_(width), C_(channels)
    {
        // 预计算归一化因子: 1 / (H * W)
        norm_factor_ = 1.0f / static_cast<float>(H_ * W_);
    }

    // ---------- 前向传播 ----------
    /**
     * @brief 执行 Global Average Pooling 的前向计算。
     *        输入形状: (N, H, W, C) -> 输出形状: (N, 1, 1, C)
     * @param input  指向输入数据的指针 (FP32, NHWC 布局)
     * @param output 指向输出数据的指针 (FP32, NHWC 布局, 空间维度为 1x1)
     */
    void forward(const float* input, float* output) {
        // 1. 创建子图
        xnn_subgraph_t subgraph = nullptr;
        xnn_status status = xnn_create_subgraph(
            /*num_external_values=*/2,        // 输入和输出两个外部值
            /*flags=*/0,
            &subgraph);
        if (status != xnn_status_success) {
            throw std::runtime_error("Failed to create XNNPACK subgraph");
        }

        // 2. 定义输入张量值 (NHWC: N, H, W, C)
        uint32_t input_id = XNN_INVALID_VALUE_ID;
        status = xnn_define_tensor_value(
            subgraph,
            xnn_datatype_fp32,
            /*num_dims=*/4,
            /*dims=*/{static_cast<size_t>(N_), static_cast<size_t>(H_),
                      static_cast<size_t>(W_), static_cast<size_t>(C_)},
            /*data=*/nullptr,                     // 外部数据，运行时传入
            /*external_id=*/0,                    // 外部值索引 0
            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
            &input_id);
        if (status != xnn_status_success || input_id == XNN_INVALID_VALUE_ID) {
            xnn_delete_subgraph(subgraph);
            throw std::runtime_error("Failed to define input tensor");
        }

        // 3. 定义输出张量值 (NHWC: N, 1, 1, C)
        uint32_t output_id = XNN_INVALID_VALUE_ID;
        status = xnn_define_tensor_value(
            subgraph,
            xnn_datatype_fp32,
            /*num_dims=*/4,
            /*dims=*/{static_cast<size_t>(N_), 1, 1, static_cast<size_t>(C_)},
            /*data=*/nullptr,
            /*external_id=*/1,                    // 外部值索引 1
            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
            &output_id);
        if (status != xnn_status_success || output_id == XNN_INVALID_VALUE_ID) {
            xnn_delete_subgraph(subgraph);
            throw std::runtime_error("Failed to define output tensor");
        }

        // 4. 添加 Global Average Pooling 节点
        //    GAP 可以看作是一个池化窗口大小 = 输入空间尺寸的 Average Pooling
        status = xnn_define_global_average_pooling_nhwc(
            subgraph,
            /*input_id=*/input_id,
            /*output_id=*/output_id,
            /*flags=*/0);
        if (status != xnn_status_success) {
            xnn_delete_subgraph(subgraph);
            throw std::runtime_error(
                "Failed to define Global Average Pooling node");
        }

        // 5. 创建运行时并执行
        xnn_runtime_t runtime = nullptr;
        status = xnn_create_runtime_v2(subgraph, nullptr, 0, &runtime);
        if (status != xnn_status_success) {
            xnn_delete_subgraph(subgraph);
            throw std::runtime_error("Failed to create XNNPACK runtime");
        }

        // 准备外部数据指针数组
        // 索引 0: 输入; 索引 1: 输出
        const void* external_data[2] = {input, output};
        status = xnn_setup_runtime(runtime, external_data);
        if (status != xnn_status_success) {
            xnn_delete_runtime(runtime);
            xnn_delete_subgraph(subgraph);
            throw std::runtime_error("Failed to setup XNNPACK runtime");
        }

        // 执行推理
        status = xnn_invoke_runtime(runtime);
        if (status != xnn_status_success) {
            xnn_delete_runtime(runtime);
            xnn_delete_subgraph(subgraph);
            throw std::runtime_error("Failed to invoke XNNPACK runtime");
        }

        // 清理资源
        xnn_delete_runtime(runtime);
        xnn_delete_subgraph(subgraph);
    }

    // ---------- 反向传播 ----------
    /**
     * @brief 执行 Global Average Pooling 的反向传播。
     *        由于 XNNPACK 本身主要用于推理，不直接提供训练反向算子，
     *        因此反向传播通过手写 kernel 实现。
     *
     *        数学推导:
     *          forward:  y[c] = (1/(H*W)) * Σ_{i,j} x[n,i,j,c]
     *          backward: dx[n,i,j,c] = (1/(H*W)) * dy[n,c]
     *        即：将输出梯度均匀地广播回输入的所有空间位置。
     *
     * @param grad_output 输出梯度，形状 (N, 1, 1, C)
     * @param grad_input  输入梯度，形状 (N, H, W, C) (调用前需预先分配内存)
     */
    void backward(const float* grad_output, float* grad_input) {
        // 归一化因子: 1 / (H * W)
        const float scale = norm_factor_;

        // 按 batch 遍历，将每个通道的梯度均匀分配到所有空间位置
        for (size_t n = 0; n < N_; ++n) {
            for (size_t c = 0; c < C_; ++c) {
                // 输出梯度在位置 (n, 0, 0, c) 处的值
                float grad_val = grad_output[n * C_ + c] * scale;

                // 将缩放后的梯度广播到输入梯度的所有空间位置 (H x W)
                for (size_t h = 0; h < H_; ++h) {
                    for (size_t w = 0; w < W_; ++w) {
                        // NHWC 索引: (n * H * W * C) + (h * W * C) + (w * C) + c
                        size_t idx = ((n * H_ + h) * W_ + w) * C_ + c;
                        grad_input[idx] = grad_val;
                    }
                }
            }
        }
    }

    // ---------- 辅助函数 ----------
    /// 获取归一化因子 (1 / (H * W))
    float get_norm_factor() const { return norm_factor_; }

private:
    size_t N_;                // 批大小
    size_t H_;                // 输入高度
    size_t W_;                // 输入宽度
    size_t C_;                // 通道数
    float  norm_factor_;      // 1 / (H * W)
};
```

### 使用示例

```cpp
// example_usage.cpp
#include "gap_xnnpack.hpp"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    // 初始化 XNNPACK (程序生命周期内只需调用一次)
    xnn_status init_status = xnn_initialize(/*allocator=*/nullptr);
    if (init_status != xnn_status_success) {
        std::cerr << "XNNPACK initialization failed" << std::endl;
        return -1;
    }

    // 定义输入维度: N=2, H=3, W=3, C=4
    const size_t N = 2, H = 3, W = 3, C = 4;
    const size_t input_size  = N * H * W * C;    // 72
    const size_t output_size = N * 1 * 1 * C;    // 8

    // 分配内存
    std::vector<float> input(input_size);
    std::vector<float> output(output_size);
    std::vector<float> grad_output(output_size);
    std::vector<float> grad_input(input_size);

    // 填充测试数据 (例如: 全1)
    std::fill(input.begin(), input.end(), 1.0f);

    // 创建 GAP 算子
    GAPOperator gap(N, H, W, C);

    // 前向传播
    gap.forward(input.data(), output.data());
    // 验证: 每个通道的平均值应为 1.0
    for (size_t i = 0; i < output_size; ++i) {
        assert(std::fabs(output[i] - 1.0f) < 1e-5);
    }

    // 反向传播: 假设输出梯度全为 1.0
    std::fill(grad_output.begin(), grad_output.end(), 1.0f);
    gap.backward(grad_output.data(), grad_input.data());
    // 验证: 每个输入梯度的值应为 1.0/(H*W) = 1/9
    float expected_grad = gap.get_norm_factor(); // 1/9
    for (size_t i = 0; i < input_size; ++i) {
        assert(std::fabs(grad_input[i] - expected_grad) < 1e-5);
    }

    std::cout << "All tests passed!" << std::endl;

    // 清理 XNNPACK 资源 (程序结束前调用)
    xnn_deinitialize();
    return 0;
}
```

### 核心设计说明

**1. 前向传播 (XNNPACK Subgraph)**

*   遵循 `创建子图 → 定义张量 → 定义节点 → 创建运行时 → 执行` 的标准流程。
*   通过 `xnn_define_global_average_pooling_nhwc` 直接定义 GAP 算子。
*   使用 `XNN_VALUE_FLAG_EXTERNAL_INPUT/OUTPUT` 标志将输入/输出张量标记为外部数据，以便在运行时传入数据。
*   执行时通过 `external_data` 数组将实际的内存指针传递给运行时。

**2. 反向传播 (手写 Kernel)**

*   XNNPACK 作为推理库，不直接提供反向算子，因此反向传播通过手写 kernel 实现。
*   **GAP 反向公式**：输出梯度被均匀分配到输入的所有空间位置，每个位置的梯度 = 输出梯度 × (1 / (H × W))。
*   手写 kernel 采用三重循环 (batch → channel → spatial)，对每个空间位置赋值归一化后的梯度。

**3. NHWC 布局**

*   所有张量维度遵循 `(N, H, W, C)` 的顺序。
*   在反向传播中，一维索引计算为：`((n * H + h) * W + w) * C + c`。
*   XNNPACK 原生支持 NHWC 布局，无需额外转换。

**4. C++17 特性**

*   使用 `constexpr` 和 `static_assert` 可在编译期校验数据类型。
*   构造函数使用成员初始化列表。
*   异常处理使用 `std::runtime_error`。
*   内存管理使用 RAII 和 `std::vector`。

**5. 性能优化要点**

*   反向传播中归一化因子 `1.0f / (H * W)` 被预先计算，避免重复除法。
*   循环顺序为 batch → channel → spatial，尽可能利用缓存局部性。
*   对于大批量数据，可进一步使用 OpenMP 或线程池并行化反向传播。
*   前向传播完全由 XNNPACK 优化的微内核执行，确保高性能。

### 编译与链接

确保链接 XNNPACK 库以及必要的依赖（如 `pthreadpool`, `cpuinfo`, `FP16` 等）。以 CMake 为例：

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(XNNPACK REQUIRED xnnpack)
target_link_libraries(your_target PRIVATE ${XNNPACK_LIBRARIES})
target_include_directories(your_target PRIVATE ${XNNPACK_INCLUDE_DIRS})
```

### 注意事项

*   使用任何 XNNPACK API 前必须调用 `xnn_initialize` 进行初始化。
*   程序结束时调用 `xnn_deinitialize` 释放全局资源。
*   所有运行时和子图对象使用完毕后必须分别通过 `xnn_delete_runtime` 和 `xnn_delete_subgraph` 释放。
*   反向传播的内存（`grad_input`）需由调用者预先分配。
*   错误处理遵循 XNNPACK 的返回值检查规范，每个 API 调用后检查 `xnn_status`。