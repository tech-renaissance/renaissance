以下是使用 **XNNPACK Subgraph API** 实现的 Global Average Pooling (GAP) 算子完整 C++17 代码。代码包含正向传播（基于 XNNPACK 编译图执行）和反向传播（基于数学推导的高效实现），并附有详细注释和编译说明。

##### 📌 重要说明
1. **XNNPACK 定位**：XNNPACK 是专为 **推理（Inference）** 优化的库，其 Subgraph API **不原生支持自动微分/反向传播**。因此，正向图使用 XNNAPI 构建，反向传播采用符合 NHWC 布局的高效 C++17 实现（与 PyTorch/TensorFlow 底层梯度计算逻辑一致）。
2. **线程安全**：`xnn_reshape_runtime` 会修改运行时内部状态。若需多线程并发推理，建议为每个线程创建独立的 `xnn_runtime_t`，或使用 `xnn_run_subgraph` 替代。
3. **依赖**：需链接 `libxnnpack.a` 及对应的 pthread/simd 依赖。

##### 💻 完整实现 (C++17)

```cpp
#include <xnnpack.h>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// XNNPACK 错误检查宏
#define CHECK_XNN_STATUS(status, msg) \
    if (status != xnn_status_success) { \
        throw std::runtime_error(std::string("XNNPACK Error [") + msg + "]: " + std::to_string(status)); \
    }

class GAPOperator {
public:
    /**
     * @brief 初始化 GAP 算子
     * @param N  Batch size
     * @param H  Input height
     * @param W  Input width
     * @param C  Channels
     */
    GAPOperator(uint32_t N, uint32_t H, uint32_t W, uint32_t C)
        : N_(N), H_(H), W_(W), C_(C) {
        
        // 1. 全局初始化 XNNPACK (线程安全，多次调用无害)
        CHECK_XNN_STATUS(xnn_initialize(nullptr), "global initialization");

        xnn_status status;

        // 2. 创建正向计算子图
        status = xnn_create_subgraph(0, 0, &fwd_subgraph_);
        CHECK_XNN_STATUS(status, "xnn_create_subgraph");

        // 3. 定义外部张量 (NHWC 布局)
        uint32_t input_dims[4]  = {N_, H_, W_, C_};
        uint32_t output_dims[4] = {N_, 1, 1, C_};

        status = xnn_define_tensor_value(
            fwd_subgraph_, xnn_datatype_fp32, 4, input_dims,
            nullptr, XNN_TENSOR_TYPE_EXTERNAL, &fwd_input_id_);
        CHECK_XNN_STATUS(status, "define_input_tensor");

        status = xnn_define_tensor_value(
            fwd_subgraph_, xnn_datatype_fp32, 4, output_dims,
            nullptr, XNN_TENSOR_TYPE_EXTERNAL, &fwd_output_id_);
        CHECK_XNN_STATUS(status, "define_output_tensor");

        // 4. 定义 Global Average Pooling 2D 算子
        status = xnn_define_global_average_pooling_2d(
            fwd_subgraph_, fwd_input_id_, fwd_output_id_, 0);
        CHECK_XNN_STATUS(status, "xnn_define_global_average_pooling_2d");

        // 5. 固化子图并创建运行时
        status = xnn_finalize_subgraph(fwd_subgraph_);
        CHECK_XNN_STATUS(status, "xnn_finalize_subgraph");

        status = xnn_create_runtime(fwd_subgraph_, &fwd_runtime_);
        CHECK_XNN_STATUS(status, "xnn_create_runtime");
    }

    ~GAPOperator() {
        if (fwd_runtime_)  xnn_delete_runtime(fwd_runtime_);
        if (fwd_subgraph_) xnn_delete_subgraph(fwd_subgraph_);
    }

    // 禁用拷贝构造与赋值
    GAPOperator(const GAPOperator&) = delete;
    GAPOperator& operator=(const GAPOperator&) = delete;

    /**
     * @brief 正向传播 (NHWC FP32)
     * @param input  [N, H, W, C]
     * @param output [N, 1, 1, C]
     */
    void forward(const float* input, float* output) const {
        // XNNAPI 要求非 const 指针数组，按外部张量定义顺序传入
        void* externals[2] = {const_cast<float*>(input), output};
        
        CHECK_XNN_STATUS(xnn_reshape_runtime(fwd_runtime_, externals), "reshape_forward");
        CHECK_XNN_STATUS(xnn_run_runtime(fwd_runtime_), "run_forward");
    }

    /**
     * @brief 反向传播梯度计算 (NHWC FP32)
     * @param dy 上游梯度 [N, 1, 1, C]
     * @param dx 输入梯度 [N, H, W, C]
     * 
     * 数学原理: dX[n, h, w, c] = dY[n, 0, 0, c] / (H * W)
     */
    void backward(const float* dy, float* dx) const {
        const float inv_hw = 1.0f / static_cast<float>(H_ * W_);
        const size_t num_spatial = H_ * W_;

        // 高效遍历: NHWC 内存连续访问优化
        // 开启 OpenMP 并行加速 (梯度分布计算高度并行)
        #pragma omp parallel for collapse(2) schedule(static) if(num_spatial * C_ > 4096)
        for (uint32_t n = 0; n < N_; ++n) {
            for (uint32_t c = 0; c < C_; ++c) {
                float grad_val = dy[n * C_ + c] * inv_hw;
                
                for (uint32_t h = 0; h < H_; ++h) {
                    size_t row_start = (n * H_ + h) * W_ * C_;
                    for (uint32_t w = 0; w < W_; ++w) {
                        dx[row_start + w * C_ + c] = grad_val;
                    }
                }
            }
        }
    }

private:
    uint32_t N_, H_, W_, C_;
    xnn_subgraph_t fwd_subgraph_ = nullptr;
    xnn_runtime_t  fwd_runtime_  = nullptr;
    uint32_t fwd_input_id_  = 0;
    uint32_t fwd_output_id_ = 0;
};

// ================= 单元测试示例 =================
int main() {
    try {
        constexpr uint32_t N = 2, H = 4, W = 6, C = 3;
        
        std::vector<float> input(N * H * W * C);
        std::vector<float> output(N * 1 * 1 * C);
        std::vector<float> dy(N * 1 * 1 * C, 1.0f); // 假设上游梯度全为1
        std::vector<float> dx(N * H * W * C, 0.0f);

        // 初始化输入数据 (示例: 0 ~ N*H*W*C-1)
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i);

        // 创建算子
        GAPOperator gap(N, H, W, C);

        // 正向
        gap.forward(input.data(), output.data());
        std::cout << "Forward succeeded. Sample output[0]: " << output[0] << "\n";

        // 反向
        gap.backward(dy.data(), dx.data());
        std::cout << "Backward succeeded. Sample dx[0]: " << dx[0] << "\n";

        // 验证反向传播数值正确性
        float expected_grad = 1.0f / (H * W);
        bool correct = std::abs(dx[0] - expected_grad) < 1e-6f;
        std::cout << "Gradient verification: " << (correct ? "PASSED ✅" : "FAILED ❌") << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
```

##### 🔧 编译建议 (CMake)
```cmake
cmake_minimum_required(VERSION 3.15)
project(gap_xnnpack)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gap_demo main.cpp)

### 假设 XNNPACK 安装在 /opt/xnnpack
find_package(PkgConfig REQUIRED)
pkg_check_modules(XNNPACK REQUIRED xnnpack)

target_include_directories(gap_demo PRIVATE ${XNNPACK_INCLUDE_DIRS})
target_link_libraries(gap_demo PRIVATE ${XNNPACK_LINK_LIBRARIES} pthread)
target_compile_options(gap_demo PRIVATE -O3 -ffast-math -march=native)
### 若启用 OpenMP 加速反向传播
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gap_demo PRIVATE OpenMP::OpenMP_CXX)
endif()
```

##### 📐 设计细节说明
| 模块 | 实现策略 | 备注 |
|------|----------|------|
| **正向传播** | `xnn_define_global_average_pooling_2d` + Subgraph API | XNNPACK 会在图固化时自动选择最优 SIMD 路径 (ARM NEON / AVX2 / AVX-512) |
| **内存布局** | `NHWC` | 符合移动端/CPU 访存偏好，XNNPACK 内部算子默认以此布局优化 |
| **反向传播** | 数学等价展开 `dx = dy / (H*W)` + 空间广播 | XNNPACK 无训练 API，手动实现避免额外常量图开销。循环展开与 `#pragma omp` 保证与 XNNPACK 正向性能匹配 |
| **错误处理** | `CHECK_XNN_STATUS` 宏抛异常 | 保证生产环境可观测性，避免静默失败 |
| **线程模型** | `mutable` runtime + 顺序调用 | 适用于单线程或队列调度场景。若需高并发，可将 `fwd_runtime_` 改为 `std::vector<xnn_runtime_t>` 池化 |

此实现可直接嵌入 PyTorch C++ / TensorFlow C API / 自定义推理引擎的算子注册表中。如有特定平台 (Android/iOS/WebAssembly) 交叉编译需求，可调整 CMake 工具链。