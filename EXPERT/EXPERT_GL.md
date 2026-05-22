使用 XNNPACK 的 Subgraph API 实现 Global Average Pooling (GAP) 算子是一个非常优雅的选择。XNNPACK 的 Subgraph API 支持算子融合和自动内存分配，并且对于 Element-wise 算子原生支持 NumPy 风格的广播机制，这为我们实现反向传播提供了极大的便利。

### 数学原理

*   **正向传播**：
    对于输入 $X$ 形状为 $(N, H, W, C)$，输出 $Y$ 形状为 $(N, 1, 1, C)$。
    $$Y_{n,c} = \frac{1}{H \times W} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} X_{n,h,w,c}$$

*   **反向传播**：
    给定上游梯度 $\frac{\partial L}{\partial Y}$ (形状为 $(N, 1, 1, C)$)，求输入梯度 $\frac{\partial L}{\partial X}$ (形状为 $(N, H, W, C)$)。
    $$\frac{\partial L}{\partial X_{n,h,w,c}} = \frac{1}{H \times W} \frac{\partial L}{\partial Y_{n,c}}$$
    即：将 $\frac{\partial L}{\partial Y}$ 乘以标量 $\frac{1}{H \times W}$，然后将其在空间维度 $H$ 和 $W$ 上进行广播。

### 实现思路

1.  **正向传播**：使用 `xnn_define_average_pooling_2d`，将 `pooling_height` 和 `pooling_width` 设置为输入的 $H$ 和 $W$，步长也为 $H$ 和 $W$，即可实现 GAP。
2.  **反向传播**：
    *   定义一个形状为 `(1, 1, 1, 1)` 的权重 Tensor，值为 $\frac{1}{H \times W}$。
    *   定义一个形状为 `(1, H, W, 1)` 的全1权重 Tensor，用于空间广播。
    *   使用 `xnn_define_multiply` 将 `grad_output` 乘以标量权重（缩放）。
    *   使用 `xnn_define_multiply` 将缩放后的梯度乘以空间全1权重，利用 XNNPACK 自动广播机制，将 $(N, 1, 1, C)$ 扩展为 $(N, H, W, C)$。

### C++17 完整实现

以下是完整的 C++ 代码，包含正向和反向传播的构建、执行以及数值正确性验证。

```cpp
#include <xnnpack.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstddef>

class GlobalAvgPool2D {
public:
    GlobalAvgPool2D(size_t N, size_t H, size_t W, size_t C) 
        : N_(N), H_(H), W_(W), C_(C) {
        xnn_status status = xnn_initialize(nullptr);
        assert(status == xnn_status_success);
    }

    ~GlobalAvgPool2D() {
        if (runtime_) xnn_delete_runtime(runtime_);
        if (subgraph_) xnn_delete_subgraph(subgraph_);
        xnn_deinitialize();
    }

    // 构建 Subgraph (包含正向和反向计算图)
    void Build() {
        xnn_status status;
        
        // 创建子图，4 个外部 Value (Input, Output, GradOutput, GradInput)
        status = xnn_create_subgraph(4, 0, &subgraph_);
        assert(status == xnn_status_success);

        // --- 定义 Tensor Values ---
        
        // Value 0: 正向输入 (N, H, W, C)
        size_t input_dims[4] = {N_, H_, W_, C_};
        status = xnn_define_tensor_value(subgraph_, xnn_datatype_fp32, 4, input_dims, nullptr, 0, 0);
        assert(status == xnn_status_success);

        // Value 1: 正向输出 (N, 1, 1, C)
        size_t output_dims[4] = {N_, 1, 1, C_};
        status = xnn_define_tensor_value(subgraph_, xnn_datatype_fp32, 4, output_dims, nullptr, 1, 0);
        assert(status == xnn_status_success);

        // Value 2: 反向传播的上游梯度 (N, 1, 1, C)
        status = xnn_define_tensor_value(subgraph_, xnn_datatype_fp32, 4, output_dims, nullptr, 2, 0);
        assert(status == xnn_status_success);

        // Value 3: 反向传播的输入梯度 (N, H, W, C)
        status = xnn_define_tensor_value(subgraph_, xnn_datatype_fp32, 4, input_dims, nullptr, 3, 0);
        assert(status == xnn_status_success);

        // Value 4: 缩放常数权重 (1, 1, 1, 1) = 1 / (H * W)
        scale_val_ = 1.0f / (H_ * W_);
        size_t scale_dims[4] = {1, 1, 1, 1};
        status = xnn_define_tensor_value(subgraph_, xnn_datatype_fp32, 4, scale_dims, &scale_val_, 4, 0);
        assert(status == xnn_status_success);

        // Value 5: 空间广播权重 (1, H, W, 1) = 1.0
        ones_val_.resize(H_ * W_, 1.0f);
        size_t ones_dims[4] = {1, H_, W_, 1};
        status = xnn_define_tensor_value(subgraph_, xnn_datatype_fp32, 4, ones_dims, ones_val_.data(), 5, 0);
        assert(status == xnn_status_success);

        // Value 6: 中间变量 - 缩放后的梯度 (N, 1, 1, C)
        status = xnn_define_tensor_value(subgraph_, xnn_datatype_fp32, 4, output_dims, nullptr, 6, 0);
        assert(status == xnn_status_success);

        // --- 定义算子 ---

        // 1. 正向传播：Average Pooling 2D (设置 Kernel 和 Stride 均为 H, W 即为 GAP)
        status = xnn_define_average_pooling_2d(
            subgraph_,
            0, 0, 0, 0,     // input padding
            H_, W_,          // pooling height, width
            H_, W_,          // stride height, width
            0, 0, 0, 0,     // output padding
            0, 1,           // input_id, output_id
            0               // flags
        );
        assert(status == xnn_status_success);

        // 2. 反向传播步骤1：将 grad_output 乘以 1/(H*W)
        status = xnn_define_multiply(subgraph_, 2, 4, 6, 0);
        assert(status == xnn_status_success);

        // 3. 反向传播步骤2：将缩放后的梯度广播至空间维度
        // XNNPACK 的 multiply 支持广播: (N, 1, 1, C) * (1, H, W, 1) -> (N, H, W, C)
        status = xnn_define_multiply(subgraph_, 6, 5, 3, 0);
        assert(status == xnn_status_success);

        // --- 编译子图 ---
        status = xnn_subgraph_compile(subgraph_, 0);
        assert(status == xnn_status_success);
    }

    // 分配 Runtime 并绑定外部内存
    void Setup(float* input, float* output, float* grad_output, float* grad_input) {
        xnn_status status;
        struct xnn_external_value externs[4] = {
            {0, input},
            {1, output},
            {2, grad_output},
            {3, grad_input}
        };

        status = xnn_create_runtime(subgraph_, &runtime_);
        assert(status == xnn_status_success);

        status = xnn_setup_runtime(runtime_, 4, externs);
        assert(status == xnn_status_success);
    }

    // 执行计算图 (正向+反向同时执行)
    void Run() {
        xnn_status status = xnn_invoke_runtime(runtime_, 0);
        assert(status == xnn_status_success);
    }

private:
    size_t N_, H_, W_, C_;
    xnn_subgraph_t subgraph_ = nullptr;
    xnn_runtime_t runtime_ = nullptr;

    // 用于权重的生命周期管理 (必须保证在 subgraph compile 期间有效)
    float scale_val_;
    std::vector<float> ones_val_;
};

int main() {
    // 测试参数：Batch=1, Height=2, Width=2, Channels=1
    const size_t N = 1, H = 2, W = 2, C = 1;
    const size_t input_size = N * H * W * C;
    const size_t output_size = N * 1 * 1 * C;

    // 准备数据
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(output_size, 0.0f);
    std::vector<float> grad_output = {1.0f}; // 上游梯度
    std::vector<float> grad_input(input_size, 0.0f);

    // 实例化并运行 GAP 算子
    GlobalAvgPool2D gap_op(N, H, W, C);
    gap_op.Build();
    gap_op.Setup(input.data(), output.data(), grad_output.data(), grad_input.data());
    gap_op.Run();

    // --- 验证正向结果 ---
    // GAP(1,2,3,4) = (1+2+3+4)/4 = 2.5
    std::cout << "Forward Output (Expected 2.5): " << output[0] << std::endl;
    assert(std::fabs(output[0] - 2.5f) < 1e-5f);

    // --- 验证反向结果 ---
    // dL/dX = dL/dY * 1/(H*W) = 1.0 * 1/4 = 0.25
    std::cout << "Backward Grad Input (Expected 0.25 for all):" << std::endl;
    for (size_t i = 0; i < input_size; ++i) {
        std::cout << grad_input[i] << " ";
        assert(std::fabs(grad_input[i] - 0.25f) < 1e-5f);
    }
    std::cout << std::endl;

    std::cout << "XNNPACK GAP Forward & Backward passed!" << std::endl;
    return 0;
}
```

### 代码解析与细节

1. **Subgraph 拓扑排序执行**：
   XNNPACK 的 `xnn_invoke_runtime` 会根据定义的算子依赖关系自动进行拓扑排序。我们定义了 `0->1` 的正向，以及 `2,4->6` 和 `6,5->3` 的反向，运行时 XNNPACK 会自动识别并按顺序执行正向传播再执行反向传播。

2. **广播机制的巧妙运用**：
   反向传播核心在于将 $(N, 1, 1, C)$ 还原为 $(N, H, W, C)$。我们构造了一个形状为 `(1, H, W, 1)` 且值全为 `1.0` 的权重 Tensor。在 NHWC 格式下：
   - $A$ 的 shape 为 `(N, 1, 1, C)`
   - $B$ 的 shape 为 `(1, H, W, 1)`
   - $A \times B$ 的 shape 自动推导为 `(N, H, W, C)`
   这种方式只占用极小的额外内存（仅 $H \times W$ 大小的参数），就能完美实现空间梯度分配。

3. **生命周期注意事项**：
   在 `xnn_define_tensor_value` 传入静态权重（如 `scale_val_` 和 `ones_val_`）时，XNNPACK 默认不拷贝数据，只持有着指针引用。必须保证这些指针在 `xnn_subgraph_compile` 调用前一直有效。因此我在类中将其声明为成员变量 `scale_val_` 和 `ones_val_` 以延长生命周期。

4. **编译方法**：
   确保你已经编译并安装了 XNNPACK，编译命令示例：
   ```bash
   g++ -std=c++17 gap_op.cpp -lxnnpack -lpthreadpool -o gap_op
   ./gap_op
   ```