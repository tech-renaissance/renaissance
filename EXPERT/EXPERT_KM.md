下面是一份完整的、可直接集成到项目中的 XNNPACK Subgraph API 版 Global Average Pooling（GAP）实现。关键设计要点：

- 正向传播：通过 `xnn_define_global_average_pooling_2d` 构建静态子图，利用 XNNPACK 的 runtime 执行高性能推理。
- 反向传播：XNNPACK 本身是纯推理库，不提供训练/反向算子，因此 backward 为手写实现。GAP 的反向逻辑极其简单——将上游梯度除以 `H×W` 后广播回输入的 spatial 维度。
- 数据格式：严格遵循 NHWC、FP32（`xnn_datatype_fp32`）。
- 输出维度：默认开启 `XNN_FLAG_KEEP_DIMS`，正向输出为 `{N, 1, 1, C}`，保持 4D NHWC 一致性；也可关闭得到 `{N, C}`。
- 资源管理：使用 C++17 RAII 封装 `xnn_subgraph_t`、`xnn_runtime_t`、`xnn_workspace_t`，避免内存泄漏。

---

下载代码

[gap_xnnpack.hpp](sandbox:///mnt/agents/output/gap_xnnpack.hpp)

---

核心代码说明

1. 正向传播（Forward）

```cpp
// 构建 Subgraph
xnn_create_subgraph(/*external_value_ids=*/2, 0, &subgraph);

// 定义 NHWC 输入 {N, H, W, C}
xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, in_dims,
                        nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);

// 定义输出 {N, 1, 1, C}（keep_dims）
xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, out_dims,
                        nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);

// 注册 GAP 节点
xnn_define_global_average_pooling_2d(subgraph, -inf, inf, input_id, output_id,
                                       XNN_FLAG_KEEP_DIMS);

// 创建 Runtime 并执行
xnn_create_runtime_v4(subgraph, nullptr, workspace, nullptr, 0, &runtime);
xnn_setup_runtime_v2(runtime, externals.size(), externals.data());
xnn_invoke_runtime(runtime);
```

2. 反向传播（Backward）

数学上，Global Average Pooling 的梯度为：

\frac{\partial \mathcal{L}}{\partial x{n,h,w,c}} = \frac{1}{H \cdot W} \cdot \frac{\partial \mathcal{L}}{\partial y{n,c}}

实现为简单的 broadcast + scale：

```cpp
const float inv_hw = 1.0f / (H * W);
for n:
  for c:
    float grad = grad_output[n*C + c] * inv_hw;
    for h, w:
        grad_input[((n*H + h)*W + w)*C + c] = grad;
```

---

编译与使用示例

```bash
# 假设 XNNPACK 已安装到系统路径
g++ -std=c++17 -O3 -DGAP_XNNPACK_STANDALONE_TEST \
    gap_xnnpack.hpp -o gap_test \
    -lxnnpack -lpthreadpool -lm
./gap_test
```

API 调用示例：

```cpp
#include "gap_xnnpack.hpp"

// 输入 2x4x4x3 (NHWC)
nn::GlobalAveragePooling2D gap(2, 4, 4, 3, /*keep_dims=*/true);

std::vector<float> input(2*4*4*3);
std::vector<float> output(2*1*1*3);
std::vector<float> grad_out(2*1*1*3, 1.0f);
std::vector<float> grad_in(2*4*4*3);

// 正向
gap.Forward(input.data(), output.data());

// 反向
gap.Backward(grad_out.data(), grad_in.data());
```

---

注意事项

1. 线程池：当前代码在 `xnn_create_runtime_v4` 中传入 `nullptr` 作为 threadpool，即单线程执行。若需多核并行，请创建 `pthreadpool_t*` 并传入。
2. XNNPACK 初始化：代码通过 `std::call_once` 保证进程级只执行一次 `xnn_initialize(nullptr)`。
3. Deprecated API：`xnn_define_global_average_pooling_2d` 在最新头文件中被标记为 `XNN_DEPRECATED`，但目前仍稳定可用。若后续版本移除，可迁移至新的 `xnn_define_static_reduce(..., xnn_reduce_mean, ...)` 配合 reduction axes `{1, 2}` 实现相同功能。citeweb_search:6#13
4. 输出 clamp：`output_min` / `output_max` 设置为 `±inf`，表示不启用输出截断（clamp）。如需 fused ReLU，可改为 `0.0f` / `+inf`。citeweb_search:6#4