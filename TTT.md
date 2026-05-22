cuDNN Frontend 里，`tanh` 属于 Pointwise 算子。正向用 `TANH_FWD`，反向用 `TANH_BWD`，直接往 `cudnn_frontend::graph::Graph` 里链式添加即可。

---

正向 Tanh

```cpp
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;

// 1. 创建 Graph
auto graph = std::make_shared<<fe::graph::Graph>();
graph->set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);

// 2. 定义输入张量
auto x = graph->tensor(
    fe::graph::Tensor_attributes()
        .set_name("x")
        .set_dim({b, c, h, w})
        .set_stride({c*h*w, h*w, w, 1})
        .set_uid(1));

// 3. 添加 Tanh 节点
auto tanh_opts = fe::graph::Pointwise_attributes()
    .set_mode(fe::PointwiseMode_t::TANH_FWD);

auto y = graph->pointwise(x, tanh_opts);
y->set_output(true)
  .set_name("y")
  .set_uid(2);

// 4. 正常 build
graph->validate();
graph->build_operation_graph(handle);
graph->create_execution_plans({fe::HeurMode_t::A});
graph->check_support();
graph->build_plans();
```

输入：`x`

输出：`y = tanh(x)`

---

反向 Tanh

反向需要两个输入：上游梯度 `dy`，以及正向的输出 `y`（cuDNN 用 `y` 来计算 `1 - y²`，比重新算 `tanh(x)` 更快）。

```cpp
// 在反向 Graph 里
auto dy = graph->tensor(
    fe::graph::Tensor_attributes()
        .set_name("dy")
        .set_dim({b, c, h, w})
        .set_stride({c*h*w, h*w, w, 1})
        .set_uid(10));

auto y = graph->tensor(
    fe::graph::Tensor_attributes()
        .set_name("y")
        .set_dim({b, c, h, w})
        .set_stride({c*h*w, h*w, w, 1})
        .set_uid(11));   // 正向输出的 y

auto tanh_bwd_opts = fe::graph::Pointwise_attributes()
    .set_mode(fe::PointwiseMode_t::TANH_BWD);

auto dx = graph->pointwise(dy, y, tanh_bwd_opts);
dx->set_output(true)
   .set_name("dx")
   .set_uid(12);
```

数学：`dx = dy * (1 - y²)`

---

完整执行示例

```cpp
// 正向
std::unordered_map<int64_t, void*> fwd_pack = {
    {1, x_dev_ptr},
    {2, y_dev_ptr}
};
graph_fwd->execute(handle, fwd_pack, workspace);

// 反向（假设 y 已保存）
std::unordered_map<int64_t, void*> bwd_pack = {
    {10, dy_dev_ptr},
    {11, y_dev_ptr},   // 正向输出
    {12, dx_dev_ptr}
};
graph_bwd->execute(handle, bwd_pack, workspace);
```

---

关键注意点

事项	说明	
反向输入	`TANH_BWD` 需要 dy 和 y（正向输出），不是 x。cuDNN 内部用 `1 - y²` 求导。	
数据类型	`set_compute_data_type(FLOAT)` 建议保持，即使 IO 是 FP16，中间用 FP32 防精度损失。	
In-place	Pointwise 支持 in-place（输出张量 UID 和输入张量 UID 指向同一地址），但反向时 `y` 和 `dx` 不能 in-place，因为 `y` 还要被读取。	
CUDA Graph	这些 pointwise 节点可以无缝嵌入你们之前的 `populate_cuda_graph` 或 stream capture 流程里。	

---

如果要做 In-Place 正向

```cpp
// y 和 x 共享内存，省显存
auto y = graph->pointwise(x, tanh_opts);
y->set_output(true)
  .set_uid(1);  // 和 x 同一个 UID，执行时传入同一指针即可
```

---

一句话：cuDNN Frontend 的 tanh 就是 `Pointwise_attributes().set_mode(TANH_FWD/TANH_BWD)`，反向传 `dy` 和 `y`，输出 `dx`。和 PyTorch 的 `torch.tanh` / `torch.nn.functional.tanh` 逻辑一致。