# 算子实现方案

## 一、体系结构：双枚举、两张表、两级分发

OOOPS_FINAL.md 定义了算子的两端世界：

```
ComputeOp（41个，逐DTensor）          RangeOp（17个，逐Region）
─────────────────────────────────    ─────────────────────────────
有 OpParams 参数                     无参数，预计算 (offset,size)
有 input_ids / output_ids            有 input_ranges / output_ranges
先 infer_shapes 再 launch            直接 launch
CPU + CUDA + cuDNN 三种路径          只有 CPU + CUDA 两种路径
```

两者通过 `GraphNode` 的 union 统一表达（[OOOPS_FINAL.md §7.1](file:///r:\renaissance\OOOPS_FINAL.md)），运行时走两张全局函数指针表：

```cpp
ComputeOpEntry g_compute_op_table[41];   // dtensor/ 各文件填充
RangeOpEntry   g_range_op_table[17];     // range/ 各文件填充
```

分发代码（[OOOPS_FINAL.md §7.3](file:///r:\renaissance\OOOPS_FINAL.md)）：

```cpp
void execute_node(const GraphNode& node, MemoryPlan& plan) {
    switch (node.kind) {
    case COMPUTE: {
        auto& e = g_compute_op_table[(size_t)node.compute_op];
        e.infer_shapes(outputs, inputs);         // 1. 推算输出形状
        e.launch_cpu(...) 或 e.launch_cuda(...); // 2. 运行时只做这一步
        break;
    }
    case RANGE: {
        auto& e = g_range_op_table[(size_t)node.range_op];
        e.launch(input_ranges, output_ranges, config);
        break;
    }
    }
}
```

**性能要点**：两张表是纯静态数组，dispatch 走 `entry[idx]` 而非 `map::find`，热路径上零hash、零分支预测失败。

## 二、CPU 算子的实现套路

### 2.1 Kernel 形态

朴素 C++ 循环，无并行，利用 `__restrict` 提示编译器做矢量化：

```cpp
void launch_tr_fc_fwd_fp32_kernel_cpu(
    const float* __restrict x, const float* __restrict w,
    const float* __restrict b, float* __restrict y,
    int batch, int in_features, int out_features,
    size_t x_row_stride, size_t w_row_stride, size_t y_row_stride)
{
    for (int b = 0; b < batch; ++b) {
        for (int o = 0; o < out_features; ++o) {
            const float* xr = x + b * x_row_stride;   // stride 非紧凑
            const float* wr = w + o * w_row_stride;
            float sum = 0.0f;
            for (int i = 0; i < in_features; ++i)
                sum += xr[i] * wr[i];
            if (b) sum += b[o];
            y[b * y_row_stride + o] = sum;
        }
    }
}
```

**为什么 stride 入参而不假设紧凑？** NHWC 布局下，FP32 要求 256 字节行对齐（64 float），FP16 要求 128 字节（64 half）。`in_features=5` 时，实际行宽 = 64 而非 5。kernel 如果按 `x[b*K+i]` 寻址，结果是错误的。stride 入参是正确性保证。

**为什么无并行？** 训练循环的多卡并行（Level 1）由 TaskBase 的多线程管理，每个 rank 一个线程。算子内部的进一步并行化收益很小——CPU 本身没有 GPU 级别的并行度，且多线程带来的 cache thrashing 可能适得其反。对于生产用例，CPU 模式仅用于验证和调试，性能瓶颈不在此。

### 2.2 Launch 外层

将 kernel 从裸函数指针包装为 `ComputeOpEntry::launch_cpu` 接口：

```cpp
static void launch_cpu_fc_fwd(const SmallVector<int32_t>& in_ids,
                              const SmallVector<int32_t>& out_ids,
                              const OpParams& params) {
    const auto* p = std::get_if<FCParams>(&params.data);
    const float* x = ptr_at(in_ids[0]);
    const float* w = ptr_at(in_ids[1]);
    const float* b = p->bias ? ptr_at(in_ids[2]) : nullptr;
    float* y = ptr_at(out_ids[0]);

    const DTensor& dx = memory_plan->dtensor(in_ids[0]);
    launch_tr_fc_fwd_fp32_kernel_cpu(
        x, w, b, y,
        dx.shape.n(), dx.shape.h() * dx.shape.w() * dx.shape.c(), p->out_features,
        dx.n_stride, memory_plan->dtensor(in_ids[1]).n_stride,
        memory_plan->dtensor(out_ids[0]).n_stride);
}
```

**关键**：`ptr_at(id)` → `Arena基址 + DTensor::offset`，绕过所有中间层次直接拿到内存地址。零虚函数、零间接。

### 2.3 文件组织

```
src/backend/ops/dtensor/fc_op.cpp
├── infer_shapes_fc_fwd/bwd    ← 形状推导（平台无关）
├── launch_cpu_fc_fwd/bwd      ← CPU 路径（统一接口）
├── launch_cuda_fc_fwd/bwd     ← CUDA 路径（统一接口，#ifdef 内）
├── kFcFwdEntry / kFcBwdEntry  ← constexpr 映射表条目（同文件定义）
```

`fc_op.cu` 只含 `__global__` kernel 和 `cudaError_t launch_*(..., cudaStream_t)` launch wrapper，不含任何注册逻辑。

## 三、CUDA 算子的实现套路

### 3.1 手写 kernel 路径（旁路）

与 CPU kernel 共享**完全相同**的参数声明（包括 `__restrict__` 的位置和顺序），内部用 **grid-stride loop** 替代双重 for：

```cpp
__global__ void tr_fc_fwd_fp32_kernel(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, float* __restrict__ y,
    int batch, int in_features, int out_features,
    size_t x_row_stride, size_t w_row_stride, size_t y_row_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_features;
    int step  = blockDim.x * gridDim.x;

    for (; idx < total; idx += step) {
        int b = idx / out_features;
        int o = idx % out_features;
        const float* xr = x + b * x_row_stride;
        const float* wr = w + o * w_row_stride;
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) sum += xr[i] * wr[i];
        if (b) sum += b[o];
        y[b * y_row_stride + o] = sum;
    }
}
```

Grid-stride loop 的模式：`idx += blockDim.x * gridDim.x` 而非 `idx += 1`，让一个线程覆盖多个输出元素。这样当 `total` 超过 `gridSize * blockSize` 时不需要多轮 launch。

对应 Launch Wrapper（[fc_op.cu:L98-L115](file:///r:\renaissance\src_legacy\backend\ops\fc_op.cu#L98)）负责三件事：计算 `(grid, block)` 配置、传 `cudaStream_t`、返 `cudaGetLastError()`。

**为什么 CPU 和 CUDA kernel 共享参数声明？** 编译期直接比对，不会出现"CPU 跑通、CUDA 炸掉"的 shape/stride 不匹配 bug。

### 3.2 cuDNN 前端图路径（主力，仅 GPU）

对于卷积、BN、FC 等重算子，主力 CUDA 实现不是手写 kernel，而是 cuDNN FE Graph：

```cpp
auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
graph->set_io_data_type(io_dtype)           // HALF or FLOAT
      .set_intermediate_data_type(FLOAT)
      .set_compute_data_type(FLOAT);

// 1. 虚拟张量 — dim/stride/uid 映射到实际内存
auto X = graph->tensor(Tensor_attributes()
    .set_dim({batch, 1, in_features})
    .set_stride({x_row_stride, in_features, 1})
    .set_uid(t_X->get_uid()));

// 2. 算子节点 — matmul / conv / pointwise
auto Y = graph->matmul(X, W, mm_attr);
Y->set_output(true);

// 3. 三步构建（耗时，仅在 Phase A compile 阶段执行一次）
graph->validate();
graph->build_operation_graph(handle);
graph->create_execution_plans({HeurMode_t::B, HeurMode_t::FALLBACK});
graph->check_support(handle);
graph->build_plans(BuildPlanPolicy_t::HEURISTICS_CHOICE);
```

这段代码只在 **compile warmup** 阶段执行一次（Phase A），产出的 `cudnn_frontend::Graph` 对象在 CUDA Graph 捕获（Phase B）时被录制为 kernel 序列。运行期不需要再调 `build_graph`。

**cuDNN 的 HEURISTICS_CHOICE + FALLBACK** 双策略是关键：B 模式选最优 heuristic engine，FALLBACK 保底不可用场景。A 模式可能在特定 GPU 上选了不支持的 engine，导致 `check_support` 失败。

### 3.3 cuDNN Graph ≠ CUDA Graph

一个必须清晰的认知链条：

```
build_graph        →  cuDNN FE Graph  （host端 engine selection，Phase A）
    ↓
execute(graph)     →  一组 kernel launch （GPU 端，在 CUDA Graph 捕获区间内）
    ↓
cudaGraphInstantiate →  cudaGraphExec_t（硬件图的固化句柄，Phase B）
    ↓
cudaGraphLaunch    →  一次 API 调用重放全部 kernel（Phase C 热路径）
```

因此 `build_graph` 从不出现在 `ComputeOpEntry` 中——它在 compile warmup 阶段单独完成，不在热路径上。

## 四、RangeOp 的实现套路

RangeOp 没有 `OpParams`，没有 DTensor ID，只有预计算的 `(offset, size)` 内存范围：

```cpp
void launch_range_zero_grad(const MemRange* inputs, const MemRange* outputs,
                            const PlanConfig& config) {
    // outputs = 从 G_BN_BIAS 到 G_DEEP_CONV 整段连续内存
    std::memset(base_ptr + outputs[0].offset, 0, outputs[0].size);
    // 或 CUDA: cudaMemsetAsync(base_ptr + offset, 0, size, stream)
}
```

为什么不需要 `OpParams`？操作语义完全由 `(offset, size)` 确定：memset、memcpy、elementwise cast、AllReduce。涉及多个 DTensor（如 `RANGE_UPDATE_WEIGHT` 同时更新 W/G/M 三个 Region）靠 Compile 期 MemoryPlan 的 `build_range_op_ranges()` 预计算好排列，kernel 按统一 index 遍历即可。

## 五、上层如何调用

### 5.1 ComputationGraph 构建期

Compile 阶段，`Compiler` 遍历 `ArchPlan`，按 `OpKind` → `GraphNode` 生成计算图节点（略）。表驱动 dispatch 确保单节点直接对应单条目。

### 5.2 图捕获期（CPU）

`capture_graph_cpu` 遍历 `ComputationGraph::nodes()`，为每个 node 生成 lambda 闭包（Legacy 做法，见 CpuGraph.md）。新版 `CapturedGraph` 应将其统一为 `vector<CpuOp>` 中的函数指针条目。

### 5.3 图捕获期（CUDA）

`capture_graph` 在 CUDA Graph 捕获区间内，通过 `g_compute_op_table[op].launch_cuda()` 或 `g_range_op_table[op].launch()` 调用对应 CUDA kernel。cuDNN 算子则调用 `build_graph` 产出的 `cudnn_frontend::Graph::execute()`。所有调用被 CUDA runtime 录制为 `cudaGraphExec_t`。

### 5.4 运行期

捕获后，`cudaGraphLaunch(exec, stream)` 一次调用替代全部算子级 dispatch。CPU 路径走 `for (op : cpu_ops) op.fn(op.ctx)` 顺序调用。两者在 `run_graph()` 的 `use_gpu` 分支内完成。

## 六、性能设计要点

| 决策 | 为什么 |
|------|--------|
| 全局数组 `g_compute_op_table[]` 而非 `std::map` | O(1) 索引，缓存友好，零 hash |
| CPU/CUDA 共享参数声明 | 编译期验证一致性，无需运行时检查 |
| `ptr_at(id)` → Arena 直接地址 | 零中间层，零虚函数，L1 cache 有效利用 |
| `__restrict` 修饰全部指针 | 编译器可做循环矢量化（CPU）和寄存器分配优化（CUDA） |
| stride 入参而非紧凑假设 | NHWC 对齐带来 ~30% 带宽提升，stride 代价是函数签名多几个参数——值得 |
| grid-stride loop（CUDA） | 避免了 `total > gridSize*blockSize` 时的多轮 launch |
| `HEURISTICS_CHOICE + FALLBACK` | 双保险 cuDNN engine 选择，不依赖特定 GPU 型号 |
| `.cu` 文件与 `.cpp` 分离 | CUDA 编译链（nvcc）和 C++ 编译链（MSVC/GCC）独立，不污染构建依赖 |
| RangeOp 预计算 (offset,size) | 把指针计算从 runtime 移到 compile，kernel 粒度从"逐 DTensor for 循环"变为"一次 kernel 覆盖整段" |

**最核心的一条**：把一切能提前做的计算放到 compile 阶段。infer_shapes（形状推导）、`build_range_op_ranges()`（offset/size 预计算）、cuDNN `build_plans`（engine 选择），全部是 compile 期完成。运行期热路径只剩：`ptr_at` 拿地址 → `launch_cpu`/`launch_cuda` 调用 kernel。没有 if/else 分支、没有 virtual dispatch、没有运行时类型推导。