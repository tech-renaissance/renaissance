# CPU 上的 Graph

## 一、什么是 CPU "图"

CPU 没有 CUDA Graph 的硬件 DAG 录制能力，旧版框架用一个简单的替代方案：**遍历 ComputationGraph 节点，把每个操作封装成 lambda，存入 `vector<function<void()>>`，运行时顺序调用**。

```cpp
// Backend 成员（task_base.cpp:L300）
std::unordered_map<std::string, std::vector<std::function<void()>>> cpu_tasks_;
```

Key 是图名称（如 `"train_forward"`），Value 是该图包含的所有操作。这就是 CPU "图"的全部数据结构——没有异步、没有流、没有依赖拓扑，有序的函数队列就是图。

## 二、捕获：遍历节点 → 封装闭包

[capture_graph_cpu()](file:///r:\renaissance\src_legacy\task\task_base.cpp#L1291-L1350) 在 `compile_capture_all_graphs` 阶段被调用，对每张图执行一次：

```cpp
void capture_graph_cpu(const string& name, const ComputationGraph& graph,
                       StreamKind, int rank) {
    auto& tasks = cpu_tasks_[name];
    tasks.clear();
    tasks.reserve(graph.nodes().size());   // 关键：防止重分配 → 引用不失效

    for (const auto& node : graph.nodes()) {
        std::visit([&](auto&& arg) {
            using T = decay_t<decltype(arg)>;

            if constexpr (is_same_v<T, ComputeNode>) {
                tasks.emplace_back([node = arg, this, rank]() {
                    execute_compute_node_cpu(node, rank);
                });
            } else if constexpr (is_same_v<T, TransferNode>) {
                tasks.emplace_back([&node = arg, this, rank]() {
                    void* dst = resolve_ptr(node.dst_id, rank);
                    const void* src = /* host_ptr 或 host_tensor.data() */;
                    size_t n = /* host_num_bytes */;
                    std::memcpy(dst, src, n);
                });
            }
        }, node);
    }
}
```

两个关键设计决策：

| 节点 | 捕获方式 | 原因 |
|------|---------|------|
| ComputeNode | `[node = arg]` 值捕获 | 可复制结构体，安全持有完整副本。执行时 `execute_compute_node_cpu` 再走 switch 分发 |
| TransferNode | `[&node = arg]` 引用捕获 | 含 `vector<Tensor>` 不可复制成员。依赖 `tasks.reserve()` 保证 vector 扩容不重分配、引用不失效 |

TransferNode 在 CPU 模式下退化为 `std::memcpy`：没有 GPU 的 H2D DMA，从主机缓冲区到内存池就是一次拷贝。

### 算子分发

`execute_compute_node_cpu`（[task_base.cpp:L800](file:///r:\renaissance\src_legacy\task\task_base.cpp#L800)）是一个大 switch，在**执行时**而非捕获时把 `OpKind` → CPU kernel：

```cpp
void execute_compute_node_cpu(const ComputeNode& node, int rank) const {
    switch (node.kind) {
        case AXPY_FWD: {
            const float* a = static_cast<const float*>(resolve_ptr(node.in_ids[0], rank));
            const float* b = static_cast<const float*>(resolve_ptr(node.in_ids[1], rank));
            float* c = static_cast<float*>(resolve_ptr(node.out_ids[0], rank));
            // shape 从 DTensor 读取，stride 从 DTensor 读取
            launch_tr_axpy_fp32_kernel_cpu(a, b, alpha, c, N, H, W, C, ...);
            break;
        }
        case RELU_FWD: launch_tr_relu_fwd_fp32_mask_kernel_cpu(x, y, mask, n); break;
        case RELU_BWD: launch_tr_relu_bwd_fp32_kernel_cpu(dy, mask, dx, n); break;
        case FC_AMP_FWD: launch_tr_fc_fwd_fp32_kernel_cpu(...); break;
        // ... CONV_FWD, CONV_DGRAD, CONV_WGRAD, FLATTEN_FWD/BWD, ADD, FILL
    }
}
```

`resolve_ptr(id, rank)` → `device_contexts_[rank]->ptr_at(id)` → **Arena 基地址 + DTensor::offset**，指向 CPU 内存池。

## 三、执行：顺序调用函数队列

[run_graph()](file:///r:\renaissance\src_legacy\task\task_base.cpp#L1355) CPU 路径只有三行：

```cpp
auto it = cpu_tasks_.find(name);
if (it == cpu_tasks_.end())
    TR_VALUE_ERROR("CPU task list '" << name << "' not found");
for (const auto& task : it->second)
    task();   // 顺序执行，阻塞完成，天然同步
```

`sync_all()` 在 CPU 模式下是空操作——每个 `task()` 返回即表示完成，无需等待异步流。

## 四、与 CUDA Graph 的对称

`capture_graph` / `run_graph` 的两条分支保证了上层代码零差异：

```cpp
void capture_graph(const string& name, const ComputationGraph& graph,
                   StreamKind stream, int rank) {
    if (use_gpu) {
        // cudaStreamBeginCapture → 逐节点执行 → cudaStreamEndCapture → instantiate
    } else {
        capture_graph_cpu(name, graph, stream, rank);
    }
}

void run_graph(const string& name, int rank, StreamKind stream) const {
    if (use_gpu) {
        // cudaGraphLaunch
    } else {
        for (const auto& task : cpu_tasks_[name])
            task();
    }
}
```

调用方不需要知道后端类型。编译管线、训练循环的代码完全不变。

## 五、局限性

- **无 Level 2 并行**。双图（`run(a, b)`）在 CPU 模式退化为串行 `run(a); run(b)`，因为没有流的概念
- **编译期指针绑定**。`resolve_ptr` 在捕获时就已经把 DTensor id 解析为具体地址封进闭包——如果地址变化（比如 Arena 重建），捕获结果作废
- **`std::function` 开销**。每个 lambda 约 32 字节 + 可能的堆分配 + 虚函数调用。一个 50 节点的图浪费 1.6KB 间接内存。可用裸函数指针 `void (*)(void*)` 替代
- **TransferNode 引用捕获的隐式契约**。`[&node = arg]` 的正确性依赖 `tasks.reserve()` 和之后不再 `push_back`。如果未来有人打破这个假设，就是悬垂引用

## 六、对新版的启示

新版 `CapturedGraph` 延续了同一哲学——用 `vector<CpuOp>` 存 CPU 路径、用 `void*` 存 CUDA 路径。Legacy 的经验表明：**统一接口比统一实现更重要**。捕获和执行的两条分支已足够干净，不需要引入虚函数、模板策略等更重的抽象机制。

改进方向：
1. `CpuOp` 用 `void (*fn)(void* ctx)` + `void* ctx`，去掉 `std::function` 的类型擦除开销
2. 延迟指针解析到运行时，不在捕获阶段把绝对地址固化进闭包