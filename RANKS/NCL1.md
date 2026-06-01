# NCCL 通信精度下降问题分析报告

## 1. 现象描述

| 配置 | 准确率表现 | 评估 |
|------|-----------|------|
| 单 RANK | 不变 | 正常 |
| 2 RANK | 准确率最高记录 | 通信正常 |
| 4 RANK | 中等 | 可能部分通信异常 |
| 8 RANK | 99.5% → 99.0% | 通信疑似未生效 |

关键线索：**8RANK 精度下降量 ≈ 每个 RANK 只训练了 1/8 样本的效果**（100 epoch ≈ 12.5 epoch 的数据量），强烈暗示 NCCL 通信在 8RANK 场景下没有实际执行。

---

## 2. 根因分析（按严重程度排序）

### 2.1 [CRITICAL] CUDA Graph 捕获期间的事件生命周期违规

**位置**：`captured_graph.cpp` 的 `capture_nccl_graph_coordinated()` 函数

**问题**：在 CUDA Graph 捕获期间（`cudaStreamBeginCapture` 与 `cudaStreamEndCapture` 之间），创建和销毁了 CUDA Event。

**代码证据**：

```cpp
// Phase 1: BeginCapture on ALL ranks (line 490)
cudaStreamBeginCapture(cap_streams[r], cudaStreamCaptureModeThreadLocal);

// Phase 2: ncclGroupStart → replay all ranks → ncclGroupEnd
ncclGroupStart();
for (int r = 0; r < num_ranks; ++r) {
    // ... 
    // 行 510-514：在捕获期间创建事件！
    for (int i = 0; i < state.num_active; ++i) {
        if (state.streams[i].last_done_event) {
            cudaEventDestroy(state.streams[i].last_done_event);  // 捕获期间销毁！
        }
        cudaEventCreateWithFlags(&state.streams[i].last_done_event, ...);  // 捕获期间创建！
    }
    // ... replay nodes ...
    // 行 559：在捕获期间销毁事件！
    state.cleanup_all_events();
}
ncclGroupEnd();

// Phase 3a: EndCapture on ALL ranks (line 566)
cudaStreamEndCapture(cap_streams[r], &captured_graphs[r]);
```

**对比正常路径**（`capture_cuda.cpp` 第 75-171 行）：

```cpp
// 正常路径：事件创建在 BeginCapture 之前
for (int i = 0; i < state.num_active; ++i) {
    cudaEventCreateWithFlags(...);  // ← 捕获前创建
}
cudaStreamBeginCapture(primary_stream, ...);  // ← 然后才开始捕获

// ... replay ...

cudaStreamEndCapture(primary_stream, &graph_obj);
cudaGraphInstantiate(&exec, graph_obj, ...);  // ← 先实例化
state.cleanup_all_events();                     // ← 后才销毁事件
```

**两个关键差异**：

| 操作 | 正常路径 (`capture_cuda`) | NCCL 路径 (`capture_nccl_graph_coordinated`) |
|------|--------------------------|---------------------------------------------|
| `cudaEventCreate` | BeginCapture **之前** | BeginCapture **之后**（违规） |
| `cudaEventDestroy` | Instantiate **之后** | EndCapture **之前**（违规） |

**CUDA 文档明确规定**：在 stream capture 期间，禁止调用任何会创建新 CUDA handle 的 API（包括 `cudaEventCreate`）。此行为属于 **undefined behavior**。

**影响**：事件被过早销毁后，CUDA Graph 中记录的 `cudaEventRecord` / `cudaStreamWaitEvent` 节点可能引用无效的内部状态。Graph 可能仍然能被实例化（不报错），但图节点执行时可能变成空操作，导致 NCCL AllReduce 实际上没有执行。

---

### 2.2 [CRITICAL] `cleanup_all_events` 在 `cudaStreamEndCapture` 之前调用

**位置**：`captured_graph.cpp` 第 559 行

```cpp
// Phase 2 循环内（行 559）
state.cleanup_all_events();   // ← 事件被销毁
}
ncclGroupEnd();

// Phase 3a（行 564-571）
for (int r = 0; r < num_ranks; ++r) {
    cudaStreamEndCapture(cap_streams[r], &captured_graphs[r]);  // ← 此时捕获还在进行中！
}
```

`cleanup_all_events()` 实现（`capture_multi_stream.cpp` 第 52-63 行）：

```cpp
void MultiStreamCaptureState::cleanup_all_events() {
    for (auto ev : temp_events) {
        if (ev) cudaEventDestroy(ev);      // 销毁临时事件
    }
    temp_events.clear();
    for (int i = 0; i < num_active; ++i) {
        if (streams[i].last_done_event) {
            cudaEventDestroy(streams[i].last_done_event);  // 销毁流事件
            streams[i].last_done_event = nullptr;
        }
    }
}
```

**问题**：`cudaStreamEndCapture` 在第 566 行才调用，而事件在第 559 行就被销毁了。此时所有 rank 的 stream 仍处于 capture 模式，图节点中记录的 `cudaEventRecord` 和 `cudaStreamWaitEvent` 操作引用了已被销毁的事件。

---

### 2.3 [HIGH] `get_or_register` 在捕获期间创建新事件

**位置**：`captured_graph.cpp` 第 505-508 行

```cpp
MultiStreamCaptureState state;           // 每次循环新建空 state
state.primary_stream = cap_streams[r];
state.get_or_register(cap_streams[r]);   // 内部调用 cudaEventCreateWithFlags ！
state.get_or_register(dc.stream(COMP_1)); // 内部调用 cudaEventCreateWithFlags ！
state.get_or_register(dc.stream(COMP_2)); // 内部调用 cudaEventCreateWithFlags ！
state.get_or_register(dc.stream(COMP_3)); // 内部调用 cudaEventCreateWithFlags ！
```

`get_or_register` 实现（`capture_multi_stream.cpp` 第 22-36 行）：

```cpp
int MultiStreamCaptureState::get_or_register(cudaStream_t s) {
    int idx = find_stream_index(s);
    if (idx >= 0) return idx;
    // ...
    cudaEventCreateWithFlags(&streams[idx].last_done_event, cudaEventDisableTiming);
    // ↑ 捕获期间创建新 CUDA handle —— 违规！
    return idx;
}
```

每个 rank 在捕获期间创建 4 个事件，8 RANK 总计 32 次违规创建。CUDA 文档明确禁止在捕获期间调用 `cudaEventCreate`。

---

### 2.4 [MEDIUM] NCCL 图执行句柄无验证保护

**位置**：`deep_learning_task.cpp` 第 761-766 行

```cpp
static const GraphSlot kRequired[] = {
    GraphSlot::XFER_A, GraphSlot::XFER_B,
    GraphSlot::FWD_BWD_DEEP_A, GraphSlot::FWD_BWD_DEEP_B,
    GraphSlot::FIRST_LAYER_BWD_A,
    GraphSlot::FIRST_LAYER_BWD_B,
    // 注意：DEEP_ALLREDUCE 和 FIRST_LAYER_ALLREDUCE 不在列表中！
};
```

而在训练循环中（第 1126-1132 行）：

```cpp
if (n_dar) cudaGraphLaunch(n_dar, s_up);   // 如果 n_dar 为 nullptr，静默跳过
sync_up();
// ...
if (n_far) cudaGraphLaunch(n_far, s_up);   // 如果 n_far 为 nullptr，静默跳过
sync_up();
```

如果通信图的 exec handle 为 nullptr（例如因为捕获失败或图损坏），则 `cudaGraphLaunch` 被静默跳过，**不会报任何错误**，梯度不会同步，训练退化到每个 RANK 独立训练。

---

### 2.5 [LOW] `insert_cross_op_barrier` 对 AllReduce 节点的默认流分类

**位置**：`capture_multi_stream.cpp` 第 80-93 行

```cpp
case RangeOp::RANGE_ACCUM_METRICS:
case RangeOp::RANGE_CLEAR:
case RangeOp::RANGE_CAST_FP32_TO_FP16:
case RangeOp::RANGE_CAST_FP16_TO_FP32:
case RangeOp::RANGE_EMA_PARAM_UPDATE:
    target_sk = StreamKind::UPDATE; break;
case RangeOp::RANGE_GRAD_SCALING:
case RangeOp::RANGE_CHECK_NAN:
    target_sk = StreamKind::COMP_1; break;
default:
    target_sk = StreamKind::COMP_1; break;  // ← RANGE_MEAN_ALLREDUCE 走这里
```

`RANGE_MEAN_ALLREDUCE` 和 `RANGE_SUM_ALLREDUCE` 未被显式列出，走 `default` 分支被映射到 `StreamKind::COMP_1`。但实际 AllReduce 操作在 `launch_allreduce_cuda_impl` 中使用的是 `StreamKind::UPDATE`。这可能导致跨算子屏障插入到错误的流上。不过对于 DEEP_COMM/FIRST_COMM 这类通常只有 1 个节点（AllReduce）的图，此问题影响较小。

---

## 3. 为什么 2 RANK 正常但 8 RANK 异常？

这是本问题最关键的线索。解释如下：

| 配置 | 捕获期间创建/销毁的事件数 | UB 触发概率 |
|------|--------------------------|------------|
| 2 RANK | 2×4 = 8 个事件 | 低——事件池小，可能未触发内存踩踏 |
| 4 RANK | 4×4 = 16 个事件 | 中等 |
| 8 RANK | 8×4 = 32 个事件 | 高——事件池大，内部状态损坏概率显著增加 |

**核心机制**：

1. `capture_nccl_graph_coordinated` 在单个线程上串行处理所有 rank
2. 每个 rank 的循环中创建/销毁 `MultiStreamCaptureState`（栈对象），内部创建事件
3. 这些事件被 `cudaEventRecord` / `cudaStreamWaitEvent` 记录到正在捕获的 CUDA Graph 中
4. 事件在 `cudaStreamEndCapture` 前被销毁，但 CUDA 驱动可能将事件操作以某种内部引用形式嵌入图节点
5. 少量 rank 时，事件池较小，内部引用可能恰好保持有效
6. 大量 rank 时，事件创建/销毁频繁，CUDA 驱动的内部事件池可能复用或失效，导致图节点中记录的事件引用变为无效

**可能的结果**：图被成功实例化（不报错），但执行时 NCCL AllReduce 节点变为空操作（no-op）。这完美解释了"程序不卡死不报错，但通信不生效"的现象。

---

## 4. 修改建议

### 4.1 修复事件生命周期（最高优先级）

**策略**：将 `capture_nccl_graph_coordinated` 的事件管理对齐到 `capture_cuda` 的模式。

具体步骤：

1. **在 Phase 1（`cudaStreamBeginCapture`）之前**创建所有 rank 的事件：

```cpp
// Phase 0: 在所有 BeginCapture 之前创建事件
std::vector<MultiStreamCaptureState> states(num_ranks);
for (int r = 0; r < num_ranks; ++r) {
    DeviceContext& dc = *contexts[r];
    cudaSetDevice(dc.device_id());
    states[r].primary_stream = cap_streams[r];
    states[r].get_or_register(cap_streams[r]);
    states[r].get_or_register(dc.stream(StreamKind::COMP_1));
    states[r].get_or_register(dc.stream(StreamKind::COMP_2));
    states[r].get_or_register(dc.stream(StreamKind::COMP_3));
    for (int i = 0; i < states[r].num_active; ++i) {
        if (states[r].streams[i].last_done_event) {
            cudaEventDestroy(states[r].streams[i].last_done_event);
        }
        cudaEventCreateWithFlags(&states[r].streams[i].last_done_event, 
                                  cudaEventDisableTiming);
    }
}

// Phase 1: BeginCapture on ALL ranks
for (int r = 0; r < num_ranks; ++r) {
    cudaSetDevice(contexts[r]->device_id());
    cudaStreamBeginCapture(cap_streams[r], cudaStreamCaptureModeThreadLocal);
}
```

2. **Phase 2 中不再创建/销毁事件**，直接使用 Phase 0 预创建的事件：

```cpp
ncclGroupStart();
for (int r = 0; r < num_ranks; ++r) {
    DeviceContext& dc = *contexts[r];
    cudaSetDevice(dc.device_id());
    auto& state = states[r];  // 使用预创建的事件
    
    // 跨流屏障引入
    if (state.num_active > 1) {
        cudaEventRecord(state.streams[0].last_done_event, state.primary_stream);
        for (int i = 1; i < state.num_active; ++i) {
            cudaStreamWaitEvent(state.streams[i].stream,
                               state.streams[0].last_done_event, 0);
        }
    }
    
    // Replay nodes（保持不变）
    // ...
    finalize_cross_stream_barrier(state);
    // 注意：不再调用 cleanup_all_events！
}
ncclGroupEnd();
```

3. **在 Phase 3b（`cudaGraphInstantiate`）之后**统一销毁事件：

```cpp
// Phase 3b: Instantiate on ALL ranks
for (int r = 0; r < num_ranks; ++r) {
    // ... cudaGraphInstantiate ...
    result.graphs[k].set_rank_exec(r, exec);
    if (captured_graphs[r]) cudaGraphDestroy(captured_graphs[r]);
}

// 在所有图实例化完成后再销毁事件
for (int r = 0; r < num_ranks; ++r) {
    cudaSetDevice(contexts[r]->device_id());
    states[r].cleanup_all_events();
}
```

### 4.2 添加 NCCL 图句柄验证（建议）

在 `build_exec_table` 中，添加对 `DEEP_ALLREDUCE` 和 `FIRST_LAYER_ALLREDUCE` 的检查：

```cpp
// 对于多卡训练，通信图必须有效
if (K > 1) {
    for (int rank = 0; rank < K; ++rank) {
        for (size_t v = 0; v < 4; ++v) {
            TR_CHECK(gpu_exec_.variant_graphs[v][rank][S(GraphSlot::DEEP_ALLREDUCE)],
                     RuntimeError, "DEEP_ALLREDUCE is nullptr for v=" << v << " rank=" << rank);
            TR_CHECK(gpu_exec_.variant_graphs[v][rank][S(GraphSlot::FIRST_LAYER_ALLREDUCE)],
                     RuntimeError, "FIRST_LAYER_ALLREDUCE is nullptr for v=" << v << " rank=" << rank);
        }
    }
}
```

### 4.3 修复 `insert_cross_op_barrier` 流映射（低优先级）

在 `capture_multi_stream.cpp` 的 `insert_cross_op_barrier` 中，显式添加 AllReduce 的流映射：

```cpp
case RangeOp::RANGE_SUM_ALLREDUCE:
case RangeOp::RANGE_MEAN_ALLREDUCE:
case RangeOp::RANGE_BN_STATS_ALLREDUCE:
    target_sk = StreamKind::UPDATE; break;
```

---

## 5. 验证方案

修复后，建议通过以下方式验证：

1. **单元测试**：在 `capture_nccl_graph_coordinated` 中，每次 `cudaEventCreate` 和 `cudaEventDestroy` 后检查 `cudaGetLastError()`，确认没有隐式错误。

2. **图节点验证**：在 `cudaGraphInstantiate` 之后，使用 `cudaGraphGetNodes` 检查图节点数量是否与预期一致（DEEP_COMM 应包含 AllReduce + scale 等节点）。

3. **梯度对比测试**：使用 `torch.distributed.all_reduce` 对同一份梯度数据进行同步，对比自定义实现的结果。在 8 RANK 下，两者的梯度值应该完全一致。

4. **端到端验证**：在 8 RANK 下运行完整训练，确认准确率恢复到 99.5% 水平。

5. **CUDA Graph 调试**：设置环境变量 `CUDA_LAUNCH_BLOCKING=1` 和 `NCCL_DEBUG=INFO` 运行，检查 NCCL 是否有警告或错误日志。

---

## 6. 总结

**根因**：`capture_nccl_graph_coordinated` 函数在 CUDA Graph 捕获期间（`BeginCapture` 与 `EndCapture` 之间）创建和销毁 CUDA Event，违反了 CUDA 编程规范。事件被过早销毁导致图节点中的事件引用失效，NCCL AllReduce 变为空操作。

**2RANK 正常、8RANK 异常的原因**：事件数量随 rank 数线性增长，UB 触发概率随事件池大小增加。2 RANK 的事件池较小，未触发内部损坏；8 RANK 的事件池较大，CUDA 驱动内部状态损坏概率显著增加。

**核心修复**：将事件创建移到 `cudaStreamBeginCapture` 之前，将事件销毁移到 `cudaGraphInstantiate` 之后。