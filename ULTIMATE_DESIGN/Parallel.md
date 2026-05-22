# 并行运行指南

## 一、两个层次

```
                         TaskBase::run("xfer_b", "fwd_bwd_deep_a")
                                       │
                  ┌────────────────────┼────────────────────┐
                  │                                         │
             Rank 0 thread                            Rank 1..N-1 threads
          cudaSetDevice(0)                          cudaSetDevice(1..N-1)
                  │                                         │
          ┌───────┴───────┐                         ┌───────┴───────┐
          │ Level 2: 双流  │                         │ Level 2: 双流  │
          │ TRANS  + COMP_1│                         │ TRANS  + COMP_1│
          └───────────────┘                         └───────────────┘
                  │                                         │
               [join]                                   [join]
                  │                                         │
                  └────────────────────┬────────────────────┘
                                       │
                                  sync_all()
                     (cudaDeviceSynchronize 所有 rank 的所有流)
```

| | Level 1（多卡，per-rank） | Level 2（同卡双图，dual-graph） |
|---|---|---|
| **并行单元** | GPU（rank） | CUDA Stream |
| **手段** | `std::thread` 每 rank 一个 | 同一线程，顺序 `cudaGraphLaunch` 到两条流 |
| **线程数** | N（= rank 数） | 0（不额外展开） |
| **同步** | `sync_all`: `cudaDeviceSynchronize` × N | GPU 硬件调度，CPU 不干预 |

▌ Level 1 用线程：每个 rank 的 CUDA 上下文（通过 `cudaSetDevice` 绑定）是线程局部的。不同 rank 的 `cudaGraphLaunch` 必须在不同线程中发出。

▌ Level 2 不用线程：`cudaGraphLaunch` 是非阻塞的——CPU 把图提交到流后立即返回。提交图 A 到 TRANS 流，再提交图 B 到 COMP_1 流，GPU 硬件调度器自动让它们并行执行。中间不需要 sync，也不需要第二个线程。

---

## 二、数据结构

### 图注册表

[task_base.h:L151-L154](file:///r:\renaissance\include_legacy\renaissance\task\task_base.h#L151)

```cpp
struct GraphEntry { ComputationGraph graph; StreamKind stream; };
std::unordered_map<std::string, GraphEntry> named_graphs_;
```

图在注册时即与其 CUDA Stream 绑定：

[task.cpp:L505-L531](file:///r:\renaissance\src_legacy\task\task.cpp#L505)

```cpp
add_graph("xfer_a",         xfer_a,         StreamKind::TRANS);
add_graph("xfer_b",         xfer_b,         StreamKind::TRANS);
add_graph("fwd_bwd_deep_a", fwd_bwd_a,      StreamKind::COMP_1);
add_graph("fwd_bwd_deep_b", fwd_bwd_b,      StreamKind::COMP_1);
add_graph("first_bwd",      first_bwd,      StreamKind::COMP_1);
```

### Per-rank 捕获句柄

[task_base.cpp:L303-L307](file:///r:\renaissance\src_legacy\task\task_base.cpp#L303)

```cpp
// captured_graphs_[graph_name][rank] = cudaGraphExec_t
std::unordered_map<std::string, std::vector<cudaGraphExec_t>> captured_graphs_;
```

同一张逻辑图在每张 GPU 上有独立的 `cudaGraphExec_t`（句柄内固化显存地址），运行时两级索引。

---

## 三、单图运行 `run(name)` — 仅 Level 1

[task_base.cpp:L2584-L2638](file:///r:\renaissance\src_legacy\task\task_base.cpp#L2584)

```cpp
void TaskBase::run(const std::string& name) {
    StreamKind stream = named_graphs_[name].stream;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_gpus_; ++i) {
        threads.emplace_back([&, i]() {
            backend_->run_graph(name, i, stream);   // 每线程一张图
        });
    }
    for (auto& t : threads) t.join();
    backend_->sync_all();
}
```

**时序**：N 个线程同时启动（`cudaSetDevice(i) → cudaGraphLaunch`），全部 join 后 `cudaDeviceSynchronize` 所有 GPU。

---

## 四、双图运行 `run(a, b)` — Level 1 + Level 2

[task_base.cpp:L2640-L2702](file:///r:\renaissance\src_legacy\task\task_base.cpp#L2640)

```cpp
void TaskBase::run(const std::string& a, const std::string& b) {
    StreamKind stream_a = named_graphs_[a].stream;
    StreamKind stream_b = named_graphs_[b].stream;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_gpus_; ++i) {
        threads.emplace_back([&, i]() {
            backend_->run_graph(a, i, stream_a);  // Level 2: TRANS 流
            backend_->run_graph(b, i, stream_b);  // Level 2: COMP_1 流
        });
    }
    for (auto& t : threads) t.join();
    backend_->sync_all();
}
```

### 时序（单 rank）

```
TRANS 流:   ──[ H2D: batch_{next} ]──────────────
                     ↑ 紧接着，无 sync
COMP_1 流:  ──[ fwd+bwd: batch_{current} ]───────
                     ↑
            GPU 硬件并行执行
```

### 底层 `run_graph`：仅 launch，不 sync

[task_base.cpp:L1384-L1387](file:///r:\renaissance\src_legacy\task\task_base.cpp#L1384)

```cpp
cudaStream_t exec_stream = device_contexts_[rank]->stream(stream);
cudaGraphLaunch(graph_exec, exec_stream);
// 无 cudaStreamSynchronize —— 立即返回
```

▌ **这就是 Level 2 用不上额外线程的根本原因**：`cudaGraphLaunch` 只提交任务到 GPU 队列、不等待执行。两次非阻塞 launch 后，GPU 自动并行。

### CPU 模式：无真实 Level 2

[task_base.cpp:L2693-L2698](file:///r:\renaissance\src_legacy\task\task_base.cpp#L2693)

```cpp
// CPU 模式：完全串行，接口一致
for (int i = 0; i < num_gpus_; ++i) {
    backend_->run_graph(a, i, stream_a);  // 等 a 跑完
    backend_->run_graph(b, i, stream_b);  // 再跑 b
}
```

CPU 路径下 `run_graph` 只是遍历 `cpu_tasks_[name]` 的函数队列逐一执行 ([task_base.cpp:L1401-L1408](file:///r:\renaissance\src_legacy\task\task_base.cpp#L1401))。

---

## 五、训练循环 — A/B 双缓冲流水线

[task.cpp:L790-L873](file:///r:\renaissance\src_legacy\task\task.cpp#L790)

```
TransferStation                          Device
  buffer_0  ──"xfer_a"──→  Device_A ──"fwd_bwd_deep_a"──→
  buffer_1  ──"xfer_b"──→  Device_B ──"fwd_bwd_deep_b"──→
```

双图的真正意义：**传输下一 batch 和计算当前 batch 重叠**。

```
Phase 1:  run("xfer_a")                      传输 batch 0（计算还没法做）

Phase 2:  for batch in 0..N-2:
              run("xfer_?","fwd_bwd_deep_?")   ← 双图：传输 batch_{next} + 计算 batch_{cur}
              run("first_layer_bwd")           ← 必须在双图 sync 之后
              scheduler.step()

Phase 3:  run("fwd_bwd_deep_?")              最后 batch（上一轮已传输）
```

以 batch=0 为例：`run("xfer_b", "fwd_bwd_deep_a")` — TRANS 流传下一批数据（batch 1），COMP_1 流同时计算当前批（batch 0）。H2D 和 kernel 执行在 GPU 上重叠。

---

## 六、要点

- **Level 1 (`std::thread`)** 是必需的：不同线程持不同 rank 的 CUDA 设备上下文
- **Level 2 (双流)** 不展开线程：`cudaGraphLaunch` 非阻塞，两次 launch 到不同流，GPU 硬件自动并行
- **`sync_all`** 只在所有 rank join 之后调用一次：走 `cudaDeviceSynchronize`
- **A/B 双缓冲** 不是为了"同一批数据并行"，而是让 H2D 与计算重叠，形成三级流水