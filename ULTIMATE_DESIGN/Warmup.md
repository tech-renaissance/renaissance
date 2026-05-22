# 预热捕获技术指南 —— Warmup & Capture Guide

## 一、编译管线四阶段

旧版 `compile()` 分解为四个阶段（[task_base.cpp:L2062-L2082](file:///r:\renaissance\src_legacy\task\task_base.cpp#L2062)）：

```
Phase 1: compile_warmup_master_gpu()    — GPU 0 单卡串行预热 cuDNN engine cache
Phase 2: compile_capture_all_graphs()   — GPU 0 串行捕获 + GPU 1~7 并行捕获
Phase 3: compile_warmup_graphs()        — Master rank 每图一次 launch + sync 验证
Phase 4: compile_mark_compiled()        — 标记编译完成
```

MLPerf 计时从 Phase 4 之后开始，前三个阶段不计时。

---

## 二、Phase 1 — 单卡 cuDNN 预热

**文件**：[task_base.cpp:L2231-L2328](file:///r:\renaissance\src_legacy\task\task_base.cpp#L2231)

```cpp
void TaskBase::compile_warmup_master_gpu() {
    int master_gpu = GlobalRegistry::instance().gpu_ids()[0];
    cudaSetDevice(master_gpu);

    for (const auto& kv : named_graphs_) {
        for (const auto& node : kv.second.graph.nodes()) {
            const auto* cn = std::get_if<ComputationGraph::ComputeNode>(&node);
            if (!cn || !Module::is_cudnn_op(cn->kind)) continue;

            // 构建 cuDNN FE graph → validate → execution_plans → build_plans
            auto cu_graph = Module::build_cudnn_graph(...);
            cu_graph->validate();
            cu_graph->create_execution_plans();
            cu_graph->build_plans(HEURISTICS_CHOICE);   // ← 触发引擎搜索

            // 用真实设备指针 execute 一次，让 cuDNN 内部 cache 落盘
            cu_graph->execute(handle, vp, temp_ws);
            ++warmed;
        }
    }
    cudaDeviceSynchronize();  // 暴露同步错误，不延迟到训练期
}
```

**原理**：`build_plans(HEURISTICS_CHOICE)` 首次调用时触发 cuDNN 引擎搜索，结果写入进程级全局 cache。主线程在 GPU 0 上串行把所有 cuDNN 算子跑一遍，后续并行捕获全部命中缓存，不再触发昂贵搜索。

**为什么必须单卡串行？** cuDNN engine cache 并非 per-handle 隔离，底层是进程级全局数据结构。如果 8 线程并发 `build_plans()`，会严重串行化（cuDNN 内部有全局锁），且可能产生损坏 cache 条目。

---

## 三、Phase 2 — 多卡并行捕获

### 3.1 顶层策略 "0 串行 + 1~7 并行"

**文件**：[task_base.cpp:L2330-L2398](file:///r:\renaissance\src_legacy\task\task_base.cpp#L2330)

```cpp
void TaskBase::compile_capture_all_graphs() {
    std::vector<std::exception_ptr> exc(num_gpus_);

    // Step 1: GPU 0 在主线程串行捕获（独占 cache，零竞争）
    {
        try {
            backend_->capture_all_graphs(0);
        } catch (...) {
            exc[0] = std::current_exception();
        }
    }

    // Step 2: GPU 1~7 并行捕获（cache 已预热，只剩读竞争）
    std::vector<std::thread> threads;
    for (int i = 1; i < num_gpus_; ++i) {
        threads.emplace_back([&, i]() {
            try {
                backend_->capture_all_graphs(i);
            } catch (...) {
                exc[i] = std::current_exception();
            }
        });
    }
    for (auto& t : threads) t.join();

    // 聚合异常：任意 rank 失败都要终止（不能在子线程直接 throw）
    for (int i = 0; i < num_gpus_; ++i) {
        if (exc[i]) std::rethrow_exception(exc[i]);
    }
}
```

**核心洞察**：不是 8 卡全开线程，而是 Rank 0 先串行、其余再并行。三步串行化策略的背后原理：

```
Step 0（Phase 1预热）: GPU 0 串行 build_plans + execute  →  进程级 cache 填满
Step 1（Phase 2前半）: GPU 0 在主线程串行 capture        →  独占 cache，绝对干净
Step 2（Phase 2后半）: GPU 1~7 并行 capture               →  cache 已热，纯读命中
```

验证结论：
- 单 GPU / CPU 模式：始终通过
- 8 GPU 全并行捕获：间歇性失败（竞争概率触发，表现为 `illegal memory access`）
- 修复后（0 串行 + 1~7 并行）：不再触发

### 3.2 单 rank 捕获入口 `capture_all_graphs(int rank)`

**文件**：[task_base.cpp:L1116-L1139](file:///r:\renaissance\src_legacy\task\task_base.cpp#L1116)

```cpp
void capture_all_graphs(int rank) {
    // 关键：子线程必须绑定设备
    int physical_gpu_id = GlobalRegistry::instance().gpu_ids()[rank];
    cudaSetDevice(physical_gpu_id);

    for (const auto& kv : owner->named_graphs_) {
        capture_graph(kv.first, kv.second.graph, kv.second.stream, rank);
    }
}
```

**要点**：
1. `cudaSetDevice` 是**每线程必须调用**的 —— CUDA 默认设备 0，新线程不绑定会导致图被错误捕获到 0 号卡
2. 按图序逐一 `capture_graph`，不是在 `named_graphs_` 上开并行（多线程的并行粒度为 rank，而非单卡内部的图）

### 3.3 单图捕获 `capture_graph(name, graph, stream, rank)`

**文件**：[task_base.cpp:L1141-L1290](file:///r:\renaissance\src_legacy\task\task_base.cpp#L1141)

这是最底层的 CUDA Graph 捕获实现，从 `cudaStreamBeginCapture` 到 `cudaGraphInstantiate`。关键片段：

```cpp
void capture_graph(const std::string& name, const ComputationGraph& graph,
                   StreamKind stream, int rank) {
    cudaSetDevice(reg.gpu_ids()[rank]);
    cudaStream_t capture_stream = device_contexts_[rank]->stream(stream);

    // ═════════ RAII Guard ═════════
    struct CudaGraphCaptureGuard {
        cudaStream_t stream = nullptr;
        bool committed = false;
        ~CudaGraphCaptureGuard() {
            if (!committed && stream) {
                cudaGraph_t dummy = nullptr;
                cudaStreamEndCapture(stream, &dummy);  // 强制结束
                if (dummy) cudaGraphDestroy(dummy);
            }
        }
    };
    CudaGraphCaptureGuard guard{capture_stream, false};

    // ═════════ Begin ═════════
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);

    // ═════════ Replay all nodes ═════════
    for (const auto& node : graph.nodes()) {
        std::visit([&](auto&& arg) {
            if constexpr (is ComputeNode) {
                execute_compute_node(arg, rank, stream);  // kernel launch
            } else if constexpr (is TransferNode) {
                cudaMemcpyAsync(dst, src, bytes, H2D, capture_stream);
            }
        }, node);
    }

    // ═════════ End + Instantiate ═════════
    cudaGraph_t graph_obj = nullptr;
    cudaStreamEndCapture(capture_stream, &graph_obj);
    guard.committed = true;           // RAII 不再介入

    cudaGraphExec_t graph_exec = nullptr;
    cudaGraphNode_t error_node;
    char error_log[2048];
    cudaGraphInstantiate(&graph_exec, graph_obj, &error_node, error_log, sizeof(error_log));

    // ═════════ Thread-safe store ═════════
    {
        std::lock_guard<std::mutex> lock(capture_mutex_);
        if (captured_graphs_.find(name) == captured_graphs_.end())
            captured_graphs_[name] = std::vector<cudaGraphExec_t>(num_gpus);
        captured_graphs_[name][rank] = graph_exec;
    }

    cudaGraphDestroy(graph_obj);      // 图对象可销毁，exec 已独立
}
```

**四项关键技术点**：

| 点 | 做法 | 原因 |
|----|------|------|
| **流模式** | `cudaStreamCaptureModeThreadLocal` | 只捕获本线程提交的 kernel，不干扰其他流上的操作 |
| **RAII Guard** | 异常析构时`cudaStreamEndCapture(&dummy)` | 若 `execute_compute_node` 抛异常而流仍处于 capture 态，后续任何 CUDA API 都报 `cudaErrorStreamCaptureUnsupported` |
| **error_log[2048]** | `cudaGraphInstantiate` 第四个参数 | `cudaGraphInstantiate` 执行内核级验证，错误信息在此 buffer；不传则静默失败 |
| **mutex + per-rank vector** | `captured_graphs_[name][rank] = exec` | 主线程（GPU 0）+ 多个子线程（GPU 1~7）并发写入同一 `captured_graphs_` map，必须加锁 |

### 3.4 数据布局

**文件**：[task_base.cpp:L291-L307](file:///r:\renaissance\src_legacy\task\task_base.cpp#L291)

```cpp
struct Backend {
    // ── Per-rank 执行引擎 ──
    std::vector<std::unique_ptr<DeviceContext>> device_contexts_;

    // ── 图表：graph_name → per_rank_vector<exec> ──
    std::unordered_map<std::string, std::vector<cudaGraphExec_t>> captured_graphs_;
    std::mutex capture_mutex_;
};
```

**两级索引**：`captured_graphs_[name][rank]`，name 区分不同子图（forward/backward/update），rank 区分不同 GPU。每个 rank 独立持有 `cudaGraphExec_t`，运行时 O(1) 查表：

```cpp
// [task_base.cpp:L1355-L1390]
cudaGraphExec_t exec = captured_graphs_[name][rank];
cudaGraphLaunch(exec, exec_stream);
```

---

## 四、Phase 3 — 捕获后 Warmup

**文件**：[task_base.cpp:L2400-L2462](file:///r:\renaissance\src_legacy\task\task_base.cpp#L2400)

```cpp
void TaskBase::compile_warmup_graphs() {
    int master_rank = 0;
    int master_gpu  = GlobalRegistry::instance().gpu_ids()[master_rank];
    cudaSetDevice(master_gpu);

    for (const auto& [name, entry] : named_graphs_) {
        auto it = backend_->captured_graphs_.find(name);
        cudaGraphExec_t exec = it->second[master_rank];
        cudaStream_t stream = device_contexts_[0]->stream(entry.stream);

        cudaGraphLaunch(exec, stream);      // 执行一次
        cudaStreamSynchronize(stream);      // 等待完成，暴露异步错误
    }
}
```

**为什么需要这一步？** `cudaGraphInstantiate` 只验证图拓扑合法性，不检查 kernel 执行结果。Warmup launch + sync 可以在 MLPerf 计时前暴露"illegal memory access"、"invalid parameter"等运行时错误。

---

## 五、异常聚合模式

并行捕获中，**不能在子线程直接 throw**（主线程收不到）。旧版使用 `std::exception_ptr` 聚合：

```cpp
std::vector<std::exception_ptr> exc(num_gpus_);

// 每个捕获路径捕获异常到 exc[rank]
try { capture_all_graphs(i); }
catch (...) { exc[i] = std::current_exception(); }

// join 后统一 rethrow
for (int i = 0; i < num_gpus_; ++i) {
    if (exc[i]) std::rethrow_exception(exc[i]);
}
```

---

## 六、新版 CapturedGraph 对接建议

### 6.1 当前 CapturedGraph 结构缺陷

```cpp
// 改前：flat vector，不区分 rank
struct PreCaptureResult {
    std::vector<CapturedGraph> graphs;  // 去重后图集，无 per-rank 维度
    GraphAtlas atlas;
};
```

多卡场景下，同一张图在不同 GPU 上有不同的 `cudaGraphExec_t`（绑定了不同的 device id）。当前数据结构无法承载 per-rank 存储。

### 6.2 建议方案

```cpp
struct PreCaptureResult {
    // 去重后的逻辑图（Key 唯一）。每个逻辑图对应一个 per-rank exec 向量。
    std::vector<CapturedGraph> graphs;

    // Per-rank exec 索引：graphs_[k].execs_[rank_]
    // 多卡: execs_[0..N-1] 各有独立 cudaGraphExec_t
    // 单卡: execs_[0] 仅一个
    std::vector<std::vector<NativeGraph>> per_rank_execs;
    // per_rank_execs_[k][r] = 第 k 张去重图在 rank r 上的 cudaGraphExec_t

    GraphAtlas atlas;
    size_t num_ranks = 1;
};
```

### 6.3 `pre_capture()` 内的三段式策略

```cpp
PreCaptureResult pre_capture(const GraphAtlas& atlas, Device* device, int num_ranks) {
    PreCaptureResult result;
    result.num_ranks = num_ranks;

    // Phase A: 去重（与当前一致：Key hash → dedup → 确定逻辑图张数 K）
    auto dedup_keys = build_dedup_keys(atlas);
    result.graphs.resize(K);       // 每种 Key 一张 CapturedGraph
    result.per_rank_execs.resize(K);
    for (auto& vec : result.per_rank_execs)
        vec.resize(num_ranks);

    // Phase B: single-rank warmup (rank 0 only)
    if (device->is_cuda()) {
        cudaSetDevice(gpu_ids[0]);
        for (int k = 0; k < K; ++k) {
            warmup_cudnn_ops(atlas, k, device);  // build_plans + execute on rank 0
        }
        cudaDeviceSynchronize();
    }

    // Phase C: capture — rank 0 serial, ranks 1..N-1 parallel
    std::vector<std::exception_ptr> exc(num_ranks);

    // Rank 0 在主线程串行
    capture_all_ranks(result, atlas, device, 0);

    // Ranks 1..N-1 并行
    std::vector<std::thread> threads;
    for (int r = 1; r < num_ranks; ++r) {
        threads.emplace_back([&, r]() {
            try { capture_all_ranks(result, atlas, device, r); }
            catch (...) { exc[r] = std::current_exception(); }
        });
    }
    for (auto& t : threads) t.join();
    for (int r = 0; r < num_ranks; ++r)
        if (exc[r]) std::rethrow_exception(exc[r]);

    // Phase D: post-capture warmup (rank 0 only)
    warmup_launch_and_sync(result, 0);

    return result;
}
```

### 6.4 RAII Guard 必须实现

在新版的 `capture()` 中存在结构风险（对算子依赖链的执行依赖稳定状态），未来对接真实 CUDA Graph 时**必须**实现 `CudaGraphCaptureGuard`：

```cpp
// 放在 capture() 内部，beginCapture 之后、execute nodes 之前
struct CaptureGuard {
    cudaStream_t stream;
    bool committed = false;
    ~CaptureGuard() {
        if (!committed && stream) {
            cudaGraph_t dummy = nullptr;
            cudaStreamEndCapture(stream, &dummy);
            if (dummy) cudaGraphDestroy(dummy);
        }
    }
} guard{capture_stream, false};
```

没有 Guard 的后果：`execute_compute_node` 抛异常后流仍处于 capture 态，程序后续任何 CUDA API 调用皆报 `cudaErrorStreamCaptureUnsupported`。

### 6.5 线程安全存储

多线程并行捕获时，写入 `per_rank_execs[k][r]` 需要保护。当前设计每个 `(k,r)` 唯一映射到一个线程，天然无竞争——但**新增 Key 到 vector 的扩容操作**（`result.graphs.push_back`）仍需要同步。建议在 Phase A 去重时先 `reserve(K)` 并 `resize` 到最终大小，Phase B/C 按索引写入，零锁竞争。

---

## 七、总结 — 关键认知

```
         Phase 1               Phase 2                  Phase 3
    ┌──────────────┐   ┌──────────────────────┐   ┌──────────────┐
    │ GPU 0 串行    │   │ Rank 0 串行           │   │ Master Rank  │
    │ 预热 cuDNN   │ → │ Rank 1~7 并行         │ → │ warmup launch│
    │ cache        │   │ (cache 已稳定,        │   │ + sync       │
    │              │   │  纯读命中)            │   │              │
    └──────────────┘   └──────────────────────┘   └──────────────┘
      主线程              主线程 + 子线程             主线程
    cudaSetDevice(0)   cudaSetDevice(id)         cudaSetDevice(0)
```

▌ **不是"并行捕获"有多难，而是"怎么让并行捕获时不触发 cuDNN 全局 cache 竞争"。**

▌ 旧版的核心不是「多线程」，而是 **三步串行化**：先用 GPU 0 独占 cache 做完预热和第一个捕获，确定 cache 已稳定、只剩纯读命中后，才对剩余 GPU 开并行。

▌ 新版对接时必须保留这个三段式策略：**预热 → 主卡串行捕获 → 其余并行捕获**。