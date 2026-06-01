# 多 RANK 训练准确率异常 — 最终根因分析与修复方案

> **日期**: 2026-06-01
> **基准**: 单 RANK 准确率 99.41%
> **现象**: 2RANK=99.62%（异常高），4RANK=99.44%（正常），8RANK=99.04%（严重偏低）
> **参考**: YVP1.md, YVP2.md, YVP3.md, YVP4.md, NCL1.md, NCL2.md, NCL3.md
> **代码依据**: 以 `captured_graph.cpp`, `capture_cuda.cpp`, `task_base.cpp`, `allreduce_op.cpp` 等源码为唯一事实依据

---

## 一、对 YVP1~YVP4 各意见的独立评估

在阅读所有 YVP 分析报告并逐一对照源码后，以下是对各关键论断的独立评估：

| 论断 | 提出者 | 评估 | 源码依据 |
|------|--------|------|----------|
| `has_nccl_ops()` 检查整图而非子图 | YVP1/2/3/4 | **确认正确，P0 根因** | [computation_graph.h:310-326](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L310-L326) 遍历 `linear_nodes_` + 所有 `graphs_` bucket；[captured_graph.cpp:284-289](file:///r:/renaissance/src/graph/captured_graph.cpp#L284-L289) 直接用整图判断 |
| Event 生命周期违规（capture 期间创建/销毁事件） | YVP1/2/3/4, NCL1 | **确认正确，P0 根因** | [captured_graph.cpp:505-514](file:///r:/renaissance/src/graph/captured_graph.cpp#L505-L514) `get_or_register` 在 `BeginCapture` 之后调用 `cudaEventCreate`；[captured_graph.cpp:559](file:///r:/renaissance/src/graph/captured_graph.cpp#L559) `cleanup_all_events` 在 `EndCapture` 之前调用 `cudaEventDestroy` |
| 缺少 `cudaDeviceSynchronize()` 在 BeginCapture 前 | YVP1/3/4 | **部分正确，非独立根因但应修复** | [capture_cuda.cpp:61](file:///r:/renaissance/src/graph/capture_cuda.cpp#L61) 有 sync；[captured_graph.cpp:483-490](file:///r:/renaissance/src/graph/captured_graph.cpp#L483-L490) 无 sync。但 `compile_capture_simple`（[task_base.cpp:456-464](file:///r:/renaissance/src/task/task_base.cpp#L456-L464)）同样无 BeginCapture 前 sync 且 8GPU 测试通过，说明 sync 缺失不独立导致问题 |
| 缺少 `cudaDeviceSynchronize()` 不成立 | YVP2 | **部分错误** | YVP2 用 Phase B2 的 sync（[captured_graph.cpp:274](file:///r:/renaissance/src/graph/captured_graph.cpp#L274)）论证，但 Phase B2 是 warmup 阶段的 sync，与 Phase B3.5 NCCL capture 之间隔了 Phase B3 的整个非 NCCL 捕获阶段 |
| 空 `ncclGroupStart/End` 污染 communicator 状态 | YVP1/2/3/4, NCL2 | **高度可能，但 NCCL 文档未明确** | 由 `has_nccl_ops()` 误判导致，~7 次空 group 调用。NCCL 文档未明确定义空 group 在 CUDA Graph 捕获下的行为，但作为间接后果合理 |
| `compile_capture_simple` 事件管理与 NCCL 路径相同 | YVP2/3 | **确认正确，关键证据** | [task_base.cpp:473-518](file:///r:/renaissance/src/task/task_base.cpp#L473-L518) 与 [captured_graph.cpp:503-559](file:///r:/renaissance/src/graph/captured_graph.cpp#L503-L559) 逐行相同的事件管理模式 |
| 缺少异常安全机制 | YVP1/3/4, NCL3 | **确认正确** | [capture_cuda.cpp:49-59](file:///r:/renaissance/src/graph/capture_cuda.cpp#L49-L59) 有 `CaptureGuard` RAII；`capture_nccl_graph_coordinated` 无 |
| `RANGE_BN_STATS_ALLREDUCE` 未注册 | YVP1/2/3/4, NCL3 | **确认正确，不影响当前 MLP 测试** | [allreduce_op.cpp:153-177](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L153-L177) 仅注册了 RANGE_SUM/MEAN_ALLREDUCE |
| `insert_cross_op_barrier` 流映射缺 AllReduce | YVP1/2/3 | **确认正确，但实际影响为零** | 对于 DEEP_COMM/FIRST_COMM 这类单节点图，`insert_cross_op_barrier` 仅在 `i > 0` 时调用（[captured_graph.cpp:531-532](file:///r:/renaissance/src/graph/captured_graph.cpp#L531-L532)），单节点图不会触发 |
| NCCL 图句柄无验证 | YVP1/2/3/4 | **确认正确** | [deep_learning_task.cpp:761-766](file:///r:/renaissance/src/task/deep_learning_task.cpp#L761-L766) 的 `kRequired` 不含 DEEP_ALLREDUCE/FIRST_LAYER_ALLREDUCE |
| `local_batch_size` 过小 | YVP1/4, NCL3 | **不是框架 bug，是超参调优问题** | `global_batch_size(128)` 语义正确，但 8RANK 下 `local_bs=16` 需要调整 lr |

---

## 二、根因分析

### 2.1 核心线索：为什么 `compile_capture_simple` 通过测试而 `pre_capture` 失败？

这是破解本问题的关键钥匙。两条路径的 NCCL 协调捕获代码**逐行相同**（包括事件创建/销毁的位置），但结果不同：

| 维度 | `compile_capture_simple`（通过） | `pre_capture`（失败） |
|------|-------------------------------|---------------------|
| 文件 | [task_base.cpp:448-559](file:///r:/renaissance/src/task/task_base.cpp#L448-L559) | [captured_graph.cpp:456-597](file:///r:/renaissance/src/graph/captured_graph.cpp#L456-L597) |
| `has_nccl_ops()` 判断范围 | 独立 `ComputationGraph`（每个 SimpleTask 一个）→ 恰好正确 | 共享 `ComputationGraph`（`train_cg_`/`infer_cg_`）→ 整图判断 |
| `capture_nccl_graph_coordinated` 调用次数 | **1 次** | **~10 次**（因 `has_nccl_ops()` 误判） |
| 其中实际含 NCCL 的调用 | 1 次 | **~3-4 次**（FIRST_COMM, DEEP_COMM, VAL_RESULT_COMM, STATS_COMM） |
| 空 `ncclGroupStart/End` 调用 | 0 次 | **~6-7 次** |
| Capture 期间 Event 创建/销毁次数 | 1× 违规 | **~10× 违规** |

**关键证据**：`compile_capture_simple` 使用的是**完全独立**的 `ComputationGraph`——每个 `SimpleTask` 自己创建一个 `ComputationGraph`，这个 graph 要么含 NCCL 要么不含，`has_nccl_ops()` 的判断自然正确。而 `pre_capture` 中，`train_cg_` 是一个**共享的** `ComputationGraph`，包含约 28 个子图（GraphId 桶），`has_nccl_ops()` 只要 `train_cg_` 中**任意一个**子图含 NCCL 就返回 true。

### 2.2 根因定义

#### 🔴 根因 1（P0）：`has_nccl_ops()` 以整图粒度做判定

**位置**: [captured_graph.cpp:284-289](file:///r:/renaissance/src/graph/captured_graph.cpp#L284-L289)

**问题代码**:
```cpp
std::vector<bool> is_nccl_key(K, false);
for (int32_t k = 0; k < K; ++k) {
    if (key_by_idx[k].cg && key_by_idx[k].cg->has_nccl_ops()) {
        is_nccl_key[k] = true;  // ← 对整图判断，而非对 gid 子图
    }
}
```

[`has_nccl_ops()`](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L310-L326) 的实现遍历 `linear_nodes_` + 所有 `graphs_` 桶：

```cpp
bool has_nccl_ops() const {
    for (const auto& node : linear_nodes_) { /* ... */ }
    for (const auto& bucket : graphs_) {       // ← 遍历全部 28 个 GraphId 桶
        for (const auto& node : bucket) { /* ... */ }
    }
    return false;
}
```

**后果链**:
```
train_cg_ 含 DEEP_COMM（有 AllReduce）→ has_nccl_ops()=true
    → train_cg_ 的 10+ 个子图全部被标记为 NCCL
    → ZERO_GRAD, OPTIMIZER, ACCUM_METRICS, EMA_UPDATE, CAST_*, NAN_CHECK, CLEAR_METRICS 等
      全部走 capture_nccl_graph_coordinated
    → 6~7 次空 ncclGroupStart/End 调用
    → 10+ 次 Event 生命周期违规（而非正确的 3~4 次）
```

真正包含 NCCL 操作的子图仅 4 个（[computation_graph.h:82-99](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L82-L99)）：
- `DEEP_COMM` — 深层梯度 AllReduce
- `FIRST_COMM` — 首层梯度 AllReduce
- `STATS_COMM` — BN 统计量 AllReduce（MLP 测试中不创建）
- `VAL_RESULT_COMM` — 验证指标 AllReduce

#### 🔴 根因 2（P0）：CUDA Event 生命周期违规

**位置**: [captured_graph.cpp:503-559](file:///r:/renaissance/src/graph/captured_graph.cpp#L503-L559)

**正确模式**（`capture_cuda.cpp`）：

| 操作 | 位置 | 说明 |
|------|------|------|
| `cudaEventCreate` | [capture_cuda.cpp:67-79](file:///r:/renaissance/src/graph/capture_cuda.cpp#L67-L79) | `BeginCapture` **之前** |
| `cudaStreamBeginCapture` | [capture_cuda.cpp:82](file:///r:/renaissance/src/graph/capture_cuda.cpp#L82) | 事件创建后才开始捕获 |
| `cudaStreamEndCapture` | [capture_cuda.cpp:145](file:///r:/renaissance/src/graph/capture_cuda.cpp#L145) | |
| `cudaGraphInstantiate` | [capture_cuda.cpp:155](file:///r:/renaissance/src/graph/capture_cuda.cpp#L155) | |
| `cleanup_all_events` | [capture_cuda.cpp:171](file:///r:/renaissance/src/graph/capture_cuda.cpp#L171) | `Instantiate` **之后** |

**违规模式**（`capture_nccl_graph_coordinated`）：

| 操作 | 位置 | 说明 |
|------|------|------|
| `cudaStreamBeginCapture` | [captured_graph.cpp:490](file:///r:/renaissance/src/graph/captured_graph.cpp#L490) | 先开始捕获 |
| `get_or_register` → `cudaEventCreate` | [captured_graph.cpp:505-508](file:///r:/renaissance/src/graph/captured_graph.cpp#L505-L508) | **捕获期间创建事件！** |
| `cudaEventDestroy` + `cudaEventCreate` | [captured_graph.cpp:510-514](file:///r:/renaissance/src/graph/captured_graph.cpp#L510-L514) | **捕获期间销毁+创建事件！** |
| `cleanup_all_events` | [captured_graph.cpp:559](file:///r:/renaissance/src/graph/captured_graph.cpp#L559) | 在 `EndCapture` 之前 |
| `cudaStreamEndCapture` | [captured_graph.cpp:566](file:///r:/renaissance/src/graph/captured_graph.cpp#L566) | 事件已销毁，图节点引用无效句柄 |

CUDA 编程规范明确规定：**stream capture 期间禁止调用任何会创建或销毁 CUDA 对象句柄的 API**（包括 `cudaEventCreate`、`cudaEventDestroy`）。此行为属 **undefined behavior**。

**关键说明**：此违规在 `compile_capture_simple`（[task_base.cpp:473-518](file:///r:/renaissance/src/task/task_base.cpp#L473-L518)）中同样存在，且该路径在 8 GPU 上测试通过。因此：**Event 生命周期违规本身不是独立根因**，需要与根因 1 的放大效应叠加才会触发 UB 表现。

#### 根因 1 × 根因 2 的叠加效应

```
根因 1: has_nccl_ops() 误判
    → 10+ 个 graph 进入 NCCL 协调捕获（而非正确的 3~4 个）
    → 每次都触发根因 2 的 Event 违规
    → 10+ 次违规 × 8 GPU × 4 event/graph = 320+ 次违规操作

根因 2: Event 生命周期违规
    → 图节点中记录的 cudaEventRecord/cudaStreamWaitEvent 引用已释放的事件句柄
    → 在 1 次违规时（compile_capture_simple），UB 未触发
    → 在 10+ 次违规时（pre_capture），UB 触发概率显著增加

叠加结果:
    → CUDA Graph 被成功实例化（不报错）
    → 运行时 NCCL AllReduce 节点静默变为 no-op
    → 梯度不同步，各 RANK 独立训练
```

### 2.3 关于 `cudaDeviceSynchronize()` 缺失的独立评估

YVP2 声称此问题不成立，理由是 Phase B2 末尾已有 sync（[captured_graph.cpp:274](file:///r:/renaissance/src/graph/captured_graph.cpp#L274)）。但此论证存在逻辑缺陷：

**时间线**:
```
Phase B2: warmup → cudaDeviceSynchronize() (line 274)  ← YVP2 引用的 sync
Phase B3: 捕获非 NCCL graph（capture_cuda，内部 sync）
Phase B3.5: capture_nccl_graph_coordinated() ← 无 sync
```

Phase B2 的 sync 发生在 warmup 结束后，但 Phase B3.5 的 NCCL 捕获在 Phase B3 的整个非 NCCL 捕获阶段之后。Phase B3 内部 `capture_cuda` 有 sync（[capture_cuda.cpp:61](file:///r:/renaissance/src/graph/capture_cuda.cpp#L61)），但 `cudaGraphDestroy`（[capture_cuda.cpp:169](file:///r:/renaissance/src/graph/capture_cuda.cpp#L169)）和 `cleanup_all_events`（[capture_cuda.cpp:171](file:///r:/renaissance/src/graph/capture_cuda.cpp#L171)）可能有异步副作用。此外，连续多次 `capture_nccl_graph_coordinated` 调用之间也缺少 sync。

**结论**: 缺少 sync 是真实存在的差异，但：
1. 不是独立根因（`compile_capture_simple` 同样没有 sync 且通过测试）
2. 应作为防御性修复加入（对齐 `capture_cuda.cpp` 的模式）

### 2.4 关于空 `ncclGroupStart/End` 的评估

这是根因 1 的间接后果。NCCL 文档对 CUDA Graph 捕获的要求是：
> "All ranks must call the same sequence of NCCL operations within the group."

当 group 中没有任何 NCCL 操作时，这是一个空 group。NCCL 文档未明确定义空 group 在 CUDA Graph 捕获模式下的行为。虽然没有直接证据证明空 group 会污染 communicator 状态，但作为一个异常调用路径，其影响不可控。

**修复策略**: 通过修复根因 1（`has_nccl_ops()` 子图级检查），空 group 调用将被自动消除（只有真正含 NCCL 的图才走协调捕获）。

### 2.5 实验数据的机理解释

| 配置 | 有效 batch（通信失效时） | 有效 epoch（通信失效时） | 实际准确率 | 解释 |
|------|----------------------|----------------------|-----------|------|
| 2RANK | 64 | 50 | 99.62% | 独立训练 + 小 batch 噪声正则化 → 泛化性能意外提升 |
| 4RANK | 32 | 25 | 99.44% | 独立训练 25 epoch，MNIST MLP 基本饱和 → 巧合正常 |
| 8RANK | 16 | 12.5 | 99.04% | 独立训练 12.5 epoch，数据严重不足 → 欠拟合 |

训练 loss 几乎重合的原因：MNIST 数据随机均匀分片，各 RANK 的本地 loss 自然相似。训练 loss 从 rank 0 读取（[deep_learning_task.cpp:1668-1669](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1668-L1669)），造成"训练正常"的假象。

验证指标同理：`VAL_RESULT_ALLREDUCE` 同样可能失效 + `fetch_from_rank(..., 0)` 只读 rank 0 → 验证准确率仅反映 rank 0 本地模型质量。

---

## 三、修复方案

### 3.1 Fix A（P0）：为 `ComputationGraph` 添加子图粒度的 `has_nccl_ops(GraphId)` 重载

**涉及文件**:
- `include/renaissance/graph/computation_graph.h`：新增 `has_nccl_ops(GraphId gid)` 重载
- `src/graph/captured_graph.cpp`：调用点改用子图粒度判定（Phase B2.5 + Phase B4 warmup）

**原理**: 将 NCCL 判定从整图粒度缩小到当前 `gid` 子图粒度，确保只有真正包含 NCCL 节点的子图才走协调捕获路径。这是消除过度调用和空 group 的根源修复。

**Step 1 — 在 `ComputationGraph` 中新增子图粒度重载**:

```cpp
// computation_graph.h，has_nccl_ops() 之后添加：
/// 检测指定 GraphId 子图中是否包含 NCCL 集合操作
bool has_nccl_ops(GraphId gid) const {
    auto is_nccl_range = [](const GraphNode& node) -> bool {
        return node.kind == GraphNode::Kind::RANGE &&
               (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE ||
                node.range_op == RangeOp::RANGE_BN_STATS_ALLREDUCE);
    };
    for (const auto& node : graphs_[static_cast<size_t>(gid)]) {
        if (is_nccl_range(node)) return true;
    }
    return false;
}
```

**Step 2 — Phase B2.5 调用点（captured_graph.cpp:284-289）**:

**修改前**:
```cpp
std::vector<bool> is_nccl_key(K, false);
for (int32_t k = 0; k < K; ++k) {
    if (key_by_idx[k].cg && key_by_idx[k].cg->has_nccl_ops()) {
        is_nccl_key[k] = true;
    }
}
```

**修改后**:
```cpp
std::vector<bool> is_nccl_key(K, false);
for (int32_t k = 0; k < K; ++k) {
    if (key_by_idx[k].cg && key_by_idx[k].cg->has_nccl_ops(key_by_idx[k].gid)) {
        is_nccl_key[k] = true;
    }
}
```

**Step 3 — Phase B4 warmup launch 跳过逻辑（captured_graph.cpp:359）**:

**修改前**:
```cpp
if (cg.key().cg && cg.key().cg->has_nccl_ops()) {
    TR_LOG_INFO("graph") << "[B4] Skip NCCL graph ...";
    continue;
}
```

**修改后**:
```cpp
if (cg.key().cg && cg.key().cg->has_nccl_ops(cg.key().gid)) {
    TR_LOG_INFO("graph") << "[B4] Skip NCCL graph ...";
    continue;
}
```

**预期效果**:
- `[B3-NCCL]` 日志从 ~10 条降至 **3-4 条**（DEEP_COMM, FIRST_COMM, VAL_RESULT_COMM, STATS_COMM）
- 消除 6-7 次空 `ncclGroupStart/End` 调用
- Event 违规次数从 10+× 降至 3-4×（与 `compile_capture_simple` 的 1+× 更接近）
- `ZERO_GRAD`, `OPTIMIZER`, `ACCUM_METRICS`, `EMA_UPDATE`, `CAST_*`, `NAN_CHECK`, `CLEAR_METRICS` 恢复正常独立捕获
- NCCL 判定逻辑集中在 `ComputationGraph` 单一位置，未来新增 NCCL RangeOp 时只需修改一处

**风险**: 低。仅改变判定粒度，不改变捕获逻辑。重载方法复用已有 `is_nccl_range` lambda 结构。

---

### 3.2 Fix B（P0）：修正 CUDA Event 生命周期

**文件**: `src/graph/captured_graph.cpp`，`capture_nccl_graph_coordinated`（第 456-597 行）

**原理**: 将 CUDA Event 的创建/销毁与 CUDA Stream Capture 的 `BeginCapture`/`EndCapture` 边界对齐。CUDA 规范要求：Event 创建必须在 `BeginCapture` 前，Event 销毁必须在 `cudaGraphInstantiate` 后。修复后与 `capture_cuda.cpp` 的正确模式完全一致。

**修改策略**: 将 `MultiStreamCaptureState` 从栈对象（每次循环创建/销毁）提升为持久对象，生命周期覆盖整个函数。

**Step 1 — Phase 0（新增）：在所有 BeginCapture 之前预创建事件**

```cpp
static void capture_nccl_graph_coordinated(...) {
    // ... 现有代码（key/mp/stream_kind 解析）...

    const int num_ranks = static_cast<int>(contexts.size());
    std::vector<cudaStream_t> cap_streams(num_ranks);
    std::vector<cudaGraph_t> captured_graphs(num_ranks, nullptr);

    // =====================================================================
    // Phase 0（新增）: 预创建所有事件（对齐 capture_cuda.cpp 模式）
    // =====================================================================
    // 注意：对于 NCCL 图（FIRST_COMM / DEEP_COMM / VAL_RESULT_COMM），
    // cap_streams[r] 即为 dc.stream(StreamKind::UPDATE)（由 deep_learning_task.cpp:594-608
    // 的 get_stream_kind() 决定）。因此 launch_allreduce_cuda_impl 中的
    // get_or_register(UPDATE) 会命中已注册的 primary stream，不会在 capture 期间创建新 event。
    std::vector<MultiStreamCaptureState> states(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        DeviceContext& dc = *contexts[r];
        cudaSetDevice(dc.device_id());
        cudaDeviceSynchronize();  // 对齐 capture_cuda.cpp:61，确保无残留异步工作

        dc.set_rank(r);
        dc.set_memory_plan(mp);
        cap_streams[r] = static_cast<cudaStream_t>(dc.stream(stream_kind));

        states[r].primary_stream = cap_streams[r];
        states[r].get_or_register(cap_streams[r]);
        states[r].get_or_register(
            static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_1)));
        states[r].get_or_register(
            static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_2)));
        states[r].get_or_register(
            static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_3)));

        for (int i = 0; i < states[r].num_active; ++i) {
            if (states[r].streams[i].last_done_event) {
                cudaEventDestroy(states[r].streams[i].last_done_event);
            }
            cudaEventCreateWithFlags(
                &states[r].streams[i].last_done_event,
                cudaEventDisableTiming);
        }
    }

    // =====================================================================
    // Phase 1: BeginCapture on ALL ranks（不变）
    // =====================================================================
    for (int r = 0; r < num_ranks; ++r) {
        cudaSetDevice(contexts[r]->device_id());
        cudaError_t cap_err = cudaStreamBeginCapture(
            cap_streams[r], cudaStreamCaptureModeThreadLocal);
        if (cap_err != cudaSuccess) {
            TR_DEVICE_ERROR("cudaStreamBeginCapture failed for rank " << r
                            << ": " << cudaGetErrorString(cap_err));
        }
    }
```

**Step 2 — Phase 2：使用预创建的事件，不再创建/销毁**

```cpp
    ncclGroupStart();
    for (int r = 0; r < num_ranks; ++r) {
        DeviceContext& dc = *contexts[r];
        cudaSetDevice(dc.device_id());
        MultiStreamCaptureState& state = states[r];  // 使用 Phase 0 预创建

        // 跨流屏障引入（事件已在 Phase 0 创建好）
        if (state.num_active > 1) {
            cudaEventRecord(state.streams[0].last_done_event, state.primary_stream);
            for (int i = 1; i < state.num_active; ++i) {
                cudaStreamWaitEvent(state.streams[i].stream,
                                   state.streams[0].last_done_event, 0);
            }
        }

        const auto& nodes = key.cg->nodes(key.gid);
        if (!nodes.empty()) {
            LOG_INFO << "[CAPTURE-CUDA] gid=" << static_cast<int>(key.gid)
                     << " nodes=" << nodes.size() << " (NCCL coordinated)";
            for (size_t i = 0; i < nodes.size(); ++i) {
                const auto& node = nodes[i];
                if (i > 0) {
                    insert_cross_op_barrier(nodes[i-1], node, state, dc);
                }
                if (node.kind == GraphNode::Kind::COMPUTE) {
                    auto& entry = g_compute_op_table[static_cast<size_t>(node.compute_op)];
                    if (entry.launch_cuda) {
                        entry.launch_cuda(node, *mp, dc, state);
                    } else {
                        StreamKind sk = get_op_default_stream(node.compute_op);
                        cudaStream_t s = static_cast<cudaStream_t>(dc.stream(sk));
                        int si = state.get_or_register(s);
                        state.output_stream_idx = si;
                        state.streams[si].has_pending_work = true;
                        cudaEventRecord(state.streams[si].last_done_event, s);
                    }
                } else {
                    auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
                    if (entry.launch_cuda) {
                        entry.launch_cuda(node, *mp, dc, state);
                    } else {
                        TR_DEVICE_ERROR("Unimplemented RangeOp in NCCL graph: "
                                        << static_cast<int>(node.range_op));
                    }
                }
            }
            finalize_cross_stream_barrier(state);
        }
        // 注意：不再调用 state.cleanup_all_events()！
    }
    ncclGroupEnd();
```

**Step 3 — Phase 3a/3b 不变，Phase 4（新增）在 Instantiate 后统一销毁**

```cpp
    // Phase 3a: EndCapture on ALL ranks（不变）
    // Phase 3b: Instantiate on ALL ranks（不变）

    // =====================================================================
    // Phase 4（新增）: 所有 Instantiate 完成后销毁事件
    // =====================================================================
    for (int r = 0; r < num_ranks; ++r) {
        cudaSetDevice(contexts[r]->device_id());
        states[r].cleanup_all_events();
    }

    CapturedGraph meta;
    meta.set_key(key);
    meta.set_is_cuda(true);
    result.graphs[k].set_metadata_from(meta);
}
```

**关键变化总结**:
1. `MultiStreamCaptureState` 从栈对象改为 `std::vector` 持久管理
2. `cudaEventCreate` 移到 `cudaStreamBeginCapture` 之前（Phase 0）
3. Phase 0 中添加 `cudaDeviceSynchronize()`，对齐 `capture_cuda.cpp:61`
4. `cleanup_all_events()` 移到 `cudaGraphInstantiate` 之后（Phase 4）
5. Phase 2 中不再有任何 event 创建/销毁操作
6. 与 `capture_cuda.cpp` 模式完全对齐

**关于 UPDATE 流的说明**: NCCL 图（FIRST_COMM, DEEP_COMM, VAL_RESULT_COMM）在 `get_stream_kind()`（[deep_learning_task.cpp:594-608](file:///r:/renaissance/src/task/deep_learning_task.cpp#L594-L608)）中映射为 `StreamKind::UPDATE`，因此 `cap_streams[r] = dc.stream(StreamKind::UPDATE)` 已覆盖 AllReduce 操作所需的流。`launch_allreduce_cuda_impl` 中 `get_or_register(dc.stream(StreamKind::UPDATE))` 会命中 Phase 0 已注册的 primary stream（指针相同），不会在 capture 期间创建新 event。

**风险**: 低。逻辑等价于 `capture_cuda.cpp`，仅将 `MultiStreamCaptureState` 从栈对象提升为 `std::vector` 管理。

---

### 3.3 Fix C（P1）：同步修正 `compile_capture_simple` 的 Event 管理

**文件**: `src/task/task_base.cpp`，`compile_capture_simple` 的 NCCL 协调捕获部分（第 448-559 行）

**原理**: 虽然 `compile_capture_simple` 当前通过测试，但其 Event 管理模式与 `capture_cuda.cpp` 存在同样的不一致。为预防未来在更复杂场景下出问题，应同步修复为与 Fix B 相同的模式。

**修改**: 参照 Fix B 的 Phase 0-4 结构，将 `task_base.cpp` 中 NCCL 协调捕获部分的事件管理修正为：
1. 预创建 `MultiStreamCaptureState` 为 `std::vector`（或单 rank 时为栈对象）
2. `cudaEventCreate` 移到 `cudaStreamBeginCapture` 之前
3. `cleanup_all_events()` 移到 `cudaGraphInstantiate` 之后
4. **防御性预注册 `StreamKind::UPDATE`**：`GraphEntry.stream` 默认为 `StreamKind::COMP_1`（[task_base.h:202-205](file:///r:/renaissance/include/renaissance/task/task_base.h#L202-L205)），但 `launch_allreduce_cuda_impl` 内部固定使用 `StreamKind::UPDATE`（[allreduce_op.cpp:40](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L40)）。若 SimpleTask 未显式将 NCCL 图的 `entry.stream` 设为 `UPDATE`，Phase 0 预注册的 `cap_streams[r]` 为 `COMP_1`，replay 时 `get_or_register(UPDATE)` 可能隐式创建事件。因此 Phase 0 应显式预注册 `StreamKind::UPDATE`：
   ```cpp
   // Fix C Phase 0 中增加：
   state.get_or_register(static_cast<cudaStream_t>(dc.stream(StreamKind::UPDATE)));
   ```

**风险**: 低。与 Fix B 完全相同的改造模式，仅应用的上下文不同。`get_or_register` 对重复流去重，即使 `cap_streams[r]` 已经是 `UPDATE` 也无害。

---

### 3.4 Fix D（P1）：添加异常安全机制

**文件**: `src/graph/captured_graph.cpp`，`capture_nccl_graph_coordinated`

**原理**: 如果 `ncclGroupStart/End` 之间的 replay 抛出异常，所有 RANK 的 stream 会永久卡在 capture 状态，且 NCCL communicator 会处于未关闭的 group 状态。需要添加异常保护，确保异常时：
1. 先关闭 NCCL group（`ncclGroupEnd()`），避免 communicator 状态泄漏
2. 再强制 `EndCapture` 以释放 stream
3. 清理已预创建的事件（来自 Fix B 的 `states`）

**修改**（包裹 Phase 2）:
```cpp
// Phase 2: ncclGroupStart → replay all ranks → ncclGroupEnd
bool capture_committed = false;

try {
    ncclGroupStart();
    for (int r = 0; r < num_ranks; ++r) {
        /* ... replay ... */
    }
    ncclGroupEnd();
    capture_committed = true;
} catch (...) {
    // 1. 必须关闭 NCCL group，否则 communicator 状态泄漏，
    //    后续任何 NCCL 操作都可能产生未定义行为或死锁
    ncclGroupEnd();

    // 2. 强制 EndCapture 以避免 stream 永久卡在 capture 状态
    for (int r = 0; r < num_ranks; ++r) {
        cudaSetDevice(contexts[r]->device_id());
        cudaGraph_t dummy = nullptr;
        cudaError_t end_err = cudaStreamEndCapture(
            cap_streams[r], &dummy);
        if (dummy) cudaGraphDestroy(dummy);
        (void)end_err;
    }

    // 3. 清理 Phase 0 预创建的事件（来自 Fix B）
    for (int r = 0; r < num_ranks; ++r) {
        cudaSetDevice(contexts[r]->device_id());
        states[r].cleanup_all_events();
    }

    throw;
}
```

**风险**: 低。仅增加异常保护路径，正常路径不受影响。`ncclGroupEnd()` 在异常路径中的调用是防御性的，NCCL 文档允许在非正常流程中调用 `ncclGroupEnd()` 来清理状态。

---

### 3.5 Fix E（P2）：注册 `RANGE_BN_STATS_ALLREDUCE` 并修正 `do_mean` 语义覆盖

**文件**: `src/backend/ops/range/allreduce_op.cpp`

**原理**: `RANGE_BN_STATS_ALLREDUCE` 在 GraphId 定义和 `has_nccl_ops()` 中已被识别，但在两方面存在遗漏：
1. `register_op_range_allreduce()` 中缺少运行时的 op 注册
2. `launch_allreduce_cuda_impl` 和 `launch_allreduce_cpu_impl` 中的 `do_mean` 条件仅匹配 `RANGE_MEAN_ALLREDUCE`，未覆盖 `RANGE_BN_STATS_ALLREDUCE`

BN 统计量（mean/variance）跨 RANK 时需要 **平均**（而非纯 sum），语义与 `RANGE_MEAN_ALLREDUCE` 相同。若 `do_mean` 不覆盖，AllReduce 只做 sum 不做 `1/world_size` post-scale → BN 统计量被放大 `world_size` 倍 → BN 行为异常。

**Step 1 — 注册 op**（在 `register_op_range_allreduce()` 函数末尾 `}` 之前添加）:
```cpp
{
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_BN_STATS_ALLREDUCE)];
    entry.op = RangeOp::RANGE_BN_STATS_ALLREDUCE;
    entry.launch_cpu = launch_allreduce_cpu_impl;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_allreduce_cuda_impl;
#endif
}
```

**Step 2 — CUDA 路径扩展 `do_mean`**（[allreduce_op.cpp:51](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L51)）:

**修改前**:
```cpp
bool do_mean = (node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE);
```

**修改后**:
```cpp
bool do_mean = (node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE ||
                node.range_op == RangeOp::RANGE_BN_STATS_ALLREDUCE);
```

**Step 3 — CPU 路径扩展 `do_mean`**（[allreduce_op.cpp:110](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L110)）:

**修改前**:
```cpp
bool do_mean = (op_ctx->range_op == RangeOp::RANGE_MEAN_ALLREDUCE);
```

**修改后**:
```cpp
bool do_mean = (op_ctx->range_op == RangeOp::RANGE_MEAN_ALLREDUCE ||
                op_ctx->range_op == RangeOp::RANGE_BN_STATS_ALLREDUCE);
```

**风险**: 低。`do_mean` 仅控制 `ncclAllReduce(ncclSum)` 之后的 `1.0f/world_size` 缩放。BN 统计量需要平均语义，加入该条件后行为正确且与 `RANGE_MEAN_ALLREDUCE` 一致。

---

### 3.6 Fix F（P2）：添加 NCCL 图句柄有效性验证

**文件**: `src/task/deep_learning_task.cpp`，`build_exec_table`（第 761-778 行）

**原理**: 当前 `kRequired` 列表（[deep_learning_task.cpp:761-766](file:///r:/renaissance/src/task/deep_learning_task.cpp#L761-L766)）不包含 `DEEP_ALLREDUCE` 和 `FIRST_LAYER_ALLREDUCE`。如果通信图捕获失败产生 nullptr，训练循环中 `if (n_dar) cudaGraphLaunch(...)` 会静默跳过（[deep_learning_task.cpp:1174](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1174)），梯度不同步且无任何错误提示。

**修改**（在 `kRequired` 检查之后添加）:
```cpp
if (K > 1) {
    for (size_t v = 0; v < 4; ++v) {
        for (int rank = 0; rank < K; ++rank) {
            TR_CHECK(gpu_exec_.variant_graphs[v][rank][S(GraphSlot::DEEP_ALLREDUCE)],
                     RuntimeError,
                     "DEEP_ALLREDUCE is nullptr for v=" << v << " rank=" << rank);
            TR_CHECK(gpu_exec_.variant_graphs[v][rank][S(GraphSlot::FIRST_LAYER_ALLREDUCE)],
                     RuntimeError,
                     "FIRST_LAYER_ALLREDUCE is nullptr for v=" << v << " rank=" << rank);
        }
    }
}
```

**风险**: 低。快速失败优于静默跳过。

---

### 3.7 Fix G（P2）：修正 `insert_cross_op_barrier` 的 AllReduce 流映射

**文件**: `src/graph/capture_multi_stream.cpp`，`insert_cross_op_barrier`（第 80-92 行）

**原理**: `RANGE_MEAN_ALLREDUCE` 等走 default 分支被映射到 `StreamKind::COMP_1`，但实际 AllReduce 操作在 `launch_allreduce_cuda_impl` 中使用 `StreamKind::UPDATE`（[allreduce_op.cpp:40](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L40)）。虽然对于 DEEP_COMM/FIRST_COMM 这类单节点图（`insert_cross_op_barrier` 仅在 `i > 0` 时调用，而单节点图不触发），实际影响为零，但语义上应修复。

**修改**（在 `RANGE_GRAD_SCALING` case 之前添加）:
```cpp
case RangeOp::RANGE_SUM_ALLREDUCE:
case RangeOp::RANGE_MEAN_ALLREDUCE:
case RangeOp::RANGE_BN_STATS_ALLREDUCE:
    target_sk = StreamKind::UPDATE; break;
```

**风险**: 低。对当前单节点 NCCL 图无影响，对未来多节点 NCCL 图提供正确性保证。

---

## 四、修复清单总览

| Fix | 优先级 | 文件 | 修改内容 | 性质 |
|-----|--------|------|----------|------|
| **A** | **P0** | `computation_graph.h`（新增重载）<br>`captured_graph.cpp:284-289`（调用点）<br>`captured_graph.cpp:359`（warmup 跳过） | 新增 `has_nccl_ops(GraphId gid)` 子图粒度重载；调用点改用子图判定 | 根源修复 |
| **B** | **P0** | `captured_graph.cpp:456-597` | Event 创建移到 `BeginCapture` 前（+ `cudaDeviceSynchronize`），销毁移到 `Instantiate` 后 | 根源修复 |
| C | P1 | `task_base.cpp:448-559` | `compile_capture_simple` 同步使用正确 Event 生命周期（+ UPDATE 流预注册） | 一致性修复 |
| D | P1 | `captured_graph.cpp:498-561` | `ncclGroupStart/End` 块包裹 try/catch（含 `ncclGroupEnd` + event 清理） | 防御性修复 |
| E | P2 | `allreduce_op.cpp:51,110,153-177` | 注册 `RANGE_BN_STATS_ALLREDUCE` + 扩展 `do_mean` 覆盖 | 功能补全 |
| F | P2 | `deep_learning_task.cpp:761-778` | 多卡时检查 `DEEP_ALLREDUCE`/`FIRST_LAYER_ALLREDUCE` 非空 | 防御性修复 |
| G | P2 | `capture_multi_stream.cpp:80-92` | AllReduce 显式映射到 `StreamKind::UPDATE` | 语义修复 |

---

## 五、修复执行顺序与验证方案

### 5.1 执行顺序

1. **同时实施 Fix A + Fix B**（P0）—— 改动集中在 `captured_graph.cpp`，互为依赖
2. 编译验证（Windows 本地增量编译，无新增 warning）
3. A100×8 端到端测试：`./mnist_best --amp --activation hardswish`
   - 如果准确率恢复至 99.40%~99.50% → 根因确认，继续
   - 如果准确率仍偏低 → 追加诊断日志（见 5.4），进一步排查
4. 实施 Fix C（P1）—— `compile_capture_simple` 一致性修复
5. 实施 Fix D（P1）—— 异常安全机制
6. 实施 Fix E/F/G（P2）—— 防御性完善
7. 最终回归测试（单 RANK + 2/4/8 RANK）确认稳定性

### 5.2 编译与日志验证

- `[B3-NCCL]` 日志数量从 `~10` 降至 `3~4`
- 确认 `capture_nccl_graph_coordinated` 中无 "capture 期间 create event" 的违规操作

### 5.3 端到端验证

| 测试项 | 预期结果 |
|--------|---------|
| 单 RANK 回归 | 保持 99.41% |
| 2RANK | 回落至 99.40%~99.50% 正常区间（不再异常偏高） |
| 4RANK | 保持 99.40%~99.50% |
| 8RANK | **恢复到 99.40%~99.50%**，与单 RANK 基本一致 |
| 收敛曲线 | 1/2/4/8 RANK 的验证曲线应基本重合 |

### 5.4 诊断手段（如问题持续）

若 Fix A+B 后准确率仍未恢复，添加以下诊断：

1. **梯度值检查**：在 `launch_allreduce_cuda_impl` 中 AllReduce 前后各打印一个梯度元素的 sum，确认通信实际生效
2. **图节点数检查**：在 `cudaGraphInstantiate` 后使用 `cudaGraphGetNodes` 验证节点数符合预期
3. **NCCL 调试日志**：设置 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH` 环境变量
4. **逐 rank 验证指标**：修改验证循环，从所有 rank 读取指标而非仅 rank 0，对比各 rank 的差异

---

## 六、预期效果

| 配置 | 当前准确率 | 预期准确率（修复后） | 理由 |
|------|-----------|-------------------|------|
| 1RANK | 99.41% | 99.41%（不变） | 单 RANK 路径不受影响 |
| 2RANK | 99.62% | 99.40%~99.50% | 通信恢复，梯度正确同步，不再异常偏高 |
| 4RANK | 99.44% | 99.40%~99.50% | 通信恢复，梯度正确同步 |
| 8RANK | 99.04% | 99.40%~99.50% | 通信恢复，梯度正确同步 |

---

## 七、对 Reviewer 反馈的评估与方案修订

在 YVP_REV.md 中，reviewer（小伙伴 K）提出了 3 个"必须修正"的问题和 2 个建议改进。以下是对每条反馈逐一的代码核查与结论：

### 7.1 问题 1：UPDATE 流预注册遗漏 → **不成立**

reviewer 认为 Fix B Phase 0 遗漏了 `StreamKind::UPDATE` 的预注册，会导致 `launch_allreduce_cuda_impl` 在 capture 期间通过 `get_or_register` 隐式创建 CUDA Event。

**代码核实**: 检查 NCCL 图的 stream 映射链路：

1. [deep_learning_task.cpp:594-608](file:///r:/renaissance/src/task/deep_learning_task.cpp#L594-L608)：`FIRST_COMM`, `DEEP_COMM`, `VAL_RESULT_COMM` 均映射为 `StreamKind::UPDATE`
2. [captured_graph.cpp:473-474](file:///r:/renaissance/src/graph/captured_graph.cpp#L473-L474)：`stream_kind = sl.stream_kind`，即 `StreamKind::UPDATE`
3. [captured_graph.cpp:489](file:///r:/renaissance/src/graph/captured_graph.cpp#L489)（旧）/ Phase 0（新）：`cap_streams[r] = dc.stream(stream_kind)` = `dc.stream(StreamKind::UPDATE)`
4. Phase 0 预注册：`states[r].get_or_register(cap_streams[r])` 已注册 `UPDATE` 流
5. [allreduce_op.cpp:40-41](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L40-L41)：`launch_allreduce_cuda_impl` 中 `ctx.stream(StreamKind::UPDATE)` 返回相同指针 → `find_stream_index` 命中 → 不创建新 event

**结论**: `UPDATE` 流通过 `cap_streams[r]` 已被隐式预注册，不存在遗漏。已在 Fix B 中添加关于此机制的说明性注释。

### 7.2 问题 2：缺少 `cudaDeviceSynchronize()` → **采纳**

YVP_FINAL §2.3 已明确指出 sync 缺失是"真实存在的差异，应作为防御性修复加入"，但 Fix B 原始代码未包含。`capture_cuda.cpp:61` 在 event 预创建之前明确调用了 `cudaDeviceSynchronize()`，Fix B 应对齐。

**修复**: 已在 Fix B Phase 0 的每个 rank 循环中添加 `cudaDeviceSynchronize()`。

### 7.3 问题 3：Fix D catch 块遗漏 `ncclGroupEnd()` → **采纳**

reviewer 指出如果异常发生在 `ncclGroupStart()` 之后、`ncclGroupEnd()` 之前，catch 块没有调用 `ncclGroupEnd()` 会导致 NCCL communicator 处于未关闭的 group 状态。这是真实的状态泄漏 bug。

**修复**: 已在 Fix D 的 catch 块中：
1. 首行添加 `ncclGroupEnd()`（关闭 NCCL group）
2. 添加 Phase 0 预创建事件的清理（`states[r].cleanup_all_events()`）
3. 改用局部 `cudaGraph_t dummy` 替代 `std::vector<cudaGraph_t> emergency_graphs`（更简洁）

### 7.4 建议 1：`has_nccl_ops(GraphId gid)` 重载 → **采纳**

reviewer 建议将 NCCL 判定逻辑从调用点内联移到 `ComputationGraph` 的方法中，避免代码重复和维护分散。

**采纳理由**: 
1. NCCL 判定逻辑集中在单一位置，未来新增 NCCL RangeOp 时只需修改一处
2. Phase B2.5（判定）+ Phase B4（warmup 跳过）复用同一方法
3. 与现有 `has_nccl_ops()`（无参数版）形成清晰的 API 分层

已在 Fix A Step 1 中添加 `has_nccl_ops(GraphId gid)` 重载。

### 7.5 建议 2：全量预注册所有流 → **不采纳（当前阶段），保留为未来选项**

reviewer 建议在 Phase 0 预注册 `dc` 提供的所有 stream kind（COMP_1/2/3、UPDATE、XFER_A、XFER_B 等），以防御未来新增 Op 引入未预注册流。

**不采纳理由**: 
1. 当前 NCCL 图仅使用 `UPDATE` + 可能的 `COMP_1/2/3`，Phase 0 已全覆盖
2. `get_or_register` 对重复流是去重的，多预注册确实无害，但当前阶段过度注册增加不必要的代码复杂度
3. 若未来引入使用新流的 NCCL 图，应在对应的 Op 实现中处理，而非在此处全面预注册

**备选方案**: 可考虑给 `MultiStreamCaptureState` 增加一个断言/调试开关，在 capture 模式下禁止 `get_or_register` 创建新 event。但这是独立的质量改进，不属于本次修复范围。

### 7.6 第二轮评审（YVP_REV2.md）

reviewer 接受了第一轮所有修订，但提出 1 个新问题 + 1 个新建议：

**7.6.1 Fix E `do_mean` 语义未覆盖 `RANGE_BN_STATS_ALLREDUCE` → 采纳**

reviewer 指出：Fix E 将 `RANGE_BN_STATS_ALLREDUCE` 注册为 `launch_allreduce_cuda_impl` / `launch_allreduce_cpu_impl`，但这两个实现内部的 `do_mean` 条件仅匹配 `RANGE_MEAN_ALLREDUCE`（[allreduce_op.cpp:51](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L51)、[allreduce_op.cpp:110](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L110)）。若 BN 统计量的语义是需要平均的（sum + `1/world_size` post-scale），则 AllReduce 只会做 sum 不会做 post-scale → BN stats 被放大 `world_size` 倍。

**代码核实**: 在 [allreduce_op.cpp:51](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L51) 和 [allreduce_op.cpp:110](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L110) 确认 `do_mean` 仅检查 `RANGE_MEAN_ALLREDUCE`，未覆盖 `RANGE_BN_STATS_ALLREDUCE`。Fix E 的说明文字也已写明 "BN 统计量使用 sum + mean 语义"，此处存在矛盾。

**修复**: 已在 Fix E 中新增 Step 2（CUDA 路径）和 Step 3（CPU 路径），将 `do_mean` 条件扩展为同时匹配 `RANGE_MEAN_ALLREDUCE || RANGE_BN_STATS_ALLREDUCE`。

**7.6.2 Fix C 需预注册 `StreamKind::UPDATE` → 采纳（防御性）**

reviewer 指出：`GraphEntry.stream` 默认为 `StreamKind::COMP_1`（[task_base.h:204](file:///r:/renaissance/include/renaissance/task/task_base.h#L204)），若 SimpleTask 未显式设置 stream 为 `UPDATE`，Fix C Phase 0 预注册的 `cap_streams[r]` 就是 `COMP_1`。而 `launch_allreduce_cuda_impl` 固定用 `StreamKind::UPDATE`（[allreduce_op.cpp:40](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L40)），replay 时会触发 `get_or_register(UPDATE)` 隐式创建 event。

**代码核实**: [task_base.h:202-205](file:///r:/renaissance/include/renaissance/task/task_base.h#L202-L205) 确认 `stream` 默认值为 `COMP_1`，[compile_capture_simple:463](file:///r:/renaissance/src/task/task_base.cpp#L463) 使用 `dc.stream(entry.stream)`。当前 `compile_capture_simple` 测试通过（可能与具体 SimpleTask 的 stream 设置有关），但作为防御性修复合理。

**修复**: 已在 Fix C 中新增第 4 点，在 Phase 0 中显式预注册 `StreamKind::UPDATE`。

---

## 八、总结

### 根本原因

`has_nccl_ops()` 以整个 `ComputationGraph` 为粒度做判定（而非当前 gid 子图），导致 `train_cg_` 中约 10 个非 NCCL 子图被错误送入 `capture_nccl_graph_coordinated`。该函数在 `cudaStreamBeginCapture` 之后创建/销毁 CUDA Event，违反 CUDA 编程规范（UB）。在 `compile_capture_simple` 路径中此违规只发生 1 次（8GPU 测试通过），但在 `pre_capture` 路径中因 `has_nccl_ops()` 误判被放大至 10+ 次，UB 触发概率显著增加，导致 NCCL AllReduce 节点在运行时静默变为 no-op，梯度不能跨 RANK 同步，多 RANK 训练退化为各 RANK 独立训练。

### 核心修复

1. **Fix A（P0）**：将 `has_nccl_ops()` 判定粒度从整图缩小到目标 gid 子图 —— 消除过度调用的根源
2. **Fix B（P0）**：将 Event 生命周期与 `capture_cuda.cpp` 正确模式完全对齐 —— 消除 CUDA 规范违规

这两项修复分别解决"为什么被放大"和"违规本身是什么"，共同确保多 RANK NCCL 协调捕获的正确性。