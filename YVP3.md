# YVP3: 多RANK训练准确率异常 — 完整修复方案

## 一、问题确认

### 1.1 现象

| RANK | Best Val Top-1 | Best Epoch | 评估 |
|------|---------------|------------|------|
| 1    | 99.41%        | 73         | 基准 |
| 2    | **99.62%**    | 87         | 异常偏高 |
| 4    | 99.44%        | 88         | 接近正常 |
| 8    | **99.04%**    | 60         | 严重偏低，后期回落至98.96% |

训练loss曲线在1/2/4/8 RANK下几乎重合，验证loss随RANK数增加而明显上升（8R=0.526 vs 1R=0.516）。

### 1.2 诊断结论

**AllReduce通信在训练路径中未实际生效。** 多RANK训练退化为各RANK独立训练，验证准确率仅反映rank 0本地模型的性能。

数据模式完美符合 "无通信独立训练" 假设：
- 2RANK：local_batch_size=64，更多梯度噪声→意外正则化→99.62%
- 8RANK：local_batch_size=16，噪声过大+有效数据仅1/8→99.04%

---

## 二、根因分析

### 2.1 直接对比：正确路径 vs 问题路径

框架中存在**两条**NCCL协调捕获路径：

| 维度 | `capture_cuda.cpp`（单RANK正常路径，已验证正确） | `capture_nccl_graph_coordinated`（多RANK问题路径） | `compile_capture_simple`（单元测试路径，已通过） |
|------|--------------------------------------------------|---------------------------------------------------|------------------------------------------------|
| `cudaDeviceSynchronize()` | **有**，在event create之前 ✅ | **无** ❌ | **无** ❌ |
| `cudaEventCreate` | **BeginCapture之前** ✅ | **BeginCapture之后** ❌ | **BeginCapture之后** ❌ |
| `cudaEventDestroy` | **Instantiate之后** ✅ | **EndCapture之前** ❌ | **EndCapture之前** ❌ |
| 调用次数 | 1次/graph | **10+次/graph**（has_nccl_ops误判） | 1次/graph |
| 调用环境 | 干净stream | 可能含warmup残留 | 全新SimpleTask，绝对干净 |

`capture_cuda.cpp` 的注释明确说明：
```cpp
// 预注册 capture stream，确保 cudaEventCreateWithFlags 在 capture 外执行
//（CUDA 文档：capture 期间禁止调用返回新 handle 的 API）
```

### 2.2 根因链

```
has_nccl_ops() 检查整个 ComputationGraph（非当前gid）
    │
    ▼
10+ 个 graph 被误标记为 NCCL → 全部走 capture_nccl_graph_coordinated
    │
    ▼
capture_nccl_graph_coordinated 被调用 10+ 次（实际只需 3~4 次）
    │
    ▼
每次调用中：BeginCapture → cudaEventCreate（违规）→ replay → cudaEventDestroy（违规）→ EndCapture
    │
    ▼
8RANK时事件池更大 + 多次违规操作累积 → CUDA Graph中的事件节点引用失效
    │
    ▼
NCCL AllReduce节点运行时静默变为no-op → 梯度不同步 + 验证指标不同步
    │
    ├─→ 各RANK独立训练 → 2R:99.62% / 4R:99.44% / 8R:99.04%
    │
    └─→ VAL_RESULT_ALLREDUCE同样失效 → 验证只反映rank 0本地
```

### 2.3 关键证据

**证据1**：`test_mean_allreduce` 单元测试在8 GPU上PASS。
- 该测试使用 `compile_capture_simple` 路径
- 只调用**1次** `ncclGroupStart`/`ncclGroupEnd`
- Stream来自全新SimpleTask，无任何pending work
- 证明 `RANGE_MEAN_ALLREDUCE` 的kernel实现和NCCL调用本身正确

**证据2**：`compile_capture_simple` 与 `capture_nccl_graph_coordinated` 的事件管理代码逐行相同。
- 差异不在于event管理的"写法"
- 差异在于**调用次数**（1次 vs 10+次）和**调用环境**（干净stream vs 可能含残留）

**证据3**：`capture_cuda.cpp`（正常单RANK路径）明确在 `BeginCapture` 之前执行 `cudaEventCreate`/`cudaEventDestroy`。
- 这是框架内部已验证的"正确模式"
- `capture_nccl_graph_coordinated` 完全违背了这个模式

**证据4**：训练loss在各RANK数下几乎一致。
- 说明前向传播正常
- 各RANK数据随机分片，分布均匀，独立训练的rank 0本地loss自然与单卡相似
- 这是"通信失效"的典型伪装

### 2.4 为什么WNF1/WNF2/WNF3的分析需要综合

| 来源 | 核心观点 | 评估 |
|------|---------|------|
| WNF1 | Event生命周期违规是致命bug | ✅ 正确，但需结合调用次数才能解释8RANK差异 |
| WNF2 | 缺少`cudaDeviceSynchronize()`是关键 | ⚠️ 重要补充，但单独不能解释全部（Phase B2已有sync） |
| WNF3 | `has_nccl_ops()`误判导致过度调用是主因 | ✅ 正确，这是将问题放大的放大器 |

**综合判断**：`has_nccl_ops()` 的误判是**必要条件**（将问题从1次放大到10+次），event在capture期间的创建/销毁是**充分条件**（违反CUDA规范导致UB），两者叠加导致8RANK时NCCL AllReduce静默失效。

---

## 三、修复方案

### Fix 1（P0）：修复 `has_nccl_ops()` 判定粒度

**问题**：当前检查整个 `ComputationGraph`，应改为只检查当前 `gid` 子图。

**文件**：`src/graph/captured_graph.cpp` Phase B2.5

**修改**：
```cpp
// 替换原有代码（line 284-289）
std::vector<bool> is_nccl_key(K, false);
for (int32_t k = 0; k < K; ++k) {
    const auto& key = key_by_idx[k];
    if (!key.cg) continue;
    bool has_nccl = false;
    for (const auto& node : key.cg->nodes(key.gid)) {
        if (node.kind == GraphNode::Kind::RANGE &&
            (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
             node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE ||
             node.range_op == RangeOp::RANGE_BN_STATS_ALLREDUCE)) {
            has_nccl = true;
            break;
        }
    }
    is_nccl_key[k] = has_nccl;
}
```

**预期效果**：`[B3-NCCL]` 日志从 ~10+ 条降到 **3~4 条**（FIRST_COMM、DEEP_COMM、VAL_RESULT_COMM、STATS_COMM）。

---

### Fix 2（P0）：将 `capture_nccl_graph_coordinated` 的事件管理对齐到 `capture_cuda.cpp`

**核心原则**：event 的 create/destroy 必须在 `BeginCapture` 之前，cleanup 必须在 `Instantiate` 之后。

**文件**：`src/graph/captured_graph.cpp`

**修改**：重构 `capture_nccl_graph_coordinated` 为四阶段：

```cpp
static void capture_nccl_graph_coordinated(...)
{
    // ... stream_kind 和 mp 解析不变 ...
    const int num_ranks = static_cast<int>(contexts.size());
    std::vector<cudaStream_t> cap_streams(num_ranks);
    std::vector<cudaGraph_t> captured_graphs(num_ranks, nullptr);

    // =====================================================================
    // Phase 0: 预创建所有事件 + Sync（对齐 capture_cuda.cpp 模式）
    // =====================================================================
    std::vector<MultiStreamCaptureState> states(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        DeviceContext& dc = *contexts[r];
        cudaSetDevice(dc.device_id());
        dc.set_rank(r);
        dc.set_memory_plan(mp);
        cap_streams[r] = static_cast<cudaStream_t>(dc.stream(stream_kind));

        states[r].primary_stream = cap_streams[r];
        states[r].get_or_register(cap_streams[r]);
        states[r].get_or_register(static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_1)));
        states[r].get_or_register(static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_2)));
        states[r].get_or_register(static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_3)));

        for (int i = 0; i < states[r].num_active; ++i) {
            if (states[r].streams[i].last_done_event) {
                cudaEventDestroy(states[r].streams[i].last_done_event);
            }
            cudaEventCreateWithFlags(&states[r].streams[i].last_done_event,
                                      cudaEventDisableTiming);
        }
    }

    // 关键：在所有 BeginCapture 之前 sync，清除任何可能的 pending work
    for (int r = 0; r < num_ranks; ++r) {
        cudaSetDevice(contexts[r]->device_id());
        cudaDeviceSynchronize();
    }

    // =====================================================================
    // Phase 1: BeginCapture on ALL ranks
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

    // =====================================================================
    // Phase 2: ncclGroupStart → replay all ranks → ncclGroupEnd
    // =====================================================================
    ncclGroupStart();
    for (int r = 0; r < num_ranks; ++r) {
        DeviceContext& dc = *contexts[r];
        cudaSetDevice(dc.device_id());
        MultiStreamCaptureState& state = states[r];

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
    }
    ncclGroupEnd();

    // =====================================================================
    // Phase 3a: EndCapture on ALL ranks
    // =====================================================================
    for (int r = 0; r < num_ranks; ++r) {
        cudaSetDevice(contexts[r]->device_id());
        cudaError_t end_err = cudaStreamEndCapture(cap_streams[r], &captured_graphs[r]);
        if (end_err != cudaSuccess) {
            TR_DEVICE_ERROR("cudaStreamEndCapture failed for rank " << r
                            << ": " << cudaGetErrorString(end_err));
        }
    }

    // =====================================================================
    // Phase 3b: Instantiate on ALL ranks
    // =====================================================================
    for (int r = 0; r < num_ranks; ++r) {
        cudaSetDevice(contexts[r]->device_id());
        cudaGraphExec_t exec = nullptr;
        cudaGraphNode_t error_node = nullptr;
        char log_buf[2048] = {};
        cudaError_t inst_err = cudaGraphInstantiate(
            &exec, captured_graphs[r], &error_node, log_buf, sizeof(log_buf));
        if (inst_err != cudaSuccess) {
            std::string info;
            if (log_buf[0] != '\0') info = std::string(" log='") + log_buf + "'";
            TR_DEVICE_ERROR("cudaGraphInstantiate failed for rank " << r
                            << ": " << cudaGetErrorString(inst_err) << info);
        }
        result.graphs[k].set_rank_exec(r, exec);
        if (captured_graphs[r]) cudaGraphDestroy(captured_graphs[r]);
    }

    // =====================================================================
    // Phase 4: 统一销毁事件（关键：必须在 Instantiate 之后）
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

**关键变化**：
1. `MultiStreamCaptureState` 从栈对象改为预分配的 `std::vector`，生命周期贯穿整个函数
2. `cudaEventCreate`/`cudaEventDestroy` 移到 `BeginCapture` 之前（Phase 0）
3. 新增 `cudaDeviceSynchronize()` 在 `BeginCapture` 之前
4. `cleanup_all_events()` 移到 `Instantiate` 之后（Phase 4）
5. Phase 2 中不再有任何 event 创建/销毁操作

---

### Fix 3（P1）：为 `compile_capture_simple` 补充一致的事件管理

**文件**：`src/task/task_base.cpp`

**理由**：`compile_capture_simple` 虽然当前通过测试，但其事件管理与 `capture_cuda.cpp` 不一致。为预防未来在更复杂场景下出问题，应同步修复。

**修改**：在 `compile_capture_simple` 的 coordinated capture 部分，参照 Fix 2 的模式：
1. 预创建 `MultiStreamCaptureState` 到 vector
2. 在 `BeginCapture` 前加 `cudaDeviceSynchronize()`
3. 将 `cleanup_all_events()` 延后到 `Instantiate` 之后

---

### Fix 4（P1）：添加异常安全机制

**文件**：`src/graph/captured_graph.cpp`

**修改**：在 `capture_nccl_graph_coordinated` 的 Phase 2 外层包裹 try/catch，如果 replay 过程中发生异常，强制对所有 rank 执行 `EndCapture` 并清理 graph，避免 stream 永远卡在 capture 状态。

```cpp
bool capture_committed = false;
std::vector<cudaGraph_t> emergency_graphs(num_ranks, nullptr);

try {
    ncclGroupStart();
    for (int r = 0; r < num_ranks; ++r) { /* replay */ }
    ncclGroupEnd();
    capture_committed = true;
} catch (...) {
    for (int r = 0; r < num_ranks; ++r) {
        cudaSetDevice(contexts[r]->device_id());
        cudaStreamEndCapture(cap_streams[r], &emergency_graphs[r]);
        if (emergency_graphs[r]) cudaGraphDestroy(emergency_graphs[r]);
    }
    throw;
}
```

---

### Fix 5（P2）：修正 `insert_cross_op_barrier` 的 AllReduce 流映射

**文件**：`src/graph/capture_multi_stream.cpp`

**修改**：
```cpp
case RangeOp::RANGE_SUM_ALLREDUCE:
case RangeOp::RANGE_MEAN_ALLREDUCE:
case RangeOp::RANGE_BN_STATS_ALLREDUCE:
    target_sk = StreamKind::UPDATE; break;
```

---

### Fix 6（P2）：注册 `RANGE_BN_STATS_ALLREDUCE`

**文件**：`src/backend/ops/range/allreduce_op.cpp`

**修改**：在 `register_op_range_allreduce()` 末尾添加：
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

---

### Fix 7（P2）：添加 NCCL graph 句柄非空断言

**文件**：`src/task/deep_learning_task.cpp` `build_exec_table`

**修改**：在多RANK训练时，将通信图加入必检列表：
```cpp
static const GraphSlot kRequiredNCCL[] = {
    GraphSlot::DEEP_ALLREDUCE,
    GraphSlot::FIRST_LAYER_ALLREDUCE,
};
if (K > 1) {
    for (size_t v = 0; v < 4; ++v) {
        for (int rank = 0; rank < K; ++rank) {
            for (auto slot : kRequiredNCCL) {
                TR_CHECK(gpu_exec_.variant_graphs[v][rank][S(slot)],
                         RuntimeError,
                         "NCCL graph slot is nullptr for v=" << v
                         << " rank=" << rank);
            }
        }
    }
}
```

---

## 四、验证计划

### 4.1 编译与日志验证
1. Windows本地增量编译通过，无新增warning
2. 运行训练，确认 `[B3-NCCL]` 日志数量从 ~10+ 降到 **3~4**
3. 确认 `capture_nccl_graph_coordinated` 中无 "capture期间create event" 的违规操作

### 4.2 功能验证
1. **`test_mean_allreduce`**：2/4/8 RANK 均通过（FP32 + AMP）
2. **`test_mean_allreduce` 前后值对比**：在 `launch_allreduce_cuda_impl` 前后打印梯度sum，确认AllReduce实际生效

### 4.3 端到端验证
| 测试项 | 预期结果 |
|--------|---------|
| 单RANK回归 | 保持 99.41% |
| 2RANK | 回落至 99.40%~99.50% 正常区间（不再异常偏高） |
| 4RANK | 保持 99.40%~99.50% |
| 8RANK | **恢复到 99.40%~99.50%**，与单RANK基本一致 |
| 收敛曲线 | 1/2/4/8 RANK 的验证曲线应基本重合 |

### 4.4 诊断环境（如问题持续）
```bash
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
```

---

## 五、总结

### 根因

`capture_nccl_graph_coordinated()` 的事件管理模式与框架内已验证正确的 `capture_cuda.cpp` 存在**三处关键差异**：
1. 缺少 `cudaDeviceSynchronize()` 在 `BeginCapture` 之前
2. `cudaEventCreate`/`cudaEventDestroy` 在 `BeginCapture` 之后执行（违反CUDA规范）
3. `cleanup_all_events()` 在 `EndCapture` 之前执行

这三个差异在单次调用时（如单元测试）可能不会触发问题，但 `has_nccl_ops()` 的误判导致该函数被调用 **10+ 次**（实际只需3~4次），违规操作在8RANK环境下累积，最终导致CUDA Graph中的事件节点引用失效，NCCL AllReduce被静默跳过。

### 修复优先级

| 优先级 | Fix | 影响 |
|--------|-----|------|
| **P0** | Fix 1: `has_nccl_ops()` 粒度修正 | 减少协调捕获调用次数，消除问题放大器 |
| **P0** | Fix 2: 事件管理对齐 `capture_cuda.cpp` | 根本修复：消除CUDA规范违规 |
| P1 | Fix 3: `compile_capture_simple` 一致性 | 预防未来隐患 |
| P1 | Fix 4: 异常安全机制 | 工程健壮性 |
| P2 | Fix 5~7: 流映射、BN注册、断言 | 防御性修复 |

### 建议执行顺序

1. 先同时实施 **Fix 1 + Fix 2**（改动集中在 `captured_graph.cpp`）
2. Windows编译验证
3. A100×8端到端测试 `mnist_best --amp --activation hardswish`
4. 若准确率恢复至 99.4%+，确认根因；若未恢复，继续实施 Fix 3~7
