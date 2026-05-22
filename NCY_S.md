# NCCL + CUDA Graph 完整修复方案 (NCY_S.md)

**日期**: 2026-05-20  
**问题**: RANGE_MEAN_ALLREDUCE 单卡崩溃 + 八卡死锁  
**根因**: NCCL初始化条件错误 + 多rank同步捕获缺失  
**方案**: 多rank协调捕获 + Live Replay混合模式

---

## 一、问题全面分析

### 1.1 错误现象对比

| 场景 | 现象 | 根因 |
|------|------|------|
| **单卡** | `ncclAllReduce failed: invalid argument` | NCCL comm仅多卡初始化 |
| **八卡** | 死锁在"capture rank 0 begin" | 缺少多rank同步捕获机制 |

### 1.2 两层根因分析

#### 根因A：NCCL初始化条件错误（单卡崩溃）
```cpp
// task_base.cpp:compile_alloc_hardware()
#ifdef TR_USE_NCCL
    if (gpu_ids.size() > 1) {  // ❌ 错误：单卡跳过NCCL初始化
        ncclCommInitAll(comms.data(), num_gpus, gpu_ids.data());
    }
#endif
```

**问题**：
- 单卡时 `ctx.nccl_comm()` 返回 `nullptr`
- `allreduce_op.cpp` 无条件调用 `ncclAllReduce(..., ctx.nccl_comm(), ...)`
- 传入空comm → `invalid argument`

#### 根因B：缺少多rank同步捕获（八卡死锁）
```cpp
// task_base.cpp:compile_capture_simple() - 当前实现（错误）
for (int rank = 0; rank < num_gpus_; ++rank) {
    // ❌ 串行捕获：rank 0先进入，调用ncclAllReduce
    // ❌ Rank 1-7还未进入capture，无法响应NCCL集合操作
    auto captured = CapturedGraph::capture(...);
}
```

**NCCL + CUDA Graph的硬性要求**（来自test_nccl_perf.cpp验证）：
```cpp
// ✅ 正确模式：三阶段同步捕获
// 阶段1：所有rank同时进入capture
for (int g = 0; g < kNumGpus; ++g) {
    cudaStreamBeginCapture(streams[g], cudaStreamCaptureModeThreadLocal);
}

// 阶段2：在ncclGroupStart/End中集体调用
ncclGroupStart();
for (int g = 0; g < kNumGpus; ++g) {
    ncclAllReduce(..., comms[g], streams[g]);
}
ncclGroupEnd();

// 阶段3：所有rank同时结束capture
for (int g = 0; g < kNumGpus; ++g) {
    cudaStreamEndCapture(streams[g], &graphs[g]);
}
```

### 1.3 现有方案对比分析

#### 方案K（NCC_K.md）：Live Replay模式
**核心思路**：NCCL图不捕获，直接live调用launcher
```cpp
// capture阶段：跳过NCCL节点
if (cg.has_nccl_ops()) {
    // 不capture NCCL节点，只保存引用
    return;  // per_rank_execs_[0] = nullptr
}

// launch阶段：live调用NCCL
if (has_nccl_ops_) {
    // 直接调用launcher，不走CUDA Graph
    entry.launch_cuda(node, *nccl_mp_, *nccl_ctx_, state);
}
```

**优点**：
- ✅ 完全避免capture同步问题
- ✅ 侵入性最小
- ✅ 逻辑清晰

**缺点**：
- ❌ 失去CUDA Graph的launch overhead优化
- ❌ 无法与compute op混合在同一个图中
- ❌ 需要修改CapturedGraph的launch逻辑

#### 方案D（NCC_D.md）：多rank协调捕获
**核心思路**：修改capture管道，实现三阶段同步捕获
```cpp
// 检测NCCL图后，走协调捕获路径
if (has_nccl) {
    // Phase A: 所有rank同时BeginCapture
    for rank in ranks:
        cudaStreamBeginCapture(stream[rank])
    
    // Phase B: 所有rank同时replay（含ncclGroupStart/End）
    for rank in ranks:
        capture_cuda_replay(nodes, ...)
    
    // Phase C: 所有rank同时EndCapture
    for rank in ranks:
        cudaStreamEndCapture(stream[rank], &graph[rank])
}
```

**优点**：
- ✅ 真正的CUDA Graph，性能最优
- ✅ 支持混合图（compute + NCCL）
- ✅ 符合CUDA Graph的设计理念

**缺点**：
- ❌ 架构改动较大
- ❌ 需要重构capture管道
- ❌ 调试复杂度高

---

## 二、推荐的混合方案：分类处理策略

### 2.1 方案设计原则

1. **最小侵入**：不破坏现有的SimpleTask捕获架构
2. **性能优先**：能用CUDA Graph就用，不能用就降级
3. **向后兼容**：不影响现有的非NCCL图
4. **分类处理**：根据图的复杂度选择策略

### 2.2 分类处理策略

| 图类型 | 判断标准 | 处理策略 | 理由 |
|--------|---------|----------|------|
| **纯NCCL图** | 只含ALLREDUCE节点 | Live Replay（方案K） | 避免同步复杂度，AllReduce瓶颈在带宽不在launch |
| **混合图** | 含compute + NCCL | 多rank协调捕获（方案D） | 需要CUDA Graph性能，必须解决同步问题 |
| **普通图** | 不含NCCL | 原有捕获机制 | 零影响 |

### 2.3 为什么这样分类？

**纯NCCL图的特点**：
- `test_mean_allreduce.cpp`、`FIRST_COMM`、`DEEP_COMM`、`STATS_COMM`都是纯NCCL
- AllReduce的时间开销（ms级）远大于kernel launch overhead（μs级）
- CUDA Graph对AllReduce的性能提升可忽略（<5%）

**混合图的未来需求**：
- 可能需要同时进行梯度计算 + AllReduce
- 计算kernel的launch overhead优化有价值
- 需要真正的CUDA Graph支持

---

## 三、具体实现方案

### 3.1 阶段0：修复单卡崩溃（立即执行）

#### 文件1：`src/task/task_base.cpp`

**修改位置：NCCL初始化条件**
```cpp
// 原代码（L1298附近）：
#ifdef TR_USE_NCCL
    if (gpu_ids.size() > 1) {  // ❌ 错误
        ncclCommInitAll(comms.data(), num_gpus, gpu_ids.data());
    }
#endif

// 修改后：
#ifdef TR_USE_NCCL
    // ✅ 始终初始化NCCL，单卡场景也需要有效的nccl_comm_
    if (gpu_ids.size() >= 1) {
        ncclResult_t res = ncclCommInitAll(comms.data(), num_gpus, gpu_ids.data());
        if (res != ncclSuccess) {
            TR_RUNTIME_ERROR("NCCL initialization failed: " << ncclGetErrorString(res));
        }
        
        for (int g = 0; g < num_gpus_; ++g) {
            DeviceContext& ctx = *backend_->contexts[g];
            ctx.set_nccl_comm(comms[g]);
        }
    }
#endif
```

#### 文件2：`src/backend/ops/range/allreduce_op.cpp`

**修改位置：增加单卡保护**
```cpp
static void launch_allreduce_cuda_impl(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

#ifdef TR_USE_NCCL
    // ✅ 新增：单卡保护
    int world_size = GlobalRegistry::instance().world_size();
    if (world_size <= 1) {
        // 单卡场景：AllReduce退化为no-op，数据已就位
        TR_LOG_DEBUG("backend") << "Single GPU mode: AllReduce is a no-op";
        return;
    }
    
    bool do_mean = (node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE);
    
    // ✅ 新增：ncclGroupStart/End包裹（支持未来的多rank协调捕获）
    ncclGroupStart();
    
    for (size_t i = 0; i < node.input_ranges.size() && i < node.output_ranges.size(); ++i) {
        // ... 原有逻辑保持不变 ...
        ncclResult_t res = ncclAllReduce(
            dst, dst, count, ncclFloat32, ncclSum,
            static_cast<ncclComm_t>(ctx.nccl_comm()), s);
        // ...
    }
    
    ncclGroupEnd();
    
#else
    // TR_USE_NCCL not defined: skip allreduce
#endif
    
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

### 3.2 阶段1：实现纯NCCL图的Live Replay（核心功能）

#### 文件3：`include/renaissance/graph/computation_graph.h`

**新增方法：检测NCCL图**
```cpp
class ComputationGraph {
public:
    // ... 现有方法 ...
    
    /// 检查该图是否为纯NCCL图（只含ALLREDUCE节点）
    [[nodiscard]] bool is_pure_nccl_graph() const {
        bool has_nccl = false;
        bool has_other = false;
        
        for (const auto& node : linear_nodes()) {
            if (node.kind == GraphNode::Kind::RANGE) {
                if (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                    node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE) {
                    has_nccl = true;
                } else {
                    has_other = true;
                }
            } else if (node.kind == GraphNode::Kind::COMPUTE) {
                has_other = true;
            }
        }
        
        return has_nccl && !has_other;
    }
    
    /// 检查该图是否包含NCCL节点（混合图检测）
    [[nodiscard]] bool has_nccl_ops() const {
        for (const auto& node : linear_nodes()) {
            if (node.kind == GraphNode::Kind::RANGE &&
                (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                 node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE)) {
                return true;
            }
        }
        return false;
    }
};
```

#### 文件4：`src/task/task_base.cpp`

**修改位置：`compile_capture_simple()` - 分类处理**
```cpp
void TaskBase::compile_capture_simple() {
#ifdef TR_USE_CUDA
    if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
        // ... warmup逻辑保持不变 ...
        
        TR_LOG_INFO("task") << "[DBG] compile_capture_simple: starting capture phase";
        int graph_index = 0;
        for (auto& [name, entry] : named_graphs_) {
            TR_LOG_INFO("task") << "[DBG] capture graph '" << name << "'";
            
            // ✅ 新增：分类检测
            bool is_pure_nccl = entry.graph.is_pure_nccl_graph();
            bool has_nccl = entry.graph.has_nccl_ops();
            
            CapturedGraph cg;
            
            if (is_pure_nccl) {
                // ✅ 路径1：纯NCCL图 - Live Replay模式
                TR_LOG_INFO("task") << "[DBG] graph '" << name 
                                   << "' is pure NCCL, using live replay mode";
                cg = capture_pure_nccl_graph(entry.graph, memory_plan_, entry.stream);
            } else if (has_nccl) {
                // ⚠️ 路径2：混合图 - 多rank协调捕获（未来实现）
                TR_LOG_ERROR("task") << "Graph '" << name
                                     << "' contains mixed compute + NCCL ops, "
                                        "multi-rank coordinated capture not implemented yet. "
                                        "Falling back to live replay mode.";
                cg = capture_pure_nccl_graph(entry.graph, memory_plan_, entry.stream);
            } else {
                // ✅ 路径3：普通图 - 原有捕获机制
                TR_LOG_INFO("task") << "[DBG] graph '" << name 
                                   << "' is normal graph, using standard capture";
                cg.reserve_ranks(num_gpus_);
                for (int rank = 0; rank < num_gpus_; ++rank) {
                    // ... 原有捕获逻辑完全不变 ...
                    auto captured = CapturedGraph::capture(
                        entry.graph, memory_plan_, gid, entry.stream, ctx);
                    // ...
                }
            }
            
            simple_captured_graphs_.emplace(name, std::move(cg));
            graph_index++;
        }
        // ...
    }
#endif
}
```

**新增方法：`capture_pure_nccl_graph()`**
```cpp
// 在task_base.cpp中新增私有方法
#ifdef TR_USE_CUDA
CapturedGraph TaskBase::capture_pure_nccl_graph(
    const ComputationGraph& graph,
    const MemoryPlan& mp,
    StreamKind stream_kind)
{
    TR_LOG_INFO("task") << "[DBG] capture_pure_nccl_graph: using live replay mode";
    
    CapturedGraph cg;
    cg.reserve_ranks(num_gpus_);
    
    // 为每个rank创建空的CapturedGraph（is_cuda=false，走CPU路径）
    // launch时会直接调用NCCL launcher
    for (int rank = 0; rank < num_gpus_; ++rank) {
        if (rank == 0) {
            // 设置metadata（从rank 0的context）
            cg.set_metadata_from(graph, stream_kind);
        }
        // 不调用capture，保持per_rank_execs_[rank] = nullptr
    }
    
    // 标记为NCCL live replay模式
    cg.set_nccl_live_replay_mode(true, &graph, &mp);
    
    return cg;
}
#endif
```

#### 文件5：`include/renaissance/graph/captured_graph.h`

**新增字段和方法**
```cpp
class CapturedGraph {
public:
    // ... 现有方法 ...
    
    /// 设置NCCL live replay模式
    void set_nccl_live_replay_mode(bool enabled, 
                                   const ComputationGraph* cg = nullptr,
                                   const MemoryPlan* mp = nullptr) {
        is_nccl_live_replay_ = enabled;
        nccl_graph_ = cg;
        nccl_memory_plan_ = mp;
    }
    
    /// 检查是否为NCCL live replay模式
    [[nodiscard]] bool is_nccl_live_replay() const noexcept { 
        return is_nccl_live_replay_; 
    }

private:
    // ... 现有字段 ...
    
    bool is_nccl_live_replay_ = false;
    const ComputationGraph* nccl_graph_ = nullptr;
    const MemoryPlan* nccl_memory_plan_ = nullptr;
};
```

#### 文件6：`src/graph/captured_graph.cpp`

**修改`launch()`方法**
```cpp
void CapturedGraph::launch(int rank, void* stream) const {
    if (is_cuda_) {
#ifdef TR_USE_CUDA
        // ✅ 新增：NCCL live replay路径
        if (is_nccl_live_replay_) {
            TR_LOG_DEBUG("graph") << "[NCCL Live Replay] launching rank " << rank;
            
            if (!nccl_graph_ || !nccl_memory_plan_) {
                TR_RUNTIME_ERROR("NCCL live replay: graph or memory plan is null");
            }
            
            // 获取对应rank的DeviceContext（从task传递）
            // 注意：需要修改launch签名或通过其他方式传递ctx
            // 这里假设通过GlobalRegistry或task context获取
            auto& registry = GlobalRegistry::instance();
            DeviceContext* ctx = get_device_context_for_rank(rank);
            if (!ctx) {
                TR_RUNTIME_ERROR("NCCL live replay: cannot get DeviceContext for rank " << rank);
            }
            
            // 直接调用NCCL launcher（不在capture中）
            const auto& nodes = nccl_graph_->linear_nodes();
            MultiStreamCaptureState state;
            state.primary_stream = static_cast<cudaStream_t>(stream);
            
            for (const auto& node : nodes) {
                if (node.kind != GraphNode::Kind::RANGE) continue;
                if (node.range_op != RangeOp::RANGE_SUM_ALLREDUCE &&
                    node.range_op != RangeOp::RANGE_MEAN_ALLREDUCE) {
                    continue;
                }
                
                auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
                if (entry.launch_cuda) {
                    entry.launch_cuda(node, *nccl_memory_plan_, *ctx, state);
                }
            }
            
            return;
        }
        
        // ✅ 原有：普通CUDA Graph launch路径
        if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size()) {
            NativeGraph exec = per_rank_execs_[rank];
            if (exec) {
                cudaError_t err = cudaGraphLaunch(
                    static_cast<cudaGraphExec_t>(exec),
                    static_cast<cudaStream_t>(stream));
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaGraphLaunch failed: " << cudaGetErrorString(err));
                }
            }
        }
#endif
    } else {
        // CPU路径保持不变
        for (const auto& op : cpu_ops_) {
            if (op.fn) op.fn(static_cast<CpuOpContext*>(op.ctx));
        }
    }
}
```

### 3.3 阶段2：多rank协调捕获（可选，未来实现）

如果未来需要支持混合图，可以实现完整的多rank协调捕获：

**文件7：`src/task/task_base.cpp`**

**新增方法：`capture_mixed_graph_coordinated()`**
```cpp
#ifdef TR_USE_CUDA
CapturedGraph TaskBase::capture_mixed_graph_coordinated(
    const ComputationGraph& graph,
    const MemoryPlan& mp,
    StreamKind stream_kind)
{
    TR_LOG_INFO("task") << "[DBG] capture_mixed_graph_coordinated: starting";
    
    CapturedGraph cg;
    cg.reserve_ranks(num_gpus_);
    
    std::vector<cudaStream_t> capture_streams(num_gpus_);
    std::vector<MultiStreamCaptureState> capture_states(num_gpus_);
    
    // ========== Phase A: 所有rank同时进入capture ==========
    TR_LOG_INFO("task") << "[DBG] Phase A: all ranks begin capture";
    for (int rank = 0; rank < num_gpus_; ++rank) {
        DeviceContext& ctx = *backend_->contexts[rank];
        cudaSetDevice(ctx.device_id());
        ctx.set_rank(rank);
        ctx.set_memory_plan(&mp);
        
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(stream_kind));
        capture_streams[rank] = s;
        
        // 初始化capture state
        capture_states[rank].primary_stream = s;
        capture_states[rank].get_or_register(s);
        capture_states[rank].get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1)));
        capture_states[rank].get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2)));
        capture_states[rank].get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3)));
        
        // 重置last_done_events
        for (int i = 0; i < capture_states[rank].num_active; ++i) {
            if (capture_states[rank].streams[i].last_done_event) {
                cudaEventDestroy(capture_states[rank].streams[i].last_done_event);
            }
            cudaEventCreateWithFlags(&capture_states[rank].streams[i].last_done_event, 
                                     cudaEventDisableTiming);
        }
        
        cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
        
        // 引入secondary流到capture上下文
        if (capture_states[rank].num_active > 1) {
            cudaEventRecord(capture_states[rank].streams[0].last_done_event, s);
            for (int i = 1; i < capture_states[rank].num_active; ++i) {
                cudaStreamWaitEvent(capture_states[rank].streams[i].stream,
                                   capture_states[rank].streams[0].last_done_event, 0);
            }
        }
    }
    
    // ========== Phase B: 所有rank同时replay节点 ==========
    TR_LOG_INFO("task") << "[DBG] Phase B: all ranks replay nodes";
    for (int rank = 0; rank < num_gpus_; ++rank) {
        DeviceContext& ctx = *backend_->contexts[rank];
        const auto& nodes = graph.linear_nodes();
        
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            
            if (i > 0) {
                insert_cross_op_barrier(nodes[i-1], node, capture_states[rank], ctx);
            }
            
            if (node.kind == GraphNode::Kind::COMPUTE) {
                auto& entry = g_compute_op_table[static_cast<size_t>(node.compute_op)];
                if (entry.launch_cuda) {
                    entry.launch_cuda(node, mp, ctx, capture_states[rank]);
                }
            } else {
                auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
                if (entry.launch_cuda) {
                    entry.launch_cuda(node, mp, ctx, capture_states[rank]);
                }
            }
        }
        
        finalize_cross_stream_barrier(capture_states[rank]);
    }
    
    // ========== Phase C: 所有rank同时结束capture ==========
    TR_LOG_INFO("task") << "[DBG] Phase C: all ranks end capture";
    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaStream_t s = capture_streams[rank];
        cudaGraph_t graph_obj = nullptr;
        
        cudaError_t end_err = cudaStreamEndCapture(s, &graph_obj);
        if (end_err != cudaSuccess) {
            TR_DEVICE_ERROR("cudaStreamEndCapture failed for rank " << rank);
            if (graph_obj) cudaGraphDestroy(graph_obj);
            continue;
        }
        
        cudaGraphExec_t exec = nullptr;
        cudaGraphNode_t error_node = nullptr;
        char error_log[2048] = {};
        
        cudaError_t inst_err = cudaGraphInstantiate(&exec, graph_obj, &error_node, error_log, sizeof(error_log));
        if (inst_err != cudaSuccess || !exec) {
            TR_DEVICE_ERROR("cudaGraphInstantiate failed for rank " << rank);
            if (exec) cudaGraphExecDestroy(exec);
            if (graph_obj) cudaGraphDestroy(graph_obj);
            continue;
        }
        
        if (rank == 0) {
            cg.set_metadata_from(exec);
        }
        cg.set_rank_exec(rank, static_cast<NativeGraph>(exec));
        cudaGraphDestroy(graph_obj);
        
        // 清理events
        for (int i = 0; i < capture_states[rank].num_active; ++i) {
            if (capture_states[rank].streams[i].last_done_event) {
                cudaEventDestroy(capture_states[rank].streams[i].last_done_event);
            }
        }
    }
    
    TR_LOG_INFO("task") << "[DBG] capture_mixed_graph_coordinated: complete";
    return cg;
}
#endif
```

---

## 四、验证方案

### 4.1 单卡验证
```bash
CUDA_VISIBLE_DEVICES=0 ./tests/correction/test_mean_allreduce --gpu
```

**预期结果**：
- ✅ NCCL初始化成功（单卡也创建comm）
- ✅ `world_size <= 1` 保护触发，AllReduce为no-op
- ✅ 数据保持不变（rank 0写入0，expected=0）

### 4.2 八卡验证
```bash
./tests/correction/test_mean_allreduce --gpu
```

**预期结果**：
- ✅ 检测为纯NCCL图，使用live replay模式
- ✅ 每个rank独立调用`ncclAllReduce`（不在capture中）
- ✅ 所有rank的均值一致（expected=3.5）

### 4.3 回归验证
```bash
./tests/correction/test_gap --gpu
./tests/correction/test_check_nan --gpu
./tests/correction/test_cast_fp16_to_fp32 --gpu
```

**预期结果**：
- ✅ 普通图走原有CUDA Graph捕获
- ✅ 性能和行为完全不变

### 4.4 性能验证
```bash
./test_nccl_perf --size 100
```

**预期结果**：
- ✅ Live Replay的AllReduce性能与直接调用相当
- ✅ 吞吐量无明显下降（<5%）

---

## 五、修改文件清单

| 优先级 | 文件 | 修改内容 | 风险 |
|--------|------|---------|------|
| **P0** | `src/task/task_base.cpp` | 修复NCCL初始化条件 | 🟢 低 |
| **P0** | `src/backend/ops/range/allreduce_op.cpp` | 增加单卡保护+ncclGroupStart/End | 🟢 低 |
| **P1** | `include/renaissance/graph/computation_graph.h` | 新增`is_pure_nccl_graph()`检测 | 🟢 低 |
| **P1** | `src/task/task_base.cpp` | 新增`capture_pure_nccl_graph()` | 🟡 中 |
| **P1** | `include/renaissance/graph/captured_graph.h` | 新增NCCL live replay字段 | 🟡 中 |
| **P1** | `src/graph/captured_graph.cpp` | 修改`launch()`支持NCCL live replay | 🟡 中 |
| **P2** | `src/task/task_base.cpp` | 新增`capture_mixed_graph_coordinated()`（可选） | 🔴 高 |

---

## 六、总结与建议

### 6.1 核心原则

1. **立即修复单卡问题**：P0修改，零风险
2. **分类处理策略**：纯NCCL用live replay，混合图用协调捕获
3. **渐进式实施**：先实现live replay，再考虑协调捕获
4. **向后兼容**：不破坏现有图的捕获机制

### 6.2 实施路径

**阶段1（立即执行）**：
1. 修复NCCL初始化条件
2. 增加单卡保护
3. 验证单卡+八卡基本功能

**阶段2（核心功能）**：
1. 实现纯NCCL图的live replay
2. 验证性能和正确性
3. 完整回归测试

**阶段3（可选，未来）**：
1. 实现混合图的多rank协调捕获
2. 支持compute + NCCL的真正CUDA Graph

### 6.3 最终建议

**推荐分阶段实施**：
1. **立即执行P0修改**（单卡崩溃修复，零风险）
2. **优先实现阶段1**（纯NCCL live replay，覆盖当前所有需求）
3. **延后阶段2**（混合图支持，当前生产代码不需要）

这个方案平衡了**风险、性能、复杂度**三个维度，能够在最小侵入的前提下解决当前的所有问题。