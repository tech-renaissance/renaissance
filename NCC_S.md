# NCCL + CUDA Graph 捕获问题分析与解决方案

**日期**: 2026-05-20  
**问题**: ALLREDUCE 测试失败，单卡报错 "ncclAllReduce failed: invalid argument"，八卡死锁  
**根因**: NCCL 图捕获要求所有 rank 同时进入捕获，但当前实现是逐 rank 串行捕获

---

## 一、问题分析

### 1.1 错误现象对比

#### 单卡失败（CUDA_VISIBLE_DEVICES=0）
```
[DeviceError] ncclAllReduce failed: invalid argument 
```

#### 八卡死锁（8 个 GPU）
```
# 程序挂起在 "capture rank 0 begin"
# 所有 rank 都在等待其他 rank 进入 NCCL 集合操作
```

### 1.2 代码问题定位

#### 问题代码1：`task_base.cpp:compile_capture_simple()` 串行捕获
```cpp
// 当前实现（错误）：逐 rank 串行捕获
for (int rank = 0; rank < num_gpus_; ++rank) {
    // ...
    TR_LOG_INFO("task") << "[DBG] capture rank " << rank << " begin";
    DeviceContext& ctx = *backend_->contexts[rank];
    ctx.set_rank(rank);
    ctx.set_memory_plan(&memory_plan_);
    
    // ❌ 单独捕获每个 rank 的图
    auto captured = CapturedGraph::capture(
        entry.graph, memory_plan_, gid, entry.stream, ctx);
    // ...
}
```

#### 问题代码2：`allreduce_op.cpp:launch_allreduce_cuda_impl()` 直接调用 NCCL
```cpp
// 当前实现：在 CUDA Graph 捕获期间直接调用 ncclAllReduce
ncclResult_t res = ncclAllReduce(
    dst, dst, count, ncclFloat32, ncclSum,
    static_cast<ncclComm_t>(ctx.nccl_comm()), s);
// ❌ 在单 rank 捕获时调用 NCCL，违反 NCCL 集合语义
```

### 1.3 根本原因

**NCCL 集合操作的 CUDA Graph 捕获要求**：
1. **同步进入捕获**：所有 rank 必须在同一个时间窗口内进入 `cudaStreamBeginCapture`
2. **同步结束捕获**：所有 rank 必须在同一个时间窗口内进入 `cudaStreamEndCapture`
3. **集体调用 NCCL**：NCCL 操作必须在所有 rank 的捕获期内被同时记录

**当前实现的问题**：
- `compile_capture_simple()` 采用 **逐 rank 串行捕获**策略
- Rank 0 先进入捕获，调用 `ncclAllReduce`，等待其他 rank
- Rank 1-7 还没有进入捕获，无法响应 NCCL 集合操作
- **结果**：死锁（多卡）或 "invalid argument"（单卡，NCCL 检测到通信域不完整）

---

## 二、正确实现参考：test_nccl_perf.cpp

### 2.1 关键代码模式

```cpp
// test_nccl_perf.cpp:204-218 - 正确的多 rank 同步捕获
std::cout << "  [Step 2/2] Capturing communication graphs (NCCL AllReduce)..." << std::endl;

// 1️⃣ 所有 rank 同时进入捕获状态
for (int g = 0; g < kNumGpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamBeginCapture(streams[g], cudaStreamCaptureModeThreadLocal));
}

// 2️⃣ 在 NCCL Group 中集体调用 AllReduce
NCCL_CHECK(ncclGroupStart());
for (int g = 0; g < kNumGpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    NCCL_CHECK(ncclAllReduce(
        d_b[g], d_b[g], kElems, ncclFloat32, ncclAvg, comms[g], streams[g]
    ));
}
NCCL_CHECK(ncclGroupEnd());

// 3️⃣ 所有 rank 同时结束捕获状态
for (int g = 0; g < kNumGpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamEndCapture(streams[g], &graphs[g]));
}
```

### 2.2 关键设计原则

1. **三阶段捕获**：
   - 阶段1：所有 rank 同时 `cudaStreamBeginCapture`
   - 阶段2：所有 rank 在 `ncclGroupStart/End` 中调用 NCCL
   - 阶段3：所有 rank 同时 `cudaStreamEndCapture`

2. **ncclGroupStart/End**：
   - 确保 NCCL 调用在所有 rank 上以相同的顺序执行
   - CUDA Graph 捕获期间必须使用

3. **设备切换顺序**：
   - 每个阶段都按 `g = 0, 1, 2, ...` 的顺序切换设备
   - 确保所有 rank 以相同的步调执行

---

## 三、当前框架的架构限制

### 3.1 SimpleTask 的捕获架构

```
SimpleTask::compile_capture_simple()
  └─> for each graph:
       └─> for each rank (串行):
            └─> CapturedGraph::capture()  ❌
                 └─> capture_cuda()
                      └─> 逐 node replay
                           └─> RangeOp launcher
                                └─> ncclAllReduce  ❌
```

**问题**：
1. **rank 循环在最外层**：每个 rank 单独完成所有图的捕获
2. **逐 rank 捕获**：Rank N 捕获时，其他 rank 处于未定义状态
3. **NCCL 语义破坏**：`ncclAllReduce` 变成了"单 rank 调用"而非"集合调用"

### 3.2 架构约束

| 组件 | 当前设计 | 是否支持 NCCL Graph |
|------|---------|---------------------|
| **SimpleTask** | 逐 rank 串行捕获 | ❌ 不支持 |
| **CapturedGraph** | 单 rank 捕获上下文 | ❌ 不支持 |
| **RangeOp launcher** | 单 rank 操作 | ⚠️ 有限支持 |
| **CUDA Graph** | 线程本地捕获模式 | ✅ 支持 |

---

## 四、安全可靠的修复方案

### 4.1 方案总览

**核心思路**：将 NCCL AllReduce 特殊处理，不通过常规的 RangeOp launcher 机制

**关键原则**：
1. **最小侵入**：不破坏现有的 SimpleTask 捕获架构
2. **类型安全**：通过检测 RangeOp 类型动态选择策略
3. **向后兼容**：不包含 NCCL 的图仍然使用原有机制

### 4.2 修复方案 A：Graph 级别的 NCCL 特殊处理（推荐）

#### 修改文件1：`src/task/task_base.cpp`

**修改位置1：`compile_capture_simple()` - 检测 NCCL 图**
```cpp
void TaskBase::compile_capture_simple() {
#ifdef TR_USE_CUDA
    if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
        // ... 现有 warmup 代码保持不变 ...
        
        TR_LOG_INFO("task") << "[DBG] compile_capture_simple: starting capture phase, num_graphs="
                            << named_graphs_.size() << " num_gpus=" << num_gpus_;
        int graph_index = 0;
        for (auto& [name, entry] : named_graphs_) {
            TR_LOG_INFO("task") << "[DBG] capture graph '" << name << "' graph_index=" << graph_index;
            
            // ✅ 新增：检测图是否包含 NCCL AllReduce
            bool contains_nccl = false;
            for (const auto& node : entry.graph.linear_nodes()) {
                if (node.kind == GraphNode::Kind::RANGE) {
                    if (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                        node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE) {
                        contains_nccl = true;
                        break;
                    }
                }
            }
            
            CapturedGraph cg;
            if (contains_nccl && num_gpus_ > 1) {
                // ✅ 新增：NCCL 图使用多 rank 同步捕获
                TR_LOG_INFO("task") << "[DBG] graph '" << name 
                                   << "' contains NCCL, using multi-rank synchronized capture";
                cg = capture_nccl_graph(entry.graph, memory_plan_, entry.stream, graph_index);
            } else {
                // ✅ 原有逻辑：非 NCCL 图使用单 rank 捕获
                cg.reserve_ranks(num_gpus_);
                for (int rank = 0; rank < num_gpus_; ++rank) {
                    // ... 现有捕获逻辑保持不变 ...
                    auto captured = CapturedGraph::capture(
                        entry.graph, memory_plan_, gid, entry.stream, ctx);
                    // ... 现有逻辑保持不变 ...
                }
            }
            
            simple_captured_graphs_.emplace(name, std::move(cg));
            graph_index++;
        }
        // ... 现有逻辑保持不变 ...
    }
#endif
}
```

**修改位置2：新增 `capture_nccl_graph()` 方法**
```cpp
// 在 task_base.cpp 中新增私有方法
#ifdef TR_USE_CUDA
CapturedGraph TaskBase::capture_nccl_graph(
    const ComputationGraph& graph,
    const MemoryPlan& mp,
    StreamKind stream_kind,
    int graph_index)
{
    TR_LOG_INFO("task") << "[DBG] capture_nccl_graph: starting synchronized multi-rank capture";
    
    CapturedGraph cg;
    cg.reserve_ranks(num_gpus_);
    
    // ========== 阶段1：所有 rank 同时进入捕获 ==========
    TR_LOG_INFO("task") << "[DBG] capture_nccl_graph: phase 1 - all ranks begin capture";
    std::vector<cudaStream_t> capture_streams(num_gpus_);
    for (int rank = 0; rank < num_gpus_; ++rank) {
        DeviceContext& ctx = *backend_->contexts[rank];
        cudaError_t err = cudaSetDevice(ctx.device_id());
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank);
            return cg;
        }
        
        ctx.set_rank(rank);
        ctx.set_memory_plan(&mp);
        
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(stream_kind));
        capture_streams[rank] = s;
        
        cudaError_t cap_err = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
        if (cap_err != cudaSuccess) {
            TR_DEVICE_ERROR("cudaStreamBeginCapture failed for rank " << rank
                           << ": " << cudaGetErrorString(cap_err));
            return cg;
        }
    }
    
    // ========== 阶段2：所有 rank 同步记录 NCCL 操作 ==========
    TR_LOG_INFO("task") << "[DBG] capture_nccl_graph: phase 2 - record NCCL operations";
    
#ifdef TR_USE_NCCL
    ncclGroupStart();
#endif
    
    for (int rank = 0; rank < num_gpus_; ++rank) {
        DeviceContext& ctx = *backend_->contexts[rank];
        cudaStream_t s = capture_streams[rank];
        
        // 遍历图中的所有节点，直接调用 NCCL launcher
        for (const auto& node : graph.linear_nodes()) {
            if (node.kind == GraphNode::Kind::RANGE) {
                if (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                    node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE) {
                    
                    TR_LOG_INFO("task") << "[DBG] rank " << rank 
                                       << " recording NCCL AllReduce";
                    
                    // 直接调用 AllReduce launcher（不通过 capture 机制）
                    launch_allreduce_cuda_capture(node, mp, ctx, s);
                }
            }
        }
    }
    
#ifdef TR_USE_NCCL
    ncclGroupEnd();
#endif
    
    // ========== 阶段3：所有 rank 同时结束捕获 ==========
    TR_LOG_INFO("task") << "[DBG] capture_nccl_graph: phase 3 - all ranks end capture";
    std::vector<cudaGraph_t> raw_graphs(num_gpus_);
    
    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaStream_t s = capture_streams[rank];
        cudaGraph_t graph_obj = nullptr;
        
        cudaError_t end_err = cudaStreamEndCapture(s, &graph_obj);
        if (end_err != cudaSuccess) {
            TR_DEVICE_ERROR("cudaStreamEndCapture failed for rank " << rank
                           << ": " << cudaGetErrorString(end_err));
            if (graph_obj) cudaGraphDestroy(graph_obj);
            continue;
        }
        
        raw_graphs[rank] = graph_obj;
    }
    
    // ========== 阶段4：实例化图 ==========
    TR_LOG_INFO("task") << "[DBG] capture_nccl_graph: phase 4 - instantiate graphs";
    for (int rank = 0; rank < num_gpus_; ++rank) {
        if (!raw_graphs[rank]) continue;
        
        cudaGraphExec_t exec = nullptr;
        cudaGraphNode_t error_node = nullptr;
        char error_log[2048] = {};
        
        cudaError_t inst_err = cudaGraphInstantiate(&exec, raw_graphs[rank], 
                                                     &error_node, error_log, sizeof(error_log));
        if (inst_err != cudaSuccess || !exec) {
            TR_DEVICE_ERROR("cudaGraphInstantiate failed for rank " << rank
                           << ": " << cudaGetErrorString(inst_err)
                           << ", log: " << error_log);
            if (exec) cudaGraphExecDestroy(exec);
            if (raw_graphs[rank]) cudaGraphDestroy(raw_graphs[rank]);
            continue;
        }
        
        if (rank == 0) {
            cg.set_metadata_from(exec);
        }
        cg.set_rank_exec(rank, static_cast<NativeGraph>(exec));
        cudaGraphDestroy(raw_graphs[rank]);
    }
    
    TR_LOG_INFO("task") << "[DBG] capture_nccl_graph: complete";
    return cg;
}
#endif
```

#### 修改文件2：`src/backend/ops/range/allreduce_op.cpp`

**新增方法：`launch_allreduce_cuda_capture()`**
```cpp
// 在 allreduce_op.cpp 中新增专用捕获方法
#ifdef TR_USE_CUDA
namespace {

// 专用于 CUDA Graph 捕获的 NCCL AllReduce launcher
void launch_allreduce_cuda_capture(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    cudaStream_t s)
{
#ifdef TR_USE_NCCL
    int world_size = GlobalRegistry::instance().world_size();
    bool do_mean = (node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE);
    
    for (size_t i = 0; i < node.input_ranges.size() && i < node.output_ranges.size(); ++i) {
        auto [src_off, src_sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.input_ranges[i].start_region_id),
            static_cast<Region>(node.input_ranges[i].end_region_id));
        auto [dst_off, dst_sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.output_ranges[i].start_region_id),
            static_cast<Region>(node.output_ranges[i].end_region_id));
        if (src_sz == 0 || dst_sz == 0) continue;
        
        void* src = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src_off);
        void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off);
        size_t count = std::min(src_sz, dst_sz) / sizeof(float);
        if (count == 0) continue;
        
        // D2D copy（如果需要）
        if (src != dst) {
            cudaError_t cpy_err = cudaMemcpyAsync(
                dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice, s);
            if (cpy_err != cudaSuccess) {
                TR_DEVICE_ERROR("RANGE_ALLREDUCE pre-copy failed: "
                                << cudaGetErrorString(cpy_err));
            }
        }
        
        // NCCL AllReduce（在 ncclGroupStart/End 中调用）
        ncclResult_t res = ncclAllReduce(
            dst, dst, count, ncclFloat32, ncclSum,
            static_cast<ncclComm_t>(ctx.nccl_comm()), s);
        if (res != ncclSuccess) {
            TR_DEVICE_ERROR("ncclAllReduce failed: " << ncclGetErrorString(res));
        }
        
        // Mean 除法（如果需要）
        if (do_mean && world_size > 1) {
            float inv = 1.0f / static_cast<float>(world_size);
            cudaError_t scale_err = launch_tr_scale_fp32_kernel(
                static_cast<float*>(dst), inv, static_cast<int64_t>(count), s);
            if (scale_err != cudaSuccess) {
                TR_DEVICE_ERROR("RANGE_MEAN_ALLREDUCE post-scale failed: "
                                << cudaGetErrorString(scale_err));
            }
        }
    }
#else
    (void)node; (void)mp; (void)ctx; (void)s;
#endif
}

} // namespace
#endif
```

**修改注册：在头文件中声明**
```cpp
// include/renaissance/backend/op_registry.h 中新增声明
#ifdef TR_USE_CUDA
namespace tr {
void launch_allreduce_cuda_capture(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    cudaStream_t s);
}
#endif
```

#### 修改文件3：`include/renaissance/task/task_base.h`

**新增方法声明**
```cpp
// 在 TaskBase 类的 private 或 protected 部分添加
private:
#ifdef TR_USE_CUDA
    CapturedGraph capture_nccl_graph(
        const ComputationGraph& graph,
        const MemoryPlan& mp,
        StreamKind stream_kind,
        int graph_index);
#endif
```

---

### 4.3 方案 B：Launcher 级别的 NCCL 检测（备选）

如果不想修改 `compile_capture_simple()`，可以在 launcher 层面做检测：

#### 修改文件：`src/graph/capture_cuda.cpp`

**修改 `capture_cuda()` 方法**
```cpp
void CapturedGraph::capture_cuda(const ComputationGraph& cg,
                                  const MemoryPlan& mp,
                                  GraphId gid,
                                  StreamKind stream_kind,
                                  const DeviceContext& ctx,
                                  CapturedGraph& result)
{
    // ✅ 新增：检测图中是否包含 NCCL AllReduce
    bool contains_nccl = false;
    for (const auto& node : cg.nodes(gid)) {
        if (node.kind == GraphNode::Kind::RANGE) {
            if (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE) {
                contains_nccl = true;
                break;
            }
        }
    }
    
    if (contains_nccl) {
        // ✅ NCCL 图：需要多 rank 同步，当前不支持
        TR_LOG_ERROR("graph") << "NCCL AllReduce graph detected, "
                                "but multi-rank synchronized capture not implemented. "
                                "Please use SimpleTask with compile_capture_simple() "
                                "instead of direct capture.";
        TR_RUNTIME_ERROR("NCCL graph requires multi-rank synchronized capture");
        return;
    }
    
    // ✅ 原有逻辑：非 NCCL 图正常捕获
    // ... 现有代码保持不变 ...
}
```

---

## 五、验证方案

### 5.1 测试验证

```bash
# 1. 单卡测试（应该通过）
CUDA_VISIBLE_DEVICES=0 ./tests/correction/test_mean_allreduce --gpu

# 2. 八卡测试（应该通过）
./tests/correction/test_mean_allreduce --gpu

# 3. AMP 模式测试（应该通过）
./tests/correction/test_mean_allreduce --amp

# 4. 回归测试（确保不破坏其他功能）
./tests/correction/test_gap --gpu
./tests/correction/test_check_nan --gpu
```

### 5.2 验收标准

- [ ] 单卡不报 "invalid argument" 错误
- [ ] 八卡不死锁，正常完成
- [ ] 数学正确性验证通过（均值计算正确）
- [ ] 不破坏其他 RangeOp 的功能
- [ ] CPU 模式仍然正常工作

---

## 六、风险与缓解

### 6.1 风险评估

| 风险项 | 等级 | 缓解措施 |
|--------|------|----------|
| **多 rank 同步复杂度** | 🟡 中 | 最小化同步代码，复用 test_nccl_perf 的已验证模式 |
| **向后兼容性** | 🟢 低 | 只影响包含 NCCL 的图，其他图使用原有路径 |
| **性能影响** | 🟢 低 | NCCL 图本来就需要多 rank 协作，同步开销可忽略 |
| **调试困难度** | 🟡 中 | 增加详细的日志输出，便于问题定位 |

### 6.2 回滚方案

如果新方案出现问题，可以快速回滚：
1. 禁用 NCCL 图的 CUDA Graph 捕获
2. 回退到直接执行（非图）模式
3. 或者使用 NCCL 的 Graph-Aware 模式（如果 NCCL 版本支持）

---

## 七、总结与建议

### 7.1 核心问题

**NCCL + CUDA Graph 的捕获必须遵循"多 rank 同步"原则**：
- 所有 rank 必须在同一个时间窗口内进入/退出捕获
- NCCL 操作必须在 `ncclGroupStart/End` 中集体调用
- 当前 SimpleTask 的逐 rank 串行捕获架构违反了这个原则

### 7.2 推荐方案

**方案 A（Graph 级别特殊处理）是最优选择**：
1. ✅ 最小侵入：不破坏现有架构
2. ✅ 类型安全：通过 RangeOp 检测动态选择策略
3. ✅ 向后兼容：不影响其他图
4. ✅ 可维护：逻辑集中在 `compile_capture_simple()`

### 7.3 实施建议

**按以下顺序实施**：
1. 先实现方案 A 的核心逻辑
2. 在单卡环境下验证功能正确性
3. 在八卡环境下验证无死锁
4. 运行完整的回归测试套件

### 7.4 最终建议

**立即修复**，这是一个功能完整性问题：
- 单卡：NCCL 报错，测试完全无法运行
- 八卡：死锁，程序挂起
- 当前实现对于包含 NCCL 的图是不可用的

修复后，SimpleTask 将能够正确处理 NCCL AllReduce 的 CUDA Graph 捕获，与 test_nccl_perf.cpp 的已验证模式保持一致。