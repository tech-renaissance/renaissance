# NCCL AllReduce CUDA Graph 捕获 —— 最终修复方案

> **输入材料**：NCY_D.md · NCY_K.md · NCY_S.md · test_nccl_perf.cpp · errors.txt
>
> **作者**：综合分析三份 NCY 方案 + 逐行验证实际代码后的产出
>
> 2026-05-20

---

## 一、问题确认

### 1.1 单卡：`invalid argument`

`errors.txt` L29 — `allreduce_op.cpp :: launch_allreduce_cuda_impl()`

**唯一根因**：[task_base.cpp L1298](file:///r:/renaissance/src/task/task_base.cpp)：`if (gpu_ids.size() > 1)` — 单卡 NCCL comm = nullptr。

### 1.2 多卡：死锁

`errors.txt` L85-86 — 挂起在 `capture rank 0 begin`

**唯一根因**：[compile_capture_simple()](file:///r:/renaissance/src/task/task_base.cpp#L308-L357)：逐 rank 串行 capture。rank 0 单独 `BeginCapture` → `ncclAllReduce`，rank 1-7 未进 capture。

---

## 二、三份 NCY 方案核心差异与评判

| | NCY_D | NCY_K | NCY_S |
|---|---|---|---|
| **核心路线** | Live Replay in SimpleTask | Live Replay in CapturedGraph::launch | 分类处理（纯 NCCL→Live、混合→协调 capture） |
| **Launcher 改动** | 仅 world_size 保护 | 仅 world_size 保护 | +ncclGroupStart/End |
| **NCY_K 的两个致命问题** | 已指出：DeviceContext 多 rank 问题 + `launch()` 未被 GPU 路径调用 | — | 继承 NCY_K 同样问题 |
| **NCY_S 的额外问题** | — | — | `is_pure_nccl_graph()` 判定器不必要（当前无混合图）; `capture_nccl_graph` 绕过 `CapturedGraph` 标准接口 |

### 关键发现

`SimpleTask::run_iter` GPU 路径（[simple_task.h L152-186](file:///r:/renaissance/include/renaissance/task/simple_task.h#L152-L186)）**直接调用 `cudaGraphLaunch(exec, stream)`**，不经过 `CapturedGraph::launch()`。因此 NCY_K / NCY_S 在 `CapturedGraph::launch` 中做 live replay 的方案 **不会生效** — GPU 路径根本不会走到那个方法。

### 决策

**采用 NCY_D 的 "Live Replay in SimpleTask" 路线，并融合 NCY_K 的 CapturedGraph 存储设计**。在此基础上做一处关键优化：**统一 run_iter 的 launch 通道** — 将 `cudaGraphLaunch` 调用改为 `cg.launch()`，使普通图和 NCCL 图共享同一入口，live replay 在 `launch()` 内部完成，从而不污染 SimpleTask 模板代码。

---

## 三、最终方案

### 3.1 战略：Live Replay（NCCL 不走 CUDA Graph）

**为什么不做 CUDA Graph**：
- AllReduce 瓶颈是通信带宽（~1.4ms / 100MB），CUDA Graph 省下的 launch overhead（~100µs）占比 <7%
- 跨 rank 协调捕获需要拆解 `capture_cuda` 并重新编排 `compile_capture_simple`，风险远超收益
- `test_nccl_perf.cpp` 的成功模式（多 rank 同步 + ncclGroupStart/End）不兼容当前 per-rank 串行 capture 框架

### 3.2 架构

```
┌─ compile ──────────────────────────────────────────────────┐
│  compile_capture_simple()                                   │
│    ├─ 普通图 → per-rank CapturedGraph::capture (不变)       │
│    └─ NCCL 图 → 存储 LiveReplayContext，skip capture        │
└─────────────────────────────────────────────────────────────┘

┌─ run_iter ─────────────────────────────────────────────────┐
│  for each rank (parallel threads):                         │
│    for each iteration:                                     │
│      cg.launch(rank, stream)  ← 统一入口                   │
│        ├─ 普通图 → cudaGraphLaunch(exec, stream)           │
│        └─ NCCL 图 → 直接调 RangeOp launcher (live replay)  │
│      cudaStreamSynchronize(stream)                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 不需要 `ncclGroupStart/End`

Live replay 不在 CUDA Graph capture 中执行，NCCL AllReduce 作为普通运行时调用，NCCL 内部自行管理跨 rank 同步，无需外部 group。

---

## 四、逐文件修改明细

### 修改 1：[src/task/task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp) — NCCL 初始化条件

```cpp
// BEFORE
if (gpu_ids.size() > 1) { ncclCommInitAll(...); }

// AFTER
if (gpu_ids.size() >= 1) { ncclCommInitAll(...); }
```

**理由**：三方共识。`ncclCommInitAll` 支持 nranks=1，单卡也需要有效的 comm。

---

### 修改 2：[src/backend/ops/range/allreduce_op.cpp](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp) — 单卡保护

在 `launch_allreduce_cuda_impl` 开头（L34 后）插入：

```cpp
int world_size = GlobalRegistry::instance().world_size();
if (world_size <= 1) {
    cudaEventRecord(state.streams[si].last_done_event, s);
    return;
}
```

**理由**：防御纵深。`last_done_event` 记录是给 `insert_cross_op_barrier` 用的，单卡复刻常规 ResourceOp 模式即可。不在此处加 `ncclGroupStart/End`（live replay 不需要）。

---

### 修改 3：[include/renaissance/graph/computation_graph.h](file:///r:/renaissance/include/renaissance/graph/computation_graph.h) — 检测 NCCL

```cpp
class ComputationGraph {
public:
    bool has_nccl_ops() const {
        for (const auto& node : linear_nodes_) {
            if (node.kind == GraphNode::Kind::RANGE &&
                (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                 node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE ||
                 node.range_op == RangeOp::RANGE_BN_STATS_ALLREDUCE)) {
                return true;
            }
        }
        return false;
    }
};
```

---

### 修改 4：[include/renaissance/graph/captured_graph.h](file:///r:/renaissance/include/renaissance/graph/captured_graph.h) — 新增 LiveReplay 字段

```cpp
class CapturedGraph {
public:
    bool is_live_replay() const noexcept { return is_live_replay_; }

    void enable_live_replay(const ComputationGraph* cg,
                            const MemoryPlan* mp,
                            const DeviceContext* const* contexts,
                            int num_ranks)
    {
        is_live_replay_ = true;
        replay_cg_  = cg;
        replay_mp_  = mp;
        replay_contexts_.assign(contexts, contexts + num_ranks);
    }

private:
    bool is_live_replay_ = false;
    const ComputationGraph*       replay_cg_ = nullptr;
    const MemoryPlan*             replay_mp_ = nullptr;
    std::vector<const DeviceContext*> replay_contexts_;  // per-rank contexts
};
```

**注意**：`replay_contexts_` 存储 `const DeviceContext*` 与 `DeviceContext` 生命周期一致（由 `TaskBase::backend_->contexts` 持有）。

---

### 修改 5：[src/task/task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp) — compile_capture_simple 添加 NCCL 检测

在 [compile_capture_simple()](file:///r:/renaissance/src/task/task_base.cpp#L308) 的 per-graph for 循环入口添加：

```cpp
for (auto& [name, entry] : named_graphs_) {
    CapturedGraph cg;

    if (entry.graph.has_nccl_ops()) {
        // NCCL 图：Live Replay
        TR_LOG_INFO("task") << "[DBG] graph '" << name
                            << "' contains NCCL, using live replay mode";
        cg.enable_live_replay(&entry.graph, &memory_plan_,
                              reinterpret_cast<const DeviceContext* const*>(
                                  backend_->contexts.data()),
                              num_gpus_);
    } else {
        // 普通图：原有 capture 逻辑（完全不变）
        cg.reserve_ranks(num_gpus_);
        for (int rank = 0; rank < num_gpus_; ++rank) {
            // ... 100% unchanged ...
        }
    }

    simple_captured_graphs_.emplace(name, std::move(cg));
    graph_index++;
}
```

---

### 修改 6：[src/graph/captured_graph.cpp](file:///r:/renaissance/src/graph/captured_graph.cpp) — launch 添加 LiveReplay 分支

在 [launch()](file:///r:/renaissance/src/graph/captured_graph.cpp#L144) 中 CUDA 分支头部添加：

```cpp
void CapturedGraph::launch(int rank, void* stream) const {
    if (is_cuda_) {
#ifdef TR_USE_CUDA
        if (is_live_replay_) {
            cudaStream_t s = static_cast<cudaStream_t>(stream);
            const DeviceContext& ctx = *replay_contexts_[rank];

            MultiStreamCaptureState state;
            state.primary_stream = s;

            // Pre-register primary stream + create event
            // (launcher 内 get_or_register 需要 last_done_event 非空)
            state.get_or_register(s);
            cudaEventCreateWithFlags(
                &state.streams[0].last_done_event, cudaEventDisableTiming);

            // Replay NCCL RangeOp nodes via registered launcher
            for (const auto& node : replay_cg_->linear_nodes()) {
                if (node.kind != GraphNode::Kind::RANGE) continue;
                auto& entry = g_range_op_table[
                    static_cast<size_t>(node.range_op)];
                if (entry.launch_cuda) {
                    entry.launch_cuda(node, *replay_mp_, ctx, state);
                }
            }

            state.cleanup_all_events();
            return;
        }

        // ── 原有 CUDA Graph 路径（完全不变）──
        if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size()) {
            NativeGraph exec = per_rank_execs_[rank];
            if (exec) {
                cudaError_t err = cudaGraphLaunch(
                    static_cast<cudaGraphExec_t>(exec),
                    static_cast<cudaStream_t>(stream));
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaGraphLaunch failed: "
                                    << cudaGetErrorString(err));
                }
            }
        }
#endif
    } else {
        // CPU 路径（完全不变）
        for (const auto& op : cpu_ops_) {
            if (op.fn) op.fn(static_cast<CpuOpContext*>(op.ctx));
        }
    }
}
```

**关键细节**：
1. `state.get_or_register(s)` 注册 primary stream 到 slot 0
2. 创建 `cudaEventDisableTiming` event → `launch_allreduce_cuda_impl` 内 `cudaEventRecord` 不会因 nullptr 崩溃
3. `g_range_op_table` 已通过 [op_registry.h](file:///r:/renaissance/include/renaissance/backend/op_registry.h#L85) 声明为 `extern`
4. `cleanup_all_events()` 销毁 event + temp_events

---

### 修改 7：[include/renaissance/task/simple_task.h](file:///r:/renaissance/include/renaissance/task/simple_task.h) — run_iter 统一 launch 入口

#### 7a. 单图 run_iter (L152-L186)

```cpp
// BEFORE
cudaGraphExec_t exec = static_cast<cudaGraphExec_t>(cg.native_exec(rank));
cudaStream_t stream = ...;
for (int i = 0; i < iterations; ++i) {
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
}

// AFTER
cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
for (int i = 0; i < iterations; ++i) {
    cg.launch(rank, static_cast<void*>(stream));
    cudaStreamSynchronize(stream);
}
```

**性能影响**：普通图多一次方法调用 + 一次 `per_rank_execs_[rank]` 查找（替代原有的局部变量 `exec`），与 `cudaGraphLaunch`（~100µs）相比可忽略。

#### 7b. 双图 run_iter (L258-L263)

同理替换两处 `cudaGraphLaunch(exec_a, s_a)` / `cudaGraphLaunch(exec_b, s_b)` → `cg_a.launch(rank, static_cast<void*>(s_a))` / `cg_b.launch(rank, static_cast<void*>(s_b))`。

---

## 五、修改文件清单

| # | 文件 | 改动 | 行数 | 风险 |
|---|------|------|------|------|
| 1 | `src/task/task_base.cpp` | NCCL init `>1`→`>=1` | 1 | 🟢 |
| 2 | `src/task/task_base.cpp` | compile_capture_simple 加 has_nccl 分支 | ~15 | 🟢 |
| 3 | `src/backend/ops/range/allreduce_op.cpp` | launcher 加 world_size 保护 | +3 | 🟢 |
| 4 | `include/renaissance/graph/computation_graph.h` | 新增 `has_nccl_ops()` | +12 | 🟢 |
| 5 | `include/renaissance/graph/captured_graph.h` | 新增 live replay fields + `enable_live_replay()` | +15 | 🟢 |
| 6 | `src/graph/captured_graph.cpp` | `launch()` 加 live replay 分支 | +30 | 🟡 |
| 7 | `include/renaissance/task/simple_task.h` | run_iter 单图/双图改用 `cg.launch()` | ~10 | 🟡 |

**未修改**（零触碰）：
- `capture_cuda.cpp` — 无 NCCL 图进入，走原路径不变
- `capture_multi_stream.cpp` — 同
- CPU 路径（`cpu_ops_` / `capture_cpu`）— 同

---

## 六、安全性验证矩阵

| 测试 | 走哪个路径 | 理由 |
|------|-----------|------|
| `test_clear --gpu` | 原有 CUDA Graph | `has_nccl=false` |
| `test_d2d_copy --gpu` | 原有 CUDA Graph | `has_nccl=false` |
| `test_check_nan --gpu` | 原有 CUDA Graph | `has_nccl=false` |
| `test_cast_fp32_to_fp16` (AMP) | 原有 CUDA Graph | `has_nccl=false` |
| `test_cast_fp16_to_fp32` (AMP) | 原有 CUDA Graph | `has_nccl=false` |
| `test_mean_allreduce --gpu` 单卡 | `launch`→world_size≤1→return | launcher 内保护 |
| `test_mean_allreduce --gpu` 8卡 | `launch`→LiveReplay→直调 launcher | `has_nccl=true` |
| `test_mean_allreduce --amp` | 同 GPU | `has_nccl=true` |

**回归保证**：非 NCCL 图的 `CapturedGraph::launch(rank, stream)` 行为等价于原 `cudaGraphLaunch(exec, stream)` — 只是多了一层函数调用包装。

---

## 七、验证计划

```bash
# 1. 单卡 GPU
CUDA_VISIBLE_DEVICES=0 ./test_mean_allreduce --gpu
# 预期：world_size=1 → 数据不变 → PASS

# 2. 多卡 GPU A100×8
./test_mean_allreduce --gpu
# 预期：mean = 3.5 → PASS

# 3. AMP
./test_mean_allreduce --amp
# 预期：同 GPU

# 4. 回归
./test_clear --gpu && ./test_d2d_copy --gpu
./test_cast_fp32_to_fp16 && ./test_cast_fp16_to_fp32
./test_check_nan --gpu
# 预期：全部 PASS
```

---

## 八、与三份 NCY 方案的关系

| NCY 方案 | 采纳了什么 | 放弃了什么 | 理由 |
|----------|-----------|-----------|------|
| **NCY_D** | ✅ Live Replay 战略 ✅ compile 阶段检测 ✅ run_iter 做 replay | ❌ inline 在 run_iter  lambda 中（代码重复） | 改为统一的 `cg.launch()` 入口避免污染 SimpleTask |
| **NCY_K** | ✅ CapturedGraph 存储 replay context ✅ `CapturedGraph::launch` 拦截 | ❌ `capture_cuda` 中拦截 NCCL ❌ 单 DeviceContext 存储 | capture_cuda 零触碰更好；多 rank 需 context 数组 |
| **NCY_S** | ✅ 防御纵深（两处单卡保护） | ❌ 分类检测 `is_pure_nccl_graph` ❌ `ncclGroupStart/End` ❌ `capture_nccl_graph` 专用方法 | 当前无混合图不需分类；live replay 不需 group；专用方法类不安全 |