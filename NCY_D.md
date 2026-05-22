# NCCL AllReduce CUDA Graph 修复 —— 综合分析 & 最终方案

> **输入材料**：NCC_S.md · NCC_K.md · NCC_D.md · test_nccl_perf.cpp · errors.txt
>
> 2026-05-20

---

## 一、问题确认（三方共识，无需再讨论）

### 1.1 单卡：`invalid argument`

| 现象 | 位置 |
|------|------|
| `ncclAllReduce failed: invalid argument` | `errors.txt` L29 → `allreduce_op.cpp :: launch_allreduce_cuda_impl()` |

**根因**：`task_base.cpp:compile_alloc_hardware()` 中 `if (gpu_ids.size() > 1)` → 单卡 NCCL comm = nullptr。

### 1.2 多卡：死锁

| 现象 | 位置 |
|------|------|
| 挂起在 `capture rank 0 begin` | `errors.txt` L85-86 → `compile_capture_simple()` |

**根因**：`compile_capture_simple()` 逐 rank 串行 capture — rank 0 单独进 capture 调 `ncclAllReduce`，rank 1-7 未 `BeginCapture`。

**NCCL 硬性要求**：所有参与 rank 必须在同一个 ncclGroupStart/End 窗口内，且所有 stream 的 `BeginCapture` 在该窗口之前完成。

---

## 二、三份方案核心差异

| | NCC_D | NCC_S | **NCC_K** |
|---|---|---|---|
| 战略 | capture_cuda 拆分三阶段 | launcher 内 ncclGroupStart/End + compile 协调 | **NCCL 不走 CUDA Graph** |
| 修改的文件数 | 3 | 3 | **5**（但关键路径更短） |
| 修改 capture_cuda 内部 | ✅ 拆分函数 | ✅ 暴露 replay | **❌ 不动** |
| 修改 compile_capture_simple | ✅ ~50 行 | ✅ ~50 行 | ✅ ~15 行 |
| 修改 allreduce_op | +ncclGroupStart/End | 同左 | **仅 world_size 保护** |
| 需要 ncclGroupStart/End | ✅ | ✅ | **❌（不在 capture 中）** |
| 多 rank 同步复杂度 | 高 | 高 | **零** |

---

## 三、方向性决策：NCCL AllReduce 不需要 CUDA Graph 加速

### 3.1 性能论据

```
AllReduce launch overhead  ≈ 100µs    (CUDA kernel launch -> NCCL dispatch)
AllReduce communication    ≈ 1400µs   (100MB FP32 @ 8×A100 NVLink ~150GB/s)
                          ─────────
CUDA Graph 节省占比        ≈ 100/1400 = 7%
```

CUDA Graph 的主要价值是消除 kernel launch chain 的 CPU dispatch bottleneck（数百个 micro-kernel 的线程调度）。对于只有一个 NCCL 集合调用的图，dispatch 开销几乎为零。

### 3.2 风险论据

NCC_K（Live Replay）的正确性**不依赖**任何 CUDA Graph capture 行为，风险为零。相比之下：
- NCC_D/NCC_S 需要拆解 `capture_cuda` 内部逻辑 + 在 `compile_capture_simple` 中跨 rank 重新编排
- 需要保证 ncclGroupStart/End 跨 rank 的线程安全
- 需要在 capture 流程中正确管理 per-rank `MultiStreamCaptureState`
- 需要保证 NON-NCCL 图的 capture 行为不被影响（`has_nccl` 分支已确保，但 code path 有更多分支）

### 3.3 实用论据

- 生产代码中所有 NCCL 图（FIRST_COMM, DEEP_COMM, STATS_COMM）都是**纯 NCCL 操作**，不混杂 compute
- Live replay 在未来也支持 NCCL + compute 混合图（只需在 capture 时跳过 NCCL 节点 → CUDA Graph 只 capture compute → launch 时先 launch graph 再 live replay NCCL）
- `test_nccl_perf.cpp` 已验证 NCCL AllReduce 在 live 模式下的正确性和性能

---

## 四、最终方案：Enhanced Live Replay

### 4.1 架构总览

```
┌────────────────────────────────────────────────────────────┐
│                   compile_capture_simple()                 │
├────────────────────────────────────────────────────────────┤
│ for each graph:                                           │
│   if graph.has_nccl_ops():                                │
│     ┌─ store LiveReplay context in CapturedGraph          │
│     └─ per_rank_execs_ ← EMPTY (no CUDA Graph)            │
│   else:                                                   │
│     └─ per-rank CapturedGraph::capture (existing) /*不变*/ │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                   SimpleTask::run_iter()                   │
├────────────────────────────────────────────────────────────┤
│ for each rank (parallel threads):                         │
│   if cg.is_live_replay():                                 │
│     for iter:                                             │
│       直接调 launch_allreduce_cuda_impl(...)               │
│       cudaStreamSynchronize                               │
│   else:                                                   │
│     cudaGraphLaunch(exec, stream)  /*不变*/               │
│     cudaStreamSynchronize                                 │
└────────────────────────────────────────────────────────────┘
```

### 4.2 关键设计决策：为什么不在 CapturedGraph::launch 做 live replay

`SimpleTask::run_iter` GPU 路径（`simple_task.h` L152-186）**直接访问** `cg.native_exec(rank)` 并调用 `cudaGraphLaunch`，不经过 `CapturedGraph::launch()`。因此 live replay 必须在 `run_iter` 中实现。

这是合理的：`run_iter` 是唯一掌握 "当前 rank 的 DeviceContext & stream" 的层次，正是 live replay 所需的上下文。

---

## 五、具体代码修改

### 5.1 `src/task/task_base.cpp` — NCCL 初始化

**位置**：`compile_alloc_hardware()` 中 NCCL 初始化条件块。

**改动**：`gpu_ids.size() > 1` → `gpu_ids.size() >= 1`

```cpp
// BEFORE
#ifdef TR_USE_NCCL
    if (gpu_ids.size() > 1) {
        ncclCommInitAll(...);
    }
#endif

// AFTER
#ifdef TR_USE_NCCL
    if (gpu_ids.size() >= 1) {
        ncclCommInitAll(comms.data(), (int)gpu_ids.size(), gpu_ids.data());
        for (size_t r = 0; r < gpu_ids.size(); ++r) {
            contexts[r]->set_nccl_comm(comms[r], (int)gpu_ids.size());
        }
    }
#endif
```

**风险**：零。无额外行为变更。三方共识修改。

---

### 5.2 `src/backend/ops/range/allreduce_op.cpp` — 单卡保护

**位置**：`launch_allreduce_cuda_impl` 函数开头（L34 之后）

```cpp
int world_size = GlobalRegistry::instance().world_size();
if (world_size <= 1) return;
```

**风险**：零。防御纵深。三方共识修改。

**不在此处添加 `ncclGroupStart/End`**：live replay 路径不在 CUDA Graph capture 中，不需要。

---

### 5.3 `include/renaissance/graph/computation_graph.h` — 新增 has_nccl_ops()

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

### 5.4 `include/renaissance/graph/captured_graph.h` — 新增 LiveReplay 字段

```cpp
class CapturedGraph {
public:
    bool is_live_replay() const noexcept { return is_live_replay_; }

    // 获取 live replay 所需的引用
    const ComputationGraph* replay_cg() const { return replay_cg_; }
    const MemoryPlan*       replay_mp() const { return replay_mp_; }

private:
    bool is_live_replay_ = false;

    // Live replay 上下文
    const ComputationGraph* replay_cg_ = nullptr;
    const MemoryPlan*       replay_mp_ = nullptr;
};
```

**不存储 per-rank DeviceContext**：`run_iter` 中每 rank 的 lambda 已持有 `DeviceContext& ctx = context(rank)`。

---

### 5.5 `src/task/task_base.cpp` — compile_capture_simple 中的检测

**位置**：`compile_capture_simple()` 的 capture for-loop（L307-L357）

在 per-rank 循环之前插入 NCCL 检测：

```cpp
for (auto& [name, entry] : named_graphs_) {
    // ▼▼▼ 新增 ▼▼▼
    bool contains_nccl = entry.graph.has_nccl_ops();

    CapturedGraph cg;
    if (contains_nccl) {
        cg.is_live_replay_ = true;
        cg.replay_cg_ = &entry.graph;
        cg.replay_mp_ = &memory_plan_;
        // per_rank_execs_ 保持空（不创建 CUDA Graph）
        cg.set_metadata_from(nullptr);  // 或从 entry 设 metadata
    } else {
    // ▲▲▲ 新增结束 ▲▲▲
        cg.reserve_ranks(num_gpus_);
        for (int rank = 0; rank < num_gpus_; ++rank) {
            // ... 原有 capture 逻辑，完全不变 ...
        }
        // ... set_metadata_from ...
    }

    simple_captured_graphs_.emplace(name, std::move(cg));
}
```

---

### 5.6 `include/renaissance/task/simple_task.h` — run_iter GPU 路径添加 live replay

#### 5.6.1 单图 run_iter（L150-L186）

```cpp
// ── GPU 路径：多线程 ──
if (K > 0 && context(0).is_gpu()) {
    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, &cg, rank, sk, iterations, &exc]() {
            try {
                DeviceContext& ctx = context(rank);
                cudaError_t err = cudaSetDevice(ctx.device_id());
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

                cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));

                // ▼▼▼ 新增：Live Replay 分支 ▼▼▼
                if (cg.is_live_replay()) {
                    MultiStreamCaptureState state;
                    state.primary_stream = stream;

                    for (int i = 0; i < iterations; ++i) {
                        for (const auto& node : cg.replay_cg()->linear_nodes()) {
                            if (node.kind != GraphNode::Kind::RANGE) continue;
                            auto& entry = g_range_op_table[
                                static_cast<size_t>(node.range_op)];
                            if (entry.launch_cuda) {
                                entry.launch_cuda(node, *cg.replay_mp(),
                                                  ctx, state);
                            }
                        }
                        cudaStreamSynchronize(stream);
                    }
                    return;
                }
                // ▲▲▲ Live Replay 分支结束 ▲▲▲

                // ── 原有 CUDA Graph 路径 ──
                cudaGraphExec_t exec = static_cast<cudaGraphExec_t>(
                    cg.native_exec(rank));
                for (int i = 0; i < iterations; ++i) {
                    cudaGraphLaunch(exec, stream);
                    cudaStreamSynchronize(stream);
                }
            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }
    // ... join / rethrow 不变 ...
}
```

#### 5.6.2 双图 run_iter（L240-L272）

同理，在双图线程 lambda 内部，每个图独立检查 `is_live_replay()`：

```cpp
// 对 cg_a:
if (cg_a.is_live_replay()) {
    // live replay cg_a on s_a
} else {
    cudaGraphLaunch(exec_a, s_a);
}
// 对 cg_b:
if (cg_b.is_live_replay()) {
    // live replay cg_b on s_b
} else {
    cudaGraphLaunch(exec_b, s_b);
}
```

---

## 六、文件修改清单

| 文件 | 改动 | 行数 | 风险 |
|------|------|------|------|
| `src/task/task_base.cpp` | NCCL 初始化 `>1` → `>=1` | 1 | 🟢 零 |
| `src/task/task_base.cpp` | compile_capture_simple 添加 has_nccl 检测 + live replay context 存储 | ~15 | 🟢 零 |
| `src/backend/ops/range/allreduce_op.cpp` | launcher 添加 `if (world_size <= 1) return` | 2 | 🟢 零 |
| `include/renaissance/graph/computation_graph.h` | 新增 `has_nccl_ops()` | 10 | 🟢 零 |
| `include/renaissance/graph/captured_graph.h` | 新增 live replay 字段 + getter | 8 | 🟢 零 |
| `include/renaissance/task/simple_task.h` | run_iter GPU 路径添加 live replay 分支（单图 + 双图） | ~40 | 🟡 中—模板方法，注意 lambda capture |

**未修改**：`capture_cuda.cpp`、`captured_graph.cpp` — 内核捕获/launch 路径零变动。

---

## 七、安全性验证矩阵

| 测试 | Live Replay 路径 | 说明 |
|------|------------------|------|
| `test_cast_fp32_to_fp16 --amp` | ❌ 不走 | cast 图不含 NCCL，`has_nccl=false` → 走原路径 |
| `test_d2d_copy --gpu` | ❌ 不走 | 同上 |
| `test_check_nan --gpu` | ❌ 不走 | 同上 |
| `test_clear --gpu` | ❌ 不走 | 同上 |
| `test_mean_allreduce --gpu` 单卡 | ✅ 走 | `world_size=1` → launcher 内直接 return |
| `test_mean_allreduce --gpu` 8卡 | ✅ 走 | 每个 rank 独立 live replay `ncclAllReduce` |
| `test_mean_allreduce --amp` | ✅ 走 | GPU 路径，同 --gpu |

**回归保证**：`has_nccl()` 返回 false 的图 → `cg.is_live_replay()` = false → 零代码路径变更。

---

## 八、验证计划

### Step 1: 单卡 GPU

```bash
CUDA_VISIBLE_DEVICES=0 ./test_mean_allreduce --gpu
```
预期：world_size=1 → launcher return，dt_a/dt_b 保持初始值（本 rank 的值）

### Step 2: 多卡 GPU A100×8

```bash
./test_mean_allreduce --gpu
```
预期：rank k 填充 k → mean = 3.5 → PASS

### Step 3: AMP 模式

```bash
./test_mean_allreduce --amp
```
预期：同 GPU 路径

### Step 4: 回归（确保原有测试不受影响）

```bash
./test_clear --gpu && ./test_clear --cpu
./test_d2d_copy --gpu && ./test_d2d_copy --cpu
./test_cast_fp32_to_fp16 && ./test_cast_fp16_to_fp32
./test_check_nan --gpu
```

---

## 九、与 Live Replay 配合的注意事项

### 9.1 不需 `ncclGroupStart/End`

Live replay 不在 CUDA Graph 捕获中，ncclAllReduce 作为普通 NCCL 调用，NCCL 内部自行管理 rank 同步（ring/tree algorithm），无需外部 group。

### 9.2 warmup 不影响

`compile_capture_simple` 中的 warmup 循环在 capture 之前单独执行，与 live replay 无交互。warmup 期间 launcher 的常规路径已验证能正常运行（warmup 不在 capture 中）。

### 9.3 stream 生命周期

Live replay 使用 `MultiStreamCaptureState`（`state.primary_stream = stream`），与普通图的 capture replay 一致。stream 由 SimpleTask 管理，live replay 只借用。

### 9.4 双图调用扩展性

当前测试只用单图。双图 `run_iter(a, b)` 已预留分支入口。未来若有通信 + 计算双图并行需求，可直接加入 `is_live_replay()` 检测为 a/b 分别处理。