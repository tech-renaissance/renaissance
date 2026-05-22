# FIX2: 多卡 CUDA Graph 捕获崩溃 — 完整解决方案

## 一、问题复述

`test_relu_fwd_bwd --amp --no-gen`：
- 1 GPU：PASS
- 2 GPU / 8 GPU：`cuDNN FE [execute AMP FWD]: ... failed with code: 11 (CUDNN_STATUS_EXECUTION_FAILED)`

崩溃位置：[relu_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp) 的 `launch_relu_amp_fwd_cuda()`，在 `compile_capture_simple()` 的 **捕获阶段** 内。

---

## 二、根因分析（综合小伙伴K/D + 专家DS/KM）

### 2.1 直接原因：全局静态 singleton cache 在多 handle 场景下触发 destructor 级联

旧代码（[relu_op.cpp:L72-L73 旧版](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp)）：

```cpp
static std::unique_ptr<AmpReluFwdCache> s_amp_fwd_cache;  // 全局唯一
static std::unique_ptr<AmpReluBwdCache> s_amp_bwd_cache;
```

执行流：
```
Phase A: warmup (GPU 0)
  → launch_relu_amp_fwd_cuda(handle_0)
  → s_amp_fwd_cache = make_unique<...>(handle_0)  → execute(handle_0) ✅

Phase B: capture (rank 0, 串行)
  → launch_relu_amp_fwd_cuda(handle_0)
  → cache命中 → execute(handle_0) ✅

Phase B: capture (rank 1, 串行)
  → launch_relu_amp_fwd_cuda(handle_1)
  → cache miss → rebuild
  → s_amp_fwd_cache = std::move(new_cache)         ← 旧 unique_ptr 析构!
  → GPU 0 的 feg::Graph::~Graph() 运行
  → 可能释放了 cuDNN FE 内部共享状态
  → build_amp_fwd_graph(handle_1) 完成
  → execute(handle_1) ❌ CUDNN_STATUS_EXECUTION_FAILED
```

### 2.2 深层原因：cuDNN Frontend 的 per-device 三层隔离

专家KM指出，cuDNN Frontend 有三层 per-device 隔离，**GPU 0 的预热缓存对 GPU 1 完全不可见**：

| 隔离层 | 说明 |
|--------|------|
| **Kernel Cache** | 每个 device 有独立的编译缓存，"Cached kernels are not shared across devices." |
| **Execution Plan** | 绑定到特定 `cudnnHandle_t`，handle 和设备强绑定 |
| **RTC 编译** | cuDNN Frontend 的 fusion 算子大量依赖 Runtime Compilation，编译出的 CUBIN 必须在目标卡上现场编译 |

专家DS进一步指出核心链路：
```
1. GPU 0 预热 → 生成 GPU 0 的 Execution Plan + Kernel → 仅缓存于 GPU 0
2. Rank 0 捕获 → 命中 GPU 0 缓存 → 成功
3. Rank 1 捕获 → 缓存不可用（是 GPU 0 的，per-device 隔离）
              → cuDNN Frontend 尝试为 GPU 1 构建新的 Execution Plan
              → 触发 RTC kernel 编译
              → 编译行为在 BeginCapture/EndCapture 窗口内 → 非法!
              → 捕获失败 / 损坏的 CUDA Graph
```

### 2.3 为什么 Legacy API "以前可以"而 Frontend 不行

| 维度 | Legacy API | cuDNN Frontend (Graph API) |
|------|-----------|---------------------------|
| 引擎类型 | 大量预编译 kernel（binary 已在 libcudnn.so） | 大量 RTC 引擎（JIT 编译） |
| Plan 构建 | 轻量，主要是 engine config 选择 | 重，包含 kernel 编译、workspace 计算 |
| 缓存位置 | 主要在 cuDNN 内部全局 heuristics cache | 显式的 **per-device KernelCache** |

Legacy API 的预编译 kernel 不需要在 capture 窗口内做 RTC，所以即使 cache 未命中，也不会触发 capture 禁止的操作。

### 2.4 为什么 Warmup.md 的"进程级全局 cache"假设不适用

[Warmup.md](file:///r:/renaissance/ULTIMATE_DESIGN/Warmup.md) §2：

> cuDNN engine cache 并非 per-handle 隔离，底层是进程级全局数据结构。如果 8 线程并发 build_plans()，会严重串行化（cuDNN 内部有全局锁），且可能产生损坏 cache 条目。

这个判断对 **Legacy API** 是正确的，对 **cuDNN Frontend API** 不适用。Frontend 的 kernel cache 是 per-device 的，不是进程级全局的。因此在 Frontend 下，"GPU 0 预热覆盖所有 GPU"的策略本身就不可行。

---

## 三、当前修复状态：做了什么、还缺什么

### 3.1 已完成 ✅：per-handle 缓存 map

[relu_op.cpp:L73-L74](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp#L73-L74)：

```cpp
static std::unordered_map<cudnnHandle_t, std::unique_ptr<AmpReluFwdCache>> s_amp_fwd_caches;
static std::unordered_map<cudnnHandle_t, std::unique_ptr<AmpReluBwdCache>> s_amp_bwd_caches;
```

**作用**：消除了 singleton `unique_ptr` 的 destructor 级联问题。GPU 0 的 `feg::Graph` 不再在 GPU 1 rebuild 时被错误析构。

**局限**：这只解决了"旧 graph 被析构"的问题，没有解决"新 graph 在 capture 窗口内被构建"的问题。

### 3.2 仍未修复 ❌：预热只覆盖 GPU 0

[task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp) `compile_capture_simple()` 第 270-288 行：

```cpp
// 仅预热 GPU 0！
cudaSetDevice(master_gpu);   // master_gpu = backend_->contexts[0]->device_id()
...
warmup_single_cudnn_op(node, memory_plan_, *backend_->contexts[0]);
```

[captured_graph.cpp](file:///r:/renaissance/src/graph/captured_graph.cpp) `pre_capture()` Phase B2 第 219-247 行：

```cpp
// 同样仅预热 GPU 0！
int master_gpu = contexts[0]->device_id();
cudaSetDevice(master_gpu);
...
warmup_single_cudnn_op(node, *mp, *contexts[0]);
```

**后果**：只有 `handle_0` 对应设备（GPU 0）的 `ExecutionPlan` 和 Kernel 被构建。对 rank > 0：
- SimpleTask 串行捕获：`build_amp_fwd_graph(handle_r)` 在 `BeginCapture/EndCapture` 之间执行 → `build_plans` 触发 per-device RTC → 非法 → `execute` 失败
- DeepLearningTask 并行捕获：同上，且多线程并发 `build_plans` → 竞态 + 非法

### 3.3 差距总结

```
            当前状态          需要达到
            ─────────        ─────────
预热范围    GPU 0 only       ALL GPUs
捕获窗口内  build_plans      只有 execute()
缓存结构    per-handle map   per-handle map ✓
```

---

## 四、完整修复方案

### 4.1 修复点 1：`compile_capture_simple()` 预热所有 rank

**文件**：[task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp) `compile_capture_simple()`

**当前**（第 270-288 行）：
```cpp
// cuDNN warmup（与 pre_capture 的 Phase B2 一致）
if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
    int master_gpu = backend_->contexts[0]->device_id();
    cudaSetDevice(master_gpu);
    cudaGetLastError();
    cudaDeviceSynchronize();

    backend_->contexts[0]->set_rank(0);
    backend_->contexts[0]->set_memory_plan(&memory_plan_);

    for (const auto& [name, entry] : named_graphs_) {
        for (const auto& node : entry.graph.linear_nodes()) {
            if (node.kind != GraphNode::Kind::COMPUTE) continue;
            if (!require_warmup(node.compute_op)) continue;
            warmup_single_cudnn_op(node, memory_plan_, *backend_->contexts[0]);
        }
    }
    cudaDeviceSynchronize();
    cudaGetLastError();
    ...
}
```

**修改为**：对所有 rank 串行预热（预热阶段本身就是串行的，没有并发竞态风险）：

```cpp
// cuDNN warmup：每个 rank 独立预热，确保所有设备的 kernel cache + plan cache 已填充
if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
    for (int rank = 0; rank < num_gpus_; ++rank) {
        DeviceContext& warm_ctx = *backend_->contexts[rank];
        cudaSetDevice(warm_ctx.device_id());
        cudaGetLastError();
        cudaDeviceSynchronize();

        warm_ctx.set_rank(rank);
        warm_ctx.set_memory_plan(&memory_plan_);

        for (const auto& [name, entry] : named_graphs_) {
            for (const auto& node : entry.graph.linear_nodes()) {
                if (node.kind != GraphNode::Kind::COMPUTE) continue;
                if (!require_warmup(node.compute_op)) continue;
                warmup_single_cudnn_op(node, memory_plan_, warm_ctx);
            }
        }
        cudaDeviceSynchronize();
        cudaGetLastError();
    }
    ...
}
```

### 4.2 修复点 2：`pre_capture()` Phase B2 预热所有 rank

**文件**：[captured_graph.cpp](file:///r:/renaissance/src/graph/captured_graph.cpp) `pre_capture()`

**当前**（第 219-247 行）：
```cpp
const bool is_cuda = (contexts[0]->is_gpu());
if (is_cuda && K > 0) {
    int master_gpu = contexts[0]->device_id();
    cudaSetDevice(master_gpu);
    ...
    for (int32_t k = 0; k < K; ++k) {
        ...
        warmup_single_cudnn_op(node, *mp, *contexts[0]);
    }
    cudaDeviceSynchronize();
}
```

**修改为**：每个 rank 独立预热（串行，在并行捕获前完成）：

```cpp
const bool is_cuda = (contexts[0]->is_gpu());
if (is_cuda && K > 0) {
    // 每个 rank 独立预热（串行执行，在并行捕获前完成）
    // cuDNN Frontend 的 kernel cache 是 per-device 的，必须每卡单独填充
    for (int r = 0; r < num_ranks; ++r) {
        int gpu_id = contexts[r]->device_id();
        cudaSetDevice(gpu_id);

        TR_LOG_INFO("graph") << "[B2] cuDNN warmup on rank " << r
                             << " (GPU " << gpu_id << ")";

        for (int32_t k = 0; k < K; ++k) {
            const auto& key = key_by_idx[k];
            if (!key.cg) continue;

            auto it = key_to_mp.find(key);
            TR_CHECK(it != key_to_mp.end() && it->second != nullptr, ValueError,
                     "No MemoryPlan for key[" << k << "]");
            const MemoryPlan* mp = it->second;

            contexts[r]->set_rank(r);
            contexts[r]->set_memory_plan(mp);

            for (const auto& node : key.cg->nodes(key.gid)) {
                if (node.kind != GraphNode::Kind::COMPUTE) continue;
                if (!require_warmup(node.compute_op)) continue;
                warmup_single_cudnn_op(node, *mp, *contexts[r]);
            }
        }
        cudaDeviceSynchronize();
    }
    TR_LOG_INFO("graph") << "[B2] cuDNN warmup completed on all " << num_ranks << " ranks";
}
```

### 4.3 修复点 3（防御层）：`launch_relu_amp_fwd_cuda` 中捕获期间禁止 rebuild

**文件**：[relu_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp)

在 `launch_relu_amp_fwd_cuda` 和 `launch_relu_amp_bwd_cuda` 中，如果检测到缓存缺失，不应静默 rebuild，而应在 `debug` 模式下报错（捕获期间不应该触发 rebuild），或者至少在非 debug 模式下发出 warning。

这一层修改是可选的防御措施——如果上述 4.1/4.2 修复正确实施，捕获期间不会触发 rebuild。但加上防御层可以提前暴露预热覆盖不完全的问题：

```cpp
if (rebuild) {
#ifndef NDEBUG
    TR_LOG_WARN("relu") << "[AMP FWD] Rebuilding cuDNN FE graph during launch. "
                        << "This should have been done during warmup. "
                        << "handle=" << handle << " N=" << N << " C=" << C;
#endif
    auto cache = std::make_unique<AmpReluFwdCache>();
    // ... 同原来 ...
}
```

### 4.4 修复点 4（可选）：`warmup_single_cudnn_op` 预热期间的 workspace 释放

在 [op_registry.cpp](file:///r:/renaissance/src/backend/op_registry.cpp) `warmup_single_cudnn_op` 中，非 Conv 算子（RELU AMP 等）通过 `entry.launch_cuda` 预热。预热完成后 `cudaStreamSynchronize` + `state.cleanup_all_events()` 清理事件。

当前实现没有问题，但需要注意：预热中创建的 `MultiStreamCaptureState` 是局部的，预热后自动销毁。`cleanup_all_events()` 销毁了事件。不需要修改。

---

## 五、修复后的完整流程

```
Phase 1: 预热 (warmup) — 所有 rank 串行
══════════════════════════════════════════════════════════════
  for rank in 0..N-1:
    cudaSetDevice(gpu_ids[rank])
    ctx.set_rank(rank)
    ctx.set_memory_plan(&mp)

    for each cuDNN node:
      → warmup_single_cudnn_op(node, mp, ctx)
        → (非Conv) entry.launch_cuda(node, mp, ctx, state)
          → launch_relu_amp_fwd_cuda(handle_r)
            → s_amp_fwd_caches.find(handle_r) → miss → rebuild
            → build_amp_fwd_graph(handle_r, ...)
              → build_operation_graph(handle_r)  ✅ 普通模式，合法
              → build_plans(HEURISTICS_CHOOSE)    ✅ 普通模式，触发 per-device RTC
              → execute(handle_r)                 ✅ 普通模式，填充 kernel cache
            → caches[handle_r] = graph
        → cudaStreamSynchronize
    cudaDeviceSynchronize

  结果：所有 rank 的 handle 都在 s_amp_fwd_caches 中有缓存条目。
       所有 rank 的 per-device kernel cache 已填充。

Phase 2: 捕获 (capture) — 所有 rank 串行 (SimpleTask) 或 0串行+1~N-1并行 (DeepLearningTask)
══════════════════════════════════════════════════════════════
  for each graph, for each rank:
    cuda_set_device
    BeginCapture(stream, ThreadLocal)

    for each node:
      → launch_relu_amp_fwd_cuda(handle_r)
        → s_amp_fwd_caches.find(handle_r) → HIT ✅
        → execute(handle_r)  ✅ 捕获窗口内只有 kernel launch，无 RTC
      → cudaEventRecord(last_done_event, s)

    EndCapture → Instantiate → exec存入per_rank_execs_[rank]

Phase 3: 运行
══════════════════════════════════════════════════════════════
  cudaGraphLaunch(exec, stream)
```

---

## 六、验证计划

### 6.1 编译

```bash
# Linux
cd build && cmake --build . --config Release -j$(nproc)
```

### 6.2 单元测试

```bash
# 1 GPU - 应 PASS
CUDA_VISIBLE_DEVICES=0 ./test_relu_fwd_bwd --amp --no-gen

# 2 GPU - 应 PASS
CUDA_VISIBLE_DEVICES=0,1 ./test_relu_fwd_bwd --amp --no-gen

# 8 GPU - 应 PASS
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./test_relu_fwd_bwd --amp --no-gen

# CPU 模式 - 应 PASS
./test_relu_fwd_bwd --cpu

# GPU FP32 模式 - 应 PASS
CUDA_VISIBLE_DEVICES=0 ./test_relu_fwd_bwd --gpu --no-gen
```

### 6.3 回归检查

- 单 GPU AMP 不应退化
- FP32 路径不受影响（FP32 不使用 cuDNN Frontend）
- CPU 模式不受影响

---

## 七、涉及文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| [task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp) | **修改** | `compile_capture_simple()` 预热循环从仅 GPU 0 改为所有 rank |
| [captured_graph.cpp](file:///r:/renaissance/src/graph/captured_graph.cpp) | **修改** | `pre_capture()` Phase B2 预热从仅 GPU 0 改为所有 rank |
| [relu_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp) | **可选** | 增加防御性 warning（捕获期间 rebuild 检测） |
| [relu_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp) | **已完成** | per-handle cache map（`unordered_map<cudnnHandle_t, unique_ptr<Cache>>`） |

---

## 八、与 Warmup.md 设计文档的关系

[Warmup.md](file:///r:/renaissance/ULTIMATE_DESIGN/Warmup.md) 的三段式策略：

> Phase 1: GPU 0 串行预热 cuDNN engine cache
> Phase 2: GPU 0 串行捕获 + GPU 1~7 并行捕获

这个策略对 **Legacy API** 是正确的（Legacy API 的 engine cache 是进程级全局的）。

对 **cuDNN Frontend API**，需要修正为：

> Phase 1: **所有 GPU 串行预热**（cuDNN Frontend 的 kernel cache 是 per-device 的）
> Phase 2: GPU 0 串行捕获 + GPU 1~7 并行捕获（预热已覆盖所有设备，捕获期间只有 cache 命中 → execute）

核心修正：预热阶段的目标设备从"GPU 0"扩展为"所有 GPU"。

Warmup.md 文档本身可能需要更新，但这是文档层面的事情，本文档聚焦代码修复。