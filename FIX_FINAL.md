# FIX_FINAL — ReLU AMP 多卡 CUDA Graph 捕获崩溃 最终解决方案

> 综合 FIX.md（四人专家组）+ FIX2.md + FIX3.md + 6 个关键文件代码审计

---

## 目 录

1. [问题现象](#一问题现象)
2. [根因分析](#二根因分析)
3. [当前代码状态](#三当前代码状态)
4. [完整修复方案](#四完整修复方案)
5. [线程安全分析](#五线程安全分析)
6. [验证计划](#六验证计划)
7. [风险与回滚](#七风险与回滚)
8. [架构启示](#八架构启示)

---

## 一、问题现象

### 1.1 复现命令

```bash
CUDA_VISIBLE_DEVICES=0       ./test_relu_fwd_bwd --amp --no-gen   # ✅ PASS
CUDA_VISIBLE_DEVICES=0,1     ./test_relu_fwd_bwd --amp --no-gen   # ❌ CRASH
CUDA_VISIBLE_DEVICES=0,1,...,7 ./test_relu_fwd_bwd --amp --no-gen # ❌ CRASH
```

### 1.2 错误信息

```
DeviceError: cuDNN FE [execute AMP FWD]: ... failed with code: 11 (CUDNN_STATUS_EXECUTION_FAILED)
at relu_op.cpp :: launch_relu_amp_fwd_cuda()
```

### 1.3 失败时机

- `compile()` → `compile_impl()` → `compile_capture_simple()` → **捕获阶段内**，rank 1+
- 日志停在 `Allocated 2 GPU device context(s)` 之后（即 `compile_alloc_hardware()` 后、`compile_mark_compiled()` 前）
- 不是 `warmup`、不是 `instantiate`、不是 `runtime`

---

## 二、根因分析

### 2.1 直接原因链

```
Phase A: warmup (仅 GPU 0)
  → launch_relu_amp_fwd_cuda(handle_0)
  → build_amp_fwd_graph(handle_0) → build_plans  ← 普通模式，合法
  → execute(handle_0) ✅

Phase B: capture rank 0
  → launch_relu_amp_fwd_cuda(handle_0)
  → 缓存命中 → execute(handle_0) ✅  ← BeginCapture 窗口内，纯 kernel launch

Phase B: capture rank 1
  → launch_relu_amp_fwd_cuda(handle_1)
  → 缓存 MISS → build_amp_fwd_graph(handle_1)  ← BeginCapture 窗口内！
    → build_operation_graph(handle_1)
    → check_support(handle_1)
    → build_plans(HEURISTICS_CHOOSE)  ← 可能触发 RTC kernel 编译
    → execute(handle_1) ❌ CUDNN_STATUS_EXECUTION_FAILED
```

### 2.2 两层根因

#### 层 1：历史已修复 — singleton unique_ptr destructor 级联

旧版代码（[relu_op.cpp 旧版](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp)）：

```cpp
static std::unique_ptr<AmpReluFwdCache> s_amp_fwd_cache;  // 全局唯一
```

rank 1 rebuild 时 `s_amp_fwd_cache = std::move(new_cache)` 触发 GPU 0 的 `feg::Graph::~Graph()`，释放了 cuDNN FE 内部可能跨 graph 共享的资源，导致新 graph 的 `execute` 失败。

#### 层 2：当前剩余问题 — 预热仅覆盖 GPU 0，捕获期触发 per-device RTC

cuDNN Frontend 的 **kernel cache 是 per-device 的**（专家KM明确指出："Cached kernels are not shared across devices"）。GPU 0 预热产生的 Execution Plan 和编译后的 CUBIN 对 GPU 1 完全不可见。当 rank 1 在 `BeginCapture/EndCapture` 窗口内执行 `build_plans` 时：

1. cuDNN Frontend 尝试为 GPU 1 构建新的 Execution Plan
2. 触发 per-device RTC kernel 编译
3. Kernel 编译行为（包括临时内存分配、NVRTC 调用）落在 CUDA Graph Capture 窗口内
4. 违反 CUDA Graph 捕获规则 → `CUDNN_STATUS_EXECUTION_FAILED`（code 11）

专家DS的总结：

> cuDNN Frontend要求所有Kernel必须在捕获前完成加载，并缓存于该进程中。在CUDA Graph Capture期间，任何需要新Kernel的操作都会被视为非法。

### 2.3 为什么 Legacy API 没有这个问题

| 维度 | Legacy API | cuDNN Frontend (Graph API) |
|------|-----------|---------------------------|
| **引擎类型** | 预编译 kernel（binary 在 libcudnn.so） | 大量 RTC 引擎（JIT 编译） |
| **Plan 构建** | 轻量 engine config 选择 | 重：kernel 编译 + workspace 计算 |
| **缓存隔离** | 进程级全局 heuristics cache | **per-device KernelCache + ExecutionPlanCache** |
| **捕获兼容** | 预编译 kernel 无需捕获窗口内 RTC | RTC 编译在捕获窗口内非法 |

### 2.4 更正 Warmup.md 的一个假设

[Warmup.md](file:///r:/renaissance/ULTIMATE_DESIGN/Warmup.md) §2 声明：

> cuDNN engine cache 并非 per-handle 隔离，底层是进程级全局数据结构。

此论断对 **Legacy cuDNN API** 成立，对 **cuDNN Frontend API** 不成立。Frontend 的 kernel cache / ExecutionPlan 都是 per-device 隔离的。因此 Warmup.md 的「GPU 0 预热即可覆盖所有 GPU」策略在引入 Frontend 算子后失效，必须修正为「**每个 rank 独立预热**」。

---

## 三、当前代码状态

对6个关键文件的审计结果：

### 3.1 [relu_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp) — ✅ per-handle map 已落地

```cpp
// L73-74
static std::unordered_map<cudnnHandle_t, std::unique_ptr<AmpReluFwdCache>> s_amp_fwd_caches;
static std::unordered_map<cudnnHandle_t, std::unique_ptr<AmpReluBwdCache>> s_amp_bwd_caches;
```

- `launch_relu_amp_fwd_cuda` (L234-296)：per-handle 查找 + 参数不匹配时 rebuild
- `launch_relu_amp_bwd_cuda` (L298-361)：同上

### 3.2 [task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp) — ❌ warmup 仅 rank 0

```cpp
// L270-288: compile_capture_simple() warmup 阶段
cudaSetDevice(master_gpu);                                    // 仅 GPU 0
warmup_single_cudnn_op(node, memory_plan_, *backend_->contexts[0]); // 仅 context[0]
```

### 3.3 [captured_graph.cpp](file:///r:/renaissance/src/graph/captured_graph.cpp) — ❌ Phase B2 仅 rank 0

```cpp
// L219-247: pre_capture() Phase B2
int master_gpu = contexts[0]->device_id();
cudaSetDevice(master_gpu);               // 仅 GPU 0
warmup_single_cudnn_op(node, *mp, *contexts[0]); // 仅 context[0]
```

### 3.4 [op_registry.cpp](file:///r:/renaissance/src/backend/op_registry.cpp) — ✅ 无需修改

`warmup_single_cudnn_op` (L82-167) 接受 `DeviceContext& ctx` 参数，内部通过 `ctx.cudnn_handle()` 和 `ctx.stream()` 获取正确的 handle 和 stream，天然支持多 rank。不需要修改。

### 3.5 [capture_cuda.cpp](file:///r:/renaissance/src/graph/capture_cuda.cpp) — ✅ 无需修改

`capture_cuda()` 在 `BeginCapture` 窗口内逐节点 replay。如果所有 cuDNN 算子在捕获前已完成预热（缓存命中），窗口内只有 `execute()`（纯 kernel launch），完全符合 CUDA Graph 规则。不需要修改。

### 3.6 [capture_multi_stream.cpp](file:///r:/renaissance/src/graph/capture_multi_stream.cpp) — ✅ 无需修改

`MultiStreamCaptureState` 和相关函数工作正常。预热期间的 `cleanup_all_events()` 在 `cudaStreamSynchronize` 之后调用，安全。

### 3.7 差距汇总

| 组件 | 状态 | 问题 |
|------|:----:|------|
| `relu_op.cpp` per-handle map | ✅ | — |
| `task_base.cpp` warmup 范围 | ❌ | 仅 GPU 0，需扩展到所有 rank |
| `captured_graph.cpp` Phase B2 范围 | ❌ | 仅 GPU 0，需扩展到所有 rank |
| `op_registry.cpp` | ✅ | 无需修改 |
| `capture_cuda.cpp` | ✅ | 无需修改 |
| `capture_multi_stream.cpp` | ✅ | 无需修改 |

---

## 四、完整修复方案

### 修改 1：[task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp) — `compile_capture_simple()`

**位置**：第 268-301 行，`compile_capture_simple()` 函数的 warmup 段

**变更内容**：将预热从仅 rank 0 改为对所有 rank 串行预热

**SEARCH**（第 268-301 行）：
```cpp
void TaskBase::compile_capture_simple() {
#ifdef TR_USE_CUDA
    // cuDNN warmup（与 pre_capture 的 Phase B2 一致）
    if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
        int master_gpu = backend_->contexts[0]->device_id();
        cudaSetDevice(master_gpu);
        cudaGetLastError();  // 清除 ArenaKeeper 初始化可能遗留的粘滞错误
        cudaDeviceSynchronize();

        // 预热前必须先设置 rank 和 memory_plan，否则 ptr_at() 会失败
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

        // 预分配 RANGE_H2D_COPY_DTENSOR 节点的 pinned memory（捕获期禁止 cudaHostAlloc）
        for (const auto& [name, entry] : named_graphs_) {
            for (const auto& node : entry.graph.linear_nodes()) {
                if (node.kind != GraphNode::Kind::RANGE) continue;
                if (node.range_op != RangeOp::RANGE_H2D_COPY_DTENSOR) continue;
                if (node.output_ranges.empty()) continue;
                const auto& seg = node.output_ranges[0];
                get_dtensor_pinned_buffer(seg.offset, seg.size);
            }
        }
    }
#endif
```

**REPLACE**：
```cpp
void TaskBase::compile_capture_simple() {
#ifdef TR_USE_CUDA
    if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
        // pinned memory 预分配（捕获期禁止 cudaHostAlloc，此操作设备无关，提前完成）
        for (const auto& [name, entry] : named_graphs_) {
            for (const auto& node : entry.graph.linear_nodes()) {
                if (node.kind != GraphNode::Kind::RANGE) continue;
                if (node.range_op != RangeOp::RANGE_H2D_COPY_DTENSOR) continue;
                if (node.output_ranges.empty()) continue;
                const auto& seg = node.output_ranges[0];
                get_dtensor_pinned_buffer(seg.offset, seg.size);
            }
        }

        // cuDNN warmup：每个 rank 独立串行预热
        // cache（per-device 的）和 s_amp_fwd_caches（per-handle 的）全部填充。
        for (int rank = 0; rank < num_gpus_; ++rank) {
            DeviceContext& warm_ctx = *backend_->contexts[rank];
            int gpu_id = warm_ctx.device_id();
            cudaSetDevice(gpu_id);
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
    }
#endif
```

**设计理由**：
1. **pinned memory 提前**：`cudaHostAlloc` 在 GPU 0 上执行一次即可（pinned memory 是设备无关的 page-locked host memory），移到循环外避免重复分配。
2. **串行安全**：`for rank` 是串行的，每个 rank 的 `build_plans` 顺序执行，无 cuDNN 全局锁竞争。这与 Warmup.md 对 Legacy API 的判断一致。
3. **必须独立预热**：cuDNN Frontend 的三层 per-device 隔离（Kernel Cache / Execution Plan / RTC）决定了 GPU 0 的预热结果对 GPU 1 不可见。

### 修改 2：[captured_graph.cpp](file:///r:/renaissance/src/graph/captured_graph.cpp) — `pre_capture()` Phase B2

**位置**：第 216-247 行，`pre_capture()` 函数的 Phase B2 段

**SEARCH**（第 216-247 行）：
```cpp
    // =====================================================================
    // Phase B2: cuDNN预热（仅CUDA，GPU 0单线程串行）
    // =====================================================================
    const bool is_cuda = (contexts[0]->is_gpu());
    if (is_cuda && K > 0) {
#ifdef TR_USE_CUDA
        int master_gpu = contexts[0]->device_id();
        cudaSetDevice(master_gpu);

        TR_LOG_INFO("graph") << "[B2] cuDNN warmup on GPU " << master_gpu
                             << " for " << K << " unique graphs";

        for (int32_t k = 0; k < K; ++k) {
            const auto& key = key_by_idx[k];
            if (!key.cg) continue;

            auto it = key_to_mp.find(key);
            TR_CHECK(it != key_to_mp.end() && it->second != nullptr, ValueError,
                     "No MemoryPlan for key[" << k << "]");
            const MemoryPlan* mp = it->second;

            for (const auto& node : key.cg->nodes(key.gid)) {
                if (node.kind != GraphNode::Kind::COMPUTE) continue;
                if (!require_warmup(node.compute_op)) continue;
                warmup_single_cudnn_op(node, *mp, *contexts[0]);
            }
        }

        cudaDeviceSynchronize();
        TR_LOG_INFO("graph") << "[B2] cuDNN warmup completed";
#endif
    }
```

**REPLACE**：
```cpp
    // =====================================================================
    // Phase B2: cuDNN预热 — 每个 rank 独立串行预热
    // cuDNN Frontend 的 kernel/plan cache 是 per-device 的，必须每卡独立填充。
    // 串行执行避免 cuDNN 全局锁竞争，预热完成后通过 s_amp_*_caches 在捕获期纯读命中。
    // =====================================================================
    const bool is_cuda = (contexts[0]->is_gpu());
    if (is_cuda && K > 0) {
#ifdef TR_USE_CUDA
        for (int rank = 0; rank < num_ranks; ++rank) {
            int gpu_id = contexts[rank]->device_id();
            cudaSetDevice(gpu_id);
            contexts[rank]->set_rank(rank);

            TR_LOG_INFO("graph") << "[B2] cuDNN warmup on rank " << rank
                                 << " (GPU " << gpu_id << ") for "
                                 << K << " unique graphs";

            for (int32_t k = 0; k < K; ++k) {
                const auto& key = key_by_idx[k];
                if (!key.cg) continue;

                auto it = key_to_mp.find(key);
                TR_CHECK(it != key_to_mp.end() && it->second != nullptr, ValueError,
                         "No MemoryPlan for key[" << k << "]");
                const MemoryPlan* mp = it->second;
                contexts[rank]->set_memory_plan(mp);

                for (const auto& node : key.cg->nodes(key.gid)) {
                    if (node.kind != GraphNode::Kind::COMPUTE) continue;
                    if (!require_warmup(node.compute_op)) continue;
                    warmup_single_cudnn_op(node, *mp, *contexts[rank]);
                }
            }
            cudaDeviceSynchronize();
        }
        TR_LOG_INFO("graph") << "[B2] cuDNN warmup completed on all "
                             << num_ranks << " ranks";
#endif
    }
```

**设计理由**：
1. **必须传 `contexts[rank]`**：`warmup_single_cudnn_op` → `entry.launch_cuda` → `launch_relu_amp_fwd_cuda` 中调用 `ctx.cudnn_handle(StreamKind::COMP_1)` 获取 handle。传入不同的 context 才能获取不同的 handle，从而填充不同的缓存条目。
2. **必须 `set_memory_plan(mp)`**：预热中 `launch_relu_amp_fwd_cuda` 调用 `ctx.ptr_at(...)` 获取设备指针。`ptr_at()` 依赖 `current_mp_` 和 `rank_for_context_` 来解析 DTensor offset。不同 key 可能对应不同的 MemoryPlan（DeepLearningTask 中不同变体可能有不同的 mp），所以需要按 key 设置。
3. **必须 `set_rank(rank)`**：`ptr_at()` 使用 `rank_for_context_` 索引 ArenaKeeper 中正确的设备内存池。
4. **注释更新**：从"GPU 0单线程串行"更正为"每个 rank 独立串行预热"。

### 修改 3（可选防御）：[relu_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp) — rebuild 防御检测

在 `launch_relu_amp_fwd_cuda` 和 `launch_relu_amp_bwd_cuda` 中增加捕获期 rebuild 的防御检测：

**位置**：[relu_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp#L273) 和 [relu_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp#L338)，`if (rebuild)` 块的开头。

**插入**：
```cpp
if (rebuild) {
#ifndef NDEBUG
    {
        cudaStreamCaptureStatus cap_status;
        cudaError_t cap_err = cudaStreamIsCapturing(
            static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1)),
            &cap_status);
        if (cap_err == cudaSuccess && cap_status != cudaStreamCaptureStatusNone) {
            TR_LOG_ERROR("relu") << "[AMP] Rebuilding cuDNN FE graph inside CUDA Graph capture! "
                                 << "handle=" << handle << " rank=" << ctx.rank_for_context()
                                 << " This indicates warmup did not cover this rank/handle. "
                                 << "Warmup MUST be performed for ALL ranks before capture.";
        }
    }
#endif
    auto cache = std::make_unique<AmpReluFwdCache>();
    // ... 原有逻辑不变 ...
```

同样逻辑插入到 `launch_relu_amp_bwd_cuda` 的 rebuild 块中。

**作用**：debug 构建中，如果因代码回归导致捕获期间触发 rebuild，会在 `TR_LOG_ERROR` 中明确提示根本原因（预热未覆盖），而非静默失败为 code 11。这是一个防御层，修改 1+2 正确实施后不会触发。

### 修改汇总

| 文件 | 行数 | 修改类型 | 必需性 |
|------|------|---------|:------:|
| [task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp) L270-301 | ~30行 | 替换 warmup 段 | **必需** |
| [captured_graph.cpp](file:///r:/renaissance/src/graph/captured_graph.cpp) L216-247 | ~30行 | 替换 Phase B2 段 | **必需** |
| [relu_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cpp) L273, L338 | +8行×2 | 插入防御检测 | 推荐 |

---

## 五、线程安全分析

### 5.1 `s_amp_fwd_caches` 的并发访问

| 阶段 | 访问模式 | 线程 | 安全 |
|------|---------|------|:----:|
| Phase B2 warmup | 写（insert） | 主线程串行 | ✅ |
| Phase B3 rank 0 capture | 读（find） | 主线程 | ✅ |
| Phase B3 rank 1+ capture | 读（find） | 多线程并行，各自访问**不同 key** | ✅ |

**关键**：Phase B2 预热完成后，`s_amp_fwd_caches` 有 N 个条目（N = num_ranks）。Phase B3 并行捕获期间，每个线程使用自己的 `handle` 查询自己的条目（`cudnnHandle_t` 值不同 → 不同 key）。对 `std::unordered_map` 的并发读是安全的（C++17 标准保证）。没有线程会写入 map（所有条目已在预热阶段插入完毕）。

### 5.2 冷启动路径防御

如果因代码回归导致 Phase B3 期间出现 cache miss（map 中不存在某个 handle），补丁的 `insert` 操作会与并行读产生 data race。修改 3 的防御检测可以在 debug 构建中提前暴露此问题。生产环境中，修改 1+2 确保此路径不会触发。

---

## 六、验证计划

### 6.1 编译

```bash
# Linux
cd build
cmake --build . --config Release -j$(nproc)
# 或
ninja -j30 test_relu_fwd_bwd
```

### 6.2 功能测试（全部应 PASS）

```bash
# 1 GPU AMP
CUDA_VISIBLE_DEVICES=0 ./test_relu_fwd_bwd --amp --no-gen

# 2 GPU AMP
CUDA_VISIBLE_DEVICES=0,1 ./test_relu_fwd_bwd --amp --no-gen

# 8 GPU AMP
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./test_relu_fwd_bwd --amp --no-gen

# CPU 模式
./test_relu_fwd_bwd --cpu

# GPU FP32 模式
CUDA_VISIBLE_DEVICES=0 ./test_relu_fwd_bwd --gpu --no-gen
```

### 6.3 回归检查项

| 检查项 | 预期 |
|--------|------|
| 单 GPU AMP 不退化 | PASS |
| FP32 CUDA kernel 路径不受影响 | PASS |
| CPU 模式不受影响 | PASS |
| 预热日志显示所有 rank | `[WARMUP] Rank 0 ... Rank 1 ...` |
| 多卡运行时无 code 11 | PASS |

### 6.4 预期日志变化

修复前：
```
[b2] cuDNN warmup on GPU 0 for 1 unique graphs
[b2] cuDNN warmup completed
[b3] Capturing rank 0 in main thread
[b3] Capturing ranks 1~7 in parallel threads
→ ❌ code 11 on rank 1
```

修复后：
```
[b2] cuDNN warmup on rank 0 (GPU 0) for 1 unique graphs
[b2] cuDNN warmup on rank 1 (GPU 1) for 1 unique graphs
...
[b2] cuDNN warmup completed on all 8 ranks
[b3] Capturing rank 0 in main thread
[b3] Capturing ranks 1~7 in parallel threads
→ ✅ 所有 rank 捕获成功
```

---

## 七、风险与回滚

### 7.1 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|:----:|------|------|
| 预热时间线性增长（N×t_warmup） | 高 | 低 | warmup 在 MLPerf 不计时阶段；ReLU build_plans ~1-2ms |
| pre_capture 多个 MemoryPlan 切换时 `set_memory_plan` 开销 | 低 | 低 | 预热已完成，不影响 Phase C 热路径 |
| Debug 防御检测的 `cudaStreamIsCapturing` 额外开销 | 仅在 debug | 低 | Release 构建中 `#ifndef NDEBUG` 跳过 |

### 7.2 回滚方案

如果此修复引发新问题，可回滚到旧预热逻辑：

```cpp
// 回滚：仅预热 GPU 0（旧行为）
cudaSetDevice(contexts[0]->device_id());
warmup_single_cudnn_op(node, *mp, *contexts[0]);
```

回滚后单卡仍可用，但多卡会恢复 code 11 崩溃。

---

## 八、架构启示

### 8.1 Warmup.md 需要更新

[Warmup.md](file:///r:/renaissance/ULTIMATE_DESIGN/Warmup.md) 的三段式策略核心不变（预热→串行捕获→并行捕获），但具体假设需要修正：

| 原假设 | 修正 |
|--------|------|
| "cuDNN engine cache 是进程级全局" | 对 Legacy API 成立；**cuDNN Frontend 的 kernel/plan cache 是 per-device 的** |
| "GPU 0 预热即可覆盖所有 GPU" | **每个 rank 必须独立预热**，因为 per-device 的 RTC 编译结果不可跨设备共享 |

### 8.2 算子开发的最佳实践

所有使用 cuDNN Frontend API 的算子必须遵循：

1. **在 `launch_cuda` 中使用 per-handle 缓存**（`unordered_map<cudnnHandle_t, ...>`），不要使用全局 singleton
2. **在 warmup 阶段为每个 handle 独立构建 graph + execute**，确保 per-device kernel cache 完整填充
3. **捕获阶段只调用 `execute()`**，不调用 `build_plans`
4. **所有 `build_plans(HEURISTICS_CHOICE)` 必须在 `cudaStreamBeginCapture` 之前完成**

### 8.3 未来优化方向

| 优化 | 说明 |
|------|------|
| **cuDNN FE 1.8.0+ `populate_cuda_graph`** | 在 capture 窗口外构建 graph，直接填充到 CUDA Graph 中，完全绕过 RTC-in-capture 问题（仅限支持 `CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API` 标记的引擎） |
| **Deviceless AOT 编译** | 将 Execution Plan 编译为平台无关的中间表示并持久化，在后续运行中直接加载（需 cuDNN FE 版本支持） |
| **并行预热** | 每个 rank 在不同线程中并行预热（而非当前串行），可减少预热时间，但需确认 cuDNN per-device 隔离足以避免竞态 |
| **cache 生命周期管理** | 在 `DeviceContext::~DeviceContext()` 中清理 `s_amp_fwd_caches` 中对应 handle 的条目，避免静态 map 无限增长 |

### 8.4 跨算子推广

此修复模式（per-handle map + 所有 rank 独立预热）适用于**所有使用了 cuDNN Frontend API 的算子**。当前仅在 `relu_op.cpp` 中实施了 per-handle map。未来其他 Frontend 算子（Conv fusion、BN 等）应遵循相同模式。`warmup_single_cudnn_op` 已经通过 `DeviceContext& ctx` 参数支持多 rank，只需确保 warmup 循环覆盖所有 rank 即可。

---

## 附：完整修改的代码 diff（精确行号）

### diff 1：task_base.cpp L268-L301

```
@@ 268,34 @@
 void TaskBase::compile_capture_simple() {
 #ifdef TR_USE_CUDA
-    // cuDNN warmup（与 pre_capture 的 Phase B2 一致）
     if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
-        int master_gpu = backend_->contexts[0]->device_id();
-        cudaSetDevice(master_gpu);
-        cudaGetLastError();  // 清除 ArenaKeeper 初始化可能遗留的粘滞错误
-        cudaDeviceSynchronize();
-
-        // 预热前必须先设置 rank 和 memory_plan，否则 ptr_at() 会失败
-        backend_->contexts[0]->set_rank(0);
-        backend_->contexts[0]->set_memory_plan(&memory_plan_);
-
+        // pinned memory 预分配 ...
         for (const auto& [name, entry] : named_graphs_) {
             for (const auto& node : entry.graph.linear_nodes()) {
-                if (node.kind != GraphNode::Kind::COMPUTE) continue;
-                if (!require_warmup(node.compute_op)) continue;
-                warmup_single_cudnn_op(node, memory_plan_, *backend_->contexts[0]);
+                if (node.kind != GraphNode::Kind::RANGE) continue;
+                ... get_dtensor_pinned_buffer ...
             }
         }
-        cudaDeviceSynchronize();
-        cudaGetLastError();
 
-        // 预分配 RANGE_H2D_COPY_DTENSOR ...
+        // cuDNN warmup：每个 rank 独立串行预热
+        for (int rank = 0; rank < num_gpus_; ++rank) {
+            DeviceContext& warm_ctx = *backend_->contexts[rank];
+            int gpu_id = warm_ctx.device_id();
+            cudaSetDevice(gpu_id);
+            cudaGetLastError();
+            cudaDeviceSynchronize();
+
+            warm_ctx.set_rank(rank);
+            warm_ctx.set_memory_plan(&memory_plan_);
+
+            for (const auto& [name, entry] : named_graphs_) {
+                for (const auto& node : entry.graph.linear_nodes()) {
+                    if (node.kind != GraphNode::Kind::COMPUTE) continue;
+                    if (!require_warmup(node.compute_op)) continue;
+                    warmup_single_cudnn_op(node, memory_plan_, warm_ctx);
+                }
+            }
+            cudaDeviceSynchronize();
+            cudaGetLastError();
+        }
     }
 #endif
```

### diff 2：captured_graph.cpp L216-L247

```
@@ 216,32 @@
-    // Phase B2: cuDNN预热（仅CUDA，GPU 0单线程串行）
+    // Phase B2: cuDNN预热 — 每个 rank 独立串行预热
     const bool is_cuda = (contexts[0]->is_gpu());
     if (is_cuda && K > 0) {
-        int master_gpu = contexts[0]->device_id();
-        cudaSetDevice(master_gpu);
-        TR_LOG_INFO("graph") << "[B2] cuDNN warmup on GPU " << master_gpu ...
-        for (int32_t k = 0; k < K; ++k) {
-            ...
-            warmup_single_cudnn_op(node, *mp, *contexts[0]);
+        for (int rank = 0; rank < num_ranks; ++rank) {
+            int gpu_id = contexts[rank]->device_id();
+            cudaSetDevice(gpu_id);
+            contexts[rank]->set_rank(rank);
+            TR_LOG_INFO("graph") << "[B2] cuDNN warmup on rank " << rank ...
+            for (int32_t k = 0; k < K; ++k) {
+                ...
+                contexts[rank]->set_memory_plan(mp);
+                warmup_single_cudnn_op(node, *mp, *contexts[rank]);
+            }
+            cudaDeviceSynchronize();
         }
-        cudaDeviceSynchronize();
-        TR_LOG_INFO("graph") << "[B2] cuDNN warmup completed";
+        TR_LOG_INFO("graph") << "[B2] cuDNN warmup completed on all " << num_ranks << " ranks";
     }
```