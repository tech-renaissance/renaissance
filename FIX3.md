# FIX3 — ReLU AMP 多卡捕获失败修复方案

> 基于 FIX.md 全量分析 + 当前代码库实际状态检查

---

## 一、当前代码状态检查

### 1.1 `relu_op.cpp` — per-handle map 已落地

```cpp
// relu_op.cpp:L73-74
static std::unordered_map<cudnnHandle_t, std::unique_ptr<AmpReluFwdCache>> s_amp_fwd_caches;
static std::unordered_map<cudnnHandle_t, std::unique_ptr<AmpReluBwdCache>> s_amp_bwd_caches;
```

✅ **已完成**：全局 `unique_ptr` 已改为 per-handle `unordered_map`。这消除了多线程并发覆盖 cache 的竞态风险，rank 0 和 rank 1 各自持有独立条目。

### 1.2 `task_base.cpp` — warmup 仍只覆盖 rank 0

```cpp
// task_base.cpp:L281-286（compile_capture_simple）
for (const auto& [name, entry] : named_graphs_) {
    for (const auto& node : entry.graph.linear_nodes()) {
        if (node.kind != GraphNode::Kind::COMPUTE) continue;
        if (!require_warmup(node.compute_op)) continue;
        warmup_single_cudnn_op(node, memory_plan_, *backend_->contexts[0]);  // ← 只传 rank 0
    }
}
```

❌ **仍有问题**：`compile_capture_simple()` 的预热阶段只调用 `backend_->contexts[0]`，只为 handle_0 建立 cache。

### 1.3 `captured_graph.cpp` — pre_capture 同样只覆盖 rank 0

```cpp
// captured_graph.cpp:L237-240（pre_capture Phase B2）
for (const auto& node : key.cg->nodes(key.gid)) {
    if (node.kind != GraphNode::Kind::COMPUTE) continue;
    if (!require_warmup(node.compute_op)) continue;
    warmup_single_cudnn_op(node, *mp, *contexts[0]);  // ← 只传 rank 0
}
```

❌ **仍有问题**：DeepLearningTask 的 `pre_capture()` Phase B2 同样只预热 `contexts[0]`。

---

## 二、问题精确定位

### 2.1 现象

| 场景 | 结果 |
|------|------|
| 单卡 (`CUDA_VISIBLE_DEVICES=0`) | ✅ PASS |
| 双卡 (`CUDA_VISIBLE_DEVICES=0,1`) | ❌ 失败，code 11 |
| 八卡 (`CUDA_VISIBLE_DEVICES=0~7`) | ❌ 失败，code 11 |

错误栈：
```
cuDNN FE [execute AMP FWD]: s_amp_fwd_caches[handle]->graph->execute(...) failed with code: 11
at relu_op.cpp :: launch_relu_amp_fwd_cuda()
```

### 2.2 根因

虽然 `s_amp_fwd_caches` 已经是 per-handle map，但 **map 在 capture 阶段之前只被 handle_0 填充**。

`compile_capture_simple()` 执行流：

```
Phase A: warmup（task_base.cpp:L270-288）
    └── 只遍历 backend_->contexts[0]
    └── 调用 warmup_single_cudnn_op → entry.launch_cuda → launch_relu_amp_fwd_cuda
    └── s_amp_fwd_caches[handle_0] = build_amp_fwd_graph(handle_0, ...)  ✅ 填充 handle_0
    └── execute 成功，cudaDeviceSynchronize 返回

Phase B: capture（task_base.cpp:L305-340）
    └── 串行循环 rank = 0 .. num_gpus_-1
    └── rank 0: CapturedGraph::capture_cuda()
        └── replay RELU_AMP_FWD → launch_relu_amp_fwd_cuda
        └── map 命中 handle_0 → 直接 execute → ✅ 成功
    └── rank 1: CapturedGraph::capture_cuda()
        └── replay RELU_AMP_FWD → launch_relu_amp_fwd_cuda
        └── map 未命中 handle_1 → rebuild!
            ├── build_amp_fwd_graph(handle_1, ...)  ← 在 capture mode 内
            │   └── finalize_cudnn_graph → build_plans(HEURISTICS_CHOICE)
            │       └── 可能触发 RTC kernel 编译 / workspace 分配
            └── s_amp_fwd_caches[handle_1]->graph->execute(handle_1, vp, nullptr)
                └── ❌ 失败，code 11
```

**code 11 的本质**：cuDNN Frontend 的 `build_plans` 在 CUDA Graph Capture 窗口内被触发，导致 kernel 编译 / workspace 分配等操作落在 `BeginCapture`/`EndCapture` 之间，违反了 CUDA Graph 的捕获规则。

### 2.3 为什么 per-handle map 没有解决问题

小伙伴之前的修复（`unique_ptr` → `unordered_map`）解决了**竞态覆盖**问题，但没有解决**缓存未命中**问题。

per-handle map 让 rank 0 和 rank 1 不再互相覆盖，但 rank 1 在 capture 阶段仍然面临 **cache miss → rebuild → 在 capture mode 内 build_plans** 的致命路径。

---

## 三、修复方案

### 3.1 核心原则（来自 Warmup.md + 专家共识）

> **所有 `build_plans` 必须在 `cudaStreamBeginCapture` 之前完成。**
> 
> Capture 窗口内只允许纯 kernel launch replay，禁止任何计划构建、内存分配、同步操作。

### 3.2 方案：预热阶段串行覆盖所有 rank

将 `compile_capture_simple()` 和 `pre_capture()` 的 Phase B2 从"只预热 GPU 0"改为"串行预热所有 GPU"。

**为什么串行安全？**
- Warmup.md 指出 cuDNN engine cache 有全局锁，**并发** `build_plans` 会 serialize 且可能损坏 cache
- 但**串行**循环每个 rank 的 `build_plans` 是安全的：前一个 handle 的 cache 写完后，再写下一个 handle，无竞态

**为什么每个 rank 都需要预热？**
- 专家 KM 指出：cuDNN Frontend 的 Kernel Cache、Execution Plan、RTC 编译结果都是 **per-device** 的
- GPU 0 上编译的 CUBIN 对 GPU 1 完全不可见
- 即使同型号 GPU，handle 也是独立的，`ExecutionPlan` 绑定到特定 handle

### 3.3 实施细节

#### 修改 1：`compile_capture_simple()`（SimpleTask 路径）

```cpp
void TaskBase::compile_capture_simple() {
#ifdef TR_USE_CUDA
    if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
        cudaGetLastError();
        cudaDeviceSynchronize();

        // 预分配 pinned memory（保持原有逻辑）
        for (const auto& [name, entry] : named_graphs_) {
            for (const auto& node : entry.graph.linear_nodes()) {
                if (node.kind != GraphNode::Kind::RANGE) continue;
                if (node.range_op != RangeOp::RANGE_H2D_COPY_DTENSOR) continue;
                if (node.output_ranges.empty()) continue;
                const auto& seg = node.output_ranges[0];
                get_dtensor_pinned_buffer(seg.offset, seg.size);
            }
        }

        // === 修改：对所有 rank 串行预热 ===
        for (int rank = 0; rank < num_gpus_; ++rank) {
            DeviceContext& ctx = *backend_->contexts[rank];
            int gpu_id = ctx.device_id();
            cudaSetDevice(gpu_id);
            cudaGetLastError();
            cudaDeviceSynchronize();

            ctx.set_rank(rank);
            ctx.set_memory_plan(&memory_plan_);

            for (const auto& [name, entry] : named_graphs_) {
                for (const auto& node : entry.graph.linear_nodes()) {
                    if (node.kind != GraphNode::Kind::COMPUTE) continue;
                    if (!require_warmup(node.compute_op)) continue;
                    warmup_single_cudnn_op(node, memory_plan_, ctx);
                }
            }
            cudaDeviceSynchronize();
            cudaGetLastError();
            TR_LOG_INFO("task") << "[WARMUP] Rank " << rank << " (GPU " << gpu_id << ") warmed";
        }
    }
#endif
    // capture 阶段保持原有逻辑不变 ...
}
```

#### 修改 2：`pre_capture()`（DeepLearningTask 路径）

```cpp
// captured_graph.cpp Phase B2
const bool is_cuda = (contexts[0]->is_gpu());
if (is_cuda && K > 0) {
#ifdef TR_USE_CUDA
    // === 修改：对所有 rank 串行预热 ===
    for (int rank = 0; rank < num_ranks; ++rank) {
        int gpu_id = contexts[rank]->device_id();
        cudaSetDevice(gpu_id);

        TR_LOG_INFO("graph") << "[B2] cuDNN warmup on GPU " << gpu_id
                             << " for " << K << " unique graphs";

        for (int32_t k = 0; k < K; ++k) {
            const auto& key = key_by_idx[k];
            if (!key.cg) continue;

            auto it = key_to_mp.find(key);
            TR_CHECK(it != key_to_mp.end() && it->second != nullptr, RuntimeError,
                     "No MemoryPlan for key[" << k << "]");
            const MemoryPlan* mp = it->second;

            for (const auto& node : key.cg->nodes(key.gid)) {
                if (node.kind != GraphNode::Kind::COMPUTE) continue;
                if (!require_warmup(node.compute_op)) continue;
                warmup_single_cudnn_op(node, *mp, *contexts[rank]);  // ← 传 contexts[rank]
            }
        }
        cudaDeviceSynchronize();
    }
    TR_LOG_INFO("graph") << "[B2] cuDNN warmup completed for all ranks";
#endif
}
```

#### 修改 3：`launch_relu_amp_fwd_cuda` 防御性断言（可选但推荐）

在 capture 阶段若仍然发生 cache miss（未来代码回归），直接报错而不是在 capture mode 内静默 rebuild：

```cpp
// relu_op.cpp launch_relu_amp_fwd_cuda
auto it_fwd = s_amp_fwd_caches.find(handle);
bool rebuild = (it_fwd == s_amp_fwd_caches.end())
    || it_fwd->second->N != N || ...;

if (rebuild) {
    // 防御：在 capture mode 内禁止 rebuild
    cudaStreamCaptureStatus capture_status;
    cudaStream_t dummy_stream = nullptr;
    cudaStreamIsCapturing(s, &capture_status);
    if (capture_status != cudaStreamCaptureStatusNone) {
        TR_DEVICE_ERROR("RELU AMP cache miss during CUDA Graph capture. "
                        "Handle " << handle << " was not warmed. "
                        "Ensure all ranks are warmed before capture.");
    }

    auto cache = std::make_unique<AmpReluFwdCache>();
    ...
}
```

> 注：此断言在 debug 构建中启用，release 构建中可省略以节省一次 API 调用。

---

## 四、验证计划

| 步骤 | 命令 | 预期结果 |
|------|------|----------|
| 1. 编译 | `ninja -j30 test_relu_fwd_bwd` | 通过 |
| 2. 单卡 | `CUDA_VISIBLE_DEVICES=0 ./test_relu_fwd_bwd --amp --no-gen` | PASS |
| 3. 双卡 | `CUDA_VISIBLE_DEVICES=0,1 ./test_relu_fwd_bwd --amp --no-gen` | PASS |
| 4. 八卡 | `./test_relu_fwd_bwd --amp --no-gen` | PASS |
| 5. CPU 模式 | `./test_relu_fwd_bwd --cpu` | PASS |
| 6. GPU FP32 | `./test_relu_fwd_bwd --gpu` | PASS |

---

## 五、风险与回滚

| 风险 | 缓解 |
|------|------|
| 预热时间线性增长（N 张卡 × 原预热时间） | 可接受：warmup 在 MLPerf 不计时阶段；实际 ReLU build_plans 极快（~1-2ms） |
| 串行预热导致大模型启动慢 | 未来优化：Conv 等重算子可用 `populate_cuda_graph`（cuDNN FE 1.8.0+） |
| 修改影响 DeepLearningTask 路径 | `pre_capture()` 改动经过相同验证流程 |

---

## 六、总结

| 问题 | 状态 |
|------|------|
| 全局 unique_ptr 竞态覆盖 | ✅ 已修复（per-handle map） |
| 只预热 rank 0 导致 rank 1+ capture 时 cache miss | ❌ **待修复（本方案）** |
| capture mode 内触发 build_plans | ❌ **待修复（本方案）** |

**一句话**：把 warmup 循环从 `contexts[0]` 扩展到 `contexts[0..N-1]`，让每个 handle 在 BeginCapture 前都有缓存。
