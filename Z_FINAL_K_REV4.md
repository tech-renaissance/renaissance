# Z_FINAL_K.md 第四次交叉审查 — 修改意见

> **审查范围**: Z_FINAL_K.md 全文（1071 行）
> **基线源码**: TR4 v4.20.1/v4.20.2，C++17
> **审查方法**: 逐节读取 Z_FINAL_K.md 后，回查 `task_base.cpp` / `deep_learning_task.cpp` / `device_context.cpp` / `computation_graph.h` / `preprocessor.h` 等关键源码进行交叉验证

---

## 审查结论

Z_FINAL_K.md 整体架构正确，核心决策（`active_memory_plan_` 指针 + 无状态 LR 查询 + `cudaMallocHost` 锁页内存 + MMP0 四阶段重叠）经源码验证成立。

**本次审查新发现 6 处问题：1 个 P0（必崩）、4 个 P1（功能缺失/路径断裂）、1 个 P2（健壮性）。** 均已在现有源码中找到直接证据。

---

## 🔴 P0: `DeviceContext::set_memory_plan()` 完全遗漏

### 问题描述

Z_FINAL_K.md §7.3 `run_train_epoch_gpu()` 中使用了 `ctx.ptr_at(lr_dtensor_id_)`：

```cpp
void* lr_dev_ptr = ctx.ptr_at(lr_dtensor_id_);   // line ~706
```

`DeviceContext::ptr_at(int)` 的实现（`src/backend/device_context.cpp:179-187`）要求 `current_mp_ != nullptr`：

```cpp
void* DeviceContext::ptr_at(int dtensor_id) const noexcept {
    TR_DEBUG_CHECK(current_mp_ != nullptr, RuntimeError,
                  "ptr_at: no MemoryPlan set (call set_memory_plan first)");
    const DTensor& dt = current_mp_->get_dtensor(dtensor_id);
    return ArenaKeeper::instance().ptr_at(rank_for_context_, static_cast<size_t>(dt.offset()));
}
```

但当前源码中，`set_memory_plan()` **仅在 SimpleTask 路径被调用**（`task_base.cpp:337, 377, 492`）：

```cpp
// compile_capture_simple() 内部
warm_ctx.set_memory_plan(&memory_plan_);        // line 337
dc.set_memory_plan(&memory_plan_);              // line 377
ctx.set_memory_plan(&memory_plan_);             // line 492
```

**DeepLearningTask 路径（`compile_impl()` 的 else 分支，`task_base.cpp:237-249`）完全没有调用 `set_memory_plan()`。**

`pre_capture()` 在**捕获阶段**会通过 `captured_graph.cpp:243` 为 CapturedGraph 内部的临时 context 设置 memory plan，但这不会持久化到 `TaskBase::backend_->contexts[]` 中。运行时 `run_train_epoch_gpu()` 使用的 `context(rank)` 返回的是 `backend_->contexts[rank]`，其 `current_mp_` 仍为 `nullptr`。

### 后果

`run_train_epoch_gpu()` 执行到 `ctx.ptr_at(lr_dtensor_id_)` 时触发 `TR_DEBUG_CHECK` 失败，进程崩溃。

### 最小侵入修复

在 `compile_impl()` 的 DeepLearningTask 分支中，`pre_capture()` 之前为每个 DeviceContext 设置 memory plan：

```cpp
// === 在 Z_FINAL_K.md §5.5 的 compile_impl() 集成版中，
//     "GraphAtlas atlas = dl->build_graph_atlas();" 之前插入 ===

for (int rank = 0; rank < num_gpus_; ++rank) {
    backend_->contexts[rank]->set_rank(rank);
    backend_->contexts[rank]->set_memory_plan(active_memory_plan_);
}
```

> **为什么放在这里**：`compile_alloc_hardware()` 已创建 `backend_->contexts`，`on_prepare()` 已切换 `active_memory_plan_`，此时设置是最佳时机。设置后，后续 `pre_capture()`、warmup、以及运行时 `run_train_epoch_gpu()` 均能正确解析 DTensor 地址。

---

## 🟡 P1-1: `compile_debug_print_and_return()` 仍用空 `memory_plan_`

### 问题描述

Z_FINAL_K.md 将 `active_memory_plan_` 在 `on_prepare()` 中切换到真实 plan，但 `compile_debug_print_and_return()`（`task_base.cpp:566-610`）仍使用 `memory_plan_`：

```cpp
void TaskBase::compile_debug_print_and_return() {
    memory_plan_.validate();                    // ← 空实例
    // ...
    for (const auto& dt : memory_plan_.dtensors()) {  // ← 空实例，size=0
        // ...
    }
}
```

当 `debug_mode_ = true`（dry-run）时，`compile_impl()` 在 `on_prepare()` 之后调用 `compile_debug_print_and_return()`。此时 `active_memory_plan_` 已指向真实 plan，但函数仍读取空实例，导致 debug 输出无任何 DTensor 信息。

### 最小侵入修复

将 `compile_debug_print_and_return()` 中所有 `memory_plan_.` 替换为 `active_memory_plan_->`：

```cpp
void TaskBase::compile_debug_print_and_return() {
    active_memory_plan_->validate();
    std::cout << active_memory_plan_->dump_layout() << "\n";
    for (const auto& dt : active_memory_plan_->dtensors()) {
        // ... 不变
    }
}
```

> 仅 3 处替换，零行为影响（SimpleTask 下两者指向同一实例）。

---

## 🟡 P1-2: StagingParamPool 自动检测使用 `linear_nodes()` 但 DL 图不存于此

### 问题描述

Z_FINAL_K.md §5.4 的 StagingParamPool 自动检测逻辑：

```cpp
for (const auto& n : dl->train_cg_->linear_nodes()) {
    if (n.kind == GraphNode::Kind::RANGE &&
        n.range_op == RangeOp::RANGE_H2D_COPY_DTENSOR) {
        need = true; break;
    }
}
```

**问题**：DeepLearningTask 的 `ComputationGraph` 由 Compiler 通过 `append(GraphId, GraphNode)` 填充 bucket（`graphs_[gid]`），**不经过 `add_linear_node()`**。`linear_nodes()` 返回的是 `linear_nodes_`，在 DeepLearningTask 场景下**为空向量**。

### 源码证据

`computation_graph.h:230-241`：
```cpp
void append(GraphId gid, const std::vector<GraphNode>& nodes) {
    auto& bucket = graphs_[static_cast<size_t>(gid)];
    bucket.insert(bucket.end(), nodes.begin(), nodes.end());
}
```

`linear_nodes()`（line 277）只返回 `linear_nodes_`，与 `append()` 的 bucket 路径完全独立。

### 后果

StagingParamPool 自动检测逻辑对 DeepLearningTask **永远不会触发**。若用户后续想切到方案 A（StagingParamPool + RANGE_H2D_COPY_DTENSOR），`has_staging_params()` 始终为 false，无法使用已有基础设施。

### 最小侵入修复

改为遍历所有 GraphId bucket：

```cpp
bool need = false;
if (dl->train_cg_) {
    for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
        for (const auto& n : dl->train_cg_->nodes(static_cast<GraphId>(gi))) {
            if (n.kind == GraphNode::Kind::RANGE &&
                n.range_op == RangeOp::RANGE_H2D_COPY_DTENSOR) {
                need = true; break;
            }
        }
        if (need) break;
    }
}
```

> 遍历 21 个 GraphId × 各 bucket 节点数，总节点数与 linear_nodes 同级，无性能问题。

---

## 🟡 P1-3: `run_train_epoch_gpu()` FIRST_FWD 空指针风险

### 问题描述

Z_FINAL_K.md §7.3 的 `run_train_epoch_gpu()` 中：

```cpp
auto gf_a  = G(FIRST_FWD_A),   gf_b  = G(FIRST_FWD_B);
// ...
cudaGraphLaunch(gf_a, s_c1);    // batches==1 分支，line ~721
cudaGraphLaunch(gf,  s_c1);     // 中间 batch 循环，line ~754
```

若 Compiler 在 **non-progressive 模式** 下未生成 `FIRST_FWD_A/B` 图（`train_cg_->nodes(FIRST_FWD_A).empty()`），则 `build_graph_atlas()` 会跳过该 GraphId，`build_exec_table()` 中 `resolve()` 返回 `nullptr`，`gf_a`/`gf_b` 为 `nullptr`。

`cudaGraphLaunch(nullptr, stream)` 返回 `cudaErrorInvalidValue`，触发运行时错误。

### 最小侵入修复

在 `run_train_epoch_gpu()` 的 `batches==1` 分支和中间循环中，增加空指针检查：

```cpp
// Phase 1: ZERO_GRAD || FIRST_FWD
if (gzg) cudaGraphLaunch(gzg, s_up);
if (gf)  cudaGraphLaunch(gf,  s_c1);   // ← 加 if (gf) 保护
sync_comp(); sync_up();
```

同理，单 batch 分支中的 `gf_a`：
```cpp
if (gzg) cudaGraphLaunch(gzg, s_up);
if (gf_a) cudaGraphLaunch(gf_a, s_c1);  // ← 加保护
```

> 即使 Compiler 确实生成了 FIRST_FWD 图，空指针检查也是零开销防御性编程。

---

## 🟡 P1-4: `name_to_gid_` 未填充，`run_impl()` / `dry_run()` 路径断裂

### 问题描述

当前源码中 `build_simple_atlas()`（`task_base.cpp:616-655`）会填充 `name_to_gid_`：

```cpp
name_to_gid_[name] = gid;   // line 647
```

Z_FINAL_K.md 废弃 `build_simple_atlas()`，改用 `DeepLearningTask::build_graph_atlas()`，但 **新函数不设置 `name_to_gid_`**。

`run_impl()`（`task_base.cpp:713-762`）在 dry-run 和命名图启动路径中依赖 `name_to_gid_`：

```cpp
auto gid_it = name_to_gid_.find(name);      // line 715 / 733
TR_CHECK(gid_it != name_to_gid_.end(), ValueError, ...);
int32_t idx = captured_result_.atlas.index(0, gid_it->second);
```

虽然真实训练走 `run_gpu()`（不经过 `run_impl()`），但 **dry-run 调试路径**和任何通过 `run(name)` 启动单个图的场景都会失败。

### 最小侵入修复

在 `build_graph_atlas()` 末尾或 `on_prepare()` 中，为 `named_graphs_` 中的每个 name 填充一个代表性映射：

```cpp
// 在 build_graph_atlas() 末尾添加：
name_to_gid_.clear();
if (train_cg_) {
    name_to_gid_["train"] = GraphId::DEEP_FWD_BWD;  // 代表性 GraphId
}
if (infer_cg_) {
    name_to_gid_["inference"] = GraphId::INF_MAIN_A;
}
```

> 注意：DeepLearningTask 一个 name 对应多个子图，`run_impl()` 只能启动一个子图。这里填充的映射主要用于 dry-run 调试验证，不代表完整训练流程。更完善的方案是后续让 `run_impl()` 支持多子图序列，但那是 Phase 2 工作。

**或者更简单的方案**：保留 `build_graph_index()` 的调用（Z_FINAL_K.md 删除了它），但 `build_graph_index()` 本身也需要 `name_to_gid_`。所以填充 `name_to_gid_` 是前提。

---

## 🟢 P2: `lr_pinned_` 分配位置建议明确在 `compile_impl()` 末尾

### 问题描述

Z_FINAL_K.md §3.4 中 `lr_pinned_` 的分配代码：

```cpp
lr_pinned_.resize(num_gpus_);
for (int rank = 0; rank < num_gpus_; ++rank) {
    cudaMallocHost(&lr_pinned_[rank], sizeof(float));
}
```

但文档未明确说明这段代码应放在哪个函数的哪个位置。若放在 `run_gpu()` 中，每次 `run()` 调用都会重新分配（虽然 `run()` 通常只调用一次）；若放在 `compile_impl()` 末尾，则与 `num_gpus_` 和 `active_memory_plan_` 的确定时机一致。

### 建议

在 Z_FINAL_K.md §5.5 的 `compile_impl()` 集成版末尾、`init_all()` 之后插入：

```cpp
compile_mark_compiled();

if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
    (void)dl;
    init_all();
}

// === lr_pinned_ 分配（COMPILED 之后，run 之前）===
#ifdef TR_USE_CUDA
if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
    if (dl->lr_dtensor_id_ >= 0) {
        dl->lr_pinned_.resize(num_gpus_);
        for (int rank = 0; rank < num_gpus_; ++rank) {
            cudaError_t err = cudaMallocHost(&dl->lr_pinned_[rank], sizeof(float));
            TR_CHECK(err == cudaSuccess, DeviceError,
                     "cudaMallocHost failed for lr_pinned_: " << cudaGetErrorString(err));
        }
    }
}
#endif
// =====================================================

TR_LOG_INFO("task") << "[DBG] compile_impl: done";
```

> 好处：分配失败在 `compile()` 阶段即暴露，不延迟到 `run()`。且生命周期与 `DeepLearningTask` 绑定，析构时统一释放。

---

## 附：已确认 Z_FINAL_K.md 正确修正的问题（REV3 遗留）

| # | 问题 | 状态 | Z_FINAL_K.md 中修正位置 |
|---|------|:----:|------------------------|
| 1 | `init_all()` 在 `MEMORY_LOCKED` 阶段调用 | ✅ | §5.5 中移至 `compile_mark_compiled()` 之后 |
| 2 | `apply_sema_switch()` 延迟一个 epoch | ✅ | §8.1 中移至 `run_train_epoch_gpu()` 之前 |
| 3 | `fetch_lr_for_batch()` 全局步数 | ✅ | §6.4 中已用 `current_epoch_ * steps_per_epoch() + batch_id` |
| 4 | `run_gpu()` 功能缺失（early_stop/save/best/final summary） | ✅ | §8.1 完整实现 |
| 5 | `lr_pinned_` 非锁页内存 | ✅ | §3.4 明确为 `std::vector<float*>` + `cudaMallocHost` |
| 6 | `on_prepare()` finalize 幂等保护 | ✅ | §2.2 中 `if (!is_finalized())` |
| 7 | `zero()`/`randn()` 替换 | ✅ | §2.2 修改清单中已列出 |
| 8 | `memory_plan_.` → `active_memory_plan_->` 8 处 | ✅ | §2.2 修改清单 |

---

## 最小修改清单汇总

| # | 文件 | 修改内容 | 严重度 |
|---|------|---------|:------:|
| 1 | `src/task/task_base.cpp` | `compile_impl()` DeepLearningTask 分支中，`pre_capture()` 前为每个 `backend_->contexts[rank]` 调用 `set_rank(rank)` + `set_memory_plan(active_memory_plan_)` | 🔴 |
| 2 | `src/task/task_base.cpp` | `compile_debug_print_and_return()` 中 3 处 `memory_plan_.` → `active_memory_plan_->` | 🟡 |
| 3 | `src/task/task_base.cpp` | StagingParamPool 自动检测：将 `train_cg_->linear_nodes()` 改为遍历所有 `GraphId` 的 `train_cg_->nodes(gid)` | 🟡 |
| 4 | `src/task/deep_learning_task.cpp` | `run_train_epoch_gpu()` 中所有 `cudaGraphLaunch(gf_*, s_c1)` 前加 `if (gf_*)` 空指针保护 | 🟡 |
| 5 | `src/task/deep_learning_task.cpp` | `build_graph_atlas()` 末尾填充 `name_to_gid_`（至少 "train"/"inference" 映射到代表性 GraphId） | 🟡 |
| 6 | `src/task/task_base.cpp` | `compile_impl()` 末尾、`init_all()` 之后，为 `DeepLearningTask` 分配 `lr_pinned_` 锁页内存 | 🟢 |

---

*审查完成。以上 6 处修改均为最小侵入式，不改变 Z_FINAL_K.md 的核心架构决策。*
