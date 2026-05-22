# Z_FINAL_K_REV3.md — 对 Z_FINAL_K.md 的最终修改优化意见

> **审阅范围**: Z_FINAL_K.md 全文 + Z_FINAL_K_REV2.md 已发现问题 + `task_base.cpp` / `deep_learning_task.cpp` / `deep_learning_task.h` / `scheduler.cpp` / `computation_graph.h` 逐行交叉验证  
> **审阅结论**: Z_FINAL_K_REV2.md 发现的 8 个问题全部成立，并在此基础上新发现 **6 个 P0/P1 关键问题** 和 **4 个 P2 优化点**。  
> **严重度说明**: 🔴 = 实施必崩或功能严重错位 / 🟡 = 特定模式触发或功能缺失 / 🟢 = 文档/健壮性优化

---

## 一、P0 关键问题（实施必崩或核心功能错位）

### 1. run_gpu() 日志 get_current_lr() 永远返回初始 LR
**状态**: Z_FINAL_K_REV2.md 已发现，仍待修复  
**源码验证** (`src/algo/scheduler.cpp:102`): `step()` 先计算 `current_lr_ = compute_lr_at_step(current_step_)`，再递增 `current_step_`。Z_FINAL_K.md 废弃了 batch 内所有 `step()` 调用，导致 `current_step_` 永远停在 0，`get_current_lr()` 永远返回 `prepare()` 后的初始值。  
**影响**: `log_epoch_results()` 在所有 epoch 打印相同的 LR，与训练实际使用的 LR（通过 `fetch_lr_for_batch()` 计算）不一致。  
**修复**（方案 A，与 K 稿"无状态查询"原则一致）：
```cpp
// run_gpu() epoch 循环末尾
log_epoch_results(0.0f, val_loss, top1, top5, 0, 0,
                  fetch_lr_for_batch(0), sec);   // ← 替换 get_current_lr()
```

---

### 2. lr_pinned_ 类型未明确 — std::vector<float> 不是锁页内存
**状态**: Z_FINAL_K_REV2.md 已发现，仍待修复  
**问题**: Z_FINAL_K.md §6.6 代码写 `lr_pinned_[rank] = lr; cudaMemcpyAsync(..., &lr_pinned_[rank], ...)`。若 `lr_pinned_` 定义为 `std::vector<float>`，其内存为分页内存。Windows 上 `cudaMemcpyAsync` 从分页内存拷贝到设备会退化为同步行为（驱动隐式锁页再传输），CPU 阻塞等待完成，抵消 UPDATE stream 重叠优势。  
**修复**（明确类型 + 分配/释放 + 修正使用语法）：
```cpp
// deep_learning_task.h
#ifdef TR_USE_CUDA
std::vector<float*> lr_pinned_;   // 每个 rank 一个 cudaMallocHost 指针
#endif

// 在 compile() 后或 run_gpu() 初始化时分配：
#ifdef TR_USE_CUDA
lr_pinned_.resize(num_gpus_);
for (int rank = 0; rank < num_gpus_; ++rank) {
    cudaMallocHost(&lr_pinned_[rank], sizeof(float));
}
#endif

// run_train_epoch_gpu() 中修正语法：
*lr_pinned_[rank] = lr;
cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                cudaMemcpyHostToDevice, s_up);

// 析构时释放（见问题 24）：
#ifdef TR_USE_CUDA
for (auto* p : lr_pinned_) { if (p) cudaFreeHost(p); }
#endif
```

---

### 3. on_prepare() 中 memory_plan_ptr_->finalize() 缺少幂等保护
**状态**: Z_FINAL_K_REV2.md 已发现，仍待修复  
**源码验证** (`include/renaissance/graph/memory_plan.h:210`): `is_finalized()` 存在，但 `finalize()` 内部是否二次幂等未在文档中明确保证。Compiler 返回的 MemoryPlan 通常已 finalize。  
**修复**:
```cpp
if (!memory_plan_ptr_->is_finalized()) {
    memory_plan_ptr_->finalize();
}
```

---

### 4. 【新增】compile_impl() 中 init_all() 在 MEMORY_LOCKED 阶段调用，phase 检查失败
**状态**: 🔴 新发现，Z_FINAL_K_REV2.md 遗漏  
**源码验证** (`src/task/task_base.cpp:1226-1227`):
```cpp
void TaskBase::init_all() {
    check_phase(Phase::COMPILED, "init_all");  // ← 要求 COMPILED
```
Z_FINAL_K.md §5.3 建议在 `compile_alloc_hardware()` 之后、`compile_mark_compiled()` 之前插入 `init_all()`。但此时 `phase_ == Phase::MEMORY_LOCKED`（`on_prepare()` 设置），`check_phase()` 会抛出 `ValueError`。  
**影响**: `compile()` 在 `init_all()` 处崩溃，训练无法启动。  
**修复**: 将 `init_all()` 移到 `compile_mark_compiled()` 之后：
```cpp
void TaskBase::compile_impl(bool debug_mode) {
    // ... 前面不变 ...
    compile_alloc_hardware();

#ifdef TR_USE_CUDA
    if (GlobalRegistry::instance().using_gpu()) {
        for (int rank = 0; rank < num_gpus_; ++rank) {
            int dev = backend_->contexts[rank]->device_id();
            cudaSetDevice(dev);
            cudaMemset(ArenaKeeper::instance().ptr_at(rank, 0), 0,
                       active_memory_plan_->total_bytes());
            cudaDeviceSynchronize();  // 每设备独立同步（见问题 6）
        }
    }
#endif

    // === F2: Atlas 构建（原 §3.3）===
    if (is_simple_task()) {
        compile_capture_simple();
    } else {
        if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
            GraphAtlas atlas = dl->build_graph_atlas();
            std::vector<DeviceContext*> ctx_ptrs;
            for (auto& ctx : backend_->contexts)
                ctx_ptrs.push_back(ctx.get());
            captured_result_ = pre_capture(atlas, ctx_ptrs);
            dl->build_exec_table();   // 不再调用 build_graph_index()
        }
    }

    compile_mark_compiled();   // ← phase_ = COMPILED

    // === F4: 权重初始化（必须在 COMPILED 之后）===
    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        (void)dl;
        init_all();
    }
}
```
> **注意**: `init_all()` 在 `compile_mark_compiled()` 之后执行是安全的。CUDA Graph 捕获的是操作序列（kernel launch / memcpy），不依赖内存初始值。权重初始化只修改 ArenaKeeper 中的数据，不影响已捕获的 CUDA Graph Exec 对象。

---

### 5. 【新增】run_gpu() 中 apply_sema_switch() 从 epoch 开头移到了 epoch 末尾，SEMA 时序错位
**状态**: 🔴 新发现，Z_FINAL_K_REV2.md 遗漏  
**源码对比**:  
- 旧代码 (`deep_learning_task.cpp:618-620`): `if (use_sema_ && epoch > 0) apply_sema_switch();` 在 `run_train_epoch_gpu()` **之前**
- Z_FINAL_K.md §8.1: `if (use_sema_ && epoch > 0) apply_sema_switch();` 在 `log_epoch_results()` **之后**

**影响**: SEMA（将 EMA 权重复制回主模型）延迟一个 epoch 生效：
- epoch 0: 训练使用原始权重 → log → 不应用 SEMA
- epoch 1: 训练仍使用原始权重（SEMA 尚未应用）→ log → 应用 SEMA（应在 epoch 1 训练前应用）
- epoch 2: 训练使用 SEMA 权重（正确，但延迟了一个 epoch）

**修复**: 保持 `apply_sema_switch()` 在 `run_train_epoch_gpu()` 之前：
```cpp
for (int epoch = 0; epoch < epochs; ++epoch) {
    current_epoch_ = epoch;
    auto epoch_start = std::chrono::steady_clock::now();

    // SEMA 必须在训练前应用
    if (use_sema_ && epoch > 0) {
        apply_sema_switch();
    }

    std::thread prep_thread([&]() { prep.train(); });
    run_train_epoch_gpu();
    prep_thread.join();
    // ... 后续不变 ...
}
```

---

## 二、P1 建议修复（特定模式触发或功能缺失）

### 6. init() 中 UNIFORM 变体错误使用正态分布
**状态**: Z_FINAL_K_REV2.md 已发现，仍待修复  
**源码位置**: Z_FINAL_K.md §5.2  
**修复**: `XAVIER_UNIFORM` / `KAIMING_UNIFORM` 改用 `uniform_real_distribution`：
```cpp
case InitKind::XAVIER_UNIFORM:
case InitKind::KAIMING_UNIFORM: {
    std::default_random_engine rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-config.scale, config.scale);
    float* data = host.data<float>();
    for (int64_t i = 0; i < host.numel(); ++i)
        data[i] = dist(rng);
    break;
}
```

---

### 7. ArenaKeeper memset 后 cudaDeviceSynchronize() 只同步最后一个设备
**状态**: Z_FINAL_K_REV2.md 已发现，仍待修复  
**源码位置**: Z_FINAL_K.md §5.1  
**修复**: 循环内每个设备独立 sync（见问题 4 的修复代码中已包含）。

---

### 8. 单 batch 路径 AMP 检查与其他路径不一致
**状态**: Z_FINAL_K_REV2.md 已发现，仍待修复  
**源码位置**: Z_FINAL_K.md §7.3 单 batch / 中间 batch / last batch 三处  
**修复**: 三处统一只保留 `ggc`（`GRAD_CONVERT`）检查，`gcn`（`CAST_AND_CHECK`）与 `ggc` 解析自同一 GraphId，重复检查无意义。

---

### 9. 【新增】fetch_lr_for_batch() 在 step_by_batch 模式下传递 epoch 内相对 batch 索引
**状态**: 🟡 新发现，Z_FINAL_K_REV2.md 遗漏  
**源码验证** (`src/algo/scheduler.cpp:117-126`):
```cpp
float LRScheduler::get_lr_by_batch(int batch_id) const {
    return compute_lr_at_step(batch_id);  // batch_id = 全局步数索引
}
```
Z_FINAL_K.md §6.4 / §7.3 中 `fetch_lr_for_batch(batch)` 的 `batch` 是 epoch 内相对索引（`0 ~ batches-2`）。对于 `step_by_batch()` 模式，`compute_lr_at_step(batch_id)` 期望的是**全局训练步数**（`epoch * steps_per_epoch + batch`）。  
**影响**: `step_by_batch()` 模式下，每个 epoch 的 LR 都重复 epoch 0 的曲线，学习率调度完全失效。（MNIST MLP 使用 `step_by_epoch()`，当前场景不触发，但作为通用基础设施严重缺陷。）  
**修复**:
```cpp
float DeepLearningTask::fetch_lr_for_batch(int batch_id) const {
    return std::visit([this, batch_id](auto&& scheduler) -> float {
        using T = std::decay_t<decltype(scheduler)>;
        if constexpr (std::is_same_v<T, std::monostate>) return 0.0f;
        if (scheduler.is_step_by_batch()) {
            int global_step = current_epoch_ * scheduler.steps_per_epoch() + batch_id;
            return scheduler.get_lr_by_batch(global_step);
        } else {
            return scheduler.get_lr_by_epoch(current_epoch_);
        }
    }, sched_cfg_);
}
```
> 注意：`scheduler.steps_per_epoch()` 已存在（`scheduler.h:59`），可直接调用。

---

### 10. 【新增】run_gpu() 遗漏 early_stop、save_model、best metrics 更新、log_final_summary
**状态**: 🟡 新发现，Z_FINAL_K_REV2.md 遗漏  
**源码对比**: 旧代码 `run_gpu()` (`deep_learning_task.cpp:537-688`) 包含：
- `did_validate` 标记与 `best_top1_` / `best_top5_` / `best_epoch_` 更新
- `early_stop_thr_` 检查与提前终止
- `should_save_this_epoch()` / `save_model_to()`
- `save_best_` 逻辑
- `log_final_summary()` 调用

Z_FINAL_K.md §8.1 的 `run_gpu()` 精简后全部遗漏。  
**影响**: 训练无 early stop、不保存模型、不记录最佳 epoch、不打印最终汇总。  
**修复**: 在 Z_FINAL_K.md §8.1 的 `run_gpu()` 中补充完整逻辑（保留旧代码全部功能）：
```cpp
TrainingResult DeepLearningTask::run_gpu() {
    // ... 前置初始化不变 ...

    for (int epoch = 0; epoch < epochs; ++epoch) {
        current_epoch_ = epoch;
        // ... progressive_resolution / SEMA / prep_thread / run_train_epoch_gpu() ...

        bool did_validate = false;
        float val_loss = 0.0f, top1 = 0.0f, top5 = 0.0f;
        float ema_top1 = 0.0f, ema_top5 = 0.0f;

        if (should_validate_this_epoch()) {
            run_val_epoch_gpu(false);
            did_validate = true;
            if (use_sema_) run_val_epoch_gpu(true);
            // TODO: 从 run_val_epoch_gpu() 获取 val_loss/top1/top5/ema_top1/ema_top5
        }

        auto epoch_end = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(epoch_end - epoch_start).count();

        if (did_validate) {
            if (top1 > best_top1_) {
                best_top1_ = top1;
                best_top5_ = top5;
                best_epoch_ = epoch + 1;
            }
            if (use_sema_ && ema_top1 > best_ema_top1_) {
                best_ema_top1_ = ema_top1;
                best_ema_top5_ = ema_top5;
            }
        }

        log_epoch_results(0.0f, val_loss, top1, top5, ema_top1, ema_top5,
                          fetch_lr_for_batch(0), sec);

        if (should_save_this_epoch()) {
            save_model_to(save_path_, false);
        }
        if (save_best_ && did_validate) {
            float best_this_epoch = use_sema_ ? std::max(top1, ema_top1) : top1;
            float best_overall = use_sema_
                ? std::max(best_top1_, best_ema_top1_) : best_top1_;
            if (best_this_epoch >= best_overall) {
                bool save_ema = use_sema_ && (ema_top1 > top1);
                save_model_to(save_best_path_, save_ema);
            }
        }

        if (did_validate) {
            float best_this_epoch = use_sema_ ? std::max(top1, ema_top1) : top1;
            if (best_this_epoch >= early_stop_thr_) {
                early_stopped_ = true;
                break;
            }
        }
    }

    log_final_summary(...);  // 总时间从 epoch 循环前的 t0 计算

    TrainingResult result;
    result.best_top1 = best_top1_;
    result.best_top5 = best_top5_;
    result.best_ema_top1 = best_ema_top1_;
    result.best_ema_top5 = best_ema_top5_;
    result.best_epoch = best_epoch_;
    return result;
}
```

---

### 11. 【新增】run_gpu() 使用不存在的 build_training_result()
**状态**: 🟡 新发现，Z_FINAL_K_REV2.md 遗漏  
**源码验证**: 全工程搜索无 `build_training_result()` 方法。旧代码直接构造 `TrainingResult` 并赋值字段。  
**影响**: 编译失败。  
**修复**: 改用直接构造（见问题 10 修复代码末尾）。

---

### 12. 【新增】kRequired 数组新增 FIRST_FWD_A/B 为必需，但 non-progressive 模式下 FIRST_FWD_B 可能为空
**状态**: 🟡 新发现，Z_FINAL_K_REV2.md 遗漏  
**源码位置**: Z_FINAL_K.md §4.1  
Z_FINAL_K.md 将 `FIRST_FWD_A` 和 `FIRST_FWD_B` 加入 `kRequired`：
```cpp
static const GraphSlot kRequired[] = {
    GraphSlot::FIRST_FWD_A,
    GraphSlot::FIRST_FWD_B,   // ← 新增
    ...
};
```
如果 Compiler 在未启用 progressive resolution 时不生成 `FIRST_FWD_B` 节点（`train_cg_->nodes(FIRST_FWD_B).empty()`），`build_graph_atlas()` 会跳过该 GraphId，`resolve()` 返回 nullptr，`TR_CHECK` 触发崩溃。  
**缓解**: MLP 当前无 progressive resolution，但需确认 Compiler 是否始终生成 `FIRST_FWD_B`（即使复制 `FIRST_FWD_A` 的节点）。若不确定，建议条件化检查：
```cpp
static const GraphSlot kRequired[] = {
    GraphSlot::XFER_A, GraphSlot::XFER_B,
    GraphSlot::FWD_BWD_DEEP_A, GraphSlot::FWD_BWD_DEEP_B,
    GraphSlot::FIRST_LAYER_BWD,
};
// FIRST_FWD_A/B 单独检查：若 Atlas 中有则必须非空
for (int rank = 0; rank < K; ++rank) {
    if (captured_result_.atlas.index(0, GraphId::FIRST_FWD_A) >= 0)
        TR_CHECK(gpu_exec_.graphs[rank][S(FIRST_FWD_A)], ...);
    if (captured_result_.atlas.index(0, GraphId::FIRST_FWD_B) >= 0)
        TR_CHECK(gpu_exec_.graphs[rank][S(FIRST_FWD_B)], ...);
}
```

---

### 13. 【新增】build_graph_atlas() 中所有 shape_id 设为 kShapeInvariant
**状态**: 🟡 新发现，Z_FINAL_K_REV2.md 遗漏  
**源码位置**: Z_FINAL_K.md §3.2  
```cpp
sl.shape_id = kShapeInvariant;  // ← 所有 GraphId 统一
```
`GraphAtlas` 使用 `(variant, gid, shape_id)` 三元组作为去重键。若 progressive resolution 启用，`FIRST_FWD_A`（低分辨率）与 `FIRST_FWD_B`（高分辨率）shape 不同。全部设为 `kShapeInvariant` 会导致 `pre_capture()` 去重错误，两个不同 shape 共享同一个 `CapturedGraph`，引发运行时 shape mismatch。  
**缓解**: MLP 当前无 progressive resolution，该问题不触发。但作为通用方案，应在文档中明确标注：
> "当前 `build_graph_atlas()` 将 `shape_id` 统一设为 `kShapeInvariant`。 progressive resolution 场景下需根据 `MemoryPlan` 中对应 DTensor 的实际 shape 动态计算 `shape_id`（建议从该 GraphId 的首个节点的 input/output DTensor 推断）。"

---

### 14. DeepLearningTask 路径缺少 StagingParamPool 自动检测
**状态**: Z_FINAL_K_REV2.md 【小伙伴D】已发现，仍待修复  
**源码验证** (`src/task/task_base.cpp:257-278`): `compile_capture_simple()` 中有 StagingParamPool 自动检测（遍历 `named_graphs_` 查找 `RANGE_H2D_COPY_DTENSOR`），但 DeepLearningTask 路径（`is_simple_task() == false`）直接进入 `build_simple_atlas()` → `pre_capture()`，**不会分配 StagingParamPool**。  
**修复**: 在 `compile_impl()` 的 else 分支（DeepLearningTask 路径）开头，复制等价的检测逻辑（见问题 4 的集成版代码，可将检测放在 ArenaKeeper memset 与 Atlas 构建之间）。

---

### 15. compile_impl() 三个片段分散，缺少集成版
**状态**: Z_FINAL_K_REV2.md 【小伙伴D】已发现，仍待修复  
**问题**: Z_FINAL_K.md 将 `compile_impl()` 的修改分散在 §3.3（Atlas 构建）、§5.1（memset）、§5.3（init_all()）三个章节，未展示最终集成顺序。  
**修复**: 已在问题 4 的修复代码中给出完整集成版（顺序：alloc hardware → memset → StagingParamPool 检测 → Atlas 构建 → mark compiled → init_all）。

---

### 16. StagingParamPool 只有注释没有实际代码
**状态**: Z_FINAL_K_REV2.md 【小伙伴D】已发现，仍待修复  
**问题**: Z_FINAL_K.md §6.5 的"方案 A: StagingParamPool（推荐）"只有注释占位，无实际 `set_param()` 调用代码。  
**修复**: 若 Compiler 已生成 `RANGE_H2D_COPY_DTENSOR` 节点，Phase 4 应补充实际代码：
```cpp
// Phase 4: LR H2D（StagingParamPool 路径）
float lr = fetch_lr_for_batch(batch);
auto& reg = GlobalRegistry::instance();
StagingParamPool* pool = reg.staging_params_ptr();
TR_CHECK(pool != nullptr, RuntimeError, "StagingParamPool not allocated");
pool->set_param(lr_slot_index_[rank], lr);  // 每个 rank 预分配一个 slot
cudaGraphLaunch(g_lr_xfer, s_up);            // g_lr_xfer 包含 RANGE_H2D_COPY_DTENSOR
```
在 `compile_impl()` 中通过 StagingParamPool 自动检测分配 slot 索引（见问题 14）。若暂不使用 StagingParamPool，Fall back 到 `cudaMallocHost` 路径也是正确的，但文档应明确当前实现走哪条路径。

---

## 三、P2 文档/结构/健壮性优化

### 17. lr_pinned_ 分配/释放代码缺失
**状态**: Z_FINAL_K_REV2.md 已发现，仍待修复  
**修复**: 见问题 2 的完整代码（含分配时机、修正后的使用语法、释放逻辑）。

---

### 18. build_graph_index() 多余调用
**状态**: Z_FINAL_K_REV2.md 已发现，仍待修复  
**问题**: `build_graph_index()` 遍历 `name_to_gid_`，但新方案中 `build_graph_atlas()` 不填充 `name_to_gid_`，该 map 为空。调用无害但多余。  
**修复**: 直接删除 `dl->build_graph_index();` 调用。

---

### 19. 【新增】init() 中 host tensor dtype 处理不完整
**状态**: 🟢 新发现  
**源码位置**: Z_FINAL_K.md §5.2  
```cpp
Tensor host(live_dt.shape, live_dt.dtype);
float* data = host.data<float>();  // ← 若 dtype == FP16，越界写入
```
`Tensor::data<float>()` 不检查 dtype，直接返回 `void*` 强转。若 `live_dt.dtype == DType::FP16`，tensor 实际分配 `numel * 2` bytes，但 `float*` 写入 `numel * 4` bytes，发生越界。  
**缓解**: MNIST MLP 权重为 FP32，当前场景安全。建议增加 dtype 断言或分路径处理：
```cpp
TR_CHECK(live_dt.dtype == DType::FP32, NotImplemented,
         "init() currently only supports FP32 weights");
```

---

### 20. 【新增】lr_pinned_ 需要自定义析构函数释放 cudaMallocHost 内存
**状态**: 🟢 新发现  
**源码验证** (`deep_learning_task.h:45`): `~DeepLearningTask() override = default;`  
`std::vector<float*>` 的默认析构只会释放指针数组本身，不会对每个 `cudaMallocHost` 指针调用 `cudaFreeHost`，导致锁页内存泄漏。  
**修复**: 声明自定义析构函数：
```cpp
// deep_learning_task.h
~DeepLearningTask() override;

// deep_learning_task.cpp
DeepLearningTask::~DeepLearningTask() {
#ifdef TR_USE_CUDA
    for (auto* p : lr_pinned_) {
        if (p) cudaFreeHost(p);
    }
    lr_pinned_.clear();
#endif
}
```

---

### 21. 【新增】DeviceContext 在 DeepLearningTask 路径可能未 set_memory_plan
**状态**: 🟢 新发现，待验证  
**问题**: Z_FINAL_K.md §7.3 使用 `ctx.ptr_at(lr_dtensor_id_)` 获取 LR 设备指针。`DeviceContext::ptr_at(int dtensor_id)` 的实现未读取，但从 `compile_capture_simple()` (`task_base.cpp:337`) 中显式调用 `ctx.set_memory_plan(&memory_plan_)` 推断，DeviceContext 可能需要绑定 memory plan 才能解析 dtensor id → offset。DeepLearningTask 路径（`pre_capture()`）未显式设置 memory plan。  
**建议**: 在 `build_graph_atlas()` 或 `pre_capture()` 之前，补充与 SimpleTask 等价的设置：
```cpp
for (int rank = 0; rank < num_gpus_; ++rank) {
    backend_->contexts[rank]->set_rank(rank);
    backend_->contexts[rank]->set_memory_plan(active_memory_plan_);
}
```
若 `pre_capture()` 内部已处理，则此设置冗余但无害。

---

### 22. memory_plan() 修改缺少 SimpleTask 回归说明
**状态**: Z_FINAL_K_REV2.md 【小伙伴D】已发现  
**修复**: 在 `task_base.h` 的 `active_memory_plan_` 声明旁增加注释：
```cpp
MemoryPlan* active_memory_plan_ = &memory_plan_;  // SimpleTask 默认指向基类实例，零影响
```

---

### 23. lr_dtensor_id_ == -1 未处理
**状态**: Z_FINAL_K_REV2.md 【小伙伴D】已发现  
**问题**: `on_prepare()` 查找 `S_SCALAR_FP32` 的循环后，若未找到，`lr_dtensor_id_` 保持默认值（未初始化或 -1）。`run_train_epoch_gpu()` 中 `ctx.ptr_at(lr_dtensor_id_)` 可能返回 nullptr 或崩溃。  
**修复**: 循环后增加断言：
```cpp
TR_CHECK(lr_dtensor_id_ >= 0, ValueError,
         "LR DTensor not found: no DTensor with region S_SCALAR_FP32");
```

---

## 四、Z_FINAL_K.md 对 step() 的描述修正

Z_FINAL_K.md §6.1 描述 `step()` 为 `"++current_step_; current_lr_ = compute_lr_at_step(current_step_)"`（先递增后计算）。  
**实际源码** (`src/algo/scheduler.cpp:102-108`): `"current_lr_ = compute_lr_at_step(current_step_); current_step_ += ..."`（先计算后递增）。  
**影响**: Z_FINAL_K.md 对 D 稿 off-by-one 的分析前提有误。但无论哪种顺序，`step_by_epoch()` 模式下 batch 内调用 `step()` 都会导致 `current_step_` 在 epoch 内爆炸性增长（从 0 → 469 × steps_per_epoch）。K 稿废弃 batch 内 `step()` 的决策仍然正确，但文档中关于 `step()` 的描述应修正为与源码一致。

---

## 五、最终修改意见汇总表

| # | 问题 | 严重度 | 来源 | 修复位置 |
|---|------|:------:|------|---------|
| 1 | `run_gpu()` 日志 `get_current_lr()` 永远返回初始 LR | 🔴 | REV2 | `run_gpu()` epoch 循环末尾 |
| 2 | `lr_pinned_` 类型未明确，非锁页内存 | 🔴 | REV2 | `deep_learning_task.h` + `.cpp` |
| 3 | `memory_plan_ptr_->finalize()` 无幂等保护 | 🔴 | REV2 | `on_prepare()` |
| 4 | `compile_impl()` 中 `init_all()` 在 MEMORY_LOCKED 阶段调用 | 🔴 | **新增** | `compile_impl()` 顺序调整 |
| 5 | `run_gpu()` 中 `apply_sema_switch()` 时序错位 | 🔴 | **新增** | `run_gpu()` epoch 循环内 |
| 6 | `init()` UNIFORM 变体用 `normal_distribution` | 🟡 | REV2 | `task_base.cpp` |
| 7 | `cudaDeviceSynchronize()` 只同步最后设备 | 🟡 | REV2 | `task_base.cpp` |
| 8 | 单 batch / 中间 batch / last batch AMP 检查不一致 | 🟡 | REV2 | `deep_learning_task.cpp` |
| 9 | `fetch_lr_for_batch()` 在 `step_by_batch` 模式下传相对索引 | 🟡 | **新增** | `fetch_lr_for_batch()` 实现 |
| 10 | `run_gpu()` 遗漏 early_stop / save / best metrics / final summary | 🟡 | **新增** | `run_gpu()` 完整实现 |
| 11 | `run_gpu()` 使用不存在的 `build_training_result()` | 🟡 | **新增** | `run_gpu()` 返回值构造 |
| 12 | `kRequired` 新增 `FIRST_FWD_B` 可能为空 | 🟡 | **新增** | `build_exec_table()` |
| 13 | `build_graph_atlas()` 所有 `shape_id` 设为 `kShapeInvariant` | 🟡 | **新增** | 文档标注限制 |
| 14 | DeepLearningTask 路径缺少 StagingParamPool 自动检测 | 🟡 | REV2 D | `compile_impl()` else 分支 |
| 15 | `compile_impl()` 三片段分散，缺集成版 | 🟡 | REV2 D | 新增集成版章节 |
| 16 | StagingParamPool 只有注释没有代码 | 🟡 | REV2 D | Phase 4 LR 传输代码 |
| 17 | `lr_pinned_` 分配/释放代码缺失 | 🟢 | REV2 | `deep_learning_task.h/cpp` |
| 18 | `build_graph_index()` 多余调用 | 🟢 | REV2 | `compile_impl()` |
| 19 | `init()` 中 host tensor FP16 越界 | 🟢 | **新增** | `task_base.cpp` dtype 断言 |
| 20 | `~DeepLearningTask()` 默认析构不释放 `cudaMallocHost` | 🟢 | **新增** | `deep_learning_task.h/cpp` |
| 21 | DeviceContext 可能未 `set_memory_plan` | 🟢 | **新增** | `compile_impl()` 验证/补充 |
| 22 | `memory_plan()` 修改缺 SimpleTask 说明 | 🟢 | REV2 D | `task_base.h` 注释 |
| 23 | `lr_dtensor_id_ == -1` 未处理 | 🟢 | REV2 D | `on_prepare()` |

---

## 六、对 Z_FINAL_K.md 的核心评价

Z_FINAL_K.md 的架构决策（`active_memory_plan_` 指针、`build_graph_atlas()` 去重、`get_lr_by_batch/epoch()` 无状态查询、`cudaMallocHost` 持久化锁页内存）**全部正确且优于 D 稿**。但实施层面存在以下结构性风险：

1. **时序错误是最大隐患**: `init_all()` 的 phase 检查、`apply_sema_switch()` 的 epoch 边界、`run_gpu()` 的功能完整性，这三处时序/遗漏问题会导致编译失败、SEMA 延迟生效、或训练结束后无输出。
2. **通用性缺陷**: `fetch_lr_for_batch()` 的相对索引、`shape_id` 的硬编码、`kRequired` 的静态断言，这三处在非 MLP 场景（step_by_batch / progressive resolution / 单 shape 变体）下会触发 bug。
3. **文档与代码的 gap**: `StagingParamPool` 只有设计无实现、`compile_impl()` 缺少集成版、若干成员变量（`train_cg_`/`infer_cg_`/`lr_pinned_`）的声明和生命周期未在修改清单中完整给出。

**建议**: 在 Z_FINAL_K.md 基础上，按本 REV3 的"问题 4 集成版代码"重写 `compile_impl()`，按"问题 5 + 10"重写 `run_gpu()`，补充所有成员变量的完整声明/分配/释放代码，即可形成可直接实施的终稿。
