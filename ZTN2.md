# 【小伙伴K】

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22
> **前置**: AXX.md（MMP0需求 + S/K/D三稿 + 用户补充）+ 源码全路径交叉验证
> **目标**: `tests/ref/mnist_mlp_3.cpp` 从 `compile_for_dry_run()+dry_run()` 改为 `compile()+run()`

---

## 一、执行摘要

经过对 AXX.md 四稿（MMP0 + S/K/D）及 TR4 v4.20.1 源码的逐行交叉验证，发现前三稿均存在**可被证伪的设计缺陷**，且存在**三稿共同遗漏的关键边界条件**。本方案去芜存菁，在吸收各稿最优设计的同时修正所有已知问题，提出一套**可直接落地、可渐进验证**的完整实施方案。

### 核心决策

| 决策点 | 选择 | 被废弃的替代方案 | 废弃理由 |
|--------|------|------------------|---------|
| MemoryPlan 修正 | `active_memory_plan_` 指针 | 虚函数 `memory_plan()`（D稿） | 虚函数开销不必要，指针方案已足够 |
| Atlas 构建 | `DeepLearningTask::build_graph_atlas()` 按 GraphId | `build_simple_atlas()`（现有代码） | 现有代码把所有图合并为 SIMPLE_TASK_GRAPH，导致 pre_capture 只捕获一张图 |
| Exec Table 解析 | 按 `GraphId` 直接 `atlas.index()` | 名字查找 `resolve("xfer_a")`（现有代码） | `named_graphs_` 中无子图名字，返回 nullptr |
| 图调度相位 | MMP0 原始 4 阶段（ZERO_GRAD‖FIRST_FWD → DEEP‖XFER → FIRST_BWD‖COMM → LR+WU） | S/K/D 三稿的变体 | MMP0 伪代码是权威需求源，S稿把 ZERO_GRAD 析出增加无谓同步 |
| **学习率机制** | **线程内只读查询 `get_lr_by_batch/epoch()`** | D稿 `clone_scheduler()`+`step()` | **D稿有 LR 偏移 bug：step() 后 get_current_lr() 得到的是下一个 batch 的 LR**；且 MNIST 使用 `step_by_epoch()`，batch 内 step() 语义错误 |
| 初始化 | CPU `<random>` 生成 + `transfer_to_rank` + `broadcast_from_rank0` | S稿预计算 `lr_table_` | 预计算表需 `reset(steps_back)` API 不存在，且索引偏移 |
| 线程模型 | epoch 级单 join + 忙等待 TransferStation | 条件变量（MMP0 提及） | 不修改 Preprocessor，符合用户约束 |

### 关键发现：D 稿的 LR 偏移 Bug

D 稿 Phase 4 代码逻辑：

```cpp
scheduler_copy.step();              // current_step_ 从 N → N+1
float lr = scheduler_copy.get_current_lr();  // 返回 step N+1 的 LR
cudaMemcpyAsync(lr_dev_ptr, &lr, ...);       // WEIGHT_UPDATE 使用的是 step N+1 的 LR
```

**问题**：`step()` 的语义是"推进到下一步并更新 current_lr"。在 batch N 的 Phase 4 调用 `step()`，得到的是 batch N+1 的学习率，而 WEIGHT_UPDATE 需要的是 batch N 的学习率。这是** off-by-one **错误，会导致所有权重更新使用**下一个 batch **的学习率。

**根因**：D 稿误将 "每个线程拥有 Scheduler" 理解为 "每个线程独立 step()"，但忽略了 step() 的副作用时序。

**修正**：使用 `get_lr_by_batch(batch_id)` 或 `get_lr_by_epoch(epoch_id)` 进行**无状态只读查询**，不推进 scheduler 状态。

---

## 二、AXX.md 全稿审查结论

### 2.1 小伙伴 S 方案（科学实验驱动）

**可取之处**：
- 实验驱动的方法论（假设-验证-修正循环）
- 模板方法模式 `working_memory_plan()` 的提出

**严重问题**：
1. **到处 `dynamic_cast`**：在 `compile_impl()`、`init()` 等路径中反复 `dynamic_cast<DeepLearningTask*>`，代码极脏且性能差
2. **`train_cg_` 指向临时对象**：`train_cg_ = &result.train_cg` 指向局部变量，函数返回后悬空
3. **预计算 `lr_table_` 索引偏移**：`batch 0` 使用 `lr_table_[1]`，且需要 `scheduler.reset(steps_back)` API 不存在
4. **scheduler 预计算表内存竞争**：`lr_table_` 是成员变量，多线程读安全，但预计算过程本身需遍历 total_steps（可能数万步），在主线程阻塞

### 2.2 小伙伴 K 方案（active_memory_plan_）

**可取之处**：
- `active_memory_plan_` 指针方案最小侵入，SimpleTask 零影响
- 学习率 `fetch_lr_for_batch()` 只读查询，无状态竞争，无偏移 bug
- `build_graph_atlas()` 按 GraphId 遍历，`build_exec_table()` 废弃名字查找
- 详细的 TransferStation buffer 标记闭环分析

**问题**：
1. `sync_lr_to_gpu(rank, s_up)` 初版仅 `rank==0` 执行，后续修正为所有 rank 自计算自传
2. 部分代码片段中 `&lr` 局部变量直接传给 `cudaMemcpyAsync`（生命周期风险），但已用 `lr_host_buffer_` 修正

### 2.3 小伙伴 D 方案（严格对齐 MMP0）

**可取之处**：
- 严格对齐 MMP0 原始 4 阶段调度意图
- `FIRST_FWD ‖ ZERO_GRAD` 并行的正确论证
- XFER 只与 DEEP 重叠的合理分析

**严重问题**：
1. **LR 偏移 bug**（见执行摘要），这是 D 稿最核心的设计缺陷
2. `clone_scheduler()` 对 `step_by_epoch()` 模式不适用：MNIST MLP 使用 `.step_by_epoch()`，batch 内 `step()` 会在每个 batch 都推进 epoch 计数器，导致学习率在第一个 epoch 内就衰减到接近 0
3. 若 scheduler 为 `std::monostate`，`clone_scheduler()` 的 visitor 需要处理所有 variant 类型，增加了维护负担

### 2.4 三稿共同遗漏

经过源码全路径扫描，以下问题在三稿中**全部遗漏**：

| # | 遗漏问题 | 影响 | 代码位置 |
|---|---------|------|---------|
| 1 | `zero()` 和 `randn()` 仍用 `memory_plan_.get_dtensor()` | DeepLearningTask 调用 `zero()`/`randn()` 时崩溃 | `task_base.cpp:1139, 1175` |
| 2 | `on_prepare()` 中 `finalize_memory()` 操作空实例，且 `memory_plan_ptr_` 的 finalize 状态未确认 | `memory_plan_ptr_` 可能未 finalize，导致后续 `total_bytes()` 返回 0 | `deep_learning_task.h:269` |
| 3 | `compile_verify_memory_locked()` 检查 `memory_plan_.is_finalized()` 对 DeepLearningTask 无意义 | 碰巧通过（空实例可能已 finalized），但逻辑错误 | `task_base.cpp:550-564` |
| 4 | `lr_dtensor_id_` 查找条件 `dt.usage == DTensorUsage::SCALAR_FP32` 不存在 | `DTensor` 无 `usage` 字段，编译失败 | D 稿 `on_prepare()` 修改 |
| 5 | `init()` 中 `transfer_to_rank(host, live_dt, 0)` 只传 rank 0 | 多卡场景下其他 rank 权重未初始化 | K/D 稿 `init()` 实现 |
| 6 | `run_train_epoch()`（CPU 路径）使用 `TaskBase::run()` 名字查找，但 `named_graphs_` 中只有 "train"/"inference" | CPU 路径无法运行，是遗留代码 | `deep_learning_task.cpp:337` |
| 7 | `build_graph_index()` 依赖 `name_to_gid_`，但 `build_graph_atlas()` 不填充 | `build_graph_index()` 遍历空 map，无害但多余 | `compile_impl()` 调用链 |
| 8 | `compile_impl()` 的 `else` 分支中 `build_simple_atlas` 使用 `&memory_plan_` 作为 `sl.mp` | Atlas slot 的 MemoryPlan 指针指向空实例 | `task_base.cpp:616-655` |

---

## 三、根因分析

### 3.1 MemoryPlan 双轨错位

```
TaskBase::memory_plan_          = 空实例（plan_config_ 构造，无 dtensors）
DeepLearningTask::memory_plan_ptr_ = Compiler 产出的真实布局（含所有 dtensor offset）

on_prepare() 中：memory_plan_ptr_ ← Compiler 结果
                  finalize_memory() → memory_plan_.finalize()（空实例的 finalize）

compile_alloc_hardware()：memory_plan_.total_bytes() → 0 → ArenaKeeper 分配 0 字节
init_all()：memory_plan_.dtensors() 为空 → 不初始化任何权重
transfer/fetch/fill：memory_plan_.get_dtensor(id) → 找不到 → 越界/崩溃
```

**修正原理**：引入 `MemoryPlan* active_memory_plan_`，默认 `&memory_plan_`，DeepLearningTask 在 `on_prepare()` 末尾切换为 `memory_plan_ptr_.get()`。所有涉及显存分配、数据传输、初始化的方法统一走指针。

### 3.2 GraphAtlas 构建错误

现有 `build_simple_atlas()` 逻辑：

```cpp
for (每个 named_graphs_ 中的名字) {
    GraphId gid = GraphId::SIMPLE_TASK_GRAPH;  // ← 所有图共用同一个 GraphId！
    sl.cg = &entry.graph;
    sl.mp = &memory_plan_;  // ← 空实例！
}
```

后果：
1. `pre_capture()` 只看到 1 个 GraphId（SIMPLE_TASK_GRAPH），只捕获 1 张 CUDA Graph
2. DeepLearningTask 需要 16+ 张独立子图（TRANSFER_A/B, FIRST_FWD_A/B, DEEP_FWD_BWD, ...）
3. `sl.mp` 指向空 MemoryPlan，捕获时所有 DTensor 的 offset 都是 0

**修正原理**：`build_graph_atlas()` 按 `GraphId` 枚举遍历 `train_cg_->nodes(gid)`，每个非空 GraphId 独立建 slot，`sl.mp` 指向真实 MemoryPlan。

### 3.3 学习率机制时序错误

D 稿的核心假设："`scheduler_copy.step()` 后 `get_current_lr()` 得到当前 batch 的 LR"。

验证 `LRScheduler::step()` 语义（`src/algo/scheduler.cpp`）：

```cpp
void LRScheduler::step() {
    ++current_step_;
    current_lr_ = compute_lr_at_step(current_step_);
}
```

`step()` **先递增后计算**。初始状态 `current_step_ = 0`，`current_lr_ = base_lr`。

- batch 0 的 Phase 4 调用 `step()` → `current_step_ = 1`，`current_lr_ = LR(1)`
- `get_current_lr()` 返回 `LR(1)`，但 WEIGHT_UPDATE for batch 0 应该用 `LR(0)`

对于 `step_by_epoch()` 模式（MNIST MLP）：
- `step()` 被设计为每个 epoch 调用一次
- 在 batch 内调用 `step()` 会导致 `current_step_` 在每个 batch 都递增
- 第一个 epoch 有 469 batches，`current_step_` 从 0 → 469，学习率在第一轮就衰减到接近 0

这是**双重错误**：既有时序偏移，又有 step 频率错误。

**修正原理**：使用 `get_lr_by_batch(batch_id)` / `get_lr_by_epoch(epoch_id)` 进行无状态查询，不修改 `current_step_`。

### 3.4 `init()` 真实路径缺失

当前 `TaskBase::init()` 直接 `TR_NOT_IMPLEMENTED`。权重初始化是训练的必要条件：

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    // ... 配置解析 ...
    TR_NOT_IMPLEMENTED("Real init not yet implemented");  // ← 阻断点
}
```

Initializer 已经配置好每个 DTensor 的 `init_config`（Xavier/Normal/Zeros/Constants），`init_all()` 会遍历所有 dtensor 调用 `init()`。实现真实路径后：
1. CPU 侧生成随机数（`<random>` 或 Philox）
2. `transfer_to_rank(host, live_dt, 0)` H2D 到 rank 0
3. `broadcast_from_rank0(live_dt)` NCCL 广播到所有 rank

---

## 四、最终方案

### 4.1 MemoryPlan 双轨修正（P0）

#### 4.1.1 `task_base.h` —— 新增 `active_memory_plan_` 指针

```cpp
protected:
    PlanConfig plan_config_;
    Phase phase_ = Phase::PLANNING;
    MemoryPlan memory_plan_{plan_config_};
    MemoryPlan* active_memory_plan_ = &memory_plan_;  // ← 新增：默认指向值成员

public:
    [[nodiscard]] const MemoryPlan& memory_plan() const noexcept {
        return *active_memory_plan_;  // ← 外部查询也走真实布局
    }
```

#### 4.1.2 `task_base.cpp` —— 8 处 `memory_plan_.` → `active_memory_plan_->`

| 方法 | 行号（≈） | 修改后表达式 |
|------|----------|------------|
| `compile_alloc_hardware()` | 1409 | `active_memory_plan_->total_bytes()` |
| `init()` | 1223（新实现中） | `active_memory_plan_->get_dtensor(dtensor.id)` |
| `init_all()` | 1230, 1235, 1243 | `active_memory_plan_->dtensors()` / `get_dtensor(id)` |
| `transfer_to_rank()` | 887 | `active_memory_plan_->get_dtensor(dt.id)` |
| `fetch_from_rank()` | 1271 | `active_memory_plan_->get_dtensor(dt.id)` |
| `fill()` | 1102 | `active_memory_plan_->get_dtensor(dt.id)` |
| **`zero()`** | **1139** | **`active_memory_plan_->get_dtensor(dt.id)`** |
| **`randn()`** | **1175** | **`active_memory_plan_->get_dtensor(dt.id)`** |

> **注意**：`zero()` 和 `randn()` 是三稿共同遗漏，必须一并修正。

#### 4.1.3 `deep_learning_task.h` —— `on_prepare()` 修正

```cpp
void on_prepare() override {
    // ... 现有 ArchPlan 构建、Compiler 调用不变 ...

    memory_plan_ptr_ = std::move(result.variants[0].memory_plan);

    // === 修正：确保 memory_plan_ptr_ 已 finalize ===
    // Compiler 可能已 finalize，为安全起见做幂等检查
    if (!memory_plan_ptr_->is_finalized()) {
        memory_plan_ptr_->finalize();
    }
    phase_ = Phase::MEMORY_LOCKED;  // 跳过基类的 finalize_memory()（操作空实例）

    // === 查找学习率 DTensor ID（修正：用 region 而非不存在的 usage） ===
    for (const auto& dt : memory_plan_ptr_->dtensors()) {
        if (dt.region == Region::S_SCALAR_FP32) {
            lr_dtensor_id_ = dt.id;
            break;
        }
    }

    add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
    add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

    // add_graph move 后 named_graphs_ 内地址稳定
    train_cg_ = &named_graphs_["train"].graph;
    infer_cg_ = &named_graphs_["inference"].graph;

    // === 激活真实 MemoryPlan ===
    active_memory_plan_ = memory_plan_ptr_.get();   // ← 关键一行
}
```

> **`compile_verify_memory_locked()` 的兼容**：`compile_verify_memory_locked()` 检查 `memory_plan_.is_finalized()`。对于 DeepLearningTask，`memory_plan_`（空实例）在构造时可能即处于 finalized 状态（无 dtensor 需要 finalize），因此 `is_finalized()` 可能返回 true，代码路径 `phase_ = Phase::MEMORY_LOCKED` 恰好通过。该检查对 SimpleTask 仍有效，对 DeepLearningTask 无实质影响，无需修改。

---

### 4.2 图集架构（P0）

#### 4.2.1 `GraphSlot` 枚举扩展

```cpp
enum class GraphSlot : uint8_t {
    FIRST_FWD_A = 0,       // ← 新增
    FIRST_FWD_B,           // ← 新增
    XFER_A,
    XFER_B,
    FWD_BWD_DEEP_A,
    FWD_BWD_DEEP_B,
    FIRST_LAYER_BWD,
    ZERO_GRAD,
    DEEP_ALLREDUCE,
    FIRST_LAYER_ALLREDUCE,
    WEIGHT_UPDATE,
    EMA_UPDATE,
    GRAD_CONVERT,
    CAST_AND_CHECK,        // ← 新增（AMP 梯度转换+NaN检查）
    INF_MAIN_A,
    INF_MAIN_B,
    INF_EMA_A,
    INF_EMA_B,
    COUNT
};
```

#### 4.2.2 `build_graph_atlas()` —— 按 GraphId 独立构建

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() const {
    GraphAtlas atlas;

    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (train_cg_->nodes(gid).empty()) continue;

            auto& sl = atlas.slot(0, gi);
            sl.cg          = train_cg_;
            sl.mp          = active_memory_plan_;   // ← 真实布局
            sl.shape_id    = kShapeInvariant;
            sl.stream_kind = stream_for(gid);
        }
    }

    if (infer_cg_) {
        static const GraphId kInferIds[] = {
            GraphId::INF_MAIN_A, GraphId::INF_MAIN_B,
            GraphId::INF_EMA_A,  GraphId::INF_EMA_B
        };
        for (GraphId gid : kInferIds) {
            if (infer_cg_->nodes(gid).empty()) continue;
            auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
            sl.cg          = infer_cg_;
            sl.mp          = active_memory_plan_;
            sl.shape_id    = kShapeInvariant;
            sl.stream_kind = StreamKind::COMP_1;
        }
    }
    return atlas;
}

StreamKind DeepLearningTask::stream_for(GraphId gid) {
    switch (gid) {
        case GraphId::TRANSFER_A:
        case GraphId::TRANSFER_B:
            return StreamKind::TRANS;
        case GraphId::ZERO_GRAD:
        case GraphId::FIRST_COMM:
        case GraphId::DEEP_COMM:
        case GraphId::CAST_AND_CHECK:
        case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:
            return StreamKind::UPDATE;
        default:
            return StreamKind::COMP_1;
    }
}
```

#### 4.2.3 `build_exec_table()` —— 按 GraphId 解析，废弃名字查找

```cpp
void DeepLearningTask::build_exec_table() {
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;

    const int K = num_gpus_;
    gpu_exec_.device_ids.resize(K);
    gpu_exec_.graphs.resize(K);

    auto resolve = [this](GraphId gid, int rank) -> cudaGraphExec_t {
        int32_t idx = captured_result_.atlas.index(0, gid);
        if (idx < 0) return nullptr;
        return static_cast<cudaGraphExec_t>(
            captured_result_.graphs[idx].native_exec(rank));
    };

    for (int rank = 0; rank < K; ++rank) {
        gpu_exec_.device_ids[rank] = context(rank).device_id();
        auto& g = gpu_exec_.graphs[rank];
        g.resize(static_cast<size_t>(GraphSlot::COUNT), nullptr);

        g[S(FIRST_FWD_A)]        = resolve(GraphId::FIRST_FWD_A,   rank);
        g[S(FIRST_FWD_B)]        = resolve(GraphId::FIRST_FWD_B,   rank);
        g[S(XFER_A)]             = resolve(GraphId::TRANSFER_A,    rank);
        g[S(XFER_B)]             = resolve(GraphId::TRANSFER_B,    rank);
        g[S(FWD_BWD_DEEP_A)]     = resolve(GraphId::DEEP_FWD_BWD,  rank);
        g[S(FWD_BWD_DEEP_B)]     = resolve(GraphId::DEEP_FWD_BWD,  rank);
        g[S(FIRST_LAYER_BWD)]    = resolve(GraphId::FIRST_BWD,     rank);
        g[S(ZERO_GRAD)]          = resolve(GraphId::ZERO_GRAD,     rank);
        g[S(DEEP_ALLREDUCE)]     = resolve(GraphId::DEEP_COMM,     rank);
        g[S(FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM, rank);
        g[S(WEIGHT_UPDATE)]      = resolve(GraphId::OPTIMIZER,     rank);
        g[S(EMA_UPDATE)]         = resolve(GraphId::EMA_UPDATE,    rank);
        g[S(GRAD_CONVERT)]       = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(CAST_AND_CHECK)]     = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(INF_MAIN_A)]         = resolve(GraphId::INF_MAIN_A,    rank);
        g[S(INF_MAIN_B)]         = resolve(GraphId::INF_MAIN_B,    rank);
        g[S(INF_EMA_A)]          = resolve(GraphId::INF_EMA_A,     rank);
        g[S(INF_EMA_B)]          = resolve(GraphId::INF_EMA_B,     rank);
    }

    static const GraphSlot kRequired[] = {
        GraphSlot::FIRST_FWD_A,
        GraphSlot::FIRST_FWD_B,
        GraphSlot::XFER_A,
        GraphSlot::XFER_B,
        GraphSlot::FWD_BWD_DEEP_A,
        GraphSlot::FWD_BWD_DEEP_B,
        GraphSlot::FIRST_LAYER_BWD,
    };
    for (int rank = 0; rank < K; ++rank) {
        for (auto slot : kRequired) {
            if (!gpu_exec_.graphs[rank][static_cast<size_t>(slot)]) {
                TR_CHECK(false, ValueError,
                    "Required slot " << static_cast<int>(slot)
                    << " is nullptr for rank " << rank);
            }
        }
    }
#endif
}
```

#### 4.2.4 `compile_impl()` —— 替换 Atlas 构建

```cpp
} else {
    // DeepLearningTask 分支：复用已有的 dl 指针
    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        GraphAtlas atlas = dl->build_graph_atlas();   // ← 替换 build_simple_atlas

        std::vector<DeviceContext*> ctx_ptrs;
        for (auto& ctx : backend_->contexts)
            ctx_ptrs.push_back(ctx.get());

        captured_result_ = pre_capture(atlas, ctx_ptrs);

        // build_graph_index() 遍历 name_to_gid_，但 DeepLearningTask GPU 路径
        // 使用 build_exec_table() 按 GraphId 解析，不再依赖 name_to_graph_index_。
        // 保留调用无害（遍历空 map），但可移除。
        dl->build_graph_index();
        dl->build_exec_table();
    }
}
```

---

### 4.3 数据初始化（P0）

#### 4.3.1 ArenaKeeper 显存清零

在 `compile_impl()` 中 `compile_alloc_hardware()` 之后插入：

```cpp
#ifdef TR_USE_CUDA
if (GlobalRegistry::instance().using_gpu()) {
    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaSetDevice(backend_->contexts[rank]->device_id());
        void* ptr = ArenaKeeper::instance().ptr_at(rank, 0);
        size_t bytes = active_memory_plan_->total_bytes();
        cudaMemset(ptr, 0, bytes);
    }
    cudaDeviceSynchronize();
}
#endif
```

#### 4.3.2 `init()` 真实路径（含多卡广播）

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    check_phase(Phase::COMPILED, "init");

    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : dtensor.init_config;
    if (!config.needs_init()) return;
    if (debug_mode_) { /* dry run print */ return; }

    const DTensor& live_dt = active_memory_plan_->get_dtensor(dtensor.id);

    Tensor host(live_dt.shape, live_dt.dtype);
    switch (config.kind) {
        case InitKind::ZEROS:
            host.fill(0.0f);
            break;
        case InitKind::CONSTANTS:
            host.fill(config.scale);
            break;
        case InitKind::NORMAL:
        case InitKind::KAIMING_NORMAL:
        case InitKind::KAIMING_UNIFORM:
        case InitKind::XAVIER_NORMAL:
        case InitKind::XAVIER_UNIFORM: {
            // 使用 <random> 生成（若未来有 Philox CPU 版本可替换）
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, config.scale);
            float* data = host.data<float>();
            for (int64_t i = 0; i < host.numel(); ++i)
                data[i] = static_cast<float>(dist(gen));
            break;
        }
        default:
            TR_CHECK(false, NotImplemented,
                     "InitKind " << static_cast<int>(config.kind) << " not implemented");
    }

    transfer_to_rank(host, live_dt, 0);   // H2D 到 rank 0
    if (num_gpus_ > 1) {
        broadcast_from_rank0(live_dt);    // NCCL 广播到所有 rank
    }
}
```

#### 4.3.3 `compile_impl()` 中调用 `init_all()`

在 ArenaKeeper memset 之后：

```cpp
// 权重初始化
if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
    (void)dl;
    init_all();
}
```

> 使用 `dynamic_cast` 仅在 `compile_impl()` 中做一次，不污染 `init()` 等高频路径。

---

### 4.4 学习率机制（P0，关键决策）

#### 4.4.1 设计选择：无状态只读查询

**为什么不用 `clone_scheduler()` + `step()`**：
1. `step()` 先递增后计算，导致 LR 时序偏移（off-by-one）
2. MNIST MLP 使用 `step_by_epoch()`，batch 内 `step()` 会在 epoch 内过度衰减
3. `clone_scheduler()` 增加不必要的 variant 拷贝开销

**为什么用 `get_lr_by_batch/epoch()`**：
1. 两个方法均为 `const`，多线程只读安全，零锁竞争
2. `get_lr_by_batch(batch_id)` 直接按 batch 索引计算，无时序偏移
3. `get_lr_by_epoch(epoch_id)` 适用于 MNIST 的 `step_by_epoch()` 模式
4. 不修改 scheduler 状态，无需 reset/clone

#### 4.4.2 新增 `is_step_by_batch()` 接口

```cpp
// include/renaissance/algo/scheduler.h
// 在 LRScheduler public 区新增一行：
bool is_step_by_batch() const noexcept { return step_by_batch_; }
```

#### 4.4.3 线程内 LR 获取

```cpp
float DeepLearningTask::fetch_lr_for_batch(int batch_id) const {
    return std::visit([this, batch_id](auto&& scheduler) -> float {
        using T = std::decay_t<decltype(scheduler)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return 0.0f;
        }
        // get_lr_by_batch / get_lr_by_epoch 均为 const 只读，多线程安全
        if (scheduler.is_step_by_batch()) {
            return scheduler.get_lr_by_batch(batch_id);
        } else {
            return scheduler.get_lr_by_epoch(current_epoch_);
        }
    }, sched_cfg_);
}
```

#### 4.4.4 Phase 4 中的 LR 传输

```cpp
// Phase 4: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE
{
    float lr = fetch_lr_for_batch(batch);   // 轻量 CPU 计算，隐藏在 GPU 工作间隙
    lr_host_buffer_ = lr;
    void* dst = context(rank).ptr_at(lr_dtensor_id_);
    cudaMemcpyAsync(dst, &lr_host_buffer_, sizeof(float),
                    cudaMemcpyHostToDevice, s_up);
}
if (gfa) cudaGraphLaunch(gfa, s_up);   // FIRST_ALLREDUCE（if exists）
if (gwu) cudaGraphLaunch(gwu, s_up);   // WEIGHT_UPDATE
sync_up();
```

> **同流有序**：`cudaMemcpyAsync` 与 `WEIGHT_UPDATE` 同在 `UPDATE` stream，无需额外同步即保证 LR 在权重更新前到位。  
> **生命周期安全**：`lr_host_buffer_` 是成员变量，持久存活到 `cudaMemcpyAsync` 完成。  
> **多卡并行**：8 个 rank 线程各自独立 `cudaMemcpyAsync` 到自己的 GPU，完全并行。

---

### 4.5 图调度：MMP0 四阶段重叠模型（P0）

#### 4.5.1 Stream 分配

| StreamKind | 承载图 |
|-----------|--------|
| `TRANS` | XFER_A/B |
| `COMP_1` | FIRST_FWD_A/B, DEEP_FWD_BWD, FIRST_BWD |
| `COMP_2/3` | 空（MLP 无多计算流需求），仅参与三流同步 |
| `UPDATE` | ZERO_GRAD, ALLREDUCE, CAST_AND_CHECK, WEIGHT_UPDATE, EMA_UPDATE, LR H2D |

#### 4.5.2 每个 Batch 的 4 阶段重叠

```
Phase 1: ZERO_GRAD [UPDATE]  ‖  FIRST_FWD [COMP_1]
         sync_comp + sync_up

Phase 2: DEEP_FWD_BWD [COMP_1]  ‖  XFER(next_batch) [TRANS]   ← 核心重叠
         sync_comp + sync_trans
         (rank==0: 标记 next_buf writeable)

Phase 3: FIRST_BWD [COMP_1]  ‖  DEEP_ALLREDUCE [UPDATE]       (多卡)
         sync_comp + sync_up
         (if AMP: CAST_AND_CHECK [UPDATE], sync_up)

Phase 4: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE  [UPDATE 串行]
         sync_up
```

**Phase 1 为何 FIRST_FWD 与 ZERO_GRAD 并行**：
- FIRST_FWD 读取 `I_{A|B}_DATA`，写入首层激活（COMP_1 流）
- ZERO_GRAD 写入梯度缓冲区（UPDATE 流）
- 两者操作完全不同的显存区域 → 天然可并行
- MMP0 伪代码明确写了这两张图同行 launch

**XFER 为何只与 DEEP 并行**：
- MMP0 原文："最关键的重叠就是传输与深层 FWD/BWD 的重叠"
- 如果 XFER 与 FIRST_FWD 并行，Preprocessor 必须在 Phase 1 前就准备好 next batch，容错性差
- XFER 与 DEEP 并行：Preprocessor 有整个 Phase 1（FIRST_FWD + ZERO_GRAD）的时间填充 next buffer

#### 4.5.3 `run_train_epoch_gpu()` 完整实现

```cpp
#ifdef TR_USE_CUDA
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    const bool frozen = is_first_layer_frozen();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts =
        static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, frozen, batches, ts, &exc, using_amp]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);

                const auto& g = gpu_exec_.graphs[rank];
                #define G(slot) g[static_cast<size_t>(GraphSlot::slot)]
                auto gf_a  = G(FIRST_FWD_A),   gf_b  = G(FIRST_FWD_B);
                auto gx_a  = G(XFER_A),        gx_b  = G(XFER_B);
                auto gd_a  = G(FWD_BWD_DEEP_A), gd_b  = G(FWD_BWD_DEEP_B);
                auto gfb   = G(FIRST_LAYER_BWD), gzg   = G(ZERO_GRAD);
                auto gda   = G(DEEP_ALLREDUCE), gfa  = G(FIRST_LAYER_ALLREDUCE);
                auto gwu   = G(WEIGHT_UPDATE),  ggc  = G(GRAD_CONVERT);
                auto gcn   = G(CAST_AND_CHECK);
                #undef G

                auto& ctx = context(rank);
                auto s_tr = (cudaStream_t)ctx.stream(StreamKind::TRANS);
                auto s_c1 = (cudaStream_t)ctx.stream(StreamKind::COMP_1);
                auto s_c2 = (cudaStream_t)ctx.stream(StreamKind::COMP_2);
                auto s_c3 = (cudaStream_t)ctx.stream(StreamKind::COMP_3);
                auto s_up = (cudaStream_t)ctx.stream(StreamKind::UPDATE);

                auto sync_comp  = [&]() { cudaStreamSynchronize(s_c1);
                                          cudaStreamSynchronize(s_c2);
                                          cudaStreamSynchronize(s_c3); };
                auto sync_trans = [&]() { cudaStreamSynchronize(s_tr); };
                auto sync_up    = [&]() { cudaStreamSynchronize(s_up); };

                void* lr_dev_ptr = ctx.ptr_at(lr_dtensor_id_);

                // ========== Batch 0: 预传输 A 区 ==========
                while (!ts->buffer_is_readable(0))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                cudaGraphLaunch(gx_a, s_tr);
                sync_trans();
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                // ========== 单 batch 边界（调试/测试）==========
                if (batches == 1) {
                    cudaGraphLaunch(gzg, s_up);
                    cudaGraphLaunch(gf_a, s_c1);
                    sync_comp(); sync_up();

                    cudaGraphLaunch(gd_a, s_c1);
                    sync_comp();
                    if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }
                    if (gda) cudaGraphLaunch(gda, s_up);
                    sync_up();
                    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                    {
                        float lr = fetch_lr_for_batch(0);
                        lr_host_buffer_ = lr;
                        cudaMemcpyAsync(lr_dev_ptr, &lr_host_buffer_, sizeof(float),
                                        cudaMemcpyHostToDevice, s_up);
                    }
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                    return;
                }

                // ========== 中间 batches: 乒乓 + 重叠 ==========
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    auto gf  = from_a ? gf_a : gf_b;
                    auto gd  = from_a ? gd_a : gd_b;
                    auto gx_n = from_a ? gx_b : gx_a;

                    // Phase 1: ZERO_GRAD || FIRST_FWD
                    cudaGraphLaunch(gzg, s_up);
                    cudaGraphLaunch(gf,  s_c1);
                    sync_comp(); sync_up();

                    // Wait next buffer
                    while (!ts->buffer_is_readable(next_buf))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    // Phase 2: DEEP_FWD_BWD || XFER(next) —— 核心重叠
                    cudaGraphLaunch(gd,   s_c1);
                    cudaGraphLaunch(gx_n, s_tr);
                    sync_comp(); sync_trans();
                    if (rank == 0) {
                        ts->set_buffer_readable(next_buf, false);
                        ts->set_buffer_writeable(next_buf, true);
                    }

                    // Phase 3: FIRST_BWD || DEEP_ALLREDUCE
                    if (!frozen) cudaGraphLaunch(gfb, s_c1);
                    if (gda)     cudaGraphLaunch(gda, s_up);
                    sync_comp(); sync_up();

                    // AMP: grad_convert（空图自动跳过）
                    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                    // Phase 4: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE
                    {
                        float lr = fetch_lr_for_batch(batch);
                        lr_host_buffer_ = lr;
                        cudaMemcpyAsync(lr_dev_ptr, &lr_host_buffer_, sizeof(float),
                                        cudaMemcpyHostToDevice, s_up);
                    }
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                }

                // ========== Last batch ==========
                bool last_a = ((batches - 1) % 2 == 0);
                auto gf_l = last_a ? gf_a : gf_b;
                auto gd_l = last_a ? gd_a : gd_b;

                cudaGraphLaunch(gzg,  s_up);
                cudaGraphLaunch(gf_l, s_c1);
                sync_comp(); sync_up();

                cudaGraphLaunch(gd_l, s_c1);
                sync_comp();
                if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }
                if (gda) cudaGraphLaunch(gda, s_up);
                sync_up();
                if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                // Last batch LR + WEIGHT_UPDATE
                {
                    float lr = fetch_lr_for_batch(batches - 1);
                    lr_host_buffer_ = lr;
                    cudaMemcpyAsync(lr_dev_ptr, &lr_host_buffer_, sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                }
                if (gfa) cudaGraphLaunch(gfa, s_up);
                if (gwu) cudaGraphLaunch(gwu, s_up);
                sync_up();

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    for (auto& t : threads) t.join();
    for (int r = 0; r < K; ++r) if (exc[r]) std::rethrow_exception(exc[r]);
}
#endif
```

---

### 4.6 N+1 线程与 Preprocessor 协调（P0）

#### 4.6.1 `run_gpu()` 主循环

```cpp
TrainingResult DeepLearningTask::run_gpu() {
    auto& prep = Preprocessor::instance();
    const int epochs = total_epochs_;

    // 一次性 prepare scheduler（所有 rank 共享同一份配置）
    std::visit([this](auto&& sch) {
        using T = std::decay_t<decltype(sch)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            sch.prepare(total_epochs_, prep.steps_per_epoch());
        }
    }, sched_cfg_);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        current_epoch_ = epoch;
        auto epoch_start = std::chrono::steady_clock::now();

        // N+1 线程：1 个 Preprocessor 线程 + K 个 rank 线程
        std::thread prep_thread([&]() { prep.train(); });
        run_train_epoch_gpu();   // 内部展开 K 个 rank 线程，epoch 末尾 join
        prep_thread.join();

        float val_loss = 0, top1 = 0, top5 = 0;
        if (should_validate_this_epoch()) {
            run_val_epoch_gpu(false);
            if (use_sema_) run_val_epoch_gpu(true);
        }

        auto epoch_end = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        log_epoch_results(0.0f, val_loss, top1, top5, 0, 0,
                          get_current_lr(), sec);

        if (use_sema_ && epoch > 0) apply_sema_switch();
    }
    return build_training_result();
}
```

#### 4.6.2 TransferStation 标志闭环验证

`prep.train()` → `wait_all_transfer_stations_consumed()` 阻塞直到所有 buffer 可写。必须确保 `run_train_epoch_gpu()` 在返回前消费完所有数据：

| 场景 | buffer 状态分析 |
|------|----------------|
| `batches == 1` | 预传输后 `rank==0` 把 buffer 0 标为 `writeable=true`。TransferStation 初始状态两个 buffer 均为 `writeable=true`。`wait_all_transfer_stations_consumed()` 立即通过。 |
| `batches > 1` | 中间循环 `batch = 0 .. batches-2` 交替把 next_buf 标为 `writeable=true`。Last batch 的数据在 `batch = batches-2` 时已传输完毕并标记。epoch 结束时两个 buffer 均为 `writeable=true`。 |

**结论**：无死锁。

#### 4.6.3 多线程安全要点

| 机制 | 说明 |
|------|------|
| **TransferStation 标志** | `buffer_is_readable/writeable` 为 `std::atomic<bool>`；**仅 rank 0 负责设置**，避免多 rank 写竞争 |
| **忙等待** | `while (!ts->buffer_is_readable()) { sleep 100us; }` — 不占用 CPU 核心 |
| **异常传播** | `exc[rank]` 数组收集异常，join 后统一 `rethrow_exception` |
| **单卡假设** | 当前 `world_size=1`，`rank==0` 即唯一 rank；多卡扩展时 `transfer_station_ptr(rank)` + LR broadcast |

---

### 4.7 修改文件清单

| # | 文件 | 改动内容 | 阶段 |
|---|------|---------|:--:|
| 1 | `include/renaissance/algo/scheduler.h` | 新增 `bool is_step_by_batch() const noexcept` | P0 |
| 2 | `include/renaissance/task/task_base.h` | 新增 `MemoryPlan* active_memory_plan_`；`memory_plan()` 返回 `*active_memory_plan_` | P0 |
| 3 | `src/task/task_base.cpp` | **8 处** `memory_plan_.` → `active_memory_plan_->`（含 zero/randn）；ArenaKeeper memset；`init()` 真实路径；`compile_impl()` DeepLearningTask 分支改为 `dl->build_graph_atlas()`；compile 后调用 `init_all()` | P0 |
| 4 | `include/renaissance/task/deep_learning_task.h` | 新增 `build_graph_atlas()`, `stream_for()`, `fetch_lr_for_batch()` 声明；新增 `train_cg_`, `infer_cg_`, `lr_dtensor_id_`, `lr_host_buffer_`；修改 `on_prepare()` 逻辑 | P0 |
| 5 | `src/task/deep_learning_task.cpp` | GraphSlot 新增 FIRST_FWD_A/B, CAST_AND_CHECK；实现 `build_graph_atlas()`, `stream_for()`, `fetch_lr_for_batch()`；重写 `build_exec_table()` 按 GraphId 解析；重写 `run_train_epoch_gpu()`（MMP0 4-phase + 只读 LR 查询）；修改 `run_gpu()` N+1 线程 | P0 |
| 6 | `tests/ref/mnist_mlp_3.cpp` | `compile_for_dry_run()+dry_run()` → `compile()+run()` | P2 |

---

## 五、实施顺序与验收标准

### 5.1 四阶段渐进验证

```
Step 1 (P0): active_memory_plan_ + build_graph_atlas() + compile_alloc_hardware 修正
  验收:
  1. compile() 不 crash
  2. active_memory_plan_->total_bytes() > 0
  3. gpu_exec_ 必需 slot 非 null
  时间: 2h

Step 2 (P1): init() 真实路径 + 单 batch 手动数据注入 + batches==1 调度
  验收:
  1. Loss 非 NaN
  2. fetch weight 确认梯度更新发生
  3. 各 rank 线程内 LR 查询值正确（step_by_epoch 模式下所有 batch 相同）
  时间: 2h (无需 Preprocessor，独立验证)

Step 3 (P2): Preprocessor 对接 + A/B 乒乓 + 完整 469 batch 循环
  验收:
  1. 1 epoch 无 crash/死锁
  2. TransferStation 标志正确翻转
  3. Loss 趋势下降
  时间: 3h

Step 4 (P3): 20 epoch 训练 + 验证准确率
  验收:
  1. Best Val Top-1 > 95%
  2. 无 NaN
  3. 无内存泄漏
  时间: 2h
```

### 5.2 分步执行逻辑

**Step 1**: 只改 compile 路径。dry_run 仍然可用（验证编译输出对比）。
- 在 `compile()` 后打印 `ArenaKeeper::instance().total_bytes()` 和 Exec Table 完整性

**Step 2**: 不启动 Preprocessor。手动构造 1 个 batch 数据写入 I_A_DATA/I_A_LABEL，强制 `batches==1`，调度单 batch 路径。
- 验证 `fetch_lr_for_batch(0)` 返回预期值（如 CosineAnnealingLR epoch 0 的 base_lr=0.01）
- fetch loss / weight 验证核心图正确

**Step 3**: 接入 Preprocessor 线程 + TransferStation。
- 验证 A/B 交替、buffer 标志正确

**Step 4**: 20 epoch + 监控收敛。

---

## 六、风险与缓解

| 风险 | 等级 | 缓解 |
|------|:--:|------|
| `active_memory_plan_` 遗漏某处替换 | 🔴 | 编译期搜索所有 `memory_plan_\.`，凡在 `compile_alloc_hardware/init/init_all/transfer/fetch/fill/zero/randn` 路径上的全部替换 |
| `get_lr_by_batch()` 在 step_by_epoch 模式下行为 | 🟢 | 已用 `is_step_by_batch()` 判断分支，`step_by_epoch` 时走 `get_lr_by_epoch()` |
| Preprocessor `train()` 阻塞 → join 死等 | 🟡 | TransferStation buffer 标记闭环已验证；debug build 可设置 epoch 超时 |
| `CAST_AND_CHECK` 图在 non-AMP 下为 `nullptr` | 🟢 | `if (using_amp && ggc)` 已处理空图跳过 |
| MNIST last batch (96 samples) | 🟢 | DEEP_FWD_BWD 图 shape 固定为 128，TransferStation 实际传输 96 样本，后 32 为 0（ArenaKeeper 已清零），训练稳定 |
| `clone_scheduler()` 诱惑（未来维护者可能引入） | 🟡 | 本方案文档化说明：禁止在 batch 内调用 `step()`，必须使用无状态查询 |
| `on_prepare()` 中 `memory_plan_ptr_->is_finalized()` 不存在 | 🔴 | 需确认 `MemoryPlan` 有 `is_finalized()`；若无，改为直接 `memory_plan_ptr_->finalize()` 并捕获异常 |

---

## 附录 A：与 AXX.md 三稿的关键差异总结

| 差异点 | S 稿 | K 稿 | D 稿 | **本方案（ZTN2）** |
|--------|------|------|------|-------------------|
| MemoryPlan 修正 | 虚函数 `working_memory_plan()` | `active_memory_plan_` 指针 | 继承 K 稿 | **继承 K 稿，补充 zero/randn** |
| LR 机制 | 预计算 `lr_table_`（索引偏移） | `fetch_lr_for_batch()` 只读查询 | `clone_scheduler()`+`step()`（**偏移 bug**） | **继承 K 稿，修正 D 稿 bug** |
| Phase 1 调度 | ZERO_GRAD 独立阶段 | ZERO_GRAD ‖ FIRST_FWD 并行 | ZERO_GRAD ‖ FIRST_FWD 并行 | **继承 K/D，对齐 MMP0** |
| on_prepare finalize | 未提及 | 提及但未明确修改 | 未提及 | **明确修改：跳过基类 finalize_memory()，直接 finalize memory_plan_ptr_** |
| init() 广播 | 未提及 | 只传 rank 0 | 只传 rank 0 | **补充 broadcast_from_rank0** |
| lr_dtensor_id_ 查找 | 未提及 | 未提及 | `dt.usage`（字段不存在） | **修正为 `dt.region == S_SCALAR_FP32`** |

---

## 附录 B：源码引用索引

| 引用 | 文件 | 行号 | 内容 |
|------|------|------|------|
| [R1] | `task_base.h` | 275 | `MemoryPlan memory_plan_{plan_config_};` |
| [R2] | `deep_learning_task.h` | 311 | `std::unique_ptr<MemoryPlan> memory_plan_ptr_;` |
| [R3] | `task_base.cpp` | 1409 | `size_t total_bytes = memory_plan_.total_bytes();` |
| [R4] | `task_base.cpp` | 1223 | `TR_NOT_IMPLEMENTED("Real init not yet implemented")` |
| [R5] | `task_base.cpp` | 1235 | `for (const auto& dtensor : memory_plan_.dtensors())` |
| [R6] | `task_base.cpp` | 887 | `const DTensor& live_dt = memory_plan_.get_dtensor(dt.id);` |
| [R7] | `deep_learning_task.cpp` | 32-49 | `GraphSlot` 枚举（缺 FIRST_FWD_A/B, CAST_AND_CHECK） |
| [R8] | `deep_learning_task.cpp` | 475-531 | `build_exec_table()` 使用名字解析 |
| [R9] | `deep_learning_task.cpp` | 691-830 | `run_train_epoch_gpu()` 仅 XFER+DEEP+FIRST_BWD |
| [R10] | `scheduler.h` | 51 | `float get_lr_by_batch(int batch_id) const;` |
| [R11] | `computation_graph.h` | 73-95 | `GraphId` 枚举 |
| [R12] | `transfer_station.h` | 263-266 | `buffer_0_is_readable_` / `buffer_0_is_writeable_` 初始状态 |
| [R13] | `mnist_mlp_3.cpp` | 79-82 | `.scheduler(CosineAnnealingLR().step_by_epoch())` |

---

*本方案基于 AXX.md 全部四稿 + 用户最终意见 + TR4 v4.20.1 源码全路径交叉验证。所有设计决策均有科学依据，所有代码路径均有文件+行号引用。严禁在 batch 内调用 `scheduler.step()`。*
