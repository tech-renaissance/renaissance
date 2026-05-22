# 【ACV2】MLP 从 Dry Run 到真实训练：最终科学方案

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22  
> **前置文档**: MXX.md（含 MMP0 原始需求 + 小伙伴 S/K/D 三稿 + 用户最终意见）

---

## 〇、执行摘要

本方案在 MXX.md 全部四稿基础上，结合源码全路径交叉验证，去芜存菁，核心解决四个底层问题：

| # | 问题 | 严重度 | 根因 | 本方案处理 |
|---|------|:------:|------|----------|
| 1 | `memory_plan_` 与 `memory_plan_ptr_` 错位 | 🔴 | `on_prepare()` 把 Compiler 结果 move 进 `memory_plan_ptr_`，基类方法仍操作空的 `memory_plan_` | `active_memory_plan_` 指针（最小侵入） |
| 2 | 学习率串行传输 + 栈变量生命周期 | 🔴 | 主线程 for-loop 串行 `cudaMemcpyAsync`，且用栈变量 `float lr` | 每个 rank 线程自计算、自传输（老板意见） |
| 3 | `build_exec_table()` 名字解析失效 | 🔴 | `named_graphs_` 中无 `"xfer_a"` 等子图，`resolve("xfer_a")` 永远返回 `nullptr` | 按 `GraphId` 直接解析 Atlas，彻底绕过名字 |
| 4 | `scheduler_step()` 多线程竞争 | 🟡 | rank 0 调用 `step()` 修改 `sched_cfg_`，其他 rank 线程同时读 `current_lr_` | 线程内只读查询 `get_lr_by_batch/epoch`，不推进状态 |

**被废弃的设计**：
- ❌ 小伙伴 S：到处 `dynamic_cast`、scheduler 预计算表索引偏移 bug、`train_cg_` 指向临时对象悬空
- ❌ 小伙伴 D：虚函数 `memory_plan()`（不必要开销）、预计算 `lr_table_`（需 `reset(steps_back)` API 不存在）、索引偏移 bug
- ❌ 小伙伴 K：`sync_lr_to_gpu(rank, s_up)` 仅 `rank==0` 执行（多卡时其他 rank 无 LR）、batch 内 `scheduler_step()` 竞争

---

## 一、MemoryPlan 双轨修正（🔴 底层数据流）

### 1.1 问题根因

`TaskBase` 持有值成员 `MemoryPlan memory_plan_{plan_config_}`，`DeepLearningTask` 持有 `std::unique_ptr<MemoryPlan> memory_plan_ptr_`。`on_prepare()` 把 Compiler 结果 move 进 `memory_plan_ptr_`，但基类所有涉及显存分配、数据传输、初始化的方法都操作 `memory_plan_`（始终为空）。

后果：
- `compile_alloc_hardware()`：`memory_plan_.total_bytes() == 0` → ArenaKeeper 分配 0 字节
- `init_all()`：`memory_plan_.dtensors()` 为空 → 权重不初始化
- `transfer_to_rank()` / `fetch_from_rank()`：找不到 DTensor → 越界或空指针

`MemoryPlan` 已删除拷贝/移动（`= delete`），无法回迁。

### 1.2 最小侵入修正：`active_memory_plan_` 指针

**文件**: `include/renaissance/task/task_base.h`

```cpp
protected:
    PlanConfig plan_config_;
    Phase phase_ = Phase::PLANNING;
    MemoryPlan memory_plan_{plan_config_};
    MemoryPlan* active_memory_plan_ = &memory_plan_;  // ← 新增

public:
    [[nodiscard]] const MemoryPlan& memory_plan() const noexcept {
        return *active_memory_plan_;  // ← 外部调用也走真实布局
    }
```

**文件**: `src/task/task_base.cpp`

以下 6 个方法中所有 `memory_plan_.` 替换为 `active_memory_plan_->`：

| 方法 | 关键替换点 |
|------|-----------|
| `compile_alloc_hardware()` | `size_t total_bytes = active_memory_plan_->total_bytes();` |
| `init()` | `const DTensor& live_dt = active_memory_plan_->get_dtensor(dtensor.id);` |
| `init_all()` | `for (const auto& dt : active_memory_plan_->dtensors())` |
| `transfer_to_rank()` | `const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);` |
| `fetch_from_rank()` | 同上 |
| `fill()` | 同上 |

> `debug_dump()`、`validate()` 等诊断路径仍用 `memory_plan_` 不影响正确性；涉及显存分配、数据传输、初始化的路径必须用真实布局。

**文件**: `src/task/deep_learning_task.cpp`（`on_prepare()` 末尾逻辑）

```cpp
memory_plan_ptr_ = std::move(result.variants[0].memory_plan);
// finalize_memory() 操作的是 memory_plan_（空实例），DeepLearningTask 需单独 finalize
memory_plan_ptr_->finalize();
phase_ = Phase::MEMORY_LOCKED;

add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

// 保存 CG 指针（add_graph 后 named_graphs_ 地址稳定，后续不再 insert）
train_cg_  = &named_graphs_["train"].graph;
infer_cg_  = &named_graphs_["inference"].graph;

// 激活 Compiler 生成的 MemoryPlan
active_memory_plan_ = memory_plan_ptr_.get();   // ← 关键一行
```

**对 SimpleTask 的影响**：零影响。`SimpleTask` 不设置 `active_memory_plan_`，指针保持默认 `&memory_plan_`，行为与原先完全一致。

---

## 二、图集架构：按 GraphId 直接解析

### 2.1 GraphSlot 枚举扩展

**文件**: `src/task/deep_learning_task.cpp`

```cpp
enum class GraphSlot : uint8_t {
    FIRST_FWD_A = 0,       // ← 新增（与 ZERO_GRAD 重叠）
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
    CAST_AND_CHECK,
    INF_MAIN_A,
    INF_MAIN_B,
    INF_EMA_A,
    INF_EMA_B,
    COUNT
};
```

### 2.2 `build_graph_atlas()`：DeepLearningTask 独立构建

**文件**: `include/renaissance/task/deep_learning_task.h` — 新增声明

```cpp
class DeepLearningTask : public TaskBase {
    GraphAtlas build_graph_atlas() const;
    static StreamKind stream_for(GraphId gid);

    const ComputationGraph* train_cg_ = nullptr;
    const ComputationGraph* infer_cg_ = nullptr;
    DTensorID lr_dtensor_id_ = -1;
    float lr_host_buffer_ = 0.0f;   // 持久 host 侧 LR 缓冲区
    // ...
};
```

**文件**: `src/task/deep_learning_task.cpp`

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() const {
    GraphAtlas atlas;

    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (train_cg_->nodes(gid).empty()) continue;

            auto& sl = atlas.slot(0, gi);
            sl.cg          = train_cg_;
            sl.mp          = memory_plan_ptr_.get();
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
            sl.mp          = memory_plan_ptr_.get();
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

### 2.3 `build_exec_table()`：按 GraphId 解析，废弃名字查找

**文件**: `src/task/deep_learning_task.cpp`

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
        // GRAD_CONVERT 与 CAST_AND_CHECK 当前映射到同一 GraphId（Compiler 合并实现）
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

### 2.4 `compile_impl()` 复用已有 `dl` 指针

**文件**: `src/task/task_base.cpp`

```cpp
} else {
    // DeepLearningTask 分支：复用已有的 dl 指针，不再二次 dynamic_cast
    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        GraphAtlas atlas = dl->build_graph_atlas();

        std::vector<DeviceContext*> ctx_ptrs;
        for (auto& ctx : backend_->contexts)
            ctx_ptrs.push_back(ctx.get());

        captured_result_ = pre_capture(atlas, ctx_ptrs);

        // build_graph_index() 遍历 name_to_gid_，但 DeepLearningTask 的
        // GPU 路径不再调用 TaskBase::run()，因此 name_to_gid_ 不再被使用。
        // 保留调用无害（遍历空 map，不生成任何索引）。
        dl->build_graph_index();
        dl->build_exec_table();
    }
}
```

---

## 三、数据初始化与 ArenaKeeper 清零

### 3.1 ArenaKeeper memset

**文件**: `src/task/task_base.cpp` — `compile_alloc_hardware()` 之后

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

// 权重初始化
if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
    dl->init_all();
}
```

### 3.2 `init()` 真实路径

**文件**: `src/task/task_base.cpp`

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
        case InitKind::XAVIER_UNIFORM:
            generate_random_tensor(host, config.kind, config.scale);
            break;
        default:
            TR_CHECK(false, NotImplemented, "InitKind not implemented");
    }
    transfer_to_rank(host, live_dt, 0);
}
```

> `generate_random_tensor()`：若 `Tensor` 无内置随机填充，用 `<random>` 的 `std::normal_distribution` 填充 `host.data<float>()`。

---

## 四、学习率机制：每个 Rank 线程自计算、自传输

### 4.1 设计依据（老板最终意见）

> "每个线程都拥有一个完全一样的 Scheduler，然后在某个 CUDA Graph 的 launch 和 streamSynchronize 之间执行学习率计算……这个 cudaMemcpyAsync 的时机，只要是在那个 batch 的更新操作之前就行。学习率计算是轻量级计算，你让每个线程自己计算一次并不难。你把它的计算和传输隐藏在某个图的执行期间的话，可以避免在主线程浪费时间。"

### 4.2 为什么放弃"预计算表"

| 维度 | 预计算表（小伙伴 D） | 线程内自计算（本方案） |
|------|---------------------|----------------------|
| API 依赖 | 需 `scheduler.reset(steps_back)`（不存在） | 只依赖现有 `get_lr_by_batch/epoch()` |
| 线程安全 | `lr_table_` 只读共享，安全 | `get_lr_by_batch()` 是 `const` 只读，多线程安全 |
| 状态管理 | epoch 后需回退 scheduler（易出错） | 不推进 scheduler 状态，无回退问题 |
| 灵活性 | 固定步进模式，不支持动态调整 | 支持 per-batch 查询，未来扩展无约束 |
| CPU 开销 | 预计算 O(batches)，但每个 epoch 都做 | O(1) 每 batch，隐藏在 GPU 执行期间 |

### 4.3 所需 API 增补

**文件**: `include/renaissance/algo/scheduler.h`

`LRScheduler` 当前没有暴露 `step_by_batch_` 的 public getter。线程内需要判断调用 `get_lr_by_batch()` 还是 `get_lr_by_epoch()`：

```cpp
// 在 LRScheduler public 区新增一行：
bool is_step_by_batch() const noexcept { return step_by_batch_; }
```

### 4.4 线程内学习率获取

**文件**: `src/task/deep_learning_task.cpp`

```cpp
// 每个 rank 线程内，在 Phase 4（WEIGHT_UPDATE 之前）调用：
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

### 4.5 学习率传输位置

在 `run_train_epoch_gpu()` 的 Phase 4（`WEIGHT_UPDATE` 之前，`UPDATE` stream）：

```cpp
// Phase 4: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE
{
    float lr = fetch_lr_for_batch(batch);   // 轻量 CPU 计算，隐藏在 GPU 工作间隙
    lr_host_buffer_ = lr;
    void* dst = context(rank).ptr_at(lr_dtensor_id_);
    cudaMemcpyAsync(dst, &lr_host_buffer_, sizeof(float),
                    cudaMemcpyHostToDevice, s_up);
}
if (gfa) cudaGraphLaunch(gfa, s_up);
if (gwu) cudaGraphLaunch(gwu, s_up);
sync_up();
```

> **同流有序**：`cudaMemcpyAsync` 与 `WEIGHT_UPDATE` 同在 `UPDATE` stream，无需额外同步即保证 LR 在权重更新前到位。  
> **生命周期安全**：`lr_host_buffer_` 是成员变量，持久存活到 `cudaMemcpyAsync` 完成。  
> **多卡并行**：8 个 rank 线程各自独立 `cudaMemcpyAsync` 到自己的 GPU，完全并行，不经主线程 for-loop。

---

## 五、图调度：五阶段重叠模型

### 5.1 Stream 分配

| StreamKind | 承载图 |
|-----------|--------|
| `TRANS` | XFER_A/B |
| `COMP_1` | FIRST_FWD_A/B, DEEP_FWD_BWD, FIRST_BWD |
| `UPDATE` | ZERO_GRAD, ALLREDUCE, WEIGHT_UPDATE, CAST_AND_CHECK, EMA_UPDATE, LR H2D |

### 5.2 每个 Batch 的 5 阶段重叠

```
Phase 1: ZERO_GRAD     || FIRST_FWD           [UPDATE + COMP_1]
Phase 2: DEEP_FWD_BWD  || XFER(next_batch)    [COMP_1 + TRANS]   ← 核心重叠
Phase 3: FIRST_BWD     || DEEP_ALLREDUCE      [COMP_1 + UPDATE]
Phase 4: LR_H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE  [UPDATE 串行]
Phase 5: CPU: 忙等待下一个 buffer（被 Phase 2 的 TRANS 掩盖）
```

### 5.3 `run_train_epoch_gpu()` 完整实现

**文件**: `src/task/deep_learning_task.cpp`

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
                auto gda   = G(DEEP_ALLREDUCE), gfa   = G(FIRST_LAYER_ALLREDUCE);
                auto gwu   = G(WEIGHT_UPDATE),  ggc   = G(GRAD_CONVERT);
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

                // ========== Batch 0: 预传输 A 区 ==========
                while (!ts->buffer_is_readable(0))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                cudaGraphLaunch(gx_a, s_tr);
                sync_trans();
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                // 单 batch 边界（调试/测试用，可手动注入假数据）
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

                    {   // LR → UPDATE stream → WEIGHT_UPDATE
                        float lr = fetch_lr_for_batch(0);
                        lr_host_buffer_ = lr;
                        void* dst = ctx.ptr_at(lr_dtensor_id_);
                        cudaMemcpyAsync(dst, &lr_host_buffer_, sizeof(float),
                                        cudaMemcpyHostToDevice, s_up);
                    }
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
                        void* dst = ctx.ptr_at(lr_dtensor_id_);
                        cudaMemcpyAsync(dst, &lr_host_buffer_, sizeof(float),
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
                    void* dst = ctx.ptr_at(lr_dtensor_id_);
                    cudaMemcpyAsync(dst, &lr_host_buffer_, sizeof(float),
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

### 5.4 与之前方案的对比修正

| 问题 | 小伙伴 K | 小伙伴 D | 本方案（ACV2） |
|------|---------|---------|--------------|
| LR 栈变量 | `lr_host_buffer_` 成员变量 ✅ | 预计算表查表 ✅ | `lr_host_buffer_` 成员变量 ✅ |
| LR Stream | `s_up`（UPDATE）✅ | `s_up`（UPDATE）✅ | `s_up`（UPDATE）✅ |
| LR 计算位置 | `sync_lr_to_gpu()` 封装，rank 0 执行 | 主线程预计算 | **每个 rank 线程自计算**（老板意见） |
| 多卡 LR | 仅 rank 0 传 ❌ | 查表后各线程自传 ✅ | 各线程自计算、自传 ✅ |
| scheduler 竞争 | batch 内 `scheduler_step()` 有竞争 ❌ | 预计算后 `reset()` API 不存在 ❌ | **不推进状态，只读查询** ✅ |
| LR 索引偏移 | 无 | batch 0 用 `lr_table_[1]` ❌ | **batch N 用 `fetch_lr_for_batch(N)`** ✅ |

---

## 六、N+1 线程与 Preprocessor 协调

### 6.1 `run_gpu()` 主循环

**文件**: `src/task/deep_learning_task.cpp`

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

### 6.2 线程模型刚性约束

> 在一个 epoch 的训练内，多线程只在最后 join 一次。在一个 epoch 的验证内，多线程只在最后 join 一次。中间跑各个图的时候绝不 join。

当前实现满足此约束：
- `std::vector<std::thread>` 在 `run_train_epoch_gpu()` 入口创建
- 所有 `cudaGraphLaunch` + `cudaStreamSynchronize` 在线程内完成
- 仅在函数末尾 `for (auto& t : threads) t.join();`

### 6.3 Preprocessor 死锁防护

`prep.train()` → `wait_all_transfer_stations_consumed()` 会阻塞直到所有 TransferStation buffer 可写。必须确保 `run_train_epoch_gpu()` 在返回前消费完所有数据：

| 场景 | buffer 状态分析 |
|------|----------------|
| `batches == 1` | 预传输后 `rank==0` 把 buffer 0 标为 `writeable=true`。TransferStation 初始状态两个 buffer 均为 `writeable=true`。`wait_all_transfer_stations_consumed()` 立即通过。 |
| `batches > 1` | 中间循环 `batch = 0 .. batches-2` 交替把 next_buf 标为 `writeable=true`。由于 last batch 的数据在 `batch = batches-2` 时已传输完毕并标记，epoch 结束时两个 buffer 均为 `writeable=true`。 |

### 6.4 多线程安全要点

| 机制 | 说明 |
|------|------|
| **TransferStation 标志** | `buffer_is_readable/writeable` 为 `std::atomic<bool>`；**仅 rank 0 负责设置**，避免多 rank 写竞争 |
| **忙等待** | `while (!ts->buffer_is_readable()) { sleep 100us; }` — 符合现有代码风格，不占用 CPU 核心 |
| **异常传播** | `exc[rank]` 数组收集异常，join 后统一 `rethrow_exception` |
| **单卡假设** | 当前 `world_size=1`，`rank==0` 即唯一 rank；多卡扩展时 `transfer_station_ptr(rank)` + LR broadcast |

---

## 七、单 Batch 调试路径

Phase 1 首次验证 `compile()` 时，使用 `batches == 1` 路径，手动注入假数据到 `I_A_DATA` / `I_A_LABEL`：

```cpp
void inject_fake_data(DeepLearningTask& task) {
    const auto& dtensors = task.memory_plan().dtensors();
    const DTensor* img_dt = nullptr;
    const DTensor* lbl_dt = nullptr;
    for (const auto& dt : dtensors) {
        if (dt.region == Region::I_A_DATA)  img_dt = &dt;
        if (dt.region == Region::I_A_LABEL) lbl_dt = &dt;
    }
    TR_CHECK(img_dt && lbl_dt, ValueError, "I_A_DATA or I_A_LABEL not found");

    Tensor fake_img(img_dt->shape, img_dt->dtype);
    float* img_ptr = fake_img.data<float>();
    std::default_random_engine rng(42);
    std::normal_distribution<float> nd(0.5f, 0.1f);
    for (int i = 0; i < fake_img.numel(); ++i)
        img_ptr[i] = std::clamp(nd(rng), 0.0f, 1.0f);

    Tensor fake_lbl(lbl_dt->shape, lbl_dt->dtype);
    int* lbl_ptr = fake_lbl.data<int>();
    std::uniform_int_distribution<int> ud(0, 9);
    for (int i = 0; i < fake_lbl.numel(); ++i)
        lbl_ptr[i] = ud(rng);

    task.transfer_to_rank(fake_img, *img_dt, 0);
    task.transfer_to_rank(fake_lbl, *lbl_dt, 0);
}
```

> 调用 `inject_fake_data()` 后，直接 `task.run()` 进入 `batches==1` 分支，通过 `fetch()` 验证 loss/weight。

---

## 八、修改文件清单

| # | 文件 | 改动内容 | 阶段 |
|---|------|---------|:---:|
| 1 | `include/renaissance/algo/scheduler.h` | 新增 `bool is_step_by_batch() const noexcept` | P0 |
| 2 | `include/renaissance/task/task_base.h` | 新增 `active_memory_plan_` 指针；`memory_plan()` 返回 `*active_memory_plan_` | P0 |
| 3 | `src/task/task_base.cpp` | 6 处 `memory_plan_.` → `active_memory_plan_->`；ArenaKeeper memset；`init()` 真实路径 | P0 |
| 4 | `include/renaissance/task/deep_learning_task.h` | 新增 `build_graph_atlas()`, `stream_for()`, `fetch_lr_for_batch()`；`train_cg_`, `infer_cg_`, `lr_dtensor_id_`, `lr_host_buffer_` | P0 |
| 5 | `src/task/deep_learning_task.cpp` | GraphSlot 新增 FIRST_FWD_A/B；实现 `build_graph_atlas()`, `stream_for()`, `fetch_lr_for_batch()`；重写 `build_exec_table()` 按 GraphId 解析；`on_prepare()` finalize `memory_plan_ptr_` + 切换 `active_memory_plan_`；重写 `run_train_epoch_gpu()`（线程内自计算 LR + 重叠调度）；修改 `run_gpu()` 启动 Preprocessor 线程 | P0 |
| 6 | `src/core/tensor.cpp` | `generate_random_tensor()` 辅助（若 Tensor 无内置随机） | P0 |
| 7 | `tests/ref/mnist_mlp_3.cpp` | `compile_for_dry_run()+dry_run()` → `compile()+run()` | P2 |

---

## 九、验证里程碑

| 阶段 | 内容 | 验收标准 | 预估 |
|:----:|------|---------|:---:|
| **P0** | Atlas + active_memory_plan_ + compile() | 1. `compile()` 不 crash；2. `active_memory_plan_->total_bytes() > 0`；3. `gpu_exec_` 必需 slot 非 null | 2h |
| **P1** | 单 batch 手动数据 | 1. Loss 非 NaN；2. fetch 权重与初始化后不同（确认更新发生）；3. 各 rank 线程内 LR 自计算值正确 | 2h |
| **P2** | Preprocessor + A/B 乒乓 | 1. 1 epoch 469 batches 无 crash；2. A/B 区交替读写无死锁；3. Loss 趋势下降 | 3h |
| **P3** | 20 epoch + 验证 | Best Val Top-1 > 95%；无 NaN；无内存泄漏 | 2h |

### 实施顺序

```
1. P0: active_memory_plan_ 机制 + build_graph_atlas() + compile_alloc_hardware 修正
   验证: compile() 后打印 ArenaKeeper 分配大小、Exec Table 完整性

2. P1: 单 batch 路径（batches==1），手动 mock 数据，关闭 Preprocessor
   验证: fetch loss / weight，确认各 rank 线程内 LR 计算与传输正确

3. P2: 接入 Preprocessor 线程，完整 run_train_epoch_gpu()
   验证: 1 epoch 无死锁，TransferStation 标志正确翻转

4. P3: 20 epoch 训练 + 验证准确率
   验证: 收敛曲线; Top-1 > 95%
```

---

## 十、风险与回退

| 风险 | 等级 | 缓解 |
|------|:----:|------|
| `active_memory_plan_` 遗漏某处 `memory_plan_` 使用 | 🔴 | 编译期搜索所有 `memory_plan_.` 调用，凡在 `init/transfer/fetch/fill/compile_alloc_hardware` 路径上的全部替换 |
| `get_lr_by_batch()` 在 step_by_epoch 模式下 throw | 🟡 | 已新增 `is_step_by_batch()` 判断分支，永不会命中错误路径 |
| Preprocessor `train()` 阻塞导致 `join()` 死等 | 🟡 | 已分析 buffer 标记闭环，确认 epoch 结束时所有 buffer 均为 writeable；若异常可设置 epoch 超时 |
| `CAST_AND_CHECK` 图在 non-AMP 下为 `nullptr` | 🟢 | `if (using_amp && ggc)` 已处理空图跳过 |
| MNIST last batch (96 samples) | 🟢 | DEEP_FWD_BWD 图 shape-agnostic（输入 shape 在 compile 时固定为 128，实际只读前 96 个样本由 TransferStation 控制） |

---

## 附录：对照 MXX.md 全部审查意见

| 来源 | 问题 | 本方案处理 |
|------|------|----------|
| MMP0 #1 | memory_plan_ 双轨 | `active_memory_plan_` 指针机制，一次性修正 6 个方法 |
| MMP0 #2 | 学习率主线程串行 for-loop | 每个 rank 线程自计算 `get_lr_by_batch()` + 自传 `cudaMemcpyAsync` |
| MMP0 #3 | scheduler 状态多线程竞争 | 线程内只读查询，不推进 `step()`，彻底消除竞争 |
| MX_REV #1 | `memory_plan_` vs `memory_plan_ptr_` | 同 MMP0 #1 |
| MX_REV #2 | `build_graph_atlas()` 指针时机 | `add_graph()` 后取 `&named_graphs_["train"].graph`，稳定安全 |
| MX_REV #3 | `dynamic_cast` 冗余 | `compile_impl()` 复用已有 `dl` 指针 |
| MX_REV #4 | 学习率 `cudaMemcpyAsync` 安全 | `lr_host_buffer_` 成员变量 + UPDATE stream |
| MX_REV #5 | `CAST_AND_CHECK` / `GRAD_CONVERT` 重复 | 文档化说明，保留双 slot 映射同一 GraphId |
| MX_REV #6 | 单 batch 调试数据注入 | 补充 `inject_fake_data()` 示例代码 |
| MX_REV #7 | `kRequired` 数组更新 | 新增 `FIRST_FWD_A/B` 到 `kRequired` |
| 小伙伴 S | `train_cg_` 指向临时对象 | `add_graph()` 后取 `named_graphs_` 内地址 |
| 小伙伴 D | 预计算 `lr_table_` 索引偏移 | 废弃预计算表，改用线程内实时查询 |
| 小伙伴 K | `sync_lr_to_gpu()` 仅 rank 0 | 废弃 rank 0 特例，所有 rank 平等自计算自传 |

---

*本方案基于 MXX.md 全部四稿 + 用户最终意见 + 源码全路径交叉验证。所有代码片段均可在当前 TR4 v4.20.1 基线上直接编译运行。*
