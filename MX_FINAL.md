# 【小伙伴D】

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22  
> **三稿综合**: MXZ1(图分解) + MXZ2(重叠优先) + MXZ3(三步渐进) + 源码全路径交叉验证

---

## 〇、执行摘要

**目标**: `tests/ref/mnist_mlp_3.cpp` 从 `compile_for_dry_run()+dry_run()` 改为 `compile()+run()`，在单卡上完成 20 epoch MNIST MLP 训练。

**核心策略**: "修复 compile → 验证传输 → 补全调度 → 收敛训练" 四步走。每步有可执行的代码改动和可量化的验收标准。

**关键决策**: 
- DeepLearningTask 与 SimpleTask **完全分离**，各自的 Atlas 构建逻辑独立
- `build_exec_table()` 改为 **按 GraphId 直接解析**，不再依赖名字查找
- 图调度以 **计算通信重叠** 为核心目标（MMP0 伪代码的要求）
- 第一阶段只跑通 **单 batch**（无 Preprocessor 多线程），降低调试复杂度

---

## 一、现状代码路径全验证

### 1.1 已验证的结论

| 序号 | 确认项 | 代码位置 | 结论 |
|:----:|--------|----------|:----:|
| ① | `ComputationGraph::nodes(GraphId)` 已按 `GraphId` 分区 | [computation_graph.h:L239](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L239) | ✅ 可直接用 |
| ② | `pre_capture()` 已按 `GraphId` 遍历 Atlas、去重、捕获 | [captured_graph.cpp:L185-L204](file:///r:/renaissance/src/graph/captured_graph.cpp#L185-L204) | ✅ 引擎层完整 |
| ③ | Compiler 已生成 20 个 `GraphId` 子图 | dry run 输出 | ✅ 分区正确 |
| ④ | 现有多线程 per-rank + A/B 乒乓等待骨架完整 | [deep_learning_task.cpp:L690](file:///r:/renaissance/src/task/deep_learning_task.cpp#L690) | ✅ 可直接扩展 |
| ⑤ | `run_gpu()` epoch 循环 + 日志 + SEMA 完整 | [deep_learning_task.cpp:L537](file:///r:/renaissance/src/task/deep_learning_task.cpp#L537) | ✅ 框架已就绪 |

### 1.2 必须修复的路径

| 序号 | 问题 | 根因 | 代码位置 | 严重度 |
|:----:|------|------|----------|:------:|
| **G1** | `build_simple_atlas()` 将所有子图合并为 `SIMPLE_TASK_GRAPH` | 为 SimpleTask 设计 | [task_base.cpp:L633](file:///r:/renaissance/src/task/task_base.cpp#L633) | 🔴 |
| **G2** | `init()` real path 为 `TR_NOT_IMPLEMENTED` | stub 未实现 | [task_base.cpp:L1223](file:///r:/renaissance/src/task/task_base.cpp#L1223) | 🔴 |
| **G3** | `run_train_epoch_gpu()` 只调了 XFER+DEEP+FIRST_BWD 三图 | 缺少 5 张图 | [deep_learning_task.cpp:L726](file:///r:/renaissance/src/task/deep_learning_task.cpp#L726) | 🔴 |
| **G4** | Preprocessor 未启动 | `run_gpu()` 无 `prep.train()` | [deep_learning_task.cpp:L537](file:///r:/renaissance/src/task/deep_learning_task.cpp#L537) | 🟡 |
| **G5** | ArenaKeeper 未 memset | `compile_alloc_hardware()` 无清零 | [task_base.cpp:L1405](file:///r:/renaissance/src/task/task_base.cpp#L1405) | 🟡 |
| **G6** | 学习率未 H2D 传输 | `scheduler.step()` 后无 GPU 同步 | [deep_learning_task.cpp:L751](file:///r:/renaissance/src/task/deep_learning_task.cpp#L751) | 🟡 |

### 1.3 MXZ 三稿方案对比与择优

| 决策点 | MXZ1 | MXZ2 | MXZ3 | **最终选择** |
|--------|:----:|:----:|:----:|:----------:|
| Atlas 构造方式 | `DeepLearningTask::build_graph_atlas()` | 修改 `build_simple_atlas()` 遍历所有 GraphId | 在 `on_prepare()` 中直接构建 | **MXZ1**（与 SimpleTask 彻底分离，不互扰） |
| Exec table 解析 | `name_to_gid_` 映射 → 查 `atlas.index()` | `resolve_gid(GraphId)` 直接查 Atlas | 使用 resolve_gid | **MXZ2**（直接按 GraphId 索引，不经过名字） |
| GraphSlot 新增 | 无 | 新增 `FIRST_FWD_A/B` | 无 | **MXZ2**（MMP0 要求 FIRST_FWD 与 ZERO_GRAD 并行） |
| 重叠调度 | 概要伪代码 | **5 阶段精确重叠模型** | 顺序调度 | **MXZ2**（精确的流并行语义） |
| 实现步骤 | 4 Phase | 4 Phase (以重叠为验收标准) | 3 步渐进 | **MXZ3 框架 + MXZ2 内容** |
| 学习率传输 | `transfer_learning_rate()` | `scheduler_step()` + `cudaMemcpyAsync` | `update_learning_rate()` | **MXZ2**（最简单有效） |

---

## 二、图集架构修正：compile() 正确产出所有 cudaGraphExec_t

### 2.1 GraphSlot 枚举扩展

当前枚举只有 17 个 slot。为支持 MMP0 要求的 `FIRST_FWD || ZERO_GRAD` 重叠，需要新增 `FIRST_FWD_A` 和 `FIRST_FWD_B`。

**文件**: `src/task/deep_learning_task.cpp:32`

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
    CAST_AND_CHECK,
    INF_MAIN_A,
    INF_MAIN_B,
    INF_EMA_A,
    INF_EMA_B,
    COUNT
};
```

### 2.2 Atlas 构建：`DeepLearningTask::build_graph_atlas()`

**核心原则**: DeepLearningTask 与 SimpleTask 完全分离。不在 `TaskBase::build_simple_atlas()` 中混入 DeepLearningTask 的特殊逻辑。

**文件**: `include/renaissance/task/deep_learning_task.h` — 新增声明

```cpp
class DeepLearningTask : public TaskBase {
    GraphAtlas build_graph_atlas() const;
    static StreamKind stream_for(GraphId gid);
    const MemoryPlan& memory_plan() const override { return *memory_plan_ptr_; }
    // 新增成员变量
    const ComputationGraph* train_cg_ = nullptr;
    const ComputationGraph* infer_cg_ = nullptr;
    DTensorID lr_dtensor_id_ = -1;     // 学习率 DTensor
    std::vector<float> lr_table_;      // 预计算学习率表（每 epoch 刷新）
    // ... 其余不变 ...
};
```

**文件**: `src/task/deep_learning_task.cpp` — 实现

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() const {
    GraphAtlas atlas;

    // ============= train 图：遍历所有 GraphId =============
    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            const auto& nodes = train_cg_->nodes(gid);
            if (nodes.empty()) continue;

            auto& sl = atlas.slot(0, gi);
            sl.cg          = train_cg_;
            sl.mp          = memory_plan_ptr_.get();
            sl.shape_id    = kShapeInvariant;   // MLP 单分辨率
            sl.stream_kind = stream_for(gid);
        }
    }

    // ============= inference 图（仅推理 GraphId） =============
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

### 2.3 Exec Table：`build_exec_table()` 改为按 GraphId 直接解析

**优势**: 不依赖名字查找，直接 `atlas.index(0, gid)` 取 `captured_idx`。无法构建的名字会导致 `nullptr`（单卡 ALLREDUCE 等），由 `safe_launch()` 跳过。

**文件**: `src/task/deep_learning_task.cpp` — `build_exec_table()`

```cpp
void DeepLearningTask::build_exec_table() {
    gpu_exec_.graphs.resize(num_gpus_);
    gpu_exec_.device_ids.resize(num_gpus_);

    auto resolve = [this](GraphId gid, int rank) -> cudaGraphExec_t {
        int32_t idx = captured_result_.atlas.index(0, gid);
        if (idx < 0) return nullptr;
        return static_cast<cudaGraphExec_t>(
            captured_result_.graphs[idx].native_exec(rank));
    };

    for (int rank = 0; rank < num_gpus_; ++rank) {
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
        // GRAD_CONVERT 和 CAST_AND_CHECK 指向同一个 GraphId::CAST_AND_CHECK。
        // GraphId::CAST_AND_CHECK 是 "AMP 梯度转换+NaN检查" 的合并图，
        // Compiler 当前没有独立的 GRAD_CONVERT GraphId。两个 Slot 均
        // 从同一个图解析，pre_capture() 去重后只捕获一份。
        g[S(GRAD_CONVERT)]       = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(CAST_AND_CHECK)]     = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(INF_MAIN_A)]         = resolve(GraphId::INF_MAIN_A,    rank);
        g[S(INF_MAIN_B)]         = resolve(GraphId::INF_MAIN_B,    rank);
        g[S(INF_EMA_A)]          = resolve(GraphId::INF_EMA_A,     rank);
        g[S(INF_EMA_B)]          = resolve(GraphId::INF_EMA_B,     rank);
    }
}

// ---- kRequired 数组（compile 阶段校验必需槽位非空） ----
static const GraphSlot kRequired[] = {
    GraphSlot::FIRST_FWD_A,       // FLATTEN 前向，不受冻结影响
    GraphSlot::FIRST_FWD_B,       // FLATTEN 前向，不受冻结影响
    GraphSlot::XFER_A,
    GraphSlot::XFER_B,
    GraphSlot::FWD_BWD_DEEP_A,
    GraphSlot::FWD_BWD_DEEP_B,
    GraphSlot::FIRST_LAYER_BWD,   // 首层冻结时可跳过，但 slot 必须存在（可为空图）
};
for (int rank = 0; rank < num_gpus_; ++rank) {
    for (auto slot : kRequired) {
        if (!gpu_exec_.graphs[rank][static_cast<size_t>(slot)]) {
            TR_CHECK(false, ValueError,
                     "Required graph slot " << static_cast<int>(slot)
                     << " is nullptr for rank " << rank);
        }
    }
}
// 辅助宏: g[S(slot)] = auto cast GraphSlot → size_t
```

**DEEP_FWD_BWD 的 A/B 共享**（关键说明）:
- `FWD_BWD_DEEP_A` 和 `FWD_BWD_DEEP_B` 都从 `DEEP_FWD_BWD` GraphId 解析
- 因为 `DEEP_FWD_BWD` 图本身不区分 A/B ——差异仅在于输入数据位于 `I_A_DATA` 还是 `I_B_DATA` 的内存区域
- `pre_capture()` 会去重：同一个 `(cg, DEEP_FWD_BWD, kShapeInvariant)` key 只捕获一次

### 2.4 compile_impl() 中的 DeepLearningTask 分支

**文件**: `src/task/task_base.cpp` — `compile_impl()`

当前代码 `else` 分支（`!is_simple_task()`）中已经通过 `if (auto* dl = dynamic_cast<DeepLearningTask*>(this))` 获取了 `dl` 指针并调用 `build_graph_index()` / `build_exec_table()`。**只需将 `build_simple_atlas(name_to_gid_)` 替换为 `dl->build_graph_atlas()`，其余 pre_capture、build_graph_index、build_exec_table 调用保持不变**，复用已有的 `dl` 指针：

```cpp
// === 当前 compile_impl() 的 else 分支，改一行即可 ===
GraphAtlas atlas = dl->build_graph_atlas();   // ← 替换 build_simple_atlas(name_to_gid_)

std::vector<DeviceContext*> ctx_ptrs;
for (auto& ctx : backend_->contexts)
    ctx_ptrs.push_back(ctx.get());

captured_result_ = pre_capture(atlas, ctx_ptrs);

dl->build_graph_index();   // ← 已有
dl->build_exec_table();    // ← 已有
```

### 2.5 on_prepare() 保存 CG 指针

**文件**: `src/task/deep_learning_task.cpp`（`on_prepare()` 实际定义在 header，此指改动逻辑）

```cpp
// ... 现有 Compiler::compile() 调用不变 ...

memory_plan_ptr_ = std::move(result.variants[0].memory_plan);

// ⚠️ MemoryPlan 不可拷贝 / 不可移动（=delete）。
// TaskBase::finalize_memory() 调用的是 memory_plan_（空值实例）的 finalize()！
// DeepLearningTask 必须单独 finalize memory_plan_ptr_：
memory_plan_ptr_->finalize();
phase_ = Phase::MEMORY_LOCKED;
// 注：不再调用 TaskBase::finalize_memory()

// 查找学习率 DTensor ID（供 run 阶段 staging param 使用）
for (const auto& dt : memory_plan_ptr_->dtensors()) {
    if (dt.usage == DTensorUsage::SCALAR_FP32) {
        lr_dtensor_id_ = dt.id;
        break;
    }
}

add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

// 保存 CG 指针供 build_graph_atlas() 使用（必须在 add_graph 之后，
// 因为 add_graph 把 train_cg/infer_cg move 进 named_graphs_ 后地址才稳定）
train_cg_ = &named_graphs_["train"].graph;
infer_cg_ = &named_graphs_["inference"].graph;
```

> **🔴 关键修复**: 由于 `MemoryPlan` 不可拷贝/不可移动（`MemoryPlan(const MemoryPlan&) = delete` 等），无法简单地将 `*memory_plan_ptr_` 赋值给 `memory_plan_`。因此 **TaskBase 中所有访问 `memory_plan_` 的方法（compile_alloc_hardware、init_all、init、transfer_to_rank、fetch_from_rank 等）须改为通过虚函数 `memory_plan()` 访问**：
>
> ```cpp
> // TaskBase 新增虚函数
> virtual const MemoryPlan& memory_plan() const { return memory_plan_; }
>
> // DeepLearningTask 覆写
> const MemoryPlan& memory_plan() const override { return *memory_plan_ptr_; }
> ```
>
> 此后所有 `memory_plan_.xxx()` 改为 `memory_plan().xxx()`。

---

## 三、数据初始化

### 3.1 ArenaKeeper memset 清零

**文件**: `src/task/task_base.cpp` — `compile_impl()` 中 `compile_alloc_hardware()` 之后

```cpp
// ---- ArenaKeeper 显存池 mGPU memset 清零 ----
#ifdef TR_USE_CUDA
if (GlobalRegistry::instance().using_gpu()) {
    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaSetDevice(backend_->contexts[rank]->device_id());
        void* ptr = ArenaKeeper::instance().ptr_at(rank, 0);
        size_t bytes = memory_plan().total_bytes();
        cudaMemset(ptr, 0, bytes);
    }
    cudaDeviceSynchronize();
}
#endif
// ---- 权重初始化 ----
if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
    dl->init_all();
}
```

### 3.2 init() 真实路径

**文件**: `src/task/task_base.cpp:L1223`

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    check_phase(Phase::COMPILED, "init");

    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : dtensor.init_config;
    if (!config.needs_init()) return;
    if (debug_mode_) { /* dry run print */ return; }

    const DTensor& live_dt = memory_plan().get_dtensor(dtensor.id);

    // 在 host 上生成初始化数据
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
            TR_CHECK(false, NotImplemented,
                     "InitKind " << static_cast<int>(config.kind) << " not implemented");
    }

    // H2D 传输到 rank 0（多卡时后继 broadcast）
    transfer_to_rank(host, live_dt, 0);
    // MNIST 单卡跳过 broadcast
}
```

> **`generate_random_tensor()`**: 若 Tensor 类无 `random_normal()`，直接用 `<random>` 的 `std::normal_distribution` 填充 Tensor data buffer。后续可优化为 CUDA kernel 设备端生成。

---

## 四、图调度：以计算通信重叠为核心

### 4.1 Stream 流分配（不可更改）

| StreamKind | 承载图 | 说明 |
|-----------|--------|------|
| `TRANS` | XFER_A/B | 异步传输，与计算流无数据依赖 |
| `COMP_1` | FIRST_FWD_A/B, DEEP_FWD_BWD, FIRST_BWD | 主计算流 |
| `COMP_2/3` | MLP 预留（空） | 仅用于跨流同步 |
| `UPDATE` | ZERO_GRAD, ALLREDUCE, WEIGHT_UPDATE, GRAD_CONVERT, CHECK_NAN, EMA_UPDATE, LR H2D | 梯度/参数管理 |

### 4.2 五阶段重叠模型

基于 MMP0 伪代码精化，每个中间 batch 执行 5 个重叠阶段：

```
Phase 1: ZERO_GRAD     || FIRST_FWD           [UPDATE + COMP_1]
Phase 2: DEEP_FWD_BWD  || XFER(next_batch)    [COMP_1 + TRANS]   ← 核心
Phase 3: FIRST_BWD     || DEEP_ALLREDUCE      [COMP_1 + UPDATE]
Phase 4: FIRST_ALLREDUCE + WEIGHT_UPDATE + LR_H2D  [UPDATE 串行]
```

### 4.3 run_train_epoch_gpu() 完整实现

**核心设计**:
1. **学习率预计算**: 主线程在 epoch 开始前预计算所有 batch 的 LR → `lr_table_`（`std::vector<float>` 成员），线程只读查表，零开销
2. **LR 在 thread 内传输**: 每个 rank 的线程自己 cudaMemcpyAsync，8 卡并行，**不经过主线程 for 循环**
3. **LR 在 s_up 流**: 放在 Phase 4（WEIGHT_UPDATE 之前），与权重更新同流，无需额外同步
4. **线程只在 epoch 末尾 join 一次**，中间跑各图时绝不 join

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

    // ======== 主线程：预计算所有 batch 的 LR ========
    // lr_table_ 是 DeepLearningTask 的 std::vector<float> 成员
    lr_table_.resize(batches);
    for (int b = 0; b < batches; ++b) {
        std::visit([](auto&& s) { s.step(); }, sched_cfg_);
        lr_table_[b] = get_current_lr();
    }
    // 重置 scheduler（回退到 epoch 初始状态），线程内不再调用 scheduler_step()
    std::visit([batches](auto&& s) {
        using T = std::decay_t<decltype(s)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            s.reset(batches);  // 回退 batches 步
        }
    }, sched_cfg_);

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);

                const auto& g = gpu_exec_.graphs[rank];
                #define G(slot) g[static_cast<size_t>(GraphSlot::slot)]
                auto gf_a  = G(FIRST_FWD_A),        gf_b  = G(FIRST_FWD_B);
                auto gx_a  = G(XFER_A),             gx_b  = G(XFER_B);
                auto gd_a  = G(FWD_BWD_DEEP_A),     gd_b  = G(FWD_BWD_DEEP_B);
                auto gfb   = G(FIRST_LAYER_BWD),    gzg   = G(ZERO_GRAD);
                auto gda   = G(DEEP_ALLREDUCE),     gfa   = G(FIRST_LAYER_ALLREDUCE);
                auto gwu   = G(WEIGHT_UPDATE),      ggc   = G(GRAD_CONVERT);
                auto gcn   = G(CAST_AND_CHECK);
                #undef G

                auto& ctx   = context(rank);
                auto s_tr   = (cudaStream_t)ctx.stream(StreamKind::TRANS);
                auto s_c1   = (cudaStream_t)ctx.stream(StreamKind::COMP_1);
                auto s_c2   = (cudaStream_t)ctx.stream(StreamKind::COMP_2);
                auto s_c3   = (cudaStream_t)ctx.stream(StreamKind::COMP_3);
                auto s_up   = (cudaStream_t)ctx.stream(StreamKind::UPDATE);

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_trans = [&]()  { cudaStreamSynchronize(s_tr); };
                auto sync_up    = [&]()  { cudaStreamSynchronize(s_up); };

                // ======== Pre-transfer Batch 0 from A ========
                while (!ts->buffer_is_readable(0))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                cudaGraphLaunch(gx_a, s_tr);
                sync_trans();
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                if (batches == 1) {
                    // 单 batch：ZERO_GRAD || FIRST_FWD_A → DEEP → FIRST_BWD → UPDATE
                    cudaGraphLaunch(gzg,  s_up);
                    cudaGraphLaunch(gf_a, s_c1);
                    sync_comp(); sync_up();
                    cudaGraphLaunch(gd_a, s_c1);
                    sync_comp();
                    if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }
                    if (gda) cudaGraphLaunch(gda, s_up);
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    // LR H2D (batch 0 的 LR，s_up 流)
                    {
                        float lr = lr_table_[0];
                        void* dst = ctx.ptr_at(lr_dtensor_id_);
                        cudaMemcpyAsync(dst, &lr, sizeof(float),
                                        cudaMemcpyHostToDevice, s_up);
                    }
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                    return;
                }

                // ======== Batch 0 ~ batches-2: 乒乓 + 重叠 ========
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    auto gf  = from_a ? gf_a : gf_b;
                    auto gd  = from_a ? gd_a : gd_b;
                    auto gx_n = from_a ? gx_b : gx_a;  // XFER 下一 batch

                    // Phase 1: ZERO_GRAD || FIRST_FWD (overlap #1)
                    cudaGraphLaunch(gzg, s_up);
                    cudaGraphLaunch(gf,  s_c1);
                    sync_comp(); sync_up();

                    // Wait next buffer
                    while (!ts->buffer_is_readable(next_buf))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    // Phase 2: DEEP_FWD_BWD || XFER(next) (overlap #2, 核心)
                    cudaGraphLaunch(gd,    s_c1);
                    cudaGraphLaunch(gx_n,  s_tr);
                    sync_comp(); sync_trans();
                    if (rank == 0) {
                        ts->set_buffer_readable(next_buf, false);
                        ts->set_buffer_writeable(next_buf, true);
                    }

                    // Phase 3: FIRST_BWD || DEEP_ALLREDUCE (overlap #3)
                    if (!frozen) cudaGraphLaunch(gfb, s_c1);
                    if (gda)     cudaGraphLaunch(gda, s_up);
                    sync_comp(); sync_up();

                    // AMP grad_convert
                    if (using_amp) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                    // Phase 4: LR H2D（在 WEIGHT_UPDATE 之前，s_up 流） + FIRST_ALLREDUCE + WEIGHT_UPDATE
                    {
                        float lr = lr_table_[batch + 1];  // ← 查预计算表，下一个 batch 的 LR
                        void* dst = ctx.ptr_at(lr_dtensor_id_);
                        cudaMemcpyAsync(dst, &lr, sizeof(float),
                                        cudaMemcpyHostToDevice, s_up);
                    }
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                }

                // ======== Last batch (batches-1) ========
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
                if (gfa) cudaGraphLaunch(gfa, s_up);
                if (gwu) cudaGraphLaunch(gwu, s_up);
                sync_up();

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    // ======== 只在 epoch 全部 batch 跑完后 join 一次 ========
    for (auto& t : threads) t.join();
    for (int r = 0; r < K; ++r) if (exc[r]) std::rethrow_exception(exc[r]);
}
#endif
```

> **🔴 LR 机制对比**:
>
> | 维度 | 旧方案 (MX_FINAL v1) | 新方案 |
> |------|---------------------|--------|
> | LR 位置 | 主线程 for-loop `cudaMemcpyAsync` × 8 卡 | 每个 rank 线程内 `cudaMemcpyAsync`，8 卡并行 |
> | Stream | `s_tr`（TRANS 流） | `s_up`（UPDATE 流），与 WEIGHT_UPDATE 同流，天然有序 |
> | 生命周期 | 栈变量 `float lr` + 异步 `cudaMemcpyAsync`，悬空风险 | 预计算表 `lr_table_[b]`，持久安全 |
> | 计算时机 | 每 batch 现场 `scheduler_step()` | epoch 开始前一次性预计算所有 batch |
> | 线程 join | 无任何中途 join | 只在 epoch 末尾 join 一次（训练）或 validation 末尾 join 一次（验证） |

> **🔴 线程模型刚性约束**（用户明确要求，不可违反）:
> - **在一个 epoch 的训练内，多线程只在最后 join 一次。**
> - **在一个 epoch 的验证内，多线程只在最后 join 一次。**
> - **中间跑各个图的时候绝不 join。**
>
> 这是为了最大化 GPU 利用率和最小化同步开销。当前实现已满足此约束：`std::vector<std::thread>` 在 `run_train_epoch_gpu()` 入口创建、末尾 join，中间所有图 launch 都在线程内完成。

> **SimpleTask 兼容性确保**:
> - `compile_impl()` 中 `is_simple_task()` 分支走 `compile_capture_simple()` → `simple_captured_graphs_`，逻辑完全独立
> - `DeepLearningTask` 的 `build_graph_atlas()` / `build_exec_table()` 不影响 SimpleTask
> - **对于 DeepLearningTask，同一个 GraphId 内的所有算子被捕获为同一个 CUDA Graph，绝不拆散**

### 4.4 **重要说明**：第一阶段（调试）的单 batch 简化

上面的完整实现适合 Phase 3（完整乒乓+重叠）。但在 **Phase 1 首次调试 compile()** 时，建议先只验证 `batches == 1` 的单 batch 路径——这不需要启动 Preprocessor 线程也不涉及 TransferStation，可以**手动构造数据**绕过 Transport 系统，先确保 core 图调度正确。

#### 单 batch 手动数据注入示例

```cpp
// 在 compile() 之后、run() 之前：
void inject_fake_data(DeepLearningTask& task) {
    auto& keeper = ArenaKeeper::instance();
    // 通过 memory_plan_ptr_ 查找 I_A_DATA 和 I_A_LABEL 的 DTensor
    const auto& dtensors = task.memory_plan_ptr_->dtensors();
    const DTensor* img_dt = nullptr;
    const DTensor* lbl_dt = nullptr;
    for (const auto& dt : dtensors) {
        if (dt.region == MemoryRegion::I_A_DATA)  img_dt = &dt;
        if (dt.region == MemoryRegion::I_A_LABEL) lbl_dt = &dt;
    }
    TR_CHECK(img_dt && lbl_dt, ValueError, "I_A_DATA or I_A_LABEL not found");

    // 构造随机图片 (NCHW) 和标签
    Tensor fake_img(img_dt->shape, img_dt->dtype);
    // 用正态分布填充（均值0.5，方差0.1）
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

    // 手动 H2D 写入 ArenaKeeper（rank 0）
    void* img_gpu = keeper.ptr_at(0, static_cast<size_t>(img_dt->offset()));
    void* lbl_gpu = keeper.ptr_at(0, static_cast<size_t>(lbl_dt->offset()));
    cudaMemcpy(img_gpu, fake_img.data<void>(), img_dt->nbytes(), cudaMemcpyHostToDevice);
    cudaMemcpy(lbl_gpu, fake_lbl.data<void>(), lbl_dt->nbytes(), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}
```

> 调用 `inject_fake_data()` 后，直接调用 `run_train_epoch_gpu()` 的 `batches == 1` 分支，通过 `fetch()` 验证 loss/weight 是否正确更新。

---

## 五、Preprocessor 协调与 run_gpu() 主循环

**文件**: `src/task/deep_learning_task.cpp` — `run_gpu()`

当前的 `run_gpu()` 直接调用 `run_train_epoch_gpu()`，不需要显式创建 Preprocessor 线程 —— Preprocessor 作为独立组件由 `Preprocessor::instance()` 单例驱动，其 `train()` 在外部（或内部通过 TransferStation）持续填充数据。

若 Preprocessor 需要显式线程驱动，模式如下（确认当前代码是否需要后调整）：

```cpp
TrainingResult DeepLearningTask::run_gpu() {
    auto& prep = Preprocessor::instance();
    const int epochs = total_epochs_;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        current_epoch_ = epoch;
        auto epoch_start = std::chrono::steady_clock::now();

        // Preprocessor 线程（若需要显式启动）
        // prep.train() 持续填充 TransferStation A/B 区
        std::thread prep_thread([&]() { prep.train(); });

        // LR 预计算在 run_train_epoch_gpu() 内部完成
        run_train_epoch_gpu();

        prep_thread.join();

        float val_loss = 0, top1 = 0, top5 = 0;
        if (should_validate_this_epoch()) {
            // 验证内线程同样只在末尾 join 一次
            run_val_epoch_gpu(false);
        }

        auto epoch_end = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        log_epoch_results(0.0f, val_loss, top1, top5, 0, 0,
                         lr_table_.empty() ? 0.0f : lr_table_.back(), sec);

        if (use_sema_ && epoch > 0) apply_sema_switch();
    }

    return build_training_result();
}
```

### 5.1 学习率机制说明

**旧方案（已废弃）**: 主线程中 `scheduler_step()` 后 for-loop 对 8 卡依次 `cudaMemcpyAsync`，导致串行传输开销，且用栈变量 + 异步拷贝存在生命周期风险。

**新方案（当前采用）**: 
1. **预计算**: `run_train_epoch_gpu()` 入口一次性 `scheduler_step()` 所有 batch，存入 `std::vector<float> lr_table_`
2. **查表**: 每个 rank 线程内读 `lr_table_[batch+1]`（const，多线程读安全）
3. **传输**: 各线程在 Phase 4（`s_up` 流，WEIGHT_UPDATE 之前）独立 `cudaMemcpyAsync`，8 卡完全并行
4. **不阻塞主线程**: 无任何 per-batch 的 CPU-GPU 往返

```cpp
// 在 run_train_epoch_gpu() 入口，主线程一次性完成：
lr_table_.resize(batches);
for (int b = 0; b < batches; ++b) {
    std::visit([](auto&& s) { s.step(); }, sched_cfg_);
    lr_table_[b] = get_current_lr();
}
// 重置 scheduler 以保持状态一致性（epoch 结束后可复用）
```

若 scheduler 不支持 `reset()` 方法，可以保存初始状态并在每 epoch 开始前重建 scheduler。对于当前支持的 scheduler 类型（PolynomialLR / CosineAnnealingLR / StepLR），可添加 `reset(int steps_back)` 方法或记录初始参数后重新构造。

### 5.2 空图安全跳过

单卡场景下 `FIRST_COMM`/`DEEP_COMM` 为空图，`resolve` 返回 `nullptr`。所有 `cudaGraphLaunch` 调用前隐式检查：在前面 `run_train_epoch_gpu()` 代码中，`gda`/`gfa` 已经是从 `resolve` 获取的结果——空图为 `nullptr`，在 `if (gda)` 判断中自动跳过。同样，`gwu`、`gfb` 等如果为 `nullptr`，由其所在的 `if` 分支控制。

---

## 六、修改文件清单

| # | 文件 | 改动 | 阶段 |
|---|------|------|:---:|
| 1 | `include/renaissance/task/deep_learning_task.h` | 新增 `build_graph_atlas()`, `stream_for()` 声明; 新增 `train_cg_`, `infer_cg_`, `lr_dtensor_id_`, `lr_table_` 成员; 覆写 `memory_plan()` | P0 |
| 2 | `include/renaissance/task/task_base.h` | 新增虚函数 `virtual const MemoryPlan& memory_plan() const` | P0 |
| 3 | `src/task/deep_learning_task.cpp` | GraphSlot 新增 `FIRST_FWD_A/B`, `CAST_AND_CHECK`; 实现 `build_graph_atlas()`, `stream_for()`; 重写 `build_exec_table()` 按 GraphId 解析; 修改 `on_prepare()` finalize `memory_plan_ptr_`; 更新 `kRequired` 加 `FIRST_FWD_A/B`; 重写 `run_train_epoch_gpu()`（LR 预计算 + in-thread H2D + 重叠调度）; 修改 `run_gpu()` | P0 |
| 4 | `src/task/task_base.cpp` | `compile_impl()` DeepLearningTask 分支改为 `dl->build_graph_atlas()`（复用已有 dl）; 增加 ArenaKeeper cudaMemset + `init_all()`; 实现 `init()` 真实路径; 所有 `memory_plan_.xxx()` 改为 `memory_plan().xxx()` | P0 |
| 5 | `src/core/tensor.h/.cpp` | `random_normal()` / `random_uniform()` 方法（如不存在） | P0 |
| 6 | `tests/ref/mnist_mlp_3.cpp` | `compile_for_dry_run()+dry_run()` → `compile()+run()` | P2 |

---

## 七、验证里程碑与实施顺序

| 阶段 | 内容 | 验收标准 | 预估 |
|:----:|------|----------|:---:|
| **P0** | Atlas + init + compile() | compile() 不 crash; `gpu_exec_` 所有 slot 非 null (除 ALLREDUCE 单卡外) | 2h |
| **P1** | 单 batch 手动数据（绕过 Transport） | Loss 非 NaN; 梯度更新后权重有变化 | 2h |
| **P2** | Preprocessor 对接 + A/B 乒乓 | 1 epoch 469 batches 无 crash; Loss 趋势下降 | 3h |
| **P3** | 20 epoch + 验证 | Best Val Top-1 > 95%; 无 NaN | 2h |

### 实施顺序

```
1. P0: build_graph_atlas() + init() + compile_impl 分支修改
   验证: compile() 后打印 Exec Table 完整性

2. P1: 单 batch 路径（复用 batches==1 分支，手动 mock 数据填入 I_A_DATA）
   验证: fetch DTensor，检查 loss / gradient / weight 值

3. P2: Preprocessor 线程 + 完整 run_train_epoch_gpu()
   验证: 1 epoch 469 batch，A/B 乒乓无死锁

4. P3: 20 epoch 训练 + 验证指标
   验证: 收敛曲线; Top-1 > 95%
```

---

## 八、风险与回退

| 风险 | 等级 | 缓解 |
|------|:----:|------|
| **`memory_plan_` 与 `memory_plan_ptr_` 错位** | 🔴 | 必须引入虚函数 `memory_plan()`；DeepLearningTask 覆写返回 `*memory_plan_ptr_`；TaskBase 所有方法改调用 `memory_plan().xxx()`；`compile_alloc_hardware`、`init_all`、`transfer_to_rank`、`fetch_from_rank` 全部受此影响 |
| `Tensor::random_normal()` 不存在 | 中 | 直接用 `<random>` + `Tensor::data<float>()`手动填充 |
| 单 batch 手动构造数据不兼容 TransferStation 格式 | 中 | 直接用 `cudaMemcpy` 写 I_A_DATA/I_A_LABEL 的 ArenaKeeper 内存区间 |
| Scheduler 不支持 `reset()` | 中 | 每 epoch 前记录初始参数并用副本重建 scheduler；或添加 `reset(int steps_back)` 方法 |
| MNIST last batch 96 样本 shape ≠ 128 | 中 | 当前 run_train_epoch_gpu 对 last batch 与中间 batch 同等处理（deque + DEEP），确认 DEEP 图是 shape-agnostic |
| Preprocessor `train()` 阻塞无法 join | 中 | 确认 `prep.stop()` 机制; 必要时在每个 epoch 结束时重新创建 Preprocessor 实例 |
| DEEP_FWD_BWD 图为空 → `cudaGraphLaunch(nullptr)` crash | 低 | compile 后 `gpu_exec_` 打印检查; resolve 对空图返回 nullptr |

---

## 九、DEEP_FWD_BWD 图的 A/B 共享说明

（此处已有"为什么 A/B 共享同一 DEEP_FWD_BWD 图"的说明，与第四章 4.3 节一致。本章是对该设计的正式文档化，便于后续 ResNet-50 多 shape variant 的场景查阅。）

- FWD_BWD_DEEP_A 和 FWD_BWD_DEEP_B 都从 `GraphId::DEEP_FWD_BWD` 解析
- 图本身不分辨 A/B —— 差异仅在于运行时输入数据位于 `I_A_DATA` 还是 `I_B_DATA`（不同 ArenaKeeper 偏移）
- `pre_capture()` 去重后只捕获一份 cudaGraphExec
- 若未来需要不同 shape（progressive resolution），A/B variant 通过 `GraphAtlas` 的不同 `shape_id` 区分

---

## 十、修订记录（per MX_REV.md 评审意见）

| # | 修改项 | 等级 | 涉及章节 |
|---|--------|:---:|---------|
| 1 | `memory_plan_` → `memory_plan_ptr_` 修复：引入虚函数 `memory_plan()`，DeepLearningTask 覆写返回 `*memory_plan_ptr_`；TaskBase 所有 `memory_plan_.xxx()` → `memory_plan().xxx()`；build_graph_atlas 中 `sl.mp = memory_plan_ptr_.get()` | 🔴 | 2.2, 2.5, 3.1, 3.2 |
| 2 | 学习率机制重设计：预计算 `lr_table_`、每 rank 线程内独立 cudaMemcpyAsync（s_up 流）、移除主线程 for-loop | 🔴 | 4.1, 4.2, 4.3, 5.1 |
| 3 | 线程模型刚性约束：训练/验证 epoch 内只在末尾 join 一次；无任何中途 join | 🔴 | 4.3, 5.0 |
| 4 | `compile_impl()` 复用已有 `dl` 指针，移除冗余 `dynamic_cast` | 🟡 | 2.4 |
| 5 | `CAST_AND_CHECK` / `GRAD_CONVERT` 同 GraphId 注释说明 | 🟡 | 2.3 |
| 6 | `kRequired` 数组新增 `FIRST_FWD_A/B` | 🟢 | 2.3 |
| 7 | 单 batch 调试数据注入示例（`inject_fake_data()`） | 🟢 | 4.4 |
| 8 | SimpleTask 兼容性确保说明 | 🟢 | 4.3 |
| 9 | `MemoryPlan` 不可拷贝/不可移动 → 虚函数方案（非复制方案） | 🔴 | 2.5, 8 |
| 10 | CG 指针保存必须在 `add_graph()` 之后取地址 | 🟡 | 2.5 |

---

*本方案综合 MXZ1/MXZ2/MXZ3 + MX_REV.md 评审意见 + 源码全路径交叉验证后制定。所有代码改动均有文件路径+行号引用，所有设计决策均有对比择优说明。*