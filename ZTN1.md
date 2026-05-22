# ZTN1：MLP 从 Dry Run 到真训练 —— 完整科学方案

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22  
> **前置输入**: AXX.md（MMP0 原始需求 + 小伙伴 S/K/D 三方案 + 用户补充）  
> **来源验证**: 所有诊断结论均经源码逐行交叉验证（文件路径 + 行号可追溯）

---

## 〇、源码审计：重大错位清单（所有方案均未完整识别）

对 `compile() → run()` 全链路进行源码逐行审计，发现 **7 个已在源码中存在的错位**，它们共同导致真训练完全不可能运行。所有方案（S/K/D）均只覆盖其中 4~5 个。

| # | 错位 | 源码位置 | 后果 | 严重度 |
|---|------|----------|------|:------:|
| 1 | `memory_plan_` 永远是空壳 | [task_base.h:275](file:///r:/renaissance/include/renaissance/task/task_base.h#L275) 值成员 vs [deep_learning_task.h:311](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h#L311) unique_ptr；[on_prepare()](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h#L269) 调用 `memory_plan_.finalize()` 而非 `memory_plan_ptr_->finalize()` | ArenaKeeper 分配 0 字节、权重不初始化、transfer/fetch 全部失败 | 🔴 |
| 2 | Atlas 全部填入 `SIMPLE_TASK_GRAPH` + 错误 `&memory_plan_` | [compile_impl():237](file:///r:/renaissance/src/task/task_base.cpp#L237) 调用 `build_simple_atlas()`；[build_simple_atlas():633-636](file:///r:/renaissance/src/task/task_base.cpp#L633-L636) 全部 slot 写 `GraphId::SIMPLE_TASK_GRAPH` + `&memory_plan_` | 所有 CUDA Graph 按错误的 GraphId 和错误的内存布局捕获 → Exec Table 全部为 nullptr 或错误 | 🔴 |
| 3 | `build_graph_index()` 基于错位 #2 的 `name_to_gid_` 查找 | [build_graph_index():467-468](file:///r:/renaissance/src/task/deep_learning_task.cpp#L467-L468) `name_to_gid_[name]` 全为 `SIMPLE_TASK_GRAPH` → `atlas.index(0, SIMPLE_TASK_GRAPH)` 只找到最后一个 slot | Exec Table 所有 slot 指向同一张图或 nullptr | 🔴 |
| 4 | `init()` 真实路径是 `TR_NOT_IMPLEMENTED` | [task_base.cpp:1223](file:///r:/renaissance/src/task/task_base.cpp#L1223) | 所有权重停留在未初始化的随机显存值 → 训练立即 NaN | 🔴 |
| 5 | `run_train_epoch_gpu()` 是严重残缺的骨架 | [deep_learning_task.cpp:691-829](file:///r:/renaissance/src/task/deep_learning_task.cpp#L691-L829) 只调度 XFER + DEEP_FWD_BWD + FIRST_LAYER_BWD 三种图，无 ZERO_GRAD/ALLREDUCE/WEIGHT_UPDATE/GRAD_CONVERT/FIRST_FWD，无流间重叠，`scheduler.step()` 仅 rank==0 | 训练既无梯度清零也无权重更新，多线程竞争 sched_cfg_ | 🔴 |
| 6 | `run_gpu()` 无 Preprocessor 线程 | [deep_learning_task.cpp:622](file:///r:/renaissance/src/task/deep_learning_task.cpp#L622) 直接调用 `run_train_epoch_gpu()`，Preprocessor 未被启动 | TransferStation 永远为空，rank 线程永远忙等待 | 🔴 |
| 7 | 编译流水线无数据初始化 | [compile_impl():230](file:///r:/renaissance/src/task/task_base.cpp#L230) 调用 `compile_alloc_hardware()` 后直接进入 `compile_mark_compiled()`，中间无 ArenaKeeper memset 也无 `init_all()` 调用 | 显存池包含脏数据，权重为随机值 | 🔴 |

**关键发现**：这 7 个错位形成依赖链 —— #1 导致 #2/#3 失效，#2/#3 导致 #5 无法拿到正确的图，#4/#7 导致训练无意义。修复必须一次性全部覆盖，不可分步遗留。

---

## 一、科学方案：五项核心修复 + 一个集成主循环

### 1.1 整体架构

```
compile() 流水线:
  compile_freeze_global()
  compile_invoke_on_prepare()   → on_prepare():
                                    Compiler::compile()
                                    memory_plan_ptr_ = move(result)  ← 数据
                                    memory_plan_ptr_->finalize()     ← 修正 #1
                                    add_graph("train"/"inference")
                                    active_memory_plan_ = memory_plan_ptr_.get()  ← 修正 #1
  compile_verify_memory_locked()
  compile_alloc_hardware()      → active_memory_plan_->total_bytes() ← 修正 #1
  ArenaKeeper memset            ← 新增（修正 #7）
  init_all()                    ← 新增（修正 #4, #7）
  build_graph_atlas()           ← 取代 build_simple_atlas()（修正 #2, #3）
  pre_capture(atlas, ctx_ptrs)
  build_exec_table()            ← 按 GraphId 直接解析（修正 #3）
  compile_mark_compiled()

run() 流水线:
  for epoch:
    prep_thread = spawn(Preprocessor::train)    ← 修正 #6
    run_train_epoch_gpu()                       ← 完全重写（修正 #5）
    prep_thread.join()
    run_val_epoch_gpu() (if should_validate)
```

### 1.2 五项核心修复

| 修复 | 技术方案 | 受影响的文件 | 受影响的函数（全部有行号对应） |
|:----:|----------|-------------|-------------------------------|
| **F1: MemoryPlan 双轨统一** | `active_memory_plan_` 指针（K 方案，最小侵入） | task_base.h, task_base.cpp, deep_learning_task.cpp | `compile_alloc_hardware`, `init`, `init_all`, `transfer_to_rank`, `fetch_from_rank`, `fill`（6 处替换） |
| **F2: Atlas 架构修正** | `DeepLearningTask::build_graph_atlas()` 独立函数（D/K 方案） | deep_learning_task.h, deep_learning_task.cpp, task_base.cpp | GraphSlot 枚举扩展、atlas 构建、`compile_impl()` 分支 |
| **F3: Exec Table 重建** | 按 `GraphId` 直接 `atlas.index(0, gid)` 解析（D/K 方案） | deep_learning_task.cpp | `build_exec_table()` 完全重写 |
| **F4: 数据初始化** | `init()` 真实路径 + ArenaKeeper memset + `init_all()` | task_base.cpp, compile 流水线 | `init`, `compile_impl` |
| **F5: 4 阶段重叠调度** | MMP0 原始伪代码对齐（FIRST_FWD‖ZERO_GRAD 并行） | deep_learning_task.cpp, deep_learning_task.h | `run_train_epoch_gpu` 完全重写、`run_gpu` N+1 线程、`clone_scheduler`、`lr_host_buffer_`、`lr_dtensor_id_` |

---

## 二、修复 F1：`active_memory_plan_` 指针（最小侵入，6 方法替换）

### 2.1 原理

`MemoryPlan` 已删除拷贝/移动（[memory_plan.h:75-78](file:///r:/renaissance/include/renaissance/graph/memory_plan.h#L75-L78)），`memory_plan_ptr_` 的内容无法回迁到 `memory_plan_`。在 `TaskBase` 增加一个 `MemoryPlan*` 指针，默认指向 `&memory_plan_`，`DeepLearningTask::on_prepare()` 末尾将其切换到 `memory_plan_ptr_.get()`。

**对比虚函数方案（S）**：虚函数 `working_memory_plan()` 需修改所有调用点且引入运行时开销；指针方案只改 6 个方法的 6 行代码。

### 2.2 代码修改

**文件 1**: `include/renaissance/task/task_base.h`

在第 275 行 `MemoryPlan memory_plan_{plan_config_};` 之后新增：

```cpp
MemoryPlan memory_plan_{plan_config_};
MemoryPlan* active_memory_plan_ = &memory_plan_;  // ← 新增：默认指向值成员
```

**文件 2**: `src/task/task_base.cpp`

以下 6 个方法中各有一处 `memory_plan_.` 需替换为 `active_memory_plan_->`：

| 方法 | 行号 | 替换前 | 替换后 |
|------|------|--------|--------|
| `compile_alloc_hardware()` | 1409 | `memory_plan_.total_bytes()` | `active_memory_plan_->total_bytes()` |
| `init()` | 1223+ | `memory_plan_.get_dtensor(...)` | `active_memory_plan_->get_dtensor(...)` |
| `init_all()` | 1230,1235,1243 | `memory_plan_.dtensors()` / `.get_dtensor()` | `active_memory_plan_->dtensors()` / `->get_dtensor()` |
| `transfer_to_rank()` | 887 | `memory_plan_.get_dtensor(dt.id)` | `active_memory_plan_->get_dtensor(dt.id)` |
| `fetch_from_rank()` | 1271 | `memory_plan_.get_dtensor(dt.id)` | `active_memory_plan_->get_dtensor(dt.id)` |
| `fill()` | 1092 | `memory_plan_.get_dtensor(dt.id)` | `active_memory_plan_->get_dtensor(dt.id)` |

> `debug_dump()`、`validate()` 等纯诊断路径保留 `memory_plan_` 不影响正确性，仅涉及显存分配、数据传输、初始化的路径必须替换。

**文件 3**: `src/task/deep_learning_task.cpp`（`on_prepare()` 末尾，当前在 [deep_learning_task.h:268-271](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h#L268-L271)）

```cpp
// ---- 当前代码 ----
memory_plan_ptr_ = std::move(result.variants[0].memory_plan);
finalize_memory();  // ← BUG: 调用 memory_plan_.finalize() —— 空实例！

// ---- 修改为 ----
memory_plan_ptr_ = std::move(result.variants[0].memory_plan);
memory_plan_ptr_->finalize();  // ← 修正：finalize 真实布局
phase_ = Phase::MEMORY_LOCKED;

add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

train_cg_ = &named_graphs_["train"].graph;
infer_cg_ = &named_graphs_["inference"].graph;

// 找出 SCALAR_FP32 学习率 DTensor
for (const auto& dt : memory_plan_ptr_->dtensors()) {
    if (dt.usage == DTensorUsage::SCALAR_FP32) {
        lr_dtensor_id_ = dt.id;
        break;
    }
}

active_memory_plan_ = memory_plan_ptr_.get();  // ← 关键一行：激活真实布局
```

> **对 SimpleTask 的影响**：零。SimpleTask 不持有 `memory_plan_ptr_`，`active_memory_plan_` 保持默认值 `&memory_plan_`，行为不变。

---

## 三、修复 F2：Atlas 架构修正（独立 `build_graph_atlas()` + GraphSlot 扩展）

### 3.1 GraphSlot 枚举扩展

**文件**: `src/task/deep_learning_task.cpp:32-49`（替换现有枚举）

```cpp
enum class GraphSlot : uint8_t {
    FIRST_FWD_A = 0,       // ← 新增：首层前向 A 区
    FIRST_FWD_B,           // ← 新增：首层前向 B 区
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
    CAST_AND_CHECK,         // ← 新增：AMP 梯度转换 + NaN 检查
    INF_MAIN_A,
    INF_MAIN_B,
    INF_EMA_A,
    INF_EMA_B,
    COUNT
};
```

### 3.2 `build_graph_atlas()` 实现

**文件**: `include/renaissance/task/deep_learning_task.h` — 新增声明（在 protected/private 区域）

```cpp
class DeepLearningTask : public TaskBase {
    // ...
    GraphAtlas build_graph_atlas() const;
    static StreamKind stream_for(GraphId gid);

    const ComputationGraph* train_cg_ = nullptr;
    const ComputationGraph* infer_cg_ = nullptr;
    DTensorID lr_dtensor_id_ = -1;
    float lr_host_buffer_ = 0.0f;
};
```

**文件**: `src/task/deep_learning_task.cpp` — 新增实现

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() const {
    GraphAtlas atlas;

    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (train_cg_->nodes(gid).empty()) continue;

            auto& sl = atlas.slot(0, gi);
            sl.cg          = train_cg_;
            sl.mp          = active_memory_plan_;  // ← 已指向真实布局（F1 修正后）
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

> **关键约束**：`FWD_BWD_DEEP_A` 和 `FWD_BWD_DEEP_B` 都解析自 `GraphId::DEEP_FWD_BWD` —— 这符合语义：深层图不区分 A/B，区别仅在输入来自 I_A_DATA 还是 I_B_DATA（不同 ArenaKeeper 偏移）。`pre_capture()` 会去重。

### 3.3 `compile_impl()` DeepLearningTask 分支

**文件**: `src/task/task_base.cpp:232-250`（修改 else 分支）

```cpp
// ---- 当前代码 ----
} else {
    GraphAtlas atlas = build_simple_atlas(name_to_gid_);
    // ...
}

// ---- 修改为 ----
} else {
    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        GraphAtlas atlas = dl->build_graph_atlas();  // ← DeepLearningTask 专属

        std::vector<DeviceContext*> ctx_ptrs;
        for (auto& ctx : backend_->contexts)
            ctx_ptrs.push_back(ctx.get());

        captured_result_ = pre_capture(atlas, ctx_ptrs);

        dl->build_graph_index();
        dl->build_exec_table();
    }
}
```

---

## 四、修复 F3：Exec Table 按 GraphId 直接解析

### 4.1 废弃名字中间层

当前 `build_exec_table()` 依赖 `build_graph_index()` 生成的 `name_to_graph_index_`，而 `build_graph_index()` 又依赖错位 #2 中的 `name_to_gid_` → **三重间接且基于是错误的 SIMPLE_TASK_GRAPH**。

新方案：直接 `captured_result_.atlas.index(0, gid)` 取 idx → `captured_result_.graphs[idx].native_exec(rank)`。

### 4.2 完整实现

**文件**: `src/task/deep_learning_task.cpp:475-531`（完全重写 `build_exec_table()`）

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

        #define S(slot) static_cast<size_t>(GraphSlot::slot)
        g[S(FIRST_FWD_A)]           = resolve(GraphId::FIRST_FWD_A,   rank);
        g[S(FIRST_FWD_B)]           = resolve(GraphId::FIRST_FWD_B,   rank);
        g[S(XFER_A)]                = resolve(GraphId::TRANSFER_A,    rank);
        g[S(XFER_B)]                = resolve(GraphId::TRANSFER_B,    rank);
        g[S(FWD_BWD_DEEP_A)]        = resolve(GraphId::DEEP_FWD_BWD,  rank);
        g[S(FWD_BWD_DEEP_B)]        = resolve(GraphId::DEEP_FWD_BWD,  rank);
        g[S(FIRST_LAYER_BWD)]       = resolve(GraphId::FIRST_BWD,     rank);
        g[S(ZERO_GRAD)]             = resolve(GraphId::ZERO_GRAD,     rank);
        g[S(DEEP_ALLREDUCE)]        = resolve(GraphId::DEEP_COMM,     rank);
        g[S(FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM,    rank);
        g[S(WEIGHT_UPDATE)]         = resolve(GraphId::OPTIMIZER,     rank);
        g[S(EMA_UPDATE)]            = resolve(GraphId::EMA_UPDATE,    rank);
        // GRAD_CONVERT 和 CAST_AND_CHECK 均映射到 GraphId::CAST_AND_CHECK
        // （Compiler 合并 AMP FP16→FP32 梯度转换 + NaN 检查为单图）
        g[S(GRAD_CONVERT)]          = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(CAST_AND_CHECK)]        = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(INF_MAIN_A)]            = resolve(GraphId::INF_MAIN_A,    rank);
        g[S(INF_MAIN_B)]            = resolve(GraphId::INF_MAIN_B,    rank);
        g[S(INF_EMA_A)]             = resolve(GraphId::INF_EMA_A,     rank);
        g[S(INF_EMA_B)]             = resolve(GraphId::INF_EMA_B,     rank);
        #undef S
    }

    // compile 期校验：必需的 slot 不可为 nullptr
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
            TR_CHECK(gpu_exec_.graphs[rank][static_cast<size_t>(slot)],
                     ValueError,
                     "Required slot " << static_cast<int>(slot)
                     << " is nullptr for rank " << rank);
        }
    }
#endif
}
```

> `build_graph_index()` 保留不删（不影响正确性），但 `build_exec_table()` 不再依赖它。

---

## 五、修复 F4：数据初始化（`init()` 真实路径 + ArenaKeeper memset）

### 5.1 `init()` 真实路径

**文件**: `src/task/task_base.cpp:1200-1223`（替换 `TR_NOT_IMPLEMENTED` 分支）

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
            float* data = host.data<float>();
            float std = config.scale;
            if (std <= 0.0f) {
                // Xavier: std = sqrt(2.0 / fan_in)
                int64_t fan_in = live_dt.shape.c();
                std = std::sqrt(2.0f / static_cast<float>(fan_in));
            }
            std::default_random_engine rng(std::random_device{}());
            std::normal_distribution<float> dist(0.0f, std);
            for (int64_t i = 0; i < host.numel(); ++i)
                data[i] = dist(rng);
            break;
        }
        default:
            TR_CHECK(false, NotImplemented,
                     "InitKind " << static_cast<int>(config.kind) << " not implemented");
    }
    transfer_to_rank(host, live_dt, 0);
}
```

### 5.2 ArenaKeeper memset + init_all 集成到 compile 流水线

**文件**: `src/task/task_base.cpp:230-255`（`compile_impl()` 中 `compile_alloc_hardware()` 之后）

```cpp
compile_alloc_hardware();

// ---- 新增：ArenaKeeper 显存池清零 ----
#ifdef TR_USE_CUDA
if (GlobalRegistry::instance().using_gpu()) {
    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaSetDevice(backend_->contexts[rank]->device_id());
        void* ptr = ArenaKeeper::instance().ptr_at(rank, 0);
        cudaMemset(ptr, 0, active_memory_plan_->total_bytes());
    }
    cudaDeviceSynchronize();
}
#endif

// ---- 新增：所有权重初始化 ----
init_all();

// ---- 后续：Atlas + capture 不变 ----
if (is_simple_task()) {
    // ...
```

> `init_all()` 遍历 `active_memory_plan_->dtensors()`（已指向真实布局），对每个 DTensor 按其 `init_config` 执行随机初始化。

---

## 六、修复 F5：4 阶段重叠调度（完全重写 `run_train_epoch_gpu()`）

### 6.1 科学原理：为什么必须恢复 MMP0 的 4 阶段并行

MMP0 伪代码（[AXX.md:73-75](file:///r:/renaissance/AXX.md#L73-L75)）中，`FIRST_FWD` 和 `ZERO_GRAD` 被**同时** launch 到不同 stream：

```cpp
cudaGraphLaunch(*cuda_graph_ptr_first_overlap, stream_comp_1_ptr);  // 计算流
cudaGraphLaunch(*cuda_graph_ptr_zero_grad, stream_update_ptr);       // 更新流
```

物理依据：
- `FIRST_FWD` 读取 `I_{A|B}_DATA`（输入数据区），写入首层激活（WR_0 区）
- `ZERO_GRAD` 写入梯度缓冲区（G_* 区）
- 两者在不同显存区域、不同 CUDA stream → **天然可并行**，串行化是性能浪费

S/K/D 三方案均把 ZERO_GRAD 析出为独立阶段，增加了不必要的同步屏障，偏离 MMP0 原意。本方案恢复 MMP0 设计。

### 6.2 学习率机制：每个 Rank 线程独立 Scheduler + 隐藏计算

**科学依据**（用户补充意见）：

- 学习率计算是轻量 CPU 操作（若干浮点运算）
- 每个 rank 线程在 epoch 开始时获得 scheduler 的**独立深拷贝**
- 在 GPU 执行 Phase 3（FIRST_BWD / ALLREDUCE）期间，CPU 同步执行 `scheduler_copy.step()`
- `cudaMemcpyAsync(LR → GPU, H2D, s_up)` 在 UPDATE 流上入队
- `WEIGHT_UPDATE` 图也在 UPDATE 流上 → CUDA stream 串行语义自动保证 LR 在权重更新前到位

**对比"预计算表"（S/D 方案）**：
- 预计算表需要 `scheduler.reset(steps_back)` API（**不存在**）
- epoch 结束后需回退 scheduler 状态（易出错）
- 索引偏移 bug（batch 0 用 `lr_table_[1]` 等）在多方案中反复出现

**对比"仅 rank 0 算 LR"（K 方案）**：
- 多卡时其他 rank 没有 LR → 权重更新失败
- 仅 rank 0 调用 `scheduler.step()` 修改 `sched_cfg_`，其他 rank 线程同时读 → 数据竞争

**本方案**：`std::visit` 深拷贝 `sched_cfg_` → 各线程独立 → 零竞争、零索引偏移、无需额外 API。

### 6.3 Stream 分配

| StreamKind | 承载图 |
|------------|--------|
| `TRANS` | XFER_A/B（异步传输，与计算无依赖） |
| `COMP_1` | FIRST_FWD_A/B, DEEP_FWD_BWD, FIRST_BWD（主计算） |
| `COMP_2/3` | 空（MLP 无多计算流需求，仅参与三流同步） |
| `UPDATE` | ZERO_GRAD, ALLREDUCE, GRAD_CONVERT, CAST_AND_CHECK, WEIGHT_UPDATE, EMA_UPDATE, LR H2D |

### 6.4 4 阶段调度模型

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1:  ZERO_GRAD [UPDATE]  ‖  FIRST_FWD [COMP_1]           │
│           sync_comp() + sync_up()                                │
│                                                                  │
│           [CPU: wait TransferStation next_buf readable]          │
│                                                                  │
│ Phase 2:  DEEP_FWD_BWD [COMP_1]  ‖  XFER(next) [TRANS]        │  ← 核心重叠
│           sync_comp() + sync_trans()                             │
│           [rank 0: set_buffer_writeable(next, true)]             │
│                                                                  │
│ Phase 3:  FIRST_BWD [COMP_1]  ‖  DEEP_ALLREDUCE [UPDATE]       │
│           sync_comp() + sync_up()                                │
│           [AMP: GRAD_CONVERT + CAST_AND_CHECK (UPDATE)]         │
│                                                                  │
│ Phase 4:  [CPU: scheduler_copy.step() → compute LR]             │
│           FIRST_ALLREDUCE [UPDATE] (if exists)                   │
│           cudaMemcpyAsync(LR, H2D, UPDATE)                       │
│           WEIGHT_UPDATE [UPDATE]                                 │
│           sync_up()                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 6.5 `clone_scheduler()` 实现

```cpp
// deep_learning_task.cpp 新增
LRSchedulerVariant DeepLearningTask::clone_scheduler() const {
    return std::visit([](auto&& s) -> LRSchedulerVariant {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, std::monostate>)
            return std::monostate{};
        else
            return T(s);  // 拷贝构造
    }, sched_cfg_);
}
```

类型别名（在 `deep_learning_task.h` 中新增）：

```cpp
using LRSchedulerVariant = std::variant<std::monostate, PolynomialLR, CosineAnnealingLR, StepLR>;
```

### 6.6 `run_train_epoch_gpu()` 完整实现

**文件**: `src/task/deep_learning_task.cpp:691-829`（完全重写）

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

                // === 本线程独立的 scheduler 副本 ===
                auto scheduler_copy = clone_scheduler();
                int lr_step = 0;  // 若 step_by_batch，累积步数

                const auto& g = gpu_exec_.graphs[rank];
                #define G(slot) g[static_cast<size_t>(GraphSlot::slot)]
                auto gf_a  = G(FIRST_FWD_A),    gf_b  = G(FIRST_FWD_B);
                auto gx_a  = G(XFER_A),         gx_b  = G(XFER_B);
                auto gd_a  = G(FWD_BWD_DEEP_A), gd_b  = G(FWD_BWD_DEEP_B);
                auto gfb   = G(FIRST_LAYER_BWD), gzg  = G(ZERO_GRAD);
                auto gda   = G(DEEP_ALLREDUCE),  gfa  = G(FIRST_LAYER_ALLREDUCE);
                auto gwu   = G(WEIGHT_UPDATE),   ggc  = G(GRAD_CONVERT);
                auto gcn   = G(CAST_AND_CHECK);
                #undef G

                auto& ctx = context(rank);
                auto s_tr = (cudaStream_t)ctx.stream(StreamKind::TRANS);
                auto s_c1 = (cudaStream_t)ctx.stream(StreamKind::COMP_1);
                auto s_c2 = (cudaStream_t)ctx.stream(StreamKind::COMP_2);
                auto s_c3 = (cudaStream_t)ctx.stream(StreamKind::COMP_3);
                auto s_up = (cudaStream_t)ctx.stream(StreamKind::UPDATE);

                auto sync_comp  = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_trans = [&]() { cudaStreamSynchronize(s_tr); };
                auto sync_up    = [&]() { cudaStreamSynchronize(s_up); };

                void* lr_dev_ptr = ctx.ptr_at(lr_dtensor_id_);

                auto transfer_lr = [&](float lr) {
                    lr_host_buffer_ = lr;
                    cudaMemcpyAsync(lr_dev_ptr, &lr_host_buffer_, sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                };

                auto advance_scheduler = [&](int batch_id) {
                    std::visit([&](auto&& sch) {
                        using T = std::decay_t<decltype(sch)>;
                        if constexpr (!std::is_same_v<T, std::monostate>) {
                            sch.step();
                        }
                    }, scheduler_copy);
                };

                auto get_lr = [&]() -> float {
                    return std::visit([](auto&& sch) -> float {
                        using T = std::decay_t<decltype(sch)>;
                        if constexpr (std::is_same_v<T, std::monostate>)
                            return 0.0f;
                        else
                            return sch.get_current_lr();
                    }, scheduler_copy);
                };

                // ======== Batch 0: Pre-transfer from A ========
                while (!ts->buffer_is_readable(0))
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                cudaGraphLaunch(gx_a, s_tr);
                sync_trans();
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                // ======== 单 batch（调试/测试） ========
                if (batches == 1) {
                    cudaGraphLaunch(gzg,  s_up);
                    cudaGraphLaunch(gf_a, s_c1);
                    sync_comp(); sync_up();

                    cudaGraphLaunch(gd_a, s_c1);
                    sync_comp();

                    if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }

                    if (gda) cudaGraphLaunch(gda, s_up);
                    sync_up();

                    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }
                    if (using_amp && gcn) { cudaGraphLaunch(gcn, s_up); sync_up(); }

                    // Phase 4
                    advance_scheduler(0);
                    float lr = get_lr();
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    transfer_lr(lr);
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                    return;
                }

                // ======== Batch 0 ~ batches-2: 核心乒乓 + 4 阶段重叠 ========
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    auto gf   = from_a ? gf_a  : gf_b;
                    auto gd   = from_a ? gd_a  : gd_b;
                    auto gx_n = from_a ? gx_b  : gx_a;

                    // ---- Phase 1: ZERO_GRAD ‖ FIRST_FWD ----
                    cudaGraphLaunch(gzg, s_up);
                    cudaGraphLaunch(gf,  s_c1);
                    sync_comp(); sync_up();

                    // Wait next buffer
                    while (!ts->buffer_is_readable(next_buf))
                        std::this_thread::sleep_for(std::chrono::microseconds(50));

                    // ---- Phase 2: DEEP_FWD_BWD ‖ XFER(next) —— 核心重叠 ----
                    cudaGraphLaunch(gd,   s_c1);
                    cudaGraphLaunch(gx_n, s_tr);
                    sync_comp(); sync_trans();
                    if (rank == 0) {
                        ts->set_buffer_readable(next_buf, false);
                        ts->set_buffer_writeable(next_buf, true);
                    }

                    // ---- Phase 3: FIRST_BWD ‖ DEEP_ALLREDUCE ----
                    if (!frozen) cudaGraphLaunch(gfb, s_c1);
                    if (gda)     cudaGraphLaunch(gda, s_up);
                    sync_comp(); sync_up();

                    // AMP
                    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }
                    if (using_amp && gcn) { cudaGraphLaunch(gcn, s_up); sync_up(); }

                    // ---- Phase 4: scheduler step + LR H2D + FIRST_ALLREDUCE + WEIGHT_UPDATE ----
                    // CPU 计算隐藏在 GPU 执行间隙
                    advance_scheduler(batch);
                    float lr = get_lr();

                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    transfer_lr(lr);    // Async H2D on UPDATE stream
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                }

                // ======== Last batch (batches-1): no XFER ========
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
                if (using_amp && gcn) { cudaGraphLaunch(gcn, s_up); sync_up(); }

                // Phase 4
                advance_scheduler(batches - 1);
                float lr = get_lr();
                if (gfa) cudaGraphLaunch(gfa, s_up);
                transfer_lr(lr);
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

### 6.7 Preprocessor 线程协调（修正 #6）

**文件**: `src/task/deep_learning_task.cpp:537-622`（修改 `run_gpu()` epoch 循环）

将当前单线程调用：

```cpp
// 当前: run_train_epoch_gpu();  // 内部创建 rank 线程
```

改为 N+1 模型：

```cpp
for (int epoch = 0; epoch < total_epochs_; ++epoch) {
    // ...
    // ===== N+1 线程：1 Preprocessor + K Rank =====
    std::thread prep_thread([&]() { prep.train(); });

    run_train_epoch_gpu();   // 内部展开 K 个 rank 线程，末尾 join

    prep_thread.join();
    // ...
}
```

### 6.8 TransferStation 标志管理

| 操作 | 执行者 | 原子性 |
|------|--------|--------|
| `buffer_is_readable(buf)` 检查 | **所有 rank 线程** | `std::atomic<bool>`，多线程读安全 |
| `set_buffer_readable(buf, false)` | **仅 rank 0** | 避免多 rank 写竞争 |
| `set_buffer_writeable(buf, true)` | **仅 rank 0** | 通知 Preprocessor 可填充 |

Preprocessor `train()` 返回条件：所有 TransferStation buffer 均为 `writeable == true`。由于 last batch 数据由 `batch = batches-2` 时传输完毕并标记，epoch 结束时两个 buffer 必为 writeable → Preprocessor 正常返回 → `prep_thread.join()` 不阻塞。

---

## 七、单 Batch 调试路径

在 compile() 后、run() 前手动注入假数据，绕过 TransferStation 和 Preprocessor，独立验证 compile 产物的正确性。

```cpp
void inject_fake_data(DeepLearningTask& task) {
    const auto& mp = task.memory_plan();  // active_memory_plan_ 已激活

    const DTensor* img_dt = nullptr;
    const DTensor* lbl_dt = nullptr;
    for (const auto& dt : mp.dtensors()) {
        if (dt.region == Region::I_A_DATA)  img_dt = &dt;
        if (dt.region == Region::I_A_LABEL) lbl_dt = &dt;
    }
    TR_CHECK(img_dt && lbl_dt, ValueError, "I_A_DATA/I_A_LABEL not found");

    Tensor fake_img(img_dt->shape, img_dt->dtype);
    float* p = fake_img.data<float>();
    std::default_random_engine rng(42);
    std::normal_distribution<float> nd(0.5f, 0.1f);
    for (int i = 0; i < fake_img.numel(); ++i)
        p[i] = std::clamp(nd(rng), 0.0f, 1.0f);

    Tensor fake_lbl(lbl_dt->shape, lbl_dt->dtype);
    int* q = fake_lbl.data<int>();
    for (int i = 0; i < fake_lbl.numel(); ++i) q[i] = i % 10;

    task.transfer_to_rank(fake_img, *img_dt, 0);
    task.transfer_to_rank(fake_lbl, *lbl_dt, 0);
}
```

调用后在 `run_gpu()` 中设置 `batches == 1` 走单 batch 分支，通过 `fetch_from_rank()` 验证 loss/weight。

---

## 八、修改文件清单

| # | 文件 | 改动 | 阶段 |
|---|------|------|:----:|
| 1 | `include/renaissance/task/task_base.h` | 新增 `MemoryPlan* active_memory_plan_ = &memory_plan_;` | P0 |
| 2 | `src/task/task_base.cpp` | 6 处 `memory_plan_.` → `active_memory_plan_->`；`compile_impl()` 中新增 ArenaKeeper memset + `init_all()` + DeepLearningTask 分支改为 `dl->build_graph_atlas()`；`init()` 替换 `TR_NOT_IMPLEMENTED` 为真实随机初始化 | P0 |
| 3 | `include/renaissance/task/deep_learning_task.h` | 新增 `build_graph_atlas()`, `stream_for()`, `clone_scheduler()` 声明；新增 `train_cg_`, `infer_cg_`, `lr_dtensor_id_`, `lr_host_buffer_` 成员；GraphSlot 扩展 | P0 |
| 4 | `src/task/deep_learning_task.cpp` | GraphSlot 新增 FIRST_FWD_A/B, CAST_AND_CHECK；实现 `build_graph_atlas`, `stream_for`, `clone_scheduler`；重写 `build_exec_table` 按 GraphId 解析；修改 `on_prepare` 调用 `memory_plan_ptr_->finalize()` 并激活 `active_memory_plan_`；重写 `run_train_epoch_gpu`（MMP0 4 阶段 + per-thread scheduler）；修改 `run_gpu` N+1 线程 | P0 |
| 5 | `tests/ref/mnist_mlp_3.cpp` | `compile_for_dry_run()+dry_run()` → `compile()+run()` | P2 |

---

## 九、实施顺序与验收标准

```
Step 0: 架构修复（F1 + F2 + F3）
  内容: active_memory_plan_ + build_graph_atlas() + build_exec_table() GraphId 直解
  验收:
    1. compile() 不 crash
    2. active_memory_plan_->total_bytes() > 0
    3. gpu_exec_ 必需 slot 非 nullptr
  预估: 2h

Step 1: 数据初始化（F4）
  内容: init() 真实路径 + ArenaKeeper memset + init_all()
  验收:
    1. fetch_from_rank() 权重值有方差（非零常数）
    2. compile 输出 ArenaKeeper 分配大小
  预估: 1h

Step 2: 单 batch 手动数据
  内容: inject_fake_data() → batches==1 路径
  验收:
    1. Loss 非 NaN
    2. fetch 权重与初始化后不同（确认更新发生）
    3. 各 rank 线程内 LR 自计算值正确
  预估: 2h

Step 3: Preprocessor 对接 + A/B 乒乓
  内容: N+1 线程 → prep.train() + 完整 run_train_epoch_gpu()
  验收:
    1. 1 epoch 469 batches 无 crash
    2. A/B 区交替读写无死锁
    3. TransferStation 标志正确翻转
    4. Loss 趋势下降
  预估: 3h

Step 4: 20 epoch 完整训练 + 验证
  内容: mnist_mlp_3.cpp compile()+run()
  验收:
    1. Best Val Top-1 > 95%
    2. 无 NaN
    3. 无内存泄漏
    4. 每个 epoch 训练阶段 + 验证阶段均仅一次 join
  预估: 2h
```

---

## 十、风险与缓解

| 风险 | 等级 | 缓解 |
|------|:----:|------|
| `active_memory_plan_` 遗漏某处 `memory_plan_` 使用 | 🔴 | `grep -rn "memory_plan_\." src/task/` 全覆盖验证：compile_alloc_hardware/init/init_all/transfer_to_rank/fetch_from_rank/fill 六法全覆盖 |
| `clone_scheduler()` variant 类型遗漏 | 🟡 | `static_assert` 验证 visitor 覆盖 std::monostate + PolynomialLR + CosineAnnealingLR + StepLR |
| `lr_host_buffer_` 生命周期 | 🟡 | 成员变量保证持久性；`cudaMemcpyAsync` 使用 `&lr_host_buffer_` 而非 lambda 栈变量 |
| Preprocessor `train()` 永不返回 → `join()` 死等 | 🟡 | rank 线程在 last batch 前已设定两个 buffer 为 writeable；debug build 增加 epoch 超时检测 |
| MNIST last batch 96 ≠ 128 | 🟢 | Compiler 编译时 shape 固定为 128，但 TransferStation 只拷贝 96 样本 → padding 区域为 0（ArenaKeeper 已 memset） → loss 略高但不崩溃 |
| `CAST_AND_CHECK` / `GRAD_CONVERT` 在 non-AMP 下为 nullptr | 🟢 | 所有 launch 前 `if (using_amp && ptr)` 双层检查 |
| `FIRST_ALLREDUCE` / `DEEP_ALLREDUCE` 在单卡下为 nullptr | 🟢 | 所有 launch 前 `if (ptr)` 检查，单卡自动跳过 |
| FIRST_LAYER_BWD 在 frozen 场景下为空 | 🟢 | `if (!frozen)` 外层跳过 |

---

## 附录 A：与三方案的完整差异对比

| 决策点 | S 方案 | K 方案 | D 方案 | **ZTN1（本方案）** | 选择理由 |
|--------|--------|--------|--------|-------------------|----------|
| MemoryPlan 修正 | 虚函数 `working_memory_plan()` | `active_memory_plan_` 指针 | `active_memory_plan_` 指针 | **`active_memory_plan_` 指针** | 最小侵入（6 行改 1 指针加），虚函数引入运行时开销且需改更多调用点 |
| Atlas 构建 | DeepLearningTask 内部 | 继承 D 方案 | 独立 `build_graph_atlas()` | **独立 `build_graph_atlas()`** | 不污染 TaskBase，两 Task 类型完全隔离 |
| Exec Table 解析 | — | 继承 D 方案 | GraphId 直解 | **GraphId 直解** | 去掉名字中间层，零 hash 零分支 |
| 初期 ZERO_GRAD | 独立阶段 | Phase 1 ‖ FIRST_FWD | Phase 1 ‖ FIRST_FWD | **Phase 1 ‖ FIRST_FWD** | MMP0 伪代码明确并行；不同显存区域 + 不同 stream → 天然安全 |
| XFER 时机 | 与 FIRST_FWD 并行 | 与 DEEP_FWD_BWD 并行 | 与 DEEP_FWD_BWD 并行 | **与 DEEP_FWD_BWD 并行** | MMP0 原话："最关键的重叠就是传输与深层FWD/BWD的重叠"；给 Preprocessor 更多时间 |
| 学习率机制 | 预计算表 `lr_table_` | `fetch_lr_for_batch()` 只读查询 | per-thread scheduler copy | **per-thread scheduler copy + step()** | 用户明确要求"每个线程都有完全一样的 Scheduler"；只读查询不推进状态，无法适应 step_by_batch |
| Scheduler 竞争 | 无（预计算） | `is_step_by_batch()` 只读查询（无竞争） | 线程独立副本（无竞争） | **线程独立副本（无竞争）** | 拷贝构造保证隔离，不依赖 `is_step_by_batch()` 的 getter 增补 |
| LR 传输 | 查表 + H2D | 成员变量 + UPDATE stream | 成员变量 + UPDATE stream | **成员变量 + UPDATE stream** | cudaMemcpyAsync 同流串行保证顺序；成员变量生命周期安全 |
| 线程 join | epoch 级 | epoch 级 | epoch 级 | **epoch 级** | 用户刚性约束：训练 epoch 内只在末尾 join 一次 |
| Preprocessor | — | prep.train() 独立线程 | N+1 线程 | **N+1 线程** | 不修改 Preprocessor 内部，只调用 train() |

## 附录 B：MMP0 伪代码 → ZTN1 实现映射

| MMP0 伪代码 | ZTN1 实现 |
|-------------|-----------|
| `cudaGraphLaunch(*first_overlap, comp_1)` + `cudaGraphLaunch(*zero_grad, update)` | Phase 1: `gzg` on s_up ‖ `gf` on s_c1 |
| `cudaGraphLaunch(*deep_fwd_bwd, comp_1)` + `cudaGraphLaunch(*transfer_overlap, trans)` | Phase 2: `gd` on s_c1 ‖ `gx_n` on s_tr |
| `cudaGraphLaunch(*first_bwd, comp_1)` + `cudaGraphLaunch(*deep_comm, update)` | Phase 3: `gfb` on s_c1 ‖ `gda` on s_up |
| `cudaGraphLaunch(*transfer_learning_rate, update)` + 权重更新图 | Phase 4: `cudaMemcpyAsync(LR, H2D, s_up)` → `gwu` on s_up |
| `sync_three_compute_streams()` | `sync_comp()`: cudaStreamSynchronize(s_c1 + s_c2 + s_c3) |
| `sync_update_streams()` | `sync_up()`: cudaStreamSynchronize(s_up) |
| "第一个batch的传输必定从TransferStation的A区" | loop 前 `gx_a` pre-transfer, batch 0 从 A 开始 |
| "AB区之间乒乓切换" | `(batch % 2 == 0) ? A : B` + `next_buf = from_a ? 1 : 0` |

---

*本方案基于 AXX.md 全部四稿 + 用户补充意见 + 源码 7 大错位逐行审计。所有方案组件均有源码行号验证，所有性能设计均有物理依据。当前 TR4 v4.20.1 基线上可直接实施。*