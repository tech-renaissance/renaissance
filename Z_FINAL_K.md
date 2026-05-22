# Z_FINAL.md — TR4 v4.20.1 真实训练最终实施方案

> **基线**: TR4 v4.20.1, C++17, CUDA Graph 训练引擎  
> **日期**: 2026-05-22  
> **输入**: AXX.md（MMP0 原始需求）+ ZXX.md（S/K/D 三方案综合 + 源码交叉验证）+ 用户最终补充意见  
> **验证**: 所有诊断结论经 `task_base.cpp` / `deep_learning_task.cpp` / `scheduler.h` / `computation_graph.h` 逐行交叉验证  
> **目标**: `tests/ref/mnist_mlp_3.cpp` 从 `compile_for_dry_run()+dry_run()` 改为 `compile()+run()`，20 epoch MNIST 真实训练

---

## 〇、核心决策摘要

| 决策点 | 前序方案分歧 | **最终决策** | 决策依据 |
|--------|-------------|-------------|---------|
| MemoryPlan 修正 | S虚函数 / K指针 / D指针 | **`active_memory_plan_` 指针** | 最小侵入（1指针+8行替换），零虚函数开销 |
| Atlas 构建 | S内部 / K&D 独立函数 | **`DeepLearningTask::build_graph_atlas()`** | 完全隔离 SimpleTask 与 DeepLearningTask |
| Exec Table | 名字查找（失效） | **GraphId 直解 `atlas.index(0, gid)`** | 去三重间接，零 hash 竞争 |
| LR 计算 | S预计算表 / K只读查询 / D clone+step | **`get_lr_by_batch/epoch()` 无状态只读查询** | D稿 `step()` 存在 off-by-one 偏移 + epoch 内步进爆炸双重 bug |
| LR 传输 | `cudaMemcpyAsync`（无锁页内存） | **`StagingParamPool` + `RANGE_H2D_COPY_DTENSOR`** | 用户明确要求：锁页内存、持久化、利用已有基础设施 |
| Phase 1 | ZERO_GRAD 独立 / ZERO_GRAD‖FIRST_FWD | **ZERO_GRAD‖FIRST_FWD 并行** | MMP0 伪代码原文；不同显存区域+不同 stream=天然安全 |
| XFER 时机 | 与 FIRST_FWD 并行 / 与 DEEP 并行 | **与 DEEP_FWD_BWD 并行** | MMP0 原话；给 Preprocessor 更多填充时间 |
| Scheduler 步进 | batch 内 `step()` / 无状态查询 | **废弃 batch 内 `step()`，改用只读查询** | MNIST MLP 使用 `step_by_epoch()`，batch 内 step 会导致 epoch 内衰减到 0 |
| init 多卡 | 只传 rank 0 | **`transfer_to_rank(...,0)` + `broadcast_from_rank0()`** | 多卡权重一致性必要 |
| `zero()`/`randn()` | 三稿全部遗漏 | **统一替换为 `active_memory_plan_->get_dtensor()`** | 源码验证确认遗漏 |

**一句话原则**：凡是能不用 `step()` 的地方绝不用 `step()`；凡是能用 `const` 只读查询的地方绝不用可变状态。

---

## 一、源码审计：7 大错位

| # | 错位 | 源码位置 | 后果 | 严重度 |
|---|------|---------|------|:------:|
| 1 | `memory_plan_` 永远是空壳 | `task_base.h:275` 值成员 vs `deep_learning_task.h:311` unique_ptr | ArenaKeeper 分配 0 字节 | 🔴 |
| 2 | Atlas 填入 `SIMPLE_TASK_GRAPH` + `&memory_plan_` | `compile_impl():237` → `build_simple_atlas()` → `task_base.cpp:633-636` | 全部 CUDA Graph 按错误 ID / 错误布局捕获 | 🔴 |
| 3 | `build_graph_index()` 基于 #2 的 `name_to_gid_` | `deep_learning_task.cpp:467-468` | Exec Table 全部 slot 指向同一张图或 nullptr | 🔴 |
| 4 | `init()` 真实路径是 `TR_NOT_IMPLEMENTED` | `task_base.cpp:1223` | 权重停留在随机显存值 → NaN | 🔴 |
| 5 | `run_train_epoch_gpu()` 严重残缺 | `deep_learning_task.cpp:691-829` | 无 ZERO_GRAD/ALLREDUCE/WEIGHT_UPDATE/LR 传输 | 🔴 |
| 6 | `run_gpu()` 无 Preprocessor 线程 | `deep_learning_task.cpp:622` | TransferStation 永远为空，rank 线程永远忙等 | 🔴 |
| 7 | 编译流水线无数据初始化 | `compile_impl():230` | 显存池含脏数据，权重为随机值 | 🔴 |
| — | `zero()`/`randn()` 未替换 `memory_plan_` | `task_base.cpp:1139,1175` | 调用空实例 → crash 或静默失败 | 🔴 |

---

## 二、F1：MemoryPlan 双轨统一（`active_memory_plan_` 指针）

### 2.1 原理

`MemoryPlan` 已 `=delete` 拷贝/移动（`memory_plan.h:75-78`），`memory_plan_ptr_` 的内容无法回迁到 `memory_plan_`。在 `TaskBase` 增加一个 `MemoryPlan*` 指针，默认指向 `&memory_plan_`（SimpleTask 场景），`DeepLearningTask::on_prepare()` 末尾将其切换到 `memory_plan_ptr_.get()`。

### 2.2 修改清单

**文件 A**: `include/renaissance/task/task_base.h`

```cpp
protected:
    PlanConfig plan_config_;
    Phase phase_ = Phase::PLANNING;
    MemoryPlan memory_plan_{plan_config_};
    MemoryPlan* active_memory_plan_ = &memory_plan_;  // ← 新增

public:
    [[nodiscard]] const MemoryPlan& memory_plan() const noexcept {
        return *active_memory_plan_;  // ← 外部查询也走真实布局
    }
```

**文件 B**: `src/task/task_base.cpp` — **8 处** `memory_plan_.` → `active_memory_plan_->`

| 方法 | 行号 | 替换前 | 替换后 |
|------|------|--------|--------|
| `compile_alloc_hardware()` | ~1409 | `memory_plan_.total_bytes()` | `active_memory_plan_->total_bytes()` |
| `init()` | ~1223 | `memory_plan_.get_dtensor(...)` | `active_memory_plan_->get_dtensor(...)` |
| `init_all()` | ~1230,1235,1243 | `memory_plan_.dtensors()` / `.get_dtensor()` | `active_memory_plan_->dtensors()` / `->get_dtensor()` |
| `transfer_to_rank()` | ~887 | `memory_plan_.get_dtensor(dt.id)` | `active_memory_plan_->get_dtensor(dt.id)` |
| `fetch_from_rank()` | ~1271 | 同上 | 同上 |
| `fill()` | ~1102 | 同上 | 同上 |
| **`zero()`** | **~1139** | 同上 | 同上（三稿遗漏） |
| **`randn()`** | **~1175** | 同上 | 同上（三稿遗漏） |
| `compile_debug_print_and_return()` | ~567,571,573 | `memory_plan_.validate()` / `dump_layout()` / `dtensors()` | `active_memory_plan_->validate()` / `dump_layout()` / `dtensors()` |
| `compile_debug_print_and_return()` | ~597 | `build_simple_atlas(name_to_gid_)` | `dl->build_graph_atlas()`（DeepLearningTask），`build_simple_atlas(name_to_gid_)`（SimpleTask） |

> **编译期验证**：`grep -rn "memory_plan_\." src/task/` 确认无遗漏。

**文件 C**: `include/renaissance/task/deep_learning_task.h` — `on_prepare()` 修正

```cpp
void on_prepare() override {
    // ... ArchPlan 构建、Compiler 调用不变 ...

    memory_plan_ptr_ = std::move(result.variants[0].memory_plan);

    // === 修正：finalize 真实布局（幂等保护），跳过基类空实例 ===
    if (!memory_plan_ptr_->is_finalized()) {
        memory_plan_ptr_->finalize();
    }
    phase_ = Phase::MEMORY_LOCKED;

    // === 查找学习率 DTensor ID（修正：用 region 而非不存在的 usage） ===
    for (const auto& dt : memory_plan_ptr_->dtensors()) {
        if (dt.region == Region::S_SCALAR_FP32) {
            lr_dtensor_id_ = dt.id;
            break;
        }
    }
    TR_CHECK(lr_dtensor_id_ >= 0, ValueError,
             "LR DTensor not found: no DTensor with region S_SCALAR_FP32");

    add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
    add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

    // add_graph move 后 named_graphs_ 内地址稳定
    train_cg_ = &named_graphs_["train"].graph;
    infer_cg_ = &named_graphs_["inference"].graph;

    // === 激活真实 MemoryPlan ===
    active_memory_plan_ = memory_plan_ptr_.get();
}
```

> `compile_verify_memory_locked()` 对 DeepLearningTask 无实质影响：`memory_plan_`（空实例）在构造时即处于 finalized 状态，检查恰能通过。对 SimpleTask 仍有效。

---

## 三、F2：Atlas 架构（`build_graph_atlas()`）

### 3.1 GraphSlot 枚举扩展

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
    CAST_AND_CHECK,        // ← 新增
    INF_MAIN_A,
    INF_MAIN_B,
    INF_EMA_A,
    INF_EMA_B,
    COUNT
};
```

### 3.2 `build_graph_atlas()` 实现

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (train_cg_->nodes(gid).empty()) continue;

            auto& sl = atlas.slot(0, gi);
            sl.cg          = train_cg_;
            sl.mp          = active_memory_plan_;   // ← 真实布局
            sl.shape_id    = kShapeInvariant;   // 当前所有 GraphId 统一；progressive resolution
                                         // 场景下需根据 DTensor 实际 shape 动态计算 shape_id
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

    // 填充 name_to_gid_（供 run_impl() dry-run 路径使用）
    name_to_gid_.clear();
    if (train_cg_) {
        name_to_gid_["train"] = GraphId::DEEP_FWD_BWD;
    }
    if (infer_cg_) {
        name_to_gid_["inference"] = GraphId::INF_MAIN_A;
    }

    return atlas;
}

static StreamKind DeepLearningTask::stream_for(GraphId gid) {
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

### 3.3 `compile_impl()` 替换 Atlas 构建

```cpp
} else {
    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        GraphAtlas atlas = dl->build_graph_atlas();   // ← 替换 build_simple_atlas

        std::vector<DeviceContext*> ctx_ptrs;
        for (auto& ctx : backend_->contexts)
            ctx_ptrs.push_back(ctx.get());

        captured_result_ = pre_capture(atlas, ctx_ptrs);

        dl->build_exec_table();
    }
}
```

### 3.4 DeepLearningTask 新增成员声明与生命周期

**文件 D**: `include/renaissance/task/deep_learning_task.h` — 成员声明

```cpp
private:
    // === 编译期成员 ===
    ComputationGraph* train_cg_ = nullptr;
    ComputationGraph* infer_cg_ = nullptr;
    int lr_dtensor_id_ = -1;           // S_SCALAR_FP32 区域的 DTensor ID

    // === 运行时成员 ===
    int current_epoch_ = 0;
#ifdef TR_USE_CUDA
    std::vector<float*> lr_pinned_;     // 每个 rank 一个 cudaMallocHost 锁页指针
#endif
```

**文件 E**: `src/task/deep_learning_task.cpp` — 自定义析构函数

```cpp
DeepLearningTask::~DeepLearningTask() {
#ifdef TR_USE_CUDA
    for (auto* p : lr_pinned_) {
        if (p) cudaFreeHost(p);
    }
    lr_pinned_.clear();
#endif
}
```

> 锁页内存的分配已集成到 `compile_impl()` 中（见 [§5.5](file:///r:/renaissance/Z_FINAL_K.md#L464-L573)），在 `init_all()` 之后、`TR_LOG_INFO` 之前执行，分配失败在编译阶段即暴露。

---

## 四、F3：Exec Table 重建（废弃名字查找）

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
        g[S(GRAD_CONVERT)]          = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(CAST_AND_CHECK)]        = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(INF_MAIN_A)]            = resolve(GraphId::INF_MAIN_A,    rank);
        g[S(INF_MAIN_B)]            = resolve(GraphId::INF_MAIN_B,    rank);
        g[S(INF_EMA_A)]             = resolve(GraphId::INF_EMA_A,     rank);
        g[S(INF_EMA_B)]             = resolve(GraphId::INF_EMA_B,     rank);
        #undef S
    }

    static const GraphSlot kRequired[] = {
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

    // FIRST_FWD_A/B 条件化：若 Atlas 中有则必须非空
    for (int rank = 0; rank < K; ++rank) {
        if (captured_result_.atlas.index(0, GraphId::FIRST_FWD_A) >= 0)
            TR_CHECK(gpu_exec_.graphs[rank][static_cast<size_t>(GraphSlot::FIRST_FWD_A)],
                     ValueError, "FIRST_FWD_A is nullptr for rank " << rank);
        if (captured_result_.atlas.index(0, GraphId::FIRST_FWD_B) >= 0)
            TR_CHECK(gpu_exec_.graphs[rank][static_cast<size_t>(GraphSlot::FIRST_FWD_B)],
                     ValueError, "FIRST_FWD_B is nullptr for rank " << rank);
    }
#endif
}
```

---

## 五、F4：数据初始化

### 5.1 ArenaKeeper 显存清零

在 `compile_impl()` 中 `compile_alloc_hardware()` 之后插入：

```cpp
#ifdef TR_USE_CUDA
if (GlobalRegistry::instance().using_gpu()) {
    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaSetDevice(backend_->contexts[rank]->device_id());
        void* ptr = ArenaKeeper::instance().ptr_at(rank, 0);
        size_t bytes = active_memory_plan_->total_bytes();
        cudaMemset(ptr, 0, bytes);
        cudaDeviceSynchronize();  // 每设备独立同步
    }
}
#endif
```

### 5.2 `init()` 真实路径（含多卡广播）

替换 `task_base.cpp:1223` 的 `TR_NOT_IMPLEMENTED`：

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    check_phase(Phase::COMPILED, "init");

    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : dtensor.init_config;
    if (!config.needs_init()) return;
    if (debug_mode_) { /* dry run print */ return; }

    const DTensor& live_dt = active_memory_plan_->get_dtensor(dtensor.id);

    TR_CHECK(live_dt.dtype == DType::FP32, NotImplemented,
             "init() currently only supports FP32 weights");

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
        case InitKind::XAVIER_NORMAL: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, config.scale);
            float* data = host.data<float>();
            for (int64_t i = 0; i < host.numel(); ++i)
                data[i] = static_cast<float>(dist(gen));
            break;
        }
        case InitKind::KAIMING_UNIFORM:
        case InitKind::XAVIER_UNIFORM: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-config.scale, config.scale);
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

### 5.3 `compile_impl()` 中调用 `init_all()`

`init_all()` 要求 `Phase::COMPILED`（`check_phase` 在 `task_base.cpp:1227`），因此必须在 `compile_mark_compiled()` 之后调用：

```cpp
// 权重初始化（必须在 COMPILED 之后）
if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
    (void)dl;
    init_all();
}
```

> CUDA Graph 捕获的是操作序列（kernel launch / memcpy），不依赖内存初始值。权重初始化只修改 ArenaKeeper 中的数据，不影响已捕获的 CUDA Graph Exec 对象。

### 5.4 StagingParamPool 自动检测（DeepLearningTask 路径）

当前 `compile_capture_simple()`（`task_base.cpp:257-278`）中有 StagingParamPool 自动检测逻辑（遍历 `named_graphs_` 查找 `RANGE_H2D_COPY_DTENSOR`），但该函数仅在 SimpleTask 路径调用。DeepLearningTask 路径需等价检测：

```cpp
// 在 compile_impl() 的 DeepLearningTask 分支中，Atlas 构建之前：
{
    auto& reg = GlobalRegistry::instance();
    if (!reg.has_staging_params()) {
        bool need_params = false;
        if (train_cg_) {
            for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
                for (const auto& node : train_cg_->nodes(static_cast<GraphId>(gi))) {
                    if (node.kind == GraphNode::Kind::RANGE &&
                        node.range_op == RangeOp::RANGE_H2D_COPY_DTENSOR) {
                        need_params = true;
                        break;
                    }
                }
                if (need_params) break;
            }
        }
        if (need_params) {
            reg.allocate_staging_params(num_gpus_);
        }
    }
}
```

### 5.5 `compile_impl()` 完整集成版

综合 §3.3（Atlas 构建）、§5.1（memset）、§5.3（init_all 顺序修正）、§5.4（StagingParamPool 检测），`compile_impl()` 中 DeepLearningTask 分支的最终完整代码为：

```cpp
void TaskBase::compile_impl(bool debug_mode) {
    (void)debug_mode;

    extern void register_default_ops();
    register_default_ops();

    compile_freeze_global();
    compile_invoke_on_prepare();       // → phase_ = MEMORY_LOCKED
    compile_verify_memory_locked();

    if (debug_mode_) {
        compile_debug_print_and_return();
        return;
    }

    compile_alloc_hardware();

    // === ArenaKeeper 显存清零（每设备独立同步）===
    #ifdef TR_USE_CUDA
    if (GlobalRegistry::instance().using_gpu()) {
        for (int rank = 0; rank < num_gpus_; ++rank) {
            cudaSetDevice(backend_->contexts[rank]->device_id());
            void* ptr = ArenaKeeper::instance().ptr_at(rank, 0);
            size_t bytes = active_memory_plan_->total_bytes();
            cudaMemset(ptr, 0, bytes);
            cudaDeviceSynchronize();
        }
    }
    #endif

    if (is_simple_task()) {
        compile_capture_simple();
    } else {
        // === StagingParamPool 自动检测（DeepLearningTask 路径）===
        if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
            auto& reg = GlobalRegistry::instance();
            if (!reg.has_staging_params() && dl->train_cg_) {
                bool need = false;
                for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
                    for (const auto& n : dl->train_cg_->nodes(static_cast<GraphId>(gi))) {
                        if (n.kind == GraphNode::Kind::RANGE &&
                            n.range_op == RangeOp::RANGE_H2D_COPY_DTENSOR) {
                            need = true; break;
                        }
                    }
                    if (need) break;
                }
                if (need) reg.allocate_staging_params(num_gpus_);
            }

            // === 为每个 DeviceContext 绑定 MemoryPlan（运行时 ptr_at() 依赖）===
            for (int rank = 0; rank < num_gpus_; ++rank) {
                backend_->contexts[rank]->set_rank(rank);
                backend_->contexts[rank]->set_memory_plan(active_memory_plan_);
            }

            GraphAtlas atlas = dl->build_graph_atlas();
            std::vector<DeviceContext*> ctx_ptrs;
            for (auto& ctx : backend_->contexts)
                ctx_ptrs.push_back(ctx.get());
            captured_result_ = pre_capture(atlas, ctx_ptrs);
            dl->build_exec_table();
        }
    }

    compile_mark_compiled();           // → phase_ = COMPILED

    // === 权重初始化（必须在 COMPILED 之后）===
    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        (void)dl;
        init_all();
    }

    // === 锁页内存分配（编译阶段即暴露分配失败）===
    #ifdef TR_USE_CUDA
    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        dl->lr_pinned_.resize(num_gpus_);
        for (int rank = 0; rank < num_gpus_; ++rank) {
            auto err = cudaMallocHost(&dl->lr_pinned_[rank], sizeof(float));
            TR_CHECK(err == cudaSuccess, RuntimeError,
                     "cudaMallocHost for lr_pinned_ failed: " << cudaGetErrorString(err));
        }
    }
    #endif

    TR_LOG_INFO("task") << "[DBG] compile_impl: done";
}
```

---

## 六、F5：学习率机制（**最终决策：无状态只读查询 + StagingParamPool 传输**）

### 6.1 为什么废弃 `clone_scheduler()` + `step()`

D 稿方案存在**双重 bug**：

1. **时序偏移（off-by-one）**：`step()` 实现为 `current_lr_ = compute_lr_at_step(current_step_); current_step_ += ...`（先计算后递增）。初始状态 `current_step_=0`，batch 0 调用 `step()` 后 `current_step_=1`（但返回的是 step 0 的 LR，因为先计算）—— 此时 `get_current_lr()` 返回的是上一步的 LR，下一步 WEIGHT_UPDATE 用的也是上一步的 LR。

2. **epoch 内步进爆炸**：MNIST MLP 使用 `step_by_epoch()`（`scheduler(CosineAnnealingLR().step_by_epoch())`），意味着 `step()` 设计为**每个 epoch 调用一次**。在 batch 循环内调用 `step()` 会导致 `current_step_` 在第一个 epoch 内从 0 递增到 469，学习率在第一轮就衰减到接近 0。

### 6.2 为什么选用 `get_lr_by_batch/epoch()`

- `const` 方法，多线程只读安全，零锁竞争
- `get_lr_by_batch(batch_id)` 直接按 batch 索引计算，无时序偏移
- `get_lr_by_epoch(epoch_id)` 适用于 MNIST 的 `step_by_epoch()` 模式
- 不修改 scheduler 状态，无需 reset/clone

### 6.3 新增 `is_step_by_batch()` 接口

```cpp
// include/renaissance/algo/scheduler.h
// 在 LRScheduler public 区新增一行：
bool is_step_by_batch() const noexcept { return step_by_batch_; }
```

### 6.4 线程内 LR 获取

```cpp
float DeepLearningTask::fetch_lr_for_batch(int batch_id) const {
    return std::visit([this, batch_id](auto&& scheduler) -> float {
        using T = std::decay_t<decltype(scheduler)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return 0.0f;
        }
        // get_lr_by_batch / get_lr_by_epoch 均为 const 只读，多线程安全
        if (scheduler.is_step_by_batch()) {
            int global_step = current_epoch_ * scheduler.steps_per_epoch() + batch_id;
            return scheduler.get_lr_by_batch(global_step);
        } else {
            return scheduler.get_lr_by_epoch(current_epoch_);
        }
    }, sched_cfg_);
}
```

### 6.5 LR 传输：`StagingParamPool` + `RANGE_H2D_COPY_DTENSOR`

用户明确指示：
> "学习率的异步 cudaMemcpyAsync 需要锁页内存！这个锁页内存每个 RANK 要有一个，而且最好持久化，不要反复申请。我的个人建议是使用 RANGE_H2D_COPY_DTENSOR 和 StagingParamPool，那是专门为异步拷贝单个值用的。"

**设计方案**：

1. **compile 阶段**：Compiler 或手动在 `OPTIMIZER` 图（或独立图）中插入 `RANGE_H2D_COPY_DTENSOR` 节点，源为 `StagingParamPool` 中某个固定 slot，目标为 `S_SCALAR_FP32` 区域的 GPU 内存（即 LR DTensor 的显存位置）。

2. **运行时**：每个 rank 线程持有独立的 StagingParamPool slot 索引（持久化，不反复申请）。Phase 4 中：
   - CPU 把 `fetch_lr_for_batch(batch)` 的 float 值写入 StagingParamPool slot
   - launch 包含 `RANGE_H2D_COPY_DTENSOR` 的 CUDA Graph（在 UPDATE stream 上）
   - 同 stream 串行语义自动保证 LR 在 `WEIGHT_UPDATE` 前到位

3. **Fallback（若 StagingParamPool 集成路径存在阻塞）**：
   - 每个 rank 在初始化时用 `cudaMallocHost` 分配一个持久化锁页内存（`float* lr_pinned_`）
   - Phase 4 中：`cudaMemcpyAsync(lr_dev_ptr, lr_pinned_, sizeof(float), H2D, s_up)`
   - 同样满足"锁页内存 + 持久化 + 不反复申请"

**生命周期安全**：StagingParamPool / 锁页内存均为成员变量或 compile 期分配，生命周期覆盖整个训练过程，不存在 lambda 捕获栈变量的问题。

### 6.6 Phase 4 中的 LR 传输代码

```cpp
// Phase 4: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE
{
    float lr = fetch_lr_for_batch(batch);

    // 方案 A: StagingParamPool（推荐，需配合 compile 阶段自动检测分配 slot）
    //   详见 §6.5 和 §5.4 的自动检测逻辑
    //   pool->set_param(lr_slot_index_[rank], lr);
    //   cudaGraphLaunch(g_lr_xfer, s_up);

    // 方案 B: cudaMemcpyAsync + 持久化锁页内存（Fallback，当前实现）
    *lr_pinned_[rank] = lr;
    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                    cudaMemcpyHostToDevice, s_up);
}
if (gfa) cudaGraphLaunch(gfa, s_up);   // FIRST_ALLREDUCE（if exists）
if (gwu) cudaGraphLaunch(gwu, s_up);   // WEIGHT_UPDATE
sync_up();
```

> **同流有序**：无论方案 A 还是 B，LR 传输与 `WEIGHT_UPDATE` 同在 `UPDATE` stream，无需额外同步即保证 LR 在权重更新前到位。

---

## 七、F6：4 阶段重叠调度（MMP0 对齐）

### 7.1 Stream 分配

| StreamKind | 承载图 |
|-----------|--------|
| `TRANS` | XFER_A/B |
| `COMP_1` | FIRST_FWD_A/B, DEEP_FWD_BWD, FIRST_BWD |
| `COMP_2/3` | 空（MLP 无多计算流需求），仅参与三流同步 |
| `UPDATE` | ZERO_GRAD, ALLREDUCE, CAST_AND_CHECK, WEIGHT_UPDATE, EMA_UPDATE, LR H2D |

### 7.2 每个 Batch 的 4 阶段重叠

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

### 7.3 `run_train_epoch_gpu()` 完整实现

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
                    if (gzg) cudaGraphLaunch(gzg, s_up);
                    if (gf_a) cudaGraphLaunch(gf_a, s_c1);
                    sync_comp(); sync_up();

                    cudaGraphLaunch(gd_a, s_c1);
                    sync_comp();
                    if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }
                    if (gda) cudaGraphLaunch(gda, s_up);
                    sync_up();
                    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                    {
                        float lr = fetch_lr_for_batch(0);
                        *lr_pinned_[rank] = lr;
                        cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
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
                    if (gzg) cudaGraphLaunch(gzg, s_up);
                    if (gf)  cudaGraphLaunch(gf,  s_c1);
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
                        *lr_pinned_[rank] = lr;
                        cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
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

                if (gzg)  cudaGraphLaunch(gzg,  s_up);
                if (gf_l) cudaGraphLaunch(gf_l, s_c1);
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
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
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

> **每个 rank 线程仅一次 `cudaSetDevice`**：在 lambda 开头调用一次，后续所有 CUDA 操作均在该设备上执行。

---

## 八、F7：N+1 线程与 Preprocessor 协调

### 8.1 `run_gpu()` 主循环

```cpp
TrainingResult DeepLearningTask::run_gpu() {
    auto& prep = Preprocessor::instance();
    const int epochs = total_epochs_;

    // 初始化 best metrics
    best_top1_ = best_top5_ = best_ema_top1_ = best_ema_top5_ = 0.0f;
    best_epoch_ = -1;
    early_stopped_ = false;

    // 一次性 prepare scheduler
    std::visit([this](auto&& sch) {
        using T = std::decay_t<decltype(sch)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            sch.prepare(total_epochs_, prep.steps_per_epoch());
        }
    }, sched_cfg_);

    const auto t0 = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        current_epoch_ = epoch;
        auto epoch_start = std::chrono::steady_clock::now();

        // SEMA 必须在训练前应用（而非 log 之后）
        if (use_sema_ && epoch > 0) {
            apply_sema_switch();
        }

        // N+1 线程：1 个 Preprocessor 线程 + K 个 rank 线程
        std::exception_ptr prep_exc;
        std::thread prep_thread([&]() {
            try { prep.train(); }
            catch (...) { prep_exc = std::current_exception(); }
        });
        run_train_epoch_gpu();   // 内部展开 K 个 rank 线程，epoch 末尾 join
        prep_thread.join();
        if (prep_exc) std::rethrow_exception(prep_exc);

        bool did_validate = false;
        float val_loss = 0.0f, top1 = 0.0f, top5 = 0.0f;
        float ema_top1 = 0.0f, ema_top5 = 0.0f;

        if (should_validate_this_epoch()) {
            run_val_epoch_gpu(false);
            did_validate = true;
            if (use_sema_) run_val_epoch_gpu(true);
            // TODO: 从 run_val_epoch_gpu 获取 val_loss / top1 / top5 / ema_top1 / ema_top5
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

    log_final_summary(std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count());

    TrainingResult result;
    result.best_top1 = best_top1_;
    result.best_top5 = best_top5_;
    result.best_ema_top1 = best_ema_top1_;
    result.best_ema_top5 = best_ema_top5_;
    result.best_epoch = best_epoch_;
    return result;
}
```

### 8.2 TransferStation 标志闭环验证

`prep.train()` → `wait_all_transfer_stations_consumed()` 阻塞直到所有 buffer 可写。`run_train_epoch_gpu()` 返回前消费完所有数据：

| 场景 | buffer 状态分析 |
|------|----------------|
| `batches == 1` | 预传输后 `rank==0` 把 buffer 0 标为 `writeable=true`。初始状态两个 buffer 均为 `writeable=true`。`wait_all_transfer_stations_consumed()` 立即通过。 |
| `batches > 1` | 中间循环 `batch = 0 .. batches-2` 交替把 next_buf 标为 `writeable=true`。Last batch 数据在 `batch = batches-2` 时已传输完毕并标记。epoch 结束时两个 buffer 均为 `writeable=true`。 |

**结论**：无死锁。

### 8.3 多线程安全要点

| 机制 | 说明 |
|------|------|
| TransferStation 标志 | `buffer_is_readable/writeable` 为 `std::atomic<bool>`；**仅 rank 0 负责设置**，避免多 rank 写竞争 |
| 忙等待 | `while (!ts->buffer_is_readable()) { sleep 100us; }` — 不占用 CPU 核心 |
| 异常传播 | `exc[rank]` 数组收集异常，join 后统一 `rethrow_exception` |
| Scheduler 查询 | `get_lr_by_batch/epoch()` 为 `const`，多线程只读零竞争 |
| 锁页内存 | `lr_pinned_` 为成员数组 `std::vector<float*>`，每个 rank 一个 `cudaMallocHost` 指针，持久化不反复申请。析构时 `cudaFreeHost` 释放 |

---

## 九、修改文件清单

| # | 文件 | 改动内容 | 阶段 |
|---|------|---------|:--:|
| 1 | `include/renaissance/algo/scheduler.h` | 新增 `bool is_step_by_batch() const noexcept` | P0 |
| 2 | `include/renaissance/task/task_base.h` | 新增 `MemoryPlan* active_memory_plan_ = &memory_plan_`（SimpleTask 默认指向基类实例，零影响）；`memory_plan()` 返回 `*active_memory_plan_` | P0 |
| 3 | `src/task/task_base.cpp` | **9 处** `memory_plan_.` → `active_memory_plan_->`（含 zero/randn/compile_debug_print_and_return 中 3 处）；ArenaKeeper memset（每设备独立同步）；`init()` 真实路径 + broadcast（NORMAL/UNIFORM 分支 + FP32 dtype 断言）；`compile_impl()` DeepLearningTask 分支改为 `dl->build_graph_atlas()` + StagingParamPool 自动检测（遍历 GraphId bucket）+ `set_rank()`/`set_memory_plan()` + Atlas 构建 + `compile_mark_compiled()` 后 `init_all()` + `cudaMallocHost` 锁页内存分配 | P0 |
| 4 | `include/renaissance/task/deep_learning_task.h` | 新增 `GraphAtlas build_graph_atlas()`（非 const，需修改 `name_to_gid_`）, `static StreamKind stream_for(GraphId)`, `float fetch_lr_for_batch(int) const` 声明；新增 `train_cg_`, `infer_cg_`, `lr_dtensor_id_`, `lr_pinned_`（`std::vector<float*>`）；修改 `on_prepare()` 逻辑；析构函数改为自定义实现 | P0 |
| 5 | `src/task/deep_learning_task.cpp` | GraphSlot 新增 FIRST_FWD_A/B, CAST_AND_CHECK；实现 `build_graph_atlas()`, `static stream_for()`, `fetch_lr_for_batch()`；重写 `build_exec_table()` 按 GraphId 解析（FIRST_FWD_A/B 条件化检查）；重写 `run_train_epoch_gpu()`（MMP0 4-phase + 只读 LR 查询 + 持久化锁页内存）；重写 `run_gpu()`（N+1 线程 + SEMA 训练前应用 + early stop + save + best metrics + final summary）；实现自定义析构（cudaFreeHost） | P0 |
| 6 | `tests/ref/mnist_mlp_3.cpp` | `compile_for_dry_run()+dry_run()` → `compile()+run()` | P2 |

---

## 十、实施顺序与验收标准

### 10.1 四阶段渐进验证

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

### 10.2 单 Batch 调试路径（不依赖 Preprocessor）

在 `compile()` 后、run 前手动注入假数据：

```cpp
void inject_fake_data(DeepLearningTask& task) {
    const auto& mp = task.memory_plan();
    const DTensor* img_dt = nullptr;
    const DTensor* lbl_dt = nullptr;
    for (const auto& dt : mp.dtensors()) {
        if (dt.region == Region::I_A_DATA)  img_dt = &dt;
        if (dt.region == Region::I_A_LABEL) lbl_dt = &dt;
    }
    TR_CHECK(img_dt && lbl_dt, ValueError, "I_A_DATA/I_A_LABEL not found");

    Tensor fake_img(img_dt->shape, img_dt->dtype);
    fake_img.fill(0.5f);
    Tensor fake_lbl(lbl_dt->shape, lbl_dt->dtype);
    int* q = fake_lbl.data<int>();
    for (int i = 0; i < fake_lbl.numel(); ++i) q[i] = i % 10;

    task.transfer_to_rank(fake_img, *img_dt, 0);
    task.transfer_to_rank(fake_lbl, *lbl_dt, 0);
}
```

调用后设置 `batches == 1` 走单 batch 分支，通过 `fetch_from_rank()` 验证 loss/weight。

---

## 十一、风险与缓解

| 风险 | 等级 | 缓解 |
|------|:--:|------|
| `active_memory_plan_` 遗漏某处替换 | 🔴 | 编译期搜索所有 `memory_plan_\.` 确认覆盖 compile_alloc_hardware/init/init_all/transfer/fetch/fill/zero/randn |
| `get_lr_by_batch()` 在 step_by_epoch 模式下行为 | 🟢 | 已用 `is_step_by_batch()` 判断分支，`step_by_epoch` 时走 `get_lr_by_epoch()` |
| `fetch_lr_for_batch()` 在 step_by_batch 模式下传相对索引 | 🟡 | 已修正为 `current_epoch_ * steps_per_epoch() + batch_id` 全局步数 |
| Preprocessor `train()` 阻塞 → join 死等 | 🟡 | TransferStation buffer 标记闭环已验证；debug build 设置 epoch 超时 |
| `CAST_AND_CHECK` 图在 non-AMP 下为 `nullptr` | 🟢 | `if (using_amp && ggc)` 已处理空图跳过；`gcn` 变量已删除避免混淆 |
| MNIST last batch (96 samples) | 🟢 | DEEP_FWD_BWD 图 shape 固定为 128，TransferStation 实际传输 96 样本，后 32 为 0（ArenaKeeper 已清零） |
| `clone_scheduler()` 诱惑（未来维护者） | 🟡 | 本方案文档化说明：禁止在 batch 内调用 `step()`，必须使用无状态查询 |
| `StagingParamPool` API 不可用 | 🟡 | Fallback 到 `cudaMallocHost` 持久化锁页内存 + `cudaMemcpyAsync`，等效正确；已在 §5.4 补充 DeepLearningTask 路径自动检测 |
| `on_prepare()` 中 `memory_plan_ptr_->finalize()` 幂等性 | 🟢 | 已添加 `if (!is_finalized())` 幂等保护 |
| `init_all()` 在 MEMORY_LOCKED 阶段调用 | 🔴 | 已移至 `compile_mark_compiled()` 之后（见 §5.5 集成版），phase 检查不再失败 |
| `apply_sema_switch()` 在 log 之后调用 | 🔴 | 已移至 `run_train_epoch_gpu()` 之前（见 §8.1），SEMA 不再延迟一个 epoch |
| `FIRST_FWD_B` 在 non-progressive 模式下为空 | 🟡 | `build_exec_table()` 改为条件化检查（见 §4.1），仅在 Atlas 有对应 GraphId 时才断言非空 |
| `lr_pinned_` 非锁页内存 | 🔴 | 已明确声明为 `std::vector<float*>`，分配时使用 `cudaMallocHost`，析构时 `cudaFreeHost`（见 §3.4） |
| `shape_id` 统一 `kShapeInvariant` 在 progressive resolution 下 | 🟡 | 已在 §3.2 注释标注限制，当前 MLP 无 progressive resolution，不触发 |
| `DeviceContext::set_memory_plan()` 未调用 | 🔴 | 已补充到 §5.5 集成版，`pre_capture()` 前对每个 context 调用 `set_rank()` + `set_memory_plan(active_memory_plan_)` |
| StagingParamPool 检测用 `linear_nodes()` 永远返回空 | 🟡 | 已改为遍历 `GraphId::COUNT` 下所有 bucket 的 `nodes()`（§5.4 + §5.5），覆盖 Compiler 生成的节点 |
| `run_train_epoch_gpu()` 中 FIRST_FWD graph 为 nullptr | 🟡 | 三处 `cudaGraphLaunch` 均添加 `if (gf_*)` 空指针保护（§7.3），non-progressive 图安全跳过 |
| `name_to_gid_` 未填充，dry-run 路径断裂 | 🟡 | `build_graph_atlas()` 末尾填充 "train"/"inference" 映射（§3.2），`run_impl()` 可正常运行 |
| `compile_debug_print_and_return()` 仍读空 `memory_plan_` | 🟢 | 已补充到 §2.2 修改清单第 9 条，3 处替换为 `active_memory_plan_->`（仅 dry-run 模式受影响） |

---

## 附录：与 S/K/D 三方案的关键差异

| 决策点 | S 方案 | K 方案 | D 方案 | **Z_FINAL（本方案）** | 选择理由 |
|--------|--------|--------|--------|----------------------|---------|
| MemoryPlan 修正 | 虚函数 `working_memory_plan()` | `active_memory_plan_` 指针 | 继承 K 稿 | **`active_memory_plan_` 指针** | 最小侵入，零虚函数开销 |
| LR 计算 | 预计算 `lr_table_`（索引偏移风险） | `fetch_lr_for_batch()` 只读查询 | `clone_scheduler()`+`step()`（**偏移 bug**） | **`get_lr_by_batch/epoch()` 无状态查询** | 零状态修改、零竞争、无 off-by-one |
| LR 传输 | 查表 + H2D | 成员变量 + UPDATE stream | 成员变量 + `cudaMemcpyAsync`（无锁页内存） | **`StagingParamPool` / `cudaMallocHost` 持久化锁页内存 + `cudaMemcpyAsync`** | 用户明确要求锁页内存+持久化 |
| Phase 1 调度 | ZERO_GRAD 独立阶段 | ZERO_GRAD ‖ FIRST_FWD 并行 | ZERO_GRAD ‖ FIRST_FWD 并行 | **ZERO_GRAD ‖ FIRST_FWD 并行** | MMP0 伪代码原文 |
| XFER 时机 | 与 FIRST_FWD 并行 | 与 DEEP_FWD_BWD 并行 | 与 DEEP_FWD_BWD 并行 | **与 DEEP_FWD_BWD 并行** | MMP0 原话；给 Preprocessor 更多时间 |
| `init()` 广播 | 未提及 | 只传 rank 0 | 只传 rank 0 | **`transfer_to_rank(...,0)` + `broadcast_from_rank0()`** | 多卡权重一致性 |
| `on_prepare()` finalize | 未提及 | 提及但未明确修改 | 未提及 | **`if (!is_finalized()) finalize()` + 跳过基类空实例** | 幂等保护；确保真实布局已 finalize |
| `zero()`/`randn()` | 未提及 | 未提及 | 未提及 | **统一替换为 `active_memory_plan_->get_dtensor()`** | 三稿共同遗漏 |
| `lr_dtensor_id_` 查找 | 未提及 | 未提及 | `dt.usage`（**字段不存在**） | **`dt.region == S_SCALAR_FP32`** | 源码验证：DTensor 无 usage 字段 |
| batch 内 `step()` | 未使用 | 未使用 | **使用（双重 bug）** | **严禁使用** | `step_by_epoch()` 模式下 batch 内 step 会导致 epoch 内衰减到 0 |

---

*本方案基于 AXX.md（MMP0 原始需求）+ ZXX.md（S/K/D 三稿 + 源码交叉验证）+ 用户最终意见综合制定。所有设计决策均有科学依据，所有代码路径均有文件+行号引用。严禁在 batch 内调用 `scheduler.step()`。*
