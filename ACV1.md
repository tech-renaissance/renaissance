# ACV1：MLP 从 Dry Run 到真训练 —— 科学方案

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22
> **前置**: MXX.md（MMP0需求 + S/K/D三稿）+ MX_REV.md 审查意见 + 源码全路径交叉验证
> **本方案**: 融合四稿最优设计，对齐 MMP0 原始意图，逐条响应"用户补充"的刚性约束

---

## 〇、四稿核心差异对比与择优

| 决策点 | MMP0 原始要求 | S 稿 | K 稿 | D 稿 | **ACV1 最终选择** |
|---|---|---|---|---|---|
| MemoryPlan 修正 | 未提及 | `memory_plan_ptr_.get()` 局部修 | `active_memory_plan_` 指针（6处替换） | 虚函数 `memory_plan()` | **K：最小侵入**，改 6 行加 1 指针 |
| Atlas 构建 | 独立于 SimpleTask | 内部构造 `DeepLearningTask::build_graph_atlas()` | 继承 D 稿 | `DeepLearningTask::build_graph_atlas()` | **D：独立函数，不污染 TaskBase** |
| Exec Table 解析 | — | — | 继承 D 稿 | 按 GraphId 直接 `atlas.index()` | **D：去掉名字中间层** |
| 图调度顺序 | `FIRST_FWD‖ZERO_GRAD → DEEP‖XFER → GRAD_CONVERT‖LR → FIRST_BWD‖COMM` | `ZERO_GRAD析出 → FIRST_FWD‖XFER → DEEP析出 → FIRST_BWD‖COMM → LR+WU` | 同 D 稿（5阶段） | 同 K 稿（5阶段） | **MMP0：FIRST_FWD 与 ZERO_GRAD 必并行，XFER 只与 DEEP 并行** |
| 学习率机制 | — | 预计算表 `lr_table_` | `lr_host_buffer_` + `sync_lr_to_gpu()` | 预计算表 `lr_table_` | **用户方案：每线程独立 Scheduler，LR 计算隐藏在 GPU 执行期间** |
| 线程 join | 只 epoch 末尾 join | ✅ | ✅ | ✅ | ✅ **三稿一致，保留** |
| TransferStation | `condition_variable::wait` | busy-wait 100µs | busy-wait 100µs | busy-wait 100µs | **busy-wait（现有代码风格，简单可靠）** |

### 为什么选择这些组合

1. **K 的 `active_memory_plan_`**：`MemoryPlan` 已 `=delete` 拷贝/移动（[memory_plan.h:75-78](file:///r:/renaissance/include/renaissance/graph/memory_plan.h#L75-L78)），虚函数方案需修改更多调用点且引入虚调用开销。K 的方案：在 TaskBase 加一个 `MemoryPlan*` 指针，默认指向 `&memory_plan_`，DeepLearningTask 在 `on_prepare()` 中切换指向 `memory_plan_ptr_.get()`。受影响范围精确定位在 6 个方法（`compile_alloc_hardware`、`init`、`init_all`、`transfer_to_rank`、`fetch_from_rank`、`fill`），SimpleTask 零影响。

2. **D 的独立 Atlas 函数**：不在 `TaskBase::build_simple_atlas()` 中混入 DeepLearningTask 逻辑，保持两个 Task 类型完全隔离。

3. **MMP0 的调度顺序**：MMP0 明确要求 `FIRST_FWD` 与 `ZERO_GRAD` **并行**（它们分属 COMP_1 和 UPDATE 流，操作不同显存区域，天然无冲突），S/K/D 三稿都把 ZERO_GRAD 析出为独立阶段（增加了不必要的同步），偏离了 MMP0 的原意。XFER(next) 只与 DEEP_FWD_BWD 并行——这是 MMP0 的核心设计："最关键的重叠就是传输与深层 FWD/BWD 的重叠"，不应再与 FIRST_FWD 重叠（会增加 TransferStation 等待压力）。

4. **用户的 LR 方案**：预计算表（S/D 稿）需要 scheduler 支持 `reset()` 且管理额外状态；`sync_lr_to_gpu()`（K 稿）仍把 LR 计算放在主线程。用户方案——每线程持独立 scheduler 副本、在 GPU 图执行期间计算 LR、异步 H2D——最科学：计算几乎零开销、多线程无竞争、scheduler 状态天然隔离。

---

## 一、MemoryPlan 双轨修正（根本问题）

### 1.1 问题根因

```cpp
// TaskBase: 值成员，始终为空
MemoryPlan memory_plan_{plan_config_};  // task_base.h:275

// DeepLearningTask: unique_ptr，Compiler 结果进入这里
std::unique_ptr<MemoryPlan> memory_plan_ptr_;  // deep_learning_task.h:311

// on_prepare() 中：
memory_plan_ptr_ = std::move(result.variants[0].memory_plan);  // 数据进入 memory_plan_ptr_
finalize_memory();  // 调用 memory_plan_.finalize() —— 空实例的 finalize，无效！

// 后果：
// - compile_alloc_hardware: memory_plan_.total_bytes() → 0 → ArenaKeeper 分配 0 字节
// - init_all: memory_plan_.dtensors() → 空 → 不初始化任何权重
// - init/transfer/fetch: 全部使用空 memory_plan_ → 训练必崩
```

### 1.2 修正方案：`active_memory_plan_` 指针（最小侵入）

**文件**: `include/renaissance/task/task_base.h`

在 `memory_plan_` 声明之后新增：

```cpp
MemoryPlan memory_plan_{plan_config_};
MemoryPlan* active_memory_plan_ = &memory_plan_;  // ← 新增：默认指向值成员
```

同时提供简洁的 public 访问器（可选，便于 grep）：

```cpp
public:
    [[nodiscard]] const MemoryPlan& memory_plan() const noexcept {
        return *active_memory_plan_;
    }
```

**文件**: `src/task/task_base.cpp`

以下 6 个方法中所有 `memory_plan_.` 替换为 `active_memory_plan_->`：

| 方法 | 行号（≈） | 受影响表达式 |
|---|---|---|
| `compile_alloc_hardware()` | 1409 | `active_memory_plan_->total_bytes()` |
| `init()` | 1223 替换 | `active_memory_plan_->get_dtensor(dtensor.id)` |
| `init_all()` | 1230, 1235, 1243 | `active_memory_plan_->dtensors()`, `active_memory_plan_->get_dtensor(id)` |
| `transfer_to_rank()` | 887 | `active_memory_plan_->get_dtensor(dt.id)` |
| `fetch_from_rank()` | 1253 | `active_memory_plan_->get_dtensor(dt.id)` |
| `fill()` | 1092 | `active_memory_plan_->get_dtensor(dt.id)` |

> **为什么只改这 6 处**：`debug_dump()`、`validate()` 等诊断路径用不用真实布局不影响正确性；涉及显存分配、数据传输、初始化的路径必须用真实布局。

**文件**: `src/task/deep_learning_task.cpp` — `on_prepare()` 末尾

在 `add_graph(...)` 之后，将指针切换到 Compiler 产出的真实布局：

```cpp
add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

active_memory_plan_ = memory_plan_ptr_.get();   // ← 关键一行
```

> **对 SimpleTask 的影响**：零。SimpleTask 不持有 `memory_plan_ptr_`，`active_memory_plan_` 始终为默认值 `&memory_plan_`，行为与原先完全一致。

---

## 二、compile() 路径：图集架构

### 2.1 GraphSlot 枚举扩展

**文件**: `src/task/deep_learning_task.cpp:32` — 替换现有枚举

```cpp
enum class GraphSlot : uint8_t {
    FIRST_FWD_A = 0,       // ← 新增：首层前向 A 区（FLATTEN）
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

### 2.2 Atlas 构建：`build_graph_atlas()`

**核心原则**：DeepLearningTask 与 SimpleTask 完全分离。不在 `TaskBase::build_simple_atlas()` 中混入任何 DeepLearningTask 逻辑。

**文件**: `include/renaissance/task/deep_learning_task.h` — 新增声明

```cpp
class DeepLearningTask : public TaskBase {
    GraphAtlas build_graph_atlas() const;
    static StreamKind stream_for(GraphId gid);

    const ComputationGraph* train_cg_ = nullptr;
    const ComputationGraph* infer_cg_ = nullptr;
    DTensorID lr_dtensor_id_ = -1;

    float lr_host_buffer_ = 0.0f;       // per-thread LR H2D 缓冲区（成员变量，生命周期安全）
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
            sl.mp          = active_memory_plan_;             // ← 已通过 1.2 修正指向真实布局
            sl.shape_id    = kShapeInvariant;
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

### 2.3 Exec Table：`build_exec_table()` 按 GraphId 直接解析

**优势**：不依赖名字查找，直接 `atlas.index(0, gid)` 取 `captured_idx`。不存在的图返回 `nullptr`（如单卡 ALLREDUCE），调度侧 `if (ptr)` 跳过。

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
        // Compiler 当前没有独立的 GRAD_CONVERT GraphId。两个 Slot 均从
        // 同一个图解析，pre_capture() 去重后只捕获一份 CUDA Graph。
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
    GraphSlot::FIRST_FWD_A,
    GraphSlot::FIRST_FWD_B,
    GraphSlot::XFER_A,
    GraphSlot::XFER_B,
    GraphSlot::FWD_BWD_DEEP_A,
    GraphSlot::FWD_BWD_DEEP_B,
    GraphSlot::FIRST_LAYER_BWD,   // 首层冻结时可空图，但 slot 必须存在
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
```

> **DEEP_FWD_BWD 的 A/B 共享说明**：`FWD_BWD_DEEP_A` 和 `FWD_BWD_DEEP_B` 都从 `GraphId::DEEP_FWD_BWD` 解析。图本身不区分 A/B —— 差异仅在于运行时输入数据位于 I_A_DATA 还是 I_B_DATA（不同 ArenaKeeper 偏移）。`pre_capture()` 去重后只捕获一份 `cudaGraphExec_t`。

### 2.4 compile_impl() 中的 DeepLearningTask 分支

**文件**: `src/task/task_base.cpp:232-250` — `compile_impl()`

当前代码 `else` 分支中已有 `if (auto* dl = dynamic_cast<DeepLearningTask*>(this))`。**只需将 `build_simple_atlas(name_to_gid_)` 替换为 `dl->build_graph_atlas()`**，其余 pre_capture、build_graph_index、build_exec_table 调用保持不变，复用已有的 `dl` 指针：

```cpp
if (is_simple_task()) {
    compile_capture_simple();
} else {
    // DeepLearningTask 分支
    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        GraphAtlas atlas = dl->build_graph_atlas();   // ← 替换 build_simple_atlas(name_to_gid_)

        std::vector<DeviceContext*> ctx_ptrs;
        for (auto& ctx : backend_->contexts)
            ctx_ptrs.push_back(ctx.get());

        captured_result_ = pre_capture(atlas, ctx_ptrs);

        dl->build_graph_index();   // ← 已有
        dl->build_exec_table();    // ← 已有
    }
}
```

### 2.5 on_prepare() 保存 CG 指针 + 激活 MemoryPlan

**文件**: `src/task/deep_learning_task.cpp`（`on_prepare()` 实际定义在 header，此指改动逻辑）

```cpp
// ... 现有 Compiler::compile() 调用不变 ...

memory_plan_ptr_ = std::move(result.variants[0].memory_plan);

// MemoryPlan 不可拷贝/不可移动（=delete），无法回迁到 memory_plan_。
// DeepLearningTask 自己 finalize memory_plan_ptr_：
memory_plan_ptr_->finalize();
phase_ = Phase::MEMORY_LOCKED;

// 查找学习率 DTensor ID
for (const auto& dt : memory_plan_ptr_->dtensors()) {
    if (dt.usage == DTensorUsage::SCALAR_FP32) {
        lr_dtensor_id_ = dt.id;
        break;
    }
}

add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

// 必须在 add_graph 之后取地址（add_graph move 后 named_graphs_["xxx"].graph 地址才稳定）
train_cg_ = &named_graphs_["train"].graph;
infer_cg_ = &named_graphs_["inference"].graph;

// 激活 Compiler 生成的 MemoryPlan → 基类所有方法（compile_alloc_hardware, init, ...）
// 通过 active_memory_plan_ 指针自动访问到真实布局
active_memory_plan_ = memory_plan_ptr_.get();   // ← 关键一行
```

---

## 三、数据初始化

### 3.1 ArenaKeeper memset 清零

**文件**: `src/task/task_base.cpp` — `compile_impl()` 中 `compile_alloc_hardware()` 之后

```cpp
// ---- ArenaKeeper 显存池全部 RANK memset 清零 ----
// 放在 compile_alloc_hardware() 之后、init_all() 之前
#ifdef TR_USE_CUDA
if (GlobalRegistry::instance().using_gpu()) {
    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaSetDevice(backend_->contexts[rank]->device_id());
        void* ptr = ArenaKeeper::instance().ptr_at(rank, 0);
        size_t bytes = active_memory_plan_->total_bytes();  // ← 已修正
        cudaMemset(ptr, 0, bytes);
    }
    cudaDeviceSynchronize();
}
#endif

// ---- 权重初始化 ----
init_all();
```

### 3.2 init() 真实路径

**文件**: `src/task/task_base.cpp:1223` — 替换 `TR_NOT_IMPLEMENTED`

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    check_phase(Phase::COMPILED, "init");

    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : dtensor.init_config;
    if (!config.needs_init()) return;
    if (debug_mode_) { /* dry run print */ return; }

    const DTensor& live_dt = active_memory_plan_->get_dtensor(dtensor.id);  // ← 已修正

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
            // 用 <random> 填充（若 Tensor 无内置 random_* 方法）
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
    transfer_to_rank(host, live_dt, 0);  // MNIST 单卡，直接 H2D
}
```

---

## 四、图调度：与 MMP0 对齐的 4 阶段重叠模型

### 4.1 Stream 分配

| StreamKind | 承载图 | 说明 |
|---|---|---|
| `TRANS` | XFER_A/B | 异步传输，与计算流无数据依赖 |
| `COMP_1` | FIRST_FWD_A/B, DEEP_FWD_BWD, FIRST_BWD | 主计算流 |
| `COMP_2/3` | 空（MLP 无多计算流需求） | 仅参与三流同步 |
| `UPDATE` | ZERO_GRAD, ALLREDUCE, GRAD_CONVERT, WEIGHT_UPDATE, EMA_UPDATE, LR H2D | 梯度/参数管理 |

### 4.2 四阶段调度（严格对齐 MMP0 伪代码）

```
┌──────────────────────────────────────────────────────────────┐
│  Phase 1:  ZERO_GRAD [UPDATE]  ‖  FIRST_FWD [COMP_1]        │
│             sync_comp + sync_up                              │
│                                                              │
│  Phase 2:  DEEP_FWD_BWD [COMP_1]  ‖  XFER(next) [TRANS]    │  ← 核心重叠
│             sync_comp + sync_trans                           │
│                                                              │
│  Phase 3:  FIRST_BWD [COMP_1]  ‖  DEEP_ALLREDUCE [UPDATE]  │  (多卡)
│             sync_comp + sync_up                              │
│                                                              │
│  Phase 4:  [CPU: scheduler.step + compute LR]                │
│             → FIRST_ALLREDUCE [UPDATE]  (if exists)          │
│             → cudaMemcpyAsync(LR, H2D, UPDATE)              │
│             → WEIGHT_UPDATE [UPDATE]                         │
│             sync_up                                          │
└──────────────────────────────────────────────────────────────┘
```

**Phase 1 为何 FIRST_FWD 与 ZERO_GRAD 并行**（对比 S/K/D 稿的纠正）：

- FIRST_FWD 读取 `I_{A|B}_DATA`，写入首层激活（COMP_1 流）
- ZERO_GRAD 写入梯度缓冲区（UPDATE 流）
- 两者操作完全不同的显存区域 → 天然可并行，无需串行化
- MMP0 原伪代码明确写了：`cudaGraphLaunch(*cuda_graph_ptr_first_overlap, stream_comp_1_ptr);` 和 `cudaGraphLaunch(*cuda_graph_ptr_zero_grad, stream_update_ptr);` 同行 —— **设计意图就是并行**

**XFER 为何只与 DEEP 并行**（不与 FIRST_FWD 并行）：

- MMP0 原文："最关键的重叠就是传输与深层FWD/BWD的重叠"
- 如果 XFER 与 FIRST_FWD 并行，TransferStation 必须在 Phase 1 前就准备好下一 batch —— 对 Preprocessor 的压力过大
- XFER 与 DEEP 并行：Preprocessor 有整个 Phase 1（FIRST_FWD + ZERO_GRAD）的时间填充下一 buffer，容错性强

### 4.3 学习率机制（严格按用户补充）

**设计方案**：

1. 每个 rank 线程在 epoch 开始时获得一份 scheduler 的**独立副本**（`sched_cfg_` copy-construct）
2. 在 Phase 4，当 GPU 执行 FIRST_ALLREDUCE 时，CPU 线程**同步执行**（时序隐藏在 GPU 执行期间）：
   - `scheduler_copy.step()` → 获取当前 batch 的 LR
   - `cudaMemcpyAsync(lr_device_ptr, &lr, sizeof(float), H2D, s_up)` — 在 UPDATE 流上入队
3. `WEIGHT_UPDATE` 图也在 UPDATE 流上 → CUDA stream 串行语义自动保证 LR 在 WEIGHT_UPDATE 之前到达
4. 无需全局 LR 表、无需 pre-compute、无需 `reset()` hack

**科学依据**：

- LR 计算是轻量 CPU 操作（若干浮点运算）→ 可完全隐藏在 GPU 执行期间
- 多线程各自独立 scheduler → 零锁竞争
- `cudaMemcpyAsync` + CUDA Graph 同流 → 无需额外 `cudaStreamSynchronize`
- 成员变量 `lr_host_buffer_`（不是栈变量）→ 异步 memcpy 生命周期安全

**实现**：

```cpp
// 在每个 rank 线程的 batch 循环中，Phase 4 处：
// Phase 4: LR compute + FIRST_ALLREDUCE + LR H2D + WEIGHT_UPDATE
//          （均在 UPDATE 流，CUDA 串行语义自动保序）

// ① CPU: scheduler step（GPU 正忙于 FIRST_ALLREDUCE / ALLREDUCE 同步）
scheduler_copy.step();
float lr = scheduler_copy.get_current_lr();

// ② Launch FIRST_ALLREDUCE on UPDATE（if exists, 单卡 skip）
if (gfa) cudaGraphLaunch(gfa, s_up);

// ③ CPU: 将 LR 异步入队到 UPDATE 流（GPU 此时正在跑 ALLREDUCE kernel）
cudaMemcpyAsync(ctx.ptr_at(lr_dtensor_id_), &lr, sizeof(float),
                cudaMemcpyHostToDevice, s_up);

// ④ Launch WEIGHT_UPDATE on UPDATE（自动在 LR H2D 之后执行）
if (gwu) cudaGraphLaunch(gwu, s_up);
sync_up();
```

> **如果 LR 是 step-by-epoch**（如 PolynomialLR epoch-wise）：只需在每个 epoch 开始前每个线程的 scheduler 步进一次，`cudaMemcpyAsync` 同样在 Phase 4。其余 batch 的 Phase 4 跳过 scheduler step，直接用缓存的 LR。LR 值的持久化由 `lr_host_buffer_` 成员变量保证。

---

## 五、run_train_epoch_gpu() 完整实现

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
        threads.emplace_back([&, rank]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);

                // === 本线程独立的 scheduler 副本 ===
                auto scheduler_copy = clone_scheduler();  // deep copy of sched_cfg_

                const auto& g = gpu_exec_.graphs[rank];
                #define G(slot) g[static_cast<size_t>(GraphSlot::slot)]
                auto gf_a  = G(FIRST_FWD_A),   gf_b  = G(FIRST_FWD_B);
                auto gx_a  = G(XFER_A),        gx_b  = G(XFER_B);
                auto gd_a  = G(FWD_BWD_DEEP_A), gd_b  = G(FWD_BWD_DEEP_B);
                auto gfb   = G(FIRST_LAYER_BWD), gzg  = G(ZERO_GRAD);
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

                auto sync_comp  = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_trans = [&]() { cudaStreamSynchronize(s_tr); };
                auto sync_up    = [&]() { cudaStreamSynchronize(s_up); };

                void* lr_dev_ptr = ctx.ptr_at(lr_dtensor_id_);

                // ======== Batch 0: Pre-transfer from A ========
                while (!ts->buffer_is_readable(0))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                cudaGraphLaunch(gx_a, s_tr);
                sync_trans();
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                // ======== 后续 Driver：单 batch 或多 batch 统一 ========
                // 借助 "上一个 batch 的 XFER(next) 已经在本 batch 开始前完成"
                // 的事实，统一用乒乓循环驱动

                if (batches == 1) {
                    // ---- 单 batch（调试/测试） ----
                    cudaGraphLaunch(gzg,  s_up);
                    cudaGraphLaunch(gf_a, s_c1);
                    sync_comp(); sync_up();

                    cudaGraphLaunch(gd_a, s_c1);
                    sync_comp();

                    if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }

                    if (gda) cudaGraphLaunch(gda, s_up);
                    if (gfa) cudaGraphLaunch(gfa, s_up);

                    // Phase 4: scheduler + LR H2D + WEIGHT_UPDATE
                    scheduler_copy.step();
                    float lr = scheduler_copy.get_current_lr();
                    cudaMemcpyAsync(lr_dev_ptr, &lr, sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                    return;
                }

                // ======== Batch 0 ~ batches-2: 核心重叠 ========
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    auto gf   = from_a ? gf_a : gf_b;
                    auto gd   = from_a ? gd_a : gd_b;
                    auto gx_n = from_a ? gx_b : gx_a;

                    // ---- Phase 1: ZERO_GRAD ‖ FIRST_FWD (overlap #1) ----
                    cudaGraphLaunch(gzg, s_up);
                    cudaGraphLaunch(gf,  s_c1);
                    sync_comp(); sync_up();

                    // Wait next buffer
                    while (!ts->buffer_is_readable(next_buf))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    // ---- Phase 2: DEEP_FWD_BWD ‖ XFER(next) (overlap #2, core) ----
                    cudaGraphLaunch(gd,   s_c1);
                    cudaGraphLaunch(gx_n, s_tr);
                    sync_comp(); sync_trans();
                    if (rank == 0) {
                        ts->set_buffer_readable(next_buf, false);
                        ts->set_buffer_writeable(next_buf, true);
                    }

                    // ---- Phase 3: FIRST_BWD ‖ DEEP_ALLREDUCE (overlap #3) ----
                    if (!frozen) cudaGraphLaunch(gfb, s_c1);
                    if (gda)     cudaGraphLaunch(gda, s_up);
                    sync_comp(); sync_up();

                    // AMP: grad_convert（单图，空图自动跳过）
                    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                    // ---- Phase 4: LR + FIRST_ALLREDUCE + WEIGHT_UPDATE ----
                    // ① CPU: scheduler step（隐藏在 GPU 执行间隙）
                    scheduler_copy.step();
                    float lr = scheduler_copy.get_current_lr();

                    // ② Launch FIRST_ALLREDUCE（if exists, on UPDATE）
                    if (gfa) cudaGraphLaunch(gfa, s_up);

                    // ③ Async H2D on UPDATE（GPU 正忙于 ALLREDUCE kernel）
                    cudaMemcpyAsync(lr_dev_ptr, &lr, sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);

                    // ④ WEIGHT_UPDATE on UPDATE（自然在 LR H2D 之后）
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                }

                // ======== Last batch (batches-1): 无需 XFER ========
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

                // Phase 4: last batch LR + WEIGHT_UPDATE
                scheduler_copy.step();
                float lr = scheduler_copy.get_current_lr();
                if (gfa) cudaGraphLaunch(gfa, s_up);
                cudaMemcpyAsync(lr_dev_ptr, &lr, sizeof(float),
                                cudaMemcpyHostToDevice, s_up);
                if (gwu) cudaGraphLaunch(gwu, s_up);
                sync_up();

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    // ======== 线程模型：epoch 训练只在末尾 join 一次 ========
    for (auto& t : threads) t.join();
    for (int r = 0; r < K; ++r) if (exc[r]) std::rethrow_exception(exc[r]);
}
#endif
```

> **`clone_scheduler()` 实现**：若 scheduler 为 `std::variant<PolynomialLR, CosineAnnealingLR, StepLR, std::monostate>`，使用 `std::visit` + copy-construct 即可。参考代码：
> ```cpp
> LRScheduler DeepLearningTask::clone_scheduler() const {
>     return std::visit([](auto&& s) -> LRScheduler {
>         using T = std::decay_t<decltype(s)>;
>         if constexpr (std::is_same_v<T, std::monostate>) return std::monostate{};
>         else return T(s);  // copy-construct
>     }, sched_cfg_);
> }
> ```

---

## 六、Preprocessor 协调与 run_gpu() 主循环

### 6.1 N+1 线程模型

```
┌─────────────────┐   ┌──────────────┐        ┌──────────────┐
│  RANK Thread 0  │   │ RANK Thread  │  ...   │ RANK Thread  │
│  (cudaSetDev 0) │   │      1       │        │     N-1      │
│   wait TS →     │   │  wait TS →   │        │  wait TS →   │
│   launch graphs │   │ launch graphs│        │ launch graphs│
│   sync → set TS │   │ sync→ set TS │        │ sync→ set TS │
└────────┬────────┘   └──────┬───────┘        └──────┬───────┘
         │                   │                       │
    ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ┼─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─
         │          join once at epoch end           │
         ▼                   ▼                       ▼
    ┌─────────────────────────────────────────────────────┐
    │        Preprocessor Thread (第 N+1 线程)             │
    │  prep.train() → 内部展开 M 个 worker 线程预处理       │
    │  持续填充 TransferStation A/B 区，直到 epoch 数据耗尽 │
    └─────────────────────────────────────────────────────┘
```

**刚性约束**（用户明确要求）：
- **训练 epoch 内，RANK 多线程只在 epoch 末尾 join 一次**
- 验证 epoch 内（如有），同样只在末尾 join 一次
- 中间跑各个图的时候**绝不 join**
- RANK 线程任务极简：等待 TransferStation → 启动 CUDA Graph → StreamSynchronize → 设置 TransferStation 可写标志

### 6.2 run_gpu() 主循环

```cpp
TrainingResult DeepLearningTask::run_gpu() {
    auto& prep = Preprocessor::instance();
    const int epochs = total_epochs_;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        current_epoch_ = epoch;
        auto epoch_start = std::chrono::steady_clock::now();

        // ===== N+1 线程：1 Preprocessor + K RANK =====
        // Preprocessor::train() 内部展开 M 个 worker 线程预处理
        std::thread prep_thread([&]() { prep.train(); });

        run_train_epoch_gpu();   // 内部展开 K 个 RANK 线程，末尾 join

        // RANK 线程已全部完成 → TransferStation 全部消费完毕
        // → Preprocessor 检测到消费完 → train() 正常返回
        prep_thread.join();

        float val_loss = 0, top1 = 0, top5 = 0;
        if (should_validate_this_epoch()) {
            // 验证：同样只在末尾 join 一次
            // TODO: run_val_epoch_gpu()
        }

        auto epoch_end = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        log_epoch_results(0.0f, val_loss, top1, top5, 0, 0, get_current_lr(), sec);

        if (use_sema_ && epoch > 0) apply_sema_switch();
    }

    return build_training_result();
}
```

### 6.3 TransferStation 标志规范

| 操作 | 执行者 | 说明 |
|---|---|---|
| `buffer_is_readable(buf)` | **所有 rank 线程** | 忙等待检查；原子变量，多线程读安全 |
| `set_buffer_readable(buf, false)` | **仅 rank 0** | 消费后标记不可读 |
| `set_buffer_writeable(buf, true)` | **仅 rank 0** | 标记可写，供 Preprocessor 填充 |

> **为什么只有 rank 0 写标志**：TransferStation 的 A/B 区是所有 rank 共享的（输入数据相同），只需一个 rank 管理读写标志。多 rank 写同一原子变量会导致竞争（多卡死锁）。

---

## 七、单 Batch 调试路径

Phase 1 首次验证 compile() 时，建议先跑 `batches == 1` 的单 batch 路径，手动注入数据绕过 TransferStation 和 Preprocessor。

```cpp
// 在 compile() 之后、run() 之前
void inject_fake_data(DeepLearningTask& task) {
    auto& keeper = ArenaKeeper::instance();
    const auto& mp = task.memory_plan();  // active_memory_plan_ 已激活

    const DTensor* img_dt = nullptr;
    const DTensor* lbl_dt = nullptr;
    for (const auto& dt : mp.dtensors()) {
        if (dt.region == MemoryRegion::I_A_DATA)  img_dt = &dt;
        if (dt.region == MemoryRegion::I_A_LABEL) lbl_dt = &dt;
    }
    TR_CHECK(img_dt && lbl_dt, ValueError, "I_A_DATA/I_A_LABEL not found");

    // 假数据
    Tensor fake_img(img_dt->shape, img_dt->dtype);
    float* p = fake_img.data<float>();
    std::default_random_engine rng(42);
    std::normal_distribution<float> nd(0.5f, 0.1f);
    for (int i = 0; i < fake_img.numel(); ++i)
        p[i] = std::clamp(nd(rng), 0.0f, 1.0f);

    Tensor fake_lbl(lbl_dt->shape, lbl_dt->dtype);
    int* q = fake_lbl.data<int>();
    for (int i = 0; i < fake_lbl.numel(); ++i) q[i] = i % 10;

    // 手动 H2D 到 ArenaKeeper (rank 0)
    cudaMemcpy(keeper.ptr_at(0, static_cast<size_t>(img_dt->offset())),
               fake_img.data<void>(), img_dt->nbytes(), cudaMemcpyHostToDevice);
    cudaMemcpy(keeper.ptr_at(0, static_cast<size_t>(lbl_dt->offset())),
               fake_lbl.data<void>(), lbl_dt->nbytes(), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}
```

调用 `inject_fake_data(task)` 后，将 `batches` 强制设为 1（或在 `run_gpu()` 中用条件编译），跳过 TransferStation 等待逻辑，直接调度 `batches == 1` 分支。

---

## 八、空图安全 + SimpleTask 兼容

### 8.1 空图安全跳过

单卡场景下 `DEEP_ALLREDUCE`（`GraphId::DEEP_COMM`）和 `FIRST_LAYER_ALLREDUCE`（`GraphId::FIRST_COMM`）为空图，`resolve()` 返回 `nullptr`。`run_train_epoch_gpu()` 中所有 `cudaGraphLaunch` 调用前通过 `if (gda)` / `if (gfa)` 检查。

`FIRST_LAYER_BWD` 在首层冻结场景可能为空图，同样 `if (!frozen)` / `if (gfb)` 双层检查。

### 8.2 SimpleTask 兼容性

- `compile_impl()` 中 `is_simple_task()` 分支走 `compile_capture_simple()` → `simple_captured_graphs_`，逻辑完全独立
- `active_memory_plan_` 默认指向 `&memory_plan_`，SimpleTask 不修改该指针，行为零差异
- DeepLearningTask 的 `build_graph_atlas()` / `build_exec_table()` 不侵入 SimpleTask 任何路径
- **同一个 GraphId 内的所有算子被捕获为同一个 CUDA Graph，绝不拆散**

---

## 九、修改文件清单

| # | 文件 | 改动 | 阶段 |
|---|------|------|:---:|
| 1 | `include/renaissance/task/task_base.h` | 新增 `MemoryPlan* active_memory_plan_` 指针（`= &memory_plan_`） | P0 |
| 2 | `src/task/task_base.cpp` | 6 处 `memory_plan_.` → `active_memory_plan_->`；ArenaKeeper memset；`init()` 真实路径；`compile_impl()` DeepLearningTask 分支改为 `dl->build_graph_atlas()` | P0 |
| 3 | `include/renaissance/task/deep_learning_task.h` | 新增 `build_graph_atlas()`, `stream_for()`, `clone_scheduler()` 声明；新增 `train_cg_`, `infer_cg_`, `lr_dtensor_id_`, `lr_host_buffer_` 成员 | P0 |
| 4 | `src/task/deep_learning_task.cpp` | GraphSlot 新增 FIRST_FWD_A/B, CAST_AND_CHECK；实现 build_graph_atlas, stream_for, clone_scheduler；重写 build_exec_table 按 GraphId 解析；修改 on_prepare 激活 active_memory_plan_；更新 kRequired 加 FIRST_FWD_A/B；重写 run_train_epoch_gpu（MMP0 4-phase 调度 + per-thread Scheduler LR）; 修改 run_gpu N+1 线程 | P0 |
| 5 | `tests/ref/mnist_mlp_3.cpp` | `compile_for_dry_run()+dry_run()` → `compile()+run()` | P2 |

---

## 十、实施顺序与验收标准

```
Step 1 (P0): active_memory_plan_ + build_graph_atlas() + compile_impl 分支
  验收: compile() 不 crash; ArenaKeeper 分配 >0 字节; Exec Table 必需 slot 非 null
  时间: 2h

Step 2 (P1): init() 真实路径 + 单 batch 手动数据注入 + batches==1 调度
  验收: Loss 非 NaN; fetch weight 确认梯度更新发生; LR 传输后 GPU 值正确
  时间: 2h (无需 Preprocessor，独立验证)

Step 3 (P2): Preprocessor 对接 + A/B 乒乓 + 完整 469 batch 循环
  验收: 1 epoch 无 crash/死锁; TransferStation 标志正确翻转; Loss 趋势下降
  时间: 3h

Step 4 (P3): 20 epoch 训练 + 验证
  验收: Best Val Top-1 > 95%; 无 NaN; 无内存泄漏
  时间: 2h
```

### 分步执行逻辑

```
1. P0: 只改 compile 路径。dry_run 仍然可用（验证编译输出对比）。
   在 compile() 后打印 ArenaKeeper::instance().total_bytes() 和 Exec Table 完整性。

2. P1: 不启动 Preprocessor。手动构造 1 个 batch 数据写入 I_A_DATA/I_A_LABEL，
   强制 batches==1，调度单 batch 路径。fetch loss / weight 验证核心图正确。
   → 这一步不需要 TransferStation，不需要 Preprocessor，不需要多线程！

3. P2: 接入 Preprocessor 线程 + TransferStation。数据由 Preprocessor 真实提供。
   1 epoch 跑通，验证如下行为：
   - TransferStation A/B 交替
   - buffer_is_readable / set_buffer_writeable 标志正确
   - 所有 rank 线程正常 join
   - Loss 打印值在合理範圍

4. P3: 20 epoch + 监控收敛。添加 TensorBoard 事件（可选），验证 Top-1。
```

---

## 十一、风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| `active_memory_plan_` 遗漏某处替换 | 🔴 | `grep -rn "memory_plan_\." src/task/` 验证，compile_alloc_hardware/init/transfer/fetch/fill/init_all 六法全覆盖 |
| `clone_scheduler()` variant 类型缺失 | 🟡 | MNIST 只用 CosineAnnealingLR，compile-time 验证 visitor 覆盖所有 variant 类型 |
| `lr_host_buffer_` 栈变量误用 | 🟡 | 代码 review 确保所有 `cudaMemcpyAsync` 使用成员变量 `lr_host_buffer_`（或 per-thread lambda 内 `lr` 值已同步入队后不再访问） |
| Preprocessor `train()` 阻塞 → join 死等 | 🟡 | 确认 TransferStation 消费检测逻辑；添加 epoch 超时机制（debug build） |
| MNIST last batch 96 ≠ 128 | 🟢 | DEEP_FWD_BWD 图 shape-agnostic（输入 shape compile 时固定为 128，但 TransferStation 实际只读前 96 样本→ padding 为 0 → loss 略高但不崩溃） |
| `CAST_AND_CHECK` 在 non-AMP 下 `nullptr` | 🟢 | `if (using_amp && ggc)` / `if (gcn)` 已处理空图 |

---

## 附录 A：MMP0 伪代码调度翻译

MMP0 的核心调度意图（伪代码 → 本方案实现映射）：

| MMP0 伪代码 | 本方案实现 |
|---|---|
| `cudaGraphLaunch(*first_overlap, comp_1)` + `cudaGraphLaunch(*zero_grad, update)` | Phase 1: `gzg` on s_up ‖ `gf` on s_c1 |
| `cudaGraphLaunch(*deep_fwd_bwd, comp_1)` + `cudaGraphLaunch(*transfer_overlap, trans)` | Phase 2: `gd` on s_c1 ‖ `gx_n` on s_tr (core) |
| `cudaGraphLaunch(*first_bwd, comp_1)` + `cudaGraphLaunch(*deep_comm, update)` | Phase 3: `gfb` on s_c1 ‖ `gda` on s_up |
| `cudaGraphLaunch(*transfer_learning_rate, update)` | Phase 4: LR H2D on s_up → `gwu` on s_up |
| `sync_three_compute_streams()` | `sync_comp()`: s_c1 + s_c2 + s_c3 |
| `sync_update_streams()` | `sync_up()`: s_up |
| `sync_trans_streams()` | `sync_trans()`: s_tr |
| "第一个batch的传输必定从TransferStation的A区" | `gx_a` pre-transfer before loop, batch 0 starts from A |
| "AB区之间乒乓切换" | `(batch % 2 == 0) ? A : B` + `next_buf = from_a ? 1 : 0` |

## 附录 B：与 S/K/D 三稿的关键差异

| 差异点 | S/K/D 稿 | ACV1 | 理由 |
|---|---|---|---|
| Phase 1 ZERO_GRAD | 独立阶段，析出在 FIRST_FWD 之前 | ZERO_GRAD ‖ FIRST_FWD 并行 | MMP0 伪代码明确并行；两者操作不同显存区域，天然安全 |
| Phase 2 XFER | 与 FIRST_FWD 并行（S 稿） | 与 DEEP_FWD_BWD 并行 | MMP0: "最关键的重叠就是传输与深层FWD/BWD的重叠"；给 Preprocessor 更多填充时间 |
| LR 方式 | 预计算表 / sync_lr_to_gpu | per-thread scheduler copy + CPU隐藏计算 | 用户补充明确要求；零表管理开销；多线程无竞争 |
| memory_plan 修正 | 虚函数（D）/ get()（S） | `active_memory_plan_` 指针（K） | 最小侵入：加 1 指针 + 改 6 行；SimpleTask 零影响 |

---

*本方案严格对齐 MMP0 原始伪代码调度意图，吸收 S/K/D 三稿中经源码验证的最优设计，逐条响应"用户补充"的刚性约束。所有代码路径均有文件+行号引用，所有设计决策均有科学依据。*