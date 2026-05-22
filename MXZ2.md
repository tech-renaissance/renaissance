# MXZ2：MLP 从 Dry Run 到真实训练——以计算通信重叠为核心的实施方案

> **版本**: V1.0  
> **日期**: 2026-05-22  
> **基线**: TR4 v4.20.4  
> **目标**: `tests/ref/mnist_mlp_3.cpp` 从 `compile_for_dry_run()+dry_run()` 改为 `compile()+run()`，**必须实现计算通信重叠**

---

## 一、执行摘要

本方案基于对 `MMP0.md`（原始需求）、`MMP2.md`（缺口分析）、`MMP1.md`/`MMP3.md`（参考方案）以及**实际代码**的完整交叉验证后制定。

**核心结论**: 当前代码在 `compile()` 真实路径下会直接崩溃（`build_exec_table()` 的 `kRequired` 检查），因为 `build_simple_atlas()` 把 Compiler 已分解的 16 个子图全部坍缩为 `SIMPLE_TASK_GRAPH`。修复 Atlas 映射后，代码骨架（多线程 per-rank、A/B 乒乓等待、CUDA Graph 启动）已经存在，但**缺少 5 张核心图的调度**，且**未启动 Preprocessor**。本方案按 **4 个 Phase** 推进，**每个 Phase 都以实现特定重叠场景为硬性验收标准**。

**5 个必须实现的重叠场景（MMP0.md 伪代码）**:

| # | 重叠场景 | 流组合 | 说明 |
|---|---------|--------|------|
| 1 | **ZERO_GRAD \|\| FIRST_FWD** | UPDATE + COMP_1 | 梯度清零与首层前向并行 |
| 2 | **DEEP_FWD_BWD \|\| XFER(next)** | COMP_1 + TRANS | 深层计算与下一 batch 传输重叠（核心） |
| 3 | **GRAD_CONVERT \|\| LR_DTENSOR_XFER** | UPDATE + TRANS | AMP 梯度转换与学习率传输并行 |
| 4 | **FIRST_LAYER_BWD \|\| DEEP_ALLREDUCE** | COMP_1 + UPDATE | 首层反向与深层梯度通信并行 |
| 5 | **FIRST_ALLREDUCE \|\| WEIGHT_UPDATE** | UPDATE + UPDATE | 串行（同流），但紧跟在重叠后 |

---

## 二、代码现状快照（经实际走读确认）

### 2.1 已具备的骨架（可直接复用）

| 组件 | 状态 | 位置 |
|------|------|------|
| `run_train_epoch_gpu()` 多线程 per-rank | ✅ 完整 | `deep_learning_task.cpp:690` |
| `cudaGraphLaunch` + `cudaStreamSynchronize` 调度 | ✅ 完整 | `deep_learning_task.cpp:729` |
| A/B 乒乓等待 `buffer_is_readable` | ✅ 完整 | `deep_learning_task.cpp:726` |
| `build_exec_table()` 框架 + `GpuExecTable` | ✅ 完整 | `deep_learning_task.cpp:475` |
| `GraphSlot` 枚举 16 个 slot | ✅ 完整 | `deep_learning_task.cpp:32` |
| `run_gpu()` epoch 循环 + 日志 + SEMA | ✅ 完整 | `deep_learning_task.cpp:537` |
| `pre_capture()` 去重 + cuDNN warmup | ✅ 完整 | `captured_graph.cpp:172` |
| `Compiler` 16 子图分解 | ✅ 完整 | `compiler.cpp` |

### 2.2 缺失或错误的关键路径

| # | 问题 | 根因 | 影响 |
|---|------|------|------|
| A | **`build_simple_atlas()` 坍缩所有子图** | 所有 named_graph 映射到 `SIMPLE_TASK_GRAPH` | `pre_capture()` 只捕获 1 张图，`build_exec_table()` `kRequired` 检查崩溃 |
| B | **`build_exec_table()` 名字映射与 `named_graphs_` 不匹配** | `resolve("fwd_bwd_deep_a")` 查无此图 | 即使 Atlas 修复后，按名字查找仍返回 `nullptr` |
| C | **`init()` / `init_all()` 抛异常** | `TR_NOT_IMPLEMENTED` | 权重未初始化，训练 loss 必 NaN |
| D | **`run_train_epoch_gpu()` 缺 5 张图** | 只调了 xfer/deep/first | 无 zero_grad/allreduce/weight_update |
| E | **未启动 Preprocessor 线程** | `run_gpu()` 无 `prep.train()` | TransferStation 永不填充 |
| F | **ArenaKeeper 未 memset 清零** | `compile_alloc_hardware()` 无清零 | 梯度/特征图含垃圾值 |
| G | **FIRST_FWD 未启动** | `GraphSlot` 无 `FIRST_FWD_A/B`，`run_train_epoch_gpu()` 未启动 | FLATTEN 正向被跳过，FC1 输入未初始化 |
| H | **学习率未 H2D 传输** | `scheduler.step()` 后无 `transfer(lr)` | GPU 优化器始终用初始 lr |

---

## 三、重叠调度模型（基于 MMP0.md 伪代码精化）

### 3.1 单 Batch 流水线（中间 batch，batch 0 ~ N-2）

假设 batch `i` 的数据已在 **A 区**（由上一轮 `XFER_A` 传输）：

```cpp
// ===== Phase 1: ZERO_GRAD || FIRST_FWD（双图并行） =====
cudaGraphLaunch(g_zero_grad,     s_update);   // UPDATE流：清零梯度
cudaGraphLaunch(g_first_fwd_a,   s_comp1);     // COMP_1流：FLATTEN正向
sync(COMP_1, COMP_2, COMP_3, UPDATE);

// ===== Phase 2: DEEP_FWD_BWD || XFER_B（核心重叠） =====
// 启动深层计算（当前batch）与下一batch传输并行
cudaGraphLaunch(g_deep_a,        s_comp1);     // COMP_1流：FC+ReLU+SoftmaxCE FWD/BWD
cudaGraphLaunch(g_xfer_b,        s_trans);     // TRANS流：batch i+1 → B区
sync(COMP_1, COMP_2, COMP_3, TRANS);
// 注：XFER_B 前已由 while(!readable) 确保 B 区就绪，launch 后两者在各自流上并行

// ===== Phase 3: GRAD_CONVERT || LR_XFER（双图并行，AMP/学习率） =====
if (amp) cudaGraphLaunch(g_grad_convert, s_update);  // UPDATE流：FP16→FP32
cudaGraphLaunch(g_lr_xfer,       s_trans);     // TRANS流：学习率 H2D
sync(UPDATE, TRANS);

// ===== Phase 4: FIRST_LAYER_BWD || DEEP_ALLREDUCE（双图并行） =====
cudaGraphLaunch(g_first_bwd,     s_comp1);     // COMP_1流：FLATTEN反向
cudaGraphLaunch(g_deep_allreduce, s_update);    // UPDATE流：深层梯度AllReduce
sync(COMP_1, COMP_2, COMP_3, UPDATE);

// ===== Phase 5: FIRST_ALLREDUCE + WEIGHT_UPDATE（串行，同UPDATE流） =====
if (multi_rank) cudaGraphLaunch(g_first_allreduce, s_update);
cudaGraphLaunch(g_weight_update, s_update);     // UPDATE流：SGD+Momentum
sync(UPDATE);

// ===== Phase 6: CPU 学习率步进 + EMA =====
if (rank == 0) scheduler.step();
// 学习率标量更新到 staging buffer（下一batch的LR_XFER使用）
if (ema) cudaGraphLaunch(g_ema_update, s_update);
sync(UPDATE);

// ===== Phase 7: 标记 B 区可写（供 Preprocessor 复用） =====
if (rank == 0) {
    ts->set_buffer_readable(1, false);
    ts->set_buffer_writeable(1, true);
}
```

### 3.2 Batch 乒乓切换规则

| batch_id | 数据来源 | FIRST_FWD | DEEP_FWD_BWD | XFER(下一batch) | FIRST_BWD | ALLREDUCE/UPDATE |
|----------|---------|-----------|--------------|-----------------|-----------|-----------------|
| 0 (预传A) | A区 | FIRST_FWD_A | DEEP_A | XFER_B | FIRST_BWD | ✓ |
| 1 | B区 | FIRST_FWD_B | DEEP_B | XFER_A | FIRST_BWD | ✓ |
| 2 | A区 | FIRST_FWD_A | DEEP_A | XFER_B | FIRST_BWD | ✓ |
| ... | ... | ... | ... | ... | ... | ... |
| N-1 (last) | A/B | FIRST_FWD_{A/B} | DEEP_{A/B} | **无** | FIRST_BWD | ✓ |

### 3.3 Stream 分配（不可更改）

| StreamKind | 承载图 | 说明 |
|-----------|--------|------|
| `TRANS` | XFER_A/B, LR_DTENSOR_XFER | H2D 异步传输，与计算流天然不冲突 |
| `COMP_1` | FIRST_FWD_A/B, DEEP_FWD_BWD, FIRST_LAYER_BWD | 主计算流 |
| `COMP_2` | （MLP 预留） | 当前 MLP 不需要 |
| `COMP_3` | （MLP 预留） | 当前 MLP 不需要 |
| `UPDATE` | ZERO_GRAD, ALLREDUCE, WEIGHT_UPDATE, GRAD_CONVERT, EMA_UPDATE | 梯度/参数更新 |

---

## 四、分阶段实施方案

### Phase 1：修复 compile() 路径——Atlas 映射 + 权重初始化

**目标**: `compile()` 成功生成所有 16 个 `GraphSlot` 的有效 `cudaGraphExec_t`，`init_all()` 完成权重初始化，无崩溃。

**重叠验收**: 本阶段不要求重叠，但 **Phase 2 起所有阶段必须以重叠为验收标准**。

#### 改动 1.1：`build_simple_atlas()` 提取 Compiler 子图

**文件**: `src/task/task_base.cpp`  
**函数**: `TaskBase::build_simple_atlas()`  
**行号**: ~616

**当前代码（错误）**:
```cpp
for (const auto& name : sorted_names) {
    const auto& entry = named_graphs_[name];
    GraphId gid = GraphId::SIMPLE_TASK_GRAPH;          // ← 致命：全部坍缩
    auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
    sl.cg = &entry.graph;
    sl.mp = &memory_plan_;
    sl.stream_kind = entry.stream;
    name_to_gid[name] = gid;
}
```

**修改为**:
```cpp
for (const auto& name : sorted_names) {
    const auto& entry = named_graphs_[name];

    // DeepLearningTask：遍历 ComputationGraph 内部分解的每个 GraphId
    for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
        GraphId gid = static_cast<GraphId>(gi);
        if (gid == GraphId::SIMPLE_TASK_GRAPH || gid == GraphId::COUNT)
            continue;

        const auto& nodes = entry.graph.nodes(gid);
        if (nodes.empty()) continue;  // 空子图跳过

        auto& sl = atlas.slot(0, gi);
        sl.cg = &entry.graph;
        sl.mp = &memory_plan_;
        sl.stream_kind = entry.stream;

        if (!memory_plan_.dtensors().empty()) {
            const auto& dt = memory_plan_.dtensors()[0];
            sl.shape_id = ShapeId{dt.shape.n(), dt.shape.h(),
                                   dt.shape.w(), dt.shape.c()};
        } else {
            sl.shape_id = kShapeInvariant;
        }

        // 保留 name → 任一 gid 的映射（用于 run() 兼容路径）
        name_to_gid[name] = gid;
    }
}
```

**验证**: `compile()` 后打印 `captured_result_.graphs.size()` 应 ≥ 11（TRANSFER_A/B, FIRST_FWD_A/B, DEEP_FWD_BWD, ZERO_GRAD, FIRST_BWD, FIRST/DEEP_COMM, CAST_AND_CHECK, OPTIMIZER）。

#### 改动 1.2：`build_exec_table()` 改为按 GraphId 直接解析

**文件**: `src/task/deep_learning_task.cpp`  
**函数**: `DeepLearningTask::build_exec_table()`  
**行号**: ~483

**当前代码（错误）**:
```cpp
auto resolve = [&](const std::string& name, int rank) -> cudaGraphExec_t {
    auto it = name_to_graph_index_.find(name);
    if (it == name_to_graph_index_.end()) return nullptr;
    return static_cast<cudaGraphExec_t>(
        captured_result_.graphs[it->second].native_exec(rank));
};
// 然后用 resolve("fwd_bwd_deep_a") 等查找 —— 这些名字在 named_graphs_ 中不存在
```

**修改为**:
```cpp
auto resolve_gid = [&](GraphId gid, int rank) -> cudaGraphExec_t {
    int32_t idx = captured_result_.atlas.index(0, gid);
    if (idx < 0) return nullptr;
    return static_cast<cudaGraphExec_t>(
        captured_result_.graphs[idx].native_exec(rank));
};

for (int rank = 0; rank < K; ++rank) {
    gpu_exec_.device_ids[rank] = context(rank).device_id();
    auto& g = gpu_exec_.graphs[rank];
    g.resize(static_cast<size_t>(GraphSlot::COUNT), nullptr);

    // 新增：FIRST_FWD_A/B slot
    g[static_cast<size_t>(GraphSlot::FIRST_FWD_A)]      = resolve_gid(GraphId::FIRST_FWD_A, rank);
    g[static_cast<size_t>(GraphSlot::FIRST_FWD_B)]      = resolve_gid(GraphId::FIRST_FWD_B, rank);

    g[static_cast<size_t>(GraphSlot::XFER_A)]           = resolve_gid(GraphId::TRANSFER_A, rank);
    g[static_cast<size_t>(GraphSlot::XFER_B)]           = resolve_gid(GraphId::TRANSFER_B, rank);
    g[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_A)]   = resolve_gid(GraphId::DEEP_FWD_BWD, rank);
    g[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_B)]   = resolve_gid(GraphId::DEEP_FWD_BWD, rank);
    // 注：DEEP_FWD_BWD 图本身不分 A/B，A/B 差异来自输入数据所在的 I_A_DATA / I_B_DATA
    // 但如果 Compiler 为 A/B 生成了不同图，则分别解析。此处假设共享同一张 DEEP_FWD_BWD 图。
    g[static_cast<size_t>(GraphSlot::FIRST_LAYER_BWD)]  = resolve_gid(GraphId::FIRST_BWD, rank);
    g[static_cast<size_t>(GraphSlot::ZERO_GRAD)]        = resolve_gid(GraphId::ZERO_GRAD, rank);
    g[static_cast<size_t>(GraphSlot::DEEP_ALLREDUCE)]   = resolve_gid(GraphId::DEEP_COMM, rank);
    g[static_cast<size_t>(GraphSlot::FIRST_LAYER_ALLREDUCE)] = resolve_gid(GraphId::FIRST_COMM, rank);
    g[static_cast<size_t>(GraphSlot::WEIGHT_UPDATE)]    = resolve_gid(GraphId::OPTIMIZER, rank);
    g[static_cast<size_t>(GraphSlot::EMA_UPDATE)]       = resolve_gid(GraphId::EMA_UPDATE, rank);
    g[static_cast<size_t>(GraphSlot::GRAD_CONVERT)]     = resolve_gid(GraphId::CAST_AND_CHECK, rank);
    g[static_cast<size_t>(GraphSlot::INF_MAIN_A)]       = resolve_gid(GraphId::INF_MAIN_A, rank);
    g[static_cast<size_t>(GraphSlot::INF_MAIN_B)]       = resolve_gid(GraphId::INF_MAIN_B, rank);
    g[static_cast<size_t>(GraphSlot::INF_EMA_A)]        = resolve_gid(GraphId::INF_EMA_A, rank);
    g[static_cast<size_t>(GraphSlot::INF_EMA_B)]        = resolve_gid(GraphId::INF_EMA_B, rank);
}
```

**注意**: `GraphSlot` 枚举需新增 `FIRST_FWD_A` 和 `FIRST_FWD_B`：

```cpp
enum class GraphSlot : uint8_t {
    FIRST_FWD_A = 0,    // ← 新增
    FIRST_FWD_B,        // ← 新增
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
    INF_MAIN_A,
    INF_MAIN_B,
    INF_EMA_A,
    INF_EMA_B,
    COUNT
};
```

**验证**: `compile()` 后 `gpu_exec_.graphs[rank][slot] != nullptr` 对所有 `kRequired` slot。

#### 改动 1.3：ArenaKeeper memset + `init_all()` 实现

**文件**: `src/task/task_base.cpp`  
**函数**: `TaskBase::compile_alloc_hardware()` 和 `TaskBase::init()`

**在 `compile_alloc_hardware()` 末尾添加**:
```cpp
// ArenaKeeper 显存池全清零（防止脏数据）
if (reg.using_gpu()) {
    for (int rank = 0; rank < num_gpus_; ++rank) {
        int gpu_id = reg.gpu_ids()[rank];
        cudaSetDevice(gpu_id);
        void* ptr = ArenaKeeper::instance().ptr_at(rank, 0);
        size_t bytes = memory_plan_.total_bytes();
        cudaMemset(ptr, 0, bytes);
    }
    cudaDeviceSynchronize();
}
```

**在 `init()` 中替换 `TR_NOT_IMPLEMENTED`**:
```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    check_phase(Phase::COMPILED, "init");
    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : dtensor.init_config;
    if (!config.needs_init()) return;
    if (debug_mode_) { /* dry run print */ return; }

    // 生成 Host 初始化数据
    Tensor host = generate_init_tensor(dtensor.shape, dtensor.dtype, config);

    // 广播到所有 rank
    for (int rank = 0; rank < num_gpus_; ++rank) {
        transfer_to_rank(host, dtensor, rank);
    }
}
```

**在 `compile_impl()` 末尾、标记 compiled 之前调用 `init_all()`**:
```cpp
if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
    dl->init_all();  // ← 新增
}
compile_mark_compiled();
```

**验证**: `compile()` 后 fetch 权重 DTensor，验证值符合 Xavier/He 分布，非全 0 或 NaN。

---

### Phase 2：Preprocessor 对接 + A 区传输验证 + 首个重叠 batch

**目标**: Preprocessor 线程与 rank 线程并发运行，A 区数据正确到达 GPU，**首个 batch 完成 ZERO_GRAD || FIRST_FWD_A 重叠**。

#### 改动 2.1：`run_gpu()` 启动 Preprocessor 线程

**文件**: `src/task/deep_learning_task.cpp`  
**函数**: `DeepLearningTask::run_gpu()`  
**行号**: ~537

**修改 epoch 循环**:
```cpp
for (int epoch = 0; epoch < total_epochs_; ++epoch) {
    current_epoch_ = epoch;
    // ...（progressive resolution 等已有代码保留）...

    // ===== 启动 Preprocessor（第 N+1 线程） =====
    std::thread prep_thread([&prep]() {
        prep.train();  // 阻塞：执行一个完整 epoch 的预处理
    });

    // ===== 训练 =====
    run_train_epoch_gpu();

    // ===== 等待 Preprocessor 完成 =====
    prep_thread.join();

    // ...（验证等已有代码保留）...
}
```

**验证**: 运行 `mnist_mlp_3.exe`，确认 Preprocessor 线程启动，TransferStation A 区被填满，`buffer_is_readable(0)` 从 false → true。

#### 改动 2.2：`run_train_epoch_gpu()` 重写——首个 batch + ZERO_GRAD || FIRST_FWD 重叠

**文件**: `src/task/deep_learning_task.cpp`  
**函数**: `DeepLearningTask::run_train_epoch_gpu()`  

**Phase 2 只跑通单 batch（batches == 1 路径或循环第一轮），验证以下重叠**:

```cpp
#ifdef TR_USE_CUDA
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    const bool frozen = is_first_layer_frozen();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    const int K = num_gpus_;
    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, frozen, batches, ts, &exc]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_first_fwd_a = g_tab[static_cast<size_t>(GraphSlot::FIRST_FWD_A)];
                auto g_first_fwd_b = g_tab[static_cast<size_t>(GraphSlot::FIRST_FWD_B)];
                auto g_xfer_a      = g_tab[static_cast<size_t>(GraphSlot::XFER_A)];
                auto g_xfer_b      = g_tab[static_cast<size_t>(GraphSlot::XFER_B)];
                auto g_deep_a      = g_tab[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_A)];
                auto g_deep_b      = g_tab[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_B)];
                auto g_first_bwd   = g_tab[static_cast<size_t>(GraphSlot::FIRST_LAYER_BWD)];
                auto g_zero_grad   = g_tab[static_cast<size_t>(GraphSlot::ZERO_GRAD)];
                auto g_deep_ar     = g_tab[static_cast<size_t>(GraphSlot::DEEP_ALLREDUCE)];
                auto g_first_ar    = g_tab[static_cast<size_t>(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto g_weight_upd  = g_tab[static_cast<size_t>(GraphSlot::WEIGHT_UPDATE)];

                DeviceContext& ctx = context(rank);
                cudaStream_t s_trans  = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_comp1  = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_comp2  = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_comp3  = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
                cudaStream_t s_update = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));

                // =============================================================
                // Phase 0: 预传输第一个 batch（A区）
                // =============================================================
                while (!ts->buffer_is_readable(0)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                cudaGraphLaunch(g_xfer_a, s_trans);
                cudaStreamSynchronize(s_trans);
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                // =============================================================
                // Phase 2A: 仅验证首个 batch 的重叠
                // =============================================================
                if (batches == 1) {
                    // ---- 重叠 #1: ZERO_GRAD || FIRST_FWD_A ----
                    cudaGraphLaunch(g_zero_grad,   s_update);
                    cudaGraphLaunch(g_first_fwd_a, s_comp1);
                    cudaStreamSynchronize(s_comp1);
                    cudaStreamSynchronize(s_comp2);
                    cudaStreamSynchronize(s_comp3);
                    cudaStreamSynchronize(s_update);

                    // ---- DEEP_FWD_BWD（无下一batch传输，因为是唯一batch） ----
                    cudaGraphLaunch(g_deep_a, s_comp1);
                    cudaStreamSynchronize(s_comp1);
                    cudaStreamSynchronize(s_comp2);
                    cudaStreamSynchronize(s_comp3);

                    // ---- FIRST_LAYER_BWD ----
                    if (!frozen) {
                        cudaGraphLaunch(g_first_bwd, s_comp1);
                        cudaStreamSynchronize(s_comp1);
                        cudaStreamSynchronize(s_comp2);
                        cudaStreamSynchronize(s_comp3);
                    }

                    // ---- ALLREDUCE + WEIGHT_UPDATE（单卡时 g_deep_ar/g_first_ar 为 nullptr，跳过） ----
                    if (g_deep_ar)   cudaGraphLaunch(g_deep_ar,  s_update);
                    if (g_first_ar)  cudaGraphLaunch(g_first_ar, s_update);
                    cudaStreamSynchronize(s_update);

                    if (g_weight_upd) cudaGraphLaunch(g_weight_upd, s_update);
                    cudaStreamSynchronize(s_update);

                    if (rank == 0) scheduler_step();
                    return;
                }

                // =============================================================
                // Phase 2B: 完整多 batch 重叠循环（Phase 3 再实现）
                // =============================================================
                // ...（留空，Phase 3 填充）...

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank) {
        if (exc[rank]) std::rethrow_exception(exc[rank]);
    }
}
#endif
```

**Phase 2 验证标准**:
- ✅ TransferStation A 区数据与 GPU `I_A_DATA`/`I_A_LABEL` 一致（fetch 前 10 个值对比）
- ✅ `ZERO_GRAD || FIRST_FWD_A` 两图并行执行，nsys/cudaProfiler 显示两条流同时活跃
- ✅ Loss 值非 NaN（fetch `loss` DTensor）
- ✅ 权重已更新（两次 fetch 同一权重对比，值有变化）

---

### Phase 3：AB 乒乓 + 完整重叠流水线

**目标**: 单 epoch 内所有 batch 完成，**核心重叠 DEEP_FWD_BWD || XFER(next) 必须生效**。

#### 改动 3.1：填充多 batch 循环

**文件**: `src/task/deep_learning_task.cpp`  
**函数**: `DeepLearningTask::run_train_epoch_gpu()`  
**位置**: Phase 2B 留空处

```cpp
// 替换原有的简单循环为完整重叠流水线
for (int batch = 0; batch < batches - 1; ++batch) {
    const bool current_from_a = (batch % 2 == 0);  // batch 0用A, batch 1用B, ...
    const int next_buf = current_from_a ? 1 : 0;   // 下一batch的缓冲

    // 取当前batch的图
    cudaGraphExec_t g_first_fwd = current_from_a ? g_first_fwd_a : g_first_fwd_b;
    cudaGraphExec_t g_deep      = current_from_a ? g_deep_a      : g_deep_b;

    // 取下一batch的传输图
    cudaGraphExec_t g_xfer_next = current_from_a ? g_xfer_b : g_xfer_a;

    // ---- 等待下一batch数据就绪（Producer-Consumer 同步点） ----
    while (!ts->buffer_is_readable(next_buf)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    // ---- 重叠 #1: ZERO_GRAD || FIRST_FWD ----
    cudaGraphLaunch(g_zero_grad,   s_update);
    cudaGraphLaunch(g_first_fwd,   s_comp1);
    cudaStreamSynchronize(s_comp1);
    cudaStreamSynchronize(s_comp2);
    cudaStreamSynchronize(s_comp3);
    cudaStreamSynchronize(s_update);

    // ---- 重叠 #2: DEEP_FWD_BWD || XFER(next)（核心） ----
    cudaGraphLaunch(g_deep,        s_comp1);   // 当前batch深层计算
    cudaGraphLaunch(g_xfer_next,   s_trans);   // 下一batch数据并行传输
    cudaStreamSynchronize(s_comp1);
    cudaStreamSynchronize(s_comp2);
    cudaStreamSynchronize(s_comp3);
    cudaStreamSynchronize(s_trans);

    // 标记下一batch缓冲区已消费（供Preprocessor复用）
    if (rank == 0) {
        ts->set_buffer_readable(next_buf, false);
        ts->set_buffer_writeable(next_buf, true);
    }

    // ---- 重叠 #4: FIRST_LAYER_BWD || DEEP_ALLREDUCE ----
    if (!frozen) {
        cudaGraphLaunch(g_first_bwd, s_comp1);
    }
    if (g_deep_ar) cudaGraphLaunch(g_deep_ar, s_update);
    cudaStreamSynchronize(s_comp1);
    cudaStreamSynchronize(s_comp2);
    cudaStreamSynchronize(s_comp3);
    cudaStreamSynchronize(s_update);

    // ---- ALLREDUCE + WEIGHT_UPDATE ----
    if (g_first_ar)  cudaGraphLaunch(g_first_ar,  s_update);
    if (g_weight_upd) cudaGraphLaunch(g_weight_upd, s_update);
    cudaStreamSynchronize(s_update);

    // ---- 学习率步进 ----
    if (rank == 0) scheduler_step();
}

// =============================================================
// Last batch（无传输）
// =============================================================
const bool last_from_a = ((batches - 1) % 2 == 0);
cudaGraphExec_t g_first_fwd_last = last_from_a ? g_first_fwd_a : g_first_fwd_b;
cudaGraphExec_t g_deep_last      = last_from_a ? g_deep_a      : g_deep_b;

// 重叠 #1: ZERO_GRAD || FIRST_FWD
cudaGraphLaunch(g_zero_grad,       s_update);
cudaGraphLaunch(g_first_fwd_last,  s_comp1);
cudaStreamSynchronize(s_comp1); cudaStreamSynchronize(s_comp2); cudaStreamSynchronize(s_comp3); cudaStreamSynchronize(s_update);

// DEEP（无XFER）
cudaGraphLaunch(g_deep_last, s_comp1);
cudaStreamSynchronize(s_comp1); cudaStreamSynchronize(s_comp2); cudaStreamSynchronize(s_comp3);

// FIRST_BWD + DEEP_ALLREDUCE
if (!frozen) cudaGraphLaunch(g_first_bwd, s_comp1);
if (g_deep_ar) cudaGraphLaunch(g_deep_ar, s_update);
cudaStreamSynchronize(s_comp1); cudaStreamSynchronize(s_comp2); cudaStreamSynchronize(s_comp3); cudaStreamSynchronize(s_update);

// ALLREDUCE + WEIGHT_UPDATE
if (g_first_ar)  cudaGraphLaunch(g_first_ar,  s_update);
if (g_weight_upd) cudaGraphLaunch(g_weight_upd, s_update);
cudaStreamSynchronize(s_update);

if (rank == 0) scheduler_step();
```

**Phase 3 验证标准**:
- ✅ 单 epoch 469 batches 全部执行，无 CUDA 错误
- ✅ nsys 或自定义计时显示 `DEEP_FWD_BWD` 与 `XFER` 存在时间重叠（GPU 利用率 > 90%）
- ✅ Loss 曲线单调下降，无 NaN
- ✅ A/B 乒乓无死锁（batch 间间隔稳定）

---

### Phase 4：完整训练 + 验证 epoch + 指标收集

**目标**: 20 epoch 完整训练，验证准确率 ≥ 98%，**重叠流水线在多 epoch 下稳定**。

#### 改动 4.1：实现 `run_val_epoch_gpu()`

**文件**: `src/task/deep_learning_task.cpp`  
**函数**: `DeepLearningTask::run_val_epoch_gpu()`  

```cpp
void DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
    auto& prep = Preprocessor::instance();
    const int val_batches = prep.val_steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    const int K = num_gpus_;
    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, val_batches, ts, validate_ema, &exc]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a = g_tab[static_cast<size_t>(GraphSlot::XFER_A)];
                auto g_xfer_b = g_tab[static_cast<size_t>(GraphSlot::XFER_B)];
                auto g_inf_a  = g_tab[static_cast<size_t>(validate_ema ? GraphSlot::INF_EMA_A : GraphSlot::INF_MAIN_A)];
                auto g_inf_b  = g_tab[static_cast<size_t>(validate_ema ? GraphSlot::INF_EMA_B : GraphSlot::INF_MAIN_B)];

                DeviceContext& ctx = context(rank);
                cudaStream_t s_trans = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_comp1 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));

                // 验证只用推理图，无反向传播
                for (int batch = 0; batch < val_batches; ++batch) {
                    const bool from_a = (batch % 2 == 0);
                    int buf = from_a ? 0 : 1;
                    while (!ts->buffer_is_readable(buf)) {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }

                    cudaGraphExec_t g_xfer = from_a ? g_xfer_a : g_xfer_b;
                    cudaGraphExec_t g_inf  = from_a ? g_inf_a  : g_inf_b;

                    // 验证阶段也可以重叠：INF || XFER(next)
                    cudaGraphLaunch(g_xfer, s_trans);
                    cudaGraphLaunch(g_inf,  s_comp1);
                    cudaStreamSynchronize(s_comp1);
                    cudaStreamSynchronize(s_trans);

                    if (rank == 0) {
                        ts->set_buffer_readable(buf, false);
                        ts->set_buffer_writeable(buf, true);
                    }
                }
            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }
    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank) {
        if (exc[rank]) std::rethrow_exception(exc[rank]);
    }
}
```

#### 改动 4.2：`run_gpu()` 中启动验证 Preprocessor

```cpp
if (should_validate_this_epoch()) {
    // 验证 epoch：Preprocessor 切到 val 模式
    std::thread prep_thread([&prep]() { prep.val(); });
    run_val_epoch_gpu(false);
    if (use_sema_) run_val_epoch_gpu(true);
    prep_thread.join();
}
```

#### 改动 4.3：指标收集

在 `run_val_epoch_gpu()` 中，每个 batch 后从 GPU fetch `loss` / `top1` / `top5` DTensor（这些 DTensor 在 `MemoryPlan` 中已分配，推理图应写入它们）。当前代码缺少指标聚合逻辑，可在 `run_gpu()` 中累加：

```cpp
float val_loss = 0.0f, top1 = 0.0f, top5 = 0.0f;
if (should_validate_this_epoch()) {
    // ... 启动验证 ...
    // 从 rank 0 fetch 指标
    val_loss = fetch_metric_from_rank(/* loss dtensor id */, 0);
    top1     = fetch_metric_from_rank(/* top1 dtensor id */, 0);
    top5     = fetch_metric_from_rank(/* top5 dtensor id */, 0);
}
```

**Phase 4 验证标准**:
- ✅ 20 epoch 完整训练无崩溃
- ✅ 单 epoch 469 batches，A/B 乒乓稳定
- ✅ 验证准确率 ≥ 98%
- ✅ nsys  profiling 显示 DEEP_FWD_BWD 与 XFER 存在显著时间重叠

---

## 五、辅助改动清单

### 5.1 学习率 H2D 传输

`scheduler.step()` 更新 CPU 侧 `current_lr_` 后，需要同步到 GPU 的 `S_SCALAR_FP32` DTensor。

**方案 A（简单）**: 在 `scheduler_step()` 后调用 `TaskBase::transfer(lr_tensor, lr_dt)`：
```cpp
void DeepLearningTask::scheduler_step() {
    std::visit([](auto&& s) { s.step(); }, sched_cfg_);
    float lr = get_current_lr();
    // 查找 MemoryPlan 中的 lr DTensor
    const DTensor& lr_dt = memory_plan_.get_lr_dtensor();  // 需要新增接口
    transfer(Tensor::scalar(lr), lr_dt);
}
```

**方案 B（CUDA Graph）**: 在 Compiler 中生成 `RANGE_H2D_COPY_DTENSOR` 节点放入 `CAST_AND_CHECK` 或独立 GraphId。但改动 Compiler 风险高，**建议 Phase 4 之前先用方案 A**。

### 5.2 空图安全跳过

单卡时 `FIRST_COMM` / `DEEP_COMM` 为空图（`nodes.empty()`），`resolve_gid` 返回 `nullptr`。`cudaGraphLaunch(nullptr, stream)` 是未定义行为，必须加保护：

```cpp
inline void safe_launch(cudaGraphExec_t g, cudaStream_t s) {
    if (g) cudaGraphLaunch(g, s);
}
```

所有 `cudaGraphLaunch` 调用替换为 `safe_launch`。

### 5.3 `DEEP_FWD_BWD` 的 A/B 区分

`GraphSlot::FWD_BWD_DEEP_A` 和 `FWD_BWD_DEEP_B` 当前都映射到同一个 `GraphId::DEEP_FWD_BWD`。这是正确的，因为 DEEP_FWD_BWD 图本身不区分 A/B，差异仅在于输入数据所在内存（`I_A_DATA` vs `I_B_DATA`）。但如果未来需要不同 shape（如 progressive resolution），则需为 A/B 分别捕获。

---

## 六、风险与缓解

| 风险 | 缓解 |
|------|------|
| `build_simple_atlas()` 修改影响 ResNet-50 图分解 | MLP 与 ResNet 共享同一 Atlas 构建逻辑，修改是通用的；ResNet 多 shape variant 已在 Atlas 的 `kMaxVariants` 中预留 |
| FIRST_FWD 与 ZERO_GRAD 并行时数据竞争 | ZERO_GRAD 清零梯度区域，FIRST_FWD 写特征区域，两者 Region 不同，无竞争 |
| DEEP_FWD_BWD 与 XFER 并行时 XFER 覆盖正在读取的数据 | XFER 写 `I_B_DATA`，DEEP 读 `I_A_DATA`，A/B 区物理分离，无竞争 |
| `cudaStreamSynchronize` 忙等待消耗 CPU | 使用 `cudaStreamSynchronize` 而非 spin-loop，CPU 会 yield；rank 线程数量 = GPU 数量，消耗可控 |
| Preprocessor `train()` 阻塞导致 `prep_thread.join()` 在 rank 线程完成后仍等待 | Preprocessor 的 `wait_all_transfer_stations_consumed()` 会等待深度学习引擎消费完所有数据，时序正确 |
| NCCL AllReduce 单卡报错 | `resolve_gid` 返回 `nullptr` 时 `safe_launch` 跳过，不会调用 NCCL |

---

## 七、实施顺序与验证检查点

```
Day 1:  Phase 1 — Atlas 映射修复 + init 实现
        └─ 检查点: compile() 后 gpu_exec_.graphs[rank][slot] != nullptr 对所有 slot

Day 2:  Phase 2 — Preprocessor 线程 + A 区传输 + ZERO_GRAD||FIRST_FWD 重叠
        └─ 检查点: fetch I_A_LABEL 前10个值与 TransferStation 原始数据一致
        └─ 检查点: nsys 显示 COMP_1 与 UPDATE 流并行

Day 3:  Phase 3 — AB 乒乓 + DEEP||XFER 核心重叠
        └─ 检查点: 单 epoch 469 batches 无崩溃
        └─ 检查点: nsys 显示 COMP_1 与 TRANS 流存在时间重叠

Day 4-5: Phase 4 — 完整训练 + 验证 epoch
        └─ 检查点: 20 epoch 训练完成，验证准确率 ≥ 98%
        └─ 检查点: Loss 曲线单调下降，无 NaN
```

---

## 八、关键代码引用汇总

| 文件 | 函数/行 | 说明 |
|------|---------|------|
| `src/task/task_base.cpp:616` | `build_simple_atlas()` | 需重写，提取 GraphId 子图 |
| `src/task/task_base.cpp:1200` | `init()` | 需实现真实初始化 |
| `src/task/task_base.cpp:1405` | `compile_alloc_hardware()` | 需加 ArenaKeeper memset |
| `src/task/deep_learning_task.cpp:32` | `GraphSlot` 枚举 | 需加 FIRST_FWD_A/B |
| `src/task/deep_learning_task.cpp:475` | `build_exec_table()` | 需改按 GraphId 解析 |
| `src/task/deep_learning_task.cpp:690` | `run_train_epoch_gpu()` | 需补全 5 张图调度 + 重叠 |
| `src/task/deep_learning_task.cpp:537` | `run_gpu()` | 需启动 Preprocessor 线程 |
| `src/task/deep_learning_task.cpp:845` | `run_val_epoch_gpu()` | 需实现验证推理 |

---

*本方案由 MLP.md + MMP2.md + 实际代码交叉验证后生成，所有重叠场景均基于 MMP0.md 伪代码和实际 CUDA Stream 架构。*
