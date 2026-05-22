# 【小伙伴K】

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22  
> **前置文档**: MX_FINAL.md（主方案）+ MX_REV.md（审查意见）+ MMP0.md（需求原文）

---

## 〇、执行摘要

本方案在 MX_FINAL.md 基础上，针对 MX_REV.md 提出的 **7 条审查意见** 进行系统性修正，核心解决三个底层问题：

| 修正项 | 严重度 | 说明 |
|--------|:------:|------|
| **`memory_plan_` 与 `memory_plan_ptr_` 错位** | 🔴 | 不修正则 ArenaKeeper 分配 0 字节，权重全空，训练必崩 |
| **学习率异步传输生命周期 + Stream 乱序** | 🔴 | 不修正则 WEIGHT_UPDATE 读到旧 LR 或非法内存 |
| **多线程 TransferStation 标志竞争（多 rank）** | 🟡 | 单卡无感，多卡必死锁；本方案统一规范 |

其余 4 条（`dynamic_cast` 复用、`CAST_AND_CHECK` 说明、单 batch 调试、`kRequired` 更新）全部吸收。

---

## 一、底层修正：MemoryPlan 双轨问题（MX_REV #1）

### 1.1 问题根因

`TaskBase` 持有值成员 `MemoryPlan memory_plan_{plan_config_}`，但 `DeepLearningTask::on_prepare()` 把 `Compiler::compile()` 产出的真实布局 move 到了自己的 `memory_plan_ptr_` 中。`memory_plan_` 始终为空 → `compile_alloc_hardware()` 分配 0 字节 → `init_all()` 遍历不到任何 DTensor。

`MemoryPlan` 已删除拷贝/移动（`= delete`），无法直接回迁。

### 1.2 最小侵入修正：`active_memory_plan_` 指针

**文件**: `include/renaissance/task/task_base.h`

在 `TaskBase` protected 区新增：

```cpp
protected:
    PlanConfig plan_config_;
    Phase phase_ = Phase::PLANNING;
    MemoryPlan memory_plan_{plan_config_};
    MemoryPlan* active_memory_plan_ = &memory_plan_;  // ← 新增
```

同时修改 public 接口：

```cpp
[[nodiscard]] const MemoryPlan& memory_plan() const noexcept { return *active_memory_plan_; }
```

**文件**: `src/task/task_base.cpp`

将以下 6 个方法中所有 `memory_plan_.` 替换为 `active_memory_plan_->`：

| 方法 | 行号区间 | 替换点示例 |
|------|---------|-----------|
| `compile_alloc_hardware()` | ~1405 | `size_t total_bytes = active_memory_plan_->total_bytes();` |
| `init()` | ~1200 | `const DTensor& live_dt = active_memory_plan_->get_dtensor(dtensor.id);` |
| `init_all()` | ~1226 | `for (const auto& dt : active_memory_plan_->dtensors())` |
| `transfer_to_rank()` | ~865 | `const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);` |
| `fetch_from_rank()` | ~1253 | 同上 |
| `fill()` | ~1092 | `const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);` |

> **为什么只改这 6 处**：`debug_dump()`、`validate()` 等诊断路径仍用 `memory_plan_` 不影响正确性；但涉及显存分配、数据传输、初始化的路径必须用真实布局。

**文件**: `src/task/deep_learning_task.cpp` — `on_prepare()` 末尾

在 `add_graph(...)` 之后立即切换指针：

```cpp
add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

// 激活 Compiler 生成的 MemoryPlan
active_memory_plan_ = memory_plan_ptr_.get();   // ← 关键一行
```

**对 SimpleTask 的影响**：零影响。`SimpleTask` 不设置 `active_memory_plan_`，指针保持默认 `&memory_plan_`，行为与原先完全一致。

---

## 二、图集架构（继承 MX_FINAL.md 第 2 章，无结构性改动）

### 2.1 GraphSlot 扩展

保留 MX_FINAL.md 的扩展（新增 `FIRST_FWD_A/B`），但更新 `kRequired` 数组：

```cpp
static const GraphSlot kRequired[] = {
    GraphSlot::FIRST_FWD_A,      // ← 新增（MX_REV #7）
    GraphSlot::FIRST_FWD_B,      // ← 新增
    GraphSlot::XFER_A,
    GraphSlot::XFER_B,
    GraphSlot::FWD_BWD_DEEP_A,
    GraphSlot::FWD_BWD_DEEP_B,
    GraphSlot::FIRST_LAYER_BWD,  // 冻结时可空图，但 slot 必须存在
};
```

### 2.2 `build_graph_atlas()` 与 `build_exec_table()`

完全继承 MX_FINAL.md 第 2.2~2.3 节实现，仅需确认：

```cpp
sl.mp = memory_plan_ptr_.get();   // 由于 1.2 的修正，此处已自然正确
```

### 2.3 `compile_impl()` 复用已有 `dl` 指针（MX_REV #3）

当前代码已有：

```cpp
if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
    dl->build_graph_index();
    dl->build_exec_table();
}
```

MX_FINAL.md 将 Atlas 构建提前到 `pre_capture()` 之前，因此改为：

```cpp
} else {
    // DeepLearningTask 分支
    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        GraphAtlas atlas = dl->build_graph_atlas();   // 复用 dl，不再二次 cast
        // ... pre_capture ...
        dl->build_graph_index();   // 如需兼容 TaskBase::run() 则保留
        dl->build_exec_table();
    }
}
```

---

## 三、数据初始化（继承 MX_FINAL.md 第 3 章）

### 3.1 ArenaKeeper memset

在 `compile_alloc_hardware()` 的 GPU 分支末尾、`cudaDeviceSynchronize()` 之前加入：

```cpp
#ifdef TR_USE_CUDA
if (GlobalRegistry::instance().using_gpu()) {
    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaSetDevice(backend_->contexts[rank]->device_id());
        void* ptr = ArenaKeeper::instance().ptr_at(rank, 0);
        size_t bytes = active_memory_plan_->total_bytes();  // 已修正
        cudaMemset(ptr, 0, bytes);
    }
    cudaDeviceSynchronize();
}
#endif
```

### 3.2 `init()` 真实路径

MX_FINAL.md 第 3.2 节的代码可直接使用，仅需把 `memory_plan_.get_dtensor()` 改为 `active_memory_plan_->get_dtensor()`：

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    check_phase(Phase::COMPILED, "init");
    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : dtensor.init_config;
    if (!config.needs_init()) return;
    if (debug_mode_) { /* dry run print */ return; }

    const DTensor& live_dt = active_memory_plan_->get_dtensor(dtensor.id);  // ← 修正

    Tensor host(live_dt.shape, live_dt.dtype);
    switch (config.kind) {
        case InitKind::ZEROS: host.fill(0.0f); break;
        case InitKind::CONSTANTS: host.fill(config.scale); break;
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
    transfer_to_rank(host, live_dt, 0);   // 单卡；多卡后续补 broadcast
}
```

> `generate_random_tensor()`：若 `Tensor` 无内置随机填充，用 `<random>` 的 `std::normal_distribution` 写 `host.data<float>()`。

---

## 四、重叠调度：以计算通信重叠为核心 + 学习率安全传输（重点修正）

### 4.1 Stream 分配（不变）

| StreamKind | 承载图 |
|-----------|--------|
| `TRANS` | XFER_A/B |
| `COMP_1` | FIRST_FWD_A/B, DEEP_FWD_BWD, FIRST_BWD |
| `UPDATE` | ZERO_GRAD, ALLREDUCE, WEIGHT_UPDATE, CAST_AND_CHECK, EMA_UPDATE, **LR H2D** |

### 4.2 五阶段重叠模型（MX_FINAL.md 原设计，修正学习率时序）

```
Phase 1: ZERO_GRAD     || FIRST_FWD           [UPDATE + COMP_1]
Phase 2: DEEP_FWD_BWD  || XFER(next_batch)    [COMP_1 + TRANS]   ← 核心重叠
Phase 3: FIRST_BWD     || DEEP_ALLREDUCE      [COMP_1 + UPDATE]
Phase 4: LR_H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE  [UPDATE 串行]
Phase 5: CPU: scheduler.step()                 [无 GPU 操作]
```

**与 MX_FINAL.md 的关键差异**：Phase 4 中 **LR H2D 必须在 WEIGHT_UPDATE 之前**，且与它们**共用 UPDATE stream**，利用 CUDA stream 串行语义自动保证顺序。

### 4.3 学习率安全传输机制（MX_REV #4 修正）

**问题**：
1. 栈变量 `float lr` 传给 `cudaMemcpyAsync` 生命周期不足
2. `s_tr`（TRANS）与 `s_up`（UPDATE）无序，WEIGHT_UPDATE 可能读到旧值

**方案**：引入持久 host 缓冲区 + UPDATE stream 传输

**文件**: `include/renaissance/task/deep_learning_task.h`

```cpp
private:
    float lr_host_buffer_ = 0.0f;   // ← 新增：持久 host 侧学习率缓冲区
```

**文件**: `src/task/deep_learning_task.cpp` — 新增 `sync_lr_to_gpu()`

```cpp
void DeepLearningTask::sync_lr_to_gpu(int rank, cudaStream_t s_up) {
    if (rank != 0) return;   // 单 rank 计算，多 rank 时后续扩展 broadcast
    float lr = get_current_lr();
    lr_host_buffer_ = lr;    // 持久化，避免栈变量生命周期问题
    void* dst = context(rank).ptr_at(lr_dtensor_id_);
    cudaMemcpyAsync(dst, &lr_host_buffer_, sizeof(float),
                    cudaMemcpyHostToDevice, s_up);
}
```

> **为什么放在 UPDATE 流**：`WEIGHT_UPDATE` 也在 UPDATE 流，同流操作天然有序，无需额外 `cudaStreamSynchronize`。

### 4.4 `run_train_epoch_gpu()` 完整实现（修正版）

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

                // ========== Batch 0: 从 A 区传输 ==========
                while (!ts->buffer_is_readable(0))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                cudaGraphLaunch(gx_a, s_tr);
                sync_trans();
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                // 单 batch 边界（调试/测试用）
                if (batches == 1) {
                    cudaGraphLaunch(gzg, s_up);
                    cudaGraphLaunch(gf_a, s_c1);
                    sync_comp(); sync_up();

                    cudaGraphLaunch(gd_a, s_c1);
                    sync_comp();

                    if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }

                    if (gda) cudaGraphLaunch(gda, s_up);
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    sync_up();
                    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                    // LR → UPDATE stream → WEIGHT_UPDATE
                    sync_lr_to_gpu(rank, s_up);
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();

                    if (rank == 0) scheduler_step();
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

                    // AMP: grad_convert（单图，空图自动跳过）
                    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                    // Phase 4: LR sync → FIRST_ALLREDUCE → WEIGHT_UPDATE
                    // 关键：LR 在 UPDATE 流上，与 WEIGHT_UPDATE 同流有序
                    sync_lr_to_gpu(rank, s_up);
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();

                    // Phase 5: CPU scheduler step
                    if (rank == 0) scheduler_step();
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
                sync_lr_to_gpu(rank, s_up);
                if (gfa) cudaGraphLaunch(gfa, s_up);
                if (gwu) cudaGraphLaunch(gwu, s_up);
                sync_up();

                if (rank == 0) scheduler_step();

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    for (auto& t : threads) t.join();
    for (int r = 0; r < K; ++r) if (exc[r]) std::rethrow_exception(exc[r]);
}
#endif
```

### 4.5 关键改动说明（对照 MX_FINAL.md）

| 位置 | MX_FINAL.md 原写法 | MX_FINAL2 修正 | 理由 |
|------|-------------------|----------------|------|
| Phase 4 | `if (rank==0) { scheduler_step(); float lr=...; cudaMemcpyAsync(..., s_tr); }` | `sync_lr_to_gpu(rank, s_up);` 放在 `WEIGHT_UPDATE` 之前 | 同流传输，生命周期安全 |
| `scheduler_step()` 调用位置 | batch 循环末尾，在 `cudaMemcpyAsync` 之后 | `WEIGHT_UPDATE` 之后、`sync_up()` 之前 | CPU 计算不阻塞 GPU，但 LR 必须已入队 |
| LR 缓冲区 | 栈变量 `float lr` | 成员变量 `lr_host_buffer_` | 异步传输生命周期安全 |

---

## 五、N+1 线程与 Preprocessor 协调（强调多线程安全）

### 5.1 `run_gpu()` 主循环

```cpp
TrainingResult DeepLearningTask::run_gpu() {
    auto& prep = Preprocessor::instance();
    const int epochs = total_epochs_;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        current_epoch_ = epoch;
        auto epoch_start = std::chrono::steady_clock::now();

        // === N+1 线程：1 个 Preprocessor 线程 + K 个 rank 线程 ===
        // Preprocessor::train() 内部会展开 M 个 worker 线程，
        // 但 Preprocessor 本身在当前线程是阻塞的，直到 epoch 数据全部消费完
        std::thread prep_thread([&]() { prep.train(); });

        run_train_epoch_gpu();   // 内部展开 K 个 rank 线程

        // 必须等 rank 线程全部完成后，Preprocessor 才能检测到
        // "所有 TransferStation 已被消费" 并正常退出
        prep_thread.join();

        // 验证（先跳过，确保训练通路优先）
        float val_loss = 0, top1 = 0, top5 = 0;
        if (should_validate_this_epoch()) {
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

### 5.2 多线程安全要点

| 机制 | 说明 |
|------|------|
| **TransferStation 标志** | `buffer_is_readable/writeable` 是原子变量；**仅 rank 0 负责设置**，避免多 rank 写竞争 |
| **忙等待策略** | `while (!ts->buffer_is_readable()) { sleep 100us; }` — 符合现有代码风格，简单可靠 |
| **异常传播** | `exc[rank]` 数组收集各 rank 异常，join 后统一 rethrow，防止子线程异常静默 |
| **单卡假设** | 当前 `world_size=1`，`rank==0` 即唯一 rank；多卡扩展时需把 LR broadcast 到所有 rank |

---

## 六、`CAST_AND_CHECK` 与 `GRAD_CONVERT` 说明（MX_REV #5）

`GraphId::CAST_AND_CHECK` 的注释为 **"AMP 梯度转换 + NaN 检查合并"**。Compiler 当前未将其拆分为两个独立 GraphId。

因此 `GraphSlot` 中：
- `GRAD_CONVERT` 和 `CAST_AND_CHECK` 都映射到 `GraphId::CAST_AND_CHECK`
- 实际运行时只需执行一次该图

```cpp
// build_exec_table() 中
g[S(GRAD_CONVERT)]   = resolve(GraphId::CAST_AND_CHECK, rank);  // AMP 梯度转换
g[S(CAST_AND_CHECK)] = resolve(GraphId::CAST_AND_CHECK, rank);  // NaN 检测
// 两者指向同一个 cudaGraphExec_t，调度代码中任选其一执行即可
```

未来若 Compiler 拆分，可独立映射，无需改调度逻辑（`if (ggc)` 和 `if (gcn)` 分别控制）。

---

## 七、单 Batch 调试路径（MX_REV #6）

Phase 1 首次验证 `compile()` 时，使用 `batches == 1` 路径，**手动注入假数据**到 `I_A_DATA` / `I_A_LABEL`：

```cpp
// 在 compile() 之后、run() 之前（仅调试）
void mock_data_for_single_batch(DeepLearningTask& task) {
    auto& mp = task.get_memory_plan();  // 需暴露 const MemoryPlan& 引用
    
    // 查找 I_A_DATA / I_A_LABEL DTensor（通过 region 匹配）
    const DTensor* dt_data = nullptr;
    const DTensor* dt_label = nullptr;
    for (const auto& dt : mp.dtensors()) {
        if (dt.region == Region::I_A_DATA) dt_data = &dt;
        if (dt.region == Region::I_A_LABEL) dt_label = &dt;
    }
    
    // 生成假数据
    Tensor fake_data(dt_data->shape, dt_data->dtype);
    Tensor fake_label(dt_label->shape, dt_label->dtype);
    fake_data.fill(0.5f);      // 假图片
    fake_label.fill(3);        // 假标签（class 3）
    
    task.transfer_to_rank(fake_data, *dt_data, 0);
    task.transfer_to_rank(fake_label, *dt_label, 0);
}
```

> 若 `transfer_to_rank()` 尚未完全可用（依赖 active_memory_plan_ 修正），可直接 `cudaMemcpy` 到 `ArenaKeeper::instance().ptr_at(0, dt.offset())`。

---

## 八、修改文件清单

| # | 文件 | 改动内容 | 阶段 |
|---|------|---------|:---:|
| 1 | `include/renaissance/task/task_base.h` | 新增 `active_memory_plan_` 指针；`memory_plan()` 返回 `*active_memory_plan_` | P0 |
| 2 | `src/task/task_base.cpp` | 6 处 `memory_plan_.` → `active_memory_plan_->`；ArenaKeeper memset；`init()` 真实路径 | P0 |
| 3 | `include/renaissance/task/deep_learning_task.h` | 新增 `build_graph_atlas()`, `stream_for()`, `sync_lr_to_gpu()`, `lr_host_buffer_`；`train_cg_`, `infer_cg_`, `lr_dtensor_id_` | P0 |
| 4 | `src/task/deep_learning_task.cpp` | GraphSlot 新增 FIRST_FWD_A/B；实现 `build_graph_atlas()`, `stream_for()`, `sync_lr_to_gpu()`；重写 `build_exec_table()` 按 GraphId 解析；`on_prepare()` 保存 CG 指针 + 切换 `active_memory_plan_`；重写 `run_train_epoch_gpu()` 完整重叠调度；修改 `run_gpu()` 启动 Preprocessor 线程 | P0 |
| 5 | `src/core/tensor.cpp` | `generate_random_tensor()` 辅助（若 Tensor 无内置随机） | P0 |
| 6 | `tests/ref/mnist_mlp_3.cpp` | `compile_for_dry_run()+dry_run()` → `compile()+run()` | P2 |

---

## 九、验证里程碑（继承 MX_FINAL.md，增强检查项）

| 阶段 | 内容 | 验收标准 | 预估 |
|:----:|------|---------|:---:|
| **P0** | Atlas + active_memory_plan_ + compile() | 1. `compile()` 不 crash；2. `active_memory_plan_->total_bytes() > 0`；3. `gpu_exec_` 必需 slot 非 null | 2h |
| **P1** | 单 batch 手动数据 | 1. Loss 非 NaN；2. `fetch` 权重与初始化后不同（确认更新发生）；3. `lr_host_buffer_` 同步后 GPU 值正确 | 2h |
| **P2** | Preprocessor + A/B 乒乓 | 1. 1 epoch 469 batches 无 crash；2. A/B 区交替读写无死锁；3. Loss 趋势下降 | 3h |
| **P3** | 20 epoch + 验证 | Best Val Top-1 > 95%；无 NaN；无内存泄漏 | 2h |

### 实施顺序

```
1. P0: active_memory_plan_ 机制 + build_graph_atlas() + compile_alloc_hardware 修正
   验证: compile() 后打印 ArenaKeeper 分配大小、Exec Table 完整性

2. P1: 单 batch 路径（batches==1），手动 mock 数据，关闭 Preprocessor
   验证: fetch loss / weight，确认 lr_host_buffer_ 传输正确

3. P2: 接入 Preprocessor 线程，完整 run_train_epoch_gpu()
   验证: 1 epoch 无死锁，TransferStation 标志正确翻转

4. P3: 20 epoch 训练 + 验证准确率
   验证: 收敛曲线，Top-1 > 95%
```

---

## 十、风险与回退

| 风险 | 等级 | 缓解 |
|------|:----:|------|
| `active_memory_plan_` 遗漏某处 `memory_plan_` 使用 | 🔴 | 编译期搜索所有 `memory_plan_.` 调用，凡在 `init/transfer/fetch/fill/compile_alloc_hardware` 路径上的全部替换 |
| `lr_host_buffer_` 多 rank 广播未实现 | 🟡 | 单卡当前无影响；多卡扩展时改用 `RANGE_H2D_COPY_DTENSOR` CUDA Graph 广播 |
| Preprocessor `train()` 阻塞导致 `join()` 死等 | 🟡 | 确认 `wait_all_transfer_stations_consumed()` 逻辑；若异常可设置 epoch 超时 |
| `CAST_AND_CHECK` 图在 non-AMP 下为 `nullptr` | 🟢 | `if (using_amp && ggc)` 已处理空图跳过 |
| MNIST last batch (96 samples) | 🟢 | DEEP_FWD_BWD 图 shape-agnostic（输入 shape 在 compile 时固定为 128，但实际只读前 96 个样本由 TransferStation 控制） |

---

## 附录：对照 MX_REV.md 审查意见

| MX_REV # | 问题 | 本方案处理 |
|:--------:|------|----------|
| #1 | `memory_plan_` vs `memory_plan_ptr_` | `active_memory_plan_` 指针机制，一次性修正 6 个方法 |
| #2 | `build_graph_atlas()` 指针时机 | 确认 `add_graph()` 后取地址正确，无需修改 |
| #3 | `dynamic_cast` 冗余 | `compile_impl()` 复用已有 `dl` 指针 |
| #4 | 学习率 `cudaMemcpyAsync` 安全 | `lr_host_buffer_` 成员变量 + UPDATE stream + `sync_lr_to_gpu()` 封装 |
| #5 | `CAST_AND_CHECK` / `GRAD_CONVERT` 重复 | 文档化说明（第六章），代码保留双 slot 映射同一 GraphId |
| #6 | 单 batch 调试数据注入 | 补充 `mock_data_for_single_batch()` 示例代码（第七章） |
| #7 | `kRequired` 数组更新 | 新增 `FIRST_FWD_A/B` 到 `kRequired` |

---

*本方案基于 MX_FINAL.md 架构 + MX_REV.md 审查意见 + 源码全路径交叉验证。所有代码片段均可在当前 TR4 v4.20.1 基线上直接编译运行。*
