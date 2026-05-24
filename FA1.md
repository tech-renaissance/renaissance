# 仅测量 H2D 传输的实现方案

## 1. 需求摘要

为目标函数 `DeepLearningTask` 添加两个方法 + 一个测试样例：

| 方法 | 功能 |
|------|------|
| `compile_h2d_only()` | 只编译/捕获 `XFER_A` 和 `XFER_B` 两个 H2D 传输图（含 last batch），不捕获验证/推理图 |
| `run_h2d_only()` | 启动 Preprocessor::train()，只交替运行 XFER_A / XFER_B，一个 epoch，计时 |

测试基于 `test_pw_ultimate.cpp` 的 PREPROCESSOR_SETTING 配置方式，但引入 DeepLearningTask。

---

## 2. 当前架构分析

### 2.1 完整 compile 流程

```
task.compile()
  └─ TaskBase::compile_impl()
       ├─ compile_freeze_global()           # 锁定 GlobalRegistry
       ├─ compile_invoke_on_prepare()       # → on_prepare()
       │    └─ Compiler::compile(plan, spec)  # 生成 MemoryPlan + ComputationGraphs (含 TRANSFER_A/B)
       │         result.train_cg  ← 含 TRANSFER_A, TRANSFER_B, DEEP_FWD_BWD, FIRST_BWD, ...
       │         result.infer_cg  ← 含 INF_MAIN_A, INF_MAIN_B, ...
       ├─ compile_alloc_hardware()           # DeviceContext + NCCL + StagingBufferPool
       └─ [DeepLearningTask 分支]
            ├─ build_graph_atlas()           # GraphId → GraphSlot 映射
            ├─ pre_capture(atlas, ctxs)      # 捕获 cudaGraph
            └─ build_exec_table()            # GraphSlot → cudaGraphExec_t 查找表
```

### 2.2 关键数据结构

**GraphSlot** (deep_learning_task.cpp:34-55) — GPU 执行时的槽位枚举：
```cpp
enum class GraphSlot : uint8_t {
    XFER_A = 0,        // ← 我们需要
    XFER_B,            // ← 我们需要
    FWD_BWD_DEEP_A,    // ✗ 不需要
    FWD_BWD_DEEP_B,    // ✗ 不需要
    ...                // 等等
};
```

**gpu_exec_.graphs[rank]** — `std::vector<cudaGraphExec_t>`，按 GraphSlot 索引。

**build_graph_atlas()** (deep_learning_task.cpp:480-519) — 遍历 `train_cg_` 和 `infer_cg_` 中所有 GraphId 生成 GraphAtlas。GraphAtlas 决定 pre_capture 捕获哪些图。

**build_exec_table()** (deep_learning_task.cpp:558-641) — 从 `captured_result_` 中 resolve 每个 GraphSlot 对应的 cudaGraphExec_t。

### 2.3 当前 run_train_epoch_gpu 的 AB 交替逻辑

```python
# 伪代码
batches = prep.steps_per_epoch()

# Batch 0 预传输
wait(buf0 readable) → XFER_A → sync

# 循环 batch = 0 .. batches-2
for batch in range(batches - 1):
    from_a = (batch % 2 == 0)
    next_buf = 1 if from_a else 0

    # Phase 1: ZERO_GRAD + FIRST_FWD
    # Phase 2: DEEP_FWD_BWD + XFER(next)     ← 传输和计算重叠
    # Phase 3: FIRST_BWD + DEEP_ALLREDUCE
    # Phase 4: WEIGHT_UPDATE

    ts->set_buffer_readable(next_buf, false)
    ts->set_buffer_writeable(next_buf, true)

# Last batch
# Phase 1-4 without XFER(next)
```

### 2.4 run_train_epoch_gpu 的和弦逻辑总结

核心思想是**流水线重叠**：当一个 batch 在 GPU 上做正向反向计算时，下一个 batch 的 H2D 数据已经在传输。

H2D 传输图分为 A 区和 B 区：
- `XFER_A`: 将 CPU staging buffer 的 zone-A 数据传输到 GPU 显存的 I_A_LABEL + I_A_DATA
- `XFER_B`: 将 CPU staging buffer 的 zone-B 数据传输到 GPU 显存的 I_B_LABEL + I_B_DATA

由于 Forward/BWD 不需要 AB 两区，只需要一个区用于计算，图执行 per batch 从两个槽位中选一个：
- A: XFER + FWD_BWD_DEEP_A + FIRST_LAYER_BWD
- B: XFER + FWD_BWD_DEEP_B + FIRST_LAYER_BWD_B

---

## 3. 实现方案

### 3.1 compile_h2d_only()

**策略：利用现有 compile 基础设施 + flags 跳步**

由于 `on_prepare()` → `Compiler::compile()` 生成 MemoryPlan 是必须的（需要 DTensor 分配区域 I_A_LABEL/I_A_DATA/I_B_LABEL/I_B_DATA），我们不能跳过。但可以：

1. 调用 TaskBase::compile_impl() → 生成完整的 MemoryPlan + ComputationGraphs
2. 在 build_graph_atlas / build_exec_table 阶段，只处理 XFER_A 和 XFER_B

**实现细节：**

```cpp
// deep_learning_task.h - 新增成员
bool compile_h2d_only_flag_ = false;

// deep_learning_task.h - 新增公开方法
void compile_h2d_only();

// deep_learning_task.h - 新增私有方法
void build_exec_table_h2d_only();
```

```cpp
// deep_learning_task.cpp
void DeepLearningTask::compile_h2d_only() {
    compile_h2d_only_flag_ = true;
    // 复用 TaskBase 的完整 compile 流程（会调 on_prepare → MemoryPlan → 硬件分配）
    // 但 build_graph_atlas 和 build_exec_table 被修改为只处理 H2D 图
    debug_mode_ = false;
    // 注意：不能直接调 compile() 因为 build_exec_table 会校验 deep 图为非空
    // 需要改 compile_impl 或通过虚拟方法介入
    TaskBase::compile();
    compile_h2d_only_flag_ = false;
}
```

**修改 build_graph_atlas()** — 当 `compile_h2d_only_flag_` 为 true 时，只包含 TRANSFER_A 和 TRANSFER_B：

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;
    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (train_cg_->nodes(gid).empty()) continue;
            
            // ═══ H2D-Only 模式：只添加 TRANSFER_A 和 TRANSFER_B ═══
            if (compile_h2d_only_flag_) {
                if (gid != GraphId::TRANSFER_A && gid != GraphId::TRANSFER_B)
                    continue;
            }
            
            auto& sl = atlas.slot(0, gi);
            sl.cg = train_cg_;
            sl.mp = active_memory_plan_;
            sl.stream_kind = stream_for(gid);
            sl.shape_id = kShapeInvariant;
        }
    }
    // H2D-Only 模式不添加推理图
    if (!compile_h2d_only_flag_ && infer_cg_) {
        // ... 推理图逻辑不变
    }
    return atlas;
}
```

**修改 build_exec_table()** — 当 `compile_h2d_only_flag_` 为 true 时，跳过 deep/bwd 的非空校验：

```cpp
void DeepLearningTask::build_exec_table() {
    // ... 前面解析 cudaGraphExec_t 的逻辑不变 ...
    
    // 仅 h2d_only 模式校验
    if (compile_h2d_only_flag_) {
        static const GraphSlot kH2DRequired[] = {
            GraphSlot::XFER_A, GraphSlot::XFER_B,
        };
        for (int rank = 0; rank < K; ++rank) {
            for (auto slot : kH2DRequired) {
                TR_CHECK(gpu_exec_.graphs[rank][static_cast<size_t>(slot)],
                         ValueError, "H2D graph slot is nullptr");
            }
        }
    } else {
        // 原有校验逻辑不变
        static const GraphSlot kRequired[] = { ... };
    }
}
```

### 3.2 run_h2d_only()

**任务：** 启动 Preprocessor::train() → 交替 launch XFER_A / XFER_B → 消费所有 batch → 计时

**核心逻辑（简化版的 AB 交替，去掉所有计算/优化器步骤）：**

```cpp
// deep_learning_task.h
H2DPerfResult run_h2d_only();

// deep_learning_task.cpp
H2DPerfResult DeepLearningTask::run_h2d_only() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(
        registry.transfer_station_ptr(0));
    const int K = num_gpus_;

    // 启动 Preprocessor 线程
    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    const auto t_start = std::chrono::steady_clock::now();

    // 启动 K 个 rank 线程
    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            cudaSetDevice(gpu_exec_.device_ids[rank]);
            auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
            const auto& g = gpu_exec_.graphs[rank];
            auto g_xfer_a = g[S(GraphSlot::XFER_A)];
            auto g_xfer_b = g[S(GraphSlot::XFER_B)];
            cudaStream_t s_trans = static_cast<cudaStream_t>(
                context(rank).stream(StreamKind::TRANS));
            auto sync_tr = [&]() { cudaStreamSynchronize(s_trans); };

            // ── Batch 0 预传输 (A 区) ──
            while (!ts->buffer_is_readable(0))
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            cudaGraphLaunch(g_xfer_a, s_trans);
            sync_tr();
            if (rank == 0) {
                ts->set_buffer_readable(0, false);
                ts->set_buffer_writeable(0, true);
            }

            // 边界：只有 1 个 batch
            if (batches == 1) return;

            // ── 主循环：batch 0 .. batches-2 ──
            // 每轮传输下一 batch 的数据
            for (int batch = 0; batch < batches - 1; ++batch) {
                bool from_a = (batch % 2 == 0);
                int next_buf = from_a ? 1 : 0;
                auto g_xfer_next = from_a ? g_xfer_b : g_xfer_a;

                // 等 Preprocessor 填好下一个 buffer
                while (!ts->buffer_is_readable(next_buf))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));

                // 启动传输
                cudaGraphLaunch(g_xfer_next, s_trans);
                sync_tr();

                // rank 0 标记 buffer 已消费，Preprocessor 可继续填充
                if (rank == 0) {
                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);
                }
            }
            // 循环结束后，所有 batches 的数据已传输完毕
            // 不需要处理 last batch（它已在循环中被传输）
        });
    }

    for (auto& t : threads) t.join();
    const auto t_end = std::chrono::steady_clock::now();

    // 等待 Preprocessor 完成（带 drain）
    // Preprocessor 可能在最后一个 batch 后还有一次写 buffer 但没人消费
    // 需要 drain 剩余 buffer 让它自然结束
    // ... drain 逻辑 ...

    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);

    // 计算统计
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    size_t total_bytes = ...;
    return { elapsed_sec, batches, total_bytes };
}
```

**A/B交替逻辑图解：**

```
Preprocessor                GPU (H2D-only)
    │                           │
    ├─ fill buf0 ──────────────►│ wait buf0 → XFER_A → sync → set_writeable(buf0)
    │                           │
    ├─ fill buf1 ──────────────►│ wait buf1 → XFER_B → sync → set_writeable(buf1)
    │                           │
    ├─ fill buf0 ──────────────►│ wait buf0 → XFER_A → sync → set_writeable(buf0)
    │   (batch 2)               │
    ├─ fill buf1 ──────────────►│ wait buf1 → XFER_B → sync → set_writeable(buf1)
    │   (batch 3)               │
    │   ...                     │   ...
    │                           │
    ├─ fill buf{n%2} (last) ───►│ last: 消耗并 set_writeable
    │                           │
    ▼ 完成                      ▼ 完成
```

**Last batch 处理：**
- `batches` 为奇数 → last batch 落在 buf0 → 已在循环最后被传为 `XFER_A`
- `batches` 为偶数 → last batch 落在 buf1 → 已在循环最后被传为 `XFER_B`
- **无需额外处理**！循环 `batch=0..batches-2` 覆盖了除 batch 0 外的所有传输

**Preprocessor drain：**
- 循环结束后，所有 batch 都已被传输且 buffer 标记为 writeable
- Preprocessor 会检测到数据集已遍历完毕 → `prep.train()` 自然返回
- 不需要额外 drain 循环

### 3.3 H2DPerfResult 结构体

```cpp
struct H2DPerfResult {
    double elapsed_seconds;      // wall-clock 时间
    int total_batches;           // 当前 epoch 的 batch 数
    size_t total_bytes_per_rank; // 单卡传输的总字节数
    double samples_per_second;   // 吞吐量
    double bandwidth_gb_s;       // 有效带宽
};
```

---

## 4. 测试样例 design

**文件：** `tests/correction/test_dl_h2d_perf.cpp`

**设计：**
- 基于 test_pw_ultimate.cpp 的 Preprocessor 配置方式（PREPROCESSOR_SETTING 链式调用）
- 引入 DeepLearningTask，配置一个最简单的 BluePrint（一个 FC 层）
- 调用 compile_h2d_only() 和 run_h2d_only()
- 输出计时和吞吐量

```cpp
// test_dl_h2d_perf.cpp 核心结构

int main(int argc, char* argv[]) {
    // 1. 解析 CLI 参数（同 test_pw_ultimate 风格）
    //    --dataset, --path, --bs, --resolution, --amp, --gpu-ids, ...

    // 2. 配置 GLOBAL_SETTING
    GLOBAL_SETTING.use_gpu(gpu_ids_str).local_batch_size(bs).auto_seed();
    if (use_amp) GLOBAL_SETTING.amp(true);

    // 3. 配置 PREPROCESSOR_SETTING（同 test_pw_ultimate）
    PREPROCESSOR_SETTING
        .dataset(dataset_type, dataset_path)
        .load_workers(n_load)
        .preprocess_workers(n_prep)
        .normalization(norm)
        .train_transforms(...)
        .val_transforms(...)
        .commit();

    // 4. 配置 DeepLearningTask
    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)));  // 最简 BluePrint
    task.loss(CrossEntropyLoss());
    task.optimizer(SGD().momentum(0.9f).weight_decay(5e-4f));
    task.num_classes(num_classes);
    task.total_epochs(1);

    // 5. 只编译 H2D 图
    task.compile_h2d_only();

    // 6. 只运行 H2D 传输
    auto result = task.run_h2d_only();

    // 7. 输出结果
    std::cout << "Elapsed: " << result.elapsed_seconds << " s\n";
    std::cout << "Batches: " << result.total_batches << "\n";
    std::cout << "Throughput: " << result.samples_per_second << " samples/s\n";
    std::cout << "Bandwidth: " << result.bandwidth_gb_s << " GB/s\n";
}
```

---

## 5. 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `include/renaissance/task/deep_learning_task.h` | 新增 `H2DPerfResult` 结构体；新增 `compile_h2d_only_flag_` 成员；新增 `compile_h2d_only()` 和 `run_h2d_only()` 公开方法声明；新增私有 helper 声明 |
| `src/task/deep_learning_task.cpp` | 实现 `compile_h2d_only()`、修改 `build_graph_atlas()` 添加 H2D-only 分支、修改 `build_exec_table()` 添加 H2D-only 校验分支、实现 `run_h2d_only()` |
| `tests/correction/test_dl_h2d_perf.cpp` | **新建** — H2D 传输性能测试 |
| `tests/correction/CMakeLists.txt` | 新增 `test_dl_h2d_perf` 编译目标 |

---

## 6. 风险与注意点

1. **compile_impl 中的 build_exec_table 校验**：当前校验要求 FWD_BWD_DEEP_A、FIRST_LAYER_BWD 等必须非空。H2D-only 模式下需要绕过，可通过 `compile_h2d_only_flag_` 条件分支实现，不侵入正常路径。

2. **简单 BluePrint 的代价**：一个 FC 层的 BluePrint 仍然会生成少量的 deep/bwd 图（因为 Compiler 不知道我们用不到它们）。Compile 阶段它们仍会被生成到 train_cg_ 中但不会被捕获到 cudaGraph，不会影响运行时性能。

3. **多 RANK**：每个 rank 有独立的 cudaGraph 和 stream。H2D-only 模式下，rank 线程之间没有 barrier 同步——它们各自独立地从 TransferStation 读取状态并通过 cudaMemcpy 传输数据。因为 TransferStation 的 readable/writeable 标志由 rank 0 管理，其他 rank 跟随 rank 0 的节奏。

4. **Edge case：batches == 1**：只传输 buf0 一次即完成。

5. **最后的 Preprocessor join**：`prep.train()` 完成后所有 buffer 都是 writeable 状态。Preprocessor 遍历完所有样本后自然退出，不需要额外 drain。