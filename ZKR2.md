# ZKR2 — `run_h2d_only()` 增强方案：CPU、多 epoch、val

> 综合阅读 YZT.md（小伙伴 S/K/D 三份方案）并进一步检查代码后的最终修改方案。

---

## 1. 代码检查补充发现

### 1.1 `run_val_epoch_gpu()` 的 buffer 释放只有 `rank == 0`

```cpp
// src/task/deep_learning_task.cpp:1565-1568
if (rank == 0) {
    ts->set_buffer_readable(buf, false);
    ts->set_buffer_writeable(buf, true);
}
```

这是**历史遗留**。在完整 val epoch 中，rank 0 释放后其他 rank 通过 `join()` 隐式同步。但 H2D-only 模式下每个 rank 独立消费自己的 TS buffer，**所有 rank 都应释放自己的 buffer**（参考 `run_h2d_only()` 已修复的 per-rank TS bug）。

### 1.2 `build_exec_table()` 在 CPU 模式下直接 return

```cpp
// src/task/deep_learning_task.cpp:574-575
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;
```

CPU 模式下 `gpu_exec_.graphs` 为空。任何尝试访问 `gpu_exec_.graphs[rank]` 的代码都会崩溃。因此 CPU 路径**绝对不能**走 GPU 图执行逻辑。

### 1.3 `num_gpus_` 在 CPU 模式下

`num_gpus_` 在 `compile_impl()` 中被设为 `reg.world_size()`。CPU 模式下 `world_size()` 通常为 1（单节点 CPU），所以 `K = 1` 是合理的。但 CPU 路径不应使用 `gpu_exec_`。

### 1.4 `Preprocessor::train()` 不依赖 GPU

`Preprocessor::train()`（`preprocessor.cpp:2193`）只操作 TransferStation 和 worker 线程，与 GPU/CPU 无关。CPU 模式下 TransferStation 仍然可用。

### 1.5 `label_smce` 复制只在 train 需要

`run_h2d_only()` 当前的 `cudaMemcpyAsync(label_smce_ptr, ...)` 只在 train epoch 有意义。val epoch 不计算 loss，不需要向 SoftmaxCE 提供标签入口。**val epoch 不应做 label_smce 复制**。

---

## 2. 三份方案的综合评估

| 维度 | 小伙伴 S | 小伙伴 K | 小伙伴 D |
|------|---------|---------|---------|
| **数据结构** | 新建 `H2DOnlyEpochResult` + `H2DOnlyResult`。一个 epoch 结构体同时容纳 train+val 字段（train 时 val 字段为 0），语义冗余。 | 新建 `H2DRunResult`，复用现有 `H2DTestResult` 作为单 epoch 单元。train/val 分离为两个向量，语义清晰。 | 直接扩展 `H2DTestResult`（添加 `epochs`/`train_epochs`/`val_epochs`）。破坏了原有结构体的单一语义，汇总字段混合 train+val，影响 `test_h2d_copy_bandwidth()` 等使用者。 |
| **函数拆分** | `run_h2d_only_epoch_gpu/cpu(bool do_validation)` 一个函数内同时处理 train+val，代码过长（~300 行）。 | `run_h2d_only_train_epoch()` + `run_h2d_only_val_epoch()`，职责分离清晰。 | `run_h2d_only_epoch(bool)` + `run_h2d_only_gpu/cpu()`，也不错，但 `run_h2d_only_gpu/cpu()` 作为多 epoch 包装层略显多余。 |
| **CPU 路径** | 每 batch 分配临时 buffer + `memcpy`（`std::vector<uint8_t> temp_buffer(buffer_size)`）。**不合理**：无意义地拷贝到临时内存，且每 batch 分配开销大。 | 简洁：`wait_buffer_readable` → `set_buffer_readable(false)` → `set_buffer_writeable(true)`。只测 Preprocessor 产出速度。 | 方案 A 要求检查 ComputationGraph CPU 执行路径，实现复杂；方案 B 需要 staging_offset 字段，目前不存在。 |
| **label_smce** | `run_h2d_only_epoch_gpu(bool)` 未区分 train/val，**若 do_validation=true 仍会复制 label_smce**。 | `run_h2d_only_train_epoch()` 有复制，`run_h2d_only_val_epoch()` 无复制，**正确**。 | 需 `if (!is_val)` 判断，可实现。 |
| **total_bytes** | `per_zone_bytes * batches * K`，**乘了 K**，改变了 `H2DTestResult` 原有的 per-rank 语义。 | 保持 `per_zone_bytes * batches`（per-rank），与现有代码一致。 | 保持原有语义。 |
| **向后兼容** | 完全废弃 `H2DTestResult`，所有调用点需重写。 | `H2DTestResult` 不变，现有测试只需外层套 `aggregate_train()`。 | `H2DTestResult` 扩展后，汇总字段语义改变（混入 val），`test_h2d_copy_bandwidth()` 等可能被影响。 |

**结论**：小伙伴 K 的大框架最优（`H2DRunResult` + `run_h2d_only_train_epoch()` / `run_h2d_only_val_epoch()`）。但三份方案都有细节问题需要修正。

---

## 3. 最终修改方案

### 3.1 数据结构：采用小伙伴 K 的设计，修正命名

```cpp
// include/renaissance/task/deep_learning_task.h

/// 单 epoch（train 或 val）的 H2D-only 统计 —— 保持现有结构不变
struct H2DTestResult {
    int    batches     = 0;
    double elapsed_us  = 0.0;
    size_t total_bytes = 0;
    double bandwidth_gbps = 0.0;
    bool   labels_ok   = true;
    bool   data_ok     = true;
    double avg_lat_us  = 0.0;
    double min_lat_us  = 0.0;
    double max_lat_us  = 0.0;
};

/// 多 epoch H2D-only 运行结果（新增）
struct H2DRunResult {
    int epochs_run = 0;                     // train epoch 数
    int vals_run   = 0;                     // val epoch 数
    std::vector<H2DTestResult> train_per_epoch;
    std::vector<H2DTestResult> val_per_epoch;
    double total_elapsed_us = 0.0;          // 全部 epoch 总 wall-clock

    H2DTestResult aggregate_train() const;
    H2DTestResult aggregate_val() const;
};
```

**为什么不采用小伙伴 S 的 `H2DOnlyEpochResult`？**
- S 的结构体把 train 和 val 合在一个 epoch 中（`train_seconds` + `val_seconds`）。但实际上 train 和 val 是两个独立的 Preprocessor 启动 + 数据传输过程。用一个结构体表示导致 val epoch 时 `train_seconds = 0` 或 train epoch 时 `val_seconds = 0`，语义不纯净。

**为什么不采用小伙伴 D 的扩展 `H2DTestResult`？**
- D 在 `H2DTestResult` 上加 `epochs` 向量，使原有汇总字段（`batches`/`elapsed_us`/`total_bytes`）变成跨 epoch 混合值。这会导致 `test_h2d_copy_bandwidth()` 等只读取 `r.bandwidth_gbps` 的代码得到"train+val 混合带宽"，失去物理意义。

### 3.2 `run_h2d_only()` 入口

```cpp
H2DRunResult DeepLearningTask::run_h2d_only() {
    H2DRunResult result;

    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;
        auto t_epoch0 = std::chrono::steady_clock::now();

        // ---- Train ----
        H2DTestResult train_res = run_h2d_only_train_epoch();
        result.train_per_epoch.push_back(train_res);
        result.epochs_run++;

        // ---- Val ----
        if (should_validate_this_epoch()) {
            H2DTestResult val_res = run_h2d_only_val_epoch();
            result.val_per_epoch.push_back(val_res);
            result.vals_run++;
        }

        auto t_epoch1 = std::chrono::steady_clock::now();
        result.total_elapsed_us += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                t_epoch1 - t_epoch0).count());
    }

    return result;
}
```

**向后兼容处理**：`total_epochs_` 默认值为 35。当前 `test_h2d_only_epoch.cpp` 没有设置 `.total_epochs()`，修改后会跑 35 个 epoch。
- **措施**：在 `test_h2d_only_epoch.cpp` 的 task 配置链上追加 `.total_epochs(1)`。
- **备选**：在 `run_h2d_only()` 内部增加显式检查：`TR_CHECK(total_epochs_ > 0, ...)`，但不用硬编码默认值覆盖。

### 3.3 `run_h2d_only_train_epoch()`：提取当前 GPU 逻辑，加入 CPU 分支

将当前 `run_h2d_only()` 的核心实现（line 2151-2257 的 GPU 逻辑）完整提取为私有方法，并在内部做 GPU/CPU 分支。

```cpp
H2DTestResult DeepLearningTask::run_h2d_only_train_epoch() {
    H2DTestResult r;
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;

    if (batches == 0) return r;

    // Preprocessor 线程
    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    // per_zone_bytes（train/val 通用）
    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }

    auto t0 = std::chrono::steady_clock::now();

#ifdef TR_USE_CUDA
    if (registry.using_gpu()) {
        // ========== GPU 路径（当前完整逻辑，一字不改）==========
        std::vector<std::exception_ptr> rank_exc(K);
        std::vector<std::thread> threads;
        threads.reserve(K);

        for (int rank = 0; rank < K; ++rank) {
            threads.emplace_back([&, rank]() {
                try {
                    cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                    if (err != cudaSuccess)
                        TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank);

                    TransferStation* ts = nullptr;
                    for (int w = 0; w < 200; ++w) {
                        ts = static_cast<TransferStation*>(
                            registry.transfer_station_ptr(rank));
                        if (ts) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

                    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                    const auto& g_tab = gpu_exec_.graphs[rank];
                    auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                    auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                    cudaStream_t s_trans = static_cast<cudaStream_t>(
                        context(rank).stream(StreamKind::TRANS));

                    const auto& bl = active_memory_plan_->baseline();
                    auto label_ptr_a    = context(rank).ptr_at(bl.label_a);
                    auto label_ptr_b    = context(rank).ptr_at(bl.label_b);
                    auto label_smce_ptr = context(rank).ptr_at(bl.label_smce);
                    size_t label_nbytes = static_cast<size_t>(
                        active_memory_plan_->get_dtensor(bl.label_a).slot_bytes());

                    for (int batch = 0; batch < batches; ++batch) {
                        int buf_id = batch % 2;
                        auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                        ts->wait_buffer_readable(buf_id);
                        cudaGraphLaunch(g_xfer, s_trans);
                        cudaStreamSynchronize(s_trans);

                        ts->set_buffer_readable(buf_id, false);
                        ts->set_buffer_writeable(buf_id, true);

                        // label_smce 复制（train only）
                        cudaMemcpyAsync(
                            label_smce_ptr,
                            (buf_id == 0) ? label_ptr_a : label_ptr_b,
                            label_nbytes,
                            cudaMemcpyDeviceToDevice, s_trans);
                        cudaStreamSynchronize(s_trans);
                    }
                } catch (...) {
                    rank_exc[rank] = std::current_exception();
                }
            });
        }

        for (auto& t : threads) t.join();
    } else
#endif
    {
        // ========== CPU 路径（新增）==========
        // CPU 模式下没有 GPU，没有 H2D 传输。
        // 测量的是 Preprocessor → TransferStation 的纯数据产出速度。
        TransferStation* ts = nullptr;
        for (int w = 0; w < 200; ++w) {
            ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
            if (ts) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!ts) TR_DEVICE_ERROR("TransferStation not ready for CPU path");

        for (int batch = 0; batch < batches; ++batch) {
            int buf_id = batch % 2;
            ts->wait_buffer_readable(buf_id);
            ts->set_buffer_readable(buf_id, false);
            ts->set_buffer_writeable(buf_id, true);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
#ifdef TR_USE_CUDA
    if (registry.using_gpu()) {
        for (int rank = 0; rank < K; ++rank)
            if (rank_exc[rank]) std::rethrow_exception(rank_exc[rank]);
    }
#endif

    double elapsed_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    r.batches     = batches;
    r.elapsed_us  = elapsed_us;
    r.total_bytes = per_zone_bytes * static_cast<size_t>(batches);
    if (batches > 0) r.avg_lat_us = elapsed_us / static_cast<double>(batches);
    if (elapsed_us > 0.0 && r.total_bytes > 0) {
        r.bandwidth_gbps = static_cast<double>(r.total_bytes) / (elapsed_us / 1e6)
                           / (1024.0 * 1024.0 * 1024.0);
    }
#ifdef TR_USE_CUDA
    if (!registry.using_gpu())
#endif
    {
        r.bandwidth_gbps = 0.0;  // CPU 路径无 H2D 传输
    }

    return r;
}
```

**CPU 路径说明**：
- 不访问 `gpu_exec_.graphs`（CPU 模式下为空）
- 不做 `cudaGraphLaunch`、`cudaMemcpyAsync`、`label_smce` 复制
- 只做 `wait_buffer_readable` → `set_buffer_readable(false)` → `set_buffer_writeable(true)`
- `bandwidth_gbps = 0`，`total_bytes` 仍保留逻辑量（用于对比"纯 Preprocessor 开销"）

### 3.4 `run_h2d_only_val_epoch()`：新增

```cpp
H2DTestResult DeepLearningTask::run_h2d_only_val_epoch() {
    H2DTestResult r;
    auto& prep = Preprocessor::instance();
    auto& registry = GlobalRegistry::instance();

    size_t num_val = registry.num_val_samples();
    int batch_size = registry.get_local_batch_size();
    if (batch_size <= 0) batch_size = 1;
    int val_batches = static_cast<int>((num_val + batch_size - 1) / batch_size);
    const int K = num_gpus_;

    if (val_batches == 0) return r;

    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.val(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }

    auto t0 = std::chrono::steady_clock::now();

#ifdef TR_USE_CUDA
    if (registry.using_gpu()) {
        std::vector<std::exception_ptr> rank_exc(K);
        std::vector<std::thread> threads;
        threads.reserve(K);

        for (int rank = 0; rank < K; ++rank) {
            threads.emplace_back([&, rank]() {
                try {
                    cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                    if (err != cudaSuccess)
                        TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank);

                    TransferStation* ts = nullptr;
                    for (int w = 0; w < 200; ++w) {
                        ts = static_cast<TransferStation*>(
                            registry.transfer_station_ptr(rank));
                        if (ts) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

                    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                    const auto& g_tab = gpu_exec_.graphs[rank];
                    auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                    auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                    cudaStream_t s_trans = static_cast<cudaStream_t>(
                        context(rank).stream(StreamKind::TRANS));

                    for (int batch = 0; batch < val_batches; ++batch) {
                        int buf_id = batch % 2;
                        auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                        ts->wait_buffer_readable(buf_id);
                        cudaGraphLaunch(g_xfer, s_trans);
                        cudaStreamSynchronize(s_trans);

                        // 所有 rank 释放自己的 buffer（修正历史遗留的 rank==0 问题）
                        ts->set_buffer_readable(buf_id, false);
                        ts->set_buffer_writeable(buf_id, true);
                    }
                } catch (...) {
                    rank_exc[rank] = std::current_exception();
                }
            });
        }

        for (auto& t : threads) t.join();
        for (int rank = 0; rank < K; ++rank)
            if (rank_exc[rank]) std::rethrow_exception(rank_exc[rank]);
    } else
#endif
    {
        // CPU 路径
        TransferStation* ts = nullptr;
        for (int w = 0; w < 200; ++w) {
            ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
            if (ts) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!ts) TR_DEVICE_ERROR("TransferStation not ready for CPU path (val)");

        for (int batch = 0; batch < val_batches; ++batch) {
            int buf_id = batch % 2;
            ts->wait_buffer_readable(buf_id);
            ts->set_buffer_readable(buf_id, false);
            ts->set_buffer_writeable(buf_id, true);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);

    double elapsed_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    r.batches     = val_batches;
    r.elapsed_us  = elapsed_us;
    r.total_bytes = per_zone_bytes * static_cast<size_t>(val_batches);
    if (val_batches > 0) r.avg_lat_us = elapsed_us / static_cast<double>(val_batches);
    if (elapsed_us > 0.0 && r.total_bytes > 0) {
        r.bandwidth_gbps = static_cast<double>(r.total_bytes) / (elapsed_us / 1e6)
                           / (1024.0 * 1024.0 * 1024.0);
    }
#ifdef TR_USE_CUDA
    if (!registry.using_gpu())
#endif
    {
        r.bandwidth_gbps = 0.0;
    }

    return r;
}
```

**val epoch 与 train epoch 的关键差异**：

| 项 | Train | Val |
|---|---|---|
| Preprocessor 启动 | `prep.train()` | `prep.val()` |
| Batch 数 | `prep.steps_per_epoch()` | `ceil(num_val_samples / batch_size)` |
| `label_smce` 复制 | ✅ 有 | ❌ 无 |
| Buffer 释放 | 所有 rank 各自释放 | 所有 rank 各自释放（已修正 `rank==0` 问题） |

### 3.5 `H2DRunResult` 聚合方法实现

```cpp
H2DTestResult H2DRunResult::aggregate_train() const {
    H2DTestResult r;
    if (train_per_epoch.empty()) return r;

    for (const auto& e : train_per_epoch) {
        r.batches     += e.batches;
        r.total_bytes += e.total_bytes;
        r.elapsed_us  += e.elapsed_us;
        if (r.min_lat_us == 0.0 || e.avg_lat_us < r.min_lat_us) r.min_lat_us = e.avg_lat_us;
        if (e.avg_lat_us > r.max_lat_us) r.max_lat_us = e.avg_lat_us;
    }
    if (r.batches > 0) r.avg_lat_us = r.elapsed_us / static_cast<double>(r.batches);
    if (r.elapsed_us > 0.0 && r.total_bytes > 0) {
        r.bandwidth_gbps = static_cast<double>(r.total_bytes) / (r.elapsed_us / 1e6)
                           / (1024.0 * 1024.0 * 1024.0);
    }
    return r;
}

H2DTestResult H2DRunResult::aggregate_val() const {
    H2DTestResult r;
    if (val_per_epoch.empty()) return r;

    for (const auto& e : val_per_epoch) {
        r.batches     += e.batches;
        r.total_bytes += e.total_bytes;
        r.elapsed_us  += e.elapsed_us;
        if (r.min_lat_us == 0.0 || e.avg_lat_us < r.min_lat_us) r.min_lat_us = e.avg_lat_us;
        if (e.avg_lat_us > r.max_lat_us) r.max_lat_us = e.avg_lat_us;
    }
    if (r.batches > 0) r.avg_lat_us = r.elapsed_us / static_cast<double>(r.batches);
    if (r.elapsed_us > 0.0 && r.total_bytes > 0) {
        r.bandwidth_gbps = static_cast<double>(r.total_bytes) / (r.elapsed_us / 1e6)
                           / (1024.0 * 1024.0 * 1024.0);
    }
    return r;
}
```

### 3.6 `test_h2d_only_epoch.cpp` 适配

```cpp
// 修改前
task.compile_h2d_only();
auto res = task.run_h2d_only();

// 修改后
task.total_epochs(1).compile_h2d_only();
auto run_res = task.run_h2d_only();
auto res = run_res.aggregate_train();  // 保持向后兼容

// 原有打印逻辑不变...
// 新增：打印 per-epoch 明细（可选）
for (size_t i = 0; i < run_res.train_per_epoch.size(); ++i) {
    std::cout << "Epoch " << (i+1) << " train: "
              << run_res.train_per_epoch[i].elapsed_us / 1e6 << " s\n";
}
```

---

## 4. 接口变更清单

### 4.1 `include/renaissance/task/deep_learning_task.h`

| 变更 | 说明 |
|---|---|
| 新增 `struct H2DRunResult` | 包含 `train_per_epoch`、`val_per_epoch`、`aggregate_train()`、`aggregate_val()` |
| `run_h2d_only()` 返回类型 | `H2DTestResult` → `H2DRunResult` |
| 新增私有方法 | `H2DTestResult run_h2d_only_train_epoch()` |
| 新增私有方法 | `H2DTestResult run_h2d_only_val_epoch()` |

### 4.2 `src/task/deep_learning_task.cpp`

| 变更 | 说明 |
|---|---|
| `run_h2d_only()` | 重写：外层 epoch 循环，返回 `H2DRunResult` |
| `run_h2d_only_train_epoch()` | 新建：提取当前 GPU 逻辑，内部做 GPU/CPU 分支，含 label_smce 复制 |
| `run_h2d_only_val_epoch()` | 新建：val H2D-only 逻辑，内部做 GPU/CPU 分支，无 label_smce 复制，所有 rank 释放 buffer |

### 4.3 `tests/correction/test_h2d_only_epoch.cpp`

| 变更 | 说明 |
|---|---|
| `.total_epochs(1)` | 在 task 配置链追加，避免默认 35 epoch |
| 返回类型适配 | `auto run_res = task.run_h2d_only(); auto res = run_res.aggregate_train();` |

---

## 5. 实施步骤

| 步骤 | 内容 | 风险 |
|------|------|------|
| 1 | `deep_learning_task.h` 中新增 `H2DRunResult`，修改 `run_h2d_only()` 签名 | 低 |
| 2 | 提取 `run_h2d_only_train_epoch()`（GPU 逻辑一字不改，仅外移） | 低 |
| 3 | 重写 `run_h2d_only()`，加入 epoch 循环 + `should_validate_this_epoch()` | 中 |
| 4 | 实现 `run_h2d_only_val_epoch()` | 中 |
| 5 | 在 `run_h2d_only_train_epoch()` / `run_h2d_only_val_epoch()` 中加入 CPU 分支 | 低 |
| 6 | 实现 `H2DRunResult::aggregate_train()` / `aggregate_val()` | 低 |
| 7 | 修改 `test_h2d_only_epoch.cpp` | 低 |
| 8 | Windows（RTX 4060）和 Linux（A100×8）回归测试 | — |

---

## 6. 测试矩阵

| 场景 | 命令/代码 | 预期 |
|---|---|---|
| GPU 单 epoch 回归 | `./test_h2d_only_epoch ... --device GPU` | 行为与修改前完全一致 |
| GPU 多 epoch | `task.total_epochs(3).compile_h2d_only(); auto r = task.run_h2d_only();` | `r.epochs_run == 3` |
| GPU 多 epoch + val | `task.total_epochs(3).validate_every(1,1).compile_h2d_only();` | `r.epochs_run == 3, r.vals_run == 3` |
| CPU 单 epoch | `./test_h2d_only_epoch ... --device CPU` | 成功跑完，`bandwidth_gbps == 0` |
| CPU val | `--device CPU` + `validate_every(1,1)` | val epoch 也能跑通 |
| 混合 train/val | `total_epochs=4, validate_every(2,0)` | epochs: T,T+V,T,T+V |
