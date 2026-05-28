# ZKR_FINAL — `run_h2d_only()` 最终增强方案

> 综合 YZT.md（三位小伙伴）、ZKR1/2/3、以及逐行代码核查后的最终实现方案。

---

## 1. 关键代码核查发现

### 1.1 CPU 有完整的 RANGE_H2D_COPY 后端

[`h2d_op.cpp:L199-L232`](file:///r:/renaissance/src/backend/ops/range/h2d_op.cpp#L199-L232):

```cpp
static void launch_range_h2d_copy_cpu(CpuOpContext* op_ctx) {
    uint8_t* staging_base = reg.staging_memory_ptr(rank);
    size_t per_zone = reg.staging_memory_size() / 2;
    size_t label_aligned = get_label_aligned();
    for (int i = 0; i < op_ctx->num_output_ranges; ++i) {
        void* dst = ArenaKeeper::instance().ptr_at(rank, range.offset);
        void* src = /* staging offset based on Region */;
        std::memcpy(dst, src, range.size);
    }
}
```

CPU 模式下的 H2D 传输 = `std::memcpy` 从 StagingBufferPool 到 DTensor 主机端地址。**此即 CPU H2D。**

### 1.2 `context().ptr_at()` 在 CPU 模式下可用

[`device_context.cpp:L179-L186`](file:///r:/renaissance/src/backend/device_context.cpp#L179-L186):

```cpp
void* DeviceContext::ptr_at(int dtensor_id) const noexcept {
    const DTensor& dt = current_mp_->get_dtensor(dtensor_id);
    return ArenaKeeper::instance().ptr_at(rank_for_context_, dt.offset());
}
```

CPU 模式下 `device_id_ < 0`，`ArenaKeeper` 返回的是 CPU 主机内存地址。GPU 模式下返回 GPU 设备内存地址。**API 对等，可对称使用。**

### 1.3 GlobalRegistry staging API 不依赖 CUDA

`reg.staging_memory_ptr(rank)` 和 `reg.staging_memory_size()` 在 CPU/GPU 模式均可用。StagingBufferPool 由 Preprocessor 分配在 NUMA 感知的主机内存上，与设备无关。

### 1.4 `prep.steps_per_epoch()` 不变

[`preprocessor.cpp:L969`](file:///r:/renaissance/src/data/preprocessor.cpp#L969)：基于 `total_train_samples` 在 `commit()` 时计算一次，`val()` 调用后不更新。**始终返回训练集 batch 数。**

### 1.5 `build_exec_table()` 在 CPU 模式下直接 return

[`deep_learning_task.cpp:L574-L575`](file:///r:/renaissance/src/task/deep_learning_task.cpp#L574-L575)：CPU 模式下 `gpu_exec_.graphs` 为空，不能访问。

---

## 2. 三方 ZKR 方案关键失误

### 2.1 ZKR1 / ZKR2 / ZKR3 的共同错误：认为 CPU 无 H2D

三个方案均建议 CPU 路径 `bandwidth_gbps = 0` 且不做实际数据复制。这是**根本性错误**——CPU 的 H2D 就是 `launch_range_h2d_copy_cpu` 所做的 `std::memcpy`。CPU 和 GPU 路径在概念上完全对称：

```
GPU H2D:  StagingPool(CPU) ──cudaMemcpyAsync──▶ DTensor(GPU)
CPU H2D:  StagingPool(CPU) ──std::memcpy──────▶ DTensor(CPU)
```

### 2.2 ZKR3 的 CPU 路径：`memcpy` 到临时 vector

```cpp
std::vector<uint8_t> temp_buffer(buffer_size);
std::memcpy(temp_buffer.data(), src_buffer, buffer_size);
// temp_buffer 随即析构，数据丢弃
```

拷贝到局部变量然后丢弃，等同于没拷贝。正确的目标是 **DTensor 所在的 ArenaKeeper 地址**。

### 2.3 ZKR2 的 `label_smce` 分析正确

ZKR2 正确指出 val epoch 不需要 `label_smce` 复制（val 不计算 loss）。ZKR1 和 ZKR3 未区分。

### 2.4 数据结构评估

| | ZKR1 | ZKR2 | ZKR3 |
|---|---|---|---|
| `H2DTestResult` | 不变 + 新增 `H2DRunResult` ✅ | 不变 + 新增 `H2DRunResult` ✅ | 扩展加 `epoch_details` 向量 ⚠️ |
| 向后兼容 | 最优 | 最优 | 汇总字段语义变化 |
| train/val 分离 | 两个独立函数 ✅ | 两个独立函数 ✅ | 四个独立函数 ⚠️（过度拆分） |

**推荐 ZKR2 的数据结构**（`H2DRunResult` + `H2DTestResult` 不变），这是最干净的设计。

---

## 3. 最终方案

### 3.1 总体架构

```
run_h2d_only() → H2DRunResult
  ├─ for epoch in [0..total_epochs_):
  │    ├─ run_h2d_only_train_epoch() → H2DTestResult
  │    │    └─ GPU 分支 (cudaGraphLaunch) / CPU 分支 (std::memcpy)
  │    └─ if should_validate_this_epoch():
  │         run_h2d_only_val_epoch() → H2DTestResult
  │              └─ GPU 分支 / CPU 分支（无 label_smce 复制）
  └─ 返回 H2DRunResult{ train_per_epoch[], val_per_epoch[] }
```

### 3.2 数据结构

[`include/renaissance/task/deep_learning_task.h`](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h)

```cpp
// H2DTestResult — 保持完全不变（单 epoch 统计单元）
struct H2DTestResult {
    int    batches        = 0;
    double elapsed_us     = 0.0;
    size_t total_bytes    = 0;
    double bandwidth_gbps = 0.0;
    bool   labels_ok      = true;
    bool   data_ok        = true;
    double avg_lat_us     = 0.0;
    double min_lat_us     = 0.0;
    double max_lat_us     = 0.0;
};

// H2DRunResult — 新增（多 epoch 容器）
struct H2DRunResult {
    int epochs_run = 0;
    int vals_run   = 0;
    std::vector<H2DTestResult> train_per_epoch;
    std::vector<H2DTestResult> val_per_epoch;
    double total_elapsed_us = 0.0;

    H2DTestResult aggregate_train() const;
    H2DTestResult aggregate_val() const;
};
```

**选择理由**：
- `H2DTestResult` 零变化 → `test_h2d_copy_bandwidth()` 等完全不受影响
- `train_per_epoch` / `val_per_epoch` 分离 → 语义明确，不会混叠
- `aggregate_train()` / `aggregate_val()` → 现有 test 一行适配

### 3.3 函数签名

```cpp
// deep_learning_task.h 中的声明

public:
    H2DRunResult run_h2d_only();           // 修改返回类型

private:
    H2DTestResult run_h2d_only_train_epoch();
    H2DTestResult run_h2d_only_val_epoch();
```

### 3.4 `run_h2d_only_train_epoch()` — 完整实现

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

    // per_zone_bytes
    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }

    auto t0 = std::chrono::steady_clock::now();

#ifdef TR_USE_CUDA
    if (registry.using_gpu()) {
        // ==================== GPU 路径 ====================
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
                    if (!ts) TR_DEVICE_ERROR("TS not ready for rank " << rank);

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
                        cudaMemcpyAsync(label_smce_ptr,
                            (buf_id == 0) ? label_ptr_a : label_ptr_b,
                            label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
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
        // ==================== CPU 路径 ====================
        // CPU H2D = std::memcpy 从 StagingBufferPool 到 DTensor 主机内存
        // 行为等价于 launch_range_h2d_copy_cpu，与 GPU cudaMemcpyAsync 语义对称

        // 预取 staging 基址和对齐信息
        void*  staging_base = registry.staging_memory_ptr(0);
        size_t per_zone     = registry.staging_memory_size() / 2;
        size_t label_aligned = static_cast<size_t>(
            DistributedTensor::compute_slot_bytes(
                Shape(registry.get_local_batch_size(), 1, 1, 1),
                DType::INT32, Region::I_A_LABEL));

        const auto& bl = active_memory_plan_->baseline();
        void* label_ptr_a    = context(0).ptr_at(bl.label_a);
        void* label_ptr_b    = context(0).ptr_at(bl.label_b);
        void* label_smce_ptr = context(0).ptr_at(bl.label_smce);
        size_t label_nbytes  = static_cast<size_t>(
            active_memory_plan_->get_dtensor(bl.label_a).slot_bytes());

        TransferStation* ts = nullptr;
        for (int w = 0; w < 200; ++w) {
            ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
            if (ts) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!ts) TR_DEVICE_ERROR("TS not ready for CPU path");

        // 预收集需要拷贝的 DTensor 列表（A 区和 B 区）
        // 使用 switch 与 launch_range_h2d_copy_cpu 风格一致
        struct CopyTask {
            void*  dst;
            void*  src_base;
            size_t nbytes;
        };
        std::vector<CopyTask> copy_a, copy_b;
        for (const auto& d : active_memory_plan_->dtensors()) {
            void* dst = context(0).ptr_at(d.id);
            size_t n  = static_cast<size_t>(d.slot_bytes());
            void* src = nullptr;
            switch (d.region) {
                case Region::I_A_LABEL: src = staging_base; break;
                case Region::I_A_DATA:  src = static_cast<uint8_t*>(staging_base) + label_aligned; break;
                case Region::I_B_LABEL: src = static_cast<uint8_t*>(staging_base) + per_zone; break;
                case Region::I_B_DATA:  src = static_cast<uint8_t*>(staging_base) + per_zone + label_aligned; break;
                default: continue;
            }
            if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
                copy_a.push_back({dst, src, n});
            else
                copy_b.push_back({dst, src, n});
        }

        for (int batch = 0; batch < batches; ++batch) {
            int buf_id = batch % 2;
            ts->wait_buffer_readable(buf_id);

            // CPU H2D：memcpy 从 staging 到 DTensor
            const auto& tasks = (buf_id == 0) ? copy_a : copy_b;
            for (const auto& t : tasks)
                std::memcpy(t.dst, t.src_base, t.nbytes);

            // label_smce 复制（train only）
            std::memcpy(label_smce_ptr,
                (buf_id == 0) ? label_ptr_a : label_ptr_b, label_nbytes);

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
    return r;
}
```

**CPU 路径核心逻辑说明**：

1. **对称性**：GPU 做 `cudaMemcpyAsync(staging→DTensor)`，CPU 做 `std::memcpy(staging→DTensor)`。目的地址都是 `context(rank).ptr_at(dt_id)`，通过 `ArenaKeeper` 解析——在 GPU 模式返回 GPU 地址，CPU 模式返回主机地址。
2. **预构建 CopyTask**：DTensor 列表和 staging 偏移在循环外预计算，避免每 batch 重复查表。A/B 两套 task 分别对应 buf_id=0/1，结构与 GPU 路径中用不同 CUDA graph 完全对称。
3. **label_smce 复制**：train epoch 必须做，与 GPU 路径对称——GPU 用 `cudaMemcpyDeviceToDevice`，CPU 用 `std::memcpy`。

### 3.5 `run_h2d_only_val_epoch()` — 实现

```cpp
H2DTestResult DeepLearningTask::run_h2d_only_val_epoch() {
    H2DTestResult r;
    auto& prep = Preprocessor::instance();
    auto& registry = GlobalRegistry::instance();

    size_t num_val   = registry.num_val_samples();
    int    batch_size = registry.get_local_batch_size();
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
                    if (!ts) TR_DEVICE_ERROR("TS not ready for rank " << rank);

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
        // CPU 路径：与 train epoch 相同的 memcpy 逻辑，无 label_smce 复制
        void*  staging_base  = registry.staging_memory_ptr(0);
        size_t per_zone      = registry.staging_memory_size() / 2;
        size_t label_aligned = static_cast<size_t>(
            DistributedTensor::compute_slot_bytes(
                Shape(registry.get_local_batch_size(), 1, 1, 1),
                DType::INT32, Region::I_A_LABEL));

        TransferStation* ts = nullptr;
        for (int w = 0; w < 200; ++w) {
            ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
            if (ts) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!ts) TR_DEVICE_ERROR("TS not ready for CPU path (val)");

        // 预收集 DTensor 拷贝任务（与 train epoch 相同，无 label_smce 复制）
        // switch 风格与 launch_range_h2d_copy_cpu 一致
        struct CopyTask { void* dst; void* src_base; size_t nbytes; };
        std::vector<CopyTask> copy_a, copy_b;
        for (const auto& d : active_memory_plan_->dtensors()) {
            void* dst = context(0).ptr_at(d.id);
            size_t n  = static_cast<size_t>(d.slot_bytes());
            void* src = nullptr;
            switch (d.region) {
                case Region::I_A_LABEL: src = staging_base; break;
                case Region::I_A_DATA:  src = static_cast<uint8_t*>(staging_base) + label_aligned; break;
                case Region::I_B_LABEL: src = static_cast<uint8_t*>(staging_base) + per_zone; break;
                case Region::I_B_DATA:  src = static_cast<uint8_t*>(staging_base) + per_zone + label_aligned; break;
                default: continue;
            }
            if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
                copy_a.push_back({dst, src, n});
            else
                copy_b.push_back({dst, src, n});
        }

        for (int batch = 0; batch < val_batches; ++batch) {
            int buf_id = batch % 2;
            ts->wait_buffer_readable(buf_id);

            const auto& tasks = (buf_id == 0) ? copy_a : copy_b;
            for (const auto& t : tasks)
                std::memcpy(t.dst, t.src_base, t.nbytes);

            // 无 label_smce 复制（val 不计算 loss）

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
    return r;
}
```

### 3.6 `run_h2d_only()` 入口

```cpp
H2DRunResult DeepLearningTask::run_h2d_only() {
    H2DRunResult result;

    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;
        auto t_epoch0 = std::chrono::steady_clock::now();

        H2DTestResult train_res = run_h2d_only_train_epoch();
        result.train_per_epoch.push_back(train_res);
        result.epochs_run++;

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

### 3.7 `H2DRunResult` 聚合方法

```cpp
H2DTestResult H2DRunResult::aggregate_train() const {
    H2DTestResult r;
    r.min_lat_us = 1e18;
    r.max_lat_us = 0.0;
    for (const auto& e : train_per_epoch) {
        r.batches     += e.batches;
        r.total_bytes += e.total_bytes;
        r.elapsed_us  += e.elapsed_us;
        if (e.avg_lat_us < r.min_lat_us) r.min_lat_us = e.avg_lat_us;
        if (e.avg_lat_us > r.max_lat_us) r.max_lat_us = e.avg_lat_us;
    }
    if (train_per_epoch.empty()) { r.min_lat_us = 0.0; }
    if (r.batches > 0) r.avg_lat_us = r.elapsed_us / r.batches;
    if (r.elapsed_us > 0.0 && r.total_bytes > 0)
        r.bandwidth_gbps = static_cast<double>(r.total_bytes) / (r.elapsed_us / 1e6)
                           / (1024.0 * 1024.0 * 1024.0);
    return r;
}

H2DTestResult H2DRunResult::aggregate_val() const {
    H2DTestResult r;
    r.min_lat_us = 1e18;
    r.max_lat_us = 0.0;
    for (const auto& e : val_per_epoch) {
        r.batches     += e.batches;
        r.total_bytes += e.total_bytes;
        r.elapsed_us  += e.elapsed_us;
        if (e.avg_lat_us < r.min_lat_us) r.min_lat_us = e.avg_lat_us;
        if (e.avg_lat_us > r.max_lat_us) r.max_lat_us = e.avg_lat_us;
    }
    if (val_per_epoch.empty()) { r.min_lat_us = 0.0; }
    if (r.batches > 0) r.avg_lat_us = r.elapsed_us / r.batches;
    if (r.elapsed_us > 0.0 && r.total_bytes > 0)
        r.bandwidth_gbps = static_cast<double>(r.total_bytes) / (r.elapsed_us / 1e6)
                           / (1024.0 * 1024.0 * 1024.0);
    return r;
}
```

### 3.8 `test_h2d_only_epoch.cpp` 适配

```cpp
// 修改
task.total_epochs(1)
    .num_classes(num_classes)
    .loss(CrossEntropyLoss())
    .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
    .scheduler(StepLR().step_size(30).gamma(0.1f));

task.compile_h2d_only();
auto run_res = task.run_h2d_only();
auto res = run_res.aggregate_train();  // 向后兼容，取 train epoch 汇总

// 原有打印逻辑完全不变...
```

---

## 4. GPU/CPU 对称性对照表

| 操作 | GPU 路径 | CPU 路径 |
|---|---|---|
| Preprocessor | `prep.train()` / `prep.val()` | 相同 |
| TransferStation 同步 | `wait_buffer_readable` / `set_buffer_writeable` | 相同 |
| 数据传输 | `cudaGraphLaunch(g_xfer)` → `cudaMemcpyAsync(staging→DTensor)` | `std::memcpy(staging→DTensor)` |
| 标签复制（train） | `cudaMemcpyAsync(I_A/B_LABEL → label_smce)` | `std::memcpy(I_A/B_LABEL → label_smce)` |
| 标签复制（val） | 无 | 无 |
| Target 地址 | `context(rank).ptr_at(dt_id)` → GPU addr | `context(0).ptr_at(dt_id)` → CPU addr |
| Rank 数 | `num_gpus_` (1~8) | 1（单 rank） |
| Bandwidth | PCIe/NVLink 带宽 | 系统内存带宽 |

---

## 5. 与三份 ZKR 提案的关键差异

| 问题 | ZKR1 | ZKR2 | ZKR3 | ZKR_FINAL |
|---|---|---|---|---|
| CPU H2D | ❌ `bandwidth=0` | ❌ `bandwidth=0` | ❌ memcpy 到临时 buffer | ✅ memcpy 到 DTensor，计算实际带宽 |
| CPU/GPU 对称性 | ❌ 不对等 | ❌ 不对等 | ❌ 不对等 | ✅ 完全对称 |
| Val batch 数 | 用 `steps_per_epoch()` ❌ | 手动计算 ✅ | 手动计算 ✅ | 手动计算 ✅ |
| 数据结构 | +`EpochDetail` ⚠️ | +`H2DRunResult` ✅ | +`EpochDetail` ⚠️ | +`H2DRunResult` ✅ |
| label_smce(val) | 未区分 | 正确排除 ✅ | 未区分 | 正确排除 ✅ |

---

## 6. 文件变更清单

| 文件 | 变更 |
|---|---|
| `include/renaissance/task/deep_learning_task.h` | +`H2DRunResult` struct；`run_h2d_only()` 返回 `H2DRunResult`；+`run_h2d_only_train_epoch()`、`run_h2d_only_val_epoch()` 声明 |
| `src/task/deep_learning_task.cpp` | 重写 `run_h2d_only()` 为 epoch 循环；抽取 `run_h2d_only_train_epoch()`；新增 `run_h2d_only_val_epoch()`；实现 `aggregate_train()`/`aggregate_val()` |
| `tests/correction/test_h2d_only_epoch.cpp` | +`.total_epochs(1)`；`run_res.aggregate_train()` 适配 |

---

## 7. 实施步骤

| 步骤 | 内容 | 风险 |
|---|---|---|
| 1 | `deep_learning_task.h`: +`H2DRunResult`，改 `run_h2d_only()` 签名，声明新私有方法 | 低 |
| 2 | 提取 `run_h2d_only_train_epoch()`：当前 GPU 逻辑完整移入，GPU/CPU 分支 | 低 |
| 3 | 新增 `run_h2d_only_val_epoch()`：参照 train，用 val_batches，无 label_smce | 中 |
| 4 | 重写 `run_h2d_only()`：外层 epoch 循环 + `should_validate_this_epoch()` | 中 |
| 5 | 实现 `H2DRunResult::aggregate_train()` / `aggregate_val()` | 低 |
| 6 | 适配 `test_h2d_only_epoch.cpp` + `.total_epochs(1)` | 低 |
| 7 | Windows RTX 4060 + Linux A100×8 编译测试 | — |