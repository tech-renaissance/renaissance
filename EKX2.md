# EKX2：重写 DeepLearningTask::compile()+run() 的完整方案

> 基于 EKX.md 的框架，对实际源码重新复查后，修正了原方案中的关键遗漏，给出可直接落地的修改计划。

---

## 1. 执行摘要

**结论先行**：`compile()` 完全不需要改；`run()` 的问题 100% 集中在 `run_train_epoch_gpu()` 和 `run_val_epoch_gpu()` 的 **H2D 子系统** 上。

通过逐行比对实际源码，确认 `run_train_epoch_gpu()`（lines 881–1453）与已验证正确的 `run_h2d_only_train_epoch()`（lines 2148–2324）存在 **6 处关键差异**（EKX.md 只列了 5 处，遗漏了第 6 处）。

**核心修正策略**：不破坏原有 pipeline 并行结构（Batch 0 预传输 + batch 0..batches-2 循环 + Last batch），只在每个 XFER 完成点插入 `label_smce` D2D 拷贝、修复 buffer 管理语义、移除诊断代码、修复 TransferStation 获取方式。改动最小、风险最低、性能不损失。

**最终目标**：跑通 `test_dl_full.cpp`（MNIST 3 epoch，best_top1 > 85%）。

---

## 2. 基于源码的复查结果

### 2.1 compile 路径：零改动（与 EKX.md 一致）

```
compile_h2d_only()              compile()
       │                            │
       └──► compile_impl() ◄────────┘
                ├──► build_graph_atlas()   // h2d_only_ 决定注册哪些 GraphId
                ├──► pre_capture()         // 对注册的所有图一视同仁捕获
                └──► dl->build_exec_table() // h2d_only_ 决定解析哪些 slot
```

- `build_graph_atlas()` line 480–534：h2d_only_=true 时只注册 TRANSFER_A/B；false 时注册全部。
- `build_exec_table()` line 573–671：h2d_only_=true 时只解析 XFER_A/B；false 时解析全部 18+ slot。
- **TRANSFER_A/B 的捕获和解析路径在两种模式下完全一致**，因此 compile() 无需任何修改。

### 2.2 run_h2d_only_train_epoch() 精确复盘

**GPU 路径**（lines 2178–2239）：

```cpp
for (int rank = 0; rank < K; ++rank) {
    threads.emplace_back([&, rank]() {
        cudaSetDevice(gpu_exec_.device_ids[rank]);
        TransferStation* ts = registry.transfer_station_ptr(rank);  // ← ⑥ 每个 rank 独立获取
        auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
        auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
        cudaStream_t s_trans = context(rank).stream(StreamKind::TRANS);
        auto label_ptr_a = context(rank).ptr_at(bl.label_a);
        auto label_ptr_b = context(rank).ptr_at(bl.label_b);
        auto label_smce_ptr = context(rank).ptr_at(bl.label_smce);
        size_t label_nbytes = ...;

        for (int batch = 0; batch < batches; ++batch) {
            int buf_id = batch % 2;
            auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

            ts->wait_buffer_readable(buf_id);           // ① 条件变量阻塞
            cudaGraphLaunch(g_xfer, s_trans);           // ② H2D
            cudaStreamSynchronize(s_trans);             // ③ sync
            ts->set_buffer_readable(buf_id, false);     // ④ 所有 rank
            ts->set_buffer_writeable(buf_id, true);     // ⑤ 所有 rank

            cudaMemcpyAsync(label_smce_ptr,             // ⑥ label_smce D2D
                (buf_id == 0) ? label_ptr_a : label_ptr_b,
                label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
            cudaStreamSynchronize(s_trans);             // ⑦ D2D sync
        }
    });
}
```

**CPU 路径**（lines 2244–2303）：同样严谨，从 StagingBufferPool 直接 `std::memcpy` 到 device context 指针，并同步 `label_smce`。

### 2.3 run_train_epoch_gpu() 问题清单（基于实际行号）

| # | 差异点 | run_h2d_only_train_epoch() ✅ | run_train_epoch_gpu() ❌ | 致命性 |
|---|--------|------------------------------|--------------------------|--------|
| 1 | **label_smce D2D 拷贝** | 每 batch 在 H2D sync 后执行（line 2225–2228） | **完全缺失** | 🔴 致命 |
| 2 | **Buffer 可读等待** | `wait_buffer_readable()` 条件变量阻塞 | `while(!buffer_is_readable) sleep(100us)` 自旋（lines 967–968, 1147–1148） | 🟡 高 |
| 3 | **Buffer 状态管理** | **所有 rank** 调用 `set_buffer_*`（lines 2222–2223） | **仅 rank 0** 调用（lines 1029–1030, 1231–1234） | 🟡 高 |
| 4 | **TransferStation 获取** | 每个 rank 调用 `transfer_station_ptr(rank)`（line 2188） | 主线程获取 `transfer_station_ptr(0)` 后共享给所有 rank（line 885） | 🟡 高 |
| 5 | **诊断代码污染** | 零诊断 | ~500 行 DIAG-* / WGHT-DBG / LOSS-DBG（lines 895–921, 971–1031, 1114–1144, 1165–1230, 1236–1348, 1371–1375, 1431–1450） | 🟠 中 |
| 6 | **magic value 初始化** | 无 | loss=3.14f, scaling=1.0f 直接写 device（lines 894–913） | 🟠 中 |

**遗漏说明**：EKX.md 列了 5 处差异，但遗漏了第 4 处（TransferStation 获取方式）和第 6 处（magic value 初始化）。第 4 处虽然对单卡无影响，但多卡下是确定的 bug；第 6 处会污染初始状态，干扰 loss 的首次计算。

### 2.4 关键发现：EKX.md 的 Phase 1 全量预传输方案不可行

EKX.md 建议 Phase 1 采用：
```
execute_h2d_epoch_gpu(batches, ...);   // 全量 H2D 先完成
for batch = 0..batches-1: compute;     // 再全量计算
```

**此方案在 A/B 双缓冲下存在确定性错误**：
- A/B 只有 2 个 GPU buffer（I_A_DATA / I_B_DATA）
- 若 batches > 2，全量 H2D 会让 batch 2 覆盖 batch 0 的 GPU 数据
- 此时 batch 0 的计算尚未执行，读到的是 batch 2 的数据

**正确做法**：保持原有 pipeline 结构（每个 batch 的 H2D 和计算成对执行），仅修复 H2D 子系统的 6 处差异。

成对执行的时序安全性证明：
1. H2D graph 将数据从 CPU staging → GPU I_A/B_DATA
2. `cudaStreamSync(s_trans)` 后，GPU 已拥有当前 batch 的完整数据
3. `set_buffer_writeable(buf_id, true)` 后，CPU staging 可被 Preprocessor 复写
4. 计算图读的是 GPU I_A/B_DATA，与 CPU staging 无关
5. 因此 batch N+2 可以安全覆盖 batch N 的 CPU/GPU buffer

### 2.5 run_val_epoch_gpu() 问题

**位置**：lines 1469–1606

| 差异点 | run_h2d_only_val_epoch() ✅ | run_val_epoch_gpu() ❌ |
|--------|---------------------------|------------------------|
| Buffer 可读等待 | `wait_buffer_readable()` | `while(sleep)` |
| set_buffer | **所有 rank**（line 2400–2401） | **仅 rank 0**（lines 1565–1568） |
| TransferStation | `transfer_station_ptr(rank)` | `transfer_station_ptr(0)` |

验证不需要 `label_smce` 拷贝（无 loss 梯度计算），但同样需要修复 buffer 管理和 ts 获取。

---

## 3. 重写方案

### 3.1 设计原则

1. **compile 零改动**：`compile()` / `build_graph_atlas()` / `build_exec_table()` / `pre_capture()` 保持现状
2. **pipeline 结构保持**：不拆分为"全量 H2D + 全量计算"，保留 Batch 0 预传输 + batch 0..batches-2 循环 + Last batch 的流水线
3. **H2D 子系统逐行对齐**：在每个 XFER 完成点插入 `label_smce` D2D 拷贝，修复 buffer 管理，用 `wait_buffer_readable` 替代自旋
4. **计算阶段不动**：ZERO_GRAD → FIRST_FWD → DEEP_FWD_BWD → FIRST_BWD → ALLREDUCE → WEIGHT_UPDATE 的启动顺序、stream 分配、同步点保持原样
5. **诊断代码全部移除**：~500 行 DIAG-* / WGHT-DBG / LOSS-DBG / PRE-TEST 全部删除
6. **magic value 初始化移除**：loss/scaling 不应被硬编码覆盖，它们由计算图或 H2D graph 正确设置

### 3.2 新 run_train_epoch_gpu() 结构

```cpp
float DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, K, using_amp, &exc]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess)
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank);

                // ===== 图句柄获取（保留原代码 lines 932–947）=====
                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a  = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b  = g_tab[S(GraphSlot::XFER_B)];
                auto g_deep_a  = g_tab[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto g_deep_b  = g_tab[S(GraphSlot::FWD_BWD_DEEP_B)];
                auto g_first   = g_tab[S(GraphSlot::FIRST_LAYER_BWD)];
                auto g_first_b = g_tab[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto g_far     = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto g_zg      = g_tab[S(GraphSlot::ZERO_GRAD)];
                auto g_dar     = g_tab[S(GraphSlot::DEEP_ALLREDUCE)];
                auto g_wu      = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
                auto g_gc      = g_tab[S(GraphSlot::GRAD_CONVERT)];
                auto g_fwd_a   = g_tab[S(GraphSlot::FIRST_FWD_A)];
                auto g_fwd_b   = g_tab[S(GraphSlot::FIRST_FWD_B)];

                // ===== Stream（保留原代码 lines 949–960）=====
                DeviceContext& ctx = context(rank);
                cudaStream_t s_up    = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                auto sync_comp = [&]() { cudaStreamSynchronize(s_c1); cudaStreamSynchronize(s_c2); };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr   = [&]() { cudaStreamSynchronize(s_trans); };

                // ===== TransferStation（对齐 h2d_only：每个 rank 独立获取）=====
                TransferStation* ts = nullptr;
                for (int w = 0; w < 200; ++w) {
                    ts = static_cast<TransferStation*>(registry.transfer_station_ptr(rank));
                    if (ts) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

                // ===== Label pointers（对齐 h2d_only）=====
                const auto& bl = active_memory_plan_->baseline();
                auto label_ptr_a = ctx.ptr_at(bl.label_a);
                auto label_ptr_b = ctx.ptr_at(bl.label_b);
                auto label_smce_ptr = ctx.ptr_at(bl.label_smce);
                size_t label_nbytes = (bl.label_a >= 0)
                    ? static_cast<size_t>(active_memory_plan_->get_dtensor(bl.label_a).slot_bytes())
                    : 0;

                float lr;
                float* lr_dev_ptr = static_cast<float*>(ctx.ptr_at(lr_dtensor_id_));
                bool frozen = is_first_layer_frozen();
                int32_t loss_id = bl.loss;

                // ================================================================
                // Batch 0 预传输（对齐 h2d_only，新增 label_smce copy）
                // ================================================================
                ts->wait_buffer_readable(0);
                if (g_xfer_a) cudaGraphLaunch(g_xfer_a, s_trans);
                sync_tr();
                ts->set_buffer_readable(0, false);
                ts->set_buffer_writeable(0, true);
                if (label_smce_ptr && label_ptr_a && label_nbytes > 0) {
                    cudaMemcpyAsync(label_smce_ptr, label_ptr_a,
                        label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
                    sync_tr();
                }

                // ===== 单 batch 边界 =====
                if (batches == 1) {
                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd_a) cudaGraphLaunch(g_fwd_a, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_a) cudaGraphLaunch(g_deep_a, s_c1);
                    sync_comp();

                    if (!frozen) {
                        if (g_first) cudaGraphLaunch(g_first, s_c1);
                    }
                    if (g_dar) cudaGraphLaunch(g_dar, s_up);
                    sync_comp(); sync_up();

                    if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(0);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_far) cudaGraphLaunch(g_far, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up);
                    sync_up();
                    return;
                }

                // ================================================================
                // 循环：batch = 0 .. batches-2（保留 pipeline 并行结构）
                // ================================================================
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;
                    auto g_fwd   = from_a ? g_fwd_a : g_fwd_b;
                    auto g_deep  = from_a ? g_deep_a : g_deep_b;
                    auto g_xfer_n = from_a ? g_xfer_b : g_xfer_a;

                    // Phase 1: ZERO_GRAD ‖ FIRST_FWD
                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                    sync_comp(); sync_up();

                    // Wait next buffer（对齐 h2d_only）
                    ts->wait_buffer_readable(next_buf);

                    // Phase 2: DEEP_FWD_BWD ‖ XFER(next)
                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
                    sync_comp(); sync_tr();

                    // XFER 收尾（对齐 h2d_only：所有 rank set_buffer + label_smce copy）
                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);
                    if (label_smce_ptr && label_nbytes > 0) {
                        void* label_src = (next_buf == 0) ? label_ptr_a : label_ptr_b;
                        cudaMemcpyAsync(label_smce_ptr, label_src,
                            label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
                        sync_tr();
                    }

                    // Phase 3: FIRST_BWD ‖ DEEP_ALLREDUCE
                    if (!frozen) {
                        auto g_first_cur = from_a ? g_first : g_first_b;
                        // 如果 FIRST_LAYER_BWD_B 未 capture，回退到 FIRST_LAYER_BWD
                        if (!g_first_cur) g_first_cur = g_first;
                        if (g_first_cur) cudaGraphLaunch(g_first_cur, s_c1);
                    }
                    if (g_dar) cudaGraphLaunch(g_dar, s_up);
                    sync_comp(); sync_up();

                    // AMP
                    if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }

                    // Phase 4: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE
                    lr = fetch_lr_for_batch(batch);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_far) cudaGraphLaunch(g_far, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up);
                    sync_up();
                }

                // ================================================================
                // Last batch (batch = batches-1)
                // ================================================================
                {
                    bool last_a = ((batches - 1) % 2 == 0);
                    auto g_fwd_l  = last_a ? g_fwd_a : g_fwd_b;
                    auto g_deep_l = last_a ? g_deep_a : g_deep_b;

                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd_l) cudaGraphLaunch(g_fwd_l, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
                    sync_comp();

                    if (!frozen) {
                        auto g_first_cur = last_a ? g_first : g_first_b;
                        if (!g_first_cur) g_first_cur = g_first;
                        if (g_first_cur) cudaGraphLaunch(g_first_cur, s_c1);
                    }
                    if (g_dar) cudaGraphLaunch(g_dar, s_up);
                    sync_comp(); sync_up();

                    if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batches - 1);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_far) cudaGraphLaunch(g_far, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up);
                    sync_up();
                }

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank)
        if (exc[rank]) std::rethrow_exception(exc[rank]);

    // 读取 train loss（仅 rank 0）
    float train_loss = 0.0f;
    if (active_memory_plan_->baseline().loss >= 0) {
        const auto& loss_dt = active_memory_plan_->get_dtensor(active_memory_plan_->baseline().loss);
        Tensor h_loss = fetch_from_rank(loss_dt, 0);
        train_loss = h_loss.data<float>()[0];
    }
    return train_loss;
}
```

### 3.3 与旧结构的差异对比

| 结构 | 旧实现 | 新实现 | 原因 |
|------|--------|--------|------|
| Batch 0 预传输后 | 仅 rank 0 set_buffer + ~150 行诊断 | **所有 rank** set_buffer + **label_smce D2D** | 对齐 h2d_only |
| batch 循环中 XFER(next) 后 | 仅 rank 0 set_buffer + ~300 行诊断 | **所有 rank** set_buffer + **label_smce D2D** | 对齐 h2d_only |
| Buffer 等待 | `while(sleep)` | `wait_buffer_readable()` | 对齐 h2d_only |
| TransferStation | 主线程 `ptr(0)` 共享 | 每个 rank `ptr(rank)` | 对齐 h2d_only |
| 诊断代码 | ~500 行 | **0 行** | 可维护性 |
| loss/scaling 初始化 | 3.14f / 1.0f 写 device | **删除** | 由计算图/H2D 正确设置 |
| FIRST_LAYER_BWD_B 回退 | 无 | `if (!g_first_cur) g_first_cur = g_first` | 健壮性 |

### 3.4 run_val_epoch_gpu() 微调

仅需 3 处修改：

```cpp
// 修改前（lines 1474, 1535–1536, 1565–1568）：
TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
while (!ts->buffer_is_readable(buf)) sleep(100us);
if (rank == 0) { ts->set_buffer_readable(buf, false); ts->set_buffer_writeable(buf, true); }

// 修改后：
TransferStation* ts = nullptr;
for (int w = 0; w < 200; ++w) {
    ts = static_cast<TransferStation*>(registry.transfer_station_ptr(rank));
    if (ts) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);
// ...
ts->wait_buffer_readable(buf);
// ...
ts->set_buffer_readable(buf, false);
ts->set_buffer_writeable(buf, true);
```

### 3.5 run_gpu() 清理

删除 **PRE-TEST 块**（lines 720–745）：预启动 DEEP + OPT 会修改权重，干扰训练初始状态。

```cpp
// 删除以下代码：
{
    cudaSetDevice(gpu_exec_.device_ids[0]);
    // ... PRE-TEST launch DEEP + OPT ...
    LOG_INFO << "[PRE-TEST] w13[0] after direct launch=" << w0;
}
```

### 3.6 run_h2d_only_train_epoch() 无需修改

用户要求 `compile_h2d_only()+run_h2d_only()` 及其测试样例继续正常运行。当前 `run_h2d_only_train_epoch()` 和 `run_h2d_only_val_epoch()` 的代码已经正确，不需要修改。其中包含的 `TR_TEST_TWO_BATCH_CORRECTION` 宏只影响 h2d-only 测试，与 `compile()+run()` 无关。

---

## 4. 修改清单

### 文件：src/task/deep_learning_task.cpp

| 行号范围 | 操作 | 说明 |
|----------|------|------|
| 720–745 | **删除** | PRE-TEST 块 |
| 881–1453 | **重写** | `run_train_epoch_gpu()` 完整替换为新实现（约 200 行） |
| 1474 | **修改** | `transfer_station_ptr(0)` → 每个 rank 独立获取 |
| 1535–1536 | **修改** | `while(sleep)` → `wait_buffer_readable()` |
| 1565–1568 | **修改** | `if(rank==0)` → 所有 rank |
| 894–913 | **删除** | loss/scaling magic value 初始化（在 run_train_epoch_gpu 新实现中不包含） |

### 文件：include/renaissance/task/deep_learning_task.h

| 操作 | 说明 |
|------|------|
| **无修改** | 不需要新增方法，所有改动在 `.cpp` 内部完成 |

---

## 5. 实施步骤

```
Step 1: 备份当前 deep_learning_task.cpp

Step 2: 重写 run_train_epoch_gpu()
        ├── 删除 lines 720–745（PRE-TEST）
        ├── 删除 lines 881–1453（旧 run_train_epoch_gpu）
        └── 写入新实现（保留 lines 932–960 的图获取和 stream 逻辑）

Step 3: 微调 run_val_epoch_gpu()
        ├── 修改 TransferStation 获取（line 1474）
        ├── 修改 buffer 等待（lines 1535–1536）
        └── 修改 set_buffer（lines 1565–1568）

Step 4: 编译验证
        ├── ninja 编译通过
        ├── 运行 test_two_batch_correction（确保 h2d_only 未受损）
        └── 运行 test_dl_full（验证目标）

Step 5: 回归测试
        ├── test_mlp_final（SimpleTask 不受影呴，确认）
        ├── test_two_batch_correction GPU/AMP/CPU
        └── test_dl_full GPU（MNIST 3 epoch，best_top1 > 85%）
```

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| label_smce D2D 拷贝后 loss 仍异常 | 中 | 需排查计算图 | 如仍异常，说明问题在 DEEP_FWD_BWD graph 内部，而非传输层；可用 test_mlp_final 的 SimpleTask 数值对比定位 |
| FIRST_LAYER_BWD_B 为 null 导致回退无效 | 低 | 首层反向缺失 | 已在代码中加入 `if (!g_first_cur) g_first_cur = g_first` 回退；若两者皆为 null，则训练不完整，需检查 Compiler 是否生成了 FIRST_BWD |
| 移除诊断代码后难以调试 | 低 | 排查困难 | 保留核心日志（epoch 序号、batch 数、最终 loss/top1）；如需诊断可临时加回 |
| 多卡下 wait_buffer_readable 死锁 | 低 | 训练 hang | `run_h2d_only` 已验证 8 卡通过；新实现使用完全相同的 wait/set 语义 |
| 删除 loss=3.14f 初始化导致首 batch loss 异常 | 低 | 首 loss 值不可预期 | 3.14f 本就是不合理的 magic value；正确的 loss 应由 SoftmaxCE atomicAdd 从 0 累加得到 |

---

## 7. 验证计划

### 7.1 立即验证（Step 4）

```bash
# H2D-only 回归（确保未受损）
test_two_batch_correction.exe --gpu
test_two_batch_correction.exe --amp
test_two_batch_correction.exe --cpu

# 目标测试
test_dl_full.exe
```

### 7.2 完整验证（Step 5）

| 测试 | 预期结果 |
|------|----------|
| `test_mlp_final --gpu` | PASS（SimpleTask 不受影响） |
| `test_mlp_final --amp` | PASS |
| `test_two_batch_correction --gpu --dataset mnist` | PASS |
| `test_two_batch_correction --gpu --dataset cifar10` | PASS |
| `test_dl_full` | MNIST 3 epoch, best_top1 > 85%, 输出 PASS |

---

## 8. 开放问题

1. **`g_first_b` (FIRST_LAYER_BWD_B) 的可靠性**：如果 Compiler 对当前模型未生成 FIRST_BWD_B，`g_first_b` 为 null，回退到 `g_first` 后 A/B batch 使用同一张 captured graph。这是否安全取决于 graph 内部是否硬编码了 buffer 地址。若不安全，则需要在 Compiler 层确保 FIRST_BWD_B 始终生成。

2. **`DEEP_FWD_BWD` 的 A/B 复用**：`FWD_BWD_DEEP_A` 和 `FWD_BWD_DEEP_B` 都 resolve 到同一个 `GraphId::DEEP_FWD_BWD`。这意味着两者使用同一张 captured graph。如果 graph 内部通过 kernel 参数动态传入 data/label 地址，则 A/B 复用是安全的；如果地址硬编码在 capture 时，则 A/B 都会读到同一套数据。当前代码已如此运行，若存在问题则不在本次传输层修复范围内。

3. **CPU 路径**：`run_cpu()` 当前为 stub。`test_dl_full` 使用 GPU 路径，CPU 路径可在 GPU 路径稳定后独立实现。

---

## 9. 总结

| 模块 | 改动量 | 说明 |
|------|--------|------|
| `compile()` / `compile_impl()` | **0 行** | 已正确编译全部图 |
| `build_graph_atlas()` | **0 行** | 已正确注册全部 GraphId |
| `build_exec_table()` | **0 行** | 已正确解析全部 slot |
| `run_gpu()` | **-26 行** | 删除 PRE-TEST |
| `run_train_epoch_gpu()` | **-570 + ~180 行** | 删除诊断代码，修复 H2D 子系统 6 处差异 |
| `run_val_epoch_gpu()` | **~10 行修改** | 修复 buffer 管理和 ts 获取 |
| `run_h2d_only_*()` | **0 行** | 保持现状 |
| **总计** | **约 -400 行** | 大量删除诊断代码，新增核心修复 |

**一句话**：保留原有 pipeline 并行结构，在每个 XFER 完成点逐行对齐 `run_h2d_only_train_epoch()` 的 H2D 语义（wait → launch → sync → set_buffer → label_smce D2D → sync），计算图启动顺序和 stream 分配完全不变，移除所有诊断代码和 magic value 初始化。
