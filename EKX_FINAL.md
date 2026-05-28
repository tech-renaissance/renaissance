# EKX_FINAL.md — 最终综合方案：重写 `compile()` + `run()`

## 0. 审阅说明

本文综合了 **EKX1.md、EKX2.md、EKX3.md** 三个方案，并基于对实际源码（`deep_learning_task.cpp` 2034 行、`task_base.cpp` 1526 行、相关头文件）的逐行复查，给出最终可实施方案。

**三篇文章已在以下问题上达成共识**：
- `compile()` 不需要任何修改 ✅
- 诊断代码必须全部移除 ✅
- Buffer 管理必须所有 rank 参与 ✅
- PRE-TEST 块必须删除 ✅
- `compile_h2d_only()` + `run_h2d_only()` 不动 ✅

**修正项（基于最新代码复查）**：
- `label_smce` **并非缺失**——`compiler.cpp` 已在 `FIRST_FWD_A/B` 图中注入 `DTensorCopy(label_a/b → label_smce)` 节点，完整训练由 CUDA Graph 自动处理。`h2d_only` 中手动 D2D 拷贝是因为不启动 FIRST_FWD 图。EKX1/EKX2/EKX3 将"label_smce 缺失"列为致命根因的结论**有误**，需修正。
- **命名不统一**：GraphId 为 `FIRST_FWD_A/B` + `FIRST_BWD/BWD_B`，GraphSlot 为 `FIRST_FWD_A/B` + `FIRST_LAYER_BWD/BWD_B`。需统一为 `FIRST_LAYER_FWD_A/B` + `FIRST_LAYER_BWD_A/B`。

**三篇文章的核心分歧只有一个**：Phase 1 采用"统一串行循环"（EKX1/EKX3）还是"保留流水线并行"（EKX2）。

**本文的裁决**：**保留流水线并行（EKX2 路线）作为最终方案**。理由见第 4 节。

---

## 1. 三方案对比

| 维度 | EKX1 | EKX2 | EKX3 |
|------|------|------|------|
| **循环结构** | 统一 `for (0..batches-1)`，先 H2D 再计算 | 保留 3 分支（预传输/循环/last batch），H2D+计算并行 | 同 EKX1：统一循环 |
| **H2D-计算关系** | 串行（无重叠） | **并行（Phase 2: DEEP ‖ XFER）** | 串行（无重叠） |
| **改动风格** | 完全重写 ~170 行 | 修复现有结构 ~180 行 | 完全重写 ~150 行 |
| **差异识别数** | 5 处 | **5 处**（label_smce 由 FIRST_FWD 图处理，不属差异；增加了 ts 获取方式 + magic value 初始化） | 5 处 |
| **FIRST_LAYER_BWD_B 处理** | null check 无 fallback | null check 无 fallback（回退到 g_first 不安全：A/B buffer 不同） | null check 无 fallback |
| **代码行数变化** | -570 + 170 行 | -570 + 180 行 | -570 + 150 行 |
| **性能** | 较低（无并行） | **高（保持并行）** | 较低（无并行） |
| **fetch_from_rank** | 正确：`fetch_from_rank(loss_dt, 0)` | 正确：`fetch_from_rank(loss_dt, 0)` | 错误：签名不匹配 |

---

## 2. 源码复查确认的完整差异清单

通过逐行比对 `run_h2d_only_train_epoch()`（lines 2148–2324）与 `run_train_epoch_gpu()`（lines 881–1453），确认 **5 处必须修复的差异**：

| # | 差异 | `run_h2d_only` ✅ | `run_train_epoch_gpu` ❌ | 致命性 |
|---|------|-------------------|--------------------------|--------|
| 1 | **Buffer 可读等待** | `wait_buffer_readable()` 条件变量阻塞 | `while(!buffer_is_readable) sleep(100us)` 自旋（lines 967–968, 1147–1148） | 🟡 高 |
| 2 | **Buffer 状态管理** | **所有 rank** 调用 `set_buffer_*`（lines 2222–2223） | **仅 rank 0** 调用（lines 1029–1030, 1231–1234） | 🟡 高 |
| 3 | **TransferStation 获取** | 每个 rank `transfer_station_ptr(rank)`（line 2188） | 主线程 `transfer_station_ptr(0)` 后共享（line 885） | 🟡 高 |
| 4 | **诊断代码污染** | 零诊断 | ~500 行 DIAG-* 散布热路径 | 🟠 中 |
| 5 | **magic value 初始化** | 无 | `loss=3.14f`, `scaling=1.0f` 直接写 device（lines 894–913） | 🟠 中 |

**关于 label_smce 的重要修正**：

- `h2d_only` 中手动 `cudaMemcpyAsync(label_smce_ptr, ...)` 是必需的，因为 `h2d_only` 只启动 XFER 图、不启动 FIRST_FWD 图，label 从 staging 传到 `I_A/B_LABEL` 后不会自动进入 `label_smce`。
- **完整训练不需要手动 D2D 拷贝**：`compiler.cpp`（lines 999–1017）在 `GraphId::FIRST_FWD_A` 和 `GraphId::FIRST_FWD_B` 中分别注入了 `DTensorCopy(I_A_LABEL → label_smce)` 和 `DTensorCopy(I_B_LABEL → label_smce)` 节点。当 `cudaGraphLaunch(g_fwd_a/b, s_c1)` 执行时，label_smce 的拷贝作为 FIRST_FWD 图的一部分自动完成，与后续 DEEP_FWD_BWD 在 `s_c1` 上的顺序执行保证了 label_smce 在 SoftmaxCE 读取前已就绪。
- **结论**：`label_smce` 不是 `run_train_epoch_gpu()` 的缺失项，而是**已由 FIRST_FWD_A/B 图处理**。EKX 三篇文章将"label_smce D2D 缺失"列为致命根因的结论**有误**，应移除所有手动 `cudaMemcpyAsync(label_smce_ptr, ...)`。

**补充说明**：

- **差异 #3（TransferStation）的补充**：代码复查确认 `GlobalRegistry::transfer_station_ptr()` 返回 `void*`，需要 `static_cast<TransferStation*>()`。当前代码（line 885）从主线程获取 `ptr(0)` 后闭包捕获，所有 rank 线程共享同一个 `ts` 指针。这在单卡下偶然正确（所有 rank 的 Preprocessor worker 共享同一个 TransferStation），但多卡下每个 rank 可能有独立的 TransferStation。

- **差异 #5（magic value）的补充**：`loss=3.14f` 会覆盖 SoftmaxCrossEntropy 的 `atomicAdd(0, ...)` 累加器，导致首次 loss 值错误；`scaling=1.0f` 可能影响 AMP 的 loss scaling 逻辑。

---

## 3. `run_val_epoch_gpu()` 的差异

`run_val_epoch_gpu()`（lines 1469–1606）同样存在 3 处差异：

| # | 差异 | `run_h2d_only_val_epoch` ✅ | `run_val_epoch_gpu` ❌ |
|---|------|---------------------------|------------------------|
| V1 | Buffer 可读等待 | `wait_buffer_readable()` | `while(sleep)`（line 1535–1536） |
| V2 | set_buffer | 所有 rank（line 2400–2401） | 仅 rank 0（lines 1565–1568） |
| V3 | TransferStation | `transfer_station_ptr(rank)` | `transfer_station_ptr(0)`（line 1474） |

验证路径不需要 `label_smce` 拷贝（验证不计算 loss 梯度，只做推理；且 `INF_MAIN_A/B` 推理图已由 Compiler 注入 `DTensorCopy(I_A/B_LABEL → label_smce)`，与训练一致）。需要把 `while(sleep)` 改为 `wait_buffer_readable()` 以对齐 h2d_only。

---

## 4. 核心决策：流水线并行 vs 统一串行

### 4.1 两种方案对比

**方案 A（EKX1/EKX3 — 统一串行）**：
```cpp
for (int batch = 0; batch < batches; ++batch) {
    // 传输
    wait_buffer → XFER → sync → set_buffer → label D2D
    // 计算
    ZG → FWD → DEEP → BWD → AR → OPT
}
```
- 简洁，与 h2d_only 循环结构完全一致
- 验证正确性最快
- **缺点**：无 H2D-计算并行，性能降低约 30–50%

**方案 B（EKX2 — 保留流水线）**：
```
Batch 0 预传输：wait → XFER_A → sync → set_buffer
循环 batch=0..batches-2：
  Phase 1: ZG ‖ FIRST_LAYER_FWD（当前 batch，内含 label_smce DTensorCopy）
  Phase 2: DEEP ‖ XFER(next batch)   ← H2D 与计算并行
  Phase 3: FIRST_LAYER_BWD_A/B ‖ DEEP_AR
  Phase 4: LR → FIRST_AR → OPT
Last batch：ZG → FWD → DEEP → BWD → AR → OPT（无 XFER）
```
- 保持原始设计意图（H2D-计算重叠）
- 改动量最小（修复式而非重写式）
- **缺点**：3 分支结构更复杂

### 4.2 裁决：方案 B

**理由**：

1. **用户明确要求性能重要**（"性能、数学正确性……都非常重要"）。方案 A 将 H2D 和计算完全串行化，对 MNIST 而言影响不大（3 epoch ~数秒），但对 CIFAR10 / ImageNet 影响显著。保留流水线结构避免了后续"重新并行化"的二次改动风险。

2. **改动更小、更安全**。方案 B 不改变原有的 Phase 顺序、Stream 分配、同步点拓扑结构。它只修复 buffer 管理、TransferStation 获取和诊断代码，计算图部分（ZG / FWD / DEEP / BWD / AR / OPT / GC / LR / FAR / WU）的启动顺序和 stream 完全不变。label_smce 由 FIRST_FWD 图内部处理，无需手动 D2D。

3. **流水线结构本身的正确性**：在 Batch 循环中，Phase 2 的 `DEEP(current) ‖ XFER(next)` 确保了：
   - 当前 batch 的 DEEP 在 `COMP_1` 上执行完成（`sync_comp()`）
   - 下一 batch 的 XFER 在 `TRANS` 上执行完成（`sync_tr()`）
   - 两条流并发执行，无数据竞争：XFER 写入 `I_B_DATA`（或 `I_A_DATA`），DEEP 读取 `I_A_DATA`（或 `I_B_DATA`），它们是不同的 GPU buffer

4. **验证策略**：如果方案 B 不通过，可以快速切换到方案 A 来二分法定位问题是出在"传输修复是否足够"还是"流水线同步是否有额外 race"。

---

## 5. 最终实现方案

### 5.1 修改文件

**核心修改文件**：`src/task/deep_learning_task.cpp`

**配套修改文件**（命名统一，必须同步修改）：
- `include/renaissance/graph/computation_graph.h`：GraphId 重命名 `FIRST_FWD_A/B` → `FIRST_LAYER_FWD_A/B`，`FIRST_BWD` → `FIRST_LAYER_BWD_A`，`FIRST_BWD_B` → `FIRST_LAYER_BWD_B`
- `src/graph/compiler.cpp`：所有 `GraphId::FIRST_FWD_A/B` 和 `GraphId::FIRST_BWD/BWD_B` 引用重命名
- `src/backend/graph_executor.cpp`：所有 `GraphId::FIRST_FWD_A/B` 和 `GraphId::FIRST_BWD/BWD_B` 引用重命名（⚠️ 遗留代码，非当前活跃执行路径，但需保持编译通过）
- `src/task/deep_learning_task.cpp`：
  - 匿名 namespace 中 GraphSlot 重命名（`FIRST_FWD_A/B` → `FIRST_LAYER_FWD_A/B`，`FIRST_LAYER_BWD` → `FIRST_LAYER_BWD_A`）
  - `stream_for()` 显式增加 `FIRST_LAYER_BWD_B` case（当前靠 default 返回 COMP_1）
  - 所有 GraphId/GraphSlot 引用更新

**不修改**：
- `src/task/task_base.cpp`（`compile_impl` 不变）
- `run_h2d_only_*()`（保持现状，确保 H2D-only 测试通过）

### 5.2 修改 A：重写 `run_train_epoch_gpu()`（lines 881–1453）

**操作**：删除 lines 881–1453（约 570 行），替换为约 180 行的新实现。

**新实现**（保留原始流水线结构，修复 6 处差异）：

```cpp
float DeepLearningTask::run_train_epoch_gpu() {
#ifdef TR_USE_CUDA
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    int32_t loss_id = active_memory_plan_->baseline().loss;

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, K, using_amp, &exc, loss_id]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess)
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));

                // ===== Graph handles（保留原有逻辑）=====
                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];

                auto g_xfer_a  = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b  = g_tab[S(GraphSlot::XFER_B)];
                auto g_deep_a  = g_tab[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto g_deep_b  = g_tab[S(GraphSlot::FWD_BWD_DEEP_B)];
                auto g_first   = g_tab[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto g_first_b = g_tab[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto g_far     = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto g_zg      = g_tab[S(GraphSlot::ZERO_GRAD)];
                auto g_dar     = g_tab[S(GraphSlot::DEEP_ALLREDUCE)];
                auto g_wu      = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
                auto g_gc      = g_tab[S(GraphSlot::GRAD_CONVERT)];
                auto g_fwd_a   = g_tab[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto g_fwd_b   = g_tab[S(GraphSlot::FIRST_LAYER_FWD_B)];

                // ===== Streams（保留原有逻辑）=====
                DeviceContext& ctx = context(rank);
                cudaStream_t s_up    = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr   = [&]() { cudaStreamSynchronize(s_trans); };

                // ===== TransferStation（修复 #4：每个 rank 独立获取）=====
                TransferStation* ts = nullptr;
                for (int w = 0; w < 200; ++w) {
                    ts = static_cast<TransferStation*>(
                        registry.transfer_station_ptr(rank));
                    if (ts) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

                float lr;
                float* lr_dev_ptr = static_cast<float*>(ctx.ptr_at(lr_dtensor_id_));
                bool frozen = is_first_layer_frozen();

                // ================================================================
                // Batch 0 预传输
                // ================================================================
                // 修复 #2（wait_buffer_readable 替代 while-sleep）
                ts->wait_buffer_readable(0);

                if (g_xfer_a) cudaGraphLaunch(g_xfer_a, s_trans);
                sync_tr();

                // 修复 #2（所有 rank set_buffer）
                ts->set_buffer_readable(0, false);
                ts->set_buffer_writeable(0, true);

                // ===== 单 batch 边界 =====
                if (batches == 1) {
                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd_a) cudaGraphLaunch(g_fwd_a, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_a) cudaGraphLaunch(g_deep_a, s_c1);
                    sync_comp();

                    if (!frozen && g_first) cudaGraphLaunch(g_first, s_c1);
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
                // 循环：batch = 0 .. batches-2（保留原有流水线结构）
                // ================================================================
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;
                    auto g_fwd   = from_a ? g_fwd_a : g_fwd_b;
                    auto g_deep  = from_a ? g_deep_a : g_deep_b;
                    auto g_xfer_n = from_a ? g_xfer_b : g_xfer_a;

                    // Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD（计算当前 batch）
                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                    sync_comp(); sync_up();

                    // 修复 #1：wait_buffer_readable 替代 while-sleep
                    ts->wait_buffer_readable(next_buf);

                    // Phase 2: DEEP_FWD_BWD ‖ XFER(next)
                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
                    sync_comp(); sync_tr();

                    // 修复 #2（所有 rank set_buffer）
                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);

                    // Phase 3: FIRST_LAYER_BWD_A/B ‖ DEEP_ALLREDUCE
                    if (!frozen) {
                        auto g_first_cur = from_a ? g_first : g_first_b;
                        if (g_first_cur) cudaGraphLaunch(g_first_cur, s_c1);
                    }
                    if (g_dar) cudaGraphLaunch(g_dar, s_up);
                    sync_comp(); sync_up();

                    // Phase 4: GRAD_CONVERT（AMP only）→ LR → FIRST_AR → OPT
                    if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batch);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_far) cudaGraphLaunch(g_far, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up);
                    sync_up();
                }

                // ================================================================
                // Last batch (batch = batches-1)：无 XFER（数据已在循环最后一轮传完）
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

    // 读取 train loss
    float train_loss = 0.0f;
    if (loss_id >= 0) {
        const auto& loss_dt = active_memory_plan_->get_dtensor(loss_id);
        Tensor h_loss = fetch_from_rank(loss_dt, 0);
        train_loss = h_loss.data<float>()[0];
    }
    return train_loss;
#else
    (void)0;
    return 0.0f;
#endif
}
```

**与旧实现的逐段对比**：

| 代码段 | 旧实现 | 新实现 | 修复的差异 |
|--------|--------|--------|-----------|
| 变量声明 | ~40 行，含 magic value 初始化 | ~20 行，无 magic value | #5 |
| TransferStation 获取 | `ptr(0)` 主线程获取后共享 | `ptr(rank)` 每 rank 带重试 | #3 |
| Batch 0 预传输 wait | `while(!buffer_is_readable) sleep(100us)` | `wait_buffer_readable(0)` | #1 |
| Batch 0 set_buffer | `if (rank == 0) { ... }` + ~150 行诊断 | 所有 rank `set_buffer_*` | #2, #4 |
| 循环 wait next | `while(!buffer_is_readable) sleep(100us)` | `wait_buffer_readable(next_buf)` | #1 |
| 循环 Phase 2 后 | `if (rank == 0) set_buffer` + ~300 行诊断 | 所有 rank `set_buffer_*` | #2, #4 |
| Loss 读取 | `cudaMemcpy + LOG_INFO` 诊断 | `fetch_from_rank(loss_dt, 0)` | #4 |
| Weight 诊断 | `cudaMemcpy w13 BEFORE/AFTER` | 删除 | #4 |

### 5.3 修改 B：微调 `run_val_epoch_gpu()`

**位置**：lines 1474, 1535–1536, 1565–1568

**修改前**：
```cpp
// line 1474
TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));

// lines 1535–1536
while (!ts->buffer_is_readable(buf))
    std::this_thread::sleep_for(std::chrono::microseconds(100));

// lines 1565–1568
if (rank == 0) {
    ts->set_buffer_readable(buf, false);
    ts->set_buffer_writeable(buf, true);
}
```

**修改后**：
```cpp
// line 1474 → 替换为：
TransferStation* ts = nullptr;
for (int w = 0; w < 200; ++w) {
    ts = static_cast<TransferStation*>(registry.transfer_station_ptr(rank));
    if (ts) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

// lines 1535–1536 → 替换为：
ts->wait_buffer_readable(buf);

// lines 1565–1568 → 替换为：
ts->set_buffer_readable(buf, false);
ts->set_buffer_writeable(buf, true);
```

### 5.4 修改 C：删除 `run_gpu()` 中的 PRE-TEST

**位置**：lines 720–745

**操作**：删除整个 `{ cudaSetDevice(...); ... LOG_INFO << "[PRE-TEST] w13[0] ..."; }` 块。

### 5.5 不修改的部分

| 代码 | 行号 | 原因 |
|------|------|------|
| `compile()` / `compile_impl()` | task_base.cpp:204–293 | 已正确，无需修改 |
| `build_graph_atlas()` | deep_learning_task.cpp:480–534 | 已正确，无需修改 |
| `build_exec_table()` | deep_learning_task.cpp:573–671 | **需修改**：`kRequired` 增加 `FIRST_LAYER_BWD_B`（见第 7 节） |
| `run_h2d_only_train_epoch()` | deep_learning_task.cpp:2148–2324 | 保持现状，确保 H2D-only 测试通过 |
| `run_h2d_only_val_epoch()` | deep_learning_task.cpp:2326–2484 | 保持现状 |
| `compile_h2d_only()` | deep_learning_task.cpp:2135–2146 | 保持现状 |
| `run_h2d_only()` | deep_learning_task.cpp:2486–2510 | 保持现状 |

---

## 6. 完整修改清单

| 文件 | 行号/范围 | 操作 | 说明 |
|------|-----------|------|------|
| `include/renaissance/graph/computation_graph.h` | ~76–81 | **重命名** | GraphId: `FIRST_FWD_A`→`FIRST_LAYER_FWD_A`, `FIRST_FWD_B`→`FIRST_LAYER_FWD_B`, `FIRST_BWD`→`FIRST_LAYER_BWD_A`, `FIRST_BWD_B`→`FIRST_LAYER_BWD_B` |
| `include/renaissance/graph/computation_graph.h` | ~106–111, ~158–163 | **重命名** | `graph_id_to_string` / `is_train_graph` 中对应 switch case |
| `src/task/deep_learning_task.cpp` | ~34–48 | **重命名** | 匿名 namespace 中 GraphSlot: `FIRST_FWD_A`→`FIRST_LAYER_FWD_A`, `FIRST_FWD_B`→`FIRST_LAYER_FWD_B`, `FIRST_LAYER_BWD`→`FIRST_LAYER_BWD_A` |
| `src/task/deep_learning_task.cpp` | ~536–555 | **增加** | `stream_for()` 显式增加 `case GraphId::FIRST_LAYER_BWD_B: return StreamKind::COMP_1;` |
| `src/graph/compiler.cpp` | ~996, ~997, ~1141, ~1185, ~1187, ~1207, ~1282–1298 | **重命名** | 所有 `GraphId::FIRST_FWD_A/B` / `GraphId::FIRST_BWD/BWD_B` 引用 |
| `src/backend/graph_executor.cpp` | ~93, ~114, ~148, ~165 | **重命名** | 所有 `GraphId::FIRST_FWD_A/B` / `GraphId::FIRST_BWD/BWD_B` 引用（⚠️ 遗留代码，保持编译即可） |
| `src/task/deep_learning_task.cpp` | ~540–543, ~616–617, ~624–625, ~635–637, ~644–648, ~661–668 | **重命名** | build_graph_atlas/build_exec_table 中 GraphId/GraphSlot 引用 |
| `src/task/deep_learning_task.cpp` | ~720–745 | **删除** | PRE-TEST 块 |
| `src/task/deep_learning_task.cpp` | ~881–1453 | **替换** | `run_train_epoch_gpu()` 完整重写（~570 → ~150 行，移除诊断+label_smce手动拷贝） |
| `src/task/deep_learning_task.cpp` | ~1474 | **替换** | TransferStation 获取：`ptr(0)` → `ptr(rank)` 带重试 |
| `src/task/deep_learning_task.cpp` | ~1535–1536 | **替换** | `while(sleep)` → `wait_buffer_readable()` |
| `src/task/deep_learning_task.cpp` | ~1565–1568 | **替换** | `if(rank==0) set_buffer` → 所有 rank |
| `src/task/deep_learning_task.cpp` | ~644–648 | **修改** | `kRequired`：将 `FIRST_LAYER_BWD` 重命名为 `FIRST_LAYER_BWD_A`，并**增加** `FIRST_LAYER_BWD_B` |
| **净变化** | | | **约 -420 行**（删除诊断代码），命名统一触及 4 个文件 |

---

## 7. 关于 `FIRST_LAYER_BWD_B` null 的说明

当前 `build_exec_table()` 中 `kRequired` 只检查 `FIRST_LAYER_BWD`（未重命名时的旧名，对应 A 侧），不检查 `FIRST_LAYER_BWD_B`（line 644–648）。命名统一后，`FIRST_LAYER_BWD` 需同步改名为 `FIRST_LAYER_BWD_A`，并**增加** `FIRST_LAYER_BWD_B`。否则 `g_first_b`（`FIRST_LAYER_BWD_B` slot）**可能为 null**。

**分析**：
- `GraphId::FIRST_LAYER_BWD_A`（旧名 `FIRST_BWD`）写回 `I_A_DATA`
- `GraphId::FIRST_LAYER_BWD_B`（旧名 `FIRST_BWD_B`）写回 `I_B_DATA`
- 两张图由 `compiler.cpp`（lines 1185–1207）在首层 Flatten BWD 时分别生成，逻辑上应当同时存在
- 若 `FIRST_LAYER_BWD_B` 为 null，B 侧 batch 会跳过首层反向，导致首层（Flatten + FC1）梯度不更新，训练不完整

**修改决定**：将 `FIRST_LAYER_BWD_B`（旧名 `FIRST_LAYER_BWD_B`）加入 `kRequired`（line 644–648），确保所有模型在编译期就检查 B 侧首层反向图是否成功捕获。

**不加回退逻辑**：旧实现（line 1352）用 `if (g_first_cur) cudaGraphLaunch(g_first_cur, s_c1)` null check + 跳过。若 BWD_B 为 null 则跳过，不会误用 BWD_A 写回 A buffer（不安全）。回退到 `g_first`（BWD_A）会导致 B 侧 batch 的梯度覆盖 A buffer，破坏数据一致性。加入 `kRequired` 后，BWD_B 缺失会在编译期直接报错，不会静默跳过。

---

## 8. 实施步骤

```
Step 1: 命名统一（4 个文件全局替换）
        ├── include/renaissance/graph/computation_graph.h
        ├── src/graph/compiler.cpp
        ├── src/backend/graph_executor.cpp
        └── src/task/deep_learning_task.cpp
        └── 编译验证通过

Step 2: 重写 run_train_epoch_gpu()（lines 881-1453）
        └── 删除旧实现，写入新实现（约 150 行，移除诊断代码+label_smce手动拷贝）

Step 3: 微调 run_val_epoch_gpu()（lines 1474, 1535-1536, 1565-1568）
        └── 3 处修改

Step 4: 修改 build_exec_table() kRequired（line 644-648）
        └── 增加 FIRST_LAYER_BWD_B

Step 5: 删除 run_gpu() PRE-TEST 块（lines 720-745）

Step 6: 编译
        └── cmake --build build --config Release

Step 7: 回归测试（确保 H2D-Only 未受损）
        ├── test_two_batch_correction --gpu --dataset mnist
        ├── test_two_batch_correction --gpu --dataset cifar10
        ├── test_two_batch_correction --amp --dataset mnist
        ├── test_two_batch_correction --amp --dataset cifar10
        ├── test_two_batch_correction --cpu --dataset mnist
        └── test_two_batch_correction --cpu --dataset cifar10
        预期：全部 6 种组合 PASS

Step 6: 目标测试
        └── test_dl_full
        预期：MNIST 3 epoch, best_top1 > 85%, 输出 PASS
```

---

## 9. 验证矩阵

### 9.1 回归验证

| 测试 | 配置 | 预期 |
|------|------|------|
| test_two_batch_correction | MNIST GPU | PASS |
| test_two_batch_correction | MNIST AMP | PASS |
| test_two_batch_correction | MNIST CPU | PASS |
| test_two_batch_correction | CIFAR10 GPU | PASS |
| test_two_batch_correction | CIFAR10 AMP | PASS |
| test_two_batch_correction | CIFAR10 CPU | PASS |
| test_mlp_final | GPU | PASS（SimpleTask 不受影响） |

### 9.2 目标验证

| 测试 | 配置 | 预期 |
|------|------|------|
| test_dl_full | MNIST GPU 3 epochs | best_top1 > 85% |

### 9.3 调试策略（如果 test_dl_full 不通过）

```
Level 1: 二分法
  └── H2D only（注释计算）→ 应与 test_two_batch_correction 一致

Level 2: 逐步加计算图
  ├── H2D + ZG + FIRST_LAYER_FWD    → 检查首层输出
  ├── + DEEP_FWD_BWD                 → 检查 loss 值（应 > 0 且正常范围）
  ├── + FIRST_LAYER_BWD_A/B          → 检查梯度非零
  └── + ALLREDUCE + OPT              → 检查权重变化、loss 下降趋势

Level 3: 切换到统一串行循环（EKX1 方案）
  └── 如果流水线版本不通过但串行版本通过 → 问题在流水线同步
  └── 如果两者都不通过 → 问题在计算图本身
```

---

## 10. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 修复 5 处差异后 loss 仍异常 | 中 | 需排查计算图 | 二分法定位（见 9.3） |
| `FIRST_LAYER_BWD_B` 为 null 导致 B 侧 batch 缺少反向 | 低 | 训练不完整 | **已在 kRequired 中强制检查**，编译期报错 |
| 多卡下 `wait_buffer_readable` 时序 | 低 | 死锁 | h2d_only 已验证 8 卡，逻辑复用 |
| 删除 magic value 初始化导致首 batch loss 异常 | 低 | 首 loss 值不可预期 | 3.14f 本就不合理；正确 loss 由 SoftmaxCE atomicAdd 从 0 累加 |
| 命名统一引入编译错误 | 低 | 编译失败 | Grep 全局替换 + 编译验证，4 个文件同步修改 |

---

## 11. 总结

| 模块 | 改动量 | 说明 |
|------|--------|------|
| `compile()` 相关 | **0 行** | 已正确，不动 |
| `run_h2d_only_*()` | **0 行** | 保持现状，确保回归 |
| `run_gpu()` | **-26 行** | 删除 PRE-TEST |
| `run_train_epoch_gpu()` | **-570 + ~150 行** | 保留流水线并行结构，修复 5 处差异，移除 label_smce 手动拷贝 |
| `run_val_epoch_gpu()` | **~12 行修改** | 修复 3 处差异 |
| `include/renaissance/graph/computation_graph.h` | **~10 行重命名** | GraphId 统一命名 |
| `src/task/deep_learning_task.cpp`（匿名 namespace） | **~4 行重命名** | GraphSlot 统一命名 |
| `src/graph/compiler.cpp` | **~10 行重命名** | GraphId 引用更新 |
| `src/backend/graph_executor.cpp` | **~4 行重命名** | GraphId 引用更新 |
| `src/task/deep_learning_task.cpp` | **~20 行重命名** | GraphId/GraphSlot 引用更新 |
| **总计** | **约 -420 行** | |

**核心原则**：保留原有流水线并行结构（Batch 0 预传输 + batch 循环 + Last batch），在 XFER 完成点逐行对齐 `run_h2d_only_train_epoch()` 的 H2D 语义（`wait_buffer_readable` → `cudaGraphLaunch(XFER)` → `sync_tr` → `set_buffer`(所有 rank) → `sync_tr`），计算图启动顺序和 stream 分配完全不变，移除全部诊断代码和 magic value 初始化。label_smce 由 `FIRST_LAYER_FWD_A/B` 图内部的 `DTensorCopy` 节点自动处理，无需手动 `cudaMemcpyAsync`。