# EHR1.md — compile()+run() 彻底重写方案

## 1. 背景与目标

### 1.1 当前状态

| 路径 | 编译 | 运行 | 状态 |
|------|------|------|------|
| `compile_h2d_only()` + `run_h2d_only()` | 仅编译 TRANSFER_A/B 图 | 仅执行 H2D 传输 | ✅ 所有数据集/场景/多RANK 均通过正确性和性能验证 |
| `compile()` + `run()` | 编译全部图（18+ GraphSlot） | 执行完整训练流水线 | ❌ 结果存在问题，test_dl_full.cpp 无法通过 |

### 1.2 目标

以 `compile_h2d_only()` + `run_h2d_only()` 为基础，彻底重写 `compile()` + `run()`，使得：

1. **传输行为完全一致**：新版 `compile()` + `run()` 的 H2D 传输行为（Buffer 管理、CUDA Graph 启动、流同步、Label 拷贝）与 `compile_h2d_only()` + `run_h2d_only()` **完全一致**。
2. **叠加计算能力**：在传输基础之上，加入 ZERO_GRAD、FIRST_FWD、DEEP_FWD_BWD、FIRST_BWD、ALLREDUCE、OPTIMIZER、INFERENCE 等计算图的执行。
3. **最终目标**：跑通 `test_dl_full.cpp`（DeepLearningTask 版，3 epoch MNIST MLP，Top-1 > 85%）。

---

## 2. 当前代码架构深度分析

### 2.1 `compile_h2d_only()` 分析

```cpp
// deep_learning_task.cpp:2135-2146
void DeepLearningTask::compile_h2d_only() {
    TR_CHECK(phase_ == Phase::PLANNING, ...);
    struct Guard { bool* p; Guard(bool* ptr) : p(ptr) { *p = true; } ~Guard() { *p = false; } } guard(&h2d_only_);
    compile();
}
```

**核心机制**：RAII guard 设置 `h2d_only_ = true`，然后调用 `compile()`。`compile()` → `compile_impl()` 的完整流程如下：

```
compile_freeze_global()
  → compile_invoke_on_prepare()     // 触发 on_prepare()，生成完整 IR（5 阶段编译）
  → compile_verify_memory_locked()  // 验证 memory plan
  → compile_alloc_hardware()        // 分配 GPU 内存（所有 tensor，不区分 h2d_only）
  → build_graph_atlas()             // ★ h2d_only_=true → 只收录 TRANSFER_A/B
  → pre_capture(atlas, ctx_ptrs)    // ★ 只 capture TRANSFER_A/B 两个 graph
  → build_exec_table()              // ★ h2d_only_=true → 只解析 XFER_A/B slot
  → compile_mark_compiled()
  → init_all() + cudaMallocHost(lr_pinned_)
```

**关键结论**：`compile()` 阶段，无论 `h2d_only_` 是 true 还是 false：
- `on_prepare()` 生成的 IR **完全相同**（均为完整蓝图编译）
- `compile_alloc_hardware()` 分配的内存 **完全相同**
- 唯一的差异在 `build_graph_atlas()` + `pre_capture()` + `build_exec_table()` 步骤中：
  - `h2d_only_=true`：只 capture 和 resolve TRANSFER_A/B
  - `h2d_only_=false`：capture 和 resolve 全部 18+ 个 GraphSlot

**这意味着**：当 `h2d_only_=false` 时，TRANSFER_A/B 的 CUDA Graph capture 流程与 `h2d_only_=true` 时**完全一致**（相同的 IR → 相同的 CUDA Graph），因为：
- `build_graph_atlas()` 中对 XFER 的处理代码路径完全相同（lines 484-495 vs 498-508）
- `stream_for(GraphId::TRANSFER_A)` = `StreamKind::TRANS`（lines 538-539）
- `build_exec_table()` 中 XFER_A/B 的 resolve 逻辑完全相同（lines 596-597 vs 612-613）

### 2.2 `run_h2d_only()` 分析

```cpp
// deep_learning_task.cpp:2486-2510
H2DRunResult DeepLearningTask::run_h2d_only() {
    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;
        H2DTestResult train_res = run_h2d_only_train_epoch();
        if (should_validate_this_epoch()) {
            H2DTestResult val_res = run_h2d_only_val_epoch();
        }
    }
}
```

#### 2.2.1 `run_h2d_only_train_epoch()` — GPU 路径（核心参考实现）

```cpp
// deep_learning_task.cpp:2178-2239 (GPU path)
for (int rank = 0; rank < K; ++rank) {
    threads.emplace_back([&, rank]() {
        cudaSetDevice(gpu_exec_.device_ids[rank]);
        // 获取 TransferStation（带重试）
        TransferStation* ts = ...;
        // 获取 XFER graph exec handles
        auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
        auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
        cudaStream_t s_trans = context(rank).stream(StreamKind::TRANS);
        // 获取 label pointer（用于 D2D copy 到 SMCE 槽位）
        auto label_ptr_a / label_ptr_b / label_smce_ptr
        size_t label_nbytes = ...

        for (int batch = 0; batch < batches; ++batch) {
            int buf_id = batch % 2;
            auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

            // ===== 核心传输三步（与新版 run() 必须保持一致）=====
            ts->wait_buffer_readable(buf_id);           // ① 等待 staging buffer 就绪
            cudaGraphLaunch(g_xfer, s_trans);           // ② 启动 CUDA Graph（H2D copy）
            cudaStreamSynchronize(s_trans);             // ③ 同步 TRANS 流
            ts->set_buffer_readable(buf_id, false);     // ④ 标记 buffer 已消费
            ts->set_buffer_writeable(buf_id, true);     // ⑤ 释放 buffer 给 Preprocessor
            // ===== 核心传输三步结束 =====

            // ⑥ Label D2D copy：从 I_A_LABEL/I_B_LABEL 拷贝到 LABEL_SMCE
            cudaMemcpyAsync(label_smce_ptr,
                (buf_id == 0) ? label_ptr_a : label_ptr_b,
                label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
            cudaStreamSynchronize(s_trans);
        }
    });
}
```

**传输模式特征**：

| 特征 | 值 |
|------|-----|
| Buffer 轮转 | `batch % 2` (A→B→A→B...) |
| XFER Graph 选择 | buf_id=0 → XFER_A, buf_id=1 → XFER_B |
| 流 | s_trans (StreamKind::TRANS) |
| 同步方式 | 每个 batch 后 `cudaStreamSynchronize(s_trans)` |
| Buffer 管理 | **所有 rank 都调用** `set_buffer_readable/set_buffer_writeable` |
| Label 处理 | D2D copy 到 LABEL_SMCE 槽位（SoftmaxCrossEntropy 的数据输入槽） |
| Prep 线程 | 函数内部 `std::thread prep_thread([&]() { prep.train(); })` |

#### 2.2.2 `run_h2d_only_val_epoch()` — GPU 路径（验证参考实现）

```cpp
// deep_learning_task.cpp:2360-2406 (GPU path)
for (int batch = 0; batch < val_batches; ++batch) {
    int buf_id = batch % 2;
    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

    ts->wait_buffer_readable(buf_id);
    cudaGraphLaunch(g_xfer, s_trans);
    cudaStreamSynchronize(s_trans);
    ts->set_buffer_readable(buf_id, false);
    ts->set_buffer_writeable(buf_id, true);
    // ⚠ 无 Label D2D copy（验证不需要 Label 参与 SoftmaxCE 的梯度计算）
}
```

### 2.3 现有 `compile()` + `run()` 分析

#### 2.3.1 `compile()` — 结论：**不需要改动**

`compile()` 调用 `compile_impl()`，由于 `h2d_only_` 默认为 `false`，走全量 GraphAtlas 路径：
- `build_graph_atlas()` 收录全部 GraphId（line 498-508）—— **包含 TRANSFER_A/B，路径与 h2d_only 一致**
- `build_exec_table()` 解析全部 18+ 个 slot（line 606-669）—— **包含 XFER_A/B，解析逻辑与 h2d_only 一致**

**`compile()` 已经正确编译了传输图和所有计算图，无需任何修改。**

#### 2.3.2 `run()` 入口

```cpp
// deep_learning_task.cpp:310-337
TrainingResult DeepLearningTask::run() {
    return run_impl(false);
}
TrainingResult DeepLearningTask::run_impl(bool dry_run) {
    if (using_gpu()) return run_gpu();
    else return run_cpu();  // run_cpu() 是 stub
}
```

#### 2.3.3 `run_gpu()` 框架（需保留）

```cpp
// deep_learning_task.cpp:677-878
TrainingResult DeepLearningTask::run_gpu() {
    // ① 日志输出、初始化（lines 684-718）
    // ② PRE-TEST：预启动 DEEP + OPT graph 验证（lines 720-745）— 可保留用于诊断
    // ③ Epoch 循环（lines 749-862）：
    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        // 渐进式分辨率调整
        // SEMA 切换
        // Prep 线程：prep.train()
        float train_loss = run_train_epoch_gpu();    // ★ 需要重写
        prep_thread.join();
        // 验证：
        if (should_validate_this_epoch()) {
            Prep 线程：prep.val()
            auto [vloss, vtop1, vtop5] = run_val_epoch_gpu(false);  // ★ 需调整
            // EMA 验证（如果启用 SEMA）
        }
        // 最佳指标追踪、日志、保存、早停
    }
    // ④ 返回 TrainingResult
}
```

**`run_gpu()` 的 epoch 循环框架本身是正确的，需要保留。** 需要替换的是其内部调用的 `run_train_epoch_gpu()` 和微调 `run_val_epoch_gpu()`。

#### 2.3.4 `run_train_epoch_gpu()` — 当前实现的问题

```cpp
// deep_learning_task.cpp:881-1453 (~570 lines)
```

**问题清单**：

| # | 问题 | 详情 |
|---|------|------|
| 1 | **诊断代码过多** | ~500 行 DIAG-* 日志代码夹杂在核心逻辑中，严重影响可读性和可维护性 |
| 2 | **Buffer 管理不一致** | 只有 rank 0 调用 `set_buffer_readable/set_buffer_writeable`（line 1231-1234），而 `run_h2d_only` 是所有 rank 都调用 |
| 3 | **缺少 Label D2D copy** | `run_h2d_only` 中显式将 I_A_LABEL / I_B_LABEL 拷贝到 LABEL_SMCE；当前 `run_train_epoch_gpu()` 没有此操作 |
| 4 | **传输与计算过度耦合** | 4-phase 流水线设计将 ZERO_GRAD、FIRST_FWD、DEEP、XFER、FIRST_BWD、ALLREDUCE、OPT 全部交错，难以独立验证每个阶段 |
| 5 | **Batch 边界处理复杂** | Batch 0 预传输、batches==1 边界、循环体(batch 0..batches-2)、last batch 四个分支各有不同逻辑 |
| 6 | **loss/scaling 初始化** | lines 894-913 用 magic value 3.14 和 1.0 初始化 loss/scaling 设备内存 |
| 7 | **冗余的 cudaMemsetAsync** | 每个 batch 的 DEEP 之前都 memset loss=0（lines 1152-1156, 1389-1393） |
| 8 | **PRE-TEST 副作用** | `run_gpu()` 中 lines 720-745 预启动 DEEP+OPT 会改变权重值（w13 被修改），然后 line 916-921 读取 BEFORE 值已经不是真正的初始值 |

#### 2.3.5 `run_val_epoch_gpu()` 分析

```cpp
// deep_learning_task.cpp:1469-1606
```

**结论**：`run_val_epoch_gpu()` 的传输模式已经非常接近 `run_h2d_only_val_epoch()`：

| 特征 | run_val_epoch_gpu() | run_h2d_only_val_epoch() |
|------|---------------------|--------------------------|
| Buffer 轮转 | `batch % 2` ✅ | `batch % 2` |
| XFER 选择 | A/B ✅ | A/B |
| Stream | s_trans ✅ | s_trans |
| 同步 | sync_tr() ✅ | cudaStreamSynchronize |
| set_buffer | **仅 rank 0** ⚠️ | **所有 rank** |
| 附加操作 | 推理 + 指标读取 | 无 |

**只需修改一点**：将 `set_buffer_readable/set_buffer_writeable` 从 "仅 rank 0" 改为 "所有 rank"，与 `run_h2d_only_val_epoch()` 保持一致。

---

## 3. 核心差异对比表

### 3.1 传输核心流程对比（GPU 路径）

```
run_h2d_only_train_epoch()               run_train_epoch_gpu() (当前)
────────────────────────────────────     ──────────────────────────────
                                         [PRE-TEST: launch DEEP+OPT]
                                         [init loss/scaling with magic values]
                                         [DIAG-INIT: scan all tensors]
                                         
batches==0? return                       batches==0? (隐式，batches=0 时为 val_batches)
                                         
Prep 线程: prep.train()                  Prep 线程: prep.train()

Batch 0:
  wait_buffer(0)                         Batch 0 预传输:
  Launch XFER_A on TRANS                   wait_buffer(0)
  sync TRANS                               Launch XFER_A on TRANS  
  set_buf_consumed(0) ← 所有 rank         sync TRANS
  label D2D copy                           [DIAG-XFER0: 全量 scan CPU GPU]
                                           set_buf_consumed(0) ← 仅 rank 0
                                         
                                         batches==1 分支:
                                           ZG → FIRST_FWD → DEEP → BWD → DAR → GC → LR → FAR → OPT
                                           全部 sync
                                         
Batch 1..batches-1:                      循环 batch=0..batches-2:
  wait_buffer(buf_id)                      Phase 1: ZG ‖ FIRST_FWD
  Launch XFER on TRANS                       sync
  sync TRANS                                [DIAG-PHASE1: 全量 scan]
  set_buf_consumed(buf_id)                Wait next buffer
  label D2D copy                          Phase 2: DEEP ‖ XFER(next)
                                             [DIAG-B0: loss/top1/scaling/w 等]
                                             sync COMP + TRANS
                                             [DIAG-S1/S2/S3/S4: 逐步验证]
                                             set_buf_consumed ← 仅 rank 0
                                           Phase 3: FIRST_BWD ‖ DEEP_AR
                                           Phase 4: LR → FIRST_AR → OPT
                                             全部 sync
                                         
                                         Last batch (batches-1):
                                           ZG → FIRST_FWD → DEEP → BWD → DAR → GC → LR → FAR → OPT

循环结构: for (batch=0; batch<batches; ++batch)   循环: for (batch=0; batch<batches-1; ++batch) + 最后 batch 单独
```

### 3.2 关键行为差异

| 行为 | run_h2d_only | run_train_epoch_gpu (当前) | 是否需要统一 |
|------|-------------|---------------------------|-------------|
| Buffer 消费标记 | 所有 rank 调用 | 仅 rank 0 调用 | ✅ 需统一 |
| Label D2D copy | 每 batch 执行 | 不执行 | ✅ 需统一 |
| XFER 后同步 | cudaStreamSynchronize(s_trans) | cudaStreamSynchronize(s_trans) | ✅ 已一致 |
| XFER Graph 选择 | buf_id % 2 → XFER_A/B | from_a → XFER_A/B | ✅ 已一致 |
| TRANS Stream | s_trans | s_trans | ✅ 已一致 |
| 诊断代码 | 无 | ~500 行 | ✅ 全部移除 |

---

## 4. 重写方案

### 4.1 总体策略

**compile() 路径：不动。**
- `compile()` → `compile_impl()` 已正确生成全部 CUDA Graph（含 XFER 和所有计算图）
- `build_graph_atlas()` 和 `build_exec_table()` 在 `h2d_only_=false` 时的行为已验证正确

**run() 路径：重写。**
- **移除** `run_train_epoch_gpu()` 当前全部实现（lines 881-1453），用新实现替代
- **微调** `run_val_epoch_gpu()`：将 set_buffer 从仅 rank 0 改为所有 rank
- **保留** `run_gpu()` 的 epoch 循环框架，内部调用新的 epoch 函数

### 4.2 新 `run_train_epoch_gpu()` 设计

**设计原则**：
1. 传输部分直接复用 `run_h2d_only_train_epoch()` 的 GPU 路径逻辑（拷贝过来，不做抽象层）
2. 在每个 batch 的传输完成后，顺序执行该 batch 的计算图
3. 不引入流水线重叠（先保证正确性，后续优化再说）
4. 移除全部诊断代码（约 500 行）

**伪代码**：

```
float run_train_epoch_gpu() {
    batches = prep.steps_per_epoch();
    K = num_gpus_;
    using_amp = registry.using_amp();
    frozen = is_first_layer_frozen();
    
    // Prep 线程已在 run_gpu() 中启动（不在函数内部管理 Prep 线程）
    // 因为 run_gpu 中有 prep_thread.join()，这里需要对齐
    
    threads[K];
    for (rank in 0..K-1):
        threads[rank] = lambda:
            cudaSetDevice(device_ids[rank]);
            
            // 获取 graph exec handles
            g_xfer_a/b = gpu_exec_.graphs[rank][XFER_A/B];
            g_deep_a/b = gpu_exec_.graphs[rank][FWD_BWD_DEEP_A/B];
            g_first_bwd_a/b = gpu_exec_.graphs[rank][FIRST_LAYER_BWD / FIRST_LAYER_BWD_B];
            g_first_fwd_a/b = gpu_exec_.graphs[rank][FIRST_FWD_A/B];
            g_zg = gpu_exec_.graphs[rank][ZERO_GRAD];
            g_dar = gpu_exec_.graphs[rank][DEEP_ALLREDUCE];
            g_far = gpu_exec_.graphs[rank][FIRST_LAYER_ALLREDUCE];
            g_wu = gpu_exec_.graphs[rank][WEIGHT_UPDATE];
            g_gc = gpu_exec_.graphs[rank][GRAD_CONVERT];  // 仅 AMP
            
            // 获取 stream handles
            s_trans, s_c1, s_c2, s_up;
            
            // 获取 TransferStation（带重试）
            ts = ...;
            
            // 获取 label pointers（用于 D2D copy）
            label_ptr_a/b, label_smce_ptr, label_nbytes;
            
            // 获取 lr device pointer
            lr_dev_ptr;
            
            for (batch = 0; batch < batches; ++batch):
                int buf_id = batch % 2;
                
                // ═══════════════════════════════════
                // Part A: H2D 传输（与 run_h2d_only_train_epoch 完全一致）
                // ═══════════════════════════════════
                ts->wait_buffer_readable(buf_id);
                cudaGraphLaunch(buf_id==0 ? g_xfer_a : g_xfer_b, s_trans);
                cudaStreamSynchronize(s_trans);
                
                ts->set_buffer_readable(buf_id, false);   // 所有 rank 调用
                ts->set_buffer_writeable(buf_id, true);   // 所有 rank 调用
                
                cudaMemcpyAsync(label_smce_ptr,
                    (buf_id == 0) ? label_ptr_a : label_ptr_b,
                    label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
                cudaStreamSynchronize(s_trans);
                
                // ═══════════════════════════════════
                // Part B: 计算
                // ═══════════════════════════════════
                bool from_a = (buf_id == 0);
                
                // Phase 1: ZERO_GRAD ‖ FIRST_FWD
                if (g_zg) cudaGraphLaunch(g_zg, s_up);
                if (from_a ? g_first_fwd_a : g_first_fwd_b)
                    cudaGraphLaunch(from_a ? g_first_fwd_a : g_first_fwd_b, s_c1);
                cudaStreamSynchronize(s_c1);
                cudaStreamSynchronize(s_up);
                
                // Phase 2: DEEP_FWD_BWD
                if (from_a ? g_deep_a : g_deep_b)
                    cudaGraphLaunch(from_a ? g_deep_a : g_deep_b, s_c1);
                cudaStreamSynchronize(s_c1);
                
                // Phase 3: FIRST_BWD ‖ DEEP_ALLREDUCE
                if (!frozen) {
                    auto g_bwd = from_a ? g_first_bwd_a : g_first_bwd_b;
                    if (g_bwd) cudaGraphLaunch(g_bwd, s_c1);
                }
                if (g_dar) cudaGraphLaunch(g_dar, s_up);
                cudaStreamSynchronize(s_c1);
                cudaStreamSynchronize(s_up);
                
                // Phase 4: GRAD_CONVERT (AMP only)
                if (using_amp && g_gc) {
                    cudaGraphLaunch(g_gc, s_up);
                    cudaStreamSynchronize(s_up);
                }
                
                // Phase 5: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE
                float lr = fetch_lr_for_batch(batch);
                *lr_pinned_[rank] = lr;
                cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                cudaMemcpyHostToDevice, s_up);
                if (g_far) cudaGraphLaunch(g_far, s_up);
                if (g_wu) cudaGraphLaunch(g_wu, s_up);
                cudaStreamSynchronize(s_up);
            end for
            
    join all threads;
    检查异常;
    
    // 读取 train loss（仅 rank 0）
    float train_loss = 0.0f;
    if (loss_id >= 0)
        train_loss = fetch_from_rank(...);
    return train_loss;
}
```

### 4.3 `run_val_epoch_gpu()` 微调

**仅需修改一处**（line 1564-1568）：

```cpp
// 当前代码（仅 rank 0）：
if (rank == 0) {
    ts->set_buffer_readable(buf, false);
    ts->set_buffer_writeable(buf, true);
}

// 改为（所有 rank）：
ts->set_buffer_readable(buf, false);
ts->set_buffer_writeable(buf, true);
```

这样使 `run_val_epoch_gpu()` 的传输行为与 `run_h2d_only_val_epoch()` 完全一致。

### 4.4 `run_gpu()` 的调整

`run_gpu()` 框架保持不变，但需调整以下细节：

1. **移除 PRE-TEST**（lines 720-745）：预启动 DEEP + OPT graph 会修改权重值，干扰训练初始状态。如果确实需要预验证 CUDA Graph 的正确性，应该在 `build_exec_table()` 之后、`compile_mark_compiled()` 之前完成，而不是在 `run()` 阶段。

2. **Prep 线程管理**：当前 `run_gpu()` 在 epoch 循环内创建 `prep_thread`，然后 `run_train_epoch_gpu()` 内部不管理 Prep 线程。新设计中 `run_train_epoch_gpu()` 也不管理 Prep 线程（由 `run_gpu()` 统一管理），保持此模式不变。

3. **SEMA 切换**：当前 `apply_sema_switch()` 是 stub（line 459），无需改动。

4. **Loss 读取**：当前 `run_train_epoch_gpu()` 返回 `float train_loss`，由 `run_gpu()` 用于日志。新实现保持此接口。

### 4.5 `run_cpu()` 不在本次范围

`run_cpu()` 当前是 stub（line 1460），不在本次重写范围内。CPU 路径可以在 GPU 路径稳定后单独实现。

---

## 5. 详细修改清单

### 文件：`src/task/deep_learning_task.cpp`

#### 修改 A：重写 `run_train_epoch_gpu()`（lines 881-1453 → 新实现）

**操作**：删除 lines 881-1453（约 570 行），替换为新实现（预计约 150 行）。

**新函数结构**：

```
float DeepLearningTask::run_train_epoch_gpu()             // line 881
{
#ifdef TR_USE_CUDA
    // 1. 获取参数 (lines 882-887，保留)
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    ...
    
    // 2. 初始化（保留 loss_id / sc_id 获取逻辑，移除 magic value 初始化）
    int32_t loss_id = active_memory_plan_->baseline().loss;
    
    // 3. 多 rank 线程
    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            // 3a. cudaSetDevice
            // 3b. 获取 graph exec handles（保留 lines 932-947，新增 g_first_bwd_b 和 g_first_fwd_a/b）
            // 3c. 获取 stream handles（保留 lines 949-960）
            // 3d. 获取 TransferStation（复用 h2d_only 的重试逻辑）
            // 3e. 获取 label pointers（复用 h2d_only 的 label copy 逻辑）
            
            // 4. Batch 循环
            for (int batch = 0; batch < batches; ++batch) {
                int buf_id = batch % 2;
                
                // === Part A: H2D 传输 ===
                ts->wait_buffer_readable(buf_id);
                cudaGraphLaunch((buf_id == 0) ? g_xfer_a : g_xfer_b, s_trans);
                cudaStreamSynchronize(s_trans);
                
                ts->set_buffer_readable(buf_id, false);
                ts->set_buffer_writeable(buf_id, true);
                
                cudaMemcpyAsync(label_smce_ptr,
                    (buf_id == 0) ? label_ptr_a : label_ptr_b,
                    label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
                cudaStreamSynchronize(s_trans);
                
                // === Part B: 计算 ===
                bool from_a = (buf_id == 0);
                
                // ZERO_GRAD ‖ FIRST_FWD
                if (g_zg) cudaGraphLaunch(g_zg, s_up);
                auto g_fwd = from_a ? g_fwd_a : g_fwd_b;
                if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                sync_comp(); sync_up();
                
                // DEEP_FWD_BWD
                auto g_deep = from_a ? g_deep_a : g_deep_b;
                if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                sync_comp();
                
                // FIRST_BWD ‖ DEEP_ALLREDUCE
                if (!frozen) {
                    auto g_bwd = from_a ? g_first_bwd_a : g_first_bwd_b;
                    if (g_bwd) cudaGraphLaunch(g_bwd, s_c1);
                }
                if (g_dar) cudaGraphLaunch(g_dar, s_up);
                sync_comp(); sync_up();
                
                // GRAD_CONVERT (AMP)
                if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }
                
                // LR → FAR → OPT
                lr = fetch_lr_for_batch(batch);
                *lr_pinned_[rank] = lr;
                cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float), cudaMemcpyHostToDevice, s_up);
                if (g_far) cudaGraphLaunch(g_far, s_up);
                if (g_wu) cudaGraphLaunch(g_wu, s_up);
                sync_up();
            }
        });
    }
    join/check;
    
    // 5. 读取 loss
    float train_loss = ...;
    return train_loss;
#else
    return 0.0f;
#endif
}
```

**关键注意事项**：
- **保留** `g_first_bwd_b` 的获取（在当前的 `run_train_epoch_gpu()` 中有 `g_first_b` = `g_tab[S(FIRST_LAYER_BWD_B)]`），因为 Double Buffer 的 B 侧需要对应 B 版本的首层反向图
- **保留** `fetch_lr_for_batch()` 和 `lr_pinned_` 机制（LR 需要在每 batch 后更新 H2D）
- **保留** `frozen` 检查（`is_first_layer_frozen()` — `freeze_after_` epoch 后跳过首层反向）
- **移除** 所有 DIAG-* 日志（DIAG-XFER0, DIAG-INIT, DIAG-PHASE1, DIAG-S1~S4, DIAG-B0, GEXEC, LOSS-DBG, WGHT-DBG 等）
- **移除** magic value 初始化（`init_val=3.14f`, `sc_val=1.0f`）
- **移除** cudaMemsetAsync(loss=0) 调用 — 这些应该由计算图内部处理

#### 修改 B：微调 `run_val_epoch_gpu()`（lines 1564-1568）

**操作**：将 `set_buffer_readable/set_buffer_writeable` 从 "仅 rank 0" 改为 "所有 rank"。

```
// 修改前 (lines 1564-1568):
if (rank == 0) {
    ts->set_buffer_readable(buf, false);
    ts->set_buffer_writeable(buf, true);
}

// 修改后:
ts->set_buffer_readable(buf, false);
ts->set_buffer_writeable(buf, true);
```

#### 修改 C：调整 `run_gpu()`（lines 720-745 和 line 909-921）

**操作**：
1. 移除 PRE-TEST 块（lines 720-745）：
   ```
   // 删除以下代码：
   {
       cudaSetDevice(gpu_exec_.device_ids[0]);
       ...
       LOG_INFO << "[PRE-TEST] w13[0] after direct launch=" << w0;
   }
   ```
2. 移除 `run_gpu()` 中的 WGHT-DBG BEFORE 读取（line 916-921）：这部分在当前 `run_train_epoch_gpu()` 内部，重写后自然消失。
3. 移除 `run_train_epoch_gpu()` 中 WGHT-DBG AFTER 读取（line 1445-1450）：新实现不再包含。

#### 修改 D：清理废弃代码（可选，Phase 2）

以下代码在重写后将不再被调用，可以在确认新实现稳定后移除：
- `run_train_epoch()`（lines 343-441）：旧的 `TaskBase::run("xfer_a")` + `TaskBase::run("fwd_bwd_deep_a")` 风格，非 GPU 路径，已被新实现取代
- `run_val_epoch()`（lines 443-453）：空的 TODO stub，已被 `run_val_epoch_gpu()` 取代

**但**：`run_train_epoch()` 和 `run_val_epoch()` 在 `deep_learning_task.h` 中被声明为 private（lines 420, 423），需要确认是否有其他调用方。如果没有，可以移除声明和实现。

---

## 6. 实现步骤（建议顺序）

### Phase 1：重写 `run_train_epoch_gpu()`
1. 在当前函数**下方**（line 1453 之后）新增 `run_train_epoch_gpu_v2()`
2. 暂时保留原 `run_train_epoch_gpu()`
3. 在 `run_gpu()` 中加一个编译开关调用 v2（或直接替换调用）
4. 编译并测试 `test_dl_full.cpp`

### Phase 2：微调 `run_val_epoch_gpu()`
1. 修改 set_buffer 调用
2. 验证 val 路径

### Phase 3：清理
1. 移除 `run_gpu()` 中的 PRE-TEST 块
2. 移除旧的 `run_train_epoch_gpu()` 实现
3. 将 v2 重命名为主函数名

---

## 7. 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 新实现与旧 Pipeline 的行为差异导致训练不收敛 | 中 | 逐步验证：先跑 1 epoch H2D only → 加 compute 跑 1 epoch → 跑 3 epoch 完整训练 |
| Label D2D copy 是否正确处理 | 低 | `run_h2d_only` 已验证此逻辑正确，Copy-Paste 过来即可 |
| 所有 rank set_buffer 是否影响 Preprocessor 行为 | 低 | `run_h2d_only` 中所有 rank 已调用 set_buffer，且通过多 RANK 测试 |
| LR pinned memory 是否正确传递 | 低 | 保留现有 `lr_pinned_` + `fetch_lr_for_batch` 机制，已经在 PRE-TEST 中得到验证 |
| 移除诊断代码后能否调试 | 低 | 可以在新实现中保留关键的 `LOG_INFO` 行（如 batch 序号、loss 值） |
| `g_first_bwd_b` 是否需要 | 中 | 需要检查 `build_exec_table()` 是否解析了 `FIRST_LAYER_BWD_B` — **已确认** line 617 有解析但 `kRequired` 中未列出，可能需要确保非 null 或回退用 `FIRST_LAYER_BWD` |

---

## 8. 需要确认的开放问题

1. **`g_first_bwd_b` (FIRST_LAYER_BWD_B) vs `g_first_bwd` (FIRST_LAYER_BWD)**：当前 `build_exec_table()` 中 `kRequired` 只检查了 `FIRST_LAYER_BWD`，未检查 `FIRST_LAYER_BWD_B`。`kRequired` 列表需要增加 `FIRST_LAYER_BWD_B` 吗？还是 B 侧可以用 A 侧的图？

2. **`g_first_fwd_b` (FIRST_FWD_B)**：类似问题 — `kRequired` 列表中没有，但如果 batch 1 需要 FIRST_FWD_B，是否需要确保非 null？

3. **CPU 路径**：`run_cpu()` 当前是 stub，不做任何事。未来是否计划支持 CPU 训练路径？如果是，`run_impl()` 中 `run_cpu()` 分支需要独立设计。

4. **`run_train_epoch()`（旧版非 GPU 路径）**：lines 343-441 目前是否还有调用方？能否安全移除？

---

## 9. 总结

本次重写的核心思路是 **"compile() 不动，run() 传输部分对齐 h2d_only，计算部分叠加到传输之后"**：

| 模块 | 改动量 | 说明 |
|------|--------|------|
| `compile()` / `compile_impl()` | **0 行** | 已正确编译所有图（含 XFER） |
| `build_graph_atlas()` | **0 行** | h2d_only_=false 时已包含全部图 |
| `build_exec_table()` | **0 行** | 已正确解析全部 GraphSlot |
| `run_gpu()` | **~15 行删除** | 移除 PRE-TEST + WGHT-DBG |
| `run_train_epoch_gpu()` | **-570 + ~150 行** | 删除旧实现，写入新实现 |
| `run_val_epoch_gpu()` | **2 行修改** | set_buffer 改为所有 rank |
| **总计** | **约 -430 行** | 大量删除诊断代码 |

新 `compile() + run()` = 现有的 `compile()` + 新的 `run_gpu()`（内部调用新的 `run_train_epoch_gpu()` + 微调后的 `run_val_epoch_gpu()`），其中：

- **传输部分**：与 `run_h2d_only_train_epoch()`/`run_h2d_only_val_epoch()` **完全一致**
- **计算部分**：在传输完成后顺序执行 ZERO_GRAD → FIRST_FWD → DEEP_FWD_BWD → FIRST_BWD → ALLREDUCE → OPTIMIZER
- **框架部分**：保留 `run_gpu()` 的多 epoch 循环、验证、SEMA、渐进分辨率、早停、模型保存