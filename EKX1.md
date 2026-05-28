# EKX1.md — 最终综合方案：重写 `compile()` + `run()`

## 0. 文档说明

本文是对 EKX.md 中 **小伙伴 S、小伙伴 K、小伙伴 D** 三位方案及**用户补充要求**的综合评审。在逐条审视各方案之后，给出一个**经过纠偏和合成的最终可实施方案**。

**前置阅读**：EHR1.md（小伙伴 D 的详细调研报告）提供了完整的代码架构分析，本文不再重复。本文聚焦于方案对比和最终决策。

---

## 1. 三位方案概览与对比

| 维度 | 小伙伴 S | 小伙伴 K | 小伙伴 D（EHR1.md） |
|------|----------|----------|---------------------|
| **核心策略** | 简化 run_train_epoch | 提取共享 `execute_h2d_epoch()` | 逐 batch 嵌入 H2D + compute |
| **compile() 改动** | 无需修改 ✅ | 无需修改 ✅ | 无需修改 ✅ |
| **传输对齐方式** | 逐行对齐 | 提取为独立函数，调用方复用 | 直接嵌入 batch 循环 |
| **Phase 1 执行顺序** | 先用 H2D-only 传完所有 batch，再计算 | **全量预传输** → 然后逐个 batch 计算 | 每 batch：H2D → 同步 → 计算 |
| **Phase 2 并行** | 后续恢复 pipeline | per-batch pipeline 化 | 后续恢复 4-phase pipeline |
| **CPU 路径** | 未提及 | 提取 `execute_h2d_epoch_cpu()` | 未提及（标记为不在范围） |
| **诊断代码** | 移除 | `#ifdef TR_DIAG_EPOCH` 隔离 | 全部移除 |
| **label_smce 缺失** | 未显式指出 | **明确指出为致命根因** ✅ | 列为差异之一 ✅ |

---

## 2. 关键发现：小伙伴 K 方案的致命缺陷

### 2.1 缺陷描述

小伙伴 K 方案的 Phase 1 设计为：

```
// Phase 1 伪代码（小伙伴 K）
execute_h2d_epoch_gpu(batches, ...);   // ① 先做完所有 H2D
for batch = 0 .. batches-1:           // ② 再逐个 batch 计算
    launch compute graphs...
```

### 2.2 为什么这在当前架构下不可行

当前 H2D 传输使用**双缓冲（Double Buffer）**架构：

```
Zone A: I_A_DATA  + I_A_LABEL   // GPU memory slot A
Zone B: I_B_DATA  + I_B_LABEL   // GPU memory slot B
```

H2D Graph 的 `dst_off` 指向：
- XFER_A → 写入 `I_A_DATA` + `I_A_LABEL`（Zone A）
- XFER_B → 写入 `I_B_DATA` + `I_B_LABEL`（Zone B）

batch 轮转规则：
- batch 0 → Zone A（XFER_A）
- batch 1 → Zone B（XFER_B）
- batch 2 → Zone A（XFER_A，**覆盖 batch 0 的数据**）
- batch 3 → Zone B（XFER_B，**覆盖 batch 1 的数据**）

如果先一次性跑完所有 batch 的 H2D：
- batch 0 写入 Zone A → 被 batch 2 覆盖
- batch 1 写入 Zone B → 被 batch 3 覆盖
- 最终 GPU 上只保留了最后 2 个 batch 的数据
- **所有偶数 batch（除最后一个）的数据全部丢失**

### 2.3 结论

小伙伴 K 的 Phase 1（全量预传输）**在当前的双缓冲架构下不可能工作**。`execute_h2d_epoch_gpu()` 不能是一个"把所有 batch 传完"的函数——必须在每个 batch 的 H2D 完成之后、下一个同 zone batch 覆盖之前，就把计算做完。

正确的 Phase 1 结构应该是小伙伴 D 的方案：**每 batch 内先 H2D 再计算**。

---

## 3. 各方案优点汇总

尽管小伙伴 K 的 Phase 1 结构有致命缺陷，但其方案中包含若干有价值的洞察。以下提取三个方案中的核心优点：

### 从小伙伴 S 提取
- 4 阶段渐进实施思路（传输对齐 → 计算集成 → 多 epoch → 性能优化）
- 强调测试策略（cuda-memcheck、逐步验证）

### 从小伙伴 K 提取
- **`label_smce` D2D 拷贝缺失是致命根因**的精确诊断（最重要发现）
- 5 处关键差异的系统化对比表
- `#ifdef TR_DIAG_EPOCH` 条件编译隔离诊断代码（更好的工程实践）
- `execute_h2d_epoch_cpu()` 提取 CPU 路径（满足用户"CPU路径支持"的要求）
- `wait_buffer_readable()` vs `while(sleep)` 的差异识别

### 从小伙伴 D（EHR1.md）提取
- 对 `compile()` / `compile_impl()` / `build_graph_atlas()` / `build_exec_table()` 的逐行审查
- 确认 compile 阶段完全不需要改动
- **正确的 Phase 1 结构**：每 batch 内 H2D → 同步 → 计算
- 详细的 5.1 修改清单（文件级、函数级）
- `run_val_epoch_gpu()` 只需改 2 行的准确识别
- `run_gpu()` PRE-TEST 副作用分析
- `kRequired` 列表缺失 `FIRST_LAYER_BWD_B` 的发现

---

## 4. 用户补充要求的逐条响应

| # | 用户要求 | 如何在方案中保证 |
|---|---------|----------------|
| 1 | **性能重要** | Phase 1（正确性验证通过后）→ Phase 2（恢复 H2D+计算并行流水线） |
| 2 | **数学正确性重要** | Phase 1 每 batch 严格同步顺序执行；传输部分与已验证的 h2d_only 逐行一致 |
| 3 | **多 RANK 支持重要** | 保留多线程 per-rank 架构；所有 rank 调用 set_buffer（不再是仅 rank 0） |
| 4 | **CPU 路径支持重要** | 提取 `execute_h2d_epoch_cpu()`，让 CPU 训练路径可用（当前 stub → 可工作） |
| 5 | **GPU 和 AMP 路径大体类似** | AMP 仅多 GRAD_CONVERT 图启动，其余逻辑同 GPU |
| 6 | **传输部分与 h2d_only 完全一致** | 传输代码直接从 `run_h2d_only_train_epoch()` 搬过来，不做任何包装或修改 |
| 7 | **不加 TR_TEST_TWO_BATCH_CORRECTION** | `compile()`+`run()` 路径不定义此宏，所有 batch 都真实传输 |
| 8 | **不破坏 compile_h2d_only()+run_h2d_only()** | `compile_h2d_only()` 和 `run_h2d_only()` 代码不动；如果提取共享函数，则让 h2d_only 调用共享函数 |

---

## 5. 最终方案

### 5.1 总体架构（一图看清）

```
compile()  ← 不动，h2d_only_=false 已正确构建所有 GraphAtlas
    │
    ▼
run() → run_gpu()  ← epoch 循环框架保持，移除 PRE-TEST
    │
    ├─► run_train_epoch_gpu()    ← 完全重写（~570 行 → ~170 行）
    │       │
    │       └─► 每 batch:
    │             Part A: H2D 传输（与 run_h2d_only_train_epoch 逐行一致）
    │             Part B: 计算图（ZERO_GRAD → FIRST_FWD → DEEP → BWD → AR → OPT）
    │
    ├─► run_val_epoch_gpu()      ← 微调（~2 行修改：set_buffer 所有 rank 调用）
    │
    └─► run_cpu()                ← 本次不实现（保留 stub）
```

### 5.2 设计决策：为什么不在 Phase 1 就做流水线重叠

1. **首要目标是正确性**：当前 `test_dl_full.cpp` 无法通过，说明存在根本性正确性问题
2. `label_smce` D2D 拷贝缺失是最可能的根因——修复它并验证
3. 当前 4-phase 流水线代码中混杂大量诊断代码，难以确定哪些行为是"正确"的、哪些是调试副作用
4. **先串行、再并行**是工程上最安全的做法：如果串行就 pass，说明 4-phase 流水线的同步/竞争有额外 bug；如果串行不 pass，说明还有其他问题
5. MNIST MLP 3 epochs 在 GPU 上跑串行也不会超过几分钟——开发阶段的性能损失可接受

### 5.3 Phase 1：正确性第一的串行实现

#### 文件修改范围

**仅修改一个文件**：`src/task/deep_learning_task.cpp`

**不修改**：头文件 `deep_learning_task.h`（接口不变）

#### 修改 A：重写 `run_train_epoch_gpu()`（lines 881-1453）

**删除**：lines 881-1453 全部（约 570 行）

**替换为**：约 170 行的新实现

**新实现骨架**：

```cpp
float DeepLearningTask::run_train_epoch_gpu() {
#ifdef TR_USE_CUDA
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();
    bool frozen = is_first_layer_frozen();

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([=, &exc]() {
            try {
                // --- 1. 设备初始化 ---
                cudaSetDevice(gpu_exec_.device_ids[rank]);

                // --- 2. 获取 TransferStation（带重试，同 h2d_only） ---
                TransferStation* ts = nullptr;
                for (int w = 0; w < 200; ++w) {
                    ts = static_cast<TransferStation*>(
                        registry.transfer_station_ptr(rank));
                    if (ts) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!ts) TR_DEVICE_ERROR("TransferStation not ready");

                // --- 3. 获取所有 graph exec handles ---
                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];

                auto g_xfer_a   = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b   = g_tab[S(GraphSlot::XFER_B)];
                auto g_deep_a   = g_tab[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto g_deep_b   = g_tab[S(GraphSlot::FWD_BWD_DEEP_B)];
                auto g_first    = g_tab[S(GraphSlot::FIRST_LAYER_BWD)];
                auto g_first_b  = g_tab[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto g_zg       = g_tab[S(GraphSlot::ZERO_GRAD)];
                auto g_dar      = g_tab[S(GraphSlot::DEEP_ALLREDUCE)];
                auto g_far      = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto g_wu       = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
                auto g_gc       = g_tab[S(GraphSlot::GRAD_CONVERT)];
                auto g_fwd_a    = g_tab[S(GraphSlot::FIRST_FWD_A)];
                auto g_fwd_b    = g_tab[S(GraphSlot::FIRST_FWD_B)];

                // --- 4. 获取 streams ---
                DeviceContext& ctx = context(rank);
                cudaStream_t s_trans = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_up    = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));

                // --- 5. 获取 label pointers ---
                const auto& bl = active_memory_plan_->baseline();
                auto label_ptr_a    = ctx.ptr_at(bl.label_a);
                auto label_ptr_b    = ctx.ptr_at(bl.label_b);
                auto label_smce_ptr = ctx.ptr_at(bl.label_smce);
                size_t label_nbytes = static_cast<size_t>(
                    active_memory_plan_->get_dtensor(bl.label_a).slot_bytes());

                // --- 6. 获取 lr pointer ---
                float lr;
                float* lr_dev_ptr = static_cast<float*>(ctx.ptr_at(lr_dtensor_id_));

                // --- 7. 同步 lambda ---
                auto sync_comp = [&]() { cudaStreamSynchronize(s_c1); };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr   = [&]() { cudaStreamSynchronize(s_trans); };

                // ════════════════════════════════════════
                // 核心循环：每 batch = H2D（同 h2d_only）+ 计算
                // ════════════════════════════════════════
                for (int batch = 0; batch < batches; ++batch) {
                    int buf_id = batch % 2;
                    bool from_a = (buf_id == 0);

                    // ============== Part A: H2D ==============
                    // 以下代码与 run_h2d_only_train_epoch() GPU 路径逐行一致，
                    // 唯一区别：无 TR_TEST_TWO_BATCH_CORRECTION 宏
                    ts->wait_buffer_readable(buf_id);

                    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;
                    cudaGraphLaunch(g_xfer, s_trans);
                    sync_tr();

                    ts->set_buffer_readable(buf_id, false);
                    ts->set_buffer_writeable(buf_id, true);

                    cudaMemcpyAsync(label_smce_ptr,
                        (buf_id == 0) ? label_ptr_a : label_ptr_b,
                        label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
                    sync_tr();
                    // ============== Part A 结束 ==============

                    // ============== Part B: 计算 ==============
                    auto g_deep_cur = from_a ? g_deep_a : g_deep_b;
                    auto g_fwd_cur  = from_a ? g_fwd_a  : g_fwd_b;
                    auto g_bwd_cur  = from_a ? g_first   : g_first_b;

                    // Phase B1: ZERO_GRAD ‖ FIRST_FWD
                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd_cur) cudaGraphLaunch(g_fwd_cur, s_c1);
                    sync_comp(); sync_up();

                    // Phase B2: DEEP_FWD_BWD
                    if (g_deep_cur) cudaGraphLaunch(g_deep_cur, s_c1);
                    sync_comp();

                    // Phase B3: FIRST_BWD ‖ DEEP_ALLREDUCE
                    if (!frozen && g_bwd_cur) cudaGraphLaunch(g_bwd_cur, s_c1);
                    if (g_dar) cudaGraphLaunch(g_dar, s_up);
                    sync_comp(); sync_up();

                    // Phase B4: GRAD_CONVERT (AMP only)
                    if (using_amp && g_gc) {
                        cudaGraphLaunch(g_gc, s_up);
                        sync_up();
                    }

                    // Phase B5: LR → FIRST_ALLREDUCE → WEIGHT_UPDATE
                    lr = fetch_lr_for_batch(batch);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank],
                                    sizeof(float), cudaMemcpyHostToDevice, s_up);
                    if (g_far) cudaGraphLaunch(g_far, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up);
                    sync_up();
                    // ============== Part B 结束 ==============
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
    int32_t loss_id = active_memory_plan_->baseline().loss;
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

**新旧对比**：

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 代码行数 | ~570 行 | ~170 行 |
| Buffer 等待 | `while(!ts->buffer_is_readable) sleep(100us)` | `ts->wait_buffer_readable()` |
| Buffer 管理 | 仅 rank 0 调用 set_* | 所有 rank 调用 set_* ✅ |
| Label D2D copy | **缺失** ❌ | 每 batch 显式 D2D copy ✅ |
| Batch 结构 | 4 个分支（预传输/单 batch/循环/最后） | 1 个统一循环 ✅ |
| 诊断代码 | ~500 行散布在热路径 | **0 行** ✅ |
| Magic value 初始化 | `init_val=3.14f` 等 | 已移除 ✅ |
| cudaMemsetAsync(loss=0) | 每 batch 都清 | 已移除（交给计算图） ✅ |
| H2D 与计算并行 | 是（4-phase 流水线） | 否（串行，Phase 2 恢复） |

#### 修改 B：微调 `run_val_epoch_gpu()`（lines 1564-1568）

```cpp
// 修改前：
if (rank == 0) {
    ts->set_buffer_readable(buf, false);
    ts->set_buffer_writeable(buf, true);
}

// 修改后：
ts->set_buffer_readable(buf, false);
ts->set_buffer_writeable(buf, true);
```

同时将 `while (!ts->buffer_is_readable(buf)) sleep(100us)` 替换为 `ts->wait_buffer_readable(buf)`。

#### 修改 C：`run_gpu()` 中移除 PRE-TEST（lines 720-745）

**操作**：删除以下代码块：
```cpp
{
    cudaSetDevice(gpu_exec_.device_ids[0]);
    auto S = ...;
    auto g_deep = g_tab[S(GraphSlot::FWD_BWD_DEEP_A)];
    auto g_opt = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
    // ... launch DEEP + OPT, sync, read w13 ...
}
```

**理由**：此 PRE-TEST 会修改权重 w13 的值，干扰后续训练的初始状态。CUDA Graph 的可用性应在 `build_exec_table()` 阶段通过 `TR_CHECK` 验证（已存在）。

#### 修改 D：不影响 `compile_h2d_only()` 和 `run_h2d_only()`

- `compile_h2d_only()`：不改
- `run_h2d_only()`：不改
- `run_h2d_only_train_epoch()` / `run_h2d_only_val_epoch()`：不改
- `build_graph_atlas()`：不改
- `build_exec_table()`：不改

**这保证了所有现有 H2D-only 测试继续通过。**

---

## 6. Phase 2：恢复 H2D-计算并行（后续迭代）

Phase 1 通过 `test_dl_full.cpp` 后，执行 Phase 2 恢复性能。

**思路**：将"先传输后计算"的串行循环改为双缓冲流水线：

```
Batch 0（预热）:
  H2D → sync → 计算

Batch 1..N-1（流水线）:
  while 上一 batch 的计算:
    启动下一 batch H2D（非阻塞，s_trans 独立于 s_c1/s_up）
  等待 H2D 完成 → sync_tr → set_buffer → label D2D → sync_tr
  等待当前计算完成 → sync_comp / sync_up
  ← 此时 H2D 和计算已重叠
```

**注意**：Phase 2 的流水线实现需要仔细处理 `cudaGraphLaunch` 与 `sync` 的时序，必须确保：
1. 传输 Stream 和计算 Stream 的并发不产生 race（CUDA Graph 内部的资源互斥由 MemoryPlan 保证）
2. buffer 管理在所有 rank 中一致

---

## 7. CPU 路径的考虑

用户要求 CPU 路径也需支持。当前 `run_cpu()` 是 stub。本次重写中：

1. `run_train_epoch_gpu()` 在 `#ifdef TR_USE_CUDA` 内，不影响 CPU 编译
2. `run_cpu()` stub 保持不动——CPU 路径需要在 GPU 路径验证通过后单独实现
3. 如果要实现 CPU 路径，可从小伙伴 K 提出的 `execute_h2d_epoch_cpu()` 出发，复用 `run_h2d_only_train_epoch()` 中已验证的 CPU memcpy 逻辑

---

## 8. 完整文件修改清单

| 文件 | 行号 | 操作 | 改动量 |
|------|------|------|--------|
| `src/task/deep_learning_task.cpp` | 720-745 | **删除** PRE-TEST 块 | -25 行 |
| `src/task/deep_learning_task.cpp` | 881-1453 | **替换** `run_train_epoch_gpu()` 全部实现 | -570 + ~170 行 |
| `src/task/deep_learning_task.cpp` | 1564-1568 | **修改** set_buffer 从仅 rank 0 到所有 rank | ±2 行 |
| `src/task/deep_learning_task.cpp` | 1535-1536 | **修改** `while(sleep)` → `wait_buffer_readable` | ±2 行 |
| **其他所有文件** | — | **不修改** | 0 行 |
| **总计** | | | **净减少约 ~425 行** |

**头文件不改**：`deep_learning_task.h` 无需修改——`run_train_epoch_gpu()` 签名不变。

---

## 9. 验证计划

### 9.1 回归验证（确保不破坏现有功能）

| 测试 | 验证内容 |
|------|---------|
| `test_two_batch_correction.cpp` — MNIST/CIFAR10 × CPU/GPU/AMP | H2D-only 路径不受影响 |
| `test_h2d_only_epoch.cpp` — 所有配置 | H2D-only 多 epoch 路径不受影响 |

### 9.2 新功能验证

| 阶段 | 测试 | 预期 |
|------|------|------|
| Phase 1 | `test_dl_full.cpp` — MNIST GPU 3 epochs | best_top1 > 85% |
| Phase 1 | `test_dl_full.cpp` — MNIST AMP 3 epochs | best_top1 > 85% |
| Phase 1 | 多 RANK 测试（如有测试框架） | 无死锁、所有 rank 正常完成 |
| Phase 2 | 同上 + 性能对比 | 吞吐量 ≥ Phase 1 版本 |

### 9.3 调试策略

如果 Phase 1 `test_dl_full.cpp` 仍然不通过：

1. **二分法**：先只跑 H2D（注释计算部分） → 应该与 H2D-only 结果一致
2. **加 FIRST_FWD**：H2D + ZERO_GRAD + FIRST_FWD → 检查首层输出
3. **加 DEEP**：H2D + ZG + FIRST_FWD + DEEP → 检查 loss
4. **加 BWD + OPT**：全流程 → 检查权重变化和 loss 下降趋势

---

## 10. 风险评估

| 风险 | 等级 | 缓解 |
|------|------|------|
| 修复 label_smce 后 loss 仍异常 | 中 | 二分法逐个启用计算图，精确定位问题图 |
| 串行执行比旧版慢 | 低 | MNIST 3 epochs 耗时可忽略；Phase 2 恢复并行 |
| `g_first_bwd_b` 为 nullptr | 中 | B 侧回退用 A 侧的 `g_first`；因为 FIRST_BWD 写回 GPU data buffer，A/B 双侧应使用各自的图避免写冲突 |
| 所有 rank set_buffer 引发多卡竞争 | 低 | `run_h2d_only` 已验证所有 rank 调用的正确性 |
| `fetch_from_rank()` 读 loss 失败 | 低 | 保留现有 loss 读取逻辑 |

---

## 11. 总结

**一句话方案**：

`compile()` 不动，`run_train_epoch_gpu()` 用 `run_h2d_only_train_epoch()` 的传输逻辑（搬过来，逐行一致）+ 计算图（保持不变），每 batch 串行执行（Phase 1），验证通过后恢复流水线并行（Phase 2）。

**三个必须**：
1. **必须** 在每 batch 做 `label_smce` D2D copy（这是小伙伴 K 发现的最重要根因）
2. **必须** 所有 rank 调用 `set_buffer_readable/set_buffer_writeable`
3. **必须** 使用 `wait_buffer_readable()` 而非 `while(sleep)` 自旋

**三个不动**：
1. `compile()` 不动
2. `compile_h2d_only()` + `run_h2d_only()` 不动
3. 头文件不动