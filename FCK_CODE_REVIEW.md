# 【小伙伴K】

> **审查范围**: `task_base.cpp`, `deep_learning_task.cpp/h`, `task_base.h`  
> **基线**: Z_FINAL_K.md v4.20.1 最终方案  
> **结论**: P0×4 + P1×5 + P2×2，共 11 处问题

---

## 🔴 P0：训练必崩 / 结果必错

### P0-1：`init()` 实现完全错误（FCK.md 核心投诉成立）

**位置**: `src/task/task_base.cpp:1242-1313`

**问题清单**:

| # | 具体问题 | 后果 |
|---|---------|------|
| 1a | 使用 `std::mt19937` 而非项目已有的 `Generator`/`philox` RNG | 随机可复现性完全丧失 |
| 1b | `KAIMING_NORMAL` 与 `XAVIER_NORMAL` 实现**一模一样**（都是 `normal(0, config.scale)`） | Kaiming 和 Xavier 数学公式完全不同，权重初始化分布错误 → Loss NaN 或收敛极慢 |
| 1c | `KAIMING_UNIFORM` 与 `XAVIER_UNIFORM` 实现**一模一样**（都是 `uniform(-config.scale, config.scale)`） | 同上 |
| 1d | 未调用项目已有的 `Initializer::apply_to_tensor()` | 重复发明轮子，且发明的轮子数学错误 |
| 1e | 种子写死为 `42 + dtensor.id`，无法从 `GlobalRegistry` 配置 | 无法做实验复现 |

**项目已有正确实现**（在 `src/core/initializer.cpp:224-265`）：

```cpp
// KAIMING_NORMAL: std = gain / sqrt(fan)
float std = cfg.scale / std::sqrt(static_cast<float>(fan));
t.normal(0.0f, std);

// XAVIER_NORMAL: std = scale * sqrt(2/(fan_in+fan_out))
float std = cfg.scale * std::sqrt(2.0f / static_cast<float>(fi + fo));
t.normal(0.0f, std);
```

`Tensor::normal()` / `Tensor::uniform()` 内部已集成 `rng.h` 的 `Generator`，多线程可复现。

**最小修复**：

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    check_phase(Phase::COMPILED, "init");
    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : dtensor.init_config;
    if (!config.needs_init()) return;
    if (debug_mode_) { /* dry run print */ return; }

    const DTensor& live_dt = active_memory_plan_->get_dtensor(dtensor.id);
    TR_CHECK(live_dt.dtype == DType::FP32, NotImplemented,
             "init() currently only supports FP32 weights");

    Tensor host(live_dt.shape, live_dt.dtype);
    Initializer::apply_to_tensor(host, live_dt.shape, config);

    transfer_to_rank(host, live_dt, 0);
    if (num_gpus_ > 1) {
        broadcast_from_rank0(live_dt);
    }
}
```

> 删除所有 `std::mt19937`/`std::normal_distribution`/`std::uniform_real_distribution` 代码。

---

### P0-2：`init()` 中 `transfer_to_rank()` 参数顺序错误

**位置**: `src/task/task_base.cpp:1311`

**当前代码**:
```cpp
for (int rank = 0; rank < num_gpus_; ++rank) {
    transfer_to_rank(host, rank, live_dt);  // ← 参数顺序错误
}
```

**函数签名**:
```cpp
void TaskBase::transfer_to_rank(const Tensor& host, const DTensor& dt, int rank);
```

**问题**: 第二个参数传入 `int rank`，但期望 `const DTensor&`。C++ 无法隐式将 `int` 转为 `const DTensor&`。若项目未编译过此代码，此处会直接编译失败。

**最小修复**:
```cpp
transfer_to_rank(host, live_dt, 0);
if (num_gpus_ > 1) {
    broadcast_from_rank0(live_dt);
}
```

---

### P0-3：`run_train_epoch_gpu()` last batch 逻辑完全崩溃

**位置**: `src/task/deep_learning_task.cpp:938-975`

**问题**: `last_batch_id` 被错误地设为 `batches - 2`，导致 **真正的 last batch (`batches - 1`) 从未被 DEEP_FWD_BWD/FIRST_BWD/ALLREDUCE/WEIGHT_UPDATE 处理**。

当前代码逻辑：
```cpp
// 最后一个 batch
{
    const bool from_a = ((batches - 1) % 2 == 1);
    int last_batch_id = batches - 2;        // ← BUG: 应为 batches - 1
    
    cudaGraphLaunch(g_last, s_c1);          // 这是 batch (batches-1) 的 DEEP_FWD_BWD
    
    // 但下面所有操作都针对 batch (batches-2)
    if (g_first) cudaGraphLaunch(g_first, s_c2);  // batch batches-2 的 FIRST_BWD
    if (g_dar)   cudaGraphLaunch(g_dar, s_c2);    // batch batches-2 的 ALLREDUCE
    
    lr = fetch_lr_for_batch(last_batch_id);       // ← batches-2 的 LR
    if (g_wu) cudaGraphLaunch(g_wu, s_up);        // batch batches-2 的 WEIGHT_UPDATE
    
    // 额外又做了一次 WEIGHT_UPDATE，还是 batches-2
    lr = fetch_lr_for_batch(batches - 2);
    if (g_wu) cudaGraphLaunch(g_wu, s_up);
}
```

**后果**:
1. batch `batches-1` 只做了一次 DEEP_FWD_BWD，没有 FIRST_BWD/ALLREDUCE/WEIGHT_UPDATE
2. batch `batches-2` 做了两次 WEIGHT_UPDATE（第二次 LR 值相同）
3. 梯度回传不完整，权重更新次数与 batch 数不匹配

**最小修复**（重写 last batch 块）：

```cpp
// ========== Last batch (batches - 1) ==========
{
    bool last_a = ((batches - 1) % 2 == 0);
    auto gf_l = last_a ? gf_a : gf_b;
    auto gd_l = last_a ? gd_a : gd_b;

    // Phase 1: ZERO_GRAD || FIRST_FWD
    if (gzg)  cudaGraphLaunch(gzg,  s_up);
    if (gf_l) cudaGraphLaunch(gf_l, s_c1);
    sync_comp(); sync_up();

    // Phase 2: DEEP_FWD_BWD (无 next XFER)
    cudaGraphLaunch(gd_l, s_c1);
    sync_comp();

    // Phase 3: FIRST_BWD || ALLREDUCE
    if (!is_first_layer_frozen()) {
        if (g_first) cudaGraphLaunch(g_first, s_c1);
    }
    if (g_dar) cudaGraphLaunch(g_dar, s_up);
    sync_comp(); sync_up();

    // AMP
    if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }

    // Phase 4: LR H2D → WEIGHT_UPDATE
    {
        float lr = fetch_lr_for_batch(batches - 1);
        *lr_pinned_[rank] = lr;
        cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                        cudaMemcpyHostToDevice, s_up);
    }
    if (g_wu) cudaGraphLaunch(gwu, s_up);
    sync_up();
}
```

---

### P0-4：`compile_mark_compiled()` 对 SimpleTask 路径缺失

**位置**: `src/task/task_base.cpp:215-292`

**问题**: `compile_mark_compiled()` 被移到了 `if (auto* dl = dynamic_cast<DeepLearningTask*>(this))` 分支内部（line 267），但 **SimpleTask 路径 (`compile_capture_simple()`) 和 else 分支均未调用**。`phase_` 停留在 `MEMORY_LOCKED`，后续 `run()` 的 `check_phase(COMPILED)` 会直接抛异常。

**最小修复**: 将 `compile_mark_compiled()` 提到 `if/else` 块之后，恢复为对所有路径统一调用：

```cpp
if (is_simple_task()) {
    compile_capture_simple();
} else {
    // ... DL / 其他路径 ...
}

compile_mark_compiled();   // ← 必须无条件调用
```

> 注意：若 `compile_mark_compiled()` 放在 DL 分支内部，需确保 `init_all()` 和 `lr_pinned_` 分配在 `compile_mark_compiled()` **之后**（因为 `init_all()` 要求 `phase_ == COMPILED`）。

---

## 🟡 P1：功能缺失 / 数据竞争

### P1-1：batch 0 XFER 与 FIRST_FWD 数据竞争

**位置**: `src/task/deep_learning_task.cpp:869-891`

**问题**: batch 0 先 launch `g_xfer_a`（`s_trans`），紧接着 launch `g_fwd_a`（`s_c1`），两者之间**无跨流同步**。FIRST_FWD 需要读取 XFER 写入的 device 内存，若 XFER 未完成，FIRST_FWD 读到的是脏数据。

当前代码：
```cpp
cudaGraphLaunch(g_xfer_a, s_trans);   // XFER batch 0 → s_trans

// Phase 1: ZERO_GRAD || FIRST_FWD
if (g_fwd_a) cudaGraphLaunch(g_fwd_a, s_c1);  // ← 需要 batch 0 数据！
sync_comp(); sync_up();

cudaStreamSynchronize(s_trans);       // ← XFER 同步在 Phase 1 之后
```

**最小修复**: batch 0 的 XFER 必须在 Phase 1 之前完成：
```cpp
while (!ts->buffer_is_readable(0)) { ... }
cudaGraphLaunch(g_xfer_a, s_trans);
cudaStreamSynchronize(s_trans);       // ← 确保数据到位
if (rank == 0) { ts->set_buffer_readable(0, false); ts->set_buffer_writeable(0, true); }

// Phase 1: ZERO_GRAD || FIRST_FWD
if (g_zg) cudaGraphLaunch(g_zg, s_up);
if (g_fwd_a) cudaGraphLaunch(g_fwd_a, s_c1);
sync_comp(); sync_up();
```

---

### P1-2：ALLREDUCE 与 WEIGHT_UPDATE 跨流数据竞争

**位置**: `src/task/deep_learning_task.cpp:894-935`

**问题**: `g_dar` (ALLREDUCE) 被 launch 到 `s_c2`，而 `g_wu` (WEIGHT_UPDATE) 被 launch 到 `s_up`。两者之间**无任何跨流同步**。WEIGHT_UPDATE 需要读取 ALLREDUCE 聚合后的梯度，若 ALLREDUCE 未完成，WEIGHT_UPDATE 使用的是未聚合的旧梯度。

当前代码：
```cpp
if (g_dar) cudaGraphLaunch(g_dar, s_c2);   // ALLREDUCE on COMP_2
sync_comp();                                // 只同步 s_c1/s_c2，不 sync s_up

// Phase 4
if (g_wu) cudaGraphLaunch(g_wu, s_up);     // WEIGHT_UPDATE on UPDATE
sync_up();
```

**根因**: `stream_for(GraphId)` 把 `FIRST_COMM`/`DEEP_COMM` 映射到了 `COMP_2`（见 P1-3）。即使映射正确（UPDATE stream），当前代码的 `sync_comp()` 也不 sync `s_up`。

**最小修复**: 将 ALLREDUCE 映射回 UPDATE stream（见 P1-3），并在 Phase 3 末尾调用 `sync_up()`：
```cpp
// Phase 3
if (!is_first_layer_frozen()) { if (g_first) cudaGraphLaunch(g_first, s_c1); }
if (g_dar) cudaGraphLaunch(g_dar, s_up);     // ← UPDATE stream
sync_comp(); sync_up();                      // ← 必须同步 UPDATE
```

---

### P1-3：`stream_for()` Stream 映射错误

**位置**: `src/task/deep_learning_task.cpp:520-538`

**当前映射**:
```cpp
case GraphId::FIRST_COMM:       return StreamKind::COMP_2;   // ← 应为 UPDATE
case GraphId::DEEP_COMM:        return StreamKind::COMP_2;   // ← 应为 UPDATE
case GraphId::CAST_AND_CHECK:   return StreamKind::COMP_1;   // ← 应为 UPDATE
case GraphId::ZERO_GRAD:        return StreamKind::UPDATE;   // ✓ 正确
```

**问题**: Z_FINAL_K.md §3.2 和 §7.1 明确指定 ALLREDUCE / CAST_AND_CHECK / ZERO_GRAD / OPTIMIZER / EMA_UPDATE 均应在 UPDATE stream。当前实现将通信算子放到了 COMP_2，导致：
1. ALLREDUCE 与 WEIGHT_UPDATE 不在同流，需要额外跨流同步（见 P1-2）
2. CAST_AND_CHECK 在 COMP_1，与 GRAD_CONVERT 流不一致

**最小修复**:
```cpp
case GraphId::ZERO_GRAD:
case GraphId::FIRST_COMM:
case GraphId::DEEP_COMM:
case GraphId::CAST_AND_CHECK:
case GraphId::OPTIMIZER:
case GraphId::EMA_UPDATE:
    return StreamKind::UPDATE;
```

---

### P1-4：单 batch 路径 (`batches == 1`) 缺少 FIRST_BWD 和 ALLREDUCE

**位置**: `src/task/deep_learning_task.cpp:837-865`

**问题**: 单 batch 路径（调试/测试场景）只做了 ZERO_GRAD → FIRST_FWD → DEEP_FWD_BWD → WEIGHT_UPDATE，**完全跳过了 FIRST_BWD 和 ALLREDUCE**。这意味着首层梯度未回传，多卡梯度未聚合。

**最小修复**: 在单 batch 路径 Phase 2/3 补充：
```cpp
// Phase 2: DEEP_FWD_BWD
cudaGraphLaunch(g_deep_a, s_c1);
sync_comp();

// Phase 3: FIRST_BWD + ALLREDUCE
if (!is_first_layer_frozen()) { if (g_first) cudaGraphLaunch(g_first, s_c1); }
if (g_dar) cudaGraphLaunch(g_dar, s_up);
sync_comp(); sync_up();

// AMP
if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }

// Phase 4: LR + WEIGHT_UPDATE
...
```

---

### P1-5：`init()` 逐 rank H2D 传输，未用 `broadcast_from_rank0()`

**位置**: `src/task/task_base.cpp:1310-1312`

**当前代码**:
```cpp
for (int rank = 0; rank < num_gpus_; ++rank) {
    transfer_to_rank(host, rank, live_dt);  // 参数顺序也错了
}
```

**问题**: 
1. 多卡场景下对每个 rank 做独立 H2D，效率低（N 次 H2D vs 1 次 H2D + 1 次 NCCL broadcast）
2. Z_FINAL_K.md §5.2 明确设计为 `transfer_to_rank(..., 0)` + `broadcast_from_rank0()`

**最小修复**: 见 P0-2 的修复代码。

---

## 🟢 P2：实现偏差

### P2-1：单 batch 路径 `g_gc` 被 launch 到 `s_c1` 而非 `s_up`

**位置**: `src/task/deep_learning_task.cpp:852-855`

```cpp
if (using_amp && g_gc) {
    cudaGraphLaunch(g_gc, s_c1);   // ← 应为 s_up
    sync_comp();
}
```

**问题**: `CAST_AND_CHECK` 的 stream kind 应为 UPDATE（见 P1-3），单 batch 路径却把它 launch 到 COMP_1。

**最小修复**: `cudaGraphLaunch(g_gc, s_up); sync_up();`

---

### P2-2：last batch 末尾无意义 launch `g_fwd_b`

**位置**: `src/task/deep_learning_task.cpp:965`

```cpp
if (g_fwd_b) cudaGraphLaunch(g_fwd_b, s_c1); // 条件保护
sync_comp();
```

**问题**: 在 Phase 4 (WEIGHT_UPDATE) 之后，又 launch 了一个 FIRST_FWD_B。last batch 结束时不需要再做 FIRST_FWD。这是 dead code，应删除。

**最小修复**: 删除这两行。

---

## 附录：修改文件汇总

| # | 文件 | 修改内容 | 严重度 |
|---|------|---------|:------:|
| 1 | `src/task/task_base.cpp` | `init()` 重写：删除 mt19937，改用 `Initializer::apply_to_tensor()` + `transfer_to_rank(...,0)` + `broadcast_from_rank0()` | 🔴 P0 |
| 2 | `src/task/task_base.cpp` | `compile_impl()` 中 `compile_mark_compiled()` 提到 `if/else` 块之后，无条件调用 | 🔴 P0 |
| 3 | `src/task/deep_learning_task.cpp` | `run_train_epoch_gpu()` last batch 块重写，修正 batch id | 🔴 P0 |
| 4 | `src/task/deep_learning_task.cpp` | batch 0 XFER 后加 `cudaStreamSynchronize(s_trans)` | 🟡 P1 |
| 5 | `src/task/deep_learning_task.cpp` | `stream_for()` FIRST_COMM/DEEP_COMM/CAST_AND_CHECK 映射为 UPDATE | 🟡 P1 |
| 6 | `src/task/deep_learning_task.cpp` | 中间 batch 循环 Phase 3 后加 `sync_up()` | 🟡 P1 |
| 7 | `src/task/deep_learning_task.cpp` | 单 batch 路径补充 FIRST_BWD + ALLREDUCE | 🟡 P1 |
| 8 | `src/task/deep_learning_task.cpp` | 单 batch 路径 `g_gc` launch 到 `s_up` | 🟢 P2 |
| 9 | `src/task/deep_learning_task.cpp` | 删除 last batch 末尾无意义的 `g_fwd_b` launch | 🟢 P2 |
