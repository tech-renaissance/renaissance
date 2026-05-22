# DeepLearningTask 第三轮代码审查报告

> **审查范围**: `deep_learning_task.cpp` (lines 795-1000), `task_base.cpp` (compile_impl)
> **对比基线**: Z_FINAL_K.md §7.3 + FCK_REVIEW_2.md（上一轮 4 处问题）
> **结论**: 上一轮 4 处问题已全部修复，但 **`run_train_epoch_gpu()` 循环结构与 Z_FINAL_K.md 存在系统性偏差**，共 **5 处问题（1 P0 + 2 P1 + 2 P2）**

---

## 已修复问题确认（4/4）

| 原编号 | 问题 | 状态 |
|--------|------|:--:|
| P0-1 | last batch 漏 batch (batches-1 的 Phase3+4 缺失) | ✅ last batch 块已补充双段 Phase3+4 |
| P1-1 | `build_exec_table()` 缺少 FIRST_FWD_A/B/CAST_AND_CHECK 填充 | ✅ lines 598-600 已补充 |
| P1-2 | `build_graph_atlas()` `seen` 集合逻辑风险 | ✅ 已改为 train/infer 显式分离结构 |
| P2 | 未使用变量 `lr_dt`、`g_far` | ✅ 已删除 |

---

## 🔴 P0: `run_train_epoch_gpu()` 循环结构与 Z_FINAL_K.md §7.3 系统性不符

**位置**: `src/task/deep_learning_task.cpp:885-1000`

当前实现采用了一种**自定义的重叠式流水线**（pre-path + loop + last-block），与 Z_FINAL_K.md §7.3 的**非重叠式统一循环**（每个迭代处理当前 batch 完整 4 phase）存在根本性偏差。这不是风格差异，而是导致 **batch ≥ 1 的 Phase 1 完全缺失** 和 **batches=2 时 XFER 缺失** 的根因。

### P0-1: Phase 1 (ZERO_GRAD ‖ FIRST_FWD) 只在 batch 0 执行一次

**当前代码** (lines 899-904):
```cpp
// 第 0 个 batch
{
    // ... XFER_A ...
    // Phase 1: ZERO_GRAD ‖ FIRST_FWD  ← 仅此处
    if (g_zg) cudaGraphLaunch(g_zg, s_up);
    if (g_fwd_a) cudaGraphLaunch(g_fwd_a, s_c1);
    sync_comp(); sync_up();

    // Phase 2: DEEP_FWD_BWD
    cudaGraphLaunch(g_deep_a, s_c1);
}

// 中间 batch (1 .. batches-2)
for (int batch = 1; batch < batches - 1; ++batch) {
    // XFER(next)
    // Phase 2: DEEP (current)
    // Phase 3+4: FIRST_BWD / DAR / LR / WU (prev)
    // ← 完全无 Phase 1
}
```

**Z_FINAL_K.md §7.3**:
```cpp
for (int batch = 0; batch < batches - 1; ++batch) {
    // Phase 1: ZERO_GRAD || FIRST_FWD  ← 每 batch 都有
    if (gzg) cudaGraphLaunch(gzg, s_up);
    if (gf)  cudaGraphLaunch(gf,  s_c1);
    sync_comp(); sync_up();
    // ... Phase 2/3/4 ...
}
```

**后果**: batch ≥ 1 永不执行 ZERO_GRAD 和 FIRST_FWD。
- ZERO_GRAD 缺失 → 梯度不清零，batch N 的梯度会累积 batch N-1 的残留梯度
- FIRST_FWD 缺失 → 首层（如 Flatten + FC1）在 batch ≥ 1 时不执行前向，后续层使用上一 batch 的旧 feature map

### P0-2: 循环起始 batch 错误 + 奇偶翻转 → batches=2 时 XFER 缺失

**当前代码**:
```cpp
for (int batch = 1; batch < batches - 1; ++batch) {  // 从 batch=1 开始
    const bool from_a = (batch % 2 == 1);             // 奇偶翻转
```

**Z_FINAL_K.md**:
```cpp
for (int batch = 0; batch < batches - 1; ++batch) {   // 从 batch=0 开始
    bool from_a = (batch % 2 == 0);                   // 偶数为 A
```

**后果**:

| batches | 当前循环执行次数 | 缺失的 XFER |
|---------|:---------------:|------------|
| 1 | 0 (单 batch 路径) | 无 |
| 2 | 0 (`batch < 1` 不成立) | **batch 1 的 XFER 完全缺失** |
| 3 | 1 (batch=1 only) | 无 |
| 469 | 467 (batch=1..467) | 无 |

当 `batches == 2` 时（小数据集或大数据量 batch size），循环体一次都不执行：
- pre-path 只做了 XFER_A (batch 0)
- last batch 块没有 XFER
- **batch 1 的数据从未从 TransferStation 传到 GPU**

### P0-3: Pre-path 越界包含 Phase 2，且缺少 `sync_comp()`

**当前代码 pre-path** (lines 887-908):
```cpp
// 第 0 个 batch
{
    while (!ts->buffer_is_readable(0)) { ... }
    cudaGraphLaunch(g_xfer_a, s_trans);
    cudaStreamSynchronize(s_trans);
    // Phase 1
    if (g_zg) cudaGraphLaunch(g_zg, s_up);
    if (g_fwd_a) cudaGraphLaunch(g_fwd_a, s_c1);
    sync_comp(); sync_up();
    // Phase 2: 直接进入循环，无 sync_comp()
    cudaGraphLaunch(g_deep_a, s_c1);  // ← 与循环 batch=1 的 DEEP 同流串行
}
```

**Z_FINAL_K.md pre-path**:
```cpp
while (!ts->buffer_is_readable(0)) { ... }
cudaGraphLaunch(gx_a, s_tr);
sync_trans();
// ← Phase 1/2 在循环 batch=0 中执行，不在 pre-path 中
```

**后果**: pre-path 的 `DEEP(0)` 在 `s_c1` 上 launch 后无 `sync_comp()`。随后循环 `batch=1` 中：
1. `DEEP(1)` 也在 `s_c1` 上 launch（同流 FIFO，隐式等待 `DEEP(0)`）
2. `FIRST_BWD(0)` 也在 `s_c1` 上 launch（隐式等待 `DEEP(1)`）

执行顺序被串化为：`DEEP(0) → DEEP(1) → FIRST_BWD(0)`。

`FIRST_BWD(0)` 被不必要地推迟到 `DEEP(1)` 之后，破坏了 "DEEP 完成后立即 FIRST_BWD" 的时序。虽然同流 FIFO 保证了正确性（无数据竞争），但 **GPU 利用率下降**，且与 Z_FINAL_K.md 设计的时序承诺不符。

### 最小修复（整体替换为 Z_FINAL_K.md §7.3 结构）

建议直接将 `run_train_epoch_gpu()` 的重叠式三段结构（pre-path / loop / last-block）替换为 Z_FINAL_K.md 的非重叠式统一循环。这是**最小侵入且最符合设计文档**的修复：

```cpp
#ifdef TR_USE_CUDA
void DeepLearningTask::run_train_epoch_gpu() {
    // ... 变量获取与 stream 初始化不变 ...

    // ========== Batch 0 预传输 ==========
    while (!ts->buffer_is_readable(0))
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    cudaGraphLaunch(gx_a, s_trans);
    sync_trans();
    if (rank == 0) {
        ts->set_buffer_readable(0, false);
        ts->set_buffer_writeable(0, true);
    }

    // ========== 单 batch 边界 ==========
    if (batches == 1) { /* ... 同 Z_FINAL_K.md ... */ return; }

    // ========== 统一循环：batch = 0 .. batches-2 ==========
    for (int batch = 0; batch < batches - 1; ++batch) {
        bool from_a  = (batch % 2 == 0);      // ← 修正奇偶
        int next_buf = from_a ? 1 : 0;
        auto gf  = from_a ? gf_a : gf_b;
        auto gd  = from_a ? gd_a : gd_b;
        auto gx_n = from_a ? gx_b : gx_a;

        // Phase 1: ZERO_GRAD || FIRST_FWD  ← 每 batch 都有
        if (gzg) cudaGraphLaunch(gzg, s_up);
        if (gf)  cudaGraphLaunch(gf,  s_c1);
        sync_comp(); sync_up();

        // Wait next buffer
        while (!ts->buffer_is_readable(next_buf))
            std::this_thread::sleep_for(std::chrono::microseconds(100));

        // Phase 2: DEEP_FWD_BWD || XFER(next)
        cudaGraphLaunch(gd,   s_c1);
        cudaGraphLaunch(gx_n, s_tr);
        sync_comp(); sync_trans();           // ← 显式同步
        if (rank == 0) {
            ts->set_buffer_readable(next_buf, false);
            ts->set_buffer_writeable(next_buf, true);
        }

        // Phase 3: FIRST_BWD || DEEP_ALLREDUCE
        if (!frozen) cudaGraphLaunch(gfb, s_c1);
        if (gda)     cudaGraphLaunch(gda, s_up);
        sync_comp(); sync_up();

        // AMP
        if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

        // Phase 4: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE
        {
            float lr = fetch_lr_for_batch(batch);
            *lr_pinned_[rank] = lr;
            cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                            cudaMemcpyHostToDevice, s_up);
        }
        if (gfa) cudaGraphLaunch(gfa, s_up);   // ← 恢复 g_fa
        if (gwu) cudaGraphLaunch(gwu, s_up);
        sync_up();
    }

    // ========== Last batch (batch = batches-1) ==========
    bool last_a = ((batches - 1) % 2 == 0);
    auto gf_l = last_a ? gf_a : gf_b;
    auto gd_l = last_a ? gd_a : gd_b;

    if (gzg)  cudaGraphLaunch(gzg,  s_up);
    if (gf_l) cudaGraphLaunch(gf_l, s_c1);
    sync_comp(); sync_up();

    cudaGraphLaunch(gd_l, s_c1);
    sync_comp();
    if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }
    if (gda) cudaGraphLaunch(gda, s_up);
    sync_up();
    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

    {
        float lr = fetch_lr_for_batch(batches - 1);
        *lr_pinned_[rank] = lr;
        cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                        cudaMemcpyHostToDevice, s_up);
    }
    if (gfa) cudaGraphLaunch(gfa, s_up);       // ← 恢复 g_fa
    if (gwu) cudaGraphLaunch(gwu, s_up);
    sync_up();
}
#endif
```

---

## 🟡 P1-1: `g_fa` (FIRST_LAYER_ALLREDUCE) 运行时缺失

**位置**: `deep_learning_task.cpp:821-831`

当前代码在 lambda 变量获取段删除了 `g_fa`：

```cpp
auto g_first   = g_tab[S(GraphSlot::FIRST_LAYER_BWD)];
auto g_zg      = g_tab[S(GraphSlot::ZERO_GRAD)];
auto g_dar     = g_tab[S(GraphSlot::DEEP_ALLREDUCE)];
// auto g_far  = ...  ← 上一轮已删除
// auto g_fa   ← 从未声明
```

后果：所有 Phase 4 代码段均缺少 `if (gfa) cudaGraphLaunch(gfa, s_up);`。

- 单卡 MLP：`gfa == nullptr`，`if (gfa)` 保护下自动跳过，**不影响功能**
- 多卡场景：`FIRST_LAYER_ALLREDUCE` 在 `build_exec_table()` 中已正确填充（line 594），但运行时永远不 launch。**首层梯度在多卡间不做 AllReduce**，导致各卡首层梯度不一致，**训练发散**

**最小修复**：恢复变量声明并在 Phase 4 补充 launch：

```cpp
auto g_fa    = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];  // ← 恢复

// 在单 batch 路径 Phase 4 和 last batch Phase 4 中补充：
if (g_fa) cudaGraphLaunch(g_fa, s_up);
```

---

## 🟡 P1-2: `build_graph_index()` 冗余

**位置**: `deep_learning_task.cpp:469-477`

`build_graph_index()` 已经无任何调用者（被 `build_graph_atlas()` 取代），且其内部逻辑与 `build_graph_atlas()` 末尾的 `name_to_gid_` 填充重复。

```cpp
void DeepLearningTask::build_graph_index() {
    name_to_gid_.clear();
    if (train_cg_) {
        name_to_gid_["train"] = GraphId::DEEP_FWD_BWD;
    }
    if (infer_cg_) {
        name_to_gid_["inference"] = GraphId::INF_MAIN_A;
    }
}
```

`build_graph_atlas()` 末尾已执行完全相同的逻辑（lines 518-524）。`build_graph_index()` 是死代码，建议删除。

---

## 🟢 P2-1: `save_best` 比较逻辑偏差

**位置**: `deep_learning_task.cpp:760-766`

**当前代码**:
```cpp
float best_this_epoch = use_sema_ ? std::max(top1, ema_top1) : top1;
if (best_this_epoch >= best_top1_ || (use_sema_ && best_this_epoch >= best_ema_top1_)) {
    bool save_ema = use_sema_ && (ema_top1 > top1);
    save_model_to(save_best_path_, save_ema);
}
```

**Z_FINAL_K.md §8.1**:
```cpp
float best_this_epoch = use_sema_ ? std::max(top1, ema_top1) : top1;
float best_overall = use_sema_ ? std::max(best_top1_, best_ema_top1_) : best_top1_;
if (best_this_epoch >= best_overall) { ... }
```

**后果**: 当 `use_sema_ = true` 且 `ema_top1 == best_ema_top1_`（EMA 没有改进）时：
- 当前条件：`best_this_epoch >= best_ema_top1_` → `true`
- 会触发不必要的模型保存

**最小修复**:
```cpp
float best_overall = use_sema_ ? std::max(best_top1_, best_ema_top1_) : best_top1_;
if (best_this_epoch >= best_overall) { ... }
```

---

## 🟢 P2-2: `run_cpu()` 路径 `run_train_epoch()` 未跟随 GPU 路径更新

**位置**: `deep_learning_task.cpp:341-439`

`run_train_epoch()`（CPU 路径）仍然使用旧的 `TaskBase::run("name")` 单图串行执行：

```cpp
TaskBase::run("xfer_a");
TaskBase::run(xfer, compute);   // 双图并行
TaskBase::run("first_layer_bwd");
```

而 GPU 路径已升级为 CUDA Graph 4-phase 调度。虽然 CPU 路径不涉及 CUDA Graph，但 CPU 路径的训练逻辑（ZERO_GRAD / FIRST_FWD / ALLREDUCE / WEIGHT_UPDATE / LR 步进）应与 GPU 路径一致。

当前 CPU 路径：
- 无 ZERO_GRAD 调用（与修复前 GPU 路径相同）
- 无 ALLREDUCE / WEIGHT_UPDATE / LR 更新
- 每个 batch 后只执行 `first_layer_bwd` 和 `scheduler.step()`

**这不是当前 MLP 训练的阻塞问题**（目标场景是 GPU），但 CPU 路径的训练语义已严重落后于 GPU 路径。建议添加 TODO 注释或统一两者逻辑。

---

## 总结

| 编号 | 问题 | 严重度 | 当前 MLP 是否触发 | 修复方式 |
|------|------|:------:|:----------------:|:--------|
| P0-1 | Phase 1 只在 batch 0 执行，batch ≥ 1 无 ZG/FIRST_FWD | 🔴 P0 | **是**，梯度累积 + 首层前向缺失 | 重写循环为 Z_FINAL_K.md §7.3 统一结构 |
| P0-2 | 循环起始 batch=1 + 奇偶翻转 → batches=2 无 XFER | 🔴 P0 | 是（小 batch 数场景） | 同上 |
| P0-3 | Pre-path 越界含 Phase 2 且无 sync_comp | 🔴 P0 | 是，性能退化 | 同上 |
| P1-1 | `g_fa` (FIRST_LAYER_ALLREDUCE) 运行时缺失 | 🟡 P1 | 否（单卡），多卡必触发 | 恢复变量声明 + Phase 4 补充 launch |
| P1-2 | `build_graph_index()` 死代码 | 🟡 P1 | 否 | 删除 |
| P2-1 | `save_best` 比较逻辑偏差 | 🟢 P2 | 否（EMA 未改进时误保存） | 改用 `best_overall` |
| P2-2 | CPU 路径训练逻辑落后于 GPU 路径 | 🟢 P2 | 否（目标 GPU） | 添加 TODO 或后续统一 |

**最优先修复：P0 三连击**。三者根因相同（自定义重叠式流水线与 Z_FINAL_K.md 不符），一次整体替换可全部解决。
