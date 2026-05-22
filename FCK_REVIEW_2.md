# DeepLearningTask 第二轮代码审查报告

> **审查范围**: `deep_learning_task.cpp`, `deep_learning_task.h`, `task_base.cpp`
> **对比基线**: FCK_CODE_REVIEW.md（上一轮 11 处问题）
> **结论**: 上一轮 11 处中已修复 9 处，**剩余 2 处 + 新增 3 处，共 5 处**

---

## 已修复问题确认（9/11）

| 原编号 | 问题 | 状态 |
|--------|------|:--:|
| P0-1 | `init()` 使用 `std::mt19937`、Kaiming/Xavier 相同实现 | ✅ 已改用 `Initializer::apply_to_tensor()` |
| P0-2 | `transfer_to_rank` 参数顺序错误 | ✅ 已修正为 `transfer_to_rank(host, live_dt, 0)` |
| P0-4 | `compile_mark_compiled()` 对 SimpleTask 缺失 | ✅ 已移到 `if/else` 块之后无条件调用 |
| P1-1 | batch 0 XFER 与 FIRST_FWD 数据竞争 | ✅ 已加 `cudaStreamSynchronize(s_trans)` |
| P1-2 | ALLREDUCE 与 WEIGHT_UPDATE 跨流数据竞争 | ✅ `stream_for()` 已将 ALLREDUCE 映射到 UPDATE stream |
| P1-3 | `stream_for()` Stream 映射错误 | ✅ FIRST_COMM/DEEP_COMM/CAST_AND_CHECK 已映射到 UPDATE |
| P1-4 | 单 batch 路径缺少 FIRST_BWD + ALLREDUCE | ✅ 单 batch 路径已补充 |
| P1-5 | `init()` 逐 rank H2D 传输 | ✅ 已改为 `rank0 + broadcast_from_rank0()` |
| P2-1 | 单 batch 路径 `g_gc` launch 到 `s_c1` | ✅ 已改为 `s_up` |
| P2-2 | last batch 末尾无意义 `g_fwd_b` launch | ✅ 已删除 |

---

## 🔴 P0：last batch 漏 batch（上一轮 P0-3 未完全修复）

**位置**: `src/task/deep_learning_task.cpp:900-968`

**问题**: 重叠式流水线设计导致 **batch `batches-1` 的 FIRST_BWD + ALLREDUCE + WEIGHT_UPDATE 从未被执行**。

**流水线追踪**（以 batches=469 为例）：

| 阶段 | 处理内容 |
|------|---------|
| batch 0 初始化块 | XFER 0 → Phase1(batch0) → Phase2(batch0 DEEP_FWD_BWD 启动) |
| 循环 batch=1 | Phase2(batch1) + Phase3+4(batch0) |
| 循环 batch=2 | Phase2(batch2) + Phase3+4(batch1) |
| ... | ... |
| 循环 batch=467 | Phase2(batch467) + Phase3+4(batch466) |
| last batch 块 | Phase2(batch468) + Phase3+4(batch467) |
| **缺失** | **Phase3+4(batch468)** |

**根因**: 循环条件 `for (int batch = 1; batch < batches - 1; ++batch)` 只到 `batch = batches-2`（即 467）。循环体每次完成的是 **prev batch** 的 Phase3+4。因此：
- batch `batches-2`（467）的 Phase2 在循环最后一次迭代启动，但它的 Phase3+4 需要在循环的下一个迭代完成——而循环已结束
- batch `batches-1`（468）的 Phase3+4 完全缺失

当前 last batch 块（lines 944-968）只完成了 batch `batches-2` 的 Phase3+4，**batch `batches-1` 的 FIRST_BWD / ALLREDUCE / WEIGHT_UPDATE 被遗漏**。

**最小修复**（在 last batch 块末尾补充 batch `batches-1` 的 Phase3+4）：

```cpp
// 最后一个 batch
{
    const bool from_a = ((batches - 1) % 2 == 0);
    cudaGraphExec_t gd_l = from_a ? g_deep_a : g_deep_b;

    // Phase 2: batch (batches-1) DEEP_FWD_BWD
    cudaGraphLaunch(gd_l, s_c1);
    sync_comp();

    // Phase 3+4: batch (batches-2)
    if (!is_first_layer_frozen()) {
        if (g_first) cudaGraphLaunch(g_first, s_c1);
    }
    if (g_dar) cudaGraphLaunch(g_dar, s_up);
    sync_comp(); sync_up();
    if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }

    lr = fetch_lr_for_batch(batches - 2);
    *lr_pinned_[rank] = lr;
    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                    cudaMemcpyHostToDevice, s_up);
    if (g_wu) cudaGraphLaunch(g_wu, s_up);
    sync_up();

    // ===== 补充：batch (batches-1) 的 Phase 3+4 =====
    if (!is_first_layer_frozen()) {
        if (g_first) cudaGraphLaunch(g_first, s_c1);
    }
    if (g_dar) cudaGraphLaunch(g_dar, s_up);
    sync_comp(); sync_up();
    if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }

    lr = fetch_lr_for_batch(batches - 1);
    *lr_pinned_[rank] = lr;
    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                    cudaMemcpyHostToDevice, s_up);
    if (g_wu) cudaGraphLaunch(g_wu, s_up);
    sync_up();
}
```

> 替代方案：将整个循环结构改为 Z_FINAL_K.md §7.3 的非重叠式（每个循环体处理当前 batch 的全部 4 phase），可大幅简化 last batch 逻辑。但侵入量更大。

---

## 🟡 P1：`build_exec_table()` 缺少 `FIRST_FWD_A/B` 和 `CAST_AND_CHECK` 填充

**位置**: `src/task/deep_learning_task.cpp:578-588`

**问题**: `build_exec_table()` 中只填充了 11 个 slot，但 GraphSlot 枚举有 17 个值。以下 slot **从未被填充**，始终为 `nullptr`：

| 缺失的 Slot | 对应 GraphId | `run_train_epoch_gpu()` 中是否使用 |
|------------|-------------|----------------------------------|
| `FIRST_FWD_A` | `GraphId::FIRST_FWD_A` | ✅ 使用（`g_fwd_a`） |
| `FIRST_FWD_B` | `GraphId::FIRST_FWD_B` | ✅ 使用（`g_fwd_b`） |
| `CAST_AND_CHECK` | `GraphId::CAST_AND_CHECK` | ❌ 未直接使用（`GRAD_CONVERT` 已映射到同一 GraphId） |

**后果**: 
- 对于当前 MLP（无 progressive resolution）：`FIRST_FWD_A/B` 在 Atlas 中索引为 -1，`g_fwd_a`/`g_fwd_b` 为 `nullptr`，`if (g_fwd_a)` 保护下自动跳过。**不影响当前训练**。
- 但如果 Compiler 未来生成 `FIRST_FWD_A/B`（progressive resolution 启用），`g_fwd_a` 将永远为 `nullptr`，Phase 1 的 FIRST_FWD 永远被跳过，**训练结果必错**。
- 后面的 `TR_CHECK`（lines 607-613）在 Atlas 有 `FIRST_FWD_A/B` 时会断言失败（因为 Exec Table 中对应 slot 为 `nullptr`），**运行时直接 crash**。

**最小修复**:

```cpp
// 在 build_exec_table() 的 slot 填充区域补充：
g[S(GraphSlot::FIRST_FWD_A)]      = resolve(GraphId::FIRST_FWD_A, rank);
g[S(GraphSlot::FIRST_FWD_B)]      = resolve(GraphId::FIRST_FWD_B, rank);
g[S(GraphSlot::CAST_AND_CHECK)]   = resolve(GraphId::CAST_AND_CHECK, rank);
```

---

## 🟡 P1：`build_graph_atlas()` 的 `seen` 去重逻辑风险

**位置**: `src/task/deep_learning_task.cpp:489-518`

**问题**: `build_graph_atlas()` 遍历 `named_graphs_`（`unordered_map`，遍历顺序不确定），对每个图的所有非空 GraphId bucket 填充 Atlas slot。`seen` 集合用于防止同一个 GraphId 被覆盖两次。

如果将来 train 图和 infer 图意外共享某个 GraphId（比如 Compiler 变更导致），`seen` 会**静默丢弃后遍历到的图**，导致 Atlas slot 的 `cg` 指针指向错误的 ComputationGraph。

**最小修复**: 改为 Z_FINAL_K.md §3.2 的显式分离结构，消除 `seen` 集合的歧义：

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (train_cg_->nodes(gid).empty()) continue;
            auto& sl = atlas.slot(0, gi);
            sl.cg = train_cg_;
            sl.mp = active_memory_plan_;
            sl.stream_kind = stream_for(gid);
            sl.shape_id = kShapeInvariant;
        }
    }

    if (infer_cg_) {
        static const GraphId kInferIds[] = {
            GraphId::INF_MAIN_A, GraphId::INF_MAIN_B,
            GraphId::INF_EMA_A,  GraphId::INF_EMA_B
        };
        for (GraphId gid : kInferIds) {
            if (infer_cg_->nodes(gid).empty()) continue;
            auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
            sl.cg = infer_cg_;
            sl.mp = active_memory_plan_;
            sl.stream_kind = StreamKind::COMP_1;
            sl.shape_id = kShapeInvariant;
        }
    }

    name_to_gid_.clear();
    if (train_cg_) name_to_gid_["train"] = GraphId::DEEP_FWD_BWD;
    if (infer_cg_) name_to_gid_["inference"] = GraphId::INF_MAIN_A;

    return atlas;
}
```

---

## 🟢 P2：未使用变量 `lr_dt` 和 `g_far`

**位置**: `src/task/deep_learning_task.cpp:835-836, 816`

```cpp
const DTensor& lr_dt = active_memory_plan_->get_dtensor(lr_dtensor_id_);  // 声明后从未使用
auto g_far = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];                  // 获取后从未 launch
```

**影响**: 仅为编译器警告/代码冗余，不影响功能。`g_far` 的缺失在当前 MLP 路径中无影响（多卡 ALLREDUCE 由 `g_dar` 处理）。

**建议**: 删除 `lr_dt` 声明；`g_far` 可保留为 TODO 注释（未来多卡 progressive resolution 可能需要 FIRST_LAYER_ALLREDUCE）。

---

## 总结

| 编号 | 问题 | 严重度 | 当前 MLP 是否触发 |
|------|------|:------:|:----------------:|
| 1 | last batch 漏 batch (batches-1 的 Phase3+4 缺失) | 🔴 P0 | **是**，训练不完整 |
| 2 | `build_exec_table()` 缺少 FIRST_FWD_A/B 填充 | 🟡 P1 | 否（MLP 无 FIRST_FWD），但启用 progressive resolution 后必崩 |
| 3 | `build_graph_atlas()` `seen` 集合逻辑风险 | 🟡 P1 | 否（当前 train/infer GraphId 不重叠） |
| 4 | 未使用变量 `lr_dt`、`g_far` | 🟢 P2 | 否 |

**最优先修复：P0-1（last batch 漏 batch）**。这是当前唯一一个在 MLP 单卡/多卡场景下都会触发的训练完整性 bug。
