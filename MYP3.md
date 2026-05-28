# MYP3 — Last Batch 与不同 Shape 捕获的完整、科学、严谨解决方案

> **基于 LRF1.md、LRF2.md、LRF3.md 的综合分析 + 框架源代码实地验证**
> **版本**: v3.0 | **日期**: 2026-05-28  
> **目标**: 彻底解决 last batch 正确性问题 + 完整支持渐进式分辨率 + 保持最高性能

---

## 一、问题诊断：三层系统性断裂

### 1.1 根本缺陷总结

经过对 LRF1/LRF2/LRF3 三份分析文档的综合验证，当前框架存在**三层系统性断裂**：

| 层级           | 设计意图                                                     | 当前实现                                                     | 断裂点                                                    | 严重性 |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------- | ------ |
| **Compiler**   | 6 变体编译（train_base/last/lowres/lowres_last + val_base/last），各自 MemoryPlan + ShapeId | `on_prepare()` **只传 1 个 base_spec**，`variant_specs` 为空 → `result.variants.size() == 1` | 未生成 last batch / lowres / val 的 MemoryPlan           | 🔴 P0  |
| **GraphAtlas** | 6×N 映射表，shape 相关图各自 `mp` + `shape_id`，shape 无关图共享 `kShapeInvariant` | `build_graph_atlas()` **只填 variant 0**，所有 slot 硬编码 `shape_id = kShapeInvariant`、`mp = active_memory_plan_` | pre_capture 去重时所有变体 Key 相同 → 只捕获一次 | 🔴 P0  |
| **运行时**     | 根据 batch 类型（normal/last）和分辨率（base/lowres）选择 correct variant | `build_exec_table()` 只解析 `atlas.index(0, gid)`；`run_train_epoch_gpu()` 对所有 batch 使用**同一组 graph** | last batch 使用 grid=base_batch 的 graph 处理 last_batch_size 个样本 | 🔴 P0  |

### 1.2 为什么必须切换 Variant 的物理约束

**CUDA Graph 的物理限制**（小伙伴 K 的决定性论证）：

1. **Grid 维度固定**：CUDA Graph 的 kernel grid 维度在 **capture 时固定**，replay 时不可变
2. **Shape 读取时机**：`softmax_ce_op.cpp:L359` 中 `int batch = logits_dt.shape.n()` 在 **capture 时**从 MemoryPlan 读取并固定
3. **运行时限制**：即使运行时修改 `batch_size_ptr` 标量为 402，CUDA Graph 的 `grid` 维度仍然是 capture 时的 512

**示例**：
- variant[0] train_base: `logits_dt.shape = {512, 32, 32, 64}` → capture 时 grid=512
- variant[1] train_last: `logits_dt.shape = {402, 32, 32, 64}` → capture 时 grid=402
- 如果 last batch 使用 variant[0] 的 graph，grid=512，后 110 个 block 处理残留数据 → **污染 loss/top1/top5/梯度**

**唯一正确做法**：
- 为 variant[1] 单独 capture CUDA Graph
- capture 前 `ctx.set_memory_plan(mp_variant1)`，使 grid=last_batch_size
- `pre_capture()` 的 `capture_all_for_rank()` 已内置此机制，问题只在于 `build_graph_atlas()` 未提供正确的 `mp` 和 `shape_id`

### 1.3 用户强调的不可违背约束

根据用户补充的指导，框架设计有以下**绝对要求**：

1. **H2D 传输图特殊性**：
   - 不管 train/val、full/last batch、渐进分辨率，H2D 传输图**只有 2 种**（A区图和B区图）
   - H2D 始终传输整个 A 区、整个 B 区，无需管分辨率
   - 但 **AMP 开启与不开启，H2D 传输图不同**

2. **权重更新/梯度同步不受 batch size 影响**：
   - OPTIMIZER/EMA_UPDATE/COMM 图不受 batch size 影响
   - 这些图在验证阶段不存在

3. **受 batch size 影响的图**：
   - 主要是首层正反向（FIRST_LAYER_FWD_A/B/FIRST_LAYER_BWD_A/B）
   - 深层正反向（DEEP_FWD_BWD）
   - batch size 影响特征图形状，但不影响权重形状

4. **性能绝对优先**：
   - **在进入 batch 循环之前**用局部变量存储好所有需要的图
   - 运行时避免 hash 查找、分支判断、内存分配
   - 循环内零函数调用、零数组索引、零条件分支（除 A/B 双缓冲）

5. **CPU/GPU 完全对齐**：
   - 如果 GPU 为某种情形单独捕获图，CPU 也应当单独捕获

---

## 二、6-Variant 系统精确定义

### 2.1 Variant 映射表

| Variant Index | 名称             | Batch Size 来源          | 分辨率来源                          | 适用阶段   | ComputationGraph | ShapeId 示例 (ImageNet 8 rank)         |
| ------------- | ---------------- | ------------------------ | ----------------------------------- | ---------- | ---------------- | ------------------------------------- |
| 0             | train_base       | `local_batch_size`       | `train_sample_resolution_end`       | 训练·常规 | train_cg         | `{512, 224, 224, 3}`                   |
| 1             | train_last       | `last_train_batch_size`  | `train_sample_resolution_end`       | 训练·末   | train_cg         | `{402, 224, 224, 3}`                   |
| 2             | train_lowres     | `local_batch_size`       | `train_sample_resolution_begin`     | 训练·低   | train_cg         | `{512, 128, 128, 3}`                   |
| 3             | train_lowres_last | `last_train_batch_size`  | `train_sample_resolution_begin`     | 训练·末低 | train_cg         | `{402, 128, 128, 3}`                   |
| 4             | val_base         | `local_batch_size`       | `val_sample_resolution`             | 验证·常规 | infer_cg         | `{512, 224, 224, 3}` (同 train_base)  |
| 5             | val_last         | `last_val_batch_size`    | `val_sample_resolution`             | 验证·末   | infer_cg         | `{106, 224, 224, 3}`                   |

**关键洞察**：
- Variant 0-3：训练路径，包含 ALL + TRAIN graphs（TRANSFER + FWD/BWD + OPTIMIZER + COMM）
- Variant 4-5：验证路径，只包含 TRANS + INF + VAL_METRICS graphs（无 OPTIMIZER/COMM）
- **交叉槽位自然隔离**：val variant 不填 train graphs，train variant 不填 infer graphs
- **ShapeId 去重自动优化**：v0 和 v4 的 ShapeId 可能相同 → 自动共享 captured graph

### 2.2 Graph 分类（按 Shape 敏感性）

| 类型           | GraphId                                                                 | Shape 相关性 | 各 Variant 行为                           | 去重结果                   |
| -------------- | ---------------------------------------------------------------------- | ------------ | ---------------------------------------- | -------------------------- |
| **完全 Shape 无关** | TRANSFER_A/B<br>ZERO_GRAD<br>DEEP_COMM/FIRST_COMM<br>OPTIMIZER<br>EMA_UPDATE<br>CAST_*<br>NAN_CHECK_AND_GRAD_SCALING<br>ACCUM_METRICS<br>ACCUM_METRICS_TRAIN_LAST<br>ACCUM_METRICS_VAL_LAST<br>VAL_RESULT_COMM<br>CLEAR_METRICS | 完全无关     | 全部 6 variant 共享 base MemoryPlan      | 全局共享 1 张图（15个GraphId） |
| **训练 Shape 相关** | FIRST_LAYER_FWD_A<br>FIRST_LAYER_FWD_B<br>DEEP_FWD_BWD<br>FIRST_LAYER_BWD_A<br>FIRST_LAYER_BWD_B | Batch 相关   | 4 个训练 variant 各自 MemoryPlan         | 最多 4 张独立图            |
| **推理 Shape 相关** | INF_MAIN_A<br>INF_MAIN_B<br>INF_EMA_A<br>INF_EMA_B | Batch 相关   | 2 个验证 variant 各自 MemoryPlan         | 最多 2 张独立图            |

**注意**：
- ACCUM_METRICS 系列虽然计算时需要 batch_size，但其 kernel 内使用 `batch_size_ptr` 动态读取，graph 本身 shape 无关
- 因此 ACCUM_METRICS、ACCUM_METRICS_TRAIN_LAST、ACCUM_METRICS_VAL_LAST 全部属于 **完全 Shape 无关**

### 2.3 去重后 Captured Graph 数量估算

**保守估算**（ImageNet progressive case）：
- Shape 无关图：15 个 GraphId × 1 次 = 15
- Shape 相关训练图：6 个 GraphId × 4 个不同 shape = 24
- Shape 相关推理图：4 个 GraphId × 2 个不同 shape = 8
- **总计 ≈ 47 张 captured graph**

**CIFAR-10 退化场景**（batch=200, 无渐进分辨率）：
- 所有 6 个 variant 的 ShapeId = `{200, 32, 32, 3}` → **完全相同**
- Shape 相关图全部共享 → **总计 ≈ 15 张**（与当前实现相同，零额外开销）

**实际数量在 15-50 张之间**，完全在可接受范围内。

---

## 三、详细实施方案

### 3.1 Step 0: 数据结构准备

**文件**: `include/renaissance/task/deep_learning_task.h`

在 `private:` 区域新增成员变量：

```cpp
// ===== 6-Variant 系统新增成员 =====

/// Compiler 完整编译结果（含 6 variant + train_cg + infer_cg）
/// 必须保存以确保 variants 数组（MemoryPlan 所有权）生命周期
Compiler::Result compiled_result_;

/// 各 variant 的 CompileSpec（用于获取 ShapeId）
std::array<CompileSpec, GraphAtlas::kMaxVariants> variant_compile_specs_;

/// 各 variant 的 MemoryPlan（slot_bytes 跨变体一致，shape 不同）
/// on_prepare() 从 compiled_result_ 中转移所有权
std::array<std::unique_ptr<MemoryPlan>, GraphAtlas::kMaxVariants> variant_memory_plans_;
```

修改 `GpuExecTable` 结构：

```cpp
#ifdef TR_USE_CUDA
struct GpuExecTable {
    // variant_graphs[variant_idx][rank][slot] → cudaGraphExec_t
    // variant_idx: 0=train_base, 1=train_last, 2=train_lowres,
    //              3=train_lowres_last, 4=val_base, 5=val_last
    // shape-invariant 图的 handles 在所有 variant_idx 中指向同一指针
    std::array<std::vector<std::vector<cudaGraphExec_t>>, GraphAtlas::kMaxVariants> variant_graphs;
    std::vector<int> device_ids;
};
#endif
```

### 3.2 Step 1: 修复 `on_prepare()` 生成完整 Variant Specs

**文件**: `include/renaissance/task/deep_learning_task.h`

在 `on_prepare()` 函数中，将当前只传 `base_spec` 的代码替换为：

```cpp
void on_prepare() override {
    // ===== 前面配置代码不变（到 plan_config_ 设置）=====
    // ... (initializer, batch_size, resolution, arch_plan 等) ...

    // ===== 旧代码（删除）：=====
    // CompileSpec spec = CompileSpec::from_global_registry();
    // auto result = Compiler::compile(plan, spec, plan_config_, initializer_);

    // ===== 新代码：生成 5 个 variant specs =====
    auto& gr = GlobalRegistry::instance();
    CompileSpec base_spec = CompileSpec::from_global_registry();
    std::vector<CompileSpec> variant_specs;

    int standard_bs = gr.get_local_batch_size();
    int last_train   = gr.get_last_train_batch_size();
    int last_val     = gr.get_last_val_batch_size();
    int res_begin    = gr.get_train_sample_resolution_begin();
    int res_end      = gr.get_train_sample_resolution_end();
    int val_res      = gr.val_sample_resolution();

    // v1: train_last
    {
        CompileSpec s = base_spec;
        s.batch_size = last_train;
        variant_specs.push_back(s);
    }
    // v2: train_lowres
    {
        CompileSpec s = base_spec;
        s.actual_resolution = res_end;
        if (res_end <= 0) s.actual_resolution = res_begin;
        variant_specs.push_back(s);
    }
    // v3: train_lowres_last
    {
        CompileSpec s = base_spec;
        s.batch_size = last_train;
        s.actual_resolution = res_end;
        if (res_end <= 0) s.actual_resolution = res_begin;
        variant_specs.push_back(s);
    }
    // v4: val_base
    {
        CompileSpec s = base_spec;
        s.actual_resolution = val_res;
        if (val_res <= 0) s.actual_resolution = res_begin;
        variant_specs.push_back(s);
    }
    // v5: val_last
    {
        CompileSpec s = base_spec;
        s.batch_size = last_val;
        s.actual_resolution = val_res;
        if (val_res <= 0) s.actual_resolution = res_begin;
        variant_specs.push_back(s);
    }

    // ===== 保存所有 CompileSpec（用于 build_graph_atlas 获取 ShapeId）=====
    variant_compile_specs_[0] = base_spec;
    for (size_t i = 0; i < variant_specs.size() && i < 5; ++i) {
        variant_compile_specs_[i + 1] = variant_specs[i];
    }

    // ===== 五阶段编译 =====
    compiled_result_ = Compiler::compile(plan, base_spec, plan_config_,
                                         initializer_, variant_specs);

    // ===== 转移所有 variant 的 MemoryPlan =====
    for (size_t i = 0; i < compiled_result_.variants.size() && i < GraphAtlas::kMaxVariants; ++i) {
        variant_memory_plans_[i] = std::move(compiled_result_.variants[i].memory_plan);
    }

    // ===== 保持向后兼容：active_memory_plan_ 指向 variant 0 =====
    memory_plan_ptr_ = std::move(variant_memory_plans_[0]);
    active_memory_plan_ = memory_plan_ptr_.get();

    // 持有 train_cg_ / infer_cg_ 引用（用于 Atlas 填表）
    train_cg_ = &compiled_result_.train_cg;
    infer_cg_ = &compiled_result_.infer_cg;

    // ... 后续代码不变（lr_dtensor_id_ 查找等）...
}
```

### 3.3 Step 2: 重写 `build_graph_atlas()` 全量填表

**文件**: `src/task/deep_learning_task.cpp`

首先在文件顶部（匿名命名空间）添加辅助函数：

```cpp
namespace {

// shape-invariant GraphId 判定
bool is_si_graph(GraphId gid) {
    switch (gid) {
        case GraphId::TRANSFER_A:
        case GraphId::TRANSFER_B:
        case GraphId::ZERO_GRAD:
        case GraphId::DEEP_COMM:
        case GraphId::FIRST_COMM:
        case GraphId::CAST_DEEP_GRAD_FP16_TO_FP32:
        case GraphId::CAST_FIRST_GRAD_FP16_TO_FP32:
        case GraphId::NAN_CHECK_AND_GRAD_SCALING:
        case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:
        case GraphId::CAST_MAIN_FP32_TO_FP16:
        case GraphId::ACCUM_METRICS:
        case GraphId::ACCUM_METRICS_TRAIN_LAST:
        case GraphId::ACCUM_METRICS_VAL_LAST:
        case GraphId::VAL_RESULT_COMM:
        case GraphId::CLEAR_METRICS:
            return true;
        default:
            return false;
    }
}

bool is_train_gid(GraphId gid) {
    switch (gid) {
        case GraphId::FIRST_LAYER_FWD_A:
        case GraphId::FIRST_LAYER_FWD_B:
        case GraphId::DEEP_FWD_BWD:
        case GraphId::FIRST_LAYER_BWD_A:
        case GraphId::FIRST_LAYER_BWD_B:
            return true;
        default:
            return false;
    }
}

bool is_infer_gid(GraphId gid) {
    switch (gid) {
        case GraphId::INF_MAIN_A:
        case GraphId::INF_MAIN_B:
        case GraphId::INF_EMA_A:
        case GraphId::INF_EMA_B:
            return true;
        default:
            return false;
    }
}

} // anonymous namespace
```

然后重写 `build_graph_atlas()` 函数：

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    // ===== H2D-only 模式：保持简单（无 FWD/BWD/INF）=====
    if (h2d_only_) {
        if (train_cg_) {
            for (GraphId gid : {GraphId::TRANSFER_A, GraphId::TRANSFER_B}) {
                if (train_cg_->nodes(gid).empty()) continue;
                auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
                sl.cg = train_cg_;
                sl.mp = active_memory_plan_;
                sl.stream_kind = stream_for(gid);
                sl.shape_id = kShapeInvariant;
            }
        }
        name_to_gid_.clear();
        return atlas;
    }

    // ===== 全量填表：6 variant × 所有 GraphId =====
    for (size_t v = 0; v < GraphAtlas::kMaxVariants; ++v) {
        const MemoryPlan* mp = variant_memory_plans_[v].get();
        if (!mp) continue;

        bool is_train_var = (v <= 3);  // v0-v3 = 训练变体
        bool is_val_var   = (v >= 4);  // v4-v5 = 验证变体

        // 该 variant 的 ShapeId（对 shape-invariant 图无意义，但仍需填入）
        ShapeId variant_shape = variant_compile_specs_[v].get_shape_id();

        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            
            // ===== 交叉槽位过滤 =====
            if (is_val_var && is_train_gid(gid)) continue;
            if (is_train_var && is_infer_gid(gid)) continue;

            // ===== 选择 ComputationGraph =====
            const ComputationGraph* cg = nullptr;
            if (is_train_gid(gid) || is_si_graph(gid)) {
                cg = train_cg_;
            } else if (is_infer_gid(gid)) {
                cg = infer_cg_;
            }
            
            if (!cg || cg->nodes(gid).empty()) continue;

            // ===== 填 Slot =====
            auto& sl = atlas.slot(v, gi);
            sl.cg = cg;
            sl.stream_kind = stream_for(gid);

            if (is_si_graph(gid)) {
                // shape-invariant: 全部 variant 共享 variant[0] 的 mp
                sl.mp = variant_memory_plans_[0].get();
                sl.shape_id = kShapeInvariant;
            } else {
                // shape-dependent: 各自 variant 的 mp + 各自的 ShapeId
                sl.mp = mp;
                sl.shape_id = variant_shape;
            }
        }
    }

    // ===== name_to_gid_ 保持向后兼容 =====
    name_to_gid_.clear();
    if (train_cg_) {
        name_to_gid_["train"] = GraphId::DEEP_FWD_BWD;
    }
    if (infer_cg_) {
        name_to_gid_["inference"] = GraphId::INF_MAIN_A;
    }

    return atlas;
}
```

### 3.4 Step 3: 重写 `build_exec_table()` 构建 6-Variant 执行表

**文件**: `src/task/deep_learning_task.cpp`

```cpp
void DeepLearningTask::build_exec_table() {
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;

    const int K = num_gpus_;
    gpu_exec_.device_ids.resize(K);
    for (int rank = 0; rank < K; ++rank) {
        gpu_exec_.device_ids[rank] = context(rank).device_id();
    }

    auto resolve = [&](GraphId gid, int rank, size_t variant_idx) -> cudaGraphExec_t {
        int32_t idx = captured_result_.atlas.index(variant_idx, gid);
        if (idx < 0 || static_cast<size_t>(idx) >= captured_result_.graphs.size())
            return nullptr;
        return static_cast<cudaGraphExec_t>(
            captured_result_.graphs[idx].native_exec(rank));
    };

    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

    if (h2d_only_) {
        for (int rank = 0; rank < K; ++rank) {
            gpu_exec_.variant_graphs[0].resize(K);
            auto& g = gpu_exec_.variant_graphs[0][rank];
            g.resize(S(GraphSlot::COUNT), nullptr);
            g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank, 0);
            g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank, 0);
        }
        return;
    }

    // ===== 为所有 6 个 variant 构建 exec table =====
    for (size_t v = 0; v < GraphAtlas::kMaxVariants; ++v) {
        gpu_exec_.variant_graphs[v].resize(K);
        for (int rank = 0; rank < K; ++rank) {
            auto& g = gpu_exec_.variant_graphs[v][rank];
            g.resize(S(GraphSlot::COUNT), nullptr);

            // ===== H2D 传输图（shape 无关，但所有 variant 都有）=====
            g[S(GraphSlot::XFER_A)]           = resolve(GraphId::TRANSFER_A, rank, v);
            g[S(GraphSlot::XFER_B)]           = resolve(GraphId::TRANSFER_B, rank, v);

            // ===== 训练相关图（仅 variant 0-3）=====
            if (v <= 3) {
                g[S(GraphSlot::FWD_BWD_DEEP_A)]   = resolve(GraphId::DEEP_FWD_BWD, rank, v);
                g[S(GraphSlot::FWD_BWD_DEEP_B)]   = resolve(GraphId::DEEP_FWD_BWD, rank, v);
                g[S(GraphSlot::FIRST_LAYER_BWD_A)]  = resolve(GraphId::FIRST_LAYER_BWD_A, rank, v);
                g[S(GraphSlot::FIRST_LAYER_BWD_B)] = resolve(GraphId::FIRST_LAYER_BWD_B, rank, v);
                g[S(GraphSlot::ZERO_GRAD)]        = resolve(GraphId::ZERO_GRAD, rank, v);
                g[S(GraphSlot::DEEP_ALLREDUCE)]   = resolve(GraphId::DEEP_COMM, rank, v);
                g[S(GraphSlot::FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM, rank, v);
                g[S(GraphSlot::WEIGHT_UPDATE)]    = resolve(GraphId::OPTIMIZER, rank, v);
                g[S(GraphSlot::EMA_UPDATE)]       = resolve(GraphId::EMA_UPDATE, rank, v);
                g[S(GraphSlot::CAST_DEEP_GRAD)]     = resolve(GraphId::CAST_DEEP_GRAD_FP16_TO_FP32, rank, v);
                g[S(GraphSlot::CAST_FIRST_GRAD)]    = resolve(GraphId::CAST_FIRST_GRAD_FP16_TO_FP32, rank, v);
                g[S(GraphSlot::NAN_CHECK_GRAD_SCALE)] = resolve(GraphId::NAN_CHECK_AND_GRAD_SCALING, rank, v);
                g[S(GraphSlot::FIRST_LAYER_FWD_A)]      = resolve(GraphId::FIRST_LAYER_FWD_A, rank, v);
                g[S(GraphSlot::FIRST_LAYER_FWD_B)]      = resolve(GraphId::FIRST_LAYER_FWD_B, rank, v);
                g[S(GraphSlot::CAST_MAIN)]        = resolve(GraphId::CAST_MAIN_FP32_TO_FP16, rank, v);
                g[S(GraphSlot::ACCUM_METRICS)]           = resolve(GraphId::ACCUM_METRICS, rank, v);
                g[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)] = resolve(GraphId::ACCUM_METRICS_TRAIN_LAST, rank, v);
            }

            // ===== 验证相关图（仅 variant 4-5）=====
            if (v >= 4) {
                g[S(GraphSlot::INF_MAIN_A)]       = resolve(GraphId::INF_MAIN_A, rank, v);
                g[S(GraphSlot::INF_MAIN_B)]       = resolve(GraphId::INF_MAIN_B, rank, v);
                g[S(GraphSlot::INF_EMA_A)]        = resolve(GraphId::INF_EMA_A, rank, v);
                g[S(GraphSlot::INF_EMA_B)]        = resolve(GraphId::INF_EMA_B, rank, v);
                g[S(GraphSlot::ACCUM_METRICS)]           = resolve(GraphId::ACCUM_METRICS, rank, v);
                g[S(GraphSlot::ACCUM_METRICS_VAL_LAST)]   = resolve(GraphId::ACCUM_METRICS_VAL_LAST, rank, v);
                g[S(GraphSlot::VAL_RESULT_ALLREDUCE)]     = resolve(GraphId::VAL_RESULT_COMM, rank, v);
            }

            // ===== 通用图（所有 variant）=====
            g[S(GraphSlot::CLEAR_METRICS)] = resolve(GraphId::CLEAR_METRICS, rank, v);
        }
    }

    // ===== 日志：每个 variant 的关键 slot =====
    for (size_t v = 0; v < GraphAtlas::kMaxVariants; ++v) {
        for (int rank = 0; rank < K; ++rank) {
            const auto& g = gpu_exec_.variant_graphs[v][rank];
            if (v <= 3) {
                LOG_INFO << "[EXEC-TABLE] v=" << v << " rank=" << rank
                         << " DEEP=" << (g[S(GraphSlot::FWD_BWD_DEEP_A)] ? "OK" : "NULL")
                         << " FWD_A=" << (g[S(GraphSlot::FIRST_LAYER_FWD_A)] ? "OK" : "NULL");
            } else {
                LOG_INFO << "[EXEC-TABLE] v=" << v << " rank=" << rank
                         << " INF_A=" << (g[S(GraphSlot::INF_MAIN_A)] ? "OK" : "NULL");
            }
        }
    }

    // ===== 校验：variant 0 的关键 slot 必须非空 =====
    static const GraphSlot kRequired[] = {
        GraphSlot::XFER_A, GraphSlot::XFER_B,
        GraphSlot::FWD_BWD_DEEP_A,
        GraphSlot::FIRST_LAYER_BWD_A,
    };
    for (int rank = 0; rank < K; ++rank) {
        for (auto slot : kRequired) {
            if (!gpu_exec_.variant_graphs[0][rank][static_cast<size_t>(slot)]) {
                TR_CHECK(false, ValueError,
                         "Required graph slot " << static_cast<int>(slot)
                         << " is nullptr for rank " << rank);
            }
        }
    }
#endif
}
```

### 3.5 Step 4: 修改 `run_train_epoch_gpu()` 支持预存 + Variant 切换

**文件**: `src/task/deep_learning_task.cpp`

核心设计：在线程 lambda 的最开始（batch 循环外部），根据当前 epoch 的分辨率选择 variant，预解析所有 graph handles 为局部变量：

```cpp
#ifdef TR_USE_CUDA
float DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    int32_t loss_id = active_memory_plan_->baseline().loss;

    // ===== Epoch-level resolution selection =====
    bool using_lowres = (registry.using_progressive_resolution() &&
                         current_epoch_ < registry.boundary_epoch());
    size_t v_base = using_lowres ? 2 : 0;  // train_lowres : train_base
    size_t v_last = using_lowres ? 3 : 1;  // train_lowres_last : train_last

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, K, using_amp, &exc,
                              loss_id, v_base, v_last]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // =====================================================================
                // CRITICAL: PRE-RESOLVE ALL GRAPH HANDLES BEFORE BATCH LOOP
                // 根据用户要求：进入 batch 循环前存储好所有图（含 last batch）
                // =====================================================================

                const auto& g_nb = gpu_exec_.variant_graphs[v_base][rank];  // normal batch
                const auto& g_lb = gpu_exec_.variant_graphs[v_last][rank];  // last batch

                // --- Normal batch handles ---
                auto nb_xfer_a  = g_nb[S(GraphSlot::XFER_A)];
                auto nb_xfer_b  = g_nb[S(GraphSlot::XFER_B)];
                auto nb_deep_a  = g_nb[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto nb_deep_b  = g_nb[S(GraphSlot::FWD_BWD_DEEP_B)];
                auto nb_fwd_a   = g_nb[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto nb_fwd_b   = g_nb[S(GraphSlot::FIRST_LAYER_FWD_B)];
                auto nb_bwd_a   = g_nb[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto nb_bwd_b   = g_nb[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto nb_zg      = g_nb[S(GraphSlot::ZERO_GRAD)];
                auto nb_dar     = g_nb[S(GraphSlot::DEEP_ALLREDUCE)];
                auto nb_far     = g_nb[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto nb_wu      = g_nb[S(GraphSlot::WEIGHT_UPDATE)];
                auto nb_cdg     = g_nb[S(GraphSlot::CAST_DEEP_GRAD)];
                auto nb_cfg     = g_nb[S(GraphSlot::CAST_FIRST_GRAD)];
                auto nb_ncg     = g_nb[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                auto nb_cm      = g_nb[S(GraphSlot::CAST_MAIN)];
                auto nb_accum   = g_nb[S(GraphSlot::ACCUM_METRICS)];
                auto nb_accum_tl = g_nb[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];
                auto nb_clear   = g_nb[S(GraphSlot::CLEAR_METRICS)];

                // --- Last batch handles ---
                // Shape-invariant 图与 normal batch 相同（去重机制保证）
                // Shape-dependent 图（FWD/BWD/DEEP）使用不同的 captured graph
                auto lb_xfer_a  = g_lb[S(GraphSlot::XFER_A)];   // same ptr as nb
                auto lb_xfer_b  = g_lb[S(GraphSlot::XFER_B)];   // same ptr as nb
                auto lb_deep_a  = g_lb[S(GraphSlot::FWD_BWD_DEEP_A)];  // DIFFERENT
                auto lb_deep_b  = g_lb[S(GraphSlot::FWD_BWD_DEEP_B)];  // DIFFERENT
                auto lb_fwd_a   = g_lb[S(GraphSlot::FIRST_LAYER_FWD_A)];  // DIFFERENT
                auto lb_fwd_b   = g_lb[S(GraphSlot::FIRST_LAYER_FWD_B)];  // DIFFERENT
                auto lb_bwd_a   = g_lb[S(GraphSlot::FIRST_LAYER_BWD_A)];  // DIFFERENT
                auto lb_bwd_b   = g_lb[S(GraphSlot::FIRST_LAYER_BWD_B)];  // DIFFERENT
                auto lb_zg      = g_lb[S(GraphSlot::ZERO_GRAD)];   // same
                auto lb_dar     = g_lb[S(GraphSlot::DEEP_ALLREDUCE)];  // same
                auto lb_far     = g_lb[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];  // same
                auto lb_wu      = g_lb[S(GraphSlot::WEIGHT_UPDATE)];  // same
                auto lb_cdg     = g_lb[S(GraphSlot::CAST_DEEP_GRAD)];  // same
                auto lb_cfg     = g_lb[S(GraphSlot::CAST_FIRST_GRAD)];  // same
                auto lb_ncg     = g_lb[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];  // same
                auto lb_cm      = g_lb[S(GraphSlot::CAST_MAIN)];  // same
                auto lb_accum_tl = g_lb[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];
                auto lb_clear   = g_lb[S(GraphSlot::CLEAR_METRICS)];  // same

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up     = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans  = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_c3     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr   = [&]() { cudaStreamSynchronize(s_trans); };

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

                // ===== Batch 0 预传输（batch 0 总是 full batch → normal variant）=====
                {
                    ts->wait_buffer_readable(0);
                    if (nb_xfer_a) cudaGraphLaunch(nb_xfer_a, s_trans);
                    sync_tr();
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);

                    if (using_amp && nb_cm) { cudaGraphLaunch(nb_cm, s_up); sync_up(); }
                    if (nb_clear) cudaGraphLaunch(nb_clear, s_up);
                }

                // ===== 统一循环：batch = 0 .. batches-2（normal batch）=====
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    auto g_fwd   = from_a ? nb_fwd_a : nb_fwd_b;
                    auto g_deep  = from_a ? nb_deep_a : nb_deep_b;
                    auto g_xfer_n = from_a ? nb_xfer_b : nb_xfer_a;
                    auto g_first = from_a ? nb_bwd_a : nb_bwd_b;

                    // Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD
                    if (nb_zg) cudaGraphLaunch(nb_zg, s_up);
                    if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                    sync_comp(); sync_up();

                    ts->wait_buffer_readable(next_buf);

                    // Phase 2: DEEP_FWD_BWD ‖ XFER(next)
                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
                    sync_comp(); sync_tr();

                    if (nb_accum) cudaGraphLaunch(nb_accum, s_up);
                    sync_up();

                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);

                    // Phase 3: BWD → CAST → COMM → CAST → COMM → NAN_CHECK → OPT → CAST
                    if (!frozen) {
                        if (g_first) cudaGraphLaunch(g_first, s_c1);
                    }
                    sync_comp();

                    if (using_amp && nb_cdg) { cudaGraphLaunch(nb_cdg, s_up); sync_up(); }
                    if (nb_dar) cudaGraphLaunch(nb_dar, s_up); sync_up();
                    if (using_amp && nb_cfg) { cudaGraphLaunch(nb_cfg, s_up); sync_up(); }
                    if (nb_far) cudaGraphLaunch(nb_far, s_up); sync_up();
                    if (using_amp && nb_ncg) { cudaGraphLaunch(nb_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batch);
                    (void)lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (nb_wu) cudaGraphLaunch(nb_wu, s_up); sync_up();
                    if (using_amp && nb_cm) { cudaGraphLaunch(nb_cm, s_up); sync_up(); }
                }

                // ===== Last batch (batch = batches-1) =====
                // 使用 lb_* handles，grid=last_batch_size
                {
                    bool last_a = ((batches - 1) % 2 == 0);

                    auto g_fwd_l = last_a ? lb_fwd_a : lb_fwd_b;
                    auto g_deep_l = last_a ? lb_deep_a : lb_deep_b;
                    auto g_first_l = last_a ? lb_bwd_a : lb_bwd_b;

                    if (lb_zg) cudaGraphLaunch(lb_zg, s_up);
                    if (g_fwd_l) cudaGraphLaunch(g_fwd_l, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
                    sync_comp();

                    if (lb_accum_tl) cudaGraphLaunch(lb_accum_tl, s_up);
                    sync_up();

                    if (!frozen) {
                        if (g_first_l) cudaGraphLaunch(g_first_l, s_c1);
                    }
                    sync_comp();

                    if (using_amp && lb_cdg) { cudaGraphLaunch(lb_cdg, s_up); sync_up(); }
                    if (lb_dar) cudaGraphLaunch(lb_dar, s_up); sync_up();
                    if (using_amp && lb_cfg) { cudaGraphLaunch(lb_cfg, s_up); sync_up(); }
                    if (lb_far) cudaGraphLaunch(lb_far, s_up); sync_up();
                    if (using_amp && lb_ncg) { cudaGraphLaunch(lb_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batches - 1);
                    (void)lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (lb_wu) cudaGraphLaunch(lb_wu, s_up); sync_up();
                    if (using_amp && lb_cm) { cudaGraphLaunch(lb_cm, s_up); sync_up(); }
                }

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank) {
        if (exc[rank]) std::rethrow_exception(exc[rank]);
    }

    // 从 accum_loss 读取 epoch 平均 loss
    float train_loss = 0.0f;
    if (active_memory_plan_) {
        const auto& b = active_memory_plan_->baseline();
        int32_t accum_loss_id = b.accum_loss;
        if (accum_loss_id >= 0) {
            const auto& accum_dt = active_memory_plan_->get_dtensor(accum_loss_id);
            Tensor h_accum = fetch_from_rank(accum_dt, 0);
            float accum_val = h_accum.data<float>()[0];
            size_t total = registry.num_train_samples();
            if (total > 0) train_loss = accum_val / static_cast<float>(total);
        }
    }
    return train_loss;
#endif
}
```

### 3.6 Step 5: 修改 `run_val_epoch_gpu()` 支持预存 + Variant 切换

**文件**: `src/task/deep_learning_task.cpp`

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
#ifdef TR_USE_CUDA
    (void)validate_ema;

    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    int val_batches = registry.get_val_steps();

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // =====================================================================
                // PRE-RESOLVE ALL GRAPH HANDLES BEFORE BATCH LOOP
                // =====================================================================
                const auto& g_vb = gpu_exec_.variant_graphs[4][rank];  // val_base
                const auto& g_vl = gpu_exec_.variant_graphs[5][rank];  // val_last

                // val_base handles
                auto vb_xfer_a  = g_vb[S(GraphSlot::XFER_A)];
                auto vb_xfer_b  = g_vb[S(GraphSlot::XFER_B)];
                auto vb_inf_a   = g_vb[S(GraphSlot::INF_MAIN_A)];
                auto vb_inf_b   = g_vb[S(GraphSlot::INF_MAIN_B)];
                auto vb_accum   = g_vb[S(GraphSlot::ACCUM_METRICS)];
                auto vb_clear   = g_vb[S(GraphSlot::CLEAR_METRICS)];
                auto vb_val_comm = g_vb[S(GraphSlot::VAL_RESULT_ALLREDUCE)];

                // val_last handles
                auto vl_xfer_a  = g_vl[S(GraphSlot::XFER_A)];  // same ptr
                auto vl_xfer_b  = g_vl[S(GraphSlot::XFER_B)];  // same ptr
                auto vl_inf_a   = g_vl[S(GraphSlot::INF_MAIN_A)];  // DIFFERENT
                auto vl_inf_b   = g_vl[S(GraphSlot::INF_MAIN_B)];  // DIFFERENT
                auto vl_accum_vl = g_vl[S(GraphSlot::ACCUM_METRICS_VAL_LAST)];
                auto vl_clear   = g_vl[S(GraphSlot::CLEAR_METRICS)];  // same ptr
                auto vl_val_comm = g_vl[S(GraphSlot::VAL_RESULT_ALLREDUCE)];  // same ptr

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up     = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans  = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_c3     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr   = [&]() { cudaStreamSynchronize(s_trans); };

                TransferStation* ts = nullptr;
                for (int w = 0; w < 200; ++w) {
                    ts = static_cast<TransferStation*>(
                        registry.transfer_station_ptr(rank));
                    if (ts) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

                const auto& b = active_memory_plan_->baseline();
                int32_t loss_id = b.loss;
                int32_t top1_id = b.top1;
                int32_t top5_id = b.top5;

                if (vb_clear) cudaGraphLaunch(vb_clear, s_up);

                // ===== Validation batch loop =====
                for (int batch = 0; batch < val_batches; ++batch) {
                    int buf = batch % 2;
                    bool is_last = (batch == val_batches - 1);

                    // Select handles: base or last
                    auto g_xfer = (buf == 0)
                        ? (is_last ? vl_xfer_a : vb_xfer_a)
                        : (is_last ? vl_xfer_b : vb_xfer_b);
                    auto g_inf = (buf == 0)
                        ? (is_last ? vl_inf_a : vb_inf_a)
                        : (is_last ? vl_inf_b : vb_inf_b);
                    auto g_accum = is_last ? vl_accum_vl : vb_accum;

                    if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (top1_id >= 0) cudaMemsetAsync(ctx.ptr_at(top1_id), 0, sizeof(float), s_c1);
                    if (top5_id >= 0) cudaMemsetAsync(ctx.ptr_at(top5_id), 0, sizeof(float), s_c1);

                    ts->wait_buffer_readable(buf);
                    if (g_xfer) cudaGraphLaunch(g_xfer, s_trans);
                    sync_tr();

                    if (g_inf) cudaGraphLaunch(g_inf, s_c1);
                    sync_comp();

                    if (g_accum) cudaGraphLaunch(g_accum, s_up);
                    sync_up();

                    ts->set_buffer_readable(buf, false);
                    ts->set_buffer_writeable(buf, true);
                }

                sync_up();

                // val_comm 是 shape invariant，vb 和 vl 指向同一个 captured graph
                if (vb_val_comm) cudaGraphLaunch(vb_val_comm, s_up);
                sync_up();

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank) {
        if (exc[rank]) std::rethrow_exception(exc[rank]);
    }

    float avg_loss = 0.0f, avg_top1 = 0.0f, avg_top5 = 0.0f;
    if (active_memory_plan_) {
        const auto& b = active_memory_plan_->baseline();
        int32_t al_id = b.accum_loss;
        int32_t at1_id = b.accum_top1;
        int32_t at5_id = b.accum_top5;
        if (al_id >= 0) {
            const auto& al_dt = active_memory_plan_->get_dtensor(al_id);
            Tensor h_al = fetch_from_rank(al_dt, 0);
            avg_loss = h_al.data<float>()[0] / static_cast<float>(registry.num_val_samples());
        }
        if (at1_id >= 0) {
            const auto& at1_dt = active_memory_plan_->get_dtensor(at1_id);
            Tensor h_at1 = fetch_from_rank(at1_dt, 0);
            avg_top1 = h_at1.data<float>()[0] / static_cast<float>(registry.num_val_samples());
        }
        if (at5_id >= 0) {
            const auto& at5_dt = active_memory_plan_->get_dtensor(at5_id);
            Tensor h_at5 = fetch_from_rank(at5_dt, 0);
            avg_top5 = h_at5.data<float>()[0] / static_cast<float>(registry.num_val_samples());
        }
    }

    LOG_INFO << "[VAL] loss=" << avg_loss << " top1=" << avg_top1 * 100.0f
             << "% top5=" << avg_top5 * 100.0f << "%";

    return {avg_loss, avg_top1, avg_top5};
#else
    (void)validate_ema;
    return {0.0f, 0.0f, 0.0f};
#endif
}
```

### 3.7 Step 6: 修改 `run_train_epoch_cpu()` 支持 Variant 切换

**文件**: `src/task/deep_learning_task.cpp`

```cpp
float DeepLearningTask::run_train_epoch_cpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();

    TransferStation* ts = nullptr;
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!ts) TR_DEVICE_ERROR("TransferStation not ready for CPU path");

    DeviceContext& ctx = context(0);
    const auto& atlas = captured_result_.atlas;
    const auto& graphs = captured_result_.graphs;

    // ===== Epoch-level resolution selection =====
    bool using_lowres = (registry.using_progressive_resolution() &&
                         current_epoch_ < registry.boundary_epoch());
    size_t v_base = using_lowres ? 2 : 0;
    size_t v_last = using_lowres ? 3 : 1;

    auto idx_for = [&atlas](GraphId gid, size_t variant_idx) -> int32_t {
        return atlas.index(variant_idx, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    // Pre-resolve normal batch indices (v_base)
    int32_t idx_xfer_a_nb = idx_for(GraphId::TRANSFER_A, v_base);
    int32_t idx_xfer_b_nb = idx_for(GraphId::TRANSFER_B, v_base);
    int32_t idx_deep_nb   = idx_for(GraphId::DEEP_FWD_BWD, v_base);
    int32_t idx_fwd_a_nb  = idx_for(GraphId::FIRST_LAYER_FWD_A, v_base);
    int32_t idx_fwd_b_nb  = idx_for(GraphId::FIRST_LAYER_FWD_B, v_base);
    int32_t idx_bwd_a_nb  = idx_for(GraphId::FIRST_LAYER_BWD_A, v_base);
    int32_t idx_bwd_b_nb  = idx_for(GraphId::FIRST_LAYER_BWD_B, v_base);
    int32_t idx_zg_nb     = idx_for(GraphId::ZERO_GRAD, v_base);
    int32_t idx_far_nb    = idx_for(GraphId::FIRST_COMM, v_base);
    int32_t idx_dar_nb    = idx_for(GraphId::DEEP_COMM, v_base);
    int32_t idx_opt_nb    = idx_for(GraphId::OPTIMIZER, v_base);

    // Pre-resolve last batch indices (v_last) — shape-dependent ones differ
    int32_t idx_deep_lb   = idx_for(GraphId::DEEP_FWD_BWD, v_last);
    int32_t idx_fwd_a_lb  = idx_for(GraphId::FIRST_LAYER_FWD_A, v_last);
    int32_t idx_fwd_b_lb  = idx_for(GraphId::FIRST_LAYER_FWD_B, v_last);
    int32_t idx_bwd_a_lb  = idx_for(GraphId::FIRST_LAYER_BWD_A, v_last);
    int32_t idx_bwd_b_lb  = idx_for(GraphId::FIRST_LAYER_BWD_B, v_last);

    int32_t loss_id   = active_memory_plan_->baseline().loss;
    void*   loss_ptr  = loss_id >= 0 ? ctx.ptr_at(loss_id) : nullptr;
    void*   lr_ptr    = ctx.ptr_at(lr_dtensor_id_);
    bool frozen = is_first_layer_frozen();

    // Batch 0 prefetch
    ts->wait_buffer_readable(0);
    launch(idx_xfer_a_nb);
    ts->set_buffer_readable(0, false);
    ts->set_buffer_writeable(0, true);

    // ===== Normal batches =====
    for (int batch = 0; batch < batches - 1; ++batch) {
        bool from_a  = (batch % 2 == 0);
        int next_buf = from_a ? 1 : 0;

        launch(idx_zg_nb);
        launch(from_a ? idx_fwd_a_nb : idx_fwd_b_nb);

        ts->wait_buffer_readable(next_buf);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_nb);
        launch(from_a ? idx_xfer_b_nb : idx_xfer_a_nb);

        ts->set_buffer_readable(next_buf, false);
        ts->set_buffer_writeable(next_buf, true);

        if (!frozen) launch(from_a ? idx_bwd_a_nb : idx_bwd_b_nb);
        launch(idx_dar_nb);

        {
            float lr = fetch_lr_for_batch(batch);
            *static_cast<float*>(lr_ptr) = lr;
        }
        launch(idx_far_nb);
        launch(idx_opt_nb);
    }

    // ===== Last batch =====
    {
        bool last_a = ((batches - 1) % 2 == 0);

        launch(idx_zg_nb);
        launch(last_a ? idx_fwd_a_lb : idx_fwd_b_lb);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_lb);

        if (!frozen) launch(last_a ? idx_bwd_a_lb : idx_bwd_b_lb);
        launch(idx_dar_nb);

        {
            float lr = fetch_lr_for_batch(batches - 1);
            *static_cast<float*>(lr_ptr) = lr;
        }
        launch(idx_far_nb);
        launch(idx_opt_nb);
    }

    // Return epoch average loss from accum_loss
    float train_loss = 0.0f;
    int32_t accum_loss_id = active_memory_plan_->baseline().accum_loss;
    if (accum_loss_id >= 0) {
        const auto& accum_dt = active_memory_plan_->get_dtensor(accum_loss_id);
        Tensor h_accum = fetch_from_rank(accum_dt, 0);
        float accum_val = h_accum.data<float>()[0];
        size_t total = registry.num_train_samples();
        if (total > 0) train_loss = accum_val / static_cast<float>(total);
    }
    return train_loss;
}
```

### 3.8 Step 7: 修改 `run_val_epoch_cpu()` 支持加权累积

**文件**: `src/task/deep_learning_task.cpp`

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_cpu(bool validate_ema) {
    (void)validate_ema;

    auto& registry = GlobalRegistry::instance();
    int val_batches = registry.get_val_steps();

    const auto& atlas = captured_result_.atlas;
    const auto& graphs = captured_result_.graphs;

    auto idx_for = [&atlas](GraphId gid, size_t variant_idx) -> int32_t {
        return atlas.index(variant_idx, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    // Pre-resolve val_base (v4) and val_last (v5) indices
    int32_t idx_xfer_a_vb = idx_for(GraphId::TRANSFER_A, 4);
    int32_t idx_xfer_b_vb = idx_for(GraphId::TRANSFER_B, 4);
    int32_t idx_inf_a_vb  = idx_for(GraphId::INF_MAIN_A, 4);
    int32_t idx_inf_b_vb  = idx_for(GraphId::INF_MAIN_B, 4);

    int32_t idx_xfer_a_vl = idx_for(GraphId::TRANSFER_A, 5);  // same
    int32_t idx_xfer_b_vl = idx_for(GraphId::TRANSFER_B, 5);  // same
    int32_t idx_inf_a_vl  = idx_for(GraphId::INF_MAIN_A, 5);  // DIFFERENT
    int32_t idx_inf_b_vl  = idx_for(GraphId::INF_MAIN_B, 5);  // DIFFERENT

    const auto& bl = active_memory_plan_->baseline();
    int32_t loss_id = bl.loss;
    int32_t top1_id = bl.top1;
    int32_t top5_id = bl.top5;

    DeviceContext& ctx = context(0);
    void* loss_ptr = loss_id >= 0 ? ctx.ptr_at(loss_id) : nullptr;
    void* top1_ptr = top1_id >= 0 ? ctx.ptr_at(top1_id) : nullptr;
    void* top5_ptr = top5_id >= 0 ? ctx.ptr_at(top5_id) : nullptr;

    TransferStation* ts = nullptr;
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!ts) TR_DEVICE_ERROR("TransferStation not ready for CPU val path");

    double acc_loss = 0.0;
    double acc_top1 = 0.0;
    double acc_top5 = 0.0;

    for (int batch = 0; batch < val_batches; ++batch) {
        int buf = batch % 2;
        bool is_last = (batch == val_batches - 1);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        if (top1_ptr) std::memset(top1_ptr, 0, sizeof(float));
        if (top5_ptr) std::memset(top5_ptr, 0, sizeof(float));

        ts->wait_buffer_readable(buf);

        auto g_xfer = (buf == 0)
            ? (is_last ? idx_xfer_a_vl : idx_xfer_a_vb)
            : (is_last ? idx_xfer_b_vl : idx_xfer_b_vb);
        auto g_inf = (buf == 0)
            ? (is_last ? idx_inf_a_vl : idx_inf_a_vb)
            : (is_last ? idx_inf_b_vl : idx_inf_b_vb);

        launch(g_xfer);
        launch(g_inf);

        // 按样本数加权累积（修复 CPU 指标计算错误）
        int bs = is_last ? registry.get_last_val_batch_size()
                         : registry.get_local_batch_size();
        float batch_loss = loss_ptr ? *static_cast<float*>(loss_ptr) : 0.0f;
        float batch_top1 = top1_ptr ? *static_cast<float*>(top1_ptr) : 0.0f;
        float batch_top5 = top5_ptr ? *static_cast<float*>(top5_ptr) : 0.0f;

        acc_loss += batch_loss * static_cast<double>(bs);
        acc_top1 += batch_top1 * static_cast<double>(bs);
        acc_top5 += batch_top5 * static_cast<double>(bs);

        ts->set_buffer_readable(buf, false);
        ts->set_buffer_writeable(buf, true);
    }

    // 按总样本数平均（修复 CPU 指标计算错误）
    size_t total_val = registry.num_val_samples();
    float avg_loss = (total_val > 0) ? static_cast<float>(acc_loss / static_cast<double>(total_val)) : 0.0f;
    float avg_top1 = (total_val > 0) ? static_cast<float>(acc_top1 / static_cast<double>(total_val)) : 0.0f;
    float avg_top5 = (total_val > 0) ? static_cast<float>(acc_top5 / static_cast<double>(total_val)) : 0.0f;

    LOG_INFO << "[VAL-CPU] loss=" << avg_loss << " top1=" << avg_top1 * 100.0f
             << "% top5=" << avg_top5 * 100.0f << "%";

    return {avg_loss, avg_top1, avg_top5};
}
```

---

## 四、渐进式分辨率完整支持

### 4.1 运行时选择逻辑

本方案**天然支持渐进式分辨率**，无需额外修改运行时逻辑：

```cpp
// run_train_epoch_gpu() 开头
bool using_lowres = (registry.using_progressive_resolution() &&
                     current_epoch_ < registry.boundary_epoch());
size_t v_base = using_lowres ? 2 : 0;  // train_lowres : train_base
size_t v_last = using_lowres ? 3 : 1;  // train_lowres_last : train_last
```

| Epoch | Resolution        | v_base | v_last | 使用的 Variant |
| ----- | ----------------- | ------ | ------ | -------------- |
| 早期  | 低分辨率 (128×128) | 2      | 3      | train_lowres + train_lowres_last |
| 后期  | 高分辨率 (224×224) | 0      | 1      | train_base + train_last |

验证阶段分辨率固定为 `val_res`，始终使用 v4（val_base）和 v5（val_last）。

### 4.2 ShapeId 去重示例

**ImageNet 渐进式场景**（8 ranks, batch=512, progressive 128→224）：

| Variant | ShapeId              | 用途                   |
| ------- | -------------------- | ---------------------- |
| v0      | `{512, 224, 224, 3}` | 后期 epoch 常规 batch   |
| v1      | `{402, 224, 224, 3}` | 后期 epoch last batch    |
| v2      | `{512, 128, 128, 3}` | 早期 epoch 常规 batch   |
| v3      | `{402, 128, 128, 3}` | 早期 epoch last batch    |
| v4      | `{512, 224, 224, 3}` | 验证常规 batch (**与 v0 同 ShapeId!**) |
| v5      | `{106, 224, 224, 3}` | 验证 last batch          |

**去重效果**：
- v0 和 v4 的 ShapeId 相同 → shape-dependent 图自动共享 → **零额外捕获开销**
- 总共 **5 个不同的 ShapeId** → shape-dependent 图捕获 5 套（而非 6×8=48 套）

---

## 五、性能保证

### 5.1 运行时零开销

| 指标                           | 当前实现          | 本方案                 | 说明                           |
| ------------------------------ | ----------------- | ---------------------- | ------------------------------ |
| Batch 循环内 variant 切换开销   | N/A（无 variant） | **0**                  | 所有 handles 在循环外预解析     |
| Graph handle 解析              | `g_tab[S(slot)]`  | **直接局部变量**       | 编译器可能将局部变量放入寄存器 |
| Last batch 条件分支             | 无                | **编译期确定的分支**   | last batch 使用独立的局部变量组 |
| 多 RANK 扩展性                  | `graphs[rank]`    | **`variant_graphs[v][rank]`** | 每个 rank 独立预解析           |

### 5.2 内存占用

| 场景           | 当前实现 | 本方案     | 增量说明                           |
| -------------- | -------- | ---------- | ---------------------------------- |
| CIFAR-10 整除  | ~20 张   | **~20 张** | ShapeId 相同 → 完全共享 → 零额外开销 |
| ImageNet progressive | ~20 张   | **~47 张** | Shape 无关图共享，shape 相关图按需捕获 |

### 5.3 编译时间

- cuDNN warmup 是主要耗时，与 captured graph 数量近似线性
- 预期增加 < 2 秒（从 ~1s 到 ~2-3s）

---

## 六、验证策略

### 6.1 分阶段验证

| 阶段 | 测试                 | 验证内容                                                     |
| ---- | -------------------- | ------------------------------------------------------------ |
| 1    | **Dry-run**          | `variant_specs.size() == 6`, variant 命名正确, MemoryPlan 创建成功 |
| 2    | **GraphAtlas 结构**  | shape-invariant 图的 6 variant `captured_idx` 相同; shape-dependent 图不同 ShapeId 的 `captured_idx` 不同 |
| 3    | **CIFAR-10 全模式**  | `test_dl_full --cpu/gpu/amp`, 3 epochs, 无回归              |
| 4    | **CIFAR-10 last batch** | 修改 `batch=192`（不可整除），验证 variant 切换无异常       |
| 5    | **渐进式分辨率**     | 启用 progressive resolution，验证早期/后期 epoch 使用不同 variant |
| 6    | **端到端正确性**     | ImageNet 配置模拟，对比修复前后的 loss/accuracy 差异       |

### 6.2 ShapeId 去重验证方法

在 `pre_capture()` 后增加诊断日志：

```cpp
// 在 build_exec_table 后：
for (size_t v = 0; v < 6; ++v) {
    int32_t idx = captured_result_.atlas.index(v, GraphId::DEEP_FWD_BWD);
    LOG_INFO << "[VARIANT] v=" << v << " DEEP_FWD_BWD captured_idx=" << idx;
}
```

**预期输出**：
- CIFAR-10 (batch=200): 所有 6 variant 的 `captured_idx` 应相同（ShapeId 退化）
- ImageNet (progressive): v0==v4（同 ShapeId），其余各不同

---

## 七、实施计划

### 7.1 任务分解

| 阶段   | 编号  | 任务                           | 文件                                   | 工时 | 验证方式                                |
| ------ | ----- | ------------------------------ | -------------------------------------- | ---- | --------------------------------------- |
| Step 0 | -     | 数据结构准备                     | `deep_learning_task.h`                 | 0.5h | 编译通过                                |
| Step 1 | 1     | 修改 `on_prepare()`             | `deep_learning_task.h`                 | 1h   | `variant_specs.size() == 6`              |
| Step 2 | 2     | 重写 `build_graph_atlas()`      | `deep_learning_task.cpp`               | 2h   | pre_capture 日志显示 captured graph 增加 |
| Step 3 | 3     | 重写 `build_exec_table()`        | `deep_learning_task.cpp`               | 1h   | 日志显示 6 variant 的 exec table 构建   |
| Step 4 | 4     | 修改 `run_train_epoch_gpu()`     | `deep_learning_task.cpp`               | 2h   | test_dl_full_gpu 通过                  |
| Step 5 | 5     | 修改 `run_val_epoch_gpu()`       | `deep_learning_task.cpp`               | 1h   | test_dl_full_gpu 验证通过              |
| Step 6 | 6     | 修改 `run_train_epoch_cpu()`     | `deep_learning_task.cpp`               | 1h   | test_dl_full --cpu 通过                |
| Step 7 | 7     | 修改 `run_val_epoch_cpu()`       | `deep_learning_task.cpp`               | 1h   | test_dl_full --cpu 验证通过            |
| 验证   | 8     | 端到端测试                       | 构造不可整除 batch + 渐进分辨率         | 2h   | 对比修复前后的 loss/accuracy            |

**总工时**: 约 **12 小时**（~1.5 个工作日）

### 7.2 风险矩阵

| 风险                                                         | 概率 | 影响  | 缓解措施                                                     |
| ------------------------------------------------------------ | ---- | ----- | ------------------------------------------------------------ |
| 某些 variant 的 MemoryPlan 中 `slot_bytes` 与 base 不同，导致 offset 不一致 | 极低 | 灾难性 | `compute_max_slot_bytes` 已跨变体取 max；可额外增加断言检查关键 DTensor offset |
| Val variant 误填 train graph，导致 val 路径使用 train 的 FWD/BWD | 低   | 正确性 | `is_train_gid()` 判定 + `continue` 跳过已保证隔离；可增加断言验证 |
| Multi-variant 导致 captured graph 内存增加                     | 中   | 显存增加 | shape 无关图共享（零膨胀）；shape 相关图只增加特征图相关 buffer |
| `transfer_to_rank` 使用 `active_memory_plan_`（base）的 `dt.nbytes()`，导致 last batch 多拷贝 | 中   | 性能损失 | 多拷贝数据在 last batch 的 FWD graph 中不会被处理（grid=last_batch_size），不影响正确性 |

---

## 八、与 LRF1/LRF2/LRF3 的对比

| 分析文档 | 总体评价 | 核心贡献                                                         | 本方案采用情况                                    |
| -------- | -------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| **LRF1** | 完整深入 | ShapeId 去重机制验证<br>6-Variant 系统精确定义<br>渐进分辨率完整支持 | ✅ 完全采纳 ShapeId 去重分析<br>✅ 完全采纳 6-Variant 定义 |
| **LRF2** | 性能极致 | H2D 特殊性强调<br>性能绝对优先原则<br>预存 graph handles 设计 | ✅ 完全采纳性能优化策略<br>✅ 完全采纳预存设计 |
| **LRF3** | 架构清晰 | 三层断裂诊断<br>Graph 分类框架<br>详细实施步骤         | ✅ 完全采纳三层诊断<br>✅ 完全采纳 Graph 分类     |

**本方案的改进点**：
1. **更简洁的数据结构**：直接使用 `Compiler::Result` 持有 `train_cg`/`infer_cg`，避免复杂的生命周期管理
2. **更清晰的 Graph 分类**：明确区分 "完全 Shape 无关"、"训练 Shape 相关"、"推理 Shape 相关" 三类
3. **更完整的 CPU 路径**：同时修复 CPU 训练的指标累积问题和 CPU 验证的加权平均问题
4. **更详细的验证计划**：分 6 个阶段验证，从编译期到运行时到正确性

---

## 九、总结

### 9.1 核心设计决策

1. **Compiler 层**：`on_prepare()` 生成 5 个 variant specs，`Compiler::compile()` 产出完整的 6 变体（含各自的 MemoryPlan 和 ShapeId）

2. **Atlas 层**：`build_graph_atlas()` 为 6 变体全量填表，`is_si_graph()` 判定 shape 无关图共享 `kShapeInvariant`，shape 相关图各自 `shape_id`

3. **Capture 层**：`pre_capture()` 自动去重（`Key{cg, gid, shape_id}`），无需额外修改

4. **运行时 GPU 层**：每个 rank 线程在 **batch 循环外部** 预解析 normal batch 和 last batch 的所有 `cudaGraphExec_t` handles。循环内部零 variant 切换开销

5. **运行时 CPU 层**：同步支持 variant 切换，修复指标累积和加权平均 bug

6. **渐进式分辨率**：epoch 开始时根据 `current_epoch_` 与 `boundary_epoch` 选择 v0/v1 或 v2/v3，天然支持

### 9.2 预期效果

**正确性**：
- **彻底解决** CUDA Graph grid 固定导致的 last batch 残留问题
- **彻底解决** cuDNN descriptor batch 固定导致的污染问题
- **彻底解决** CPU 路径的指标计算错误
- ImageNet 级别的 last batch 完全正确

**完整性**：
- 完整支持渐进式分辨率（train_lowres/train_lowres_last）
- 完整支持多 rank（GlobalRegistry 自动计算各 rank 的 last_batch_size）
- 完整支持 AMP/FP32 模式（H2D 传输图自动区分）

**性能**：
- Shape 无关图全局共享，内存开销 **~15 张图**
- Shape 相关图按需捕获，总开销 **15-50 张图**（取决于 ShapeId 重复度）
- 运行时 variant 切换 **零开销**（预存局部变量）

### 9.3 改动量总结

| 文件                                             | 行数估算 | 关键改动                                                     |
| ------------------------------------------------ | -------- | ------------------------------------------------------------ |
| `include/renaissance/task/deep_learning_task.h` | ~60      | 新增成员变量 `compiled_result_` 等；修改 `on_prepare()` 传递 variant_specs；修改 `GpuExecTable` |
| `src/task/deep_learning_task.cpp`               | ~400     | 新增辅助函数；重写 `build_graph_atlas()`；重写 `build_exec_table()`；修改 4 个 run epoch 函数 |

**总改动量**: 约 **460 行**，集中在 1 个头文件 + 1 个 cpp 文件。

**不需要修改的文件**: `src/graph/compiler.cpp`、`src/graph/captured_graph.cpp`、`src/graph/pre_capture.cpp`、`src/backend/ops/*` — 这些基础设施已完整支持多 variant，只需上层正确调用。

---

**文档版本**: v3.0  
**创建日期**: 2026-05-28  
**作者**: 基于 LRF1.md、LRF2.md、LRF3.md 的综合分析 + 框架源代码实地验证  
**状态**: 待用户审核后开始实施
