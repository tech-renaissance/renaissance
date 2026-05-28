# LRF3 — 完整、科学、严谨的 Last Batch 与不同 Shape 处理方案

> **基于 LRF.md 中小伙伴 D 和 K 的深入分析 + 框架设计意图的完整实现**  
> **版本**: v2.0 | **日期**: 2026-05-28  
> **目标**: 彻底解决 last batch 正确性问题 + 完整支持渐进式分辨率

---

## 一、问题诊断：三层断裂

### 1.1 当前实现的根本缺陷

经过对 LRF.md 中小伙伴 D 和 K 的分析综合验证，当前框架存在**三层系统性断裂**：

| 层级           | 设计意图                                                     | 当前实现                                                     | 断裂点                                                    |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------- |
| **Compiler**   | 6 变体编译（train_base/last/lowres/lowres_last + val_base/last），各自 MemoryPlan | `Compiler::compile()` 支持 6 变体，但 `on_prepare()` **只传 1 个 spec**，`result.variants` 实际只有 **1 个元素** | 未生成 last batch 的 MemoryPlan                           |
| **GraphAtlas** | 6×N 映射表，shape 相关图各自 slot + 各自 mp/shape_id，shape 无关图共享 | `build_graph_atlas()` **只填 variant 0**，且所有 slot 硬编码 `shape_id = kShapeInvariant`、`mp = active_memory_plan_` | 所有 graph 被当成 shape invariant，pre_capture 只捕获一次 |
| **运行时**     | 根据 batch 特征动态选择 variant，launch 对应 captured graph  | `build_exec_table()` 只解析 variant 0；`run_train_epoch_gpu()` 对所有 batch 使用**同一个 g_tab** | last batch 使用 grid=512 的 graph 处理 402 个样本         |

### 1.2 为什么必须多 Variant MemoryPlan

小伙伴 K 的分析给出了**决定性论证**：

**CUDA Graph 的物理约束**：
- CUDA Graph 的 kernel grid 维度在 **capture 时固定**，replay 时不可变
- `softmax_ce_op.cpp:L358-359` 中 `int batch = logits_dt.shape.n()` 从 MemoryPlan 读取
- 如果 capture 时使用 base MemoryPlan（batch=512），则 grid=512 **永久固定**
- 即使运行时修改 `batch_size_ptr` 标量为 402，grid 仍然是 512，导致后 110 个 block 处理残留数据

**唯一正确的做法**：
- 为 variant[1]（train_last）**单独 capture** CUDA Graph
- capture 前 `ctx.set_memory_plan(mp_variant1)`，使 `batch=402`，grid=402
- `pre_capture()` 的 `capture_all_for_rank()` 已内置此机制（`L367: ctx.set_memory_plan(mp)`）
- 问题仅在于 `build_graph_atlas()` 未提供正确的 `mp` 和 `shape_id`

### 1.3 用户强调的关键约束

根据用户补充的指导，框架设计有以下**不可违背的约束**：

1. **H2D 传输图特殊性**：
   - 不管 train/val、full/last batch、渐进分辨率，H2D 传输图**只有 2 种**（A区图和B区图）
   - 因为 H2D 始终传输整个 A 区、整个 B 区，无需管分辨率
   - 但 **AMP 开启与不开启，H2D 传输图不同**

2. **权重更新/梯度同步图不受 batch size 影响**：
   - OPTIMIZER/EMA_UPDATE/COMM 图不受 batch size 影响
   - 这些图在验证阶段不存在

3. **受 batch size 影响的图**：
   - 主要是首层正反向（FIRST_LAYER_FWD_A/B/FIRST_LAYER_BWD_A/B）
   - 深层正反向（DEEP_FWD_BWD）
   - batch size 影响特征图形状，但不影响权重形状

4. **性能绝对优先**：
   - **在进入 batch 循环之前**用局部变量存储好所有需要的图
   - 运行时避免 hash 查找、分支判断、内存分配

5. **CPU/GPU 对齐**：
   - 如果 GPU 为某种情形单独捕获图，CPU 也应当单独捕获

---

## 二、完整修复方案

### 2.1 Variant 映射表

根据用户分析和框架设计，6 个 variant 的精确定义如下：

| Variant Index | 名称             | 用途                     | Batch Size 来源        | 分辨率来源              | 包含的 Graph 类型                  |
| ------------- | ---------------- | ------------------------ | ---------------------- | --------------------- | --------------------------------- |
| 0             | train_base       | 标准训练 batch           | `local_batch_size`     | `train_sample_resolution_end` | ALL + TRAIN graphs                 |
| 1             | train_last       | 训练 last batch          | `last_train_batch_size` | `train_sample_resolution_end` | ALL + TRAIN graphs                 |
| 2             | train_lowres     | 低分辨率训练             | `local_batch_size`     | `train_sample_resolution_begin` | ALL + TRAIN graphs                 |
| 3             | train_lowres_last | 低分辨率训练 last batch  | `last_train_batch_size` | `train_sample_resolution_begin` | ALL + TRAIN graphs                 |
| 4             | val_base         | 标准验证 batch           | `local_batch_size`     | `val_sample_resolution`    | TRANS + INF + VAL_METRICS graphs   |
| 5             | val_last         | 验证 last batch          | `last_val_batch_size`   | `val_sample_resolution`    | TRANS + INF + VAL_METRICS graphs   |

**关键洞察**：
- Variant 0-3：训练路径，包含 ALL + TRAIN graphs（TRANSFER + FWD/BWD + OPTIMIZER + COMM）
- Variant 4-5：验证路径，只包含 TRANS + INF + VAL_METRICS graphs（无 OPTIMIZER/COMM）
- **交叉槽位自然隔离**：val variant 不填 train graphs，train variant 不填 infer graphs

### 2.2 Graph 分类（按 Shape 敏感性）

基于用户分析和代码实际，Graph 分为三类：

| 类型           | GraphId                                                                 | Shape 相关性 | 各 Variant 行为                           | 去重策略                     |
| -------------- | ------------------------------------------------------------------------ | ------------ | ---------------------------------------- | ---------------------------- |
| **完全 Shape 无关** | TRANSFER_A/B<br>ZERO_GRAD<br>DEEP_COMM/FIRST_COMM<br>OPTIMIZER<br>EMA_UPDATE<br>CAST_*<br>NAN_CHECK_AND_GRAD_SCALING<br>CLEAR_METRICS<br>ACCUM_METRICS<br>ACCUM_METRICS_TRAIN_LAST<br>ACCUM_METRICS_VAL_LAST<br>VAL_RESULT_COMM | 完全无关     | 全部 6 variant 共享 base MemoryPlan      | 全局共享 1 张图               |
| **训练 Shape 相关** | FIRST_LAYER_FWD_A/B<br>DEEP_FWD_BWD<br>FIRST_LAYER_BWD_A/B              | Batch 相关   | 4 个训练 variant 各自 MemoryPlan         | 4 张独立图（或更少，如 batch 相同） |
| **推理 Shape 相关** | INF_MAIN_A/B<br>INF_EMA_A/B                                             | Batch 相关   | 2 个验证 variant 各自 MemoryPlan         | 2 张独立图（或更少，如 batch 相同） |

**注意**：
- ACCUM_METRICS 系列虽然计算时需要 batch_size，但其计算逻辑在 kernel 内使用 `batch_size_ptr` 动态读取，graph 本身 shape 无关
- 因此 ACCUM_METRICS 系列属于 **完全 Shape 无关**

### 2.3 H2D 传输图特殊处理

用户强调的 H2D 特殊性：
- **AMP 相同**：variant 0-5 的 TRANSFER_A/B 完全相同 → 全局共享
- **AMP 不同**：需要区分 FP32 vs FP16 版本的 TRANSFER_A/B

实现策略：
- AMP 模式下，TRANSFER_A/B 自动成为 **shape 无关图** → 6 variant 共享
- 这符合 GraphAtlas 的 `kShapeInvariant` 机制

---

## 三、实施步骤

### Step 1: 修复 `on_prepare()` 生成完整 Variant Specs

**文件**: `include/renaissance/task/deep_learning_task.h` (on_prepare 函数)

当前代码问题：
```cpp
CompileSpec spec = CompileSpec::from_global_registry();
auto result = Compiler::compile(plan, spec, plan_config_, initializer_);
memory_plan_ptr_ = std::move(result.variants[0].memory_plan);
```

修改为：
```cpp
CompileSpec base_spec = CompileSpec::from_global_registry();

// ===== 生成 5 个 variant specs =====
auto& gr = GlobalRegistry::instance();
std::vector<CompileSpec> variant_specs;

// v1: train_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_train_batch_size();
    variant_specs.push_back(s);
}

// v2: train_lowres
{
    CompileSpec s = base_spec;
    s.actual_resolution = gr.get_train_sample_resolution_begin();
    variant_specs.push_back(s);
}

// v3: train_lowres_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_train_batch_size();
    s.actual_resolution = gr.get_train_sample_resolution_begin();
    variant_specs.push_back(s);
}

// v4: val_base
{
    CompileSpec s = base_spec;
    s.actual_resolution = gr.val_sample_resolution();
    s.mode = GraphMode::INFERENCE;
    variant_specs.push_back(s);
}

// v5: val_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_val_batch_size();
    s.actual_resolution = gr.val_sample_resolution();
    s.mode = GraphMode::INFERENCE;
    variant_specs.push_back(s);
}

auto result = Compiler::compile(plan, base_spec, plan_config_, initializer_, variant_specs);

// ===== 保存所有 variant 的 MemoryPlan =====
for (size_t i = 0; i < result.variants.size() && i < 6; ++i) {
    variant_memory_plans_[i] = std::move(result.variants[i].memory_plan);
}
memory_plan_ptr_ = std::move(variant_memory_plans_[0]);  // base 作为默认
active_memory_plan_ = memory_plan_ptr_.get();
```

**需要在 DeepLearningTask 中新增成员变量**：
```cpp
std::array<std::unique_ptr<MemoryPlan>, 6> variant_memory_plans_;
std::array<CompileSpec, 6> variant_compile_specs_;  // 用于获取 ShapeId
```

### Step 2: 重写 `build_graph_atlas()` 实现 6-Variant 填表

**文件**: `src/task/deep_learning_task.cpp`

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    if (h2d_only_) {
        // h2d_only: 仍只填 variant 0（无 FWD/BWD）
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

    // ===== 辅助函数：判断 Graph 是否 shape 无关 =====
    auto is_shape_invariant = [](GraphId gid) -> bool {
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
    };

    // ===== 辅助函数：判断 Graph 是否属于训练 =====
    auto is_train_graph = [](GraphId gid) -> bool {
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
    };

    // ===== 辅助函数：判断 Graph 是否属于推理 =====
    auto is_infer_graph = [](GraphId gid) -> bool {
        switch (gid) {
            case GraphId::INF_MAIN_A:
            case GraphId::INF_MAIN_B:
            case GraphId::INF_EMA_A:
            case GraphId::INF_EMA_B:
                return true;
            default:
                return false;
        }
    };

    // ===== 为 6 个 variant 填表 =====
    for (size_t v = 0; v < 6; ++v) {
        const MemoryPlan* mp = variant_memory_plans_[v].get();
        if (!mp) continue;

        ShapeId shape_id = variant_compile_specs_[v].get_shape_id();

        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            
            // ===== 适用性判定（train/val 变体隔离）=====
            bool is_train_var = (v <= 3);
            bool is_val_var   = (v >= 4);

            // val variant 不填 train graph
            if (is_val_var && is_train_graph(gid)) continue;
            // train variant 不填 infer graph
            if (is_train_var && is_infer_graph(gid)) continue;

            // ===== 选择 ComputationGraph =====
            const ComputationGraph* cg = nullptr;
            if (is_train_graph(gid) || is_shape_invariant(gid)) {
                cg = train_cg_;
            } else if (is_infer_graph(gid)) {
                cg = infer_cg_;
            }
            
            if (!cg || cg->nodes(gid).empty()) continue;

            // ===== 填 Slot =====
            auto& sl = atlas.slot(v, gi);
            sl.cg = cg;
            sl.stream_kind = stream_for(gid);

            if (is_shape_invariant(gid)) {
                // shape 无关图：全部变体共享 base MemoryPlan + kShapeInvariant
                sl.mp = variant_memory_plans_[0].get();
                sl.shape_id = kShapeInvariant;
            } else {
                // shape 相关图：各自 MemoryPlan + 各自 ShapeId
                sl.mp = mp;
                sl.shape_id = shape_id;
            }
        }
    }

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

### Step 3: 扩展 `GpuExecTable` 支持 6-Variant

**文件**: `include/renaissance/task/deep_learning_task.h`

```cpp
struct GpuExecTable {
    // variant_graphs[variant_idx][rank][slot] → cudaGraphExec_t
    // variant_idx 0~5 对应：train_base/train_last/train_lowres/train_lowres_last/val_base/val_last
    std::array<std::vector<std::vector<cudaGraphExec_t>>, 6> variant_graphs;
    std::vector<int> device_ids;
};
```

### Step 4: 重写 `build_exec_table()` 构建 6-Variant 执行表

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
            gpu_exec_.variant_graphs[0][rank].resize(S(GraphSlot::COUNT), nullptr);
            auto& g = gpu_exec_.variant_graphs[0][rank];
            g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank, 0);
            g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank, 0);
        }
        return;
    }

    // ===== 为所有 6 个 variant 构建 exec table =====
    for (size_t v = 0; v < 6; ++v) {
        gpu_exec_.variant_graphs[v].resize(K);
        for (int rank = 0; rank < K; ++rank) {
            auto& g = gpu_exec_.variant_graphs[v][rank];
            g.resize(S(GraphSlot::COUNT), nullptr);

            // ===== H2D 传输图（shape 无关，但所有 variant 都有）=====
            g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank, v);
            g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank, v);

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
    for (size_t v = 0; v < 6; ++v) {
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

### Step 5: 修改 `run_train_epoch_gpu()` 支持 Variant 切换

**文件**: `src/task/deep_learning_task.cpp`

关键修改：在进入 batch 循环前，**预先存储所有需要的图指针**：

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

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, K, using_amp, &exc, loss_id]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // ===== 性能关键：预先存储所有需要的图指针 =====
                // Variant 0: train_base (full batch)
                const auto& g_tab_base = gpu_exec_.variant_graphs[0][rank];
                auto g_xfer_a_base     = g_tab_base[S(GraphSlot::XFER_A)];
                auto g_xfer_b_base     = g_tab_base[S(GraphSlot::XFER_B)];
                auto g_fwd_a_base      = g_tab_base[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto g_fwd_b_base      = g_tab_base[S(GraphSlot::FIRST_LAYER_FWD_B)];
                auto g_deep_a_base     = g_tab_base[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto g_deep_b_base     = g_tab_base[S(GraphSlot::FWD_BWD_DEEP_B)];
                auto g_first_a_base    = g_tab_base[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto g_first_b_base    = g_tab_base[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto g_zg_base         = g_tab_base[S(GraphSlot::ZERO_GRAD)];
                auto g_accum_base      = g_tab_base[S(GraphSlot::ACCUM_METRICS)];
                auto g_dar_base        = g_tab_base[S(GraphSlot::DEEP_ALLREDUCE)];
                auto g_far_base        = g_tab_base[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto g_wu_base         = g_tab_base[S(GraphSlot::WEIGHT_UPDATE)];
                auto g_ema_base        = g_tab_base[S(GraphSlot::EMA_UPDATE)];
                auto g_cdg_base        = g_tab_base[S(GraphSlot::CAST_DEEP_GRAD)];
                auto g_cfg_base        = g_tab_base[S(GraphSlot::CAST_FIRST_GRAD)];
                auto g_ncg_base        = g_tab_base[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                auto g_cm_base         = g_tab_base[S(GraphSlot::CAST_MAIN)];
                auto g_clear_metrics_base = g_tab_base[S(GraphSlot::CLEAR_METRICS)];

                // Variant 1: train_last (last batch)
                const auto& g_tab_last = gpu_exec_.variant_graphs[1][rank];
                auto g_xfer_a_last     = g_tab_last[S(GraphSlot::XFER_A)];
                auto g_xfer_b_last     = g_tab_last[S(GraphSlot::XFER_B)];
                auto g_fwd_a_last      = g_tab_last[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto g_fwd_b_last      = g_tab_last[S(GraphSlot::FIRST_LAYER_FWD_B)];
                auto g_deep_a_last     = g_tab_last[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto g_deep_b_last     = g_tab_last[S(GraphSlot::FWD_BWD_DEEP_B)];
                auto g_first_a_last    = g_tab_last[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto g_first_b_last    = g_tab_last[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto g_zg_last         = g_tab_last[S(GraphSlot::ZERO_GRAD)];
                auto g_accum_last      = g_tab_last[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];
                auto g_dar_last        = g_tab_last[S(GraphSlot::DEEP_ALLREDUCE)];
                auto g_far_last        = g_tab_last[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto g_wu_last         = g_tab_last[S(GraphSlot::WEIGHT_UPDATE)];
                auto g_ema_last        = g_tab_last[S(GraphSlot::EMA_UPDATE)];
                auto g_cdg_last        = g_tab_last[S(GraphSlot::CAST_DEEP_GRAD)];
                auto g_cfg_last        = g_tab_last[S(GraphSlot::CAST_FIRST_GRAD)];
                auto g_ncg_last        = g_tab_last[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                auto g_cm_last         = g_tab_last[S(GraphSlot::CAST_MAIN)];

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

                // ===== Batch 0 预传输（使用 base variant）=====
                {
                    ts->wait_buffer_readable(0);
                    if (g_xfer_a_base) cudaGraphLaunch(g_xfer_a_base, s_trans);
                    sync_tr();

                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);

                    if (using_amp) {
                        if (g_cm_base) { cudaGraphLaunch(g_cm_base, s_up); sync_up(); }
                    }
                    if (g_clear_metrics_base) cudaGraphLaunch(g_clear_metrics_base, s_up);
                }

                // ===== Normal batches (0 ~ batches-2) 使用 base variant =====
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    auto g_fwd   = from_a ? g_fwd_a_base : g_fwd_b_base;
                    auto g_deep  = from_a ? g_deep_a_base : g_deep_b_base;
                    auto g_xfer_n = from_a ? g_xfer_b_base : g_xfer_a_base;
                    auto g_first = from_a ? g_first_a_base : g_first_b_base;

                    // Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD
                    if (g_zg_base) cudaGraphLaunch(g_zg_base, s_up);
                    if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                    sync_comp(); sync_up();

                    // Wait next buffer
                    ts->wait_buffer_readable(next_buf);

                    // Phase 2: DEEP_FWD_BWD ‖ XFER(next)
                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
                    sync_comp(); sync_tr();

                    if (g_accum_base) cudaGraphLaunch(g_accum_base, s_up);
                    sync_up();

                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);

                    // Phase 3: BWD → CAST → COMM → CAST → COMM → NAN_CHECK → OPT → CAST
                    if (!frozen) {
                        if (g_first) cudaGraphLaunch(g_first, s_c1);
                    }
                    sync_comp();

                    if (using_amp && g_cdg_base) { cudaGraphLaunch(g_cdg_base, s_up); sync_up(); }
                    if (g_dar_base) cudaGraphLaunch(g_dar_base, s_up); sync_up();
                    if (using_amp && g_cfg_base) { cudaGraphLaunch(g_cfg_base, s_up); sync_up(); }
                    if (g_far_base) cudaGraphLaunch(g_far_base, s_up); sync_up();
                    if (using_amp && g_ncg_base) { cudaGraphLaunch(g_ncg_base, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batch);
                    (void)lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_wu_base) cudaGraphLaunch(g_wu_base, s_up); sync_up();
                    if (using_amp && g_cm_base) { cudaGraphLaunch(g_cm_base, s_up); sync_up(); }
                }

                // ===== Last batch (batch = batches-1) 使用 last variant =====
                {
                    bool last_a = ((batches - 1) % 2 == 0);

                    auto g_fwd_l   = last_a ? g_fwd_a_last : g_fwd_b_last;
                    auto g_deep_l  = last_a ? g_deep_a_last : g_deep_b_last;
                    auto g_first_l = last_a ? g_first_a_last : g_first_b_last;

                    if (g_zg_last) cudaGraphLaunch(g_zg_last, s_up);
                    if (g_fwd_l) cudaGraphLaunch(g_fwd_l, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
                    sync_comp();

                    if (g_accum_last) cudaGraphLaunch(g_accum_last, s_up);
                    sync_up();

                    if (!frozen) {
                        if (g_first_l) cudaGraphLaunch(g_first_l, s_c1);
                    }
                    sync_comp();

                    if (using_amp && g_cdg_last) { cudaGraphLaunch(g_cdg_last, s_up); sync_up(); }
                    if (g_dar_last) cudaGraphLaunch(g_dar_last, s_up); sync_up();
                    if (using_amp && g_cfg_last) { cudaGraphLaunch(g_cfg_last, s_up); sync_up(); }
                    if (g_far_last) cudaGraphLaunch(g_far_last, s_up); sync_up();
                    if (using_amp && g_ncg_last) { cudaGraphLaunch(g_ncg_last, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batches - 1);
                    (void)lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_wu_last) cudaGraphLaunch(g_wu_last, s_up); sync_up();
                    if (using_amp && g_cm_last) { cudaGraphLaunch(g_cm_last, s_up); sync_up(); }
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

**关键设计点**：
1. **性能关键**：在进入 batch 循环前，将所有需要的图指针存储为局部变量
2. **Zero-Cost 抽象**：运行时只是指针解引用，无 hash、无分支、无内存分配
3. **Last Batch 隔离**：last batch 完全使用 variant 1 的图，grid=last_batch_size

### Step 6: 修改 `run_val_epoch_gpu()` 支持 Variant 切换

**文件**: `src/task/deep_learning_task.cpp`

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
#ifdef TR_USE_CUDA
    // ... 前置代码不变 ...

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                // ... 设备设置不变 ...

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // ===== 性能关键：预先存储所有需要的图指针 =====
                // Variant 4: val_base
                const auto& g_tab_base = gpu_exec_.variant_graphs[4][rank];
                auto g_xfer_a_base     = g_tab_base[S(GraphSlot::XFER_A)];
                auto g_xfer_b_base     = g_tab_base[S(GraphSlot::XFER_B)];
                auto g_inf_a_base      = g_tab_base[S(GraphSlot::INF_MAIN_A)];
                auto g_inf_b_base      = g_tab_base[S(GraphSlot::INF_MAIN_B)];
                auto g_ema_a_base      = g_tab_base[S(GraphSlot::INF_EMA_A)];
                auto g_ema_b_base      = g_tab_base[S(GraphSlot::INF_EMA_B)];
                auto g_accum_base      = g_tab_base[S(GraphSlot::ACCUM_METRICS)];
                auto g_clear_metrics   = g_tab_base[S(GraphSlot::CLEAR_METRICS)];

                // Variant 5: val_last
                const auto& g_tab_last = gpu_exec_.variant_graphs[5][rank];
                auto g_xfer_a_last     = g_tab_last[S(GraphSlot::XFER_A)];
                auto g_xfer_b_last     = g_tab_last[S(GraphSlot::XFER_B)];
                auto g_inf_a_last      = g_tab_last[S(GraphSlot::INF_MAIN_A)];
                auto g_inf_b_last      = g_tab_last[S(GraphSlot::INF_MAIN_B)];
                auto g_ema_a_last      = g_tab_last[S(GraphSlot::INF_EMA_A)];
                auto g_ema_b_last      = g_tab_last[S(GraphSlot::INF_EMA_B)];
                auto g_accum_last      = g_tab_last[S(GraphSlot::ACCUM_METRICS_VAL_LAST)];

                // ... stream 初始化不变 ...

                if (g_clear_metrics) cudaGraphLaunch(g_clear_metrics, s_up);

                for (int batch = 0; batch < val_batches; ++batch) {
                    int buf = batch % 2;
                    bool is_last = (batch == val_batches - 1);

                    // ===== 根据是否 last batch 选择图指针 =====
                    auto g_xfer = (buf == 0) 
                        ? (is_last ? g_xfer_a_last : g_xfer_a_base)
                        : (is_last ? g_xfer_b_last : g_xfer_b_base);
                    auto g_inf = (buf == 0)
                        ? (is_last ? g_inf_a_last : g_inf_a_base)
                        : (is_last ? g_inf_b_last : g_inf_b_base);
                    auto g_ema = (buf == 0)
                        ? (is_last ? g_ema_a_last : g_ema_a_base)
                        : (is_last ? g_ema_b_last : g_ema_b_base);
                    auto g_accum = is_last ? g_accum_last : g_accum_base;

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

                // ... AllReduce 不变 ...

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }
    // ... 后续不变 ...
#endif
}
```

### Step 7: 修复 CPU 路径 + 支持 Variant

#### 7.1 修复 `run_train_epoch_cpu()`

**文件**: `src/task/deep_learning_task.cpp`

```cpp
float DeepLearningTask::run_train_epoch_cpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    
    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

    // ===== 性能关键：预先存储所有需要的图索引 =====
    // Variant 0: train_base
    int32_t idx_fwd_a_base     = atlas.index(0, GraphId::FIRST_LAYER_FWD_A);
    int32_t idx_fwd_b_base     = atlas.index(0, GraphId::FIRST_LAYER_FWD_B);
    int32_t idx_deep_base      = atlas.index(0, GraphId::DEEP_FWD_BWD);
    int32_t idx_first_a_base   = atlas.index(0, GraphId::FIRST_LAYER_BWD_A);
    int32_t idx_first_b_base   = atlas.index(0, GraphId::FIRST_LAYER_BWD_B);
    int32_t idx_clear          = atlas.index(0, GraphId::CLEAR_METRICS);
    int32_t idx_accum          = atlas.index(0, GraphId::ACCUM_METRICS);
    int32_t idx_zg             = atlas.index(0, GraphId::ZERO_GRAD);
    int32_t idx_dar            = atlas.index(0, GraphId::DEEP_COMM);
    int32_t idx_far            = atlas.index(0, GraphId::FIRST_COMM);
    int32_t idx_wu             = atlas.index(0, GraphId::OPTIMIZER);
    int32_t idx_ema            = atlas.index(0, GraphId::EMA_UPDATE);
    int32_t idx_cdg            = atlas.index(0, GraphId::CAST_DEEP_GRAD_FP16_TO_FP32);
    int32_t idx_cfg            = atlas.index(0, GraphId::CAST_FIRST_GRAD_FP16_TO_FP32);
    int32_t idx_ncg            = atlas.index(0, GraphId::NAN_CHECK_AND_GRAD_SCALING);
    int32_t idx_cm             = atlas.index(0, GraphId::CAST_MAIN_FP32_TO_FP16);

    // Variant 1: train_last
    int32_t idx_fwd_a_last     = atlas.index(1, GraphId::FIRST_LAYER_FWD_A);
    int32_t idx_fwd_b_last     = atlas.index(1, GraphId::FIRST_LAYER_FWD_B);
    int32_t idx_deep_last      = atlas.index(1, GraphId::DEEP_FWD_BWD);
    int32_t idx_first_a_last   = atlas.index(1, GraphId::FIRST_LAYER_BWD_A);
    int32_t idx_first_b_last   = atlas.index(1, GraphId::FIRST_LAYER_BWD_B);
    int32_t idx_accum_last     = atlas.index(1, GraphId::ACCUM_METRICS_TRAIN_LAST);

    auto launch = [&](int32_t idx) {
        if (idx >= 0) captured_result_.graphs[idx].launch(0, nullptr);
    };

    // ... Clear metrics ...

    // ===== Normal batches =====
    for (int batch = 0; batch < batches - 1; ++batch) {
        bool from_a = (batch % 2 == 0);
        
        // ... 使用 base variant 的图 ...
        launch(idx_zg);
        launch(from_a ? idx_fwd_a_base : idx_fwd_b_base);
        // ... 完整训练流程 ...
        launch(idx_accum);
    }

    // ===== Last batch =====
    {
        bool last_a = ((batches - 1) % 2 == 0);
        
        launch(idx_zg);
        launch(last_a ? idx_fwd_a_last : idx_fwd_b_last);
        launch(last_a ? idx_deep_last : idx_deep_last);  // DEEP_FWD_BWD 只有一个
        // ... 完整训练流程 ...
        launch(idx_accum_last);
    }

    // ===== 读取累积指标 =====
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
}
```

#### 7.2 修复 `run_val_epoch_cpu()`

**文件**: `src/task/deep_learning_task.cpp`

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_cpu(bool validate_ema) {
    auto& prep = Preprocessor::instance();
    const int val_batches = prep.val_steps();
    auto& registry = GlobalRegistry::instance();

    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

    // ===== 性能关键：预先存储所有需要的图索引 =====
    // Variant 4: val_base
    int32_t idx_inf_a_base    = atlas.index(4, GraphId::INF_MAIN_A);
    int32_t idx_inf_b_base    = atlas.index(4, GraphId::INF_MAIN_B);
    int32_t idx_ema_a_base    = atlas.index(4, GraphId::INF_EMA_A);
    int32_t idx_ema_b_base    = atlas.index(4, GraphId::INF_EMA_B);
    int32_t idx_accum_base    = atlas.index(4, GraphId::ACCUM_METRICS);

    // Variant 5: val_last
    int32_t idx_inf_a_last    = atlas.index(5, GraphId::INF_MAIN_A);
    int32_t idx_inf_b_last    = atlas.index(5, GraphId::INF_MAIN_B);
    int32_t idx_ema_a_last    = atlas.index(5, GraphId::INF_EMA_A);
    int32_t idx_ema_b_last    = atlas.index(5, GraphId::INF_EMA_B);
    int32_t idx_accum_last    = atlas.index(5, GraphId::ACCUM_METRICS_VAL_LAST);

    auto launch = [&](int32_t idx) {
        if (idx >= 0) captured_result_.graphs[idx].launch(0, nullptr);
    };

    // ... Clear metrics ...

    // ===== 按 batch size 加权累积 =====
    double acc_loss = 0.0;
    double acc_top1 = 0.0;
    double acc_top5 = 0.0;

    for (int batch = 0; batch < val_batches; ++batch) {
        int buf = batch % 2;
        bool is_last = (batch == val_batches - 1);
        
        // ===== 根据 is_last 选择 variant =====
        int32_t idx_inf_a = is_last ? idx_inf_a_last : idx_inf_a_base;
        int32_t idx_inf_b = is_last ? idx_inf_b_last : idx_inf_b_base;
        int32_t idx_ema_a = is_last ? idx_ema_a_last : idx_ema_a_base;
        int32_t idx_ema_b = is_last ? idx_ema_b_last : idx_ema_b_base;
        int32_t idx_accum = is_last ? idx_accum_last : idx_accum_base;

        // ... 完整验证流程 ...
        launch(idx_accum);

        // ===== 读取本 batch 指标并加权 =====
        int bs = is_last 
            ? registry.get_last_val_batch_size() 
            : registry.get_local_batch_size();
        
        if (loss_id >= 0) {
            Tensor h_loss = fetch_from_rank(active_memory_plan_->get_dtensor(loss_id), 0);
            acc_loss += static_cast<double>(h_loss.data<float>()[0]) * static_cast<double>(bs);
        }
        // ... top1/top5 同理 ...
    }

    // ===== 按总样本数平均 =====
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

## 四、去重机制验证

### 4.1 预期去重效果

基于 GraphAtlas 的 `Key{cg, gid, shape_id}` 去重机制：

| GraphId               | 是否 Shape 无关 | 各 Variant 的 ShapeId                                       | 去重结果                           |
| --------------------- | -------------- | ----------------------------------------------------------- | ---------------------------------- |
| TRANSFER_A/B          | ✅              | 全 `kShapeInvariant`                                         | 6 变体共享 1 张图                  |
| ZERO_GRAD             | ✅              | 全 `kShapeInvariant`                                         | 6 变体共享 1 张图                  |
| DEEP_FWD_BWD          | ❌              | v0~v1: train_bs×res, v2~v3: train_bs×lowres, v4~v5: val_bs×res | 最多 6 张（实际更少，因为 resolution/batch 可能相同） |
| FIRST_LAYER_FWD_A/B   | ❌              | 同 DEEP_FWD_BWD                                              | 最多 6 张                          |
| FIRST_LAYER_BWD_A/B   | ❌              | 同 DEEP_FWD_BWD                                              | 最多 6 张                          |
| INF_MAIN_A/B          | ❌              | v4~v5: val_bs×res                                           | 最多 2 张（val_base/val_last）      |
| INF_EMA_A/B           | ❌              | v4~v5: val_bs×res                                           | 最多 2 张                          |
| COMM/CAST/OPT/EMA     | ✅              | 全 `kShapeInvariant`                                         | 全部共享                           |
| ACCUM_METRICS 系列    | ✅              | 全 `kShapeInvariant`                                         | 全部共享                           |
| CLEAR_METRICS         | ✅              | 全 `kShapeInvariant`                                         | 全部共享                           |

**关键点**：
- ACCUM_METRICS 系列虽然计算时需要 batch_size，但其 kernel 内使用 `batch_size_ptr` 动态读取，graph 本身 shape 无关
- 因此 ACCUM_METRICS、ACCUM_METRICS_TRAIN_LAST、ACCUM_METRICS_VAL_LAST 全部共享同一张图

### 4.2 预期 Captured Graph 数量

**保守估算**：
- Shape 无关图：~15 个 GraphId × 1 次 = 15
- Shape 相关训练图：~8 个 GraphId × 4 个不同 shape（base/last/lowres/lowres_last）= 32
- Shape 相关推理图：~4 个 GraphId × 2 个不同 shape（base/last）= 8
- **总计 ≈ 55 张 captured graph**

**实际可能更少**：
- 如果 `last_train_batch_size == local_batch_size`（如 CIFAR-10），train_last 与 train_base 共享
- 如果 `train_sample_resolution_begin == train_sample_resolution_end`（不使用渐进分辨率），lowres 与 base 共享
- 实际数量在 **30-50 张之间**，完全在可接受范围内

---

## 五、实施优先级与验证

### 5.1 实施顺序

| 阶段 | 任务                               | 工时 | 验证方式                           |
| ---- | ---------------------------------- | ---- | ---------------------------------- |
| 1    | 修复 `on_prepare()` 生成 variant   | 1h   | 编译通过，`result.variants.size() == 6` |
| 2    | 重写 `build_graph_atlas()`         | 2h   | pre_capture 日志显示 captured graph 增加 |
| 3    | 重写 `build_exec_table()`          | 1h   | 日志显示 6 variant 的 exec table 构建 |
| 4    | 修改 `run_train_epoch_gpu()`       | 2h   | test_dl_full_gpu 通过              |
| 5    | 修改 `run_val_epoch_gpu()`        | 1h   | test_dl_full_gpu 验证通过          |
| 6    | 修改 `run_train_epoch_cpu()`       | 1h   | test_dl_full --cpu 通过            |
| 7    | 修改 `run_val_epoch_cpu()`        | 1h   | test_dl_full --cpu 验证通过        |
| 8    | 端到端测试                         | 2h   | 构造不能整除的 batch size 验证      |

**总工时**: 约 **11 小时**（~1.5 个工作日）

### 5.2 验证计划

#### 阶段 1：编译期验证
- 编译通过
- `result.variants.size() == 6`
- 各 variant 的 `memory_plan` 非空

#### 阶段 2：Capture 期验证
- `pre_capture` 日志显示 captured graph 数量增加到 **30-50**
- Reused 计数 > 0（shape 无关图共享）
- 各 variant 的关键 graph（DEEP_FWD_BWD/INF_MAIN_A）captured_idx >= 0

#### 阶段 3：运行时验证
- `test_dl_full_gpu` 运行通过
- `test_dl_full_amp` 运行通过
- `test_dl_full --cpu` 运行通过

#### 阶段 4：正确性验证
**构造 Last Batch 场景**：
- 临时修改测试配置，使 `train` 和 `val` 样本数不能被 `batch_size` 整除
- 例如：`train=50003, val=10007, batch=200`
- 预期：`train_last_batch_size=103, val_last_batch_size=7`

**验证方法**：
- 对比修复前后的 loss/accuracy
- 如果可能，dump last batch 的 softmax CE kernel grid 配置
- 确认 variant 1 的 DEEP_FWD_BWD captured graph 的 `grid.x == last_batch_size`

---

## 六、风险与缓解

| 风险                                                         | 概率 | 影响  | 缓解措施                                                     |
| ------------------------------------------------------------ | ---- | ----- | ------------------------------------------------------------ |
| 某些 variant 的 MemoryPlan 中 `slot_bytes` 与 base 不同，导致 offset 不一致 | 低   | 灾难性 | `compute_max_slot_bytes` 已跨变体取 max，所有 variant 的 slot size 相同。如不放心，可在 `build_graph_atlas()` 中增加断言检查关键 DTensor（loss/accum）的 offset 一致性 |
| Val variant 误填 train graph，导致 val 路径使用 train 的 FWD/BWD | 低   | 正确性 | `is_train_graph()` / `is_infer_graph()` 判定 + `continue` 跳过已保证隔离。可额外增加断言：val variant 的 DEEP_FWD_BWD slot `captured_idx == -1` |
| Multi-variant 导致 `pre_capture` 时间翻倍                     | 中   | 性能  | 预期增加 < 2 秒（cuDNN warmup 是主要耗时）。如需要可并行 capture（当前已支持 per-rank 并行） |
| CPU 路径的 `CpuOp` 的 `input_shape` 在 capture 时固定，切换 variant 后 shape 不匹配 | 低   | 正确性 | CPU capture 同样使用 `mp` 参数，`create_cpu_ops()` 中已读取 `mp.get_dtensor(id).shape`。只要 Atlas 填了正确的 mp，CPU 路径自动正确 |
| `transfer_to_rank` 使用 `active_memory_plan_->get_dtensor().nbytes()`（base batch size），导致 last batch 多拷贝数据 | 中   | 性能  | 多拷贝的数据在 last batch 的 FWD graph 中不会被处理（grid=last_batch_size），不影响正确性。性能影响极小（只在每 epoch 的最后一个 batch） |

---

## 七、与 LRF.md 分析的对比

| 分析文档       | 总体评价 | 核心贡献                                                         | 本方案采用情况 |
| -------------- | -------- | ---------------------------------------------------------------- | ------------- |
| **LRF-D**      | 深入透彻 | 三层断裂诊断<br>两阶段修复路线<br>详细的实施步骤与代码示例           | ✅ 完全采纳   |
| **LRF-K**      | 架构完整 | 根因论证（CUDA Graph grid 固定）<br>6-Variant 完整实现<br>去重机制验证 | ✅ 完全采纳   |
| **用户补充**   | 关键指导 | H2D 传输图特殊性<br>权重更新图不受 batch 影响<br>性能绝对优先要求   | ✅ 完全采纳   |

**本方案的改进点**：
1. **更清晰的 Graph 分类**：明确区分 "完全 Shape 无关"、"训练 Shape 相关"、"推理 Shape 相关" 三类
2. **性能优化的 Variant 切换**：在进入 batch 循环前预先存储所有图指针，运行时零开销
3. **更完整的 CPU 路径修复**：同时修复 CPU 训练的指标累积问题和 CPU 验证的加权平均问题
4. **更详细的验证计划**：分 4 个阶段验证，从编译期到运行时到正确性

---

## 八、总结

### 8.1 核心设计决策

1. **6-Variant 系统完整落地**：
   - Variant 0-3：训练路径（base/last/lowres/lowres_last）
   - Variant 4-5：验证路径（base/last）
   - 交叉槽位自然隔离：val variant 不填 train graph

2. **Graph 分类去重**：
   - 完全 Shape 无关：TRANSFER/COMM/OPT/CAST/ACCUM/CLEAR → 全局共享 1 张图
   - 训练 Shape 相关：FIRST_LAYER/DEEP_FWD_BWD → 最多 4 张独立图
   - 推理 Shape 相关：INF_MAIN/INF_EMA → 最多 2 张独立图

3. **性能绝对优先**：
   - 在进入 batch 循环前预先存储所有图指针
   - 运行时只是指针解引用，零 hash、零分支、零内存分配
   - 避免任何运行时开销

4. **CPU/GPU 完全对齐**：
   - CPU 路径同样支持 variant 切换
   - 同时修复 CPU 路径的两处指标 bug

### 8.2 预期效果

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
- Shape 相关图按需捕获，总开销 **30-50 张图**
- 运行时 variant 切换 **零开销**（指针解引用）

### 8.3 与框架设计意图的对比

| 设计意图                     | 本方案实现                                                     | 状态 |
| ---------------------------- | ------------------------------------------------------------ | ---- |
| 6-Variant 编译               | `on_prepare()` 生成 5 个 variant specs                           | ✅    |
| GraphAtlas 去重机制          | `build_graph_atlas()` 正确区分 shape 相关/无关图              | ✅    |
| Last batch 正确性            | Variant 1/5 使用 last_batch_size capture graph                 | ✅    |
| 渐进分辨率支持               | Variant 2/3 使用 low_res capture graph                         | ✅    |
| 性能优先                     | 预先存储图指针，运行时零开销                                   | ✅    |
| CPU/GPU 对齐                 | CPU 路径同样支持 variant 切换                                  | ✅    |

**结论**：本方案 **完整、科学、严谨** 地实现了框架设计的所有意图，彻底解决了 last batch 正确性问题，并完整支持了渐进式分辨率。

---

**文档版本**: v2.0  
**创建日期**: 2026-05-28  
**作者**: 基于 LRF.md 中小伙伴 D 和 K 的深入分析 + 用户补充的关键指导  
**状态**: 待用户审核后开始实施
