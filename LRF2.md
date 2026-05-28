# Last Batch 与渐进式分辨率 —— 完整、科学、严谨的变体捕获与运行时切换方案

> 基于 LRF.md 中用户补充的核心设计原则，结合源代码实地验证，提出一套零妥协、高性能、完全支持 6-Variant GraphAtlas 的方案。

---

## 一、设计原则（来自用户补充，最高优先级）

1. **H2D 传输图始终只有 2 种**：A 区图和 B 区图。与训练/验证、完整/不完整 batch、渐进式分辨率均无关。AMP 开启与否会影响 H2D 图。
2. **权重更新/梯度同步不受 batch size 影响**：这些图在验证阶段不存在。受 batch size 影响的主要是首层正反向、深层正反向。
3. **不同 MemoryPlan 中同一 ID 的 DTensor 总是拥有相同的 slot bytes**（`compute_max_slot_bytes` 跨变体取 max），但具体 shape 不同。
4. **ShapeId 是去重键**：`Key{cg, gid, shape_id}`。相同的 shape_id → 相同的 CapturedGraph。
5. **GPU 为准，CPU 对齐**：如果 GPU 为某种情形单独捕获图，CPU 也应单独捕获。
6. **性能至上**：进入 batch 循环之前，用局部变量存储好这个 epoch 需要用到的所有图（包括 last batch）。循环内部零函数调用、零哈希、零分支（除了 A/B 双缓冲的 trivial 条件）。
7. **多 RANK 全程支持**：每个 rank 的线程在启动时独立预解析自己的 graph handles。

---

## 二、问题根因

### 2.1 三层断裂

| 层级 | 设计意图 | 当前实现 | 断裂点 |
|---|---|---|---|
| **Compiler** | `on_prepare()` 生成 base_spec + 5 个 variant_specs，`compile()` 产出 6 个 Variant | `on_prepare()` **只传 1 个 base_spec**，`variant_specs` 为空 → `result.variants.size() == 1` | 未生成 last batch / lowres / val 的 MemoryPlan |
| **GraphAtlas** | 6×28 映射表，shape 相关图各自 `mp` + `shape_id`，shape 无关图共享 `kShapeInvariant` | `build_graph_atlas()` **只填 variant 0**，所有 slot 硬编码 `shape_id = kShapeInvariant`、`mp = active_memory_plan_` | pre_capture 去重时所有变体 Key 相同 → 只捕获一次 |
| **运行时** | 根据 batch 类型（normal/last）和分辨率（base/lowres）选择 correct variant | `build_exec_table()` 只解析 `atlas.index(0, gid)`；`run_train_epoch_gpu()` 对所有 batch 使用**同一组局部变量** | last batch 使用 grid=base_batch 的 CUDA Graph 处理 last_batch_size 个样本 |

### 2.2 为什么必须切换 Variant

`Compiler::create_memory_plans()` 中 `max_slots` 保证了**跨变体 offset 一致性**，但 `DTensor::shape` 在各变体中不同：

- variant[0] train_base: `logits_dt.shape = {512, 32, 32, 64}` → capture 时 grid=512
- variant[1] train_last: `logits_dt.shape = {402, 32, 32, 64}` → capture 时 grid=402

`softmax_ce_op.cpp` 中：`int batch = logits_dt.shape.n();` 在 **capture 时**读取并固定进 CUDA Graph。运行时即使修改 `batch_size_ptr` 标量，`grid`（即 `batch`）仍然是 capture 时的值。

**唯一正确做法**：为 variant[1] 单独 capture，capture 前 `ctx.set_memory_plan(mp_v1)`，使 grid=402。

`pre_capture()` 的 `capture_all_for_rank()` 已内置此机制（`contexts[rank]->set_memory_plan(mp)`），问题只在于上游 `build_graph_atlas()` 未提供正确的 `mp` 和 `shape_id`。

---

## 三、6-Variant 系统精确定义

| Variant | 索引 | 名称 | batch_size | resolution | 适用阶段 | ComputationGraph |
|---|---|---|---|---|---|---|
| 0 | `train_base` | `local_batch_size` | `train_res_begin` | 训练（常规 batch） | `train_cg` |
| 1 | `train_last` | `last_train_batch_size` | `train_res_begin` | 训练（last batch） | `train_cg` |
| 2 | `train_lowres` | `local_batch_size` | `train_res_end` | 训练（低分辨率常规 batch） | `train_cg` |
| 3 | `train_lowres_last` | `last_train_batch_size` | `train_res_end` | 训练（低分辨率 last batch） | `train_cg` |
| 4 | `val_base` | `local_batch_size` | `val_res` | 验证（常规 batch） | `infer_cg` |
| 5 | `val_last` | `last_val_batch_size` | `val_res` | 验证（last batch） | `infer_cg` |

### 3.1 各 GraphId 的 Shape 属性

| GraphId | 是否 Shape 相关 | 各 Variant 的 ShapeId | 去重结果 |
|---|---|---|---|
| `TRANSFER_A/B` | ❌ 无关 | 全 `kShapeInvariant` | 6 变体共享 1 张图 |
| `ZERO_GRAD` | ❌ 无关 | 全 `kShapeInvariant` | 6 变体共享 1 张图 |
| `DEEP_FWD_BWD` | ✅ 相关 | v0:{512,32,32,3}, v1:{402,32,32,3}, v2:{512,16,16,3}, v3:{402,16,16,3}, v4:{512,32,32,3}, v5:{106,32,32,3} | 6 张独立图（6 个不同 ShapeId） |
| `FIRST_LAYER_FWD_A/B` | ✅ 相关 | 同上 | 各 6 张独立图 |
| `FIRST_LAYER_BWD_A/B` | ✅ 相关 | 同上 | 各 6 张独立图 |
| `INF_MAIN_A/B` | ✅ 相关 | v4/v5 使用 | 各 2 张（val_base/last） |
| `INF_EMA_A/B` | ✅ 相关 | v4/v5 使用 | 各 2 张 |
| `COMM` (FIRST/DEEP/STATS) | ❌ 无关 | 全 `kShapeInvariant` | 全部共享 |
| `CAST_*` | ❌ 无关 | 全 `kShapeInvariant` | 全部共享 |
| `OPTIMIZER` | ❌ 无关 | 全 `kShapeInvariant` | 全部共享 |
| `EMA_UPDATE` | ❌ 无关 | 全 `kShapeInvariant` | 全部共享 |
| `ACCUM_METRICS*` | ❌ 无关 | 全 `kShapeInvariant` | 全部共享（但 GraphId 不同，不互相共享） |
| `VAL_RESULT_COMM` | ❌ 无关 | 全 `kShapeInvariant` | 全部共享 |
| `CLEAR_METRICS` | ❌ 无关 | 全 `kShapeInvariant` | 全部共享 |

### 3.2 去重后 Captured Graph 数量估算

- Shape 无关图：~15 个 GraphId × 1 次 = 15
- Shape 相关图：~10 个 GraphId，按实际出现的不同 ShapeId 计数
  - DEEP_FWD_BWD: 6 个（6 个 variant 各不相同）
  - FIRST_LAYER_FWD_A: 6 个
  - FIRST_LAYER_FWD_B: 6 个
  - FIRST_LAYER_BWD_A: 6 个
  - FIRST_LAYER_BWD_B: 6 个
  - INF_MAIN_A: 2 个（v4, v5）
  - INF_MAIN_B: 2 个
  - INF_EMA_A: 2 个
  - INF_EMA_B: 2 个
- **总计 ≈ 53 张 captured graph**
- 相比当前（~20 张）增加约 2.6 倍，仍在可接受范围内。cuDNN warmup 是主要耗时，与 graph 数量近似线性，系数很小。

---

## 四、修改文件清单

| 文件 | 修改内容 | 行数估算 |
|---|---|---|
| `include/renaissance/task/deep_learning_task.h` | ① 新增 `variant_memory_plans_` / `variant_shape_ids_` / `compiled_result_` 成员<br>② 扩展 `GpuExecTable` 支持 6 variant<br>③ `on_prepare()` 生成 variant specs | ~50 |
| `src/task/deep_learning_task.cpp` | ① 重写 `build_graph_atlas()`<br>② 重写 `build_exec_table()`<br>③ 重写 `run_train_epoch_gpu()`（循环外预解析）<br>④ 重写 `run_val_epoch_gpu()`<br>⑤ 重写 `run_train_epoch_cpu()`<br>⑥ 重写 `run_val_epoch_cpu()` | ~350 |

> **不需要修改的文件**：`src/graph/compiler.cpp`、`src/graph/captured_graph.cpp`、`src/backend/ops/*` —— 这些基础设施已完全支持多 variant，只需上层正确调用。

---

## 五、详细实现

### 5.1 `DeepLearningTask` 新增成员（`deep_learning_task.h`）

在 `private:` 区域新增：

```cpp
// ===== 6-Variant 系统 =====
/// 各变体的独立 MemoryPlan（slot_bytes 相同，shape 不同）
std::array<std::unique_ptr<MemoryPlan>, GraphAtlas::kMaxVariants> variant_memory_plans_;

/// 各变体的 ShapeId（用于 Atlas 填表和 pre_capture 去重）
std::array<ShapeId, GraphAtlas::kMaxVariants> variant_shape_ids_;

/// 保存 Compiler::Result 以确保 ComputationGraph 生命周期（train_cg_ / infer_cg_ 指向此处）
Compiler::Result compiled_result_;
```

### 5.2 `GpuExecTable` 扩展（`deep_learning_task.h`）

```cpp
#ifdef TR_USE_CUDA
struct GpuExecTable {
    // variant_graphs[variant_idx][rank][slot] → cudaGraphExec_t
    // variant_idx 0~5 对应：train_base/train_last/train_lowres/train_lowres_last/val_base/val_last
    std::array<std::vector<std::vector<cudaGraphExec_t>>, GraphAtlas::kMaxVariants> variant_graphs;
    std::vector<int> device_ids;
};
GpuExecTable gpu_exec_;
std::vector<float*> lr_pinned_;
#endif
```

### 5.3 `on_prepare()` 修改（`deep_learning_task.h`）

将当前仅传 `base_spec` 的代码：

```cpp
CompileSpec spec = CompileSpec::from_global_registry();
auto result = Compiler::compile(plan, spec, plan_config_, initializer_);
memory_plan_ptr_ = std::move(result.variants[0].memory_plan);
```

替换为：

```cpp
CompileSpec base_spec = CompileSpec::from_global_registry();
auto& gr = GlobalRegistry::instance();

// ===== 生成 5 个 variant specs =====
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
    s.actual_resolution = gr.get_train_sample_resolution_end();
    variant_specs.push_back(s);
}
// v3: train_lowres_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_train_batch_size();
    s.actual_resolution = gr.get_train_sample_resolution_end();
    variant_specs.push_back(s);
}
// v4: val_base
{
    CompileSpec s = base_spec;
    s.actual_resolution = gr.val_sample_resolution();
    variant_specs.push_back(s);
}
// v5: val_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_val_batch_size();
    s.actual_resolution = gr.val_sample_resolution();
    variant_specs.push_back(s);
}

auto result = Compiler::compile(plan, base_spec, plan_config_, initializer_, variant_specs);
compiled_result_ = std::move(result);  // 保存所有权

// ===== 保存所有 variant 的 MemoryPlan 和 ShapeId =====
for (size_t i = 0; i < compiled_result_.variants.size() && i < GraphAtlas::kMaxVariants; ++i) {
    variant_memory_plans_[i] = std::move(compiled_result_.variants[i].memory_plan);
    // ShapeId 从 CompileSpec 推导
    if (i == 0) {
        variant_shape_ids_[i] = base_spec.get_shape_id();
    } else if (i - 1 < variant_specs.size()) {
        variant_shape_ids_[i] = variant_specs[i - 1].get_shape_id();
    }
}

memory_plan_ptr_ = std::move(variant_memory_plans_[0]);  // base 作为默认
active_memory_plan_ = memory_plan_ptr_.get();

// ComputationGraph 指针从 compiled_result_ 获取（生命周期由 compiled_result_ 保证）
train_cg_ = &compiled_result_.train_cg;
infer_cg_ = &compiled_result_.infer_cg;
```

> **关键说明**：`compiled_result_` 是新增的成员变量，持有 `train_cg` 和 `infer_cg` 的所有权。`train_cg_` / `infer_cg_` 指向 `compiled_result_` 内部，不再依赖 `named_graphs_` 的生命周期。
> 
> `add_graph("train", std::move(result.train_cg), ...)` 这行需要移除或修改，因为 `result.train_cg` 已被 `compiled_result_` 持有。实际上当前代码中 `add_graph` 将 `result.train_cg` move 进 `named_graphs_`，然后 `train_cg_ = &named_graphs_["train"].graph`。这种方式也可以保留，只需确保 `named_graphs_` 的生命周期覆盖整个训练过程即可。
> 
> 为最小化改动，可以保留 `add_graph()` 调用，但 `compiled_result_` 仍然需要保存以保留 `variants` 数组（内含 MemoryPlan 所有权）。或者将 `add_graph()` 改为不 move，而是直接引用 `compiled_result_.train_cg`。
> 
> **推荐**：保留 `add_graph()` 不变（`named_graphs_` 持有 ComputationGraph），`compiled_result_` 只用于保存 `variants`（MemoryPlan 所有权）。`train_cg_` 仍指向 `named_graphs_` 中的图。

### 5.4 `build_graph_atlas()` 重写（`deep_learning_task.cpp`）

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

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

    for (size_t v = 0; v < GraphAtlas::kMaxVariants; ++v) {
        const MemoryPlan* mp = variant_memory_plans_[v].get();
        if (!mp) continue;

        bool is_train_var = (v <= 3);
        bool is_val_var   = (v >= 4);

        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);

            // ===== 1. 选择 ComputationGraph =====
            bool gid_is_train = is_train_graph(gid);
            bool gid_is_infer = is_inference_graph(gid);
            bool gid_is_shape_inv = is_shape_invariant_graph(gid);

            const ComputationGraph* cg = nullptr;

            if (is_train_var) {
                // Train variant: 需要 train graph（含 transfer）和 shape invariant 图
                // 纯推理图（INF_*）在 train variant 中不需要
                if (gid_is_infer && !gid_is_shape_inv) continue;
                cg = train_cg_;
            } else { // val variant
                // Val variant: 需要推理图（INF_*）、transfer 图、shape invariant 图
                // Train-only shape 相关图（FWD/BWD/DEEP）不需要
                if (gid_is_train && !gid_is_infer && !gid_is_shape_inv) continue;
                if (gid_is_infer || gid_is_shape_inv) {
                    cg = gid_is_infer ? infer_cg_ : train_cg_;
                }
            }

            if (!cg || cg->nodes(gid).empty()) continue;

            // ===== 2. 填 Slot =====
            auto& sl = atlas.slot(v, gi);
            sl.cg = cg;
            sl.stream_kind = stream_for(gid);

            if (gid_is_shape_inv) {
                // Shape 无关图：全部变体共享 variant[0] 的 MemoryPlan + kShapeInvariant
                sl.mp = variant_memory_plans_[0].get();
                sl.shape_id = kShapeInvariant;
            } else {
                // Shape 相关图：各自 MemoryPlan + 各自 ShapeId
                sl.mp = mp;
                sl.shape_id = variant_shape_ids_[v];
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

> **去重验证**：Shape 无关图（如 ZERO_GRAD）在所有 variant 中 `cg` 相同、`shape_id = kShapeInvariant`，所以 `Key{cg, gid, shape_id}` 完全相同 → `pre_capture` 自动去重，全局只捕获一次。
> 
> Shape 相关图（如 DEEP_FWD_BWD）在不同 variant 中 `shape_id` 不同（batch_size 不同）→ Key 不同 → 分别捕获。

### 5.5 `build_exec_table()` 重写（`deep_learning_task.cpp`）

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

    // 为所有 6 个 variant 预 build exec table
    for (size_t v = 0; v < GraphAtlas::kMaxVariants; ++v) {
        gpu_exec_.variant_graphs[v].resize(K);
        for (int rank = 0; rank < K; ++rank) {
            auto& g = gpu_exec_.variant_graphs[v][rank];
            g.resize(S(GraphSlot::COUNT), nullptr);

            g[S(GraphSlot::XFER_A)]           = resolve(GraphId::TRANSFER_A, rank, v);
            g[S(GraphSlot::XFER_B)]           = resolve(GraphId::TRANSFER_B, rank, v);
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
            g[S(GraphSlot::INF_MAIN_A)]       = resolve(GraphId::INF_MAIN_A, rank, v);
            g[S(GraphSlot::INF_MAIN_B)]       = resolve(GraphId::INF_MAIN_B, rank, v);
            g[S(GraphSlot::INF_EMA_A)]        = resolve(GraphId::INF_EMA_A, rank, v);
            g[S(GraphSlot::INF_EMA_B)]        = resolve(GraphId::INF_EMA_B, rank, v);
            g[S(GraphSlot::ACCUM_METRICS)]           = resolve(GraphId::ACCUM_METRICS, rank, v);
            g[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)] = resolve(GraphId::ACCUM_METRICS_TRAIN_LAST, rank, v);
            g[S(GraphSlot::ACCUM_METRICS_VAL_LAST)]   = resolve(GraphId::ACCUM_METRICS_VAL_LAST, rank, v);
            g[S(GraphSlot::VAL_RESULT_ALLREDUCE)]     = resolve(GraphId::VAL_RESULT_COMM, rank, v);
            g[S(GraphSlot::CLEAR_METRICS)]            = resolve(GraphId::CLEAR_METRICS, rank, v);
        }
    }

    // 日志：每个 variant 的关键 slot
    for (size_t v = 0; v < GraphAtlas::kMaxVariants; ++v) {
        for (int rank = 0; rank < K; ++rank) {
            const auto& g = gpu_exec_.variant_graphs[v][rank];
            LOG_INFO << "[EXEC-TABLE] v=" << v << " rank=" << rank
                     << " DEEP=" << (g[S(GraphSlot::FWD_BWD_DEEP_A)] ? "OK" : "NULL")
                     << " FWD_A=" << (g[S(GraphSlot::FIRST_LAYER_FWD_A)] ? "OK" : "NULL")
                     << " INF_A=" << (g[S(GraphSlot::INF_MAIN_A)] ? "OK" : "NULL");
        }
    }

    // 校验：variant 0 的关键 slot 必须非空
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

### 5.6 `run_train_epoch_gpu()` 重写（`deep_learning_task.cpp`）

**核心设计**：在线程 lambda 的最开始（batch 循环外部），根据当前 epoch 的分辨率选择 variant，预解析所有 graph handles 为局部变量。循环内部只使用局部变量，零 variant 切换开销。

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
    bool using_lowres = (get_current_train_resolution() !=
                         registry.train_sample_resolution_begin());
    size_t v_base = using_lowres ? 2 : 0;  // train_lowres : train_base
    size_t v_last = using_lowres ? 3 : 1;  // train_lowres_last : train_last

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, K, using_amp, &exc,
                              loss_id, v_base, v_last]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // =====================================================================
                // CRITICAL: PRE-RESOLVE ALL GRAPH HANDLES BEFORE BATCH LOOP
                // 根据用户要求：进入 batch 循环前存储好所有图（含 last batch）
                // =====================================================================

                const auto& g_nb = gpu_exec_.variant_graphs[v_base][rank];
                const auto& g_lb = gpu_exec_.variant_graphs[v_last][rank];

                // --- Normal batch handles (variant v_base) ---
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
                auto nb_clear   = g_nb[S(GraphSlot::CLEAR_METRICS)];

                // --- Last batch handles (variant v_last) ---
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

                    // Wait next buffer
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
}
#endif
```

> **性能保证**：所有 `cudaGraphExec_t` handles 在 lambda 开始处（batch 循环外部）预解析为局部变量。Normal batch 循环和 last batch 代码路径各自使用独立的局部变量组。循环内部**零 variant 切换开销**、零数组索引计算（除了 A/B 双缓冲的 trivial 条件）、零函数调用。

### 5.7 `run_val_epoch_gpu()` 重写（`deep_learning_task.cpp`）

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
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

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

                auto g_val_comm = (val_batches > 0 && val_batches - 1 == val_batches - 1)
                    ? vl_val_comm : vb_val_comm;
                // 实际上 val_comm 是 shape invariant，vb 和 vl 指向同一个 captured graph
                // 可以直接用 vb_val_comm
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

### 5.8 `run_train_epoch_cpu()` 重写（`deep_learning_task.cpp`）

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
    bool using_lowres = (get_current_train_resolution() !=
                         registry.train_sample_resolution_begin());
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
    // Shape-invariant indices are the same (dedup), but we keep separate vars for symmetry
    int32_t idx_zg_lb     = idx_for(GraphId::ZERO_GRAD, v_last);     // same as nb
    int32_t idx_far_lb    = idx_for(GraphId::FIRST_COMM, v_last);    // same as nb
    int32_t idx_dar_lb    = idx_for(GraphId::DEEP_COMM, v_last);     // same as nb
    int32_t idx_opt_lb    = idx_for(GraphId::OPTIMIZER, v_last);     // same as nb

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

        launch(idx_zg_lb);
        launch(last_a ? idx_fwd_a_lb : idx_fwd_b_lb);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_lb);

        if (!frozen) launch(last_a ? idx_bwd_a_lb : idx_bwd_b_lb);
        launch(idx_dar_lb);

        {
            float lr = fetch_lr_for_batch(batches - 1);
            *static_cast<float*>(lr_ptr) = lr;
        }
        launch(idx_far_lb);
        launch(idx_opt_lb);
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

### 5.9 `run_val_epoch_cpu()` 重写（`deep_learning_task.cpp`）

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

        // 按样本数加权累积（修复 B5）
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

    // 按总样本数平均（修复 B5）
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

## 六、渐进式分辨率集成

本方案天然支持渐进式分辨率，无需额外修改运行时逻辑：

```cpp
// run_train_epoch_gpu() 开头
bool using_lowres = (get_current_train_resolution() !=
                     registry.train_sample_resolution_begin());
size_t v_base = using_lowres ? 2 : 0;  // train_lowres : train_base
size_t v_last = using_lowres ? 3 : 1;  // train_lowres_last : train_last
```

| Epoch | Resolution | v_base | v_last |
|---|---|---|---|
| 早期（< boundary_epoch） | `train_res_end`（低分辨率） | 2 | 3 |
| 后期（≥ boundary_epoch） | `train_res_begin`（高分辨率） | 0 | 1 |

验证阶段分辨率固定为 `val_res`，始终使用 v4（val_base）和 v5（val_last）。

---

## 七、性能保证

| 指标 | 当前实现 | 本方案 | 说明 |
|---|---|---|---|
| Batch 循环内 variant 切换开销 | N/A（无 variant） | **0** | 所有 handles 在循环外预解析为局部变量 |
| Graph handle 解析 | `g_tab[S(slot)]` 数组索引 | **直接局部变量** | 编译器可能将局部变量放入寄存器 |
| Last batch 条件分支 | 无（当前错误地用同一 graph） | **编译期确定的分支**（分开的代码块） | last batch 使用独立的局部变量组 |
| 多 RANK 扩展性 | `gpu_exec_.graphs[rank]` | **`gpu_exec_.variant_graphs[v][rank]`** | 每个 rank 独立预解析自己的 handles |
| 内存占用 | ~20 张 captured graph | **~53 张** | shape 无关图零膨胀，shape 相关图按实际不同 ShapeId 计数 |
| 编译时间（cuDNN warmup） | ~1s | **~2s** | 与 unique graph 数量近似线性 |

---

## 八、验证计划

### Phase 1：编译与基础功能（1 小时）

1. 修改 `on_prepare()` 生成 variant specs
2. 修改 `build_graph_atlas()` 全量填表
3. 编译通过
4. **验证**：`pre_capture` 日志显示 `captured` 计数 ≈ 53，`reused` 计数 > 0（shape 无关图共享）

### Phase 2：运行时 GPU（2 小时）

1. 修改 `build_exec_table()`
2. 修改 `run_train_epoch_gpu()` 和 `run_val_epoch_gpu()`
3. 运行 `test_dl_full_gpu` 和 `test_dl_full_amp`
4. **验证**：CIFAR-10（整除 batch）无回归；构造不可整除 batch（如 batch=192）验证 last batch 行为

### Phase 3：运行时 CPU（1 小时）

1. 修改 `run_train_epoch_cpu()` 和 `run_val_epoch_cpu()`
2. 运行 `test_dl_full`（CPU 模式）
3. **验证**：CPU 路径指标与 GPU 路径对齐

### Phase 4：渐进式分辨率（1 小时）

1. 启用 `progressive_resize()` 配置
2. 验证早期 epoch 使用 v2/v3（低分辨率），后期使用 v0/v1（高分辨率）
3. **验证**：`pre_capture` 日志中不同 epoch 的 captured graph 数量一致（因为所有 variant 在 compile 时已全部捕获）

### Phase 5：端到端正确性（2 小时）

1. ImageNet 配置模拟（batch=512, world_size=8）
2. 验证 `last_train_batch_size = 402` 时，variant 1 的 DEEP_FWD_BWD grid=402
3. 对比修复前后的 loss/accuracy 差异

---

## 九、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| `build_graph_atlas()` 中 val 变体误填了 train-only shape 相关图 | 低 | 正确性错误 | `is_train_graph() && !is_shape_invariant_graph()` 在 val variant 中显式 `continue` 跳过 |
| `variant_memory_plans_` 中某些 variant 的 slot_bytes 与 base 不一致 | 极低 | 灾难性（指针错乱） | `compute_max_slot_bytes` 已跨变体取 max；可额外增加断言检查关键 DTensor offset |
| 多 variant 导致 captured graph 内存增加 | 中 | 显存增加 | shape 无关图共享（零膨胀）；shape 相关图只增加特征图相关 buffer，权重 buffer 不增加 |
| `transfer_to_rank` 使用 `active_memory_plan_`（base）的 `dt.nbytes()`，导致 last batch 多拷贝 | 中 | 性能损失 | 多拷贝数据在 last batch 的 FWD graph 中不会被处理（grid=last_batch_size），不影响正确性。后续可优化 `transfer_to_rank` 接受 actual_batch_size |
| CPU 路径的 `graphs[idx].launch()` 中 `CpuOp` 的 shape 在 capture 时固定 | 低 | CPU last batch 仍错误 | CPU capture 同样使用 `mp` 参数，`create_cpu_ops()` 读取 `mp.get_dtensor(id).shape`。Atlas 填了正确的 mp 后自动正确 |

---

## 十、总结

本方案**彻底根除 last batch 正确性 bug**，核心设计决策：

1. **Compiler 层**：`on_prepare()` 生成 5 个 variant specs，`Compiler::compile()` 产出完整的 6 变体（含各自的 MemoryPlan 和 ShapeId）。
2. **Atlas 层**：`build_graph_atlas()` 为 6 变体全量填表，`is_shape_invariant_graph()` 判定 shape 无关图共享 `kShapeInvariant`，shape 相关图各自 `shape_id`。
3. **Capture 层**：`pre_capture()` 自动去重（`Key{cg, gid, shape_id}`），无需额外修改。
4. **运行时 GPU 层**：每个 rank 线程在 **batch 循环外部** 预解析 normal batch 和 last batch 的所有 `cudaGraphExec_t` handles。循环内部零 variant 切换开销。
5. **运行时 CPU 层**：同步支持 variant 切换，修复指标累积和加权平均 bug。
6. **渐进式分辨率**：epoch 开始时根据 `get_current_train_resolution()` 选择 v0/v1 或 v2/v3，天然支持。

**改动量**：约 400 行，集中在 1 个头文件 + 1 个 cpp 文件。**不修改任何 kernel、Compiler 核心、pre_capture/capture 基础设施。**
