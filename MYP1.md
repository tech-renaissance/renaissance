# MYP1 — 多 Shape 捕获、Last Batch 处理与渐进式分辨率的完备方案

> **综合 LRF1/LRF2/LRF3 三部分析文档 + 源码全量验证 + 修正缺陷，给出唯一权威实施方案**

---

## 一、问题根因：三层断裂与 CUDA Graph 物理约束

### 1.1 三层系统性断裂

| 层级 | 设计意图 | 当前实现 | 断裂点 |
|------|---------|---------|--------|
| **Compiler** | `Compiler::compile()` **已支持** `variant_specs` 参数（`compiler.h:L115-L125`），可生成 6 个 Variant（含各自 MemoryPlan） | `on_prepare()`（`deep_learning_task.h:L287-L288`）只传 1 个 `base_spec`，`variant_specs` 为空 → `result.variants.size() == 1` | 未生成 last batch / lowres / val 的 MemoryPlan |
| **GraphAtlas** | 6×28 映射表，shape 相关图各自 `mp` + `shape_id`，shape 无关图共享 `kShapeInvariant` | `build_graph_atlas()`（`deep_learning_task.cpp:L490-L541`）**只填 variant 0**，所有 slot 硬编码 `shape_id = kShapeInvariant`、`mp = active_memory_plan_` | `pre_capture()` 去重时所有 variant 的 `Key` 相同 → 只捕获一次 |
| **运行时** | 根据 batch 类型（normal/last）和分辨率（base/alt）选择 correct variant | `build_exec_table()`（`deep_learning_task.cpp:L598`）只 `atlas.index(0, gid)`；`run_train_epoch_gpu()`（`deep_learning_task.cpp:L913`）所有 batch 使用同一 `g_tab = gpu_exec_.graphs[rank]` | last batch 使用 grid=standard_bs 的 CUDA Graph 处理 last_batch_size 个样本 |

### 1.2 为什么必须切换 Variant（CUDA Graph 物理约束）

CUDA Graph 的 kernel grid 维度在 **capture 时固定**，replay 时不可变：

```cpp
// softmax_ce_op.cpp 中
int batch = logits_dt.shape.n();    // ← 从 MemoryPlan 读取，capture 时固定进 graph
```

如果 capture 时使用 base MemoryPlan（batch=512），则 `grid.x = 512` **永久固定**。即使运行时修改 `batch_size_ptr` 标量为 402，grid 仍然是 512，导致后 110 个 block 处理**残留/越界数据**。

**唯一正确做法**：为 variant[1]（train_last, batch=402）**单独 capture** CUDA Graph。`pre_capture()` 的 `capture_all_for_rank()` 已内置此机制（`ctx.set_memory_plan(mp)`），问题仅在于 `build_graph_atlas()` 未提供正确的 `mp` 和 `shape_id`。

### 1.3 与 LRF1-3 的差异修正

| 议题 | LRF1 | LRF2 | LRF3 | **MYP1 采纳** |
|------|------|------|------|------------|
| Variant 命名 | v0=res_begin, v2=res_end | 同 LRF1 | v0=res_end, v2=res_begin | **采用 `compile_spec.h:L28-L33` 官方注释**：v0=res_begin, v2=res_end |
| ShapeId 来源 | `variant_compile_specs_[]` | 同 LRF1 | `variant_compile_specs_[]` | **采用 `CompileSpec::get_shape_id()`**，从 `compile_spec.h:L66-L69` 计算 |
| 新增成员 | `compiled_result_` 存 Compiler::Result | `variant_memory_plans_` | `variant_compile_specs_` + `variant_memory_plans_` | **采用 LRF3：`variant_memory_plans_`（6个）+ `variant_compile_specs_`（6个）** |
| GpuExecTable | `variant_graphs[v][rank][slot]` | 同 LRF1 | 同 LRF1 | **统一采用三级数组** |
| CPU 路径 | 含 variant 切换 + 指标修复 | 含 variant 切换 | 含 variant 切换 + 加权平均修复 | **采用完整修复** |
| EMA validation | 未提及 | 未提及 | 赋值 but skip | **沿用 `(void)validate_ema` 跳过** |

---

## 二、6-Variant 精确定义（基于源码全量验证）

### 2.1 Variant 映射表

根据 `compile_spec.h:L28-L33` 官方注释 + `CompileSpec::from_global_registry()`（`compile_spec.h:L49-L58`）实际行为：

| Variant | 名称 | batch_size | actual_resolution | 来源 API | ComputationGraph |
|---------|------|-----------|-------------------|---------|-----------------|
| 0 | `train_base` | `global_batch_size / world_size` | `train_sample_resolution_begin()` | `CompileSpec::from_global_registry()` | `train_cg` |
| 1 | `train_last` | `num_train % (global_bs / world_size)` | `train_sample_resolution_begin()` | `global_registry.cpp:L2085-L2088` | `train_cg` |
| 2 | `train_lowres` | `global_batch_size / world_size` | `train_sample_resolution_end()` | `global_registry.cpp` | `train_cg` |
| 3 | `train_lowres_last` | `num_train % (global_bs / world_size)` | `train_sample_resolution_end()` | 同上 | `train_cg` |
| 4 | `val_base` | `global_batch_size / world_size` | `val_sample_resolution()` | `global_registry.cpp` | `infer_cg` |
| 5 | `val_last` | `num_val % (global_bs / world_size)` | `val_sample_resolution()` | `global_registry.cpp:L2089-L2092` | `infer_cg` |

### 2.2 各 Variant 的 ShapeId 四元组

`ShapeId{batch_size, actual_resolution, actual_resolution, num_color_channels}` — 来自 `compile_spec.h:L66-L69`

以 CIFAR-10（standard_bs=512, res=32, C=3）为例：
- **v0**: `ShapeId{512, 32, 32, 3}`
- **v1**: `ShapeId{512, 32, 32, 3}` （整除时 last_bs == standard_bs → **自然去重**）
- **v2**: `ShapeId{512, 32, 32, 3}` （单分辨率时 res_begin == res_end → 自然去重）
- **v3**: `ShapeId{512, 32, 32, 3}` （同上）
- **v4**: `ShapeId{512, 32, 32, 3}`
- **v5**: `ShapeId{512, 32, 32, 3}`

全部 **6 变体同 ShapeId → 零额外开销**。

以 ImageNet（standard_bs=512, last_bs=402, res=224, begin=128, end=224, C=3）为例：
- **v0**: `ShapeId{512, 224, 224, 3}`
- **v1**: `ShapeId{402, 224, 224, 3}`
- **v2**: `ShapeId{512, 128, 128, 3}`
- **v3**: `ShapeId{402, 128, 128, 3}`
- **v4**: `ShapeId{512, 224, 224, 3}` → **与 v0 相同**（去重：val_base = train_base 共用）
- **v5**: `ShapeId{106, 224, 224, 3}` （假设 val_last=106）

共 **5 个唯一 ShapeId**（v0/v4 共享, v1, v2, v3, v5）。

### 2.3 渐进式分辨率下的 Variant 选择

`get_current_train_resolution()`（`deep_learning_task.h:L528`）调用链 → `global_registry.cpp`：
```cpp
epoch < boundary_epoch → train_sample_resolution_begin()
epoch >= boundary_epoch → train_sample_resolution_end()
```

`run_gpu()`（`deep_learning_task.cpp:L760-L790`）每 epoch 调用 `set_current_resolution_train(new_res)` 更新。

**Variant 选择逻辑**（在 `run_train_epoch_gpu()` 内部）：
```cpp
int current_res = get_current_train_resolution();
int res_begin = registry.train_sample_resolution_begin();
int res_end   = registry.train_sample_resolution_end();

bool at_begin_res = (current_res == res_begin);
// at_begin_res → v0 (base), v1 (last)
// !at_begin_res → v2 (lowres), v3 (lowres_last)
```

| Epoch | Resolution | v_base | v_last |
|-------|-----------|--------|--------|
| 0..boundary-1 | `train_sample_resolution_begin()` | 0 | 1 |
| boundary..N | `train_sample_resolution_end()` | 2 | 3 |

---

## 三、ShapeId 去重机制全量验证

### 3.1 去重核心逻辑

`pre_capture()`（`captured_graph.cpp:L182-L201`）：

```cpp
CapturedGraph::Key key{cg, gid, shape_id};
// 遍历已捕获的图，若 Key 相同 → 复用
// 不同 → 新建
```

`Key` = `{ComputationGraph*, GraphId, ShapeId}` → 同 `cg`, `gid`, `shape_id` → 共享同一张 captured graph。

### 3.2 Graph 分类（按 Shape 敏感性，基于 `computation_graph.h:L73-L102` 枚举）

**完全 Shape 无关**（全部 6 variant 共享 `kShapeInvariant` = `{0,0,0,0}`）：

| GraphId | 说明 |
|---------|------|
| `TRANSFER_A`, `TRANSFER_B` | H2D 传输，始终传输整个缓冲区的全部字节 |
| `ZERO_GRAD` | 梯度清零 |
| `FIRST_COMM`, `DEEP_COMM`, `STATS_COMM` | AllReduce 通信 |
| `CAST_DEEP_GRAD_FP16_TO_FP32`, `CAST_FIRST_GRAD_FP16_TO_FP32` | AMP cast |
| `NAN_CHECK_AND_GRAD_SCALING` | AMP NaN 检查 |
| `OPTIMIZER` | 权重更新（权重 shape 不随 batch 变化） |
| `EMA_UPDATE` | EMA 参数更新 |
| `CAST_MAIN_FP32_TO_FP16`, `CAST_EMA_FP32_TO_FP16` | 模型 cast |
| `ACCUM_METRICS`, `ACCUM_METRICS_TRAIN_LAST`, `ACCUM_METRICS_VAL_LAST` | 指标累积（内部 `batch_size_ptr` 动态读取） |
| `VAL_RESULT_COMM` | 验证结果 AllReduce |
| `CLEAR_METRICS` | 累积区清零 |

**Shape 相关**（不同 variant → 不同 ShapeId → 各自 capture）：

| GraphId | 说明 | 受影响的 factor |
|---------|------|---------------|
| `FIRST_LAYER_FWD_A`, `FIRST_LAYER_FWD_B` | 首层前向 | batch, resolution |
| `DEEP_FWD_BWD` | 深层前向+反向融合 | batch, resolution |
| `FIRST_LAYER_BWD_A`, `FIRST_LAYER_BWD_B` | 首层反向 | batch, resolution |
| `INF_MAIN_A`, `INF_MAIN_B` | 推理主模型 | batch, resolution |
| `INF_EMA_A`, `INF_EMA_B` | 推理 EMA 模型 | batch, resolution |

### 3.3 预期 Captured Graph 数量

| 类别 | GraphId 数 | 唯一 ShapeId 数 | 图数 |
|------|-----------|---------------|------|
| Shape 无关 | ~16（不含 SIMPLE_TASK_GRAPH, STATS_COMM 可能未使用） | 1 (kShapeInvariant) | ~15 |
| Shape 相关训练 | 6 (FIRST_LAYER_FWD_A/B + DEEP_FWD_BWD + FIRST_LAYER_BWD_A/B) | 最多 4 (v0/v1/v2/v3) | 最多 24 |
| Shape 相关推理 | 4 (INF_MAIN_A/B + INF_EMA_A/B) | 最多 2 (v4/v5) | 最多 8 |
| **总计** | | | **最多 ~47，实际 ~30-47** |

**去重乘积**：
- CIFAR-10 不整除 + single_resolution：只有 2 个唯一 ShapeId（v0=v2=v4 共享, v1=v3=v5 共享）
- ImageNet 渐进分辨率 + 不整除：5 个唯一 ShapeId

---

## 四、完整实施方案

### 改动文件清单（不含 compiler / captured_graph / kernel）

| 文件 | 改动点 | 行数估算 |
|------|--------|---------|
| `include/task/deep_learning_task.h` | 新增成员 `variant_memory_plans_`、`variant_compile_specs_`、`compiled_result_`；扩展 `GpuExecTable` | ~40 |
| `src/task/deep_learning_task.cpp` | 重写 `on_prepare()`、`build_graph_atlas()`、`build_exec_table()`、`run_train_epoch_gpu()`、`run_val_epoch_gpu()`、`run_train_epoch_cpu()`、`run_val_epoch_cpu()` | ~380 |
| **总计** | **2 文件** | **~420 行** |

> **不修改**：`compiler.cpp`、`captured_graph.cpp`、`compile_spec.h`、`computation_graph.h`，以及任何 kernel 文件。

---

### Step 0: `DeepLearningTask` 新增成员（`deep_learning_task.h`）

在 `private:` 区域 `memory_plan_ptr_` 成员下方新增：

```cpp
// ===== 6-Variant 系统 =====
/// 各变体的独立 MemoryPlan（slot_bytes 相同，shape 不同）
std::array<std::unique_ptr<MemoryPlan>, GraphAtlas::kMaxVariants> variant_memory_plans_;

/// 各变体的 CompileSpec（用于 ShapeId 推导）
std::array<CompileSpec, GraphAtlas::kMaxVariants> variant_compile_specs_;

/// 持有 Compiler::Result 所有权（保证 ComputationGraph 和 Variant 数组生命周期）
Compiler::Result compiled_result_;
```

修改 `GpuExecTable`：

```cpp
#ifdef TR_USE_CUDA
struct GpuExecTable {
    // variant_graphs[variant_idx][rank][slot] → cudaGraphExec_t
    // variant_idx 0~5 对应 train_base / train_last / train_lowres / train_lowres_last / val_base / val_last
    std::array<std::vector<std::vector<cudaGraphExec_t>>, GraphAtlas::kMaxVariants> variant_graphs;
    std::vector<int> device_ids;
};
GpuExecTable gpu_exec_;
std::vector<float*> lr_pinned_;
#endif
```

---

### Step 1: `on_prepare()` 生成完整 6-Variant（`deep_learning_task.h`）

替换 `on_prepare()` 中的 `Compiler::compile()` 调用（当前 `deep_learning_task.h:L287-L346`）：

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
    s.actual_resolution = gr.train_sample_resolution_end();
    variant_specs.push_back(s);
}
// v3: train_lowres_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_train_batch_size();
    s.actual_resolution = gr.train_sample_resolution_end();
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

// ===== 保存所有权并抽取指针 =====
for (size_t i = 0; i < result.variants.size() && i < GraphAtlas::kMaxVariants; ++i) {
    variant_memory_plans_[i] = std::move(result.variants[i].memory_plan);
    if (i == 0) {
        variant_compile_specs_[i] = base_spec;
    } else if (i - 1 < variant_specs.size()) {
        variant_compile_specs_[i] = variant_specs[i - 1];
    }
}
variant_compile_specs_[4] = variant_specs[3];  // val_base
variant_compile_specs_[5] = variant_specs[4];  // val_last

memory_plan_ptr_ = std::move(variant_memory_plans_[0]);
if (!memory_plan_ptr_->is_finalized()) {
    memory_plan_ptr_->finalize();
}
active_memory_plan_ = memory_plan_ptr_.get();

// 保留原有 add_graph() 调用，ComputationGraph 所有权移入 named_graphs_
// compiled_result_.variants 仍需持有（因为 MemoryPlan unique_ptr 已移出到 variant_memory_plans_）
```

> **注意**：`compiled_result_` 需保存以确保 `Compiler::Result::variants` 数组生命周期覆盖 `variant_memory_plans_` 的使用期。实际上 `variant_memory_plans_[i]` 已通过 `std::move` 取得所有权，`compiled_result_` 可以析构。

> **简化**：不需要 `compiled_result_` 成员（因为 MemoryPlan 所有权已移交）。可直接让 `result` 作为局部变量析构。

---

### Step 2: `build_graph_atlas()` 全量 6-Variant 填表（`deep_learning_task.cpp`）

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

    // ===== Graph 分类谓词（基于 computation_graph.h 枚举验证）=====
    auto is_shape_invariant = [](GraphId gid) -> bool {
        switch (gid) {
            case GraphId::TRANSFER_A: case GraphId::TRANSFER_B:
            case GraphId::ZERO_GRAD:
            case GraphId::FIRST_COMM: case GraphId::DEEP_COMM: case GraphId::STATS_COMM:
            case GraphId::CAST_DEEP_GRAD_FP16_TO_FP32: case GraphId::CAST_FIRST_GRAD_FP16_TO_FP32:
            case GraphId::NAN_CHECK_AND_GRAD_SCALING:
            case GraphId::OPTIMIZER: case GraphId::EMA_UPDATE:
            case GraphId::CAST_MAIN_FP32_TO_FP16: case GraphId::CAST_EMA_FP32_TO_FP16:
            case GraphId::ACCUM_METRICS: case GraphId::ACCUM_METRICS_TRAIN_LAST:
            case GraphId::ACCUM_METRICS_VAL_LAST:
            case GraphId::VAL_RESULT_COMM: case GraphId::CLEAR_METRICS:
                return true;
            default: return false;
        }
    };

    auto is_train_shape_related = [](GraphId gid) -> bool {
        switch (gid) {
            case GraphId::FIRST_LAYER_FWD_A: case GraphId::FIRST_LAYER_FWD_B:
            case GraphId::DEEP_FWD_BWD:
            case GraphId::FIRST_LAYER_BWD_A: case GraphId::FIRST_LAYER_BWD_B:
                return true;
            default: return false;
        }
    };

    auto is_infer_shape_related = [](GraphId gid) -> bool {
        switch (gid) {
            case GraphId::INF_MAIN_A: case GraphId::INF_MAIN_B:
            case GraphId::INF_EMA_A: case GraphId::INF_EMA_B:
                return true;
            default: return false;
        }
    };

    // ===== 为 6 个 variant 填表 =====
    for (size_t v = 0; v < GraphAtlas::kMaxVariants; ++v) {
        const MemoryPlan* mp = variant_memory_plans_[v].get();
        if (!mp) continue;

        ShapeId shape_id = variant_compile_specs_[v].get_shape_id();

        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);

            // 确定 ComputationGraph
            const ComputationGraph* cg = nullptr;
            if (is_train_shape_related(gid) || is_shape_invariant(gid)) {
                cg = train_cg_;
            } else if (is_infer_shape_related(gid)) {
                cg = infer_cg_;
            }
            if (!cg || cg->nodes(gid).empty()) continue;

            auto& sl = atlas.slot(v, gi);
            sl.cg = cg;
            sl.stream_kind = stream_for(gid);

            if (is_shape_invariant(gid)) {
                sl.mp = variant_memory_plans_[0].get();
                sl.shape_id = kShapeInvariant;
            } else {
                sl.mp = mp;
                sl.shape_id = shape_id;
            }
        }
    }

    name_to_gid_.clear();
    if (train_cg_) name_to_gid_["train"] = GraphId::DEEP_FWD_BWD;
    if (infer_cg_) name_to_gid_["inference"] = GraphId::INF_MAIN_A;

    return atlas;
}
```

---

### Step 3: `build_exec_table()` 6-Variant 预解析（`deep_learning_task.cpp`）

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
#endif
}
```

---

### Step 4: `run_train_epoch_gpu()` 变体切换 + 预解析（`deep_learning_task.cpp`）

**核心原则**：在 batch 循环外部完成所有 graph 预解析，循环内部零数组索引。

```cpp
#ifdef TR_USE_CUDA
float DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    int32_t loss_id = active_memory_plan_->baseline().loss;

    // ===== Epoch-level resolution → variant selection =====
    int current_res = registry.get_current_resolution_train();
    int res_begin = registry.train_sample_resolution_begin();
    int res_end   = registry.train_sample_resolution_end();
    bool at_begin_res = (current_res == res_begin);
    size_t v_base = at_begin_res ? 0 : 2;
    size_t v_last = at_begin_res ? 1 : 3;

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

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

                // ===== PRE-RESOLVE: normal batch handles (v_base) =====
                const auto& g_n = gpu_exec_.variant_graphs[v_base][rank];
                auto n_xfer_a  = g_n[S(GraphSlot::XFER_A)];
                auto n_xfer_b  = g_n[S(GraphSlot::XFER_B)];
                auto n_deep_a  = g_n[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto n_deep_b  = g_n[S(GraphSlot::FWD_BWD_DEEP_B)];
                auto n_fwd_a   = g_n[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto n_fwd_b   = g_n[S(GraphSlot::FIRST_LAYER_FWD_B)];
                auto n_bwd_a   = g_n[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto n_bwd_b   = g_n[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto n_zg      = g_n[S(GraphSlot::ZERO_GRAD)];
                auto n_dar     = g_n[S(GraphSlot::DEEP_ALLREDUCE)];
                auto n_far     = g_n[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto n_wu      = g_n[S(GraphSlot::WEIGHT_UPDATE)];
                auto n_cdg     = g_n[S(GraphSlot::CAST_DEEP_GRAD)];
                auto n_cfg     = g_n[S(GraphSlot::CAST_FIRST_GRAD)];
                auto n_ncg     = g_n[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                auto n_cm      = g_n[S(GraphSlot::CAST_MAIN)];
                auto n_accum   = g_n[S(GraphSlot::ACCUM_METRICS)];
                auto n_clear   = g_n[S(GraphSlot::CLEAR_METRICS)];

                // ===== PRE-RESOLVE: last batch handles (v_last) =====
                const auto& g_l = gpu_exec_.variant_graphs[v_last][rank];
                auto l_xfer_a  = g_l[S(GraphSlot::XFER_A)];      // same ptr (shape-invariant)
                auto l_xfer_b  = g_l[S(GraphSlot::XFER_B)];      // same ptr
                auto l_deep_a  = g_l[S(GraphSlot::FWD_BWD_DEEP_A)]; // DIFFERENT grid
                auto l_deep_b  = g_l[S(GraphSlot::FWD_BWD_DEEP_B)]; // DIFFERENT grid
                auto l_fwd_a   = g_l[S(GraphSlot::FIRST_LAYER_FWD_A)]; // DIFFERENT
                auto l_fwd_b   = g_l[S(GraphSlot::FIRST_LAYER_FWD_B)]; // DIFFERENT
                auto l_bwd_a   = g_l[S(GraphSlot::FIRST_LAYER_BWD_A)]; // DIFFERENT
                auto l_bwd_b   = g_l[S(GraphSlot::FIRST_LAYER_BWD_B)]; // DIFFERENT
                auto l_zg      = g_l[S(GraphSlot::ZERO_GRAD)];   // same ptr
                auto l_dar     = g_l[S(GraphSlot::DEEP_ALLREDUCE)];  // same ptr
                auto l_far     = g_l[S(GraphSlot::FIRST_LAYER_ALLREDUCE)]; // same ptr
                auto l_wu      = g_l[S(GraphSlot::WEIGHT_UPDATE)];  // same ptr
                auto l_cdg     = g_l[S(GraphSlot::CAST_DEEP_GRAD)];  // same ptr
                auto l_cfg     = g_l[S(GraphSlot::CAST_FIRST_GRAD)]; // same ptr
                auto l_ncg     = g_l[S(GraphSlot::NAN_CHECK_GRAD_SCALE)]; // same ptr
                auto l_cm      = g_l[S(GraphSlot::CAST_MAIN)];     // same ptr
                auto l_accum_t = g_l[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];
                auto l_clear   = g_l[S(GraphSlot::CLEAR_METRICS)]; // same ptr

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_c3 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_up = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr = [&]() { cudaStreamSynchronize(s_trans); };

                TransferStation* ts = ... /* 保持不变 */;

                float lr;
                float* lr_dev_ptr = static_cast<float*>(ctx.ptr_at(lr_dtensor_id_));
                bool frozen = is_first_layer_frozen();

                // ===== Batch 0 预传输 (使用 normal variant) =====
                {
                    ts->wait_buffer_readable(0);
                    if (n_xfer_a) cudaGraphLaunch(n_xfer_a, s_trans);
                    sync_tr();
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                    if (using_amp && n_cm) { cudaGraphLaunch(n_cm, s_up); sync_up(); }
                    if (n_clear) cudaGraphLaunch(n_clear, s_up);
                }

                // ===== Normal batches (0 .. batches-2) 使用 n_* =====
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;
                    auto g_fwd   = from_a ? n_fwd_a : n_fwd_b;
                    auto g_deep  = from_a ? n_deep_a : n_deep_b;
                    auto g_xfer_n = from_a ? n_xfer_b : n_xfer_a;
                    auto g_first = from_a ? n_bwd_a : n_bwd_b;

                    if (n_zg) cudaGraphLaunch(n_zg, s_up);
                    if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                    sync_comp(); sync_up();

                    ts->wait_buffer_readable(next_buf);

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
                    sync_comp(); sync_tr();

                    if (n_accum) cudaGraphLaunch(n_accum, s_up);
                    sync_up();

                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);

                    if (!frozen && g_first) cudaGraphLaunch(g_first, s_c1);
                    sync_comp();

                    if (using_amp && n_cdg) { cudaGraphLaunch(n_cdg, s_up); sync_up(); }
                    if (n_dar) cudaGraphLaunch(n_dar, s_up); sync_up();
                    if (using_amp && n_cfg) { cudaGraphLaunch(n_cfg, s_up); sync_up(); }
                    if (n_far) cudaGraphLaunch(n_far, s_up); sync_up();
                    if (using_amp && n_ncg) { cudaGraphLaunch(n_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batch);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (n_wu) cudaGraphLaunch(n_wu, s_up); sync_up();
                    if (using_amp && n_cm) { cudaGraphLaunch(n_cm, s_up); sync_up(); }
                }

                // ===== Last batch (batch = batches-1) 使用 l_* =====
                {
                    bool last_a = ((batches - 1) % 2 == 0);
                    auto g_fwd_l  = last_a ? l_fwd_a : l_fwd_b;
                    auto g_deep_l = last_a ? l_deep_a : l_deep_b;
                    auto g_first_l = last_a ? l_bwd_a : l_bwd_b;

                    if (l_zg) cudaGraphLaunch(l_zg, s_up);
                    if (g_fwd_l) cudaGraphLaunch(g_fwd_l, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
                    sync_comp();

                    if (l_accum_t) cudaGraphLaunch(l_accum_t, s_up);
                    sync_up();

                    if (!frozen && g_first_l) cudaGraphLaunch(g_first_l, s_c1);
                    sync_comp();

                    if (using_amp && l_cdg) { cudaGraphLaunch(l_cdg, s_up); sync_up(); }
                    if (l_dar) cudaGraphLaunch(l_dar, s_up); sync_up();
                    if (using_amp && l_cfg) { cudaGraphLaunch(l_cfg, s_up); sync_up(); }
                    if (l_far) cudaGraphLaunch(l_far, s_up); sync_up();
                    if (using_amp && l_ncg) { cudaGraphLaunch(l_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batches - 1);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (l_wu) cudaGraphLaunch(l_wu, s_up); sync_up();
                    if (using_amp && l_cm) { cudaGraphLaunch(l_cm, s_up); sync_up(); }
                }

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank)
        if (exc[rank]) std::rethrow_exception(exc[rank]);

    float train_loss = 0.0f;
    if (active_memory_plan_) {
        int32_t al_id = active_memory_plan_->baseline().accum_loss;
        if (al_id >= 0) {
            const auto& al_dt = active_memory_plan_->get_dtensor(al_id);
            Tensor h_al = fetch_from_rank(al_dt, 0);
            train_loss = h_al.data<float>()[0] / static_cast<float>(registry.num_train_samples());
        }
    }
    return train_loss;
}
#endif
```

---

### Step 5: `run_val_epoch_gpu()` 变体切换（`deep_learning_task.cpp`）

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
                if (err != cudaSuccess)
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank);

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                const auto& g_vb = gpu_exec_.variant_graphs[4][rank];  // val_base
                const auto& g_vl = gpu_exec_.variant_graphs[5][rank];  // val_last

                auto vb_xfer_a = g_vb[S(GraphSlot::XFER_A)];
                auto vb_xfer_b = g_vb[S(GraphSlot::XFER_B)];
                auto vb_inf_a  = g_vb[S(GraphSlot::INF_MAIN_A)];
                auto vb_inf_b  = g_vb[S(GraphSlot::INF_MAIN_B)];
                auto vb_accum  = g_vb[S(GraphSlot::ACCUM_METRICS)];
                auto vb_clear  = g_vb[S(GraphSlot::CLEAR_METRICS)];
                auto vb_vcomm  = g_vb[S(GraphSlot::VAL_RESULT_ALLREDUCE)];

                auto vl_xfer_a = g_vl[S(GraphSlot::XFER_A)];  // same
                auto vl_xfer_b = g_vl[S(GraphSlot::XFER_B)];  // same
                auto vl_inf_a  = g_vl[S(GraphSlot::INF_MAIN_A)];  // DIFFERENT
                auto vl_inf_b  = g_vl[S(GraphSlot::INF_MAIN_B)];  // DIFFERENT
                auto vl_accum  = g_vl[S(GraphSlot::ACCUM_METRICS_VAL_LAST)];
                auto vl_clear  = g_vl[S(GraphSlot::CLEAR_METRICS)];  // same
                auto vl_vcomm  = g_vl[S(GraphSlot::VAL_RESULT_ALLREDUCE)]; // same

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_c3 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
                auto sync_comp = [&]() { cudaStreamSynchronize(s_c1); cudaStreamSynchronize(s_c2); cudaStreamSynchronize(s_c3); };
                auto sync_up = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr = [&]() { cudaStreamSynchronize(s_trans); };

                TransferStation* ts = /* ... 标准获取逻辑 ... */;

                const auto& b = active_memory_plan_->baseline();
                int32_t loss_id = b.loss, top1_id = b.top1, top5_id = b.top5;

                if (vb_clear) cudaGraphLaunch(vb_clear, s_up);

                for (int batch = 0; batch < val_batches; ++batch) {
                    int buf = batch % 2;
                    bool is_last = (batch == val_batches - 1);

                    auto g_xfer = (buf == 0)
                        ? (is_last ? vl_xfer_a : vb_xfer_a)
                        : (is_last ? vl_xfer_b : vb_xfer_b);
                    auto g_inf = (buf == 0)
                        ? (is_last ? vl_inf_a : vb_inf_a)
                        : (is_last ? vl_inf_b : vb_inf_b);
                    auto g_accum = is_last ? vl_accum : vb_accum;

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
                if (vb_vcomm) cudaGraphLaunch(vb_vcomm, s_up);
                sync_up();

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank)
        if (exc[rank]) std::rethrow_exception(exc[rank]);

    float avg_loss = 0.0f, avg_top1 = 0.0f, avg_top5 = 0.0f;
    if (active_memory_plan_) {
        const auto& bl = active_memory_plan_->baseline();
        if (bl.accum_loss >= 0) {
            Tensor h = fetch_from_rank(active_memory_plan_->get_dtensor(bl.accum_loss), 0);
            avg_loss = h.data<float>()[0] / static_cast<float>(registry.num_val_samples());
        }
        if (bl.accum_top1 >= 0) {
            Tensor h = fetch_from_rank(active_memory_plan_->get_dtensor(bl.accum_top1), 0);
            avg_top1 = h.data<float>()[0] / static_cast<float>(registry.num_val_samples());
        }
        if (bl.accum_top5 >= 0) {
            Tensor h = fetch_from_rank(active_memory_plan_->get_dtensor(bl.accum_top5), 0);
            avg_top5 = h.data<float>()[0] / static_cast<float>(registry.num_val_samples());
        }
    }
    return {avg_loss, avg_top1, avg_top5};
#else
    (void)validate_ema;
    return {0.0f, 0.0f, 0.0f};
#endif
}
```

---

### Step 6: `run_train_epoch_cpu()` 变体支持 + 指标修复（`deep_learning_task.cpp`）

**修复要点**：
1. 支持 variant 切换（`atlas.index(v, gid)` 代替 `atlas.index(0, gid)`）
2. last batch 使用不同的 shape-dependent graph index
3. **从 `accum_loss` 读取 epoch 平均 loss**（当前只读 last batch 的 `loss_id`）

```cpp
float DeepLearningTask::run_train_epoch_cpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();

    TransferStation* ts = /* ... 标准获取 ... */;

    DeviceContext& ctx = context(0);
    const auto& atlas = captured_result_.atlas;
    const auto& graphs = captured_result_.graphs;

    // ===== Epoch-level variant selection =====
    int current_res = registry.get_current_resolution_train();
    int res_begin = registry.train_sample_resolution_begin();
    bool at_begin_res = (current_res == res_begin);
    size_t v_base = at_begin_res ? 0 : 2;
    size_t v_last = at_begin_res ? 1 : 3;

    auto idx_for = [&atlas](GraphId gid, size_t variant_idx) -> int32_t {
        return atlas.index(variant_idx, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    // Pre-resolve normal batch indices (v_base)
    int32_t idx_xfer_a = idx_for(GraphId::TRANSFER_A, v_base);
    int32_t idx_xfer_b = idx_for(GraphId::TRANSFER_B, v_base);
    int32_t idx_deep_n = idx_for(GraphId::DEEP_FWD_BWD, v_base);
    int32_t idx_fwd_a_n = idx_for(GraphId::FIRST_LAYER_FWD_A, v_base);
    int32_t idx_fwd_b_n = idx_for(GraphId::FIRST_LAYER_FWD_B, v_base);
    int32_t idx_bwd_a_n = idx_for(GraphId::FIRST_LAYER_BWD_A, v_base);
    int32_t idx_bwd_b_n = idx_for(GraphId::FIRST_LAYER_BWD_B, v_base);
    int32_t idx_zg   = idx_for(GraphId::ZERO_GRAD, v_base);
    int32_t idx_far  = idx_for(GraphId::FIRST_COMM, v_base);
    int32_t idx_dar  = idx_for(GraphId::DEEP_COMM, v_base);
    int32_t idx_opt  = idx_for(GraphId::OPTIMIZER, v_base);

    // Pre-resolve last batch indices (v_last) — shape-dependent ones differ
    int32_t idx_deep_l = idx_for(GraphId::DEEP_FWD_BWD, v_last);
    int32_t idx_fwd_a_l = idx_for(GraphId::FIRST_LAYER_FWD_A, v_last);
    int32_t idx_fwd_b_l = idx_for(GraphId::FIRST_LAYER_FWD_B, v_last);
    int32_t idx_bwd_a_l = idx_for(GraphId::FIRST_LAYER_BWD_A, v_last);
    int32_t idx_bwd_b_l = idx_for(GraphId::FIRST_LAYER_BWD_B, v_last);

    int32_t loss_id = active_memory_plan_->baseline().loss;
    void* loss_ptr = loss_id >= 0 ? ctx.ptr_at(loss_id) : nullptr;
    void* lr_ptr   = ctx.ptr_at(lr_dtensor_id_);
    bool frozen = is_first_layer_frozen();

    // Batch 0 prefetch
    ts->wait_buffer_readable(0);
    launch(idx_xfer_a);
    ts->set_buffer_readable(0, false);
    ts->set_buffer_writeable(0, true);

    // ===== Normal batches (0 .. batches-2) =====
    for (int batch = 0; batch < batches - 1; ++batch) {
        bool from_a = (batch % 2 == 0);
        int next_buf = from_a ? 1 : 0;

        launch(idx_zg);
        launch(from_a ? idx_fwd_a_n : idx_fwd_b_n);

        ts->wait_buffer_readable(next_buf);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_n);
        launch(from_a ? idx_xfer_b : idx_xfer_a);

        ts->set_buffer_readable(next_buf, false);
        ts->set_buffer_writeable(next_buf, true);

        if (!frozen) launch(from_a ? idx_bwd_a_n : idx_bwd_b_n);
        launch(idx_dar);

        {
            float lr = fetch_lr_for_batch(batch);
            *static_cast<float*>(lr_ptr) = lr;
        }
        launch(idx_far);
        launch(idx_opt);
    }

    // ===== Last batch (batches-1) 使用 v_last indices =====
    {
        bool last_a = ((batches - 1) % 2 == 0);

        launch(idx_zg);
        launch(last_a ? idx_fwd_a_l : idx_fwd_b_l);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_l);

        if (!frozen) launch(last_a ? idx_bwd_a_l : idx_bwd_b_l);
        launch(idx_dar);

        {
            float lr = fetch_lr_for_batch(batches - 1);
            *static_cast<float*>(lr_ptr) = lr;
        }
        launch(idx_far);
        launch(idx_opt);
    }

    // ===== 从 accum_loss 读取（修复：原只读 last batch loss_id）=====
    float train_loss = 0.0f;
    int32_t accum_loss_id = active_memory_plan_->baseline().accum_loss;
    if (accum_loss_id >= 0) {
        const auto& accum_dt = active_memory_plan_->get_dtensor(accum_loss_id);
        Tensor h_accum = fetch_from_rank(accum_dt, 0);
        train_loss = h_accum.data<float>()[0] / static_cast<float>(registry.num_train_samples());
    }
    return train_loss;
}
```

---

### Step 7: `run_val_epoch_cpu()` 变体支持 + 加权平均修复（`deep_learning_task.cpp`）

**修复要点**：
1. `atlas.index(4/5, gid)` 区分 val_base / val_last
2. 按 **加权平均**（`batch_loss × batch_size`）代替简单平均（`acc_loss / val_batches`）

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_cpu(bool validate_ema) {
    (void)validate_ema;

    auto& registry = GlobalRegistry::instance();
    int val_batches = registry.get_val_steps();

    const auto& atlas = captured_result_.atlas;
    const auto& graphs = captured_result_.graphs;

    auto idx_for = [&atlas](GraphId gid, size_t v) -> int32_t {
        return atlas.index(v, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    int32_t idx_xfer_a_vb = idx_for(GraphId::TRANSFER_A, 4);
    int32_t idx_xfer_b_vb = idx_for(GraphId::TRANSFER_B, 4);
    int32_t idx_inf_a_vb  = idx_for(GraphId::INF_MAIN_A, 4);
    int32_t idx_inf_b_vb  = idx_for(GraphId::INF_MAIN_B, 4);

    int32_t idx_xfer_a_vl = idx_for(GraphId::TRANSFER_A, 5);  // same
    int32_t idx_xfer_b_vl = idx_for(GraphId::TRANSFER_B, 5);  // same
    int32_t idx_inf_a_vl  = idx_for(GraphId::INF_MAIN_A, 5);  // DIFFERENT
    int32_t idx_inf_b_vl  = idx_for(GraphId::INF_MAIN_B, 5);  // DIFFERENT

    const auto& bl = active_memory_plan_->baseline();
    int32_t loss_id = bl.loss, top1_id = bl.top1, top5_id = bl.top5;
    DeviceContext& ctx = context(0);
    void* loss_ptr = loss_id >= 0 ? ctx.ptr_at(loss_id) : nullptr;
    void* top1_ptr = top1_id >= 0 ? ctx.ptr_at(top1_id) : nullptr;
    void* top5_ptr = top5_id >= 0 ? ctx.ptr_at(top5_id) : nullptr;

    TransferStation* ts = /* ... 标准获取 ... */;

    double acc_loss = 0.0, acc_top1 = 0.0, acc_top5 = 0.0;

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

        // ===== 按 batch_size 加权累积（修复：原为简单平均 acc / val_batches）=====
        int bs = is_last ? registry.get_last_val_batch_size()
                         : registry.get_local_batch_size();
        float b_loss = loss_ptr ? *static_cast<float*>(loss_ptr) : 0.0f;
        float b_top1 = top1_ptr ? *static_cast<float*>(top1_ptr) : 0.0f;
        float b_top5 = top5_ptr ? *static_cast<float*>(top5_ptr) : 0.0f;

        acc_loss += static_cast<double>(b_loss) * static_cast<double>(bs);
        acc_top1 += static_cast<double>(b_top1) * static_cast<double>(bs);
        acc_top5 += static_cast<double>(b_top5) * static_cast<double>(bs);

        ts->set_buffer_readable(buf, false);
        ts->set_buffer_writeable(buf, true);
    }

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

## 五、渐进式分辨率集成

### 5.1 分辨率选择机制

`run_gpu()`（`deep_learning_task.cpp:L760-L790`）每 epoch 开头调用：

```cpp
if (reg.using_progressive_resolution()) {
    int new_res = get_current_train_resolution();
    const_cast<GlobalRegistry&>(reg).set_current_resolution_train(new_res);
    // ... crop / resize 配置 ...
}
```

`get_current_train_resolution()`（`deep_learning_task.h:L528`）：
- `epoch < boundary_epoch` → `train_sample_resolution_begin()`
- `epoch >= boundary_epoch` → `train_sample_resolution_end()`

### 5.2 变体选择

`run_train_epoch_gpu()` 开头从 `registry.get_current_resolution_train()` 读取当前分辨率并选择变体：

```cpp
int current_res = registry.get_current_resolution_train();
bool at_begin_res = (current_res == registry.train_sample_resolution_begin());
size_t v_base = at_begin_res ? 0 : 2;
size_t v_last = at_begin_res ? 1 : 3;
```

### 5.3 收敛保证

| 场景 | 分辨率数量 | 变体数 | 说明 |
|------|---------|--------|------|
| 单分辨率 | 1 (begin == end) | v0-v3 all same ShapeId | ShapeId 去重 → 无需额外 capture |
| 双分辨率 | 2 (begin != end) | v0/v1 vs v2/v3 different ShapeIds | 各分辨率独立 capture |
| 动态多分辨率（crop 渐变） | N | 同双分辨率 | 配置仅改变 `set_current_resolution_train()` 返回值，不改变体数 |

---

## 六、性能保证

### 6.1 Batch 循环内零开销

| 指标 | 当前 | 本方案 | 优化来源 |
|------|------|--------|---------|
| Graph handle 解析 | `g_tab[S(slot)]` 数组索引 | **直接局部变量** | 编译器可能将局部变量放入寄存器 |
| Last batch 分支 | 无（错误地同 graph） | **编译期分离**（独立代码块） | 独立局部变量组 |
| 多 RANK 扩展性 | `gpu_exec_.graphs[rank]` | `gpu_exec_.variant_graphs[v][rank]` | 每个 rank 独立预解析 |
| Captured graph 内存 | ~20 张 | **~30-47 张** | 同 ShapeId 去重零膨胀 |

### 6.2 编译时间估算

- Shape 无关图（约 15 个 GraphId）：始终 15 次 capture
- Shape 相关图（最多 10 个 GraphId × 最多 5 个唯一 ShapeId）：最多 50 次
- 额外的 cuDNN warmup：与 unique graph 数量近似线性，典型值 < 1s
- **总增加**：< 2s（相比当前）

---

## 七、多 RANK 兼容性

本方案**每一层**均完整支持多 RANK：

1. **`on_prepare()`**：`base_spec.batch_size = gr.get_local_batch_size()` — 已包含 `world_size` 除数
2. **`build_graph_atlas()`**：6×N 表非 rank 相关（`ComputationGraph` 在所有 rank 间共享）
3. **`build_exec_table()`**：`resolve(gid, rank, variant_idx)` — 为每个 `(rank, variant)` 独立解析
4. **`run_train_epoch_gpu()`**：每个 rank 线程从 `gpu_exec_.variant_graphs[v_base][rank]` 独立读取
5. **`GpuExecTable`**：`variant_graphs[v][rank][slot]` 三维数组，各 rank 独立

---

## 八、CPU 路径完整修复清单

| Bug | 位置 | 当前行为 | 修复 |
|-----|------|---------|------|
| **B1**: 无 variant 支持 | `run_train_epoch_cpu()` L1298-1299 | `atlas.index(0, gid)` — 只查 variant 0 | → `atlas.index(v_base/v_last, gid)` |
| **B2**: 无 variant 支持 | `run_val_epoch_cpu()` L1564-1565 | `atlas.index(0, gid)` — 只查 variant 0 | → `atlas.index(4/5, gid)` |
| **B3**: train loss 只读 last batch | `run_train_epoch_cpu()` L1397-1399 | `return h_loss.data<float>()[0]` — 仅最后 batch | → 从 `accum_loss` 读取并除以 `num_train_samples()` |
| **B4**: val 指标简单平均 | `run_val_epoch_cpu()` L1622-1624 | `acc_loss / val_batches` | → 加权平均：`sum(batch_loss × batch_size) / total_val` |

---

## 九、验证策略

### Phase 1：编译验证（1h）

- 编译通过，`result.variants.size() == 6`
- `pre_capture` 日志中 `captured` 计数 ≈ 目标值

### Phase 2：CIFAR-10 无回归（1h）

- `test_dl_full_gpu` 通过（整除 batch，6 变体同 ShapeId）
- `test_dl_full_amp` 通过

### Phase 3：不可整除验证（2h）

- 构造 `train=50003, val=10007, batch=200` 场景
- 验证 `last_train_batch_size=103, last_val_batch_size=7`
- 对比修复前后 loss/accuracy

### Phase 4：渐进式分辨率（1h）

- 启用 `progressive_resize(128, 224)` + `boundary_epoch(10)`
- 验证 epoch 0-9 使用 v0/v1（低分辨率），epoch 10+ 使用 v2/v3（高分辨率）

### Phase 5：CPU 路径（1h）

- `test_dl_full --cpu` 通过
- 验证 CPU 指标与 GPU 一致

### Phase 6：多 RANK（1h）

- `world_size=4` 配置下运行
- 验证各 rank 指标一致

---

## 十、修改文件总览

| 文件 | 修改行数 | 关键改动 |
|------|---------|---------|
| `include/task/deep_learning_task.h` | ~40 | `variant_memory_plans_`(6), `variant_compile_specs_`(6), 新的 `GpuExecTable` |
| `src/task/deep_learning_task.cpp` | ~380 | `on_prepare()` 传 5 variant_specs, `build_graph_atlas()` 6变体填表, `build_exec_table()` 6变体预解析, `run_train_epoch_gpu()` epoch级变体选择+预解析, `run_val_epoch_gpu()` 变体选择, `run_train_epoch_cpu()` 变体+指标修复, `run_val_epoch_cpu()` 变体+加权平均 |

**总计**: 2 文件，约 420 行。

**不修改**: `compiler.cpp`, `captured_graph.cpp`, `compile_spec.h`, `computation_graph.h` 及所有 kernel 文件、H2D 传输代码。

---

## 十一、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `build_graph_atlas()` val 变体误填 train shape 图 | 低 | 正确性 | `is_train_shape_related()` / `is_infer_shape_related()` 谓词 + cg 判定保证隔离 |
| `variant_memory_plans_[]` slot_bytes 不一致 | 极低 | 灾难 | `compute_max_slot_bytes` 已跨变体取 max；增加 DTensor offset 断言 |
| `transfer_to_rank` 使用 `active_memory_plan_` 的 `nbytes()` | 中 | 性能（仅 last batch 多拷贝） | last batch 的 FWD graph grid=last_batch_size，多拷贝数据不被处理，正确性无影响 |
| `get_current_val_resolution()` 不存在 | — | — | `val_sample_resolution()` 固定不变，始终 v4/v5；无需动态选择 |
| `CLEAR_METRICS` 在 GPU 路径的时序 | 中 | 首次 batch 0 可能异常 | 当前代码 `on_prepare` 后 CUDA `memset` 已在 `capture` 阶段清零；本方案不改变时序 |

---

## 十二、与 LRF1/LRF2/LRF3 的差异修正总结

| 修正点 | LRF1/2/3 的行为 | MYP1 修正 |
|--------|---------------|----------|
| Variant 命名与分辨率映射 | LRF3 将 v0 关联到 `res_end`，与 `compile_spec.h:L28-L33` 官方注释不一致 | **MYP1 严格遵循官方注释**：v0=res_begin, v2=res_end |
| `variant_compile_specs_[]` 数量 | LRF2 未保存 CompileSpec | **MYP1 保存全部 6 个**，用于 `get_shape_id()` |
| `compiled_result_` 成员 | LRF1 引入 `compiled_result_` 持所有权 | **MYP1 不需要**（MemoryPlan 已 move 出；ComputationGraph 已 move 入 `named_graphs_`） |
| GpuExecTable 数组大小 | LRF1/LRF2 使用 `GraphAtlas::kMaxVariants` | **MYP1 统一使用 `GraphAtlas::kMaxVariants`** |
| CPU 指标累积 | LRF1 从 `accum_loss` 读取，LRF2 保留 | **MYP1 确认从 `accum_loss` 读取** |
| CPU val 加权平均 | LRF1/LRF2 有加权平均，LRF3 有 | **MYP1 统一采用加权平均** |
| `STATS_COMM` 分类 | LRF1/2/3 未明确 | **MYP1 归入 shape 无关**（通信操作与 batch_size 无关） |
| `CAST_EMA_FP32_TO_FP16` 分类 | LRF1/2/3 未列出 | **MYP1 归入 shape 无关**（cast 操作与 batch 无关） |
| 渐进式分辨率 epoch-level 变体选择 | LRF1/2 使用 `get_current_train_resolution()` vs `train_res_begin` 比较 | **MYP1 同样采用**，并验证了 `run_gpu()` 中已有 `set_current_resolution_train()` |