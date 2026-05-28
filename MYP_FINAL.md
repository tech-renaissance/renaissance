# MYP_FINAL — Last Batch 与不同 Shape 图捕获/调用的终极完备方案

> **综合 MYP1 / MYP2 / MYP3 三份方案 + 源码全量验证 + 修正差异，给出唯一权威最终方案**
> **版本**: FINAL | **日期**: 2026-05-28
> **目标**: 完美支持 last batch + 渐进式分辨率 + ShapeId 去重 + batch 循环内零开销

---

## 一、MYP1/MYP2/MYP3 关键差异与修正

### 1.1 三大分歧点与源码裁决

| 分歧点 | MYP1 | MYP2 | MYP3 | **源码裁决** |
|--------|------|------|------|-------------|
| Variant 命名 (res_begin ↔ res_end) | v0=res_begin, v2=res_end ✅ | v0=res_begin, v2=res_end ✅ | v0=res_end, v2=res_begin ❌ | **`compile_spec.h:L28-L33`** 注释明确：variant 0 = train_res_begin，variant 2 = train_res_end |
| Graph 分类辅助函数 | 自定 lambda | 复用 `computation_graph.h` 已有函数 ✅ | 自定 namespace 函数 | **`computation_graph.h:L149-L213`** 已定义 `is_shape_invariant_graph()` / `is_train_graph()` / `is_inference_graph()`，已含 `STATS_COMM` 和 `CAST_EMA_FP32_TO_FP16` |
| `compiled_result_` 成员 | 讨论后认为不需要 ✅ | 明确不添加 ✅ | 添加 ❌ | **`on_prepare()` 已将 `train_cg`/`infer_cg` move 入 `named_graphs_`，`MemoryPlan` move 入 `variant_memory_plans_`** — `compiled_result_` 无持有必要 |
| 渐进分辨率选择 | `current_resolution_train()` ✅ | `current_resolution_train()` ✅ | `current_epoch_ < boundary_epoch()` ❌ | **`run_gpu():L760-L790`** 每 epoch 通过 `set_current_resolution_train()` 设置动态分辨率（支持 gradual crop/resize 渐变），必须用 `registry.current_resolution_train()`（`global_registry.h:L758`）读取 |
| ACCUM_METRICS 去重 | 三者均 shape-invariant | 三者 shape-invariant + 三个不同 GraphId（彼此独立）✅ | 三者均 shape-invariant | **`computation_graph.h:L96-L98`** 三个 GraphId 各不相同，`Key{cg, gid, shape_id}` 中 `gid` 不同 → 三者互不共享，各自在 6 variant 内共享 |

### 1.2 各文档优势吸收

| 吸收点 | 来源 | 说明 |
|--------|------|------|
| ShapeId 去重机制完整分析 | MYP1 | 从 `captured_graph.cpp` 验证了 `Key{cg, gid, shape_id}` 的完整逻辑 |
| 复用已有辅助函数 | MYP2 | `computation_graph.h` 的 `is_shape_invariant_graph()` 等已枚举完整 |
| 交叉槽位隔离逻辑 | MYP2 | Train variant 不填 inference 图，Val variant 不填 train 图 |
| CPU 标量显式写入 | MYP2 | GPU 路径 GraphNode 自动绑定 `batch_size_ptr`，CPU 路径需手动写入 |
| 三层断裂诊断框架 | MYP3 | 清晰呈现 Compiler → GraphAtlas → Runtime 断裂链 |
| `compile_spec.h` 官方注释六变体 | MYP1 | 严格遵循 L28-L33 官方文档 |
| `get_shape_id()` 作为去重键来源 | MYP1 | `CompileSpec::get_shape_id()` 直接从 `batch_size`、`actual_resolution`、`num_color_channels` 计算 |

---

## 二、问题根因精确定位

### 2.1 三层系统性断裂（源码验证）

| 层级 | 设计意图 | 当前实现 | 断裂点 |
|------|---------|---------|--------|
| **Compiler** | 6 变体编译，各含独立 MemoryPlan + ShapeId | `on_prepare()`（`deep_learning_task.h:L287-L288`）只传 1 个 `base_spec`，`variant_specs` 为空 → `result.variants.size() == 1` | 只产 1 个 Variant |
| **GraphAtlas** | 6×28 映射表，shape 相关图各自 `mp` + `shape_id` | `build_graph_atlas()`（`deep_learning_task.cpp:L490-L541`）只填 variant 0，所有 slot 硬编码 `shape_id = kShapeInvariant` | pre_capture 去重时所有变体 Key 相同 |
| **运行时** | 根据 batch 类型选 correct variant | `build_exec_table()`（L598）只 `atlas.index(0, gid)`；`run_train_epoch_gpu()`（L913）所有 batch 同一 `g_tab = gpu_exec_.graphs[rank]` | last batch 使用 grid=standard_bs 的 graph |

### 2.2 CUDA Graph 物理约束（不可违背）

CUDA Graph 的 kernel grid 维度在 **capture 时固定**，replay 时不可变。kernel 内部从 MemoryPlan 的 DTensor shape 读取 batch 维度（如 `logits_dt.shape.n()`），该值在 capture 时写入 graph。即使运行时修改标量 `batch_size_ptr`，grid 维度不变。

**后果**: 如果 capture 时 batch=512，last batch 402 个样本用同一 graph → grid 仍是 512 → 后 110 个 block 处理残留/越界数据 → 污染 loss/top1/梯度。

**唯一正确做法**: 为每个唯一的 `ShapeId{batch_size, resolution, resolution, C}` 单独 capture CUDA Graph。

---

## 三、6-Variant 精确定义（遵循 `compile_spec.h` 官方注释）

### 3.1 Variant 映射表

根据 `compile_spec.h:L28-L33` 官方注释 + `CompileSpec::from_global_registry()`（L49-L58）实际行为：

| Variant | 名称 | batch_size | actual_resolution | CG 来源 | ShapeId 来源 |
|---------|------|-----------|-------------------|---------|-------------|
| 0 | `train_base` | `get_local_batch_size()` | `train_sample_resolution_begin()` | `train_cg` | `CompileSpec::get_shape_id()` |
| 1 | `train_last` | `get_last_train_batch_size()` | `train_sample_resolution_begin()` | `train_cg` | 同上 |
| 2 | `train_lowres` | `get_local_batch_size()` | `train_sample_resolution_end()` | `train_cg` | 同上 |
| 3 | `train_lowres_last` | `get_last_train_batch_size()` | `train_sample_resolution_end()` | `train_cg` | 同上 |
| 4 | `val_base` | `get_local_batch_size()` | `val_sample_resolution()` | `infer_cg` | 同上 |
| 5 | `val_last` | `get_last_val_batch_size()` | `val_sample_resolution()` | `infer_cg` | 同上 |

### 3.2 ShapeId 公式（来自 `compile_spec.h:L66-L69`）

```cpp
ShapeId{ batch_size, actual_resolution, actual_resolution, num_color_channels }
```

### 3.3 各 Variant 的 ShapeId 示例

**CIFAR-10**（batch=200, res=32, C=3, 整除且单分辨率）：
- 全部 6 variant ShapeId = `{200, 32, 32, 3}` → **自然去重为 1 个唯一 ShapeId**

**ImageNet 渐进分辨率**（8 ranks, batch=512, last_train=402, last_val=106, res_begin=128, res_end=224, C=3）：
- v0: `{512, 224, 224, 3}`
- v1: `{402, 224, 224, 3}`
- v2: `{512, 128, 128, 3}`
- v3: `{402, 128, 128, 3}`
- v4: `{512, 224, 224, 3}` → **与 v0 同 ShapeId，自动复用**
- v5: `{106, 224, 224, 3}`
- **5 个唯一 ShapeId**

---

## 四、Graph 分类与去重机制

### 4.1 直接复用 `computation_graph.h` 已有函数

源码（`computation_graph.h:L149-L213`）已定义三个判定函数，**不重新定义**：

- `is_shape_invariant_graph(gid)` — 含 TRANSFER_A/B, ZERO_GRAD, COMM×3, CAST_×4, NAN_CHECK, EMA_UPDATE, ACCUM_METRICS×3, VAL_RESULT_COMM, CLEAR_METRICS
- `is_train_graph(gid)` — 含 TRANSFER_A/B, FWD_A/B, DEEP_FWD_BWD, ZERO_GRAD, BWD_A/B, CAST_×3, COMM×3, EMA_UPDATE, ACCUM_METRICS, ACCUM_METRICS_TRAIN_LAST, CLEAR_METRICS
- `is_inference_graph(gid)` — 含 INF_MAIN_A/B, INF_EMA_A/B

**注意**: `OPTIMIZER` **不在** `is_train_graph()` 中（源码注释：LARS 下节点数依赖层数）。

### 4.2 三类 Graph 与去重行为

| 类型 | GraphId | 行为 |
|------|---------|------|
| **完全 Shape 无关** | `is_shape_invariant_graph()` 返回 true 的全部 GraphId | 6 variant 共享 `kShapeInvariant` → pre_capture 必然碰撞 → 全局复用 **1 张图** |
| **训练 Shape 相关** | FIRST_LAYER_FWD_A/B, DEEP_FWD_BWD, FIRST_LAYER_BWD_A/B | 按各自 variant 的 ShapeId 去重捕获，最多 4 套 |
| **推理 Shape 相关** | INF_MAIN_A/B, INF_EMA_A/B | 按各自 variant 的 ShapeId 去重捕获，最多 2 套 |
| **训练专属非 Shape 相关** | OPTIMIZER | `is_train_graph()` 返回 false（LARS 下节点数依赖层数），`is_shape_invariant_graph()` 也返回 false。不被交叉隔离跳过 → 每个 variant 独立捕获，但拓扑相同 |

### 4.3 ACCUM_METRICS 系列特别说明

`ACCUM_METRICS`、`ACCUM_METRICS_TRAIN_LAST`、`ACCUM_METRICS_VAL_LAST` 是 **三个不同的 GraphId**。虽然每个都是 shape-invariant（kernel 内通过 `batch_size_ptr` 动态读取 batch），但 `pre_capture()` 的 `Key{cg, gid, shape_id}` 中 `gid` 不同 → **三者互不共享**，各自在 6 variant 内共享 1 张图。这是正确的设计：normal batch 累积中除以 `local_batch_size`，last batch 累积中除以 `last_batch_size`。

### 4.4 去重后 Captured Graph 数量估算

| 场景 | shape-invariant 图 | 训练 shape 相关 | 推理 shape 相关 | OPTIMIZER | 总计 |
|------|-------------------|----------------|----------------|-----------|------|
| CIFAR-10 (整除, 单分辨率) | ~17 张 | ~5 GraphId × 1 ShapeId = 5 | ~4 GraphId × 1 ShapeId = 4 | 1 GraphId × 1 ShapeId = 1 | **~27 张** |
| ImageNet (不整除, 渐进分辨率) | ~17 张 | ~5 GraphId × 4 ShapeId = 20 | ~4 GraphId × 2 ShapeId = 8 | 1 GraphId × 4 ShapeId = 4 | **~49 张** |

---

## 五、完整实施方案

### 改动范围

| 文件 | 改动点 | 行数估算 |
|------|--------|---------|
| `include/renaissance/task/deep_learning_task.h` | 新增 `variant_memory_plans_`、`variant_compile_specs_`；扩展 `GpuExecTable`；修改 `on_prepare()` | ~50 |
| `src/task/deep_learning_task.cpp` | 重写 `build_graph_atlas()`、`build_exec_table()`、`run_train_epoch_gpu()`、`run_val_epoch_gpu()`、`run_train_epoch_cpu()`、`run_val_epoch_cpu()` | ~350 |
| **总计** | **2 文件** | **~400 行** |

**不修改**: `compiler.cpp`、`captured_graph.cpp`、`compile_spec.h`、`computation_graph.h` 及所有 kernel 文件。

---

### Step 0: 数据结构准备（`deep_learning_task.h`）

在 `private:` 区域 `memory_plan_ptr_` 下方新增：

```cpp
// ===== 6-Variant 系统 =====
/// 各变体的独立 MemoryPlan（slot_bytes 跨变体一致，shape 不同）
std::array<std::unique_ptr<MemoryPlan>, GraphAtlas::kMaxVariants> variant_memory_plans_;

/// 各变体的 CompileSpec（用于 build_graph_atlas 获取 ShapeId）
std::array<CompileSpec, GraphAtlas::kMaxVariants> variant_compile_specs_;
```

修改 `GpuExecTable`：

```cpp
#ifdef TR_USE_CUDA
struct GpuExecTable {
    // variant_graphs[variant_idx][rank][slot] → cudaGraphExec_t
    std::array<std::vector<std::vector<cudaGraphExec_t>>, GraphAtlas::kMaxVariants> variant_graphs;
    std::vector<int> device_ids;
};
GpuExecTable gpu_exec_;
std::vector<float*> lr_pinned_;
#endif
```

> **不添加 `compiled_result_`**: `Compiler::Result::train_cg` / `infer_cg` 已通过 `add_graph()` move 入 `named_graphs_`（`deep_learning_task.h:L343-L344`），生命周期由 `named_graphs_` 保证。`variant_memory_plans_` 已通过 `std::move` 取得各 Variant 的 MemoryPlan 所有权。

---

### Step 1: `on_prepare()` 生成完整 6-Variant（`deep_learning_task.h`）

替换 `on_prepare()` 中 L287-L288 的单 spec 调用：

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

// ===== 保存 CompileSpec（用于 build_graph_atlas 获取 ShapeId）=====
variant_compile_specs_[0] = base_spec;
for (size_t i = 0; i < variant_specs.size() && i < GraphAtlas::kMaxVariants - 1; ++i) {
    variant_compile_specs_[i + 1] = variant_specs[i];
}

auto result = Compiler::compile(plan, base_spec, plan_config_,
                                initializer_, variant_specs);

// ===== 转移所有 variant 的 MemoryPlan =====
for (size_t i = 0; i < result.variants.size() && i < GraphAtlas::kMaxVariants; ++i) {
    variant_memory_plans_[i] = std::move(result.variants[i].memory_plan);
}

// ===== 保持向后兼容 =====
memory_plan_ptr_ = std::move(variant_memory_plans_[0]);
if (!memory_plan_ptr_->is_finalized()) {
    memory_plan_ptr_->finalize();
}
active_memory_plan_ = memory_plan_ptr_.get();
phase_ = Phase::MEMORY_LOCKED;

// 后续代码不变（lr_dtensor_id_ 查找、标量初始化、add_graph 等）
// add_graph("train", ...) 和 add_graph("inference", ...) 保持不变
```

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

    // ===== 全量填表：6 variant × 所有 GraphId =====
    for (size_t v = 0; v < GraphAtlas::kMaxVariants; ++v) {
        const MemoryPlan* mp = variant_memory_plans_[v].get();
        if (!mp) continue;

        bool is_train_var = (v <= 3);
        ShapeId variant_shape = variant_compile_specs_[v].get_shape_id();

        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);

            // ===== 交叉槽位隔离 =====
            if (is_train_var) {
                // Train variant: 不需要纯推理图
                if (is_inference_graph(gid) && !is_shape_invariant_graph(gid)) continue;
            } else {
                // Val variant: 不需要 train-only shape 相关图
                if (is_train_graph(gid) && !is_inference_graph(gid) &&
                    !is_shape_invariant_graph(gid)) continue;
            }

            // ===== 选择 ComputationGraph =====
            const ComputationGraph* cg = is_inference_graph(gid) ? infer_cg_ : train_cg_;
            if (!cg || cg->nodes(gid).empty()) continue;

            auto& sl = atlas.slot(v, gi);
            sl.cg = cg;
            sl.stream_kind = stream_for(gid);

            if (is_shape_invariant_graph(gid)) {
                sl.mp = active_memory_plan_;
                sl.shape_id = kShapeInvariant;
            } else {
                sl.mp = mp;
                sl.shape_id = variant_shape;
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
        gpu_exec_.variant_graphs[0].resize(K);
        for (int rank = 0; rank < K; ++rank) {
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
            g[S(GraphSlot::CLEAR_METRICS)]    = resolve(GraphId::CLEAR_METRICS, rank, v);

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
            } else {
                g[S(GraphSlot::INF_MAIN_A)]       = resolve(GraphId::INF_MAIN_A, rank, v);
                g[S(GraphSlot::INF_MAIN_B)]       = resolve(GraphId::INF_MAIN_B, rank, v);
                g[S(GraphSlot::INF_EMA_A)]        = resolve(GraphId::INF_EMA_A, rank, v);
                g[S(GraphSlot::INF_EMA_B)]        = resolve(GraphId::INF_EMA_B, rank, v);
                g[S(GraphSlot::ACCUM_METRICS)]           = resolve(GraphId::ACCUM_METRICS, rank, v);
                g[S(GraphSlot::ACCUM_METRICS_VAL_LAST)]   = resolve(GraphId::ACCUM_METRICS_VAL_LAST, rank, v);
                g[S(GraphSlot::VAL_RESULT_ALLREDUCE)]     = resolve(GraphId::VAL_RESULT_COMM, rank, v);
            }
        }
    }
#endif
}
```

---

### Step 4: `run_train_epoch_gpu()` — Variant 切换 + 预解析（`deep_learning_task.cpp`）

**核心原则**: 在 batch 循环外部完成 **normal batch 和 last batch 两组** graph handle 的预解析。

```cpp
#ifdef TR_USE_CUDA
float DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    // ===== Epoch-level variant 选择 =====
    int current_res = registry.current_resolution_train();
    bool at_begin_res = (current_res == registry.train_sample_resolution_begin());
    size_t v_base = at_begin_res ? 0 : 2;
    size_t v_last = at_begin_res ? 1 : 3;

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    int32_t loss_id = active_memory_plan_->baseline().loss;

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, K, using_amp, &exc,
                              loss_id, v_base, v_last]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // ============================================================
                // PRE-RESOLVE: normal batch handles (v_base)
                // ============================================================
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

                // ============================================================
                // PRE-RESOLVE: last batch handles (v_last)
                // shape-invariant 图与 normal batch 指向同一 captured graph
                // shape-dependent 图使用不同的 captured graph
                // ============================================================
                const auto& g_l = gpu_exec_.variant_graphs[v_last][rank];
                auto l_deep_a  = g_l[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto l_deep_b  = g_l[S(GraphSlot::FWD_BWD_DEEP_B)];
                auto l_fwd_a   = g_l[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto l_fwd_b   = g_l[S(GraphSlot::FIRST_LAYER_FWD_B)];
                auto l_bwd_a   = g_l[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto l_bwd_b   = g_l[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto l_accum_tl = g_l[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up    = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_c3    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_up = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr = [&]() { cudaStreamSynchronize(s_trans); };

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

                // Batch 0 预传输
                ts->wait_buffer_readable(0);
                if (n_xfer_a) cudaGraphLaunch(n_xfer_a, s_trans);
                sync_tr();
                ts->set_buffer_readable(0, false);
                ts->set_buffer_writeable(0, true);
                if (using_amp && n_cm) { cudaGraphLaunch(n_cm, s_up); sync_up(); }
                if (n_clear) cudaGraphLaunch(n_clear, s_up);

                // ===== Normal batches (0 .. batches-2) =====
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
                    (void)lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (n_wu) cudaGraphLaunch(n_wu, s_up); sync_up();
                    if (using_amp && n_cm) { cudaGraphLaunch(n_cm, s_up); sync_up(); }
                }

                // ===== Last batch (batches-1) — 使用 l_* handles =====
                {
                    bool last_a = ((batches - 1) % 2 == 0);
                    auto g_fwd_l  = last_a ? l_fwd_a : l_fwd_b;
                    auto g_deep_l = last_a ? l_deep_a : l_deep_b;
                    auto g_first_l = last_a ? l_bwd_a : l_bwd_b;

                    if (n_zg) cudaGraphLaunch(n_zg, s_up);
                    if (g_fwd_l) cudaGraphLaunch(g_fwd_l, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
                    sync_comp();

                    if (l_accum_tl) cudaGraphLaunch(l_accum_tl, s_up);
                    sync_up();

                    if (!frozen && g_first_l) cudaGraphLaunch(g_first_l, s_c1);
                    sync_comp();

                    if (using_amp && n_cdg) { cudaGraphLaunch(n_cdg, s_up); sync_up(); }
                    if (n_dar) cudaGraphLaunch(n_dar, s_up); sync_up();
                    if (using_amp && n_cfg) { cudaGraphLaunch(n_cfg, s_up); sync_up(); }
                    if (n_far) cudaGraphLaunch(n_far, s_up); sync_up();
                    if (using_amp && n_ncg) { cudaGraphLaunch(n_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batches - 1);
                    (void)lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (n_wu) cudaGraphLaunch(n_wu, s_up); sync_up();
                    if (using_amp && n_cm) { cudaGraphLaunch(n_cm, s_up); sync_up(); }
                }

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank)
        if (exc[rank]) std::rethrow_exception(exc[rank]);

    // 从 accum_loss 读取 epoch 平均 loss
    float train_loss = 0.0f;
    if (active_memory_plan_) {
        const auto& b = active_memory_plan_->baseline();
        int32_t accum_loss_id = b.accum_loss;
        if (accum_loss_id >= 0) {
            const auto& accum_dt = active_memory_plan_->get_dtensor(accum_loss_id);
            Tensor h_accum = fetch_from_rank(accum_dt, 0);
            float val = h_accum.data<float>()[0];
            size_t total = registry.num_train_samples();
            if (total > 0) train_loss = val / static_cast<float>(total);
        }
    }
    return train_loss;
}
#endif
```

---

### Step 5: `run_val_epoch_gpu()` — Variant 切换（`deep_learning_task.cpp`）

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

                const auto& g_vb = gpu_exec_.variant_graphs[4][rank];
                const auto& g_vl = gpu_exec_.variant_graphs[5][rank];

                auto vb_xfer_a  = g_vb[S(GraphSlot::XFER_A)];
                auto vb_xfer_b  = g_vb[S(GraphSlot::XFER_B)];
                auto vb_inf_a   = g_vb[S(GraphSlot::INF_MAIN_A)];
                auto vb_inf_b   = g_vb[S(GraphSlot::INF_MAIN_B)];
                auto vb_accum   = g_vb[S(GraphSlot::ACCUM_METRICS)];
                auto vb_clear   = g_vb[S(GraphSlot::CLEAR_METRICS)];
                auto vb_comm    = g_vb[S(GraphSlot::VAL_RESULT_ALLREDUCE)];

                auto vl_inf_a   = g_vl[S(GraphSlot::INF_MAIN_A)];
                auto vl_inf_b   = g_vl[S(GraphSlot::INF_MAIN_B)];
                auto vl_accum   = g_vl[S(GraphSlot::ACCUM_METRICS_VAL_LAST)];

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up    = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_c3    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_up = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr = [&]() { cudaStreamSynchronize(s_trans); };

                TransferStation* ts = nullptr;
                for (int w = 0; w < 200; ++w) {
                    ts = static_cast<TransferStation*>(
                        registry.transfer_station_ptr(rank));
                    if (ts) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

                const auto& b = active_memory_plan_->baseline();
                int32_t loss_id = b.loss, top1_id = b.top1, top5_id = b.top5;

                if (vb_clear) cudaGraphLaunch(vb_clear, s_up);

                for (int batch = 0; batch < val_batches; ++batch) {
                    int buf = batch % 2;
                    bool is_last = (batch == val_batches - 1);

                    auto g_xfer = (buf == 0) ? vb_xfer_a : vb_xfer_b;
                    auto g_inf  = (buf == 0)
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
                if (vb_comm) cudaGraphLaunch(vb_comm, s_up);
                sync_up();

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank)
        if (exc[rank]) std::rethrow_exception(exc[rank]);

    float avg_loss = 0.0f, avg_top1 = 0.0f, avg_top5 = 0.0f;
    if (active_memory_plan_) {
        const auto& b = active_memory_plan_->baseline();
        size_t n = registry.num_val_samples();
        if (b.accum_loss >= 0 && n > 0) {
            Tensor h = fetch_from_rank(active_memory_plan_->get_dtensor(b.accum_loss), 0);
            avg_loss = h.data<float>()[0] / static_cast<float>(n);
        }
        if (b.accum_top1 >= 0 && n > 0) {
            Tensor h = fetch_from_rank(active_memory_plan_->get_dtensor(b.accum_top1), 0);
            avg_top1 = h.data<float>()[0] / static_cast<float>(n);
        }
        if (b.accum_top5 >= 0 && n > 0) {
            Tensor h = fetch_from_rank(active_memory_plan_->get_dtensor(b.accum_top5), 0);
            avg_top5 = h.data<float>()[0] / static_cast<float>(n);
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

### Step 6: `run_train_epoch_cpu()` — Variant 支持 + 指标修复 + 标量写入（`deep_learning_task.cpp`）

**修复的 Bug**:
- **B1**: `atlas.index(0, gid)` → `atlas.index(v_base/v_last, gid)` — variant 切换
- **B3**: 从 `accum_loss` 读取 epoch 平均 loss（原只读 last batch 的 `loss_id`）
- **B5**: 缺少 ACCUM_METRICS / ACCUM_METRICS_TRAIN_LAST 调用 — accum_loss 永远为 0
- **B6**: 缺少 batch_size 标量写入 — CPU 路径 `launch()` 无自动 DTensor 绑定机制，ACCUM_METRICS kernel 读取 `b.local_batch_size` 的初始值会导致 NaN

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

    int current_res = registry.current_resolution_train();
    bool at_begin_res = (current_res == registry.train_sample_resolution_begin());
    size_t v_base = at_begin_res ? 0 : 2;
    size_t v_last = at_begin_res ? 1 : 3;

    auto idx_for = [&atlas](GraphId gid, size_t v) -> int32_t {
        return atlas.index(v, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    // Shape-invariant (variant 0 即可，去重后相同)
    int32_t idx_xfer_a = idx_for(GraphId::TRANSFER_A, 0);
    int32_t idx_xfer_b = idx_for(GraphId::TRANSFER_B, 0);
    int32_t idx_zg     = idx_for(GraphId::ZERO_GRAD, 0);
    int32_t idx_dar    = idx_for(GraphId::DEEP_COMM, 0);
    int32_t idx_far    = idx_for(GraphId::FIRST_COMM, 0);
    int32_t idx_opt    = idx_for(GraphId::OPTIMIZER, 0);
    int32_t idx_clear  = idx_for(GraphId::CLEAR_METRICS, 0);

    // Shape-dependent: normal batch
    int32_t idx_deep_nb  = idx_for(GraphId::DEEP_FWD_BWD, v_base);
    int32_t idx_fwd_a_nb = idx_for(GraphId::FIRST_LAYER_FWD_A, v_base);
    int32_t idx_fwd_b_nb = idx_for(GraphId::FIRST_LAYER_FWD_B, v_base);
    int32_t idx_bwd_a_nb = idx_for(GraphId::FIRST_LAYER_BWD_A, v_base);
    int32_t idx_bwd_b_nb = idx_for(GraphId::FIRST_LAYER_BWD_B, v_base);

    // Shape-dependent: last batch
    int32_t idx_deep_lb  = idx_for(GraphId::DEEP_FWD_BWD, v_last);
    int32_t idx_fwd_a_lb = idx_for(GraphId::FIRST_LAYER_FWD_A, v_last);
    int32_t idx_fwd_b_lb = idx_for(GraphId::FIRST_LAYER_FWD_B, v_last);
    int32_t idx_bwd_a_lb = idx_for(GraphId::FIRST_LAYER_BWD_A, v_last);
    int32_t idx_bwd_b_lb = idx_for(GraphId::FIRST_LAYER_BWD_B, v_last);

    const auto& bl = active_memory_plan_->baseline();
    int32_t loss_id = bl.loss;
    void* loss_ptr = loss_id >= 0 ? ctx.ptr_at(loss_id) : nullptr;
    void* lr_ptr   = ctx.ptr_at(lr_dtensor_id_);
    // 标量指针：ACCUM_METRICS kernel 通过 DTensor 地址读取 batch_size
    void* local_bs_ptr  = bl.local_batch_size >= 0 ? ctx.ptr_at(bl.local_batch_size) : nullptr;
    void* last_bs_ptr   = bl.last_train_batch_size >= 0 ? ctx.ptr_at(bl.last_train_batch_size) : nullptr;
    bool frozen = is_first_layer_frozen();

    if (idx_clear >= 0) launch(idx_clear);

    // Batch 0 prefetch
    ts->wait_buffer_readable(0);
    launch(idx_xfer_a);
    ts->set_buffer_readable(0, false);
    ts->set_buffer_writeable(0, true);

    // Normal batches
    for (int batch = 0; batch < batches - 1; ++batch) {
        bool from_a  = (batch % 2 == 0);
        int next_buf = from_a ? 1 : 0;

        launch(idx_zg);
        launch(from_a ? idx_fwd_a_nb : idx_fwd_b_nb);

        ts->wait_buffer_readable(next_buf);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_nb);
        launch(from_a ? idx_xfer_b : idx_xfer_a);

        ts->set_buffer_readable(next_buf, false);
        ts->set_buffer_writeable(next_buf, true);

        if (!frozen) launch(from_a ? idx_bwd_a_nb : idx_bwd_b_nb);
        launch(idx_dar);

        // 写入 local_batch_size 标量，然后发射 ACCUM_METRICS
        if (local_bs_ptr) *static_cast<int32_t*>(local_bs_ptr) = registry.get_local_batch_size();
        int32_t idx_accum_nb = idx_for(GraphId::ACCUM_METRICS, v_base);
        if (idx_accum_nb >= 0) launch(idx_accum_nb);

        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batch);
        launch(idx_far);
        launch(idx_opt);
    }

    // Last batch — 使用 v_last indices
    {
        bool last_a = ((batches - 1) % 2 == 0);

        launch(idx_zg);
        launch(last_a ? idx_fwd_a_lb : idx_fwd_b_lb);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_lb);

        if (!frozen) launch(last_a ? idx_bwd_a_lb : idx_bwd_b_lb);
        launch(idx_dar);

        // 写入 last_train_batch_size 标量，然后发射 ACCUM_METRICS_TRAIN_LAST
        if (last_bs_ptr) *static_cast<int32_t*>(last_bs_ptr) = registry.get_last_train_batch_size();
        int32_t idx_accum_lb = idx_for(GraphId::ACCUM_METRICS_TRAIN_LAST, v_last);
        if (idx_accum_lb >= 0) launch(idx_accum_lb);

        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batches - 1);
        launch(idx_far);
        launch(idx_opt);
    }

    float train_loss = 0.0f;
    if (bl.accum_loss >= 0) {
        Tensor h = fetch_from_rank(active_memory_plan_->get_dtensor(bl.accum_loss), 0);
        float val = h.data<float>()[0];
        size_t total = registry.num_train_samples();
        if (total > 0) train_loss = val / static_cast<float>(total);
    }
    return train_loss;
}
```

---

### Step 7: `run_val_epoch_cpu()` — Variant 支持 + 加权平均（`deep_learning_task.cpp`）

**修复的 Bug**:
- **B2**: `atlas.index(0, gid)` → `atlas.index(4/5, gid)` — variant 切换
- **B4**: 简单平均 `acc_loss / val_batches` → 加权平均 `sum(batch_loss × batch_size) / total_val`

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

    int32_t idx_xfer_a = idx_for(GraphId::TRANSFER_A, 4);
    int32_t idx_xfer_b = idx_for(GraphId::TRANSFER_B, 4);
    int32_t idx_inf_a_vb = idx_for(GraphId::INF_MAIN_A, 4);
    int32_t idx_inf_b_vb = idx_for(GraphId::INF_MAIN_B, 4);
    int32_t idx_inf_a_vl = idx_for(GraphId::INF_MAIN_A, 5);
    int32_t idx_inf_b_vl = idx_for(GraphId::INF_MAIN_B, 5);

    const auto& bl = active_memory_plan_->baseline();
    int32_t loss_id = bl.loss, top1_id = bl.top1, top5_id = bl.top5;

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

    double acc_loss = 0.0, acc_top1 = 0.0, acc_top5 = 0.0;

    for (int batch = 0; batch < val_batches; ++batch) {
        int buf = batch % 2;
        bool is_last = (batch == val_batches - 1);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        if (top1_ptr) std::memset(top1_ptr, 0, sizeof(float));
        if (top5_ptr) std::memset(top5_ptr, 0, sizeof(float));

        ts->wait_buffer_readable(buf);

        launch(buf == 0 ? idx_xfer_a : idx_xfer_b);
        launch(buf == 0
            ? (is_last ? idx_inf_a_vl : idx_inf_a_vb)
            : (is_last ? idx_inf_b_vl : idx_inf_b_vb));

        int bs = is_last ? registry.get_last_val_batch_size()
                         : registry.get_local_batch_size();
        float b_loss = loss_ptr ? *static_cast<float*>(loss_ptr) : 0.0f;
        float b_top1 = top1_ptr ? *static_cast<float*>(top1_ptr) : 0.0f;
        float b_top5 = top5_ptr ? *static_cast<float*>(top5_ptr) : 0.0f;

        acc_loss += b_loss * static_cast<double>(bs);
        acc_top1 += b_top1 * static_cast<double>(bs);
        acc_top5 += b_top5 * static_cast<double>(bs);

        ts->set_buffer_readable(buf, false);
        ts->set_buffer_writeable(buf, true);
    }

    size_t total_val = registry.num_val_samples();
    float avg_loss = (total_val > 0) ? static_cast<float>(acc_loss / static_cast<double>(total_val)) : 0.0f;
    float avg_top1 = (total_val > 0) ? static_cast<float>(acc_top1 / static_cast<double>(total_val)) : 0.0f;
    float avg_top5 = (total_val > 0) ? static_cast<float>(acc_top5 / static_cast<double>(total_val)) : 0.0f;

    return {avg_loss, avg_top1, avg_top5};
}
```

---

## 六、渐进式分辨率完整支持

### 6.1 运行时选择逻辑

`run_gpu()`（`deep_learning_task.cpp:L760-L790`）每 epoch 开头调用 `set_current_resolution_train()`，支持 gradual crop/resize 渐变。因此 variant 选择应使用 `registry.current_resolution_train()` 运行时读取：

```cpp
int current_res = registry.current_resolution_train();
bool at_begin_res = (current_res == registry.train_sample_resolution_begin());
size_t v_base = at_begin_res ? 0 : 2;
size_t v_last = at_begin_res ? 1 : 3;
```

| Epoch 阶段 | `current_resolution_train()` | v_base | v_last |
|-----------|----------------------------------|--------|--------|
| 早期 (< boundary) | `train_sample_resolution_begin()` | 0 | 1 |
| 后期 (≥ boundary) | `train_sample_resolution_end()` | 2 | 3 |
| 渐进渐变 (crop/resize) | 每 epoch 线性插值 | 根据返回值判断 | 根据返回值判断 |

验证阶段始终使用 v4/v5，分辨率固定为 `val_sample_resolution()`。

### 6.2 ShapeId 去重的自然收敛

- 单分辨率（begin == end）：v0=v2, v1=v3 → 训练只有 2 个唯一 ShapeId
- 双分辨率（begin != end）：v0≠v2, v1≠v3 → 训练有 4 个唯一 ShapeId
- ImageNet 场景：v0 和 v4 同 ShapeId → train_base + val_base 共享 shape 相关图

---

## 七、性能保证与开销分析

### 7.1 Batch 循环内零开销

| 指标 | 当前实现 | 本方案 | 说明 |
|------|---------|--------|------|
| Graph handle 解析 | `g_tab[S(slot)]` 数组索引 | **直接局部变量** | 编译器可将局部变量放入寄存器 |
| Last batch 分支 | 无（错误地同 graph） | **独立代码块 + 独立局部变量组** | 编译期完全确定，零运行时分支 |
| Variant 切换 | N/A（无 variant） | **循环外预解析** | 零开销 |
| 多 RANK 扩展 | `gpu_exec_.graphs[rank]` | `gpu_exec_.variant_graphs[v][rank]` | 每 rank 独立预解析 |

### 7.2 内存/编译时间

| 场景 | 当前 | 本方案 |
|------|------|--------|
| CIFAR-10 (整除, 单分辨率) | ~23 张图 | **~27 张图** (ShapeId 退化 → 零额外开销) |
| ImageNet (不整除, 渐进分辨率) | ~23 张图 | **~53 张图** (按 5 个唯一 ShapeId 捕获) |
| 编译时间增加 | — | **< 2s** (cuDNN warmup 线性增长) |

### 7.3 shape-invariant 图的零膨胀

`is_shape_invariant_graph()` 返回 true 的全部 GraphId（~17 个），6 variant 中全部使用 `kShapeInvariant` 作为 shape_id → `pre_capture()` 的 `Key{cg, gid, ShapeId{0,0,0,0}}` 完全相同 → 全局只捕获 **1 套**，无论 variant 数量。

---

## 八、CPU 路径完整修复清单

| Bug | 当前行为 | 修复 |
|-----|---------|------|
| **B1**: train 无 variant | `atlas.index(0, gid)` — 只查 variant 0 | → `atlas.index(v_base/v_last, gid)` |
| **B2**: val 无 variant | `atlas.index(0, gid)` — 只查 variant 0 | → `atlas.index(4/5, gid)` |
| **B3**: train loss 只读 last batch | `return h_loss.data<float>()[0]` — 仅最后 batch 的 loss | → 从 `accum_loss` 读取并除以 `num_train_samples()` |
| **B4**: val 指标简单平均 | `acc_loss / val_batches` — 忽略 last batch 样本数少 | → 加权平均 `sum(b × bs) / total_val` |
| **B5**: 缺少 ACCUM_METRICS 调用 | CPU 路径未发射 ACCUM_METRICS/ACCUM_METRICS_TRAIN_LAST | → 在 BWD+COMM 后发射 ACCUM_METRICS（norm batch）或 ACCUM_METRICS_TRAIN_LAST（last batch） |
| **B6**: 缺少 batch_size 标量写入 | CPU 路径 `launch()` 无 CUDA Graph 自动绑定机制 | → launch 前写入 `local_batch_size` 或 `last_train_batch_size` 到 MemoryPlan baseline 的对应 DTensor |

---

## 九、验证策略

### Phase 1: 编译期验证
- `result.variants.size() == 6`
- 各 variant MemoryPlan 非空，`slot_bytes` 跨变体一致
- `pre_capture` 日志：shape-invariant 图 6 variant `captured_idx` 相同

### Phase 2: CIFAR-10 回归 (整除, 单分辨率)
- `test_dl_full_gpu` / `test_dl_full_amp` 通过
- 6 variant 同 ShapeId → 退化，额外开销为零

### Phase 3: 不可整除验证
- 构造 `train=50003, val=10007, batch=200` 场景
- 验证 `last_train_batch_size=103, last_val_batch_size=7`
- 对比修复前后 loss/accuracy（预期：修复后不再被残留数据污染）

### Phase 4: 渐进式分辨率
- 启用 `progressive_resize(128, 224)` + `boundary_epoch(10)`
- 验证 epoch 0-9 使用 v0/v1，epoch 10+ 使用 v2/v3

### Phase 5: CPU 路径
- `test_dl_full --cpu` 通过
- 验证 CPU 指标与 GPU 对齐

### Phase 6: 多 RANK
- `world_size=4` 运行
- 各 rank 指标一致

---

## 十、改动文件总览

| 文件 | 行数 | 关键改动 |
|------|------|---------|
| `include/renaissance/task/deep_learning_task.h` | ~50 | `variant_memory_plans_[6]` + `variant_compile_specs_[6]`；`GpuExecTable` 改为 `variant_graphs[6]`；`on_prepare()` 传 5 variant_specs |
| `src/task/deep_learning_task.cpp` | ~350 | `build_graph_atlas()` 6变体填表；`build_exec_table()` 6变体预解析；`run_train_epoch_gpu/cpu()` variant切换+预解析；`run_val_epoch_gpu/cpu()` variant切换+指标修复 |

**总计**: 2 文件，约 400 行。

**不修改**: `compiler.cpp`、`captured_graph.cpp`、`compile_spec.h`、`computation_graph.h`、所有 kernel/op 文件。

---

## 十一、风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `variant_memory_plans_[]` slot_bytes 不一致 | 极低 | 灾难性 | `Compiler::compute_max_slot_bytes` 已跨变体取 max；可增加 DTensor offset 断言 |
| Val variant 误填 train graph | 低 | 正确性 | `is_train_graph && !is_inference_graph && !is_shape_invariant_graph` → `continue` 已保证隔离 |
| `transfer_to_rank` 使用 base `active_memory_plan_` 的 `nbytes()`，last batch 多拷贝 | 中 | 性能 | 多拷贝数据在 last FWD graph 中不会被处理（grid=last_batch_size），正确性无影响 |
| Multi-variant 编译时间增加 | 中 | 开发体验 | 预期 < 2s；可考虑并行 capture（后续优化） |

---

## 十二、核心设计决策总结

1. **严格遵循 `compile_spec.h:L28-L33`**: v0=train_base (res_begin), v2=train_lowres (res_end)
2. **复用 `computation_graph.h` 已有函数**: `is_shape_invariant_graph()` / `is_train_graph()` / `is_inference_graph()`，不重新定义
3. **不添加 `compiled_result_`**: MemoryPlan 所有权已 move 到 `variant_memory_plans_`，ComputationGraph 已 move 到 `named_graphs_`
4. **使用 `current_resolution_train()`**: 支持 progressive_crop/resize 的 gradual 渐变，而非简单 `epoch < boundary` 判断
5. **预解析所有 handles**: 在 batch 循环外将 normal batch 和 last batch 两组 `cudaGraphExec_t` / `int32_t idx` 全部预解析为局部变量
6. **ShapeId 天然去重**: 整除时 last_bs == standard_bs → v0=v1 同 ShapeId，res_begin==res_end 时 v0=v2 同 ShapeId → zero extra captures
7. **CPU/GPGU 完全对齐**: CPU 路径同步支持 variant 切换 + 指标修复
8. **不修改基础设施**: Compiler / pre_capture / kernel 已完整支持，仅 orchestration 层改动

---

**文档版本**: FINAL
**创建日期**: 2026-05-28
**基于**: MYP1.md + MYP2.md + MYP3.md (综合分析) + 框架源代码全量验证
**状态**: 待实施