# LRF1: 多形状捕获 & Last Batch & 渐进式分辨率 — 完整解决方案

> **基于 DKS_FINAL.md / 小伙伴K方案 / 用户补充 + ShapeId 去重机制源码验证**
> **版本**: v1.0 | **日期**: 2026-05-28

---

## 一、问题域建模

### 1.1 输入形状的完整维度空间

训练和验证的输入由两个独立维度的笛卡尔积构成：

```
训练空间: {res_begin, res_end} × {standard_batch, last_batch} = 最多 4 种 shape
验证空间: {val_res}           × {standard_batch, last_batch} = 最多 2 种 shape
```

| 变体索引 | 名称 | 场景 | ShapeId 四元组 |
|:--------:|------|------|---------------|
| **v0** | train_base | 训练·标准分辨率·标准batch | `{standard_bs, res_begin, res_begin, C}` |
| **v1** | train_last | 训练·标准分辨率·末batch | `{last_train_bs, res_begin, res_begin, C}` |
| **v2** | train_lowres | 训练·另一分辨率·标准batch | `{standard_bs, res_end, res_end, C}` |
| **v3** | train_lowres_last | 训练·另一分辨率·末batch | `{last_train_bs, res_end, res_end, C}` |
| **v4** | val_base | 验证·标准batch | `{standard_bs, val_res, val_res, C}` |
| **v5** | val_last | 验证·末batch | `{last_val_bs, val_res, val_res, C}` |

**关键退化**：当 `res_begin == res_end`（非渐进式）或 `standard_bs == last_bs`（整除）时，对应的 ShapeId 相同 → ShapeId 去重自动合并 → **零额外捕获开销**。

### 1.2 图分类：shape-dependent vs shape-invariant

通过对 26 个 GraphId 的逐一分析，根据是否依赖输入 shape 分为两类：

#### shape-invariant 图（统一共享，kShapeInvariant）

| GraphId | 不依赖 shape 的原因 |
|---------|-------------------|
| `TRANSFER_A`, `TRANSFER_B` | H2D 传整个 buffer，与 batch/resolution 无关 |
| `ZERO_GRAD` | 权重形状固定 |
| `DEEP_COMM`, `FIRST_COMM` | AllReduce 梯度，梯度形状 = 权重形状 |
| `CAST_DEEP_GRAD_FP16_TO_FP32` | 权重的 fp16→fp32 转换 |
| `CAST_FIRST_GRAD_FP16_TO_FP32` | 同上 |
| `NAN_CHECK_AND_GRAD_SCALING` | NaN 检查和梯度缩放，不依赖特征图 |
| `OPTIMIZER` | 权重更新，仅涉及权重大小 |
| `EMA_UPDATE` | EMA 权重平滑，仅涉及权重 |
| `CAST_MAIN_FP32_TO_FP16` | 权重的 fp32→fp16 转换 |
| `ACCUM_METRICS` | 只读 1 个标量 loss/top1/top5 |
| `ACCUM_METRICS_TRAIN_LAST` | 同上 |
| `ACCUM_METRICS_VAL_LAST` | 同上 |
| `VAL_RESULT_COMM` | AllReduce 标量 |
| `CLEAR_METRICS` | 清零标量 buffer |

**数量：15 个**，全部共享 `kShapeInvariant` → pre_capture 去重后仅捕获 **15 张图**。

#### shape-dependent 图（按 ShapeId 分别捕获）

| GraphId | 依赖 shape 的原因 |
|---------|------------------|
| `FIRST_LAYER_FWD_A`, `FIRST_LAYER_FWD_B` | 特征图 batch 维度 = 输入 batch |
| `DEEP_FWD_BWD` | 深层前向+反向，特征图/probs/logits 的 batch 维度 |
| `FIRST_LAYER_BWD_A`, `FIRST_LAYER_BWD_B` | 首层反向梯度图 batch 维度 |
| `INF_MAIN_A`, `INF_MAIN_B` | 推理特征图 batch 维度 |
| `INF_EMA_A`, `INF_EMA_B` | EMA 推理特征图 batch 维度 |

**数量：8 个**，每个不同 ShapeId 捕获一次。

### 1.3 ShapeId 去重示例

**场景 A: CIFAR-10 (batch=200, 无渐进分辨率)**

```
standard_bs = 200, last_train_bs = 200, last_val_bs = 200
res_begin = 32, res_end = 32, val_res = 32
```
全部 6 个 variant 的 ShapeId = `{200, 32, 32, 3}` → **8 个 shape-dependent 图只捕获 1 套 → 零额外开销**。

**场景 B: ImageNet (8 ranks, batch=512, 渐进 128→224)**

```
standard_bs = 512, last_train_bs = 402, last_val_bs = 106
res_begin = 128, res_end = 224, val_res = 224
```
| Variant | ShapeId | 用途 |
|---------|---------|------|
| v0 | `{512, 128, 128, 3}` | 早期 epoch 常规 batch |
| v1 | `{402, 128, 128, 3}` | 早期 epoch last batch |
| v2 | `{512, 224, 224, 3}` | 后期 epoch 常规 batch |
| v3 | `{402, 224, 224, 3}` | 后期 epoch last batch |
| v4 | `{512, 224, 224, 3}` | 验证常规 batch (**与 v2 同 ShapeId!**) |
| v5 | `{106, 224, 224, 3}` | 验证 last batch |

5 个不同的 ShapeId → shape-dependent 图捕获 5 套（而非 6×8=48 套）。ShapeId 去重自动复用了 v2 和 v4。

**shape-dependent 图捕获总数：15 (invariant) + 8×5 = 55 张**。当前（1 variant）捕获约 23 张。增加约 2.3 倍，完全可接受。

---

## 二、ShapeId 去重机制 — 基础设施验证

### 2.1 机制原理

框架已具备完整的 ShapeId 去重基础设施，本方案**无需修改 Compiler 内核、无需修改 pre_capture 去重逻辑**。去重流程：

```
build_graph_atlas()
    ↓ 为每个 Slot 填入 cg + mp + shape_id
pre_capture()
    ↓ Phase B1: 遍历所有 Slot，构建 Key{cg, gid, shape_id}
    ↓ Key 相同 → captured_idx 复用 → result.reused++
    ↓ Key 不同 → captured_idx 新分配 → result.captured++
capture_all_for_rank()
    ↓ ctx.set_memory_plan(mp)  ← 用该变体的 MemoryPlan 捕获
    ↓ CapturedGraph::capture(*cg, *mp, gid, shape, stream, ctx)
    ↓ CUDA graph 的 grid 维度 = mp 中 DTensor 的 shape.n
```

**核心代码验证**（[captured_graph.cpp:L182-L201](file:///r:/renaissance/src/graph/captured_graph.cpp#L182-L201)）:

```cpp
for (size_t vi = 0; vi < GraphAtlas::kMaxVariants; ++vi) {
    for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
        auto& slot = result.atlas.slot(vi, gi);
        if (!slot.cg || !slot.mp) continue;

        CapturedGraph::Key key{slot.cg, static_cast<GraphId>(gi), slot.shape_id};

        auto it = seen.find(key);
        if (it != seen.end()) {
            slot.captured_idx = it->second;    // ← 复用
            ++result.reused;
        } else {
            slot.captured_idx = static_cast<int32_t>(seen.size());  // ← 新增
            seen[key] = slot.captured_idx;
            ++result.captured;
        }
    }
}
```

### 2.2 ShapeId 与 MemoryPlan 的关系

`CompileSpec::get_shape_id()` ([compile_spec.h:L66](file:///r:/renaissance/include/renaissance/graph/compile_spec.h#L66)):

```cpp
ShapeId get_shape_id() const noexcept {
    return ShapeId{ batch_size, actual_resolution,
                    actual_resolution, num_color_channels };
}
```

不同 CompileSpec 产生不同 MemoryPlan，其中 DTensor 形状不同但 slot_bytes 相同（Compiler Phase 2 的 `compute_max_slot_bytes` 取跨变体 max）。capture 时 `ctx.set_memory_plan(mp)` 使 `softmax_ce_op.cpp:L359` 的 `logits_dt.shape.n()` 读取到该变体的 batch 维度，从而产生正确 grid 维度的 CUDA Graph。

---

## 三、核心方案设计

### 3.1 总体架构

```
on_prepare()
    ├─ 生成 5 个 CompileSpec variant_specs
    ├─ Compiler::compile(base_spec, ..., variant_specs)
    ├─ 产出 6 个 MemoryPlan（variant_memory_plans_[0..5]）
    └─ 保存 Compiler::Result（含 6 变体 + train_cg + infer_cg）

build_graph_atlas()
    ├─ 计算各 variant 的 ShapeId（从 CompileSpec 获取）
    ├─ shape-invariant 图：全部 6 variant 填 mp[0] + kShapeInvariant
    └─ shape-dependent 图：各自 variant 填 mp[v] + ShapeId[v]

pre_capture()  ← 基础设施，无需修改
    ├─ 遍历 Atlas → Key{cg, gid, shape_id} 去重
    └─ 每 unique key 捕获一次 CapturedGraph

build_exec_table()
    ├─ 构建 per-variant-per-rank 的 exec table
    └─ shape-invariant 图在所有 variant 中指向同一 cudaGraphExec_t

run_train_epoch_gpu()
    ├─ 进入循环前：预存 variant 0/1 的所有 graph handles
    ├─ 循环内：is_last ? gs_last : gs_base
    └─ 零 variant 查找、零 hash、纯指针切换

run_val_epoch_gpu()
    └─ 同理，variant 4/5
```

### 3.2 关键设计原则

**原则 1: 不碰 Compiler 和 pre_capture 内核**。这两个模块已完整支持 6 variant，只需上层正确调用。

**原则 2: 所有 graph 在 batch 循环前预存储**。这是用户明确要求的性能优化。循环内部只有局部变量选择，无任何 map/atlas 查找。

**原则 3: ShapeId 是唯一真理来源**。去重、variant 选择、图共享全部基于 ShapeId，零冗余逻辑。

**原则 4: CPU 严格对齐 GPU**。如果 GPU 为某 variant 单独捕获了图，CPU 也必须捕获。

**原则 5: AMP 不影响 variant 布局**。AMP 只改变 DTensor dtype 和算子选择，不改变 shape，属于 CompileSpec 内部差异。

---

## 四、详细实现

### 4.1 Step 0: 数据结构准备

**文件**: `include/renaissance/task/deep_learning_task.h`

新增以下成员：

```cpp
// ===== Variant 系统新增成员 =====

/// Compiler 完整编译结果（含 6 variant + train_cg + infer_cg）
Compiler::Result compiled_result_;

/// 各 variant 的 CompileSpec（用于获取 ShapeId）
std::array<CompileSpec, 6> variant_compile_specs_;

/// 各 variant 的 MemoryPlan（slot_bytes 跨变体一致，shape 不同）
/// on_prepare() 从 compiled_result_ 中转移所有权
std::array<std::unique_ptr<MemoryPlan>, 6> variant_memory_plans_;
```

修改 `GpuExecTable`：

```cpp
struct GpuExecTable {
    // variant_graphs[variant_idx][rank][slot] → cudaGraphExec_t
    // variant_idx: 0=train_base, 1=train_last, 2=train_lowres,
    //              3=train_lowres_last, 4=val_base, 5=val_last
    // shape-invariant 图的 handles 在所有 variant_idx 中指向同一指针
    std::array<std::vector<std::vector<cudaGraphExec_t>>, 6> variant_graphs;
    std::vector<int> device_ids;
};
```

### 4.2 Step 1: `on_prepare()` — 传递 variant_specs

**文件**: `include/renaissance/task/deep_learning_task.h` (on_prepare 内联实现)

**改动点**: L288 附近

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
    for (size_t i = 0; i < compiled_result_.variants.size() && i < 6; ++i) {
        variant_memory_plans_[i] = std::move(compiled_result_.variants[i].memory_plan);
    }

    // ===== 保持向后兼容：active_memory_plan_ 指向 variant 0 =====
    memory_plan_ptr_ = std::move(variant_memory_plans_[0]);
    active_memory_plan_ = memory_plan_ptr_.get();

    // 持有 train_cg_ / infer_cg_ 引用（用于 Atlas 填表）
    train_cg_ = compiled_result_.train_cg.get();
    infer_cg_ = compiled_result_.infer_cg.get();

    // ... 后续代码不变（lr_dtensor_id_ 查找等）...
}
```

### 4.3 Step 2: `build_graph_atlas()` — 全量填表

**文件**: `src/task/deep_learning_task.cpp`

**当前问题**: 只填 variant 0，且所有 Slot 都设 `kShapeInvariant`。

**新实现**:

```cpp
namespace {
// shape-invariant GraphId 判定（独立函数，供 GraphAtlas::build 也可用）
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
} // anonymous

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
    const size_t num_v = compiled_result_.variants.size();
    for (size_t v = 0; v < num_v && v < 6; ++v) {
        bool is_train_var = (v <= 3);  // v0-v3 = 训练变体
        bool is_val_var   = (v >= 4);  // v4-v5 = 验证变体

        const MemoryPlan* mp_v = variant_memory_plans_[v].get();
        if (!mp_v) continue;

        // 该 variant 的 ShapeId（对 shape-invariant 图无意义，但仍需填入）
        ShapeId variant_shape = variant_compile_specs_[v].get_shape_id();

        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);

            // ===== 交叉槽位过滤 =====
            if (is_val_var && is_train_gid(gid)) continue;
            if (is_train_var && is_infer_gid(gid)) continue;

            // ===== 选择 ComputationGraph =====
            const ComputationGraph* cg = nullptr;
            if (is_train_gid(gid) || is_si_graph(gid) || gid == GraphId::TRANSFER_A ||
                gid == GraphId::TRANSFER_B) {
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
                sl.mp = mp_v;
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

**验证点**:
- shape-invariant 图（ZERO_GRAD 等）: variant 0/1/2/3/4/5 的 `captured_idx` 全部相同
- shape-dependent 图（DEEP_FWD_BWD）: 不同 ShapeId 的 variant 有不同的 `captured_idx`
- v2 和 v4 若 ShapeId 相同（如 `{512,224,224,3}`），则其 shape-dependent 图的 `captured_idx` 相同

### 4.4 Step 3: `build_exec_table()` — per-variant-per-rank

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
        gpu_exec_.variant_graphs[0].resize(K);
        for (int rank = 0; rank < K; ++rank) {
            gpu_exec_.variant_graphs[0][rank].resize(S(GraphSlot::COUNT), nullptr);
            auto& g = gpu_exec_.variant_graphs[0][rank];
            g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank, 0);
            g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank, 0);
        }
        return;
    }

    const size_t num_v = compiled_result_.variants.size();

    for (size_t v = 0; v < num_v && v < 6; ++v) {
        gpu_exec_.variant_graphs[v].resize(K);
        for (int rank = 0; rank < K; ++rank) {
            auto& g = gpu_exec_.variant_graphs[v][rank];
            g.resize(S(GraphSlot::COUNT), nullptr);

            g[S(GraphSlot::XFER_A)]            = resolve(GraphId::TRANSFER_A, rank, v);
            g[S(GraphSlot::XFER_B)]            = resolve(GraphId::TRANSFER_B, rank, v);
            g[S(GraphSlot::FWD_BWD_DEEP_A)]    = resolve(GraphId::DEEP_FWD_BWD, rank, v);
            g[S(GraphSlot::FWD_BWD_DEEP_B)]    = resolve(GraphId::DEEP_FWD_BWD, rank, v);
            g[S(GraphSlot::FIRST_LAYER_BWD_A)] = resolve(GraphId::FIRST_LAYER_BWD_A, rank, v);
            g[S(GraphSlot::FIRST_LAYER_BWD_B)] = resolve(GraphId::FIRST_LAYER_BWD_B, rank, v);
            g[S(GraphSlot::ZERO_GRAD)]         = resolve(GraphId::ZERO_GRAD, rank, v);
            g[S(GraphSlot::DEEP_ALLREDUCE)]    = resolve(GraphId::DEEP_COMM, rank, v);
            g[S(GraphSlot::FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM, rank, v);
            g[S(GraphSlot::WEIGHT_UPDATE)]     = resolve(GraphId::OPTIMIZER, rank, v);
            g[S(GraphSlot::EMA_UPDATE)]        = resolve(GraphId::EMA_UPDATE, rank, v);
            g[S(GraphSlot::CAST_DEEP_GRAD)]    = resolve(GraphId::CAST_DEEP_GRAD_FP16_TO_FP32, rank, v);
            g[S(GraphSlot::CAST_FIRST_GRAD)]   = resolve(GraphId::CAST_FIRST_GRAD_FP16_TO_FP32, rank, v);
            g[S(GraphSlot::NAN_CHECK_GRAD_SCALE)] = resolve(GraphId::NAN_CHECK_AND_GRAD_SCALING, rank, v);
            g[S(GraphSlot::FIRST_LAYER_FWD_A)] = resolve(GraphId::FIRST_LAYER_FWD_A, rank, v);
            g[S(GraphSlot::FIRST_LAYER_FWD_B)] = resolve(GraphId::FIRST_LAYER_FWD_B, rank, v);
            g[S(GraphSlot::CAST_MAIN)]         = resolve(GraphId::CAST_MAIN_FP32_TO_FP16, rank, v);
            g[S(GraphSlot::INF_MAIN_A)]        = resolve(GraphId::INF_MAIN_A, rank, v);
            g[S(GraphSlot::INF_MAIN_B)]        = resolve(GraphId::INF_MAIN_B, rank, v);
            g[S(GraphSlot::INF_EMA_A)]         = resolve(GraphId::INF_EMA_A, rank, v);
            g[S(GraphSlot::INF_EMA_B)]         = resolve(GraphId::INF_EMA_B, rank, v);
            g[S(GraphSlot::ACCUM_METRICS)]     = resolve(GraphId::ACCUM_METRICS, rank, v);
            g[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)] = resolve(GraphId::ACCUM_METRICS_TRAIN_LAST, rank, v);
            g[S(GraphSlot::ACCUM_METRICS_VAL_LAST)]   = resolve(GraphId::ACCUM_METRICS_VAL_LAST, rank, v);
            g[S(GraphSlot::VAL_RESULT_ALLREDUCE)]     = resolve(GraphId::VAL_RESULT_COMM, rank, v);
            g[S(GraphSlot::CLEAR_METRICS)]     = resolve(GraphId::CLEAR_METRICS, rank, v);
        }
    }

    // 校验 variant 0 的关键 slot
    static const GraphSlot kRequired[] = {
        GraphSlot::XFER_A, GraphSlot::XFER_B,
        GraphSlot::FWD_BWD_DEEP_A,
        GraphSlot::FIRST_LAYER_BWD_A,
    };
    for (int rank = 0; rank < K; ++rank) {
        for (auto slot : kRequired) {
            TR_CHECK(gpu_exec_.variant_graphs[0][rank][static_cast<size_t>(slot)],
                     ValueError, "Required slot nullptr for rank " << rank);
        }
    }
#endif
}
```

### 4.5 Step 4: `run_train_epoch_gpu()` — 预存 graph + variant 切换

**文件**: `src/task/deep_learning_task.cpp`

**核心改动**: 在线程 lambda 内部，在 batch 循环前预存两组 graph handles，循环内根据 `is_last` 选择。

```cpp
#ifdef TR_USE_CUDA
float DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    // ===== 确定本 epoch 使用的 training variant 对 =====
    // progressive resolution: epoch < boundary → res_begin (v0/v1), else → res_end (v2/v3)
    bool use_lowres = false;
    if (registry.using_progressive_resolution()) {
        int boundary = registry.boundary_epoch();
        use_lowres = (current_epoch_ < boundary);
    }
    size_t v_base = use_lowres ? 2 : 0;   // train_base 或 train_lowres
    size_t v_last = use_lowres ? 3 : 1;   // train_last 或 train_lowres_last

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    int32_t loss_id = active_memory_plan_->baseline().loss;

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, K, using_amp, &exc, loss_id,
                              v_base, v_last]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // ======== 预存储：variant base 的所有 graph ========
                const auto& g_base = gpu_exec_.variant_graphs[v_base][rank];
                auto g_fwd_a_0   = g_base[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto g_fwd_b_0   = g_base[S(GraphSlot::FIRST_LAYER_FWD_B)];
                auto g_deep_0    = g_base[S(GraphSlot::FWD_BWD_DEEP_A)];    // A/B 同图
                auto g_first_a_0 = g_base[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto g_first_b_0 = g_base[S(GraphSlot::FIRST_LAYER_BWD_B)];

                // ======== 预存储：variant last 的所有 graph ========
                const auto& g_last = gpu_exec_.variant_graphs[v_last][rank];
                auto g_fwd_a_l   = g_last[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto g_fwd_b_l   = g_last[S(GraphSlot::FIRST_LAYER_FWD_B)];
                auto g_deep_l    = g_last[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto g_first_a_l = g_last[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto g_first_b_l = g_last[S(GraphSlot::FIRST_LAYER_BWD_B)];

                // ======== shape-invariant 图（从任意 variant 取，均相同）=======
                auto g_xfer_a        = g_base[S(GraphSlot::XFER_A)];
                auto g_xfer_b        = g_base[S(GraphSlot::XFER_B)];
                auto g_zg            = g_base[S(GraphSlot::ZERO_GRAD)];
                auto g_dar           = g_base[S(GraphSlot::DEEP_ALLREDUCE)];
                auto g_far           = g_base[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto g_wu            = g_base[S(GraphSlot::WEIGHT_UPDATE)];
                auto g_cdg           = g_base[S(GraphSlot::CAST_DEEP_GRAD)];
                auto g_cfg           = g_base[S(GraphSlot::CAST_FIRST_GRAD)];
                auto g_ncg           = g_base[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                auto g_cm            = g_base[S(GraphSlot::CAST_MAIN)];
                auto g_accum         = g_base[S(GraphSlot::ACCUM_METRICS)];
                auto g_accum_tl      = g_base[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];
                auto g_clear         = g_base[S(GraphSlot::CLEAR_METRICS)];

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
                auto sync_up  = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr  = [&]() { cudaStreamSynchronize(s_trans); };

                // ... TransferStation 获取等不变 ...

                if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }

                // ===== Batch 0 预传输（始终用 variant base，batch 0 总是 full）=====
                ts->wait_buffer_readable(0);
                if (g_xfer_a) cudaGraphLaunch(g_xfer_a, s_trans);
                sync_tr();
                ts->set_buffer_readable(0, false);
                ts->set_buffer_writeable(0, true);
                if (g_clear) cudaGraphLaunch(g_clear, s_up);

                // ===== 统一循环：batch 0 .. batches-2（常规 batch → variant base）=====
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    auto g_fwd   = from_a ? g_fwd_a_0 : g_fwd_b_0;
                    auto g_deep  = g_deep_0;
                    auto g_xfer_n = from_a ? g_xfer_b : g_xfer_a;
                    auto g_first_cur = from_a ? g_first_a_0 : g_first_b_0;

                    // Phase 1: ZERO_GRAD ‖ FIRST_FWD
                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                    sync_comp(); sync_up();

                    ts->wait_buffer_readable(next_buf);

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
                    sync_comp(); sync_tr();

                    if (g_accum) cudaGraphLaunch(g_accum, s_up);
                    sync_up();
                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);

                    if (!frozen) {
                        if (g_first_cur) cudaGraphLaunch(g_first_cur, s_c1);
                    }
                    sync_comp();

                    if (using_amp && g_cdg) { cudaGraphLaunch(g_cdg, s_up); sync_up(); }
                    if (g_dar) cudaGraphLaunch(g_dar, s_up); sync_up();
                    if (using_amp && g_cfg) { cudaGraphLaunch(g_cfg, s_up); sync_up(); }
                    if (g_far) cudaGraphLaunch(g_far, s_up); sync_up();
                    if (using_amp && g_ncg) { cudaGraphLaunch(g_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batch);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up); sync_up();
                    if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }
                }

                // ===== Last batch → variant last =====
                {
                    bool last_a = ((batches - 1) % 2 == 0);
                    auto g_fwd_l = last_a ? g_fwd_a_l : g_fwd_b_l;
                    auto g_first_l = last_a ? g_first_a_l : g_first_b_l;

                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd_l) cudaGraphLaunch(g_fwd_l, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
                    sync_comp();

                    if (g_accum_tl) cudaGraphLaunch(g_accum_tl, s_up);
                    sync_up();

                    if (!frozen) {
                        if (g_first_l) cudaGraphLaunch(g_first_l, s_c1);
                    }
                    sync_comp();

                    if (using_amp && g_cdg) { cudaGraphLaunch(g_cdg, s_up); sync_up(); }
                    if (g_dar) cudaGraphLaunch(g_dar, s_up); sync_up();
                    if (using_amp && g_cfg) { cudaGraphLaunch(g_cfg, s_up); sync_up(); }
                    if (g_far) cudaGraphLaunch(g_far, s_up); sync_up();
                    if (using_amp && g_ncg) { cudaGraphLaunch(g_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batches - 1);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up); sync_up();
                    if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }
                }

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    // ... join + rethrow + accum_loss 读取不变 ...
}
#endif
```

**性能分析**:
- 循环内 `g_deep`、`g_fwd` 等为局部 `cudaGraphExec_t` 变量，直接 `cudaGraphLaunch`，**零函数调用、零查表**
- `is_last` 判断仅在 last batch 块中使用一次条件分支
- 预存的 graph handles 总数 ≈ 30（15 shape-invariant + 10 shape-dependent for both variants + 5 extra），全部在栈上

### 4.6 Step 5: `run_val_epoch_gpu()` — 同理 pre-store

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
    // ...

    size_t v_val_base = 4;
    size_t v_val_last = 5;

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            // ...

            // ===== Pre-store: variant 4 & 5 的 INF graph =====
            const auto& gv4 = gpu_exec_.variant_graphs[4][rank];
            auto g_inf_a_4 = gv4[S(GraphSlot::INF_MAIN_A)];
            auto g_inf_b_4 = gv4[S(GraphSlot::INF_MAIN_B)];

            const auto& gv5 = gpu_exec_.variant_graphs[5][rank];
            auto g_inf_a_5 = gv5[S(GraphSlot::INF_MAIN_A)];
            auto g_inf_b_5 = gv5[S(GraphSlot::INF_MAIN_B)];

            // shape-invariant 从任意 variant 取
            auto g_xfer_a     = gv4[S(GraphSlot::XFER_A)];
            auto g_xfer_b     = gv4[S(GraphSlot::XFER_B)];
            auto g_accum      = gv4[S(GraphSlot::ACCUM_METRICS)];
            auto g_accum_vl   = gv4[S(GraphSlot::ACCUM_METRICS_VAL_LAST)];
            auto g_val_comm   = gv4[S(GraphSlot::VAL_RESULT_ALLREDUCE)];
            auto g_clear      = gv4[S(GraphSlot::CLEAR_METRICS)];

            if (g_clear) cudaGraphLaunch(g_clear, s_up);

            for (int batch = 0; batch < val_batches; ++batch) {
                int buf = batch % 2;
                bool is_last = (batch == val_batches - 1);

                if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                if (top1_id >= 0) cudaMemsetAsync(ctx.ptr_at(top1_id), 0, sizeof(float), s_c1);
                if (top5_id >= 0) cudaMemsetAsync(ctx.ptr_at(top5_id), 0, sizeof(float), s_c1);

                ts->wait_buffer_readable(buf);

                auto g_xfer = (buf == 0) ? g_xfer_a : g_xfer_b;
                if (g_xfer) cudaGraphLaunch(g_xfer, s_trans);
                sync_tr();

                auto g_inf = is_last
                    ? ((buf == 0) ? g_inf_a_5 : g_inf_b_5)
                    : ((buf == 0) ? g_inf_a_4 : g_inf_b_4);
                if (g_inf) cudaGraphLaunch(g_inf, s_c1);
                sync_comp();

                auto g_va = is_last ? g_accum_vl : g_accum;
                if (g_va) cudaGraphLaunch(g_va, s_up);
                sync_up();

                ts->set_buffer_readable(buf, false);
                ts->set_buffer_writeable(buf, true);
            }

            if (g_val_comm) cudaGraphLaunch(g_val_comm, s_up);
            sync_up();
        });
    }
    // ...
}
```

### 4.7 Step 6: CPU 路径 — 完全对齐 GPU

**文件**: `src/task/deep_learning_task.cpp`

CPU 路径的本质差异：不使用 CUDA Graph，而是 `graphs[idx].launch(0, nullptr)` 即时执行。CPU 的 `CpuOp` 会从 `DeviceContext::memory_plan()` 获取已设置的 MemoryPlan 以读取 DTensor shape。`pre_capture()` 中 `ctx.set_memory_plan(mp)` 对 CPU 路径同样生效。

#### `run_train_epoch_cpu()`

```cpp
float DeepLearningTask::run_train_epoch_cpu() {
    // ... 前置不变 ...

    const auto& atlas = captured_result_.atlas;
    const auto& graphs = captured_result_.graphs;

    // ===== 支持 variant 的 idx_for =====
    auto idx_for = [&atlas](GraphId gid, size_t v) -> int32_t {
        return atlas.index(v, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    // ===== 确定 variant =====
    bool use_lowres = false;
    if (registry.using_progressive_resolution()) {
        int boundary = registry.boundary_epoch();
        use_lowres = (current_epoch_ < boundary);
    }
    size_t v_base = use_lowres ? 2 : 0;
    size_t v_last = use_lowres ? 3 : 1;

    // ===== Pre-store shape-invariant 图的 idx（从 variant 0）=====
    int32_t idx_xfer_a = idx_for(GraphId::TRANSFER_A, 0);
    int32_t idx_xfer_b = idx_for(GraphId::TRANSFER_B, 0);
    int32_t idx_zg     = idx_for(GraphId::ZERO_GRAD, 0);
    int32_t idx_far    = idx_for(GraphId::FIRST_COMM, 0);
    int32_t idx_dar    = idx_for(GraphId::DEEP_COMM, 0);
    int32_t idx_opt    = idx_for(GraphId::OPTIMIZER, 0);
    int32_t idx_clear  = idx_for(GraphId::CLEAR_METRICS, 0);
    int32_t idx_accum  = idx_for(GraphId::ACCUM_METRICS, 0);
    int32_t idx_accum_tl = idx_for(GraphId::ACCUM_METRICS_TRAIN_LAST, 0);

    // ===== Pre-store variant base 的 shape-dependent 图 idx =====
    int32_t idx_fwd_a_0  = idx_for(GraphId::FIRST_LAYER_FWD_A, v_base);
    int32_t idx_fwd_b_0  = idx_for(GraphId::FIRST_LAYER_FWD_B, v_base);
    int32_t idx_deep_0   = idx_for(GraphId::DEEP_FWD_BWD, v_base);
    int32_t idx_bwd_a_0  = idx_for(GraphId::FIRST_LAYER_BWD_A, v_base);
    int32_t idx_bwd_b_0  = idx_for(GraphId::FIRST_LAYER_BWD_B, v_base);

    // ===== Pre-store variant last 的 shape-dependent 图 idx =====
    int32_t idx_fwd_a_l  = idx_for(GraphId::FIRST_LAYER_FWD_A, v_last);
    int32_t idx_fwd_b_l  = idx_for(GraphId::FIRST_LAYER_FWD_B, v_last);
    int32_t idx_deep_l   = idx_for(GraphId::DEEP_FWD_BWD, v_last);
    int32_t idx_bwd_a_l  = idx_for(GraphId::FIRST_LAYER_BWD_A, v_last);
    int32_t idx_bwd_b_l  = idx_for(GraphId::FIRST_LAYER_BWD_B, v_last);

    // ===== CLEAR_METRICS =====
    if (idx_clear >= 0) launch(idx_clear);

    // ===== H2D pre-fetch 不变 =====
    ts->wait_buffer_readable(0);
    launch(idx_xfer_a);
    ts->set_buffer_readable(0, false);
    ts->set_buffer_writeable(0, true);

    if (batches == 1) {
        launch(idx_zg);
        launch(idx_fwd_a_0);
        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_0);
        if (!frozen) launch(idx_bwd_a_0);
        launch(idx_dar);
        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(0);
        launch(idx_far);
        launch(idx_opt);

        // ===== 读取 accum_loss 而非 loss_id（和 GPU 对齐）=====
        if (accum_loss_id >= 0) {
            Tensor h_al = fetch_from_rank(
                active_memory_plan_->get_dtensor(accum_loss_id), 0);
            float acc_val = h_al.data<float>()[0];
            size_t total = registry.num_train_samples();
            train_loss = (total > 0) ? acc_val / static_cast<float>(total) : 0.0f;
        }
        return train_loss;
    }

    // ===== 循环：常规 batch（variant base）=====
    for (int batch = 0; batch < batches - 1; ++batch) {
        bool from_a  = (batch % 2 == 0);
        int next_buf = from_a ? 1 : 0;

        launch(idx_zg);
        launch(from_a ? idx_fwd_a_0 : idx_fwd_b_0);

        ts->wait_buffer_readable(next_buf);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_0);
        launch(from_a ? idx_xfer_b : idx_xfer_a);

        ts->set_buffer_readable(next_buf, false);
        ts->set_buffer_writeable(next_buf, true);

        // ACCUM_METRICS（需要预先写入 batch_size 标量）
        *static_cast<int32_t*>(ctx.ptr_at(registry.local_batch_size_id())) =
            registry.get_local_batch_size();
        if (idx_accum >= 0) launch(idx_accum);

        if (!frozen) launch(from_a ? idx_bwd_a_0 : idx_bwd_b_0);
        launch(idx_dar);

        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batch);
        launch(idx_far);
        launch(idx_opt);
    }

    // ===== Last batch（variant last）=====
    {
        bool last_a = ((batches - 1) % 2 == 0);

        launch(idx_zg);
        launch(last_a ? idx_fwd_a_l : idx_fwd_b_l);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_l);

        // ACCUM_METRICS_TRAIN_LAST
        *static_cast<int32_t*>(ctx.ptr_at(registry.last_train_batch_size_id())) =
            registry.get_last_train_batch_size();
        if (idx_accum_tl >= 0) launch(idx_accum_tl);

        if (!frozen) launch(last_a ? idx_bwd_a_l : idx_bwd_b_l);
        launch(idx_dar);

        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batches - 1);
        launch(idx_far);
        launch(idx_opt);
    }

    // ===== 从 accum_loss 读取（与 GPU 路径对齐）=====
    if (accum_loss_id >= 0) {
        const auto& accum_dt = active_memory_plan_->get_dtensor(accum_loss_id);
        Tensor h_accum = fetch_from_rank(accum_dt, 0);
        float accum_val = h_accum.data<float>()[0];
        size_t total = GlobalRegistry::instance().num_train_samples();
        if (total > 0) train_loss = accum_val / static_cast<float>(total);
    }

    return train_loss;
}
```

**注意**: CPU 路径的 `ACCUM_METRICS` / `ACCUM_METRICS_TRAIN_LAST` 需要在 launch 前手动写入 `batch_size` 标量到 DTensor 地址（GPU 路径通过 `input_ids` 自动绑定）。需要从 active_memory_plan 获取 `local_batch_size` / `last_train_batch_size` 的 DTensor ID。如果当前 MemoryPlan 没有暴露这些 ID，需在 Step 7 中解决。

#### `run_val_epoch_cpu()`

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_cpu(bool validate_ema) {
    // ... 前置不变 ...

    auto idx_for = [&atlas](GraphId gid, size_t v) -> int32_t {
        return atlas.index(v, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    // shape-invariant
    int32_t idx_xfer_a = idx_for(GraphId::TRANSFER_A, 0);
    int32_t idx_xfer_b = idx_for(GraphId::TRANSFER_B, 0);

    // val_base (v4)
    int32_t idx_inf_a_4 = idx_for(GraphId::INF_MAIN_A, 4);
    int32_t idx_inf_b_4 = idx_for(GraphId::INF_MAIN_B, 4);

    // val_last (v5)
    int32_t idx_inf_a_5 = idx_for(GraphId::INF_MAIN_A, 5);
    int32_t idx_inf_b_5 = idx_for(GraphId::INF_MAIN_B, 5);

    // ... TransferStation 不变 ...

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
            ? (is_last ? idx_inf_a_5 : idx_inf_a_4)
            : (is_last ? idx_inf_b_5 : idx_inf_b_4));

        float batch_loss = loss_ptr ? *static_cast<float*>(loss_ptr) : 0.0f;
        float batch_top1 = top1_ptr ? *static_cast<float*>(top1_ptr) : 0.0f;
        float batch_top5 = top5_ptr ? *static_cast<float*>(top5_ptr) : 0.0f;

        int bs = is_last ? registry.get_last_val_batch_size()
                         : registry.get_local_batch_size();
        acc_loss += batch_loss * static_cast<float>(bs);
        acc_top1 += batch_top1 * static_cast<float>(bs);
        acc_top5 += batch_top5 * static_cast<float>(bs);

        ts->set_buffer_readable(buf, false);
        ts->set_buffer_writeable(buf, true);
    }

    size_t total_val = registry.num_val_samples();
    float avg_loss = (total_val > 0) ? static_cast<float>(acc_loss / static_cast<double>(total_val)) : 0.0f;
    float avg_top1 = (total_val > 0) ? static_cast<float>(acc_top1 / static_cast<double>(total_val)) : 0.0f;
    float avg_top5 = (total_val > 0) ? static_cast<float>(acc_top5 / static_cast<double>(total_val)) : 0.0f;

    // ... LOG 不变 ...
    return {avg_loss, avg_top1, avg_top5};
}
```

### 4.8 Step 7: 接入点清单（ensure all wiring works）

以下改动需要逐项确认：

| 编号 | 位置 | 内容 |
|:----:|------|------|
| 7.1 | `deep_learning_task.h` | 新增成员 `compiled_result_`, `variant_compile_specs_`, `variant_memory_plans_` |
| 7.2 | `deep_learning_task.h` | 修改 `GpuExecTable` 为 `variant_graphs[6]` |
| 7.3 | `deep_learning_task.h` `on_prepare()` | 传递 5 个 variant_specs 给 Compiler，保存各 variant MemoryPlan |
| 7.4 | `deep_learning_task.cpp` | 新增 `is_si_graph()`, `is_train_gid()`, `is_infer_gid()` 辅助函数 |
| 7.5 | `deep_learning_task.cpp` `build_graph_atlas()` | 全量填表 |
| 7.6 | `deep_learning_task.cpp` `build_exec_table()` | per-variant-per-rank 构建 |
| 7.7 | `deep_learning_task.cpp` `run_train_epoch_gpu()` | 预存 + variant 切换 |
| 7.8 | `deep_learning_task.cpp` `run_val_epoch_gpu()` | 预存 + variant 切换 |
| 7.9 | `deep_learning_task.cpp` `run_train_epoch_cpu()` | variant 切换 + ACCUM_METRICS 累积 |
| 7.10 | `deep_learning_task.cpp` `run_val_epoch_cpu()` | variant 切换 + 加权平均修正 |
| 7.11 | `deep_learning_task.cpp` `compile_h2d_only()` | 适配新的 `compile()` 调用（可选，h2d_only 可保持简单 1 variant） |

### 4.9 Step 8: SOFTMAX_CE kernel 边界修复（同步进行）

本方案不依赖 kernel 修复（variant 系统从根因上消除 last batch 问题），但 **kernel 修复作为防御层** 仍应同步实施，以防未来有其他路径触发：

**文件**: `src/backend/ops/dtensor/softmax_ce_op.cu` + `softmax_ce_op.cpp`

| 改动 | 位置 | 内容 |
|:----:|------|------|
| K1 | `softmax_ce_fwd_kernel` L75 | `if (b >= *batch_size_ptr) return;` |
| K2 | `softmax_ce_inf_kernel` L178 | 同上 |
| K3 | `softmax_ce_bwd_kernel` L307-L334 | 增加 `batch_size_ptr` 参数，`total = (*batch_size_ptr) * num_classes` |
| K4 | `launch_softmax_ce_bwd_fp32/amp` L405-L431 | 新增 `batch_size_ptr` 参数 |
| K5 | BWD dispatch L513-L575 | 传递 `batch_size_ptr` |

---

## 五、渐进式分辨率完整支持

### 5.1 运行时选择逻辑

```cpp
size_t v_base, v_last;
if (GlobalRegistry::instance().using_progressive_resolution()) {
    int boundary = GlobalRegistry::instance().boundary_epoch();
    if (current_epoch_ < boundary) {
        v_base = 2;  // train_lowres (epoch < boundary → 使用 res_begin)
        v_last = 3;  // train_lowres_last
    } else {
        v_base = 0;  // train_base (epoch >= boundary → 使用 res_end)
        v_last = 1;  // train_last
    }
} else {
    v_base = 0;  // 非渐进: 只有一种分辨率
    v_last = 1;
}
```

**注意**: 这里的"lowres"命名可能与实际分辨率大小不一致（见 1.1 节分析），但 variant 的 ShapeId 是由 `train_sample_resolution_begin()` / `train_sample_resolution_end()` 的实际值决定的，所以运行时效果是正确的。

### 5.2 边界 epoch 切换时的行为

当 `current_epoch_ == boundary` 时，`run_train_epoch_gpu()` 使用 v0/v1（res_end 分辨率），而 `run_val_epoch_gpu()` 始终使用 v4/v5（val_res 分辨率）。这是预期行为，因为验证分辨率与训练分辨率独立。

### 5.3 编译时处理 same-shape 退化

Compiler 的 `share_or_clone` 已经在 Phase 5 处理了 variant 间 MemoryPlan 共享。当 `res_begin == res_end` 且 `last_bs == standard_bs` 时，所有 6 个 variant 的 ShapeId 相同，pre_capture 自动将所有 shape-dependent 图合并为 1 套 — **零额外开销**。

---

## 六、CPU 路径累积指标修复

CPU 路径存在两处需要额外修复：

### 6.1 训练累积（修复 B4）

当前 `run_train_epoch_cpu()` 缺少 ACCUM_METRICS。需要：

1. 添加 `CLEAR_METRICS` 调用
2. 循环内添加 `ACCUM_METRICS`（写入 `local_batch_size` 标量后 launch）
3. 最后 batch 使用 `ACCUM_METRICS_TRAIN_LAST`（写入 `last_train_batch_size` 标量后 launch）
4. 返回值从 `loss_id` 改为读 `accum_loss_id`

**前提**: MemoryPlan 的 baseline 需要暴露 `local_batch_size`、`last_train_batch_size` 等标量的 DTensor ID。当前 `deep_learning_task.cpp` 中已有 `active_memory_plan_->baseline().local_batch_size` 等字段。

### 6.2 验证加权修正（修复 B5）

已在 Step 6 的 `run_val_epoch_cpu()` 中包含。

---

## 七、多 RANK 支持

本方案的所有设计均与多 RANK 兼容：

- `build_exec_table()` 中 `per-rank` 构建 exec table：每个 rank 独立调用 `atlas.index(variant, gid)` + `native_exec(rank)`
- `pre_capture` 已支持 rank 0 串行 + rank 1~N-1 并行捕获
- 不同 rank 的 `cudaGraphExec_t` 句柄独立，但指向同一套去重后的 `CapturedGraph`
- H2D TRANSFER 图不受 variant 影响（永远是 A/B 两套 buffer）

---

## 八、验证策略

### 8.1 分阶段验证

| 阶段 | 测试 | 验证内容 |
|:----:|------|----------|
| 1 | **Dry-run** | `variant_specs.size() == 6`, variant 命名正确, MemoryPlan 创建成功 |
| 2 | **GraphAtlas 结构** | shape-invariant 图的 6 variant `captured_idx` 相同; shape-dependent 图不同 ShapeId 的 `captured_idx` 不同 |
| 3 | **CIFAR-10 全模式** | `test_dl_full --cpu/gpu/amp`, 3 epochs, 无回归 |
| 4 | **CIFAR-10 last batch** | 修改 `batch=192`（不可整除），验证 variant 切换无异常 |
| 5 | **dry_run 日志** | `pre_capture` 输出 `captured/reused` 计数验证去重效果 |

### 8.2 ShapeId 去重验证方法

在 `pre_capture()` 后增加诊断日志：

```cpp
// 在 build_exec_table 后：
for (size_t v = 0; v < 6; ++v) {
    int32_t idx = captured_result_.atlas.index(v, GraphId::DEEP_FWD_BWD);
    LOG_INFO << "[VARIANT] v=" << v << " DEEP_FWD_BWD captured_idx=" << idx;
}
```

CIFAR-10 (batch=200): 所有 6 variant 的 `captured_idx` 应相同（ShapeId 退化）。
ImageNet (progressive): v0, v1, v2, v3, v4, v5 中 v2==v4（同 ShapeId），其余各不同。

---

## 九、修正变更文件清单

| 文件 | 行数估算 | 关键改动 |
|------|:--------:|----------|
| `include/renaissance/task/deep_learning_task.h` | ~60 | 新增成员变量 `compiled_result_` 等；修改 `on_prepare()` 传递 variant_specs；修改 `GpuExecTable` |
| `src/task/deep_learning_task.cpp` | ~250 | 新增辅助函数；重写 `build_graph_atlas()`；重写 `build_exec_table()`；修改 4 个 run epoch 函数 |
| `src/backend/ops/dtensor/softmax_ce_op.cu` | ~30 | FWD/INF/BWD kernel 边界修复 |
| `src/backend/ops/dtensor/softmax_ce_op.cpp` | ~20 | BWD dispatch 传 `batch_size_ptr` |
| **总计** | **~360** | |

**不需要修改的文件**: `src/graph/compiler.cpp`, `src/graph/captured_graph.cpp`, `src/graph/pre_capture.cpp`, `include/renaissance/graph/*.h` — 这些基础设施已完整支持。

---

## 十、总结

本方案的核心思想：

1. **ShapeId 是唯一真理来源** — 去重、variant 选择、图复用全部由 ShapeId 自然驱动，零冗余逻辑。

2. **不重复造轮子** — Compiler 的 6 variant 创建、pre_capture 的 `Key{cg, gid, shape_id}` 去重、`ctx.set_memory_plan(mp)` 的 shape 注入 — 这些基础设施已就绪，本方案只是正确调用。

3. **退化零开销** — 当 `standard_bs == last_bs` 或 `res_begin == res_end` 时，ShapeId 相同 → 捕获的图数完全等于当前（1 variant）方案。

4. **性能极致** — 所有 graph handles 在循环前预存为局部变量，循环内零查表、零 hash。

5. **CPU/GPU 统一** — 两者共享同一套 variant 选择逻辑，CPU 的 `graphs[idx].launch()` 自然跟随 GPU 的捕获策略。