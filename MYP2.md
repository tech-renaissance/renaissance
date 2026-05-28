# MYP2: Last Batch、不同 Shape 捕获与渐进式分辨率 —— 完整实施方案

> 综合 LRF1 / LRF2 / LRF3 三份方案之精华，结合源代码实地验证，给出可直接落地、零妥协、高性能的最终方案。

---

## 一、执行摘要

| 问题 | 根因 | 修复方式 |
|---|---|---|
| Last batch 使用 base batch 的 CUDA Graph → grid 过大，处理残留数据 | `build_graph_atlas()` 只填 variant 0，所有 slot `shape_id = kShapeInvariant` | 6-Variant 全量填表，`pre_capture` 自动按 `ShapeId` 去重捕获 |
| CPU 训练路径无指标累积 | 未调用 `ACCUM_METRICS` / `CLEAR_METRICS` | 补全累积逻辑，batch 循环前写入 `batch_size` 标量 |
| CPU 验证路径按 batch 数平均 | 未按样本数加权 | `acc_loss += batch_loss * bs`，最后除 `num_val_samples` |
| 渐进式分辨率未接入 | 运行时无 variant 切换 | epoch 开头根据 `get_current_train_resolution()` 选 v0/v1 或 v2/v3 |

**核心设计**：不修改 Compiler / pre_capture / kernel，仅修改 `DeepLearningTask` 的 orchestration 层（~400 行）。

---

## 二、问题根因与关键约束

### 2.1 三层断裂（与前三份方案一致）

| 层级 | 设计意图 | 当前实现 | 断裂点 |
|---|---|---|---|
| **Compiler** | 6 变体编译 | `on_prepare()` 只传 1 个 base_spec | 只产出 1 个 Variant |
| **GraphAtlas** | 6×28 映射表，shape 相关图各自 `mp` + `shape_id` | 只填 variant 0，全部 `kShapeInvariant` | pre_capture 只捕获 1 套图 |
| **运行时** | 按 batch 类型选 correct variant | `build_exec_table()` 只解析 `atlas.index(0, gid)` | last batch 使用 grid=base_batch 的图 |

### 2.2 用户关键约束（不可违背）

1. **H2D 传输图只有 2 种**（A/B），与分辨率/batch 无关，AMP 不同则不同。
2. **权重更新/梯度同步不受 batch size 影响** —— OPTIMIZER/COMM/EMA 属于 shape-invariant。
3. **受 batch size 影响的图**：首层正反向、深层正反向（特征图 batch 维度）。
4. **性能至上**：进入 batch 循环前，局部变量存储好所有图（含 last batch）。
5. **CPU/GPU 对齐**：GPU 单独捕获的情形，CPU 也必须单独捕获。
6. **多 RANK**：每个 rank 线程独立预解析自己的 handles。

### 2.3 基础设施状态（已就绪，无需修改）

- `Compiler::compile(base, ..., variant_specs)` —— 已支持 6 variant
- `pre_capture()` 的 `Key{cg, gid, shape_id}` 去重 —— 已就绪
- `capture_all_for_rank()` 的 `ctx.set_memory_plan(mp)` —— 已就绪
- `computation_graph.h` 的 `is_shape_invariant_graph()` / `is_train_graph()` / `is_inference_graph()` —— 已就绪

---

## 三、6-Variant 系统精确定义

| Variant | 索引 | 场景 | batch_size | resolution | CG 来源 |
|---|---|---|---|---|---|
| 0 | train_base | 训练·常规 batch | `local_batch_size` | `train_sample_resolution_begin()` | train_cg |
| 1 | train_last | 训练·last batch | `last_train_batch_size` | `train_sample_resolution_begin()` | train_cg |
| 2 | train_lowres | 训练·低分辨率常规 | `local_batch_size` | `train_sample_resolution_end()` | train_cg |
| 3 | train_lowres_last | 训练·低分辨率 last | `last_train_batch_size` | `train_sample_resolution_end()` | train_cg |
| 4 | val_base | 验证·常规 batch | `local_batch_size` | `val_sample_resolution()` | infer_cg |
| 5 | val_last | 验证·last batch | `last_val_batch_size` | `val_sample_resolution()` | infer_cg |

**退化零开销**：当 `begin == end`（非渐进）或 `standard_bs == last_bs`（整除）时，对应 variant 的 `ShapeId` 相同 → `pre_capture` 自动合并 → 捕获图数不增加。

---

## 四、Graph 分类与去重机制

### 4.1 三类 Graph

| 类型 | GraphId | Shape 相关 | 去重行为 |
|---|---|---|---|
| **完全 Shape 无关** | TRANSFER_A/B, ZERO_GRAD, DEEP_COMM, FIRST_COMM, STATS_COMM, CAST_*, NAN_CHECK, OPTIMIZER, EMA_UPDATE, CLEAR_METRICS, VAL_RESULT_COMM | ❌ | 6 variant 共享 1 张图 |
| **训练 Shape 相关** | FIRST_LAYER_FWD_A/B, DEEP_FWD_BWD, FIRST_LAYER_BWD_A/B | ✅ Batch | 按 ShapeId 分别捕获 |
| **推理 Shape 相关** | INF_MAIN_A/B, INF_EMA_A/B | ✅ Batch | 按 ShapeId 分别捕获 |

**ACCUM_METRICS 系列说明**：`ACCUM_METRICS`、`ACCUM_METRICS_TRAIN_LAST`、`ACCUM_METRICS_VAL_LAST` 三个 GraphId **各不相同**，所以即使都是 shape-invariant，三者之间**不互相共享**（`Key` 中 `gid` 不同）。每个在 6 variant 中各自共享 1 张图。

### 4.2 去重后 Captured Graph 数量估算

**场景 A: CIFAR-10 (batch=200, 无渐进)**
- 全部 6 variant ShapeId = `{200,32,32,3}` → shape 相关图只捕获 1 套
- 总计：~15 (invariant) + ~10 (shape-dep, 1 套) + 3 (ACCUM* 独立) = **~28 张**
- 相比当前（~23 张）增加极小

**场景 B: ImageNet (batch=512, 渐进 128→224)**
- 5 个不同 ShapeId：`{512,128,128,3}`, `{402,128,128,3}`, `{512,224,224,3}`, `{402,224,224,3}`, `{106,224,224,3}`
- 注意：v2 `{512,224,224,3}` 与 v4 `{512,224,224,3}` **同 ShapeId** → 自动复用
- shape 相关训练图：~6 个 GraphId × 4 个不同 train ShapeId = 24
- shape 相关推理图：~4 个 GraphId × 2 个不同 val ShapeId = 8
- invariant：~15 张
- ACCUM*：3 张（各自独立）
- **总计 ≈ 50 张**，完全可接受

---

## 五、详细实现

### 5.1 头文件修改 (`deep_learning_task.h`)

**新增成员变量**（`private:` 区域）：

```cpp
// ===== 6-Variant 系统 =====
/// 各变体的独立 MemoryPlan（slot_bytes 跨变体一致，shape 不同）
std::array<std::unique_ptr<MemoryPlan>, 6> variant_memory_plans_;

/// 各变体的 CompileSpec（用于 build_graph_atlas 获取 ShapeId）
std::array<CompileSpec, 6> variant_specs_;
```

> **不添加 `compiled_result_`**：原因见 7.2 节。`Compiler::Result::train_cg` / `infer_cg` 需要 move 进 `named_graphs_` 供 `TaskBase::run()` 使用，由 `named_graphs_` 保证生命周期。

**修改 `GpuExecTable`**：

```cpp
#ifdef TR_USE_CUDA
struct GpuExecTable {
    // variant_graphs[variant_idx][rank][slot] → cudaGraphExec_t
    std::array<std::vector<std::vector<cudaGraphExec_t>>, 6> variant_graphs;
    std::vector<int> device_ids;
};
GpuExecTable gpu_exec_;
std::vector<float*> lr_pinned_;
#endif
```

### 5.2 `on_prepare()` 修改

替换原有单 spec 调用：

```cpp
// ===== 旧代码（删除） =====
// CompileSpec spec = CompileSpec::from_global_registry();
// auto result = Compiler::compile(plan, spec, plan_config_, initializer_);

// ===== 新代码 =====
CompileSpec base_spec = CompileSpec::from_global_registry();
auto& gr = GlobalRegistry::instance();

// 生成 5 个 variant specs
std::vector<CompileSpec> variant_specs_vec;

// v1: train_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_train_batch_size();
    variant_specs_vec.push_back(s);
}
// v2: train_lowres
{
    CompileSpec s = base_spec;
    s.actual_resolution = gr.get_train_sample_resolution_end();
    variant_specs_vec.push_back(s);
}
// v3: train_lowres_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_train_batch_size();
    s.actual_resolution = gr.get_train_sample_resolution_end();
    variant_specs_vec.push_back(s);
}
// v4: val_base
{
    CompileSpec s = base_spec;
    s.actual_resolution = gr.val_sample_resolution();
    variant_specs_vec.push_back(s);
}
// v5: val_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_val_batch_size();
    s.actual_resolution = gr.val_sample_resolution();
    variant_specs_vec.push_back(s);
}

auto result = Compiler::compile(plan, base_spec, plan_config_,
                                initializer_, variant_specs_vec);

// 保存所有 variant 的 MemoryPlan（先提取，再 move cg）
for (size_t i = 0; i < result.variants.size() && i < 6; ++i) {
    variant_memory_plans_[i] = std::move(result.variants[i].memory_plan);
}
variant_specs_[0] = base_spec;
for (size_t i = 0; i < variant_specs_vec.size() && i < 5; ++i) {
    variant_specs_[i + 1] = variant_specs_vec[i];
}

memory_plan_ptr_ = std::move(variant_memory_plans_[0]);
active_memory_plan_ = memory_plan_ptr_.get();

// 后续代码不变：add_graph → train_cg_ / infer_cg_ 指向 named_graphs_
```

### 5.3 `build_graph_atlas()` 重写

**核心原则**：复用 `computation_graph.h` 中已有的 `is_shape_invariant_graph()` / `is_train_graph()` / `is_inference_graph()`，不重新定义。

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

    for (size_t v = 0; v < 6; ++v) {
        const MemoryPlan* mp = variant_memory_plans_[v].get();
        if (!mp) continue;

        bool is_train_var = (v <= 3);
        ShapeId shape_id = variant_specs_[v].get_shape_id();

        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);

            // ===== 交叉槽位隔离 =====
            if (is_train_var) {
                // Train variant：不需要纯推理图（INF_* 且非 shape-invariant）
                if (is_inference_graph(gid) && !is_shape_invariant_graph(gid)) continue;
            } else {
                // Val variant：不需要 train-only shape 相关图
                if (is_train_graph(gid) && !is_inference_graph(gid) &&
                    !is_shape_invariant_graph(gid)) continue;
            }

            // ===== 选择 ComputationGraph =====
            const ComputationGraph* cg = is_inference_graph(gid) ? infer_cg_ : train_cg_;
            if (!cg || cg->nodes(gid).empty()) continue;

            // ===== 填 Slot =====
            auto& sl = atlas.slot(v, gi);
            sl.cg = cg;
            sl.stream_kind = stream_for(gid);

            if (is_shape_invariant_graph(gid)) {
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

### 5.4 `build_exec_table()` 重写

```cpp
void DeepLearningTask::build_exec_table() {
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;

    const int K = num_gpus_;
    gpu_exec_.device_ids.resize(K);
    for (int rank = 0; rank < K; ++rank) {
        gpu_exec_.device_ids[rank] = context(rank).device_id();
    }

    auto resolve = [&](GraphId gid, int rank, size_t v) -> cudaGraphExec_t {
        int32_t idx = captured_result_.atlas.index(v, gid);
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

    for (size_t v = 0; v < 6; ++v) {
        gpu_exec_.variant_graphs[v].resize(K);
        for (int rank = 0; rank < K; ++rank) {
            auto& g = gpu_exec_.variant_graphs[v][rank];
            g.resize(S(GraphSlot::COUNT), nullptr);

            // 所有 variant 都有的图
            g[S(GraphSlot::XFER_A)]            = resolve(GraphId::TRANSFER_A, rank, v);
            g[S(GraphSlot::XFER_B)]            = resolve(GraphId::TRANSFER_B, rank, v);
            g[S(GraphSlot::ZERO_GRAD)]         = resolve(GraphId::ZERO_GRAD, rank, v);
            g[S(GraphSlot::DEEP_ALLREDUCE)]    = resolve(GraphId::DEEP_COMM, rank, v);
            g[S(GraphSlot::FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM, rank, v);
            g[S(GraphSlot::WEIGHT_UPDATE)]     = resolve(GraphId::OPTIMIZER, rank, v);
            g[S(GraphSlot::EMA_UPDATE)]        = resolve(GraphId::EMA_UPDATE, rank, v);
            g[S(GraphSlot::CAST_DEEP_GRAD)]    = resolve(GraphId::CAST_DEEP_GRAD_FP16_TO_FP32, rank, v);
            g[S(GraphSlot::CAST_FIRST_GRAD)]   = resolve(GraphId::CAST_FIRST_GRAD_FP16_TO_FP32, rank, v);
            g[S(GraphSlot::NAN_CHECK_GRAD_SCALE)] = resolve(GraphId::NAN_CHECK_AND_GRAD_SCALING, rank, v);
            g[S(GraphSlot::CAST_MAIN)]         = resolve(GraphId::CAST_MAIN_FP32_TO_FP16, rank, v);
            g[S(GraphSlot::CLEAR_METRICS)]     = resolve(GraphId::CLEAR_METRICS, rank, v);

            if (v <= 3) {  // Train variants
                g[S(GraphSlot::FWD_BWD_DEEP_A)]    = resolve(GraphId::DEEP_FWD_BWD, rank, v);
                g[S(GraphSlot::FWD_BWD_DEEP_B)]    = resolve(GraphId::DEEP_FWD_BWD, rank, v);
                g[S(GraphSlot::FIRST_LAYER_BWD_A)] = resolve(GraphId::FIRST_LAYER_BWD_A, rank, v);
                g[S(GraphSlot::FIRST_LAYER_BWD_B)] = resolve(GraphId::FIRST_LAYER_BWD_B, rank, v);
                g[S(GraphSlot::FIRST_LAYER_FWD_A)] = resolve(GraphId::FIRST_LAYER_FWD_A, rank, v);
                g[S(GraphSlot::FIRST_LAYER_FWD_B)] = resolve(GraphId::FIRST_LAYER_FWD_B, rank, v);
                g[S(GraphSlot::ACCUM_METRICS)]     = resolve(GraphId::ACCUM_METRICS, rank, v);
                g[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)] = resolve(GraphId::ACCUM_METRICS_TRAIN_LAST, rank, v);
            } else {  // Val variants
                g[S(GraphSlot::INF_MAIN_A)]       = resolve(GraphId::INF_MAIN_A, rank, v);
                g[S(GraphSlot::INF_MAIN_B)]       = resolve(GraphId::INF_MAIN_B, rank, v);
                g[S(GraphSlot::INF_EMA_A)]        = resolve(GraphId::INF_EMA_A, rank, v);
                g[S(GraphSlot::INF_EMA_B)]        = resolve(GraphId::INF_EMA_B, rank, v);
                g[S(GraphSlot::ACCUM_METRICS)]    = resolve(GraphId::ACCUM_METRICS, rank, v);
                g[S(GraphSlot::ACCUM_METRICS_VAL_LAST)] = resolve(GraphId::ACCUM_METRICS_VAL_LAST, rank, v);
                g[S(GraphSlot::VAL_RESULT_ALLREDUCE)]  = resolve(GraphId::VAL_RESULT_COMM, rank, v);
            }
        }
    }

    // 日志与校验（略，同 LRF2）
#endif
}
```

### 5.5 `run_train_epoch_gpu()` 重写

**性能核心**：线程 lambda 开始处预解析所有 handles，循环内零查表。

```cpp
#ifdef TR_USE_CUDA
float DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    // ===== Epoch-level variant 选择 =====
    bool is_begin_res = (get_current_train_resolution() ==
                         registry.train_sample_resolution_begin());
    size_t v_base = is_begin_res ? 0 : 2;  // begin_res → v0/v1, end_res → v2/v3
    size_t v_last = is_begin_res ? 1 : 3;

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

                // ===== PRE-RESOLVE ALL HANDLES (loop external) =====
                const auto& g_nb = gpu_exec_.variant_graphs[v_base][rank];
                const auto& g_lb = gpu_exec_.variant_graphs[v_last][rank];

                // Normal batch
                auto nb_xfer_a = g_nb[S(GraphSlot::XFER_A)];
                auto nb_xfer_b = g_nb[S(GraphSlot::XFER_B)];
                auto nb_deep   = g_nb[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto nb_fwd_a  = g_nb[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto nb_fwd_b  = g_nb[S(GraphSlot::FIRST_LAYER_FWD_B)];
                auto nb_bwd_a  = g_nb[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto nb_bwd_b  = g_nb[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto nb_zg     = g_nb[S(GraphSlot::ZERO_GRAD)];
                auto nb_dar    = g_nb[S(GraphSlot::DEEP_ALLREDUCE)];
                auto nb_far    = g_nb[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto nb_wu     = g_nb[S(GraphSlot::WEIGHT_UPDATE)];
                auto nb_cdg    = g_nb[S(GraphSlot::CAST_DEEP_GRAD)];
                auto nb_cfg    = g_nb[S(GraphSlot::CAST_FIRST_GRAD)];
                auto nb_ncg    = g_nb[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                auto nb_cm     = g_nb[S(GraphSlot::CAST_MAIN)];
                auto nb_accum  = g_nb[S(GraphSlot::ACCUM_METRICS)];
                auto nb_clear  = g_nb[S(GraphSlot::CLEAR_METRICS)];

                // Last batch (shape-dependent differ, invariant same)
                auto lb_deep   = g_lb[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto lb_fwd_a  = g_lb[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto lb_fwd_b  = g_lb[S(GraphSlot::FIRST_LAYER_FWD_B)];
                auto lb_bwd_a  = g_lb[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto lb_bwd_b  = g_lb[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto lb_accum_tl = g_lb[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];
                // shape-invariant 可用 nb_*（同一指针），也可用 g_lb 取

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up    = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_c3    = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

                auto sync_comp = [&]() { cudaStreamSynchronize(s_c1);
                                         cudaStreamSynchronize(s_c2);
                                         cudaStreamSynchronize(s_c3); };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr   = [&]() { cudaStreamSynchronize(s_trans); };

                TransferStation* ts = nullptr;
                for (int w = 0; w < 200; ++w) {
                    ts = static_cast<TransferStation*>(registry.transfer_station_ptr(rank));
                    if (ts) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

                float lr; float* lr_dev_ptr = static_cast<float*>(ctx.ptr_at(lr_dtensor_id_));
                bool frozen = is_first_layer_frozen();

                // Batch 0 prefetch (always full batch)
                ts->wait_buffer_readable(0);
                if (nb_xfer_a) cudaGraphLaunch(nb_xfer_a, s_trans); sync_tr();
                ts->set_buffer_readable(0, false); ts->set_buffer_writeable(0, true);
                if (using_amp && nb_cm) { cudaGraphLaunch(nb_cm, s_up); sync_up(); }
                if (nb_clear) cudaGraphLaunch(nb_clear, s_up);

                // Normal batches (0 .. batches-2)
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;
                    auto g_fwd  = from_a ? nb_fwd_a : nb_fwd_b;
                    auto g_deep = from_a ? nb_deep : nb_deep;  // A/B same handle
                    auto g_xfer_n = from_a ? nb_xfer_b : nb_xfer_a;
                    auto g_first  = from_a ? nb_bwd_a : nb_bwd_b;

                    if (nb_zg) cudaGraphLaunch(nb_zg, s_up);
                    if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                    sync_comp(); sync_up();

                    ts->wait_buffer_readable(next_buf);
                    if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
                    sync_comp(); sync_tr();

                    if (nb_accum) cudaGraphLaunch(nb_accum, s_up); sync_up();
                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);

                    if (!frozen && g_first) cudaGraphLaunch(g_first, s_c1);
                    sync_comp();

                    if (using_amp && nb_cdg) { cudaGraphLaunch(nb_cdg, s_up); sync_up(); }
                    if (nb_dar) cudaGraphLaunch(nb_dar, s_up); sync_up();
                    if (using_amp && nb_cfg) { cudaGraphLaunch(nb_cfg, s_up); sync_up(); }
                    if (nb_far) cudaGraphLaunch(nb_far, s_up); sync_up();
                    if (using_amp && nb_ncg) { cudaGraphLaunch(nb_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batch);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (nb_wu) cudaGraphLaunch(nb_wu, s_up); sync_up();
                    if (using_amp && nb_cm) { cudaGraphLaunch(nb_cm, s_up); sync_up(); }
                }

                // Last batch
                {
                    bool last_a = ((batches - 1) % 2 == 0);
                    auto g_fwd_l  = last_a ? lb_fwd_a : lb_fwd_b;
                    auto g_deep_l = last_a ? lb_deep : lb_deep;
                    auto g_first_l = last_a ? lb_bwd_a : lb_bwd_b;

                    if (nb_zg) cudaGraphLaunch(nb_zg, s_up);  // shape-invariant, use nb
                    if (g_fwd_l) cudaGraphLaunch(g_fwd_l, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
                    sync_comp();

                    if (lb_accum_tl) cudaGraphLaunch(lb_accum_tl, s_up); sync_up();

                    if (!frozen && g_first_l) cudaGraphLaunch(g_first_l, s_c1);
                    sync_comp();

                    if (using_amp && nb_cdg) { cudaGraphLaunch(nb_cdg, s_up); sync_up(); }
                    if (nb_dar) cudaGraphLaunch(nb_dar, s_up); sync_up();
                    if (using_amp && nb_cfg) { cudaGraphLaunch(nb_cfg, s_up); sync_up(); }
                    if (nb_far) cudaGraphLaunch(nb_far, s_up); sync_up();
                    if (using_amp && nb_ncg) { cudaGraphLaunch(nb_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batches - 1);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (nb_wu) cudaGraphLaunch(nb_wu, s_up); sync_up();
                    if (using_amp && nb_cm) { cudaGraphLaunch(nb_cm, s_up); sync_up(); }
                }
            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank) if (exc[rank]) std::rethrow_exception(exc[rank]);

    // 从 accum_loss 读取 epoch 平均 loss
    float train_loss = 0.0f;
    if (active_memory_plan_) {
        const auto& b = active_memory_plan_->baseline();
        if (b.accum_loss >= 0) {
            const auto& dt = active_memory_plan_->get_dtensor(b.accum_loss);
            Tensor h = fetch_from_rank(dt, 0);
            float val = h.data<float>()[0];
            size_t total = registry.num_train_samples();
            if (total > 0) train_loss = val / static_cast<float>(total);
        }
    }
    return train_loss;
}
#endif
```

### 5.6 `run_val_epoch_gpu()` 重写

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
#ifdef TR_USE_CUDA
    (void)validate_ema;
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    int val_batches = registry.get_val_steps();

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads; threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);
                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // PRE-RESOLVE
                const auto& g_vb = gpu_exec_.variant_graphs[4][rank];
                const auto& g_vl = gpu_exec_.variant_graphs[5][rank];

                auto vb_xfer_a = g_vb[S(GraphSlot::XFER_A)];
                auto vb_xfer_b = g_vb[S(GraphSlot::XFER_B)];
                auto vb_inf_a  = g_vb[S(GraphSlot::INF_MAIN_A)];
                auto vb_inf_b  = g_vb[S(GraphSlot::INF_MAIN_B)];
                auto vb_accum  = g_vb[S(GraphSlot::ACCUM_METRICS)];
                auto vb_clear  = g_vb[S(GraphSlot::CLEAR_METRICS)];
                auto vb_comm   = g_vb[S(GraphSlot::VAL_RESULT_ALLREDUCE)];

                auto vl_inf_a  = g_vl[S(GraphSlot::INF_MAIN_A)];
                auto vl_inf_b  = g_vl[S(GraphSlot::INF_MAIN_B)];
                auto vl_accum  = g_vl[S(GraphSlot::ACCUM_METRICS_VAL_LAST)];

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up = ..., s_trans = ..., s_c1 = ...;  // 同前
                auto sync_comp = ...; auto sync_up = ...; auto sync_tr = ...;

                TransferStation* ts = ...;  // 同前获取

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
                    if (g_xfer) cudaGraphLaunch(g_xfer, s_trans); sync_tr();
                    if (g_inf) cudaGraphLaunch(g_inf, s_c1); sync_comp();
                    if (g_accum) cudaGraphLaunch(g_accum, s_up); sync_up();

                    ts->set_buffer_readable(buf, false);
                    ts->set_buffer_writeable(buf, true);
                }

                sync_up();
                if (vb_comm) cudaGraphLaunch(vb_comm, s_up); sync_up();
            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank) if (exc[rank]) std::rethrow_exception(exc[rank]);

    // 读取 accum_* 按 num_val_samples 平均
    float avg_loss = 0, avg_top1 = 0, avg_top5 = 0;
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
    LOG_INFO << "[VAL] loss=" << avg_loss << " top1=" << avg_top1 * 100.0f
             << "% top5=" << avg_top5 * 100.0f << "%";
    return {avg_loss, avg_top1, avg_top5};
#else
    (void)validate_ema; return {0.0f, 0.0f, 0.0f};
#endif
}
```

### 5.7 `run_train_epoch_cpu()` 重写

```cpp
float DeepLearningTask::run_train_epoch_cpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();

    TransferStation* ts = nullptr;
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break; std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!ts) TR_DEVICE_ERROR("TransferStation not ready for CPU path");

    DeviceContext& ctx = context(0);
    const auto& atlas = captured_result_.atlas;
    const auto& graphs = captured_result_.graphs;

    // Variant 选择
    bool is_begin_res = (get_current_train_resolution() ==
                         registry.train_sample_resolution_begin());
    size_t v_base = is_begin_res ? 0 : 2;
    size_t v_last = is_begin_res ? 1 : 3;

    auto idx_for = [&atlas](GraphId gid, size_t v) -> int32_t {
        return atlas.index(v, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    // Pre-resolve shape-invariant (variant 0 即可，去重后相同)
    int32_t idx_xfer_a = idx_for(GraphId::TRANSFER_A, 0);
    int32_t idx_xfer_b = idx_for(GraphId::TRANSFER_B, 0);
    int32_t idx_zg     = idx_for(GraphId::ZERO_GRAD, 0);
    int32_t idx_dar    = idx_for(GraphId::DEEP_COMM, 0);
    int32_t idx_far    = idx_for(GraphId::FIRST_COMM, 0);
    int32_t idx_opt    = idx_for(GraphId::OPTIMIZER, 0);
    int32_t idx_clear  = idx_for(GraphId::CLEAR_METRICS, 0);

    // Pre-resolve shape-dependent: normal batch
    int32_t idx_deep_nb  = idx_for(GraphId::DEEP_FWD_BWD, v_base);
    int32_t idx_fwd_a_nb = idx_for(GraphId::FIRST_LAYER_FWD_A, v_base);
    int32_t idx_fwd_b_nb = idx_for(GraphId::FIRST_LAYER_FWD_B, v_base);
    int32_t idx_bwd_a_nb = idx_for(GraphId::FIRST_LAYER_BWD_A, v_base);
    int32_t idx_bwd_b_nb = idx_for(GraphId::FIRST_LAYER_BWD_B, v_base);
    int32_t idx_accum_nb = idx_for(GraphId::ACCUM_METRICS, v_base);

    // Pre-resolve shape-dependent: last batch
    int32_t idx_deep_lb  = idx_for(GraphId::DEEP_FWD_BWD, v_last);
    int32_t idx_fwd_a_lb = idx_for(GraphId::FIRST_LAYER_FWD_A, v_last);
    int32_t idx_fwd_b_lb = idx_for(GraphId::FIRST_LAYER_FWD_B, v_last);
    int32_t idx_bwd_a_lb = idx_for(GraphId::FIRST_LAYER_BWD_A, v_last);
    int32_t idx_bwd_b_lb = idx_for(GraphId::FIRST_LAYER_BWD_B, v_last);
    int32_t idx_accum_lb = idx_for(GraphId::ACCUM_METRICS_TRAIN_LAST, v_last);

    int32_t loss_id = active_memory_plan_->baseline().loss;
    void* loss_ptr = loss_id >= 0 ? ctx.ptr_at(loss_id) : nullptr;
    void* lr_ptr   = ctx.ptr_at(lr_dtensor_id_);
    bool frozen = is_first_layer_frozen();

    // CLEAR_METRICS
    if (idx_clear >= 0) launch(idx_clear);

    // Batch 0 prefetch
    ts->wait_buffer_readable(0);
    launch(idx_xfer_a);
    ts->set_buffer_readable(0, false); ts->set_buffer_writeable(0, true);

    // Normal batches
    for (int batch = 0; batch < batches - 1; ++batch) {
        bool from_a = (batch % 2 == 0);
        int next_buf = from_a ? 1 : 0;

        launch(idx_zg);
        launch(from_a ? idx_fwd_a_nb : idx_fwd_b_nb);
        ts->wait_buffer_readable(next_buf);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_nb);
        launch(from_a ? idx_xfer_b : idx_xfer_a);
        ts->set_buffer_readable(next_buf, false); ts->set_buffer_writeable(next_buf, true);

        // ACCUM_METRICS: 需先写入 local_batch_size 标量
        const auto& bl = active_memory_plan_->baseline();
        if (bl.local_batch_size >= 0) {
            *static_cast<int32_t*>(ctx.ptr_at(bl.local_batch_size)) =
                registry.get_local_batch_size();
        }
        if (idx_accum_nb >= 0) launch(idx_accum_nb);

        if (!frozen) launch(from_a ? idx_bwd_a_nb : idx_bwd_b_nb);
        launch(idx_dar);
        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batch);
        launch(idx_far); launch(idx_opt);
    }

    // Last batch
    {
        bool last_a = ((batches - 1) % 2 == 0);
        launch(idx_zg);
        launch(last_a ? idx_fwd_a_lb : idx_fwd_b_lb);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep_lb);

        // ACCUM_METRICS_TRAIN_LAST: 先写入 last_train_batch_size 标量
        const auto& bl = active_memory_plan_->baseline();
        if (bl.last_train_batch_size >= 0) {
            *static_cast<int32_t*>(ctx.ptr_at(bl.last_train_batch_size)) =
                registry.get_last_train_batch_size();
        }
        if (idx_accum_lb >= 0) launch(idx_accum_lb);

        if (!frozen) launch(last_a ? idx_bwd_a_lb : idx_bwd_b_lb);
        launch(idx_dar);
        *static_cast<float*>(lr_ptr) = fetch_lr_for_batch(batches - 1);
        launch(idx_far); launch(idx_opt);
    }

    // 返回 epoch 平均 loss（从 accum_loss）
    float train_loss = 0.0f;
    const auto& bl = active_memory_plan_->baseline();
    if (bl.accum_loss >= 0) {
        Tensor h = fetch_from_rank(active_memory_plan_->get_dtensor(bl.accum_loss), 0);
        float val = h.data<float>()[0];
        size_t total = registry.num_train_samples();
        if (total > 0) train_loss = val / static_cast<float>(total);
    }
    return train_loss;
}
```

### 5.8 `run_val_epoch_cpu()` 重写

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

    // Pre-resolve
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
        if (ts) break; std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
        float batch_loss = loss_ptr ? *static_cast<float*>(loss_ptr) : 0.0f;
        float batch_top1 = top1_ptr ? *static_cast<float*>(top1_ptr) : 0.0f;
        float batch_top5 = top5_ptr ? *static_cast<float*>(top5_ptr) : 0.0f;

        acc_loss += batch_loss * static_cast<double>(bs);
        acc_top1 += batch_top1 * static_cast<double>(bs);
        acc_top5 += batch_top5 * static_cast<double>(bs);

        ts->set_buffer_readable(buf, false); ts->set_buffer_writeable(buf, true);
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

## 六、渐进式分辨率集成

```cpp
// run_train_epoch_gpu() / run_train_epoch_cpu() 开头
bool is_begin_res = (get_current_train_resolution() ==
                     registry.train_sample_resolution_begin());
size_t v_base = is_begin_res ? 0 : 2;   // begin → v0/v1, end → v2/v3
size_t v_last = is_begin_res ? 1 : 3;
```

| Epoch 阶段 | `get_current_train_resolution()` | v_base | v_last |
|---|---|---|---|
| 早期 (< boundary) | `train_sample_resolution_begin()` | 0 | 1 |
| 后期 (≥ boundary) | `train_sample_resolution_end()` | 2 | 3 |

验证阶段始终使用 v4/v5，与训练分辨率无关。

---

## 七、性能分析

| 指标 | 当前实现 | 本方案 | 说明 |
|---|---|---|---|
| Batch 循环内 variant 切换 | 无（错误地同 graph） | **0 开销** | 循环外预解析为局部变量 |
| Graph handle 访问 | `g_tab[S(slot)]` 数组索引 | **直接局部变量** | 编译器可能放入寄存器 |
| Last batch 条件判断 | 无 | **仅 A/B 双缓冲条件** | `from_a ? nb_fwd_a : nb_fwd_b` |
| 多 RANK | `gpu_exec_.graphs[rank]` | **`gpu_exec_.variant_graphs[v][rank]`** | 每 rank 独立预解析 |
| Captured graph 内存 | ~23 张 | **28~50 张** | shape 无关图零膨胀，shape 相关按 ShapeId 计数 |
| 编译时间 | ~1s | **~1.5s** | cuDNN warmup 线性增长，系数小 |

---

## 八、验证计划

| 阶段 | 验证内容 | 通过标准 |
|---|---|---|
| **1. 编译期** | `result.variants.size() == 6`，各 variant MemoryPlan 非空 | 编译通过 |
| **2. Atlas 结构** | pre_capture 日志：shape-invariant 图 6 variant `captured_idx` 相同；shape-dependent 图不同 ShapeId 的 `captured_idx` 不同 | `reused > 0` |
| **3. CIFAR-10 回归** | `test_dl_full_gpu/amp`，3 epochs | loss/top1 与修复前一致（CIFAR-10 整除，variant 退化） |
| **4. Last batch 场景** | 修改配置使 `train/val` 不能被 batch 整除，运行 GPU/AMP | 无异常，last batch 使用正确 variant |
| **5. CPU 路径** | `test_dl_full --cpu` | 指标与 GPU 对齐 |
| **6. 渐进分辨率** | 启用 progressive resize，跨 boundary epoch | 早期用 v0/v1，后期用 v2/v3 |

---

## 九、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| `variant_memory_plans_` slot_bytes 不一致 | 极低 | 灾难性 | `compute_max_slot_bytes` 已跨变体取 max；可断言关键 DTensor offset |
| Val variant 误填 train-only shape 图 | 低 | 正确性错误 | `build_graph_atlas()` 中 `is_train_graph && !is_inference_graph && !is_shape_invariant` → `continue` |
| `add_graph()` 后 `train_cg_` 指针失效 | 低 | 崩溃 | `named_graphs_` 生命周期覆盖整个训练，`train_cg_` 指向其内部，安全 |
| CPU ACCUM_METRICS 标量未写入 | 中 | 指标错误 | 本方案明确在 launch 前写入 `local_batch_size` / `last_train_batch_size` |
| Multi-variant 捕获时间增加 | 中 | 编译慢 | 预期 < 0.5s 增加；可并行 capture |

---

## 十、实施检查清单（Checklist）

| # | 文件 | 位置 | 改动内容 | 状态 |
|---|------|------|----------|------|
| 1 | `deep_learning_task.h` | private 成员 | 新增 `variant_memory_plans_[6]`, `variant_specs_[6]` | ⬜ |
| 2 | `deep_learning_task.h` | `GpuExecTable` | 改为 `variant_graphs[6]` | ⬜ |
| 3 | `deep_learning_task.h` | `on_prepare()` | 生成 5 个 variant_specs，保存 MemoryPlan | ⬜ |
| 4 | `deep_learning_task.cpp` | `build_graph_atlas()` | 全量 6-variant 填表 | ⬜ |
| 5 | `deep_learning_task.cpp` | `build_exec_table()` | per-variant-per-rank 构建 | ⬜ |
| 6 | `deep_learning_task.cpp` | `run_train_epoch_gpu()` | 预解析 + variant 切换 | ⬜ |
| 7 | `deep_learning_task.cpp` | `run_val_epoch_gpu()` | 预解析 + variant 切换 | ⬜ |
| 8 | `deep_learning_task.cpp` | `run_train_epoch_cpu()` | variant 切换 + ACCUM_METRICS + 标量写入 | ⬜ |
| 9 | `deep_learning_task.cpp` | `run_val_epoch_cpu()` | variant 切换 + 加权平均 | ⬜ |

**不需要修改的文件**：
- `src/graph/compiler.cpp` — 已支持 variant_specs
- `src/graph/captured_graph.cpp` — 去重机制已就绪
- `src/backend/ops/*` — kernel 无需修改（variant 从根因解决）

---

## 十一、总结

### 11.1 相比 LRF1/LRF2/LRF3 的改进

| 方面 | LRF1 | LRF2 | LRF3 | **MYP2** |
|---|---|---|---|---|
| `compiled_result_` 成员 | ✅ 添加 | ✅ 添加 | ✅ 添加 | **❌ 不添加**（避免与 `named_graphs_` 冲突） |
| `is_shape_invariant` 辅助函数 | 重新定义匿名函数 | 复用已有 | 重新定义 lambda | **复用 `computation_graph.h` 已有函数** |
| ACCUM_METRICS* 去重 | 未明确 | 未明确 | 错误声称三者共享同一张图 | **明确三者各自独立**（不同 GraphId） |
| 渐进分辨率映射 | v0=base(res_end) | v0=base(res_begin) ✅ | v0=base(res_end) | **v0=base(res_begin) ✅**（与 `CompileSpec::from_global_registry()` 一致） |
| CPU 标量写入 | 提到但未完整 | 未提及 | 提到前提 | **完整写入 `local_batch_size` / `last_train_batch_size`** |
| 实施检查清单 | 有接入点清单 | 无 | 有阶段验证 | **有完整 9 项 checklist + 状态列** |

### 11.2 核心设计决策

1. **不添加 `compiled_result_`**：`Compiler::Result::train_cg` 必须 move 到 `named_graphs_` 供 `TaskBase::run()` 使用。改为只保存 `variant_memory_plans_` + `variant_specs_`。
2. **复用已有判断函数**：`is_shape_invariant_graph()` / `is_train_graph()` / `is_inference_graph()` 已在 `computation_graph.h` 中定义，不再重复。
3. **ACCUM_METRICS* 独立**：三个不同 GraphId，各自在 6 variant 中共享，但三者之间不互相共享。
4. **CPU 标量显式写入**：GPU 路径通过 GraphNode `input_ids` 自动绑定 `batch_size_ptr`；CPU 路径需在 `launch()` 前手动写入标量 DTensor。
5. **循环外预解析**：所有 `cudaGraphExec_t` / `int32_t idx` 在 batch 循环外部解析为局部变量，循环内零查表。

**改动量**：~400 行，2 个文件（1 头文件 + 1 cpp）。不修改 Compiler / pre_capture / kernel。
