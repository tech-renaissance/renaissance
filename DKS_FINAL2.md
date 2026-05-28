# Last Batch 根本修复方案 —— 完整实现变体捕获与运行时切换

> 基于 DKS2.md 的深入分析，本方案彻底根除 last batch 正确性 bug，实现框架设计意图中的 6 变体 GraphAtlas 机制。

---

## 一、问题根因（彻底版）

### 1.1 三层断裂

| 层级 | 设计意图 | 当前实现 | 断裂点 |
|---|---|---|---|
| **Compiler** | 6 变体编译（train_base/last/lowres/lowres_last + val_base/last），各自 MemoryPlan | `Compiler::compile()` 支持 6 变体，但 `on_prepare()` **只传 1 个 spec**，`result.variants` 实际只有 **1 个元素** | 未生成 last batch 的 MemoryPlan |
| **GraphAtlas** | 6×N 映射表，shape 相关图各自 slot + 各自 mp/shape_id，shape 无关图共享 | `build_graph_atlas()` **只填 variant 0**，且所有 slot 硬编码 `shape_id = kShapeInvariant`、`mp = active_memory_plan_` | 所有 graph 被当成 shape invariant，pre_capture 只捕获一次 |
| **运行时** | 根据 batch 特征动态选择 variant，launch 对应 captured graph | `build_exec_table()` 只解析 variant 0；`run_train_epoch_gpu()` 对所有 batch 使用**同一个 g_tab** | last batch 使用 grid=512 的 graph 处理 402 个样本 |

### 1.2 为什么"不切换 MemoryPlan 必定行不通"

`Compiler::create_memory_plans()` 中虽然使用 `max_slots` 保证了**跨变体 offset 一致性**（V4.20.2 设计），但 `DTensor::shape` 在各变体中是不同的：

- variant[0] train_base: `logits_dt.shape = {512, 32, 32, 64}`
- variant[1] train_last: `logits_dt.shape = {402, 32, 32, 64}`

`softmax_ce_op.cpp:L358-359`：
```cpp
const DTensor& logits_dt = mp.get_dtensor(ids_in[0]);
int batch = logits_dt.shape.n();   // ← capture 时从 mp 读取，固定进 CUDA Graph
```

如果 capture 时 `mp` 是 variant[0] 的 MemoryPlan，`batch=512`，grid=512，此值在 CUDA Graph replay 时**不可变**。即使运行时把 `batch_size_ptr` 标量改为 402，`batch`（grid 维度）仍然是 512。

**唯一正确的做法：为 variant[1] 单独 capture 一次 CUDA Graph，capture 前 `ctx.set_memory_plan(mp_variant1)`，使 `batch=402`，grid=402。**

`pre_capture()` 的 `capture_all_for_rank()` 已经内置了此机制（`L367: ctx.set_memory_plan(mp)`），问题只在于上游的 `build_graph_atlas()` 没有提供正确的 `mp` 和 `shape_id`。

---

## 二、修复目标

1. **Compiler 层**：`on_prepare()` 生成 5 个 variant specs，使 `Compiler::compile()` 产出完整的 6 变体
2. **Atlas 层**：`build_graph_atlas()` 为 6 变体全量填表，shape 相关图各自 mp + shape_id，shape 无关图共享
3. **Capture 层**：`pre_capture()` 自动去重 + 分别捕获，无需额外修改
4. **运行时 GPU 层**：`run_train_epoch_gpu()` / `run_val_epoch_gpu()` 根据 batch 类型选择 correct variant
5. **运行时 CPU 层**：`run_train_epoch_cpu()` / `run_val_epoch_cpu()` 同步支持 variant 切换
6. **去重效率**：shape 无关图（COMM/CAST/CLEAR/ACCUM 等）全局只捕获一次，不膨胀

---

## 三、修改文件清单

| 文件 | 修改内容 | 行数估算 |
|---|---|---|
| `include/renaissance/task/deep_learning_task.h` | ① 扩展 `GpuExecTable` 支持 6 variant<br>② 增加 `variant_memory_plans_`<br>③ `on_prepare()` 生成 variant specs | ~40 |
| `src/task/deep_learning_task.cpp` | ① 重写 `build_graph_atlas()`<br>② 重写 `build_exec_table()`<br>③ 修改 `run_train_epoch_gpu()`<br>④ 修改 `run_val_epoch_gpu()`<br>⑤ 修改 `run_train_epoch_cpu()`<br>⑥ 修改 `run_val_epoch_cpu()` | ~200 |
| `src/task/deep_learning_task.cpp` | 新增 `slot_to_graph_id()` 辅助映射 | ~30 |

> **不需要修改的文件**：`src/graph/compiler.cpp`、`src/graph/captured_graph.cpp`、`src/graph/pre_capture.cpp`、`src/backend/ops/*` —— 这些基础设施已完全支持多 variant，只需上层正确调用。

---

## 四、详细实现

### 4.1 `GpuExecTable` 扩展（`deep_learning_task.h`）

```cpp
struct GpuExecTable {
    // variant_graphs[variant_idx][rank][slot] → cudaGraphExec_t
    // variant_idx 0~5 对应：train_base/train_last/train_lowres/train_lowres_last/val_base/val_last
    std::array<std::vector<std::vector<cudaGraphExec_t>>, 6> variant_graphs;
    std::vector<int> device_ids;
};
```

### 4.2 `on_prepare()` 修改（`deep_learning_task.h` 内联实现）

当前代码：
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

// ===== 保存所有 variant 的 MemoryPlan =====
for (size_t i = 0; i < result.variants.size() && i < 6; ++i) {
    variant_memory_plans_[i] = std::move(result.variants[i].memory_plan);
}
memory_plan_ptr_ = std::move(variant_memory_plans_[0]);  // base 作为默认
active_memory_plan_ = memory_plan_ptr_.get();
```

> **说明**：`variant_memory_plans_` 是 `DeepLearningTask` 新增成员：
> ```cpp
> std::array<std::unique_ptr<MemoryPlan>, 6> variant_memory_plans_;
> ```
> 虽然 cross-variant offset 一致（`max_slots` 保证），但保留各 variant 的 MemoryPlan 是 Atlas 正确填表的必要条件。

### 4.3 `build_graph_atlas()` 重写（`deep_learning_task.cpp`）

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;
    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

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
        return atlas;
    }

    const size_t num_variants = variant_memory_plans_.size();
    
    for (size_t v = 0; v < num_variants; ++v) {
        const MemoryPlan* mp = variant_memory_plans_[v].get();
        if (!mp) continue;

        ShapeId shape_id = (v < 6 && variant_memory_plans_[v]) 
            ? ShapeId{mp->input_batch_size(), mp->input_resolution_h(), 
                      mp->input_resolution_w(), mp->input_channels()}
            : kShapeInvariant;
        // 更简单的做法：在 on_prepare() 中保存各 variant 的 CompileSpec，直接取 get_shape_id()
        // 这里假设 variant_memory_plans_ 已保存，shape_id 从 mp 推导
        // 实际上 mp 可能没有 batch_size/resolution 的 getter，更可靠的方式是 on_prepare 保存 specs：
        // std::array<CompileSpec, 6> variant_specs_;
        // shape_id = variant_specs_[v].get_shape_id();

        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            
            // ===== 适用性判定（train/val 变体隔离）=====
            bool is_train = is_train_graph(gid);
            bool is_infer = is_inference_graph(gid);
            bool is_train_var = (v <= 3);
            bool is_val_var   = (v >= 4);

            if (is_val_var && is_train) continue;
            if (is_train_var && is_infer) continue;

            // ===== 选择 ComputationGraph =====
            const ComputationGraph* cg = nullptr;
            if (is_train || is_transfer_graph(gid)) {
                cg = train_cg_;
            } else if (is_infer) {
                cg = infer_cg_;
            }
            if (!cg || cg->nodes(gid).empty()) continue;

            // ===== 填 Slot =====
            auto& sl = atlas.slot(v, gi);
            sl.cg = cg;
            sl.stream_kind = stream_for(gid);

            if (is_shape_invariant_graph(gid)) {
                // shape 无关图：全部变体共享 variant[0] 的 MemoryPlan + kShapeInvariant
                sl.mp = variant_memory_plans_[0].get();
                sl.shape_id = kShapeInvariant;
            } else {
                // shape 相关图：各自 MemoryPlan + 各自 ShapeId
                sl.mp = mp;
                sl.shape_id = shape_id;  // 需要正确构造
            }
        }
    }

    return atlas;
}
```

> **关于 ShapeId 的获取**：`MemoryPlan` 当前可能没有 `input_batch_size()` 等 getter。更可靠的做法是在 `on_prepare()` 中保存各 variant 的 `CompileSpec`：
> ```cpp
> std::array<CompileSpec, 6> variant_compile_specs_;
> // on_prepare 中填充
> // build_graph_atlas 中：sl.shape_id = variant_compile_specs_[v].get_shape_id();
> ```

### 4.4 `build_exec_table()` 重写（`deep_learning_task.cpp`）

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

    // 为所有 6 个 variant 预 build exec table
    for (size_t v = 0; v < 6; ++v) {
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
    for (size_t v = 0; v < 6; ++v) {
        for (int rank = 0; rank < K; ++rank) {
            const auto& g = gpu_exec_.variant_graphs[v][rank];
            LOG_INFO << "[EXEC-TABLE] v=" << v << " rank=" << rank
                     << " DEEP=" << (g[S(GraphSlot::FWD_BWD_DEEP_A)] ? "OK" : "NULL")
                     << " FWD_A=" << (g[S(GraphSlot::FIRST_LAYER_FWD_A)] ? "OK" : "NULL");
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

> **去重验证**：shape 无关图（如 ZERO_GRAD）在各 variant 中 `captured_idx` 相同（因为 `build_graph_atlas` 中全部指向 variant[0] 的 mp + kShapeInvariant，Key 相同）。`build_exec_table` 中 `resolve(..., v)` 返回的 `cudaGraphExec_t` 在各 variant 中也是同一个指针值。存储上多存了 5 个重复指针，但正确性无损，且避免了运行时条件判断。

### 4.5 `run_train_epoch_gpu()` 修改（`deep_learning_task.cpp`）

核心变化：在线程 lambda 内部引入 `variant_idx` 切换逻辑。

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

                // ===== 辅助：根据 batch 类型选择 variant =====
                auto get_g_tab = [&](bool is_last) -> const std::vector<cudaGraphExec_t>& {
                    size_t v = is_last ? 1 : 0;  // train_base=0, train_last=1
                    return gpu_exec_.variant_graphs[v][rank];
                };

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

                // ===== Batch 0 预传输（始终用 variant 0，因为 batch 0 总是 full batch）=====
                {
                    const auto& g_tab = get_g_tab(false);
                    auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                    auto g_clear_metrics = g_tab[S(GraphSlot::CLEAR_METRICS)];

                    ts->wait_buffer_readable(0);
                    if (g_xfer_a) cudaGraphLaunch(g_xfer_a, s_trans);
                    sync_tr();

                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);

                    if (using_amp) {
                        auto g_cm = g_tab[S(GraphSlot::CAST_MAIN)];
                        if (g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }
                    }
                    if (g_clear_metrics) cudaGraphLaunch(g_clear_metrics, s_up);
                }

                // ===== 统一循环：batch = 0 .. batches-2（normal batch）=====
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    const auto& g_tab = get_g_tab(false);  // normal batch → variant 0
                    auto g_fwd   = from_a ? g_tab[S(GraphSlot::FIRST_LAYER_FWD_A)]
                                          : g_tab[S(GraphSlot::FIRST_LAYER_FWD_B)];
                    auto g_deep  = from_a ? g_tab[S(GraphSlot::FWD_BWD_DEEP_A)]
                                          : g_tab[S(GraphSlot::FWD_BWD_DEEP_B)];
                    auto g_xfer_n = from_a ? g_tab[S(GraphSlot::XFER_B)]
                                           : g_tab[S(GraphSlot::XFER_A)];
                    auto g_zg = g_tab[S(GraphSlot::ZERO_GRAD)];
                    auto g_accum = g_tab[S(GraphSlot::ACCUM_METRICS)];
                    auto g_first = from_a ? g_tab[S(GraphSlot::FIRST_LAYER_BWD_A)]
                                          : g_tab[S(GraphSlot::FIRST_LAYER_BWD_B)];
                    auto g_dar = g_tab[S(GraphSlot::DEEP_ALLREDUCE)];
                    auto g_far = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                    auto g_wu  = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
                    auto g_cdg = g_tab[S(GraphSlot::CAST_DEEP_GRAD)];
                    auto g_cfg = g_tab[S(GraphSlot::CAST_FIRST_GRAD)];
                    auto g_ncg = g_tab[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                    auto g_cm  = g_tab[S(GraphSlot::CAST_MAIN)];

                    // Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD
                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
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

                    if (g_accum) cudaGraphLaunch(g_accum, s_up);
                    sync_up();

                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);

                    // Phase 3: BWD → CAST → COMM → CAST → COMM → NAN_CHECK → OPT → CAST
                    if (!frozen) {
                        if (g_first) cudaGraphLaunch(g_first, s_c1);
                    }
                    sync_comp();

                    if (using_amp && g_cdg) { cudaGraphLaunch(g_cdg, s_up); sync_up(); }
                    if (g_dar) cudaGraphLaunch(g_dar, s_up); sync_up();
                    if (using_amp && g_cfg) { cudaGraphLaunch(g_cfg, s_up); sync_up(); }
                    if (g_far) cudaGraphLaunch(g_far, s_up); sync_up();
                    if (using_amp && g_ncg) { cudaGraphLaunch(g_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batch);
                    (void)lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up); sync_up();
                    if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }
                }

                // ===== Last batch (batch = batches-1) =====
                {
                    bool last_a = ((batches - 1) % 2 == 0);
                    const auto& g_tab = get_g_tab(true);  // ← last batch → variant 1

                    auto g_fwd_l = last_a ? g_tab[S(GraphSlot::FIRST_LAYER_FWD_A)]
                                          : g_tab[S(GraphSlot::FIRST_LAYER_FWD_B)];
                    auto g_deep_l = last_a ? g_tab[S(GraphSlot::FWD_BWD_DEEP_A)]
                                           : g_tab[S(GraphSlot::FWD_BWD_DEEP_B)];
                    auto g_zg = g_tab[S(GraphSlot::ZERO_GRAD)];
                    auto g_accum_train_last = g_tab[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];
                    auto g_first = last_a ? g_tab[S(GraphSlot::FIRST_LAYER_BWD_A)]
                                          : g_tab[S(GraphSlot::FIRST_LAYER_BWD_B)];
                    auto g_dar = g_tab[S(GraphSlot::DEEP_ALLREDUCE)];
                    auto g_far = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                    auto g_wu  = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
                    auto g_cdg = g_tab[S(GraphSlot::CAST_DEEP_GRAD)];
                    auto g_cfg = g_tab[S(GraphSlot::CAST_FIRST_GRAD)];
                    auto g_ncg = g_tab[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                    auto g_cm  = g_tab[S(GraphSlot::CAST_MAIN)];

                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd_l) cudaGraphLaunch(g_fwd_l, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
                    sync_comp();

                    if (g_accum_train_last) cudaGraphLaunch(g_accum_train_last, s_up);
                    sync_up();

                    if (!frozen) {
                        if (g_first) cudaGraphLaunch(g_first, s_c1);
                    }
                    sync_comp();

                    if (using_amp && g_cdg) { cudaGraphLaunch(g_cdg, s_up); sync_up(); }
                    if (g_dar) cudaGraphLaunch(g_dar, s_up); sync_up();
                    if (using_amp && g_cfg) { cudaGraphLaunch(g_cfg, s_up); sync_up(); }
                    if (g_far) cudaGraphLaunch(g_far, s_up); sync_up();
                    if (using_amp && g_ncg) { cudaGraphLaunch(g_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batches - 1);
                    (void)lr;
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
}
#endif
```

> **关键变化**：
> 1. `get_g_tab(false)` 返回 variant 0（train_base）的 exec table
> 2. `get_g_tab(true)` 返回 variant 1（train_last）的 exec table
> 3. Batch 0 预传输也使用 variant 0（batch 0 总是 full batch）
> 4. last batch 的 `g_deep_l`、`g_fwd_l` 等从 variant 1 获取，grid=last_batch_size

### 4.6 `run_val_epoch_gpu()` 修改（`deep_learning_task.cpp`）

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
#ifdef TR_USE_CUDA
    // ... 前置代码不变 ...

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                // ... 设备设置不变 ...

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // ===== 辅助：根据 batch 类型选择 variant =====
                auto get_g_tab = [&](bool is_last) -> const std::vector<cudaGraphExec_t>& {
                    size_t v = is_last ? 5 : 4;  // val_base=4, val_last=5
                    return gpu_exec_.variant_graphs[v][rank];
                };

                // ... stream 初始化不变 ...

                if (g_clear_metrics) cudaGraphLaunch(g_clear_metrics, s_up);

                for (int batch = 0; batch < val_batches; ++batch) {
                    int buf = batch % 2;
                    bool is_last = (batch == val_batches - 1);
                    const auto& g_tab = get_g_tab(is_last);  // ← 动态切换

                    if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (top1_id >= 0) cudaMemsetAsync(ctx.ptr_at(top1_id), 0, sizeof(float), s_c1);
                    if (top5_id >= 0) cudaMemsetAsync(ctx.ptr_at(top5_id), 0, sizeof(float), s_c1);

                    ts->wait_buffer_readable(buf);

                    auto g_xfer = (buf == 0) ? g_tab[S(GraphSlot::XFER_A)]
                                              : g_tab[S(GraphSlot::XFER_B)];
                    if (g_xfer) cudaGraphLaunch(g_xfer, s_trans);
                    sync_tr();

                    auto g_inf = (buf == 0) ? g_tab[S(GraphSlot::INF_MAIN_A)]
                                             : g_tab[S(GraphSlot::INF_MAIN_B)];
                    if (g_inf) cudaGraphLaunch(g_inf, s_c1);
                    sync_comp();

                    auto g_accum = is_last ? g_tab[S(GraphSlot::ACCUM_METRICS_VAL_LAST)]
                                           : g_tab[S(GraphSlot::ACCUM_METRICS)];
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

### 4.7 CPU 路径修改

#### `run_train_epoch_cpu()`

```cpp
float DeepLearningTask::run_train_epoch_cpu() {
    // ... 前置不变 ...

    auto idx_for = [&atlas](GraphId gid, size_t variant_idx) -> int32_t {
        return atlas.index(variant_idx, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    // normal batches: variant 0
    for (int batch = 0; batch < batches - 1; ++batch) {
        bool from_a = (batch % 2 == 0);
        // ... 使用 idx_for(GraphId::DEEP_FWD_BWD, 0) ...
    }

    // last batch: variant 1
    {
        bool last_a = ((batches - 1) % 2 == 0);
        // ... 使用 idx_for(GraphId::DEEP_FWD_BWD, 1) ...
    }

    // ... 指标读取不变 ...
}
```

#### `run_val_epoch_cpu()`

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_cpu(bool validate_ema) {
    // ... 前置不变 ...

    auto idx_for = [&atlas](GraphId gid, size_t variant_idx) -> int32_t {
        return atlas.index(variant_idx, gid);
    };

    for (int batch = 0; batch < val_batches; ++batch) {
        int buf = batch % 2;
        size_t v = (batch == val_batches - 1) ? 5 : 4;  // val_base=4, val_last=5

        // ... 使用 idx_for(GraphId::INF_MAIN_A, v) ...
    }

    // 指标加权修正（小伙伴 D 指出的 bug）
    float avg_loss = static_cast<float>(acc_loss / registry.num_val_samples());
    float avg_top1 = static_cast<float>(acc_top1 / registry.num_val_samples());
    float avg_top5 = static_cast<float>(acc_top5 / registry.num_val_samples());

    LOG_INFO << "[VAL-CPU] loss=" << avg_loss << " top1=" << avg_top1 * 100.0f
             << "% top5=" << avg_top5 * 100.0f << "%";
    return {avg_loss, avg_top1, avg_top5};
}
```

> **同时修复 CPU 路径的两处指标 bug**（小伙伴 D 已指出）：
> 1. `run_train_epoch_cpu()` 当前只取最后一个 batch 的 loss，需要改为累加所有 batch（variant 切换后自然修复，因为需要遍历所有 batch）
> 2. `run_val_epoch_cpu()` 按 `val_batches` 除 → 改为按 `num_val_samples()` 除

---

## 五、去重机制验证

`pre_capture()` 的去重逻辑（`Key{cg, gid, shape_id}`）：

| GraphId | 是否 shape invariant | 各 variant 的 shape_id | 去重结果 |
|---|---|---|---|
| TRANSFER_A/B | ✅ | 全 `kShapeInvariant` | 6 变体共享 1 张图 |
| ZERO_GRAD | ✅ | 全 `kShapeInvariant` | 6 变体共享 1 张图 |
| DEEP_FWD_BWD | ❌ | v0:{512,32,32,3}, v1:{402,32,32,3}, v4:{512,32,32,3}, v5:{106,32,32,3} | 4 张独立图（train_base/last + val_base/last，train_lowres 如有则更多） |
| FIRST_LAYER_FWD_A | ❌ | 同上 | 4 张独立图 |
| FIRST_LAYER_FWD_B | ❌ | 同上 | 4 张独立图（与 A 不共享，因为 GraphId 不同） |
| FIRST_LAYER_BWD_A/B | ❌ | 同上 | 各 4 张独立图 |
| INF_MAIN_A/B | ❌ | 同上（val 专用） | 各 2 张（val_base/last） |
| COMM/CAST/ACCUM/CLEAR | ✅ | 全 `kShapeInvariant` | 全部共享 |

**去重后总 captured graph 数估算**：
- Shape 无关图：~15 个 GraphId × 1 次 = 15
- Shape 相关图：~8 个 GraphId × 4 个不同 shape = 32
- **总计 ≈ 47 张 captured graph**
- 相比当前（~20 张）增加约 2.3 倍，完全在可接受范围内

---

## 六、渐进式分辨率支持

本方案顺便完整支持了 progressive resolution：

| Epoch | 分辨率 | 训练 variant |
|---|---|---|
| 早期 | 低分辨率 | v2 (train_lowres) 或 v3 (train_lowres_last) |
| 后期 | 高分辨率 | v0 (train_base) 或 v1 (train_last) |

运行时只需在 `run_train_epoch_gpu()` 的 `get_g_tab` 前增加分辨率判断：
```cpp
auto get_g_tab = [&](bool is_last) -> const std::vector<cudaGraphExec_t>& {
    int res = registry.get_current_train_resolution();  // 或从 epoch 推导
    size_t v;
    if (res == gr.train_sample_resolution_begin()) {
        v = is_last ? 1 : 0;  // base res
    } else {
        v = is_last ? 3 : 2;  // lowres
    }
    return gpu_exec_.variant_graphs[v][rank];
};
```

> 当前 `run_train_epoch_gpu()` 尚未接入 progressive resolution 的 variant 选择，但本方案的结构已天然支持，只需一行判断即可扩展。

---

## 七、实施顺序与验证

### Phase 1：Compiler 层（1 小时）
1. 修改 `on_prepare()` 生成 5 个 variant specs
2. 传入 `Compiler::compile()` 的重载（带 `variant_specs` 参数）
3. 保存 `variant_memory_plans_`

**验证**：编译通过，`result.variants.size() == 6`

### Phase 2：Atlas 层（1 小时）
1. 重写 `build_graph_atlas()` 为 6 变体全量填表
2. 正确区分 `is_shape_invariant_graph`

**验证**：`pre_capture` 日志显示 captured graph 数量增加（~40-50），reused 计数 > 0（shape 无关图共享）

### Phase 3：运行时 GPU 层（2 小时）
1. 扩展 `GpuExecTable`
2. 重写 `build_exec_table()`
3. 修改 `run_train_epoch_gpu()` 和 `run_val_epoch_gpu()`

**验证**：
- `test_dl_full_gpu` 运行通过
- `test_dl_full_amp` 运行通过
- 检查 `pre_capture` 日志中 variant 1 的 DEEP_FWD_BWD captured graph 的 grid 参数（可通过 `cudaGraphGetNodes` + `cudaGraphKernelNodeGetParams` 读取 batch 维度）

### Phase 4：运行时 CPU 层（1 小时）
1. 修改 `run_train_epoch_cpu()` 和 `run_val_epoch_cpu()`
2. 修复指标加权 bug

**验证**：`test_dl_full` CPU 版本运行通过，loss/top1 与 GPU 对齐

### Phase 5：端到端测试（2 小时）
1. 故意构造不能整除的 batch size（如 train=50003，val=10007），验证 last batch 行为
2. 对比修复前后的 loss/accuracy 差异
3. 如果可能，dump last batch 的 softmax CE kernel grid 配置，确认 `grid.x == last_batch_size`

---

## 八、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| `variant_memory_plans_` 中某些 variant 的 `slot_bytes` 与 base 不同，导致 offset 不一致 | 低 | 灾难性（指针错乱） | `compute_max_slot_bytes` 已跨变体取 max，所有 variant 的 slot size 相同，offset 一致。如不放心，可在 `build_graph_atlas()` 中增加断言检查关键 DTensor（loss/accum）的 offset 一致性。 |
| `build_graph_atlas()` 中 val 变体误填了 train graph，导致 val 路径使用了 train 的 FWD/BWD graph | 低 | 正确性错误 | `is_train_graph()` / `is_inference_graph()` 判定 + `continue` 跳过已保证隔离。可额外增加断言：val variant 的 DEEP_FWD_BWD slot `captured_idx == -1`。 |
| 多 variant 导致 `pre_capture` 时间翻倍（~40 张图 vs ~20 张） | 中 | 编译时间增加 | 预期增加 < 2 秒（cuDNN warmup 是主要耗时，与 graph 数量线性相关但系数很小）。如需要可并行 capture（当前已支持 per-rank 并行）。 |
| CPU 路径的 `graphs[idx].launch()` 中 `CpuOp` 的 `input_shape` 在 capture 时固定，切换 variant 后 shape 不匹配 | 低 | CPU last batch 仍错误 | CPU capture 同样使用 `mp` 参数，`create_cpu_ops()` 中已读取 `mp.get_dtensor(id).shape`。只要 Atlas 填了正确的 mp，CPU 路径也会正确。 |
| `transfer_to_rank` 使用 `active_memory_plan_->get_dtensor().nbytes()`（base batch size），导致 last batch 多拷贝数据 | 中 | 性能损失 / 潜在正确性影响 | 多拷贝的数据在 last batch 的 FWD graph 中不会被处理（grid=last_batch_size），所以不影响正确性。性能影响极小（多拷贝 < 20% 数据，且只在每 epoch 的最后一个 batch）。可在后续优化 `transfer_to_rank` 使其接受 actual_batch_size 参数。 |

---

## 九、总结

本方案**不修改任何 kernel、不修改 Compiler 核心、不修改 pre_capture/capture 基础设施**，只修改 `DeepLearningTask` 的上层 orchestration 代码。核心改动量约 **250 行**，集中在 1 个头文件 + 1 个 cpp 文件。

**关键设计决策**：
1. **运行时切换 variant 而非增加 GraphSlot**：`GraphSlot` 保持 A/B double-buffering 语义，variant 切换在更高层处理，代码侵入最小
2. **pre-build per-variant exec table**：运行时 `O(1)` 数组索引，零 hash、零分支
3. **完全复用现有去重机制**：`is_shape_invariant_graph()` + `pre_capture()` 的 `Key{cg, gid, shape_id}` 自动完成去重，shape 无关图零膨胀
4. **CPU/GPU 路径统一**：两者都支持 variant 切换，同时修复 CPU 路径的指标加权 bug
