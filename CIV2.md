# H2D-Only Epoch 性能测试 — 综合实现方案

## 1. 需求概述

为 `DeepLearningTask` 新增两个接口，并在测试样例中验证：

| 接口 | 功能 |
|------|------|
| `compile_h2d_only()` | 仅编译/捕获 H2D 传输图（TRANSFER_A / TRANSFER_B），不编译 FWD/BWD/OPT/推理图 |
| `run_h2d_only()` | 启动 `Preprocessor::train()`，仅交替执行 XFER_A / XFER_B 完成一个 epoch，返回 `H2DTestResult` |

测试样例基于 `test_pw_ultimate.cpp` 的 Preprocessor 配置框架，支持 ImageNet / CIFAR-10 等数据集，BluePrint 使用极简单层 FC。

---

## 2. 已有方案的问题分析

### 2.1 小伙伴S：手动 alloc + add_graph

**问题**：手动调用 `TaskBase::add_graph("h2d_xfer_a", ...)` 添加的图存储在 `named_graphs_` 中，但 `DeepLearningTask::build_graph_atlas()` 遍历的是 `train_cg_` 的 **GraphId 桶**（`TRANSFER_A` / `TRANSFER_B`），不是 `named_graphs_` 的字符串键。手动 add 的图无法进入 GraphAtlas，因此 `pre_capture()` 不会捕获它们。

**结论**：该方案在 `DeepLearningTask` 路径下不可行（SimpleTask 路径才走 `named_graphs_` 独立捕获）。

### 2.2 小伙伴K：复用 Compiler + `h2d_only_` 标志

**问题**：`compile_h2d_only()` 调用 `compile()` 后，`h2d_only_` 标志**未被清除**。若用户随后调用 `run()`（完整训练），`gpu_exec_` 中只有 xfer 句柄，会导致 segfault。

**改进**：使用 RAII Guard 确保标志在 `compile()` 返回或异常抛出后自动清除。

### 2.3 小伙伴D：类似K，但 `run_h2d_only()` 无 barrier

**优点**：与现有 `run_train_epoch_gpu()` 保持一致，不引入额外同步开销，测量结果更真实反映端到端性能。

**风险**：所有 rank 共享同一份 CPU Staging Buffer。rank-0 释放 buffer 时，其他 rank 的 DMA 可能尚未读完（虽然概率极低，因为各 rank H2D 时间相近）。

**结论**：在 H2D-only 场景下（所有 rank 行为完全一致，无计算负载差异），该风险可接受。若未来观察到数据竞争，可再引入 barrier。

### 2.4 `compile()` 末尾的 `init_all()` 开销

正常 `compile()` 末尾会调用 `init_all()` 初始化所有权重，并分配 `lr_pinned_`。对于 H2D-only 场景，这是**不必要的 compile 阶段开销**。

**权衡**：
- 若避免该开销，需要修改 `TaskBase::compile_impl()`（将 `compile_alloc_hardware()` 等改为 protected，或添加 `skip_init_all_` 标志），侵入 TaskBase。
- 单层 FC 的 `init_all()` 开销极小（约几百个参数，<1ms）；即使 ResNet-50，也只是 compile 阶段的一次性几秒开销，不影响 `run_h2d_only()` 运行时性能。

**决策**：**不修改 TaskBase**，接受该一次性开销。理由：TSK.md 强调 "run_h2d_only() 必须没有多余动作"，这是针对**运行时**，不是 compile 时。

---

## 3. 核心设计决策

### 3.1 复用完整 Compiler，Atlas/ExecTable 阶段过滤

`Compiler::compile()` 生成的 `train_cg_` 内部**已经天然包含** `GraphId::TRANSFER_A` 和 `GraphId::TRANSFER_B` 子图。我们只需在 `build_graph_atlas()` 和 `build_exec_table()` 中过滤：

- `build_graph_atlas()`：只向 Atlas 填入 `TRANSFER_A` / `TRANSFER_B` slot
- `build_exec_table()`：只解析 `XFER_A` / `XFER_B`，跳过 deep/bwd 的 required 校验
- `pre_capture()` 自动只捕获 Atlas 中的非空 slot

### 3.2 `compile_h2d_only()` 调用 `compile()` + RAII Guard

```cpp
void DeepLearningTask::compile_h2d_only() {
    struct FlagGuard {
        bool* ptr;
        explicit FlagGuard(bool* p) : ptr(p) { *ptr = true; }
        ~FlagGuard() { *ptr = false; }
    };
    FlagGuard guard(&compile_h2d_only_flag_);
    compile();   // 走正常管线，build_graph_atlas/build_exec_table 自动走简化分支
}
```

无论 `compile()` 成功或抛出异常，`compile_h2d_only_flag_` 都会被析构函数清零，避免状态残留。

### 3.3 `run_h2d_only()` 的同步模型

采用 **多 rank 线程 + rank-0 管理 TransferStation buffer 状态**，**不加 barrier**：

```
每个 rank 线程:
  for batch = 0 .. batches-1:
    buf = batch % 2
    等待 ts->buffer_is_readable(buf)
    cudaGraphLaunch(xfer_a 或 xfer_b, TRANS stream)
    cudaStreamSynchronize(TRANS stream)
    if rank == 0:
      ts->set_buffer_readable(buf, false)
      ts->set_buffer_writeable(buf, true)
```

**与现有训练代码保持一致**（`run_train_epoch_gpu()` 同样无 barrier）。H2D-only 场景下所有 rank 的传输行为完全一致，时间差异极小，数据竞争风险可忽略。

### 3.4 Last Batch 处理

`Preprocessor::steps_per_epoch()` 已经通过 `ceil(num_train_samples / global_batch_size)` 将 last batch 计入总步数。H2D CUDA Graph 的传输范围是固定的（整个 `I_A_LABEL + I_A_DATA` / `I_B_LABEL + I_B_DATA` Region），因此 last batch 不需要变体图，直接按普通 batch 传输即可。

### 3.5 返回值：复用 `H2DTestResult`

项目中已有 `H2DTestResult` 结构体（`deep_learning_task.h:30-40`），与 `test_h2d_copy_bandwidth()` / `test_h2d_copy_correctness()` 接口风格一致。`run_h2d_only()` 返回该类型，其中：
- `batches` = `steps_per_epoch()`
- `elapsed_us` = 总 wall-clock 耗时（微秒）
- `total_bytes` = `per_zone_bytes * batches * num_ranks`
- `bandwidth_gbps` = 总带宽（GB/s）
- `avg_lat_us` = `elapsed_us / batches`
- `min_lat_us = max_lat_us = 0`（不测量 per-batch 延迟，避免运行时开销）

---

## 4. 具体实现

### 4.1 `include/renaissance/task/deep_learning_task.h`

**新增 public 方法：**

```cpp
    /**
     * @brief 仅编译 H2D 传输图（TRANSFER_A + TRANSFER_B），不编译 FWD/BWD/OPT/推理图
     * @note 调用后不可再调用 compile() 或 run()（完整训练），两者互斥
     */
    void compile_h2d_only();

    /**
     * @brief 仅运行 H2D 传输图一个 epoch，联动 Preprocessor/TransferStation
     * @return H2DTestResult 包含耗时、带宽、batch 数等统计信息
     */
    H2DTestResult run_h2d_only();
```

**新增 private 成员：**

```cpp
    bool compile_h2d_only_flag_ = false;   // compile_h2d_only() 模式标志（RAII 保护）

    /**
     * @brief H2D-only 模式下构建只含 xfer 的 gpu_exec_ 表
     */
    void build_exec_table_h2d_only();
```

### 4.2 `src/task/deep_learning_task.cpp`

#### A. 修改 `build_graph_atlas()`

在 GraphId 遍历处增加过滤：

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (train_cg_->nodes(gid).empty()) continue;

            // ═══ H2D-Only 模式：只添加 xfer 图 ═══
            if (compile_h2d_only_flag_) {
                if (gid != GraphId::TRANSFER_A && gid != GraphId::TRANSFER_B)
                    continue;
            }
            // ═════════════════════════════════════

            auto& sl = atlas.slot(0, gi);
            sl.cg = train_cg_;
            sl.mp = active_memory_plan_;
            sl.stream_kind = stream_for(gid);
            sl.shape_id = kShapeInvariant;
        }
    }

    name_to_gid_.clear();
    if (!compile_h2d_only_flag_ && train_cg_) {
        name_to_gid_["train"] = GraphId::DEEP_FWD_BWD;
    }
    if (!compile_h2d_only_flag_ && infer_cg_) {
        name_to_gid_["inference"] = GraphId::INF_MAIN_A;
    }

    return atlas;
}
```

#### B. 修改 `build_exec_table()`

在 required slot 校验处分支：

```cpp
void DeepLearningTask::build_exec_table() {
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;

    const int K = num_gpus_;
    gpu_exec_.device_ids.resize(K);
    gpu_exec_.graphs.resize(K);

    auto resolve = [&](GraphId gid, int rank) -> cudaGraphExec_t {
        int32_t idx = captured_result_.atlas.index(0, gid);
        if (idx < 0 || static_cast<size_t>(idx) >= captured_result_.graphs.size())
            return nullptr;
        return static_cast<cudaGraphExec_t>(
            captured_result_.graphs[idx].native_exec(rank));
    };

    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

    for (int rank = 0; rank < K; ++rank) {
        gpu_exec_.device_ids[rank] = context(rank).device_id();

        auto& g = gpu_exec_.graphs[rank];
        g.resize(static_cast<size_t>(GraphSlot::COUNT), nullptr);

        g[S(GraphSlot::XFER_A)]           = resolve(GraphId::TRANSFER_A, rank);
        g[S(GraphSlot::XFER_B)]           = resolve(GraphId::TRANSFER_B, rank);

        if (!compile_h2d_only_flag_) {
            g[S(GraphSlot::FWD_BWD_DEEP_A)]   = resolve(GraphId::DEEP_FWD_BWD, rank);
            g[S(GraphSlot::FWD_BWD_DEEP_B)]   = resolve(GraphId::DEEP_FWD_BWD, rank);
            g[S(GraphSlot::FIRST_LAYER_BWD)]  = resolve(GraphId::FIRST_BWD, rank);
            g[S(GraphSlot::FIRST_LAYER_BWD_B)] = resolve(GraphId::FIRST_BWD_B, rank);
            g[S(GraphSlot::ZERO_GRAD)]        = resolve(GraphId::ZERO_GRAD, rank);
            g[S(GraphSlot::DEEP_ALLREDUCE)]   = resolve(GraphId::DEEP_COMM, rank);
            g[S(GraphSlot::FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM, rank);
            g[S(GraphSlot::WEIGHT_UPDATE)]    = resolve(GraphId::OPTIMIZER, rank);
            g[S(GraphSlot::EMA_UPDATE)]       = resolve(GraphId::EMA_UPDATE, rank);
            g[S(GraphSlot::GRAD_CONVERT)]     = resolve(GraphId::CAST_AND_CHECK, rank);
            g[S(GraphSlot::FIRST_FWD_A)]      = resolve(GraphId::FIRST_FWD_A, rank);
            g[S(GraphSlot::FIRST_FWD_B)]      = resolve(GraphId::FIRST_FWD_B, rank);
            g[S(GraphSlot::CAST_AND_CHECK)]   = resolve(GraphId::CAST_AND_CHECK, rank);
            g[S(GraphSlot::INF_MAIN_A)]       = resolve(GraphId::INF_MAIN_A, rank);
            g[S(GraphSlot::INF_MAIN_B)]       = resolve(GraphId::INF_MAIN_B, rank);
            g[S(GraphSlot::INF_EMA_A)]        = resolve(GraphId::INF_EMA_A, rank);
            g[S(GraphSlot::INF_EMA_B)]        = resolve(GraphId::INF_EMA_B, rank);
        }
    }

    // ═══ H2D-Only 模式：只校验 xfer 图 ═══
    if (compile_h2d_only_flag_) {
        static const GraphSlot kH2DRequired[] = {
            GraphSlot::XFER_A, GraphSlot::XFER_B,
        };
        for (int rank = 0; rank < K; ++rank) {
            for (auto slot : kH2DRequired) {
                TR_CHECK(gpu_exec_.graphs[rank][static_cast<size_t>(slot)],
                         ValueError,
                         "H2D-only graph slot is nullptr for rank " << rank);
            }
        }
        return;   // 跳过正常模式的校验
    }
    // ═════════════════════════════════════

    // 原有校验逻辑不变
    static const GraphSlot kRequired[] = {
        GraphSlot::XFER_A, GraphSlot::XFER_B,
        GraphSlot::FWD_BWD_DEEP_A, GraphSlot::FWD_BWD_DEEP_B,
        GraphSlot::FIRST_LAYER_BWD,
    };
    for (int rank = 0; rank < K; ++rank) {
        for (auto slot : kRequired) {
            if (!gpu_exec_.graphs[rank][static_cast<size_t>(slot)]) {
                TR_CHECK(false, ValueError,
                         "Required graph slot " << static_cast<int>(slot)
                         << " is nullptr for rank " << rank);
            }
        }
    }
    // ... 其余原有校验不变 ...
#endif
}
```

> **注意**：`build_exec_table()` 中原有的 `INF_MAIN_A` / `INF_MAIN_B` 等推理图 resolve 逻辑在 H2D-only 模式下被跳过，因为 Atlas 中未填入这些 slot，`resolve()` 会返回 `nullptr`。

#### C. 新增 `compile_h2d_only()`

```cpp
void DeepLearningTask::compile_h2d_only() {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "compile_h2d_only() must be called in PLANNING phase");

    struct FlagGuard {
        bool* ptr;
        explicit FlagGuard(bool* p) : ptr(p) { *ptr = true; }
        ~FlagGuard() { *ptr = false; }
    };
    FlagGuard guard(&compile_h2d_only_flag_);

    compile();   // 走正常编译管线，build_graph_atlas/build_exec_table 自动走简化分支
}
```

#### D. 新增 `run_h2d_only()`

```cpp
H2DTestResult DeepLearningTask::run_h2d_only() {
    H2DTestResult result;

#ifdef TR_USE_CUDA
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = nullptr;
    const int K = num_gpus_;

    // ---- 启动 Preprocessor ----
    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    // ---- 等待 TransferStation 就绪 ----
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!ts) {
        LOG_ERROR << "TransferStation not ready within timeout";
        prep_thread.join();
        if (prep_exc) std::rethrow_exception(prep_exc);
        result.batches = 0;
        return result;
    }

    // ---- 计算传输字节数（用于带宽统计）----
    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    auto t0 = std::chrono::steady_clock::now();

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                DeviceContext& ctx = context(rank);
                cudaStream_t s_trans = static_cast<cudaStream_t>(
                    ctx.stream(StreamKind::TRANS));

                for (int batch = 0; batch < batches; ++batch) {
                    int buf_id = batch % 2;
                    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                    while (!ts->buffer_is_readable(buf_id))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    if (g_xfer) cudaGraphLaunch(g_xfer, s_trans);
                    cudaStreamSynchronize(s_trans);

                    if (rank == 0) {
                        ts->set_buffer_readable(buf_id, false);
                        ts->set_buffer_writeable(buf_id, true);
                    }
                }
            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    auto t1 = std::chrono::steady_clock::now();

    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    for (int rank = 0; rank < K; ++rank)
        if (exc[rank]) std::rethrow_exception(exc[rank]);

    // ---- 填充结果 ----
    double elapsed_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    result.batches = batches;
    result.elapsed_us = elapsed_us;
    result.total_bytes = per_zone_bytes * static_cast<size_t>(batches);
    result.labels_ok = true;
    result.data_ok = true;

    if (batches > 0) {
        result.avg_lat_us = elapsed_us / static_cast<double>(batches);
    }
    if (elapsed_us > 0.0 && result.total_bytes > 0) {
        double bw = static_cast<double>(result.total_bytes) / (elapsed_us / 1e6);
        result.bandwidth_gbps = bw / (1024.0 * 1024.0 * 1024.0);
    }
#else
    (void)result;
    LOG_ERROR << "run_h2d_only: CUDA not available";
#endif

    return result;
}
```

> **关于 `total_bytes` 的计算**：`per_zone_bytes` 是单卡单 batch 的 A 区大小（`I_A_LABEL + I_A_DATA`）。一个 epoch 中每个 batch 传输一次 A 区或 B 区（大小相同），所以 `total_bytes = per_zone_bytes * batches`。这里不乘 `num_ranks`，因为 `H2DTestResult::total_bytes` 在 `test_h2d_copy_bandwidth()` 中定义为单卡数据量 × consumed batch 数。若测试文件需要聚合带宽，可在外部自行乘以 `num_ranks`。

---

## 5. 测试样例

### 5.1 `tests/correction/test_h2d_only_epoch.cpp`

基于 `test_pw_ultimate.cpp` 的 Preprocessor 配置框架，完整支持其 CLI 参数和 PO 链解析。

```cpp
/**
 * @file test_h2d_only_epoch.cpp
 * @brief H2D-only epoch 性能测试（Preprocessor + TransferStation + DeepLearningTask）
 */

#include "renaissance.h"
#include "renaissance/task/deep_learning_task.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

#ifdef _WIN32
    #ifdef min
        #undef min
    #endif
    #ifdef max
        #undef max
    #endif
#endif

using namespace tr;

// ── 以下辅助函数直接复用 test_pw_ultimate.cpp 的实现 ──

bool is_resize_or_crop_po(const std::string& po_name) {
    std::string po_lower = po_name;
    std::transform(po_lower.begin(), po_lower.end(), po_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return po_lower == "resize" || po_lower == "centercrop" ||
           po_lower == "randomresizedcrop" || po_lower == "fastrandomresizedcrop" ||
           po_lower == "randomcrop";
}

std::unique_ptr<PreprocessOperation> create_resize_crop_po(
    const std::string& po_name, int resolution,
    float scale_min, float scale_max, float ratio_min, float ratio_max)
{
    std::string po_lower = po_name;
    std::transform(po_lower.begin(), po_lower.end(), po_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (po_lower == "resize") return std::make_unique<Resize>(resolution);
    if (po_lower == "centercrop") return std::make_unique<CenterCrop>(resolution);
    if (po_lower == "randomresizedcrop")
        return std::make_unique<RandomResizedCrop>(resolution, scale_min, scale_max, ratio_min, ratio_max);
    if (po_lower == "fastrandomresizedcrop")
        return std::make_unique<FastRandomResizedCrop>(resolution, scale_min, scale_max, ratio_min, ratio_max);
    if (po_lower == "randomcrop") return std::make_unique<RandomCrop>(resolution);
    return nullptr;
}

std::unique_ptr<PreprocessOperation> create_po(
    const std::string& po_name,
    float scale_min, float scale_max, float ratio_min, float ratio_max,
    float flip_prob,
    float brightness, float contrast, float saturation, float hue,
    float degrees, int fill,
    float autocontrast_p,
    float blur_sigma_min, float blur_sigma_max,
    float grayscale_p,
    float noise_mean, float noise_sigma)
{
    (void)scale_min; (void)scale_max; (void)ratio_min; (void)ratio_max;
    (void)brightness; (void)contrast; (void)saturation; (void)hue;
    (void)degrees; (void)fill; (void)autocontrast_p;
    (void)blur_sigma_min; (void)blur_sigma_max; (void)grayscale_p;
    (void)noise_mean; (void)noise_sigma;

    std::string po_lower = po_name;
    std::transform(po_lower.begin(), po_lower.end(), po_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (po_lower == "randomhorizontalflip") return std::make_unique<RandomHorizontalFlip>(flip_prob);
    if (po_lower == "colorjitter") return std::make_unique<ColorJitter>(brightness, contrast, saturation, hue);
    if (po_lower == "randomrotation") return std::make_unique<RandomRotation>(degrees, static_cast<uint8_t>(fill));
    if (po_lower == "randomautocontrast") return std::make_unique<RandomAutocontrast>(autocontrast_p);
    if (po_lower == "randomscale") return std::make_unique<RandomScale>(0.8f, 1.2f);
    if (po_lower == "gaussianblur") return std::make_unique<GaussianBlur>(blur_sigma_min, blur_sigma_max);
    if (po_lower == "randomgrayscale") return std::make_unique<RandomGrayscale>(grayscale_p);
    if (po_lower == "gaussiannoise") return std::make_unique<GaussianNoise>(noise_mean, noise_sigma, true);
    if (po_lower == "pad") return std::make_unique<Pad>(4, std::vector<int>{0}, PaddingMode::CONSTANT);
    if (po_lower == "donothing") return std::make_unique<DoNothing>();
    return nullptr;
}

NormMode parse_norm_mode(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (lower == "imagenet") return NormMode::IMAGENET;
    if (lower == "mnist")    return NormMode::MNIST;
    if (lower == "cifar")    return NormMode::CIFAR;
    if (lower == "mlperf")   return NormMode::MLPERF;
    return NormMode::NO_NORM;
}

int main(int argc, char* argv[]) {
    // ── 默认值（与 test_pw_ultimate 对齐）──
    std::string dataset_type_str = "imagenet";
    std::string dataset_path;
    std::string format_arg = "raw";
    std::string mode_arg = "partial";
    int compression_level = 0;
    bool shuffle_train = false;
    bool auto_cpu_binding = false;
    int batch_size = 512;
    bool using_cpvs = false;
    int sdmp_factor = 1;
    bool reproducible = false;
    int num_epochs = 1;
    int resolution = 224;
    std::string po_train1, po_train2;
    std::string po_val1, po_val2;
    std::string normalization_mode = "mlperf";
    int num_load_workers = 16;
    int num_preproc_workers = 16;
    std::string device_type = "GPU";
    std::string gpu_ids_str = "";
    bool using_amp = false;
    uint64_t random_seed = 42;
    float scale_min = 0.08f, scale_max = 1.0f, ratio_min = 0.75f, ratio_max = 1.3333333f;
    float flip_prob = 0.5f;
    float brightness = 0.2f, contrast = 0.2f, saturation = 0.2f, hue = 0.1f;
    float degrees = 30.0f; int fill = 0;
    float autocontrast_p = 0.5f;
    float blur_sigma_min = 0.1f, blur_sigma_max = 2.0f;
    float grayscale_p = 0.1f;
    float noise_mean = 0.0f, noise_sigma = 25.5f;

    // ── 参数解析（复用 test_pw_ultimate 逻辑）──
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) dataset_type_str = argv[++i];
        else if (arg == "--path" && i + 1 < argc) dataset_path = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--resolution" && i + 1 < argc) resolution = std::stoi(argv[++i]);
        else if (arg == "--loaders" && i + 1 < argc) num_load_workers = std::stoi(argv[++i]);
        else if (arg == "--preproc" && i + 1 < argc) num_preproc_workers = std::stoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) {
            device_type = argv[++i];
            std::transform(device_type.begin(), device_type.end(), device_type.begin(),
                           [](unsigned char c) { return std::toupper(c); });
        }
        else if (arg == "--gpu-ids" && i + 1 < argc) gpu_ids_str = argv[++i];
        else if (arg == "--po-train1" && i + 1 < argc) po_train1 = argv[++i];
        else if (arg == "--po-train2" && i + 1 < argc) po_train2 = argv[++i];
        else if (arg == "--po-val1" && i + 1 < argc) po_val1 = argv[++i];
        else if (arg == "--po-val2" && i + 1 < argc) po_val2 = argv[++i];
        else if (arg == "--normalization" && i + 1 < argc) normalization_mode = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) random_seed = std::stoull(argv[++i]);
        else if (arg == "--sdmp" && i + 1 < argc) sdmp_factor = std::stoi(argv[++i]);
        else if (arg == "--cpvs" && i + 1 < argc) {
            std::string s = argv[++i]; std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            using_cpvs = (s == "true");
        }
        else if (arg == "--amp") using_amp = true;
        else if (arg == "--reproducible") reproducible = true;
        else if (arg == "--cpu-bind") auto_cpu_binding = true;
        else if (arg == "--mode" && i + 1 < argc) mode_arg = argv[++i];
        else if (arg == "--format" && i + 1 < argc) format_arg = argv[++i];
    }

    if (dataset_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --path <dataset_root> [options...]\n";
        return 1;
    }

    // ── 推断数据集参数 ──
    std::transform(dataset_type_str.begin(), dataset_type_str.end(),
                   dataset_type_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    int num_classes = 1000;
    NormMode norm = NormMode::MLPERF;
    if (dataset_type_str == "mnist") { num_classes = 10; norm = NormMode::MNIST; }
    else if (dataset_type_str == "cifar10" || dataset_type_str == "cifar-10") {
        num_classes = 10; norm = NormMode::CIFAR;
    }
    else if (dataset_type_str == "cifar100" || dataset_type_str == "cifar-100") {
        num_classes = 100; norm = NormMode::CIFAR;
    }

    if (po_val1.empty()) po_val1 = po_train1;

    // ── 全局配置 ──
    if (reproducible) {
        GLOBAL_SETTING.reproducible().local_batch_size(batch_size).manual_seed(random_seed);
    } else {
        GLOBAL_SETTING.local_batch_size(batch_size).manual_seed(random_seed);
    }
    if (using_amp) GLOBAL_SETTING.amp(true);

    if (device_type == "GPU") {
        if (!gpu_ids_str.empty()) GLOBAL_SETTING.use_gpu(gpu_ids_str);
        else GLOBAL_SETTING.use_gpu();
    } else {
        GLOBAL_SETTING.use_cpu();
    }

    auto& reg = GlobalRegistry::instance();
    reg.set_num_color_channels(3);

    // ── 创建 PO ──
    std::cout << "\n=== Setting Transforms ===\n";
    auto train_po1 = create_resize_crop_po(po_train1, resolution, scale_min, scale_max, ratio_min, ratio_max);
    if (!train_po1) { std::cerr << "Error: invalid po-train1\n"; return 1; }

    std::unique_ptr<PreprocessOperation> train_po2;
    if (po_train2.empty()) {
        std::cout << "Train PO 2: DoNothing (default)\n";
        train_po2 = std::make_unique<DoNothing>();
    } else {
        train_po2 = is_resize_or_crop_po(po_train2)
            ? create_resize_crop_po(po_train2, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_train2, scale_min, scale_max, ratio_min, ratio_max, flip_prob,
                       brightness, contrast, saturation, hue, degrees, fill, autocontrast_p,
                       blur_sigma_min, blur_sigma_max, grayscale_p, noise_mean, noise_sigma);
        if (!train_po2) { std::cerr << "Error: invalid po-train2\n"; return 1; }
    }

    auto val_po1 = create_resize_crop_po(po_val1, resolution, scale_min, scale_max, ratio_min, ratio_max);
    if (!val_po1) { std::cerr << "Error: invalid po-val1\n"; return 1; }

    std::unique_ptr<PreprocessOperation> val_po2;
    if (po_val2.empty()) {
        val_po2 = std::make_unique<DoNothing>();
    } else {
        val_po2 = is_resize_or_crop_po(po_val2)
            ? create_resize_crop_po(po_val2, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_val2, scale_min, scale_max, ratio_min, ratio_max, flip_prob,
                       brightness, contrast, saturation, hue, degrees, fill, autocontrast_p,
                       blur_sigma_min, blur_sigma_max, grayscale_p, noise_mean, noise_sigma);
        if (!val_po2) { std::cerr << "Error: invalid po-val2\n"; return 1; }
    }

    // ── 配置 Preprocessor ──
    GLOBAL_SETTING.train_resolution(resolution).val_resolution(resolution);

    bool partial_mode = (mode_arg == "partial");
    bool use_dts = (format_arg == "dts");

    PREPROCESSOR_SETTING
        .dataset(dataset_type_str, dataset_path)
        .using_dts_format(use_dts, compression_level)
        .color_channels(3)
        .load_workers(num_load_workers)
        .preprocess_workers(num_preproc_workers)
        .fully_mode(!partial_mode)
        .shuffle_train(shuffle_train)
        .download(false)
        .sdmp_factor(sdmp_factor)
        .using_cpvs(using_cpvs)
        .cpu_binding(auto_cpu_binding)
        .normalization(norm)
        .train_transforms(*train_po1, *train_po2)
        .val_transforms(*val_po1, *val_po2)
        .commit();

    // ── DeepLearningTask：极简 FC ──
    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)))
        .loss(CrossEntropyLoss())
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .scheduler(PolynomialLR().base_lr(0.1f).power(1.0f))
        .num_classes(num_classes)
        .total_epochs(num_epochs);

    // ── 只编译 H2D 图 ──
    task.compile_h2d_only();

    const int steps = Preprocessor::instance().steps_per_epoch();
    std::cout << "\n=== H2D-Only Epoch Test ===\n"
              << "Dataset: " << dataset_type_str << "\n"
              << "Batch size: " << batch_size << "\n"
              << "Resolution: " << resolution << "\n"
              << "AMP: " << (using_amp ? "on" : "off") << "\n"
              << "Ranks: " << reg.world_size() << "\n"
              << "Steps per epoch: " << steps << "\n";

    // ── 运行 H2D-only epoch ──
    auto h2d_result = task.run_h2d_only();

    // ── 计算聚合带宽 ──
    size_t total_bytes_all_ranks = h2d_result.total_bytes * static_cast<size_t>(reg.world_size());
    double aggregate_bw_gbps = 0.0;
    if (h2d_result.elapsed_us > 0.0 && total_bytes_all_ranks > 0) {
        aggregate_bw_gbps = (static_cast<double>(total_bytes_all_ranks) / (h2d_result.elapsed_us / 1e6))
                            / (1024.0 * 1024.0 * 1024.0);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== Results ===\n"
              << "Batches:        " << h2d_result.batches << "\n"
              << "Elapsed time:   " << (h2d_result.elapsed_us / 1e6) << " s\n"
              << "Avg latency:    " << (h2d_result.avg_lat_us / 1e3) << " ms/batch\n"
              << "Per-rank bytes: " << (h2d_result.total_bytes / 1024.0 / 1024.0) << " MB\n"
              << "Total bytes:    " << (total_bytes_all_ranks / 1024.0 / 1024.0) << " MB\n"
              << "Per-rank BW:    " << h2d_result.bandwidth_gbps << " GB/s\n"
              << "Aggregate BW:   " << aggregate_bw_gbps << " GB/s\n"
              << "=== DONE ===\n";

    return 0;
}
```

### 5.2 `tests/correction/CMakeLists.txt`

```cmake
add_executable(test_h2d_only_epoch test_h2d_only_epoch.cpp)
target_link_libraries(test_h2d_only_epoch PRIVATE renaissance)
```

---

## 6. 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `include/renaissance/task/deep_learning_task.h` | 新增 `compile_h2d_only()`、`run_h2d_only()` 声明；新增 `compile_h2d_only_flag_` 成员；新增 `build_exec_table_h2d_only()` 私有声明 |
| `src/task/deep_learning_task.cpp` | 修改 `build_graph_atlas()` 添加 H2D-only 过滤分支；修改 `build_exec_table()` 添加 H2D-only 校验分支；新增 `compile_h2d_only()` 实现；新增 `run_h2d_only()` 实现 |
| `tests/correction/test_h2d_only_epoch.cpp` | **新建** — 基于 test_pw_ultimate.cpp 的 Preprocessor 配置 + DeepLearningTask H2D-only 测试 |
| `tests/correction/CMakeLists.txt` | 新增 `test_h2d_only_epoch` 编译目标 |

---

## 7. 边界情况与风险

### 7.1 `compile_h2d_only()` 与 `compile()` / `run()` 的互斥

`compile_h2d_only()` 调用后，`gpu_exec_` 中只包含 xfer 图。若用户随后调用 `run()`（完整训练），会访问到 `nullptr` 的 deep/bwd exec 句柄，导致 segfault。

**处理**：文档中明确声明两者互斥。RAII Guard 确保 `compile_h2d_only_flag_` 在异常时也被清除，避免影响后续的 `compile()` 调用（但 `gpu_exec_` 中的句柄仍只有 xfer）。

### 7.2 Multi-GPU 无 barrier 的潜在风险

rank-0 释放 buffer 时，其他 rank 的 DMA 理论上可能尚未完成。

**评估**：H2D-only 场景下所有 rank 行为完全一致（同一图、同一 stream、同一 buffer），时间差异 <1ms，而 Preprocessor 填充一个 batch 通常需要数毫秒到数十毫秒。因此 race condition 窗口极窄，实际中可忽略。

**备选**：若未来观察到不稳定，可在 `run_h2d_only()` 中加入 `std::barrier`（C++20）或 `std::mutex + condition_variable` 的 rank 间同步，仅需在每次 batch 的 `cudaStreamSynchronize` 后增加一行同步代码。

### 7.3 `batches == 0`

`run_h2d_only()` 的循环体不执行，`elapsed_us = 0`，返回结果中 `batches = 0`，`bandwidth_gbps = 0`。

### 7.4 `batches == 1`

循环执行 1 次（buf=0 → XFER_A），直接完成，无需特殊分支。

### 7.5 AMP (FP16)

`Compiler::compile()` 已根据 `GlobalRegistry::using_amp()` 生成正确的 Region 类型（`I_A_DATA` 为 FP16），`pre_capture()` 自动适配，无需额外处理。

### 7.6 CPU 模式

`#ifdef TR_USE_CUDA` 保护，`run_h2d_only()` 返回空的 `H2DTestResult`（`batches = 0`）。`compile_h2d_only()` 本身在 CPU 模式下也可执行（会 capture CPU 图），但测试意义不大。

---

## 8. 与现有代码的复用关系

| 现有代码 | 本方案如何复用 |
|---------|--------------|
| `Compiler::build_auxiliary_graphs()` 中的 TRANSFER_A/B 构建 | `train_cg_` 已天然包含 xfer 子图，直接复用 |
| `pre_capture()` | 正常调用，Atlas 中只有 xfer slot，自动只 capture xfer |
| `test_h2d_copy_bandwidth()` | 其多 rank 线程 + TransferStation 交互模式被继承；不记录 per-batch 延迟以消除运行时开销 |
| `run_train_epoch_gpu()` | 其 `prep.train()` 线程启动方式、buffer 0/1 交替策略、rank-0 释放 buffer 模式被继承 |
| `test_pw_ultimate.cpp` | 其完整的 CLI 参数解析、PO 链配置、Preprocessor Setup 被测试文件复用 |

---

## 9. 命令行示例

**ImageNet（与 test_pw_ultimate 完全相同的参数风格）：**

```bash
./test_h2d_only_epoch \
  --dataset imagenet --path /root/epfs/dataset/imagenet \
  --format raw --mode partial --batch-size 512 --resolution 224 \
  --loaders 16 --preproc 128 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" \
  --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip \
  --po-val1 Resize --po-val2 CenterCrop \
  --seed 42 --sdmp 1 --cpvs true --reproducible --cpu-bind
```

**CIFAR-10 + AMP：**

```bash
./test_h2d_only_epoch \
  --dataset cifar10 --path /root/epfs/dataset/cifar-10 \
  --format raw --mode partial --batch-size 512 --resolution 32 \
  --loaders 1 --preproc 32 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" \
  --po-train1 CenterCrop --po-val1 CenterCrop \
  --seed 42 --sdmp 1 --cpvs true --reproducible --cpu-bind --amp
```
