# H2D-Only Epoch 测试方案

## 1. 目标

为 `DeepLearningTask` 新增两个接口：
- `compile_h2d_only()` —— 只编译/捕获 H2D 传输图（TRANSFER_A / TRANSFER_B）。
- `run_h2d_only()` —— 只交替执行 H2D 传输图，联动 Preprocessor + TransferStation，跑完一个 epoch 的 train 数据，并返回 wall-clock 耗时（秒）。

并提供一个测试样例 `test_h2d_only_epoch.cpp`（基于 `test_pw_ultimate.cpp` 的 Preprocessor 配置框架），用于测量 **Preprocessor 与异步 H2D 传输对接后的每 epoch 总耗时**。

---

## 2. 核心设计决策

### 2.1 复用完整 Compiler，但只 Capture Xfer 图

`Compiler::compile()` 生成的 `ComputationGraph`（`train_cg_`）内部**已经天然包含** `GraphId::TRANSFER_A` 和 `GraphId::TRANSFER_B` 子图（见 `compiler.cpp:1422~1446`）。我们不需要修改 Compiler，只需要在 **GraphAtlas 构建**和 **ExecTable 构建**两个阶段做过滤：

- `build_graph_atlas()`：若处于 `h2d_only_` 模式，只向 Atlas 填入 `TRANSFER_A` / `TRANSFER_B` 两个 slot。
- `build_exec_table()`：若处于 `h2d_only_` 模式，只解析 `XFER_A` / `XFER_B` 两个 slot，跳过其余所有 slot 及“Required”校验。

这样 `pre_capture()` 只会捕获这两张图，`gpu_exec_` 中也只会有 xfer 的执行句柄。

### 2.2 `compile_h2d_only()` 直接调用 `compile()`

`TaskBase::compile_alloc_hardware()` 等关键步骤是 **private** 的，`DeepLearningTask` 无法从外部直接调用。因此 `compile_h2d_only()` 的实现策略是：

1. 设置标志位 `h2d_only_ = true;`
2. 调用 `compile();` 走正常深度学习编译管线
3. 在 `build_graph_atlas()` / `build_exec_table()` 内部根据 `h2d_only_` 走简化分支

> **副作用说明**：正常 `compile()` 末尾会执行 `init_all()`（初始化所有权重）和 `lr_pinned_` 分配。这些在 `h2d_only` 场景下虽非必需，但**仅发生在 compile 阶段**，不影响 `run_h2d_only()` 的运行时性能，且不会导致错误。为保持最小侵入，接受该一次性开销。

### 2.3 `run_h2d_only()` 的同步模型

参考 `test_h2d_copy_bandwidth()`（`deep_learning_task.cpp:1944~2103`）和 `run_train_epoch_gpu()`（`deep_learning_task.cpp:850~1423`），采用 **多 rank 线程 + rank-0 管理 TransferStation buffer 状态** 的模型。

**关键正确性保障**：所有 rank 共享同一份 CPU Staging Buffer（TransferStation）。若 rank-0 完成 H2D 后立即把 buffer 设为可写，而 rank-5 的 DMA 尚未读完，会产生数据竞争。因此每次 batch 传输完成后、rank-0 释放 buffer 前，加入 **rank 间 barrier**，确保**所有 rank 的当前 batch H2D 都已完成**。

**执行流程（单 epoch）**：

```
启动 prep.train() 线程
每个 rank 线程：
  for batch = 0 .. batches-1:
    buf = batch % 2
    等待 ts->buffer_is_readable(buf)
    cudaGraphLaunch(xfer_a 或 xfer_b, TRANS stream)
    cudaStreamSynchronize(TRANS stream)
    barrier(所有 rank)          <-- 保证所有 rank H2D 完成
    if rank == 0:
      ts->set_buffer_readable(buf, false)
      ts->set_buffer_writeable(buf, true)
join 所有 rank 线程
join prep 线程
返回耗时
```

### 2.4 Last Batch 处理

`Preprocessor::steps_per_epoch()` 已经通过 `ceil(num_train_samples / batch_size)` 把 last batch 计入总步数。H2D CUDA Graph 的传输范围是固定的（整个 `I_A_LABEL` + `I_A_DATA` / `I_B_LABEL` + `I_B_DATA` Region），因此 last batch 不需要变体图，直接和普通 batch 一样传输即可。TransferStation 的实际有效数据量由 `get_buffer_actual_transfer_bytes()` 描述，但 H2D 图本身始终传输固定大小。

---

## 3. 具体修改

### 3.1 `include/renaissance/task/deep_learning_task.h`

在 `public` 段新增：

```cpp
    /**
     * @brief 仅编译 H2D 传输图（TRANSFER_A + TRANSFER_B），不编译 FWD/BWD/OPT 等
     */
    void compile_h2d_only();

    /**
     * @brief 仅运行 H2D 传输图一个 epoch，联动 Preprocessor/TransferStation
     * @return wall-clock 耗时（秒）
     */
    double run_h2d_only();
```

在 `private` 段新增：

```cpp
    bool h2d_only_ = false;   // compile_h2d_only() 模式标志

    /**
     * @brief H2D-only 模式下构建只含 xfer 的 gpu_exec_ 表
     */
    void build_exec_table_h2d_only();
```

### 3.2 `src/task/deep_learning_task.cpp`

#### A. 修改 `build_graph_atlas()`

在函数开头增加 `h2d_only_` 分支：

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    // ========== H2D-Only 模式：只填 xfer 图 ==========
    if (h2d_only_) {
        if (train_cg_) {
            for (GraphId gid : {GraphId::TRANSFER_A, GraphId::TRANSFER_B}) {
                if (train_cg_->nodes(gid).empty()) continue;
                auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
                sl.cg = train_cg_;
                sl.mp = active_memory_plan_;
                sl.stream_kind = StreamKind::TRANS;
                sl.shape_id = kShapeInvariant;
            }
        }
        // name_to_gid_ 在 h2d_only 下不需要映射 train/inference
        name_to_gid_.clear();
        return atlas;
    }
    // ==================================================

    // 原有逻辑保持不变 ...
    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            // ... 原代码 ...
        }
    }
    // ...
}
```

#### B. 修改 `build_exec_table()`

在函数开头增加 `h2d_only_` 分发：

```cpp
void DeepLearningTask::build_exec_table() {
    if (h2d_only_) {
        build_exec_table_h2d_only();
        return;
    }
    // 原有逻辑保持不变 ...
}
```

#### C. 新增 `build_exec_table_h2d_only()`

```cpp
void DeepLearningTask::build_exec_table_h2d_only() {
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
        g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank);
        g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank);
    }
#endif
}
```

#### D. 新增 `compile_h2d_only()`

```cpp
void DeepLearningTask::compile_h2d_only() {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "compile_h2d_only() must be called in PLANNING phase");

    h2d_only_ = true;
    compile();   // 走正常编译管线，但 build_graph_atlas/build_exec_table 会走简化分支
}
```

> 注意：`compile()` 内部会调用 `on_prepare()` → `build_graph_atlas()` → `pre_capture()` → `build_exec_table()`，由于 `h2d_only_ == true`，这些都会自动走 H2D-only 分支。

#### E. 新增 `run_h2d_only()`

```cpp
double DeepLearningTask::run_h2d_only() {
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
    TR_CHECK(ts != nullptr, RuntimeError,
             "TransferStation not ready within timeout");

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    // ---- rank 间 barrier ----
    std::mutex mtx;
    std::condition_variable cv;
    int barrier_count = 0;

    auto sync_barrier = [&](int total) {
        std::unique_lock<std::mutex> lk(mtx);
        ++barrier_count;
        if (barrier_count == total) {
            barrier_count = 0;
            cv.notify_all();
        } else {
            cv.wait(lk, [&] { return barrier_count == 0; });
        }
    };

    auto t0 = std::chrono::steady_clock::now();

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);
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

                    sync_barrier(K);   // 所有 rank H2D 完成后才释放 buffer

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

    return std::chrono::duration<double>(t1 - t0).count();
#else
    return 0.0;
#endif
}
```

---

### 3.3 新增测试文件 `tests/correction/test_h2d_only_epoch.cpp`

基于 `test_pw_ultimate.cpp` 的 Preprocessor 配置，但使用 `DeepLearningTask` + `compile_h2d_only()` / `run_h2d_only()`。

```cpp
/**
 * @file test_h2d_only_epoch.cpp
 * @brief 仅测量 H2D 传输的每 epoch 耗时（Preprocessor + TransferStation + DeepLearningTask）
 */

#include "renaissance.h"
#include "renaissance/task/deep_learning_task.h"
#include <iostream>
#include <iomanip>

using namespace tr;

int main(int argc, char* argv[]) {
    // 简化参数：只支持数据集路径和少量关键参数
    std::string dataset_path;
    std::string dataset_type = "imagenet";
    int batch_size = 512;
    int resolution = 224;
    bool use_amp = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc) dataset_path = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc) dataset_type = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--resolution" && i + 1 < argc) resolution = std::stoi(argv[++i]);
        else if (arg == "--amp") use_amp = true;
    }

    if (dataset_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --path <dataset_root> [--dataset imagenet|mnist|cifar10]"
                     " [--batch-size N] [--resolution N] [--amp]\n";
        return 1;
    }

    // ---- 框架全局配置 ----
    GLOBAL_SETTING.use_gpu().auto_seed();
    GLOBAL_SETTING.local_batch_size(batch_size)
                  .train_resolution(resolution)
                  .val_resolution(resolution);
    if (use_amp) GLOBAL_SETTING.amp(true);

    auto& reg = GlobalRegistry::instance();
    reg.set_num_color_channels(3);

    int num_classes = 1000;
    NormMode norm = NormMode::MLPERF;
    if (dataset_type == "mnist") { num_classes = 10; norm = NormMode::MNIST; }
    else if (dataset_type == "cifar10") { num_classes = 10; norm = NormMode::CIFAR; }
    else if (dataset_type == "cifar100") { num_classes = 100; norm = NormMode::CIFAR; }

    // ---- Preprocessor 配置（与 test_pw_ultimate 一致）----
    PREPROCESSOR_SETTING
        .dataset(dataset_type, dataset_path)
        .color_channels(3)
        .load_workers(4)
        .preprocess_workers(8)
        .cpu_binding(false)
        .normalization(norm)
        .train_transforms(
            FastRandomResizedCrop(resolution, {0.08f, 1.0f}, {0.75f, 1.333f}),
            RandomHorizontalFlip())
        .val_transforms(
            Resize(256),
            CenterCrop(224))
        .partial_mode(true)
        .commit();

    // ---- DeepLearningTask：极简模型（单层 FC）----
    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)))   // 简单 FC，不关心准确率
        .loss(CrossEntropyLoss())
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .scheduler(PolynomialLR().base_lr(0.1f).power(1.0f))
        .num_classes(num_classes)
        .total_epochs(1);

    task.compile_h2d_only();

    const int steps = Preprocessor::instance().steps_per_epoch();
    std::cout << "=== H2D-Only Epoch Test ===\n"
              << "Dataset: " << dataset_type << "\n"
              << "Batch size: " << batch_size << "\n"
              << "Resolution: " << resolution << "\n"
              << "AMP: " << (use_amp ? "on" : "off") << "\n"
              << "Ranks: " << reg.world_size() << "\n"
              << "Steps per epoch: " << steps << "\n";

    double elapsed_sec = task.run_h2d_only();

    // 计算总传输字节数 = steps * per_zone_bytes * num_ranks
    size_t per_zone_bytes = 0;
    const auto& mp = task.memory_plan();
    for (const auto& d : mp.dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }
    size_t total_bytes = per_zone_bytes * static_cast<size_t>(steps) * static_cast<size_t>(reg.world_size());
    double bandwidth_gbps = 0.0;
    if (elapsed_sec > 0.0) {
        bandwidth_gbps = (static_cast<double>(total_bytes) / elapsed_sec)
                         / (1024.0 * 1024.0 * 1024.0);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Elapsed time: " << elapsed_sec << " s\n"
              << "Total bytes: " << (total_bytes / 1024.0 / 1024.0) << " MB\n"
              << "Aggregate BW: " << bandwidth_gbps << " GB/s\n"
              << "=== DONE ===\n";

    return 0;
}
```

### 3.4 `tests/correction/CMakeLists.txt`

新增测试目标：

```cmake
add_executable(test_h2d_only_epoch test_h2d_only_epoch.cpp)
target_link_libraries(test_h2d_only_epoch PRIVATE renaissance)
```

---

## 4. 与现有代码的关系

| 现有代码 | 本方案如何复用 |
|---------|--------------|
| `Compiler::build_auxiliary_graphs()` 中的 TRANSFER_A/B 构建 | 直接复用，`train_cg_` 已包含 xfer 子图 |
| `pre_capture()` | 正常调用，Atlas 中只有 xfer slot，因此只 capture xfer |
| `test_h2d_copy_bandwidth()` | 其多 rank 线程 + barrier + TransferStation 交互模式被 `run_h2d_only()` 继承 |
| `run_train_epoch_gpu()` | 其 `prep.train()` 线程启动方式和 buffer 0/1 交替策略被继承 |
| `test_pw_ultimate.cpp` | 其 Preprocessor 配置代码被测试文件直接复用 |

---

## 5. 边界情况处理

| 场景 | 处理 |
|-----|------|
| `batches == 1` | 循环只执行 1 次（buf=0 → xfer_a），无特殊分支需求 |
| `batches == 0` | `run_h2d_only()` 循环体不执行，立即返回 0.0 秒 |
| AMP (FP16) | `Compiler` 已根据 `GlobalRegistry::using_amp()` 生成正确 Region（I_A_DATA 为 FP16），`pre_capture()` 自动适配 |
| Multi-GPU | 每个 rank 独立 launch 各自 capture 的 xfer graph，共享 TransferStation CPU buffer，通过 barrier 保证安全 |
| CPU 模式 | `#ifdef TR_USE_CUDA` 保护，`run_h2d_only()` 直接返回 0.0；编译阶段无影响 |

---

## 6. 风险与注意事项

1. **`h2d_only_` 标志的副作用**：`compile_h2d_only()` 调用后 `h2d_only_` 保持 `true`。若用户随后调用 `run()`（完整训练），`gpu_exec_` 中只有 xfer 句柄，会导致 segfault。建议文档中明确：**`compile_h2d_only()` 和 `compile()` 互斥，不可混用**。

2. **Barrier 开销**：每次 batch 后 barrier 会引入微小同步开销（约几微秒到几十微秒）。由于本测试目的是测量 **Preprocessor 与 H2D 对接后的端到端耗时**（而非裸 H2D 带宽），该 barrier 是正确性必需的，不算“多余动作”。若未来需要测裸带宽，应直接使用 `test_h2d_copy_bandwidth()`。

3. **Init All 开销**：`compile()` 末尾的 `init_all()` 会初始化所有权重。对于大型模型（如 ResNet-50），这可能需要几百毫秒到几秒。由于本方案模型是单层 FC，该开销可忽略。若未来需要为大型模型做 `compile_h2d_only()`，可考虑在 `TaskBase::compile_impl()` 中增加 `h2d_only_mode_` 保护以跳过 `init_all()`，但这需要修改 `TaskBase`，侵入更大。
