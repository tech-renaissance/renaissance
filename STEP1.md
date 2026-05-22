# DeepLearningTask GPU 训练 — 四步测试计划（详细版）

> 基于 [STEP.md](file:///r:/renaissance/STEP.md) 思路展开，结合当前代码现状制定可操作的详细步骤。

---

## 前置条件

### 测试目标代码

我们修改的关键文件：

| 文件 | 角色 |
|------|------|
| `src/task/task_base.cpp` | `compile_impl()` 支持 DL 分支、`init()` 重写、`active_memory_plan_` |
| `src/task/deep_learning_task.cpp` | `build_graph_atlas()`、`build_exec_table()`、`run_train_epoch_gpu()`、`run_gpu()` |
| `include/renaissance/task/deep_learning_task.h` | 新增成员 `lr_pinned_`、`train_cg_`、`infer_cg_`，新方法声明 |
| `include/renaissance/algo/scheduler.h` | 新增 `is_step_by_batch()` |
| `tests/ref/mnist_mlp_3.cpp` | 已改为 `compile()` + `run()` |

### 参照测试文件

- **`tests/ref/mnist_mlp_3.cpp`** — DeepLearningTask 完整示例：`PREPROCESSOR_SETTING → GLOBAL_SETTING → DeepLearningTask → compile() → run()`
- **`tests/correction/test_h2d_copy_a.cpp`** — A 区传输正确性测试模板：`SimpleTask + RANGE_H2D_COPY_A → staging_memory_ptr → memcpy → run("xfer_a") → fetch_from_rank → 逐字节对比`
- **`tests/correction/perf_h2d_copy_a.cpp`** — A 区传输性能测试模板：warmup + iterations 循环 + 带宽计算
- **`tests/correction/CMakeLists.txt`** — CMake 构建模板：`add_executable + target_link_libraries + setup_gpu_runtime_env`

---

## 第一步：跑通 compile，run 只打印不执行

### 目标

验证 `compile()` 流程能完整走通：
1. `on_prepare()` → Compiler 生成 IR
2. `compile_impl()` → ArenaKeeper memset → `build_graph_atlas()` → `pre_capture()` → `build_exec_table()` → `compile_mark_compiled()` → `init_all()` → `cudaMallocHost`
3. `run()` 进入 `run_gpu()` → `run_train_epoch_gpu()` 启动 rank 线程

但不真正执行 CUDA Graph，每个 Graph 只打印一行 `"[DRY] Launching GraphSlot::XXX on stream YYY"`。

### 实现方案

#### 1.1 创建测试文件

在 `tests/correction/` 新建 `test_dl_dry.cpp`（或使用 `tests/ref/mnist_mlp_3.cpp` 的 dry 变体）。

```cpp
#include "renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    GLOBAL_SETTING
        .use_gpu("0")
        .manual_seed(42)
        .local_batch_size(5000)    // ← 大幅减小 batch 数
        .train_resolution(28)
        .val_resolution(28)
        .amp(false);

    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1)
        .load_workers(1)
        .preprocess_workers(1)
        .cpu_binding(false)        // ← 单卡测试不需要复杂绑核
        .normalization(NormMode::MNIST)
        .train_transforms(DoNothing())
        .val_transforms(DoNothing())
        .commit();

    BluePrint mlp = seq(fc(512, true), relu(), fc(256, true), relu(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(1)           // ← 只跑 1 个 epoch
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .scheduler(CosineAnnealingLR().base_lr(0.01f).eta_min(1e-5f).step_by_epoch())
        .validate_every(1, 1)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    task.compile();  // ← 核心验证点：走完整个 compile 流程

    std::cout << "=== compile() PASSED ===\n";

    auto result = task.run();  // ← 进入 run_gpu() → run_train_epoch_gpu()
    // 预期：每个 batch 打印一行 Graph 信息，无 CUDA 错误

    return 0;
}
```

#### 1.2 添加 DRY_RUN_CUDA_GRAPH 宏

在 `deep_learning_task.cpp` 的 `run_train_epoch_gpu()` 顶部添加条件编译：

```cpp
// 文件顶部或 build_exec_table() 之后
// #define DRY_RUN_CUDA_GRAPH  // 取消注释以启用 dry-run 打印模式

// 在 run_train_epoch_gpu() 的 lambda 中，将所有 cudaGraphLaunch 替换为：

#ifdef DRY_RUN_CUDA_GRAPH
#define LAUNCH_OR_PRINT(graph, stream, name)                               \
    do {                                                                    \
        if (graph) {                                                        \
            std::cout << "[DRY] rank=" << rank << " batch=" << batch       \
                      << " launch " << name << " stream="                   \
                      << (stream == s_c1 ? "COMP_1" :                      \
                          stream == s_c2 ? "COMP_2" :                      \
                          stream == s_up  ? "UPDATE" : "TRANS")            \
                      << std::endl;                                         \
        }                                                                   \
    } while(0)
#else
#define LAUNCH_OR_PRINT(graph, stream, name) \
    do { if (graph) cudaGraphLaunch(graph, stream); } while(0)
#endif
```

> **注意**：这是临时调试宏，验证完后删除。真正用于 Step 1 时手动替换 `cudaGraphLaunch(g_xxx, s_yyy)`。

#### 1.3 Preprocessor 注释掉

在 `run_gpu()` 中注释掉 Preprocessor 线程：

```cpp
// std::thread prep_thread([&]() { ... });
// prep_thread.join();
```

改为直接跳过数据传输，用假数据填充 TransferStation buffer。

#### 1.4 预期输出

```
=== compile() PASSED ===
[DRY] rank=0 batch=0 launch ZERO_GRAD stream=UPDATE
[DRY] rank=0 batch=0 launch FIRST_FWD_A stream=COMP_1
[DRY] rank=0 batch=0 launch DEEP_FWD_BWD stream=COMP_1
[DRY] rank=0 batch=0 launch XFER_B stream=TRANS
[DRY] rank=0 batch=0 launch FIRST_LAYER_BWD stream=COMP_1
[DRY] rank=0 batch=0 launch DEEP_ALLREDUCE stream=UPDATE
[DRY] rank=0 batch=0 launch WEIGHT_UPDATE stream=UPDATE
...
```

#### 1.5 成功标准

- [ ] `compile()` 返回且 `phase_ == COMPILED`
- [ ] `build_graph_atlas()` 中 `train_cg_` 非空，atlas slot 数量正确
- [ ] `build_exec_table()` 中所有 kRequired GraphSlot 都非空
- [ ] `init_all()` 不触发 phase 检查失败
- [ ] `cudaMallocHost` 都成功
- [ ] `run_gpu()` → `run_train_epoch_gpu()` 线程启动正常
- [ ] 打印的 Graph 序列符合 Z_FINAL_K.md §7.3 四阶段预期
- [ ] `run_gpu()` 正常返回 `TrainingResult`

#### 1.6 CMakeLists.txt 添加

```cmake
# tests/correction/CMakeLists.txt
add_executable(test_dl_dry test_dl_dry.cpp)
target_link_libraries(test_dl_dry PRIVATE renaissance)
target_compile_definitions(test_dl_dry PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_dl_dry)
endif()
set_target_properties(test_dl_dry PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
```

---

## 第二步：实测 A 区 Transfer 正确性

### 目标

验证 TransferStation 写入 → Transfer Graph 异步 H2D → GPU 取回的往返数据一致性。只跑一次，只跑 Transfer Graph。

### 挑战分析

DeepLearningTask 不使用 SimpleTask 的 `StagingBufferPool` + `RANGE_H2D_COPY_A` 路径。它的 Transfer Graph 由 Compiler 自动生成，data layout 由 MemoryPlan 决定。

**需要额外开发的调试接口**（临时，仅测试用）：

#### 2.1 添加 `debug_copy_staging_to_host()`

在 `DeepLearningTask` 上添加调试方法：

```cpp
// deep_learning_task.h — 临时调试接口（测试完成后移除）
public:
    // 将 TransferStation 的指定 buffer 数据完整复制到 host 内存
    // 返回的 vector 布局与 TransferStation 内存布局一致
    std::vector<uint8_t> debug_copy_staging_to_host(int engine_id, int buffer_id);

    // 从 GPU 取回 Transfer 目标区域的数据
    Tensor debug_fetch_transfer_destination(int rank);
```

实现思路：

```cpp
// deep_learning_task.cpp
std::vector<uint8_t> DeepLearningTask::debug_copy_staging_to_host(int engine_id, int buffer_id) {
    auto& reg = GlobalRegistry::instance();
    auto* ts = static_cast<TransferStation*>(reg.transfer_station_ptr(engine_id));
    uint8_t* src = ts->get_buffer_ptr(buffer_id);  // buffer_labels_ 起始地址
    size_t bytes = ts->get_buffer_actual_transfer_bytes(buffer_id);
    std::vector<uint8_t> copy(bytes);
    std::memcpy(copy.data(), src, bytes);
    return copy;
}

Tensor DeepLearningTask::debug_fetch_transfer_destination(int rank) {
    // 找一个已知存在于 Transfer 目标区域的 DTensor
    // 因为 Transfer Graph 传输 I_A_LABEL + I_A_DATA 到 GPU 对应区域
    // 取回 I_A_DATA 的 DTensor 内容
    for (const auto& dt : active_memory_plan_->dtensors()) {
        if (dt.region == Region::I_A_DATA) {
            return fetch_from_rank(dt, rank);
        }
    }
    TR_CHECK(false, ValueError, "No I_A_DATA DTensor found");
    return Tensor();
}
```

#### 2.2 创建测试文件 `test_dl_xfer_a.cpp`

```cpp
#include "renaissance.h"
#include <iostream>
#include <cstring>
#include <iomanip>

using namespace tr;

int main() {
    GLOBAL_SETTING
        .use_gpu("0")
        .manual_seed(42)
        .local_batch_size(128)
        .train_resolution(28)
        .val_resolution(28)
        .amp(false);

    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1)
        .load_workers(1)
        .preprocess_workers(1)
        .cpu_binding(false)
        .normalization(NormMode::MNIST)
        .train_transforms(DoNothing())
        .val_transforms(DoNothing())
        .commit();

    BluePrint mlp = seq(fc(512, true), relu(), fc(256, true), relu(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(1)
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .scheduler(CosineAnnealingLR().base_lr(0.01f).eta_min(1e-5f).step_by_epoch())
        .validate_every(1, 1)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    task.compile();

    // ── 手动启动 Preprocessor 准备一个 batch ──
    auto& prep = Preprocessor::instance();
    // prep.train() 会启动所有 worker 线程，这里需要一种方式只准备一个 batch
    // 方案：直接操作 TransferStation，手动填充 staging buffer
    // （需要深层了解 TransferStation 内存布局）

    auto& reg = GlobalRegistry::instance();
    auto* ts = static_cast<TransferStation*>(reg.transfer_station_ptr(0));
    int batch_size = reg.get_local_batch_size();
    int resolution = reg.train_sample_resolution_begin();
    int channels = reg.num_color_channels();

    // 填充 buffer 0：手工写入已知模式的 label + image 数据
    uint8_t* buf0 = ts->get_buffer_ptr(0);
    size_t label_bytes = batch_size * sizeof(int32_t);
    // Image 数据从 label_aligned 偏移开始
    size_t label_aligned = ((label_bytes + 16 + 255) / 256) * 256;

    // 写入 label：0, 1, 2, ..., batch_size-1
    auto* labels = reinterpret_cast<int32_t*>(buf0);
    for (int i = 0; i < batch_size; ++i) labels[i] = i % 10;

    // 写入 image：已知递增模式
    auto* images = reinterpret_cast<float*>(buf0 + label_aligned);
    int total_pixels = batch_size * resolution * resolution * channels;
    for (int i = 0; i < total_pixels; ++i) {
        images[i] = static_cast<float>(i % 997) * 0.001f;
    }

    ts->set_buffer_readable(0, true);

    // ── 保存本地副本 ──
    auto local_copy = task.debug_copy_staging_to_host(0, 0);

    // ── 运行 Transfer Graph（仅此一张图） ──
    // 方案 A：用手动构造的 SimpleTask 跑 RANGE_H2D_COPY 图（SimpleTask 路径已验证）
    // 方案 B：用 DeepLearningTask 的 run_train_epoch_gpu() 但只跑 Transfer 部分
    // 方案 B 更接近真实路径，但需要 hack 跳过其他图

    // 推荐方案 A（可操作性强）：
    //   将 DeepLearningTask 的 Transfer Graph 导出到 SimpleTask 执行
    //   或直接复用 test_h2d_copy_a.cpp 的框架，用 MemoryPlan 的 I_A 区域信息

    // 鉴于 DeepLearningTask 的 Transfer Graph 由 Compiler 生成，且命名为 "train"，
    // 我们可以通过 named_graphs_ 访问它（需要 friend 或 public getter）

    // 实际可行方案：直接在 DeepLearningTask 内部 hack
    // 在 run_train_epoch_gpu() 中加宏，只跑第一个 batch 的第一个 Graph

    // ── 取回 GPU 数据 ──
    Tensor gpu_data = task.debug_fetch_transfer_destination(0);

    // ── 逐字节对比 ──
    // 对比 local_copy 中 image 区域 vs gpu_data
    const float* gpu_f = gpu_data.data<float>();
    const float* local_f = reinterpret_cast<const float*>(
        local_copy.data() + label_aligned);
    double max_diff = 0.0;
    int64_t numel = gpu_data.numel();
    for (int64_t i = 0; i < numel; ++i) {
        double d = std::abs(static_cast<double>(gpu_f[i]) - static_cast<double>(local_f[i]));
        if (d > max_diff) max_diff = d;
    }

    std::cout << "Transfer A correct test: max|diff| = " << max_diff << std::endl;
    bool pass = (max_diff < 1e-6);
    std::cout << (pass ? "PASS" : "FAIL") << std::endl;

    return pass ? 0 : 1;
}
```

#### 2.3 实际可行方案评估

由于 DeepLearningTask 的 Transfer Graph 嵌入在完整的训练流水线中，单独提取运行一条 Transfer Graph 需要较大 hack 工作量。**更务实的做法是：**

1. **Step 2 使用 SimpleTask + RANGE_H2D_COPY_A 路径**（已有 `test_h2d_copy_a.cpp` 验证），这验证了硬件 H2D 传输的底层正确性
2. **DeepLearningTask 的 Transfer Graph 正确性通过 Step 1 的 dry-run 打印来推断**——我们能看到 Transfer Graph 的 launch 是否正确
3. **Step 4 跑通后，通过 loss 曲线下降来间接验证** Transfer Graph 的正确性

> **如果必须验证 DeepLearningTask 特有路径**，建议采用：在 `run_train_epoch_gpu()` 中添加 `#define DEBUG_TRANSFER_ONLY` 模式，第一个 batch 只跑 XFER + 取回对比，不跑其他图。

#### 2.4 成功标准

- [ ] `task.compile()` 成功
- [ ] `ts->get_buffer_ptr(0)` 返回非空
- [ ] `debug_copy_staging_to_host()` 返回完整数据副本
- [ ] `debug_fetch_transfer_destination()` 取得 GPU 端数据
- [ ] 逐字节 `max|diff| < 1e-6`

---

## 第三步：实测 AB 区传输性能

### 目标

验证 XFER_A/B 交替传输的异步流水线正确性和带宽。

### 实现方案

创建 `tests/correction/perf_dl_xfer_ab.cpp`：

```cpp
#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>

using namespace tr;
using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

int main() {
    GLOBAL_SETTING
        .use_gpu("0")
        .manual_seed(42)
        .local_batch_size(128)
        .train_resolution(28)
        .val_resolution(28)
        .amp(false);

    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1)
        .load_workers(1)
        .preprocess_workers(1)
        .cpu_binding(false)
        .normalization(NormMode::MNIST)
        .train_transforms(DoNothing())
        .val_transforms(DoNothing())
        .commit();

    BluePrint mlp = seq(fc(512, true), relu(), fc(256, true), relu(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(1)
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .scheduler(CosineAnnealingLR().base_lr(0.01f).eta_min(1e-5f).step_by_epoch())
        .validate_every(1, 1)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    task.compile();

    // ── 方案：Hack run_train_epoch_gpu() 为 XFER-only 模式 ──
    // 通过 #define PERF_XFER_ONLY 使 GPU 线程只执行：
    //   for batch in 0..batches-1:
    //     t0 = now()
    //     cudaGraphLaunch(xfer, s_trans)
    //     cudaStreamSynchronize(s_trans)
    //     t1 = now()
    //     record(t1-t0)
    //
    // 注：需要 Preprocessor 提供数据填充 TransferStation buffer

    // 临时 hack：手动填充 A/B 两个 buffer，然后运行 xfer-only 循环
    auto& reg = GlobalRegistry::instance();
    auto* ts = static_cast<TransferStation*>(reg.transfer_station_ptr(0));
    int batch_size = reg.get_local_batch_size();
    int resolution = reg.train_sample_resolution_begin();

    size_t label_bytes = batch_size * sizeof(int32_t);
    size_t label_aligned = ((label_bytes + 16 + 255) / 256) * 256;

    // 填充 buffer 0 和 buffer 1
    for (int buf = 0; buf < 2; ++buf) {
        uint8_t* ptr = ts->get_buffer_ptr(buf);
        auto* labels = reinterpret_cast<int32_t*>(ptr);
        for (int i = 0; i < batch_size; ++i) labels[i] = i % 10;
        ts->set_buffer_readable(buf, true);
        ts->set_buffer_writeable(buf, false);
    }

    // ── 使用 SimpleTask + RANGE_H2D_COPY_A/B 性能测试 ──
    // 这是最可靠的测量方式（perf_h2d_copy_a.cpp 已验证）
    // 构建两个图：RANGE_H2D_COPY_A 和 RANGE_H2D_COPY_B

    SimpleTask xfer_task;
    Shape label_shape{batch_size, 1, 1, 1};
    Shape data_shape{batch_size, resolution, resolution, static_cast<int>(reg.num_color_channels())};

    DTensor d_label_a = xfer_task.alloc(label_shape, DType::INT32, Region::I_A_LABEL);
    DTensor d_data_a  = xfer_task.alloc(data_shape, DType::FP32, Region::I_A_DATA);
    DTensor d_label_b = xfer_task.alloc(label_shape, DType::INT32, Region::I_B_LABEL);
    DTensor d_data_b  = xfer_task.alloc(data_shape, DType::FP32, Region::I_B_DATA);

    xfer_task.finalize_memory();
    const auto& mp = xfer_task.memory_plan();

    // A 区传输图
    ComputationGraph g_a;
    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_A;
        node.output_ranges = { mp.region_range(Region::I_A_LABEL), mp.region_range(Region::I_A_DATA) };
        g_a.append(std::move(node));
    }
    xfer_task.add_graph("xfer_a", std::move(g_a), StreamKind::TRANS);

    // B 区传输图
    ComputationGraph g_b;
    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_B;
        node.output_ranges = { mp.region_range(Region::I_B_LABEL), mp.region_range(Region::I_B_DATA) };
        g_b.append(std::move(node));
    }
    xfer_task.add_graph("xfer_b", std::move(g_b), StreamKind::TRANS);

    xfer_task.compile();

    // ── 预热 ──
    const int warmup = 20;
    const int iterations = 200;
    for (int i = 0; i < warmup; ++i) {
        xfer_task.run("xfer_a");
        xfer_task.run("xfer_b");
    }

    // ── 测量 A/B 轮流 ──
    std::vector<double> latencies_us;
    latencies_us.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t0 = Clock::now();
        xfer_task.run("xfer_a");
        auto t1 = Clock::now();
        xfer_task.run("xfer_b");
        auto t2 = Clock::now();
        // 记录 AB 交替的总时间
        double us_ab = static_cast<double>(
            std::chrono::duration_cast<us>(t2 - t0).count());
        latencies_us.push_back(us_ab);
    }

    double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    double mean = sum / iterations;

    // 计算带宽
    size_t label_slot = static_cast<size_t>(DistributedTensor::compute_slot_bytes(
        label_shape, DType::INT32, Region::I_A_LABEL));
    size_t data_slot = static_cast<size_t>(DistributedTensor::compute_slot_bytes(
        data_shape, DType::FP32, Region::I_A_DATA));
    size_t per_xfer_bytes = label_slot + data_slot;
    size_t ab_bytes = per_xfer_bytes * 2;  // A + B 两个区
    double ab_mb = static_cast<double>(ab_bytes) / (1024.0 * 1024.0);
    double bw_gb_s = ab_mb / (mean / 1e6) / 1024.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== AB Transfer Performance ===\n";
    std::cout << "  AB bytes:        " << ab_bytes << " (" << ab_mb << " MB)\n";
    std::cout << "  Mean latency:    " << mean << " us/AB-pair\n";
    std::cout << "  Effective BW:    " << bw_gb_s << " GB/s\n";

    const double min_expected_bw = 2.0;  // RTX 4060 PCIe 保守下限
    bool pass = (bw_gb_s >= min_expected_bw);
    std::cout << "  " << (pass ? "PASS" : "WARNING: below expected bandwidth")
              << std::endl;

    return pass ? 0 : 1;
}
```

#### 3.1 简化方案

如果不想创建新的 SimpleTask，可以直接复用现有的 `perf_h2d_copy_a.cpp`，扩展为交替跑 A 和 B：

```bash
# 编译后运行
perf_h2d_copy_a --gpu --iter 200
```

关键是在 `tests/correction/CMakeLists.txt` 中添加对应的 perf 测试条目。

#### 3.2 成功标准

- [ ] XFER_A 和 XFER_B 交替执行无 CUDA 错误
- [ ] 单卡 H2D 带宽 ≥ 2 GB/s（PCIE 3.0 x16 理论 ~12 GB/s，取保守下限）
- [ ] 延迟稳定（stddev < mean * 20%）

---

## 第四步：完整 GPU 训练

### 目标

所有图全部加上，跑完整真实训练，验证 loss 下降和收敛。

### 实现方案

基于 `tests/ref/mnist_mlp_3.cpp`，但做以下调整以适配首次调试：

#### 4.1 创建 `tests/correction/test_dl_full.cpp`

```cpp
#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace tr;

int main() {
    GLOBAL_SETTING
        .use_gpu("0")
        .manual_seed(42)
        .local_batch_size(128)
        .train_resolution(28)
        .val_resolution(28)
        .amp(false);

    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1)
        .load_workers(1)
        .preprocess_workers(4)
        .cpu_binding(true)
        .normalization(NormMode::MNIST)
        .train_transforms(DoNothing())
        .val_transforms(DoNothing())
        .commit();

    BluePrint mlp = seq(fc(512, true), relu(), fc(256, true), relu(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(3)              // ← 少量 epoch 快速验证
        .optimizer(SGD()
            .momentum(0.9f)
            .weight_decay(5e-4f)
            .nesterov(false)
            .dampening(0.0f))
        .scheduler(CosineAnnealingLR()
            .base_lr(0.01f)
            .eta_min(1e-5f)
            .step_by_epoch())
        .validate_every(1, 1)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    const auto t0 = std::chrono::steady_clock::now();

    task.compile();

    const auto result = task.run();

    const auto t1 = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=====================================================\n"
              << " DL GPU Full Training Test\n"
              << "-----------------------------------------------------\n"
              << " Best Val Top-1:  " << result.best_top1 * 100.0f << " %\n"
              << " Best Epoch:      " << result.best_epoch << "\n"
              << " Total Time:      " << elapsed << " s\n"
              << "=====================================================\n";

    // 基本合理性检查：3 个 epoch MNIST MLP 应该达到 > 85% Top-1
    bool pass = (result.best_top1 > 0.85f);
    std::cout << (pass ? "PASS" : "FAIL — accuracy below 85%") << std::endl;

    return pass ? 0 : 1;
}
```

#### 4.2 CMakeLists.txt 添加

```cmake
# 第四步：完整 GPU 训练测试
add_executable(test_dl_full test_dl_full.cpp)
target_link_libraries(test_dl_full PRIVATE renaissance)
target_compile_definitions(test_dl_full PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_dl_full)
endif()
set_target_properties(test_dl_full PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_dl_full PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_dl_full PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_dl_full: DeepLearningTask GPU full training test")
```

#### 4.3 成功标准

| 指标 | 标准 |
|------|------|
| `compile()` | 无错误返回 |
| `run()` | 不崩溃，无 CUDA 错误 |
| Training loss | 逐渐下降（epoch 3 < epoch 1） |
| Val Top-1 @ epoch 3 | ≥ 85%（MNIST 3-layer MLP 合理下限） |
| 无内存泄漏 | `cudaFreeHost` 在析构时正常调用 |

---

## 总体测试流程

```
Step 1  ──→  test_dl_dry.cpp      验证 compile() + run() 骨架
            (epochs=1, batch=5000, 只打印不执行, Preprocessor 注释)

Step 2  ──→  test_h2d_copy_a.cpp   复用已有测试验证底层 H2D 正确性
            (+ DeepLearningTask 专用调试接口)

Step 3  ──→  perf_h2d_copy_a.cpp   复用已有测试验证底层 H2D 带宽
            + perf_dl_xfer_ab.cpp  AB 交替传输带宽

Step 4  ──→  test_dl_full.cpp      完整训练，验证 loss 下降 + Top-1 收敛
            (epochs=3, batch=128, 正常 Preprocessor, 所有 Graph)
```

## CMakeLists.txt 完整新增条目

在 `tests/correction/CMakeLists.txt` 末尾添加：

```cmake
# ============================================================================
# DeepLearningTask GPU 训练四步测试
# ============================================================================

# Step 1: Dry-run compile + print-only run
add_executable(test_dl_dry test_dl_dry.cpp)
target_link_libraries(test_dl_dry PRIVATE renaissance)
target_compile_definitions(test_dl_dry PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_dl_dry)
endif()
set_target_properties(test_dl_dry PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_dl_dry PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_dl_dry PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_dl_dry: DeepLearningTask GPU dry-run compile+print test (Step 1)")

# Step 2: A-zone transfer correctness (复用 test_h2d_copy_a 基础设施)
# 注：DeepLearningTask 专用的 transfer 验证脚本按需创建 test_dl_xfer_a.cpp

# Step 3: AB zone transfer performance
add_executable(perf_dl_xfer_ab perf_dl_xfer_ab.cpp)
target_link_libraries(perf_dl_xfer_ab PRIVATE renaissance)
target_compile_definitions(perf_dl_xfer_ab PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(perf_dl_xfer_ab)
endif()
set_target_properties(perf_dl_xfer_ab PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(perf_dl_xfer_ab PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(perf_dl_xfer_ab PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - perf_dl_xfer_ab: AB zone transfer performance test (Step 3)")

# Step 4: Full GPU training
add_executable(test_dl_full test_dl_full.cpp)
target_link_libraries(test_dl_full PRIVATE renaissance)
target_compile_definitions(test_dl_full PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_dl_full)
endif()
set_target_properties(test_dl_full PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_dl_full PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_dl_full PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_dl_full: DeepLearningTask GPU full training test (Step 4)")
```

---

## 附录 A：调试宏速查

| 宏 | 用途 | 位置 |
|----|------|------|
| `DRY_RUN_CUDA_GRAPH` | 替换 cudaGraphLaunch 为打印 | `run_train_epoch_gpu()` |
| `DEBUG_TRANSFER_ONLY` | 只跑 Transfer Graph + 取回对比 | `run_train_epoch_gpu()` |
| `PERF_XFER_ONLY` | 只跑 Transfer Graph + 计时 | `run_train_epoch_gpu()` |
| `TR_LOG_LEVEL=1` | 启用最小日志 | CMake 编译定义 |

## 附录 B：常见问题预判

| 问题 | 原因 | 排查方法 |
|------|------|----------|
| `compile()` 崩溃 | `active_memory_plan_` 指向空、`init_all()` phase 检查失败 | Step 1 仔细看编译输出和 TR_CHECK 断言 |
| `pre_capture()` 失败 | MemoryPlan data layout 不对、context 未 set_memory_plan | 检查 `build_graph_atlas()` 输出 |
| `cudaGraphLaunch` 报 `cudaErrorInvalidValue` | Graph 为 nullptr、stream 不正确 | Step 1 打印确认 Graph 非空 |
| loss 不下降 | ALLREDUCE 或 WEIGHT_UPDATE 未正确执行 | Step 4 对比 reference mnist_mlp_3 的 loss 曲线 |
| H2D 带宽异常低 | 未使用锁页内存、同步点过多 | Step 3 确认 `cudaMallocHost` 分配成功 |