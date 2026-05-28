# ZKR1: YZT 评审与最终方案

## 1. 额外代码检查结果

在评审三位小伙伴的方案前，已补充检查以下关键代码：

### 1.1 `prep.steps_per_epoch()` 对 train/val 的行为

[`preprocessor.cpp:L969`](file:///r:/renaissance/src/data/preprocessor.cpp#L969)：

```cpp
steps_per_epoch_ = static_cast<int>((total_train_samples + global_batch_size - 1) / global_batch_size);
```

**关键事实**：`steps_per_epoch_` 在 `commit()` 时基于 `total_train_samples` 计算一次，之后 `train()` 和 `val()` 调用都不会改变它。调用 `prep.steps_per_epoch()` 后返回的始终是 **训练集的 batch 数**，不是验证集的。

**影响**：任何用 `prep.steps_per_epoch()` 来获取 val batch 数的方案都是错误的。Val batch 数必须手动计算：
```cpp
int val_batches = static_cast<int>((num_val_samples + batch_size - 1) / batch_size);
```

### 1.2 `RANGE_H2D_COPY` 的 CPU 后端

[`h2d_op.cpp:L199-L232`](file:///r:/renaissance/src/backend/ops/range/h2d_op.cpp#L199-L232)：

```cpp
static void launch_range_h2d_copy_cpu(CpuOpContext* op_ctx) {
    uint8_t* staging_base = reg.staging_memory_ptr(rank);
    size_t per_zone = reg.staging_memory_size() / 2;
    size_t label_aligned = get_label_aligned();
    // ...
    std::memcpy(dst, src, range.size);
}
```

CPU 后端已完整实现，通过 `std::memcpy` 从 StagingBufferPool 拷贝到 ArenaKeeper 目标地址。但注意：与 GPU 路径使用 DTensor 地址不同，CPU 路径使用 `ArenaKeeper::ptr_at()`。这意味着 CPU 模式下 DTensor 的 `slot_bytes()` 偏移量与 GPU 模式一致，但最终地址来源不同。

### 1.3 TransferStation 的 buffer 指针 API

`get_buffer_ptr(int buf_id)` — 返回 `buffer_labels_[buf_id]`（标签区首地址）
`get_buffer_actual_transfer_bytes(int buf_id)` — 返回标签区 + 图像区的总字节数

这些在 GPU 路径通过 CUDA graph 使用，CPU 路径下是同一块主机内存。

### 1.4 `total_epochs_` 默认值

[`deep_learning_task.h:L345`](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h#L345)：`int total_epochs_ = 35;`

如果 `run_h2d_only()` 启用多 epoch 循环，`test_h2d_only_epoch.cpp` 若未显式设 `.total_epochs(1)` 会跑 35 个 epoch，这不是期望的行为。

---

## 2. 三方方案对比

### 2.1 数据结构

| | 小伙伴S | 小伙伴K | 小伙伴D (YBR1) |
|---|---|---|---|
| 结构体 | `H2DOnlyEpochResult` + `H2DOnlyResult`（全新） | `H2DTestResult`（不变）+ `H2DRunResult`（新增） | `H2DTestResult`（扩展 `EpochDetail` 向量） |
| 向后兼容 | ❌ 破坏性变更 | ✅ `H2DTestResult` 完全不变 | ⚠️ 新增字段但不破坏 |
| train/val 区分 | per-epoch 结构体含 `train_`/`val_` 字段对 | 分两个 vector：`train_per_epoch` / `val_per_epoch` | 扁平 `epochs` 向量 + `is_val` 标志 |

### 2.2 Val batch 数计算

| 小伙伴S | 小伙伴K | 小伙伴D (YBR1) |
|---|---|---|
| ❌ 用 `prep.steps_per_epoch()`（错误） | ✅ 手动计算 `(num_val+batch_size-1)/batch_size` | ❌ 建议用 `prep.steps_per_epoch()`（错误） |

### 2.3 label_smce 复制

| 小伙伴S | 小伙伴K | 小伙伴D (YBR1) |
|---|---|---|
| ❌ 未提及 | ✅ 明确 Val 不需要 | ⚠️ 假设 train/val 逻辑一致 |

### 2.4 CPU 路径

| 小伙伴S | 小伙伴K | 小伙伴D (YBR1) |
|---|---|---|
| ⚠️ `memcpy` 到一个临时 vector `<uint8_t>`（每 batch 分配，浪费；未复制到 DTensor 目标地址） | ✅ 正确处理：CPU 无 H2D，仅消费 staging buffer 信号量 | ⚠️ 提到 ComputationGraph CPU 后端，但未验证可行性 |

### 2.5 `total_epochs_=35` 兜底

| 小伙伴S | 小伙伴K | 小伙伴D (YBR1) |
|---|---|---|
| ❌ 未提及 | ✅ 推荐 test 显式 `.total_epochs(1)` | ❌ 未提及 |

### 2.6 多 rank TransferStation

| 小伙伴S | 小伙伴K | 小伙伴D (YBR1) |
|---|---|---|
| ❌ 未显式获取每个 rank 的 TransferStation | ✅ 每个 rank 获取自己的 `transfer_station_ptr(rank)` | ✅ 每个 rank 获取自己的 |

### 2.7 GPU 路径中 `run_h2d_only_epoch_gpu()` 的 train/val 实现方式

| 小伙伴S | 小伙伴K | 小伙伴D (YBR1) |
|---|---|---|
| ⚠️ 一个函数 `run_h2d_only_epoch_gpu(bool)` 内嵌大段 if/else，train 和 val 代码大量重复 | ✅ 两个独立函数 `run_h2d_only_train_epoch()` / `run_h2d_only_val_epoch()` | ⚠️ 一个函数 `run_h2d_only_epoch(bool)` |

### 2.8 代码重复度

| 小伙伴S | 小伙伴K | 小伙伴D (YBR1) |
|---|---|---|
| ⚠️ 高：GPU rank 线程逻辑在 train 和 val 段几乎全文重复 | ⚠️ 中：train 和 val 是两个函数，但 rank 线程核心逻辑（wait→launch→sync→release）相同 | ⚠️ 中：一个函数 + bool 参数，但 val 段的 batch 计算和 label copy 逻辑不同 |

---

## 3. 关键错误标记（P0 — 必须修复才能用）

### P0-1: Val batch 数（小伙伴S + 小伙伴D）

`prep.steps_per_epoch()` 永远返回训练集 batch 数。Val 必须用 `ceil(num_val_samples / local_batch_size)`。

### P0-2: CPU 路径 `memcpy` 目标（小伙伴S）

`memcpy(temp_buffer.data(), src, size)` 拷贝到一个局部的 `std::vector<uint8_t>`，每 batch 都在栈上分配一个新 vector。正确的目标是 DTensor 所在的 ArenaKeeper 地址，这与 `launch_range_h2d_copy_cpu` 一致。

### P0-3: 默认 35 epoch（三人都应注意）

`total_epochs_` 默认值 35，必须在 test 文件中显式 `.total_epochs(1)`。

---

## 4. 最终推荐方案

综合三方优点，推荐以 **小伙伴K 的方案为主体**，吸收小伙伴D 的向后兼容思路和结构清晰度，修正小伙伴S 中发现的问题。

### 4.1 数据结构

```cpp
// H2DTestResult — 保持不变（单 epoch 统计单元）
struct H2DTestResult {
    int    batches        = 0;
    double elapsed_us     = 0.0;
    size_t total_bytes    = 0;
    double bandwidth_gbps = 0.0;
    bool   labels_ok      = true;
    bool   data_ok        = true;
    double avg_lat_us     = 0.0;
    double min_lat_us     = 0.0;
    double max_lat_us     = 0.0;
};

// H2DRunResult — 新增（多 epoch 容器）
struct H2DRunResult {
    int epochs_run = 0;
    int vals_run   = 0;
    std::vector<H2DTestResult> train_per_epoch;
    std::vector<H2DTestResult> val_per_epoch;
    double total_elapsed_us = 0.0;

    H2DTestResult aggregate_train() const;
    H2DTestResult aggregate_val() const;
};
```

**选择理由**：
- `H2DTestResult` 完全不变 → `test_h2d_copy_bandwidth()` 等旧测试零影响
- `H2DRunResult` 明确分离 train/val → 用户无需通过 `is_val` 标志筛选
- `aggregate_train()` / `aggregate_val()` 便捷方法 → 现有 test 只需 `auto res = run_res.aggregate_train()` 即可适配

### 4.2 函数拆分

```cpp
public:
    H2DRunResult run_h2d_only();

private:
    H2DTestResult run_h2d_only_train_epoch();
    H2DTestResult run_h2d_only_val_epoch();
```

**选择理由**：
- 两个独立函数比一个带 `bool` 的函数更清晰地表达了"train 和 val 是不同的 epoch"
- train 有 `label_smce` 复制，val 没有 → 各自独立实现，避免 if/else 分支
- GPU/CPU 路径在函数内部通过 `#ifdef TR_USE_CUDA` + `using_gpu()` 二分

### 4.3 GPU 路径核心逻辑

```cpp
H2DTestResult DeepLearningTask::run_h2d_only_train_epoch() {
    H2DTestResult r;
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    // ...

    std::thread prep_thread([&]() { prep.train(); });

#ifdef TR_USE_CUDA
    if (GlobalRegistry::instance().using_gpu()) {
        // 完整的当前 GPU 逻辑：
        //   K 个 rank 线程
        //     for batch: wait_readable → cudaGraphLaunch → sync
        //              → set_readable(false) → set_writeable(true)
        //              → cudaMemcpyAsync(label → label_smce)
        //     join
    } else
#endif
    {
        // CPU 路径（见 4.5）
    }

    prep_thread.join();
    // 填充 r，返回
}

H2DTestResult DeepLearningTask::run_h2d_only_val_epoch() {
    H2DTestResult r;
    auto& prep = Preprocessor::instance();
    auto& registry = GlobalRegistry::instance();
    int val_batches = (num_val + batch_size - 1) / batch_size;
    // ...
    // 与 train 结构相同，但：
    //   - prep.val()
    //   - 无 label_smce 复制
    //   - batch 数 = val_batches
}
```

### 4.4 epoch 循环

```cpp
H2DRunResult DeepLearningTask::run_h2d_only() {
    H2DRunResult result;

    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;
        auto t0 = std::chrono::steady_clock::now();

        // Train
        auto train_res = run_h2d_only_train_epoch();
        result.train_per_epoch.push_back(train_res);
        result.epochs_run++;

        // Val
        if (should_validate_this_epoch()) {
            auto val_res = run_h2d_only_val_epoch();
            result.val_per_epoch.push_back(val_res);
            result.vals_run++;
        }

        auto t1 = std::chrono::steady_clock::now();
        result.total_elapsed_us += duration_cast<microseconds>(t1 - t0).count();
    }
    return result;
}
```

### 4.5 CPU 路径

CPU 模式下没有真正的 H2D（数据已在主机内存）。`launch_range_h2d_copy_cpu` 已实现了 `memcpy`，但这对 H2D 带宽测试无意义——这只是 CPU 内存拷贝，带宽是系统内存带宽。

**设计原则**：CPU 路径只消费 TransferStation 的信号量，不执行实际的数据拷贝。这样测量的是 Preprocessor 产出 staging buffer 的纯速度（即数据准备的 CPU 端瓶颈），与 GPU H2D 带宽形成对比。

```cpp
// CPU 路径（train epoch 内）
TransferStation* ts = transfer_station_ptr(0);
for (int batch = 0; batch < batches; ++batch) {
    int buf_id = batch % 2;
    ts->wait_buffer_readable(buf_id);
    ts->set_buffer_readable(buf_id, false);
    ts->set_buffer_writeable(buf_id, true);
}
r.bandwidth_gbps = 0.0;  // CPU 无 H2D，带宽为 0
r.total_bytes   = per_zone_bytes * batches;  // 记录逻辑数据量
```

**无需** `memcpy` 到临时 buffer 或 DTensor。`bandwidth = 0` 明确表达了"不是 H2D"。

### 4.6 test_h2d_only_epoch.cpp 适配

```cpp
// 改动：显式设 total_epochs(1)，适配 H2DRunResult
task.total_epochs(1)
    .num_classes(num_classes)
    // ...
    .compile_h2d_only();

auto run_res = task.run_h2d_only();

// 取第一个（唯一）train epoch 结果
assert(run_res.epochs_run == 1);
auto res = run_res.aggregate_train();

// 后续打印逻辑完全不变
std::cout << "Batches: " << res.batches << "\n";
std::cout << "Elapsed: " << res.elapsed_us / 1e6 << " s\n";
// ...
```

---

## 5. 与小伙伴S的差异说明

| 问题 | 小伙伴S | 最终方案 |
|---|---|---|
| `prep.steps_per_epoch()` 用于 val | 用（❌） | 手动计算 `(num_val+batch_size-1)/batch_size`（✅） |
| CPU 路径 `memcpy` 目标 | 临时 vector（❌） | 不拷贝，仅消费信号量（✅） |
| `total_epochs_=35` 默认 | 未处理（❌） | test 显式 `.total_epochs(1)`（✅） |
| label_smce 复制 | val 也复制（❌） | val 不复制（✅） |
| 结构体兼容性 | 全新 struct（❌） | 保留 `H2DTestResult`（✅） |

小伙伴S 方案中最严重的问题是 val batch 数错误和 `total_epochs_=35` 未处理，会导致测试行为不符合预期。

## 6. 与小伙伴D的差异说明

| 问题 | 小伙伴D (YBR1) | 最终方案 |
|---|---|---|
| 数据结构 | 扩展 `H2DTestResult` 加 `epochs` 向量 | 新增 `H2DRunResult` 分离 train/val |
| `prep.steps_per_epoch()` 用于 val | 建议用（❌） | 手动计算 |
| CPU 路径 | ComputationGraph 后端或手动 memcpy | 仅消费信号量 |
| train/val 函数 | 一个 `run_h2d_only_epoch(bool)` | 两个独立函数 |

小伙伴D 的核心问题是建议用 `prep.steps_per_epoch()` 获取 val batch 数，这在代码审查中已证实为无效。`EpochDetail` 扁平向量 + `is_val` 标志不如分离向量清晰。

## 7. 文件变更清单

| 文件 | 变更 |
|---|---|
| `include/renaissance/task/deep_learning_task.h` | 新增 `H2DRunResult` 结构体；`run_h2d_only()` 返回 `H2DRunResult`；新增 `run_h2d_only_train_epoch()` / `run_h2d_only_val_epoch()` 声明 |
| `src/task/deep_learning_task.cpp` | 重写 `run_h2d_only()` 为 epoch 循环；抽取 `run_h2d_only_train_epoch()`；新增 `run_h2d_only_val_epoch()`；每个函数内 GPU/CPU 分支 |
| `tests/correction/test_h2d_only_epoch.cpp` | 加 `.total_epochs(1)`；适配 `H2DRunResult`，用 `aggregate_train()` 取结果 |

---

## 8. 实施顺序

| 步骤 | 内容 | 风险 |
|---|---|---|
| 1 | `deep_learning_task.h`: 新增 `H2DRunResult`，改 `run_h2d_only()` 签名 | 低 |
| 2 | 提取 `run_h2d_only_train_epoch()`：将当前 GPU 代码移入，保持逻辑不变 | 低 |
| 3 | 重写 `run_h2d_only()`：外层 epoch 循环 + `should_validate_this_epoch()` | 中 |
| 4 | 实现 `run_h2d_only_val_epoch()`：参考 train epoch，用 val_batches，无 label_smce | 中 |
| 5 | train/val 函数内加 CPU 路径分支 | 低 |
| 6 | 适配 `test_h2d_only_epoch.cpp` + `.total_epochs(1)` | 低 |
| 7 | 编译 + Windows RTX 4060 单 GPU 测试 | — |
| 8 | Linux A100×8 多 GPU 测试 | — |