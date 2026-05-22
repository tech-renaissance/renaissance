# DeepLearningTask GPU 训练 — 最终测试计划

> **地位**：本文是 [STEP.md](file:///r:/renaissance/STEP.md)、[STEP1.md](file:///r:/renaissance/STEP1.md)、[STX1.md](file:///r:/renaissance/STX1.md) 的**合并终稿**，自包含、可直接按序执行。
> 三步走：Dry-Run → Transfer 逐步验证 → 全图真训。每一步都给出**完整代码、确切行号、判断标准**。

---

## 环境准备

```powershell
# 一次配置，后续不再改动
mkdir build\release -Force
cd build\release
cmake ..\.. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
```

后续每步只执行 `cmake --build . --config Release --target <target> -- /m`。

---

## 第一步：Dry-Run — compile 全流程 + print-only

### 为什么先做这一步
`compile()` 内部包含 **13 个关键步骤**（Arena memset → atlas 构建 → pre_capture → exec_table → mark_compiled → init_all → cudaMallocHost）。任何一个出错都会导致后续步骤无法定位问题。

### 1A：代码修改

**文件**: `src/task/deep_learning_task.cpp`

#### (a) 在 L786 之前插入 DRY_RUN 宏

```cpp
// ========== [DRY-RUN: 第一步专用, 测试后立即删除] ==========
#define DRY_RUN_CUDA_GRAPH
#ifdef DRY_RUN_CUDA_GRAPH
#include <iostream>
#define IGNORE_UNUSED(x) (void)(x)

#define L_OR_P(g, s, label)                       do { if (g) { std::cout << "[DRY] r" << rank << " b" << batch << " " << label << " s=" << (s==s_c1?"C1":s==s_c2?"C2":s==s_up?"UP":"TR") << std::endl; } } while(0)
#define LX_OR_P(g, s, label)                      do { if (g) { std::cout << "[DRY] r" << rank << " b" << batch << " " << label << " s=TR" << std::endl; } } while(0)
#define LS_OR_P(g, s, label)                      do { if (g) { std::cout << "[DRY] r" << rank << " " << label << " s=" << (s==s_c1?"C1":s==s_c2?"C2":s==s_up?"UP":"TR") << std::endl; } } while(0)
#else
#define L_OR_P(g, s, label)  do { if (g) cudaGraphLaunch(g, s); } while(0)
#define LX_OR_P(g, s, label) L_OR_P(g, s, label)
#define LS_OR_P(g, s, label) L_OR_P(g, s, label)
#endif
// ========== [DRY-RUN END] ==========
```

`LS_OR_P` ——用于 single-batch 和 last-batch 块（batch 变量未定义或已有 `from_a`）。

#### (b) 替换所有 cudaGraphLaunch

| 区域 | 当前行附近 | 替换 |
|------|-----------|------|
| pre-transmit | L845 `cudaGraphLaunch(g_xfer_a, s_trans)` | `LX_OR_P(g_xfer_a, s_trans, "XFER_A(pre)")` |
| single batch | L854 `cudaGraphLaunch(g_zg, s_up)` | `LS_OR_P(g_zg, s_up, "ZERO_GRAD")` |
| single batch | L855 `cudaGraphLaunch(g_fwd_a, s_c1)` | `LS_OR_P(g_fwd_a, s_c1, "FIRST_FWD")` |
| single batch | L858 `cudaGraphLaunch(g_deep_a, s_c1)` | `LS_OR_P(g_deep_a, s_c1, "DEEP_FWD_BWD")` |
| single batch | L862 `cudaGraphLaunch(g_first, s_c1)` | `LS_OR_P(g_first, s_c1, "FIRST_BWD")` |
| single batch | L864 `cudaGraphLaunch(g_dar, s_up)` | `LS_OR_P(g_dar, s_up, "DEEP_ALLREDUCE")` |
| single batch | L867 `cudaGraphLaunch(g_gc, s_up)` | `LS_OR_P(g_gc, s_up, "CAST_AND_CHECK")` |
| single batch | L875 `cudaGraphLaunch(g_far, s_up)` | `LS_OR_P(g_far, s_up, "FIRST_ALLREDUCE")` |
| single batch | L876 `cudaGraphLaunch(g_wu, s_up)` | `LS_OR_P(g_wu, s_up, "WEIGHT_UPDATE")` |
| 循环 Phase 1 | L890 `cudaGraphLaunch(g_zg, s_up)` | `L_OR_P(g_zg, s_up, "ZERO_GRAD")` |
| 循环 Phase 1 | L891 `cudaGraphLaunch(g_fwd, s_c1)` | `L_OR_P(g_fwd, s_c1, "FIRST_FWD")` |
| 循环 Phase 2 | L899 `cudaGraphLaunch(g_deep, s_c1)` | `L_OR_P(g_deep, s_c1, "DEEP_FWD_BWD")` |
| 循环 Phase 2 | L900 `cudaGraphLaunch(g_xfer_n, s_trans)` | `LX_OR_P(g_xfer_n, s_trans, "XFER")` |
| 循环 Phase 3 | L909 `cudaGraphLaunch(g_first, s_c1)` | `L_OR_P(g_first, s_c1, "FIRST_BWD")` |
| 循环 Phase 3 | L911 `cudaGraphLaunch(g_dar, s_up)` | `L_OR_P(g_dar, s_up, "DEEP_ALLREDUCE")` |
| 循环 AMP | L915 `cudaGraphLaunch(g_gc, s_up)` | `L_OR_P(g_gc, s_up, "CAST_AND_CHECK")` |
| 循环 Phase 4 | L924 `cudaGraphLaunch(g_far, s_up)` | `L_OR_P(g_far, s_up, "FIRST_ALLREDUCE")` |
| 循环 Phase 4 | L925 `cudaGraphLaunch(g_wu, s_up)` | `L_OR_P(g_wu, s_up, "WEIGHT_UPDATE")` |
| last batch | L935-960 各 cudaGraphLaunch | 同上模式替换为 `LS_OR_P` |

#### (c) 跳过 TransferStation 等待

在 L843 行 `while (!ts->buffer_is_readable(0))` **之前**：

```cpp
ts->set_buffer_readable(0, true); ts->set_buffer_readable(1, true);
ts->set_buffer_writeable(0, false); ts->set_buffer_writeable(1, false);
```

在 L895 循环内 `while (!ts->buffer_is_readable(next_buf))` **之前**同样加两行。

#### (d) 跳过 Preprocessor 线程

在 `run_gpu()` 中 L710-716，注释掉 `prep_thread`：

```cpp
// [DRY-RUN]
// std::exception_ptr prep_exc;
// std::thread prep_thread(...);
// prep_thread.join();
// if (prep_exc) std::rethrow_exception(prep_exc);
```

### 1B：测试文件

**创建**: `tests/correction/test_dl_dry.cpp`

```cpp
#include "renaissance.h"
#include <iostream>
using namespace tr;

int main() {
    GLOBAL_SETTING
        .use_gpu("0").manual_seed(42)
        .local_batch_size(5000)   // ← 大幅缩减 batch 数
        .train_resolution(28).val_resolution(28).amp(false);

    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1).load_workers(1).preprocess_workers(1)
        .cpu_binding(false).normalization(NormMode::MNIST)
        .train_transforms(DoNothing()).val_transforms(DoNothing())
        .commit();

    BluePrint mlp = seq(fc(512, true), relu(), fc(256, true), relu(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp).loss(CrossEntropyLoss())
        .total_epochs(1)
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .scheduler(CosineAnnealingLR().base_lr(0.01f).eta_min(1e-5f).step_by_epoch())
        .validate_every(1, 1)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    std::cout << "=== Starting compile() ===\n";
    task.compile();
    std::cout << "=== compile() PASSED ===\n";

    auto result = task.run();
    std::cout << "=== run() PASSED, best_epoch=" << result.best_epoch << " ===\n";
    return 0;
}
```

### 1C：编译 + 运行

```powershell
cmake --build . --config Release --target test_dl_dry -- /m
.\bin\tests\correction\test_dl_dry.exe
```

### 1D：判断标准

| 阶段 | 检查内容 | 怎么判断 |
|------|----------|----------|
| compile 中 | `train_cg_` 非空 | 程序不崩溃到 L248 之后 |
| compile 中 | atlas slot >= 10 | 无 TR_CHECK 失败 |
| compile 中 | pre_capture 成功 | `captured_result_.graphs.size() >= 10` |
| compile 中 | exec_table 全部非空 | 无 TR_CHECK `"Required graph slot ...nullptr"` |
| compile 中 | phase == COMPILED | 无 `check_phase` 失败 |
| compile 中 | init_all 成功 | 无 TR_CHECK 失败 |
| compile 中 | cudaMallocHost 成功 | 无 `cudaMallocHost failed` |
| run 中 | batch 0 pre-transmit | `[DRY] r0 XFER_A(pre) s=TR` |
| run 中 | 四阶段序列 | 每 batch `ZERO_GRAD → FIRST_FWD → DEEP_FWD_BWD\|XFER → FIRST_BWD\|DEEP_ALLREDUCE → WEIGHT_UPDATE` |
| run 中 | XFER 交替 | 偶数 batch 印 `XFER` 表示下一个 buffer 正确 |
| run 结束 | TrainingResult 返回 | `best_epoch=-1`（无验证数据） |

### 1E：清理

**立即执行**（进入第二步前）：
1. 删除 `#define DRY_RUN_CUDA_GRAPH` 至 `#endif` 整个宏块
2. 恢复所有 `L_OR_P`/`LX_OR_P`/`LS_OR_P` → `cudaGraphLaunch`
3. 删除 buffer 手动标记行
4. 恢复 Preprocessor 线程的 4 行代码

---

## 第二步：Transfer 逐层验证

> 分为 2A（硬件正确性）+ 2B（DL 路径正确性）+ 2C（AB 带宽）三个子步。2A 用已有 SimpleTask 测试，不改 DL 源码。2B 可选，2C 用新 SimpleTask 测 AB 交替。

### 2A：硬件 H2D 正确性（已有测试，不改代码）

```powershell
cmake --build . --config Release --target test_h2d_copy_a -- /m
.\bin\tests\correction\test_h2d_copy_a.exe
```

**判断**：输出 `max|diff| = 0.000000` 或 `< 1e-6`。

### 2B：DeepLearningTask 路径 Transfer 验证（可选）

如果怀疑 Compiler 生成的 Transfer Graph 与 SimpleTask `RANGE_H2D_COPY_A` 有差异，在 `run_train_epoch_gpu()` 的 batch 0 pre-transmit 块（L845 sync_tr 后）插入验证块：

```cpp
// ===== [STEP 2B: 验证后删除] =====
{
    uint8_t* src = ts->get_buffer_ptr(0);
    size_t label_bytes = (size_t)registry.get_local_batch_size() * sizeof(int32_t);
    size_t label_align = ((label_bytes + 16 + 255) / 256) * 256;
    size_t buf_bytes = label_align + (size_t)registry.get_local_batch_size() * 28 * 28 * 1 * sizeof(float);
    std::vector<uint8_t> local(buf_bytes);
    std::memcpy(local.data(), src, buf_bytes);

    const DTensor* ia = nullptr;
    for (auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_DATA) { ia = &d; break; }
    }
    TR_CHECK(ia, ValueError, "no I_A_DATA DTensor");

    Tensor g = fetch_from_rank(*ia, rank);
    auto* gf = g.data<float>();
    auto* lf = (const float*)(local.data() + label_align);
    double md = 0;
    for (int64_t i = 0; i < g.numel(); ++i) {
        double d = std::abs((double)gf[i] - (double)lf[i]);
        if (d > md) md = d;
    }
    TR_CHECK(md < 1e-6, ValueError, "Transfer corruption: max|diff|=" << md);
    std::cout << "[STEP2B] rank=" << rank << " OK\n";
}
// ===== [STEP 2B END] =====
```

同时需在 `run_gpu()` 中手动填充 buffer 0（用已知递增模式），注释 Prep 线程。判断：`max|diff| < 1e-6`。**验证后删除所有 2B 代码**。

### 2C：AB 区交替传输带宽

**创建**: `tests/correction/perf_dl_xfer_ab.cpp`

```cpp
#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cmath>
using namespace tr;
using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

int main() {
    GLOBAL_SETTING
        .use_gpu("0").manual_seed(42)
        .local_batch_size(128).train_resolution(28).val_resolution(28).amp(false);
    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1).load_workers(1).preprocess_workers(1)
        .cpu_binding(false).normalization(NormMode::MNIST)
        .train_transforms(DoNothing()).val_transforms(DoNothing()).commit();

    auto& reg = GlobalRegistry::instance();
    int bs = reg.get_local_batch_size();
    int res = reg.train_sample_resolution_begin();
    int ch = reg.num_color_channels();

    auto* ts = static_cast<TransferStation*>(reg.transfer_station_ptr(0));
    for (int b = 0; b < 2; ++b) {
        uint8_t* p = ts->get_buffer_ptr(b);
        auto* lb = (int32_t*)p;
        for (int i = 0; i < bs; ++i) lb[i] = i % 10;
        ts->set_buffer_readable(b, true);
        ts->set_buffer_writeable(b, false);
    }

    SimpleTask task;
    Shape ls{bs, 1, 1, 1};
    Shape ds{bs, res, res, ch};
    task.alloc(ls, DType::INT32, Region::I_A_LABEL);
    task.alloc(ds, DType::FP32, Region::I_A_DATA);
    task.alloc(ls, DType::INT32, Region::I_B_LABEL);
    task.alloc(ds, DType::FP32, Region::I_B_DATA);
    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph ga, gb;
    { GraphNode n; n.kind = GraphNode::Kind::RANGE; n.range_op = RangeOp::RANGE_H2D_COPY_A;
      n.output_ranges = {mp.region_range(Region::I_A_LABEL), mp.region_range(Region::I_A_DATA)};
      ga.append(std::move(n)); }
    { GraphNode n; n.kind = GraphNode::Kind::RANGE; n.range_op = RangeOp::RANGE_H2D_COPY_B;
      n.output_ranges = {mp.region_range(Region::I_B_LABEL), mp.region_range(Region::I_B_DATA)};
      gb.append(std::move(n)); }
    task.add_graph("xfer_a", std::move(ga), StreamKind::TRANS);
    task.add_graph("xfer_b", std::move(gb), StreamKind::TRANS);
    task.compile(false);

    const int warmup = 20, iters = 200;
    for (int i = 0; i < warmup; ++i) { task.run("xfer_a"); task.run("xfer_b"); }

    std::vector<double> lats;
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        task.run("xfer_a");
        task.run("xfer_b");
        auto t1 = Clock::now();
        lats.push_back((double)std::chrono::duration_cast<us>(t1 - t0).count());
    }
    double mean = std::accumulate(lats.begin(), lats.end(), 0.0) / iters;

    size_t per_bytes = (size_t)DistributedTensor::compute_slot_bytes(ls, DType::INT32, Region::I_A_LABEL)
                     + (size_t)DistributedTensor::compute_slot_bytes(ds, DType::FP32, Region::I_A_DATA);
    double bw = (per_bytes * 2.0) / (mean / 1e6) / (1024.0*1024.0*1024.0);

    std::cout << std::fixed << std::setprecision(2)
              << "AB bytes: " << (per_bytes*2) << " (" << (per_bytes*2/1024.0/1024.0) << " MB)\n"
              << "Mean: " << mean << " us  BW: " << bw << " GB/s\n"
              << (bw >= 2.0 ? "PASS" : "WARN: below 2 GB/s") << std::endl;
    return bw >= 2.0 ? 0 : 1;
}
```

```powershell
cmake --build . --config Release --target perf_dl_xfer_ab -- /m
.\bin\tests\correction\perf_dl_xfer_ab.exe
```

**判断**：带宽 ≥ 2 GB/s。

---

## 第三步：完整 GPU 训练

### 3A：检查环境

确认以下已清理（第一步和第二步的临时代码）：
- [ ] `DRY_RUN_CUDA_GRAPH` 宏删除
- [ ] `L_OR_P`/`LX_OR_P`/`LS_OR_P` → `cudaGraphLaunch`
- [ ] 手动 buffer 标记删除
- [ ] 2B 验证块删除
- [ ] Preprocessor 线程恢复

### 3B：测试文件

**创建**: `tests/correction/test_dl_full.cpp`

```cpp
#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
using namespace tr;

int main() {
    GLOBAL_SETTING
        .use_gpu("0").manual_seed(42)
        .local_batch_size(128)
        .train_resolution(28).val_resolution(28).amp(false);

    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1).load_workers(1).preprocess_workers(4)
        .cpu_binding(true).normalization(NormMode::MNIST)
        .train_transforms(DoNothing()).val_transforms(DoNothing()).commit();

    BluePrint mlp = seq(fc(512, true), relu(), fc(256, true), relu(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp).loss(CrossEntropyLoss())
        .total_epochs(3)
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f).nesterov(false).dampening(0.0f))
        .scheduler(CosineAnnealingLR().base_lr(0.01f).eta_min(1e-5f).step_by_epoch())
        .validate_every(1, 1)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    auto t0 = std::chrono::steady_clock::now();
    task.compile();
    auto result = task.run();
    auto t1 = std::chrono::steady_clock::now();

    std::cout << std::fixed << std::setprecision(3)
              << "\n=====================================\n"
              << " Best Top-1: " << result.best_top1 * 100.0f << "%\n"
              << " Best Epoch: " << result.best_epoch << "\n"
              << " Time: " << std::chrono::duration<double>(t1-t0).count() << " s\n"
              << "=====================================\n"
              << (result.best_top1 > 0.85f ? "PASS" : "FAIL (< 85%)") << std::endl;
    return result.best_top1 > 0.85f ? 0 : 1;
}
```

### 3C：编译 + 运行

```powershell
cmake --build . --config Release --target test_dl_full -- /m
.\bin\tests\correction\test_dl_full.exe
```

### 3D：判断标准

| 阶段 | 检查点 | 通过条件 |
|------|--------|----------|
| compile | 无异常 | 无 TR_CHECK / 无 `cudaError` |
| epoch 1 日志 | train_loss | < 2.0（随机初始 ~2.3） |
| epoch 1 日志 | lr 值 | ≈ 0.01 ± 0.001 |
| epoch 2 日志 | train_loss ↓ | < epoch 1 的值 |
| epoch 2 验证 | val_top1 | ≥ 70% |
| epoch 3 验证 | val_top1 | ≥ 85% 且 > epoch 2 |
| 最终输出 | best_epoch | ≥ 1 |
| 析构 | ~DeepLearningTask | cudaFreeHost 均成功 |

### 3E：如果 loss 不下降

```
Step 3.1: 确认 Preprocessor 数据正确
  → 注释 run_train_epoch_gpu() 的 loop，只跑 1 个 batch 并打印 sample[0] 的 pixel 值
Step 3.2: 检查 ALLREDUCE 是否生效
  → fetch_from_rank 取 fc1.weight 在 ALLREDUCE 前后对比（应变化 ≈ grad_size * lr）
Step 3.3: 检查 WEIGHT_UPDATE 是否生效
  → 同上手法在 WEIGHT_UPDATE 前后对比
Step 3.4: 检查 optimizer LR 是否正确写入
  → 确认 lr_dev_ptr 指向的 GPU 地址内容 = fetch_lr_for_batch 值
```

---

## CMakeLists.txt 修改

**文件**: `tests/correction/CMakeLists.txt`，末尾追加：

```cmake
# ============================================================================
# DeepLearningTask GPU 四步测试（最终版）
# ============================================================================

function(setup_dl_test name)
    target_link_libraries(${name} PRIVATE renaissance)
    target_compile_definitions(${name} PRIVATE TR_LOG_LEVEL=1)
    if(TR_USE_CUDA)
        setup_gpu_runtime_env(${name})
    endif()
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
        WIN32_EXECUTABLE FALSE)
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    endif()
endfunction()

add_executable(test_dl_dry test_dl_dry.cpp)
setup_dl_test(test_dl_dry)

add_executable(perf_dl_xfer_ab perf_dl_xfer_ab.cpp)
setup_dl_test(perf_dl_xfer_ab)

add_executable(test_dl_full test_dl_full.cpp)
setup_dl_test(test_dl_full)
```

---

## 执行检查清单

```
[ ] Step 1: DRY_RUN 宏插入                      → 编译、运行
[ ] Step 1: 所有 cudaGraphLaunch 替换为 L_OR_P   → 打印正确序列
[ ] Step 1: buffer + Prep 跳过                   → 无 Hang
[ ] Step 1: compile() PASS → run() PRINT → run() 返回
[ ] Step 1 清理: 删除所有 DRY_RUN 标记

[ ] Step 2A: test_h2d_copy_a.exe                 → max|diff| < 1e-6
[ ] Step 2B: (可选) DL Transfer 验证              → max|diff| < 1e-6
[ ] Step 2C: perf_dl_xfer_ab.exe                 → BW ≥ 2 GB/s

[ ] Step 3: 确认所有临时代码已清理
[ ] Step 3: test_dl_full.exe                     → Top-1 ≥ 85%
[ ] Step 3: cudaFreeHost 正常（进程退出无异常）
```

---

## 各步参数速查

| 参数 | Step 1 | Step 2A | Step 2C | Step 3 |
|------|--------|---------|---------|--------|
| `local_batch_size` | 5000 | 128 | 128 | 128 |
| `total_epochs` | 1 | — | — | 3 |
| Preprocessor | 跳过 | 跳过 | 手工 | 正常 |
| CUDA Graph | 只打印 | 全跑 | 全跑 | 全跑 |
| 改 DL 源码 | 是（大量） | 否 | 否 | **否** |
| 新建文件 | test_dl_dry.cpp | — | perf_dl_xfer_ab.cpp | test_dl_full.cpp |