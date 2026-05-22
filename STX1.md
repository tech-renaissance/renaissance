# DeepLearningTask GPU 训练 — 精密四步测试计划（代码修改级）

> 基于 [STEP.md](file:///r:/renaissance/STEP.md) 四步思路，**逐行**描述每一步需要的代码修改。
> 与 [STEP1.md](file:///r:/renaissance/STEP1.md)（方案级）互补——本文负责"改哪行、判断什么"，STEP1.md 负责"为什么这样设计"。

---

## 基础设施：编译命令

```powershell
# 在项目根目录下
mkdir build\release -Force
cd build\release
cmake ..\.. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
cmake --build . --config Release --target test_dl_dry test_dl_full -- /m
```

后续每一步只需要 `cmake --build . --config Release --target <target> -- /m`。

---

## 第一步：Dry-Run — 跑通 compile + print-only

### 1.0 前置修改：添加 DRY_RUN 开关宏

**文件**: `src/task/deep_learning_task.cpp`，行 786 之前插入：

```cpp
// ========== [DRY-RUN: 测试用, Step 1 完成后删除此块] ==========
#define DRY_RUN_CUDA_GRAPH
#ifdef DRY_RUN_CUDA_GRAPH
#include <iostream>
#define LAUNCH_OR_PRINT(g, s, label)                                    \
    do {                                                                \
        if (g) {                                                        \
            std::cout << "[DRY] rank=" << rank << " batch=" << batch   \
                      << " " << label                                   \
                      << " s=" << (s == s_c1 ? "COMP_1" :              \
                                   s == s_c2 ? "COMP_2" :              \
                                   s == s_up  ? "UPDATE" : "TRANS")    \
                      << std::endl;                                     \
        }                                                               \
    } while(0)
#define LAUNCH_XFER_OR_PRINT(g, s, label)                               \
    do {                                                                \
        if (g) {                                                        \
            std::cout << "[DRY] rank=" << rank << " batch=" << batch   \
                      << " " << label                                   \
                      << " s=TRANS" << std::endl;                       \
        }                                                               \
    } while(0)
#else
#define LAUNCH_OR_PRINT(g, s, label) \
    do { if (g) cudaGraphLaunch(g, s); } while(0)
#define LAUNCH_XFER_OR_PRINT(g, s, label) LAUNCH_OR_PRINT(g, s, label)
#endif
// ========== [DRY-RUN END] ==========
```

### 1.1 修改 run_train_epoch_gpu() — 替换 cudaGraphLaunch 调用

**文件**: `src/task/deep_learning_task.cpp`，对以下所有 `cudaGraphLaunch` 进行替换：

| 原代码位置（行号） | 原调用 | 替换为 |
|---|---|---|
| L845 | `cudaGraphLaunch(g_xfer_a, s_trans);` | `LAUNCH_XFER_OR_PRINT(g_xfer_a, s_trans, "XFER_A(pre)");` |
| L854 | `cudaGraphLaunch(g_zg, s_up);` | `LAUNCH_OR_PRINT(g_zg, s_up, "ZERO_GRAD");` |
| L855 | `cudaGraphLaunch(g_fwd_a, s_c1);` | `LAUNCH_OR_PRINT(g_fwd_a, s_c1, "FIRST_FWD_A");` |
| L858 | `cudaGraphLaunch(g_deep_a, s_c1);` | `LAUNCH_OR_PRINT(g_deep_a, s_c1, "DEEP_FWD_BWD");` |
| L862 | `cudaGraphLaunch(g_first, s_c1);` | `LAUNCH_OR_PRINT(g_first, s_c1, "FIRST_BWD");` |
| L864 | `cudaGraphLaunch(g_dar, s_up);` | `LAUNCH_OR_PRINT(g_dar, s_up, "DEEP_ALLREDUCE");` |
| L867 | `cudaGraphLaunch(g_gc, s_up);` | `LAUNCH_OR_PRINT(g_gc, s_up, "CAST_AND_CHECK");` |
| L875 | `cudaGraphLaunch(g_far, s_up);` | `LAUNCH_OR_PRINT(g_far, s_up, "FIRST_ALLREDUCE");` |
| L876 | `cudaGraphLaunch(g_wu, s_up);` | `LAUNCH_OR_PRINT(g_wu, s_up, "WEIGHT_UPDATE");` |
| L889-899 | 循环内 Phase 1-4 全部替换 | 见下方详细清单 |

**循环内替换清单**（行 889-899 区域，循环内每条）：

```
L890  cudaGraphLaunch(g_zg, s_up)    → LAUNCH_OR_PRINT(g_zg, s_up, "ZERO_GRAD")
L891  cudaGraphLaunch(g_fwd, s_c1)   → LAUNCH_OR_PRINT(g_fwd, s_c1, "FIRST_FWD")
L899  cudaGraphLaunch(g_deep, s_c1)  → LAUNCH_OR_PRINT(g_deep, s_c1, "DEEP_FWD_BWD")
L900  cudaGraphLaunch(g_xfer_n, s_trans) → LAUNCH_XFER_OR_PRINT(g_xfer_n, s_trans, "XFER")
L909  cudaGraphLaunch(g_first, s_c1)  → LAUNCH_OR_PRINT(g_first, s_c1, "FIRST_BWD")
L911  cudaGraphLaunch(g_dar, s_up)   → LAUNCH_OR_PRINT(g_dar, s_up, "DEEP_ALLREDUCE")
L914  cudaGraphLaunch(g_gc, s_up)    → LAUNCH_OR_PRINT(g_gc, s_up, "CAST_AND_CHECK")
L924  cudaGraphLaunch(g_far, s_up)   → LAUNCH_OR_PRINT(g_far, s_up, "FIRST_ALLREDUCE")
L925  cudaGraphLaunch(g_wu, s_up)    → LAUNCH_OR_PRINT(g_wu, s_up, "WEIGHT_UPDATE")
```

**Last batch 区域**（行 935-960 附近）同理全部替换。

### 1.2 跳过 TransferStation 等待 — 不做真正数据传输

**文件**: `src/task/deep_learning_task.cpp`

在第 843 行 `while (!ts->buffer_is_readable(0))` 循环**之前**，手动设置 buffer 状态：

```cpp
// [DRY-RUN] 手动标记 buffer 可读，跳过实际数据传输
ts->set_buffer_readable(0, true);
ts->set_buffer_readable(1, true);
ts->set_buffer_writeable(0, false);
ts->set_buffer_writeable(1, false);
```

在循环内 `while (!ts->buffer_is_readable(next_buf))`（行 895）**之前**同样设置：

```cpp
// [DRY-RUN 循环内] 手动标记
ts->set_buffer_readable(0, true);
ts->set_buffer_readable(1, true);
```

### 1.3 跳过 Preprocessor 线程

**文件**: `src/task/deep_learning_task.cpp`，行 710-716，注释掉：

```cpp
// [DRY-RUN] Preprocessor 线程注释
// std::exception_ptr prep_exc;
// std::thread prep_thread([&]() { ... });
// prep_thread.join();
// if (prep_exc) std::rethrow_exception(prep_exc);
```

### 1.4 验证子步骤与检查点

**编译阶段**（`task.compile()` 执行完毕）：

| 检查点 | 判断方式 | 判断值 |
|--------|----------|--------|
| `on_prepare()` 存入 `train_cg_` | 在 `compile_impl()` L248 后加 `printf("train_cg_ nodes: %zu\n", dl->train_cg_->total_nodes());` | `> 0` |
| `build_graph_atlas()` atlas 非空 | `printf("atlas slot count: %d\n", atlas.total_slots());` | `>= 10` |
| `pre_capture()` 成功 | `printf("captured %zu graphs\n", captured_result_.graphs.size());` | `>= 10` |
| `build_exec_table()` 全部非空 | `printf("g_tab[0..N]: %p %p ...\n", g[0], g[1], ...);` | 所有 `!= nullptr` |
| `compile_mark_compiled()` phase | `printf("phase: %d (expect %d)\n", phase_, Phase::COMPILED);` | `phase_ == 2` |
| `init_all()` 正常 | 打印 `"init_all PASSED"` | 无 TR_CHECK 失败 |
| `cudaMallocHost` 成功 | 打印 `"lr_pinned_ allocated %p\n"` | `!= nullptr` |

**运行阶段**（`task.run()` 执行中）：

| 检查点 | 判断方式 |
|--------|----------|
| 循环批次正确 | 核对 `batch=0..N-2` 打印序列 + last batch |
| XFER_A/B 交替 | 检查 `batch 偶数→XFER_B`、`batch 奇数→XFER_A` 模式 |
| 四阶段序列正确 | 每 batch 内部 Phase1→Phase2→Phase3→Phase4 顺序 |
| `lr_pinned_` 被写入 | 在 fetch_lr_for_batch 后打印 `"lr=%.6f\n"` |

### 1.5 第一步成功后的清理

**必须还原**（进入 Step 2 之前）：
1. 删除 `DRY_RUN_CUDA_GRAPH` 宏及所有 `LAUNCH_OR_PRINT` / `LAUNCH_XFER_OR_PRINT`
2. 删除 `ts->set_buffer_readable/writeable` 手动设置
3. 恢复 Preprocessor 线程代码
4. 恢复所有 `cudaGraphLaunch` → `LAUNCH_OR_PRINT` 替换为原始调用

---

## 第二步：A 区 Transfer 逐字节正确性

### 2.1 代码修改计划

**核心思路**：在 `run_train_epoch_gpu()` 的 batch 0 pre-transmit 块之后、统一循环之前插入对比逻辑。

**文件**: `src/task/deep_learning_task.cpp`

在第 850 行 `sync_tr()` 之后、第 852 行单 batch 判断之前插入：

```cpp
// ========== [STEP 2: 传输正确性验证, 测试完后删除] ==========
{
    // 1) 本地副本
    uint8_t* src = ts->get_buffer_ptr(0);
    // TransferStation 的 buffer 布局:
    //   [0..label_aligned)        : int32_t labels[batch_size]
    //   [label_aligned..buf_size) : float   images[total_pixels]
    size_t label_bytes = static_cast<size_t>(registry.get_local_batch_size()) * sizeof(int32_t);
    size_t label_aligned = ((label_bytes + 16 + 255) / 256) * 256;
    size_t buf_bytes = label_aligned
        + static_cast<size_t>(registry.get_local_batch_size())
            * 28 * 28 * 1 * sizeof(float);

    std::vector<uint8_t> local_copy(buf_bytes);
    std::memcpy(local_copy.data(), src, buf_bytes);

    // 2) 从 GPU 取回 I_A_DATA
    const DTensor* ia_data = nullptr;
    for (const auto& dt : active_memory_plan_->dtensors()) {
        if (dt.region == Region::I_A_DATA) { ia_data = &dt; break; }
    }
    if (!ia_data) {
        std::cerr << "STEP2 FAIL: no I_A_DATA DTensor in memory_plan\n";
        exc[rank] = std::make_exception_ptr(std::runtime_error("no I_A_DATA"));
        return;
    }

    Tensor gpu_tensor = fetch_from_rank(*ia_data, rank);
    const float* gpu_f = gpu_tensor.data<float>();
    const float* local_f = reinterpret_cast<const float*>(local_copy.data() + label_aligned);

    int64_t numel = gpu_tensor.numel();
    double max_diff = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
        double d = std::abs(static_cast<double>(gpu_f[i]) - static_cast<double>(local_f[i]));
        if (d > max_diff) { max_diff = d; }
    }

    if (max_diff > 1e-6) {
        std::cerr << "STEP2 FAIL: max|diff| = " << max_diff
                  << " (expected 0.0)\n";
        exc[rank] = std::make_exception_ptr(
            std::runtime_error("transfer corruption detected"));
        return;
    }
    std::cout << "[STEP2] rank=" << rank << " transfer OK, max|diff|="
              << std::scientific << max_diff << std::endl;
}
// ========== [STEP 2 END] ==========
```

添加后立即 `return;` 退出函数（只测传输不分发）。

### 2.2 前置条件 — 手动填充 TransferStation buffer

在 `run_gpu()` 中，`prep_thread` 启动之前插入：

```cpp
// [STEP 2] 手动填充已知模式数据到 buffer 0
{
    auto* ts2 = static_cast<TransferStation*>(reg.transfer_station_ptr(0));
    uint8_t* buf0 = ts2->get_buffer_ptr(0);
    int batch_sz = reg.get_local_batch_size();
    size_t label_bytes = static_cast<size_t>(batch_sz) * sizeof(int32_t);
    size_t label_aligned = ((label_bytes + 16 + 255) / 256) * 256;

    auto* labels = reinterpret_cast<int32_t*>(buf0);
    for (int i = 0; i < batch_sz; ++i) labels[i] = i % 10;

    auto* images = reinterpret_cast<float*>(buf0 + label_aligned);
    int total_px = batch_sz * 28 * 28 * 1;
    for (int i = 0; i < total_px; ++i)
        images[i] = static_cast<float>(i % 997) * 0.001f;

    ts2->set_buffer_readable(0, true);
}
```

### 2.3 验证子步骤

| 子步骤 | 判断 |
|--------|------|
| `ts->get_buffer_ptr(0)` 返回非空 | `REQUIRE(ret != nullptr)` |
| `active_memory_plan_->dtensors()` 能找到 `I_A_DATA` | `REQUIRE(ia_data != nullptr)` |
| `fetch_from_rank()` 返回 tensor shape 正确 | `tensor.shape == [batch, 28, 28, 1]` |
| 逐 float 对比 `max|diff| < 1e-6` | **核心判定** |

### 2.4 成功后的清理

删除 `[STEP 2]` 块（包含手动 buffer 填充），恢复 `run_train_epoch_gpu()` 正常流程。

---

## 第三步：AB 区传输性能

### 3.1 方法选择

**首选方案**：复用 `tests/correction/perf_h2d_copy_a.cpp` 框架，扩展为 AB 交替。

不修改 DeepLearningTask 源码。使用 SimpleTask + `RANGE_H2D_COPY_A` + `RANGE_H2D_COPY_B` 两个图。原因：
- deep_learning_task 的 transfer 图嵌入流水线，单抽出来需要改大量代码
- SimpleTask 的 RANGE_H2D_COPY 图已验证正确性（Step 2 已覆盖底层硬件正确性）
- 传输性能取决于 PCIe 带宽，与上层图结构无关

### 3.2 测试文件

**创建**: `tests/correction/perf_dl_xfer_ab.cpp`（完整代码见 STEP1.md §3）

### 3.3 测量指标

| 指标 | 计算方式 | 参考值（RTX 4060 PCIe 4.0 x8） |
|------|----------|------|
| 单次 AB 延迟 | `mean(latencies)` | ≤ 500 µs @ batch=128 MNIST |
| 有效带宽 | `2 × per_xfer_bytes / mean` | ≥ 4 GB/s |
| 抖动 | `stddev / mean` | ≤ 20% |

### 3.4 验证检查点

- [ ] 200 次迭代无 CUDA 错误
- [ ] 带宽 ≥ 4 GB/s（PCIe 4.0 x8 理论 ~8 GB/s，保守下限）
- [ ] 延迟 stddev < mean × 0.3

---

## 第四步：完整 GPU 训练

### 4.1 前置条件 — 代码已全部还原

确认以下内容**未残留**：
- [ ] `DRY_RUN_CUDA_GRAPH` 宏已删除
- [ ] 所有 `LAUNCH_OR_PRINT` 已恢复为 `cudaGraphLaunch`
- [ ] STEP 2 的手动 buffer 填充和对比代码已删除
- [ ] Preprocessor 线程已恢复
- [ ] `run_train_epoch_gpu()` 末尾无 `return;`

### 4.2 测试文件

**创建**: `tests/correction/test_dl_full.cpp`（完整代码见 STEP1.md §4.1）

**关键参数**（选自 STEP.md）:

```cpp
.total_epochs(3)
.local_batch_size(128)
```

### 4.3 运行时验证 — 无额外代码修改

仅通过 stdout 输出判断。关键输出包括：

```text
[EPOCH 1/3] train_loss=1.234  val_loss=0.876  val_top1=72.3%  lr=0.010000   time=12.3s
[EPOCH 2/3] train_loss=0.456  val_loss=0.321  val_top1=88.5%  lr=0.009000   time=11.8s
[EPOCH 3/3] train_loss=0.234  val_loss=0.198  val_top1=91.2%  lr=0.007000   time=11.6s
```

### 4.4 验证检查点（按顺序）

| 阶段 | 检查点 | 判断 |
|------|--------|------|
| compile | `compile()` 无异常退出 | 无 TR_CHECK/Exception |
| run epoch 1 | 训练 loss 打印 | `train_loss < 2.0`（随机初始 ≈ 2.3） |
| run epoch 1 | `fetch_lr_for_batch()` | `lr ≈ 0.01`（`cosine base_lr`） |
| run epoch 2 | training loss ↓ | 相比 epoch 1 显著下降 |
| run epoch 2 | val top-1 | ≥ 70% |
| run epoch 3 | val top-1 | ≥ 85% ↑（比 epoch 2 更高） |
| run end | `log_final_summary` | 打印总时间、最佳指标 |
| run end | `TrainingResult::best_epoch` | ≥ 1 |
| post-run | ~DeepLearningTask() 不崩 | `cudaFreeHost` 成功 |

### 4.5 如果 loss 不下降 — 排查流程图

```
run() 执行无错误但 loss 不下降
        ↓
    确认 Preprocessor 有真实数据
        ↓ (YES)
    在 batch 0 的 ALLREDUCE 前后分别 fetch_from_rank 取权重对比
        ↓ (权重在 ALLREDUCE 后无变化)
    检查 DEEP_ALLREDUCE Graph 是否正确 launch
        → gpu_exec_.graphs[0][S(DEEP_ALLREDUCE)] != nullptr?
        → cudaGraphLaunch 返回值是否为 cudaSuccess?
        ↓
    在 WEIGHT_UPDATE 前后取权重对比
        ↓
    如果还无变化 → 检查 MemoryPlan 中 optimizer 的 DTensor offset 是否正确
```

---

## CMakeLists.txt 修改清单

**文件**: `tests/correction/CMakeLists.txt`，末尾追加：

```cmake
# ============================================================================
# DeepLearningTask GPU 四步测试
# ============================================================================

set(DL_TEST_OPTIONS /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)

function(setup_dl_test target_name)
    target_link_libraries(${target_name} PRIVATE renaissance)
    target_compile_definitions(${target_name} PRIVATE TR_LOG_LEVEL=1)
    if(TR_USE_CUDA)
        setup_gpu_runtime_env(${target_name})
    endif()
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
        WIN32_EXECUTABLE FALSE
    )
    if(MSVC)
        target_compile_options(${target_name} PRIVATE ${DL_TEST_OPTIONS})
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    endif()
endfunction()

# Step 1: Dry-run
add_executable(test_dl_dry test_dl_dry.cpp)
setup_dl_test(test_dl_dry)

# Step 3: AB transfer perf
add_executable(perf_dl_xfer_ab perf_dl_xfer_ab.cpp)
setup_dl_test(perf_dl_xfer_ab)

# Step 4: Full training
add_executable(test_dl_full test_dl_full.cpp)
setup_dl_test(test_dl_full)

# Steps 2 + 3 备用：复用已有 SimpleTask 测试（已存在，无需重新添加）
# test_h2d_copy_a  ← Step 2 首选
# perf_h2d_copy_a  ← Step 3 首选
```

---

## 测试执行序列

```powershell
# === Step 1 ===
# 1. 添加 DRY_RUN 宏和 LAUNCH_OR_PRINT 替换
# 2. 注释 Preprocessor，手动设置 buffer
# 3. 创建 test_dl_dry.cpp（见 STEP1.md §1.1）
cd build\release
cmake --build . --config Release --target test_dl_dry -- /m
.\bin\tests\correction\test_dl_dry.exe
# 检查输出中 [DRY] 序列的四阶段模式
# 检查 output 无 "FAIL"、无 "cudaError"

# === 清理 DRY_RUN ===
# 4. 删除 DRY_RUN_CUDA_GRAPH 宏块
# 5. 恢复所有 cudaGraphLaunch
# 6. 恢复 Preprocessor 线程

# === Step 2 ===
# 7. 添加 STEP 2 transfer 验证块到 run_train_epoch_gpu
# 8. 添加手动 buffer 填充到 run_gpu
# 9. 编译 test_h2d_copy_a（已是现有目标）
cmake --build . --config Release --target test_h2d_copy_a -- /m
.\bin\tests\correction\test_h2d_copy_a.exe
# 检查 max|diff| < 1e-6

# === 清理 STEP 2 ===
# 10. 删除 STEP 2 验证块和手动 buffer 填充

# === Step 3 ===
# 11. 创建 perf_dl_xfer_ab.cpp（见 STEP1.md §3）
cmake --build . --config Release --target perf_h2d_xfer_ab -- /m
.\bin\tests\correction\perf_dl_xfer_ab.exe
# 检查带宽 ≥ 4 GB/s

# === Step 4 ===
# 12. 确认所有测试代码已清理
# 13. 创建 test_dl_full.cpp（见 STEP1.md §4.1）
cmake --build . --config Release --target test_dl_full -- /m
.\bin\tests\correction\test_dl_full.exe
# 检查 Top-1 ≥ 85%
```

---

## 代码修改生命周期

```
Step 1 ──→ 添加 DRY_RUN 宏 ──→ 测试 ──→ ✂ 删除 DRY_RUN 宏
Step 2 ──→ 添加验证块   ──→ 测试 ──→ ✂ 删除验证块
Step 3 ──→ 不修改 DL 源码（使用 SimpleTask）
Step 4 ──→ 不修改 DL 源码（纯运行时测试）
```

> **唯一需要恢复的代码修改**：Step 1 的 DRY_RUN 宏 + Step 2 的验证块。这些都不是持久性代码。

---

## 附录 A：逐阶段代码定位速查

| 检查目标 | 代码位置（文件:行） |
|----------|-----|
| compile 入口 | `task_base.cpp:214` `compile_impl()` |
| DL 分支 Arena memset | `task_base.cpp:238-245` |
| build_graph_atlas | `deep_learning_task.cpp:489` |
| set_rank + set_memory_plan | `task_base.cpp:257-260` |
| pre_capture | `task_base.cpp:262` |
| build_exec_table | `deep_learning_task.cpp:553` |
| compile_mark_compiled | `task_base.cpp:277` |
| init_all | `task_base.cpp:1227` (调用 `init` 循环) |
| cudaMallocHost | `task_base.cpp:283-290` |
| run_gpu epoch loop | `deep_learning_task.cpp:669-768` |
| run_train_epoch_gpu | `deep_learning_task.cpp:787-920` |
| batch 0 pre-transmit | `deep_learning_task.cpp:842-850` |
| 统一循环 Phase 1 | `deep_learning_task.cpp:889-892` |
| 统一循环 Phase 4 | `deep_learning_task.cpp:919-925` |
| last batch | `deep_learning_task.cpp:930-960` |

## 附录 B：配置差异表（各步对比）

| 参数 | Step 1 | Step 2 | Step 3 | Step 4 |
|------|--------|--------|--------|--------|
| `local_batch_size` | **5000** | 128 | 128 | 128 |
| `total_epochs` | 1 | 1 | 1 | **3** |
| `preprocess_workers` | 1 | 1 | 1 | **4** |
| `cpu_binding` | false | false | false | **true** |
| `amp` | false | false | false | false |
| Preprocessor | **注释** | **手动填充** | 手动填充 | **正常** |
| CUDA Graph | **只打印** | 只跑 XFER | 只跑 XFER | **全跑** |
| 代码修改程度 | 大量（DRY_RUN 宏） | 中等（验证块） | 无（新建文件） | **无** |