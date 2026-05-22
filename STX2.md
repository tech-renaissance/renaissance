# STX2.md — DeepLearningTask 编译前最终测试路线图

> **版本**: v1.0 | **日期**: 2026-05-22  
> **基线**: TR4 v4.20.1, C++17, CUDA Graph 训练引擎  
> **目标**: `mnist_mlp_3.cpp` 20 epoch 真实训练（784-512-256-10 MLP）  
> **验证文档**: Z_FINAL_K.md §7.3 四阶段重叠调度  
> **策略**: 渐进式验证 — 每步只引入一个新复杂度，最小代码侵入

---

## 0. 代码基线确认

以下行号基于 **2026-05-22 当前 `main` 分支** 的精确状态。

| 文件 | 关键区域 | 当前行号 |
|------|---------|---------|
| `tests/ref/mnist_mlp_3.cpp` | 配置入口 | 17–113 |
| `src/task/deep_learning_task.cpp` | `run_gpu()` + `run_train_epoch_gpu()` | 624–974 |
| `src/task/deep_learning_task.cpp` | `build_graph_atlas()` / `build_exec_table()` | 478–617 |
| `src/task/task_base.cpp` | `compile_impl()` / `init_all()` | 214–293 / 1276–1301 |
| `include/renaissance/task/deep_learning_task.h` | 类声明 | 1–450 |

### 0.1 `run_train_epoch_gpu()` 当前结构（精简版）

```cpp
// src/task/deep_learning_task.cpp:787–974
#ifdef TR_USE_CUDA
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();   // line 789
    // ... 变量声明 ...
    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, ...]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);   // line 803
                // ... 获取 stream、graph handle ...
                // ========== Batch 0 预传输 ==========   // line 842
                while (!ts->buffer_is_readable(0)) sleep;
                cudaGraphLaunch(g_xfer_a, s_trans); sync_tr();   // line 845–846
                if (rank == 0) { ts->set_buffer_readable(0, false); ts->set_buffer_writeable(0, true); }
                // ========== 单 batch 边界 ==========   // line 852
                if (batches == 1) { ... cudaGraphLaunch(...) ... return; }
                // ========== 统一循环 batch=0..batches-2 ==========   // line 881
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a = (batch % 2 == 0);
                    // Phase 1: ZG ‖ FWD   (line 889–892)
                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                    sync_comp(); sync_up();
                    // Wait next buffer   (line 894–896)
                    while (!ts->buffer_is_readable(next_buf)) sleep;
                    // Phase 2: DEEP ‖ XFER(next)   (line 898–905)
                    cudaGraphLaunch(g_deep, s_c1);
                    cudaGraphLaunch(g_xfer_n, s_trans);
                    sync_comp(); sync_tr();
                    if (rank == 0) { set_readable(false); set_writeable(true); }
                    // Phase 3: FIRST ‖ DAR   (line 907–912)
                    if (!frozen) if (g_first) cudaGraphLaunch(g_first, s_c1);
                    if (g_dar) cudaGraphLaunch(g_dar, s_up);
                    sync_comp(); sync_up();
                    // AMP   (line 914–915)
                    if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }
                    // Phase 4: LR H2D → FAR → WU   (line 917–926)
                    lr = fetch_lr_for_batch(batch);
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float), H2D, s_up);
                    if (g_far) cudaGraphLaunch(g_far, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up);
                    sync_up();
                }
                // ========== Last batch (batches-1) ==========   // line 929
                { ... Phase 1→4 无 XFER ... }
            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }
    // join + rethrow
}
#endif
```

### 0.2 核心设施可用性确认

| 设施 | 状态 | 证据 |
|------|------|------|
| `active_memory_plan_` 指针 | ✅ 已替换 | `task_base.cpp:265`, `deep_learning_task.h:265` |
| `GraphAtlas` 去重捕获 | ✅ 已实现 | `deep_learning_task.cpp:478–517` |
| `pre_capture()` | ✅ 已调用 | `task_base.cpp:262` |
| `StagingParamPool` / `lr_pinned_` 锁页内存 | ✅ 已分配 | `task_base.cpp:284–289` |
| `TransferStation` 双缓冲 | ✅ 可用 | `transfer_station.cpp:1050–1075` |
| `cudaGraphLaunch` 直接调用 | ✅ 已实现 | `deep_learning_task.cpp:845, 854, 855, ...`（共 25 处） |
| `run_gpu()` N+1 线程 | ✅ 已实现 | `deep_learning_task.cpp:710–717` |

### 0.3 已知 Stub（不影响 GPU 训练路径）

| 函数 | 状态 | 影响 |
|------|------|------|
| `run_val_epoch_gpu()` | 仅打印 stub | Step 4 无法获得 val accuracy，只能验证"不崩溃" |
| `save_model_to()` | 仅打印 stub | Step 4 不保存模型 |
| `run_cpu()` / `run_train_epoch_cpu()` | 返回 debug_stub | CPU 路径不可用 |

> **Step 4 判定调整**：由于 `run_val_epoch_gpu()` 为 stub，accuracy 收敛性验证暂缓；Step 4 的核心目标是"所有图完整执行、不崩溃、无 CUDA 错误"。

---

## 测试策略总览

### 核心设计原则

1. **一处修改原则**：每步只在 `deep_learning_task.cpp` 的 `run_train_epoch_gpu()` rank 线程 lambda 开头插入一个条件块；外加 `mnist_mlp_3.cpp` 的配置调整。
2. **编译时切换**：使用顶部宏控制 `kStx2Mode`，改一行即切换步骤。
3. **零副作用回滚**：删除条件块 + 恢复 `mnist_mlp_3.cpp` 配置 = 完全回到基线。

### 四步流程

```
Step 1  ──→  改 kStx2Mode=STEP1, epochs=1, batch=5000
             只打印调度序列，不执行任何 CUDA Graph
             验证：compile 成功、调度顺序正确、A/B 切换无误

Step 2  ──→  改 kStx2Mode=STEP2, epochs=1, batch=128
             只执行 XFER_A，手动填充已知数据，取回 GPU 后逐元素对比
             验证：label 和 data 逐元素一致

Step 3  ──→  改 kStx2Mode=STEP3, epochs=1, batch=128
             循环执行 XFER_A/B，测 launch+sync 耗时，算等效带宽
             验证：无 CUDA 错误、带宽 ≥ 2 GB/s

Step 4  ──→  改 kStx2Mode=NONE（或删除宏块），epochs=20, batch=128
             所有图全开，Preprocessor 正常跑
             验证：20 epoch 完整跑完、不崩溃、无 CUDA 错误
```

---

## Step 1: Dry Run — 只打印调度序列

### 1.1 目标

验证 `compile()` 完整链路（on_prepare → build_graph_atlas → pre_capture → build_exec_table → init_all → cudaMallocHost）无错误；验证 `run_train_epoch_gpu()` 的调度顺序与 Z_FINAL_K.md §7.3 一致。

### 1.2 修改清单

#### 修改 A — `src/task/deep_learning_task.cpp`（添加测试模式控制）

**位置**：文件顶部，`#include <unordered_map>`（line 29）之后，插入以下宏控制块：

```cpp
// ===== STX2 测试模式控制 =====
// 取消注释以下一个 #define 以启用对应步骤
// #define STX2_STEP1   // Step 1: Dry Run — 只打印不执行
// #define STX2_STEP2   // Step 2: A 区 Transfer 正确性
// #define STX2_STEP3   // Step 3: AB 区 Transfer 性能
// Step 4: 全部注释掉 = 完整训练

#ifdef STX2_STEP1
constexpr bool stx2_dry_run = true;
constexpr bool stx2_xfer_a_only = false;
constexpr bool stx2_xfer_ab_perf = false;
#elif defined(STX2_STEP2)
constexpr bool stx2_dry_run = false;
constexpr bool stx2_xfer_a_only = true;
constexpr bool stx2_xfer_ab_perf = false;
#elif defined(STX2_STEP3)
constexpr bool stx2_dry_run = false;
constexpr bool stx2_xfer_a_only = false;
constexpr bool stx2_xfer_ab_perf = true;
#else
constexpr bool stx2_dry_run = false;
constexpr bool stx2_xfer_a_only = false;
constexpr bool stx2_xfer_ab_perf = false;
#endif
// ===== STX2 END =====
```

#### 修改 B — `src/task/deep_learning_task.cpp`（Dry Run 条件块）

**位置**：`run_train_epoch_gpu()` 函数内，rank 线程 lambda 的 `try {`（line 802）之后，**第一个语句之前**，插入：

```cpp
                // ===== STX2 Step 1: Dry Run =====
                if constexpr (stx2_dry_run) {
                    LOG_INFO << "[STX2-Step1] === START === rank=" << rank
                             << " batches=" << batches << " K=" << K
                             << " amp=" << using_amp
                             << " frozen=" << is_first_layer_frozen();
                    for (int batch = 0; batch < batches; ++batch) {
                        bool from_a = (batch % 2 == 0);
                        int next_buf = from_a ? 1 : 0;
                        auto g_fwd = from_a ? g_fwd_a : g_fwd_b;
                        auto g_deep = from_a ? g_deep_a : g_deep_b;
                        auto g_xfer_n = from_a ? g_xfer_b : g_xfer_a;
                        LOG_INFO << "[STX2-Step1] batch=" << batch
                                 << " from_a=" << from_a
                                 << " next_buf=" << next_buf
                                 << " | P1: ZG=" << (g_zg ? "Y" : "N")
                                 << " FWD=" << (g_fwd ? (from_a ? "A" : "B") : "N")
                                 << " | P2: DEEP=" << (g_deep ? (from_a ? "A" : "B") : "N")
                                 << " XFER=" << (g_xfer_n ? (from_a ? "B" : "A") : "N")
                                 << " | P3: FIRST=" << (g_first ? "Y" : "N")
                                 << " DAR=" << (g_dar ? "Y" : "N")
                                 << " | P4: LR=" << fetch_lr_for_batch(batch)
                                 << " FAR=" << (g_far ? "Y" : "N")
                                 << " WU=" << (g_wu ? "Y" : "N");
                    }
                    // Last batch (无 XFER)
                    {
                        bool last_a = ((batches - 1) % 2 == 0);
                        auto g_fwd_l = last_a ? g_fwd_a : g_fwd_b;
                        auto g_deep_l = last_a ? g_deep_a : g_deep_b;
                        LOG_INFO << "[STX2-Step1] LAST batch=" << (batches - 1)
                                 << " FWD=" << (g_fwd_l ? (last_a ? "A" : "B") : "N")
                                 << " DEEP=" << (g_deep_l ? (last_a ? "A" : "B") : "N")
                                 << " (no XFER)";
                    }
                    LOG_INFO << "[STX2-Step1] === END === rank=" << rank;
                    return;  // 直接返回，跳过所有 CUDA 调用
                }
```

#### 修改 C — `tests/ref/mnist_mlp_3.cpp`（缩减规模）

**位置**：
- line 30: `.local_batch_size(128)` → `.local_batch_size(5000)`
- line 73: `.total_epochs(20)` → `.total_epochs(1)`

### 1.3 编译命令

```powershell
cd R:\renaissance
cmake -B build/windows-msvc-release -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build/windows-msvc-release --target mnist_mlp_3 --config Release
```

### 1.4 运行命令

```powershell
.\build\windows-msvc-release\bin\tests\ref\mnist_mlp_3.exe
```

### 1.5 预期输出

```
==================================================
 Tech-Renaissance Training Started
--------------------------------------------------
 GPU IDs: [0]
 Local batch size: 5000
 ...
==================================================
[STX2-Step1] === START === rank=0 batches=12 K=1 amp=0 frozen=0
[STX2-Step1] batch=0 from_a=1 next_buf=1 | P1: ZG=Y FWD=A | P2: DEEP=A XFER=B | P3: FIRST=Y DAR=Y | P4: LR=0.01 FAR=Y WU=Y
[STX2-Step1] batch=1 from_a=0 next_buf=0 | P1: ZG=Y FWD=B | P2: DEEP=B XFER=A | P3: FIRST=Y DAR=Y | P4: LR=0.01 FAR=Y WU=Y
[STX2-Step1] batch=2 from_a=1 next_buf=1 | P1: ZG=Y FWD=A | P2: DEEP=A XFER=B | P3: FIRST=Y DAR=Y | P4: LR=0.01 FAR=Y WU=Y
...
[STX2-Step1] batch=10 from_a=0 next_buf=0 | P1: ZG=Y FWD=B | P2: DEEP=B XFER=A | P3: FIRST=Y DAR=Y | P4: LR=0.01 FAR=Y WU=Y
[STX2-Step1] LAST batch=11 FWD=B DEEP=B (no XFER)
[STX2-Step1] === END === rank=0
 Training Complete
--------------------------------------------------
 Best Val Top-1: 0.00%
 ...
```

### 1.6 通过/失败判定标准

| # | 检查项 | 通过标准 | 失败排查 |
|---|--------|---------|---------|
| 1.1 | `compile()` 返回 | 不 crash、不 assert | 检查 `build_graph_atlas()` 中 `train_cg_` 是否非空 |
| 1.2 | `build_exec_table()` | `kRequired` 全部非 nullptr | 检查 `captured_result_.atlas.index(0, gid)` 是否 ≥ 0 |
| 1.3 | `init_all()` | 不触发 phase 检查失败 | 检查 `active_memory_plan_` 是否已设置 |
| 1.4 | `cudaMallocHost` | 成功（无 DeviceError） | 检查 `num_gpus_` 是否与 `gpu_ids()` 一致 |
| 1.5 | batch 数量 | `batches == 12`（60000/5000） | 检查 `Preprocessor::steps_per_epoch()` |
| 1.6 | A/B 交替 | batch 偶数 `from_a=1`（FWD=A, XFER=B），奇数相反 | 检查循环逻辑 |
| 1.7 | next_buf | A→1, B→0 | 检查 `int next_buf = from_a ? 1 : 0` |
| 1.8 | Last batch | 无 XFER，FWD/DEEP 与 `(batches-1)%2` 一致 | 检查 last batch 块 |
| 1.9 | Graph 非空 | ZG/FIRST/DAR/FAR/WU 均显示 `Y` | 检查 `build_exec_table()` resolve 结果 |
| 1.10 | 程序正常退出 | 返回码 0，无 CUDA 错误 | 检查 stderr |

### 1.7 回滚指令

```powershell
# 1. 恢复 mnist_mlp_3.cpp
git checkout tests/ref/mnist_mlp_3.cpp

# 2. 删除 deep_learning_task.cpp 中的 STX2 宏块
# 手动删除：
#   - 文件顶部的 "===== STX2 测试模式控制 =====" 块（约 20 行）
#   - lambda 中的 "===== STX2 Step 1: Dry Run =====" 块（约 35 行）
```

---

## Step 2: A 区 Transfer 正确性验证

### 2.1 目标

验证 `XFER_A` CUDA Graph 能将 TransferStation buffer 0 的数据完整传输到 GPU 显存对应位置（I_A_LABEL / I_A_DATA）。对比本地副本与 GPU 取回数据是否逐元素一致。

### 2.2 修改清单

#### 修改 A — `src/task/deep_learning_task.cpp`（启用 Step 2）

**位置**：顶部宏控制块，取消 `STX2_STEP2` 的注释：

```cpp
// #define STX2_STEP1   // Step 1: Dry Run — 只打印不执行
#define STX2_STEP2   // Step 2: A 区 Transfer 正确性   ← 取消注释这一行
// #define STX2_STEP3   // Step 3: AB 区 Transfer 性能
```

#### 修改 B — `src/task/deep_learning_task.cpp`（Step 2 条件块）

**位置**：与 Step 1 的 Dry Run 块**同一位置**（`try {` 之后），在 Dry Run 块**下方**插入：

```cpp
                // ===== STX2 Step 2: XFER_A Correctness =====
                if constexpr (stx2_xfer_a_only) {
                    // 手动填充 buffer 0（已知模式，避免依赖 Preprocessor）
                    {
                        int bs = registry.get_local_batch_size();
                        int res = registry.train_sample_resolution_begin();
                        int ch = registry.num_color_channels();

                        // Labels: 0,1,2,...,9 循环
                        int32_t* labels = reinterpret_cast<int32_t*>(ts->get_buffer_ptr(0));
                        for (int i = 0; i < bs; ++i) labels[i] = i % 10;

                        // Data: 已知递增模式 (i % 997) * 0.001f
                        float* data = reinterpret_cast<float*>(ts->get_image_data_ptr(0));
                        int64_t data_count = static_cast<int64_t>(bs) * res * res * ch;
                        for (int64_t i = 0; i < data_count; ++i) {
                            data[i] = static_cast<float>(i % 997) * 0.001f;
                        }
                        ts->set_buffer_readable(0, true);
                        LOG_INFO << "[STX2-Step2] Manually filled buffer 0: bs=" << bs
                                 << " labels=" << bs << " data_count=" << data_count;
                    }

                    // 等待 buffer 0 可读（已满足，但为了逻辑完整性保留）
                    while (!ts->buffer_is_readable(0))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    // 执行 XFER_A
                    cudaGraphLaunch(g_xfer_a, s_trans);
                    sync_tr();
                    LOG_INFO << "[STX2-Step2] XFER_A launched and synced";

                    // 查找 I_A_LABEL 和 I_A_DATA DTensor
                    const DTensor* dt_label = nullptr;
                    const DTensor* dt_data = nullptr;
                    for (const auto& dt : active_memory_plan_->dtensors()) {
                        if (dt.region == Region::I_A_LABEL) dt_label = &dt;
                        if (dt.region == Region::I_A_DATA) dt_data = &dt;
                    }

                    bool pass = true;

                    // ---- 验证 Labels ----
                    if (dt_label) {
                        Tensor gpu_label = fetch_from_rank(*dt_label, rank);
                        const int32_t* host_labels = reinterpret_cast<const int32_t*>(
                            ts->get_buffer_ptr(0));
                        const int32_t* gpu_labels = gpu_label.data<int32_t>();
                        int label_count = gpu_label.numel();
                        for (int i = 0; i < label_count; ++i) {
                            if (host_labels[i] != gpu_labels[i]) {
                                LOG_ERROR << "[STX2-Step2] LABEL mismatch at " << i
                                          << ": host=" << host_labels[i]
                                          << " gpu=" << gpu_labels[i];
                                pass = false; break;
                            }
                        }
                        if (pass) {
                            LOG_INFO << "[STX2-Step2] LABEL check: " << label_count
                                     << " elements OK";
                        }
                    } else {
                        LOG_ERROR << "[STX2-Step2] I_A_LABEL DTensor not found in MemoryPlan";
                        pass = false;
                    }

                    // ---- 验证 Data ----
                    if (dt_data) {
                        Tensor gpu_data = fetch_from_rank(*dt_data, rank);
                        float* host_data = reinterpret_cast<float*>(ts->get_image_data_ptr(0));
                        const float* gpu_data_ptr = gpu_data.data<float>();
                        int data_count = gpu_data.numel();
                        for (int i = 0; i < data_count; ++i) {
                            if (std::abs(host_data[i] - gpu_data_ptr[i]) > 1e-6f) {
                                LOG_ERROR << "[STX2-Step2] DATA mismatch at " << i
                                          << ": host=" << host_data[i]
                                          << " gpu=" << gpu_data_ptr[i];
                                pass = false; break;
                            }
                        }
                        if (pass) {
                            LOG_INFO << "[STX2-Step2] DATA check: " << data_count
                                     << " elements OK";
                        }
                    } else {
                        LOG_ERROR << "[STX2-Step2] I_A_DATA DTensor not found in MemoryPlan";
                        pass = false;
                    }

                    LOG_INFO << "[STX2-Step2] === RESULT: " << (pass ? "PASS" : "FAIL")
                             << " ===";
                    return;  // 只验证一个 batch，直接返回
                }
```

#### 修改 C — `tests/ref/mnist_mlp_3.cpp`（恢复真实 batch size）

**位置**：
- line 30: `.local_batch_size(5000)` → `.local_batch_size(128)`（如果 Step 1 改了的话）
- line 73: `.total_epochs(1)` → `.total_epochs(1)`（保持 1，只跑一个 epoch）

> **注意**：Step 2 使用 `batch_size=128`，因为 Transfer 正确性应在真实 batch size 下验证。

### 2.3 编译命令

```powershell
cmake --build build/windows-msvc-release --target mnist_mlp_3 --config Release
```

### 2.4 运行命令

```powershell
.\build\windows-msvc-release\bin\tests\ref\mnist_mlp_3.exe
```

### 2.5 预期输出

```
...
[STX2-Step2] Manually filled buffer 0: bs=128 labels=128 data_count=100352
[STX2-Step2] XFER_A launched and synced
[STX2-Step2] LABEL check: 128 elements OK
[STX2-Step2] DATA check: 100352 elements OK
[STX2-Step2] === RESULT: PASS ===
Training Complete
...
```

### 2.6 通过/失败判定标准

| # | 检查项 | 通过标准 | 失败排查 |
|---|--------|---------|---------|
| 2.1 | `compile()` 成功 | 同 Step 1.1–1.4 | 同 Step 1 |
| 2.2 | Buffer 填充 | `Manually filled buffer 0` 日志出现 | 检查 `get_buffer_ptr(0)` 是否非空 |
| 2.3 | XFER_A 执行 | `XFER_A launched and synced` 日志出现 | 检查 `g_xfer_a` 是否为 nullptr |
| 2.4 | I_A_LABEL 找到 | `LABEL check: N elements OK` | 检查 MemoryPlan 中是否有 I_A_LABEL |
| 2.5 | I_A_DATA 找到 | `DATA check: N elements OK` | 检查 MemoryPlan 中是否有 I_A_DATA |
| 2.6 | Label 一致性 | 所有 128 个 label 完全相等 | 检查 `get_buffer_ptr()` 返回的是否为 labels 区 |
| 2.7 | Data 一致性 | 所有 pixel 的 `|diff| < 1e-6` | 检查 `get_image_data_ptr()` 返回的是否为 data 区 |
| 2.8 | 最终结果 | `[STX2-Step2] === RESULT: PASS ===` | 逐行检查 mismatch 日志定位问题 |

### 2.7 回滚指令

```powershell
# 1. 恢复 mnist_mlp_3.cpp
git checkout tests/ref/mnist_mlp_3.cpp

# 2. 注释掉 STX2_STEP2
# 在 deep_learning_task.cpp 顶部宏控制块中，将 #define STX2_STEP2 改回 // #define STX2_STEP2

# 3. 保留 STX2 宏块（Step 3/4 还要用）
```

---

## Step 3: AB 区 Transfer 性能测试

### 3.1 目标

验证 `XFER_A` / `XFER_B` 交替传输无 CUDA 错误；测量单 batch H2D 传输的等效带宽。

### 3.2 修改清单

#### 修改 A — `src/task/deep_learning_task.cpp`（启用 Step 3）

**位置**：顶部宏控制块：

```cpp
// #define STX2_STEP1
// #define STX2_STEP2
#define STX2_STEP3   // Step 3: AB 区 Transfer 性能   ← 取消注释这一行
```

#### 修改 B — `src/task/deep_learning_task.cpp`（Step 3 条件块）

**位置**：与 Step 1/2 的块**同一位置**（`try {` 之后），在其下方插入：

```cpp
                // ===== STX2 Step 3: XFER_A/B Performance =====
                if constexpr (stx2_xfer_ab_perf) {
                    // 手动填充 A/B buffer（已知模式，避免依赖 Preprocessor）
                    {
                        int bs = registry.get_local_batch_size();
                        int res = registry.train_sample_resolution_begin();
                        int ch = registry.num_color_channels();
                        int64_t data_count = static_cast<int64_t>(bs) * res * res * ch;

                        for (int buf = 0; buf < 2; ++buf) {
                            int32_t* labels = reinterpret_cast<int32_t*>(ts->get_buffer_ptr(buf));
                            for (int i = 0; i < bs; ++i) labels[i] = (i + buf * 100) % 10;
                            float* data = reinterpret_cast<float*>(ts->get_image_data_ptr(buf));
                            for (int64_t i = 0; i < data_count; ++i) {
                                data[i] = static_cast<float>((i + buf * 1000) % 997) * 0.001f;
                            }
                            ts->set_buffer_readable(buf, true);
                        }
                        LOG_INFO << "[STX2-Step3] Manually filled buffer 0/1: bs=" << bs
                                 << " data_count=" << data_count;
                    }

                    // 预热
                    const int warmup = 20;
                    for (int i = 0; i < warmup; ++i) {
                        cudaGraphLaunch(g_xfer_a, s_trans); sync_tr();
                        cudaGraphLaunch(g_xfer_b, s_trans); sync_tr();
                    }
                    LOG_INFO << "[STX2-Step3] Warmup " << warmup << " iterations done";

                    // 测量
                    const int iterations = 200;
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < iterations; ++i) {
                        cudaGraphLaunch(g_xfer_a, s_trans); sync_tr();
                        cudaGraphLaunch(g_xfer_b, s_trans); sync_tr();
                    }
                    auto t1 = std::chrono::high_resolution_clock::now();

                    double total_us = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
                    double per_xfer_us = total_us / (iterations * 2.0);

                    size_t xfer_bytes = ts->get_buffer_actual_transfer_bytes(0);
                    double xfer_mb = static_cast<double>(xfer_bytes) / (1024.0 * 1024.0);
                    double bw_gb_s = xfer_mb / (per_xfer_us / 1e6) / 1024.0;

                    LOG_INFO << "[STX2-Step3] === RESULT ===";
                    LOG_INFO << "[STX2-Step3] batch_size=" << registry.get_local_batch_size()
                             << " xfer_bytes=" << xfer_bytes
                             << " total_time=" << (total_us / 1000.0) << " ms"
                             << " per_xfer=" << per_xfer_us << " us"
                             << " BW=" << bw_gb_s << " GB/s";

                    const double min_expected_bw = 2.0;  // PCIe 保守下限
                    bool pass = (bw_gb_s >= min_expected_bw);
                    LOG_INFO << "[STX2-Step3] Bandwidth " << (pass ? "PASS" : "FAIL")
                             << " (threshold: " << min_expected_bw << " GB/s)";
                    return;
                }
```

#### 修改 C — `tests/ref/mnist_mlp_3.cpp`

保持 `batch_size=128`, `epochs=1`（同 Step 2）。

### 3.3 编译命令

```powershell
cmake --build build/windows-msvc-release --target mnist_mlp_3 --config Release
```

### 3.4 运行命令

```powershell
.\build\windows-msvc-release\bin\tests\ref\mnist_mlp_3.exe
```

### 3.5 预期输出

```
...
[STX2-Step3] Manually filled buffer 0/1: bs=128 data_count=100352
[STX2-Step3] Warmup 20 iterations done
[STX2-Step3] === RESULT ===
[STX2-Step3] batch_size=128 xfer_bytes=402944 total_time=123.45 ms per_xfer=308.62 us BW=1.27 GB/s
[STX2-Step3] Bandwidth PASS (threshold: 2.0 GB/s)
Training Complete
...
```

> 注：上述数值为示例，实际取决于硬件。MNIST 128 batch 的数据量约为 402 KB（128×28×28×1×4 + 128×4 + padding）。

### 3.6 通过/失败判定标准

| # | 检查项 | 通过标准 | 失败排查 |
|---|--------|---------|---------|
| 3.1 | `compile()` 成功 | 同 Step 1.1–1.4 | 同 Step 1 |
| 3.2 | Warmup 完成 | `Warmup 20 iterations done` | 检查 `g_xfer_a/b` 是否为 nullptr |
| 3.3 | 测量完成 | `=== RESULT ===` 出现 | 检查 CUDA 错误（如 `cudaErrorInvalidValue`） |
| 3.4 | 无 CUDA 错误 | stderr 无 `cudaError` | 检查 `cudaGetLastError()` |
| 3.5 | 带宽达标 | `BW >= 2.0 GB/s` | 如果低于阈值：检查是否使用锁页内存（`get_buffer_ptr` 应返回锁页内存） |
| 3.6 | 迭代稳定 | `per_xfer` 时间在合理范围（100–2000 us） | 如果异常大：检查同步点是否过多 |

### 3.7 回滚指令

```powershell
# 1. 恢复 mnist_mlp_3.cpp
git checkout tests/ref/mnist_mlp_3.cpp

# 2. 注释掉 STX2_STEP3
# 在 deep_learning_task.cpp 顶部宏控制块中，将 #define STX2_STEP3 改回 // #define STX2_STEP3
```

---

## Step 4: 完整训练（20 epochs）

### 4.1 目标

所有 CUDA Graph 全部启用，Preprocessor 正常供应数据，验证 20 epoch 完整训练不崩溃、无 CUDA 错误。

> **当前限制**：`run_val_epoch_gpu()` 和 `save_model_to()` 为 stub，因此无法验证 val accuracy 收敛和模型保存。本 Step 的核心目标是"训练主干路径贯通"。

### 4.2 修改清单

#### 修改 A — `src/task/deep_learning_task.cpp`（关闭所有 STX2 模式）

**位置**：顶部宏控制块，全部注释掉：

```cpp
// #define STX2_STEP1
// #define STX2_STEP2
// #define STX2_STEP3
```

#### 修改 B — `tests/ref/mnist_mlp_3.cpp`（恢复完整配置）

**位置**：
- line 30: `.local_batch_size(128)`（确认是 128）
- line 73: `.total_epochs(1)` → `.total_epochs(20)`

### 4.3 编译命令

```powershell
cmake --build build/windows-msvc-release --target mnist_mlp_3 --config Release
```

### 4.4 运行命令

```powershell
.\build\windows-msvc-release\bin\tests\ref\mnist_mlp_3.exe
```

### 4.5 预期输出

```
==================================================
 Tech-Renaissance Training Started
--------------------------------------------------
 GPU IDs: [0]
 Local batch size: 128
 World size: 1
 Total batch size: 128
 Total epochs: 20
 AMP: disabled
 SEMA: disabled
 Validate every: 1 epochs, offset: 1
 Early stop by Top-1: 0.759
==================================================
DeepLearningTask::run_val_epoch_gpu() — main model (stub)
     1 |     0.0000 |     0.0000 |    0.00% |  0.010000 |    1.2s
DeepLearningTask::run_val_epoch_gpu() — main model (stub)
     2 |     0.0000 |     0.0000 |    0.00% |  0.009876 |    1.1s
...
    20 |     0.0000 |     0.0000 |    0.00% |  0.000010 |    1.0s
==================================================
 Training Complete
--------------------------------------------------
 Best Val Top-1: 0.00%
 Best Epoch: -1
 Total Time: 0h 0m 25s
=====================================================
 Tech-Renaissance MNIST 3-Layer MLP (784-512-256-10)
-----------------------------------------------------
 Best Val Top-1:    0.000 %
 Best Epoch:        -1
 Total Time:        25.000 s
=====================================================
```

> 注：`train_loss` 和 `val_loss/top1` 当前为 0，因为 loss 收集和验证推理尚未实现。关键是看 epoch 是否从 1 跑到 20，每行日志间隔时间是否稳定。

### 4.6 通过/失败判定标准

| # | 检查项 | 通过标准 | 失败排查 |
|---|--------|---------|---------|
| 4.1 | `compile()` 成功 | 同 Step 1.1–1.4 | 同 Step 1 |
| 4.2 | Epoch 计数 | 输出显示 epoch 1 → 20 | 检查 `total_epochs_` 是否为 20 |
| 4.3 | 每 epoch 时间稳定 | 无明显异常（如 epoch 3 突然从 1s 变成 10s） | 检查是否有 CUDA 同步阻塞 |
| 4.4 | 无 CUDA 错误 | stderr 无 `cudaError`，程序返回码 0 | 检查 `cudaGetLastError()` |
| 4.5 | 无死锁 | 程序在合理时间内完成（20 epoch MNIST MLP 约 20–60s） | 检查 `buffer_is_readable` 死等 |
| 4.6 | LR 递减 | `log_epoch_results` 中的 LR 列从 0.01 逐步减小到接近 1e-5 | 检查 `CosineAnnealingLR` 配置 |
| 4.7 | Preprocessor 正常 | 无 "TransferStation buffer deadlock" 相关日志 | 检查 `prep_thread` 是否正常退出 |
| 4.8 | 无内存泄漏 | `nvidia-smi` 显存使用稳定（epoch 间不持续增长） | 检查 `cudaMalloc`/`cudaFree` 配对 |

### 4.7 回滚指令

```powershell
# 完全回滚到基线
git checkout tests/ref/mnist_mlp_3.cpp
git checkout src/task/deep_learning_task.cpp
```

> 如果 `deep_learning_task.cpp` 中除了 STX2 块外没有其他修改，上述命令即可完全恢复。

---

## 附录 A：常见问题排查

### A.1 `compile()` 阶段

| 现象 | 根因 | 排查 |
|------|------|------|
| `active_memory_plan_` nullptr assert | `on_prepare()` 未执行或 `memory_plan_ptr_` 未移动 | 检查 `compile_invoke_on_prepare()` 是否被调用 |
| `LR DTensor not found` | MemoryPlan 中无 `Region::S_SCALAR_FP32` | 检查 Compiler 是否生成了 LR DTensor |
| `build_exec_table()` assert | `resolve(GraphId)` 返回 nullptr | 检查 `captured_result_.graphs` 大小和 atlas index |
| `cudaMallocHost failed` | 系统内存不足或 CUDA 驱动问题 | 检查 `num_gpus_` 和系统可用内存 |

### A.2 `run()` 阶段

| 现象 | 根因 | 排查 |
|------|------|------|
| `cudaErrorInvalidValue` on launch | Graph handle 为 nullptr | Step 1 打印确认 `g_xfer_a` 等是否非空 |
| 死锁（程序卡住） | `buffer_is_readable` 永远 false | 检查 Preprocessor 是否正常启动；检查 `set_buffer_writeable` 是否被调用 |
| `cudaErrorLaunchFailure` | Graph 内部 kernel 参数错误 | 检查 `pre_capture()` 阶段是否有错误被忽略 |
| Step 2 label/data mismatch | TransferStation 布局与 GPU DTensor 布局不一致 | 确认 `get_buffer_ptr()` 返回 labels 区，`get_image_data_ptr()` 返回 data 区 |
| Step 3 带宽异常低 | 未使用锁页内存或同步点过多 | 确认 `get_buffer_ptr()` 返回的是锁页内存（TransferStation 应使用 `cudaMallocHost` 或 `mlock`） |

### A.3 环境检查

```powershell
# 确认 CUDA 可用
nvidia-smi

# 确认编译器
cl /?   # MSVC
# 或
g++ --version

# 确认 CMake
cmake --version

# 构建前清理（如有需要）
Remove-Item -Recurse -Force build/windows-msvc-release
cmake -B build/windows-msvc-release -S . -DCMAKE_BUILD_TYPE=Release
```

---

## 附录 B：所有修改汇总

### B.1 `src/task/deep_learning_task.cpp` 修改清单

| 修改 | 行号附近 | 内容 | 影响步骤 |
|------|---------|------|---------|
| 添加 STX2 宏控制块 | 顶部（`#include` 之后） | 定义 `stx2_dry_run` / `stx2_xfer_a_only` / `stx2_xfer_ab_perf` | 全部 |
| 添加 Step 1 Dry Run 块 | `run_train_epoch_gpu()` lambda `try {` 后 | 打印调度序列后 `return` | Step 1 |
| 添加 Step 2 XFER_A 块 | 同位置（Step 1 块下方） | 手动填充 buffer → XFER_A → fetch → 对比 | Step 2 |
| 添加 Step 3 XFER_AB 块 | 同位置（Step 2 块下方） | 手动填充 buffer → warmup → 测量 → 算带宽 | Step 3 |

### B.2 `tests/ref/mnist_mlp_3.cpp` 修改清单

| 步骤 | `.local_batch_size` | `.total_epochs` |
|------|---------------------|-----------------|
| Step 1 | 5000 | 1 |
| Step 2 | 128 | 1 |
| Step 3 | 128 | 1 |
| Step 4 | 128 | 20 |

### B.3 一键修改命令（PowerShell）

```powershell
# === Step 1: Dry Run ===
# 1. 改 mnist_mlp_3.cpp
(Get-Content tests/ref/mnist_mlp_3.cpp) `
    -replace '\.local_batch_size\(128\)', '.local_batch_size(5000)' `
    -replace '\.total_epochs\(20\)', '.total_epochs(1)' `
    | Set-Content tests/ref/mnist_mlp_3.cpp

# 2. 改 deep_learning_task.cpp 顶部宏
(Get-Content src/task/deep_learning_task.cpp) `
    -replace '// #define STX2_STEP1', '#define STX2_STEP1' `
    | Set-Content src/task/deep_learning_task.cpp

# === Step 4: 完全回滚 ===
git checkout tests/ref/mnist_mlp_3.cpp
git checkout src/task/deep_learning_task.cpp
```

> ⚠️ **注意**：上述 PowerShell 命令假设 `deep_learning_task.cpp` 除了 STX2 块外没有其他未提交的修改。如果有其他修改，请使用 `git diff` 确认后再执行。

---

**技术觉醒团队 · 编译测试组**  
*文档版本：v1.0 | 基于 STEP.md / STEP2.md 演进 | 2026-05-22*
