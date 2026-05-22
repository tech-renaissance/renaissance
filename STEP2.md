# STEP2.md — DeepLearningTask 编译与运行测试详细计划

> **承接**: STEP.md（四步走思路）  
> **基线**: TR4 v4.20.1, `mnist_mlp_3.cpp`, `deep_learning_task.cpp` (含 CUDA Graph 捕获)  
> **验证文档**: Z_FINAL_K.md §7.3 流水线时序  
> **日期**: 2026-05-22

---

## 0. 前置条件与环境

### 0.1 确认编译环境

| 检查项 | 命令/方法 | 预期结果 |
|--------|----------|---------|
| CMake 配置 | `cmake -B build/windows-msvc-release` | 成功生成 build 目录 |
| SimpleTask 可编译 | `cmake --build build --target test_mlp_final` | 编译成功 |
| SimpleTask 可运行 | `build/Release/test_mlp_final.exe --gpu` | PASS |
| 编译器版本 | MSVC 2022 / GCC 11+ | 支持 C++17 |

### 0.2 确认 GPU 环境

| 检查项 | 命令/方法 | 预期结果 |
|--------|----------|---------|
| CUDA 可用 | `nvidia-smi` | GPU 在线 |
| CUDA 运行时 | `nvcc --version` | CUDA 11.8+ / 12.x |
| 单卡测试 | `test_mlp_final --gpu` | 通过 |

### 0.3 测试入口文件

使用 `tests/ref/mnist_mlp_3.cpp` 作为测试入口，基于以下配置：

```cpp
GLOBAL_SETTING
    .use_gpu("0")
    .manual_seed(42)
    .local_batch_size(128)   // 第一步可调为 5000/10000
    .train_resolution(28)
    .val_resolution(28)
    .amp(false);
```

---

## 第一步：Compile Dry Run（只编译，只打印调度信息）

### 1.1 目标

验证 `compile()` 完整链路无错误，同时验证 `run()` 的调度逻辑（图名、stream、batch 顺序）与 Z_FINAL_K.md 一致。不真正执行 CUDA Graph，不启动 Preprocessor。

### 1.2 代码修改点

#### A. 修改 `mnist_mlp_3.cpp`（缩减规模）

```cpp
// 修改前
task.total_epochs(20);
GLOBAL_SETTING.local_batch_size(128);

// 修改后（仅第一步）
task.total_epochs(1);
GLOBAL_SETTING.local_batch_size(5000);  // MNIST 共 60000 样本 → 12 个 batch
```

> 注意：`local_batch_size(5000)` 后 `steps_per_epoch()` 会变为 12，大幅降低第一步打印量。

#### B. 修改 `run_gpu()` — 注释掉 Preprocessor 线程

文件：`src/task/deep_learning_task.cpp`

```cpp
// 修改前（约 line 709-717）
std::thread prep_thread([&]() {
    try { prep.train(); }
    catch (...) { prep_exc = std::current_exception(); }
});
run_train_epoch_gpu();
prep_thread.join();

// 修改后（仅第一步）
// std::thread prep_thread([&]() {
//     try { prep.train(); }
//     catch (...) { prep_exc = std::current_exception(); }
// });
run_train_epoch_gpu();
// prep_thread.join();
// if (prep_exc) std::rethrow_exception(prep_exc);
```

#### C. 修改 `run_train_epoch_gpu()` — 注释掉所有 CUDA Graph Launch

文件：`src/task/deep_learning_task.cpp`（约 line 845-958）

将 `cudaGraphLaunch(...)` 和 `cudaMemcpyAsync(...)` 替换为打印。推荐用 `#ifdef` 包围而非直接注释，方便恢复：

```cpp
#ifdef STEP1_DRY_RUN
    #define STEP1_LOG_LAUNCH(g, stream, name) \
        LOG_INFO << "[STEP1] Launch graph: " << name \
                 << " on stream=" << stream
#else
    #define STEP1_LOG_LAUNCH(g, stream, name) \
        cudaGraphLaunch(g, stream)
#endif
```

然后替换所有 launch 点（共 15 处）：

| 行号附近 | 原代码 | 替换为 |
|----------|--------|--------|
| ~845 | `cudaGraphLaunch(g_xfer_a, s_trans);` | `STEP1_LOG_LAUNCH(g_xfer_a, s_trans, "XFER_A");` |
| ~854 | `cudaGraphLaunch(g_zg, s_up);` | `STEP1_LOG_LAUNCH(g_zg, s_up, "ZERO_GRAD");` |
| ~855 | `cudaGraphLaunch(g_fwd_a, s_c1);` | `STEP1_LOG_LAUNCH(g_fwd_a, s_c1, "FIRST_FWD_A");` |
| ~858 | `cudaGraphLaunch(g_deep_a, s_c1);` | `STEP1_LOG_LAUNCH(g_deep_a, s_c1, "DEEP_A");` |
| ~862 | `cudaGraphLaunch(g_first, s_c1);` | `STEP1_LOG_LAUNCH(g_first, s_c1, "FIRST_BWD");` |
| ~864 | `cudaGraphLaunch(g_dar, s_up);` | `STEP1_LOG_LAUNCH(g_dar, s_up, "DEEP_ALLREDUCE");` |
| ~867 | `cudaGraphLaunch(g_gc, s_up);` | `STEP1_LOG_LAUNCH(g_gc, s_up, "GRAD_CONVERT");` |
| ~872 | `cudaMemcpyAsync(...);` | `LOG_INFO << "[STEP1] LR H2D memcpy";` |
| ~875 | `cudaGraphLaunch(g_far, s_up);` | `STEP1_LOG_LAUNCH(g_far, s_up, "FIRST_ALLREDUCE");` |
| ~876 | `cudaGraphLaunch(g_wu, s_up);` | `STEP1_LOG_LAUNCH(g_wu, s_up, "WEIGHT_UPDATE");` |

循环体（~899-925）和 last batch（~939-957）的对应位置同理替换。

**简化做法**：直接在 `run_train_epoch_gpu()` 函数开头添加一个 `bool dry_run = true;`，在每个 launch 点加 `if (!dry_run)` 保护：

```cpp
bool dry_run = true;  // 仅第一步设为 true

// 使用处
if (g_xfer_a) {
    if (!dry_run) cudaGraphLaunch(g_xfer_a, s_trans);
    else LOG_INFO << "[STEP1] Would launch XFER_A on TRANS stream";
}
```

> **保留 sync 调用**：`sync_comp()`、`sync_up()`、`sync_tr()` 保留（sync 空流是 no-op，但保留可验证逻辑完整性）。

#### D. 编译选项

在 CMake 中添加 `-DSTEP1_DRY_RUN=1`（可选，也可直接改源码）。

### 1.3 成功标准

编译成功，运行输出符合以下预期：

```
[STEP1] Launch graph: XFER_A on stream=TRANS
[STEP1] Launch graph: ZERO_GRAD on stream=UPDATE
[STEP1] Launch graph: FIRST_FWD_A on stream=COMP_1
[STEP1] Launch graph: DEEP_A on stream=COMP_1
[STEP1] Launch graph: XFER_B on stream=TRANS
[STEP1] Launch graph: FIRST_BWD on stream=COMP_1
[STEP1] Launch graph: DEEP_ALLREDUCE on stream=UPDATE
[STEP1] LR H2D memcpy
[STEP1] Launch graph: FIRST_ALLREDUCE on stream=UPDATE
[STEP1] Launch graph: WEIGHT_UPDATE on stream=UPDATE
...
```

**检查项**：
1. `compile()` 无 crash、无 assert 失败
2. `build_graph_atlas()` 输出的 GraphId 非空
3. `pre_capture()` 捕获成功（可临时加 LOG 打印 captured/reused 数量）
4. `build_exec_table()` 无 nullptr assert
5. `run()` 输出 12 个 batch 的调度信息（batch 0→11）
6. 单 batch 路径和 last batch 路径均触发（可通过打印确认）

### 1.4 失败排查

| 失败现象 | 排查方向 |
|----------|---------|
| compile() crash | 检查 `build_graph_atlas()` 中 train_cg_->nodes(gid) 是否非空 |
| build_exec_table() assert | 检查 `resolve()` 返回的 cudaGraphExec_t 是否为 nullptr |
| run() 无输出 | 检查 `run_gpu()` 是否被调用（`GlobalRegistry::using_gpu()`） |
| batch 数不对 | 检查 `Preprocessor::steps_per_epoch()` 返回值 |

---

## 第二步：A 区 Transfer 验证（单 batch 传输正确性）

### 2.1 目标

验证 `XFER_A` CUDA Graph 能正确将 TransferStation buffer 0 的数据 H2D 到 GPU 显存对应位置。对比本地副本与 GPU 数据是否一致。

### 2.2 代码修改点

#### A. 恢复 `XFER_A` 的真实 launch

将第一步的 `dry_run` 设为 `false`，但**仅保留 `XFER_A` 的 launch**，其余图继续注释掉或走 dry run。

#### B. 在 `run_train_epoch_gpu()` 中添加数据副本保存与验证代码

文件：`src/task/deep_learning_task.cpp`

在 `Batch 0 预传输` 段添加（约 line 842-850 之间）：

```cpp
// ========== Batch 0 预传输 ==========
while (!ts->buffer_is_readable(0))
    std::this_thread::sleep_for(std::chrono::microseconds(100));

// ==== STEP2: 保存本地副本 ====
size_t xfer_bytes = ts->get_buffer_actual_transfer_bytes(0);
uint8_t* buffer_ptr = ts->get_buffer_ptr(0);
std::vector<uint8_t> host_copy(xfer_bytes);
std::memcpy(host_copy.data(), buffer_ptr, xfer_bytes);
LOG_INFO << "[STEP2] Saved local copy of buffer 0: " << xfer_bytes << " bytes";

// 执行 XFER_A
cudaGraphLaunch(g_xfer_a, s_trans);
sync_tr();

// ==== STEP2: 从 GPU 取回并对比 ====
// 构造一个临时 DTensor 用于 fetch
DTensor xfer_dt;
xfer_dt.id = -999;  // 临时 ID，不会参与后续计算
xfer_dt.shape = Shape{1, 1, 1, static_cast<int>(xfer_bytes)};
xfer_dt.dtype = DType::INT8;  // 逐字节对比用 INT8
xfer_dt.region = Region::I_A_DATA;  // 占位
xfer_dt.offset_ = 0;  // 从 Arena 起始取（注意：这是简化的做法）

// ... 需要知道 XFER_A 实际写入的 GPU 地址 ...
```

> **注意**：上述简化做法有问题，因为 XFER_A 的目标地址不是 Arena 起始。需要找到正确的 GPU 地址。

**正确做法**：XFER_A 图内部使用 `RANGE_H2D_COPY_A` 节点，目标地址是 `MemoryPlan` 中 `I_A_DATA` 区域的 GPU 地址。但 `run_train_epoch_gpu()` 中不方便直接获取该地址。

更实际的做法：在 `compile_impl()` 中或 `on_prepare()` 末尾，记录 input DTensor 的 ID，然后在 `run_train_epoch_gpu()` 中通过 `ctx.ptr_at(input_id)` 获取 GPU 地址。

但 `ctx.ptr_at(dtensor_id)` 返回的是 `void*`，它查找 ArenaKeeper 中 `rank` 对应设备从 `dtensor.offset()` 开始的地址。所以我们需要知道 input DTensor 的 offset。

**最简方案**：遍历 `active_memory_plan_->dtensors()` 查找 `Region::I_A_DATA`，然后直接 `fetch_from_rank`：

```cpp
// 在 run_train_epoch_gpu() 中添加（临时调试代码）
DTensor input_dt;
bool found_input = false;
for (const auto& dt : active_memory_plan_->dtensors()) {
    if (dt.region == Region::I_A_DATA) {
        input_dt = dt;
        found_input = true;
        break;
    }
}
if (!found_input) {
    LOG_ERROR << "[STEP2] No I_A_DATA DTensor found in MemoryPlan";
    return;
}

// XFER_A 执行后，从 GPU 取回
Tensor fetched = fetch_from_rank(input_dt, rank);

// 对比：TransferStation buffer 的数据 vs GPU 取回的数据
// 注意：TransferStation buffer 包含 labels + padding + data
// 而 fetch_from_rank 返回的是 DTensor 对应的完整 slot（含 padding）
// 需要小心处理 padding 的差异
```

实际上，对比逻辑更复杂。让我重新设计...

**STEP2 推荐做法**：不直接对比 raw bytes，而是对比有效数据。

在 TransferStation 的 `request_write_slot` 中，Worker 写入的数据包含 label 和 image data。XFER_A 图将这些数据 H2D 到 GPU。GPU 上的 DTensor 布局可能与 TransferStation buffer 的布局不完全相同（因为 DTensor 可能有不同的 padding 策略）。

所以更实际的做法是：
1. 从 TransferStation buffer 提取 labels 和 image data
2. 从 GPU 取回 input DTensor
3. 对比有效元素（labels 和 pixels）

但这仍然需要知道 input DTensor 的具体 shape。

**最实用的方案**：在 `run_train_epoch_gpu()` 中，XFER_A 之后，直接读取 GPU 显存（通过 `cudaMemcpy` 从 `ctx.ptr_at(input_dtensor_id)`）到一个主机 Tensor，然后对比 TransferStation buffer 中的数据。

但 `input_dtensor_id` 未知...

**最佳方案**：在 `compile_impl()` 末尾或 `on_prepare()` 中，保存 input DTensor 的 ID 到成员变量 `input_dtensor_id_`。

```cpp
// 在 deep_learning_task.h 中添加
int input_dtensor_id_ = -1;

// 在 on_prepare() 末尾添加
for (const auto& dt : active_memory_plan_->dtensors()) {
    if (dt.region == Region::I_A_DATA) {
        input_dtensor_id_ = dt.id;
        break;
    }
}
```

然后在 `run_train_epoch_gpu()` 中：

```cpp
// XFER_A 之后
void* gpu_input_ptr = ctx.ptr_at(input_dtensor_id_);
const DTensor& input_dt = active_memory_plan_->get_dtensor(input_dtensor_id_);
size_t input_bytes = input_dt.slot_bytes();

std::vector<uint8_t> gpu_copy(input_bytes);
cudaMemcpy(gpu_copy.data(), gpu_input_ptr, input_bytes, cudaMemcpyDeviceToHost);

// 对比 TransferStation buffer 与 gpu_copy
// 注意：TransferStation buffer 的大小和 GPU buffer 的大小可能不同
// 因为 TransferStation 使用 align_256，而 MemoryPlan 也使用 align_256
// 如果两者对齐策略一致，大小应该相同
```

嗯，这变得越来越复杂。也许 STEP.md 的意图更直接："把A区的数据全部复制到一块跟A区一样大的内存里，然后执行A区异步传输图，然后用fetch方法从显存取回A区数据，再跟本地留存的数据副本进行对比，看是否逐字节相同。"

"fetch方法"可能指的是 `TaskBase::fetch_from_rank()`。而 "A区" 在 TransferStation 中指的是 buffer 0。在 GPU 上，A 区数据对应的是 input DTensor。

如果 XFER_A 图在捕获时已经绑定了 TransferStation buffer 0 的地址和 GPU input DTensor 的地址，那么执行 XFER_A 后，数据就已经在 GPU 上了。

对比方案：
1. TransferStation buffer 0 的数据 → `host_copy_A`
2. 执行 XFER_A
3. GPU input DTensor → `gpu_copy_A`（通过 `fetch_from_rank`）
4. 逐字节对比 `host_copy_A` 和 `gpu_copy_A`

但问题是 `host_copy_A` 是 TransferStation buffer 的原始布局，而 `gpu_copy_A` 是 DTensor 的布局（可能有 padding 差异）。如果 XFER_A 图只传输有效数据（不含 padding），那么 padding 区域的对比会失败。

解决方案：只对比 XFER_A 实际传输的字节数（`ts->get_buffer_actual_transfer_bytes(0)`）对应的 GPU 区域。

或者更简单：在 `run_train_epoch_gpu()` 中不追求逐字节对比，而是验证 key 字段（如第一个 label、第一个 pixel）是否正确传输。如果 label 和 pixel 正确，说明传输逻辑正确。

但 STEP.md 明确说了"逐字节相同"...

好，我换一个思路。也许我们可以在 `DeepLearningTask` 中添加一个专门的调试接口：

```cpp
bool DeepLearningTask::debug_verify_xfer_a(int rank);
```

这个接口：
1. 获取 TransferStation buffer 0 的指针和大小
2. 执行 XFER_A
3. 获取 XFER_A 在 GPU 上的目标地址（通过解析 CapturedGraph 或 MemoryPlan）
4. 从 GPU 复制回主机
5. 对比

但解析 CapturedGraph 很复杂...

**最终方案**：在 `run_train_epoch_gpu()` 中直接通过 `cudaMemcpy` 读取 GPU 上 XFER_A 目标区域。为了找到目标区域，我们使用 MemoryPlan 中 `I_A_DATA` 区域的 DTensor。

具体步骤：
1. 在 `compile_impl()` 末尾或 `on_prepare()` 中，查找并保存 `input_data_dtensor_id_`（region == I_A_DATA）
2. 在 `run_train_epoch_gpu()` 中：
   a. 获取 TransferStation buffer 0 的 `buffer_ptr` 和 `xfer_bytes`
   b. `memcpy(host_copy, buffer_ptr, xfer_bytes)`
   c. 执行 `cudaGraphLaunch(g_xfer_a, s_trans); sync_tr();`
   d. `void* gpu_ptr = ctx.ptr_at(input_data_dtensor_id_);`
   e. `cudaMemcpy(gpu_host_copy, gpu_ptr, xfer_bytes, cudaMemcpyDeviceToHost);`
   f. `memcmp(host_copy, gpu_host_copy, xfer_bytes) == 0`

注意：这里假设 XFER_A 传输的字节数等于 `input_dt.slot_bytes()`，且目标地址从 `input_dt.offset()` 开始。这可能不完全正确，因为 XFER_A 图内部可能使用 `RANGE_H2D_COPY_A`，它可能只传输有效数据而不是整个 slot。但 `slot_bytes()` 是 DTensor 的完整 slot 大小（含 padding），而 TransferStation 的 `get_buffer_actual_transfer_bytes()` 也是含 padding 的（因为 TransferStation 的 buffer 布局也考虑了对齐）。如果两者对齐方式一致，padding 区域应该都是对齐字节，可以逐字节对比。

实际上，为了简化，我们可以对比 `std::min(xfer_bytes, slot_bytes())` 字节，这样更安全。

好，我先把这些思考写入 STEP2.md，但保持简洁和可执行。

实际上，也许我过度复杂化了。让我想想最简单的方法...

STEP.md 说"这个可能需要额外为DeepLearningTask类添加调试接口，但行为基本上就是：Preprocessor准备了一个batch的数据在A区，你把A区的数据全部复制到一块跟A区一样大的内存里，然后执行A区异步传输图，然后用fetch方法从显存取回A区数据，再跟本地留存的数据副本进行对比，看是否逐字节相同。"

也许"fetch方法"不是 `fetch_from_rank`，而是直接用 `cudaMemcpy` 从 GPU 复制回主机。

"A区"在 TransferStation 中是 buffer 0。在 GPU 上，"A区"是某个 DTensor（可能是 I_A_DATA）。

如果我们假设 XFER_A 图将 TransferStation buffer 0 的 `get_buffer_actual_transfer_bytes(0)` 字节传输到了 GPU 的某个地址，那么对比方法：
1. 保存 buffer 0 的 `get_buffer_actual_transfer_bytes(0)` 字节到 host_copy
2. 执行 XFER_A
3. 从 GPU 目标地址复制相同字节数到 gpu_copy
4. 对比

关键：如何知道 GPU 目标地址？

XFER_A 是 CUDA Graph，捕获时已经记录了 H2D 拷贝的源地址和目标地址。源地址是 TransferStation buffer 0 的地址（在 CPU 内存中），目标地址是 GPU 显存中的某个地址。

但 XFER_A 图在每次 launch 时是否可以更新参数？CUDA Graph 允许参数更新（通过 `cudaGraphExecKernelNodeSetParams`），但我们的 CapturedGraph 可能不支持参数更新。

如果 XFER_A 在捕获时绑定了固定的源地址和目标地址，那么每次 launch 都使用相同的地址。这意味着：
- 源地址 = TransferStation buffer 0 的固定地址（由 StagingBufferPool 分配）
- 目标地址 = GPU ArenaKeeper 中 input DTensor 的固定地址

如果这样，我们可以在捕获后获取目标地址...

但这很复杂。也许更简单：XFER_A 图内部使用的不是 `cudaMemcpy`，而是 `RANGE_H2D_COPY_A` 自定义 RangeOp，它可能使用 `cudaMemcpyAsync` 并传递参数。参数在图捕获时就已经固定了。

无论如何，为了验证传输正确性，最实用的方案是：
1. 在 `run_train_epoch_gpu()` 中直接读取 GPU 显存（从 ArenaKeeper 起始地址 + input DTensor offset）
2. 与 TransferStation buffer 0 对比

让我把具体实现细节写入 STEP2.md。

实际上，让我换一个更实际的角度。也许不需要这么精确地找到 GPU 地址。我们可以：
1. 在 `run_train_epoch_gpu()` 中，XFER_A 之前，从 TransferStation buffer 0 读取 label 和第一个 pixel
2. XFER_A 之后，从 GPU 显存读取对应位置的 label 和第一个 pixel（通过 `ctx.ptr_at(input_label_id)` 和 `ctx.ptr_at(input_data_id)`）
3. 对比这些 key 值

这比逐字节对比更简单，且足以验证传输正确性。

但 STEP.md 明确说"逐字节相同"... 好吧，让我提供一个详细的逐字节对比方案。

好的，我已经花了太多时间思考细节。让我直接写出 STEP2.md，在第二步中提供清晰但不过度复杂的方案。

让我再快速检查一下是否有遗漏的关键信息...

关于 `fetch_from_rank`：它在 `task_base.cpp` 中，通过 `active_memory_plan_->get_dtensor(dt.id)` 获取 live DTensor，然后 `cudaMemcpy` 回主机。所以 `fetch_from_rank(input_dt, 0)` 是正确的用法。

好的，现在让我写 STEP2.md。内容需要：
1. 承接 STEP.md 的四步思路
2. 详细说明每一步的代码修改点
3. 提供验证方法和成功标准
4. 提供失败排查指南

我会使用清晰、可执行的格式。

在写之前，让我再确认一下一些关键信息：
- `run_gpu()` 中的 N+1 线程结构
- `run_train_epoch_gpu()` 中的 K 个 rank 线程
- 单 batch 路径 vs 多 batch 路径
- compile_impl() 的完整流程

这些我已经很熟悉了。让我开始写 STEP2.md。

计划结构：
1. 概述
2. 前置条件
3. 第一步：Compile Dry Run
4. 第二步：A 区 Transfer 验证
5. 第三步：AB 区传输带宽测试
6. 第四步：端到端真实训练
7. 回滚指南

每个步骤包含：
- 目标
- 修改文件清单
- 具体修改（代码片段）
- 编译命令
- 运行命令
- 预期输出
- 成功标准
- 失败排查

让我开始写。