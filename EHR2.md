# EHR2 — 以 `compile_h2d_only()+run_h2d_only()` 为基线重写 `compile()+run()` 的调研与方案

> 目标：使 `compile()+run()` 的传输层行为与已验证正确的 `compile_h2d_only()+run_h2d_only()` **完全一致**，在此基础上叠加训练计算图，最终跑通 `test_dl_full.cpp`。

---

## 1. 执行摘要

`compile_h2d_only()+run_h2d_only()` 已在 **MNIST/CIFAR10/IMAGENET × CPU/GPU/AMP × 多 RANK** 全矩阵通过数学正确性验证。其成功不依赖于特殊的编译路径——它走的正是 `TaskBase::compile()` 的统一管线，只是通过 `h2d_only_` 标志让 `build_graph_atlas()` 仅注册 `TRANSFER_A/B` 两张图。

`compile()+run()` 的问题**不在 compile 阶段**（图捕获、MemoryPlan、Arena 分配均正确），**而在 run 阶段的 epoch 执行逻辑**：`run_train_epoch_gpu()` 的 H2D 传输子系统与 `run_h2d_only_train_epoch()` 存在 **5 处关键差异**，其中 **label_smce 的 D2D 拷贝缺失** 和 **buffer 状态管理不一致** 是最可能导致 `test_dl_full.cpp` 失败的根因。

**重构核心策略**：将 `run_h2d_only_train_epoch()` 的 GPU/CPU H2D 逻辑提取为可复用的 `execute_h2d_epoch()`，让 `run_train_epoch_gpu()` 和 `run_val_epoch_gpu()` 在 H2D 阶段直接调用它，确保传输行为 100% 一致；计算图部分保持现有的 `cudaGraphLaunch` 启动逻辑不变。

---

## 2. `compile_h2d_only()+run_h2d_only()` 成功要素拆解

### 2.1 compile 阶段

```cpp
void DeepLearningTask::compile_h2d_only() {
    struct Guard { bool* p; Guard(bool* ptr) : p(ptr) { *p = true; } ~Guard() { *p = false; } } guard(&h2d_only_);
    compile();  // ← 就是普通的 compile()
}
```

- `h2d_only_ = true` 仅影响 `build_graph_atlas()`：只向 `GraphAtlas` 注册 `GraphId::TRANSFER_A` 和 `TRANSFER_B`
- `pre_capture(atlas, ctx_ptrs)` 捕获这两张图，结果存入 `captured_result_`
- `build_exec_table()` 将 `captured_result_` 映射到 `gpu_exec_.graphs[rank][GraphSlot::XFER_A/B]`
- **结论**：compile 管线本身是通用且正确的，`compile()` 不需要任何修改

### 2.2 run 阶段 — `run_h2d_only_train_epoch()` GPU 路径

```cpp
for (int batch = 0; batch < batches; ++batch) {
    int buf_id = batch % 2;
    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

    ts->wait_buffer_readable(buf_id);           // ① 阻塞等待 Preprocessor 填好 buffer

    cudaGraphLaunch(g_xfer, s_trans);           // ② 启动 H2D captured graph
    cudaStreamSynchronize(s_trans);             // ③ 确保 H2D 完成

    ts->set_buffer_readable(buf_id, false);     // ④ 标记 buffer 已消费
    ts->set_buffer_writeable(buf_id, true);     // ⑤ 允许 Preprocessor 复写

    cudaMemcpyAsync(label_smce_ptr,             // ⑥ label_smce D2D 拷贝
        (buf_id == 0) ? label_ptr_a : label_ptr_b,
        label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
    cudaStreamSynchronize(s_trans);             // ⑦ 确保 D2D 完成
}
```

**成功关键**：
1. **每个 rank 独立管理 buffer 状态**（`wait`/`set` 都在 rank 线程内）
2. **H2D graph 完成后显式 sync**，不依赖隐式同步
3. **`label_smce` 显式 D2D 拷贝**——H2D graph 只把 label 写到 `I_A/B_LABEL`，loss 计算需要的是 `label_smce`，必须二次拷贝
4. **CPU 路径同样严谨**：`std::memcpy` 从 `StagingBufferPool` 直接拷贝到 `context(0).ptr_at(d.id)`

---

## 3. `compile()+run()` 问题根因分析

### 3.1 `run_train_epoch_gpu()` 的 H2D 逻辑（当前实现）

```cpp
// Batch 0 预传输
while (!ts->buffer_is_readable(0)) sleep(100us);
if (g_xfer_a) cudaGraphLaunch(g_xfer_a, s_trans);
sync_tr();
if (rank == 0) {                              // ← 仅 rank 0 管理 buffer
    ts->set_buffer_readable(0, false);
    ts->set_buffer_writeable(0, true);
}

// batch 0 ~ batches-2 循环
for (int batch = 0; batch < batches - 1; ++batch) {
    // Phase 1: ZERO_GRAD ‖ FIRST_FWD
    // Phase 2: DEEP_FWD_BWD ‖ XFER(next)
    // Phase 3: FIRST_BWD ‖ DEEP_ALLREDUCE
    // Phase 4: LR → FIRST_ALLREDUCE → WEIGHT_UPDATE
}

// Last batch
// ...
```

### 3.2 与 `run_h2d_only_train_epoch()` 的 5 处关键差异

| # | 差异点 | `run_h2d_only_train_epoch()` ✅ | `run_train_epoch_gpu()` ❌ | 影响 |
|---|--------|-------------------------------|---------------------------|------|
| 1 | **label_smce D2D 拷贝** | 每次 H2D 后显式 `cudaMemcpyAsync(label_smce_ptr, ...)` | **完全缺失** | loss 计算读到的 label 是旧值或垃圾，导致 loss NaN/错误 |
| 2 | **Buffer 可读等待** | `ts->wait_buffer_readable(buf_id)`（条件变量阻塞） | `while (!ts->buffer_is_readable(buf_id)) sleep(100us)`（自旋睡眠） | 时序脆弱，高负载下可能丢数据 |
| 3 | **Buffer 状态管理** | **每个 rank** 都调用 `set_buffer_readable/writeable` | **仅 rank 0** 调用 | 多卡下非 rank-0 的 Preprocessor worker 可能永远等不到 writeable 信号 |
| 4 | **H2D 后 sync 粒度** | `cudaStreamSynchronize(s_trans)` 在 graph launch 后立即调用 | `sync_tr()` 与 `sync_comp()` 成对出现，但中间夹着计算图 | H2D 是否完成取决于后续计算图的 sync 时机，存在 race |
| 5 | **诊断代码污染** | 零诊断代码，纯生产逻辑 | 大量 `cudaMemcpy` + `LOG_INFO` 散布在热路径（~200 行诊断） | 隐式同步 + 时序扰动 + 代码可读性极差 |

### 3.3 为什么 label_smce 缺失是致命问题

- H2D captured graph（`RANGE_H2D_COPY`）的 `dst_off` 指向 `I_A_LABEL` / `I_B_LABEL`（region 49/51）
- Loss kernel（SoftmaxCrossEntropy）读取的 label 地址是 `label_smce`（region 64，baseline 中的 `loss_label`）
- `run_h2d_only_train_epoch()` 通过 `cudaMemcpyAsync(label_smce_ptr, label_ptr_a/b, ...)` 完成从 `I_A/B_LABEL` → `label_smce` 的 D2D 拷贝
- `run_train_epoch_gpu()` **没有这个拷贝**，SoftmaxCE 读到的 `label_smce` 可能是：
  - 上一 batch 的残留值（导致 loss 正确但梯度方向错误）
  - 初始零值（导致 loss 为 0，梯度为 0，模型不更新）
  - 随机值（导致 loss NaN）

这与 `test_dl_full.cpp` 报告的现象（loss 异常、模型不收敛、best_top1 < 85%）完全吻合。

---

## 4. 架构对比：`compile()+run()` vs `compile_h2d_only()+run_h2d_only()`

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          COMPILE PHASE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  compile_h2d_only()                    compile()                         │
│       │                                    │                            │
│       ▼                                    ▼                            │
│  h2d_only_ = true                   h2d_only_ = false (default)         │
│       │                                    │                            │
│       └──► compile_impl() ◄────────────────┘                            │
│                │                                                        │
│                ├──► build_graph_atlas()                                 │
│                │      h2d_only_ ? {TRANSFER_A, TRANSFER_B}              │
│                │      : {TRANSFER_A/B, FIRST_FWD_A/B, DEEP_FWD_BWD,     │
│                │        FIRST_BWD, ZERO_GRAD, DEEP_COMM, FIRST_COMM,    │
│                │        OPTIMIZER, EMA_UPDATE, CAST_AND_CHECK, ...}     │
│                │                                                        │
│                ├──► pre_capture(atlas, ctx_ptrs)                        │
│                │      捕获 atlas 中所有 registered 图                     │
│                │                                                        │
│                └──► dl->build_exec_table()                              │
│                       将 captured_result_ → gpu_exec_.graphs[rank][slot] │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            RUN PHASE                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  run_h2d_only()                        run() → run_gpu()                 │
│       │                                    │                            │
│       ▼                                    ▼                            │
│  run_h2d_only_train_epoch()          run_train_epoch_gpu()               │
│       │                                    │                            │
│       ├──► prep_thread (train)             ├──► prep_thread (train)     │
│       │                                    │                            │
│       ├──► per-rank threads                ├──► per-rank threads        │
│       │      wait_buffer_readable()        │      while(buffer_is_readable) sleep
│       │      cudaGraphLaunch(XFER)         │      cudaGraphLaunch(XFER)  │
│       │      cudaStreamSync(s_trans)       │      sync_tr()              │
│       │      set_buffer_readable(false)    │      if(rank==0) set_buffer_readable
│       │      set_buffer_writeable(true)    │      if(rank==0) set_buffer_writeable
│       │      cudaMemcpyAsync(label_smce)   │      [MISSING]              │
│       │      cudaStreamSync(s_trans)       │      [MISSING]              │
│       │                                    │                            │
│       │      [NO COMPUTE]                  │      cudaGraphLaunch(ZERO_GRAD)
│       │                                    │      cudaGraphLaunch(FIRST_FWD)
│       │                                    │      cudaGraphLaunch(DEEP_FWD_BWD)
│       │                                    │      cudaGraphLaunch(FIRST_BWD)
│       │                                    │      cudaGraphLaunch(DEEP_ALLREDUCE)
│       │                                    │      cudaGraphLaunch(WEIGHT_UPDATE)
│       │                                    │                            │
│       └──► prep_thread.join()              └──► prep_thread.join()      │
└─────────────────────────────────────────────────────────────────────────┘
```

**核心结论**：两张图的 compile 阶段完全共享同一套管线，差异**仅在于 run 阶段 epoch 函数中的 H2D 子系统实现**。只要将 `run_h2d_only_train_epoch()` 的 H2D 逻辑原封不动地嵌入 `run_train_epoch_gpu()`，就能保证传输行为一致。

---

## 5. 重构方案

### 5.1 设计原则

1. **零改动 compile 阶段**：`compile()`、`build_graph_atlas()`、`pre_capture()`、`build_exec_table()` 均保持现状
2. **H2D 逻辑 100% 复用**：提取 `run_h2d_only_train_epoch()` 的 GPU/CPU 路径为独立函数，让 `run_train_epoch_gpu()` 和 `run_val_epoch_gpu()` 直接调用
3. **计算图逻辑不变**：`ZERO_GRAD`、`FIRST_FWD`、`DEEP_FWD_BWD`、`FIRST_BWD`、`ALLREDUCE`、`WEIGHT_UPDATE` 的启动顺序和 stream 分配保持现有实现
4. **诊断代码隔离**：将 `run_train_epoch_gpu()` 中 ~200 行诊断代码迁移到 `#ifdef TR_DIAG_EPOCH` 条件编译块或独立函数

### 5.2 Phase 1：提取公共 H2D 执行层（P0）

#### 5.2.1 新增 `execute_h2d_epoch_gpu()`

将 `run_h2d_only_train_epoch()` 中 `if (registry.using_gpu()) { ... }` 的完整内容提取为一个私有方法：

```cpp
// DeepLearningTask 私有方法
float DeepLearningTask::execute_h2d_epoch_gpu(
    int batches,
    void* label_smce_ptr,
    void* label_ptr_a,
    void* label_ptr_b,
    size_t label_nbytes,
    TransferStation* ts)
{
    const int K = num_gpus_;
    std::vector<std::exception_ptr> rank_exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess)
                    TR_DEVICE_ERROR("cudaSetDevice failed: " << cudaGetErrorString(err));

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                cudaStream_t s_trans = static_cast<cudaStream_t>(
                    context(rank).stream(StreamKind::TRANS));

                for (int batch = 0; batch < batches; ++batch) {
                    int buf_id = batch % 2;
                    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                    ts->wait_buffer_readable(buf_id);

                    cudaGraphLaunch(g_xfer, s_trans);
                    cudaStreamSynchronize(s_trans);

                    ts->set_buffer_readable(buf_id, false);
                    ts->set_buffer_writeable(buf_id, true);

                    cudaMemcpyAsync(label_smce_ptr,
                        (buf_id == 0) ? label_ptr_a : label_ptr_b,
                        label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
                    cudaStreamSynchronize(s_trans);
                }
            } catch (...) {
                rank_exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank)
        if (rank_exc[rank]) std::rethrow_exception(rank_exc[rank]);

    return 0.0f;  // H2D-only 无 loss
}
```

**修改范围**：`src/task/deep_learning_task.cpp` 新增方法，原 `run_h2d_only_train_epoch()` 的 GPU 路径替换为对 `execute_h2d_epoch_gpu()` 的调用。

#### 5.2.2 新增 `execute_h2d_epoch_cpu()`

同理提取 CPU 路径：

```cpp
void DeepLearningTask::execute_h2d_epoch_cpu(
    int batches,
    void* label_smce_ptr,
    void* label_ptr_a,
    void* label_ptr_b,
    size_t label_nbytes,
    TransferStation* ts)
{
    // 原 run_h2d_only_train_epoch() CPU 路径的完整逻辑
    // 包括：copy_a/copy_b 构建、batch 循环、std::memcpy、label_smce memcpy
}
```

#### 5.2.3 重写 `run_train_epoch_gpu()` — H2D 阶段对齐

**当前 `run_train_epoch_gpu()` 的 batch 循环结构**：

```
Batch 0 预传输
for batch = 0 .. batches-2:
    Phase 1: ZERO_GRAD ‖ FIRST_FWD
    Phase 2: DEEP_FWD_BWD ‖ XFER(next)      ← 计算与传输并行
    Phase 3: FIRST_BWD ‖ DEEP_ALLREDUCE
    Phase 4: LR → FIRST_ALLREDUCE → WEIGHT_UPDATE
Last batch:
    Phase 1: ZERO_GRAD ‖ FIRST_FWD
    Phase 2: DEEP_FWD_BWD
    Phase 3: FIRST_BWD ‖ DEEP_ALLREDUCE
    Phase 4: LR → FIRST_ALLREDUCE → WEIGHT_UPDATE
```

**重构后的结构（Phase 1：先对齐，牺牲并行）**：

```
// 1. H2D 传输阶段（与 run_h2d_only_train_epoch 完全一致）
execute_h2d_epoch_gpu(batches, label_smce_ptr, label_ptr_a, label_ptr_b, label_nbytes, ts);

// 2. 计算阶段（串行执行，与 H2D 不重叠）
for batch = 0 .. batches-1:
    bool from_a = (batch % 2 == 0);
    auto g_fwd   = from_a ? g_fwd_a   : g_fwd_b;
    auto g_deep  = from_a ? g_deep_a  : g_deep_b;
    auto g_first_cur = from_a ? g_first : g_first_b;

    Phase 1: ZERO_GRAD ‖ FIRST_FWD
    sync_comp(); sync_up();

    Phase 2: DEEP_FWD_BWD
    sync_comp();

    Phase 3: FIRST_BWD ‖ DEEP_ALLREDUCE
    sync_comp(); sync_up();

    if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }

    Phase 4: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE
    sync_up();
```

**为什么先牺牲并行？**
- `test_dl_full.cpp` 的首要目标是**数学正确性**（best_top1 > 85%）
- H2D-计算并行是性能优化，应在确认正确性后引入
- 先对齐 H2D 逻辑可以排除传输层的一切不确定性，如果此时 `test_dl_full` 通过，则证明问题确实只在传输层

**Phase 2 再引入并行（见 5.4）**：在确认正确后，将 `execute_h2d_epoch_gpu()` 从"全量预传输"改为"pipelined 双缓冲"（batch N 计算的同时传输 batch N+1），恢复性能。

#### 5.2.4 重写 `run_val_epoch_gpu()` — H2D 阶段对齐

当前 `run_val_epoch_gpu()` 的 H2D 逻辑：

```cpp
while (!ts->buffer_is_readable(buf)) sleep(100us);
auto g_xfer = (buf == 0) ? g_xfer_a : g_xfer_b;
if (g_xfer) cudaGraphLaunch(g_xfer, s_trans);
sync_tr();
// 没有 set_buffer_readable/writeable！
```

问题：
1. 使用 `sleep` 而非 `wait_buffer_readable`
2. 没有 `set_buffer_readable/writeable`
3. 没有 `label_smce` 拷贝（验证不需要 loss，但需要 label 用于 top1/top5 计算）

**修改**：将验证的 H2D 逻辑也替换为调用 `execute_h2d_epoch_gpu()`（验证时 `batches = val_batches`，无 label_smce 需求可传 `nullptr`）。

### 5.3 诊断代码隔离（P1）

`run_train_epoch_gpu()` 当前包含约 **200 行** 散布在热路径的诊断代码（`[DIAG-XFER0]`、`[DIAG-PHASE1]`、`[DIAG-B0]`、`[DIAG-S1~S4]` 等）。这些代码：
- 在 H2D 和计算之间插入大量 `cudaMemcpy`（隐式同步）
- 污染 `s_trans`、`s_c1`、`s_up` stream 的时序
- 使代码难以阅读和维护

**修改方案**：将所有诊断块包裹在条件编译中：

```cpp
#ifdef TR_DIAG_EPOCH
    // 原诊断代码块
#endif
```

默认不定义 `TR_DIAG_EPOCH`，诊断代码被编译器完全消除，零运行时开销。

### 5.4 Phase 2：恢复 H2D-计算并行（P2，后续迭代）

在 `test_dl_full` 通过且传输层确认无误后，将串行执行恢复为双缓冲并行：

```
Batch 0:
    execute_h2d_batch_gpu(batch=0, buf=0)   // 同步完成
    launch ZERO_GRAD + FIRST_FWD
    launch DEEP_FWD_BWD
    sync

Batch 1..batches-1:
    // 背景：当前 batch 的数据已在上一轮 H2D 完成
    // Step 1: 启动下一轮 H2D（与当前计算并行）
    ts->wait_buffer_readable(next_buf);
    cudaGraphLaunch(g_xfer_n, s_trans);       // 非阻塞，s_trans 独立

    // Step 2: 当前 batch 计算
    launch ZERO_GRAD + FIRST_FWD (s_c1/s_up)
    launch DEEP_FWD_BWD (s_c1)
    sync_comp();

    // Step 3: H2D 完成后的收尾（此时 s_trans 和 s_c1 都空闲）
    sync_tr();
    ts->set_buffer_readable(next_buf, false);
    ts->set_buffer_writeable(next_buf, true);
    cudaMemcpyAsync(label_smce_ptr, ..., s_trans);
    sync_tr();

    // Step 4: Post-batch（ALLREDUCE + WEIGHT_UPDATE）
    launch FIRST_BWD + DEEP_ALLREDUCE
    launch WEIGHT_UPDATE
    sync_up();
```

此结构与 `run_h2d_only_train_epoch()` 的 H2D 逻辑**完全一致**，只是将 H2D 调用从"全量 epoch 预传输"改为"按 batch pipeline"。

---

## 6. 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `src/task/deep_learning_task.cpp` | 新增方法 + 重写 | 新增 `execute_h2d_epoch_gpu()`、`execute_h2d_epoch_cpu()`；重写 `run_train_epoch_gpu()` 的 H2D 部分；重写 `run_val_epoch_gpu()` 的 H2D 部分；隔离诊断代码 |
| `include/renaissance/task/deep_learning_task.h` | 新增声明 | 声明 `execute_h2d_epoch_gpu()`、`execute_h2d_epoch_cpu` |
| `src/task/deep_learning_task.cpp` | 修改 | `run_h2d_only_train_epoch()` 和 `run_h2d_only_val_epoch()` 改为调用新的 `execute_h2d_epoch_*()` |

---

## 7. 实施步骤

```
Step 1: 新增 execute_h2d_epoch_gpu() / execute_h2d_epoch_cpu()
        └── 从 run_h2d_only_train_epoch() 中提取，零逻辑改动

Step 2: 修改 run_h2d_only_train_epoch() / run_h2d_only_val_epoch()
        └── 替换为对 execute_h2d_epoch_*() 的调用
        └── 验证：test_two_batch_correction 所有 6 种组合仍 PASS

Step 3: 重写 run_train_epoch_gpu() — 串行版本（Phase 1）
        └── 先 execute_h2d_epoch_gpu() 完成全部 H2D
        └── 再逐个 batch 启动计算图
        └── 隔离诊断代码到 #ifdef TR_DIAG_EPOCH

Step 4: 重写 run_val_epoch_gpu()
        └── 同样调用 execute_h2d_epoch_gpu() 做 H2D

Step 5: 编译并运行 test_dl_full.cpp
        └── 预期：MNIST GPU 3 epochs，best_top1 > 85%

Step 6: （后续）恢复 H2D-计算并行（Phase 2）
        └── 将 execute_h2d_epoch_gpu() 从"全量预传输"改为"per-batch pipeline"
```

---

## 8. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| `label_smce` D2D 拷贝引入后，loss 仍异常 | 中 | 需回到传输层排查 | Step 1~2 确保 H2D-only 路径不受影响，可快速二分 |
| 串行 H2D+计算导致性能下降 >30% | 低 | test 运行时间增加 | test_dl_full 只有 3 epochs MNIST，性能非瓶颈；Phase 2 恢复并行 |
| 多卡下 `wait_buffer_readable` 与 `set_buffer_*` 的线程安全 | 低 | 死锁或 race | `run_h2d_only_train_epoch()` 已验证 8 卡正确，逻辑复用 |
| 计算图 stream 依赖被破坏 | 低 | 计算错误 | 计算图启动顺序和 stream 不变，仅 H2D 时序调整 |

---

## 9. 一句话总结

**`compile()` 不需要改，`run_train_epoch_gpu()` 的 H2D 子系统需要整体替换为 `run_h2d_only_train_epoch()` 的已验证实现。** 计算图部分（ZERO_GRAD → FIRST_FWD → DEEP_FWD_BWD → FIRST_BWD → ALLREDUCE → WEIGHT_UPDATE）保持现有 CUDA Graph 启动逻辑不变。先串行对齐确保正确性，再逐步恢复 H2D-计算并行以优化性能。
