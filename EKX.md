# 【今日话题：重写DeepLearningTask的compile()+run()】

# 【背景】

此前我们使用compile()+run()的组合，但是似乎结果出现了一些问题。
如今我们的compile_h2d_only()+run_h2d_only()已经在所有数据集（MNIST/CIFAR/IMAGENET）、所有场景（CPU/GPU/AMP）、多RANK下都通过了数学正确性验证和性能测试，是很好的基础。
我的计划是以compile_h2d_only()+run_h2d_only()为基础来彻底重写compile()+run()，使得compile()+run()至少在传输方面的行为与compile_h2d_only()+run_h2d_only()**完全一致**。也就是说，新版的compile()+run()将会相当于在compile_h2d_only()+run_h2d_only()的基础上加上了其他图的编译和运行，而传输逻辑不变。
咱们先别改代码，先对当前代码进行一次完整的调研，说说要实现这样的改动需要怎么修改。
我们的最终目的，还是要跑通DeepLearningTask版的test_dl_full.cpp的测试（而其SimpleTask版的成功样例是test_mlp_final.cpp）。
调查分析完后，把你的方案写在根目录下的EHR1.md



# 【小伙伴S】

# 

## 执行摘要

经过全面的代码审查，确认`compile_h2d_only()` + `run_h2d_only()`已经建立了可靠的传输基础。现在需要将这个成功的传输逻辑完整移植到`compile()` + `run()`中，同时保留其他图的编译和执行能力。

---

## 1. 当前成功架构分析：H2D-Only模式

### 1.1 compile_h2d_only() 的核心机制

**位置**: `src/task/deep_learning_task.cpp:2135-2146`

```cpp
void DeepLearningTask::compile_h2d_only() {
    struct Guard {
        bool* p;
        Guard(bool* ptr) : p(ptr) { *p = true; }
        ~Guard() { *p = false; }
    } guard(&h2d_only_);  // ← 关键：设置h2d_only_标志

    compile();  // ← 调用完整compile()，但h2d_only_=true改变行为
}
```

**关键洞察**:

- H2D-Only模式通过`h2d_only_`标志改变`build_graph_atlas()`的行为
- 当`h2d_only_=true`时，只注册传输图（TRANSFER_A/B），跳过其他图
- 编译路径与完整compile()完全相同，只是图集合不同

### 1.2 run_h2d_only() 的成功传输逻辑

**位置**: `src/task/deep_learning_task.cpp:2148-2248`

**成功要素**:

1. **精确的同步机制**:

   ```cpp
   ts->wait_buffer_readable(buf_id);       // 等待Preprocessor生产数据
   cudaGraphLaunch(g_xfer, s_trans);        // 启动传输图
   cudaStreamSynchronize(s_trans);          // 确保传输完成
   ts->set_buffer_readable(buf_id, false);  // 通知TransferStation已消费
   ts->set_buffer_writeable(buf_id, true);  // 通知TransferStation可复用
   ```

2. **正确的设备管理**:

   ```cpp
   cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
   ```

3. **安全的指针获取**:

   ```cpp
   auto label_ptr_a = context(rank).ptr_at(bl.label_a);
   auto label_ptr_b = context(rank).ptr_at(bl.label_b);
   auto label_smce_ptr = context(rank).ptr_at(bl.label_smce);
   ```

4. **AB区乒乓机制**:

   ```cpp
   int buf_id = batch % 2;
   auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;
   ```

---

## 2. 当前问题分析：完整compile()+run()的缺陷

### 2.1 compile() 中的图构建

**位置**: `src/task/task_base.cpp:204-278`

**当前流程**:

```cpp
void TaskBase::compile() {
    debug_mode_ = false;
    compile_impl(false);
}

void TaskBase::compile_impl(bool debug_mode) {
    compile_freeze_global();
    compile_invoke_on_prepare();
    compile_verify_memory_locked();
    compile_alloc_hardware();

    if (is_simple_task()) {
        compile_capture_simple();
    } else {
        // DeepLearningTask路径
        GraphAtlas atlas = dl->build_graph_atlas();  // ← 构建所有图
        captured_result_ = pre_capture(atlas, ctx_ptrs);  // ← 捕获所有图
        dl->build_exec_table();  // ← 构建执行表
    }

    compile_mark_compiled();
}
```

**关键发现**:

- `build_graph_atlas()`已经通过`h2d_only_`标志正确处理H2D图
- `pre_capture()`和`build_exec_table()`对所有图一视同仁
- 编译阶段无需修改，已经支持H2D-Only逻辑

### 2.2 run_train_epoch() 中的传输问题

**位置**: `src/task/deep_learning_task.cpp:881-1464`

**当前实现的问题**:

1. **复杂的图依赖管理**:

   ```cpp
   auto g_xfer_a  = g_tab[S(GraphSlot::XFER_A)];
   auto g_xfer_b  = g_tab[S(GraphSlot::XFER_B)];
   auto g_deep_a  = g_tab[S(GraphSlot::FWD_BWD_DEEP_A)];
   auto g_deep_b  = g_tab[S(GraphSlot::FWD_BWD_DEEP_B)];
   auto g_first   = g_tab[S(GraphSlot::FIRST_LAYER_BWD)];
   // ... 还有10多个图 ...
   ```

2. **不清晰的同步点**:

   - 多个Stream之间的同步关系复杂
   - TransferStream与其他Stream的同步可能存在race condition

3. **批次循环逻辑复杂**:

   - Batch 0预传输、正常批次、最后批次的处理逻辑分散
   - 难以验证传输逻辑的正确性

---

## 3. 重写方案：传输逻辑完全一致性

### 3.1 核心原则

**"传输逻辑不变性原则"**: 新版`run_train_epoch()`的传输部分必须与`run_h2d_only()`逐行对齐，只在传输完成后插入其他图的执行。

### 3.2 compile() 修改

**位置**: `src/task/task_base.cpp:204-278`

**修改方案**: **无需修改**

**理由**:

- `compile()`已经通过`h2d_only_`标志正确支持H2D-Only模式
- 图构建、捕获、执行表构建逻辑对所有图一视同仁
- 编译阶段已经满足要求

### 3.3 run_train_epoch() 重写

**位置**: `src/task/deep_learning_task.cpp:881-1464`

#### 3.3.1 基本结构重构

```cpp
float DeepLearningTask::run_train_epoch_gpu() {
    // ... 初始化代码保持不变 ...
    
    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, ts, K, using_amp, &exc]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

                // ========== 获取图指针（与run_h2d_only完全一致） ==========
                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                cudaStream_t s_trans = static_cast<cudaStream_t>(
                    context(rank).stream(StreamKind::TRANS));

                const auto& bl = active_memory_plan_->baseline();
                auto label_ptr_a = context(rank).ptr_at(bl.label_a);
                auto label_ptr_b = context(rank).ptr_at(bl.label_b);
                auto label_smce_ptr = context(rank).ptr_at(bl.label_smce);
                size_t label_nbytes = static_cast<size_t>(
                    active_memory_plan_->get_dtensor(bl.label_a).slot_bytes());

                // ========== 获取其他图指针（新增） ==========
                auto g_deep_a  = g_tab[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto g_deep_b  = g_tab[S(GraphSlot::FWD_BWD_DEEP_B)];
                auto g_first   = g_tab[S(GraphSlot::FIRST_LAYER_BWD)];
                auto g_zg      = g_tab[S(GraphSlot::ZERO_GRAD)];
                auto g_wu      = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
                cudaStream_t s_c1 = static_cast<cudaStream_t>(
                    context(rank).stream(StreamKind::COMP_1));
                cudaStream_t s_up = static_cast<cudaStream_t>(
                    context(rank).stream(StreamKind::UPDATE));

                // ========== 批次循环：传输逻辑与run_h2d_only完全一致 ==========
                for (int batch = 0; batch < batches; ++batch) {
                    int buf_id = batch % 2;
                    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                    // 传输阶段（与run_h2d_only逐行对齐）
                    ts->wait_buffer_readable(buf_id);
                    cudaGraphLaunch(g_xfer, s_trans);
                    cudaStreamSynchronize(s_trans);
                    ts->set_buffer_readable(buf_id, false);
                    ts->set_buffer_writeable(buf_id, true);

                    cudaMemcpyAsync(label_smce_ptr,
                        (buf_id == 0) ? label_ptr_a : label_ptr_b,
                        label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
                    cudaStreamSynchronize(s_trans);

                    // ========== 计算阶段（新增，传输完成后执行） ==========
                    auto g_deep = (buf_id == 0) ? g_deep_a : g_deep_b;
                    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                    cudaStreamSynchronize(s_c1);

                    if (g_first) cudaGraphLaunch(g_first, s_c1);
                    cudaStreamSynchronize(s_c1);

                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    cudaStreamSynchronize(s_up);

                    if (g_wu) cudaGraphLaunch(g_wu, s_up);
                    cudaStreamSynchronize(s_up);
                }
            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    // ... 返回loss值 ...
}
```

#### 3.3.2 关键修改点

**修改1: 简化传输逻辑**

```cpp
// 修改前：复杂的条件判断和同步
if (g_xfer_a) cudaGraphLaunch(g_xfer_a, s_trans);
sync_tr();
// ... 大量诊断代码 ...

// 修改后：与run_h2d_only完全一致
ts->wait_buffer_readable(buf_id);
cudaGraphLaunch(g_xfer, s_trans);
cudaStreamSynchronize(s_trans);
ts->set_buffer_readable(buf_id, false);
ts->set_buffer_writeable(buf_id, true);
```

**修改2: 明确的计算阶段插入**

```cpp
// 传输完成后立即启动计算
auto g_deep = (buf_id == 0) ? g_deep_a : g_deep_b;
if (g_deep) cudaGraphLaunch(g_deep, s_c1);
cudaStreamSynchronize(s_c1);
```

**修改3: 移除冗余的诊断代码**

- 删除`[DIAG-XFER0]`相关的GPU内存扫描代码
- 保留必要的错误检查和日志

---

## 4. 实施计划

### 4.1 Phase 1: 传输逻辑对齐（P0）

**目标**: 确保`run_train_epoch()`的传输部分与`run_h2d_only()`完全一致

**步骤**:

1. 复制`run_h2d_only()`中的传输代码到`run_train_epoch()`
2. 保留必要的变量声明和初始化
3. 移除复杂的批次分类逻辑（Batch 0预传输、正常批次等）
4. 统一使用简单的`buf_id = batch % 2`逻辑

**验证**:

- 运行`test_h2d_only_epoch.cpp`确保H2D-Only模式仍然工作
- 运行基本的单epoch训练测试

### 4.2 Phase 2: 计算图集成（P1）

**目标**: 在传输完成后正确插入计算图的执行

**步骤**:

1. 从当前`run_train_epoch()`中提取图获取逻辑
2. 在传输循环中插入计算图启动代码
3. 确保Stream同步的正确性
4. 处理图指针为空的情况（向后兼容）

**验证**:

- 运行MNIST单epoch测试，验证loss下降
- 对比SimpleTask版本的数值结果

### 4.3 Phase 3: 多epoch和验证（P2）

**目标**: 支持完整的训练循环和验证

**步骤**:

1. 恢复验证逻辑（`run_val_epoch()`）
2. 支持早停、EMA等高级功能
3. 恢复性能指标收集

**验证**:

- 运行3 epoch MNIST训练，验证准确率>85%
- 运行CIFAR-10和ImageNet测试

### 4.4 Phase 4: 性能优化和清理（P3）

**目标**: 性能调优和代码清理

**步骤**:

1. 分析Stream overlap机会
2. 优化同步点减少等待时间
3. 清理调试代码和注释

---

## 5. 关键注意事项

### 5.1 同步语义保持

- TransferStream必须在启动计算前完成同步
- 计算Stream之间可能需要同步（取决于图依赖）
- UpdateStream必须等待计算Stream完成

### 5.2 内存一致性

- 标签数据的D2D拷贝必须在传输Stream上完成
- 计算图启动前确保所有数据都已在GPU上

### 5.3 错误处理

- 保留所有CUDA错误检查
- 维护异常传播机制
- 确保资源清理正确

### 5.4 向后兼容性

- 支持图指针为空的情况（某些图可能不存在）
- 支持CPU模式（`#ifdef TR_USE_CUDA`）
- 支持不同GPU数量的配置

---

## 6. 测试策略

### 6.1 单元测试

1. **传输正确性**: `test_h2d_only_epoch.cpp`所有配置
2. **计算正确性**: 对比SimpleTask数值结果
3. **同步正确性**: 使用cuda-memcheck检测race condition

### 6.2 集成测试

1. **MNIST**: 3 epoch训练，目标准确率>85%
2. **CIFAR-10**: 单epoch测试，验证性能和正确性
3. **ImageNet**: 单batch测试，验证大规模数据

### 6.3 回归测试

1. 确保SimpleTask模式不受影响
2. 验证CPU模式仍然工作
3. 检查多GPU配置的正确性

---

## 7. 风险评估

### 7.1 高风险区域

1. **Stream同步**: 多Stream并发可能导致race condition
2. **内存覆盖**: AB区乒乓机制的时序错误
3. **图依赖**: 计算图的启动顺序可能影响正确性

### 7.2 缓解措施

1. 分阶段实施，每阶段充分测试
2. 保留当前代码作为fallback
3. 使用CUDA工具（nsight, cuda-memcheck）验证

---

## 8. 成功标准

### 8.1 功能正确性

- ✅ `test_dl_full.cpp` MNIST 3 epoch准确率>85%
- ✅ 所有数据集（MNIST/CIFAR/ImageNet）单batch训练无错误
- ✅ H2D-Only模式测试继续通过

### 8.2 性能指标

- ✅ 训练吞吐量不低于SimpleTask版本的90%
- ✅ 传输带宽与H2D-Only模式一致
- ✅ GPU利用率>80%

### 8.3 代码质量

- ✅ 代码结构与`run_h2d_only()`清晰对应
- ✅ 同步逻辑清晰可验证
- ✅ 错误处理完整

---

## 9. 总结

**核心洞察**: `compile_h2d_only()`已经证明`compile()`框架本身是正确的，问题出在`run_train_epoch()`的传输逻辑实现上。通过将验证过的H2D-Only传输逻辑逐行移植到`run_train_epoch()`，然后在传输完成后插入计算图执行，可以最小化风险地实现目标。

**实施路径**: 遵循"传输逻辑不变性原则"，分4个阶段逐步实施，每个阶段都有明确的验证标准，确保最终实现既正确又高效。

这个方案将使`compile()` + `run()`成为`compile_h2d_only()` + `run_h2d_only()`的超集，传输行为完全一致，同时支持完整的训练功能。

# 【小伙伴K】



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

| #    | 差异点                  | `run_h2d_only_train_epoch()` ✅                              | `run_train_epoch_gpu()` ❌                                    | 影响                                                         |
| ---- | ----------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | **label_smce D2D 拷贝** | 每次 H2D 后显式 `cudaMemcpyAsync(label_smce_ptr, ...)`      | **完全缺失**                                                 | loss 计算读到的 label 是旧值或垃圾，导致 loss NaN/错误       |
| 2    | **Buffer 可读等待**     | `ts->wait_buffer_readable(buf_id)`（条件变量阻塞）          | `while (!ts->buffer_is_readable(buf_id)) sleep(100us)`（自旋睡眠） | 时序脆弱，高负载下可能丢数据                                 |
| 3    | **Buffer 状态管理**     | **每个 rank** 都调用 `set_buffer_readable/writeable`        | **仅 rank 0** 调用                                           | 多卡下非 rank-0 的 Preprocessor worker 可能永远等不到 writeable 信号 |
| 4    | **H2D 后 sync 粒度**    | `cudaStreamSynchronize(s_trans)` 在 graph launch 后立即调用 | `sync_tr()` 与 `sync_comp()` 成对出现，但中间夹着计算图      | H2D 是否完成取决于后续计算图的 sync 时机，存在 race          |
| 5    | **诊断代码污染**        | 零诊断代码，纯生产逻辑                                      | 大量 `cudaMemcpy` + `LOG_INFO` 散布在热路径（~200 行诊断）   | 隐式同步 + 时序扰动 + 代码可读性极差                         |

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

| 文件                                            | 修改类型        | 说明                                                         |
| ----------------------------------------------- | --------------- | ------------------------------------------------------------ |
| `src/task/deep_learning_task.cpp`               | 新增方法 + 重写 | 新增 `execute_h2d_epoch_gpu()`、`execute_h2d_epoch_cpu()`；重写 `run_train_epoch_gpu()` 的 H2D 部分；重写 `run_val_epoch_gpu()` 的 H2D 部分；隔离诊断代码 |
| `include/renaissance/task/deep_learning_task.h` | 新增声明        | 声明 `execute_h2d_epoch_gpu()`、`execute_h2d_epoch_cpu`      |
| `src/task/deep_learning_task.cpp`               | 修改            | `run_h2d_only_train_epoch()` 和 `run_h2d_only_val_epoch()` 改为调用新的 `execute_h2d_epoch_*()` |

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

| 风险                                                       | 概率 | 影响              | 缓解措施                                                     |
| ---------------------------------------------------------- | ---- | ----------------- | ------------------------------------------------------------ |
| `label_smce` D2D 拷贝引入后，loss 仍异常                   | 中   | 需回到传输层排查  | Step 1~2 确保 H2D-only 路径不受影响，可快速二分              |
| 串行 H2D+计算导致性能下降 >30%                             | 低   | test 运行时间增加 | test_dl_full 只有 3 epochs MNIST，性能非瓶颈；Phase 2 恢复并行 |
| 多卡下 `wait_buffer_readable` 与 `set_buffer_*` 的线程安全 | 低   | 死锁或 race       | `run_h2d_only_train_epoch()` 已验证 8 卡正确，逻辑复用       |
| 计算图 stream 依赖被破坏                                   | 低   | 计算错误          | 计算图启动顺序和 stream 不变，仅 H2D 时序调整                |

---

## 9. 一句话总结

**`compile()` 不需要改，`run_train_epoch_gpu()` 的 H2D 子系统需要整体替换为 `run_h2d_only_train_epoch()` 的已验证实现。** 计算图部分（ZERO_GRAD → FIRST_FWD → DEEP_FWD_BWD → FIRST_BWD → ALLREDUCE → WEIGHT_UPDATE）保持现有 CUDA Graph 启动逻辑不变。先串行对齐确保正确性，再逐步恢复 H2D-计算并行以优化性能。

# 【小伙伴D】

# 

## 1. 背景与目标

### 1.1 当前状态

| 路径                                    | 编译                        | 运行               | 状态                                            |
| --------------------------------------- | --------------------------- | ------------------ | ----------------------------------------------- |
| `compile_h2d_only()` + `run_h2d_only()` | 仅编译 TRANSFER_A/B 图      | 仅执行 H2D 传输    | ✅ 所有数据集/场景/多RANK 均通过正确性和性能验证 |
| `compile()` + `run()`                   | 编译全部图（18+ GraphSlot） | 执行完整训练流水线 | ❌ 结果存在问题，test_dl_full.cpp 无法通过       |

### 1.2 目标

以 `compile_h2d_only()` + `run_h2d_only()` 为基础，彻底重写 `compile()` + `run()`，使得：

1. **传输行为完全一致**：新版 `compile()` + `run()` 的 H2D 传输行为（Buffer 管理、CUDA Graph 启动、流同步、Label 拷贝）与 `compile_h2d_only()` + `run_h2d_only()` **完全一致**。
2. **叠加计算能力**：在传输基础之上，加入 ZERO_GRAD、FIRST_FWD、DEEP_FWD_BWD、FIRST_BWD、ALLREDUCE、OPTIMIZER、INFERENCE 等计算图的执行。
3. **最终目标**：跑通 `test_dl_full.cpp`（DeepLearningTask 版，3 epoch MNIST MLP，Top-1 > 85%）。

---

## 2. 当前代码架构深度分析

### 2.1 `compile_h2d_only()` 分析

```cpp
// deep_learning_task.cpp:2135-2146
void DeepLearningTask::compile_h2d_only() {
    TR_CHECK(phase_ == Phase::PLANNING, ...);
    struct Guard { bool* p; Guard(bool* ptr) : p(ptr) { *p = true; } ~Guard() { *p = false; } } guard(&h2d_only_);
    compile();
}
```

**核心机制**：RAII guard 设置 `h2d_only_ = true`，然后调用 `compile()`。`compile()` → `compile_impl()` 的完整流程如下：

```
compile_freeze_global()
  → compile_invoke_on_prepare()     // 触发 on_prepare()，生成完整 IR（5 阶段编译）
  → compile_verify_memory_locked()  // 验证 memory plan
  → compile_alloc_hardware()        // 分配 GPU 内存（所有 tensor，不区分 h2d_only）
  → build_graph_atlas()             // ★ h2d_only_=true → 只收录 TRANSFER_A/B
  → pre_capture(atlas, ctx_ptrs)    // ★ 只 capture TRANSFER_A/B 两个 graph
  → build_exec_table()              // ★ h2d_only_=true → 只解析 XFER_A/B slot
  → compile_mark_compiled()
  → init_all() + cudaMallocHost(lr_pinned_)
```

**关键结论**：`compile()` 阶段，无论 `h2d_only_` 是 true 还是 false：

- `on_prepare()` 生成的 IR **完全相同**（均为完整蓝图编译）
- `compile_alloc_hardware()` 分配的内存 **完全相同**
- 唯一的差异在 `build_graph_atlas()` + `pre_capture()` + `build_exec_table()` 步骤中：
  - `h2d_only_=true`：只 capture 和 resolve TRANSFER_A/B
  - `h2d_only_=false`：capture 和 resolve 全部 18+ 个 GraphSlot

**这意味着**：当 `h2d_only_=false` 时，TRANSFER_A/B 的 CUDA Graph capture 流程与 `h2d_only_=true` 时**完全一致**（相同的 IR → 相同的 CUDA Graph），因为：

- `build_graph_atlas()` 中对 XFER 的处理代码路径完全相同（lines 484-495 vs 498-508）
- `stream_for(GraphId::TRANSFER_A)` = `StreamKind::TRANS`（lines 538-539）
- `build_exec_table()` 中 XFER_A/B 的 resolve 逻辑完全相同（lines 596-597 vs 612-613）

### 2.2 `run_h2d_only()` 分析

```cpp
// deep_learning_task.cpp:2486-2510
H2DRunResult DeepLearningTask::run_h2d_only() {
    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;
        H2DTestResult train_res = run_h2d_only_train_epoch();
        if (should_validate_this_epoch()) {
            H2DTestResult val_res = run_h2d_only_val_epoch();
        }
    }
}
```

#### 2.2.1 `run_h2d_only_train_epoch()` — GPU 路径（核心参考实现）

```cpp
// deep_learning_task.cpp:2178-2239 (GPU path)
for (int rank = 0; rank < K; ++rank) {
    threads.emplace_back([&, rank]() {
        cudaSetDevice(gpu_exec_.device_ids[rank]);
        // 获取 TransferStation（带重试）
        TransferStation* ts = ...;
        // 获取 XFER graph exec handles
        auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
        auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
        cudaStream_t s_trans = context(rank).stream(StreamKind::TRANS);
        // 获取 label pointer（用于 D2D copy 到 SMCE 槽位）
        auto label_ptr_a / label_ptr_b / label_smce_ptr
        size_t label_nbytes = ...

        for (int batch = 0; batch < batches; ++batch) {
            int buf_id = batch % 2;
            auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

            // ===== 核心传输三步（与新版 run() 必须保持一致）=====
            ts->wait_buffer_readable(buf_id);           // ① 等待 staging buffer 就绪
            cudaGraphLaunch(g_xfer, s_trans);           // ② 启动 CUDA Graph（H2D copy）
            cudaStreamSynchronize(s_trans);             // ③ 同步 TRANS 流
            ts->set_buffer_readable(buf_id, false);     // ④ 标记 buffer 已消费
            ts->set_buffer_writeable(buf_id, true);     // ⑤ 释放 buffer 给 Preprocessor
            // ===== 核心传输三步结束 =====

            // ⑥ Label D2D copy：从 I_A_LABEL/I_B_LABEL 拷贝到 LABEL_SMCE
            cudaMemcpyAsync(label_smce_ptr,
                (buf_id == 0) ? label_ptr_a : label_ptr_b,
                label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
            cudaStreamSynchronize(s_trans);
        }
    });
}
```

**传输模式特征**：

| 特征            | 值                                                           |
| --------------- | ------------------------------------------------------------ |
| Buffer 轮转     | `batch % 2` (A→B→A→B...)                                     |
| XFER Graph 选择 | buf_id=0 → XFER_A, buf_id=1 → XFER_B                         |
| 流              | s_trans (StreamKind::TRANS)                                  |
| 同步方式        | 每个 batch 后 `cudaStreamSynchronize(s_trans)`               |
| Buffer 管理     | **所有 rank 都调用** `set_buffer_readable/set_buffer_writeable` |
| Label 处理      | D2D copy 到 LABEL_SMCE 槽位（SoftmaxCrossEntropy 的数据输入槽） |
| Prep 线程       | 函数内部 `std::thread prep_thread([&]() { prep.train(); })`  |

#### 2.2.2 `run_h2d_only_val_epoch()` — GPU 路径（验证参考实现）

```cpp
// deep_learning_task.cpp:2360-2406 (GPU path)
for (int batch = 0; batch < val_batches; ++batch) {
    int buf_id = batch % 2;
    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

    ts->wait_buffer_readable(buf_id);
    cudaGraphLaunch(g_xfer, s_trans);
    cudaStreamSynchronize(s_trans);
    ts->set_buffer_readable(buf_id, false);
    ts->set_buffer_writeable(buf_id, true);
    // ⚠ 无 Label D2D copy（验证不需要 Label 参与 SoftmaxCE 的梯度计算）
}
```

### 2.3 现有 `compile()` + `run()` 分析

#### 2.3.1 `compile()` — 结论：**不需要改动**

`compile()` 调用 `compile_impl()`，由于 `h2d_only_` 默认为 `false`，走全量 GraphAtlas 路径：

- `build_graph_atlas()` 收录全部 GraphId（line 498-508）—— **包含 TRANSFER_A/B，路径与 h2d_only 一致**
- `build_exec_table()` 解析全部 18+ 个 slot（line 606-669）—— **包含 XFER_A/B，解析逻辑与 h2d_only 一致**

**`compile()` 已经正确编译了传输图和所有计算图，无需任何修改。**

#### 2.3.2 `run()` 入口

```cpp
// deep_learning_task.cpp:310-337
TrainingResult DeepLearningTask::run() {
    return run_impl(false);
}
TrainingResult DeepLearningTask::run_impl(bool dry_run) {
    if (using_gpu()) return run_gpu();
    else return run_cpu();  // run_cpu() 是 stub
}
```

#### 2.3.3 `run_gpu()` 框架（需保留）

```cpp
// deep_learning_task.cpp:677-878
TrainingResult DeepLearningTask::run_gpu() {
    // ① 日志输出、初始化（lines 684-718）
    // ② PRE-TEST：预启动 DEEP + OPT graph 验证（lines 720-745）— 可保留用于诊断
    // ③ Epoch 循环（lines 749-862）：
    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        // 渐进式分辨率调整
        // SEMA 切换
        // Prep 线程：prep.train()
        float train_loss = run_train_epoch_gpu();    // ★ 需要重写
        prep_thread.join();
        // 验证：
        if (should_validate_this_epoch()) {
            Prep 线程：prep.val()
            auto [vloss, vtop1, vtop5] = run_val_epoch_gpu(false);  // ★ 需调整
            // EMA 验证（如果启用 SEMA）
        }
        // 最佳指标追踪、日志、保存、早停
    }
    // ④ 返回 TrainingResult
}
```

**`run_gpu()` 的 epoch 循环框架本身是正确的，需要保留。** 需要替换的是其内部调用的 `run_train_epoch_gpu()` 和微调 `run_val_epoch_gpu()`。

#### 2.3.4 `run_train_epoch_gpu()` — 当前实现的问题

```cpp
// deep_learning_task.cpp:881-1453 (~570 lines)
```

**问题清单**：

| #    | 问题                       | 详情                                                         |
| ---- | -------------------------- | ------------------------------------------------------------ |
| 1    | **诊断代码过多**           | ~500 行 DIAG-* 日志代码夹杂在核心逻辑中，严重影响可读性和可维护性 |
| 2    | **Buffer 管理不一致**      | 只有 rank 0 调用 `set_buffer_readable/set_buffer_writeable`（line 1231-1234），而 `run_h2d_only` 是所有 rank 都调用 |
| 3    | **缺少 Label D2D copy**    | `run_h2d_only` 中显式将 I_A_LABEL / I_B_LABEL 拷贝到 LABEL_SMCE；当前 `run_train_epoch_gpu()` 没有此操作 |
| 4    | **传输与计算过度耦合**     | 4-phase 流水线设计将 ZERO_GRAD、FIRST_FWD、DEEP、XFER、FIRST_BWD、ALLREDUCE、OPT 全部交错，难以独立验证每个阶段 |
| 5    | **Batch 边界处理复杂**     | Batch 0 预传输、batches==1 边界、循环体(batch 0..batches-2)、last batch 四个分支各有不同逻辑 |
| 6    | **loss/scaling 初始化**    | lines 894-913 用 magic value 3.14 和 1.0 初始化 loss/scaling 设备内存 |
| 7    | **冗余的 cudaMemsetAsync** | 每个 batch 的 DEEP 之前都 memset loss=0（lines 1152-1156, 1389-1393） |
| 8    | **PRE-TEST 副作用**        | `run_gpu()` 中 lines 720-745 预启动 DEEP+OPT 会改变权重值（w13 被修改），然后 line 916-921 读取 BEFORE 值已经不是真正的初始值 |

#### 2.3.5 `run_val_epoch_gpu()` 分析

```cpp
// deep_learning_task.cpp:1469-1606
```

**结论**：`run_val_epoch_gpu()` 的传输模式已经非常接近 `run_h2d_only_val_epoch()`：

| 特征        | run_val_epoch_gpu() | run_h2d_only_val_epoch() |
| ----------- | ------------------- | ------------------------ |
| Buffer 轮转 | `batch % 2` ✅       | `batch % 2`              |
| XFER 选择   | A/B ✅               | A/B                      |
| Stream      | s_trans ✅           | s_trans                  |
| 同步        | sync_tr() ✅         | cudaStreamSynchronize    |
| set_buffer  | **仅 rank 0** ⚠️     | **所有 rank**            |
| 附加操作    | 推理 + 指标读取     | 无                       |

**只需修改一点**：将 `set_buffer_readable/set_buffer_writeable` 从 "仅 rank 0" 改为 "所有 rank"，与 `run_h2d_only_val_epoch()` 保持一致。

---

## 3. 核心差异对比表

### 3.1 传输核心流程对比（GPU 路径）

```
run_h2d_only_train_epoch()               run_train_epoch_gpu() (当前)
────────────────────────────────────     ──────────────────────────────
                                         [PRE-TEST: launch DEEP+OPT]
                                         [init loss/scaling with magic values]
                                         [DIAG-INIT: scan all tensors]
                                         
batches==0? return                       batches==0? (隐式，batches=0 时为 val_batches)
                                         
Prep 线程: prep.train()                  Prep 线程: prep.train()

Batch 0:
  wait_buffer(0)                         Batch 0 预传输:
  Launch XFER_A on TRANS                   wait_buffer(0)
  sync TRANS                               Launch XFER_A on TRANS  
  set_buf_consumed(0) ← 所有 rank         sync TRANS
  label D2D copy                           [DIAG-XFER0: 全量 scan CPU GPU]
                                           set_buf_consumed(0) ← 仅 rank 0
                                         
                                         batches==1 分支:
                                           ZG → FIRST_FWD → DEEP → BWD → DAR → GC → LR → FAR → OPT
                                           全部 sync
                                         
Batch 1..batches-1:                      循环 batch=0..batches-2:
  wait_buffer(buf_id)                      Phase 1: ZG ‖ FIRST_FWD
  Launch XFER on TRANS                       sync
  sync TRANS                                [DIAG-PHASE1: 全量 scan]
  set_buf_consumed(buf_id)                Wait next buffer
  label D2D copy                          Phase 2: DEEP ‖ XFER(next)
                                             [DIAG-B0: loss/top1/scaling/w 等]
                                             sync COMP + TRANS
                                             [DIAG-S1/S2/S3/S4: 逐步验证]
                                             set_buf_consumed ← 仅 rank 0
                                           Phase 3: FIRST_BWD ‖ DEEP_AR
                                           Phase 4: LR → FIRST_AR → OPT
                                             全部 sync
                                         
                                         Last batch (batches-1):
                                           ZG → FIRST_FWD → DEEP → BWD → DAR → GC → LR → FAR → OPT

循环结构: for (batch=0; batch<batches; ++batch)   循环: for (batch=0; batch<batches-1; ++batch) + 最后 batch 单独
```

### 3.2 关键行为差异

| 行为            | run_h2d_only                   | run_train_epoch_gpu (当前)     | 是否需要统一 |
| --------------- | ------------------------------ | ------------------------------ | ------------ |
| Buffer 消费标记 | 所有 rank 调用                 | 仅 rank 0 调用                 | ✅ 需统一     |
| Label D2D copy  | 每 batch 执行                  | 不执行                         | ✅ 需统一     |
| XFER 后同步     | cudaStreamSynchronize(s_trans) | cudaStreamSynchronize(s_trans) | ✅ 已一致     |
| XFER Graph 选择 | buf_id % 2 → XFER_A/B          | from_a → XFER_A/B              | ✅ 已一致     |
| TRANS Stream    | s_trans                        | s_trans                        | ✅ 已一致     |
| 诊断代码        | 无                             | ~500 行                        | ✅ 全部移除   |

---

## 4. 重写方案

### 4.1 总体策略

**compile() 路径：不动。**

- `compile()` → `compile_impl()` 已正确生成全部 CUDA Graph（含 XFER 和所有计算图）
- `build_graph_atlas()` 和 `build_exec_table()` 在 `h2d_only_=false` 时的行为已验证正确

**run() 路径：重写。**

- **移除** `run_train_epoch_gpu()` 当前全部实现（lines 881-1453），用新实现替代
- **微调** `run_val_epoch_gpu()`：将 set_buffer 从仅 rank 0 改为所有 rank
- **保留** `run_gpu()` 的 epoch 循环框架，内部调用新的 epoch 函数

### 4.2 新 `run_train_epoch_gpu()` 设计

**设计原则**：

1. 传输部分直接复用 `run_h2d_only_train_epoch()` 的 GPU 路径逻辑（拷贝过来，不做抽象层）
2. 在每个 batch 的传输完成后，顺序执行该 batch 的计算图
3. 不引入流水线重叠（先保证正确性，后续优化再说）
4. 移除全部诊断代码（约 500 行）

**伪代码**：

```
float run_train_epoch_gpu() {
    batches = prep.steps_per_epoch();
    K = num_gpus_;
    using_amp = registry.using_amp();
    frozen = is_first_layer_frozen();
    
    // Prep 线程已在 run_gpu() 中启动（不在函数内部管理 Prep 线程）
    // 因为 run_gpu 中有 prep_thread.join()，这里需要对齐
    
    threads[K];
    for (rank in 0..K-1):
        threads[rank] = lambda:
            cudaSetDevice(device_ids[rank]);
            
            // 获取 graph exec handles
            g_xfer_a/b = gpu_exec_.graphs[rank][XFER_A/B];
            g_deep_a/b = gpu_exec_.graphs[rank][FWD_BWD_DEEP_A/B];
            g_first_bwd_a/b = gpu_exec_.graphs[rank][FIRST_LAYER_BWD / FIRST_LAYER_BWD_B];
            g_first_fwd_a/b = gpu_exec_.graphs[rank][FIRST_FWD_A/B];
            g_zg = gpu_exec_.graphs[rank][ZERO_GRAD];
            g_dar = gpu_exec_.graphs[rank][DEEP_ALLREDUCE];
            g_far = gpu_exec_.graphs[rank][FIRST_LAYER_ALLREDUCE];
            g_wu = gpu_exec_.graphs[rank][WEIGHT_UPDATE];
            g_gc = gpu_exec_.graphs[rank][GRAD_CONVERT];  // 仅 AMP
            
            // 获取 stream handles
            s_trans, s_c1, s_c2, s_up;
            
            // 获取 TransferStation（带重试）
            ts = ...;
            
            // 获取 label pointers（用于 D2D copy）
            label_ptr_a/b, label_smce_ptr, label_nbytes;
            
            // 获取 lr device pointer
            lr_dev_ptr;
            
            for (batch = 0; batch < batches; ++batch):
                int buf_id = batch % 2;
                
                // ═══════════════════════════════════
                // Part A: H2D 传输（与 run_h2d_only_train_epoch 完全一致）
                // ═══════════════════════════════════
                ts->wait_buffer_readable(buf_id);
                cudaGraphLaunch(buf_id==0 ? g_xfer_a : g_xfer_b, s_trans);
                cudaStreamSynchronize(s_trans);
                
                ts->set_buffer_readable(buf_id, false);   // 所有 rank 调用
                ts->set_buffer_writeable(buf_id, true);   // 所有 rank 调用
                
                cudaMemcpyAsync(label_smce_ptr,
                    (buf_id == 0) ? label_ptr_a : label_ptr_b,
                    label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
                cudaStreamSynchronize(s_trans);
                
                // ═══════════════════════════════════
                // Part B: 计算
                // ═══════════════════════════════════
                bool from_a = (buf_id == 0);
                
                // Phase 1: ZERO_GRAD ‖ FIRST_FWD
                if (g_zg) cudaGraphLaunch(g_zg, s_up);
                if (from_a ? g_first_fwd_a : g_first_fwd_b)
                    cudaGraphLaunch(from_a ? g_first_fwd_a : g_first_fwd_b, s_c1);
                cudaStreamSynchronize(s_c1);
                cudaStreamSynchronize(s_up);
                
                // Phase 2: DEEP_FWD_BWD
                if (from_a ? g_deep_a : g_deep_b)
                    cudaGraphLaunch(from_a ? g_deep_a : g_deep_b, s_c1);
                cudaStreamSynchronize(s_c1);
                
                // Phase 3: FIRST_BWD ‖ DEEP_ALLREDUCE
                if (!frozen) {
                    auto g_bwd = from_a ? g_first_bwd_a : g_first_bwd_b;
                    if (g_bwd) cudaGraphLaunch(g_bwd, s_c1);
                }
                if (g_dar) cudaGraphLaunch(g_dar, s_up);
                cudaStreamSynchronize(s_c1);
                cudaStreamSynchronize(s_up);
                
                // Phase 4: GRAD_CONVERT (AMP only)
                if (using_amp && g_gc) {
                    cudaGraphLaunch(g_gc, s_up);
                    cudaStreamSynchronize(s_up);
                }
                
                // Phase 5: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE
                float lr = fetch_lr_for_batch(batch);
                *lr_pinned_[rank] = lr;
                cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                cudaMemcpyHostToDevice, s_up);
                if (g_far) cudaGraphLaunch(g_far, s_up);
                if (g_wu) cudaGraphLaunch(g_wu, s_up);
                cudaStreamSynchronize(s_up);
            end for
            
    join all threads;
    检查异常;
    
    // 读取 train loss（仅 rank 0）
    float train_loss = 0.0f;
    if (loss_id >= 0)
        train_loss = fetch_from_rank(...);
    return train_loss;
}
```

### 4.3 `run_val_epoch_gpu()` 微调

**仅需修改一处**（line 1564-1568）：

```cpp
// 当前代码（仅 rank 0）：
if (rank == 0) {
    ts->set_buffer_readable(buf, false);
    ts->set_buffer_writeable(buf, true);
}

// 改为（所有 rank）：
ts->set_buffer_readable(buf, false);
ts->set_buffer_writeable(buf, true);
```

这样使 `run_val_epoch_gpu()` 的传输行为与 `run_h2d_only_val_epoch()` 完全一致。

### 4.4 `run_gpu()` 的调整

`run_gpu()` 框架保持不变，但需调整以下细节：

1. **移除 PRE-TEST**（lines 720-745）：预启动 DEEP + OPT graph 会修改权重值，干扰训练初始状态。如果确实需要预验证 CUDA Graph 的正确性，应该在 `build_exec_table()` 之后、`compile_mark_compiled()` 之前完成，而不是在 `run()` 阶段。

2. **Prep 线程管理**：当前 `run_gpu()` 在 epoch 循环内创建 `prep_thread`，然后 `run_train_epoch_gpu()` 内部不管理 Prep 线程。新设计中 `run_train_epoch_gpu()` 也不管理 Prep 线程（由 `run_gpu()` 统一管理），保持此模式不变。

3. **SEMA 切换**：当前 `apply_sema_switch()` 是 stub（line 459），无需改动。

4. **Loss 读取**：当前 `run_train_epoch_gpu()` 返回 `float train_loss`，由 `run_gpu()` 用于日志。新实现保持此接口。

### 4.5 `run_cpu()` 不在本次范围

`run_cpu()` 当前是 stub（line 1460），不在本次重写范围内。CPU 路径可以在 GPU 路径稳定后单独实现。

---

## 5. 详细修改清单

### 文件：`src/task/deep_learning_task.cpp`

#### 修改 A：重写 `run_train_epoch_gpu()`（lines 881-1453 → 新实现）

**操作**：删除 lines 881-1453（约 570 行），替换为新实现（预计约 150 行）。

**新函数结构**：

```
float DeepLearningTask::run_train_epoch_gpu()             // line 881
{
#ifdef TR_USE_CUDA
    // 1. 获取参数 (lines 882-887，保留)
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    ...
    
    // 2. 初始化（保留 loss_id / sc_id 获取逻辑，移除 magic value 初始化）
    int32_t loss_id = active_memory_plan_->baseline().loss;
    
    // 3. 多 rank 线程
    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            // 3a. cudaSetDevice
            // 3b. 获取 graph exec handles（保留 lines 932-947，新增 g_first_bwd_b 和 g_first_fwd_a/b）
            // 3c. 获取 stream handles（保留 lines 949-960）
            // 3d. 获取 TransferStation（复用 h2d_only 的重试逻辑）
            // 3e. 获取 label pointers（复用 h2d_only 的 label copy 逻辑）
            
            // 4. Batch 循环
            for (int batch = 0; batch < batches; ++batch) {
                int buf_id = batch % 2;
                
                // === Part A: H2D 传输 ===
                ts->wait_buffer_readable(buf_id);
                cudaGraphLaunch((buf_id == 0) ? g_xfer_a : g_xfer_b, s_trans);
                cudaStreamSynchronize(s_trans);
                
                ts->set_buffer_readable(buf_id, false);
                ts->set_buffer_writeable(buf_id, true);
                
                cudaMemcpyAsync(label_smce_ptr,
                    (buf_id == 0) ? label_ptr_a : label_ptr_b,
                    label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
                cudaStreamSynchronize(s_trans);
                
                // === Part B: 计算 ===
                bool from_a = (buf_id == 0);
                
                // ZERO_GRAD ‖ FIRST_FWD
                if (g_zg) cudaGraphLaunch(g_zg, s_up);
                auto g_fwd = from_a ? g_fwd_a : g_fwd_b;
                if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                sync_comp(); sync_up();
                
                // DEEP_FWD_BWD
                auto g_deep = from_a ? g_deep_a : g_deep_b;
                if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                sync_comp();
                
                // FIRST_BWD ‖ DEEP_ALLREDUCE
                if (!frozen) {
                    auto g_bwd = from_a ? g_first_bwd_a : g_first_bwd_b;
                    if (g_bwd) cudaGraphLaunch(g_bwd, s_c1);
                }
                if (g_dar) cudaGraphLaunch(g_dar, s_up);
                sync_comp(); sync_up();
                
                // GRAD_CONVERT (AMP)
                if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }
                
                // LR → FAR → OPT
                lr = fetch_lr_for_batch(batch);
                *lr_pinned_[rank] = lr;
                cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float), cudaMemcpyHostToDevice, s_up);
                if (g_far) cudaGraphLaunch(g_far, s_up);
                if (g_wu) cudaGraphLaunch(g_wu, s_up);
                sync_up();
            }
        });
    }
    join/check;
    
    // 5. 读取 loss
    float train_loss = ...;
    return train_loss;
#else
    return 0.0f;
#endif
}
```

**关键注意事项**：

- **保留** `g_first_bwd_b` 的获取（在当前的 `run_train_epoch_gpu()` 中有 `g_first_b` = `g_tab[S(FIRST_LAYER_BWD_B)]`），因为 Double Buffer 的 B 侧需要对应 B 版本的首层反向图
- **保留** `fetch_lr_for_batch()` 和 `lr_pinned_` 机制（LR 需要在每 batch 后更新 H2D）
- **保留** `frozen` 检查（`is_first_layer_frozen()` — `freeze_after_` epoch 后跳过首层反向）
- **移除** 所有 DIAG-* 日志（DIAG-XFER0, DIAG-INIT, DIAG-PHASE1, DIAG-S1~S4, DIAG-B0, GEXEC, LOSS-DBG, WGHT-DBG 等）
- **移除** magic value 初始化（`init_val=3.14f`, `sc_val=1.0f`）
- **移除** cudaMemsetAsync(loss=0) 调用 — 这些应该由计算图内部处理

#### 修改 B：微调 `run_val_epoch_gpu()`（lines 1564-1568）

**操作**：将 `set_buffer_readable/set_buffer_writeable` 从 "仅 rank 0" 改为 "所有 rank"。

```
// 修改前 (lines 1564-1568):
if (rank == 0) {
    ts->set_buffer_readable(buf, false);
    ts->set_buffer_writeable(buf, true);
}

// 修改后:
ts->set_buffer_readable(buf, false);
ts->set_buffer_writeable(buf, true);
```

#### 修改 C：调整 `run_gpu()`（lines 720-745 和 line 909-921）

**操作**：

1. 移除 PRE-TEST 块（lines 720-745）：

   ```
   // 删除以下代码：
   {
       cudaSetDevice(gpu_exec_.device_ids[0]);
       ...
       LOG_INFO << "[PRE-TEST] w13[0] after direct launch=" << w0;
   }
   ```

2. 移除 `run_gpu()` 中的 WGHT-DBG BEFORE 读取（line 916-921）：这部分在当前 `run_train_epoch_gpu()` 内部，重写后自然消失。

3. 移除 `run_train_epoch_gpu()` 中 WGHT-DBG AFTER 读取（line 1445-1450）：新实现不再包含。

#### 修改 D：清理废弃代码（可选，Phase 2）

以下代码在重写后将不再被调用，可以在确认新实现稳定后移除：

- `run_train_epoch()`（lines 343-441）：旧的 `TaskBase::run("xfer_a")` + `TaskBase::run("fwd_bwd_deep_a")` 风格，非 GPU 路径，已被新实现取代
- `run_val_epoch()`（lines 443-453）：空的 TODO stub，已被 `run_val_epoch_gpu()` 取代

**但**：`run_train_epoch()` 和 `run_val_epoch()` 在 `deep_learning_task.h` 中被声明为 private（lines 420, 423），需要确认是否有其他调用方。如果没有，可以移除声明和实现。

---

## 6. 实现步骤（建议顺序）

### Phase 1：重写 `run_train_epoch_gpu()`

1. 在当前函数**下方**（line 1453 之后）新增 `run_train_epoch_gpu_v2()`
2. 暂时保留原 `run_train_epoch_gpu()`
3. 在 `run_gpu()` 中加一个编译开关调用 v2（或直接替换调用）
4. 编译并测试 `test_dl_full.cpp`

### Phase 2：微调 `run_val_epoch_gpu()`

1. 修改 set_buffer 调用
2. 验证 val 路径

### Phase 3：清理

1. 移除 `run_gpu()` 中的 PRE-TEST 块
2. 移除旧的 `run_train_epoch_gpu()` 实现
3. 将 v2 重命名为主函数名

---

## 7. 风险评估

| 风险                                            | 等级 | 缓解措施                                                     |
| ----------------------------------------------- | ---- | ------------------------------------------------------------ |
| 新实现与旧 Pipeline 的行为差异导致训练不收敛    | 中   | 逐步验证：先跑 1 epoch H2D only → 加 compute 跑 1 epoch → 跑 3 epoch 完整训练 |
| Label D2D copy 是否正确处理                     | 低   | `run_h2d_only` 已验证此逻辑正确，Copy-Paste 过来即可         |
| 所有 rank set_buffer 是否影响 Preprocessor 行为 | 低   | `run_h2d_only` 中所有 rank 已调用 set_buffer，且通过多 RANK 测试 |
| LR pinned memory 是否正确传递                   | 低   | 保留现有 `lr_pinned_` + `fetch_lr_for_batch` 机制，已经在 PRE-TEST 中得到验证 |
| 移除诊断代码后能否调试                          | 低   | 可以在新实现中保留关键的 `LOG_INFO` 行（如 batch 序号、loss 值） |
| `g_first_bwd_b` 是否需要                        | 中   | 需要检查 `build_exec_table()` 是否解析了 `FIRST_LAYER_BWD_B` — **已确认** line 617 有解析但 `kRequired` 中未列出，可能需要确保非 null 或回退用 `FIRST_LAYER_BWD` |

---

## 8. 需要确认的开放问题

1. **`g_first_bwd_b` (FIRST_LAYER_BWD_B) vs `g_first_bwd` (FIRST_LAYER_BWD)**：当前 `build_exec_table()` 中 `kRequired` 只检查了 `FIRST_LAYER_BWD`，未检查 `FIRST_LAYER_BWD_B`。`kRequired` 列表需要增加 `FIRST_LAYER_BWD_B` 吗？还是 B 侧可以用 A 侧的图？

2. **`g_first_fwd_b` (FIRST_FWD_B)**：类似问题 — `kRequired` 列表中没有，但如果 batch 1 需要 FIRST_FWD_B，是否需要确保非 null？

3. **CPU 路径**：`run_cpu()` 当前是 stub，不做任何事。未来是否计划支持 CPU 训练路径？如果是，`run_impl()` 中 `run_cpu()` 分支需要独立设计。

4. **`run_train_epoch()`（旧版非 GPU 路径）**：lines 343-441 目前是否还有调用方？能否安全移除？

---

## 9. 总结

本次重写的核心思路是 **"compile() 不动，run() 传输部分对齐 h2d_only，计算部分叠加到传输之后"**：

| 模块                           | 改动量             | 说明                           |
| ------------------------------ | ------------------ | ------------------------------ |
| `compile()` / `compile_impl()` | **0 行**           | 已正确编译所有图（含 XFER）    |
| `build_graph_atlas()`          | **0 行**           | h2d_only_=false 时已包含全部图 |
| `build_exec_table()`           | **0 行**           | 已正确解析全部 GraphSlot       |
| `run_gpu()`                    | **~15 行删除**     | 移除 PRE-TEST + WGHT-DBG       |
| `run_train_epoch_gpu()`        | **-570 + ~150 行** | 删除旧实现，写入新实现         |
| `run_val_epoch_gpu()`          | **2 行修改**       | set_buffer 改为所有 rank       |
| **总计**                       | **约 -430 行**     | 大量删除诊断代码               |

新 `compile() + run()` = 现有的 `compile()` + 新的 `run_gpu()`（内部调用新的 `run_train_epoch_gpu()` + 微调后的 `run_val_epoch_gpu()`），其中：

- **传输部分**：与 `run_h2d_only_train_epoch()`/`run_h2d_only_val_epoch()` **完全一致**
- **计算部分**：在传输完成后顺序执行 ZERO_GRAD → FIRST_FWD → DEEP_FWD_BWD → FIRST_BWD → ALLREDUCE → OPTIMIZER
- **框架部分**：保留 `run_gpu()` 的多 epoch 循环、验证、SEMA、渐进分辨率、早停、模型保存



# 【用户补充】

性能、数学正确性、多RANK支持、CPU/GPU/AMP路径支持都非常重要（GPU和AMP路径大体类似），一定要确保传输的部分跟compile_h2d_only()+run_h2d_only()一样，因为那是通过了验证的。

另外，compile()+run()就没必要设TR_TEST_TWO_BATCH_CORRECTION这个宏了。

还有，你可以对deep_learning_task.cpp的所有代码进行修改，但你要保证compile_h2d_only()+run_h2d_only()及其测试样例还能正常运行。





