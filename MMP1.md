# 【从 Dry Run 到真实训练：MLP 实现方案】

**版本**: V1.0  
**日期**: 2026-05-22  
**基线**: TR4 v4.20.1  
**目标**: 将 `tests/ref/mnist_mlp_3.cpp` 从 `compile_for_dry_run()+dry_run()` 改为 `compile()+run()`

---

## 一、当前状态诊断

### 1.1 已具备的能力

| 模块 | 状态 | 验证 |
|------|:----:|------|
| 核心算子(FC/Tanh/ReLU/SoftmaxCE/FLATTEN,FWD+BWD) | ✅ | `test_mlp_final.cpp` 三模式通过 |
| RANGE_H2D_COPY_A/B (异步传输) | ✅ | `test_h2d_copy_a/b.cpp` pass |
| 优化器 (SGD+Momentum weight/bias) | ✅ | `test_sgd_weight/bias.cpp` pass |
| Scheduler (CosineAnnealing/Step/PolynomialLR) | ✅ | 单元测试 pass |
| Initializer (DTensor 初始化) | ✅ | 接口完整 |
| compile_for_dry_run() | ✅ | `mnist_mlp_3.exe` 输出完整 GraphAtlas |
| SimpleTask::compile()+run() | ✅ | 多RANK NCCL graph 捕获+启动 works |

### 1.2 核心差距

| 问题 | 严重度 | 位置 |
|------|:------:|------|
| **`build_simple_atlas()` 将所有图合并为一个 `SIMPLE_TASK_GRAPH`** | 🔴 致命 | [task_base.cpp:L633](file:///r:/renaissance/src/task/task_base.cpp#L633) |
| **`run_gpu()` 缺少 ZERO_GRAD/ALLREDUCE/OPTIMIZER/CHECK_NAN** | 🔴 致命 | [deep_learning_task.cpp:L690-L830](file:///r:/renaissance/src/task/deep_learning_task.cpp#L690-L830) |
| **数据初始化缺失** (ArenaKeeper memset + Initializer) | 🔴 致命 | 编译阶段 |
| **Preprocessor 线程协调缺失** | 🟡 严重 | 执行阶段 |
| **验证 epoch 未实现** | 🟡 严重 | `run_val_epoch_gpu()` |
| AMP 转换图 (GRAD_CONVERT) 未接入 | 🟢 低 | 非必需(AMP=false for MNIST) |
| 推理图 (INF_MAIN/INF_EMA) 未接入 | 🟢 低 | 验证 epoch 需要 |

---

## 二、图分解分析

### 2.1 Dry Run 输出的真实图结构

Compiler 已经正确按 `GraphId` 分解了 train graph：

```
GraphId 0 / TRANSFER_A:     RANGE_H2D_COPY_A           [I_A → device]
GraphId 1 / TRANSFER_B:     RANGE_H2D_COPY_B           [I_B → device]
GraphId 2 / FIRST_FWD_A:    FLATTEN_FP32_FWD           [data from A]
GraphId 3 / FIRST_FWD_B:    FLATTEN_FP32_FWD           [data from B]
GraphId 4 / DEEP_FWD_BWD:   FC1→ReLU1→FC2→ReLU2→FC3   [完整FWD+BWD]
                            →SoftmaxCE_FWD→SoftmaxCE_BWD
                            →FC3_BWD→ReLU2_BWD→FC2_BWD
                            →ReLU1_BWD→FC1_BWD
GraphId 5 / ZERO_GRAD:      RANGE_CLEAR                [梯度清零]
GraphId 6 / FIRST_BWD:      FLATTEN_FP32_BWD           [首层反向]
GraphId 7 / FIRST_COMM:     RANGE_SUM_ALLREDUCE        [首层梯度同步]
GraphId 8 / DEEP_COMM:      RANGE_SUM_ALLREDUCE        [深层梯度同步]
GraphId 9 / CAST_AND_CHECK: RANGE_CHECK_NAN            [NaN检测]
GraphId 11/ OPTIMIZER:      RANGE_UPDATE_WEIGHT_MOMENTUM
                            RANGE_UPDATE_BIAS_MOMENTUM
```

### 2.2 图分拆到 GraphSlot 的映射

MLP 不需要 A/B 两份 DEEP_FWD_BWD（不像 ResNet 有分辨率差异），但需要 A/B 两份 XFER 和 FIRST_FWD：

```
XFER_A             ← TRANSFER_A
XFER_B             ← TRANSFER_B
FWD_BWD_DEEP_A     ← FIRST_FWD_A + DEEP_FWD_BWD (来自A区)
FWD_BWD_DEEP_B     ← FIRST_FWD_B + DEEP_FWD_BWD (来自B区)
FIRST_LAYER_BWD    ← FIRST_BWD
ZERO_GRAD          ← ZERO_GRAD
DEEP_ALLREDUCE     ← DEEP_COMM
FIRST_LAYER_ALLREDUCE ← FIRST_COMM
WEIGHT_UPDATE      ← OPTIMIZER
CAST_AND_CHECK     ← CAST_AND_CHECK
INF_MAIN_A/B       ← INF_MAIN_A/B  (验证用)
```

### 2.3 关键设计决策：图要"拆分"还是"合并"

当前 Compiler 已经按 `GraphId` 拆分了子图。问题是 `build_simple_atlas()` 将它们全部合并。

**推荐方案**：修复 `build_simple_atlas()`，改为从 `ComputationGraph` 中提取每个 `GraphId` 的子图单独捕获。

Compiler 生成的 `ComputationGraph` 内部已有 `nodes(GraphId)` 方法，按 GraphId 分别提取节点即可。参考 SimpleTask 的 `CapturedGraph::capture()`，每个子图独立捕获成 `cudaGraphExec_t`。

### 2.4 GraphSlot → GraphId → 去重

多 shape 变体（progressive resolution）情况下，同一个 GraphId 在不同 shape 下可能有不同实现。但 MLP MNIST 只有一种 shape `(128, 28, 28, 1)`，不涉及变体。只需处理 shape variant 0。

---

## 三、分步实现计划

### 阶段 0：修复 `compile()` → 正确捕获所有子图

**目标**: `compile()` 后，所有 GraphSlot 对应的 cudaGraphExec_t 均非空

**改动文件**: `src/task/task_base.cpp`、`src/task/deep_learning_task.cpp`

**步骤**:

1. **重写 `build_simple_atlas()`**
   - 从 Compiler 生成的 `named_graphs_` 中提取每个 `GraphId` 的子图
   - 为每个 GraphId 创建独立的 Atlas Slot
   - 不再将所有图合并为 `SIMPLE_TASK_GRAPH`
   
   映射逻辑：
   ```
   graph "train" → 提取 TRANSFER_A, TRANSFER_B, FIRST_FWD_A, FIRST_FWD_B,
                    DEEP_FWD_BWD, ZERO_GRAD, FIRST_BWD, FIRST_COMM,
                    DEEP_COMM, CAST_AND_CHECK, OPTIMIZER
   graph "inference" → 提取 INF_MAIN_A, INF_MAIN_B
   ```

2. **扩展 `build_exec_table()`** 
   - 当前已按 GraphSlot 枚举索引，但需要确认 "xfer_a" 等名字能否正确解析
   - 在 `name_to_gid_` 中建立正确的映射：
     ```
     "xfer_a"        → TRANSFER_A
     "xfer_b"        → TRANSFER_B
     "fwd_bwd_deep_a" → {FIRST_FWD_A, DEEP_FWD_BWD} 组合
     "fwd_bwd_deep_b" → {FIRST_FWD_B, DEEP_FWD_BWD} 组合
     "first_layer_bwd" → FIRST_BWD
     "zero_grad"     → ZERO_GRAD
     "deep_allreduce" → DEEP_COMM
     "first_layer_allreduce" → FIRST_COMM
     "weight_update"  → OPTIMIZER
     "cast_and_check" → CAST_AND_CHECK
     ```

3. **数据初始化**
   - ArenaKeeper 全部 RANK `cudaMemset(0)` 
   - 调用 Initializer 初始化所有权重 DTensor
   - 分配学习率张量（已实现，确认可用）
   
   位置：`compile_impl()` 中 `compile_alloc_hardware()` 之后、`pre_capture()` 之前

4. **Warmup**
   - 参考 SimpleTask 的 warmup 循环
   - 遍历 DEEP_FWD_BWD 子图中所有需要 warmup 的 cuDNN 算子
   - 确保 warmup 在 graph capture 之前完成

**验证**:
- `compile()` 后打印 `gpu_exec_` 表，确认所有 Required slots 非空
- 每个 RANK 的 cudaGraphExec_t 指针均非 `nullptr`

---

### 阶段 1：A 区传输对接验证

**目标**: 与 Preprocessor 成功对接，A 区数据正确传输到 GPU

**改动文件**: `src/task/deep_learning_task.cpp`

**步骤**:

1. **在 `run_gpu()` epoch 循环中启动 Preprocessor**
   ```cpp
   // 主线程启动 Preprocessor（训练模式）
   std::thread prep_thread([&prep]() {
       prep.train();  // 内部启动上百个预处理线程
   });
   ```

2. **`run_train_epoch_gpu()` 第一阶段：仅测 A 区传输**
   - 等待 TransferStation buffer[0] readable
   - 启动 `XFER_A` 图 (`cudaGraphLaunch`)
   - `cudaStreamSynchronize(s_trans)` 等待传输完成
   - 从 `I_A_DATA`/`I_A_LABEL` fetch 前几个值，与 TransferStation 原始数据对比
   - 标记 buffer[0] writable

3. **停止 Preprocessor**
   ```cpp
   prep.stop();
   prep_thread.join();
   ```

**验证标准**:
- 传输的数据与 TransferStation 原始数据一致
- 标签（label）和图像像素值完全匹配
- 无 CUDA 错误

---

### 阶段 2：A/B 乒乓 + 完整单 Batch

**目标**: 单个 epoch 内完成传输↔计算乒乓 + ZERO_GRAD + ALLREDUCE + OPTIMIZER

**改动文件**: `src/task/deep_learning_task.cpp`

**步骤**:

1. **`run_train_epoch_gpu()` 完整单 batch 流程**:
   ```
   for each batch:
     1. ZERO_GRAD       (UPDATE流)  ← 清零上轮梯度
     2. XFER_A/B        (TRANS流)   ← 传输当前batch数据
     3. FWD_BWD_DEEP_A/B(COMP_1流)  ← 所有层FWD+BWD (与传输并行!)
     4. csync COMP_1/2/3/TRANS      ← 同步所有计算流+传输流
     5. FIRST_LAYER_BWD (COMP_1流)  ← FLATTEN BWD
     6. sync COMP_1/2/3
     7. DEEP_ALLREDUCE  (UPDATE流)  ← 深层梯度AllReduce
        FIRST_LAYER_ALLREDUCE(UPDATE) ← 首层梯度AllReduce (可以与FIRST_BWD并行!)
     8. sync UPDATE
     9. WEIGHT_UPDATE   (UPDATE流)  ← SGD+Momentum权重更新
     10. sync UPDATE
     11. scheduler.step()
   ```

2. **传输与计算的重叠策略**（MLP.md 的核心思路）:
   ```
   Batch N:   [XFER_A] ──→ [FIRST_FWD_A+DEEP_FWD_BWD] ──→ [FIRST_BWD+ALLREDUCE] ──→ [OPTIMIZER]
   Batch N+1:         [XFER_B] ──→ [FIRST_FWD_B+DEEP_FWD_BWD] ──→ [FIRST_BWD+ALLREDUCE]
              ↑ 传输与深度计算在 COMP_1 和 TRANS 流上并行 ↑
   ```

3. **last batch 特殊处理**:
   - 最后一个 batch 不需要传输（上个 batch 已传输）
   - 直接使用对应 DEEP_FWD_BWD 图

4. **空图跳过**:
   - 非 AMP 模式下 `GRAD_CONVERT` 为空 → `cudaGraphExec_t == nullptr` → 跳过
   - 单卡模式下 `FIRST_COMM`/`DEEP_COMM` 为空 → 跳过

**验证标准**:
- 单个 epoch 所有 batch 完整执行
- A/B 乒乓无卡死
- Loss 值在合理范围（不出现 NaN/Inf）

---

### 阶段 3：完整训练 + 验证 + 指标

**目标**: 20 epoch 完整训练并验证

**改动文件**: `src/task/deep_learning_task.cpp`

**步骤**:

1. **完善 `run_gpu()` 主循环**
   - 参考现有 [run_gpu():L537-L688](file:///r:/renaissance/src/task/deep_learning_task.cpp#L537-L688) 框架
   - 加入真正的 Preprocessor 线程启动/停止
   - 加入 epoch 计时与日志

2. **实现 `run_val_epoch_gpu()`**
   - Preprocessor 切换到 VALIDATION 模式
   - 使用 INF_MAIN_A/B 推理图
   - 收集 loss、Top-1 指标

3. **指标收集**
   - 从 GPU fetch Loss DTensor → CPU Tensor
   - Top-1 统计（SoftmaxCE 已输出 probs → argmax → compare with labels）

4. **学习率调度**
   - 每个 batch 后 `scheduler.step()`（已实现）
   - 确认在 rank=0 上单次执行（避免多 RANK 重复执行）

5. **模型保存** (可选，作为 bonus)
   - `save_model_to(path)` 接口已存在
   - 保存训练中最佳模型

**验证标准**:
- 20 epoch 完整训练无崩溃
- 单 epoch 约 469 batches 全部正确处理
- MNIST 验证准确率达到 98%+

---

## 四、CUDA Stream 架构

已有 5 个物理流 ([types.h:L220](file:///r:/renaissance/include/renaissance/core/types.h#L220))：

```
StreamKind::TRANS   → 传输流 (H2D, 学习率传输)
StreamKind::COMP_1  → 计算流1 (DEEP_FWD_BWD)
StreamKind::COMP_2  → 计算流2 (备用, MLP不需要)
StreamKind::COMP_3  → 计算流3 (备用)
StreamKind::UPDATE  → 更新流 (ZERO_GRAD, ALLREDUCE, OPTIMIZER)
```

**MLP 各 GraphSlot 分配的 Stream**:

| GraphSlot | StreamKind | 说明 |
|-----------|-----------|------|
| XFER_A/B | TRANS | H2D 异步传输 |
| FWD_BWD_DEEP_A/B | COMP_1 | 所有 FC+ReLU+SoftmaxCE |
| ZERO_GRAD | UPDATE | 梯度清零 |
| FIRST_LAYER_BWD | COMP_1 | FLATTEN BWD |
| DEEP_ALLREDUCE | UPDATE | 深层梯度 AllReduce |
| FIRST_LAYER_ALLREDUCE | UPDATE | 首层梯度 AllReduce |
| WEIGHT_UPDATE | UPDATE | SGD+Momentum 更新 |
| CAST_AND_CHECK | UPDATE | NaN 检测 |

**Multi-stream capture 已有支持** ([compile_capture_simple:L388-L415](file:///r:/renaissance/src/task/task_base.cpp#L388-L415)):

SimpleTask 中 NCCL 图捕获时创建了 COMP_1/2/3 三个流的交叉事件同步。
DeepLearningTask 需要为每个子图单独指定正确的 StreamKind，通过 `Slot.stream_kind` 传递。

---

## 五、N+1 线程架构

参考 MLP.md 的线程模型：

```cpp
// Thread 0~K-1: GPU RANK threads
for (rank = 0; rank < K; ++rank) {
    threads.push_back(std::thread([rank]() {
        cudaSetDevice(device_ids[rank]);
        // wait TransferStation, launch CUDA graphs, sync streams
    }));
}

// Thread K: Preprocessor thread (主线程or单独线程)
prep.train();  // 可阻塞或异步

// Join
for (auto& t : rank_threads) t.join();
prep.stop();
```

**注意**：Preprocessor 的 `train()` 和 `val()` 方法需要确认是阻塞式还是异步启动、如何停止。当前 `Preprocessor::instance()` 是单例，通过 `GlobalRegistry` 访问 `TransferStation`。

---

## 六、风险点

| 风险 | 缓解 |
|------|------|
| **图分解 `build_simple_atlas()` 改动影响 ResNet-50** | MLP 和 ResNet 走不同 shape variant，MLP 只改 variant 0 |
| **TransferStation A/B 标志竞争** | 使用现有 `buffer_is_readable/writable` 原子变量 |
| **cudaGraphLaunch 在非捕获模式失败** | 确保所有 cuDNN 操作在 compile warmup 中已执行 |
| **Preprocessor 数据不匹配** | 阶段1 先做数据对比验证 |
| **NCCL AllReduce 单卡报错** | 空图检查跳过 |
| **last batch shape 不同** | MNIST 60000 samples @128 = 468 full + 1 partial(96)，需确认是否 already handled |

---

## 七、实施顺序总结

```
Day 1: 阶段0 (compile修复) → compile() 正确生成所有 GraphSlot 的 cudaGraphExec_t
Day 2: 阶段1 (A区传输验证) → 数据从 Preprocessor 正确到达 GPU
Day 3: 阶段2 (A/B乒乓+单batch) → run_gpu() 单epoch 完整执行
Day 4-5: 阶段3 (完整训练+验证) → 20 epoch 训练 + 验证准确率达标
```