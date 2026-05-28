# AMP 模式系统性缺陷分析与修复方案

## 摘要

经对 `src/graph/compiler.cpp`、`src/backend/ops/range/optimizer_op.cpp`/`optimizer_op.cu`、`src/backend/ops/dtensor/softmax_ce_op.cu`、`src/graph/capture_multi_stream.cpp`、`src/task/deep_learning_task.cpp`、`src/backend/graph_executor.cpp` 及 `src/graph/layer_descriptor_registry.cpp` 的完整代码级审查，当前 AMP 实现存在 **2 个致命缺陷、3 个高危缺陷、2 个中危缺陷**。这些缺陷构成了一条完整的 "AMP 必炸链"，解释了为何 `test_dl_full --amp` 要么 NaN、要么 train_loss=0、要么 val_loss 爆炸。

---

## 一、致命缺陷（P0）

### 1.1 Optimizer `input_ids` 索引错位：compiler.cpp vs optimizer_op.cpp

#### 证据

`src/graph/compiler.cpp` 构建非-LARS Weight/Bias 优化器节点时，**无条件**把 `scaling` 插入到固定索引：

```cpp
// Weight (line ~1584)
node.input_ids.push_back(scalar_ids.lr);      // idx 0
node.input_ids.push_back(scalar_ids.wd);      // idx 1
node.input_ids.push_back(scalar_ids.scaling); // idx 2  ← 插入 scaling
if (weight_needs_m) {
    node.input_ids.push_back(scalar_ids.beta); // idx 3
}
if (weight_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2); // idx 4
    node.input_ids.push_back(scalar_ids.eps);   // idx 5
}
node.input_ids.push_back(scalar_ids.has_nan); // last

// Bias (line ~1647)
node.input_ids.push_back(scalar_ids.lr);      // idx 0
node.input_ids.push_back(scalar_ids.scaling); // idx 1  ← 插入 scaling
...
```

但 `src/backend/ops/range/optimizer_op.cpp` 的 launcher 按**硬编码索引**读取标量：

| 优化器 | launcher 读取的标量索引 | compiler 实际放的值（同索引） | 结果 |
|--------|------------------------|------------------------------|------|
| **Weight SGD** | `0=lr, 1=wd, 2=scaling` | `2=scaling` | ✅ 唯一正确 |
| **Weight Momentum** | `0=lr, 1=wd, 2=beta` | `2=scaling` | ❌ **beta = scaling** |
| **Weight Nesterov** | `0=lr, 1=wd, 2=beta` | `2=scaling` | ❌ **beta = scaling** |
| **Weight Adam** | `0=lr, 1=wd, 2=b1, 3=b2, 4=eps` | `2=scaling, 3=beta, 4=beta2` | ❌ **三级错位** |
| **Weight AdamW** | `0=lr, 1=wd, 2=b1, 3=b2, 4=eps` | `2=scaling, 3=beta, 4=beta2` | ❌ **三级错位** |
| **Bias SGD** | `0=lr, 1=scaling` | `1=scaling` | ✅ 正确 |
| **Bias Momentum** | `0=lr, 1=beta` | `1=scaling` | ❌ **beta = scaling** |
| **Bias Nesterov** | `0=lr, 1=beta` | `1=scaling` | ❌ **beta = scaling** |
| **Bias Adam** | `0=lr, 1=b1, 2=b2, 3=eps` | `1=scaling, 2=beta, 3=beta2` | ❌ **三级错位** |

#### 影响

- **AMP 模式下（scaling = 65536）**：
  - Momentum/Nesterov：`m = m * 65536 + g`，动量一 step 就指数爆炸。
  - Adam/AdamW：`b1 = 65536`，一阶矩更新变成 `m = m * 65536 + (1-65536)*g`（负数且巨大）；`b2 = 0.9`（应为 0.999），二阶矩 decay 过快；`eps = 0.999`（应为 1e-8），分母被撑大。训练完全不可收敛。
- **FP32 模式下（scaling = 1.0）**：
  - Momentum：`beta = 1.0`，变成**不衰减的累积动量**（`m = m + g`）。
  - Adam：`b1 = 1.0`，一阶矩完全冻结（`m = m * 1.0 + 0 * g = m`）；`b2 = 0.9`，`eps = 0.999`。Adam 退化为一个错误的伪-RMSprop。
  - **FP32 下 97.61% 的 "正常" 是假象**——Adam 实际在完全错误的超参数组合下运行，只是 `scaling=1.0` 让数值没有爆炸。

---

### 1.2 Optimizer Kernel 缺失 Gradient Unscaling

#### 证据

`src/backend/ops/range/optimizer_op.cu` 中，**只有 SGD kernel 对梯度做了 unscaling**：

```cpp
// SGD kernel
float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
float g_i = g[i] * _inv_scaling;   // ✅

// Momentum kernel
float g_i = g[i];                  // ❌ 直接用了 scaled 梯度

// Nesterov kernel
float g_i = g[i];                  // ❌

// Adam kernel
float g_i = g[i];                  // ❌

// AdamW kernel
float g_i = g[i];                  // ❌
```

对应的 CPU launcher（`optimizer_op.cpp`）同样只有 `sgd_update_cpu` 做了 `inv_scaling`。

#### 影响

AMP 的完整链路：
1. Forward：`loss` 被乘以 `scaling`（65536）。
2. Backward：由于 loss 被 scaled，通过链式法则，**所有梯度天然携带 `scaling` 倍放大**。
3. Optimizer：**必须**在更新前将 `g` 除以 `scaling`（unscaling），否则权重更新步长是正常值的 65536 倍。

SGD 做了 unscaling，所以 SGD 在 AMP 下理论上能工作（如果 input_ids 没错位的话）。  
Momentum/Adam/AdamW **既读错了参数，又没做 unscaling**，双重 bug 叠加，导致 AMP 训练在第一个 step 就发散。

---

## 二、高危缺陷（P1）

### 2.1 `SOFTMAX_CE_AMP_INF` 在推理/验证时错误乘以 scaling

#### 证据

`src/backend/ops/dtensor/softmax_ce_op.cu`，`softmax_ce_inf_kernel` line 242：

```cpp
if (c == label_b) {
    float sample_loss = -logf(prob + 1e-8f);
    float scaling = *scaling_ptr;
    atomicAdd(loss, sample_loss * inv_batch * scaling);  // ❌ 推理不应乘 scaling
}
```

#### 影响

- `val_loss` 被放大了 `scaling` 倍（65536 倍）。即使 `ACCUM_METRICS` 正常工作，`avg_loss` 也会报告 ~150,000 而非 ~2.3。
- **Scaling 的唯一目的是在 backward 时放大梯度以避免 FP16 下溢**。推理/验证不需要 backward，因此 loss 必须是 unscaled 的真实值。

---

### 2.2 `inv_scaling` 被错误赋值为 `1.0f / batch_size`

#### 证据

`softmax_ce_fwd_kernel` line 141 和 `softmax_ce_inf_kernel` line 299：

```cpp
*inv_scaling = 1.0f / static_cast<float>(*batch_size_ptr);  // BUG
```

#### 影响

- `inv_scaling` 的语义是 `1.0f / scaling`，供 backward 计算 `scale = scaling * inv_scaling` 使用。
- 当前实现导致 backward 的 `scale = scaling / batch_size`。这不是 GradScaler 语义。
- 如果修复为 `*inv_scaling = 1.0f / (*scaling_ptr)`，则 backward 的 `scale = 1.0f`，这是正确的——因为 forward 时 loss 已被 scaled，backward 通过 autodiff 自动传播 scaling，不需要在 backward kernel 里额外乘 scale。

---

### 2.3 `GraphExecutor::check_nan_flag()` 是 TODO 存根

#### 证据

`src/backend/graph_executor.cpp` line 228-230：

```cpp
bool GraphExecutor::check_nan_flag() const {
    return false;  // TODO: 当前为存根
}
```

#### 影响

- `GraphExecutor::run_train_step()` 和 `run_train_step_last_batch()` 在 launch `OPTIMIZER` 前调用 `check_nan_flag()`。由于永远返回 `false`：
  - 即使 `RANGE_CHECK_NAN` 检测到了 NaN，`GraphExecutor` 也会**继续执行 optimizer update**。
  - `RANGE_GRAD_SCALING`（动态折半 scaling）在 `GraphExecutor` 路径下**永远不会被触发**。
- 虽然当前 `test_dl_full` 使用 `DeepLearningTask` 的手动 loop（不经过 `GraphExecutor`），但这个 TODO 说明框架的 NaN 检测端到端链路**尚未闭合**。任何迁移到 `GraphExecutor` 的路径都会立即失效。

---

## 三、中危缺陷（P2）

### 3.1 `g_accum` 在 Training Loop 中执行顺序错位

#### 证据

`src/task/deep_learning_task.cpp` training loop（line 1197-1214）：

```cpp
// Batch loop 内部
if (g_accum) cudaGraphLaunch(g_accum, s_up);   // 累加 metrics
sync_up();
...
cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);  // 清零 loss
if (g_deep) cudaGraphLaunch(g_deep, s_c1);     // 计算 loss + backward
sync_comp();
```

`g_accum` 被放在了 `g_deep` **之前**执行。

#### 影响

- Batch 0：`g_accum` 读取的是 `loss` 的初始值（0），累加 0。
- Batch 1：`g_accum` 读取的是 Batch 0 的 loss，累加正确。
- ...
- **最后一个 batch 的 loss 永远不被累加**。
- FP32 下 batches 多（~300），漏一个误差仅 0.3%，看不出来；AMP 下调试时 batches 少，偏差被放大。
- 此外，`g_accum`（`s_up`）和 `cudaMemsetAsync(loss, 0)`（`s_c1`）之间**没有 stream 同步**。虽然 `sync_up()` 等待 `g_accum` 完成，但 `cudaMemsetAsync` 在另一个 stream 上，`g_accum` 可能和上一个 batch 末尾的 `cudaMemsetAsync` race，导致读到 0。这可能是 `train_loss = 0` 的辅助原因。

---

### 3.2 `insert_cross_op_barrier` 不处理 RANGE Nodes

#### 证据

`src/graph/capture_multi_stream.cpp` line 69-85：

```cpp
void insert_cross_op_barrier(...) {
    int out_idx = state.output_stream_idx;
    if (out_idx < 0) return;

    if (next_node.kind == GraphNode::Kind::COMPUTE) {  // ❌ 只处理 COMPUTE
        StreamKind target_sk = get_op_default_stream(next_node.compute_op);
        ...
    }
}
```

#### 影响

- 如果 Captured Graph 内部存在 "COMPUTE node（如 `SOFTMAX_CE_AMP_FWD` 在 `COMP_1`）→ RANGE node（如 `RANGE_ACCUM_METRICS` 在 `UPDATE`）" 的相邻关系，`insert_cross_op_barrier` **不会**插入 `cudaStreamWaitEvent`。
- 这意味着 RANGE node 可能在 COMPUTE node 完成前就读取输出（如 `loss`），导致读到旧值或 0。
- 虽然当前 `g_accum` 在 training loop 中是**单独 launch** 的（不在 `DEEP_FWD_BWD` graph 内部），但如果未来将 `ACCUM_METRICS` 放入 graph 内部，这个 barrier 缺失会成为 race condition 的根源。

---

## 四、正确实现 AMP 的完整方案

### 4.1 设计原则（PyTorch GradScaler 语义）

当前框架的 AMP 架构设计（FP32 master weight + FP16 working weight + `CAST_MAIN_FP32_TO_FP16`）是正确的。修复只需补齐实现层面的 bug：

1. **Forward Loss Scaling**：`SOFTMAX_CE_AMP_FWD` 将 `loss` 乘以 `scaling`。✅ 已有，但 `inv_scaling` 写错。
2. **Backward Gradient Scaling**：由于 loss 被 scaled，backward 自动产生 scaled 梯度。不需要在 backward kernel 里额外乘 scale。✅ 理论正确，但 `inv_scaling` 修复前 `scale = scaling / batch_size` 是错的。
3. **Gradient Unscaling（Optimizer）**：optimizer 在更新前必须将梯度除以 `scaling`。❌ 缺失。
4. **NaN Detection & Dynamic Scaling**：检查梯度是否有 NaN/Inf，若有则跳过 update 并折半 `scaling`。❌ `GraphExecutor` 路径下 TODO 存根。
5. **Metrics Reporting**：train/val loss 必须报告 **unscaled** 真实值。❌ `SOFTMAX_CE_AMP_INF` 错误乘了 scaling。

---

### 4.2 具体修复清单

#### Step 1：统一 Optimizer `input_ids` 顺序（根治错位）

**修改 `src/graph/compiler.cpp`**

将所有 Weight/Bias 优化器节点的 `input_ids` 统一为以下顺序（不存在的参数跳过，但保持相对位置）：

```
[lr, wd, beta, beta2, eps, scaling, has_nan]
```

具体修改：
- **Weight**：`lr` (0), `wd` (1), `beta` (2, 若 needs_m), `beta2` (3, 若 needs_v), `eps` (4, 若 needs_v), `scaling` (倒数第二), `has_nan` (最后)。
- **Bias**：`lr` (0), `beta` (1, 若 needs_m), `beta2` (2, 若 needs_v), `eps` (3, 若 needs_v), `scaling` (倒数第二), `has_nan` (最后)。

**修改 `src/backend/ops/range/optimizer_op.cpp`**

按统一索引读取标量：
- SGD Weight: `0=lr, 1=wd, 2=scaling, last=has_nan`
- Momentum/Nesterov Weight: `0=lr, 1=wd, 2=beta, 3=scaling, last=has_nan`
- Adam/AdamW Weight: `0=lr, 1=wd, 2=b1, 3=b2, 4=eps, 5=scaling, last=has_nan`
- SGD Bias: `0=lr, 1=scaling, last=has_nan`
- Momentum/Nesterov Bias: `0=lr, 1=beta, 2=scaling, last=has_nan`
- Adam Bias: `0=lr, 1=b1, 2=b2, 3=eps, 4=scaling, last=has_nan`

#### Step 2：给所有 Optimizer Kernel 添加 Unscaling

**修改 `src/backend/ops/range/optimizer_op.cu`**

给 `update_momentum_kernel`、`update_nesterov_kernel`、`update_adam_kernel`、`update_adamw_kernel` 各添加 `const float* scaling` 参数：

```cpp
float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
float g_i = g[i] * _inv_scaling;
```

**修改 `src/backend/ops/range/optimizer_op.cpp`**

- 更新所有 launcher 的 `extern` 声明和调用，传入 `scaling` 指针。
- CPU launcher（`momentum_update_cpu`、`nesterov_update_cpu`、`adam_update_cpu`、`adamw_update_cpu`）同步添加 `scaling` 参数并做 unscaling。

#### Step 3：修复 `SOFTMAX_CE_AMP_INF` 的 Loss 计算

**修改 `src/backend/ops/dtensor/softmax_ce_op.cu`**

`softmax_ce_inf_kernel` 中：

```cpp
// 原来
atomicAdd(loss, sample_loss * inv_batch * scaling);
// 改为
atomicAdd(loss, sample_loss * inv_batch);
```

同时移除 kernel 中的 `printf` 调试语句（`[SOFTMAX-INF]`）。

#### Step 4：修复 `inv_scaling` 赋值

**修改 `src/backend/ops/dtensor/softmax_ce_op.cu`**

`softmax_ce_fwd_kernel` 和 `softmax_ce_inf_kernel` 中：

```cpp
// 原来
*inv_scaling = 1.0f / static_cast<float>(*batch_size_ptr);
// 改为
*inv_scaling = 1.0f / (*scaling_ptr);
```

同时移除 `printf` 调试语句（`[SOFTMAX-FWD]`）。

#### Step 5：实现 `GraphExecutor::check_nan_flag()`

**修改 `src/backend/graph_executor.cpp`**

```cpp
bool GraphExecutor::check_nan_flag() const {
    if (optimizer_scalar_ids_.has_nan < 0) return false;
    const int32_t* ptr = static_cast<const int32_t*>(
        ctx_.ptr_at(optimizer_scalar_ids_.has_nan));
    int32_t flag = 0;
#ifdef TR_USE_CUDA
    if (ctx_.is_gpu()) {
        cudaMemcpy(&flag, ptr, sizeof(int32_t), cudaMemcpyDeviceToHost);
    } else
#endif
    {
        flag = *ptr;
    }
    return flag != 0;
}
```

#### Step 6：修复 Training Loop 中 `g_accum` 的执行顺序

**修改 `src/task/deep_learning_task.cpp`**

将 `g_accum` 的 launch 移到 `g_deep` **之后**，紧跟在 `sync_comp()` 后面：

```cpp
// Phase 2: DEEP_FWD_BWD
if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (g_deep) cudaGraphLaunch(g_deep, s_c1);
sync_comp();

// === 累加 metrics（必须在 g_deep 完成后）===
if (g_accum) cudaGraphLaunch(g_accum, s_up);
sync_up();
```

同时处理 `batches == 1` 的特殊分支，确保 `g_accum_train_last` 也在 `g_deep_a` 之后。

#### Step 7：修复 `insert_cross_op_barrier` 对 RANGE Nodes 的支持

**修改 `src/graph/capture_multi_stream.cpp`**

添加 RANGE op 的 stream 推断：

```cpp
void insert_cross_op_barrier(...) {
    int out_idx = state.output_stream_idx;
    if (out_idx < 0) return;

    StreamKind target_sk;
    if (next_node.kind == GraphNode::Kind::COMPUTE) {
        target_sk = get_op_default_stream(next_node.compute_op);
    } else if (next_node.kind == GraphNode::Kind::RANGE) {
        switch (next_node.range_op) {
            case RangeOp::RANGE_ACCUM_METRICS:
            case RangeOp::RANGE_CLEAR:
            case RangeOp::RANGE_CAST_FP32_TO_FP16:
            case RangeOp::RANGE_EMA_PARAM_UPDATE:
                target_sk = StreamKind::UPDATE; break;
            case RangeOp::RANGE_GRAD_SCALING:
            case RangeOp::RANGE_CHECK_NAN:
                target_sk = StreamKind::COMP_1; break;
            default:
                target_sk = StreamKind::COMP_1; break;
        }
    } else {
        return;
    }

    cudaStream_t target_s = static_cast<cudaStream_t>(ctx.stream(target_sk));
    int target_idx = state.find_stream_index(target_s);
    if (target_idx >= 0 && target_idx != out_idx) {
        cudaStreamWaitEvent(target_s, state.streams[out_idx].last_done_event, 0);
    }
}
```

#### Step 8：清理 Debug Prints

移除 `src/backend/ops/dtensor/softmax_ce_op.cu` 中的 `[SOFTMAX-FWD]`、`[SOFTMAX-INF]` `printf`；移除 `src/task/deep_learning_task.cpp` 中的 `[NAN-CHECK-MAIN]`、`[LOOP-CHECK]`、`[GC-CHECK]`、`[DIAG-STEPS]`、`[ACCUM-DIAG]` 等调试输出（保留可选的 `TR_USE_CUDA` 块但清理 noisy printf）。

---

### 4.3 验证计划

| 验证项 | 命令 | 预期结果 |
|--------|------|----------|
| FP32 基线 | `test_dl_full --gpu` | ~97.6% top-1，loss ~2.3 |
| AMP 修复后 | `test_dl_full --amp` | 不 NaN，train_loss ~2.3，val_loss ~2.3，最终 top-1 ≥ 95% |
| Metrics 正确性 | 观察 epoch 0 batch 0/1 | train_loss 不为 0，且随 epoch 下降 |
| Scaling 动态调整 | 故意用极大 initial scaling | 若梯度溢出，`scaling` 应被折半，训练继续而非崩溃 |
| Val Loss 不爆炸 | 任意 epoch | val_loss 与 train_loss 同量级，不乘以 65536 |

---

## 五、总结

当前代码的 AMP 缺陷不是单一 bug，而是一条 **"loss scaling → gradient scaling → parameter misread → missing unscaling → NaN"** 的完整故障链：

1. `SOFTMAX_CE_AMP_FWD` 正确地将 loss 乘以 `scaling`，但 `inv_scaling` 写错，导致 backward 语义偏差。
2. `SOFTMAX_CE_AMP_INF` 在 val 时也乘了 `scaling`，导致 val loss 虚假爆炸。
3. Optimizer `input_ids` 错位，使得 Momentum 的 `beta = scaling`（65536），Adam 的 `b1 = scaling`、`b2 = beta`、`eps = beta2`。
4. Optimizer kernel 缺失 unscaling，使得梯度未还原就参与权重更新。
5. `check_nan_flag()` 是 TODO，GradScaler 的动态保护机制在 `GraphExecutor` 路径下失效。
6. `g_accum` 顺序错位，导致 train metrics 系统性漏统计。

**修复后，AMP 模式将遵循标准 GradScaler 语义**：
- Forward: `loss_scaled = loss_true * scaling`
- Backward: 梯度天然被 scaled（通过 autodiff 传播）
- Optimizer: `g_unscaled = g_scaled / scaling`，然后用正确的超参数更新 FP32 master weight
- Cast: 优化器更新后，FP32 master weight 被 cast 到 FP16 working weight
- NaN Check: 若梯度溢出，自动折半 `scaling` 并跳过本次 update
