# AMP 模式缺陷交叉验证与权威修复方案

> 基于 NWX1.md、NWX2.md、NWX3.md 的交叉验证，结合进一步代码审查，对 AMP 训练失败的根因做出最终裁决，并提供可直接执行的修复方案。

---

## 一、三个分析文件的共识与分歧裁决

| 问题 | NWX1 | NWX2 | NWX3 | 裁决 |
|------|------|------|------|------|
| **P0-A: optimizer input_ids 错位** | ✅ 致命 | ✅ 致命 | ✅ 致命 | **共识，确认属实** |
| **P0-B: optimizer 缺失 unscaling** | ✅ 致命 | ✅ 致命 | ✅ 致命 | **共识，确认属实** |
| **P1: SOFTMAX_CE_AMP_INF 乘 scaling** | ✅ 高危 | ✅ 高 | 未提及 | **确认属实** |
| **inv_scaling = 1/batch_size** | ❌ 认为是高危 bug | ✅ 认为"数学正确、命名误导" | ❌ 认为是 bug | **NWX2 正确，非 bug** |
| **g_accum 顺序错位** | ❌ 认为是中危缺陷 | ⚠️ 认为"待验证、看似正确" | 未提及 | **确认属实，且是 train_loss=0 的主因** |
| **check_nan_flag() TODO** | ❌ 认为是高危缺陷 | 未提及 | 未提及 | **确认属实** |
| **insert_cross_op_barrier 不处理 RANGE** | ❌ 认为是中危缺陷 | 未提及 | 未提及 | **确认属实，未来隐患** |
| **TR_AMP_INITIAL_SCALING 当前值** | 65536.0f | 1.1f | 1.1f | **实际为 1.1f（新发现）** |

---

## 二、独立验证的关键发现

### 2.1 `TR_AMP_INITIAL_SCALING` 实际为 `1.1f`（新发现）

**文件**: `include/renaissance/core/global_config.h:70`

```cpp
#define TR_AMP_INITIAL_SCALING  1.1f
```

此前有记录称已改为 `65536.0f`，但代码实际为 `1.1f`。`1.1f` 是一个非标准的保守值，不足以体现 AMP 的梯度精度提升优势。修复所有 bug 后，应逐步测试 `scaling = 4 → 16 → 64 → 128 → 256 → 512 → 1024 → 65536`，找到当前模型/数据/批次下的最大安全值。

### 2.2 `inv_scaling = 1.0f / batch_size` 不是 bug（关键裁决）

**代码位置**: `softmax_ce_op.cu:141`

```cpp
*inv_scaling = 1.0f / static_cast<float>(*batch_size_ptr);
```

**backward 使用处**: `softmax_ce_op.cu:329`

```cpp
float scale = (*scaling_ptr) * (*inv_scaling_ptr);
// = scaling * (1/batch_size) = scaling / batch_size
```

**数学推导**:
- 标准 SoftmaxCE 的 gradient = `(softmax - one_hot) / batch_size`
- AMP 下 forward 的 `loss = mean_CE * scaling`
- 通过链式法则，backward 的 gradient = `(softmax - one_hot) * scaling / batch_size`
- 当前代码 `scale = scaling / batch_size` **恰好等于此公式**

**结论**: `inv_scaling` 变量名应改为 `inv_batch_samples` 以避免误导，但**计算逻辑完全正确，不应修改**。若按 NWX1/NWX3 的建议改为 `1.0f / scaling`，则 backward 的 `scale = 1.0f`，梯度不再被 scaled，AMP 的 loss scaling 机制将完全失效。

### 2.3 `g_accum` 顺序错位是 `train_loss=0` 的确定根因（关键裁决）

**代码位置**: `deep_learning_task.cpp:1197-1216`

```cpp
if (g_accum) cudaGraphLaunch(g_accum, s_up);   // ← 在 g_deep 之前！
sync_up();
...
cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (g_deep) cudaGraphLaunch(g_deep, s_c1);
sync_comp();
```

**执行时序分析**（以 batches=300 为例）:

| Batch | g_accum 读取的 loss | g_deep 计算的 loss | 说明 |
|-------|---------------------|-------------------|------|
| 0 | 0（初始值） | batch 0 的 loss | batch 0 的 loss 未被累加 |
| 1 | **0 或 batch 0**（race） | batch 1 的 loss | `g_accum`(s_up) 与 `cudaMemsetAsync`(s_c1) 跨 stream race |
| 2 | **0 或 batch 1**（race） | batch 2 的 loss | 同上 |
| ... | ... | ... | ... |
| 299 | **0 或 batch 298**（race） | batch 299 的 loss | batch 299 的 loss 未被累加 |

**为什么 NWX2 的"看似正确"判断是错的**:
- NWX2 误以为 `g_fwd`（first layer forward）已经计算了 loss。实际上 `g_fwd` 只包含 Flatten+FC1+Tanh1，`g_deep` 才包含 FC2+Tanh2+FC3+SoftmaxCE。
- `g_accum` 在 `g_deep` 之前，意味着它永远累加的是**上一个 batch** 的 loss，且**最后一个 batch 的 loss 永远丢失**。
- 更致命的是，`g_accum` 在 `UPDATE` stream 上，`cudaMemsetAsync(loss, 0)` 在 `COMP_1` stream 上。两者之间没有 `cudaStreamWaitEvent`。GPU 调度器可能先执行 `cudaMemsetAsync`，导致 `g_accum` 读到 0。这是 `train_loss=0` 的精确解释。

### 2.4 `check_nan_flag()` TODO 意味着 GradScaler 动态保护未闭合

**代码位置**: `graph_executor.cpp:228-230`

```cpp
bool GraphExecutor::check_nan_flag() const {
    return false;  // TODO
}
```

虽然 `test_dl_full` 使用 `DeepLearningTask` 手动 loop 而非 `GraphExecutor`，但 `GraphExecutor` 是框架设计的标准执行路径。TODO 存根意味着：
- 任何迁移到 `GraphExecutor` 的训练代码都会**在梯度溢出时继续更新权重**（本该跳过）。
- `RANGE_GRAD_SCALING` 的折半机制在 `GraphExecutor` 路径下**永远不会被触发**。

### 2.5 `insert_cross_op_barrier` 不处理 RANGE nodes 是结构性隐患

**代码位置**: `capture_multi_stream.cpp:76`

```cpp
if (next_node.kind == GraphNode::Kind::COMPUTE) {  // ❌ 只处理 COMPUTE
```

当前 `g_accum` 是单独 launch 的，不在 Captured Graph 内部，所以此 bug 不直接影响 `test_dl_full`。但如果未来将 `ACCUM_METRICS` 嵌入 `DEEP_FWD_BWD` graph（这是合理的优化），`RANGE_ACCUM_METRICS`（UPDATE stream）将在 `SOFTMAX_CE_AMP_FWD`（COMP_1 stream）完成前执行，导致 race。

---

## 三、完整问题清单（经交叉验证后的最终版）

### P0-1: Optimizer `input_ids` 索引错位 [致命]
- **文件**: `compiler.cpp:1584-1597` (Weight), `1647-1658` (Bias)
- **影响**: Momentum/Adam/AdamW 读取到错误的超参数（beta=scaling, b1=scaling, b2=beta, eps=beta2）
- **触发条件**: 所有非 SGD 优化器，只要 `scaling ≠ 1` 就炸；`scaling=1` 时参数错位被掩盖但仍在错误超参数下运行

### P0-2: Optimizer kernel 缺失 gradient unscaling [致命]
- **文件**: `optimizer_op.cu:42-138`（4 个非 SGD kernel）
- **影响**: 梯度未还原，有效学习率放大 `scaling` 倍
- **触发条件**: 所有非 SGD 优化器，AMP 模式下必然触发

### P0-3: `g_accum` 执行顺序错位 + 跨 stream race [致命]
- **文件**: `deep_learning_task.cpp:1197` (统一循环), `1120` (batches==1)
- **影响**: `train_loss=0`，最后一个 batch 的 loss 永远不被累加；`g_accum`(UPDATE) 与 `cudaMemsetAsync(loss=0)`(COMP_1) 跨 stream race
- **触发条件**: 所有训练模式（FP32/AMP 均存在，只是 FP32 下 batches 多误差小）

### P1-1: `SOFTMAX_CE_AMP_INF` 在 val 时乘 scaling [高危]
- **文件**: `softmax_ce_op.cu:242`
- **影响**: val_loss 被放大 `scaling` 倍
- **触发条件**: `--amp` 模式下每次 validation

### P1-2: `GraphExecutor::check_nan_flag()` 是 TODO 存根 [高危]
- **文件**: `graph_executor.cpp:228-230`
- **影响**: GradScaler 动态保护机制在标准执行路径下完全失效
- **触发条件**: 使用 `GraphExecutor` 的任何训练代码

### P2-1: `insert_cross_op_barrier` 不处理 RANGE nodes [中危]
- **文件**: `capture_multi_stream.cpp:76`
- **影响**: Captured Graph 内 RANGE node 与上游 COMPUTE node 可能跨 stream race
- **触发条件**: RANGE node 紧跟 COMPUTE node 且两者在不同 stream 时

### P2-2: `TR_AMP_INITIAL_SCALING = 1.1f` 非标准 [低/建议]
- **文件**: `global_config.h:70`
- **影响**: 无法充分发挥 AMP 的梯度精度提升优势
- **建议**: 修复上述 bug 后，逐步提升至 65536.0f（PyTorch GradScaler 默认值）

---

## 四、科学修复方案

### 4.1 设计原则（基于 NVIDIA AMP / PyTorch GradScaler 标准）

```
Forward:  loss_scaled = loss_true * scaling
          （仅训练时；推理/验证时不乘）

Backward: gradient = ∂loss_true/∂W * scaling
          （通过链式法则自动传播，backward kernel 无需额外处理）
          
Optimizer: g_unscaled = g_scaled / scaling
           （所有优化器必须在更新前做 unscaling）
           
NaN Check: 若梯度有 NaN/Inf → has_nan=1
           → 跳过 optimizer update
           → scaling *= 0.5
           → 下一 batch 继续训练
```

### 4.2 修复步骤

#### Step 1: 统一 Optimizer `input_ids` 顺序（根治 P0-1）

**修改 `src/graph/compiler.cpp`**

统一顺序：`[lr, wd, beta, beta2, eps, scaling, has_nan]`（不存在的参数跳过，但保持相对位置）

```cpp
// Weight 节点（修正后）
node.input_ids.push_back(scalar_ids.lr);       // idx 0
node.input_ids.push_back(scalar_ids.wd);       // idx 1
if (weight_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);  // idx 2
}
if (weight_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2); // idx 3
    node.input_ids.push_back(scalar_ids.eps);   // idx 4
}
node.input_ids.push_back(scalar_ids.scaling);   // 倒数第二
node.input_ids.push_back(scalar_ids.has_nan);   // last

// Bias 节点（修正后）
node.input_ids.push_back(scalar_ids.lr);        // idx 0
if (bias_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);   // idx 1
}
if (bias_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2);  // idx 2
    node.input_ids.push_back(scalar_ids.eps);    // idx 3
}
node.input_ids.push_back(scalar_ids.scaling);   // 倒数第二
node.input_ids.push_back(scalar_ids.has_nan);   // last
```

**修改 `src/backend/ops/range/optimizer_op.cpp`**

按统一索引读取：

| Launcher | scaling 索引 |
|----------|-------------|
| Weight SGD | idx 2（不变）|
| Weight Momentum | idx 3 |
| Weight Nesterov | idx 3 |
| Weight Adam | idx 5 |
| Weight AdamW | idx 5 |
| Bias SGD | idx 1（不变）|
| Bias Momentum | idx 2 |
| Bias Nesterov | idx 2 |
| Bias Adam | idx 4 |

#### Step 2: 给所有非 SGD Optimizer Kernel 添加 Unscaling（修复 P0-2）

**修改 `src/backend/ops/range/optimizer_op.cu`**

给 4 个 kernel 添加 `const float* scaling` 参数：

```cpp
// 以 momentum 为例
__global__ void update_momentum_kernel(
    ..., const float* __restrict__ scaling)  // NEW
{
    ...
    float _inv = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    for (...) {
        float g_i = g[i] * _inv;  // ← 添加 unscaling
        m[i] = m[i] * _beta + g_i;
        w[i] = w[i] * (1.0f - _lr * _wd) - _lr * m[i];
    }
}
```

同步修改：
- `update_nesterov_kernel`
- `update_adam_kernel`
- `update_adamw_kernel`
- 8 个 launcher 的 `extern` 声明和调用
- 4 个 CPU 更新函数（`momentum_update_cpu` 等）

#### Step 3: 修复 `g_accum` 执行顺序（修复 P0-3）

**修改 `src/task/deep_learning_task.cpp`**

**统一循环**（batch = 0 .. batches-2）：

```cpp
// === 修改前 ===
if (g_accum) cudaGraphLaunch(g_accum, s_up);
sync_up();
...
cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (g_deep) cudaGraphLaunch(g_deep, s_c1);
sync_comp();

// === 修改后 ===
if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (g_deep) cudaGraphLaunch(g_deep, s_c1);
sync_comp();

if (g_accum) cudaGraphLaunch(g_accum, s_up);
sync_up();
```

**`batches == 1` 特殊分支**：

```cpp
// === 修改前 ===
if (g_accum_train_last) cudaGraphLaunch(g_accum_train_last, s_up);
sync_up();
if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (g_deep_a) cudaGraphLaunch(g_deep_a, s_c1);
sync_comp();

// === 修改后 ===
if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (g_deep_a) cudaGraphLaunch(g_deep_a, s_c1);
sync_comp();

if (g_accum_train_last) cudaGraphLaunch(g_accum_train_last, s_up);
sync_up();
```

**原理**: `g_accum` 必须在 `g_deep` 完成后执行，确保读取的是当前 batch 已计算好的 `loss`。同时将 `cudaMemsetAsync(loss, 0)` 放在 `g_deep` 之前（同 stream），利用 stream 顺序保证 `loss` 在 `g_deep` 前清零、在 `g_accum` 后有值。

#### Step 4: 修复 `SOFTMAX_CE_AMP_INF` 的 Loss 计算（修复 P1-1）

**修改 `src/backend/ops/dtensor/softmax_ce_op.cu`**

```cpp
// 修改前
atomicAdd(loss, sample_loss * inv_batch * scaling);

// 修改后
atomicAdd(loss, sample_loss * inv_batch);
```

同步移除 kernel 中的 `printf` 调试语句。

#### Step 5: 实现 `GraphExecutor::check_nan_flag()`（修复 P1-2）

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

#### Step 6: 修复 `insert_cross_op_barrier` 对 RANGE nodes 的支持（修复 P2-1）

**修改 `src/graph/capture_multi_stream.cpp`**

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
    // ... cudaStreamWaitEvent ...
}
```

#### Step 7: 清理 Debug Prints

移除 `softmax_ce_op.cu` 中的 `[SOFTMAX-FWD]`、`[SOFTMAX-INF]` `printf`；移除 `deep_learning_task.cpp` 中的 `[NAN-CHECK-MAIN]`、`[LOOP-CHECK]`、`[GC-CHECK]`、`[DIAG-STEPS]` 等 noisy 调试输出。

#### Step 8: 调整 `TR_AMP_INITIAL_SCALING`（建议，非紧急）

**修改 `include/renaissance/core/global_config.h`**

```cpp
#define TR_AMP_INITIAL_SCALING  65536.0f
```

**注意**: 此修改必须在 P0-1/P0-2 修复后才能生效，否则 scaling=65536 会立即爆炸。

---

## 五、验证计划

### Phase 1: 基础功能验证
1. `test_dl_full --gpu` → 确认 FP32 基线 ~97.6%
2. `test_dl_full --amp` + `TR_AMP_INITIAL_SCALING=1.0f` → 验证 P0-1/P0-2 修复后 AMP 能跑（scaling=1 时无 unscaling 需求，主要验证索引正确）
3. `test_dl_full --amp` + `TR_AMP_INITIAL_SCALING=4.0f` → 验证 unscaling 生效
4. 观察 train_loss 和 val_loss → 两者应同量级（~2.3），train_loss 不应为 0

### Phase 2: Scaling 阶梯测试
逐步提升 `TR_AMP_INITIAL_SCALING`：
- `4.0f` → `16.0f` → `64.0f` → `256.0f` → `1024.0f` → `65536.0f`
- 每次运行 `test_dl_full --amp`
- 记录准确率、最终 scaling 值（是否被 NaN 检测折半）
- 找到不触发 NaN 的最大 scaling

### Phase 3: NaN 检测验证
- 故意使用极大的 initial scaling（如 `1e6`）
- 观察 `CHECK_NAN` 是否能正确检测 NaN
- 观察 `GRAD_SCALING` 是否能正确折半 scaling
- 观察 optimizer 是否跳过 update（`has_nan=1` 时 kernel 应直接 return）

---

## 六、为什么修复后能工作

修复后的 AMP 完整链路：

```
[Forward]
  FC_AMP_FWD:    FP16 x @ FP16 w → FP16 y (cuDNN FE 1x1 conv)
  SOFTMAX_CE_AMP_FWD:  loss = mean_CE * scaling (scaling=65536)

[Backward]
  SOFTMAX_CE_AMP_BWD:  d_logits = (softmax-one_hot) * scaling / batch_size
  FC_AMP_BWD:          dW = X^T @ dY (FP32 master grad)
                       dX = dY @ W (FP16)

[Grad Cast & Check]
  CAST_FP16_TO_FP32:   conv gradients FP16 → FP32
  CHECK_NAN:           扫描所有 FP32 grads，has_nan = 1 if overflow
  GRAD_SCALING:        if has_nan: scaling *= 0.5

[Optimizer]
  Momentum/Adam:       g_unscaled = g_scaled / scaling
                       update FP32 master weights
                       (has_nan=1 时 kernel 直接 return，跳过 update)

[Cast Back]
  CAST_MAIN_FP32_TO_FP16:  FP32 master w → FP16 working w
```

这条链路与 PyTorch 的 `torch.cuda.amp.GradScaler` 完全等价，是工业界标准实现。

---

**报告日期**: 2026-05-28
**交叉验证来源**: NWX1.md + NWX2.md + NWX3.md + 独立代码审查
**置信度**: 极高 — 所有问题有具体代码位置、数学推导和修复方案
