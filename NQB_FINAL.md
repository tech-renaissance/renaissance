# NQB_FINAL — AMP 最终裁决与完整修复方案

> 基于六份分析文档（NWX1, NWX2, NWX3, NQB1, NQB2, NQB3）的交叉验证 + AVY_REV 同行评审修正

---

## 一、分析流程与争议裁决

### 1.1 五轮七文档演进

| 轮次 | 文档 | 发现 |
|------|------|------|
| 第一轮 | NWX1 (K) | 发现 P0-A, P0-B, P1-1, P1-2, P2-1, P2-2（最全面） |
| 第一轮 | NWX2 (D) | 发现 P0-A, P0-B, P1-1；正确判定 inv_scaling 非 bug |
| 第一轮 | NWX3 | 发现 P0-A, P0-B；重复 NWX1/NWX2 |
| 第二轮 | NQB1 (我) | 裁决 inv_scaling 非 bug（数学证明）；但**错误判定** g_accum 非 bug |
| 第二轮 | NQB2 | 正确指出 g_accum 是 bug + 跨 stream race；确认 TR_AMP_INITIAL_SCALING=1.1f |
| 第二轮 | NQB3 | 综合评判，正确采纳 inv_scaling 非 bug 裁决 |
| 第三轮 | NQB_FINAL v1 | 最终裁决：采纳 NQB2 的 g_accum 判定，修正 NQB1 错误 |
| **第四轮** | **AVY_REV (同行评审)** | 确认 6/6 核心诊断正确，补充 P1-0 FWD loss scaling + 3 处实施细节修正 |
| **第四轮** | **NQB_FINAL v2** | 融入全部修正：P1-0 新 bug + g_accum 精确位置 + CPU 索引表 + Adam 偏置校正 |

### 1.2 最终裁决矩阵

| 编号 | 问题 | NWX1 | NWX2 | NWX3 | NQB1 | NQB2 | NQB3 | AVY | **最终** |
|------|------|------|------|------|------|------|------|-----|----------|
| P0-A | input_ids 索引错位 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **✓ bug** |
| P0-B | 非SGD kernel 缺失 unscaling | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **✓ bug** |
| P0-C | g_accum 顺序错位 + race | ✅ | — | — | ✗ | ✅ | ⚠️ | ✅ | **✓ bug** |
| P1-0 | **FWD loss 错误乘 scaling**（新） | — | — | — | — | — | — | ✅ | **✓ bug** |
| P1-1 | INF 推理时乘 scaling | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | **✓ bug** |
| P1-2 | check_nan_flag() TODO | ✅ | — | — | — | ✅ | — | ✅ | **✓ bug** |
| P2-1 | cross_op_barrier 仅处理 COMPUTE | ✅ | — | — | — | ✅ | — | ✅ | **✓ bug** |
| P2-2 | Adam/AdamW 缺少偏置校正 | — | — | — | — | — | — | ✅ | **⚠ 已知限制** |
| — | inv_scaling=1/batch_size | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | **✗ 非 bug** |

---

## 二、P0-C：g_accum 顺序错误的彻底分析

### 2.1 NQB1 为什么错了

NQB1 的推理假设 `g_fwd` 包含 loss 计算，因此 `FWD→ACCUM→memset→BWD` 的时序是正确的。

但代码实际结构（`deep_learning_task.cpp:1028-1029`）：

```cpp
auto g_fwd_a = g_tab[S(GraphSlot::FIRST_LAYER_FWD_A)];  // → GraphId::FIRST_LAYER_FWD_A
auto g_deep_a = g_tab[S(GraphSlot::FWD_BWD_DEEP_A)];    // → GraphId::DEEP_FWD_BWD
```

`FIRST_LAYER_FWD_A` 是 pipeline 优化的**第一层前向**（Flatten + FC1 + Tanh1），**不包含** FC2/Tanh2/FC3/SoftmaxCE，因此**不计算 loss**。loss 由 `DEEP_FWD_BWD`（即 `g_deep`）在完整前向过程中计算。

### 2.2 执行时序追踪

**训练循环**（`batches=300` 为例）：

```cpp
// Epoch 开始
g_clear_metrics                     // 清零累计指标

// 统一循环: batch = 0 .. 298
for (int batch = 0; batch < 299; ++batch) {
    g_zg  (s_up)                    // 零梯度
    g_fwd (s_c1)                    // 第一层前向（无 loss）
    sync
    g_accum (s_up)                  // ← 读取 loss（来自上一个 batch 的 g_deep）
    sync
    memset(loss, 0)                 // 清零 loss
    g_deep (s_c1)                   // 完整前向+反向（计算 loss）
    ...
}

// 最后一个 batch: batch = 299
    g_zg → g_fwd_l → sync → g_accum_now → sync → memset(loss,0) → g_deep_l → sync
```

逐 batch 追踪：

| 时刻 | g_accum 实际读取的 loss | g_deep 写入的 loss | 说明 |
|------|------------------------|-------------------|------|
| Batch 0 | **初始值 0** | batch 0 loss | batch 0 的 loss 未被累加 |
| Batch 1 | **batch 0 loss** | batch 1 loss | 延时一个 batch |
| Batch 2 | batch 1 loss | batch 2 loss | |
| ... | ... | ... | |
| Batch 298 | batch 297 loss | batch 298 loss | |
| Last batch | **batch 298 loss** | **batch 299 loss** | **batch 299 的 loss 永不累加** |

### 2.3 与验证循环对比

**验证循环**（`deep_learning_task.cpp:1718-1824`）是正确的：

```cpp
memset(loss, 0)
xfer → g_inf (计算 loss) → sync
g_accum (读取 loss) → sync        // ← 正确：先计算后累加
```

训练循环的 g_accum 应与验证循环保持一致：先 `g_deep`（计算 loss），再 `g_accum`（累加 loss）。

### 2.4 跨 stream race 风险

`g_accum` 在 `UPDATE` stream (`s_up`)，`memset(loss, 0)` 在 `COMP_1` stream (`s_c1`)。虽然在 `sync_up()` 和 `sync_comp()` 之间有同步，但若未来修改了这两组 kernel 的相对位置或新增了跨 stream 依赖，可能导致 `g_accum` 在 `memset` 之后但 `g_deep` 之前执行读到 0。修复后 g_accum 移到 g_deep 之后，消除此隐患。

### 2.5 batch=1 路径的致命后果

```cpp
// batches == 1 (deep_learning_task.cpp:1100-1163)
g_clear_metrics → g_zg → g_fwd_a → sync
g_accum_train_last (s_up) → sync       // ← 读的是初始值 0！
memset(loss,0) → g_deep_a (s_c1)        // ← 唯一 batch 的 loss 刚被计算但不会被累加
...
return;                                   // ← 直接返回，loss 再也无法累加
```

**唯一 batch 的 loss 永远丢失**，train_loss 恒为 0。

---

## 三、P1-0：FWD loss 错误乘 scaling 的深入分析（AVY_REV 新发现）

### 3.1 问题代码

**`softmax_ce_op.cu:136`**：

```cpp
atomicAdd(loss, sample_loss * inv_batch * scaling);
```

在标准 AMP（PyTorch GradScaler）语义中：
- Forward 应计算未缩放的 loss：`loss = CE_mean`
- Backward 使用 scaled gradient：`gradient = ∂loss/∂W × scaling`
- Metrics 显示未缩放的 loss

当前把 `scaling` 乘进了 loss 标量，导致：
- `train_loss = CE_mean × scaling`（被虚高）
- 修复 P1-1 后 `val_loss = CE_mean`（正常）
- 两者相差 scaling 倍（若 scaling=65536，train_loss 是 val_loss 的 65536 倍）

### 3.2 为什么移除 FWD scaling 不影响 backward

BWD kernel 独立读取 scaling（[`softmax_ce_op.cu:329`](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu#L329)）：

```cuda
__global__ void softmax_ce_bwd_kernel(
    const float* probs, const int* labels,
    const float* scaling_ptr,        // ← 独立读取，不依赖 FWD loss 值
    const float* inv_scaling_ptr,
    void* grad_ptr, ...)
{
    float scale = (*scaling_ptr) * (*inv_scaling_ptr);  // = scaling / batch_size
    float g = prob;
    if (c == label) g -= 1.0f;
    g *= scale;  // = (prob - one_hot) × scaling / N
}
```

梯度缩放因子 `scale = scaling / batch_size` 是 BWD kernel 从 `scaling_ptr` 独立计算的，**完全不依赖 FWD 写入的 loss 值**。FWD loss 乘 scaling 唯一的作用是把 train_loss 虚高 scaling 倍。

### 3.3 结论

**FWD 核验必须移除 `* scaling`**。这同时修复 P1-0（train_loss 虚高）和 P1-1（val_loss 虚高），使两者恢复一致。

---

## 四、完整问题清单

### P0（致命 — 阻塞 AMP 训练）

| 编号 | 问题 | 文件:行号 | 表象 |
|------|------|----------|------|
| P0-A | compiler input_ids 索引错位 | compiler.cpp:1584-1597, 1647-1658 | beta/scaling 参数窜位 |
| P0-B | Momentum/Nesterov/Adam/AdamW 缺失 unscaling | optimizer_op.cu:42-158, optimizer_op.cpp | 梯度被放大 scaling 倍未恢复 |
| P0-C | g_accum 在 g_deep 之前执行 | deep_learning_task.cpp:1197, 1120, 1299-1301 | train_loss=0，最后一个 batch 漏统 |

### P1（高危）

| 编号 | 问题 | 文件:行号 | 影响 |
|------|------|----------|------|
| P1-0 | **SOFTMAX_CE_AMP_FWD 错误乘 scaling** | softmax_ce_op.cu:136 | train_loss 虚高 scaling 倍（AVY_REV 新增） |
| P1-1 | SOFTMAX_CE_AMP_INF 错误乘 scaling | softmax_ce_op.cu:242 | val_loss 虚高 scaling 倍 |
| P1-2 | GraphExecutor::check_nan_flag() TODO 存根 | graph_executor.cpp:228-230 | 标准路径 NaN 保护失效 |

### P2（中危 + 已知限制）

| 编号 | 问题 | 文件:行号 | 影响 |
|------|------|----------|------|
| P2-1 | insert_cross_op_barrier 仅处理 COMPUTE | capture_multi_stream.cpp:76 | 含 RANGE 的复合图可能跨 stream race |
| P2-2 | Adam/AdamW 缺少偏置校正（bias correction） | optimizer_op.cu adam/adamw kernel | 初始几步有效 lr 异常大（非 AMP 特有） |

### 明确排除

| 标记 | 问题 | 排除原因 |
|------|------|----------|
| — | inv_scaling = 1/batch_size | 数学证明正确。BWD 中 `scale = scaling/batch_size` 是 CE 梯度的标准形式。仅命名误导 |

---

## 五、修复方案

### 5.1 设计原则

AMP 标准链路（与 PyTorch `torch.cuda.amp.GradScaler` 等价）：

```
FWD (train):        loss = CE_mean                       ← 不乘 scaling
BWD:                gradient = (prob-one_hot) × scaling / N
FWD (val/infer):    loss = CE_mean                       ← 不乘 scaling
OPT:                g_unscaled = g / scaling → 更新权重
CHECK_NAN:          扫描梯度 → has_nan flag
GRAD_SCALING:       has_nan=1 → scaling /= 2; epoch_end → scaling = max(scaling×2, init)
```

关键设计决策：**BWD kernel 独立读取 `scaling_ptr` 计算梯度缩放因子**，不依赖 FWD 的 loss 值。FWD loss 只需写 CE_mean 用于正确的 metrics 上报。

### 5.2 Step 0: 移除 FWD loss 的 scaling 乘法（P1-0）【新增】

**文件**: `src/backend/ops/dtensor/softmax_ce_op.cu:136`

```cpp
// 修改前：
atomicAdd(loss, sample_loss * inv_batch * scaling);

// 修改后：
atomicAdd(loss, sample_loss * inv_batch);
```

同步移除 `printf("[SOFTMAX-FWD]...")` 调试输出（L142）。

### 5.3 Step 1: 修正 compiler.cpp 的 optimizer input_ids 顺序（P0-A）

**文件**: `src/graph/compiler.cpp`

统一顺序：`[lr, wd, beta*, beta2*, eps*, scaling, has_nan]`，`*` 表示按需插入。

```cpp
// ========== Weight 优化器节点 ==========
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

// ========== Bias 优化器节点 ==========
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

### 5.4 Step 2: 为全部非 SGD kernel 添加 unscaling（P0-B）

#### 5.4.1 CUDA kernel 修改

**文件**: `src/backend/ops/range/optimizer_op.cu`

修改 4 个 kernel（momentum, nesterov, adam, adamw），每个添加：

```cuda
// 新增参数
const float* scaling,

// kernel 开头添加（参考 SGD kernel 的实现）
float _inv = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;

// 所有 g[i] 使用处添加 unscaling
float g_i = g[i] * _inv;
```

同步更新 7 个非 SGD launcher 函数（共 9 个中的 7 个：momentum/nesterov/adam/adamw weight 4 个 + momentum/nesterov/adam bias 3 个；SGD weight/bias 已有 scaling，无需修改）的 extern 声明和签名。

#### 5.4.2 CUDA launcher 的 scalar_ptr 索引修正

**文件**: `src/backend/ops/range/optimizer_op.cpp`

| CUDA Launcher | scaling 索引 | has_nan 来源 | 附注 |
|---|---|---|---|
| Weight SGD | `scalar_ptr<2>` | `input_ids.back()` | 不变 |
| Weight Momentum | `scalar_ptr<3>` | `input_ids.back()` | 原 `scalar_ptr<2>` 是 beta，需下移 |
| Weight Nesterov | `scalar_ptr<3>` | `input_ids.back()` | 同上 |
| Weight Adam | `scalar_ptr<5>` | `input_ids.back()` | 原无 scaling 读取，需新增 |
| Weight AdamW | `scalar_ptr<5>` | `input_ids.back()` | 同上 |
| Bias SGD | `scalar_ptr<1>` | `input_ids.back()` | 不变 |
| Bias Momentum | `scalar_ptr<2>` | `input_ids.back()` | 原 `scalar_ptr<1>` 是 beta，需下移 |
| Bias Nesterov | `scalar_ptr<2>` | `input_ids.back()` | 同上 |
| Bias Adam | `scalar_ptr<4>` | `input_ids.back()` | 原无 scaling 读取，需新增 |

#### 5.4.3 CPU 更新函数修正

**文件**: `src/backend/ops/range/optimizer_op.cpp`

为 4 个非 SGD CPU 更新函数添加 `float scaling` 参数和 unscaling：

```cpp
// momentum_update_cpu — 修改前（L290-295）：
static void momentum_update_cpu(float* w, const float* g, float* m,
    size_t n, float lr, float wd, float beta) {
    for (size_t i = 0; i < n; ++i) {
        m[i] = m[i] * beta + g[i];
        w[i] = w[i] * (1.0f - lr * wd) - lr * m[i];
    }
}

// momentum_update_cpu — 修改后：
static void momentum_update_cpu(float* w, const float* g, float* m,
    size_t n, float lr, float wd, float beta, float scaling) {
    float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        float g_i = g[i] * inv_scaling;
        m[i] = m[i] * beta + g_i;
        w[i] = w[i] * (1.0f - lr * wd) - lr * m[i];
    }
}
```

同样修改 `nesterov_update_cpu`, `adam_update_cpu`, `adamw_update_cpu`。

#### 5.4.4 CPU launcher 索引与 num_inputs 阈值修正

| CPU Launcher | 修正后 scalar 读取 | num_inputs 阈值 | update 调用（新增 scaling） |
|---|---|---|---|
| Weight SGD | lr=0, wd=1, scaling=2 | ≥3（不变） | `sgd_update_cpu(wp,gp,N,lr,wd,scaling)` |
| Weight Momentum | lr=0, wd=1, beta=2, scaling=3 | ≥5 | `momentum_update_cpu(wp,gp,mp,N,lr,wd,beta,scaling)` |
| Weight Nesterov | lr=0, wd=1, beta=2, scaling=3 | ≥5 | `nesterov_update_cpu(wp,gp,mp,N,lr,wd,beta,scaling)` |
| Weight Adam | lr=0, wd=1, b1=2, b2=3, eps=4, scaling=5 | ≥7 | `adam_update_cpu(wp,gp,mp,vp,N,lr,wd,b1,b2,eps,scaling)` |
| Weight AdamW | lr=0, wd=1, b1=2, b2=3, eps=4, scaling=5 | ≥7 | `adamw_update_cpu(wp,gp,mp,vp,N,lr,wd,b1,b2,eps,scaling)` |
| Bias SGD | lr=0, scaling=1 | ≥2（不变） | `sgd_update_cpu(wp,gp,N,lr,0.0f,scaling)` |
| Bias Momentum | lr=0, beta=1, scaling=2 | ≥4 | `momentum_update_cpu(wp,gp,mp,N,lr,0.0f,beta,scaling)` |
| Bias Nesterov | lr=0, beta=1, scaling=2 | ≥4 | `nesterov_update_cpu(wp,gp,mp,N,lr,0.0f,beta,scaling)` |
| Bias Adam | lr=0, b1=1, b2=2, eps=3, scaling=4 | ≥6 | `adam_update_cpu(wp,gp,mp,vp,N,lr,0.0f,b1,b2,eps,scaling)` |

`num_inputs` 阈值 = 标量数量 + 1（因为 input_ids 末尾还有 has_nan）。

### 5.5 Step 3: 修正 g_accum 执行顺序（P0-C）

**文件**: `src/task/deep_learning_task.cpp`

#### 5.5.1 统一循环修正（batch = 0 .. batches-2）

当前代码结构 [deep_learning_task.cpp:L1169-1286](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1169-L1286)：

```
Phase 1: g_zg ‖ g_fwd       → sync_comp; sync_up
         g_accum             → sync_up            ← ❌ 错误位置
         memset(loss,0)
Phase 2: g_deep ‖ g_xfer_n  → sync_comp; sync_tr
         [debug checks]
Phase 3: g_first ‖ g_dar     → sync_comp; sync_up
         g_gc → ... → sync_up
```

修复后：

```
Phase 1: g_zg ‖ g_fwd       → sync_comp; sync_up
         memset(loss,0)
Phase 2: g_deep ‖ g_xfer_n  → sync_comp; sync_tr
         g_accum             → sync_up            ← ✅ loss 已计算，安全读取
         [debug checks]                          ← g_accum 已完成（sync_up），后续操作无冲突
Phase 3: g_first ‖ g_dar     → sync_comp; sync_up
         g_gc → ... → sync_up
```

精确修改方案（移除 g_accum 从 Phase 1 后，插入到 Phase 2 后）：

```cpp
// ========== 修改内容1：删除 Phase 1 后的 g_accum（L1197-1198）==========
// 修改前：
    sync_comp(); sync_up();
    // ... debug checks ...
    if (g_accum) cudaGraphLaunch(g_accum, s_up);
    sync_up();
    ts->wait_buffer_readable(next_buf);

// 修改后：
    sync_comp(); sync_up();
    // ... debug checks ...（保留）
    // ← g_accum 移走，删除这两行
    ts->wait_buffer_readable(next_buf);

// ========== 修改内容2：在 Phase 2 后插入 g_accum（L1216 之后）==========
// 修改前：
    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
    sync_comp(); sync_tr();

    // [debug checks at L1218-1229]
    // ts->set_buffer_readable/writeable
    // Phase 3: g_first, g_dar ...

// 修改后：
    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
    sync_comp(); sync_tr();

    if (g_accum) cudaGraphLaunch(g_accum, s_up);
    sync_up();                                 // ← 新增：确保 g_accum 完成

    // [debug checks at L1218-1229]            ← 保留原有代码不变
    // ts->set_buffer_readable/writeable
    // Phase 3: g_first, g_dar ...
```

#### 5.5.2 batches == 1 特殊分支修正（L1120-1126）

```cpp
// 修改前：
if (g_accum_train_last) cudaGraphLaunch(g_accum_train_last, s_up);
sync_up();
if (loss_id >= 0)
    cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (g_deep_a) cudaGraphLaunch(g_deep_a, s_c1);
sync_comp();

// 修改后：
if (loss_id >= 0)
    cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (g_deep_a) cudaGraphLaunch(g_deep_a, s_c1);
sync_comp();

if (g_accum_train_last) cudaGraphLaunch(g_accum_train_last, s_up);
sync_up();
```

#### 5.5.3 最后一 batch 特殊分支修正（L1298-1308）

```cpp
// 修改前：
{
    cudaGraphExec_t g_accum_now = g_accum;
    if (g_accum_train_last) g_accum_now = g_accum_train_last;
    if (g_accum_now) cudaGraphLaunch(g_accum_now, s_up);
}
sync_up();
if (loss_id >= 0)
    cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
sync_comp();

// 修改后：
if (loss_id >= 0)
    cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
sync_comp();

{
    cudaGraphExec_t g_accum_now = g_accum;
    if (g_accum_train_last) g_accum_now = g_accum_train_last;
    if (g_accum_now) cudaGraphLaunch(g_accum_now, s_up);
}
sync_up();
```

### 5.6 Step 4: 修复 SOFTMAX_CE_AMP_INF 移除 scaling（P1-1）

**文件**: `src/backend/ops/dtensor/softmax_ce_op.cu:242`

```cpp
// 修改前
atomicAdd(loss, sample_loss * inv_batch * scaling);

// 修改后
atomicAdd(loss, sample_loss * inv_batch);
```

同步移除 `printf("[SOFTMAX-INF]...")` 调试输出（L300-302）。

### 5.7 Step 5: 实现 check_nan_flag()（P1-2）

**文件**: `src/backend/graph_executor.cpp:228-230`

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
    { flag = *ptr; }
    return flag != 0;
}
```

### 5.8 Step 6: 扩展 insert_cross_op_barrier 支持 RANGE nodes（P2-1）

**文件**: `src/graph/capture_multi_stream.cpp:69-85`

```cpp
void insert_cross_op_barrier(const GraphNode& prev_node,
                              const GraphNode& next_node,
                              MultiStreamCaptureState& state,
                              const DeviceContext& ctx) {
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
        cudaStreamWaitEvent(target_s,
            state.streams[out_idx].last_done_event, 0);
    }
}
```

### 5.9 Step 7: 清理调试输出

移除以下文件中的 debug `printf`：
- `softmax_ce_op.cu`: `[SOFTMAX-FWD]` (L142), `[SOFTMAX-INF]` (L300-302)
- `deep_learning_task.cpp`: `[NAN-CHECK-MAIN]`, `[LOOP-CHECK]`, `[GC-CHECK]`, `[DIAG-STEPS]`, `[ACCUM-DIAG]`, `[TRAIN-CHECK]`
- `check_op.cpp`: `[CHECK-NAN-DEBUG]` (L68-70)

### 5.10 Step 8: 调整初始 scaling（建议）

**文件**: `include/renaissance/core/global_config.h:70`

```cpp
#define TR_AMP_INITIAL_SCALING  4.0f   // 从保守值开始，逐步增大
```

修复所有 bug 后在 Phase 2 验证中逐步增大至 65536。

---

## 六、涉及文件汇总

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `src/backend/ops/dtensor/softmax_ce_op.cu` | FWD 移除 scaling 乘法 + INF 移除 scaling 乘法 + 清理 printf | P0/P1 |
| `src/graph/compiler.cpp` | 修正 optimizer input_ids 顺序 | P0 |
| `src/backend/ops/range/optimizer_op.cu` | 4 kernel + 8 launcher 加 unscaling 参数 | P0 |
| `src/backend/ops/range/optimizer_op.cpp` | CUDA launcher 索引 + CPU 函数 unscaling + CPU 索引/阈值 | P0 |
| `src/task/deep_learning_task.cpp` | 3 处 g_accum 移到 g_deep 之后 | P0 |
| `src/backend/graph_executor.cpp` | 实现 check_nan_flag() | P1 |
| `src/graph/capture_multi_stream.cpp` | insert_cross_op_barrier 支持 RANGE | P2 |

---

## 七、验证计划

### Phase 0: 基础验证（修复 P0 + P1-0/P1-1 后立即执行）

```
# 1. FP32 基线（确认未被破坏）
test_dl_full --gpu
# 预期: 97.6% top-1, loss 正常下降, train_loss ≠ 0

# 2. AMP scaling=1.0（验证索引正确 + unscaling 无副作用）
test_dl_full --amp   (TR_AMP_INITIAL_SCALING=1.0f)
# 预期: 无NaN, loss 正常, 准确率接近 FP32 基线, train_loss ≠ 0

# 3. 确认 train_loss 不再为 0
# 观察输出中的 avg_loss 值 > 0 且随 epoch 下降

# 4. 确认 train_loss 与 val_loss 同量级
# 修复 P1-0/P1-1 后两者应都在 ~2.3→0.1 范围
```

### Phase 1: 梯度缩放验证

```
# 逐步增大 scaling
scaling=2.0  → test_dl_full --amp  → 预期: 无NaN, 准确率 ≥ 95%
scaling=4.0  → test_dl_full --amp  → 预期: 无NaN
scaling=8.0  → test_dl_full --amp  → 预期: 无NaN
scaling=16.0 → test_dl_full --amp  → 预期: 无NaN
scaling=32.0 → test_dl_full --amp  → 预期: 无NaN
scaling=64.0 → test_dl_full --amp  → 预期: 无NaN（如遇 NaN 应观察到 CHECK_NAN 触发并折半 scaling）
```

### Phase 2: 完整系统验证

```
# 1. NaN 保护机制
故意设置 TR_AMP_INITIAL_SCALING=1e7
→ CHECK_NAN 应检测 NaN
→ GRAD_SCALING 应折半 scaling
→ 下一 batch 继续训练（不崩溃）

# 2. 最终性能验证
TR_AMP_INITIAL_SCALING=65536
→ 准确率应 ≥ 97%（FP32 水平）
→ train_loss 正常下降
→ val_loss 与 train_loss 一致
```

---

## 八、已知限制

| 限制 | 说明 | 影响范围 |
|------|------|----------|
| P2-2: Adam/AdamW 缺少偏置校正 | 当前 kernel 无时间步 `t` 参数，`m`/`v` 无 `1/(1-β^t)` 校正。初始几步有效 lr 异常大 | 使用 Adam/AdamW 优化器时 |
| inv_scaling 命名误导 | 实际存储 `1/batch_size` 而非 `1/scaling`，建议重命名为 `inv_batch_samples` | 代码可读性 |
| GraphExecutor 路径 NaN 保护 | P1-2 仅修复了 `check_nan_flag()` 读取，`GraphExecutor::step()` 中是否调用该方法尚需验证 | GraphExecutor 使用者 |

---

## 九、为什么 FP32 能正常工作

FP32 模式下：
- P0-A：`scaling=1.0` → beta 被错读为 1.0（≈0.9 勉强可用）
- P0-B：`scaling=1.0` → unscaling÷1 无影响
- P0-C：g_accum 错位虽然存在，但 300 个 batch 只漏 1 个（~0.3% 误差），FP32 训练曲线本身强烈，看不出来
- P1-0/P1-1：FP32 模式不使用 AMP kernel → 不触发
- 结论：**FP32 的"97.6% 正常"是 bug 被 `scaling=1` 掩盖的假象**

---

## 十、对之前错误分析的修正声明

NQB1 中关于 g_accum 执行顺序的结论**有误**。错误原因：误以为 `g_fwd` 包含完整前向计算（含 loss），实际 `g_fwd` 来自 `GraphSlot::FIRST_LAYER_FWD_A`，只是第一层前向的 pipeline 优化，不包含 loss。loss 由 `g_deep`（`GraphSlot::FWD_BWD_DEEP_A`）在完整前向中计算。NQB2 的交叉验证正确发现并纠正了此错误。

此外，NQB_FINAL v1 遗漏了 P1-0（FWD loss scaling），该问题是 AVY_REV 同行评审发现的。BWD kernel 独立读取 scaling，FWD loss 无需乘 scaling。

---

## 十一、总结

| | 数量 | 详情 |
|------|------|------|
| **P0 致命 bug** | 3 | input_ids 错位 + 缺失 unscaling + g_accum 顺序错误 |
| **P1 高危 bug** | 3 | FWD loss 乘 scaling（新）+ INF loss 乘 scaling + check_nan_flag TODO |
| **P2 中危 bug** | 1 | cross_op_barrier 仅处理 COMPUTE |
| **已知限制** | 2 | Adam 偏置校正缺失 + inv_scaling 命名误导 |
| **明确排除** | 1 | inv_scaling = 1/batch_size 数学正确 |
| **涉及文件** | 7 | softmax_ce_op, compiler, optimizer_op.cu/.cpp, deep_learning_task, graph_executor, capture_multi_stream |

修复后，AMP 模式将实现完整的 GradScaler 语义：loss 无缩放上报 → backward 独立梯度缩放 → optimizer unscaling 恢复 → NaN 检测及折半 → 动态 scaling 调整。与 PyTorch `torch.cuda.amp.GradScaler` 等价。

---

**修订历史**：
- v1 (2026-05-28): 基于 NWX1/NWX2/NWX3/NQB1/NQB2/NQB3 六文档交叉验证的初版
- v2 (2026-05-28): 融入 AVY_REV 同行评审的 4 项修正：P1-0 FWD loss scaling 新 bug、g_accum 精确插入位置、CPU 索引对照表、Adam 偏置校正已知限制
- v3 (2026-05-28): 融入 JSG 评审的 3 处文字修正：CUDA launcher 数量 8→7、g_accum/g_first 串行描述修正、Step 7 清理列表补充 check_op.cpp