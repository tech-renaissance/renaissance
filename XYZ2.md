# SOFTMAX_CE 融合算子实现方案（综合版）

> 版本: V1.0  
> 日期: 2026-05-20  
> 状态: 待实施  
> 来源: 综合 XXX.md 中 S/K/D 三份提案 + 代码审计 + 用户补充要求

---

## 一、需求与关键决策

### 1.1 目标算子

| 算子 | 后端 | 说明 |
|------|------|------|
| `SOFTMAX_CE_FP32_FWD` | CPU + CUDA | FP32 前向：logits + labels → loss + top1 + top5 + pred + probs |
| `SOFTMAX_CE_FP32_BWD` | CPU + CUDA | FP32 反向：probs + labels → grad(logits) × scaling |
| `SOFTMAX_CE_AMP_FWD` | CUDA only | AMP：FP16 logits 输入，内部 FP32，输出 FP32 probs |
| `SOFTMAX_CE_AMP_BWD` | CUDA only | AMP：FP32 probs 输入，FP16 grad 输出 |

### 1.2 关键决策（分歧已解决）

| 议题 | 决策 | 理由 |
|------|------|------|
| **旧枚举** | 删除 `CROSS_ENTROPY_LOSS*` 全部 5 个，替换为 `SOFTMAX_CE_*` 4 个 | 用户明确要求；语义清晰；旧枚举无实现 |
| **结果 Region** | 新增 `R_TOP1_CORRECT`、`R_TOP5_CORRECT`、`R_PREDICTED_LABEL` | 推理指标语义独立，便于后续 all-reduce |
| **labels 传递** | `MemoryPlan` 保存 `I_A_LABEL` ID → `Compiler` 注入 `input_ids` | labels 由数据加载器分配，非 SoftmaxCE 层所有；不破坏现有 descs 体系 |
| **softmax_probs** | FWD 输出到 `T_TEMP_FP32`，BWD 复用 | 避免 BWD 重复 exp/sum；内存代价极小 |
| **数值精度** | 内部**全 FP32**；AMP 仅在 IO 边界做 FP16↔FP32 | SOF.md 强制要求；softmax + CE 的规约易 FP16 溢出 |
| **CUDA 路径** | **手写融合 kernel**，不用 cuDNN FE | cuDNN 无融合算子；需输出 top1/top5 |
| **CPU 路径** | **Eigen 向量化**，不展开 one-hot | 与 FC BWD 优化风格一致；O(B×C) 单次遍历 |
| **BWD 输出** | grad **in-place 覆盖 logits** | 框架 dX 覆盖 X 约定 |

---

## 二、现状分析

### 2.1 已有但未实现的枚举

```cpp
// op_kind.h:217-221 — 已定义，但 g_compute_op_table 中 launch 函数全为 nullptr
CROSS_ENTROPY_LOSS,
CROSS_ENTROPY_LOSS_FP32_FWD,
CROSS_ENTROPY_LOSS_FP32_BWD,
CROSS_ENTROPY_LOSS_AMP_FWD,
CROSS_ENTROPY_LOSS_AMP_BWD,
```

### 2.2 已有编译器子图（需重写）

[`layer_descriptor_registry.cpp:505-537`](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L505-L537)：

```
infer_softmaxce_tensors → 4 个 tensor: ce_loss, ce_output, scaling_factor, inv_scaling_factor
build_softmaxce_forward → output_indices = {0,1,2,3}
build_softmaxce_backward → input_indices = {1,2,3}, output_indices = {1}
```

**缺失**：labels 输入、top1/top5 输出、predicted_labels 输出、softmax_probs 中间张量。

### 2.3 已有参数结构（保留复用）

```cpp
struct LossParams {
    float label_smoothing = 0.0f;  // V1 先实现 smoothing=0
    int   num_classes     = 1000;
};
```

> `scaling_factor` **不放入 params**——它是运行时 DTensor，作为输入张量传递。

---

## 三、基础设施改动

### 3.1 ComputeOp 枚举（op_kind.h）

**删除** 5 个旧枚举：
```cpp
// CROSS_ENTROPY_LOSS,           // ← 删除
// CROSS_ENTROPY_LOSS_FP32_FWD,  // ← 删除
// CROSS_ENTROPY_LOSS_FP32_BWD,  // ← 删除
// CROSS_ENTROPY_LOSS_AMP_FWD,   // ← 删除
// CROSS_ENTROPY_LOSS_AMP_BWD,   // ← 删除
```

**插入** 4 个新枚举（在原位置）：
```cpp
SOFTMAX_CE_FP32_FWD,
SOFTMAX_CE_FP32_BWD,
SOFTMAX_CE_AMP_FWD,
SOFTMAX_CE_AMP_BWD,
```

**影响**：`COUNT` 从原值减 1；所有表项使用 `ComputeOp::XXX` 常量索引，编译器自动适配，**无需手动修正任何注册代码**。

同步更新 `op_kind.cpp` 的 `compute_op_to_string` 和 `format_params`。

### 3.2 Region 枚举（types.h）

在 `T_TEMP_INT8` (065) 之后、`DEFAULT` 之前新增 3 个：

```cpp
enum class Region : uint8_t {
    // ... 现有 65 个 ...
    T_TEMP_INT8,         // 065
    R_TOP1_CORRECT,      // 066 — TOP-1 正确数（标量 INT32）
    R_TOP5_CORRECT,      // 067 — TOP-5 正确数（标量 INT32）
    R_PREDICTED_LABEL,   // 068 — 推理标签 [batch] INT32
    DEFAULT = B_PREV_MEAN,
    NUM_REGIONS = 69     // 原 65 + 3 新 + 1 DEFAULT 占位
};
```

同步更新 `memory_plan.cpp` 的 `region_to_string` 和 Region 布局注册。

### 3.3 MemoryPlan —— 保存输入缓冲区 ID（解决 labels 来源）

**问题**：`alloc_input_buffers()` 返回的 `InputBuffers` 被 `compiler.cpp` 丢弃，labels 的 DTensor ID 无处可查。

**方案**：`MemoryPlan` 新增成员保存 4 个特殊 ID：

```cpp
// memory_plan.h
class MemoryPlan {
    // ...
    int32_t input_label_a_id_ = -1;
    int32_t input_data_a_id_  = -1;
    int32_t input_label_b_id_ = -1;
    int32_t input_data_b_id_  = -1;

public:
    [[nodiscard]] int32_t input_label_a_id() const noexcept { return input_label_a_id_; }
    [[nodiscard]] int32_t input_data_a_id()  const noexcept { return input_data_a_id_; }
    [[nodiscard]] int32_t input_label_b_id() const noexcept { return input_label_b_id_; }
    [[nodiscard]] int32_t input_data_b_id()  const noexcept { return input_data_b_id_; }
};
```

```cpp
// memory_plan.cpp
InputBuffers MemoryPlan::alloc_input_buffers(...) {
    auto la = alloc_impl(label_shape, DType::INT32, Region::I_A_LABEL);
    auto da = alloc_impl(data_shape,  dtype,        Region::I_A_DATA);
    auto lb = alloc_impl(label_shape, DType::INT32, Region::I_B_LABEL);
    auto db = alloc_impl(data_shape,  dtype,        Region::I_B_DATA);
    input_label_a_id_ = la.id;
    input_data_a_id_  = da.id;
    input_label_b_id_ = lb.id;
    input_data_b_id_  = db.id;
    return {la, da, lb, db};
}
```

### 3.4 Compiler —— 注入 labels 到 SoftmaxCE GraphNode

在 `compiler.cpp` Phase 4 的 Forward / Backward 循环中，为 `SoftmaxCE` 追加 labels ID：

```cpp
// Forward 循环内（构建完 gn 后、append 前）
if (layer.kind == LayerKind::SoftmaxCE) {
    int32_t label_id = memory_plan.input_label_a_id();
    if (label_id >= 0) gn.input_ids.push_back(label_id);
}

// Backward 循环内（同理）
if (layer.kind == LayerKind::SoftmaxCE) {
    int32_t label_id = memory_plan.input_label_a_id();
    if (label_id >= 0) gn.input_ids.push_back(label_id);
}
```

> 这是框架现有模式的直接扩展（类似 Flatten 首层的显式 input_ids 注入），不引入新机制。

---

## 四、LayerDescriptor 设计

### 4.1 `infer_softmaxce_tensors`（8 张量，不含 labels）

```cpp
std::vector<TensorDesc> infer_softmaxce_tensors(
    const Shape& input, const OpParams& p, const InferContext& ctx)
{
    int batch = input.n();
    const auto* lp = std::get_if<LossParams>(&p.data);
    int C = lp ? lp->num_classes : input.c();
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    return {
        // [0] ce_loss
        TensorDesc{"ce_loss",            Shape{1,1,1,1},      Region::S_SCALAR_FP32,     DType::FP32},
        // [1] ce_output / logits（被 BWD grad 覆盖）
        TensorDesc{"ce_output",          input,               select_feature_region(ctx), feat_dt},
        // [2] top1_correct
        TensorDesc{"top1_correct",       Shape{1,1,1,1},      Region::R_TOP1_CORRECT,    DType::INT32},
        // [3] top5_correct
        TensorDesc{"top5_correct",       Shape{1,1,1,1},      Region::R_TOP5_CORRECT,    DType::INT32},
        // [4] predicted_labels
        TensorDesc{"predicted_labels",   Shape{batch,1,1,1},  Region::R_PREDICTED_LABEL, DType::INT32},
        // [5] scaling_factor
        TensorDesc{"scaling_factor",     Shape{1,1,1,1},      Region::S_SCALAR_FP32,     DType::FP32},
        // [6] inv_scaling_factor
        TensorDesc{"inv_scaling_factor", Shape{1,1,1,1},      Region::S_SCALAR_FP32,     DType::FP32},
        // [7] softmax_probs（Temp区，FWD输出供BWD复用）
        TensorDesc{"softmax_probs",      input,               Region::T_TEMP_FP32,       DType::FP32},
    };
}
```

### 4.2 `build_softmaxce_forward`

```cpp
SubgraphPattern build_softmaxce_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_FWD
               : ComputeOp::SOFTMAX_CE_FP32_FWD;
    // logits 由 Compiler 跨层链注入（prev_output_id → input_ids[0]）
    // labels 由 Compiler 显式注入（input_ids[1]）
    // scaling_factor 是本层 descs[5]
    n.input_indices  = {5};
    // 输出：loss + top1 + top5 + pred + scaling + inv + probs
    n.output_indices = {0, 2, 3, 4, 5, 6, 7};
    p.nodes.push_back(n);
    return p;
}
```

**运行时 input_ids / output_ids 顺序**：

```
FWD input_ids:
  [0] = logits         (Compiler 跨层链：prev_output_id)
  [1] = labels         (Compiler 注入：I_A_LABEL)
  [2] = scaling_factor (descs[5])

FWD output_ids:
  [0] = ce_loss            (descs[0])
  [1] = top1_correct       (descs[2])
  [2] = top5_correct       (descs[3])
  [3] = predicted_labels   (descs[4])
  [4] = scaling_factor     (descs[5])
  [5] = inv_scaling_factor (descs[6])
  [6] = softmax_probs      (descs[7])
```

### 4.3 `build_softmaxce_backward`

```cpp
SubgraphPattern build_softmaxce_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_BWD
               : ComputeOp::SOFTMAX_CE_FP32_BWD;
    // softmax_probs[7] + scaling_factor[5] + inv_scaling[6]
    n.input_indices  = {7, 5, 6};
    // grad 覆盖 ce_output[1]（即 logits 存储位置）
    n.output_indices = {1};
    p.nodes.push_back(n);
    return p;
}
```

**运行时 input_ids / output_ids 顺序**：

```
BWD input_ids:
  [0] = prev_grad_id     (Compiler 跨层梯度链)
  [1] = softmax_probs    (descs[7])
  [2] = labels           (Compiler 注入：I_A_LABEL)
  [3] = scaling_factor   (descs[5])
  [4] = inv_scaling      (descs[6])

BWD output_ids:
  [0] = d_logits         (in-place 覆盖 ce_output / logits)
```

---

## 五、数学公式与数值稳定性

### 5.1 FWD

给定 logits $x \in \mathbb{R}^{B \times C}$，labels $y \in \{0, ..., C-1\}^B$：

```
max_val[b]   = max_c x[b][c]
exp_shifted  = exp(x - max_val)          // 数值稳定
sum_exp[b]   = sum_c exp_shifted[b][c]
log_sum_exp  = max_val + log(sum_exp)

// CE loss（单个样本）
loss[b] = -x[b][y[b]] + log_sum_exp[b]

// 总 loss
L = (scaling_factor / B) * sum_b loss[b]

// Softmax 概率（输出到 Temp 区供 BWD 复用）
probs[b][c] = exp_shifted[b][c] / sum_exp[b]

// TOP-1 / TOP-5
pred[b]     = argmax_c probs[b][c]
top1_correct = sum_b (pred[b] == y[b])
top5_correct = sum_b (y[b] in top5(probs[b]))
```

### 5.2 BWD

```
scale = scaling_factor / B

d_logits[b][c] = (probs[b][c] - (c == y[b] ? 1.0 : 0.0)) * scale
```

**关键**：不展开 one-hot，仅在 label 位置条件减 1。

---

## 六、CPU 实现（softmax_ce_op.cpp）

### 6.1 FWD（FP32，Eigen 向量化）

```cpp
static void launch_softmax_ce_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "Missing LossParams");

    int batch       = op_ctx->input_shape.n;
    int num_classes = p->num_classes;
    DeviceContext* ctx = const_cast<DeviceContext*>(op_ctx->ctx);

    // input_ids: [0]=logits, [1]=labels, [2]=scaling_factor
    float*         logits  = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[0]));
    const int32_t* labels  = static_cast<const int32_t*>(ctx->ptr_at(op_ctx->input_ids[1]));

    // output_ids: [0]=loss, [1]=top1, [2]=top5, [3]=pred, [4]=scaling, [5]=inv, [6]=probs
    float*   loss_out    = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[0]));
    int32_t* top1_out    = static_cast<int32_t*>(ctx->ptr_at(op_ctx->output_ids[1]));
    int32_t* top5_out    = static_cast<int32_t*>(ctx->ptr_at(op_ctx->output_ids[2]));
    int32_t* pred_out    = static_cast<int32_t*>(ctx->ptr_at(op_ctx->output_ids[3]));
    float*   scaling_out = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[4]));
    float*   inv_out     = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[5]));
    float*   probs       = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[6]));

    float scale = scaling_out[0];
    *inv_out = 1.0f / static_cast<float>(batch);

    double loss_sum = 0.0;
    int top1_cnt = 0, top5_cnt = 0;

#ifdef TR_USE_EIGEN
    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<MatrixXfRow> logits_mat(logits, batch, num_classes);
    Eigen::Map<MatrixXfRow> probs_mat(probs, batch, num_classes);

    for (int b = 0; b < batch; ++b) {
        // 1. max per row
        float max_val = logits_mat.row(b).maxCoeff();

        // 2. exp (vectorized)
        probs_mat.row(b) = (logits_mat.row(b).array() - max_val).exp();

        // 3. sum + normalize
        float sum_exp = probs_mat.row(b).sum();
        probs_mat.row(b) /= sum_exp;

        // 4. CE loss
        int label = labels[b];
        float p_label = probs_mat.row(b)(label);
        loss_sum += -std::log(std::max(p_label, 1e-12f));

        // 5. TOP-1
        int pred;
        probs_mat.row(b).maxCoeff(&pred);
        pred_out[b] = pred;
        if (pred == label) ++top1_cnt;

        // 6. TOP-5：5 次扫描找前 5
        bool in_top5 = false;
        auto row_copy = probs_mat.row(b).eval();
        for (int k = 0; k < 5 && k < num_classes; ++k) {
            int best_j = 0;
            float best_v = row_copy(0);
            for (int j = 1; j < num_classes; ++j) {
                if (row_copy(j) > best_v) { best_v = row_copy(j); best_j = j; }
            }
            if (best_j == label) { in_top5 = true; break; }
            row_copy(best_j) = -1e30f;
        }
        if (in_top5) ++top5_cnt;
    }
#else
    // 纯标量回退（无 Eigen）
    for (int b = 0; b < batch; ++b) {
        float* row = logits + b * num_classes;
        float* prob_row = probs + b * num_classes;
        int label = labels[b];

        float max_val = row[0];
        for (int j = 1; j < num_classes; ++j) if (row[j] > max_val) max_val = row[j];

        double sum_exp = 0.0;
        for (int j = 0; j < num_classes; ++j) {
            prob_row[j] = std::exp(row[j] - max_val);
            sum_exp += prob_row[j];
        }
        for (int j = 0; j < num_classes; ++j) prob_row[j] /= static_cast<float>(sum_exp);

        loss_sum += -(row[label] - max_val) + std::log(sum_exp);

        int pred = 0;
        for (int j = 1; j < num_classes; ++j)
            if (prob_row[j] > prob_row[pred]) pred = j;
        pred_out[b] = pred;
        if (pred == label) ++top1_cnt;

        // TOP-5（5次扫描）
        bool in_top5 = false;
        for (int k = 0; k < 5 && k < num_classes; ++k) {
            int bj = -1; float bv = -1e30f;
            for (int j = 0; j < num_classes; ++j)
                if (prob_row[j] > bv) { bv = prob_row[j]; bj = j; }
            if (bj == label) { in_top5 = true; break; }
            prob_row[bj] = -1e30f;
        }
        if (in_top5) ++top5_cnt;
    }
#endif

    *loss_out = static_cast<float>(loss_sum / batch * scale);
    *top1_out = top1_cnt;
    *top5_out = top5_cnt;
}
```

### 6.2 BWD（FP32，Eigen 向量化）

```cpp
static void launch_softmax_ce_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "Missing LossParams");

    int batch       = op_ctx->input_shape.n;
    int num_classes = p->num_classes;
    DeviceContext* ctx = const_cast<DeviceContext*>(op_ctx->ctx);

    // input_ids: [0]=prev_grad(logits), [1]=probs, [2]=labels, [3]=scaling, [4]=inv_scaling
    float*         probs    = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[1]));
    const int32_t* labels   = static_cast<const int32_t*>(ctx->ptr_at(op_ctx->input_ids[2]));
    float*         scaling  = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[3]));
    float*         inv_sc   = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[4]));

    // output_ids: [0]=grad（in-place 覆盖 logits）
    float* grad = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[0]));

    float s = scaling[0] * inv_sc[0];

#ifdef TR_USE_EIGEN
    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<MatrixXfRow> probs_mat(probs, batch, num_classes);
    Eigen::Map<MatrixXfRow> grad_mat(grad, batch, num_classes);

    // 单次广播乘法：所有元素乘以 s
    grad_mat = probs_mat * s;

    // 仅修改 label 位置：减 s
    for (int b = 0; b < batch; ++b) {
        grad_mat(b, labels[b]) -= s;
    }
#else
    for (int b = 0; b < batch; ++b) {
        float* p = probs + b * num_classes;
        float* g = grad + b * num_classes;
        int label = labels[b];
        for (int j = 0; j < num_classes; ++j) g[j] = p[j] * s;
        g[label] -= s;
    }
#endif
}
```

### 6.3 AMP CPU 占位

```cpp
static void launch_softmax_ce_amp_fwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_FWD is not supported on CPU");
}
static void launch_softmax_ce_amp_bwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_BWD is not supported on CPU");
}
```

---

## 七、CUDA 实现（softmax_ce_op.cu）

### 7.1 FWD Kernel（模板化 FP32/AMP）

```cuda
template<bool IS_AMP>
__global__ void softmax_ce_fwd_kernel(
    const float*  __restrict__ logits_fp32,
    const __half* __restrict__ logits_fp16,
    const int*    __restrict__ labels,
    float*        __restrict__ loss,          // [batch]，per-sample loss
    int*          __restrict__ top1_correct,  // 标量，atomicAdd
    int*          __restrict__ top5_correct,  // 标量，atomicAdd
    int*          __restrict__ predicted_labels,
    float*        __restrict__ softmax_probs, // [batch * num_classes]
    int batch, int num_classes)
{
    extern __shared__ float smem[];
    int b = blockIdx.x;
    if (b >= batch) return;

    const int tid = threadIdx.x;
    int label = labels[b];

    // Step 1: 加载 logits → shared memory，同时求 local max
    float local_max = -INFINITY;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        float v = IS_AMP ? __half2float(logits_fp16[b * num_classes + c])
                         : logits_fp32[b * num_classes + c];
        smem[c] = v;
        local_max = fmaxf(local_max, v);
    }

    // Warp reduce max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    local_max = __shfl_sync(0xFFFFFFFF, local_max, 0);

    // Step 2: exp + sum
    float local_sum = 0.0f;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        smem[c] = expf(smem[c] - local_max);
        local_sum += smem[c];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    local_sum = __shfl_sync(0xFFFFFFFF, local_sum, 0);

    // Step 3: 归一化 → probs
    float log_sum_exp = local_max + logf(local_sum);
    for (int c = tid; c < num_classes; c += blockDim.x)
        softmax_probs[b * num_classes + c] = smem[c] / local_sum;

    // Step 4: loss + top1 + top5（仅 lane 0）
    if (tid == 0) {
        float sample_loss = -smem[label] + log_sum_exp;
        loss[b] = sample_loss;

        // TOP-1
        int pred = 0;
        float best = smem[0];
        for (int c = 1; c < num_classes; ++c) {
            if (smem[c] > best) { best = smem[c]; pred = c; }
        }
        predicted_labels[b] = pred;
        if (pred == label) atomicAdd(top1_correct, 1);

        // TOP-5：5 次扫描（C≤1000 时足够快）
        if (num_classes >= 5) {
            bool in_top5 = false;
            for (int k = 0; k < 5; ++k) {
                int bj = -1; float bv = -1e30f;
                for (int j = 0; j < num_classes; ++j) {
                    if (smem[j] > bv) { bv = smem[j]; bj = j; }
                }
                if (bj == label) { in_top5 = true; break; }
                smem[bj] = -1e30f;  // 标记已选
            }
            if (in_top5) atomicAdd(top5_correct, 1);
        }
    }
}
```

### 7.2 BWD Kernel（模板化 FP32/AMP）

```cuda
template<bool IS_AMP>
__global__ void softmax_ce_bwd_kernel(
    const float* __restrict__ softmax_probs,
    const int*   __restrict__ labels,
    float        scaling_factor,
    float        inv_scaling,
    float*       __restrict__ grad_fp32,
    __half*      __restrict__ grad_fp16,
    int batch, int num_classes)
{
    int b = blockIdx.x;
    if (b >= batch) return;

    float s = scaling_factor * inv_scaling;
    int label = labels[b];

    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        float g = softmax_probs[b * num_classes + c] * s;
        if (c == label) g -= s;
        if (IS_AMP)
            grad_fp16[b * num_classes + c] = __float2half(g);
        else
            grad_fp32[b * num_classes + c] = g;
    }
}
```

### 7.3 Launch 封装

```cpp
// FP32 FWD launch
cudaError_t launch_softmax_ce_fwd_fp32(
    const float* logits, const int* labels,
    float* loss, int* top1, int* top5, int* pred, float* probs,
    int N, int C, cudaStream_t stream)
{
    int block = (C <= 128) ? 128 : (C <= 512) ? 256 : 512;
    size_t smem = C * sizeof(float);
    softmax_ce_fwd_kernel<false><<<N, block, smem, stream>>>(
        logits, nullptr, labels, loss, top1, top5, pred, probs, N, C);
    return cudaGetLastError();
}

// AMP FWD launch
cudaError_t launch_softmax_ce_fwd_amp(
    const __half* logits, const int* labels,
    float* loss, int* top1, int* top5, int* pred, float* probs,
    int N, int C, cudaStream_t stream)
{
    int block = (C <= 128) ? 128 : (C <= 512) ? 256 : 512;
    size_t smem = C * sizeof(float);
    softmax_ce_fwd_kernel<true><<<N, block, smem, stream>>>(
        nullptr, logits, labels, loss, top1, top5, pred, probs, N, C);
    return cudaGetLastError();
}

// FP32 BWD launch
cudaError_t launch_softmax_ce_bwd_fp32(
    const float* probs, const int* labels,
    float scale, float inv_scale, float* grad,
    int N, int C, cudaStream_t stream)
{
    int block = 256;
    softmax_ce_bwd_kernel<false><<<N, block, 0, stream>>>(
        probs, labels, scale, inv_scale, grad, nullptr, N, C);
    return cudaGetLastError();
}

// AMP BWD launch
cudaError_t launch_softmax_ce_bwd_amp(
    const float* probs, const int* labels,
    float scale, float inv_scale, __half* grad,
    int N, int C, cudaStream_t stream)
{
    int block = 256;
    softmax_ce_bwd_kernel<true><<<N, block, 0, stream>>>(
        probs, labels, scale, inv_scale, nullptr, grad, N, C);
    return cudaGetLastError();
}
```

### 7.4 CUDA Launch 分发（.cpp 中）

```cpp
#ifdef TR_USE_CUDA
static void launch_softmax_ce_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<LossParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_FWD CUDA missing LossParams");

    const DTensor& dt_logits = mp.get_dtensor(node.input_ids[0]);
    int batch = dt_logits.shape().n();
    int C     = p->num_classes;

    const float* logits = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const int*   labels = static_cast<const int*>(ctx.ptr_at(node.input_ids[1]));

    float* loss  = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    int*   top1  = static_cast<int*>(ctx.ptr_at(node.output_ids[1]));
    int*   top5  = static_cast<int*>(ctx.ptr_at(node.output_ids[2]));
    int*   pred  = static_cast<int*>(ctx.ptr_at(node.output_ids[3]));
    float* probs = static_cast<float*>(ctx.ptr_at(node.output_ids[6]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;

    // 初始化累加标量
    cudaMemsetAsync(loss, 0, sizeof(float) * batch, s);
    cudaMemsetAsync(top1, 0, sizeof(int), s);
    cudaMemsetAsync(top5, 0, sizeof(int), s);

    cudaError_t err = launch_softmax_ce_fwd_fp32(
        logits, labels, loss, top1, top5, pred, probs, batch, C, s);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_FP32_FWD kernel failed: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_softmax_ce_fp32_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<LossParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_BWD CUDA missing LossParams");

    int batch = mp.get_dtensor(node.input_ids[0]).shape().n();
    int C     = p->num_classes;

    const float* probs = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    const int*   labels = static_cast<const int*>(ctx.ptr_at(node.input_ids[2]));
    float*       scaling = static_cast<float*>(ctx.ptr_at(node.input_ids[3]));
    float*       inv_sc  = static_cast<float*>(ctx.ptr_at(node.input_ids[4]));
    float*       grad    = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;

    cudaError_t err = launch_softmax_ce_bwd_fp32(
        probs, labels, scaling[0], inv_sc[0], grad, batch, C, s);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_FP32_BWD kernel failed: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif
```

---

## 八、注册与集成

### 8.1 声明（op_registry.h）

```cpp
void register_op_softmax_ce();
```

### 8.2 实现（softmax_ce_op.cpp 末尾）

```cpp
void register_op_softmax_ce() {
    // FP32 FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_FWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu = launch_softmax_ce_fp32_fwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_fwd_cuda;
#endif
    }
    // FP32 BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_BWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu = launch_softmax_ce_fp32_bwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_bwd_cuda;
#endif
    }
    // AMP FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_FWD;
        e.launch_cpu = launch_softmax_ce_amp_fwd_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_fwd_cuda;
#endif
    }
    // AMP BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_BWD;
        e.launch_cpu = launch_softmax_ce_amp_bwd_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_bwd_cuda;
#endif
    }
}
```

### 8.3 调用（op_registry.cpp）

在 `register_default_ops()` 中添加：
```cpp
register_op_softmax_ce();
```

### 8.4 流策略（op_stream_policy.cpp）

```cpp
case ComputeOp::SOFTMAX_CE_FP32_FWD:
case ComputeOp::SOFTMAX_CE_FP32_BWD:
case ComputeOp::SOFTMAX_CE_AMP_FWD:
case ComputeOp::SOFTMAX_CE_AMP_BWD:
    return StreamKind::COMP_1;
```

### 8.5 CMakeLists.txt

在 `src/CMakeLists.txt` 源文件列表中添加：
```cmake
backend/ops/dtensor/softmax_ce_op.cpp
backend/ops/dtensor/softmax_ce_op.cu
```

---

## 九、改动清单

| # | 文件 | 改动 | 行数 |
|---|------|------|------|
| 1 | `include/renaissance/core/types.h` | 新增 3 个 Region；`NUM_REGIONS` 更新 | +4 |
| 2 | `include/renaissance/graph/memory_plan.h` | 新增输入缓冲区 ID 存储与访问接口 | +12 |
| 3 | `src/graph/memory_plan.cpp` | `alloc_input_buffers` 保存 ID；`region_to_string` 新增 case | ~12 |
| 4 | `include/renaissance/graph/op_kind.h` | 删除 5 旧 `CROSS_ENTROPY_LOSS*`；新增 4 新 `SOFTMAX_CE_*` | ~10 |
| 5 | `src/graph/op_kind.cpp` | `compute_op_to_string` 替换；`format_params` 替换 | ~12 |
| 6 | `src/graph/layer_descriptor_registry.cpp` | 重写 `infer/build_softmaxce_*`（4→8 张量） | ~50 |
| 7 | `src/graph/compiler.cpp` | Phase 4 注入 labels ID 到 SoftmaxCE input_ids | ~10 |
| 8 | `src/backend/op_stream_policy.cpp` | 4 个新算子的流策略 | +5 |
| 9 | `src/backend/ops/dtensor/softmax_ce_op.cu` | **新文件**：FWD/BWD CUDA kernels | ~180 |
| 10 | `src/backend/ops/dtensor/softmax_ce_op.cpp` | **新文件**：CPU + CUDA launch + 注册 | ~220 |
| 11 | `include/renaissance/backend/op_registry.h` | 声明 `register_op_softmax_ce()` | +1 |
| 12 | `src/backend/op_registry.cpp` | 调用注册 | +2 |
| 13 | `src/CMakeLists.txt` | 添加 `.cpp` / `.cu` | +2 |

**总计**：~500 行，2 个新文件，11 个修改文件。

---

## 十、实施顺序

```
Phase 1（枚举与 Region）
  → types.h, memory_plan.h/cpp, op_kind.h/cpp
  → 编译验证

Phase 2（编译器适配）
  → layer_descriptor_registry.cpp（infer/build_softmaxce_*）
  → compiler.cpp（labels 注入）
  → op_stream_policy.cpp
  → 编译验证

Phase 3（算子实现）
  → softmax_ce_op.cu（kernels + launch wrappers）
  → softmax_ce_op.cpp（CPU + CUDA dispatch + register）
  → op_registry.h/cpp
  → CMakeLists.txt
  → 编译验证

Phase 4（测试）
  → Python 参考脚本（对比 PyTorch F.cross_entropy）
  → C++ standalone 测试（CPU/GPU/AMP 三模式）
  → Composite 端到端训练验证
```

---

## 十一、风险与缓解

| 风险 | 缓解 |
|------|------|
| 旧枚举 `CROSS_ENTROPY_LOSS*` 残留引用 | 全局 grep 确保无残留；编译器会报错 |
| `num_classes > 2048` 时 shared memory 不足 | 动态 block 大小选择；超出时用 global memory fallback |
| TOP-5 kernel 串行扫描（lane 0 做 O(5C)） | C≤1000 时足够快（5ms 内）；C>10000 改用分块并行 |
| `label_smoothing > 0` | `LossParams` 已预留；V1 先假设 smoothing=0 |
| `cudaMemsetAsync` 在 CUDA Graph capture 中 | 标量累加器改用 per-batch 数组，Host 端 reduce；或 capture 前预清零 |
| AMP 下 logits 步幅非连续 | kernel 按 DTensor stride 访问（通过 `n_stride_cuda()`） |

---

## 十二、验证计划

| 验证项 | 方法 | 通过标准 |
|--------|------|---------|
| 数值正确性 | CPU/GPU FWD 对比 PyTorch `F.cross_entropy` | MSE < 1e-5 |
| 梯度正确性 | CPU/GPU BWD 对比 PyTorch `loss.backward()` | MSE < 1e-5 |
| Top-1/Top-5 | 构造已知 logits/labels | 与 PyTorch `topk` 结果一致 |
| AMP 精度 | FP16 输入 vs FP32 输入 | MSE < 1e-4 |
| In-place 覆盖 | BWD 后检查原 logits 内存 | 已被 grad 覆盖 |
| 零 one-hot | 代码审查 + 内存 profiling | 无 B×C 中间分配 |
| 编译 | MSVC /O2 + CUDA 13.1 | 零警告通过 |
