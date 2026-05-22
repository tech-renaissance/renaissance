# SOFTMAX_CE 融合算子实现方案（最终综合版）

**版本**: V4.0
**日期**: 2026-05-19
**基于**: SOF.md + SOFX.md（S/K/D 三份提案）+ 代码审计 + 用户补充要求

---

## 一、需求摘要

实现 4 个融合算子：

| 算子 | 说明 |
|------|------|
| `SOFTMAX_CE_FP32_FWD` | FP32 前向：logits + labels → loss + softmax probs + TOP-1/5 + 预测标签 |
| `SOFTMAX_CE_FP32_BWD` | FP32 反向：probs + labels → grad(logits) × scaling_factor |
| `SOFTMAX_CE_AMP_FWD` | AMP 前向：FP16 logits → FP32 内部 → loss + probs(FP32) + TOP-1/5 |
| `SOFTMAX_CE_AMP_BWD` | AMP 反向：probs(FP32) → grad(FP16) × scaling_factor |

**全局约束**：

- CUDA 绝不展开 one-hot（浪费显存和带宽）
- CPU 也尽量不展开
- AMP 内部全部 FP32 运算（softmax + CE 涉及规约，FP16 极易出错）
- 中间有意义的张量用 Temp 区，不用 workspace
- 算子名必须叫 `SOFTMAX_CE_*`，不符合的旧枚举要删除替换

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

### 2.2 已有参数结构

```cpp
// op_kind.h:47-50
struct LossParams {
    float label_smoothing = 0.0f;
    int num_classes = 1000;
};
```

### 2.3 已有编译器子图

[`layer_descriptor_registry.cpp:505-537`](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L505-L537)：

```
forward  →  output_indices = {0, 1, 2, 3}
            → 4 个 tensor: ce_loss, ce_output, scaling_factor, inv_scaling_factor

backward →  input_indices  = {1, 2, 3}
            output_indices = {1}
            → 读取 ce_output + scaling + inv_scaling，grad 覆盖 ce_output
```

**问题**：缺少 labels 输入、TOP-1/5 输出、softmax_probs 中间张量。

### 2.4 参考算子

| 模式 | 参考 |
|------|------|
| CUDA 自定义 kernel（非 cuDNN FE） | [`relu_op.cu`](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cu) |
| CUDA half↔float 转换 | [`gap_op.cu:L54-L56`](file:///r:/renaissance/src/backend/ops/dtensor/gap_op.cu#L54-L56) |
| CPU Eigen 优化 | [`fc_op.cpp:647-711`](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L647-L711) |
| 算子注册 | `register_op_fc()` / `register_op_relu()` |
| 流策略 | [`op_stream_policy.cpp`](file:///r:/renaissance/src/backend/op_stream_policy.cpp) |

---

## 三、核心设计决策

### 3.1 算子枚举：删除旧 + 新增

在 `op_kind.h` 中，**删除** 5 个旧枚举：

```cpp
// 删除 ↓
CROSS_ENTROPY_LOSS,
CROSS_ENTROPY_LOSS_FP32_FWD,
CROSS_ENTROPY_LOSS_FP32_BWD,
CROSS_ENTROPY_LOSS_AMP_FWD,
CROSS_ENTROPY_LOSS_AMP_BWD,
```

在原位置**插入** 4 个新枚举：

```cpp
// 新增 ↓
SOFTMAX_CE_FP32_FWD,
SOFTMAX_CE_FP32_BWD,
SOFTMAX_CE_AMP_FWD,
SOFTMAX_CE_AMP_BWD,
```

**影响分析**：旧枚举 5 个 → 新枚举 4 个，后续所有枚举值向前偏移 1 位。`g_compute_op_table` 使用 `ComputeOp::XXX` 常量索引，编译器自动适配，无需手动修正任何注册代码。`COUNT` 从旧值减 1。

### 3.2 参数结构

`LossParams` **保留**，不新增结构体。但 `scaling_factor` **不放入 params**——它是一个运行时 DTensor（来自 loss scaling），应作为输入张量传递。

```cpp
// LossParams 保持不变
struct LossParams {
    float label_smoothing = 0.0f;  // V1 先实现 smoothing=0, V2 扩展
    int num_classes = 1000;
};
```

### 3.3 新 Region

在 `types.h` 的 `Region` 枚举末尾（`DEFAULT` 之前）新增 3 个 Region：

```cpp
R_TOP1_CORRECT,      // TOP-1 正确数，标量 INT32
R_TOP5_CORRECT,      // TOP-5 正确数，标量 INT32
R_PREDICTED_LABEL,   // 推理结果标签值，[batch] INT32
```

**设计理由**：

- `R_TOP1_CORRECT` 和 `R_TOP5_CORRECT` 后续需要 `RANGE_OP` 做跨 rank 的 sum-all-reduce，放在专用结果区便于集体操作
- `R_PREDICTED_LABEL` 是 [batch] 大小，不需要 all-reduce，但先放在结果区统一管理
- 不放入 `S_SCALAR_INT32` 区：TOP-1/5 是推理指标，语义不属于标量

### 3.4 FWD 输出 softmax_probs 到 Temp 区

**决策**：FWD 将 `softmax_probs` 保存到 Temp 区，BWD 直接读取，不重复计算。

**理由**：

- BWD 需要 softmax_probs 来计算 `grad = (probs - onehot) × scale`
- 如果 BWD 重新计算，等同于再做一次 FWD 的 exp/sum 操作，浪费计算
- 内存代价极小：`batch × num_classes × 4` bytes，对于典型配置（batch=7, num_classes=1000）= 28 KB

**SOF.md 准则**：中间有意义的张量用 Temp 区，不是 workspace。softmax_probs 是一个有明确语义的张量（BWD 的显式输入），符合 Temp 区使用规范。

---

## 四、数学基础

### 4.1 FWD

给定 logits $x \in \mathbb{R}^{B \times C}$，labels $y \in \{0, ..., C-1\}^B$：

1. 数值稳定：$x'_{ij} = x_{ij} - \max_k x_{ik}$
2. Softmax：$p_{ij} = \frac{\exp(x'_{ij})}{\sum_k \exp(x'_{ik})}$
3. CE Loss：$L_i = -\log(p_{i, y_i})$
4. 总 Loss：$L = \frac{\text{scaling\_factor}}{B} \sum_i L_i$
5. TOP-1：$\arg\max_j p_{ij} == y_i$
6. TOP-5：$y_i \in \text{top-5}(p_{ij})$

### 4.2 BWD

$$\frac{\partial L}{\partial x_{ij}} = \frac{\text{scaling\_factor}}{B} \left(p_{ij} - \mathbf{1}[j = y_i]\right)$$

**关键优化**：不展开 B×C 的 one-hot 矩阵，仅在 label 位置减 1：

```
grad[i][c] = probs[i][c] * scale / B
grad[i][label[i]] -= scale / B
```

---

## 五、I/O 张量布局

### 5.1 `infer_softmaxce_tensors` 返回值（严格顺序）

```cpp
std::vector<TensorDesc> infer_softmaxce_tensors(...) {
    // input 是上一层的输出 shape: [batch, 1, 1, num_classes]
    int batch = input.n();
    const auto* lp = std::get_if<LossParams>(&p.data);
    int C = lp ? lp->num_classes : input.c();

    return {
        // [0] labels 输入
        {"labels",             Shape{batch,1,1,1},  Region::I_A_LABEL,          DType::INT32},

        // [1] ce_loss 输出
        {"ce_loss",            Shape{1,1,1,1},      Region::S_SCALAR_FP32,      DType::FP32},

        // [2] top1_correct 输出
        {"top1_correct",       Shape{1,1,1,1},      Region::R_TOP1_CORRECT,     DType::INT32},

        // [3] top5_correct 输出
        {"top5_correct",       Shape{1,1,1,1},      Region::R_TOP5_CORRECT,     DType::INT32},

        // [4] predicted_labels 输出
        {"predicted_labels",   Shape{batch,1,1,1},  Region::R_PREDICTED_LABEL,  DType::INT32},

        // [5] scaling_factor 输出（运行时由 loss scaling 填入）
        {"scaling_factor",     Shape{1,1,1,1},      Region::S_SCALAR_FP32,      DType::FP32},

        // [6] inv_scaling_factor 输出
        {"inv_scaling_factor", Shape{1,1,1,1},      Region::S_SCALAR_FP32,      DType::FP32},

        // [7] softmax_probs 中间张量 (Temp区)
        {"softmax_probs",      input,               Region::T_TEMP_FP32,        DType::FP32},
    };
}
```

### 5.2 子图 `build_softmaxce_forward`

```cpp
SubgraphPattern build_softmaxce_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;

    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_FWD
               : ComputeOp::SOFTMAX_CE_FP32_FWD;
    // input_indices:  labels[0]
    // logits 由编译器作为隐式输入提供（前一层的输出，input_ids[0]）
    n.input_indices  = {0};
    // output_indices: loss[1], top1[2], top5[3], pred[4], scaling[5], inv[6], probs[7]
    n.output_indices = {1, 2, 3, 4, 5, 6, 7};
    p.nodes.push_back(n);
    return p;
}
```

**算子运行时 `input_ids` / `output_ids` 的最终顺序**：

```
FWD input_ids:
  [0] = logits         (编译器从上一层自动连线)
  [1] = labels         (来自 descs[0])

FWD output_ids:
  [0] = ce_loss            (descs[1])
  [1] = top1_correct       (descs[2])
  [2] = top5_correct       (descs[3])
  [3] = predicted_labels   (descs[4])
  [4] = scaling_factor     (descs[5])
  [5] = inv_scaling_factor (descs[6])
  [6] = softmax_probs      (descs[7])
```

### 5.3 子图 `build_softmaxce_backward`

```cpp
SubgraphPattern build_softmaxce_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;

    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_BWD
               : ComputeOp::SOFTMAX_CE_FP32_BWD;
    // input_indices: softmax_probs[7], labels[0], scaling_factor[5], inv_scaling[6]
    n.input_indices  = {7, 0, 5, 6};
    // output: 空 (in-place 覆盖，grad 写到 logits 的梯度槽)
    n.output_indices = {};
    p.nodes.push_back(n);
    return p;
}
```

**算子运行时 `input_ids` / `output_ids` 的最终顺序**：

```
BWD input_ids:
  [0] = softmax_probs    (descs[7])
  [1] = labels           (descs[0])
  [2] = scaling_factor   (descs[5])
  [3] = inv_scaling      (descs[6])
  [4] = logits           (编译器提供，用于获取 shape)

BWD output_ids:
  [0] = dL/d(logits)     (in-place 覆盖 logits 的梯度槽)
```

---

## 六、CPU 实现

### 6.1 FWD（FP32，Eigen 优化）

```cpp
static void launch_softmax_ce_fwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_FWD CPU missing LossParams");

    int batch       = op_ctx->input_shape.n;
    int num_classes = p->num_classes;

    // input_ids: [0]=logits, [1]=labels
    float*         logits  = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[0]));
    const int32_t* labels  = static_cast<const int32_t*>(ctx->ptr_at(op_ctx->input_ids[1]));

    // output_ids: [0]=loss, [1]=top1, [2]=top5, [3]=pred, [4]=scaling, [5]=inv, [6]=probs
    float*   loss_out   = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[0]));
    int32_t* top1_out   = static_cast<int32_t*>(ctx->ptr_at(op_ctx->output_ids[1]));
    int32_t* top5_out   = static_cast<int32_t*>(ctx->ptr_at(op_ctx->output_ids[2]));
    int32_t* pred_out   = static_cast<int32_t*>(ctx->ptr_at(op_ctx->output_ids[3]));
    float*   scaling    = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[4]));
    float*   inv_scaling = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[5]));
    float*   probs      = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[6]));

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<MatrixXfRow> logits_mat(logits, batch, num_classes);
    Eigen::Map<MatrixXfRow> probs_mat(probs, batch, num_classes);

    double loss_sum = 0.0;
    int top1_cnt = 0, top5_cnt = 0;

    for (int b = 0; b < batch; ++b) {
        // 1. max per row
        float max_val = logits_mat.row(b).maxCoeff();

        // 2. exp (vectorized by Eigen)
        probs_mat.row(b) = (logits_mat.row(b).array() - max_val).exp();

        // 3. sum
        float sum_exp = probs_mat.row(b).sum();

        // 4. normalize (in-place)
        probs_mat.row(b) /= sum_exp;

        // 5. CE loss
        int label = labels[b];
        float p_label = probs_mat.row(b)(label);
        loss_sum += -std::log(std::max(p_label, 1e-12f));

        // 6. TOP-1
        int top1_idx;
        probs_mat.row(b).maxCoeff(&top1_idx);
        pred_out[b] = top1_idx;
        if (top1_idx == label) ++top1_cnt;

        // 7. TOP-5: 5 次线性扫描找前 5 大概率
        bool label_in_top5 = false;
        auto row_copy = probs_mat.row(b).eval();
        for (int k = 0; k < 5 && k < num_classes; ++k) {
            int best_idx = 0;
            float best_val = row_copy(0);
            for (int j = 1; j < num_classes; ++j) {
                if (row_copy(j) > best_val) {
                    best_val = row_copy(j);
                    best_idx = j;
                }
            }
            if (best_idx == label) { label_in_top5 = true; break; }
            row_copy(best_idx) = -1e30f;
        }
        if (label_in_top5) ++top5_cnt;
    }

    float scale = scaling[0];
    *loss_out    = static_cast<float>(loss_sum / batch * scale);
    *top1_out    = top1_cnt;
    *top5_out    = top5_cnt;
    *inv_scaling = 1.0f / batch;
}
```

**复杂度**：FWD O(B×C) 遍历 1 次（max + exp 合并为 Eigen 单次遍历），TOP-5 O(B×5C)≈O(B×C)。

### 6.2 BWD（FP32，Eigen 优化）

```cpp
static void launch_softmax_ce_bwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_BWD CPU missing LossParams");

    int batch       = op_ctx->input_shape.n;
    int num_classes = p->num_classes;

    // input_ids: [0]=probs, [1]=labels, [2]=scaling, [3]=inv_scaling, [4]=logits
    float*         probs       = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[0]));
    const int32_t* labels      = static_cast<const int32_t*>(ctx->ptr_at(op_ctx->input_ids[1]));
    float*         scaling     = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[2]));
    float*         inv_scaling = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[3]));

    // output_ids: [0]=grad (in-place on logits)
    float* grad = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[0]));

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<MatrixXfRow> probs_mat(probs, batch, num_classes);
    Eigen::Map<MatrixXfRow> grad_mat(grad, batch, num_classes);

    float s = scaling[0] * inv_scaling[0];

    // 单次广播乘法：所有元素乘以 s
    grad_mat = probs_mat * s;

    // 仅修改 label 位置：减 s
    for (int b = 0; b < batch; ++b) {
        grad_mat(b, labels[b]) -= s;
    }
}
```

**复杂度**：BWD O(B×C) 1 次 Eigen 广播乘法 + O(B) 次减法。one-hot 未展开。

### 6.3 AMP CPU

```cpp
static void launch_softmax_ce_fwd_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_FWD is not supported on CPU");
}
static void launch_softmax_ce_bwd_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_BWD is not supported on CPU");
}
```

---

## 七、CUDA 实现

### 7.1 新增文件

```
src/backend/ops/dtensor/softmax_ce_op.cu   — CUDA kernels
src/backend/ops/dtensor/softmax_ce_op.cpp  — 分发 + 注册
```

`CMakeLists.txt` 中添加配对编译。

### 7.2 FWD Kernel

每个 block 处理一个 batch item，使用 shared memory 做行内 reduction：

```cuda
template<bool IS_AMP>
__global__ void softmax_ce_fwd_kernel(
    const T*      __restrict__ logits,    // float 或 __half
    const int*    __restrict__ labels,
    float*        __restrict__ ce_loss,   // [batch]，后续 CPU reduce
    int*          __restrict__ top1_correct,
    int*          __restrict__ top5_correct,
    int*          __restrict__ predicted_labels,
    float*        __restrict__ softmax_probs, // [batch * num_classes]

    int batch, int num_classes)
{
    extern __shared__ float smem[];       // 需要 num_classes * sizeof(float)
    int b = blockIdx.x;
    if (b >= batch) return;

    const T* logit_row = logits + b * num_classes;
    float*  prob_row  = softmax_probs + b * num_classes;
    int     label     = labels[b];

    // Step 1: 加载 logits → shared memory，同时找最大值
    float max_val = -INFINITY;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        float v = IS_AMP ? __half2float(logit_row[c]) : logit_row[c];
        smem[c] = v;
        max_val = fmaxf(max_val, v);
    }

    // warp-level reduce max
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // Step 2: softmax = exp(x - max) / sum(exp(x - max))
    float sum_exp = 0.0f;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        smem[c] = expf(smem[c] - max_val);
        sum_exp += smem[c];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

    // Step 3: 归一化 → 写入 probs
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        smem[c] /= sum_exp;
        prob_row[c] = smem[c];
    }

    // Step 4: CE loss (仅 warp lane 0)
    float sample_loss = 0.0f;
    int   top1_idx    = 0;
    float top1_prob   = smem[0];
    if (threadIdx.x == 0) {
        float p_label = smem[label];
        sample_loss   = -logf(fmaxf(p_label, 1e-12f));
        ce_loss[b]    = sample_loss;

        // TOP-1: 遍历找 argmax
        for (int c = 1; c < num_classes; ++c) {
            if (smem[c] > top1_prob) {
                top1_prob = smem[c];
                top1_idx  = c;
            }
        }
        predicted_labels[b] = top1_idx;
        if (top1_idx == label) atomicAdd(top1_correct, 1);

        // TOP-5: 5 次扫描找前 5
        // 注意：不能修改 smem（已经被 __syncthreads 保护）
        // 在寄存器中维护 top-5 列表
        bool label_in_top5 = false;
        int top5_idx[5] = {};
        float top5_val[5] = {};
        for (int k = 0; k < 5 && k < num_classes; ++k) {
            int best_j = 0;
            float best_v = smem[0];
            for (int j = 1; j < num_classes; ++j) {
                if (smem[j] > best_v) {
                    bool already = false;
                    for (int p = 0; p < k; ++p) {
                        if (j == top5_idx[p]) { already = true; break; }
                    }
                    if (!already) { best_v = smem[j]; best_j = j; }
                }
            }
            top5_idx[k] = best_j;
            top5_val[k] = best_v;
            if (best_j == label) label_in_top5 = true;
        }
        if (label_in_top5) atomicAdd(top5_correct, 1);
    }
}
```

### 7.3 BWD Kernel

极简单，不需要 shared memory，每个线程直接计算：

```cuda
template<bool IS_AMP>
__global__ void softmax_ce_bwd_kernel(
    const float* __restrict__ softmax_probs, // always FP32
    const int*   __restrict__ labels,
    float scaling_factor,
    float inv_scaling,
    T*           __restrict__ grad_logits,   // float 或 __half
    int batch, int num_classes)
{
    int b = blockIdx.x;
    if (b >= batch) return;

    int label = labels[b];
    float s  = scaling_factor * inv_scaling;

    const float* prob_row = softmax_probs + b * num_classes;
    T*           grad_row = grad_logits + b * num_classes;

    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        float g = prob_row[c] * s;
        if (c == label) g -= s;

        if (IS_AMP)
            grad_row[c] = __float2half(g);
        else
            grad_row[c] = g;
    }
}
```

### 7.4 Launch 封装（`.cu` 外部声明）

```cpp
// FP32
cudaError_t launch_softmax_ce_fwd_fp32(
    const float* logits, const int* labels,
    float* loss, int* top1, int* top5, int* pred,
    float* probs, int N, int C, cudaStream_t s);

cudaError_t launch_softmax_ce_bwd_fp32(
    const float* probs, const int* labels,
    float scaling, float inv_scaling,
    float* grad, int N, int C, cudaStream_t s);

// AMP
cudaError_t launch_softmax_ce_fwd_amp(
    const __half* logits, const int* labels,
    float* loss, int* top1, int* top5, int* pred,
    float* probs, int N, int C, cudaStream_t s);

cudaError_t launch_softmax_ce_bwd_amp(
    const float* probs, const int* labels,
    float scaling, float inv_scaling,
    __half* grad, int N, int C, cudaStream_t s);
```

### 7.5 CUDA Launch 分发（`.cpp` 中）

以 FWD 为例：

```cpp
static void launch_softmax_ce_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<LossParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_FWD CUDA missing LossParams");

    int batch       = mp.get_dtensor(node.input_ids[0]).n();
    int num_classes = p->num_classes;

    const float* logits = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const int*   labels = static_cast<const int*>(ctx.ptr_at(node.input_ids[1]));

    float*  loss   = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    int*    top1   = static_cast<int*>(ctx.ptr_at(node.output_ids[1]));
    int*    top5   = static_cast<int*>(ctx.ptr_at(node.output_ids[2]));
    int*    pred   = static_cast<int*>(ctx.ptr_at(node.output_ids[3]));
    float*  probs  = static_cast<float*>(ctx.ptr_at(node.output_ids[6]));

    cudaStream_t s  = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;

    // 初始化标量输出为零
    cudaMemsetAsync(loss, 0, sizeof(float) * batch, s);
    cudaMemsetAsync(top1, 0, sizeof(int), s);
    cudaMemsetAsync(top5, 0, sizeof(int), s);

    int block_size = 256;
    size_t smem   = num_classes * sizeof(float);

    cudaError_t err = launch_softmax_ce_fwd_fp32(
        logits, labels, loss, top1, top5, pred, probs,
        batch, num_classes, s);

    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_FP32_FWD kernel failed: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

**注意**：FWD kernel 写 per-sample loss 到 `loss[batch]` 数组，需要在 launch 后做 CPU-side reduce（或者在第二个 kernel 中做）。实际实现中，可以在 kernel 内用 `atomicAdd` 到标量 loss，或在 launch 后调用 `cub::DeviceReduce`。

**推荐方案**：kernel 内部已使用 `atomicAdd` 累加 `top1_correct`/`top5_correct`。对于 loss，kernel 写 per-sample 值到临时数组 `loss[batch]`，然后额外 launch 一个 reduction kernel 或调用 cub 做 sum，最后乘以 `scale/batch`。

### 7.6 Stream 策略

在 [`op_stream_policy.cpp`](file:///r:/renaissance/src/backend/op_stream_policy.cpp) 添加：

```cpp
case ComputeOp::SOFTMAX_CE_FP32_FWD:
case ComputeOp::SOFTMAX_CE_FP32_BWD:
case ComputeOp::SOFTMAX_CE_AMP_FWD:
case ComputeOp::SOFTMAX_CE_AMP_BWD:
    return StreamKind::COMP_1;
```

### 7.7 为什么不用 cuDNN Frontend？

- cuDNN 没有 fused Softmax+CrossEntropy 算子
- 拆分为两个 cuDNN FE 节点会失去融合优势，中间 probs 需要从显存回读再写回
- TOP-1/TOP-5/预测标签无法用 cuDNN FE 表达
- 自定义 kernel 可将 max + exp + sum + log + topk 全部融合在一个 kernel 中

---

## 八、注册与集成

### 8.1 注册函数

在 [`op_registry.h`](file:///r:/renaissance/include/renaissance/backend/op_registry.h) 添加声明：

```cpp
void register_op_softmax_ce();
```

### 8.2 注册实现

```cpp
void register_op_softmax_ce() {
    auto& t = g_compute_op_table;

    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_FWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu = launch_softmax_ce_fwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fwd_cuda;
#endif
    }
    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_BWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu = launch_softmax_ce_bwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_bwd_cuda;
#endif
    }
    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_FWD;
        e.launch_cpu = launch_softmax_ce_fwd_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fwd_amp_cuda;
#endif
    }
    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_BWD;
        e.launch_cpu = launch_softmax_ce_bwd_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_bwd_amp_cuda;
#endif
    }
}
```

### 8.3 在 `op_registry.cpp` 中调用

在 `register_all_ops()` 或等价入口函数中添加：

```cpp
register_op_softmax_ce();
```

### 8.4 warmup（CUDA Graph 场景）

由于 FWD kernel 使用动态 shared memory (`num_classes * sizeof(float)`)，建议在 `require_warmup` 中添加 AMP 版本以触发 CUDA Graph 预热：

```cpp
case ComputeOp::SOFTMAX_CE_AMP_FWD:
case ComputeOp::SOFTMAX_CE_AMP_BWD:
    return true;
```

---

## 九、改动清单

| # | 文件 | 改动性质 | 行数 |
|---|------|----------|------|
| 1 | `include/renaissance/graph/op_kind.h` | 删除 5 个 `CROSS_ENTROPY_LOSS_*`，新增 4 个 `SOFTMAX_CE_*` | ~12 |
| 2 | `include/renaissance/core/types.h` | 新增 `R_TOP1_CORRECT`、`R_TOP5_CORRECT`、`R_PREDICTED_LABEL`，更新 `NUM_REGIONS` | +4 |
| 3 | `src/graph/op_kind.cpp` | 更新 `compute_op_to_string`（删除 5 旧 case + 新增 4 新 case） | ~9 |
| 4 | `src/graph/layer_descriptor_registry.cpp` | 重写 `infer_softmaxce_tensors` (4→8)、更新 forward/backward 子图 | ~55 |
| 5 | `src/graph/memory_plan.cpp` | 注册 3 个新 Region 的布局属性 | ~9 |
| 6 | `src/backend/ops/dtensor/softmax_ce_op.cu` | **新文件**：FWD + BWD CUDA kernels（FP32 + AMP） | ~220 |
| 7 | `src/backend/ops/dtensor/softmax_ce_op.cpp` | **新文件**：CPU 实现 + CUDA launch 分发 + 注册 | ~280 |
| 8 | `include/renaissance/backend/op_registry.h` | 声明 `register_op_softmax_ce()` | +1 |
| 9 | `src/backend/op_registry.cpp` | 调用 `register_op_softmax_ce()` + `require_warmup` 添加 | ~3 |
| 10 | `src/backend/op_stream_policy.cpp` | 4 个 CE 算子的流策略 | +4 |
| 11 | `src/CMakeLists.txt` | 添加 `.cu` + `.cpp` 到源文件列表 | +2 |

**总改动量**：约 600 行，2 个新文件 + 9 个修改文件，零接口破坏。

---

## 十、实施步骤

### Phase 1：枚举和 Region 定义（Step 1→2→3→5）

- 删除旧 `CROSS_ENTROPY_LOSS_*`，插入 `SOFTMAX_CE_*`
- 新增 3 个 Region 枚举值
- 更新 `compute_op_to_string` 和 `region_to_string`
- 在 `memory_plan.cpp` 注册新 Region（INT32 类型，无需梯度）

**验证**：编译通过，`g_compute_op_table` 大小正确

### Phase 2：编译器 LayerDescriptor 适配（Step 4）

- 重写 `infer_softmaxce_tensors`（8 个 tensor）
- 重写 `build_softmaxce_forward`（labels 输入 + 7 个输出）
- 重写 `build_softmaxce_backward`（probs + labels + scaling 输入 + in-place 输出）

**验证**：编译通过，图拓扑打印正确

### Phase 3：算子实现（Step 6→7→11）

- 创建 `softmax_ce_op.cu`：
  - FWD kernel（FP32 + AMP，shared memory per-block reduction）
  - BWD kernel（FP32 + AMP，无 shared memory，条件减法）
  - Launch wrapper（`extern "C"` 导出）
- 创建 `softmax_ce_op.cpp`：
  - CPU FWD（Eigen 向量化）
  - CPU BWD（Eigen 向量化）
  - CUDA launch 分发（从 `.cu` 调用 kernel）
  - AMP CPU not_supported
- 更新 `CMakeLists.txt`

**验证**：编译通过，无符号未定义

### Phase 4：注册与流策略（Step 8→9→10）

- 注册 4 个算子到 `g_compute_op_table`
- 添加流策略（COMP_1）
- 调用 `register_op_softmax_ce()`

**验证**：编译通过，`g_compute_op_table` 对应 entry 非 nullptr

### Phase 5：测试验证

5.1 **Python 参考脚本**：

```python
# test_softmax_ce.py — 生成 TSR 格式参考数据
import torch
import torch.nn.functional as F

batch, num_classes = 7, 1000
logits = torch.randn(batch, num_classes) * 2.0
labels = torch.randint(0, num_classes, (batch,))

# FWD
softmax = F.softmax(logits, dim=1)
loss = F.cross_entropy(logits, labels, reduction='mean')
pred = logits.argmax(dim=1)
top1 = (pred == labels).sum().item()
top5 = 0
for i in range(batch):
    _, top5_idx = torch.topk(logits[i], 5)
    top5 += labels[i].item() in top5_idx.tolist()

# BWD
logits.requires_grad_(True)
loss.backward()
grad = logits.grad  # ∂L/∂x

# Save as TSR...
```

5.2 **C++ 正确性测试**（参考 `test_flatten_fc_relu_fc.cpp` 模式）：

- CPU FWD/BWD vs PyTorch（MSE 校验）
- GPU FWD/BWD vs PyTorch
- AMP FWD/BWD vs PyTorch（较高容差，允许 FP16 转换误差）

5.3 **Composite 测试**：

- 构建包含 SoftmaxCE 的端到端模型
- 验证 loss 收敛和 accuracy 正确

---

## 十一、风险与注意事项

| 风险 | 缓解 |
|------|------|
| Shared memory 不足（num_classes=10000 需要 40KB） | 用 `cudaFuncSetAttribute` 指定动态 SMEM；超出硬件限制时 fallback 到 global memory |
| TOP-5 线程 0 串行 O(C²) | 对 C≤1000 足够快（1000×5=5000 次比较），C>10000 可改为 shared memory 分块 partial sort |
| `label_smoothing > 0` | `LossParams` 已预留字段，V1 先实现 smoothing=0，V2 扩展 |
| 数值下溢（log(0)） | 所有 `log` 前加 `max(val, 1e-12f)` |
| CUDA kernel 中 blockIdx.x 为 0 的线程做 loss reduce 的竞态 | 用单独的 reduction kernel 或 cub 做 post-kernel reduction |
| 枚举值重排导致旧代码索引错位 | 已验证：所有表项用 `ComputeOp::XXX` 常量，编译器自动适配 |

---

## 十二、总结

本方案综合了 S/K/D 三份提案的精华，完全满足 SOF.md 和用户补充要求：

1. **算子命名**：`SOFTMAX_CE_*`，旧枚举全部删除替换
2. **专用结果区**：`R_TOP1_CORRECT`、`R_TOP5_CORRECT`、`R_PREDICTED_LABEL`
3. **融合优势**：FWD 保存 `softmax_probs` 到 Temp 区，BWD 零重计算
4. **零 one-hot 展开**：CUDA/CPU 均不构建 B×C 的 one-hot 矩阵
5. **AMP 全 FP32 内部**：仅输入/输出做 FP16↔FP32 转换
6. **CPU Eigen 向量化**：maxCoeff + exp + sum 一次遍历，BWD 广播乘法
7. **CUDA 自定义 kernel**：shared memory per-block reduction，不使用 cuDNN FE
8. **约 600 行新代码**，2 个新文件 + 9 个修改文件
9. **架构完全对齐**：复用 `LossParams`、`CpuOpContext`/`MultiStreamCaptureState`、流策略、Region 体系