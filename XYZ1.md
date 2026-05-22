# SOFTMAX_CE 融合算子 — 完整实现方案

**版本**: V1.0  
**日期**: 2026-05-19  
**基于**: SOF.md + S/K/D 三份提案 + 代码审计 + 用户补充要求

---

## 一、需求回顾

实现 4 个融合算子：

| 算子 | 说明 |
|------|------|
| `SOFTMAX_CE_FP32_FWD` | FP32 前向：logits + labels → loss + softmax_probs + TOP-1/5 + 预测标签 |
| `SOFTMAX_CE_FP32_BWD` | FP32 反向：probs + labels → grad(logits) × scaling_factor |
| `SOFTMAX_CE_AMP_FWD` | AMP 前向：FP16 logits → FP32 内部运算 → probs(FP32) + loss + TOP-1/5 |
| `SOFTMAX_CE_AMP_BWD` | AMP 反向：probs(FP32) → grad(FP16) × scaling_factor |

**核心约束**：

- CUDA 绝不展开 one-hot（浪费显存和带宽）
- CPU 也绝不展开 one-hot（条件减法即可）
- AMP 内部全部 FP32 运算（softmax + CE 涉及规约，FP16 极易出错）
- 中间有意义的张量放 Temp 区，不用 workspace
- 算子名必须叫 `SOFTMAX_CE_*`，旧枚举 `CROSS_ENTROPY_LOSS_*` 删除替换

---

## 二、用户补充要求（关键）

来自 XXX.md 末尾的用户补充，这段话纠正了三份提案中的过度设计：

> labels只是一个普通的张量而已！只不过它的形状是[batch size,1,1,1]，必定compact，数据类型是INT32，区域是I_A_LABEL或I_B_LABEL！你设计算子的时候，只需要接受并处理张量，标签就是输入张量之一，它跟FC层的权重是一个道理！你的算子接受DTensor的传入，然后处理它，就这么简单！
>
> 你的SOFTMAX_CE系列算子毕竟只是一个普通算子，你要让它符合我们的算子设计的API风格！它充其量只是一个涉及的IO较多的算子而已！

**解读**：

- labels 是"一个普通张量"——不需要专门设计注入机制
- 放入 `infer_softmaxce_tensors` 返回值，编译器正常分配并连线
- 不需要修改 `MemoryPlan` 增加特殊 ID 存储
- 不需要在 `compiler.cpp` 中增加特殊的 labels 注入逻辑
- 算子 API 风格与 FC/ReLU 等完全一致

---

## 三、现有基础设施（可直接复用或修改）

### 3.1 已定义的枚举和参数

```cpp
// op_kind.h — 旧枚举（将删除）
CROSS_ENTROPY_LOSS,              // 无方向版本
CROSS_ENTROPY_LOSS_FP32_FWD,     // FP32 前向
CROSS_ENTROPY_LOSS_FP32_BWD,     // FP32 反向
CROSS_ENTROPY_LOSS_AMP_FWD,      // AMP 前向
CROSS_ENTROPY_LOSS_AMP_BWD,      // AMP 反向

// op_kind.h — 已存在的参数结构（复用）
struct LossParams {
    float label_smoothing = 0.0f;  // V1 仅支持 smoothing=0
    int num_classes = 1000;
};
```

### 3.2 已定义的编译器框架

从 [`layer_descriptor_registry.cpp:505-537`](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L505-L537)：

- `infer_softmaxce_tensors` — 现有返回 4 个 tensor（需扩展至 8 个）
- `build_softmaxce_forward` — 现有 output_indices = {0,1,2,3}（需扩展）
- `build_softmaxce_backward` — 现有 input_indices = {1,2,3}（需扩展）
- `SoftmaxCELayerParams` → `LossParams` 转换已在 [`compiler.cpp:559-562`](file:///r:/renaissance/src/graph/compiler.cpp#L559-L562) 实现
- `get_layer_output_id(SoftmaxCE)` → `idx=1` (ce_output) 已存在
- `get_grad_output_id(SoftmaxCE)` → `idx=1` (in-place 覆盖 ce_output) 已存在

### 3.3 已有但未注册的算子

`g_compute_op_table` 中 `CROSS_ENTROPY_LOSS_*` 的 5 个 entry 的 `launch_cpu` 和 `launch_cuda` 均为 `nullptr`。删除后替换为 `SOFTMAX_CE_*` 的 4 个 entry。

### 3.4 参考算子

| 模式 | 参考 |
|------|------|
| CUDA 自定义 kernel（非 cuDNN FE） | [`relu_op.cu`](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cu) |
| CUDA half↔float 转换 | [`gap_op.cu:L54-L56`](file:///r:/renaissance/src/backend/ops/dtensor/gap_op.cu#L54-L56) |
| CPU Eigen 优化 | [`fc_op.cpp:647-711`](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L647-L711) |
| 算子注册模式 | `register_op_fc()` / `register_op_relu()` |
| 流策略 | [`op_stream_policy.cpp`](file:///r:/renaissance/src/backend/op_stream_policy.cpp) |

---

## 四、核心设计决策

### 4.1 算子枚举：删除旧 + 新增

在 `op_kind.h` 中删除 5 个旧枚举，在原位置插入 4 个新枚举：

```cpp
// ===== 删除以下 5 个 =====
// CROSS_ENTROPY_LOSS,
// CROSS_ENTROPY_LOSS_FP32_FWD,
// CROSS_ENTROPY_LOSS_FP32_BWD,
// CROSS_ENTROPY_LOSS_AMP_FWD,
// CROSS_ENTROPY_LOSS_AMP_BWD,

// ===== 新增以下 4 个 =====
SOFTMAX_CE_FP32_FWD,
SOFTMAX_CE_FP32_BWD,
SOFTMAX_CE_AMP_FWD,
SOFTMAX_CE_AMP_BWD,
```

**影响**：后续所有枚举值向前偏移 1 位（5→4），`COUNT` 减 1。`g_compute_op_table` 使用 `ComputeOp::XXX` 常量索引，编译器自动适配。

### 4.2 参数结构：复用 LossParams

不新增 `SoftmaxCEParams`。`LossParams` 已有 `num_classes` 和 `label_smoothing`，足够使用。

`scaling_factor` **不放入 params**——它是一个运行时 DTensor（来自 loss scaling），作为 FWD 的输入张量之一传入。

### 4.3 Region：新增 1 个 R_METRIC_INT32

在 `types.h` 的 `Region` 枚举中，`T_TEMP_INT8` 之后新增：

```cpp
R_METRIC_INT32,      // 066 — 推理指标（TOP-1/TOP-5 正确数、预测标签）
```

更新 `NUM_REGIONS = 66`。

**选择 1 个 Region 而非 3 个的理由**：

- K 方案已验证：1 个 Region 同时容纳 TOP-1、TOP-5、predicted_labels 三个 INT32 张量，内存布局统一
- 用户强调"简单"，3 个 Region（D 方案）增加了不必要复杂度
- 后续 RANGE_OP 的 sum-all-reduce 可以通过 DTensor ID 精确指定目标张量，不需要依赖 Region 级别区分
- 未来扩展其他推理指标只需增加新 DTensor，不需要新 Region

### 4.4 FWD 输出 softmax_probs 到 Temp 区

FWD 将 `softmax_probs` 保存到 `T_TEMP_FP32` 区，BWD 直接读取，**不重复计算**。

三个提案一致同意此决策。内存代价极小（典型 batch=7, num_classes=1000 → 28 KB）。

### 4.5 labels 是最普通的 DTensor

遵循用户补充要求：

- labels 放入 `infer_softmaxce_tensors` 返回值的第 0 位
- Region 使用 `I_A_LABEL`（训练）或 `I_B_LABEL`（验证）
- Shape `[batch, 1, 1, 1]`，DType `INT32`
- 子图通过 `input_indices = {0}` 引用
- 编译器在 Phase 4 正常分配 ID 并连线
- **无需修改 MemoryPlan**，**无需 compiler 注入逻辑**

### 4.6 手写 CUDA kernel，不用 cuDNN FE

- cuDNN 没有 fused Softmax+CrossEntropy 算子
- 拆分为两个 cuDNN FE 节点会失去融合优势
- TOP-1/TOP-5/预测标签无法用 cuDNN FE 表达
- 自定义 kernel 可将 max + exp + sum + log + topk 全部融合在一个 kernel 中

### 4.7 CPU 用 Eigen，不展开 one-hot

使用 Eigen 的 row-wise maxCoeff、array exp、sum 等向量化操作。BWD 用条件减法：`grad[i][c] = prob * scale; if (c==label) grad[i][c] -= scale;`。零 one-hot 展开。

---

## 五、数学基础

### 5.1 FWD

给定 logits $x \in \mathbb{R}^{B \times C}$，labels $y \in \{0, ..., C-1\}^B$：

1. **数值稳定**：$x'_{ij} = x_{ij} - \max_k x_{ik}$
2. **Softmax**：$p_{ij} = \dfrac{\exp(x'_{ij})}{\sum_k \exp(x'_{ik})}$
3. **CE Loss**：$L_i = -\log(p_{i, y_i})$
4. **总 Loss**：$L = \dfrac{\text{scaling\_factor}}{B} \sum_i L_i$
5. **TOP-1**：$\arg\max_j p_{ij} = y_i$
6. **TOP-5**：$y_i \in \text{top-5}(p_{ij})$
7. **inv_scaling**：$1.0 / B$（用于 BWD 恢复梯度）

### 5.2 BWD

$$\frac{\partial L}{\partial x_{ij}} = \frac{\text{scaling\_factor}}{B} \left(p_{ij} - \mathbf{1}[j = y_i]\right)$$

**关键优化**：不展开 B×C 的 one-hot 矩阵，仅在 label 位置减 1：

```
for each c:
    grad[i][c] = probs[i][c] * scale * inv_scaling
grad[i][label[i]] -= scale * inv_scaling
```

---

## 六、I/O 张量布局

### 6.1 `infer_softmaxce_tensors`（8 张量）

```cpp
std::vector<TensorDesc> infer_softmaxce_tensors(
    const Shape& input, const OpParams& p, const InferContext& ctx)
{
    int batch = input.n();
    const auto* lp = std::get_if<LossParams>(&p.data);
    int num_classes = lp ? lp->num_classes : input.c();
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    return {
        // [0] labels — 普通 DTensor，I_A_LABEL 区域
        TensorDesc{"labels",             Shape{batch,1,1,1},      Region::I_A_LABEL,          DType::INT32},

        // [1] ce_loss — 标量
        TensorDesc{"ce_loss",            Shape{1,1,1,1},          Region::S_SCALAR_FP32,      DType::FP32},

        // [2] ce_output / logits — 输入来自前一层，BWD 梯度 in-place 覆盖
        TensorDesc{"ce_output",          input,                   select_feature_region(ctx), feat_dt},

        // [3] scaling_factor — 标量，运行时由 loss scaling 传入
        TensorDesc{"scaling_factor",     Shape{1,1,1,1},          Region::S_SCALAR_FP32,      DType::FP32},

        // [4] inv_scaling_factor — 标量，FWD 计算输出（1/batch）
        TensorDesc{"inv_scaling_factor", Shape{1,1,1,1},          Region::S_SCALAR_FP32,      DType::FP32},

        // [5] top1_correct — INT32 标量
        TensorDesc{"top1_correct",       Shape{1,1,1,1},          Region::R_METRIC_INT32,     DType::INT32},

        // [6] top5_correct — INT32 标量
        TensorDesc{"top5_correct",       Shape{1,1,1,1},          Region::R_METRIC_INT32,     DType::INT32},

        // [7] predicted_labels — [batch] INT32
        TensorDesc{"predicted_labels",   Shape{batch,1,1,1},      Region::R_METRIC_INT32,     DType::INT32},

        // [8] softmax_probs — Temp 区，FWD 输出供 BWD 复用
        TensorDesc{"softmax_probs",      input,                   Region::T_TEMP_FP32,         DType::FP32},
    };
}
```

### 6.2 子图 `build_softmaxce_forward`

```cpp
SubgraphPattern build_softmaxce_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 9) return p;

    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_FWD
               : ComputeOp::SOFTMAX_CE_FP32_FWD;
    // labels[0] 作为输入（编译器 auto-wire logits 为 input_ids[0]）
    n.input_indices  = {0, 3};   // labels + scaling_factor
    n.output_indices = {1, 4, 5, 6, 7, 8};  // loss + inv_scaling + top1 + top5 + pred + softmax_probs
    p.nodes.push_back(n);
    return p;
}
```

**算子运行时 `input_ids`**（编译器自动解析后）：
```
input_ids:
  [0] = logits          (编译器从上一层自动连线)
  [1] = labels          (通过 input_indices={0}  → descs[0])
  [2] = scaling_factor  (通过 input_indices={3}  → descs[3])

output_ids:
  [0] = ce_loss             (descs[1])
  [1] = inv_scaling_factor  (descs[4])
  [2] = top1_correct        (descs[5])
  [3] = top5_correct        (descs[6])
  [4] = predicted_labels    (descs[7])
  [5] = softmax_probs       (descs[8])
```

### 6.3 子图 `build_softmaxce_backward`

```cpp
SubgraphPattern build_softmaxce_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 9) return p;

    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_BWD
               : ComputeOp::SOFTMAX_CE_FP32_BWD;
    // softmax_probs[8] + labels[0] + scaling_factor[3] + inv_scaling[4]
    n.input_indices  = {8, 0, 3, 4};
    // 梯度 in-place 覆盖 ce_output[2]
    n.output_indices = {2};
    p.nodes.push_back(n);
    return p;
}
```

**算子运行时 `input_ids`**：
```
input_ids:
  [0] = logits          (编译器从上一层自动连线 — grad 覆盖目标)
  [1] = softmax_probs   (descs[8])
  [2] = labels          (descs[0])
  [3] = scaling_factor  (descs[3])
  [4] = inv_scaling     (descs[4])

output_ids:
  [0] = dL/d(logits)    (descs[2] — in-place 覆盖 logits)
```

### 6.4 `compiler.cpp` 需要同步更新

现有的两个 lambda 函数需要更新 index：

```cpp
// get_layer_output_id — ce_output 从 descs[1] 变为 descs[2]
case LayerKind::SoftmaxCE: idx = 2; break;

// get_grad_output_id — 梯度 in-place 覆盖 ce_output descs[2]
case LayerKind::SoftmaxCE: idx = 2; break;
```

---

## 七、CPU 实现

### 7.1 FWD（FP32，Eigen 向量化）

```cpp
static void launch_softmax_ce_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_FWD CPU missing LossParams");

    int batch       = op_ctx->input_shape.n;
    int num_classes = p->num_classes;
    auto* ctx = const_cast<DeviceContext*>(op_ctx->ctx);

    // input_ids: [0]=logits, [1]=labels, [2]=scaling_factor
    const float*   logits  = static_cast<const float*>(ctx->ptr_at(op_ctx->input_ids[0]));
    const int32_t* labels  = static_cast<const int32_t*>(ctx->ptr_at(op_ctx->input_ids[1]));
    float*         scaling = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[2]));

    // output_ids: [0]=loss, [1]=inv_scaling, [2]=top1, [3]=top5, [4]=pred, [5]=probs
    float*   loss_out    = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[0]));
    float*   inv_scaling = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[1]));
    int32_t* top1_out    = static_cast<int32_t*>(ctx->ptr_at(op_ctx->output_ids[2]));
    int32_t* top5_out    = static_cast<int32_t*>(ctx->ptr_at(op_ctx->output_ids[3]));
    int32_t* pred_out    = static_cast<int32_t*>(ctx->ptr_at(op_ctx->output_ids[4]));
    float*   probs       = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[5]));

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const MatrixXfRow> logits_mat(logits, batch, num_classes);
    Eigen::Map<MatrixXfRow>       probs_mat(probs,  batch, num_classes);

    double loss_sum = 0.0;
    int    top1_cnt = 0, top5_cnt = 0;

    for (int b = 0; b < batch; ++b) {
        // 1. max per row（数值稳定）
        float max_val = logits_mat.row(b).maxCoeff();

        // 2. exp (Eigen 向量化)
        probs_mat.row(b) = (logits_mat.row(b).array() - max_val).exp();

        // 3. sum
        float sum_exp = probs_mat.row(b).sum();

        // 4. normalize (in-place)
        probs_mat.row(b) /= sum_exp;

        // 5. CE loss
        int   label   = labels[b];
        float p_label = probs_mat.row(b)(label);
        loss_sum += -std::log(std::max(p_label, 1e-12f));

        // 6. TOP-1
        int top1_idx;
        probs_mat.row(b).maxCoeff(&top1_idx);
        pred_out[b] = top1_idx;
        if (top1_idx == label) ++top1_cnt;

        // 7. TOP-5: 5 次线性扫描找前 5 大概率值
        bool label_in_top5 = false;
        Eigen::RowVectorXf row_copy = probs_mat.row(b);
        for (int k = 0; k < 5 && k < num_classes; ++k) {
            int best_idx = 0;
            float best_val = row_copy(0);
            for (int j = 1; j < num_classes; ++j) {
                if (row_copy(j) > best_val) { best_val = row_copy(j); best_idx = j; }
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

### 7.2 BWD（FP32，Eigen 向量化）

```cpp
static void launch_softmax_ce_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_BWD CPU missing LossParams");

    int batch       = op_ctx->input_shape.n;
    int num_classes = p->num_classes;
    auto* ctx = const_cast<DeviceContext*>(op_ctx->ctx);

    // input_ids: [0]=logits(grad目标), [1]=probs, [2]=labels, [3]=scaling, [4]=inv_scaling
    const float*   probs       = static_cast<const float*>(ctx->ptr_at(op_ctx->input_ids[1]));
    const int32_t* labels      = static_cast<const int32_t*>(ctx->ptr_at(op_ctx->input_ids[2]));
    float*         scaling     = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[3]));
    float*         inv_scaling = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[4]));

    // output_ids: [0]=grad (in-place 覆盖 logits)
    float* grad = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[0]));

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const MatrixXfRow> probs_mat(probs, batch, num_classes);
    Eigen::Map<MatrixXfRow>       grad_mat(grad,  batch, num_classes);

    // 单次广播乘法：所有元素乘以 scale * inv_scaling
    float s = scaling[0] * inv_scaling[0];
    grad_mat = probs_mat * s;

    // 仅修改 label 位置：减 s
    for (int b = 0; b < batch; ++b) {
        grad_mat(b, labels[b]) -= s;
    }
}
```

### 7.3 AMP CPU — 不支持

```cpp
static void launch_softmax_ce_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_* is not supported on CPU");
}
```

---

## 八、CUDA 实现

### 8.1 新增文件

```
src/backend/ops/dtensor/softmax_ce_op.cu   — CUDA kernels
src/backend/ops/dtensor/softmax_ce_op.cpp  — 分发 + 注册
```

### 8.2 FWD Kernel（FP32 + AMP 模板化）

```cuda
template<typename T, bool IS_AMP>
__global__ void softmax_ce_fwd_kernel(
    const T*      __restrict__ logits,           // float 或 __half
    const int*    __restrict__ labels,
    float*        __restrict__ ce_loss,          // 标量，atomicAdd 累加
    float*        __restrict__ inv_scaling,
    int*          __restrict__ top1_correct,     // 标量，atomicAdd
    int*          __restrict__ top5_correct,     // 标量，atomicAdd
    int*          __restrict__ predicted_labels, // [batch]
    float*        __restrict__ softmax_probs,    // [batch * num_classes]
    int batch, int num_classes)
{
    extern __shared__ float smem[];
    int b = blockIdx.x;
    if (b >= batch) return;

    const T* logit_row = logits + b * num_classes;
    float*   prob_row  = softmax_probs + b * num_classes;
    int      label     = labels[b];

    // 1. 加载 logits → shared memory + 找局部最大值
    float max_val = -INFINITY;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        float v = IS_AMP ? __half2float(logit_row[c]) : (float)logit_row[c];
        smem[c] = v;
        max_val = fmaxf(max_val, v);
    }

    // 2. warp-level reduce max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // 3. exp + sum
    float sum_exp = 0.0f;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        smem[c] = expf(smem[c] - max_val);
        sum_exp += smem[c];
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

    // 4. 归一化写入 softmax_probs
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        smem[c] /= sum_exp;
        prob_row[c] = smem[c];
    }

    // 5. Thread 0: loss + top1 + top5 + predicted_label
    if (threadIdx.x == 0) {
        // CE loss (数值稳定：-log(p_label))
        float p_label = smem[label];
        float ce = -logf(fmaxf(p_label, 1e-12f));
        atomicAdd(ce_loss, ce);

        // TOP-1: argmax
        int top1_idx = 0;
        float top1_val = smem[0];
        for (int c = 1; c < num_classes; ++c) {
            if (smem[c] > top1_val) { top1_val = smem[c]; top1_idx = c; }
        }
        predicted_labels[b] = top1_idx;
        if (top1_idx == label) atomicAdd(top1_correct, 1);

        // TOP-5: 5 次扫描找前 5 大
        bool label_in_top5 = false;
        int top5_idx[5] = {};
        for (int k = 0; k < 5 && k < num_classes; ++k) {
            int best_j = 0;
            float best_v = smem[0];
            for (int j = 1; j < num_classes; ++j) {
                bool already = false;
                for (int p = 0; p < k; ++p) {
                    if (j == top5_idx[p]) { already = true; break; }
                }
                if (!already && smem[j] > best_v) { best_v = smem[j]; best_j = j; }
            }
            top5_idx[k] = best_j;
            if (best_j == label) label_in_top5 = true;
        }
        if (label_in_top5) atomicAdd(top5_correct, 1);
    }
}
```

**关键设计**：

- 每个 block 处理一个 sample
- shared memory 大小 = `num_classes * sizeof(float)`（典型 1000×4=4KB，轻松容纳）
- 数值稳定：CE loss 用 `-log(prob_label)` 而非 `-logits[label] + log_sum_exp`（shared memory 中已有归一化 prob）
- TOP-1/TOP-5 通过 `atomicAdd` 原子累加
- 注意：`ce_loss` 是 per-batch 累加标量，launch 后需乘以 `scaling_factor / batch`

### 8.3 BWD Kernel

```cuda
template<typename T, bool IS_AMP>
__global__ void softmax_ce_bwd_kernel(
    const float* __restrict__ softmax_probs,  // always FP32
    const int*   __restrict__ labels,
    float scaling_factor,
    float inv_scaling,
    T*           __restrict__ grad_logits,     // float 或 __half
    int batch, int num_classes)
{
    int b = blockIdx.x;
    if (b >= batch) return;

    float s  = scaling_factor * inv_scaling;
    int label = labels[b];

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

**特点**：

- 无需 shared memory，无 reduction
- 条件判断 `c == label` 替代 one-hot 展开
- AMP 仅在写回时做 `__float2half`

### 8.4 Launch Wrapper 声明

```cpp
// FP32
extern cudaError_t launch_softmax_ce_fwd_fp32(
    const float* logits, const int* labels,
    float* loss, float* inv_scaling,
    int* top1, int* top5, int* pred,
    float* probs, int N, int C, cudaStream_t s);

extern cudaError_t launch_softmax_ce_bwd_fp32(
    const float* probs, const int* labels,
    float scaling, float inv_scaling,
    float* grad, int N, int C, cudaStream_t s);

// AMP
extern cudaError_t launch_softmax_ce_fwd_amp(
    const __half* logits, const int* labels,
    float* loss, float* inv_scaling,
    int* top1, int* top5, int* pred,
    float* probs, int N, int C, cudaStream_t s);

extern cudaError_t launch_softmax_ce_bwd_amp(
    const float* probs, const int* labels,
    float scaling, float inv_scaling,
    __half* grad, int N, int C, cudaStream_t s);
```

### 8.5 CUDA Launch 分发（`.cpp`）

```cpp
#ifdef TR_USE_CUDA
static void launch_softmax_ce_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<LossParams>(&node.params.data);
    int batch       = mp.get_dtensor(node.input_ids[0]).n();
    int num_classes = p->num_classes;

    const float* logits  = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const int*   labels  = static_cast<const int*>(ctx.ptr_at(node.input_ids[1]));
    float*       scaling = static_cast<float*>(ctx.ptr_at(node.input_ids[2]));

    float* loss    = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    float* inv_s   = static_cast<float*>(ctx.ptr_at(node.output_ids[1]));
    int*   top1    = static_cast<int*>(ctx.ptr_at(node.output_ids[2]));
    int*   top5    = static_cast<int*>(ctx.ptr_at(node.output_ids[3]));
    int*   pred    = static_cast<int*>(ctx.ptr_at(node.output_ids[4]));
    float* probs   = static_cast<float*>(ctx.ptr_at(node.output_ids[5]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;

    // 初始化累加标量为 0
    cudaMemsetAsync(loss, 0, sizeof(float), s);
    cudaMemsetAsync(top1, 0, sizeof(int), s);
    cudaMemsetAsync(top5, 0, sizeof(int), s);

    int threads = 256;
    if (num_classes <= 128) threads = 128;
    else if (num_classes <= 512) threads = 256;
    else threads = 512;
    size_t smem = num_classes * sizeof(float);

    cudaError_t err = launch_softmax_ce_fwd_fp32(
        logits, labels, loss, inv_s, top1, top5, pred, probs,
        batch, num_classes, s);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_FP32_FWD kernel failed: " << cudaGetErrorString(err));

    // Post-kernel: loss = loss_sum / batch * scaling_factor
    // 可以在第二个 micro-kernel 中完成，或通过 cub 做 device-side reduction
    // 简化方案：此时 loss 中存的是 sum(ce_i)，在 CPU 端完成缩放：
    //   float host_loss; cudaMemcpy(&host_loss, loss, sizeof(float), cudaMemcpyDeviceToHost);
    //   host_loss = host_loss / batch * scaling[0]; cudaMemcpy(loss, &host_loss, sizeof(float), ...);
    // 实际实现中将缩放整合进 kernel 或用 post-processing kernel

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif
```

### 8.6 Stream 策略

在 [`op_stream_policy.cpp`](file:///r:/renaissance/src/backend/op_stream_policy.cpp) 添加：

```cpp
case ComputeOp::SOFTMAX_CE_FP32_FWD:
case ComputeOp::SOFTMAX_CE_FP32_BWD:
case ComputeOp::SOFTMAX_CE_AMP_FWD:
case ComputeOp::SOFTMAX_CE_AMP_BWD:
    return StreamKind::COMP_1;
```

---

## 九、注册与集成

### 9.1 注册函数声明

在 [`op_registry.h`](file:///r:/renaissance/include/renaissance/backend/op_registry.h) 添加：

```cpp
void register_op_softmax_ce();
```

### 9.2 注册实现

```cpp
void register_op_softmax_ce() {
    auto& t = g_compute_op_table;

    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_FWD;
        e.launch_cpu = launch_softmax_ce_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_fwd_cuda;
#endif
    }
    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_BWD;
        e.launch_cpu = launch_softmax_ce_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_bwd_cuda;
#endif
    }
    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_FWD;
        e.launch_cpu = launch_softmax_ce_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_fwd_cuda;
#endif
    }
    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_BWD;
        e.launch_cpu = launch_softmax_ce_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_bwd_cuda;
#endif
    }
}
```

### 9.3 调用注册

在 [`op_registry.cpp`](file:///r:/renaissance/src/backend/op_registry.cpp) 的 `register_all_ops()` 中添加：

```cpp
register_op_softmax_ce();
```

### 9.4 warmup（CUDA Graph 场景）

```cpp
case ComputeOp::SOFTMAX_CE_AMP_FWD:
case ComputeOp::SOFTMAX_CE_AMP_BWD:
    return true;
```

---

## 十、改动清单

| # | 文件 | 改动性质 | 行数 |
|---|------|----------|------|
| 1 | `include/renaissance/graph/op_kind.h` | 删除 5 个 `CROSS_ENTROPY_LOSS_*`，新增 4 个 `SOFTMAX_CE_*` | ~12 |
| 2 | `include/renaissance/core/types.h` | 新增 `R_METRIC_INT32`（066），更新 `NUM_REGIONS=66` | +2 |
| 3 | `src/graph/op_kind.cpp` | 更新 `compute_op_to_string`（5 旧→4 新） | ~9 |
| 4 | `src/graph/layer_descriptor_registry.cpp` | 重写 `infer_softmaxce_tensors` (4→9)、更新 forward/backward/inference 子图 | ~50 |
| 5 | `src/graph/memory_plan.cpp` | `R_METRIC_INT32` Region 注册（INT32 类型、无需梯度、无需 momentum） | ~5 |
| 6 | `src/graph/compiler.cpp` | 更新 `get_layer_output_id`(idx 1→2)、`get_grad_output_id`(idx 1→2) | ~4 |
| 7 | `src/backend/ops/dtensor/softmax_ce_op.cu` | **新文件**：FWD + BWD CUDA kernels（FP32 + AMP 模板化） | ~250 |
| 8 | `src/backend/ops/dtensor/softmax_ce_op.cpp` | **新文件**：CPU 实现 + CUDA launch 分发 + 注册 | ~300 |
| 9 | `include/renaissance/backend/op_registry.h` | 声明 `register_op_softmax_ce()` | +1 |
| 10 | `src/backend/op_registry.cpp` | 调用 `register_op_softmax_ce()`；`require_warmup` 添加 2 项 | ~3 |
| 11 | `src/backend/op_stream_policy.cpp` | 4 个 CE 算子的流策略（`COMP_1`） | +4 |
| 12 | `src/CMakeLists.txt` | 添加 `.cu` + `.cpp` 到源文件列表 | +2 |

**总改动量**：约 640 行，2 个新文件 + 10 个修改文件，零接口破坏。

---

## 十一、实施步骤

### Phase 1：基础设施（Step 1 → 2 → 3 → 5）

1. 在 `op_kind.h` 中删除 5 个旧枚举，插入 4 个新枚举
2. 在 `types.h` 中新增 `R_METRIC_INT32`，更新 `NUM_REGIONS`
3. 在 `op_kind.cpp` 中更新 `compute_op_to_string`
4. 在 `memory_plan.cpp` 中注册新 Region

**验证**：编译通过，`g_compute_op_table` 大小正确，`Region` 枚举值正确

### Phase 2：编译器适配（Step 4 → 6）

5. 重写 `infer_softmaxce_tensors`（4→9 张量）
6. 重写 `build_softmaxce_forward` / `build_softmaxce_backward` / `build_softmaxce_inference`
7. 更新 `compiler.cpp` 中两个 lambda 的 index（1→2）

**验证**：编译通过，编译器能正确分配 labels/ce_output/scaling/probs 等张量的 DTensor ID

### Phase 3：算子实现（Step 7 → 8 → 12）

8. 创建 `softmax_ce_op.cu`：
   - FWD kernel（模板化 FP32+AMP，shared memory per-block reduction）
   - BWD kernel（模板化 FP32+AMP，直接条件减法，无 shared memory）
   - Launch wrapper 函数（4 个 export）
9. 创建 `softmax_ce_op.cpp`：
   - CPU FWD（Eigen 向量化 softmax + TOP-1/5 扫描）
   - CPU BWD（Eigen broadcast 乘法 + 条件减法）
   - CUDA launch 分发（4 个 static 函数）
   - AMP CPU not_supported 占位（2 个）
   - `register_op_softmax_ce()` 注册函数
10. 更新 `CMakeLists.txt` 添加 `.cu` + `.cpp`

**验证**：编译通过，无符号未定义

### Phase 4：注册与集成（Step 9 → 10 → 11）

11. 在 `op_registry.h` 声明 `register_op_softmax_ce()`
12. 在 `op_registry.cpp` 中调用注册函数；在 `require_warmup` 添加 AMP 条目
13. 在 `op_stream_policy.cpp` 添加 4 个 CE 算子的流策略

**验证**：编译通过，`g_compute_op_table` 对应 4 个 entry 非 nullptr

### Phase 5：测试验证

#### 5.1 Python 参考脚本

```python
import torch
import torch.nn.functional as F

batch, num_classes = 7, 1000
logits = torch.randn(batch, num_classes) * 2.0
labels = torch.randint(0, num_classes, (batch,))

# FWD
probs = F.softmax(logits, dim=1)
loss  = F.cross_entropy(logits, labels, reduction='mean')
pred  = torch.argmax(logits, dim=1)
top1  = (pred == labels).sum().item()
top5  = 0
for i in range(batch):
    _, top5_idx = torch.topk(logits[i], 5)
    top5 += int(labels[i].item() in top5_idx.tolist())

# BWD
logits = logits.clone().requires_grad_(True)
loss2  = F.cross_entropy(logits, labels, reduction='mean')
loss2.backward()
grad  = logits.grad

# Save TSR...
```

#### 5.2 C++ 正确性测试（参考 `test_flatten_fc_relu_fc.cpp`）

- CPU FWD vs PyTorch（loss、probs、top1、top5 MSE < 1e-5）
- CPU BWD vs PyTorch grad
- GPU FWD/BWD vs PyTorch
- AMP FWD/BWD vs PyTorch（较高容差 1e-4，允许 FP16 转换误差）

#### 5.3 Composite 测试

- 构建端到端训练模型（至少 FC → SoftmaxCE）
- 验证 loss 收敛和 gradient 不爆炸

---

## 十二、风险与缓解

| 风险 | 缓解 |
|------|------|
| CPU TOP-5 O(B×5C) 开销大 | 对 C≤10000 可接受（1000×5=5000 次比较/样本）；超大 C 可改为 partial_sort |
| CUDA shared memory 超限（C>12000 需 48KB） | `cudaFuncSetAttribute` 指定动态 SMEM；超限时改为 global memory 分块 |
| `label_smoothing > 0` 未实现 | `LossParams` 已预留，V1 仅实现 smoothing=0，V2 扩展 |
| `log(0)` 数值下溢 | `max(p_label, 1e-12f)` 保护 |
| CE loss 标量累加 kernel 内 atomicAdd 竞态 | 先写 per-sample loss 到 `loss[batch]` 数组，再用 cub::DeviceReduce 或额外 kernel 求和 |
| 枚举值重排导致旧代码索引错位 | 所有表项使用 `ComputeOp::XXX` 常量，编译器自动适配 |
| AMP 下 logits 步幅非连续 | kernel 中按 DTensor stride 访问（通过 `n_stride_cuda()` 等） |

---

## 十三、总结

1. **算子名**：`SOFTMAX_CE_*`（4 个），旧 `CROSS_ENTROPY_LOSS_*`（5 个）全部删除替换
2. **Region**：新增 1 个 `R_METRIC_INT32`，同时容纳 TOP-1/TOP-5/predicted_labels
3. **Params**：复用 `LossParams`，scaling_factor 是运行时 DTensor 输入
4. **labels**：最普通的 DTensor（`I_A_LABEL`），放入 `infer_softmaxce_tensors`，编译器正常处理
5. **FWD→BWD 复用**：`softmax_probs` 写入 `T_TEMP_FP32`，BWD 直接读取，零重计算
6. **零 one-hot 展开**：CPU/CUDA 均用条件减法 `if (c==label) g -= s`
7. **AMP 全 FP32 内部**：仅 logits 输入做 FP16→FP32，grad 输出做 FP32→FP16
8. **CPU Eigen 向量化**：row-wise maxCoeff + array exp + sum，BWD 一次广播乘法
9. **CUDA 手写 kernel**：shared memory per-block reduction（FWD），单个条件减法循环（BWD）
10. **约 640 行新代码**，2 个新文件 + 10 个修改文件，**零接口破坏**
11. **API 风格完全对齐**：与 FC/ReLU/GAP 等现有算子一致，labels 是普通张量，遵循标准 `CpuOpContext`/`MultiStreamCaptureState` 接口