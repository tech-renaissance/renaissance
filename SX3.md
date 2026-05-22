# SOFTMAX_CE 融合算子最优综合方案

## 版本
**V3.0 最终综合版**  
日期: 2026-05-19  
编制: 技术觉醒团队  
状态: 待实施  
基于: 小伙伴S、K、D的讨论综合

---

## 一、需求共识与关键决策

### 1.1 核心需求

实现4个SOFTMAX_CE融合算子：
- `SOFTMAX_CE_FP32_FWD` - FP32前向（CPU + CUDA双后端）
- `SOFTMAX_CE_FP32_BWD` - FP32反向（CPU + CUDA双后端）
- `SOFTMAX_CE_AMP_FWD` - AMP前向（仅CUDA，logits FP16输入）
- `SOFTMAX_CE_AMP_BWD` - AMP反向（仅CUDA，梯度FP16输出）

### 1.2 关键约束（必须遵守）

| 约束 | 说明 | 严重性 |
|------|------|--------|
| **绝对不展开one-hot** | CUDA和CPU都避免创建one-hot张量 | 🔴 强制 |
| **中间运算全FP32** | 无论FP32/AMP，softmax+CE计算必须FP32 | 🔴 强制 |
| **梯度覆盖输入** | d_logits覆盖logits存储位置 | 🔴 强制 |
| **Temp区存有意义张量** | 不用workspace存softmax_probs等 | 🔴 强制 |
| **数值稳定** | 使用log-sum-exp技巧 | 🔴 强制 |
| **CUDA手写kernel** | 不用cuDNN FE（融合更高效） | 🟡 建议 |

### 1.3 重大分歧点与最终决策

#### 分歧1：是否新增Region？

- **小伙伴K**：新增R-Series（R_TOP1_CORRECT、R_TOP5_CORRECT、R_PREDICTED_LABEL）
- **小伙伴D**：新增R_RESULT_INT32
- **小伙伴S**：不新增，复用S-Series和T-Series

**✅ 最终决策：新增R-Series**

**理由**：
1. 推理结果是业务语义明确的张量（top1/top5准确率、预测标签），值得独立Region
2. 避免与现有S-Series（标量区）、T-Series（临时张量）语义混淆
3. 便于未来扩展其他推理指标（top-10、mAP等）
4. 小伙伴K已给出详细实现方案，验证可行

**新增Region**：
```cpp
// types.h: Region枚举末尾（DEFAULT之前）
// R-Series: 推理结果（INT32）
R_TOP1_CORRECT,      // 066 - TOP-1正确数
R_TOP5_CORRECT,      // 067 - TOP-5正确数  
R_PREDICTED_LABEL,   // 068 - 预测标签[batch]

NUM_REGIONS = 69  // 原65 + 3个新区 + 1个DEFAULT占位
```

#### 分歧2：使用现有CROSS_ENTROPY_LOSS_*还是新增SOFTMAX_CE_*？

- **小伙伴D**：复用现有CROSS_ENTROPY_LOSS_FP32_FWD/BWD等
- **小伙伴K、S**：新增SOFTMAX_CE_*系列

**✅ 最终决策：新增SOFTMAX_CE_*系列**

**理由**：
1. 语义清晰：SOFTMAX_CE明确表示融合算子
2. 行为差异：现有CROSS_ENTROPY_LOSS_*未注册实现，新增可避免破坏性
3. 参数结构差异：需要SoftmaxCEParams（num_classes、scaling_factor），而非通用LossParams
4. I/O数量差异：FWD输出6个张量（vs原有4个），需要新的编译器子图

**新增ComputeOp**：
```cpp
// op_kind.h: CROSS_ENTROPY_LOSS_AMP_BWD之后、ALLREDUCE_SUM之前
// === Softmax + CrossEntropy 融合算子 ===
SOFTMAX_CE_FP32_FWD,  // FP32前向
SOFTMAX_CE_FP32_BWD,  // FP32反向
SOFTMAX_CE_AMP_FWD,   // AMP前向（仅CUDA）
SOFTMAX_CE_AMP_BWD,   // AMP反向（仅CUDA）
```

#### 分歧3：softmax_probs是否需要FWD输出给BWD复用？

- **小伙伴K**：FWD输出softmax_probs到Temp区，BWD复用避免重复计算
- **小伙伴D**：BWD重新计算softmax_probs（简化内存管理）
- **小伙伴S**：未明确说明

**✅ 最终决策：FWD输出softmax_probs到Temp区，BWD复用**

**理由**：
1. 性能优化：避免BWD重复exp/log/sum运算（计算密集型）
2. 内存开销可接受：FP32 softmax_probs = batch×num_classes×4字节（batch=256, class=1000 → 1MB）
3. 符合框架设计：Temp区专门用于跨算子复用的中间张量
4. 小伙伴K已验证可行性

**Temp区分配**：
```cpp
TensorDesc{"softmax_probs", Shape{batch,1,1,num_classes}, Region::T_TEMP_FP32, DType::FP32}
```

#### 分歧4：BWD输入是否需要labels？

- **小伙伴D**：强调BWD需要labels作为输入（用于确定one-hot位置）
- **小伙伴K、S**：未明确讨论

**✅ 最终决策：BWD需要labels作为input_ids[1]**

**理由**：
1. 梯度公式：grad = softmax_probs - one_hot(label)
2. 不展开one-hot，但需要label索引确定减1的位置
3. 编译器子图需要明确labels依赖

---

## 二、I/O张量布局最终设计

### 2.1 FWD版本

```cpp
输入 (input_ids):
  [0] logits           [batch, 1, 1, num_classes]  FP32/FP16(AMP)  F_FEATURE_FP32/FP16
  [1] labels           [batch] 紧凑INT32            I_A_LABEL       DType::INT32

输出 (output_ids):
  [0] ce_loss          [1,1,1,1]  FP32              S_SCALAR_FP32
  [1] softmax_probs    [batch,1,1,num_classes]  FP32  T_TEMP_FP32     ← BWD复用
  [2] scaling_factor   [1,1,1,1]  FP32              S_SCALAR_FP32
  [3] inv_scaling      [1,1,1,1]  FP32              S_SCALAR_FP32
  [4] top1_correct     [1,1,1,1]  INT32             R_TOP1_CORRECT
  [5] top5_correct     [1,1,1,1]  INT32             R_TOP5_CORRECT
  [6] predicted_labels [batch,1,1,1]  INT32         R_PREDICTED_LABEL
```

**说明**：
- `inv_scaling`是`1.0/scaling_factor`，用于BWD恢复梯度
- `softmax_probs`输出到Temp区，BWD作为input_ids[0]复用
- `predicted_labels`是argmax(logits)，用于准确率统计

### 2.2 BWD版本

```cpp
输入 (input_ids):
  [0] softmax_probs    [batch,1,1,num_classes]  FP32  T_TEMP_FP32  ← FWD输出
  [1] labels           [batch] 紧凑INT32         I_A_LABEL    DType::INT32
  [2] scaling_factor   [1,1,1,1]  FP32           S_SCALAR_FP32
  [3] inv_scaling      [1,1,1,1]  FP32           S_SCALAR_FP32

输出 (output_ids):
  [0] d_logits         [batch,1,1,num_classes]  FP32/FP16(AMP)  覆盖logits存储位置
```

**说明**：
- BWD接收FWD的`softmax_probs`，避免重复计算exp/sum
- 梯度直接覆盖原logits的物理存储（通过MemoryPlan的alias机制）
- AMP模式：FP32计算梯度，最后转为FP16写入

---

## 三、数学公式与数值稳定性

### 3.1 FWD：Softmax + Cross Entropy

#### 数学定义

对于单个样本 $i$，logits $\mathbf{z} = [z_1, ..., z_K]$，标签 $y \in \{0, ..., K-1\}$：

1. **Softmax概率**：
   $$\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$

2. **Cross Entropy Loss**：
   $$L_i = -\log(\text{softmax}(z_y)) = -z_y + \log\left(\sum_{k=1}^{K} e^{z_k}\right)$$

3. **总Loss（带scaling）**：
   $$L = \frac{\text{scaling\_factor}}{N} \sum_{i=1}^{N} L_i$$

#### 数值稳定算法（log-sum-exp技巧）

```python
# 伪代码：单个样本的处理
max_val = max(logits)                    # z_max
exp_shifted = exp(logits - max_val)      # 避免overflow
sum_exp = sum(exp_shifted)               # Σe^(z_j - z_max)
log_sum_exp = max_val + log(sum_exp)     # log(Σe^z_j)

# CE loss
ce_loss = -logits[label] + log_sum_exp    # -z_y + log(Σe^z_j)

# Softmax概率
softmax_probs = exp_shifted / sum_exp    # e^(z_j - z_max) / Σe^(z_k - z_max)
```

**关键**：
- 减去max_val避免exp overflow
- softmax_probs输出供BWD复用

### 3.2 BWD：梯度计算

#### 数学推导

$$\frac{\partial L_i}{\partial z_j} = \frac{\partial}{\partial z_j} \left(-\log\frac{e^{z_y}}{\sum_k e^{z_k}}\right) 
= \frac{\partial}{\partial z_j} \left(-z_y + \log\sum_k e^{z_k}\right) 
= \text{softmax}(z_j) - \mathbb{1}[j = y]$$

总梯度（带scaling）：
$$\frac{\partial L}{\partial z_{ij}} = \frac{\text{scaling\_factor}}{N} \left(\text{softmax}(z_{ij}) - \mathbb{1}[j = y_i]\right)$$

#### 算法步骤

```python
# 伪代码：单个样本的梯度
# softmax_probs来自FWD输出（避免重复计算）
grad = softmax_probs.copy()
grad[label] -= 1.0          # one-hot：label位置减1
grad *= (scaling_factor / batch_size)  # 应用scaling
```

**关键**：
- 不展开one-hot，直接条件判断
- 直接覆盖logits存储位置

---

## 四、实现架构

### 4.1 文件结构（新增2个文件）

```
src/backend/ops/dtensor/
├── softmax_ce_op.cpp    (~200行) CPU实现 + CUDA launch + 注册
└── softmax_ce_op.cu     (~180行) CUDA kernels
```

### 4.2 参数结构定义

```cpp
// op_params.h
struct SoftmaxCEParams {
    int num_classes;       // 类别数（如1000 for ImageNet）
    float scaling_factor;  // 损失缩放因子（如1.0f）
    
    SoftmaxCEParams() = default;
    SoftmaxCEParams(int nc, float sf) : num_classes(nc), scaling_factor(sf) {}
    
    OpParams to_op_params() const {
        OpParams params;
        params.data = *this;
        return params;
    }
};
```

### 4.3 流策略

根据`op_stream_policy.cpp`：
- FWD/BWD均使用**COMP_1**流（与其他损失算子一致）
- AMP模式无特殊流处理

---

## 五、详细实现方案

### 5.1 CPU实现（softmax_ce_op.cpp）

#### 5.1.1 前向（FP32）

```cpp
static void launch_softmax_ce_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<SoftmaxCEParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "Missing SoftmaxCEParams");
    
    int num_classes = p->num_classes;
    float scaling_factor = p->scaling_factor;
    int batch_size = op_ctx->input_shape.n;
    
    // 输入
    const float* logits = static_cast<const float*>(
        op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const int32_t* labels = static_cast<const int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    
    // 输出
    float* ce_loss = static_cast<float*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    float* softmax_probs = static_cast<float*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));
    float* scaling_factor_out = static_cast<float*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[2]));
    float* inv_scaling = static_cast<float*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[3]));
    int32_t* top1_correct = static_cast<int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[4]));
    int32_t* top5_correct = static_cast<int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[5]));
    int32_t* predicted_labels = static_cast<int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[6]));
    
    // 初始化标量输出
    *ce_loss = 0.0f;
    *scaling_factor_out = scaling_factor;
    *inv_scaling = 1.0f / scaling_factor;
    *top1_correct = 0;
    *top5_correct = 0;
    
    // OpenMP并行处理batch
    #pragma omp parallel for reduction(+:*ce_loss, *top1_correct, *top5_correct)
    for (int i = 0; i < batch_size; ++i) {
        const float* logits_i = logits + i * num_classes;
        float* probs_i = softmax_probs + i * num_classes;
        int32_t label = labels[i];
        
        // 1. 找最大值（数值稳定）
        float max_val = logits_i[0];
        int pred = 0;
        for (int j = 1; j < num_classes; ++j) {
            if (logits_i[j] > max_val) {
                max_val = logits_i[j];
                pred = j;
            }
        }
        
        // 2. 计算exp和sum（log-sum-exp）
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            float exp_val = std::exp(logits_i[j] - max_val);
            probs_i[j] = exp_val;
            sum_exp += exp_val;
        }
        
        // 3. 归一化得到softmax概率
        for (int j = 0; j < num_classes; ++j) {
            probs_i[j] /= sum_exp;
        }
        
        // 4. 计算CE loss: -log(softmax_probs[label])
        float log_sum_exp = max_val + std::log(sum_exp);
        float sample_loss = -logits_i[label] + log_sum_exp;
        *ce_loss += sample_loss;
        
        // 5. 统计准确率
        predicted_labels[i] = pred;
        if (pred == label) {
            (*top1_correct)++;
        }
        
        // 6. Top-5准确率（num_classes >= 5时）
        if (num_classes >= 5) {
            // 部分排序找前5（比全排序快）
            std::vector<std::pair<float, int>> indexed_probs(num_classes);
            for (int j = 0; j < num_classes; ++j) {
                indexed_probs[j] = {logits_i[j], j};
            }
            std::partial_sort(indexed_probs.begin(), indexed_probs.begin() + 5,
                             indexed_probs.end(),
                             [](const auto& a, const auto& b) { return a.first > b.first; });
            for (int k = 0; k < 5; ++k) {
                if (indexed_probs[k].second == label) {
                    (*top5_correct)++;
                    break;
                }
            }
        }
    }
    
    // 7. 平均loss并应用scaling
    *ce_loss = (*ce_loss / batch_size) * scaling_factor;
}
```

**关键优化**：
- OpenMP并行处理batch
- 部分排序找Top-5（比全排序快）
- softmax_probs输出到Temp区供BWD复用

#### 5.1.2 反向（FP32）

```cpp
static void launch_softmax_ce_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<SoftmaxCEParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "Missing SoftmaxCEParams");
    
    int num_classes = p->num_classes;
    float scaling_factor = p->scaling_factor;
    int batch_size = op_ctx->input_shape.n;
    
    // 输入（复用FWD的softmax_probs）
    const float* softmax_probs = static_cast<const float*>(
        op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const int32_t* labels = static_cast<const int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    
    // 输出（覆盖logits）
    float* d_logits = static_cast<float*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    
    float scale = scaling_factor / batch_size;
    
    // OpenMP并行处理batch
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        const float* probs_i = softmax_probs + i * num_classes;
        float* grad_i = d_logits + i * num_classes;
        int32_t label = labels[i];
        
        // 梯度 = softmax_probs - one_hot(label)
        // one-hot隐式表示：label位置减1，其他不变
        for (int j = 0; j < num_classes; ++j) {
            grad_i[j] = probs_i[j];
        }
        grad_i[label] -= 1.0f;
        
        // 应用scaling
        for (int j = 0; j < num_classes; ++j) {
            grad_i[j] *= scale;
        }
    }
}
```

**关键优化**：
- 直接复用FWD的softmax_probs，避免重复计算
- OpenMP并行处理batch
- 梯度直接覆盖logits存储位置

#### 5.1.3 AMP占位函数

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

### 5.2 CUDA实现（softmax_ce_op.cu）

#### 5.2.1 前向Kernel（FP32）

```cuda
__global__ void softmax_ce_fwd_fp32_kernel(
    const float* __restrict__ logits,         // [batch, num_classes]
    const int32_t* __restrict__ labels,       // [batch]
    float* __restrict__ ce_loss,              // [1] 标量
    float* __restrict__ softmax_probs,        // [batch, num_classes]
    int32_t* __restrict__ predicted_labels,   // [batch]
    int32_t* __restrict__ top1_correct,       // [1] 标量
    int32_t* __restrict__ top5_correct,       // [1] 标量
    int batch_size,
    int num_classes,
    float scaling_factor)
{
    // 每个block处理一个样本
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    const float* logits_i = logits + batch_idx * num_classes;
    float* probs_i = softmax_probs + batch_idx * num_classes;
    int32_t label = labels[batch_idx];
    
    // Shared memory用于规约
    extern __shared__ float sdata[];
    
    // 1. 找最大值（warp reduction）
    float max_val = -INFINITY;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        max_val = fmaxf(max_val, logits_i[j]);
    }
    max_val = warp_reduce_max(max_val);  // 自定义warp reduction
    
    // 2. 计算exp和sum（warp reduction）
    float sum_exp = 0.0f;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float exp_val = expf(logits_i[j] - max_val);
        sdata[j] = exp_val;  // 保存到shared memory
        sum_exp += exp_val;
    }
    sum_exp = warp_reduce_sum(sum_exp);  // 自定义warp reduction
    
    // 3. 归一化得到softmax概率
    float log_sum_exp = max_val + logf(sum_exp);
    for (int j = tid; j < num_classes; j += blockDim.x) {
        probs_i[j] = sdata[j] / sum_exp;
    }
    
    // 4. 计算CE loss: -log(softmax_probs[label])
    float sample_loss = -logits_i[label] + log_sum_exp;
    
    // 5. 找argmax（warp reduction）
    int pred = argmax(logits_i, num_classes, tid);
    predicted_labels[batch_idx] = pred;
    
    // 6. 统计准确率
    if (tid == 0) {
        atomicAdd(ce_loss, sample_loss);
        if (pred == label) atomicAdd(top1_correct, 1);
        
        // Top-5准确率（需要额外逻辑）
        if (num_classes >= 5) {
            bool in_top5 = check_top5(logits_i, label, num_classes);
            if (in_top5) atomicAdd(top5_correct, 1);
        }
    }
    
    // 7. Block结束后归一化loss
    if (blockIdx.x == 0 && tid == 0) {
        *ce_loss = (*ce_loss / batch_size) * scaling_factor;
    }
}
```

**关键优化**：
- Warp reduction替代shared memory原子操作
- 每个block处理一个样本，避免跨block同步
- softmax_probs输出到global memory供BWD复用

#### 5.2.2 反向Kernel（FP32）

```cuda
__global__ void softmax_ce_bwd_fp32_kernel(
    const float* __restrict__ softmax_probs,  // [batch, num_classes]
    const int32_t* __restrict__ labels,        // [batch]
    float* __restrict__ d_logits,              // [batch, num_classes] 覆盖logits
    int batch_size,
    int num_classes,
    float scaling_factor)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    const float* probs_i = softmax_probs + batch_idx * num_classes;
    float* grad_i = d_logits + batch_idx * num_classes;
    int32_t label = labels[batch_idx];
    
    float scale = scaling_factor / batch_size;
    
    // 梯度 = softmax_probs - one_hot(label)
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float grad = probs_i[j];
        if (j == label) grad -= 1.0f;
        grad_i[j] = grad * scale;
    }
}
```

**关键优化**：
- 直接复用FWD的softmax_probs，无需重复计算
- 简单的逐元素操作，无需规约
- 梯度直接覆盖logits存储位置

#### 5.2.3 AMP Kernel（FP16输入/输出）

**前向（FP16→FP32→FP32）**：

```cuda
__global__ void softmax_ce_amp_fwd_kernel(
    const half* __restrict__ logits_fp16,     // [batch, num_classes] FP16
    const int32_t* __restrict__ labels,
    float* __restrict__ ce_loss,
    float* __restrict__ softmax_probs,        // FP32输出（BWD复用）
    int32_t* __restrict__ predicted_labels,
    int32_t* __restrict__ top1_correct,
    int32_t* __restrict__ top5_correct,
    int batch_size,
    int num_classes,
    float scaling_factor)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // 1. 加载FP16并转换为FP32
    const half* logits_i_fp16 = logits_fp16 + batch_idx * num_classes;
    float logits_i[128];  // 假设max 128 classes（栈分配）
    
    for (int j = tid; j < num_classes; j += blockDim.x) {
        logits_i[j] = __half2float(logits_i_fp16[j]);
    }
    
    // 2. 后续逻辑与FP32版本完全相同
    // ... (省略，参考FP32版本)
}
```

**反向（FP32→FP16）**：

```cuda
__global__ void softmax_ce_amp_bwd_kernel(
    const float* __restrict__ softmax_probs,  // [batch, num_classes] FP32
    const int32_t* __restrict__ labels,
    half* __restrict__ d_logits_fp16,         // [batch, num_classes] FP16输出
    int batch_size,
    int num_classes,
    float scaling_factor)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* probs_i = softmax_probs + batch_idx * num_classes;
    int32_t label = labels[batch_idx];
    
    float scale = scaling_factor / batch_size;
    
    // 计算FP32梯度
    float grad[128];
    for (int j = tid; j < num_classes; j += blockDim.x) {
        grad[j] = (probs_i[j] - (j == label ? 1.0f : 0.0f)) * scale;
    }
    
    // 转换为FP16并写入
    half* grad_i_fp16 = d_logits_fp16 + batch_idx * num_classes;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        grad_i_fp16[j] = __float2half(grad[j]);
    }
}
```

**关键**：
- 内部计算全FP32
- 仅在输入/输出边界做FP16↔FP32转换

#### 5.2.4 辅助函数

```cuda
// Warp reduction for max
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Warp reduction for sum
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Argmax
__device__ int argmax(const float* data, int n, int tid) {
    int max_idx = 0;
    float max_val = data[0];
    
    for (int j = tid; j < n; j += blockDim.x) {
        if (data[j] > max_val) {
            max_val = data[j];
            max_idx = j;
        }
    }
    
    // Warp reduction找最大索引
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }
    
    return max_idx;
}

// 检查label是否在top-5
__device__ bool check_top5(const float* logits, int label, int num_classes) {
    // 简化实现：找前5个最大值
    float top5_vals[5] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    
    for (int j = 0; j < num_classes; ++j) {
        if (logits[j] > top5_vals[4]) {
            top5_vals[4] = logits[j];
            // 冒泡排序
            for (int k = 4; k > 0 && top5_vals[k] > top5_vals[k-1]; --k) {
                float tmp = top5_vals[k];
                top5_vals[k] = top5_vals[k-1];
                top5_vals[k-1] = tmp;
            }
        }
    }
    
    return logits[label] >= top5_vals[4];
}
```

### 5.3 CUDA Launch函数

```cpp
static void launch_softmax_ce_fp32_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<SoftmaxCEParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "Missing SoftmaxCEParams");
    
    const DTensor& dt_logits = mp.get_dtensor(node.input_ids[0]);
    int batch_size = dt_logits.shape().n();
    int num_classes = p->num_classes;
    
    const float* logits = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const int32_t* labels = static_cast<const int32_t*>(ctx.ptr_at(node.input_ids[1]));
    
    float* ce_loss = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    float* softmax_probs = static_cast<float*>(ctx.ptr_at(node.output_ids[1]));
    int32_t* predicted_labels = static_cast<int32_t*>(ctx.ptr_at(node.output_ids[6]));
    int32_t* top1_correct = static_cast<int32_t*>(ctx.ptr_at(node.output_ids[4]));
    int32_t* top5_correct = static_cast<int32_t*>(ctx.ptr_at(node.output_ids[5]));
    
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    
    // 初始化标量输出为0
    cudaMemsetAsync(ce_loss, 0, sizeof(float), s);
    cudaMemsetAsync(top1_correct, 0, sizeof(int32_t), s);
    cudaMemsetAsync(top5_correct, 0, sizeof(int32_t), s);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = batch_size;
    size_t shared_mem_size = num_classes * sizeof(float);
    
    softmax_ce_fwd_fp32_kernel<<<grid_size, block_size, shared_mem_size, s>>>(
        logits, labels, ce_loss, softmax_probs, predicted_labels,
        top1_correct, top5_correct, batch_size, num_classes, p->scaling_factor);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("SOFTMAX_CE_FP32_FWD kernel failed: " << cudaGetErrorString(err));
    }
    
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

### 5.4 算子注册

```cpp
void register_op_softmax_ce() {
    // ===== SOFTMAX_CE_FP32_FWD =====
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
    
    // ===== SOFTMAX_CE_FP32_BWD =====
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
    
    // ===== SOFTMAX_CE_AMP_FWD (CUDA only) =====
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_FWD;
        e.launch_cpu = launch_softmax_ce_amp_fwd_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_fwd_cuda;
#endif
    }
    
    // ===== SOFTMAX_CE_AMP_BWD (CUDA only) =====
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

---

## 六、LayerDescriptor修改

### 6.1 更新`infer_softmaxce_tensors`

```cpp
std::vector<TensorDesc> infer_softmaxce_tensors(
    const Shape& input, const OpParams& p, const InferContext& ctx)
{
    // input shape: [batch, 1, 1, num_classes]
    int batch = input.n();
    const auto* lp = std::get_if<SoftmaxCEParams>(&p.data);
    int num_classes = lp ? lp->num_classes : input.c();
    
    return {
        // 输入
        TensorDesc{"logits", input, Region::F_FEATURE_FP32, DType::FP32},
        TensorDesc{"labels", Shape{batch,1,1,1}, Region::I_A_LABEL, DType::INT32},
        
        // 输出
        TensorDesc{"ce_loss", Shape{1,1,1,1}, Region::S_SCALAR_FP32, DType::FP32},
        TensorDesc{"softmax_probs", input, Region::T_TEMP_FP32, DType::FP32},
        TensorDesc{"scaling_factor", Shape{1,1,1,1}, Region::S_SCALAR_FP32, DType::FP32},
        TensorDesc{"inv_scaling", Shape{1,1,1,1}, Region::S_SCALAR_FP32, DType::FP32},
        TensorDesc{"top1_correct", Shape{1,1,1,1}, Region::R_TOP1_CORRECT, DType::INT32},
        TensorDesc{"top5_correct", Shape{1,1,1,1}, Region::R_TOP5_CORRECT, DType::INT32},
        TensorDesc{"predicted_labels", Shape{batch,1,1,1}, Region::R_PREDICTED_LABEL, DType::INT32},
    };
}
```

### 6.2 更新`build_softmaxce_forward`

```cpp
SubgraphPattern build_softmaxce_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    
    auto* fwd_node = p.add_node(GraphNode::Kind::COMPUTE);
    fwd_node->compute_op = ComputeOp::SOFTMAX_CE_FP32_FWD;
    fwd_node->params = SoftmaxCEParams(/*num_classes*/, /*scaling_factor*/).to_op_params();
    
    // 输入：logits + labels
    fwd_node->input_ids = {0, 1};  // logits(input_ids[0]) + labels(input_ids[1])
    
    // 输出：7个张量
    fwd_node->output_ids = {2, 3, 4, 5, 6, 7, 8};  // ce_loss + softmax_probs + scaling + inv_scaling + top1 + top5 + pred_labels
    
    return p;
}
```

### 6.3 更新`build_softmaxce_backward`

```cpp
SubgraphPattern build_softmaxce_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    
    auto* bwd_node = p.add_node(GraphNode::Kind::COMPUTE);
    bwd_node->compute_op = ComputeOp::SOFTMAX_CE_FP32_BWD;
    bwd_node->params = SoftmaxCEParams(/*num_classes*/, /*scaling_factor*/).to_op_params();
    
    // 输入：softmax_probs + labels + scaling + inv_scaling
    bwd_node->input_ids = {3, 1, 4, 5};  // softmax_probs + labels + scaling + inv_scaling
    
    // 输出：d_logits（覆盖logits存储位置）
    bwd_node->output_ids = {0};  // logits
    
    return p;
}
```

---

## 七、实施步骤

### 阶段1：基础设施修改（2-3小时）

1. **op_kind.h**: 新增4个ComputeOp枚举值
2. **types.h**: 新增3个Region枚举值，更新NUM_REGIONS=69
3. **op_kind.cpp**: 更新compute_op_to_string函数

### 阶段2：LayerDescriptor更新（1-2小时）

1. **layer_descriptor_registry.cpp**: 
   - 更新infer_softmaxce_tensors
   - 更新build_softmaxce_forward/backward

### 阶段3：算子实现（8-12小时）

#### 3.1 CPU实现（3-4小时）
1. 创建softmax_ce_op.cpp
2. 实现FWD/BWD的CPU版本
3. 添加AMP占位函数

#### 3.2 CUDA实现（5-8小时）
1. 创建softmax_ce_op.cu
2. 实现FWD/BWD的FP32 kernel
3. 实现FWD/BWD的AMP kernel
4. 实现warp reduction等辅助函数

#### 3.3 CUDA Launch函数（2-3小时）
1. 实现FWD/BWD的FP32 launch函数
2. 实现FWD/BWD的AMP launch函数

### 阶段4：注册与集成（1-2小时）

1. 实现register_op_softmax_ce()
2. 在op_registry.cpp中调用注册函数
3. 更新CMakeLists.txt添加新文件

### 阶段5：测试验证（4-6小时）

#### 5.1 单元测试
1. 数值正确性：对比PyTorch输出
2. 梯度检查：finite difference验证
3. Top-1/Top-5准确率验证

#### 5.2 集成测试
1. 完整训练流程验证
2. Loss收敛性检查
3. 性能profiling

#### 5.3 边界测试
1. batch_size=1
2. num_classes=2（最小类别数）
3. num_classes=10000（大类别数）
4. logits全相同/极端值

**总耗时估算**：16-25小时（2-3个工作日）

---

## 八、性能优化建议

### 8.1 CPU优化

1. **SIMD向量化**：使用AVX2/AVX-512加速exp/log
2. **缓存友好**：确保logits按行连续访问
3. **OpenMP调优**：根据batch_size动态调整线程数

### 8.2 CUDA优化

1. **Warp reduction**：避免shared memory原子操作
2. **循环展开**：减少分支预测失败
3. **Tensor Core**：AMP模式利用Tensor Core加速FP16→FP32转换
4. **内存合并访问**：确保coalesced global memory访问

### 8.3 数值优化

1. **Kahan求和**：提高大batch下的精度
2. **log/exp查表**：对固定num_classes可用查表加速
3. **混合精度**：AMP模式探索TF32支持（A100+）

---

## 九、风险与注意事项

### 9.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 数值溢出/下溢 | 中 | 高 | 严格使用log-sum-exp技巧 |
| Top-5实现复杂度高 | 低 | 中 | 优先完成Top-1，后续补充 |
| AMP精度损失 | 低 | 中 | 严格FP32中间计算 |
| Warp reduction正确性 | 中 | 高 | 详细单元测试验证 |
| 内存布局冲突 | 低 | 中 | 仔细设计Region分配 |

### 9.2 架构兼容性

1. **Region冲突**：新增R-Series需验证与现有MemoryPlan兼容
2. **编译器子图**：需要更新infer/build_softmaxce_*函数
3. **梯度覆盖约定**：确保d_logits物理上覆盖logits
4. **流策略**：使用COMP_1流，避免与其他算子冲突

### 9.3 测试策略

1. **数值验证**：对比PyTorch的F.cross_entropy + softmax
2. **梯度检查**：使用finite difference验证BWD正确性
3. **性能对比**：对比分离实现（softmax + ce）的性能提升
4. **边界测试**：极端batch_size、num_classes下的行为

---

## 十、总结

### 10.1 核心优势

1. **融合设计**：避免one-hot展开，节省显存和带宽
2. **数值稳定**：使用log-sum-exp技巧
3. **性能优化**：FWD输出softmax_probs供BWD复用
4. **架构对齐**：遵循现有算子的实现模式

### 10.2 关键决策

1. **新增Region**：R-Series存放推理结果（语义清晰）
2. **新增ComputeOp**：SOFTMAX_CE_*系列（避免破坏现有API）
3. **Temp区复用**：softmax_probs供BWD复用（性能优化）
4. **不展开one-hot**：CUDA/CPU都避免创建one-hot张量

### 10.3 预期收益

- **显存节省**：不展开one-hot，节省batch×num_classes×4字节
- **性能提升**：融合减少内存访问，预期加速1.5-2×
- **数值稳定**：log-sum-exp避免overflow/underflow
- **功能完整**：Top-1/Top-5准确率，预测标签输出

**实施优先级**：高（核心损失函数，影响所有训练任务）

建议按阶段逐步实施，优先完成FP32版本并验证正确性，后续补充AMP版本和性能优化。
