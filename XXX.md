

# 【今日话题：SOFTMAX_CE算子的实现】

# 【导言】

我们需要实现4个SOFTMAX_CE相关的算子：

SOFTMAX_CE_FP32_FWD
SOFTMAX_CE_FP32_BWD
SOFTMAX_CE_AMP_FWD
SOFTMAX_CE_AMP_BWD

我们框架一期强制使用交叉熵损失，所以softmax和crossentropy总是连在一起的，这是一个融合算子





CUDA绝对不要展开为one-hot，浪费内存和时间
CPU也尽量不要，除非真的没有更好的办法，或者说展开放在workspace能做到更快



logits[batch_size,1,1,num_classes]（AMP下是FP16，否则就是FP32）

scaling factor（FP32）

标签（INT32），在输入缓冲区，是一个DTensor，但其元素必定在内存上紧凑、连续，且有效元素数就等于batch size

FWD版：
根据logits和标签，计算LOSS（标量）
计算出TOP-1正确数和TOP-5正确数和推理结果标号值（INT32，就是要用来与标签对比的那个），放在结果region（好像现在还缺这个region，需要新增，它只放结果张量INT32，TOP-1正确数一个DTensor、TOP-5正确数另一个DTensor）



BWD版：
求出grad，覆盖logits（如果没法一个操作完成或者会污染输入，那可以先计算出放在另一个DTensor，然后再复制到logits）
真的，我们不缺显存空间，主要是担心读写次数，需要节约显存带宽
因为我们框架的约定就是特征图梯度覆盖输入特征图、dX覆盖X。
梯度需要乘以scaling factor（这是一个被分配到scalar区的DTensor，需要显式指定它作为输入张量之一）之后再写入到logits所在的张量

特别强调，无论是AMP还是FP32，Softmax和CrossEntropy的运算和中间张量都必须是FP32。也就是说，对于AMP的情形，也就只有logits输入是FP16、最终的grad要转为FP16再写入，但除此之外全都是FP32运算。
因为Softmax和CrossEntropy是复杂操作而且带规约，用FP16极其容易出错。

实现要好好参考其他算子，比如FC算子和ReLU算子。

如果涉及新建中间张量，应该申请Temp区的DTensor，而不是放在workspace（workspace的含义是执行一个单独运算时需要的中间缓冲区，用于cuDNN的算子或CPU的单个运算需要的中间缓存，但对于有意义的张量，最好使用Temp区，避免跟某个流的workspace冲突）
简单的判断准则：CUDA的情况，你只需要指定每个流的workspace，而不需要手动去往里面写东西，不要把它作为任何算子的输出；CPU的情况，与CUDA相似，但是如果单算子的执行需要用到中间缓存（比如你求一个矩阵乘法）来存储无明确意义的中间数据，就用workspace。





# 【小伙伴S】



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

| 约束                   | 说明                                 | 严重性 |
| ---------------------- | ------------------------------------ | ------ |
| **绝对不展开one-hot**  | CUDA和CPU都避免创建one-hot张量       | 🔴 强制 |
| **中间运算全FP32**     | 无论FP32/AMP，softmax+CE计算必须FP32 | 🔴 强制 |
| **梯度覆盖输入**       | d_logits覆盖logits存储位置           | 🔴 强制 |
| **Temp区存有意义张量** | 不用workspace存softmax_probs等       | 🔴 强制 |
| **数值稳定**           | 使用log-sum-exp技巧                  | 🔴 强制 |
| **CUDA手写kernel**     | 不用cuDNN FE（融合更高效）           | 🟡 建议 |

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

| 风险                 | 概率 | 影响 | 缓解措施                |
| -------------------- | ---- | ---- | ----------------------- |
| 数值溢出/下溢        | 中   | 高   | 严格使用log-sum-exp技巧 |
| Top-5实现复杂度高    | 低   | 中   | 优先完成Top-1，后续补充 |
| AMP精度损失          | 低   | 中   | 严格FP32中间计算        |
| Warp reduction正确性 | 中   | 高   | 详细单元测试验证        |
| 内存布局冲突         | 低   | 中   | 仔细设计Region分配      |

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

# 【小伙伴K】



> 版本: V1.0  
> 日期: 2026-05-16  
> 状态: 待实施

---

## 一、核心决策

| 问题        | 决策                                                         |
| ----------- | ------------------------------------------------------------ |
| 算子命名    | **替换**旧枚举：`CROSS_ENTROPY_LOSS*` → `SOFTMAX_CE_*`（4个） |
| labels 来源 | 数据加载器写入 `I_A_LABEL` / `I_B_LABEL`，算子**直接读取**不复制 |
| labels 传递 | `MemoryPlan` 显式保存输入缓冲区 DTensor ID，`Compiler` 注入到 GraphNode.input_ids |
| 结果区      | **新增** `R_METRIC_INT32`（Top-1/Top-5 正确数），`predicted_labels` 放 `T_TEMP_INT32` |
| FWD 中间量  | `softmax_probs` 输出到 `T_TEMP_FP32`（供 BWD 复用），**不走 workspace** |
| BWD 输出    | grad **in-place 覆盖 logits**（框架 dX 覆盖 X 约定），乘以 scaling_factor |
| 数值精度    | **无论 FP32/AMP，内部计算全 FP32**；AMP 仅 logits 输入/grad 输出做 FP16↔FP32 转换 |
| CUDA 路径   | **手写融合 kernel**，不用 cuDNN FE（无对应融合算子，且需输出 top1/top5） |
| CPU 路径    | 朴素循环实现，**不展开 one-hot**，不用 Eigen（无矩阵运算）   |

---

## 二、需要修改的文件（10个）

| #    | 文件                                        | 改动性质                                                     | 行数估算 |
| ---- | ------------------------------------------- | ------------------------------------------------------------ | -------- |
| 1    | `include/renaissance/core/types.h`          | 新增 `R_METRIC_INT32` Region；`NUM_REGIONS` +1               | +3       |
| 2    | `include/renaissance/graph/memory_plan.h`   | 新增输入缓冲区 ID 存储与访问接口                             | +10      |
| 3    | `src/graph/memory_plan.cpp`                 | `alloc_input_buffers` 保存 ID；`region_to_string` 新增 case  | ~10      |
| 4    | `include/renaissance/graph/op_kind.h`       | 删除 5 个旧 `CROSS_ENTROPY_LOSS*`；新增 4 个 `SOFTMAX_CE_*`  | ~10      |
| 5    | `src/graph/op_kind.cpp`                     | 字符串映射同步替换                                           | ~10      |
| 6    | `src/graph/layer_descriptor_registry.cpp`   | 重写 `infer/build_softmaxce_*`；扩展 TensorDesc 列表         | ~40      |
| 7    | `src/graph/compiler.cpp`                    | Phase 4 为 SoftmaxCE 注入 labels ID 到 input_ids             | ~15      |
| 8    | `src/backend/ops/dtensor/softmax_ce_op.cpp` | **新文件**：CPU launch + CUDA launch 分发 + 注册             | ~250     |
| 9    | `src/backend/ops/dtensor/softmax_ce_op.cu`  | **新文件**：CUDA 融合 kernels（FP32 + AMP）                  | ~280     |
| 10   | `src/CMakeLists.txt`                        | 添加 `.cpp` / `.cu`                                          | +2       |
| 11   | `include/renaissance/backend/op_registry.h` | 声明 `register_op_softmax_ce()`                              | +1       |
| 12   | `src/backend/op_registry.cpp`               | 调用注册；`require_warmup` 判断（手写 kernel 不需要 warmup） | +2       |
| 13   | `src/backend/op_stream_policy.cpp`          | 4 个新算子的流策略（`COMP_1`）                               | +5       |

---

## 三、基础设施改动

### 3.1 MemoryPlan —— 保存输入缓冲区 ID

**问题**：当前 `alloc_input_buffers()` 返回的 `InputBuffers` 被 `compiler.cpp` 丢弃，labels 的 DTensor ID 无处可查。

**方案**：`MemoryPlan` 新增成员保存这 4 个特殊 ID：

```cpp
// memory_plan.h —— MemoryPlan 类内新增
int32_t input_label_a_id_ = -1;
int32_t input_data_a_id_  = -1;
int32_t input_label_b_id_ = -1;
int32_t input_data_b_id_  = -1;

public:
[[nodiscard]] int32_t input_label_a_id() const noexcept { return input_label_a_id_; }
[[nodiscard]] int32_t input_data_a_id()  const noexcept { return input_data_a_id_; }
[[nodiscard]] int32_t input_label_b_id() const noexcept { return input_label_b_id_; }
[[nodiscard]] int32_t input_data_b_id()  const noexcept { return input_data_b_id_; }
```

```cpp
// memory_plan.cpp —— alloc_input_buffers 中保存
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

### 3.2 Region 枚举 —— 新增结果区

```cpp
// types.h
enum class Region : uint8_t {
    // ... 现有 65 个 ...
    T_TEMP_INT8,          // 065
    R_METRIC_INT32,       // 066  ← 新增：Top-1/Top-5 正确数、其他推理指标
    DEFAULT = B_PREV_MEAN,
    NUM_REGIONS = 66      // 65 → 66
};
```

同步更新 `memory_plan.cpp` 的 `region_to_string`。

### 3.3 ComputeOp 枚举 —— 替换旧名

```cpp
// op_kind.h
enum class ComputeOp : uint16_t {
    // ... 其他算子 ...

    // 删除以下 5 个旧枚举（及其在 op_kind.cpp / format_params 中的 case）：
    // CROSS_ENTROPY_LOSS,
    // CROSS_ENTROPY_LOSS_FP32_FWD,
    // CROSS_ENTROPY_LOSS_FP32_BWD,
    // CROSS_ENTROPY_LOSS_AMP_FWD,
    // CROSS_ENTROPY_LOSS_AMP_BWD,

    // 新增 4 个：
    SOFTMAX_CE_FP32_FWD,
    SOFTMAX_CE_FP32_BWD,
    SOFTMAX_CE_AMP_FWD,
    SOFTMAX_CE_AMP_BWD,

    ALLREDUCE_SUM,
    // ...
};
```

> **注意**：`LossParams` 结构体**保留复用**，无需新增 `SoftmaxCEParams`。`num_classes` 和 `label_smoothing` 已足够。

---

## 四、LayerDescriptor 与 Compiler 改动

### 4.1 TensorDesc 列表（8张量）

```cpp
std::vector<TensorDesc> infer_softmaxce_tensors(
    const Shape& input, const OpParams& p, const InferContext& ctx)
{
    int batch = input.n();
    const auto* lp = std::get_if<LossParams>(&p.data);
    int num_classes = lp ? lp->num_classes : input.c();
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    return {
        // 0: loss（标量）
        TensorDesc{"ce_loss",            Shape{1,1,1,1},      Region::S_SCALAR_FP32,   DType::FP32},
        // 1: ce_output / logits（输入来自前一层，输出 softmax_probs 或 grad 覆盖）
        TensorDesc{"ce_output",          input,               select_feature_region(ctx), feat_dt},
        // 2: scaling_factor（标量）
        TensorDesc{"scaling_factor",     Shape{1,1,1,1},      Region::S_SCALAR_FP32,   DType::FP32},
        // 3: inv_scaling_factor（标量）
        TensorDesc{"inv_scaling_factor", Shape{1,1,1,1},      Region::S_SCALAR_FP32,   DType::FP32},
        // 4: top1_correct（标量 INT32）
        TensorDesc{"top1_correct",       Shape{1,1,1,1},      Region::R_METRIC_INT32,  DType::INT32},
        // 5: top5_correct（标量 INT32）
        TensorDesc{"top5_correct",       Shape{1,1,1,1},      Region::R_METRIC_INT32,  DType::INT32},
        // 6: predicted_labels（batch 个 INT32）
        TensorDesc{"predicted_labels",   Shape{batch,1,1,1},  Region::T_TEMP_INT32,    DType::INT32},
        // 7: softmax_probs（FWD 输出供 BWD 复用）
        TensorDesc{"softmax_probs",      input,               Region::T_TEMP_FP32,     DType::FP32},
    };
}
```

### 4.2 SubgraphPattern

```cpp
SubgraphPattern build_softmaxce_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_FWD
               : ComputeOp::SOFTMAX_CE_FP32_FWD;
    // input_indices 仅含 descs 内部张量：scaling_factor(2)
    // labels 由 Compiler 显式注入（见 4.3）
    n.input_indices  = {2};           // scaling_factor
    n.output_indices = {0, 4, 5, 6, 7}; // loss + top1 + top5 + pred + probs
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_softmaxce_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_BWD
               : ComputeOp::SOFTMAX_CE_FP32_BWD;
    // BWD 读取：softmax_probs(7) + scaling_factor(2)
    // labels 由 Compiler 显式注入
    n.input_indices  = {7, 2};
    // 输出：grad 覆盖 ce_output(1) —— 通过 idx=1 让 compiler 映射到 logits 张量
    n.output_indices = {1};
    p.nodes.push_back(n);
    return p;
}
```

### 4.3 Compiler —— 注入 labels

在 `compiler.cpp` Phase 4 的 Forward 和 Backward 循环中，为 `SoftmaxCE` 层追加 labels ID：

```cpp
// Forward 循环内，构建完 gn 后、append 前：
if (layer.kind == LayerKind::SoftmaxCE) {
    int32_t label_id = memory_plan.input_label_a_id();  // 训练用 A 桶
    if (label_id >= 0) {
        gn.input_ids.push_back(label_id);
    }
}

// Backward 循环内，同理：
if (layer.kind == LayerKind::SoftmaxCE) {
    int32_t label_id = memory_plan.input_label_a_id();
    if (label_id >= 0) {
        gn.input_ids.push_back(label_id);
    }
}
```

> **注入后的 input_ids 顺序**：
>
> - FWD: `[prev_output_id(logits), scaling_factor_tid, label_tid]`
> - BWD: `[prev_grad_id, softmax_probs_tid, scaling_factor_tid, label_tid]`

---

## 五、算子 I/O 约定

### 5.1 FWD（FP32 与 AMP 共用结构）

| 接口          | 索引 | 内容             | 说明                               |
| ------------- | ---- | ---------------- | ---------------------------------- |
| input_ids[0]  | —    | logits           | 跨层链注入，FP32(AMP下FP16)        |
| input_ids[1]  | —    | scaling_factor   | S_SCALAR_FP32，标量                |
| input_ids[2]  | —    | labels           | I_A_LABEL，INT32，batch 个紧凑连续 |
| output_ids[0] | —    | ce_loss          | S_SCALAR_FP32，标量                |
| output_ids[1] | —    | top1_correct     | R_METRIC_INT32，标量               |
| output_ids[2] | —    | top5_correct     | R_METRIC_INT32，标量               |
| output_ids[3] | —    | predicted_labels | T_TEMP_INT32，[batch]              |
| output_ids[4] | —    | softmax_probs    | T_TEMP_FP32，[batch, num_classes]  |

### 5.2 BWD

| 接口          | 索引 | 内容               | 说明                           |
| ------------- | ---- | ------------------ | ------------------------------ |
| input_ids[0]  | —    | prev_grad / logits | 跨层链注入，BWD 覆盖目标       |
| input_ids[1]  | —    | softmax_probs      | T_TEMP_FP32，FWD 输出          |
| input_ids[2]  | —    | scaling_factor     | S_SCALAR_FP32                  |
| input_ids[3]  | —    | labels             | I_A_LABEL，INT32               |
| output_ids[0] | —    | grad_logits        | **in-place 覆盖 input_ids[0]** |

---

## 六、CUDA Kernel 设计

### 6.1 FWD Kernel（模板化 FP32/AMP）

```cuda
template<bool IS_AMP>
__global__ void softmax_ce_fwd_kernel(
    const float*  logits_fp32,   // IS_AMP=false
    const __half* logits_fp16,   // IS_AMP=true
    const int*    labels,
    float*        loss,          // 标量，atomicAdd 累加
    int*          top1_correct,  // 标量，atomicAdd
    int*          top5_correct,  // 标量，atomicAdd
    int*          predicted_labels,  // [batch]
    float*        softmax_probs, // [batch, num_classes] Temp区
    float         scaling_factor,
    int           batch,
    int           num_classes)
{
    // 每个 block 处理一个 sample
    int b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ float smem[];
    float* s_logits = smem;  // [blockSize] 或 [num_classes] 视策略而定

    const int tid = threadIdx.x;
    int label = labels[b];

    // 1. 加载 logits 到 FP32 寄存器/共享内存，同时求 local max
    float local_max = -INFINITY;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        float v = IS_AMP ? __half2float(logits_fp16[b * num_classes + c])
                         : logits_fp32[b * num_classes + c];
        // 可写入 smem 供后续复用
        smem[c] = v;  // 若 num_classes <= max_shared_mem / 4
        local_max = fmaxf(local_max, v);
    }

    // 2. Warp + Block 归约求 global_max
    // ... (shuffle + tree reduction) ...
    float global_max = ...;

    // 3. 计算 exp 和 sum_exp
    float local_sum = 0.0f;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        float e = expf(smem[c] - global_max);
        smem[c] = e;  // 复用 smem 存 exp
        local_sum += e;
    }
    // Block 归约求 sum_exp
    float sum_exp = ...;

    // 4. 归一化写入 softmax_probs + 维护 Top-5
    Top5 local_top5;  // 寄存器内小根堆/插入排序，5个元素
    for (int c = tid; c < num_classes; c += blockDim.x) {
        float prob = smem[c] / sum_exp;
        softmax_probs[b * num_classes + c] = prob;
        local_top5.push(prob, c);  // 或直接用 logits 值维护 top5
    }
    // Warp 级 Top-5 合并 ...

    // 5. Thread 0 计算 loss、top1、top5、predicted_label
    if (tid == 0) {
        float label_prob = smem[label] / sum_exp;  // 注意：若 smem 被覆盖为 exp，需用原始 logits
        // 更稳妥：ce = -(logits[label] - global_max) + log(sum_exp)
        float ce = -smem_original[label] + global_max + logf(sum_exp);
        atomicAdd(loss, ce);

        int pred = ...;  // 从 Top-5 取第 0 个
        predicted_labels[b] = pred;
        if (pred == label) atomicAdd(top1_correct, 1);
        if (top5_contains_label) atomicAdd(top5_correct, 1);
    }
}
```

**关键优化**：

- **数值稳定**：`ce = -logits[label] + max + log(sum_exp)`，不通过 `log(prob)` 间接计算
- **共享内存布局**：若 `num_classes=1000`，需 `1000*4=4KB`，单 block 轻松容纳
- **Top-5**：每线程维护局部 Top-5，warp shuffle 合并，block 级再合并
- **初始化**：`loss/top1/top5` 标量需在 kernel launch 前 `cudaMemsetAsync(..., 0)`
- **Block 大小**：根据 num_classes 动态选择（128/256/512）

### 6.2 BWD Kernel

```cuda
template<bool IS_AMP>
__global__ void softmax_ce_bwd_kernel(
    const float* probs,      // [batch, num_classes] FWD输出的Temp区
    const int*   labels,
    float        scaling_factor,
    float*       grad_fp32,  // IS_AMP=false
    __half*      grad_fp16,  // IS_AMP=true
    int          batch,
    int          num_classes)
{
    int b = blockIdx.x;
    if (b >= batch) return;

    float scale = scaling_factor / batch;
    int label = labels[b];

    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        float g = probs[b * num_classes + c] * scale;
        if (c == label) g -= scale;
        if (IS_AMP)
            grad_fp16[b * num_classes + c] = __float2half(g);
        else
            grad_fp32[b * num_classes + c] = g;
    }
}
```

**特点**：

- 无需 shared memory reduction，每个线程独立计算
- 条件判断 `c == label` 替代 one-hot 展开
- AMP 版本仅在写入时做 `__float2half`

---

## 七、CPU 实现

### 7.1 FWD CPU

```cpp
static void launch_softmax_ce_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    float* logits = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float* scaling = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    int* labels = static_cast<int*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));

    float* loss = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    int* top1 = static_cast<int*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));
    int* top5 = static_cast<int*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[2]));
    int* pred = static_cast<int*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[3]));
    float* probs = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[4]));

    int batch = op_ctx->input_shape.n;
    int num_classes = op_ctx->input_shape.c;
    double loss_sum = 0.0;
    int t1 = 0, t5 = 0;

    for (int b = 0; b < batch; ++b) {
        float* row = logits + b * num_classes;
        float* prob_row = probs + b * num_classes;
        int label = labels[b];

        float max_val = row[0];
        for (int i = 1; i < num_classes; ++i) max_val = std::max(max_val, row[i]);

        double sum_exp = 0.0;
        for (int i = 0; i < num_classes; ++i) {
            prob_row[i] = std::exp(row[i] - max_val);
            sum_exp += prob_row[i];
        }
        for (int i = 0; i < num_classes; ++i) prob_row[i] /= (float)sum_exp;

        loss_sum += -(row[label] - max_val) + std::log(sum_exp);

        // Top-1
        int best_idx = 0;
        for (int i = 1; i < num_classes; ++i)
            if (prob_row[i] > prob_row[best_idx]) best_idx = i;
        pred[b] = best_idx;
        if (best_idx == label) ++t1;

        // Top-5：5次扫描找前5大（1000*5=5000次比较，可接受）
        bool in_top5 = false;
        for (int k = 0; k < 5; ++k) {
            int bi = -1;
            float bv = -1e30f;
            for (int i = 0; i < num_classes; ++i) {
                if (prob_row[i] > bv) { bv = prob_row[i]; bi = i; }
            }
            if (bi == label) in_top5 = true;
            prob_row[bi] = -1e30f;  // 标记已选
        }
        if (in_top5) ++t5;
    }

    loss[0] = (float)(loss_sum / batch * scaling[0]);
    top1[0] = t1;
    top5[0] = t5;
}
```

### 7.2 BWD CPU

```cpp
static void launch_softmax_ce_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    float* probs = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    float* scaling = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));
    int* labels = static_cast<int*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[3]));
    float* grad = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int batch = op_ctx->input_shape.n;
    int num_classes = op_ctx->input_shape.c;
    float scale = scaling[0] / batch;

    for (int b = 0; b < batch; ++b) {
        float* p = probs + b * num_classes;
        float* g = grad + b * num_classes;
        int label = labels[b];
        for (int i = 0; i < num_classes; ++i) {
            float v = p[i] * scale;
            if (i == label) v -= scale;
            g[i] = v;
        }
    }
}
```

### 7.3 AMP CPU —— 不支持

```cpp
static void launch_softmax_ce_amp_fwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_FWD is not supported on CPU");
}
```

---

## 八、CUDA Launch 分发层

```cpp
#ifdef TR_USE_CUDA
static void launch_softmax_ce_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const DTensor& dt_logits = mp.get_dtensor(node.input_ids[0]);
    int batch = dt_logits.shape().n();
    int num_classes = dt_logits.shape().c();

    const float* logits = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const int* labels   = static_cast<const int*>(ctx.ptr_at(node.input_ids[2]));

    float* loss = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    int* top1   = static_cast<int*>(ctx.ptr_at(node.output_ids[1]));
    int* top5   = static_cast<int*>(ctx.ptr_at(node.output_ids[2]));
    int* pred   = static_cast<int*>(ctx.ptr_at(node.output_ids[3]));
    float* probs = static_cast<float*>(ctx.ptr_at(node.output_ids[4]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;

    // 初始化累加标量为 0
    cudaMemsetAsync(loss, 0, sizeof(float), s);
    cudaMemsetAsync(top1, 0, sizeof(int), s);
    cudaMemsetAsync(top5, 0, sizeof(int), s);

    int block_size = (num_classes <= 128) ? 128 : (num_classes <= 512) ? 256 : 512;
    size_t shared_mem = num_classes * sizeof(float);  // 存 logits → exp

    softmax_ce_fwd_kernel<false><<<batch, block_size, shared_mem, s>>>(
        logits, nullptr, labels, loss, top1, top5, pred, probs,
        scaling_factor_value, batch, num_classes);

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif
```

> **scaling_factor_value 获取**：从 `node.input_ids[1]` 对应的 DTensor 读取。注意：在 CUDA graph capture 中，若 scaling_factor 是 host 常量，可直接传入 kernel 参数；若为 device 内存，需在 kernel 中再读一次。

---

## 九、注册

```cpp
void register_op_softmax_ce() {
    // FP32 FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_FWD;
        e.launch_cpu = launch_softmax_ce_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_fwd_cuda;
#endif
    }
    // FP32 BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_BWD;
        e.launch_cpu = launch_softmax_ce_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_bwd_cuda;
#endif
    }
    // AMP FWD / BWD（CPU 抛错）
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_FWD;
        e.launch_cpu = launch_softmax_ce_amp_fwd_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_fwd_cuda;
#endif
    }
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

## 十、验证计划

| 验证项        | 方法                                                        |
| ------------- | ----------------------------------------------------------- |
| 数值正确性    | 与 PyTorch `F.cross_entropy` + `F.softmax` 对比，MSE < 1e-5 |
| Top-1/Top-5   | 构造已知 logits/labels，验证正确数符合预期                  |
| 梯度正确性    | 与 PyTorch `loss.backward()` 对比 grad                      |
| AMP 精度      | FP16 输入下，内部 FP32 计算，结果与 FP32 版本 MSE < 1e-4    |
| In-place 覆盖 | 验证 BWD 后原 logits 内存已被 grad 覆盖                     |
| 零 one-hot    | 代码审查：CPU/CUDA 均无 `num_classes x batch` 的中间分配    |
| 编译通过      | MSVC /O2 + CUDA 13.1 全量编译无警告                         |

---

## 十一、实施顺序（推荐）

```
Phase 1（编译器框架）:
  → types.h (R_METRIC_INT32)
  → memory_plan.h/cpp (保存输入缓冲区 ID)
  → op_kind.h/cpp (替换枚举与字符串)
  → layer_descriptor_registry.cpp (扩展 TensorDesc)
  → compiler.cpp (注入 labels)
  → op_stream_policy.cpp (流策略)
  → 编译验证

Phase 2（算子实现）:
  → softmax_ce_op.cu (CUDA kernels)
  → softmax_ce_op.cpp (CPU + launch 分发 + 注册)
  → op_registry.h/cpp (调用注册)
  → CMakeLists.txt
  → 编译验证

Phase 3（测试）:
  → Python 参考脚本（对比 PyTorch）
  → C++ standalone 测试
  → Composite 端到端训练验证
```

---

## 十二、风险与缓解

| 风险                                       | 缓解                                                         |
| ------------------------------------------ | ------------------------------------------------------------ |
| `CROSS_ENTROPY_LOSS*` 被其他代码引用       | 全局 grep 替换；若遗留引用会导致编译错误，易发现             |
| Top-5 CUDA kernel 复杂度高                 | 先用 warp-shuffle + 寄存器 Top-5 实现；若超时再优化          |
| labels ID 注入时机                         | 确保在 `gn.input_ids` 已填入 descs 映射后、append 前注入     |
| AMP 下 logits 步幅非连续                   | kernel 中按 DTensor stride 访问（通过 `n_stride_cuda()` 等） |
| `cudaMemsetAsync` 在 CUDA Graph capture 中 | 使用 `cudaGraph` 兼容的初始化方式，或在 capture 前预清零     |



# 【小伙伴D】



**版本**: V4.0
**日期**: 2026-05-19
**基于**: SOF.md + SOFX.md（S/K/D 三份提案）+ 代码审计 + 用户补充要求

---

## 一、需求摘要

实现 4 个融合算子：

| 算子                  | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| `SOFTMAX_CE_FP32_FWD` | FP32 前向：logits + labels → loss + softmax probs + TOP-1/5 + 预测标签 |
| `SOFTMAX_CE_FP32_BWD` | FP32 反向：probs + labels → grad(logits) × scaling_factor    |
| `SOFTMAX_CE_AMP_FWD`  | AMP 前向：FP16 logits → FP32 内部 → loss + probs(FP32) + TOP-1/5 |
| `SOFTMAX_CE_AMP_BWD`  | AMP 反向：probs(FP32) → grad(FP16) × scaling_factor          |

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

| 模式                              | 参考                                                         |
| --------------------------------- | ------------------------------------------------------------ |
| CUDA 自定义 kernel（非 cuDNN FE） | [`relu_op.cu`](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cu) |
| CUDA half↔float 转换              | [`gap_op.cu:L54-L56`](file:///r:/renaissance/src/backend/ops/dtensor/gap_op.cu#L54-L56) |
| CPU Eigen 优化                    | [`fc_op.cpp:647-711`](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L647-L711) |
| 算子注册                          | `register_op_fc()` / `register_op_relu()`                    |
| 流策略                            | [`op_stream_policy.cpp`](file:///r:/renaissance/src/backend/op_stream_policy.cpp) |

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

| #    | 文件                                        | 改动性质                                                     | 行数 |
| ---- | ------------------------------------------- | ------------------------------------------------------------ | ---- |
| 1    | `include/renaissance/graph/op_kind.h`       | 删除 5 个 `CROSS_ENTROPY_LOSS_*`，新增 4 个 `SOFTMAX_CE_*`   | ~12  |
| 2    | `include/renaissance/core/types.h`          | 新增 `R_TOP1_CORRECT`、`R_TOP5_CORRECT`、`R_PREDICTED_LABEL`，更新 `NUM_REGIONS` | +4   |
| 3    | `src/graph/op_kind.cpp`                     | 更新 `compute_op_to_string`（删除 5 旧 case + 新增 4 新 case） | ~9   |
| 4    | `src/graph/layer_descriptor_registry.cpp`   | 重写 `infer_softmaxce_tensors` (4→8)、更新 forward/backward 子图 | ~55  |
| 5    | `src/graph/memory_plan.cpp`                 | 注册 3 个新 Region 的布局属性                                | ~9   |
| 6    | `src/backend/ops/dtensor/softmax_ce_op.cu`  | **新文件**：FWD + BWD CUDA kernels（FP32 + AMP）             | ~220 |
| 7    | `src/backend/ops/dtensor/softmax_ce_op.cpp` | **新文件**：CPU 实现 + CUDA launch 分发 + 注册               | ~280 |
| 8    | `include/renaissance/backend/op_registry.h` | 声明 `register_op_softmax_ce()`                              | +1   |
| 9    | `src/backend/op_registry.cpp`               | 调用 `register_op_softmax_ce()` + `require_warmup` 添加      | ~3   |
| 10   | `src/backend/op_stream_policy.cpp`          | 4 个 CE 算子的流策略                                         | +4   |
| 11   | `src/CMakeLists.txt`                        | 添加 `.cu` + `.cpp` 到源文件列表                             | +2   |

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

| 风险                                                       | 缓解                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| Shared memory 不足（num_classes=10000 需要 40KB）          | 用 `cudaFuncSetAttribute` 指定动态 SMEM；超出硬件限制时 fallback 到 global memory |
| TOP-5 线程 0 串行 O(C²)                                    | 对 C≤1000 足够快（1000×5=5000 次比较），C>10000 可改为 shared memory 分块 partial sort |
| `label_smoothing > 0`                                      | `LossParams` 已预留字段，V1 先实现 smoothing=0，V2 扩展      |
| 数值下溢（log(0)）                                         | 所有 `log` 前加 `max(val, 1e-12f)`                           |
| CUDA kernel 中 blockIdx.x 为 0 的线程做 loss reduce 的竞态 | 用单独的 reduction kernel 或 cub 做 post-kernel reduction    |
| 枚举值重排导致旧代码索引错位                               | 已验证：所有表项用 `ComputeOp::XXX` 常量，编译器自动适配     |

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



# 【用户补充】

我发现某些小伙伴花了大量的时间精力去纠结“label从何而来”。这是个很简单的问题！label只是一个普通的张量而已！只不过它的形状是[batch size,1,1,1]，必定compact，数据类型是INT32，区域是I_A_LABEL或I_B_LABEL！你设计算子的时候，只需要接受并处理张量，标签就是输入张量之一，它跟FC层的权重是一个道理！你的算子接受DTensor的传入，然后处理它，就这么简单！

另外你的SOFTMAX_CE系列算子毕竟只是一个普通算子，你要让它符合我们的算子设计的API风格！它充其量只是一个涉及的IO较多的算子而已！

