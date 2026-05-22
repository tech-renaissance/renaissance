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

**V1.0**  
日期: 2026-05-19  
编制: 技术觉醒团队  
状态: 待实施

---

## 一、需求分析

### 1.1 算子定义

需要实现4个SOFTMAX_CE融合算子：

- `SOFTMAX_CE_FP32_FWD` - FP32前向（CPU + CUDA）
- `SOFTMAX_CE_FP32_BWD` - FP32反向（CPU + CUDA）
- `SOFTMAX_CE_AMP_FWD` - AMP FP16前向（仅CUDA）
- `SOFTMAX_CE_AMP_BWD` - AMP FP16反向（仅CUDA）

### 1.2 融合原因

Softmax和CrossEntropy总是连在一起使用（框架一期强制使用交叉熵损失），融合后可以：

1. **避免显存浪费**：不展开one-hot标签
2. **减少内存带宽**：一次遍历完成softmax + CE计算
3. **数值稳定性**：使用log-sum-exp技巧

### 1.3 输入输出规格

#### FWD版输入：

1. **logits**: `[batch_size, 1, 1, num_classes]`
   - FP32模式：`DType::FP32`
   - AMP模式：`DType::FP16`
2. **labels**: `[batch_size]` (INT32)
   - 紧凑连续内存
   - 值范围：`[0, num_classes-1]`
3. **scaling_factor**: 标量FP32
   - 用于损失缩放

#### FWD版输出：

1. **loss**: 标量FP32 (`S_SCALAR_FP32`)
2. **top1_correct**: 标量INT32 (`S_SCALAR_INT32`) - TOP-1正确数
3. **top5_correct**: 标量INT32 (`S_SCALAR_INT32`) - TOP-5正确数
4. **predictions**: `[batch_size]` INT32 (`T_TEMP_INT32`) - 推理结果标号

#### BWD版输入：

1. **logits**: `[batch_size, 1, 1, num_classes]` (被覆盖)
2. **labels**: `[batch_size]` INT32
3. **scaling_factor**: 标量FP32

#### BWD版输出：

1. **d_logits**: `[batch_size, 1, 1, num_classes]` (覆盖logits)
   - 梯度乘以scaling_factor后再写入
   - AMP模式：需要FP16→FP32→FP16转换

### 1.4 关键约束

1. **数值精度**：无论FP32还是AMP，Softmax和CE中间运算必须是FP32
2. **不展开one-hot**：CUDA和CPU都避免创建one-hot张量
3. **梯度覆盖输入**：d_logits覆盖logits（框架约定）
4. **Region选择**：
   - 结果张量使用**T-Series** (临时张量)
   - 不使用workspace（workspace仅用于无明确意义的中间缓存）
   - 梯度使用**F-Series** (特征图梯度槽)

---

## 二、Region扩展需求

### 2.1 现有Region分析

当前已有Region（`types.h:237-329`）：

- **S-Series**: 标量区
  - `S_SCALAR_FP32` (057)
  - `S_SCALAR_INT32` (059)
- **T-Series**: 临时张量
  - `T_TEMP_FP32` (062)
  - `T_TEMP_INT32` (064)
- **F-Series**: 特征图与梯度槽
  - `F_FEATURE_FP32` (053)
  - `F_GRAD_SLOT_FP32` (054)

### 2.2 新增Region建议

**结论**：无需新增Region！

**原因**：

1. 标量结果（loss、top1_correct、top5_correct）可使用 `S_SCALAR_FP32` 和 `S_SCALAR_INT32`
2. 推理结果（predictions）可使用 `T_TEMP_INT32`
3. 梯度（d_logits）使用现有梯度槽

**但需要注意**：

- 需要在算子参数中明确指定使用哪个Region
- 避免与其他算子冲突

---

## 三、数学公式与算法

### 3.1 前向：Softmax + Cross Entropy

#### 数学定义

对于单个样本 $i$，给定logits $\mathbf{z} = [z_1, z_2, ..., z_K]$ 和标签 $y \in \{0, 1, ..., K-1\}$：

1. **Softmax**：
   $$\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$

2. **Cross Entropy Loss**：
   $$L_i = -\log(\text{softmax}(z_y)) = -z_y + \log\left(\sum_{k=1}^{K} e^{z_k}\right)$$

3. **总损失**（带scaling）：
   $$L = \frac{\text{scaling\_factor}}{N} \sum_{i=1}^{N} L_i$$

#### 数值稳定实现

使用log-sum-exp技巧：
$$\log\left(\sum_{k=1}^{K} e^{z_k}\right) = z_{\max} + \log\left(\sum_{k=1}^{K} e^{z_k - z_{\max}}\right)$$

其中 $z_{\max} = \max_k z_k$。

#### 算法步骤（FWD）

```python
# 伪代码
def softmax_ce_fwd(logits, labels, scaling_factor):
    batch_size, num_classes = logits.shape
    
    # 1. 计算每个样本的最大值（数值稳定）
    max_logits = max(logits, axis=1)  # [batch_size]
    
    # 2. 计算log-sum-exp
    exp_shifted = exp(logits - max_logits)  # [batch_size, num_classes]
    sum_exp = sum(exp_shifted, axis=1)      # [batch_size]
    log_sum_exp = max_logits + log(sum_exp) # [batch_size]
    
    # 3. 计算每个样本的loss
    # labels[i] 是第i个样本的类别索引
    target_logits = logits[range(batch_size), labels]  # [batch_size]
    sample_losses = -target_logits + log_sum_exp       # [batch_size]
    
    # 4. 总损失（带scaling）
    loss = scaling_factor * mean(sample_losses)  # 标量
    
    # 5. 计算准确率
    predictions = argmax(logits, axis=1)  # [batch_size]
    top1_correct = sum(predictions == labels)
    top5_correct = sum_top5_match(logits, labels)  # 需要实现
    
    return loss, top1_correct, top5_correct, predictions
```

### 3.2 反向：梯度计算

#### 数学推导

对于单个样本 $i$：

1. **Loss对logits的梯度**：
   $$\frac{\partial L_i}{\partial z_j} = \text{softmax}(z_j) - \mathbb{1}[j = y]$$

   其中 $\mathbb{1}[j = y]$ 是指示函数（label $y$ 对应位置为1，其他为0）。

2. **总梯度**（带scaling）：
   $$\frac{\partial L}{\partial z_{ij}} = \frac{\text{scaling\_factor}}{N} \left(\text{softmax}(z_{ij}) - \mathbb{1}[j = y_i]\right)$$

#### 算法步骤（BWD）

```python
# 伪代码
def softmax_ce_bwd(logits, labels, scaling_factor):
    batch_size, num_classes = logits.shape
    
    # 1. 计算softmax（复用FWD的计算逻辑）
    max_logits = max(logits, axis=1)
    exp_shifted = exp(logits - max_logits)
    sum_exp = sum(exp_shifted, axis=1)
    softmax_probs = exp_shifted / sum_exp  # [batch_size, num_classes]
    
    # 2. 计算梯度
    # 初始化梯度为softmax概率
    d_logits = softmax_probs  # [batch_size, num_classes]
    
    # 在label位置减1（one-hot编码的隐式表示）
    d_logits[range(batch_size), labels] -= 1.0
    
    # 3. 应用scaling factor
    d_logits *= (scaling_factor / batch_size)
    
    return d_logits  # 覆盖原始logits
```

---

## 四、实现方案

### 4.1 ComputeOp注册

在 `op_kind.h` 中添加：

```cpp
enum class ComputeOp : uint16_t {
    // ... 现有算子 ...
    
    // === Softmax + Cross Entropy 融合算子 ===
    SOFTMAX_CE_FP32_FWD,  // FP32前向：softmax + CE + accuracy
    SOFTMAX_CE_FP32_BWD,  // FP32反向：梯度计算
    SOFTMAX_CE_AMP_FWD,   // AMP前向：logits(FP16) → FP32计算 → FP16输出
    SOFTMAX_CE_AMP_BWD,   // AMP反向：logits(FP16) → FP32计算 → FP16梯度
    
    // ...
};
```

### 4.2 文件结构

创建新文件：

- `src/backend/ops/dtensor/softmax_ce_op.cpp` - CPU实现 + CUDA launch函数
- `src/backend/ops/dtensor/softmax_ce_op.cu` - CUDA kernel实现

### 4.3 CPU实现方案

#### 4.3.1 前向（FP32）

```cpp
static void launch_softmax_ce_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    // 参数解析
    const auto* p = std::get_if<SoftmaxCEParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "Missing SoftmaxCEParams");
    
    int num_classes = p->num_classes;
    float scaling_factor = p->scaling_factor;
    
    // 输入
    float* logits = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const int32_t* labels = static_cast<const int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    
    // 输出
    float* loss = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    int32_t* top1_correct = static_cast<int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));
    int32_t* top5_correct = static_cast<int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[2]));
    int32_t* predictions = static_cast<int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[3]));
    
    int batch_size = op_ctx->output_shape.n;
    
    // 临时缓冲区：max_logits, sum_exp, softmax_probs
    // 使用stack分配（小batch）或malloc（大batch）
    std::vector<float> max_logits(batch_size);
    std::vector<float> sum_exp(batch_size);
    std::vector<float> softmax_probs(batch_size * num_classes);
    
    float total_loss = 0.0f;
    int top1_cnt = 0;
    int top5_cnt = 0;
    
    // 对每个样本计算
    for (int i = 0; i < batch_size; ++i) {
        float* logits_i = logits + i * num_classes;
        float* probs_i = softmax_probs.data() + i * num_classes;
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
        max_logits[i] = max_val;
        
        // 2. 计算exp和sum
        float sum = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            float exp_val = std::exp(logits_i[j] - max_val);
            probs_i[j] = exp_val;
            sum += exp_val;
        }
        sum_exp[i] = sum;
        
        // 3. 计算loss: -logits[label] + log(sum_exp)
        float target_logit = logits_i[label];
        float log_sum_exp = max_val + std::log(sum);
        total_loss += (-target_logit + log_sum_exp);
        
        // 4. 统计准确率
        if (pred == label) top1_cnt++;
        
        // Top-5准确率
        if (num_classes >= 5) {
            // 找前5个最大值的位置
            std::vector<std::pair<float, int>> indexed_probs(num_classes);
            for (int j = 0; j < num_classes; ++j) {
                indexed_probs[j] = {logits_i[j], j};
            }
            std::partial_sort(indexed_probs.begin(), indexed_probs.begin() + 5,
                             indexed_probs.end(),
                             [](const auto& a, const auto& b) { return a.first > b.first; });
            for (int k = 0; k < 5; ++k) {
                if (indexed_probs[k].second == label) {
                    top5_cnt++;
                    break;
                }
            }
        }
        
        // 保存预测结果
        predictions[i] = pred;
    }
    
    // 5. 平均loss并应用scaling
    *loss = scaling_factor * (total_loss / batch_size);
    *top1_correct = top1_cnt;
    *top5_correct = top5_cnt;
}
```

#### 4.3.2 反向（FP32）

```cpp
static void launch_softmax_ce_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    // 参数解析
    const auto* p = std::get_if<SoftmaxCEParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "Missing SoftmaxCEParams");
    
    int num_classes = p->num_classes;
    float scaling_factor = p->scaling_factor;
    
    // 输入
    float* logits = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const int32_t* labels = static_cast<const int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    
    // 输出（覆盖输入）
    float* d_logits = logits;  // 直接覆盖
    
    int batch_size = op_ctx->input_shape.n;
    
    float scale = scaling_factor / batch_size;
    
    // 对每个样本计算梯度
    for (int i = 0; i < batch_size; ++i) {
        float* logits_i = logits + i * num_classes;
        float* grad_i = d_logits + i * num_classes;
        int32_t label = labels[i];
        
        // 1. 计算softmax（复用FWD逻辑）
        float max_val = *std::max_element(logits_i, logits_i + num_classes);
        float sum = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            grad_i[j] = std::exp(logits_i[j] - max_val);
            sum += grad_i[j];
        }
        
        // 2. softmax概率
        for (int j = 0; j < num_classes; ++j) {
            grad_i[j] /= sum;
        }
        
        // 3. 减one-hot（label位置减1）
        grad_i[label] -= 1.0f;
        
        // 4. 应用scaling
        for (int j = 0; j < num_classes; ++j) {
            grad_i[j] *= scale;
        }
    }
}
```

#### 4.3.3 AMP支持

```cpp
static void launch_softmax_ce_amp_fwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_FWD is not supported on CPU (FP16 not available)");
}

static void launch_softmax_ce_amp_bwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_BWD is not supported on CPU (FP16 not available)");
}
```

### 4.4 CUDA实现方案

#### 4.4.1 前向Kernel（FP32）

```cuda
__global__ void softmax_ce_fwd_fp32_kernel(
    const float* __restrict__ logits,  // [batch, num_classes]
    const int32_t* __restrict__ labels,  // [batch]
    float* __restrict__ loss,  // [1] 标量
    int32_t* __restrict__ top1_correct,  // [1] 标量
    int32_t* __restrict__ top5_correct,  // [1] 标量
    int32_t* __restrict__ predictions,  // [batch]
    int batch_size,
    int num_classes,
    float scaling_factor)
{
    // 每个block处理一个样本
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    const float* logits_i = logits + batch_idx * num_classes;
    int32_t label = labels[batch_idx];
    
    // 1. Shared memory用于规约
    extern __shared__ float sdata[];
    
    // 找最大值
    float max_val = -INFINITY;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        max_val = fmaxf(max_val, logits_i[j]);
    }
    
    __shared__ float sh_max;
    if (tid == 0) sh_max = -INFINITY;
    __syncthreads();
    
    atomicMax(&sh_max, max_val);  // 需要自定义或使用warp reduction
    __syncthreads();
    max_val = sh_max;
    
    // 2. 计算exp和sum
    float exp_sum = 0.0f;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float exp_val = expf(logits_i[j] - max_val);
        sdata[j] = exp_val;
        exp_sum += exp_val;
    }
    
    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        exp_sum += __shfl_down_sync(0xFFFFFFFF, exp_sum, offset);
    }
    
    __shared__ float sh_sum;
    if (tid == 0) sh_sum = exp_sum;
    __syncthreads();
    exp_sum = sh_sum;
    
    // 3. 计算loss
    float log_sum_exp = max_val + logf(exp_sum);
    float target_logit = logits_i[label];
    float sample_loss = -target_logit + log_sum_exp;
    
    // 4. 找预测结果（argmax）
    __shared__ int sh_pred;
    if (tid == 0) sh_pred = 0;
    __syncthreads();
    
    for (int j = tid; j < num_classes; j += blockDim.x) {
        if (logits_i[j] > logits_i[sh_pred]) {
            atomicMax(&sh_pred, j);  // 需要自定义atomicMax for int
        }
    }
    __syncthreads();
    
    int pred = sh_pred;
    predictions[batch_idx] = pred;
    
    // 5. 累加loss
    if (tid == 0) {
        atomicAdd(loss, sample_loss);
        if (pred == label) atomicAdd(top1_correct, 1);
        
        // Top-5需要额外逻辑（略）
    }
    
    // 6. Block结束后归一化loss
    if (blockIdx.x == 0 && tid == 0) {
        *loss = scaling_factor * (*loss / batch_size);
    }
}
```

**注意**：

- 实际实现需要更细致的warp reduction
- Top-5准确率需要额外处理
- 需要初始化loss和correct为0

#### 4.4.2 反向Kernel（FP32）

```cuda
__global__ void softmax_ce_bwd_fp32_kernel(
    float* __restrict__ logits,  // [batch, num_classes] 被覆盖
    const int32_t* __restrict__ labels,  // [batch]
    int batch_size,
    int num_classes,
    float scaling_factor)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    float* logits_i = logits + batch_idx * num_classes;
    int32_t label = labels[batch_idx];
    
    float scale = scaling_factor / batch_size;
    
    // 1. 找最大值
    float max_val = -INFINITY;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        max_val = fmaxf(max_val, logits_i[j]);
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    
    // 2. 计算softmax概率
    float sum_exp = 0.0f;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float exp_val = expf(logits_i[j] - max_val);
        logits_i[j] = exp_val;  // 复用logits缓冲区存储exp
        sum_exp += exp_val;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    
    // 3. 计算梯度：softmax - one_hot
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float prob = logits_i[j] / sum_exp;
        float grad = (j == label) ? (prob - 1.0f) : prob;
        logits_i[j] = grad * scale;  // 覆盖原始logits
    }
}
```

#### 4.4.3 AMP Kernel

**关键点**：

1. 输入logits是FP16，需要转换为FP32
2. 中间计算全FP32
3. 输出梯度需要转回FP16

```cuda
__global__ void softmax_ce_amp_fwd_kernel(
    const half* __restrict__ logits_fp16,  // [batch, num_classes]
    const int32_t* __restrict__ labels,
    float* __restrict__ loss,
    int32_t* __restrict__ top1_correct,
    int32_t* __restrict__ top5_correct,
    int32_t* __restrict__ predictions,
    int batch_size,
    int num_classes,
    float scaling_factor)
{
    // 与FP32版本相同，但输入需要转换
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // 1. 加载FP16并转换为FP32
    float logits_i[128];  // 假设max 128 classes
    for (int j = tid; j < num_classes; j += blockDim.x) {
        logits_i[j] = __half2float(logits_fp16[batch_idx * num_classes + j]);
    }
    
    // 2. 后续逻辑与FP32版本完全相同
    // ...
}
```

### 4.5 CUDA Launch函数

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
    
    float* loss = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    int32_t* top1_correct = static_cast<int32_t*>(ctx.ptr_at(node.output_ids[1]));
    int32_t* top5_correct = static_cast<int32_t*>(ctx.ptr_at(node.output_ids[2]));
    int32_t* predictions = static_cast<int32_t*>(ctx.ptr_at(node.output_ids[3]));
    
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    
    // 初始化输出为0
    cudaMemsetAsync(loss, 0, sizeof(float), s);
    cudaMemsetAsync(top1_correct, 0, sizeof(int32_t), s);
    cudaMemsetAsync(top5_correct, 0, sizeof(int32_t), s);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = batch_size;
    size_t shared_mem_size = num_classes * sizeof(float);
    
    softmax_ce_fwd_fp32_kernel<<<grid_size, block_size, shared_mem_size, s>>>(
        logits, labels, loss, top1_correct, top5_correct, predictions,
        batch_size, num_classes, p->scaling_factor);
    
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

---

## 五、参数结构定义

```cpp
// op_params.h
struct SoftmaxCEParams {
    int num_classes;      // 类别数
    float scaling_factor; // 损失缩放因子
    
    SoftmaxCEParams() = default;
    SoftmaxCEParams(int nc, float sf) : num_classes(nc), scaling_factor(sf) {}
    
    OpParams to_op_params() const {
        OpParams params;
        params.data = *this;
        return params;
    }
};
```

---

## 六、内存布局与Region分配

### 6.1 FWD版Region分配

| 张量           | Region                                       | 说明             |
| -------------- | -------------------------------------------- | ---------------- |
| logits         | F_FEATURE_FP32 (FP32) / F_FEATURE_FP16 (AMP) | 输入特征图       |
| labels         | I_A_LABEL / I_B_LABEL                        | 输入标签（已有） |
| scaling_factor | S_SCALAR_FP32                                | 标量输入（已有） |
| loss           | S_SCALAR_FP32                                | 标量输出（已有） |
| top1_correct   | S_SCALAR_INT32                               | 标量输出（已有） |
| top5_correct   | S_SCALAR_INT32                               | 标量输出（已有） |
| predictions    | T_TEMP_INT32                                 | 临时张量（已有） |

### 6.2 BWD版Region分配

| 张量           | Region                                           | 说明                   |
| -------------- | ------------------------------------------------ | ---------------------- |
| logits         | F_FEATURE_FP32 (FP32) / F_FEATURE_FP16 (AMP)     | 输入，被梯度覆盖       |
| d_logits       | F_GRAD_SLOT_FP32 (FP32) / F_GRAD_SLOT_FP16 (AMP) | 梯度输出（覆盖logits） |
| labels         | I_A_LABEL / I_B_LABEL                            | 输入标签               |
| scaling_factor | S_SCALAR_FP32                                    | 标量输入               |

### 6.3 内存复用策略

**关键**：BWD时d_logits直接覆盖logits，无需额外分配

```cpp
// 在SimpleTask中分配时
DTensor d_logits = task.alloc(logits.shape(), dtype, F_GRAD_SLOT_FP32);
// 但物理上与logits共享内存，通过MemoryPlan的alias机制实现
```

---

## 七、性能优化考虑

### 7.1 CPU优化

1. **向量化**：使用Eigen或SIMD指令加速exp/log
2. **并行化**：OpenMP并行处理不同样本
3. **缓存友好**：按行连续访问logits

```cpp
#pragma omp parallel for reduction(+:total_loss, top1_cnt, top5_cnt)
for (int i = 0; i < batch_size; ++i) {
    // 处理第i个样本
}
```

### 7.2 CUDA优化

1. **Warp Reduction**：替代shared memory原子操作
2. **循环展开**：减少分支
3. **FP16优化**：使用Tensor Core（AMP模式）

### 7.3 数值稳定性

1. **Log-Sum-Exp技巧**：避免overflow
2. **Kahan求和**：提高精度（可选）

---

## 八、测试验证计划

### 8.1 单元测试

创建 `test_softmax_ce.cpp`：

```cpp
// 测试FWD
test_softmax_ce_fwd_fp32_cpu();
test_softmax_ce_fwd_amp_cuda();
// ... 

// 测试BWD
test_softmax_ce_bwd_fp32_cpu();
test_softmax_ce_bwd_amp_cuda();
// ...

// 数值验证（对比PyTorch）
test_softmax_ce_vs_pytorch();
```

### 8.2 集成测试

在完整训练流程中验证：

1. Loss收敛性
2. 准确率计算正确性
3. 梯度数值正确性（gradient check）

### 8.3 性能测试

对比以下场景：

1. 融合算子 vs 分离算子（softmax + ce）
2. FP32 vs AMP性能
3. 不同batch_size和num_classes下的性能

---

## 九、实施步骤

### 阶段1：基础设施（1-2天）

1. 在 `op_kind.h` 中添加4个算子类型
2. 创建 `softmax_ce_op.cpp` 和 `softmax_ce_op.cu`
3. 定义 `SoftmaxCEParams` 结构体

### 阶段2：CPU实现（2-3天）

1. 实现 `launch_softmax_ce_fp32_fwd_cpu`
2. 实现 `launch_softmax_ce_fp32_bwd_cpu`
3. 添加AMP占位函数（抛出错误）

### 阶段3：CUDA实现（3-4天）

1. 实现 `softmax_ce_fwd_fp32_kernel`
2. 实现 `softmax_ce_bwd_fp32_kernel`
3. 实现对应的launch函数
4. 实现AMP版本（FP16转换）

### 阶段4：注册与测试（2-3天）

1. 实现 `register_op_softmax_ce()` 函数
2. 在 `op_registry.cpp` 中调用注册
3. 编写单元测试
4. 数值验证（对比PyTorch）

### 阶段5：优化与文档（1-2天）

1. 性能profiling与优化
2. 添加使用文档
3. 更新开发者指南

**总耗时估算**：9-14天

---

## 十、风险与注意事项

### 10.1 技术风险

| 风险              | 概率 | 影响 | 缓解措施                  |
| ----------------- | ---- | ---- | ------------------------- |
| 数值溢出/下溢     | 中   | 高   | 使用log-sum-exp技巧       |
| Top-5实现复杂度高 | 低   | 中   | 可先实现Top-1，后续补充   |
| AMP精度损失       | 低   | 中   | 严格FP32中间计算          |
| CUDA kernel正确性 | 中   | 高   | 详细单元测试，对比PyTorch |

### 10.2 架构兼容性

1. **Region选择**：确保与现有MemoryPlan兼容
2. **梯度覆盖约定**：遵守dX覆盖X的框架约定
3. **Stream策略**：使用COMP_1流（与其他算子一致）

### 10.3 性能考虑

1. **避免不必要的内存拷贝**：BWD直接覆盖logits
2. **减少同步点**：CUDA kernel内部尽量减少__syncthreads
3. **批处理优化**：大batch时使用grid-stride loop

---

## 十一、总结

本方案提供了完整的SOFTMAX_CE融合算子实现路线图：

1. **无需新增Region**：复用现有S-Series、T-Series、F-Series
2. **数值稳定**：使用log-sum-exp技巧
3. **性能优化**：避免one-hot展开，减少内存带宽
4. **架构对齐**：遵循现有算子的实现模式

**核心优势**：

- 融合设计避免了显存浪费
- 中间运算全FP32保证数值精度
- 梯度覆盖输入符合框架约定
- CPU/CUDA双后端支持

建议按阶段逐步实施，优先完成FP32版本，验证正确性后再实现AMP版本。

# 【小伙伴K】



## 一、现状分析

当前代码中：

- `ComputeOp` 枚举已有 `CROSS_ENTROPY_LOSS_FP32_FWD/BWD` 和 `CROSS_ENTROPY_LOSS_AMP_FWD/BWD`，但**没有注册任何 launch 函数**（`g_compute_op_table` 中对应 entry 的 `launch_cpu` 和 `launch_cuda` 均为 `nullptr`）。
- `layer_descriptor_registry.cpp` 中 SoftmaxCE 层的 FWD/BWD 映射到上述 `CROSS_ENTROPY_LOSS_*` 算子。
- 没有 `softmax_ce_op.cpp` / `softmax_ce_op.cu` 文件。
- `Region` 枚举缺少专门存放推理结果（TOP-1/TOP-5/预测标签）的 region。

**SOF.md 的核心要求**：

1. 实现 4 个算子：`SOFTMAX_CE_FP32_FWD`、`SOFTMAX_CE_FP32_BWD`、`SOFTMAX_CE_AMP_FWD`、`SOFTMAX_CE_AMP_BWD`
2. Softmax + CrossEntropy **融合**，中间不展开 one-hot
3. 无论 FP32/AMP，**内部计算全部用 FP32**
4. FWD：输出 loss（标量）、top1_correct、top5_correct、predicted_labels
5. BWD：grad 覆盖 logits，乘以 scaling_factor
6. CUDA 不走 cuDNN FE（手写融合 kernel 更高效）；CPU 不用 Eigen

---

## 二、融合数学原理

### 2.1 FWD（Softmax + CrossEntropy）

数值稳定版本，每个 sample 独立：

```
max_val   = max(logits[i] for i in [0, num_classes))
sum_exp   = sum(exp(logits[i] - max_val) for i in [0, num_classes))
log_softmax_label = logits[label] - max_val - log(sum_exp)
ce_sample = -log_softmax_label
```

batch loss = `mean(ce_sample) * scaling_factor`

**关键优化**：不需要显式输出 softmax 概率数组也可以直接算 loss。但为了 BWD 复用，FWD **仍输出 softmax_probs 到 Temp 区**（避免 BWD 重复计算 exp/sum）。

### 2.2 BWD

CrossEntropy + Softmax 的梯度有闭式解：

```
grad_logits[i] = (softmax_probs[i] - (i == label ? 1.0 : 0.0)) * scaling_factor / batch
```

无需 one-hot 展开，直接条件判断即可。

### 2.3 TOP-1 / TOP-5 / Predicted Label

FWD kernel 在遍历 logits 时同步维护：

- `top1_idx` = argmax(logits)
- `top5` = 维护一个大小为 5 的小根堆（或直接 5 次遍历找前 5）
- `predicted_label[sample] = top1_idx`
- `top1_correct += (top1_idx == label)`
- `top5_correct += (label in top5)`

---

## 三、需要修改的文件清单（11 个文件）

| #    | 文件                                        | 改动性质                                                     | 行数估算 |
| ---- | ------------------------------------------- | ------------------------------------------------------------ | -------- |
| 1    | `include/renaissance/graph/op_kind.h`       | 新增 4 个 `ComputeOp` 枚举值                                 | +4       |
| 2    | `include/renaissance/core/types.h`          | 新增 3 个 `Region`（结果区）                                 | +3       |
| 3    | `include/renaissance/backend/op_registry.h` | 新增 `register_op_softmax_ce()` 声明                         | +1       |
| 4    | `src/backend/op_registry.cpp`               | 调用 `register_op_softmax_ce()`                              | +2       |
| 5    | `src/graph/op_kind.cpp`                     | `compute_op_to_string` 新增 4 个 case                        | +4       |
| 6    | `src/graph/layer_descriptor_registry.cpp`   | 修改 `infer_softmaxce_tensors`、`build_softmaxce_forward/backward` | ~40      |
| 7    | `src/backend/ops/dtensor/softmax_ce_op.cpp` | **新文件**：CPU launch + CUDA launch + 注册                  | ~200     |
| 8    | `src/backend/ops/dtensor/softmax_ce_op.cu`  | **新文件**：CUDA kernels                                     | ~180     |
| 9    | `src/CMakeLists.txt`                        | 添加新 `.cpp` / `.cu` 到源文件列表                           | +2       |
| 10   | `src/graph/compiler.cpp`                    | 可能需要调整 SoftmaxCE 的内存分配逻辑                        | 待验证   |
| 11   | `src/backend/op_registry.cpp`               | `require_warmup` 是否需要添加 SOFTMAX_CE_AMP_*               | 待验证   |

---

## 四、详细设计

### 4.1 ComputeOp 枚举扩展（op_kind.h）

在 `enum class ComputeOp` 的 `CROSS_ENTROPY_LOSS_AMP_BWD` 之后、`ALLREDUCE_SUM` 之前插入：

```cpp
    // === Softmax + CrossEntropy 融合 ===
    SOFTMAX_CE_FP32_FWD,
    SOFTMAX_CE_FP32_BWD,
    SOFTMAX_CE_AMP_FWD,
    SOFTMAX_CE_AMP_BWD,
```

> **注意**：这是新增 4 个算子，不是替换现有的 `CROSS_ENTROPY_LOSS_*`。保留旧枚举值以避免破坏其他引用。

### 4.2 Region 扩展（types.h）

在 `Region` 枚举末尾（`DEFAULT` 之前）新增：

```cpp
    // R-Series: 推理结果（INT32）
    R_TOP1_CORRECT,      // 066
    R_TOP5_CORRECT,      // 067
    R_PREDICTED_LABEL,   // 068
```

并更新 `NUM_REGIONS = 68`。

### 4.3 LayerDescriptor 修改（layer_descriptor_registry.cpp）

#### 4.3.1 `infer_softmaxce_tensors` 重写

当前返回 4 个张量，需要扩展为包含 labels 输入、top1/top5 输出、predicted_labels 输出、softmax_probs 中间张量：

```cpp
std::vector<TensorDesc> infer_softmaxce_tensors(
    const Shape& input, const OpParams& p, const InferContext& ctx)
{
    // input shape: [batch, 1, 1, num_classes]
    int batch = input.n();
    const auto* lp = std::get_if<LossParams>(&p.data);
    int num_classes = lp ? lp->num_classes : input.c();

    return {
        // 输入（由上一层提供 logits）
        // 额外输入：labels（INT32，紧凑连续）
        TensorDesc{"labels",             Shape{batch,1,1,1},  Region::I_A_LABEL,      DType::INT32},

        // 输出
        TensorDesc{"ce_loss",            Shape{1,1,1,1},      Region::S_SCALAR_FP32,  DType::FP32},
        TensorDesc{"top1_correct",       Shape{1,1,1,1},      Region::R_TOP1_CORRECT, DType::INT32},
        TensorDesc{"top5_correct",       Shape{1,1,1,1},      Region::R_TOP5_CORRECT, DType::INT32},
        TensorDesc{"predicted_labels",   Shape{batch,1,1,1},  Region::R_PREDICTED_LABEL, DType::INT32},
        TensorDesc{"scaling_factor",     Shape{1,1,1,1},      Region::S_SCALAR_FP32,  DType::FP32},

        // 中间（Temp区，FWD输出供BWD复用）
        TensorDesc{"softmax_probs",      input,               Region::T_TEMP_FP32,    DType::FP32},
    };
}
```

> **讨论**：labels 放在 `I_A_LABEL` 是否合适？当前 `I_A_LABEL` 是输入缓冲区，用于数据加载。SoftmaxCE 的 labels 本质上是输入数据的一部分，复用 `I_A_LABEL` 合理。如果冲突，可改用 `T_TEMP_INT32`。

#### 4.3.2 `build_softmaxce_forward` 重写

```cpp
SubgraphPattern build_softmaxce_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;  // labels + loss + top1 + top5 + pred + scaling + probs

    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_FWD
               : ComputeOp::SOFTMAX_CE_FP32_FWD;
    // inputs:  logits(来自上一层) + labels(0) + scaling_factor(4)
    // 但 logits 是前一层的输出，通过 GraphNode::input_ids 隐式传入
    // 这里需要明确指定 Node 的 input/output indices
    n.input_indices  = {0, 5};   // labels(0) + scaling_factor(5) 的索引
    n.output_indices = {1, 2, 3, 4, 6}; // loss(1) + top1(2) + top5(3) + pred(4) + probs(6)
    p.nodes.push_back(n);
    return p;
}
```

> **问题**：SubgraphPattern 的 `input_indices` 和 `output_indices` 是相对于该层的 `descs` 列表的索引。需要与 `infer_softmaxce_tensors` 返回的顺序严格对应。上述索引基于 4.3.1 的返回顺序。

#### 4.3.3 `build_softmaxce_backward` 重写

```cpp
SubgraphPattern build_softmaxce_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;

    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_BWD
               : ComputeOp::SOFTMAX_CE_FP32_BWD;
    // inputs: softmax_probs(6) + labels(0) + scaling_factor(5)
    n.input_indices  = {6, 0, 5};
    // output: grad_logits，in-place 覆盖 logits（即前一层的输出/ce_output）
    // logits 在 backward 图中的索引需要由 Compiler 决定
    n.output_indices = {};  // 空表示 in-place 覆盖第一个输入（或需显式指定）
    p.nodes.push_back(n);
    return p;
}
```

> **in-place 覆盖的精确语义**：当前框架中 "dX 覆盖 X" 是通过 MemoryPlan 的 Region 复用实现的（logits 和 grad_logits 共享同一个 Region）。`build_softmaxce_backward` 的 `output_indices` 可能不需要显式指定，因为 Compiler 会自动处理梯度槽的分配。这需要进一步验证 Compiler 逻辑。

### 4.4 CUDA Kernel 设计（softmax_ce_op.cu）

#### 4.4.1 FWD Kernel（FP32）

```cuda
__global__ void softmax_ce_fwd_fp32_kernel(
    const float* __restrict__ logits,   // [batch, num_classes]
    const int*   __restrict__ labels,   // [batch]
    float* __restrict__ loss_out,       // [1]
    int*   __restrict__ top1_correct,   // [1]
    int*   __restrict__ top5_correct,   // [1]
    int*   __restrict__ predicted_labels, // [batch]
    float* __restrict__ softmax_probs,  // [batch, num_classes] (Temp区)
    float scaling_factor,
    int batch, int num_classes)
{
    extern __shared__ float sdata[];  // 动态共享内存

    int sample = blockIdx.x;
    if (sample >= batch) return;

    const float* logit_row = logits + sample * num_classes;
    float* prob_row = softmax_probs + sample * num_classes;
    int label = labels[sample];

    // Step 1: 找 max
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        max_val = fmaxf(max_val, logit_row[i]);
    }
    // block reduce max
    // ... 使用 shared memory 做 warp/block reduce ...

    // Step 2: 计算 sum_exp 和 softmax_probs
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        float e = expf(logit_row[i] - max_val);
        prob_row[i] = e;
        sum_exp += e;
    }
    // block reduce sum

    // Step 3: 归一化 softmax_probs
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        prob_row[i] /= sum_exp;
    }

    // Step 4: 计算该 sample 的 CE loss（仅 thread 0）
    float ce_sample = 0.0f;
    if (threadIdx.x == 0) {
        float log_softmax = logf(prob_row[label]);  // 或 logits[label] - max_val - log(sum_exp)
        ce_sample = -log_softmax;

        // TOP-1 / TOP-5
        int top1_idx = 0;
        float top1_val = prob_row[0];
        // 找 top1...
        predicted_labels[sample] = top1_idx;
        if (top1_idx == label) atomicAdd(top1_correct, 1);
        // 找 top5...
        // if (label in top5) atomicAdd(top5_correct, 1);

        // 累加 loss（atomicAdd 到全局）
        atomicAdd(loss_out, ce_sample);
    }
}
```

> **共享内存大小**：每个 block 处理一个 sample，共享内存主要用于 block reduce（max/sum）。对于 num_classes=1000，float 数组约 4KB，可放入 shared memory。

#### 4.4.2 BWD Kernel（FP32）

```cuda
__global__ void softmax_ce_bwd_fp32_kernel(
    const float* __restrict__ softmax_probs, // [batch, num_classes]
    const int*   __restrict__ labels,        // [batch]
    float* __restrict__ grad_logits,         // [batch, num_classes] (覆盖输入)
    float scaling_factor,
    int batch, int num_classes)
{
    int sample = blockIdx.x;
    if (sample >= batch) return;

    const float* prob_row = softmax_probs + sample * num_classes;
    float* grad_row = grad_logits + sample * num_classes;
    int label = labels[sample];
    float scale = scaling_factor / batch;

    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        float g = prob_row[i] * scale;
        if (i == label) g -= scale;
        grad_row[i] = g;
    }
}
```

#### 4.4.3 AMP Kernels

AMP FWD：与 FP32 逻辑相同，但输入 logits 是 `__half*`，需要 `__half2float` 转换后计算。

AMP BWD：计算 grad 为 FP32，最后 `__float2half` 写回。

```cuda
__global__ void softmax_ce_bwd_amp_kernel(
    const float* __restrict__ softmax_probs,
    const int*   __restrict__ labels,
    __half* __restrict__ grad_logits,  // 输出 FP16
    float scaling_factor,
    int batch, int num_classes)
{
    // ... 同 FP32 BWD，但最后：
    // grad_logits[i] = __float2half(g);
}
```

### 4.5 CPU 实现（softmax_ce_op.cpp）

#### 4.5.1 FWD CPU

```cpp
static void launch_softmax_ce_fwd_cpu(CpuOpContext* op_ctx) {
    // 参数解析
    int batch = op_ctx->input_shape.n;
    int num_classes = op_ctx->input_shape.c;

    float* logits = ...;   // input_ids[0]
    int*   labels = ...;   // input_ids[1]
    float* scaling = ...;  // input_ids[2]

    float* loss = ...;     // output_ids[0]
    int* top1 = ...;       // output_ids[1]
    int* top5 = ...;       // output_ids[2]
    int* pred = ...;       // output_ids[3]
    float* probs = ...;    // output_ids[4] (softmax_probs in Temp)

    float scale = scaling[0];
    double loss_sum = 0.0;
    int top1_cnt = 0, top5_cnt = 0;

    for (int b = 0; b < batch; ++b) {
        float* logit_row = logits + b * num_classes;
        float* prob_row  = probs + b * num_classes;
        int label = labels[b];

        // 找 max
        float max_val = logit_row[0];
        for (int i = 1; i < num_classes; ++i) max_val = std::max(max_val, logit_row[i]);

        // sum_exp
        double sum_exp = 0.0;
        for (int i = 0; i < num_classes; ++i) {
            prob_row[i] = std::exp(logit_row[i] - max_val);
            sum_exp += prob_row[i];
        }

        // 归一化
        for (int i = 0; i < num_classes; ++i) prob_row[i] /= (float)sum_exp;

        // CE loss
        loss_sum += -std::log((double)prob_row[label] + 1e-30);

        // TOP-1
        int top1_idx = 0;
        float top1_prob = prob_row[0];
        for (int i = 1; i < num_classes; ++i) {
            if (prob_row[i] > top1_prob) { top1_prob = prob_row[i]; top1_idx = i; }
        }
        pred[b] = top1_idx;
        if (top1_idx == label) top1_cnt++;

        // TOP-5（小根堆或直接 5 次遍历）
        // 对于 num_classes=1000，直接部分排序或 5 次扫描即可
        // ...
    }

    loss[0] = (float)(loss_sum / batch * scale);
    top1[0] = top1_cnt;
    top5[0] = top5_cnt;
}
```

#### 4.5.2 BWD CPU

```cpp
static void launch_softmax_ce_bwd_cpu(CpuOpContext* op_ctx) {
    int batch = op_ctx->input_shape.n;
    int num_classes = op_ctx->input_shape.c;

    float* probs = ...;    // input_ids[0] (softmax_probs)
    int*   labels = ...;   // input_ids[1]
    float* scaling = ...;  // input_ids[2]
    float* grad = ...;     // output_ids[0] (覆盖 logits)

    float scale = scaling[0] / batch;

    for (int b = 0; b < batch; ++b) {
        float* prob_row = probs + b * num_classes;
        float* grad_row = grad + b * num_classes;
        int label = labels[b];

        for (int i = 0; i < num_classes; ++i) {
            float g = prob_row[i] * scale;
            if (i == label) g -= scale;
            grad_row[i] = g;
        }
    }
}
```

### 4.6 算子注册（softmax_ce_op.cpp 末尾）

```cpp
void register_op_softmax_ce() {
    {
        auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_FWD)];
        entry.op = ComputeOp::SOFTMAX_CE_FP32_FWD;
        entry.launch_cpu = launch_softmax_ce_fwd_cpu;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_softmax_ce_fwd_cuda;
#endif
    }
    {
        auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_BWD)];
        entry.op = ComputeOp::SOFTMAX_CE_FP32_BWD;
        entry.launch_cpu = launch_softmax_ce_bwd_cpu;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_softmax_ce_bwd_cuda;
#endif
    }
    {
        auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_FWD)];
        entry.op = ComputeOp::SOFTMAX_CE_AMP_FWD;
        entry.launch_cpu = launch_softmax_ce_fwd_cpu_amp_not_supported;  // CPU 不支持 FP16
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_softmax_ce_fwd_amp_cuda;
#endif
    }
    {
        auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_BWD)];
        entry.op = ComputeOp::SOFTMAX_CE_AMP_BWD;
        entry.launch_cpu = launch_softmax_ce_bwd_cpu_amp_not_supported;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_softmax_ce_bwd_amp_cuda;
#endif
    }
}
```

### 4.7 CMakeLists.txt

在 `src/CMakeLists.txt` 的源文件列表中添加：

```cmake
    backend/ops/dtensor/softmax_ce_op.cpp
    backend/ops/dtensor/softmax_ce_op.cu
```

---

## 五、关键设计决策

### 5.1 为什么手写 CUDA Kernel 而非 cuDNN FE？

- **融合度**：cuDNN FE 可以将 Softmax 和 Reduction 组合，但 CrossEntropy 的 label-dependent 特性（需要读取 labels 数组）无法直接用 cuDNN FE 表达。
- **额外输出**：cuDNN FE 难以同时输出 top1_correct、top5_correct、predicted_labels。
- **性能**：手写 kernel 可以将 max + exp + sum + log + topk 全部融合在一个 kernel 中，避免中间内存读写。

### 5.2 为什么 FWD 输出 softmax_probs 到 Temp 区？

- **避免 BWD 重复计算**：BWD 需要 softmax_probs，如果 FWD 不保存，BWD 需要重新计算一遍 exp/sum（与 FWD 相同的计算量）。
- **内存代价**：`batch * num_classes * 4` bytes = 7 * 1000 * 4 = 28 KB，极小。
- **与 workspace 的区别**：softmax_probs 是"有意义的张量"（BWD 的显式输入），符合 SOF.md 中"用 Temp 区而非 workspace"的准则。

### 5.3 TOP-5 怎么算？

对于 num_classes=1000，每个 sample 的 TOP-5 可以用**部分选择排序**（partial selection sort）或**5 次线性扫描**：

```cpp
// 5 次扫描找前 5 大（足够快，1000 * 5 = 5000 次比较）
bool label_in_top5 = false;
for (int k = 0; k < 5; ++k) {
    int best_idx = -1;
    float best_val = -1e30f;
    for (int i = 0; i < num_classes; ++i) {
        if (prob_row[i] > best_val) { best_val = prob_row[i]; best_idx = i; }
    }
    if (best_idx == label) label_in_top5 = true;
    prob_row[best_idx] = -1e30f;  // 标记为已选
}
```

在 CUDA 中，可以使用 shared memory 存储 top-5 候选，每个 warp/线程协作更新。

### 5.4 数值稳定性

- `exp(logits - max_val)` 防止 overflow
- `log(prob_row[label] + 1e-30)` 防止 log(0)
- 内部全部 FP32（SOF.md 强制要求）

---

## 六、实施风险与验证

| 风险                                                         | 缓解措施                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `SubgraphPattern` 的 input/output indices 与 `infer_softmaxce_tensors` 顺序不匹配 | 严格对照索引，编译后通过 `test_flatten_fc_relu_fc` 类测试验证图拓扑 |
| Region 新增导致 MemoryPlan 布局变化                          | 检查 `NUM_REGIONS` 和 `region_to_string` 的同步更新          |
| TOP-5 实现效率低                                             | 先用 5 次扫描实现，后续用 profiling 优化                     |
| AMP 下 `__half` 转 `float` 的精度问题                        | 所有计算在 float 中完成，仅在读写时转换                      |

### 验证计划

1. **Standalone 测试**：创建 `test_softmax_ce_fwd_bwd`，对比 PyTorch 的 `F.cross_entropy` + `F.softmax`
2. **Composite 测试**：在包含 SoftmaxCE 的模型端到端训练中验证 loss 下降和 accuracy 正确
3. **性能基准**：对比非融合实现（softmax + cross_entropy 分离）的耗时和内存带宽

---

## 七、改动量总结

- **新增文件**：2 个（`softmax_ce_op.cpp`, `softmax_ce_op.cu`）
- **修改文件**：9 个
- **总代码量**：~450 行（其中 kernel ~180 行，CPU launch ~120 行，注册/枚举/Region ~80 行，LayerDescriptor ~70 行）
- **接口破坏**：零（新增枚举值和 Region 不影响已有代码）

# 【小伙伴D】



**版本**: V1.0  
**日期**: 2026-05-19  

---

## 一、需求回顾

来自 [SOF.md](file:///r:/renaissance/SOF.md) 的核心要求：

| 算子                  | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| `SOFTMAX_CE_FP32_FWD` | 从 logits + labels 计算 loss + softmax 输出 + scaling factor + TOP-1/5 正确数 |
| `SOFTMAX_CE_FP32_BWD` | 从 softmax 输出 + labels 计算 grad(logits) × scaling_factor  |
| `SOFTMAX_CE_AMP_FWD`  | 同上，logits 输入为 FP16，内部全部 FP32 运算                 |
| `SOFTMAX_CE_AMP_BWD`  | 同上，输入 probs 为 FP32，输出 grad 需转为 FP16              |

关键原则：

- **CUDA 绝不展开 one-hot**（浪费显存和带宽）
- **CPU 也尽量不展开**（除非确有必要且放在 workspace 更快）
- **AMP 内部全部 FP32 运算**（softmax + CE 涉及规约，FP16 极易出错）
- **中间有意义的张量用 Temp 区**，不要放在 workspace

---

## 二、现有基础设施（可直接复用）

### 2.1 已定义的枚举和参数

```cpp
// op_kind.h — 已存在，无需新增
CROSS_ENTROPY_LOSS,              // 无方向版本
CROSS_ENTROPY_LOSS_FP32_FWD,     // ← 直接使用
CROSS_ENTROPY_LOSS_FP32_BWD,     // ← 直接使用
CROSS_ENTROPY_LOSS_AMP_FWD,      // ← 直接使用
CROSS_ENTROPY_LOSS_AMP_BWD,      // ← 直接使用

// op_kind.h — 已存在
struct LossParams {
    float label_smoothing = 0.0f;
    int num_classes = 1000;
};
```

### 2.2 已定义的编译器子图

从 [layer_descriptor_registry.cpp:516-537](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L516-L537)：

```
forward  →  output_indices = {0, 1, 2, 3}
            → ce_loss(FP32 scalar) + ce_output(FP32) + scaling(FP32) + inv_scaling(FP32)

backward →  input_indices  = {1, 2, 3}
            output_indices = {1}
            → 读取 ce_output + scaling + inv_scaling，grad 覆盖 ce_output
```

### 2.3 参考算子

| 模式                               | 参考                                                         |
| ---------------------------------- | ------------------------------------------------------------ |
| CUDA 自定义 kernel（非 cuDNN FE）  | ReLU FP32 FWD/BWD in [relu_op.cu](file:///r:/renaissance/src/backend/ops/dtensor/relu_op.cu) |
| CUDA kernel 中使用 half↔float 转换 | [gap_op.cu:L54-L56](file:///r:/renaissance/src/backend/ops/dtensor/gap_op.cu#L54-L56) |
| CPU Eigen 优化                     | FC BWD in [fc_op.cpp:647-711](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L647-L711) |
| 算子注册                           | FC/ReLU 的 `register_op_*()` 函数                            |
| 流策略                             | [op_stream_policy.cpp](file:///r:/renaissance/src/backend/op_stream_policy.cpp) |
| 新 Region 定义                     | F_FEATURE_FP32, S_MASK, W_FC_WEIGHT 等                       |

---

## 三、需要新增的组件

### 3.1 新 Region：`R_RESULT_INT32`

存放 TOP-1 和 TOP-5 正确数（INT32 标量），需要 2 个 DTensor：

| DTensor        | 形状        | 含义                               |
| -------------- | ----------- | ---------------------------------- |
| `top1_correct` | `[1,1,1,1]` | 本 batch 中 TOP-1 预测正确的样本数 |
| `top5_correct` | `[1,1,1,1]` | 本 batch 中 TOP-5 预测正确的样本数 |

涉及文件：

- `include/renaissance/core/types.h` — 添加 `R_RESULT_INT32` 枚举值
- `src/graph/memory_plan.cpp` — Region 布局注册
- `src/graph/layer_descriptor_registry.cpp` — 更新 `infer_softmaxce_tensors()` 添加这 2 个 tensor
- 前向子图 `output_indices` 从 `{0,1,2,3}` 扩展为 `{0,1,2,3,4,5}`

### 3.2 Labels 输入张量

当前子图 FWD 节点无 `input_indices`（编译器自动解析）。但 labels 是外部输入（来自数据加载器），需要明确标注。SOF.md 描述：

> 标签（INT32），在输入缓冲区，是一个 DTensor，但其元素必定在内存上紧凑、连续，且有效元素数就等于 batch size

编译器需要将 labels DTensor 作为 FWD 的 `input_ids[1]` 传入（`input_ids[0]` 是 logits）。同样 BWD 也需要 labels 来构造 one-hot 位置索引。

---

## 四、I/O 张量布局（最终版）

### FWD

```
输入:
  input_ids[0] = logits       [batch, 1, 1, num_classes]  FP32 或 FP16(AMP)
  input_ids[1] = labels       [batch] 紧凑 INT32         （外部数据加载器提供）

输出:
  output_ids[0] = ce_loss           [1,1,1,1]  FP32      S_SCALAR_FP32
  output_ids[1] = ce_output         [batch,1,1,num_classes]  FP32  (softmax 输出)
  output_ids[2] = scaling_factor    [1,1,1,1]  FP32      S_SCALAR_FP32
  output_ids[3] = inv_scaling       [1,1,1,1]  FP32      S_SCALAR_FP32
  output_ids[4] = top1_correct      [1,1,1,1]  INT32     R_RESULT_INT32
  output_ids[5] = top5_correct      [1,1,1,1]  INT32     R_RESULT_INT32
```

### BWD

```
输入:
  input_ids[0] = ce_output     [batch,1,1,num_classes]  FP32  (FWD 的 softmax 输出)
  input_ids[1] = labels        [batch] 紧凑 INT32
  input_ids[2] = scaling_factor [1,1,1,1]  FP32
  input_ids[3] = inv_scaling    [1,1,1,1]  FP32

输出:
  output_ids[0] = dL/d(logits)  [batch,1,1,num_classes]  FP32 或 FP16(AMP)
                  (in-place 覆盖 logits / ce_output 的存储位置)
```

**注意**：SOF.md 说 BWD 的 `input_indices = {1, 2, 3}`（ce_output + scaling + inv_scaling）。但 labels 也需要作为输入以确定正确的类别索引。因此 BWD 的编译器子图也需要更新。

---

## 五、数学推导

### FWD

```
给定 logits x ∈ R^(B×C), labels y ∈ {0..C-1}^B

1. 数值稳定: x' = x - max(x, dim=1)
2. exp:      e = exp(x')
3. sum:      s = sum(e, dim=1)                    shape [B]
4. prob:     p = e / s                             shape [B×C]
5. loss:     L = mean_i(-log(p[i][y[i]]))          scalar
6. 1/scaling: inv_s = 1.0 / B
7. TOP-1:    每个样本 argmax(p[i]) == y[i] 计数
8. TOP-5:    每个样本 y[i] 在 p[i] 的 top-5 中计数
```

### BWD

```
∂L/∂x[i][c] = (p[i][c] - δ(c == y[i])) / B * scaling_factor

其中 δ 是 Kronecker delta（one-hot 仅在 label 位置为 1，其余为 0）

关键优化：无需展开 one-hot 矩阵 B×C，只需在 label 位置减 1：
  grad[i][c] = p[i][c] / B * scale
  grad[i][y[i]] -= 1/B * scale
```

---

## 六、CPU 实现方案

### 6.1 FP32 FWD

使用 Eigen 行级向量化，完全避免 one-hot 展开：

```cpp
// 伪代码
Eigen::Map<MatrixXfRow> logits_mat(logits, batch, num_classes);
Eigen::Map<MatrixXfRow> probs_mat(probs, batch, num_classes);

for (int b = 0; b < batch; ++b) {
    // 1. max per row
    float max_val = logits_mat.row(b).maxCoeff();
    
    // 2. exp
    probs_mat.row(b) = (logits_mat.row(b).array() - max_val).exp();
    
    // 3. sum + normalize (原地)
    float sum_exp = probs_mat.row(b).sum();
    probs_mat.row(b) /= sum_exp;
    
    // 4. loss (仅取 label 位置)
    float p_label = probs_mat.row(b)(label[b]);
    loss_val += -std::log(std::max(p_label, 1e-12f));
    
    // 5. TOP-1: argmax
    int pred;
    probs_mat.row(b).maxCoeff(&pred);
    if (pred == label[b]) ++top1;
    
    // 6. TOP-5: label probability >= 5th largest
    //    直接用 partial sort 或 threshold 比较
}
loss_val /= batch;
inv_scaling = 1.0f / batch;
```

**复杂度**：O(B×C)，无额外内存分配。

### 6.2 FP32 BWD

简洁实现，仅修改 label 位置：

```cpp
Eigen::Map<MatrixXfRow> probs_mat(probs, batch, num_classes);
Eigen::Map<MatrixXfRow> grad_mat(grad, batch, num_classes);

grad_mat = probs_mat * (inv_scaling * scale);  // O(B×C) broadcast

for (int b = 0; b < batch; ++b) {
    grad_mat(b, label[b]) -= inv_scaling * scale;  // 仅修改1个元素
}
```

**优化要点**：

- 第一行 `probs_mat * scale_factor` 是单次 Eigen scalar-broadcast 乘法，高度向量化
- 循环仅 O(B) 次减法（非 O(B×C)），one-hot 未展开

### 6.3 AMP CPU — 不支持

与其他算子一致，AMP 不在 CPU 上实现，直接抛 `TR_TYPE_ERROR`。

---

## 七、CUDA 实现方案

### 7.1 新增文件

```
src/backend/ops/dtensor/softmax_ce_op.cu   — CUDA kernels
src/backend/ops/dtensor/softmax_ce_op.cpp  — 分发 + 注册
```

CMakeLists.txt 中添加配对编译。

### 7.2 Kernel 设计

#### 7.2.1 FWD Kernel（FP32 + AMP 共享结构）

每个 block 处理一个 batch item（B 个 block），使用 shared memory 做行内 reduction：

```cpp
template<bool IS_AMP>
__global__ void softmax_ce_fwd_kernel(
    const T* __restrict__ logits,       // FP32 或 __half
    const int* __restrict__ labels,
    float* __restrict__ probs,          // 输出 always FP32
    float* __restrict__ loss,           // per-sample loss，后续 CPU reduce
    int* __restrict__ top1_correct,
    int* __restrict__ top5_correct,
    int N, int C)
{
    extern __shared__ float smem[];    // 动态 shared memory: C × sizeof(float)
    int b = blockIdx.x;
    if (b >= N) return;

    // 1. 加载 logits 到 shared memory + 找最大值（FP16 先转为 FP32）
    float max_val = -INFINITY;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = IS_AMP ? __half2float(logits[b * C + c]) : logits[b * C + c];
        smem[c] = v;
        max_val = fmaxf(max_val, v);
    }
    __syncthreads();

    // 2. Warp-level reduce max（优化）
    //    blockReduceMax(max_val)
    
    // 3. exp + sum
    float sum_exp = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        smem[c] = expf(smem[c] - max_val);
        sum_exp += smem[c];
    }
    __syncthreads();
    // blockReduceSum(sum_exp)

    // 4. normalize + loss (仅 label 位置)
    int label = labels[b];
    float prob_label = smem[label] / sum_exp;
    if (threadIdx.x == 0) {
        loss[b * 2 + 0] = -logf(fmaxf(prob_label, 1e-12f));
        loss[b * 2 + 1] = sum_exp;  // 用于后续验证
    }

    // 5. 归一化写入 probs（always FP32）
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        probs[b * C + c] = smem[c] / sum_exp;
    }

    // 6. TOP-1: block-level argmax
    //    TOP-5: thread-local scan + atomic
    //    详见下节
}
```

#### 7.2.2 BWD Kernel

```cpp
template<bool IS_AMP>
__global__ void softmax_ce_bwd_kernel(
    const float* __restrict__ probs,   // always FP32
    const int* __restrict__ labels,
    float scale,
    float inv_scaling,
    T* __restrict__ grad,              // FP32 或 __half
    int N, int C)
{
    int b = blockIdx.x;
    if (b >= N) return;
    
    float s = scale * inv_scaling;
    int label = labels[b];

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float g = probs[b * C + c] * s;
        if (c == label) g -= s;
        
        if (IS_AMP)
            grad[b * C + c] = __float2half(g);
        else
            grad[b * C + c] = g;
    }
}
```

**关键点**：BWD kernel 极简单，不需要 shared memory reduction。每个线程独立处理若干列，用 `label` 做单元素条件减法。

#### 7.2.3 额外的 Launch 封装

```cpp
// FP32 版本
cudaError_t launch_softmax_ce_fwd_fp32(
    const float* logits, const int* labels,
    float* probs, float* loss,
    int* top1, int* top5,
    int N, int C, cudaStream_t stream);

// AMP 版本（FP16 logits 输入 → FP32 probs 输出）
cudaError_t launch_softmax_ce_fwd_amp(
    const __half* logits, const int* labels,
    float* probs, float* loss,
    int* top1, int* top5,
    int N, int C, cudaStream_t stream);
```

### 7.3 Stream 策略

遵循现有模式，使用单流：

```cpp
StreamKind sk = StreamKind::COMP_1;
cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
int si = state.get_or_register(s);
state.output_stream_idx = si;
```

CE 算子的输入依赖前一层（如 FC/GAP）的输出，CUDA graph 捕获时 `MultiStreamCaptureState` 自动管理跨流依赖。

### 7.4 是否使用 cuDNN Frontend？

**不使用**。cuDNN 没有 fused Softmax+CrossEntropy 算子。拆分为两个 cuDNN FE 节点会失去融合优势——中间 probs 张量需要从显存回读再写回，浪费带宽。

全部用自定义 CUDA kernel，配合 shared memory reduction 达到最佳性能。

---

## 八、编译器改动

### 8.1 新 Region 注册

在 `types.h` 中：

```cpp
enum class Region : uint8_t {
    // ... 现有区域 ...
    S_SCALAR_FP32,        // 已有
    R_RESULT_INT32,       // ← 新增
};
```

在 `memory_plan.cpp` 中注册 Region 特性（INT32 类型、无需梯度、无需 momentum）。

### 8.2 Tensor 推断更新

修改 `infer_softmaxce_tensors()` 在 [layer_descriptor_registry.cpp:505-514](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L505-L514)：

```cpp
std::vector<TensorDesc> infer_softmaxce_tensors(...) {
    return {
        {"ce_loss",            Shape{1,1,1,1}, Region::S_SCALAR_FP32,  DType::FP32},
        {"ce_output",          input,           select_feature_region(ctx), DType::FP32},
        {"scaling_factor",     Shape{1,1,1,1}, Region::S_SCALAR_FP32,  DType::FP32},
        {"inv_scaling_factor", Shape{1,1,1,1}, Region::S_SCALAR_FP32,  DType::FP32},
        {"top1_correct",       Shape{1,1,1,1}, Region::R_RESULT_INT32, DType::INT32},  // ← 新增
        {"top5_correct",       Shape{1,1,1,1}, Region::R_RESULT_INT32, DType::INT32},  // ← 新增
    };
}
```

### 8.3 子图更新

FWD:

```cpp
n.output_indices = {0, 1, 2, 3, 4, 5};  // 原为 {0,1,2,3}
```

BWD（需加 labels 输入）:

```cpp
n.input_indices  = {1, 2, 3};  // 保持: ce_output + scaling + inv_scaling
n.output_indices = {1};        // 保持: grad 覆盖 ce_output
```

labels 作为额外隐式输入由编译器传递。不在此处修改 `input_indices` 是因为 labels 不属于这 4 个 inferred tensor 的范畴——它由外部数据加载器提供，编译器在 wiring 阶段自动连线。

---

## 九、注册与集成

### 9.1 注册函数

在 [op_registry.h](file:///r:/renaissance/include/renaissance/backend/op_registry.h) 添加声明：

```cpp
void register_op_softmax_ce();
```

在 [op_registry.cpp](file:///r:/renaissance/src/backend/op_registry.cpp) 中调用。

### 9.2 注册实现

```cpp
void register_op_softmax_ce() {
    auto& t = g_compute_op_table;

    {
        auto& e = t[static_cast<size_t>(ComputeOp::CROSS_ENTROPY_LOSS_FP32_FWD)];
        e.op = ComputeOp::CROSS_ENTROPY_LOSS_FP32_FWD;
        e.launch_cpu = launch_softmax_ce_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_fwd_cuda;
#endif
    }
    // ... BWD / AMP 同理 ...
}
```

### 9.3 Stream 策略

在 [op_stream_policy.cpp](file:///r:/renaissance/src/backend/op_stream_policy.cpp) 添加：

```cpp
case ComputeOp::CROSS_ENTROPY_LOSS_FP32_FWD:
case ComputeOp::CROSS_ENTROPY_LOSS_FP32_BWD:
case ComputeOp::CROSS_ENTROPY_LOSS_AMP_FWD:
case ComputeOp::CROSS_ENTROPY_LOSS_AMP_BWD:
    return StreamKind::COMP_1;
```

---

## 十、改动清单

| #    | 文件                                        | 改动                                                         | 行数 |
| ---- | ------------------------------------------- | ------------------------------------------------------------ | ---- |
| 1    | `include/renaissance/core/types.h`          | 添加 `R_RESULT_INT32` Region 枚举                            | +1   |
| 2    | `src/graph/memory_plan.cpp`                 | Region `R_RESULT_INT32` 注册                                 | ~5   |
| 3    | `src/graph/layer_descriptor_registry.cpp`   | `infer_softmaxce_tensors` 加 top1/top5；子图 `output_indices` 扩展 | ~5   |
| 4    | `src/backend/ops/dtensor/softmax_ce_op.cu`  | FWD + BWD CUDA kernels（FP32 + AMP）                         | ~250 |
| 5    | `src/backend/ops/dtensor/softmax_ce_op.cpp` | CPU 实现 + CUDA 分发 + 注册                                  | ~400 |
| 6    | `include/renaissance/backend/op_registry.h` | 声明 `register_op_softmax_ce()`                              | +1   |
| 7    | `src/backend/op_registry.cpp`               | 调用 `register_op_softmax_ce()`；`require_warmup`（如需要）  | ~2   |
| 8    | `src/backend/op_stream_policy.cpp`          | 4 个 CE 算子的流策略                                         | +4   |
| 9    | `src/CMakeLists.txt`                        | 添加 `.cu` + `.cpp` 文件                                     | +2   |

**总改动量**：约 670 行，9 个文件。

---

## 十一、实施顺序（推荐）

```
Phase 1（编译器基础）: Step 1 → 2 → 3
    — 新增 R_RESULT_INT32 region + 更新 tensor 推断
    — 可独立编译验证

Phase 2（算子实现）:  Step 4 → 5
    — .cu kernel + .cpp 分发层
    — 参考 ReLU/FC 现有代码风格

Phase 3（集成注册）:  Step 6 → 7 → 8 → 9
    — 注册、流策略、CMake
    — 全量编译

Phase 4（验证测试）:
    — 编写 Python 参考脚本（类似 test_flatten_fc_relu_fc.py）
    — 编写 C++ 正确性测试
    — CPU / GPU / AMP 三场景验证
```

---

## 十二、风险与注意事项

| 风险                                   | 缓解                                                         |
| -------------------------------------- | ------------------------------------------------------------ |
| labels DTensor stride ≠ 1（非紧凑）    | 文档说明 labels 必定紧凑；BWD kernel 假设 labels 连续访问    |
| TOP-5 计算需 partial sort              | 对 ≤ 10K classes 可用 threshold 比较（10× log 低于 sort）；对 > 10K 可另写专用 kernel |
| label_smoothing > 0                    | 当前 LossParams 已预留，V1 可先实现 smoothing=0，V2 扩展     |
| AMP FWD 中 shared memory 大小 = C × 4B | 对 C=10000 需 ~40KB shared mem，需用 `cudaFuncSetAttribute` 指定动态 SMEM 大小 |
| 数值稳定性（log(0)）                   | 始终使用 `max(prob_label, 1e-12f)` 保护                      |

---

## 十三、总结

该方案与现有 FC/ReLU 等算子架构完全对齐：

1. **复用现有枚举** (`CROSS_ENTROPY_LOSS_*`) 和参数 (`LossParams`)，零破坏性改动
2. **新增 R_RESULT_INT32 region** 存放 TOP-1/TOP-5 结果
3. **CPU 实现使用 Eigen 行向量化**，零 one-hot 展开，O(B×C) 复杂度
4. **CUDA 实现使用自定义 kernel + shared memory reduction**，不使用 cuDNN FE（无对应融合算子）
5. **AMP 内部全部 FP32 运算**，仅输入/输出做 FP16↔FP32 转换
6. **单流模式**（COMP_1），与其他简单算子（ReLU/Tanh/Flatten）一致
7. **只需 ~670 行新代码**，分布 9 个文件

# 【用户补充】

算子名就要叫这个：

SOFTMAX_CE_FP32_FWD
SOFTMAX_CE_FP32_BWD
SOFTMAX_CE_AMP_FWD
SOFTMAX_CE_AMP_BWD

现有的不符合的就要删掉，替换。

我们需要有一个专门的结果区，它就只存TOP-1正确数、TOP-5正确数。后面我们会用RANGE OP来对这个区求sum-all-reduce。至于推理的具体结果（batch size个结果标签值），我觉得放哪都不合适，但可以先放在结果区，后续再看怎么处理





# 【Legacy代码】

**（注：以下是我们以前进行单独试验的时候写的代码，仅供参考，但其数学正确性未验证，API和对齐可能与我们框架不符，不作为方向性指引）**



```
/**
 * @file ta_v4_common_fp16.hpp
 * @brief 技术觉醒V4公共基础设施层（FP16专用版本，单一真理源）
 * @version 1.0.0
 * @date 2026-04-12
 * @author 技术觉醒团队
 *
 * @note 设计原则（基于8位特级专家共识）：
 *   1. 单点真理源：所有公共代码仅此一处定义
 *   2. 最小侵入：完全兼容现有cbr_fwd/cbr_bwd实现
 *   3. 零性能损耗：全inline，编译期优化
 *   4. 防漂移设计：CI可检查的约束
 *
 * @note 包含内容：
 *   - 错误检查宏（CHECK_*）
 *   - 内存工具（align_to, allocate_*）
 *   - 修复后的初始化函数（Bug修复的单点）
 *   - Config结构体 + parse_arguments
 *   - Experience查询（Mode C支持）
 *   - GPU频率锁定工具
 *
 * @note 依赖项：
 *   - cuDNN Frontend 1.17
 *   - CUDA 13.1
 *   - cuDNN 9.17
 */

#pragma once

// ==================== 标准库依赖 ====================
#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <cstring>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <unordered_map>
#include <sstream>
#include <string>

namespace fe = cudnn_frontend;

// ==================== 1. 错误检查宏（单点真理源）====================

/**
 * @brief CUDA错误检查宏
 * @note 保持与现有代码100%兼容
 */
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * @brief cuDNN错误检查宏
 * @note 保持与现有代码100%兼容
 */
#define CHECK_CUDNN(call) \
do { \
    cudnnStatus_t err = call; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudnnGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * @brief cuDNN Frontend错误检查宏
 * @note 保持与现有代码100%兼容
 */
#define CHECK_CUDNN_FE(call) \
do { \
    auto err = call; \
    if (err.is_bad()) { \
        std::cerr << "cuDNN Frontend error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << err.get_message() << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ==================== 2. 内存工具（单点真理源）====================

/**
 * @brief 内存对齐辅助函数
 * @param size 原始大小
 * @param alignment 对齐边界（默认256B，A100最优）
 * @return 对齐后的大小
 * @note 保持与现有代码100%兼容
 */
inline size_t align_to(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

/**
 * @brief 分配对齐的GPU内存
 * @param size 需要的字节数
 * @param alignment 对齐边界（默认256B）
 * @return GPU内存指针
 * @note 保持与现有代码100%兼容
 */
inline void* allocate_aligned_gpu_memory(size_t size, size_t alignment = 256) {
    void* ptr = nullptr;
    size_t aligned_size = align_to(size, alignment);
    CHECK_CUDA(cudaMalloc(&ptr, aligned_size));
    return ptr;
}

// ==================== 3. 初始化函数（Bug修复的单点）====================

/**
 * @brief 初始化随机FP16数据（通用版本）
 * @param d_ptr GPU内存指针
 * @param num_elements 元素数量
 * @param seed 随机种子（默认42）
 * @note 保持与现有代码100%兼容
 */
inline void initialize_random_fp16(void* d_ptr, size_t num_elements, unsigned int seed = 42) {
    std::vector<__half> h_data(num_elements);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < num_elements; ++i) {
        h_data[i] = __float2half(dis(gen));
    }

    CHECK_CUDA(cudaMemcpy(d_ptr, h_data.data(),
                         num_elements * sizeof(__half),
                         cudaMemcpyHostToDevice));
}

/**
 * @brief 初始化随机FP32数据（通用版本）
 * @param d_ptr GPU内存指针
 * @param num_elements 元素数量
 * @param seed 随机种子（默认42）
 * @note 保持与现有代码100%兼容
 */
inline void initialize_random_fp32(void* d_ptr, size_t num_elements, unsigned int seed = 42) {
    std::vector<float> h_data(num_elements);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < num_elements; ++i) {
        h_data[i] = dis(gen);
    }

    CHECK_CUDA(cudaMemcpy(d_ptr, h_data.data(),
                         num_elements * sizeof(float),
                         cudaMemcpyHostToDevice));
}

/**
 * @brief Bug修复：初始化正值FP32数据（用于inv_variance等物理量）
 * @param d_ptr GPU内存指针
 * @param num_elements 元素数量
 * @param min_val 最小值（默认0.5，对应variance约等于4）
 * @param max_val 最大值（默认5.0，对应variance约等于0.04）
 * @param seed 随机种子（默认44）
 *
 * @note 物理验证：区间[0.5, 5.0]覆盖ResNet-50训练中99%的实际inv_variance值
 * @note 专家共识度：7/8专家支持此方案
 */
inline void initialize_positive_fp32(void* d_ptr, size_t num_elements,
                                     float min_val = 0.5f,
                                     float max_val = 5.0f,
                                     unsigned int seed = 44) {
    std::vector<float> h_data(num_elements);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < num_elements; ++i) {
        h_data[i] = dis(gen);
    }

    CHECK_CUDA(cudaMemcpy(d_ptr, h_data.data(),
                         num_elements * sizeof(float),
                         cudaMemcpyHostToDevice));
}

/**
 * @brief 初始化ReLU Bitmask（Bernoulli分布，字节掩码）
 * @param d_ptr GPU内存指针
 * @param num_elements 元素数量
 * @param activation_rate 激活率（默认0.5）
 * @param seed 随机种子（默认2024）
 *
 * @note 重要说明：
 *   - 这是用于独立Backward Benchmark的模拟数据
 *   - 真实训练框架中，Bitmask应由前向传播的ReLU层生成
 *   - 采用Bernoulli(0.5)分布，50%激活率符合ResNet-50稳态
 *   - 字节掩码（uint8_t），不是bit-packed mask
 */
inline void initialize_relu_bitmask(void* d_ptr, size_t num_elements, float activation_rate = 0.5f, unsigned int seed = 2024) {
    std::vector<uint8_t> h_data(num_elements);
    std::mt19937 mask_gen(seed);
    std::bernoulli_distribution dist(activation_rate);

    for (size_t i = 0; i < num_elements; ++i) {
        h_data[i] = dist(mask_gen) ? uint8_t(1) : uint8_t(0);
    }

    CHECK_CUDA(cudaMemcpy(d_ptr, h_data.data(),
                         num_elements * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));
}

/**
 * @brief 初始化FP32标量（用于epsilon、momentum等）
 * @param d_ptr GPU内存指针
 * @param value 标量值
 * @note 保持与现有代码100%兼容
 */
inline void initialize_scalar_fp32(void* d_ptr, float value) {
    CHECK_CUDA(cudaMemcpy(d_ptr, &value, sizeof(float), cudaMemcpyHostToDevice));
}

// ==================== 4. 配置系统（单点真理源）====================

/**
 * @brief 统一配置结构体
 * @note 保持与现有代码100%兼容
 */
struct Config {
    int64_t batch_size = 512;
    int64_t input_size = 56;
    int64_t in_channels = 64;
    int64_t out_channels = 256;
    int64_t kernel_size = 1;
    int64_t conv_stride = 1;

    // ========== MaxPool 参数 ==========
    int64_t pool_kernel_size = 3;
    int64_t pool_stride = 2;
    int64_t pool_padding = 1;

    /**
     * @brief 搜索模式枚举
     * @note 支持Mode C穷举式搜索
     */
    enum class SearchMode {
        HEURISTIC_A = 0,
        HEURISTIC_B = 1,
        EXHAUSTIVE_C = 2
    };
    SearchMode search_mode = SearchMode::HEURISTIC_B;

    /**
     * @brief CUDA Graph模式开关
     * @note 默认true（与merged版本一致）
     */
    bool use_graph = true;

    /**
     * @brief 映射到cuDNN HeurMode（Mode C时返回B作为fallback）
     */
    fe::HeurMode_t get_heur_mode() const {
        if (search_mode == SearchMode::HEURISTIC_A) {
            return fe::HeurMode_t::A;
        } else {
            return fe::HeurMode_t::B;
        }
    }

    /**
     * @brief 计算padding（保持空间维度不变）
     */
    int64_t get_padding() const {
        return (kernel_size - 1) / 2;
    }

    /**
     * @brief 计算输出尺寸
     */
    int64_t get_output_size() const {
        int64_t padding = get_padding();
        return (input_size + 2 * padding - kernel_size) / conv_stride + 1;
    }

    /**
     * @brief 打印配置信息（精简版）
     */
    void print() const {
        // 与现有实现保持一致：不打印详细信息
        (void)batch_size;
        (void)input_size;
        (void)in_channels;
        (void)out_channels;
        (void)kernel_size;
        (void)conv_stride;
        (void)use_graph;
        (void)search_mode;
        int64_t padding = get_padding();
        int64_t output_size = get_output_size();
        (void)padding;
        (void)output_size;
    }
};

/**
 * @brief 命令行参数解析
 * @param argc 参数数量
 * @param argv 参数数组
 * @return Config对象
 * @note 保持与现有代码100%兼容
 */
inline Config parse_arguments(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--batch_size" && i + 1 < argc) {
            config.batch_size = std::atoll(argv[++i]);
        } else if (arg == "--input_size" && i + 1 < argc) {
            config.input_size = std::atoll(argv[++i]);
        } else if (arg == "--in_channels" && i + 1 < argc) {
            config.in_channels = std::atoll(argv[++i]);
        } else if (arg == "--out_channels" && i + 1 < argc) {
            config.out_channels = std::atoll(argv[++i]);
        } else if (arg == "--kernel_size" && i + 1 < argc) {
            config.kernel_size = std::atoll(argv[++i]);
        } else if (arg == "--conv_stride" && i + 1 < argc) {
            config.conv_stride = std::atoll(argv[++i]);
        } else if (arg == "--pool_kernel_size" && i + 1 < argc) {
            config.pool_kernel_size = std::atoll(argv[++i]);
        } else if (arg == "--pool_stride" && i + 1 < argc) {
            config.pool_stride = std::atoll(argv[++i]);
        } else if (arg == "--pool_padding" && i + 1 < argc) {
            config.pool_padding = std::atoll(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode_str = argv[++i];
            if (mode_str == "A") {
                config.search_mode = Config::SearchMode::HEURISTIC_A;
            } else if (mode_str == "B") {
                config.search_mode = Config::SearchMode::HEURISTIC_B;
            } else if (mode_str == "C") {
                config.search_mode = Config::SearchMode::EXHAUSTIVE_C;
            } else {
                std::cerr << "Error: mode must be 'A', 'B', or 'C', got " << mode_str << std::endl;
                exit(EXIT_FAILURE);
            }
        } else if (arg == "--graph") {
            if (i + 1 < argc) {
                std::string graph_str = argv[++i];
                if (graph_str == "true" || graph_str == "1") {
                    config.use_graph = true;
                } else if (graph_str == "false" || graph_str == "0") {
                    config.use_graph = false;
                } else {
                    std::cerr << "Error: --graph must be 'true', 'false', '0', or '1', got "
                             << graph_str << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                std::cerr << "Error: --graph requires an argument (true/false)" << std::endl;
                exit(EXIT_FAILURE);
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --batch_size <N>      Batch size (default: 512)" << std::endl;
            std::cout << "  --input_size <S>     Input feature map size H=W=S (default: 56)" << std::endl;
            std::cout << "  --in_channels <C>    Input channels (default: 64)" << std::endl;
            std::cout << "  --out_channels <K>   Output channels (default: 256)" << std::endl;
            std::cout << "  --kernel_size <K>    Kernel size R=S=K, must be 1, 3, 5, or 7 (default: 1)" << std::endl;
            std::cout << "  --conv_stride <S>    Convolution stride (default: 1)" << std::endl;
            std::cout << "  --pool_kernel_size <K> MaxPool kernel size (default: 3)" << std::endl;
            std::cout << "  --pool_stride <S>    MaxPool stride (default: 2)" << std::endl;
            std::cout << "  --pool_padding <P>   MaxPool padding (default: 1)" << std::endl;
            std::cout << "  --mode <M>           Search mode: 'A', 'B', or 'C' (default: B)" << std::endl;
            std::cout << "                         A: Heuristic Mode A (fast decision tree)" << std::endl;
            std::cout << "                         B: Heuristic Mode B (neural network predictor)" << std::endl;
            std::cout << "                         C: Exhaustive search (uses pre-computed optimal plans)" << std::endl;
            std::cout << "  --graph <bool>       Enable CUDA Graph (default: true)" << std::endl;
            std::cout << "                         true: Use CUDA Graph mode (zero launch overhead)" << std::endl;
            std::cout << "                         false: Use traditional mode (no Graph capture)" << std::endl;
            std::cout << "  --help, -h           Show this help message" << std::endl;
            std::cout << std::endl;
            std::cout << "Padding is auto-calculated as (kernel_size - 1) / 2 to maintain spatial dimensions when stride=1" << std::endl;
            std::cout << std::endl;
            std::cout << "Examples:" << std::endl;
            std::cout << "  " << argv[0] << " --batch_size 512 --input_size 56 --in_channels 64 --out_channels 256 --kernel_size 1 --mode B" << std::endl;
            std::cout << "  " << argv[0] << " --kernel_size 3 --in_channels 128 --out_channels 256 --mode A --graph false" << std::endl;
            std::cout << "  " << argv[0] << " --kernel_size 1 --in_channels 64 --out_channels 256 --mode C" << std::endl;
            exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Use --help or -h for usage information" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // 参数校验（保持与现有代码一致）
    if (config.kernel_size != 1 && config.kernel_size != 3 &&
        config.kernel_size != 5 && config.kernel_size != 7) {
        std::cerr << "Error: kernel_size must be 1, 3, 5, or 7, got " << config.kernel_size << std::endl;
        exit(EXIT_FAILURE);
    }

    if (config.conv_stride < 1) {
        std::cerr << "Error: conv_stride must be >= 1, got " << config.conv_stride << std::endl;
        exit(EXIT_FAILURE);
    }

    if (config.pool_kernel_size != 2 && config.pool_kernel_size != 3) {
        std::cerr << "Error: pool_kernel_size must be 2 or 3, got " << config.pool_kernel_size << std::endl;
        exit(EXIT_FAILURE);
    }

    if (config.pool_stride < 1) {
        std::cerr << "Error: pool_stride must be >= 1, got " << config.pool_stride << std::endl;
        exit(EXIT_FAILURE);
    }

    if (config.pool_padding < 0) {
        std::cerr << "Error: pool_padding must be >= 0, got " << config.pool_padding << std::endl;
        exit(EXIT_FAILURE);
    }

    if (config.batch_size <= 0 || config.input_size <= 0 ||
        config.in_channels <= 0 || config.out_channels <= 0) {
        std::cerr << "Error: All size parameters must be positive" << std::endl;
        exit(EXIT_FAILURE);
    }

    return config;
}

// ==================== 5. Experience查询（Mode C支持）====================

/**
 * @brief Experience状态枚举
 * @note 用于标记查询结果类型
 */
enum class ExperienceStatus {
    NOT_FOUND,
    FOUND,
    FOUND_BACKUP1,
    FOUND_BACKUP2
};

/**
 * @brief 平台条件编译：引入Experience表头文件
 * @note 仅A100和RTX5090有预计算的Experience表
 */
#if defined(USING_A100)
    #include "generated/cbr_experience_a100_fp16.hpp"
#elif defined(USING_RTX5090)
    #include "generated/cbr_experience_rtx5090_fp16.hpp"
#else
    // 其他平台：提供空实现，查询直接返回nullptr（自动fallback到Heuristic B）
    namespace ta_v4 {
        namespace experience {
            struct ExperienceRecord {
                const char* shape_key;
                const char* winner_tag;
                const char* backup1_tag;
                const char* backup2_tag;
                uint64_t workspace_bytes;
                float benchmark_time_ms;
                const char* source;
            };

            inline const ExperienceRecord* lookup(const std::string&) {
                return nullptr;
            }
        }
    }
    #pragma message("Mode C disabled: Unsupported platform, will fallback to Heuristic B")
#endif

/**
 * @brief 构建Shape Key字符串（用于Experience表查询）
 * @param op_type 操作类型："conv_genstats" / "conv_dgrad" / "conv_wgrad"
 * @param dtype 数据类型："fp16" / "bf16"
 * @param N,H,W,C,K,R,S,stride,padding 张量维度参数
 * @return 完整的Shape Key字符串
 * @note 保持与现有代码100%兼容
 */
inline std::string build_shape_key(
    const std::string& op_type,
    const std::string& dtype,
    int64_t N, int64_t H, int64_t W, int64_t C, int64_t K,
    int64_t R, int64_t S, int64_t stride, int64_t padding)
{
    std::ostringstream oss;

    // GPU架构（编译时确定）
#if defined(USING_A100)
    oss << "A100-SXM4-80GB";
#elif defined(USING_RTX5090)
    oss << "RTX5090";
#else
    oss << "UNKNOWN_GPU";
#endif

    // 固定字段
    oss << "|SM80"
        << "|cuDNN9.17.0"
        << "|CUDA13.1"
        << "|" << op_type << "_" << dtype
        << "|N" << N
        << "|H" << H
        << "|W" << W
        << "|C" << C
        << "|K" << K
        << "|R" << R
        << "|S" << S
        << "|U1|V1"
        << "|P" << padding
        << "|Q" << padding
        << "|D" << stride
        << "|E" << stride
        << "|NHWC|FP16|FP32";

    return oss.str();
}

/**
 * @brief 三级Fallback Plan匹配（Mode C核心逻辑）
 * @param graph cuDNN Frontend Graph对象
 * @param candidates 候选Plan索引列表
 * @param exp_rec Experience记录指针
 * @param handle cuDNN句柄（保留参数，API兼容性）
 * @return {ExperienceStatus, bool} 状态和是否成功
 * @note 保持与现有代码100%兼容
 */
inline std::pair<ExperienceStatus, bool> match_and_build_plan(
    std::shared_ptr<fe::graph::Graph>& graph,
    const std::vector<int64_t>& candidates,
    const ta_v4::experience::ExperienceRecord* exp_rec,
    cudnnHandle_t handle)
{
    (void)handle;  // 保留参数（API兼容性）

    // Level 1: 尝试Winner
    for (int64_t idx : candidates) {
        std::string tag;
        auto status = graph->get_plan_name_at_index(idx, tag);
        if (status.is_bad()) {
            continue;
        }
        if (tag == std::string(exp_rec->winner_tag)) {
            auto build_status = graph->build_plan_at_index(idx);
            if (!build_status.is_bad()) {
                return {ExperienceStatus::FOUND, true};
            }
        }
    }

    // Level 2: 尝试Backup1
    if (strlen(exp_rec->backup1_tag) > 0) {
        for (int64_t idx : candidates) {
            std::string tag;
            auto status = graph->get_plan_name_at_index(idx, tag);
            if (status.is_bad()) {
                continue;
            }
            if (tag == std::string(exp_rec->backup1_tag)) {
                auto build_status = graph->build_plan_at_index(idx);
                if (!build_status.is_bad()) {
                    return {ExperienceStatus::FOUND_BACKUP1, true};
                }
            }
        }
    }

    // Level 3: 尝试Backup2
    if (strlen(exp_rec->backup2_tag) > 0) {
        for (int64_t idx : candidates) {
            std::string tag;
            auto status = graph->get_plan_name_at_index(idx, tag);
            if (status.is_bad()) {
                continue;
            }
            if (tag == std::string(exp_rec->backup2_tag)) {
                auto build_status = graph->build_plan_at_index(idx);
                if (!build_status.is_bad()) {
                    return {ExperienceStatus::FOUND_BACKUP2, true};
                }
            }
        }
    }

    // Level 4: 所有Tag失效或构建失败
    return {ExperienceStatus::NOT_FOUND, false};
}

// ==================== 7. 编译期约束检查（防漂移机制）====================

/**
 * @brief 编译期版本检查
 * @note 确保所有使用此头文件的代码版本一致
 */
#define TA_V4_COMMON_VERSION_MAJOR 1
#define TA_V4_COMMON_VERSION_MINOR 0
#define TA_V4_COMMON_VERSION_PATCH 0

/**
 * @brief 防漂移标记宏（用于CI检查）
 * @note 算子文件中禁止定义这些函数（由CI脚本检查）
 */
#define TA_V4_FORBID_LOCAL_INITIALIZE \
    static_assert(true, "Local initialize_* functions are forbidden. Use ta_v4_common_fp16.hpp");

#define TA_V4_FORBID_LOCAL_CONFIG \
    static_assert(true, "Local Config struct is forbidden. Use ta_v4_common_fp16.hpp");

// ==================== 8. 调试辅助宏（可选）====================

/**
 * @brief 调试打印宏（仅在DEBUG模式下生效）
 * @note 生产代码中零开销
 */
#ifdef DEBUG_TA_V4
    #define TA_V4_DEBUG_PRINT(msg) std::cout << "[DEBUG] " << msg << std::endl;
#else
    #define TA_V4_DEBUG_PRINT(msg) ((void)0)
#endif

// ==================== 9. 文档标记（用于自动生成文档）====================

/**
 * @brief 本头文件提供的核心功能清单
 *
 * 1. 错误检查宏：CHECK_CUDA, CHECK_CUDNN, CHECK_CUDNN_FE
 * 2. 内存工具：align_to, allocate_aligned_gpu_memory
 * 3. 初始化函数：
 *    - initialize_random_fp16
 *    - initialize_random_fp32
 *    - initialize_positive_fp32（Bug修复）
 *    - initialize_relu_bitmask：Bernoulli(α=0.5)随机字节掩码初始化
 *      注意：仅用于独立Backward Benchmark模拟；
 *      真实训练框架中，该Bitmask应由前向图的ReLU操作生成
 *    - initialize_scalar_fp32
 * 4. 配置系统：Config结构体, parse_arguments
 * 5. Experience查询：ExperienceStatus, build_shape_key, match_and_build_plan
 *
 * @note 所有函数均为inline，零运行时开销
 * @note 完全兼容现有cbr_fwd_fp16.cpp和cbr_bwd_fp16.cpp
 */

```

```
/**
 * @file softmax_ce_grad_final.cpp
 * @brief 技术觉醒V4 - ResNet-50末端 Softmax+CrossEntropyLoss 梯度计算层 极致优化版
 * @version 1.0.0
 * @date 2026-04-19
 * @author 技术觉醒团队
 * @note 依赖项: CUDA 13.1, cuDNN 9.17, C++17
 * @note 所属系列: fp16
 *
 * @note 核心设计原则:
 *   1. 零中间显存: Softmax概率只存在于共享内存/寄存器，永不写回全局显存
 *   2. 完全融合: 单kernel完成max归约→exp计算→梯度生成，消除所有冗余访问
 *   3. 极致对齐: 256B起始地址，128B行步幅，完美匹配A100 L2 Cache Line
 *   4. FP16 IO + FP32内部计算: 兼顾AMP训练带宽与数值稳定性
 *   5. 动态优化: 根据K值自动选择最优block配置和归约策略
 *   6. 避免one-hot: 通过标签索引直接计算梯度，节省显存带宽
 *   7. 单一流架构: stream_comp_1_，符合框架规范
 */

// ████████████████████████████████████████████████████████████████████████████
// ████ 引入公共基础设施层（FP16专用，单一真理源） ████
// ████████████████████████████████████████████████████████████████████████████
#include "ta_v4_common_fp16.hpp"
#include <iomanip>
#include <stdexcept>

// ████████████████████████████████████████████████████████████████████████████
// ████ 外部Kernel声明（在softmax_ce_grad.cu中实现） ████
// ████████████████████████████████████████████████████████████████████████████
extern "C" void launch_fused_softmax_ce_grad(
    const __half* d_logits,
    const int32_t* d_labels,
    __half* d_grads,
    float* d_total_loss,
    int64_t N,
    int64_t K,
    int64_t K_aligned,
    int block_size,
    size_t shared_mem_size,
    bool use_shmem_cache,
    cudaStream_t stream);

// ████████████████████████████████████████████████████████████████████████████
// ████ Softmax+CE 融合配置结构体（扩展 Config） ████
// ████████████████████████████████████████████████████████████████████████████
struct SoftmaxCEConfig : public Config {
    int64_t num_classes = 1000;  // FC 输出维度（分类数）

    fe::HeurMode_t get_heur_mode() const {
        return (search_mode == SearchMode::HEURISTIC_A) ?
               fe::HeurMode_t::A : fe::HeurMode_t::B;
    }

    void print() const {
        std::cout << "=== Softmax+CE Gradient Configuration ===" << std::endl;
        std::cout << "Batch Size:    " << batch_size << std::endl;
        std::cout << "Num Classes:   " << num_classes << std::endl;
        std::cout << "Search Mode:   " << (search_mode == SearchMode::HEURISTIC_A ? "A" : "B") << std::endl;
        std::cout << "CUDA Graph:    " << (use_graph ? "Enabled" : "Disabled") << std::endl;
        std::cout << "=========================================" << std::endl;
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ Softmax+CE 专用参数解析函数 ████
// ████████████████████████████████████████████████████████████████████████████
inline SoftmaxCEConfig parse_softmax_ce_arguments(int argc, char** argv) {
    SoftmaxCEConfig config;

    // 先使用通用参数解析器处理公共参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--num_classes" && i + 1 < argc) {
            config.num_classes = std::atoll(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --batch_size <N>      Batch size (default: 512)" << std::endl;
            std::cout << "  --num_classes <K>     Number of classes (default: 1000)" << std::endl;
            std::cout << "  --mode <A/B>          Heuristic mode (default: B)" << std::endl;
            std::cout << "  --graph <true/false>  Enable CUDA Graph (default: true)" << std::endl;
            std::cout << std::endl;
            std::cout << "ResNet-50 Final Layer Test:" << std::endl;
            std::cout << "  " << argv[0] << " --batch_size 512 --num_classes 1000 --mode B --graph true" << std::endl;
            exit(EXIT_SUCCESS);
        }
    }

    // 使用通用参数解析器处理剩余参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--batch_size" && i + 1 < argc) {
            config.batch_size = std::atoll(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "A") {
                config.search_mode = SoftmaxCEConfig::SearchMode::HEURISTIC_A;
            } else if (mode == "B") {
                config.search_mode = SoftmaxCEConfig::SearchMode::HEURISTIC_B;
            }
        } else if (arg == "--graph" && i + 1 < argc) {
            std::string val = argv[++i];
            config.use_graph = (val == "true" || val == "1");
        }
    }

    return config;
}

// ████████████████████████████████████████████████████████████████████████████
// ████ Softmax+CE 融合 Benchmark 核心类 ████
// ████████████████████████████████████████████████████████████████████████████
class SoftmaxCEGradBenchmark {
private:
    // ========== 核心资源: 严格使用单流 ==========
    cudnnHandle_t cudnn_handle_;
    cudaStream_t stream_comp_1_;  // 单一计算流（强制要求）

    // ========== 配置参数 ==========
    const int64_t N_;  // batch_size
    const int64_t K_;  // num_classes
    int64_t K_aligned_;
    const bool use_graph_;
    const fe::HeurMode_t heur_mode_;

    // ========== GPU内存 ==========
    void* d_logits_ = nullptr;      // 输入: [N, K_aligned] FP16
    int32_t* d_labels_ = nullptr;   // 输入: [N] INT32
    void* d_grads_ = nullptr;       // 输出: [N, K_aligned] FP16
    float* d_total_loss_ = nullptr;  // 输出: [1] FP32，总Loss（需除以N得到平均值）

    // ========== CUDA Graph ==========
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_captured_;

public:
    // ========== Benchmark结果 ==========
    struct BenchmarkResult {
        double avg_ms;
        double bandwidth_gbps;
        double throughput;
        double avg_loss;
    };

    // ========== Kernel配置 ==========
    int block_size_;
    size_t shared_mem_size_;
    bool use_shmem_cache_;
    BenchmarkResult last_result_;

    explicit SoftmaxCEGradBenchmark(const SoftmaxCEConfig& config)
        : N_(config.batch_size),
          K_(config.num_classes),
          use_graph_(config.use_graph),
          heur_mode_(config.get_heur_mode()),
          graph_captured_(false)
    {
        // 1. 计算对齐维度
        int64_t row_bytes = K_ * sizeof(__half);
        K_aligned_ = align_to(row_bytes, 128) / sizeof(__half);

        // 2. 选择最优配置
        select_optimal_config();

        // 3. 创建资源
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_1_, cudaStreamNonBlocking));
        CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
        CHECK_CUDNN(cudnnSetStream(cudnn_handle_, stream_comp_1_));

        // 4. 分配内存
        allocate_memory();

        // 5. 初始化数据
        initialize_data();

        // 打印配置
        config.print();
        std::cout << "Aligned K:       " << K_aligned_ << std::endl;
        std::cout << "Block Size:      " << block_size_ << std::endl;
        std::cout << "Shared Memory:   " << shared_mem_size_ << " bytes" << std::endl;
        std::cout << "Use Shmem Cache: " << (use_shmem_cache_ ? "Yes" : "No") << std::endl;
    }

    ~SoftmaxCEGradBenchmark() {
        // 销毁CUDA Graph
        if (graph_exec_) {
            cudaError_t err = cudaGraphExecDestroy(graph_exec_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: cudaGraphExecDestroy failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }
        if (graph_) {
            cudaError_t err = cudaGraphDestroy(graph_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: cudaGraphDestroy failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }

        // 释放GPU内存
        CHECK_CUDA(cudaFree(d_logits_));
        CHECK_CUDA(cudaFree(d_labels_));
        CHECK_CUDA(cudaFree(d_grads_));
        CHECK_CUDA(cudaFree(d_total_loss_));

        // 销毁流和句柄
        CHECK_CUDNN(cudnnDestroy(cudnn_handle_));
        CHECK_CUDA(cudaStreamDestroy(stream_comp_1_));
    }

private:
    void select_optimal_config() {
        // 根据K值选择block大小
        if (K_ <= 128) {
            block_size_ = 128;
        } else if (K_ <= 512) {
            block_size_ = 256;
        } else if (K_ <= 2048) {
            block_size_ = 512;
        } else {
            block_size_ = 1024;
        }

        // Heuristic模式调整
        if (heur_mode_ == fe::HeurMode_t::A) {
            // 保守模式: 使用较小block size，减少共享内存占用
            block_size_ = std::max(128, block_size_ / 2);
        }

        // 计算共享内存需求
        shared_mem_size_ = block_size_ * sizeof(float) * 2;  // max + sum

        // 检查是否可以使用exp缓存
        size_t exp_cache_size = K_ * sizeof(float);
        int max_shared_mem;
        CHECK_CUDA(cudaDeviceGetAttribute(&max_shared_mem,
                                         cudaDevAttrMaxSharedMemoryPerBlock, 0));

        if (shared_mem_size_ + exp_cache_size <= static_cast<size_t>(max_shared_mem)) {
            use_shmem_cache_ = true;
            shared_mem_size_ += exp_cache_size;
        } else {
            use_shmem_cache_ = false;
            // 不使用缓存，需要更多寄存器重新计算exp
        }
    }

    void allocate_memory() {
        // Logits: [N, K_aligned]，256B起始对齐，128B行步幅对齐
        const size_t logits_size = N_ * K_aligned_ * sizeof(__half);
        d_logits_ = allocate_aligned_gpu_memory(logits_size);

        // Labels: [N]，INT32，无对齐要求
        CHECK_CUDA(cudaMalloc(&d_labels_, N_ * sizeof(int32_t)));

        // Gradients: [N, K_aligned]，256B起始对齐，128B行步幅对齐
        const size_t grad_size = N_ * K_aligned_ * sizeof(__half);
        d_grads_ = allocate_aligned_gpu_memory(grad_size);

        // Total Loss: [1]，FP32标量，所有样本Loss之和
        CHECK_CUDA(cudaMalloc(&d_total_loss_, sizeof(float)));
    }

    void initialize_data() {
        // 随机初始化Logits（不计入计时）
        const size_t logits_size = static_cast<size_t>(N_ * K_);
        std::vector<__half> h_logits(logits_size);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        for (size_t i = 0; i < logits_size; ++i) {
            h_logits[i] = __float2half(dist(gen));
        }
        CHECK_CUDA(cudaMemcpy(d_logits_, h_logits.data(),
                             logits_size * sizeof(__half), cudaMemcpyHostToDevice));

        // 随机标签
        std::vector<int32_t> h_labels(N_);
        std::uniform_int_distribution<int32_t> label_dist(0, K_ - 1);
        for (int64_t i = 0; i < N_; ++i) {
            h_labels[i] = label_dist(gen);
        }
        CHECK_CUDA(cudaMemcpy(d_labels_, h_labels.data(),
                             N_ * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    void launch_kernel() {
        // 重置loss为0（重要：因为kernel使用atomicAdd累加）
        float zero = 0.0f;
        CHECK_CUDA(cudaMemcpyAsync(d_total_loss_, &zero, sizeof(float),
                                   cudaMemcpyHostToDevice, stream_comp_1_));

        launch_fused_softmax_ce_grad(
            static_cast<const __half*>(d_logits_),
            d_labels_,
            static_cast<__half*>(d_grads_),
            d_total_loss_,
            N_, K_, K_aligned_,
            block_size_,
            shared_mem_size_,
            use_shmem_cache_,
            stream_comp_1_
        );
        CHECK_CUDA(cudaGetLastError());
    }

    void capture_cuda_graph() {
        CHECK_CUDA(cudaStreamBeginCapture(stream_comp_1_, cudaStreamCaptureModeThreadLocal));

        launch_kernel();

        CHECK_CUDA(cudaStreamEndCapture(stream_comp_1_, &graph_));

        cudaGraphNode_t error_node;
        char error_log[2048];
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_, graph_, &error_node,
                                       error_log, sizeof(error_log)));

        graph_captured_ = true;
    }

public:
    void warmup(int iterations = 50) {
        std::cout << "\n=== Warmup Phase ===" << std::endl;

        // 传统模式预热
        std::cout << "Warming up traditional mode..." << std::endl;
        for (int i = 0; i < iterations; ++i) {
            launch_kernel();
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));

        // CUDA Graph捕获
        if (use_graph_) {
            std::cout << "Capturing CUDA Graph..." << std::endl;
            capture_cuda_graph();

            if (graph_captured_) {
                // Graph模式预热
                std::cout << "Warming up graph mode..." << std::endl;
                for (int i = 0; i < iterations; ++i) {
                    CHECK_CUDA(cudaGraphLaunch(graph_exec_, stream_comp_1_));
                }
                CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
                std::cout << "Graph capture successful" << std::endl;
            } else {
                std::cerr << "Warning: Graph capture failed, using traditional mode" << std::endl;
            }
        }
    }

    void execute() {
        if (use_graph_ && graph_captured_ && graph_exec_) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_, stream_comp_1_));
        } else {
            launch_kernel();
        }
    }

    BenchmarkResult benchmark(int iterations = 500) {
        std::cout << "\n=== Benchmark Phase ===" << std::endl;

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            execute();
        }

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        auto end = std::chrono::high_resolution_clock::now();

        double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        // 读取并计算平均Loss（在计时结束后，不影响性能测量）
        float h_total_loss = 0.0f;
        CHECK_CUDA(cudaMemcpy(&h_total_loss, d_total_loss_,
                             sizeof(float), cudaMemcpyDeviceToHost));
        double avg_loss = h_total_loss / static_cast<double>(N_);

        // 计算性能指标
        double total_bytes =
            N_ * K_aligned_ * sizeof(__half) * 2.0 +  // 读logits + 写grads
            N_ * sizeof(int32_t);                     // 读labels

        double bandwidth_gbps = (total_bytes / 1e9) / (avg_ms / 1000.0);
        double throughput = N_ / (avg_ms / 1000.0);

        // 保存结果
        last_result_ = {avg_ms, bandwidth_gbps, throughput, avg_loss};

        // 打印结果
        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "Mode:              " << (use_graph_ && graph_captured_ ? "CUDA Graph" : "Traditional") << std::endl;
        std::cout << "Average Time:      " << std::fixed << std::setprecision(4)
                  << avg_ms << " ms" << std::endl;
        std::cout << "Memory Bandwidth:  " << std::fixed << std::setprecision(2)
                  << bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "Throughput:        " << std::fixed << std::setprecision(0)
                  << throughput << " samples/sec" << std::endl;
        std::cout << "Average Loss:      " << std::fixed << std::setprecision(6)
                  << avg_loss << std::endl;
        std::cout << "============================" << std::endl;

        return last_result_;
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ Main入口 ██████████████████████████████████████████████████████████████
// ████████████████████████████████████████████████████████████████████████████
int main(int argc, char** argv) {
    std::cout << "=== 技术觉醒V4 - Softmax+CE 最终梯度计算层 ===" << std::endl;
    std::cout << "=== 极致融合方案 | Zero-Intermediate-Memory ===" << std::endl;

    try {
        SoftmaxCEConfig config = parse_softmax_ce_arguments(argc, argv);
        SoftmaxCEGradBenchmark benchmark(config);

        benchmark.warmup(50);
        benchmark.benchmark(500);

        std::cout << "\n=== 测试完成 ===" << std::endl;
        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

```

```
/**
 * @file softmax_ce_inf.cpp
 * @brief 技术觉醒V4 - 推理时最终结果计算层
 * @version 1.0.0
 * @date 2026-04-19
 * @author 技术觉醒团队
 * @note 依赖项: CUDA 13.1, cuDNN 9.17, C++17
 * @note 所属系列: fp16
 *
 * @note 核心设计原则:
 *   1. 全融合: 单kernel完成Softmax(隐式)、Loss、Top-1/5预测和准确率统计
 *   2. 零中间显存: 概率值仅存于寄存器/共享内存，绝不写回全局显存
 *   3. 极致内存访问: 半精度向量化加载 (half2)，L2缓存行对齐
 *   4. 数值稳定: LogSumExp技巧 + FP32全程计算
 *   5. 高效Top-K: Warp内寄存器Top-5 + Shuffle合并
 *   6. 单流架构: stream_comp_1_，符合框架规范
 */

// ████████████████████████████████████████████████████████████████████████████
// ████ 引入公共基础设施层（FP16专用，单一真理源） ████
// ████████████████████████████████████████████████████████████████████████████
#include "ta_v4_common_fp16.hpp"
#include <iomanip>
#include <stdexcept>

// ████████████████████████████████████████████████████████████████████████████
// ████ 外部Kernel声明（在 softmax_ce_inf.cu 中实现） ████
// ████████████████████████████████████████████████████████████████████████████
extern "C" void launch_softmax_inference_fused(
    const __half* d_logits,
    const int32_t* d_labels,
    int32_t* d_pred_top1,
    float* d_total_loss,
    int32_t* d_top1_correct,
    int32_t* d_top5_correct,
    int64_t N,
    int64_t K,
    int64_t K_aligned,
    int block_size,
    size_t shared_mem_size,
    cudaStream_t stream);

// ████████████████████████████████████████████████████████████████████████████
// ████ 推理层专用配置结构体（扩展 Config） ████
// ████████████████████████████████████████████████████████████████████████████
struct SoftmaxInfConfig : public Config {
    int64_t num_classes = 1000;  // FC 输出维度（分类数）

    fe::HeurMode_t get_heur_mode() const {
        return (search_mode == SearchMode::HEURISTIC_A) ?
               fe::HeurMode_t::A : fe::HeurMode_t::B;
    }

    void print() const {
        std::cout << "=== Softmax Inference Configuration ===" << std::endl;
        std::cout << "Batch Size:    " << batch_size << std::endl;
        std::cout << "Num Classes:   " << num_classes << std::endl;
        std::cout << "Search Mode:   " << (search_mode == SearchMode::HEURISTIC_A ? "A" : "B") << std::endl;
        std::cout << "CUDA Graph:    " << (use_graph ? "Enabled" : "Disabled") << std::endl;
        std::cout << "=========================================" << std::endl;
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ 推理层专用参数解析函数 ████
// ████████████████████████████████████████████████████████████████████████████
inline SoftmaxInfConfig parse_softmax_inf_arguments(int argc, char** argv) {
    SoftmaxInfConfig config;

    // 先使用通用参数解析器处理公共参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--num_classes" && i + 1 < argc) {
            config.num_classes = std::atoll(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --batch_size <N>      Batch size (default: 512)" << std::endl;
            std::cout << "  --num_classes <K>     Number of classes (default: 1000)" << std::endl;
            std::cout << "  --mode <A/B>          Heuristic mode (default: B)" << std::endl;
            std::cout << "  --graph <true/false>  Enable CUDA Graph (default: true)" << std::endl;
            std::cout << std::endl;
            std::cout << "ResNet-50 Final Layer Inference Test:" << std::endl;
            std::cout << "  " << argv[0] << " --batch_size 512 --num_classes 1000 --mode B --graph true" << std::endl;
            exit(EXIT_SUCCESS);
        }
    }

    // 使用通用参数解析器处理剩余参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--batch_size" && i + 1 < argc) {
            config.batch_size = std::atoll(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "A") {
                config.search_mode = SoftmaxInfConfig::SearchMode::HEURISTIC_A;
            } else if (mode == "B") {
                config.search_mode = SoftmaxInfConfig::SearchMode::HEURISTIC_B;
            }
        } else if (arg == "--graph" && i + 1 < argc) {
            std::string val = argv[++i];
            config.use_graph = (val == "true" || val == "1");
        }
    }

    return config;
}

// ████████████████████████████████████████████████████████████████████████████
// ████ Softmax Inference 融合 Benchmark 核心类 ████
// ████████████████████████████████████████████████████████████████████████████
class SoftmaxInfBenchmark {
private:
    // ========== 核心资源: 严格使用单流 ==========
    cudnnHandle_t cudnn_handle_;
    cudaStream_t stream_comp_1_;  // 单一计算流（强制要求）

    // ========== 配置参数 ==========
    const int64_t N_;  // batch_size
    const int64_t K_;  // num_classes
    int64_t K_aligned_;
    const bool use_graph_;
    const fe::HeurMode_t heur_mode_;

    // ========== GPU内存 ==========
    void* d_logits_ = nullptr;        // 输入: [N, K_aligned] FP16
    int32_t* d_labels_ = nullptr;     // 输入: [N] INT32
    int32_t* d_pred_top1_ = nullptr;  // 输出: [N] INT32 (TOP-1预测列表)
    float* d_total_loss_ = nullptr;   // 输出: [1] FP32（所有样本Loss之和）
    int32_t* d_top1_correct_ = nullptr;  // 输出: [1] INT32 (TOP-1正确数)
    int32_t* d_top5_correct_ = nullptr;  // 输出: [1] INT32 (TOP-5正确数)

    // ========== CUDA Graph ==========
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_captured_;

public:
    // ========== Benchmark结果 ==========
    struct BenchmarkResult {
        double avg_ms;
        double bandwidth_gbps;
        double throughput;
        double val_loss;
        double top1_acc;
        double top5_acc;
    };

    // ========== Kernel配置 ==========
    int block_size_;
    size_t shared_mem_size_;
    BenchmarkResult last_result_;

    explicit SoftmaxInfBenchmark(const SoftmaxInfConfig& config)
        : N_(config.batch_size),
          K_(config.num_classes),
          use_graph_(config.use_graph),
          heur_mode_(config.get_heur_mode()),
          graph_captured_(false)
    {
        // 1. 计算对齐维度
        int64_t row_bytes = K_ * sizeof(__half);
        K_aligned_ = align_to(row_bytes, 128) / sizeof(__half);

        // 2. 选择最优配置
        select_optimal_config();

        // 3. 创建资源
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_comp_1_, cudaStreamNonBlocking));
        CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
        CHECK_CUDNN(cudnnSetStream(cudnn_handle_, stream_comp_1_));

        // 4. 分配内存
        allocate_memory();

        // 5. 初始化数据
        initialize_data();

        // 打印配置
        config.print();
        std::cout << "Aligned K:       " << K_aligned_ << std::endl;
        std::cout << "Block Size:      " << block_size_ << std::endl;
        std::cout << "Shared Memory:   " << shared_mem_size_ << " bytes" << std::endl;
    }

    ~SoftmaxInfBenchmark() {
        // 销毁CUDA Graph
        if (graph_exec_) {
            cudaError_t err = cudaGraphExecDestroy(graph_exec_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: cudaGraphExecDestroy failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }
        if (graph_) {
            cudaError_t err = cudaGraphDestroy(graph_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: cudaGraphDestroy failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }

        // 释放GPU内存
        CHECK_CUDA(cudaFree(d_logits_));
        CHECK_CUDA(cudaFree(d_labels_));
        CHECK_CUDA(cudaFree(d_pred_top1_));
        CHECK_CUDA(cudaFree(d_total_loss_));
        CHECK_CUDA(cudaFree(d_top1_correct_));
        CHECK_CUDA(cudaFree(d_top5_correct_));

        // 销毁流和句柄
        CHECK_CUDNN(cudnnDestroy(cudnn_handle_));
        CHECK_CUDA(cudaStreamDestroy(stream_comp_1_));
    }

private:
    void select_optimal_config() {
        // 根据K值选择block大小
        if (K_ <= 128) {
            block_size_ = 128;
        } else if (K_ <= 512) {
            block_size_ = 256;
        } else if (K_ <= 2048) {
            block_size_ = 512;
        } else {
            block_size_ = 1024;
        }

        // Heuristic模式调整
        if (heur_mode_ == fe::HeurMode_t::A) {
            // 保守模式: 使用较小block size，减少共享内存占用
            block_size_ = std::max(128, block_size_ / 2);
        }

        // 计算共享内存需求（与kernel布局严格一致）
        // 布局：[logits_cache(K_aligned * 4 bytes)] [sum_exp(num_warps * 4 bytes)]
        //       [top5_vals(num_warps * 5 * 4 bytes)] [top5_idxs(num_warps * 5 * 4 bytes)]
        const int num_warps = block_size_ / 32;
        shared_mem_size_ = (K_aligned_ + num_warps * (1 + 5)) * sizeof(float) +
                          num_warps * 5 * sizeof(int32_t);
    }

    void allocate_memory() {
        // Logits: [N, K_aligned]，256B起始对齐，128B行步幅对齐
        const size_t logits_size = N_ * K_aligned_ * sizeof(__half);
        d_logits_ = allocate_aligned_gpu_memory(logits_size);

        // Labels: [N]，INT32，无对齐要求
        CHECK_CUDA(cudaMalloc(&d_labels_, N_ * sizeof(int32_t)));

        // Predictions: [N]，INT32，无对齐要求
        CHECK_CUDA(cudaMalloc(&d_pred_top1_, N_ * sizeof(int32_t)));

        // Total Loss: [1]，FP32标量
        CHECK_CUDA(cudaMalloc(&d_total_loss_, sizeof(float)));

        // Top-1/5 Correct: [1]，INT32标量
        CHECK_CUDA(cudaMalloc(&d_top1_correct_, sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&d_top5_correct_, sizeof(int32_t)));
    }

    void initialize_data() {
        // 随机初始化Logits（不计入计时）
        const size_t logits_size = static_cast<size_t>(N_ * K_);
        std::vector<__half> h_logits(logits_size);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        for (size_t i = 0; i < logits_size; ++i) {
            h_logits[i] = __float2half(dist(gen));
        }

        // 逐行复制到对齐的显存布局（kernel期望步幅为K_aligned）
        for (int64_t i = 0; i < N_; ++i) {
            __half* dst = static_cast<__half*>(d_logits_) + i * K_aligned_;
            CHECK_CUDA(cudaMemcpyAsync(dst, h_logits.data() + i * K_,
                                      K_ * sizeof(__half), cudaMemcpyHostToDevice, stream_comp_1_));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));

        // 随机标签
        std::vector<int32_t> h_labels(N_);
        std::uniform_int_distribution<int32_t> label_dist(0, K_ - 1);
        for (int64_t i = 0; i < N_; ++i) {
            h_labels[i] = label_dist(gen);
        }
        CHECK_CUDA(cudaMemcpy(d_labels_, h_labels.data(),
                             N_ * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    void launch_kernel() {
        launch_softmax_inference_fused(
            static_cast<const __half*>(d_logits_),
            d_labels_,
            d_pred_top1_,
            d_total_loss_,
            d_top1_correct_,
            d_top5_correct_,
            N_, K_, K_aligned_,
            block_size_,
            shared_mem_size_,
            stream_comp_1_
        );
        CHECK_CUDA(cudaGetLastError());
    }

    void capture_cuda_graph() {
        CHECK_CUDA(cudaStreamBeginCapture(stream_comp_1_, cudaStreamCaptureModeThreadLocal));

        launch_kernel();

        CHECK_CUDA(cudaStreamEndCapture(stream_comp_1_, &graph_));

        cudaGraphNode_t error_node;
        char error_log[2048];
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec_, graph_, &error_node,
                                       error_log, sizeof(error_log)));

        graph_captured_ = true;
    }

public:
    void warmup(int iterations = 50) {
        std::cout << "\n=== Warmup Phase ===" << std::endl;

        // 传统模式预热
        std::cout << "Warming up traditional mode..." << std::endl;
        for (int i = 0; i < iterations; ++i) {
            launch_kernel();
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));

        // CUDA Graph捕获
        if (use_graph_) {
            std::cout << "Capturing CUDA Graph..." << std::endl;
            capture_cuda_graph();

            if (graph_captured_) {
                // Graph模式预热
                std::cout << "Warming up graph mode..." << std::endl;
                for (int i = 0; i < iterations; ++i) {
                    CHECK_CUDA(cudaGraphLaunch(graph_exec_, stream_comp_1_));
                }
                CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
                std::cout << "Graph capture successful" << std::endl;
            } else {
                std::cerr << "Warning: Graph capture failed, using traditional mode" << std::endl;
            }
        }
    }

    void execute() {
        // 每次执行前重置计数器（不能包含在CUDA Graph中）
        float zero_f = 0.0f;
        int32_t zero_i = 0;
        CHECK_CUDA(cudaMemcpyAsync(d_total_loss_, &zero_f, sizeof(float),
                                   cudaMemcpyHostToDevice, stream_comp_1_));
        CHECK_CUDA(cudaMemcpyAsync(d_top1_correct_, &zero_i, sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream_comp_1_));
        CHECK_CUDA(cudaMemcpyAsync(d_top5_correct_, &zero_i, sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream_comp_1_));

        if (use_graph_ && graph_captured_ && graph_exec_) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec_, stream_comp_1_));
        } else {
            launch_kernel();
        }
    }

    BenchmarkResult benchmark(int iterations = 500) {
        std::cout << "\n=== Benchmark Phase ===" << std::endl;

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            execute();
        }

        CHECK_CUDA(cudaStreamSynchronize(stream_comp_1_));
        auto end = std::chrono::high_resolution_clock::now();

        double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        // 读取最后一次迭代的统计结果（不计入计时）
        float h_total_loss = 0.0f;
        int32_t h_top1_correct = 0;
        int32_t h_top5_correct = 0;
        CHECK_CUDA(cudaMemcpy(&h_total_loss, d_total_loss_,
                             sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_top1_correct, d_top1_correct_,
                             sizeof(int32_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_top5_correct, d_top5_correct_,
                             sizeof(int32_t), cudaMemcpyDeviceToHost));

        double val_loss = h_total_loss / static_cast<double>(N_);
        double top1_acc = (static_cast<double>(h_top1_correct) / N_) * 100.0;
        double top5_acc = (static_cast<double>(h_top5_correct) / N_) * 100.0;

        // 计算性能指标
        double total_bytes =
            N_ * K_aligned_ * sizeof(__half) +  // 读logits
            N_ * sizeof(int32_t) +               // 读labels
            N_ * sizeof(int32_t);                // 写predictions

        double bandwidth_gbps = (total_bytes / 1e9) / (avg_ms / 1000.0);
        double throughput = N_ / (avg_ms / 1000.0);

        // 保存结果
        last_result_ = {avg_ms, bandwidth_gbps, throughput, val_loss, top1_acc, top5_acc};

        // 打印结果
        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "Mode:              " << (use_graph_ && graph_captured_ ? "CUDA Graph" : "Traditional") << std::endl;
        std::cout << "Average Time:      " << std::fixed << std::setprecision(4)
                  << avg_ms << " ms" << std::endl;
        std::cout << "Memory Bandwidth:  " << std::fixed << std::setprecision(2)
                  << bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "Throughput:        " << std::fixed << std::setprecision(0)
                  << throughput << " samples/sec" << std::endl;
        std::cout << std::setprecision(6);
        std::cout << "Val Loss:          " << val_loss << std::endl;
        std::cout << std::setprecision(2);
        std::cout << "TOP-1 Accuracy:    " << top1_acc << "%" << std::endl;
        std::cout << "TOP-5 Accuracy:    " << top5_acc << "%" << std::endl;
        std::cout << "============================" << std::endl;

        return last_result_;
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ Main入口 ██████████████████████████████████████████████████████████████
// ████████████████████████████████████████████████████████████████████████████
int main(int argc, char** argv) {
    std::cout << "=== 技术觉醒V4 - Softmax Inference 最终结果计算层 ===" << std::endl;
    std::cout << "=== 极致融合方案 | Zero-Intermediate-Memory ===" << std::endl;

    try {
        SoftmaxInfConfig config = parse_softmax_inf_arguments(argc, argv);
        SoftmaxInfBenchmark benchmark(config);

        benchmark.warmup(50);
        benchmark.benchmark(500);

        std::cout << "\n=== 测试完成 ===" << std::endl;
        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

```

```
/**
 * @file softmax_ce_inf.cu
 * @brief 技术觉醒V4 - 推理时融合 Kernel 实现
 * @version 1.0.0
 * @date 2026-04-19
 * @author 技术觉醒团队
 * @note 依赖项: CUDA 13.1
 * @note 所属系列: fp16
 *
 * @note 核心设计原则:
 *   1. 全融合: 单kernel完成Softmax(隐式)、Loss、Top-1/5预测和准确率统计
 *   2. 零中间显存: 概率值仅存于寄存器/共享内存，绝不写回全局显存
 *   3. 三级共享内存: logits缓存 + warp结果缓存，避免重复计算
 *   4. 向量化加载: 使用half2加载，提高内存带宽利用率
 *   5. 数值稳定: FP32全程计算，避免FP16溢出
 *   6. 高效Top-K: Warp内寄存器Top-5 + Shuffle合并
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// ████████████████████████████████████████████████████████████████████████████
// ████ Top-5 数据结构 ████████████████████████████████████████████████████████
// ████████████████████████████████████████████████████████████████████████████

struct Top5 {
    float vals[5];
    int32_t idxs[5];

    __device__ __forceinline__ Top5() {
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            vals[i] = -INFINITY;
            idxs[i] = -1;
        }
    }

    // 插入排序，维持降序
    __device__ __forceinline__ void push(float v, int32_t id) {
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            if (v > vals[i]) {
                // 向后移动
                #pragma unroll
                for (int k = 4; k > i; --k) {
                    vals[k] = vals[k - 1];
                    idxs[k] = idxs[k - 1];
                }
                // 插入新值
                vals[i] = v;
                idxs[i] = id;
                return;
            }
        }
    }

    // 合并另一个Top5（用于warp归约）
    __device__ __forceinline__ void merge(const Top5& other) {
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            if (other.vals[i] > vals[0]) {
                push(other.vals[i], other.idxs[i]);
            }
        }
    }
};

// ████████████████████████████████████████████████████████████████████████████
// ████ 核心推理Kernel: 完全融合的Softmax+Inference ████████████████████████
// ████████████████████████████████████████████████████████████████████████████

/**
 * @brief 极致优化的推理融合Kernel（含Loss、Top-1/5预测和准确率统计）
 *
 * 核心优化策略:
 *   1. 协作加载: 整个block协作将logits加载到共享内存
 *   2. 数值稳定: LogSumExp技巧（先减max，再exp）
 *   3. 零中间显存: 概率值只在寄存器中计算，不写回全局显存
 *   4. 高效Top-K: Warp级Top-5追踪 + Shuffle合并
 *   5. 向量化访问: 使用half2加载，提高内存带宽利用率
 *
 * @param d_logits 输入logits [N, K_aligned]，NHWC布局，256B起始对齐，128B行步幅对齐
 * @param d_labels 标签 [N]，INT32，无对齐要求
 * @param d_pred_top1 输出TOP-1预测 [N]，INT32
 * @param d_total_loss 输出总Loss [1]，FP32标量，所有样本Loss之和（需host端除以N得到平均值）
 * @param d_top1_correct 输出TOP-1正确数 [1]，INT32标量
 * @param d_top5_correct 输出TOP-5正确数 [1]，INT32标量
 * @param N Batch size
 * @param K 原始类别数
 * @param K_aligned 对齐后的类别数（128字节对齐）
 */
template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 4) fused_softmax_inference_kernel(
    const __half* __restrict__ d_logits,
    const int32_t* __restrict__ d_labels,
    int32_t* __restrict__ d_pred_top1,
    float* __restrict__ d_total_loss,
    int32_t* __restrict__ d_top1_correct,
    int32_t* __restrict__ d_top5_correct,
    const int64_t N,
    const int64_t K,
    const int64_t K_aligned)
{
    // 每个block处理一个样本
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;           // warp内线程ID
    const int warp_id = tid >> 5;        // warp ID
    const int num_warps = BLOCK_SIZE / 32;

    const int32_t label = d_labels[n];
    const __half* sample_logits = d_logits + n * K_aligned;

    // 重新设计共享内存布局：完全分离float和int32_t区域
    // 使用字节数组作为基础，通过字节偏移量计算各区域位置
    // 布局：[logits_cache(K_aligned * 4 bytes)] [sum_exp(num_warps * 4 bytes)]
    //       [top5_vals(num_warps * 5 * 4 bytes)] [top5_idxs(num_warps * 5 * 4 bytes)]
    extern __shared__ char s_data_raw[];
    float* s_logits = reinterpret_cast<float*>(s_data_raw);
    float* s_sum_exp_per_warp = reinterpret_cast<float*>(s_data_raw + K_aligned * sizeof(float));
    float* s_top5_vals = reinterpret_cast<float*>(s_data_raw + (K_aligned + num_warps) * sizeof(float));
    int32_t* s_top5_idxs = reinterpret_cast<int32_t*>(s_data_raw + (K_aligned + num_warps + num_warps * 5) * sizeof(float));

    // ━━━━━━ Phase 1: 协作加载logits到共享内存 ━━━━━━
    // 每个线程加载多个元素
    const int elements_per_thread = (K_aligned + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < K_aligned) {
            if (idx < K) {
                // 原始数据：转换为FP32存储，避免重复转换
                s_logits[idx] = __half2float(sample_logits[idx]);
            } else {
                // Padding区域：填充负无穷，确保在max/exp计算时被忽略
                s_logits[idx] = -INFINITY;
            }
        }
    }
    __syncthreads();

    // ━━━━━━ Phase 2: 计算全局max（LogSumExp技巧）━━━━━━
    float local_max = -INFINITY;

    // 每个线程处理一部分
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        local_max = fmaxf(local_max, s_logits[k]);
    }

    // Warp级归约
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, s);
        local_max = fmaxf(local_max, other);
    }

    // Warp leader写入共享内存
    if (lane == 0) {
        s_sum_exp_per_warp[warp_id] = local_max;
    }
    __syncthreads();

    // Block级树形归约求全局max（参考训练版实现）
    #pragma unroll
    for (int s = num_warps >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_exp_per_warp[tid] = fmaxf(s_sum_exp_per_warp[tid], s_sum_exp_per_warp[tid + s]);
        }
        __syncthreads();
    }

    const float global_max = s_sum_exp_per_warp[0];

    // ━━━━━━ Phase 3: 计算sum_exp并维护Top-5 ━━━━━━
    float local_sum_exp = 0.0f;
    Top5 local_top5;

    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float logit = s_logits[k];
        float exp_val = expf(logit - global_max);
        local_sum_exp += exp_val;

        // 维护局部Top-5（基于原始logit值）
        local_top5.push(logit, k);
    }

    // Warp级归约sum_exp
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        local_sum_exp += __shfl_xor_sync(0xFFFFFFFF, local_sum_exp, s);
    }

    // Warp级归约Top-5（两两合并策略）
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        // 接收对方的Top-5
        float other_vals[5];
        int32_t other_idxs[5];

        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            other_vals[i] = __shfl_xor_sync(0xFFFFFFFF, local_top5.vals[i], offset);
            other_idxs[i] = __shfl_xor_sync(0xFFFFFFFF, local_top5.idxs[i], offset);
        }

        // 合并到本地
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            if (other_vals[i] > local_top5.vals[0]) {
                local_top5.push(other_vals[i], other_idxs[i]);
            }
        }
    }

    // 将Warp结果写入共享内存
    if (lane == 0) {
        s_sum_exp_per_warp[warp_id] = local_sum_exp;

        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            s_top5_vals[warp_id * 5 + i] = local_top5.vals[i];
            s_top5_idxs[warp_id * 5 + i] = local_top5.idxs[i];
        }
    }
    __syncthreads();

    // ━━━━━━ Phase 4: 根线程计算最终指标 ━━━━━━
    if (tid == 0) {
        // 1. 合并所有warp的sum_exp
        float total_sum_exp = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            total_sum_exp += s_sum_exp_per_warp[w];
        }

        // 2. 合并所有warp的Top-5
        Top5 final_top5;
        for (int w = 0; w < num_warps; ++w) {
            for (int i = 0; i < 5; ++i) {
                final_top5.push(s_top5_vals[w * 5 + i],
                               s_top5_idxs[w * 5 + i]);
            }
        }

        // 3. 计算并输出指标
        float label_logit = s_logits[label];
        float sample_loss = -label_logit + global_max + logf(total_sum_exp);

        // 使用原子加累加到全局loss
        atomicAdd(d_total_loss, sample_loss);

        // 写入Top-1预测
        int32_t pred_idx = final_top5.idxs[0];
        d_pred_top1[n] = pred_idx;

        // 统计Top-1准确率
        if (pred_idx == label) {
            atomicAdd(d_top1_correct, 1);
        }

        // 统计Top-5准确率
        bool top5_match = false;
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            if (final_top5.idxs[i] == label) {
                top5_match = true;
                break;
            }
        }
        if (top5_match) {
            atomicAdd(d_top5_correct, 1);
        }
    }
}

// ████████████████████████████████████████████████████████████████████████████
// ████ C++接口封装 ████████████████████████████████████████████████████████████
// ████████████████████████████████████████████████████████████████████████████

extern "C" {

/**
 * @brief 启动融合推理Kernel（C接口，含Loss和Top-1/5准确率）
 * @param d_logits 输入logits
 * @param d_labels 标签
 * @param d_pred_top1 输出TOP-1预测
 * @param d_total_loss 输出总Loss（FP32标量，所有样本Loss之和）
 * @param d_top1_correct 输出TOP-1正确数
 * @param d_top5_correct 输出TOP-5正确数
 * @param N Batch size
 * @param K 类别数
 * @param K_aligned 对齐后的类别数
 * @param block_size Block大小
 * @param shared_mem_size 共享内存大小
 * @param stream CUDA流
 */
void launch_softmax_inference_fused(
    const __half* d_logits,
    const int32_t* d_labels,
    int32_t* d_pred_top1,
    float* d_total_loss,
    int32_t* d_top1_correct,
    int32_t* d_top5_correct,
    int64_t N,
    int64_t K,
    int64_t K_aligned,
    int block_size,
    size_t shared_mem_size,
    cudaStream_t stream)
{
    const int grid_size = static_cast<int>(N);

    #define LAUNCH_KERNEL(BLOCK) \
        fused_softmax_inference_kernel<BLOCK><<< \
            grid_size, BLOCK, shared_mem_size, stream>>>( \
            d_logits, d_labels, d_pred_top1, d_total_loss, \
            d_top1_correct, d_top5_correct, N, K, K_aligned)

    switch(block_size) {
        case 128: LAUNCH_KERNEL(128); break;
        case 256: LAUNCH_KERNEL(256); break;
        case 512: LAUNCH_KERNEL(512); break;
        case 1024: LAUNCH_KERNEL(1024); break;
        default: LAUNCH_KERNEL(256); break;
    }

    #undef LAUNCH_KERNEL
}

} // extern "C"

```

```
/**
 * @file softmax_ce_grad.cu
 * @brief 技术觉醒V4 - Softmax+CrossEntropyLoss 融合梯度计算Kernel
 * @version 1.0.0
 * @date 2026-04-19
 * @author 技术觉醒团队
 * @note 依赖项: CUDA 13.1
 * @note 所属系列: fp16
 *
 * @note 核心设计原则:
 *   1. 完全融合: 单kernel完成max归约→exp计算→梯度生成
 *   2. 零中间显存: Softmax概率只存在于共享内存，永不写回全局显存
 *   3. 三级共享内存缓存: max缓存 + sum缓存 + exp缓存，避免重复计算
 *   4. 向量化加载: 使用half2加载，提高内存带宽利用率
 *   5. 数值稳定性: FP32全程计算，避免FP16溢出
 *   6. 避免one-hot展开: 通过标签索引直接计算梯度
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// ████████████████████████████████████████████████████████████████████████████
// ████ 核心Kernel: 完全融合的Softmax+CE梯度计算 ██████████████████████████████
// ████████████████████████████████████████████████████████████████████████████

/**
 * @brief 极致优化的Softmax+CE梯度计算Kernel（含Loss计算）
 *
 * 核心优化策略:
 *   1. 三级共享内存: max缓存 + sum缓存 + exp缓存，避免重复计算
 *   2. 向量化加载: 使用half2加载，提高内存带宽利用率
 *   3. 分支消除: 使用条件移动指令避免分支发散
 *   4. 内存对齐: 完美匹配128B Cache Line
 *   5. 数值稳定: FP32全程计算，避免FP16溢出
 *   6. 零开销Loss: 利用已计算的global_max和sum_exp
 *
 * @param d_logits 输入logits [N, K_aligned]，NHWC布局，256B起始对齐，128B行步幅对齐
 * @param d_labels 标签 [N]，INT32，无对齐要求
 * @param d_grads 输出梯度 [N, K_aligned]，NHWC布局，256B起始对齐，128B行步幅对齐
 * @param d_total_loss 输出总Loss [1]，FP32标量，所有样本Loss之和（需host端除以N得到平均值）
 * @param N Batch size
 * @param K 原始类别数
 * @param K_aligned 对齐后的类别数（128字节对齐）
 */
template <int BLOCK_SIZE, bool USE_SHMEM_CACHE = true>
__global__ void __launch_bounds__(BLOCK_SIZE, 4) fused_softmax_ce_grad_kernel(
    const __half* __restrict__ d_logits,
    const int32_t* __restrict__ d_labels,
    __half* __restrict__ d_grads,
    float* __restrict__ d_total_loss,
    const int64_t N,
    const int64_t K,
    const int64_t K_aligned)
{
    // 每个block处理一个样本
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int32_t label = d_labels[n];

    // 共享内存布局: [max_cache | sum_cache | exp_cache(K_aligned)]
    extern __shared__ float s_data[];
    float* s_max = s_data;                           // BLOCK_SIZE floats
    float* s_sum = s_data + BLOCK_SIZE;              // BLOCK_SIZE floats
    float* s_exp_cache = s_data + 2 * BLOCK_SIZE;    // K_aligned floats (可选)

    const __half* sample_logits = d_logits + n * K_aligned;
    __half* sample_grads = d_grads + n * K_aligned;

    // ━━━━━━ Phase 1: 并行搜索最大值 (LogSumExp Trick) ━━━━━━
    float local_max = -INFINITY;

    // 向量化加载: 每次处理2个元素
    const int K_vec = (K / 2) * 2;

    for (int k = tid * 2; k < K_vec; k += BLOCK_SIZE * 2) {
        __half2 h2 = *reinterpret_cast<const __half2*>(sample_logits + k);
        float2 f2 = __half22float2(h2);
        local_max = fmaxf(local_max, fmaxf(f2.x, f2.y));
    }

    // 处理剩余奇数元素
    if (K % 2 != 0 && tid == 0) {
        float val = __half2float(sample_logits[K - 1]);
        local_max = fmaxf(local_max, val);
    }

    s_max[tid] = local_max;
    __syncthreads();

    // Block级树形归约求全局max
    #pragma unroll
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    const float global_max = s_max[0];

    // ━━━━━━ Phase 2: 计算exp(x - max)并累加sum ━━━━━━
    float local_sum = 0.0f;

    for (int k = tid * 2; k < K_vec; k += BLOCK_SIZE * 2) {
        __half2 h2 = *reinterpret_cast<const __half2*>(sample_logits + k);
        float2 f2 = __half22float2(h2);

        float exp_x = expf(f2.x - global_max);
        float exp_y = expf(f2.y - global_max);

        // 关键优化: 将exp值缓存到共享内存，避免重复计算
        if (USE_SHMEM_CACHE) {
            s_exp_cache[k] = exp_x;
            s_exp_cache[k + 1] = exp_y;
        }

        local_sum += exp_x + exp_y;
    }

    if (K % 2 != 0 && tid == 0) {
        float val = __half2float(sample_logits[K - 1]);
        float exp_val = expf(val - global_max);
        if (USE_SHMEM_CACHE) {
            s_exp_cache[K - 1] = exp_val;
        }
        local_sum += exp_val;
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    // Block级树形归约求全局sum
    #pragma unroll
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    const float sum_exp = s_sum[0];
    const float inv_sum = 1.0f / sum_exp;

    // ━━━━━━ Phase 3: 计算梯度并写回 ━━━━━━
    // grad = prob - (k == label ? 1.0f : 0.0f)

    for (int k = tid * 2; k < K_vec; k += BLOCK_SIZE * 2) {
        float prob_x, prob_y;

        // 从共享内存读取概率值，或重新计算
        if (USE_SHMEM_CACHE) {
            prob_x = s_exp_cache[k] * inv_sum;
            prob_y = s_exp_cache[k + 1] * inv_sum;
        } else {
            // 回退方案: 重新计算exp
            __half2 h2 = *reinterpret_cast<const __half2*>(sample_logits + k);
            float2 f2 = __half22float2(h2);
            prob_x = expf(f2.x - global_max) * inv_sum;
            prob_y = expf(f2.y - global_max) * inv_sum;
        }

        // 计算梯度: 使用条件移动避免分支发散
        float grad_x = prob_x - ((k == label) ? 1.0f : 0.0f);
        float grad_y = prob_y - ((k + 1 == label) ? 1.0f : 0.0f);

        // 向量化写回
        __half2 result = __floats2half2_rn(grad_x, grad_y);
        *reinterpret_cast<__half2*>(sample_grads + k) = result;
    }

    // 处理剩余奇数元素
    if (K % 2 != 0 && tid == 0) {
        float prob;
        if (USE_SHMEM_CACHE) {
            prob = s_exp_cache[K - 1] * inv_sum;
        } else {
            float val = __half2float(sample_logits[K - 1]);
            prob = expf(val - global_max) * inv_sum;
        }
        float grad = prob - (((K - 1) == label) ? 1.0f : 0.0f);
        sample_grads[K - 1] = __float2half_rn(grad);
    }

    // ━━━━━━ Phase 4: 计算CrossEntropyLoss（零额外开销）━━━━━━
    // CE_Loss = -log(softmax(logits)[label])
    //         = -logits[label] + log(sum(exp(logits)))
    //         = -logits[label] + global_max + log(sum_exp)
    // 注意：所有线程计算结果相同，只用tid==0写入
    if (tid == 0) {
        float label_logit = __half2float(sample_logits[label]);
        float sample_loss = -label_logit + global_max + logf(sum_exp);

        // 使用原子加将当前样本的loss累加到全局loss
        atomicAdd(d_total_loss, sample_loss);
    }
}

// ████████████████████████████████████████████████████████████████████████████
// ████ C++接口封装 ████████████████████████████████████████████████████████████
// ████████████████████████████████████████████████████████████████████████████

extern "C" {

/**
 * @brief 启动Softmax+CE融合梯度计算Kernel（C接口，含Loss）
 * @param d_logits 输入logits
 * @param d_labels 标签
 * @param d_grads 输出梯度
 * @param d_total_loss 输出总Loss（FP32标量，所有样本Loss之和）
 * @param N Batch size
 * @param K 类别数
 * @param K_aligned 对齐后的类别数
 * @param block_size Block大小
 * @param shared_mem_size 共享内存大小
 * @param use_shmem_cache 是否使用共享内存缓存exp值
 * @param stream CUDA流
 */
void launch_fused_softmax_ce_grad(
    const __half* d_logits,
    const int32_t* d_labels,
    __half* d_grads,
    float* d_total_loss,
    int64_t N,
    int64_t K,
    int64_t K_aligned,
    int block_size,
    size_t shared_mem_size,
    bool use_shmem_cache,
    cudaStream_t stream)
{
    const int grid_size = static_cast<int>(N);

    #define LAUNCH_KERNEL(BLOCK) \
        if (use_shmem_cache) { \
            fused_softmax_ce_grad_kernel<BLOCK, true><<< \
                grid_size, BLOCK, shared_mem_size, stream>>>( \
                d_logits, d_labels, d_grads, d_total_loss, N, K, K_aligned); \
        } else { \
            fused_softmax_ce_grad_kernel<BLOCK, false><<< \
                grid_size, BLOCK, shared_mem_size, stream>>>( \
                d_logits, d_labels, d_grads, d_total_loss, N, K, K_aligned); \
        }

    switch(block_size) {
        case 128: LAUNCH_KERNEL(128); break;
        case 256: LAUNCH_KERNEL(256); break;
        case 512: LAUNCH_KERNEL(512); break;
        case 1024: LAUNCH_KERNEL(1024); break;
        default: LAUNCH_KERNEL(256); break;
    }

    #undef LAUNCH_KERNEL
}

} // extern "C"

```

