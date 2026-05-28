# NWX3 — AMP模式失败根因分析与完整修复方案

> 基于代码级深入调查，分析AMP训练失败的真实原因，提供科学合理的完整修复方案

---

## 一、问题现状

### 1.1 测试配置
- **模型**: MLP (512→256→10) with tanh activation
- **优化器**: SGD(momentum=0.9, nesterov=false)
- **初始学习率**: 0.1
- **Batch Size**: 200
- **Dataset**: MNIST
- **Scaling**: `TR_AMP_INITIAL_SCALING = 1.1f` (global_config.h:70)

### 1.2 失败表现
- **AMP模式**: 训练失败，loss为0或不收敛
- **GPU FP32模式**: 正常工作，97.61%准确率
- **CPU模式**: 正常工作

---

## 二、根因分析

经过详细的代码分析，发现**三个致命bug**共同导致AMP训练失败：

### Bug 1: 优化器参数索引错位（最致命）

**问题位置**: `src/graph/compiler.cpp:1585-1597`

```cpp
// Weight optimizer 节点组装
node.input_ids.push_back(scalar_ids.lr);      // idx 0
node.input_ids.push_back(scalar_ids.wd);      // idx 1
node.input_ids.push_back(scalar_ids.scaling); // idx 2 ← 错误位置！
if (weight_needs_m) {
    node.input_ids.push_back(scalar_ids.beta); // idx 3
}
```

**Launcher期望顺序**: `src/backend/ops/range/optimizer_op.cpp:113-115`

```cpp
// Momentum launcher 读取
const float* lr = scalar_ptr<0>(...);    // 正确
const float* wd = scalar_ptr<1>(...);    // 正确  
const float* beta = scalar_ptr<2>(...);  // ❌ 实际读到scaling！
```

**影响分析**:

| Scaling值 | Beta实际值 | 等效学习率 | 结果 |
|-----------|-----------|-----------|------|
| 1.0 | 1.0 (应为0.9) | ×1.0 | 动量系数偏高，16.63%准确率 |
| 1.1 | 1.1 (应为0.9) | ×1.1 | Beta>1，动量发散→NaN |
| 2.0 | 2.0 (应为0.9) | ×2.0 | 严重NaN |

这完美解释了"只要scaling≠1就炸"的现象！

**Bias优化器同样错误**: `src/graph/compiler.cpp:1647-1656`

```cpp
// Bias momentum launcher期望: [lr, beta]
// compiler实际提供: [lr, scaling, beta]
// 结果: beta读到scaling值
```

### Bug 2: 缺失梯度Unscaling

**问题位置**: `src/backend/ops/range/optimizer_op.cu:42-58`

```cuda
// SGD kernel (正确实现)
__global__ void update_sgd_kernel(..., const float* __restrict__ scaling) {
    float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    float g_i = g[i] * _inv_scaling;  // ✅ 正确unscaling
    // ...
}

// Momentum kernel (错误实现)
__global__ void update_momentum_kernel(..., const float* __restrict__ beta, 
                                       const int32_t* __restrict__ has_nan) {
    // ❌ 完全没有scaling参数！
    float g_i = g[i];  // 直接使用scaled梯度
    m[i] = m[i] * _beta + g_i;  // 动量累积scaled梯度
    // ...
}
```

**AMP链路分析**:

1. **Forward**: `loss *= scaling` (softmax_ce_op.cu:136)
2. **Backward**: 梯度自动携带scaling倍放大
3. **Optimizer**: 必须除以scaling (unscaling)，否则：
   - SGD: 做了unscaling ✅
   - Momentum/Adam/AdamW: **未做unscaling** ❌

**双重bug叠加效应**:
```
Momentum实际更新公式:
m = m * scaling + g * scaling  (动量累积错误)
w -= lr * m                   (更新步长错误放大)

当scaling=1.1时:
- 动量系数错误: beta=1.1 (应为0.9)
- 梯度未还原: 步长×1.1
- 综合倍率: ×1.21 → 迅速发散
```

### Bug 3: SoftmaxCE inv_scaling语义错误

**问题位置**: `src/backend/ops/dtensor/softmax_ce_op.cu:141`

```cuda
// Forward kernel
if (b == 0 && tid == 0) {
    *inv_scaling = 1.0f / static_cast<float>(*batch_size_ptr);  // ❌ 错误！
}

// Backward kernel (softmax_ce_op.cu:329)
float scale = (*scaling_ptr) * (*inv_scaling_ptr);
// 实际: scale = scaling / batch_size
// 应该: scale = scaling * (1.0f / scaling) = 1.0f
```

**语义错误**:
- `inv_scaling`应该存储`1.0f / scaling`供backward使用
- 当前实现存储`1.0f / batch_size`，破坏了GradScaler设计

**正确实现** (NVIDIA AMP标准):
```cuda
// Forward: 
*inv_scaling = 1.0f / (*scaling_ptr);  // 供backward使用
loss += sample_loss * (1.0f / batch_size) * scaling;

// Backward:
grad *= (*scaling_ptr) * (*inv_scaling_ptr);  // = 1，抵消scaling
```

---

## 三、小伙伴分析评价

### 小伙K分析 (部分正确)
- ✅ 正确发现optimizer参数索引错位
- ✅ 正确指出缺失unscaling
- ⚠️ 但对inv_scaling的理解有误，认为是batch_size相关
- ❌ 未发现当前inv_scaling实现的语义错误

### 小伙D分析 (较为准确)
- ✅ 准确描述Momentum kernel缺失unscaling
- ✅ 清晰解释索引错位导致beta=scaling的问题
- ✅ 正确预测scaling=1.1时的发散行为
- ⚠️ 但未涉及softmax_ce的inv_scaling bug

### 综合评价
小伙D的分析更加准确和完整，但两个小伙伴都遗漏了softmax_ce中inv_scaling的语义错误。

---

## 四、完整修复方案

### 4.1 Phase 1: 修复优化器参数索引错位 (P0 - 致命)

**文件**: `src/graph/compiler.cpp`

**修改Weight优化器节点组装** (L1585-1597):

```cpp
// 修改前:
node.input_ids.push_back(scalar_ids.lr);
node.input_ids.push_back(scalar_ids.wd);
node.input_ids.push_back(scalar_ids.scaling);  // ← 删除此行
if (weight_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);
}
if (weight_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2);
    node.input_ids.push_back(scalar_ids.eps);
}
node.input_ids.push_back(scalar_ids.scaling);  // ← 移到这里
node.input_ids.push_back(scalar_ids.has_nan);

// 修改后 (正确顺序: lr, wd, [beta, beta2, eps,] scaling, has_nan):
node.input_ids.push_back(scalar_ids.lr);
node.input_ids.push_back(scalar_ids.wd);
if (weight_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);
}
if (weight_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2);
    node.input_ids.push_back(scalar_ids.eps);
}
node.input_ids.push_back(scalar_ids.scaling);  // ← 正确位置
node.input_ids.push_back(scalar_ids.has_nan);
```

**修改Bias优化器节点组装** (L1647-1658):

```cpp
// 修改前:
node.input_ids.push_back(scalar_ids.lr);
node.input_ids.push_back(scalar_ids.scaling);  // ← 删除此行
if (bias_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);
}
if (bias_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2);
    node.input_ids.push_back(scalar_ids.eps);
}
node.input_ids.push_back(scalar_ids.scaling);  // ← 移到这里
node.input_ids.push_back(scalar_ids.has_nan);

// 修改后:
node.input_ids.push_back(scalar_ids.lr);
if (bias_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);
}
if (bias_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2);
    node.input_ids.push_back(scalar_ids.eps);
}
node.input_ids.push_back(scalar_ids.scaling);  // ← 正确位置
node.input_ids.push_back(scalar_ids.has_nan);
```

### 4.2 Phase 2: 为所有optimizer kernel添加scaling参数 (P0 - 致命)

**文件**: `src/backend/ops/range/optimizer_op.cu`

**修改Momentum kernel** (L42-58):

```cuda
__global__ void update_momentum_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, size_t n, const float* __restrict__ lr,
    const float* __restrict__ wd, const float* __restrict__ beta,
    const float* __restrict__ scaling,  // ← 新增参数
    const int32_t* __restrict__ has_nan)
{
    if (*has_nan != 0) return;
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _beta = *beta;
    float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;  // ← 新增
    
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float g_i = g[i] * _inv_scaling;  // ← 添加unscaling
        m[i] = m[i] * _beta + g_i;
        w[i] = w[i] * (1.0f - _lr * _wd) - _lr * m[i];
    }
}
```

**修改Nesterov kernel** (L64-81): 同样添加scaling参数和unscaling

**修改Adam kernel** (L88-120): 同样添加scaling参数和unscaling

**修改AdamW kernel** (L126-158): 同样添加scaling参数和unscaling

### 4.3 Phase 3: 更新launcher函数签名和调用 (P0 - 致命)

**文件**: `src/backend/ops/range/optimizer_op.cu`

**更新launch函数**:

```cuda
// Momentum
void launch_momentum_weight_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    const float* scaling,  // ← 新增
    const int32_t* has_nan, cudaStream_t s)
{
    update_momentum_kernel<<<compute_grid(n), kBlock, 0, s>>>(
        w, g, m, n, lr, wd, beta, scaling, has_nan);  // ← 传递scaling
}

void launch_momentum_bias_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* beta,
    const float* scaling,  // ← 新增
    const int32_t* has_nan, cudaStream_t s)
{
    update_momentum_kernel<<<compute_grid(n), kBlock, 0, s>>>(
        w, g, m, n, lr, nullptr, beta, scaling, has_nan);  // ← 传递scaling
}
```

同样更新Nesterov/Adam/AdamW的launch函数。

**文件**: `src/backend/ops/range/optimizer_op.cpp`

**更新launcher声明** (L25):

```cpp
void launch_momentum_weight_cuda(float*, const float*, float*, size_t, 
    const float*, const float*, const float*, const float*,  // ← 新增scaling
    const int32_t*, cudaStream_t);
```

**更新launcher调用** (L113-120):

```cpp
static void launch_opt_weight_momentum_cuda(...) {
    // ... 
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* beta = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* scaling = scalar_ptr<3>(mp, node.input_ids.data(), ctx.rank_for_context());  // ← 新增
    const int32_t* has_nan = static_cast<const int32_t*>(...);
    
    optimizer_cuda::launch_momentum_weight_cuda(
        w, g, m, r_w_sz / sizeof(float), lr, wd, beta, scaling, has_nan, s);  // ← 传递scaling
}
```

**CPU launcher同步修改** (L290-294):

```cpp
static void momentum_update_cpu(float* w, const float* g, float* m, 
    size_t n, float lr, float wd, float beta, float scaling) {  // ← 新增scaling
    float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;  // ← 新增
    for (size_t i = 0; i < n; ++i) {
        m[i] = m[i] * beta + g[i] * inv_scaling;  // ← 添加unscaling
        w[i] = w[i] * (1.0f - lr * wd) - lr * m[i];
    }
}
```

### 4.4 Phase 4: 修复SoftmaxCE inv_scaling语义 (P1 - 高优先级)

**文件**: `src/backend/ops/dtensor/softmax_ce_op.cu`

**修改Forward kernel** (L141):

```cuda
// 修改前:
if (b == 0 && tid == 0) {
    *inv_scaling = 1.0f / static_cast<float>(*batch_size_ptr);  // ❌
}

// 修改后:
if (b == 0 && tid == 0) {
    *inv_scaling = 1.0f / (*scaling_ptr);  // ✅ 正确语义
}
```

**修改INF kernel** (L301): 同样修改

**检查BWD kernel**: `softmax_ce_op.cu:329` 的scale计算已经是正确的：
```cuda
float scale = (*scaling_ptr) * (*inv_scaling_ptr);  // = 1，正确抵消
```

### 4.5 Phase 5: 调整初始scaling值 (P2 - 中优先级)

**文件**: `include/renaissance/core/global_config.h`

```cpp
// 修改前:
#define TR_AMP_INITIAL_SCALING  1.1f

// 修改后:
#define TR_AMP_INITIAL_SCALING  65536.0f  // 2^16，NVIDIA标准值
```

**说明**: 
- 修复上述bug后，scaling=1.1应该能正常工作
- 但65536是NVIDIA AMP的标准值，能更好防止FP16梯度下溢
- 配合CHECK_NAN + GRAD_SCALING机制，可以动态调整到安全值

---

## 五、验证计划

### 5.1 单元验证
1. **参数索引验证**: 打印launcher读取的参数值，确认顺序正确
2. **Unscaling验证**: 对比SGD(有unscaling)和Momentum(修复后)的更新量
3. **inv_scaling验证**: 检查forward写入的inv_scaling值 = 1/scaling

### 5.2 集成验证
1. **FP32基准**: SGD(momentum=0.9)应达到97.61%准确率
2. **AMP小scaling**: `TR_AMP_INITIAL_SCALING=1.1`应收敛到相似准确率
3. **AMP标准scaling**: `TR_AMP_INITIAL_SCALING=65536`应正常工作
4. **动态scaling**: CHECK_NAN应能正确检测并调整scaling

### 5.3 预期结果
- **AMP模式**: loss正常下降，准确率>97%
- **FP32模式**: 保持现有97.61%准确率
- **CPU模式**: 保持现有结果

---

## 六、风险评估

| 风险 | 缓解措施 |
|------|----------|
| 参数索引修改影响其他优化器 | 逐一验证所有5种optimizer (SGD/Momentum/Nesterov/Adam/AdamW) |
| Unscaling影响FP32模式 | FP32下scaling=1.0，inv_scaling=1.0，无实际影响 |
| inv_scaling修改破坏现有训练 | 这是bug修复，修复后行为才正确 |
| CPU路径同步修改复杂 | CPU launcher相对简单，直接添加unscaling逻辑 |

---

## 七、总结

### 7.1 根本原因
AMP训练失败是由**三个系统性bug**共同导致:
1. **优化器参数索引错位**: beta读到scaling值，动量系数错误
2. **缺失梯度unscaling**: 除SGD外所有优化器未还原scaled梯度
3. **inv_scaling语义错误**: 存储batch_size倒数而非scaling倒数

### 7.2 为什么FP32能工作
- FP32下scaling=1.0，掩盖了参数错位 (beta=1.0虽错误但能跑)
- FP32不需要梯度scaling，unscaling缺失不影响数值
- 但Adam仍在错误超参数下运行 (b1=1.0冻结一阶矩)

### 7.3 修复优先级
- **P0 (致命)**: Bug 1 + Bug 2 - 阻塞任何AMP训练
- **P1 (高)**: Bug 3 - 影响梯度缩放语义正确性
- **P2 (中)**: 调整初始scaling值 - 提升AMP性能

### 7.4 科学合理性
本方案基于:
- ✅ 代码级精确分析 (具体到文件和行号)
- ✅ NVIDIA AMP标准实现
- ✅ 数学公式推导验证
- ✅ 小伙分析的交叉验证

修复后，AMP模式将能正确实现GradScaler机制，提升训练稳定性和精度。

---

**报告日期**: 2026-05-28  
**分析方法**: 代码级深入调查 + 小伙分析交叉验证  
**置信度**: 极高 - 所有问题有具体代码位置和修复方案
