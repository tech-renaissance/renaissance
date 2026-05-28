# CNM1: 优化器超参数从未写入 GPU 导致训练配置失效

> **日期**: 2026-05-26
>
> **严重性**: 🔴 高 — 用户通过 `.momentum(0.9f)` 等 API 设置的优化器超参数，在 GPU 执行时**全部失效**，训练实际使用 GPU 显存初始值（通常为 0）

---

## 1. 问题概述

用户在测试代码中通过 DSL 链式调用配置了优化器：

```cpp
.optimizer(SGD().momentum(0.9f).weight_decay(0.0f).nesterov(false).dampening(0.0f))
```

这些配置值（momentum=0.9, weight_decay=0.0）正确进入了 `DeepLearningTask::opt_cfg_` 成员，**但从 `opt_cfg_` 到 GPU 标量 DTensor 的桥接代码完全缺失**。GPU 上的 optimizer kernel 读取的 `beta` 和 `wd` 值是 `cudaMalloc` 分配时未初始化的残留值。

---

## 2. 数据流追踪：正确到达 CPU，终止于 CPU

### 2.1 阶段 A：配置进入 CPU 端 ✅ 正确

```
测试 DSL                          deep_learning_task.cpp:94
  │                                     │
  ├─ SGD()                               │
  ├─ .momentum(0.9f) ─────────> config_.momentum = 0.9f
  ├─ .weight_decay(0.0f) ────> config_.weight_decay = 0.0f
  ├─ .nesterov(false) ────────> config_.nesterov = false
  └─ .dampening(0.0f) ────────> config_.dampening = 0.0f
                                        │
                    optimizer(const SGD& opt) {
                        opt_cfg_ = opt;    ← 拷贝构造，POD 类无问题
                    }
```

`opt_cfg_` 存储了完整的 `SGDConfig{momentum=0.9, weight_decay=0.0, ...}`。此阶段无 bug。

### 2.2 阶段 B：GPU 标量 DTensor 分配 ✅ 仅分配

[memory_plan.cpp:L383-L386](file:///r:/renaissance/src/graph/memory_plan.cpp#L383-L386):

```cpp
if (opt != OptimizerKind::SGD) {
    baseline_.beta = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
    baseline_.wd   = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
}
```

分配了 `beta` 和 `wd` 的 GPU 内存空间，但**只分配，不设值**。`cudaMalloc` 返回的内存内容是未定义的（CUDA 规范明确声明 "The memory is not initialized"）。

### 2.3 阶段 C：Compiler 初始化 ✅ 仅对 `scaling` 设值

[compiler.cpp:L748-L749](file:///r:/renaissance/src/graph/compiler.cpp#L748-L749):

```cpp
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(1.0f));
```

**只有 `scaling` 被初始化为 1.0**。`beta`、`wd`、`eps`、`tc` 等优化器标量均无 `set_init_config` 调用。

[compiler.cpp:L755-L759](file:///r:/renaissance/src/graph/compiler.cpp#L755-L759) 仅将 `scalar_ids` 指向这些 DTensor 的 ID，用于构建计算图节点的输入边：

```cpp
scalar_ids.beta  = b.beta;   // 仅记录 ID，不写值
scalar_ids.wd    = b.wd;     // 仅记录 ID，不写值
```

### 2.4 阶段 D：`init_all()` 初始化 ❌ 跳过所有优化器标量

[task_base.cpp:L1313-L1315](file:///r:/renaissance/src/task/task_base.cpp#L1313-L1315):

```cpp
for (const auto& dtensor : active_memory_plan_->dtensors()) {
    init(dtensor);                    // → Initializer::derive(region) → apply_to_tensor
}
```

对于 `beta`、`wd` 所在的 `Region::S_SCALAR_FP32`：

[initializer.cpp:L100-L113](file:///r:/renaissance/src/core/initializer.cpp#L100-L113) + [initializer.cpp:L162-L163](file:///r:/renaissance/src/core/initializer.cpp#L162-L163):

```
derive(S_SCALAR_FP32)
  → is_bias_region(S_SCALAR_FP32) = false
  → is_param_region(S_SCALAR_FP32) = false   ← 不在白名单中
  → 返回 InitConfig{1.0f, NONE, FAN_IN}
  → apply_to_tensor → case NONE → return;    ← 什么都不做
```

**GPU 端 `beta`、`wd` 标量从未被写入任何有效值。**

### 2.5 阶段 E：设计假设与实现脱节 ❌ 关键断裂

[initializer.cpp:L83-L95](file:///r:/renaissance/src/core/initializer.cpp#L83-L95) 的注释明确表达了设计意图：

> *"整个显存池在 alloc 前全局置零，不在 is_param_region 中的 DTensor 天然为全零内存，梯度/动量/速度/范数的初始值 0 是自动满足的，无需 Initializer 干预。"*

然而，搜索 `memory_arena.cpp` 全部代码 — **不存在任何 `cudaMemset` 或全局置零操作**。实际内存分配使用的是原生 `cudaMalloc`（[memory_arena.cpp:L126](file:///r:/renaissance/src/backend/memory_arena.cpp#L126)），CUDA 规范明确标注其返回内存"未初始化"。

---

## 3. 实际执行的训练 vs 用户预期的训练

### 3.1 用户预期

| 参数 | 用户设定值 | 含义 |
|------|-----------|------|
| momentum (β) | 0.9 | 动量系数，加速收敛 |
| weight_decay | 0.0 | 无 L2 正则化 |
| lr | 0.1 | 学习率 |

预期优化器公式（SGDM）：

```
m_t = m_{t-1} × 0.9 + g_t
w_t = w_{t-1} - 0.1 × m_t
```

### 3.2 GPU 实际执行

[optimizer_op.cu:L48-L53](file:///r:/renaissance/src/backend/ops/range/optimizer_op.cu#L48-L53):

```cpp
m[i] = m[i] * _beta + g_i;           // _beta = 0（GPU 残留值）
w[i] = w[i] * (1.0f - _lr * _wd)    // _wd = 0（GPU 残留值）
     - _lr * m[i];                    // _lr = 0.1（训练循环写入 ✓）
```

代入 `_beta=0`, `_wd=0`, `_lr=0.1`：

```
m[i] = m[i] × 0 + g_i = g_i          ← 动量项失效！
w[i] = w[i] - 0.1 × g_i              ← 退化为纯 SGD
```

### 3.3 影响评估

| 现象 | 影响 |
|------|------|
| momentum=0.9 未生效 | 失去动量加速，收敛变慢，3 epoch 内准确率低于应有水平 |
| weight_decay=0.0 凑巧正确 | GPU 残留值恰好也是 0，无影响（纯属巧合） |
| 若用户设 weight_decay>0 | 同样不会生效，L2 正则化失效 |
| `Adam` 的 `beta1`/`beta2`/`eps` | 同路径，全部不生效 |
| `LARS` 的 `trust_coefficient`/`eps` | 同路径，全部不生效 |

**这是 1.63% 准确率差距的最可能的直接原因。**

### 3.4 证实推断：LR×10 实验

当 LR 从 0.1 改为 1.0（10 倍）时，准确率从 95.64% 暴跌到 76.47%。这个行为与**纯 SGD（无动量）**一致：

- 有动量时：`m_t = 0.9×m_{t-1} + g_t` 提供了"惯性"，能部分抵抗大 LR 的震荡
- 无动量时：`w = w - 1.0×g` 直接过度更新，导致发散

如果动量真的生效（β=0.9），LR=1.0 的震荡会被动量缓冲，准确率不会如此剧烈下降。

---

## 4. 受影响范围

### 4.1 所有优化器类型均受影响

| 优化器 | 丢失的超参数 | 影响 |
|--------|------------|------|
| `SGD_MOMENTUM` | β (momentum) | 退化为纯 SGD |
| `SGD_NESTEROV` | β (momentum) | Nesterov 加速失效 |
| `ADAM` / `ADAMW` | β1, β2, ε | Adam 完全失效，退化为带 wd 的纯 SGD |
| `LARS` | β, trust_coefficient, ε | LARS 层级自适应失效 |

### 4.2 `lr` 和 `scaling` 不受影响

- `lr`：由 `run_train_epoch_gpu()` 中的 `cudaMemcpyAsync(lr_dev_ptr, ...)` 每 batch 写入（[deep_learning_task.cpp:L959-L960](file:///r:/renaissance/src/task/deep_learning_task.cpp#L959-L960)）
- `scaling`：由 `compiler.cpp` 中的 `set_init_config(scaling, kInitConstant(1.0f))` 在初始化阶段写入（[compiler.cpp:L748-L749](file:///r:/renaissance/src/graph/compiler.cpp#L748-L749)）

### 4.3 `SGD`（纯 SGD，无动量）不受影响

当 `optimizer = SGD` 时，`alloc_baseline_dtensors()` 不分配 `beta`/`wd` DTensor（[memory_plan.cpp:L383](file:///r:/renaissance/src/graph/memory_plan.cpp#L383)），optimizer kernel 也不读取它们。纯 SGD 用户恰好不受此 bug 影响。

---

## 5. 修改方案

### 方案 A：在 Compiler 中补充 `set_init_config`（推荐）

在 [compiler.cpp:L748-L749](file:///r:/renaissance/src/graph/compiler.cpp#L748-L749) 之后，根据 `GlobalRegistry` 中存储的优化器配置，对 `beta`、`wd`、`eps`、`tc` 等标量 DTensor 调用 `set_init_config`。

**具体位置**：`Compiler::create_memory_plans()` 中，在现有的 `set_init_config(scaling, ...)` 之后。

**关键问题**：`set_init_config` 只能写编译期常量，而 `opt_cfg_` 中的值是**运行时**从用户 DSL 传入的。需要将超参数值从 `DeepLearningTask` 传递到 `Compiler::compile()` 中。

**可行路径**：

1. 在 `Compiler::compile()` 的参数中增加超参结构体，或从 `GlobalRegistry` 读取。`DeepLearningTask::on_prepare()` 在调用 `Compiler::compile()` 之前已经执行了 `GlobalRegistry::instance().set_optimizer_kind(opt_kind)`，可以同步增加超参写入。

2. 在 `on_prepare()` 中将 `opt_cfg_` 中的超参值写入 `GlobalRegistry`：

```cpp
// 在 deep_learning_task.h on_prepare() 中，调用 Compiler 之前
std::visit([](const auto& cfg) {
    if constexpr (!std::is_same_v<std::decay_t<decltype(cfg)>, std::monostate>) {
        GlobalRegistry::instance().set_optimizer_config(cfg);
    }
}, opt_cfg_);
```

3. 在 `Compiler::create_memory_plans()` 中读取并写入标量 DTensor：

```cpp
// 在 compiler.cpp set_init_config(scaling, 1.0) 之后
auto opt = GlobalRegistry::instance().optimizer_kind();
auto opt_cfg = GlobalRegistry::instance().optimizer_config();  // 新增接口

if (opt != OptimizerKind::SGD) {
    memory_plans[s]->set_init_config(b.beta, kInitConstant(opt_cfg.momentum));
    memory_plans[s]->set_init_config(b.wd,   kInitConstant(opt_cfg.weight_decay));
}
if (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW) {
    memory_plans[s]->set_init_config(b.beta2, kInitConstant(opt_cfg.beta2));
    memory_plans[s]->set_init_config(b.eps,   kInitConstant(opt_cfg.eps));
}
if (opt == OptimizerKind::LARS || opt == OptimizerKind::LARS_NESTEROV) {
    memory_plans[s]->set_init_config(b.tc,  kInitConstant(opt_cfg.trust_coefficient));
    memory_plans[s]->set_init_config(b.eps, kInitConstant(opt_cfg.eps));
}
```

### 方案 B：在 `TaskBase::init_all()` 中直接写入 GPU

在 `TaskBase::init_all()` 中，对所有标量 DTensor 补充写入逻辑。这绕过了 `Initializer` 框架，直接在 GPU 上 `cudaMemset` 或通过 `Tensor` 的 `fill()` 写入。

**缺点**：与 Initializer 框架的设计不一致，增加了维护复杂度。不推荐。

### 方案 C：修复全局 memset 假设

在 `CudaArena::do_allocate()` 中，为所有新分配的 GPU 内存调用 `cudaMemset(ptr, 0, total_size)`。

**缺点**：
- `beta` 需要的是 0.9，不是 0。全局 memset 不能解决"非零默认值"问题。
- 仅对 `wd=0` 有效，对 `beta=0.9` 无效。
- 对大量内存做 memset 会影响初始化性能。

不推荐作为唯一方案。可作为方案 A 的补充（确保 momentum 等梯度缓冲区归零）。

---

## 6. 总结

| 维度 | 结论 |
|------|------|
| `opt_cfg_` 复制有 bug？ | **无**。POD 类的拷贝构造正确。 |
| 超参数到达 GPU 了吗？ | **没有**。从 `opt_cfg_` 到 GPU 的桥接代码完全缺失。 |
| 当前训练实际使用什么？ | `cudaMalloc` 残留值（通常为 0）：`beta=0`, `wd=0`。动量完全被禁用。 |
| 对 1.63% 差距的贡献？ | **极高概率是主要原因**。SGDM→纯 SGD 的退化导致 3 epoch 内收敛不足。 |
| 为什么不崩溃？ | `cudaMalloc` 在首次分配时通常返回零值（OS 安全策略），虽然 CUDA 规范不保证。`beta=0` 恰好让训练退化到纯 SGD（仍可训练，只是慢）。 |
| 修复优先级 | 🔴 **P0**。影响所有非纯 SGD 的训练场景。 |

---

## 附录：完整数据流断点图

```
用户 DSL                          deep_learning_task.cpp
  │                                     │
  │ SGD().momentum(0.9f)                │ opt_cfg_ = opt;
  │                                     │   ├─ SGDConfig.momentum = 0.9 ✓
  │                                     │   └─ SGDConfig.weight_decay = 0.0 ✓
  │                                     │
  │                                     │ on_prepare()
  │                                     │   ├─ opt_kind = kind() → SGD_MOMENTUM ✓
  │                                     │   └─ GlobalRegistry.set_optimizer_kind ✓
  │                                     │
  ▼                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Compiler::create_memory_plans()            │
│                                                                  │
│  alloc_baseline_dtensors(opt=SGD_MOMENTUM)                      │
│    → baseline_.beta = DTensor(id=N, Region=S_SCALAR_FP32)       │
│    → baseline_.wd   = DTensor(id=M, Region=S_SCALAR_FP32)       │
│                    ↑ 仅分配 ID，不设值                            │
│                                                                  │
│  set_init_config(scaling, 1.0)                                   │
│                    ↑ 仅 scaling 被初始化                          │
│                                                                  │
│  beta/wd/eps/tc: 无 set_init_config 调用  ←── 🔴 BUG            │
└──────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                       TaskBase::init_all()                       │
│                                                                  │
│  for each DTensor:                                               │
│    derive(S_SCALAR_FP32) → NONE → apply_to_tensor → return;     │
│                    ↑ 跳过，不做任何写入                           │
│                                                                  │
│  设计假设: "全局 memset 已归零" → 但 memset 不存在  ←── 🔴 BUG   │
└──────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                  GPU 标量 DTensor (cudaMalloc 残留)               │
│                                                                  │
│  beta (id=N): [ 0.0 ]  ← 用户期望 0.9  🔴                       │
│  wd   (id=M): [ 0.0 ]  ← 用户期望 0.0  ✓ (巧合)                 │
│  lr          : [ 0.1 ]  ← 训练循环 cudaMemcpyAsync 写入 ✓        │
│  scaling     : [ 1.0 ]  ← set_init_config 写入 ✓                │
└──────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                optimizer_op.cu (GPU kernel)                      │
│                                                                  │
│  m[i] = m[i] * _beta + g_i      // _beta=0, 动量失效             │
│  w[i] = w[i] * (1.0-_lr*_wd)   // _wd=0, 无正则化               │
│       - _lr * m[i]              // _lr=0.1                       │
│                                                                  │
│  实际: w[i] = w[i] - 0.1 * g_i  ← 纯 SGD                        │
│  预期: w[i] = w[i] - 0.1 * (0.9*m_{t-1} + g_t)  ← SGDM         │
└──────────────────────────────────────────────────────────────────┘
```