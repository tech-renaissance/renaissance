# TR4 MLP MNIST 精度修复报告：未初始化 Optimizer Scalar 导致 Momentum 失效

## 1. 问题现象

在 MLP MNIST 训练任务中（batch=200，3 epochs，Kaiming Uniform 初始化，SGD + Momentum=0.9），Tech-Renaissance（TR）的验证准确率显著低于 PyTorch 参考实现：

| Epoch | TR（修复前） | PyTorch | 差距 |
|-------|-------------|---------|------|
| 0     | 92.52%      | 96.20%  | -3.68% |
| 1     | 94.27%      | 96.64%  | -2.37% |
| 2     | 95.64%      | 97.27%  | -1.63% |

该差距在所有超参数对齐（batch size、初始化、loss scaling、TF32 关闭）后仍然稳定复现。

## 2. 根因分析

### 2.1 核心结论

**Optimizer 标量 `beta`（momentum 系数）和 `wd`（weight decay）的 DTensor 在 GPU 上从未被写入正确数值，全局 `cudaMemset(0)` 后保持 `0.0f`。**

对于 `SGD_MOMENTUM` 更新核（`optimizer_op.cu`）：
```cpp
float _beta = *beta;  // 读取到 0.0f
m[i] = m[i] * _beta + g_i;  // 退化为 m[i] = g_i（纯 SGD，无动量）
```

Momentum=0.9 对 MNIST 早期收敛至关重要。缺失动量导致：
- Epoch 0 差距最大（~3.7%），因为动量对早期梯度方向平滑作用最强
- 后续 epoch 差距逐渐缩小但仍存在，因为无动量 SGD 收敛更慢

### 2.2 代码路径

`MemoryPlan::alloc_baseline_dtensors()` 分配了以下标量 DTensor：
- `baseline_.lr`（学习率）
- `baseline_.scaling`（loss scaling）
- `baseline_.beta`（momentum / Adam beta1）
- `baseline_.wd`（weight decay）

这些 DTensor 的 `init_config` 默认为 `InitKind::NONE`。`TaskBase::init_all()` 会跳过 `NONE` 的 DTensor。

`scaling` 的 `init_config` 在 `Compiler::compile()` 中被显式设置为 `CONSTANTS(1.0f)`，因此正确初始化。
**但 `beta` 和 `wd` 的 `init_config` 从未被设置。**

### 2.3 `set_init_config` 的隐藏 Bug

在尝试修复时，最初在 `DeepLearningTask::on_prepare()` 中调用 `active_memory_plan_->set_init_config(id, ...)` 来设置 `beta`/`wd` 的 `init_config`。

然而发现 **`MemoryPlan::set_init_config()` 只修改 `entries_[idx].dt.init_config`，没有同步 `dtensor_cache_`**。

`init_all()` 遍历的是 `dtensors()` → `dtensor_cache_`，因此即使 `set_init_config` 被调用，`init_all()` 仍然看到旧的 `NONE`，跳过初始化。

这是本次修复中最隐蔽的一点：**修改了配置，但遍历路径看不到**。

## 3. 修复内容

### 3.1 文件 1：`include/renaissance/task/deep_learning_task.h`

在 `on_prepare()` 中，`lr_dtensor_id_` 查找之后、`add_graph()` 之前，添加优化器标量 `init_config` 设置逻辑：

```cpp
// 设置优化器标量 DTensor 的 init_config，使 init_all() 能将其初始化为正确常数
auto set_scalar_init = [this](int32_t id, float value) {
    if (id >= 0) active_memory_plan_->set_init_config(id, InitConfig{value, InitKind::CONSTANTS, FanMode::FAN_IN});
};
if (auto* sgd = std::get_if<SGD>(&opt_cfg_)) {
    Optimizer opt = *sgd;
    if (const auto* cfg = opt.as<SGDConfig>()) {
        set_scalar_init(active_memory_plan_->beta_id(), cfg->momentum);
        set_scalar_init(active_memory_plan_->wd_id(),   cfg->weight_decay);
    }
} else if (auto* lars = std::get_if<LARS>(&opt_cfg_)) {
    // ... LARS: beta, wd, tc, eps
} else if (auto* adam = std::get_if<Adam>(&opt_cfg_)) {
    // ... Adam: beta1, wd, beta2, eps
} else if (auto* adamw = std::get_if<AdamW>(&opt_cfg_)) {
    // ... AdamW: beta1, wd, beta2, eps
}
```

同时添加 `#include "renaissance/core/init_config.h"`。

### 3.2 文件 2：`src/graph/memory_plan.cpp`

修复 `MemoryPlan::set_init_config`，同步更新 `dtensor_cache_`：

```cpp
void MemoryPlan::set_init_config(int32_t id, const InitConfig& config) {
    auto it = id_to_idx_.find(id);
    TR_CHECK(it != id_to_idx_.end(), IndexError,
             "DTensor id " << id << " not found in MemoryPlan");

    entries_[it->second].dt.init_config = config;
    // 同步更新 dtensor_cache_，确保 init_all() 等遍历路径可见
    if (!dtensor_cache_.empty() && it->second < static_cast<int32_t>(dtensor_cache_.size())) {
        dtensor_cache_[it->second].init_config = config;
    }
}
```

### 3.3 文件 3：`src/graph/memory_plan.cpp`（回滚）

此前在 `alloc_baseline_dtensors()` 中尝试过直接为 `scaling` 设置 `init_config`，但发现 `compiler.cpp` 已在 `Compiler::compile()` 中完成同样的设置，因此回滚了该冗余修改。

## 4. 修复效果

重新编译运行 `test_dl_full.exe`（batch=200，3 epochs）：

| Epoch | TR（修复前） | TR（修复后） | PyTorch | 修复后 vs PyTorch |
|-------|-------------|-------------|---------|-----------------|
| 0     | 92.52%      | **96.33%**  | 96.20%  | +0.13% |
| 1     | 94.27%      | **97.15%**  | 96.64%  | +0.51% |
| 2     | 95.64%      | **97.61%**  | 97.27%  | +0.34% |

- Epoch 0 准确率从 92.52% 跃升至 **96.33%**，与 PyTorch 的 96.20% 基本持平
- Epoch 2 准确率达到 **97.61%**，超过 PyTorch 的 97.27%
- 收敛曲线形态与 PyTorch 高度一致

`init_all()` 日志也确认 `beta`（id=11）和 `wd`（id=12）的 `init` 从 `0`（NONE）变为 `2`（CONSTANTS）：
```
[DEBUG] dtensor[11] id=11 region=57 shape=[1,1,1,1] init=2
[DEBUG] dtensor[12] id=12 region=57 shape=[1,1,1,1] init=2
```

## 5. 经验总结

1. **标量初始化不容忽视**：GPU 标量 DTensor 虽然仅占 4 字节，但如果值错误，会在 kernel 中被广播到整个梯度/更新计算，造成系统性偏差。
2. **初始化器路径必须完整**：`init_all()` → `init(dtensor)` → `dtensor.init_config` 的链路中，任何环节的配置遗漏都会导致静默失败。
3. **配置修改需同步所有视图**：`MemoryPlan` 同时维护 `entries_` 和 `dtensor_cache_`，修改时必须确保两者一致，否则会出现"修改了但遍历不到"的诡异现象。
4. **早期收敛差距是动量缺失的标志性信号**：Epoch 0 差距 3.7% 是典型特征——动量对早期训练阶段的梯度方向平滑和加速作用最为明显。

## 6. 附录：`inv_scaling` 的澄清

`inv_scaling`（SoftmaxCE 输出的 `1/batch` 缩放因子）是 **FWD kernel 的运行时输出**，每次前向传播都被 kernel 覆盖：
```cpp
if (b == 0 && tid == 0) *inv_scaling = inv_batch;  // 1.0f / batch
```
BWD kernel 在 FWD 之后执行，读取的总是正确的运行时值。因此 `inv_scaling` 不需要初始化阶段设置，其 DTensor 的 `init_config` 保持 `NONE` 是正确且无害的。

`scaling`（loss scaling 系数）已由 `compiler.cpp` 显式设置为 `CONSTANTS(1.0f)`，无需额外处理。

---

**报告日期**：2026-05-26
**修复版本**：4.20.3+  
**作者**：Kimi Code CLI
