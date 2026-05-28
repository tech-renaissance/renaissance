# ABP2: Optimizer Scalar 修复后全面代码复盘报告

> **日期**: 2026-05-25
> **范围**: CNM1/CNM2 已修复的 Optimizer Scalar 初始化问题 + 周边代码全面审查
> **结论**: 核心 Bug 已彻底修复，训练精度与 PyTorch 一致。另发现 3 个代码质量/潜在风险问题，1 个已知的次要差异。

---

## 1. 已修复问题回顾（CNM1/CNM2 结论确认）

### 1.1 Optimizer Scalar 初始化链路 ✅ 已修复

| 检查项 | 状态 | 证据 |
|--------|------|------|
| `on_prepare()` 中为 beta/wd/beta2/eps/tc 设置 `init_config` | ✅ | `deep_learning_task.h:L307-L341` |
| `set_init_config()` 同步 `dtensor_cache_` | ✅ | `memory_plan.cpp:L903-L912` |
| `init_all()` 遍历 `dtensor_cache_` 并执行 `CONSTANTS` 初始化 | ✅ | `task_base.cpp:L1304-L1328` |
| `compile()` 全局 `cudaMemset(0)` 在 `pre_capture` 之前执行 | ✅ | `task_base.cpp:L240-L244` |
| 修复后训练精度（3 epochs, batch=200） | ✅ | Epoch 0: 96.33% vs PyTorch 96.20% |

**验证方式**: 运行 `test_dl_full.exe`，`init_all()` 日志确认 `beta(id=11)` 和 `wd(id=12)` 的 `init` 从 `0(NONE)` 变为 `2(CONSTANTS)`。

---

## 2. 新发现问题

### 2.1 🟡 GPU 训练路径缺少 `scheduler.step()` 调用

**现象**:
- CPU 路径 `run_train_epoch_cpu()` 在 **每个 batch 后** 调用 `scheduler.step()`（`deep_learning_task.cpp:L373-L376, L411-L416, L435-L439`）
- GPU 路径 `run_train_epoch_gpu()` **全程未调用** `scheduler.step()`

**证据**:
```cpp
// CPU 路径（deep_learning_task.cpp:L411-L416）
std::visit([](auto&& scheduler) {
    using T = std::decay_t<decltype(scheduler)>;
    if constexpr (!std::is_same_v<T, std::monostate>) {
        scheduler.step();
    }
}, sched_cfg_);

// GPU 路径（deep_learning_task.cpp:L864-L1072）—— 搜索 "scheduler" / "step()" 无匹配
```

**影响分析**:
- `fetch_lr_for_batch()` 使用的是**无状态查询**（`scheduler.cpp:L117-L134`）:
  - `get_lr_by_batch(batch_id)` → `compute_lr_at_step(batch_id)`
  - `get_lr_by_epoch(epoch_id)` → `compute_lr_at_step(epoch_id * steps_per_epoch)`
- 这两个函数**不读取** `scheduler.current_step_`，仅根据传入参数直接计算 LR
- **当前配置下无实际影响**: `StepLR(step_size=10).step_by_epoch()` + 3 epochs，LR 恒定为 0.1
- **但存在代码不一致和潜在风险**:
  - 若用户切换为 `step_by_batch()` 模式，GPU 路径的 `scheduler.current_step_` 永远为 0，而 `fetch_lr_for_batch()` 使用的是独立计算的 `global_step`，仍不受影响
  - 但如果未来有人修改 `fetch_lr_for_batch()` 使用 `scheduler.get_current_lr()`（状态查询），GPU 路径将完全失效

**修改建议**:
在 `run_train_epoch_gpu()` 的适当位置（例如 epoch 结束后、所有 rank 的线程 join 之后）添加：
```cpp
std::visit([](auto&& sch) {
    using T = std::decay_t<decltype(sch)>;
    if constexpr (!std::is_same_v<T, std::monostate>) sch.step();
}, sched_cfg_);
```

---

### 2.2 🟡 `lr_dtensor_id_` 在 `on_prepare()` 中的脆弱查找逻辑

**现象**:
`on_prepare()` 中通过遍历 `S_SCALAR_FP32` 区域找到第一个匹配的 DTensor 作为 `lr`：
```cpp
// deep_learning_task.h:L297-L305
lr_dtensor_id_ = -1;
for (const auto& dt : active_memory_plan_->dtensors()) {
    if (dt.region == Region::S_SCALAR_FP32) {
        lr_dtensor_id_ = dt.id;
        break;
    }
}
```

**证据**:
- 当前分配顺序下（`alloc_baseline_dtensors`），`S_SCALAR_FP32` 中的第一个 DTensor 确实是 `lr`（ID 5），第二个是 `scaling`（ID 6），第三个是 `beta`（ID 10+）
- **但是** `compile()` 中（`task_base.cpp:L265`）无条件覆盖了该值：
  ```cpp
  dl->lr_dtensor_id_ = active_memory_plan_->lr_id();  // 通过 baseline_.lr 直接查询
  ```

**影响分析**:
- **当前无功能性影响**: `compile()` 的覆盖确保 `lr_dtensor_id_` 始终正确
- **代码质量问题**:
  1. `on_prepare()` 中的 `TR_CHECK(lr_dtensor_id_ >= 0, ...)` 实际上检查的是**脆弱查找**的结果，而真实值在 `compile()` 中重新设置
  2. 如果未来有人在 `lr` 之前添加新的 `S_SCALAR_FP32` DTensor，`on_prepare()` 会找到错误的标量并抛出异常（或被覆盖后静默修复）
  3. 存在冗余逻辑：`on_prepare()` 查找 → `compile()` 覆盖

**修改建议**:
删除 `on_prepare()` 中的遍历查找逻辑和 `TR_CHECK`，统一在 `compile()` 中通过 `lr_id()` 设置。或者，如果需要在 `on_prepare()` 阶段就确定 `lr_dtensor_id_`，应改为：
```cpp
lr_dtensor_id_ = active_memory_plan_->lr_id();
TR_CHECK(lr_dtensor_id_ >= 0, ValueError, "LR DTensor not found");
```

---

### 2.3 🟡 SGD Momentum 的 Weight Decay 公式与 PyTorch 不一致

**现象**:
TR 的 `update_momentum_kernel`（`optimizer_op.cu:L39-L53`）实现：
```cpp
m[i] = m[i] * _beta + g_i;
w[i] = w[i] * (1.0f - _lr * _wd) - _lr * m[i];
```

PyTorch `torch.optim.SGD` 的数学公式（L2 regularization）：
```
buffer = momentum * buffer + (grad + weight_decay * param)
param = param - lr * buffer
```

**证据**:
- 当 `beta=0`（无 momentum）时，两者等价：
  - TR: `w = w - lr*wd*w - lr*g`
  - PyTorch: `w = w - lr*(g + wd*w) = w - lr*g - lr*wd*w`
- 当 `beta > 0` 且 `wd > 0` 时，两者**不等价**：
  - TR: weight decay 不经过 momentum buffer（decoupled）
  - PyTorch: weight decay 经过 momentum buffer（L2 regularization）

**影响分析**:
- **当前测试中无影响**: `test_dl_full.cpp` 配置 `weight_decay(0.0f)`
- **潜在影响**: 如果用户设置 `weight_decay > 0`，TR 与 PyTorch 的收敛行为将出现系统性差异
- 这不是一个"bug"，而是一个**设计差异**。但文档/注释中未明确说明 TR 的 SGD 使用 decoupled weight decay

**修改建议**:
在 `optimizer_op.cu` 的注释或相关文档中明确标注：
> "TR 的 SGD Momentum 使用 decoupled weight decay（`w = w*(1-lr*wd) - lr*m`），与 PyTorch 的 L2 regularization（`m = beta*m + (g + wd*w)`）在 `momentum>0 && wd>0` 时不等价。"

如果需要与 PyTorch 完全对齐，需修改 kernel：
```cpp
// PyTorch 风格 L2 regularization
float g_i = g[i] + _wd * w[i];  // L2 grad
m[i] = m[i] * _beta + g_i;
w[i] = w[i] - _lr * m[i];
```

---

### 2.4 🟢 `run_train_epoch_gpu()` 返回最后一个 batch 的 loss（已知次要差异）

**现象**:
`run_train_epoch_gpu()` 返回 `fetch_from_rank(loss_dt, 0)`，即 rank 0 上最后一个 batch 的 loss 值。

**证据**:
```cpp
// deep_learning_task.cpp:L1064-L1071
float train_loss = 0.0f;
if (loss_id >= 0) {
    const auto& loss_dt = active_memory_plan_->get_dtensor(loss_id);
    Tensor h_loss = fetch_from_rank(loss_dt, 0);
    train_loss = h_loss.data<float>()[0];
}
return train_loss;
```

**影响分析**:
- 仅影响 `log_epoch_results()` 中打印的 `train_loss`
- **不影响权重更新**（每个 batch 的梯度计算和更新是正确的）
- 与 PyTorch 的 epoch 平均 loss 不一致，但属于已知的日志层面差异

**修改建议**:
如需与 PyTorch 对齐，可在 `run_train_epoch_gpu()` 中累加每个 batch 的 loss（需从 GPU 读取每个 batch 的 loss 值，有性能开销），或接受当前行为并在文档中注明。

---

## 3. 其他检查项（确认无问题）

| 检查项 | 结论 | 证据 |
|--------|------|------|
| `scaling` 初始化 | ✅ 正确 | `compiler.cpp:L748-L749` 设置为 `CONSTANTS(1.0f)` |
| `inv_scaling` 无需初始化 | ✅ 正确 | FWD kernel 每次前向覆盖（`softmax_ce_op.cu:L139`） |
| Kaiming init 公式 | ✅ 与 PyTorch 一致 | `gain = sqrt(2/(1+a²))`（`initializer.cpp`） |
| Bias 初始化（ZEROS） | ✅ 与 PyTorch 一致 | `initializer.cpp:derive()` 中 `is_bias_region()` 返回 ZEROS |
| Data preprocessing | ✅ 与 PyTorch 一致 | `(pixel/255 - 0.1307)/0.3081` |
| TF32 关闭 | ✅ 已验证 | PyTorch 侧 `torch.backends.cuda.matmul.allow_tf32 = False` |
| Philox RNG | ✅ 分布正确 | 已验证 uniform 分布 |
| SoftmaxCE BWD gradient | ✅ 正确 | `scale = scaling * inv_scaling = 1.0/batch`，与 PyTorch `reduction='mean'` 一致 |
| `lr` 每 batch 更新时机 | ✅ 正确 | `cudaMemcpyAsync` 在 `g_wu` 之前，位于 `s_up` 流 |
| `loss` 每 batch 清零 | ✅ 正确 | `cudaMemsetAsync(ctx.ptr_at(loss_id), 0, ...)` 在每个 batch 的 FWD 前执行 |
| 全局 arena `cudaMemset(0)` | ✅ 已执行 | `task_base.cpp:L240-L244`，在 `pre_capture` 之前 |

---

## 4. 总结

### 4.1 修复状态

| 问题 | 严重性 | 状态 |
|------|--------|------|
| Optimizer Scalar 未初始化（CNM1/CNM2） | 🔴 高 | **已修复**，精度与 PyTorch 一致 |
| GPU 路径缺少 `scheduler.step()` | 🟡 中 | **代码不一致**，当前无功能影响，建议补充 |
| `lr_dtensor_id_` 脆弱查找 | 🟡 低 | **代码质量问题**，被 `compile()` 覆盖，建议清理 |
| SGD Weight Decay 公式差异 | 🟡 低 | **设计差异**，`wd=0` 时无影响，建议文档标注 |
| Train loss 返回最后一个 batch | 🟢 信息 | **已知差异**，仅影响日志 |

### 4.2 关键结论

1. **核心 Bug 已根治**: Optimizer Scalar 初始化修复后，MLP MNIST 3-epoch 训练精度从 95.64% 提升到 97.61%，超过 PyTorch 的 97.27%。
2. **无其他功能性 Bug**: 经过对 `on_prepare()`、`compile()`、`run_train_epoch_gpu()`、`run_val_epoch_gpu()`、算子 kernel、`scheduler`、`initializer` 等关键路径的全面审查，未发现其他导致精度差异的 Bug。
3. **建议的后续动作**:
   - [ ] 在 `run_train_epoch_gpu()` 中补充 `scheduler.step()`（保持与 CPU 路径一致）
   - [ ] 清理 `on_prepare()` 中脆弱的 `lr_dtensor_id_` 查找逻辑
   - [ ] 在文档/注释中明确 TR SGD 的 decoupled weight decay 设计
