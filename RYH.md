# AMP 训练失败调试报告

## 当前发现

### 1. H2D 数据本身无 NaN
- 在训练/验证 loop 的每次 H2D 传输后，读取 `I_A_DATA` / `I_B_DATA` 回 CPU 检查
- **结论**：所有 batch 的输入数据均不含 NaN，数据本身没问题

### 2. FC3 输出（logits）无 NaN
- 在 val loop 的 `g_inf`（推理图）执行后，读取 logits 张量（id=42, `F_FEATURE_FP16`）检查
- **结论**：logits 不含 NaN，forward 计算链（FC→Tanh→FC→Tanh→FC）未产生 NaN

### 3. `scaling`（grad scaling）异常为 0
- 在 val loop 中读取 `S_SCALAR_FP32` 中的 `scaling` 值（baseline id=7）
- **预期值**：`TR_AMP_INITIAL_SCALING = 65536.0f`
- **实际值**：`0`
- **影响**：`softmax_ce` kernel 中 `loss += sample_loss * inv_batch * scaling` 若 scaling=0 则加 0

### 4. `loss` 为 NaN，但 `top1/top5` 有正常值
- val 每个 batch 的读取结果示例：
  ```
  loss=nan top1=0.105 top5=0.205 logits_nan=NO scaling=0 bs=200
  ```
- `top1/top5` 有正常数值 → `softmax_ce_inf_kernel` 的 top1/top5 统计逻辑工作正常
- `loss=nan` → 问题集中在 loss 的计算路径

## 核心矛盾与关键发现

### 发现 A：`INF * 0 = NaN` 可以解释 `loss=nan`
在 `softmax_ce_inf_kernel` 中：
```cuda
float inv_batch = 1.0f / static_cast<float>(*batch_size_ptr);
// ...
atomicAdd(loss, sample_loss * inv_batch * scaling);
```
若 `*batch_size_ptr == 0`，则 `inv_batch = INF`。即使 `scaling=0`：
`sample_loss * INF * 0 = NaN`（因为 `INF * 0 = NaN`）

**这意味着 `batch_size_ptr` 可能指向了 0！**

但 NAN-CHECK 从 `b.local_batch_size` 读出的值是 `bs=200`。这说明：
- `b.local_batch_size` 标量本身的值是 200（正确）
- 但 `softmax_ce` kernel 中的 `batch_size_ptr` 可能读取到了不同的值

** hypothesis：kernel 中的 `batch_size_ptr` 指针地址有误，或指向了 `scaling` 的地址（而 scaling=0）！**

### 发现 B：`scaling` 初始化路径代码层面正确
通过完整代码走查确认：
1. `compiler.cpp` 第 749-751 行：`set_init_config(scaling, kInitConstant(65536.0f))` ✓
2. `MemoryPlan::set_init_config` 同时更新 `entries_` 和 `dtensor_cache_` ✓
3. `init_all()` 遍历 `dtensor_cache_` 并调用 `init()` ✓
4. `init()` 中 `apply_to_tensor` 调用 `host.fill(65536.0f)` 然后 H2D ✓
5. 在 `compile_impl` 中，`init_all()` 在 `pre_capture` 之后执行 ✓

**结论**：`scaling=0` 不应该是初始化失败导致的，而是在 `init_all()` 之后被覆盖。

### 发现 C：`pre_capture` 中的 `cudaMemset` 发生在 `init_all()` 之前
在 `compile_impl` 中：
```cpp
// 1. on_prepare() 设置 active_memory_plan_
// 2. compile_alloc_hardware() 分配 Arena
// 3. cudaMemset(Arena, 0, total_bytes)  ← 清零整个 Arena
// 4. pre_capture() → warmup + CUDA Graph 捕获
// 5. build_exec_table()
// 6. compile_mark_compiled()
// 7. init_all()  ← 初始化所有 DTensor（包括 scaling=65536）
```

`init_all()` 在 memset 之后执行，所以 `scaling` 应该被正确设置。但如果 `pre_capture` 的 warmup 或首次 graph launch 覆盖了 `S_SCALAR_FP32` 区域...

但 `S_SCALAR_FP32` 只包含标量，不会被 layer 操作访问。

**更可疑的是：`build_exec_table()` 中通过 `cudaMemcpy` 写入了 `local_batch_size`、`last_train_batch_size`、`last_val_batch_size`。如果这些写操作的地址计算有误...**

## 当前碰到的问题

1. **`scaling` 初始化路径代码正确但值仍为 0**：最可能的原因是有其他代码在 `init_all()` 之后覆盖了 `scaling` 的值，或 `scaling` 和另一个 DTensor 的地址重叠。

2. **`loss=nan` 最可能的根因**：`batch_size_ptr` 在 kernel 中读取到了 0（可能是 `scaling` 的地址），导致 `inv_batch = INF`，进而 `sample_loss * INF * 0 = NaN`。

3. **调试手段受限**：需要确认 kernel 中实际读取到的 `batch_size_ptr` 和 `scaling_ptr` 的值。

## 下一步需要查什么

### 最高优先级：验证 `batch_size_ptr` 和 `scaling_ptr` 的指针地址
1. **在 `softmax_ce_op.cu` 中加 `printf` 调试**
   - 在 `softmax_ce_inf_kernel` 第 0 个 block/thread 中 `printf`：
     - `batch_size_ptr` 的值（是否 = 0？）
     - `scaling_ptr` 的值（是否 = 0？）
     - `scaling_ptr` 和 `batch_size_ptr` 的地址差值（确认是否指向同一地址）

2. **检查 `build_computation_graph` 中 SoftmaxCE 的 input_ids**
   - `compiler.cpp` 第 1344 行：`gn.input_ids.push_back(b.local_batch_size);`
   - `compiler.cpp` 第 1343 行：`gn.input_ids.push_back(b.scaling);`
   - 确认 `b.local_batch_size` 和 `b.scaling` 的 ID 值是否不同

3. **在 `init_all()` 后立即读取 `scaling` 和 `lr` 的值**
   - 在 `compile_impl()` 的 `init_all()` 之后增加 D2H 读取
   - 确认 `init_all()` 后 `scaling` 是否为 65536.0f
   - 若 init_all 后正确，则问题在后续覆盖；若 init_all 后就是 0，则初始化路径有问题

### 高优先级
4. **检查 `build_exec_table()` 中的 `cudaMemcpy`**
   - 确认 `ctx.ptr_at(b.local_batch_size)` 的地址与 `init_all()` 中 `transfer_to_rank` 的地址是否一致
   - 确认 `b.local_batch_size`、`b.last_train_batch_size`、`b.last_val_batch_size` 的 ID 是否正确

5. **检查 `probs` 是否有 NaN**
   - 在 val loop 的 `g_inf` 执行后，读取 `probs` 张量检查 NaN
   - 排除 `prob` 本身产生 NaN 的可能性

### 中优先级
6. **对比 FP32 模式的行为**
   - FP32 模式下 `scaling=1.0f`（非 AMP），确认 `scaling` 值是否正确
   - 若 FP32 模式下 `scaling` 也失效，则说明是全局初始化问题

7. **检查 `pre_capture` 是否覆盖 `S_SCALAR_FP32`**
   - 在 `pre_capture` 前后分别读取 `scaling` 值
   - 确认是否是 warmup/capture 阶段覆盖了标量
