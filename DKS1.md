# DKS1: Last Batch 处理修复方案

## 一、问题综述

综合 **小伙伴S**、**小伙伴K**、**小伙伴D** 三份独立审查意见，当前代码在 last batch 处理上存在以下问题：

| 路径 | 子问题 | 严重性 | 发现者 |
|------|--------|--------|--------|
| GPU 训练/验证 | CUDA Graph 固定 grid 导致阻塞超出 `last_batch_size` 的样本执行 kernel，处理残留数据并污染 loss/top1/top5 | **🔴 致命** | 小伙伴K |
| GPU 训练/验证 | cuDNN Conv/FC/BN descriptor 的 batch size 固定为 `local_batch_size`，last batch 时处理额外样本 | **🔴 致命** | 小伙伴K |
| CPU 训练 | `run_train_epoch_cpu` 完全没有指标累积，只返回最后一个 batch 的 loss | **🔴 致命** | 小伙伴D |
| CPU 验证 | `run_val_epoch_cpu` 按 batch 数平均（`/val_batches`），非按样本数加权（应`/num_val_samples`） | 🟡 中等 | 小伙伴S、小伙伴D |
| CPU 验证 | 未使用 `*batch_size` 加权累积（直接加 `batch_loss` 而非 `batch_loss * batch_size`） | 🟡 中等 | 小伙伴D |

**小伙伴S** 的审查确认了 GPU 路径的图选择（`ACCUM_METRICS_TRAIN_LAST` / `ACCUM_METRICS_VAL_LAST`）和最终除法（除以总样本数）是正确的——但未发现 CUDA Graph grid 维度固化导致的数据污染问题。

---

## 二、根因分析

### 2.1 CUDA Graph Grid 维度固化（核心 Bug）

**现象**：[softmax_ce_op.cu:L75](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu#L75)

```cpp
// FWD/INF kernel (每 sample 一个 block)
int b = blockIdx.x;
if (b >= batch) return;   // batch 是 capture 时的 local_batch_size (如 128)
```

Graph capture 时 `batch = local_batch_size = 128`，grid = 128 blocks。Replay 时 128 个 block 全部启动。对于 last batch (如 80 samples)：

| blockIdx.x | `b >= batch` (128) | 实际访问的 `labels[b]` / `logits[b*stride]` | 后果 |
|:----------:|:------------------:|---------------------------------------------|------|
| 0~79 | 不触发 | 有效样本数据 | ✅ 正确 |
| **80~127** | **永不触发** | **上一 batch 残留数据** | ❌ 污染 loss/top1/top5 |

同理 [BWD kernel L318](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu#L318)：`total = batch * num_classes` 固定在 capture 时的值，grid 覆盖额外的 `(local_bs - last_bs) * num_classes` 个元素，读取残留 probs/labels 产生错误梯度。

**cuDNN 操作**（Conv/FC/BN）：其 descriptor 中的 batch size 也在 capture 时固定，无法通过 kernel 修改规避。

### 2.2 残留数据来源

[transfer_station.cpp:L375-L420](file:///r:/renaissance/src/data/transfer_station.cpp#L375-L420)：`reset_and_update()` 为性能优化**明确移除**了 buffer 的 memset。当 last batch 只有 `last_batch_size` 个有效样本被 H2D 传输时，buffer 中 `[last_bs .. local_bs-1]` 区域仍保留着**上一 batch 的旧 labels 和 data**。

### 2.3 CPU 路径的独立问题

- **`run_train_epoch_cpu`**([L1397-L1400](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1397-L1400))：直接读取最后 batch 的 `loss_id` 返回，未使用 `g_accum` / `g_accum_train_last` 累积
- **`run_val_epoch_cpu`**([L1622-L1624](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1622-L1624))：`acc_loss / val_batches` 按 batch 数除，而 GPU 路径是 `accum_loss / num_val_samples()` 按样本数除

---

## 三、修复方案

### 修复 A：SOFTMAX_CE kernel 使用动态 `batch_size_ptr` 做边界检查（优先级最高）

**文件**：[softmax_ce_op.cu](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu)

**A1. FWD kernel (L75)**：
```cpp
// 改前:
if (b >= batch) return;

// 改后:
if (b >= *batch_size_ptr) return;  // 使用运行时 batch size，block 索引超出有效样本数则返回
```

**A2. INF kernel (L178)**：同上修改。

**A3. BWD kernel (L313-L318)**：
```cpp
// 改前:
int total = batch * num_classes;
if (idx >= total) return;

// 改后: BWD kernel 需要新增 batch_size_ptr 参数
// 签名增加: const int32_t* __restrict__ batch_size_ptr,
int total = (*batch_size_ptr) * num_classes;
if (idx >= total) return;
```

**A4. BWD launch 函数**：`launch_softmax_ce_bwd_fp32` / `launch_softmax_ce_bwd_amp` 需要新增 `batch_size_ptr` 参数。

**A5. BWD dispatch**：`launch_softmax_ce_fp32_bwd_cuda` / `launch_softmax_ce_amp_bwd_cuda` 在 [softmax_ce_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cpp) 中传递 `batch_size_ptr`。

> **重要**：修复 A 只能保护 SOFTMAX_CE kernel，cuDNN Conv/FC/BN 仍需要修复 B 配合。

### 修复 B：Last batch 执行前清零残留数据区（保护 cuDNN 算子）

**文件**：[deep_learning_task.cpp](file:///r:/renaissance/src/task/deep_learning_task.cpp)

**原理**：在 last batch 的 FWD/INF graph 启动前，对 labels 和 data 的残留区做 `cudaMemsetAsync(0)`。清零后：

- cuDNN Conv/FC/BN 处理的全零数据产生全零输出，不传播错误信号
- 修复 A 确保 SOFTMAX_CE 跳过残留 block（不读 labels/logits）
- label=0（零值）被 FWD kernel 读取后产生的 loss 贡献被修复 A 的边界检查跳过

**需要清零的区域**：

| 区域 | 起始偏移 | 大小 |
|------|----------|------|
| labels | `label_base + last_bs * sizeof(int)` | `(local_bs - last_bs) * sizeof(int)` |
| data buffer A | `data_a_base + last_bs * sample_stride_a` | `(local_bs - last_bs) * sample_stride_a` |
| data buffer B | `data_b_base + last_bs * sample_stride_b` | `(local_bs - last_bs) * sample_stride_b` |

**B1. 训练 last batch (L1037-L1057 之前插入)**：
```cpp
// 训练 last batch: 清零残留数据区
{
    auto& registry = GlobalRegistry::instance();
    int last_bs = registry.get_last_train_batch_size();
    int local_bs = registry.get_local_batch_size();
    if (last_bs < local_bs) {
        const auto& b = active_memory_plan_->baseline();
        // 清零 labels 残留区
        int32_t label_id = last_a ? b.label_a : b.label_b;
        const DTensor& label_dt = active_memory_plan_->get_dtensor(label_id);
        char* label_base = static_cast<char*>(ctx.ptr_at(label_id));
        size_t label_extra = static_cast<size_t>(local_bs - last_bs) * sizeof(int32_t);
        cudaMemsetAsync(label_base + last_bs * sizeof(int32_t), 0, label_extra, s_trans);

        // 清零 data 残留区 (I_A_DATA 或 I_B_DATA)
        int32_t data_id = last_a ? b.data_a : b.data_b;
        const DTensor& data_dt = active_memory_plan_->get_dtensor(data_id);
        char* data_base = static_cast<char*>(ctx.ptr_at(data_id));
        size_t sample_stride = data_dt.n_stride_cuda() * data_dt.element_size();
        size_t data_extra = static_cast<size_t>(local_bs - last_bs) * sample_stride;
        cudaMemsetAsync(data_base + last_bs * sample_stride, 0, data_extra, s_trans);
        cudaStreamSynchronize(s_trans);
    }
}
```

**B2. 验证 last batch (L1475 循环内，判断 `is_last` 时插入)**：
- 同样逻辑，使用 `get_last_val_batch_size()`、`b.label_smce`（验证用 labels）、`b.data_a`/`b.data_b`

**关于 `need_filling_` 的说明**：训练路径中如果 `need_filling_` 为 true（默认行为），TransferStation 会对最后一个 batch 做 filling（用第一个样本填充），使其达到 `local_batch_size`。此时 `last_bs` 会等于 `local_bs`，残留数据问题不触发。但为防御性编程，修复 B 中的 `if (last_bs < local_bs)` 条件确保只在真正需要时执行。

### 修复 C：CPU 训练路径 — 添加指标累积

**文件**：[deep_learning_task.cpp](file:///r:/renaissance/src/task/deep_learning_task.cpp)

**C1. 修改 `run_train_epoch_cpu`**：与 GPU 路径对齐，使用 `g_accum` / `g_accum_train_last` 累积 loss/top1/top5。

需要新增的关键步骤：
- 在 batch 循环前调用 `CLEAR_METRICS`
- 每个常规 batch 后调用 `ACCUM_METRICS`（使用 `local_batch_size`）
- 最后 batch 后调用 `ACCUM_METRICS_TRAIN_LAST`（使用 `last_train_batch_size`）
- 返回 `accum_loss / num_train_samples()`

由于 CPU 路径不使用 CUDA Graph，累积图需要以即时模式执行（`graphs[idx].launch(0, nullptr)`）。

**C2. 修改 `run_val_epoch_cpu`**：
```cpp
// 改前:
acc_loss += batch_loss;                           // L1614
float avg_loss = acc_loss / val_batches;           // L1622

// 改后:
int bs = (batch == val_batches - 1)
    ? GlobalRegistry::instance().get_last_val_batch_size()
    : GlobalRegistry::instance().get_local_batch_size();
acc_loss += batch_loss * static_cast<float>(bs);  // 加权累积
// ...
float avg_loss = acc_loss / GlobalRegistry::instance().num_val_samples();  // 按样本数除
```

### 修复 D：统一使用 GlobalRegistry 的 last batch 计算

**文件**：[deep_learning_task.cpp](file:///r:/renaissance/src/task/deep_learning_task.cpp)、[task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp#L295-L305)

当前 GPU 路径已正确从 `GlobalRegistry` 获取并写入 `last_train_batch_size` 和 `last_val_batch_size` 标量。需确认：
- CPU 路径也使用相同来源
- 不在 DeepLearningTask 中独立计算 last batch size

**GlobalRegistry 计算逻辑**([global_registry.cpp:L2085-L2099](file:///r:/renaissance/src/core/global_registry.cpp#L2085-L2099))：
```
padded = ceil(num_samples / world_size) * world_size           # 向上 pad 到 world_size 倍
per_rank = padded / world_size                                 # 每个 rank 样本数
last_bs = per_rank % local_batch_size                          # 余数
         == 0 ? local_batch_size : last_bs                     # 整除则 = local_bs
```

---

## 四、修复执行顺序

| 顺序 | 修复 | 影响范围 | 风险 |
|:----:|------|----------|------|
| 1 | **A**: SOFTMAX_CE kernel 边界检查 | `softmax_ce_op.cu` (+cpp) | 低，仅改 kernel 内 guard |
| 2 | **B**: Last batch 残留数据清零 | `deep_learning_task.cpp` | 中，需验证 memset 地址/大小 |
| 3 | **C**: CPU 路径指标累积修复 | `deep_learning_task.cpp` | 中，需验证 CPU 图即时执行 |
| 4 | **D**: 确认依赖 GlobalRegistry | 审阅即可 | 低 |

---

## 五、测试验证

修复后应运行以下测试确认正确性：

1. **`test_dl_full --cpu`**：验证 CPU 路径累积正确
2. **`test_dl_full --gpu`**：验证 GPU FP32 最后 batch 不产生污染
3. **`test_dl_full --amp`**：验证 AMP 最后 batch 不产生污染
4. **`test_dl_full_gpu`**：独立 GPU 测试，确保复活
5. **`test_dl_full_amp`**：独立 AMP 测试，确保复活

CIFAR-10 数据集（50000 train / 10000 val, batch=200, world_size=1）：
- `steps_per_epoch = 250`, `last_train_batch_size = 200`（整除，无最后 batch 问题）
- 需构造一个不能整除的场景（如手动设置 `batch=192`）来验证 last batch fix

---

## 六、备选方案（未采纳）

| 方案 | 描述 | 未采纳原因 |
|------|------|------------|
| 单独 capture last batch graph | 为 `DEEP_FWD_BWD_LAST` 等新增 GraphId | 内存翻倍，代码侵入大，约需新增 10+ GraphId |
| TransferStation 恢复 memset | 在 buffer 切换时清零 | 性能回退（每秒约 600MB 无用 memset），之前特意移除 |
| 仅改 kernel 不处理 cuDNN | 只改 SOFTMAX_CE，忽略 cuDNN | cuDNN Conv/FC/BN 仍会处理残留数据，污染 feature maps 和梯度 |