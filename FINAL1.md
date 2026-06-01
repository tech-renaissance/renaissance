# 多 RANK 训练验证准确率异常 — 最终根因分析报告

> **日期**: 2026-06-01
> **关键发现者**: 姜总工
> **结论**: `VAL_RESULT_ALLREDUCE`（即 `RANGE_MEAN_ALLREDUCE` on `VAL_RESULT_COMM` graph）的 NCCL 通信完全失效

---

## 一、现象回顾

| RANK | Best Val Top-1 | 评估 |
|------|---------------|------|
| 1    | 99.41%        | 基准 |
| 2    | **99.62%**    | 异常偏高 |
| 4    | 99.44%        | 接近正常 |
| 8    | **99.04%**    | 严重偏低 |

- 训练 loss 曲线各 RANK 几乎重合
- 验证 loss 随 RANK 数增加而明显上升
- 单 RANK 和多 RANK 在数学上不等价

---

## 二、姜总工的关键洞察（本次分析的转折点）

> RANKS/*.txt 的数据当中，RANK=2 的时候，val 正确率乘以 100 必定是 2 的倍数；
> RANK=4 的时候，val 正确率乘以 100 必定是 4 的倍数；
> RANK=8 的时候，val 正确率乘以 100 必定是 8 的倍数！！！

> 极有可能是 VAL_RESULT_COMM 失效了。——这导致通信失败，得到的数据并没有在多个
> RANK 上平均，最后打印的时候又乘以 RANK 数，结果就导致正确样本数只能是 RANK
> 的整数倍——因为它只统计了一个 RANK！

---

## 三、数学推导：AllReduce 失效的演绎证明

### 3.1 单 batch 的 top1 计算

[softmax_ce_op.cu:306](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu#L299-L306)

```cuda
float inv_batch = 1.0f / static_cast<float>(batch);
// ...
if (global_top1_cls == label_b) atomicAdd(top1, inv_batch);
```

每 batch 统计的正确样本数 = `top1 × batch_size` = **整数**。

### 3.2 ACCUM_METRICS 累加

[accum_op.cu:20](file:///r:/renaissance/src/backend/ops/range/accum_op.cu#L20)

```cuda
*accum_top1 += (*batch_top1) * bs;
```

其中 `bs = batch_size`（int → float），`batch_top1 = correct_in_batch / batch_size`。

```
accum_top1 = Σ(correct_in_batch / batch × batch)
           = Σ(correct_in_batch)
           = 该 RANK 的正确样本总数（整数！）
```

**关键结论：累加完成后，每个 RANK 的 `accum_top1` 是一个精确的整数（以 float 存储）。**

### 3.3 VAL_RESULT_ALLREDUCE

[allreduce_op.cpp:77-92](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L77-L92)

```cpp
ncclAllReduce(dst, dst, count, ncclFloat32, ncclSum, ncclComm, s);  // sum
float inv = 1.0f / static_cast<float>(world_size);                    // mean scale
launch_tr_scale_fp32_kernel(dst, inv, count, s);                      // post-scale
```

#### 若 AllReduce 正常工作：

```
accum_top1_after = Σ(correct_count_i) / world_size       (所有 RANK 求平均)
avg_top1 = (Σ(correct_i) / world_size) / val_samples_per_rank()
         = (Σ(correct_i) / world_size) / (10000 / world_size)
         = Σ(correct_i) / 10000
         = 全局准确率 ✅
```

#### 若 AllReduce 完全失效（NCCL 通信未执行，scale 也未执行）：

```
accum_top1_after = correct_count_rank0                  (整数!)
avg_top1 = correct_count_rank0 / val_samples_per_rank()
         = correct_count_rank0 / (10000 / world_size)
         = correct_count_rank0 × world_size / 10000
```

**打印时：**
```
displayed_correct_count = avg_top1 × 10000
                        = correct_count_rank0 × world_size
```

因为 `correct_count_rank0` 是整数（ACCUM_METRICS 的整数累加特性），所以 **显示的正确样本数 = world_size × 整数**，也就是 **必定是 world_size 的整数倍**！

这正是姜总工口中的 "最后打印的时候又乘以 RANK 数" —— 除以 `val_samples_per_rank()`（即 `10000/world_size`）而不是 `10000`，等价于把单个 RANK 的数据放大了 `world_size` 倍。

### 3.4 为何正确的 AllReduce 不会产生此效果？

若 AllReduce 正常：`accum_top1_after = Σ(correct_i) / world_size`，这是一个**浮点值**（多个整数求和后再除以 world_size），不再是整数，因此不满足被 world_size 整除的性质。

---

## 四、数据验证

### 4.1 Best Accuracy 验证

| RANK | Best Acc | 正确样本数 | world_size | 整除验证 |
|------|----------|-----------|-----------|---------|
| 1 | 99.41% | 9941 | 1 | 9941 ÷ 1 = 9941 ✓ |
| 2 | 99.62% | 9962 | 2 | 9962 ÷ 2 = 4981 ✓ |
| 4 | 99.44% | 9944 | 4 | 9944 ÷ 4 = 2486 ✓ |
| 8 | 99.04% | 9904 | 8 | 9904 ÷ 8 = 1238 ✓ |

### 4.2 每个 epoch 逐一验证（以 RANK=8 为例）

| Epoch | Val Acc | 正确样本数 | ÷ 8 |
|-------|---------|-----------|-----|
| 1 | 10.08% | 1008 | 126 |
| 2 | 94.16% | 9416 | 1177 |
| 5 | 96.08% | 9608 | 1201 |
| 10 | 97.28% | 9728 | 1216 |
| 20 | 98.08% | 9808 | 1226 |
| 30 | 98.48% | 9848 | 1231 |
| 40 | 98.48% | 9848 | 1231 |
| 50 | 98.48% | 9848 | 1231 |
| 60 | **99.04%** | 9904 | 1238 |
| 70 | 98.80% | 9880 | 1235 |
| 80 | 98.72% | 9872 | 1234 |
| 90 | 98.96% | 9896 | 1237 |
| 100 | 98.96% | 9896 | 1237 |

**RANK=2、RANK=4 的每个 epoch 同样严格遵守。无一例外。** 这是 **数学必然**，不是巧合。

---

## 五、根因确认

### 5.1 直接根因

**`VAL_RESULT_ALLREDUCE` graph 的 NCCL AllReduce 操作完全失效。**

从代码路径来看（[deep_learning_task.cpp:1581-1658](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1581-L1658)）：

```cpp
auto vb_comm = g_vb[S(GraphSlot::VAL_RESULT_ALLREDUCE)];  // variant 4 的 VAL_RESULT_COMM graph
// ... 验证循环 ...
if (vb_comm) cudaGraphLaunch(vb_comm, s_up);  // launch 了但内部 NCCL 未生效
```

`vb_comm` 的 graph 被成功 launch，但 graph 内部的 `ncclAllReduce` 调用可能因 CUDA Graph 捕获问题而成为空操作。

### 5.2 可能的底层原因

参照之前 `NCL1.md` 的分析：`capture_nccl_graph_coordinated` 函数在 CUDA Graph 捕获期间（`BeginCapture` 与 `EndCapture` 之间）创建和销毁 CUDA Event，违反了 CUDA 编程规范。事件被过早销毁导致图节点中的事件引用失效，NCCL AllReduce 变为空操作。

[allreduce_op.cpp:40-101](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L40-L101) 中，`launch_allreduce_cuda_impl` 使用 `StreamKind::UPDATE`，在 graph capture 期间，此函数被调用时会在 capture 流上隐式创建事件。如果这些事件的创建/销毁时机违规，就会导致生成的 graph 节点中的 NCCL 通信失效。

### 5.3 为何训练不受影响？

训练时的梯度 AllReduce（`DEEP_COMM` / `FIRST_COMM`）走的是 `variant[0..3]` 的 graph。虽然这些 graph 也可能有同样的 capture 问题，但：

1. **如果训练 AllReduce 也失效**：在 8 RANK 下，每 RANK 仅用 `local_bs=16` 独立训练 100 epoch，有效只看到 12.5 epoch 的数据量（100 × 16 / 128），准确率应该远低于 99.04%。但实际显示 99.04%，说明训练路径有一定程度的通信在工作。

2. **如果训练 AllReduce 正常**：VAL_RESULT_COMM 是一个不同的 graph（`GraphId::VAL_RESULT_COMM`），其 stream 映射、捕获时机可能与训练路径不同，导致只有验证的 AllReduce 失效。

### 5.4 与小伙伴 K 发现的 variant 5 问题的关系

小伙伴 K 在 `EKR.md` 中发现的 variant 5（val_last）的 accum 写入 variant 5 的 Region 但被从 variant 4 读取的问题，**也是一个真实存在的 bug**。但姜总工发现的 AllReduce 失效问题是**更根本的原因**，因为：

- **AllReduce 失效**：解释了正确样本数是 world_size 倍数的精确数学规律
- **Variant 5 偏移**：导致 last batch 指标丢失（每个 RANK 丢 2~16 个样本，约占 0.16%）

两个问题是**正交的**，都导致验证准确率失真。

---

## 六、修复建议

### 修复 A：修复 CUDA Graph 捕获中的 Event 生命周期（P0）

在 `capture_nccl_graph_coordinated` 中：
- 将 Event 的创建移到 `cudaStreamBeginCapture` 之前
- 将 Event 的销毁移到 `cudaGraphInstantiate` 之后

确保 graph 捕获期间不创建/销毁任何 CUDA 资源。

### 修复 B：修复 Variant 5 的 accum 写入/读取不一致（P1）

确保 `ACCUM_METRICS_VAL_LAST` 的 output 写入到与 `VAL_RESULT_ALLREDUCE` 相同 memory offset 的 accum region，或验证结束后手动合并 variant 5 的指标到 variant 4。

### 验证方案

修复后：
1. 所有 RANK 的 val accuracy × 10000 不再必须是 world_size 的整数倍（即 AllReduce 真正生效）
2. RANK=1/2/4/8 的准确率应收敛到一致水平（99.4% 附近）
3. 训练 loss 和验证 loss 在跨 RANK 时保持一致

---

## 七、关键代码位置索引

| 代码 | 位置 | 作用 |
|------|------|------|
| softmax_ce INF kernel | [softmax_ce_op.cu:306](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu#L299-L306) | 计算 batch top1（整数个正确样本 / batch） |
| ACCUM_METRICS kernel | [accum_op.cu:20](file:///r:/renaissance/src/backend/ops/range/accum_op.cu#L20) | 累加 batch 正确样本数（整数） |
| VAL_RESULT_ALLREDUCE | [allreduce_op.cpp:77-92](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L77-L92) | NCCL AllReduce(sum) + 1/world_size scale |
| 验证结果读取 | [deep_learning_task.cpp:1673-1696](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1673-L1696) | fetch_from_rank(0) + 除以 val_samples_per_rank() |
| VAL_RESULT_COMM launch | [deep_learning_task.cpp:1658](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1658) | `cudaGraphLaunch(vb_comm, s_up)` |
| NCCL graph capture | [captured_graph.cpp:460-560](file:///r:/renaissance/src/graph/captured_graph.cpp#L460-L560) | `capture_nccl_graph_coordinated` — Event 生命周期违规 |
| Variant 5 accum 偏移 | [deep_learning_task.cpp:1581-1594](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1581-L1594) | `vl_accum` 写 variant 5，`vb_comm` 读 variant 4 |

---

*报告生成时间: 2026-06-01*
*关键贡献者: 姜总工 — VAL_RESULT_COMM 失效假说与正确样本数整除性证明*