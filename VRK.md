# 【小伙伴D】

## CPU 路径失败根因分析报告
### 1. 问题现象
模式 Epoch 0 Epoch 1 Epoch 2 Best GPU FP32 95.89% 96.93% 97.33% 97.33% PASS GPU AMP 95.89% 96.82% 97.33% 97.33% PASS CPU FP32 58.96% 49.46% 37.46% 58.96% FAIL

CPU 准确率不仅远低于 GPU，而且 逐 epoch 下降 ，说明模型权重在训练过程中不断被污染。

### 2. 根因定位 2.1 核心矛盾：GPU Transfer 用 capture-time 数据，CPU Transfer 用 runtime 数据
GPU Transfer dtensor_copy_op.cpp:47-75 ：

CPU Transfer dtensor_copy_op.cpp:25-41 ：

CPU 版本通过 dev.memory_plan() 在运行时查当前 MemoryPlan，而不是用 capture 时烘焙的值！
 2.2 ctx.memory_plan_ 在 capture 后指向谁？
追踪调用链：

1. task_base.cpp:246-248 ：capture 前设为 active_memory_plan_ （variant 0）
2. captured_graph.cpp:243 ：B2 warmup 阶段对每个 capture key 调用：
3. captured_graph.cpp:272 ：B3 capture 阶段再次迭代所有 variant：
内部对每个不同的 ShapeId 调用 ctx.set_memory_plan(mp) 。

4. B3 结束后， ctx.memory_plan_ 保持为 最后一个 capture key 的 MemoryPlan 。
5. 由于 variant 1 (batch=101 的 last batch) 的 ShapeId{101,28,28,1} 与 variant 0 的 ShapeId{199,28,28,1} 不同，它们产生不同的 capture keys。按遍历顺序，variant 1 的 key 排在 variant 0 之后 ，所以 capture 结束后 ctx.memory_plan_ = variant 1 的 MemoryPlan。
6. task_base.cpp:251 后： 没有任何代码恢复 ctx.set_memory_plan(active_memory_plan_) ！ 2.3 后果
CPU 训练循环在 run_train_epoch_cpu 中：

Transfer graph（shape-invariant，仅有 1 个 captured graph，用 active_memory_plan_ capture）被启动：

正确值应为 199×4 = 796 bytes。

对于 I_A_DATA（图像数据），损失更严重：

- 正确： 199 × 28 × 28 × 4 = 624,064 bytes
- 实际： 101 × 28 × 28 × 4 = 316,864 bytes
- 仅有 50.8% 的数据被正确拷贝！
DTensor 应拷贝 实拷贝 缺失率 I_A_LABEL 796B 404B 49.2% I_A_DATA 624,064B 316,864B 49.2%

约 49% 的样本数据是未初始化的 Arena 内存垃圾！ 模型在这些垃圾数据上"学习"，权重被持续污染，准确率必然下降。
 2.4 为什么 GPU 不受影响？
GPU Transfer 函数签名是 const MemoryPlan& mp （ 值传递 ），capture 时传入的是 variant 0 的 MemoryPlan。CUDA Graph 将 cudaMemcpyAsync(dst, src, nbytes, ...) 的 nbytes 参数硬编码在图中，replay 时不会重新查询 DeviceContext 的 memory_plan_ 。
 2.5 为什么 VZT1 无法修复此问题？
VZT1 保证了所有 DTensor 的 offset 跨 variant 一致，所以 ptr_at(id) 能返回正确的物理地址。但 VZT1 不保证 shape 一致——variant 0 的 shape 是 {199,28,28,1} ，variant 1 的是 {101,28,28,1} 。

CPU Transfer 用 dev.memory_plan() 在运行时查 DTensor → 拿到 variant 1 的 DTensor → 调用 nbytes() → shape.numel() * dsize → 用 101 算字节数。这个流程不受 VZT1 影响，因为 VZT1 只统一了 slot_bytes，不改变 shape。

### 3. 修改方案
Root cause : ctx.memory_plan_ 在 pre_capture() 后指向最后一个 capture 的 variant（variant 1），但运行时 CPU kernel 通过 dev.memory_plan() 查询它来获取元数据。

Fix ：在 pre_capture() 返回后、训练开始前， 恢复所有 context 的 MemoryPlan 为 active_memory_plan_ （即 variant 0） 。
 修改位置： src/task/task_base.cpp
在 task_base.cpp:251 之后添加：
 为什么只需要 1 行代码（3 行含循环）？
1. Transfer 图 （shape-invariant）只有 1 个 captured graph，使用时需要 variant 0 的 Metadata（nbytes = max）。恢复后满足。
2. Compute 图 （shape-dependent）的 CPU kernel 使用 op_ctx->input_shape.n （capture 时烘焙的 batch 大小）， 不依赖 dev.memory_plan() 获取形状。恢复不影响。
3. Optimizer CPU kernel 通过 dev.memory_plan() 查 has_nan offset，但 VZT1 已保证 offset 跨 variant 一致。恢复不影响。
4. GPU 路径 ： build_exec_table() 已预解析了 CUDA graph handles，CUDA graph replay 不依赖 ctx.memory_plan_ 。恢复不影响 GPU。 为什么之前 batch_size=200 时 CPU 正常？
batch_size=200 时没有 last batch variant。所有 variant 的 shape 完全相同（ {200,28,28,1} ）， capture_all_for_rank 遍历的 capture keys 都共享同一 ShapeId → 去重后只有一组 captured graph → ctx.memory_plan_ 始终是 variant 0。

### 4. 总结
维度 GPU CPU（当前） CPU（修复后） Transfer nbytes capture 时硬编码 ✓ runtime 查 variant 1 ✗ runtime 查 variant 0 ✓ ctx.memory_plan_ 来源 不依赖 capture_all_for_rank 残留 set_memory_plan(active_mp) 受影响范围 无 所有 dev.memory_plan() 调用者 无

修改量 ：1 个文件（ task_base.cpp ），新增 3 行代码。