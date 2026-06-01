# NCCL 协调多 RANK CUDA Graph 捕获审查报告

> 审查范围：`git diff` 涉及的 8 个文件（`captured_graph.cpp` 为主）
> 审查目标：确认小伙伴将 `compile_capture_simple` 的协调捕获机制移植到 `pre_capture` 的正确性
> 测试状态：多 RANK 已跑通（不卡死），但准确率不高

---

## 一、小伙伴修改的核心正确性 ✅

### 1.1 NCCL 协调捕获的结构完全正确

`capture_nccl_graph_coordinated()` 遵循了 NCCL + CUDA Graph 的铁律：

```
Phase 1: 所有 RANK 同时 cudaStreamBeginCapture
Phase 2: ncclGroupStart → 逐个 RANK replay → ncclGroupEnd
Phase 3a: 所有 RANK 同时 cudaStreamEndCapture
Phase 3b: 所有 RANK cudaGraphInstantiate
```

这与 `task_base.cpp` 中 `compile_capture_simple`（已通过 `test_mean_allreduce` 验证）的模板完全一致。多 RANK 卡死问题的根因（各 RANK 独立 begin/end capture）已被正确修复。

### 1.2 NCCL / 非 NCCL 路径分离正确

- `has_nccl_ops()` 检测包含 `RANGE_SUM_ALLREDUCE`、`RANGE_MEAN_ALLREDUCE`、`RANGE_BN_STATS_ALLREDUCE`
- 非 NCCL graph 保持原有 `capture_all_for_rank` 独立 capture 路径不变
- warmup 阶段正确跳过 NCCL graph（只跑 rank 0 时 launch NCCL graph 会死锁）

### 1.3 AKT_FINAL 修改全部正确

- `RANGE_MEAN_ALLREDUCE` 替换（3 处）✅
- `ACCUM_METRICS` 顺序调整 ✅
- 除数改为 `*_samples_per_rank()` ✅
- `std::round` 移除 ✅
- `global_batch_size()` API 正确 ✅

---

## 二、`capture_nccl_graph_coordinated` 的工程隐患

### 🔴 P1：缺少异常安全机制（最严重）

**问题**：如果 `ncclGroupStart/End` 之间发生异常（kernel launch 失败、`TR_DEVICE_ERROR` 触发、内存不足等），**所有 RANK 的 stream 会永远卡在 capture 状态**。后续任何 CUDA 操作都会报 `cudaErrorStreamCaptureImplicit`，整个进程崩溃且无法恢复。

**对比原版**：`capture_cuda.cpp` 中有 `CaptureGuard` RAII：
```cpp
struct CaptureGuard {
    cudaStream_t stream = nullptr;
    bool committed = false;
    ~CaptureGuard() {
        if (!committed && stream) {
            cudaGraph_t dummy = nullptr;
            cudaStreamEndCapture(stream, &dummy);
            if (dummy) cudaGraphDestroy(dummy);
        }
    }
};
```

**多 RANK 下更危险**：8 张卡同时 begin capture，如果 rank 3 在 replay 时抛出异常，其他 7 个 rank 的 stream 也不会被 end capture。这不是单 rank 自己能恢复的。

**建议修复**：
```cpp
// Phase 1: BeginCapture 之后
bool capture_committed = false;
std::vector<cudaGraph_t> emergency_graphs(num_ranks, nullptr);

try {
    ncclGroupStart();
    for (int r = 0; r < num_ranks; ++r) { /* replay */ }
    ncclGroupEnd();
    capture_committed = true;
} catch (...) {
    for (int r = 0; r < num_ranks; ++r) {
        cudaSetDevice(contexts[r]->device_id());
        cudaStreamEndCapture(cap_streams[r], &emergency_graphs[r]);
        if (emergency_graphs[r]) cudaGraphDestroy(emergency_graphs[r]);
    }
    throw;
}
```

### 🟡 P2：`cudaStreamBeginCapture` 失败不 abort

```cpp
cudaError_t cap_err = cudaStreamBeginCapture(cap_streams[r], ...);
if (cap_err != cudaSuccess) {
    TR_DEVICE_ERROR(...);  // 只是 log，不 throw！
}
```

`TR_DEVICE_ERROR` 宏定义为 `TR_THROW(DeviceError, ...)`，确实会 throw。但代码在 throw 之后没有任何 cleanup，结合 P1 的问题，失败的 rank 和其他成功的 rank 会处于不一致状态。

**建议**：在 Phase 1 的循环中加入 rollback 逻辑，如果某个 rank 的 BeginCapture 失败，已经 begin 的 rank 需要被 end capture。

### 🟡 P3：缺少 `cudaGetLastError` 诊断

`capture_cuda.cpp` 在**每个 node 之后**和 **finalize barrier 之后**都检查了 `cudaGetLastError()`：
```cpp
cudaError_t node_err = cudaGetLastError();
if (node_err != cudaSuccess) { TR_DEVICE_ERROR(...); }
```

小伙伴的代码缺少这些检查。如果某个 kernel 在 capture 期间静默失败（如 `cudaErrorLaunchOutOfResources`），问题会延迟到 `EndCapture` 或 `Instantiate` 才暴露，错误信息模糊（如 "operation failed"），极难定位。

### 🟢 关于 `cleanup_all_events` 的位置（不是 bug）

`state.cleanup_all_events()` 在 `ncclGroupEnd()` 之后、`cudaStreamEndCapture` 之前调用。**这是安全的**，因为 `task_base.cpp` 的 `compile_capture_simple`（已验证正确）也是这个顺序。NCCL group capture 的 event 节点在 graph 中引用的是 event handle 的"快照"，destroy 原始 event 后 graph 节点仍然有效。

---

## 三、准确率不高的根因分析

用户反馈：多 RANK "不卡死了，也能跑出结果，但准确率不高"。通信已通，但数学不正确。

### 🔴 最直接原因：`RANGE_BN_STATS_ALLREDUCE` 未注册

`allreduce_op.cpp` 的 `register_op_range_allreduce()` 只注册了：
- `RANGE_SUM_ALLREDUCE`
- `RANGE_MEAN_ALLREDUCE`

**没有注册 `RANGE_BN_STATS_ALLREDUCE`**。

`compiler.cpp:1566` 生成了 `STATS_COMM` graph：
```cpp
train_cg.append_range(GraphId::STATS_COMM,
    RangeOp::RANGE_BN_STATS_ALLREDUCE, {r_bn}, {r_bn});
```

当 `has_nccl_ops()` 检测到 `RANGE_BN_STATS_ALLREDUCE` 时，`STATS_COMM` 被标记为 NCCL graph，走 `capture_nccl_graph_coordinated`。但在 replay 时：
```cpp
auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
if (entry.launch_cuda) {
    entry.launch_cuda(node, *mp, dc, state);
} else {
    TR_DEVICE_ERROR("Unimplemented RangeOp in NCCL graph: " ...);
}
```

由于 `entry.launch_cuda == nullptr`，会调用 `TR_DEVICE_ERROR` 抛出异常。

**对于当前 MNIST MLP 测试**：`mnist_best.cpp` 使用的网络是纯 MLP（`fc` + activation），没有 BN 层。`compiler.cpp` 中 `has_bn=false`，`STATS_COMM` graph 不会被创建。因此 **BN_STATS 未注册不是当前准确率问题的根因**。

**但如果测试网络有 BN（如 ResNet、DenseNet）**，BN 的 running mean/variance 只在本地 batch 上统计，不进行跨 RANK 同步。这会导致：
- 训练时：各 RANK 的 BN 统计量偏离全局统计量，梯度估计有偏
- 验证时：rank 0 的 BN 统计量只反映本地数据分布，inference 不稳定
- **最终表现：能收敛，但准确率显著低于单 RANK 基线**

**建议修复**：在 `register_op_range_allreduce()` 中为 `RANGE_BN_STATS_ALLREDUCE` 注册处理函数，复用 `launch_allreduce_cuda_impl`（使用 MEAN-all-reduce 语义）：
```cpp
{
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_BN_STATS_ALLREDUCE)];
    entry.op = RangeOp::RANGE_BN_STATS_ALLREDUCE;
    entry.launch_cpu = launch_allreduce_cpu_impl;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_allreduce_cuda_impl;
#endif
}
```

### 🟡 次可能原因：`global_batch_size(128)` 导致 `local_batch_size` 过小

小伙伴将 `mnist_best.cpp` 中的 `local_batch_size(128)` 改为了 `global_batch_size(128)`。

- **单 RANK**：`world_size=1` → `local_batch_size = 128 / 1 = 128`，和原来一致
- **多 RANK（8卡）**：`world_size=8` → `local_batch_size = 128 / 8 = 16`

**有效 batch size 从 1024 降到了 128**（原来是 8 卡 × 128 = 1024）。更重要的是，每个 RANK 的 `local_batch_size` 从 128 骤降到 16。

AdamW 的 `base_lr=0.001f` 是针对较大 batch size（128+）调的经验值。当 `local_batch_size=16` 时：
- 梯度噪声增大 8 倍
- 有效学习率相对放大（如果按 linear scaling rule，lr 应降至 ~0.000125）
- 训练可能不稳定，收敛到次优解

**建议验证**：
1. 单 RANK 但显式设置 `local_batch_size(16)`，lr 保持 0.001，看准确率是否也下降
2. 如果下降，说明是 lr/batch_size 不匹配问题，需按 local_batch_size 调整 lr
3. 或者将 `global_batch_size(128)` 改回 `global_batch_size(1024)`，使 `local_batch_size=128`

### 🟡 第三可能原因：MEAN_ALLREDUCE 的 scale kernel 在 capture 期间的正确性

`launch_tr_scale_fp32_kernel` 是 `extern "C"` 的 CUDA kernel launcher。在 graph capture 期间调用它会生成 kernel 节点。但如果该 kernel 使用了某些 CUDA 特性（如动态并行、CUDA Graph 不支持的 API），capture 可能失败或生成错误的 graph。

**验证方法**：单 RANK 和多 RANK 的 loss 曲线对比。如果多 RANK loss 震荡剧烈或收敛到更高值，说明梯度缩放有问题。

---

## 四、修改建议清单

| 优先级 | 文件 | 修改内容 | 影响 |
|--------|------|----------|------|
| P0 | `allreduce_op.cpp` | 注册 `RANGE_BN_STATS_ALLREDUCE` | 修复 BN 统计量同步，直接解决准确率问题 |
| P1 | `captured_graph.cpp` | 为 `capture_nccl_graph_coordinated` 添加异常安全机制（try/catch + 强制 EndCapture） | 防止异常后 stream 卡死 |
| P1 | `captured_graph.cpp` | Phase 1 中 `BeginCapture` 失败时 rollback 已成功的 rank | 避免 RANK 间状态不一致 |
| P2 | `captured_graph.cpp` | Phase 2 中加入 `cudaGetLastError` 检查 | 提升诊断能力 |
| P2 | `captured_graph.cpp` | Phase 1 中增加 `world_size == num_ranks` 断言 | 防御配置不一致 |
| P3 | `captured_graph.cpp` | `cudaStreamEndCapture` 失败时 destroy graph | 避免内存泄漏 |

---

## 五、验证计划

1. **确认测试网络是否有 BN**：检查 `has_bn` 的值。如果没有 BN，`RANGE_BN_STATS_ALLREDUCE` 未注册不会触发，需要排查其他原因。
2. **单 RANK vs 多 RANK loss 曲线对比**：如果多 RANK loss 明显偏离单 RANK，说明梯度 allreduce 或 scale 有问题。
3. **多 RANK 梯度数值检查**：在 `launch_allreduce_cuda_impl` 中打印梯度 sum 的前后值，确认 `ncclAllReduce` 和 scale 都生效。
4. **注册 BN_STATS 后重新测试**：如果网络有 BN，注册后准确率应恢复到单 RANK 水平。

---

## 六、结论

小伙伴的修改在 **NCCL 协调捕获的核心逻辑上是正确的**，成功解决了多 RANK 卡死问题。但存在以下需要修复的问题：

1. **准确率不高的最可能原因**：`RANGE_BN_STATS_ALLREDUCE` 未注册（如果网络有 BN）
2. **工程隐患**：`capture_nccl_graph_coordinated` 缺少异常安全机制和错误诊断
3. **防御性改进**：`world_size` 一致性断言、graph 内存管理

建议优先修复 P0（BN_STATS 注册）和 P1（异常安全），然后重新进行多 RANK 准确率测试。
