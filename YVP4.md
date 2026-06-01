# 多RANK训练准确率异常综合分析报告 (YVP4.md)

**日期**: 2026-06-01  
**测试平台**: A100×8  
**分析依据**: 1RANK=99.41%, 2RANK=99.62%, 4RANK=99.44%, 8RANK=99.04%  
**参考文档**: WNF1.md, WNF2.md, WNF3.md, NCL1-3.md, AKT_FINAL.md  
**代码检查**: captured_graph.cpp, allreduce_op.cpp, deep_learning_task.cpp, compiler.cpp

---

## 一、问题现象总结

### 1.1 准确率异常数据

| RANK数 | 最佳准确率 | 最佳轮次 | 与基准差异 | 异常评估 |
|--------|-----------|----------|------------|----------|
| 1RANK | 99.41% | 73 | 基准 | 正常 |
| 2RANK | 99.62% | 87 | +0.21% | **异常偏高** |
| 4RANK | 99.44% | 88 | +0.03% | 正常偏高 |
| 8RANK | 99.04% | 60 | -0.37% | **异常偏低** |

### 1.2 训练曲线特征

- **训练loss**: 所有RANK数的训练loss曲线几乎完全重合（epoch 2均为~1.0857）
- **验证loss**: 8RANK验证loss明显偏高（8R=0.526 vs 1R=0.516）
- **收敛速度**: 8RANK在60轮达到最佳后开始下降，2RANK和4RANK持续改善到87-88轮

### 1.3 核心异常

1. **2RANK异常偏高**: 达到99.62%，超过MLP理论极限
2. **8RANK异常偏低**: 比基准低0.37%，且提前过拟合
3. **梯度同步失效**: 训练loss相似但验证准确率差异巨大

---

## 二、根因分析

### 🔴 P0-1: `has_nccl_ops()` 检查整图而非子图

**位置**: `include/renaissance/graph/computation_graph.h:310-326`  
**严重程度**: 致命 (P0)

**问题描述**:
```cpp
bool has_nccl_ops() const {
    // 检查整个ComputationGraph的所有节点
    for (const auto& node : linear_nodes_) { /* ... */ }
    for (const auto& bucket : graphs_) {
        for (const auto& node : bucket) { /* ... */ }
    }
}
```

**问题**: 只要`train_cg_`中有一个子图包含NCCL节点（如`DEEP_COMM`），则所有子图都被标记为NCCL图。

**后果**:
- 28个unique graph中，实际只有3-4个包含NCCL节点
- 但10+个graph被误标记为NCCL图
- 所有被误标记的graph都走`capture_nccl_graph_coordinated`路径
- 导致大量不必要的`ncclGroupStart/End`调用

**数学证据**:
```
正常情况: 3-4个NCCL graph × 1次ncclGroup调用 = 3-4次调用
实际情况: 10+个graph × 1次ncclGroup调用 = 10+次调用
多余调用: 6-7次空ncclGroup调用
```

---

### 🔴 P0-2: `capture_nccl_graph_coordinated` 缺少 `cudaDeviceSynchronize()`

**位置**: `src/graph/captured_graph.cpp:483-495`  
**严重程度**: 致命 (P0)

**问题描述**:
`capture_nccl_graph_coordinated`直接调用`cudaStreamBeginCapture`，没有任何同步：
```cpp
// Phase 1: BeginCapture on ALL ranks
for (int r = 0; r < num_ranks; ++r) {
    cudaStreamBeginCapture(cap_streams[r], ...);  // ← 直接开始，没有sync！
}
```

**对比正常路径** (`src/graph/capture_cuda.cpp:61-62`):
```cpp
cudaDeviceSynchronize();  // ← 清除pending work
// 然后才开始capture
cudaStreamBeginCapture(primary_stream, ...);
```

**问题根因**:
1. Phase B2的cuDNN warmup在每个rank的stream上执行了大量kernels
2. 如果没有`cudaDeviceSynchronize()`，这些pending work会被`BeginCapture`捕获为external dependencies
3. CUDA Graph的external dependency机制在8RANK下处理混乱，导致后续NCCL节点被静默跳过

**为什么2RANK正常、8RANK异常**:
- 2RANK: pending work少，事件池小，external dependency处理正常
- 8RANK: pending work多，事件池大，external dependency处理混乱

---

### 🔴 P0-3: NCCL communicator状态被空group调用污染

**严重程度**: 致命 (P0)

**问题描述**:
由于P0-1，10+个graph被误标记为NCCL图，每个都会调用：
```cpp
ncclGroupStart();
// 对于ZERO_GRAD等不含NCCL的graph，这里没有任何ncclAllReduce调用
ncclGroupEnd();
```

**后果**:
1. 大量空`ncclGroupStart/End`调用累积污染NCCL communicator状态
2. 当真正包含NCCL节点的graph被捕获时，communicator状态已异常
3. Graph能成功实例化，但运行时`ncclAllReduce`节点变为no-op

**完美解释实验现象**:
- **不卡死**: Graph成功实例化并launch
- **不报错**: NCCL节点静默跳过，无错误日志
- **通信不生效**: 梯度同步失效，各RANK独立训练

---

### 🟡 P1-1: Event生命周期违规

**位置**: `src/graph/captured_graph.cpp:510-515,559`  
**严重程度**: 高危 (P1)

**问题描述**:
```cpp
// Phase 2中（在BeginCapture之后）
cudaEventCreateWithFlags(&state.streams[i].last_done_event, ...);  // ← 创建事件

// Phase 2结束前
state.cleanup_all_events();  // ← 销毁事件

// Phase 3a
cudaStreamEndCapture(cap_streams[r], &captured_graphs[r]);  // ← graph节点还引用着已销毁的事件
```

**对比**: `test_mean_allreduce`使用相同的event管理方式但通过了测试，说明这个问题**不是独立根因**，但会加剧P0-2的影响。

---

### 🟡 P1-2: 验证指标同步失效

**位置**: `src/task/deep_learning_task.cpp:1662-1797`  
**严重程度**: 高危 (P1)

**问题描述**:
```cpp
// 验证阶段只从rank 0读取结果
Tensor h_al = fetch_from_rank(al_dt, 0);  // ← 只读rank 0
float accum_loss = h_al.data<float>()[0];
avg_loss = accum_loss / static_cast<float>(n);
```

**如果`VAL_RESULT_COMM`失效**: 只反映rank 0的本地模型性能，无法解释为何rank 0性能随RANK数变化。

**真正解释**: 梯度同步失效 → 各RANK模型diverge → rank 0模型质量随有效batch size减小而下降。

---

### 🟢 P2-1: `RANGE_BN_STATS_ALLREDUCE`未注册

**位置**: `src/backend/ops/range/allreduce_op.cpp:153-177`  
**严重程度**: 中危 (P2)

**问题**: BN统计量同步未注册，对当前MLP测试无影响，但对含BN的网络会造成准确率下降。

---

### 🟢 P2-2: `local_batch_size`过小

**问题描述**: `global_batch_size(128)`在8RANK下导致`local_batch_size=16`。

**影响**: 如果通信修复，此问题会成为次要瓶颈。学习率相对小batch size过大，需要调整。

---

## 三、问题因果关系链

```
has_nccl_ops()检查整图 (P0-1)
    │
    ▼
10+个graph被误标记为NCCL → 全部走capture_nccl_graph_coordinated
    │
    ▼
大量空ncclGroupStart/End调用 → communicator状态污染 (P0-3)
    │
    ▼
capture_nccl_graph_coordinated缺少cudaDeviceSynchronize() (P0-2)
    │
    ▼
BeginCapture捕获warmup的pending work为external dependencies
    │
    ▼
8RANK时事件池大 + pending work多 → external dependency处理混乱
    │
    ▼
NCCL AllReduce节点在运行时被静默跳过 (no-op)
    │
    ├─→ 梯度不同步 → 各RANK独立训练
    │       │
    │       └─→ 2RANK: 有效batch=64, 梯度噪声起正则化作用 → 99.62%
    │       └─→ 4RANK: 有效batch=32, 接近正常 → 99.44%
    │       └─→ 8RANK: 有效batch=16, 噪声过大+数据量少 → 99.04%
    │
    └─→ VAL_RESULT_COMM失效 → 只反映rank 0本地性能 (P1-2)
            │
            └─→ 验证准确率差异进一步放大
```

---

## 四、修复方案

### 4.1 P0-1修复: 修正`has_nccl_ops()`为子图级别检查

**文件**: `src/graph/captured_graph.cpp:284-289`

**修改前**:
```cpp
for (int32_t k = 0; k < K; ++k) {
    if (key_by_idx[k].cg && key_by_idx[k].cg->has_nccl_ops()) {
        is_nccl_key[k] = true;  // ← 检查整图
    }
}
```

**修改后**:
```cpp
for (int32_t k = 0; k < K; ++k) {
    const auto& key = key_by_idx[k];
    if (!key.cg) continue;
    
    // 只检查当前gid的子图
    bool has_nccl = false;
    for (const auto& node : key.cg->nodes(key.gid)) {
        if (node.kind == GraphNode::Kind::RANGE &&
            (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
             node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE ||
             node.range_op == RangeOp::RANGE_BN_STATS_ALLREDUCE)) {
            has_nccl = true;
            break;
        }
    }
    is_nccl_key[k] = has_nccl;
}
```

**预期效果**: `[B3-NCCL]`日志从~10条减少到~3-4条

---

### 4.2 P0-2修复: 添加`cudaDeviceSynchronize()`

**文件**: `src/graph/captured_graph.cpp:483`

**修改前**:
```cpp
// Phase 1: BeginCapture on ALL ranks
for (int r = 0; r < num_ranks; ++r) {
    cudaStreamBeginCapture(cap_streams[r], ...);
}
```

**修改后**:
```cpp
// Phase 1: BeginCapture on ALL ranks
for (int r = 0; r < num_ranks; ++r) {
    DeviceContext& dc = *contexts[r];
    cudaSetDevice(dc.device_id());
    dc.set_rank(r);
    dc.set_memory_plan(mp);
    cap_streams[r] = static_cast<cudaStream_t>(dc.stream(stream_kind));
    
    // ← 添加同步，清除warmup的pending work
    cudaDeviceSynchronize();
    
    cudaError_t cap_err = cudaStreamBeginCapture(cap_streams[r], cudaStreamCaptureModeThreadLocal);
    // ...
}
```

---

### 4.3 P0-3修复: 移除空group调用（通过P0-1实现）

P0-1修复后，只有真正包含NCCL的graph才会调用`capture_nccl_graph_coordinated`，自动消除空group调用问题。

---

### 4.4 P1-1修复: 修正Event生命周期

**文件**: `src/graph/captured_graph.cpp:456-597`

**修改策略**: 将event创建移到`BeginCapture`之前，event销毁移到`Instantiate`之后。

**Step 1**: 在Phase 1之前预创建所有events
**Step 2**: Phase 2使用预创建的events，不再创建/销毁  
**Step 3**: 在Phase 3b之后统一销毁events

---

### 4.5 P1-2修复: 验证通信正确性检查

**文件**: `src/task/deep_learning_task.cpp`

添加验证逻辑，确认`VAL_RESULT_COMM`是否成功：
```cpp
// 在vb_comm launch后添加检查
cudaStreamSynchronize(s_up);
float check_sum = 0.0f;
for (int rank = 0; rank < world_size; ++rank) {
    Tensor h_check = fetch_from_rank(al_dt, rank);
    check_sum += h_check.data<float>()[0];
}
float expected = check_sum / world_size;
if (std::abs(check_sum - expected) > 0.01f) {
    TR_LOG_WARN("task") << "VAL_RESULT_COMM may have failed";
}
```

---

### 4.6 P2-1修复: 注册`RANGE_BN_STATS_ALLREDUCE`

**文件**: `src/backend/ops/range/allreduce_op.cpp`

添加注册：
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

---

## 五、验证计划

### 阶段1: P0修复验证

1. **实施P0-1, P0-2修复**
2. **检查日志**: `[B3-NCCL]`条目应从~10减少到~3-4
3. **单元测试**: `test_mean_allreduce`应继续通过
4. **多RANK测试**: 重新运行1/2/4/8RANK测试

**预期结果**: 
- 8RANK准确率恢复到99.4%以上
- 各RANK数准确率收敛到99.4%-99.5%区间
- 2RANK不再异常偏高

### 阶段2: P1修复验证

1. **实施P1-1, P1-2修复**
2. **添加验证通信检查**
3. **长期稳定性测试**: 运行完整100轮训练

**预期结果**:
- 验证指标同步正常
- 各RANK验证准确率一致

### 阶段3: P2修复验证

1. **实施P2-1修复**
2. **测试含BN网络**: 确认BN统计量同步正常

---

## 六、结论

**主要根因**: `has_nccl_ops()`检查整图 + 缺少`cudaDeviceSynchronize()` + 空ncclGroup调用污染，三重因素导致NCCL通信在8RANK下静默失效。

**修复优先级**:
1. 🔴 **P0-1**: 修正`has_nccl_ops()` - 立即修复
2. 🔴 **P0-2**: 添加`cudaDeviceSynchronize()` - 立即修复  
3. 🔴 **P0-3**: 空group调用自动消除 - 通过P0-1实现
4. 🟡 **P1-1**: Event生命周期 - 高优先级
5. 🟡 **P1-2**: 验证通信检查 - 高优先级
6. 🟢 **P2-1**: BN_STATS注册 - 中优先级

**预期效果**: 修复P0-1和P0-2后，8RANK准确率应恢复到99.4%以上，与单RANK基准一致。2RANK的异常虚高现象应消失。

**技术保证**: P0修复确保NCCL通信在多RANK下正常工作，P1修复提供工程健壮性，P2修复为未来BN网络测试做准备。
