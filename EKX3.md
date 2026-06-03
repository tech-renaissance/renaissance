# EKX3: MaxPool 首层可用性问题 — 综合诊断与修改意见

> 基于 HFE1/HFE2/HFE3 及代码深度复查，对 `mnist_best_maxpool` 失败根因进行交叉验证，并补充 HFE 文档未覆盖的关联问题。

---

## 一、现象确认（与 HFE 一致）

| 模式 | 现象 | 根因归属 |
|------|------|----------|
| **GPU** | `compile()` 阶段崩溃（`0xC0000005` 越界） | MaxPool FWD/BWD 首层 `input_ids` 为空 → `node.input_ids[0]` 越界访问 |
| **CPU** | 准确率锁定 **11.35%**（随机猜测） | MaxPool FWD `input_shape` 全零 → kernel 零次循环 → 输出全零 → 模型不学习 |
| **AMP** | 即使修复首层绑定，仍会在 FC 层断言失败 | `fc_op.cpp` 要求 `H==1 && W==1`，但 MaxPool 输出为 `[14,14,1]` |

---

## 二、问题清单（逐项验证与补充）

### P0-1: 首层 MaxPool FWD 未绑定数据输入 ✅（HFE 已发现，确认正确）

**位置**: `src/graph/compiler.cpp` Phase 1（第 1076–1215 行）

**机制**:
1. `build_maxpool_forward` 未设置 `input_indices` → 为空
2. Compiler 通用逻辑 `if (prev_output_id >= 0)` 对首层失效（`prev_output_id = -1`）
3. 仅有 **Flatten** 享有首层兜底（硬编码 `I_A_DATA` / `I_B_DATA`）
4. MaxPool 无兜底 → `gn.input_ids = {}`

**代码证据**:
```cpp
// layer_descriptor_registry.cpp:388-396
SubgraphPattern build_maxpool_forward(...) {
    SubgraphPattern::Node n;
    n.output_indices = {0, 1};
    // ❌ n.input_indices 未设置
}

// compiler.cpp:1168-1170
if (prev_output_id >= 0) {
    gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
}
// 首层 prev_output_id = -1，跳过
```

**验证结论**: HFE 分析完全正确。

---

### P0-2: 首层 MaxPool BWD 未绑定 X 与 dX ✅（HFE 已发现，确认正确）

**位置**: `src/graph/compiler.cpp` Phase 2（第 1226–1355 行）

**机制**:
1. `layer_input_ids[0] = prev_output_id = -1`
2. MaxPool BWD 特殊处理块（第 1298–1304 行）条件 `it->second >= 0` 不满足
3. BWD `gn.input_ids` 最终为 `{dY, Y, mask}`（缺 X）
4. BWD `gn.output_ids` 最终为 `{}`（缺 dX）

**与 Flatten BWD 对比**:
```cpp
// compiler.cpp:1313-1324
if (layer.is_first_layer) {
    if (gn.compute_op == ComputeOp::FLATTEN_..._BWD) {
        gn.output_ids = {1};   // I_A_DATA
        train_cg.append(FIRST_LAYER_BWD_A, gn);
        gn.output_ids = {3};   // I_B_DATA
        train_cg.append(FIRST_LAYER_BWD_B, gn);
        continue;  // 跳过后续通用处理
    }
}
```

MaxPool BWD **缺少**等效的首层 A/B 双缓冲处理。

**补充发现**: Phase 2 中 `backward_graph_id` 对首层固定为 `FIRST_LAYER_BWD_A`，但 Flatten 通过显式 `append(FIRST_LAYER_BWD_B)` 补充了 B 桶。MaxPool 若仅修复 `layer_input_ids`，只会 append 到 A 桶，**B 桶仍缺失**。

---

### P1: 首层输入 ID 硬编码风险 ⚠️（HFE 未提及）

HFE1 建议修复代码中直接写 `gn.input_ids = {1}` 和 `layer_input_ids[l] = 1`。

**问题**: `1` 是 `I_A_DATA` 的 DTensor ID 的**经验值**，而非**契约值**。

**实际分配顺序**（`memory_plan.cpp:359-362`）:
```cpp
auto la = alloc_impl(label_shape, DType::INT32, Region::I_A_LABEL);  // id=0
auto da = alloc_impl(data_shape, input_dtype, Region::I_A_DATA);     // id=1
auto lb = alloc_impl(label_shape, DType::INT32, Region::I_B_LABEL);  // id=2
auto db = alloc_impl(data_shape, input_dtype, Region::I_B_DATA);     // id=3
```

若 `alloc_baseline_dtensors` 调用时机变化（如用户先 `alloc` 了其他张量再创建 `DeepLearningTask`），ID 会漂移。

**正确做法**: 使用 `memory_plan.baseline().data_a` 和 `memory_plan.baseline().data_b`。

---

### P2: FC AMP 的 `H==1 && W==1` 断言 —— 网络拓扑问题 ⚠️（HFE2 已发现）

**位置**: `src/backend/ops/dtensor/fc_op.cpp`

**现象**:
- MaxPool 输出 shape: `[N, 14, 14, 1]`
- FC 层（尤其是 AMP 路径）要求输入 `H==1 && W==1`

**结论**: 这不是 MaxPool 算子本身的 bug，而是 `mnist_best_maxpool.cpp` 的网络设计问题。即使修复了首层绑定，不加 `flatten()` 的话 AMP 模式仍会崩溃。

**建议**: 在 BluePrint 中 MaxPool 后显式插入 `flatten()`，或修改 compiler 的 `step5_normalize_flatten()` 使其在检测到 `H*W > 1` 时自动插入 Flatten。

---

### P3: MaxPool GPU BWD 的 Y 被前层 in-place 覆写风险 ⚠️（HFE 未提及）

**位置**: `src/backend/ops/dtensor/maxpool_op.cpp` GPU BWD launcher

**机制**:
- MaxPool → FC 的拓扑中，FC BWD 的 dX 是 **in-place** 到 FC 输入（即 MaxPool 输出 Y）
- FC BWD 先执行，覆写了 Y 内存区域为 dY（MaxPool 的梯度）
- MaxPool BWD 后执行，需要读取 Y（前向输出）作为 `cudnnPoolingBackward` 的 `y` 参数

```cpp
// maxpool_op.cpp:549-556
cudnnPoolingBackward(
    handle, cache.pool_desc,
    &alpha,
    cache.y_desc,  ctx.ptr_at(node.input_ids[1]),   // Y ← 已被 FC BWD 覆写为 dY！
    cache.dy_desc, ctx.ptr_at(node.input_ids[0]),   // dY
    cache.x_desc,  ctx.ptr_at(node.input_ids[3]),   // X
    &beta,
    cache.dx_desc, ctx.ptr_at(node.output_ids[0])); // dX
```

**风险评估**:
- CPU BWD 明确不使用 Y（`input_ids[1]` 被 `(void)` 掉）
- cuDNN Legacy `CUDNN_POOLING_MAX_DETERMINISTIC` 的 backward 实现**可能**从 X 和 pooling 描述符重新计算 max 位置，而非依赖 Y
- 但若 cuDNN 内部用 Y 做验证或索引回查，则会产生**静默的梯度错误**

**建议**: 在 MaxPool BWD 的 GPU launcher 中增加 `TR_CHECK`，确保 `node.input_ids[0] != node.input_ids[1]`（即 dY 和 Y 不是同一个 DTensor）。若相同，应改用 CPU-style mask-based BWD，或在 FWD 阶段将 Y 拷贝到安全缓冲区。

> **当前优先级**: 这是潜在的正确性隐患，但不是导致 `mnist_best_maxpool` 崩溃/不学习的直接原因。修复 P0-1/P0-2 后再验证。

---

### P4: MaxPool INF 输出不必要的 mask 🔧（HFE 未提及，低优先级）

```cpp
// layer_descriptor_registry.cpp:409-417
SubgraphPattern build_maxpool_inference(...) {
    n.output_indices = {0, 1};  // Y + mask
}
```

INF 不需要 mask，但 `infer_maxpool_tensors` 仍为 mask 分配了 `Region::S_MASK` 内存。建议改为 `output_indices = {0}`，节省显存。

---

### P5: MaxPool FWD mask 的 virtual/real 与 MemoryPlan 分配不匹配 🔧（HFE 未提及）

**位置**: `maxpool_op.cpp:281-296`

```cpp
try {
    M->set_output(false)          // virtual tensor
     .set_data_type(fe::DataType_t::INT8);
    mask_virtual = true;
} catch (...) {
    M->set_output(true)           // real tensor
     .set_name("mp_mask")
     .set_dim(...)
     .set_data_type(fe::DataType_t::INT8);
}
```

若 cuDNN 支持 virtual mask，`graph->execute` 不会向 mask 显存写入数据。但 MemoryPlan 仍通过 `infer_maxpool_tensors` 为 `pool_mask` 分配了 `S_MASK` 区域。

**影响**: 每个 MaxPool 层浪费 `N*OH*OW*C` 字节的显存。

**建议**: 若确定 virtual mask 在目标 cuDNN 版本（v9.17）上稳定工作，可在 `infer_maxpool_tensors` 中不返回 mask，或在 `build_maxpool_forward` 中根据运行时判断省略 mask 分配。但为保持 CPU/GPU 路径统一，当前实现可接受。

---

## 三、修改意见（综合方案）

### 修改 1: `compiler.cpp` Phase 1 — 首层 FWD 通用兜底（修复 P0-1）

**位置**: 第 1165 行（Flatten 特殊处理块之后）

```cpp
// [修改] 所有首层且无显式输入的操作，自动绑定数据缓冲区
if (layer.is_first_layer && gn.input_ids.empty()) {
    // 使用 memory_plan 的 baseline 获取正确的 data_a / data_b ID
    const auto& b = memory_plan.baseline();
    
    // A 桶
    GraphNode gn_a = gn;
    gn_a.input_ids = {b.data_a};  // I_A_DATA，避免硬编码 id=1
    train_cg.append(graph_id, gn_a);
    
    // B 桶
    GraphNode gn_b = gn;
    gn_b.input_ids = {b.data_b};  // I_B_DATA，避免硬编码 id=3
    train_cg.append(graph_id_b, gn_b);
    
    // 同步设置 layer_input_ids，供 BWD 使用
    layer_input_ids[l] = b.data_a;
    
    // 更新 prev_output_id 并跳过通用 append
    int32_t out_id = get_layer_output_id(layer.kind, tensor_ids);
    if (out_id >= 0) prev_output_id = out_id;
    continue;
}
```

**关键改进**（对比 HFE 方案）:
- 使用 `memory_plan.baseline().data_a` / `data_b`，而非硬编码 `1`/`3`
- 同步设置 `layer_input_ids[l]`，一举两得修复 P0-2
- `continue` 跳过后续通用 append，避免重复

---

### 修改 2: `compiler.cpp` Phase 2 — 首层 BWD A/B 双缓冲（修复 P0-2）

**位置**: 第 1298–1304 行

```cpp
if (gn.compute_op == ComputeOp::MAXPOOL_FP32_BWD || gn.compute_op == ComputeOp::MAXPOOL_AMP_BWD) {
    if (layer.is_first_layer) {
        const auto& b = memory_plan.baseline();
        
        // A 桶：X = I_A_DATA，dX → I_A_DATA
        gn.input_ids.push_back(b.data_a);
        gn.output_ids = {b.data_a};
        train_cg.append(GraphId::FIRST_LAYER_BWD_A, gn);
        
        // B 桶：X = I_B_DATA，dX → I_B_DATA
        gn.input_ids.back() = b.data_b;
        gn.output_ids = {b.data_b};
        train_cg.append(GraphId::FIRST_LAYER_BWD_B, gn);
        continue;
    } else {
        auto it = layer_input_ids.find(l);
        if (it != layer_input_ids.end() && it->second >= 0) {
            gn.input_ids.push_back(it->second);
            gn.output_ids = {it->second};
        }
    }
}
```

**关键改进**（对比 HFE 方案）:
- 显式处理 `FIRST_LAYER_BWD_A` 和 `FIRST_LAYER_BWD_B` 双桶
- 使用 `memory_plan.baseline()` 而非硬编码 ID
- `continue` 跳过后续通用 append

---

### 修改 3: `mnist_best_maxpool.cpp` — 网络拓扑修正（修复 P2）

```cpp
BluePrint ultimate_mlp = seq(
    maxpool(3, 2, 1),
    flatten(1),           // [NEW] 将 [14,14,1] 展平为 196
    fc(512, true),
    make_activation(cfg.activation),
    fc(512, true),
    make_activation(cfg.activation),
    fc(256, true),
    make_activation(cfg.activation),
    fc(10, true)
);
```

同时更新打印字符串（第 201 行）以反映实际拓扑。

---

### 修改 4: `layer_descriptor_registry.cpp` — INF 不输出 mask（修复 P4，可选）

```cpp
SubgraphPattern build_maxpool_inference(...) {
    // ...
    n.output_indices = {0};  // 仅 Y，不分配 mask
    // ...
}
```

---

### 修改 5: `maxpool_op.cpp` — GPU BWD Y/dY 安全检查（修复 P3，可选）

```cpp
static void launch_maxpool_bwd_cuda_impl(...) {
    // ...
    TR_CHECK(node.input_ids[0] != node.input_ids[1], RuntimeError,
             "MaxPool BWD: dY and Y must not alias (Y was overwritten by previous in-place BWD)");
    // ...
}
```

若此断言在训练中触发，说明需要为 MaxPool 输出引入 shadow buffer，或改用 mask-based BWD（与 CPU 路径一致）。

---

## 四、影响范围评估

| 修改 | 影响范围 | 风险 |
|------|---------|------|
| **修改 1** | 所有首层且无 `input_indices` 的算子（MaxPool、GAP、Dropout、ReLU 等） | 低：仅当 `gn.input_ids.empty() && layer.is_first_layer` 时触发，不影响现有 Flatten/Conv/FC 路径 |
| **修改 2** | 首层 MaxPool BWD | 低：仅在 `layer.is_first_layer` 时触发新分支，非首层走原有 `else` 路径 |
| **修改 3** | 仅 `mnist_best_maxpool.cpp` 测试样例 | 无 |
| **修改 4** | MaxPool INF 内存占用 | 极低：节省显存，无副作用 |
| **修改 5** | GPU BWD 运行时检查 | 极低：仅增加断言，不改动计算逻辑 |

---

## 五、验证路径（修复后）

1. **编译**: `ninja -j30 mnist_best_maxpool`
2. **CPU 模式**: `mnist_best_maxpool.exe --cpu` → 准确率应在 epoch 1-2 内显著高于 11.35%
3. **GPU 模式**: `mnist_best_maxpool.exe --gpu` → 应正常完成 `compile()` 并开始训练
4. **AMP 模式**: `mnist_best_maxpool.exe --amp` → 需配合 `flatten()` 插入后才能通过（否则仍触发 FC 断言）
5. **单元测试回归**: `test_maxpool_fwd_bwd.exe --cpu --gpu --amp` → 确保修改不影响手动构图的 SimpleTask 路径
6. **其他首层算子回归**: 验证 `mnist_best_adamw.exe`（首层为 FC）不受影响

---

## 六、总结

| 问题 | 严重程度 | 是否被 HFE 覆盖 | 当前状态 |
|------|---------|----------------|----------|
| P0-1 首层 FWD 无输入绑定 | **致命** | ✅ HFE1/HFE2 | 待修复 |
| P0-2 首层 BWD 缺 X/dX | **致命** | ✅ HFE1 | 待修复 |
| P1 硬编码 ID 风险 | 中 | ❌ | 建议同步修复 |
| P2 FC AMP 断言 | **致命**（AMP 路径） | ✅ HFE2 | 需修改测试样例拓扑 |
| P3 Y 被 in-place 覆写 | 潜在隐患 | ❌ | 可选，建议加断言监控 |
| P4 INF 多余 mask | 低 | ❌ | 可选优化 |

**核心判断**: HFE 文档对 P0-1/P0-2 的诊断是**完全准确的**。本报告在此基础上补充了：
1. **P1**: 硬编码 ID 的隐患及修正方案
2. **P0-2 补充**: BWD A/B 双缓冲的缺失
3. **P3**: GPU BWD Y/dY 别名的潜在正确性风险
4. **P2 重申**: AMP 路径必须配合 `flatten()` 使用

修复优先级：**P0-1 + P0-2 + P2** 为阻断性修复，必须完成；其余为可选增强。
