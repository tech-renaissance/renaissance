# EKX2: MaxPool 首层输入绑定缺失 -- 独立诊断与修改意见

## 一、问题确认

阅读 HFE1/HFE2/HFE3 三份报告并核对相关代码后，**完全认同核心结论**：

> MaxPool 作为网络首层时，Compiler 未为其注入 I_A_DATA/I_B_DATA，导致 FWD input_ids 为空；同时 layer_input_ids[0] = -1 导致 BWD 缺少原始输入 X 和 dX 输出槽位。

这是 mnist_best_maxpool.exe CPU 模式准确率锁定 11.35%、GPU 模式直接崩溃的根本原因。

---

## 二、补充细节与修正

### 2.1 Infer 图同样受影响（三份 HFE 报告均未提及）

src/graph/compiler.cpp 第 1439-1474 行构建推理图（INF）时，Flatten 有特殊处理：

```cpp
if (pattern_node.op == ComputeOp::FLATTEN_FP32_FWD ||
    pattern_node.op == ComputeOp::FLATTEN_AMP_FWD) {
    if (layer.is_first_layer) {
        gn_a.input_ids = {1};  // I_A_DATA
        ...
    }
}
```

MaxPool 的 build_maxpool_inference 同样未声明 input_indices：

```cpp
SubgraphPattern build_maxpool_inference(...) {
    // n.output_indices = {0, 1};  // Y + mask
    // n.input_indices 未设置 -> 空
}
```

因此 --inf（验证/推理阶段）也会遇到完全相同的空输入问题。修复时必须同时覆盖 **FWD、BWD、INF** 三个图。

### 2.2 CPU 模式 11.35% 的准确机制（比 HFE2 更精确）

HFE2 认为 CPU 内核读了 DTensor ID=0 的内存。实际流程更精确的是 HFE3 的分析：

1. capture_cpu.cpp 第 40-45 行：
   ```cpp
   for (size_t i = 0; i < node.input_ids.size() && i < 12; ++i) { ... }
   ```
   node.input_ids.empty() -> num_inputs = 0，input_shape **不被设置**。

2. CpuOpContext::input_shape 是默认构造的 ShapeId{0,0,0,0}。

3. maxpool_op.cpp CPU FWD：
   ```cpp
   int N = op_ctx->input_shape.n;  // N = 0
   ```
   外层循环 for (int n = 0; n < N; ++n) -> **零次执行**，Y buffer 完全不被写入。

4. FC 接收到的是 **全零输入**（或内存初始化值），输出由 bias 主导，模型学到几乎恒定的预测，top1 锁定在随机水平（约10%），实测 11.35% 在此范围内。

### 2.3 HFE1 修改方案 1 的潜在缺陷

HFE1 的修改 1 建议在 Flatten 处理块后插入 continue 来跳过通用 append。虽然该方案在 continue 之前手动更新了 prev_output_id，但 **continue 会跳过通用 append 中的所有后续逻辑**（如第 1190-1203 行的 SoftmaxCE 基线注入、FC debug 打印等）。更重要的是，这是一种"硬编码分支"风格，与 Flatten 的特殊处理耦合在一起，后续若通用 append 增加新逻辑，这个 continue 会造成维护隐患。

**推荐做法**：不要 continue，而是在通用 append 之前用条件注入 input_ids，让通用路径仍然执行。

### 2.4 layer_input_ids 必须同步修正

layer_input_ids[l] 的语义是"该层前向输入 X 的 DTensor ID"，用于 BWD 时路由 dX。

当前代码第 1080 行：
```cpp
layer_input_ids[l] = prev_output_id;  // 首层 -> -1
```

即使 FWD 兜底把 gn.input_ids 设为 {1}，如果不同时修正 layer_input_ids[l]，BWD 的 if (it->second >= 0) 仍然不进入，X 和 dX 仍缺失。

**因此修复必须是成对的**：FWD 注入 I_A_DATA 的同时，把 layer_input_ids[l] 设为 1。

---

## 三、当前 MaxPool 实现的其他问题

### 3.1 build_maxpool_forward / build_maxpool_inference 的 input_indices 设计

MaxPool FWD 的 build_maxpool_forward 未设置 input_indices 是**有意为之**的：MaxPool 没有权重参数，唯一的输入是前一层输出（或首层数据），应由 Compiler 的跨层链注入。

这种设计本身没错，但 Compiler 的跨层链对首层失效（prev_output_id = -1），且只有 Flatten 有首层兜底。这是 **Compiler 的通用缺失**，不是 MaxPool 算子本身的 bug。

### 3.2 BWD 的 input_indices 与 Compiler 注入顺序的隐含契约

build_maxpool_backward：
```cpp
n.input_indices  = {0, 1};   // pool_output, pool_mask
n.output_indices = {};        // dX in-place (compiler routes to X)
```

Compiler 处理后的实际 binding（由注释确认）：
```
input_ids[0] = dY   (prev_grad_id, compiler 跨层梯度链注入到开头)
input_ids[1] = Y    (pool_output, map_indices({0}))
input_ids[2] = mask (pool_mask,  map_indices({1}))
input_ids[3] = X    (layer_input_ids[l], compiler push_back)
output_ids[0] = dX  (layer_input_ids[l], compiler 设置)
```

这个契约在 layer_input_ids[l] >= 0 时是正确的。但首层时 layer_input_ids[0] = -1，契约断裂。

### 3.3 CPU BWD 内核中 input_ids[3] 的访问

maxpool_op.cpp CPU BWD：
```cpp
(void)op_ctx->ctx->ptr_at(op_ctx->input_ids[3]);  // X，声明需要但实现未使用
```

CPU BWD 实现使用 mask 路由，确实不需要 X。但接口统一要求 input_ids[3] 存在。如果 input_ids 只有 2 个元素（dY 和 mask），访问 [3] 是越界。这进一步说明 BWD 的 input binding 必须完整。

---

## 四、修改意见

### 4.1 修改 1：FWD 图首层兜底（compiler.cpp）

**位置**：Phase 4 正向循环，第 1167-1170 行（跨层输入链）之后。

```cpp
// 跨层输入链：注入前一层输出DTensor ID
if (prev_output_id >= 0) {
    gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
}

// [NEW] 首层兜底：无显式输入且无跨层输入时，绑定 I_A_DATA
if (layer.is_first_layer && gn.input_ids.empty()) {
    gn.input_ids = {1};  // I_A_DATA
    layer_input_ids[l] = 1;  // 同步修正 BWD 所需的 X 引用
}
```

**关键点**：
- 不 continue，让通用 append（第 1205-1209 行）正常执行，SoftmaxCE 基线注入等逻辑不受影响。
- layer_input_ids[l] = 1 同步修正，确保 BWD 能找到 X。
- 此修改同时覆盖 MaxPool、GAP、Dropout 及所有未来无 input_indices 的首层算子。

### 4.2 修改 2：INF 推理图首层兜底（compiler.cpp）

**位置**：INF 循环，第 1472-1474 行（跨层输入链）之后。

```cpp
if (prev_inf_output_id >= 0) {
    gn.input_ids.insert(gn.input_ids.begin(), prev_inf_output_id);
}

// [NEW] 首层兜底
if (layer.is_first_layer && gn.input_ids.empty()) {
    gn.input_ids = {1};  // I_A_DATA
}
```

INF 没有 BWD，无需 layer_input_ids。

### 4.3 修改 3：BWD 图首层 MaxPool 处理（compiler.cpp）

**位置**：BWD 循环，第 1298-1304 行。

当前代码：
```cpp
if (gn.compute_op == ComputeOp::MAXPOOL_FP32_BWD || gn.compute_op == ComputeOp::MAXPOOL_AMP_BWD) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) {
        gn.input_ids.push_back(it->second);  // X
        gn.output_ids = {it->second};         // dX in-place to X
    }
}
```

修改为：
```cpp
if (gn.compute_op == ComputeOp::MAXPOOL_FP32_BWD || gn.compute_op == ComputeOp::MAXPOOL_AMP_BWD) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) {
        gn.input_ids.push_back(it->second);  // X
        gn.output_ids = {it->second};         // dX in-place to X
    } else if (layer.is_first_layer) {
        // 首层兜底：X 和 dX 都指向 I_A_DATA
        gn.input_ids.push_back(1);
        gn.output_ids = {1};
    }
}
```

**关键点**：
- 不 continue，让通用 append（第 1336 行 train_cg.append）和 prev_grad_id 更新（第 1353-1356 行）正常执行。
- 由于修改 1 中已经设置了 layer_input_ids[l] = 1，理论上 else if (layer.is_first_layer) 分支不会走到。但保留这个兜底是防御性编程，防止修改 1 被回滚或遗漏。

**关于 BWD 双缓冲的说明**：
BWD 循环中 backward_graph_id 已按 A/B 区分：
```cpp
GraphId backward_graph_id = layer.is_first_layer ? GraphId::FIRST_LAYER_BWD_A : GraphId::DEEP_FWD_BWD;
```

但 B 桶在哪里？检查代码后发现 BWD 循环中没有显式的 B 桶 append。这可能意味着首层 BWD 只构建 A 桶，B 桶由其他机制处理，或者这是一个现有框架的已知简化。此问题超出 MaxPool 修复范围，暂不深入。

---

## 五、修改后预期行为

| 模式 | 修复前 | 修复后 |
|------|--------|--------|
| CPU | 11.35% 恒定，模型不学习 | 正常学习，准确率几轮内上升 |
| GPU | compile() 崩溃（空 vector 越界） | 正常训练，不再崩溃 |
| AMP | compile() 崩溃 | 仍可能崩溃于 FC AMP 的 H==1 && W==1 断言（需额外修复） |
| INF | 首层空输入，推理崩溃 | 正常推理 |

---

## 六、验证步骤

1. 应用上述三处修改。
2. 增量编译 ninja -j30 mnist_best_maxpool。
3. 运行 mnist_best_maxpool.exe --cpu 50 秒：观察 val top1 是否脱离 11.35%。
4. 运行 mnist_best_maxpool.exe --gpu 50 秒：确认不再崩溃。
5. 运行 mnist_best_maxpool.exe --amp：若仍崩溃，检查是否为 FC AMP 断言（非 MaxPool 问题）。
6. 重新运行 test_maxpool_fwd_bwd（CPU/GPU/AMP）：确认单元测试仍通过（不应受影响，因为 SimpleTask 手动构图不受 Compiler 修改影响）。
