# HFE1: MaxPool 首层失败根因分析

## 问题现象

`mnist_best_maxpool.cpp` 测试 MaxPool 可用性：
- **GPU 模式**: 立即崩溃，exit code `-1073741819` (0xC0000005, 内存访问越界)
- **CPU 模式**: 不崩溃但准确率始终 11.35%（≈ 随机猜测，10分类基线）

## 网络结构

```cpp
seq(
    maxpool(3, 2, 1),   // 首层！kernel=3, stride=2, padding=1
    fc(512, true),       // 第一层 FC：784→512
    relu(),
    fc(512, true),
    relu(),
    fc(256, true),
    relu(),
    fc(10, true)
);
```

MaxPool 是 BluePrint 的第一个算子，即架构的**首层 (is_first_layer = true)**。

---

## 根因 #1: 首层 MaxPool 前向传播未注入输入数据

### 位置

[compiler.cpp](file:///r:/renaissance/src/graph/compiler.cpp) 第 1076-1215 行，Phase 1 前向图构建。

### 详细追踪

1. `prev_output_id` 初始化为 `-1`（第 994 行）

2. 遍历 Layer 0 (MaxPool, `is_first_layer = true`):
   - `layer_input_ids[0] = prev_output_id` → `-1`（第 1080 行）
   - `forward_pattern = build_maxpool_forward(...)` 返回：
     ```cpp
     n.op = MAXPOOL_FP32_FWD;  // 或 AMP 变体
     n.output_indices = {0, 1};  // pool_output, pool_mask
     n.input_indices = {};       // 空！未设置
     ```
   - `gn.input_ids = map_indices({})` → **空 `{}`**（第 1138 行）
   - `gn.output_ids = map_indices({0, 1})` → `{pool_output_id, pool_mask_id}`

3. Flatten 特殊处理（第 1142-1165 行）：
   - MaxPool 不是 Flatten，**跳过**
   - 这是**唯一**将 `I_A_DATA`/`I_B_DATA` 注入首层的地方

4. 跨层输入链注入（第 1168-1170 行）：
   ```cpp
   if (prev_output_id >= 0) {
       gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
   }
   ```
   - `prev_output_id = -1` → **条件不满足，跳过**

5. 最终 append（第 1205-1209 行）：
   ```
   gn.input_ids = {}          ← 空！
   gn.output_ids = {pool_output_id, pool_mask_id}
   ```
   被添加到 `FIRST_LAYER_FWD_A` 和 `FIRST_LAYER_FWD_B` 两个图。

### 后果

| 后端 | 症状 | 原因 |
|------|------|------|
| **GPU** | warmup 阶段崩溃 | `warmup_single_cudnn_op` → `launch_cuda` → `launch_maxpool_fwd_cuda_impl` 访问 `node.input_ids[0]` 时越界 |
| **CPU** | 准确率 11.35% | `capture_cpu` 中 `input_ids` 为空，`input_shape` 未设置，MaxPool CPU kernel 读取未初始化内存作为输入，输出垃圾数据。FC 层在垃圾数据上训练，无法学习。 |

### 对比：Flatten 作为首层时为什么正常

Flatten 作为首层时（第 1144-1159 行）：
```cpp
if (layer.is_first_layer) {
    gn_a.input_ids = {1};  // I_A_DATA ← 硬编码注入
    gn_b.input_ids = {3};  // I_B_DATA ← 硬编码注入
}
```

MaxPool 缺少这个注入逻辑。

---

## 根因 #2: 首层 MaxPool 反向传播缺少 X 和 dX

### 位置

[compiler.cpp](file:///r:/renaissance/src/graph/compiler.cpp) 第 1298-1304 行，Phase 2 反向图构建。

### 详细追踪

1. MaxPool BWD 子图模式（[layer_descriptor_registry.cpp](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp) 第 398-407 行）：
   ```cpp
   n.input_indices  = {0, 1};   // pool_output, pool_mask
   n.output_indices = {};        // dX in-place (compiler routes to X)
   ```

2. 编译器处理（第 1260-1304 行）：
   ```cpp
   gn.input_ids = map_indices({0, 1}) = {pool_output_id, pool_mask_id};
   gn.output_ids = map_indices({}) = {};  // 空
   ```

3. 跨层梯度注入（第 1269-1273 行）：
   ```cpp
   if (prev_grad_id >= 0) {
       gn.input_ids.insert(gn.input_ids.begin(), prev_grad_id);
   }
   ```
   - `prev_grad_id` = FC 层的梯度输出（pool_output_id），≥0 → 注入
   - `gn.input_ids = {pool_output_id, pool_output_id, pool_mask_id}`  ← dY, Y, mask

4. MaxPool BWD 特殊处理（第 1298-1304 行）：
   ```cpp
   auto it = layer_input_ids.find(l);   // l=0
   // layer_input_ids[0] = -1  (因为 prev_output_id 初始为 -1)
   if (it != layer_input_ids.end() && it->second >= 0) {
       // it->second = -1, 条件不满足！
       gn.input_ids.push_back(it->second);  // ← 从未执行
       gn.output_ids = {it->second};        // ← 从未执行
   }
   ```

5. 最终 append：
   ```
   gn.input_ids = {pool_output_id, pool_output_id, pool_mask_id}  ← 缺 X
   gn.output_ids = {}                                              ← 空！
   ```

### 后果

MaxPool BWD kernel 期望：
```
input_ids[0] = dY  (prev_grad_id)     ✓
input_ids[1] = Y   (pool_output)      ✓
input_ids[2] = mask (pool_mask)       ✓
input_ids[3] = X   (original input)   ✗ 缺失！
output_ids[0] = dX                     ✗ 缺失！
```

- GPU: `ctx.ptr_at(node.input_ids[3])` 越界访问 → crash
- CPU: `op_ctx->input_ids[3]` 未初始化，读垃圾内存 → 梯度传播错误

### 对比：Flatten BWD 首层处理

Flatten BWD 首层有正确的处理（第 1313-1318 行）：
```cpp
if (layer.is_first_layer) {
    gn.output_ids = {1};  // 首层A：写回 I_A_DATA
    train_cg.append(GraphId::FIRST_LAYER_BWD_A, gn);
    gn.output_ids = {3};  // 首层B：写回 I_B_DATA
    train_cg.append(GraphId::FIRST_LAYER_BWD_B, gn);
}
```

MaxPool BWD 需要类似的首层特殊处理。

---

## 根因 #3 (次要): warmup 对 MaxPool 的处理不完整

### 位置

[op_registry.cpp](file:///r:/renaissance/src/backend/op_registry.cpp) 第 134-219 行，`warmup_single_cudnn_op`。

### 现状

- MaxPool FWD/INF 在 `require_warmup()` 中已注册（第 82-83 行）
- `warmup_single_cudnn_op` 对 MaxPool 走通用 `launch_cuda` 路径（第 199-218 行）
- 通用路径直接调用 `entry.launch_cuda(node, mp, ctx, state)`，这会触发 cuDNN Frontend graph 构建和执行

**如果根因 #1 修复后，warmup 应该能正常工作**，因为通用路径会调用 `launch_maxpool_fwd_cuda_impl`，该函数内部会构建 cuDNN FE graph 并执行。但如果 warmup 阶段需要特殊处理（如 Conv 那样预先构建 graph），则需要额外修改。

### 当前影响

由于根因 #1 导致 `node.input_ids` 为空，warmup 在调用 `launch_cuda` 时也会崩溃，所以 GPU 崩溃发生在 warmup 阶段而非执行阶段。

---

## 修改方案

### 修改 1: compiler.cpp — 前向图首层数据注入（修复根因 #1）

**文件**: `src/graph/compiler.cpp`  
**位置**: 第 1141-1165 行之后、第 1167 行之前

在 Flatten 特殊处理之后，添加非 Flatten 首层算子的数据注入逻辑：

```cpp
// [NEW] 非Flatten首层算子：显式注入数据缓冲区 (I_A_DATA / I_B_DATA)
if (layer.is_first_layer) {
    // 记录首层输入ID供反向传播使用
    layer_input_ids[l] = 1;  // 使用 I_A_DATA (id=1) 作为 X 引用

    GraphNode gn_a = gn;
    gn_a.input_ids.insert(gn_a.input_ids.begin(), 1);  // I_A_DATA
    train_cg.append(graph_id, gn_a);

    GraphNode gn_b = gn;
    gn_b.input_ids.insert(gn_b.input_ids.begin(), 3);  // I_B_DATA
    train_cg.append(graph_id_b, gn_b);

    // 跳过下方的通用 append（第 1205-1209 行）
    int32_t out_id = get_layer_output_id(layer.kind, tensor_ids);
    if (out_id >= 0) prev_output_id = out_id;
    continue;
}
```

**关键点**:
- `layer_input_ids[l] = 1` 确保反向传播能找到 X（I_A_DATA 的 id=1）
- 分别创建 A/B 双缓冲节点，与 Flatten 首层处理模式一致
- 使用 `continue` 跳过第 1205-1209 行的通用 append

### 修改 2: compiler.cpp — 反向图首层 MaxPool BWD（修复根因 #2）

**文件**: `src/graph/compiler.cpp`  
**位置**: 第 1298-1304 行

修改 MaxPool BWD 的首层处理：

```cpp
if (gn.compute_op == ComputeOp::MAXPOOL_FP32_BWD || gn.compute_op == ComputeOp::MAXPOOL_AMP_BWD) {
    if (layer.is_first_layer) {
        // 首层：X = I_A_DATA / I_B_DATA，dX 写回对应缓冲区
        gn.input_ids.push_back(1);   // X = I_A_DATA
        gn.output_ids = {1};         // dX in-place → I_A_DATA
        train_cg.append(GraphId::FIRST_LAYER_BWD_A, gn);

        gn.input_ids.back() = 3;     // X = I_B_DATA
        gn.output_ids = {3};         // dX in-place → I_B_DATA
        train_cg.append(GraphId::FIRST_LAYER_BWD_B, gn);
        continue;
    } else {
        auto it = layer_input_ids.find(l);
        if (it != layer_input_ids.end() && it->second >= 0) {
            gn.input_ids.push_back(it->second);  // X (original input)
            gn.output_ids = {it->second};         // dX in-place to X
        }
    }
}
```

### 修改 3 (可选): warmup 增强

如果修复 #1 后 warmup 仍不稳定，可在 `warmup_single_cudnn_op` 中为 MaxPool 添加专用预热路径（类似 Conv），预构建 graph 并验证 workspace 大小。但预期通用 `launch_cuda` 路径即可满足。

---

## 验证方式

修改后重新运行：
```
# GPU 模式
mnist_best_maxpool.exe --gpu

# AMP 模式
mnist_best_maxpool.exe --amp

# CPU 模式
mnist_best_maxpool.exe --cpu
```

预期：
- GPU/AMP: 不再崩溃，正常训练，准确率 > 97%
- CPU: 准确率不再卡在 11.35%，正常学习

---

## 总结

| 根因 | 严重程度 | 影响范围 |
|------|---------|---------|
| #1 前向首层无数据注入 | **致命** | 所有非 Flatten 首层算子（MaxPool、Conv、GAP 等） |
| #2 反向首层缺 X/dX | **致命** | MaxPool 作为首层时的反向传播 |
| #3 warmup 不完整 | 次要 | 仅 GPU，修复 #1 后可能自动解决 |

核心问题：**编译器假设首层永远是 Flatten**，只有 Flatten 有 `I_A_DATA`/`I_B_DATA` 注入逻辑。当 MaxPool（或其他算子）作为首层时，数据管道断裂。