# EKX1: MaxPool 首层失败 — 综合分析与修改方案

## 0. 前提确认

已阅读 HFE1.md、HFE2.md、HFE3.md，它们的核心结论一致：**MaxPool 作为首层时，Compiler 没有绑定输入数据缓冲区 `I_A_DATA`/`I_B_DATA`，导致 FWD 的 `input_ids` 为空，BWD 的 X/dX 缺失。**

单元测试 `test_maxpool_fwd_bwd` 通过是因为它使用 `SimpleTask` 手动构图，`input_ids` 由用户显式指定，不经过 Compiler 的自动构图逻辑。

以下是我的独立检查分析，补充 HFE*.md 未覆盖或值得深化的部分。

---

## 1. 算子层面：MaxPool 的实现正确性检查

### 1.1 Tensor 声明 (`infer_maxpool_tensors`)

[`layer_descriptor_registry.cpp:371-386`](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L371-L386)

```cpp
return {
    TensorDesc{"pool_output", out, select_feature_region(ctx), feat_dt},
    TensorDesc{"pool_mask",   out, Region::S_MASK,        DType::INT8}
};
```

- **只声明了两个输出张量**（pool_output, pool_mask），没有声明输入张量。
- **设计与 Flatten、ReLU、Tanh 等一致**：这些算子都不声明输入张量，依赖 Compiler 跨层输入链注入前一层输出。
- **对比 FC**：`infer_fc_tensors` 明确声明 `{input, weight, output, ...}`，并在 `build_fc_forward` 中设置 `input_indices = {0, 1}`。FC 作为首层时靠自身声明的 `input_indices` 获取数据，不依赖跨层链。

**结论：Tensor 声明本身没有错误，但隐去了对 Compiler 跨层链的依赖。**

### 1.2 前向子图 (`build_maxpool_forward`)

[`layer_descriptor_registry.cpp:388-396`](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L388-L396)

```cpp
n.op = ...MAXPOOL_XXX_FWD;
n.output_indices = {0, 1};
// n.input_indices 未设置 → 默认空
```

`output_indices = {0, 1}` 映射到 tensor_ids：`{pool_output_id, pool_mask_id}`。正确。

### 1.3 反向子图 (`build_maxpool_backward`)

[`layer_descriptor_registry.cpp:398-407`](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L398-L407)

```cpp
n.input_indices  = {0, 1};   // pool_output, pool_mask
n.output_indices = {};        // dX in-place (compiler routes to X)
```

`input_indices = {0, 1}` 映射为 `{pool_output_id, pool_mask_id}`。反向图还需要：
- `dY`（由 Compiler 跨层梯度链注入）
- `X`（由 Compiler 通过 `layer_input_ids` 注入）
- `dX`（由 Compiler 通过 `layer_input_ids` 设置 output）

### 1.4 GPU FWD 内核 (`launch_maxpool_fwd_cuda_impl`)

[`maxpool_op.cpp:311-362`](file:///r:/renaissance/src/graph/maxpool_op.cpp#L311-L362)

- 使用 cuDNN Frontend `Resample` API（`MAXPOOL` + `NEG_INF_PAD` + `generate_index=true`）
- 生成 INT8 mask（优先 virtual tensor，fallback real）
- 有完善的 graph 缓存机制 (`s_maxpool_fwd_caches`)
- AMP 模式下使用 `padded_c()` 确保 TensorCore 对齐
- 输入绑定：`node.input_ids[0]` → X, `node.output_ids[0]` → Y, `node.output_ids[1]` → mask

**实现正确。但依赖 `node.input_ids[0]` 有效。**

### 1.5 GPU BWD 内核 (`launch_maxpool_bwd_cuda_impl`)

[`maxpool_op.cpp:510-559`](file:///r:/renaissance/src/graph/maxpool_op.cpp#L510-L559)

- 使用 cuDNN Legacy API (`cudnnPoolingBackward`)，模式 `CUDNN_POOLING_MAX_DETERMINISTIC`
- **输入绑定**：
  - `input_ids[0]` = dY (prev_grad_id, Compiler 注入)
  - `input_ids[1]` = Y (pool_output, tensor_ids[0])
  - `input_ids[2]` = mask (pool_mask, tensor_ids[1])
  - `input_ids[3]` = X (原始输入, Compiler 通过 layer_input_ids 注入)
  - `output_ids[0]` = dX (in-place 覆写 X buffer)
- 有完善的 descriptor 缓存机制

**实现正确。但依赖 `input_ids[3]` 和 `output_ids[0]` 有效。**

### 1.6 CPU FWD/BWD 内核

[`maxpool_op.cpp:45-137`](file:///r:/renaissance/src/graph/maxpool_op.cpp#L45-L137)

- NHWC 布局，C 为最内维 (`idx = ((n*H+ih)*W+iw)*C + c`)
- Mask 编码：`max_kh * k + max_kw`（BWD 解码：`m/k` 和 `m%k`）
- BWD 使用 `CUDNN_POOLING_MAX_DETERMINISTIC` 等价模式，不依赖 FWD mask 也能 rerun 找到 max 位置

**数学实现正确。问题在于获取不到正确的输入 shape 和数据指针。**

### 1.7 算子注册 & warmup

[`op_registry.cpp:72-124`](file:///r:/renaissance/src/graph/op_registry.cpp#L72-L124)

- `require_warmup()` 已注册 `MAXPOOL_FP32_FWD`, `MAXPOOL_AMP_FWD`, `MAXPOOL_FP32_INF`, `MAXPOOL_AMP_INF`
- 但 `MAXPOOL_*_BWD` **未注册** warmup — BWD 走 Legacy API 非 Frontend，无需 warmup，正确。
- `warmup_single_cudnn_op` 对 MaxPool 走通用 `launch_cuda` 路径（line 199-218），该路径直接调用 `launch_maxpool_fwd_cuda_impl`，内部会构建 cuDNN FE graph 并执行。

**warmup 逻辑本身正确。但通用 `launch_cuda` 路径在访问 `node.input_ids[0]` 时因 input_ids 为空而越界。**

---

## 2. 编译器层面：首层数据注入的漏洞

### 2.1 Phase 1：前向图构建

[`compiler.cpp:1076-1215`](file:///r:/renaissance/src/graph/compiler.cpp#L1076-L1215)

流程追踪（以 MaxPool 为 Layer 0, `is_first_layer = true`）：

| 步骤 | 行号 | 操作 | 结果 |
|------|------|------|------|
| 初始化 | 994 | `prev_output_id = -1` | — |
| 记录输入 | 1080 | `layer_input_ids[0] = -1` | 空值 |
| map_indices | 1138 | `gn.input_ids = {}`（空） | `input_indices` 未声明 |
| Flatten检查 | 1141-1165 | MaxPool ≠ Flatten，**跳过** | — |
| 跨层注入 | 1167-1170 | `prev_output_id = -1` → **跳过** | — |
| 最终 append | 1205-1209 | `gn.input_ids = {}` **直接 append** | 空输入！ |

**对比 Flatten 首层**（line 1144-1159）：
```cpp
if (layer.is_first_layer) {
    gn_a.input_ids = {1};  // I_A_DATA  ← 硬编码注入
    gn_b.input_ids = {3};  // I_B_DATA
    ...
    continue;  // 跳过通用 append
}
```

**MaxPool（以及 ReLU、Tanh、SiLU 等所有不声明 `input_indices` 的算子）缺少这段逻辑。**

### 2.2 Phase 2：反向图构建

[`compiler.cpp:1226-1359`](file:///r:/renaissance/src/graph/compiler.cpp#L1226-L1359)

MaxPool BWD 节点处理（line 1298-1304）：

```cpp
if (gn.compute_op == MAXPOOL_XXX_BWD) {
    auto it = layer_input_ids.find(l);  // l=0
    // layer_input_ids[0] = -1
    if (it != layer_input_ids.end() && it->second >= 0) {
        // ✗ 条件不满足，整块被跳过！
        gn.input_ids.push_back(it->second);  // X 从未注入
        gn.output_ids = {it->second};         // dX 从未设置
    }
}
```

**结果：**
- `gn.input_ids = {dY, pool_output, pool_mask}` — 缺少 X
- `gn.output_ids = {}` — 缺少 dX

**对比 Flatten BWD 首层**（line 1311-1325）：
```cpp
if (layer.is_first_layer) {
    gn.output_ids = {1};  // dX → I_A_DATA
    train_cg.append(FIRST_LAYER_BWD_A, gn);
    gn.output_ids = {3};  // dX → I_B_DATA
    train_cg.append(FIRST_LAYER_BWD_B, gn);
    continue;
}
```

MaxPool BWD 需要类似处理，但更复杂：不仅需要设置 `output_ids`，还需要注入 `X`（`input_ids[3]`）。

### 2.3 Phase 4：推理图构建（同样受影响）

[`compiler.cpp:1382-1500`](file:///r:/renaissance/src/graph/compiler.cpp#L1382-L1500)

INF 图与 FWD 图有相同的结构：
- Flatten 有首层特殊处理（line 1446-1470）
- MaxPool 无特殊处理
- 跨层注入 `prev_inf_output_id >= 0` 对首层为 false

**影响：如果进行推理/验证，MaxPool INF 节点同样 `input_ids` 为空。**

### 2.4 `prev_grad_id` 传播链断裂

[`compiler.cpp:1342-1356`](file:///r:/renaissance/src/graph/compiler.cpp#L1342-L1356)

```cpp
int32_t grad_id = get_grad_output_id(layer.kind, tensor_ids);
// MaxPool → idx = -1 (in-place)
if (grad_id < 0 && layer.kind == LayerKind::MaxPool) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) grad_id = it->second;
    // layer_input_ids[0] = -1 → grad_id stays -1
}
if (grad_id >= 0) prev_grad_id = grad_id;
```

对于首层 MaxPool，`grad_id = -1` → `prev_grad_id` 保持为上一个值。由于 MaxPool 是第一层，这实际上是**正确的行为**（首层没有"更上一层"需要传播梯度）。但这也意味着如果以后有更复杂的首层结构（如多个首层算子），这个逻辑需要重新审阅。

---

## 3. CPU 模式 11.35% 准确率的完整因果链

### 3.1 FWD 路径

[`capture_cpu.cpp:40-65`](file:///r:/renaissance/src/graph/capture_cpu.cpp#L40-L65)

```
node.input_ids = {} (空)
  → for loop: input_ids.size() == 0, 不填充任何 input_ids
  → op_ctx->input_shape 保持默认 ShapeId{0,0,0,0}

node.output_ids = {pool_output_id, pool_mask_id} (有值)
  → op_ctx->output_shape = shape of pool_output = {N, OH, OW, C}
```

MaxPool CPU FWD 内核（line 50-55）：
```cpp
int N = op_ctx->input_shape.n;  // = 0
int C = op_ctx->input_shape.c;  // = 0
int H = op_ctx->input_shape.h;  // = 0
int W = op_ctx->input_shape.w;  // = 0
// → 所有外循环跳过，pool_output/mask 保持未初始化值（垃圾数据）
```

FC 层读取垃圾 pool_output → 输出随机值 → Loss ≈ random baseline。

### 3.2 BWD 路径

```
node.input_ids = {dY_id, pool_output_id, pool_mask_id} (3个元素，缺X)

capture_cpu:
  → input_shape = shape of input_ids[0] = shape of dY = {N, OH, OW, C}
  → output_shape = ShapeId{0,0,0,0} (output_ids 为空)
```

MaxPool CPU BWD 内核（line 106-114）：
```cpp
int N = op_ctx->output_shape.n;   // = 0
int C = op_ctx->output_shape.c;   // = 0
int IH = op_ctx->output_shape.h;  // = 0
int IW = op_ctx->output_shape.w;  // = 0
// dx_elems = 0 → memset(dx, 0, 0) 什么都不做
// 后续循环不执行 → dX 完全不更新
```

梯度消失 + 前向垃圾数据 = 模型完全不学习 = 准确率 11.35%（10 分类随机基线）。

---

## 4. 问题总结

### 4.1 根因矩阵

| 组件 | 问题 | 严重程度 | 影响 |
|------|------|---------|------|
| `build_maxpool_forward` | 未声明 `input_indices` | 设计选择（非错误） | 依赖 Compiler 跨层链 |
| Compiler Phase 1 (FWD) | 首层无 `I_A_DATA`/`I_B_DATA` 注入 | **致命** | GPU crash / CPU 无效训练 |
| Compiler Phase 2 (BWD) | `layer_input_ids[0] = -1` → X/dX 缺失 | **致命** | GPU crash / CPU 梯度断裂 |
| Compiler Phase 4 (INF) | 同 Phase 1，推理图首层无输入 | **致命** | 推理/验证时 crash 或错误 |
| Warmup | 通用 `launch_cuda` 路径访问空 `input_ids[0]` | 症状 | 是 Phase 1 问题的直接后果 |

### 4.2 受影响的算子（完整列表）

以下算子都未在 `build_*_forward` 中声明 `input_indices`，如果在 Compiler 自动构图中作为首层，均会触发相同问题：

| 算子 | `input_indices` 声明 | 首层是否受影响 |
|------|---------------------|---------------|
| **MaxPool** | 空 | ✅ **已确认崩溃/不学习** |
| **ReLU** | 空 | ✅ 潜伏 |
| **Tanh** | 空 | ✅ 潜伏 |
| **SiLU** | 空 | ✅ 潜伏 |
| **ReLU6** | 空 | ✅ 潜伏 |
| **LeakyReLU** | 空 | ✅ 潜伏 |
| **Hardswish** | 空 | ✅ 潜伏 |
| **ELU** | 空 | ✅ 潜伏 |
| **Sigmoid** | 空 | ✅ 潜伏 |
| **Dropout** | 空 | ✅ 潜伏 |
| **GAP** | 空（且 `infer_gap_tensors` 也未声明输入） | ✅ 潜伏 |
| **Flatten** | 空但有 Compiler 显式兜底 | ❌ 正常 |
| **FC** | `{0,1}`（显式声明） | ❌ 正常 |
| **Conv** | `{0}`（显式声明 weight） | ❌ 正常 |

### 4.3 为什么单元测试 `test_maxpool_fwd_bwd` 通过

[`tests/op/test_maxpool_fwd_bwd.cpp`](file:///r:/renaissance/tests/op/test_maxpool_fwd_bwd.cpp)

该测试使用 `SimpleTask`，在 `build_cg()` 中手动添加 GraphNode 并显式指定 `input_ids`：
- FWD: `input_ids = {x_id}`, `output_ids = {y_id, mask_id}` — 显式绑定，不依赖 Compiler
- BWD: `input_ids = {dy_id, y_id, mask_id, x_id}`, `output_ids = {x_id}` — 完整正确的绑定

这解释了**算子数学正确性**与**端到端训练失败**之间的矛盾：算子本身没问题，Compiler 的构图逻辑在首层情况下有漏洞。

---

## 5. 修改方案

### 5.1 方案选择

经过对三种方案的评估：

- **方案 A**（HFE1 提出）：在 Compiler 中为 MaxPool 单独添加首层处理
- **方案 B**（HFE2 提出）：修改 `build_maxpool_forward` 添加 `input_indices`
- **方案 C**（HFE3 提出）：Compile 中通用处理所有 `input_indices` 为空的首层节点

**推荐方案 C**，理由：
1. **覆盖面最广**：一次修复 MaxPool + ReLU + Tanh + ... 所有潜伏算子
2. **与 Flatten 风格一致**：都是 Compiler 层兜底
3. **不影响已有正确逻辑**：Con、FC 等有显式 `input_indices` 的算子不受影响
4. **符合架构设计意图**：`input_indices` 为空的操作本来就应该由 Compiler 跨层链注入输入

### 5.2 具体修改

#### 修改 1: 前向图（Phase 1）— 非 Flatten 首层数据注入

**文件**: `src/graph/compiler.cpp`
**位置**: 在 Flatten 特殊处理块结束（约 line 1165）之后，跨层输入链注入（line 1167）之前

```cpp
// [NEW] 非Flatten首层算子：无显式 input_indices → 显式注入 I_A_DATA / I_B_DATA
// 覆盖 MaxPool、ReLU、Tanh、SiLU、Dropout、GAP 等依赖跨层链注入输入的算子
if (layer.is_first_layer && gn.input_ids.empty()) {
    // 记录首层输入 ID 供反向传播使用（使用 I_A_DATA 的 DTensor id=1 作为 X 引用）
    layer_input_ids[l] = 1;

    GraphNode gn_a = gn;
    gn_a.input_ids = {1};  // I_A_DATA
    train_cg.append(graph_id, gn_a);

    GraphNode gn_b = gn;
    gn_b.input_ids = {3};  // I_B_DATA
    train_cg.append(graph_id_b, gn_b);

    // 跳过下方的通用 append（line 1205-1209）
    int32_t out_id = get_layer_output_id(layer.kind, tensor_ids);
    if (out_id >= 0) prev_output_id = out_id;
    continue;
}
```

**关键说明**：
- `layer_input_ids[l] = 1`：将首层输入标记为 I_A_DATA 的 DTensor ID。BWD 通过此 ID 注入 X 和设置 dX 输出。
- `gn.input_ids.empty()`：确保此分支不影响已有显式 `input_indices` 的算子（FC、Conv 等）。
- 双缓冲：同时 append 到 A/B 桶，与 Flatten 首层模式一致。

#### 修改 2: 反向图（Phase 2）— MaxPool BWD 首层处理

**文件**: `src/graph/compiler.cpp`
**位置**: line 1298-1304（MaxPool BWD 特殊处理块）

```cpp
if (gn.compute_op == ComputeOp::MAXPOOL_FP32_BWD || gn.compute_op == ComputeOp::MAXPOOL_AMP_BWD) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) {
        if (layer.is_first_layer) {
            // 首层 MaxPool BWD：X = I_A_DATA/I_B_DATA，dX 写回对应缓冲区
            // input_ids 当前为：{dY, pool_output, pool_mask}
            // 需要追加 X，并设置 dX output

            GraphNode gn_a = gn;
            gn_a.input_ids.push_back(1);   // X = I_A_DATA
            gn_a.output_ids = {1};         // dX → I_A_DATA
            train_cg.append(GraphId::FIRST_LAYER_BWD_A, gn_a);

            GraphNode gn_b = gn;
            gn_b.input_ids.push_back(3);   // X = I_B_DATA
            gn_b.output_ids = {3};         // dX → I_B_DATA
            train_cg.append(GraphId::FIRST_LAYER_BWD_B, gn_b);
        } else {
            // 非首层：原逻辑不变
            gn.input_ids.push_back(it->second);  // X (original input)
            gn.output_ids = {it->second};         // dX in-place to X
        }
    }
    continue;  // 跳过下方的通用 append（line 1336-1339）
}
```

**关键说明**：
- 首层时，`layer_input_ids[0] = 1`（由修改 1 设置），所以 `it->second >= 0` 条件满足。
- `input_ids[3] = 1/3`（I_A_DATA / I_B_DATA）：cuDNN Legacy API `cudnnPoolingBackward` 需要 X 来 rerun 找到 max 位置（`CUDNN_POOLING_MAX_DETERMINISTIC` 模式）。
- `output_ids = {1/3}`：dX 原地写回 I_A_DATA / I_B_DATA 缓冲区。
- 必须 `continue` 跳过下方的通用双缓冲 append（line 1336-1339），因为首层 A/B 桶已经分别 append 了不同的节点。

#### 修改 3: 推理图（Phase 4）— 非 Flatten 首层数据注入

**文件**: `src/graph/compiler.cpp`
**位置**: 在 INF 图的 Flatten 特殊处理块结束（约 line 1470）之后，跨层输入链注入（line 1472）之前

```cpp
// [NEW] 非Flatten首层算子推理图：无显式 input_indices → 注入 I_A_DATA / I_B_DATA
if (layer.is_first_layer && gn.input_ids.empty()) {
    GraphNode gn_a = gn;
    gn_a.input_ids = {1};  // I_A_DATA
    infer_cg.append(infer_graph_id, gn_a);

    GraphNode gn_b = gn;
    gn_b.input_ids = {3};  // I_B_DATA
    infer_cg.append(infer_graph_id_b, gn_b);

    int32_t inf_out_id = get_layer_output_id(layer.kind, tensor_ids);
    if (inf_out_id >= 0) prev_inf_output_id = inf_out_id;
    continue;
}
```

#### 修改 4（可选）: 非 Flatten 首层算子的通用 BWD 处理

当前，以下 BWD 算子也存在类似问题（`layer_input_ids[l] = -1` 导致 dX in-place 逻辑不执行）：

- FC BWD（line 1275-1282）
- Tanh/ReLU/SiLU/ReLU6/LeakyReLU/Hardswish/ELU/Sigmoid/Dropout BWD（line 1284-1297）

如果修复后这些算子被放在首层使用，BWD 的 `layer_input_ids[l]` 会被修改 1 正确设为 1，`it->second >= 0` 条件满足，原逻辑就能正确执行。但它们的 BWD **output_ids** 会设为 `it->second`（= 1 即 I_A_DATA），而首层的双缓冲 append（line 1337-1338）会把同一个节点同时 append 到 `FIRST_LAYER_BWD_A` 和 `FIRST_LAYER_BWD_B`，两个图都写入 I_A_DATA（id=1），I_B_DATA（id=3）的梯度丢失。

**当前建议**：此修改 4 暂不实施。因为当前场景（mnist_best_maxpool）MaxPool 是 BluePrint 第一个元素，其后的算子都是非首层。如果需要支持 ReLU/Tanh 等作为首层，再按需扩展。

### 5.3 不需要修改的部分

以下代码不需要修改，确认无问题：

1. **`maxpool_op.cpp`**：算子实现正确，单元测试通过。所有问题在 Compiler 层。
2. **`op_registry.cpp` warmup**：修复 Compiler 后，`node.input_ids[0]` 有效，通用 `launch_cuda` warmup 路径可正常工作。
3. **`layer_descriptor_registry.cpp` `infer_maxpool_tensors` / `build_maxpool_*`**：设计选择（不声明输入张量依赖跨层链）本身是合理的，只需 Compiler 补齐首层兜底。
4. **`capture_cpu.cpp`**：其行为正确——按 `node.input_ids` 填充上下文。修复 Compiler 后 input_ids 正确，capture 结果自然正确。

---

## 6. 验证预期

修复后运行 `mnist_best_maxpool`：

| 模式 | 修复前 | 修复后预期 |
|------|--------|-----------|
| CPU | 11.35% 不变 | 准确率正常上升，最终 > 97% |
| GPU | compile 阶段 crash | 正常训练，准确率 > 97% |
| AMP | compile 阶段 crash | 正常训练，但可能因 FC AMP 的 `H==1 && W==1` 检查在其他位置报错（见 HFE2 5.2 节） |

---

## 7. 总结

| 项目 | 结论 |
|------|------|
| **根因** | Compiler 假设首层必定是 Flatten 或 FC/Conv 等有显式 `input_indices` 的算子。MaxPool（及 ReLU 等）无 `input_indices` 且无首层兜底逻辑，导致数据管道断裂。 |
| **核心修复** | 在 Compiler Phase 1 中，对首层且 `input_ids` 为空的节点显式注入 `I_A_DATA`（id=1）/ `I_B_DATA`（id=3），并设置 `layer_input_ids[l] = 1`。 |
| **BWD 修复** | 首层 MaxPool BWD 需要向 A/B 桶分别注入 X 和设置 dX 为 I_A_DATA / I_B_DATA。 |
| **INF 修复** | 推理图同样需要首层数据注入。 |
| **覆盖范围** | 修改 1 + 修改 3 一次性修复 MaxPool、ReLU、Tanh、SiLU、Dropout、GAP 等所有潜伏的首层问题。 |