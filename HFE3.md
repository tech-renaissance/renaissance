# mnist_best_maxpool 失败根因分析

## 一、现象

| 模式 | 现象 |
|------|------|
| **GPU** | 运行时报内存越界（`Invalid DTensor id` 或 `cuda`/`std::vector` 越界） |
| **CPU** | 准确率恒定在 **11.35%**（MNIST 随机猜测水平），模型完全没有学习 |

---

## 二、问题定位

### 2.1 核心结论

**MaxPool 作为网络首层时，FWD/BWD 的输入绑定双双缺失。**

具体而言：
1. **FWD**：首层 MaxPool 的 `GraphNode::input_ids` 为空，没有绑定到 `I_A_DATA`/`I_B_DATA`
2. **BWD**：首层 MaxPool 的 `layer_input_ids[0] = -1`，导致 BWD 缺少原始输入 X，且 dX 输出未设置

### 2.2 代码级证据

#### 证据 A：`build_maxpool_forward` 未声明 `input_indices`

```cpp
// src/graph/layer_descriptor_registry.cpp : 388-396
SubgraphPattern build_maxpool_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::MAXPOOL_AMP_FWD : ComputeOp::MAXPOOL_FP32_FWD;
    n.output_indices = {0, 1};
    p.nodes.push_back(n);
    return p;
}
```

`input_indices` 未设置 → 为空。

#### 证据 B：Compiler Phase 1 的首层输入注入有漏洞

```cpp
// src/graph/compiler.cpp : 994, 1077-1080, 1167-1170
int32_t prev_output_id = -1;
// ...
std::unordered_map<size_t, int32_t> layer_input_ids;
for (size_t l = 0; l < arch.layers().size(); ++l) {
    layer_input_ids[l] = prev_output_id;  // 第一层 → layer_input_ids[0] = -1
    // ...
    gn.input_ids = map_indices(pattern_node.input_indices);  // MaxPool → 空
    // ...
    if (prev_output_id >= 0) {
        gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
    }
}
```

- `prev_output_id` 初始为 `-1`
- 首层（l=0）时 `layer_input_ids[0] = -1`
- `prev_output_id >= 0` 为 **false**，不会注入输入
- MaxPool FWD 的 `gn.input_ids` 最终为 **空**

#### 证据 C：Flatten 有显式首层处理，MaxPool 没有

```cpp
// src/graph/compiler.cpp : 1141-1165
// Flatten: 显式输入来自数据缓冲区(首层)或前一层输出(非首层)
if (pattern_node.op == ComputeOp::FLATTEN_FP32_FWD ||
    pattern_node.op == ComputeOp::FLATTEN_AMP_FWD) {
    if (layer.is_first_layer) {
        gn_a.input_ids = {1};  // I_A_DATA
        gn_b.input_ids = {3};  // I_B_DATA
        // ...
    }
    continue;
}
```

MaxPool 没有类似的 `if (layer.is_first_layer)` 分支。

#### 证据 D：BWD 节点组装同样缺失

```cpp
// src/graph/compiler.cpp : 1298-1304
if (gn.compute_op == ComputeOp::MAXPOOL_FP32_BWD || gn.compute_op == ComputeOp::MAXPOOL_AMP_BWD) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) {
        gn.input_ids.push_back(it->second);   // X (original input)
        gn.output_ids = {it->second};          // dX in-place to X
    }
}
```

- 首层时 `layer_input_ids[0] = -1`
- `it->second >= 0` 为 **false**
- BWD `gn.input_ids` 保持为 `{dY, mask}`（缺少 X）
- BWD `gn.output_ids` 保持为 **空**（缺少 dX）

### 2.3 为什么 GPU 报越界，CPU 卡 11.35%

#### GPU 路径（CUDA Graph capture）

```cpp
// maxpool_op.cpp : 319
const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
```

FWD 的 `node.input_ids` 为空，`node.input_ids[0]` 访问空 `std::vector` → **未定义行为/越界**。

BWD 同理：`node.input_ids[3]` 越界（只有 2 个元素）。

#### CPU 路径（capture_cpu）

```cpp
// capture_cpu.cpp : 40-45, 48-65
for (size_t i = 0; i < node.input_ids.size() && i < 12; ++i) { ... }
// input_ids 为空 → input_shape 不设置（默认 ShapeId{0,0,0,0}）
```

```cpp
// maxpool_op.cpp CPU FWD
int N = op_ctx->input_shape.n, C = op_ctx->input_shape.c;
int H = op_ctx->input_shape.h, W = op_ctx->input_shape.w;
// N=0, C=0, H=0, W=0 → 所有循环不执行 → Y 全为 0
```

```cpp
// maxpool_op.cpp CPU BWD
int N = op_ctx->output_shape.n, C = op_ctx->output_shape.c;
// output_shape 同样为 {0,0,0,0} → dx_elems = 0
// memset(dx, 0, 0) + 零次循环 → dX 不更新
```

**整个网络接收全 0 输入、梯度不更新** → 模型完全不学习 → 准确率恒为随机水平（11.35%）。

---

## 三、为什么其他算子作为首层时正常

| 首层算子 | `build_forward` 有 `input_indices`？ | Compiler 有首层特殊处理？ | 结果 |
|----------|--------------------------------------|---------------------------|------|
| **FC** | ✅ 有（FP32: {0,1}，AMP: {5,1}） | ❌ 无（靠 `input_indices` 自身） | 正常 |
| **Flatten** | ❌ 无 | ✅ 有（显式绑定 `I_A_DATA`/`I_B_DATA`） | 正常 |
| **MaxPool** | ❌ 无 | ❌ 无 | **崩溃/不学习** |
| **ReLU/Tanh/...** | ❌ 无 | ❌ 无 | **理论上同病**（但未在首层使用过） |

> 注：ReLU、Tanh、SiLU 等激活函数的 `build_*_forward` 同样未设置 `input_indices`，如果它们被放在首层，也会遇到完全相同的问题。

---

## 四、修改建议

### 方案 A：在 `compiler.cpp` 中为 MaxPool 添加首层特殊处理（推荐，与 Flatten 风格一致）

在 `src/graph/compiler.cpp` Phase 1 的 Flatten 首层处理块之后，增加 MaxPool（以及所有无 `input_indices` 的算子）的首层分支：

```cpp
// 在 Flatten 特殊处理块之后（约第 1165 行之后）

// MaxPool / ReLU / Tanh / SiLU / ... 首层：显式绑定 I_A_DATA / I_B_DATA
if (!pattern_node.input_indices.empty()) {
    // 已有显式输入索引，走常规路径
} else if (layer.is_first_layer) {
    // 无 input_indices 且为首层 → 输入来自数据缓冲区
    GraphNode gn_a = gn;
    gn_a.input_ids = {1};  // I_A_DATA
    train_cg.append(graph_id, gn_a);

    GraphNode gn_b = gn;
    gn_b.input_ids = {3};  // I_B_DATA
    train_cg.append(graph_id_b, gn_b);
    continue;  // 跳过后续通用处理
}
```

> 但注意：MaxPool FWD 的 `output_indices = {0, 1}` 需要保留，上述代码需要确保 `gn_a.output_ids` / `gn_b.output_ids` 已正确设置。

### 方案 B：修改 `build_maxpool_forward`，添加输入 TensorDesc 并设置 `input_indices`

```cpp
// layer_descriptor_registry.cpp
std::vector<TensorDesc> infer_maxpool_tensors(...) {
    return {
        TensorDesc{"pool_input",  input, select_feature_region(ctx), feat_dt},  // 新增
        TensorDesc{"pool_output", out,   select_feature_region(ctx), feat_dt},
        TensorDesc{"pool_mask",   out,   Region::S_MASK,        DType::INT8}
    };
}

SubgraphPattern build_maxpool_forward(...) {
    // ...
    n.input_indices  = {0};    // pool_input
    n.output_indices = {1, 2}; // pool_output, pool_mask
    // ...
}
```

但此方案会改变 `descs` 的索引映射，需要同步修改 `build_maxpool_backward`（`input_indices` 需要从 `{0,1}` 改为 `{1,2}` 或相应值），影响面较大。

### 方案 C：通用修复——在 Compiler Phase 1 中统一处理所有无 `input_indices` 的首层节点

将现有代码：
```cpp
if (prev_output_id >= 0) {
    gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
}
```

改为：
```cpp
if (prev_output_id >= 0) {
    gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
} else if (layer.is_first_layer && gn.input_ids.empty()) {
    // 首层且无显式输入 → 绑定数据缓冲区
    gn.input_ids = {1};  // I_A_DATA
    // 双缓冲 B 桶在后续统一处理
}
```

此方案最简洁，且同时修复 ReLU/Tanh/SiLU 等算子潜在的首层问题。

### 关于 BWD 的修复

BWD 的问题（`layer_input_ids[0] = -1`）是 FWD 问题的直接后果。一旦 FWD 正确绑定了首层输入，`layer_input_ids[0]` 需要被设置为**首层输入张量的 ID**（即 `I_A_DATA` 对应的 DTensor ID）。

但注意：`layer_input_ids[l]` 的语义是"该层前向输入 X 的 DTensor ID"，用于 BWD 时路由 dX。如果首层输入是 `I_A_DATA`（固定 baseline ID = 1 或 3），那么 `layer_input_ids[0]` 应该被设为 1（或 3，但通常 A 桶是主路径）。

然而，当前 `layer_input_ids[0] = prev_output_id = -1` 的赋值在第 1080 行。如果改为在首层特殊处理时同步设置 `layer_input_ids[0] = 1`，BWD 的 `if (it->second >= 0)` 就能进入，X 和 dX 就能正确绑定。

---

## 五、验证思路

修复后，可通过以下方式验证：

1. **打印 ComputationGraph**：在 `compiler.cpp` 或 `task_base.cpp` 中打印首层 MaxPool FWD/BWD 的 `input_ids` / `output_ids`，确认：
   - FWD: `input_ids` 包含 `1`（I_A_DATA），`output_ids` 包含 pool_output 和 pool_mask
   - BWD: `input_ids` 包含 dY、Y、mask、X，`output_ids` 包含 X（in-place）

2. **运行 `mnist_best_maxpool --cpu`**：准确率应在几轮内上升，不再卡在 11.35%

3. **运行 `mnist_best_maxpool --gpu`**：不再报越界错误，训练正常进行

4. **对比 `test_maxpool_fwd_bwd`**：该测试使用 `SimpleTask` 手动构图（非 `DeepLearningTask` 自动 IR），`input_ids` 是显式指定的，因此不受此 bug 影响。这也解释了为什么单元测试通过但端到端训练失败。

---

## 六、总结

| 项目 | 结论 |
|------|------|
| **根因** | MaxPool 的 `build_maxpool_forward` 未设置 `input_indices`；Compiler 对首层无 `input_indices` 的算子未注入 `I_A_DATA`/`I_B_DATA`；同时 `layer_input_ids[0] = -1` 导致 BWD 缺失 X 和 dX |
| **影响范围** | 所有将 MaxPool（以及 ReLU/Tanh/SiLU/...）放在首层的 `DeepLearningTask` 自动构图场景 |
| **不影响** | `SimpleTask` 手动构图（如 `test_maxpool_fwd_bwd`），因为 `input_ids` 是手动显式指定的 |
| **小伙伴结论** | ✅ **正确**："maxpool 作为首层时，输入节点未绑定" |
| **推荐修复** | **方案 C**：在 Compiler Phase 1 中统一处理 `gn.input_ids.empty() && layer.is_first_layer` 的情况，注入 `I_A_DATA`（ID=1），并同步修正 `layer_input_ids[0]` |
