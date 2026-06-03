# HFE2: MaxPool 首层输入绑定缺失诊断报告

## 1. 现象

- `mnist_best_maxpool.exe --cpu`：能跑通，但 val top1 锁定在 **11.35%**（随机猜测水平）。
- `mnist_best_maxpool.exe --gpu`：`compile()` 阶段直接崩溃，日志仅打印 `Device: GPU [0]` 和 `Random Seed`。

## 2. 根因定位

### 2.1 MaxPool FWD 未声明 `input_indices`

在 `src/graph/layer_descriptor_registry.cpp` 中：

```cpp
SubgraphPattern build_maxpool_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::MAXPOOL_AMP_FWD : ComputeOp::MAXPOOL_FP32_FWD;
    n.output_indices = {0, 1};
    // ❌ n.input_indices 未设置，默认为空
    p.nodes.push_back(n);
    return p;
}
```

MaxPool 前向节点没有声明任何 `input_indices`。

### 2.2 Compiler Phase 4 的跨层输入链对首层失效

在 `src/graph/compiler.cpp` 的 FWD 构建逻辑中：

```cpp
// 跨层输入链：注入前一层输出DTensor ID
if (prev_output_id >= 0) {
    gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
}
```

- **非首层**：`prev_output_id >= 0`，跨层链正常工作，MaxPool 获得前一层输出作为输入。
- **首层**：`prev_output_id = -1`，跨层链不执行。由于 `build_maxpool_forward` 未声明 `input_indices`，`gn.input_ids` 最终为空。

### 2.3 首层无兜底绑定

Compiler 中仅有 **Flatten** 享有显式的首层特殊处理：

```cpp
if (pattern_node.op == ComputeOp::FLATTEN_FP32_FWD ||
    pattern_node.op == ComputeOp::FLATTEN_AMP_FWD) {
    if (layer.is_first_layer) {
        gn_a.input_ids = {1};  // I_A_DATA
        ...
    }
}
```

MaxPool、GAP、Dropout 等同样未声明 `input_indices` 的操作，**没有类似的首层兜底逻辑**。

### 2.4 为什么 CPU 没崩溃但结果错误

在 `src/graph/capture_cpu.cpp` 中：

```cpp
for (size_t i = 0; i < node.input_ids.size() && i < 12; ++i) {
    op_ctx->input_ids[op_ctx->num_inputs++] = node.input_ids[i];
}
```

如果 `node.input_ids` 为空，则 `num_inputs = 0`，但 `input_ids[12]` 数组被默认零初始化。

在 `src/backend/ops/dtensor/maxpool_op.cpp` 的 CPU 内核中：

```cpp
const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
```

`input_ids[0] == 0`，访问的是 **DTensor ID 为 0 的内存区域**。该区域恰好是有效内存（可能是第一个分配的 tensor），因此未触发崩溃，但 MaxPool 处理的是错误数据，导致模型完全无法学习（11.35% 随机精度）。

### 2.5 为什么 GPU 直接崩溃

在 GPU 内核中：

```cpp
const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
```

`node.input_ids` 是空 `std::vector`，访问 `[0]` 导致 **未定义行为**（越界访问），在 MSVC 下直接 abort。

## 3. 影响范围

| 操作 | `input_indices` 为空 | 作为首层时是否受影响 |
|------|---------------------|---------------------|
| MaxPool | ✅ | ✅ 崩溃/错误 |
| GAP | ✅ | ✅ 崩溃/错误 |
| Dropout | ✅ | ✅ 崩溃/错误 |
| Flatten | ✅ | ❌ 已有显式兜底 |
| ReLU 等激活 | ✅（理论上） | 通常不作为首层，风险低 |

> 注：在正常 CNN 中首层是 Conv，Conv 的 `build_conv_forward` 声明了 `input_indices = {0}`（weight），不受此问题影响。

## 4. 修复方案

### 方案 A：Compiler 层通用兜底（推荐）

在 `src/graph/compiler.cpp` 的 Phase 4 中，对所有首层且无显式输入的操作，自动绑定 `I_A_DATA`。

**修改位置 1：FWD 训练图**

在第 1170 行（跨层输入链）之后添加：

```cpp
// 首层兜底：无显式输入且无跨层输入时，绑定 I_A_DATA
if (layer.is_first_layer && gn.input_ids.empty()) {
    gn.input_ids = {1};  // I_A_DATA
}
```

**修改位置 2：INF 推理图**

在第 1474 行（跨层输入链）之后添加同样的逻辑。

**优点**：
- 一次修复覆盖 MaxPool、GAP、Dropout 及未来可能出现的类似操作。
- 不影响已有逻辑（Conv、FC 等已有 `input_indices` 的操作不会进入此分支）。
- 与 Flatten 的首层处理逻辑对齐。

### 方案 B：逐个操作在 `build_xxx_forward` 中声明 `input_indices`

例如修改 `build_maxpool_forward`：

```cpp
SubgraphPattern build_maxpool_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = ...;
    n.input_indices = {};   // 数据输入由 compiler 跨层链/首层兜底注入
    n.output_indices = {0, 1};
    p.nodes.push_back(n);
    return p;
}
```

此方案**无法单独解决问题**，因为即使显式声明空 `input_indices`，首层仍需要 compiler 的兜底逻辑。方案 B 必须与方案 A 结合使用，或改为在 `build_maxpool_forward` 中声明某个特殊索引让 compiler 识别为首层输入——但这会引入新的约定，不如方案 A 简洁。

## 5. 其他关联问题

### 5.1 `mnist_best_maxpool.cpp` 的语义不匹配

当前代码注释和打印信息写的是 `Network: 784→1024→512→256→10`，实际网络是：

```cpp
seq(
    maxpool(3, 2, 1),   // [28,28,1] -> [14,14,1]
    fc(512, true),      // 输入 14*14*1=196
    ...
)
```

应更新注释和打印信息以反映实际拓扑。

### 5.2 AMP 路径的 FC `H==1 && W==1` 断言

`src/backend/ops/dtensor/fc_op.cpp` 中：

```cpp
TR_DEBUG_CHECK(dt_x.h() == 1 && dt_x.w() == 1, ShapeError,
               "FC_AMP_FWD input must have H=1, W=1. ...");
```

MaxPool 输出 `[14,14,1]`，即使修复了首层输入绑定，FC AMP 仍会在此断言处崩溃。

**解决方向**：
- 在 MaxPool 后显式插入 `flatten(1)`（用户代码层面）。
- 或修改 `step5_normalize_flatten()`，使其不仅检查 `C%8`，还检查 `H*W > 1`。

## 6. 验证步骤（修复后）

1. 应用方案 A 修改 `compiler.cpp`。
2. 增量编译 `ninja -j30 mnist_best_maxpool`。
3. 运行 `mnist_best_maxpool.exe --cpu`：应能看到 val top1 开始正常上升（不再是 11.35%）。
4. 运行 `mnist_best_maxpool.exe --gpu`：应能正常启动训练，不再崩溃。
5. 运行 `mnist_best_maxpool.exe --amp`：仍会崩溃于 FC AMP 的 `H==1 && W==1` 断言（需额外修复）。
