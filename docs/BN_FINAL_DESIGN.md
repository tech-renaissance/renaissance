# DTY_FINAL：BN1D / BN2D 单算子最终实现方案

> **版本**: V1.2  
> **日期**: 2026-06-07  
> **依据**: DTY0.md（需求说明）、DTY1.md~DTY4.md（四份方案），以及 B_REV.md（小伙伴审查意见）。基于现有框架代码（`layer_descriptor_registry.cpp`、`compiler.cpp`、`memory_plan.h/cpp`、`types.h`、`op_kind.h`、`op_stream_policy.cpp`、`op_registry.cpp`）的全面审计。  
> **定位**: 本文件是 BN1D/BN2D 实现的**唯一最终方案**，综合了 DTY1~DTY4 的全部正确结论与 B_REV.md 审查意见，修正了各方案中的错误和不一致之处，与现有代码状态完全对齐。

---

## 一、各 DTY 方案差异裁决

DTY1~DTY4 在核心设计上高度一致，但在以下关键细节上存在分歧。本节逐一裁决。

### 1.1 BWD 的 `input_indices`

| 方案 | BWD input_indices | 问题 |
|------|-------------------|------|
| DTY1 | `{0, 2, 11, 12}` | 索引 2 是 `bn_output`（Y），不是 X。`batchnorm_backward` 需要原始输入 X，Y 无法满足。且 Compiler 已通过特殊处理追加 X，重复包含索引 2 会导致错误双 X 输入。 |
| DTY2 | `{0, 11, 12}` | **正确**。仅 weight、saved_mean、saved_inv_var。X 由 Compiler 特殊处理追加。 |
| DTY3 | `{0, 11, 12}` | **正确**。同 DTY2。 |
| DTY4 | `{0, 2, 11, 12}` | 同 DTY1 的错误。 |

**裁决**: `{0, 11, 12}`。不包含索引 2（bn_output），因为 X 由 Compiler 通过 `layer_input_ids[l]` 追加。

### 1.2 BWD 的 `output_indices`（dX 覆盖目标）

| 方案 | BWD output_indices | 问题 |
|------|---------------------|------|
| DTY1 | `{7, 8}` | **正确**。dX 由 Compiler 特殊处理插入 `output_ids[0]`。 |
| DTY2 | `{7, 8}` | **正确**。 |
| DTY3 | `{7, 8}` | **正确**。 |
| DTY4 | `{2, 7, 8}` | **错误**。dX 写入 bn_output（索引 2），违反框架铁律 "dX 覆盖 X"。 |

**裁决**: `{7, 8}`。dX 由 Compiler 特殊处理注入 `output_ids[0]`（`layer_input_ids[l]`）。

### 1.3 `eq_scale` / `eq_bias` 是否使用占位符

| 方案 | 处理方式 | 问题 |
|------|---------|------|
| DTY1 | 始终真实张量 | **正确**。推理时必定需要这两个张量。 |
| DTY2 | 始终真实张量 | **正确**。 |
| DTY3 | 始终真实张量 | **正确**。 |
| DTY4 | `ctx.bn_folded` 条件判断 | **错误**。单算子 BN 推理需要 eq_scale/eq_bias，占位符会导致无内存可用。 |

**裁决**: 始终申请真实张量，移除 `ctx.bn_folded` 条件判断。

### 1.4 FWD 的 `input_indices`

| 方案 | FWD input_indices | 裁决 |
|------|-------------------|------|
| DTY1~DTY4 | `{0, 1, 5, 6}` | **一致通过**。weight(0), bias(1), next_mean(5), next_var(6)。 |

### 1.5 FWD 的 `output_indices`

| 方案 | FWD output_indices | 裁决 |
|------|---------------------|------|
| DTY1~DTY4 | `{2, 11, 12}` | **一致通过**。output(2), saved_mean(11), saved_inv_var(12)。 |

### 1.6 INF 的 `input_indices`

| 方案 | INF input_indices | 裁决 |
|------|-------------------|------|
| DTY1~DTY4 | `{10, 9}` | **一致通过**。eq_scale(10), eq_bias(9)。 |

### 1.7 裁决总结

| 决策项 | 最终裁决 | 采纳方案 |
|--------|---------|---------|
| 张量总数 | 13 张量 | DTY1/2/3/4 一致 |
| FWD `input_indices` | `{0, 1, 5, 6}` | 一致 |
| FWD `output_indices` | `{2, 11, 12}` | 一致 |
| BWD `input_indices` | `{0, 11, 12}` | DTY2/DTY3 |
| BWD `output_indices` | `{7, 8}` | DTY1/2/3 |
| dX 覆盖目标 | `layer_input_ids[l]`（X） | DTY1/2/3 |
| INF `input_indices` | `{10, 9}` | 一致 |
| `eq_scale`/`eq_bias` | 始终真实张量 | DTY1/2/3 |
| Running stats 绑定 | B_NEXT（原地更新） | 一致 |
| 全局标量 | Baseline 申请 | 一致 |
| STATS_COMM 范围 | 4 个区 | 一致 |
| `get_grad_output_id` | `-1` | DTY2/3 |

---

## 二、核心设计原则

1. **单算子路径**：使用 cuDNN Frontend `batchnorm` / `batchnorm_backward` / Pointwise（INF），不经过 `BN Finalize` + `GenStats`。
2. **兼容 BN Finalize 融合路径**：为将来的 CBR / ConvBN 融合算子预留全部必需的统计量张量，确保 RANGE OP 批量操作时张量一一对应。
3. **BN1D 与 BN2D 张量完全对齐**：申请相同数量、种类、Region、dtype 的张量，仅特征图形状不同（BN1D: `[N,1,1,C]`，BN2D: `[N,H,W,C]`）。
4. **dX 覆盖 X**：反向传播的梯度输出覆盖正向的原始输入（`layer_input_ids[l]`），同张量分时复用。
5. **所有 BN 参数张量均为 FP32**：即使在 AMP 模式下，仅输入/输出特征图为 FP16，其余 BN 参数（γ, β, running_mean/var, 梯度, saved_mean, saved_inv_var 等）均为 FP32。
6. **不使用 GRAD_SLOT 区**：特征图梯度放回特征图位置（dX 覆盖 X）。
7. **CPU 实现用 Eigen3 + 朴素循环 fallback**。AMP 仅 GPU 支持。
8. **NHWC 布局**：特征图 stride `{H*W*C, 1, W*C, C}`（`stride[1]=1`，C 维连续）。逐通道参数 stride `{C, 1, C, C}`。全局标量形状 `[1,1,1,1]`。
9. **BN1D 的 `H=W=1`**，但 cuDNN Frontend 中仍使用 4D `{N, C, 1, 1}`。

---

## 三、张量申请方案（BN 层 13 张量 + 全局 2 标量）

### 3.1 每个 BN 层的 13 张量

| 索引 | 名称 | Region | DType | NHWC 物理形状 | 含义 |
|:----:|------|--------|:-----:|---------------|------|
| 0 | `bn_weight` | `W_BN_WEIGHT` | FP32 | `[1,1,1,C]` | 可学习缩放 γ |
| 1 | `bn_bias` | `W_BN_BIAS` | FP32 | `[1,1,1,C]` | 可学习偏置 β |
| 2 | `bn_output` | `F_FEATURE_FP*` | FP32/FP16 | 同输入 | 主要输出 Y；BWD 时作为 dY 输入 |
| 3 | `bn_prev_mean` | `B_PREV_MEAN` | FP32 | `[1,1,1,C]` | 上一轮全局 running mean（融合算子路径输入） |
| 4 | `bn_prev_var` | `B_PREV_VAR` | FP32 | `[1,1,1,C]` | 上一轮全局 running variance（融合算子路径输入） |
| 5 | `bn_next_mean` | `B_NEXT_MEAN` | FP32 | `[1,1,1,C]` | 更新后 running mean（单算子路径原地更新） |
| 6 | `bn_next_var` | `B_NEXT_VAR` | FP32 | `[1,1,1,C]` | 更新后 running variance（单算子路径原地更新） |
| 7 | `bn_weight_grad` | `G_BN_WEIGHT` | FP32 | `[1,1,1,C]` | dγ（梯度） |
| 8 | `bn_bias_grad` | `G_BN_BIAS` | FP32 | `[1,1,1,C]` | dβ（梯度） |
| 9 | `bn_eq_bias` | `W_EQ_BIAS` | FP32 | `[1,1,1,C]` | 推理等效偏置 `β − μ × γ/√(σ²+ε)` |
| 10 | `bn_eq_scale` | `W_EQ_SCALE` | FP32 | `[1,1,1,C]` | 推理等效缩放 `γ/√(σ²+ε)` |
| 11 | `bn_saved_mean` | `T_TEMP_FP32` | FP32 | `[1,1,1,C]` | **新增**：FWD 保存的当前 batch 均值 μ，供 BWD 使用 |
| 12 | `bn_saved_inv_var` | `T_TEMP_FP32` | FP32 | `[1,1,1,C]` | **新增**：FWD 保存的当前 batch 逆标准差 `1/√(σ²+ε)`，供 BWD 使用 |

### 3.2 全局标量（Compiler baseline 申请，所有 BN 层共用）

| 名称 | Region | DType | 形状 | 含义 |
|------|--------|:-----:|------|------|
| `bn_epsilon` | `S_SCALAR_FP32` | FP32 | `[1,1,1,1]` | 数值稳定常数 ε，默认 1e-5 |
| `bn_momentum` | `S_SCALAR_FP32` | FP32 | `[1,1,1,1]` | EMA 动量，默认 0.1 |

### 3.3 与当前代码（11 张量）的差异

当前代码（`infer_bn_tensors` 函数）返回 11 张量（索引 0-10）。本方案新增 2 张量：

- **`bn_saved_mean`（索引 11）**：cuDNN `batchnorm` FWD 的 5 个输出之一（`saved_mean`），BWD 的 `batchnorm_backward` 必须使用此值。BN Finalize 同样输出此张量（名为 `MEAN`），为融合路径预留。
- **`bn_saved_inv_var`（索引 12）**：cuDNN `batchnorm` FWD 的 5 个输出之一（`saved_invariance`），BWD 必须使用。BN Finalize 同样输出此张量（名为 `INV_VARIANCE`），为融合路径预留。

**为什么放在 `T_TEMP_FP32` 而非专门分区？** 这两个张量不需要跨层批量操作（RANGE OP），只是单层内 FWD→BWD 的临时桥接。放在专门分区会浪费珍贵的连续排列区域。`T_TEMP_FP32` 是通用临时区，语义完全匹配。

### 3.4 不由 BN 层申请的张量

| 张量 | 谁申请 | 形状 | 原因 |
|------|--------|------|------|
| `d_sum` | 前置 Conv（如有） | `[1,1,1,C]` | GenStats 由 Conv 产生，BN 单算子不使用。若 BN 也申请会重复创建 |
| `d_sq_sum` | 前置 Conv（如有） | `[1,1,1,C]` | 同上 |
| `accum_count` | 前置 Conv（如有） | `[1,1,1,2]` INT32 | 同上。INT64 用两个 INT32 表示 |

---

## 四、LayerDescriptor 修改（`layer_descriptor_registry.cpp`）

### 4.1 `infer_bn_tensors` — 返回 13 张量

```cpp
std::vector<TensorDesc> infer_bn_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    (void)params;
    int ch = input.c();
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    Shape pshape{1, 1, 1, ch};

    // 0: weight
    { TensorDesc d; d.name="bn_weight"; d.shape=pshape; d.region=Region::W_BN_WEIGHT; d.dtype=DType::FP32; descs.push_back(d); }
    // 1: bias
    { TensorDesc d; d.name="bn_bias"; d.shape=pshape; d.region=Region::W_BN_BIAS; d.dtype=DType::FP32; descs.push_back(d); }
    // 2: output
    { TensorDesc d; d.name="bn_output"; d.shape=input; d.region=select_feature_region(ctx); d.dtype=feat_dt; descs.push_back(d); }
    // 3-6: 四大统计量（全部申请以保证 RANGE OP 对齐）
    { TensorDesc d; d.name="bn_prev_mean"; d.shape=pshape; d.region=Region::B_PREV_MEAN; d.dtype=DType::FP32; descs.push_back(d); }
    { TensorDesc d; d.name="bn_prev_var";  d.shape=pshape; d.region=Region::B_PREV_VAR;  d.dtype=DType::FP32; descs.push_back(d); }
    { TensorDesc d; d.name="bn_next_mean"; d.shape=pshape; d.region=Region::B_NEXT_MEAN; d.dtype=DType::FP32; descs.push_back(d); }
    { TensorDesc d; d.name="bn_next_var";  d.shape=pshape; d.region=Region::B_NEXT_VAR;  d.dtype=DType::FP32; descs.push_back(d); }
    // 7-8: 梯度
    { TensorDesc d; d.name="bn_weight_grad"; d.shape=pshape; d.region=Region::G_BN_WEIGHT; d.dtype=DType::FP32; descs.push_back(d); }
    { TensorDesc d; d.name="bn_bias_grad";   d.shape=pshape; d.region=Region::G_BN_BIAS;   d.dtype=DType::FP32; descs.push_back(d); }
    // 9-10: eq_bias/eq_scale — 始终为真实张量（推理必需）
    { TensorDesc d; d.name="bn_eq_bias";  d.shape=pshape; d.region=Region::W_EQ_BIAS;  d.dtype=DType::FP32; descs.push_back(d); }
    { TensorDesc d; d.name="bn_eq_scale"; d.shape=pshape; d.region=Region::W_EQ_SCALE; d.dtype=DType::FP32; descs.push_back(d); }
    // 11-12: saved_mean / saved_inv_var（新增，T_TEMP_FP32）
    { TensorDesc d; d.name="bn_saved_mean";    d.shape=pshape; d.region=Region::T_TEMP_FP32; d.dtype=DType::FP32; descs.push_back(d); }
    { TensorDesc d; d.name="bn_saved_inv_var"; d.shape=pshape; d.region=Region::T_TEMP_FP32; d.dtype=DType::FP32; descs.push_back(d); }

    return descs;  // 共 13 张量
}
```

**关键变更**：
- 新增索引 11（`bn_saved_mean`）和 12（`bn_saved_inv_var`），`T_TEMP_FP32` 区。
- 索引 9（`bn_eq_bias`）和 10（`bn_eq_scale`）**始终为真实张量**，不再使用 `ctx.bn_folded` 条件判断。

### 4.2 `build_bn_forward`

```cpp
SubgraphPattern build_bn_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 13) return p;   // 改为 13
    bool is_1d = (descs[2].shape.h() == 1 && descs[2].shape.w() == 1);
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    n.op = amp ? (is_1d ? ComputeOp::BN1D_AMP_FWD : ComputeOp::BN2D_AMP_FWD)
               : (is_1d ? ComputeOp::BN1D_FP32_FWD : ComputeOp::BN2D_FP32_FWD);
    // input_indices: weight(0), bias(1), next_mean(5), next_var(6)
    // Compiler 会在 begin 注入 X（前层输出），在末尾注入 bn_epsilon 和 bn_momentum
    n.input_indices  = {0, 1, 5, 6};
    // output_indices: output(2), saved_mean(11), saved_inv_var(12)
    // next_mean(5) 和 next_var(6) 由 cuDNN 原地更新，不在 output_indices 中显式列出
    n.output_indices = {2, 11, 12};
    p.nodes.push_back(n);
    return p;
}
```

**最终 GraphNode.input_ids 顺序**（Compiler 注入后）：
```
[0] X            (前层输出，Compiler 注入)
[1] weight       (索引 0)
[2] bias         (索引 1)
[3] next_mean    (索引 5)
[4] next_var     (索引 6)
[5] bn_epsilon   (baseline, Compiler 注入)
[6] bn_momentum  (baseline, Compiler 注入)
```

### 4.3 `build_bn_backward`

```cpp
SubgraphPattern build_bn_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 13) return p;   // 改为 13
    bool is_1d = (descs[2].shape.h() == 1 && descs[2].shape.w() == 1);
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    n.op = amp ? (is_1d ? ComputeOp::BN1D_AMP_BWD : ComputeOp::BN2D_AMP_BWD)
               : (is_1d ? ComputeOp::BN1D_FP32_BWD : ComputeOp::BN2D_FP32_BWD);
    // input_indices: weight(0), saved_mean(11), saved_inv_var(12)
    //   - Compiler 在 begin 注入 dY（prev_grad_id）
    //   - Compiler 在末尾追加 X（layer_input_ids[l]）——见 5.3 节
    //   - batchnorm_backward API 参数为 (loss, input, scale)，不需要 bias
    n.input_indices  = {0, 11, 12};
    // output_indices: 仅 weight_grad(7) 和 bias_grad(8)
    //   - dX 由 Compiler 特殊处理插入到 output_ids[0]（layer_input_ids[l]）
    n.output_indices = {7, 8};
    p.nodes.push_back(n);
    return p;
}
```

**关键解释**：
- BWD 不包含索引 2（bn_output）。`batchnorm_backward` 需要的是原始输入 X，不是 Y。X 由 Compiler 通过 `layer_input_ids[l]` 追加。
- 不包含 `prev_mean`/`prev_var`（索引 3、4），取而代之的是 FWD 输出的 `saved_mean`（索引 11）和 `saved_inv_var`（索引 12）。
- `batchnorm_backward` API 参数为 `(loss, input, scale)`，不需要 bias。

**最终 GraphNode.input_ids 顺序**（Compiler 注入后）：
```
[0] dY            (prev_grad_id, Compiler 注入)
[1] weight        (索引 0)
[2] saved_mean    (索引 11)
[3] saved_inv_var (索引 12)
[4] X             (layer_input_ids[l], Compiler 特殊处理追加)
```

**最终 GraphNode.output_ids 顺序**：
```
[0] dX            (layer_input_ids[l], in-place 覆盖 X)
[1] weight_grad   (索引 7)
[2] bias_grad     (索引 8)
```

### 4.4 `build_bn_inference`

```cpp
SubgraphPattern build_bn_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 13) return p;   // 改为 13
    bool is_1d = (descs[2].shape.h() == 1 && descs[2].shape.w() == 1);
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    n.op = amp ? (is_1d ? ComputeOp::BN1D_AMP_INF : ComputeOp::BN2D_AMP_INF)
               : (is_1d ? ComputeOp::BN1D_FP32_INF : ComputeOp::BN2D_FP32_INF);
    // INF 使用 Pointwise MUL(eq_scale) + ADD(eq_bias)
    // Compiler 在 begin 注入 X
    n.input_indices  = {10, 9};   // eq_scale, eq_bias
    n.output_indices = {2};       // output
    p.nodes.push_back(n);
    return p;
}
```

---

## 五、Compiler 层修改（`compiler.cpp`）

### 5.1 `get_grad_output_id` 修正

当前代码（[compiler.cpp:1043](file:///r:/renaissance/src/graph/compiler.cpp#L1043)）：
```cpp
case LayerKind::Bn1d: case LayerKind::Bn2d: idx = 2; break;  // dX inplace to bn_output
```

**修正为**：
```cpp
case LayerKind::Bn1d: case LayerKind::Bn2d: idx = -1; break;  // dX in-place to X, handled by special logic
```

### 5.2 `get_grad_output_id` 的 fallback 列表中加入 Bn1d/Bn2d

当前代码（[compiler.cpp:1444-1452](file:///r:/renaissance/src/graph/compiler.cpp#L1444-L1452)）的 fallback 列表不包含 Bn1d/Bn2d。修正为：

```cpp
if (grad_id < 0 && (layer.kind == LayerKind::FC ||
    layer.kind == LayerKind::Conv ||
    layer.kind == LayerKind::Bn1d || layer.kind == LayerKind::Bn2d ||  // [NEW]
    layer.kind == LayerKind::FCBNReLU || layer.kind == LayerKind::GapFC ||
    // ... 其余层保持不变 ...
    layer.kind == LayerKind::Flatten)) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) grad_id = it->second;
}
```

### 5.3 BWD 构建循环中新增 BN 特殊处理块

在 BWD 构建循环中，在所有其他层的特殊处理（FC/Conv、Tanh/ReLU、Flatten、SoftmaxCE 等）完成之后、`train_cg.append` 之前，新增：

```cpp
// BN BWD: 追加原始输入 X，dX in-place 覆盖 X
if (gn.compute_op == ComputeOp::BN1D_FP32_BWD || gn.compute_op == ComputeOp::BN1D_AMP_BWD ||
    gn.compute_op == ComputeOp::BN2D_FP32_BWD || gn.compute_op == ComputeOp::BN2D_AMP_BWD) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) {
        gn.input_ids.push_back(it->second);                      // 追加 X
        gn.output_ids.insert(gn.output_ids.begin(), it->second); // dX in-place to X (output_ids[0])
    }
}
```

### 5.4 BN FWD 注入全局标量

在 FWD 构建循环中（[compiler.cpp:1136-1248](file:///r:/renaissance/src/graph/compiler.cpp#L1136-L1248)），在 SoftmaxCE 注入之后、`train_cg.append` 之前，新增：

```cpp
// BN FWD: 注入全局标量 bn_epsilon 和 bn_momentum
if (is_bn_fwd_op(gn.compute_op)) {
    const auto& b = memory_plan.baseline();
    if (b.bn_epsilon >= 0)  gn.input_ids.push_back(b.bn_epsilon);
    if (b.bn_momentum >= 0) gn.input_ids.push_back(b.bn_momentum);
}
```

辅助函数：

```cpp
inline bool is_bn_fwd_op(ComputeOp op) {
    return op == ComputeOp::BN1D_FP32_FWD || op == ComputeOp::BN1D_AMP_FWD ||
           op == ComputeOp::BN2D_FP32_FWD || op == ComputeOp::BN2D_AMP_FWD;
}
```

### 5.5 `RANGE_BN_STATS_ALLREDUCE` — 扩展为四个区

当前代码（[compiler.cpp:1742-1748](file:///r:/renaissance/src/graph/compiler.cpp#L1742-L1748)）仅同步 `B_NEXT_MEAN → B_NEXT_VAR`（2 个区）：

```cpp
MemRange r_bn = memory_plan.region_range(
    Region::B_NEXT_MEAN, Region::B_NEXT_VAR);
train_cg.append_range(GraphId::STATS_COMM,
    RangeOp::RANGE_BN_STATS_ALLREDUCE, {r_bn}, {r_bn});
```

**修正为**：

```cpp
if (has_bn) {
    MemRange r_prev = memory_plan.region_range(Region::B_PREV_MEAN, Region::B_PREV_VAR);
    MemRange r_next = memory_plan.region_range(Region::B_NEXT_MEAN, Region::B_NEXT_VAR);
    train_cg.append_range(GraphId::STATS_COMM,
        RangeOp::RANGE_BN_STATS_ALLREDUCE, {r_prev, r_next}, {r_prev, r_next});
}
```

**理由**：
- 融合算子路径的有效输入在 `B_PREV_MEAN` / `B_PREV_VAR`，必须同步。
- 单算子路径的有效输入在 `B_NEXT_MEAN` / `B_NEXT_VAR`，必须同步。
- 如果只同步 `B_NEXT_*`，融合算子缺少同步；只同步 `B_PREV_*`，单算子缺少同步。
- 必须四个区都同步，才能同时覆盖两条路径。

如果 `RANGE_BN_STATS_ALLREDUCE` 当前实现不支持多 Range 输入，可拆分为两次独立的 `append_range` 调用。

### 5.6 `RANGE_D2D_COPY` — 保持不变

当前代码（[compiler.cpp:2067-2086](file:///r:/renaissance/src/graph/compiler.cpp#L2067-L2086)）已正确实现 `B_NEXT_MEAN → B_PREV_MEAN`、`B_NEXT_VAR → B_PREV_VAR`，无需修改。此逻辑同时服务于单算子路径和融合算子路径。

### 5.7 BN 不同层 epsilon/momentum 不一致校验

在 Compiler 中新增校验：

```cpp
float first_eps = -1.0f, first_mom = -1.0f;
for (const auto& layer : arch.layers()) {
    if (layer.kind == LayerKind::Bn1d || layer.kind == LayerKind::Bn2d) {
        const auto& bp = std::get<BNParams>(layer.params);
        if (first_eps < 0) { first_eps = bp.eps; first_mom = bp.momentum; }
        else if (bp.eps != first_eps || bp.momentum != first_mom) {
            TR_COMPILE_ERROR("All BN layers must share the same epsilon and momentum "
                             "for RANGE OP batch operations.");
        }
    }
}
```

---

## 六、MemoryPlan 层修改（`memory_plan.h` / `memory_plan.cpp`）

### 6.1 `BaselineIds` 新增字段

在 [memory_plan.h:152-177](file:///r:/renaissance/include/renaissance/graph/memory_plan.h#L152-L177) 中新增：

```cpp
struct BaselineIds {
    // ... 现有字段 ...
    int32_t bn_epsilon  = -1;   // [NEW] 全局 BN epsilon，S_SCALAR_FP32
    int32_t bn_momentum = -1;   // [NEW] 全局 BN momentum，S_SCALAR_FP32
};
```

### 6.2 `alloc_baseline_dtensors` 中无条件申请全局标量

两个 `[1,1,1,1]` FP32 标量仅占 8 字节，开销可忽略。无条件申请可避免签名变更和条件判断的复杂性。

在两个 `alloc_baseline_dtensors` 重载的末尾（Dropout seed 之后）新增：

```cpp
// Step 7: BN 全局标量（无条件申请，开销极小）
Shape scalar_shape{1, 1, 1, 1};
baseline_.bn_epsilon  = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
baseline_.bn_momentum = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
```

### 6.3 `init_variant_scalars` 中初始化标量

在 `deep_learning_task.cpp` 的 `init_variant_scalars` 中新增：

```cpp
if (b.bn_epsilon >= 0) {
    float eps = 1e-5f;  // 从用户配置读取
    for (int rank = 0; rank < num_gpus_; ++rank) {
        // ... 遍历 variant ...
        cudaMemcpy(ctx.ptr_at(b.bn_epsilon), &eps, sizeof(float), cudaMemcpyHostToDevice);
    }
}
if (b.bn_momentum >= 0) {
    float mom = 0.1f;   // 从用户配置读取
    for (int rank = 0; rank < num_gpus_; ++rank) {
        // ... 遍历 variant ...
        cudaMemcpy(ctx.ptr_at(b.bn_momentum), &mom, sizeof(float), cudaMemcpyHostToDevice);
    }
}
```

---

## 七、GPU 后端实现（cuDNN Frontend）

### 7.1 文件组织

- 新建 `src/backend/ops/dtensor/bn_op.cpp`（CUDA 实现）
- 在 `src/backend/op_registry.cpp` 的 `register_default_ops()` 中调用 `register_op_bn()`

### 7.2 FWD — cuDNN Frontend `batchnorm` API

```cpp
// 输出顺序: [Y, saved_mean, saved_invariance, next_running_mean, next_running_variance]
std::array<std::shared_ptr<Tensor_attributes>, 5> batchnorm(
    std::shared_ptr<Tensor_attributes>& input,    // [N, C, H, W], NHWC stride
    std::shared_ptr<Tensor_attributes>& scale,    // [1, C, 1, 1], FP32
    std::shared_ptr<Tensor_attributes>& bias,     // [1, C, 1, 1], FP32
    Batchnorm_attributes attributes);
```

**图构建**：

```cpp
auto graph = std::make_shared<fe::graph::Graph>();
auto dt = is_fp16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT;
graph->set_io_data_type(dt).set_compute_data_type(fe::DataType_t::FLOAT);

auto X = graph->tensor(fe::graph::Tensor_attributes()
    .set_name("bn_x")
    .set_dim({N, C, H, W})
    .set_stride({int64_t(H)*W*C, 1, int64_t(W)*C, C})
    .set_data_type(dt));

auto make_param = [&](const char* name) {
    return graph->tensor(fe::graph::Tensor_attributes()
        .set_name(name)
        .set_dim({1, C, 1, 1})
        .set_stride({C, 1, C, C})
        .set_data_type(fe::DataType_t::FLOAT));
};

auto scale     = make_param("bn_scale");
auto bias      = make_param("bn_bias");
auto prev_mean = make_param("bn_prev_mean");   // 运行时绑定到 B_NEXT_MEAN
auto prev_var  = make_param("bn_prev_var");    // 运行时绑定到 B_NEXT_VAR

auto epsilon  = graph->tensor(fe::graph::Tensor_attributes()
    .set_name("bn_epsilon")
    .set_dim({1, 1, 1, 1})
    .set_stride({1, 1, 1, 1})
    .set_data_type(fe::DataType_t::FLOAT));
auto momentum = graph->tensor(fe::graph::Tensor_attributes()
    .set_name("bn_momentum")
    .set_dim({1, 1, 1, 1})
    .set_stride({1, 1, 1, 1})
    .set_data_type(fe::DataType_t::FLOAT));

auto bn_opts = fe::graph::Batchnorm_attributes()
    .set_previous_running_stats(prev_mean, prev_var, momentum)
    .set_epsilon(epsilon)
    .set_compute_data_type(fe::DataType_t::FLOAT);

auto [Y, saved_mean, saved_inv_var, next_mean, next_var] =
    graph->batchnorm(X, scale, bias, bn_opts);

Y->set_output(true).set_data_type(dt);
saved_mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
saved_inv_var->set_output(true).set_data_type(fe::DataType_t::FLOAT);
next_mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
next_var->set_output(true).set_data_type(fe::DataType_t::FLOAT);
```

**VariantPack 绑定（关键！）**：

```cpp
// 单算子路径：prev 和 next 绑定到同一个 buffer（B_NEXT_MEAN / B_NEXT_VAR）
// 实现原地更新，与 Backend API cudnnBatchNormalizationForwardTraining 语义一致
vp[X]            = d_bn_input;        // F_FEATURE_*
vp[scale]        = d_scale;           // W_BN_WEIGHT
vp[bias]         = d_bias;            // W_BN_BIAS
vp[prev_mean]    = d_next_mean;       // B_NEXT_MEAN  ← prev 绑定到 next 的 buffer
vp[prev_var]     = d_next_var;        // B_NEXT_VAR   ← 同上
vp[epsilon]      = d_bn_epsilon;      // S_SCALAR_FP32 (baseline)
vp[momentum]     = d_bn_momentum;     // S_SCALAR_FP32 (baseline)

vp[Y]            = d_bn_output;       // F_FEATURE_*
vp[saved_mean]   = d_saved_mean;      // T_TEMP_FP32
vp[saved_inv_var] = d_saved_inv_var;  // T_TEMP_FP32
vp[next_mean]    = d_next_mean;       // B_NEXT_MEAN（与 prev 同一个 pointer）
vp[next_var]     = d_next_var;        // B_NEXT_VAR（与 prev 同一个 pointer）
```

### 7.3 BWD — cuDNN Frontend `batchnorm_backward` API

```cpp
// 输出顺序: [dX, dscale, dbias]
std::array<std::shared_ptr<Tensor_attributes>, 3> batchnorm_backward(
    std::shared_ptr<Tensor_attributes>& loss,     // dY, [N, C, H, W]
    std::shared_ptr<Tensor_attributes>& input,    // X,  [N, C, H, W]
    std::shared_ptr<Tensor_attributes>& scale,    // γ,  [1, C, 1, 1], FP32
    Batchnorm_backward_attributes attributes);
```

**图构建**：

```cpp
auto dY = graph->tensor(...);  // [N,C,H,W], dt
auto X  = graph->tensor(...);  // [N,C,H,W], dt
auto scale = make_param("bn_scale");
auto saved_mean = make_param("bn_saved_mean");
auto saved_inv_var = make_param("bn_saved_inv_var");

auto dbn_opts = fe::graph::Batchnorm_backward_attributes()
    .set_saved_mean_and_inv_variance(saved_mean, saved_inv_var)
    .set_compute_data_type(fe::DataType_t::FLOAT);

auto [dX, dscale, dbias] = graph->batchnorm_backward(dY, X, scale, dbn_opts);

dX->set_output(true).set_data_type(dt);
dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);
```

**VariantPack 绑定**：

```cpp
vp[dY]           = d_dy;              // input_ids[0] — 上游梯度
vp[X]            = d_x;               // input_ids[4] — X (layer_input_ids[l])
vp[scale]        = d_weight;          // input_ids[1] — W_BN_WEIGHT
vp[saved_mean]   = d_saved_mean;      // input_ids[2] — T_TEMP_FP32
vp[saved_inv_var] = d_saved_inv_var;  // input_ids[3] — T_TEMP_FP32

vp[dX]           = d_x;               // dX 覆盖 X！（layer_input_ids[l]）
vp[dscale]       = d_weight_grad;     // G_BN_WEIGHT
vp[dbias]        = d_bias_grad;       // G_BN_BIAS
```

### 7.4 INF — Pointwise MUL + ADD

```cpp
auto X = graph->tensor(...);  // [N,C,H,W], dt
auto eq_scale = make_param("eq_scale");  // W_EQ_SCALE
auto eq_bias  = make_param("eq_bias");   // W_EQ_BIAS

auto scaled = graph->pointwise(X, eq_scale,
    fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));
scaled->set_data_type(dt);

auto Y = graph->pointwise(scaled, eq_bias,
    fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
Y->set_output(true).set_data_type(dt);
```

### 7.5 BN1D vs BN2D 的唯一差异

| 参数 | BN1D | BN2D |
|------|------|------|
| 输入形状（cuDNN dim） | `{N, C, 1, 1}` | `{N, C, H, W}` |
| 输入形状（物理） | `[N, 1, 1, C]` | `[N, H, W, C]` |

其余所有代码（图构建、VariantPack 绑定、张量布局）完全相同。

### 7.6 流分配

BN 已分配到 `COMP_2`（[op_stream_policy.cpp:47-54](file:///r:/renaissance/src/backend/op_stream_policy.cpp#L47-L54)），无需修改。

### 7.7 图缓存策略

参考 `conv_op.cpp` 的缓存模式，建立 BN 专用的图缓存。缓存 key 包含：`handle_ptr, N, H, W, C, is_fp16, is_1d`。

### 7.8 算子注册

```cpp
void register_op_bn() {
    auto& table = g_compute_op_table;

    // FP32 FWD
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_FWD)].launch_cpu = launch_bn_fp32_fwd_cpu;
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_FWD)].launch_cuda = launch_bn_fp32_fwd_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_FWD)].launch_cpu = launch_bn_fp32_fwd_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_FWD)].launch_cuda = launch_bn_fp32_fwd_cuda;

    // FP32 BWD
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_BWD)].launch_cpu = launch_bn_fp32_bwd_cpu;
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_BWD)].launch_cuda = launch_bn_fp32_bwd_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_BWD)].launch_cpu = launch_bn_fp32_bwd_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_BWD)].launch_cuda = launch_bn_fp32_bwd_cuda;

    // FP32 INF
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_INF)].launch_cpu = launch_bn_fp32_inf_cpu;
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_INF)].launch_cuda = launch_bn_fp32_inf_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_INF)].launch_cpu = launch_bn_fp32_inf_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_INF)].launch_cuda = launch_bn_fp32_inf_cuda;

    // AMP FWD（CPU 不支持，报错）
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_FWD)].launch_cpu = launch_bn_amp_not_supported_cpu;
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_FWD)].launch_cuda = launch_bn_amp_fwd_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_FWD)].launch_cpu = launch_bn_amp_not_supported_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_FWD)].launch_cuda = launch_bn_amp_fwd_cuda;

    // AMP BWD
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_BWD)].launch_cpu = launch_bn_amp_not_supported_cpu;
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_BWD)].launch_cuda = launch_bn_amp_bwd_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_BWD)].launch_cpu = launch_bn_amp_not_supported_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_BWD)].launch_cuda = launch_bn_amp_bwd_cuda;

    // AMP INF
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_INF)].launch_cpu = launch_bn_amp_not_supported_cpu;
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_INF)].launch_cuda = launch_bn_amp_inf_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_INF)].launch_cpu = launch_bn_amp_not_supported_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_INF)].launch_cuda = launch_bn_amp_inf_cuda;
}
```

---

## 八、CPU 后端实现

### 8.1 FWD

**关键绑定：`running_mean`/`running_var` 参数绑定到 `B_NEXT_MEAN`/`B_NEXT_VAR`（与 GPU 一致），而非 `B_PREV_MEAN`/`B_PREV_VAR`。** 否则 CPU 路径会错误地更新 prev 区，导致下一轮被 RANGE_D2D_COPY 误覆盖。

```cpp
void bn_fwd_cpu(
    const float* x, float* y,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float* saved_mean, float* saved_inv_var,
    int N, int H, int W, int C,
    float eps, float momentum)
{
    int spatial = N * H * W;
    for (int c = 0; c < C; ++c) {
        // 1. 计算 batch 均值
        double sum = 0;
        for (int i = 0; i < spatial; ++i)
            sum += x[i * C + c];
        float mean = sum / spatial;
        saved_mean[c] = mean;

        // 2. 计算 batch 方差
        double sq_sum = 0;
        for (int i = 0; i < spatial; ++i) {
            float diff = x[i * C + c] - mean;
            sq_sum += diff * diff;
        }
        float var = sq_sum / spatial;
        float inv_std = 1.0f / std::sqrt(var + eps);
        saved_inv_var[c] = inv_std;

        // 3. 更新 running stats（原地更新）
        // momentum 在 cuDNN 中作为 exponentialAverageFactor：
        //   running = (1-momentum) * old + momentum * batch
        // 当 momentum=0.1 时: running = 0.9 * old + 0.1 * batch（平滑 EMA）
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
        running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var;

        // 4. 归一化 + 仿射
        for (int i = 0; i < spatial; ++i)
            y[i * C + c] = gamma[c] * (x[i * C + c] - mean) * inv_std + beta[c];
    }
}
```

### 8.2 BWD

```cpp
void bn_bwd_cpu(
    const float* dy, const float* x,
    const float* gamma,
    const float* saved_mean, const float* saved_inv_var,
    float* dx, float* dgamma, float* dbeta,
    int N, int H, int W, int C)
{
    int spatial = N * H * W;
    for (int c = 0; c < C; ++c) {
        float mean = saved_mean[c];
        float inv_std = saved_inv_var[c];

        double dy_sum = 0, dy_xmu_sum = 0;
        for (int i = 0; i < spatial; ++i) {
            int idx = i * C + c;
            dy_sum += dy[idx];
            dy_xmu_sum += dy[idx] * (x[idx] - mean);
        }

        dgamma[c] = dy_xmu_sum * inv_std;
        dbeta[c] = dy_sum;

        float inv_spatial = 1.0f / spatial;
        for (int i = 0; i < spatial; ++i) {
            int idx = i * C + c;
            float x_mu = x[idx] - mean;
            dx[idx] = gamma[c] * inv_std * (
                dy[idx]
                - inv_spatial * dy_sum
                - inv_spatial * dy_xmu_sum * inv_std * inv_std * x_mu
            );
        }
    }
}
```

### 8.3 INF

```cpp
void bn_inf_cpu(
    const float* x, float* y,
    const float* eq_scale, const float* eq_bias,
    int N, int H, int W, int C)
{
    int spatial = N * H * W;
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < spatial; ++i)
            y[i * C + c] = x[i * C + c] * eq_scale[c] + eq_bias[c];
}
```

### 8.4 Eigen3 优化

用 `TR_USE_EIGEN` 宏包裹。对每个 channel 用 Eigen 的 `array` 操作加速。若 Eigen3 不可用或形状不匹配，自动 fallback 到朴素循环。

---

## 九、eq_scale / eq_bias 预计算

### 9.1 公式

```
eq_scale[c] = gamma[c] / sqrt(running_var[c] + epsilon)
eq_bias[c]  = beta[c] - running_mean[c] * eq_scale[c]
```

### 9.2 生成策略

**在 BN INF 算子内部首次执行时生成**。

理由：
1. 实现简单，不改动 Compiler 和 Graph 结构。
2. 每个 BN 层首次推理有额外开销（约 < 100μs），可忽略。
3. RANGE OP 无法跨 Region（`W_BN_WEIGHT`、`B_PREV_VAR`、`S_SCALAR_FP32`）、变长度地执行 `gamma / sqrt(var + eps)` 这类复杂计算。

**实现**：在 BN INF 的 `launch_cuda` / `launch_cpu` 中，检查一个 layer-local `bool eq_ready` 标志。若未生成，从 `gamma`（W_BN_WEIGHT）、`beta`（W_BN_BIAS）、`running_mean`（B_PREV_MEAN 或 B_NEXT_MEAN，训练结束后两者内容相同）、`running_var`（B_PREV_VAR 或 B_NEXT_VAR）、`bn_epsilon`（S_SCALAR_FP32）计算 `eq_scale` 和 `eq_bias`，写入 `W_EQ_SCALE` 和 `W_EQ_BIAS` 区。

---

## 十、初始化

| 张量 | 初始化值 | 说明 |
|:---|:---|:---|
| `bn_weight` (γ) | 1.0f | 标准 BN 初始化 |
| `bn_bias` (β) | 0.0f | 标准 BN 初始化 |
| `bn_prev_mean` | 0.0f | 初始 running mean |
| `bn_prev_var` | 1.0f | 初始 running var |
| `bn_next_mean` | 0.0f | 与 prev 一致 |
| `bn_next_var` | 1.0f | 与 prev 一致 |
| `bn_eq_scale` | 0.0f | 首次推理时由 INF 算子重新计算 |
| `bn_eq_bias` | 0.0f | 首次推理时由 INF 算子重新计算 |
| `bn_saved_mean` | 0.0f | 临时量，运行时覆写 |
| `bn_saved_inv_var` | 0.0f | 临时量，运行时覆写 |
| `bn_epsilon` (全局) | 1e-5f | H2D 初始化 |
| `bn_momentum` (全局) | 0.1f | H2D 初始化 |

---

## 十一、关键设计决策与理由

### 11.1 为什么 running stats 用 B_NEXT 而非 B_PREV？

cuDNN 的 `batchnorm` 在 Backend API 层面是原地更新 running stats。框架每 batch 执行 `RANGE_D2D_COPY(B_NEXT → B_PREV)`。如果 cuDNN 更新 B_PREV，会被 RANGE_D2D_COPY 用未赋值的 B_NEXT 覆盖，导致数据破坏。使用 B_NEXT 则安全：cuDNN 更新 B_NEXT → RANGE_D2D_COPY 将正确的 B_NEXT 复制到 B_PREV。

### 11.2 为什么 BWD 用 `saved_mean`/`saved_inv_var` 而非 `prev_mean`/`prev_var`？

`prev_mean`/`prev_var` 是全局 EMA 累积统计量（跨 batch），而 BWD 需要的是当前 batch 的统计量。BatchNorm FWD 输出 `saved_mean`（当前 batch 的 μ）和 `saved_invariance`（当前 batch 的 1/σ），BWD 必须使用这些值。

### 11.3 为什么 BWD 的 `input_indices` 不含索引 2（bn_output）？

`batchnorm_backward` 的第二个参数需要的是原始输入 X，不是 Y（BN 输出）。X 是前一层输出（`layer_input_ids[l]`），bn_output 是本层输出 Y，两者是不同的 DTensor。正确的 X 由 Compiler 通过 `layer_input_ids[l]` 追加到 `input_ids` 末尾。

### 11.4 为什么 dX 必须覆盖 X 而非 bn_output？

框架铁律 "dX 覆盖 X"。dX 是前一层 BWD 的 dY，必须写入前一层输出（即本层输入 X）的 buffer。写入 bn_output 会破坏梯度链。

### 11.5 为什么 BN1D 也要申请全部 13 张量？

RANGE OP 批量操作要求所有 BN 层的张量一一对应。如果 BN1D 少申请某张量，会导致 RANGE OP 遍历时张量错位，引发极其隐蔽的 bug。浪费的 KB 级显存微不足道。

### 11.6 为什么 `bn_epsilon` 和 `bn_momentum` 是全局的？

- 所有 BN 层应使用相同的 ε 和 momentum，这是标准做法。
- 全局共享使得 RANGE OP 可以统一操作，无需逐层处理。
- 如果允许逐层不同，无法使用 RANGE OP 批量更新。如果用户配置了不同值，Compiler 应报错。

### 11.7 为什么 INF 用 Pointwise MUL+ADD 而非 `batchnorm` 推理 API？

- 推理时 BN 已化简为仿射变换 `Y = X * eq_scale + eq_bias`，无需访问 running stats。
- `eq_scale` 和 `eq_bias` 在训练收敛后预计算一次，之后直接使用。

---

## 十二、实施清单

### Phase 1：LayerDescriptor（`layer_descriptor_registry.cpp`）
- [ ] `infer_bn_tensors`：返回 13 张量（新增 saved_mean/saved_inv_var），eq_scale/eq_bias 始终为真实张量
- [ ] `build_bn_forward`：`descs.size() < 13`，`input_indices = {0, 1, 5, 6}`，`output_indices = {2, 11, 12}`
- [ ] `build_bn_backward`：`descs.size() < 13`，`input_indices = {0, 11, 12}`，`output_indices = {7, 8}`
- [ ] `build_bn_inference`：`descs.size() < 13`，`input_indices = {10, 9}`，`output_indices = {2}`

### Phase 2：Compiler（`compiler.cpp`）
- [ ] `get_grad_output_id`：Bn1d/Bn2d 从 `2` 改为 `-1`
- [ ] `get_grad_output_id` fallback 列表：加入 `Bn1d`/`Bn2d`
- [ ] BWD 构建循环：新增 BN BWD 特殊处理块（追加 X，dX in-place 到 X）
- [ ] FWD 构建循环：新增 BN 全局标量注入（`bn_epsilon`、`bn_momentum`）
- [ ] `RANGE_BN_STATS_ALLREDUCE`：范围从 2 区扩展为 4 区
- [ ] 新增 BN 不同层 epsilon/momentum 不一致校验

### Phase 3：MemoryPlan（`memory_plan.h` + `.cpp`）
- [ ] `BaselineIds` 新增 `bn_epsilon`、`bn_momentum` 字段
- [ ] `alloc_baseline_dtensors` 中**无条件**申请这两个标量（`S_SCALAR_FP32`，`Shape{1,1,1,1}`）

### Phase 4：Task 层（`deep_learning_task.cpp`）
- [ ] `init_variant_scalars` 中 H2D 初始化 `bn_epsilon` 和 `bn_momentum`

### Phase 5：后端实现
- [ ] 新建 `src/backend/ops/dtensor/bn_op.cpp`：GPU FWD/BWD/INF（cuDNN Frontend）
- [ ] 实现 CPU FWD/BWD/INF（朴素循环 + Eigen3 可选）
- [ ] 实现 AMP 不支持 CPU 的报错 stub
- [ ] 实现图缓存（参考 Conv 模式）
- [ ] 实现 BN INF 内部的 `eq_scale`/`eq_bias` 生成逻辑
- [ ] 在 `op_registry.cpp` 中调用 `register_op_bn()`

### Phase 6：初始化
- [ ] `initializer.cpp`：BN 的 γ=1, β=0；running_mean=0, running_var=1；saved_mean/saved_inv_var=0；eq_scale/eq_bias=0

### Phase 7：测试
- [ ] 与 PyTorch `nn.BatchNorm1d` / `nn.BatchNorm2d` 对比数值（MSE < 1e-4）
- [ ] 验证 running stats 是否正确更新（连续多个 batch 观察 running_mean 变化）
- [ ] 验证多 RANK 下统计量同步后数值一致
- [ ] 验证 BN INF 输出与 PyTorch `eval()` 模式一致（MSE < 1e-4）
- [ ] 验证 dX 覆盖 X（BWD 后 `layer_input_ids[l]` buffer 内容为 dX）
- [ ] 端到端梯度链测试：构建 Conv→BN→FC 小网络，验证 BWD 梯度能从 FC 正确传回 Conv，中间 BN 的 dX 正确覆盖 X 并被前层 Conv 作为 dY 消费
- [ ] A/B 双缓冲测试：验证 BN running stats 在双缓冲切换后仍然正确（如果框架使用 A/B 数据双缓冲）

### Phase 8：融合层兼容性（`layer_descriptor_registry.cpp`）
`infer_bn_tensors` 从 11 张量扩展为 13 张量后，以下融合层的 `descs.size()` 检查和 ReLU mask 索引需同步更新（融合层未来将重写，本次仅做最小兼容性修改）：

| 融合层 | 原总张量 | 新总张量 | `descs.size()` 检查 | mask 索引变化 |
|--------|:--------:|:--------:|---------------------|---------------|
| BNReLU | 12 | 14 | `< 12` → `< 14` | 11 → 13 |
| ConvBN | 17 | 19 | `< 17` → `< 19` | 无 mask |
| CBR | 18 | 20 | `< 18` → `< 20` | 17 → 19 |
| FCBNReLU | 19 | 21 | `< 19` → `< 21` | 18 → 20 |

- [ ] `infer_bnrelu_tensors` / `build_bnrelu_*`：`descs.size() < 12` → `< 14`，mask 索引 11 → 13
- [ ] `infer_convbn_tensors` / `build_convbn_*`：`descs.size() < 17` → `< 19`
- [ ] `infer_convbnrelu_tensors` / `build_convbnrelu_*`：`descs.size() < 18` → `< 20`，mask 索引 17 → 19
- [ ] `infer_fcbnrelu_tensors` / `build_fcbnrelu_*`：`descs.size() < 19` → `< 21`，mask 索引 18 → 20

> **注意**: Bottleneck 层（`BottleneckProjection`、`BottleneckIdentity`）内部使用 `infer_bn_tensors`/`infer_convbnrelu_tensors`/`infer_convbn_tensors`，其总张量数也会相应增加（BottleneckProjection: 71→79，BottleneckIdentity: 54→60），但硬编码的 `build_bottleneck_*` 内部索引极为复杂，将在融合层重写时一并修正。当前仅需更新 `descs.size()` 检查，确保不会因检查失败而 crash。

---

## 十三、附录：cuDNN Frontend API 关键信息

### FWD（`batchnorm`）
- 输出顺序：`[Y, saved_mean, saved_invariance, next_running_mean, next_running_variance]`
- 输入：`(input, scale, bias)` + `Batchnorm_attributes`（含 `set_previous_running_stats`, `set_epsilon`）
- `set_previous_running_stats(prev_rm, prev_rv, momentum)` 中 `prev_rm`/`prev_rv` 和返回的 `next_rm`/`next_rv` 可以使用同一个 `Tensor_attributes`（原地更新），cuDNN Frontend 支持。

### BWD（`batchnorm_backward`）
- 输出顺序：`[dX, dscale, dbias]`
- 输入：`(loss, input, scale)` + `Batchnorm_backward_attributes`（含 `set_saved_mean_and_inv_variance`）
- `set_saved_mean_and_inv_variance(saved_mean, saved_inv_variance)` 使用 FWD 输出的 `saved_mean` 和 `saved_invariance`。

### BN Finalize（融合路径，当前不使用）
- 输出顺序：`[EQ_SCALE, EQ_BIAS, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR]`
- 其中 MEAN 对应 BatchNorm 的 `saved_mean`，INV_VARIANCE 对应 BatchNorm 的 `saved_invariance`。

---

*本方案综合了 DTY0.md 需求说明、DTY1.md~DTY4.md 四份分析方案、B_REV.md 审查意见，以及现有框架代码的全面审计。所有关键设计决策均有充分理由，裁决了各方案之间的分歧，与现有代码状态完全对齐。*
