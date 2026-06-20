# CBR（Conv+BN+ReLU）融合算子 AMP 实现方案（最终版）

> 综合 EBH0/EBH1/EBH2/EBH3 四版方案与当前代码实际状态，经关键事实核查后形成的最终可实施方案。本轮仅输出方案，不修改代码。

---

## 1. 目标与范围

### 1.1 新增算子（仅 AMP）

| 算子 | 语义 |
|------|------|
| `CBR_AMP_FWD` | Conv AMP FWD → BN2D AMP FWD → ReLU AMP FWD |
| `CBR_AMP_BWD` | ReLU AMP BWD → BN2D AMP BWD → Conv AMP BWD（含 dX） |
| `CBR_AMP_BWD_FIRST_LAYER` | ReLU AMP BWD → BN2D AMP BWD → Conv AMP BWD_FIRST_LAYER（仅 wgrad） |
| `CBR_AMP_INF` | Conv AMP INF → BN2D AMP INF → ReLU AMP INF |

### 1.2 明确不实现

- GPU FP32 版、CPU 版 CBR。
- 非 AMP 路径下，`cbr()` / `conv_bn_relu()` 自动拆分为 `Conv + Bn2d + ReLU`。

### 1.3 BluePrint 与 ArchPlan 行为

- `cbr(out_ch, k, s, p, momentum=0.1, eps=1e-5)` 与 `conv_bn_relu(out_ch, k, s, p, momentum=0.1, eps=1e-5)` 为别名，生成相同 `NodeKind::CBR` 节点。参数风格与 `bn(momentum, eps)` 一致，`momentum` 在前。
- AMP（`fuse=true`）下直接生成 `LayerKind::CBR`。
- FP32/CPU 下展开为 `Conv + Bn2d + ReLU` 三个独立层。
- AMP 下，连续的 `Conv + Bn2d + ReLU` 在 `step9_merge_triple` 自动合并为 `LayerKind::CBR`。
- `cbr` 可作为首层。
- `expand_primitive_impl` 对 `NodeKind::CBR`（用户显式调用 `cbr()`/`conv_bn_relu()`）在 `fuse=true` 时直接 emit `LayerKind::CBR`，不会拆成三个 primitive 层；`step9_merge_triple` 仅处理用户显式写的 `Conv; Bn2d; ReLU;` primitive 序列。两个路径互不重叠，避免重复合并。

---

## 2. 关键事实核查与方案定型

### 2.1 ReLU AMP 是 out-of-place，不是 in-place

经代码核查，分立路径中 ReLU 有独立的输出张量：

- `infer_relu_tensors` 返回 `{relu_output, relu_mask}` 两个张量。
- `build_relu_forward` 的 `output_indices = {0, 1}`，没有 `input_indices`。
- `launch_relu_amp_fwd_cuda` 中：
  ```cpp
  __half* x = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));   // BN 输出
  __half* y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));  // ReLU 自己的输出
  ```
- CUDA kernel 签名：`relu_fwd_amp_mask_kernel(const __half* x, __half* y, ...)`，`x` 与 `y` 不重叠。

**结论**：分立路径的真实数据流为 `X → conv_output → bn_output → relu_output`，三个特征图是**三个独立张量**。CBR 必须保留这三个独立张量，ReLU 不能 in-place 到 `bn_output`。

### 2.2 不能复用 `infer_convbnrelu_tensors`

当前 `infer_convbnrelu_tensors` 使用 `infer_conv_tensors`（6 张量），缺少 `bn_sum`/`bn_sq_sum`。而 `CONV_AMP_FWD` 的 `build_conv_forward` 输出索引为 `{1, 6, 7}`，必须包含这两个统计量。因此该函数不能直接用于独立 CBR 层。

### 2.3 最终选型

| 议题 | 选择 | 理由 |
|------|------|------|
| 张量布局 | **23 张量** | Conv(8) + BN(13) + ReLU output(1) + ReLU mask(1)，与分立路径三个独立特征图对齐 |
| LayerParam | **`CbrLayerParams`** | 独立于 `OpParams` 的 `CBRParams`，字段风格与 `ConvLayerParams` 一致，且保留 eps/momentum |
| 后端实现 | **复制核心逻辑** | 不修改现有 static launch 函数，隔离性好；LKW.md 明确允许“复制单算子代码” |
| 流策略 | **内部 Conv(COMP_1)→BN(COMP_2)→ReLU(COMP_3)** | 与分立算子完全一致；代表流统一为 **COMP_1**（FWD/INF/BWD/BWD_FIRST_LAYER），内部子操作仍按单算子原流执行 |

---

## 3. CBR 独立层张量布局（23 张量）

> **注**：`EBH_REQ.md` 中 `n1+n2+n3-2` 的公式是从全量张量角度描述的。按当前 `layer_descriptor_registry` 的 descriptor 计数方式：Conv 子部分贡献 8 张量（含 `bn_sum`/`bn_sq_sum`），BN 子部分贡献 13 张量，ReLU 子部分仅贡献独立的 `relu_output` 和 `relu_mask` 2 张量（其输入 `bn_output` 已含在 BN 的 13 张量中）。因此 CBR descriptor 总张量数为 **8 + 13 + 2 = 23**。其中 `conv_output`（索引 1）与 `bn_output`（索引 10）分别作为 BN/ReLU 的输入被复用，不重复分配。这与分立路径的三个独立特征图完全对齐。

新增 `infer_cbr_tensors`，内部拼接 `infer_conv_tensors_with_bn_stats(8)` + `infer_bn_tensors(13)` + ReLU output(1) + ReLU mask(1)。

| 索引 | 名称 | 来源 | Region | dtype | 说明 |
|------|------|------|--------|-------|------|
| 0 | `conv_weight` | Conv | `W_FIRST_CONV` / `W_DEEP_CONV` | FP32 | Conv 主权重 |
| 1 | `conv_output` | Conv | `F_FEATURE_FP16` | FP16 | Conv 输出 = BN 输入 |
| 2 | `conv_grad_slot` | Conv | `F_GRAD_SLOT_FP16` | FP16 | 梯度槽（dX 原位目标） |
| 3 | `conv_weight_grad` | Conv | `G_FIRST_CONV` / `G_DEEP_CONV` | FP32 | Conv 权重梯度 |
| 4 | `conv_amp_w_fp16` | Conv | `A_FIRST_CONV` / `A_DEEP_CONV` | FP16 | AMP 工作权重 |
| 5 | `conv_amp_g_fp16` | Conv | `G_FIRST_CONV_FP16` / `G_DEEP_CONV_FP16` | FP16 | AMP FP16 权重梯度 |
| 6 | `conv_bn_sum` | Conv | `T_TEMP_FP32` | FP32 | 真实输出张量，与独立 `CONV_AMP_FWD` 的 `sum` 完全等价，写入全局显存 |
| 7 | `conv_bn_sq_sum` | Conv | `T_TEMP_FP32` | FP32 | 真实输出张量，与独立 `CONV_AMP_FWD` 的 `sq_sum` 完全等价，写入全局显存 |
| 8 | `bn_weight` | BN | `W_BN_WEIGHT` | FP32 | γ |
| 9 | `bn_bias` | BN | `W_BN_BIAS` | FP32 | β |
| 10 | `bn_output` | BN | `F_FEATURE_FP16` | FP16 | BN 输出 = ReLU 输入 |
| 11 | `bn_prev_mean` | BN | `B_PREV_MEAN` | FP32 | running mean（旧） |
| 12 | `bn_prev_var` | BN | `B_PREV_VAR` | FP32 | running var（旧） |
| 13 | `bn_next_mean` | BN | `B_NEXT_MEAN` | FP32 | running mean（新，原地更新） |
| 14 | `bn_next_var` | BN | `B_NEXT_VAR` | FP32 | running var（新，原地更新） |
| 15 | `bn_weight_grad` | BN | `G_BN_WEIGHT` | FP32 | dγ |
| 16 | `bn_bias_grad` | BN | `G_BN_BIAS` | FP32 | dβ |
| 17 | `bn_eq_bias` | BN | `W_EQ_BIAS` | FP32 | 推理预计算 bias |
| 18 | `bn_eq_scale` | BN | `W_EQ_SCALE` | FP32 | 推理预计算 scale |
| 19 | `bn_saved_mean` | BN | `T_TEMP_FP32` | FP32 | FWD→BWD 桥接 |
| 20 | `bn_saved_inv_var` | BN | `T_TEMP_FP32` | FP32 | FWD→BWD 桥接 |
| 21 | `relu_output` | 新增 | `F_FEATURE_FP16` | FP16 | **层最终输出** |
| 22 | `relu_mask` | 新增 | `S_MASK` | INT8 | ReLU mask |

- **层最终输出索引**：`21`。
- **BWD dX 原位目标**：本层输入 X（由 Compiler 注入到 `output_ids[0]`）。

### 3.1 新增 `infer_cbr_tensors`

```cpp
std::vector<TensorDesc> infer_cbr_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<CBRParams>(params.data)) {
        LOG_WARN << "infer_cbr_tensors: params is not CBRParams";
        return descs;
    }
    const auto& cbr = std::get<CBRParams>(params.data);

    // Conv 部分：8 张量（含 bn_sum/bn_sq_sum）
    OpParams conv_op{ConvParams{cbr.conv}};
    auto conv_d = infer_conv_tensors_with_bn_stats(input, conv_op, ctx);
    descs.insert(descs.end(), conv_d.begin(), conv_d.end());

    // BN 部分：13 张量
    Shape conv_out = compute_conv_output(input, cbr.conv);

    // BN 通道对齐检查：CBR 内部 BN 子操作受 cuDNN TensorCore 约束，
    // 要求 Conv 输出通道数（即 BN 输入通道数）为 8 的倍数
    if (ctx.enable_amp && (conv_out.c() % 8 != 0)) {
        int aligned_c = ((conv_out.c() + 7) / 8) * 8;
        TR_CHECK(false, ValueError,
            "CBR AMP requires conv output channels to be a multiple of 8 "
            "(cuDNN TensorCore constraint for internal BN). "
            "Current out_ch=" << conv_out.c() << " is not aligned. "
            "Insert channel_padding() before CBR or set out_ch to "
            << aligned_c << ".");
    }

    OpParams bn_op{BNParams{cbr.bn}};
    auto bn_d = infer_bn_tensors(conv_out, bn_op, ctx);
    descs.insert(descs.end(), bn_d.begin(), bn_d.end());

    // ReLU 部分：2 张量（独立输出 + mask）
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    descs.push_back(TensorDesc{"cbr_relu_output", conv_out, select_feature_region(ctx), feat_dt});
    descs.push_back(TensorDesc{"cbr_relu_mask",   conv_out, Region::S_MASK,        DType::INT8});

    return descs;  // 共 23 张量
}
```

---

## 4. 子图模式

### 4.1 FWD

```cpp
SubgraphPattern build_cbr_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 23) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_FWD;
    // input: amp_w(4), bn_weight(8), bn_bias(9), next_mean(13), next_var(14)
    // X 由 Compiler 在 begin 注入；bn_epsilon / bn_momentum 由 Compiler 在末尾追加
    n.input_indices  = {4, 8, 9, 13, 14};
    // output: conv_output(1), bn_sum(6), bn_sq_sum(7), bn_output(10),
    //         saved_mean(19), saved_inv_var(20), relu_output(21), relu_mask(22)
    n.output_indices = {1, 6, 7, 10, 19, 20, 21, 22};
    p.nodes.push_back(n);
    return p;
}
```

### 4.2 BWD

```cpp
SubgraphPattern build_cbr_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 23) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_BWD;
    // input: amp_w(4), bn_weight(8), saved_mean(19), saved_inv_var(20), relu_mask(22)
    // dY 由 Compiler 在 begin 注入；X 由 Compiler 在末尾追加
    n.input_indices  = {4, 8, 19, 20, 22};
    // output: conv_amp_g_fp16(5), bn_weight_grad(15), bn_bias_grad(16)
    //         scratch: conv_output(1), bn_output(10)
    // dX 由 Compiler 以 in-place 方式注入到 output_ids[0]
    n.output_indices = {5, 15, 16, 1, 10};
    // Compiler 在 BWD 构建时会把 dX 以 in-place 方式插入 output_ids[0]，
    // 因此实际 output_ids 顺序为：
    // [0]=dX, [1]=conv_amp_g_fp16, [2]=bn_weight_grad, [3]=bn_bias_grad,
    // [4]=conv_output(scratch), [5]=bn_output(scratch)
    p.nodes.push_back(n);
    return p;
}
```

> 注意：AMP 下 Conv BWD 的 FP16 权重梯度输出到 `conv_amp_g_fp16`（索引 5），后续由 Compiler 自动插入 `RANGE_CAST_FP16_TO_FP32` 转为 `conv_weight_grad`（索引 3）。因此这里必须用索引 5，不能用 3。

### 4.3 INF

```cpp
SubgraphPattern build_cbr_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 23) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_INF;
    // input: amp_w(4), bn_weight(8), bn_bias(9), eq_scale(18), eq_bias(17),
    //        next_mean(13), next_var(14)
    // X 由 Compiler 在 begin 注入；bn_epsilon 由 Compiler 在末尾追加
    n.input_indices  = {4, 8, 9, 18, 17, 13, 14};
    // output: conv_output(1), bn_sum(6), bn_sq_sum(7), bn_output(10),
    //         relu_output(21), relu_mask(22)
    n.output_indices = {1, 6, 7, 10, 21, 22};
    p.nodes.push_back(n);
    return p;
}
```

### 4.4 注册

```cpp
static const LayerDescriptor cbr_desc = {
    infer_cbr_tensors,
    build_cbr_forward,
    build_cbr_backward,
    build_cbr_inference
};

// get_output_shape:
case LayerKind::CBR: return find(21);  // relu_output

// get_layer_descriptor:
case LayerKind::CBR: return cbr_desc;
```

---

## 5. 各层改动清单

### 5.1 `include/renaissance/graph/op_kind.h`

1. 在 `ComputeOp` 枚举中新增 4 个值，追加在 `CHANNEL_PADDING_AMP_BWD_FIRST_LAYER` 之后、`COUNT` 之前，避免中间插入导致后续大量枚举值后移：
   ```cpp
   CBR_AMP_FWD,
   CBR_AMP_BWD,
   CBR_AMP_BWD_FIRST_LAYER,
   CBR_AMP_INF,
   ```
2. `CBRParams` 已存在，无需改动。

### 5.1b `include/renaissance/graph/blueprint.h`

修改 `CBRParam` 结构体，增加 `momentum` 和 `eps` 字段（与 `BNParam` 风格一致，`momentum` 在前）：

```cpp
struct CBRParam {
    int out_ch; int k; int s; int p;
    float momentum = 0.1f;
    float eps = 1e-5f;
};
```

修改工厂函数签名：

```cpp
inline Layer cbr(int out_ch, int k, int s, int p,
                 double momentum = 0.1, double eps = 1e-5) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::CBR,
        detail::CBRParam{out_ch, k, s, p,
                         static_cast<float>(momentum),
                         static_cast<float>(eps)}));
}

inline Layer conv_bn_relu(int out_ch, int k, int s, int p,
                          double momentum = 0.1, double eps = 1e-5) {
    return cbr(out_ch, k, s, p, momentum, eps);
}
```

### 5.2 `src/graph/op_kind.cpp`

- `compute_op_to_string()` 增加 4 个 CBR case。
- `format_params()` 增加：
  ```cpp
  case ComputeOp::CBR_AMP_FWD:
  case ComputeOp::CBR_AMP_BWD:
  case ComputeOp::CBR_AMP_BWD_FIRST_LAYER:
  case ComputeOp::CBR_AMP_INF: {
      if (auto* cp = std::get_if<CBRParams>(&p.data)) {
          oss << "out_ch=" << cp->conv.out_channels
              << ",kernel=" << cp->conv.kernel_h << "x" << cp->conv.kernel_w
              << ",stride=" << cp->conv.stride_h << "x" << cp->conv.stride_w
              << ",pad=" << cp->conv.pad_h << "x" << cp->conv.pad_w
              << ",eps=" << cp->bn.eps << ",momentum=" << cp->bn.momentum;
      }
      break;
  }
  ```

### 5.3 `include/renaissance/graph/arch_plan.h`

1. `LayerKind` 枚举在 `GapFC` 之前新增 `CBR`。
2. 新增 `CbrLayerParams`：
   ```cpp
   struct CbrLayerParams {
       int out_ch, k, s, p;
       float eps = 1e-5f;
       float momentum = 0.1f;
       bool operator==(const CbrLayerParams& o) const {
           return out_ch == o.out_ch && k == o.k && s == o.s && p == o.p &&
                  eps == o.eps && momentum == o.momentum;
       }
   };
   ```
3. `LayerParam` variant 中加入 `CbrLayerParams`。

### 5.4 `src/graph/arch_plan_expand.cpp`

修改 `NodeKind::CBR` 分支，根据 `fuse` 参数分支，且使用 `CBRParam` 中的 eps/momentum：

```cpp
case NodeKind::CBR: {
    auto& p = std::get<CBRParam>(node.payload);
    if (fuse) {
        CbrLayerParams cbr{p.out_ch, p.k, p.s, p.p, p.eps, p.momentum};
        out.push_back({LayerKind::CBR, cbr, "cbr", {}, {}, false, false, src_id});
    } else {
        out.push_back({LayerKind::Conv, ConvLayerParams{p.out_ch, p.k, p.s, p.p},
                       "conv_cbr", {}, {}, false, false, src_id});
        out.push_back({LayerKind::Bn2d, BNParams{p.eps, p.momentum},
                       "bn_cbr", {}, {}, false, false, src_id});
        out.push_back({LayerKind::ReLU, EmptyParams{},
                       "relu_cbr", {}, {}, false, false, src_id});
    }
    current_c = p.out_ch;
    break;
}
```

> **注意**：`NodeKind::CBR`（用户显式调用 `cbr()`/`conv_bn_relu()`）在 `fuse=true` 时直接 emit `LayerKind::CBR`，不会拆成三个 primitive 层；`step9_merge_triple` 仅处理用户显式写的 `Conv; Bn2d; ReLU;` primitive 序列。两个路径互不重叠。

### 5.5 `src/graph/arch_plan_shape.cpp`

`recompute_shapes_from` 新增：

```cpp
case LayerKind::CBR: {
    auto& p = std::get<CbrLayerParams>(layer.params);
    cur = {cur.n(),
           (cur.h() + 2 * p.p - p.k) / p.s + 1,
           (cur.w() + 2 * p.p - p.k) / p.s + 1,
           p.out_ch};
    break;
}
```

### 5.6 `src/graph/arch_plan_merge.cpp`

恢复 `step9_merge_triple()`，仅在 AMP 下执行：

```cpp
void ArchPlan::step9_merge_triple() {
    if (!GlobalRegistry::instance().using_amp()) return;

    auto build_cbr = [](const ArchLayer& conv, const ArchLayer& bn, const ArchLayer& relu) -> LayerParam {
        auto& cp = std::get<ConvLayerParams>(conv.params);
        float eps = 1e-5f, momentum = 0.1f;
        if (std::holds_alternative<BNParams>(bn.params)) {
            auto& bp = std::get<BNParams>(bn.params);
            eps = bp.eps; momentum = bp.momentum;
        }
        return CbrLayerParams{cp.out_ch, cp.k, cp.s, cp.p, eps, momentum};
    };
    merge_pattern_triple(LayerKind::Conv, LayerKind::Bn2d, LayerKind::ReLU,
                         LayerKind::CBR, build_cbr);
}
```

> **备注**：当前 `arch_plan_expand.cpp` 中非 CBR 路径的 BN 展开使用的是 `BNParams{}`（默认值），`step9_merge_triple` 中的 `std::holds_alternative<BNParams>` 实际会走 false 分支，回退到 `1e-5f/0.1f`。如果用户显式写 `Conv; bn(momentum, eps); ReLU;` 序列，`bn()` 会带 `BNParams`，合并时会正确提取对应的 eps/momentum。如果用户写 `cbr()`，则 expand 直接生成 `LayerKind::CBR`，不会走到 step9。

### 5.7 `src/graph/arch_plan_normalize.cpp`

`get_effective_output_c_at` 新增：

```cpp
case LayerKind::CBR:
    return std::get<CbrLayerParams>(layers_[idx].params).out_ch;
```

### 5.8 `src/graph/arch_plan_format.cpp`

- `kind_name()` 新增 `case LayerKind::CBR: return "CBR";`。
- `params_str()` 新增：
  ```cpp
  case LayerKind::CBR: {
      auto& p = std::get<CbrLayerParams>(l.params);
      snprintf(buf, sizeof(buf), "out=%d k=%d s=%d p=%d eps=%g mom=%g",
               p.out_ch, p.k, p.s, p.p, p.eps, p.momentum);
      break;
  }
  ```

### 5.9 `src/graph/arch_plan_yaml.cpp`

- `kind_from_name()` 新增 `"CBR" → LayerKind::CBR`。
- `to_yaml()` 中 `params` 序列化新增：
  ```cpp
  case LayerKind::CBR: {
      auto& p = std::get<CbrLayerParams>(l.params);
      pnode["out_ch"] = p.out_ch; pnode["k"] = p.k; pnode["s"] = p.s; pnode["p"] = p.p;
      pnode["eps"] = p.eps; pnode["momentum"] = p.momentum;
      break;
  }
  ```
- `from_yaml()` 中反序列化新增：
  ```cpp
  case LayerKind::CBR:
      layer.params = CbrLayerParams{
          pnode["out_ch"].get_value<int>(), pnode["k"].get_value<int>(),
          pnode["s"].get_value<int>(), pnode["p"].get_value<int>(),
          pnode.contains("eps") ? pnode["eps"].get_value<float>() : 1e-5f,
          pnode.contains("momentum") ? pnode["momentum"].get_value<float>() : 0.1f};
      break;
  ```

### 5.10 `src/graph/layer_descriptor_registry.cpp`

- 新增 `infer_cbr_tensors`（23 张量，见第 3 节）。
- 新增 `build_cbr_forward` / `build_cbr_backward` / `build_cbr_inference`（见第 4 节）。
- `get_output_shape()` 新增 `case LayerKind::CBR: return find(21);`。
- `get_layer_descriptor()` 新增 `case LayerKind::CBR: return cbr_desc;`。

### 5.11 `src/graph/compiler.cpp`

#### 5.11.1 `convert_to_op_params`

新增：

```cpp
if (std::holds_alternative<CbrLayerParams>(lp)) {
    auto& p = std::get<CbrLayerParams>(lp);
    ConvParams cp{p.out_ch, p.k, p.k, p.s, p.s, p.p, p.p, 1, 1, 1};
    BNParams bp{p.eps, p.momentum};  // BNParams 字段顺序为 eps, momentum
    return OpParams{CBRParams{cp, bp}};
}
```

#### 5.11.2 `is_weight_bearing`

新增：

```cpp
case LayerKind::CBR: return true;
```

#### 5.11.3 `ArchPlanCompiler::compile_layer`

新增：

```cpp
case LayerKind::CBR:
    compile_cbr(layer, idx);
    break;
```

新增私有函数 `compile_cbr`：

```cpp
void compile_cbr(const ArchLayer& layer, size_t idx) {
    auto& p = std::get<CbrLayerParams>(layer.params);
    int in_c = layer.in_shape.c();
    Shape w_shape{p.out_ch, p.k, p.k, in_c};
    bool is_first = layer.is_first_layer;
    alloc_conv_group(w_shape, is_first);

    int ch = layer.out_shape.c();
    Shape param_shape{ch};
    alloc_bn_group(param_shape, idx, layer);
    alloc_bn_bias_group(param_shape);
    memory_plan_.alloc_bn_stats(param_shape);
}
```

#### 5.11.4 `get_layer_output_id` / `get_grad_output_id`

```cpp
// get_layer_output_id:
case LayerKind::CBR: idx = 21; break;  // relu_output

// get_grad_output_id:
case LayerKind::CBR: idx = -1; break;  // dX in-place to X
```

#### 5.11.5 首层 FWD/INF 注入

将 `CBR` 加入 `Conv/FC` 首层特化分支（训练图约 1185 行，推理图约 1626 行）：

**训练图**（约 1185 行）：
```cpp
if (layer.is_first_layer && !gn.input_ids.empty() &&
    (layer.kind == LayerKind::Conv ||
     layer.kind == LayerKind::FC ||
     layer.kind == LayerKind::CBR)) {
    // 注入 I_A_DATA / I_B_DATA 到 input_ids 开头
}
```

**推理图**（约 1626 行）：
```cpp
if (layer.is_first_layer && !gn.input_ids.empty() &&
    (layer.kind == LayerKind::Conv ||
     layer.kind == LayerKind::FC ||
     layer.kind == LayerKind::CBR)) {
    // 注入 I_A_DATA / I_B_DATA 到 input_ids 开头
}
```

#### 5.11.6 首层 BWD 特化

`to_first_layer_bwd_op` 增加：

```cpp
case ComputeOp::CBR_AMP_BWD: return ComputeOp::CBR_AMP_BWD_FIRST_LAYER;
```

新增判断：

```cpp
inline bool is_cbr_bwd_op(ComputeOp op) {
    return op == ComputeOp::CBR_AMP_BWD || op == ComputeOp::CBR_AMP_BWD_FIRST_LAYER;
}
```

在 BWD 构建循环中，为 `is_conv_bwd_op || is_cbr_bwd_op` 统一处理：

- 非首层：追加 X 到 input_ids，dX in-place 到 output_ids[0]。
- 首层：分别追加 `b.data_a` / `b.data_b`，生成 `FIRST_LAYER_BWD_A` / `FIRST_LAYER_BWD_B` 两个节点。

#### 5.11.7 梯度追踪

`get_grad_output_id` fallback 中增加 `LayerKind::CBR`：

```cpp
if (grad_id < 0 && (... || layer.kind == LayerKind::CBR || ...)) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) grad_id = it->second;
}
```

#### 5.11.8 BN FWD/INF 标量注入

当前 Compiler 只为 `is_bn_fwd_op` / `is_bn_inf_op` 追加标量。由于 CBR 的 `compute_op` 不是 BN，需要新增分支或扩展现有判断。

推荐在 FWD 循环中新增：

```cpp
// CBR FWD: 注入全局标量 bn_epsilon 和 bn_momentum
if (gn.compute_op == ComputeOp::CBR_AMP_FWD) {
    const auto& b = memory_plan.baseline();
    if (b.bn_epsilon >= 0)  gn.input_ids.push_back(b.bn_epsilon);
    if (b.bn_momentum >= 0) gn.input_ids.push_back(b.bn_momentum);
}
```

在 INF 循环中新增：

```cpp
// CBR INF: 注入全局标量 bn_epsilon
if (gn.compute_op == ComputeOp::CBR_AMP_INF) {
    const auto& b = memory_plan.baseline();
    if (b.bn_epsilon >= 0) gn.input_ids.push_back(b.bn_epsilon);
}
```

#### 5.11.9 首层合法性校验

约 2301 行的首层校验 switch 增加 `LayerKind::CBR`：

```cpp
switch (first_layer.kind) {
    case LayerKind::Flatten:
    case LayerKind::Conv:
    case LayerKind::ChannelPadding:
    case LayerKind::CBR:
        return;  // 合法首层，静默通过
    default:
        TR_CHECK(false, ValueError, ...);
}
```

#### 5.11.10 `build_auxiliary_graphs` — BN 统计量维护（严重，不可遗漏）

`build_auxiliary_graphs` 中 `has_bn` 检测仅检查 `Bn1d`/`Bn2d`，遗漏 `CBR`。若网络仅含 CBR 层，会导致：

1. **STATS_COMM 缺失**：`RANGE_BN_STATS_ALLREDUCE` 不创建，分布式训练 running mean/var 无法同步。
2. **Running stats 传递断裂**：`RANGE_D2D_COPY`（`B_NEXT_MEAN→B_PREV_MEAN` 等）不创建。
3. **eps/momentum 一致性校验跳过**。

需修改三处：

**a) has_bn 检测**（约 1744 行）：

```cpp
if (k == LayerKind::Bn1d || k == LayerKind::Bn2d || k == LayerKind::CBR) {
    has_bn = true;
    break;
}
```

**b) eps/momentum 一致性校验**（约 1754 行）：

```cpp
if (layer.kind == LayerKind::Bn1d || layer.kind == LayerKind::Bn2d || layer.kind == LayerKind::CBR) {
    float eps = -1.0f, mom = -1.0f;
    if (std::holds_alternative<BNParams>(layer.params)) {
        const auto& bp = std::get<BNParams>(layer.params);
        eps = bp.eps; mom = bp.momentum;
    } else if (std::holds_alternative<CbrLayerParams>(layer.params)) {
        const auto& cp = std::get<CbrLayerParams>(layer.params);
        eps = cp.eps; mom = cp.momentum;
    }
    if (eps >= 0) {
        if (first_eps < 0) { first_eps = eps; first_mom = mom; }
        else if (eps != first_eps || mom != first_mom) {
            TR_CHECK(false, ValueError,
                     "All BN/CBR layers must share the same epsilon and momentum "
                     "for RANGE OP batch operations.");
        }
    }
}
```

**c) `convert_to_op_params`** 中 `BNParams` 构造必须保持字段顺序 `{eps, momentum}`（已确认正确）：

```cpp
BNParams bp{p.eps, p.momentum};  // BNParams 字段顺序为 eps, momentum
```

#### 5.11.11 `is_bn_like` 扩展以支持 BN3 检测

`src/graph/compiler.cpp:36-44` 的 `is_bn_like` 只覆盖 `Bn1d`/`Bn2d`。若 CBR 位于 `Add2End` 之前（例如 ResNet block 内部用 CBR 替代 `Conv+BN+ReLU`），其内部 BN 应被识别为 BN3，否则 zero-gamma 初始化会漏掉。将 `LayerKind::CBR` 加入判断：

```cpp
static bool is_bn_like(LayerKind k) {
    switch (k) {
        case LayerKind::Bn1d:
        case LayerKind::Bn2d:
        case LayerKind::CBR:
            return true;
        default:
            return false;
    }
}
```

这样 `Compiler::is_bn3_layer` 和 `alloc_bn_group` 中的 `mark_bn3_if_needed` 会自动覆盖 CBR。

#### 5.11.12 `DeepLearningTask::init_variant_scalars` 读取 CBR 的 eps/momentum

`src/task/deep_learning_task.cpp:605-614` 只从 `Bn1d`/`Bn2d` 提取 BN 全局参数。如果网络里只有 CBR 层，会 fallback 到默认值，可能与 `CbrLayerParams` 中的值不一致。增加 `LayerKind::CBR` 分支：

```cpp
for (const auto& layer : arch_plan_.layers()) {
    if (layer.kind == LayerKind::Bn1d || layer.kind == LayerKind::Bn2d ||
        layer.kind == LayerKind::CBR) {
        if (std::holds_alternative<BNParams>(layer.params)) {
            const auto& bp = std::get<BNParams>(layer.params);
            bn_eps = bp.eps; bn_mom = bp.momentum; break;
        } else if (std::holds_alternative<CbrLayerParams>(layer.params)) {
            const auto& cp = std::get<CbrLayerParams>(layer.params);
            bn_eps = cp.eps; bn_mom = cp.momentum; break;
        }
    }
}
```

### 5.12 `src/backend/op_stream_policy.cpp`

新增（代表流统一为 COMP_1，用户指示。内部子操作仍按单算子原流执行：ReLU BWD 在 COMP_3，BN BWD 在 COMP_2，Conv BWD 在 COMP_3/COMP_1）：

```cpp
// CBR 所有变体代表流统一为 COMP_1
case ComputeOp::CBR_AMP_FWD:
case ComputeOp::CBR_AMP_INF:
case ComputeOp::CBR_AMP_BWD:
case ComputeOp::CBR_AMP_BWD_FIRST_LAYER:
    return StreamKind::COMP_1;
```

### 5.13 `include/renaissance/backend/op_registry.h` / `src/backend/op_registry.cpp`

- `op_registry.h` 声明 `void register_op_cbr();`。
- `op_registry.cpp` 的 `register_default_ops()` 中调用 `register_op_cbr();`。
- `require_warmup()` 增加 4 个 CBR 算子，返回 `true`。
- `warmup_single_cudnn_op()` 中把 CBR 四个算子加入 Conv 特判分支（约 138 行），走 `entry.launch_cuda(...)` 后 `cudaDeviceSynchronize()` 全同步路径，**不走**非 Conv 分支的 COMP_1 流同步。因为 CBR 内部跨 COMP_1/2/3，只有全同步才能正确预热所有子操作的 cuDNN graph。若走非 Conv 分支，仅预热 COMP_1 会漏掉 COMP_2/3 上的 kernel，导致首次 capture 时才 build。

```cpp
if (node.compute_op == ComputeOp::CONV_AMP_FWD ||
    // ... 现有 Conv 算子 ...
    node.compute_op == ComputeOp::CBR_AMP_FWD ||
    node.compute_op == ComputeOp::CBR_AMP_BWD ||
    node.compute_op == ComputeOp::CBR_AMP_BWD_FIRST_LAYER ||
    node.compute_op == ComputeOp::CBR_AMP_INF) {
    // 直接 launch_cuda + cudaDeviceSynchronize()
}
```

### 5.14 `src/CMakeLists.txt`

追加：

```cmake
backend/ops/dtensor/cbr_op.cpp
```

### 5.15 `tests/op/CMakeLists.txt`

新增测试目标 `test_cbr_amp`（源文件 `tests/op/test_cbr_amp.cpp`），链接、GPU 环境、输出目录、编译选项与现有 `test_conv_fwd_bwd` 一致。

---

## 6. 后端实现方案

### 6.1 新增文件

| 文件 | 说明 |
|------|------|
| `src/backend/ops/dtensor/cbr_op.cpp` | CBR AMP 算子的 launch_cuda / 注册 |

### 6.2 实现策略：复制核心逻辑

CBR 的 4 个 launch 函数在 `cbr_op.cpp` 中**自行完整实现** Conv/BN/ReLU 三个子操作的等价逻辑，不直接调用现有 `static` 的 launch 函数，也不暴露它们。

理由：
1. LKW.md 明确要求 CBR 算子要有自己的实现文件，"哪怕是复制单算子的代码也好"。
2. 现有 Conv/BN/ReLU 的 launch 函数均为 `static`，跨文件无法直接调用；暴露它们会侵入现有算子封装。
3. 自行实现可精确控制每个子操作的流、workspace、graph cache，确保与分立算子字节级一致。

### 6.3 各子操作的具体实现方式

#### 6.3.1 Conv 子操作：完整复制并重命名所有函数符号

`conv_op_impl.cpp` 中的 `build_conv_amp_fwd_graph`、`build_conv_amp_inf_graph`、`build_conv_amp_wgrad_graph`、`build_conv_amp_dgrad_graph` 等函数**均为非 static 的外部链接符号**（[conv_op_impl.cpp:306-427](file:///r:/renaissance/src/backend/ops/dtensor/conv_op_impl.cpp#L306-L427)）。`conv_op.cpp` 已通过 `#include "conv_op_impl.cpp"` 定义了一次，如果 `cbr_op.cpp` 再同样 include，链接时会产生多重定义错误。

即使是匿名 namespace 包裹 include 的方式，也可能因 `using namespace`、模板特化等细微问题导致编译失败。因此采用**与 BN 相同的复制策略**：将 `conv_op_impl.cpp` 中需要的代码完整复制到 `cbr_op.cpp`，并重命名所有具有外部链接的符号。

复制范围与重命名规则：

| 原名称 | 新名称 |
|--------|--------|
| `ConvGraphCacheKey` | `CBRConvGraphCacheKey` |
| `ConvGraphCacheKeyHasher` | `CBRConvGraphCacheKeyHasher` |
| `ConvGraphCache` | `CBRConvGraphCache` |
| `s_conv_fwd_cache`（FWD + INF 共用） | `s_cbr_conv_fwd_cache` |
| `s_conv_wgrad_cache` | `s_cbr_conv_wgrad_cache` |
| `s_conv_dgrad_cache` | `s_cbr_conv_dgrad_cache` |
| `build_conv_amp_fwd_graph` | `build_cbr_conv_amp_fwd_graph` |
| `build_conv_amp_inf_graph` | `build_cbr_conv_amp_inf_graph` |
| `build_conv_amp_wgrad_graph` | `build_cbr_conv_amp_wgrad_graph` |
| `build_conv_amp_dgrad_graph` | `build_cbr_conv_amp_dgrad_graph` |

> **重要**：`build_cbr_conv_amp_fwd_graph` **不能**把 AMP FWD 转发到 AMP INF。它必须与独立 `CONV_AMP_FWD` 使用**完全相同的 cuDNN FE 图结构**——即 `conv_fprop + genstats`，`sum`/`sq_sum` 作为真实输出节点（`set_output(true)`），绑定到 CBR 张量索引 6/7，写入全局显存。这样才能字节级复现 `Conv+BN2D+ReLU` 的串联结果。独立 `CONV_AMP_FWD` 当前没有将 `sum`/`sq_sum` 设为虚张量，CBR 也必须保持一致。

复制整个 `conv_op_impl.cpp` 文件内容时，以下 **static/inline 辅助函数**也必须一并复制，否则编译会失败（它们的名称不与其他翻译单元冲突，无需重命名）：

| 原名称 | 说明 |
|--------|------|
| `make_nhwc_stride` | inline，构造 NHWC stride |
| `make_krsc_stride` | inline，构造 KRSC stride |
| `get_or_build_cache` | static 模板，cache 查找/构建 |
| `update_conv_tensor_to_id` | static，更新 tensor_to_id 映射 |

#### 6.3.2 BN 子操作：完整复制并重命名所有全局 cache 变量

BN 的 cuDNN FE graph 构建、`BNGraphCacheKey`、cache map（`s_bn_fwd_caches` / `s_bn_bwd_caches`）、`update_bn_tensor_to_id`、`float_to_bits` 等都直接写在 `bn_op.cpp` 中，没有拆成独立的 `_impl.cpp`。因此 `cbr_op.cpp` 需要**完整复制**这部分代码。

> **注意**：`s_bn_fwd_caches` 和 `s_bn_bwd_caches` 是全局变量（外部链接），不是 static。重命名的目的是避免跨翻译单元重复定义，而非 static 冲突。`s_bn_inf_caches` 虽然存在声明，但 BN INF 实际走自定义 kernel `launch_tr_bn_inf_kernel`，不经过 cuDNN FE graph cache，**无需复制**。

复制范围与重命名规则：

| 原名称 | 新名称 |
|--------|--------|
| `BNGraphCacheKey` | `CBRBNGraphCacheKey` |
| `BNGraphCacheKeyHash` | `CBRBNGraphCacheKeyHash` |
| `BNGraphCache` | `CBRBNGraphCache` |
| `s_bn_fwd_caches` | `s_cbr_bn_fwd_caches` |
| `s_bn_bwd_caches` | `s_cbr_bn_bwd_caches` |
| `update_bn_tensor_to_id` | `update_cbr_bn_tensor_to_id` |
| `float_to_bits`（匿名 namespace 内） | 直接复制，匿名 namespace 天然隔离 |

`create_cudnn_graph` / `finalize_cudnn_graph` 是 `cudnn_utils.h` 中的 `inline` 公共函数，`#include` 头文件即可调用，**无需复制**。

`launch_bn_fwd_cuda` / `launch_bn_bwd_cuda` / `launch_bn_inf_cuda` 的核心逻辑：内联到 CBR launch 函数中。

> **重要**：复制 BN FWD 代码时，`next_mean`/`next_var` 的原地更新逻辑必须显式强制绑定到输入 running buffer。独立 `bn_op.cpp` 可能根据 `node.output_ids.size() >= 5` 判断 `next_rm`/`next_rv` 的绑定位置，但 CBR 的 `output_ids.size() == 8`，`output_ids[3]`/`output_ids[4]` 不是 `next_mean`/`next_var`（而是 `saved_mean`/`saved_inv_var`）。必须去掉 `output_ids.size() >= 5` 的 fallback 逻辑，改为显式：
> ```cpp
> name_to_id["next_rm"] = mp.get_dtensor(node.input_ids[4]).id;  // next_mean
> name_to_id["next_rv"] = mp.get_dtensor(node.input_ids[5]).id;  // next_var
> // saved_mean/saved_inv_var 输出到 output_ids[4]/output_ids[5]
> name_to_id["saved_mean"]    = mp.get_dtensor(node.output_ids[4]).id;
> name_to_id["saved_inv_var"] = mp.get_dtensor(node.output_ids[5]).id;
> ```

#### 6.3.3 ReLU 子操作：extern 声明 CUDA kernel 包装函数

ReLU 的 kernel 包装函数定义在 `relu_op.cu` 中，非 static，可直接 extern 声明调用。在 `cbr_op.cpp` 顶部添加：

```cpp
extern cudaError_t launch_relu_amp_fwd_mask_kernel(
    const __half* x, __half* y, int8_t* mask,
    int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_amp_bwd_kernel(
    const __half* dY, const int8_t* mask, __half* dX,
    int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_amp_inf_kernel(
    const __half* x, __half* y, int64_t n, cudaStream_t stream);
```

BN INF 使用的自定义 kernel 也需要 extern 声明：

```cpp
extern "C" cudaError_t launch_tr_bn_inf_kernel(
    const void* x,
    const float* gamma, const float* beta,
    const float* running_mean, const float* running_var,
    float eps, void* y,
    int N, int C, int H, int W,
    bool is_fp16, cudaStream_t stream);
```

#### 6.3.4 复制 BN/Conv 代码时的 params 访问方式

`bn_op.cpp` 中通过 `node.params.bn()` 获取 BN 参数（[bn_op.cpp:351](file:///r:/renaissance/src/backend/ops/dtensor/bn_op.cpp#L351)），但 `cbr_op.cpp` 的 `node.params` 类型是 `CBRParams`，应改为 `node.params.cbr().bn()`。同理 Conv 参数访问也应改为 `node.params.cbr().conv()`。复制代码时务必注意，避免照搬 `node.params.bn()` 取到默认值。

### 6.4 核心原则

- **流**：子操作内部显式使用对应流，不调用 `get_op_default_stream(node.compute_op)`。各子操作执行流：
  - `COMP_1`：Conv FWD / Conv INF / Conv BWD wgrad（含首层 BWD_FIRST_LAYER）
  - `COMP_2`：BN FWD / BN BWD / BN INF
  - `COMP_3`：ReLU FWD / ReLU BWD / ReLU INF / Conv BWD dgrad（非首层）
- **Workspace**：`DeviceContext::ensure_workspace_grow(StreamKind sk, size_t size)` 是按流独立管理 workspace 的，不同流的 workspace 互不共享。每个子操作在自身所在流上独立调用：
  - `COMP_1`：Conv FWD/INF/BWD_FIRST_LAYER；
  - `COMP_2`：BN FWD/BWD/INF；
  - `COMP_3`：ReLU 不需要 workspace，但 Conv BWD dgrad 需要。
- **Graph Cache**：Conv 通过复制并重命名 `conv_op_impl.cpp` 中的代码获得独立 cache；BN 通过重命名 static 变量获得独立 cache；两者天然互不污染。
- **state.output_stream_idx**：最终设置为最后一个子操作的流索引。

### 6.5 跨子操作同步

> **必须注意**：CBR 内部不能依赖 Compiler 自动插入算子间同步。复制 Conv/BN/ReLU 代码时，必须完整保留每个子操作内部的流切换与 event 同步逻辑：每个子操作执行前 `cudaStreamWaitEvent`（等待上游子操作在其输出流上 record 的 event），执行后 `cudaEventRecord` 并更新 `state.output_stream_idx`。任何省略都会导致跨流竞争条件。

每个子操作内部已经执行了：
1. 注册/获取自身 stream；
2. 若 `state.output_stream_idx` 指向不同 stream，则 `cudaStreamWaitEvent`；
3. 执行 kernel；
4. `cudaEventRecord` 并更新 `state.output_stream_idx`。

因此 CBR 只需把同一个 `MultiStreamCaptureState& state` 依次传给三个子操作，依赖链自动形成：
- FWD/INF：`COMP_1(Conv) → COMP_2(BN) → COMP_3(ReLU)`。
- BWD：`COMP_3(ReLU) → COMP_2(BN) → Conv BWD（wgrad on COMP_1，dgrad on COMP_3；首层仅 COMP_1 wgrad、无 dgrad）`。

> **注意**：`state.output_stream_idx` 必须跟随最后一个子操作的实际流，**不能强制改回 COMP_1**。`get_op_default_stream(CBR_AMP_BWD)` 返回 COMP_1 是"对外代表流"（用于调度），但内部 `state.output_stream_idx` 最终由 Conv BWD 的 dgrad 设置（COMP_3，非首层）或 wgrad 设置（COMP_1，首层）。下游算子通过 `state.output_stream_idx` 决定在哪个流上等待 CBR 完成，所以必须指向 CBR 内部最后一个 kernel 实际所在的流。

### 6.6 CBR_AMP_FWD 内部流程

```cpp
static void launch_cbr_amp_fwd_cuda(...)
{
    // input_ids:  [0]=X [1]=amp_w [2]=bn_w [3]=bn_b [4]=next_mean [5]=next_var [6]=eps [7]=mom
    // output_ids: [0]=conv_output [1]=bn_sum [2]=bn_sq_sum [3]=bn_output
    //             [4]=saved_mean [5]=saved_inv_var [6]=relu_output [7]=relu_mask

    // 1) Conv FWD on COMP_1：完整复现独立 CONV_AMP_FWD 的 conv_fprop + genstats；
    //    sum/sq_sum 作为真实输出（set_output(true)），写入 CBR 张量索引 6/7。
    //    最终输出 conv_output。
    // 2) BN FWD on COMP_2：输入 conv_output，输出 bn_output + saved_mean + saved_inv_var
    // 3) ReLU FWD on COMP_3：输入 bn_output，输出 relu_output + relu_mask
}
```

### 6.7 CBR_AMP_BWD 内部流程

> **output_ids 顺序契约**（Compiler 处理后，与 `build_cbr_backward` 子图 `output_indices` 对应）：
> ```
> [0] = dX target (X_id / data_a / data_b)  ← Compiler 注入
> [1] = conv_amp_g_fp16   (index 5)
> [2] = bn_weight_grad    (index 15)
> [3] = bn_bias_grad      (index 16)
> [4] = conv_output scratch (index 1)
> [5] = bn_output scratch   (index 10)
> ```

```cpp
static void launch_cbr_amp_bwd_cuda(...)
{
    // input_ids:  [0]=dY [1]=amp_w [2]=bn_w [3]=saved_mean [4]=saved_inv_var [5]=mask [6]=X
    // output_ids: [0]=dX target [1]=conv_amp_g [2]=dγ [3]=dβ [4]=conv_output(scratch) [5]=bn_output(scratch)

    // 1) ReLU BWD on COMP_3：dY * mask → bn_output(scratch)
    // 2) BN BWD on COMP_2：输入 bn_output(scratch)，输出 conv_output(scratch) + dγ + dβ
    // 3) Conv BWD on COMP_1/3：输入 conv_output(scratch)，输出 dX + conv_amp_g
}
```

### 6.8 CBR_AMP_BWD_FIRST_LAYER 内部流程

与 `CBR_AMP_BWD` 相同，只是第 3 步 Conv BWD 替换为首层版本（仅 wgrad，不写 dX）。

> **注意**：首层 BWD 的 `output_ids[0]` 被 Compiler 注入 `data_a`/`data_b`（A/B 双缓冲），但首层特化只算 wgrad 不写 dX。**launch 函数必须跳过 `output_ids[0]` 的写入**，否则会破坏层输入缓冲区。

### 6.9 CBR_AMP_INF 内部流程

```cpp
static void launch_cbr_amp_inf_cuda(...)
{
    // input_ids:  [0]=X [1]=amp_w [2]=bn_w [3]=bn_b [4]=eq_scale [5]=eq_bias
    //             [6]=next_mean [7]=next_var [8]=eps
    // output_ids: [0]=conv_output [1]=bn_sum(reserved) [2]=bn_sq_sum(reserved) [3]=bn_output
    //             [4]=relu_output [5]=relu_mask

    // 1) Conv INF on COMP_1：输出 conv_output；bn_sum/bn_sq_sum 按独立 CONV_AMP_INF 方式预留分配但不绑定、不写入（纯 conv_fprop，无 GenStats）
    // 2) BN INF on COMP_2
    // 3) ReLU INF on COMP_3
}
```

### 6.10 注册

参考 `register_op_conv` 的模式（[conv_op.cpp:1356](file:///r:/renaissance/src/backend/ops/dtensor/conv_op.cpp#L1356-L1427)），`register_op_cbr` 的函数体本身在 `#ifdef TR_USE_CUDA` 之外，仅 `launch_cuda` 赋值放在 guard 内部，确保非 CUDA 构建时不会链接失败：

```cpp
void register_op_cbr() {
    // CBR_AMP_FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_FWD)];
        e.op = ComputeOp::CBR_AMP_FWD;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_fwd_cuda;
#endif
    }
    // CBR_AMP_BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_BWD)];
        e.op = ComputeOp::CBR_AMP_BWD;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_bwd_cuda;
#endif
    }
    // CBR_AMP_BWD_FIRST_LAYER
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_BWD_FIRST_LAYER)];
        e.op = ComputeOp::CBR_AMP_BWD_FIRST_LAYER;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_bwd_first_layer_cuda;
#endif
    }
    // CBR_AMP_INF
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_INF)];
        e.op = ComputeOp::CBR_AMP_INF;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_inf_cuda;
#endif
    }
}
```

其中 `launch_cbr_amp_cpu_not_supported` 是一个统一的 CPU 占位函数，直接 `TR_CHECK(false, "CBR AMP is CUDA-only.")`。

```cpp
static void launch_cbr_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_CHECK(false, NotImplementedError, "CBR AMP is CUDA-only.");
}
```

### 6.11 CBR 节点输入输出索引契约（Compiler 处理后）

> **这是实现中最容易出错的地方**。CBR 节点的 `input_ids`/`output_ids` 顺序与独立 Conv/BN/ReLU 算子不同，复制代码时不能直接照搬独立算子的 `name_to_id` 映射。下面给出 Compiler 处理后的最终索引契约，CBR 后端实现必须严格按此映射。

#### CBR_AMP_FWD

```
// input_ids:
//   [0] = X                  (Compiler 注入)
//   [1] = amp_w              (CBR 张量索引 4)
//   [2] = bn_weight          (CBR 张量索引 8)
//   [3] = bn_bias            (CBR 张量索引 9)
//   [4] = next_mean          (CBR 张量索引 13)
//   [5] = next_var           (CBR 张量索引 14)
//   [6] = bn_epsilon         (scalar, Compiler 追加)
//   [7] = bn_momentum        (scalar, Compiler 追加)

// output_ids:
//   [0] = conv_output        (CBR 张量索引 1)
//   [1] = bn_sum             (CBR 张量索引 6)
//   [2] = bn_sq_sum          (CBR 张量索引 7)
//   [3] = bn_output          (CBR 张量索引 10)
//   [4] = saved_mean         (CBR 张量索引 19)
//   [5] = saved_inv_var      (CBR 张量索引 20)
//   [6] = relu_output        (CBR 张量索引 21)
//   [7] = relu_mask          (CBR 张量索引 22)
```

子操作映射：

| 子操作 | 输入索引 | 输出索引 | 说明 |
|--------|---------|---------|------|
| Conv FWD | `input[0]=X`, `input[1]=amp_w` | `output[0]=conv_output`, `output[1]=bn_sum`, `output[2]=bn_sq_sum` | 与独立 CONV_AMP_FWD 一致 |
| BN FWD | `input[0]=conv_output`, `input[1]=bn_w`, `input[2]=bn_b`, `input[3]=next_mean`, `input[4]=next_var`, `input[5]=eps`, `input[6]=momentum` | `output[0]=bn_output`, `output[1]=saved_mean`, `output[2]=saved_inv_var` | next_mean/next_var 原地更新到 input[3]/input[4] |
| ReLU FWD | `input[0]=bn_output` | `output[0]=relu_output`, `output[1]=relu_mask` | |

#### CBR_AMP_BWD

```
// input_ids:
//   [0] = dY                (Compiler 注入)
//   [1] = amp_w             (CBR 张量索引 4)
//   [2] = bn_weight         (CBR 张量索引 8)
//   [3] = saved_mean        (CBR 张量索引 19)
//   [4] = saved_inv_var     (CBR 张量索引 20)
//   [5] = relu_mask         (CBR 张量索引 22)
//   [6] = X                 (Compiler 追加)

// output_ids:
//   [0] = dX target         (Compiler in-place 注入，不能写入)
//   [1] = conv_amp_g        (CBR 张量索引 5)
//   [2] = bn_weight_grad    (CBR 张量索引 15)
//   [3] = bn_bias_grad      (CBR 张量索引 16)
//   [4] = conv_output scratch (CBR 张量索引 1，BN BWD 的 dX 输出，Conv BWD 的 dY 输入)
//   [5] = bn_output scratch   (CBR 张量索引 10，ReLU BWD 的 dX 输出，BN BWD 的 dY 输入)
```

子操作映射（以下索引均为 CBR 的 `input_ids`/`output_ids` 全局索引）：

| 子操作 | 输入（CBR input_ids） | 输出（CBR output_ids） | 说明 |
|--------|----------------------|----------------------|------|
| ReLU BWD | `input[0]=dY`, `input[5]=relu_mask` | `output[5]=bn_output(scratch)` | |
| BN BWD | `input[0]=bn_output(scratch)`, `input[2]=bn_w`, `input[3]=saved_mean`, `input[4]=saved_inv_var`（X 复用 `output_ids[4]`） | `output[4]=conv_output(scratch, dX_BN)`, `output[2]=bn_weight_grad(dγ)`, `output[3]=bn_bias_grad(dβ)` | BN BWD 的 X 输入是 `conv_output`，复用 `output_ids[4]` 并原地写回为 `dX_BN`；`input_ids[6]` 是 CBR 层输入 `X_prev`，仅供 Conv BWD 使用 |
| Conv BWD | `input[0]=conv_output(scratch)`, `input[1]=amp_w`, `input[6]=X_prev` | `output[0]=dX`, `output[1]=conv_amp_g` | `input[6]` 是 CBR 层输入 X_prev（Compiler 追加），仅用于 Conv BWD。非首层；首层跳过 dX 写入 |

#### CBR_AMP_INF

```
// input_ids:
//   [0] = X                  (Compiler 注入)
//   [1] = amp_w              (CBR 张量索引 4)
//   [2] = bn_weight          (CBR 张量索引 8)
//   [3] = bn_bias            (CBR 张量索引 9)
//   [4] = eq_scale           (CBR 张量索引 18)
//   [5] = eq_bias            (CBR 张量索引 17)
//   [6] = next_mean          (CBR 张量索引 13)
//   [7] = next_var           (CBR 张量索引 14)
//   [8] = bn_epsilon         (scalar, Compiler 追加)

// output_ids:
//   [0] = conv_output        (CBR 张量索引 1)
//   [1] = bn_sum             (CBR 张量索引 6, reserved)
//   [2] = bn_sq_sum          (CBR 张量索引 7, reserved)
//   [3] = bn_output          (CBR 张量索引 10)
//   [4] = relu_output        (CBR 张量索引 21)
//   [5] = relu_mask          (CBR 张量索引 22)
```

子操作映射（以下索引均为 CBR 的 `input_ids`/`output_ids` 全局索引）：

| 子操作 | 输入（CBR input_ids） | 输出（CBR output_ids） | 说明 |
|--------|----------------------|----------------------|------|
| Conv INF | `input[0]=X`, `input[1]=amp_w` | `output[0]=conv_output` | 纯 conv_fprop，无 GenStats；bn_sum/bn_sq_sum 预留不写入 |
| BN INF | `input[0]=conv_output`, `input[2]=bn_w(gamma)`, `input[3]=bn_b(beta)`, `input[6]=next_mean(rm)`, `input[7]=next_var(rv)`, `input[8]=eps` | `output[3]=bn_output` | 使用 `launch_tr_bn_inf_kernel`；input[1]/[4]/[5]（amp_w/eq_scale/eq_bias）不传给 kernel |
| ReLU INF | `input[0]=bn_output` | `output[4]=relu_output` | |

> **注意**：BN INF 的 `y` 输出到 `output_ids[3]`（bn_output），不是 `output_ids[0]`。`launch_tr_bn_inf_kernel` 签名: `(x, gamma, beta, rm, rv, eps, y, N, C, H, W, is_fp16, stream)`，eq_scale/eq_bias 不在该 kernel 参数中（kernel 内部自行计算）。

---

## 7. 测试方案

### 7.1 测试目标

不对比 PyTorch，只验证 CBR 与独立 `Conv+BN2D+ReLU` 结果一致。

### 7.2 测试文件：`tests/op/test_cbr_amp.cpp`

构建两张等价图：

1. **CBR 路径**：单个 `CBR_AMP_FWD` / `CBR_AMP_BWD` / `CBR_AMP_INF` 节点，使用 23 张量布局。
2. **分立路径**：`CONV_AMP_FWD → BN2D_AMP_FWD → RELU_AMP_FWD`（或反向的 BWD 链）。

共享同一份 X、W、γ、β、running mean/var、dY。

### 7.3 测试内容

1. **FWD 等价性**
   - 对比 `relu_output`、`saved_mean`、`saved_inv_var`、`relu_mask`。
   - 对比更新后的 `next_mean`、`next_var`。
   - 对比 `bn_sum`/`bn_sq_sum`（与独立 `CONV_AMP_FWD` 输出的 `sum`/`sq_sum` 字节级一致）。

2. **BWD 等价性**
   - 对比 `dX`、`conv_weight_grad`、`bn_weight_grad`、`bn_bias_grad`。

3. **BWD_FIRST_LAYER 等价性**
   - 将 CBR 作为首层，验证 `dX` 不被写入，且 `conv_weight_grad` 与分立路径一致。

4. **INF 等价性**
   - 对比 `relu_output`。
   - `bn_sum`/`bn_sq_sum` 在 INF 中预留不写入（与独立 `CONV_AMP_INF` 一致），不做对比。

### 7.4 通过标准

- FP16 张量：由于使用完全相同的 kernel、完全相同的顺序，实际差异应接近 0。最终回归标准 **MSE < 1e-6**。若初期调试因环境抖动难以达到，可临时放宽到 1e-4 定位问题，但合并前必须收紧到 1e-6。
- INT8 mask：逐字节完全一致。
- FP32 统计量（running stats、bn_sum/bn_sq_sum）：MSE < 1e-6。

> **调试提示**：若 MSE > 1e-6，应首先检查：① `sum`/`sq_sum` 是否为真实输出（`set_output(true)`）；② 各子操作的 tensor 映射是否正确（参见 6.11 节索引契约）；③ 流同步是否完整（参见 6.5 节）。

---

## 8. 实施步骤（建议顺序）

1. **枚举与类型**：`op_kind.h` + `op_kind.cpp` + `arch_plan.h`。
2. **ArchPlan 层处理**：`arch_plan_expand.cpp` + `arch_plan_shape.cpp` + `arch_plan_merge.cpp` + `arch_plan_format.cpp` + `arch_plan_yaml.cpp` + `arch_plan_normalize.cpp`。
3. **描述符与编译器**：`layer_descriptor_registry.cpp` + `compiler.cpp`。
4. **注册与策略**：`op_registry.h/.cpp` + `op_stream_policy.cpp`。
5. **后端实现**：新建 `cbr_op.cpp`。
6. **构建系统**：`src/CMakeLists.txt`。
7. **测试**：新建 `test_cbr_amp.cpp` + `tests/op/CMakeLists.txt`。
8. **验证**：全量 `ninja`，运行 `test_cbr_amp` 与现有回归测试。

---

## 9. 风险与注意事项

| 风险 | 影响 | 缓解 |
|------|------|------|
| ReLU 误设为 in-place | 与分立路径张量布局不一致，结果正确但无法字节级对齐 | 已纠正：ReLU 独立输出张量，23 张量布局 |
| sum/sq_sum 误设为虚张量 | cuDNN FE 图结构不同，无法保证 conv_output 字节级一致 | 已纠正：sum/sq_sum 保持真实输出，与独立 CONV_AMP_FWD 完全一致 |
| 张量索引错误 | shape 不匹配、越界 | 严格按 23 张量索引；代码中加注释；测试覆盖；参见 6.11 节索引契约 |
| BWD output_indices 误用 index 3 | Conv AMP BWD 期望 FP16 dW（index 5），用 index 3 会类型崩溃 | 使用 `{5,15,16,1,10}` |
| BN next_mean/next_var 绑定错误 | CBR output_ids.size()=8 导致独立 BN 的 fallback 逻辑误判 | 显式强制绑定到 input_ids（见 6.3.2 节） |
| `has_bn` 遗漏 `LayerKind::CBR` | 分布式训练 BN 统计量不同步、running stats 传递断裂 | 在 `build_auxiliary_graphs` 三处检测中加入 `LayerKind::CBR`（见 5.11.10） |
| `is_bn_like` 遗漏 `LayerKind::CBR` | BN3 检测遗漏，zero-gamma 初始化漏掉 | 加入 `is_bn_like`（见 5.11.11） |
| `init_variant_scalars` 遗漏 CBR | 仅有 CBR 层时 BN 全局参数取默认值 | 增加 CBR 分支（见 5.11.12） |
| Conv/BN graph cache 符号冲突 | 多重定义或链接失败 | 完整复制并重命名所有符号（Conv 见 6.3.1，BN 见 6.3.2） |
| 首层 BWD 误写 `output_ids[0]` | 破坏层输入缓冲区 | 首层 BWD launch 跳过 dX 写入（见 6.8 注意） |
| 复制 BN 代码时照搬 `node.params.bn()` | 取到默认值，行为偏离 | 改为 `node.params.cbr().bn()`（见 6.3.4） |
| 流同步遗漏 | 结果非确定性错误 | 复用子操作内部的 event wait/record（见 6.5 节） |
| 首层 BWD 双缓冲错误 | 训练首层崩溃 | 严格复用 Conv 首层 BWD 的 A/B 逻辑 |
| CBR 自动合并与 Block 融合冲突 | Block 检测失败 | step9 在 step7 之后执行 |
| 结果与分立路径存在 FP16 累积差异 | 测试 MSE 超标 | kernel 相同、顺序相同；调试阶段先按 MSE < 1e-6 验证；若超标按 7.4 节调试提示排查 |

---

## 10. 结论

本最终方案通过：

1. **23 张量布局**：严格保留 `conv_output`、`bn_output`、`relu_output` 三个独立特征图，与分立路径对齐；
2. **`CbrLayerParams`**：清晰的 ArchLayer 参数类型，保留 eps/momentum；
3. **复制核心逻辑**：在独立 `cbr_op.cpp` 中完整实现三个子操作，不侵入现有算子；
4. **完整的 Compiler 改造**：覆盖 `is_weight_bearing`、首层注入、BWD 特化、BN 标量注入、梯度追踪等所有边角；
5. **对比测试**：验证 CBR 与 `Conv+BN2D+ReLU` 字节级一致。

可在不破坏现有 Block 融合与 FP32 路径的前提下，完成 AMP 版 CBR 融合算子。
