# 融合算子开发教程

> **版本**: V1.0  
> **更新日期**: 2026-06-21  
> **适用范围**: Renaissance 框架，当前仅支持 AMP 模式  
> **示例**: CBR（Conv + BN2D + ReLU）融合算子

---

## 目录

1. [架构概览：框架分层与数据流](#一架构概览框架分层与数据流)
2. [设计原则与约束](#二设计原则与约束)
3. [Step 1 — 算子枚举与参数定义](#三step-1--算子枚举与参数定义)
4. [Step 2 — BluePrint 用户 API](#四step-2--blueprint-用户-api)
5. [Step 3 — ArchPlan 层类型与融合规则](#五step-3--archplan-层类型与融合规则)
6. [Step 4 — LayerDescriptor：张量推断与子图模式](#六step-4--layerdescriptor张量推断与子图模式)
7. [Step 5 — Compiler 编译器改造](#七step-5--compiler-编译器改造)
8. [Step 6 — 后端实现与算子注册](#八step-6--后端实现与算子注册)
9. [Step 7 — 分布式训练适配](#九step-7--分布式训练适配)
10. [Step 8 — 测试与验证](#十step-8--测试与验证)
11. [完整检查清单](#十一完整检查清单)
12. [附录：CBR 实施涉及的全部文件](#十二附录cbr-实施涉及的全部文件)

---

## 一、架构概览：框架分层与数据流

### 1.1 五层架构

Renaissance 框架从用户代码到 GPU 执行，经过五个层次：

```
BluePrint DSL         用户编写模型结构（cbr(64, 3, 1, 1)）
    │
    ▼
ArchPlan              中间表示层（LayerKind + LayerParam + 形状）
    │  ├─ expand:    BluePrint → ArchPlan 展开
    │  ├─ merge:     AMP 下自动融合相邻算子
    │  └─ shape:     推导每层输出形状
    │
    ▼
LayerDescriptor       张量布局 + 子图模式（每个 LayerKind 的"知识契约"）
    │  ├─ infer_tensors:   返回该层所有张量（输入/输出/权重/中间量）
    │  ├─ build_forward:   前向子图（ComputeOp → 输入/输出张量索引）
    │  ├─ build_backward:  反向子图
    │  └─ build_inference: 推理子图
    │
    ▼
Compiler               生成 ComputationGraph（GraphNode + 内存分配 + 标量注入）
    │
    ▼
Backend                执行（launch_cuda / cuDNN graph / CUDA kernel）
```

### 1.2 关键数据结构

| 结构 | 所在层 | 作用 |
|------|--------|------|
| `NodeKind` / `CBRParam` | BluePrint | 用户 DSL 的节点类型与参数 |
| `LayerKind` / `CbrLayerParams` | ArchPlan | 中间表示的层类型与参数 |
| `ComputeOp` / `CBRParams` | 跨层 | 后端算子的枚举与参数包 |
| `TensorDesc` | LayerDescriptor | 描述一个张量的名称、形状、Region、数据类型 |
| `SubgraphPattern` | LayerDescriptor | 描述一个 ComputeOp 的输入/输出张量索引 |
| `GraphNode` | Compiler → Backend | 计算图中的节点，携带 input_ids/output_ids |

### 1.3 融合算子的特殊性

融合算子同时具备多个子算子的属性。以 CBR 为例：

- 它**有 Conv 权重**（需要 `is_weight_bearing = true`）
- 它**包含 BN**（需要 `is_bn_like = true`，需要 BN 统计量同步）
- 它**可作为首层**（需要首层注入 `I_A_DATA` / `I_B_DATA`）
- 它**输出 ReLU 结果**（`get_layer_output_id` 返回 `relu_output` 的索引）

框架中所有 `layer.kind == LayerKind::Conv` 或 `layer.kind == LayerKind::Bn2d` 的判断，都要检查是否需要加入 `LayerKind::CBR`。

---

## 二、设计原则与约束

### 2.1 AMP 专用

当前框架**仅在 AMP 模式下支持融合算子**。CBR 只实现四个变体：

| ComputeOp | 用途 |
|-----------|------|
| `CBR_AMP_FWD` | 前向传播 |
| `CBR_AMP_BWD` | 反向传播 |
| `CBR_AMP_BWD_FIRST_LAYER` | 首层反向传播（仅 wgrad，无 dgrad） |
| `CBR_AMP_INF` | 推理 |

不需要实现 FP32 版本或 CPU 版本。CPU 路径注册为 `launch_cbr_amp_cpu_not_supported`，调用时抛出 `NotImplementedError`。

### 2.2 字节级一致性

融合算子的结果必须与分立算子组合（Conv + BN2D + ReLU）**字节级一致**（MSE = 0）。这意味着：

- **不修改任何 kernel 实现**：Conv 用 cuDNN FE graph，BN 用 cuDNN FE graph，ReLU 用手写 CUDA kernel——与分立算子完全相同。
- **不优化中间张量**：即使 Conv 的输出只被 BN 消费、BN 的输出只被 ReLU 消费，仍然完整写入中间张量，与分立路径一致。
- **不转发到不同算子**：例如 `CBR_AMP_FWD` 中的 Conv 子操作必须调用 `CONV_AMP_FWD` 的 kernel，不能转发到 `CONV_AMP_INF`。

### 2.3 张量布局规则

融合算子的张量 = 各子算子张量的**并集**，共享中间特征图张量：

```
n_cbr = n_conv + n_bn + n_relu - 2
```

减 2 是因为 Conv 的输出 = BN 的输入（1 个共享张量），BN 的输出 = ReLU 的输入（1 个共享张量）。CBR 共 23 张量：

| 来源 | 张量数 | 内容 |
|------|--------|------|
| Conv（含 bn_stats） | 8 | X, W, Y, sum, sq_sum, amp_w, amp_g, amp_m |
| BN | 13 | scale, bias, saved_mean, saved_inv_var, next_rm, next_rv, dX, dS, dB, running_mean, running_var, ... |
| ReLU | 2 | relu_output, relu_mask |

### 2.4 流同步规约

CBR 内部子操作可能跨多个 CUDA stream：
- Conv → `COMP_1`
- BN → `COMP_2`
- ReLU → `COMP_3`

**FWD 链**: `Conv(COMP_1) → BN(COMP_2) → ReLU(COMP_3)`  
**BWD 链**: `ReLU(COMP_3) → BN(COMP_2) → Conv wgrad(COMP_1) / dgrad(COMP_3)`

每个子操作结束后记录 `cudaEventRecord`，下一子操作前通过 `cudaStreamWaitEvent` 等待。**代表流**（`get_op_default_stream`）统一设为 `COMP_1`，仅用于框架层面的依赖分析。

### 2.5 Graph Cache 隔离

CBR 的 Conv 和 BN 子操作使用 cuDNN FE graph，需要独立的 cache 变量（如 `s_cbr_conv_fwd_cache`），不能与分立算子共享 cache 变量名，否则会导致链接符号冲突。

---

## 三、Step 1 — 算子枚举与参数定义

**涉及文件**: `include/renaissance/graph/op_kind.h` + `src/graph/op_kind.cpp`

### 3.1 定义参数包

融合算子的参数是各子算子参数的组合。CBR 复用已有的 `ConvParams` 和 `BNParams`：

```cpp
// include/renaissance/graph/op_kind.h
struct CBRParams {
    ConvParams conv;
    BNParams bn;
};
```

### 3.2 新增 ComputeOp 枚举

为融合算子的四个变体添加枚举值（注意放在 `COUNT` 之前）：

```cpp
enum class ComputeOp : uint16_t {
    // ... 已有枚举 ...
    CBR_AMP_FWD,
    CBR_AMP_BWD,
    CBR_AMP_BWD_FIRST_LAYER,
    CBR_AMP_INF,
    COUNT,
};
```

### 3.3 加入 OpParams variant

`OpParams` 是一个 `std::variant`，需要加入 `CBRParams` 并提供构造函数和访问器：

```cpp
struct OpParams {
    std::variant<ConvParams, BNParams, CBRParams, ...> data;

    explicit OpParams(CBRParams p) : data(std::move(p)) {}
    const CBRParams& cbr() const { return std::get<CBRParams>(data); }
};
```

### 3.4 补充名称字符串

在 `src/graph/op_kind.cpp` 中：

- `compute_op_to_string()`: 为四个 `CBR_AMP_*` 返回可读名称。
- `format_op_params()`: 格式化 `CBRParams` 为调试字符串（如 `"out_ch=64, k=3, s=1, p=1, eps=1e-5, momentum=0.1"`）。

---

## 四、Step 2 — BluePrint 用户 API

**涉及文件**: `include/renaissance/graph/blueprint.h`

### 4.1 新增 NodeKind

在 `detail::NodeKind` 枚举中添加：

```cpp
enum class NodeKind : uint16_t {
    // ... 已有节点 ...
    CBR,  // 融合 Conv + BN2D + ReLU
};
```

### 4.2 定义参数结构体

```cpp
struct CBRParam {
    int out_ch, k, s, p;
    float momentum = 0.1f;
    float eps = 1e-5f;
};
```

### 4.3 提供工厂函数

```cpp
inline Layer cbr(int out_ch, int k, int s, int p,
                 double momentum = 0.1, double eps = 1e-5) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::CBR,
        detail::CBRParam{out_ch, k, s, p,
                         static_cast<float>(momentum),
                         static_cast<float>(eps)}));
}

// 别名
inline Layer conv_bn_relu(int out_ch, int k, int s, int p,
                          double momentum = 0.1, double eps = 1e-5) {
    return cbr(out_ch, k, s, p, momentum, eps);
}
```

> **注意**：`cbr()` 和 `conv_bn_relu()` 产生完全相同的 `Layer` 对象。两者参数列表与 `conv()` 一致，额外多了 `momentum` 和 `eps`。

---

## 五、Step 3 — ArchPlan 层类型与融合规则

**涉及文件**: 8 个文件

### 5.1 新增 LayerKind 与 LayerParam

**文件**: `include/renaissance/graph/arch_plan.h`

```cpp
enum class LayerKind : uint16_t {
    Conv, Bn1d, Bn2d, ReLU, ..., CBR, ...
};

struct CbrLayerParams {
    int out_ch, k, s, p;
    float eps = 1e-5f;
    float momentum = 0.1f;
    bool operator==(const CbrLayerParams& o) const { ... }
};

using LayerParam = std::variant<..., CbrLayerParams, EmptyParams>;
```

### 5.2 BluePrint → ArchPlan 展开

**文件**: `src/graph/arch_plan_expand.cpp`

当 BluePrint 中有 `NodeKind::CBR` 节点时：

- **AMP 模式（`fuse == true`）**：直接产出 `LayerKind::CBR`，参数从 `CBRParam` 转为 `CbrLayerParams`。
- **非 AMP 模式**：拆分为 `Conv → Bn2d → ReLU` 三个独立层。

```cpp
case NodeKind::CBR: {
    auto& p = std::get<CBRParam>(node.payload);
    if (fuse) {
        // AMP：保持融合
        out.push_back({LayerKind::CBR,
                       CbrLayerParams{p.out_ch, p.k, p.s, p.p, p.eps, p.momentum},
                       "cbr", {}, {}, false, false, src_id});
    } else {
        // 非 AMP：拆分为独立三层
        out.push_back({LayerKind::Conv, ConvLayerParams{p.out_ch, p.k, p.s, p.p}, ...});
        out.push_back({LayerKind::Bn2d, BNParams{p.eps, p.momentum}, ...});
        out.push_back({LayerKind::ReLU, EmptyParams{}, ...});
    }
    current_c = p.out_ch;
    break;
}
```

### 5.3 自动融合规则

**文件**: `src/graph/arch_plan_merge.cpp`

**仅 AMP 模式下**，自动将连续的 `Conv + Bn2d + ReLU` 合并为 `CBR`。这意味着用户即使写分立算子，在 AMP 下也会被自动融合。

```cpp
void ArchPlan::step9_merge_triple() {
    if (!GlobalRegistry::instance().using_amp()) return;

    auto build_cbr = [](const ArchLayer& conv, const ArchLayer& bn,
                        const ArchLayer& /*relu*/) -> LayerParam {
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

### 5.4 ArchPlan 工具函数

以下 4 个文件需要为 `LayerKind::CBR` 添加 case 分支：

| 文件 | 作用 | 实现 |
|------|------|------|
| `arch_plan_shape.cpp` | 形状推导 | 使用 Conv 公式 `(H + 2P - K) / S + 1` 计算输出 H/W，C = `out_ch` |
| `arch_plan_normalize.cpp` | 输出通道 | 返回 `CbrLayerParams::out_ch` |
| `arch_plan_format.cpp` | 格式化 | 返回 `"CBR"` 和参数字符串 `"out_ch=64, k=3, s=1, p=1"` |
| `arch_plan_yaml.cpp` | YAML 序列化 | 读写 `CBR` 层类型和参数，支持从 YAML 配置文件加载 |

---

## 六、Step 4 — LayerDescriptor：张量推断与子图模式

**涉及文件**: `src/graph/layer_descriptor_registry.cpp`

这是融合算子的**核心设计环节**。每个 `LayerKind` 都在此注册一个 `LayerDescriptor`：

```cpp
struct LayerDescriptor {
    InferFn infer_tensors;    // 张量推断
    BuildFn build_forward;    // 前向子图模式
    BuildFn build_backward;   // 反向子图模式
    BuildFn build_inference;  // 推理子图模式
};
```

### 6.1 张量推断 `infer_tensors`

给定输入形状，返回该层需要的**所有**张量（输入、输出、权重、梯度、中间量）的 `TensorDesc` 列表。融合算子的张量 = 各子算子张量的并集。

```
infer_tensors(input_shape, params, ctx) → vector<TensorDesc>
```

CBR 的实现逻辑：

```cpp
std::vector<TensorDesc> infer_cbr_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    const auto& cbr = std::get<CBRParams>(params.data);

    // 1. Conv 部分（8 张量，含 bn_sum/bn_sq_sum）
    OpParams conv_op{ConvParams{cbr.conv}};
    auto conv_d = infer_conv_tensors_with_bn_stats(input, conv_op, ctx);
    descs.insert(descs.end(), conv_d.begin(), conv_d.end());

    // 2. BN 部分（13 张量）
    Shape conv_out = compute_conv_output(input, cbr.conv);
    OpParams bn_op{BNParams{cbr.bn}};
    auto bn_d = infer_bn_tensors(conv_out, bn_op, ctx);
    descs.insert(descs.end(), bn_d.begin(), bn_d.end());

    // 3. ReLU 部分（2 张量：独立输出 + mask）
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    descs.push_back({"cbr_relu_output", conv_out, select_feature_region(ctx), feat_dt});
    descs.push_back({"cbr_relu_mask",   conv_out, Region::S_MASK,        DType::INT8});

    return descs;  // 共 23 张量
}
```

> **AMP 通道对齐检查**：由于 CBR 内部包含 BN 子操作，需检查 `conv_out.c()` 是否为 8 的倍数（cuDNN TensorCore 要求），与 `infer_bn_tensors` 保持一致。

### 6.2 子图模式 `build_forward / backward / inference`

每个子图模式描述一个 `ComputeOp` 节点，其输入/输出是张量列表中的索引。

以 CBR_AMP_FWD 为例：

```cpp
SubgraphPattern build_cbr_forward(const OpParams&, const vector<TensorDesc>& descs) {
    SubgraphPattern p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_FWD;

    // 输入索引: amp_w(4), bn_weight(8), bn_bias(9), next_mean(13), next_var(14)
    // X(0) 和 bn_epsilon/bn_momentum 由 Compiler 后续注入，不在此声明
    n.input_indices  = {4, 8, 9, 13, 14};

    // 输出索引: conv_output(1), bn_sum(6), bn_sq_sum(7), bn_output(10),
    //          saved_mean(19), saved_inv_var(20), relu_output(21), relu_mask(22)
    n.output_indices = {1, 6, 7, 10, 19, 20, 21, 22};

    p.nodes.push_back(n);
    return p;
}
```

> **关键约定**：`input_indices` 和 `output_indices` 是相对于 `infer_tensors` 返回的 `TensorDesc` 列表的索引。Compiler 会将它们映射为全局张量 ID。

### 6.3 注册描述符

```cpp
static const LayerDescriptor cbr_desc = {
    infer_cbr_tensors,
    build_cbr_forward,
    build_cbr_backward,
    build_cbr_inference
};

const LayerDescriptor& get_layer_descriptor(LayerKind kind) {
    switch (kind) {
        // ... 其他 case ...
        case LayerKind::CBR: return cbr_desc;
    }
}
```

### 6.4 输出张量索引

指定该层对外可见的输出张量索引。CBR 的输出是 `relu_output`（张量索引 21）：

```cpp
int get_layer_output_tensor_index(LayerKind kind) {
    switch (kind) {
        // ... 其他 case ...
        case LayerKind::CBR: return 21;  // relu_output
    }
}
```

---

## 七、Step 5 — Compiler 编译器改造

**涉及文件**: `src/graph/compiler.cpp`

Compiler 是变更最密集的模块。需要处理以下 10 个方面：

### 7.1 层分发

在编译主循环中为 `LayerKind::CBR` 添加分支：

```cpp
case LayerKind::CBR:
    compile_cbr(layer, idx);
    break;
```

### 7.2 参数转换

`convert_to_op_params()` 将 `CbrLayerParams` 转为 `CBRParams`：

```cpp
if (std::holds_alternative<CbrLayerParams>(lp)) {
    auto& p = std::get<CbrLayerParams>(lp);
    ConvParams cp{p.out_ch, p.k, p.k, p.s, p.s, p.p, p.p, 1, 1, 1};
    BNParams bp{p.eps, p.momentum};
    return OpParams{CBRParams{cp, bp}};
}
```

### 7.3 权重分配

`compile_cbr()` 需要同时分配：
- **Conv 权重**：通过 `alloc_conv_group()` 分配 W 和 amp_w
- **BN 参数**：通过 `alloc_bn_group()` 分配 scale、bias、running_mean、running_var
- **BN 统计量**：通过 `memory_plan_.alloc_bn_stats()` 分配

### 7.4 属性判断

CBR 同时具备 Conv、BN、ReLU 的属性：

```cpp
// CBR 包含 BN 子操作 → 需要 BN 标量注入（eps/momentum）
static bool is_bn_like(LayerKind k) {
    switch (k) {
        case LayerKind::Bn1d: case LayerKind::Bn2d:
        case LayerKind::CBR: return true;
        default: return false;
    }
}

// CBR 包含 Conv 和 BN 权重 → 需要权重持久化
static bool is_weight_bearing(LayerKind k) {
    switch (k) {
        case LayerKind::Conv: case LayerKind::Bn1d: case LayerKind::Bn2d:
        case LayerKind::CBR: case LayerKind::FC: return true;
        default: return false;
    }
}
```

### 7.5 首层处理

- **首层输入注入**：CBR 作为首层时，FWD/INF 需要注入 `I_A_DATA` / `I_B_DATA`（与 Conv/FC 一致）。
- **首层合法性校验**：`validate_first_layer()` 允许 `LayerKind::CBR` 作为首层。
- **首层 BWD 算子映射**：`to_first_layer_bwd_op()` 将 `CBR_AMP_BWD` 映射为 `CBR_AMP_BWD_FIRST_LAYER`。

### 7.6 标量注入

CBR 的 BN 子操作需要全局标量：

```cpp
// FWD: 注入 bn_epsilon + bn_momentum
if (gn.compute_op == ComputeOp::CBR_AMP_FWD) {
    if (b.bn_epsilon >= 0)  gn.input_ids.push_back(b.bn_epsilon);
    if (b.bn_momentum >= 0) gn.input_ids.push_back(b.bn_momentum);
}

// INF: 仅注入 bn_epsilon
if (gn.compute_op == ComputeOp::CBR_AMP_INF) {
    if (b.bn_epsilon >= 0) gn.input_ids.push_back(b.bn_epsilon);
}
```

### 7.7 BWD 双缓冲

CBR 的反向传播需要双缓冲（与 Conv BWD 一致），通过辅助函数 `is_cbr_bwd_op()` 判断：

```cpp
inline bool is_cbr_bwd_op(ComputeOp op) {
    return op == ComputeOp::CBR_AMP_BWD || op == ComputeOp::CBR_AMP_BWD_FIRST_LAYER;
}

// 在双缓冲分配处
if (is_conv_bwd_op(gn.compute_op) || is_cbr_bwd_op(gn.compute_op)) {
    // 分配双缓冲
}
```

### 7.8 输出/梯度输出索引

```cpp
// 前向输出：relu_output（索引 21）
auto get_layer_output_id = [](LayerKind kind, ...) -> int32_t {
    switch (kind) {
        case LayerKind::CBR: return 21; break;
        // ...
    }
};

// 梯度输出：dX in-place 到 X（索引 -1 表示 in-place）
auto get_grad_output_id = [](LayerKind kind, ...) -> int32_t {
    switch (kind) {
        case LayerKind::CBR: return -1; break;  // dX in-place to X
        // ...
    }
};
```

### 7.9 BN 统计量维护

在 `build_auxiliary_graphs()` 中，检测 `has_bn` 时需要将 `LayerKind::CBR` 加入判断，否则分布式训练时 BN running stats 无法同步：

```cpp
bool has_bn = false;
for (const auto& layer : arch.layers()) {
    auto k = layer.kind;
    if (k == LayerKind::Bn1d || k == LayerKind::Bn2d || k == LayerKind::CBR) {
        has_bn = true;
        break;
    }
}
```

同时，需要校验所有 BN/CBR 使用相同的 `eps/momentum`。

### 7.10 CBR 专用的 BWD 辅助函数

```cpp
inline bool is_cbr_bwd_op(ComputeOp op) {
    return op == ComputeOp::CBR_AMP_BWD || op == ComputeOp::CBR_AMP_BWD_FIRST_LAYER;
}
```

这个函数在双缓冲分配、首层 BWD 处理等场景中与 `is_conv_bwd_op()` 并列使用。

---

## 八、Step 6 — 后端实现与算子注册

**涉及文件**: 4 个文件

### 8.1 后端实现文件

**新建**: `src/backend/ops/dtensor/cbr_op.cpp`

实现四个 launch 函数：

```cpp
static void launch_cbr_amp_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state);

static void launch_cbr_amp_bwd_cuda(...);
static void launch_cbr_amp_bwd_first_layer_cuda(...);
static void launch_cbr_amp_inf_cuda(...);
```

**实现要点**：

1. **子操作顺序调用**：FWD = Conv FWD → BN FWD → ReLU FWD；BWD = ReLU BWD → BN BWD → Conv BWD/wgrad。

2. **流同步**：每个子操作结束后 `cudaEventRecord(last_done_event)`，下一子操作前 `cudaStreamWaitEvent`。不要使用 `cudaStreamSynchronize` 或 `cudaDeviceSynchronize`。

3. **Graph Cache 独立**：Conv 和 BN 的 cuDNN FE graph cache 需要复制并重命名（如 `s_cbr_conv_fwd_cache`），避免与分立算子的 cache 符号冲突。构建逻辑完全一致，直接复制。

4. **ReLU 使用外部 kernel**：通过 `extern` 声明引用 `relu_op.cu` 中的 kernel 函数。

5. **`padded_elems()`**：ReLU 子操作的元素数量使用 `dt.padded_elems()`，与分立 ReLU AMP 一致。

6. **`output_stream_idx`**：每个子操作结束后设置 `state.output_stream_idx` 为当前子操作的流索引，下一个子操作据此等待。

7. **CPU 不支持**：注册 `launch_cpu = launch_cbr_amp_cpu_not_supported`。

### 8.2 算子注册

**文件**: `include/renaissance/backend/op_registry.h` + `src/backend/op_registry.cpp`

```cpp
// 声明
void register_op_cbr();

// 在 register_default_ops() 中调用
void register_default_ops() {
    register_op_conv();
    register_op_bn();
    register_op_relu();
    register_op_cbr();   // 新增
}
```

**Warmup 支持**：`require_warmup()` 和 `warmup_single_cudnn_op()` 中需要加入四个 `CBR_AMP_*` 枚举值。

### 8.3 流策略

**文件**: `src/backend/op_stream_policy.cpp`

```cpp
case ComputeOp::CBR_AMP_FWD:
case ComputeOp::CBR_AMP_BWD:
case ComputeOp::CBR_AMP_BWD_FIRST_LAYER:
case ComputeOp::CBR_AMP_INF:
    return StreamKind::COMP_1;
```

代表流仅用于框架层面 `insert_cross_op_barrier` 的依赖分析。CBR 内部子操作按各自需要的流执行。

### 8.4 构建系统

**文件**: `src/CMakeLists.txt`

将 `cbr_op.cpp` 加入源文件列表：

```cmake
backend/ops/dtensor/cbr_op.cpp
```

---

## 九、Step 7 — 分布式训练适配

**涉及文件**: `src/task/deep_learning_task.cpp`

在 `init_variant_scalars()` 中，为 CBR 层提取 BN 的 eps/momentum 参数（分布式训练需要这些参数）：

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

---

## 十、Step 8 — 测试与验证

**涉及文件**: 5 个新文件 + `tests/op/CMakeLists.txt`

### 10.1 测试策略：等价性对比

为每个 ComputeOp 变体（FWD / BWD / BWD_FIRST_LAYER / INF）编写独立测试。每个测试构造两条等价路径：

- **CBR 路径**：使用 `CBR_AMP_*` 算子
- **分立路径**：使用 `CONV_AMP_*` + `BN2D_AMP_*` + `RELU_AMP_*` 算子

两者使用相同的输入和初始权重，对比所有关键输出张量的 MSE。MSE = 0 即字节级一致。

### 10.2 测试文件

| 文件 | 测试内容 |
|------|----------|
| `tests/op/cbr_amp_test_utils.hpp` | 共享工具：参数生成、权重初始化、MSE 计算 |
| `tests/op/test_cbr_amp_fwd.cpp` | 前向传播等价性 |
| `tests/op/test_cbr_amp_inf.cpp` | 推理等价性 |
| `tests/op/test_cbr_amp_bwd.cpp` | 反向传播等价性 |
| `tests/op/test_cbr_amp_bwd_first_layer.cpp` | 首层反向传播等价性 |

### 10.3 对比的关键张量

| 测试 | 对比张量 |
|------|----------|
| FWD | relu_output, relu_mask, saved_mean, saved_inv_var, next_mean, next_var, bn_sum, bn_sq_sum |
| INF | relu_output, next_mean, next_var |
| BWD | dX, dW, dS, dB |
| BWD_FIRST_LAYER | dW, dS, dB（无 dX） |

> **注意**：`relu_mask` 是 INT8 类型，需要使用 `compute_mse_int8` 而非 `compute_mse_fp32`。

### 10.4 CMake 配置

在 `tests/op/CMakeLists.txt` 中添加 4 个测试目标：

```cmake
foreach(cbr_test IN ITEMS test_cbr_amp_fwd test_cbr_amp_bwd
                         test_cbr_amp_bwd_first_layer test_cbr_amp_inf)
    add_executable(${cbr_test} ${cbr_test}.cpp)
    target_link_libraries(${cbr_test} PRIVATE renaissance)
endforeach()
```

---

## 十一、完整检查清单

创建融合算子时，按以下清单逐项确认：

### 枚举与参数（3 项）
- [ ] `op_kind.h`：新增 `CBRParams` 结构体 + `ComputeOp` 枚举值 + `OpParams` variant/访问器
- [ ] `op_kind.cpp`：新增 `compute_op_to_string` 与 `format_op_params` 分支
- [ ] `blueprint.h`：新增 `NodeKind` + 参数结构体 + 工厂函数

### ArchPlan 层（6 项）
- [ ] `arch_plan.h`：新增 `LayerKind` + `CbrLayerParams` + `LayerParam` variant
- [ ] `arch_plan_expand.cpp`：BluePrint → ArchPlan 展开（AMP 保持 / 非 AMP 拆分）
- [ ] `arch_plan_merge.cpp`：自动融合规则（仅 AMP 模式）
- [ ] `arch_plan_shape.cpp`：形状推导
- [ ] `arch_plan_format.cpp` + `arch_plan_yaml.cpp`：格式化与序列化
- [ ] `arch_plan_normalize.cpp`：输出通道读取

### LayerDescriptor（4 项）
- [ ] `infer_*_tensors`：张量布局（子算子张量并集，共享中间张量）
- [ ] `build_*_forward / backward / inference`：子图模式（input/output 索引）
- [ ] 注册 `cbr_desc` 到 `get_layer_descriptor()`
- [ ] `get_layer_output_tensor_index`：指定对外输出张量索引

### Compiler（8 项）
- [ ] `compile_cbr()`：权重分配 + 形状计算
- [ ] `convert_to_op_params()`：`CbrLayerParams` → `CBRParams` 转换
- [ ] `is_bn_like()` / `is_weight_bearing()`：属性判断
- [ ] 首层处理：输入注入 + 合法性校验 + `to_first_layer_bwd_op()`
- [ ] 标量注入：FWD 注入 `bn_epsilon` + `bn_momentum`；INF 注入 `bn_epsilon`
- [ ] BWD 双缓冲：`is_cbr_bwd_op()` 辅助函数
- [ ] `get_layer_output_id()` / `get_grad_output_id()`：输出/梯度输出索引
- [ ] `build_auxiliary_graphs()`：`has_bn` 检测 + BN 参数一致性校验

### 后端（5 项）
- [ ] `cbr_op.cpp`（新建）：4 个 launch 函数 + `register_op_cbr()`
- [ ] `op_registry.h` + `op_registry.cpp`：声明/调用注册 + warmup 支持
- [ ] `op_stream_policy.cpp`：默认流策略
- [ ] `src/CMakeLists.txt`：加入 `cbr_op.cpp`

### 分布式训练（1 项）
- [ ] `deep_learning_task.cpp`：`init_variant_scalars()` 中提取 CBR 的 eps/momentum

### 测试（2 项）
- [ ] 测试文件（4 个）：FWD / INF / BWD / BWD_FIRST_LAYER 等价性测试
- [ ] `tests/op/CMakeLists.txt`：测试目标

---

## 十二、附录：CBR 实施涉及的全部文件

### 修改文件（16 个）

| 文件 | 修改内容 |
|------|----------|
| `include/renaissance/graph/op_kind.h` | `CBRParams`、`ComputeOp` 枚举、`OpParams` variant |
| `src/graph/op_kind.cpp` | 名称字符串、格式化输出 |
| `include/renaissance/graph/blueprint.h` | `NodeKind::CBR`、`CBRParam`、`cbr()`/`conv_bn_relu()` |
| `include/renaissance/graph/arch_plan.h` | `LayerKind::CBR`、`CbrLayerParams`、`LayerParam` variant |
| `src/graph/arch_plan_expand.cpp` | BluePrint 展开（AMP 保持 / 非 AMP 拆分） |
| `src/graph/arch_plan_merge.cpp` | `step9_merge_triple()` 自动融合 |
| `src/graph/arch_plan_shape.cpp` | 形状推导 |
| `src/graph/arch_plan_normalize.cpp` | 输出通道读取 |
| `src/graph/arch_plan_format.cpp` | 格式化打印 |
| `src/graph/arch_plan_yaml.cpp` | YAML 序列化 |
| `src/graph/layer_descriptor_registry.cpp` | 张量推断、子图模式、描述符注册 |
| `src/graph/compiler.cpp` | 编译器全链路适配（10 个方面） |
| `include/renaissance/backend/op_registry.h` | `register_op_cbr()` 声明 |
| `src/backend/op_registry.cpp` | 注册调用、warmup |
| `src/backend/op_stream_policy.cpp` | 默认流策略 |
| `src/task/deep_learning_task.cpp` | eps/momentum 提取 |
| `src/CMakeLists.txt` | 加入 `cbr_op.cpp` |
| `tests/op/CMakeLists.txt` | 测试目标 |

### 新建文件（6 个）

| 文件 | 内容 |
|------|------|
| `src/backend/ops/dtensor/cbr_op.cpp` | 后端实现（4 个 launch 函数 + 算子注册） |
| `tests/op/cbr_amp_test_utils.hpp` | 测试共享工具 |
| `tests/op/test_cbr_amp_fwd.cpp` | FWD 等价性测试 |
| `tests/op/test_cbr_amp_inf.cpp` | INF 等价性测试 |
| `tests/op/test_cbr_amp_bwd.cpp` | BWD 等价性测试 |
| `tests/op/test_cbr_amp_bwd_first_layer.cpp` | BWD_FIRST_LAYER 等价性测试 |