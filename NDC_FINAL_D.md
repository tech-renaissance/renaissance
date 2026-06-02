# MaxPool 与 Dropout 最终实现方案（NDC_FINAL）

> 版本：FINAL v4  
> 日期：2026-06-02  
> 修订：根据 R_REV3.md 审查意见修正（P0-1 + P1-2/3 + P2-4 + P3-5 + 正方形核约束）  
> 核心约束：dX 覆盖 X；CPU/GPU 接口对齐；算子内部不临时分配内存；Dropout per-RANK 随机可复现

---

## 一、核心设计铁律

### 1.1 不可违背的原则

| # | 原则 | 说明 |
|---|------|------|
| 1 | **dX 覆盖 X** | BWD 梯度输出写入原始输入 X 的 buffer，不单独分配 grad_slot。MaxPool 删除 `pool_grad_slot`，Dropout 不设 grad_slot。 |
| 2 | **张量数守恒** | 同一 LayerKind 在 FWD/BWD/INF 下 `infer_tensors` 返回张量数量、名称、顺序完全一致。INF 分配 mask 但不写。 |
| 3 | **mask 始终 INT8 / S_MASK** | MaxPool index 和 Dropout mask 统一 `Region::S_MASK`、`DType::INT8`。 |
| 4 | **签名完整性** | 每张量必须出现在 kernel 参数列表和 SubgraphPattern indices 中，即使不被读写。 |
| 5 | **禁止算子内部临时分配内存** | 禁止 `new`/`malloc`/`cudaMalloc`。GPU workspace 用 `ctx.ensure_workspace_grow()`；CPU 临时内存用 `ctx->ensure_cpu_workspace_grow()`（MaxPool/Dropout 不需求）。 |
| 6 | **CPU/GPU 接口对齐** | 同一 ComputeOp 的 CPU launcher 和 CUDA launcher 输入输出张量语义完全一致。 |

### 1.2 数学定义

#### MaxPool

- **约束**：当前框架仅支持正方形核（`kernel_h == kernel_w`，`stride_h == stride_w`，`pad_h == pad_w`），由 `PoolLayerParams`（单字段 `k, s, p`）保证。
- **FWD**：`Y[n,oh,ow,c] = max_{kh,kw} X[n, oh·s+kh-pad, ow·s+kw-pad, c]`，同时生成 INT8 mask（局部偏移 `kh*k+kw`，`k == kernel_h == kernel_w`）。
- **BWD**：`dX[n,h,w,c] = dY[n,h',w',c]` 当且仅当 `(h,w)` 是 `(h',w')` 窗口内 max 位置（DETERMINISTIC：第一个 max）；否则 0。
- **INF**：同 FWD，mask 生成但不消费。

#### Dropout（Inverted Dropout）

| 模式 | 公式 | mask 值 | 说明 |
|------|------|---------|------|
| **FWD** | `Y = X * M / (1-p)` | `M ∈ {0, 1}` | `M ~ Bernoulli(1-p)`，`scale = 1/(1-p)` |
| **BWD** | `dX = dY * M / (1-p)` | `M ∈ {0, 1}` | 同 mask 同 scale |
| **INF** | `Y = X` | 不写入 | **恒等映射**，无随机无缩放 |

---

## 二、MaxPool 实现方案

### 2.1 张量布局（2 张量）

| 索引 | 名称 | 形状 | Region | DType | FWD 读写 | BWD 读写 | INF 读写 |
|------|------|------|--------|-------|----------|----------|----------|
| 0 | `pool_output` | `[N, OH, OW, C]` | F_FEATURE_FP32/FP16 | FP32/FP16 | 写 | 读 | 写 |
| 1 | `pool_mask` | `[N, OH, OW, C]` | S_MASK | INT8 | 写(虚拟) | 不读 | 写(虚拟) |

**关键设计决策**：
- 删除 `pool_grad_slot`（dX 覆盖 X）。
- `pool_index` → `pool_mask`，dtype `INT32` → `INT8`。
- cuDNN FE 中 mask 设为 virtual tensor（`set_output(false)`），不写显存。若版本不支持则 fallback 为 real tensor。
- BWD 的 cuDNN Legacy API 不使用 mask（内部重算 max 位置），但 mask 必须出现在 `input_ids` 中（张量数一致性）。

### 2.2 ComputeOp 枚举拆分

将现有的类型多态 `MAXPOOL_FWD` / `MAXPOOL_BWD` 拆分为 6 个精度分离枚举：

```cpp
// 替换原有 MAXPOOL_FWD / MAXPOOL_BWD
MAXPOOL_FP32_FWD,
MAXPOOL_FP32_BWD,
MAXPOOL_FP32_INF,
MAXPOOL_AMP_FWD,
MAXPOOL_AMP_BWD,
MAXPOOL_AMP_INF,
```

原有的 `MAXPOOL_FWD` / `MAXPOOL_BWD` 确认无其他引用后删除。同步修改 `op_kind.cpp` 的 `compute_op_to_string()`。

### 2.3 LayerDescriptor 修正

**修改文件**：`src/graph/layer_descriptor_registry.cpp`

```cpp
std::vector<TensorDesc> infer_maxpool_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    TR_CHECK(std::holds_alternative<PoolParams>(params.data), ValueError,
             "infer_maxpool_tensors expects PoolParams");
    const auto& pp = std::get<PoolParams>(params.data);

    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    int oh = (input.h() + 2 * pp.pad_h - pp.kernel_h) / pp.stride_h + 1;
    int ow = (input.w() + 2 * pp.pad_w - pp.kernel_w) / pp.stride_w + 1;
    Shape out{input.n(), oh, ow, input.c()};

    return {
        TensorDesc{"pool_output", out, select_feature_region(ctx), feat_dt},  // idx 0
        TensorDesc{"pool_mask",   out, Region::S_MASK,        DType::INT8}    // idx 1
    };
}
```

**变更点**：删除 `pool_grad_slot`，`pool_index`→`pool_mask`，dtype `INT32`→`INT8`。

### 2.4 SubgraphPattern

```cpp
// FWD: X(跨层注入) → Y(idx0), mask(idx1)
SubgraphPattern build_maxpool_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    bool amp = GlobalRegistry::instance().using_amp();
    n.op = amp ? ComputeOp::MAXPOOL_AMP_FWD : ComputeOp::MAXPOOL_FP32_FWD;
    n.output_indices = {0, 1};
    p.nodes.push_back(n);
    return p;
}

// BWD: dY(compiler注入) + Y(idx0) + mask(idx1) → dX(覆盖X, compiler处理)
SubgraphPattern build_maxpool_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    bool amp = GlobalRegistry::instance().using_amp();
    n.op = amp ? ComputeOp::MAXPOOL_AMP_BWD : ComputeOp::MAXPOOL_FP32_BWD;
    n.input_indices  = {0, 1};  // Y, mask (dY 由 compiler 注入)
    n.output_indices = {};      // 空：compiler 处理为 layer_input_ids[l]
    p.nodes.push_back(n);
    return p;
}

// INF: X(跨层注入) → Y(idx0), mask(idx1, 分配但不写)
SubgraphPattern build_maxpool_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    bool amp = GlobalRegistry::instance().using_amp();
    n.op = amp ? ComputeOp::MAXPOOL_AMP_INF : ComputeOp::MAXPOOL_FP32_INF;
    n.output_indices = {0, 1};
    p.nodes.push_back(n);
    return p;
}
```

**BWD input/output 绑定**：
- `node.input_ids[0] = prev_grad_id`（dY，compiler 注入）
- `node.input_ids[1] = Y_id`（pool_output，pattern `input_indices[0]`）
- `node.input_ids[2] = mask_id`（pool_mask，pattern `input_indices[1]`，Legacy API 不使用但必须存在）
- `node.output_ids[0] = X_id`（compiler in-place：`layer_input_ids[l]`）

### 2.5 Compiler 集成

**修改文件**：`src/graph/compiler.cpp`

#### 2.5.1 `get_layer_output_id`（前向输出传递）

```cpp
case LayerKind::Dropout:  idx = 0; break;  // dropout_output at index 0
```

#### 2.5.2 `get_grad_output_id`

```cpp
case LayerKind::MaxPool:  idx = -1; break;  // in-place（原 idx=2 删除）
```

#### 2.5.3 BWD in-place 处理

在 compiler.cpp BWD 遍历中（~line 1288），将 MaxPool BWD 和 Dropout BWD 加入 in-place 分支：

```cpp
if (gn.compute_op == ComputeOp::MAXPOOL_FP32_BWD || gn.compute_op == ComputeOp::MAXPOOL_AMP_BWD ||
    gn.compute_op == ComputeOp::DROPOUT_FP32_BWD || gn.compute_op == ComputeOp::DROPOUT_AMP_BWD ||
    gn.compute_op == ComputeOp::RELU_FP32_BWD || ...) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) {
        gn.output_ids = {it->second};  // dX in-place to X
    }
}
```

#### 2.5.4 `prev_grad_id` fallback 列表

```cpp
if (grad_id < 0 && (layer.kind == LayerKind::FC ||
    layer.kind == LayerKind::FCBNReLU || ... ||
    layer.kind == LayerKind::MaxPool ||    // 新增
    layer.kind == LayerKind::Dropout)) {   // 新增
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) grad_id = it->second;
}
```

#### 2.5.5 `compile_layer` no-op

MaxPool 已在 break 列表中。Dropout 新增：

```cpp
case LayerKind::Dropout:
    break;  // 无权重
```

### 2.6 CUDA 后端实现

#### 2.6.1 FWD / INF：cuDNN Frontend Resample MAXPOOL

**参考实现**：`cbrp_fwd.cpp` 中 `build_maxpool_graph()`（已验证性能）。

技术路径：cuDNN Frontend `resample(MAXPOOL)` + `set_generate_index(true)` 生成 INT8 mask，mask 设为 virtual tensor。使用 per-shape cache（同 GAP 的 `gap_op.cpp` 模式）。

**新建文件**：`src/backend/ops/dtensor/maxpool_op.cpp` + `maxpool_op.cu`

```cpp
// Graph cache key
struct MaxPoolFwdCacheKey {
    uint64_t handle_bits;
    int32_t n, c, h, w, k, s, p;
    bool is_amp;
    bool operator==(const MaxPoolFwdCacheKey& o) const { /* 全字段比较 */ }
};

// cuDNN FE graph 构建（参照 cbrp_fwd.cpp:741-862，使用 shared_ptr<Tensor_attributes> variant pack）
// 注：项目 FE execute 使用 shared_ptr<fe::graph::Tensor_attributes> 作为 variant pack key，
//     而非 int64_t uid（参照 cbrp_fwd.cpp:161-165）

struct MaxPoolFwdCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> x_attr;
    std::shared_ptr<fe::graph::Tensor_attributes> y_attr;
    std::shared_ptr<fe::graph::Tensor_attributes> mask_attr;
    size_t workspace_size = 0;
    bool mask_is_virtual = false;
};

// 构建 cuDNN FE graph
auto graph = std::make_shared<fe::graph::Graph>();
graph->set_io_data_type(is_amp ? fe::DataType_t::HALF : fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

// X tensor：NHWC 通过 set_stride 表达
auto X = graph->tensor(fe::graph::Tensor_attributes()
    .set_name("pool_x")
    .set_dim({N, C, H, W})
    .set_stride({n_stride, c_stride, h_stride, w_stride})
    .set_data_type(to_fe_dtype(dt_x.dtype)));
// 其中 c_stride = 1, w_stride = C, h_stride = W*C, n_stride = H*W*C

// Resample MAXPOOL（参照 cbrp_fwd.cpp:758-769）
// 注：当前框架仅支持正方形核（kernel_h == kernel_w），PoolLayerParams 保证 h==w。
fe::graph::Resample_attributes pool_attr;
pool_attr.set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
         .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
         .set_window({params.kernel_h, params.kernel_w})
         .set_stride({params.stride_h, params.stride_w})
         .set_pre_padding({params.pad_h, params.pad_w})
         .set_post_padding({params.pad_h, params.pad_w})
         .set_generate_index(true)
         .set_compute_data_type(fe::DataType_t::FLOAT);

auto outs = graph->resample(X, pool_attr);

// 保存 shared_ptr 到 cache（后续 execute 用作 variant pack key）
cache.x_attr = X;
cache.y_attr = outs[0];
cache.mask_attr = outs[1];

// Y 输出
outs[0]->set_output(true).set_data_type(to_fe_dtype(dt_y.dtype))...;

// Mask：virtual tensor（参照 cbrp_fwd.cpp:789-803）
try {
    outs[1]->set_output(false).set_data_type(fe::DataType_t::INT8);
    cache.mask_is_virtual = true;
} catch (const fe::cudnnException&) {
    outs[1]->set_output(true).set_data_type(fe::DataType_t::INT8)...;
    cache.mask_is_virtual = false;
}

// 构建 + workspace
graph->validate();
graph->build_operation_graph(handle);
graph->create_execution_plans({fe::HeurMode_t::B});
graph->check_support(handle);
graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
cache.graph = graph;
cache.workspace_size = graph->get_workspace_size();

// 执行（shared_ptr-based variant pack，同 cbrp_fwd.cpp:161-165 模式）
ctx.ensure_workspace_grow(sk, cache.workspace_size);
std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
vp[cache.x_attr] = ctx.ptr_at(node.input_ids[0]);
vp[cache.y_attr] = ctx.ptr_at(node.output_ids[0]);
if (!cache.mask_is_virtual) {
    vp[cache.mask_attr] = ctx.ptr_at(node.output_ids[1]);
}
cache.graph->execute(handle, vp, ctx.workspace(sk));
```

- FWD 和 INF 共享同一个 FE graph cache。
- 使用 `StreamKind::COMP_2`，跨流同步用 `MultiStreamCaptureState` 的 event 机制。
- 使用预分配 GPU workspace（`ctx.ensure_workspace_grow` → `ctx.workspace`），不在算子内分配。

#### 2.6.2 BWD：cuDNN Legacy API

**参考实现**：`cbrp_bwd_no_dgrad.cpp` 中 `initialize_maxpool_descriptors()` + `execute_maxpool_bwd_legacy()`（已验证性能）。

技术路径：`cudnnPoolingBackward` + `CUDNN_POOLING_MAX_DETERMINISTIC` + `CUDNN_NOT_PROPAGATE_NAN`。不使用 mask（Legacy API 内部重算 max 位置）。

```cpp
// input binding:
//   input_ids[0] = dY (prev_grad_id)
//   input_ids[1] = Y  (pool_output)
//   input_ids[2] = mask (pool_mask, unused)
//   output_ids[0] = dX (= X, in-place)

void* dy = ctx.ptr_at(node.input_ids[0]);
void* y  = ctx.ptr_at(node.input_ids[1]);
const int8_t* unused_mask = static_cast<const int8_t*>(ctx.ptr_at(node.input_ids[2]));
(void)unused_mask;
void* dx = ctx.ptr_at(node.output_ids[0]);  // dx == x buffer (in-place)

// Per-shape cache 结构体（descriptor 生命周期管理）：
struct MaxPoolBwdCache {
    cudnnPoolingDescriptor_t pool_desc = nullptr;
    cudnnTensorDescriptor_t x_desc = nullptr, y_desc = nullptr;
    cudnnTensorDescriptor_t dx_desc = nullptr, dy_desc = nullptr;
    // 其他缓存字段...

    ~MaxPoolBwdCache() {
        if (pool_desc) cudnnDestroyPoolingDescriptor(pool_desc);
        if (x_desc)   cudnnDestroyTensorDescriptor(x_desc);
        if (y_desc)   cudnnDestroyTensorDescriptor(y_desc);
        if (dx_desc)  cudnnDestroyTensorDescriptor(dx_desc);
        if (dy_desc)  cudnnDestroyTensorDescriptor(dy_desc);
    }
    MaxPoolBwdCache(const MaxPoolBwdCache&) = delete;
    MaxPoolBwdCache& operator=(const MaxPoolBwdCache&) = delete;
};

// 创建 descriptor（须先 Create 再 Set；正方形核，h==w）
cudnnCreatePoolingDescriptor(&pool_desc);
cudnnSetPooling2dDescriptor(pool_desc,
    CUDNN_POOLING_MAX_DETERMINISTIC,
    CUDNN_NOT_PROPAGATE_NAN,
    params.kernel_h, params.kernel_h,
    params.pad_h, params.pad_h,
    params.stride_h, params.stride_h);

// 类似创建 x_desc, dx_desc = [N, IH, IW, C]（NHWC）；y_desc, dy_desc = [N, OH, OW, C]（NHWC）
cudnnCreateTensorDescriptor(&x_desc);
cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NHWC, dt, N, C, IH, IW);
// ... 同理创建 dx_desc, y_desc, dy_desc

const float alpha = 1.0f, beta = 0.0f;
cudnnPoolingBackward(handle, pool_desc, &alpha,
    y_desc,  y,
    dy_desc, dy,
    x_desc,  dx,   // x == dx (in-place, cuDNN 文档明确支持)
    &beta,
    dx_desc, dx);
```

**关键点**：
- `x == dx`（同一指针），cuDNN 允许 `cudnnPoolingBackward` 的 x 和 dx 重叠。
- `y` 和 `dy` 与 `dx` 不重叠（MemoryPlan 保证）。
- FP32: `CUDNN_DATA_FLOAT`，AMP: `CUDNN_DATA_HALF`。
- BWD **不需要** workspace（Legacy API 无额外内存需求）。
- 描述符可加入 per-shape cache 避免重复创建/销毁。

#### 2.6.3 CUDA Launcher 签名

```cpp
// FWD / INF / BWD 通用 launcher 模式（同 ReLU/GAP）
static void launch_maxpool_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state);
```

#### 2.6.4 内存管理

| 路径 | 所需内存 | 来源 |
|------|----------|------|
| FWD/INF GPU | cuDNN FE workspace | `ctx.ensure_workspace_grow(sk, ...)` + `ctx.workspace(sk)` |
| BWD GPU | 无 | Legacy API 不需要额外内存 |
| FWD/INF CPU | 无 | 朴素循环，寄存器/栈上变量 |
| BWD CPU | 无 | 利用 mask 直接定位，无需 workspace |

### 2.7 CPU 后端实现（FP32 only）

#### 2.7.1 FWD

```cpp
static void launch_maxpool_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    int8_t*    mask = static_cast<int8_t*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));

    auto params = std::get<PoolParams>(op_ctx->params.data);
    int N = op_ctx->input_shape.n, IH = op_ctx->input_shape.h;
    int IW = op_ctx->input_shape.w, C = op_ctx->input_shape.c;
    int OH = op_ctx->output_shape.h, OW = op_ctx->output_shape.w;
    // 注：当前框架仅支持正方形核，kernel_h == kernel_w == k 等
    int k = params.kernel_h, s = params.stride_h, p = params.pad_h;

    for (int n = 0; n < N; ++n)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow)
          for (int c = 0; c < C; ++c) {
            float max_val = -std::numeric_limits<float>::infinity();
            int max_kh = -1, max_kw = -1;
            for (int kh = 0; kh < k; ++kh)
              for (int kw = 0; kw < k; ++kw) {
                int ih = oh * s - p + kh;
                int iw = ow * s - p + kw;
                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                  float val = x[((n * IH + ih) * IW + iw) * C + c];
                  if (val > max_val) { max_val = val; max_kh = kh; max_kw = kw; }
                }
              }
            y[((n * OH + oh) * OW + ow) * C + c] = max_val;
            // 编码：max_kh * k + max_kw（k == kernel_w，正方形核下等价于标准行优先）
            mask[((n * OH + oh) * OW + ow) * C + c] =
                (max_kh >= 0) ? static_cast<int8_t>(max_kh * k + max_kw) : -1;
          }

#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}
```

#### 2.7.2 BWD

CPU BWD 利用 FWD 写入 mask 的局部偏移（`max_kh * k + max_kw`，正方形核下 `k == kernel_w`）直接定位 max 位置，不重算、不读取 X。无需 cpu_workspace。

```cpp
static void launch_maxpool_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const float* dy = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const int8_t* mask = static_cast<const int8_t*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    float* dx = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    auto params = std::get<PoolParams>(op_ctx->params.data);
    int N = op_ctx->output_shape.n, OH = op_ctx->output_shape.h;
    int OW = op_ctx->output_shape.w, C = op_ctx->output_shape.c;
    int IH = op_ctx->input_shape.h, IW = op_ctx->input_shape.w;
    // 注：正方形核，k == kernel_h == kernel_w，s/p 同理
    int k = params.kernel_h, s = params.stride_h, p = params.pad_h;

    // 清零 dX（因为 DETERMINISTIC 语义：只写第一个 max 位置）
    int64_t dx_elems = static_cast<int64_t>(N) * IH * IW * C;
    for (int64_t i = 0; i < dx_elems; ++i) dx[i] = 0.0f;

    for (int n = 0; n < N; ++n)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow)
          for (int c = 0; c < C; ++c) {
            int8_t m = mask[((n * OH + oh) * OW + ow) * C + c];
            if (m < 0) continue;
            int max_kh = m / k;   // 解码：m = max_kh * k + max_kw
            int max_kw = m % k;
            int ih = oh * s - p + max_kh;
            int iw = ow * s - p + max_kw;
            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
              dx[((n * IH + ih) * IW + iw) * C + c] +=
                  dy[((n * OH + oh) * OW + ow) * C + c];
            }
          }
}
```

**安全性分析**：BWD 流程为"先清零 dx → 后累加 dy"，全程不读取 X（仅依赖 mask）。由于 dX 覆盖 X，清零操作的同时销毁了 X 的原始值，这完全安全——后续计算不再需要 X。重叠窗口（stride < kernel）的多个输出位置映射到同一输入位置时，使用 `+=` 正确累加梯度。

#### 2.7.3 INF

INF 复用 FWD CPU launcher（mask 被写入但不消费，语义上 INF 不需要 mask 值，但写入不影响正确性）。

#### 2.7.4 AMP CPU

`launch_not_supported_cpu`（抛出 `TR_TYPE_ERROR`）。

### 2.8 流策略与 Warmup

```cpp
// op_stream_policy.cpp
case ComputeOp::MAXPOOL_FP32_FWD: case ComputeOp::MAXPOOL_AMP_FWD:
case ComputeOp::MAXPOOL_FP32_BWD: case ComputeOp::MAXPOOL_AMP_BWD:
case ComputeOp::MAXPOOL_FP32_INF: case ComputeOp::MAXPOOL_AMP_INF:
    return StreamKind::COMP_2;
```

Warmup：
- FWD/INF：需要（cuDNN Frontend graph build）
- BWD：不需要（Legacy API）

在 `require_warmup` 中注册（`op_registry.cpp`）：

```cpp
// 替换原有 MAXPOOL_FWD / MAXPOOL_BWD 的 warmup 条目
case ComputeOp::MAXPOOL_FP32_FWD: case ComputeOp::MAXPOOL_AMP_FWD:
case ComputeOp::MAXPOOL_FP32_INF: case ComputeOp::MAXPOOL_AMP_INF:
    return true;
// BWD 不需要 warmup
```

### 2.9 清理已有 CBRP（ConvBNReLUMaxPool）不完整实现

当前项目中存在一个与 MaxPool 紧耦合的四元融合算子 ConvBNReLUMaxPool（CBRP），其张量推断直接调用 `infer_maxpool_tensors`。MaxPool 从 3-tensor 改为 2-tensor 后，CBRP 的 19 张量布局会整体断裂。由于本次仅实现 MaxPool 算子本身、不实现相关融合算子，需要删除 CBRP 的不完整实现。

**策略选择**：**策略 A（保留前端 DSL，展开时去融合）**——`blueprint.h` 中保留 `NodeKind::CBRP`、`CBRPParam`、`conv_bn_relu_pool()` builder；`arch_plan_expand.cpp` 中无论 fuse 参数如何，都展开为独立的 Conv + Bn2d + ReLU + MaxPool 四个 ArchLayer。用户仍可在模型定义中写 `conv_bn_relu_pool(...)`，底层退化为独立算子执行。

#### 2.9.1 删除清单

**类型定义层（`op_kind.h` + `arch_plan.h`）**：

| 文件 | 删除内容 |
|------|----------|
| `op_kind.h` | 枚举：`CBRP_AMP_FWD`, `CBRP_AMP_BWD`, `CBRP_AMP_INF` |
| `op_kind.h` | 结构体：`CBRPParams` |
| `op_kind.h` | `OpParams` variant 中的 `CBRPParams` |
| `op_kind.h` | `OpParams` 构造函数 `OpParams(CBRPParams)` 和 accessor `cbrp()` |
| `arch_plan.h` | `LayerKind::ConvBNReLUMaxPool` |
| `arch_plan.h` | 结构体：`CBRPLayerParams` |
| `arch_plan.h` | `LayerParam` variant 中的 `CBRPLayerParams` |

**前端展开层（`arch_plan_expand.cpp` + `arch_plan_merge.cpp`）**：

| 文件 | 修改内容 |
|------|----------|
| `arch_plan_expand.cpp` | `NodeKind::CBRP` 的 `fuse=true` 分支：删除推入 `LayerKind::ConvBNReLUMaxPool` 的代码，改为与 `fuse=false` 一致——展开为独立的 Conv + Bn2d + ReLU + MaxPool 四个 ArchLayer |
| `arch_plan_merge.cpp` | 删除 `ConvBNReLUMaxPool` 的 merge 规则（`merge_conv_bn_relu_maxpool` 或类似函数） |

**描述符与编译层（`layer_descriptor_registry.cpp` + `compiler.cpp`）**：

| 文件 | 删除内容 |
|------|----------|
| `layer_descriptor_registry.cpp` | `infer_convbnrelump_tensors` 函数（含内部 `pool_d = infer_maxpool_tensors(...)` 调用） |
| `layer_descriptor_registry.cpp` | `build_convbnrelump_forward/backward/inference` |
| `layer_descriptor_registry.cpp` | `get_output_shape` 中 `ConvBNReLUMaxPool: return find(17)` |
| `layer_descriptor_registry.cpp` | `get_layer_descriptor` 中 `ConvBNReLUMaxPool: return cbrp_desc` |
| `compiler.cpp` | `get_grad_output_id` 中 `ConvBNReLUMaxPool: idx = 2/17` |
| `compiler.cpp` | `compile_layer` 中 `ConvBNReLUMaxPool` 的 compile 分支 |
| `compiler.cpp` | `convert_to_op_params` 中 `CBRPLayerParams → CBRPParams` 转换 |
| `compiler.cpp` | `is_fused_layer` 中 `case LayerKind::ConvBNReLUMaxPool:`（~line 43） |
| `compiler.cpp` | `is_weight_bearing` 中 `case LayerKind::ConvBNReLUMaxPool:`（~line 57） |
| `compiler.cpp` | `build_aux_graphs` 中 `has_bn` 判断的 `ConvBNReLUMaxPool`（~line 1560） |

**Shape/Normalize/Format/YAML 层**：

| 文件 | 删除内容 |
|------|----------|
| `arch_plan_shape.cpp` | `LayerKind::ConvBNReLUMaxPool` 的 shape 推断 case |
| `arch_plan_normalize.cpp` | `ConvBNReLUMaxPool` 的 `out_ch` 查询 case |
| `arch_plan_format.cpp` | `ConvBNReLUMaxPool` 的 `to_string` 和 format case |
| `arch_plan_yaml.cpp` | `ConvBNReLUMaxPool` 的 YAML 序列化/反序列化 |

**字符串与策略层（`op_kind.cpp` + `op_stream_policy.cpp` + `op_registry.cpp`）**：

| 文件 | 删除内容 |
|------|----------|
| `op_kind.cpp` | `compute_op_to_string` 中 `CBRP_AMP_FWD/BWD/INF` case |
| `op_kind.cpp` | `format_params` 中 `CBRPParams` 分支 |
| `op_stream_policy.cpp` | `CBRP_AMP_FWD/BWD/INF` 的流映射 |
| `op_registry.cpp` | `require_warmup` 中 `CBRP_AMP_FWD/BWD/INF` case |

#### 2.9.2 保留项

| 文件 | 保留内容 |
|------|----------|
| `blueprint.h` | `NodeKind::CBRP`、`CBRPParam` 结构体 |
| `blueprint.h` | `conv_bn_relu_pool()` 前端 builder 函数 |

---

## 三、Dropout 实现方案

### 3.1 前端全链路补齐

Dropout 当前在框架中完全缺失（`arch_plan_expand.cpp` 中 `NodeKind::Dropout` 直接 `break`），需要完整的前端实现。

#### 3.1.1 `op_kind.h` — 新增 `DropoutParams`

```cpp
struct DropoutParams {
    float p = 0.5f;  // 丢弃率 [0.0, 1.0)
    // 注：建议在 convert_to_op_params 或算子入口处添加 TR_CHECK(p >= 0.0f && p < 1.0f) 边界验证，
    //     防止 p=1.0 导致 scale = inf。
};
```

`OpParams` variant 加入 `DropoutParams`：

```cpp
struct OpParams {
    std::variant<
        std::monostate,
        ConvParams, PoolParams, FCParams, BNParams, LossParams,
        UpdateParams, EMAParams, AllReduceParams, AxpyParams,
        CastParams, FlattenParams, CBRParams,
        BottleneckParams, GapFCParams,
        DropoutParams   // <-- 新增
    > data = std::monostate{};

    explicit OpParams(DropoutParams p) : data(std::move(p)) {}
    const DropoutParams& dropout() const { return std::get<DropoutParams>(data); }
};
// 注：CBRPParams 已在 §2.9 中删除，最终 variant 不含 CBRPParams。
```

ComputeOp 枚举（已存在，无需修改）：

```cpp
DROPOUT_AMP_FWD, DROPOUT_AMP_BWD, DROPOUT_AMP_INF,
DROPOUT_FP32_FWD, DROPOUT_FP32_BWD, DROPOUT_FP32_INF,
```

`op_kind.cpp` 中 `format_params` 新增 Dropout 分支：

```cpp
// 在 switch(op) 结构中的 Dropout case：
case ComputeOp::DROPOUT_FP32_FWD: case ComputeOp::DROPOUT_FP32_BWD:
case ComputeOp::DROPOUT_FP32_INF: case ComputeOp::DROPOUT_AMP_FWD:
case ComputeOp::DROPOUT_AMP_BWD: case ComputeOp::DROPOUT_AMP_INF: {
    if (auto* dp = std::get_if<DropoutParams>(&p.data)) {
        oss << "p=" << dp->p;
    }
    break;
}
```

同时将 `MAXPOOL_FWD/BWD` 的 format_params 分支更新为 6 个新枚举：

```cpp
case ComputeOp::MAXPOOL_FP32_FWD: case ComputeOp::MAXPOOL_FP32_BWD:
case ComputeOp::MAXPOOL_AMP_FWD: case ComputeOp::MAXPOOL_AMP_BWD:
case ComputeOp::MAXPOOL_FP32_INF: case ComputeOp::MAXPOOL_AMP_INF: {
    if (auto* pp = std::get_if<PoolParams>(&p.data)) {
        oss << "kernel=" << pp->kernel_h << "x" << pp->kernel_w
            << ",stride=" << pp->stride_h << "x" << pp->stride_w
            << ",pad=" << pp->pad_h << "x" << pp->pad_w;
    }
    break;
}
```

#### 3.1.2 `arch_plan.h` — 新增 `LayerKind::Dropout`

```cpp
enum class LayerKind : uint16_t {
    // ... 现有 ...
    Dropout,   // <-- 新增
};

struct DropoutLayerParams {
    float p = 0.5f;
    bool operator==(const DropoutLayerParams& o) const { return p == o.p; }
};

using LayerParam = std::variant<
    // ... 现有 ...
    DropoutLayerParams,   // <-- 新增
    EmptyParams
>;
```

#### 3.1.3 `arch_plan_expand.cpp`

在 `expand_primitive_impl` 函数（`arch_plan_expand.cpp` line ~20-133）的 switch 中添加 `NodeKind::Dropout` case，替换当前的 `break;`（当前代码 line 90-92: `case NodeKind::Dropout: { break; }`）。`expand_tree` 中已有的 `NodeKind::Dropout` 调用 `expand_primitive_impl` 的逻辑无需修改。

```cpp
// 在 expand_primitive_impl 的 switch 中，将现有的 case NodeKind::Dropout: { break; }
// 替换为：
case NodeKind::Dropout: {
    auto& p = std::get<DropoutParam>(node.payload);
    out.push_back({LayerKind::Dropout, DropoutLayerParams{p.p},
                   "dropout", {}, {}, false, false, src_id});
    break;
}
```

#### 3.1.4 `arch_plan_shape.cpp`

```cpp
case LayerKind::Dropout:
    break;  // 形状不变
```

#### 3.1.5 `compiler.cpp` — 其他集成点

```cpp
// convert_to_op_params（在 ~line 566 的 return OpParams{} 之前插入）
if (std::holds_alternative<DropoutLayerParams>(lp)) {
    auto& p = std::get<DropoutLayerParams>(lp);
    return OpParams{DropoutParams{p.p}};
}

// get_output_shape（layer_descriptor_registry.cpp 中的形状查询函数）
case LayerKind::Dropout:  return find(0);

// get_grad_output_id
case LayerKind::Dropout:  idx = -1; break;  // in-place

// compile_layer no-op
case LayerKind::Dropout:
    break;
```

### 3.2 张量布局（2 张量）

| 索引 | 名称 | 形状 | Region | DType | FWD 读写 | BWD 读写 | INF 读写 |
|------|------|------|--------|-------|----------|----------|----------|
| 0 | `dropout_output` | `[N, 1, 1, C]` | F_FEATURE_FP32/FP16 | FP32/FP16 | 写 | 写(dX) | 写 |
| 1 | `dropout_mask` | `[N, 1, 1, C]` | S_MASK | INT8 | 写 | 读 | 不写 |

**约束**：Dropout 仅支持 2D 输入 `[N, 1, 1, C]`（全连接层后）。在 `infer_dropout_tensors` 中验证 `input.h() == 1 && input.w() == 1`。

### 3.3 LayerDescriptor

```cpp
std::vector<TensorDesc> infer_dropout_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    (void)params;
    // 注：当前方案限制 Dropout 仅支持 [N,1,1,C] 输入（项目级约束，非数学限制）。
    // 这是因为当前 ResNet 架构中 Dropout 仅用于 FC 层之后，
    // kernel 内部使用 N*C 遍历 + stride 映射物理地址的方式处理 AMP padding。
    // 如需支持 [N,H,W,C] 输入，kernel 需改为四维 N*H*W*C 遍历。
    TR_CHECK(input.h() == 1 && input.w() == 1, ShapeError,
             "Dropout only supports [N,1,1,C] input (FC layer only in current arch)");
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return {
        TensorDesc{"dropout_output", input, select_feature_region(ctx), feat_dt},
        TensorDesc{"dropout_mask",   input, Region::S_MASK,        DType::INT8}
    };
}
```

### 3.4 SubgraphPattern

```cpp
// FWD: X(跨层注入) → Y(idx0), mask(idx1)
SubgraphPattern build_dropout_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    bool amp = GlobalRegistry::instance().using_amp();
    n.op = amp ? ComputeOp::DROPOUT_AMP_FWD : ComputeOp::DROPOUT_FP32_FWD;
    n.output_indices = {0, 1};
    p.nodes.push_back(n);
    return p;
}

// BWD: dY(compiler注入) + mask(idx1) → dX(覆盖X)
SubgraphPattern build_dropout_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    bool amp = GlobalRegistry::instance().using_amp();
    n.op = amp ? ComputeOp::DROPOUT_AMP_BWD : ComputeOp::DROPOUT_FP32_BWD;
    n.input_indices  = {1};     // mask only（dY 由 compiler 注入）
    n.output_indices = {};      // 空：compiler 处理
    p.nodes.push_back(n);
    return p;
}

// INF: X(跨层注入) → Y(idx0), mask(idx1, 分配但不写)
SubgraphPattern build_dropout_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    bool amp = GlobalRegistry::instance().using_amp();
    n.op = amp ? ComputeOp::DROPOUT_AMP_INF : ComputeOp::DROPOUT_FP32_INF;
    n.output_indices = {0, 1};
    p.nodes.push_back(n);
    return p;
}
```

注册到 `get_layer_descriptor()`：

```cpp
static const LayerDescriptor dropout_desc = {
    infer_dropout_tensors, build_dropout_forward,
    build_dropout_backward, build_dropout_inference
};
// ...
case LayerKind::Dropout: return dropout_desc;
```

`get_output_shape`（`layer_descriptor_registry.cpp`）新增：

```cpp
case LayerKind::Dropout:  return find(0);  // 输出形状 = 输入形状
```

### 3.5 随机种子管理（Per-RANK 可复现）

#### 3.5.1 设计决策

| 组件 | 位置 | 说明 |
|------|------|------|
| `dropout_seed` | baseline INT32 `[1,1,1,2]` | 64-bit seed 拆为两个 INT32，per-rank 不同 |
| offset | 元素索引 `idx` | Kernel 中 `philox_uniform_float(seed, idx)` |

**选择理由**：
1. 无需 device atomic（避免 CUDA Graph 并发 replay 竞争）。
2. Per-RANK 种子独立：`rank_seed = SplitMix64(global_seed + rank)`。
3. 可复现：相同 seed + 相同输入 → 相同 mask。
4. 元素索引作为 Philox counter 偏移，天然 CUDA Graph 兼容。

**局限性**：同一 CUDA Graph 多次 replay 产生相同 mask。每 training step 只 forward 一次，不同 step 输入不同，确定性 dropout 不影响训练正确性（TensorFlow `tf.nn.dropout` 在 graph 模式下的行为）。

#### 3.5.2 Baseline 分配

**修改文件**：`include/renaissance/graph/memory_plan.h` + `src/graph/memory_plan.cpp`

```cpp
// memory_plan.h: BaselineIds 新增
struct BaselineIds {
    // ... 现有 ...
    int32_t dropout_seed = -1;   // INT32 [1,1,1,2]
};
```

```cpp
// memory_plan.cpp: alloc_baseline_dtensors
Shape seed_shape{1, 1, 1, 2};
auto seed_dt = alloc_impl(seed_shape, DType::INT32, Region::S_SCALAR_INT32);
set_init_config(seed_dt.id, InitConfig{0.0f, InitKind::NONE, FanMode::FAN_IN});  // 避免随机初始化
baseline_.dropout_seed = seed_dt.id;
```

> **注**：`memory_plan.cpp` 中 `alloc_baseline_dtensors` 有两个重载（5参数版 line 353 和 8参数版 line 412），**两个重载均需添加上述 dropout_seed 分配代码**。

#### 3.5.3 种子初始化（`init_all`）

**修改文件**：`src/task/task_base.cpp`（或 `init_all` 所在文件）

插入位置：在标准 DTensor 初始化循环之后、ZERO_GAMMA 策略之前。

```cpp
void TaskBase::init_all() {
    // --- 第 1 步：标准初始化全部 DTensor ---
    for (const auto& dtensor : active_memory_plan_->dtensors()) {
        init(dtensor);
    }

    // --- [插入点] 第 1.5 步：初始化 per-rank dropout seed ---
    // 为每个 rank 衍生独立 seed（从全局随机种子派生）
    uint64_t global_seed = GlobalRegistry::instance().seed();
    for (int rank = 0; rank < num_gpus_; ++rank) {
        // SplitMix64 衍生 per-rank seed
        uint64_t z = global_seed + static_cast<uint64_t>(rank) + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        uint64_t rank_seed = z ^ (z >> 31);

        int32_t seed_data[2] = {
            static_cast<int32_t>(rank_seed & 0xFFFFFFFFULL),
            static_cast<int32_t>(rank_seed >> 32)
        };

        Tensor host_seed(Shape{1, 1, 1, 2}, DType::INT32);
        memcpy(host_seed.data(), seed_data, sizeof(seed_data));
        transfer_to_rank(host_seed, active_memory_plan_->get_dtensor(baseline_.dropout_seed), rank);
    }

    // --- 第 2 步：ZERO_GAMMA 策略 ---
    if (initializer_.is_zero_gamma()) { ... }
}
```

#### 3.5.4 种子读取（Backend）

```cpp
// GPU 模式下 seed 存在 device 端，需 D2H 拷贝后读取
// （seed 共 8 字节，每次 launch 拷贝开销可忽略）
static uint64_t get_dropout_seed(const DeviceContext& ctx, int32_t seed_id) {
    int32_t seed_data[2];
    void* d_ptr = ctx.ptr_at(seed_id);
    cudaMemcpy(seed_data, d_ptr, sizeof(seed_data), cudaMemcpyDeviceToHost);
    return (static_cast<uint64_t>(seed_data[1]) << 32)
         | static_cast<uint32_t>(seed_data[0]);
}
```

### 3.6 CUDA 后端实现

**新建文件**：`src/backend/ops/dtensor/dropout_op.cpp` + `dropout_op.cu`

#### 3.6.1 FWD Kernel（Philox RNG）

```cuda
__global__ void dropout_fp32_fwd_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int8_t* __restrict__ mask,
    int N, int C, int64_t feat_n_stride, int64_t mask_n_stride,
    uint64_t seed, float p, float scale)
{
    int64_t total = static_cast<int64_t>(N) * C;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += gridDim.x * blockDim.x) {
        int n = static_cast<int>(idx / C);
        int c = static_cast<int>(idx % C);
        int64_t feat_idx = n * feat_n_stride + c;
        int64_t mask_idx = n * mask_n_stride + c;

        float r = tr::detail::philox_uniform_float(seed, idx);
        int8_t m = (r >= p) ? 1 : 0;
        mask[mask_idx] = m;
        y[feat_idx] = m ? (x[feat_idx] * scale) : 0.0f;
    }
}
```

**AMP 版本**：`const __half*` / `__half*`，内部转 float 计算后写回：
```cuda
float xf = __half2float(x[feat_idx]);
y[feat_idx] = m ? __float2half(xf * scale) : __float2half(0.0f);
```

#### 3.6.2 BWD Kernel

```cuda
__global__ void dropout_fp32_bwd_kernel(
    const float* __restrict__ dY,
    const int8_t* __restrict__ mask,
    float* __restrict__ dX,
    int N, int C, int64_t feat_n_stride, int64_t mask_n_stride,
    float scale)
{
    int64_t total = static_cast<int64_t>(N) * C;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += gridDim.x * blockDim.x) {
        int n = static_cast<int>(idx / C);
        int c = static_cast<int>(idx % C);
        int64_t feat_idx = n * feat_n_stride + c;
        int64_t mask_idx = n * mask_n_stride + c;
        dX[feat_idx] = mask[mask_idx] ? (dY[feat_idx] * scale) : 0.0f;
    }
}
```

#### 3.6.3 INF Kernel

```cuda
__global__ void dropout_fp32_inf_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const int8_t* __restrict__ unused_mask,
    int N, int C, int64_t feat_n_stride)
{
    (void)unused_mask;
    int64_t total = static_cast<int64_t>(N) * C;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += gridDim.x * blockDim.x) {
        int n = static_cast<int>(idx / C);
        int c = static_cast<int>(idx % C);
        int64_t feat_idx = n * feat_n_stride + c;
        y[feat_idx] = x[feat_idx];  // identity
    }
}
```

#### 3.6.4 Launch CUDA 函数

```cpp
// FP32 FWD
static void launch_dropout_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_mask = mp.get_dtensor(node.output_ids[1]);

    float* x = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    float* y = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    int8_t* mask = static_cast<int8_t*>(ctx.ptr_at(node.output_ids[1]));

    auto params = std::get<DropoutParams>(node.params.data);
    float p = params.p;
    float scale = 1.0f / (1.0f - p);

    int N = dt_x.n(), C = dt_x.c();
    int64_t feat_n_stride = dt_x.n_stride_cuda();
    int64_t mask_n_stride = dt_mask.n_stride_cuda();

    uint64_t seed = get_dropout_seed(ctx, mp.baseline().dropout_seed);

    StreamKind sk = StreamKind::COMP_2;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);

    // cross-stream sync ...
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    int64_t total = static_cast<int64_t>(N) * C;
    int block_size = 256;
    int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);

    dropout_fp32_fwd_kernel<<<grid_size, block_size, 0, s>>>(
        x, y, mask, N, C, feat_n_stride, mask_n_stride, seed, p, scale);

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// BWD: input_ids[0]=dY(compiler注入), input_ids[1]=mask, output_ids[0]=dX(=X)
// INF: output_ids[1] mask 参数存在但不写
```

#### 3.6.5 内存管理

| 路径 | 所需内存 | 来源 |
|------|----------|------|
| FWD GPU | 无 | Philox kernel 无 workspace |
| BWD GPU | 无 | 纯逐元素操作 |
| INF GPU | 无 | 纯 identity |
| CPU 全部 | 无 | 朴素循环，无临时内存 |

seed 存储在 baseline 张量中（预先分配），不在算子内分配。

### 3.7 CPU 后端实现（FP32 only）

```cpp
static void launch_dropout_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    int8_t*    mask = static_cast<int8_t*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));

    auto params = std::get<DropoutParams>(op_ctx->params.data);
    float p = params.p;
    float scale = 1.0f / (1.0f - p);

    int N = op_ctx->input_shape.n, C = op_ctx->input_shape.c;
    int64_t total = static_cast<int64_t>(N) * C;

    uint64_t seed = GlobalRegistry::instance().seed();  // CPU 单 rank，直接用 global seed

    for (int64_t idx = 0; idx < total; ++idx) {
        int n = static_cast<int>(idx / C);
        int c = static_cast<int>(idx % C);
        int64_t feat_idx = n * op_ctx->n_stride + c;
        int64_t mask_idx = n * op_ctx->n_stride + c;

        float r = tr::detail::philox_uniform_float(seed, idx);
        int8_t m = (r >= p) ? 1 : 0;
        mask[mask_idx] = m;
        y[feat_idx] = m ? (x[feat_idx] * scale) : 0.0f;
    }

#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

static void launch_dropout_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const float* dY = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const int8_t* mask = static_cast<const int8_t*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    float* dX = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    auto params = std::get<DropoutParams>(op_ctx->params.data);
    float scale = 1.0f / (1.0f - params.p);

    int N = op_ctx->input_shape.n, C = op_ctx->input_shape.c;
    int64_t total = static_cast<int64_t>(N) * C;

    for (int64_t idx = 0; idx < total; ++idx) {
        int n = static_cast<int>(idx / C);
        int c = static_cast<int>(idx % C);
        int64_t feat_idx = n * op_ctx->n_stride + c;
        int64_t mask_idx = n * op_ctx->n_stride + c;
        dX[feat_idx] = mask[mask_idx] ? (dY[feat_idx] * scale) : 0.0f;
    }
}

static void launch_dropout_fp32_inf_cpu(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    const int8_t* unused_mask = static_cast<const int8_t*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));
    (void)unused_mask;

    int N = op_ctx->input_shape.n, C = op_ctx->input_shape.c;
    int64_t total = static_cast<int64_t>(N) * C;

    for (int64_t idx = 0; idx < total; ++idx) {
        int n = static_cast<int>(idx / C);
        int c = static_cast<int>(idx % C);
        int64_t feat_idx = n * op_ctx->n_stride + c;
        y[feat_idx] = x[feat_idx];
    }
}

// AMP CPU: launch_not_supported_cpu
```

### 3.8 流策略与 Warmup

```cpp
// op_stream_policy.cpp
case ComputeOp::DROPOUT_FP32_FWD: case ComputeOp::DROPOUT_AMP_FWD:
case ComputeOp::DROPOUT_FP32_BWD: case ComputeOp::DROPOUT_AMP_BWD:
case ComputeOp::DROPOUT_FP32_INF: case ComputeOp::DROPOUT_AMP_INF:
    return StreamKind::COMP_2;
```

Warmup：Dropout 使用自定义 Philox kernel，但必须在 `require_warmup` 中注册（与 ReLU/Tanh 等自定义 kernel 算子一致），确保 CUDA Graph capture 前 kernel 已加载到 GPU。

```cpp
// op_registry.cpp: require_warmup 新增
case ComputeOp::DROPOUT_FP32_FWD: case ComputeOp::DROPOUT_AMP_FWD:
case ComputeOp::DROPOUT_FP32_BWD: case ComputeOp::DROPOUT_AMP_BWD:
case ComputeOp::DROPOUT_FP32_INF: case ComputeOp::DROPOUT_AMP_INF:
    return true;
```

---

## 四、算子注册

### 4.1 声明

**修改文件**：`include/renaissance/backend/op_registry.h`

```cpp
void register_op_maxpool();
void register_op_dropout();
```

### 4.2 调用

**修改文件**：`src/backend/op_registry.cpp`

```cpp
void register_default_ops() {
    // ... 现有注册 ...
    register_op_maxpool();   // <-- 新增
    register_op_dropout();   // <-- 新增
}
```

### 4.3 MaxPool 注册函数

```cpp
// maxpool_op.cpp
void register_op_maxpool() {
    auto reg_fp32 = [](ComputeOp op, auto cu, auto cpu) {
        auto& e = g_compute_op_table[static_cast<size_t>(op)];
        e.op = op;
#ifdef TR_USE_CUDA
        e.launch_cuda = cu;
#endif
#ifdef TR_USE_EIGEN
        e.launch_cpu = cpu;
#endif
    };
    auto reg_amp = [](ComputeOp op, auto cu) {
        auto& e = g_compute_op_table[static_cast<size_t>(op)];
        e.op = op;
#ifdef TR_USE_CUDA
        e.launch_cuda = cu;
#endif
        e.launch_cpu = launch_not_supported_cpu;
    };

    reg_fp32(ComputeOp::MAXPOOL_FP32_FWD, launch_maxpool_fp32_fwd_cuda, launch_maxpool_fp32_fwd_cpu);
    reg_fp32(ComputeOp::MAXPOOL_FP32_BWD, launch_maxpool_fp32_bwd_cuda, launch_maxpool_fp32_bwd_cpu);
    reg_fp32(ComputeOp::MAXPOOL_FP32_INF, launch_maxpool_fp32_inf_cuda, launch_maxpool_fp32_inf_cpu);
    reg_amp(ComputeOp::MAXPOOL_AMP_FWD, launch_maxpool_amp_fwd_cuda);
    reg_amp(ComputeOp::MAXPOOL_AMP_BWD, launch_maxpool_amp_bwd_cuda);
    reg_amp(ComputeOp::MAXPOOL_AMP_INF, launch_maxpool_amp_inf_cuda);
}
```

### 4.4 Dropout 注册函数

```cpp
// dropout_op.cpp
void register_op_dropout() {
    auto reg_fp32 = [](ComputeOp op, auto cu, auto cpu) {
        auto& e = g_compute_op_table[static_cast<size_t>(op)];
        e.op = op;
#ifdef TR_USE_CUDA
        e.launch_cuda = cu;
#endif
#ifdef TR_USE_EIGEN
        e.launch_cpu = cpu;
#endif
    };
    auto reg_amp = [](ComputeOp op, auto cu) {
        auto& e = g_compute_op_table[static_cast<size_t>(op)];
        e.op = op;
#ifdef TR_USE_CUDA
        e.launch_cuda = cu;
#endif
        e.launch_cpu = launch_not_supported_cpu;
    };

    reg_fp32(ComputeOp::DROPOUT_FP32_FWD, launch_dropout_fp32_fwd_cuda, launch_dropout_fp32_fwd_cpu);
    reg_fp32(ComputeOp::DROPOUT_FP32_BWD, launch_dropout_fp32_bwd_cuda, launch_dropout_fp32_bwd_cpu);
    reg_fp32(ComputeOp::DROPOUT_FP32_INF, launch_dropout_fp32_inf_cuda, launch_dropout_fp32_inf_cpu);
    reg_amp(ComputeOp::DROPOUT_AMP_FWD, launch_dropout_amp_fwd_cuda);
    reg_amp(ComputeOp::DROPOUT_AMP_BWD, launch_dropout_amp_bwd_cuda);
    reg_amp(ComputeOp::DROPOUT_AMP_INF, launch_dropout_amp_inf_cuda);
}
```

---

## 五、CPU/GPU 接口对齐总结

### 5.1 张量访问一致性

| 算子 | 模式 | GPU input_ids | GPU output_ids | CPU input_ids | CPU output_ids |
|------|------|---------------|----------------|---------------|----------------|
| MaxPool | FWD | [X] | [Y, mask] | [X] | [Y, mask] |
| MaxPool | BWD | [dY, Y, mask] | [dX(=X)] | [dY, Y, mask] | [dX(=X)] |
| MaxPool | INF | [X] | [Y, mask] | [X] | [Y, mask] |
| Dropout | FWD | [X] | [Y, mask] | [X] | [Y, mask] |
| Dropout | BWD | [dY, mask] | [dX(=X)] | [dY, mask] | [dX(=X)] |
| Dropout | INF | [X] | [Y, mask] | [X] | [Y, mask] |

### 5.2 参数语义一致性

| 参数 | GPU 来源 | CPU 来源 |
|------|----------|----------|
| PoolParams | `node.params.data` | `op_ctx->params.data` |
| DropoutParams | `node.params.data` | `op_ctx->params.data` |
| Dropout seed | `ctx.ptr_at(baseline.dropout_seed)` | `GlobalRegistry::instance().seed()` |
| 形状/stride | `mp.get_dtensor(...)` | `op_ctx->input_shape/output_shape/n_stride` |

---

## 六、AMP Padding 处理

- **MaxPool**：cuDNN FE/Legacy 内部自行处理 padding。NHWC stride 通过 `set_stride` 正确传递。无需手动干预。
- **Dropout `[N,1,1,C]`**：kernel 遍历逻辑元素数 `N*C`（不是 `padded_elems()`），使用 `n_stride_cuda()` 计算物理地址：
  - 特征图索引 = `n * feat_n_stride + c`（`feat_n_stride = padded_c`）
  - mask 索引 = `n * mask_n_stride + c`（mask 紧凑无 padding）
  - 不会越界写入 S_MASK buffer。

---

## 七、完整文件修改清单

### 7.1 新增文件

| 文件 | 内容 |
|------|------|
| `src/backend/ops/dtensor/maxpool_op.cpp` | MaxPool CPU launchers + CUDA launchers + `register_op_maxpool()` |
| `src/backend/ops/dtensor/maxpool_op.cu` | MaxPool cuDNN FE FWD/INF graph cache + Legacy BWD descriptors cache |
| `src/backend/ops/dtensor/dropout_op.cpp` | Dropout CPU launchers + CUDA launchers + `register_op_dropout()` |
| `src/backend/ops/dtensor/dropout_op.cu` | Dropout Philox kernel（FWD/BWD/INF × FP32/AMP） |

### 7.2 修改文件

| # | 文件 | 修改内容 |
|---|------|----------|
| 1 | `include/renaissance/graph/op_kind.h` | ① 删除 `MAXPOOL_FWD/BWD`，新增 6 个精度分离枚举；② 新增 `DropoutParams`；③ `OpParams` variant 加入 `DropoutParams`；④ 删除 `CBRP_AMP_FWD/BWD/INF` 枚举和 `CBRPParams` |
| 2 | `src/graph/op_kind.cpp` | ① 新增 6 个 MaxPool 枚举 `to_string`；② 更新 `format_params` 中 `MAXPOOL_FWD/BWD` 为 6 个新枚举；③ 新增 `DropoutParams` 的 `format_params` 分支；④ 删除 CBRP 相关 `to_string`/`format_params` 分支 |
| 3 | `include/renaissance/graph/arch_plan.h` | ① `LayerKind` 添加 `Dropout`；② 新增 `DropoutLayerParams`；③ `LayerParam` 加入；④ 删除 `LayerKind::ConvBNReLUMaxPool` 和 `CBRPLayerParams` |
| 4 | `src/graph/arch_plan_expand.cpp` | `NodeKind::Dropout` 不再 `break`，展开为 `LayerKind::Dropout` |
| 5 | `src/graph/arch_plan_shape.cpp` | `LayerKind::Dropout` → `break`（形状不变） |
| 6 | `src/graph/layer_descriptor_registry.cpp` | ① MaxPool 2-tensor 修正（删除 `pool_grad_slot`，mask INT8）；② 新增 Dropout 完整描述符；③ `get_layer_descriptor` 注册 Dropout；④ `get_output_shape` 新增 `LayerKind::Dropout` case |
| 7 | `src/graph/compiler.cpp` | ① `convert_to_op_params` 加入 Dropout；② `get_output_shape` 加入 Dropout；③ `get_grad_output_id` MaxPool 改 `-1` + Dropout 加 `-1`；④ BWD in-place 分支加入 MaxPool/Dropout BWD；⑤ `prev_grad_id` fallback 加入 MaxPool/Dropout；⑥ `compile_layer` break 加入 Dropout |
| 8 | `include/renaissance/graph/memory_plan.h` | `BaselineIds` 新增 `int32_t dropout_seed = -1;` |
| 9 | `src/graph/memory_plan.cpp` | `alloc_baseline_dtensors` 分配 `dropout_seed`：`Shape{1,1,1,2}, INT32, S_SCALAR_INT32` |
| 10 | `src/backend/op_stream_policy.cpp` | ① 新增 6 个 MaxPool → `COMP_2`；② 新增 6 个 Dropout → `COMP_2` |
| 11 | `src/backend/op_registry.cpp` | ① `register_default_ops` 调用 `register_op_maxpool/dropout`；② `require_warmup`：删除旧 `MAXPOOL_FWD/BWD` case，MaxPool FWD/INF 需要 warmup（4 个新枚举），Dropout 全部 6 个枚举需要 warmup（自定义 kernel 预加载） |
| 12 | `include/renaissance/backend/op_registry.h` | 声明 `register_op_maxpool()`、`register_op_dropout()` |
| 13 | `src/task/task_base.cpp` | `init_all()` 中用 SplitMix64 为每个 rank 衍生 seed，`transfer_to_rank` 到 `dropout_seed` |
| 14 | `src/backend/CMakeLists.txt` | 新增 `maxpool_op.cpp/.cu`、`dropout_op.cpp/.cu` |
| 15 | `src/graph/arch_plan_expand.cpp` | CBRP：删除 `fuse=true` 推入 `ConvBNReLUMaxPool` 分支，统一展开为 Conv+Bn2d+ReLU+MaxPool |
| 16 | `src/graph/arch_plan_merge.cpp` | 删除 `ConvBNReLUMaxPool` 的 merge 规则 |
| 17 | `src/graph/arch_plan_shape.cpp` | 删除 `ConvBNReLUMaxPool` 的 shape 推断 case |
| 18 | `src/graph/arch_plan_normalize.cpp` | 删除 `ConvBNReLUMaxPool` 的 `out_ch` 查询 case |
| 19 | `src/graph/arch_plan_format.cpp` | 删除 `ConvBNReLUMaxPool` 的 format case |
| 20 | `src/graph/arch_plan_yaml.cpp` | 删除 `ConvBNReLUMaxPool` 的 YAML 序列化/反序列化 |
| 21 | `src/graph/layer_descriptor_registry.cpp` | 删除 `infer_convbnrelump_tensors`、`build_convbnrelump_*`、`get_output_shape`/`get_layer_descriptor` 中 CBRP 相关代码 |
| 22 | `src/graph/compiler.cpp` | 删除 `get_grad_output_id`、`compile_layer`、`convert_to_op_params` 等中 CBRP 相关代码 |
| 23 | `src/backend/op_stream_policy.cpp` | 删除 CBRP_AMP_* 的流映射 |
| 24 | `src/backend/op_registry.cpp` | 删除 `require_warmup` 中 CBRP_AMP_* case |
| 25 | `include/renaissance/graph/blueprint.h` | 保留 `NodeKind::CBRP`、`CBRPParam`、`conv_bn_relu_pool()`（不改动） |

---

## 八、现有代码已知问题汇总

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| 1 | `layer_descriptor_registry.cpp:384` | `pool_index` dtype 为 `INT32` | 改为 `INT8`，改名 `pool_mask` |
| 2 | `layer_descriptor_registry.cpp:385` | `pool_grad_slot` 存在 | 删除 |
| 3 | `layer_descriptor_registry.cpp` | `infer_maxpool_tensors` 返回 3 张量 | 改为 2 张量 |
| 4 | `layer_descriptor_registry.cpp` | `build_maxpool_backward` 的 `output_indices = {2}` | 改为 `{}` |
| 5 | `arch_plan_expand.cpp:90` | `NodeKind::Dropout` 直接 `break` | 展开为 `LayerKind::Dropout` |
| 6 | `op_kind.h:213-214` | `MAXPOOL_FWD/BWD` 未区分 FP32/AMP | 拆分为 6 个枚举 |
| 7 | `compiler.cpp:1047` | `get_grad_output_id(MaxPool)` 返回 `idx=2` | 改为 `idx=-1` |
| 8 | `compiler.cpp` | BWD in-place 分支不含 MaxPool/Dropout | 添加 |
| 9 | `compiler.cpp` | `prev_grad_id` fallback 不含 MaxPool/Dropout | 添加 |
| 10 | `op_registry.cpp` | 未调用 `register_op_maxpool/dropout` | 添加 |
| 11 | `op_stream_policy.cpp` | 无 Dropout 流映射 | 添加 `COMP_2` |
| 12 | `memory_plan.h` | 无 `dropout_seed` 字段 | 添加 |
| 13 | `op_kind.cpp` | `format_params` 中 `MAXPOOL_FWD/BWD` 未拆分 | 改为 6 个新枚举（FP32_FWD/BWD/INF, AMP_FWD/BWD/INF） |
| 14 | `layer_descriptor_registry.cpp` | `build_maxpool_inference` 使用 `MAXPOOL_FWD` 枚举且 `output_indices = {0}`（不含 mask） | 改为 `MAXPOOL_FP32_INF/AMP_INF`，`output_indices = {0, 1}` |
| 15 | `op_kind.h, arch_plan.h 等` | CBRP（ConvBNReLUMaxPool）不完整的融合算子实现 | 删除并改为独立展开（见 §2.9） |

---

## 九、实施顺序

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | CBRP 清理 | 删除 ConvBNReLUMaxPool 融合算子，`NodeKind::CBRP` 展开为独立四层（见 §2.9） |
| **P0** | 类型定义修改 | `op_kind.h`（MaxPool 枚举拆分 + CBRP 删除 + DropoutParams）、`arch_plan.h`（CBRP 删除 + Dropout LayerKind/Params） |
| **P0** | LayerDescriptor 修正 | MaxPool → 2-tensor + Dropout 新增完整描述符 |
| **P0** | ArchPlan 展开 | `NodeKind::Dropout` 不再 break |
| **P0** | Compiler 集成 | `get_grad_output_id`、BWD in-place、`prev_grad_id` fallback、`convert_to_op_params` |
| **P0** | MaxPool 后端 | `maxpool_op.cpp/.cu`：cuDNN FE FWD/INF + Legacy BWD + CPU |
| **P0** | Dropout 后端 | `dropout_op.cpp/.cu`：Philox kernel FWD/BWD/INF + CPU |
| **P1** | 流策略 + 注册 | `op_stream_policy.cpp`、`op_registry.cpp/.h` |
| **P1** | Baseline 种子 | `memory_plan.h/cpp` 分配 `dropout_seed`；`init_all` 初始化 per-rank seed |
| **P1** | CMake | 编译列表更新 |
| **P2** | 测试 | MaxPool 数值正确性、Dropout 随机可复现性、多 RANK 差异 |

**建议顺序**：先清理 CBRP（消除不完整融合算子对 MaxPool 的依赖），再实现 MaxPool（修改已有基础设施，风险可控），最后实现 Dropout（全新实现）。

---

## 十、验证计划

1. **MaxPool FWD**：与 PyTorch `F.max_pool2d(kernel_size=k, stride=s, padding=p)` 对比，误差 < 1e-5。
2. **MaxPool BWD**：与 PyTorch autograd 对比，误差 < 1e-5。验证 dX 覆盖 X 后 X buffer 内容正确。
3. **MaxPool INF**：输出与 FWD 一致。
4. **Dropout mask 统计**：`p=0.5`，`N=10000, C=1024`，mask=1 比例 ≈ 0.5（±0.01）。
5. **Dropout FWD/BWD**：固定 seed，验证 `Y = X * mask * scale`，`dX = dY * mask * scale`。
6. **Dropout INF**：严格验证 `Y == X`（逐元素相等）。
7. **多 RANK**：不同 RANK 产生不同 mask（相同输入、相同 global seed）。
8. **CPU/GPU 对齐**：同一算子的 CPU 和 GPU 路径输出一致（允许 float 误差 < 1e-5）。
9. **dX 覆盖 X**：验证 BWD 后 X buffer 内容为正确的 dX。

---

## 十一、风险与注意事项

### 11.1 MaxPool

1. **cuDNN virtual INT8 tensor**：部分 cuDNN 版本/硬件不支持 virtual INT8 tensor，需 try-catch fallback 为 real tensor（参照 `cbrp_fwd.cpp:789-803`）。
2. **Legacy API in-place**：`cudnnPoolingBackward` 允许 x 和 dx 为同一指针，需实测确认。
3. **FE graph cache key**：包含 `handle_bits` 区分不同 cudnnHandle（因 FE graph 绑定 handle）。
4. **AMP padding**：cuDNN 内部通过 `set_stride` 处理，无需手动干预。
5. **CBRP 清理**：须先删除 ConvBNReLUMaxPool 融合算子，再做 MaxPool 的 3→2 tensor 修改，否则 CBRP 的 `infer_convbnrelump_tensors` 调用修改后的 `infer_maxpool_tensors` 会导致张量数量不匹配。

### 11.2 Dropout

1. **Philox 实现**：需确认 `tr::detail::philox_uniform_float(seed, idx)` 已实现并可用。
2. **CUDA Graph 确定性**：同一 graph 多次 replay 产生相同 mask。每个 batch 只 forward 一次，不影响训练正确性。
3. **AMP padding**：kernel 遍历 `N*C` 逻辑元素，通过 stride 计算物理地址，不触碰 padding。
4. **Dropout 仅支持 2D**：输入必须是 `[N,1,1,C]`，在 `infer_dropout_tensors` 中 assert。
5. **Seed 初始化时机**：`dropout_seed` 的 baseline 分配先于 `init_all` 执行。确保 `alloc_baseline_dtensors` 在 `init_all` 之前调用。