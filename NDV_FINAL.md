# 卷积六大算子最终实现方案

> 版本: FINAL 1.3（修订版）  
> 日期: 2026-06-03  
> 来源: NDV1.md + NDV2.md + NDV3.md 综合分析 + 全量代码审查 + N_REV.md / N_REV2.md / N_REV3.md 评审修订  
> 修订记录: 采纳小伙伴C与小伙伴K的评审意见，补充 compiler.cpp 剩余两处修改（get_grad_output_id + grad_id 追踪）  
> 范围: CONV_FP32_FWD / CONV_FP32_BWD / CONV_FP32_INF / CONV_AMP_FWD / CONV_AMP_BWD / CONV_AMP_INF

---

## 一、方案对比与决策

三个方案（NDV1/NDV2/NDV3）在核心设计上高度一致，差异集中在以下 5 个决策点。本方案逐一给出最终裁决及理由。

### 决策 1: bn_stats 形状

| 方案 | 形状 | 裁决 |
|------|------|:---:|
| NDV1/NDV2 | `{1, 1, 1, K}` | 不采用 |
| NDV3 | `{1, 1, 1, 2*K}` | **采用** |

**理由**: MemoryPlan 按 `shape.elem_count() * sizeof(dtype)` 分配内存。形状 `{1,1,1,K}` 仅分配 `K*sizeof(float)` 字节，不足以容纳 `sum`(K个float) + `sq_sum`(K个float) 共 `2*K*sizeof(float)` 字节。形状 `{1,1,1,2*K}` 确保分配足够。

### 决策 2: bn_stats 是输入还是输出

| 方案 | 做法 | 裁决 |
|------|------|:---:|
| NDV1/NDV2 | bn_stats 仅作为 input_indices 成员 | **采用** |
| NDV3 | bn_stats 同时作为 FWD 的 output_indices 成员 | 不采用 |

**理由**: bn_stats 本质是 T_TEMP_FP32 区的预分配张量，AMP FWD 通过 VariantPack 指针写入，其他 5 个算子不写入。放入 input_indices 即可保证 Compiler 将其纳入 GraphNode，算子内通过 ptr_at 获取指针。放入 output_indices 反而引入不必要的复杂性（需在 Compiler 处理多输出）。

### 决策 3: BWD 的 input_indices 是否包含 output(Y)

| 方案 | 做法 | 裁决 |
|------|------|:---:|
| NDV1/NDV2 | `{weight, output, grad_slot}` 含 output | 不采用 |
| NDV3 | `{weight, grad_slot}` 不含 output | **采用** |

**理由**: cuDNN `conv_wgrad(dY, X)` 和 `conv_dgrad(dY, W)` 均不需要 Y。BWD 的 input_indices 包含 output(Y) 是冗余的，且会占用 Compiler 注入的 dY 之后的位置，导致 input_ids 语义错位。NDV3 的修正正确。

### 决策 4: Graph 缓存结构

| 方案 | 结构 | 裁决 |
|------|------|:---:|
| NDV1 | `uid_to_dtensor_id` (int64_t → int64_t) | 不采用 |
| NDV2 | `uid_to_tensor_id` (vector of pair) | 不采用 |
| NDV3 | `tensor_to_id` (shared_ptr\<Tensor_attributes\> → int64_t) | **采用** |

**理由**: NDV3 模式与 `fc_op.cpp` 的 `FcAmpFwdCache` 完全一致，已在生产代码中验证。`shared_ptr<Tensor_attributes>` 作为 key 可以直接从 graph builder 返回的 tensor 对象获取，无需额外维护 uid 映射。

### 决策 5: Graph 构建函数签名

| 方案 | 签名 | 裁决 |
|------|------|:---:|
| NDV1 | 旧签名（vector\<shared_ptr\> + vector\<Shape\> + vector\<DTensor\>） | 不采用 |
| NDV2/NDV3 | 直接传入 DTensor 引用 | **采用** |

**理由**: 旧签名间接传递 DTensor，导致 builder 内部需要从 vectors 中按索引取数据，容易出错。新签名直接传入相关 DTensor，类型安全且清晰。

### 决策 6: bn_stats 加入 infer_conv_tensors 的方式 ★ 修订

| 方案 | 做法 | 裁决 |
|------|------|:---:|
| NDV_FINAL 1.0 | 直接修改 `infer_conv_tensors` 返回 7 张量 | 不采用 |
| **修订方案** | 新建 `infer_conv_tensors_with_bn_stats`，保持 `infer_conv_tensors` 不变 | **采用** |

**理由**: `infer_conv_tensors` 被融合层（CBR/Bottleneck/BasicBlock/InvResidual）内部调用。若直接改为 7 张量，所有融合层的硬编码索引全部错位，影响 20+ 个 build 函数。新建独立函数、纯 Conv 层注册改用新函数，融合层完全不受影响。

---

## 二、参数约束

BluePrint/ArchPlan 层已保证，算子内不检查。

| 参数 | 允许值 |
|------|--------|
| kernel_h / kernel_w | 1, 3, 5, 7（正方形） |
| stride_h / stride_w | 1, 2 |
| dilation | 固定 1 |
| groups | 固定 1（普通卷积） |
| bias | **不支持** |
| padding | 无限制 |

---

## 三、核心铁律

### 3.1 内存管理（最高优先级）

**算子内严禁 `cudaMalloc` / `cudaFree`。** 所有临时 workspace 通过 `ctx.ensure_workspace_grow(StreamKind, size)` 复用 DeviceContext 的 per-stream workspace。`ctx.workspace(StreamKind)` 获取指针。

这也是 warmup 阶段的要求：`warmup_single_cudnn_op` 中现有的 `cudaMalloc`/`cudaFree` 必须改为 `ensure_workspace_grow`。

### 3.2 权重布局 KRSC

权重物理布局 `{K, R, S, C}`，C 最内层连续。cuDNN FE 中 `set_dim({K, C, R, S})`，`set_stride` 使用 DTensor 真实值：

```
dt_w.n_stride_cuda() = R*S*C   (K 步幅)
dt_w.c_stride_cuda() = 1       (C 步幅，NHWC)
dt_w.h_stride_cuda() = S*C     (R 步幅)
dt_w.w_stride_cuda() = C       (S 步幅)
```

### 3.3 NHWC 布局

特征图物理布局 NHWC。cuDNN FE `set_dim({N, C, H, W})`，`set_stride({N_stride, C_stride, H_stride, W_stride})`，`C_stride = 1` 表示 NHWC。

**铁律: 所有 cuDNN FE Tensor 的 stride 必须使用 DTensor 真实 stride (`dt.n_stride_cuda()` 系列)，禁止手写 `align_up`。**

### 3.4 dX 覆盖 X（In-Place BWD）

BWD 梯度输出 dX 复用 X 的同一 DTensor buffer（grad_slot，索引 2）。

**必须先算 dW（wgrad），再算 dX（dgrad），避免 X 在 wgrad 消费前被覆盖。**

| 阶段 | StreamKind | 同步 |
|------|-----------|------|
| wgrad | COMP_1 | 完成后 `cudaEventRecord(s_dw)` |
| dgrad | COMP_3 | 启动前 `cudaStreamWaitEvent(s_dx, dW_event)` |
| 代表流 | COMP_3 | 与 FC BWD 一致 |

### 3.5 张量对齐

同一层输入输出张量的数量、种类、形状在 CPU/GPU/FP32/AMP 下完全一致。bn_stats 在所有 6 个算子中均存在于 input_indices，但仅 `CONV_AMP_FWD` 真正写入。

---

## 四、张量布局（7 张量）

### 4.1 完整张量清单

| 索引 | 名称 | 形状 | Region | DType | 说明 |
|------|------|------|--------|-------|------|
| 0 | `conv_weight` | `{K, R, S, C}` | W_FIRST_CONV / W_DEEP_CONV | FP32 | 主权重 |
| 1 | `conv_output` | `{N, OH, OW, K}` | F_FEATURE_FP32 / F_FEATURE_FP16 | varies | 输出特征图 |
| 2 | `conv_grad_slot` | `{N, H, W, C}` | F_GRAD_SLOT_FP32 / F_GRAD_SLOT_FP16 | varies | 梯度槽（dX in-place） |
| 3 | `conv_weight_grad` | `{K, R, S, C}` | G_FIRST_CONV / G_DEEP_CONV | FP32 | FP32 权重梯度 |
| 4 | `conv_amp_w_fp16` | `{K, R, S, C}` | A_FIRST_CONV / A_DEEP_CONV | FP16 | AMP 工作权重 |
| 5 | `conv_amp_g_fp16` | `{K, R, S, C}` | G_FIRST_CONV_FP16 / G_DEEP_CONV_FP16 | FP16 | AMP 权重梯度（BWD 直接产 FP16） |
| **6** | **`conv_bn_stats`** | **`{1, 1, 1, 2*K}`** | **T_TEMP_FP32** | **FP32** | **BN 统计量（新增）** |

### 4.2 bn_stats 规格

- **形状**: `{1, 1, 1, 2 * out_channels}`，即 `{1, 1, 1, 2*K}`
- **区域**: `T_TEMP_FP32`（`cuda_alignment = 1`，不使用 padding）
- **内存布局**: 前半段 `[0, K)` 存 `sum`，后半段 `[K, 2*K)` 存 `sq_sum`
- **写入**: 仅 `CONV_AMP_FWD` 通过 GenStats 写入；其余 5 个算子不写入、不 memset
- **GenStats 绑定**: `sum` → `bn_stats_ptr`; `sq_sum` → `bn_stats_ptr + K * sizeof(float)`

### 4.3 Layer Descriptor 中的修改（避免融合层索引偏移）

**P0 风险**: `infer_conv_tensors` 被 `infer_convbnrelu_tensors` / `infer_convbn_tensors` 等融合层内部调用。若直接将其从 6 张量改为 7 张量（加入 bn_stats），会导致所有融合层（CBR/Bottleneck/BasicBlock/InvResidual）的硬编码索引全部错位。例如 `build_bottleneck_proj_forward` 的 20 个 `input_indices` 全部需要重新计算。

**解决方案**: 保持 `infer_conv_tensors` 返回 6 张量不变，新建 `infer_conv_tensors_with_bn_stats` 函数返回 7 张量（6 张原始 + bn_stats）。纯 Conv 层的 `LayerDescriptor` 注册改用新函数，融合层完全不受影响。

```cpp
// ===== 保持 infer_conv_tensors 不变（仍返回 6 张量）=====
// 融合层内部继续调用此函数，索引不受影响

// ===== 新增 infer_conv_tensors_with_bn_stats（返回 7 张量）=====
std::vector<TensorDesc> infer_conv_tensors_with_bn_stats(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    auto descs = infer_conv_tensors(input, params, ctx);  // 0-5
    const auto& cp = std::get<ConvParams>(params.data);
    // 6: bn_stats → T_TEMP_FP32, 形状 {1,1,1,2*K}
    descs.push_back(TensorDesc{"conv_bn_stats",
        Shape{1, 1, 1, cp.out_channels * 2},
        Region::T_TEMP_FP32, DType::FP32});
    return descs;
}

// ===== LayerDescriptor 注册改用新函数 =====
// get_layer_descriptor() 中:
static const LayerDescriptor conv_desc = {
    infer_conv_tensors_with_bn_stats,   // ← 改用 7 张量版本
    build_conv_forward,
    build_conv_backward,
    build_conv_inference
};

// ===== build_conv_forward =====
if (descs.size() < 7) return p;  // ★ 6→7（新增 bn_stats）
bool amp = GlobalRegistry::instance().using_amp();
SubgraphPattern::Node n;
n.op = amp ? ComputeOp::CONV_AMP_FWD : ComputeOp::CONV_FP32_FWD;
n.input_indices  = amp ? std::vector<size_t>{4, 6} : std::vector<size_t>{0, 6};
n.output_indices = {1};

// ===== build_conv_backward =====
if (descs.size() < 7) return p;  // ★ 6→7（新增 bn_stats）
// ★★ 关键修正 (N_REV2 Issue C-A): grad_slot(2) 位于 F_GRAD_SLOT Region，
//   而 X 位于 F_FEATURE Region，两者是不同的内存池，grad_slot 不能替代 X。
//   因此 input_indices 仅含 weight，X 由 Compiler 通过 layer_input_ids[l] 追加。
//   输出 dX 也由 Compiler 注入为 output_ids[0]（in-place 到 X），
//   故 output_indices 仅含 dW（weight_grad）。
n.op = amp ? ComputeOp::CONV_AMP_BWD : ComputeOp::CONV_FP32_BWD;
n.input_indices  = amp ? std::vector<size_t>{4} : std::vector<size_t>{0};
//   [weight]；dY 由 Compiler 注入到 input_ids[0]，X 由 Compiler 追加到 input_ids[2]
n.output_indices = amp ? std::vector<size_t>{5} : std::vector<size_t>{3};
//   [dW(weight_grad)]；dX 由 Compiler 注入为 output_ids[0]（in-place 到 X）

// ===== build_conv_inference =====
if (descs.size() < 7) return p;  // ★ 6→7（新增 bn_stats）
n.op = amp ? ComputeOp::CONV_AMP_INF : ComputeOp::CONV_FP32_INF;
n.input_indices  = amp ? std::vector<size_t>{4, 6} : std::vector<size_t>{0, 6};
n.output_indices = {1};
```

### 4.4 各算子 GraphNode 最终 I/O

Compiler 构建后，每个算子的实际 `input_ids` / `output_ids`：

| 算子 | node.input_ids | node.output_ids | Stream |
|------|---------------|-----------------|--------|
| CONV_FP32_FWD | `[0]=X, [1]=W(0), [2]=bn_stats(6)` | `[0]=Y(1)` | COMP_1 |
| CONV_FP32_INF | `[0]=X, [1]=W(0), [2]=bn_stats(6)` | `[0]=Y(1)` | COMP_1 |
| CONV_FP32_BWD | `[0]=dY, [1]=W(0), [2]=X(layer_input)` | `[0]=dX(layer_input), [1]=dW(3)` | dW:COMP_1, dX:COMP_3 |
| CONV_AMP_FWD | `[0]=X, [1]=W_fp16(4), [2]=bn_stats(6)` | `[0]=Y(1)` | COMP_1 |
| CONV_AMP_INF | `[0]=X, [1]=W_fp16(4), [2]=bn_stats(6)` | `[0]=Y(1)` | COMP_1 |
| CONV_AMP_BWD | `[0]=dY, [1]=W_fp16(4), [2]=X(layer_input)` | `[0]=dX(layer_input), [1]=dW_fp16(5)` | dW:COMP_1, dX:COMP_3 |

---

## 五、CUDA 实现

### 5.1 通用辅助函数

```cpp
// NHWC stride 映射（特征图）
inline std::vector<int64_t> make_nhwc_stride(const DTensor& dt) {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
}

// KRSC stride 映射（权重）
// dim={K, C, R, S}，stride 通过 DTensor 真实值隐式表达 KRSC 物理布局
// ★ 实现与 make_nhwc_stride 完全一致，因为 DTensor 的 stride 方法已基于 padded_c
//   隐式表达了 KRSC 布局。两个函数等价但语义不同，保留以区分特征图和权重。
inline std::vector<int64_t> make_krsc_stride(const DTensor& dt) {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
}
```

### 5.2 Per-Shape Graph Cache

参考 `fc_op.cpp` 的 `FcAmpFwdCache`，完全复用其模式：

```cpp
struct ConvGraphCacheKey {
    uint64_t handle_bits;
    int32_t  N, H, W, C, K, R, S;
    int32_t  pad_h, pad_w;
    int32_t  stride_h, stride_w;
    bool     is_amp;
    ComputeOp op;   // 区分 FWD/INF/BWD-wgrad/BWD-dgrad

    bool operator==(const ConvGraphCacheKey& o) const = default;
};

struct ConvGraphCacheKeyHasher {
    size_t operator()(const ConvGraphCacheKey& k) const {
        size_t h = std::hash<uint64_t>()(k.handle_bits);
        h ^= std::hash<int32_t>()(k.N)  << 1;
        h ^= std::hash<int32_t>()(k.H)  << 2;
        h ^= std::hash<int32_t>()(k.W)  << 3;
        h ^= std::hash<int32_t>()(k.C)  << 4;
        h ^= std::hash<int32_t>()(k.K)  << 5;
        h ^= std::hash<int32_t>()(k.R)  << 6;
        h ^= std::hash<int32_t>()(k.S)  << 7;
        h ^= std::hash<int32_t>()(k.pad_h) << 8;
        h ^= std::hash<int32_t>()(k.pad_w) << 9;
        h ^= std::hash<int32_t>()(k.stride_h) << 10;
        h ^= std::hash<int32_t>()(k.stride_w) << 11;
        h ^= std::hash<bool>()(k.is_amp) << 12;
        h ^= std::hash<int>()(static_cast<int>(k.op)) << 13;
        return h;
    }
};

struct ConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size;
};

// 静态缓存：按算子类型分独立 map
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHasher>
    s_conv_fwd_cache;     // FWD + INF 共用
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHasher>
    s_conv_wgrad_cache;   // BWD wgrad
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHasher>
    s_conv_dgrad_cache;   // BWD dgrad
```

**Cache 查找/构建模板**:

```cpp
template<typename CacheMap>
static ConvGraphCache& get_or_build_cache(
    CacheMap& cache_map,
    const ConvGraphCacheKey& key,
    std::function<ConvGraphCache()> builder)
{
    auto it = cache_map.find(key);
    if (it != cache_map.end()) return it->second;
    auto [inserted_it, _] = cache_map.emplace(key, builder());
    return inserted_it->second;
}
```

### 5.3 Variant Pack 填充

```cpp
// 从 cache 的 tensor_to_id 映射填充 Variant Pack
std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
for (const auto& [tensor_attr, dt_id] : cache.tensor_to_id) {
    vp[tensor_attr] = ctx.ptr_at(static_cast<int>(dt_id));
}
```

### 5.4 CONV_FP32_FWD / CONV_FP32_INF

**Graph 构建**（`conv_op_impl.cpp`）:

```cpp
ConvGraphCache build_conv_fp32_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_cudnn_graph(DType::FP32);

    auto X = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("X")
        .set_dim(to_fe_dim(dt_x.shape))
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::FLOAT));

    auto W = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("W")
        .set_dim(to_fe_dim(dt_w.shape))
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::FLOAT));

    auto opts = fe::graph::Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto Y = graph->conv_fprop(X, W, opts);
    Y->set_output(true)
      .set_name("Y")
      .set_dim(to_fe_dim(dt_y.shape))
      .set_stride(make_nhwc_stride(dt_y))
      .set_data_type(fe::DataType_t::FLOAT);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X] = dt_x.id;
    cache.tensor_to_id[W] = dt_w.id;
    cache.tensor_to_id[Y] = dt_y.id;
    return cache;
}
```

**Launch 执行**:

```cpp
static void launch_conv_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto& cp = node.params.conv();
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);

    StreamKind sk = StreamKind::COMP_1;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
    int si = state.get_or_register(s);

    // 等待前序
    if (state.output_stream_idx >= 0) {
        cudaStreamWaitEvent(s,
            state.streams[state.output_stream_idx].last_done_event, 0);
    }

    // Cache
    ConvGraphCacheKey key{
        reinterpret_cast<uint64_t>(h),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(), dt_w.c(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        false, ComputeOp::CONV_FP32_FWD
    };
    auto& cache = get_or_build_cache(s_conv_fwd_cache, key, [&]() {
        return build_conv_fp32_fwd_graph(dt_x, dt_w, dt_y, cp, h);
    });

    // Workspace
    ctx.ensure_workspace_grow(sk, cache.workspace_size);

    // Variant Pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
    for (const auto& [ta, tid] : cache.tensor_to_id) {
        vp[ta] = ctx.ptr_at(static_cast<int>(tid));
    }

    // Execute
    TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ctx.workspace(sk)),
                      "CONV_FP32_FWD execute");

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

**INF 与 FWD 在 cuDNN 层面完全等价**（都是 `conv_fprop`），共用 graph 构建函数和 cache。Cache key 中 `op` 字段区分，或 INF 直接查 FWD 的 cache（key.op 设为 FWD）。

### 5.5 CONV_FP32_BWD（双图 + 跨流同步）

**必须拆分为两个独立 Graph**: cuDNN FE Graph 的 `execute()` 绑定到单个 `cudnnHandle_t`（即单个 stream），无法将 wgrad 和 dgrad 分配到不同 stream。

**WGrad Graph（COMP_1）**:

```cpp
ConvGraphCache build_conv_fp32_wgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_x, const DTensor& dt_dw,
    const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_cudnn_graph(DType::FP32);

    auto dY = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("dY")
        .set_dim(to_fe_dim(dt_dy.shape))
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::FLOAT));

    auto X = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("X")
        .set_dim(to_fe_dim(dt_x.shape))
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::FLOAT));

    auto opts = fe::graph::Conv_wgrad_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto dW = graph->conv_wgrad(dY, X, opts);
    dW->set_output(true)
       .set_name("dW")
       .set_dim(to_fe_dim(dt_dw.shape))
       .set_stride(make_krsc_stride(dt_dw))
       .set_data_type(fe::DataType_t::FLOAT);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[X]  = dt_x.id;
    cache.tensor_to_id[dW] = dt_dw.id;
    return cache;
}
```

**DGrad Graph（COMP_3）**:

```cpp
ConvGraphCache build_conv_fp32_dgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_w, const DTensor& dt_dx,
    const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_cudnn_graph(DType::FP32);

    auto dY = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("dY")
        .set_dim(to_fe_dim(dt_dy.shape))
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::FLOAT));

    auto W = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("W")
        .set_dim(to_fe_dim(dt_w.shape))
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::FLOAT));

    auto opts = fe::graph::Conv_dgrad_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto dX = graph->conv_dgrad(dY, W, opts);
    dX->set_output(true)
       .set_name("dX")
       .set_dim(to_fe_dim(dt_dx.shape))
       .set_stride(make_nhwc_stride(dt_dx))  // 必须与 X 完全一致
       .set_data_type(fe::DataType_t::FLOAT);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[W]  = dt_w.id;
    cache.tensor_to_id[dX] = dt_dx.id;
    return cache;
}
```

**Launch 同步**:

```cpp
static void launch_conv_fp32_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto& cp = node.params.conv();
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_x  = mp.get_dtensor(node.input_ids[2]);
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);
    const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]);

    cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    cudaStream_t s_dx = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
    cudnnHandle_t h_dw = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_1));
    cudnnHandle_t h_dx = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_3));
    int i_dw = state.get_or_register(s_dw);
    int i_dx = state.get_or_register(s_dx);

    // 等待前序（两流都 wait）
    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
        cudaStreamWaitEvent(s_dx, state.streams[out_idx].last_done_event, 0);
    }

    // === COMP_1: WGrad ===
    ConvGraphCacheKey key_w{
        reinterpret_cast<uint64_t>(h_dw),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(), dt_w.c(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        false, ComputeOp::CONV_FP32_BWD
    };
    auto& cache_w = get_or_build_cache(s_conv_wgrad_cache, key_w, [&]() {
        return build_conv_fp32_wgrad_graph(dt_dy, dt_x, dt_dw, cp, h_dw);
    });
    ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_w;
    for (const auto& [ta, tid] : cache_w.tensor_to_id) {
        vp_w[ta] = ctx.ptr_at(static_cast<int>(tid));
    }
    TR_CUDNN_FE_CHECK(cache_w.graph->execute(h_dw, vp_w, ctx.workspace(StreamKind::COMP_1)),
                      "CONV_FP32_BWD wgrad execute");
    cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    state.streams[i_dw].has_pending_work = true;

    // === COMP_3: DGrad（等待 WGrad 完成，保护 X） ===
    cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);

    ConvGraphCacheKey key_x{
        reinterpret_cast<uint64_t>(h_dx),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(), dt_w.c(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        false, ComputeOp::CONV_FP32_BWD
    };
    auto& cache_x = get_or_build_cache(s_conv_dgrad_cache, key_x, [&]() {
        return build_conv_fp32_dgrad_graph(dt_dy, dt_w, dt_dx, cp, h_dx);
    });
    ctx.ensure_workspace_grow(StreamKind::COMP_3, cache_x.workspace_size);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_x;
    for (const auto& [ta, tid] : cache_x.tensor_to_id) {
        vp_x[ta] = ctx.ptr_at(static_cast<int>(tid));
    }
    TR_CUDNN_FE_CHECK(cache_x.graph->execute(h_dx, vp_x, ctx.workspace(StreamKind::COMP_3)),
                      "CONV_FP32_BWD dgrad execute");
    cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
    state.streams[i_dx].has_pending_work = true;

    state.output_stream_idx = i_dx;
}
```

### 5.6 CONV_AMP_FWD（Conv + GenStats）

**Graph 构建**（遵循 `cbr_fwd_fp16.cpp` 已验证模式：conv_out 直接 `set_output(true)`，无需 `set_is_virtual`）:

```cpp
ConvGraphCache build_conv_amp_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const DTensor& dt_bn, const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_cudnn_graph(DType::FP16);

    int64_t PC_x = dt_x.padded_c();  // ★ AMP: padded_c()
    int64_t PC_w = dt_w.padded_c();
    int64_t K = dt_y.c();

    auto X = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), PC_x, dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), PC_w, dt_w.h(), dt_w.w()})
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = fe::graph::Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto conv_out = graph->conv_fprop(X, W, opts);

    // ★ 直接 set_output(true)，与 cbr_fwd_fp16.cpp 一致。
    //    conv_out 设为 output 后仍可作为 genstats 的输入，graph 不会断裂。
    conv_out->set_output(true)
             .set_name("Y")
             .set_dim(to_fe_dim(dt_y.shape))
             .set_stride(make_nhwc_stride(dt_y))
             .set_data_type(fe::DataType_t::HALF);

    // GenStats
    auto genstats_opts = fe::graph::Genstats_attributes()
        .set_name("genstats")
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto genstats_outputs = graph->genstats(conv_out, genstats_opts);
    auto sum    = genstats_outputs[0];
    auto sq_sum = genstats_outputs[1];

    sum->set_output(true)
        .set_name("sum")
        .set_dim({1, K, 1, 1})
        .set_stride({K, 1, K, K})
        .set_data_type(fe::DataType_t::FLOAT);

    sq_sum->set_output(true)
          .set_name("sq_sum")
          .set_dim({1, K, 1, 1})
          .set_stride({K, 1, K, K})
          .set_data_type(fe::DataType_t::FLOAT);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X]        = dt_x.id;
    cache.tensor_to_id[W]        = dt_w.id;
    cache.tensor_to_id[conv_out] = dt_y.id;
    cache.tensor_to_id[sum]      = dt_bn.id;    // bn_stats 前半段
    cache.tensor_to_id[sq_sum]   = dt_bn.id;    // 同 id，通过偏移区分
    cache.bn_stats_offset        = K;            // ★ sq_sum 偏移 K 个 float 元素
    return cache;
}
```

**ConvGraphCache 扩展**（新增 `bn_stats_offset` 字段）:

```cpp
struct ConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size;
    int64_t bn_stats_offset = 0;  // sq_sum 相对于 sum 的偏移（元素数），仅 AMP FWD 使用
};
```

**Variant Pack 中 bn_stats 映射**（通过 name 判断 sq_sum，与 cbr_fwd_fp16.cpp 现有模式一致；若需更稳健，可改用 `cache.sq_sum_tensor` 指针比较）:

```cpp
// 在 Launch 中：
float* bn_stats_ptr = static_cast<float*>(ctx.ptr_at(node.input_ids[2]));
for (const auto& [ta, tid] : cache.tensor_to_id) {
    void* ptr = ctx.ptr_at(static_cast<int>(tid));
    // ★ 通过 Tensor_attributes 对象指针判断是否为 sq_sum，而非 get_name()
    if (tid == dt_bn.id && cache.bn_stats_offset > 0 &&
        ta->get_name() == "sq_sum") {
        ptr = static_cast<float*>(ptr) + cache.bn_stats_offset;
    }
    vp[ta] = ptr;
}
```

**单流 COMP_1**。

### 5.7 CONV_AMP_BWD（双图 + 跨流同步 + FP16 dW）

**WGrad Graph（COMP_1）**:

```cpp
ConvGraphCache build_conv_amp_wgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_x, const DTensor& dt_dw,
    const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_cudnn_graph(DType::FP16);

    auto dY = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("dY")
        .set_dim({dt_dy.n(), dt_dy.padded_c(), dt_dy.h(), dt_dy.w()})
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::HALF));

    auto X = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), dt_x.padded_c(), dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = fe::graph::Conv_wgrad_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto dW = graph->conv_wgrad(dY, X, opts);
    dW->set_output(true)
       .set_name("dW")
       .set_dim(to_fe_dim(dt_dw.shape))
       .set_stride(make_krsc_stride(dt_dw))
       .set_data_type(fe::DataType_t::HALF);  // ★ AMP BWD 输出 FP16 dW

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[X]  = dt_x.id;
    cache.tensor_to_id[dW] = dt_dw.id;
    return cache;
}
```

**DGrad Graph（COMP_3）**:

```cpp
ConvGraphCache build_conv_amp_dgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_w, const DTensor& dt_dx,
    const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_cudnn_graph(DType::FP16);

    auto dY = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("dY")
        .set_dim({dt_dy.n(), dt_dy.padded_c(), dt_dy.h(), dt_dy.w()})
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), dt_w.padded_c(), dt_w.h(), dt_w.w()})
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = fe::graph::Conv_dgrad_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto dX = graph->conv_dgrad(dY, W, opts);
    dX->set_output(true)
       .set_name("dX")
       .set_dim({dt_dx.n(), dt_dx.padded_c(), dt_dx.h(), dt_dx.w()})
       .set_stride(make_nhwc_stride(dt_dx))
       .set_data_type(fe::DataType_t::HALF);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[W]  = dt_w.id;
    cache.tensor_to_id[dX] = dt_dx.id;
    return cache;
}
```

**流同步代码与 FP32 BWD 完全一致**（见 5.5 节 Launch 模板）。

**关键点**:
- dW 输出 FP16 到 `G_*_CONV_FP16`（索引 5），后续由 Compiler 插入 `RANGE_CAST_FP16_TO_FP32` 批量转换到 G_*_CONV（索引 3）
- `set_dim` 使用 `dt.padded_c()`，`set_stride` 使用 DTensor 真实值
- dX 的 stride 必须与 X 完全一致（包括 padded_c）

### 5.8 CONV_AMP_INF

纯 `conv_fprop`，**无 GenStats**:

```cpp
ConvGraphCache build_conv_amp_inf_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_cudnn_graph(DType::FP16);

    auto X = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), dt_x.padded_c(), dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), dt_w.padded_c(), dt_w.h(), dt_w.w()})
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = fe::graph::Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto Y = graph->conv_fprop(X, W, opts);
    Y->set_output(true)
      .set_name("Y")
      .set_dim(to_fe_dim(dt_y.shape))
      .set_stride(make_nhwc_stride(dt_y))
      .set_data_type(fe::DataType_t::HALF);

    finalize_cudnn_graph(graph.get(), handle);

    ConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X] = dt_x.id;
    cache.tensor_to_id[W] = dt_w.id;
    cache.tensor_to_id[Y] = dt_y.id;
    return cache;
}
```

单流 COMP_1。bn_stats 在 input_ids 中但 Graph 不引用。

---

## 六、CPU 实现

### 6.1 FWD / INF

- 共用 `launch_conv_fwd_cpu()`
- 优先 XNNPACK: `xnn_define_convolution_2d`，NHWC layout，filter 格式 KRSC
- Fallback: naive 嵌套循环
- INF 与 FWD 计算逻辑完全相同

### 6.2 BWD

- `launch_conv_bwd_cpu()`: Naive 6 重循环
- 先 memset dX=0、dW=0，再累加
- 单线程顺序执行（先 dW 后 dX，天然满足 dX 覆盖 X 约束）
- 注意 NHWC 内存访问模式: `X[n*H*W*C + (ih*W+iw)*C + c]`

### 6.3 AMP CPU

不支持，统一抛出异常:

```cpp
static void launch_conv_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("Conv AMP operators do not support CPU execution");
}
```

---

## 七、枚举与注册表扩展

### 7.1 `include/renaissance/graph/op_kind.h`

```cpp
enum class ComputeOp : uint16_t {
    // ... 现有 ...
    CONV_FP32_FWD,
    CONV_FP32_BWD,
    CONV_FP32_INF,      // ★ 新增
    CONV_AMP_FWD,
    CONV_AMP_BWD,
    CONV_AMP_INF,       // ★ 新增
    // ...
};
```

### 7.2 `src/graph/op_kind.cpp`

`compute_op_to_string()` 和 `format_params()` 需覆盖全部 6 个算子变体:

```cpp
// compute_op_to_string():
case ComputeOp::CONV_FP32_INF: return "CONV_FP32_INF";
case ComputeOp::CONV_AMP_INF:  return "CONV_AMP_INF";

// format_params() 扩展现有 case:
case ComputeOp::CONV_FP32_FWD:
case ComputeOp::CONV_FP32_BWD:
case ComputeOp::CONV_FP32_INF:      // ★ 新增
case ComputeOp::CONV_AMP_FWD:
case ComputeOp::CONV_AMP_BWD:
case ComputeOp::CONV_AMP_INF: {     // ★ 新增
    if (auto* cp = std::get_if<ConvParams>(&p.data)) {
        oss << "out_ch=" << cp->out_channels
            << ",kernel=" << cp->kernel_h << "x" << cp->kernel_w
            << ",stride=" << cp->stride_h << "x" << cp->stride_w
            << ",pad=" << cp->pad_h << "x" << cp->pad_w;
    }
    break;
}
```

### 7.3 `include/renaissance/core/types.h`

```cpp
enum class ConvSearchMode : uint8_t {
    HEURISTIC_B = 0,   // 默认
    EXHAUSTIVE_C = 1   // 仅 AMP Conv，仅 A100/RTX5090
};
```

### 7.4 `include/renaissance/core/global_registry.h` + `src/core/global_registry.cpp`

遵循 `set_optimizer_kind` / `set_conv_init_kind` 的固定模式：setter 返回 `void`，带幂等检查；getter 返回枚举值。

```cpp
// global_registry.h
class GlobalRegistry {
public:
    void set_conv_search_mode(ConvSearchMode mode);
    ConvSearchMode conv_search_mode() const;
private:
    // ★ -1 = unset sentinel，与 fixed_optimizer_kind_ / fixed_conv_init_kind_ 等统一
    std::atomic<int> fixed_conv_search_mode_{-1};
};

// global_registry.cpp
void GlobalRegistry::set_conv_search_mode(ConvSearchMode mode) {
    int value = static_cast<int>(mode);
    int old = fixed_conv_search_mode_.load(std::memory_order_relaxed);

    if (old == -1) {
        fixed_conv_search_mode_.store(value, std::memory_order_release);
        return;
    }
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify conv_search_mode after initialization.");
    }
    if (old == value) return;  // 幂等允许
    TR_VALUE_ERROR("Cannot modify conv_search_mode after first assignment.");
}

ConvSearchMode GlobalRegistry::conv_search_mode() const {
    int v = fixed_conv_search_mode_.load(std::memory_order_relaxed);
    if (v == -1) return ConvSearchMode::HEURISTIC_B;  // ★ 未设置时返回默认值
    return static_cast<ConvSearchMode>(v);
}
```

### 7.5 `src/backend/op_stream_policy.cpp`

```cpp
case ComputeOp::CONV_FP32_INF:
case ComputeOp::CONV_AMP_INF:
    return StreamKind::COMP_1;
```

### 7.6 `src/backend/op_registry.cpp`

**`require_warmup()`**: 新增 `CONV_FP32_INF`、`CONV_AMP_INF`。

**★ 删除旧的 extern 声明**: 现有 `op_registry.cpp` line 126-132 的 `build_conv_fwd_graph` (旧签名，返回 `shared_ptr<Graph>`) 和 `build_conv_bwd_graph` 前向声明不再匹配新签名（返回 `ConvGraphCache`），必须删除。

**`warmup_single_cudnn_op()`**: **保留 Conv 专用路径，修复 workspace 管理。**

- 移除 `cudaMalloc`/`cudaFree`，改用 `ctx.ensure_workspace_grow(StreamKind::COMP_1, ws)`（FWD/INF）
- 不统一走通用 `launch_cuda`，原因：BWD 的 dgrad 在 COMP_3 执行，通用 warmup 路径仅同步 COMP_1，可能导致 warmup 在 dgrad 完成前返回
- 推荐做法：
  - FWD/INF: 专用路径，构建 graph → `ensure_workspace_grow` → execute，同步 COMP_1
  - BWD: 专用路径，分别构建 wgrad(COMP_1) 和 dgrad(COMP_3) graph，双流执行 + 同步所有涉及流
- 或退而求其次：BWD 走通用 `launch_cuda`，但修改 warmup_single_cudnn_op 遍历 `state.streams` 中所有 `has_pending_work` 的流分别同步

### 7.7 算子注册

```cpp
void register_op_conv() {
    // CONV_FP32_FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_FWD)];
        e.op = ComputeOp::CONV_FP32_FWD;
        e.launch_cpu = launch_conv_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_conv_fp32_fwd_cuda;
#endif
    }
    // CONV_FP32_BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_BWD)];
        e.op = ComputeOp::CONV_FP32_BWD;
        e.launch_cpu = launch_conv_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_conv_fp32_bwd_cuda;
#endif
    }
    // CONV_FP32_INF（复用 FWD 的 CPU + GPU launch）
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_INF)];
        e.op = ComputeOp::CONV_FP32_INF;
        e.launch_cpu = launch_conv_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_conv_fp32_inf_cuda;  // 内部调用 fwd 实现
#endif
    }
#ifdef TR_USE_CUDA
    // CONV_AMP_FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_AMP_FWD)];
        e.op = ComputeOp::CONV_AMP_FWD;
        e.launch_cpu = launch_conv_amp_cpu_not_supported;
        e.launch_cuda = launch_conv_amp_fwd_cuda;
    }
    // CONV_AMP_BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_AMP_BWD)];
        e.op = ComputeOp::CONV_AMP_BWD;
        e.launch_cpu = launch_conv_amp_cpu_not_supported;
        e.launch_cuda = launch_conv_amp_bwd_cuda;
    }
    // CONV_AMP_INF
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_AMP_INF)];
        e.op = ComputeOp::CONV_AMP_INF;
        e.launch_cpu = launch_conv_amp_cpu_not_supported;
        e.launch_cuda = launch_conv_amp_inf_cuda;
    }
#endif
}
```

---

## 八、文件变更清单

| # | 文件 | 变更 | 说明 |
|---|------|------|------|
| 1 | `include/renaissance/graph/op_kind.h` | 修改 | 新增 `CONV_FP32_INF`、`CONV_AMP_INF` |
| 2 | `src/graph/op_kind.cpp` | 修改 | `compute_op_to_string()` + `format_params()` 补充 |
| 3 | `include/renaissance/core/types.h` | 修改 | 新增 `ConvSearchMode` 枚举 |
| 4 | `include/renaissance/core/global_registry.h` | 修改 | 新增 `fixed_conv_search_mode_` + setter/getter |
| 5 | `src/core/global_registry.cpp` | 修改 | 实现 setter/getter（幂等检查） |
| 6 | `src/backend/op_stream_policy.cpp` | 修改 | 新增 INF → COMP_1 |
| 7 | `src/backend/op_registry.cpp` | 修改 | `require_warmup()` 加 INF；warmup 修复 workspace |
| 8 | `src/graph/layer_descriptor_registry.cpp` | 修改 | 新增 `infer_conv_tensors_with_bn_stats`；修正 `build_conv_*` 的 I/O（`descs.size()` 检查改为 `< 7`） |
| 9 | `src/graph/compiler.cpp` | 修改 | ★ 三处修改：(1) 追加 X + dX in-place；(2) `get_grad_output_id` Conv 改为 -1；(3) `grad_id<0` 分支加入 Conv |
| 10 | `src/backend/ops/dtensor/conv_op_impl.cpp` | **重写** | 6 个 cuDNN FE Graph 构建函数 |
| 11 | `src/backend/ops/dtensor/conv_op.cpp` | **重写** | 6 个 launch + CPU + 注册 |

> **conv_op_legacy.cpp**: 根目录下的该文件包含旧的 AMP graph builder 和 CPU XNNPACK 实现，不在 CMakeLists.txt 中（不编译）。保留作为 CPU XNNPACK 实现的参考，不需要修改或加入构建。新的 CPU 实现直接写入 `conv_op.cpp`。

> **compiler.cpp**: ★★ 必须进行三处修改（仿 FC BWD 模式），缺一不可：
>
> **修改 1** — 在 BWD GraphNode 构建阶段（紧接着 FC BWD 处理分支后），追加 X 为输入并设 dX in-place：
> ```cpp
> if (gn.compute_op == ComputeOp::CONV_FP32_BWD || gn.compute_op == ComputeOp::CONV_AMP_BWD) {
>     auto it = layer_input_ids.find(l);
>     if (it != layer_input_ids.end() && it->second >= 0) {
>         gn.input_ids.push_back(it->second);                      // 追加 X
>         gn.output_ids.insert(gn.output_ids.begin(), it->second); // dX in-place to X (output_ids[0])
>     }
> }
> ```
>
> **修改 2** — `get_grad_output_id` lambda（约 line 1040），将 Conv 从固定索引改为 in-place 模式：
> ```cpp
> // 改前: case LayerKind::Conv: idx = 2; break;  // grad_slot
> // 改后:
> case LayerKind::Conv: idx = -1; break;  // ★ dX in-place to X, 同 FC/MaxPool/ReLU 模式
> ```
>
> **修改 3** — `grad_id < 0` 追踪分支（约 line 1378），将 `LayerKind::Conv` 加入列表：
> ```cpp
> if (grad_id < 0 && (layer.kind == LayerKind::FC ||
>     layer.kind == LayerKind::Conv ||   // ★ 新增
>     layer.kind == LayerKind::FCBNReLU || layer.kind == LayerKind::GapFC ||
>     // ... 其余不变 ...
> ```
>
> 修改 2 和 3 共同确保 `prev_grad_id` 正确追踪到 X 张量，前一层 BWD 的 dX 才能写到正确的内存地址。遗漏任一处都会导致反向传播链断裂。
>
> 处理后 `input_ids = {dY, W, X}`，`output_ids = {dX, dW}`，与 Launch 函数期望一致。

---

## 九、实施顺序

```
Phase 1: 基础设施（无风险）
  1. op_kind.h/cpp: 加 INF 枚举 + 字符串化
  2. types.h: 加 ConvSearchMode
  3. global_registry.h/cpp: 加 conv_search_mode
  4. op_stream_policy.cpp: INF 流映射
  5. op_registry.cpp: require_warmup 扩展

Phase 2: Layer Descriptor（低风险）
  6. layer_descriptor_registry.cpp:
     - 新增 infer_conv_tensors_with_bn_stats（调用 infer_conv_tensors + bn_stats）
     - 修正 build_conv_* 的 ComputeOp（动态选 FP32/AMP）和 I/O（BWD 去掉 output）
     - 注册 conv_desc 改用 infer_conv_tensors_with_bn_stats

Phase 3: FP32 重写（中风险）
  7. conv_op_impl.cpp: 删除旧 build_conv_fwd_graph / build_conv_bwd_graph
  8. conv_op_impl.cpp: 新增 build_conv_fp32_fwd_graph（返回 ConvGraphCache）
  9. conv_op_impl.cpp: 新增 build_conv_fp32_wgrad_graph + build_conv_fp32_dgrad_graph
  10. conv_op.cpp: 重写 launch_conv_fp32_fwd_cuda（cache + workspace 复用）
  11. conv_op.cpp: 新增 launch_conv_fp32_inf_cuda（复用 FWD）
  12. conv_op.cpp: 重写 launch_conv_fp32_bwd_cuda（双图双流）
  13. conv_op.cpp: 更新 register_op_conv() 注册 FP32 三个算子

Phase 4: AMP 实现（高风险）
  14. conv_op_impl.cpp: 新增 build_conv_amp_fwd_graph（Conv + GenStats）
  15. conv_op_impl.cpp: 新增 build_conv_amp_inf_graph（纯 conv_fprop）
  16. conv_op_impl.cpp: 新增 build_conv_amp_wgrad_graph + build_conv_amp_dgrad_graph
  17. conv_op.cpp: 新增 launch_conv_amp_fwd_cuda / launch_conv_amp_inf_cuda
  18. conv_op.cpp: 新增 launch_conv_amp_bwd_cuda（双图双流，FP16 dW）
  19. conv_op.cpp: 完善 register_op_conv() 注册全部 6 个算子

Phase 5: Warmup 修复（低风险）
  20. op_registry.cpp: warmup 移除 cudaMalloc/Free，改用 ensure_workspace_grow

Phase 6: EXHAUSTIVE_C 集成（后续）
  21. cudnn_utils.h 或 conv_op_impl.cpp: 新增 `finalize_cudnn_graph` 重载
      `finalize_cudnn_graph(Graph*, handle, ConvSearchMode, exp_op_name)`
      用于 EXHAUSTIVE_C 模式下的引擎加载
```

---

## 十、风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| **dX 覆盖 X 数据竞争** | 训练结果完全错误 | BWD 严格 `wgrad(COMP_1) → event → dgrad(COMP_3)` |
| **padded_c 遗漏** | cuDNN FE error 8 | AMP 所有 `set_dim` 用 `dt.padded_c()`；FP32 用 `dt.c()` |
| **Stride 手写错误** | 静默数据错乱 | 强制 `make_nhwc_stride(dt)` / `make_krsc_stride(dt)` |
| **Workspace 泄漏** | OOM/崩溃 | `ensure_workspace_grow`，禁止 cudaMalloc/Free |
| **bn_stats 内存不足** | GenStats 越界写 | 形状 `{1,1,1,2*K}` 确保 `2*K*sizeof(float)` 空间 |
| **bn_stats 指针偏移** | GenStats 写错位置 | `sum → bn_ptr`, `sq_sum → bn_ptr + K*sizeof(float)` |
| **Graph Cache key 冲突** | 命中错误缓存 | key 包含 handle/shape/params/is_amp/op 全部字段 |
| **INF 错误触发 GenStats** | AMP INF 意外写 bn_stats | AMP INF 独立构建无 GenStats 的 graph |
| **CPU BWD 性能** | CPU 训练极慢 | naive 实现仅保证 correctness；生产以 GPU 为主 |
| **融合层索引偏移** | CBR/Bottleneck 等全部失效 | 保持 `infer_conv_tensors` 不变，新建 `infer_conv_tensors_with_bn_stats`；融合层不受影响 |
| **AMP FWD graph 断裂** | fe error / GenStats 无输出 | 遵循 `cbr_fwd_fp16.cpp` 模式：conv_out 直接 `set_output(true)`，不用 `set_is_virtual` |
| **Warmup BWD 多流不同步** | warmup 在 dgrad 完成前返回 | 保留专用 warmup 路径，BWD 分别同步 COMP_1 和 COMP_3 |

---

## 十一、参考代码索引

| 功能 | 文件 | 关键函数/类 |
|------|------|-------------|
| FE Graph Cache + Workspace | `fc_op.cpp` | `FcAmpFwdCache`、`launch_fc_amp_fwd_cuda()` |
| 多流 BWD 同步 | `fc_op.cpp` | `launch_fc_amp_bwd_cuda()` |
| AMP FWD Graph (Conv+GenStats) | `cbr_fwd_fp16.cpp` | `build_conv_genstats_graph()` |
| AMP BWD Graph (WGrad/DGrad) | `cbr_bwd_fp16.cpp` | `build_conv_wgrad_graph()` / `build_conv_dgrad_graph()` |
| AMP INF Graph | `cbr_inf_fp16.cpp` | `try_build_single_graph()` |
| padded_c 用法 | `maxpool_op.cpp` | `build_maxpool_fwd_graph()` |
| FE 公共辅助 | `cudnn_utils.h` | `create_cudnn_graph()`、`finalize_cudnn_graph()`、`to_fe_dim()` |
| 算子注册 | `fc_op.cpp` | `register_op_fc()` |
| Layer Descriptor 模式 | `layer_descriptor_registry.cpp` | BN 的 `build_bn_*` 动态 op 选择 |
| CPU XNNPACK | `conv_op.cpp` | `launch_conv_fwd_cpu_xnnpack()` |
| CPU Naive BWD | `conv_op.cpp` | `launch_conv_bwd_cpu()` |
| GlobalRegistry fixed 模式 | `global_registry.cpp` | `fixed_optimizer_kind_` |
| Stream 策略 | `op_stream_policy.cpp` | `get_op_default_stream()` |

---

*本方案综合 NDV1/NDV2/NDV3 三个方案，逐项裁决了 5 个关键分歧点，并基于 fc_op.cpp / cbr_*_fp16.cpp / maxpool_op.cpp / cudnn_utils.h 等现有参考代码，形成可直接编码的最终方案。核心要求 "算子内不得申请临时内存" 通过 `ensure_workspace_grow` 统一满足。*