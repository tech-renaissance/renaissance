# 卷积六大算子实现方案（NDV2）

> 基于NDV.md需求定义、现有代码真实状态、以及cbr_*/fc_op/maxpool_op参考代码的综合分析。  
> 版本: 2.0  
> 日期: 2026-06-02  

---

## 一、现状诊断（基于真实代码）

### 1.1 现有算子状态

| 算子 | 枚举 | launch_cuda | launch_cpu | graph缓存 | 双流BWD |
|------|:----:|:-----------:|:----------:|:---------:|:-------:|
| CONV_FP32_FWD | ✅ | ✅(重建/无缓存) | ✅ XNNPACK | ❌ | N/A |
| CONV_FP32_BWD | ✅ | ✅(单流COMP_1) | ✅ naive | ❌ | ❌ |
| CONV_FP32_INF | ❌ | — | — | — | — |
| CONV_AMP_FWD | ✅ | ❌(无实现) | — | — | — |
| CONV_AMP_BWD | ✅ | ❌(无实现) | — | — | — |
| CONV_AMP_INF | ❌ | — | — | — | — |

### 1.2 关键代码缺陷

1. **conv_op.cpp 每次launch重建Graph** + `cudaMalloc/Free` workspace（热路径性能极差）。
2. **CONV_FP32_BWD单流执行**：全部在COMP_1，未拆分为wgrad@COMP_1 + dgrad@COMP_3。
3. **缺少INF枚举**：`op_kind.h`无`CONV_FP32_INF`、`CONV_AMP_INF`。
4. **缺少bn_stats张量**：`infer_conv_tensors`返回6个张量，缺第7个。
5. **build_conv_*硬编码AMP**：`build_conv_forward`固定输出`CONV_AMP_FWD`，不根据`using_amp()`切换。
6. **conv_op_impl.cpp无AMP支持**：graph builder硬编码`DataType_t::FLOAT`。
7. **缺少ConvSearchMode/EXHAUSTIVE_C**：`types.h`和`global_registry.h`均不存在相关定义。
8. **warmup_single_cudnn_op仅处理FWD**：BWD无预热路径，INF无预热路径。

### 1.3 参考代码关键发现

| 文件 | 关键模式 |
|------|----------|
| `fc_op.cpp` | `FcAmpFwdCache` per-shape缓存 + `ctx.ensure_workspace_grow()`；BWD dW@COMP_1→event→dX@COMP_3同步模板 |
| `maxpool_op.cpp` | `padded_c()`在`set_dim`中的使用；Legacy API fallback模式 |
| `cbr_fwd_fp16.cpp` | Conv+GenStats融合图；`genstats(conv_out, attrs)`API；EXHAUSTIVE_C查表流程 |
| `cbr_bwd_fp16.cpp` | WGrad/DGrad独立graph；3-stream backward fork-join |
| `cbr_inf_fp16.cpp` | 纯conv_fprop无GenStats；single-graph mainline |
| `conv_op_legacy.cpp` | XNNPACK NHWC/KRSC实现参考；naive BWD dgrad+wgrad参考 |

---

## 二、约束总览（来自NDV.md）

### 2.1 卷积参数（BluePrint/ArchPlan层已校验，算子内不检查）

| 参数 | 约束 |
|------|------|
| kernel_size | 仅1,3,5,7（正方形） |
| stride | 仅1,2 |
| dilation | 固定1 |
| groups | 固定1（普通卷积） |
| bias | **不支持** |
| padding | 无限制 |

### 2.2 框架铁律

- **dX覆盖X**：BWD梯度输出覆盖原始输入，共用同一DTensor。
- **先dW后dX**：dW@COMP_1 → `cudaEventRecord` → dX@COMP_3等待该event。
- **张量对齐**：CPU/GPU/FP32/AMP同一层张量数量/种类/形状完全一致。
- **禁止算子内临时内存分配**：workspace用`ctx.ensure_workspace_grow()`。
- **NHWC物理布局**：cuDNN FE `set_stride({N_stride, C_stride, H_stride, W_stride})`，`C_stride=1`。
- **KRSC权重**：权重形状`{K,R,S,C}`，cuDNN filter dim `{K,C,R,S}`。

---

## 三、张量布局

### 3.1 Conv层张量清单（7个）

当前`infer_conv_tensors`返回6个，**新增第7个bn_stats**：

| 索引 | 名称 | Region | Dtype | 说明 |
|------|------|--------|-------|------|
| 0 | conv_weight | W_FIRST_CONV/W_DEEP_CONV | FP32 | 主权重 |
| 1 | conv_output | F_FEATURE_FP32/FP16 | 随AMP | 输出特征图 |
| 2 | conv_grad_slot | F_GRAD_SLOT_FP32/FP16 | 随AMP | 梯度槽（dX in-place覆盖X） |
| 3 | conv_weight_grad | G_FIRST_CONV/G_DEEP_CONV | FP32 | FP32权重梯度 |
| 4 | conv_amp_w_fp16 | A_FIRST_CONV/A_DEEP_CONV | FP16 | AMP工作权重 |
| 5 | conv_amp_g_fp16 | G_FIRST_CONV_FP16/G_DEEP_CONV_FP16 | FP16 | AMP权重梯度 |
| **6** | **conv_bn_stats** | **T_TEMP_FP32** | **FP32** | **BN统计量（新增）** |

### 3.2 bn_stats规格

- **形状**：`Shape{1, 1, 1, cp.out_channels}`（即`{1,1,1,K}`）。不使用padding（T_TEMP_FP32区`cuda_alignment=1`）。
- **大小**：实际需容纳`sum`+`sq_sum`两个`{1,K,1,1}`张量，共`2*K*sizeof(float)`字节。MemoryPlan分配时确保≥此大小。
- **写入**：仅`CONV_AMP_FWD`通过GenStats写入。其余5个算子不写入、不memset。

### 3.3 算子I/O绑定

**FWD / INF**（X由跨层链注入，不在input_indices中）：

```cpp
// FP32 FWD/INF
n.op = CONV_FP32_FWD / CONV_FP32_INF;
n.input_indices  = {0, 6};       // weight, bn_stats(占位)
n.output_indices = {1};          // output

// AMP FWD/INF
n.op = CONV_AMP_FWD / CONV_AMP_INF;
n.input_indices  = {4, 6};       // amp_w_fp16, bn_stats
n.output_indices = {1};          // output
```

**BWD**（dY由Compiler跨层注入为input_ids[0]）：

```cpp
// FP32 BWD
n.op = CONV_FP32_BWD;
n.input_indices  = {0, 1, 2};    // weight, output(=dY来源), grad_slot(=X)
n.output_indices = {2, 3};       // dX(in-place), dW→weight_grad

// AMP BWD
n.op = CONV_AMP_BWD;
n.input_indices  = {4, 1, 2};    // amp_w_fp16, output, grad_slot
n.output_indices = {2, 5};       // dX(in-place), dW→amp_g_fp16(FP16)
```

> 注：小伙伴S/K/C的某些方案中`input_indices`不含bn_stats或bn_stats作为输出，但根据NDV.md原文"卷积的正向应该有2个输入、2个输出——权重、特征图输入、特征图输出、bn_stats"，bn_stats是**输入**（作为占位），**不是输出**。Graph中bn_stats仅在AMP FWD的GenStats中被引用，其他算子Graph中不引用bn_stats节点，但input_indices仍保留以保持layer层面张量对齐。

---

## 四、基础设施扩展

### 4.1 `include/renaissance/graph/op_kind.h`

```cpp
enum class ComputeOp : uint16_t {
    // ... existing ...
    CONV_FP32_FWD,
    CONV_FP32_BWD,
    CONV_FP32_INF,      // ← 新增
    CONV_AMP_FWD,
    CONV_AMP_BWD,
    CONV_AMP_INF,       // ← 新增
    // ...
    COUNT,
};
```

### 4.2 `src/graph/op_kind.cpp`

```cpp
std::string compute_op_to_string(ComputeOp op) {
    switch (op) {
        // ...
        case ComputeOp::CONV_FP32_INF: return "CONV_FP32_INF";
        case ComputeOp::CONV_AMP_INF:  return "CONV_AMP_INF";
        // ...
    }
}
```

### 4.3 `include/renaissance/core/types.h`

```cpp
enum class ConvSearchMode : uint8_t {
    HEURISTIC_B = 0,    // 默认
    EXHAUSTIVE_C = 1    // 仅AMP Conv，仅A100/RTX5090
};
```

### 4.4 `include/renaissance/core/global_registry.h`

遵循**fixed变量**模式（与`fixed_optimizer_kind_`一致）：

```cpp
class GlobalRegistry {
public:
    void set_conv_search_mode(ConvSearchMode mode);
    ConvSearchMode conv_search_mode() const;
    // 链式调用包装（可选）
    GlobalRegistry& conv_search_mode(ConvSearchMode mode) {
        set_conv_search_mode(mode); return *this;
    }
private:
    std::atomic<int> fixed_conv_search_mode_{-1};  // -1 = unset
};
```

### 4.5 `src/core/global_registry.cpp`

```cpp
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
    if (old == value) return;
    TR_VALUE_ERROR("Cannot modify conv_search_mode after first assignment.");
}

ConvSearchMode GlobalRegistry::conv_search_mode() const {
    int v = fixed_conv_search_mode_.load(std::memory_order_relaxed);
    if (v == -1) return ConvSearchMode::HEURISTIC_B;  // 默认值
    return static_cast<ConvSearchMode>(v);
}
```

### 4.6 `src/backend/op_stream_policy.cpp`

```cpp
case ComputeOp::CONV_FP32_INF:
case ComputeOp::CONV_AMP_INF:
    return StreamKind::COMP_1;
```

### 4.7 `src/backend/op_registry.cpp`

`require_warmup`新增：

```cpp
case ComputeOp::CONV_FP32_INF:
case ComputeOp::CONV_AMP_INF:
    return true;
```

`warmup_single_cudnn_op`需扩展：当前只处理`CONV_FP32_FWD`和`CONV_AMP_FWD`，需统一处理全部6个算子。BWD因拆双图，走generic `launch_cuda`路径进行预热（首次执行即构建并缓存graph）。

---

## 五、Layer Descriptor与Compiler修改

### 5.1 `src/graph/layer_descriptor_registry.cpp`

**`infer_conv_tensors`**：在现有6张量后追加：

```cpp
{ TensorDesc d;
  d.name = "conv_bn_stats";
  d.shape = Shape{1, 1, 1, cp.out_channels};
  d.region = Region::T_TEMP_FP32;
  d.dtype = DType::FP32;
  descs.push_back(d);
}
```

**`build_conv_forward/backward/inference`**：根据`using_amp()`动态选择：

```cpp
bool amp = GlobalRegistry::instance().using_amp();
SubgraphPattern::Node n;
// forward:
n.op = amp ? ComputeOp::CONV_AMP_FWD : ComputeOp::CONV_FP32_FWD;
n.input_indices  = amp ? std::vector<size_t>{4, 6} : std::vector<size_t>{0, 6};
n.output_indices = {1};
// backward:
n.op = amp ? ComputeOp::CONV_AMP_BWD : ComputeOp::CONV_FP32_BWD;
n.input_indices  = amp ? std::vector<size_t>{4, 1, 2} : std::vector<size_t>{0, 1, 2};
n.output_indices = amp ? std::vector<size_t>{2, 5} : std::vector<size_t>{2, 3};
// inference:
n.op = amp ? ComputeOp::CONV_AMP_INF : ComputeOp::CONV_FP32_INF;
n.input_indices  = amp ? std::vector<size_t>{4, 6} : std::vector<size_t>{0, 6};
n.output_indices = {1};
```

### 5.2 `src/graph/compiler.cpp`

当前Compiler已正确处理Conv的跨层注入（FWD注入prev_output，BWD注入prev_grad和X）。因新增bn_stats张量后张量索引不变（bn_stats是第7个，追加在末尾），Compiler的`get_layer_output_id`(idx=1)和`get_grad_output_id`(idx=2)逻辑**无需修改**。

但需确认：`build_computation_graph`中BWD节点的`output_indices`为`{2}`（dX in-place）还是`{2,3}`（dX+dW）。当前代码可能只处理in-place dX，需要检查BWD是否自动追加weight_grad输出。参考FC BWD的处理方式。

---

## 六、cuDNN FE Graph缓存设计

### 6.1 Cache结构（参考fc_op.cpp的`FcAmpFwdCache`）

```cpp
struct ConvGraphCacheKey {
    uint64_t handle_bits;
    int32_t  N, H, W, C;
    int32_t  K, R, S;
    int32_t  pad_h, pad_w;
    int32_t  stride_h, stride_w;
    bool     is_amp;       // FP32 vs AMP
    ComputeOp op_kind;     // FWD/INF/BWD_wgrad/BWD_dgrad

    bool operator==(const ConvGraphCacheKey& o) const { /*...*/ }
};

struct ConvGraphCacheKeyHash {
    size_t operator()(const ConvGraphCacheKey& k) const { /*...*/ }
};

struct ConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    size_t workspace_size;
    // uid → dtensor id映射，用于快速填充Variant Pack
    std::vector<std::pair<int64_t, int32_t>> uid_to_tensor_id;
};
```

### 6.2 缓存实例

```cpp
// 每个算子类型独立缓存
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHash>
    s_conv_fp32_fwd_cache;
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHash>
    s_conv_fp32_wgrad_cache;
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHash>
    s_conv_fp32_dgrad_cache;
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHash>
    s_conv_amp_fwd_cache;
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHash>
    s_conv_amp_wgrad_cache;
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHash>
    s_conv_amp_dgrad_cache;
// INF可共用FWD的cache（key中加is_inf标记区分，或单独cache但构建逻辑相同）
```

### 6.3 通用执行流程

```cpp
template<typename CacheMap>
static ConvGraphCache& get_or_build_cache(
    CacheMap& cache_map,
    const ConvGraphCacheKey& key,
    cudnnHandle_t handle,
    std::function<ConvGraphCache(cudnnHandle_t)> builder)
{
    auto it = cache_map.find(key);
    if (it != cache_map.end()) return it->second;
    auto cache = builder(handle);
    auto [inserted_it, ok] = cache_map.emplace(key, std::move(cache));
    return inserted_it->second;
}
```

---

## 七、各算子详细实现

### 7.1 通用辅助函数

#### `make_nhwc_stride`

```cpp
inline std::vector<int64_t> make_nhwc_stride(const DTensor& dt) {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
}
```

#### `create_conv_graph`

```cpp
inline auto create_conv_graph(DType dtype) {
    auto g = std::make_shared<fe::graph::Graph>();
    if (dtype == DType::FP16) {
        g->set_io_data_type(fe::DataType_t::HALF)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
    } else {
        g->set_io_data_type(fe::DataType_t::FLOAT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
    }
    return g;
}
```

#### `finalize_conv_graph`

```cpp
inline void finalize_conv_graph(
    fe::graph::Graph* graph,
    cudnnHandle_t handle,
    ConvSearchMode mode,
    const char* exp_op_name)  // e.g., "conv_genstats", "conv_wgrad"
{
    if (mode == ConvSearchMode::EXHAUSTIVE_C && exp_op_name) {
#if defined(USING_A100) || defined(USING_RTX5090)
        // 查表逻辑（参考cbr_fwd_fp16.cpp）
        // ...
#endif
    }
    // fallback
    graph->create_execution_plans({fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK});
    graph->check_support(handle);
    graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
}
```

### 7.2 CONV_FP32_FWD / CONV_FP32_INF

**Graph构建**（`conv_op_impl.cpp`）：

```cpp
std::shared_ptr<fe::graph::Graph> build_conv_fp32_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_conv_graph(DType::FP32);

    auto X = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), dt_x.c(), dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::FLOAT));

    auto W = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), dt_w.c(), dt_w.h(), dt_w.w()})
        .set_stride(make_nhwc_stride(dt_w))
        .set_data_type(fe::DataType_t::FLOAT));

    auto opts = fe::graph::Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto Y = graph->conv_fprop(X, W, opts);
    Y->set_output(true)
      .set_name("Y")
      .set_dim({dt_y.n(), dt_y.c(), dt_y.h(), dt_y.w()})
      .set_stride(make_nhwc_stride(dt_y))
      .set_data_type(fe::DataType_t::FLOAT);

    finalize_conv_graph(graph.get(), handle, ConvSearchMode::HEURISTIC_B, nullptr);
    return graph;
}
```

**Launch**（`conv_op.cpp`）：

```cpp
static void launch_conv_fp32_fwd_cuda(...) {
    cudaStream_t s = ctx.stream(StreamKind::COMP_1);
    cudnnHandle_t h = ctx.cudnn_handle(StreamKind::COMP_1);
    int si = state.get_or_register(s);

    // 等待前序
    if (state.output_stream_idx >= 0) {
        cudaStreamWaitEvent(s, state.streams[state.output_stream_idx].last_done_event, 0);
    }

    // cache lookup / build
    ConvGraphCacheKey key{...};
    auto& cache = get_or_build_cache(s_conv_fp32_fwd_cache, key, h,
        [&](cudnnHandle_t hh) { return build_conv_fp32_fwd_graph(dt_x, dt_w, dt_y, cp, hh); });

    // workspace
    ctx.ensure_workspace_grow(StreamKind::COMP_1, cache.workspace_size);

    // variant pack
    auto vp = fe::graph::VariantPack();
    for (auto& [uid, tid] : cache.uid_to_tensor_id) {
        vp.set_workspace_pointer(uid, ctx.ptr_at(tid));
    }

    // execute
    cache.graph->execute(h, vp, ctx.workspace(StreamKind::COMP_1));

    // event
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

**INF与FWD完全等价**，launch函数可调用同一函数体，仅注册时`ComputeOp`不同。

### 7.3 CONV_FP32_BWD

**拆分为两个独立Graph**：

```cpp
// WGrad Graph (COMP_1)
std::shared_ptr<fe::graph::Graph> build_conv_fp32_wgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_x, const DTensor& dt_dw,
    const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_conv_graph(DType::FP32);
    auto dY = graph->tensor(...).set_data_type(FLOAT);
    auto X  = graph->tensor(...).set_data_type(FLOAT);
    auto dW = graph->conv_wgrad(dY, X, wgrad_opts);
    dW->set_output(true)
        .set_dim({dt_dw.n(), dt_dw.c(), dt_dw.h(), dt_dw.w()})
        .set_stride(make_nhwc_stride(dt_dw))
        .set_data_type(FLOAT);
    finalize_conv_graph(graph.get(), handle, HEURISTIC_B, nullptr);
    return graph;
}

// DGrad Graph (COMP_3)
std::shared_ptr<fe::graph::Graph> build_conv_fp32_dgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_w, const DTensor& dt_dx,
    const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_conv_graph(DType::FP32);
    auto dY = graph->tensor(...).set_data_type(FLOAT);
    auto W  = graph->tensor(...).set_data_type(FLOAT);
    auto dX = graph->conv_dgrad(dY, W, dgrad_opts);
    dX->set_output(true)
        .set_dim({dt_dx.n(), dt_dx.c(), dt_dx.h(), dt_dx.w()})
        .set_stride(make_nhwc_stride(dt_dx))  // 必须与X完全一致
        .set_data_type(FLOAT);
    finalize_conv_graph(graph.get(), handle, HEURISTIC_B, nullptr);
    return graph;
}
```

**Launch**（参考fc_op.cpp双流模式）：

```cpp
static void launch_conv_fp32_bwd_cuda(...) {
    cudaStream_t s_dw = ctx.stream(StreamKind::COMP_1);
    cudaStream_t s_dx = ctx.stream(StreamKind::COMP_3);
    cudnnHandle_t h_dw = ctx.cudnn_handle(StreamKind::COMP_1);
    cudnnHandle_t h_dx = ctx.cudnn_handle(StreamKind::COMP_3);
    int i_dw = state.get_or_register(s_dw);
    int i_dx = state.get_or_register(s_dx);

    // 等待前序（两流都wait）
    if (state.output_stream_idx >= 0) {
        auto& ev = state.streams[state.output_stream_idx].last_done_event;
        cudaStreamWaitEvent(s_dw, ev, 0);
        cudaStreamWaitEvent(s_dx, ev, 0);
    }

    // === COMP_1: WGrad ===
    auto& cache_w = get_or_build_cache(s_conv_fp32_wgrad_cache, key_w, h_dw, ...);
    ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);
    cache_w.graph->execute(h_dw, vp_w, ctx.workspace(StreamKind::COMP_1));
    cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    state.streams[i_dw].has_pending_work = true;

    // === COMP_3: DGrad（等待WGrad完成，保护X）===
    cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);
    auto& cache_x = get_or_build_cache(s_conv_fp32_dgrad_cache, key_x, h_dx, ...);
    ctx.ensure_workspace_grow(StreamKind::COMP_3, cache_x.workspace_size);
    cache_x.graph->execute(h_dx, vp_x, ctx.workspace(StreamKind::COMP_3));
    cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
    state.streams[i_dx].has_pending_work = true;

    state.output_stream_idx = i_dx;
}
```

### 7.4 CONV_AMP_FWD

**Graph构建**（参考cbr_fwd_fp16.cpp）：

```cpp
std::shared_ptr<fe::graph::Graph> build_conv_amp_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const DTensor& dt_bn, const ConvParams& cp, cudnnHandle_t handle)
{
    auto graph = create_conv_graph(DType::FP16);

    // ★ AMP用padded_c()而非c()
    int64_t PC_x = dt_x.padded_c();
    int64_t PC_w = dt_w.padded_c();
    int64_t K = dt_y.c();  // output channels

    auto X = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), PC_x, dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), PC_w, dt_w.h(), dt_w.w()})
        .set_stride(make_nhwc_stride(dt_w))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = fe::graph::Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto conv_out = graph->conv_fprop(X, W, opts);

    // GenStats
    auto genstats_opts = fe::graph::Genstats_attributes()
        .set_name("genstats")
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto [sum, sq_sum] = graph->genstats(conv_out, genstats_opts);

    // Y输出
    conv_out->set_output(true)
             .set_name("Y")
             .set_dim({dt_y.n(), dt_y.c(), dt_y.h(), dt_y.w()})
             .set_stride(make_nhwc_stride(dt_y))
             .set_data_type(fe::DataType_t::HALF);

    // sum/sq_sum → bn_stats
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

    // EXHAUSTIVE_C（仅AMP FWD）
    auto mode = GlobalRegistry::instance().conv_search_mode();
    finalize_conv_graph(graph.get(), handle, mode, "conv_genstats");
    return graph;
}
```

**Variant Pack中bn_stats映射**：

```cpp
float* bn_ptr = static_cast<float*>(ctx.ptr_at(node.input_ids[2]));  // bn_stats
vp.set_workspace_pointer(uid_sum,    bn_ptr);
vp.set_workspace_pointer(uid_sq_sum, bn_ptr + K);  // 偏移K个float
```

**注意**：`conv_out`在调用`genstats`后不要先`set_output(true)`，等genstats附加完成后再设置。否则graph拓扑会断裂。

### 7.5 CONV_AMP_BWD

**两个独立Graph**（数据类型HALF，dW输出FP16）：

```cpp
// WGrad
auto dW = graph->conv_wgrad(dY, X, wgrad_opts);
dW->set_output(true)
   .set_dim({dt_dw.n(), dt_dw.c(), dt_dw.h(), dt_dw.w()})
   .set_stride(make_nhwc_stride(dt_dw))
   .set_data_type(fe::DataType_t::HALF);  // ★ FP16输出

// DGrad
auto dX = graph->conv_dgrad(dY, W, dgrad_opts);
dX->set_output(true)
   .set_dim({dt_dx.n(), dt_dx.padded_c(), dt_dx.h(), dt_dx.w()})
   .set_stride(make_nhwc_stride(dt_dx))
   .set_data_type(fe::DataType_t::HALF);
```

**流同步**：与FP32 BWD完全一致（`wgrad@COMP_1 → event → dgrad@COMP_3`）。

**EXHAUSTIVE_C**：WGrad查`"conv_wgrad"`，DGrad查`"conv_dgrad"`。

### 7.6 CONV_AMP_INF

纯`conv_fprop`，**无GenStats**。结构与FP32 FWD相同，仅数据类型为HALF：

```cpp
graph->set_io_data_type(HALF)
      ->set_compute_data_type(FLOAT);
// X(HALF), W(HALF) → conv_fprop → Y(HALF)
// 无GenStats，bn_stats不占位在Graph中
```

---

## 八、CPU实现

### 8.1 FWD / INF

- **共用同一函数**：`launch_conv_fwd_cpu()`。
- **XNNPACK优先**：使用`xnn_define_convolution_2d`，filter格式KRSC（OHWI）。参考`conv_op_legacy.cpp`的XNNPACK实现。
- **Stride repacking**：若DTensor stride非紧凑，使用`repack_nhwc_to_dense`/`repack_dense_to_nhwc`辅助函数（参考`conv_op_legacy.cpp`）。
- **Fallback**：XNNPACK不支持时回退到naive嵌套循环。

### 8.2 BWD

- **Naive实现**：6重循环计算dgrad+wgrad。先memset dX=0、dW=0，再累加。
- **顺序执行**：单线程先算dW后算dX，天然满足约束。
- **参考**：`conv_op_legacy.cpp`中的`launch_tr_conv_dgrad_fp32_kernel_cpu`和`launch_tr_conv_wgrad_fp32_kernel_cpu`。

### 8.3 AMP CPU

不支持。统一指向：

```cpp
static void launch_conv_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    TR_THROW_NOT_SUPPORTED_GPU("Conv AMP operators do not support CPU execution");
}
```

---

## 九、算子注册

```cpp
void register_op_conv() {
    // FP32 FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_FWD)];
        e.op = ComputeOp::CONV_FP32_FWD;
        e.launch_cpu = launch_conv_fwd_cpu;
        e.launch_cuda = launch_conv_fp32_fwd_cuda;
    }
    // FP32 BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_BWD)];
        e.op = ComputeOp::CONV_FP32_BWD;
        e.launch_cpu = launch_conv_bwd_cpu;
        e.launch_cuda = launch_conv_fp32_bwd_cuda;
    }
    // FP32 INF（复用FWD的CPU/GPU launch）
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_INF)];
        e.op = ComputeOp::CONV_FP32_INF;
        e.launch_cpu = launch_conv_fwd_cpu;           // INF=FWD
        e.launch_cuda = launch_conv_fp32_inf_cuda;    // 可调用fwd实现
    }
#ifdef TR_USE_CUDA
    // AMP FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_AMP_FWD)];
        e.op = ComputeOp::CONV_AMP_FWD;
        e.launch_cpu = launch_conv_amp_cpu_not_supported;
        e.launch_cuda = launch_conv_amp_fwd_cuda;
    }
    // AMP BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_AMP_BWD)];
        e.op = ComputeOp::CONV_AMP_BWD;
        e.launch_cpu = launch_conv_amp_cpu_not_supported;
        e.launch_cuda = launch_conv_amp_bwd_cuda;
    }
    // AMP INF
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

## 十、文件变更清单

| 序号 | 文件 | 变更 | 优先级 |
|:----:|------|------|:------:|
| 1 | `include/renaissance/graph/op_kind.h` | 新增`CONV_FP32_INF`、`CONV_AMP_INF` | P0 |
| 2 | `src/graph/op_kind.cpp` | `compute_op_to_string`补充 | P0 |
| 3 | `include/renaissance/core/types.h` | 新增`ConvSearchMode`枚举 | P0 |
| 4 | `include/renaissance/core/global_registry.h` | 新增`set_conv_search_mode`/`conv_search_mode` | P0 |
| 5 | `src/core/global_registry.cpp` | 实现setter/getter | P0 |
| 6 | `src/backend/op_stream_policy.cpp` | 新增INF流映射 | P0 |
| 7 | `src/backend/op_registry.cpp` | `require_warmup`加INF；`warmup_single_cudnn_op`扩展 | P0 |
| 8 | `src/graph/layer_descriptor_registry.cpp` | 加bn_stats；修正`build_conv_*` | P1 |
| 9 | `src/backend/ops/dtensor/conv_op_impl.cpp` | **重写**：全部6个graph builder | P1 |
| 10 | `src/backend/ops/dtensor/conv_op.cpp` | **重写**：6个launch + CPU + 注册 | P1 |
| 11 | `src/graph/compiler.cpp` | 确认BWD output_indices处理（如需要） | P2 |

---

## 十一、实施优先级

```
Phase 1: 基础设施（无风险，1-2天）
  ├─ op_kind.h/cpp: 加INF枚举
  ├─ types.h: 加ConvSearchMode
  ├─ global_registry.h/cpp: 加conv_search_mode
  ├─ op_stream_policy.cpp: INF映射
  └─ op_registry.cpp: require_warmup扩展

Phase 2: FP32完整改造（中风险，2-3天）
  ├─ layer_descriptor_registry.cpp: bn_stats + build_conv_*修正
  ├─ conv_op_impl.cpp: build_conv_fp32_fwd_graph（带缓存）
  ├─ conv_op_impl.cpp: build_conv_fp32_wgrad/dgrad_graph
  ├─ conv_op.cpp: launch_conv_fp32_fwd_cuda（缓存+workspace复用）
  ├─ conv_op.cpp: launch_conv_fp32_bwd_cuda（拆双图+双流同步）
  ├─ conv_op.cpp: launch_conv_fp32_inf_cuda（复用FWD）
  └─ 编译验证FP32 FWD/BWD/INF

Phase 3: AMP实现（高风险，3-4天）
  ├─ conv_op_impl.cpp: build_conv_amp_fwd_graph（Conv+GenStats）
  ├─ conv_op_impl.cpp: build_conv_amp_wgrad/dgrad_graph
  ├─ conv_op_impl.cpp: build_conv_amp_inf_graph
  ├─ conv_op.cpp: launch_conv_amp_fwd_cuda
  ├─ conv_op.cpp: launch_conv_amp_bwd_cuda（双流同步+FP16 dW）
  ├─ conv_op.cpp: launch_conv_amp_inf_cuda
  ├─ conv_op.cpp: 完善register_op_conv()
  └─ 编译验证全部6个算子

Phase 4: EXHAUSTIVE_C（低风险，1天）
  ├─ 接入include/generated/经验表
  ├─ 平台检测（A100/RTX5090）
  └─ fallback到Heuristic B
```

---

## 十二、风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| **dX覆盖X数据竞争** | 训练结果完全错误 | BWD严格`wgrad(COMP_1)→event→dgrad(COMP_3)` |
| **padded_c遗漏** | cuDNN FE找不到plan（error 8） | AMP所有`set_dim`用`dt.padded_c()`，FP32用`dt.c()` |
| **Stride手写错误** | 静默数据错乱 | 强制用`dt.n_stride_cuda()`系列，禁止手算 |
| **Workspace泄漏** | OOM/性能差 | `ctx.ensure_workspace_grow()`，禁止cudaMalloc/Free |
| **bn_stats布局** | GenStats写错位置 | `sum→bn_ptr`, `sq_sum→bn_ptr+K*sizeof(float)` |
| **Graph Cache key冲突** | 缓存命中错误graph | key包含handle/shape/params/is_amp/op_kind全部字段 |
| **CPU BWD性能** | CPU训练极慢 | naive实现仅保证correctness，生产以GPU为主 |
| **INF复用FWD缓存** | 意外GenStats | INF单独cache或key中加`is_inf`标记 |

---

## 十三、参考索引

| 功能 | 文件 | 关键函数 |
|------|------|----------|
| AMP FWD Graph | `cbr_fwd_fp16.cpp` | `build_conv_genstats_graph()` |
| AMP BWD Graph | `cbr_bwd_fp16.cpp` | `build_conv_wgrad_graph()`/`build_conv_dgrad_graph()` |
| AMP INF Graph | `cbr_inf_fp16.cpp` | `try_build_single_graph()` |
| 多流BWD同步 | `fc_op.cpp` | `launch_fc_amp_bwd_cuda()` |
| Graph缓存+Workspace | `fc_op.cpp` | `FcAmpFwdCache` |
| padded_c用法 | `maxpool_op.cpp` | `build_maxpool_fwd_graph()` |
| XNNPACK CPU | `conv_op_legacy.cpp` | `launch_tr_conv_fwd_fp32_kernel_cpu()` |
| CPU naive BWD | `conv_op_legacy.cpp` | `launch_tr_conv_dgrad/wgrad_fp32_kernel_cpu()` |
| 算子添加流程 | `ATN2.md` | 完整checklist |
