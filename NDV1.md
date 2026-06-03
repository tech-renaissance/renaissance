> 版本: 1.0.0
> 日期: 2026-06-03
> 来源: NDV.md（4位小伙伴S/K/C/D观点综合）+ 代码现状分析
> 范围: CONV_FP32_FWD / CONV_FP32_BWD / CONV_FP32_INF / CONV_AMP_FWD / CONV_AMP_BWD / CONV_AMP_INF

---

## 一、现状诊断

### 1.1 已有实现

| 算子 | 状态 | 问题 |
|------|------|------|
| `CONV_FP32_FWD` | 已注册，有CPU+CUDA launch | CUDA: 每调用 `cudaMalloc`/`cudaFree` workspace；无 per-shape Graph Cache；输出 Y 的 stride 手写 `align_up(K,8)` 而非 DTensor 真实 stride |
| `CONV_FP32_BWD` | 已注册，有CPU+CUDA launch | CUDA: 单图单流（COMP_1），未按规范拆分为 wgrad(COMP_1) + dgrad(COMP_3)；同样存在 workspace 泄漏和 stride 手写问题 |
| `CONV_FP32_INF` | **缺失** | `ComputeOp` 枚举无此定义 |
| `CONV_AMP_FWD`  | 枚举已有，无 launch 实现 | `conv_op_impl.cpp` 中 `build_graph_fwd` 硬编码 `FLOAT`，未接入 launch 路径 |
| `CONV_AMP_BWD`  | 枚举已有，无 launch 实现 | 同上 |
| `CONV_AMP_INF`  | **缺失** | `ComputeOp` 枚举无此定义 |

### 1.2 配套基础设施缺口

| 组件 | 缺口 |
|------|------|
| `op_kind.h` | 缺 `CONV_FP32_INF`、`CONV_AMP_INF` |
| `types.h` | 缺 `ConvSearchMode` 枚举 |
| `global_registry.h/cpp` | 缺 `conv_search_mode()` setter/getter |
| `layer_descriptor_registry.cpp` | 缺第7张量 bn_stats；`build_conv_*` 硬编码 AMP |
| `op_registry.cpp` | `require_warmup` 缺 INF；warmup 路径缺 INF/AMP |
| `op_stream_policy.cpp` | 缺 INF 流映射 |

---

## 二、参数约束（BluePrint/ArchPlan 层已保证，算子内不检查）

| 参数 | 允许值 |
|------|--------|
| kernel_size | 1, 3, 5, 7（正方形，kernel_h == kernel_w） |
| stride | 1, 2（stride_h == stride_w） |
| dilation | 固定 1 |
| groups | 固定 1（普通卷积，无 Depthwise） |
| bias | **不支持** |
| padding | 无限制 |

---

## 三、核心设计原则

### 3.1 权重布局：KRSC

权重的物理布局为 KRSC，即形状 `{K, R, S, C}`，其中 C 是最内层维度（连续存储）。这是需求文档反复强调的硬性约束：

- 在 cuDNN Frontend 中，`set_dim({K, C, R, S})`，`set_stride` 使用 DTensor 真实值
- `dt_w.n_stride_cuda()` = `R*S*C`（K 步幅）
- `dt_w.c_stride_cuda()` = `1`（C 步幅，NHWC 语义）
- `dt_w.h_stride_cuda()` = `S*C`（R 步幅）
- `dt_w.w_stride_cuda()` = `C`（S 步幅）
- XNNPACK 的 filter 格式为 `{K, R, S, C}`，与框架布局天然一致

### 3.2 NHWC 物理布局与 cuDNN FE Stride

框架张量物理布局为 NHWC。cuDNN FE 的 `set_dim` 顺序为 `{N, C, H, W}`（逻辑声明），`set_stride` 顺序也为 `{N, C, H, W}`，其中 **`stride[1]=1` 表示 C 维度连续，即 NHWC**。

**铁律：所有 cuDNN FE Tensor 的 stride 必须使用 DTensor 真实 stride，禁止手写 `align_up`。** 现有代码中 `build_conv_fwd_graph` 对 Y 的 stride 手写 `K_aligned` 必须修正。

### 3.3 dX 覆盖 X（In-Place BWD）

反向传播时 dX 复用 X 的同一 DTensor buffer（索引 2 — grad_slot）。**必须先算 dW（wgrad），再算 dX（dgrad），避免 X 在 wgrad 消费前被覆盖。**

| 阶段 | StreamKind | 同步 |
|------|-----------|------|
| wgrad | COMP_1 | 完成后 `cudaEventRecord(state.streams[i_dw].last_done_event, s_dw)` |
| dgrad | COMP_3 | 启动前 `cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0)` |
| output_stream_idx | — | 设为 `i_dx`（dX 所在流） |

**此规则适用于 FP32 BWD 和 AMP BWD。** 现有的单图 BWD 必须拆分为两个独立 Graph，因为 cuDNN FE Graph 的 `execute()` 绑定到单个 `cudnnHandle_t`（即单个 stream），无法将 wgrad 和 dgrad 分配到不同 stream。

COMP_3 而非 COMP_2 的原因：与 FC 算子对齐，且 COMP_2 留给 BN 算子。

### 3.4 内存管理铁律

- **禁止在算子内 `cudaMalloc`/`cudaFree`**。所有临时 workspace 通过 `ctx.ensure_workspace_grow(StreamKind, size)` 复用 DeviceContext 管理的 per-stream workspace
- 如需额外临时张量，应在 Compiler 阶段申请 T_TEMP_FP32/FP16 区 DTensor

### 3.5 张量对齐原则

同一层输入输出张量的数量、种类、形状在 CPU/GPU/FP32/AMP 下完全一致。bn_stats 在所有 6 个算子中均作为输入存在，但仅 `CONV_AMP_FWD` 真正写入。

---

## 四、张量布局（7 张量）

### 4.1 Conv 层完整张量清单

| 索引 | 名称 | 形状 | Region | DType | 说明 |
|------|------|------|--------|-------|------|
| 0 | conv_weight | `{K, R, S, C}` | W_FIRST_CONV / W_DEEP_CONV | FP32 | 主权重（KRSC 格式） |
| 1 | conv_output | `{N, OH, OW, K}` | F_FEATURE_FP32 / F_FEATURE_FP16 | varies | 输出特征图 |
| 2 | conv_grad_slot | `{N, H, W, C}` | F_GRAD_SLOT_FP32 / F_GRAD_SLOT_FP16 | varies | 梯度槽（dX 覆盖 X，in-place） |
| 3 | conv_weight_grad | `{K, R, S, C}` | G_FIRST_CONV / G_DEEP_CONV | FP32 | 权重梯度（经 CAST 后） |
| 4 | conv_amp_w_fp16 | `{K, R, S, C}` | A_FIRST_CONV / A_DEEP_CONV | FP16 | AMP 工作权重（!AMP 时占位） |
| 5 | conv_amp_g_fp16 | `{K, R, S, C}` | G_FIRST_CONV_FP16 / G_DEEP_CONV_FP16 | FP16 | AMP 权重梯度（BWD 直接产出 FP16） |
| **6** | **conv_bn_stats** | `{1, 1, 1, K}` | **T_TEMP_FP32** | **FP32** | **BN 统计量（新增）** |

### 4.2 bn_stats 规格

- 形状 `{1, 1, 1, K}`（4D NHWC），不使用 padding
- 区域 `T_TEMP_FP32`（cuda_alignment=1，无 padding）
- 仅 `CONV_AMP_FWD` 通过 GenStats 写入 sum + sq_sum；其余 5 个算子不写入、不 memset
- GenStats 输出两个 `{1, K, 1, 1}` 张量，Variant Pack 中 `sum` 指向 bn_stats 首地址，`sq_sum` 指向 `bn_stats_ptr + K * sizeof(float)`

### 4.3 Layer Descriptor 修正

**`infer_conv_tensors`** 在现有 6 张量后追加 bn_stats（索引 6）。

**`build_conv_forward`**：根据 `using_amp()` 动态选择 `ComputeOp` 和 `input_indices`：
- FP32: `op=CONV_FP32_FWD`, `input_indices={0,6}`, `output_indices={1}`
- AMP: `op=CONV_AMP_FWD`, `input_indices={4,6}`, `output_indices={1}`

**`build_conv_backward`**：
- FP32: `op=CONV_FP32_BWD`, `input_indices={0,1,2}`, `output_indices={2,3}`
- AMP: `op=CONV_AMP_BWD`, `input_indices={4,1,2}`, `output_indices={2,5}`

**`build_conv_inference`**：
- FP32: `op=CONV_FP32_INF`, `input_indices={0,6}`, `output_indices={1}`
- AMP: `op=CONV_AMP_INF`, `input_indices={4,6}`, `output_indices={1}`

### 4.4 各算子 I/O 速查表

| 算子 | Inputs | Outputs | Stream |
|------|--------|---------|--------|
| CONV_FP32_FWD | X, W(0), bn_stats(6) | Y(1) | COMP_1 |
| CONV_FP32_INF | X, W(0), bn_stats(6) | Y(1) | COMP_1 |
| CONV_FP32_BWD | dY, W(0), Y(1), X(2), bn_stats(6) | dX(2), dW(3) | dW:COMP_1, dX:COMP_3 |
| CONV_AMP_FWD | X, W_fp16(4), bn_stats(6) | Y(1) | COMP_1 |
| CONV_AMP_INF | X, W_fp16(4), bn_stats(6) | Y(1) | COMP_1 |
| CONV_AMP_BWD | dY, W_fp16(4), Y(1), X(2), bn_stats(6) | dX(2), dW_fp16(5) | dW:COMP_1, dX:COMP_3 |

---

## 五、CUDA 实现方案

### 5.1 Per-Shape Graph Cache

参考 FC 算子的 `FcAmpFwdCache` 机制，为 Conv 实现 per-shape 缓存：

```cpp
struct ConvGraphCacheKey {
    uint64_t handle_bits;       // cudnnHandle 指针
    int32_t N, H, W, C;         // 输入形状
    int32_t K, R, S;            // 权重形状
    int32_t pad_h, pad_w;
    int32_t stride_h, stride_w;
    bool is_amp;                // AMP vs FP32
    ComputeOp op;               // 区分 FWD/BWD-wgrad/BWD-dgrad/INF
    bool operator==(const ConvGraphCacheKey& o) const = default;
};

struct ConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<int64_t, int64_t> uid_to_dtensor_id;
    size_t workspace_size;
};

// 静态缓存 map
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHash> s_conv_caches;
```

Cache 命中时：
1. 复用已构建的 Graph 对象
2. 通过 `uid_to_dtensor_id` 映射填充 Variant Pack（`ctx.ptr_at(id)` 取指针）
3. `ctx.ensure_workspace_grow(sk, cache.workspace_size)` 复用 workspace

### 5.2 NHWC Stride 映射规范

```cpp
// 特征图 NHWC：dim={N, C, H, W}
auto make_nhwc_stride = [](const DTensor& dt) {
    return std::vector<int64_t>{
        dt.n_stride_cuda(),   // N 步幅
        dt.c_stride_cuda(),   // C 步幅（应为 1）
        dt.h_stride_cuda(),   // H 步幅
        dt.w_stride_cuda()    // W 步幅
    };
};

// 权重 KRSC：dim={K, C, R, S}
auto make_krsc_stride = [](const DTensor& dt) {
    return std::vector<int64_t>{
        dt.n_stride_cuda(),   // K 步幅（= R*S*C）
        dt.c_stride_cuda(),   // C 步幅（= 1）
        dt.h_stride_cuda(),   // R 步幅（= S*C）
        dt.w_stride_cuda()    // S 步幅（= C）
    };
};
```

**AMP 模式下 `padded_c()`**：cuDNN FE 的 `set_dim` 必须使用 `dt.padded_c()`（对齐到 8 的倍数），但 `set_stride` 使用 `dt.c_stride_cuda()` 等真实值。FP32 模式下 `padded_c() == c()`。

### 5.3 CONV_FP32_FWD / CONV_FP32_INF

**cuDNN 操作**：`conv_fprop` only（无 GenStats，FP32 不支持 GenStats）

**Graph 构建**：

```cpp
graph->set_io_data_type(FLOAT).set_compute_data_type(FLOAT);

auto X = graph->tensor(Tensor_attributes()
    .set_dim({N, C, H, W})
    .set_stride(make_nhwc_stride(dt_x))
    .set_data_type(FLOAT));

auto W = graph->tensor(Tensor_attributes()
    .set_dim({K, C, R, S})       // K,R,S,C → cuDNN dim K,C,R,S
    .set_stride(make_krsc_stride(dt_w))
    .set_data_type(FLOAT));

auto conv_opts = Conv_fprop_attributes()
    .set_padding({pad_h, pad_w})
    .set_stride({stride_h, stride_w})
    .set_dilation({1, 1});

auto Y = graph->conv_fprop(X, W, conv_opts);
Y->set_output(true)
  .set_dim({N, K, OH, OW})
  .set_stride(make_nhwc_stride(dt_y))  // ★ 必须用 DTensor 真实 stride
  .set_data_type(FLOAT);
```

**关键修正**：现有代码对 Y 手写 `align_up_c(K)` 作为 stride，必须改为 `dt_y.n_stride_cuda()` 系列。

**INF 与 FWD 在 cuDNN 层面完全等价**（都是 `conv_fprop`），可共用 graph 构建函数和 cache。

**单流 COMP_1**。

### 5.4 CONV_FP32_BWD（双图 + 跨流同步）

拆分为两个独立 Graph：

**WGrad Graph（COMP_1）**：

```cpp
wgrad_graph->set_io_data_type(FLOAT).set_compute_data_type(FLOAT);
auto dY = wgrad_graph->tensor(...FLOAT...);
auto X  = wgrad_graph->tensor(...FLOAT...);
auto dW = wgrad_graph->conv_wgrad(dY, X, wgrad_opts);
dW->set_output(true)
   .set_dim({K, R, S, C})            // 输出 KRSC
   .set_stride(make_krsc_stride(dt_dw))
   .set_data_type(FLOAT);
```

**DGrad Graph（COMP_3）**：

```cpp
dgrad_graph->set_io_data_type(FLOAT).set_compute_data_type(FLOAT);
auto dY = dgrad_graph->tensor(...FLOAT...);
auto W  = dgrad_graph->tensor(...FLOAT...);
auto dX = dgrad_graph->conv_dgrad(dY, W, dgrad_opts);
dX->set_output(true)
   .set_dim({N, C, H, W})
   .set_stride(make_nhwc_stride(dt_dx))  // 与 X 完全一致（dX 覆盖 X）
   .set_data_type(FLOAT);
```

**Launch 同步**：

```cpp
cudaStream_t s_dw = ctx.stream(StreamKind::COMP_1);
cudaStream_t s_dx = ctx.stream(StreamKind::COMP_3);
cudnnHandle_t h_dw = ctx.cudnn_handle(StreamKind::COMP_1);
cudnnHandle_t h_dx = ctx.cudnn_handle(StreamKind::COMP_3);
int i_dw = state.get_or_register(s_dw);
int i_dx = state.get_or_register(s_dx);

// 等待前序算子
int out_idx = state.output_stream_idx;
if (out_idx >= 0) {
    cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
    cudaStreamWaitEvent(s_dx, state.streams[out_idx].last_done_event, 0);
}

// COMP_1: wgrad
auto& cache_w = get_or_build_wgrad_cache(...);
void* ws_w = ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);
cache_w.graph->execute(h_dw, vp_w, ws_w);
cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
state.streams[i_dw].has_pending_work = true;

// COMP_3: dgrad（等待 wgrad 完成，保护 X）
cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);
auto& cache_x = get_or_build_dgrad_cache(...);
void* ws_x = ctx.ensure_workspace_grow(StreamKind::COMP_3, cache_x.workspace_size);
cache_x.graph->execute(h_dx, vp_x, ws_x);
cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
state.streams[i_dx].has_pending_work = true;

state.output_stream_idx = i_dx;
```

### 5.5 CONV_AMP_FWD（Conv + GenStats）

**参考**：`cbr_fwd_fp16.cpp` 的 `build_conv_genstats_graph()`

**Graph 构建**：

```cpp
graph->set_io_data_type(HALF)
     ->set_intermediate_data_type(FLOAT)
     ->set_compute_data_type(FLOAT);

auto X = graph->tensor(...)
    ->set_dim({N, dt_x.padded_c(), H, W})  // AMP: padded_c()
    ->set_stride(make_nhwc_stride(dt_x))
    ->set_data_type(HALF);

auto W = graph->tensor(...)
    ->set_dim({K, dt_w.padded_c(), R, S})
    ->set_stride(make_krsc_stride(dt_w))
    ->set_data_type(HALF);

auto conv_out = graph->conv_fprop(X, W, conv_opts);

// GenStats 附加
auto genstats_attrs = Genstats_attributes()
    .set_compute_data_type(FLOAT);
auto [sum, sq_sum] = graph->genstats(conv_out, genstats_attrs);

// Y 输出
conv_out->set_output(true)
    ->set_dim({N, K, OH, OW})
    ->set_stride(make_nhwc_stride(dt_y))
    ->set_data_type(HALF);

// sum / sq_sum 输出到 bn_stats 的两个半区
sum->set_output(true)
    ->set_dim({1, K, 1, 1})
    ->set_stride({K, 1, K, K})
    ->set_data_type(FLOAT);
sq_sum->set_output(true)
    ->set_dim({1, K, 1, 1})
    ->set_stride({K, 1, K, K})
    ->set_data_type(FLOAT);
```

**Variant Pack 中 bn_stats 映射**：

```cpp
float* bn_stats_ptr = static_cast<float*>(ctx.ptr_at(node.input_ids[2])); // bn_stats 是输入
vp[uid_sum]    = bn_stats_ptr;
vp[uid_sq_sum] = bn_stats_ptr + K;  // 偏移 K 个 float
```

**单流 COMP_1**。

### 5.6 CONV_AMP_BWD（双图 + 跨流同步 + FP16 dW）

与 FP32 BWD 完全相同的流架构，但数据类型为 HALF，且 **dW 输出 FP16**：

**WGrad Graph（COMP_1）**：

```cpp
wgrad_graph->set_io_data_type(HALF)->set_compute_data_type(FLOAT);
auto dY = wgrad_graph->tensor(...)->set_data_type(HALF);
auto X  = wgrad_graph->tensor(...)->set_data_type(HALF);
auto dW = wgrad_graph->conv_wgrad(dY, X, wgrad_opts);
dW->set_output(true)
   ->set_dim({K, R, S, C})
   ->set_stride(make_krsc_stride(dt_dw))
   ->set_data_type(HALF);  // ★ AMP BWD 输出 FP16 dW
```

**DGrad Graph（COMP_3）**：

```cpp
dgrad_graph->set_io_data_type(HALF)->set_compute_data_type(FLOAT);
auto dY = dgrad_graph->tensor(...)->set_data_type(HALF);
auto W  = dgrad_graph->tensor(...)->set_data_type(HALF);
auto dX = dgrad_graph->conv_dgrad(dY, W, dgrad_opts);
dX->set_output(true)
   ->set_dim({N, dt_dx.padded_c(), H, W})
   ->set_stride(make_nhwc_stride(dt_dx))
   ->set_data_type(HALF);
```

**关键点**：
- dW 输出 FP16 到 `G_*_CONV_FP16`（索引 5），不同于 FC 的 FP32 dW。后续由 Compiler 插入 `RANGE_CAST_FP16_TO_FP32` 批量转换到 G_*_CONV（索引 3）
- dX 的 stride 必须与 X 完全一致（包括 `padded_c`）
- AMP 使用 `dt.padded_c()` 作为 `set_dim` 参数
- 流同步代码与 FP32 BWD 完全一致

### 5.7 CONV_AMP_INF

纯 `conv_fprop`，**无 GenStats**：

```cpp
graph->set_io_data_type(HALF)
     ->set_intermediate_data_type(HALF)
     ->set_compute_data_type(FLOAT);

auto X = graph->tensor(...)->set_data_type(HALF);
auto W = graph->tensor(...)->set_data_type(HALF);
auto Y = graph->conv_fprop(X, W, conv_opts);
Y->set_output(true)->set_data_type(HALF);
// 无 GenStats，bn_stats 仅占位
```

单流 COMP_1。

---

## 六、EXHAUSTIVE_C 穷举式搜索

### 6.1 平台支持

| 平台 | EXHAUSTIVE_C | 回退模式 |
|------|-------------|----------|
| A100-SXM4-80GB (sm_80) | 支持 | Heuristic B |
| RTX 5090 (sm_100) | 支持 | Heuristic B |
| 其他 | 不支持 | Heuristic B |

### 6.2 集成流程

```cpp
if (GlobalRegistry::instance().conv_search_mode() == ConvSearchMode::EXHAUSTIVE_C) {
#if defined(USING_A100) || defined(USING_RTX5090)
    std::string key = build_shape_key(op_name, "fp16", N, H, W, C, K, R, S, stride, pad);
    auto exp_rec = ta_v4::experience::lookup(key);
    if (exp_rec != nullptr) {
        graph->create_execution_plans({HeurMode_t::A, HeurMode_t::B});
        auto [status, matched] = match_and_build_plan(graph, candidates, exp_rec, handle);
        if (!matched) goto fallback;
    } else { goto fallback; }
#else
    goto fallback;
#endif
} else {
fallback:
    graph->create_execution_plans({HeurMode_t::B, HeurMode_t::FALLBACK});
    graph->check_support(handle);
    graph->build_plans(BuildPlanPolicy_t::HEURISTICS_CHOICE);
}
```

### 6.3 Shape Key 前缀

| 算子 | Key 前缀 |
|------|---------|
| AMP FWD | `"conv_genstats"` |
| AMP BWD wgrad | `"conv_wgrad"` |
| AMP BWD dgrad | `"conv_dgrad"` |
| AMP INF | `"conv_fprop"` |

---

## 七、CPU 实现

### 7.1 FWD / INF

共用 `launch_conv_fwd_cpu()`：
- 优先 XNNPACK（`xnn_define_convolution_2d`，NHWC layout，filter 格式 KRSC）
- 若 XNNPACK 不支持，fallback 到 naive 嵌套循环
- INF 与 FWD 计算逻辑完全相同，仅算子枚举不同

### 7.2 BWD

`launch_conv_bwd_cpu()`：
- Naive 6 重循环实现 dgrad + wgrad
- 先 memset dX=0、dW=0，再累加
- 单线程顺序执行（先 dW 后 dX，天然满足 dX 覆盖 X 约束）
- 注意 C 维度连续（NHWC layout），内存访问模式 `X[n*H*W*C + (ih*W+iw)*C + c]`

### 7.3 AMP CPU

不支持，统一指向 `launch_conv_amp_cpu_not_supported()`，抛出异常。

---

## 八、枚举与注册表扩展

### 8.1 `include/renaissance/graph/op_kind.h`

```cpp
enum class ComputeOp : uint16_t {
    // 在 CONV_AMP_BWD 之后新增：
    CONV_FP32_INF,   // FP32 推理
    CONV_AMP_INF,    // AMP 推理
    // ...
    COUNT,
};
```

### 8.2 `include/renaissance/core/types.h`

```cpp
enum class ConvSearchMode : uint8_t {
    HEURISTIC_B = 0,   // 默认：cuDNN 启发式 Mode B
    EXHAUSTIVE_C = 1   // 穷举式搜索（仅 AMP Conv，仅 A100/RTX5090）
};
```

### 8.3 `include/renaissance/core/global_registry.h` + `src/core/global_registry.cpp`

```cpp
// global_registry.h
class GlobalRegistry {
public:
    GlobalRegistry& conv_search_mode(ConvSearchMode mode);  // Setter（链式调用）
    ConvSearchMode conv_search_mode() const;                // Getter
private:
    std::atomic<int> fixed_conv_search_mode_{
        static_cast<int>(ConvSearchMode::HEURISTIC_B)
    };
};

// global_registry.cpp
GlobalRegistry& GlobalRegistry::conv_search_mode(ConvSearchMode mode) {
    fixed_conv_search_mode_.store(static_cast<int>(mode));
    return *this;
}
ConvSearchMode GlobalRegistry::conv_search_mode() const {
    return static_cast<ConvSearchMode>(fixed_conv_search_mode_.load());
}
```

遵循 fixed 型变量规则：允许幂等赋值，禁止非幂等修改。

### 8.4 `src/backend/op_stream_policy.cpp`

```cpp
case ComputeOp::CONV_FP32_INF:
case ComputeOp::CONV_AMP_INF:
    return StreamKind::COMP_1;
```

### 8.5 `src/backend/op_registry.cpp`

`require_warmup()` 新增 `CONV_FP32_INF`、`CONV_AMP_INF`。`warmup_single_cudnn_op()` 扩展为统一处理 FWD/INF/AMP 变体。

---

## 九、权重初始化

权重初始化由 `Initializer::derive(Region)` 和 `Initializer::apply_to_tensor()` 统一处理，算子层无需改动：

| 区域 | 初始化方式 |
|------|-----------|
| W_FIRST_CONV / W_DEEP_CONV | TRUNC_NORMAL |
| A_FIRST_CONV / A_DEEP_CONV | TRUNC_NORMAL |
| G_FIRST_CONV / G_DEEP_CONV | ZEROS |
| G_FIRST_CONV_FP16 / G_DEEP_CONV_FP16 | ZEROS |

---

## 十、文件变更清单

| # | 文件 | 变更类型 | 说明 |
|---|------|----------|------|
| 1 | `include/renaissance/graph/op_kind.h` | 修改 | 新增 `CONV_FP32_INF`、`CONV_AMP_INF` |
| 2 | `src/graph/op_kind.cpp` | 修改 | `compute_op_to_string()` + `format_params()` 补充 |
| 3 | `include/renaissance/core/types.h` | 修改 | 新增 `ConvSearchMode` 枚举 |
| 4 | `include/renaissance/core/global_registry.h` | 修改 | 新增 `fixed_conv_search_mode_` + setter/getter |
| 5 | `src/core/global_registry.cpp` | 修改 | 实现 setter/getter（幂等检查） |
| 6 | `src/backend/ops/dtensor/conv_op.cpp` | **重写** | 6 个 launch 函数 + CPU kernels + 注册入口 |
| 7 | `src/backend/ops/dtensor/conv_op_impl.cpp` | **重写** | 所有 cuDNN FE Graph 构建函数（FP32 + AMP） |
| 8 | `src/backend/op_registry.cpp` | 修改 | `require_warmup()` 加 INF；warmup 路径扩展 |
| 9 | `src/backend/op_stream_policy.cpp` | 修改 | 新增 INF 默认流 = COMP_1 |
| 10 | `src/graph/layer_descriptor_registry.cpp` | 修改 | `infer_conv_tensors` 加 bn_stats；`build_conv_*` 修正 ComputeOp 选择 |

---

## 十一、实施优先级

### Phase 1: 基础设施（无风险，先合入）

1. `op_kind.h`：新增 `CONV_FP32_INF`、`CONV_AMP_INF`
2. `op_kind.cpp`：字符串化补充
3. `types.h`：新增 `ConvSearchMode` 枚举
4. `global_registry.h/cpp`：新增 `conv_search_mode` setter/getter
5. `op_stream_policy.cpp`：新增 INF 流映射
6. `op_registry.cpp`：`require_warmup` 加 INF

### Phase 2: Layer Descriptor 修正（低风险）

7. `layer_descriptor_registry.cpp`：`infer_conv_tensors` 加 bn_stats；`build_conv_*` 修正

### Phase 3: FP32 完整实现（中风险）

8. `conv_op_impl.cpp`：重写 `build_conv_fwd_graph`（修正 Y stride 为 DTensor 真实值，添加 per-shape cache）
9. `conv_op_impl.cpp`：拆分 `build_conv_bwd_graph` 为 `build_conv_wgrad_graph` + `build_conv_dgrad_graph`
10. `conv_op.cpp`：重写 `launch_conv_fwd_cuda`（workspace 复用 + cache）、新增 `launch_conv_fp32_inf_cuda`
11. `conv_op.cpp`：重写 `launch_conv_bwd_cuda`（双流 + 跨流同步 + workspace 复用）
12. `conv_op.cpp`：`register_op_conv()` 注册 CONV_FP32_FWD/BWD/INF

### Phase 4: AMP 实现（高风险，需仔细验证）

13. `conv_op_impl.cpp`：新增 AMP FWD/INF Graph builders（Conv + GenStats / Conv only）
14. `conv_op_impl.cpp`：新增 AMP BWD Graph builders（wgrad FP16 + dgrad FP16）
15. `conv_op.cpp`：新增 AMP FWD/BWD/INF launch
16. `conv_op.cpp`：完善 `register_op_conv()`，注册全部 6 个算子

### Phase 5: EXHAUSTIVE_C 与优化

17. EXHAUSTIVE_C 搜索集成（仅 AMP Conv，A100/RTX5090）
18. warmup 路径扩展覆盖 INF 和 AMP 变体

---

## 十二、关键风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **dX 覆盖 X 数据竞争** | 训练结果完全错误 | BWD 严格 `wgrad(COMP_1) → event → dgrad(COMP_3)`；单流内 cuDNN FE execute 顺序执行 |
| **padded_c vs c 混用** | cuDNN FE 找不到执行计划（error code 8） | AMP 模式所有 `set_dim` 必须用 `dt.padded_c()`；FP32 用 `dt.c()` |
| **Stride 手写错误** | 静默数据错乱 | 强制使用 `dt.n_stride_cuda()` 系列，禁止手算 `align_up` |
| **KRSC 布局误解** | 权重读写错位 | 权重 dim 为 `{K, C, R, S}`（cuDNN FE），stride 通过 DTensor 真实值隐式表达 KRSC 物理布局 |
| **BN Stats 指针偏移** | GenStats 写错位置 | `sum` → `bn_stats_ptr`；`sq_sum` → `bn_stats_ptr + K * sizeof(float)` |
| **Workspace 越界/泄漏** | 随机崩溃或 OOM | 禁止 `cudaMalloc/Free`；统一用 `ctx.ensure_workspace_grow()` |
| **INF 与 FWD Graph 混用** | 意外触发 GenStats（INF 不应有） | AMP INF 独立构建无 GenStats 的 graph；Cache key 包含 `op` 字段 |
| **CPU BWD 性能** | CPU 训练极慢 | 当前 naive 实现仅保证 correctness；生产训练以 GPU 为主 |

---

## 十三、参考代码索引

| 功能 | 参考文件 | 关键函数/类 |
|------|----------|-------------|
| AMP FWD Graph (Conv+GenStats) | `cbr_fwd_fp16.cpp` | `build_conv_genstats_graph()` |
| AMP BWD Graph (WGrad/DGrad) | `cbr_bwd_fp16.cpp` | `build_conv_wgrad_graph()` / `build_conv_dgrad_graph()` |
| AMP INF Graph | `cbr_inf_fp16.cpp` | `try_build_single_graph()` |
| FE Graph Cache + Workspace | `fc_op.cpp` | `FcAmpFwdCache`、`launch_fc_amp_fwd_cuda()` |
| 多流 BWD 同步 | `fc_op.cpp` | `launch_fc_amp_bwd_cuda()` |
| FP32 Graph 构建（需重构） | `conv_op_impl.cpp` | `build_conv_fwd_graph()` / `build_conv_bwd_graph()` |
| AMP padded_c 使用 | `maxpool_op.cpp` | MaxPool AMP FWD/INF 实现 |
| CPU XNNPACK FWD | `conv_op.cpp` | `launch_conv_fwd_cpu_xnnpack()` |
| CPU Naive BWD | `conv_op.cpp` | `launch_conv_bwd_cpu()` |
| 算子注册 | `fc_op.cpp` | `register_op_fc()` |
| Experience 查询 | `ta_v4_common_fp16.hpp` | `build_shape_key()`、`match_and_build_plan()` |
| Layer Descriptor 模式 | `layer_descriptor_registry.cpp` | FC/BN/ReLU 的 `build_*` 模式 |
| cuDNN FE 公共辅助 | `cudnn_utils.h` | `create_cudnn_graph()`、`finalize_cudnn_graph()` |

---

*本方案综合了 NDV.md 中 4 位小伙伴（S/K/C/D）的分析视角，去重合并后形成统一实施路线。方案与框架现有基础设施（DTensor 对齐、Region 体系、MultiStreamCaptureState、GlobalRegistry fixed 变量体系、Initializer 初始化链）完全兼容。*