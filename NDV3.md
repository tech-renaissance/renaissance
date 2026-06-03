# 卷积六大算子实现方案（NDV3）

> 版本: 1.0.0  
> 日期: 2026-06-02  
> 范围: CONV_FP32_FWD / CONV_FP32_BWD / CONV_FP32_INF / CONV_AMP_FWD / CONV_AMP_BWD / CONV_AMP_INF  
> 目标平台: Windows MSVC + CUDA 13.1 + cuDNN 9.17

---

## 1. 概述

本方案基于 NDV.md（多轮方案讨论）及全量代码调研，制定六大卷积算子的完整实现路径。核心目标：

1. **补全缺失算子**：`CONV_FP32_INF`、`CONV_AMP_INF` 枚举缺失，AMP FWD/BWD launch 缺失。
2. **修正现有缺陷**：FP32 BWD 单流单图需拆为双流；workspace 禁止 `cudaMalloc`；stride 禁止手写。
3. **新增 bn_stats**：`infer_conv_tensors` 从 6 张量扩展为 7 张量。
4. **EXHAUSTIVE_C**：为 AMP Conv 接入平台特化的穷举搜索。

---

## 2. 现状诊断（基于代码实际状态）

### 2.1 算子状态

| 算子 | 状态 | 说明 |
|------|------|------|
| `CONV_FP32_FWD` | 已注册 | `launch_conv_fwd_cuda` → `build_conv_fwd_graph`，但 Y stride 手写 `align_up_c(K)`，workspace 每轮 `cudaMalloc/Free` |
| `CONV_FP32_BWD` | 已注册 | `launch_conv_bwd_cuda` → `build_conv_bwd_graph` 单图单流（COMP_1），未拆双图；input 映射存在隐患（见 2.2） |
| `CONV_FP32_INF` | **缺失** | `ComputeOp` 枚举无定义 |
| `CONV_AMP_FWD` | 半残 | 枚举已有，但无 launch 实现；`conv_op_impl.cpp` 中有旧 AMP graph builder（硬编码 FLOAT，未接入） |
| `CONV_AMP_BWD` | 半残 | 同上 |
| `CONV_AMP_INF` | **缺失** | `ComputeOp` 枚举无定义 |

### 2.2 关键缺陷

1. **BWD 未拆双图**：现有 `build_conv_bwd_graph` 将 `conv_dgrad` + `conv_wgrad` 放在**同一 Graph**，但 cuDNN FE Graph 的 `execute()` 绑定到单个 stream。框架要求 dW@COMP_1 → event → dX@COMP_3，**单图无法满足跨流需求**。
2. **Workspace 临时分配**：`conv_op.cpp` FWD/BWD 每次 launch 都 `cudaMalloc/Free` workspace。必须改为 `ctx.ensure_workspace_grow()` 复用 DeviceContext 的 per-stream workspace。
3. **输出 stride 手写**：`conv_op_impl.cpp` 中 Y/dX/dW 的 stride 手写 `align_up_c(K)` 等，而非使用 DTensor 真实 stride。这会导致 AMP 模式下（`padded_c ≠ c`）数据错位。
4. **缺少 bn_stats**：`infer_conv_tensors` 返回 6 张量，缺第 7 个 bn_stats。`build_conv_forward/backward/inference` 硬编码 `CONV_AMP_FWD/BWD`，未根据 `using_amp()` 动态选择。
5. **BWD input 映射隐患**：现有 `build_conv_backward` 的 `input_indices = {0,1,2}`（weight, output, grad_slot）。Compiler Phase 2 插入 dY 到开头后，`node.input_ids = [dY, weight, output, grad_slot]`。但 `launch_conv_bwd_cuda` 只检查 `size >= 3` 并取 `[0]=dY, [1]=X?, [2]=W?`，实际 `[1]=weight, [2]=output`，**命名与语义错位**。且 BWD 数学上不需要 output(Y)，只需 dY/W/X。
6. **warmup 路径缺失**：`op_registry.cpp` 的 `warmup_single_cudnn_op()` 只处理 FWD，未覆盖 INF 和 AMP 变体。

---

## 3. 核心设计原则

### 3.1 参数约束（BluePrint/ArchPlan 层已保证，算子内不检查）

| 参数 | 约束 |
|------|------|
| `kernel_h / kernel_w` | 仅 1, 3, 5, 7（正方形） |
| `stride_h / stride_w` | 仅 1 或 2 |
| `dilation` | 固定 1 |
| `groups` | 固定 1（普通卷积，无 Depthwise） |
| `bias` | **不支持**（无 bias 分区） |
| `padding` | 无限制 |

### 3.2 框架铁律

- **dX 覆盖 X**：BWD 梯度输出覆盖原始输入，共用同一 DTensor。
- **先 dW 后 dX**：BWD 必须先完成 wgrad（COMP_1），再启动 dgrad（COMP_3）。`cudaStreamWaitEvent(s_dx, dW_event, 0)`。
- **张量对齐**：同一层输入输出张量的数量、种类、形状在 CPU/GPU/FP32/AMP 下完全一致。
- **禁止算子内临时内存分配**：workspace 用 `ctx.ensure_workspace_grow()`；DTensor 在 Compiler 阶段申请。
- **NHWC 物理布局**：cuDNN FE `set_stride({N_stride, C_stride, H_stride, W_stride})`，`C_stride=1` 表示 NHWC。
- **Stride 必须用 DTensor 真实值**：禁止手写 `align_up`。

### 3.3 AMP  vs FP32 的维度差异

| 场景 | `set_dim` 的 C 维度 | Stride |
|------|---------------------|--------|
| FP32 | `dt.c()`（无 padding） | `dt.n_stride_cuda()` 等真实值 |
| AMP | `dt.padded_c()`（对齐到 8） | `dt.n_stride_cuda()` 等真实值 |

> ⚠️ `to_fe_stride_nhwc()` 基于紧凑 shape + alignment 计算，**与 DTensor 真实 CUDA stride 不同**，不可用于 Graph 构建。

---

## 4. 张量布局（扩展后：7 张量）

### 4.1 Conv 层张量清单

| 索引 | 名称 | Shape (NHWC) | Region | DType | 说明 |
|------|------|-------------|--------|-------|------|
| 0 | `conv_weight` | `{K,R,S,C}` | W_FIRST_CONV / W_DEEP_CONV | FP32 | 主权重 |
| 1 | `conv_output` | `{N,OH,OW,K}` | F_FEATURE_FP32/FP16 | varies | 输出特征图 |
| 2 | `conv_grad_slot` | `{N,H,W,C}` | F_GRAD_SLOT_FP32/FP16 | varies | 梯度槽（dX in-place 覆盖 X） |
| 3 | `conv_weight_grad` | `{K,R,S,C}` | G_FIRST_CONV / G_DEEP_CONV | FP32 | 权重梯度（FP32 区） |
| 4 | `conv_amp_w_fp16` | `{K,R,S,C}` | A_FIRST_CONV / A_DEEP_CONV | FP16 | AMP 工作权重（!AMP 时占位） |
| 5 | `conv_amp_g_fp16` | `{K,R,S,C}` | G_*_CONV_FP16 | FP16 | AMP 权重梯度（BWD 直接产 FP16） |
| **6** | **`conv_bn_stats`** | **`{1,1,1,2*K}`** | **T_TEMP_FP32** | **FP32** | **BN 统计量（新增）** |

### 4.2 bn_stats 规格

- **大小**：`2 * K` 个 float，前半段存 `sum`，后半段存 `sq_sum`。
- **区域**：`T_TEMP_FP32`（`cuda_alignment = 1`，不使用 padding）。
- **写入情况**：仅 `CONV_AMP_FWD` 通过 GenStats 写入；其余 5 个算子**不写入、不 memset**。
- **Graph 绑定**：AMP FWD 的 GenStats 输出两个 `{1,K,1,1}` 张量，`sum` 指向 bn_stats 首地址，`sq_sum` 指向 `bn_stats_ptr + K * sizeof(float)`。

### 4.3 SubgraphPattern 修正

```cpp
// build_conv_forward
bool amp = GlobalRegistry::instance().using_amp();
SubgraphPattern::Node n;
n.op = amp ? ComputeOp::CONV_AMP_FWD : ComputeOp::CONV_FP32_FWD;
n.input_indices  = amp ? std::vector<size_t>{4, 6} : std::vector<size_t>{0, 6};
//   [weight, bn_stats]；X 由 Compiler 跨层链注入 input_ids[0]
n.output_indices = {1, 6};   // output + bn_stats

// build_conv_backward
n.op = amp ? ComputeOp::CONV_AMP_BWD : ComputeOp::CONV_FP32_BWD;
n.input_indices  = amp ? std::vector<size_t>{4, 2} : std::vector<size_t>{0, 2};
//   [weight, grad_slot(=X)]；dY 由 Compiler 跨层链注入 input_ids[0]
//   注：BWD 不需要 output(Y)，cuDNN conv_bwd 只需 dY/W/X
n.output_indices = amp ? std::vector<size_t>{2, 5} : std::vector<size_t>{2, 3};
//   dX(in-place to grad_slot), dW

// build_conv_inference
n.op = amp ? ComputeOp::CONV_AMP_INF : ComputeOp::CONV_FP32_INF;
n.input_indices  = amp ? std::vector<size_t>{4, 6} : std::vector<size_t>{0, 6};
n.output_indices = {1};
```

### 4.4 各算子 I/O 速查表（Compiler 构建后的 GraphNode）

| 算子 | node.input_ids | node.output_ids | 说明 |
|------|---------------|-----------------|------|
| `CONV_FP32_FWD` | `[0]=X, [1]=W(0), [2]=bn_stats(6)` | `[0]=Y(1), [1]=bn_stats(6)` | bn_stats 占位 |
| `CONV_FP32_INF` | `[0]=X, [1]=W(0), [2]=bn_stats(6)` | `[0]=Y(1)` | 同 FWD，无 bn_stats 输出 |
| `CONV_FP32_BWD` | `[0]=dY, [1]=W(0), [2]=X(2)` | `[0]=dX(2), [1]=dW(3)` | dX in-place |
| `CONV_AMP_FWD` | `[0]=X, [1]=W_fp16(4), [2]=bn_stats(6)` | `[0]=Y(1), [1]=bn_stats(6)` | GenStats 写 bn_stats |
| `CONV_AMP_INF` | `[0]=X, [1]=W_fp16(4), [2]=bn_stats(6)` | `[0]=Y(1)` | 无 GenStats |
| `CONV_AMP_BWD` | `[0]=dY, [1]=W_fp16(4), [2]=X(2)` | `[0]=dX(2), [1]=dW_fp16(5)` | dW 输出 FP16 |

---

## 5. CUDA 实现方案

### 5.1 通用设计：Per-Shape Graph Cache

参考 `fc_op.cpp` 的 `FcAmpFwdCache`，为 Conv 所有算子构建 shape-keyed cache。

```cpp
struct ConvGraphCacheKey {
    uint64_t handle_bits;
    int32_t  N, H, W, C, K, R, S;
    int32_t  pad, stride;
    bool     is_amp;
    ComputeOp op;   // 区分 FWD/INF/BWD-wgrad/BWD-dgrad

    bool operator==(const ConvGraphCacheKey& o) const { ... }
};

struct ConvGraphCacheKeyHasher {
    size_t operator()(const ConvGraphCacheKey& k) const { ... }
};

struct ConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size;
};

// 独立缓存：FWD / INF / BWD-wgrad / BWD-dgrad
static std::unordered_map<ConvGraphCacheKey, ConvGraphCache, ConvGraphCacheKeyHasher>
    s_conv_fwd_caches, s_conv_inf_caches,
    s_conv_wgrad_caches, s_conv_dgrad_caches;
```

Cache 在 warmup 阶段预热（`warmup_single_cudnn_op` 扩展）。命中时复用 Graph + 重新填充 Variant Pack。

**Workspace 管理铁律**：
```cpp
void* ws = ctx.ensure_workspace_grow(stream_kind, cache.workspace_size);
// 禁止 cudaMalloc / cudaFree
```

### 5.2 NHWC ↔ cuDNN FE 的 stride 映射

```cpp
auto make_nhwc_stride = [](const DTensor& dt) -> std::vector<int64_t> {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
};

// 特征图（X/Y/dY/dX）：dim={N, C, H, W}，stride 真实值
// 权重（W/dW）：dim={K, C, R, S}，stride 真实值
//   (dt.shape 为 {K,R,S,C}，n_stride=R*S*C, c_stride=1, h_stride=S*C, w_stride=C)
```

### 5.3 Graph 构建函数签名重构

现有 `build_conv_fwd_graph` 签名复杂且不接收输出 DTensor。新签名直接传入关键 DTensor：

```cpp
// FP32 / AMP 共用 FWD/INF Graph（通过 dtype 区分）
ConvGraphCache build_conv_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const ConvParams& params, cudnnHandle_t handle, DType dtype);

// BWD 拆分为两个独立 Graph
ConvGraphCache build_conv_wgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_x, const DTensor& dt_dw,
    const ConvParams& params, cudnnHandle_t handle, DType dtype);

ConvGraphCache build_conv_dgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_w, const DTensor& dt_dx,
    const ConvParams& params, cudnnHandle_t handle, DType dtype);

// AMP FWD 专用（Conv + GenStats）
ConvGraphCache build_conv_amp_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const DTensor& dt_bn_stats,
    const ConvParams& params, cudnnHandle_t handle);
```

### 5.4 FP32 FWD / INF（CONV_FP32_FWD、CONV_FP32_INF）

**Graph 构建**：
```cpp
auto graph = create_cudnn_graph(DType::FP32);
auto X = graph->tensor(... FLOAT, make_nhwc_stride(dt_x) ...);
auto W = graph->tensor(... FLOAT, make_nhwc_stride(dt_w) ...);
auto Y = graph->conv_fprop(X, W, conv_opts);
Y->set_output(true)
  .set_dim(to_fe_dim(dt_y.shape))   // {N, K, OH, OW}
  .set_stride(make_nhwc_stride(dt_y))
  .set_data_type(DataType_t::FLOAT);
finalize_cudnn_graph(graph.get(), handle);
```

**Launch 执行**：
```cpp
cudaStream_t s = ctx.stream(StreamKind::COMP_1);
cudnnHandle_t h = ctx.cudnn_handle(StreamKind::COMP_1);
int si = state.get_or_register(s);

// 等待前序
if (state.output_stream_idx >= 0) {
    cudaStreamWaitEvent(s, state.streams[state.output_stream_idx].last_done_event, 0);
}

// Cache lookup / build
auto& cache = get_or_build_cache(key, h, ...);
ctx.ensure_workspace_grow(StreamKind::COMP_1, cache.workspace_size);

// Variant Pack
std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
for (const auto& [tensor, id] : cache.tensor_to_id) {
    vp[tensor] = (id >= 0) ? ctx.ptr_at(id) : nullptr;
}

cache.graph->execute(h, vp, ctx.workspace(StreamKind::COMP_1));

state.output_stream_idx = si;
state.streams[si].has_pending_work = true;
cudaEventRecord(state.streams[si].last_done_event, s);
```

- INF 与 FWD **共用同一个 graph builder 和 cache**（key 中 `op` 字段区分，或 INF 直接查 FWD cache）。
- bn_stats 不在 Graph 中定义，但作为 `node.output_ids[1]` 存在（AMP FWD 需要）。

### 5.5 FP32 BWD（CONV_FP32_BWD）— 拆双图 + 跨流同步

**WGrad Graph（COMP_1）**：
```cpp
auto graph = create_cudnn_graph(DType::FP32);
auto dY = graph->tensor(... FLOAT ...);
auto X  = graph->tensor(... FLOAT ...);
auto dW = graph->conv_wgrad(dY, X, wgrad_opts);
dW->set_output(true)
   .set_dim(to_fe_dim(dt_dw.shape))
   .set_stride(make_nhwc_stride(dt_dw))
   .set_data_type(DataType_t::FLOAT);
finalize_cudnn_graph(graph.get(), handle);
```

**DGrad Graph（COMP_3）**：
```cpp
auto graph = create_cudnn_graph(DType::FP32);
auto dY = graph->tensor(... FLOAT ...);
auto W  = graph->tensor(... FLOAT ...);
auto dX = graph->conv_dgrad(dY, W, dgrad_opts);
dX->set_output(true)
   .set_dim(to_fe_dim(dt_dx.shape))
   .set_stride(make_nhwc_stride(dt_dx))   // 必须与 X 完全一致
   .set_data_type(DataType_t::FLOAT);
finalize_cudnn_graph(graph.get(), handle);
```

**Launch 同步代码**：
```cpp
cudaStream_t s_dw = ctx.stream(StreamKind::COMP_1);
cudaStream_t s_dx = ctx.stream(StreamKind::COMP_3);
cudnnHandle_t h_dw = ctx.cudnn_handle(StreamKind::COMP_1);
cudnnHandle_t h_dx = ctx.cudnn_handle(StreamKind::COMP_3);
int i_dw = state.get_or_register(s_dw);
int i_dx = state.get_or_register(s_dx);

// 等待前序（两流都 wait）
int out_idx = state.output_stream_idx;
if (out_idx >= 0) {
    cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
    cudaStreamWaitEvent(s_dx, state.streams[out_idx].last_done_event, 0);
}

// COMP_1: wgrad
auto& cache_w = get_or_build_wgrad_cache(...);
ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);
cache_w.graph->execute(h_dw, vp_w, ctx.workspace(StreamKind::COMP_1));
cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
state.streams[i_dw].has_pending_work = true;

// COMP_3: dgrad（等待 wgrad 完成，保护 X）
cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);
auto& cache_x = get_or_build_dgrad_cache(...);
ctx.ensure_workspace_grow(StreamKind::COMP_3, cache_x.workspace_size);
cache_x.graph->execute(h_dx, vp_x, ctx.workspace(StreamKind::COMP_3));
cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
state.streams[i_dx].has_pending_work = true;

state.output_stream_idx = i_dx;
```

### 5.6 AMP FWD（CONV_AMP_FWD）— Conv + GenStats

**Graph 构建**（参考 `cbr_fwd_fp16.cpp`）：
```cpp
auto graph = create_cudnn_graph(DType::FP16);  // I/O = HALF

auto X = graph->tensor(... HALF, make_nhwc_stride(dt_x) ...);
auto W = graph->tensor(... HALF, make_nhwc_stride(dt_w) ...);

auto conv_out = graph->conv_fprop(X, W, conv_opts);
conv_out->set_is_virtual(true).set_data_type(DataType_t::FLOAT);

auto [sum, sq_sum] = graph->genstats(conv_out, Genstats_attributes()
    .set_compute_data_type(DataType_t::FLOAT));

// Y 输出（conv_out 本身）
conv_out->set_output(true)
        ->set_dim(to_fe_dim(dt_y.shape))
        ->set_stride(make_nhwc_stride(dt_y))
        ->set_data_type(DataType_t::HALF);

// sum / sq_sum 输出到 bn_stats
sum->set_output(true)
    ->set_dim({1, K, 1, 1})
    ->set_stride({K, 1, K, K})
    ->set_data_type(DataType_t::FLOAT);
sq_sum->set_output(true)
      ->set_dim({1, K, 1, 1})
      ->set_stride({K, 1, K, K})
      ->set_data_type(DataType_t::FLOAT);
```

**Variant Pack 中 bn_stats 映射**：
```cpp
float* bn_stats_ptr = static_cast<float*>(ctx.ptr_at(node.output_ids[1]));
vp[uid_sum]    = bn_stats_ptr;
vp[uid_sq_sum] = bn_stats_ptr + K;  // 偏移 K 个 float
```

**EXHAUSTIVE_C 分支**（AMP FWD 专用）：
```cpp
if (GlobalRegistry::instance().conv_search_mode() == ConvSearchMode::EXHAUSTIVE_C) {
#if defined(USING_A100) || defined(USING_RTX5090)
    std::string key = build_shape_key("conv_genstats", "fp16",
        N, H, W, C, K, R, S, stride, pad);
    auto exp_rec = ta_v4::experience::lookup(key);
    if (exp_rec != nullptr) {
        graph->create_execution_plans({HeurMode_t::A, HeurMode_t::B});
        auto [status, matched] = match_and_build_plan(
            graph, candidates, exp_rec, handle);
        if (matched) goto done;
    }
#endif
}
// fallback
finalize_cudnn_graph(graph.get(), handle);
done:
```

### 5.7 AMP INF（CONV_AMP_INF）

纯 `conv_fprop`，**无 GenStats**：
```cpp
auto graph = create_cudnn_graph(DType::FP16);
// X, W, Y 同 AMP FWD
finalize_cudnn_graph(graph.get(), handle);
```

- bn_stats 作为 `node.input_ids[2]` 传入算子，但 Graph 中不引用。
- 单流 COMP_1。

### 5.8 AMP BWD（CONV_AMP_BWD）— 双图 + 跨流同步

与 FP32 BWD **完全相同的流架构**，仅数据类型为 HALF，且 dW 输出 FP16：

**WGrad Graph（COMP_1）**：
```cpp
auto graph = create_cudnn_graph(DType::FP16);
// dY(HALF), X(HALF)
auto dW = graph->conv_wgrad(dY, X, wgrad_opts);
dW->set_output(true)
   ->set_dim(to_fe_dim(dt_dw.shape))
   ->set_stride(make_nhwc_stride(dt_dw))
   ->set_data_type(DataType_t::HALF);   // ★ AMP BWD 输出 FP16 dW
```

**DGrad Graph（COMP_3）**：
```cpp
auto graph = create_cudnn_graph(DType::FP16);
// dY(HALF), W(HALF)
auto dX = graph->conv_dgrad(dY, W, dgrad_opts);
dX->set_output(true)
   ->set_dim(to_fe_dim(dt_dx.shape))
   ->set_stride(make_nhwc_stride(dt_dx))
   ->set_data_type(DataType_t::HALF);
```

**流同步代码与 FP32 BWD 完全一致**（见 5.5）。

**EXHAUSTIVE_C**：WGrad 查 `"conv_wgrad"`，DGrad 查 `"conv_dgrad"`。

---

## 6. CPU 实现

### 6.1 FWD / INF

- 复用现有 `launch_conv_fwd_cpu_xnnpack`（XNNPACK NHWC）。
- `CONV_FP32_INF` 与 `CONV_FP32_FWD` 共用同一 CPU launch。
- AMP 算子 CPU 不支持：`launch_conv_amp_cpu_not_supported()`。

### 6.2 BWD

- 复用现有 `launch_conv_bwd_cpu_naive`（Naive 六重循环）。
- 先 memset dX=0、dW=0，再累加。
- 单线程顺序执行（先 dW 后 dX，天然满足约束）。

---

## 7. 枚举与注册表扩展

### 7.1 `include/renaissance/graph/op_kind.h`

```cpp
enum class ComputeOp : uint16_t {
    // ... 现有 ...
    CONV_FP32_FWD,
    CONV_FP32_BWD,
    CONV_FP32_INF,      // 新增
    CONV_AMP_FWD,
    CONV_AMP_BWD,
    CONV_AMP_INF,       // 新增
    // ...
};
```

### 7.2 `include/renaissance/core/types.h`

```cpp
enum class ConvSearchMode : uint8_t {
    HEURISTIC_B = 0,   // 默认
    EXHAUSTIVE_C = 1   // 仅 AMP Conv，仅 A100/RTX5090
};
```

### 7.3 `global_registry.h` + `global_registry.cpp`

```cpp
class GlobalRegistry {
public:
    GlobalRegistry& conv_search_mode(ConvSearchMode mode);
    ConvSearchMode conv_search_mode() const;
private:
    std::atomic<int> fixed_conv_search_mode_{0};  // 0 = HEURISTIC_B
};
```

实现遵循 fixed 型变量规则（幂等赋值允许，非幂等报错）。

### 7.4 `src/backend/op_stream_policy.cpp`

```cpp
case ComputeOp::CONV_FP32_INF:
case ComputeOp::CONV_AMP_INF:
    return StreamKind::COMP_1;
```

### 7.5 `src/backend/op_registry.cpp`

- `require_warmup()` 新增 `CONV_FP32_INF`、`CONV_AMP_INF`。
- `warmup_single_cudnn_op()`：
  - 扩展专用预热路径覆盖 FWD/INF（两者 Graph 结构相同）。
  - 移除 `cudaMalloc/Free`，改用 `ctx.ensure_workspace_grow()`。
  - BWD 的 warmup 走通用 `launch_cuda` 路径（双图会在 launch 中构建）。

---

## 8. Layer Descriptor 与 Compiler 修改

### 8.1 `src/graph/layer_descriptor_registry.cpp`

**`infer_conv_tensors`**：新增第 7 个张量 bn_stats：
```cpp
{ TensorDesc d;
  d.name = "conv_bn_stats";
  d.shape = Shape{1, 1, 1, cp.out_channels * 2};
  d.region = Region::T_TEMP_FP32;
  d.dtype = DType::FP32;
  descs.push_back(d);
}
```

**`build_conv_forward/backward/inference`**：修正 ComputeOp 选择和 I/O（见 4.3）。

### 8.2 `src/graph/compiler.cpp`

`alloc_conv_group` **无需**新增 bn_stats 分配逻辑——`infer_conv_tensors` 返回的 TensorDesc（Region::T_TEMP_FP32）已由 Compiler 统一处理。

---

## 9. 文件改动清单

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `include/renaissance/graph/op_kind.h` | 修改 | 新增 `CONV_FP32_INF`、`CONV_AMP_INF` |
| 2 | `src/graph/op_kind.cpp` | 修改 | `compute_op_to_string()` / `format_params()` 补充 |
| 3 | `include/renaissance/core/types.h` | 修改 | 新增 `ConvSearchMode` |
| 4 | `include/renaissance/core/global_registry.h` | 修改 | 新增 `fixed_conv_search_mode_` + setter/getter |
| 5 | `src/core/global_registry.cpp` | 修改 | 实现 setter/getter（幂等检查） |
| 6 | `src/backend/op_stream_policy.cpp` | 修改 | 新增 INF stream = COMP_1 |
| 7 | `src/backend/op_registry.cpp` | 修改 | `require_warmup()` 加 INF；warmup 路径扩展并修复 workspace |
| 8 | `src/graph/layer_descriptor_registry.cpp` | 修改 | 新增 bn_stats；修正 `build_conv_*` 的 I/O 和 ComputeOp |
| 9 | `src/backend/ops/dtensor/conv_op.cpp` | **重写** | 6 个 launch 函数 + CPU kernels + 注册入口 |
| 10 | `src/backend/ops/dtensor/conv_op_impl.cpp` | **重写** | 所有 cuDNN FE Graph builders（FP32 + AMP） |

---

## 10. 实施优先级

按以下顺序实现，每一步都可独立编译验证：

### Phase 1: 基础设施（无风险）
1. `op_kind.h`：新增 `CONV_FP32_INF`、`CONV_AMP_INF`
2. `types.h`：新增 `ConvSearchMode`
3. `global_registry.h/cpp`：新增 `conv_search_mode`
4. `op_stream_policy.cpp`：新增 INF 流策略
5. `op_registry.cpp`：`require_warmup()` 扩展

### Phase 2: Layer Descriptor（低风险）
6. `layer_descriptor_registry.cpp`：
   - `infer_conv_tensors` 加 bn_stats（第 7 张量）
   - `build_conv_*` 修正 ComputeOp 选择、I/O、input_indices

### Phase 3: FP32 INF（低风险）
7. 实现 `launch_conv_fp32_inf_cuda`（复用 FWD graph builder，仅枚举不同）
8. CPU INF 共用 FWD 函数
9. 注册到 `g_compute_op_table`

### Phase 4: FP32 BWD 双流改造（中风险）
10. 将 `build_conv_bwd_graph` 拆为 `build_conv_wgrad_graph` + `build_conv_dgrad_graph`
11. `launch_conv_bwd_cuda` 改为双流 + event 同步
12. 添加 per-shape cache + workspace 复用
13. 修正 BWD 的 input 映射（dY/W/X）

### Phase 5: FP32 FWD 改造（低风险）
14. 修复 workspace 管理（`cudaMalloc` → `ensure_workspace_grow`）
15. 添加 per-shape cache，修正 Y stride 为 DTensor 真实值

### Phase 6: AMP FWD（中风险）
16. 实现 `build_conv_amp_fwd_graph`（Conv+GenStats，HALF I/O）
17. 实现 `launch_conv_amp_fwd_cuda`
18. bn_stats Variant Pack 映射（sum/sq_sum 分半区）
19. 使用 `padded_c()` 设置 dim

### Phase 7: AMP INF（低风险）
20. 纯 `conv_fprop` HALF，无 GenStats
21. 复用 AMP FWD 的大部分逻辑

### Phase 8: AMP BWD（高风险）
22. 实现 `build_conv_amp_wgrad_graph` + `build_conv_amp_dgrad_graph`
23. 流同步与 FP32 BWD 一致
24. dW 输出 FP16（HALF），写入 `G_*_CONV_FP16`
25. EXHAUSTIVE_C 分别查询 wgrad/dgrad

### Phase 9: 统一注册与验证
26. 完善 `register_op_conv()`，注册全部 6 个算子
27. 编译并通过现有测试（确保 FP32 FWD/BWD 不被破坏）

---

## 11. 风险与缓解

| 风险点 | 影响 | 缓解措施 |
|--------|------|----------|
| **dX 覆盖 X 数据竞争** | 训练结果完全错误 | BWD 严格拆双图：`wgrad(COMP_1) → event → dgrad(COMP_3)` |
| **padded_c vs c 混用** | cuDNN FE 找不到执行计划 | AMP 所有 `set_dim` 用 `dt.padded_c()`；FP32 用 `dt.c()` |
| **Stride 手写错误** | 静默数据错乱 | 强制使用 `make_nhwc_stride(dt)`，禁止手算 stride |
| **BN Stats 指针偏移** | GenStats 写错位置 | `sum` → `bn_stats_ptr`；`sq_sum` → `bn_stats_ptr + K * sizeof(float)` |
| **Workspace 泄漏/每轮分配** | 随机崩溃或 OOM | 统一 `ctx.ensure_workspace_grow()`；禁止 `cudaMalloc/Free` |
| **INF 与 FWD Graph 混用** | INF 意外触发 GenStats | Cache key 包含 `op` 字段；INF 独立查 FWD cache 但构建时不加 GenStats |
| **CPU BWD 性能** | CPU 训练极慢 | 当前 naive 实现仅保证 correctness；生产训练以 GPU 为主 |
| **Layer Descriptor 张量数不一致** | MemoryPlan 错位 | 确保 `infer_conv_tensors` 返回 7 个；所有 `build_conv_*` 的 `descs.size()` 检查从 `< 6` 改为 `< 7` |
| **Compiler BWD input 映射** | 现有 BWD input 语义错位 | 修正 `build_conv_backward` 的 `input_indices` 为 `{weight, grad_slot}`，去掉不需要的 output |

---

## 12. 参考代码索引

| 功能 | 参考文件 | 关键函数/类 |
|------|---------|------------|
| AMP FWD Graph (Conv+GenStats) | `cbr_fwd_fp16.cpp` | `build_conv_genstats_graph()` |
| AMP BWD Graph (WGrad/DGrad) | `cbr_bwd_fp16.cpp` | `build_conv_wgrad_graph()` / `build_conv_dgrad_graph()` |
| AMP INF Graph | `cbr_inf_fp16.cpp` | `try_build_single_graph()` |
| FE Graph Cache + Workspace | `fc_op.cpp` | `FcAmpFwdCache`、`launch_fc_amp_fwd_cuda()` |
| 多流 BWD 同步 | `fc_op.cpp` | `launch_fc_amp_bwd_cuda()` |
| FP32 Graph 构建 | `conv_op_impl.cpp` | 现有 `build_conv_fwd_graph()` / `build_conv_bwd_graph()`（需重构） |
| CPU XNNPACK FWD | `conv_op.cpp` | `launch_conv_fwd_cpu_xnnpack()` |
| CPU Naive BWD | `conv_op.cpp` | `launch_conv_bwd_cpu()` |
| 算子注册 | `fc_op.cpp` | `register_op_fc()` |
| FE 公共辅助 | `cudnn_utils.h` | `create_cudnn_graph()`、`finalize_cudnn_graph()`、`to_fe_dim()` |

---

*本方案已与框架现有基础设施（DTensor 对齐、Region 体系、MultiStreamCaptureState、GlobalRegistry fixed 变量体系）完全兼容，可直接进入编码阶段。*
