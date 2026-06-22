# ETK2：BN 推理 eq_scale/eq_bias 预计算最终方案

> **版本**: 2.0  
> **日期**: 2026-06-22  
> **目标**: 消除 BN INF（含 CBR INF 的 BN 段）每次推理时重复计算 `eq_scale`/`eq_bias` 的开销，改为每 epoch 训练结束后一次性预计算并写入张量，后续 INF 直接读取。  
> **约束**: 新增 DTensor 算子**不得在算子内申请任何额外内存**，只能使用输入/输出张量。  
> **结果**: 数值与当前代码完全一致，推理侧计算量减少、内存带宽降低。

---

## 1. 现状与问题（基于代码审计）

### 1.1 BN 张量布局

标准 BN 层在 `layer_descriptor_registry.cpp:185-235` 中分配 13 张量：

| 局部索引 | 名称 | Region | 说明 |
|:---:|:---|:---|:---|
| 0 | weight (gamma) | `W_BN_WEIGHT` | 可训练权重 |
| 1 | bias (beta) | `W_BN_BIAS` | 可训练偏置 |
| 2 | output | 特征区 | 前向/推理输出 |
| 3 | prev_mean | `B_PREV_MEAN` | 上一轮 running mean |
| 4 | prev_var | `B_PREV_VAR` | 上一轮 running var |
| 5 | next_mean | `B_NEXT_MEAN` | 当前 running mean |
| 6 | next_var | `B_NEXT_VAR` | 当前 running var |
| 7 | weight_grad | `G_BN_WEIGHT` | 权重梯度 |
| 8 | bias_grad | `G_BN_BIAS` | 偏置梯度 |
| 9 | eq_bias | `W_EQ_BIAS` | **推理等效 bias** |
| 10 | eq_scale | `W_EQ_SCALE` | **推理等效 scale** |
| 11 | saved_mean | `T_TEMP_FP32` | FWD→BWD 桥接 |
| 12 | saved_inv_var | `T_TEMP_FP32` | FWD→BWD 桥接 |

CBR 层（23 张量）中 BN 部分在索引 8-20，相对偏移 `+8`：
- `bn_weight = 8`, `bn_bias = 9`
- `next_mean = 13`, `next_var = 14`
- `eq_bias = 17`, `eq_scale = 18`

### 1.2 当前 INF 行为

| 路径 | 位置 | 行为 |
|:---|:---|:---|
| CPU BN INF | `bn_op.cpp:171-213` | 8 输入 `[X, eq_scale, eq_bias, gamma, beta, running_mean, running_var, eps]`，每次 INF 重新计算 `eq_scale`/`eq_bias` 写入 `W_EQ_*`，再读回做 affine。 |
| GPU BN INF | `bn_op.cpp:589-644` → `bn_op.cu:60-109` | 8 输入同上，但融合 kernel `tr_bn_inf_fp32/fp16_kernel` 每个线程重新计算 `scale/bias`，**完全不读写** `W_EQ_*` 张量。 |
| GPU CBR INF | `cbr_op.cpp:1093-1224` | 9 输入含 `eq_scale/eq_bias`，但 BN 段同样调用 `launch_tr_bn_inf_kernel` 临时计算，不读 `eq_scale/eq_bias`。 |

### 1.3 已有可用 kernel

`bn_op.cu:20-37` 的 `tr_bn_compute_eq_params_kernel` 已按 channel 并行实现：

```cuda
inv_std = 1.0f / sqrtf(running_var[c] + eps);
eq_scale[c] = gamma[c] * inv_std;
eq_bias[c]  = beta[c] - running_mean[c] * eq_scale[c];
```

该 kernel 目前**未被任何代码调用**，可直接复用于 `BN_UPDATE_EQ_PARAMS` 的 GPU 路径。

---

## 2. 设计决策

### 2.1 调度位置：放在 `run_train_epoch_gpu/cpu()` 末尾

- **GPU**: 在 last batch 结束后、函数返回前，于每个 rank 线程内 launch `UPDATE_BN_INF_PARAMS`。此时 `OPTIMIZER`、`CAST_MAIN_FP32_TO_FP16` 已完成，`B_NEXT_MEAN/B_NEXT_VAR` 已通过 `STATS_COMM` 同步为最终值。
- **CPU**: 放在 `run_train_epoch_cpu()` 函数最末尾（所有 batch 处理完、`active_memory_plan_` 已恢复后、计算 `train_loss` 前）。这同时覆盖 `batches == 1` 的早返回路径。

**为什么不放在 `run_gpu()` 中训练与验证之间？**
放在 epoch 训练函数末尾可利用已存在的 rank 线程并行 launch，避免在 `run_gpu()` 中再 spawn 线程或串行遍历 rank。逻辑上仍保证“训练结束后、验证前”。

### 2.2 流选择：`UPDATE` 流

- `UPDATE_BN_INF_PARAMS` 是参数维护图，语义上与 `OPTIMIZER`/`CAST_MAIN` 同类，使用 `StreamKind::UPDATE`。
- 在 last batch 末尾，已有 `sync_up()` / `sync_comp()` 模式。新增 `cudaGraphLaunch(n_upd_bn, s_up); sync_up();` 可自然保证 UPDATE 流上的写入对后续验证的 COMP 流可见。

### 2.3 构图方式：按 Region 批量迭代

在 `Compiler::build_auxiliary_graphs()` 中，使用 `MemoryPlan::get_ids_by_region()` 获取以下 6 个 Region 的 DTensor ID 向量：

- 输入：`W_BN_WEIGHT`（gamma）、`W_BN_BIAS`（beta）、`B_NEXT_MEAN`、`B_NEXT_VAR`
- 输出：`W_EQ_SCALE`、`W_EQ_BIAS`

同一索引 `i` 在 6 个向量中对应同一 BN/CBR 实例（分配顺序一致），因此可直接配对构造 `n` 个 `BN_UPDATE_EQ_PARAMS` 节点。

**优点**：自动覆盖 `Bn1d`/`Bn2d`/`CBR`/`BottleneckProjection`/`BottleneckIdentity`/`BasicBlockProjection`/`BasicBlockIdentity`/`InvResidual*` 等所有含 BN 的层，无需按 `LayerKind` 写分支。

### 2.4 eps 来源：与当前代码一致

当前代码对 eps 的获取采用“CPU 读 device 标量、GPU 读 `node.params`”的分工：

- **CPU BN FWD/INF**：从 `input_ids` 中的全局 `bn_epsilon` 标量张量读取。
- **GPU BN FWD/INF**：从 `node.params.bn().eps` 读取（CUDA Graph 捕获时 kernel 参数按 host 值捕获，避免捕获期 D2H 同步）。

`BN_UPDATE_EQ_PARAMS` 遵循同一约定：

- **CPU launch**：从 `input_ids[4]`（`bn_epsilon` 标量）读取 eps。
- **GPU launch**：从 `node.params.bn().eps` 读取 eps。

因此构图时：
1. 仍需把 `memory_plan.baseline().bn_epsilon` 追加为第 5 个输入（供 CPU 使用）。
2. 同时把公共 `eps` 写入 `node.params`（供 GPU 使用）。

### 2.5 内存约束

新增算子与改造后的 INF 必须**只使用输入/输出张量指针**，禁止：
- `malloc` / `new` / `cudaMalloc` / `cudaMallocHost`
- 临时 `std::vector` 缓存
- 静态 thread_local 缓冲区
- workspace 扩容

`BN_UPDATE_EQ_PARAMS` CPU 路径：仅 5 个输入指针 + 2 个输出指针，循环计算。  
`BN_UPDATE_EQ_PARAMS` GPU 路径：复用现有 kernel，无额外显存。  
新 INF kernel：仅读取 `X/eq_scale/eq_bias`，写入 `Y`，无额外显存。

---

## 3. 详细改动

### 3.1 新增 `GraphId::UPDATE_BN_INF_PARAMS`

**文件**: `include/renaissance/graph/computation_graph.h`

在 `LARS_DEEP_CONV_OPT` 与 `COUNT` 之间插入：

```cpp
UPDATE_BN_INF_PARAMS,  ///< 训练后更新 BN 推理参数 (eq_scale/eq_bias)
```

同步修改三处：
- `graph_id_to_string()`: 新增 `case GraphId::UPDATE_BN_INF_PARAMS: return "UPDATE_BN_INF_PARAMS";`
- `is_shape_invariant_graph()`: 新增 `case GraphId::UPDATE_BN_INF_PARAMS: return true;`（只操作 per-channel 1D 张量，与 batch/分辨率无关）
- `is_train_graph()`: 新增 `case GraphId::UPDATE_BN_INF_PARAMS: return true;`（属于训练阶段生命周期）

### 3.2 新增 `ComputeOp::BN_UPDATE_EQ_PARAMS`

**文件**: `include/renaissance/graph/op_kind.h`

在 BN 区域（`BN2D_AMP_INF` 之后）新增：

```cpp
BN_UPDATE_EQ_PARAMS,  ///< 根据 running_mean/var 更新 eq_scale/eq_bias
```

**文件**: `src/graph/op_kind.cpp`
- `compute_op_to_string()`: 新增 `case ComputeOp::BN_UPDATE_EQ_PARAMS: return "BN_UPDATE_EQ_PARAMS";`
- `format_params()`: 在 BN 参数打印分支中新增 `case ComputeOp::BN_UPDATE_EQ_PARAMS:`（复用 `BNParams` 打印 `eps/momentum`）

**文件**: `src/backend/op_stream_policy.cpp`

```cpp
case ComputeOp::BN_UPDATE_EQ_PARAMS:
    return StreamKind::UPDATE;
```

**输入/输出张量**:

| 索引 | 张量 | Region | Shape | 说明 |
|:---:|:---|:---|:---|:---|
| 0 | gamma | `W_BN_WEIGHT` | [1,1,1,C] | 可训练权重 |
| 1 | beta | `W_BN_BIAS` | [1,1,1,C] | 可训练偏置 |
| 2 | running_mean | `B_NEXT_MEAN` | [1,1,1,C] | 当前 running mean |
| 3 | running_var | `B_NEXT_VAR` | [1,1,1,C] | 当前 running var |
| 4 | bn_epsilon | `S_SCALAR_FP32` | [1,1,1,1] | 全局 eps 标量（CPU 路径使用） |
| 0 | eq_scale | `W_EQ_SCALE` | [1,1,1,C] | 输出：推理等效 scale |
| 1 | eq_bias | `W_EQ_BIAS` | [1,1,1,C] | 输出：推理等效 bias |

> GPU 路径从 `node.params.bn().eps` 读取 eps，与当前 GPU BN FWD/INF 一致。

### 3.3 实现 `BN_UPDATE_EQ_PARAMS` 算子

#### CPU 路径（`src/backend/ops/dtensor/bn_op.cpp`）

```cpp
static void launch_bn_update_eq_params_cpu(CpuOpContext* op_ctx) {
    // input_ids:  [0]=gamma, [1]=beta, [2]=running_mean, [3]=running_var, [4]=bn_epsilon
    // output_ids: [0]=eq_scale, [1]=eq_bias
    const float* gamma = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const float* beta  = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    const float* rm    = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));
    const float* rv    = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[3]));
    const float* eps_ptr = (op_ctx->num_inputs >= 5)
        ? static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[4]))
        : nullptr;

    float* eq_scale = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    float* eq_bias  = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));

    int C = op_ctx->input_shape.c;
    float eps = eps_ptr ? *eps_ptr : 1e-5f;  // 与当前 CPU BN FWD/INF 一致

    for (int c = 0; c < C; ++c) {
        float inv_std = 1.0f / std::sqrt(rv[c] + eps);
        eq_scale[c] = gamma[c] * inv_std;
        eq_bias[c]  = beta[c] - rm[c] * eq_scale[c];
    }
}
```

**注意**:
- `op_ctx->input_shape.c` 来自第一个输入 gamma 的 shape `{1,1,1,C}`。
- eps 读取方式与当前 `launch_bn_fp32_fwd_cpu` / `launch_bn_fp32_inf_cpu` 完全一致：从 device 标量 `bn_epsilon` 读取。

#### GPU 路径（`src/backend/ops/dtensor/bn_op.cpp` + `bn_op.cu`）

`bn_op.cpp` 新增 launch wrapper：

```cpp
#ifdef TR_USE_CUDA
static void launch_bn_update_eq_params_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=gamma, [1]=beta, [2]=running_mean, [3]=running_var
    // output_ids: [0]=eq_scale, [1]=eq_bias
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[0]);
    int C = dt_w.shape.c();

    StreamKind sk = get_op_default_stream(node.compute_op);  // UPDATE
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(stream);

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_stream = state.streams[out_idx].stream;
        if (prev_stream != stream) {
            cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
        }
    }
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    float eps = 1e-5f;
    if (!node.params.is_empty()) {
        eps = node.params.bn().eps;  // 与当前 GPU BN FWD/INF 一致
    }

    cudaError_t err = launch_tr_bn_compute_eq_params_kernel(
        static_cast<const float*>(ctx.ptr_at(node.input_ids[0])),  // gamma
        static_cast<const float*>(ctx.ptr_at(node.input_ids[1])),  // beta
        static_cast<const float*>(ctx.ptr_at(node.input_ids[2])),  // running_mean
        static_cast<const float*>(ctx.ptr_at(node.input_ids[3])),  // running_var
        eps,
        static_cast<float*>(ctx.ptr_at(node.output_ids[0])),       // eq_scale
        static_cast<float*>(ctx.ptr_at(node.output_ids[1])),       // eq_bias
        C,
        stream);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("BN_UPDATE_EQ_PARAMS kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}
#endif
```

复用 `bn_op.cu:20-55` 的 `launch_tr_bn_compute_eq_params_kernel`，**不新增显存申请**。

#### 算子注册

在 `register_op_bn()` 末尾新增：

```cpp
// BN_UPDATE_EQ_PARAMS
table[static_cast<size_t>(ComputeOp::BN_UPDATE_EQ_PARAMS)].launch_cpu = launch_bn_update_eq_params_cpu;
#ifdef TR_USE_CUDA
table[static_cast<size_t>(ComputeOp::BN_UPDATE_EQ_PARAMS)].launch_cuda = launch_bn_update_eq_params_cuda;
#endif
```

### 3.4 修改 BN INF（CPU）

**文件**: `src/backend/ops/dtensor/bn_op.cpp`，`launch_bn_fp32_inf_cpu`

从 8 输入改为 3 输入：`[X, eq_scale, eq_bias]`。删除重新计算段，直接读取预计算值。

```cpp
static void launch_bn_fp32_inf_cpu(CpuOpContext* op_ctx) {
    // input_ids:  [0]=X, [1]=eq_scale, [2]=eq_bias
    // output_ids: [0]=Y
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const float* eq_scale = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    const float* eq_bias  = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));
    float* y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n;
    int C = op_ctx->input_shape.c;
    int H = op_ctx->input_shape.h;
    int W = op_ctx->input_shape.w;
    int spatial = N * H * W;

    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < spatial; ++i) {
            y[i * C + c] = x[i * C + c] * eq_scale[c] + eq_bias[c];
        }
    }
}
```

### 3.5 修改 BN INF（GPU）

**文件**: `src/backend/ops/dtensor/bn_op.cu`

新增两个只读 `eq_scale/eq_bias` 的 kernel：

```cuda
__global__ void tr_bn_inf_eq_fp32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ eq_scale,
    const float* __restrict__ eq_bias,
    float* __restrict__ y,
    int N, int C, int H, int W)
{
    int spatial = N * H * W;
    int total = spatial * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % C;
    int nhw = idx / C;
    y[nhw * C + c] = x[nhw * C + c] * eq_scale[c] + eq_bias[c];
}

__global__ void tr_bn_inf_eq_fp16_kernel(
    const __half* __restrict__ x,
    const float* __restrict__ eq_scale,
    const float* __restrict__ eq_bias,
    __half* __restrict__ y,
    int N, int C, int H, int W)
{
    int spatial = N * H * W;
    int total = spatial * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % C;
    int nhw = idx / C;
    float val = __half2float(x[nhw * C + c]) * eq_scale[c] + eq_bias[c];
    y[nhw * C + c] = __float2half(val);
}

extern "C" cudaError_t launch_tr_bn_inf_eq_kernel(
    const void* x,
    const float* eq_scale,
    const float* eq_bias,
    void* y,
    int N, int C, int H, int W,
    bool is_fp16,
    cudaStream_t stream)
{
    const int block_size = 256;
    int spatial = N * H * W;
    int total = spatial * C;
    const int grid_size = (total + block_size - 1) / block_size;
    if (is_fp16) {
        tr_bn_inf_eq_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const __half*>(x), eq_scale, eq_bias,
            static_cast<__half*>(y), N, C, H, W);
    } else {
        tr_bn_inf_eq_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(x), eq_scale, eq_bias,
            static_cast<float*>(y), N, C, H, W);
    }
    return cudaGetLastError();
}
```

**文件**: `src/backend/ops/dtensor/bn_op.cpp`，`launch_bn_inf_cuda`

改为读取 `eq_scale/eq_bias`：

```cpp
static void launch_bn_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=X, [1]=eq_scale, [2]=eq_bias
    // output_ids: [0]=Y
    TR_CHECK(node.input_ids.size() >= 3, ShapeError,
             "BN INF requires at least 3 inputs. Got " << node.input_ids.size());
    TR_CHECK(node.output_ids.size() >= 1, ShapeError,
             "BN INF requires at least 1 output. Got " << node.output_ids.size());

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    Shape shape = dt_x.shape;
    int N = shape.n(), H = shape.h(), W = shape.w(), C = shape.c();
    bool is_fp16 = (dt_x.dtype == DType::FP16);

    StreamKind sk = get_op_default_stream(node.compute_op);  // COMP_2
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(stream);

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_stream = state.streams[out_idx].stream;
        if (prev_stream != stream) {
            cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
        }
    }
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const float* d_eq_scale = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    const float* d_eq_bias  = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));
    void* d_y = ctx.ptr_at(node.output_ids[0]);

    cudaError_t err = launch_tr_bn_inf_eq_kernel(
        ctx.ptr_at(node.input_ids[0]),
        d_eq_scale, d_eq_bias,
        d_y, N, C, H, W, is_fp16, stream);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("BN INF kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}
```

旧的 `tr_bn_inf_fp32/fp16_kernel` 与 `launch_tr_bn_inf_kernel` 保留但不调用，后续清理。

### 3.6 修改 CBR INF

**文件**: `src/backend/ops/dtensor/cbr_op.cpp`，`launch_cbr_amp_inf_cuda`

精简 BN 段输入读取：

```cpp
// ── 2) BN INF on COMP_2（直接读取 eq_scale/eq_bias）─────────────
{
    StreamKind sk = StreamKind::COMP_2;
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(stream);

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_stream = state.streams[out_idx].stream;
        if (prev_stream != stream) {
            cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
        }
    }
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_x = mp.get_dtensor(node.output_ids[0]);  // conv_output
    Shape shape = dt_x.shape;
    int N = shape.n(), H = shape.h(), W = shape.w(), C = shape.c();
    bool is_fp16 = (dt_x.dtype == DType::FP16);

    // 修改：input_ids[2]=eq_scale, input_ids[3]=eq_bias
    const float* eq_scale = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));
    const float* eq_bias  = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));
    void* y = ctx.ptr_at(node.output_ids[3]);  // bn_output

    cudaError_t err = launch_tr_bn_inf_eq_kernel(
        ctx.ptr_at(node.output_ids[0]),  // x = conv_output
        eq_scale, eq_bias,
        y, N, C, H, W, is_fp16, stream);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CBR_AMP_INF bn kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}
```

### 3.7 修改 LayerDescriptor 的 input_indices

**文件**: `src/graph/layer_descriptor_registry.cpp`

#### `build_bn_inference`（L276-294）

```cpp
// 修改前: {10, 9, 0, 1, 5, 6}
// 修改后: 仅保留 eq_scale(10), eq_bias(9)；X 由 Compiler 在 begin 注入
n.input_indices = {10, 9};
```

#### `build_cbr_inference`（L505-519）

```cpp
// 修改前: {4, 8, 9, 18, 17, 13, 14}
// 修改后: 仅保留 amp_w(4), eq_scale(18), eq_bias(17)；X 由 Compiler 在 begin 注入
n.input_indices = {4, 18, 17};
```

### 3.8 Compiler 修改

**文件**: `src/graph/compiler.cpp`

#### 3.8.1 移除 BN/CBR INF 的 eps 注入

删除 L1707-1717：

```cpp
// BN INF: 注入全局标量 bn_epsilon（用于 eq_scale/eq_bias 惰性生成）
if (is_bn_inf_op(gn.compute_op)) {
    const auto& b = memory_plan.baseline();
    if (b.bn_epsilon >= 0) gn.input_ids.push_back(b.bn_epsilon);
}

// CBR INF: 注入全局标量 bn_epsilon
if (gn.compute_op == ComputeOp::CBR_AMP_INF) {
    const auto& b = memory_plan.baseline();
    if (b.bn_epsilon >= 0) gn.input_ids.push_back(b.bn_epsilon);
}
```

#### 3.8.2 在 `build_auxiliary_graphs` 中构建 `UPDATE_BN_INF_PARAMS` 图

在函数末尾（`NAN_CHECK_AND_GRAD_SCALING` 之后、`RANGE_D2D_COPY` 之前或之后均可）新增：

```cpp
// 7. UPDATE_BN_INF_PARAMS 图：每个 epoch 末统一预计算所有 BN/CBR 层的 eq_scale/eq_bias
if (memory_plan.is_region_populated(Region::W_BN_WEIGHT)) {
    const auto& w_ids  = memory_plan.get_ids_by_region(Region::W_BN_WEIGHT);
    const auto& b_ids  = memory_plan.get_ids_by_region(Region::W_BN_BIAS);
    const auto& rm_ids = memory_plan.get_ids_by_region(Region::B_NEXT_MEAN);
    const auto& rv_ids = memory_plan.get_ids_by_region(Region::B_NEXT_VAR);
    const auto& es_ids = memory_plan.get_ids_by_region(Region::W_EQ_SCALE);
    const auto& eb_ids = memory_plan.get_ids_by_region(Region::W_EQ_BIAS);

    size_t n = w_ids.size();
    TR_CHECK(n == b_ids.size() && n == rm_ids.size() && n == rv_ids.size()
             && n == es_ids.size() && n == eb_ids.size(),
             ShapeError, "BN update eq params: region count mismatch");

    // 所有 BN/CBR 层共享同一 eps，取第一个有效值即可
    float common_eps = 1e-5f;
    for (const auto& layer : arch.layers()) {
        if (layer.kind == LayerKind::Bn1d || layer.kind == LayerKind::Bn2d ||
            layer.kind == LayerKind::CBR) {
            if (std::holds_alternative<BNParams>(layer.params)) {
                common_eps = std::get<BNParams>(layer.params).eps;
                break;
            } else if (std::holds_alternative<CbrLayerParams>(layer.params)) {
                common_eps = std::get<CbrLayerParams>(layer.params).eps;
                break;
            }
        }
    }

    int32_t eps_id = memory_plan.baseline().bn_epsilon;
    for (size_t i = 0; i < n; ++i) {
        GraphNode node;
        node.kind = GraphNode::Kind::COMPUTE;
        node.compute_op = ComputeOp::BN_UPDATE_EQ_PARAMS;
        node.params = OpParams(BNParams{common_eps, 0.0f});  // momentum 无关，供 GPU 路径使用
        node.input_ids  = {w_ids[i], b_ids[i], rm_ids[i], rv_ids[i]};
        if (eps_id >= 0) node.input_ids.push_back(eps_id);  // 供 CPU 路径使用
        node.output_ids = {es_ids[i], eb_ids[i]};
        train_cg.append(GraphId::UPDATE_BN_INF_PARAMS, std::move(node));
    }
}
```

### 3.9 DeepLearningTask 修改

**文件**: `src/task/deep_learning_task.cpp`

#### 3.9.1 新增 GraphSlot

在 `enum class GraphSlot`（L37-69）中，`CLEAR_METRICS` 之前插入：

```cpp
UPDATE_BN_INF_PARAMS,  ///< 训练后更新 BN 的 eq_scale/eq_bias
```

#### 3.9.2 新增 `stream_for` 分支

在 `stream_for()`（L520-556）中新增：

```cpp
case GraphId::UPDATE_BN_INF_PARAMS:
    return StreamKind::UPDATE;
```

#### 3.9.3 `build_exec_table` 中解析

在 `v <= 3` 分支中新增：

```cpp
g[S(GraphSlot::UPDATE_BN_INF_PARAMS)] = resolve(GraphId::UPDATE_BN_INF_PARAMS, rank, v);
```

由于 `UPDATE_BN_INF_PARAMS` 是 shape-invariant，所有 variant 共享同一 captured graph。

#### 3.9.4 GPU 训练循环: `run_train_epoch_gpu`

在 pre-resolve 区域（L1054-1081）新增：

```cpp
auto n_upd_bn = g_n[S(GraphSlot::UPDATE_BN_INF_PARAMS)];
```

在 last batch 块结束后（L1235 的 `}` 之后、`catch` 之前）新增：

```cpp
// 每个 epoch 训练结束后更新 BN 推理参数 (eq_scale/eq_bias)
if (n_upd_bn) { cudaGraphLaunch(n_upd_bn, s_up); sync_up(); }
```

**时序**: 位于 `CAST_MAIN`（AMP）+ `sync_up()` 之后，所有 rank 的 UPDATE 流同步完成才返回，保证后续验证读到最新 `eq_scale/eq_bias`。

#### 3.9.5 CPU 训练循环: `run_train_epoch_cpu`

在预解析区域（L1461-1473 附近）新增：

```cpp
int32_t idx_upd_bn = idx_for(GraphId::UPDATE_BN_INF_PARAMS, 0);
```

在函数最末尾（L1631 的 `train_loss` 计算之前）新增：

```cpp
// 每个 epoch 训练结束后更新 BN 推理参数
if (idx_upd_bn >= 0) launch(idx_upd_bn);
```

**注意**: 放在函数末尾可同时覆盖 `batches == 1` 的早返回路径。需要把该 launch 放在 `batches == 1` 分支的 `return train_loss;` 之前，以及 `batches > 1` 的 last batch 块之后。最简洁的做法是把 `batches == 1` 的 `return` 改为跳出到公共尾部，或在两处分别插入。推荐在公共尾部统一插入，并调整 `batches == 1` 分支不直接 return。

---

## 4. 关键时序

```text
for each epoch:
  for batch in normal batches:
    FWD → BWD → COMM → OPTIMIZER → CAST_MAIN (if AMP)
  last batch:
    FWD → BWD → COMM → OPTIMIZER → CAST_MAIN (if AMP)
    UPDATE_BN_INF_PARAMS   ← UPDATE 流，一次性更新所有 BN 层 eq_scale/eq_bias
    sync_up()              ← 确保写入完成
  if validation:
    INF_MAIN_A/B  ← BN INF 直接读 eq_scale/eq_bias
```

---

## 5. 数值等价性证明

修改前后数学公式完全一致：

```text
修改前（每次 INF 计算）:
  inv_std   = 1 / sqrt(running_var + eps)
  eq_scale  = gamma * inv_std
  eq_bias   = beta - running_mean * eq_scale
  y         = x * eq_scale + eq_bias

修改后:
  UPDATE_BN_INF_PARAMS（每 epoch 一次）:
    inv_std  = 1 / sqrt(running_var + eps)
    eq_scale = gamma * inv_std
    eq_bias  = beta - running_mean * eq_scale
  BN INF（每次验证）:
    y = x * eq_scale + eq_bias
```

- 同一 epoch 内 `running_mean`/`running_var` 不变（训练结束后才更新）。
- `gamma`/`beta` 在优化器更新后不变直到下一 epoch。
- `eps` 为同一常量。
- 因此每个验证 batch 读取的 `eq_scale`/`eq_bias` 与旧代码现场计算的值逐位相同。

---

## 6. 不变式与约束保证

| 保证项 | 说明 |
|:---|:---|
| 不申请额外内存 | `BN_UPDATE_EQ_PARAMS` CPU/GPU、新 INF kernel 均只使用输入输出张量指针 |
| 不影响训练 | 更新图在训练完成后执行，不参与 FWD/BWD/COMM/OPTIMIZER |
| 不影响 AMP/FP32 | 公式与精度无关，gamma/beta/mean/var/eq 均为 FP32 |
| 不影响首层类型 | 仅处理含 BN 的 Region，不关心首层结构 |
| CPU/GPU 一致 | 双路径公式等价，各自使用相同精度的 sqrt |
| 不影响多卡 | 每 rank 独立更新；running stats 已通过 `STATS_COMM` 同步 |
| 自动覆盖复合层 | Region 迭代覆盖 Bottleneck/BasicBlock/InvResidual 内部 BN |

---

## 7. 边界情况

| 场景 | 处理 |
|:---|:---|
| 模型无 BN 层 | `W_BN_WEIGHT` region 为空，图为空，handle 为 nullptr，跳过 |
| `batches == 1`（GPU） | 唯一 batch 走 last batch 块，UPDATE 正常执行 |
| `batches == 1`（CPU） | 需将 UPDATE launch 放在公共尾部，避免早 return 跳过 |
| epoch 0 首次验证 | 默认 `offset=1`，epoch 0 训练结束后执行 UPDATE，验证前已就绪 |
| `offset=0` 提前验证 | 当前默认配置无此问题；若需支持，可在首次验证前额外执行一次 UPDATE |
| 多分辨率训练 | UPDATE 为 shape-invariant，6 个 variant 共享同一 CapturedGraph |
| SEMA 切换 | `apply_sema_switch()` 在 epoch 开头执行，UPDATE 使用切换后的主模型参数，行为正确 |

---

## 8. 实施顺序

1. `include/renaissance/graph/op_kind.h` + `src/graph/op_kind.cpp`：新增 `BN_UPDATE_EQ_PARAMS`
2. `src/backend/op_stream_policy.cpp`：映射到 `UPDATE` 流
3. `src/backend/ops/dtensor/bn_op.cpp` / `bn_op.cu`：实现 update 算子、新增 INF kernel、改造 BN INF
4. `src/backend/ops/dtensor/cbr_op.cpp`：改造 CBR INF 读取 eq_scale/eq_bias
5. `include/renaissance/graph/computation_graph.h`：新增 `GraphId::UPDATE_BN_INF_PARAMS`
6. `src/graph/layer_descriptor_registry.cpp`：精简 BN/CBR INF 的 input_indices
7. `src/graph/compiler.cpp`：移除 INF eps 注入；构建 UPDATE_BN_INF_PARAMS 图
8. `src/task/deep_learning_task.cpp`：新增 GraphSlot、stream_for、build_exec_table、GPU/CPU 训练循环 launch

---

## 9. 验证计划

1. **编译检查**: 全量 ninja 编译通过，无新增 warning。
2. **单元验证**: 小网络（如 `tests/bn/test_bn_fwd_bwd`）跑 1 个 epoch，手动读取 `W_EQ_SCALE`/`W_EQ_BIAS`，验证与 `gamma/beta/running stats` 手工计算一致。
3. **集成验证**: VGG16BN AMP 跑 1-3 个 epoch：
   - 确认 `UPDATE_BN_INF_PARAMS` 在每个 epoch 末成功执行。
   - 验证阶段 BN INF / CBR INF 不再访问 `B_NEXT_MEAN/B_NEXT_VAR`。
4. **精度回归**: 相同随机种子下，对比修改前后验证 loss/acc，差异在 FP32/FP16 累加顺序误差范围内（~1e-5）。
5. **性能验证**: 使用 Nsight Compute 或日志确认：
   - 验证阶段 INF kernel 不再出现 `sqrt`/`rcp` 指令。
   - `UPDATE_BN_INF_PARAMS` 每个 epoch 只执行一次，耗时远小于一个验证 batch。
6. **边界测试**: `batches == 1`、无 BN 模型、CPU 路径均通过。
