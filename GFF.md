# GAP 算子实现方案 — 最终完整版

**撰写时间**：2026-05-18
**分析基础**：GPF.md（S/K/D 三方终版方案 + 用户最终约束）、框架现有代码、legacy `gap.cpp`/`gap_backward.cu`

---

## 一、总览

实现 **GAP_FP32_FWD、GAP_FP32_BWD、GAP_AMP_FWD、GAP_AMP_BWD** 四个算子，采用经过 S/K/D 三方论证与用户约束校验后收敛的方案：

| 决策项 | 选型 | 理由 |
|--------|------|------|
| FWD (CUDA) | **cuDNN FE Resample** (AVGPOOL, generate_index=false) | legacy 验证 A100 性能极佳；三方一致 |
| BWD (CUDA) | **自定义 CUDA kernel** (grid-stride loop, stride-based indexing) | cuDNN Resample_backward 支持不佳；K/D 论证、用户确认；零 workspace |
| BWD FP16 内积 | **FP32** | 大 H×W 下 FP16 累加可能溢出 |
| CPU FP32 | **XNNPACK** (`xnn_define_global_average_pooling_2d` FWD + 手动循环 BWD) | 四位专家 (KM/QW/DSX/GL) 一致选用；FWD 自动 SIMD (AVX-512/NEON)；BWD 极简无依赖 |
| CPU AMP | **不支持**（`TR_TYPE_ERROR`）| 与 flatten/FC 一致 |
| 图缓存 | **Per-shape** (handle + NCHW + is_amp) | FC 算子已验证 |
| Workspace | **DeviceContext 池化管理**；BWD 零 workspace | 不额外 cudaMalloc |

---

## 二、文件变更清单（完整）

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `include/renaissance/graph/op_kind.h` | 修改 | 删 `GAP_FWD`/`GAP_BWD`，增 4 个精度特化枚举 |
| 2 | `src/graph/op_kind.cpp` | 修改 | `compute_op_to_string` 和 `to_string_params` 新增 case |
| 3 | `src/graph/layer_descriptor_registry.cpp` | 修改 | `build_gap_forward`/`backward` 根据 descs dtype 选择算子 |
| 4 | `src/backend/op_registry.cpp` | 修改 | `require_warmup` 更新；`register_default_ops` 加 `register_op_gap()` |
| 5 | `include/renaissance/backend/op_registry.h` | 修改 | 声明 `void register_op_gap();` |
| 6 | `src/backend/ops/dtensor/gap_op.cu` | **新建** | FP32 + AMP backward kernel ×2 + launch wrapper |
| 7 | `src/backend/ops/dtensor/gap_op.cpp` | **新建** | CPU(XNNPACK) + CUDA 桥接 + cuDNN FE FWD + cache + 注册 |
| 8 | `src/CMakeLists.txt` | 修改 | `RENAISSANCE_SOURCES` 追加 `gap_op.cpp`、`gap_op.cu` |

---

## 三、修改 op_kind.h — 枚举替换

**删除**：
```cpp
GAP_FWD,
GAP_BWD,
```

**替换为**（插入原 GAP_FWD 位置附近，保持与 FLATTEN/FC 模式一致）：
```cpp
GAP_FP32_FWD,
GAP_FP32_BWD,
GAP_AMP_FWD,
GAP_AMP_BWD,
```

由于 `kComputeOpCount` 是**末尾自动推导**（最后一个枚举值 + 1），不依赖具体枚举数，因此改成 4 个后无需手工调整计数。

---

## 四、修改 op_kind.cpp — 字符串表

### 4.1 compute_op_to_string

原代码：
```cpp
case ComputeOp::GAP_FWD:               return "GAP_FWD";
case ComputeOp::GAP_BWD:               return "GAP_BWD";
```

改为：
```cpp
case ComputeOp::GAP_FP32_FWD:          return "GAP_FP32_FWD";
case ComputeOp::GAP_FP32_BWD:          return "GAP_FP32_BWD";
case ComputeOp::GAP_AMP_FWD:           return "GAP_AMP_FWD";
case ComputeOp::GAP_AMP_BWD:           return "GAP_AMP_BWD";
```

### 4.2 compute_op_to_string_params

新增 case（GAP 无参数，不做输出）：
```cpp
case ComputeOp::GAP_FP32_FWD:
case ComputeOp::GAP_FP32_BWD:
case ComputeOp::GAP_AMP_FWD:
case ComputeOp::GAP_AMP_BWD:
    break;  // GAP has no params
```

---

## 五、修改 layer_descriptor_registry.cpp — 精度派发

### 5.1 build_gap_forward

原代码：
```cpp
n.op = ComputeOp::GAP_FWD;
```

改为：
```cpp
n.op = (descs[0].dtype == DType::FP16)
       ? ComputeOp::GAP_AMP_FWD
       : ComputeOp::GAP_FP32_FWD;
```

`descs[0]` 是 `gap_output` (feat_dt)，由 `infer_gap_tensors` 提供，其 dtype 正确反映 AMP/FP32 模式。

### 5.2 build_gap_backward

原代码：
```cpp
n.op = ComputeOp::GAP_BWD;
```

改为：
```cpp
n.op = (descs[1].dtype == DType::FP16)
       ? ComputeOp::GAP_AMP_BWD
       : ComputeOp::GAP_FP32_BWD;
```

`descs[1]` 是 `gap_grad_slot` (feat_dt)，dtype 与 forward 一致。

---

## 六、修改 op_registry — warmup 与注册

### 6.1 op_registry.h

添加声明：
```cpp
void register_op_gap();
```

### 6.2 op_registry.cpp — require_warmup

原代码：
```cpp
case ComputeOp::GAP_FWD:        case ComputeOp::GAP_BWD:
    return true;
```

改为：
```cpp
case ComputeOp::GAP_FP32_FWD:
case ComputeOp::GAP_AMP_FWD:
    return true;
```

仅 FWD 需要 warmup（cuDNN FE 图构建）。BWD 是纯 kernel，无需 warmup。

### 6.3 op_registry.cpp — register_default_ops

添加：
```cpp
register_op_gap();
```

---

## 七、gap_op.cu — CUDA 反向 Kernel

新建文件，位于 `src/backend/ops/dtensor/gap_op.cu`。

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// FP32 backward kernel
__global__ void gap_bwd_fp32_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale)
{
    const int total = N * H * W * C;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        int c     = idx % C;
        int tmp   = idx / C;
        int w     = tmp % W;
        tmp      /= W;
        int h     = tmp % H;
        int n     = tmp / H;

        size_t dy_idx = n * dy_n_stride + c * dy_c_stride;
        size_t dx_idx = n * dx_n_stride + h * dx_h_stride + w * dx_w_stride + c * dx_c_stride;
        dx[dx_idx] = dy[dy_idx] * scale;
    }
}

// ====================================================================
// AMP backward kernel — FP16 I/O, FP32 内部运算
// ====================================================================
__global__ void gap_bwd_amp_kernel(
    const __half* __restrict__ dy,
    __half* __restrict__ dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale)
{
    // 数学: dx[n,c,h,w] = dy[n,c] / (H*W)
    // 第几步——总输出元素——粒度
    const int total = N * H * W * C;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        int c     = idx % C;
        int tmp   = idx / C;
        int w     = tmp % W;
        tmp      /= W;
        int h     = tmp % H;
        int n     = tmp / H;

        size_t dy_idx = n * dy_n_stride + c * dy_c_stride;
        size_t dx_idx = n * dx_n_stride + h * dx_h_stride + w * dx_w_stride + c * dx_c_stride;

        float dy_val = __half2float(dy[dy_idx]);   // FP16→FP32
        dx[dx_idx] = __float2half(dy_val * scale); // FP32 乘 accumulate，写回 FP16
    }
}

// ====================================================================
// Launch wrapper — FP32
// ====================================================================
cudaError_t launch_gap_bwd_fp32(
    const float* dy, float* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale, cudaStream_t stream)
{
    int total = N * H * W * C;
    int block = 256;
    int grid  = (total + block - 1) / block;
    gap_bwd_fp32_kernel<<<grid, block, 0, stream>>>(
        dy, dx, N, H, W, C,
        dy_n_stride, dy_c_stride,
        dx_n_stride, dx_h_stride, dx_w_stride, dx_c_stride,
        scale);
    return cudaGetLastError();
}

cudaError_t launch_gap_bwd_amp(
    const __half* dy, __half* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale, cudaStream_t stream)
{
    int total = N * H * W * C;
    int block = 256;
    int grid  = (total + block - 1) / block;
    gap_bwd_amp_kernel<<<grid, block, 0, stream>>>(
        dy, dx, N, H, W, C,
        dy_n_stride, dy_c_stride,
        dx_n_stride, dx_h_stride, dx_w_stride, dx_c_stride,
        scale);
    return cudaGetLastError();
}
```

### 设计说明

- **grid-stride loop**：`idx += blockDim.x * gridDim.x` 确保覆盖全部 `N×C×H×W` 个元素，grid 越小越宽泛。
- **Stride-based**：不假设紧凑布局。dy/dx 各取 6 个 stride 参数，AMP C 通道 padding 时仍正确。
- **Decompose idx → (n, h, w, c)**：基于 NHWC 的逻辑索引展开。`idx/(C·W·H)→n, rem/(C·W)→h, rem/C→w, rem% C→c`。这等价于遍历 NHWC 物理顺序。
- **Scale 预计算**：`scale = 1.0f / (H * W)` 在 host 侧算好传入，kernel 内仅做乘法。
- **__restrict__**：引导 NVCC 生成更优加载指令。

---

## 八、gap_op.cpp — 算子主体

新建文件，位于 `src/backend/ops/dtensor/gap_op.cpp`。

### 8.1 头文件与条件编译结构

```cpp
#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/tensor/dtype.h"
#include "renaissance/core/logger.h"

#ifdef TR_USE_XNNPACK
#  include <xnnpack.h>
#endif

#ifdef TR_USE_CUDA
#  include "renaissance/backend/cudnn_utils.h"   // 已提供 namespace fe = cudnn_frontend;
#  include <cudnn_frontend.h>
#  include <cuda_runtime.h>
#  include <cudnn.h>
#  include <unordered_map>
#  include <memory>
#endif

namespace tr {

// ====================================================================
// CPU 路径 — XNNPACK Subgraph API (FWD) / Fallback 朴素循环
//
// FWD: 优先 XNNPACK subgraph 缓存于 per-shape map，
//      调用 xnn_reshape_runtime + xnn_invoke_runtime 执行；
//      若无 XNNPACK 则回退到朴素四重循环。
// BWD: dx[n,h,w,c] = dy[n,c] / (H*W)，无条件朴素循环。
//      四位专家共识：GAP BWD 标量乘法 + 空间广播，无需引入外部库。
// ====================================================================

#ifdef TR_USE_XNNPACK

// ──── XNNPACK cached subgraph FWD ──────────────────────────────

struct XNNGapCacheEntry {
    xnn_subgraph_t subgraph;
    xnn_runtime_t  runtime;
    uint32_t       input_id;
    uint32_t       output_id;
};

// TODO: 添加 static destructor 释放 xnn_delete_runtime / xnn_delete_subgraph
static std::unordered_map<uint64_t, XNNGapCacheEntry> s_gap_xnn_caches;

static void xnn_check(xnn_status s, const char* tag) {
    if (s != xnn_status_success) {
        TR_TYPE_ERROR("XNNPACK " << tag << " failed, status=" << static_cast<int>(s));
    }
}

static uint64_t xnn_shape_key(int N, int H, int W, int C) {
    return (static_cast<uint64_t>(N) << 48)
         | (static_cast<uint64_t>(H) << 32)
         | (static_cast<uint64_t>(W) << 16)
         | (static_cast<uint64_t>(C));
}

static XNNGapCacheEntry build_xnn_gap_fwd(int N, int H, int W, int C) {
    static std::once_flag init_flag;
    std::call_once(init_flag, [] { xnn_check(xnn_initialize(nullptr), "init"); });

    XNNGapCacheEntry e{};
    xnn_check(xnn_create_subgraph(2, 0, &e.subgraph), "subgraph");

    size_t in_dims[4]  = {static_cast<size_t>(N), static_cast<size_t>(H),
                          static_cast<size_t>(W), static_cast<size_t>(C)};
    size_t out_dims[4] = {static_cast<size_t>(N), 1, 1, static_cast<size_t>(C)};

    xnn_check(xnn_define_tensor_value(
        e.subgraph, xnn_datatype_fp32, 4, in_dims, nullptr,
        0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &e.input_id), "input");

    xnn_check(xnn_define_tensor_value(
        e.subgraph, xnn_datatype_fp32, 4, out_dims, nullptr,
        1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &e.output_id), "output");

    // xnn_define_global_average_pooling_2d 已弃用，改用 static_reduce
    size_t reduce_axes[2] = {1, 2};  // H 和 W 维度求 mean
    xnn_check(xnn_define_static_reduce(
        e.subgraph, xnn_reduce_mean,
        2, reduce_axes,
        e.input_id, e.output_id,
        XNN_FLAG_KEEP_DIMS), "gap_reduce");

    xnn_check(xnn_create_runtime_v4(e.subgraph, nullptr, nullptr,
        nullptr, 0, &e.runtime), "runtime");

    return e;
}

static void launch_gap_fp32_fwd_cpu_xnn(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n, H = op_ctx->input_shape.h;
    int W = op_ctx->input_shape.w, C = op_ctx->input_shape.c;

    uint64_t key = xnn_shape_key(N, H, W, C);
    auto it = s_gap_xnn_caches.find(key);
    if (it == s_gap_xnn_caches.end()) {
        auto cache = build_xnn_gap_fwd(N, H, W, C);
        it = s_gap_xnn_caches.emplace(key, cache).first;
    }
    const auto& e = it->second;

    void* externals[2] = {const_cast<float*>(x), y};
    xnn_check(xnn_reshape_runtime(e.runtime), "reshape");
    xnn_check(xnn_setup_runtime_v2(e.runtime, 2, externals), "setup");
    xnn_check(xnn_invoke_runtime(e.runtime), "invoke");
}

#else // !TR_USE_XNNPACK — Fallback 朴素循环

static void launch_gap_fp32_fwd_cpu_fallback(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n, H = op_ctx->input_shape.h;
    int W = op_ctx->input_shape.w, C = op_ctx->input_shape.c;
    float scale = 1.0f / static_cast<float>(H * W);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    sum += x[((n * H + h) * W + w) * C + c];
                }
            }
            y[n * C + c] = sum * scale;
        }
    }
}

#endif // TR_USE_XNNPACK

// ──── BWD 朴素循环（无条件，不依赖任何外部库）─────────────────

static void launch_gap_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const float* dy = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       dx = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->output_shape.n, H = op_ctx->output_shape.h;
    int W = op_ctx->output_shape.w, C = op_ctx->output_shape.c;
    float scale = 1.0f / static_cast<float>(H * W);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float g = dy[n * C + c] * scale;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    dx[((n * H + h) * W + w) * C + c] = g;
                }
            }
        }
    }
}



### 8.3 AMP CPU 不支持

```cpp
static void launch_gap_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("GAP_AMP_FWD/BWD not supported on CPU (use GPU)");
}
```

### 8.4 CUDA 桥接 — 外部 kernel 声明

```cpp
#ifdef TR_USE_CUDA

cudaError_t launch_gap_bwd_fp32(
    const float* dy, float* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale, cudaStream_t stream);

cudaError_t launch_gap_bwd_amp(
    const __half* dy, __half* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale, cudaStream_t stream);
```

### 8.5 cuDNN FE Forward Graph 构建函数

```cpp
struct GapFwdCacheKey {
    uint64_t handle_bits;
    int32_t  n, c, h, w;
    bool     is_amp;

    bool operator==(const GapFwdCacheKey& o) const {
        return handle_bits == o.handle_bits && n == o.n && c == o.c
            && h == o.h && w == o.w && is_amp == o.is_amp;
    }
};

struct GapFwdCacheKeyHasher {
    size_t operator()(const GapFwdCacheKey& k) const {
        size_t hv = std::hash<uint64_t>{}(k.handle_bits);
        hv ^= std::hash<int32_t>{}(k.n) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(k.c) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(k.h) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(k.w) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<bool>{}(k.is_amp) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        return hv;
    }
};

struct GapFwdCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size = 0;
};

static std::unordered_map<GapFwdCacheKey, GapFwdCache, GapFwdCacheKeyHasher>
    s_gap_fwd_caches;

static GapFwdCache build_gap_fwd_graph(
    cudnnHandle_t   handle,
    const DTensor&  dt_x,
    const DTensor&  dt_y,
    bool            is_amp)
{
    int N = dt_x.n(), C = dt_x.c(), H = dt_x.h(), W = dt_x.w();

    auto graph = create_cudnn_graph(is_amp ? DType::FP16 : DType::FP32);
    // create_cudnn_graph 内部已设置 io_data_type、intermediate_data_type、compute_data_type

    // 输入张量 X: [N, C, H, W] NHWC
    int64_t uid_x = 700;
    auto attr_x = std::make_shared<fe::graph::Tensor_attributes>();
    attr_x->set_name("gap_x")
          .set_dim({N, C, H, W})
          .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                       dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
          .set_data_type(to_fe_dtype(dt_x.dtype))
          .set_uid(uid_x);
    auto X = graph->tensor(attr_x);

    // Resample: AVGPOOL, 全空间覆盖 → 输出 1×1
    fe::graph::Resample_attributes pool_opts;
    pool_opts.set_resampling_mode(fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING)
             .set_padding_mode(fe::PaddingMode_t::ZERO_PAD)
             .set_window({H, W})
             .set_stride({H, W})
             .set_pre_padding({0, 0})
             .set_post_padding({0, 0})
             .set_generate_index(false);  // GAP 反向不需要 index

    auto resample_outputs = graph->resample(X, pool_opts);
    auto Y = resample_outputs[0];

    // 输出张量 Y: [N, C, 1, 1] NHWC
    int64_t uid_y = 701;
    Y->set_output(true)
     .set_name("gap_y")
     .set_dim({N, C, 1, 1})
     .set_stride({dt_y.n_stride_cuda(), dt_y.c_stride_cuda(),
                  dt_y.h_stride_cuda(), dt_y.w_stride_cuda()})
     .set_data_type(to_fe_dtype(dt_y.dtype))
     .set_uid(uid_y);

    finalize_cudnn_graph(graph.get(), handle);

    GapFwdCache cache;
    cache.graph          = graph;
    cache.tensor_to_id[attr_x] = uid_x;
    cache.tensor_to_id[Y]      = uid_y;
    cache.workspace_size = graph->get_workspace_size();

    return cache;
}
```

### 8.6 CUDA Launch 函数

```cpp
static void launch_gap_fp32_fwd_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);

    cudaStream_t  s      = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int           si     = state.get_or_register(s);
    state.output_stream_idx        = si;
    state.streams[si].has_pending_work = true;

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_1));

    GapFwdCacheKey key;
    key.handle_bits = reinterpret_cast<uint64_t>(handle);
    key.n = dt_x.n();  key.c = dt_x.c();  key.h = dt_x.h();  key.w = dt_x.w();
    key.is_amp = false;

    auto it = s_gap_fwd_caches.find(key);
    if (it == s_gap_fwd_caches.end()) {
        auto cache = build_gap_fwd_graph(handle, dt_x, dt_y, /*is_amp=*/false);
        ctx.ensure_workspace_grow(StreamKind::COMP_1, cache.workspace_size);
        it = s_gap_fwd_caches.emplace(key, std::move(cache)).first;
    }
    const auto& cache = it->second;

    std::unordered_map<int64_t, void*> vp;
    vp[700] = ctx.ptr_at(node.input_ids[0]);
    vp[701] = ctx.ptr_at(node.output_ids[0]);

    TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ctx.workspace(StreamKind::COMP_1)),
                      "GAP_FP32_FWD");
    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_gap_amp_fwd_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    // 与 launch_gap_fp32_fwd_cuda 结构相同，仅 is_amp=true
    // (相同结构，key.is_amp = true, build_gap_fwd_graph(..., true))
}

static void launch_gap_fp32_bwd_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);

    cudaStream_t s  = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int          si = state.get_or_register(s);
    state.output_stream_idx        = si;
    state.streams[si].has_pending_work = true;

    const float* dy = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    float*       dx = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int   N = dt_dy.n(),  C = dt_dy.c();
    int   H = dt_dx.h(),  W = dt_dx.w();
    float scale = 1.0f / static_cast<float>(H * W);

    cudaError_t err = launch_gap_bwd_fp32(
        dy, dx, N, H, W, C,
        dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
        dt_dx.n_stride_cuda(), dt_dx.h_stride_cuda(),
        dt_dx.w_stride_cuda(), dt_dx.c_stride_cuda(),
        scale, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("GAP_FP32_BWD kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_gap_amp_bwd_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    // 与 launch_gap_fp32_bwd_cuda 结构相同:
    //   取 dy/dx 为 const __half* / __half*
    //   调用 launch_gap_bwd_amp(...)
}
```

### 8.7 算子注册

```cpp
void register_op_gap() {
    auto& table = g_compute_op_table;

    // --- GAP_FP32_FWD: CPU(XNNPACK优先, fallback朴素) + CUDA(cuDNN FE) ---
    {
        auto& e = table[static_cast<size_t>(ComputeOp::GAP_FP32_FWD)];
        e.op = ComputeOp::GAP_FP32_FWD;
#ifdef TR_USE_XNNPACK
        e.launch_cpu = launch_gap_fp32_fwd_cpu_xnn;
#else
        e.launch_cpu = launch_gap_fp32_fwd_cpu_fallback;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_gap_fp32_fwd_cuda;
#endif
    }

    // --- GAP_FP32_BWD: CPU(朴素循环, 无条件) + CUDA(自定义kernel) ---
    {
        auto& e = table[static_cast<size_t>(ComputeOp::GAP_FP32_BWD)];
        e.op = ComputeOp::GAP_FP32_BWD;
        e.launch_cpu = launch_gap_fp32_bwd_cpu;  // 无条件：BWD 不依赖任何外部库
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_gap_fp32_bwd_cuda;
#endif
    }

    // --- GAP_AMP_FWD: CPU(不支持) + CUDA(cuDNN FE) ---
    {
        auto& e = table[static_cast<size_t>(ComputeOp::GAP_AMP_FWD)];
        e.op = ComputeOp::GAP_AMP_FWD;
        e.launch_cpu = launch_gap_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_gap_amp_fwd_cuda;
#endif
    }

    // --- GAP_AMP_BWD: CPU(不支持) + CUDA(自定义kernel) ---
    {
        auto& e = table[static_cast<size_t>(ComputeOp::GAP_AMP_BWD)];
        e.op = ComputeOp::GAP_AMP_BWD;
        e.launch_cpu = launch_gap_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_gap_amp_bwd_cuda;
#endif
    }
}

} // namespace tr
```

---

## 九、修改 CMakeLists.txt

在 `RENAISSANCE_SOURCES` 的 `backend/ops/dtensor/` block 中添加：

```cmake
backend/ops/dtensor/gap_op.cpp
backend/ops/dtensor/gap_op.cu
```

---

## 十、不可违反的约束

### 10.1 不要参考 MaxPool

MaxPool 尚未实现，其 `MAXPOOL_FWD`/`MAXPOOL_BWD` 是**占位枚举**，无对应 `register_op_maxpool()`。GAP 完全独立，不依赖也不参考 MaxPool。

### 10.2 GAP 算子极简

- **FWD**：1 输入 (x)，1 输出 (y)。无权重、无 bias、无 mask。
- **BWD**：1 输入 (dy)，1 输出 (dx)。同样无权重、无 mask。
- `num_inputs == 1`，`num_outputs == 1`。
- 不需要像 FC/Conv 那样传递参数结构体。

### 10.3 dX 覆盖 X（in-place gradient）

框架强制反向梯度写入**与正向输入同一内存区域**。即 BWD 的 `dx` DTensor 与 FWD 的 `x` DTensor 是同一个物理池中的同一块内存。

对 GAP 的影响：
- **无负面影响**：GAP backward 运算 `dx[n,c,h,w] = dy[n,c]/HW` 不依赖 x，覆盖 x 不造成数据竞态。
- **Stride 继承**：dx 的 stride 与 x 完全相同。
- 实现中，通过 `mp.get_dtensor(node.output_ids[0])` 获取 dx DTensor，其 shape/stride 自动正确。

### 10.4 绝对不要参考 MaxPool

我们这句话说了两遍，原因是我们发现了前面有人在方案里参考 maxpool，这是不合理的。

---

## 十一、实施步骤

| # | 操作 | 预估 | 验证 |
|---|------|------|------|
| 1 | 修改 `op_kind.h`：替换枚举 (Delete + Insert 4) | 3 min | grep 旧枚举名，确认无残留 |
| 2 | 修改 `op_kind.cpp`：to_string (×4) + to_string_params (×4) | 3 min | 编译通过即可 |
| 3 | 修改 `layer_descriptor_registry.cpp`：build_gap_forward/backward (×2) | 3 min | 人工 review dtype 判断逻辑 |
| 4 | 修改 `op_registry.cpp/.h`：require_warmup (×2) + register (×1) + 声明 | 3 min | 编译通过 |
| 5 | 新建 `gap_op.cu`：完整复制上述 CUDA kernel 代码 | 8 min | 单元编译 CUDA 源文件 |
| 6 | 新建 `gap_op.cpp`：完整复制上述算子主体代码 | 12 min | 单元编译 C++ 源文件 |
| 7 | 修改 `src/CMakeLists.txt`：追加两行 | 1 min | 编译通过 |
| 8 | 完整编译 (`cmake --build build/`) | 3 min | 0 error |
| 9 | 运行 GAP 测试验证数值正确性 | 5 min | MaxMSE < 1e-6 |
| **总计** | | **~41 min** | |