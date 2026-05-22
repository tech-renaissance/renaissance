# FC 算子 + Workspace 最终科学方案

## 引言：四轮讨论的进化与收敛

本方案综合了 FFF.md（导言需求）→ FFF2.md（第一轮方案）→ FFF3.md（兜底讨论）→ FFF4.md（方法论收敛）的全部讨论，以及三位小伙伴 S、K、D 的方案迭代，形成最终版。

| 议题 | FFF.md | FFF2.md | FFF3.md | **FFF4.md（收敛）** |
|------|--------|---------|---------|---------------------|
| workspace 所有者 | Task 或 DeviceContext | 分歧 | → DeviceContext | **→ DeviceContext（全票通过）** |
| 分配策略 | 临时→compile正式 | 512MB初始/Task预分配 | → grow-on-demand | **→ ensure_workspace_grow 精确按需** |
| 512MB 盲猜 | FFF.md #15 兜底 | D 仍采用 | S/K/D 一致否定 | **→ 全部放弃，精确优于盲猜** |
| Conv 算子 | 要改 | D 方案含修复 | 用户：不管 Conv | **→ 彻底排除出方案范围** |
| 中央收集器 | 无 | D 提出 | 争议 | **→ K/D 达成共识：仅 FC 时不需要** |
| Task 介入 | 迷茫 | S/K 主张 Task 介入 | 用户：DC全权 | **→ 零 Task 介入（全票通过）** |
| const 问题 | 未触及 | 未触及 | D 首次指出 | **→ S/K/D 各自处理，需统一** |

**核心变化**：小伙伴 S（FFF4）明显倾斜到小伙伴 D（FFF3）的"精确按需 + 算子级 Lazy Grow"路线，只是试图保留中央收集器。小伙伴 K（FFF4）彻底放弃 Task 预分配，走向纯粹的"算子级 Lazy Grow"。小伙伴 D（FFF4）坚持 const 安全的实现路径。**三人已解决除 const 实现细节外的全部分歧**。

### 用户明确指令（按时间线汇总）

| 指令 | 来源 |
|------|------|
| "bias 必定是 FP32" | FFF.md #6 |
| "cuDNN 接受 3D 输入" | FFF.md #2 |
| "C 通道必定连续" | FFF.md #13 |
| "每流一个、一次申请永久复用" | FFF.md #11 |
| "warmup 发现、compile 分配" | FFF.md #14, #16 |
| "把 workspace 管理全权交给 DeviceContext" | FFF2.md 用户补充 |
| "完全不用管 Conv 算子，当它不存在" | FFF3.md 用户补充 |
| "初始申请 512MB 至少是个兜底，希望有更好的" | FFF2.md 用户补充 |

---

## 最终方案

### 一、核心原则

```
原则 1: DeviceContext 是 workspace 唯一所有者，Task 零介入
原则 2: workspaces_[5] 天然实现每流一个 workspace
原则 3: 首次使用按需精确分配（cuDNN 报告多大就分多大）
原则 4: 不够大→释放旧→申请新，warmup 结束即得精确值
原则 5: Capture 和 Runtime 直接复用，零额外分配开销
原则 6: 算子内部绝不 cudaMalloc/Free，统一 ctx.workspace()
原则 7: 全部分配操作对 launch_cuda 透明（const 方法支持）
```

### 二、架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase B2: Warmup（每个 rank 串行）                              │
│  ─────────────────────────────                                  │
│  warmup_single_cudnn_op() → entry.launch_cuda(node, mp, ctx)   │
│    └→ cache miss → rebuild=true                                 │
│       └→ build_fc_amp_fwd_graph(handle, ...)                    │
│       └→ ws_needed = graph->get_workspace_size()                │
│       └→ ctx.ensure_workspace_grow(COMP_1, ws_needed)  ← 精确! │
│       └→ graph->execute(handle, vp, ws)                         │
│    └→ cudaStreamSynchronize  → 完成预热                          │
│                                                                  │
│  Phase B3: Capture（每个 rank，cudaStreamBeginCapture 内）       │
│  ───────────────────────────────────────────                    │
│  entry.launch_cuda(node, mp, ctx, state)                        │
│    └→ cache hit → rebuild=false                                  │
│    └→ ws = ctx.workspace(COMP_1)  ← warmup 已确定大小            │
│    └→ graph->execute(handle, vp, ws)  ← 纯 cache hit             │
│                                                                  │
│  Runtime: 每个 epoch → cudaGraphLaunch → workspace 自动复用      │
└─────────────────────────────────────────────────────────────────┘
```

**关键洞察**：`ensure_workspace_grow` **只在 `rebuild=true` 时调用**。warmup 阶段触发 rebuild → 精确分配。capture 阶段 cache hit → 直接复用。无需中央收集器、无需 compile 流程改动、无需 Phase B2.5 插入。时序天然正确。

### 三、基础设施：DeviceContext

#### 3.1 现状分析

当前 `DeviceContext` 已有完善的 workspace 基础设施（[device_context.h](file:///r:/renaissance/include/renaissance/backend/device_context.h)）：

```cpp
struct WSpace { void* ptr = nullptr; size_t size = 0; };
WSpace workspaces_[5];                          // 每流一个
void* workspace(StreamKind) const;              // 只读查询
size_t workspace_size(StreamKind) const;        // 只读查询
void pre_allocate_workspace(StreamKind, size_t); // 幂等预分配
void ensure_workspace(StreamKind, size_t);       // 不够大→抛异常
void free_workspaces();                          // 析构清理
```

**问题**：`ensure_workspace` 在不够大时**抛异常**，不支持扩容。warmup 场景下我们需要"不够大就扩容"而非抛异常。

#### 3.2 新增 `ensure_workspace_grow` 方法

```cpp
// device_context.h — 在 public 区新增
void ensure_workspace_grow(StreamKind kind, size_t req_size) const;
```

```cpp
// device_context.cpp — 实现
void DeviceContext::ensure_workspace_grow(StreamKind kind, size_t req_size) const {
#ifdef TR_USE_CUDA
    if (device_id_ < 0) return;

    constexpr size_t alignment = 256;
    size_t aligned = ((req_size + alignment - 1) / alignment) * alignment;
    size_t idx = stream_index(kind);
    auto& ws = workspaces_[idx];

    if (ws.size >= aligned) return;

    if (ws.ptr) {
        cudaFree(ws.ptr);
        ws.ptr = nullptr;
        ws.size = 0;
    }

    int prev_device = -1;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id_);

    cudaError_t err = cudaMalloc(&ws.ptr, aligned);
    cudaSetDevice(prev_device);

    if (err != cudaSuccess) {
        ws.ptr = nullptr;
        ws.size = 0;
        TR_GPU_OOM("Failed to grow workspace to " << aligned
                   << " bytes for stream " << static_cast<int>(kind)
                   << " on device " << device_id_
                   << ": " << cudaGetErrorString(err));
    }

    ws.size = aligned;
    TR_LOG_INFO("backend") << "DeviceContext " << device_id_
                           << ": workspace[stream " << static_cast<int>(kind)
                           << "] grown to " << aligned << " bytes";
#else
    (void)kind;
    (void)req_size;
#endif
}
```

#### 3.3 const 安全性：`workspaces_` 标记为 `mutable`

小伙伴 D（FFF3）首先指出：`launch_cuda` 签名是 `const DeviceContext&`，不能调用非 const 方法。小伙伴 S（FFF4）的做法是 `const_cast<DeviceContext&>(ctx).ensure_workspace_grow(...)`，const_cast 暴露在调用端，不够优雅。

**最优解**：将 `workspaces_` 标记为 `mutable`。

```cpp
// device_context.h — 仅改一个关键字
mutable WSpace workspaces_[5];
```

理由：
1. **逻辑 const 语义**：workspace 是句柄缓存，其扩容不影响 `DeviceContext` 的逻辑状态（设备ID、流、cuDNN handle 均不变）
2. **零 const_cast**：`ensure_workspace_grow` 声明为 `const`，实现中可直接修改 `workspaces_[idx]`，接口干净
3. **C++ 标准实践**：缓存/memoization/lazy-init 用 `mutable` 是标准惯用法
4. **对现有代码零影响**：`workspace()` 和 `workspace_size()` 已是 `const`，`free_workspaces()` 是析构函数调用的 private 方法（本身就是非 const），均不受影响

与三种 const 解决路径的对比：

| 路径 | 调用端 | 实现端 | 风险 |
|------|--------|--------|------|
| S: `const_cast<DeviceContext&>(ctx)` 外露 | ❌ 丑陋 | 非 const | 调用端误用风险 |
| K: 不改签名（有编译错误） | ❌ 编译失败 | 非 const | 不可行 |
| **D: `mutable WSpace` + const 方法** | ✅ 干净 | ✅ 干净 | ✅ 零风险 |

### 四、FC_AMP 算子实现

#### 4.1 总体架构

```
FC_FP32_FWD/BWD  →  fc_op.cu 手写 kernel   →  workspace: 不需要   →  保持不变
FC_AMP_FWD/BWD   →  fc_op.cpp cuDNN FE    →  workspace: 需要      →  全新实现
```

#### 4.2 Per-handle Cache（完全对标 relu_op.cpp / tanh_op.cpp）

```cpp
#ifdef TR_USE_CUDA
namespace feg = cudnn_frontend::graph;

struct FcAmpFwdCache {
    cudnnHandle_t handle = nullptr;
    int64_t batch = -1, in_features = -1, out_features = -1;
    int64_t x_n_stride = -1, w_w_stride = -1, y_n_stride = -1;
    bool has_bias = false;
    std::shared_ptr<feg::Graph> graph;
    std::shared_ptr<feg::Tensor_attributes> x_attr, w_attr, b_attr, y_attr;
};

struct FcAmpBwdCache {
    cudnnHandle_t handle = nullptr;
    int64_t batch = -1, in_features = -1, out_features = -1;
    int64_t dy_n_stride = -1, w_w_stride = -1, dx_n_stride = -1;
    std::shared_ptr<feg::Graph> graph;
    std::shared_ptr<feg::Tensor_attributes> dy_attr, w_attr, dx_attr;
};

static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpFwdCache>>
    s_fc_amp_fwd_caches;
static std::unordered_map<cudnnHandle_t, std::unique_ptr<FcAmpBwdCache>>
    s_fc_amp_bwd_caches;
#endif
```

**设计依据**：
- `unordered_map<cudnnHandle_t, unique_ptr<Cache>>` 是 relu_op.cpp 和 tanh_op.cpp 验证成功的多 GPU 安全模式
- Cache 按 `handle` 而非进程级单例索引，解决 multi-GPU 下 graph 析构冲突
- `bool has_bias` 参与 rebuild 判定：bias 有无改变 graph 结构，必须重建

#### 4.3 Forward Graph：`Y = Matmul(X, W) + bias`

```
IO 类型:      HALF  (FP16)
Intermediate: FLOAT (FP32)
Compute:      FLOAT (FP32, 利用 Tensor Core)
Bias 类型:    FLOAT (FP32) ← FFF.md #6 硬性要求

cuDNN 3D 布局（cuDNN Matmul 要求 3D 输入）：
  X: {batch,  1, in_features}     stride: {x_n_stride, in_features, 1}
  W: {1, in_features, out_features} stride: {in_features*w_w_stride, w_w_stride, 1}
  B: {1, 1, out_features}         data_type = FLOAT (独立于 IO 类型！)
  Y: {batch, 1, out_features}     stride: {y_n_stride, out_features, 1}
```

**stride 来源**：使用框架 API `dt_x.n_stride_cuda()`、`dt_w.w_stride_cuda()`、`dt_y.n_stride_cuda()`，**不**使用旧版 `get_row_stride()` 手工对齐。这是新版算子的统一规范，与 relu_op / tanh_op 一致。

```cpp
static std::shared_ptr<feg::Graph> build_fc_amp_fwd_graph(
    cudnnHandle_t handle,
    int64_t batch, int64_t in_features, int64_t out_features,
    int64_t x_n_stride, int64_t w_w_stride, int64_t y_n_stride,
    bool has_bias,
    std::shared_ptr<feg::Tensor_attributes>& out_x,
    std::shared_ptr<feg::Tensor_attributes>& out_w,
    std::shared_ptr<feg::Tensor_attributes>& out_b,
    std::shared_ptr<feg::Tensor_attributes>& out_y)
{
    auto g = std::make_shared<feg::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
     .set_intermediate_data_type(fe::DataType_t::FLOAT)
     .set_compute_data_type(fe::DataType_t::FLOAT);

    // X: {batch, 1, in_features}
    out_x = g->tensor(feg::Tensor_attributes()
        .set_name("x")
        .set_dim({batch, 1, in_features})
        .set_stride({x_n_stride, in_features, 1})
        .set_data_type(fe::DataType_t::HALF));

    // W: {1, in_features, out_features}
    out_w = g->tensor(feg::Tensor_attributes()
        .set_name("w")
        .set_dim({1, in_features, out_features})
        .set_stride({in_features * w_w_stride, w_w_stride, 1})
        .set_data_type(fe::DataType_t::HALF));

    auto mm_attr = feg::Matmul_attributes()
        .set_name("FC_Matmul")
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Y = g->matmul(out_x, out_w, mm_attr);
    Y->set_name("Y_mm")
      .set_dim({batch, 1, out_features})
      .set_stride({y_n_stride, out_features, 1})
      .set_data_type(fe::DataType_t::HALF);

    if (has_bias) {
        // 【FFF.md #6】Bias 永远是 FP32，即使在 AMP 模式下
        out_b = g->tensor(feg::Tensor_attributes()
            .set_name("bias")
            .set_dim({1, 1, out_features})
            .set_stride({out_features, out_features, 1})
            .set_data_type(fe::DataType_t::FLOAT));

        auto add_attr = feg::Pointwise_attributes()
            .set_name("FC_AddBias")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        auto Y_final = g->pointwise(Y, out_b, add_attr);
        Y_final->set_name("Y")
                .set_dim({batch, 1, out_features})
                .set_stride({y_n_stride, out_features, 1})
                .set_data_type(fe::DataType_t::HALF)
                .set_output(true);
        out_y = Y_final;
    } else {
        Y->set_output(true);
        out_y = Y;
    }

    TR_CUDNN_FE_CHECK(g->validate(), "validate FC AMP FWD");
    TR_CUDNN_FE_CHECK(g->build_operation_graph(handle), "build FC op graph");
    TR_CUDNN_FE_CHECK(g->create_execution_plans(
        {fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}), "create FC plans");
    TR_CUDNN_FE_CHECK(g->check_support(handle), "check FC support");
    TR_CUDNN_FE_CHECK(g->build_plans(
        fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "build FC plans");

    return g;
}
```

#### 4.4 Backward Graph：`dX = Matmul(dY, W)`

```
dX = dY @ W  其中 W.shape = [out_features, in_features]

dY: {batch, 1, out_features}    stride: {dy_n_stride, out_features, 1}
W:  {1, out_features, in_features}  stride: {out_features*w_w_stride, w_w_stride, 1}
dX: {batch, 1, in_features}     stride: {dx_n_stride, in_features, 1}
```

BWD 不需要 bias。`dW` 的计算由框架统一处理（不在本方案范围）。

```cpp
static std::shared_ptr<feg::Graph> build_fc_amp_bwd_graph(
    cudnnHandle_t handle,
    int64_t batch, int64_t in_features, int64_t out_features,
    int64_t dy_n_stride, int64_t w_w_stride, int64_t dx_n_stride,
    std::shared_ptr<feg::Tensor_attributes>& out_dy,
    std::shared_ptr<feg::Tensor_attributes>& out_w,
    std::shared_ptr<feg::Tensor_attributes>& out_dx)
{
    auto g = std::make_shared<feg::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
     .set_intermediate_data_type(fe::DataType_t::FLOAT)
     .set_compute_data_type(fe::DataType_t::FLOAT);

    out_dy = g->tensor(feg::Tensor_attributes()
        .set_name("dy")
        .set_dim({batch, 1, out_features})
        .set_stride({dy_n_stride, out_features, 1})
        .set_data_type(fe::DataType_t::HALF));

    out_w = g->tensor(feg::Tensor_attributes()
        .set_name("w")
        .set_dim({1, out_features, in_features})
        .set_stride({out_features * w_w_stride, w_w_stride, 1})
        .set_data_type(fe::DataType_t::HALF));

    auto mm_attr = feg::Matmul_attributes()
        .set_name("FC_Bwd_Matmul")
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto dX = g->matmul(out_dy, out_w, mm_attr);
    dX->set_name("dx")
       .set_dim({batch, 1, in_features})
       .set_stride({dx_n_stride, in_features, 1})
       .set_data_type(fe::DataType_t::HALF)
       .set_output(true);

    TR_CUDNN_FE_CHECK(g->validate(), "validate FC AMP BWD");
    TR_CUDNN_FE_CHECK(g->build_operation_graph(handle), "build FC BWD op graph");
    TR_CUDNN_FE_CHECK(g->create_execution_plans(
        {fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}), "create FC BWD plans");
    TR_CUDNN_FE_CHECK(g->check_support(handle), "check FC BWD support");
    TR_CUDNN_FE_CHECK(g->build_plans(
        fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "build FC BWD plans");

    return g;
}
```

#### 4.5 Launch 函数

```cpp
static void launch_fc_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<FCParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_AMP_FWD missing FCParams");
    TR_CHECK(node.input_ids.size() >= 2, ShapeError,
             "FC_AMP_FWD requires at least 2 inputs (x, w)");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);

    __half* x = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    __half* w = static_cast<__half*>(ctx.ptr_at(node.input_ids[1]));
    __half* y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    float* b = nullptr;
    if (p->bias && node.input_ids.size() > 2) {
        b = static_cast<float*>(ctx.ptr_at(node.input_ids[2]));
    }

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(
        ctx.cudnn_handle(StreamKind::COMP_1));

    int64_t batch        = dt_x.n();
    int64_t in_features  = dt_x.h() * dt_x.w() * dt_x.c();
    int64_t out_features = p->out_features;
    int64_t x_ns = dt_x.n_stride_cuda();
    int64_t w_ws = dt_w.w_stride_cuda();
    int64_t y_ns = dt_y.n_stride_cuda();

    auto it = s_fc_amp_fwd_caches.find(handle);
    bool rebuild = (it == s_fc_amp_fwd_caches.end())
        || it->second->batch != batch
        || it->second->in_features != in_features
        || it->second->out_features != out_features
        || it->second->x_n_stride != x_ns
        || it->second->w_w_stride != w_ws
        || it->second->y_n_stride != y_ns
        || it->second->has_bias != p->bias;

    if (rebuild) {
#ifndef NDEBUG
        {
            cudaStreamCaptureStatus cap_status;
            cudaError_t cap_err = cudaStreamIsCapturing(s, &cap_status);
            if (cap_err == cudaSuccess && cap_status != cudaStreamCaptureStatusNone) {
                TR_LOG_ERROR("fc") << "[AMP FWD] Rebuilding cuDNN FE graph "
                                   << "inside CUDA Graph capture! handle="
                                   << handle;
            }
        }
#endif
        auto cache = std::make_unique<FcAmpFwdCache>();
        cache->handle = handle;
        cache->batch = batch;
        cache->in_features = in_features;
        cache->out_features = out_features;
        cache->x_n_stride = x_ns;
        cache->w_w_stride = w_ws;
        cache->y_n_stride = y_ns;
        cache->has_bias = p->bias;

        cache->graph = build_fc_amp_fwd_graph(
            handle, batch, in_features, out_features,
            x_ns, w_ws, y_ns, p->bias,
            cache->x_attr, cache->w_attr, cache->b_attr, cache->y_attr);

        size_t ws_needed = cache->graph->get_workspace_size();
        if (ws_needed > 0) {
            ctx.ensure_workspace_grow(StreamKind::COMP_1, ws_needed);
        }

        s_fc_amp_fwd_caches[handle] = std::move(cache);
    }

    auto& cache = *s_fc_amp_fwd_caches[handle];
    std::unordered_map<std::shared_ptr<feg::Tensor_attributes>, void*> vp = {
        {cache.x_attr, x}, {cache.w_attr, w}, {cache.y_attr, y}
    };
    if (p->bias && b) vp[cache.b_attr] = b;

    void* ws = ctx.workspace(StreamKind::COMP_1);
    TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ws), "execute FC AMP FWD");
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

**BWD launch 同理**，省略重复代码，结构完全对标。

#### 4.6 CPU 不支持路径

AMP 算子仅在 CUDA 上可用（FP16 无 CPU 支持）：

```cpp
static void launch_fc_amp_cpu_not_supported(CpuOpContext* ctx) {
    (void)ctx;
    TR_TYPE_ERROR("FC_AMP is not supported on CPU (FP16 not available)");
}
```

### 五、集成改动

#### 5.1 `require_warmup()` 添加 FC_AMP

[op_registry.cpp](file:///r:/renaissance/src/backend/op_registry.cpp)：

```cpp
case ComputeOp::FC_AMP_FWD:
case ComputeOp::FC_AMP_BWD:
    return true;
```

#### 5.2 `register_op_fc()` 分拆注册

[fc_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp)：

```cpp
void register_op_fc() {
    // ---- FC_FP32_FWD/BWD: 保持不变，指向手写 kernel ----
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_FP32_FWD)];
        e.op = ComputeOp::FC_FP32_FWD;
        e.launch_cpu = launch_fc_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_fc_fwd_cuda;
#endif
    }
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_FP32_BWD)];
        e.op = ComputeOp::FC_FP32_BWD;
        e.launch_cpu = launch_fc_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_fc_bwd_cuda;
#endif
    }

    // ---- FC_AMP_FWD/BWD: 指向新的 cuDNN FE launch 函数 ----
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_AMP_FWD)];
        e.op = ComputeOp::FC_AMP_FWD;
        e.launch_cpu = launch_fc_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_fc_amp_fwd_cuda;
#endif
    }
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_AMP_BWD)];
        e.op = ComputeOp::FC_AMP_BWD;
        e.launch_cpu = launch_fc_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_fc_amp_bwd_cuda;
#endif
    }
}
```

#### 5.3 不需要改的文件

| 文件 | 为什么不改 |
|------|-----------|
| `captured_graph.cpp` | 无需 Phase B2.5 插入——warmup 中的 `ensure_workspace_grow` 已自动完成分配 |
| `task_base.cpp` | Task 零介入 workspace —— DeviceContext 全权管理 |
| `capture_cuda.cpp` | 通用 `entry.launch_cuda()` 路径无需特殊处理 |
| `fc_op.cu` | FP32 手写 kernel 不变 |
| `op_registry.cpp` | 不需要中央收集器——只有 FC 需要 workspace，集中式管理过度设计 |

### 六、改单清单

| 步骤 | 文件 | 内容 | 改动量 |
|------|------|------|--------|
| 1 | `include/renaissance/backend/device_context.h` | `workspaces_` 加 `mutable`；声明 `ensure_workspace_grow(kind, size) const` | 2行 |
| 2 | `src/backend/device_context.cpp` | 实现 `ensure_workspace_grow` | ~45行 |
| 3 | `src/backend/ops/dtensor/fc_op.cpp` | 新增 `FcAmpFwdCache`/`FcAmpBwdCache` + static cache map | ~30行 |
| 4 | `src/backend/ops/dtensor/fc_op.cpp` | 新增 `build_fc_amp_fwd_graph()` | ~55行 |
| 5 | `src/backend/ops/dtensor/fc_op.cpp` | 新增 `build_fc_amp_bwd_graph()` | ~40行 |
| 6 | `src/backend/ops/dtensor/fc_op.cpp` | 新增 `launch_fc_amp_fwd_cuda()` | ~70行 |
| 7 | `src/backend/ops/dtensor/fc_op.cpp` | 新增 `launch_fc_amp_bwd_cuda()` | ~60行 |
| 8 | `src/backend/ops/dtensor/fc_op.cpp` | 新增 `launch_fc_amp_cpu_not_supported()` | ~4行 |
| 9 | `src/backend/ops/dtensor/fc_op.cpp` | 分拆 `register_op_fc()`（AMP→新launch） | ~20行 |
| 10 | `src/backend/op_registry.cpp` | `require_warmup()` 添加 `FC_AMP_FWD/BWD` | 2行 |
| 11 | 编译 + 测试 | FC FP32 + FC AMP，单 GPU + 多 GPU | — |

**总计**：约 320 行新增代码，2 个文件基础设施改动，1 个文件核心实现。

### 七、方案对比：为什么这是最优解

| 维度 | 小伙伴 S（FFF4 混合） | 小伙伴 K（FFF4 纯Lazy） | 小伙伴 D（FFF3 const法） | **本方案** |
|------|----------------------|------------------------|-------------------------|-----------|
| 管理主体 | ✅ 纯 DC | ✅ 纯 DC | ✅ 纯 DC | ✅ 纯 DC |
| Task 介入 | ✅ 零介入 | ✅ 零介入 | ✅ 零介入 | ✅ 零介入 |
| 中央收集器 | ❌ 保留 | ✅ 不需要 | ✅ 不需要 | ✅ 不需要 |
| 初始浪费 | ✅ 零 | ✅ 零 | ✅ 零 | ✅ 零 |
| compile 流程改动 | ❌ 需插入 B2.5 | ✅ 零改动 | ✅ 零改动 | ✅ 零改动 |
| const 安全性 | ❌ `const_cast<DC&>(ctx)` 外露 | ❌ 编译错误 | ✅ const 方法 | ✅ `mutable` + const 方法 |
| 代码侵入 | 中 | 最低 | 最低 | ✅ 最低 |

本方案实质上是**小伙伴 K 的"算子级 Lazy Grow"逻辑 + 小伙伴 D 的 const 安全性设计**的融合，同时采纳了小伙伴 S 关于精确按需的核心主张。三人方案的优势全部保留，各自的缺陷全部避开。

### 八、核心口诀

```
DeviceContext 全权管，
算子自己按需 grow，
Task 完全不介入，
Compile 流程零改动，
mutable 保证 const 安全，
一次分配终生复用。
```

### 九、实施步骤（建议顺序）

1. **第1步**：`device_context.h` — 加 `mutable` + 声明 `ensure_workspace_grow`
2. **第2步**：`device_context.cpp` — 实现 `ensure_workspace_grow`
3. **第3步**：`fc_op.cpp` — 实现 cache 结构 + `build_fc_amp_fwd_graph` + `launch_fc_amp_fwd_cuda`
4. **第4步**：`fc_op.cpp` — 实现 `build_fc_amp_bwd_graph` + `launch_fc_amp_bwd_cuda`
5. **第5步**：`fc_op.cpp` — 添加 CPU not-supported stub + 分拆 `register_op_fc`
6. **第6步**：`op_registry.cpp` — `require_warmup` 添加 FC_AMP
7. **第7步**：编译 → 单 GPU FC_FP32 测试 → 单 GPU FC_AMP 测试 → 多 GPU 测试

### 十、与 relu_op.cpp 的对标验证

| 组件 | relu_op.cpp | **fc_op.cpp（新）** | 一致性 |
|------|-------------|---------------------|--------|
| Cache 结构 | `ReluAmpCache` per handle | `FcAmpFwdCache`/`FcAmpBwdCache` per handle | ✅ |
| Cache 容器 | `unordered_map<cudnnHandle_t, unique_ptr<Cache>>` | 同 | ✅ |
| launch_cuda 签名 | `(const GraphNode&, const MemoryPlan&, const DeviceContext&, MCState&)` | 同 | ✅ |
| rebuild 判定 | shape/stride 变化检测 | 同 + has_bias | ✅ |
| `#ifndef NDEBUG` 检测 | 有 | 有 | ✅ |
| cuDNN FE validate/build | 有 | 有 | ✅ |
| workspace 参数 | `nullptr`（不需要） | `ctx.workspace(COMP_1)` | ⚠️ 唯一差异 |
| CPU 不支持路径 | `TR_TYPE_ERROR` | 同 | ✅ |

**唯一差异**是 `graph->execute(handle, vp, workspace_ptr)` 传的是 `ctx.workspace()` 而非 `nullptr`。这是 FC Matmul 的内在需求，而非架构偏差。