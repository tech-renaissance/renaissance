# LPD2：LARS 与 LARS_NESTEROV 完整测试样例实现方案

> 调查日期：2026-05-21
> 调查范围：LARD.md、tests/correction/*、src/backend/ops/*、src/graph/compiler.cpp、src/graph/capture_*.cpp
> 结论：Backend Kernel 是唯一阻塞项；SimpleTask 已完全支持 ComputeOp 测试。

---

## 一、现状调查

### 1.1 已就绪的设施

| 设施 | 位置 | 状态 |
|------|------|------|
| **LARS ComputeOp 枚举** | `include/renaissance/graph/op_kind.h:234-237` | ✅ `LARS_UPDATE`、`LARS_NESTEROV_UPDATE`、`LARS_COMPUTE_TRUST_RATIO` |
| **LARS 配置系统** | `include/renaissance/algo/optimizer.h:90-401` | ✅ `LARSConfig` 含 `trust_coefficient`、`eps`、`momentum`、`weight_decay` |
| **编译器图生成** | `src/graph/compiler.cpp:1392-1434` | ✅ 自动注入 `LARS_COMPUTE_TRUST_RATIO` + `LARS_UPDATE` 节点 |
| **N-Series 内存区域** | `include/renaissance/core/types.h:302-304` | ✅ `N_FC_WEIGHT`、`N_FIRST_CONV`、`N_DEEP_CONV` |
| **MemoryPlan 分配** | `src/graph/memory_plan.cpp:322-333` | ✅ `alloc_norm_fc_weight()` 等接口已就绪 |
| **CPU/GPU 捕获路径** | `src/graph/capture_cpu.cpp:34` / `capture_cuda.cpp:104` | ✅ **COMPUTE 节点在 CPU 和 CUDA Graph 捕获中均已完整支持** |
| **SimpleTask ComputeOp 支持** | `src/graph/capture_cpu.cpp:31` / `capture_cuda.cpp:47` | ✅ 优先使用 `cg.linear_nodes()`，SimpleTask 手动绘图完全兼容 |

### 1.2 唯一缺失：Backend Kernel 实现

`src/backend/ops/` 下搜索 `LARS` → **0 hits**。

| 缺失项 | 影响 |
|--------|------|
| 无 `lars_op.cpp` / `lars_op.cu` | 3 个 ComputeOp 无 CPU/CUDA launch 函数 |
| 无算子表注册 | `op_registry.cpp:39-54` 的 `register_default_ops()` 未调用 `register_op_lars()` |
| 无 reduce kernel | `infra_kernels.cu` 只有 scale / philox，无 sum-of-squares reduction |

> **关键发现**：`capture_cuda.cpp:104-110` 明确处理 `GraphNode::Kind::COMPUTE`，调用 `entry.launch_cuda(node, mp, ctx, state)`；`capture_cpu.cpp:34-70` 同理。这意味着 **SimpleTask 的 CUDA Graph / CPU 路径已经天然支持 ComputeOp 测试**，只要 backend 把 launch 函数注册进 `g_compute_op_table` 即可。

---

## 二、测试样例设计

### 2.1 参考模板：test_nesterov_weight.cpp

现有 9 个测试的结构：

```cpp
// 1. alloc W/G/M/V + scalar(lr,wd,beta,...)
// 2. finalize_memory()
// 3. ComputationGraph g { GraphNode(RANGE); input_ranges/output_ranges; input_ids }
// 4. task.add_graph(name, g, StreamKind::UPDATE)
// 5. compile()
// 6. transfer host data → device
// 7. task.run(name)
// 8. fetch result → compare with host reference
```

### 2.2 LARS 测试的核心差异

LARS 是 **ComputeOp（逐 DTensor）**，不是 RangeOp（逐 Region）：

| 差异点 | RangeOp（如 Nesterov） | ComputeOp（LARS） |
|--------|------------------------|-------------------|
| `node.kind` | `GraphNode::Kind::RANGE` | `GraphNode::Kind::COMPUTE` |
| 操作字段 | `node.range_op` | `node.compute_op` |
| 输入/输出 | `input_ranges` / `output_ranges`（Region 范围） | `input_ids` / `output_ids`（DTensor ID 列表） |
| 是否需要 `mp.region_range()` | ✅ 是 | ❌ 否，直接用 DTensor ID |
| Graph 节点数 | 1 个节点覆盖全部 | 2N 个节点（每层：trust_ratio + update） |

### 2.3 test_lars_weight.cpp 结构

```cpp
int main(int argc, char* argv[]) {
    // --cpu / --gpu 参数解析（同现有 9 个测试）
    // ...

    SimpleTask task;
    Shape shape_a{4, 8, 4, 4};   // 512 elts
    Shape shape_b{8, 4, 8, 2};   // 512 elts

    // 1. 分配 W / G / M / N（N region 存 trust ratio，标量）
    DTensor d_w_a = task.alloc(shape_a, DType::FP32, Region::W_FC_WEIGHT);
    DTensor d_w_b = task.alloc(shape_b, DType::FP32, Region::W_FC_WEIGHT);
    DTensor d_g_a = task.alloc(shape_a, DType::FP32, Region::G_FC_WEIGHT);
    DTensor d_g_b = task.alloc(shape_b, DType::FP32, Region::G_FC_WEIGHT);
    DTensor d_m_a = task.alloc(shape_a, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_m_b = task.alloc(shape_b, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_n_a = task.alloc(Shape{1}, DType::FP32, Region::N_FC_WEIGHT);
    DTensor d_n_b = task.alloc(Shape{1}, DType::FP32, Region::N_FC_WEIGHT);

    // 2. 分配标量：lr, beta, tc, wd, eps
    DTensor d_lr   = task.alloc_scalar(DType::FP32);
    DTensor d_beta = task.alloc_scalar(DType::FP32);
    DTensor d_tc   = task.alloc_scalar(DType::FP32);
    DTensor d_wd   = task.alloc_scalar(DType::FP32);
    DTensor d_eps  = task.alloc_scalar(DType::FP32);

    task.finalize_memory();

    // 3. 构建 ComputationGraph（COMPUTE 节点，非 RANGE）
    ComputationGraph g;
    {
        // Step 1: LARS_COMPUTE_TRUST_RATIO for dt_a
        GraphNode n1;
        n1.kind = GraphNode::Kind::COMPUTE;
        n1.compute_op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
        n1.input_ids  = {d_w_a.id, d_g_a.id, d_tc.id, d_wd.id, d_eps.id};
        n1.output_ids = {d_n_a.id};
        g.append(std::move(n1));

        // Step 2: LARS_UPDATE for dt_a
        GraphNode n2;
        n2.kind = GraphNode::Kind::COMPUTE;
        n2.compute_op = ComputeOp::LARS_UPDATE;
        n2.input_ids  = {d_w_a.id, d_g_a.id, d_m_a.id, d_n_a.id,
                         d_lr.id, d_beta.id, d_wd.id};
        n2.output_ids = {d_w_a.id, d_m_a.id};
        g.append(std::move(n2));

        // Step 3: LARS_COMPUTE_TRUST_RATIO for dt_b
        GraphNode n3;
        n3.kind = GraphNode::Kind::COMPUTE;
        n3.compute_op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
        n3.input_ids  = {d_w_b.id, d_g_b.id, d_tc.id, d_wd.id, d_eps.id};
        n3.output_ids = {d_n_b.id};
        g.append(std::move(n3));

        // Step 4: LARS_UPDATE for dt_b
        GraphNode n4;
        n4.kind = GraphNode::Kind::COMPUTE;
        n4.compute_op = ComputeOp::LARS_UPDATE;
        n4.input_ids  = {d_w_b.id, d_g_b.id, d_m_b.id, d_n_b.id,
                         d_lr.id, d_beta.id, d_wd.id};
        n4.output_ids = {d_w_b.id, d_m_b.id};
        g.append(std::move(n4));
    }
    task.add_graph("lars_weight", std::move(g), StreamKind::UPDATE);
    task.compile();

    // 4. 初始化 host 数据并 transfer
    // ...

    // 5. run
    task.run("lars_weight");

    // 6. fetch & validate
    // ...
}
```

**关键**：`task.add_graph()` 的第二个参数 `StreamKind::UPDATE` 会作为该图的 capture stream。ComputeOp 的 launch 函数内部通过 `ctx.stream(StreamKind::COMP_1)` 获取实际执行流，但 `StreamKind::UPDATE` 是 SimpleTask 的 primary capture stream，与现有测试保持一致。

### 2.4 test_lars_nesterov_weight.cpp 结构

与 `test_lars_weight.cpp` 几乎完全相同，仅两处不同：

1. `n2.compute_op = ComputeOp::LARS_NESTEROV_UPDATE;`
2. `n4.compute_op = ComputeOp::LARS_NESTEROV_UPDATE;`

### 2.5 主机参考公式（来自 LARD.md / LAR_FINAL3.md）

**LARS Weight**：

```
Step 1: η = tc · ‖W‖₂ / (‖G‖₂ + wd·‖W‖₂ + ε)
        若 ‖W‖₂ < 1e-12 或 ‖G‖₂ < 1e-12: η = 1.0
        η = min(η, 100.0)

Step 2: G' = G + wd·W

Step 3: M_new = β · M + η · G'

Step 4: W = W - lr · M_new
```

**LARS_NESTEROV Weight**：

```
Step 1~3 同上

Step 4: W = W - lr · (η·G' + β·M_new)
        = W - lr · (β²·M + (1+β)·η·G')
```

主机参考实现（C++）：

```cpp
auto expected_lars = [&](const Tensor& w, const Tensor& g,
                         const Tensor& m_in) -> Tensor {
    // Step 1: compute norms & trust ratio
    double sum_w_sq = 0.0, sum_g_sq = 0.0;
    for (int64_t i = 0; i < w.numel(); ++i) {
        float wv = w.data<float>()[i];
        float gv = g.data<float>()[i];
        sum_w_sq += (double)wv * (double)wv;
        sum_g_sq += (double)gv * (double)gv;
    }
    double w_norm = std::sqrt(sum_w_sq);
    double g_norm = std::sqrt(sum_g_sq);
    double eta = 1.0;
    if (w_norm >= 1e-12 && g_norm >= 1e-12) {
        eta = (double)tc_val * w_norm / (g_norm + (double)wd_val * w_norm + (double)eps_val);
        eta = std::min(eta, 100.0);
    }

    // Step 2~4: element-wise update
    Tensor e_w(w.shape(), DType::FP32);
    Tensor e_m(w.shape(), DType::FP32);
    for (int64_t i = 0; i < w.numel(); ++i) {
        float wv = w.data<float>()[i];
        float gv = g.data<float>()[i];
        float mv = m_in.data<float>()[i];
        float gp = gv + wd_val * wv;                    // G'
        float m_new = beta_val * mv + (float)eta * gp;  // M_new
        e_w.data<float>()[i] = wv - lr_val * m_new;     // W update
        e_m.data<float>()[i] = m_new;                   // M update
    }
    return e_w;  // 验证 W 即可，M 可额外验证
};
```

> **注意**：主机使用 `double` 做 reduction 累加，避免 FP32 累加顺序差异导致的误判。GPU kernel 内用 `float` 累加，最终 `max|diff|` 可能略高于纯 element-wise 测试（因为有 reduce 顺序差异），但仍应在 `1e-5` 以内。

---

## 三、Backend Kernel 实现方案

### 3.1 文件清单

```
src/backend/ops/dtensor/
├── lars_op.cpp      # CPU launch + 算子表注册
├── lars_op.cu       # CUDA kernel + CUDA launch
```

```
src/backend/op_registry.cpp   # register_default_ops() 追加 register_op_lars();
```

### 3.2 Kernel 1: LARS_COMPUTE_TRUST_RATIO

**输入**：W, G（device float*），tc, wd, eps（标量 device float*）
**输出**：η（device float*，单个元素）

**CUDA Kernel**：

```cpp
__global__ void lars_trust_ratio_kernel(
    const float* __restrict__ w,
    const float* __restrict__ g,
    float* __restrict__ out_eta,
    size_t n,
    float tc, float wd, float eps)
{
    __shared__ float sm[2];  // sm[0]=sum_w_sq, sm[1]=sum_g_sq
    if (threadIdx.x == 0) { sm[0] = 0.0f; sm[1] = 0.0f; }
    __syncthreads();

    float local_w = 0.0f, local_g = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i];
        float gv = g[i];
        local_w += wv * wv;
        local_g += gv * gv;
    }

    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_w += __shfl_down_sync(0xFFFFFFFF, local_w, offset);
        local_g += __shfl_down_sync(0xFFFFFFFF, local_g, offset);
    }

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&sm[0], local_w);
        atomicAdd(&sm[1], local_g);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float w_norm = sqrtf(sm[0]);
        float g_norm = sqrtf(sm[1]);
        float eta = 1.0f;
        if (w_norm >= 1e-12f && g_norm >= 1e-12f) {
            eta = tc * w_norm / (g_norm + wd * w_norm + eps);
            if (eta > 100.0f) eta = 100.0f;
        }
        *out_eta = eta;
    }
}
```

**设计说明**：
- 不用 CUB / Thrust：保持与现有 5 个 optimizer kernel 风格一致，零外部依赖
- `atomicAdd` 到 shared mem：每个 block 2 次，grid 最多 65535 blocks，竞争可忽略
- `__shfl_down_sync`：Warp 级 reduce，比 shared mem 更快

**CPU Fallback**：

```cpp
static void launch_lars_trust_ratio_cpu(CpuOpContext* ctx) {
    const float* w = ...;  // 从 ctx->input_ids[0] 解析
    const float* g = ...;
    float* eta_out = ...;
    float tc  = *scalar_ptr(ctx, 2);
    float wd  = *scalar_ptr(ctx, 3);
    float eps = *scalar_ptr(ctx, 4);
    int64_t n = ctx->total_elements;

    double sum_w_sq = 0.0, sum_g_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum_w_sq += (double)w[i] * (double)w[i];
        sum_g_sq += (double)g[i] * (double)g[i];
    }
    double w_norm = std::sqrt(sum_w_sq);
    double g_norm = std::sqrt(sum_g_sq);
    double eta = 1.0;
    if (w_norm >= 1e-12 && g_norm >= 1e-12) {
        eta = tc * w_norm / (g_norm + wd * w_norm + eps);
        eta = std::min(eta, 100.0);
    }
    *eta_out = (float)eta;
}
```

### 3.3 Kernel 2: LARS_UPDATE

**输入**：W, G, M, η, lr, β, wd
**输出**：W, M

```cpp
__global__ void lars_update_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, size_t n,
    const float* __restrict__ lr,
    const float* __restrict__ beta,
    const float* __restrict__ wd,
    const float* __restrict__ eta)
{
    float _lr   = *lr;
    float _beta = *beta;
    float _wd   = wd ? *wd : 0.0f;
    float _eta  = *eta;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i];
        float gv = g[i];
        float gp = gv + _wd * wv;                // G'
        float m_new = _beta * m[i] + _eta * gp;  // M_new
        w[i] = wv - _lr * m_new;                 // W update
        m[i] = m_new;                            // M update
    }
}
```

### 3.4 Kernel 3: LARS_NESTEROV_UPDATE

```cpp
__global__ void lars_nesterov_update_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, size_t n,
    const float* __restrict__ lr,
    const float* __restrict__ beta,
    const float* __restrict__ wd,
    const float* __restrict__ eta)
{
    float _lr   = *lr;
    float _beta = *beta;
    float _wd   = wd ? *wd : 0.0f;
    float _eta  = *eta;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i];
        float gv = g[i];
        float gp = gv + _wd * wv;                     // G'
        float m_new = _beta * m[i] + _eta * gp;       // M_new
        m[i] = m_new;
        // W = W - lr * (η·G' + β·M_new)
        w[i] = wv - _lr * (_eta * gp + _beta * m_new);
    }
}
```

### 3.5 CUDA Launch 函数签名

参考 `axpy_op.cpp` 的 `launch_axpy_fwd_cuda` 模式：

```cpp
static void launch_lars_trust_ratio_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    // 1. 从 node.input_ids[0/1] 解析 w/g 的 device ptr
    // 2. 从 node.input_ids[2/3/4] 解析 tc/wd/eps 标量 ptr（通过 ctx.ptr_at()）
    // 3. 从 node.output_ids[0] 解析 eta 输出 ptr
    // 4. 获取元素数 n = mp.get_dtensor(node.input_ids[0]).numel()
    // 5. 选择 StreamKind::UPDATE 流
    // 6. launch kernel
}
```

> **Stream 选择**：LARS 更新走 `StreamKind::UPDATE`（与现有 optimizer RangeOp 一致）。`LARS_COMPUTE_TRUST_RATIO` 的归约与 `LARS_UPDATE` 的 elementwise 可以在同一流上顺序执行，无需跨流同步。

### 3.6 算子表注册

```cpp
void register_op_lars() {
    // LARS_COMPUTE_TRUST_RATIO
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_COMPUTE_TRUST_RATIO)];
        e.op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
        e.launch_cpu = launch_lars_trust_ratio_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_lars_trust_ratio_cuda;
#endif
    }
    // LARS_UPDATE
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_UPDATE)];
        e.op = ComputeOp::LARS_UPDATE;
        e.launch_cpu = launch_lars_update_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_lars_update_cuda;
#endif
    }
    // LARS_NESTEROV_UPDATE
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_NESTEROV_UPDATE)];
        e.op = ComputeOp::LARS_NESTEROV_UPDATE;
        e.launch_cpu = launch_lars_nesterov_update_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_lars_nesterov_update_cuda;
#endif
    }
    TR_LOG_DEBUG("backend") << "LARS operators registered (CPU+CUDA)";
}
```

并在 `op_registry.cpp:54` 的 `register_default_ops()` 末尾追加：

```cpp
register_op_lars();
```

---

## 四、修改清单（按优先级）

| 优先级 | 文件 | 修改内容 | 行数估计 |
|--------|------|----------|----------|
| **P0** | `src/backend/ops/dtensor/lars_op.cu` | 3 个 CUDA kernel + 3 个 launch 函数 | ~120 |
| **P0** | `src/backend/ops/dtensor/lars_op.cpp` | 3 个 CPU launch + `register_op_lars()` | ~150 |
| **P0** | `src/backend/op_registry.cpp` | `register_default_ops()` 追加 `register_op_lars()` | ~1 |
| **P1** | `tests/correction/test_lars_weight.cpp` | LARS 测试样例（2 DTensor × 2 ComputeOp） | ~160 |
| **P1** | `tests/correction/test_lars_nesterov_weight.cpp` | LARS_NESTEROV 测试样例 | ~160 |
| **P2** | `tests/correction/CMakeLists.txt` | 追加 `test_lars_weight`、`test_lars_nesterov_weight` | ~20 |
| **P2** | `src/backend/CMakeLists.txt` | 若未用 glob，追加 `lars_op.cpp` / `lars_op.cu` | ~2 |

---

## 五、风险与验证计划

| 风险 | 说明 | 缓解措施 |
|------|------|----------|
| **R1: GPU reduce 顺序差异** | GPU block-level 并行 reduce 与 CPU 顺序累加结果可能差 `~1e-7` | 阈值维持 `1e-5`；CPU 路径用 `double` 累加 |
| **R2: `||W||₂ < 1e-12` fallback** | 若测试数据全 0，`η=1.0` 的 fallback 路径必须被覆盖 | 测试数据用非零 fill（如 0.5f），同时加一组边界值测试 |
| **R3: N region 数据生命周期** | `d_n_a` 在 `task.run()` 后是否仍有效？（是，DTensor 生命周期与 Task 绑定） | 无需处理 |
| **R4: ComputeOp 在 SimpleTask 中首次使用** | 现有 9 个测试全为 RangeOp，ComputeOp 测试路径未经验证 | 先跑 `--cpu` 验证 CPU 路径，再跑 `--gpu` 验证 CUDA Graph 路径 |
| **R5: 多 rank 标量广播** | tc/wd/eps 是标量，只需 rank 0 transfer，是否需 `broadcast_from_rank0`？ | 是，参照现有测试模式，所有标量都需 broadcast |

---

## 六、实施路线图

```
Phase 1: Backend Kernel（可独立编译验证）
  ├── lars_op.cu       → ninja 编译通过
  ├── lars_op.cpp      → ninja 编译通过
  └── op_registry.cpp  → 链接通过

Phase 2: 测试样例（依赖 Phase 1）
  ├── test_lars_weight.cpp          → --cpu PASS → --gpu PASS
  └── test_lars_nesterov_weight.cpp → --cpu PASS → --gpu PASS

Phase 3: 精度调优（若 max|diff| > 1e-5）
  ├── 检查 reduce 累加精度（是否需 Kahan / double buffer）
  └── 检查 sqrt / division 的 FP32 行为
```

---

## 七、结论

1. **框架的 LARS "图纸"已经画好**：编译器会生成图、内存已分配区域、配置系统完整、CUDA Graph / CPU 捕获路径均已支持 COMPUTE 节点。
2. **唯一阻塞项是 3 个 ComputeOp 的 backend kernel**：约 300 行代码（CUDA + CPU + 注册），参考 `axpy_op.cpp/cu` 的 ComputeOp 模式即可实现。
3. **范数必须求，但可以融合**：`LARS_COMPUTE_TRUST_RATIO` 一个 kernel 内完成 `‖W‖₂` + `‖G‖₂` + `η` 计算，无需独立范数算子，也无需中间存储。
4. **SimpleTask 已天然支持 ComputeOp 测试**：`capture_cpu.cpp` 和 `capture_cuda.cpp` 都完整处理了 `GraphNode::Kind::COMPUTE`，测试代码只需把 `node.kind` 设为 `COMPUTE`、用 `input_ids/output_ids` 代替 `input_ranges/output_ranges` 即可。
