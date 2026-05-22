# LX_FINAL：LARS / LARS_NESTEROV 完整实现方案

> 综合 LPD1 / LPD2 / LPD3 三份分析文档 + 代码核实
> 数学公式经 LAR_FINAL3.md CLOSED.py / OPEN.py 验证
> 2026-05-21

---

## 〇、核心结论

**Compiler 层图纸已画好。唯一阻塞项：3 个 ComputeOp 的 backend kernel 实现。**

| 层面 | 完成度 | 缺失 |
|------|--------|------|
| ComputeOp 枚举 + op_kind 字符串映射 | ✅ | — |
| OptimizerKind 枚举 + LARSConfig | ✅ | — |
| Compiler 图注入（LARS → 2N 个 COMPUTE 节点） | ✅ | — |
| MemoryPlan N_* 条件分配 | ✅ | — |
| ScalarIds（tc/wd/eps/lr/beta）| ✅ | — |
| Bias 复用（LARS Bias → RANGE_UPDATE_BIAS_MOMENTUM/NESTEROV） | ✅ | — |
| is_shape_invariant_graph 已移除 OPTIMIZER | ✅ | — |
| **Backend kernel** | ❌ | lars_op.cu/lars_op.cpp 不存在 |
| **op_registry 注册** | ❌ | register_op_lars() 未调用 |
| **正确性测试** | ❌ | test_lars_weight / test_lars_nesterov_weight 不存在 |

---

## 一、符号约定

| 符号 | 含义 | 在 kernel 中的表示 |
|------|------|-------------------|
| W    | 参数权重 | input_ids[0] |
| G    | 原始梯度（weight decay 还未施加） | input_ids[1] |
| M    | 动量缓冲（不含 lr 乘数） | input_ids[2] / output_ids[1] |
| η    | LARS trust ratio = tc·‖W‖₂ / (‖G‖₂ + wd·‖W‖₂ + ε) | input_ids[3] / N_* 标量 |
| tc   | trust coefficient | scalar_ids.tc |
| wd   | weight decay | scalar_ids.wd |
| lr   | learning rate | scalar_ids.lr |
| β    | momentum coefficient | scalar_ids.beta |
| ε    | 数值保护小量 | scalar_ids.eps |
| G'   | 施加 weight decay 后的梯度：G' = G + wd·W | 临时变量 |

---

## 二、数学公式

### 2.1 LARS_COMPUTE_TRUST_RATIO

```
sum_w² = Σ W[i]²,   sum_g² = Σ G[i]²
‖W‖₂ = sqrt(sum_w²),   ‖G‖₂ = sqrt(sum_g²)

if ‖W‖₂ < 1e-12  ||  ‖G‖₂ < 1e-12:
    η = 1.0
else:
    η = tc · ‖W‖₂ / (‖G‖₂ + wd · ‖W‖₂ + ε)
η = min(η, 100.0)
```

### 2.2 LARS_UPDATE

```
G' = G + wd · W
M_new = β · M_old + η · G'
W_new = W_old - lr · M_new
```

### 2.3 LARS_NESTEROV_UPDATE

```
G' = G + wd · W
M_new = β · M_old + η · G'
W_new = W_old - lr · (η · G' + β · M_new)
```

### 2.4 LARS Bias（退化）

LARS Bias 的 `trust_coefficient=0`，公式退化为纯 Momentum/Nesterov。因此直接复用现有 RangeOp：
- `RANGE_UPDATE_BIAS_MOMENTUM`（LARS）
- `RANGE_UPDATE_BIAS_NESTEROV`（LARS_NESTEROV）

Compiler 已在 [compiler.cpp:1505-1512](file:///r:/renaissance/src/graph/compiler.cpp#L1505-L1512) 正确映射，无需新 kernel。

---

## 三、Backend Kernel 实现

### 3.1 新增文件

```
src/backend/ops/dtensor/
├── lars_op.cu       ← 新增：3 个 CUDA kernel + 3 个 launch 函数（~120 行）
├── lars_op.cpp      ← 新增：3 个 CPU fallback + register_op_lars()（~150 行）
```

### 3.2 lars_op.cu — 三个 CUDA Kernel

参照 `optimizer_op.cu` 的简洁风格：无外部库依赖（不用 CUB/Thrust），无分支，per-kernel 独立。

**kernel 1: `lars_trust_ratio_kernel`**

```
__global__ void lars_trust_ratio_kernel(
    const float* __restrict__ w,        // input_ids[0]
    const float* __restrict__ g,        // input_ids[1]
    float* __restrict__ out_eta,        // output_ids[0]
    size_t n,
    const float* __restrict__ tc,       // input_ids[2] — device pointer
    const float* __restrict__ wd,       // input_ids[3]
    const float* __restrict__ eps)      // input_ids[4]
{
    float _tc = *tc, _wd = *wd, _eps = *eps;

    __shared__ float sm[2];             // sm[0] = sum_w², sm[1] = sum_g²
    if (threadIdx.x == 0) { sm[0] = 0.0f; sm[1] = 0.0f; }
    __syncthreads();

    // Step 1: grid-stride loop 累加
    float local_w2 = 0.0f, local_g2 = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i], gv = g[i];
        local_w2 += wv * wv;
        local_g2 += gv * gv;
    }

    // Step 2: warp-level reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_w2 += __shfl_down_sync(0xFFFFFFFF, local_w2, offset);
        local_g2 += __shfl_down_sync(0xFFFFFFFF, local_g2, offset);
    }

    // Step 3: block-level atomicAdd to shared memory
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&sm[0], local_w2);
        atomicAdd(&sm[1], local_g2);
    }
    __syncthreads();

    // Step 4: thread 0 计算 η
    if (threadIdx.x == 0) {
        float w_norm = sqrtf(sm[0]);
        float g_norm = sqrtf(sm[1]);
        float eta = 1.0f;
        if (w_norm >= 1e-12f && g_norm >= 1e-12f) {
            eta = _tc * w_norm / (g_norm + _wd * w_norm + _eps);
            if (eta > 100.0f) eta = 100.0f;
        }
        *out_eta = eta;
    }
}
```

**kernel 2: `lars_update_kernel`**

```
__global__ void lars_update_kernel(
    float* __restrict__ w,              // input_ids[0] / output_ids[0]
    const float* __restrict__ g,        // input_ids[1]
    float* __restrict__ m,              // input_ids[2] / output_ids[1]
    size_t n,
    const float* __restrict__ eta,      // input_ids[3]
    const float* __restrict__ lr,       // input_ids[4]
    const float* __restrict__ beta,     // input_ids[5]
    const float* __restrict__ wd)       // input_ids[6]
{
    float _lr = *lr, _beta = *beta, _wd = *wd, _eta = *eta;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i], gv = g[i];
        float gp = gv + _wd * wv;                  // G'
        float m_new = _beta * m[i] + _eta * gp;    // M_new
        w[i] = wv - _lr * m_new;                   // W update
        m[i] = m_new;                              // M update
    }
}
```

**kernel 3: `lars_nesterov_update_kernel`**

与 kernel 2 完全相同，唯更新公式一行不同：

```
// 最后一行为（代替 w[i] = wv - _lr * m_new）：
w[i] = wv - _lr * (_eta * gp + _beta * m_new);
```

### 3.3 lars_op.cpp — CPU Fallback + CUDA Launch + 注册

**3.3.1 CUDA launch 函数**（以 trust_ratio 为例）

```
static void launch_lars_trust_ratio_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    // 1. 解析 DTensor ID → ArenaKeeper ptr
    //    tc/wd/eps 都是 device pointer → 与 optimizer_op.cu 中 lr/wd/beta 一致
    const DTensor& w_dt  = mp.get_dtensor(node.input_ids[0]);
    const DTensor& g_dt  = mp.get_dtensor(node.input_ids[1]);

    size_t n = w_dt.numel();
    if (n == 0) return;

    const float* w   = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const float* g   = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    const float* tc  = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));
    const float* wd  = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));
    const float* eps = static_cast<const float*>(ctx.ptr_at(node.input_ids[4]));
    float* eta = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int block = 256;
    int grid = std::min(static_cast<int>((n + block - 1) / block), 65535);

    // 2. 选择 UPDATE stream（与 optimizer RangeOp 一致）
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    // 3. launch kernel — 标量以 device pointer 传入（与现有 5 个 optimizer kernel 一致）
    lars_trust_ratio_kernel<<<grid, block, 0, s>>>(w, g, eta, n, tc, wd, eps);

    // 4. record event for cross-op barrier
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

`launch_lars_update_cuda` 和 `launch_lars_nesterov_update_cuda` 同理，只是 input_ids 布局不同（多了 M_old、η、lr、beta 等）。

**3.3.2 CPU fallback**（用于 `--cpu` 测试路径）

> 注意：`capture_cpu.cpp:48-50` 中 `total_elements` 从 `output_ids[0]` 计算，
> 对 trust_ratio 这是 N_* 标量=1。正确做法是用 `input_shape` 从 `input_ids[0]`（W）获取元素数。

```
// trust_ratio CPU: double 累加 → sqrt → clamp → 写入标量
static void launch_lars_trust_ratio_cpu(CpuOpContext* ctx) {
    // 实际 API：const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[...])
    const float* w   = static_cast<const float*>(
        const_cast<DeviceContext*>(ctx->ctx)->ptr_at(ctx->input_ids[0]));
    const float* g   = static_cast<const float*>(
        const_cast<DeviceContext*>(ctx->ctx)->ptr_at(ctx->input_ids[1]));
    float tc  = *static_cast<const float*>(
        const_cast<DeviceContext*>(ctx->ctx)->ptr_at(ctx->input_ids[2]));
    float wd  = *static_cast<const float*>(
        const_cast<DeviceContext*>(ctx->ctx)->ptr_at(ctx->input_ids[3]));
    float eps = *static_cast<const float*>(
        const_cast<DeviceContext*>(ctx->ctx)->ptr_at(ctx->input_ids[4]));
    float* eta = static_cast<float*>(
        const_cast<DeviceContext*>(ctx->ctx)->ptr_at(ctx->output_ids[0]));

    // 用 input_shape 计算元素数（不是 total_elements，因为 trust_ratio 的
    // output 是标量 N_*，total_elements == 1）
    int64_t n = static_cast<int64_t>(ctx->input_shape.n) *
                static_cast<int64_t>(ctx->input_shape.h) *
                static_cast<int64_t>(ctx->input_shape.w) *
                static_cast<int64_t>(ctx->input_shape.c);

    double sum_w2 = 0.0, sum_g2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum_w2 += (double)w[i] * (double)w[i];
        sum_g2 += (double)g[i] * (double)g[i];
    }
    double w_norm = std::sqrt(sum_w2), g_norm = std::sqrt(sum_g2);
    double eta_val = 1.0;
    if (w_norm >= 1e-12 && g_norm >= 1e-12) {
        eta_val = (double)tc * w_norm /
                  (g_norm + (double)wd * w_norm + (double)eps);
        eta_val = std::min(eta_val, 100.0);
    }
    *eta = (float)eta_val;
}

// lars_update_cpu / lars_nesterov_update_cpu 同理（单线程 for 循环）
```

**3.3.3 注册函数**

```
void register_op_lars() {
    auto& t0 = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_COMPUTE_TRUST_RATIO)];
    t0.op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
    t0.launch_cpu = launch_lars_trust_ratio_cpu;
#ifdef TR_USE_CUDA
    t0.launch_cuda = launch_lars_trust_ratio_cuda;
#endif

    auto& t1 = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_UPDATE)];
    t1.op = ComputeOp::LARS_UPDATE;
    t1.launch_cpu = launch_lars_update_cpu;
#ifdef TR_USE_CUDA
    t1.launch_cuda = launch_lars_update_cuda;
#endif

    auto& t2 = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_NESTEROV_UPDATE)];
    t2.op = ComputeOp::LARS_NESTEROV_UPDATE;
    t2.launch_cpu = launch_lars_nesterov_update_cpu;
#ifdef TR_USE_CUDA
    t2.launch_cuda = launch_lars_nesterov_update_cuda;
#endif

    TR_LOG_DEBUG("backend") << "LARS operators registered (CPU+CUDA)";
}
```

---

## 四、现有代码修改

### 4.1 op_registry.h — 声明注册函数

在 [op_registry.h:114](file:///r:/renaissance/include/renaissance/backend/op_registry.h#L114) 之后添加：

```cpp
void register_op_lars();
```

### 4.2 op_registry.cpp — 调用注册

在 [op_registry.cpp:54](file:///r:/renaissance/src/backend/op_registry.cpp#L54) 的 `register_default_ops()` 末尾追加：

```cpp
register_op_lars();
```

### 4.3 src/CMakeLists.txt — 显式添加新文件

在 [src/CMakeLists.txt:86-87](file:///r:/renaissance/src/CMakeLists.txt#L86-L87) 的
`backend/ops/range/optimizer_op.cu` 之后追加：

```cmake
    backend/ops/dtensor/lars_op.cpp
    backend/ops/dtensor/lars_op.cu
```

> 注意：backend 源文件是显式 `list(APPEND RENAISSANCE_SOURCES ...)`（非 GLOB），
> 不添加则 lars_op.cpp / lars_op.cu 不会被编译。

### 4.4 无需修改的部分

| 组件 | 现状 | 理由 |
|------|------|------|
| Compiler 图注入 | ✅ 已完整 | [compiler.cpp:1392-1434](file:///r:/renaissance/src/graph/compiler.cpp#L1392-L1434) 正确生成 LARS_COMPUTE_TRUST_RATIO + LARS_UPDATE/NESTEROV_UPDATE |
| MemoryPlan N_* | ✅ 已完整 | [compiler.cpp:843-847](file:///r:/renaissance/src/graph/compiler.cpp#L843-L847) 条件分配 Shape{1} 标量 |
| ScalarIds | ✅ 已完整 | [memory_plan.cpp:388-391](file:///r:/renaissance/src/graph/memory_plan.cpp#L388-L391) 条件分配 tc/eps |
| Bias 复用 | ✅ 已完整 | LARS Bias → RANGE_UPDATE_BIAS_MOMENTUM/NESTEROV |

---

## 五、测试样例设计

### 5.1 测试架构

SimpleTask 的 capture 路径在 [capture_cuda.cpp:104-111](file:///r:/renaissance/src/graph/capture_cuda.cpp#L104-L111) 已完整支持 `GraphNode::Kind::COMPUTE`——它直接查 `g_compute_op_table` 调用 `launch_cuda`。只要 P3 完成注册，COMPUTE 节点就可以在 CUDA Graph 内正常工作。

**测试流程**（与现有 9 个 RangeOp 测试保持一致）：

```
SimpleTask task → alloc W/G/M/N + 5 scalars → finalize_memory()
→ 手动构造 ComputationGraph（COMPUTE 节点, 每个 layer 2 个节点）
→ task.add_graph("lars_weight", g, StreamKind::UPDATE)
→ compile() → transfer host data → run() → fetch → 与 CPU 参考实现对比
```

### 5.2 DTensor 布局

```
Region::W_FC_WEIGHT:  d_w_a (shape=4×8×4×4=512), d_w_b (shape=8×4×8×2=512)
Region::G_FC_WEIGHT:  d_g_a (同 shape_a),        d_g_b (同 shape_b)
Region::M_FC_WEIGHT:  d_m_a (同 shape_a),        d_m_b (同 shape_b)
Region::N_FC_WEIGHT:  d_n_a (Shape{1} 标量),     d_n_b (Shape{1} 标量)
S_SCALAR_FP32:        d_lr, d_wd, d_beta, d_tc, d_eps
```

### 5.3 Graph 结构（4 个 COMPUTE 节点）

```
GraphId::OPTIMIZER:
  [COMPUTE] LARS_COMPUTE_TRUST_RATIO  input_ids={w_a, g_a, tc, wd, eps}              output_ids={n_a}
  [COMPUTE] LARS_UPDATE               input_ids={w_a, g_a, m_a, n_a, lr, beta, wd}  output_ids={w_a, m_a}
  [COMPUTE] LARS_COMPUTE_TRUST_RATIO  input_ids={w_b, g_b, tc, wd, eps}              output_ids={n_b}
  [COMPUTE] LARS_UPDATE               input_ids={w_b, g_b, m_b, n_b, lr, beta, wd}  output_ids={w_b, m_b}
```

**关键注意事项**：

- trust_ratio 和 update 的 **input_ids 布局不同**：trust_ratio 需要 tc/wd/eps；update 需要 lr/beta/wd。与 Compiler 注入（compiler.cpp:1416-1431）完全一致。
- 测试仅用 `W_FC_WEIGHT` region（SimpleTask 无 FirstConv/DeepConv 层结构），
  但足以验证 kernel 数学的正确性。
- N_* 标量通过 `output_ids` 写入后由下一个 COMPUTE 节点的 `input_ids` 读取——
  CUDA Graph 串行捕获保证两节点之间的依赖。

### 5.4 test_lars_weight.cpp 验证公式

```
主机参考实现：
  η = cpu_compute_trust_ratio(h_w, h_g, tc, wd, eps)
  expected_w[i] = w[i] - lr * (beta * m[i] + eta * (g[i] + wd * w[i]))
  expected_m[i] = beta * m[i] + eta * (g[i] + wd * w[i])

验证：
  max|device_w[i] - expected_w[i]| < 1e-5
  max|device_m[i] - expected_m[i]| < 1e-5
```

### 5.5 test_lars_nesterov_weight.cpp 验证公式

唯一差异在最后一步：
```
  expected_w[i] = w[i] - lr * (eta * (g[i] + wd * w[i]) + beta * (beta * m[i] + eta * (g[i] + wd * w[i])))
```

其余结构与 test_lars_weight 相同。

### 5.6 CMakeLists.txt 注册

在 [tests/correction/CMakeLists.txt](file:///r:/renaissance/tests/correction/CMakeLists.txt) 现有 9 个测试之后，各追加 `test_lars_weight` 和 `test_lars_nesterov_weight` 的 CMake 条目（每项 ~10 行）。

---

## 六、实施清单

| # | 文件 | 操作 | 内容 | 预估行数 |
|---|------|------|------|----------|
| M1 | `src/backend/ops/dtensor/lars_op.cu` | **新建** | 3 CUDA kernel + 3 launch | ~120 |
| M2 | `src/backend/ops/dtensor/lars_op.cpp` | **新建** | 3 CPU fallback + register_op_lars() | ~150 |
| M3 | `include/renaissance/backend/op_registry.h` | 修改 | 声明 `register_op_lars()` | +1 |
| M4 | `src/backend/op_registry.cpp` | 修改 | `register_default_ops()` 加一行 | +1 |
| M5 | `src/CMakeLists.txt` | 修改 | 显式追加 `lars_op.cpp` / `lars_op.cu` | +2 |
| M6 | `tests/correction/test_lars_weight.cpp` | **新建** | LARS 正确性测试 | ~170 |
| M7 | `tests/correction/test_lars_nesterov_weight.cpp` | **新建** | LARS_NESTEROV 正确性测试 | ~170 |
| M8 | `tests/correction/CMakeLists.txt` | 修改 | 注册两个新测试 | ~30 |

### 实施顺序

```
P1 (M1+M2):    Kernel 层 → 编译通过（ninja renaissance）
P2 (M3+M4+M5): 注册+CMake 层 → 链接通过
P3 (M6+M7+M8): 测试层 → --cpu PASS → --gpu PASS
```

P1 和 P2 是依赖关系，P3 依赖 P2。P1 内部 M1/M2 可并行。

### 验证命令

```bash
# 编译
ninja renaissance test_lars_weight test_lars_nesterov_weight

# CPU 验证（先跑，排除 CUDA 复杂性）
./build/bin/tests/correction/test_lars_weight --cpu
./build/bin/tests/correction/test_lars_nesterov_weight --cpu

# GPU 验证（CUDA Graph capture）
./build/bin/tests/correction/test_lars_weight --gpu
./build/bin/tests/correction/test_lars_nesterov_weight --gpu
```

---

## 七、风险与缓解

| # | 风险 | 等级 | 缓解 |
|---|------|:--:|------|
| R1 | Trust ratio GPU reduce 与 CPU 顺序累加结果差异 | 低 | 阈值维持 1e-5；CPU 用 double 累加 |
| R2 | `atomicAdd` 到 shared mem 的精度 | 低 | float atomicAdd 精度足够，验证中会暴露 |
| R3 | COMPUTE 节点在 SimpleTask CUDA Graph 中首次使用 | 中 | capture_cuda.cpp 已支持 COMPUTE 调度，先跑 --cpu 再跑 --gpu |
| R4 | Grid size min((n+255)/256, 65535) 对大 DTensor 不充分 | 低 | ResNet-50 最大 FC 层 2048×1000=2M elts→~8000 blocks，远小于 65535 |
| R5 | 标量 tc/wd/eps 的 device ptr 在 capture 前就绪 | 低 | transfer_to_rank + compile 在 run 之前，数据已就绪 |

---

## 八、与现有优化器测试体系的对照

```
现有 9 测试（全部 PASS）:
  test_sgd_weight/bias  test_momentum_weight/bias  test_nesterov_weight/bias
  test_adam_weight/bias  test_adamw_weight

新增 2 测试:
  test_lars_weight  test_lars_nesterov_weight

覆盖矩阵:
  优化器          Weight 测试                  Bias 测试
  ─────────────────────────────────────────────────────
  SGD             test_sgd_weight             test_sgd_bias
  SGD_MOMENTUM    test_momentum_weight        test_momentum_bias
  SGD_NESTEROV    test_nesterov_weight        test_nesterov_bias
  LARS            test_lars_weight ★          ≡ test_momentum_bias ★
  LARS_NESTEROV   test_lars_nesterov_weight ★ ≡ test_nesterov_bias ★
  ADAM            test_adam_weight            test_adam_bias
  ADAMW           test_adamw_weight           ≡ test_adam_bias

★ LARS Bias 退化为 Momentum/Nesterov（tc=0, wd=0），已有测试覆盖
```

至此优化器全家桶（7 种 × Weight + Bias）正确性全覆盖。