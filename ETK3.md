# ETK3：BN INF eq_scale/eq_bias 预计算最终方案

> **目标**：消除 BN 推理（INF）每次执行时重复计算 `eq_scale/eq_bias` 的开销，改为每个 epoch 训练结束后一次性预计算并写入 `W_EQ_SCALE`/`W_EQ_BIAS`，后续 INF 直接读取。  
> **约束**：结果必须与当前代码一致；新增 DTensor 算子只能使用输入/输出张量，不得在算子内部申请额外内存。  
> **范围**：CPU/GPU 双路径、FP32/AMP 双精度、BN/CBR 两种含 BN 结构、单卡/多卡训练。

---

## 1. 现状与问题

当前 BN INF 与 CBR INF 中的 BN 段每次都会从 `gamma/beta/running_mean/running_var` 现场计算

```
eq_scale = gamma / sqrt(running_var + eps)
eq_bias  = beta - running_mean * eq_scale
```

然后再做 `y = x * eq_scale + eq_bias`。这导致：

1. **`W_EQ_SCALE`/`W_EQ_BIAS` 张量被浪费**：`infer_bn_tensors` 已经为每个 BN 层分配了这两个 FP32 张量，但 GPU INF 从不读取它们。
2. **重复计算**：验证阶段同一 epoch 的 running stats 不变，却为每个 batch 的每个像素重复做 `sqrt` 与乘加。

---

## 2. 核心设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 新增算子 | 单一 `ComputeOp::BN_UPDATE_EQ_PARAMS` | 1D/2D、FP32/AMP 的数学完全相同（per-channel），避免枚举爆炸。 |
| 图构建方式 | 在 `Compiler::build_auxiliary_graphs()` 中按 `Region` 批量生成节点 | `MemoryPlan::get_ids_by_region()` 对 `W_BN_WEIGHT/W_BN_BIAS/B_NEXT_MEAN/B_NEXT_VAR/W_EQ_SCALE/W_EQ_BIAS` 返回的 ID 天然按层对齐，自动覆盖 `Bn1d/Bn2d/CBR` 及所有展开后的复合层。 |
| `eps` 来源 | CPU：从全局 `bn_epsilon` 标量张量读取；GPU：从 `node.params.bn().eps` 读取 | 与当前代码行为保持一致：CPU BN FWD/INF 读输入标量，GPU BN INF/CBR INF 读 `node.params`。 |
| update 图流 | `StreamKind::COMP_2`（与 BN INF 同流） | 避免跨流同步，更新完成后 `sync_comp()` 即可保证后续 INF 读到最新值。 |
| update 图属性 | `shape-invariant` + `train-only` | 只操作 per-channel 张量，6 个 variant 共享一份 CapturedGraph。 |
| INF 输入精简 | BN INF 只保留 `eq_scale/eq_bias`；CBR INF 只保留 `amp_w/eq_scale/eq_bias` | 减少无效输入，kernel 只读取预计算结果。 |

---

## 3. 整体时序

```text
每个 epoch:
  所有训练 batch:
    FWD → BWD → COMM → OPTIMIZER → CAST_MAIN (AMP)
  last batch 结束后:
    UPDATE_BN_INF_PARAMS   (每个 rank 在 COMP_2 上更新所有层的 eq_scale/eq_bias)
  如果本 epoch 需要验证:
    INF_MAIN_A/B 直接读取 eq_scale/eq_bias
```

---

## 4. 详细改动

### 4.1 枚举与分类

#### `include/renaissance/graph/computation_graph.h`

- 在 `LARS_DEEP_CONV_OPT` 与 `COUNT` 之间插入：

```cpp
UPDATE_BN_INF_PARAMS,   // 每个 epoch 末预计算 BN 推理等效参数
```

- `graph_id_to_string()` 增加对应 `case`。
- `is_shape_invariant_graph()` 返回 `true`。
- `is_train_graph()` 返回 `true`。

#### `include/renaissance/graph/op_kind.h`

在 BatchNorm 区末尾新增：

```cpp
BN_UPDATE_EQ_PARAMS,
```

#### `src/graph/op_kind.cpp`

- `compute_op_to_string()` 增加 `BN_UPDATE_EQ_PARAMS` 分支。
- `format_params()` 中将其与现有 BN 算子归入同一分支（打印 `eps/momentum`）。

#### `src/backend/op_stream_policy.cpp`

```cpp
case ComputeOp::BN_UPDATE_EQ_PARAMS:
    return StreamKind::COMP_2;
```

#### `src/task/deep_learning_task.cpp`

- `GraphSlot` 枚举末尾新增 `UPDATE_BN_INF_PARAMS`。
- `stream_for()` 中：

```cpp
case GraphId::UPDATE_BN_INF_PARAMS:
    return StreamKind::COMP_2;
```

- `build_exec_table()` 的 `v <= 3` 分支中增加：

```cpp
g[S(GraphSlot::UPDATE_BN_INF_PARAMS)] = resolve(GraphId::UPDATE_BN_INF_PARAMS, rank, v);
```

### 4.2 新增 `BN_UPDATE_EQ_PARAMS` 算子

#### CPU 实现：`src/backend/ops/dtensor/bn_op.cpp`

新增：

```cpp
static void launch_bn_update_eq_params_cpu(CpuOpContext* op_ctx) {
    // input_ids:  [0]=gamma, [1]=beta, [2]=running_mean, [3]=running_var, [4]=bn_epsilon
    // output_ids: [0]=eq_scale, [1]=eq_bias
    const float* gamma = ...;
    const float* beta  = ...;
    const float* rm    = ...;
    const float* rv    = ...;
    float* eq_scale = ...;
    float* eq_bias  = ...;

    int C = op_ctx->input_shape.c;
    float eps = 1e-5f;
    if (op_ctx->num_inputs >= 5 && op_ctx->input_ids[4] >= 0) {
        eps = *static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[4]));
    }

    for (int c = 0; c < C; ++c) {
        float inv_std = 1.0f / std::sqrt(rv[c] + eps);
        eq_scale[c] = gamma[c] * inv_std;
        eq_bias[c]  = beta[c]  - rm[c] * eq_scale[c];
    }
}
```

> 说明：与现有 **CPU BN FWD/INF** 一致，从输入标量 `bn_epsilon` 读取 `eps`；不申请任何额外内存。

注册：

```cpp
table[static_cast<size_t>(ComputeOp::BN_UPDATE_EQ_PARAMS)].launch_cpu = launch_bn_update_eq_params_cpu;
```

#### GPU 实现：`src/backend/ops/dtensor/bn_op.cpp/.cu`

复用 `bn_op.cu` 中已有的 `tr_bn_compute_eq_params_kernel` / `launch_tr_bn_compute_eq_params_kernel`。

新增 GPU launch wrapper：

```cpp
static void launch_bn_update_eq_params_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  gamma, beta, running_mean, running_var
    // output_ids: eq_scale, eq_bias
    const DTensor& dt_gamma = mp.get_dtensor(node.input_ids[0]);
    int C = dt_gamma.shape.c();

    StreamKind sk = get_op_default_stream(node.compute_op);  // COMP_2
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(stream);
    // 标准流同步 boilerplate ...

    const float* gamma = ...;
    const float* beta  = ...;
    const float* rm    = ...;
    const float* rv    = ...;
    float* eq_scale = ...;
    float* eq_bias  = ...;

    float eps = 1e-5f;
    if (!node.params.is_empty()) eps = node.params.bn().eps;
    // 与现有 GPU BN INF / CBR INF 一致，eps 来自 node.params

    cudaError_t err = launch_tr_bn_compute_eq_params_kernel(
        gamma, beta, rm, rv, eps, eq_scale, eq_bias, C, stream);
    // 错误处理 ...
    cudaEventRecord(state.streams[si].last_done_event, stream);
}
```

注册：

```cpp
#ifdef TR_USE_CUDA
table[static_cast<size_t>(ComputeOp::BN_UPDATE_EQ_PARAMS)].launch_cuda = launch_bn_update_eq_params_cuda;
#endif
```

### 4.3 修改 BN INF：直接读取 `eq_scale/eq_bias`

#### 标准 BN

`src/graph/layer_descriptor_registry.cpp` 的 `build_bn_inference`：

```cpp
n.input_indices = {10, 9};   // eq_scale, eq_bias（原：{10,9,0,1,5,6}）
n.output_indices = {2};      // bn_output
```

`src/graph/compiler.cpp`：删除 BN INF 的 `bn_epsilon` 注入逻辑（原 `is_bn_inf_op` 分支）。

`src/backend/ops/dtensor/bn_op.cpp`：重写 `launch_bn_fp32_inf_cpu`：

```cpp
static void launch_bn_fp32_inf_cpu(CpuOpContext* op_ctx) {
    // input_ids:  [0]=X, [1]=eq_scale, [2]=eq_bias
    // output_ids: [0]=Y
    const float* x = ...;
    const float* eq_scale = ...;
    const float* eq_bias  = ...;
    float* y = ...;

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

`src/backend/ops/dtensor/bn_op.cu`：新增/替换为只读 kernel：

```cpp
__global__ void tr_bn_inf_from_eq_fp32_kernel(
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

__global__ void tr_bn_inf_from_eq_fp16_kernel(
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

extern "C" cudaError_t launch_tr_bn_inf_from_eq_kernel(
    const void* x, const float* eq_scale, const float* eq_bias,
    void* y, int N, int C, int H, int W, bool is_fp16, cudaStream_t stream)
{
    const int block_size = 256;
    int total = N * H * W * C;
    int grid_size = (total + block_size - 1) / block_size;
    if (is_fp16) {
        tr_bn_inf_from_eq_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const __half*>(x), eq_scale, eq_bias,
            static_cast<__half*>(y), N, C, H, W);
    } else {
        tr_bn_inf_from_eq_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(x), eq_scale, eq_bias,
            static_cast<float*>(y), N, C, H, W);
    }
    return cudaGetLastError();
}
```

`src/backend/ops/dtensor/bn_op.cpp` 的 `launch_bn_inf_cuda` 改为：

```cpp
static void launch_bn_inf_cuda(...) {
    // input_ids:  [0]=X, [1]=eq_scale, [2]=eq_bias
    // output_ids: [0]=Y
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    int N=dt_x.shape.n(), H=dt_x.shape.h(), W=dt_x.shape.w(), C=dt_x.shape.c();
    bool is_fp16 = (dt_x.dtype == DType::FP16);

    // 流同步 boilerplate ...

    launch_tr_bn_inf_from_eq_kernel(
        ctx.ptr_at(node.input_ids[0]),
        static_cast<const float*>(ctx.ptr_at(node.input_ids[1])),
        static_cast<const float*>(ctx.ptr_at(node.input_ids[2])),
        ctx.ptr_at(node.output_ids[0]),
        N, C, H, W, is_fp16, stream);
}
```

旧的 `tr_bn_inf_fp32/fp16_kernel` 与 `launch_tr_bn_inf_kernel` 不再使用，可删除。

### 4.4 修改 CBR INF

`src/graph/layer_descriptor_registry.cpp` 的 `build_cbr_inference`：

```cpp
n.input_indices = {4, 18, 17};   // amp_w, eq_scale, eq_bias
n.output_indices = {1, 6, 7, 10, 21, 22};
```

`src/graph/compiler.cpp`：删除 CBR INF 的 `bn_epsilon` 注入逻辑。

`src/backend/ops/dtensor/cbr_op.cpp` 的 `launch_cbr_amp_inf_cuda` 中 BN 段改为：

```cpp
const DTensor& dt_x = mp.get_dtensor(node.output_ids[0]);  // conv_output
int N=dt_x.shape.n(), H=dt_x.shape.h(), W=dt_x.shape.w(), C=dt_x.shape.c();
bool is_fp16 = (dt_x.dtype == DType::FP16);

const float* eq_scale = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));
const float* eq_bias  = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));
void* y = ctx.ptr_at(node.output_ids[3]);  // bn_output

launch_tr_bn_inf_from_eq_kernel(
    ctx.ptr_at(node.output_ids[0]), eq_scale, eq_bias, y,
    N, C, H, W, is_fp16, stream);
```

### 4.5 构建 `UPDATE_BN_INF_PARAMS` 图

在 `src/graph/compiler.cpp` 的 `Compiler::build_auxiliary_graphs()` 中，紧接 BN 一致性校验块（`if (has_bn) { ... }`）之后、优化器图构建之前插入，以复用该校验块求出的 `first_eps` / `first_mom`：

```cpp
// UPDATE_BN_INF_PARAMS: 每个 epoch 末一次性预计算所有 BN 层的 eq_scale/eq_bias
if (has_bn && memory_plan.config().bn_folded) {
    const auto& w_ids  = memory_plan.get_ids_by_region(Region::W_BN_WEIGHT);
    const auto& b_ids  = memory_plan.get_ids_by_region(Region::W_BN_BIAS);
    const auto& rm_ids = memory_plan.get_ids_by_region(Region::B_NEXT_MEAN);
    const auto& rv_ids = memory_plan.get_ids_by_region(Region::B_NEXT_VAR);
    const auto& es_ids = memory_plan.get_ids_by_region(Region::W_EQ_SCALE);
    const auto& eb_ids = memory_plan.get_ids_by_region(Region::W_EQ_BIAS);

    size_t n = w_ids.size();
    TR_CHECK(n == b_ids.size() && n == rm_ids.size() && n == rv_ids.size()
             && n == es_ids.size() && n == eb_ids.size(),
             ShapeError, "BN_UPDATE_EQ_PARAMS region count mismatch");

    for (size_t i = 0; i < n; ++i) {
        GraphNode node;
        node.kind = GraphNode::Kind::COMPUTE;
        node.compute_op = ComputeOp::BN_UPDATE_EQ_PARAMS;

        // 供 GPU launch 使用：与 GPU BN INF/CBR INF 一致，从 node.params.bn().eps 读取
        float eps = (first_eps >= 0.0f) ? first_eps : 1e-5f;
        float mom = (first_mom >= 0.0f) ? first_mom : 0.1f;
        node.params = OpParams(BNParams{eps, mom});

        node.input_ids  = {w_ids[i], b_ids[i], rm_ids[i], rv_ids[i]};
        // 供 CPU launch 使用：与 CPU BN FWD/INF 一致，追加全局 bn_epsilon 标量
        if (b.bn_epsilon >= 0) node.input_ids.push_back(b.bn_epsilon);
        node.output_ids = {es_ids[i], eb_ids[i]};
        train_cg.append(GraphId::UPDATE_BN_INF_PARAMS, std::move(node));
    }
}
```

> 说明：
> - GPU 路径与现有 **GPU BN INF / CBR INF** 一致，从 `node.params.bn().eps` 读取 `eps`；
> - CPU 路径与现有 **CPU BN FWD / INF** 一致，从输入标量 `bn_epsilon` 读取 `eps`；
> - `first_eps` / `first_mom` 复用 `build_auxiliary_graphs()` 开头 BN 一致性校验时已求出的值。  
> 该图不依赖 batch size 或空间分辨率，因此是 shape-invariant，可在 6 个 variant 间共享。

### 4.6 训练循环调用

#### GPU 路径：`src/task/deep_learning_task.cpp` 的 `run_train_epoch_gpu()`

在预解析区增加：

```cpp
auto n_update_bn = g_n[S(GraphSlot::UPDATE_BN_INF_PARAMS)];
```

在 last batch 块末尾（`if (using_amp && n_cm) { ... sync_up(); }` 之后、`catch` 之前）增加：

```cpp
if (n_update_bn) cudaGraphLaunch(n_update_bn, s_c2);
sync_comp();
```

> 此时 optimizer、LARS、CAST_MAIN 已全部完成，`sync_comp()` 保证更新结果对后续验证 INF 可见。

#### CPU 路径：`src/task/deep_learning_task.cpp` 的 `run_train_epoch_cpu()`

预解析：

```cpp
int32_t idx_update_bn = idx_for(GraphId::UPDATE_BN_INF_PARAMS, 0);
```

在 last batch 块结束、`ctx.set_memory_plan(active_memory_plan_)` 恢复之后、`train_loss` 计算之前增加：

```cpp
if (idx_update_bn >= 0) launch(idx_update_bn);
```

---

## 5. 同步与正确性

| 关键点 | 说明 |
|---|---|
| 与 optimizer 的时序 | last batch 在 `WEIGHT_UPDATE`、`LARS_*`、`CAST_MAIN` 之后执行 update；`sync_up()`/`sync_comp()` 已保证 `W_BN_*`、`B_NEXT_*` 为最终值。 |
| 与验证 INF 的时序 | update 在 `COMP_2` 上执行，随后 `sync_comp()`；`run_train_epoch_gpu()` 返回后所有 rank 的 COMP 流空闲，下一阶段的 INF 图读取到最新 `W_EQ_*`。 |
| 与 `STATS_COMM` 的时序 | `STATS_COMM` 在最后一个 batch 中已 AllReduce 并复制 `B_NEXT → B_PREV`；update 读取 `B_NEXT`（即本 epoch 最终统计量）。 |
| 多卡一致性 | 各 rank 独立更新自己的 `W_EQ_*`；running stats 已经过 `STATS_COMM` 同步，结果一致。 |
| SEMA | 当前 `run_val_epoch_gpu(true)` 未实际启用 EMA 验证；若未来启用，需额外维护 EMA 对应的 `E_EQ_SCALE/E_EQ_BIAS`，本方案不改变现有 SEMA 主模型行为。 |

---

## 6. 内存约束自查

- `BN_UPDATE_EQ_PARAMS`：仅读取 `gamma/beta/running_mean/running_var`，写入 `eq_scale/eq_bias`；不分配任何临时缓冲区。
- 新 BN INF kernel：仅读取 `X/eq_scale/eq_bias`，写入 `Y`；不分配任何临时缓冲区。
- CBR INF：复用同一 kernel，不新增缓冲区。
- 所有改动均遵守“新增 DTensor 算子只能在输入/输出张量上操作”的约束。

---

## 7. 文件改动清单

| 文件 | 改动 |
|---|---|
| `include/renaissance/graph/computation_graph.h` | 新增 `GraphId::UPDATE_BN_INF_PARAMS`；更新字符串/分类函数。 |
| `include/renaissance/graph/op_kind.h` | 新增 `ComputeOp::BN_UPDATE_EQ_PARAMS`。 |
| `src/graph/op_kind.cpp` | 新增字符串映射与参数打印。 |
| `src/backend/op_stream_policy.cpp` | 新算子映射到 `COMP_2`。 |
| `src/backend/ops/dtensor/bn_op.cpp` | 新增 CPU/GPU update 实现；重写 CPU INF；重写 GPU INF launch；注册新算子。 |
| `src/backend/ops/dtensor/bn_op.cu` | 新增 `tr_bn_inf_from_eq_*_kernel` 与 launch wrapper；保留 `tr_bn_compute_eq_params_kernel`。 |
| `src/backend/ops/dtensor/cbr_op.cpp` | CBR INF 改用 `eq_scale/eq_bias`。 |
| `src/graph/layer_descriptor_registry.cpp` | 修改 `build_bn_inference` / `build_cbr_inference` 的 `input_indices`。 |
| `src/graph/compiler.cpp` | 构建 `UPDATE_BN_INF_PARAMS` 图；删除 BN/CBR INF 的 `bn_epsilon` 注入。 |
| `src/task/deep_learning_task.cpp` | 新增 `GraphSlot`、`stream_for`、`build_exec_table` 解析；GPU/CPU 训练循环在 epoch 末调用。 |

---

## 8. 风险与缓解

| 风险 | 说明 | 缓解 |
|---|---|---|
| 数值等价性 | 公式与原地计算完全一致 | 验证阶段对比 `eq_scale/eq_bias` 与手工计算结果 |
| 首次验证时机 | 若 `val_offset=0`，epoch 0 训练结束后即验证，此时 `W_EQ_*` 尚未更新 | 默认配置 `val_offset=1`；若需支持 `offset=0`，在首次验证前强制跑一次 `UPDATE_BN_INF_PARAMS` |
| 跨流可见性 | update 与 INF 流分离 | 将 update 放在 `COMP_2`（BN INF 同流），并在 epoch 末 `sync_comp()` |
| EMA 验证 | 当前 EMA 推理路径未实际构建 | 未来启用时需为 EMA 维护独立 `E_EQ_SCALE/E_EQ_BIAS` |
| 无 BN 模型 | `UPDATE_BN_INF_PARAMS` 图为空 | 调度代码中判空跳过 |

---

## 9. 测试计划

1. **编译通过**：全量 ninja 编译无错误。
2. **单元验证**：单卡 CPU/GPU 小网络，读取 `W_EQ_SCALE/W_EQ_BIAS`，确认与 `gamma/beta/running stats` 计算结果一致。
3. **集成验证**：VGG16BN AMP 跑 1 个 epoch，确认：
   - `UPDATE_BN_INF_PARAMS` 每个 epoch 末执行一次；
   - 验证阶段 BN INF / CBR INF 不再访问 `B_NEXT_MEAN/B_NEXT_VAR`。
4. **精度回归**：相同随机种子下，修改前后 loss/acc 一致（允许 FP16 累加顺序差异 `~1e-5`）。
5. **性能验证**：
   - 验证阶段 INF kernel 中不再出现 `sqrt`/`rcp`；
   - `UPDATE_BN_INF_PARAMS` 耗时远小于一个验证 batch。
6. **边界测试**：
   - `batches == 1`；
   - 模型不含 BN；
   - 多卡训练。
