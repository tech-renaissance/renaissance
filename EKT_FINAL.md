# EKT_FINAL：SoftmaxCE GPU 确定性改造 — 最终综合方案

> 版本: V1.1 | 日期: 2026-06-12 | 综合 EKT1 / EKT2 / EKT3 的最终方案，已纳入 S_REV.md 评审意见

---

## 一、方案综合评估

### 1.1 根因回顾

当前 `softmax_ce_op.cu` 中不可复现性来源于三个 `atomicAdd` 调用点：

```cpp
// FWD:  atomicAdd(loss, sample_loss * inv_batch);
// INF:  atomicAdd(loss, sample_loss * inv_batch);
// INF:  atomicAdd(top1, inv_batch);
// INF:  atomicAdd(top5, inv_batch);
```

`atomicAdd` 保证原子性但不保证跨 block 执行顺序，浮点加法不满足结合律，导致多次运行结果在 ULP 级别不一致。BWD kernel 为纯逐元素操作，无跨 block 归约，完全确定性，**本次不修改**。

### 1.2 三份 EKT 方案关键差异对比

| 维度 | EKT1 | EKT2 | EKT3 | **EKT_FINAL 选择** |
|------|------|------|------|---------------------|
| Compiler output_ids 顺序 | 不改注入 → top1/5 在 [7]/[8] | **显式重排** → top1/5 在 [4]/[5] | 不改注入 → top1/5 在 [7]/[8] | **EKT2：显式重排** |
| CPU INF 是否需要改索引 | 是 ([4]→[7], [5]→[8]) | **否** | 是 ([4]→[7], [5]→[8]) | **EKT2：不改 CPU** |
| scaling 是否乘入 loss | 否（保持 GPU 现状） | 是（reduce 阶段乘） | 是（main kernel 乘） | **EKT1：不乘** |
| Reduce kernel 线程数 | BLOCK_DIM（树形归约） | 1（单线程顺序） | BLOCK_DIM（树形归约） | **EKT1/EKT3：BLOCK_DIM** |
| Reduce kernel 参数 | `batch` (int) | `batch` + `scaling` | `batch_size_ptr` | **EKT1：`batch`** |
| 统一 reduce kernel | 是（`compute_top_metrics`） | 是 | 是 | **是** |
| 新增 ComputeOp | 否 | 否 | 否 | **否** |

> 注：EKT1 的 partial 张量在 `input_ids` 中传递，output_ids 保持 6 个，top1/top5 仍在 [4]/[5]（因为编译器追加 baseline scalar 时填到已有位置）。EKT2 通过显式重排将 top1/top5 放在 [4]/[5]，partial 放在 [6]/[7]/[8]。EKT_FINAL 采用 EKT2 的重排方式。

### 1.3 各维度选择理由

**1. Compiler 显式重排 (EKT2)**：将 `top1`/`top5` 保持在 `output_ids[4]`/`[5]`，使得 CPU INF 和 CUDA dispatch 中所有对 `top1`/`top5` 的索引引用完全不需要修改，改动面最小，风险最低。

**2. 不乘 scaling (EKT1)**：当前 GPU 的 loss 计算不乘 scaling（与 CPU 的 `*loss *= scaling` 存在历史差异）。本次目标仅为消除不可复现性，不应该同时改变 loss 数值语义。scaling 对齐问题应作为独立 issue 处理。

**3. BLOCK_DIM 树形归约 (EKT1/EKT3)**：相比单线程顺序累加（EKT2），树形归约在大 batch 下有更好的并行度，且 shared memory tree reduction 的配对模式完全由 `blockDim.x` 固定，确定性不变。

**4. 使用 `batch` 参数而非 `batch_size_ptr`**：在 CUDA dispatch 中 `batch = logits_dt.shape.n()` 已经是实际 batch size，kernel 的 grid 也按此启动。不需要额外读取 `batch_size_ptr`。

---

## 二、最终 output_ids 布局

### 2.1 设计原则

通过 Compiler 显式重排，使 `top1`/`top5` 保持在索引 [4]/[5]，partial 缓冲区放在 [6]/[7]/[8]。这样 CPU 和 CUDA dispatch 中所有现有索引引用都不需要更改。

### 2.2 最终布局

```
索引  张量            来源                  区域              DType
[0]   loss            baseline scalar       S_SCALAR_FP32     FP32
[1]   inv_scaling     desc[3]               S_SCALAR_FP32     FP32
[2]   pred            desc[2]               R_PREDICTED_LABEL INT32
[3]   probs           desc[1]               T_TEMP_FP32       FP32
[4]   top1            baseline scalar       S_SCALAR_FP32     FP32
[5]   top5            baseline scalar       S_SCALAR_FP32     FP32
[6]   loss_partial    desc[4] (NEW)         T_TEMP_FP32       FP32
[7]   top1_partial    desc[5] (NEW)         T_TEMP_INT32      INT32
[8]   top5_partial    desc[6] (NEW)         T_TEMP_INT32      INT32
```

### 2.3 与现有代码的兼容性

| 现有代码位置 | 原索引 | 新索引 | 是否需要改 |
|-------------|--------|--------|-----------|
| CPU FWD: `loss` | [0] | [0] | 否 |
| CPU FWD: `inv_sc` | [1] | [1] | 否 |
| CPU FWD: `pred` | [2] | [2] | 否 |
| CPU FWD: `probs` | [3] | [3] | 否 |
| CPU FWD: `top1` | [4] | [4] | 否 |
| CPU FWD: `top5` | [5] | [5] | 否 |
| CPU INF: `loss` | [0] | [0] | 否 |
| CPU INF: `top1` | [4] | [4] | 否 |
| CPU INF: `top5` | [5] | [5] | 否 |
| CUDA FWD: `loss` | [0] | [0] | 否 |
| CUDA FWD: `top1` | [4] | [4] | 否 |
| CUDA FWD: `top5` | [5] | [5] | 否 |
| CUDA INF: 所有字段 | [0]..[5] | [0]..[5] | 否 |

**仅新增** `ids_out[6]`/`[7]`/`[8]` 的 partial 指针读取。现有代码索引完全不变。

---

## 三、涉及文件与改动摘要

| 文件 | 改动内容 |
|------|---------|
| `src/graph/layer_descriptor_registry.cpp` | ① `infer_softmaxce_tensors` 新增 3 个 TEMP desc（共 7 个 desc）；② `build_softmaxce_forward/inference` 更新 `output_indices`；③ `build_softmaxce_backward` desc 数量检查更新为 `< 7` |
| `src/graph/compiler.cpp` | 两处 SoftmaxCE 注入逻辑显式重排 `output_ids` |
| `src/backend/ops/dtensor/softmax_ce_op.cu` | ① FWD/INF kernel 签名 + 逻辑改写；② 新增 `softmax_ce_reduce_kernel`；③ 更新 4 个 launch wrapper |
| `src/backend/ops/dtensor/softmax_ce_op.cpp` | ① 前向声明更新；② CUDA dispatch 新增 partial 指针读取；③ CPU 路径不变 |

**BWD 完全不变**（纯逐元素，无归约，本已确定）。

> **实施时需同步更新的注释**：
> - `softmax_ce_op.cpp` 头部注释：`FWD: 6 output` → `FWD: 9 output`（6 个原有 + 3 个 partial）
> - `layer_descriptor_registry.cpp` SoftmaxCE 段注释：`// SoftmaxCE — 4张量` → `// SoftmaxCE — 7张量`
> - `compiler.cpp` 注入段注释：补充 partial 输出索引说明

---

## 四、LayerDescriptor 改动

文件: `src/graph/layer_descriptor_registry.cpp`

### 4.1 `infer_softmaxce_tensors` — 新增 3 个 TEMP desc

> 注意：`batch_shape` 的 C=1，`padded_c=1`。kernel 中直接使用 `partial[b]` 做 sample-index 索引，**依赖 T_TEMP_FP32 / T_TEMP_INT32 区域 alignment=1，不引入 channel padding**。如果将来 TEMP 区域引入 padding，此处需要改为 `partial[b * stride]`。

```cpp
std::vector<TensorDesc> infer_softmaxce_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    Shape batch_shape{input.n(), 1, 1, 1};
    return {
        // --- 原有 4 个 (索引 0..3) 不变 ---
        TensorDesc{"ce_output",          input,           select_feature_region(ctx), feat_dt},     // 0
        TensorDesc{"softmax_probs",      input,           Region::T_TEMP_FP32,       DType::FP32},  // 1
        TensorDesc{"pred_labels",        batch_shape,     Region::R_PREDICTED_LABEL, DType::INT32}, // 2
        TensorDesc{"inv_scaling_factor", Shape{1,1,1,1},   Region::S_SCALAR_FP32,     DType::FP32},  // 3
        // --- 新增 3 个 TEMP (索引 4..6) ---
        TensorDesc{"loss_partial",       batch_shape,     Region::T_TEMP_FP32,       DType::FP32},  // 4
        TensorDesc{"top1_partial",       batch_shape,     Region::T_TEMP_INT32,      DType::INT32}, // 5
        TensorDesc{"top5_partial",       batch_shape,     Region::T_TEMP_INT32,      DType::INT32}, // 6
    };
}
```

> `input.n()` 在每个 variant 下分别为 `local_batch_size` 或 `last_*_batch_size`，MemoryPlan 为每个 variant 独立分配正确大小的 TEMP。

### 4.2 `build_softmaxce_forward` — 更新 output_indices

```cpp
SubgraphPattern build_softmaxce_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;   // 4 → 7
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::SOFTMAX_CE_AMP_FWD
                                                   : ComputeOp::SOFTMAX_CE_FP32_FWD;
    n.output_indices = {3, 2, 1, 4, 5, 6};   // 新增 4,5,6
    // → [inv_scaling, pred, probs, loss_partial, top1_partial, top5_partial]
    p.nodes.push_back(n);
    return p;
}
```

### 4.3 `build_softmaxce_inference` — 同 4.2

```cpp
SubgraphPattern build_softmaxce_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::SOFTMAX_CE_AMP_INF
                                                   : ComputeOp::SOFTMAX_CE_FP32_INF;
    n.output_indices = {3, 2, 1, 4, 5, 6};
    p.nodes.push_back(n);
    return p;
}
```

### 4.4 `build_softmaxce_backward` — desc 数量检查更新

```cpp
SubgraphPattern build_softmaxce_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;   // 4 → 7，与 FWD/INF 统一
    // output_indices = {0}  不变
    // input_indices  = {0, 1, 3}  不变
    // ...
}
```

---

## 五、Compiler 改动

文件: `src/graph/compiler.cpp`

### 5.1 训练路径注入（约 L1195-1208）— 显式重排

```cpp
if (gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_FWD ||
    gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_FWD  ||
    gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_INF ||
    gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_INF) {
    const auto& b = memory_plan.baseline();

    // inputs 不变
    gn.input_ids.push_back(b.scaling);
    gn.input_ids.push_back(b.local_batch_size);
    gn.input_ids.push_back(b.label_smce);
    gn.input_ids.push_back(b.label_smoothing);

    // 显式重排 outputs：
    // 目标: [loss, inv_scaling, pred, probs, top1, top5, loss_p, top1_p, top5_p]
    // orig = [inv_scaling(3), pred(2), probs(1), loss_p(4), top1_p(5), top5_p(6)]
    auto orig = gn.output_ids;
    gn.output_ids.clear();
    gn.output_ids.push_back(b.loss);                                   // 0: loss
    gn.output_ids.insert(gn.output_ids.end(), orig.begin(), orig.begin() + 3); // 1,2,3: inv_sc, pred, probs
    gn.output_ids.push_back(b.top1);                                   // 4: top1
    gn.output_ids.push_back(b.top5);                                   // 5: top5
    gn.output_ids.insert(gn.output_ids.end(), orig.begin() + 3, orig.end());   // 6,7,8: loss_p, top1_p, top5_p
}
```

### 5.2 推理路径注入（约 L1597-1609）— 同 5.1

```cpp
// 与训练路径完全相同的重排逻辑
if (gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_FWD ||
    gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_FWD  ||
    gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_INF ||
    gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_INF) {
    // ... 同上 ...
}
```

### 5.3 BWD 注入 — 不变

```cpp
// (约 L1400-1406) 完全不变
```

---

## 六、CUDA Kernel 改动

文件: `src/backend/ops/dtensor/softmax_ce_op.cu`

### 6.1 FWD Kernel 改写

**关键变更**:
1. `atomicAdd(loss, ...)` → `loss_partial[b] = sample_loss * inv_batch`
2. 不乘 scaling（保持 GPU 现有行为）
3. 保留 `top1`/`top5`/`pred` 参数用于接口对齐，但不操作（FWD 不计算 top metrics）
4. 不写 `top1_partial`/`top5_partial`（R_RESULT 区由 ZERO_GRAD 图清零，无需冗余写 0）

```cpp
template <bool IS_AMP>
__global__ void softmax_ce_fwd_kernel(
    const void*  __restrict__ logits_ptr,
    const int*   __restrict__ labels,
    float* __restrict__ loss,              // 保留参数，不再写入
    float* __restrict__ top1,              // 保留参数，接口对齐
    float* __restrict__ top5,              // 保留参数，接口对齐
    int*   __restrict__ pred,              // 保留参数，接口对齐
    float* __restrict__ probs,             // ids_out[3]
    float* __restrict__ inv_scaling,       // ids_out[1]
    const float* __restrict__ scaling_ptr,
    const int32_t* __restrict__ batch_size_ptr,
    const float* __restrict__ label_smoothing_ptr,
    float* __restrict__ loss_partial,      // [NEW] ids_out[6]
    int*   __restrict__ top1_partial,      // [NEW] ids_out[7] — FWD 不操作
    int*   __restrict__ top5_partial,      // [NEW] ids_out[8] — FWD 不操作
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    // FWD 不计算 top1/top5/pred，保留参数仅用于接口对齐
    (void)loss; (void)top1; (void)top5; (void)pred;
    (void)scaling_ptr; (void)batch_size_ptr;
    (void)top1_partial; (void)top5_partial;

    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_sum = smem + WARP_COUNT;

    int b = blockIdx.x;
    if (b >= batch) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int label_b = static_cast<int>(labels[b]);
    float inv_batch = 1.0f / static_cast<float>(batch);

    // ===== Step 1-3: Max + Exp + Sum + Normalize + sum_log (完全不变) =====
    // ... (保持原代码 L85-L151 不变) ...

    // ===== Step 4: 写入 partial 缓冲区 (替代 atomicAdd) =====
    if (tid == 0) {
        float prob_y = probs[b * probs_stride + label_b];
        float sample_loss = -one_minus_ls * logf(prob_y + 1e-8f)
                            - ls_over_K * sum_log;
        loss_partial[b] = sample_loss * inv_batch;   // [CHANGED] 直接写入，无竞争
    }

    if (b == 0 && tid == 0) {
        *inv_scaling = inv_batch;
    }
}
```

### 6.2 INF Kernel 改写

**关键变更**:
1. `atomicAdd(loss, ...)` → `loss_partial[b] = sample_loss * inv_batch`
2. `atomicAdd(top1, inv_batch)` → `top1_partial[b] = 0/1` (INT32)
3. `atomicAdd(top5, inv_batch)` → `top5_partial[b] = 0/1` (INT32)
4. `inv_scaling` 写入条件改为 `b == 0 && tid == 0`（避免 block 0 全部 256 线程冗余写）

```cpp
template <bool IS_AMP>
__global__ void softmax_ce_inf_kernel(
    const void*  __restrict__ logits_ptr,
    const int*   __restrict__ labels,
    float* __restrict__ loss,              // 保留参数，不再写入
    float* __restrict__ top1,              // 保留参数，不再写入
    float* __restrict__ top5,              // 保留参数，不再写入
    int*   __restrict__ pred,              // ids_out[2]
    float* __restrict__ probs,             // ids_out[3]
    float* __restrict__ inv_scaling,       // ids_out[1]
    const float* __restrict__ scaling_ptr,
    const int32_t* __restrict__ batch_size_ptr,
    const float* __restrict__ label_smoothing_ptr,
    float* __restrict__ loss_partial,      // [NEW] ids_out[6]
    int*   __restrict__ top1_partial,      // [NEW] ids_out[7]
    int*   __restrict__ top5_partial,      // [NEW] ids_out[8]
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    (void)loss; (void)top1; (void)top5;
    (void)scaling_ptr; (void)batch_size_ptr;

    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_sum = smem + WARP_COUNT;
    float* s_top1_val = smem + 2 * WARP_COUNT;
    int*   s_top1_cls = reinterpret_cast<int*>(smem + 2 * WARP_COUNT + BLOCK_DIM);

    int b = blockIdx.x;
    if (b >= batch) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int label_b = static_cast<int>(labels[b]);
    float inv_batch = 1.0f / static_cast<float>(batch);

    // ===== Step 1-3: Max + Exp + Sum + Normalize + sum_log + top1 (完全不变) =====
    // ... (保持原代码 L209-L287 不变) ...

    // ===== Step 4: 写入 partial (替代所有 atomicAdd) =====
    if (tid == 0) {
        float ls = label_smoothing_ptr[0];
        float prob_y = probs[b * probs_stride + label_b];
        float sample_loss = -(1.0f - ls) * logf(prob_y + 1e-8f)
                            - (ls / static_cast<float>(num_classes)) * sum_log;
        loss_partial[b] = sample_loss * inv_batch;           // [CHANGED]

        int global_top1_cls = s_top1_cls[0];
        float global_top1_val = s_top1_val[0];
        for (int i = 1; i < BLOCK_DIM; ++i) {
            if (s_top1_val[i] > global_top1_val) {
                global_top1_val = s_top1_val[i];
                global_top1_cls = s_top1_cls[i];
            }
        }
        pred[b] = global_top1_cls;
        top1_partial[b] = (global_top1_cls == label_b) ? 1 : 0;  // [CHANGED] INT32

        if (num_classes >= 5) {
            bool in_top5 = false;
            for (int k = 0; k < 5; ++k) {
                int best_j = -1;
                float best_prob = -1.0f;
                for (int j = 0; j < num_classes; ++j) {
                    float p = probs[b * probs_stride + j];
                    if (p > best_prob) { best_prob = p; best_j = j; }
                }
                if (best_j == label_b) { in_top5 = true; break; }
                probs[b * probs_stride + best_j] = -best_prob;
            }
            for (int j = 0; j < num_classes; ++j) {
                float p = probs[b * probs_stride + j];
                if (p < 0.0f) probs[b * probs_stride + j] = -p;
            }
            top5_partial[b] = in_top5 ? 1 : 0;
        } else {
            top5_partial[b] = top1_partial[b];
        }
    }

    if (b == 0 && tid == 0) {
        *inv_scaling = 1.0f / static_cast<float>(batch);   // [CHANGED] 加 tid==0 条件
    }
}
```

### 6.3 新增: 统一确定性归约 Kernel

**功能**: 单 block 确定性归约，FWD/INF 共用。BLOCK_DIM 线程树形归约，固定线程→索引映射。

**FWD 行为**: 只归约 `loss_partial → loss`，不操作 top1/top5（R_RESULT 区由 ZERO_GRAD 图清零，无需写 0）。

**INF 行为**: 归约 `loss_partial → loss`，`top1_partial → top1`，`top5_partial → top5`。

```cpp
/**
 * @brief 统一确定性归约 kernel（FWD/INF 共用）
 *
 * 单 block 启动，按固定顺序累加 partial 缓冲区到标量输出。
 * - 始终归约 loss_partial → loss
 * - compute_top_metrics=true 时额外归约 top1_partial → top1, top5_partial → top5
 *   compute_top_metrics=false 时不操作 top1/top5（R_RESULT 区由 ZERO_GRAD 图保证清零）
 *
 * 确定性保证：
 *   1. 每个线程按固定 stride (tid, tid+BLOCK_DIM, ...) 读取，顺序唯一
 *   2. shared memory tree reduction 固定配对模式
 *   3. top1/top5 用 INT32 累加，整数加法满足结合律
 *   4. 最终只做一次 float(sum) * inv_batch，消除浮点累加误差
 */
__global__ void softmax_ce_reduce_kernel(
    const float* __restrict__ loss_partial,
    const int*   __restrict__ top1_partial,
    const int*   __restrict__ top5_partial,
    float* __restrict__ loss,
    float* __restrict__ top1,
    float* __restrict__ top5,
    int batch,
    bool compute_top_metrics)
{
    extern __shared__ char smem_raw[];
    float* s_loss = reinterpret_cast<float*>(smem_raw);
    int*   s_top1 = reinterpret_cast<int*>(smem_raw + BLOCK_DIM * sizeof(float));
    int*   s_top5 = reinterpret_cast<int*>(smem_raw + BLOCK_DIM * (sizeof(float) + sizeof(int)));

    int tid = threadIdx.x;
    float loss_sum = 0.0f;
    int top1_sum = 0, top5_sum = 0;

    // 步长累加: 每个线程按固定间隔读取，保证确定性
    for (int i = tid; i < batch; i += blockDim.x) {
        loss_sum += loss_partial[i];
        if (compute_top_metrics) {
            top1_sum += top1_partial[i];
            top5_sum += top5_partial[i];
        }
    }

    s_loss[tid] = loss_sum;
    if (compute_top_metrics) {
        s_top1[tid] = top1_sum;
        s_top5[tid] = top5_sum;
    }
    __syncthreads();

    // Tree reduction in shared memory (固定配对模式)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_loss[tid] += s_loss[tid + s];
            if (compute_top_metrics) {
                s_top1[tid] += s_top1[tid + s];
                s_top5[tid] += s_top5[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *loss = s_loss[0];
        if (compute_top_metrics) {
            float inv_batch = 1.0f / static_cast<float>(batch);
            *top1 = static_cast<float>(s_top1[0]) * inv_batch;   // INT32 → FP32
            *top5 = static_cast<float>(s_top5[0]) * inv_batch;   // INT32 → FP32
        }
        // FWD 路径 (compute_top_metrics=false): 不写 top1/top5
        // R_RESULT 区由 ZERO_GRAD 图在每个 batch 之初清零，无需冗余写 0
    }
}
```

### 6.4 BWD Kernel

**完全不变**。纯逐元素操作，无归约、无原子操作。

---

## 七、Launch Wrapper 改动

文件: `src/backend/ops/dtensor/softmax_ce_op.cu`

### 7.1 FWD Launch (FP32)

```cpp
cudaError_t launch_softmax_ce_fwd_fp32(
    cudaStream_t s,
    const float* logits, const int* labels,
    float* loss, float* top1, float* top5,    // 保留参数，kernel 不再直接写入
    int* pred, float* probs,                   // 保留参数，kernel 不再写入
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial,                       // [NEW] ids_out[6]
    int* top1_partial,                         // [NEW] ids_out[7]
    int* top5_partial,                         // [NEW] ids_out[8]
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    // Step 1: FWD main kernel — 每个 block 写入 loss_partial[b]
    //         top1/top5/pred 参数传入但不操作（接口对齐）
    size_t smem = static_cast<size_t>(WARP_COUNT) * 2 * sizeof(float);
    softmax_ce_fwd_kernel<false><<<batch, BLOCK_DIM, smem, s>>>(
        static_cast<const void*>(logits), labels,
        loss, top1, top5, pred, probs,
        inv_scaling, scaling, batch_size, label_smoothing,
        loss_partial, top1_partial, top5_partial,
        batch, logits_stride, probs_stride, num_classes);

    // Step 2: 确定性归约 — 单 block 累加 loss_partial → loss
    //         FWD 不计算 top metrics; top1/top5 由 R_RESULT 区 ZERO_GRAD 保证为 0
    size_t reduce_smem = BLOCK_DIM * sizeof(float)
                       + BLOCK_DIM * sizeof(int) * 2;
    softmax_ce_reduce_kernel<<<1, BLOCK_DIM, reduce_smem, s>>>(
        loss_partial, top1_partial, top5_partial,
        loss, top1, top5, batch, false);

    return cudaGetLastError();
}
```

### 7.2 FWD Launch (AMP)

```cpp
cudaError_t launch_softmax_ce_fwd_amp(
    cudaStream_t s,
    const __half* logits, const int* labels,
    float* loss, float* top1, float* top5, int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size, const float* label_smoothing,
    float* loss_partial, int* top1_partial, int* top5_partial,   // [NEW]
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    size_t smem = static_cast<size_t>(WARP_COUNT) * 2 * sizeof(float);
    softmax_ce_fwd_kernel<true><<<batch, BLOCK_DIM, smem, s>>>(
        static_cast<const void*>(logits), labels,
        loss, top1, top5, pred, probs,
        inv_scaling, scaling, batch_size, label_smoothing,
        loss_partial, top1_partial, top5_partial,
        batch, logits_stride, probs_stride, num_classes);

    size_t reduce_smem = BLOCK_DIM * sizeof(float)
                       + BLOCK_DIM * sizeof(int) * 2;
    softmax_ce_reduce_kernel<<<1, BLOCK_DIM, reduce_smem, s>>>(
        loss_partial, top1_partial, top5_partial,
        loss, top1, top5, batch, false);

    return cudaGetLastError();
}
```

### 7.3 INF Launch (FP32)

```cpp
cudaError_t launch_softmax_ce_inf_fp32(
    cudaStream_t s,
    const float* logits, const int* labels,
    float* loss, float* top1, float* top5,
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial,                       // [NEW] ids_out[6]
    int* top1_partial,                         // [NEW] ids_out[7]
    int* top5_partial,                         // [NEW] ids_out[8]
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    size_t smem = (static_cast<size_t>(WARP_COUNT) * 2 + BLOCK_DIM) * sizeof(float)
                + BLOCK_DIM * sizeof(int);

    // Step 1: INF main kernel — 每个 block 写入 partial[b]
    softmax_ce_inf_kernel<false><<<batch, BLOCK_DIM, smem, s>>>(
        static_cast<const void*>(logits), labels, loss, top1, top5, pred, probs,
        inv_scaling, scaling, batch_size, label_smoothing,
        loss_partial, top1_partial, top5_partial,
        batch, logits_stride, probs_stride, num_classes);

    // Step 2: 确定性归约 — 单 block 累加 → loss, top1, top5
    size_t reduce_smem = BLOCK_DIM * sizeof(float)
                       + BLOCK_DIM * sizeof(int) * 2;
    softmax_ce_reduce_kernel<<<1, BLOCK_DIM, reduce_smem, s>>>(
        loss_partial, top1_partial, top5_partial,
        loss, top1, top5, batch, true);    // INF: 计算 top metrics

    return cudaGetLastError();
}
```

### 7.4 INF Launch (AMP)

```cpp
cudaError_t launch_softmax_ce_inf_amp(
    cudaStream_t s,
    const __half* logits, const int* labels,
    float* loss, float* top1, float* top5,
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial, int* top1_partial, int* top5_partial,   // [NEW]
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    size_t smem = (static_cast<size_t>(WARP_COUNT) * 2 + BLOCK_DIM) * sizeof(float)
                + BLOCK_DIM * sizeof(int);
    softmax_ce_inf_kernel<true><<<batch, BLOCK_DIM, smem, s>>>(
        static_cast<const void*>(logits), labels, loss, top1, top5, pred, probs,
        inv_scaling, scaling, batch_size, label_smoothing,
        loss_partial, top1_partial, top5_partial,
        batch, logits_stride, probs_stride, num_classes);

    size_t reduce_smem = BLOCK_DIM * sizeof(float)
                       + BLOCK_DIM * sizeof(int) * 2;
    softmax_ce_reduce_kernel<<<1, BLOCK_DIM, reduce_smem, s>>>(
        loss_partial, top1_partial, top5_partial,
        loss, top1, top5, batch, true);

    return cudaGetLastError();
}
```

### 7.5 BWD Launch

**完全不变**。

---

## 八、CPU 算子接口对齐

文件: `src/backend/ops/dtensor/softmax_ce_op.cpp`

### 8.1 前向声明更新

```cpp
#ifdef TR_USE_CUDA
extern cudaError_t launch_softmax_ce_fwd_fp32(
    cudaStream_t s,
    const float* logits, const int* labels,
    float* loss, float* top1, float* top5,
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial, int* top1_partial, int* top5_partial,   // [NEW]
    int batch, int logits_stride, int probs_stride, int num_classes);

extern cudaError_t launch_softmax_ce_fwd_amp(
    cudaStream_t s,
    const __half* logits, const int* labels,
    float* loss, float* top1, float* top5,
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial, int* top1_partial, int* top5_partial,   // [NEW]
    int batch, int logits_stride, int probs_stride, int num_classes);

extern cudaError_t launch_softmax_ce_inf_fp32(
    cudaStream_t s,
    const float* logits, const int* labels,
    float* loss, float* top1, float* top5,
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial, int* top1_partial, int* top5_partial,   // [NEW]
    int batch, int logits_stride, int probs_stride, int num_classes);

extern cudaError_t launch_softmax_ce_inf_amp(
    cudaStream_t s,
    const __half* logits, const int* labels,
    float* loss, float* top1, float* top5,
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial, int* top1_partial, int* top5_partial,   // [NEW]
    int batch, int logits_stride, int probs_stride, int num_classes);

// BWD 声明不变
#endif
```

### 8.2 CPU FWD Launch — 不变

```cpp
static void launch_softmax_ce_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    // ids_out 布局: [loss, inv_sc, pred, probs, top1, top5, loss_p, top1_p, top5_p]
    // 索引 [0]..[5] 与现有代码完全一致，无需修改
    // ids_out[6..8] 的 partial 指针存在但 CPU 不使用
    // top1/top5 由 R_RESULT 区 ZERO_GRAD 图保证为 0，CPU 无需额外写 0

    // ... 现有代码完全不变 ...
}
```

### 8.3 CPU INF Launch — 不变

```cpp
static void launch_softmax_ce_fp32_inf_cpu(CpuOpContext* op_ctx) {
    // ids_out 布局: [loss, inv_sc, pred, probs, top1, top5, loss_p, top1_p, top5_p]
    // top1 仍在 ids_out[4], top5 仍在 ids_out[5] — 与现有代码完全一致

    // ... 现有代码完全不变 ...
}
```

### 8.4 CPU BWD Launch — 不变

### 8.5 CUDA Dispatch 函数更新

仅需在每个 CUDA dispatch 函数中新增 3 行 partial 指针读取：

```cpp
// FWD CUDA dispatch 中新增:
float* loss_partial = static_cast<float*>(ctx.ptr_at(ids_out[6]));  // [NEW]
int*   top1_partial = static_cast<int*>(ctx.ptr_at(ids_out[7]));    // [NEW]
int*   top5_partial = static_cast<int*>(ctx.ptr_at(ids_out[8]));    // [NEW]

// 调用 launch_softmax_ce_fwd_fp32 时传入新参数
cudaError_t err_fwd = launch_softmax_ce_fwd_fp32(
    s, logits, labels, loss, top1, top5, pred, probs,
    inv_sc, scaling, batch_size, ls_ptr,
    loss_partial, top1_partial, top5_partial,   // [NEW]
    batch, stride, probs_stride, num_cls);
```

INF CUDA dispatch 同理，读取 `ids_out[6..8]` 并传入 launch wrapper。

---

## 九、接口变更汇总

### 9.1 FWD 接口

| 位置 | 当前 | 变更后 |
|------|------|--------|
| `ids_in` 数量 | 5 | 5（不变） |
| `ids_out` 数量 | 6 | **9** |
| `ids_out` 布局 | `[loss, inv_sc, pred, probs, top1, top5]` | `[loss, inv_sc, pred, probs, top1, top5, loss_p, top1_p, top5_p]` |
| loss 写入方式 | `atomicAdd(loss, ...)` | `loss_partial[b] = ...` → reduce kernel |
| top1/top5 | FWD kernel 不操作 | FWD kernel 不操作（R_RESULT 区由 ZERO_GRAD 图保证为 0） |
| CPU 索引 | 全部不变 | 全部不变 |

> **说明**: FWD 仍然不计算 top1/top5。训练图里的 `RANGE_ACCUM_METRICS` 会读 top1/top5，所以训练 metric 的 top1/top5 仍然是 0（pre-existing 行为）。如果需要训练精度指标，应作为后续需求单独实现。

### 9.2 INF 接口

| 位置 | 当前 | 变更后 |
|------|------|--------|
| `ids_in` 数量 | 5 | 5（不变） |
| `ids_out` 数量 | 6 | **9** |
| `ids_out` 布局 | `[loss, inv_sc, pred, probs, top1, top5]` | `[loss, inv_sc, pred, probs, top1, top5, loss_p, top1_p, top5_p]` |
| loss 写入方式 | `atomicAdd(loss, ...)` | `loss_partial[b] = ...` → reduce kernel |
| top1/top5 写入方式 | `atomicAdd(top1, inv_batch)` | `top1_partial[b] = 0/1` (INT32) → reduce kernel |
| CPU 索引 | 全部不变 | 全部不变 |

### 9.3 BWD 接口

**不变**。`ids_in`: [grad_output, probs, inv_scaling, scaling, label, ls] (6)，`ids_out`: [dlogits] (1)。

---

## 十、边界条件处理

| 场景 | 处理方式 |
|------|---------|
| **last batch** (实际 batch < variant batch) | TEMP 按 variant 的 `input.n()` 分配；kernel grid 为 `logits_dt.shape.n()`（实际 batch）；reduce kernel 只读 `partial[0..batch-1]`；`inv_batch = 1.0f / batch` 正确 |
| `batch == 1` | reduce kernel: `loss = partial[0]`，正确 |
| `num_classes < 5` | INF kernel: `top5_partial[b] = top1_partial[b]`（与现有逻辑一致） |
| label smoothing | loss 公式不变，只是写入 partial 缓冲区 |
| AMP (FP16 logits) | logits 为 FP16，内部计算用 FP32，`loss_partial` 为 FP32，规约确定性不变 |
| 多 SoftmaxCE 层 | 每层独立 TEMP 缓冲区（`infer_softmaxce_tensors` 每层调用一次），无竞态 |
| FWD top1/top5 | FWD kernel 不操作 top1_partial/top5_partial；reduce kernel 在 `compute_top_metrics=false` 时不写 top1/top5；R_RESULT 区由 ZERO_GRAD 图保证为 0 |

---

## 十一、确定性保证分析

| 组件 | 为什么确定 |
|------|-----------|
| FWD/INF main kernel | 每个 block 写入 `partial[b]`（b = blockIdx.x），无竞争 |
| reduce kernel | 单 block，线程按 `tid, tid+BLOCK_DIM, ...` 固定 stride 读取，shared memory tree reduction 固定配对模式 |
| top1/top5 计数 | INT32 整数加法 → 精确，最终 `float(sum) / batch` 只做一次除法，消除浮点累加误差 |
| CPU | 串行确定性，不依赖 partial 缓冲区 |
| 同 stream 顺序发射 | main kernel + reduce kernel 在同一个 stream 上顺序执行，无竞态 |

---

## 十二、内存与性能影响

| 指标 | 影响 |
|------|------|
| 新增显存 (per layer) | `variant_batch * (4 + 4 + 4) = 12 * variant_batch` 字节（batch=128 时仅 1.5 KiB） |
| 额外 kernel 发射 | FWD/INF 各 +1 个单 block reduce kernel（~256 线程） |
| 性能影响 | 单 block reduce kernel 启动开销 ~几微秒，可忽略 |
| 消除 atomicAdd 竞争 | 大 batch 下 block 间无全局内存竞争，可能略微提升性能 |
| 消除冗余写 0 | FWD 不再写 top1_partial/top5_partial 和 reduce kernel 的 top1/top5，节省少量 global memory 写入 |

---

## 十三、实施顺序

### Phase 1: LayerDescriptor + Compiler（验证图构建与内存分配）

1. 修改 `infer_softmaxce_tensors` 新增 3 个 TEMP desc（共 7 个 desc）
2. 修改 `build_softmaxce_forward/inference` 的 `output_indices` 为 `{3, 2, 1, 4, 5, 6}`
3. 修改 `build_softmaxce_backward` 的 desc 数量检查为 `< 7`
4. 修改 `compiler.cpp` 两处注入逻辑，显式重排 `output_ids`
5. 同步更新文件头部/段注释（`softmax_ce_op.cpp`、`layer_descriptor_registry.cpp`、`compiler.cpp`）
6. **编译验证**：确认图节点 `output_ids` 变为 9 个，MemoryPlan 正确分配 TEMP

### Phase 2: CUDA Kernel 改造

7. 修改 `softmax_ce_fwd_kernel` 签名与逻辑（新增 3 个 partial 参数，保留 top1/top5/pred 参数，删除 `atomicAdd`，添加 `(void)` 消除警告）
8. 修改 `softmax_ce_inf_kernel` 签名与逻辑（新增 3 个 partial 参数，删除所有 `atomicAdd`，修正 `inv_scaling` 写入条件，添加 `(void)` 消除警告）
9. 新增 `softmax_ce_reduce_kernel`
10. 更新 4 个 launch wrapper (`fwd_fp32/amp`, `inf_fp32/amp`)

### Phase 3: CPU 接口对齐

11. 更新 `softmax_ce_op.cpp` 中的前向声明
12. CUDA dispatch 函数新增 `ids_out[6..8]` 的 partial 指针读取
13. CPU launch 函数不变（接口已对齐）

### Phase 4: 验证

14. 编译通过，无 unused parameter 警告
15. 单卡固定输入 100 次运行，loss/top1/top5 完全一致
16. last batch 正确性验证
17. CPU vs GPU 一致性验证（注意 loss 的 scaling 差异，见 14.2）
18. DDP 多卡验证

---

## 十四、验证计划

### 14.1 确定性验证

```cpp
// 固定输入，100 次运行，结果应 bitwise 一致
for (int run = 0; run < 100; ++run) {
    float loss, top1, top5;
    run_softmax_ce_inf(logits, labels, &loss, &top1, &top5);
    // 100 次 loss/top1/top5 完全一致
}
```

### 14.2 CPU/GPU 一致性

> **重要**: 当前 CPU 的 loss 会乘以 scaling（`*loss *= scaling`），而 GPU 不乘。这是历史遗留差异，本次不修正。在进行 CPU/GPU 一致性验证时，需要将 CPU 输出的 loss 除以 scaling 再与 GPU 比较，或者在测试代码中临时关闭 CPU 的 scaling 乘法。

同一组随机数据分别跑 CPU 与 GPU（FP32），比较：
- `top1` / `top5`：允许 1e-6 级 FP32 尾差
- `loss`：将 CPU 的 `loss / scaling` 与 GPU 的 `loss` 比较，允许 1e-6 级 FP32 尾差

### 14.3 Last Batch

- 构造实际 batch = local_batch_size - 7
- 确认 `inv_scaling == 1.0f / actual_batch`
- 确认 top1/top5 = count / actual_batch，而非除以 local_batch_size

### 14.4 DDP

- 各 rank local loss/top1/top5 一致
- allreduce 后 global accum metrics 与单卡 global batch 结果一致

### 14.5 训练回归

- NIN CIFAR-10 训练一个 epoch，loss 曲线与改造前一致
- VGG16 ImageNet 训练若干 epoch，确认 top1 走势无异常

---

## 十五、风险与回滚

| 风险 | 缓解 |
|------|------|
| `output_ids` 从 6→9 导致其他模块索引错位 | 通过显式重排使前 6 个索引语义不变；所有对 `top1`/`top5` 的引用通过 `baseline().top1`/`.top5` 的 tensor ID 进行，不依赖 output 顺序 |
| CPU INF 的 `top1`/`top5` 索引不变 | 已验证 indices [4]/[5] 保持不变 |
| AMP 的 `__half` 与 partial 的 FP32/INT32 混用 | partial 缓冲区始终为 FP32/INT32，只有 logits 是 FP16 |
| loss 不乘 scaling 与 CPU 不一致 | 这是**历史遗留差异**，本次不修正。14.2 验证计划已说明如何规避 |
| 未来 TEMP 区域引入 channel padding | 已在 4.1 节添加注释说明依赖 `padded_c=1`，若引入 padding 需改为 `partial[b * stride]` |
| **回滚** | 恢复 `layer_descriptor_registry.cpp`、`compiler.cpp`、`softmax_ce_op.cu`、`softmax_ce_op.cpp` 即可 |

---

## 十六、与 EKT1/EKT2/EKT3 的关键差异总结

| 设计决策 | EKT1 | EKT2 | EKT3 | **EKT_FINAL** | 理由 |
|---------|------|------|------|---------------|------|
| Compiler 重排 | 否 | 是 | 否 | **是** | 最小化现有代码改动 |
| scaling 乘入 loss | 否 | 是 | 是 | **否** | 保持 GPU 现有行为，避免改变训练语义 |
| Reduce 线程数 | BLOCK_DIM | 1 | BLOCK_DIM | **BLOCK_DIM** | 更好并行度，树形归约确定性不变 |
| Reduce 参数 | `batch` | `batch`+`scaling` | `batch_size_ptr` | **`batch`** | `batch` 已是实际值，无需额外指针 |
| CPU 索引改动 | 需要 | 不需要 | 需要 | **不需要** | 显式重排使前 6 个索引不变 |
| Kernel 参数顺序 | 紧凑 | 保留 top1/5/pred | 保留 top1/5/pred | **保留 top1/5/pred** | 接口对齐，便于回滚 |
| FWD 写 top1/top5=0 | 是 | 是 | 是 | **否** | R_RESULT 区由 ZERO_GRAD 图保证清零，写 0 是冗余操作 |
| INF inv_scaling 写入条件 | `b==0` | `b==0` | `b==0` | **`b==0 && tid==0`** | 避免 256 线程冗余写 |

---

## 十七、结论

本方案综合 EKT1/EKT2/EKT3 的精华，并纳入了 S_REV.md 的评审意见，采用以下核心策略：

1. **Layer-level TEMP 张量**：通过 `infer_softmaxce_tensors` 在 `T_TEMP_FP32` / `T_TEMP_INT32` 区域预分配 `loss_partial`、`top1_partial`、`top5_partial`，每个 variant 独立分配，避免全局共享竞态。
2. **Compiler 显式重排**：保持 `top1`/`top5` 在 `output_ids[4]`/`[5]`，使 CPU 和 CUDA dispatch 现有代码无需修改索引。
3. **FWD/INF kernel 写 partial**：移除所有 `atomicAdd`，每个 block 写入独立的 `partial[blockIdx.x]`，无竞争。FWD 不操作 top1/top5 相关缓冲区。
4. **统一确定归约 kernel**：单 block BLOCK_DIM 线程树形归约，top1/top5 用 INT32 累加，最后转 FP32，消除浮点累加顺序不确定性。FWD 路径不写 top1/top5（R_RESULT 区由 ZERO_GRAD 图保证为 0）。
5. **保持 GPU 现有 loss 语义**：不乘 scaling，避免改变训练行为。
6. **CPU 接口对齐**：`output_ids` 布局一致，CPU 不使用 partial 但接口包含相应指针。
7. **工程细节完善**：消除 unused 参数警告、修正 INF `inv_scaling` 写入条件、统一 desc 数量检查、添加依赖注释、修正验证计划。

该方案满足全部 7 条硬性要求，改动范围最小化，性能影响可忽略，且彻底消除 SoftmaxCE FWD/INF 的不可复现问题。