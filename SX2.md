# SOFTMAX_CE 融合算子 —— 最优综合实施方案

> 版本: V1.0  
> 日期: 2026-05-16  
> 状态: 待实施

---

## 一、核心决策

| 问题 | 决策 |
|------|------|
| 算子命名 | **替换**旧枚举：`CROSS_ENTROPY_LOSS*` → `SOFTMAX_CE_*`（4个） |
| labels 来源 | 数据加载器写入 `I_A_LABEL` / `I_B_LABEL`，算子**直接读取**不复制 |
| labels 传递 | `MemoryPlan` 显式保存输入缓冲区 DTensor ID，`Compiler` 注入到 GraphNode.input_ids |
| 结果区 | **新增** `R_METRIC_INT32`（Top-1/Top-5 正确数），`predicted_labels` 放 `T_TEMP_INT32` |
| FWD 中间量 | `softmax_probs` 输出到 `T_TEMP_FP32`（供 BWD 复用），**不走 workspace** |
| BWD 输出 | grad **in-place 覆盖 logits**（框架 dX 覆盖 X 约定），乘以 scaling_factor |
| 数值精度 | **无论 FP32/AMP，内部计算全 FP32**；AMP 仅 logits 输入/grad 输出做 FP16↔FP32 转换 |
| CUDA 路径 | **手写融合 kernel**，不用 cuDNN FE（无对应融合算子，且需输出 top1/top5） |
| CPU 路径 | 朴素循环实现，**不展开 one-hot**，不用 Eigen（无矩阵运算） |

---

## 二、需要修改的文件（10个）

| # | 文件 | 改动性质 | 行数估算 |
|---|------|---------|---------|
| 1 | `include/renaissance/core/types.h` | 新增 `R_METRIC_INT32` Region；`NUM_REGIONS` +1 | +3 |
| 2 | `include/renaissance/graph/memory_plan.h` | 新增输入缓冲区 ID 存储与访问接口 | +10 |
| 3 | `src/graph/memory_plan.cpp` | `alloc_input_buffers` 保存 ID；`region_to_string` 新增 case | ~10 |
| 4 | `include/renaissance/graph/op_kind.h` | 删除 5 个旧 `CROSS_ENTROPY_LOSS*`；新增 4 个 `SOFTMAX_CE_*` | ~10 |
| 5 | `src/graph/op_kind.cpp` | 字符串映射同步替换 | ~10 |
| 6 | `src/graph/layer_descriptor_registry.cpp` | 重写 `infer/build_softmaxce_*`；扩展 TensorDesc 列表 | ~40 |
| 7 | `src/graph/compiler.cpp` | Phase 4 为 SoftmaxCE 注入 labels ID 到 input_ids | ~15 |
| 8 | `src/backend/ops/dtensor/softmax_ce_op.cpp` | **新文件**：CPU launch + CUDA launch 分发 + 注册 | ~250 |
| 9 | `src/backend/ops/dtensor/softmax_ce_op.cu` | **新文件**：CUDA 融合 kernels（FP32 + AMP） | ~280 |
| 10 | `src/CMakeLists.txt` | 添加 `.cpp` / `.cu` | +2 |
| 11 | `include/renaissance/backend/op_registry.h` | 声明 `register_op_softmax_ce()` | +1 |
| 12 | `src/backend/op_registry.cpp` | 调用注册；`require_warmup` 判断（手写 kernel 不需要 warmup） | +2 |
| 13 | `src/backend/op_stream_policy.cpp` | 4 个新算子的流策略（`COMP_1`） | +5 |

---

## 三、基础设施改动

### 3.1 MemoryPlan —— 保存输入缓冲区 ID

**问题**：当前 `alloc_input_buffers()` 返回的 `InputBuffers` 被 `compiler.cpp` 丢弃，labels 的 DTensor ID 无处可查。

**方案**：`MemoryPlan` 新增成员保存这 4 个特殊 ID：

```cpp
// memory_plan.h —— MemoryPlan 类内新增
int32_t input_label_a_id_ = -1;
int32_t input_data_a_id_  = -1;
int32_t input_label_b_id_ = -1;
int32_t input_data_b_id_  = -1;

public:
[[nodiscard]] int32_t input_label_a_id() const noexcept { return input_label_a_id_; }
[[nodiscard]] int32_t input_data_a_id()  const noexcept { return input_data_a_id_; }
[[nodiscard]] int32_t input_label_b_id() const noexcept { return input_label_b_id_; }
[[nodiscard]] int32_t input_data_b_id()  const noexcept { return input_data_b_id_; }
```

```cpp
// memory_plan.cpp —— alloc_input_buffers 中保存
InputBuffers MemoryPlan::alloc_input_buffers(...) {
    auto la = alloc_impl(label_shape, DType::INT32, Region::I_A_LABEL);
    auto da = alloc_impl(data_shape,  dtype,        Region::I_A_DATA);
    auto lb = alloc_impl(label_shape, DType::INT32, Region::I_B_LABEL);
    auto db = alloc_impl(data_shape,  dtype,        Region::I_B_DATA);
    input_label_a_id_ = la.id;
    input_data_a_id_  = da.id;
    input_label_b_id_ = lb.id;
    input_data_b_id_  = db.id;
    return {la, da, lb, db};
}
```

### 3.2 Region 枚举 —— 新增结果区

```cpp
// types.h
enum class Region : uint8_t {
    // ... 现有 65 个 ...
    T_TEMP_INT8,          // 065
    R_METRIC_INT32,       // 066  ← 新增：Top-1/Top-5 正确数、其他推理指标
    DEFAULT = B_PREV_MEAN,
    NUM_REGIONS = 66      // 65 → 66
};
```

同步更新 `memory_plan.cpp` 的 `region_to_string`。

### 3.3 ComputeOp 枚举 —— 替换旧名

```cpp
// op_kind.h
enum class ComputeOp : uint16_t {
    // ... 其他算子 ...

    // 删除以下 5 个旧枚举（及其在 op_kind.cpp / format_params 中的 case）：
    // CROSS_ENTROPY_LOSS,
    // CROSS_ENTROPY_LOSS_FP32_FWD,
    // CROSS_ENTROPY_LOSS_FP32_BWD,
    // CROSS_ENTROPY_LOSS_AMP_FWD,
    // CROSS_ENTROPY_LOSS_AMP_BWD,

    // 新增 4 个：
    SOFTMAX_CE_FP32_FWD,
    SOFTMAX_CE_FP32_BWD,
    SOFTMAX_CE_AMP_FWD,
    SOFTMAX_CE_AMP_BWD,

    ALLREDUCE_SUM,
    // ...
};
```

> **注意**：`LossParams` 结构体**保留复用**，无需新增 `SoftmaxCEParams`。`num_classes` 和 `label_smoothing` 已足够。

---

## 四、LayerDescriptor 与 Compiler 改动

### 4.1 TensorDesc 列表（8张量）

```cpp
std::vector<TensorDesc> infer_softmaxce_tensors(
    const Shape& input, const OpParams& p, const InferContext& ctx)
{
    int batch = input.n();
    const auto* lp = std::get_if<LossParams>(&p.data);
    int num_classes = lp ? lp->num_classes : input.c();
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    return {
        // 0: loss（标量）
        TensorDesc{"ce_loss",            Shape{1,1,1,1},      Region::S_SCALAR_FP32,   DType::FP32},
        // 1: ce_output / logits（输入来自前一层，输出 softmax_probs 或 grad 覆盖）
        TensorDesc{"ce_output",          input,               select_feature_region(ctx), feat_dt},
        // 2: scaling_factor（标量）
        TensorDesc{"scaling_factor",     Shape{1,1,1,1},      Region::S_SCALAR_FP32,   DType::FP32},
        // 3: inv_scaling_factor（标量）
        TensorDesc{"inv_scaling_factor", Shape{1,1,1,1},      Region::S_SCALAR_FP32,   DType::FP32},
        // 4: top1_correct（标量 INT32）
        TensorDesc{"top1_correct",       Shape{1,1,1,1},      Region::R_METRIC_INT32,  DType::INT32},
        // 5: top5_correct（标量 INT32）
        TensorDesc{"top5_correct",       Shape{1,1,1,1},      Region::R_METRIC_INT32,  DType::INT32},
        // 6: predicted_labels（batch 个 INT32）
        TensorDesc{"predicted_labels",   Shape{batch,1,1,1},  Region::T_TEMP_INT32,    DType::INT32},
        // 7: softmax_probs（FWD 输出供 BWD 复用）
        TensorDesc{"softmax_probs",      input,               Region::T_TEMP_FP32,     DType::FP32},
    };
}
```

### 4.2 SubgraphPattern

```cpp
SubgraphPattern build_softmaxce_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_FWD
               : ComputeOp::SOFTMAX_CE_FP32_FWD;
    // input_indices 仅含 descs 内部张量：scaling_factor(2)
    // labels 由 Compiler 显式注入（见 4.3）
    n.input_indices  = {2};           // scaling_factor
    n.output_indices = {0, 4, 5, 6, 7}; // loss + top1 + top5 + pred + probs
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_softmaxce_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_BWD
               : ComputeOp::SOFTMAX_CE_FP32_BWD;
    // BWD 读取：softmax_probs(7) + scaling_factor(2)
    // labels 由 Compiler 显式注入
    n.input_indices  = {7, 2};
    // 输出：grad 覆盖 ce_output(1) —— 通过 idx=1 让 compiler 映射到 logits 张量
    n.output_indices = {1};
    p.nodes.push_back(n);
    return p;
}
```

### 4.3 Compiler —— 注入 labels

在 `compiler.cpp` Phase 4 的 Forward 和 Backward 循环中，为 `SoftmaxCE` 层追加 labels ID：

```cpp
// Forward 循环内，构建完 gn 后、append 前：
if (layer.kind == LayerKind::SoftmaxCE) {
    int32_t label_id = memory_plan.input_label_a_id();  // 训练用 A 桶
    if (label_id >= 0) {
        gn.input_ids.push_back(label_id);
    }
}

// Backward 循环内，同理：
if (layer.kind == LayerKind::SoftmaxCE) {
    int32_t label_id = memory_plan.input_label_a_id();
    if (label_id >= 0) {
        gn.input_ids.push_back(label_id);
    }
}
```

> **注入后的 input_ids 顺序**：
> - FWD: `[prev_output_id(logits), scaling_factor_tid, label_tid]`
> - BWD: `[prev_grad_id, softmax_probs_tid, scaling_factor_tid, label_tid]`

---

## 五、算子 I/O 约定

### 5.1 FWD（FP32 与 AMP 共用结构）

| 接口 | 索引 | 内容 | 说明 |
|------|------|------|------|
| input_ids[0] | — | logits | 跨层链注入，FP32(AMP下FP16) |
| input_ids[1] | — | scaling_factor | S_SCALAR_FP32，标量 |
| input_ids[2] | — | labels | I_A_LABEL，INT32，batch 个紧凑连续 |
| output_ids[0] | — | ce_loss | S_SCALAR_FP32，标量 |
| output_ids[1] | — | top1_correct | R_METRIC_INT32，标量 |
| output_ids[2] | — | top5_correct | R_METRIC_INT32，标量 |
| output_ids[3] | — | predicted_labels | T_TEMP_INT32，[batch] |
| output_ids[4] | — | softmax_probs | T_TEMP_FP32，[batch, num_classes] |

### 5.2 BWD

| 接口 | 索引 | 内容 | 说明 |
|------|------|------|------|
| input_ids[0] | — | prev_grad / logits | 跨层链注入，BWD 覆盖目标 |
| input_ids[1] | — | softmax_probs | T_TEMP_FP32，FWD 输出 |
| input_ids[2] | — | scaling_factor | S_SCALAR_FP32 |
| input_ids[3] | — | labels | I_A_LABEL，INT32 |
| output_ids[0] | — | grad_logits | **in-place 覆盖 input_ids[0]** |

---

## 六、CUDA Kernel 设计

### 6.1 FWD Kernel（模板化 FP32/AMP）

```cuda
template<bool IS_AMP>
__global__ void softmax_ce_fwd_kernel(
    const float*  logits_fp32,   // IS_AMP=false
    const __half* logits_fp16,   // IS_AMP=true
    const int*    labels,
    float*        loss,          // 标量，atomicAdd 累加
    int*          top1_correct,  // 标量，atomicAdd
    int*          top5_correct,  // 标量，atomicAdd
    int*          predicted_labels,  // [batch]
    float*        softmax_probs, // [batch, num_classes] Temp区
    float         scaling_factor,
    int           batch,
    int           num_classes)
{
    // 每个 block 处理一个 sample
    int b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ float smem[];
    float* s_logits = smem;  // [blockSize] 或 [num_classes] 视策略而定

    const int tid = threadIdx.x;
    int label = labels[b];

    // 1. 加载 logits 到 FP32 寄存器/共享内存，同时求 local max
    float local_max = -INFINITY;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        float v = IS_AMP ? __half2float(logits_fp16[b * num_classes + c])
                         : logits_fp32[b * num_classes + c];
        // 可写入 smem 供后续复用
        smem[c] = v;  // 若 num_classes <= max_shared_mem / 4
        local_max = fmaxf(local_max, v);
    }

    // 2. Warp + Block 归约求 global_max
    // ... (shuffle + tree reduction) ...
    float global_max = ...;

    // 3. 计算 exp 和 sum_exp
    float local_sum = 0.0f;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        float e = expf(smem[c] - global_max);
        smem[c] = e;  // 复用 smem 存 exp
        local_sum += e;
    }
    // Block 归约求 sum_exp
    float sum_exp = ...;

    // 4. 归一化写入 softmax_probs + 维护 Top-5
    Top5 local_top5;  // 寄存器内小根堆/插入排序，5个元素
    for (int c = tid; c < num_classes; c += blockDim.x) {
        float prob = smem[c] / sum_exp;
        softmax_probs[b * num_classes + c] = prob;
        local_top5.push(prob, c);  // 或直接用 logits 值维护 top5
    }
    // Warp 级 Top-5 合并 ...

    // 5. Thread 0 计算 loss、top1、top5、predicted_label
    if (tid == 0) {
        float label_prob = smem[label] / sum_exp;  // 注意：若 smem 被覆盖为 exp，需用原始 logits
        // 更稳妥：ce = -(logits[label] - global_max) + log(sum_exp)
        float ce = -smem_original[label] + global_max + logf(sum_exp);
        atomicAdd(loss, ce);

        int pred = ...;  // 从 Top-5 取第 0 个
        predicted_labels[b] = pred;
        if (pred == label) atomicAdd(top1_correct, 1);
        if (top5_contains_label) atomicAdd(top5_correct, 1);
    }
}
```

**关键优化**：
- **数值稳定**：`ce = -logits[label] + max + log(sum_exp)`，不通过 `log(prob)` 间接计算
- **共享内存布局**：若 `num_classes=1000`，需 `1000*4=4KB`，单 block 轻松容纳
- **Top-5**：每线程维护局部 Top-5，warp shuffle 合并，block 级再合并
- **初始化**：`loss/top1/top5` 标量需在 kernel launch 前 `cudaMemsetAsync(..., 0)`
- **Block 大小**：根据 num_classes 动态选择（128/256/512）

### 6.2 BWD Kernel

```cuda
template<bool IS_AMP>
__global__ void softmax_ce_bwd_kernel(
    const float* probs,      // [batch, num_classes] FWD输出的Temp区
    const int*   labels,
    float        scaling_factor,
    float*       grad_fp32,  // IS_AMP=false
    __half*      grad_fp16,  // IS_AMP=true
    int          batch,
    int          num_classes)
{
    int b = blockIdx.x;
    if (b >= batch) return;

    float scale = scaling_factor / batch;
    int label = labels[b];

    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        float g = probs[b * num_classes + c] * scale;
        if (c == label) g -= scale;
        if (IS_AMP)
            grad_fp16[b * num_classes + c] = __float2half(g);
        else
            grad_fp32[b * num_classes + c] = g;
    }
}
```

**特点**：
- 无需 shared memory reduction，每个线程独立计算
- 条件判断 `c == label` 替代 one-hot 展开
- AMP 版本仅在写入时做 `__float2half`

---

## 七、CPU 实现

### 7.1 FWD CPU

```cpp
static void launch_softmax_ce_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    float* logits = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float* scaling = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    int* labels = static_cast<int*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));

    float* loss = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    int* top1 = static_cast<int*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));
    int* top5 = static_cast<int*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[2]));
    int* pred = static_cast<int*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[3]));
    float* probs = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[4]));

    int batch = op_ctx->input_shape.n;
    int num_classes = op_ctx->input_shape.c;
    double loss_sum = 0.0;
    int t1 = 0, t5 = 0;

    for (int b = 0; b < batch; ++b) {
        float* row = logits + b * num_classes;
        float* prob_row = probs + b * num_classes;
        int label = labels[b];

        float max_val = row[0];
        for (int i = 1; i < num_classes; ++i) max_val = std::max(max_val, row[i]);

        double sum_exp = 0.0;
        for (int i = 0; i < num_classes; ++i) {
            prob_row[i] = std::exp(row[i] - max_val);
            sum_exp += prob_row[i];
        }
        for (int i = 0; i < num_classes; ++i) prob_row[i] /= (float)sum_exp;

        loss_sum += -(row[label] - max_val) + std::log(sum_exp);

        // Top-1
        int best_idx = 0;
        for (int i = 1; i < num_classes; ++i)
            if (prob_row[i] > prob_row[best_idx]) best_idx = i;
        pred[b] = best_idx;
        if (best_idx == label) ++t1;

        // Top-5：5次扫描找前5大（1000*5=5000次比较，可接受）
        bool in_top5 = false;
        for (int k = 0; k < 5; ++k) {
            int bi = -1;
            float bv = -1e30f;
            for (int i = 0; i < num_classes; ++i) {
                if (prob_row[i] > bv) { bv = prob_row[i]; bi = i; }
            }
            if (bi == label) in_top5 = true;
            prob_row[bi] = -1e30f;  // 标记已选
        }
        if (in_top5) ++t5;
    }

    loss[0] = (float)(loss_sum / batch * scaling[0]);
    top1[0] = t1;
    top5[0] = t5;
}
```

### 7.2 BWD CPU

```cpp
static void launch_softmax_ce_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    float* probs = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    float* scaling = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));
    int* labels = static_cast<int*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[3]));
    float* grad = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int batch = op_ctx->input_shape.n;
    int num_classes = op_ctx->input_shape.c;
    float scale = scaling[0] / batch;

    for (int b = 0; b < batch; ++b) {
        float* p = probs + b * num_classes;
        float* g = grad + b * num_classes;
        int label = labels[b];
        for (int i = 0; i < num_classes; ++i) {
            float v = p[i] * scale;
            if (i == label) v -= scale;
            g[i] = v;
        }
    }
}
```

### 7.3 AMP CPU —— 不支持

```cpp
static void launch_softmax_ce_amp_fwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_FWD is not supported on CPU");
}
```

---

## 八、CUDA Launch 分发层

```cpp
#ifdef TR_USE_CUDA
static void launch_softmax_ce_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const DTensor& dt_logits = mp.get_dtensor(node.input_ids[0]);
    int batch = dt_logits.shape().n();
    int num_classes = dt_logits.shape().c();

    const float* logits = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const int* labels   = static_cast<const int*>(ctx.ptr_at(node.input_ids[2]));

    float* loss = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    int* top1   = static_cast<int*>(ctx.ptr_at(node.output_ids[1]));
    int* top5   = static_cast<int*>(ctx.ptr_at(node.output_ids[2]));
    int* pred   = static_cast<int*>(ctx.ptr_at(node.output_ids[3]));
    float* probs = static_cast<float*>(ctx.ptr_at(node.output_ids[4]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;

    // 初始化累加标量为 0
    cudaMemsetAsync(loss, 0, sizeof(float), s);
    cudaMemsetAsync(top1, 0, sizeof(int), s);
    cudaMemsetAsync(top5, 0, sizeof(int), s);

    int block_size = (num_classes <= 128) ? 128 : (num_classes <= 512) ? 256 : 512;
    size_t shared_mem = num_classes * sizeof(float);  // 存 logits → exp

    softmax_ce_fwd_kernel<false><<<batch, block_size, shared_mem, s>>>(
        logits, nullptr, labels, loss, top1, top5, pred, probs,
        scaling_factor_value, batch, num_classes);

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif
```

> **scaling_factor_value 获取**：从 `node.input_ids[1]` 对应的 DTensor 读取。注意：在 CUDA graph capture 中，若 scaling_factor 是 host 常量，可直接传入 kernel 参数；若为 device 内存，需在 kernel 中再读一次。

---

## 九、注册

```cpp
void register_op_softmax_ce() {
    // FP32 FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_FWD;
        e.launch_cpu = launch_softmax_ce_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_fwd_cuda;
#endif
    }
    // FP32 BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_BWD;
        e.launch_cpu = launch_softmax_ce_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_bwd_cuda;
#endif
    }
    // AMP FWD / BWD（CPU 抛错）
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_FWD;
        e.launch_cpu = launch_softmax_ce_amp_fwd_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_fwd_cuda;
#endif
    }
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_BWD;
        e.launch_cpu = launch_softmax_ce_amp_bwd_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_bwd_cuda;
#endif
    }
}
```

---

## 十、验证计划

| 验证项 | 方法 |
|--------|------|
| 数值正确性 | 与 PyTorch `F.cross_entropy` + `F.softmax` 对比，MSE < 1e-5 |
| Top-1/Top-5 | 构造已知 logits/labels，验证正确数符合预期 |
| 梯度正确性 | 与 PyTorch `loss.backward()` 对比 grad |
| AMP 精度 | FP16 输入下，内部 FP32 计算，结果与 FP32 版本 MSE < 1e-4 |
| In-place 覆盖 | 验证 BWD 后原 logits 内存已被 grad 覆盖 |
| 零 one-hot | 代码审查：CPU/CUDA 均无 `num_classes x batch` 的中间分配 |
| 编译通过 | MSVC /O2 + CUDA 13.1 全量编译无警告 |

---

## 十一、实施顺序（推荐）

```
Phase 1（编译器框架）:
  → types.h (R_METRIC_INT32)
  → memory_plan.h/cpp (保存输入缓冲区 ID)
  → op_kind.h/cpp (替换枚举与字符串)
  → layer_descriptor_registry.cpp (扩展 TensorDesc)
  → compiler.cpp (注入 labels)
  → op_stream_policy.cpp (流策略)
  → 编译验证

Phase 2（算子实现）:
  → softmax_ce_op.cu (CUDA kernels)
  → softmax_ce_op.cpp (CPU + launch 分发 + 注册)
  → op_registry.h/cpp (调用注册)
  → CMakeLists.txt
  → 编译验证

Phase 3（测试）:
  → Python 参考脚本（对比 PyTorch）
  → C++ standalone 测试
  → Composite 端到端训练验证
```

---

## 十二、风险与缓解

| 风险 | 缓解 |
|------|------|
| `CROSS_ENTROPY_LOSS*` 被其他代码引用 | 全局 grep 替换；若遗留引用会导致编译错误，易发现 |
| Top-5 CUDA kernel 复杂度高 | 先用 warp-shuffle + 寄存器 Top-5 实现；若超时再优化 |
| labels ID 注入时机 | 确保在 `gn.input_ids` 已填入 descs 映射后、append 前注入 |
| AMP 下 logits 步幅非连续 | kernel 中按 DTensor stride 访问（通过 `n_stride_cuda()` 等） |
| `cudaMemsetAsync` 在 CUDA Graph capture 中 | 使用 `cudaGraph` 兼容的初始化方式，或在 capture 前预清零 |
