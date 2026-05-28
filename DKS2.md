# Last Batch 处理 — 代码审查与修复建议

> 基于 LST.md 中三位小伙伴的分析，结合对实际代码的深入检查，综合报告。

---

## 一、框架设计意图 vs 当前实现

### 1.1 设计意图（来自 `Compiler::Result` 和 `GraphAtlas`）

框架设计上明确支持 **6 个变体**：

| 变体索引 | 用途 | batch_size |
|---|---|---|
| v=0 | train_base（训练常规 batch） | local_batch_size |
| v=1 | train_last（训练 last batch） | last_train_batch_size |
| v=2 | train_lowres（低分辨率训练） | local_batch_size 或其他 |
| v=3 | ... | ... |
| v=4 | val_base（验证常规 batch） | local_batch_size |
| v=5 | val_last（验证 last batch） | last_val_batch_size |

`GraphAtlas::build()` 的设计（见 `graph_atlas.h` 注释和 `CAP_FINAL.md`）：
- **Shape 无关图**（TRANSFER/COMM/OPTIMIZER/CLEAR 等）：`is_shape_invariant_graph = true`，全部 6 变体共享同一个 MemoryPlan，shape_id = `kShapeInvariant`
- **Shape 相关图**（DEEP_FWD_BWD、FIRST_LAYER_FWD_A/B、FIRST_LAYER_BWD_A/B、INF_MAIN_A/B 等）：`is_shape_invariant_graph = false`，**每个变体各自使用独立的 MemoryPlan 和 ShapeId**，应分别捕获 CUDA Graph

用户明确指示：
> "按照我们的框架，对于512/402的训练会分别捕获Graph，对于512/106的验证会分别捕获Graph，并且在run的时候，根据形状选择正确的Graph。"

### 1.2 当前实现状态

| 组件 | 状态 | 说明 |
|---|---|---|
| `GraphAtlas::build()` | ❌ **未实现** | `src/graph/graph_atlas.cpp` **文件不存在**，只有头文件声明 |
| `deep_learning_task.cpp` 运行时变体切换 | ❌ **未实现** | `build_exec_table()` 中 `atlas.index(0, gid)` **硬编码 variant 0**，没有 last batch 的 graph 选择逻辑 |
| `run_train_epoch_gpu()` last batch graph | ❌ **使用相同 graph** | `g_deep_a/g_deep_b`、`g_fwd_a/g_fwd_b` 与 normal batch 完全相同，没有 last batch 专用 slot |
| `run_val_epoch_gpu()` last batch graph | ❌ **使用相同 graph** | `g_inf_a_exec/g_inf_b_exec` 与 normal batch 完全相同 |
| `run_train_epoch_cpu()` last batch graph | ❌ **使用相同 graph** | `idx_deep`（DEEP_FWD_BWD）单一 graph |
| `run_val_epoch_cpu()` last batch graph | ❌ **使用相同 graph** | `idx_inf_a/idx_inf_b` 单一 graph |

**结论：框架的 6 变体设计在 Compiler 层已就绪（`variants` 向量已存在），但 GraphAtlas 构建和运行时变体切换完全没有落地。**

---

## 二、GPU 路径详细分析

### 2.1 Metric Accumulation（小伙伴 S 和 D 认为正确的部分）

```cpp
// compiler.cpp
node.input_ids = { b.local_batch_size, b.loss, b.top1, b.top5 };        // ACCUM_METRICS
node.input_ids = { b.last_train_batch_size, b.loss, b.top1, b.top5 };   // ACCUM_METRICS_TRAIN_LAST
node.input_ids = { b.last_val_batch_size, b.loss, b.top1, b.top5 };     // ACCUM_METRICS_VAL_LAST
```

```cpp
// accum_op.cu
float bs = static_cast<float>(*bs_ptr);
*al_p  += (*loss_p) * bs;
*at1_p += (*top1_p) * bs;
```

✅ **Metric accumulation 本身的加权逻辑是正确的。** `ACCUM_METRICS_TRAIN_LAST` 确实使用了 `last_train_batch_size`，`accum_metrics_kernel` 按实际 batch size 加权累加，最后除以总样本数。

**但前提是 `loss_p` 和 `top1_p` 必须只反映实际样本的计算结果。**

### 2.2 核心 Bug：FWD/BWD/INF Graph 使用固定 Shape

CUDA Graph 在 capture 时，kernel 的 **grid 维度是固定的**。以 `softmax_ce_inf_kernel` 为例：

```cpp
// capture 时：grid = local_batch_size（如 128），不可变
softmax_ce_inf_kernel<<<batch, BLOCK_DIM, smem, s>>>(
    ..., batch, ..., batch_size_ptr, ...);

// kernel 内部：
int b = blockIdx.x;
if (b >= batch) return;        // batch = 128，永远不会触发
float inv_batch = 1.0f / (*batch_size_ptr);  // last batch 时 = 1/80
```

Last batch 只有 80 个有效样本，但 **128 个 block 全部执行**：
- 前 80 个 block：处理有效数据，正确
- 后 48 个 block：读取上一 batch 残留的 labels/logits，计算 softmax + CE，执行 `atomicAdd(loss, ...)` 和 `atomicAdd(top1, ...)`

**这导致 loss、top1、top5 被残留样本污染。**

同样的问题存在于：
- `softmax_ce_bwd_kernel`：残留样本产生错误 gradient，`dlogit_b[c] = (prob - 1) * scale`，污染整个 backward 的 gradient
- cuDNN Conv/FC/BN：descriptor 中的 N=batch 在 capture 时固定为 `local_batch_size`，cuDNN 会处理全部 128 个样本
  - Conv FWD：后面 48 个样本的 feature maps 被计算
  - BN FWD：running mean/variance 的统计包含了残留样本（如果 BN 内部按 N 做 reduction）
  - Conv BWD：dX 和 dW 包含了残留样本的 gradient

**Metric accumulation 的加权再正确也没用，因为输入的 loss/top1/top5 本身已经被污染。**

### 2.3 小伙伴 K 的判断：正确 ✅

小伙伴 K 指出的 "存在严重的 last batch 正确性 Bug" 是准确的。问题不在于 metric accumulation 的加权逻辑，而在于**产生 loss/top1/top5 的上游 graph 使用了错误的 shape**。

### 2.4 小伙伴 S 和 D 的判断：部分正确，但忽略了根本问题

小伙伴 S 和 D 只检查了 metric accumulation 和最终除法，没有向上追溯到 FWD/BWD kernel 的 grid 维度固定问题。

---

## 三、CPU 路径详细分析

### 3.1 指标累积方式（小伙伴 D 正确指出）

```cpp
// run_val_epoch_cpu() — line 1622-1624
float avg_loss = static_cast<float>(acc_loss / val_batches);  // ❌ 按 batch 数除
float avg_top1 = static_cast<float>(acc_top1 / val_batches);
float avg_top5 = static_cast<float>(acc_top5 / val_batches);
```

而 GPU 路径：
```cpp
avg_loss = accum_loss / static_cast<float>(registry.num_val_samples());  // ✅ 按样本数除
```

如果 last val batch 的样本数（如 106）与常规 batch（512）不同，CPU 路径会把 106 个样本的 loss 和 512 个样本的 loss 等权对待。

**影响：对于 ImageNet val（50000 样本，batch=512，last=106），错误约为 (512-106)/50000 ≈ 0.8%，不可忽略。**

### 3.2 CPU 训练路径的指标累积

```cpp
// run_train_epoch_cpu() — line 1397-1400
if (loss_id >= 0) {
    Tensor h_loss = fetch_from_rank(active_memory_plan_->get_dtensor(loss_id), 0);
    train_loss = h_loss.data<float>()[0];
}
```

❌ **完全没有指标累积**。多 batch 场景下，只返回了最后一个 batch 的 `loss_id` 值，不是整轮平均 loss。对比 GPU 路径使用的是 `accum_loss / num_train_samples()`。

### 3.3 CPU 的 Softmax CE 是否也有 shape 问题？

```cpp
// softmax_ce_op.cpp — CPU launch
int batch = op_ctx->input_shape.n;
softmax_ce_fwd_inner(logits, labels, loss, inv_sc, probs, batch, num_cls, scaling[0]);
```

CPU graph 的 `input_shape.n` 在 capture 时固定为 `local_batch_size`（因为 CPU capture 使用的是 `CompileSpec` 中的 shape）。如果 `input_shape.n` 固定，那么 `softmax_ce_fwd_inner` 会处理超出实际样本数的残留数据，与 GPU 路径有相同的污染问题。

**需要确认 CPU capture 代码中 `input_shape.n` 是否固定。** 但从设计上看，如果只有一个 graph variant，那它就是固定的。

---

## 四、多 RANK 情形（用户特别提醒）

用户给出的例子：
- ImageNet 训练集 1281167，验证集 50000
- world_size = 8，local_batch_size = 512
- Pad 后：训练 1281168 / 验证 50000
- 每 RANK：训练 160146 / 验证 6250
- steps：训练 313 / 验证 13
- last batch size：训练 402 / 验证 106

当前代码中的 `get_last_train_batch_size()`：
```cpp
int GlobalRegistry::get_last_train_batch_size() const {
    size_t per_rank = train_samples_per_rank();
    int bs = get_local_batch_size();
    int rem = static_cast<int>(per_rank % static_cast<size_t>(bs));
    return (rem == 0) ? bs : static_cast<int>(rem);
}
```

这里 `per_rank` 已经考虑了 pad 到 world_size 整数倍。计算逻辑是正确的。

**但当前实现没有利用这个正确的 `last_batch_size` 来构建独立的 graph。** 正确的 `last_batch_size` 只被用在了 `ACCUM_METRICS_TRAIN_LAST` 的加权上，而上游的 FWD/BWD graph 仍然是 512 的 shape。

---

## 五、修复建议

### 方案 1：短期修复 — 清零残留数据（最小侵入，立即可用）

在 XFER graph 完成后，对超出 `last_batch_size` 的 labels 和数据执行 `cudaMemsetAsync`，使残留样本对后续计算无贡献。

**修改位置**：`deep_learning_task.cpp` 的 `run_train_epoch_gpu()` 和 `run_val_epoch_gpu()`

```cpp
// 在 ts->set_buffer_readable(buf, false) 之后，launch FWD/INF 之前
if (batch == batches - 1) {
    int last_bs = registry.get_last_train_batch_size();
    int pad = local_bs - last_bs;
    if (pad > 0) {
        // 清零 labels（超出 last_bs 的部分）
        int32_t label_id = (buf == 0) ? b.label_a : b.label_b;
        void* label_ptr = ctx.ptr_at(label_id);
        // label 是 int32，每个样本 1 个元素
        cudaMemsetAsync((int32_t*)label_ptr + last_bs, 0,
                        pad * sizeof(int32_t), s_trans);

        // 清零 data（超出 last_bs 的部分）
        int32_t data_id = (buf == 0) ? b.data_a : b.data_b;
        void* data_ptr = ctx.ptr_at(data_id);
        // data 是 CHW 格式，每个样本 numel = C*H*W
        int data_numel = d_data_a.n_ * d_data_a.h_ * d_data_a.w_ * d_data_a.c_;
        cudaMemsetAsync((uint8_t*)data_ptr + last_bs * data_numel * elem_size, 0,
                        pad * data_numel * elem_size, s_trans);
    }
}
```

**优点**：
- 不需要改动 GraphAtlas 或 Compiler
- 不需要为 last batch 单独 capture graph
- 对 loss/accuracy/gradient 的影响为 0（清零后，softmax 给所有 class 相同概率，但 `atomicAdd` 仍会被执行）

**缺点**：
- 仍有多余的 block/thread 在执行（浪费计算）
- cuDNN Conv/BN 仍处理额外样本（虽然输入为 0，但 BN running stats 可能被轻微影响）
- 不是根本解决，是 workaround

**注意**：即使 labels 和数据清零，后面的 block 仍然会执行并贡献 `loss`（因为 `atomicAdd(loss, sample_loss * inv_batch)` 对每个 block 都执行）。要使贡献为 0，需要让 `sample_loss` 为 0，这需要 labels 指向一个概率为 1 的 class（不可能）。所以**清零 workaround 对 loss 不够完美**。

### 方案 2：根本修复 — 实现 GraphAtlas + 运行时变体切换

**步骤 1：实现 `GraphAtlas::build()`**

已有头文件声明，需要创建 `src/graph/graph_atlas.cpp`：

```cpp
GraphAtlas GraphAtlas::build(const Compiler::Result& compiled,
                              const std::array<ShapeId, 6>& variant_shapes) {
    GraphAtlas atlas;
    for (size_t v = 0; v < compiled.variants.size() && v < kMaxVariants; ++v) {
        for (size_t g = 0; g < kMaxGraphIds; ++g) {
            GraphId gid = static_cast<GraphId>(g);
            bool is_train = is_train_graph(gid);
            bool is_infer = is_inference_graph(gid);
            bool is_xfer  = is_transfer_graph(gid);
            bool is_train_var = (v <= 3);
            bool is_val_var   = (v >= 4);

            if (is_val_var && is_train) continue;
            if (is_train_var && is_infer) continue;

            auto& s = atlas.slot(v, gid);
            if (is_train || is_xfer) {
                s.cg = &compiled.train_cg;
            } else {
                s.cg = &compiled.infer_cg;
            }
            if (is_shape_invariant_graph(gid)) {
                s.mp = &compiled.variants[0].memory_plan;
                s.shape_id = kShapeInvariant;
            } else {
                s.mp = &compiled.variants[v].memory_plan;
                s.shape_id = variant_shapes[v];
            }
            s.stream_kind = stream_for(gid);  // 需要添加映射
        }
    }
    return atlas;
}
```

**步骤 2：修改 `deep_learning_task.cpp` 的 GraphSlot 和 `build_exec_table()`**

需要为 last batch 增加新的 GraphSlot：

```cpp
enum class GraphSlot : uint8_t {
    // ... existing slots ...
    FWD_BWD_DEEP_A_LAST,      // last batch variant
    FWD_BWD_DEEP_B_LAST,
    FIRST_LAYER_FWD_A_LAST,
    FIRST_LAYER_FWD_B_LAST,
    FIRST_LAYER_BWD_A_LAST,
    FIRST_LAYER_BWD_B_LAST,
    INF_MAIN_A_LAST,
    INF_MAIN_B_LAST,
    // ...
};
```

并在 `build_exec_table()` 中解析变体索引：

```cpp
// 需要扩展 atlas.index() 支持变体选择
// 或者为每个变体单独 build exec table
```

**步骤 3：运行时选择变体**

```cpp
// run_train_epoch_gpu() 中
for (int batch = 0; batch < batches - 1; ++batch) {
    // normal batch: use variant 0 (train_base)
    activate_variant(0);
    // ... normal graph launch ...
}
// last batch: use variant 1 (train_last)
activate_variant(1);
// ... last batch graph launch ...
```

但这与当前的 double buffering（A/B）机制冲突。需要重新设计。

**优点**：根本解决，符合框架设计意图
**缺点**：改动量大，涉及 Compiler、GraphAtlas、DeepLearningTask 三层，需要数天开发

### 方案 3：折中方案 — 修改 Kernel 使用 `batch_size_ptr` 控制边界

对自定义 kernel（softmax_ce_fwd/inf/bwd），把边界检查从固定的 `batch` 参数改为动态的 `batch_size_ptr`：

```cpp
// softmax_ce_fwd_kernel
int b = blockIdx.x;
if (b >= *batch_size_ptr) return;   // 用动态值替代固定值
float inv_batch = 1.0f / (*batch_size_ptr);
```

```cpp
// softmax_ce_bwd_kernel
int total = (*batch_size_ptr) * num_classes;
if (idx >= total) return;
```

**优点**：
- 不需要改动 GraphAtlas 或 Compiler
- 自定义 kernel 的污染问题完全解决

**缺点**：
- cuDNN 操作无法修改（Conv/BN 的 descriptor batch size 仍固定）
- 需要修改所有自定义 kernel，可能有遗漏

### 综合建议

| 优先级 | 方案 | 工作量 | 效果 |
|---|---|---|---|
| P0 | **方案 2（根本修复）** | 大（3-5 天） | 完全正确，符合设计意图 |
| P1 | **方案 3（kernel 修改）** | 中（1 天） | 解决自定义 kernel 问题，cuDNN 仍残留 |
| P2 | **CPU 路径修复** | 小（0.5 天） | 修正指标累积和加权方式 |

**推荐执行顺序**：
1. 立即修复 CPU 路径的指标累积 bug（小伙伴 D 已指出）
2. 同步修改 softmax CE kernel 的边界检查（方案 3）
3. 在下一个迭代中实现 GraphAtlas 变体切换（方案 2）

---

## 六、CPU 路径具体修复代码

### 6.1 `run_train_epoch_cpu()` — 增加指标累积

当前代码只返回最后一个 batch 的 loss：
```cpp
if (loss_id >= 0) {
    Tensor h_loss = fetch_from_rank(active_memory_plan_->get_dtensor(loss_id), 0);
    train_loss = h_loss.data<float>()[0];
}
```

应改为累加所有 batch 的 loss，然后除以总样本数：
```cpp
// 需要在循环内累加
float total_loss = 0.0f;
int total_samples = 0;
for (int batch = 0; batch < batches; ++batch) {
    // ... launch graph ...
    int bs = (batch == batches - 1) ? registry.get_last_train_batch_size()
                                     : registry.get_local_batch_size();
    if (loss_ptr) total_loss += (*static_cast<float*>(loss_ptr)) * bs;
    total_samples += bs;
}
train_loss = total_loss / total_samples;
```

### 6.2 `run_val_epoch_cpu()` — 按样本数加权

```cpp
// 当前：按 batch 数平均
float avg_loss = acc_loss / val_batches;

// 应改为：按样本数加权
int total_val_samples = registry.num_val_samples();
float avg_loss = acc_loss / total_val_samples;
float avg_top1 = acc_top1 / total_val_samples;
float avg_top5 = acc_top5 / total_val_samples;
```

但注意：`acc_loss` 当前存储的是每个 batch 的**每样本平均 loss** 之和，不是总 loss 之和。需要改为存储 `batch_loss * batch_size`：

```cpp
// 循环内：
float batch_loss = loss_ptr ? *static_cast<float*>(loss_ptr) : 0.0f;
int bs = (batch == val_batches - 1) ? registry.get_last_val_batch_size()
                                       : registry.get_local_batch_size();
acc_loss += batch_loss * bs;
acc_top1 += batch_top1 * bs;
acc_top5 += batch_top5 * bs;

// 循环后：
float avg_loss = acc_loss / registry.num_val_samples();
```

---

## 七、总结

| 问题 | 严重程度 | 负责小伙伴 | 状态 |
|---|---|---|---|
| GPU last batch 使用固定 shape 的 CUDA Graph | 🔴 **P0 — 正确性 Bug** | K 已指出 | 未修复 |
| CPU 训练无指标累积，只取 last batch loss | 🟡 **P1 — 正确性 Bug** | D 已指出 | 未修复 |
| CPU 验证按 batch 数平均而非样本数加权 | 🟡 **P1 — 正确性 Bug** | D 已指出 | 未修复 |
| GraphAtlas::build() 未实现 | 🟡 **P1 — 设计缺失** | — | 未实现 |
| Metric accumulation 加权逻辑 | 🟢 正确 | S、D 已确认 | 已正确 |

**三位小伙伴的分析都有价值，但都只覆盖了部分视角：**
- **S**：确认了 metric accumulation 的正确性，但忽略了上游 FWD/BWD graph 的 shape 问题
- **K**：正确指出了 GPU kernel grid 固定的根本问题，但对 CPU 路径和 accum 加权的分析不够深入
- **D**：正确指出了 CPU 路径的三处问题，但没有涉及 GPU 路径

**当前最紧迫的修复：**
1. 修改 `softmax_ce_fwd/inf/bwd_kernel` 使用 `*batch_size_ptr` 控制边界（方案 3）
2. 修复 CPU 路径的指标累积和加权（小伙伴 D 的代码建议）
3. 在下个迭代中实现 `GraphAtlas::build()` 和运行时变体切换（方案 2）
