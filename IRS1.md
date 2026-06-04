# 卷积算子实现审查报告 (IRS1)

## 审查范围

对照 [FIX.md](file:///r:/renaissance/FIX.md) 和 [RESULTS.md](file:///r:/renaissance/RESULTS.md)，逐一检查以下文件中与卷积算子相关的全部代码：

- [src/backend/ops/dtensor/conv_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/conv_op.cpp) — launch 函数 & CPU 实现
- [src/backend/ops/dtensor/conv_op_impl.cpp](file:///r:/renaissance/src/backend/ops/dtensor/conv_op_impl.cpp) — cuDNN graph 构建
- [src/graph/layer_descriptor_registry.cpp](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp) — 张量描述 & build 函数
- [src/graph/compiler.cpp](file:///r:/renaissance/src/graph/compiler.cpp) — 图层编译（训练图 & 推理图）

---

## 一、小伙伴 K 的问题核查

### Issue K1: CPU BWD 中 `op_ctx->input_shape` 误用为 X 的 shape ✅ 已修复

小伙伴指出：BWD 中 `op_ctx->input_shape` 返回的是 dY 的 shape `[N,OH,OW,K]`，但被代码当作 X 的 shape 使用。

**核查结果：已经修复。** 当前代码在 [conv_op.cpp:649-656](file:///r:/renaissance/src/backend/ops/dtensor/conv_op.cpp#L649-L656)：

```cpp
int N = op_ctx->input_shape.n;   // dY 的 batch（与 X 的 batch 相同）
int OH = op_ctx->input_shape.h;  // dY 的 spatial H
int OW = op_ctx->input_shape.w;  // dY 的 spatial W

// ★ H / IW / C 必须从 output_shape（dX）获取，因为 input_shape 是 dY 的 shape
int H  = op_ctx->output_shape.h;
int IW = op_ctx->output_shape.w;
int C  = op_ctx->output_shape.c;
```

H/IW/C 已正确从 `output_shape`（dX）获取，避免了 shape 错误。该修复参照了 FC BWD 的做法（FC BWD 也是从 `output_shape` 获取 `in_features`）。RESULTS.md 中 CPU BWD MSE 正常（dX=9.3e-14, dW=2.7e-10）也佐证了此修复有效。

**结论：无问题。**

---

### Issue K2: naive FWD 中存在死代码 ✅部分采纳

小伙伴指出 `const float* B = nullptr;` 和 `if (false) sum += 0.0f;` 是历史遗留的死代码。

**核查结果：确实存在。** 当前代码 [conv_op.cpp:563](file:///r:/renaissance/src/backend/ops/dtensor/conv_op.cpp#L563) 和 [conv_op.cpp:608](file:///r:/renaissance/src/backend/ops/dtensor/conv_op.cpp#L608)：

```cpp
const float* B = nullptr;  // no bias in conv
// ...
if (false) sum += 0.0f;  // 抑制 unused 警告
```

这是代码质量问题（不影响功能），但 `if (false)` 是一种不优雅的警告抑制方式。框架明确不支持卷积 bias，这些代码属于历史遗留。

**结论：真实但不严重。建议清理。**

**修改建议**：删除 `const float* B = nullptr;` 和 `if (false) sum += 0.0f;`。如果 `sum` 变量的 `unused` 警告仍有意义，直接用 `(void)sum;` 或 `[[maybe_unused]]` 抑制。

---

## 二、小伙伴 C 的问题核查

### Issue C1: FWD 与 INF 使用独立 cache，未共享 [细枝末节，可忽略]

小伙伴指出 FWD 和 INF 在 cache key 中使用了不同的 `ComputeOp`，导致两者无法共享 graph。

**核查结果：确实如此。** [conv_op_impl.cpp](file:///r:/renaissance/src/backend/ops/dtensor/conv_op_impl.cpp) 中 `ConvGraphCacheKey` 包含 `ComputeOp op` 字段，FWD 和 INF 的 key 不同，graph 不共享：

```cpp
ConvGraphCacheKey key_fwd{..., false, ComputeOp::CONV_FP32_FWD};
ConvGraphCacheKey key_inf{..., false, ComputeOp::CONV_FP32_INF};
```

**分析**：FWD 和 INF 的 cuDNN graph 结构完全相同（都是 conv_fprop），仅 `ComputeOp` 枚举值不同。不共享的后果是每个 shape 要多构建一个 graph，仅影响 graph 构建阶段的性能（一次性开销），**不影响运行时的数值正确性**。属于细枝末节。

**结论：真实但无实际影响。可改可不改。**

---

### Issue C2: 编译器中对首层 Conv 的 X 注入缺失 🔴 确认！严重 bug

小伙伴指出：当 Conv 作为网络第一层时，编译器没有将输入数据张量 X 注入到 `input_ids` 中，导致 FWD 和 BWD 都会缺少 X 输入。

**核查结果：确认存在！这是一个严重的设计缺陷。** 分析如下：

#### 根因分析

MaxPool（参考基准）通过空 `input_indices` 触发编译器的首层注入逻辑：

```cpp
// layer_descriptor_registry.cpp — build_maxpool_forward
SubgraphPattern build_maxpool_forward(...) {
    n.input_indices = {};           // ← 空！
    n.output_indices = {0};
    // ...
}
```

编译器利用空 `input_indices` 触发注入（[compiler.cpp:1169](file:///r:/renaissance/src/graph/compiler.cpp#L1169)）：
```cpp
if (layer.is_first_layer && gn.input_ids.empty()) {
    // 注入 I_A_DATA / I_B_DATA ...
}
```

但 Conv 的 `build_conv_forward` 有显式的 `input_indices`（[layer_descriptor_registry.cpp:130](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L130)）：
```cpp
// FP32: input_indices = {0, 6}  → {W_id, bn_stats_id}
// AMP:  input_indices = {4, 6}  → {amp_w_id, bn_stats_id}
```

所以 `gn.input_ids` **非空** → `gn.input_ids.empty()` 检查不触发 → 又因为 `prev_output_id = -1`，`prev_output_id >= 0` 检查也不触发 → **X 未被注入**。

#### 影响范围

此 bug 在以下**所有**路径中都存在：

| 路径 | 文件位置 | 影响 |
|---|---|---|
| 训练图 FWD（常规路径） | [compiler.cpp:1185](file:///r:/renaissance/src/graph/compiler.cpp#L1185) | 首层 Conv FWD 缺 X |
| 训练图 FWD（per-shape cache） | [compiler.cpp:1209-1226](file:///r:/renaissance/src/graph/compiler.cpp#L1209-L1226) | 首层 Conv FWD 缺 X |
| 训练图 BWD | [compiler.cpp:1301-1306](file:///r:/renaissance/src/graph/compiler.cpp#L1301-L1306) | 首层 Conv BWD 缺 X 和 dX(in-place) |
| 推理图 FWD | [compiler.cpp:1514-1531](file:///r:/renaissance/src/graph/compiler.cpp#L1514-L1531) | 首层 Conv INF 缺 X  |

**FP32 和 AMP 两种精度均受影响。**

#### 为什么 standalone 测试没发现？

`test_conv_fwd_bwd.cpp` 使用 `SimpleTask`，直接指定了完整的 `input_ids = {X, W, bn_stats}` 和 `output_ids`，不经过 `compiler.cpp` 的首层注入逻辑。`DeepLearningTask` 才会触发此 bug。

#### 运行时后果

以训练图 FWD 为例，首层 Conv 的 `gn.input_ids = {W_id, bn_stats_id}`（无 X_id），`launch_conv_fp32_fwd_cuda` 执行时：
- `node.input_ids[0]` 取到的是 W 的指针（当作 X 用）→ **数据完全错误**
- `node.input_ids[1]` 取到的是 bn_stats 的指针（当作 W 用）→ **shape 不匹配导致 crash 或静默错误**
- `node.input_ids[2]` 越界 → **UB / crash**

BWD 同理：`gn.input_ids = {dY_id, W_id}`（无 X_id 和 dX_inplace），也会 crash。

#### 修改建议

在 `compiler.cpp` 中新增一个首层注入分支，专门处理 `input_indices` 非空但 X 仍缺失的情况。关键位置有三处：

**1. 训练图 FWD（常规路径）** — 在现有 `gn.input_ids.empty()` 检查后、`prev_output_id >= 0` 之前插入：

```cpp
// 首层 Conv/FC 等有显式 input_indices 的算子：仍需注入 I_A_DATA / I_B_DATA
if (layer.is_first_layer) {
    const auto& b = memory_plan.baseline();
    layer_input_ids[l] = b.data_a;

    // ★ 显式 input_indices 的首层算子直接注入 I_A_DATA 作为首元素
    gn.input_ids.insert(gn.input_ids.begin(), b.data_a);

    // 双缓冲：B 桶使用 I_B_DATA
    GraphNode gn_b = gn;
    gn_b.input_ids[0] = b.data_b;
    train_cg.append(graph_id_b, gn_b);

    // 继续走 prev_output_id 注入（prev_output_id == -1 所以跳过）
}
// else if (prev_output_id >= 0) { ... }  // 非首层的跨层注入保持不变
```

**2. 训练图 FWD（per-shape cache）** — 在 `gn_a`/`gn_b` 创建后，`train_cg.append` 之前，做同样的首层注入。

**3. 推理图** — 在 `compiler.cpp:1514` 现有空 input_ids 检查后，新增非空 input_ids 的首层注入。

**4. BWD 路径** — 需要确保 `layer_input_ids[l]` 在首层 FWD 阶段被正确设置为 `b.data_a`（上述 FWD 修改已包含 `layer_input_ids[l] = b.data_a`），这样 BWD 的 `it->second >= 0` 检查才能通过，X 才会被正确追加。

---

## 三、自主发现的额外问题

### 额外问题 1: per-shape cache 路径的首层注入

训练图 per-shape cache 路径（[compiler.cpp:1209-1226](file:///r:/renaissance/src/graph/compiler.cpp#L1209-L1226)）也有完全相同的首层注入缺失问题。

```cpp
if (use_per_shape_cache && ...) {
    gn_compute_cg_a = std::move(gn_a);
    gn_compute_cg_b = std::move(gn_b);

    gn_a = res_a.first;  // input_ids = {W_id, bn_stats_id}，无 X
    gn_b = res_b.first;  // 同上

    train_cg.append(graph_id, gn_a);   // X 缺失
    train_cg.append(graph_id_b, gn_b); // X 缺失
}
```

**修改建议**：与上述"训练图 FWD 常规路径"的修改一并处理。可以在 per-shape cache 路径中做同样的注入，或重构为统一的注入逻辑。

### 额外问题 2: XNNPACK 和 naive FWD 中 `input_shape` 的潜在问题

当前 `launch_conv_fwd_cpu`（XNNPACK）和 `launch_conv_fwd_cpu_naive` 都通过 `op_ctx->input_shape` 获取 N/H/W/C：

```cpp
int N  = op_ctx->input_shape.n;
int H  = op_ctx->input_shape.h;
int IW = op_ctx->input_shape.w;
int C  = op_ctx->input_shape.c;
```

这依赖于 X 是第一个输入（FWD 中如此）。如果编译器首层注入修复后，X 被正确注入为 `input_ids[0]`，则此代码正确。当前在 standalone 测试中正确（因为测试直接指定了正确的 input_ids 顺序），但 DeepLearningTask 中若首层注入缺失，`input_shape` 会反映 W 或 bn_stats 的 shape（完全错误）。

此问题的修复依赖于"额外问题 1"的修复（编译器首层注入）。

---

## 总结

| # | 来源 | 问题 | 严重程度 | 结论 |
|---|---|---|---|---|
| K1 | FIX.md | CPU BWD input_shape 误用 | — | ✅ 已修复 |
| K2 | FIX.md | naive FWD 死代码 | 低 | 建议清理 |
| C1 | FIX.md | FWD/INF cache 不共享 | 低 | 细枝末节，可忽略 |
| C2 | FIX.md | **首层 Conv X 注入缺失** | **高 / 关键** | 🔴 必须修复 |
| E1 | 自主 | per-shape cache 路径首层注入缺失 | 高 | 与 C2 一并修复 |
| E2 | 自主 | CPU FWD input_shape 依赖首层注入正确性 | 中 | 依赖 C2/E1 修复 |

### 额外问题 3: 测试样例不应使用 `run_iter`

[test_conv_fwd_bwd.cpp:343-355](file:///r:/renaissance/tests/op/test_conv_fwd_bwd.cpp#L343-L355) 使用 `task.run_iter()` 来跑 FWD 和 BWD：

```cpp
task.run_iter("fwd", cfg.warmup);      // warmup
task.run_iter("fwd", cfg.iterations);  // 多次迭代
// ...
task.run_iter("bwd", cfg.warmup);
task.run_iter("bwd", cfg.iterations);
```

这是数学正确性测试而非性能 benchmark，只需要跑一次验证结果即可。多次迭代会浪费时间且引入不必要的 `run_iter` 带来的开销（如每个 stage 的同步）。应该改为 `task.run()`。

**修改建议**：将 `run_iter` 替换为 `task.run()`。FC 的参考测试 [test_fc_fwd_bwd.cpp](file:///r:/renaissance/tests/op/test_fc_fwd_bwd.cpp) 使用的是 `task.run()`，卷积测试应该对齐。

---

## 总结

| # | 来源 | 问题 | 严重程度 | 结论 |
|---|---|---|---|---|
| K1 | FIX.md | CPU BWD input_shape 误用 | — | ✅ 已修复 |
| K2 | FIX.md | naive FWD 死代码 | 低 | 建议清理 |
| C1 | FIX.md | FWD/INF cache 不共享 | 低 | 细枝末节，可忽略 |
| C2 | FIX.md | **首层 Conv X 注入缺失** | **高 / 关键** | 🔴 必须修复 |
| E1 | 自主 | per-shape cache 路径首层注入缺失 | 高 | 与 C2 一并修复 |
| E2 | 自主 | CPU FWD input_shape 依赖首层注入正确性 | 中 | 依赖 C2/E1 修复 |
| E3 | 自主 | 测试样例用 `run_iter` 而非 `run` | 低 | 数学测试只需跑一次 |

**核心结论**：问题 C2（及衍生的 E1/E2）是一个真实存在的严重 bug。当 Conv 作为 DeepLearningTask 的第一层时，训练图和推理图都会因为缺少 X 输入数据而起 crash。86% 的 FIX.md 意见是正确的（已修复或采纳），14% 属于细枝末节。E3 是对测试代码的规范性问题。