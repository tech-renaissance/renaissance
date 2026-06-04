# IRS3 — 卷积算子最终检查意见

> 检查范围：FIX.md / RESULTS.md 中已报告的问题 + 独立深度代码审查
> 检查文件：
> - `src/backend/ops/dtensor/conv_op.cpp`
> - `src/backend/ops/dtensor/conv_op_impl.cpp`
> - `src/graph/compiler.cpp`
> - `src/graph/layer_descriptor_registry.cpp`
> - `src/backend/op_registry.cpp`
> - `src/backend/op_stream_policy.cpp`
> - `src/graph/op_kind.cpp`
> - `src/graph/capture_cpu.cpp`
>
> 日期：2026-06-02

---

## 一、已确认修复的问题（无需再动）

| 问题 | 状态 | 说明 |
|------|------|------|
| `bn_stats_ptr` 冗余变量 | ✅ 已修 | `launch_conv_amp_fwd_cuda` 中已删除未使用的 `bn_stats_ptr` |
| CPU BWD `input_shape` 误用为 X 形状 | ✅ 已修 | 代码 line 653–656 已改用 `output_shape.h/w/c` 获取 X 维度；`N` 从 `input_shape.n` 取（dY 与 X batch 相同，正确）；`OH/OW` 从 `input_shape.h/w` 取（dY 的 spatial，正确） |
| `make_nhwc_stride`/`make_krsc_stride` 编译错误 | ✅ 已修 | namespace 隔离 + 删除重复定义 |
| CPU FWD XNNPACK 将 bn_stats 误作 bias | ✅ 已修 | XNNPACK 路径已显式跳过 bias（`bias_id = XNN_INVALID_VALUE_ID`） |
| 文件末尾缺少换行符 | ✅ 已修 | — |

---

## 二、已确认正确的设计点（无需修改）

| 检查项 | 结论 |
|--------|------|
| `compiler.cpp` `get_grad_output_id(Conv) = -1` + `grad_id < 0` 回退到 `layer_input_ids` | ✅ 正确 |
| BWD 双图双流：`wgrad@COMP_1 → cudaEventRecord → dgrad@COMP_3 (cudaStreamWaitEvent)` | ✅ 正确 |
| AMP graph builder 使用 `padded_c()` 设置 operand dim，stride 用真实值 | ✅ 正确 |
| `conv_out->set_output(true)` 后直接作为 genstats 输入 | ✅ 正确（cuDNN FE 允许 output tensor 同时作为后续节点输入） |
| `sum`/`sq_sum` 指针偏移 `+K` 个 float | ✅ 正确 |
| 6 个算子全部注册，CPU fallback 完整 | ✅ 正确 |
| `require_warmup()` 包含 INF，warmup 走 `launch_cuda + cudaDeviceSynchronize` | ✅ 正确 |
| `state.output_stream_idx = i_dx` | ✅ 正确 |
| `infer_conv_tensors_with_bn_stats` 返回 7 张量，`conv_desc` 注册正确 | ✅ 正确 |
| `build_conv_*` 的 `descs.size() < 7` 检查 | ✅ 正确 |
| `op_stream_policy.cpp`: INF → COMP_1 | ✅ 正确 |
| `op_kind.cpp`: 6 个 Conv 算子全部在 `compute_op_to_string` 和 `format_params` 中 | ✅ 正确 |
| `workspace` 管理：全部使用 `ctx.ensure_workspace_grow` | ✅ 正确 |

---

## 三、仍存在的问题（按严重程度排序）

### 问题 1：首层 Conv 的输入 X 缺失 — 训练图 FWD / 推理图 FWD / 训练图 BWD 均受影响（严重）

**根因**：`compiler.cpp` Phase 4 中，首层的 `prev_output_id = -1`，因此 `if (prev_output_id >= 0)` 不注入 X。而 `build_conv_forward` 的 `input_indices` 只包含 weight 和 bn_stats（`{0,6}` 或 `{4,6}`），导致 `gn.input_ids` 不为空，不触发 `[FIX] 非Flatten首层算子：无显式 input_indices → 显式注入 I_A_DATA / I_B_DATA` 分支。

**后果**：

| 场景 | `node.input_ids` 实际内容 | launch 函数期望 | 结果 |
|------|--------------------------|----------------|------|
| 首层 FP32 FWD | `{W, bn_stats}` (2 个) | `[0]=X, [1]=W` | `node.input_ids[0]` 取到 W（被当作 X），`[1]` 取到 bn_stats（被当作 W）。形状完全不匹配，cuDNN FE 报 `BAD_PARAM` 或崩溃。 |
| 首层 AMP FWD | `{W, bn_stats}` (2 个) | `[0]=X, [1]=W, [2]=bn_stats` | `node.input_ids[0]` 取到 W（被当作 X），`[1]` 取到 bn_stats（被当作 W），`[2]` 越界。 |
| 首层 FP32/AMP BWD | `{dY, W}` (2 个) | `[0]=dY, [1]=W, [2]=X` | `node.input_ids[2]` 越界访问。 |
| 首层推理图 FP32/AMP | `{W, bn_stats}` (2 个) | 同 FWD | 同 FWD。 |

**非首层不受影响**：`prev_output_id >= 0`，X 正确注入到 `gn.input_ids` 开头；BWD 中 `layer_input_ids[l] >= 0`，X 正确追加。

**修复意见（`compiler.cpp`）**：

1. **训练图 FWD**（约 line 1184–1187）：
   ```cpp
   // 原代码
   if (prev_output_id >= 0) {
       gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
   }
   ```
   修改为：
   ```cpp
   if (prev_output_id >= 0) {
       gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
   } else if (layer.is_first_layer && !gn.input_ids.empty() &&
              (layer.kind == LayerKind::Conv || layer.kind == LayerKind::FC)) {
       // 首层 Conv/FC 有显式参数输入（weight/bias/bn_stats），需要注入数据输入
       const auto& b = memory_plan.baseline();
       layer_input_ids[l] = b.data_a;
       gn.input_ids.insert(gn.input_ids.begin(), b.data_a);
   }
   ```

2. **推理图 FWD**（约 line 1529–1531）：
   ```cpp
   // 原代码
   if (prev_inf_output_id >= 0) {
       gn.input_ids.insert(gn.input_ids.begin(), prev_inf_output_id);
   }
   ```
   修改为：
   ```cpp
   if (prev_inf_output_id >= 0) {
       gn.input_ids.insert(gn.input_ids.begin(), prev_inf_output_id);
   } else if (layer.is_first_layer && !gn.input_ids.empty() &&
              (layer.kind == LayerKind::Conv || layer.kind == LayerKind::FC)) {
       const auto& b = memory_plan.baseline();
       gn.input_ids.insert(gn.input_ids.begin(), b.data_a);
   }
   ```

   > 说明：训练图的修改同时修复了 BWD（因为 `layer_input_ids[l]` 被正确设置为 `b.data_a`，BWD 中的 `if (it->second >= 0)` 条件即可满足，从而正确追加 X）。推理图只需修复 FWD。

   > 补充：MaxPool 的首层注入之所以正确，是因为它的 `build_maxpool_forward` 中 `input_indices` 为空，所以 `gn.input_ids.empty()` 为 true，自然走 `[FIX]` 分支注入 `b.data_a`。Conv/FC 的问题在于 `input_indices` 包含了 weight/bn_stats 等参数，导致 `gn.input_ids` 不为空。

---

### 问题 2：FWD 与 INF cache 仍未共享（小伙伴C指出，未修复）

**位置**：`src/backend/ops/dtensor/conv_op.cpp`

`ConvGraphCacheKey` 包含 `ComputeOp op` 字段，导致 FWD 和 INF 使用不同的 key：

| 函数 | key.op |
|------|--------|
| `launch_conv_fp32_fwd_cuda` | `CONV_FP32_FWD` |
| `launch_conv_fp32_inf_cuda` | `CONV_FP32_INF` |
| `launch_conv_amp_fwd_cuda` | `CONV_AMP_FWD` |
| `launch_conv_amp_inf_cuda` | `CONV_AMP_INF` |

FWD 与 INF 在 cuDNN FE 层面 graph 结构完全相同（都是纯 `conv_fprop`），共享同一个 `s_conv_fwd_cache` map，但 key 不同导致 cache 不命中。每次从 FWD 切换到 INF（如 warmup 或实际推理）都会多构建一次 graph。

**修复意见**：将 INF launch 函数中的 `key.op` 改为对应的 FWD 值：

- `launch_conv_fp32_inf_cuda`: `ComputeOp::CONV_FP32_INF` → `ComputeOp::CONV_FP32_FWD`
- `launch_conv_amp_inf_cuda`: `ComputeOp::CONV_AMP_INF` → `ComputeOp::CONV_AMP_FWD`

或者从 `ConvGraphCacheKey` 中移除 `op` 字段（因为 FWD/INF 的 graph 构建函数已经相同，key 的其余字段已足以区分 shape/handle）。

---

### 问题 3：CPU FWD naive 残留死代码（小伙伴K指出，未修复）

**位置**：`src/backend/ops/dtensor/conv_op.cpp` line 563、608

```cpp
const float* B = nullptr;   // 从未使用
// ...
if (false) sum += 0.0f;     // 永远不会执行
```

**修复意见**：直接删除 `B` 变量声明和 `if (false)` 整行。注释 `// 框架不支持卷积 bias，第 3 个输入是 bn_stats，不使用` 可以保留在 `num_inputs >= 2` 的检查附近或删除。

---

## 四、低风险观察项（非 bug，仅建议）

### 观察 1：`ConvGraphCacheKeyHasher` 的 hash 质量

当前 hasher 使用简单的 `h ^= hash(field) << n` 组合，字段间 XOR 叠加，在 `unordered_map` 负载较高时碰撞率可能偏高。功能性上无问题（`unordered_map` 会处理碰撞），但建议未来改用更 robust 的组合方式（如 `boost::hash_combine` 风格的 `h = h * 31 + hash(field)`）。

### 观察 2：`get_or_build_cache` 无线程安全保护

`std::unordered_map` 的读写无 mutex。当前框架在 warmup 阶段单线程构建 graph，运行时只读 cache，因此无数据竞争。但如果未来支持多线程并行构图，需要加锁。当前无需修改。

### 观察 3：AMP cache key 使用逻辑 C 而非 padded C

`ConvGraphCacheKey` 中 `C = dt_x.c()` 是逻辑 C，而 AMP graph builder 使用 `dt_x.padded_c()` 设置 dim。在固定 alignment 策略下（`cuda_alignment=4/8`），相同逻辑 C 的 padded C 是确定的，因此不会导致错误 cache 共享。但如果未来 alignment 策略变化，建议 cache key 也包含 padded C。

---

## 五、总结

| 优先级 | 问题 | 状态 | 修复位置 |
|--------|------|------|----------|
| **P0（严重）** | 首层 Conv X 注入缺失，导致 FWD/BWD 越界或错误张量绑定 | ❌ 未修复 | `compiler.cpp` 训练图 FWD + 推理图 FWD |
| **P1（性能）** | FWD/INF cache 不共享 | ❌ 未修复 | `conv_op.cpp` INF launch 函数的 `key.op` |
| **P2（代码整洁）** | CPU FWD naive 死代码（`B` 变量 + `if(false)`） | ❌ 未修复 | `conv_op.cpp` line 563、608 |

**其余所有设计点和实现细节经再三确认均正确。**
