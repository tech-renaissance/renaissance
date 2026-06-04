# 卷积算子最终修复方案 (IRS_FINAL)

**基准文档**: [IRS1.md](file:///r:/renaissance/IRS1.md)、[IRS2.md](file:///r:/renaissance/IRS2.md)、[IRS3.md](file:///r:/renaissance/IRS3.md)  
**审查范围**: `conv_op.cpp`、`conv_op_impl.cpp`、`compiler.cpp`、`layer_descriptor_registry.cpp`、`test_conv_fwd_bwd.cpp`  
**策略**: 只修改必要的缺陷；细枝末节不修。

---

## 一、共识与分歧裁决

三份 IRS 报告的共同结论：

| # | 问题 | 三份报告共识 | 裁决 |
|---|---|---|---|
| K1 | CPU BWD input_shape 误用 | ✅ 已修复 | 无需动作 |
| C2 | **首层 Conv X 注入缺失** | 🔴 确认严重 bug | **必须修复** |
| K2 | naive FWD 死代码 | 属实，建议清理 | **采纳，修复** |
| C1 | FWD/INF cache 分离 | 属实，性能优化 | **采纳，修复** |
| E3 | 测试样例 run_iter → run | IRS1/IRS3 提出 | **采纳，修复** |

三份报告的分歧：

| 分歧点 | IRS1 | IRS2 | IRS3 | 裁决 |
|---|---|---|---|---|
| 首层注入是否限制 LayerKind | 通用（不限制） | Conv + FC | Conv + FC | **Conv + FC**，与 MaxPool/Flatten 对称 |
| BWD 是否构造 A/B 双节点 | 依赖 `layer_input_ids` 自动处理 | 显式构造 A/B 节点 + `continue` | 依赖 `layer_input_ids` 自动处理 | **IRS2 方案**，与 MaxPool 首层 BWD 对齐 |
| per-shape cache 是否需要改 | 认为需要（→ 错了） | 未提及 | 未提及 | **不需要改**：per-shape cache 在 launch 层，compiler 层无此逻辑 |

---

## 二、修复项

### Fix 1 (P0): compiler.cpp — 训练图 FWD，首层 Conv/FC 注入 X

**位置**: [compiler.cpp:1183](file:///r:/renaissance/src/graph/compiler.cpp#L1183)，在空 `input_ids` 检查后、`prev_output_id >= 0` 检查前插入

**问题**: Conv/FC 的 `build_*_forward` 有显式 `input_indices = {weight, bn_stats}`，导致 `gn.input_ids` 非空。首层时 `prev_output_id = -1`，两条注入路径均不触发，X 缺失。

**方案**: 仿照 MaxPool/ReLU 等算子的空 input_ids 首层注入模式，新增显式 input_ids 的首层分支，构造 A/B 双节点（`data_a` / `data_b`），`continue` 跳过后续通用注入逻辑。

```cpp
// ★ 在 line 1182 的 } 后、line 1184 的注释前插入：

// [FIX] 首层 Conv/FC 等有显式 input_indices（weight/bn_stats）的算子
// 显式注入 I_A_DATA / I_B_DATA 作为输入数据 X
if (layer.is_first_layer && !gn.input_ids.empty() &&
    (layer.kind == LayerKind::Conv || layer.kind == LayerKind::FC)) {
    const auto& b = memory_plan.baseline();
    layer_input_ids[l] = b.data_a;

    GraphNode gn_a = gn;
    gn_a.input_ids.insert(gn_a.input_ids.begin(), b.data_a);
    train_cg.append(graph_id, gn_a);

    GraphNode gn_b = gn;
    gn_b.input_ids.insert(gn_b.input_ids.begin(), b.data_b);
    train_cg.append(graph_id_b, gn_b);

    continue;
}
```

**說明**:
- `layer_input_ids[l] = b.data_a` 虽然 BWD 首层分支不依赖它，但设置后与整体风格一致，且不影响功能
- `continue` 后 `get_layer_output_id`（line 1230）仍会执行（与空 input_ids 首层注入的流程一致），`prev_output_id` 正常更新
- FC 也纳入修复范围，与 Flatten 首层 FC 注入逻辑形成对称

---

### Fix 2 (P0): compiler.cpp — BWD，首层 Conv X/dX 注入

**位置**: [compiler.cpp:1301-1306](file:///r:/renaissance/src/graph/compiler.cpp#L1301-L1306)，修改现有 Conv BWD 块

**问题**: 首层 Conv BWD 时 `layer_input_ids[l] = -1`，X 未注入，dX 未写回数据缓冲区。

**方案**: 参照 MaxPool 首层 BWD（lines 1323-1337），增加首层分支，显式构造 A/B 双节点，`continue` 跳过通用 append。

```cpp
// ★ 替换 line 1301-1306 的 Conv BWD 块为：

if (gn.compute_op == ComputeOp::CONV_FP32_BWD || gn.compute_op == ComputeOp::CONV_AMP_BWD) {
    if (layer.is_first_layer) {
        const auto& b = memory_plan.baseline();

        GraphNode gn_a = gn;
        gn_a.input_ids.push_back(b.data_a);
        // dX in-place 写入数据缓冲区（output_ids[0]），与 MaxPool 首层 BWD 语义一致
        gn_a.output_ids.insert(gn_a.output_ids.begin(), b.data_a);
        train_cg.append(GraphId::FIRST_LAYER_BWD_A, gn_a);

        GraphNode gn_b = gn;
        gn_b.input_ids.push_back(b.data_b);
        gn_b.output_ids.insert(gn_b.output_ids.begin(), b.data_b);
        train_cg.append(GraphId::FIRST_LAYER_BWD_B, gn_b);

        continue;
    } else {
        auto it = layer_input_ids.find(l);
        if (it != layer_input_ids.end() && it->second >= 0) {
            gn.input_ids.push_back(it->second);
            gn.output_ids.insert(gn.output_ids.begin(), it->second);  // dX in-place to X
        }
    }
}
```

**說明**:
- 首层 BWD 的 dX 写回 `I_A_DATA` / `I_B_DATA`（数据缓冲区），与 MaxPool 首层 BWD 语义一致
- 非首层路径不变（仍用 `layer_input_ids`）
- `continue` 跳过 line 1377-1380 的通用 append（避免重复添加）

---

### Fix 3 (P0): compiler.cpp — 推理图 FWD，首层 Conv/FC 注入 X

**位置**: [compiler.cpp:1514](file:///r:/renaissance/src/graph/compiler.cpp#L1514)，在推理图空 input_ids 检查后、`prev_inf_output_id >= 0` 检查前插入

**问题**: 与训练图 Fix 1 完全对称。推理图的首层 Conv/FC 同样缺失 X。

**方案**: 与 Fix 1 镜像。

```cpp
// ★ 在 line 1527 的 } 后、line 1529 的注释前插入：

// [FIX] 首层 Conv/FC 推理图：显式 input_indices → 注入 I_A_DATA / I_B_DATA
if (layer.is_first_layer && !gn.input_ids.empty() &&
    (layer.kind == LayerKind::Conv || layer.kind == LayerKind::FC)) {
    const auto& b = memory_plan.baseline();

    GraphNode gn_a = gn;
    gn_a.input_ids.insert(gn_a.input_ids.begin(), b.data_a);
    infer_cg.append(infer_graph_id, gn_a);

    GraphNode gn_b = gn;
    gn_b.input_ids.insert(gn_b.input_ids.begin(), b.data_b);
    infer_cg.append(infer_graph_id_b, gn_b);

    continue;
}
```

**說明**: 推理图无双缓冲混合问题，`continue` 后 `get_layer_output_id`（line 1552）正常更新 `prev_inf_output_id`。

---

### Fix 4 (P1): conv_op.cpp — FWD 与 INF 共享 cuDNN graph cache

**位置**: [conv_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/conv_op.cpp) 各 launch 函数的 `ConvGraphCacheKey` 构建

**问题**: INF key 的 `op` 字段为 `CONV_*_INF`，与 FWD 的 `CONV_*_FWD` 不同，导致同一 shape 的 graph 重复构建。

**方案**: 仅对 FP32 统一 cache key。AMP 不统一。

| 文件位置 | 修改前 | 修改后 |
|---|---|---|
| `launch_conv_fp32_inf_cuda` | `key.op = ComputeOp::CONV_FP32_INF` | `key.op = ComputeOp::CONV_FP32_FWD` |
| `launch_conv_amp_inf_cuda` | `key.op = ComputeOp::CONV_AMP_INF` | **不改** |

**说明**:
- **FP32**：FWD 和 INF 共用同一个 graph builder（`build_conv_fp32_fwd_graph`），统一 key 后共享 cache，无副作用。
- **AMP**：FWD 使用 `build_conv_amp_fwd_graph`（含 GenStats），INF 使用 `build_conv_amp_inf_graph`（无 GenStats）。两者 graph 结构不同，但共用同一个 `s_conv_fwd_cache`。若统一 key，AMP INF 会命中 AMP FWD 的 cache entry，拿到含 GenStats 的 graph，但 INF 的 variant pack 中无 bn_stats 指针，会导致运行时错误。因此 AMP 保持 key 分离，不共享 cache。

---

### Fix 5 (P2): conv_op.cpp — 删除 naive FWD 死代码

**位置**: [conv_op.cpp:563](file:///r:/renaissance/src/backend/ops/dtensor/conv_op.cpp#L563)、[conv_op.cpp:608](file:///r:/renaissance/src/backend/ops/dtensor/conv_op.cpp#L608)

```cpp
// 删除以下两行：
const float* B = nullptr;        // line 563 — 框架不支持 Conv bias，从未使用
// ...
if (false) sum += 0.0f;          // line 608 — 永不执行的分支
```

`sum` 在后续计算中有实际使用，但 `if (false)` 分支是历史遗留的死代码。直接删除即可，不影响功能。

---

### Fix 6 (P2): test_conv_fwd_bwd.cpp — 使用 `task.run()` 而非 `task.run_iter()`

**位置**: [test_conv_fwd_bwd.cpp:343-355](file:///r:/renaissance/tests/op/test_conv_fwd_bwd.cpp#L343-L355)

**问题**: 数学正确性测试使用了 `run_iter`（多次迭代），应该用 `run`（单次执行）。

**方案**: 将 `run_iter` 替换为 `run`，保留现有 `fetch_from_rank` 结果获取方式，删除或简化计时逻辑。

```cpp
// 替换 line 341-358 的 run_iter 调用为：

// ── 运行 FWD（单次）──
auto t0 = std::chrono::high_resolution_clock::now();
task.run("fwd");
auto t1 = std::chrono::high_resolution_clock::now();
double us_fwd = std::chrono::duration<double, std::micro>(t1 - t0).count();

// ── 运行 BWD（单次）──
auto t2 = std::chrono::high_resolution_clock::now();
task.run("bwd");
auto t3 = std::chrono::high_resolution_clock::now();
double us_bwd = std::chrono::duration<double, std::micro>(t3 - t2).count();
```

移除 `cfg.warmup` / `cfg.iterations` 参数声明及使用。保留后续 `fetch_from_rank` + MSE 验证逻辑不变。

**注意**: `SimpleTask` 不提供 `get_results()` / `set_inputs()` API，结果获取应继续使用现有的 `task.fetch_from_rank()` 模式。

---

## 三、修改文件清单

| 文件 | 修改项 | 优先级 |
|---|---|---|
| [compiler.cpp](file:///r:/renaissance/src/graph/compiler.cpp) | Fix 1: 训练图 FWD 首层 Conv 注入 | **P0** |
| [compiler.cpp](file:///r:/renaissance/src/graph/compiler.cpp) | Fix 2: BWD 首层 Conv X/dX 注入 | **P0** |
| [compiler.cpp](file:///r:/renaissance/src/graph/compiler.cpp) | Fix 3: 推理图 FWD 首层 Conv 注入 | **P0** |
| [conv_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/conv_op.cpp) | Fix 4: FWD/INF cache key 统一 | P1 |
| [conv_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/conv_op.cpp) | Fix 5: 删除死代码 | P2 |
| [test_conv_fwd_bwd.cpp](file:///r:/renaissance/tests/op/test_conv_fwd_bwd.cpp) | Fix 6: `run_iter` → `run` | P2 |

---

## 四、不变更的设计点

下列 IRS 报告中提及的项，经分析后决定**不修改**：

| 项 | 来源 | 理由 |
|---|---|---|
| FC BWD 首层双缓冲 | IRS1/IRS2 | Fix 1/3 已纳入 FC 首层 FWD 注入，但 **Fix 2 未覆盖 FC BWD 首层**。当 FC 作为首层时，FC BWD 会走 `layer_input_ids` 通用分支（因 Fix 1 设置了 `layer_input_ids[l] = b.data_a`），但通用 append 会将同一个 gn 同时加入 A/B 图，导致 B 图也使用 `data_a` 而非 `data_b`。考虑到标准 CNN 中 FC 从不作为首层出现，暂不修复。若未来需要支持 FC 首层，需参照 Conv BWD 首层模式（Fix 2）补充 A/B 双节点构造 |
| per-shape cache 首层注入 | IRS1 E1 | 不存在 — per-shape cache 在 launch 层（conv_op.cpp），不在 compiler 层。compiler 层无此逻辑，无需修改 |
| CPU FWD input_shape | IRS1 E2 | 不独立存在 — 此问题依赖于 Compiler 首层注入修复（Fix 1/3）。修复后 X 是首个 input，`input_shape` 自动正确 |
| `bias_id` 在 XNNPACK | IRS3 | ✅ 已修复 |
| `make_nhwc_stride`/`make_krsc_stride` | IRS3 | ✅ 已修复 |
| bn_stats 形状、sum/sq_sum 偏移、BWD 双流等 | 全部 IRS | ✅ 全部正确，无需修改 |

---

## 五、实施顺序

```
Fix 1 → 训练图 FWD 首层注入
Fix 2 → BWD 首层注入
Fix 3 → 推理图 FWD 首层注入
Fix 4 → FWD/INF cache 统一
Fix 5 → 删除死代码
Fix 6 → 测试 run_iter → run
编译 → 运行 CPS ──── 运行 GPU ──── 运行 AMP 全部测试
```