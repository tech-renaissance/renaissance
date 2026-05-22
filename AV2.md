# AV2.md — RangeOp 体系终局审计报告

> 审计时间：2026-05-20
> 审计依据：RN_FINAL.md · RG1.md · RG2.md · RG3.md · RN_REVIEW.md · REPORT.md
> 审计范围：op_kind.h/cpp · memory_plan.h/cpp · computation_graph.h · compiler.cpp · capture_cpu/cuda.cpp · op_registry.h/cpp · 全部 9 个 range op 实现文件
> 编译状态：✅ 122/122 目标通过（vcvars64 + ninja -j30）

---

## 一、已确认修复的历史问题

以下问题在 RN_REVIEW.md 中被标记为 P0/P1，经本次审计确认**已修复**：

| 历史问题 | 修复状态 | 验证位置 |
|---------|---------|---------|
| ALLREDUCE 用了错误的 `RANGE_MEAN_ALLREDUCE` | ✅ 已修复 | `compiler.cpp` L1370-1376 现用 `RANGE_SUM_ALLREDUCE` |
| Optimizer 标量参数重复 beta、缺少 beta2 | ✅ 已修复 | `graph_executor.h` 已含 `beta2` 字段；`compiler.cpp` Weight Adam 推 `[lr,wd,beta,beta2,eps]` |
| `check_op.cu` 数据竞争 `s_has_nan[0]=1` | ✅ 已修复 | 已改为 `atomicOr(&s_has_nan[0], 1)` |
| `cast_op.cu` 缺少 `#include <algorithm>` | ✅ 已修复 | 已添加 |
| `append_if_non_empty` lambda 依赖旧 API | ✅ 已修复 | 该 lambda 已删除，全部改为 `region_range()` 直接构造 |
| `OptimizerScalarIds` 缺少 `beta2` | ✅ 已修复 | `graph_executor.h` L24 已添加 `beta2` |
| Bias scalar 映射错位（beta 读成 wd） | ✅ 已修复 | `compiler.cpp` 不再推 `wd` 到 Bias，launcher 索引与推入顺序匹配 |
| Compiler 空 Region 过滤失效 | ✅ 已修复 | 全面改用 `is_region_populated()` |

---

## 二、当前真实存在的问题（按严重度排序）

### 🔴 P0：回归性问题 — Bias SGD CPU launcher 的 `num_inputs` 守卫未同步

**位置**：`src/backend/ops/range/optimizer_op.cpp` L363

**根因**：小伙伴在修复 Bias scalar 映射时（删除 Bias 的 `wd` 推入），`launch_opt_bias_sgd_cpu` 的守卫条件未同步调整：

```cpp
// 修复前：Compiler 推 [lr, wd] → num_inputs = 2 → 检查 < 2 → 2<2=false → PASS
// 修复后：Compiler 推 [lr]     → num_inputs = 1 → 检查 < 2 → 1<2=true  → 错误返回！
if (!mp || op_ctx->num_input_ranges < 2 || op_ctx->num_inputs < 2) return;
//                                                                    ↑ 应从 2 改为 1
```

**影响范围**：仅 **CPU 路径的 Bias SGD**。CUDA launcher 无 `num_inputs` 检查，不受影响。Bias Momentum/Nesterov/Adam 的检查值在删除 `wd` 后恰好对齐，也不受影响。

**修复方案**：将 `op_ctx->num_inputs < 2` 改为 `op_ctx->num_inputs < 1`（或直接删除 `num_inputs` 检查，保留 `num_input_ranges < 2` 即可）。

---

### 🟡 P1：H2D 算子仍依赖旧 API `get_range_op_range()`

**位置**：`src/backend/ops/range/h2d_op.cpp` L104（CUDA）、L144（CPU）

**问题**：`launch_range_h2d_copy_cuda` 和 `launch_range_h2d_copy_cpu` 在运行时调用 `mp.get_range_op_range(node.range_op)`，完全忽略 `node.output_ranges` 中 Compiler 推入的延迟态 `MemRange`。

**后果**：功能正确（旧表中的范围和 `region_range()` 解析结果一致），但 H2D 是**最后一个未完全迁移到 `region_range()` 的 RangeOp**。`get_range_op_range()` 是 Phase 4 计划删除的方法，若不迁移，Phase 4 清理时会阻断编译。

**修复方案**：
1. H2D CUDA launcher 改为遍历 `node.output_ranges`，对每个 `MemRange` 调用 `resolve_region_bounds()` 获取实际 offset/size。
2. H2D CPU launcher 同理，遍历 `op_ctx->output_ranges`。

---

### 🟡 P2：`build_range_op_ranges()` 和 `range_op_ranges_` 仍残留

**位置**：
- `src/graph/memory_plan.cpp` L643-740（`build_range_op_ranges()` 实现）
- `include/renaissance/graph/memory_plan.h` L296-297（`range_op_ranges_` 数组声明）
- `include/renaissance/graph/memory_plan.h` L235（`get_range_op_range()` 声明）
- `src/graph/memory_plan.cpp` L819-823（`get_range_op_range()` 实现）

**问题**：Phase 4 应删除的 10 个旧枚举对应的预计算范围仍存在于代码中。这些代码在当前编译后**没有任何调用方**（Compiler 已全部改用 `region_range()`），但物理上仍占用编译时间、二进制体积和认知负担。

**修复方案**：Phase 4 直接删除：
1. `memory_plan.h` 中删除 `RangeOpRange` 结构体、`range_op_ranges_` 数组、`get_range_op_range()` 方法。
2. `memory_plan.cpp` 中删除 `build_range_op_ranges()` 函数及 `finalize()` 中对它的调用。
3. `op_kind.h` 中删除 10 个旧枚举值。
4. `op_kind.cpp` 中删除对应的 `range_op_to_string()` 分支。
5. 更新 `COUNT` 从 31 → 22。

---

### 🟡 P3：Compiler 中部分 RangeOp 未使用 `is_region_populated()` 过滤

**位置**：
- `compiler.cpp` L1368-1377（ALLREDUCE FIRST_COMM / DEEP_COMM）
- `compiler.cpp` L1460-1495（Weight 优化器）
- `compiler.cpp` L1521-1554（Bias 优化器）
- `compiler.cpp` L1560-1565（EMA_UPDATE）
- `compiler.cpp` L1636-1641（SEMA_SWITCH）
- `compiler.cpp` L1646-1658（EMA CAST）

**问题**：这些 RangeOp 节点在构图时**无条件生成**，不检查对应 Region 是否为空。虽然 capture 期 launcher 会检查 `size == 0` 并跳过，但图中会多出一些无意义的空节点。

**评估**：这是**设计选择而非 bug**。旧代码中 `if (r.size > 0)` 的检查因延迟态 `size` 恒为 0 而实际失效，导致旧代码也从未正确过滤。当前无条件生成 + launcher 运行时跳过的策略与旧行为等价，不会引入新的运行时错误。但如需减少图中空节点数量，可在构图前增加 `is_region_populated()` 检查。

---

### 🟢 P4：`__launch_bounds__` 统一为 128（设计偏差，非 bug）

**位置**：`src/backend/ops/range/optimizer_op.cu` L15

**问题**：RN_FINAL.md §9.3 建议 Adam/AdamW 用 `__launch_bounds__(128)`（寄存器压力大），SGD/Momentum/Nesterov 用 `__launch_bounds__(256)`（寄存器压力小）。当前实现统一使用 `__launch_bounds__(128, 2)`。

**评估**：不会导致错误，SGD/Momentum/Nesterov 仍可正确运行。仅可能在超大规模参数场景下损失少量 occupancy。属于**性能优化项**，非正确性 bug。

---

### 🟢 P5：Adam/AdamW 缺少 bias correction

**位置**：`src/backend/ops/range/optimizer_op.cu` L82-132

**问题**：当前 kernel 直接使用原始 `m[i]` 和 `v[i]`，未执行标准 Adam 的 bias correction：
```cpp
// 标准 PyTorch Adam:
m_hat = m[i] / (1 - powf(b1, t));
v_hat = v[i] / (1 - powf(b2, t));
// 当前实现缺少此步骤
```

**评估**：RN_FINAL.md 对此**未作要求**。PyTorch 默认开启 bias correction，但部分实现（如 Hugging Face 的 `AdamW`）允许关闭。若需与 PyTorch 默认行为逐 bit 对齐，需补充 `current_step` 标量参数并在 kernel 中应用 correction。属于**设计选择**。

---

### 🟢 P6：RangeOp 专用单元测试缺失

**位置**：`tests/` 目录

**问题**：RN_FINAL.md §7.1 和 RG3.md 均建议为每个新 RangeOp 编写独立单元测试（如 `test_range_clear.cpp`），验证单 Region / 多 Region / 空 Region 场景。当前 `tests/` 目录中**没有任何 RangeOp 专用测试**。

**评估**：不阻塞 Phase 4，但应在清理完成后补充，以确保 22 个 RangeOp 的 CPU + CUDA 路径均被覆盖。

---

## 三、修复方案汇总

| 优先级 | 问题 | 文件 | 修改内容 | 工作量 |
|:------:|------|------|---------|:------:|
| **P0** | Bias SGD CPU 回归 | `optimizer_op.cpp` L363 | `num_inputs < 2` → `< 1` | 1 行 |
| **P1** | H2D 旧 API 依赖 | `h2d_op.cpp` | CUDA/CPU launcher 改遍历 `node.output_ranges` + `resolve_region_bounds()` | ~20 行 |
| **P2** | 死代码清理 | `memory_plan.h/cpp` | 删除 `build_range_op_ranges()`、`range_op_ranges_`、`get_range_op_range()`、`RangeOpRange`、`OpSegment` | ~120 行 |
| **P2** | 旧枚举清理 | `op_kind.h/cpp` | 删除 10 个旧枚举及 `range_op_to_string()` 分支；COUNT 31→22 | ~30 行 |
| **P3** | 空节点过滤 | `compiler.cpp` | ALLREDUCE / EMA / 优化器 构图前增加 `is_region_populated()` 检查（可选） | ~10 行 |
| **P4** | `__launch_bounds__` 拆分 | `optimizer_op.cu` | Adam/AdamW 保持 128；SGD/Momentum/Nesterov 改为 256 | 3 行 |
| **P5** | Adam bias correction | `optimizer_op.cu` + `compiler.cpp` | 新增 `current_step` 标量 DTensor，kernel 中应用 `m/(1-b1^t)`、`v/(1-b2^t)`（可选） | ~15 行 |
| **P6** | 单元测试 | `tests/range/` | 新建 `test_range_clear.cpp`、`test_range_cast.cpp` 等（可选） | ~200 行 |

---

## 四、下一步计划建议

### 立即执行（1 小时内）
1. 修复 P0 回归：`optimizer_op.cpp` L363 `num_inputs < 2` → `< 1`。
2. 编译验证（`ninja -j30`）。

### Phase 4 清理（1-2 天）
1. 迁移 H2D 到 `node.output_ranges`（最后一个旧 API 用户）。
2. 删除 10 个旧枚举 + `build_range_op_ranges()` + `range_op_ranges_` + `get_range_op_range()`。
3. 更新 `range_op_to_string()`，`COUNT` 自动变为 22。
4. 编译 + 运行全量 `tests/` 回归测试。

### 可选优化（后续迭代）
1. 按算法拆分 `__launch_bounds__`（128 vs 256）。
2. 补充 Adam bias correction（如需 PyTorch 对齐）。
3. 编写 RangeOp 专用单元测试。
