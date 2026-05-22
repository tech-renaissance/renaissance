# LARS / LARS_NESTEROV 完整测试样例实施方案

## 〇、调查结论

### 当前状态

| 组件 | 状态 | 位置 |
|------|------|------|
| ComputeOp 枚举（LARS_UPDATE / LARS_NESTEROV_UPDATE / LARS_COMPUTE_TRUST_RATIO） | ✅ | [op_kind.h:235-237](file:///r:/renaissance/include/renaissance/graph/op_kind.h#L235-L237) |
| Compiler 图注入（为 LARS 生成 2N 个 COMPUTE 节点） | ✅ | [compiler.cpp:1392-1434](file:///r:/renaissance/src/graph/compiler.cpp#L1392-L1434) |
| MemoryPlan N_* 条件分配（Shape{1} 标量） | ✅ | [compiler.cpp:843-847](file:///r:/renaissance/src/graph/compiler.cpp#L843-L847) |
| 优化器标量分配（tc/wd/eps/beta/lr） | ✅ | [memory_plan.cpp:379-391](file:///r:/renaissance/src/graph/memory_plan.cpp#L379-L391) |
| OptimizerScalarIds 结构体 | ✅ | [graph_executor.h:21-28](file:///r:/renaissance/include/renaissance/backend/graph_executor.h#L21-L28) |
| Bias 复用（LARS Bias → RANGE_UPDATE_BIAS_MOMENTUM/NESTEROV） | ✅ | [compiler.cpp:1505-1512](file:///r:/renaissance/src/graph/compiler.cpp#L1505-L1512) |
| is_shape_invariant_graph 已移除 OPTIMIZER | ✅ | [computation_graph.h:165](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L165) |
| op_kind.cpp 字符串映射 | ✅ | [op_kind.cpp:136-138](file:///r:/renaissance/src/graph/op_kind.cpp#L136-L138) |
| **Kernel 实现（3 个 ComputeOp）** | ❌ | 不存在 |
| **op_registry 注册** | ❌ | [op_registry.cpp:39-54](file:///r:/renaissance/src/backend/op_registry.cpp#L39-L54) 无 `register_op_lars()` |
| **正确性测试样例** | ❌ | 不存在 |

### 核心结论

Compiler 层的图纸已经画好——图结构、内存布局、标量传输路径全部就绪。唯一阻塞点是 **backend kernel 层：3 个 ComputeOp 的 CUDA kernel + CPU fallback + op_registry 注册** 完全缺失。加上测试样例，共需完成 3 个修改点 + 2 个新测试文件。

---

## 一、需要实现的 Kernel

遵循 `optimizer_op.cu` 的现有风格（无分支、per-kernel 独立 `__global__`），在以下位置新增文件：

```
src/backend/ops/dtensor/
├── lars_op.cpp        ← 新增：CPU launch + CPU fallback + op_registry 注册
├── lars_op.cu         ← 新增：3 个 CUDA kernel + 3 个 CUDA launch 函数
```

### 1.1 `lars_op.cu` — 3 个 CUDA Kernel

**kernel 1: `lars_compute_trust_ratio_kernel`**

```
功能: 双归约求 η = tc·‖W‖₂ / (‖G_raw‖₂ + wd·‖W‖₂ + eps)
输入: w, g (device const float*), n (元素数), tc, wd, eps (标量 float)
输出: out_eta (device float*, 1 个元素)
处理: block-local warp reduce → atomicAdd 到 shared mem → block 0 thread 0 写 η
特殊: 若 ‖W‖₂ < 1e-12 或 ‖G_raw‖₂ < 1e-12: η = 1.0; clamp η = min(η, 100.0)
```

**kernel 2: `lars_update_kernel`**

```
功能: LARS Weight 更新
输入: w, g, m (device float*), n, lr*, beta*, wd*, eta*
公式: G' = G_raw + wd·W
     M_new = beta·M_old + eta·G'
     W = W - lr·M_new
输出: w, m (原地更新)
```

**kernel 3: `lars_nesterov_update_kernel`**

```
功能: LARS Nesterov Weight 更新
输入: 同 lars_update_kernel
公式: G' = G_raw + wd·W
     M_new = beta·M_old + eta·G'
     W = W - lr·(eta·G' + beta·M_new)
输出: w, m (原地更新)
```

**符号对照**（来自 LAR_FINAL3.md §4.1）：

| symbol | 含义 | 来源 |
|--------|------|------|
| W | 参数权重 | w 参数 |
| G_raw | 原始梯度 | g 参数 |
| M_old | 旧动量缓冲 | m 参数（输入） |
| M_new | 新动量缓冲 | m 参数（输出） |
| η | trust ratio | eta 标量（由 trust_ratio kernel 写入 N_*） |
| tc | trust coefficient | scalar_ids.tc |
| wd | weight decay | scalar_ids.wd |
| eps | 数值保护 | scalar_ids.eps |
| lr | 学习率 | scalar_ids.lr |
| beta | momentum 系数 | scalar_ids.beta |

### 1.2 `lars_op.cpp` — CPU Fallback + CUDA Launch

参照 [axpy_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/axpy_op.cpp) 的注册模式：

1. **CUDA launch 函数签名**（与 `ComputeOpEntry.launch_cuda` 一致）：

```cpp
static void launch_lars_compute_trust_ratio_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state);
```

2. **DTensor 指针解析**：通过 `input_ids`/`output_ids` 索引获取 `mp.get_dtensor(id)`，再用 `ctx.ptr_at(id)` 获取设备指针。

3. **标量解析**：tc/wd/eps/lr/beta 通过 `input_ids` 传入，使用与 [optimizer_op.cpp:44-50](file:///r:/renaissance/src/backend/ops/range/optimizer_op.cpp#L44-L50) 相同的 `scalar_ptr<>()` 工具从 `ArenaKeeper` 解析设备指针。

4. **流选择**：使用 `StreamKind::UPDATE`（与 optimizer RangeOp 一致）。

5. **CPU fallback**：
   - `lars_compute_trust_ratio_cpu()`：逐元素累加求 sum_w2、sum_g2，然后算 η
   - `lars_update_cpu()`：单线程循环，按 LARS 公式更新
   - `lars_nesterov_update_cpu()`：单线程循环，按 Nesterov 公式更新

6. **注册函数**：

```cpp
void register_op_lars() {
    // LARS_COMPUTE_TRUST_RATIO
    auto& trust_entry = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_COMPUTE_TRUST_RATIO)];
    trust_entry.op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
    trust_entry.launch_cpu = launch_lars_compute_trust_ratio_cpu;
#ifdef TR_USE_CUDA
    trust_entry.launch_cuda = launch_lars_compute_trust_ratio_cuda;
#endif

    // LARS_UPDATE
    auto& update_entry = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_UPDATE)];
    update_entry.op = ComputeOp::LARS_UPDATE;
    update_entry.launch_cpu = launch_lars_update_cpu;
#ifdef TR_USE_CUDA
    update_entry.launch_cuda = launch_lars_update_cuda;
#endif

    // LARS_NESTEROV_UPDATE
    auto& nest_entry = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_NESTEROV_UPDATE)];
    nest_entry.op = ComputeOp::LARS_NESTEROV_UPDATE;
    nest_entry.launch_cpu = launch_lars_nesterov_update_cpu;
#ifdef TR_USE_CUDA
    nest_entry.launch_cuda = launch_lars_nesterov_update_cuda;
#endif
}
```

---

## 二、需要修改的现有代码

### 2.1 `src/backend/op_registry.cpp` — 注册 LARS ops

在 `register_default_ops()` 中（[L54](file:///r:/renaissance/src/backend/op_registry.cpp#L54) 之后）添加一行：

```cpp
register_op_lars();
```

### 2.2 `include/renaissance/backend/op_registry.h` — 声明注册函数

在现有声明列表末尾添加：

```cpp
void register_op_lars();
```

---

## 三、测试样例设计

### 3.1 测试架构选择

LARS 的 trust ratio 计算和更新都是逐 DTensor 的 **ComputeOp**（非 RangeOp）。`SimpleTask` 的 capture 路径已支持 COMPUTE 节点（[capture_cuda.cpp:104-111](file:///r:/renaissance/src/graph/capture_cuda.cpp#L104-L111)），因此可以直接用 SimpleTask 手动构图测试，无需 DeepLearningTask + Compiler。

### 3.2 `tests/correction/test_lars_weight.cpp` — LARS Weight 正确性测试

**测试策略**：

- 构造一个简单的两层场景（两个 Weight DTensor），模拟 Compiler 对 LARS 生成的两个 COMPUTE 节点
- 验证 `LARS_COMPUTE_TRUST_RATIO` → `LARS_UPDATE` 的完整 pipeline
- **不**验证 Bias（Bias 走 RangeOp，已有 `test_momentum_bias.cpp` 覆盖）

**DTensor 布局**：

```
Region::W_FC_WEIGHT:  d_w_a (shape=2×8×4×4=256), d_w_b (shape=4×4×8×4=512)
Region::G_FC_WEIGHT:  d_g_a (shape=2×8×4×4=256), d_g_b (shape=4×4×8×4=512)
Region::M_FC_WEIGHT:  d_m_a (shape=2×8×4×4=256), d_m_b (shape=4×4×8×4=512)
Region::N_FC_WEIGHT:  d_eta_a (shape=1, 标量),    d_eta_b (shape=1, 标量)

S_SCALAR_FP32:        d_lr, d_wd, d_beta, d_tc, d_eps
```

**关键：手动构造 COMPUTE 节点**（对照 LARD.md 中 Compiler 生成的图结构）：

```cpp
ComputationGraph g;

// Layer a: trust_ratio + update
GraphNode trust_a;
trust_a.kind = GraphNode::Kind::COMPUTE;
trust_a.compute_op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
trust_a.input_ids  = {d_w_a.id, d_g_a.id, d_tc.id, d_wd.id, d_eps.id};
trust_a.output_ids = {d_eta_a.id};
g.append(std::move(trust_a));

GraphNode update_a;
update_a.kind = GraphNode::Kind::COMPUTE;
update_a.compute_op = ComputeOp::LARS_UPDATE;
update_a.input_ids  = {d_w_a.id, d_g_a.id, d_m_a.id, d_eta_a.id,
                       d_lr.id, d_beta.id, d_wd.id};
update_a.output_ids = {d_w_a.id, d_m_a.id};
g.append(std::move(update_a));

// Layer b: 同理
// ...
```

**关键注意事项**：

1. **trust_ratio 和 update 的 input_ids 布局不同**：trust_ratio 需要 tc/wd/eps；update 需要 lr/beta/wd。这是与 Lod 的 Compiler 注入 (compiler.cpp:1416-1431) 一致的。

2. **kernal 通过 `ctx.ptr_at(id)` 获取指针**：标量 DTensor 的名词来自 `ArenaKeeper` 的 `ptr_at(rank, mp.get_dtensor(id).offset())`，与现有 RangeOp 一致。

3. **TEST only uses W_FC_WEIGHT region** (FC weights only), since SimpleTask has no FirstConv/DeepConv layer structure. This is sufficient to verify the kernel math.

**GRAPH ID**: Use `GraphId::OPTIMIZER` (value 10) to match Compiler's layout.

**STREAM**: Use `StreamKind::UPDATE` (same as RangeOp optimizer tests).

**验证公式**（来自 LAR_FINAL3.md §4.1，**为 LARS / 非 Nesterov**）：

```
η = tc · ‖W‖₂ / (‖G‖₂ + wd · ‖W‖₂ + ε)
若 ‖W‖₂ < 1e-12 或 ‖G‖₂ < 1e-12: η = 1.0
η = min(η, 100.0)

G' = G + wd · W
M = β · M_old + η · G'
W = W - lr · M
```

### 3.3 `tests/correction/test_lars_nesterov_weight.cpp` — LARS Nesterov Weight 正确性测试

结构与 §3.2 完全一致，唯二不同：
1. Update 节点的 `compute_op` 设为 `ComputeOp::LARS_NESTEROV_UPDATE`
2. 验证公式不同：

```
η = tc · ‖W‖₂ / (‖G‖₂ + wd · ‖W‖₂ + ε)    （同 LARS）
η = min(η, 100.0)

G' = G + wd · W
M = β · M_old + η · G'
W = W - lr · (η · G' + β · M)               ← Nesterov: 前瞻项含 M_new
```

### 3.4 `tests/correction/CMakeLists.txt` — 注册两个新测试

在现有的 `test_adam_bias` 条目之后添加：

```cmake
# === LARS 优化器测试 ===
add_executable(test_lars_weight test_lars_weight.cpp)
target_link_libraries(test_lars_weight PRIVATE renaissance)
target_compile_definitions(test_lars_weight PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_lars_weight)
endif()
set_target_properties(test_lars_weight PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_lars_weight PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_lars_weight PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_lars_weight")

add_executable(test_lars_nesterov_weight test_lars_nesterov_weight.cpp)
target_link_libraries(test_lars_nesterov_weight PRIVATE renaissance)
target_compile_definitions(test_lars_nesterov_weight PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_lars_nesterov_weight)
endif()
set_target_properties(test_lars_nesterov_weight PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_lars_nesterov_weight PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_lars_nesterov_weight PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_lars_nesterov_weight")
```

---

## 四、实施顺序

```
P1: lars_op.cu     — 3 个 CUDA kernel 实现
P2: lars_op.cpp    — CPU fallback + CUDA launch + register_op_lars()
P3: op_registry.h  — 声明 register_op_lars()
P4: op_registry.cpp — register_default_ops() 中调用
P5: test_lars_weight.cpp          — 正确性测试
P6: test_lars_nesterov_weight.cpp — 正确性测试
P7: CMakeLists.txt — 注册测试目标
```

**依赖**：P1→P2→P3+P4→P5+P6→P7。P3 是纯声明不可 bloking，P3+P4 同批；P5+P6 同批；P7 同批。

---

## 五、不需要实现 / 不需要修改的

| 项目 | 理由 |
|------|------|
| Compiler 层 | 已完整：图注入、MemoryPlan、ScalarIds（见 §〇） |
| Bias 测试 | LARS Bias 退化为 Momentum/Nesterov，已有 test_momentum_bias / test_nesterov_bias 覆盖 |
| LARS_COMPUTE_TRUST_RATIO 的 Nesterov 变体 | 公式与 LARS 完全相同（η 无关 Nesterov） |
| RangeOp 新增 | LARS Weight 走 ComputeOp，不新增 RangeOp |
| is_shape_invariant_graph | 已移除 OPTIMIZER |
| allreduce 图的 nccl graph 判定 | 无关 |

---

## 六、风险点

| # | 风险 | 缓解 |
|---|------|------|
| R1 | COMPUTE 节点在 capture 时 `g_compute_op_table` 的 `launch_cuda` 为空 → `replay_compute_node_default` 被调用（空操作） | P3 必须在测试运行前完成注册 |
| R2 | `LARS_COMPUTE_TRUST_RATIO` 归约结果写入 N_* 标量后，`LARS_UPDATE` 跨 kernel 读取——需确保 capture 内的依赖关系 | 两个 COMPUTE 节点在同一 `GraphId::OPTIMIZER` 图内串行 capture，保证顺序 |
| R3 | trust_ratio kernel 使用 `atomicAdd` 的 shared mem，可能影响 A100 SM 调度 | 仅 2 个 atomicAdd 到同一个 block 内的 shared mem，竞争概率极低 |
| R4 | N_* 标量通过 `output_ids` 传给下一个 COMPUTE 节点，capture 期间 CUDA graph 能看到这个依赖 | 与 RangeOp 的 `input_ranges`/`output_ranges` 不同，COMPUTE 节点靠 CUDA stream 串行保证顺序 |

---

## 附录 A：测试标量值建议

为确保验证的有效性，使用与 CLOSED.py 参考实现一致的标量值：

| 标量 | LARS | LARS_NESTEROV |
|------|------|---------------|
| lr | 0.001 | 0.001 |
| wd | 5e-5 | 8e-5 |
| beta| 0.9 | 0.9 |
| tc  | 0.001 | 0.001 |
| eps | 1e-8 | 1e-8 |

权重/梯度/动量的初值用显式均匀填充（非随机），保证多 rank 可复现。

## 附录 B：与现有 9 个 RangeOp 测试的对照

| 现有测试 | 类型 | 对应 LARS 等价 |
|----------|------|---------------|
| test_sgd_weight.cpp | RangeOp | — |
| test_sgd_bias.cpp | RangeOp | — |
| test_momentum_weight.cpp | RangeOp | — |
| test_momentum_bias.cpp | RangeOp | **LARS Bias** (tc=0 退化) |
| test_nesterov_weight.cpp | RangeOp | — |
| test_nesterov_bias.cpp | RangeOp | **LARS_NESTEROV Bias** (tc=0 退化) |
| test_adam_weight.cpp | RangeOp | — |
| test_adamw_weight.cpp | RangeOp | — |
| test_adam_bias.cpp | RangeOp | — |
| **test_lars_weight.cpp** | **ComputeOp** | **新增** |
| **test_lars_nesterov_weight.cpp** | **ComputeOp** | **新增** |

这样优化器全家桶（SGD/Momentum/Nesterov/Adam/AdamW/LARS/LARS_NESTEROV × Weight/Bias）的正确性就全覆盖了。