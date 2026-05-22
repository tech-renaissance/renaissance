# 三流架构 最终修复方案

**文档编号**：FIX-MULTI-FINAL  
**日期**：2026-05-19  
**编制**：技术觉醒团队（S / K / D 三方共识）  
**范围**：capture 框架层 4 处修改，算子层零改动  

---

## 一、共识结论

经 S、K、D 三轮独立审查，三方一致确认：

**算子实现（fc_op.cpp / gap_op.cpp / op_stream_policy.cpp）完全正确，无需任何修改。**

当前代码存在 3 个真实问题，均在框架层，修复仅需 **2 个文件、约 20 行**。

---

## 二、修改清单

| # | 文件 | 行号 | 改动 | 优先级 |
|---|------|:---:|------|:---:|
| 1 | `include/renaissance/graph/capture_multi_stream.h` | 32 | `kMaxActiveStreams: 3 → 5` | 🔴 必须 |
| 2 | `src/graph/capture_multi_stream.cpp` | 1 | 添加 `#include "renaissance/backend/op_stream_policy.h"` | 🟡 配套 |
| 3 | `src/graph/capture_multi_stream.cpp` | 68-80 | `insert_cross_op_barrier` 星型广播 → 精确 pre-node wait | 🔴 必须 |
| 4 | `src/graph/capture_cuda.cpp` | 103-113 | 删除独立 pre-node wait（INSERT 已承包） | 🟡 配套 |

---

## 三、具体 diff

### 3.1 修改 1：kMaxActiveStreams 扩容

**文件**：`include/renaissance/graph/capture_multi_stream.h`

```diff
-struct MultiStreamCaptureState {
-    static constexpr int kMaxActiveStreams = 3;
+struct MultiStreamCaptureState {
+    static constexpr int kMaxActiveStreams = 5;

     PerStreamState streams[kMaxActiveStreams] = {};
```

**理由**：底层 `DeviceContext` 实际上创建了 5 个 CUDA 流（`streams_[5]`）、5 个 cuBLAS handle 和 5 个 cuDNN handle，对应 `StreamKind` 枚举的全部 5 个值（TRANS / COMP_1 / COMP_2 / COMP_3 / UPDATE）。`kMaxActiveStreams = 3` 与之严重不匹配：

- 预注册阶段 `get_or_register(primary)` + COMP_1/2/3 可能已达 4 个流（当 primary 不在 COMP_1/2/3 中时），直接触发 `TR_CHECK(num_active < 3)` 崩溃
- 即使预注册恰好 3 个，`replay_range_node_default` 注册 `UPDATE` 流时也会崩溃
- 扩容到 5 与底层完全对齐，消除崩溃风险

---

### 3.2 修改 2：添加依赖头文件

**文件**：`src/graph/capture_multi_stream.cpp`

```diff
 #include "renaissance/graph/capture_multi_stream.h"
 #include "renaissance/graph/computation_graph.h"
 #include "renaissance/backend/device_context.h"
 #include "renaissance/backend/graph_executor.h"
+#include "renaissance/backend/op_stream_policy.h"
 #include "renaissance/core/logger.h"
```

**理由**：修改后的 `insert_cross_op_barrier` 需要调用 `get_op_default_stream`。

---

### 3.3 修改 3：INSERT 星型广播 → 精确 pre-node wait

**文件**：`src/graph/capture_multi_stream.cpp`

```diff
-void insert_cross_op_barrier(const GraphNode& /*prev_node*/,
-                              const GraphNode& /*next_node*/,
+void insert_cross_op_barrier(const GraphNode& /*prev_node*/,
+                              const GraphNode& next_node,
                               MultiStreamCaptureState& state,
-                              const DeviceContext& /*ctx*/) {
+                              const DeviceContext& ctx) {
     int out_idx = state.output_stream_idx;
     if (out_idx < 0) return;
-    for (int i = 0; i < state.num_active; ++i) {
-        if (i == out_idx) continue;
-        if (!state.streams[i].has_pending_work) continue;
-        cudaStreamWaitEvent(state.streams[i].stream,
-                           state.streams[out_idx].last_done_event, 0);
+
+    if (next_node.kind == GraphNode::Kind::COMPUTE) {
+        StreamKind target_sk = get_op_default_stream(next_node.compute_op);
+        cudaStream_t target_s = static_cast<cudaStream_t>(ctx.stream(target_sk));
+        int target_idx = state.find_stream_index(target_s);
+        if (target_idx >= 0 && target_idx != out_idx) {
+            cudaStreamWaitEvent(target_s,
+                state.streams[out_idx].last_done_event, 0);
+        }
     }
 }
```

**理由**：

- **去冗余**：原星型广播使所有曾活跃的流都 wait 上一个输出流，产生不必要的跨流依赖边（Graph 节点数增多）
- **语义正确**：上一个算子的输出流仅需被下一个算子的**代表流**等待——其他流的依赖应由算子自治机制内部处理（FC_BWD 已验证）。广播等待覆盖了框架不该关心的范围内依赖
- **消除 has_pending_work 累积效应**：原实现依赖 `has_pending_work` 决策，该标志永不重置导致曾用过的流持续参与后续所有 INSERT 的 wait。改为精确依赖后不再依赖此标志
- **与 finalize 不冲突**：分析已验证 finalize 的星型收束与算子自治等待作用于不同时间线（早期 vs 晚期事件），形成单向 DAG 链，无回路

---

### 3.4 修改 4：删除冗余 pre-node wait

**文件**：`src/graph/capture_cuda.cpp`

```diff
     if (i > 0) {
         insert_cross_op_barrier(nodes[i-1], node, state, ctx);
-
-        if (state.output_stream_idx >= 0
-            && node.kind == GraphNode::Kind::COMPUTE) {
-            StreamKind target_sk = get_op_default_stream(node.compute_op);
-            cudaStream_t target_s = static_cast<cudaStream_t>(ctx.stream(target_sk));
-            int target_idx = state.find_stream_index(target_s);
-            if (target_idx >= 0 && target_idx != state.output_stream_idx) {
-                cudaStreamWaitEvent(target_s,
-                    state.streams[state.output_stream_idx].last_done_event, 0);
-            }
-        }
     }
```

**理由**：修改后的 `insert_cross_op_barrier` 已精确完成 pre-node wait（"下一个算子的代表流等上一个算子的输出流"）。capture 循环内的独立 pre-node wait 代码块是重复逻辑，删除以简化。

---

## 四、修改后的依赖验证

### 4.1 单节点 FC_BWD（primary = COMP_3）

```
无 INSERT（i == 0）
dummy join: COMP_1 wait COMP_3, COMP_2 wait COMP_3
FC_BWD:     COMP_3 wait COMP_1（dX 等 dW）
finalize:   COMP_3 wait COMP_1, COMP_3 wait COMP_2

时间线: dummy → dW → ev_dw → dX → ev_dx → primary 收束  ✅
```

### 4.2 多节点 GAP_BWD → FC_BWD（primary = COMP_2）

```
COMP_2: GAP_BWD → ev_gap → dB → ev_db → wait ev_dw → wait ev_dx（finalize）
COMP_1: wait ev_gap（自治）→ dW → ev_dw
COMP_3: wait ev_gap（INSERT）→ wait ev_dw（内部）→ dX → ev_dx

全局链: ev_gap → dW → ev_dw → dX → ev_dx → COMP_2 收束  ✅
       ev_gap → dB → ev_db → COMP_2 收束                ✅
```

---

## 五、已验证无需修改的部位

| 组件 | 结论 | 审查人 |
|------|:---:|:---:|
| `fc_op.cpp` (AMP + FP32 BWD) | ✅ 正确 | S, K, D |
| `fc_op.cpp` (FWD) | ✅ 正确 | D |
| `gap_op.cpp` (FWD + BWD) | ✅ 正确 | D |
| `op_stream_policy.cpp` | ✅ 正确 | S, K, D |
| `finalize_cross_stream_barrier` | ✅ 正确（最终 DAG 链无回路） | K, D |
| dummy join | ✅ 正确（ThreadLocal 标准模式） | D |
| 事件预注册与重建 | ✅ 正确 | D |
| FC_BWD 算子自治等待 | ✅ 正确（真实数据依赖） | S, K, D |

---

## 六、已知但不处理的问题

| 问题 | 风险 | 处理 |
|------|:---:|------|
| `has_pending_work` 永不重置 | 🟢 无（改为精确 INSERT 后已无关） | 不处理 |
| `alloc_temp_event` 为死代码 | 🟢 无（无调用点） | 不处理，后续可清理 |
| `prev_node` 参数未使用 | 🟢 无（代码异味） | 不处理，后续可移除 |
| `replay_range_node_default` 无自治 wait | 🟡 低（非新问题，RangeOp 专用实现需自行处理） | 不处理 |
| GAP FWD 全局缓存 per-handle 键 | 🟡 低（仅多 GPU 部署时有风险） | 不处理 |

---

## 七、执行说明

1. 按第三节 diff 依次修改 2 个文件（`capture_multi_stream.h/cpp` + `capture_cuda.cpp`）
2. 重新编译：`configure.py` + `cmake --build`
3. 运行验证：`test_fc_amp` + `test_gap_perf` + `test_gap_fc_perf`（含单节点与多节点）
4. 确认 A100 数据无回归

---

## 八、参考文档

| 文档 | 内容 |
|------|------|
| `FIX_MULTI.md` | D 的初审报告 |
| `FIX_MULTI2.md` | D 初审 + K 复审 |
| `FIX_MULTI3.md` | S / K / D 三方终审 |
| `MULTI_D.md` | 三流架构综合验证报告 |
| `cbr_fwd_fp16.cpp` | 链式事件拓扑参考实现 |