# renAIssance 三流架构技术白皮书

**文档编号**：TR4-MULTI-STREAM-FINAL  
**版本**：3.0（终版）  
**日期**：2026-05-19  
**编制**：技术觉醒团队（S / K / D 三方共识）  
**平台**：NVIDIA A100 80GB / CUDA 13.1 / cuDNN 9.17  
**结论**：三流架构是 DTensor 算子系统的唯一生产基线，不可回退。

---

## 执行摘要

renAIssance 框架在 DTensor 算子系统中采用了**三流 CUDA 架构**。该架构将 GPU 计算任务按功能特征分配到三条独立 CUDA Stream 上，通过有向无环图（DAG）表达真实数据依赖，以事件（Event）实现 GPU 侧同步，并在 CUDA Graph 中固化为静态拓扑。

**核心结论（一条就够）**：

> 三流架构的 GAP+FC 联合执行耗时 **587.23 μs**，单流架构为 **614.57 μs**，三流节省 **27.34 μs（−4.5%）**。CUDA Graph 联合执行比分别跑两份单独 Graph 快 **15.55 μs（−2.6%）**。数学正确性验证 6/6 通过（CPU FP32 / GPU FP32 / GPU AMP），覆盖 test_fc_fwd_bwd 与 test_gap。三流架构不是幻觉，是真实有效的工程优化。

---

## 一、架构总览

### 1.1 三条流的职责

renAIssance 底层 `DeviceContext` 维护 5 条 CUDA Stream（对应 `StreamKind` 枚举），其中 3 条用于计算：

| 流枚举值 | 物理流 | 分配对象 | 设计意图 |
|----------|:------:|------|------|
| `COMP_1` | Stream 1 | Conv/FC FWD、FC BWD dW | 主干计算流（cublasGemmEx） |
| `COMP_2` | Stream 2 | GAP、BN、Pooling、FC BWD dB | 池化与归约流（cuDNN FE / Reduction） |
| `COMP_3` | Stream 3 | Activation、FC BWD dX | 逐元素与输出流（FC BWD 最终输出） |

**关键设计原则**：COMP_1 是"重计算流"（GEMM 密集），COMP_2 是"归约流"（Reduction / Pooling），COMP_3 是"轻量结果流"（Elementwise）。三条流各司其职，资源需求互补而非竞争。

### 1.2 静态流策略

[`op_stream_policy.cpp`](file:///r:/renaissance/src/backend/op_stream_policy.cpp) 提供了算子到流的静态映射函数 `get_op_default_stream(ComputeOp)`。每个算子声明一个**代表流**——即该算子的"输出流"，下游算子通过框架的 `insert_cross_op_barrier` 等待此流。

| 算子 | 代表流 | 实际工作流 |
|------|:---:|------|
| FC / CONV FWD | COMP_1 | COMP_1（单流） |
| FC / CONV BWD | COMP_3 | COMP_1(dW) + COMP_2(dB) + COMP_3(dX) |
| GAP / MaxPool | COMP_2 | COMP_2（单流） |
| BN | COMP_2 | COMP_2（单流） |
| ReLU / Tanh | COMP_3 | COMP_3（单流） |

**注意**：FC/CONV BWD 声明代表流为 COMP_3，因为 dX 是该算子的最终输出，下游算子需要等 dX 完成。但 dW 和 dB 可在 COMP_1/COMP_2 上并发执行。这个语义的精妙之处在于：**声明流告诉框架依赖方向，实际工作流由算子内部决定**。

### 1.3 Capture 流容量

`MultiStreamCaptureState::kMaxActiveStreams = 5`，与 `DeviceContext` 底层 5 条 CUDA Stream 完全对齐。预注册阶段注册 primary + COMP_1/2/3（最多 4 条），`replay_range_node_default` 按需注册 `UPDATE` 流（第 5 条）。此前 `kMaxActiveStreams = 3` 的配置曾导致含 RangeOp 的图在 `get_or_register(primary)`（当 primary ≠ COMP_1/2/3 时）直接触发崩溃——已修正。

### 1.4 设备上下文中的多流支持

[`DeviceContext`](file:///r:/renaissance/include/renaissance/backend/device_context.h) 为每个 `StreamKind` 维护：

- 独立的 `cudaStream_t`
- 独立的 `cublasHandle_t`（per-stream cuBLAS 句柄）
- 独立的 `cudnnHandle_t`（per-stream cuDNN 句柄，GAP 使用）
- 独立的 `workspace`（per-stream 临时缓冲区，避免竞态）

```cpp
// 算子获取 per-stream 资源的典型模式
cudaStream_t s = ctx.stream(StreamKind::COMP_1);
cublasHandle_t h = ctx.cublas_handle(StreamKind::COMP_1);
void* ws = ctx.workspace(StreamKind::COMP_1);
```

---

## 二、CUDA Graph 捕获机制

三流架构的真正威力在与 CUDA Graph 结合时释放。Graph 将多流拓扑固化为静态 GPU 侧执行图，消除所有 CPU 侧 launch 和同步开销。

### 2.1 捕获流程

[`capture_cuda.cpp`](file:///r:/renaissance/src/graph/capture_cuda.cpp) 的完整捕获流程：

```
阶段 1: 准备（BeginCapture 之前）
  1a. 确定 primary stream（取第一个算子的代表流）
  1b. 预注册 3 个 compute 流（事件在 capture 外创建，合规）
      └─ `kMaxActiveStreams = 5`，支持后续 UPDATE/TRANS 流按需注册
  1c. 销毁 warmup 期间的旧事件，重建新事件（确保事件无历史状态）

阶段 2: 开始捕获
  2a. cudaStreamBeginCapture(primary, cudaStreamCaptureModeThreadLocal)
  2b. Record dummy event on primary → 所有 secondary 流 wait 它
      └─ 将 secondary 流引入 CUDA Graph 捕获上下文

阶段 3: 逐算子的 replay 循环
  for each node:
    3a. insert_cross_op_barrier(prev_node, cur_node)
        └─ 下一个节点的代表流 wait 上一个节点的输出流
    3b. 调用算子的 launch_cuda 函数（算子自治处理内部多流同步）
    3c. cudaGetLastError 检查

阶段 4: 收束
  4a. finalize_cross_stream_barrier
      └─ primary stream wait 所有有 pending work 的 secondary 流
  4b. cudaStreamEndCapture → 获取 cudaGraph_t
  4c. cudaGraphInstantiate → 获取 cudaGraphExec_t（可执行句柄）
```

### 2.2 cross-op barrier（精确依赖 chain）

[`insert_cross_op_barrier`](file:///r:/renaissance/src/graph/capture_multi_stream.cpp#L69-L85) 采用**精确依赖传递**（而非星型广播）：

```cpp
void insert_cross_op_barrier(const GraphNode&, const GraphNode& next_node,
                              MultiStreamCaptureState& state, const DeviceContext& ctx) {
    int out_idx = state.output_stream_idx;
    if (out_idx < 0) return;

    if (next_node.kind == GraphNode::Kind::COMPUTE) {
        StreamKind target_sk = get_op_default_stream(next_node.compute_op);
        cudaStream_t target_s = static_cast<cudaStream_t>(ctx.stream(target_sk));
        int target_idx = state.find_stream_index(target_s);
        if (target_idx >= 0 && target_idx != out_idx) {
            cudaStreamWaitEvent(target_s,
                state.streams[out_idx].last_done_event, 0);
        }
    }
}
```

**核心逻辑**：只让"下一个算子的代表流"等待"上一个算子的输出流"。不广播，不通知无关的流。这保证了依赖图是单向 DAG 链，不会产生循环。

### 2.3 finalize barrier（总收束）

[`finalize_cross_stream_barrier`](file:///r:/renaissance/src/graph/capture_multi_stream.cpp#L87-L94) 在 Graph 末尾执行，保证 primary stream 在 EndCapture 前看到所有计算流的结果：

```cpp
void finalize_cross_stream_barrier(MultiStreamCaptureState& state) {
    for (int i = 0; i < state.num_active; ++i) {
        if (state.streams[i].stream == state.primary_stream) continue;
        if (!state.streams[i].has_pending_work) continue;
        cudaStreamWaitEvent(state.primary_stream,
                           state.streams[i].last_done_event, 0);
    }
}
```

### 2.4 事件拓扑的全局一致性

经过上述 precise INSERT + 算子自治 wait + finalize 收束，CUDA Graph 中的事件拓扑始终是**单向 DAG**：

```
GAP_BWD(COMP_2) → FC_BWD:

COMP_2: [GAP kernel] ─ev_gap→ [dB kernel] ─ev_db→ wait(ev_dw) → wait(ev_dx) (finalize)
COMP_1: ─wait(ev_gap)→ [dW GEMM] ─ev_dw→
COMP_3: ─wait(ev_gap)→ wait(ev_dw) → [dX GEMM] ─ev_dx→

所有边方向一致，无回路。验证通过。
```

---

## 三、算子开发指南：在三流架构下写算子

本章是本文档的**核心实用价值**。后面的开发者读此章即可掌握所有必需的接口与规范。

### 3.1 你只需要实现一个函数

每个算子需要注册一个 launch 函数：

```cpp
using LaunchCudaFn = void(*)(const GraphNode& node,
                               const MemoryPlan& mp,
                               const DeviceContext& ctx,
                               MultiStreamCaptureState& state);
```

在你的算子源文件中：

```cpp
void launch_my_op_cuda(const GraphNode& node, const MemoryPlan& mp,
                       const DeviceContext& ctx,
                       MultiStreamCaptureState& state) {
    // 你的实现
}

// 注册（通常在文件末尾的静态初始化中）
REGISTER_COMPUTE_OP(my_op, launch_my_op_cuda);
```

### 3.2 算子分类与模板

#### 类型 A：单流算子（绝大多数算子）

如果算子的工作天然属于一个流（如 GAP 全在 COMP_2、BN 全在 COMP_2、ReLU 全在 COMP_3），只需：

```cpp
void launch_my_single_stream_op(...) {
    // 1. 获取流
    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));

    // 2. 注册流并设置状态
    int si = state.get_or_register(s);
    state.streams[si].has_pending_work = true;   // ← 先标记
    state.output_stream_idx = si;                // ← 设置输出流

    // 3. 实际计算（cuDNN / cuBLAS / custom kernel）
    my_compute_kernel<<<..., s>>>(...);

    // 4. 记录完成事件
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

#### 类型 B：多流并发算子（如 FC BWD）

如果算子内部需要多流并发，需要自己管理内部事件同步。以 FC_BWD 为典范：

```cpp
void launch_fc_amp_bwd_cuda(...) {
    // 1. 获取三流
    cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));  // dW 计算
    cudaStream_t s_db = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));  // dB 计算
    cudaStream_t s_dx = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));  // dX 计算
    int i_dw = state.get_or_register(s_dw);
    int i_db = state.get_or_register(s_db);
    int i_dx = state.get_or_register(s_dx);

    // 2. 【关键】算子自治等待上游输出
    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s_dw) cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
        if (prev_s != s_db) cudaStreamWaitEvent(s_db, state.streams[out_idx].last_done_event, 0);
        if (prev_s != s_dx) cudaStreamWaitEvent(s_dx, state.streams[out_idx].last_done_event, 0);
    }

    // 3. 并发工作：dB 在 COMP_2
    launch_db_kernel(..., s_db);
    cudaEventRecord(state.streams[i_db].last_done_event, s_db);
    state.streams[i_db].has_pending_work = true;

    // 4. 顺序工作：dW 在 COMP_1
    cublasGemmEx(h_dw, ...);
    cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    state.streams[i_dw].has_pending_work = true;

    // 5. 跨流同步：dX 在 COMP_3，等待 dW 完成
    cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);
    cublasGemmEx(h_dx, ...);
    cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
    state.streams[i_dx].has_pending_work = true;

    // 6. 设置输出流（dX 是最终输出）
    state.output_stream_idx = i_dx;  // ← COMP_3
}
```

### 3.3 必须遵守的规则

| # | 规则 | 原因 |
|---|------|------|
| 1 | **必须先 `has_pending_work = true`，再做计算** | 该标志仅影响 `finalize_cross_stream_barrier` 收束。注意：precise INSERT 不再使用 `has_pending_work` 做决策 |
| 2 | **必须设置 `output_stream_idx`** | 框架需要知道下游该等哪条流 |
| 3 | **不要调用 `alloc_temp_event`** | 在 capture 中创建事件是 CUDA Graph 禁止的操作 |
| 4 | **使用 per-stream handle** | `ctx.cublas_handle(sk)` 而非默认 handle，避免竞态 |
| 5 | **算子自治 wait 上游** | 多流算子必须在所有使用的流上 wait `state.streams[out_idx].last_done_event` |
| 6 | **不要手动调用 `cudaEventCreateWithFlags`** | 事件已在预注册阶段创建好 |
| 7 | **集合类操作（H2D/D2H）用 TRANS/UPDATE 流** | 不与 COMP_*/COMP_*/COMP_3 的计算流混淆 |

### 3.4 流策略声明

在 [`op_stream_policy.cpp`](file:///r:/renaissance/src/backend/op_stream_policy.cpp) 中为新算子添加 case：

```cpp
case ComputeOp::MY_OP_BWD:
    return StreamKind::COMP_3;  // 如果 dX 是最终输出
```

声明逻辑：
- FWD 输出在 COMP_1 → 返回 COMP_1
- BWD dX 输出在 COMP_3 → 返回 COMP_3
- 池化类 → 返回 COMP_2
- 逐元素类 → 返回 COMP_3

---

## 四、实测数据

> 数据来源：[`docs/MULTI_RESULTS.md`](file:///r:/renaissance/docs/MULTI_RESULTS.md)（2026-05-19, A100）  
> 测试配置：GAP [512,7,7,2048]，FC In=2048, Out=1000，AMP 模式

### 4.1 三流架构数据

| 测试 | FWD (μs) | BWD (μs) | 总计 (μs) |
|------|:---:|:---:|:---:|
| GAP 单独 | 87.97 | 407.55 | 495.52 |
| FC 单独 | 50.64 | 56.62 | 107.26 |
| **算术和** | **138.61** | **464.17** | **602.78** |
| **GAP+FC 联合** | **131.18** | **456.05** | **587.23** |
| **联合 vs 算术和** | **−7.43 (−5.4%)** | **−8.12 (−1.7%)** | **−15.55 (−2.6%)** |

### 4.2 单流 vs 三流

| 测试 | 单流 (μs) | 三流 (μs) | 节省 |
|------|:---:|:---:|:---:|
| **GAP+FC 总计** | **614.57** | **587.23** | **−27.34 (−4.5%)** |

### 4.3 数学正确性

流分配只影响执行调度，不影响计算语义。所有正确性测试通过：

| 测试 | 精度 | 判定 |
|------|------|:---:|
| `test_fc_fwd_bwd` | FP16 + FP32（CPU + GPU 全覆盖） | PASS |
| `test_gap` | FP32 | PASS |
| `test_gap --amp` | FP16 | PASS（MaxMSE 2.19E-11） |

---

## 五、FAQ：回应常见质疑

### Q1："又不是每个算子都有内部并发，三流架构到底有什么用？"

**答**：三条流各司其职，即使某算子只用一条流（如 GAP 独占 COMP_2），三流基础设施的价值体现在：

1. **跨算子执行不互相阻塞**：GAP(COMP_2) 和 FC(COMP_1) 在 Graph 中通过事件接力，不影响各流的资源分配
2. **为后续扩展提供基础**：后续开发 Conv BWD 等多流算子时，不需要额外搭建流基础设施
3. **NCCL 通信重叠和优化器融合都需要独立流**：三流架构为这些系统级优化提供了插入点

### Q2："单流不是更简单吗？为什么要引入多流复杂度？"

**答**：实际数据推翻了这个假设。单流架构在同一 Graph 内比三流慢 **4.5%（27.34 μs）**。CUDA Graph 捕获机制封装了绝大部分多流复杂性，算子开发者只需遵循本文第三章的模板。

### Q3："联合执行不应该等于 GAP + FC 叠加吗？为什么会更快？"

**答**：分别跑两份 Graph 需要两次 `cudaGraphLaunch` + 两次 `cudaStreamSynchronize`。联合 Graph 只需一次 launch + 一次 sync。省掉的开销（约 13-15 μs）就是 net gain。

### Q4："cbr_fwd_fp16.cpp 用链式事件同步，你们呢？"

**答**：cbr_fwd 是手写融合模块，可以手动编排事件链。DTensor 框架是通用图计算系统，不知道全局拓扑。我们采用的 precise INSERT（只让 need-to-wait 的流 wait）+ 算子自治同步（多流算子在内部管理自己的跨流依赖）+ finalize 收束，三者共同构成了与 cbr 等效但更通用的单向 DAG 同步拓扑。经严格依赖分析，**无循环、无冗余**。

### Q5："星型广播 barrier 是否会产生流间循环依赖？"

**答**：原星型 `insert_cross_op_barrier`（所有已活跃流 wait 输出流）与 `finalize`（primary 等所有非 primary）会形成双向依赖边，在特定拓扑下产生循环。当前的 precise INSERT（只让下游代表流 wait 上游输出流）已消除此问题——依赖图退化为单向 DAG 链，算子内部自治同步走同向边，经事件时间线分析，无任何回路。`has_pending_work` 也不再参与 INSERT 决策。

---

## 六、技术演进路线

三流架构不仅是当前最优解，更是未来系统级优化的基础设施：

```
当前：COMP_1 ∥ COMP_2 ∥ COMP_3（计算内部并发）
   ↓
短期：+ STREAM_COMM（NCCL AllReduce 重叠，遮掩 60-80% 通信延迟）
   ↓
中期：+ STREAM_UPDATE（LARS 优化器融合，计算-通信-更新三级流水线）
   ↓
长期：自适应 per-layer 流策略（基于 profiler 反馈的动态分配）
```

每一步都需要三流基础设施提供的独立流句柄、per-stream 资源隔离和事件同步机制。**回退单流将 pre-close 所有系统级优化的路径。**

---

## 七、代码索引

| 文件 | 职责 | 阅读顺序 |
|------|------|:---:|
| `include/renaissance/graph/capture_multi_stream.h` | 多流状态结构 + barrier 声明 | 1 |
| `src/graph/capture_multi_stream.cpp` | INSERT / finalize 实现 + 流注册 | 2 |
| `src/graph/capture_cuda.cpp` | Graph 捕获主循环 | 3 |
| `src/backend/op_stream_policy.cpp` | 静态流映射 | 4 |
| `src/backend/ops/dtensor/fc_op.cpp` | FC 三流 BWD 典范实现 | 5 |
| `src/backend/ops/dtensor/gap_op.cpp` | GAP 单流典范实现 | 6 |
| `include/renaissance/backend/device_context.h` | per-stream 资源接口 | 7 |

---

## 八、最终裁决

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   三流架构 = renAIssance DTensor 算子系统的唯一生产基线    │
│                                                         │
│   证据：                                                  │
│   ✅ 联合执行比单流快 4.5%（27 μs）                        │
│   ✅ CUDA Graph 联合 vs 个体和快 2.6%（15 μs）            │
│   ✅ 依赖图拓扑单向 DAG，零循环                             │
│   ✅ 数学正确性验证通过                                     │
│   ✅ 为 NCCL 通信重叠 / 优化器融合提供基础设施              │
│                                                         │
│   禁则：                                                  │
│   ❌ 回退单流（−4.5% 性能倒退）                            │
│   ❌ 移除事件同步（破坏正确性）                             │
│   ❌ 共享 Workspace（引入竞态）                            │
│   ❌ 在 capture 中创建事件（CUDA Graph 禁止）              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

**附录 A：参考文档**

| 文档 | 内容 |
|------|------|
| `docs/MULTI_RESULTS.md` | 最终 A100 测试数据 |
| `docs/MULTI_REV.md` | 本文档审核意见 |
| `FIX_MULTI.md` / `FIX_MULTI2.md` / `FIX_MULTI3.md` | 问题定位与修复方案演进（根目录） |
| `FIX_MULTI_FINAL.md` | 最终修复方案（根目录，4 处修改，已应用） |
| `cbr_fwd_fp16.cpp` | 旧版 CBR 手写链式事件参考（根目录） |

**附录 B：文档修订历史**

| 版本 | 日期 | 修订内容 |
|------|------|------|
| 1.0 | 2026-05-19 | 初版，基于未修正的 In=1024 数据 |
| 2.0 | 2026-05-19 | 修正 FC 维度对齐，发现联合更快 |
| **3.0** | **2026-05-19** | **终版：整合三方共识, 添加算子开发指南, precise INSERT 分析, FAQ** |

---

**文档完**

> 三流架构已被严格验证。后续工作：在此架构上开发 Conv BWD 多流实现，引入 NCCL 通信流进行通信-计算重叠实验。
