# 图捕获与运行机制 —— 最终方案

> 基于 ULTIMATE_DESIGN 全系列 13 份文档、当前代码库完整检查、proposal1/2/3 的批判性继承、
> Legacy 生产验证经验（src_legacy/task/task_base.cpp、include_legacy/renaissance/backend/device_context.h）、
> 以及 OOOPS_FINAL.md / REGION_FINAL.md 等设计文档的综合分析。
>
> 本方案为正式通过的设计规范。实施时必须以此为唯一权威。

---

## 一、现状精准诊断

### 1.1 已完成的资产（无需大改）

| 模块 | 状态 | 位置 |
|------|------|------|
| **Compiler** | 五阶段编译完整实现 | `include/renaissance/graph/compiler.h` |
| **MemoryPlan** | 65 Region + DTensor + RangeOpRange 预计算 | `include/renaissance/graph/memory_plan.h` |
| **ComputationGraph** | 17 GraphId 桶 + GraphNode COMPUTE/RANGE union | `include/renaissance/graph/computation_graph.h` |
| **GraphAtlas** | 6x17 Slot 表 + 三阶段填表接口 | `include/renaissance/graph/graph_atlas.h` |
| **ShapeId** | (N,H,W,C) 四元组 + kShapeInvariant | `include/renaissance/graph/shape_id.h` |
| **CompileSpec** | 6 变体定义 | `include/renaissance/graph/compile_spec.h` |
| **ArenaKeeper** | 多卡内存池单例，`ptr_at(rank, offset)` 直接解析 | `include/renaissance/backend/memory_arena.h` |
| **GlobalRegistry** | `using_gpu()` / `gpu_ids()` 运行时后端判断 | `include/renaissance/core/global_registry.h` |
| **Legacy DeviceContext** | 5 流 + cuDNN handle + workspace + ptr_table（生产验证） | `include_legacy/renaissance/backend/device_context.h` |

### 1.2 当前代码的致命缺陷

| 问题 | 位置 | 严重 |
|------|------|:--:|
| `#ifdef TR_USE_CUDA` 切割类定义 — CUDA 编译 + `use_cpu()` 时无 CPU 存储空间 | `captured_graph.h:L32-40, L149-153` | **P0** |
| `CpuOp` 用 `std::function<void()> execute` — ~32 bytes + 堆分配 + 虚调用 | `captured_graph.h:L37-39` | **P1** |
| `capture()` / `launch()` 为 dry-run placeholder — 完全未实现 | `captured_graph.cpp:L30-45, L73-83` | **P0** |
| `PreCaptureResult` 无 per-rank 维度 — 多卡场景下无法承载 per-GPU exec | `captured_graph.h:L169-176` | **P0** |
| `capture()` 接受虚构的 `Device*` 参数 — 这个类完全未实现 | `captured_graph.h:L89` | **P0** |
| 无 GraphExecutor 类 — 运行时路由、A/B 双缓冲、per-rank 调度全部缺失 | — | **P0** |
| 无多流捕获 — 用户反复强调的硬性需求 | — | **P0** |

### 1.3 Legacy 已验证但当前完全缺失的能力

1. 真实 CUDA Graph 捕获：`cudaStreamBeginCapture` -> replay -> `cudaStreamEndCapture` -> `cudaGraphInstantiate`
2. RAII CaptureGuard：异常时强制结束 capture 态，防止流永久锁死
3. "GPU 0 串行预热 -> Rank 0 串行捕获 -> Rank 1~N-1 并行捕获" 三步串行化（避免 cuDNN 全局 cache 竞争）
4. Per-rank `cudaGraphExec_t` 存储
5. `error_log[2048]` 传入 `cudaGraphInstantiate` 用于诊断
6. `std::exception_ptr` 跨线程异常聚合
7. A/B 双缓冲训练循环
8. CPU/CUDA 统一接口下的双分支执行
9. 多流捕获：CBR 拆三流，跨流 Event 依赖链

### 1.4 对 proposal1/2/3 的评述

| 维度 | proposal1 | proposal2 | proposal3 | **P_FINAL** |
|------|-----------|-----------|-----------|-------------|
| **多流捕获** | 硬性需求，Full Design（MultiStreamCaptureState） | 硬性需求，CBR 三流详尽 | 硬性需求，MultiStreamCaptureContext | **综合三者最佳，精确实施** |
| **Device 抽象** | 基于 legacy DeviceContext 重构适配 | 直接复用 legacy 具体类 | 纯虚 `DeviceContext` 基类 | **基于 legacy 重构适配新版 MemoryPlan** |
| **per-rank 存储** | CapturedGraph 内部集成 | 同左 | PreCaptureResult 外部数组 | **CapturedGraph 内部集成**（launch(rank,stream) 单点分发） |
| **遍历接口** | `cg.nodes(gid)` | 同左 | `cg.nodes(gid)` | **严格使用 `cg.nodes(gid)`** |
| **BN Stats** | Compiler 入图 | Compiler 入图 | 未明确 | **Compiler 入图（RANGE_BN_STATS_COPY 作为 GraphNode）** |
| **LR 更新** | H2D + Compiler 标量 ID | H2D + Compiler 标量 ID | 未提及 | **H2D + OptimizerScalarIds** |
| **launch 分发** | `if (is_cuda_)` 直接分支 | `if (is_cuda_)` 直接分支 | 未明确 | **`if (is_cuda_)` 直接分支** |
| **GraphId->Stream** | 不绑定！IR 不应含后端信息 | `gid_to_stream_kind()` 自由函数 | 未明确 | **`gid_to_stream_kind()` 自由函数（辅助），IR 不含 Stream** |
| **CpuOp** | `void(*fn)(void*)+ctx`（16 bytes） | 同左 | 同左 | **继承，零虚调用** |
| **事件机制** | MultiStreamCaptureState 追踪器 | CBR 三流事件链 | MultiStreamCaptureContext | **综合：追踪器 + 算子内事件链 + 算子间屏障** |
| **PreCaptureResult** | 含统计字段 | 删除统计字段 | 含 per_rank_execs 数组 | **保留必要字段** |

**proposal1 的贡献**：四条铁的定律、MultiStreamCaptureState 追踪器、算子间跨流屏障的完整设计。
**proposal2 的贡献**：cudnn handle 多流独立、CBR 内部事件链的精确实现、`gid_to_stream_kind()` 自由函数、对 `rank` 参数不应传入 `capture()` 的批判。
**proposal3 的贡献**：CaptureGuard 强制要求、渐进式实施策略思路。

本方案继承上述全部优点，并在此基础上提供更精确的实施细节。

---

## 二、核心认知框架

### 2.1 四条铁的定律

**定律一：ComputationGraph 是纯算子拓扑 IR，不含任何后端执行信息。**

`GraphNode` 中不存储 stream、event、device handle。哪个流上执行、是否需要跨流事件——这是 capture 引擎在 Phase B 需要动态决策的，不是 Compiler 在 Phase A 需要提前指定的。

**定律二：同一 CUDA Graph 内可以有 COMP_1/COMP_2/COMP_3，但绝不能混入 TRANS。**

计算流（COMP_i）和传输流（TRANS）**绝对不会**出现在同一 CUDA Graph 中。它们会被捕获为不同的 CUDA Graph，然后通过并行跑两张独立的 CUDA Graph（Level 2 并行）来实现计算与通信的重叠。

**定律三：多流捕获的跨流依赖通过 cudaEvent 动态插入，不污染 IR。**

算子内部事件依赖链由算子 replay 函数维护。算子间跨流屏障由 capture 循环在算子边界自动插入。不能因为多算子的串联捕获就破坏每个算子内部的事件依赖顺序。

**定律四：CapturedGraph 是平台相关的可执行实体。**

与平台无关的 ComputationGraph/MemoryPlan 不同，CapturedGraph 完成了指针绑定、预热、捕获、实例化，就等运行。`is_cuda_` 给出肯定答案。去重键 `{cg*, gid, shape_id}` 决定"是否同一张图"，`per_rank_execs_[N]` 决定"这张图在每张卡上用哪个 exec handle"。

### 2.2 三阶段生命周期

```
Phase A（编译期，MLPerf 不计时）
═══════════════════════════════════════════════════════════════════
  Compiler::compile(arch, base_spec, plan_config, initializer, variant_specs)
    ├─ A1: derive_all_shapes          -> all_shapes[6][layer][tensor]
    ├─ A2: compute_max_slot_bytes     -> max_slots（跨变体取 max）
    ├─ A3: create_memory_plans        -> 6x MemoryPlan（独立对象，offset 一致）
    │     └─ 产出 OptimizerScalarIds{lr, beta, tc, wd, eps}
    ├─ A4: build_computation_graph    -> train_cg + infer_cg（17 GraphId 桶）
    │     ├─ build_forward            -> TRANSFER_A/B, FIRST_FWD_A/B, DEEP_FWD_BWD
    │     ├─ build_backward           -> FIRST_BWD
    │     ├─ build_inference          -> INF_MAIN_A/B, INF_EMA_A/B
    │     └─ build_auxiliary_graphs   -> ZERO_GRAD, CAST_AND_CHECK,
    │           FIRST_COMM, DEEP_COMM, STATS_COMM, OPTIMIZER, EMA_UPDATE
    │           （含 RANGE_BN_STATS_COPY 等 RangeOp 节点）
    └─ A5: share_or_clone             -> variants[0~5] 指向 cg/mp
           │
           ▼
  GraphAtlas::build(Result, input_shapes[6])
    ├─ 训练变体 0~3 -> 填训练图（shape 无关图填 kShapeInvariant）
    ├─ 验证变体 4~5 -> 填推理图
    └─ 交叉槽位 -> cg=nullptr
           │
           ▼
  GraphAtlas（Slot{cg, mp, shape_id} 已填，captured_idx = -1）

Phase B（预热捕获期，MLPerf 不计时）
═══════════════════════════════════════════════════════════════════
  pre_capture(GraphAtlas, device_contexts[0~N-1])
    ├─ B1: 去重（遍历 6x17 Slot）
    │     Key{cg*, gid, shape_id} -> unordered_map -> K 张逻辑图（K~27~31）
    │     graphs_.resize(K)，每张 reserve_ranks(N)
    │
    ├─ B2: cuDNN 预热（仅 CUDA，GPU 0 单线程串行）
    │     cudaSetDevice(gpu_ids[0])
    │     遍历全部 ComputeNode 中的 cuDNN 算子
    │     build_plans(HEURISTICS_CHOICE) + execute on GPU 0
    │     cudaDeviceSynchronize()
    │
    ├─ B3: 捕获
    │     ├─ Rank 0: 主线程串行 capture_all_for_rank(0)
    │     └─ Rank 1~N-1: 子线程并行 capture_all_for_rank(r)
    │         -> 将返回的 exec 合并入 rank 0 创建的 CapturedGraph
    │         异常聚合: std::exception_ptr[N]
    │
    └─ B4: 捕获后 warmup（仅 Rank 0）
          每图 cudaGraphLaunch + cudaStreamSynchronize
          暴露异步运行时错误（illegal memory access 等）
           │
           ▼
  PreCaptureResult { graphs[K], atlas（captured_idx 已填）, stats }

Phase C（运行期，MLPerf 计时开始）
═══════════════════════════════════════════════════════════════════
  每 rank 一个 GraphExecutor 线程
    ├─ launch(GraphId gid)
    │     idx = atlas_.index(variant, gid)      // O(1) 数组双下标
    │     graphs_[idx].launch(rank_, stream)     // is_cuda_ 分支 + launch
    │
    ├─ launch_dual(GraphId gid1, GraphId gid2)
    │     graphs_[idx1].launch(rank_, stream1)   // 非阻塞
    │     graphs_[idx2].launch(rank_, stream2)   // 非阻塞
    │     GPU 硬件自动并行
    │
    └─ sync_all()
          CUDA: cudaDeviceSynchronize()
          CPU:  空操作
```

---

## 三、CapturedGraph —— 双后端可执行图

### 3.1 设计原则

1. **`#ifdef TR_USE_CUDA` 只进函数体，不进类定义** — `TR_USE_CUDA` 控制 CUDA API 调用是否存在，不控制数据布局。`void*` 类型的 cuda 数据和 `vector<CpuOp>` 始终同时存在。
2. **后端运行时确定，运行期不变** — `is_cuda_` 在 capture 时根据 DeviceContext 一次性设定。CPU 8GB 内存与 GPU 32GB 显存的配置完全不同，动态切换后端是伪需求。
3. **per_rank_execs_ 封装在 CapturedGraph 内部** — `launch(rank, stream)` 一个调用完成所有后端分发，消除外部两层索引。
4. **CpuOp 用裸函数指针** — `void(*fn)(void*ctx) + void*ctx`，16 bytes 栈分配，零虚调用，替代 `std::function<void()>`。
5. **launch 用直接 `if(is_cuda_)` 分支** — 后端从不切换，分支预测 100% 命中，~1ns 开销。比函数指针间接调用更简洁。

### 3.2 类定义

```cpp
// ===== include/renaissance/graph/captured_graph.h =====
#pragma once

#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/shape_id.h"
#include "renaissance/graph/graph_atlas.h"
#include "renaissance/graph/memory_plan.h"
#include <vector>
#include <cstdint>

namespace tr {

class DeviceContext;

using NativeGraph = void*;

struct CpuOp {
    void (*fn)(void* ctx) = nullptr;
    void* ctx = nullptr;
};

class CapturedGraph {
public:
    CapturedGraph() = default;
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;
    CapturedGraph(CapturedGraph&&) noexcept = default;
    CapturedGraph& operator=(CapturedGraph&&) noexcept = default;
    ~CapturedGraph();

    // ===== 去重键 =====
    struct Key {
        const ComputationGraph* cg = nullptr;
        GraphId                 gid = GraphId::TRANSFER_A;
        ShapeId                 shape{};
        bool operator==(const Key& o) const noexcept {
            return cg == o.cg && gid == o.gid && shape == o.shape;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            size_t h = std::hash<const ComputationGraph*>{}(k.cg);
            h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.gid))
                  + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= ShapeIdHash{}(k.shape)
                  + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    static CapturedGraph capture(const ComputationGraph& cg,
                                  const MemoryPlan& mp,
                                  GraphId gid,
                                  ShapeId shape_id,
                                  const DeviceContext& ctx);

    void launch(int rank, void* stream) const;

    [[nodiscard]] bool is_cuda() const noexcept { return is_cuda_; }
    [[nodiscard]] const Key& key() const noexcept { return key_; }
    [[nodiscard]] size_t num_ranks() const noexcept { return per_rank_execs_.size(); }

    std::string debug_dump() const;

    void reserve_ranks(size_t num_ranks);
    void set_rank_exec(int rank, NativeGraph exec);
    [[nodiscard]] const std::vector<NativeGraph>& per_rank_execs() const noexcept {
        return per_rank_execs_;
    }

private:
    Key key_;
    bool is_cuda_ = false;
    std::vector<NativeGraph> per_rank_execs_;
    std::vector<CpuOp> cpu_ops_;
};

// ===== PreCaptureResult =====
struct PreCaptureResult {
    std::vector<CapturedGraph> graphs;
    GraphAtlas atlas;

    size_t total_slots = 0;
    size_t captured     = 0;
    size_t reused       = 0;
};

PreCaptureResult pre_capture(const GraphAtlas& compile_atlas,
                              const std::vector<DeviceContext*>& contexts);

} // namespace tr
```

### 3.3 capture() —— 单 rank 捕获入口

```cpp
CapturedGraph CapturedGraph::capture(const ComputationGraph& cg,
                                      const MemoryPlan& mp,
                                      GraphId gid,
                                      ShapeId shape_id,
                                      const DeviceContext& ctx) {
    CapturedGraph result;
    result.key_ = Key{&cg, gid, shape_id};

    if (ctx.is_gpu()) {
        result.is_cuda_ = true;
        result.per_rank_execs_.resize(1, nullptr);
#ifdef TR_USE_CUDA
        capture_cuda(cg, mp, gid, ctx, result);
#endif
    } else {
        result.is_cuda_ = false;
        capture_cpu(cg, mp, gid, ctx, result.cpu_ops_);
    }

    return result;
}
```

### 3.4 launch()

```cpp
void CapturedGraph::launch(int rank, void* stream) const {
    if (is_cuda_) {
#ifdef TR_USE_CUDA
        TR_DEBUG_CHECK(rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size(),
                       IndexError, "CapturedGraph::launch rank out of range");
        NativeGraph exec = per_rank_execs_[rank];
        if (exec) {
            cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec),
                           static_cast<cudaStream_t>(stream));
        }
#endif
    } else {
        (void)rank;
        for (const auto& op : cpu_ops_) {
            op.fn(op.ctx);
        }
    }
}

CapturedGraph::~CapturedGraph() {
    if (is_cuda_) {
#ifdef TR_USE_CUDA
        for (auto exec : per_rank_execs_) {
            if (exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec));
        }
#endif
    }
}
```

### 3.5 关键决策解释

**为什么不用 `CpuOp` + `void* cuda_graph_` 分别存储？**

因为多卡场景下，CUDA 路径的 `cudaGraphExec_t` 是 **per-rank** 的（每张 GPU 有独立句柄），而 CPU 路径的 `cpu_ops_` 是所有 rank 共享的（CPU 没有 per-rank 概念）。`per_rank_execs_` 是一个 `vector<NativeGraph>`，`rank` 作为下标直接索引——这比 `std::unordered_map<rank, exec>` 快得多，也更简单。

**为什么 launch 用 `if (is_cuda_)` 而非函数指针？**

`is_cuda_` 在 capture 时确定，此后至程序结束不变。CPU 的分支预测器会将此分支预测为 always-taken（或 always-not-taken），开销 ~1ns。函数指针间接调用（`launch_fn_(this, rank, stream)`）约 2ns + 需要统一签名，优势不明显。

---

## 四、多流捕获 —— 硬性需求，完整设计

### 4.1 核心认知

**不在 ComputationGraph 中预设 StreamKind。** `GraphId` 是平台无关的 IR 概念。一个 `DEEP_FWD_BWD` 中可能包含 CBR1 的 Conv+GenStats（COMP_1）+ BN Finalize（COMP_2）+ BN Apply+ReLU（COMP_3），然后是 CBR2 的同样三流模式。流的选择和事件依赖的插入完全由 capture 引擎在 Phase B 动态完成。

**哪些流可以出现在同一个 CUDA Graph 内？**
- `COMP_1`、`COMP_2`、`COMP_3` — **可以**同图（算子的多流实现）
- `TRANS` — **绝不**与计算流同图（传输图与计算图天生分离）
- `UPDATE` — 可以与计算流同图，取决于该 GraphId 桶

**框架保证**：计算流和传输流绝对不存在被捕获到同一个 CUDA Graph 的情况——它们只会被捕获成不同的 CUDA Graph，然后通过 Level 2 并行跑两张独立的 CUDA Graph 来实现计算通信重叠。

### 4.2 MultiStreamCaptureState —— 多流追踪器

```cpp
struct MultiStreamCaptureState {
    cudaStream_t primary_stream = nullptr;

    struct PerStreamState {
        cudaStream_t stream = nullptr;
        cudaEvent_t  last_done_event = nullptr;
        bool         has_pending_work = false;
    };
    static constexpr int kMaxActiveStreams = 3;
    PerStreamState streams[kMaxActiveStreams];
    int num_active = 0;

    int output_stream_idx = -1;

    std::vector<cudaEvent_t> temp_events;

    int get_or_register(cudaStream_t s) {
        for (int i = 0; i < num_active; ++i) {
            if (streams[i].stream == s) return i;
        }
        int idx = num_active++;
        streams[idx].stream = s;
        cudaEventCreate(&streams[idx].last_done_event);
        return idx;
    }

    cudaEvent_t alloc_temp_event() {
        cudaEvent_t ev = nullptr;
        cudaEventCreate(&ev);
        temp_events.push_back(ev);
        return ev;
    }

    void cleanup_all_events() {
        for (auto ev : temp_events) cudaEventDestroy(ev);
        temp_events.clear();
        for (int i = 0; i < num_active; ++i) {
            if (streams[i].last_done_event) {
                cudaEventDestroy(streams[i].last_done_event);
            }
        }
    }
};
```

### 4.3 capture_cuda() —— 完整的 CUDA 捕获

```cpp
void CapturedGraph::capture_cuda(const ComputationGraph& cg,
                                  const MemoryPlan& mp,
                                  GraphId gid,
                                  const DeviceContext& ctx,
                                  CapturedGraph& result) {
#ifdef TR_USE_CUDA
    cudaStream_t primary_stream = select_primary_capture_stream(gid, ctx);

    struct CaptureGuard {
        cudaStream_t stream = nullptr;
        bool committed = false;
        ~CaptureGuard() {
            if (!committed && stream) {
                cudaGraph_t dummy = nullptr;
                cudaStreamEndCapture(stream, &dummy);
                if (dummy) cudaGraphDestroy(dummy);
            }
        }
    } guard{primary_stream, false};

    cudaStreamBeginCapture(primary_stream, cudaStreamCaptureModeThreadLocal);

    MultiStreamCaptureState state;
    state.primary_stream = primary_stream;

    const auto& nodes = cg.nodes(gid);
    if (!nodes.empty()) {
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];

            if (i > 0) {
                insert_cross_op_barrier(nodes[i-1], node, state, ctx);
            }

            if (node.kind == GraphNode::Kind::COMPUTE) {
                replay_compute_node_multi_stream(node, mp, ctx, state);
            } else {
                replay_range_node_multi_stream(node, mp, ctx, state);
            }
        }

        finalize_cross_stream_barrier(state);
    }

    cudaGraph_t graph_obj = nullptr;
    cudaStreamEndCapture(primary_stream, &graph_obj);
    guard.committed = true;

    state.cleanup_all_events();

    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t error_node = nullptr;
    char error_log[2048] = {};
    cudaError_t err = cudaGraphInstantiate(&exec, graph_obj,
                                           &error_node, error_log, sizeof(error_log));
    if (err != cudaSuccess || !exec) {
        std::string msg = "cudaGraphInstantiate failed: gid=";
        msg += graph_id_to_string(gid);
        msg += " rank=" + std::to_string(ctx.device_id());
        msg += ": " + std::string(error_log);
        TR_DEVICE_ERROR(msg);
    }

    result.per_rank_execs_[0] = static_cast<NativeGraph>(exec);
    cudaGraphDestroy(graph_obj);
#endif
}

cudaStream_t select_primary_capture_stream(GraphId gid, const DeviceContext& ctx) {
    switch (gid) {
        case GraphId::TRANSFER_A:
        case GraphId::TRANSFER_B:
            return ctx.stream(StreamKind::TRANS);
        case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:
        case GraphId::FIRST_COMM:
        case GraphId::DEEP_COMM:
        case GraphId::STATS_COMM:
            return ctx.stream(StreamKind::UPDATE);
        default:
            return ctx.stream(StreamKind::COMP_1);
    }
}

// 运行时 launch 流的映射函数（与 select_primary_capture_stream 共享 switch 逻辑，
// 但语义不同：前者选 BeginCapture 主流，后者选 cudaGraphLaunch 的目标流）
StreamKind gid_to_stream_kind(GraphId gid) noexcept {
    switch (gid) {
        case GraphId::TRANSFER_A:
        case GraphId::TRANSFER_B:
            return StreamKind::TRANS;
        case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:
        case GraphId::FIRST_COMM:
        case GraphId::DEEP_COMM:
        case GraphId::STATS_COMM:
            return StreamKind::UPDATE;
        default:
            return StreamKind::COMP_1;
    }
}
```

### 4.4 算子内多流 Replay（CBR Forward 示例）

```cpp
void replay_cbr_node_in_capture(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t s1 = ctx.stream(StreamKind::COMP_1);
    cudaStream_t s2 = ctx.stream(StreamKind::COMP_2);
    cudaStream_t s3 = ctx.stream(StreamKind::COMP_3);

    cudaEvent_t ev_s1 = state.alloc_temp_event();
    cudaEvent_t ev_s2 = state.alloc_temp_event();

    // S1: Conv + GenStats
    launch_conv_genstats(node, mp, ctx, s1);
    cudaEventRecord(ev_s1, s1);

    // S2: BN Finalize（等 S1 的 sum/sq_sum）
    cudaStreamWaitEvent(s2, ev_s1, 0);
    launch_bn_finalize(node, mp, ctx, s2);
    cudaEventRecord(ev_s2, s2);

    // S3: BN Apply + ReLU（等 S1 的 conv_out + S2 的 eq_scale）
    cudaStreamWaitEvent(s3, ev_s1, 0);
    cudaStreamWaitEvent(s3, ev_s2, 0);
    launch_bn_apply_relu(node, mp, ctx, s3);

    // 更新 state：标记三条流"已有工作"、记录最后完成事件
    int i1 = state.get_or_register(s1);
    int i2 = state.get_or_register(s2);
    int i3 = state.get_or_register(s3);

    cudaEventRecord(state.streams[i1].last_done_event, s1);
    cudaEventRecord(state.streams[i2].last_done_event, s2);
    cudaEventRecord(state.streams[i3].last_done_event, s3);

    state.streams[i1].has_pending_work = true;
    state.streams[i2].has_pending_work = true;
    state.streams[i3].has_pending_work = true;

    state.output_stream_idx = i3;
}
```

### 4.5 算子间跨流屏障

**问题**：CBR1 的 S3 写入 `d_relu_1_out`，CBR2 的 S1 读取 `d_relu_1_out`。两条流不同——CUDA 不自动保证顺序。不显式同步 = 静默数据竞争。

**解决**：capture 循环在开始 replay CBR2 之前，自动插入跨流 barrier。

```cpp
void insert_cross_op_barrier(
    const GraphNode& prev_node,
    const GraphNode& next_node,
    MultiStreamCaptureState& state,
    const DeviceContext& ctx)
{
    int out_idx = state.output_stream_idx;
    if (out_idx < 0) return;

    for (int i = 0; i < state.num_active; ++i) {
        cudaStreamWaitEvent(state.streams[i].stream,
                           state.streams[out_idx].last_done_event, 0);
    }
}

void finalize_cross_stream_barrier(MultiStreamCaptureState& state) {
    for (int i = 0; i < state.num_active; ++i) {
        if (state.streams[i].stream == state.primary_stream) continue;
        if (!state.streams[i].has_pending_work) continue;
        cudaStreamWaitEvent(state.primary_stream,
                           state.streams[i].last_done_event, 0);
    }
}
```

### 4.6 多流捕获规则清单

| 规则 | 说明 |
|------|------|
| 单流 begin/end | `BeginCapture` 和 `EndCapture` 必须在同一条流上 |
| ThreadLocal 模式 | 只捕获当前线程的 kernel，多线程互不干扰 |
| 同线程自动入图 | begin 后同线程内所有流操作全部自动入图 |
| 跨流必须显式同步 | `cudaEventRecord` + `cudaStreamWaitEvent` 是唯一跨流依赖表达方式 |
| 跨组件必须屏障 | N+1 层读 N 层输出时，三条流全部等待 N 层完成事件 |
| Graph 内无阻塞 API | 禁止 `cudaStreamSynchronize` 或 `cudaDeviceSynchronize` |
| 独立 handle + workspace | 每条流独占 cuDNN handle + workspace（`DeviceContext::cudnn_handle(StreamKind)` 返回 per-stream handle） |
| 非阻塞流 | 全部用 `cudaStreamNonBlocking` 创建 |
| 临时事件生命周期 | 通过 `MultiStreamCaptureState::alloc_temp_event()` 管理，在 `cudaStreamEndCapture` 后统一销毁 |

### 4.7 CBR Backward 采用三流顺序（非并行）

> **决策依据**：MultiStreamTest.md 经 6 次独立重复实验、144 个数据点的 A100 实测，
> 结论：CBR 反向传播的三流顺序与三流并行性能等价（全局差异 <1%，噪声级），
> 但三流顺序天然兼容现有 Compiler 的梯度覆盖设计。详见 [MultiStreamTest.md](MultiStreamTest.md)。

**CBR Forward 与 Backward 的拓扑差异**：

| 方向 | 拓扑 | 原因 |
|------|------|------|
| **Forward** | S1[Conv+GenStats] → Fork S2[BN Finalize] ∥ S3[BN Apply+ReLU] → Join | Conv 输出和 BN stats 可天然 fork，BN Finalize 与 BN Apply 共享 S1 产出 |
| **Backward** | S1[BN BWD] → S3[WGrad] → S2[DGrad] → S1 | WGrad 必须先读取 `d_input`，DGrad 才能覆盖写入 `d_grad_input` |

**三流并行的工程代价**：DGrad 和 WGrad 并发执行时，DGrad 写入 `d_grad_input` 的同时 WGrad 正在读取 `d_input`。若梯度覆盖输入（现有 Compiler 设计），则产生**数据竞争**。改为独立显存则需要每个 Conv 层的输入特征图翻倍 + Compiler 内存模型重构。

**三流顺序的天然兼容性**：WGrad 先执行完毕（读取 `d_input`），DGrad 后执行（写入 `d_grad_input` 覆盖 `d_input`），无需任何 Compiler 改动。

**Backward replay 实现示意**：

```cpp
void replay_cbr_bwd_node_in_capture(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s1 = ctx.stream(StreamKind::COMP_1);
    cudaStream_t s3 = ctx.stream(StreamKind::COMP_3);

    cudaEvent_t ev_s1 = state.alloc_temp_event();
    cudaEvent_t ev_s3 = state.alloc_temp_event();

    // S1: BN Backward
    launch_bn_backward(node, mp, ctx, s1);
    cudaEventRecord(ev_s1, s1);

    // S3: WGrad（等 BN BWD 完成，读取 d_input）
    cudaStreamWaitEvent(s3, ev_s1, 0);
    launch_wgrad(node, mp, ctx, s3);
    cudaEventRecord(ev_s3, s3);

    // S2: DGrad（等 WGrad 完成，安全覆盖 d_input）
    cudaStream_t s2 = ctx.stream(StreamKind::COMP_2);
    cudaStreamWaitEvent(s2, ev_s3, 0);
    launch_dgrad(node, mp, ctx, s2);

    int i1 = state.get_or_register(s1);
    int i2 = state.get_or_register(s2);
    int i3 = state.get_or_register(s3);
    cudaEventRecord(state.streams[i1].last_done_event, s1);
    cudaEventRecord(state.streams[i2].last_done_event, s2);
    cudaEventRecord(state.streams[i3].last_done_event, s3);
    state.streams[i1].has_pending_work = true;
    state.streams[i2].has_pending_work = true;
    state.streams[i3].has_pending_work = true;
    state.output_stream_idx = i2;
}
```

`insert_cross_op_barrier` 和 `finalize_cross_stream_barrier`（§4.5）保持不变，三流顺序的跨算子屏障无需任何修改。

**实施要求**：上述 Backward replay 与 §4.4 的 Forward replay 并列为 `capture_cuda()` 中 `replay_compute_node_multi_stream` 的两大分支，由节点的算子类型 dispatch。

---

## 五、DeviceContext —— 从 Legacy 重构

### 5.1 设计原则

**不推翻重建，而是适配重构。** Legacy `DeviceContext`（`include_legacy/renaissance/backend/device_context.h`）已经完整实现了 5 流管理、workspace 管理。新版需要适配：
- `ptr_at(id)` 从查 `ptr_table_` 改为**查 MemoryPlan 的 DTensor + ArenaKeeper::ptr_at(rank, offset)**
- 增加 `current_mp_` 指针，支持运行时 MemoryPlan 切换
- cuDNN handle 从单 handle 改为 per-stream handles 数组（`cudnn_handles_[5]`），多流捕获安全基础
- 去掉 `graph_execs_` 命名图映射（CapturedGraph 自己管理 exec handles）
- 去掉 `create_temp_event`/`destroy_temp_event`（事件生命周期由 `MultiStreamCaptureState` 统一管理）
- 保留流管理、workspace 管理

### 5.2 接口设计

```cpp
// ===== include/renaissance/backend/device_context.h =====
#pragma once

#include "renaissance/core/types.h"
#include <cstddef>
#include <cstdint>

namespace tr {

class MemoryPlan;

class DeviceContext {
public:
    explicit DeviceContext(int device_id);
    ~DeviceContext();

    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&&) = delete;
    DeviceContext& operator=(DeviceContext&&) = delete;

    [[nodiscard]] int  device_id() const noexcept { return device_id_; }
    [[nodiscard]] bool is_gpu() const noexcept { return device_id_ >= 0; }
    [[nodiscard]] int  rank_for_context() const noexcept { return rank_for_context_; }
    void set_rank(int rank) noexcept { rank_for_context_ = rank; }
    void set_memory_plan(const MemoryPlan* mp) noexcept { current_mp_ = mp; }

    [[nodiscard]] void* ptr_at(int dtensor_id) const noexcept;

    [[nodiscard]] void* stream(StreamKind kind) const;
    void synchronize_all() const;
    void synchronize_stream(StreamKind kind) const;
    void device_sync() const;

    [[nodiscard]] void* cudnn_handle(StreamKind kind) const noexcept {
        return cudnn_handles_[stream_index(kind)];
    }

    [[nodiscard]] void* workspace(StreamKind kind) const;
    [[nodiscard]] size_t workspace_size(StreamKind kind) const;
    void pre_allocate_workspace(StreamKind kind, size_t size);
    void ensure_workspace(StreamKind kind, size_t req_size);

private:
    int device_id_;
    int rank_for_context_ = 0;
    const MemoryPlan* current_mp_ = nullptr;
    void* cudnn_handles_[5] = {};
    void* streams_[5] = {};
    struct WSpace { void* ptr = nullptr; size_t size = 0; };
    WSpace workspaces_[5];
};

inline int stream_index(StreamKind k) noexcept {
    switch (k) {
        case StreamKind::TRANS:   return 0;
        case StreamKind::COMP_1:  return 1;
        case StreamKind::COMP_2:  return 2;
        case StreamKind::COMP_3:  return 3;
        case StreamKind::UPDATE:  return 4;
    }
    return 0;
}

} // namespace tr
```

### 5.3 关键实现

```cpp
void* DeviceContext::ptr_at(int dtensor_id) const noexcept {
    TR_DEBUG_CHECK(dtensor_id >= 0, IndexError,
                  "ptr_at: invalid dtensor_id " << dtensor_id);
    const DTensor& dt = current_mp_->dtensor(dtensor_id);
    return ArenaKeeper::instance().ptr_at(
        rank_for_context_, dt.global_offset);
}

DeviceContext::DeviceContext(int device_id) : device_id_(device_id) {
    if (device_id_ < 0) return;
#ifdef TR_USE_CUDA
    cudaSetDevice(device_id_);
    for (int i = 0; i < 5; ++i) {
        cudaStreamCreateWithFlags(
            reinterpret_cast<cudaStream_t*>(&streams_[i]),
            cudaStreamNonBlocking);
        cudnnCreate(reinterpret_cast<cudnnHandle_t*>(&cudnn_handles_[i]));
        cudnnSetStream(reinterpret_cast<cudnnHandle_t>(cudnn_handles_[i]),
                       reinterpret_cast<cudaStream_t>(streams_[i]));
    }
#endif
}
```

---

## 六、pre_capture() —— 去重 + 三段式捕获

### 6.1 Phase B1：去重

```cpp
PreCaptureResult pre_capture(const GraphAtlas& compile_atlas,
                              const std::vector<DeviceContext*>& contexts) {
    const int num_ranks = static_cast<int>(contexts.size());
    PreCaptureResult result;
    result.atlas = compile_atlas;

    // B1: 去重
    std::unordered_map<CapturedGraph::Key, int32_t, CapturedGraph::KeyHash> seen;
    std::unordered_map<CapturedGraph::Key, const MemoryPlan*, CapturedGraph::KeyHash> key_to_mp;

    for (size_t vi = 0; vi < GraphAtlas::kMaxVariants; ++vi) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            auto& slot = result.atlas.slot(vi, gi);
            if (!slot.cg || !slot.mp) continue;

            CapturedGraph::Key key{slot.cg, static_cast<GraphId>(gi), slot.shape_id};
            key_to_mp[key] = slot.mp;

            auto it = seen.find(key);
            if (it != seen.end()) {
                slot.captured_idx = it->second;
                ++result.reused;
            } else {
                slot.captured_idx = static_cast<int32_t>(seen.size());
                seen[key] = slot.captured_idx;
                ++result.captured;
            }
            ++result.total_slots;
        }
    }

    const int32_t K = static_cast<int32_t>(seen.size());
    result.graphs.resize(K);
    for (auto& cg : result.graphs) cg.reserve_ranks(num_ranks);

    std::vector<CapturedGraph::Key> key_by_idx(K);
    for (const auto& [key, idx] : seen) key_by_idx[idx] = key;
```

### 6.2 Phase B2：cuDNN 预热

```cpp
    // B2: cuDNN 预热（仅 CUDA，GPU 0 单线程串行）
    const bool is_cuda = (contexts[0]->is_gpu());
    if (is_cuda && K > 0) {
        int master_gpu = contexts[0]->device_id();
#ifdef TR_USE_CUDA
        cudaSetDevice(master_gpu);
        for (int32_t k = 0; k < K; ++k) {
            const auto& key = key_by_idx[k];
            if (!key.cg) continue;
            for (const auto& node : key.cg->nodes(key.gid)) {
                if (node.kind != GraphNode::Kind::COMPUTE) continue;
                if (!is_cudnn_op(node.compute_op)) continue;
                warmup_single_cudnn_op(node, compile_atlas, *contexts[0]);
            }
        }
        cudaDeviceSynchronize();
#endif
    }
```

### 6.3 Phase B3：捕获（0 串行 + 1~N-1 并行）

```cpp
    // B3: 捕获
    std::vector<std::exception_ptr> exc(num_ranks);

    // Rank 0 在主线程串行
    try {
        capture_all_for_rank(result, key_by_idx, key_to_mp, *contexts[0], 0);
    } catch (...) {
        exc[0] = std::current_exception();
    }

    // Rank 1~N-1 并行
    if (num_ranks > 1) {
        std::vector<std::thread> threads;
        for (int r = 1; r < num_ranks; ++r) {
            threads.emplace_back([&, r]() {
                try {
                    capture_all_for_rank(
                        result, key_by_idx, key_to_mp, *contexts[r], r);
                } catch (...) {
                    exc[r] = std::current_exception();
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    for (int r = 0; r < num_ranks; ++r) {
        if (exc[r]) std::rethrow_exception(exc[r]);
    }
```

**并发安全**：`result.graphs[k].set_rank_exec(rank, exec)` 中每个 `(k, rank)` 唯一映射到一个线程，纯索引写入，零锁竞争。Phase B1 已 `resize(K)`，无扩容风险。

### 6.4 Phase B4：warmup launch

```cpp
    // B4: warmup launch（仅 Rank 0）
    if (is_cuda && K > 0) {
#ifdef TR_USE_CUDA
        cudaSetDevice(contexts[0]->device_id());
        for (const auto& cg : result.graphs) {
            StreamKind sk = gid_to_stream_kind(cg.key().gid);
            void* stream = contexts[0]->stream(sk);
            cg.launch(0, stream);
        }
        contexts[0]->synchronize_all();
#endif
    }

    return result;
}
```

### 6.5 capture_all_for_rank

```cpp
void capture_all_for_rank(PreCaptureResult& result,
                           const std::vector<CapturedGraph::Key>& key_by_idx,
                           const std::unordered_map<CapturedGraph::Key, const MemoryPlan*,
                                                    CapturedGraph::KeyHash>& key_to_mp,
                           DeviceContext& ctx,
                           int rank) {
    ctx.set_rank(rank);

    if (ctx.is_gpu()) {
#ifdef TR_USE_CUDA
        cudaSetDevice(ctx.device_id());
#endif
    }

    const int32_t K = static_cast<int32_t>(key_by_idx.size());
    for (int32_t k = 0; k < K; ++k) {
        const auto& key = key_by_idx[k];
        auto it = key_to_mp.find(key);
        TR_CHECK(it != key_to_mp.end() && it->second != nullptr, RuntimeError,
                 "No MemoryPlan found for key");
        const MemoryPlan* mp = it->second;

        ctx.set_memory_plan(mp);
        auto cg = CapturedGraph::capture(*key.cg, *mp, key.gid, key.shape, ctx);
        result.graphs[k].set_rank_exec(rank, cg.per_rank_execs()[0]);
    }
}
```

---

## 七、GraphExecutor —— per-rank 运行调度器

### 7.1 设计定位

每个 rank（每个 GPU 线程）持有一个 GraphExecutor 实例。它知道：
- 自己属于哪个 rank
- 当前是训练还是验证
- 当前是标准 batch 还是 last batch
- 当前分辨率阶段（渐进式）
- A/B 双缓冲状态
- 是否跳过首层反向（首层冻结）

### 7.2 类定义与核心实现

```cpp
// ===== include/renaissance/backend/graph_executor.h =====
#pragma once

#include "renaissance/graph/captured_graph.h"
#include "renaissance/graph/graph_atlas.h"
#include "renaissance/backend/device_context.h"
#include <cstdint>

namespace tr {

class GraphExecutor {
public:
    GraphExecutor(int rank,
                  const DeviceContext& ctx,
                  const PreCaptureResult& pre_capture_result);

    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;

    void set_training(bool v) noexcept       { is_training_ = v; }
    void set_last_batch(bool v) noexcept     { is_last_batch_ = v; }
    void set_low_resolution(bool v) noexcept { use_low_res_ = v; }
    void set_skip_first_bwd(bool v) noexcept { skip_first_bwd_ = v; }
    void toggle_ab() noexcept                { ab_toggle_ = !ab_toggle_; }
    void reset_ab(bool to_a = true) noexcept { ab_toggle_ = !to_a; }

    [[nodiscard]] int  rank() const noexcept { return rank_; }
    [[nodiscard]] int  device_id() const noexcept { return ctx_.device_id(); }
    [[nodiscard]] bool is_gpu() const noexcept { return ctx_.is_gpu(); }

    void launch(GraphId gid) const;
    void launch_dual(GraphId gid1, GraphId gid2) const;
    void sync_all() const { ctx_.synchronize_all(); }

    void run_train_step();
    void run_val_step();

private:
    int rank_;
    const DeviceContext& ctx_;
    const PreCaptureResult& pc_;

    bool is_training_      = true;
    bool is_last_batch_    = false;
    bool use_low_res_      = false;
    bool skip_first_bwd_   = false;
    bool ab_toggle_        = false;

    int32_t resolve_variant() const noexcept;
    int32_t resolve_idx(GraphId gid) const;
};
```

**变体路由**：

```cpp
int32_t GraphExecutor::resolve_variant() const noexcept {
    if (is_training_) {
        return is_last_batch_
            ? (use_low_res_ ? 3 : 1)
            : (use_low_res_ ? 2 : 0);
    } else {
        return is_last_batch_ ? 5 : 4;
    }
}

int32_t GraphExecutor::resolve_idx(GraphId gid) const {
    return pc_.atlas.index(resolve_variant(), gid);
}

void GraphExecutor::launch(GraphId gid) const {
    int32_t idx = resolve_idx(gid);
    if (idx < 0) return;
    pc_.graphs[idx].launch(rank_, ctx_.stream(gid_to_stream_kind(gid)));
}

void GraphExecutor::launch_dual(GraphId gid1, GraphId gid2) const {
    int32_t idx1 = resolve_idx(gid1);
    int32_t idx2 = resolve_idx(gid2);

    StreamKind sk1 = gid_to_stream_kind(gid1);
    StreamKind sk2 = gid_to_stream_kind(gid2);

    TR_DEBUG_CHECK(sk1 != sk2, RuntimeError,
                   "launch_dual: " << static_cast<int>(gid1) << " and "
                   << static_cast<int>(gid2) << " map to same StreamKind, "
                   "cannot parallelize");

    if (idx1 >= 0) pc_.graphs[idx1].launch(rank_, ctx_.stream(sk1));
    if (idx2 >= 0) pc_.graphs[idx2].launch(rank_, ctx_.stream(sk2));
}
```

### 7.3 训练 Step（严格对齐 Guide.md）

```cpp
void GraphExecutor::run_train_step() {
    if (is_last_batch_) {
        run_train_step_last_batch();
        return;
    }

    // 0. 更新 LR（H2D，数据量 4 bytes，同步完成，不计 MLPerf 计时）
    update_lr_scalar(current_lr_, optimizer_scalar_ids_, ctx_);

    GraphId xfer_gid      = ab_toggle_ ? GraphId::TRANSFER_A : GraphId::TRANSFER_B;
    GraphId first_fwd_gid = ab_toggle_ ? GraphId::FIRST_FWD_A : GraphId::FIRST_FWD_B;

    // 1. Transfer 到计算区
    launch(xfer_gid);

    // 2. ZERO GRAD
    launch(GraphId::ZERO_GRAD);

    // 3. 首层正向
    launch(first_fwd_gid);

    // 4. 双图并行：Transfer 下一 batch + 深层正反向
    GraphId next_xfer = ab_toggle_ ? GraphId::TRANSFER_B : GraphId::TRANSFER_A;
    launch_dual(next_xfer, GraphId::DEEP_FWD_BWD);
    sync_all();

    // 5. 梯度 CAST + NaN check（RANGE_NAN_CHECK_ALL_G 写入 S_MASK）
    launch(GraphId::CAST_AND_CHECK);

    // 6. 深层通信 + 首层反向
    if (!skip_first_bwd_) {
        launch_dual(GraphId::DEEP_COMM, GraphId::FIRST_BWD);
    } else {
        launch(GraphId::DEEP_COMM);
    }
    sync_all();

    // 7. 首层通信
    if (!skip_first_bwd_) {
        launch(GraphId::FIRST_COMM);
    }

    // 8. BN 统计量通信
    launch(GraphId::STATS_COMM);

    // 9. NaN 检查分支（D2H 读 S_MASK，数据量极小）
    bool has_nan = check_nan_flag();
    if (!has_nan) {
        // 10. 优化器更新（含 RANGE_BN_STATS_COPY：next→prev）
        // Compiler 必须保证 RANGE_BN_STATS_COPY 在 OPTIMIZER 桶中
        // 且位于所有权重更新 kernel 之后、桶结束之前
        launch(GraphId::OPTIMIZER);

        // 11. EMA 更新
        launch(GraphId::EMA_UPDATE);
    } else {
        // NaN：跳过更新，由上层 Task 调整 AMP loss scaling
        on_nan_detected();
    }

    // 12. 权重 CAST FP32→FP16（独立 shape 无关 Graph，双图并行；见 §8.2）
    // GraphId::CAST_MAIN_FP32_TO_FP16：将主模型 FP32 权重转 FP16 复制到训练区
    // GraphId::CAST_EMA_FP32_TO_FP16：将 EMA 模型 FP32 权重转 FP16 复制到推理区
    // 若 Compiler 设计为将 CAST 合并入 EMA_UPDATE（RANGE 节点），则此步省略
    launch_dual(GraphId::CAST_MAIN_FP32_TO_FP16,
                GraphId::CAST_EMA_FP32_TO_FP16);
    sync_all();

    toggle_ab();
}

void GraphExecutor::run_train_step_last_batch() {
    // 0. 更新 LR
    update_lr_scalar(current_lr_, optimizer_scalar_ids_, ctx_);

    GraphId first_fwd_gid = ab_toggle_ ? GraphId::FIRST_FWD_A : GraphId::FIRST_FWD_B;

    // 1. ZERO GRAD（shape 无关）
    launch(GraphId::ZERO_GRAD);

    // 2. 首层正向（shape 不同，resolve_variant() 自动选对图）
    launch(first_fwd_gid);

    // 3. 深层正反向（单图，不双图并行——无下一 batch）
    launch(GraphId::DEEP_FWD_BWD);
    sync_all();

    // 4. 梯度 CAST + NaN check
    launch(GraphId::CAST_AND_CHECK);

    // 5. 深层通信 + 首层反向
    if (!skip_first_bwd_) {
        launch_dual(GraphId::DEEP_COMM, GraphId::FIRST_BWD);
    } else {
        launch(GraphId::DEEP_COMM);
    }
    sync_all();

    // 6~12. 后续与标准 batch 完全相同
    if (!skip_first_bwd_) {
        launch(GraphId::FIRST_COMM);
    }
    launch(GraphId::STATS_COMM);

    bool has_nan = check_nan_flag();
    if (!has_nan) {
        launch(GraphId::OPTIMIZER);
        launch(GraphId::EMA_UPDATE);
    } else {
        on_nan_detected();
    }

    launch_dual(GraphId::CAST_MAIN_FP32_TO_FP16,
                GraphId::CAST_EMA_FP32_TO_FP16);
    sync_all();

    toggle_ab();
}
```

### 7.4 验证 Step

> **推理结果累加**：Guide.md 要求"把结果复制或加到主模型结果区"。此操作由 Compiler 在 Phase A4 将 `RANGE_ADD` 或 `RANGE_COPY` 节点放入 `INF_MAIN_A/B` 和 `INF_EMA_A/B` 图末尾。`run_val_step()` 只需 launch 对应图，累加由 GPU 自动完成。

```cpp
void GraphExecutor::run_val_step() {
    GraphId inf_main = ab_toggle_ ? GraphId::INF_MAIN_A : GraphId::INF_MAIN_B;
    GraphId xfer     = ab_toggle_ ? GraphId::TRANSFER_B : GraphId::TRANSFER_A;

    launch_dual(inf_main, xfer);
    sync_all();

    GraphId inf_ema = ab_toggle_ ? GraphId::INF_EMA_A : GraphId::INF_EMA_B;
    launch(inf_ema);
    sync_all();

    toggle_ab();
}
```

---

## 八、去重与运行时索引

### 8.1 去重键

`Key{cg*, gid, shape_id}` —— 三维全部相等才算同一张图。ShapeId 是值比较（而非 MemoryPlan 指针比较），不同变体的相同输入尺寸自动碰撞 -> 复用一张 CapturedGraph。

### 8.2 shape 无关图

8 种 GraphId 标记 `is_shape_invariant_graph()` = true：
```
TRANSFER_A, TRANSFER_B, ZERO_GRAD, CAST_AND_CHECK,
FIRST_COMM, DEEP_COMM, STATS_COMM, EMA_UPDATE
```

若 `CAST_MAIN_FP32_TO_FP16` 和 `CAST_EMA_FP32_TO_FP16` 采用独立 GraphId（非合并入 EMA_UPDATE 的 RANGE 节点），则也应加入 shape 无关列表，总数变为 10 种。CAST 操作只依赖 Region 大小，不依赖 batch size。

Compile 期统一填入 `kShapeInvariant{0,0,0,0}`。Phase B 去重时 6 个变体的同一种 GraphId 全部产生相同 Key -> 首次未命中（1 张），后续 5 次命中 -> 8 张图服务 48 个槽位。

### 8.3 数量概算

|   | 种类 | 槽位 | 捕获 | 复用 |
|---|------|:--:|:--:|:--:|
| shape 无关 | 8~10 | 48~60 | 8~10 | 40~50 |
| 训练 shape 相关 | 4~8 | 16~32 | 16~32 | 0 |
| 推理 shape 相关 | 2~4 | 4~8 | 4~8 | 0 |
| **合计** |   | **68~100** | **28~50** | **40~50** |

典型（2 分辨率 + EMA 禁用 + no last batch 特殊形状 + CAST 独立）：约 29 张 CUDA Graph。

### 8.4 Phase C O(1) 索引

```cpp
int32_t idx = pc_.atlas.index(resolve_variant(), gid);  // table_[6][17] 数组读
graphs_[idx].launch(rank_, stream);                      // is_cuda_ 分支 -> launch
```

零 hash、零 map::find、零虚函数。

---

## 九、CPU 后端

### 9.1 CpuOp

```cpp
struct CpuOp {
    void (*fn)(void* ctx) = nullptr;
    void* ctx = nullptr;
};  // 共 16 bytes，栈分配，零虚调用
```

vs 现状 `std::function<void()>`（~32 bytes + 堆分配 + 虚调用），每 50 节点图节省 800 bytes。

### 9.2 捕获与 CpuOpContext

**设计决策：ctx 包含 `DeviceContext* + DTensorID[]`，运行时动态解析地址。**

原因：CPU 模式下 ArenaKeeper 地址不变（无 GC/迁移），但出于与 CUDA 路径一致的考虑、以及未来可能支持 CPU 端 DTensor 重排，采用与 CUDA 路径相同的动态解析策略。

```cpp
struct CpuOpContext {
    const DeviceContext* ctx = nullptr;
    int32_t input_ids[8] = {};
    int32_t output_ids[8] = {};
    int num_inputs = 0;
    int num_outputs = 0;
    OpParams params;
};

// capture_cpu 中：
CpuOpContext* op_ctx = alloc_cpu_op_context();
op_ctx->ctx = &device_ctx;
// ... 从 GraphNode 填充 input_ids/output_ids/params ...
cpu_ops_.push_back(CpuOp{entry.launch_cpu, op_ctx});

// launch_cpu 示例：
static void launch_cpu_fc_fwd(void* raw_ctx) {
    auto* c = static_cast<CpuOpContext*>(raw_ctx);
    const float* x = static_cast<const float*>(c->ctx->ptr_at(c->input_ids[0]));
    float* y = static_cast<float*>(c->ctx->ptr_at(c->output_ids[0]));
    // ...
}
```

### 9.3 执行

```cpp
for (const auto& op : cpu_ops_) op.fn(op.ctx);  // ~2ns/op
```

CPU"图"本质是有序函数队列——无异步、无流、无依赖拓扑。`sync_all()` 在 CPU 模式为空操作。

---

## 十、并行运行

### 10.1 Level 1（多卡 per-rank）

CUDA 设备上下文（`cudaSetDevice`）是线程局部的。不同 rank 的 `cudaGraphLaunch` 必须在不同线程中发出。

```cpp
std::vector<std::thread> threads;
for (auto& ex : executors) {
    threads.emplace_back([&]() {
        cudaSetDevice(ex->device_id());
        ex->launch(gid);
    });
}
```

### 10.2 Level 2（同卡双图）

`cudaGraphLaunch` 非阻塞。两次 launch 到不同 StreamKind（如 TRANS + COMP_1），GPU 硬件自动并行。**不需要额外线程。**

**约束**：每卡同时最多 2 张 CUDA Graph，必须绑定到**不同 StreamKind**。不允许同卡 3 张 CUDA Graph 同时跑。

### 10.3 CPU 退化

CPU 路径 Level 2 退化为串行：`run(a); run(b)`。无异步、无流。

---

## 十一、辅助机制

### 11.1 LR 标量更新

**H2D 到 `S_SCALAR_FP32`**。数据量 4 bytes，同步完成，不计 MLPerf 计时。

Compiler 在 Phase A3 产出：
```cpp
struct OptimizerScalarIds { int32_t lr, beta, tc, wd, eps; };
```

运行期：
```cpp
void update_lr(float lr, const OptimizerScalarIds& ids, DeviceContext& ctx) {
    void* lr_ptr = ctx.ptr_at(ids.lr);
    cudaMemcpyAsync(lr_ptr, &lr, 4, H2D, ctx.stream(StreamKind::UPDATE));
    cudaStreamSynchronize(ctx.stream(StreamKind::UPDATE));
}
```

**为什么不用 `cudaGraphExecKernelNodeSetParams`**：需要精确定位图中哪个 kernel 节点读取 lr，所有含 lr 的图都需要修改，易遗漏。H2D 数据量极小（4 bytes），维护成本远低于节点参数修改。

### 11.2 BN Stats 入图

**全部由 Compiler 负责。**`RANGE_BN_STATS_COPY` 作为 RANGE 节点，在 `build_auxiliary_graphs` 阶段放入 `OPTIMIZER` 桶。

**顺序约束**：`RANGE_BN_STATS_COPY` 必须是 `OPTIMIZER` 桶中的**最后一个 RANGE 节点**——在所有优化器权重更新 kernel 之后、桶结束之前。这确保 BN 统计量的 `next→prev` 在权重更新完成之后、EMA 权重更新之前执行，精确对齐 Guide.md 的步骤顺序。capture 遍历自然执行。无需 capture 末尾追加。

### 11.3 Running Stats（跨卡同步）

每个 epoch 末尾跨卡同步 BN 统计量。通信图本身是 shape 无关的——`STATS_COMM` GraphId 在 Compile 期以 `kShapeInvariant` 标注，Phase B 全局复用一张图。

---

## 十二、异常安全与诊断

### 12.1 RAII CaptureGuard（强制）

```cpp
struct CaptureGuard {
    cudaStream_t stream;
    bool committed = false;
    ~CaptureGuard() {
        if (!committed && stream) {
            cudaGraph_t dummy = nullptr;
            cudaStreamEndCapture(stream, &dummy);
            if (dummy) cudaGraphDestroy(dummy);
        }
    }
};
```

无 Guard 的后果：算子 replay 中 cudnn kernel 失败抛异常 -> 流锁死在 capture 态 -> 后续任何 CUDA API 皆报 `cudaErrorStreamCaptureUnsupported` -> 程序无法恢复。

### 12.2 异常聚合

```cpp
std::vector<std::exception_ptr> exc(num_ranks);
// 子线程: try { capture } catch (...) { exc[rank] = std::current_exception(); }
// 主线程 join 后统一 rethrow
```

### 12.3 cudaGraphInstantiate 诊断

```cpp
char error_log[2048] = {};
cudaGraphInstantiate(&exec, graph_obj, &error_node, error_log, sizeof(error_log));
```

### 12.4 Phase B4 warmup launch

`cudaGraphInstantiate` 只验证拓扑——warmup launch + sync 在 MLPerf 计时前暴露 runtime 错误（illegal memory access 等）。

---

## 十三、性能保证

### 13.1 Phase C 热路径（一条指令链）

```cpp
int32_t idx  = atlas_.index(variant, gid);      // table_ 二维数组读，~1ns
auto&    cg   = graphs_[idx];                    // vector 单下标，~1ns
cg.launch(rank_, stream);                        // is_cuda_ 分支(~1ns) + launch(~10ns)
// 总计 <15ns，远低于 50ns 要求
```

零 hash、零 map::find、零虚函数。

### 13.2 全部性能决策清单

| 决策 | 收益 |
|------|------|
| Phase C `atlas_.index(variant, gid)` | O(1) 纯数组双下标 |
| 全局数组 dispatch | O(1)，无 hash，缓存友好 |
| `ptr_at(id)` -> DTensor offset + Arena base_ptr | 零中间层，零虚函数 |
| `__restrict` 修饰全部指针 | 编译器矢量化 |
| stride 入参（NHWC 256 字节对齐） | ~30% 带宽提升 |
| grid-stride loop（CUDA） | 单 kernel 覆盖全部输出 |
| `HEURISTICS_CHOICE + FALLBACK` | cuDNN engine 双保险 |
| ShapeId 值去重 | 同尺寸跨变体自动复用 |
| everything-in-compile | Phase C 热路径只剩 ptr_at + launch |
| `cudaGraphLaunch` 非阻塞 | CPU 不等 GPU，Level 2 零线程 |
| CpuOp 裸函数指针 | 零堆分配，零虚调用 |
| BN Stats 入图（Compiler） | 运行期零 CPU 介入 |
| Phase B1 resize(K) | Phase B3 并行写入零扩容 |

### 13.3 禁止在 Phase C 做的事

- CUDA Graph 捕获 API
- cuDNN `build_graph` / `build_plans`
- `map::find` / `unordered_map::operator[]`
- `infer_shapes`
- polling / busy-wait
- CUDA Graph 内 `cudaStreamSynchronize` / `cudaDeviceSynchronize`
- 默认流（全部用 `cudaStreamNonBlocking`）
- 同卡 3 张 CUDA Graph 同时跑

---

## 十四、文件组织

```
include/renaissance/graph/
├── captured_graph.h          # CapturedGraph + CpuOp + NativeGraph + PreCaptureResult
├── graph_atlas.h             # GraphAtlas（已完成，不变）
├── computation_graph.h       # GraphNode + GraphId（已完成，不变）
└── pre_capture.h             # pre_capture() 声明

include/renaissance/backend/
├── device_context.h          # DeviceContext（从 legacy 重构）
├── graph_executor.h          # GraphExecutor per-rank 调度器
└── memory_arena.h            # ArenaKeeper（已完成，不变）

src/graph/
├── captured_graph.cpp        # capture + launch + 析构
├── capture_cuda.cpp          # CUDA capture 主逻辑 + MultiStreamCaptureState + RAII Guard
├── capture_cpu.cpp           # CPU capture 逻辑
├── capture_multi_stream.cpp  # 多流 replay（算子内事件链 + 算子间屏障）
└── pre_capture.cpp           # pre_capture() + capture_all_for_rank() + warmup

src/backend/
├── device_context.cpp        # DeviceContext 实现
├── graph_executor.cpp        # GraphExecutor
└── ops/
    ├── dtensor/              # ComputeOp CPU + CUDA 实现
    └── range/                # RangeOp CPU + CUDA 实现
```

---

## 十五、实施路线图

### Phase 1（P0 — 核心骨架修复）

| # | 任务 | 产出 |
|---|------|------|
| 1 | CapturedGraph 去 `#ifdef`，增加 `is_cuda_` + `per_rank_execs_` + `CpuOp{fn, ctx}` | `captured_graph.h` |
| 2 | DeviceContext 从 legacy 重构适配新版（`ptr_at` 换算法，per-stream cuDNN handles） | `device_context.h/cpp` |
| 3 | capture_cpu() 实现 | `capture_cpu.cpp` |
| 4 | pre_capture() 去重逻辑 | `pre_capture.cpp` |

### Phase 2（P0 — 单卡多流 CUDA 捕获）

| # | 任务 | 产出 |
|---|------|------|
| 5 | MultiStreamCaptureState（含事件池 + `gid_to_stream_kind` 映射）+ RAII CaptureGuard 实现 | `capture_multi_stream.cpp` |
| 6 | capture_cuda()：BeginCapture -> replay -> EndCapture -> Instantiate + `cleanup_all_events()` | `capture_cuda.cpp` |
| 7 | CBR 三流 replay + 算子内事件链（使用 `state.alloc_temp_event()`） | `capture_multi_stream.cpp` |
| 8 | GraphExecutor 骨架：`launch(gid)` + `launch_dual(gid1,gid2)` + `resolve_variant()` | `graph_executor.h/cpp` |

### Phase 3（P0 — 多卡并行捕获）

| # | 任务 | 产出 |
|---|------|------|
| 9 | cuDNN 预热（GPU 0 串行 build_plans + execute） | `pre_capture.cpp` |
| 10 | 三段式捕获（0 串行 + 1~N-1 并行）+ 异常聚合 | `pre_capture.cpp` |
| 11 | Phase B4 warmup launch + sync | `pre_capture.cpp` |
| 12 | per-rank `set_rank_exec` 批量合并 | `pre_capture.cpp` |

### Phase 4（P1 — 工作流集成）

| # | 任务 |
|---|------|
| 13 | `run_train_step()`：A/B 双缓冲 + 双图并行 + 首层冻结 + NaN 分支 + last batch 处理 + 权重 CAST |
| 14 | `run_val_step()`：推理 + 双缓冲 + 结果累加（Compiler 入图） |
| 15 | BN Stats 入图（Compiler Phase A4，`RANGE_BN_STATS_COPY` 作为 OPTIMIZER 桶最后节点） |
| 16 | LR H2D 更新到 S_SCALAR_FP32（step 开始时 `update_lr_scalar()`） |
| 17 | NaN 检查工作流：RANGE_NAN_CHECK_ALL_G -> CPU D2H 读标志位 -> 分支（`check_nan_flag()`） |

### Phase 5（P2 — 算子填充与验收）

| # | 任务 |
|---|------|
| 18 | 填充 `g_compute_op_table`：每个 ComputeOp 的 CPU + CUDA launch |
| 19 | 填充 `g_range_op_table`：每个 RangeOp 的实现 |
| 20 | cuDNN FE Graph 集成：build_graph 在 compile 阶段，execute 在 capture 阶段 |

---

## 十六、关键认知总结

1. **ComputationGraph 不对 Stream 做任何假设。** 流选择是 capture 引擎在 Phase B 根据算子类型动态完成的。GraphId 是纯平台无关 IR。

2. **同一 CUDA Graph 内可用 COMP_1/COMP_2/COMP_3，但决不会混合 TRANS。** 计算通信重叠通过并行跑两张独立的 CUDA Graph 实现。

3. **多流捕获通过 MultiStreamCaptureState 追踪事件依赖链。** 算子内部事件链由算子 replay 函数维护。算子间跨流屏障由 capture 循环在算子边界自动插入。不在 GraphNode 中持久化流/事件信息。

4. **CapturedGraph 是平台相关的可执行实体。** `is_cuda_` 给出肯定答案。`per_rank_execs_[N]` 让 `launch(rank, stream)` 一行完成所有分发。

5. **`#ifdef TR_USE_CUDA` 只进函数体。** 类定义始终含有 `per_rank_execs_`（vector<void*>）和 `cpu_ops_`。数据布局的编译期切割是根本错误的设计。

6. **继承 legacy DeviceContext 的设计智慧。** 不推翻重建——5 流管理、per-stream cuDNN handles（`cudnn_handles_[5]`）、per-stream workspace、非阻塞流策略全部保留。只适配新版 MemoryPlan（`ptr_at` 从 ptr_table 改为 DTensor offset + ArenaKeeper）。

7. **"GPU 0 串行预热 -> Rank 0 串行捕获 -> Rank 1~N-1 并行捕获" 不是噱头。** cuDNN 全局 engine cache 竞争是物理现实。Legacy 已验证：8 GPU 全并行间歇性失败（illegal memory access）；三步串行化后不再触发。

8. **Level 2 并行不需要额外线程。** `cudaGraphLaunch` 非阻塞，两次 launch 到不同 StreamKind 后 GPU 硬件自动并行。CPU 路径退化为串行。

9. **一切提前到 compile。** `infer_shapes`、`build_range_op_ranges`、cuDNN `build_plans`、BN Stats 入图——全部在 Phase A 完成。Phase C 热路径只剩 `ptr_at` + `launch`。

10. **ShapeId 值去重而非指针去重。** 不同变体的 MemoryPlan 是不同的对象指针。ShapeId 值比较使同尺寸跨变体自动复用。

11. **CPU "图"就是函数队列。** `vector<CpuOp>` 顺序调用，但接口与 CUDA 完全一致（`launch(rank, stream)`），调用方零后端感知。

12. **严格使用 `cg.nodes(gid)`。** `linear_nodes_` 是 SimpleTask 手动绘图遗留。正式 Compiler 产出填在 17 个 GraphId 桶内。

13. **RAII CaptureGuard 是强制要求。** 无 Guard 的 capture 异常 = 流永久锁死在 capture 态 = 程序后续不可运行。

14. **H2D 4 bytes 替代 `cudaGraphExecKernelNodeSetParams`。** 数据量接近零、维护成本远低于遍历图中特定 kernel 节点修改参数。

15. **Compile 阶段的 Running Stats 入图 > capture 末尾追加。** Compiler 是 IR 的唯一权威。后插入逻辑与 Compiler 未来的布局调整必然冲突。

16. **GraphExecutor 是对 Guide.md 训练/验证工作流的精确编码。** run_train_step 和 run_val_step 每行都与 Guide.md 的步骤一一对应。不是近似——是精确翻译。

17. **DeviceContext 的 `ptr_at` 核心重构：不再预先建表，而是动态从 MemoryPlan 查 offset + ArenaKeeper::ptr_at(rank, offset)。** 这使 DeviceContext 从"绑定唯一的 MemoryPlan/Arena"切换到"可随图集切换 MemoryPlan"模式，是图集切换的技术基础。