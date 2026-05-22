# UTK_FINAL：RANGE 异步传输三大算子实现与测试 —— 终版科学方案

> 综合 UTK1 / UTK2 / UTK3 三稿 + 当前代码 Cross-check
> 2026-05-21 · 终版

---

## 〇、文档溯源与综合决策

| 源文档 | 定位 | 被采纳的核心贡献 |
|--------|------|:--|
| [UTK1.md](file:///r:/renaissance/UTK1.md) | 完整技术方案 | GlobalRegistry 管道、Region→offset 映射、编译期检查 |
| [UTK2.md](file:///r:/renaissance/UTK2.md) | 多 RANK 原生方案 | `compile_capture_simple` 自动分配 StagingBufferPool、per-rank 测试数据 pattern、`g.append_range()` API |
| [UTK3.md](file:///r:/renaissance/UTK3.md) | 测试专项方案 | 完整测试代码模板、buffer 计算公式、overlap 测试框架、成功标准 |

### 终版综合决策

| 分歧点 | UTK1 | UTK2 | UTK3 | **UTK_FINAL 裁决** |
|--------|:--:|:--:|:--:|------|
| A/B kernel 指针获取 | GlobalRegistry 新字段 + TransferStation 注册 | 现场计算 `label_aligned` | 现场计算 `label_aligned` | **现场计算**：代码自包含，无耦合，公式与 TransferStation 完全一致 |
| StagingBufferPool 分配 | 测试显式分配 | `compile_capture_simple` **自动分配** | 测试显式分配 | **自动分配**：测试仅需设置 GlobalRegistry 参数，无需关心 buffer 公式 |
| DTENSOR per-rank | Phase 2 | Phase 1（原生支持） | 保持现状 | **Phase 1 保持现状**：现有 `fill_transfer_buffer` 路径已正确 |
| 测试验证指标 | 逐元素 max\|diff\| | 逐元素精确比较 | 逐元素 max\|diff\| | **MSE + max\|diff\| 双重指标**（按用户要求） |

---

## 一、问题诊断：当前代码唯一关键缺陷

> 经 UTK1/2/3 三稿交叉验证，问题收敛到**唯一一点**。

### 1.1 缺陷定位

[文件：src/backend/ops/range/h2d_op.cpp:L104-L116](file:///r:/renaissance/src/backend/ops/range/h2d_op.cpp#L104-L116)

```cpp
static void* s_placeholder_h2d = nullptr;
if (!s_placeholder_h2d) {
    cudaHostAlloc(&s_placeholder_h2d, 4096, cudaHostAllocDefault);  // ← 仅 4KB
}
void* src = s_placeholder_h2d;  // ← 永远指向未初始化内存
```

**完整影响链**：

```
CUDA Graph Capture 期：
  cudaMemcpyAsync(dst, s_placeholder_h2d, dst_size, TRANS_stream)
  → CUDA 将「src = s_placeholder_h2d 地址」记录到 graph memcpy node

CUDA Graph Instantiate 期：
  → src 指针被固定为 s_placeholder_h2d 的虚拟地址

运行时 cudaGraphLaunch 期：
  → memcpy 从 s_placeholder_h2d 读取垃圾数据 → 传输到 GPU
  → s_placeholder_h2d 仅 4096 字节，典型 per_zone 为数百 KB~数 MB
  → 越界读取行为未定义
```

### 1.2 DTENSOR 算子状态：完全重构

#### 1.2.1 现有实现的致命缺陷

`RANGE_H2D_COPY_DTENSOR` 的现有实现在 **graph 构造** 和 **kernel** 两个层面均存在严重问题：

| 层面 | 缺陷 | 后果 |
|------|------|------|
| **Graph 构造** | [UTK_FINAL §3.4](file:///r:/renaissance/UTK_FINAL.md#L465) 使用 `mp.region_range(d_v.region)` 构造 `output_ranges` | `region_range(Region)` 返回 `MemRange{0, 0, r, r}`（延迟解析：offset=0, size=0）；`launch_range_h2d_copy_dtensor_cuda` 在 L66 检测到 `seg.size == 0` 直接 return → 不执行任何 memcpy |
| **Kernel** | `s_pinned_map` 是全局单例，所有 rank 共享 | 无法区分 per-rank 数据，不支持 per-batch per-rank 不同参数（DeepLearningTask 场景） |
| **设计哲学** | 被归类为 RANGE OP，却处理点对点 DTensor 传输 | 概念混淆，应独立为专用通道 |

**完整影响链**（graph 缺陷）：

```
Graph 构造：
  mp.region_range(Region::W_FC_WEIGHT)  →  MemRange{offset=0, size=0, start=044, end=044}

CUDA Graph Capture  L65-66：
  seg.size == 0  →  return（不记录任何 cudaMemcpyAsync node）

CUDA Graph Launch：
  → Graph 中无 memcpy node → 什么都不执行
  → fetch 回来的是 device memory 原有值（未初始化或之前的值）
```

#### 1.2.2 正确构造：参考 `dual_graph.cpp`

[dual_graph.cpp:L93-L98](file:///r:/renaissance/tests/graph/dual_graph.cpp#L93-L98) 是唯一正确使用 DTENSOR 的代码（✓ 已验证通过）：

```cpp
const DTensor& live_b = task.memory_plan().get_dtensor(d_buf_b.id);
xfer_node.output_ranges = {
    MemRange{static_cast<uint64_t>(live_b.offset()),       // ← 真实设备偏移
             static_cast<uint64_t>(live_b.slot_bytes()),   // ← 真实槽位字节数
             static_cast<int32_t>(live_b.region),          // ← 真实 Region ID
             static_cast<int32_t>(live_b.region)}
};
```

**关键区别**：`mp.get_dtensor(id)` 返回 `finalize()` 之后的真实 DTensor（含 `offset()` 和 `slot_bytes()`），而 `mp.region_range(Region)` 返回延迟解析的占位 MemRange（offset=0, size=0）。

#### 1.2.3 整体重构方案：引入 StagingParamPool

根据 UTK_REV.md 用户补充的设计意见，`RANGE_H2D_COPY_DTENSOR` 的本质**不是 RANGE OP**，而是简单的**点对点 per-RANK 参数传输**：

```
设计原理：
  1. StagingParamPool：独立的 per-rank 小参数区（256 字节 = 64×FP32），
     与 StagingBufferPool（A/B 标签/数据区）完全分开，两次不同的分配
  2. 每个 execution 仅传输 sizeof(float) = 4 字节（一个 FP32 标量）
  3. 典型用途：per-batch per-rank 学习率（LR）更新
  4. 参数放 data[0]（绝对主流场景），data[1..63] 预留给未来扩展
```

GPU 模式使用 `cudaHostAlloc`（pinned），CPU 模式使用 `malloc`（普通内存）—— 因此类名选用 `StagingParamPool` 而非 "Pinned"，以反映两种内存模式。

> **⚠️ Phase 1 限制**：hardcode 参数槽 index=0（仅支持 LR，即 data[0]）。多参数支持（任意 slot）在 Phase 2 扩展。
> CPU 模式下 `StagingParamPool` 使用 `malloc` 分配（非 pinned），因此命名避用 "Pin"。

### 1.3 五层基础架构状态

```
Layer 5: StagingBufferPool / StagingParamPool   🌕 A/B ✅, DTENSOR 新增
Layer 4: CUDA Graph capture (CapturedGraph)      ✅
Layer 3: Compiler (append_h2d)                   ✅
Layer 2: Backend Kernel (h2d_op.cpp)             🌕 A/B + DTENSOR 均需修复
Layer 1: Enum 定义 (GraphId / Region / Op)       ✅
```

---

## 二、StagingParamPool：DTENSOR 专用 per-RANK 参数区

> 用户设计指令（RL_REV2.md §用户补充）：
> "不破坏现有的 StagingBuffer 的功能，给它一个额外分配小锁页内存的功能。固定 256 字节（64 个 FP32）。
> 每个 RANK 对应一个，保存各自的指针。capture 之前确定好。"

### 2.1 类设计

```cpp
/**
 * @class StagingParamPool
 * @brief DTENSOR H2D 传输的 per-rank 小参数区
 *
 * 设计：
 * - 每个 rank 分配 256 字节（64 × FP32），GPU 模式用 cudaHostAlloc pinned，
 *   CPU 模式用 malloc
 * - 独立于 StagingBufferPool — 两次不同的分配，生命周期各自管理
 * - 指针在 compile 前确定，capture 时直接引用 — 零运行时开销
 * - Phase 1：仅使用 slot[0]（LR 的 data[0]）
 */
class StagingParamPool {
public:
    StagingParamPool(const std::vector<int>& gpu_ids, size_t bytes_per_rank = 256);
    ~StagingParamPool();

    StagingParamPool(const StagingParamPool&) = delete;
    StagingParamPool& operator=(const StagingParamPool&) = delete;

    void* ptr(int rank) const;
    void set_param(int rank, int slot, float value);
    float param(int rank, int slot) const;
    int num_ranks() const;

private:
    std::vector<int> gpu_ids_;
    std::vector<void*> ptrs_;
    size_t bytes_per_rank_;
};
```

### 2.2 实现要点

```cpp
StagingParamPool::StagingParamPool(const std::vector<int>& gpu_ids, size_t bytes_per_rank)
    : gpu_ids_(gpu_ids), bytes_per_rank_(bytes_per_rank) {
    int n = static_cast<int>(gpu_ids_.size());
    ptrs_.resize(n, nullptr);
    for (int i = 0; i < n; ++i) {
#ifdef TR_USE_CUDA
        if (gpu_ids_[i] >= 0) {
            cudaSetDevice(gpu_ids_[i]);
            cudaMallocHost(&ptrs_[i], bytes_per_rank_);
        } else
#endif
        {
            ptrs_[i] = std::malloc(bytes_per_rank_);
        }
        std::memset(ptrs_[i], 0, bytes_per_rank_);
    }
}

StagingParamPool::~StagingParamPool() {
    for (size_t i = 0; i < ptrs_.size(); ++i) {
        if (!ptrs_[i]) continue;
#ifdef TR_USE_CUDA
        if (gpu_ids_[i] >= 0) cudaFreeHost(ptrs_[i]);
        else
#endif
        std::free(ptrs_[i]);
    }
}

void* StagingParamPool::ptr(int rank) const {
    if (rank < 0 || rank >= num_ranks()) {
        TR_INDEX_ERROR("StagingParamPool rank out of range: " << rank
                       << ", num_ranks=" << num_ranks());
    }
    return ptrs_[rank];
}

void StagingParamPool::set_param(int rank, int slot, float value) {
    if (rank < 0 || rank >= num_ranks()) {
        TR_INDEX_ERROR("StagingParamPool rank out of range: " << rank);
    }
    if (slot < 0 || static_cast<size_t>(slot) >= bytes_per_rank_ / sizeof(float)) {
        TR_INDEX_ERROR("StagingParamPool slot out of range: " << slot);
    }
    float* base = static_cast<float*>(ptrs_[rank]);
    base[slot] = value;
}

float StagingParamPool::param(int rank, int slot) const {
    if (rank < 0 || rank >= num_ranks()) {
        TR_INDEX_ERROR("StagingParamPool rank out of range: " << rank);
    }
    if (slot < 0 || static_cast<size_t>(slot) >= bytes_per_rank_ / sizeof(float)) {
        TR_INDEX_ERROR("StagingParamPool slot out of range: " << slot);
    }
    float* base = static_cast<float*>(ptrs_[rank]);
    return base[slot];
}

int StagingParamPool::num_ranks() const {
    return static_cast<int>(ptrs_.size());
}
```

### 2.3 GlobalRegistry 集成

```cpp
// global_registry.h 新增
// 签名与 allocate_staging_memory 一致：gpu_ids 从内部获取
GlobalRegistry& allocate_staging_params(size_t bytes_per_rank = 256);
bool has_staging_params() const;
void* staging_params_ptr(int rank) const;
size_t staging_params_bytes() const;

// global_registry.cpp 新增
struct StagingParamsState {
    std::unique_ptr<StagingParamPool> pool;
    std::vector<int> gpu_ids;
    size_t bytes_per_rank = 0;
};
static StagingParamsState s_params_state;

GlobalRegistry& GlobalRegistry::allocate_staging_params(size_t bytes_per_rank) {
    // gpu_ids 从内部获取，与 allocate_staging_memory 风格一致
    // CPU 模式处理：fixed_gpu_ids_ 可能为空，仿照 allocate_staging_memory
    std::vector<int> ids = this->gpu_ids();
    if (ids.empty()) {
        int ws = world_size();
        ids.resize(ws, -1);  // CPU 模式：每个 rank gpu_id=-1 → malloc 分支
    }
    s_params_state.pool = std::make_unique<StagingParamPool>(ids, bytes_per_rank);
    s_params_state.gpu_ids   = ids;
    s_params_state.bytes_per_rank = bytes_per_rank;
    return *this;
}

bool GlobalRegistry::has_staging_params() const {
    return s_params_state.pool != nullptr;
}

void* GlobalRegistry::staging_params_ptr(int rank) const {
    if (!s_params_state.pool) {
        TR_RUNTIME_ERROR("StagingParamPool not allocated");
    }
    return s_params_state.pool->ptr(rank);
}

size_t GlobalRegistry::staging_params_bytes() const {
    return s_params_state.bytes_per_rank;
}
```

> **CPU 模式说明**：CPU 模式下 `gpu_ids_` = `{-1}`。`StagingParamPool` 构造时检测 `gpu_id < 0`，
> 使用 `std::malloc` 分配普通分页内存（用户指令："与 GPU 做法对齐，但分配的是普通分页内存"）。

### 2.4 compile_capture_simple 自动分配（GPU + CPU 通用）

[文件：src/task/task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp)，在 `compile_capture_simple()` 的 **`#ifdef TR_USE_CUDA` 块之前**（作为 GPU/CPU 公共路径）加入：

```cpp
void TaskBase::compile_capture_simple() {

    // ====== 自动检测 DTENSOR 节点并分配 StagingParamPool ======
    // GPU/CPU 公共路径：compile_capture_simple 无论设备模式均需分配
    {
        auto& reg = GlobalRegistry::instance();
        if (!reg.has_staging_params()) {
            bool need_params = false;
            for (const auto& [name, entry] : named_graphs_) {
                for (const auto& node : entry.graph.linear_nodes()) {
                    if (node.kind == GraphNode::Kind::RANGE &&
                        node.range_op == RangeOp::RANGE_H2D_COPY_DTENSOR) {
                        need_params = true;
                        break;
                    }
                }
            }
            if (need_params) {
                reg.allocate_staging_params(256);
            }
        }
    }
    // ============================================================

#ifdef TR_USE_CUDA
    if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
        // ... 原有 StagingBufferPool A/B 自动分配 + DTENSOR 预分配 + warmup + capture ...
```

**关键点**：
- StagingParamPool 分配在 GPU/CPU 公共路径（`#ifdef TR_USE_CUDA` 之前），确保 `--cpu` 测试也能获取分配
- StagingBufferPool A/B 自动分配仍保留在 `#ifdef TR_USE_CUDA ... is_gpu()` 内（仅 GPU 需要 pinned memory）
- CPU 模式下 `gpu_ids_ = {-1}`，`StagingParamPool` 内部走 `malloc` 分支

### 2.5 DTENSOR Kernel 重写

[文件：src/backend/ops/range/h2d_op.cpp](file:///r:/renaissance/src/backend/ops/range/h2d_op.cpp)，将 `launch_range_h2d_copy_dtensor_cuda`（L49-L79）替换为：

```cpp
/**
 * RANGE_H2D_COPY_DTENSOR 的 CUDA 捕获期 replay 函数
 *
 * 从 StagingParamPool 获取 per-rank 参数区指针作为 src，
 * 从 ArenaKeeper 获取 DTensor 设备指针作为 dst。
 * 每个 execution 传输 sizeof(float) = 4 字节（LR → data[0]）。
 *
 * Phase 1 限制：hardcoded slot=0（仅支持一个 FP32 参数 = LR）。
 * Phase 2 将槽位编码于 GraphNode，支持多参数传输。
 */
static void launch_range_h2d_copy_dtensor_cuda(
    const GraphNode& node,
    const MemoryPlan& /*mp*/,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));

    int si = state.get_or_register(stream);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    auto& reg = GlobalRegistry::instance();
    int rank = ctx.rank_for_context();

    if (!reg.has_staging_params()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: StagingParamPool not allocated. "
                        "Ensure compile_capture_simple() auto-allocates.");
    }

    if (node.output_ranges.empty()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: empty output_ranges");
    }

    const auto& seg = node.output_ranges[0];
    if (seg.size < sizeof(float)) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: output_ranges[0].size="
                        << seg.size << " < sizeof(float)");
    }

    void* src = reg.staging_params_ptr(rank);       // StagingParamPool[rank] base
    void* dst = ArenaKeeper::instance().ptr_at(rank, seg.offset);

    cudaError_t err = cudaMemcpyAsync(dst, src, sizeof(float),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR cudaMemcpyAsync failed: "
                        << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}
```

**核心变更总结**：

| 项目 | 旧实现 | 新实现 |
|------|--------|--------|
| SRC 指针 | `s_pinned_map[offset]`（全局单例） | `StagingParamPool.ptr(rank)`（per-rank） |
| 传输大小 | `seg.size`（DTensor 的总 slot_bytes） | `sizeof(float)`（固定 4 字节） |
| 分配时机 | `compile_capture_simple()` 按 offset 逐个分配 | `compile_capture_simple()` 一次性分配 256B × nRanks |
| 多 RANK 支持 | ❌ 所有 rank 共享同一 buffer | ✅ 每个 rank 独立 256B |
| 内存区域 | 与 `s_pinned_map` 混在 h2d_op.cpp 内部 | `StagingParamPool` 独立类，与 StagingBufferPool 对称 |
| CPU 路径 | no-op（不执行 memcpy） | `std::memcpy(dst, ParamPool[rank], 4)` host-to-host |
| 分配 TODO: 通用 | `#ifdef TR_USE_CUDA ... is_gpu()` 内 | `#ifdef TR_USE_CUDA` 之前（GPU+CPU 公共路径） |

#### 2.5.1 CPU Kernel 重写

[文件：src/backend/ops/range/h2d_op.cpp](file:///r:/renaissance/src/backend/ops/range/h2d_op.cpp)，将 `launch_range_h2d_copy_dtensor_cpu`（L157-L165）替换为：

```cpp
/**
 * RANGE_H2D_COPY_DTENSOR 的 CPU 实现
 *
 * GPU/CPU 路径对齐：均从 StagingParamPool[rank] 读取 data[0] 执行 memcpy。
 * CPU 模式使用普通分页内存（malloc），GPU 模式使用 pinned memory（cudaHostAlloc）。
 * StagingParamPool 在 compile_capture_simple 的公共路径中分配。
 */
static void launch_range_h2d_copy_dtensor_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    auto& reg = GlobalRegistry::instance();

    if (!reg.has_staging_params()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: StagingParamPool not allocated");
    }

    if (op_ctx->num_output_ranges == 0) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: empty output_ranges");
    }

    const auto& seg = op_ctx->output_ranges[0];
    if (seg.size < sizeof(float)) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: output_ranges[0].size="
                        << seg.size << " < sizeof(float)");
    }

    int rank = ctx.rank_for_context();
    void* src = reg.staging_params_ptr(rank);       // StagingParamPool[rank] base
    void* dst = ArenaKeeper::instance().ptr_at(rank, seg.offset);
    std::memcpy(dst, src, sizeof(float));
}
```

---

## 三、Kernel 修复（一）：A/B 算子 —— src 指针绑定 StagingBufferPool

### 3.1 核心原理

CUDA Graph 的 `cudaMemcpyAsync(H2D)` node 在 capture 时记录 src/dst 指针。**只要 capture 时 src 指向 `cudaHostAlloc` 分配的 pinned memory，且该地址在 capture→replay 期间不变，graph replay 就能正确读取数据。**

StagingBufferPool 用 `cudaMallocHost` 分配，地址固定，生命周期覆盖整个训练过程 → 完全满足。

**方案**：删掉 `s_placeholder_h2d`，在 capture 阶段直接从 StagingBufferPool 取真实 per-rank 指针。

### 3.2 数据布局

```
StagingBufferPool 单 rank 布局：

A 区 (offset = 0):
  [0,                   label_aligned)  → I_A_LABEL (050)
  [label_aligned,        per_zone)      → I_A_DATA  (051)

B 区 (offset = per_zone):
  [per_zone,             per_zone + label_aligned)  → I_B_LABEL (052)
  [per_zone + label_aligned, 2*per_zone)            → I_B_DATA  (053)
```

其中：
```
label_aligned = align_up_256(local_batch_size * sizeof(int32_t) + 16)

data_raw      = local_batch_size * resolution² * C_eff * elem_size
data_aligned  = align_up_256(data_raw + 16)
per_zone      = label_aligned + data_aligned

C_eff = (using_amp && num_color_channels == 3) ? 4 : num_color_channels
```

`label_aligned` 的计算公式与 [transfer_station.cpp](file:///r:/renaissance/src/data/transfer_station.cpp) 中的 `configure()` **完全一致** — 两者使用 GlobalRegistry 中的同一组参数。

### 3.3 修改后代码

[文件：src/backend/ops/range/h2d_op.cpp](file:///r:/renaissance/src/backend/ops/range/h2d_op.cpp)，将 `launch_range_h2d_copy_cuda`（L92-L127）替换为：

```cpp
/// 对齐辅助函数：先加 16 字节 XNNPACK padding，再对齐到 256 字节边界
/// 公式与 TransferStation::configure() 完全一致
static size_t align_up_256_with_padding(size_t raw) {
    return ((raw + 16 + 255) / 256) * 256;
}

/// 从 GlobalRegistry 参数计算 label_aligned
static size_t get_label_aligned() {
    auto& reg = GlobalRegistry::instance();
    int local_batch_size = reg.get_local_batch_size();
    size_t label_raw = static_cast<size_t>(local_batch_size) * sizeof(int32_t);
    return align_up_256_with_padding(label_raw);
}

/**
 * RANGE_H2D_COPY_A/B 的 CUDA 捕获期 replay 函数
 *
 * 从 GlobalRegistry 获取 StagingBufferPool 的真实 per-rank 指针作为 src，
 * 确保 CUDA Graph capture 期记录的 memcpy src 指向正确的 pinned memory。
 * 调用方必须在 compile 之前分配 StagingBufferPool（SimpleTask 由
 * compile_capture_simple() 自动检测并分配）。
 */
static void launch_range_h2d_copy_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));

    int si = state.get_or_register(stream);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    auto& reg = GlobalRegistry::instance();
    int rank = ctx.rank_for_context();

    if (!reg.has_staging_memory()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_A/B: StagingBufferPool not allocated. "
                        "Call GlobalRegistry::allocate_staging_memory() "
                        "or ensure compile_capture_simple() auto-allocates.");
    }

    uint8_t* staging_base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
    size_t per_zone = reg.staging_memory_size() / 2;
    size_t label_aligned = get_label_aligned();

    for (const auto& range : node.output_ranges) {
        auto [dst_off, dst_size] = mp.resolve_region_bounds(
            static_cast<Region>(range.start_region_id),
            static_cast<Region>(range.end_region_id));
        if (dst_size == 0) continue;

        void* dst = ArenaKeeper::instance().ptr_at(rank, dst_off);
        void* src = nullptr;

        Region start_region = static_cast<Region>(range.start_region_id);
        switch (start_region) {
            case Region::I_A_LABEL: src = staging_base; break;
            case Region::I_A_DATA:  src = staging_base + label_aligned; break;
            case Region::I_B_LABEL: src = staging_base + per_zone; break;
            case Region::I_B_DATA:  src = staging_base + per_zone + label_aligned; break;
            default:
                TR_DEVICE_ERROR("RANGE_H2D_COPY: unexpected region "
                                << static_cast<int>(start_region));
        }

        cudaError_t err = cudaMemcpyAsync(dst, src, dst_size,
                                          cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("RANGE_H2D_COPY cudaMemcpyAsync failed: "
                            << cudaGetErrorString(err));
        }
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}
```

### 3.4 compile_capture_simple 自动分配 StagingBufferPool

[文件：src/task/task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp)，在 `compile_capture_simple()` 的 `#ifdef TR_USE_CUDA` 块开头加入：

```cpp
void TaskBase::compile_capture_simple() {
#ifdef TR_USE_CUDA
    if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {

        // ====== 自动检测 A/B 传输节点并分配 StagingBufferPool ======
        bool need_staging = false;
        for (const auto& [name, entry] : named_graphs_) {
            for (const auto& node : entry.graph.linear_nodes()) {
                if (node.kind != GraphNode::Kind::RANGE) continue;
                if (node.range_op == RangeOp::RANGE_H2D_COPY_A ||
                    node.range_op == RangeOp::RANGE_H2D_COPY_B) {
                    need_staging = true;
                    break;
                }
            }
        }

        if (need_staging) {
            auto& reg = GlobalRegistry::instance();
            if (!reg.has_staging_memory()) {
                int batch_size = reg.get_local_batch_size();
                int resolution = reg.max_sample_resolution();
                int channels = reg.num_color_channels();
                bool using_amp = reg.using_amp();

                size_t elem_size = using_amp ? sizeof(uint16_t) : sizeof(float);
                int effective_c = (using_amp && channels == 3) ? 4 : channels;

                size_t label_aligned = ((static_cast<size_t>(batch_size)
                                        * sizeof(int32_t) + 16 + 255) / 256) * 256;
                size_t data_raw = static_cast<size_t>(batch_size) * resolution
                                  * resolution * effective_c * elem_size;
                size_t data_aligned = ((data_raw + 16 + 255) / 256) * 256;
                size_t per_zone = label_aligned + data_aligned;

                reg.allocate_staging_memory(per_zone * 2);
            }
        }
        // ============================================================

        // ... 原有 DTENSOR 预分配 + warmup + capture 代码继续 ...
```

**设计要点**：
- 幂等：若 `has_staging_memory() == true`（测试已手动分配），则跳过
- 自动：测试无需关心 buffer 大小公式，仅需在 `compile()` 前设置 GlobalRegistry 参数

### 3.5 修改文件总览（A/B 部分）

| # | 文件 | 操作 | 内容 |
|---|------|:--:|------|
| K-1 | `h2d_op.cpp` | 修改 | 删除 `s_placeholder_h2d`；新增 `get_label_aligned()`；重写 `launch_range_h2d_copy_cuda` 使用 `staging_memory_ptr(rank)` |
| K-2 | `task_base.cpp` | 修改 | `compile_capture_simple()` 增加 A/B 检测 + StagingBufferPool 自动分配 |

**仅两处修改，无需新增 GlobalRegistry 字段，无需修改 TransferStation。**

---

## 三、数学正确性测试（三个算子）

### 3.1 统一测试方法论

```
测试流程（所有算子通用）：

1. GlobalRegistry 配置参数（必须包含 local_batch_size + train_resolution + set_num_color_channels，
   compile_capture_simple 自动分配 StagingBufferPool 时依赖这些参数计算 buffer 大小，缺一不可）
2. SimpleTask 分配 DTensor → finalize_memory()
3. 构造包含 RANGE_H2D_COPY_XXX 节点的 ComputationGraph
4. task.add_graph(name, graph, StreamKind::TRANS)
5. task.compile()  （compile_capture_simple 自动分配 buffer）
6. 将已知数据填充到 StagingBufferPool 或 DTENSOR pinned buffer
7. task.run("graph_name")  （异步 CUDA Graph Launch）
8. 对每个 rank：task.fetch_from_rank(dt, rank)  （同步取回）
9. 计算 MSE + max|diff|，与预期值比较
```

### 3.2 test_h2d_copy_a.cpp — RANGE_H2D_COPY_A

#### 测试参数

```cpp
const int batch_size = 4;
const int resolution = 8;
const int channels = 1;
```
```

#### GlobalRegistry 配置

> **⚠️ 关键**：缺少 `train_resolution` 会导致 `compile_capture_simple` 自动分配 StagingBufferPool 时
> `max_sample_resolution()` 返回 0 → buffer 计算为极小值 → H2D 拷贝越界。

```cpp
GLOBAL_SETTING.train_resolution(resolution).val_resolution(resolution);
auto& reg = GlobalRegistry::instance();
reg.local_batch_size(batch_size);
reg.train_resolution(resolution);
reg.set_num_color_channels(channels);
```

#### DTensor 分配

```cpp
DTensor d_label = task.alloc(Shape{batch_size, 1, 1, 1}, DType::INT32, Region::I_A_LABEL);
DTensor d_data  = task.alloc(Shape{batch_size, 8, 8, 1},  DType::FP32,  Region::I_A_DATA);
```

#### 图构造

```cpp
ComputationGraph g;
{
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_H2D_COPY_A;
    node.output_ranges = {
        mp.region_range(Region::I_A_LABEL),
        mp.region_range(Region::I_A_DATA),
    };
    g.append(std::move(node));
}
task.add_graph("xfer_a", std::move(g), StreamKind::TRANS);
```

#### 数据准备与 StagingBuffer 填充

```cpp
// 准备主机参考数据
auto h_label = Tensor::zeros({4, 1, 1, 1}, DType::INT32);
for (int i = 0; i < 4; ++i) h_label.data<int32_t>()[i] = i * 10 + 7;  // 7, 17, 27, 37

auto h_data = Tensor::zeros({4, 8, 8, 1}, DType::FP32);  // 256 elements
for (int64_t i = 0; i < h_data.numel(); ++i) {
    h_data.data<float>()[i] = static_cast<float>(i % 997) * 0.001f;
}

// 填充各 rank 的 StagingBufferPool A 区
size_t label_raw = batch_size * sizeof(int32_t);
size_t label_aligned = ((label_raw + 16 + 255) / 256) * 256;

for (int rank = 0; rank < num_ranks; ++rank) {
    uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
    std::memcpy(base, h_label.data<int32_t>(), label_raw);
    std::memcpy(base + label_aligned, h_data.data<float>(),
                h_data.numel() * sizeof(float));
}
```

#### 执行与验证（MSE 指标）

```cpp
task.run("xfer_a");

bool all_pass = true;
for (int rank = 0; rank < num_ranks; ++rank) {
    auto out_label = task.fetch_from_rank(d_label, rank);
    auto out_data  = task.fetch_from_rank(d_data,  rank);

    // Label：逐元素精确比较（INT32）+ 跟踪 flag 用于输出
    bool label_ok = true;
    for (int64_t i = 0; i < h_label.numel(); ++i) {
        if (out_label.data<int32_t>()[i] != h_label.data<int32_t>()[i]) {
            label_ok = false;
            all_pass = false;
        }
    }

    // Data：MSE + max|diff|
    double mse = 0.0;
    double max_diff = 0.0;
    for (int64_t i = 0; i < h_data.numel(); ++i) {
        double diff = static_cast<double>(out_data.data<float>()[i]) -
                      static_cast<double>(h_data.data<float>()[i]);
        mse += diff * diff;
        double abs_diff = std::abs(diff);
        if (abs_diff > max_diff) max_diff = abs_diff;
    }
    mse /= h_data.numel();

    std::cout << "  Rank " << rank
              << " label_ok=" << (label_ok ? "true" : "false")
              << " data_MSE=" << std::scientific << mse
              << " max|diff|=" << max_diff;
    if (max_diff > 1e-6) { std::cout << "  FAIL"; all_pass = false; }
    std::cout << std::endl;
}
```

#### 期望输出

```
RANGE_H2D_COPY_A (N ranks): PASS
  - 所有 rank label 逐元素精确匹配
  - 所有 rank data MSE ≈ 0, max|diff| < 1e-6
```

### 3.3 test_h2d_copy_b.cpp — RANGE_H2D_COPY_B

与 test_h2d_copy_a 完全同构（参数 `batch_size=4, resolution=8, channels=1` 相同），差异仅在于：

| 项目 | test_h2d_copy_a | test_h2d_copy_b |
|------|:--:|:--:|
| DTensor Region | `I_A_LABEL`, `I_A_DATA` | `I_B_LABEL`, `I_B_DATA` |
| RangeOp | `RANGE_H2D_COPY_A` | `RANGE_H2D_COPY_B` |
| 图名 | `"xfer_a"` | `"xfer_b"` |
| Staging 填充偏移 | `base`（A 区起点） | `base + per_zone`（B 区起点） |

**B 区填充代码**：

> **⚠️ 关键**：与 test_h2d_copy_a 相同，缺少 `train_resolution` 会导致 StagingBufferPool 自动分配崩溃。

```cpp
GLOBAL_SETTING.train_resolution(resolution).val_resolution(resolution);
auto& reg = GlobalRegistry::instance();
reg.local_batch_size(batch_size);
reg.train_resolution(resolution);
reg.set_num_color_channels(channels);
// AMP 默认为 false，无需显式设置以使用 FP32 模式

size_t per_zone = reg.staging_memory_size() / 2;
uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
uint8_t* zone_b = base + per_zone;
std::memcpy(zone_b, h_label.data<int32_t>(), label_raw);
std::memcpy(zone_b + label_aligned, h_data.data<float>(),
            h_data.numel() * sizeof(float));
```

其余 DTensor 形状、数据生成、验证逻辑（含 `label_ok` 变量）与 test_h2d_copy_a 完全相同。

### 3.4 test_h2d_copy_dtensor.cpp — RANGE_H2D_COPY_DTENSOR

#### 设计要点

`RANGE_H2D_COPY_DTENSOR` 的新测试数据路径（基于 StagingParamPool 重构）：

1. Graph 构造时，用 `mp.get_dtensor(d_v.id)` 的手动 MemRange（含真实 offset + slot_bytes）— 不再使用 `mp.region_range()`
2. `task.compile()` → `compile_capture_simple` 自动检测 DTENSOR 节点 → 分配 StagingParamPool（256B × nRanks）
3. 将参考值写入 StagingParamPool：`reg.staging_params_ptr(rank)` → `reinterpret_cast<float*>(ptr)[0] = lr_value`
4. `task.run("h2d_dtensor")` → `cudaMemcpyAsync(dst, ParamPool[rank], sizeof(float), H2D)` → 4 字节从 ParamPool 传输到 DTensor
5. `task.fetch_from_rank(d_v, rank)` → 验证 DTensor 第一个 4 字节是否匹配参考值

#### 完整代码

```cpp
#include "renaissance.h"
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace tr;

int main(int argc, char* argv[]) {
    bool use_gpu = true;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") use_gpu = false;
        else if (arg == "--gpu") use_gpu = true;
    }
    if (use_gpu) GLOBAL_SETTING.use_gpu().auto_seed();
    else GLOBAL_SETTING.use_cpu().auto_seed();
    const int resolution = 1;
    GLOBAL_SETTING.train_resolution(resolution).val_resolution(resolution);

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    SimpleTask task;

    // 标量 DTensor：Shape{1,1,1,1} FP32 = 4 字节 = sizeof(float)
    // RANGE_H2D_COPY_DTENSOR 每次传输恰好 sizeof(float) = 4 字节
    Shape shape{1, 1, 1, 1};
    DTensor d_v = task.alloc(shape, DType::FP32, Region::S_SCALAR_FP32);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    // ==== Graph 构造：必须用 mp.get_dtensor(id) 获取真实 offset + slot_bytes ====
    ComputationGraph g;
    {
        const auto& dt_info = mp.get_dtensor(d_v.id);
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_DTENSOR;
        node.output_ranges = {
            MemRange{static_cast<uint64_t>(dt_info.offset()),
                     static_cast<uint64_t>(dt_info.slot_bytes()),
                     static_cast<int32_t>(dt_info.region),
                     static_cast<int32_t>(dt_info.region)}
        };
        g.append(std::move(node));
    }
    task.add_graph("h2d_dtensor", std::move(g), StreamKind::TRANS);

    // compile() 自动检测 DTENSOR 节点 → 分配 StagingParamPool
    task.compile();

    // ==== 准备 per-rank 不同参考值，写入 StagingParamPool ====
    // Phase 1 限制：仅使用 slot[0]（LR data[0]）
    // per-rank 不同值：检测 cross-rank 指针混淆
    std::vector<float> expected_values(num_ranks);
    for (int rank = 0; rank < num_ranks; ++rank) {
        expected_values[rank] = 0.001f * static_cast<float>(rank + 1);
        if (reg.has_staging_params()) {
            float* param = static_cast<float*>(reg.staging_params_ptr(rank));
            param[0] = expected_values[rank];  // data[0] = LR
        }
    }

    // 执行异步传输 CUDA Graph（每个 rank 从自己的 StagingParamPool[rank] 传输 4B）
    task.run("h2d_dtensor");

    // 同步取回 → 与各自 per-rank expected 比较
    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        auto out = task.fetch_from_rank(d_v, rank);

        float fetched = out.data<float>()[0];
        float expected = expected_values[rank];
        double diff = static_cast<double>(fetched) -
                      static_cast<double>(expected);
        double mse = diff * diff;          // 单标量 MSE = diff²

        std::cout << "  Rank " << rank
                  << " fetched=" << std::fixed << fetched
                  << " expected=" << expected
                  << " diff=" << std::scientific << std::abs(diff);
        if (std::abs(diff) > 1e-6) {
            std::cout << "  FAIL";
            all_pass = false;
        }
        std::cout << std::endl;
    }

    std::cout << "\nRANGE_H2D_COPY_DTENSOR (" << num_ranks << " ranks): "
              << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}
```

#### 期望输出

```
RANGE_H2D_COPY_DTENSOR (N ranks): PASS
  - 每个 rank 的标量 DTensor[0] 精确匹配 StagingParamPool[rank][0] = 0.001×(rank+1)
  - |diff| < 1e-6
```

**与旧版测试的关键差异**：

| 项目 | 旧版（已废弃） | 新版 |
|------|------|------|
| Graph 构造 | `mp.region_range(d_v.region)` → offset=0, size=0 | `mp.get_dtensor(id)` → 真实 offset + slot_bytes |
| 数据源 | `fill_transfer_buffer(host, dt)` → `s_pinned_map` | `StagingParamPool.set_param(rank, 0, value)` |
| 传输大小 | `seg.size`（slot_bytes） | `sizeof(float)`（固定 4B） |
| 验证范围 | 整个 DTensor 256 元素 | 首个 FP32 标量（data[0]） |
| 多 RANK | 全局单例共享 | 每个 rank 独立 256B |

---

## 四、性能测试（仅 RANGE_H2D_COPY_A）

### 4.1 `perf_h2d_copy_a.cpp` — H2D 带宽与延迟

#### 设计策略

| 指标 | 方法 | 公式 |
|------|------|------|
| H2D **延迟** (latency) | 小数据量（1 batch），Warmup 排除首次编译开销 | `total_time / iter_count` |
| H2D **带宽** (bandwidth) | 大数据量（max batch / max resolution），测量总耗时 | `total_bytes / avg_time` |
| 稳定性 | 多次迭代取均值 + 标准差 | `N_iter ≥ 100` |

#### 测试代码

```cpp
/**
 * @file perf_h2d_copy_a.cpp
 * @brief RANGE_H2D_COPY_A 异步 H2D 传输性能测试
 *
 * 测量指标:
 *   - H2D 延迟 (us/iter): 小 batch 的传输耗时
 *   - H2D 有效带宽 (GB/s): bytes / time
 *   - 迭代稳定性: mean ± stddev
 *
 * 使用方法:
 *   perf_h2d_copy_a --gpu [--amp] [--warmup N] [--iter N]
 *   perf_h2d_copy_a --cpu  (CPU 模式基准对比)
 *
 * 默认参数（按用户要求 RL_REV2.md）:
 *   512 batch size, 224×224 resolution, 3 color channels
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include <numeric>

using namespace tr;
// microsecond 时间单位
using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

/// 对齐辅助函数
static size_t align256(size_t raw) {
    return ((raw + 16 + 255) / 256) * 256;
}

/// 计算 StagingBufferPool per_zone 大小
static size_t calc_per_zone(int n, int res, int ch, bool amp) {
    int c_eff = (amp && ch == 3) ? 4 : ch;
    size_t es = amp ? 2 : 4;
    size_t la = align256(n * 4);
    size_t da = align256(static_cast<size_t>(n) * res * res * c_eff * es);
    return la + da;
}

int main(int argc, char* argv[]) {
    bool use_gpu = true;
    bool use_amp = false;
    int warmup = 20;
    int iterations = 200;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") use_gpu = false;
        else if (arg == "--gpu") use_gpu = true;
        else if (arg == "--amp") use_amp = true;
        else if (arg == "--warmup" && i+1 < argc) warmup = std::stoi(argv[++i]);
        else if (arg == "--iter"   && i+1 < argc) iterations = std::stoi(argv[++i]);
    }

    if (use_gpu) GLOBAL_SETTING.use_gpu().auto_seed();
    else GLOBAL_SETTING.use_cpu().auto_seed();
    GLOBAL_SETTING.train_resolution(resolution).val_resolution(resolution);

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    // 性能测试参数（按用户要求 RL_REV2.md 默认值）
    const int batch_size = 512;
    const int resolution = 224;
    const int channels = 3;
    const bool is_amp = use_amp;

    reg.local_batch_size(batch_size);
    reg.train_resolution(resolution);
    reg.set_num_color_channels(channels);
    if (use_amp) reg.amp(true);

    SimpleTask task;

    int effective_c = (is_amp && channels == 3) ? 4 : channels;
    DType dtype = is_amp ? DType::FP16 : DType::FP32;
    Shape shape{batch_size, effective_c, resolution, resolution};
    DTensor d_data = task.alloc(shape, dtype, Region::I_A_DATA);
    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g;
    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_A;
        node.output_ranges = { mp.region_range(Region::I_A_DATA) };
        g.append(std::move(node));
    }
    task.add_graph("xfer_a_perf", std::move(g), StreamKind::TRANS);
    task.compile();

    // 填充系统 StagingBufferPool（compile_capture_simple 已自动分配）
    size_t elem_size = is_amp ? sizeof(uint16_t) : sizeof(float);
    size_t total_transfer_bytes = static_cast<size_t>(shape.numel()) * elem_size;
    size_t per_zone = reg.staging_memory_size() / 2;
    size_t label_aligned = align256(batch_size * 4);

    for (int rank = 0; rank < num_ranks; ++rank) {
        uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
        float* data = reinterpret_cast<float*>(base + label_aligned);
        for (int64_t i = 0; i < shape.numel(); ++i) data[i] = 1.0f;
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(iterations);

    // Warmup（排除首次 CUDA Graph launch 开销）
    for (int i = 0; i < warmup; ++i) {
        task.run("xfer_a_perf");
    }

    // 正式测量
    for (int i = 0; i < iterations; ++i) {
        auto t0 = Clock::now();
        task.run("xfer_a_perf");
        auto t1 = Clock::now();

        double us_elapsed = static_cast<double>(
            std::chrono::duration_cast<us>(t1 - t0).count());
        latencies_us.push_back(us_elapsed);
    }

    // 统计
    double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    double mean = sum / iterations;
    double variance = 0.0;
    for (double v : latencies_us) {
        double d = v - mean;
        variance += d * d;
    }
    variance /= iterations;
    double stddev = std::sqrt(variance);

    double total_mb = static_cast<double>(total_transfer_bytes) / (1024.0 * 1024.0);
    double bandwidth_gb_s = total_mb / (mean / 1e6) / 1024.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== RANGE_H2D_COPY_A Performance ===" << std::endl;
    std::cout << "  Batch size:       " << batch_size << std::endl;
    std::cout << "  Resolution:       " << resolution << "x" << resolution << std::endl;
    std::cout << "  Channels:         " << channels << std::endl;
    std::cout << "  Transfer bytes:   " << total_transfer_bytes
              << " (" << total_mb << " MB)" << std::endl;
    std::cout << "  Num ranks:        " << num_ranks << std::endl;
    std::cout << "  Warmup iters:     " << warmup << std::endl;
    std::cout << "  Measure iters:    " << iterations << std::endl;
    std::cout << "  Mean latency:     " << mean << " us/iter" << std::endl;
    std::cout << "  Std dev:          " << stddev << " us" << std::endl;
    std::cout << "  Effective BW:     " << bandwidth_gb_s << " GB/s" << std::endl;

    // 带宽合理性校验（PCIe Gen4 x16 理论 ~32 GB/s，40%~80% 合理范围）
    double min_expected = num_ranks * 2.0;   // 至少 2 GB/s per rank
    if (bandwidth_gb_s < min_expected) {
        std::cout << "  WARNING: Bandwidth below expected minimum ("
                  << min_expected << " GB/s)" << std::endl;
    }

    return 0;
}
```

#### 期望输出示例

```
=== RANGE_H2D_COPY_A Performance ===
  Batch size:       512
  Resolution:       224x224
  Channels:         3
  Amp:              false
  Transfer bytes:   77070336 (73.50 MB)
  Num ranks:        1
  Warmup iters:     20
  Measure iters:    200
  Mean latency:     7200.00 us/iter
  Std dev:          150.00 us
  Effective BW:     10.21 GB/s
```

---

## 五、CMakeLists.txt 注册

[文件：tests/correction/CMakeLists.txt](file:///r:/renaissance/tests/correction/CMakeLists.txt)，在文件末尾追加：

```cmake
# ============================================================
# RANGE_H2D_COPY 系列：数学正确性测试 + 性能测试
# ============================================================

set(H2D_CORRECTION_TESTS
    test_h2d_copy_a
    test_h2d_copy_b
    test_h2d_copy_dtensor
)

foreach(test_name IN LISTS H2D_CORRECTION_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE renaissance)
    target_compile_definitions(${test_name} PRIVATE TR_LOG_LEVEL=1)
    if(TR_USE_CUDA)
        setup_gpu_runtime_env(${test_name})
    endif()
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
        WIN32_EXECUTABLE FALSE
    )
    if(MSVC)
        target_compile_options(${test_name} PRIVATE /W4 /utf-8
                               /wd4244 /wd4505 /wd4458 /wd4127)
    else()
        target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic
                               -Wno-unused-function)
    endif()
    message(STATUS "  - ${test_name}: RANGE_H2D_COPY correction test")
endforeach()

# TRANSFER_A 性能测试（单独目录或 tests/perf/）
add_executable(perf_h2d_copy_a perf_h2d_copy_a.cpp)
target_link_libraries(perf_h2d_copy_a PRIVATE renaissance)
target_compile_definitions(perf_h2d_copy_a PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(perf_h2d_copy_a)
endif()
set_target_properties(perf_h2d_copy_a PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(perf_h2d_copy_a PRIVATE /W4 /utf-8
                           /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(perf_h2d_copy_a PRIVATE -Wall -Wextra -Wpedantic
                           -Wno-unused-function)
endif()
message(STATUS "  - perf_h2d_copy_a: RANGE_H2D_COPY_A perf test")
```

---

## 六、完整修改清单

### 6.1 Kernel 修复

| # | 文件 | 操作 | 内容 |
|---|------|:--:|------|
| **F1** | `h2d_op.cpp` | 修改 | 删除 `s_placeholder_h2d` + `s_pinned_map` + `lookup_pinned_for_capture` + `get_dtensor_pinned_buffer`（旧 DTENSOR 实现全部移除）；新增 `#include "renaissance/core/global_registry.h"`、`#include "renaissance/core/staging_param_pool.h"`；新增 `get_label_aligned()`、`align_up_256_with_padding()`；重写 `launch_range_h2d_copy_cuda`（A/B）和 `launch_range_h2d_copy_dtensor_cuda`（StagingParamPool）；重写 `launch_range_h2d_copy_dtensor_cpu`（std::memcpy） |
| **F2** | `task_base.cpp` | 修改 | `compile_capture_simple()`: **删除旧 DTENSOR 预分配循环（L263-271）**；增加 A/B 检测 + StagingBufferPool 自动分配；增加 DTENSOR 检测 + StagingParamPool 自动分配（`#ifdef TR_USE_CUDA` 之前）；**删除 `fill_transfer_buffer` 实现（L1021-1062）** |
| **F3** | `staging_param_pool.h` | **新增** | StagingParamPool 类声明（`ptr()`, `set_param()`, `param()`） |
| **F4** | `staging_param_pool.cpp` | **新增** | StagingParamPool 实现（GPU: cudaHostAlloc, CPU: malloc） |
| **F5** | `global_registry.h` | 修改 | 新增 `allocate_staging_params()`, `has_staging_params()`, `staging_params_ptr()` |
| **F6** | `global_registry.cpp` | 修改 | 新增 `s_params_state` 静态状态 + `allocate_staging_params` 实现（含空 gpu_ids 处理） |

### 6.2 旧符号删除

| # | 文件 | 操作 | 内容 |
|---|------|:--:|------|
| **D1** | `op_registry.h:L121` | 删除 | `void* get_dtensor_pinned_buffer(size_t, size_t);` 声明 |
| **D2** | `task_base.h:L120` | 删除 | `void fill_transfer_buffer(const Tensor&, const DTensor&);` 声明 |
| **D3** | `dual_graph.cpp` | 修改 | L132 处 `fill_transfer_buffer` 调用注释掉，添加 `// TODO(BROKEN): fill_transfer_buffer removed. Replace with RANGE_H2D_COPY_A/B + StagingBufferPool for >256B DTensor` |

> **dual_graph.cpp 兼容处理**：`dual_graph.cpp:132` 中 `fill_transfer_buffer(host_b_next[0], d_buf_b)` 传输的 `d_buf_b`
> 是 1024-element FP32 tensor（4KB），超出 StagingParamPool 的 256B 容量。两种处理方式：
> - 若 d_buf_b 是标量参数 → 改为 `StagingParamPool.set_param(rank, 0, val)` + RANGE_H2D_COPY_DTENSOR
> - 若 d_buf_b 是完整 batch 数据 → 改用 `RANGE_H2D_COPY_A/B` + StagingBufferPool（更大容量）
> 
> **Phase 1 代码级操作**：在 `dual_graph.cpp:132` 注释掉该行，并在行前添加 TODO 注释，防止编译断裂。

### 6.3 测试文件

| # | 文件 | 类型 | 测试算子 |
|---|------|:--:|------|
| **T1** | `test_h2d_copy_a.cpp` | 数学正确性 | `RANGE_H2D_COPY_A` |
| **T2** | `test_h2d_copy_b.cpp` | 数学正确性 | `RANGE_H2D_COPY_B` |
| **T3** | `test_h2d_copy_dtensor.cpp` | 数学正确性 | `RANGE_H2D_COPY_DTENSOR` |
| **T4** | `perf_h2d_copy_a.cpp` | 性能测试 | `RANGE_H2D_COPY_A` |

### 6.4 构建系统

| # | 文件 | 操作 | 内容 |
|---|------|:--:|------|
| **B1** | `tests/correction/CMakeLists.txt` | 修改 | 追加 4 个测试 targets |
| **B2** | `src/CMakeLists.txt` | 修改 | 追加 `src/core/staging_param_pool.cpp` 到源文件列表（否则链接缺少 `StagingParamPool` 符号） |

**总计：14 个文件（8 修改 + 6 新增），新增 1 个基础设施类 StagingParamPool。**

---

## 七、实施顺序

```
Phase 1: Kernel 修复（~1h）
  1. F1: h2d_op.cpp — 重写 launch_range_h2d_copy_cuda
  2. F2: task_base.cpp — compile_capture_simple 自动分配 StagingBufferPool
  └→ 验证：编译通过，不引入新 warning

Phase 2: 数学正确性测试（~2h）
  3. T1: test_h2d_copy_a.cpp  → 编译运行 → PASS
  4. T2: test_h2d_copy_b.cpp  → 编译运行 → PASS
  5. T3: test_h2d_copy_dtensor.cpp → 编译运行 → PASS

Phase 3: 性能测试（~1h）
  6. T4: perf_h2d_copy_a.cpp  → 编译运行 → 带宽合理

Phase 4: 构建集成（~0.5h）
  7. B1: CMakeLists.txt → cmake --build → 全部编译通过

总工时: ~5.5h (新增 StagingParamPool 类 + GlobalRegistry 集成)
```

---

## 八、成功标准

### 8.1 编译

```bash
cmake --build build --target test_h2d_copy_a test_h2d_copy_b \
                                      test_h2d_copy_dtensor perf_h2d_copy_a
```

### 8.2 数学正确性

```bash
./build/bin/tests/correction/test_h2d_copy_a --gpu
# 期望: RANGE_H2D_COPY_A: PASS
#       所有 rank MSE ≈ 0, max|diff| < 1e-6

./build/bin/tests/correction/test_h2d_copy_b --gpu
# 期望: RANGE_H2D_COPY_B: PASS
#       所有 rank MSE ≈ 0, max|diff| < 1e-6

./build/bin/tests/correction/test_h2d_copy_dtensor --gpu
# 期望: RANGE_H2D_COPY_DTENSOR: PASS
#       StagingParamPool[rank][0] 值精确传输，|diff| < 1e-6
```

### 9.3 性能

```bash
./build/bin/tests/correction/perf_h2d_copy_a --gpu      # FP32, 512bs
./build/bin/tests/correction/perf_h2d_copy_a --gpu --amp  # FP16/AMP
./build/bin/tests/correction/perf_h2d_copy_a --cpu        # CPU 基准
# 期望:
#   - 延迟 < 10000 us/iter (batch=512, res=224, C=3, FP32)
#   - 有效带宽 > 2 GB/s per rank
#   - 标准差 < 均值 * 10%
```

---

## 九、风险矩阵

| # | 风险 | 等级 | 缓解 |
|---|------|:--:|------|
| R1 | `get_label_aligned()` 与 TransferStation 公式不一致（`+16` 是对齐到 16 字节边界） | 🟢 | 公式完全相同：`align_up_256(raw + 16)`。若 TransferStation 未来修改公式，`h2d_op.cpp` 需同步。 |
| R2 | DeepLearningTask 路径中 `launch_range_h2d_copy_cuda` 的 capture 发生在 `pre_capture()` 中，此时 StagingBufferPool 已分配 | 🟢 | `DeepLearningTask::on_prepare()` 在 `compile_impl:235-237` 调用 Preprocessor 分配 StagingBufferPool，在 `pre_capture:243` 之前。已验证。 |
| R3 | CPU 模式下 `launch_range_h2d_copy_dtensor_cpu` 使用 `std::memcpy`，与 GPU 路径行为一致（均从 StagingParamPool 读取）。 | 🟢 | [§2.5.1](file:///r:/renaissance/UTK_FINAL.md#L346-L380) 已重写 CPU kernel。`--cpu` 测试可直接运行，无需特殊处理。 |
| R4 | 性能测试的 StagingBufferPool 由 `compile_capture_simple` 自动分配，需确保 GlobalRegistry 参数在 compile 前正确设置 | 🟢 | 性能测试在 `task.compile()` 之前调用 `reg.local_batch_size(batch_size)` 等设置。`get_label_aligned()` 从 GlobalRegistry 读取参数。 |
| R5 | A/B CPU kernel `launch_range_h2d_copy_cpu` 仍为 zero-fill（写入全零值，非 StagingBufferPool 数据） | 🟡 | `--cpu` 模式下 A/B 测试无法验证数据内容。建议 `--cpu` 仅用于 zero-fill 基准对比，或待后续统一修复 CPU 路径实现与 GPU 对齐的 `cudaMemcpyAsync`→`std::memcpy`。 |

---

## 十、与 UTK1/2/3 的关键决策对照

| 议题 | UTK1 | UTK2 | UTK3 | **UTK_FINAL** |
|------|------|------|------|:--|
| A/B src 指针修复 | GlobalRegistry 新字段 | 现场计算 `label_aligned` | 现场计算 | **UTK2 方案**（无耦合，自包含） |
| StagingBufferPool 分配 | 测试显式分配 | `compile_capture_simple` 自动分配 | 测试显式分配 | **UTK2 方案**（自动化） |
| DTENSOR 数据源 | `s_pinned_map` 全局单例 | per-rank vector | 保持单 map | **StagingParamPool per-rank**（用户指令） |
| DTENSOR 传输大小 | seg.size（slot_bytes） | seg.size | seg.size | **sizeof(float)**（固定 4B） |
| 测试验证指标 | max\|diff\| | 精确逐元素 | max\|diff\| | **MSE + max\|diff\| 双重指标** |
| 性能测试默认参数 | 128bs/3ch/224res | 未涉及 | 未涉及 | **512bs/3ch/224res** + --amp 选项 |
| 修改文件总数 | 10 | 10 | 10 | **14**（含 StagingParamPool + 旧符号删除 + dual_graph 兼容） |

> 审查记录：RL_REV2.md（3 问题 → 全部修复）、YD.md（3 问题 → 全部修复）、MW.md（4 问题 + 2 建议 → 全部修复）。

---

## 附录 A：StagingBufferPool 容量公式（参考）

```
label_raw      = local_batch_size × sizeof(int32_t)
label_aligned  = align_up_256(label_raw + 16)

C_eff          = (using_amp && channels == 3) ? 4 : channels
elem_size      = using_amp ? sizeof(uint16_t) : sizeof(float)
data_raw       = local_batch_size × resolution² × C_eff × elem_size
data_aligned   = align_up_256(data_raw + 16)

per_zone       = label_aligned + data_aligned
total          = per_zone × 2    (A + B)
```

## 附录 B：关键代码位置索引

| 组件 | 文件 | 行号 |
|------|------|------|
| 算子枚举 | `op_kind.h` | 248-250 |
| GraphId 枚举 | `computation_graph.h` | 74-75 |
| StreamKind::TRANS | `types.h` | 220 |
| Region::I_A_LABEL/DATA | `types.h` | 307-310 |
| A/B CUDA launcher | `h2d_op.cpp` | 92-127 |
| DTENSOR CUDA launcher | `h2d_op.cpp` | 49-79 |
| get_dtensor_pinned_buffer | `h2d_op.cpp` | 167-184 |
| fill_transfer_buffer | `task_base.cpp` | 1021-1062 |
| compile_capture_simple | `task_base.cpp` | 256-480 |
| run(a, b) dual-graph | `task_base.cpp` | 711-789 |
| append_h2d (Compiler) | `compiler.cpp` | 1340-1364 |
| test_sgd_weight 参考 | `test_sgd_weight.cpp` | 全文件 |