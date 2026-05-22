# UTK2：RANGE 异步传输三大算子实现与测试方案（多 RANK 原生设计）

> 综合 TRA.md / TRX1.md / TRX2.md / TRX3.md 及当前代码审查
> 修订：2026-05-16 — 全部设计 natively 支持多 RANK

---

## 一、现状综合诊断

### 1.1 五层基础设施状态

| 层级 | 组件 | 文件 | 状态 |
|------|------|------|:--:|
| L1 枚举 | GraphId::TRANSFER_A/B | `computation_graph.h` | ✅ |
| L1 枚举 | RangeOp::H2D_COPY_A/B/DTENSOR | `op_kind.h` | ✅ |
| L1 枚举 | StreamKind::TRANS / Region::I_A_*/I_B_* | `types.h` | ✅ |
| L2 Backend | `register_op_range_h2d()` | `h2d_op.cpp` | ⚠️ A/B 指针未绑定 |
| L2 Backend | `get_dtensor_pinned_buffer()` | `h2d_op.cpp` | ⚠️ 全局单例，非 per-rank |
| L3 Compiler | `append_h2d(TRANSFER_A/B)` | `compiler.cpp:1340` | ✅ |
| L4 CUDA Graph | `CapturedGraph::capture()` / `pre_capture()` | `captured_graph.cpp` | ✅ |
| L5 数据层 | `StagingBufferPool` / `TransferStation` | `staging_buffer_pool.cpp` | ✅ |
| 运行时 | GraphSlot + GpuExecTable | `deep_learning_task.cpp` | ✅ |

### 1.2 当前代码四大关键问题

**问题 1：A/B 算子的 src 指针是 4096 字节 placeholder，未绑定 StagingBufferPool**

`h2d_op.cpp:104-116`：
```cpp
static void* s_placeholder_h2d = nullptr;
if (!s_placeholder_h2d) {
    cudaHostAlloc(&s_placeholder_h2d, 4096, cudaHostAllocDefault);
}
// capture 阶段 src = s_placeholder_h2d（固定 4KB）
```

- CUDA Graph 的 memcpy node 在 capture 时固定记录 src 指针
- 运行时没有任何 `cudaGraphExecMemcpyNodeSetParams` 调用替换该指针
- 实际传输的是 4096 字节未初始化内存中的垃圾数据
- 即使被替换，4096 字节也远小于典型 per_zone（数 MB 级）

**问题 2：SimpleTask 的 `compile_capture_simple()` 不为 A/B 预分配 StagingBufferPool**

```cpp
// task_base.cpp:263-271
for (const auto& node : entry.graph.linear_nodes()) {
    if (node.range_op != RangeOp::RANGE_H2D_COPY_DTENSOR) continue;  // ← 只处理 DTENSOR
    get_dtensor_pinned_buffer(seg.offset, seg.size);
}
```

- 对 H2D_COPY_A/B 节点**没有任何预分配或检查**
- SimpleTask 不经过 Preprocessor/TransferStation，不会自动分配 StagingBufferPool
- 若测试代码未手动分配，capture 阶段将拿不到正确 src 指针

**问题 3：DTENSOR 的 pinned buffer 是全局单例，非 per-rank**

`h2d_op.cpp:35-36`：
```cpp
static std::unordered_map<size_t, PinnedBuffer> s_pinned_map;
```

- key 仅为 `device_offset`，不含 `rank`
- 所有 rank 的 CUDA Graph capture 记录的是**同一个 host 指针**
- 多 rank 场景下，若各 rank 写入不同数据（如不同 LR），会互相覆盖
- `compile_capture_simple()` 的预分配只在 rank loop 之前执行一次，仅分配一个 buffer

**问题 4：DeepLearningTask 每 batch 同步所有 stream，未释放 overlap 潜力**

`deep_learning_task.cpp:770-773`：
```cpp
cudaGraphLaunch(g_xfer, s_trans);
cudaGraphLaunch(g_deep, s_comp1);
cudaStreamSynchronize(s_comp1);
cudaStreamSynchronize(s_comp2);
cudaStreamSynchronize(s_comp3);
cudaStreamSynchronize(s_trans);  // ← 阻塞传输流
```

- 传输图与计算图虽分属不同 stream，但每轮都同步全部 stream
- A/B 双缓冲的设计天然支持「下一 batch 传输」与「当前 batch 计算」并行
- 当前同步策略偏保守，overlap 潜力未释放（功能正确，性能未达最优）

---

## 二、核心设计决策

### 决策 1：A/B 算子 —— capture 阶段直接使用 StagingBufferPool 真实指针

**不采用** `cudaGraphExecMemcpyNodeSetParams` 运行时替换方案。理由：
1. 实现复杂，需遍历 graph 节点、保存 node handle、运行时批量替换
2. 用户明确要求 "StagingBufferPool 必须在 compile 之初就已分配"（TRA.md 用户补充 C1）
3. **最简单可靠的方案**：compile 前确保 StagingBufferPool 已分配，capture 阶段直接从 GlobalRegistry 取真实指针

```cpp
auto& reg = GlobalRegistry::instance();
void* staging_base = reg.staging_memory_ptr(rank);  // per-rank 独立指针
size_t per_zone = reg.staging_memory_size() / 2;

// I_A_LABEL → base + 0
// I_A_DATA  → base + label_aligned
// I_B_LABEL → base + per_zone
// I_B_DATA  → base + per_zone + label_aligned
```

**label_aligned 的计算**：直接从 GlobalRegistry 取参数现场计算（batch_size、resolution、channels、amp），公式与 TransferStation 完全一致。无需新增 GlobalRegistry 字段。

### 决策 2：DTENSOR 算子 —— pinned buffer 改为 per-rank

将全局 `s_pinned_map` 改为 `std::vector<std::unordered_map<size_t, PinnedBuffer>>`，索引为 `rank`。

**理由**：
1. 用户明确要求 "每个 RANK 对应一个" pinned buffer（TRA.md 用户补充 C4）
2. DeepLearningTask 的 per-batch LR 更新需要各 rank 独立
3. 测试代码从一开始就要验证多 rank 并行传输

**修改范围**：
- `h2d_op.cpp`：内部数据结构 + `launch_range_h2d_copy_dtensor_cuda` 使用 `ctx.rank_for_context()`
- `op_registry.h`：新增 `get_dtensor_pinned_buffer_for_rank(int rank, ...)` 公开 API
- `task_base.h/.cpp`：`fill_transfer_buffer` 增加 `int rank = 0` 默认参数
- `task_base.cpp`：`compile_capture_simple()` 为所有 rank 预分配 DTENSOR pinned buffer

### 决策 3：SimpleTask compile 路径 —— 自动检测并分配 StagingBufferPool

在 `compile_capture_simple()` 开头：
1. 遍历所有 graph 的 RANGE 节点
2. 若检测到 H2D_COPY_A/B 且 `has_staging_memory() == false`
3. 自动计算 buffer 大小并调用 `allocate_staging_memory()`

这样测试代码无需手动分配 StagingBufferPool，同时保留测试手动分配的能力（幂等设计）。

### 决策 4：Overlap 测试 —— 利用 `run(a,b)` 双图并发接口

`TaskBase::run(a,b)` 的实现：
```cpp
// 同时 launch 两张图到各自 stream
cap_a->second.launch(rank, stream_a);
cap_b->second.launch(rank, stream_b);
// 然后同步两个 stream
ctx.synchronize_stream(entry_a.stream);
ctx.synchronize_stream(entry_b.stream);
```

这已经支持两张图的并发执行。测试通过构造「传输图 + 计算图」，用 `run("xfer", "compute")` 验证数据正确性，即可间接证明 overlap 机制工作正常。

---

## 三、代码修改方案

### 修改 1：`h2d_op.cpp` —— A/B 算子使用真实 StagingBufferPool 指针 + DTENSOR per-rank

**文件**：`src/backend/ops/range/h2d_op.cpp`

**A. A/B 算子部分**：

```cpp
#ifdef TR_USE_CUDA

// 辅助函数：计算 label_aligned（与 TransferStation 公式一致）
static size_t calculate_label_aligned() {
    auto& reg = GlobalRegistry::instance();
    int batch_size = reg.get_local_batch_size();
    size_t label_raw = static_cast<size_t>(batch_size) * sizeof(int32_t);
    return utils::align_up_256(label_raw + 16);
}

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

    int rank = ctx.rank_for_context();
    auto& reg = GlobalRegistry::instance();

    void* staging_base = nullptr;
    size_t per_zone = 0;
    size_t label_aligned = 0;
    bool use_real_buffer = false;

    if (reg.has_staging_memory()) {
        staging_base = reg.staging_memory_ptr(rank);
        size_t total = reg.staging_memory_size();
        per_zone = total / 2;
        label_aligned = calculate_label_aligned();
        use_real_buffer = true;
    } else {
        // Fallback：向后兼容，使用 placeholder
        static void* s_fallback = nullptr;
        if (!s_fallback) {
            cudaHostAlloc(&s_fallback, 4096, cudaHostAllocDefault);
        }
        staging_base = s_fallback;
        per_zone = 4096;
    }

    for (const auto& range : node.output_ranges) {
        auto [dst_off, dst_size] = mp.resolve_region_bounds(
            static_cast<Region>(range.start_region_id),
            static_cast<Region>(range.end_region_id));
        if (dst_size == 0) continue;

        void* dst = ArenaKeeper::instance().ptr_at(rank, dst_off);
        void* src = nullptr;

        Region start_region = static_cast<Region>(range.start_region_id);
        switch (start_region) {
            case Region::I_A_LABEL:
                src = staging_base;
                break;
            case Region::I_A_DATA:
                src = static_cast<uint8_t*>(staging_base) + label_aligned;
                break;
            case Region::I_B_LABEL:
                src = static_cast<uint8_t*>(staging_base) + per_zone;
                break;
            case Region::I_B_DATA:
                src = static_cast<uint8_t*>(staging_base) + per_zone + label_aligned;
                break;
            default:
                TR_DEVICE_ERROR("RANGE_H2D_COPY: unexpected region "
                                << static_cast<int>(start_region));
        }

        // 安全校验：若使用真实 buffer，确保不越界
        if (use_real_buffer) {
            size_t region_offset = static_cast<uint8_t*>(src) - static_cast<uint8_t*>(staging_base);
            if (region_offset + dst_size > per_zone) {
                TR_DEVICE_ERROR("RANGE_H2D_COPY: region overflow! offset="
                                << region_offset << " size=" << dst_size
                                << " per_zone=" << per_zone);
            }
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

#endif // TR_USE_CUDA
```

**B. DTENSOR 算子 per-rank 改造部分**：

```cpp
// 删除旧的：static std::unordered_map<size_t, PinnedBuffer> s_pinned_map;
// 替换为 per-rank vector

struct PinnedBuffer { void* ptr = nullptr; size_t size = 0; };

#ifdef TR_USE_CUDA
static std::vector<std::unordered_map<size_t, PinnedBuffer>> s_pinned_maps_cuda;
#else
static std::vector<std::unordered_map<size_t, PinnedBuffer>> s_pinned_maps_cpu;
#endif

static auto& get_pinned_maps() {
#ifdef TR_USE_CUDA
    return s_pinned_maps_cuda;
#else
    return s_pinned_maps_cpu;
#endif
}

static void ensure_pinned_maps_initialized() {
    auto& maps = get_pinned_maps();
    if (!maps.empty()) return;
    int ws = GlobalRegistry::instance().world_size();
    if (ws <= 0) ws = 1;
    maps.resize(ws);
}

// 内部 per-rank 实现
static void* get_dtensor_pinned_buffer_internal(int rank, size_t device_offset, size_t size) {
    ensure_pinned_maps_initialized();
    auto& maps = get_pinned_maps();
    if (rank < 0 || rank >= static_cast<int>(maps.size())) {
        TR_DEVICE_ERROR("get_dtensor_pinned_buffer: invalid rank " << rank
                        << " (world_size=" << maps.size() << ")");
    }
    auto& map = maps[rank];
    auto it = map.find(device_offset);
    if (it != map.end() && it->second.size >= size) {
        return it->second.ptr;
    }
    void* ptr = nullptr;
#ifdef TR_USE_CUDA
    cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
#else
    ptr = std::malloc(size);
#endif
    map[device_offset] = {ptr, size};
    return ptr;
}

// 公开 API：向后兼容（rank=0）
void* get_dtensor_pinned_buffer(size_t device_offset, size_t size) {
    return get_dtensor_pinned_buffer_internal(0, device_offset, size);
}

// 新增公开 API：per-rank
void* get_dtensor_pinned_buffer_for_rank(int rank, size_t device_offset, size_t size) {
    return get_dtensor_pinned_buffer_internal(rank, device_offset, size);
}

// 捕获期查找（per-rank）
static void* lookup_pinned_for_capture(int rank, size_t device_offset, size_t size) {
    ensure_pinned_maps_initialized();
    auto& maps = get_pinned_maps();
    if (rank < 0 || rank >= static_cast<int>(maps.size())) {
        TR_DEVICE_ERROR("lookup_pinned_for_capture: invalid rank " << rank);
    }
    auto& map = maps[rank];
    auto it = map.find(device_offset);
    if (it == map.end() || it->second.size < size) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: pinned buffer not pre-allocated for rank="
                        << rank << " offset=" << device_offset << " size=" << size);
    }
    return it->second.ptr;
}

// 修改 launch_range_h2d_copy_dtensor_cuda
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

    if (node.output_ranges.empty()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: empty output_ranges");
    }

    const auto& seg = node.output_ranges[0];
    if (seg.size == 0) return;

    int rank = ctx.rank_for_context();
    void* dst = ArenaKeeper::instance().ptr_at(rank, seg.offset);
    void* src = lookup_pinned_for_capture(rank, seg.offset, seg.size);

    cudaError_t err = cudaMemcpyAsync(dst, src, seg.size,
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR cudaMemcpyAsync failed: "
                        << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}
```

### 修改 2：`op_registry.h` —— 新增 per-rank API 声明

**文件**：`include/renaissance/backend/op_registry.h`

在 `get_dtensor_pinned_buffer` 声明下方添加：

```cpp
/// RANGE_H2D_COPY_DTENSOR 专用：获取指定 RANK 的 pinned memory buffer
/// rank: 逻辑 rank 索引（0-based）
/// device_offset: DTensor 在 ArenaKeeper 中的字节偏移
/// size: 所需的 buffer 字节数
void* get_dtensor_pinned_buffer_for_rank(int rank, size_t device_offset, size_t size);
```

### 修改 3：`task_base.h` + `task_base.cpp` —— `fill_transfer_buffer` 增加 rank 参数

**文件**：`include/renaissance/task/task_base.h`

修改声明（增加默认参数）：
```cpp
/**
 * @brief 将host数据填入RANGE_H2D_COPY_DTENSOR专用的pinned memory
 * @param host 主机张量
 * @param dtensor 目标分布式张量
 * @param rank 目标逻辑 rank（默认 0）
 * @note 仅填充指定 rank 的 pinned buffer，不触发GPU传输；实际H2D由run("xfer")中的图完成
 */
void fill_transfer_buffer(const Tensor& host, const DTensor& dtensor, int rank = 0);
```

**文件**：`src/task/task_base.cpp`

修改实现：
```cpp
void TaskBase::fill_transfer_buffer(const Tensor& host, const DTensor& dt, int rank) {
    check_run_eligibility(debug_mode_);
    if (debug_mode_) {
        std::cout << "[DRY RUN] skip fill_transfer_buffer " << dt.id << " rank=" << rank << "\n";
        return;
    }

    TR_CHECK(host.dtype() == dt.dtype, ValueError, "DType mismatch: host vs dtensor");
    TR_CHECK(host.numel() >= dt.shape.numel(), ValueError,
             "Host tensor too small");

    const DTensor& live_dt = memory_plan_.get_dtensor(dt.id);
    size_t offset = static_cast<size_t>(live_dt.offset());
    size_t size = static_cast<size_t>(live_dt.slot_bytes());

    // 使用 per-rank API
    void* pinned = get_dtensor_pinned_buffer_for_rank(rank, offset, size);
    if (!pinned) {
        TR_DEVICE_ERROR("fill_transfer_buffer: no pinned buffer for DTensor id="
                        << dt.id << " rank=" << rank << " offset=" << offset);
    }

    if (dt.is_compact()) {
        std::memcpy(pinned, host.data<void>(), static_cast<size_t>(dt.nbytes()));
    } else {
        // ... 原有非紧凑拷贝逻辑 ...
    }
}
```

### 修改 4：`task_base.cpp` —— `compile_capture_simple()` 自动分配 + per-rank 预分配

**文件**：`src/task/task_base.cpp`

在 `compile_capture_simple()` 的 `#ifdef TR_USE_CUDA` 块开头替换为：

```cpp
void TaskBase::compile_capture_simple() {
#ifdef TR_USE_CUDA
    if (!backend_->contexts.empty() && backend_->contexts[0]->is_gpu()) {
        // ========== 自动检测 A/B 传输并分配 StagingBufferPool ==========
        bool has_ab_transfer = false;
        for (const auto& [name, entry] : named_graphs_) {
            for (const auto& node : entry.graph.linear_nodes()) {
                if (node.kind != GraphNode::Kind::RANGE) continue;
                if (node.range_op == RangeOp::RANGE_H2D_COPY_A ||
                    node.range_op == RangeOp::RANGE_H2D_COPY_B) {
                    has_ab_transfer = true;
                    break;
                }
            }
        }

        if (has_ab_transfer) {
            auto& reg = GlobalRegistry::instance();
            if (!reg.has_staging_memory()) {
                int batch_size = reg.get_local_batch_size();
                int resolution = reg.current_resolution_train();
                int channels = reg.num_color_channels();
                bool using_amp = false;
                try { using_amp = reg.using_amp(); } catch (const TRException&) {}

                size_t elem_size = using_amp ? sizeof(uint16_t) : sizeof(float);
                int effective_c = (using_amp && channels == 3) ? 4 : channels;

                size_t label_raw = static_cast<size_t>(batch_size) * sizeof(int32_t);
                size_t label_aligned = utils::align_up_256(label_raw + 16);
                size_t data_raw = static_cast<size_t>(batch_size) * resolution * resolution
                                  * effective_c * elem_size;
                size_t data_aligned = utils::align_up_256(data_raw + 16);
                size_t per_zone = label_aligned + data_aligned;
                size_t total_size = per_zone * 2;

                reg.allocate_staging_memory(total_size);
            }
        }
        // ================================================================

        // ========== 为所有 rank 预分配 DTENSOR pinned buffer ==========
        for (const auto& [name, entry] : named_graphs_) {
            for (const auto& node : entry.graph.linear_nodes()) {
                if (node.kind != GraphNode::Kind::RANGE) continue;
                if (node.range_op != RangeOp::RANGE_H2D_COPY_DTENSOR) continue;
                if (node.output_ranges.empty()) continue;
                const auto& seg = node.output_ranges[0];
                for (int rank = 0; rank < num_gpus_; ++rank) {
                    get_dtensor_pinned_buffer_for_rank(rank, seg.offset, seg.size);
                }
            }
        }
        // ================================================================

        // ... 原有 warmup 代码继续 ...
```

### 修改 5：`tests/correction/CMakeLists.txt` —— 注册三个新测试

**文件**：`tests/correction/CMakeLists.txt`

在末尾 `endforeach()` 之后添加：

```cmake
# RANGE_H2D_COPY 系列正确性测试（natively 多 RANK）
set(H2D_TESTS
    test_h2d_copy_a
    test_h2d_copy_b
    test_h2d_copy_dtensor
)
foreach(test_name IN LISTS H2D_TESTS)
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
        target_compile_options(${test_name} PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
    else()
        target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
    endif()
    message(STATUS "  - ${test_name}: RANGE_H2D_COPY correction test")
endforeach()
```

---

## 四、测试方案（natively 多 RANK）

### 4.1 测试总览

| 测试文件 | 算子 | 验证目标 | 多 RANK 设计 |
|---------|------|---------|-------------|
| `test_h2d_copy_a.cpp` | RANGE_H2D_COPY_A | A 区 label+data 异步传输正确性 | 每个 rank 填充**不同**数据，fetch 后分别验证 |
| `test_h2d_copy_b.cpp` | RANGE_H2D_COPY_B | B 区 label+data 异步传输正确性 | 同上，B 区偏移 |
| `test_h2d_copy_dtensor.cpp` | RANGE_H2D_COPY_DTENSOR | 标量 DTensor 精确传输正确性 | 每个 rank 写入**不同**LR值，fetch 后分别验证 |

### 4.2 `test_h2d_copy_a.cpp`

```cpp
#include "renaissance.h"
#include <iostream>
#include <cmath>

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

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    // 小参数加速测试
    reg.local_batch_size(4)
       .train_resolution(8)
       .val_resolution(8)
       .color_channels(1);

    SimpleTask task;

    DTensor d_label = task.alloc(Shape{4,1,1,1}, DType::INT32, Region::I_A_LABEL);
    DTensor d_data  = task.alloc(Shape{4,8,8,1}, DType::FP32,  Region::I_A_DATA);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g;
    g.append_range(GraphId::TRANSFER_A, RangeOp::RANGE_H2D_COPY_A,
                   {}, {mp.region_range(Region::I_A_LABEL),
                        mp.region_range(Region::I_A_DATA)});
    task.add_graph("xfer_a", std::move(g), StreamKind::TRANS);
    task.compile();

    // 计算 StagingBufferPool 布局
    size_t label_raw = 4 * sizeof(int32_t);
    size_t label_aligned = utils::align_up_256(label_raw + 16);

    // 为每个 rank 准备**不同**的数据（rank-specific pattern）
    std::vector<Tensor> rank_labels(num_ranks);
    std::vector<Tensor> rank_data(num_ranks);

    for (int rank = 0; rank < num_ranks; ++rank) {
        rank_labels[rank] = Tensor::zeros({4,1,1,1}, DType::INT32);
        for (int i = 0; i < 4; ++i) {
            rank_labels[rank].data<int32_t>()[i] = rank * 1000 + i * 10 + 1;
        }

        rank_data[rank] = Tensor::zeros({4,8,8,1}, DType::FP32);
        for (int i = 0; i < rank_data[rank].numel(); ++i) {
            rank_data[rank].data<float>()[i] = static_cast<float>(rank * 10)
                                               + static_cast<float>(i) * 0.1f + 0.5f;
        }
    }

    // 填充各 rank 的 StagingBufferPool A 区
    for (int rank = 0; rank < num_ranks; ++rank) {
        uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
        std::memcpy(base, rank_labels[rank].data<int32_t>(), 4 * sizeof(int32_t));
        std::memcpy(base + label_aligned, rank_data[rank].data<float>(),
                    4 * 8 * 8 * 1 * sizeof(float));
    }

    // 执行传输（所有 rank 并行）
    task.run("xfer_a");

    // 逐个 rank 验证（各自数据应独立正确）
    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor out_label = task.fetch_from_rank(d_label, rank);
        Tensor out_data  = task.fetch_from_rank(d_data, rank);

        for (int i = 0; i < 4; ++i) {
            if (out_label.data<int32_t>()[i] != rank_labels[rank].data<int32_t>()[i]) {
                std::cout << "  Rank " << rank << " label[" << i << "] mismatch: "
                          << out_label.data<int32_t>()[i] << " != "
                          << rank_labels[rank].data<int32_t>()[i] << "\n";
                all_pass = false;
            }
        }

        double max_diff = 0.0;
        for (int64_t i = 0; i < out_data.numel(); ++i) {
            double diff = std::abs(
                static_cast<double>(out_data.data<float>()[i]) -
                static_cast<double>(rank_data[rank].data<float>()[i]));
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff > 1e-6) {
            std::cout << "  Rank " << rank << " data max|diff|=" << max_diff << " FAIL\n";
            all_pass = false;
        } else {
            std::cout << "  Rank " << rank << " OK (max|diff|=" << max_diff << ")\n";
        }
    }

    std::cout << "\nRANGE_H2D_COPY_A (" << num_ranks << " ranks): "
              << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}
```

### 4.3 `test_h2d_copy_b.cpp`

与 A 区测试结构相同，差异点：
- `Region::I_B_LABEL` / `Region::I_B_DATA`
- `RangeOp::RANGE_H2D_COPY_B`
- StagingBufferPool 写入 B 区：`base + per_zone` 偏移
- 每个 rank 使用与 A 区测试**不同**的 rank-specific pattern（避免数据巧合）

```cpp
// 核心差异
size_t per_zone = reg.staging_memory_size() / 2;

// B 区写入：
std::memcpy(base + per_zone, rank_labels[rank].data<int32_t>(), ...);
std::memcpy(base + per_zone + label_aligned, rank_data[rank].data<float>(), ...);
```

### 4.4 `test_h2d_copy_dtensor.cpp`

```cpp
#include "renaissance.h"
#include <iostream>
#include <cmath>

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

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    SimpleTask task;

    // 分配标量 DTensor（模拟学习率）
    DTensor d_lr = task.alloc_scalar(DType::FP32);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g;
    g.append_range(GraphId::SIMPLE_TASK_GRAPH, RangeOp::RANGE_H2D_COPY_DTENSOR,
                   {}, {mp.region_range(d_lr.region)});
    task.add_graph("h2d_dtensor", std::move(g), StreamKind::TRANS);
    task.compile();

    // 为每个 rank 准备**不同**的 LR 值
    std::vector<float> host_lrs(num_ranks);
    for (int rank = 0; rank < num_ranks; ++rank) {
        host_lrs[rank] = 0.001f * (rank + 1);  // rank 0: 0.001, rank 1: 0.002, ...
    }

    // 填充各 rank 的 per-rank pinned buffer
    size_t offset = mp.get_dtensor(d_lr.id).offset;
    size_t size   = mp.get_dtensor(d_lr.id).slot_bytes();
    for (int rank = 0; rank < num_ranks; ++rank) {
        void* pinned = get_dtensor_pinned_buffer_for_rank(rank, offset, size);
        std::memcpy(pinned, &host_lrs[rank], sizeof(float));
    }

    // 执行传输
    task.run("h2d_dtensor");

    // 逐个 rank 验证（各自 LR 应独立正确）
    bool pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor fetched = task.fetch_from_rank(d_lr, rank);
        float actual = fetched.data<float>()[0];
        float expected = host_lrs[rank];
        float diff = std::abs(actual - expected);
        if (diff > 1e-7f) {
            std::cout << "Rank " << rank << " mismatch: expected=" << expected
                      << " actual=" << actual << " diff=" << diff << " FAIL\n";
            pass = false;
        } else {
            std::cout << "Rank " << rank << " OK: lr=" << actual << "\n";
        }
    }

    std::cout << "\nRANGE_H2D_COPY_DTENSOR (" << num_ranks << " ranks): "
              << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? 0 : 1;
}
```

### 4.5 Overlap 验证思路（可选增强）

利用 `run("xfer_a", "compute")` 接口同时 launch 传输图和计算图：

```cpp
// 构造计算图（如 IDENTITY 或简单 element-wise）
ComputationGraph comp_g;
// ... append COMPUTE node on d_data ...
task.add_graph("compute", std::move(comp_g), StreamKind::COMP_1);
task.compile();

// 同时启动两张图（所有 rank 并行）
task.run("xfer_a", "compute");

// 验证每个 rank 的计算结果正确 → 说明传输在计算前已完成
```

---

## 五、实施步骤

| # | 任务 | 文件 | 优先级 | 预估工时 |
|---|------|------|:--:|:--:|
| 1 | 修改 `launch_range_h2d_copy_cuda` 使用真实指针 | `h2d_op.cpp` | P0 | 1h |
| 2 | DTENSOR pinned buffer 改为 per-rank | `h2d_op.cpp` + `op_registry.h` | P0 | 1h |
| 3 | `fill_transfer_buffer` 增加 rank 参数 | `task_base.h` + `task_base.cpp` | P0 | 0.5h |
| 4 | 修改 `compile_capture_simple` 自动分配 + per-rank 预分配 | `task_base.cpp` | P0 | 1h |
| 5 | 编写 `test_h2d_copy_a.cpp`（多 rank 不同数据） | `tests/correction/` | P1 | 1h |
| 6 | 编写 `test_h2d_copy_b.cpp`（多 rank 不同数据） | `tests/correction/` | P1 | 0.5h |
| 7 | 编写 `test_h2d_copy_dtensor.cpp`（多 rank 不同 LR） | `tests/correction/` | P1 | 1h |
| 8 | 修改 `CMakeLists.txt` 注册测试 | `tests/correction/CMakeLists.txt` | P1 | 0.5h |
| 9 | 编译并运行测试（多 rank GPU + CPU） | — | P0 | 1h |
| 10 | DeepLearningTask 同步优化（event-based overlap） | `deep_learning_task.cpp` | P2 | 2h |

**合计**：核心功能 + 测试 ≈ 7.5 小时，全部完成 ≈ 10 小时。

---

## 六、风险与缓解

| # | 风险 | 等级 | 缓解 |
|---|------|:--:|------|
| R1 | `calculate_label_aligned()` 与 TransferStation 公式不一致 | 低 | 公式完全相同（`align_up_256(raw + 16)`），TransferStation::configure() 已验证 |
| R2 | CPU 路径 H2D 为零填充，无法验证数据正确性 | 低 | CPU 路径设计如此（数据在 transfer_to_rank 中处理）。A/B 测试主要验证 GPU 路径 |
| R3 | AMP 模式下 channels padding 到 4 | 中 | `effective_c = (using_amp && channels == 3) ? 4 : channels` 已在自动分配逻辑中处理 |
| R4 | `fill_transfer_buffer` 新增默认参数导致 ABI 变化 | 低 | 默认参数在声明处，`.cpp` 实现同步修改。`dual_graph.cpp` 等现有调用无需改动 |
| R5 | per-rank pinned map 的 `world_size` 在捕获前未确定 | 低 | `ensure_pinned_maps_initialized()` 从 GlobalRegistry 读取 world_size；若未设置则默认 1 |

---

## 七、附录：StagingBufferPool 容量公式

```
label_raw     = batch_size * sizeof(int32_t)
label_aligned = align_up_256(label_raw + 16)

data_raw      = batch_size * h * w * c * elem_size
                // FP32: elem_size = 4, c = num_color_channels
                // FP16: elem_size = 2, c = (num_color_channels == 3) ? 4 : num_color_channels
data_aligned  = align_up_256(data_raw + 16)

per_zone      = label_aligned + data_aligned
total         = per_zone * 2   // A + B 双区
```

**调用方契约**：`allocate_staging_memory(total)` 必须在 `compile()` 之前完成。`compile_capture_simple()` 已自动处理此契约。
