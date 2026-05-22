# UTK1：RANGE 异步传输三大算子实现与测试 —— 最终综合方案

> 综合 TRX1 / TRX2 / TRX3 三份方案 + 当前代码审查
> 2026-05-21

---

## 〇、文档溯源

| 源文档 | 作者 | 定位 | 核心贡献 |
|--------|------|------|---------|
| [TRX1.md](file:///r:/renaissance/TRX1.md) | 本人 | 完整技术方案 | DTensorPinnedPool per-rank 设计、Region→offset 映射、GlobalRegistry 管道、完整修改清单 |
| [TRX2.md](file:///r:/renaissance/TRX2.md) | 小伙伴 K | 精简实施方案 | `get_h2d_src_ptr()` 辅助函数、fallback 策略、CPU 路径 zero-fill 问题识别、P1-P4 优先级体系 |
| [TRX3.md](file:///r:/renaissance/TRX3.md) | 小伙伴 S | 测试专项方案 | 三层测试体系（unit/integration/perf）、overlap 测试设计、完整测试代码模板、buffer 计算辅助函数、成功标准 |

本文档综合三份方案的最优设计，消除分歧，形成唯一的、可执行的最终方案。

---

## 一、问题定位：当前代码的唯一关键缺陷

经过 TRX1/2/3 三份方案交叉验证，当前代码的 **所有问题收敛到一点**：

### 缺陷：`h2d_op.cpp` 中 `s_placeholder_h2d` 未绑定 StagingBufferPool

[文件：h2d_op.cpp:104-116](file:///r:/renaissance/src/backend/ops/range/h2d_op.cpp#L104-L116)

```cpp
static void* s_placeholder_h2d = nullptr;
if (!s_placeholder_h2d) {
    cudaHostAlloc(&s_placeholder_h2d, 4096, cudaHostAllocDefault);
}
void* src = s_placeholder_h2d;  // ← 永远指向 4096 字节的未初始化内存
```

**影响链**：
1. CUDA Graph capture 期，`cudaMemcpyAsync(dst, s_placeholder_h2d, dst_size, ...)` 被记录为 graph memcpy node
2. Graph instantiate 时，src 指针被固定为 `s_placeholder_h2d` 的地址
3. 运行时 `cudaGraphLaunch` 执行 memcpy — 传输的是 **placeholder 中的垃圾数据**，而非 StagingBufferPool 中的实际数据
4. placeholder 只有 4096 字节，远小于实际传输量（label_aligned + data_aligned 通常为数百 KB 到数 MB）

**TRX1/TRX2/TRX3 对此问题的一致判断**：必须在 capture 之前/之中使用 StagingBufferPool 的真实指针。

**分歧点**已消除：
- TRX1 提出通过 GlobalRegistry 传递 `staging_memory_ptr(rank)` 和 `label_aligned` → ✅ 采纳
- TRX2 提出 `get_h2d_src_ptr()` 辅助函数 + fallback 策略 → ✅ 采纳
- TRX3 提出 `set_h2d_placeholder_ptr()` 外部注入 → ❌ 不采纳（增加不必要的全局状态，且 `launch_cuda` 函数签名不可变）

---

## 二、最终方案：修改 `launch_range_h2d_copy_cuda` 直接使用 StagingBufferPool 指针

### 2.1 设计原理

> — 用户 C1（TRA.md）：StagingBufferPool 必须在 compile 之初就已分配，确保知道每块 buffer 的指针。

> — 用户 C5：传输图和计算图必须是在不同 stream 上的不同 CUDA Graph。

CUDA Graph 的 `cudaMemcpyAsync` node 在 capture 时会记录 src/dst 指针和大小。**只要 src 指针指向的 pinned memory 在运行时地址不变，graph replay 时就能正确读取**。

StagingBufferPool 使用 `cudaMallocHost` 分配的 pinned memory 地址固定，生命周期覆盖整个训练过程 → 完全满足条件。

**因此**：在 capture 阶段直接使用 `GlobalRegistry::staging_memory_ptr(rank)` 作为 src 指针，即可彻底解决问题。

### 2.2 A/B 区数据布局与指针计算

TransferStation 双区布局（[transfer_station.h:L16-L22](file:///r:/renaissance/include/renaissance/data/transfer_station.h#L16-L22)）：

```
StagingBufferPool 一块 buffer 的完整布局（per rank）:

A 区 (offset 0):
  [0,           label_aligned)  → I_A_LABEL 数据
  [label_aligned, per_zone)    → I_A_DATA  数据

B 区 (offset per_zone):
  [per_zone,              per_zone + label_aligned)  → I_B_LABEL 数据
  [per_zone + label_aligned, 2*per_zone)            → I_B_DATA  数据
```

其中：
```
label_aligned = align_up_256(local_batch_size * 4 + 16)
data_aligned  = align_up_256(local_batch_size * H * W * C_eff * elem_size + 16)
per_zone      = label_aligned + data_aligned
C_eff = (AMP && channels == 3) ? 4 : channels
```

### 2.3 修改后的 kernel 代码

[文件：src/backend/ops/range/h2d_op.cpp](file:///r:/renaissance/src/backend/ops/range/h2d_op.cpp)，替换 `launch_range_h2d_copy_cuda` 函数体（L92-127）：

```cpp
/// 由 GlobalRegistry 确定当前 rank 的 staging buffer 基地址和区域偏移
static void* get_h2d_src_for_region(int rank, Region region, size_t* out_offset) {
    auto& reg = GlobalRegistry::instance();
    if (!reg.has_staging_memory()) {
        // Fallback：SimpleTask 未分配 StagingBufferPool 的场景
        // 仍然使用 placeholder（此时传输数据为未定义内容）
        // 调用方应在 compile 前确保 StagingBufferPool 已分配
        static void* s_fallback = nullptr;
        if (!s_fallback) cudaHostAlloc(&s_fallback, 4096, cudaHostAllocDefault);
        if (out_offset) *out_offset = 0;
        return s_fallback;
    }

    size_t total = reg.staging_memory_size();
    size_t per_zone = total / 2;
    uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));

    // TransferStation 的 label_aligned 已在 configure 时注册
    size_t label_aligned = reg.staging_label_aligned();

    if (out_offset) *out_offset = 0;

    switch (region) {
        case Region::I_A_LABEL: return base;                           // A label
        case Region::I_A_DATA:  return base + label_aligned;            // A data
        case Region::I_B_LABEL: return base + per_zone;                 // B label
        case Region::I_B_DATA:  return base + per_zone + label_aligned; // B data
        default:
            TR_DEVICE_ERROR("get_h2d_src_for_region: unexpected region "
                            << static_cast<int>(region));
            return nullptr;
    }
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

    for (const auto& range : node.output_ranges) {
        auto [dst_off, dst_size] = mp.resolve_region_bounds(
            static_cast<Region>(range.start_region_id),
            static_cast<Region>(range.end_region_id));
        if (dst_size == 0) continue;

        void* dst = ArenaKeeper::instance().ptr_at(rank, dst_off);

        // 从 StagingBufferPool 获取当前 region 对应的 src 指针
        void* src = get_h2d_src_for_region(
            rank, static_cast<Region>(range.start_region_id), nullptr);

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

### 2.4 配套修改：TransferStation 注册 label_aligned

[文件：src/data/transfer_station.cpp](file:///r:/renaissance/src/data/transfer_station.cpp)，在 `configure()` 末尾（L307 附近）添加：

```cpp
// 将 label_aligned 和 per_zone 注册到 GlobalRegistry，
// 供 h2d_op.cpp 的 launch_range_h2d_copy_cuda 使用
GlobalRegistry::instance().set_staging_label_aligned(label_aligned_);
GlobalRegistry::instance().set_staging_per_zone(bytes_per_transfer_);
```

### 2.5 配套修改：GlobalRegistry 新增字段

[文件：include/renaissance/core/global_registry.h](file:///r:/renaissance/include/renaissance/core/global_registry.h)，在 staging memory 方法附近新增：

```cpp
// === staging buffer 布局参数（供 h2d_op.cpp 使用）===

/// 设置 staging buffer 的 label 区对齐后大小
void set_staging_label_aligned(size_t value);
/// 获取 staging buffer 的 label 区对齐后大小
size_t staging_label_aligned() const;

/// 设置 staging buffer 单区（A 或 B）对齐后总大小
void set_staging_per_zone(size_t value);
/// 获取 staging buffer 单区对齐后总大小
size_t staging_per_zone() const;
```

[文件：src/core/global_registry.cpp](file:///r:/renaissance/src/core/global_registry.cpp)，实现：

```cpp
void GlobalRegistry::set_staging_label_aligned(size_t value) {
    staging_label_aligned_ = value;
}
size_t GlobalRegistry::staging_label_aligned() const {
    return staging_label_aligned_;
}
void GlobalRegistry::set_staging_per_zone(size_t value) {
    staging_per_zone_ = value;
}
size_t GlobalRegistry::staging_per_zone() const {
    return staging_per_zone_;
}
```

成员变量：
```cpp
size_t staging_label_aligned_ = 0;
size_t staging_per_zone_ = 0;
```

---

## 三、SimpleTask 测试中 StagingBufferPool 的分配

### 3.1 方案选择

| 方案 | 来源 | 描述 | 评价 |
|------|------|------|:--:|
| A：compile_capture_simple 自动检测并分配 | TRX1 | 在 compile_capture_simple 开头检测 H2D 节点，自动分配 StagingBufferPool | 自动化好，但与 SimpleTask "不初始化 Preprocessor" 的设计哲学冲突 |
| **B：测试代码显式分配** | TRX2/TRX3 | 测试代码在 `task.compile()` 之前调用 `reg.allocate_staging_memory(total)` | ✅ **采纳**：显式控制，测试意图清晰，与 SimpleTask 设计一致 |
| C：外部 set_h2d_placeholder_ptr 注入 | TRX3 | compile 后调用 set_h2d_placeholder_ptr 替换 placeholder | ❌ 不采纳：增加不需要的全局状态 |

**最终决策**：采用方案 B。

### 3.2 辅助：增加编译期检查

参照 [TRX2.md](file:///r:/renaissance/TRX2.md) 的 P2 建议，在 `compile_capture_simple()` 开头增加检查：

```cpp
// compile_capture_simple() 中增加（warmup 之前）
for (const auto& [name, entry] : named_graphs_) {
    for (const auto& node : entry.graph.linear_nodes()) {
        if (node.kind != GraphNode::Kind::RANGE) continue;
        if (node.range_op == RangeOp::RANGE_H2D_COPY_A ||
            node.range_op == RangeOp::RANGE_H2D_COPY_B) {
            TR_CHECK(GlobalRegistry::instance().has_staging_memory(),
                     RuntimeError,
                     "Graph '" << name << "' contains H2D_COPY_A/B but "
                     "StagingBufferPool is not allocated. "
                     "Call GlobalRegistry::allocate_staging_memory() "
                     "before task.compile().");
        }
    }
}
```

### 3.3 Buffer 大小计算辅助函数

基于 [TRX3.md](file:///r:/renaissance/TRX3.md) 的 buffer 计算公式，提供辅助函数（可在 test 中 inline 定义）：

```cpp
static size_t align_up_256(size_t x) {
    return ((x + 16 + 255) / 256) * 256;
}

/// 计算 StagingBufferPool 的总大小（A + B 双区）
/// @param batch_size  本地 batch size
/// @param h           样本高度（resolution）
/// @param w           样本宽度（resolution）
/// @param c           颜色通道数
/// @param is_amp      是否 AMP 模式（FP16）
static size_t calc_staging_total_bytes(int batch_size, int h, int w, int c, bool is_amp) {
    int c_eff = (is_amp && c == 3) ? 4 : c;
    size_t elem_size = is_amp ? sizeof(uint16_t) : sizeof(float);

    size_t label_raw = static_cast<size_t>(batch_size) * sizeof(int32_t);
    size_t label_aligned = align_up_256(label_raw);

    size_t data_raw = static_cast<size_t>(batch_size) * h * w * c_eff * elem_size;
    size_t data_aligned = align_up_256(data_raw);

    size_t per_zone = label_aligned + data_aligned;
    return per_zone * 2;  // A + B
}
```

### 3.4 测试中 StagingBufferPool 数据填充

由于 SimpleTask 不使用 TransferStation，测试需要直接写入 StagingBufferPool：

```cpp
// 填充 A 区
uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(0));
size_t label_aligned = reg.staging_label_aligned();

// Label 数据: 写入 base[0, label_raw)
std::memcpy(base, h_label_data, label_raw);

// Data 数据: 写入 base[label_aligned, ...)
std::memcpy(base + label_aligned, h_data_data, data_raw);

// 类似地填充 B 区（如果需要测试 B）
size_t per_zone = reg.staging_per_zone();
std::memcpy(base + per_zone, h_label_data_b, label_raw);
std::memcpy(base + per_zone + label_aligned, h_data_data_b, data_raw);
```

---

## 四、RANGE_H2D_COPY_DTENSOR 现状评估

### 4.1 当前实现已基本正确

`RANGE_H2D_COPY_DTENSOR` 的路径是完整的：

- `get_dtensor_pinned_buffer(offset, size)` → 分配/查找 pinned buffer
- `compile_capture_simple()` L263-271 → capture 前预分配
- `launch_range_h2d_copy_dtensor_cuda` → `lookup_pinned_for_capture` → `cudaMemcpyAsync`

**SRC 指针在 capture 时就是真正的 pinned buffer 地址** — 这是正确的原因：`get_dtensor_pinned_buffer` 在 capture 前被调用，返回的 `cudaHostAlloc` 地址在整个生命周期内不变。

与 A/B 算子的关键区别：A/B 的 src 是 StagingBufferPool 的 buffer（大块、双缓冲、与 TransferStation 耦合），而 DTENSOR 的 src 是独立的 per-DTensor pinned buffer（小块、按需分配、简单直接）。

### 4.2 已知局限（不影响 Phase 1）

| 局限 | 影响 | 处理 |
|------|------|------|
| `s_pinned_map` 是全局单例，不区分 rank | 多 rank 场景下所有 rank 共享同一 pinned buffer → 无法为不同 rank 传输不同值 | **Phase 1 不处理**。SimpleTask 单 rank 测试不受影响。DeepLearningTask 多 rank per-batch LR 更新场景留待 Phase 2。 |
| CPU 路径 zero-fill | CPU 测试只能验证图执行，不能验证数据 | 接受。H2D 本质是 GPU 概念，CPU 路径的实际数据填充在 `transfer_to_rank()` 中完成。 |

### 4.3 DTensorPinnedPool 延迟到 Phase 2

[TRX1.md](file:///r:/renaissance/TRX1.md) 提出的 `DTensorPinnedPool` per-rank 管理是更完善的方案，但它增加了新类和新 API。考虑到：
1. 当前 DTENSOR 路径在 SingleTask 下完全正确
2. DeepLearningTask 的 per-batch LR 更新尚未实现（该功能本身是 TODO）
3. 优先保证 A/B 算子的修复和全部三个算子的测试落地

**DTensorPinnedPool 标记为 Phase 2 优化项**。

---

## 五、正确性测试方案

### 5.1 测试层次

```
Level 1: 单元正确性测试 (tests/correction/)
  ├── test_h2d_copy_a.cpp         — RANGE_H2D_COPY_A ✅ P1
  ├── test_h2d_copy_b.cpp         — RANGE_H2D_COPY_B ✅ P1
  └── test_h2d_copy_dtensor.cpp   — RANGE_H2D_COPY_DTENSOR ✅ P1

Level 2: 集成测试 (tests/integration/) — Phase 2
  └── test_transfer_overlap.cpp   — 计算-传输重叠验证

Level 3: 性能测试 (tests/perf/) — Phase 2
  └── perf_h2d_bandwidth.cpp     — H2D 带宽 + 重叠加速比
```

### 5.2 测试公共框架

三个测试共享以下框架（提取自 [TRX3.md](file:///r:/renaissance/TRX3.md) 模板 + [test_sgd_weight.cpp](file:///r:/renaissance/tests/correction/test_sgd_weight.cpp) 参考）：

```cpp
#include "renaissance.h"
#include <iostream>
#include <cstring>
#include <cmath>

using namespace tr;

// === Buffer 大小计算辅助函数 ===
static size_t align_up_256(size_t x) {
    return ((x + 16 + 255) / 256) * 256;
}
static size_t calc_per_zone_bytes(int batch_size, int h, int w, int c, bool is_amp) {
    int c_eff = (is_amp && c == 3) ? 4 : c;
    size_t es = is_amp ? sizeof(uint16_t) : sizeof(float);
    return align_up_256(batch_size * 4) + align_up_256(batch_size * h * w * c_eff * es);
}

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

    // === 测试参数 ===
    const int batch_size = 4;
    const int H = 8, W = 8, C = 1;
    const bool is_amp = false;

    // === 配置 GlobalRegistry ==
    reg.local_batch_size(batch_size);

    // === 分配 StagingBufferPool ===
    size_t per_zone = calc_per_zone_bytes(batch_size, H, W, C, is_amp);
    size_t label_aligned = align_up_256(batch_size * 4);
    reg.allocate_staging_memory(per_zone * 2);
    reg.set_staging_label_aligned(label_aligned);
    reg.set_staging_per_zone(per_zone);

    // === 构造 SimpleTask ===
    SimpleTask task;
    // ... 分配 DTensor ...
    task.finalize_memory();

    // === 构造图 ===
    ComputationGraph g;
    // ... 添加 RANGE node ...
    task.add_graph("graph_name", std::move(g), StreamKind::TRANS);
    task.compile();

    // === 填充 StagingBufferPool ===
    uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(0));
    // ... 填充 label 和 data ...

    // === 执行 ===
    task.run("graph_name");

    // === 校验 ===
    // ... fetch + compare ...
}
```

### 5.3 test_h2d_copy_a.cpp 详细设计

**测试目标**：验证 `RANGE_H2D_COPY_A` 将 StagingBufferPool A 区的 label + data 正确传输到 `I_A_LABEL` + `I_A_DATA` 区域。

**DTensor 分配**：
```cpp
Shape label_shape{batch_size, 1, 1, 1};              // N×1×1×1 = 4 elements INT32
Shape data_shape{batch_size, H, W, C};               // 4×8×8×1 = 256 elements FP32
DTensor d_label = task.alloc(label_shape, DType::INT32, Region::I_A_LABEL);
DTensor d_data  = task.alloc(data_shape,  DType::FP32,  Region::I_A_DATA);
```

**图构造**：
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

**数据填充**（A 区 = base + 0）：
```cpp
uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(0));
size_t label_aligned = reg.staging_label_aligned();

// Label: 0, 1, 2, 3
int32_t* label_ptr = reinterpret_cast<int32_t*>(base);
for (int i = 0; i < batch_size; ++i) label_ptr[i] = i;

// Data: 0.0, 1.0, 2.0, ..., 255.0
float* data_ptr = reinterpret_cast<float*>(base + label_aligned);
for (int i = 0; i < 256; ++i) data_ptr[i] = static_cast<float>(i);

// 多 rank：复制相同数据到其他 rank
for (int r = 1; r < num_ranks; ++r) {
    std::memcpy(reg.staging_memory_ptr(r), base, per_zone);
}
```

**验证**：
```cpp
for (int rank = 0; rank < num_ranks; ++rank) {
    Tensor out_label = task.fetch_from_rank(d_label, rank);
    Tensor out_data  = task.fetch_from_rank(d_data,  rank);

    for (int i = 0; i < batch_size; ++i) {
        if (out_label.data<int32_t>()[i] != i) { /* FAIL */ }
    }
    for (int i = 0; i < 256; ++i) {
        float expected = static_cast<float>(i);
        float diff = std::abs(out_data.data<float>()[i] - expected);
        if (diff > 1e-7f) { /* FAIL */ }
    }
}
```

**CPU 模式**：`launch_range_h2d_copy_cpu` 是 zero-fill，因此 CPU 测试只验证 dst 区域被填充为 0（而非验证数据正确性）。GPU 模式才是真正的数据完整性测试。

### 5.4 test_h2d_copy_b.cpp（与 A 的差异）

| 项目 | test_h2d_copy_a | test_h2d_copy_b |
|------|:--:|:--:|
| DTensor Region | `I_A_LABEL`, `I_A_DATA` | `I_B_LABEL`, `I_B_DATA` |
| RangeOp | `RANGE_H2D_COPY_A` | `RANGE_H2D_COPY_B` |
| 图名 | `"xfer_a"` | `"xfer_b"` |
| 数据填充偏移 | `base + 0` (A 区) | `base + per_zone` (B 区) |

其余完全相同。

### 5.5 test_h2d_copy_dtensor.cpp 详细设计

**测试目标**：验证 `RANGE_H2D_COPY_DTENSOR` 从 pinned buffer 传输单个 DTensor 的正确性。

**DTensor 分配**：
```cpp
DTensor d_w = task.alloc(Shape{2, 4, 4, 4}, DType::FP32, Region::W_FC_WEIGHT);
// 2×4×4×4 = 128 elements = 512 bytes
```

**图构造**：
```cpp
ComputationGraph g;
{
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_H2D_COPY_DTENSOR;
    node.output_ranges = { mp.dtensor_range(d_w.id) };
    g.append(std::move(node));
}
task.add_graph("h2d_dtensor", std::move(g), StreamKind::TRANS);
```

**数据准备**：利用现有的 `transfer_to_rank()` → `get_dtensor_pinned_buffer()` 路径：
```cpp
Tensor h_w = Tensor::fill({2, 4, 4, 4}, DType::FP32, 0);
for (int i = 0; i < h_w.numel(); ++i) h_w.data<float>()[i] = static_cast<float>(i);

// transfer_to_rank 会自动把数据写入 DTENSOR 对应的 pinned buffer
task.transfer_to_rank(h_w, d_w, 0);
if (num_ranks > 1) task.broadcast_from_rank0(d_w);
```

**关键**：`transfer_to_rank` 在 GPU 模式下将数据写入 pinned buffer（通过 `get_dtensor_pinned_buffer`），而非直接写入 device memory。之后 `RANGE_H2D_COPY_DTENSOR` 的 CUDA Graph 从同一 pinned buffer 读取数据 → 传输到 device。

**验证**：
```cpp
for (int rank = 0; rank < num_ranks; ++rank) {
    Tensor out = task.fetch_from_rank(d_w, rank);
    for (int i = 0; i < out.numel(); ++i) {
        float expected = static_cast<float>(i);
        float diff = std::abs(out.data<float>()[i] - expected);
        if (diff > 1e-7f) { /* FAIL */ }
    }
}
```

---

## 六、实施清单

### 6.1 Phase 1：核心修复 + 测试（本次必须完成）

| # | 文件 | 操作 | 内容 |
|---|------|------|------|
| **H-1** | `h2d_op.cpp` | **修改** | 删除 `s_placeholder_h2d`；新增 `get_h2d_src_for_region()`；重写 `launch_range_h2d_copy_cuda` |
| **H-2** | `global_registry.h` | **修改** | 新增 `staging_label_aligned_` / `staging_per_zone_` 字段 + setter/getter |
| **H-3** | `global_registry.cpp` | **修改** | 实现新增 setter/getter |
| **H-4** | `transfer_station.cpp` | **修改** | `configure()` 末尾注册 `label_aligned_` 和 `bytes_per_transfer_` |
| **H-5** | `task_base.cpp` | **修改** | `compile_capture_simple()` 增加 H2D_COPY_A/B 的 StagingBufferPool 检查 |
| **T-1** | `test_h2d_copy_a.cpp` | **新增** | RANGE_H2D_COPY_A 正确性测试 |
| **T-2** | `test_h2d_copy_b.cpp` | **新增** | RANGE_H2D_COPY_B 正确性测试 |
| **T-3** | `test_h2d_copy_dtensor.cpp` | **新增** | RANGE_H2D_COPY_DTENSOR 正确性测试 |
| **T-4** | `CMakeLists.txt` | **修改** | 注册 three 个新测试 |

### 6.2 Phase 2：增强项（后续）

| # | 内容 | 来源 |
|---|------|------|
| E-1 | DTensorPinnedPool per-rank 管理 | TRX1 |
| E-2 | test_transfer_overlap 集成测试 | TRX3 |
| E-3 | 计算通信重叠 event-based 优化 | TRX2 P3 |
| E-4 | `launch_range_h2d_copy_cpu` 改为实际 memcpy（如需 CPU 测试） | TRX2 |

---

## 七、与三份 TRX 文档的关键差异对照

| 议题 | TRX1 方案 | TRX2 方案 | TRX3 方案 | **UTK1 最终方案** |
|------|-----------|-----------|-----------|-----------------|
| A/B kernel 修复 | Region→offset switch + GlobalRegistry 管道 | `get_h2d_src_ptr(is_a_zone)` 简化 | `set_h2d_placeholder_ptr()` 外部注入 | **综合 TRX1+TRX2**：Region→offset switch + `get_h2d_src_for_region()` 辅助函数 + fallback |
| StagingBufferPool 分配 | `compile_capture_simple` 自动分配 | 测试代码显式分配 + 编译期检查 | 测试代码显式构造 | **采纳 TRX2/3 方案**：测试显式分配 + 编译期检查 |
| DTENSOR | DTensorPinnedPool 新类 | 保持现状 | 保持现状 | **Phase 1 保持现状**，DTensorPinnedPool 推迟到 Phase 2 |
| CPU 路径 | 未提及 | 明确 zero-fill 问题 | 未提及 | **采纳 TRX2 分析**：CPU 路径不验证数据，GPU 是主测试目标 |
| 测试层次 | 3 个 unit test | 3 个 unit test | 3 层（unit/integration/perf） | **Phase 1 3 个 unit test**，Phase 2 增加 integration + perf |
| 测试 Buffersize 计算 | inline 在测试中 | 测试代码自行计算 | `calc_staging_total_bytes()` 辅助函数 | **采纳 TRX3**：提供 `calc_staging_total_bytes()` 辅助函数 |
| overlay 测试 | 未提及 | 未提及 | 详细设计 | **Phase 2** |

---

## 八、实施顺序

```
Step 1: H-2 + H-3  GlobalRegistry 新增 staging_label_aligned / staging_per_zone
Step 2: H-4        TransferStation::configure() 注册到 GlobalRegistry
Step 3: H-5        task_base.cpp 编译期检查（可选，提升错误信息质量）
Step 4: H-1        h2d_op.cpp 重写 launch_range_h2d_copy_cuda
      └── 验证：编译通过
Step 5: T-1 + T-2 + T-3  编写三个测试文件
Step 6: T-4              注册到 CMakeLists.txt
      └── 验证：三个测试全部 PASS（GPU 模式）
```

---

## 九、风险矩阵

| # | 风险 | 概率 | 影响 | 缓解 |
|---|------|:--:|:--:|------|
| R1 | `GlobalRegistry::allocate_staging_memory()` 的签名不接受 `gpu_ids`（目前由 `Preprocessor` 设置） | 中 | 中 | 已在 `global_registry.h:515` 确认签名 `allocate_staging_memory(size_t bytes_per_device)` — 它内部使用已注册的 `gpu_ids()` |
| R2 | 测试中的 StagingBufferPool 大小计算与 TransferStation 公式不完全一致 | 低 | 高 | 使用与 `transfer_station.cpp:248-249` 完全相同的公式；测试前在 transfer_station.cpp 中加断言验证 |
| R3 | `launch_range_h2d_copy_cuda` 在 DeepLearningTask 路径中未经充分测试 | 低 | 中 | DeepLearningTask 已有 `h2d_op.cpp` 的 H2D 路径，修复后只是 src 指针变化，不改变图结构 |
| R4 | CPU 测试失败（zero-fill 预期 ≠ data 预期） | 中 | 低 | CPU 测试使用独立的验证路径（验证 zero-fill 而非数据内容），或跳过 `--cpu` 模式 |

---

## 十、成功标准

```bash
# 编译
cmake --build build --target test_h2d_copy_a test_h2d_copy_b test_h2d_copy_dtensor

# 运行
./build/bin/tests/correction/test_h2d_copy_a --gpu
# 预期输出: RANGE_H2D_COPY_A: PASS

./build/bin/tests/correction/test_h2d_copy_b --gpu
# 预期输出: RANGE_H2D_COPY_B: PASS

./build/bin/tests/correction/test_h2d_copy_dtensor --gpu
# 预期输出: RANGE_H2D_COPY_DTENSOR: PASS
```