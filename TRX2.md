# TRX2：RANGE 异步传输三大算子实现与测试方案

> 基于 TRA.md 审查结论 + 当前代码全面分析
> 2026-05-21

---

## 一、现状回顾

### 1.1 已就绪的基础设施

| 组件 | 文件 | 状态 |
|------|------|:--:|
| RangeOp 枚举（H2D_COPY_A/B/DTENSOR） | `op_kind.h:248-250` | ✅ |
| Backend CUDA kernel（placeholder 版） | `h2d_op.cpp` | ⚠️ 需修正 |
| Backend CPU fallback | `h2d_op.cpp:137-165` | ✅ |
| Compiler 图注入（TRANSFER_A/B） | `compiler.cpp:1340-1364` | ✅ |
| Stream 管理（TRANS = index 0） | `device_context.cpp` | ✅ |
| StagingBufferPool（NUMA 感知） | `staging_buffer_pool.cpp` | ✅ |
| TransferStation（A/B 双缓冲） | `transfer_station.cpp/h` | ✅ |
| DTENSOR pinned buffer 管理 | `h2d_op.cpp:167-187` | ✅ |
| GraphSlot + GpuExecTable | `deep_learning_task.cpp` | ✅ |

### 1.2 当前代码中的关键缺陷

**缺陷 1：`s_placeholder_h2d` 未绑定实际 StagingBufferPool 指针**

`h2d_op.cpp:104-107`：
```cpp
static void* s_placeholder_h2d = nullptr;
if (!s_placeholder_h2d) {
    cudaHostAlloc(&s_placeholder_h2d, 4096, cudaHostAllocDefault);
}
// capture 阶段 src = s_placeholder_h2d（仅 4096 字节）
```

- 实际 StagingBufferPool 每区容量为 **数 MB 级**（label_aligned + data_aligned）
- 4096 字节 placeholder 远远不够
- CUDA Graph 的 memcpy node 在 capture 时记录 src 指针，**运行期不会自动替换**
- 当前代码中**未找到** `cudaGraphExecMemcpyNodeSetParams` 的调用

**缺陷 2：计算与传输未真正重叠**

`deep_learning_task.cpp:767-773`：
```cpp
cudaGraphLaunch(g_xfer, s_trans);
cudaGraphLaunch(g_deep, s_comp1);
cudaStreamSynchronize(s_comp1);  // ← 阻塞计算流
cudaStreamSynchronize(s_trans);  // ← 阻塞传输流
```

两图虽分属不同 stream，但每 batch 后都用 `cudaStreamSynchronize` 全部同步，**没有利用双缓冲实现 overlap**。

**缺陷 3：StagingBufferPool 分配时机未与 compile 严格绑定**

- `GlobalRegistry::allocate_staging_memory()` 是独立调用的
- `TransferStation::configure()` 会检查 `has_staging_memory()`，但 SimpleTask 不经过 TransferStation
- SimpleTask 测试若不预先分配 StagingBufferPool，H2D_COPY_A/B 的 capture 将拿不到正确 src 指针

---

## 二、三大算子精确定义

### 2.1 RANGE_H2D_COPY_A

| 属性 | 定义 |
|------|------|
| 类型 | RANGE（固定范围传输） |
| GraphId | `TRANSFER_A` |
| 输出范围 | `I_A_LABEL` (050) + `I_A_DATA` (051) |
| 传输字节数 | `label_aligned + data_aligned`（compile 前确定） |
| src 指针 | StagingBufferPool 的 A 区首地址（compile 前确定） |
| Stream | `StreamKind::TRANS` |
| 并发特性 | 可与 COMP 流上的计算图并行 |

**核心约束**：`dst` 地址、`src` 地址、`传输字节数` 都必须在 **CUDA Graph capture 之前** 完全确定。

### 2.2 RANGE_H2D_COPY_B

| 属性 | 定义 |
|------|------|
| 类型 | RANGE（固定范围传输） |
| GraphId | `TRANSFER_B` |
| 输出范围 | `I_B_LABEL` (052) + `I_B_DATA` (053) |
| 传输字节数 | 同 A 区（双缓冲对称） |
| src 指针 | StagingBufferPool 的 B 区首地址（compile 前确定） |
| Stream | `StreamKind::TRANS` |

### 2.3 RANGE_H2D_COPY_DTENSOR

| 属性 | 定义 |
|------|------|
| 类型 | RANGE（按 DTensor 精确传输） |
| 用途 | 标量更新（lr/wd/beta 等）、SimpleTask H2D、DeepLearningTask per-batch 标量注入 |
| src 指针 | `get_dtensor_pinned_buffer(offset, size)` 管理的 per-rank pinned buffer |
| dst 指针 | DTensor 的 device offset（通过 `MemoryPlan::get_dtensor(id).offset()` 确定） |
| 传输字节数 | `dtensor.slot_bytes()`（capture 前确定） |
| Stream | `StreamKind::TRANS` |

**设计要点**：
- 每个 RANK 有独立的 pinned buffer（通过 `s_pinned_map[device_offset]` 索引）
- 标量连续排列，每个 4 字节 FP32
- 多线程场景下：每个线程先写本地 pinned buffer，再统一 launch H2D

---

## 三、关键设计决策

### 决策 1：Capture 阶段直接使用 StagingBufferPool 真实指针

**问题**：`s_placeholder_h2d` 太小且固定，运行时替换复杂。

**方案**：在 capture 之前确保 StagingBufferPool 已分配，capture 阶段直接从 StagingBufferPool 取指针。

```cpp
// h2d_op.cpp 修正思路
#ifdef TR_USE_CUDA
static void* get_staging_ptr_for_rank(int rank, size_t* out_size) {
    auto& reg = GlobalRegistry::instance();
    if (!reg.has_staging_memory()) return nullptr;
    if (out_size) *out_size = reg.staging_memory_size() / 2;  // per_zone
    return reg.staging_memory_ptr(rank);
}
#endif
```

capture 阶段 `launch_range_h2d_copy_cuda` 中：
```cpp
// A 区：src = base + 0
// B 区：src = base + per_zone
size_t per_zone = 0;
void* base = get_staging_ptr_for_rank(ctx.rank_for_context(), &per_zone);
void* src = (is_a_zone) ? base : (static_cast<uint8_t*>(base) + per_zone);
```

### 决策 2：StagingBufferPool 分配时机

**必须在 `compile()` 之前完成**：

```cpp
// SimpleTask 测试路径示例
auto& reg = GlobalRegistry::instance();
reg.local_batch_size(4)
   .train_resolution(28)
   .val_resolution(28)
   .color_channels(1);  // 根据实际数据集

size_t label_raw = local_batch_size * sizeof(int32_t);
size_t data_raw  = local_batch_size * resolution * resolution * channels * 4;
size_t per_zone  = align_up_256(label_raw + 16) + align_up_256(data_raw + 16);
reg.allocate_staging_memory(per_zone * 2);  // A+B 双区

// 然后：task.finalize_memory() → add_graph() → compile()
```

**DeepLearningTask 路径**：`Preprocessor` 已在 `configure()` 中检查 `has_staging_memory()`，但 `allocate_staging_memory()` 的调用方需确保参数正确（AMP 时 channels padding 到 4）。

### 决策 3：计算通信重叠的同步策略

**当前问题**：`cudaStreamSynchronize(s_comp1)` + `cudaStreamSynchronize(s_trans)` 串行化了两流。

**目标方案**：使用 `cudaEvent` 实现跨流依赖，让传输和计算真正并行。

```cpp
// 理想流程（单 batch，伪代码）
cudaGraphLaunch(g_xfer_next, s_trans);     // 启动下一 batch 的传输
cudaGraphLaunch(g_deep_cur,  s_comp1);     // 启动当前 batch 的计算

// 计算流等待传输完成（只需要数据就位，不需要传输流完全空闲）
cudaStreamWaitEvent(s_comp1, event_xfer_done, 0);

// 只同步计算流（传输可在后台继续）
cudaStreamSynchronize(s_comp1);
cudaStreamSynchronize(s_comp2);
cudaStreamSynchronize(s_comp3);
// ← s_trans 不需要在这里同步！
```

**但注意**：当前 A/B 双缓冲的设计中，**下一 batch 的传输** 和 **当前 batch 的计算** 操作的是不同的 buffer（A vs B），天然无数据竞争。因此实际上可以：
1. 启动 `g_xfer_b`（传 B 区）
2. 启动 `g_deep_a`（算 A 区）
3. 只同步计算流
4. 传输流在后台继续

**Phase 1 实现**：先保证正确性（保留当前同步策略），Phase 2 再优化为 event-based overlap。

### 决策 4：RANGE_H2D_COPY_DTENSOR 的运行时标量更新

对于 DeepLearningTask 的 per-batch 学习率更新：

```cpp
// 每个 rank 一个线程
void* pinned_lr = get_dtensor_pinned_buffer(lr_offset, 4);
*static_cast<float*>(pinned_lr) = current_lr;  // CPU 写入 pinned memory
// 然后 launch RANGE_H2D_COPY_DTENSOR 图（已 capture，只拷贝 4 字节）
```

由于 `RANGE_H2D_COPY_DTENSOR` 的 src 是 `get_dtensor_pinned_buffer()` 管理的固定 pinned buffer，而 **不是** StagingBufferPool，所以它的指针在 capture 时就是正确的（通过 `lookup_pinned_for_capture`），无需替换。

---

## 四、实施方案

### P1：修正 `h2d_op.cpp` 的 src 指针（最高优先级）

**文件**：`src/backend/ops/range/h2d_op.cpp`

**修改内容**：
1. 删除 `s_placeholder_h2d`，改为从 `GlobalRegistry::staging_memory_ptr(rank)` 获取真实指针
2. A 区 src = base + 0，B 区 src = base + per_zone
3. 若 StagingBufferPool 未分配，fallback 到 `s_placeholder_h2d`（保证 SimpleTask 不强制依赖 StagingBufferPool）

```cpp
#ifdef TR_USE_CUDA
static void* get_h2d_src_ptr(int rank, bool is_a_zone, size_t* out_size) {
    auto& reg = GlobalRegistry::instance();
    if (reg.has_staging_memory()) {
        size_t total = reg.staging_memory_size();
        size_t per_zone = total / 2;
        uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
        if (out_size) *out_size = per_zone;
        return is_a_zone ? base : (base + per_zone);
    }
    // Fallback：SimpleTask 未分配 StagingBufferPool 时，用 placeholder
    static void* s_fallback = nullptr;
    if (!s_fallback) cudaHostAlloc(&s_fallback, 4096, cudaHostAllocDefault);
    if (out_size) *out_size = 4096;
    return s_fallback;
}
#endif
```

### P2：SimpleTask compile 流程添加 StagingBufferPool 支持

**文件**：`src/task/task_base.cpp`

**修改内容**：在 `compile_capture_simple()` 中，若检测到 `RANGE_H2D_COPY_A/B` 节点且 `GlobalRegistry::has_staging_memory()` 为 false，给出明确错误提示（而非静默使用 fallback）。

```cpp
// compile_capture_simple() 开头增加检查
for (const auto& [name, entry] : named_graphs_) {
    for (const auto& node : entry.graph.linear_nodes()) {
        if (node.kind != GraphNode::Kind::RANGE) continue;
        if (node.range_op == RangeOp::RANGE_H2D_COPY_A ||
            node.range_op == RangeOp::RANGE_H2D_COPY_B) {
            TR_CHECK(GlobalRegistry::instance().has_staging_memory(),
                     RuntimeError,
                     "Graph '" << name << "' contains H2D_COPY_A/B but "
                     "StagingBufferPool not allocated. Call "
                     "GlobalRegistry::allocate_staging_memory() before compile()");
        }
    }
}
```

### P3：计算通信重叠优化（Phase 2）

**文件**：`src/task/deep_learning_task.cpp`

**修改内容**：
1. 在 `run_iter()` 中，将 `cudaStreamSynchronize(s_trans)` 从每 batch 末尾移除
2. 改为：计算流通过 `cudaStreamWaitEvent` 等待上一 batch 传输完成事件
3. 在 A→B 切换时，确保 buffer 状态一致性（TransferStation 的 readable/writeable 标志已处理）

**此优化依赖当前同步策略已被充分验证，建议在 H2D 测试 PASS 后再实施。**

### P4：三大算子正确性测试

#### 4.4.1 `test_h2d_copy_a` / `test_h2d_copy_b`

**设计思路**：模拟 DeepLearningTask 的 A/B 双缓冲传输路径。

```cpp
// 测试架构
SimpleTask task;

// 1. 配置 GlobalRegistry（必须！）
auto& reg = GlobalRegistry::instance();
reg.local_batch_size(4)
   .train_resolution(8)   // 小分辨率，加速测试
   .val_resolution(8)
   .color_channels(1);

size_t label_raw = 4 * sizeof(int32_t);
size_t data_raw  = 4 * 8 * 8 * 1 * 4;
size_t per_zone  = align_up_256(label_raw + 16) + align_up_256(data_raw + 16);
reg.allocate_staging_memory(per_zone * 2);

// 2. 分配 DTensor（I_A_LABEL, I_A_DATA）
DTensor d_label = task.alloc(Shape{4,1,1,1}, DType::INT32, Region::I_A_LABEL);
DTensor d_data  = task.alloc(Shape{4,8,8,1}, DType::FP32,  Region::I_A_DATA);

// 3. 准备 host 数据并写入 StagingBufferPool
Tensor h_label = Tensor::fill({4,1,1,1}, DType::INT32, /*pattern*/);
Tensor h_data  = Tensor::fill({4,8,8,1}, DType::FP32,  /*pattern*/);

// 手动写入 StagingBufferPool 的 A 区
void* base = reg.staging_memory_ptr(0);
std::memcpy(base, h_label.data<void>(), h_label.nbytes());
std::memcpy(static_cast<uint8_t*>(base) + label_aligned,
            h_data.data<void>(), h_data.nbytes());

// 4. 构建 RANGE_H2D_COPY_A 图并执行
ComputationGraph g;
g.append_range(GraphId::TRANSFER_A, RangeOp::RANGE_H2D_COPY_A,
               {}, {mp.region_range(Region::I_A_LABEL),
                    mp.region_range(Region::I_A_DATA)});
task.add_graph("xfer_a", std::move(g), StreamKind::TRANS);
task.compile();
task.run("xfer_a");

// 5. fetch 验证
Tensor out_label = task.fetch_from_rank(d_label, 0);
Tensor out_data  = task.fetch_from_rank(d_data,  0);
// compare with h_label / h_data
```

**关键注意事项**：
- StagingBufferPool 的 layout 必须严格遵循 TransferStation 的公式：`label_aligned = align_up_256(label_raw + 16)`
- 测试需要同时覆盖 `--cpu` 和 `--gpu`
- CPU 路径下 `launch_range_h2d_copy_cpu` 是 zero-fill，所以此测试在 CPU 模式下**不能**验证数据正确性（只能验证范围/零填充）。CPU 测试需要修改 kernel 为实际 memcpy，或者单独测 GPU 路径。

**修正 CPU 路径**：当前 `launch_range_h2d_copy_cpu` 只做 `memset(dst, 0, size)`，这对正确性测试不够。但用户说"SimpleTask 不要求初始化 Preprocessor/TransferStation"，所以 CPU 路径下数据从哪来？

实际上，对于 CPU 路径，H2D 传输退化为 **Host-to-Host memcpy**（因为 CPU 模式下 device 和 host 共享内存）。所以 CPU 路径的 `launch_range_h2d_copy_cpu` 应该执行实际的数据复制，而不是 zero-fill。

但当前的 zero-fill 设计是有原因的：在 DeepLearningTask CPU 路径中，数据准备由 TransferStation 直接写入目标内存，不需要额外的 H2D 拷贝。所以 `launch_range_h2d_copy_cpu` 只是占位。

**测试方案折中**：
- `test_h2d_copy_a/b` **仅测 GPU 路径**（`--gpu`），因为 H2D 的本质是异步 GPU 传输
- 或者：在 CPU 模式下，手动将 src 数据复制到 StagingBufferPool，然后运行图，验证 dst 被 zero-fill（验证图执行了，但不验证数据内容）

更合理的方案：**测试同时验证 GPU 路径的数据正确性和 CPU 路径的 zero-fill 行为**。

#### 4.4.2 `test_h2d_copy_dtensor`

**设计思路**：验证标量/小数据 DTENSOR 的精确传输。

```cpp
SimpleTask task;

// 分配一个小 DTensor
Shape shape{2, 4, 4, 4};  // 128 elements
DTensor d_w = task.alloc(shape, DType::FP32, Region::W_FC_WEIGHT);

// 准备 host 数据
Tensor h_w = Tensor::fill(shape, DType::FP32, 0.42f);

// 通过 transfer_to_rank 写入 pinned buffer（自动调用 get_dtensor_pinned_buffer）
task.transfer_to_rank(h_w, d_w, 0);
if (num_ranks > 1) task.broadcast_from_rank0(d_w);

// 构建 RANGE_H2D_COPY_DTENSOR 图
// 注意： SimpleTask 的 H2D_COPY_DTENSOR 节点需要 output_ranges 指向 W_FC_WEIGHT 区域
ComputationGraph g;
g.append_range(GraphId::SIMPLE_TASK_GRAPH, RangeOp::RANGE_H2D_COPY_DTENSOR,
               {}, {mp.region_range(Region::W_FC_WEIGHT)});
task.add_graph("h2d_dtensor", std::move(g), StreamKind::TRANS);
task.compile();

// 这里有一个问题：transfer_to_rank 已经把数据写到 device memory 了（CPU 路径）
// 或者 pinned buffer 了（GPU 路径）。再跑一次 H2D 拷贝是幂等的。
// 更清晰的测试：先把 dst 清零，再运行 H2D 拷贝，验证数据恢复。
```

**更干净的 DTENSOR 测试设计**：

```cpp
// 1. 清零 dst（RANGE_CLEAR）
// 2. 准备新的 host 数据到 pinned buffer
// 3. 运行 RANGE_H2D_COPY_DTENSOR
// 4. fetch 验证
```

但 `RANGE_H2D_COPY_DTENSOR` 的 capture 逻辑是从 `s_pinned_map` 查找 pinned buffer，而 `transfer_to_rank()` 会调用 `fill_transfer_buffer()` → `get_dtensor_pinned_buffer()` → 分配/查找 pinned buffer。所以流程是兼容的。

#### 4.4.3 `test_transfer_overlap`（可选，Phase 2）

验证计算通信重叠的正确性：
1. 捕获两张图：传输图（TRANS stream）和 计算图（COMP stream）
2. 连续 launch 两张图
3. 验证计算结果正确（说明传输已完成且数据有效）
4. 测量时间确认有 overlap（计算时间 < 传输时间 + 计算时间）

---

## 五、实施清单

| # | 任务 | 文件 | 操作 | 优先级 |
|---|------|------|------|:--:|
| M1 | 修正 `launch_range_h2d_copy_cuda` 的 src 指针 | `h2d_op.cpp` | 替换 s_placeholder 为 StagingBufferPool 真实指针 | P0 |
| M2 | compile 前 StagingBufferPool 检查 | `task_base.cpp` | 若图含 H2D_COPY_A/B 则强制检查 has_staging_memory | P1 |
| M3 | 修正 CPU H2D fallback（可选） | `h2d_op.cpp` | 若需要 CPU 测试验证数据，改为实际 memcpy | P2 |
| M4 | `test_h2d_copy_a.cpp` | `tests/correction/` | 新建：A 区 H2D 正确性测试 | P1 |
| M5 | `test_h2d_copy_b.cpp` | `tests/correction/` | 新建：B 区 H2D 正确性测试 | P1 |
| M6 | `test_h2d_copy_dtensor.cpp` | `tests/correction/` | 新建：DTENSOR H2D 正确性测试 | P1 |
| M7 | CMakeLists.txt 注册 | `tests/correction/CMakeLists.txt` | 追加 3 个测试 | P1 |
| M8 | 计算通信重叠优化 | `deep_learning_task.cpp` | event-based 同步替代全流同步 | P3 |

---

## 六、风险与缓解

| # | 风险 | 等级 | 缓解 |
|---|------|:--:|------|
| R1 | StagingBufferPool 指针在 capture 后改变 | 低 | StagingBufferPool 生命周期覆盖整个训练，compile 后不再释放/重分配 |
| R2 | CPU 路径 H2D 测试无法验证真实数据 | 低 | CPU 模式下 StagingBufferPool 用 `std::malloc`，可直接 memcpy；但当前 zero-fill 设计下只验证 zero-fill 行为 |
| R3 | AMP 模式下 FP16 数据布局 | 中 | 测试需区分 FP32/FP16，AMP 时 channels padding 到 4，StagingBufferPool 容量公式已处理 |
| R4 | 多 rank 测试时 StagingBufferPool per-rank 指针 | 低 | `staging_memory_ptr(rank)` 返回各 rank 独立指针，测试 broadcast 后验证各 rank |

---

## 七、附录：StagingBufferPool 容量计算公式

```
label_raw     = local_batch_size * sizeof(int32_t)
label_aligned = ((label_raw + 16 + 255) / 256) * 256

data_raw      = local_batch_size * h * w * c * elem_size
                // FP32: elem_size = 4, c = num_color_channels
                // FP16: elem_size = 2, c = (num_color_channels == 3) ? 4 : num_color_channels
data_aligned  = ((data_raw + 16 + 255) / 256) * 256

per_zone      = label_aligned + data_aligned
total         = per_zone * 2   // A + B 双区
```

**调用方必须确保**：`allocate_staging_memory(total)` 在 `compile()` 之前完成。
