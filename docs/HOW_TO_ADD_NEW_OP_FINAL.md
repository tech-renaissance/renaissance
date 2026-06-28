# 如何添加新算子 · 最终版

> 基于 MaxPool 和 Dropout 的完整落地经验撰写。覆盖从类型定义到性能测试的全流程，适用于 Renaissance TR4 框架。

---

## 零、决策树：你的算子需要走完整流程吗？

```
┌─ 是否只通过 SimpleTask 手动构图验证？
│  ├─ 是 → 阶段一~三 + 阶段六
│  └─ 否（需要 DeepLearningTask / YAML 自动构图）→ 全部阶段
```

**最快验证路径**：只走阶段一~三 + 写一个 SimpleTask 测试，8 个文件改动即可跑通。MaxPool 和 Dropout 都先走此路径验证，再接入 ArchPlan。

---

## 一、类型系统（`op_kind.h` / `op_kind.cpp`）

### 1.1 参数结构体

```cpp
// include/renaissance/graph/op_kind.h

struct PoolParams {
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
};

struct DropoutParams {
    float p = 0.5f;
};
```

### 1.2 ComputeOp 枚举

命名约定：`{OP}_{FP32|AMP}_{FWD|BWD|INF}`。至少 6 个变体（FP32 三模式 + AMP 三模式）。AMP 变体仅 CUDA 实现，CPU 指向 `launch_not_supported_cpu`。

```cpp
enum class ComputeOp : uint16_t {
    MAXPOOL_FP32_FWD, MAXPOOL_FP32_BWD, MAXPOOL_FP32_INF,
    MAXPOOL_AMP_FWD,  MAXPOOL_AMP_BWD,  MAXPOOL_AMP_INF,
    DROPOUT_FP32_FWD, DROPOUT_FP32_BWD, DROPOUT_FP32_INF,
    DROPOUT_AMP_FWD,  DROPOUT_AMP_BWD,  DROPOUT_AMP_INF,
};
```

### 1.3 缝合到 OpParams

三处改动：variant 列表、显式构造函数、访问器。

```cpp
struct OpParams {
    std::variant<..., PoolParams, DropoutParams> data = std::monostate{};
    explicit OpParams(PoolParams p)    : data(std::move(p)) {}
    explicit OpParams(DropoutParams p) : data(std::move(p)) {}
    const PoolParams&    pool()    const { return std::get<PoolParams>(data); }
    const DropoutParams& dropout() const { return std::get<DropoutParams>(data); }
};
```

### 1.4 字符串化

在 `src/graph/op_kind.cpp` 中补充 `compute_op_to_string()` 和 `format_params()`。

---

## 二、后端实现（`src/backend/ops/dtensor/xxx_op.cpp` / `.cu`）

### 2.1 代码结构

```
namespace tr {

// ── CPU 实现 ──
static void launch_xxx_fp32_fwd_cpu(CpuOpContext* op_ctx) { ... }
static void launch_xxx_fp32_bwd_cpu(CpuOpContext* op_ctx) { ... }
static void launch_xxx_fp32_inf_cpu(CpuOpContext* op_ctx) { ... }

#ifdef TR_USE_CUDA
// ── CUDA launcher ──
static void launch_xxx_fp32_fwd_cuda(const GraphNode&, const MemoryPlan&,
                                      const DeviceContext&, MultiStreamCaptureState&) { ... }
// ── CUDA kernel (放 .cu 文件) ──
extern "C" __global__ void xxx_fwd_kernel(...) { ... }
#endif

// ── 注册函数 ──
void register_op_xxx() {
    auto& table = g_compute_op_table;
    {
        auto& e = table[static_cast<size_t>(ComputeOp::XXX_FP32_FWD)];
        e.op = ComputeOp::XXX_FP32_FWD;
        e.launch_cpu = launch_xxx_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_xxx_fp32_fwd_cuda;
#endif
    }
    // ... BWD、INF、AMP 变体同理
}

} // namespace tr
```

### 2.2 关键约定

| 规则 | 说明 |
|------|------|
| CPU 接口 | `CpuOpContext*`，通过 `op_ctx->ptr_at()` 获取指针 |
| CUDA 接口 | `(GraphNode, MemoryPlan, DeviceContext, MultiStreamCaptureState)` |
| dX 覆盖 X | BWD 的 output_ids 指向 X 的 DTensor ID，gradient 直接写入 X 的内存 |
| 禁止内部临时分配 | 如需 workspace，用 `ctx.ensure_workspace_grow()` 预申请 |
| 命名空间 | `.cpp` 前向声明和 `.cu` 实现在同一个 `namespace tr {}`，否则 LNK2019 |
| Stream | 通过 `get_op_default_stream(node.compute_op)` 获取，在 `state` 中注册 |

### 2.3 cuDNN FE 图缓存

对于使用 cuDNN Frontend 的算子（如 MaxPool FWD/INF），必须按 `(handle, shape, params)` 做图缓存，避免每次 launch 都重建图：

```cpp
struct MaxPoolFwdCacheKey {
    uint64_t handle_bits;
    int n, c, h, w, k, s, p;
    bool is_amp;
    // operator== + hash
};

static std::unordered_map<MaxPoolFwdCacheKey, MaxPoolFwdCache> s_maxpool_fwd_caches;
```

---

## 三、全局注册

### 3.1 声明 + 挂载

| 文件 | 操作 |
|------|------|
| `include/renaissance/backend/op_registry.h` | 声明 `void register_op_xxx();` |
| `src/backend/op_registry.cpp` | 在 `register_default_ops()` 中调用 |

### 3.2 Warmup 决策

在 `src/backend/op_registry.cpp` 的 `require_warmup()` 中：

| 实现方式 | 需要 warmup？ | 原因 |
|----------|:---:|------|
| cuDNN Frontend（如 MaxPool FWD/INF） | **是** | FE 需要在 capture 前建 execution plan 并执行一次 |
| cuDNN Legacy API（如 MaxPool BWD） | 否 | 首次调用时自初始化 |
| 手写 CUDA kernel（如 Dropout 全部） | 否 | 无预热需求，且可能因缺少 baseline 而崩溃 |

### 3.3 Stream 策略

在 `src/backend/op_stream_policy.cpp` 中指定默认 stream。池化/逐元素类算子通常用 `COMP_2`，与 GAP、BN 共用。

### 3.4 CMake 源文件

```cmake
# src/CMakeLists.txt
list(APPEND RENAISSANCE_SOURCES
    backend/ops/dtensor/maxpool_op.cpp
    backend/ops/dtensor/dropout_op.cpp
    backend/ops/dtensor/dropout_op.cu
)
```

---

## 四、ArchPlan 集成（如需自动构图）

如果算子需要被 `DeepLearningTask` 的 YAML → IR 流水线识别，完成以下步骤。

### 4.1 定义 LayerKind 与 LayerParams

```cpp
// include/renaissance/graph/arch_plan.h

enum class LayerKind : uint16_t {
    MaxPool,  // 已有
    Dropout,  // 新增
};

struct DropoutLayerParams {
    float p = 0.5f;
    bool operator==(const DropoutLayerParams& o) const { return p == o.p; }
};

// LayerParam variant 中加入 DropoutLayerParams
```

### 4.2 注册 LayerDescriptor

文件：`src/graph/layer_descriptor_registry.cpp`

每个 `LayerKind` 需要四个函数，打包为 `LayerDescriptor`（即 `std::tuple<InferFn, BuildFwdFn, BuildBwdFn, BuildInfFn>`）：

#### (a) 张量推断 `infer_xxx_tensors`

返回该层产生的所有 DTensor 描述。输出张量数量 = `output_indices` 引用的上限 + 1。

```cpp
// MaxPool: 产生 pool_output + pool_mask (2 个输出)
// Dropout: 产生 dropout_output + dropout_mask (2 个输出)
```

#### (b) 前向子图 `build_xxx_forward`

```cpp
SubgraphPattern::Node n;
n.op = using_amp() ? XXX_AMP_FWD : XXX_FP32_FWD;
n.output_indices = {0, 1};  // 两个输出
```

#### (c) 反向子图 `build_xxx_backward`

**关键：input_indices 只声明 "额外" 输入**。dY 由 Compiler 自动 prepend；X 对 MaxPool 由 Compiler 追加。

| 算子 | SubgraphPattern 声明 | Compiler 补充后实际输入 |
|------|---------------------|------------------------|
| MaxPool BWD | `input_indices = {0, 1}` (Y, mask) | `{dY, Y, mask, X}` → `{X}` |
| Dropout BWD | `input_indices = {1}` (mask) | `{dY, mask}` → `{X}` |

`output_indices = {}` — dX in-place，由 Compiler 路由到 X 的 DTensor ID。

#### (d) 推理子图 `build_xxx_inference`

与 FWD 类似，但使用 `XXX_INF` 枚举。即使 mask 不被消费，仍需分配以保持张量计数一致。

#### 注册

```cpp
static const LayerDescriptor dropout_desc = {
    infer_dropout_tensors, build_dropout_forward,
    build_dropout_backward, build_dropout_inference
};
// get_layer_descriptor() 中:
case LayerKind::Dropout: return dropout_desc;
```

### 4.3 Compiler 集成

文件：`src/graph/compiler.cpp`

在以下位置添加分支（按编译流程顺序）：

| 位置 | 作用 | MaxPool / Dropout 的值 |
|------|------|------------------------|
| `compile_primitive()` | 标记为 primitive（无内部子结构） | `case LayerKind::MaxPool:` / `case LayerKind::Dropout:` |
| `build_computation_graph()` FWD 索引映射 | 指定该层 forward 输出在全局 tensor 列表中的起始索引 | 均为 `idx = 0` |
| `build_computation_graph()` BWD grad 槽映射 | 指定梯度写回哪个 slot | 均为 `idx = -1`（in-place） |
| `build_computation_graph()` BWD 节点生成后处理 | **MaxPool 特殊**：追加 X 作为第 4 输入；Dropout：仅设置 output_ids | 见下方代码 |
| grad_id 追踪 | 让前一层找到正确的梯度 ID | 加入 `LayerKind::MaxPool` / `LayerKind::Dropout` |
| no-op break 列表 | INF 模式下的断点，避免无意义继续 | 加入 `LayerKind::Dropout` |

**BWD 节点生成后的 in-place 处理**（MaxPool 特殊）：

```cpp
// Dropout: dX 覆盖 X（2输入 → 1输出）
if (gn.compute_op == ComputeOp::DROPOUT_FP32_BWD || gn.compute_op == ComputeOp::DROPOUT_AMP_BWD) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) {
        gn.output_ids = {it->second};
    }
}
// MaxPool: dX 覆盖 X，且需要 X 作为 cuDNN Legacy API 的额外输入（4输入 → 1输出）
if (gn.compute_op == ComputeOp::MAXPOOL_FP32_BWD || gn.compute_op == ComputeOp::MAXPOOL_AMP_BWD) {
    auto it = layer_input_ids.find(l);
    if (it != layer_input_ids.end() && it->second >= 0) {
        gn.input_ids.push_back(it->second);  // 追加 X
        gn.output_ids = {it->second};         // dX in-place
    }
}
```

### 4.4 ArchPlan 辅助文件

| 文件 | 需要添加的内容 |
|------|---------------|
| `src/graph/arch_plan_expand.cpp` | `expand_primitive_impl()`: `NodeKind::Dropout` → `ArchLayer{LayerKind::Dropout, ...}` |
| `src/graph/arch_plan_shape.cpp` | `recompute_shapes_from()`: 下采样算子计算 OH/OW，形状不变算子直接 break |
| `src/graph/arch_plan_format.cpp` | `kind_name()`、`params_str()` |
| `src/graph/arch_plan_normalize.cpp` | `get_effective_output_c_at()`: 输出 C 不变的层归入 default 分支（continue） |
| `src/graph/arch_plan_yaml.cpp` | `kind_from_name()`、`to_yaml()`、`from_yaml()` |
| `src/graph/arch_plan_merge.cpp` | 如有融合规则，更新 |

---

## 五、基础设施扩展（按需）

仅当算子需要跨层共享的全局资源时才需要。Dropout 的 per-RANK 随机种子是典型例子。

### 5.1 MemoryPlan Baseline

```cpp
// include/renaissance/graph/memory_plan.h
struct BaselineIds {
    int32_t dropout_seed = -1;  // shape {1,1,1,2}, DType::INT32
};
void set_baseline_dropout_seed(int32_t id) noexcept { baseline_.dropout_seed = id; }

// src/graph/memory_plan.cpp — alloc_baseline_dtensors() 的两个重载都要加
Shape seed_shape{1, 1, 1, 2};
auto seed_dt = alloc_impl(seed_shape, DType::INT32, Region::S_SCALAR_INT32);
baseline_.dropout_seed = seed_dt.id;
```

### 5.2 TaskBase 初始化

```cpp
// include/renaissance/task/task_base.h
void set_dropout_seed_id(int32_t id) { memory_plan_.set_baseline_dropout_seed(id); }

// src/task/task_base.cpp — init_all() 中
// 用 SplitMix64 从 global_seed + rank_id 派生 per-RANK seed
// 通过 transfer_to_rank() 写入 GPU 显存
```

### 5.3 SimpleTask 暴露

```cpp
// include/renaissance/task/simple_task.h
using TaskBase::set_dropout_seed_id;
```

**为什么 SimpleTask 需要手动设置？** `DeepLearningTask` 在构造期自动调用 `alloc_baseline_dtensors()`；`SimpleTask` 不走这个流程，测试代码必须手动 `alloc()` + `set_xxx_id()`。

---

## 六、测试

### 6.1 正确性测试（`tests/op/`）

| 算子类型 | 策略 | 示例 |
|----------|------|------|
| 数值可严格对齐（如 MaxPool） | PyTorch 生成 TSR 参考数据 → MSE 对比 | `test_maxpool_fwd_bwd.cpp` |
| 存在随机性（如 Dropout） | 属性自验证：`mask==1 ⇒ y==x*scale`，`mask==0 ⇒ y==0` | `test_dropout.cpp` |
| 梯度归属验证 | 小张量 + 打印可视化，手动验证梯度位置 | `test_maxpool_visual.cpp` |

- 使用 `SimpleTask` 构建独立的 FWD/BWD/INF 图
- 支持 `--cpu` / `--gpu` / `--amp`
- 数学测试用 `run()` 跑一次，不要用 `run_iter()`

### 6.2 性能测试（`tests/perf/`）

- 固定输入形状，预热 5 次，计时 100 次取平均
- FWD/BWD/INF 分别计时
- 支持 `--cpu` / `--gpu` / `--amp`

### 6.3 CMake 注册

```cmake
# tests/op/CMakeLists.txt 或 tests/perf/CMakeLists.txt
add_executable(test_xxx test_xxx.cpp)
target_link_libraries(test_xxx PRIVATE renaissance)
target_compile_definitions(test_xxx PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_xxx)
endif()
set_target_properties(test_xxx PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/op
    WIN32_EXECUTABLE FALSE
)
```

---

## 七、AMP 避坑指南

### 7.1 cuDNN FE 的 `set_dim` 必须用 `padded_c()`

AMP 模式下 FP16 张量的 C 通道被自动 padding 到 8 的倍数（TensorCore 对齐）。`set_dim({N, C, H, W})` 使用逻辑 C=2 会导致 cuDNN FE 报 error code 8。

```cpp
// 错误
attr_x->set_dim({N, dt_x.c(), H, W});

// 正确
int64_t PC = dt_x.padded_c();
attr_x->set_dim({N, PC, H, W})
      .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(), ...});
```

**stride 本身已经基于 padded_c 计算**，不需要修改。

### 7.2 cuDNN Legacy API 用 `SetTensor4dDescriptorEx`

BWD 使用 `cudnnSetTensor4dDescriptor` 时 NHWC 格式会按逻辑 C 计算紧凑 stride，但 AMP 内存按 padded_c 布局，导致读错位置。

```cpp
cudnnSetTensor4dDescriptorEx(cache.x_desc, CUDNN_TENSOR_NHWC, dt,
    dt_x.n(), dt_x.c(), dt_x.h(), dt_x.w(),
    dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
    dt_x.h_stride_cuda(), dt_x.w_stride_cuda());
```

### 7.3 手写 CUDA kernel 用 stride 而非逻辑 C

```cpp
// 正确：用 stride 跳过 padding
int idx = n * feat_n_stride + c;  // 而非 n * feat_c * feat_h * feat_w + c
```

### 7.4 Mask 的 Region 和 DType

- MaxPool mask: `Region::S_MASK`, `DType::INT8`（局部偏移编码或 0/1）
- Dropout mask: `Region::S_MASK`, `DType::INT8`（0/1 布尔）
- S_MASK 在 CUDA 下对齐到 8，padded_c 也是 8 的倍数

---

## 八、最终 Checklist

| # | 文件 | 必须？ | 操作 |
|---|------|:---:|------|
| 1 | `include/renaissance/graph/op_kind.h` | ✅ | 参数结构体 + ComputeOp 枚举 + OpParams variant |
| 2 | `src/graph/op_kind.cpp` | ✅ | `compute_op_to_string()` + `format_params()` |
| 3 | `src/backend/ops/dtensor/xxx_op.cpp` | ✅ | CPU + CUDA launcher + 注册函数 |
| 4 | `src/backend/ops/dtensor/xxx_op.cu` | 按需 | 手写 CUDA kernel |
| 5 | `include/renaissance/backend/op_registry.h` | ✅ | 声明注册函数 |
| 6 | `src/backend/op_registry.cpp` | ✅ | 挂载 + warmup 决策 |
| 7 | `src/backend/op_stream_policy.cpp` | ✅ | Stream 策略 |
| 8 | `src/CMakeLists.txt` | ✅ | 添加源文件 |
| 9 | `include/renaissance/graph/arch_plan.h` | 如需 | LayerKind + LayerParams |
| 10 | `src/graph/layer_descriptor_registry.cpp` | 如需 | 四个函数 + LayerDescriptor 注册 |
| 11 | `src/graph/compiler.cpp` | 如需 | 5 处分支（详见 4.3） |
| 12 | `src/graph/arch_plan_expand.cpp` | 如需 | `expand_primitive_impl()` |
| 13 | `src/graph/arch_plan_shape.cpp` | 如需 | `recompute_shapes_from()` |
| 14 | `src/graph/arch_plan_format.cpp` | 如需 | `kind_name()` + `params_str()` |
| 15 | `src/graph/arch_plan_normalize.cpp` | 如需 | `get_effective_output_c_at()` |
| 16 | `src/graph/arch_plan_yaml.cpp` | 如需 | YAML 序列化/反序列化 |
| 17 | `include/renaissance/graph/memory_plan.h` | 按需 | Baseline 字段 |
| 18 | `src/graph/memory_plan.cpp` | 按需 | Baseline 分配 |
| 19 | `include/renaissance/task/task_base.h` | 按需 | setter 接口 |
| 20 | `src/task/task_base.cpp` | 按需 | `init_all()` 初始化 |
| 21 | `include/renaissance/task/simple_task.h` | 按需 | `using TaskBase::set_xxx` |
| 22 | `tests/op/CMakeLists.txt` | ✅ | 正确性测试目标 |
| 23 | `tests/perf/CMakeLists.txt` | 建议 | 性能测试目标 |
| 24 | `tests/op/test_xxx.cpp` | ✅ | 正确性测试 |
| 25 | `tests/perf/test_xxx_perf.cpp` | 建议 | 性能测试 |

---

> **黄金法则**：先 SimpleTask 跑通，再接入 ArchPlan。分阶段验证，避免一次性改动过大导致调试困难。
