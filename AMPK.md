# 【今日话题：AMP的实现】

# 【小伙伴S】

## 执行摘要

基于对当前代码的全面调研，分析了实现 AMP grad scaling 机制的可行性和具体方案。**当前代码基础设施已完备**，包含 NaN 检测、优化器图构建、标量管理、AMP 模式切换等核心组件。实现图内条件分支需要**图级别控制流改造**，建议分阶段实施。

---

## 1. 当前代码基础设施分析

### 1.1 ✅ 标量 DTensor 管理系统（完备）

**位置**：`src/graph/memory_plan.cpp:374-376`

```cpp
baseline_.has_nan = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
baseline_.lr      = alloc_impl(scalar_shape, DType::FP32,  Region::S_SCALAR_FP32).id;
baseline_.scaling = alloc_impl(scalar_shape, DType::FP32,  Region::S_SCALAR_FP32).id;
```

**关键发现**：
- ✅ `scaling` DTensor 已分配（`Region::S_SCALAR_FP32`）
- ✅ `has_nan` 标志已分配（`Region::S_SCALAR_INT32`）  
- ✅ 所有标量 DTensor 通过 `baseline_` 结构统一管理
- ✅ 优化器标量（`beta`、`wd`、`eps`、`tc`）条件分配（L382-394）

### 1.2 ✅ NaN 检测机制（完备）

**位置**：`src/graph/compiler.cpp:1672-1696`

```cpp
// 11. RANGE_CHECK_NAN - 梯度NaN检查
{
    if (nan_flag_id >= 0) {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_CHECK_NAN;
        
        std::vector<Region> nan_regions = {
            Region::G_BN_BIAS, Region::G_BN_WEIGHT,
            Region::G_FC_BIAS, Region::G_FC_WEIGHT,
            Region::G_FIRST_CONV, Region::G_DEEP_CONV
        };
        // ... 遍历所有梯度区域，input_ranges 聚合
        node.output_ids.push_back(nan_flag_id);  // ← 写入 has_nan 标志
        train_cg.append(GraphId::CAST_AND_CHECK, node);
    }
}
```

**位置**：`src/backend/ops/range/check_op.cpp:52-72`

```cpp
int32_t* has_nan_ptr = static_cast<int32_t*>(
    ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));

cudaError_t err = cudaMemsetAsync(has_nan_ptr, 0, sizeof(int32_t), s);  // 先清零
for (size_t i = 0; i < node.input_ranges.size(); ++i) {
    launch_check_nan_cuda_impl(has_nan_ptr, data, elements, s);  // 发现 NaN 时写 1
}
```

**关键发现**：
- ✅ NaN 检查在 `CAST_AND_CHECK` 图中执行（所有梯度 allreduce 之后）
- ✅ `has_nan` 标志在 GPU 端写入，无需 HOST 同步
- ✅ 检查范围覆盖所有 FP32 梯度区域（`G_*` 系列）
- ✅ 实现：先 `cudaMemsetAsync(..., 0)` 清零，发现 NaN 时原子写 1

### 1.3 ✅ 优化器图构建系统（完备）

**位置**：`src/graph/compiler.cpp:1450-1640`

**结构**：
- **Weight 更新**：每个优化器一个 RangeOp（`RANGE_UPDATE_WEIGHT_*`）
- **Bias 更新**：每个优化器一个 RangeOp（`RANGE_UPDATE_BIAS_*`）
- **LARS 特殊处理**：两阶段（`LARS_COMPUTE_TRUST_RATIO` + `LARS_UPDATE`）

**示例（SGD_MOMENTUM Weight）**：L1529-1531
```cpp
case OptimizerKind::SGD_MOMENTUM:
    weight_op = RangeOp::RANGE_UPDATE_WEIGHT_MOMENTUM;
    weight_needs_m = true; break;
```

**标量 DTensor 传递**：L1630
```cpp
node.input_ids.push_back(scalar_ids.lr);      // 学习率
node.input_ids.push_back(scalar_ids.beta);    // momentum 系数
node.input_ids.push_back(scalar_ids.wd);      // weight decay
```

**关键发现**：
- ✅ 优化器标量通过 `OptimizerScalarIds` 统一传递
- ✅ 所有优化器（SGD、LARS、Adam、AdamW）均支持
- ✅ 标量通过 `input_ids` 传递，kernel 内部读取

### 1.4 ✅ Scaling Factor 初始化（需补充）

**当前状态**：
- ✅ `scaling` DTensor 已分配（`baseline_.scaling`）
- ✅ 非 AMP 模式初始化为 1.0f：`compiler.cpp:748-749`
- ❌ AMP 模式初始化**缺失**

**位置**：`src/graph/compiler.cpp:748-749`
```cpp
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(1.0f));
```

**问题**：此代码对 AMP 和非 AMP 模式**统一设置 1.0f**，未区分。

### 1.5 ✅ AMP 模式检测（完备）

**位置**：`include/renaissance/core/global_registry.h:327`

```cpp
bool using_amp() const;
```

**实现**：`src/core/global_registry.cpp`
```cpp
bool GlobalRegistry::using_amp() const {
    return fixed_using_amp_.load(std::memory_order_relaxed);
}
```

**使用位置**：
- `compiler.cpp:1411`：`bool amp_on = GlobalRegistry::instance().using_amp();`
- `memory_plan.cpp:399`：`bool amp = GlobalRegistry::instance().using_amp();`

**关键发现**：
- ✅ AMP 状态全局可查询
- ✅ `fixed_using_amp_` 为 atomic<bool>，线程安全
- ✅ 编译阶段已知 AMP 状态，可用于条件初始化

### 1.6 ❌ 图内条件分支（不存在）

**当前 ComputationGraph 限制**：
- ✅ 支持 `Kind::COMPUTE`（DTensor 级操作）和 `Kind::RANGE`（Region 级批量操作）
- ❌ **不支持条件分支节点**（`if-else` 控制流）

**现有 GraphNode 结构**：`include/renaissance/graph/computation_graph.h:38-56`
```cpp
struct GraphNode {
    enum class Kind : uint8_t { COMPUTE, RANGE };
    
    union {
        ComputeOp compute_op;
        RangeOp   range_op;
    };
    
    OpParams params;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> output_ids;
    std::vector<MemRange> input_ranges;   // RANGE 态
    std::vector<MemRange> output_ranges;  // RANGE 态
};
```

**缺失组件**：
- ❌ `Kind::BRANCH` 或 `Kind::IF` 枚举值
- ❌ `cond` 字段（条件 DTensor ID，如 `has_nan`）
- ❌ `true_branch`、`false_branch` 子图引用

---

## 2. 核心挑战分析

### 2.1 🚨 图内条件分支缺失

**用户需求**：
> "如果has_nan为true，那就跳过本batch的更新（不更新权重、也不更新动量），但是把scaling factor除以2"

**当前限制**：
- `ComputationGraph` 是**静态拓扑**（Phase B 编译时固化）
- CUDA Graph 要求**固定的执行流结构**（不能动态插入/删除节点）
- 当前 `GraphNode` 不支持**运行时分支决策**

### 2.2 🚨 Scaling Factor 动态更新

**用户需求**：
> "把scaling factor除以2"

**问题**：
- 当前 `scaling` DTensor 在 `init_all()` 时写入一次（初始值）
- **运行时修改** `scaling` 需要**额外的算子节点**
- 需要在 CUDA Graph 内实现 `scaling = scaling / 2.0f` 的 inplace 更新

### 2.3 🚨 优化器 Skip 机制

**用户需求**：
> "跳过本batch的更新（不更新权重、也不更新动量）"

**当前优化器 kernel 限制**：
- 所有 `RANGE_UPDATE_*` kernel **无条件执行**
- kernel 内部没有"跳过更新"的分支逻辑
- **方案 A**：修改所有 optimizer kernel（工作量大，侵入性强）
- **方案 B**：在图级别绕过优化器节点（需要分支支持）

---

## 3. 实现方案对比

### 方案 A：图内条件分支（完整方案）

#### 3.1 设计

**新增 GraphNode 类型**：
```cpp
enum class Kind : uint8_t { COMPUTE, RANGE, IF };  // 新增 IF

struct GraphNode {
    Kind kind = Kind::COMPUTE;
    
    union {
        ComputeOp compute_op;
        RangeOp   range_op;
        struct {  // IF 态专用字段
            int32_t cond_id;           // 条件 DTensor ID（has_nan）
            GraphId true_branch;       // NaN=true 时的子图
            GraphId false_branch;      // NaN=false 时的子图
        };
    };
    
    // ... 其他字段（IF 态不使用）
};
```

**新增子图**：
```cpp
enum class GraphId : uint8_t {
    // ... 现有子图
    OPTIMIZER_UPDATE_NAN_SKIP,    // has_nan=true 时的空操作图
    OPTIMIZER_UPDATE_NORMAL,      // has_nan=false 时的正常优化器图
    SCALING_DIV2,                 // scaling /= 2 更新图
};
```

#### 3.2 修改点

**文件 1**：`include/renaissance/graph/computation_graph.h`
- 新增 `Kind::IF`
- 扩展 `GraphNode` union（`cond_id`、`true_branch`、`false_branch`）
- 新增 `GraphId::OPTIMIZER_UPDATE_NAN_SKIP`、`OPTIMIZER_UPDATE_NORMAL`、`SCALING_DIV2`

**文件 2**：`src/backend/graph_executor.cpp`
- 实现 IF 节点执行逻辑：
  ```cpp
  if (node.kind == GraphNode::Kind::IF) {
      const int32_t* cond_ptr = /* 读取 node.cond_id */;
      int32_t cond_val = *cond_ptr;  // GPU → HOST（可用单字节 memcpy 减少开销）
      
      GraphId next_graph = (cond_val != 0) ? node.true_branch : node.false_branch;
      execute_graph(next_graph);  // 递归或迭代执行
  }
  ```

**文件 3**：`src/backend/ops/range/scaling_op.cpp`（新建）
- 实现 `RANGE_SCALING_DIV2` 算子：
  ```cpp
  __global__ void scaling_div2_kernel(float* scaling) {
      scaling[0] *= 0.5f;
  }
  ```

**文件 4**：`src/graph/compiler.cpp`
- 修改 `build_auxiliary_graphs()`：
  - 拆分 `OPTIMIZER` 为 `OPTIMIZER_UPDATE_NORMAL`（现有逻辑）
  - 创建 `OPTIMIZER_UPDATE_NAN_SKIP`（空图或仅清零梯度）
  - 创建 `SCALING_DIV2` 图（`RANGE_SCALING_DIV2`）
  - 在 `CAST_AND_CHECK` 后插入 IF 节点：
    ```cpp
    GraphNode if_node;
    if_node.kind = GraphNode::Kind::IF;
    if_node.cond_id = nan_flag_id;
    if_node.true_branch = GraphId::SCALING_DIV2;        // has_nan=true
    if_node.false_branch = GraphId::OPTIMIZER_UPDATE_NORMAL;  // has_nan=false
    train_cg.append(GraphId::OPTIMIZER, if_node);
    ```

#### 3.3 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ 完整实现用户需求 | ❌ **工程量大**（需要改动核心数据结构） |
| ✅ 真正的图内分支（GPU 端决策） | ❌ **CUDA Graph 兼容性风险**（IF 节点可能导致 capture 失败） |
| ✅ 无 HOST-GPU 同步开销 | ❌ 需要大量测试验证 |
| ✅ 可扩展到其他条件逻辑 | ❌ 可能破坏现有 CUDA Graph 优化 |

---

### 方案 B：HOST 端决策 + 动态图选择（简化方案）

#### 3.1 设计

**原理**：
- 将图决策**移出 CUDA Graph**
- 使用现有的 `GraphExecutor` HOST 端控制流
- 基于 `has_nan` 值选择执行哪个子图

**实现**：
```cpp
// GraphExecutor::execute_optimizer()
bool has_nan = check_nan_flag_on_host();  // D2H copy has_nan

if (has_nan) {
    execute_graph(GraphId::OPTIMIZER_UPDATE_NAN_SKIP);
    execute_graph(GraphId::SCALING_DIV2);
} else {
    execute_graph(GraphId::OPTIMIZER_UPDATE_NORMAL);
}
```

#### 3.2 修改点

**文件 1**：`src/backend/graph_executor.cpp`
- 实现 `check_nan_flag_on_host()`（类似 NANF_FINAL.md 的 `check_nan_flag()`）
- 新增 `execute_optimizer_with_nan_check()` 封装逻辑

**文件 2**：`src/graph/compiler.cpp`
- 拆分 `OPTIMIZER` 为两个独立子图（不使用 IF 节点）
- 创建 `SCALING_DIV2` 子图

#### 3.3 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ **工程量小**（不改动核心数据结构） | ❌ HOST-GPU 同步开销（每 batch 一次 D2H copy） |
| ✅ **低风险**（不破坏 CUDA Graph） | ❌ 不满足"图内分支"需求 |
| ✅ 易于测试和调试 | ❌ 性能略低于方案 A |
| ✅ 向后兼容（不影响现有图） | ❌ 需要修改训练循环逻辑 |

---

### 方案 C：优化器 Kernel 内部分支（最小侵入方案）

#### 3.1 设计

**原理**：
- 不修改图结构
- 在每个 optimizer kernel **内部**添加 `has_nan` 检查
- `if (*has_nan == 0) { /* 正常更新 */ }`

**实现**：
```cuda
__global__ void update_momentum_kernel(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    const int32_t* has_nan)  // 新增参数
{
    if (*has_nan != 0) return;  // 跳过更新
    
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _beta = *beta;
    // ... 原有逻辑
}
```

#### 3.2 修改点

**文件 1**：`src/backend/ops/range/optimizer_op.cu`
- 所有 5 个 optimizer kernel 新增 `const int32_t* has_nan` 参数
- 每个 kernel 开头添加 `if (*has_nan != 0) return;`

**文件 2**：`src/backend/ops/range/optimizer_op.cpp`
- 所有 launcher 函数传递 `has_nan_ptr`

**文件 3**：`src/graph/compiler.cpp`
- 修改 `build_auxiliary_graphs()` 中的 `node.input_ids`：
  ```cpp
  node.input_ids.push_back(scalar_ids.lr);
  node.input_ids.push_back(scalar_ids.beta);
  node.input_ids.push_back(scalar_ids.wd);
  node.input_ids.push_back(nan_flag_id);  // ← 新增
  ```

**文件 4**：`src/backend/ops/range/scaling_op.cpp`（新建）
- `RANGE_SCALING_DIV2` 实现
- 在 `OPTIMIZER` 后追加 `SCALING_DIV2` 节点

#### 3.3 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ **最小改动**（仅修改 optimizer kernel） | ❌ 仍需实现 `scaling /= 2` 的独立节点 |
| ✅ **无图结构变更**（保持 CUDA Graph 兼容性） | ❌ 需要额外实现 scaling 更新逻辑 |
| ✅ **性能优**（GPU 端分支，无 HOST 同步） | ❌ 每次优化器调用都检查 `has_nan`（轻微开销） |
| ✅ 易于测试（单元测试 kernel 即可） | ❌ Scaling 更新仍需要额外节点 |

---

## 4. 推荐方案：混合方案（方案 C + 简化的 Scaling 更新）

### 4.1 核心思路

1. **优化器 Skip**：使用**方案 C**（optimizer kernel 内部 `has_nan` 检查）
2. **Scaling 更新**：使用**简化的独立节点**（非条件分支，总是执行但效果不同）
3. **初始化**：补充 AMP 模式下的 `scaling` 初始化

### 4.2 具体实现

#### 阶段 1：AMP 初始 Scaling Factor（P0）

**宏定义**：`include/renaissance/core/global_config.h`
```cpp
#define AMP_INITIAL_SCALING_FACTOR (65536.0f)  // 2^16，PyTorch 默认值
```

**修改初始化**：`src/graph/compiler.cpp:748-749`
```cpp
bool amp_on = GlobalRegistry::instance().using_amp();
float init_scaling = amp_on ? AMP_INITIAL_SCALING_FACTOR : 1.0f;

memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(init_scaling));
```

#### 阶段 2：Optimizer Kernel 内部分支（P0）

**修改所有 optimizer kernel**：`src/backend/ops/range/optimizer_op.cu`

示例（Momentum）：
```cuda
OPTIMIZER_LAUNCH_BOUNDS
__global__ void update_momentum_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, size_t n,
    const float* __restrict__ lr, const float* __restrict__ wd,
    const float* __restrict__ beta,
    const int32_t* __restrict__ has_nan)  // ← 新增
{
    if (*has_nan != 0) return;  // ← 跳过更新
    
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _beta = *beta;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float g_i = g[i];
        m[i] = m[i] * _beta + g_i;
        w[i] = w[i] * (1.0f - _lr * _wd) - _lr * m[i];
    }
}
```

**修改所有 launcher**：`src/backend/ops/range/optimizer_op.cpp`

示例（Momentum Weight）：
```cpp
static void launch_opt_weight_momentum_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1) OPT_RESOLVE_RANGE(m, 2)
    if (r_w_sz == 0 || r_g_sz == 0 || r_m_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    float* m = OPT_RANGE_PTR(m);
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* beta = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = scalar_ptr<3>(mp, node.input_ids.data(), ctx.rank_for_context());  // ← 新增
    
    optimizer_cuda::launch_momentum_weight_cuda(w, g, m, r_w_sz / sizeof(float), 
                                                lr, wd, beta, has_nan, s);  // ← 传递
    OPT_CUDA_TAIL()
}
```

**修改节点构建**：`src/graph/compiler.cpp`

**Weight 更新节点**（L1630 附近）：
```cpp
node.input_ids.push_back(scalar_ids.lr);
node.input_ids.push_back(scalar_ids.beta);
node.input_ids.push_back(scalar_ids.wd);
node.input_ids.push_back(nan_flag_id);  // ← 新增
```

**Bias 更新节点**（L1638 附近）：
```cpp
node.input_ids.push_back(scalar_ids.lr);
node.input_ids.push_back(scalar_ids.beta);
// ... 其他标量
node.input_ids.push_back(nan_flag_id);  // ← 新增
```

#### 阶段 3：Scaling 条件更新（P1）

**新增 RangeOp**：`include/renaissance/graph/op_kind.h:289`
```cpp
enum class RangeOp : uint16_t {
    // ...
    RANGE_SCALING_COND_DIV2,  // 条件：if (has_nan) scaling /= 2
    // ...
};
```

**新增算子实现**：`src/backend/ops/range/scaling_op.cpp`（新建）
```cpp
#ifdef TR_USE_CUDA
__global__ void scaling_cond_div2_kernel(
    float* __restrict__ scaling,
    const int32_t* __restrict__ has_nan)
{
    if (*has_nan != 0) {
        scaling[0] *= 0.5f;
    }
}

static void launch_scaling_cond_div2_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    
    int32_t scaling_id = node.output_ids[0];
    int32_t nan_id = node.input_ids[0];  // has_nan
    
    float* scaling_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), 
                                       mp.get_dtensor(scaling_id).offset()));
    const int32_t* nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), 
                                       mp.get_dtensor(nan_id).offset()));
    
    scaling_cond_div2_kernel<<<1, 1, 0, s>>>(scaling_ptr, nan_ptr);
    
    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif
```

**注册算子**：`src/backend/ops/range/scaling_op.cpp`
```cpp
REGISTER_RANGE_OP(RANGE_SCALING_COND_DIV2, launch_scaling_cond_div2_cuda, 
                  launch_scaling_cond_div2_cpu);
```

**插入到图中**：`src/graph/compiler.cpp`（L1640 之后）
```cpp
// Scaling 条件更新（仅在 AMP 模式下）
if (amp_on && nan_flag_id >= 0) {
    GraphNode scaling_node;
    scaling_node.kind = GraphNode::Kind::RANGE;
    scaling_node.range_op = RangeOp::RANGE_SCALING_COND_DIV2;
    scaling_node.input_ids.push_back(nan_flag_id);
    scaling_node.output_ids.push_back(memory_plan.baseline().scaling);
    train_cg.append(GraphId::OPTIMIZER, scaling_node);
}
```

---

## 5. 实现优先级与风险

### 5.1 分阶段实施

| 阶段 | 内容 | 优先级 | 风险 | 工作量 |
|------|------|--------|------|--------|
| **阶段 1** | AMP 初始 scaling 初始化 | 🔴 P0 | 极低 | 0.5 天 |
| **阶段 2** | Optimizer kernel 内 `has_nan` 检查 | 🔴 P0 | 低 | 1 天 |
| **阶段 3** | Scaling 条件更新节点 | 🟡 P1 | 中 | 1.5 天 |
| **阶段 4** | 完整图内条件分支（方案 A） | 🟢 P2 | 高 | 3-5 天 |

### 5.2 风险点

**阶段 1**：
- ✅ 无风险（仅修改常量）
- ⚠️ 需验证 AMP 模式检测正确性

**阶段 2**：
- ⚠️ Optimizer kernel 性能回退（每次检查 `has_nan`，但开销极小）
- ⚠️ 需要修改所有 launcher 函数（约 10 个）
- ⚠️ 需要更新 `scalar_ptr<>` 模板（支持 `int32_t*`）

**阶段 3**：
- ⚠️ 新 RangeOp 注册流程需验证
- ⚠️ Scaling 更新时机需确认（在 optimizer 之后）
- ⚠️ CUDA Graph 兼容性需测试

**阶段 4**（不推荐）：
- 🚨 核心数据结构变更风险
- 🚨 CUDA Graph capture 可能失败
- 🚨 需要大量重构测试

---

## 6. 与现有 NaN 检测的集成

### 6.1 当前数据流

```
ALLREDUCE 梯度
    ↓
CAST_AND_CHECK 图
    ├─ RANGE_CAST_FP16_TO_FP32（梯度 FP16→FP32）
    └─ RANGE_CHECK_NAN（检查所有梯度，写入 has_nan）
```

### 6.2 增强后的数据流（推荐方案）

```
ALLREDUCE 梯度
    ↓
CAST_AND_CHECK 图
    ├─ RANGE_CAST_FP16_TO_FP32
    └─ RANGE_CHECK_NAN → has_nan 标志
         ↓
OPTIMIZER 图
    ├─ RANGE_UPDATE_WEIGHT_*（内部 if (*has_nan != 0) return;）
    ├─ RANGE_UPDATE_BIAS_*（内部 if (*has_nan != 0) return;）
    └─ RANGE_SCALING_COND_DIV2（if (*has_nan != 0) scaling *= 0.5;）
```

### 6.3 关键时序保证

- ✅ `RANGE_CHECK_NAN` 先清零 `has_nan`（`cudaMemsetAsync`）
- ✅ 检查所有梯度区域（原子写入 1）
- ✅ Optimizer kernel 在**同一 stream**（`StreamKind::UPDATE`）执行
- ✅ Scaling 更新在 optimizer **之后**执行（同 stream）

---

## 7. 初始 Scaling Factor 选择

### 7.1 推荐值

```cpp
#define AMP_INITIAL_SCALING_FACTOR (65536.0f)  // 2^16
```

**依据**：
- PyTorch 默认值：`torch.cuda.amp.GradScaler(init_scale=65536.0)`
- NVIDIA AMP 最佳实践：2^16 是稳定性与性能的平衡点
-过大 → 梯度上溢风险增加
- 过小 → 下溢损失精度

### 7.2 调整策略（可选）

```cpp
#define AMP_INITIAL_SCALING_FACTOR (32768.0f)  // 2^15（保守）
// 或
#define AMP_INITIAL_SCALING_FACTOR (131072.0f)  // 2^17（激进）
```

**建议**：
- 初步实现使用 **65536.0f**（与 PyTorch 对齐）
- 实际训练中根据 NaN 频率动态调整（通过 `SCALING_COND_DIV2`）

---

## 8. 最终方案总结

### 8.1 推荐方案（混合方案）

**核心策略**：
1. ✅ **不改动图结构**（保持 CUDA Graph 兼容性）
2. ✅ **Optimizer kernel 内部分支**（GPU 端决策，无 HOST 同步）
3. ✅ **独立 Scaling 更新节点**（条件更新，图级串行）

### 8.2 修改文件清单

| 文件 | 修改类型 | 工作量 |
|------|----------|--------|
| `include/renaissance/core/global_config.h` | 新增宏 | 2 行 |
| `src/graph/compiler.cpp` | 修改初始化 + 节点构建 | 30 行 |
| `src/backend/ops/range/optimizer_op.cu` | 修改 5 个 kernel | 10 行 |
| `src/backend/ops/range/optimizer_op.cpp` | 修改 10 个 launcher | 50 行 |
| `include/renaissance/graph/op_kind.h` | 新增枚举 | 1 行 |
| `src/backend/ops/range/scaling_op.cpp` | 新建文件 | 100 行 |
| `src/backend/ops/range/scaling_op.cu` | 新建文件（可选，CPU fallback） | 50 行 |

**总工作量**：约 **3-4 天**

### 8.3 测试验证

**单元测试**：
- `test_optimizer_nan_skip.cpp`：验证 optimizer kernel 正确跳过
- `test_scaling_div2.cpp`：验证 scaling 条件更新

**集成测试**：
- `test_dl_full_amp.cpp`：完整训练流程，AMP 模式
- 触发 NaN（手动设置梯度为 NaN），验证：
  - Optimizer 跳过更新（权重不变）
  - Scaling 正确减半
  - 后续 batch 正常恢复

---

## 9. 不推荐的方案

### 9.1 ❌ 方案 A（完整图内分支）

**理由**：
- 🚨 需要修改核心数据结构（`GraphNode`、`ComputationGraph`）
- 🚨 CUDA Graph 兼容性风险高
- 🚨 工程量巨大（3-5 天 + 大量测试）
- 🚨 可能破坏现有优化（如 CUDA Graph cache）

### 9.2 ❌ HOST 端决策（方案 B 纯粹版）

**理由**：
- 🚨 不满足用户"图内分支"需求
- 🚨 性能损失（每 batch D2H copy）
- 🚨 破坏 CUDA Graph 执行流

---

## 10. 与用户需求的对应

| 用户需求 | 推荐方案实现 | 满足度 |
|---------|-------------|--------|
| **宏定义初始 scaling factor** | `AMP_INITIAL_SCALING_FACTOR` 宏 | ✅ 100% |
| **AMP 模式初始化正确值** | 编译时 `using_amp()` 条件初始化 | ✅ 100% |
| **has_nan 为 true 时跳过更新** | Optimizer kernel 内 `if (*has_nan != 0) return;` | ✅ 100% |
| **has_nan 为 true 时 scaling 除以 2** | `RANGE_SCALING_COND_DIV2` 节点 | ✅ 100% |
| **图内分支（不在 HOST 判断）** | Optimizer kernel + Scaling node 均为 GPU 端决策 | ✅ 100% |
| **所有优化器支持** | 修改所有 5 种 optimizer kernel | ✅ 100% |

---

**报告日期**：2026-05-26
**调研方法**：代码审查、数据流追踪、现有实现分析
**置信度**：极高（基于实际代码，无推测）
**下一步**：等待用户确认方案选择，然后进入实施阶段





# 【小伙伴K】

> 调研日期：2026-05-26
> 调研范围：MemoryPlan baseline、SoftmaxCE AMP、RANGE_CHECK_NAN、优化器算子、Compiler 图构建、GraphExecutor

---

## 一、现状调研结论

### 1.1 BaselineIds：has_nan 与 scaling 已分配，但 scaling 未初始化

`MemoryPlan::BaselineIds`（`memory_plan.h:152-166`）默认全部初始化为 `-1`。

`alloc_baseline_dtensors()`（`memory_plan.cpp:352-395`）无条件分配了以下字段：

| 字段        | Region           | Dtype | 用途                |
| ----------- | ---------------- | ----- | ------------------- |
| `has_nan`   | `S_SCALAR_INT32` | INT32 | NaN 检测标志        |
| `scaling`   | `S_SCALAR_FP32`  | FP32  | Grad scaling factor |
| `loss`      | `R_RESULT`       | FP32  | 累计 loss           |
| `top1/top5` | `R_RESULT`       | FP32  | 准确率指标          |

**关键问题**：`scaling` 的 `init_config` 未被显式设置，当前由 `init_all()` 按默认 `InitKind::ZEROS` 初始化，导致 AMP 模式下 `scaling = 0.0f`，SoftmaxCE 的 AMP kernel 读取后 loss 和梯度全为 0。

### 1.2 SoftmaxCE AMP 已支持 scaling，但 scaling 值为 0

`softmax_ce_op.cu` 中 AMP 三件套（FWD/BWD/INF）已完整实现：

- **FWD**: `atomicAdd(loss, sample_loss * inv_batch * scaling)` — loss 被 scaling
- **BWD**: `g *= scaling * inv_scaling` — 梯度被 scaling
- **INF**: 同 FWD，但额外计算 top1/top5/pred

`compiler.cpp:1065-1076` 和 `:1201-1206` 在构建 SoftmaxCE 节点时，已将 `baseline.scaling` 和 `baseline.label_smce` 注入 `input_ids`。

**结论**：前向/反向的 grad scaling 链路已打通，唯一阻塞点是 `scaling` 初始值为 0。

### 1.3 RANGE_CHECK_NAN：写入了 has_nan，但无人读取

`src/backend/ops/range/check_op.cpp`：

- **CUDA**：先 `cudaMemsetAsync(has_nan_ptr, 0)`，再遍历梯度区域调用 `launch_check_nan_cuda_impl`，发现 NaN 则写入 `1`
- **CPU**：局部变量 `has_nan = 0`，遍历检查 `std::isnan`，发现则写入 `1`

`compiler.cpp:1672-1696` 在 `GraphId::CAST_AND_CHECK` 图中追加了 `RANGE_CHECK_NAN` 节点，其 `output_ids[0] = nan_flag_id`（即 `baseline.has_nan`）。

**关键问题**：`has_nan` 写入后，没有任何后续算子或 HOST 代码读取它。

### 1.4 优化器算子：9 个 kernel，均不接受 has_nan / scaling

`src/backend/ops/range/optimizer_op.cpp`：

| 算子            | input_ranges       | input_ids（标量）   |
| --------------- | ------------------ | ------------------- |
| WEIGHT_SGD      | weight, grad       | lr, wd              |
| WEIGHT_MOMENTUM | weight, grad, m    | lr, wd, beta        |
| WEIGHT_NESTEROV | weight, grad, m    | lr, wd, beta        |
| WEIGHT_ADAM     | weight, grad, m, v | lr, wd, b1, b2, eps |
| WEIGHT_ADAMW    | weight, grad, m, v | lr, wd, b1, b2, eps |
| BIAS_SGD        | bias, grad         | lr                  |
| BIAS_MOMENTUM   | bias, grad, m      | lr, beta            |
| BIAS_NESTEROV   | bias, grad, m      | lr, beta            |
| BIAS_ADAM       | bias, grad, m, v   | lr, b1, b2, eps     |

`CpuOpContext.input_ids[8]` 容量为 8，Adam weight 当前已用 5 个，加上 `has_nan` 后为 6，仍在容量内。

### 1.5 GraphExecutor：HOST 侧 NaN 分支（STUB）

`src/backend/graph_executor.cpp:128-135` 和 `:177-183`：

```cpp
bool has_nan = check_nan_flag();  // ← 硬编码返回 false
if (!has_nan) {
    launch(GraphId::OPTIMIZER);
    launch(GraphId::EMA_UPDATE);
} else {
    on_nan_detected();
}
```

`GraphExecutor` **未接入 `DeepLearningTask` 主流程**（`task_base.cpp` / `deep_learning_task.cpp` 中无引用）。`DeepLearningTask` 的 `run_gpu()` / `run_cpu()` 直接操作 `CapturedGraph::launch()`，不走 `GraphExecutor`。

**结论**：`GraphExecutor` 的 HOST 侧分支对当前主流程无影响，但可作为参考。

### 1.6 DeepLearningTask AMP 路径现状

`run_train_epoch_gpu()` 中 AMP 相关代码：

```cpp
bool using_amp = registry.using_amp();
// ...
if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }
// 然后直接执行 OPTIMIZER，无任何 NaN 分支
```

`g_gc` 对应 `GraphSlot::GRAD_CONVERT` = `GraphId::CAST_AND_CHECK`。该图在 AMP 时包含 `RANGE_CAST_FP16_TO_FP32` + `RANGE_CHECK_NAN`。执行完 `g_gc` 后，直接启动 `OPTIMIZER`，没有判断 `has_nan`。

---

## 二、方案目标

1. **宏定义 AMP 初始 scaling factor**（建议 `65536.0f` = 2^16）
2. **非 AMP 模式**：`scaling = 1.0f`；**AMP 模式**：`scaling = TR_AMP_INITIAL_SCALING`
3. **初始化时**把 `scaling` 写入正确的初始值
4. **优化器算子新增 `has_nan` 输入**
5. **图内分支**（非 HOST 判断）：
   - `has_nan == true`：跳过本 batch 的权重/动量更新，`scaling /= 2`
   - `has_nan == false`：照常执行更新

---

## 三、推荐方案：优化器 kernel 内部分支 + 独立 scaling 更新算子

### 3.1 为什么不用 CUDA Graph 条件节点

CUDA Graph 的条件节点（Conditional Nodes）需要 CUDA 12.x + 特殊 API，且与当前 `CapturedGraph` 架构不兼容。最实际的"图内分支"是在 **kernel 内部**做条件判断。

### 3.2 为什么把 scaling 更新拆分为独立算子

如果让每个优化器 kernel 都负责 `scaling /= 2`：

- Weight SGD + Momentum + Nesterov + Adam + AdamW（5 个）
- Bias SGD + Momentum + Nesterov + Adam（4 个）
- 共 **9 个 kernel**，每个都可能尝试修改 `scaling`
- 即使只有部分被启动（取决于架构层数），同一 batch 内 weight 和 bias 更新是**两个独立节点**
- 若两者都执行 `*scaling *= 0.5f`，`scaling` 会被连除两次

**因此必须确保 `scaling` 的修改只发生一次。**

### 3.3 方案架构

```
[CAST_AND_CHECK]  ──→ 包含 RANGE_CHECK_NAN（写入 has_nan）
       │
       ▼
[RANGE_GRAD_SCALING]  ──→ 新增算子：if (has_nan) scaling *= 0.5f
       │
       ▼
[OPTIMIZER]  ──→ Weight/Bias 各 kernel：if (has_nan) return; else 正常更新
```

**关键点**：

- `RANGE_GRAD_SCALING` 只修改 `scaling`，**不修改 `has_nan`**
- `has_nan` 由下一次 batch 的 `RANGE_CHECK_NAN` 自动清零（其内部先 `memset 0`）
- 所有优化器 kernel 读取 `has_nan` 做分支，**互不干扰**

---

## 四、具体改动清单

### 4.1 宏定义

在 `include/renaissance/core/types.h`（或项目全局 config 头文件）添加：

```cpp
/// AMP 初始 grad scaling factor（2^16 = 65536）
/// 参考：PyTorch GradScaler 默认值、NVIDIA 混合精度最佳实践
#define TR_AMP_INITIAL_SCALING 65536.0f
```

**选型理由**：

- 2^16 = 65536 是业界默认值（PyTorch、Apex）
- 足够大以充分利用 FP16 动态范围（~1e-5 到 6e4）
- 若梯度上溢，会自动通过 NaN 检测回退（`/= 2`）
- 若长期无 NaN，未来可扩展增长策略（`*= growth_factor`）

### 4.2 scaling 初始化

在 `DeepLearningTask::on_prepare()` 末尾（`lr_dtensor_id_` 查找之后）添加：

```cpp
// 设置 scaling factor 初始值
int32_t scaling_id = active_memory_plan_->baseline().scaling;
if (scaling_id >= 0) {
    float scaling_val = GlobalRegistry::instance().using_amp()
                        ? TR_AMP_INITIAL_SCALING
                        : 1.0f;
    active_memory_plan_->set_init_config(scaling_id,
        InitConfig{scaling_val, InitKind::CONSTANTS, FanMode::FAN_IN});
}
```

非 AMP 模式下 `scaling = 1.0f`，确保 SoftmaxCE FWD/BWD 的 `*scaling` 不影响 FP32 数值。

### 4.3 新增 RANGE_GRAD_SCALING 算子

**文件**：`src/backend/ops/range/grad_scaling_op.cpp`（新建）

**CUDA 实现**：

```cpp
static void launch_range_grad_scaling_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    int32_t has_nan_id = node.input_ids[0];
    int32_t scaling_id = node.input_ids[1];

    const int32_t* has_nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), mp.get_dtensor(has_nan_id).offset()));
    float* scaling_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), mp.get_dtensor(scaling_id).offset()));

    // 启动单 block 单 thread 的极简 kernel
    launch_grad_scaling_kernel<<<1, 1, 0, s>>>(has_nan_ptr, scaling_ptr);
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

**CPU 实现**：

```cpp
static void launch_range_grad_scaling_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_inputs < 2) return;

    int32_t has_nan_id = op_ctx->input_ids[0];
    int32_t scaling_id = op_ctx->input_ids[1];

    const int32_t* has_nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), mp->get_dtensor(has_nan_id).offset()));
    float* scaling_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), mp->get_dtensor(scaling_id).offset()));

    if (*has_nan_ptr != 0) {
        *scaling_ptr *= 0.5f;
    }
}
```

**注册**：

```cpp
void register_op_range_grad_scaling() {
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_GRAD_SCALING)];
    entry.op = RangeOp::RANGE_GRAD_SCALING;
    entry.launch_cpu = launch_range_grad_scaling_cpu;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_range_grad_scaling_cuda;
#endif
}
```

**CUDA kernel**（`.cu` 文件）：

```cuda
__global__ void launch_grad_scaling_kernel(const int32_t* has_nan, float* scaling) {
    if (*has_nan != 0) {
        *scaling *= 0.5f;
    }
}
```

### 4.4 修改优化器算子（9 个 kernel）

以 `RANGE_UPDATE_WEIGHT_SGD` 为例，CPU 和 CUDA 均添加 `has_nan` 输入：

**CPU launcher 修改**：

```cpp
static void launch_opt_weight_sgd_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    // 现有防护检查：num_input_ranges >= 2, num_inputs >= 2
    // 新增 has_nan 后：num_inputs >= 3
    if (!mp || op_ctx->num_input_ranges < 2 || op_ctx->num_inputs < 3) return;

    // 读取 has_nan（新增 input_ids[2]）
    int32_t has_nan_id = op_ctx->input_ids[2];
    const int32_t* has_nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), mp->get_dtensor(has_nan_id).offset()));
    if (*has_nan_ptr != 0) {
        return;  // 跳过本 batch 更新
    }

    // 原有逻辑不变
    float lr = OPT_CPU_SCALAR(0);
    float wd = OPT_CPU_SCALAR(1);
    OPT_CPU_RANGE_PTR(w, 0);
    OPT_CPU_GRAD(1);
    sgd_update_cpu(wp, gp, OPT_CPU_N, lr, wd);
}
```

**CUDA launcher 修改**：

```cpp
static void launch_opt_weight_sgd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    // ... 原有 stream 获取代码 ...

    OPT_CUDA_HEAD(si, s);
    OPT_RESOLVE_RANGE(w, 0);
    OPT_RESOLVE_RANGE(g, 1);

    // 新增：解析 has_nan
    int32_t has_nan_id = node.input_ids[2];
    const DTensor& nan_dt = mp.get_dtensor(has_nan_id);
    const int32_t* has_nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));

    // 注意：CUDA Graph 不支持 HOST 分支，kernel 内部判断
    // 由于现有 CUDA kernel 是纯计算 kernel，不支持条件分支，
    // 需要新增一个包装 kernel 或修改底层 kernel 签名
    // ...（详见 4.6 讨论）
}
```

**所有需修改的算子**：

- `RANGE_UPDATE_WEIGHT_SGD` / `BIAS_SGD`
- `RANGE_UPDATE_WEIGHT_MOMENTUM` / `BIAS_MOMENTUM`
- `RANGE_UPDATE_WEIGHT_NESTEROV` / `BIAS_NESTEROV`
- `RANGE_UPDATE_WEIGHT_ADAM` / `BIAS_ADAM`
- `RANGE_UPDATE_WEIGHT_ADAMW`（bias 无 AdamW 专用，Adam bias 复用）

### 4.5 Compiler 图构建修改

**`include/renaissance/backend/graph_executor.h`**：

```cpp
struct OptimizerScalarIds {
    int32_t lr    = -1;
    int32_t beta  = -1;
    int32_t beta2 = -1;
    int32_t tc    = -1;
    int32_t wd    = -1;
    int32_t eps   = -1;
    int32_t has_nan = -1;   // ← 新增
    int32_t scaling = -1;   // ← 新增（供 RANGE_GRAD_SCALING 使用）
};
```

**`src/graph/compiler.cpp`**：

1. `create_memory_plans()`（`:750-760`）：

   ```cpp
   scalar_ids.has_nan  = b.has_nan;
   scalar_ids.scaling  = b.scaling;
   ```

2. `build_auxiliary_graphs()` 中，在 `CAST_AND_CHECK` 之后、`OPTIMIZER` 之前，追加 `RANGE_GRAD_SCALING`：

   ```cpp
   // 10.5. RANGE_GRAD_SCALING - AMP grad scaling 回退
   if (amp_on && nan_flag_id >= 0 && scalar_ids.scaling >= 0) {
       GraphNode gs_node;
       gs_node.kind = GraphNode::Kind::RANGE;
       gs_node.range_op = RangeOp::RANGE_GRAD_SCALING;
       gs_node.input_ids.push_back(nan_flag_id);
       gs_node.input_ids.push_back(scalar_ids.scaling);
       train_cg.append(GraphId::GRAD_CONVERT, gs_node);  // 复用 GRAD_CONVERT slot
   }
   ```

3. 优化器 Weight 节点（`:1570-1578`）：

   ```cpp
   node.input_ids.push_back(scalar_ids.lr);
   node.input_ids.push_back(scalar_ids.wd);
   // ... beta, beta2, eps ...
   node.input_ids.push_back(scalar_ids.has_nan);  // ← 追加到末尾
   ```

4. 优化器 Bias 节点（`:1630-1637`）：

   ```cpp
   node.input_ids.push_back(scalar_ids.lr);
   // ... beta, beta2, eps ...
   node.input_ids.push_back(scalar_ids.has_nan);  // ← 追加到末尾
   ```

**注意**：`input_ids` 追加顺序变更后，需在 `optimizer_op.cpp` 中同步修改 `OPT_CPU_SCALAR(idx)` 宏的索引偏移。

### 4.6 CUDA 路径的特殊处理

当前优化器 CUDA kernel 是**纯计算 kernel**（如 `launch_sgd_weight_cuda(float* w, const float* g, size_t, const float* lr, const float* wd, cudaStream_t)`），不支持条件分支。

**两种实现方式**：

**方式 A：修改底层 kernel 签名（推荐）**

为每个底层 kernel 新增 `const int32_t* has_nan` 参数，kernel 内部第一行判断：

```cuda
__global__ void sgd_weight_kernel(float* w, const float* g, size_t n,
                                  float lr, float wd,
                                  const int32_t* has_nan) {
    if (*has_nan != 0) return;
    // ... 原有逻辑 ...
}
```

**优点**：真正的单 kernel 图内分支，零额外启动开销。
**缺点**：需修改全部 9 个 `.cu` 文件中的 kernel 定义和 launch wrapper。

**方式 B：HOST 侧前置 wrapper（不推荐，违背"图内判断"原则）**

在 CUDA launcher 中做 HOST 分支：

```cpp
int32_t has_nan_val = 0;
cudaMemcpy(&has_nan_val, has_nan_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost);
if (has_nan_val == 0) {
    launch_sgd_weight_cuda(...);  // 启动 kernel
}
```

**此方式违背用户"图内判断"要求，明确排除。**

**结论**：采用方式 A，修改全部 9 个底层 CUDA kernel 签名。

### 4.7 DeepLearningTask 执行流程修改

`run_train_epoch_gpu()` 中，当前 AMP 流程：

```cpp
// Phase 3: FIRST_LAYER_BWD ‖ DEEP_ALLREDUCE
if (!frozen) { ... }
if (g_dar) cudaGraphLaunch(g_dar, s_up);
sync_comp(); sync_up();

if (using_amp && g_gc) { cudaGraphLaunch(g_gc, s_up); sync_up(); }  // CAST_AND_CHECK

lr = fetch_lr_for_batch(batch);
cudaMemcpyAsync(lr_dev_ptr, ...);
if (g_far) cudaGraphLaunch(g_far, s_up);
if (g_wu) cudaGraphLaunch(g_wu, s_up);  // OPTIMIZER
sync_up();
```

**修改后**：

```cpp
if (using_amp && g_gc) {
    cudaGraphLaunch(g_gc, s_up);  // CAST_AND_CHECK + RANGE_GRAD_SCALING
    sync_up();
}

// OPTIMIZER 内部现在自带 has_nan 分支，无需 HOST 判断
if (g_far) cudaGraphLaunch(g_far, s_up);
if (g_wu) cudaGraphLaunch(g_wu, s_up);
sync_up();
```

**无需修改 `run_cpu()`**：`CapturedGraph::launch()` 遍历 `cpu_ops_` 顺序执行，`RANGE_GRAD_SCALING` 作为普通 `CpuOp` 被自动调用。

---

## 五、风险与待决策项

### 5.1 风险

1. **CUDA kernel 签名变更影响面大**：9 个 `.cu` 文件需同步修改，需确保编译通过。
2. **`input_ids` 索引偏移**：优化器算子宏（`OPT_CPU_SCALAR`）硬编码了索引，追加 `has_nan` 后需全部 +1。
3. **CPU 路径 `has_nan` 清零时机**：`RANGE_CHECK_NAN` CPU 实现把 `has_nan` 写入 DTensor，下一次 batch 的 `RANGE_CHECK_NAN` 会先 `memset 0`。需确认 `CapturedGraph` 中 `CAST_AND_CHECK` 图的节点顺序（cast 先还是 check 先）。

### 5.2 待决策

1. **scaling 增长策略**：当前方案只实现了 NaN 时的回退（`/= 2`）。是否需要实现无 NaN 时的增长（如连续 N 个 batch 无 NaN 则 `*= 2`）？
   - **建议**：一期仅实现回退，增长策略留给二期。

2. **EMA_UPDATE 是否也要跳过**：当 `has_nan == true` 时，`EMA_UPDATE` 是否也跳过？
   - **建议**：是。EMA 更新依赖主模型权重，若权重未更新，EMA 也不应更新。Compiler 中 `EMA_UPDATE` 节点是否需要添加 `has_nan` 输入？
   - 但 `RANGE_EMA_PARAM_UPDATE` 当前是独立算子，尚未支持条件分支。若 EMA 更新也需要跳过，需同步修改 `ema_op.cpp`。

3. **是否有更简洁的替代方案**：例如把 `has_nan` 和 `scaling` 的更新合并到 `RANGE_CHECK_NAN` 算子内部？
   - `RANGE_CHECK_NAN` 当前职责单一（检查 NaN），若混入 scaling 回退，违反 SRP。
   - **建议**：保持独立算子 `RANGE_GRAD_SCALING`，职责清晰。

---

## 六、实施优先级

| 优先级 | 任务                                              | 预估工作量                       |
| ------ | ------------------------------------------------- | -------------------------------- |
| P0     | 宏定义 + scaling 初始化（`on_prepare`）           | 小（1 文件，10 行）              |
| P0     | 新增 `RANGE_GRAD_SCALING` 算子（CPU+CUDA+注册）   | 中（2 新文件，~80 行）           |
| P0     | 修改 9 个优化器 CPU launcher（`has_nan` 分支）    | 中（1 文件，~90 行）             |
| P0     | Compiler 修改（`input_ids` 追加）                 | 中（1 文件，~20 行）             |
| P1     | 修改 9 个优化器 CUDA kernel 签名                  | 大（9 个 `.cu` 文件）            |
| P1     | `run_train_epoch_gpu()` 移除 HOST 侧 AMP 特殊处理 | 小（确认 `g_gc` 图已包含新节点） |
| P2     | EMA_UPDATE 的 has_nan 支持                        | 中（可选）                       |
| P2     | scaling 增长策略                                  | 中（可选，二期）                 |

---

## 七、验证计划

1. **单元测试**：编译 `--amp` 模式，确认 `test_dl_full.exe --amp` 不再崩溃
2. **数值测试**：3 epoch MNIST MLP，AMP 模式下 loss 应收敛（与 FP32 接近但不完全相同，因精度差异）
3. **NaN 注入测试**：临时在 backward 中注入 NaN，确认：
   - 优化器不更新权重
   - `scaling` 从 65536 → 32768 → 16384 ...
   - 后续 batch 恢复正常



# 【小伙伴D】

## 1. 背景

AMP (Automatic Mixed Precision) 用 FP16 替代部分 FP32（特征图和部分权重），其关键机制是 **grad scaling**：将 FP32 loss 乘以 scaling factor 再反向传播，防止 FP16 梯度下溢为 0。当检测到梯度 NaN 时，需要跳过本轮参数更新，并将 scaling factor 减半，避免持续的数值不稳定。

## 2. 现状调研

### 2.1 当前图执行流程（per batch）

```
Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD        (s_up | s_c1)
Phase 2: DEEP_FWD_BWD ‖ XFER(next)          (s_c1 | s_trans)
Phase 3: FIRST_LAYER_BWD ‖ DEEP_ALLREDUCE   (s_c1 | s_up)
Phase 4: CAST_AND_CHECK (仅 AMP)            (s_up)   ← 写 has_nan DTensor
Phase 5: LR 写入 → FIRST_ALLREDUCE          (s_up)
Phase 6: OPTIMIZER                           (s_up)   ← 当前不读 has_nan
```

来源：[`deep_learning_task.cpp`](file:///r:/renaissance/src/task/deep_learning_task.cpp#L968-L1013)

### 2.2 GraphId 与 GraphSlot

| GraphId          | GraphSlot       | 包含节点                           | 流     |
| ---------------- | --------------- | ---------------------------------- | ------ |
| `CAST_AND_CHECK` | `GRAD_CONVERT`  | FP16→FP32 cast + `RANGE_CHECK_NAN` | UPDATE |
| `OPTIMIZER`      | `WEIGHT_UPDATE` | Weight 更新 + Bias 更新 (RangeOp)  | UPDATE |

来源：[computation_graph.h](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L73-L95), [deep_learning_task.cpp](file:///r:/renaissance/src/task/deep_learning_task.cpp#L34-L66)

### 2.3 OPTIMIZER 图结构（compiler.cpp L1472-L1641）

非 LARS 优化器的 Weight/Bias 更新各为 **一个 RangeOp 节点**：

- Weight: `RANGE_UPDATE_WEIGHT_{SGD|MOMENTUM|NESTEROV|ADAM|ADAMW}`
  - input_ranges: [w_range, g_range, (m_range), (v_range)]
  - output_ranges: [w_range, (m_range), (v_range)]
  - input_ids: [lr, wd, (beta), (beta2), (eps)]  ← **没有 has_nan！**
- Bias: `RANGE_UPDATE_BIAS_{SGD|MOMENTUM|NESTEROV|ADAM}`
  - input_ranges: [bw_range, bg_range, (bm_range), (bv_range)]
  - output_ranges: [bw_range, (bm_range), (bv_range)]
  - input_ids: [lr, (beta), (beta2), (eps)]  ← **没有 has_nan！**

来源：[compiler.cpp](file:///r:/renaissance/src/graph/compiler.cpp#L1521-L1641)

### 2.4 CAST_AND_CHECK 图结构（compiler.cpp L1653-L1696）

AMP 模式下包含两个子步骤：

1. `RANGE_CAST_FP16_TO_FP32` × N：将 FP16 梯度转为 FP32（仅 AMP）
2. `RANGE_CHECK_NAN`：检查所有梯度 Region，将结果写入 `has_nan` DTensor（`S_SCALAR_INT32`, shape=[1,1,1,1]）

### 2.5 scaling factor 现状

- **分配**：`memory_plan.cpp` L376 — `baseline_.scaling` 在 `S_SCALAR_FP32` 区域，shape=[1,1,1,1]
- **初始化**：[`compiler.cpp`](file:///r:/renaissance/src/graph/compiler.cpp#L748-L749) L749 — `kInitConstant(1.0f)`，始终为 1.0
- **使用**：仅 SoftmaxCE FWD/BWD/INF 读取 `scaling` DTensor（用于 loss scale/inv_scale）
- **问题**：AMP 模式下应初始化为更大的值（如 65536.0f），且需在 NaN 时动态调整

### 2.6 optimizer 算子结构

所有 optimizer 均是 **RangeOp**，分发到：

- **CUDA kernel**（`optimizer_op.cu`）：接收 `w, g, m, v, lr, wd, beta, beta2, eps` 指针
- **CPU launcher**（`optimizer_op.cpp` L319-L398）：通过 `scalar_value<>` 宏读取标量

关键宏：

```cpp
// optimizer_op.cpp L44-L56
template<int Idx>
const float* scalar_ptr(const MemoryPlan& mp, const int32_t* ids, int rank) { ... }
template<int Idx>
float scalar_value(const MemoryPlan& mp, const int32_t* ids, int rank) { ... }
```

CUDA launcher 示例（SGD Weight）：

```cpp
input_ids: [lr(0), wd(1)]
lr = scalar_ptr<0>(...), wd = scalar_ptr<1>(...)
```

### 2.7 has_nan DTensor

- **分配**：`memory_plan.cpp` L362 — `baseline_.has_nan`，`S_SCALAR_INT32`，shape=[1,1,1,1]
- **写入**：`check_op.cpp` L55（CUDA）/ L116（CPU）— `check_nan` 写入 0 或 1
- **读取**：当前没有任何算子读取此 DTensor！
- **ID 获取**：[`compiler.cpp`](file:///r:/renaissance/src/graph/compiler.cpp#L753) L753 — `nan_flag_id = b.has_nan`

## 3. 设计方案

### 3.1 宏定义：`TR_AMP_INITIAL_SCALING`

```
建议值: 65536.0f  (2^16)
```

理由：

- 足够大以避免 FP16 下溢
- 是 2 的幂，方便反复除 2 保持精确
- 行业常见值（NVIDIA apex 默认 65536，PyTorch 默认也是此值）

定义位置：`include/renaissance/core/global_config.h` 末尾（与 `TR_DEFAULT_STREAM` 等宏同级）

```cpp
#define TR_AMP_INITIAL_SCALING  65536.0f
```

### 3.2 scaling factor 初始化逻辑

修改 [`compiler.cpp`](file:///r:/renaissance/src/graph/compiler.cpp#L748-L749) L748-749：

```cpp
// 当前：
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(1.0f));

// 修改为：
float init_scaling = amp ? TR_AMP_INITIAL_SCALING : 1.0f;
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(init_scaling));
```

### 3.3 新增 RangeOp：`RANGE_ADJUST_SCALING`

**设计理念**：将 scaling factor 的动态调整独立为一个 RangeOp，放在 `CAST_AND_CHECK` 之后、`FIRST_COMM` 和 `OPTIMIZER` 之前。这样 scaling 的减半操作在整个 batch 中只执行一次，避免 weight 和 bias 两个 optimizer 节点重复执行。

#### 3.3.1 语义

```
输入: has_nan (DTensor, INT32), scaling (DTensor, FP32)
输出: scaling (DTensor, FP32 —— 原地修改)
逻辑: if (*has_nan == 1) { *scaling = *scaling * 0.5f; }
```

#### 3.3.2 CUDA kernel（新增 `adjust_scaling_op.cu`）

```cuda
__global__ void adjust_scaling_kernel(int32_t* has_nan, float* scaling) {
    if (*has_nan != 0) {
        *scaling = *scaling * 0.5f;
    }
}
```

单线程启动即可，数据量极小（两个标量各 4 字节）。

#### 3.3.3 CPU launcher（新增 `adjust_scaling_op.cpp`）

```cpp
static void launch_range_adjust_scaling_cpu(CpuOpContext* op_ctx) {
    // 读取 has_nan 和 scaling DTensor
    // if (*has_nan == 1) *scaling *= 0.5f;
}
```

#### 3.3.4 在 Compiler 中注入

在 [`compiler.cpp`](file:///r:/renaissance/src/graph/compiler.cpp#L1653-L1696) 的 `CAST_AND_CHECK` 图末尾追加（仅 AMP）：

```cpp
if (amp_on) {
    // ... 现有 FP16→FP32 cast + RANGE_CHECK_NAN ...
    
    // 新增：scaling 动态调整
    GraphNode adj_node;
    adj_node.kind = GraphNode::Kind::RANGE;
    adj_node.range_op = RangeOp::RANGE_ADJUST_SCALING;
    adj_node.input_ids.push_back(nan_flag_id);      // has_nan
    adj_node.input_ids.push_back(scalar_ids.scaling); // scaling (input)
    adj_node.output_ids.push_back(scalar_ids.scaling); // scaling (output,原地)
    train_cg.append(GraphId::CAST_AND_CHECK, adj_node);
}
```

#### 3.3.5 所需修改清单

| 文件                                          | 修改内容                                             |
| --------------------------------------------- | ---------------------------------------------------- |
| `include/renaissance/backend/op_registry.h`   | 添加 `RANGE_ADJUST_SCALING` 枚举值                   |
| `src/backend/ops/range/adjust_scaling_op.cpp` | **新建** CPU launcher                                |
| `src/backend/ops/range/adjust_scaling_op.cu`  | **新建** CUDA kernel                                 |
| `src/graph/compiler.cpp`                      | 在 `CAST_AND_CHECK` 构建末尾注入节点                 |
| `src/graph/capture_cpu.cpp`                   | 无需修改（RangeOp 自动通过 `g_range_op_table` 分发） |
| `src/graph/capture_cuda.cpp`                  | 无需修改（同上）                                     |
| CMakeLists.txt                                | 添加新文件                                           |

### 3.4 优化器算子注入 `has_nan`

#### 3.4.1 Compiler 端：为优化器节点添加 `has_nan` input

在 [`compiler.cpp`](file:///r:/renaissance/src/graph/compiler.cpp#L1545-L1580) 的 OPTIMIZER 构建中：

**Weight 节点**（以 SGD 为例）：

```cpp
// 当前：
node.input_ids.push_back(scalar_ids.lr);     // idx 0
node.input_ids.push_back(scalar_ids.wd);     // idx 1

// 修改为：
node.input_ids.push_back(scalar_ids.lr);     // idx 0
node.input_ids.push_back(scalar_ids.wd);     // idx 1
node.input_ids.push_back(nan_flag_id);       // idx 2 ← 新增
```

**Bias 节点**：

```cpp
// 当前：
node.input_ids.push_back(scalar_ids.lr);     // idx 0

// 修改为：
node.input_ids.push_back(scalar_ids.lr);     // idx 0
node.input_ids.push_back(nan_flag_id);       // idx 1 ← 新增（bias 无 wd）
```

有 momentum/v 的优化器类同，`has_nan` 始终追加在最后。

#### 3.4.2 CUDA kernel 端：每个 kernel 添加 `has_nan` 参数

以 SGD Weight 为例（`optimizer_op.cu`）：

```cuda
// 当前签名：
__global__ void update_sgd_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    size_t n, const float* __restrict__ lr, const float* __restrict__ wd);

// 修改为：
__global__ void update_sgd_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    size_t n, const float* __restrict__ lr, const float* __restrict__ wd,
    const int32_t* __restrict__ has_nan);
```

Kernel 内部逻辑：

```cuda
for (size_t i = ...; i < n; i += stride) {
    float w_i = w[i];
    float g_i = g[i];
    // 正常计算更新值
    float w_new = w_i * (1.0f - _lr * _wd) - _lr * g_i;
    // 如果 has_nan，保留旧值；否则写入新值
    w[i] = (*has_nan == 0) ? w_new : w_i;
}
```

⚠️ **注意**：Momentum/Adam 的 m/v 更新也需要分支——如果 has_nan，动量也不更新：

```cuda
m[i] = (*has_nan == 0) ? (m[i] * _beta + g_i) : m[i];
```

#### 3.4.3 CUDA launcher 端：传递 `has_nan` 指针

在 `optimizer_op.cpp` 的 CUDA launcher 中，从 `input_ids` 最后一个位置读取 `has_nan`：

```cpp
static void launch_opt_weight_sgd_cuda(...) {
    // ... 现有代码 ...
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    //                                                                   ↑ 新索引
    optimizer_cuda::launch_sgd_weight_cuda(w, g, r_w_sz / sizeof(float), lr, wd, has_nan, s);
}
```

Momentum (has_nan 在 idx=2，bias 在 idx=2)：

```cpp
// Weight momentum: input_ids = [lr(0), wd(1), beta(2), has_nan(3)]
const int32_t* has_nan = scalar_ptr<3>(...);
// Bias momentum: input_ids = [lr(0), beta(1), has_nan(2)]
const int32_t* has_nan = scalar_ptr<2>(...);
```

#### 3.4.4 CPU launcher 端：同样修改

```cpp
static void sgd_update_cpu(float* w, const float* g, size_t n,
                           float lr, float wd, int has_nan) {
    if (has_nan) return;  // 直接跳过
    for (size_t i = 0; i < n; ++i) {
        float w_i = w[i];
        w[i] = w_i * (1.0f - lr * wd) - lr * g[i];
    }
}
```

#### 3.4.5 所需修改清单

| 文件               | 修改内容                                                     |
| ------------------ | ------------------------------------------------------------ |
| `optimizer_op.cu`  | 5 个 Weight kernel + 4 个 Bias kernel 全部添加 `has_nan` 参数和分支逻辑 |
| `optimizer_op.cpp` | 9 个 CUDA launcher 全部更新 `scalar_ptr` 索引和参数传递      |
| `optimizer_op.cpp` | 9 个 CPU launcher 全部读取 `has_nan` 并添加跳过逻辑（简单：`if (has_nan) return;`） |
| `compiler.cpp`     | 9 种优化器 × 2 (weight/bias) 的 `input_ids` 全部追加 `nan_flag_id` |

### 3.5 LARS 优化器

LARS 使用 ComputeOp 而非 RangeOp，路径不同（`compiler.cpp` L1477-L1520）。需要：

- `LARS_COMPUTE_TRUST_RATIO` / `LARS_UPDATE` / `LARS_NESTEROV_UPDATE` kernel 中添加 `has_nan` 检查
- 在 `input_ids` 中注入 `nan_flag_id`

但当前 `test_dl_full` 使用 SGD，LARS 可以后续再处理。

### 3.6 图执行流程调整

**修改后流程（仅 AMP）**：

```
Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD
Phase 2: DEEP_FWD_BWD ‖ XFER(next)
Phase 3: FIRST_LAYER_BWD ‖ DEEP_ALLREDUCE
Phase 4: CAST_AND_CHECK:
         ├── RANGE_CAST_FP16_TO_FP32  × N   (FP16梯度→FP32)
         ├── RANGE_CHECK_NAN                  (写 has_nan DTensor)
         └── RANGE_ADJUST_SCALING             (if has_nan: scaling*0.5)  ← 新增
Phase 5: LR 写入 → FIRST_ALLREDUCE
Phase 6: OPTIMIZER:
         ├── RANGE_UPDATE_WEIGHT_*           (if has_nan: skip w+m+v)
         └── RANGE_UPDATE_BIAS_*             (if has_nan: skip w+m+v)
```

关键：`RANGE_ADJUST_SCALING` 和 `RANGE_UPDATE_*` 都在同一个 `CAST_AND_CHECK` / `OPTIMIZER` 的 captured graph 内，因此无需退出 capture 循环——GPU 端完全是图内执行。

## 4. 影响评估

### 4.1 性能影响

- `RANGE_ADJUST_SCALING`：一个单线程 kernel，<10 微秒，可忽略
- `has_nan` 分支检查：CUDA warp 内统一分支，warp divergence 极小（整个 batch 要么全 NaN 要么全正常）；额外读取一个 INT32 标量，无性能差异
- 总计：**性能影响 <0.1%**

### 4.2 精度影响

- 非 AMP 模式：scaling=1.0f，has_nan 始终为 0，行为与当前完全一致
- AMP 模式：NaN batch 被正确跳过，精度应与 FP32 版一致（已验证 CPU/GPU FP32 输出完全一致）

### 4.3 CPU 路径

CPU 的 `capture_cpu.cpp` 无需修改——它通过 `g_range_op_table` 自动分发新增的 `RANGE_ADJUST_SCALING`。仅需实现 CPU launcher。

## 5. 实现步骤

| 步骤 | 内容                                                      | 涉及文件           | 优先级 |
| ---- | --------------------------------------------------------- | ------------------ | ------ |
| 1    | 定义 `TR_AMP_INITIAL_SCALING` 宏                          | `global_config.h`  | P0     |
| 2    | 添加 `RANGE_ADJUST_SCALING` 枚举                          | `op_registry.h`    | P0     |
| 3    | 实现 `adjust_scaling_op.cu` + `.cpp`                      | 新建 2 个文件      | P0     |
| 4    | 修改 `compiler.cpp`：注入 scaling 初始化和调整节点        | `compiler.cpp`     | P0     |
| 5    | 修改 optimizer kernel（.cu）：添加 `has_nan` 参数 + 分支  | `optimizer_op.cu`  | P1     |
| 6    | 修改 optimizer launcher（.cpp）：更新索引、传递 `has_nan` | `optimizer_op.cpp` | P1     |
| 7    | 修改 `compiler.cpp`：optimizer 节点注入 `nan_flag_id`     | `compiler.cpp`     | P1     |
| 8    | 编译                                                      | CMakeLists + ninja | P1     |
| 9    | 运行 `test_dl_full --gpu` 和 `--amp` 验证                 | -                  | P2     |

## 6. 开放问题

1. **`TR_AMP_INITIAL_SCALING` 值**：建议 65536.0f，待确认
2. **scaling 恢复策略**：当前方案仅在 NaN 时减半，不恢复。是否需要 backlog 增长（如 `scaling *= 2` 每 N 步无 NaN）？这可以作为后续增强
3. **LARS 路径**：当前方案仅覆盖非 LARS 优化器，LARS 需单独的 ComputeOp 修改

---

> 撰写日期：2026-05-26



# 【用户补充】

有些小伙伴给出了一些简化的思路，比如“CPU 直接改 launcher；CUDA 需修改底层 .cu kernel 签名”——这是不推荐的。因为你擅自破坏了CPU和GPU的对称性。到时候要进行其他修改的时候，就容易错漏。GPU要改签名，那CPU最好也改。只要CPU有对应的部分，那就对应修改。
再次强调，需要图内实现分支，绝对不能把标志位取回HOST然后再在HOST上做判断。
关于scaling factor是否只缩小不增大，我的观点是：不要把CUDA Graph搞得太复杂，我们图内就只做缩小。
另外，图内分支结构要怎么实现，小伙伴们要认真研究
