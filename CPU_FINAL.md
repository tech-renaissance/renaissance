# CPU Workspace 设计与 FC BWD 性能优化 —— 最终实施方案

## 版本

**V4.0 Final**  
日期: 2026-05-19  
编制: 技术觉醒团队  
基于: 小伙伴K、小伙伴S、小伙伴D 全部提案的交叉验证与决策收敛

---

## 一、决策收敛：各提案异同与最终选择

### 1.1 共识项（全员一致，直接采纳）

| 议题 | 共识 | 决策 |
|------|------|------|
| 根因 L1 | FC BWD 右操作数 RowMajor → Eigen GEMM 列跨步 → 缓存命中率低 | ✅ 通过 ColMajor 重排修复 |
| Workspace 位置 | `DeviceContext` 中新增 `mutable WSpace cpu_workspace_` | ✅ 采纳 |
| 接口模式 | 复用 `ensure_workspace_grow` 的 `const` + `mutable` 模式 | ✅ 采纳 |
| 大小确定 | `compile_capture_simple()` 后 dry-run 一次 | ✅ 采纳 |
| 改动规模 | ~100 行，4 个文件，零接口破坏 | ✅ 一致 |

### 1.2 分歧项（需要决策收敛）

| 议题 | D 方案 | S 方案 | K 方案 | **最终决策** | 理由 |
|------|--------|--------|--------|-------------|------|
| 内存分配 | `_aligned_malloc`/`posix_memalign` | `mi_malloc_aligned` | `mi_malloc_aligned` | **`mi_malloc_aligned`** | 项目已链接 mimalloc；跨平台统一；[memory_arena.h:28](file:///r:/renaissance/include/renaissance/backend/memory_arena.h#L28) 已 include `<mimalloc.h>` |
| 对齐值 | 32B | 64B | 64B | **64B** | cache line 大小 + AVX-512 兼容；对齐开销 32 字节 vs MB 级 workspace 可忽略 |
| OpenMP | 未涉及 | 未涉及 | **`Eigen::setNbThreads(1)`** | **采纳** | 这是 K 方案最大的差异化贡献：Composite BWD 比 standalone 之和多 5.7×，OpenMP 并行对 batch=7 小矩阵是负收益 |

### 1.3 根因分层（科学方法论）

| 层级 | 问题 | 影响量级 | 确定性 | 修复方案 |
|------|------|---------|--------|---------|
| **L1：GEMM 跨步** | BWD dW/dX 的右操作数为 RowMajor，列访问 stride 巨大 | standalone BWD 慢 2~5× | ✅ 确认 | ColMajor 重排（workspace） |
| **L2：OpenMP 并行** | Eigen 对小矩阵（batch=7）仍可能触发多线程，同步开销 > 计算收益 | Composite 额外慢 3~5× | ⚠️ 高概率 | `Eigen::setNbThreads(1)` |
| **L3：未知因素** | Arena cache 布局、频率缩放等 | 剩余开销 | ❓ 待验证 | 后续 profiling |

---

## 二、实施概览

```
改动文件：4 个
改动位置：8 处
总改动量：约 135 行代码
安全等级：零接口破坏，零 const_cast 新增
实施顺序：Step A → B → C → D → E（线性，前后依赖）
```

---

## 三、详细实施步骤

---

### Step A：DeviceContext 头文件扩展

**文件**：`include/renaissance/backend/device_context.h`

**位置**：在 `ensure_workspace_grow` 声明（第 83 行）之后、`nccl_comm()` 声明（第 85 行）之前。

**当前代码**（第 83-85 行）：
```cpp
    void ensure_workspace_grow(StreamKind kind, size_t req_size) const;  // 精确按需扩容

    [[nodiscard]] void* nccl_comm() const noexcept { return nccl_comm_; }
```

**修改后**：
```cpp
    void ensure_workspace_grow(StreamKind kind, size_t req_size) const;  // 精确按需扩容

    // ───────────────────────────────────────────────────────────────────
    // CPU Workspace（单流全局共享，避免算子内反复 malloc/free）
    // 语义完全对齐 GPU 的 per-stream workspace，由 compile 阶段 dry-run 确定大小
    // ───────────────────────────────────────────────────────────────────
    [[nodiscard]] void* cpu_workspace() const noexcept { return cpu_workspace_.ptr; }
    [[nodiscard]] size_t cpu_workspace_size() const noexcept { return cpu_workspace_.size; }
    void ensure_cpu_workspace_grow(size_t req_size) const;

    [[nodiscard]] void* nccl_comm() const noexcept { return nccl_comm_; }
```

**位置**：在 `mutable WSpace workspaces_[5];` 声明（第 101 行）之后、`void* nccl_comm_` 声明（第 103 行）之前。

**当前代码**（第 101-103 行）：
```cpp
    mutable WSpace workspaces_[5];  // mutable for const-safe workspace growth

    void* nccl_comm_ = nullptr;
```

**修改后**：
```cpp
    mutable WSpace workspaces_[5];       // mutable for const-safe workspace growth
    mutable WSpace cpu_workspace_;       // CPU 单流 workspace（GPU 模式不使用但安全保留）

    void* nccl_comm_ = nullptr;
```

**验证点**：头文件编译通过，无新 warning。

---

### Step B：DeviceContext 构造函数 —— 禁用 Eigen OpenMP

**文件**：`src/backend/device_context.cpp`

**位置**：构造函数中，第 40 行 `if (device_id_ < 0) return;` 之前插入。

**当前代码**（第 39-40 行）：
```cpp
DeviceContext::DeviceContext(int device_id) : device_id_(device_id) {
    if (device_id_ < 0) return;
```

**前置条件**：`device_context.cpp` 需要新增 Eigen 头文件。在 `#ifdef TR_USE_CUDA` 块下方、`namespace tr` 之前添加：

```cpp
#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#endif

#ifdef TR_USE_EIGEN
#include <Eigen/Core>
#endif
```

**修改后**：
```cpp
DeviceContext::DeviceContext(int device_id) : device_id_(device_id) {
    if (device_id_ < 0) {
        // CPU 模式：Eigen 对小矩阵的 OpenMP 并行是负收益
        // 线程分发/同步开销远超计算收益，全局禁用
#ifdef TR_USE_EIGEN
#ifdef _OPENMP
        Eigen::setNbThreads(1);
        TR_LOG_INFO("backend") << "DeviceContext CPU: Eigen OpenMP disabled (setNbThreads=1)";
#endif
#endif
        return;
    }
```

**说明**：此修改解决 L2 层问题。放在构造函数的原因：
- CPU 模式只有一个 DeviceContext，初始化一次即可
- 不影响 GPU 路径（GPU 路径不执行此分支）
- 比 CMakeLists.txt 加宏更灵活（运行时决定，不需重新编译）
- 比算子内每次设置更简洁（零运行时重复调用开销）
- `#ifdef TR_USE_EIGEN` 防御性宏保证无 Eigen 环境编译不失败

**验证点**：CPU 模式下日志输出 `"DeviceContext CPU: Eigen OpenMP disabled (setNbThreads=1)"`。

---

### Step C：实现 `ensure_cpu_workspace_grow` + 析构补充

**文件**：`src/backend/device_context.cpp`

#### C1. 实现 `ensure_cpu_workspace_grow`

**位置**：在 `ensure_workspace_grow` 函数之后（第 397 行）、`destroy_streams` 函数之前（第 402 行）。

**插入新函数**：
```cpp
// ---------------------------------------------------------------------------
// ensure_cpu_workspace_grow: CPU workspace 精确按需扩容
// ---------------------------------------------------------------------------
void DeviceContext::ensure_cpu_workspace_grow(size_t req_size) const {
    // 1. 零需求快速路径
    if (req_size == 0) return;

    // 2. 64 字节对齐（cache line + AVX-512 兼容）
    constexpr size_t kAlign = 64;
    size_t aligned = (req_size + kAlign - 1) & ~(kAlign - 1);

    // 3. mutable 引用 —— 允许 const 方法中修改 workspaces_（与 GPU 模式一致）
    auto& ws = cpu_workspace_;

    // 4. 已足够大，直接返回
    if (ws.size >= aligned) return;

    // 5. 释放旧内存（存在时）
    if (ws.ptr) {
        mi_free(ws.ptr);
        ws.ptr  = nullptr;
        ws.size = 0;
    }

    // 6. 申请新对齐内存（mimalloc，项目统一分配器）
    ws.ptr = mi_malloc_aligned(aligned, kAlign);
    if (!ws.ptr) {
        ws.size = 0;
        TR_GPU_OOM("Failed to allocate CPU workspace of " << aligned << " bytes");
    }
    ws.size = aligned;

    TR_LOG_INFO("backend") << "DeviceContext CPU: workspace grown to "
                           << aligned << " bytes";
}
```

**说明**：
- `mi_malloc_aligned` / `mi_free` 来自 `<mimalloc.h>`，已在 [device_context.cpp:10](file:///r:/renaissance/src/backend/device_context.cpp#L10) 通过 `memory_arena.h` → [memory_arena.h:28](file:///r:/renaissance/include/renaissance/backend/memory_arena.h#L28) 链式引入，**无需新增 include**
- `const` 方法 + `mutable WSpace` 与 GPU 的 `ensure_workspace_grow` 完全一致
- 对齐值 64B：cache line 大小（64B）避免 false sharing；AVX-512 兼容（512-bit = 64B）

#### C2. 析构函数添加 CPU workspace 释放

**位置**：析构函数末尾 `#endif` 之后（第 155-156 行）。

**当前代码**（第 138-156 行）：
```cpp
DeviceContext::~DeviceContext() {
#ifdef TR_USE_CUDA
    // 1. 先销毁 handle（遵循 RAII：依赖资源先于被依赖资源销毁）
    // ... GPU cleanup ...
    // 2. 再销毁 stream
    destroy_streams();
    // 3. 最后释放 workspace
    free_workspaces();
#endif
}
```

**修改后**：
```cpp
DeviceContext::~DeviceContext() {
#ifdef TR_USE_CUDA
    // ... 现有 GPU cleanup（不变）...
    destroy_streams();
    free_workspaces();
#endif

    // CPU workspace 清理（GPU/CPU 模式统一处理）
    // 放在 #ifdef TR_USE_CUDA 外部，CPU-only 模式才能执行
    if (cpu_workspace_.ptr) {
        mi_free(cpu_workspace_.ptr);
        cpu_workspace_.ptr  = nullptr;
        cpu_workspace_.size = 0;
    }
}
```

**设计要点**：`free_workspaces()` 保持 `#ifdef TR_USE_CUDA` 不变。CPU workspace 有自己的独立释放路径，不与 GPU 的 per-stream workspace 混合。

**验证点**：编译通过；程序结束时 `cpu_workspace_.ptr` 变为 `nullptr`。

---

### Step D：FC BWD 双重优化（ColMajor 重排 + 清理 Profiling 代码）

**文件**：`src/backend/ops/dtensor/fc_op.cpp`

**位置**：`launch_fc_bwd_cpu_eigen` 函数（第 647-713 行）。

**改动说明**：
1. **必删**：`#include <chrono>`（第 21 行附近，前一轮 profiling 代码遗留）
2. **必加**：`#include <algorithm>`（第 21 行附近，新增代码使用 `std::max`，MSVC 虽可能间接引入但不是标准保证）
3. **必删**：函数内 profiling 代码（`call_cnt`、`t_start`/`t1`/`t2`/`t3`、`std::chrono::high_resolution_clock`、`std::cout << "[FC_BWD_TIMING]"`）
4. **新增**：`using MatrixXfCol` 类型别名
5. **新增**：workspace 计算与获取
6. **修改**：dW 和 dX 的 GEMM 路径 → 右操作数 ColMajor 重排

**当前代码**（第 647-713 行）：
```cpp
static void launch_fc_bwd_cpu_eigen(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<FCParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_BWD CPU EIGEN missing FCParams");

    bool has_bias = p->bias;
    int x_idx = op_ctx->num_inputs - 1;

    TR_CHECK(op_ctx->num_inputs >= 4, ShapeError,
             "FC_BWD CPU EIGEN requires at least 4 inputs. Got " << op_ctx->num_inputs);
    TR_CHECK(op_ctx->num_outputs >= 3, ShapeError,
             "FC_BWD CPU EIGEN requires at least 3 outputs. Got " << op_ctx->num_outputs);

    float* dy = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    float* w  = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* x  = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[x_idx]));
    float* dx = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));
    float* dw = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[1]));
    float* db = has_bias ? static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[2])) : nullptr;

    int batch        = op_ctx->input_shape.n;
    int out_features = op_ctx->input_shape.c;
    int in_features  = op_ctx->output_shape.h * op_ctx->output_shape.w * op_ctx->output_shape.c;

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    static int call_cnt = 0;
    ++call_cnt;
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. db = reduce_sum(dY, dim=0)
    if (has_bias && db != nullptr) {
        Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
        Eigen::Map<Eigen::RowVectorXf> db_vec(db, out_features);
        db_vec.noalias() = dY_mat.colwise().sum();
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // 2. dW = dY^T @ X
    if (dw != nullptr) {
        Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
        Eigen::Map<MatrixXfRow> X_mat(x, batch, in_features);
        Eigen::Map<MatrixXfRow> dW_mat(dw, out_features, in_features);
        dW_mat.noalias() = dY_mat.transpose() * X_mat;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // 3. dX = dY @ W
    Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
    Eigen::Map<MatrixXfRow> W_mat(w, out_features, in_features);
    Eigen::Map<MatrixXfRow> dX_mat(dx, batch, in_features);
    dX_mat.noalias() = dY_mat * W_mat;

    auto t3 = std::chrono::high_resolution_clock::now();

    if (call_cnt >= 6 && call_cnt <= 25) {
        auto db_us  = std::chrono::duration<double, std::micro>(t1 - t_start).count();
        auto dw_us  = std::chrono::duration<double, std::micro>(t2 - t1).count();
        auto dx_us  = std::chrono::duration<double, std::micro>(t3 - t2).count();
        auto tot_us = std::chrono::duration<double, std::micro>(t3 - t_start).count();
        std::cout << "[FC_BWD_TIMING] call=" << call_cnt
                  << " in=" << in_features << " out=" << out_features
                  << " db=" << db_us << "us dw=" << dw_us << "us dx=" << dx_us
                  << "us total=" << tot_us << "us" << std::endl;
    }
}
```

**修改后（完整函数）**：
```cpp
static void launch_fc_bwd_cpu_eigen(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<FCParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_BWD CPU EIGEN missing FCParams");

    bool has_bias = p->bias;
    int x_idx = op_ctx->num_inputs - 1;

    TR_CHECK(op_ctx->num_inputs >= 4, ShapeError,
             "FC_BWD CPU EIGEN requires at least 4 inputs. Got " << op_ctx->num_inputs);
    TR_CHECK(op_ctx->num_outputs >= 3, ShapeError,
             "FC_BWD CPU EIGEN requires at least 3 outputs. Got " << op_ctx->num_outputs);

    float* dy = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    float* w  = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* x  = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[x_idx]));
    float* dx = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));
    float* dw = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[1]));
    float* db = has_bias ? static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[2])) : nullptr;

    int batch        = op_ctx->input_shape.n;
    int out_features = op_ctx->input_shape.c;
    int in_features  = op_ctx->output_shape.h * op_ctx->output_shape.w * op_ctx->output_shape.c;

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXfCol = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    // ── Workspace 计算与获取 ──
    // dW 和 dX 顺序执行，共享同一块 workspace：
    //   dW: dY^T[out×batch] × X_cm[batch×in]  → ws = sizeof(X_cm) = batch×in×4
    //   dX: dY[batch×out]   × W_cm[out×in]    → ws = sizeof(W_cm) = out×in×4
    // 取最大值：max(batch×in, out×in) × sizeof(float)
    size_t w_cm_bytes = static_cast<size_t>(out_features) * in_features * sizeof(float);
    size_t x_cm_bytes = static_cast<size_t>(batch)       * in_features * sizeof(float);
    size_t ws_needed  = std::max(w_cm_bytes, x_cm_bytes);
    constexpr size_t kAlign = 64;
    ws_needed = (ws_needed + kAlign - 1) & ~(kAlign - 1);

    const DeviceContext* ctx = op_ctx->ctx;
    ctx->ensure_cpu_workspace_grow(ws_needed);  // const-safe（mutable WSpace）
    float* ws_ptr = static_cast<float*>(ctx->cpu_workspace());

    // 1. db = reduce_sum(dY, dim=0)  —— 不涉及 GEMM，不变
    if (has_bias && db != nullptr) {
        Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
        Eigen::Map<Eigen::RowVectorXf> db_vec(db, out_features);
        db_vec.noalias() = dY_mat.colwise().sum();
    }

    // 2. dW = dY^T @ X
    //    原方案：右操作数 X 为 RowMajor → 列访问 stride 巨大 → 缓存命中率低
    //    优化：将 X 复制到 ColMajor 缓冲区 → 恢复 A-ColMajor × B-ColMajor 最优路径
    if (dw != nullptr) {
        Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
        Eigen::Map<MatrixXfRow> X_mat(x, batch, in_features);
        Eigen::Map<MatrixXfRow> dW_mat(dw, out_features, in_features);

        Eigen::Map<MatrixXfCol> X_cm(ws_ptr, batch, in_features);
        X_cm = X_mat;  // RowMajor → ColMajor 重排（memcpy）
        dW_mat.noalias() = dY_mat.transpose() * X_cm;
    }

    // 3. dX = dY @ W
    //    原方案：右操作数 W 为 RowMajor → 列访问 stride 巨大
    //    优化：将 W 复制到 ColMajor 缓冲区（复用 workspace）→ 恢复最优路径
    Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
    Eigen::Map<MatrixXfRow> dX_mat(dx, batch, in_features);

    Eigen::Map<MatrixXfCol> W_cm(ws_ptr, out_features, in_features);
    {
        Eigen::Map<MatrixXfRow> W_mat(w, out_features, in_features);
        W_cm = W_mat;  // RowMajor → ColMajor 重排（memcpy）
    }
    dX_mat.noalias() = dY_mat * W_cm;
}
```

**关键细节**：
- dW 中 `X_cm = X_mat` 复制和 `dW_mat = dY^T * X_cm` GEMM 不同时使用 `ws_ptr` 上的相同 `Map`，无生命周期冲突
- dW 的 `X_cm` 用完后，dX 的 `W_cm` 重新映射到同一块 `ws_ptr`，作用域隔离保证正确性
- **必删**：`#include <chrono>` 行（第 21 行附近），仅服务于已清理的 profiling 代码，删除后无其他引用
- **必加**：`#include <algorithm>`（放在 `<chrono>` 被删除的位置附近），提供 `std::max` 的标准保证

**验证点**：
- 编译通过，无 `call_cnt`、`chrono`、`high_resolution_clock` 等残余符号
- `MatrixXfCol` 类型已声明
- `#include <algorithm>` 已添加，`#include <chrono>` 已删除

---

### Step E：compile_capture_simple 添加 CPU Dry-Run

**文件**：`src/task/task_base.cpp`

**位置**：在 `compile_capture_simple()` 函数的 capture 循环结束后（第 369 行 `}` 闭合后）、函数返回前（第 371 行 `}` 之前）。

**当前代码**（第 365-371 行）：
```cpp
        TR_LOG_INFO("task") << "[DBG] emplace simple_captured_graphs_ ...";
        simple_captured_graphs_.emplace(name, std::move(cg));
        TR_LOG_INFO("task") << "[DBG] emplace done";
        graph_index++;
    }
    TR_LOG_INFO("task") << "[DBG] compile_capture_simple: all done";
}
```

**修改后**：
```cpp
        TR_LOG_INFO("task") << "[DBG] emplace simple_captured_graphs_ ...";
        simple_captured_graphs_.emplace(name, std::move(cg));
        TR_LOG_INFO("task") << "[DBG] emplace done";
        graph_index++;
    }

    // ── CPU Dry-Run：对所有 CPU 图执行一次 launch，触发 workspace 扩容 ──
    // 安全性前提：
    //   1. compile_alloc_hardware() 已在之前调用 → ArenaKeeper 已初始化 → ptr_at() 可用
    //   2. ctx.set_rank() / set_memory_plan() 已在 capture 循环中设置
    //   3. compile 阶段 Arena 数据未初始化，dry-run 写入无害
    //   4. 后续 run() 前 transfer_to_rank() 会重新填充输入数据
    // 开销：仅执行 1 次，~500 µs 级别，可忽略
    for (auto& [name, cg] : simple_captured_graphs_) {
        if (cg.is_cuda()) continue;
        cg.launch(0, nullptr);  // CPU launch 不依赖 stream 参数
    }

    TR_LOG_INFO("task") << "[DBG] compile_capture_simple: all done";
}
```

**可行性交叉验证**：
- [task_base.cpp:191-195](file:///r:/renaissance/src/task/task_base.cpp#L191-L195)：`compile_alloc_hardware()` 在 `compile_capture_simple()` 之前调用
- [task_base.cpp:337-338](file:///r:/renaissance/src/task/task_base.cpp#L337-L338)：capture 循环中 `ctx.set_rank(rank)` 和 `ctx.set_memory_plan()` 已设置
- [captured_graph.cpp:161-162](file:///r:/renaissance/src/graph/captured_graph.cpp#L161-L162)：CPU `launch()` 忽略 stream 参数，`nullptr` 安全

**验证点**：
- 编译通过
- `ensure_cpu_workspace_grow` 日志在 compile 阶段出现（而非 run 阶段）

---

## 四、验证计划

### 4.1 编译验证

```bash
cd r:\renaissance\build
ninja
# 预期：0 errors, 0 warnings
```

### 4.2 正确性验证

```bash
# Step 1：Standalone FC（覆盖两种尺寸）
.\tests\correction\test_fc_fwd_bwd.exe --cpu --batch 7 --in 784 --out 512
.\tests\correction\test_fc_fwd_bwd.exe --cpu --batch 7 --in 512 --out 256
# 预期：MSE < 1e-5，与优化前一致（ColMajor 重排是精确 bit-wise 复制，无舍入误差）

# Step 2：Composite（全链路正确性）
.\tests\correction\test_flatten_fc_relu_fc.exe --cpu
# 预期：全部 12 个 MSE 检查 PASS
```

### 4.3 性能验证

| 测试 | 优化前 BWD | 优化后预期 BWD | 加速比 |
|------|-----------|---------------|--------|
| FC1 standalone (784→512) | ~568 µs | **~155 µs** | ~3.7× |
| FC2 standalone (512→256) | ~81 µs | **~30 µs** | ~2.7× |
| Composite BWD | ~3682 µs | **~210 µs** | **~17.5×** |

**说明**：Composite 的 17.5× 加速来自 L1（ColMajor 重排 ~3×）+ L2（禁用 OpenMP ~5×）。若 L2 的实际贡献低于预估，Composite BWD 可能为 ~570 µs + 未知开销，仍低于 1000 µs。

### 4.4 Workspace 行为验证

通过 `TR_LOG_INFO` 日志确认：

1. **compile 阶段**（dry-run）：
   ```
   DeviceContext CPU: workspace grown to 1644800 bytes   ← FC1 触发（1.57 MB）
   ```
   FC2 不应再有 grow 日志（复用）

2. **run 阶段**：
   不应再有 `"workspace grown"` 日志（大小已固定）

3. **程序退出**：
   无 `mi_free` 相关的崩溃或泄漏

### 4.5 OpenMP 验证

在 `launch_fc_bwd_cpu_eigen` 内临时加入（验证后删除）：
```cpp
std::cout << "Eigen threads: " << Eigen::nbThreads() << std::endl;
```
**预期输出**：始终为 `1`。

---

## 五、风险评估与回退

| 风险 | 概率 | 影响 | 监测方法 | 回退措施 |
|------|------|------|---------|---------|
| `mi_malloc_aligned` 版本不兼容 | 极低 | 低 | 编译错误 | 替换为 `_aligned_malloc`/`posix_memalign`，2 行改动 |
| `Eigen::setNbThreads(1)` 影响其他 CPU 算子性能 | 极低 | 低 | 性能回归测试 | 所有 CPU 算子均为单线程串行；未来需并行时可改为动态策略 |
| Dry-run 触发未知 side-effect | 极低 | 中 | correctness 测试 | 当前 CPU 算子（FC/ReLU/Flatten/GAP）均无 I/O side-effect；若后期新增有 side-effect 算子，可在 `CpuOpContext` 加 `dry_run` 标志 |
| ColMajor 重排引入数值误差 | 无 | 无 | MSE 检查 | 重排是 bit-wise 复制，无浮点运算；MSE 验证可捕获任何异常 |
| Composite ~5434 µs 未知开销仍存在 | 中 | 中 | 性能测试 | L1+L2 修复后，通过 VTune/perf/缩放实验隔离剩余因素 |

---

## 六、改动汇总表

| Step | 文件 | 行号范围 | 改动性质 | 行数 |
|------|------|---------|---------|------|
| A1 | [device_context.h](file:///r:/renaissance/include/renaissance/backend/device_context.h#L83-L85) | ~83 | 在 `ensure_workspace_grow` 后插入 3 个 CPU workspace 方法声明 | +5 |
| A2 | [device_context.h](file:///r:/renaissance/include/renaissance/backend/device_context.h#L101-L103) | ~101 | 在 `workspaces_[5]` 后插入 `cpu_workspace_` 成员 | +2 |
| B1 | [device_context.cpp](file:///r:/renaissance/src/backend/device_context.cpp#L18-L20) | ~18 | `#ifdef TR_USE_CUDA` 块下方添加 `#ifdef TR_USE_EIGEN` → `#include <Eigen/Core>` | +3 |
| B2 | [device_context.cpp](file:///r:/renaissance/src/backend/device_context.cpp#L39-L40) | 40 | CPU 分支插入 `Eigen::setNbThreads(1)`（带 `TR_USE_EIGEN`/`_OPENMP` 双宏保护） | +7 |
| C1 | [device_context.cpp](file:///r:/renaissance/src/backend/device_context.cpp#L397-L402) | ~398 | 插入 `ensure_cpu_workspace_grow` 完整实现 | +40 |
| C2 | [device_context.cpp](file:///r:/renaissance/src/backend/device_context.cpp#L155-L156) | ~155 | 析构函数 `#endif` 后插入 CPU workspace 释放 | +7 |
| D | [fc_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L647-L713) | 647-713 | 整体替换 `launch_fc_bwd_cpu_eigen`；顶部删 `<chrono>`、加 `<algorithm>` | ~60 |
| E | [task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp#L369-L371) | ~369 | 在 capture 循环闭合后插入 CPU dry-run 循环 | +10 |

**总改动量**：~135 行，4 个文件，零接口破坏，零 `const_cast` 新增。

---

## 七、实施检查清单

实施时逐项勾选，确保无遗漏：

### Phase 1：编译前检查
- [ ] 确认 `r:\renaissance\build` 目录存在，CMake 已配置
- [ ] 确认 `ninja` 在 `PATH` 中
- [ ] 确认 mimalloc 版本 ≥ v1.7（检查 `third_party/mimalloc` 或 `CMakeLists.txt`）
- [ ] 确认 `device_context.cpp` 已通过 `#include "memory_arena.h"` 间接包含 `<mimalloc.h>`

### Phase 2：代码修改
- [ ] Step A1：`device_context.h` 添加 3 个 CPU workspace 方法声明（第 83 行后）
- [ ] Step A2：`device_context.h` 添加 `cpu_workspace_` 成员变量（第 101 行后）
- [ ] Step B1：`device_context.cpp` 在 `#ifdef TR_USE_CUDA` 块下方添加 `#include <Eigen/Core>`（`#ifdef TR_USE_EIGEN` 保护）
- [ ] Step B2：`device_context.cpp` 构造函数 CPU 分支添加 `Eigen::setNbThreads(1)`（`TR_USE_EIGEN` + `_OPENMP` 双宏保护）
- [ ] Step C1：`device_context.cpp` 添加 `ensure_cpu_workspace_grow` 实现
- [ ] Step C2：`device_context.cpp` 析构函数 `#endif` 后添加 `mi_free(cpu_workspace_)`
- [ ] Step D：`fc_op.cpp` 删 `<chrono>`、加 `<algorithm>`，替换 `launch_fc_bwd_cpu_eigen`
- [ ] Step E：`task_base.cpp` capture 循环后添加 CPU dry-run

### Phase 3：编译
- [ ] `ninja` 编译通过，0 errors，0 warnings

### Phase 4：测试
- [ ] `test_fc_fwd_bwd.exe --cpu` standalone FC 正确性 PASS
- [ ] `test_flatten_fc_relu_fc.exe --cpu` composite 正确性 PASS
- [ ] 确认 `Eigen::setNbThreads(1)` 日志出现在 compile 阶段
- [ ] 确认 `"workspace grown to"` 日志仅出现在 FC1 dry-run
- [ ] 确认 run 阶段无新增 `"workspace grown"` 日志
- [ ] 确认 BWD 性能提升符合预期

### Phase 5：清理
- [ ] 确认 `fc_op.cpp` 中 `#include <chrono>` 已删除、`#include <algorithm>` 已添加
- [ ] 确认函数内无 `call_cnt`、`high_resolution_clock`、`std::cout << "[FC_BWD_TIMING]"` 等 profiling 残留
- [ ] 确认 `const_cast` 数量和位置与修改前一致（仅 `ptr_at` 调用处 6 个）

---

## 八、总结

本方案融合了小伙伴K、S、D 的全部有效分析，形成了如下最终决策：

| 优化 | 来源 | 解决的问题 | 预期收益 |
|------|------|-----------|---------|
| ColMajor 重排（workspace） | K/S/D 共识 | L1：GEMM 跨步 | standalone BWD 3.7× |
| `Eigen::setNbThreads(1)` | K 独家 | L2：OpenMP 负收益 | composite BWD 额外 5× |
| CPU workspace 架构 | K/S/D 共识 | 避免算子内 malloc/free | 零分配开销 |
| Dry-run 大小确定 | K/S/D 共识 | 与 GPU warmup 对齐 | 确定性行为 |
| mimalloc 对齐分配 | S/K 共识 | 跨平台统一 | 可移植性 |

**关键里程碑**：
- 改动量：~135 行，4 个文件，8 处修改（B 拆为 B1/B2）
- 安全性：零接口破坏，零 `const_cast` 新增，`TR_USE_EIGEN` 防御性宏保护
- 性能目标：Composite BWD 从 ~3682 µs 降至 ~210 µs（~17.5×）
- L3 未知开销：列为后续 profiling 任务，不影响本方案实施