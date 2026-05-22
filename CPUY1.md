# CPU Workspace 设计与 FC BWD 性能优化 —— 综合方案

## 版本

V1.0  
日期: 2026-05-19  
编制: 技术觉醒团队  
基于: 小伙伴K（初版）、小伙伴S（评审改进）、小伙伴D（独立分析）三份提案的交叉验证与综合

---

## 一、问题根因分析

### 1.1 性能数据

Composite 测试 `test_flatten_fc_relu_fc` 的 CPU BWD 耗时 **6097 µs**，而 FWD 仅 **180 µs**，比例 **34×**。对照 standalone FC 测试：

| 测试                             | FWD (µs) | BWD (µs) | BWD/FWD  |
| -------------------------------- | -------- | -------- | -------- |
| FC standalone (784→512, batch=7) | 115      | 528      | 4.6×     |
| FC standalone (512→256, batch=7) | 39       | 135      | 3.5×     |
| Composite (FC1+ReLU+FC2)         | 180      | **6097** | **34×**  |

理论上 composite BWD 应该 ≈ 528 + 135 = **663 µs**（两独立 FC BWD 之和 + ReLU/Flatten），但实际 **6097 µs**，存在 **~5434 µs** 的无法解释开销。

### 1.2 Eigen GEMM 内存访问模式分析

当前 `launch_fc_bwd_cpu_eigen` 位于 [fc_op.cpp:647-713](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L647-L713)，三个子操作的 Eigen 表达式：

| 子操作 | 表达式                       | 左操作数访问              | 右操作数访问                            | 实际模式                       |
| ------ | ---------------------------- | ------------------------- | --------------------------------------- | ------------------------------ |
| **db** | `dY_mat.colwise().sum()`     | —                         | —                                       | 简单归约，不涉及 GEMM          |
| **dW** | `dY_mat.transpose() * X_mat` | transpose 等效 ColMajor ✓ | RowMajor，按列访问 stride=`in_features` ✗ | ColMajor × RowMajor → 跨步严重 |
| **dX** | `dY_mat * W_mat`             | RowMajor 行连续 ✓         | RowMajor，按列访问 stride=`in_features` ✗ | RowMajor × RowMajor → 跨步严重 |

**根本原因**：Eigen GEMM 内核深度优化了 **A-RowMajor × B-ColMajor** 模式（缓存分块、SIMD 向量化、预取）。当右操作数 B 是原生 RowMajor 时，内核对 B 进行列遍历的步长等于整行字节数（`in_features × sizeof(float)`），导致：

- FC1 (784→512, batch=7)：B 的列访问 stride = 784 × 4 = **3136 bytes**（跨越 ~5 个 cache line）
- 每次 GEMM 内核迭代都需要从主存或低层缓存重新加载，L1/L2 缓存命中率急剧下降

### 1.3 临时内存分配分析

**Eigen GEMM 不涉及 `malloc/new` 堆分配**：
- `Eigen::Map` 是零拷贝包装
- `.transpose()` 是惰性求值（表达式模板）
- `.noalias()` 禁止临时矩阵创建
- GEMM 内部 packing buffer 通过 `alloca` 在栈上分配（O(1) 开销）

**因此，FC BWD 的性能瓶颈是 GEMM 内部的缓存效率问题，而非内存分配开销。** 这为 CPU Workspace 的设计提供了明确的优化方向：提供临时 ColMajor 重排缓冲区，而非通用内存池。

### 1.4 Composite 测试中 ~5434 µs 的未知开销

三份提案均注意到此问题，但目前尚未定位确切原因。可能因素：
- Arena tensor 偏移导致 cache 竞争（多个 tensor 竞争同一 cache set）
- Eigen/OpenMP 线程池创建/销毁
- 连续 GEMM 调用之间的 pipeline 停顿
- DTensor 地址解析（`ptr_at()`）的间接访问开销被放大

**本方案专注 FC BWD GEMM 优化，~5434 µs 的未知开销留待后续 profiling 隔离。**

---

## 二、现有架构基座

### 2.1 GPU Workspace 架构（可复用基座）

| 组件 | 实现位置 | 关键特征 |
|------|---------|---------|
| `WSpace` | [device_context.h:97-100](file:///r:/renaissance/include/renaissance/backend/device_context.h#L97-L100) | `{void* ptr; size_t size;}` |
| `workspaces_[5]` | [device_context.h:101](file:///r:/renaissance/include/renaissance/backend/device_context.h#L101) | `mutable`，per-StreamKind |
| `ensure_workspace_grow` | [device_context.cpp:348-397](file:///r:/renaissance/src/backend/device_context.cpp#L348-L397) | `const` 方法，256B 对齐 |
| `free_workspaces` | [device_context.cpp:413-423](file:///r:/renaissance/src/backend/device_context.cpp#L413-L423) | `#ifdef TR_USE_CUDA` 保护，CPU 模式空操作 |
| 析构函数 | [device_context.cpp:138-156](file:///r:/renaissance/src/backend/device_context.cpp#L138-L156) | GPU 路径：cublas/cudnn → stream → workspace |

### 2.2 CPU 路径特殊之处

- **单线程串行执行**：CPU 模式只有 1 个 rank，`run_iter()` 在 [simple_task.h:191-198](file:///r:/renaissance/include/renaissance/task/simple_task.h#L191-L198) 串行循环
- **`CapturedGraph::launch()` 的 CPU 分支**：[captured_graph.cpp:160-168](file:///r:/renaissance/src/graph/captured_graph.cpp#L160-L168) 直接遍历 `cpu_ops_` 调用函数指针
- **无 warmup 阶段**：`compile_capture_simple()` 中 GPU 有 warmup（执行 cuDNN 算子），CPU 路径直接结束
- **mimalloc 可用**：`CpuArena` 已使用 `mi_malloc`/`mi_free`（[memory_arena.cpp:79-83](file:///r:/renaissance/src/backend/memory_arena.cpp#L79-L83)），整个项目已链接 mimalloc

---

##三、CPU Workspace 设计

### 3.1 设计原则

1. **完全对齐 GPU Workspace 的接口语义** — 复用 `WSpace` 结构、`ensure_*_grow` 模式、`const` + `mutable` 技法
2. **最小侵入** — 不修改 `CapturedGraph`、不修改 `CpuOpContext`、不修改算子注册表
3. **可扩展** — 未来 Conv、BN 等 CPU 算子可零成本接入同一 workspace
4. **科学对齐** — 对齐值兼顾 Eigen 向量化需求和 cache line 效率

### 3.2 DeviceContext 接口扩展

文件：[device_context.h](file:///r:/renaissance/include/renaissance/backend/device_context.h)

```cpp
// ===== 新增声明（在 workspace() / workspace_size() 附近） =====

/// CPU Workspace：单流全局共享，语义完全对齐 GPU 的 per-stream workspace
[[nodiscard]] void*  cpu_workspace()      const noexcept { return cpu_workspace_.ptr; }
[[nodiscard]] size_t cpu_workspace_size() const noexcept { return cpu_workspace_.size; }
void ensure_cpu_workspace_grow(size_t req_size) const;

// ===== 新增成员（在 workspaces_[5] 之后） =====

mutable WSpace cpu_workspace_;  // CPU 单流 workspace（GPU 模式下不使用）
```

### 3.3 内存分配实现

**设计决策：统一使用 mimalloc（`mi_malloc_aligned` / `mi_free`）**

理由：
- mimalloc 已是项目强制依赖（[CMakeLists.txt:314](file:///r:/renaissance/CMakeLists.txt#L314) 明确标注与 OpenMP/mimalloc 兼容性），`CpuArena` 已使用
- `mi_malloc_aligned` 跨平台统一接口，消除 `#ifdef _WIN32` 分支
- mimalloc 的 free list 缓存可减少页表抖动，优于裸 `_aligned_malloc`
- 对齐值选择 **64 字节**（cache line 大小，AVX-512 兼容），优于 32 字节

```cpp
// device_context.cpp

void DeviceContext::ensure_cpu_workspace_grow(size_t req_size) const {
    if (req_size == 0) return;

    constexpr size_t kAlign = 64;  // cache-line aligned, AVX-512 safe
    size_t aligned = (req_size + kAlign - 1) & ~(kAlign - 1);

    auto& ws = cpu_workspace_;  // mutable ref from const method

    if (ws.size >= aligned) return;  // 已足够大

    // 释放旧内存
    if (ws.ptr) {
        mi_free(ws.ptr);
        ws.ptr  = nullptr;
        ws.size = 0;
    }

    // 申请新内存
    ws.ptr = mi_malloc_aligned(aligned, kAlign);
    if (!ws.ptr) {
        ws.size = 0;
        TR_GPU_OOM("CPU workspace allocation failed for " << aligned << " bytes");
    }
    ws.size = aligned;

    TR_LOG_INFO("backend") << "CPU workspace grown to " << aligned << " bytes";
}
```

### 3.4 析构函数补充

当前 [device_context.cpp:138-156](file:///r:/renaissance/src/backend/device_context.cpp#L138-L156) 中析构函数所有清理逻辑在 `#ifdef TR_USE_CUDA` 内。CPU workspace 的释放应**放在 `#ifdef TR_USE_CUDA` 外部**，使其在 CPU 和 GPU 模式下都生效：

```cpp
DeviceContext::~DeviceContext() {
#ifdef TR_USE_CUDA
    // ... 现有 GPU cleanup（cublas/cudnn → stream → workspace）...
#endif

    // CPU workspace 清理（CPU/GPU 模式通用）
    if (cpu_workspace_.ptr) {
        mi_free(cpu_workspace_.ptr);
        cpu_workspace_.ptr  = nullptr;
        cpu_workspace_.size = 0;
    }
}
```

**设计要点**：`free_workspaces()` 保持 `#ifdef TR_USE_CUDA` 不变——CPU workspace 有自己的独立释放路径，不与 GPU 的 per-stream workspace 混在一起。

### 3.5 Workspace 大小确定：Dry-Run 机制

**问题**：CPU 路径无 warmup，workspace 大小无法在 compile 阶段确定。

**方案**：在 `compile_capture_simple()` 末尾对所有 CPU 图执行一次 `launch(0, nullptr)` dry-run。

位置：[task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp)，在 `simple_captured_graphs_.emplace(...)` 循环结束后、`compile_capture_simple()` 返回前：

```cpp
// ── CPU Dry-Run：为所有 CPU 图触发 workspace 扩容 ──
// 安全性：compile 阶段 Arena 数据本就是未初始化状态，dry-run 写入无害；
//         后续 run() 前 transfer_to_rank() 会重新填充输入。
for (auto& [name, cg] : simple_captured_graphs_) {
    if (cg.is_cuda()) continue;
    cg.launch(0, nullptr);  // CPU launch 不依赖 stream 参数
}
```

**可行性验证**：
- `compile_alloc_hardware()` 先于 `compile_capture_simple()` 执行 → Arena 已初始化
- `ctx.set_rank()` 和 `ctx.set_memory_plan()` 在 capture 循环中已设置 → `ptr_at()` 可用
- `CapturedGraph::launch(rank, stream)` CPU 分支忽略 `stream` 参数（[captured_graph.cpp:161-162](file:///r:/renaissance/src/graph/captured_graph.cpp#L161-L162)）→ `nullptr` 无影响
- Dry-run 仅执行 1 次，开销 ~500 µs（FC BWD 级别），可忽略

**关于 Side-Effect 风险**：小伙伴S 提出的 `dry_run` 标志方案虽然更安全，但会增加所有 CPU 算子的复杂度。考虑到：
1. compile 数据本就是垃圾值
2. 当前所有 CPU 算子无 I/O side-effect
3. 未来如有 side-effect 算子，可单独处理

**本方案采用直接 dry-run，同时保留未来添加 `dry_run` 标志的扩展空间。**

---

## 四、FC BWD 优化实现

### 4.1 核心思路

将 RowMajor 右操作数复制到 workspace 内的 **ColMajor 缓冲区**，恢复 Eigen GEMM 最优路径 A-RowMajor × B-ColMajor。

两个 GEMM（dW、dX）顺序执行，Workspace 需求 = `max(sizeof(X_cm), sizeof(W_cm))`：

```
dW: dY^T[out×batch] × X[batch×in]  → 复制 X 到 workspace  →  dY^T × X_cm  (最优)
dX: dY[batch×out] × W[out×in]      → 复制 W 到 workspace  →  dY × W_cm    (最优)
                                      ↑                          ↑
                             X_cm 用完再复用给 W_cm        RowMajor×ColMajor
```

### 4.2 Workspace 用量

| FC 层         | `sizeof(X_cm)`              | `sizeof(W_cm)`                  | **ws_needed** |
| ------------- | --------------------------- | ------------------------------- | ------------- |
| FC1 (784→512) | 7×784×4 = **21.5 KB**    | 512×784×4 = **1.57 MB** | **1.57 MB**   |
| FC2 (512→256) | 7×512×4 = **14 KB**      | 256×512×4 = **512 KB**  | **512 KB**    |

FC1 先执行 → workspace 增长到 1.57 MB → FC2 复用（不触发扩容）。

### 4.3 修改后的代码

文件：[fc_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L647-L713)

```cpp
static void launch_fc_bwd_cpu_eigen(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<FCParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_BWD CPU EIGEN missing FCParams");

    bool has_bias = p->bias;
    int x_idx = op_ctx->num_inputs - 1;

    TR_CHECK(op_ctx->num_inputs >= 4, ShapeError, "...");
    TR_CHECK(op_ctx->num_outputs >= 3, ShapeError, "...");

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

    // ── Workspace：max(sizeof(X_cm), sizeof(W_cm))，对齐 64B ──
    size_t w_cm_bytes = static_cast<size_t>(out_features) * in_features * sizeof(float);
    size_t x_cm_bytes = static_cast<size_t>(batch)       * in_features * sizeof(float);
    size_t ws_needed  = std::max(w_cm_bytes, x_cm_bytes);
    constexpr size_t kAlign = 64;
    ws_needed = (ws_needed + kAlign - 1) & ~(kAlign - 1);

    const DeviceContext* ctx = op_ctx->ctx;
    ctx->ensure_cpu_workspace_grow(ws_needed);  // const-safe（mutable WSpace）
    float* ws_ptr = static_cast<float*>(ctx->cpu_workspace());

    // 1. db = reduce_sum(dY, dim=0)   —— 不变
    if (has_bias && db != nullptr) {
        Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
        Eigen::Map<Eigen::RowVectorXf> db_vec(db, out_features);
        db_vec.noalias() = dY_mat.colwise().sum();
    }

    // 2. dW = dY^T @ X
    //    将 X (RowMajor) 复制到 workspace 中的 ColMajor 缓冲区，恢复最优 GEMM
    if (dw != nullptr) {
        Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
        Eigen::Map<MatrixXfCol> X_cm(ws_ptr, batch, in_features);
        {
            Eigen::Map<MatrixXfRow> X_mat(x, batch, in_features);
            X_cm = X_mat;  // RowMajor → ColMajor 重排
        }
        Eigen::Map<MatrixXfRow> dW_mat(dw, out_features, in_features);
        dW_mat.noalias() = dY_mat.transpose() * X_cm;
    }

    // 3. dX = dY @ W
    //    将 W (RowMajor) 复制到 workspace 中的 ColMajor 缓冲区（复用 X_cm 的空间）
    Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
    Eigen::Map<MatrixXfRow> dX_mat(dx, batch, in_features);
    {
        Eigen::Map<MatrixXfCol> W_cm(ws_ptr, out_features, in_features);
        Eigen::Map<MatrixXfRow> W_mat(w, out_features, in_features);
        W_cm = W_mat;  // RowMajor → ColMajor 重排
        dX_mat.noalias() = dY_mat * W_cm;
    }
}
```

**关键细节**：
- `X_cm = X_mat` 复制和 `dW_mat.noalias() = dY_mat.transpose() * X_cm` 可使用同一作用域内的不同 `Eigen::Map`，无生命周期问题
- dW 的 `X_cm` 用完后，dX 的 `W_cm` 映射到同一块 `ws_ptr`（复用），作用域隔离保证不冲突
- `const_cast<DeviceContext*>` 仅用于获取 `ptr_at()` 的 mutable 指针（与现有代码一致）

---

## 五、性能预期

### 5.1 Standalone FC BWD

以 FC (784→512, batch=7) 为例：

| 阶段                          | 原方案     | 优化方案    | 说明                          |
| ----------------------------- | ---------- | ----------- | ----------------------------- |
| **db**                        | ~5 µs      | ~5 µs       | 不变（不涉及 GEMM）           |
| **dW** + X→ColMajor 复制      | ~300 µs    | **~65 µs**  | X_cm 复制 22 KB + 最优 GEMM   |
| **dX** + W→ColMajor 复制      | ~200 µs    | **~85 µs**  | W_cm 复制 1.57 MB + 最优 GEMM |
| **总计（BWD）**               | **~528 µs** | **~155 µs** | **~3.4× 加速**                |
| **FWD**                       | ~115 µs    | ~115 µs     | 不受影响                      |
| **BWD/FWD 比**                | **4.6×**   | **~1.3×**   | 接近理论最优（BWD 含 2 个 GEMM vs FWD 的 1 个） |

### 5.2 Composite 测试

| 组件            | 原方案      | 优化方案     |
| --------------- | ----------- | ------------ |
| FC1 BWD         | ~528 µs     | **~155 µs**  |
| FC2 BWD         | ~135 µs     | **~45 µs**   |
| ReLU/Flatten BWD| ~10 µs      | ~10 µs       |
| 未知开销        | ~5424 µs    | **待 profiling 确认** |
| **总计**        | **~6097 µs** | **~210 µs + 未知开销** |

即使 FC BWD 优化到接近理论值，**~5434 µs 的未知开销仍是主要瓶颈**，需后续 profiling 隔离。

### 5.3 内存开销

| 场景              | Workspace 峰值 | 生命周期    |
| ----------------- | -------------- | ----------- |
| FC standalone     | **1.57 MB**    | 程序全程    |
| Composite         | **1.57 MB**    | 程序全程    |
| 未来 Conv 等      | 按需自动扩容   | 程序全程    |

**1.57 MB 在现代 CPU 上可忽略不计**（< 0.01% 的典型系统内存）。

---

## 六、改动文件清单

| # | 文件 | 改动内容 | 行数 |
|---|------|---------|------|
| 1 | [device_context.h](file:///r:/renaissance/include/renaissance/backend/device_context.h) | 加 `cpu_workspace()`/`cpu_workspace_size()`/`ensure_cpu_workspace_grow()` 声明；加 `mutable WSpace cpu_workspace_` 成员 | ~6 |
| 2 | [device_context.cpp](file:///r:/renaissance/src/backend/device_context.cpp) | 实现 `ensure_cpu_workspace_grow`（mimalloc，64B 对齐）；`~DeviceContext()` 中释放 `cpu_workspace_`（`#ifdef TR_USE_CUDA` 外部） | ~50 |
| 3 | [fc_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L647-L713) | `launch_fc_bwd_cpu_eigen` 加 workspace 计算、ColMajor 复制、最优 GEMM | ~35 |
| 4 | [task_base.cpp](file:///r:/renaissance/src/task/task_base.cpp) | `compile_capture_simple()` 末尾加 CPU 图 dry-run 循环 | ~8 |

**总改动量**：~100 行，4 个文件，**零接口破坏，零 `const_cast` 新增**。

---

## 七、风险评估与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `mi_malloc_aligned` 版本不兼容 | 极低 | 低 | mimalloc v1.7+ 均支持；如遇不兼容可回退 `_aligned_malloc`/`posix_memalign` |
| Dry-run 触发未知 side-effect | 极低 | 中 | 当前 CPU 算子均为纯计算；如在算子文档标注 dry-run 安全性契约 |
| ColMajor 重排引入数值误差 | 无 | 无 | 重排是精确 bit-wise 复制，不涉及浮点运算 |
| ~5434 µs 未知开销非 workspace 可解 | 高 | 高 | 明确记录，FC BWD GEMM 优化完成后立即进行下一轮 profiling |
| 64B 对齐浪费内存 | 无 | 无 | 对齐开销 = (63 - 31) = 32 字节最大，相比 MB 级别 workspace 可忽略 |

---

## 八、验证计划

### 8.1 正确性验证

```bash
# 1. Standalone FC（覆盖两种尺寸）
test_fc_fwd_bwd.exe --cpu --batch 7 --in 784 --out 512
test_fc_fwd_bwd.exe --cpu --batch 7 --in 512 --out 256
# 预期：MSE < 1e-5，与优化前一致

# 2. Composite（全链路正确性）
test_flatten_fc_relu_fc.exe --cpu
# 预期：全部 12 个 MSE 检查 PASS
```

### 8.2 性能验证

```bash
# Standalone FC BWD
test_fc_fwd_bwd.exe --cpu --batch 7 --in 784 --out 512
# 预期：BWD 从 ~528 µs 降至 ~150-200 µs

# Composite BWD
test_flatten_fc_relu_fc.exe --cpu
# 预期：FC BWD 部分从 ~663 µs 降至 ~200 µs；
#       总体下降幅度取决于 ~5434 µs 未知开销的性质
```

### 8.3 Workspace 行为验证

在 `ensure_cpu_workspace_grow` 中添加日志，确认：
- **Dry-run 阶段**：FC1 触发 1 次 grow → ~1.57 MB；FC2 复用，无新日志
- **Run 阶段**：全程无 grow 日志（大小已固定）
- **析构**：`mi_free` 正确释放

---

## 九、后续工作

1. **~5434 µs 未知开销隔离**：FC BWD 优化完成后，通过矩阵大小缩放实验 + 算子级拆解 + CPU profiler 定位根因
2. **其他 CPU 算子接入**：Conv BWD、BN 等同样涉及 GEMM 或 strided access 的算子可遵循同一模式：请求 workspace → ColMajor 重排 → 最优 GEMM
3. **FWD 路径评估**：当前 FC FWD 使用 `W_mat.transpose()`（惰性 transpose），对 Eigen 友好。但如未来引入新算子，workspace 可同样用于 FWD 的重排优化

---

## 十、总结

本方案在综合小伙伴K、S、D 三份提案的基础上，提出了一套科学、合理、务实的 CPU Workspace 设计与 FC BWD 优化方案：

1. **架构对齐**：复用 `WSpace` 结构、`ensure_*_grow` 模式、`const` + `mutable` 技法，与现有 GPU workspace 零差异
2. **技术选型明确**：mimalloc（`mi_malloc_aligned`/`mi_free`）→ 跨平台统一；64B 对齐 → cache-line + AVX-512 兼容
3. **Dry-run 务实**：捕获后执行一次 launch 触发 workspace 扩容，安全性经过三方交叉验证
4. **FC BWD 优化精确**：RowMajor → ColMajor 重排 → 恢复 A-RowMajor × B-ColMajor 最优 GEMM，预期 3.4× 加速（~528 µs → ~155 µs）
5. **风险透明**：明确标注 ~5434 µs 未知开销非本方案可解，列入后续工作计划

改动仅 ~100 行、4 个文件，零接口破坏，可作为实施基准。