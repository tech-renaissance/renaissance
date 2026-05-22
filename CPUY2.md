# CPU Workspace 设计与 FC BWD 优化综合方案

## 一、数据回顾与问题诊断

### 1.1 实测数据

| 测试场景 | FWD | BWD | BWD/FWD | 备注 |
|---------|-----|-----|---------|------|
| Standalone FC (784→512, batch=7) | 113.5 us | **568.2 us** | 5.0× | 单节点，1 FC BWD |
| Standalone FC (512→256, batch=7) | 38.0 us | **81.2 us** | 2.1× | 单节点，1 FC BWD |
| Composite (Flatten+FC+ReLU+FC) | 153.7 us | **3681.8 us** | **24.0×** | 4 节点，2 FC BWD |

**关键异常**：Composite BWD（3682 us）远大于两个 Standalone FC BWD 之和（568 + 81 = **649 us**），差距约 **5.7 倍**。即使叠加 ReLU/Flatten BWD（~20 us），理论值仅 ~670 us。

这意味着：**FC BWD 的 GEMM 跨步问题只能解释一部分 slowdown，Composite 场景中还存在一个更大的系统性开销。**

### 1.2 根因分层

| 层级 | 问题描述 | 影响量级 | 确定性 |
|------|---------|---------|--------|
| **L1: GEMM 跨步** | BWD dW/dX 的右操作数为 RowMajor，Eigen GEMM 列访问 stride 极大 | BWD 慢 2~5× | ✅ 已确认 |
| **L2: OpenMP 并行** | 项目 CMake 启用了 OpenMP，Eigen 对小矩阵（batch=7）仍可能触发多线程，同步开销远超计算收益 | Composite 额外慢 3~5× | ⚠️ 高概率 |
| **L3: 其他** | Arena cache 布局、run_iter 循环开销、数据依赖等 | 未知 | ❓ 待验证 |

**核心结论**：如果只修复 L1（ColMajor 重排），Standalone FC BWD 会大幅改善，但 Composite BWD 可能仍有显著差距。必须**同时处理 L1 + L2**。

---

## 二、CPU Workspace 设计

### 2.1 设计原则

完全对齐 GPU Workspace 的接口语义，复用现有基础设施：

| 特性 | GPU Workspace | CPU Workspace（新增） |
|------|--------------|---------------------|
| 管理者 | `DeviceContext` | `DeviceContext` |
| 数量 | 5 个（per StreamKind） | **1 个**（CPU 单流串行） |
| 分配器 | `cudaMalloc` | **`mi_malloc_aligned`**（复用项目统一分配器） |
| 释放器 | `cudaFree` | **`mi_free`** |
| 对齐 | 256 B | **64 B**（cache line + AVX-512） |
| 扩容接口 | `ensure_workspace_grow` | **`ensure_cpu_workspace_grow`** |
| 大小确定 | warmup 阶段 | **capture 后 dry-run** |

**选择 mimalloc 而非平台 API**：项目已在 `memory_arena.h` / `tensor.cpp` 中统一使用 mimalloc，`device_context.cpp` 通过 `#include "renaissance/backend/memory_arena.h"` 已间接包含 `<mimalloc.h>`，无需新增依赖。

### 2.2 DeviceContext 扩展

```cpp
// include/renaissance/backend/device_context.h

class DeviceContext {
public:
    // ... existing ...

    // ── 新增：CPU Workspace（单流全局共享） ──
    [[nodiscard]] void* cpu_workspace() const noexcept { return cpu_workspace_.ptr; }
    [[nodiscard]] size_t cpu_workspace_size() const noexcept { return cpu_workspace_.size; }
    void ensure_cpu_workspace_grow(size_t req_size) const;

private:
    // ... existing workspaces_[5] ...
    mutable WSpace cpu_workspace_;  // CPU 单流全局 workspace
};
```

### 2.3 扩容实现

```cpp
// src/backend/device_context.cpp

void DeviceContext::ensure_cpu_workspace_grow(size_t req_size) const {
    if (req_size == 0) return;

    constexpr size_t kAlign = 64;
    size_t aligned = (req_size + kAlign - 1) & ~(kAlign - 1);

    auto& ws = cpu_workspace_;
    if (ws.size >= aligned) return;

    if (ws.ptr) {
        mi_free(ws.ptr);
        ws.ptr = nullptr;
        ws.size = 0;
    }

    ws.ptr = mi_malloc_aligned(aligned, kAlign);
    if (!ws.ptr) {
        ws.size = 0;
        TR_GPU_OOM("Failed to allocate CPU workspace of " << aligned << " bytes");
    }
    ws.size = aligned;

    TR_LOG_INFO("backend") << "CPU workspace grown to " << aligned << " bytes";
}
```

### 2.4 析构补充

```cpp
DeviceContext::~DeviceContext() {
#ifdef TR_USE_CUDA
    // ... existing GPU cleanup ...
#endif
    if (cpu_workspace_.ptr) {
        mi_free(cpu_workspace_.ptr);
        cpu_workspace_.ptr = nullptr;
        cpu_workspace_.size = 0;
    }
}
```

### 2.5 Capture 阶段 Dry-Run

在 `compile_capture_simple()` 的 capture 循环结束后，对所有 CPU 图执行一次 dry-run，触发 workspace 按需扩容：

```cpp
// src/task/task_base.cpp :: compile_capture_simple()

// 在 simple_captured_graphs_.emplace(...) 循环结束后：

// CPU Warmup: dry-run once to let CPU ops determine workspace sizes.
// Safety: compile phase data is uninitialized; inputs will be refilled
// before the first real run().
for (auto& [name, cg] : simple_captured_graphs_) {
    if (!cg.is_cuda()) {
        cg.launch(0, nullptr);
    }
}
```

---

## 三、FC BWD 双重优化

### 3.1 优化一：ColMajor 重排（修复 L1）

将跨步访问的右操作数复制到 workspace 内的 ColMajor 缓冲区，恢复 Eigen GEMM 最优路径。

**Workspace 用量**：
```
ws_needed = max(out_features * in_features, batch * in_features) * sizeof(float)
```
dW 和 dX 顺序执行，复用同一块内存。

**修改后的 `launch_fc_bwd_cpu_eigen`**：

```cpp
static void launch_fc_bwd_cpu_eigen(CpuOpContext* op_ctx) {
    // ... 参数解析 ...
    int batch = op_ctx->input_shape.n;
    int out_features = op_ctx->input_shape.c;
    int in_features = op_ctx->output_shape.h * op_ctx->output_shape.w * op_ctx->output_shape.c;

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXfCol = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    // ── Workspace ──
    size_t w_bytes = static_cast<size_t>(out_features) * in_features * sizeof(float);
    size_t x_bytes = static_cast<size_t>(batch)       * in_features * sizeof(float);
    size_t ws_needed = std::max(w_bytes, x_bytes);
    constexpr size_t kAlign = 64;
    ws_needed = (ws_needed + kAlign - 1) & ~(kAlign - 1);

    const DeviceContext* ctx = op_ctx->ctx;
    ctx->ensure_cpu_workspace_grow(ws_needed);
    float* ws_ptr = static_cast<float*>(ctx->cpu_workspace());

    // 1. db
    if (has_bias && db != nullptr) {
        Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
        Eigen::Map<Eigen::RowVectorXf> db_vec(db, out_features);
        db_vec.noalias() = dY_mat.colwise().sum();
    }

    // 2. dW = dY^T @ X  (X_cm: ColMajor)
    if (dw != nullptr) {
        Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
        Eigen::Map<MatrixXfRow> X_mat(x, batch, in_features);
        Eigen::Map<MatrixXfRow> dW_mat(dw, out_features, in_features);

        Eigen::Map<MatrixXfCol> X_cm(ws_ptr, batch, in_features);
        X_cm = X_mat;
        dW_mat.noalias() = dY_mat.transpose() * X_cm;
    }

    // 3. dX = dY @ W  (W_cm: ColMajor, reuse workspace)
    Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
    Eigen::Map<MatrixXfRow> dX_mat(dx, batch, in_features);

    Eigen::Map<MatrixXfCol> W_cm(ws_ptr, out_features, in_features);
    {
        Eigen::Map<MatrixXfRow> W_mat(w, out_features, in_features);
        W_cm = W_mat;
    }
    dX_mat.noalias() = dY_mat * W_cm;
}
```

### 3.2 优化二：禁用 Eigen OpenMP（修复 L2）

**问题**：CMakeLists.txt 启用了 OpenMP，`find_package(OpenMP)` + `set(CMAKE_CXX_FLAGS ...)`。Eigen 会自动检测 `_OPENMP` 并在 GEMM 中启用多线程。对于 batch=7 的超小矩阵，线程分发/同步开销远大于计算收益，且 **Composite 场景中连续调用不同尺寸的 GEMM 会反复触发线程池策略评估**，这是 Composite BWD 比 Standalone 之和慢 5.7 倍的最可能解释。

**方案**：在 `DeviceContext` 构造函数中（CPU 模式，`device_id_ < 0`），全局禁用 Eigen 多线程：

```cpp
// src/backend/device_context.cpp

DeviceContext::DeviceContext(int device_id) : device_id_(device_id) {
    if (device_id_ < 0) {
        // CPU 模式：Eigen 对小矩阵的 OpenMP 并行收益为负，全局禁用
#ifdef TR_USE_EIGEN
#ifdef _OPENMP
        Eigen::setNbThreads(1);
        TR_LOG_INFO("backend") << "DeviceContext CPU: Eigen OpenMP disabled (setNbThreads=1)";
#endif
#endif
        return;
    }
    // ... existing GPU init ...
}
```

**为什么放在 DeviceContext 构造函数**：
- CPU 模式只有一个 DeviceContext，初始化一次即可
- 不影响 GPU 路径（GPU 路径不执行此分支）
- 比 CMakeLists.txt 加宏更灵活（运行时决定，无需重新编译）
- 比算子内局部设置更简洁（零运行时重复调用开销）

**兼容性说明**：
- 禁用 OpenMP 后，Eigen 仍保留完整的 SSE/AVX SIMD 向量化（单线程最优）
- 未来若 CPU 支持大 batch 并行训练，可改为动态策略（根据 batch size 决定是否启用）

---

## 四、性能预期

### 4.1 Standalone FC BWD

| 阶段 | 原始 | 优化后 | 说明 |
|------|------|--------|------|
| dW | ~300 us | ~60 us | ColMajor 消除跨步 |
| dX | ~200 us | ~40 us | ColMajor 消除跨步 |
| 复制开销 | - | ~50 us | X_cm(22KB) + W_cm(1.57MB) |
| **总计** | **~568 us** | **~150 us** | **3.8× 提升** |

### 4.2 Composite BWD

**仅做 ColMajor 重排（不做 OpenMP 禁用）**：
- FC1 BWD: 568 → 150 us
- FC2 BWD: 81 → ~30 us（按比例）
- ReLU + Flatten: ~20 us
- **理论合计**: ~200 us
- **但实际可能仍有 ~1000+ us**（OpenMP 额外开销未消除）

**ColMajor 重排 + OpenMP 禁用**：
- 消除 OpenMP 线程同步开销后，Composite BWD 应接近各节点优化后之和
- **预期总计**: ~200 us
- **相比原始 3682 us，提升约 18×**

### 4.3 Workspace 内存

| 场景 | 大小 | 说明 |
|------|------|------|
| FC1 (784→512) | 1.57 MB | W_cm [512, 784] |
| FC2 (512→256) | 512 KB | W_cm [256, 512]，复用已有 workspace |

---

## 五、改动文件清单

| 文件 | 改动内容 | 行数 |
|------|---------|------|
| `include/renaissance/backend/device_context.h` | 添加 `cpu_workspace()` / `cpu_workspace_size()` / `ensure_cpu_workspace_grow()` 声明；添加 `mutable WSpace cpu_workspace_` | ~8 行 |
| `src/backend/device_context.cpp` | 1. 构造函数 CPU 分支添加 `Eigen::setNbThreads(1)`<br>2. 实现 `ensure_cpu_workspace_grow`<br>3. 析构函数补充 `mi_free(cpu_workspace_)` | ~60 行 |
| `src/backend/ops/dtensor/fc_op.cpp` | `launch_fc_bwd_cpu_eigen` 增加 workspace 计算、ColMajor 复制、最优 GEMM | ~35 行 |
| `src/task/task_base.cpp` | `compile_capture_simple()` 末尾增加 CPU 图 dry-run 循环 | ~8 行 |

**总改动量**：约 110 行，4 个文件，零接口破坏。

---

## 六、验证计划

### 6.1 功能正确性

```bash
test_fc_fwd_bwd.exe --cpu --batch 7 --in 784 --out 512
test_fc_fwd_bwd.exe --cpu --batch 7 --in 512 --out 256
test_flatten_fc_relu_fc.exe --cpu
```
**预期**：全部 PASS，MSE 与优化前一致。

### 6.2 性能验证

| 指标 | 优化前 | 优化后预期 | 验证方法 |
|------|--------|-----------|---------|
| Standalone FC1 BWD | ~568 us | ~150 us | `test_fc_fwd_bwd.exe --cpu` |
| Standalone FC2 BWD | ~81 us | ~30 us | `test_fc_fwd_bwd.exe --cpu` |
| Composite BWD | ~3682 us | **~200 us** | `test_flatten_fc_relu_fc.exe --cpu` |
| Workspace grow 次数 | N/A | 1 次（FC1 dry-run） | `ensure_cpu_workspace_grow` 日志 |

### 6.3 OpenMP 验证

在 `launch_fc_bwd_cpu_eigen` 内临时加入：
```cpp
std::cout << "Eigen threads: " << Eigen::nbThreads() << std::endl;
```
**预期输出**：始终为 `1`。

---

## 七、风险与回退

| 风险 | 概率 | 影响 | 回退措施 |
|------|------|------|---------|
| ColMajor 重排引入数值误差 | 极低 | 低 | 重排是精确的 bit-wise 复制，无舍入；MSE 验证可捕获 |
| `Eigen::setNbThreads(1)` 影响其他算子 | 低 | 中 | 所有 CPU 算子均为单线程串行执行，无负面影响；未来需大 batch 并行时可改为动态策略 |
| mimalloc 对齐分配失败 | 极低 | 高 | `mi_malloc_aligned` 回退到 `_aligned_malloc`（Windows）或 `posix_memalign`（Linux） |
| Dry-run 污染后续数据 | 极低 | 中 | compile 阶段数据未初始化，run() 前 `transfer_to_rank` 会重新填充输入 |

---

## 八、总结

本方案在小伙伴 K/S/D 分析的基础上，**增加了一个关键优化：禁用 Eigen OpenMP**。

**原因**：Composite BWD 比 Standalone FC BWD 之和多出约 5.7 倍开销，远超 GEMM 跨步能解释的范围。在项目已启用 OpenMP、CPU 为单线程串行架构、且 batch=7 极小的背景下，Eigen 的多线程并行是最大嫌疑。

**实施顺序**：
1. 实现 CPU Workspace（`DeviceContext` 扩展）
2. 实现 FC BWD ColMajor 重排
3. 在 `DeviceContext` CPU 构造函数中禁用 Eigen OpenMP
4. 添加 dry-run 确定 workspace 大小
5. 运行验证计划，确认正确性和性能收益

**预期最终收益**：
- Standalone FC BWD：**3.8×** 加速
- Composite BWD：**~18×** 加速（从 3682 us 降至 ~200 us）
