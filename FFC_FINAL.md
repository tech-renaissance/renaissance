# FC 算子 2D/4D 输入兼容性 — 最终修改方案

> 版本: 1.0  
> 日期: 2026-06-09  
> 方案整合自: FFC1.md, FFC2.md, FFC3.md  
> 状态: 待实施（先不改代码）

---

## 1. 核心结论

FC 算子的 **FP32/CPU 路径已正确兼容 2D/4D 输入**，无需修改。**AMP 路径存在 3 类 Bug**，需要修复。所有修改集中在 `fc_op.cpp` 的 `FC_AMP_FWD` 和 `FC_AMP_BWD` 两个函数内。

---

## 2. 输入类型与框架保障

### 2.1 框架规定的 Flatten 插入规则

```cpp
// arch_plan_normalize.cpp:145-156
int out_c = get_effective_output_c_at(insert_at - 1, input_.c());
needs_flatten = (out_c % 8 != 0);
```

| C % 8 | Flatten | FC 收到的输入 | Shape |
|-------|---------|-------------|-------|
| == 0 | 不插入 | 原始 4D 张量 | `[N, H, W, C]` |
| != 0 | 插入 | Flatten 重排后的 2D 张量 | `[N, 1, 1, Cf]`，且 `Cf % 8 == 0` |

### 2.2 关键推论：`padded_c == C` 恒成立

由于 FC 的输入在两种情况下均满足 `C % 8 == 0`，而 AMP 模式下 FP16 特征区的 `cuda_alignment = 8`，故：

```
padded_c = align_up(C, 8) = C
n_stride_cuda = padded_c × W × H = C × W × H = H × W × C = in_features
```

**数据在内存中完全紧凑连续，无须处理 stride 与 in_features 不等的 padding 问题。**

### 2.3 等效 C 语义

| 输入类型 | 等效 C (`c_equal`) | 等效 n_stride |
|---------|-------------------|---------------|
| 4D `[N, H, W, C]` | H × W × C | H × W × C |
| 2D `[N, 1, 1, Cf]` | Cf (= 1 × 1 × Cf) | Cf |

FC 层一律使用 `H × W × C`（等效 C）进行运算，不做 `H=1, W=1` 假设。

---

## 3. 各路径审查结论

### 3.1 FP32 CUDA（`FC_FP32_FWD` / `FC_FP32_BWD`）— **已正确，无需修改**

- `in_features = dt_x.shape.h() × dt_x.shape.w() × dt_x.shape.c()` ✓
- stride 使用 `n_stride_cuda()` ✓
- cuBLAS GEMM 参数与 FP32 紧凑布局兼容 ✓
- 无 `H=1, W=1` 检查 ✓

### 3.2 CPU Eigen / 朴素实现 — **已正确，无需修改**

- 均使用 `H×W×C` 作为 `in_features` ✓
- CPU 路径 DTensor 恒为紧凑（`padded_c == C`） ✓

### 3.3 Compiler / LayerDescriptor / ArchPlan — **已正确，无需修改**

| 组件 | 文件:行号 | 代码 | 状态 |
|------|----------|------|------|
| `infer_fc_tensors` | `layer_descriptor_registry.cpp:286` | `in_feat = input.h() * input.w() * input.c()` | ✓ |
| `compile_fc` | `compiler.cpp:326` | `in_features = h * w * c` | ✓ |
| arch_plan_shape | `arch_plan_shape.cpp:43` | FC 输出 `{1,1,1, out_features}` | ✓ |

### 3.4 辅助 Kernel（`fc_op.cu`）— **无需修改**

- `fc_fwd_bias_add_amp_kernel` / `fc_bwd_db_amp_kernel` 均通过 `y_ns` / `dy_n_stride` 参数化 stride，不依赖 H/W 值 ✓

---

## 4. AMP 路径问题清单

### 4.1 Bug 1：硬编码 `H=1, W=1` 断言检查

**文件**: `fc_op.cpp`

| 函数 | 行号 | 代码 | 严重度 |
|------|------|------|--------|
| `FC_AMP_FWD` | 276-278 | `TR_DEBUG_CHECK(dt_x.h() == 1 && dt_x.w() == 1, ...)` | **P0** |
| `FC_AMP_BWD` | 367-368 | `TR_DEBUG_CHECK(dt_dy.h() == 1 && dt_dy.w() == 1, ...)` | P1 |
| `FC_AMP_BWD` | 370-372 | `TR_DEBUG_CHECK(dt_x.h() == 1 && dt_x.w() == 1, ...)` | **P0** |

**影响**: 4D 输入 `[N,H,W,C]`（`C%8==0`，无 Flatten）直接触发 Debug 断言崩溃。

**修复**: 删除全部三处 `TR_DEBUG_CHECK`。

> **关于 `dt_dy` 的检查（367-368）**：FC 输出恒为 `[N,1,1,out_features]`，故 `dt_dy.h()==1 && dt_dy.w()==1` 在数学上恒成立。为统一性和防御性，一并删除。

### 4.2 Bug 2：`in_features` 使用 `dt_x.c()` 而非 `H×W×C`

**文件**: `fc_op.cpp`

| 函数 | 行号 | 当前代码 | 修复 |
|------|------|---------|------|
| `FC_AMP_FWD` cache key | 292 | `dt_x.n(), dt_x.c(), dt_w.n()` | `dt_x.n(), dt_x.h()*dt_x.w()*dt_x.c(), dt_w.n()` |
| `FC_AMP_BWD` | 383 | `int64_t in_features = dt_x.c();` | `int64_t in_features = dt_x.h() * dt_x.w() * dt_x.c();` |

**影响**: 4D 输入时 `in_features` 偏小 `1/(H×W)` 倍，GEMM 维度错误，计算结果完全错误。

### 4.3 Bug 3：cuDNN FE 1×1 Conv 图与 4D 输入不兼容（`FC_AMP_FWD` 专属）

**文件**: `fc_op.cpp:110-223`（`build_fc_amp_fwd_conv_graph`）

**根因**: cuDNN FE 构建的 1×1 卷积图对 `[N,H,W,C]` 输入产出 `[N,H,W,O]`，而 FC 要求输出 `[N,1,1,O]`。shape 不匹配导致 cuDNN 执行失败。

**连带问题**: 整个 cuDNN FE 基础设施（`FcAmpFwdCacheKey`, `FcAmpFwdCache`, `s_fc_amp_fwd_caches`, `build_fc_amp_fwd_conv_graph`, `update_fc_amp_tensor_to_id`，约 170 行）都是为 cuDNN FE Graph 服务的，修复后全部不再需要。

---

## 5. 修改方案

### 5.1 策略选择：cuDNN FE → cuBLAS

| 维度 | 方案 A: 切换 cuBLAS（采纳） | 方案 B: 保留 cuDNN FE + 等效 shape |
|------|--------------------------|----------------------------------|
| 4D 兼容 | ✓ 天然支持（GEMM 基于 stride） | ⚠️ 需手动构造等效 dim/stride |
| 代码量 | ~70 行新增，~170 行删除 | ~20 行修改 |
| 与 FP32 一致性 | ✓ 同一模式 | ✗ 不同实现路径 |
| 与 AMP BWD 一致性 | ✓ 都使用 cuBLAS | ✗ FWD 用 cuDNN，BWD 用 cuBLAS |
| 正确性风险 | 低（cuBLAS GEMM 成熟可靠） | 中（构造等效 stride 有出错风险） |
| 缓存复杂度 | 无需缓存 | 仍需 per-shape 缓存 |

**采纳方案 A**。理由：
1. 与 `FC_FP32_FWD` 实现模式统一，易于维护
2. GEMM 基于 stride 参数，天然支持任意 `(H,W)` 的 4D 输入
3. 与 `FC_AMP_BWD`（已使用 cuBLAS）一致
4. 删除 ~170 行 cuDNN FE 基础设施，净减少 ~100 行代码

### 5.2 删除清单（cuDNN FE 基础设施）

**文件**: `fc_op.cpp`

```cpp
// 删除以下全部内容（第 65-248 行区域）：
//
//   struct FcAmpFwdCacheKey { ... };          // 行 69-83
//   struct FcAmpFwdCacheKeyHasher { ... };    // 行 87-96
//   struct FcAmpFwdCache { ... };             // 行 98-106
//   s_fc_amp_fwd_caches                       // 行 108
//   build_fc_amp_fwd_conv_graph(...)          // 行 110-223
//   namespace {} 闭合花括号                    // 行 225
//   update_fc_amp_tensor_to_id(...)           // 行 234-248
```

同时删除 `fc_op.cpp` 顶部不再需要的 include（如果它们不再被其他函数使用）：
- `#include "renaissance/backend/cudnn_utils.h"` — 如果 Conv 等其他 op 仍需要，保留
- `#include <cudnn_frontend.h>` 和 `#include <cudnn_frontend/graph_interface.h>` — 同上

> **注意**: 这些头文件可能被 Conv 算子的代码使用，须确认后再决定是否删除。保守做法：保留头文件不删。

### 5.3 新增：`launch_fc_amp_fwd_cuda`（cuBLAS GEMM 版本）

**文件**: `fc_op.cpp`

在（删除 cuDNN FE 基础设施后的）`FC_AMP_FWD` 位置写入新实现。核心 GEMM 调用与 `FC_FP32_FWD` 结构一致，仅类型从 `float/FP32` 切换为 `__half/FP16`。

```cpp
static void launch_fc_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<FCParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_AMP_FWD missing FCParams");

    bool has_bias = p->bias;
    TR_CHECK(node.input_ids.size() >= 3, ShapeError,
             "FC_AMP_FWD requires at least 3 inputs. Got " << node.input_ids.size());
    TR_CHECK(node.output_ids.size() >= 1, ShapeError,
             "FC_AMP_FWD requires at least 1 output");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);

    __half* x = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    __half* w = static_cast<__half*>(ctx.ptr_at(node.input_ids[1]));
    float*  b = has_bias ? static_cast<float*>(ctx.ptr_at(node.input_ids[2])) : nullptr;
    __half* y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    int batch        = dt_x.shape.n();
    int in_features  = dt_x.shape.h() * dt_x.shape.w() * dt_x.shape.c();
    int out_features = p->out_features;

    int x_ns = static_cast<int>(dt_x.n_stride_cuda());
    int w_ns = static_cast<int>(dt_w.n_stride_cuda());
    int y_ns = static_cast<int>(dt_y.n_stride_cuda());

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cublasHandle_t cublas_handle = static_cast<cublasHandle_t>(
        ctx.cublas_handle(sk));

    float alpha = 1.0f;
    float beta  = 0.0f;

    // FWD: Y = X @ W^T
    // 等价 cuBLAS: C = α · op(W) · op(X) + β · C
    //   op(W) = W^T  → W[O,I] 变为 I×O (CUBLAS_OP_T)
    //   op(X) = X    → X[B,I] 变为 I×B (CUBLAS_OP_N, col-major 视角)
    //   m=O, n=B, k=I
    cublasStatus_t cb_status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,                     // W^T
        CUBLAS_OP_N,                     // X 保持
        out_features,                    // m = O
        batch,                           // n = B
        in_features,                     // k = I (= H×W×C)
        &alpha,
        w, CUDA_R_16F, w_ns,
        x, CUDA_R_16F, x_ns,
        &beta,
        y, CUDA_R_16F, y_ns,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (cb_status != CUBLAS_STATUS_SUCCESS) {
        TR_DEVICE_ERROR("FC_AMP_FWD cublasGemmEx failed: " << cb_status);
    }

    // Bias add (FP32 bias → FP16 output)
    if (has_bias && b != nullptr) {
        cudaError_t err = launch_fc_fwd_bias_add_amp_kernel(
            y, b, batch, out_features, static_cast<size_t>(y_ns), s);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("FC_AMP_FWD bias-add kernel failed: "
                            << cudaGetErrorString(err));
        }
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

### 5.4 修改：`launch_fc_amp_bwd_cuda`

**文件**: `fc_op.cpp`

仅改三处（删除两段检查 + 修正一行计算）：

#### 5.4.1 删除第 367-372 行

```cpp
// --- 删除 start ---
TR_DEBUG_CHECK(dt_dy.h() == 1 && dt_dy.w() == 1, ShapeError,
               "FC_AMP_BWD dy input must have H=1, W=1. Got H=" << dt_dy.h()
               << ", W=" << dt_dy.w());
TR_DEBUG_CHECK(dt_x.h() == 1 && dt_x.w() == 1, ShapeError,
               "FC_AMP_BWD x input must have H=1, W=1. Got H=" << dt_x.h()
               << ", W=" << dt_x.w());
// --- 删除 end ---
```

#### 5.4.2 修正第 383 行

```cpp
// 修改前:
int64_t in_features  = dt_x.c();

// 修改后:
int64_t in_features  = dt_x.shape.h() * dt_x.shape.w() * dt_x.shape.c();
```

> 注：除 `in_features` 外，BWD 的其他代码（stride 使用、GEMM 调用、流管理、三流并行）无需修改。

---

## 6. 关于 FP16 溢出风险分析

### 6.1 旧方案（cuDNN FE）的 clamp

旧方案在 cuDNN FE Graph 中通过 `clamp_min=-65504` / `clamp_max=65504` 对 FP32 中间结果做裁剪后再写回 FP16。

### 6.2 新方案（cuBLAS）的溢出处理

- cuBLAS 使用 `CUBLAS_COMPUTE_32F`：乘法和累加在 FP32 精度下进行，不存在中间溢出
- 输出写回 FP16 时，由 cuBLAS 内部执行 FP32→FP16 转换
- 对于超出 FP16 范围 `[-65504, 65504]` 的值，转换行为取决于 cuBLAS 实现（通常为饱和转换或产生 Inf）

### 6.3 是否需要独立 clamp kernel？

| 考虑因素 | 分析 |
|---------|------|
| FC 输入范围 | FC 通常接在 BN/Conv 之后，特征值已被归一化到合理范围 |
| 权重范围 | 权重初始化后值较小，训练中通过 optimizer 维持在有限范围内 |
| GEMM 结果 | 单层 FC 的矩阵乘法结果极少超过 FP16 范围 |
| 与 FP32 一致性 | `FC_FP32_FWD` 无 clamp（FP32 范围足够大，无需 clamp） |
| 与 AMP BWD 一致性 | `FC_AMP_BWD` 当前使用 cuBLAS 且无 clamp，已验证可正常工作 |
| 工程开销 | 新增 clamp kernel 需修改 `fc_op.cu`，增加约 50 行代码 |

**建议**: 第一期不加独立 clamp kernel，理由：
1. `FC_AMP_BWD` 的 cuBLAS 路径当前无 clamp 且已验证正确
2. FP16 溢出在单层 FC 的正常训练中极其罕见
3. 如需 clamp，后续可作为独立优化项添加，不影响当前方案的架构

**验证策略**: 若训练中出现 NaN/Inf，优先排查是否是 FC 溢出导致。若是，再补加 clamp kernel。

---

## 7. 修改文件清单

| 文件 | 修改内容 | 净变化 |
|------|---------|--------|
| `src/backend/ops/dtensor/fc_op.cpp` | 删除 cuDNN FE 基础设施 + 删除 H=1,W=1 检查 + 重写 `FC_AMP_FWD` + 修正 `FC_AMP_BWD` in_features | 删 ~180 行，增 ~75 行 |

其余文件 **无需修改**（详见第 3 节审查结论）。

---

## 8. 正确性验证

### 8.1 现有测试（回归验证）

```bash
# FP32 GPU (2D 输入 [N,1,1,I])
test_fc_fwd_bwd.exe --gpu --batch 8 --in 2048 --out 512

# AMP GPU (2D 输入 [N,1,1,I])
test_fc_fwd_bwd.exe --amp --batch 8 --in 2048 --out 512
```

预期：修改前后结果一致（AMP FWD 的 cuBLAS 输出应与旧的 cuDNN FE 输出数值上等效）。

### 8.2 4D 输入验证

构造 BluePrint：`Conv(3×3, C=64) → BN → FC(512)`

- C=64 满足 `C%8==0`，框架不插入 Flatten
- FC 收到 4D 输入 `[N,7,7,64]`，`in_features = 7×7×64 = 3136`
- 训练若干 iteration，验证：
  - 无 `ShapeError` 崩溃
  - loss 正常下降
  - 无 NaN/Inf

---

## 9. 附录：cuBLAS GEMM 在 NHWC 布局下的正确性论证

FC 的 GEMM 调用模式如下（以 FWD 为例）：

```
cuBLAS 列主序视角:
  C[m×n] = α · op(A)[m×k] · op(B)[k×n] + β · C[m×n]

实参映射:
  op(A) = W^T  (CUBLAS_OP_T)   →  W 的列主序视角为 [I,O]，转置后为 [O,I]
  op(B) = X    (CUBLAS_OP_N)   →  X 的列主序视角为 [I,B]
  m = O, n = B, k = I

物理内存:
  W: RowMajor [O,I], lda = w_ns = n_stride_cuda
  X: RowMajor [B,I], ldb = x_ns = n_stride_cuda
  Y: RowMajor [B,O], ldc = y_ns = n_stride_cuda

正确性条件:
  lda >= max(1, m) for CUBLAS_OP_T → lda >= max(1, k) → w_ns >= I  ✓
  ldb >= max(1, m) for CUBLAS_OP_N → ldb >= max(1, k) → x_ns >= I  ✓
  ldc >= max(1, m)                  → ldc >= m       → y_ns >= O  ✓
```

当 `C%8==0` 时 `n_stride_cuda == H×W×C == in_features`，上述条件全部满足。✓

---

## 10. 与三份原方案的差异总结

| 项目 | FFC1 | FFC2 | FFC3 | FFC_FINAL |
|------|------|------|------|-----------|
| FP32 路径 | ✓ 正确 | ✓ 正确 | ✓ 正确 | ✓ 正确（同） |
| AMP FWD 方案 | cuBLAS | cuBLAS（方案A）| cuBLAS（方案A）| **cuBLAS** |
| AMP BWD 方案 | 统一修改 | 统一修改 | 统一修改 | 同 |
| clamp 处理 | 不需要（饱和转换）| 未明确 | 需要新增 kernel | **不需要（与 BWD 一致）** |
| padded_c != C | 未讨论 | 未讨论 | 末讨论但指出 FP32 无此风险 | **确认恒有 padded_c==C** |
| 4D + C%8!=0 | 未讨论 | 未讨论 | 提到应避免 | **不会发生（Flatten 已保障）** |
| cuDNN FE 备用方案 | 无 | 方案B（保留）| 方案B（保留）| **不保留（代码净减少）** |