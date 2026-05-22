# FC 算子性能优化方案

## 1. 现状概览

### 当前各路径实现与耗时（shape: batch=512, in=1024, out=1000）

| 路径 | 算力提供 | FWD (μs) | BWD (μs) | 方法 |
|------|---------|----------|----------|------|
| CPU FP32 | Eigen3 | 2,077 | 8,024 | GEMM（已优化） |
| GPU FP32 | 手写 kernel | **10,683** | **2,693** | 朴素逐元素 CUDA kernel |
| AMP FP16 | cuDNN FE + cuBLAS | **3,079** | **152** | cuDNN FE Matmul / cuBLAS HGEMM |

### 核心问题

GPU FP32 FWD（10.7ms）比 CPU 的 Eigen3（2.1ms）还慢 5 倍。
GPU FP32 BWD（2.7ms）虽然比 CPU（8.0ms）快 3 倍，但与 AMP BWD（0.15ms）差距 18 倍。
AMP FWD（3.1ms）虽然合理，但与同路径的 AMP BWD（0.15ms）差距 20 倍。

### 根因

**GPU FP32 全路径（FWD + BWD dW + BWD dX）全部使用手写的朴素逐元素 CUDA kernel**，位于
[fc_op.cu](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cu)（共 210 行，4 个 kernel）。
这些 kernel 的特点是：

- **无 shared memory tiling**：每个线程独立从 global memory 读取所有操作数
- **无 warp-level 协作**：线程间无数据复用
- **内存访问模式差**：每个线程做 `in_features`（最坏 1024 次）global memory 读取，无 coalesced 策略

以 `fc_fwd_fp32_kernel`（[fc_op.cu L20-L50](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cu#L20-L50)）为例：
```
每个线程：for (i = 0; i < 1024; i++) sum += x_row[i] * w_row[i];
```
总共 512K 线程 × 1024 次循环 × 2 读 = **~1B global memory 访问**。没有任何 L1/shared 缓存利用。

与之对比，**AMP BWD 使用 cuBLAS**（[fc_op.cpp L377-L422](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L377-L422)），
3 次 `cublasGemmEx` 调用完成 db/dW/dX，仅 **152 μs**。

## 2. 事件同步检查

用户特别要求检查 "是否有同步缺失导致耗时极短"。

### AMP BWD（[fc_op.cpp L367-L424](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L367-L424)）

```
1. db kernel launch  ──── (COMP_1 stream)
2. dW: cublasGemmEx   ── (COMP_1 stream)
3. cudaEventRecord(dw_done) + cudaStreamWaitEvent  ← 事件屏障
4. dX: cublasGemmEx   ── (COMP_1 stream)
5. cudaEventRecord(last_done_event) ← 跨算子屏障
```

✅ 正确：所有操作在同一 CUDA 流，顺序执行。事件屏障确保 dW 完全完成后才开始 dX（此时 dX 可安全覆写 X）。`last_done_event` 供框架跨算子同步。

### GPU FP32 BWD（[fc_op.cpp L524-L558](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L524-L558)）

同 AMP BWD 的同步模式。✅ 正确。

### 耗时解释

| 路径 | BWD 耗时 | 同步状态 | 归因 |
|------|---------|---------|------|
| AMP BWD | 152 μs | ✅ | cuBLAS Tensor Core HGEMM，极快（正常） |
| GPU FP32 BWD | 2,693 μs | ✅ | 手写 kernel，无 tiling，慢但无缺失（正常） |

**无同步缺失。** AMP BWD 的 152 μs 是 cuBLAS FP16 的正常性能水平。

## 3. 优化方案

### 3.1 总体策略

**将所有 FP32/AMP 的矩阵运算统一迁移到 cuBLAS，与 AMP BWD 对齐。**

| 变更路径 | 当前方法 | 改后方法 | 预期加速 |
|---------|---------|---------|---------|
| **GPU FP32 FWD** | 手写 kernel | cuBLAS SGEMM | **~50x** (10.7ms → ~200μs) |
| **GPU FP32 BWD dW** | 手写 kernel | cuBLAS SGEMM | **~18x** (含在 2.7ms 内) |
| **GPU FP32 BWD dX** | 手写 kernel | cuBLAS SGEMM | **~18x** (含在 2.7ms 内) |
| **GPU FP32 BWD db** | 手写 kernel (keep) | 保留（非瓶颈） | — |
| **AMP FWD** | cuDNN FE Matmul | cuBLAS HGEMM | **~20x** (3.1ms → ~150μs) |
| **AMP BWD dX/dW/db** | cuBLAS (keep) | 保留 | — |

### 3.2 GPU FP32 FWD → cuBLAS SGEMM

**数学**：Y[b, o] = Σ_i X[b, i] · W[o, i]

**cuBLAS 参数推导**（col-major layout）：

- W 在 NHWC 中 shape = `[O, 1, 1, I]`，flat 偏移 `w[o, i] = o·w_ns + i`
- W 按 col-major 解释为 `[I, O]`：`W_col[i, o] = i + o·w_ns = o·w_ns + i = W[o, i]`，即 `W_col = W^T`
- `CUBLAS_OP_T` 将 W_col 转为 `[O, I]` = W
- X 按 col-major 解释为 `[I, B]`：`X_col = X^T`
- `CUBLAS_OP_N` 保持 X_col = X^T = `[I, B]`

**调用**（等价于 `Y^T = W @ X^T`，结果 C 按 col-major `[O, B]` 与 NHWC `Y[B,1,1,O]` 布局一致）：

```cpp
cublasGemmEx(handle,
    CUBLAS_OP_T,                           // W_col → W [O, I]
    CUBLAS_OP_N,                           // X_col → X^T [I, B]
    out_features,                          // m (C rows = O)
    batch,                                 // n (C cols = B)
    in_features,                           // k
    &alpha,
    w,  CUDA_R_32F,  static_cast<int>(dt_w.n_stride_cuda()),  // lda ≥ I
    x,  CUDA_R_32F,  static_cast<int>(dt_x.n_stride_cuda()),  // ldb ≥ I
    &beta,
    y,  CUDA_R_32F,  static_cast<int>(dt_y.n_stride_cuda()),  // ldc ≥ O
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

**bias 处理**：cuBLAS 不支持内联 bias。需要追加一个小 kernel：
```
// Y[b, o] += B[o]（单次 kernel launch，O 个线程各负责一列）
```
可以参考现有 `fc_bwd_db_amp_kernel`（[fc_op.cu L118-L132](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cu#L118-L132)）的模式写一个简单的 FP32 bias add kernel（或者也改用 cuBLAS `cublasSaxpy` 逐行处理，但 kernel 更简单直接）。

### 3.3 GPU FP32 BWD dW → cuBLAS SGEMM

**数学**：dW[o, i] = Σ_b dY[b, o] · X[b, i]

与 AMP BWD dW 完全相同的 cuBLAS 调用模式（仅数据类型从 FP16 换为 FP32）：

```cpp
cublasGemmEx(handle,
    CUBLAS_OP_N,                           // X_col [I,B] no transpose
    CUBLAS_OP_T,                           // dY_col [O,B] transposed
    in_features, out_features, batch,
    &alpha,
    x,  CUDA_R_32F,  x_ns,
    dy, CUDA_R_32F,  dy_ns,
    &beta,
    dw, CUDA_R_32F,  dw_ns,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

### 3.4 GPU FP32 BWD dX → cuBLAS SGEMM

**数学**：dX[b, i] = Σ_o dY[b, o] · W[o, i]

与 AMP BWD dX 完全相同的 cuBLAS 调用模式：

```cpp
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    in_features, batch, out_features,
    &alpha,
    w,  CUDA_R_32F,  w_ns,
    dy, CUDA_R_32F,  dy_ns,
    &beta,
    dx, CUDA_R_32F,  dx_ns,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

### 3.5 AMP FWD → cuBLAS HGEMM

**数学**：与 FP32 FWD 完全相同，仅数据类型变化。

与 §3.2 的 FP32 FWD 调用完全一致，仅 `CUDA_R_32F` → `CUDA_R_16F`，指针类型变 `__half*`：

```cpp
cublasGemmEx(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    out_features, batch, in_features,
    &alpha,
    w,  CUDA_R_16F,  w_ns,
    x,  CUDA_R_16F,  x_ns,
    &beta,
    y,  CUDA_R_16F,  y_ns,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

AMP FWD 的 bias 需要单独的 FP16→FP32 混合精度 bias add kernel（bias 为 FP32）。

### 3.6 GPU FP32 BWD db — 保留不动

`fc_bwd_db_fp32_kernel`（[fc_op.cu L79-L93](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cu#L79-L93)）是个简单的 reduction kernel，
O 个线程各做 B 次加法。在本测试中这是 O=1000 线程 × 512 次循环 ≈ 512K 次加法，与 GEMM 的 524M 次 FMA 相比微不足道。保留不动。

### 3.7 cuBLAS handle 管理

**当前状态**（[fc_op.cpp L341-L362](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L341-L362)）：
```cpp
static std::unordered_map<cudaStream_t, cublasHandle_t> s_cublas_handles;
// + CublasHandleCleanup RAII
```

✅ 已实现 RAII 清理（[fc_op.cpp L343-L350](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L343-L350)）。

**建议**：将 cuBLAS handle 从 AMP BWD 内部提升为 `DeviceContext` 级别的共享资源，
避免 3 个 launch 函数各维护一套 `static unordered_map`。

## 4. 代码清理

### 4.1 删除所有手写 FP32 CUDA kernel

[fc_op.cu](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cu) 中可以删除：
- `fc_fwd_fp32_kernel`（L20-L50）
- `fc_bwd_fp32_kernel`（L52-L77）
- `fc_bwd_dw_fp32_kernel`（L95-L116）
- `launch_fc_fwd_kernel`（L134-L148）
- `launch_fc_bwd_kernel`（L150-L164）
- `launch_fc_bwd_dw_kernel`（L180-L194）

保留（仍被 AMP BWD 使用）：
- `fc_bwd_db_amp_kernel`（L118-L132）
- `launch_fc_bwd_db_amp_kernel`（L196-L208）
- `fc_bwd_db_fp32_kernel`（L79-L93）
- `launch_fc_bwd_db_kernel`（L166-L178）

### 4.2 删除 AMP FWD cuDNN FE 图构建

[fc_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp) 中可以删除：
- `FcAmpFwdCache` 结构体（L78-L85）
- `s_fc_amp_fwd_caches` 静态 map（L87-L88）
- `build_fc_amp_fwd_graph` 函数（L94-L174）
- `launch_fc_amp_fwd_cuda` 中的 cuDNN FE 重建逻辑（L225-L269）

仅保留 cuBLAS 调用 + bias add kernel 的简洁实现。

### 4.3 新增：bias add kernel

FP32 FWD bias add（简单 kernel）：
```cpp
__global__ void fc_fwd_bias_add_fp32_kernel(
    float* y, const float* b,
    int batch, int out_features, size_t y_n_stride)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_features) return;
    float bval = b[o];
    for (int b = 0; b < batch; ++b)
        y[b * y_n_stride + o] += bval;
}
```

AMP FWD bias add（FP16 数据 + FP32 bias）：
```cpp
__global__ void fc_fwd_bias_add_amp_kernel(
    __half* y, const float* b,
    int batch, int out_features, size_t y_n_stride)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_features) return;
    float bval = b[o];
    for (int b = 0; b < batch; ++b)
        y[b * y_n_stride + o] = __float2half(
            __half2float(y[b * y_n_stride + o]) + bval);
}
```

## 5. 修改清单

| 优先级 | 文件 | 变更 | 影响 |
|--------|------|------|------|
| 🔴 P0 | `fc_op.cpp` L178-L281 | 重写 `launch_fc_amp_fwd_cuda`：cuDNN FE → cuBLAS + bias kernel | AMP FWD 性能 ~20x |
| 🔴 P0 | `fc_op.cpp` L427-L472 | 重写 `launch_fc_fwd_cuda`：手写 kernel → cuBLAS + bias kernel | FP32 FWD 性能 ~50x |
| 🔴 P0 | `fc_op.cpp` L524-L541 | BWD dW/dX：手写 kernel → cuBLAS（参考 AMP BWD 模式） | FP32 BWD 性能 ~18x |
| 🟡 P1 | `fc_op.cpp` L78-L174 | 删除 `FcAmpFwdCache` + `build_fc_amp_fwd_graph` | 减少 ~100 行代码 |
| 🟡 P1 | `fc_op.cu` L20-L77, L95-L116, L134-L164, L180-L194 | 删除 FP32 手写 kernel | 减少 ~120 行代码 |
| 🟡 P1 | `fc_op.cu` | 新增 `fc_fwd_bias_add_fp32_kernel` + `fc_fwd_bias_add_amp_kernel` | 新增 ~30 行 |
| 🟢 P2 | `fc_op.cpp` L341-L362 | cuBLAS handle 提升为共享（避免 3 个 launch 各维护一份） | 代码整洁 |

## 6. 预期结果

| 路径 | 当前 FWD | 优化后 FWD | 当前 BWD | 优化后 BWD |
|------|---------|-----------|---------|-----------|
| GPU FP32 | 10,683 μs | **~200 μs** | 2,693 μs | **~350 μs** |
| AMP FP16 | 3,079 μs | **~150 μs** | 152 μs | 152 μs（不变） |

优化后所有 GPU 路径均统一使用 cuBLAS，代码量减少 ~200 行，FP32 BWD 仍保留完整的事件同步（db → dW → barrier → dX → last_done_event）。