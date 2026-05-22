# 【今日话题：FC的最终优化】







# 【小伙伴S】

## 性能现状分析

### 当前性能数据 (batch=512, in=1024, out=1000)

| 模式     | FWD耗时  | BWD耗时 | 实现方式          |
| -------- | -------- | ------- | ----------------- |
| CPU FP32 | 5,905μs  | 6,697μs | Eigen3            |
| GPU FP32 | 10,683μs | 2,693μs | 手写CUDA kernel   |
| AMP FP16 | 3,079μs  | 152μs   | cuDNN FE + cuBLAS |

### 关键问题识别

1. **GPU FP32 FWD异常慢** (10.7ms) - 比CPU还慢！
2. **AMP FP16 FWD不够快** (3.1ms) - 应该接近BWD的性能
3. **GPU FP32 BWD性能可接受** (2.7ms) - 但仍有优化空间
4. **AMP FP16 BWD性能优秀** (152μs) - 这是正确的实现方向

---

## 根本原因分析

### 1. GPU FP32 FWD性能问题 (fc_op.cu)

**当前实现**：

```cpp
// fc_fwd_fp32_kernel - 朴素矩阵乘法
for (int i = 0; i < in_features; ++i) {
    sum += x_row[i] * w_row[i];  // 每个线程独立做1024次乘加
}
```

**性能瓶颈**：

- **无共享内存优化**：每次都从global memory读取数据
- **无向量化**：没有利用CUDA的warp-level原语
- **无分块(tiling)**：数据重用效率低
- **无Tensor Core利用**：FP32应该使用Tensor Core加速

**对比AMP FP16 BWD**：

```cpp
// 使用cuBLAS GEMM：高度优化的库函数
cublasGemmEx(cublas_handle, ..., CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

### 2. AMP FP16 FWD性能问题 (fc_op.cpp)

**当前实现**：

```cpp
// 使用cuDNN FE构建计算图
g->set_io_data_type(fe::DataType_t::HALF)
 .set_compute_data_type(fe::DataType_t::FLOAT);
```

**性能瓶颈**：

- **cuDNN FE overhead**：每次执行都有图执行开销
- **没有使用Tensor Core**：虽然设置了FP16 compute，但可能没有正确启用
- **图构建复杂**：相比直接cuBLAS调用有额外抽象层

### 3. GPU FP32 BWD性能可接受但有优化空间

**当前实现**：

```cpp
// 3个独立kernel调用：db, dW, dX
launch_fc_bwd_db_kernel(...);     // 朴素reduce
launch_fc_bwd_dw_kernel(...);     // 朴素矩阵乘法
launch_fc_bwd_kernel(...);        // 朴素矩阵乘法
```

**可优化点**：

- **dW计算**：可以合并为单个GEMM
- **dX计算**：可以优化为单个GEMM
- **内存访问**：优化coalescing和缓存利用

### 4. 同步机制检查

**当前代码**：

```cpp
cudaEventRecord(state.streams[si].last_done_event, s);  // ✅ 有事件记录
```

**结论**：同步机制正确，不存在"同步缺失"导致的虚假性能数据。

---

## 优化建议方案

### 方案A：统一使用cuBLAS (推荐)

#### 1. GPU FP32 FWD → cuBLAS

**当前**：手写kernel (10.7ms)
**优化后**：cuBLAS GEMM (预期<1ms)

```cpp
// 建议：使用cublasGemmEx替代手写kernel
cublasStatus_t status = cublasGemmEx(
    cublas_handle,
    CUBLAS_OP_N, CUBLAS_OP_T,  // W需要转置：Y = X @ W^T
    batch, out_features, in_features,
    &alpha,
    x, CUDA_R_32F, x_row_stride,
    w, CUDA_R_32F, w_row_stride,
    &beta,
    y, CUDA_R_32F, y_row_stride,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);  // 启用Tensor Core
```

#### 2. GPU FP32 BWD → cuBLAS (3个GEMM)

**当前**：3个手写kernel (2.7ms)
**优化后**：3个cuBLAS调用 (预期<1ms)

```cpp
// dB: reduce sum可以使用cublasGemmEx技巧或专用reduce kernel
// dW: GEMM dW = dY^T @ X
cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
             out_features, in_features, batch, ...);

// dX: GEMM dX = dY @ W
cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
             batch, in_features, out_features, ...);
```

#### 3. AMP FP16 FWD → cuBLAS

**当前**：cuDNN FE (3.1ms)
**优化后**：cuBLAS FP16 GEMM (预期<500μs)

```cpp
// 直接使用cuBLAS FP16，避免cuDNN FE开销
cublasStatus_t status = cublasGemmEx(
    cublas_handle,
    CUBLAS_OP_N, CUBLAS_OP_T,
    batch, out_features, in_features,
    &alpha,
    x, CUDA_R_16F, x_row_stride,
    w, CUDA_R_16F, w_row_stride,
    &beta,
    y, CUDA_R_16F, y_row_stride,
    CUBLAS_COMPUTE_32F,  // 计算用FP32保证精度
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);  // 强制启用Tensor Core
```

### 方案B：混合方案 (渐进式优化)

#### 阶段1：GPU FP32优先修复

1. GPU FP32 FWD改为cuBLAS
2. GPU FP32 BWD改为cuBLAS
3. 验证性能提升

#### 阶段2：AMP FP16优化

1. AMP FP16 FWD改为cuBLAS
2. 保持AMP FP16 BWD的cuBLAS实现
3. 统一接口风格

#### 阶段3：CPU优化

1. 保持Eigen3实现(已经够好)
2. 可选：添加OpenMP并行优化

---

## 实现细节建议

### 1. cuBLAS handle管理

**当前问题**：可能存在handle创建/销毁开销
**建议**：在DeviceContext中缓存cuBLAS handle

```cpp
// 在DeviceContext中添加
class DeviceContext {
    cublasHandle_t cublas_handle_ = nullptr;
    
    cublasHandle_t cublas_handle(StreamKind kind) {
        if (!cublas_handle_) {
            cublasCreate(&cublas_handle_);
        }
        return cublas_handle_;
    }
};
```

### 2. 内存布局优化

**当前**：NHWC布局的stride处理
**建议**：确保stride参数正确传递给cuBLAS

```cpp
// 关键：正确设置stride
// X: [B,1,1,I] -> 2D view [B,I] with leading dim = x_row_stride
// W: [O,1,1,I] -> 2D view [O,I] with leading dim = w_row_stride  
// Y: [B,1,1,O] -> 2D view [B,O] with leading dim = y_row_stride
```

### 3. Tensor Core强制启用

```cpp
// 关键参数组合
CUBLAS_COMPUTE_32F                    // FP32累加精度
CUBLAS_GEMM_DEFAULT_TENSOR_OP         // 强制Tensor Core
CUBLAS_POINTER_MODE_HOST              // alpha/beta在host
```

### 4. FP16精度保障

```cpp
// 对于AMP FP16，保持FP32计算精度
CUBLAS_COMPUTE_32F                    // 内部FP32计算
CUDA_R_16F                            // I/O FP16存储
```

---

## 预期性能提升

| 模式         | 当前耗时 | 预期耗时 | 提升倍数   |
| ------------ | -------- | -------- | ---------- |
| GPU FP32 FWD | 10,683μs | <1,000μs | **10x+**   |
| GPU FP32 BWD | 2,693μs  | <800μs   | **3x+**    |
| AMP FP16 FWD | 3,079μs  | <500μs   | **6x+**    |
| AMP FP16 BWD | 152μs    | 152μs    | **已最优** |

---

## 实现优先级

### P0 (立即修复)

1. **GPU FP32 FWD改为cuBLAS** - 10x性能提升，最严重问题
2. **GPU FP32 BWD改为cuBLAS** - 3x性能提升，显著改善

### P1 (重要优化)

3. **AMP FP16 FWD改为cuBLAS** - 6x性能提升，与BWD对齐
4. **统一代码风格** - 所有GPU实现都用cuBLAS

### P2 (可选优化)

5. **添加性能基准测试** - 建立性能回归测试
6. **CPU OpenMP优化** - 进一步提升CPU性能

---

## 风险评估

### 低风险

- cuBLAS是成熟稳定的库
- 与现有AMP BWD实现一致
- 测试覆盖充分

### 需要验证

1. **stride正确性**：NHWC布局的2D view映射
2. **数值精度**：cuBLAS与手写kernel的舍入差异
3. **内存对齐**：确保满足cuBLAS的对齐要求
4. **小batch性能**：cuBLAS在小shape下可能有overhead

---

## 总结

**核心建议**：统一使用cuBLAS实现所有GPU版本的FC算子。

**关键理由**：

1. AMP FP16 BWD已经证明cuBLAS性能优异(152μs)
2. cuBLAS针对Tensor Core做了深度优化
3. 代码简化，维护性提升
4. 与现有高性能实现保持一致

**实施策略**：先修复GPU FP32(获得10x提升)，再优化AMP FP16(获得6x提升)，最终实现所有GPU版本的性能统一。



# 【小伙伴K】

> 基于 FC_OPT2.md 的测试数据与 `src/backend/ops/dtensor/fc_op.cpp` + `fc_op.cu` 源码的逐项审查。
> 结论先行：**GPU FP32 路径是当前最大瓶颈，AMP FWD 仍有调优空间；同步机制无缺失但存在冗余。**

---

## 一、性能数据回顾（batch=512, in=1024, out=1000）

| 情形                 | FWD (μs)   | BWD (μs)  | 实现方式             | 问题评级 |
| -------------------- | ---------- | --------- | -------------------- | -------- |
| CPU FP32 + bias      | 5,905      | 6,697     | Eigen3 / 朴素        | —        |
| CPU FP32 no-bias     | 1,800      | 6,534     | Eigen3 / 朴素        | —        |
| **GPU FP32 + bias**  | **10,683** | **2,693** | **手写 CUDA kernel** | **🔴 P0** |
| **GPU FP32 no-bias** | **10,513** | **2,559** | **手写 CUDA kernel** | **🔴 P0** |
| AMP FP16 + bias      | 3,079      | 152       | cuDNN FE + cuBLAS    | 🟡 P1     |
| AMP FP16 no-bias     | 3,053      | 152       | cuDNN FE + cuBLAS    | 🟡 P1     |

**核心异常**：

- GPU FP32 FWD 比 CPU 还慢 ~1.8×，比 AMP FP16 FWD 慢 ~3.5×。
- AMP FP16 BWD 极快（152 μs），但 `run_iter` 内部已调用 `cudaStreamSynchronize`，计时准确。**不存在同步缺失。**

---

## 二、源码逐项审查

### 2.1 GPU FP32 路径（🔴 P0）

#### FWD：`launch_fc_fwd_cuda` → `fc_fwd_fp32_kernel`

```cpp
// fc_op.cu 第20-50行
__global__ void fc_fwd_fp32_kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_features;
    int step  = blockDim.x * gridDim.x;
    for (; idx < total; idx += step) {
        // 每个线程串行累加 in_features=1024 个 float
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += x_row[i] * w_row[i];
        }
    }
}
```

**问题清单**：

1. **无共享内存（Shared Memory）优化**：每个线程独立读取权重行 `w_row`，导致 `batch × out_features` 次全局内存重复读取权重。对于大 shape，权重读取量膨胀到 `(512×1000)×1024×4B ≈ 2GB`，严重浪费显存带宽。
2. **无寄存器分块（Tiling）**：内层循环步长为 1，无法隐藏访存延迟。
3. **无 Tensor Core 利用**：RTX 4060 支持 FP32 Tensor Core，但手写 kernel 完全未使用。
4. **Grid 配置过于保守**：`grid_size = (batch×out_features + 255)/256`，对于 512×1000=512K 的输出，grid≈2000，block=256。这个配置没有考虑 occupancy 和 SM 利用率。
5. **未调用 cuBLAS**：`cublasSgemm` 在相同 shape 下通常可达 **~200-500 μs**，当前 10.7 ms 慢了 **20-50 倍**。

#### BWD：`launch_fc_bwd_cuda` → `fc_bwd_fp32_kernel` / `fc_bwd_dw_kernel` / `fc_bwd_db_kernel`

BWD 分三步：

1. `fc_bwd_db_fp32_kernel`：每个 output channel 一个线程，串行累加 batch 个值。out=1000 时仅发射 1000 个线程（grid=4, block=256），SM 利用率 < 1%。
2. `fc_bwd_dw_fp32_kernel`：每个 (o, i) 一对一个线程，串行累加 batch 个值。共 1000×1024=1M 线程，grid≈3906。但每个线程内层循环 512 次，计算强度低，无共享内存优化。
3. `fc_bwd_fp32_kernel`（dX）：与 FWD 类似的问题，每个 (batch, in) 一个线程，串行累加 out_features 个值。

**结论**：GPU FP32 路径的 **FWD + BWD 全部三个子操作** 都应替换为 cuBLAS `cublasSgemm` 和 `cublasSgemv`（或 `cublasSger`）。这是性能提升最大、改动最明确的优化项。

---

### 2.2 AMP FP16 路径（🟡 P1）

#### FWD：`launch_fc_amp_fwd_cuda` → cuDNN FE Matmul + Pointwise ADD

```cpp
// fc_op.cpp 第92-174行
auto g = std::make_shared<feg::Graph>();
g->set_io_data_type(fe::DataType_t::HALF)
 .set_intermediate_data_type(fe::DataType_t::FLOAT)
 .set_compute_data_type(fe::DataType_t::FLOAT);
// ... reshape, matmul, pointwise add ...
```

**问题清单**：

1. **cuDNN FE Graph 构建开销**：每次 shape 变化时（或首次运行时）需要 `validate → build_operation_graph → create_execution_plans → check_support → build_plans`。虽然已做 per-handle cache，但 cache miss 时的延迟不可忽略。
2. **3D Reshape 的复杂性**：NHWC `[B,1,1,I]` → 3D `[B,1,I]`，权重 `[O,1,1,I]` → 3D `[1,I,O]`，stride 映射为 `{w_n_stride, 1, w_n_stride}`。这种非标准 stride 可能导致 cuDNN 选择次优的 execution plan。
3. **cuDNN FE 的 workspace**：`cache->graph->get_workspace_size()` 可能需要额外显存分配。
4. **用户建议**："AMP FP16 BWD 使用的是 cuBLAS，FWD 也应该改成 cuBLAS，实现同一层级的性能。"

**量化对比**：

- AMP FWD（cuDNN FE）：~3,070 μs
- AMP BWD（cuBLAS GemmEx ×2 + 手写 db kernel）：~152 μs
- 即使 FWD 只做一次 Matmul + BiasAdd，3 ms 也显著高于 BWD 的两次 GemmEx（152 μs）。这个差距部分源于 cuDNN FE 的 orchestration 开销，部分源于 FE 选择的 kernel 效率不如 cuBLAS 直接。

**建议**：AMP FWD 改用 `cublasGemmEx` + 轻量 bias-add kernel（或 cuDNN Pointwise 单独做 ADD）。

#### BWD：`launch_fc_amp_bwd_cuda` → cublasGemmEx + 手写 db kernel

```cpp
// fc_op.cpp 第289-425行
// 1. db = reduce_sum(dY)  [手写 kernel]
// 2. dW = cublasGemmEx
// 3. event barrier
// 4. dX = cublasGemmEx
```

**问题清单**：

1. **`fc_bwd_db_amp_kernel` 效率极低**：与 FP32 dB kernel 相同的问题，out=1000 时仅 1000 个线程，grid=4。batch=512 的 reduce 在单个线程内串行完成。

   - **建议**：改用 `cublasGemvEx`（`dB = 1^T @ dY`）或 `thrust::reduce_by_key`，或至少用 warp-shuffle 并行归约。

2. **同流事件屏障冗余**：

   ```cpp
   cudaEvent_t dw_done = state.alloc_temp_event();
   cudaEventRecord(dw_done, s);
   cudaStreamWaitEvent(s, dw_done, 0);  // 在同一流 s 上等待 s 的事件 = no-op
   ```

   在同一个 CUDA stream 中，所有操作天然按 FIFO 顺序执行。`cudaStreamWaitEvent(s, dw_done, 0)` 在同一流上是 **完全冗余的 no-op**。

   - 但它在 CUDA Graph 捕获中会被记录为一个 graph node，在 replay 时同样成为 no-op，不会导致同步缺失。
   - **真正需要事件屏障的场景**：如果 dW 和 dX 被分配到 **不同 stream** 并行执行，才需要跨流同步。当前它们都在 `StreamKind::COMP_1`（即同一个 stream）上，所以此代码应删除或改为有意义的跨流同步。

---

### 2.3 CPU 路径（🟢 可接受）

#### FWD：`launch_fc_fwd_cpu_eigen`

```cpp
Y_mat.noalias() = X_mat * W_mat.transpose();
if (has_bias) Y_mat.rowwise() += b_vec;
```

Eigen3 的矩阵乘法已经高度优化，no-bias 路径（纯 `gemm`）约 1.8 ms，bias 路径（`gemm + rowwise_add`）约 5.9 ms。差距较大可能是因为 `rowwise()` 操作在 Eigen3 中引入了额外的内存遍历，或者 OpenMP 在纯 GEMM 路径上的并行效率更高。

**建议（低优先级）**：

- 检查 `rowwise() +=` 是否可以通过 `Y_mat = X_mat * W_mat.transpose(); Y_mat += b_vec.replicate(batch, 1);` 优化。
- 或者将 bias broadcast 与 GEMM 融合为一个 kernel 调用（但这需要手写 CPU kernel，与使用 Eigen3 的初衷相悖）。

#### BWD：`launch_fc_bwd_cpu_eigen`

```cpp
db_vec.noalias() = dY_mat.colwise().sum();
dW_mat.noalias() = dY_mat.transpose() * X_mat;
dX_mat.noalias() = dY_mat * W_mat;
```

Eigen3 实现正确，无显著问题。

---

### 2.4 cuBLAS Handle 管理（🟡 P1）—— 需重点重构

#### 当前实现（fc_op.cpp 第341-362行）

```cpp
static std::unordered_map<cudaStream_t, cublasHandle_t> s_cublas_handles;

static struct CublasHandleCleanup {
    ~CublasHandleCleanup() {
        for (auto& kv : s_cublas_handles) {
            if (kv.second) cublasDestroy(kv.second);
        }
        s_cublas_handles.clear();
    }
} s_cublas_cleanup;

cublasHandle_t cublas_handle = nullptr;
auto it = s_cublas_handles.find(s);
if (it == s_cublas_handles.end()) {
    cublasStatus_t cs = cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, s);
    s_cublas_handles[s] = cublas_handle;
} else {
    cublas_handle = it->second;
}
```

#### 问题深度分析

**问题 1：线程安全性缺失（Critical）**

`run_iter` 在 GPU 路径下为每个 rank 启动独立线程（`std::thread`）：

```cpp
// simple_task.h 第157-181行
for (int rank = 0; rank < K; ++rank) {
    threads.emplace_back([this, &cg, rank, sk, iterations, &exc]() {
        DeviceContext& ctx = context(rank);
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        // ... cudaGraphLaunch + cudaStreamSynchronize ...
    });
}
```

如果多个 rank 映射到同一个物理 GPU（单 GPU 多 rank 场景），它们共享同一个 `cudaStream_t`（因为 `DeviceContext` 的 stream 是按 device 创建的）。此时多个线程并发访问 `s_cublas_handles`（`std::unordered_map`），而标准库 unordered_map **不是线程安全的**。这会导致：

- `find` + `insert` 竞态条件，可能重复创建 handle
- 迭代器失效，导致段错误
- 即使当前测试是单 rank，代码层面存在隐患

**问题 2：Handle 与 Device 的隐式绑定（Critical）**

`cublasCreate` 的文档明确说明：创建的 handle 绑定到**当前 CUDA device**（调用 `cublasCreate` 时的 active device）。当前代码：

```cpp
cublasStatus_t cs = cublasCreate(&cublas_handle);  // 绑定到当前 device
cublasSetStream(cublas_handle, s);                 // 绑定到 stream s
```

在 `launch_fc_amp_bwd_cuda` 中，调用前没有显式 `cudaSetDevice`。如果 CUDA 运行时当前 device 与 `s` 所属的 device 不一致（多 GPU 场景），`cublasCreate` 会绑定到错误的 device。后续在该 handle 上执行 `cublasGemmEx` 时，可能产生 `CUBLAS_STATUS_NOT_INITIALIZED` 或静默错误。

**问题 3：生命周期与 CUDA Graph 捕获不兼容（High）**

`cublasGemmEx` 在 CUDA Graph 捕获期间的行为需要特别注意：

- cuBLAS 在 graph capture 时会记录其内部的工作描述（work descriptor），而不是立即执行
- 如果 `cublas_handle` 在 capture 之后被销毁（或 stream 被重新绑定），graph replay 时会失败
- 当前 `s_cublas_cleanup` 是 static 析构，在程序退出时销毁 handle。但如果模块被动态卸载（如 Python 扩展），handle 可能在 graph replay 之前被销毁

**问题 4：仅在 AMP BWD 中使用（Medium）**

GPU FP32 FWD/BWD 完全未使用 cuBLAS，导致性能差。而 AMP FWD 使用 cuDNN FE，也未使用 cuBLAS。这意味着项目中 cuBLAS 的利用极不充分，每个需要矩阵乘法的算子各自为政：

- AMP BWD：cuBLAS GemmEx
- AMP FWD：cuDNN FE Matmul
- GPU FP32 FWD/BWD：手写 kernel

这种碎片化导致维护困难，且无法统一调优。

**问题 5：Key 类型选择不当（Low）**

以 `cudaStream_t`（即 `CUstream_st*`）作为 unordered_map 的 key 是可行的（指针可哈希），但语义上不够清晰。stream 是 per-device 的资源，而 handle 的创建需要考虑 device + stream 的组合。如果两个 device 上有相同的 stream 地址值（不同进程或不同 CUDA context），map 会错误地复用 handle。

---

#### 推荐方案：DeviceContext 集成（与 cuDNN handle 对齐）

`DeviceContext` 已经以完全相同的方式管理了 5 个 per-stream `cudnnHandle_t`（见 `device_context.h` 第66-68行和 `device_context.cpp` 第64-79行）。cuBLAS handle 应遵循完全一致的模式。

**Step 1：device_context.h 修改**

在 `class DeviceContext` 中新增 `cublas_handles_` 数组和访问方法：

```cpp
// device_context.h（在 cudnn_handle 旁边新增）
class DeviceContext {
public:
    // ... 现有接口 ...

    [[nodiscard]] void* cudnn_handle(StreamKind kind) const noexcept {
        return cudnn_handles_[stream_index(kind)];
    }

    // 新增：per-stream cuBLAS handle
    [[nodiscard]] void* cublas_handle(StreamKind kind) const noexcept {
        return cublas_handles_[stream_index(kind)];
    }

private:
    // ... 现有成员 ...
    void* cudnn_handles_[5] = {};
    void* cublas_handles_[5] = {};  // 新增
    void* streams_[5] = {};
    // ...
};
```

**Step 2：device_context.cpp 修改**

在构造函数中，与 cuDNN handle 创建完全对称地创建 cuBLAS handle：

```cpp
// device_context.cpp DeviceContext::DeviceContext(int device_id)
for (int i = 0; i < 5; ++i) {
    // 1. 创建 stream（已有）
    err = cudaStreamCreateWithFlags(...);
    // ...

    // 2. 创建 cuDNN handle（已有）
    cudnnStatus_t cudnn_err = cudnnCreate(...);
    cudnn_err = cudnnSetStream(...);
    // ...

    // 3. 新增：创建 cuBLAS handle
    cublasStatus_t cublas_err = cublasCreate(
        reinterpret_cast<cublasHandle_t*>(&cublas_handles_[i]));
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        // 清理已创建的 cuBLAS handles（反向遍历）
        for (int j = i - 1; j >= 0; --j) {
            if (cublas_handles_[j]) {
                cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_handles_[j]));
                cublas_handles_[j] = nullptr;
            }
        }
        // 同时清理 cuDNN handles 和 streams（保持与现有错误处理一致）
        // ...
        TR_DEVICE_ERROR("cublasCreate failed for stream " << i
                        << " on device " << device_id_
                        << ": " << cublas_err);
    }

    cublas_err = cublasSetStream(
        reinterpret_cast<cublasHandle_t>(cublas_handles_[i]),
        reinterpret_cast<cudaStream_t>(streams_[i]));
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        // 同样的清理逻辑 ...
        TR_DEVICE_ERROR("cublasSetStream failed for stream " << i
                        << " on device " << device_id_
                        << ": " << cublas_err);
    }
}

TR_LOG_INFO("backend") << "DeviceContext " << device_id_
                       << ": created 5 non-blocking streams + per-stream cuDNN + cuBLAS handles";
```

在析构函数中对称销毁：

```cpp
// device_context.cpp DeviceContext::~DeviceContext()
for (int i = 0; i < 5; ++i) {
    if (cublas_handles_[i]) {
        cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_handles_[i]));
        cublas_handles_[i] = nullptr;
    }
    if (cudnn_handles_[i]) {
        cudnnDestroy(reinterpret_cast<cudnnHandle_t>(cudnn_handles_[i]));
        cudnn_handles_[i] = nullptr;
    }
}
```

**Step 3：fc_op.cpp 中删除全局 static map，改用 ctx.cublas_handle()**

```cpp
// 删除以下全部代码（fc_op.cpp 第341-362行）
// static std::unordered_map<cudaStream_t, cublasHandle_t> s_cublas_handles;
// static struct CublasHandleCleanup { ... } s_cublas_cleanup;
// auto it = s_cublas_handles.find(s);
// ...

// 改为：
cublasHandle_t cublas_handle = static_cast<cublasHandle_t>(
    ctx.cublas_handle(StreamKind::COMP_1));
```

**Step 4：GPU FP32 路径也接入 cuBLAS**

`launch_fc_fwd_cuda` 和 `launch_fc_bwd_cuda` 中同样通过 `ctx.cublas_handle()` 获取 handle，调用 `cublasSgemm` / `cublasSgemv`。这样所有 4 个 FC GPU 路径（FP32 FWD/BWD、AMP FWD/BWD）统一使用 DeviceContext 管理的 cuBLAS handle。

---

#### 方案优势分析

| 维度            | 全局 static map 方案             | DeviceContext 集成方案                       |
| --------------- | -------------------------------- | -------------------------------------------- |
| **线程安全**    | ❌ 无锁保护，多 rank 竞态         | ✅ DeviceContext 实例 per-rank，天然隔离      |
| **Device 绑定** | ❌ 隐式绑定，易错                 | ✅ 构造函数内显式 `cudaSetDevice` 后创建      |
| **生命周期**    | ❌ static 析构时机不可控          | ✅ 与 DeviceContext 同生命周期，RAII          |
| **CUDA Graph**  | ⚠️ handle 存活需保证到 graph 销毁 | ✅ handle 存活期覆盖所有 graph 生命周期       |
| **多流支持**    | ⚠️ per-stream map 需要额外 key    | ✅ 原生 5 个 per-stream handle，与 cuDNN 对齐 |
| **代码一致性**  | ❌ 仅 AMP BWD 使用，碎片化        | ✅ 所有路径统一从 ctx 获取                    |
| **错误处理**    | ❌ 创建失败仅打印错误             | ✅ 构造时失败可清理已创建资源并抛异常         |

---

#### 边界情况与兼容性

1. **CPU 模式**：`device_context.h` 中 `cublas_handle()` 返回 `void*`。在 CPU 模式下（`device_id_ < 0`），`cublas_handles_` 数组保持全零，GPU 算子不会被调用到（`#ifdef TR_USE_CUDA` 保护）。无需额外处理。

2. **CUDA Graph 捕获期间的 handle 使用**：cuBLAS 在 graph capture 时，其内部 kernel launch 会被记录到 graph 中。由于 `cublas_handle` 和 `stream` 的绑定关系在 DeviceContext 构造时就固定了，且 handle 的生命周期与 DeviceContext 相同（长于任何 CUDA Graph），graph replay 时不会遇到 handle 失效问题。

3. **cuBLAS 版本兼容性**：`cublasCreate` / `cublasDestroy` / `cublasSetStream` 是 cuBLAS API 的基础函数，从 CUDA 4.0 起就存在，与当前 CUDA 13.1 完全兼容。

4. **与现有 cuDNN handle 的共存**：cuBLAS 和 cuDNN handle 互不干扰。AMP FWD 当前使用 cuDNN FE（需要 cuDNN handle），AMP BWD 使用 cuBLAS。即使未来 AMP FWD 也改用 cuBLAS，两个库在 same stream 上也可以安全交错执行。

---

#### 迁移步骤建议

```
Step A（零风险准备）：
  1. 在 device_context.h/.cpp 中新增 cublas_handles_[5] 和 cublas_handle() 方法
  2. 编译验证 DeviceContext 构造/析构无泄漏
  3. 不修改任何算子代码

Step B（替换 AMP BWD）：
  1. 删除 fc_op.cpp 中的 s_cublas_handles 全局 map 和 CublasHandleCleanup
  2. launch_fc_amp_bwd_cuda 中改用 ctx.cublas_handle(StreamKind::COMP_1)
  3. 运行 test_fc_fwd_bwd --amp，验证功能与性能不变

Step C（扩展 GPU FP32）：
  1. launch_fc_fwd_cuda 中调用 cublasSgemm 替代 launch_fc_fwd_kernel
  2. launch_fc_bwd_cuda 中调用 cublasSgemm/Sgemv 替代手写 kernels
  3. 运行 test_fc_fwd_bwd --gpu，验证精度和性能提升

Step D（可选：AMP FWD 统一）：
  1. launch_fc_amp_fwd_cuda 中从 cuDNN FE 改为 cublasGemmEx + bias kernel
  2. 验证精度与性能
```

**回滚策略**：如果某步出现问题，只需回滚对应算子的 `.cpp` 文件。DeviceContext 的修改是增量添加，不会影响其他算子。

---

## 三、优化建议汇总（按优先级排序）

### 🔴 P0：GPU FP32 路径全面替换为 cuBLAS

| 操作                  | 当前实现    | 建议替换                        | 预期性能         |
| --------------------- | ----------- | ------------------------------- | ---------------- |
| FWD (Y = X @ W^T + b) | 手写 kernel | `cublasSgemm` + bias add kernel | ~500 μs（↓ 20×） |
| BWD dB                | 手写 kernel | `cublasSgemv` 或 `cublasSger`   | ~50 μs           |
| BWD dW                | 手写 kernel | `cublasSgemm`                   | ~500 μs          |
| BWD dX                | 手写 kernel | `cublasSgemm`                   | ~500 μs          |

**具体方案**：

- FWD：`cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, batch, in_features, &alpha, W, in_features, X, in_features, &beta, Y, out_features)`，然后调用轻量 bias-add kernel（或 `cublasSgeam`）。
- BWD dW：`cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, batch, &alpha, X, in_features, dY, out_features, &beta, dW, in_features)`。
- BWD dX：`cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, batch, out_features, &alpha, W, in_features, dY, out_features, &beta, dX, in_features)`。
- BWD dB：`cublasSgemv(handle, CUBLAS_OP_T, batch, out_features, &alpha, dY, out_features, ones, 1, &beta, dB, 1)`，其中 `ones` 是全 1 向量。

### 🟡 P1：AMP FWD 改用 cuBLAS GemmEx

- 用 `cublasGemmEx` 替代 cuDNN FE Matmul，消除 FE Graph 构建和 3D reshape 的开销。
- Bias add 可保留 cuDNN Pointwise 或手写轻量 kernel。
- 预期性能：从 ~3,070 μs 降至 ~500-800 μs。

### 🟡 P1：AMP BWD dB 改用 cuBLAS/cuDNN Reduce

- 当前 `fc_bwd_db_amp_kernel` 在 out=1000 时仅利用 1000 个线程，SM 利用率 < 1%。
- 建议：
  - 方案 A：`cublasGemvEx` 用全 1 向量做矩阵-向量乘法。
  - 方案 B：手写 warp-shuffle + shared memory 并行归约 kernel，每个 output channel 用一个 warp（32 线程）并行 reduce batch 维度。
  - 方案 C：调用 cuDNN ReduceTensor（如果项目已依赖 cuDNN）。

### 🟡 P1：cuBLAS Handle 管理重构（详见 §2.4）

**目标**：将全局 static `std::unordered_map<cudaStream_t, cublasHandle_t>` 替换为 `DeviceContext` 内 per-stream 数组 `cublas_handles_[5]`，与现有 `cudnn_handles_[5]` 对齐。

**关键修改点**：

- `device_context.h`：新增 `cublas_handle(StreamKind kind)` 方法
- `device_context.cpp`：构造函数内对称创建/绑定 5 个 cuBLAS handle；析构函数内对称销毁
- `fc_op.cpp`：删除 `s_cublas_handles` 全局 map 和 `CublasHandleCleanup`；所有 4 个 GPU 路径统一从 `ctx.cublas_handle()` 获取

**预期收益**：

- 消除多线程竞态和 device 绑定隐患
- 为 GPU FP32 路径接入 cuBLAS 扫清基础设施障碍
- 所有 FC 算子使用统一的 handle 来源，维护性提升

### 🟢 P2：删除冗余的同流事件屏障

```cpp
// fc_op.cpp 第399-401行（AMP BWD）和第544-546行（FP32 BWD）
cudaEvent_t dw_done = state.alloc_temp_event();
cudaEventRecord(dw_done, s);
cudaStreamWaitEvent(s, dw_done, 0);  // 冗余：同一流天然顺序
```

- **删除理由**：在同一个 `cudaStream_t` 上，`cudaStreamWaitEvent(s, ev, 0)` 其中 `ev` 也是在该流上记录的，等于 no-op。
- **保留条件**：如果未来要将 dW 和 dX 分配到不同 stream 并行计算，才需要保留跨流同步。当前代码应在注释中明确说明此 barrier 的意图，或暂时删除以简化 CUDA Graph。
- **注意**：`alloc_temp_event()` 创建的事件在 `cleanup_all_events()` 中批量销毁。每次捕获都创建新事件，虽不泄漏，但增加了不必要的 CUDA Graph node。

### 🟢 P2：CPU Bias-Add 路径优化（低优先级）

- 调查 `Y_mat.rowwise() += b_vec` 为何导致 FWD 从 1.8 ms 升至 5.9 ms（~3.3×）。
- 可能的修复：将 bias 预 broadcast 为 `[batch, out_features]` 矩阵再一次性 add，或检查 Eigen3 的 `rowwise()` 是否未向量化。

---

## 四、关于"同步缺失"的专项澄清

FC_OPT2.md 中用户提出：

> "检查算子执行完运算后是否有同步计算流。如果同步缺失，就会出现耗时极短的情况！"

经审查 `include/renaissance/task/simple_task.h` 第172-176行：

```cpp
for (int i = 0; i < iterations; ++i) {
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);  // ✅ 每次迭代都同步
}
```

**结论**：`run_iter` 在 GPU 路径下**每次迭代都调用了 `cudaStreamSynchronize`**，计时结果反映的是完整的 GPU 执行时间（包含 kernel launch、执行、显存回写）。

**AMP FP16 BWD 的 152 μs 是真实值**，原因不是"同步缺失"，而是：

1. cublasGemmEx 调用的是高度优化的 cuBLAS kernel，利用了 Tensor Core。
2. 两个 GemmEx 的 MNK 尺寸（1024×1000×512 和 1024×512×1000）恰好落在 cuBLAS 的 sweet spot。
3. 当前 dB kernel 虽然低效，但执行时间极短（< 20 μs），对总时间影响小。

---

## 五、实施路线图建议

```
Phase 1（最高收益）:
  └─ GPU FP32 FWD/BWD 全部替换为 cuBLAS Sgemm
  └─ 预期：FWD 从 10.7ms → ~500μs，BWD 从 2.7ms → ~600μs

Phase 2（统一架构）:
  └─ AMP FWD 从 cuDNN FE 改为 cuBLAS GemmEx + bias kernel
  └─ AMP BWD dB 从手写 kernel 改为 cuBLAS GemvEx 或 warp-reduce
  └─ cuBLAS handle 移入 DeviceContext

Phase 3（细节打磨）:
  └─ 删除冗余的同流 event barrier
  └─ 调查 CPU bias-add 性能 anomaly
```

---

*报告生成时间：2026-05-18*
*审查范围：fc_op.cpp (799 lines), fc_op.cu (212 lines), simple_task.h (413 lines), capture_multi_stream.cpp (91 lines)*



# 【小伙伴D】

## 1. 现状概览

### 当前各路径实现与耗时（shape: batch=512, in=1024, out=1000）

| 路径     | 算力提供          | FWD (μs)   | BWD (μs)  | 方法                           |
| -------- | ----------------- | ---------- | --------- | ------------------------------ |
| CPU FP32 | Eigen3            | 2,077      | 8,024     | GEMM（已优化）                 |
| GPU FP32 | 手写 kernel       | **10,683** | **2,693** | 朴素逐元素 CUDA kernel         |
| AMP FP16 | cuDNN FE + cuBLAS | **3,079**  | **152**   | cuDNN FE Matmul / cuBLAS HGEMM |

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

| 路径         | BWD 耗时 | 同步状态 | 归因                                       |
| ------------ | -------- | -------- | ------------------------------------------ |
| AMP BWD      | 152 μs   | ✅        | cuBLAS Tensor Core HGEMM，极快（正常）     |
| GPU FP32 BWD | 2,693 μs | ✅        | 手写 kernel，无 tiling，慢但无缺失（正常） |

**无同步缺失。** AMP BWD 的 152 μs 是 cuBLAS FP16 的正常性能水平。

## 3. 优化方案

### 3.1 总体策略

**将所有 FP32/AMP 的矩阵运算统一迁移到 cuBLAS，与 AMP BWD 对齐。**

| 变更路径             | 当前方法           | 改后方法       | 预期加速                   |
| -------------------- | ------------------ | -------------- | -------------------------- |
| **GPU FP32 FWD**     | 手写 kernel        | cuBLAS SGEMM   | **~50x** (10.7ms → ~200μs) |
| **GPU FP32 BWD dW**  | 手写 kernel        | cuBLAS SGEMM   | **~18x** (含在 2.7ms 内)   |
| **GPU FP32 BWD dX**  | 手写 kernel        | cuBLAS SGEMM   | **~18x** (含在 2.7ms 内)   |
| **GPU FP32 BWD db**  | 手写 kernel (keep) | 保留（非瓶颈） | —                          |
| **AMP FWD**          | cuDNN FE Matmul    | cuBLAS HGEMM   | **~20x** (3.1ms → ~150μs)  |
| **AMP BWD dX/dW/db** | cuBLAS (keep)      | 保留           | —                          |

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

| 优先级 | 文件                                               | 变更                                                         | 影响               |
| ------ | -------------------------------------------------- | ------------------------------------------------------------ | ------------------ |
| 🔴 P0   | `fc_op.cpp` L178-L281                              | 重写 `launch_fc_amp_fwd_cuda`：cuDNN FE → cuBLAS + bias kernel | AMP FWD 性能 ~20x  |
| 🔴 P0   | `fc_op.cpp` L427-L472                              | 重写 `launch_fc_fwd_cuda`：手写 kernel → cuBLAS + bias kernel | FP32 FWD 性能 ~50x |
| 🔴 P0   | `fc_op.cpp` L524-L541                              | BWD dW/dX：手写 kernel → cuBLAS（参考 AMP BWD 模式）         | FP32 BWD 性能 ~18x |
| 🟡 P1   | `fc_op.cpp` L78-L174                               | 删除 `FcAmpFwdCache` + `build_fc_amp_fwd_graph`              | 减少 ~100 行代码   |
| 🟡 P1   | `fc_op.cu` L20-L77, L95-L116, L134-L164, L180-L194 | 删除 FP32 手写 kernel                                        | 减少 ~120 行代码   |
| 🟡 P1   | `fc_op.cu`                                         | 新增 `fc_fwd_bias_add_fp32_kernel` + `fc_fwd_bias_add_amp_kernel` | 新增 ~30 行        |
| 🟢 P2   | `fc_op.cpp` L341-L362                              | cuBLAS handle 提升为共享（避免 3 个 launch 各维护一份）      | 代码整洁           |

## 6. 预期结果

| 路径     | 当前 FWD  | 优化后 FWD  | 当前 BWD | 优化后 BWD     |
| -------- | --------- | ----------- | -------- | -------------- |
| GPU FP32 | 10,683 μs | **~200 μs** | 2,693 μs | **~350 μs**    |
| AMP FP16 | 3,079 μs  | **~150 μs** | 152 μs   | 152 μs（不变） |

优化后所有 GPU 路径均统一使用 cuBLAS，代码量减少 ~200 行，FP32 BWD 仍保留完整的事件同步（db → dW → barrier → dX → last_done_event）。





# 【用户补充】

脱离cuDNN或cuBLAS的手写kernel性能不好，这个显然需要改进。刚才小伙伴K提出的cuBLAS handle放到DeviceContext的思路，大家重点分析，探讨科学性可行性和具体的正确做法。我们最终是要实现性能最优的ResNet-50的单机多卡AMP训练，并且我们一定会捕获CUDA Graph，所以如何在架构上确保合理，非常重要。

另外关于具体如何用cuBLAS/cuDNN技术路线来实现性能优化，大家一定要认真给出方案。
