/**
 * @file softmax_ce_op.cu
 * @brief SOFTMAX_CE 融合算子（Softmax + CrossEntropy）CUDA kernel 实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: cuda_runtime.h, cuda_fp16.h
 * @note 所属系列: backend/ops/dtensor
 * @note 包含 FP32 和 AMP FP16 两个变体，FWD/INF/BWD + reduce 四 kernel 完全独立
 * @note FWD: 仅计算 loss + probs + inv_scaling（训练用，轻量）
 * @note INF: 保留完整 top1/top5/pred + loss + probs + inv_scaling（推理用）
 * @note 核心修正：全局 reduce + DTensor-only（kernel 内读 scaling/写 inv_scaling）
 * @note V2.2.0: 用 const void* / void* 替代 float* 传 logits/grad，消除 strict aliasing violation
 * @note V2.3.0: FWD 与 INF 拆分为独立 kernel，FWD 移除 top1/top5/pred 计算
 * @note V3.0.0: 移除所有 atomicAdd，改用 per-sample partial + 确定性单 block reduce kernel
 *               消除跨 block 浮点累加顺序不确定性；INF 的 top1/top5 用 INT32 partial 累加
 */

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>



namespace tr {

#define WARP_SIZE  32
#define BLOCK_DIM  256
#define WARP_COUNT 8

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (other > val) val = other;
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Softmax + CrossEntropy FWD kernel
 *
 * 每个 block 处理一个 sample。写入 loss_partial[b] 替代 atomicAdd。
 * top1/top5/pred 参数保留用于接口对齐，FWD 不操作这些张量。
 * @param logits_stride  样本间 stride（logits 张量；FEATURE_FP16 可能有对齐 padding）
 * @param probs_stride   样本间 stride（probs 张量，FP32 紧凑）
 *
 * 共享内存布局：
 *   [0..WARP_COUNT-1]                : warp max buffer
 *   [WARP_COUNT..2*WARP_COUNT-1]     : warp sum buffer
 */
template <bool IS_AMP>
__global__ void softmax_ce_fwd_kernel(
    const void*  __restrict__ logits_ptr,
    const int*   __restrict__ labels,
    float* __restrict__ loss,              // 保留参数，不再写入
    float* __restrict__ top1,              // 保留参数，接口对齐
    float* __restrict__ top5,              // 保留参数，接口对齐
    int*   __restrict__ pred,              // 保留参数，接口对齐
    float* __restrict__ probs,             // ids_out[3]
    float* __restrict__ inv_scaling,       // ids_out[1]
    const float* __restrict__ scaling_ptr,
    const int32_t* __restrict__ batch_size_ptr,
    const float* __restrict__ label_smoothing_ptr,
    float* __restrict__ loss_partial,      // [NEW] ids_out[6]
    int*   __restrict__ top1_partial,      // [NEW] ids_out[7] — FWD 不操作
    int*   __restrict__ top5_partial,      // [NEW] ids_out[8] — FWD 不操作
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    // FWD 不计算 top1/top5/pred，保留参数仅用于接口对齐
    (void)loss; (void)top1; (void)top5; (void)pred;
    (void)scaling_ptr; (void)batch_size_ptr;
    (void)top1_partial; (void)top5_partial;

    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_sum = smem + WARP_COUNT;

    int b = blockIdx.x;
    if (b >= batch) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int label_b = static_cast<int>(labels[b]);
    float inv_batch = 1.0f / static_cast<float>(batch);

    // ===== Step 1: Global max =====
    float local_max = -INFINITY;
    for (int c = tid; c < num_classes; c += BLOCK_DIM) {
        float val;
        if constexpr (IS_AMP) {
            val = __half2float(static_cast<const __half*>(logits_ptr)[b * logits_stride + c]);
        } else {
            val = static_cast<const float*>(logits_ptr)[b * logits_stride + c];
        }
        if (val > local_max) local_max = val;
    }
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) s_max[warp_id] = local_max;
    __syncthreads();

    float max_val = (tid < WARP_COUNT) ? s_max[tid] : -INFINITY;
    max_val = warp_reduce_max(max_val);
    if (lane_id == 0 && warp_id == 0) s_max[0] = max_val;
    __syncthreads();
    max_val = s_max[0];

    // ===== Step 2: Exp + global sum =====
    float local_sum = 0.0f;
    for (int c = tid; c < num_classes; c += BLOCK_DIM) {
        float val;
        if constexpr (IS_AMP) {
            val = __half2float(static_cast<const __half*>(logits_ptr)[b * logits_stride + c]);
        } else {
            val = static_cast<const float*>(logits_ptr)[b * logits_stride + c];
        }
        float e = expf(val - max_val);
        local_sum += e;
        probs[b * probs_stride + c] = e;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) s_sum[warp_id] = local_sum;
    __syncthreads();

    float sum = (tid < WARP_COUNT) ? s_sum[tid] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (lane_id == 0 && warp_id == 0) s_sum[0] = sum;
    __syncthreads();
    sum = s_sum[0];
    float inv_sum = 1.0f / (sum + 1e-8f);

    // ===== Step 3: Normalize probs + accumulate sum_log_p =====
    float ls = label_smoothing_ptr[0];
    float one_minus_ls = 1.0f - ls;
    float ls_over_K = ls / static_cast<float>(num_classes);

    float local_sum_log = 0.0f;
    for (int c = tid; c < num_classes; c += BLOCK_DIM) {
        float prob = probs[b * probs_stride + c] * inv_sum;
        probs[b * probs_stride + c] = prob;
        local_sum_log += logf(prob + 1e-8f);
    }

    // 复用 s_sum 做 warp reduce（Step 2 后 s_sum 空闲）
    local_sum_log = warp_reduce_sum(local_sum_log);
    if (lane_id == 0) s_sum[warp_id] = local_sum_log;
    __syncthreads();

    float sum_log = (tid < WARP_COUNT) ? s_sum[tid] : 0.0f;
    sum_log = warp_reduce_sum(sum_log);
    if (lane_id == 0 && warp_id == 0) s_sum[0] = sum_log;
    __syncthreads();
    sum_log = s_sum[0];

    // ===== Step 4: Thread 0 writes loss to partial buffer (no atomicAdd) =====
    if (tid == 0) {
        float prob_y = probs[b * probs_stride + label_b];
        float sample_loss = -one_minus_ls * logf(prob_y + 1e-8f)
                            - ls_over_K * sum_log;
        loss_partial[b] = sample_loss * inv_batch;   // 直接写入，无竞争
    }

    if (b == 0 && tid == 0) {
        *inv_scaling = inv_batch;
    }
}

/**
 * @brief Softmax + CrossEntropy INF kernel（推理版 — 含 top1/top5/pred）
 *
 * 每个 block 处理一个 sample。写入 partial 缓冲区替代 atomicAdd。
 * top1/top5 用 INT32 partial 累计，由 reduce kernel 确定性归约。
 * @param logits_stride  样本间 stride（logits 张量）
 * @param probs_stride   样本间 stride（probs 张量）
 *
 * 共享内存布局：
 *   [0..WARP_COUNT-1]                : warp max buffer
 *   [WARP_COUNT..2*WARP_COUNT-1]     : warp sum buffer
 *   [2*WARP_COUNT..2*WARP_COUNT+BLOCK_DIM-1]           : top1 val buffer
 *   [2*WARP_COUNT+BLOCK_DIM..2*WARP_COUNT+2*BLOCK_DIM-1] (int) : top1 cls buffer
 */
template <bool IS_AMP>
__global__ void softmax_ce_inf_kernel(
    const void*  __restrict__ logits_ptr,
    const int*   __restrict__ labels,
    float* __restrict__ loss,              // 保留参数，不再写入
    float* __restrict__ top1,              // 保留参数，不再写入
    float* __restrict__ top5,              // 保留参数，不再写入
    int*   __restrict__ pred,              // ids_out[2]
    float* __restrict__ probs,             // ids_out[3]
    float* __restrict__ inv_scaling,       // ids_out[1]
    const float* __restrict__ scaling_ptr,
    const int32_t* __restrict__ batch_size_ptr,
    const float* __restrict__ label_smoothing_ptr,
    float* __restrict__ loss_partial,      // [NEW] ids_out[6]
    int*   __restrict__ top1_partial,      // [NEW] ids_out[7]
    int*   __restrict__ top5_partial,      // [NEW] ids_out[8]
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    (void)loss; (void)top1; (void)top5;
    (void)scaling_ptr; (void)batch_size_ptr;

    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_sum = smem + WARP_COUNT;
    float* s_top1_val = smem + 2 * WARP_COUNT;
    int*   s_top1_cls = reinterpret_cast<int*>(smem + 2 * WARP_COUNT + BLOCK_DIM);

    int b = blockIdx.x;
    if (b >= batch) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int label_b = static_cast<int>(labels[b]);
    float inv_batch = 1.0f / static_cast<float>(batch);

    float local_max = -INFINITY;
    for (int c = tid; c < num_classes; c += BLOCK_DIM) {
        float val;
        if constexpr (IS_AMP) {
            val = __half2float(static_cast<const __half*>(logits_ptr)[b * logits_stride + c]);
        } else {
            val = static_cast<const float*>(logits_ptr)[b * logits_stride + c];
        }
        if (val > local_max) local_max = val;
    }
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) s_max[warp_id] = local_max;
    __syncthreads();

    float max_val = (tid < WARP_COUNT) ? s_max[tid] : -INFINITY;
    max_val = warp_reduce_max(max_val);
    if (lane_id == 0 && warp_id == 0) s_max[0] = max_val;
    __syncthreads();
    max_val = s_max[0];

    float local_sum = 0.0f;
    for (int c = tid; c < num_classes; c += BLOCK_DIM) {
        float val;
        if constexpr (IS_AMP) {
            val = __half2float(static_cast<const __half*>(logits_ptr)[b * logits_stride + c]);
        } else {
            val = static_cast<const float*>(logits_ptr)[b * logits_stride + c];
        }
        float e = expf(val - max_val);
        local_sum += e;
        probs[b * probs_stride + c] = e;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) s_sum[warp_id] = local_sum;
    __syncthreads();

    float sum = (tid < WARP_COUNT) ? s_sum[tid] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (lane_id == 0 && warp_id == 0) s_sum[0] = sum;
    __syncthreads();
    sum = s_sum[0];
    float inv_sum = 1.0f / (sum + 1e-8f);

    float local_top1_val = -INFINITY;
    int local_top1_cls = -1;

    // ===== Step 3: Normalize + sum_log + top1 收集 =====
    float local_sum_log = 0.0f;
    for (int c = tid; c < num_classes; c += BLOCK_DIM) {
        float prob = probs[b * probs_stride + c] * inv_sum;
        probs[b * probs_stride + c] = prob;
        local_sum_log += logf(prob + 1e-8f);

        float raw;
        if constexpr (IS_AMP) {
            raw = __half2float(static_cast<const __half*>(logits_ptr)[b * logits_stride + c]);
        } else {
            raw = static_cast<const float*>(logits_ptr)[b * logits_stride + c];
        }
        if (raw > local_top1_val) {
            local_top1_val = raw;
            local_top1_cls = c;
        }
    }

    // Warp reduce sum_log（复用 Step 2 后空闲的 s_sum）
    local_sum_log = warp_reduce_sum(local_sum_log);
    if (lane_id == 0) s_sum[warp_id] = local_sum_log;

    s_top1_val[tid] = local_top1_val;
    s_top1_cls[tid] = local_top1_cls;
    __syncthreads();

    // Block reduce sum_log
    float sum_log = (tid < WARP_COUNT) ? s_sum[tid] : 0.0f;
    sum_log = warp_reduce_sum(sum_log);
    if (lane_id == 0 && warp_id == 0) s_sum[0] = sum_log;
    __syncthreads();
    sum_log = s_sum[0];

    // ===== Step 4: Write partial buffers (no atomicAdd) =====
    if (tid == 0) {
        float ls = label_smoothing_ptr[0];
        float prob_y = probs[b * probs_stride + label_b];
        float sample_loss = -(1.0f - ls) * logf(prob_y + 1e-8f)
                            - (ls / static_cast<float>(num_classes)) * sum_log;
        loss_partial[b] = sample_loss * inv_batch;           // 直接写入，无竞争

        int global_top1_cls = s_top1_cls[0];
        float global_top1_val = s_top1_val[0];
        for (int i = 1; i < BLOCK_DIM; ++i) {
            if (s_top1_val[i] > global_top1_val) {
                global_top1_val = s_top1_val[i];
                global_top1_cls = s_top1_cls[i];
            }
        }
        pred[b] = global_top1_cls;
        top1_partial[b] = (global_top1_cls == label_b) ? 1 : 0;  // INT32

        if (num_classes >= 5) {
            bool in_top5 = false;
            for (int k = 0; k < 5; ++k) {
                int best_j = -1;
                float best_prob = -1.0f;
                for (int j = 0; j < num_classes; ++j) {
                    float p = probs[b * probs_stride + j];
                    if (p > best_prob) {
                        best_prob = p;
                        best_j = j;
                    }
                }
                if (best_j == label_b) { in_top5 = true; break; }
                probs[b * probs_stride + best_j] = -best_prob;
            }
            for (int j = 0; j < num_classes; ++j) {
                float p = probs[b * probs_stride + j];
                if (p < 0.0f) probs[b * probs_stride + j] = -p;
            }
            top5_partial[b] = in_top5 ? 1 : 0;
        } else {
            top5_partial[b] = top1_partial[b];
        }
    }

    if (b == 0 && tid == 0) {
        *inv_scaling = 1.0f / static_cast<float>(batch);
    }
}

/**
 * @brief 统一确定性归约 kernel（FWD/INF 共用）
 *
 * 单 block 启动，按固定顺序累加 partial 缓冲区到标量输出。
 * - 始终归约 loss_partial → loss
 * - compute_top_metrics=true 时额外归约 top1_partial → top1, top5_partial → top5
 *   compute_top_metrics=false 时不操作 top1/top5（R_RESULT 区由 ZERO_GRAD 图保证清零）
 *
 * 确定性保证：
 *   1. 每个线程按固定 stride (tid, tid+BLOCK_DIM, ...) 读取，顺序唯一
 *   2. shared memory tree reduction 固定配对模式
 *   3. top1/top5 用 INT32 累加，整数加法满足结合律
 *   4. 最终只做一次 float(sum) * inv_batch，消除浮点累加误差
 */
__global__ void softmax_ce_reduce_kernel(
    const float* __restrict__ loss_partial,
    const int*   __restrict__ top1_partial,
    const int*   __restrict__ top5_partial,
    float* __restrict__ loss,
    float* __restrict__ top1,
    float* __restrict__ top5,
    int batch,
    bool compute_top_metrics)
{
    extern __shared__ char smem_raw[];
    float* s_loss = reinterpret_cast<float*>(smem_raw);
    int*   s_top1 = reinterpret_cast<int*>(smem_raw + BLOCK_DIM * sizeof(float));
    int*   s_top5 = reinterpret_cast<int*>(smem_raw + BLOCK_DIM * (sizeof(float) + sizeof(int)));

    int tid = threadIdx.x;
    float loss_sum = 0.0f;
    int top1_sum = 0, top5_sum = 0;

    // 步长累加: 每个线程按固定间隔读取，保证确定性
    for (int i = tid; i < batch; i += blockDim.x) {
        loss_sum += loss_partial[i];
        if (compute_top_metrics) {
            top1_sum += top1_partial[i];
            top5_sum += top5_partial[i];
        }
    }

    s_loss[tid] = loss_sum;
    if (compute_top_metrics) {
        s_top1[tid] = top1_sum;
        s_top5[tid] = top5_sum;
    }
    __syncthreads();

    // Tree reduction in shared memory (固定配对模式)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_loss[tid] += s_loss[tid + s];
            if (compute_top_metrics) {
                s_top1[tid] += s_top1[tid + s];
                s_top5[tid] += s_top5[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *loss = s_loss[0];
        if (compute_top_metrics) {
            float inv_batch = 1.0f / static_cast<float>(batch);
            *top1 = static_cast<float>(s_top1[0]) * inv_batch;   // INT32 → FP32
            *top5 = static_cast<float>(s_top5[0]) * inv_batch;   // INT32 → FP32
        }
        // FWD 路径 (compute_top_metrics=false): 不写 top1/top5
        // R_RESULT 区由 ZERO_GRAD 图在每个 batch 之初清零，无需冗余写 0
    }
}

/**
 * @brief Softmax + CrossEntropy BWD kernel
 *
 * 1D grid，每个线程处理一个 (batch, class) 对。
 * @param probs_stride  样本间 stride（probs 张量）
 * @param grad_stride   样本间 stride（grad 张量，AMP 下可能与 probs 不同）
 */
template <bool IS_AMP>
__global__ void softmax_ce_bwd_kernel(
    const float* __restrict__ probs,
    const int*   __restrict__ labels,
    const float* __restrict__ scaling_ptr,
    const float* __restrict__ inv_scaling_ptr,
    const float* __restrict__ label_smoothing_ptr,   // [NEW]
    void*  __restrict__ grad_ptr,
    int batch, int probs_stride, int grad_stride, int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_classes;
    if (idx >= total) return;

    int b = idx / num_classes;
    int c = idx % num_classes;

    float scale = (*scaling_ptr) * (*inv_scaling_ptr);
    float prob = probs[b * probs_stride + c];
    int label = static_cast<int>(labels[b]);
    float ls = label_smoothing_ptr[0];
    float g = prob - ls / static_cast<float>(num_classes);
    if (c == label) g -= (1.0f - ls);
    g *= scale;

    if constexpr (IS_AMP) {
        static_cast<__half*>(grad_ptr)[b * grad_stride + c] = __float2half(g);
    } else {
        static_cast<float*>(grad_ptr)[b * grad_stride + c] = g;
    }
}

cudaError_t launch_softmax_ce_fwd_fp32(
    cudaStream_t s,
    const float* logits, const int* labels,
    float* loss, float* top1, float* top5,    // 保留参数，kernel 不再直接写入
    int* pred, float* probs,                   // 保留参数，kernel 不再写入
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial,                       // [NEW] ids_out[6]
    int* top1_partial,                         // [NEW] ids_out[7]
    int* top5_partial,                         // [NEW] ids_out[8]
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    // Step 1: FWD main kernel — 每个 block 写入 loss_partial[b]
    //         top1/top5/pred 参数传入但不操作（接口对齐）
    size_t smem = static_cast<size_t>(WARP_COUNT) * 2 * sizeof(float);
    softmax_ce_fwd_kernel<false><<<batch, BLOCK_DIM, smem, s>>>(
        static_cast<const void*>(logits), labels,
        loss, top1, top5, pred, probs,
        inv_scaling, scaling, batch_size, label_smoothing,
        loss_partial, top1_partial, top5_partial,
        batch, logits_stride, probs_stride, num_classes);

    // Step 2: 确定性归约 — 单 block 累加 loss_partial → loss
    //         FWD 不计算 top metrics; top1/top5 由 R_RESULT 区 ZERO_GRAD 保证为 0
    size_t reduce_smem = BLOCK_DIM * sizeof(float)
                       + BLOCK_DIM * sizeof(int) * 2;
    softmax_ce_reduce_kernel<<<1, BLOCK_DIM, reduce_smem, s>>>(
        loss_partial, top1_partial, top5_partial,
        loss, top1, top5, batch, false);

    return cudaGetLastError();
}

cudaError_t launch_softmax_ce_fwd_amp(
    cudaStream_t s,
    const __half* logits, const int* labels,
    float* loss, float* top1, float* top5,    // 保留参数，kernel 不再直接写入
    int* pred, float* probs,                   // 保留参数，kernel 不再写入
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial,                       // [NEW] ids_out[6]
    int* top1_partial,                         // [NEW] ids_out[7]
    int* top5_partial,                         // [NEW] ids_out[8]
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    size_t smem = static_cast<size_t>(WARP_COUNT) * 2 * sizeof(float);
    softmax_ce_fwd_kernel<true><<<batch, BLOCK_DIM, smem, s>>>(
        static_cast<const void*>(logits), labels,
        loss, top1, top5, pred, probs,
        inv_scaling, scaling, batch_size, label_smoothing,
        loss_partial, top1_partial, top5_partial,
        batch, logits_stride, probs_stride, num_classes);

    size_t reduce_smem = BLOCK_DIM * sizeof(float)
                       + BLOCK_DIM * sizeof(int) * 2;
    softmax_ce_reduce_kernel<<<1, BLOCK_DIM, reduce_smem, s>>>(
        loss_partial, top1_partial, top5_partial,
        loss, top1, top5, batch, false);

    return cudaGetLastError();
}

cudaError_t launch_softmax_ce_inf_fp32(
    cudaStream_t s,
    const float* logits, const int* labels,
    float* loss, float* top1, float* top5,    // 保留参数，kernel 不再直接写入
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial,                       // [NEW] ids_out[6]
    int* top1_partial,                         // [NEW] ids_out[7]
    int* top5_partial,                         // [NEW] ids_out[8]
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    size_t smem = (static_cast<size_t>(WARP_COUNT) * 2 + BLOCK_DIM) * sizeof(float)
                + BLOCK_DIM * sizeof(int);

    // Step 1: INF main kernel — 每个 block 写入 partial[b]
    softmax_ce_inf_kernel<false><<<batch, BLOCK_DIM, smem, s>>>(
        static_cast<const void*>(logits), labels, loss, top1, top5, pred, probs,
        inv_scaling, scaling, batch_size, label_smoothing,
        loss_partial, top1_partial, top5_partial,
        batch, logits_stride, probs_stride, num_classes);

    // Step 2: 确定性归约 — 单 block 累加 → loss, top1, top5
    size_t reduce_smem = BLOCK_DIM * sizeof(float)
                       + BLOCK_DIM * sizeof(int) * 2;
    softmax_ce_reduce_kernel<<<1, BLOCK_DIM, reduce_smem, s>>>(
        loss_partial, top1_partial, top5_partial,
        loss, top1, top5, batch, true);    // INF: 计算 top metrics

    return cudaGetLastError();
}

cudaError_t launch_softmax_ce_inf_amp(
    cudaStream_t s,
    const __half* logits, const int* labels,
    float* loss, float* top1, float* top5,    // 保留参数，kernel 不再直接写入
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    const int32_t* batch_size,
    const float* label_smoothing,
    float* loss_partial,                       // [NEW] ids_out[6]
    int* top1_partial,                         // [NEW] ids_out[7]
    int* top5_partial,                         // [NEW] ids_out[8]
    int batch, int logits_stride, int probs_stride, int num_classes)
{
    size_t smem = (static_cast<size_t>(WARP_COUNT) * 2 + BLOCK_DIM) * sizeof(float)
                + BLOCK_DIM * sizeof(int);
    softmax_ce_inf_kernel<true><<<batch, BLOCK_DIM, smem, s>>>(
        static_cast<const void*>(logits), labels, loss, top1, top5, pred, probs,
        inv_scaling, scaling, batch_size, label_smoothing,
        loss_partial, top1_partial, top5_partial,
        batch, logits_stride, probs_stride, num_classes);

    size_t reduce_smem = BLOCK_DIM * sizeof(float)
                       + BLOCK_DIM * sizeof(int) * 2;
    softmax_ce_reduce_kernel<<<1, BLOCK_DIM, reduce_smem, s>>>(
        loss_partial, top1_partial, top5_partial,
        loss, top1, top5, batch, true);

    return cudaGetLastError();
}

cudaError_t launch_softmax_ce_bwd_fp32(
    cudaStream_t s,
    const float* probs, const int* labels,
    const float* scaling, const float* inv_scaling,
    const float* label_smoothing,
    float* grad, int batch, int probs_stride, int grad_stride, int num_classes)
{
    int total = batch * num_classes;
    int grid = (total + BLOCK_DIM - 1) / BLOCK_DIM;
    softmax_ce_bwd_kernel<false><<<grid, BLOCK_DIM, 0, s>>>(
        probs, labels, scaling, inv_scaling, label_smoothing,
        static_cast<void*>(grad),
        batch, probs_stride, grad_stride, num_classes);
    return cudaGetLastError();
}

cudaError_t launch_softmax_ce_bwd_amp(
    cudaStream_t s,
    const float* probs, const int* labels,
    const float* scaling, const float* inv_scaling,
    const float* label_smoothing,
    __half* grad, int batch, int probs_stride, int grad_stride, int num_classes)
{
    int total = batch * num_classes;
    int grid = (total + BLOCK_DIM - 1) / BLOCK_DIM;
    softmax_ce_bwd_kernel<true><<<grid, BLOCK_DIM, 0, s>>>(
        probs, labels, scaling, inv_scaling, label_smoothing,
        static_cast<void*>(grad),
        batch, probs_stride, grad_stride, num_classes);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA
