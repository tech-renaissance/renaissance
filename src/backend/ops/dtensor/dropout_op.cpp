/**
 * @file dropout_op.cpp
 * @brief Dropout 算子实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/op_stream_policy.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/types.h"
#include "renaissance/core/global_registry.h"
#include "renaissance/core/rng.h"
#include "renaissance/core/logger.h"

#ifdef TR_USE_CUDA
#  include <cuda_runtime.h>
#  include <cuda_fp16.h>
#  include <cstdint>
#endif

#include <cstring>

namespace tr {

// ============================================================================
// Philox RNG (用于确定性随机数生成)
// ============================================================================

namespace detail {

inline uint64_t philox_round(uint64_t k0, uint64_t k1, uint64_t c0, uint64_t c1) {
    // Philox 2x64 单轮（简化版）
    uint64_t hi = 0, lo = 0;

    // mulhilo: multiply two 64-bit numbers and get high 64 bits
    auto mulhilo = [](uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
#ifdef _MSC_VER
        lo = a * b;
        hi = __umulh(a, b);
#else
        __uint128_t p = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
        lo = static_cast<uint64_t>(p);
        hi = static_cast<uint64_t>(p >> 64);
#endif
    };

    mulhilo(0xD2B74407B1CE6E93ULL, k0, hi, lo);
    k0 = hi ^ k1 ^ c0;
    k1 = lo ^ c1;
    return k0;
}

inline float philox_uniform_float(uint64_t& seed_lo, uint64_t& seed_hi, uint64_t offset) {
    // Philox 2x64 一步生成
    uint64_t counter = seed_lo + offset;
    uint64_t key_lo = seed_hi;
    uint64_t key_hi = seed_lo ^ 0x9e3779b97f4a7c15ULL;

    // 10 rounds
    for (int i = 0; i < 10; ++i) {
        uint64_t hi = 0, lo = 0;
#ifdef _MSC_VER
        lo = 0xD2B74407B1CE6E93ULL * counter;
        hi = __umulh(0xD2B74407B1CE6E93ULL, counter);
#else
        __uint128_t mul = static_cast<__uint128_t>(0xD2B74407B1CE6E93ULL) * static_cast<__uint128_t>(counter);
        lo = static_cast<uint64_t>(mul);
        hi = static_cast<uint64_t>(mul >> 64);
#endif
        uint64_t new_counter = hi ^ key_lo ^ key_hi;
        uint64_t new_key = lo;
        counter = new_counter;
        key_lo = new_key;
        key_hi = key_lo ^ key_hi;
    }

    // 将 counter 转换为 [0, 1) 的 float
    uint32_t upper = static_cast<uint32_t>(counter >> 32);
    return static_cast<float>(upper) * (1.0f / 4294967296.0f);
}

} // namespace detail

// ============================================================================
// CPU 实现
// ============================================================================

static void launch_not_supported_cpu(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("AMP Dropout not supported on CPU");
}

static void launch_dropout_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    int8_t*    mask = static_cast<int8_t*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));

    float p = op_ctx->params.dropout().p;
    TR_CHECK(p >= 0.0f && p < 1.0f, ValueError, "Dropout p must be in [0.0, 1.0)");
    float scale = 1.0f / (1.0f - p);
    int total = op_ctx->input_shape.n * op_ctx->input_shape.h
              * op_ctx->input_shape.w * op_ctx->input_shape.c;

    // 从 dropout_seed tensor 读取 → Xorshift64* 旋转 → 写回（与 GPU 行为对齐）
    // init_all() 每次都会重新初始化，保证跨 run 可复现
    int32_t* seed_ptr = static_cast<int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->ctx->memory_plan()->baseline().dropout_seed));
    uint64_t s = (static_cast<uint64_t>(static_cast<uint32_t>(seed_ptr[1])) << 32)
               | static_cast<uint32_t>(seed_ptr[0]);
    if (s == 0) s = 0x9e3779b97f4a7c15ULL;
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    s *= 0x2545F4914F6CDD1DULL;
    seed_ptr[0] = static_cast<int32_t>(s & 0xFFFFFFFFULL);
    seed_ptr[1] = static_cast<int32_t>(s >> 32);

    uint64_t seed_lo = s;
    uint64_t seed_hi = s ^ 0x9e3779b97f4a7c15ULL;

    for (int i = 0; i < total; ++i) {
        float r = detail::philox_uniform_float(seed_lo, seed_hi, static_cast<uint64_t>(i));
        if (r < p) {
            y[i] = 0.0f;
            mask[i] = 0;
        } else {
            y[i] = x[i] * scale;
            mask[i] = 1;
        }
    }

#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

static void launch_dropout_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const float* dy = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const int8_t* mask = static_cast<const int8_t*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    float*       dx = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    float p = op_ctx->params.dropout().p;
    TR_CHECK(p >= 0.0f && p < 1.0f, ValueError, "Dropout p must be in [0.0, 1.0)");
    float scale = 1.0f / (1.0f - p);
    int total = op_ctx->output_shape.n * op_ctx->output_shape.h
              * op_ctx->output_shape.w * op_ctx->output_shape.c;

    for (int i = 0; i < total; ++i) {
        dx[i] = mask[i] ? dy[i] * scale : 0.0f;
    }
}

static void launch_dropout_fp32_inf_cpu(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int total = op_ctx->input_shape.n * op_ctx->input_shape.h
              * op_ctx->input_shape.w * op_ctx->input_shape.c;

    // Inference: identity
    std::memcpy(y, x, total * sizeof(float));
}

// ============================================================================
// CUDA 实现
// ============================================================================

#ifdef TR_USE_CUDA

// CUDA kernel declarations (defined in .cu file)
cudaError_t launch_rotate_dropout_seed(int32_t* seed_ptr, cudaStream_t stream);
cudaError_t launch_dropout_fwd_kernel(
    const float* x, float* y, int8_t* mask,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float p, float scale,
    const int32_t* seed_ptr,
    cudaStream_t stream);

cudaError_t launch_dropout_bwd_kernel(
    const float* dy, const int8_t* mask, float* dx,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float scale,
    cudaStream_t stream);

cudaError_t launch_dropout_inf_kernel(
    const float* x, float* y,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    cudaStream_t stream);

cudaError_t launch_dropout_fwd_amp_kernel(
    const __half* x, __half* y, int8_t* mask,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float p, float scale,
    const int32_t* seed_ptr,
    cudaStream_t stream);

cudaError_t launch_dropout_bwd_amp_kernel(
    const __half* dy, const int8_t* mask, __half* dx,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float scale,
    cudaStream_t stream);

cudaError_t launch_dropout_inf_amp_kernel(
    const __half* x, __half* y,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    cudaStream_t stream);

static void launch_dropout_fp32_fwd_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    const float* x = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    float*       y = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    int8_t*    mask = static_cast<int8_t*>(ctx.ptr_at(node.output_ids[1]));

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);

    float p = node.params.dropout().p;
    TR_CHECK(p >= 0.0f && p < 1.0f, ValueError, "Dropout p must be in [0.0, 1.0)");
    float scale = 1.0f / (1.0f - p);

    int32_t* seed_ptr = static_cast<int32_t*>(
        ctx.ptr_at(mp.baseline().dropout_seed));

    const DTensor& dt_mask = mp.get_dtensor(node.output_ids[1]);
    int64_t feat_n_stride = dt_x.n_stride_cuda();
    int64_t feat_h_stride = dt_x.h_stride_cuda();
    int64_t feat_w_stride = dt_x.w_stride_cuda();
    int64_t mask_n_stride = dt_mask.n_stride_cuda();
    int64_t mask_h_stride = dt_mask.h_stride_cuda();
    int64_t mask_w_stride = dt_mask.w_stride_cuda();

    // 确定性种子旋转：同一 stream 上先旋转 seed，再执行 dropout FWD
    launch_rotate_dropout_seed(seed_ptr, s);

    cudaError_t err = launch_dropout_fwd_kernel(
        x, y, mask, dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        feat_n_stride, feat_h_stride, feat_w_stride,
        mask_n_stride, mask_h_stride, mask_w_stride,
        p, scale, seed_ptr, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("DROPOUT_FP32_FWD kernel failed: " << cudaGetErrorString(err));
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_dropout_fp32_bwd_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    const float* dy = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const int8_t* mask = static_cast<const int8_t*>(ctx.ptr_at(node.input_ids[1]));
    float*       dx = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);
    const DTensor& dt_mask = mp.get_dtensor(node.input_ids[1]);
    int64_t feat_n_stride = dt_dx.n_stride_cuda();
    int64_t feat_h_stride = dt_dx.h_stride_cuda();
    int64_t feat_w_stride = dt_dx.w_stride_cuda();
    int64_t mask_n_stride = dt_mask.n_stride_cuda();
    int64_t mask_h_stride = dt_mask.h_stride_cuda();
    int64_t mask_w_stride = dt_mask.w_stride_cuda();

    float scale = 1.0f / (1.0f - node.params.dropout().p);

    cudaError_t err = launch_dropout_bwd_kernel(
        dy, mask, dx, dt_dx.n(), dt_dx.h(), dt_dx.w(), dt_dx.c(),
        feat_n_stride, feat_h_stride, feat_w_stride,
        mask_n_stride, mask_h_stride, mask_w_stride,
        scale, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("DROPOUT_FP32_BWD kernel failed: " << cudaGetErrorString(err));
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_dropout_fp32_inf_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    const float* x = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    float*       y = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    int64_t feat_n_stride = dt_x.n_stride_cuda();
    int64_t feat_h_stride = dt_x.h_stride_cuda();
    int64_t feat_w_stride = dt_x.w_stride_cuda();

    cudaError_t err = launch_dropout_inf_kernel(
        x, y, dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        feat_n_stride, feat_h_stride, feat_w_stride, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("DROPOUT_FP32_INF kernel failed: " << cudaGetErrorString(err));
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_dropout_amp_fwd_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    const __half* x = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    __half*       y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
    int8_t*    mask = static_cast<int8_t*>(ctx.ptr_at(node.output_ids[1]));

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_mask = mp.get_dtensor(node.output_ids[1]);
    int64_t feat_n_stride = dt_x.n_stride_cuda();
    int64_t feat_h_stride = dt_x.h_stride_cuda();
    int64_t feat_w_stride = dt_x.w_stride_cuda();
    int64_t mask_n_stride = dt_mask.n_stride_cuda();
    int64_t mask_h_stride = dt_mask.h_stride_cuda();
    int64_t mask_w_stride = dt_mask.w_stride_cuda();

    float p = node.params.dropout().p;
    TR_CHECK(p >= 0.0f && p < 1.0f, ValueError, "Dropout p must be in [0.0, 1.0)");
    float scale = 1.0f / (1.0f - p);

    int32_t* seed_ptr = static_cast<int32_t*>(
        ctx.ptr_at(mp.baseline().dropout_seed));

    // 确定性种子旋转：同一 stream 上先旋转 seed，再执行 dropout FWD
    launch_rotate_dropout_seed(seed_ptr, s);

    cudaError_t err = launch_dropout_fwd_amp_kernel(
        x, y, mask, dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        feat_n_stride, feat_h_stride, feat_w_stride,
        mask_n_stride, mask_h_stride, mask_w_stride,
        p, scale, seed_ptr, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("DROPOUT_AMP_FWD kernel failed: " << cudaGetErrorString(err));
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_dropout_amp_bwd_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    const __half* dy = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    const int8_t* mask = static_cast<const int8_t*>(ctx.ptr_at(node.input_ids[1]));
    __half*       dx = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);
    const DTensor& dt_mask = mp.get_dtensor(node.input_ids[1]);
    int64_t feat_n_stride = dt_dx.n_stride_cuda();
    int64_t feat_h_stride = dt_dx.h_stride_cuda();
    int64_t feat_w_stride = dt_dx.w_stride_cuda();
    int64_t mask_n_stride = dt_mask.n_stride_cuda();
    int64_t mask_h_stride = dt_mask.h_stride_cuda();
    int64_t mask_w_stride = dt_mask.w_stride_cuda();

    float scale = 1.0f / (1.0f - node.params.dropout().p);

    cudaError_t err = launch_dropout_bwd_amp_kernel(
        dy, mask, dx, dt_dx.n(), dt_dx.h(), dt_dx.w(), dt_dx.c(),
        feat_n_stride, feat_h_stride, feat_w_stride,
        mask_n_stride, mask_h_stride, mask_w_stride,
        scale, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("DROPOUT_AMP_BWD kernel failed: " << cudaGetErrorString(err));
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_dropout_amp_inf_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    const __half* x = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    __half*       y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    int64_t feat_n_stride = dt_x.n_stride_cuda();
    int64_t feat_h_stride = dt_x.h_stride_cuda();
    int64_t feat_w_stride = dt_x.w_stride_cuda();

    cudaError_t err = launch_dropout_inf_amp_kernel(
        x, y, dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        feat_n_stride, feat_h_stride, feat_w_stride, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("DROPOUT_AMP_INF kernel failed: " << cudaGetErrorString(err));
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

// ============================================================================
// 注册
// ============================================================================

void register_op_dropout() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::DROPOUT_FP32_FWD)];
        e.op = ComputeOp::DROPOUT_FP32_FWD;
        e.launch_cpu = launch_dropout_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_dropout_fp32_fwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::DROPOUT_FP32_BWD)];
        e.op = ComputeOp::DROPOUT_FP32_BWD;
        e.launch_cpu = launch_dropout_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_dropout_fp32_bwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::DROPOUT_FP32_INF)];
        e.op = ComputeOp::DROPOUT_FP32_INF;
        e.launch_cpu = launch_dropout_fp32_inf_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_dropout_fp32_inf_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::DROPOUT_AMP_FWD)];
        e.op = ComputeOp::DROPOUT_AMP_FWD;
        e.launch_cpu = launch_not_supported_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_dropout_amp_fwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::DROPOUT_AMP_BWD)];
        e.op = ComputeOp::DROPOUT_AMP_BWD;
        e.launch_cpu = launch_not_supported_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_dropout_amp_bwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::DROPOUT_AMP_INF)];
        e.op = ComputeOp::DROPOUT_AMP_INF;
        e.launch_cpu = launch_not_supported_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_dropout_amp_inf_cuda;
#endif
    }
}

} // namespace tr