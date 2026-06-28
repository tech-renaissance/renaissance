/**
 * @file axpy_op.cpp
 * @brief AXPY算子实现：launch_cuda / launch_cpu / 算子表注册
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: op_registry.h, device_context.h, capture_multi_stream.h
 * @note 所属系列: backend/ops/dtensor
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

#include <cstdint>
#include <cstddef>
#include <variant>

namespace tr {

#ifdef TR_USE_CUDA

// CUDA kernel外部声明（在axpy_op.cu中实现）
extern cudaError_t launch_axpy_fwd_kernel(
    const float* a, const float* b, float alpha, float* c,
    int N, int H, int W, int C,
    size_t stride_n, size_t stride_h, size_t stride_w, size_t stride_c,
    cudaStream_t stream);

/**
 * @brief AXPY前向CUDA launch函数
 *
 * 实现：c = alpha * a + b
 * 使用COMP_1单流执行
 */
static void launch_axpy_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<AxpyParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "AXPY_FWD missing AxpyParams");

    TR_CHECK(node.input_ids.size() >= 2, ShapeError,
             "AXPY_FWD requires 2 inputs (a, b)");
    TR_CHECK(node.output_ids.size() >= 1, ShapeError,
             "AXPY_FWD requires 1 output (c)");

    // 获取DTensor引用
    const DTensor& dt_a = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_b = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_c = mp.get_dtensor(node.output_ids[0]);

    // 解析设备指针
    const float* a = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const float* b = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    float* c = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    // 提取形状信息（NHWC）- 使用快速访问方法
    int N = dt_a.n();
    int H = dt_a.h();
    int W = dt_a.w();
    int C = dt_a.c();

    // 提取 CUDA 对齐 stride（含 padding 间距）
    size_t stride_n = static_cast<size_t>(dt_a.n_stride_cuda());
    size_t stride_h = static_cast<size_t>(dt_a.h_stride_cuda());
    size_t stride_w = static_cast<size_t>(dt_a.w_stride_cuda());
    size_t stride_c = static_cast<size_t>(dt_a.c_stride_cuda());

    // 获取alpha参数
    float alpha = p->alpha;

    // 使用COMP_1流
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    // 启动kernel
    cudaError_t err = launch_axpy_fwd_kernel(
        a, b, alpha, c, N, H, W, C,
        stride_n, stride_h, stride_w, stride_c, s);

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("AXPY_FWD kernel launch failed: " << cudaGetErrorString(err));
    }

    // 记录完成事件
    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

/**
 * @brief AXPY前向CPU实现
 *
 * 实现：c = alpha * a + b
 * 纯CPU版本，使用四重循环遍历NHWC张量
 */
static void launch_axpy_fwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<AxpyParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "AXPY_FWD CPU missing AxpyParams");

    TR_CHECK(op_ctx->num_inputs >= 2, ShapeError,
             "AXPY_FWD CPU requires 2 inputs");
    TR_CHECK(op_ctx->num_outputs >= 1, ShapeError,
             "AXPY_FWD CPU requires 1 output");

    // 解析设备指针
    const float* a = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    const float* b = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* c = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));

    // 从输入形状获取NHWC信息 - 使用快速访问方法
    int N = op_ctx->input_shape.n;  // ShapeId已经存储了正确的值
    int H = op_ctx->input_shape.h;
    int W = op_ctx->input_shape.w;
    int C = op_ctx->input_shape.c;

    // CPU上所有DTensor按框架设计必定紧凑（有效元素间无padding），
    // 因此直接使用紧凑stride即可，无需也不应从DTensor获取cuda_alignment()计算的stride
    size_t stride_n = H * W * C;
    size_t stride_h = W * C;
    size_t stride_w = C;
    size_t stride_c = 1;

    // 获取alpha参数
    float alpha = p->alpha;

    // CPU四重循环实现AXPY
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int ch = 0; ch < C; ++ch) {
                    size_t off = n * stride_n + h * stride_h +
                                 w * stride_w + ch * stride_c;
                    c[off] = alpha * a[off] + b[off];
                }
            }
        }
    }
}

/**
 * @brief 注册AXPY算子到全局算子表
 */
void register_op_axpy() {
    auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::AXPY_FWD)];
    entry.op = ComputeOp::AXPY_FWD;
    entry.launch_cpu = launch_axpy_fwd_cpu;

#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_axpy_fwd_cuda;
#endif

    TR_LOG_DEBUG("backend") << "AXPY_FWD operator registered (FP32, CPU+CUDA)";
}

} // namespace tr