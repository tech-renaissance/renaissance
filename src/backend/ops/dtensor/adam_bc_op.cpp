/**
 * @file adam_bc_op.cpp
 * @brief Adam / AdamW Bias Correction — CPU kernel + 注册
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include <cmath>
#include <cstdint>

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

namespace tr {
extern void launch_scalar_increment_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state);
extern void launch_adam_bias_correction_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state);
} // namespace tr
#endif

namespace tr {

static void launch_scalar_increment_cpu(CpuOpContext* op_ctx) {
    int32_t* step_ptr = static_cast<int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    *step_ptr += 1;
}

static void launch_adam_bias_correction_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 3 && op_ctx->num_outputs >= 2,
             ShapeError, "ADAM_BIAS_CORRECTION requires 3 inputs + 2 outputs");

    const int32_t* step_ptr = static_cast<const int32_t*>(
        op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const float* b1_ptr = static_cast<const float*>(
        op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    const float* b2_ptr = static_cast<const float*>(
        op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));
    float* bc1_ptr = static_cast<float*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    float* bc2_ptr = static_cast<float*>(
        op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));

    int32_t t = *step_ptr;
    float b1 = *b1_ptr;
    float b2 = *b2_ptr;
    *bc1_ptr = 1.0f / (1.0f - std::pow(b1, static_cast<float>(t)));
    *bc2_ptr = 1.0f / (1.0f - std::pow(b2, static_cast<float>(t)));
}

void register_op_adam_bc() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::SCALAR_INCREMENT)];
        e.op = ComputeOp::SCALAR_INCREMENT;
        e.launch_cpu = launch_scalar_increment_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_scalar_increment_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::ADAM_BIAS_CORRECTION)];
        e.op = ComputeOp::ADAM_BIAS_CORRECTION;
        e.launch_cpu = launch_adam_bias_correction_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_adam_bias_correction_cuda;
#endif
    }

    TR_LOG_DEBUG("backend") << "Adam Bias Correction ops registered (CPU+CUDA)";
}

} // namespace tr