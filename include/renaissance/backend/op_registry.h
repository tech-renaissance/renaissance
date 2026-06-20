/**
 * @file op_registry.h
 * @brief 算子注册表：ComputeOp/RangeOp → launch 函数指针的全局映射表
 * @version 4.21.0
 * @date 2026-05-16
 * @author 技术觉醒团队
 * @note 依赖项: op_kind.h, computation_graph.h, device_context.h
 * @note 所属系列: backend
 */

#pragma once

#include "renaissance/graph/op_kind.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/memory_plan.h"
#include <cstddef>

#ifdef TR_USE_CUDA
#include "renaissance/graph/capture_multi_stream.h"
#endif

namespace tr {

class DeviceContext;

struct MemRangeInfo {
    uint64_t offset = 0;
    uint64_t size = 0;
    int32_t start_region_id = -1;
    int32_t end_region_id = -1;
};

struct CpuOpContext {
    const DeviceContext* ctx = nullptr;
    int32_t input_ids[12] = {};
    int32_t output_ids[12] = {};
    int num_inputs = 0;
    int num_outputs = 0;
    int64_t total_elements = 0;
    ShapeId input_shape{};
    ShapeId output_shape{};
    OpParams params;
    RangeOp range_op = RangeOp::UNKNOWN;  ///< RANGE 节点使用

    // DTensor stride（从 MemoryPlan 提取，CPU 路径需与 CUDA 路径对齐）
    int64_t n_stride = 0;
    int64_t h_stride = 0;
    int64_t w_stride = 0;
    int64_t c_stride = 0;

    // RangeOp 专用字段（V4.21 新增）
    MemRangeInfo input_ranges[12];
    MemRangeInfo output_ranges[12];
    int num_input_ranges = 0;
    int num_output_ranges = 0;
};

struct ComputeOpEntry {
    ComputeOp op = ComputeOp::UNKNOWN;

    void (*launch_cpu)(CpuOpContext* ctx) = nullptr;

#ifdef TR_USE_CUDA
    void (*launch_cuda)(const GraphNode& node,
                        const MemoryPlan& mp,
                        const DeviceContext& ctx,
                        MultiStreamCaptureState& state) = nullptr;
#endif
};

struct RangeOpEntry {
    RangeOp op = RangeOp::UNKNOWN;

    void (*launch_cpu)(CpuOpContext* ctx) = nullptr;

#ifdef TR_USE_CUDA
    void (*launch_cuda)(const GraphNode& node,
                        const MemoryPlan& mp,
                        const DeviceContext& ctx,
                        MultiStreamCaptureState& state) = nullptr;
#endif
};

extern ComputeOpEntry g_compute_op_table[];
extern RangeOpEntry   g_range_op_table[];

constexpr size_t kComputeOpCount = static_cast<size_t>(ComputeOp::COUNT);
constexpr size_t kRangeOpCount   = static_cast<size_t>(RangeOp::COUNT);

#ifdef TR_USE_CUDA
bool require_warmup(ComputeOp op) noexcept;
void warmup_single_cudnn_op(const GraphNode& node,
                             const MemoryPlan& mp,
                             DeviceContext& ctx);
#endif

void register_default_ops();
void register_op_relu();
void register_op_identity();
void register_op_tanh();
void register_op_silu();
void register_op_relu6();
void register_op_leaky_relu();
void register_op_hardswish();
void register_op_elu();
void register_op_sigmoid();
void register_op_fc();
void register_op_flatten();
void register_op_channel_padding();
void register_op_axpy();
void register_op_conv();
void register_op_gap();
void register_op_softmax_ce();
void register_op_range_h2d();

void register_op_range_clear();
void register_op_range_d2d_copy();
void register_op_range_cast();
void register_op_range_check_nan();
void register_op_range_allreduce();
void register_op_range_optimizer();
void register_op_lars();
void register_op_dtensor_copy();
void register_op_range_grad_scaling();
void register_op_range_accum_metrics();
void register_op_adam_bc();
void register_op_maxpool();
void register_op_avgpool();
void register_op_dropout();
void register_op_bn();
void register_op_cbr();

} // namespace tr