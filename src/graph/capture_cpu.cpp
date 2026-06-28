/**
 * @file capture_cpu.cpp
 * @brief CPU 后端图捕获：收集 CpuOp 序列
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#include "renaissance/graph/captured_graph.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/backend/op_registry.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/shape_id.h"
#include "renaissance/tensor/distributed_tensor.h"

namespace tr {


static CpuOpContext* alloc_cpu_op_context() {
    return new CpuOpContext();
}

void capture_cpu(const ComputationGraph& cg,
                  const MemoryPlan& mp,
                  GraphId gid,
                  const DeviceContext& ctx,
                  std::vector<CpuOp>& cpu_ops) {
    // 获取节点列表：优先使用linear_nodes_（SimpleTask手动绘图），其次使用nodes(gid)（Compiler自动构图）
    const auto& nodes = cg.linear_nodes().empty() ? cg.nodes(gid) : cg.linear_nodes();

    for (const auto& node : nodes) {
        if (node.kind == GraphNode::Kind::COMPUTE) {
            CpuOpContext* op_ctx = alloc_cpu_op_context();
            op_ctx->ctx = &ctx;
            op_ctx->num_inputs = 0;
            op_ctx->num_outputs = 0;

            for (size_t i = 0; i < node.input_ids.size() && i < 12; ++i) {
                op_ctx->input_ids[op_ctx->num_inputs++] = node.input_ids[i];
            }
            for (size_t i = 0; i < node.output_ids.size() && i < 12; ++i) {
                op_ctx->output_ids[op_ctx->num_outputs++] = node.output_ids[i];
            }
            op_ctx->params = node.params;

            if (!node.output_ids.empty()) {
                const DTensor& dt = mp.get_dtensor(node.output_ids[0]);
                op_ctx->total_elements = dt.numel();
                op_ctx->output_shape = ShapeId{
                    dt.shape.n(), dt.shape.h(),
                    dt.shape.w(), dt.shape.c()};
            }
            if (!node.input_ids.empty()) {
                const DTensor& dt_in = mp.get_dtensor(node.input_ids[0]);
                op_ctx->input_shape = ShapeId{
                    dt_in.shape.n(), dt_in.shape.h(),
                    dt_in.shape.w(), dt_in.shape.c()};
                // 填充 CPU 紧凑 stride — 框架保证 CPU 上 DTensor 必定紧凑
                op_ctx->n_stride = dt_in.n_stride_cpu();
                op_ctx->h_stride = dt_in.h_stride_cpu();
                op_ctx->w_stride = dt_in.w_stride_cpu();
                op_ctx->c_stride = dt_in.c_stride_cpu();
            }

            CpuOp op;
            op.fn = g_compute_op_table[static_cast<size_t>(node.compute_op)].launch_cpu;
            op.ctx = op_ctx;
            cpu_ops.push_back(op);
        } else {
            CpuOpContext* op_ctx = alloc_cpu_op_context();
            op_ctx->ctx = &ctx;
            op_ctx->range_op = node.range_op;
            op_ctx->params = node.params;

            for (size_t i = 0; i < node.input_ranges.size() && i < 12; ++i) {
                auto [off, sz] = mp.resolve_region_bounds(
                    static_cast<Region>(node.input_ranges[i].start_region_id),
                    static_cast<Region>(node.input_ranges[i].end_region_id));
                op_ctx->input_ranges[i] = {off, sz,
                    node.input_ranges[i].start_region_id,
                    node.input_ranges[i].end_region_id};
            }
            for (size_t i = 0; i < node.output_ranges.size() && i < 12; ++i) {
                auto [off, sz] = mp.resolve_region_bounds(
                    static_cast<Region>(node.output_ranges[i].start_region_id),
                    static_cast<Region>(node.output_ranges[i].end_region_id));
                op_ctx->output_ranges[i] = {off, sz,
                    node.output_ranges[i].start_region_id,
                    node.output_ranges[i].end_region_id};
            }
            op_ctx->num_input_ranges  = static_cast<int>(node.input_ranges.size());
            op_ctx->num_output_ranges = static_cast<int>(node.output_ranges.size());

            for (size_t i = 0; i < node.input_ids.size() && i < 12; ++i) {
                op_ctx->input_ids[i] = node.input_ids[i];
            }
            for (size_t i = 0; i < node.output_ids.size() && i < 12; ++i) {
                op_ctx->output_ids[i] = node.output_ids[i];
            }
            op_ctx->num_inputs  = static_cast<int>(node.input_ids.size());
            op_ctx->num_outputs = static_cast<int>(node.output_ids.size());

            auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
            TR_CHECK(entry.launch_cpu != nullptr, RuntimeError,
                     "RangeOp " << range_op_to_string(node.range_op)
                     << " has no CPU implementation");
            CpuOp op;
            op.fn = entry.launch_cpu;
            op.ctx = op_ctx;
            cpu_ops.push_back(op);
        }
    }
}

} // namespace tr
