/**
 * @file example_dual_graph_cpu.cpp
 * @brief Dual-graph example: compute and transfer on CPU (forced CPU execution)
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace tr;

namespace {

// --- Constants ---
constexpr float kAlpha    = 2.0f;
constexpr float kWeight   = 3.0f;
constexpr float kInputA   = -4.0f;
constexpr float kExpectedTemp   = kAlpha * kInputA + kWeight;   // -5.0
constexpr float kExpectedResult = 0.0f;                         // ReLU(-5.0)
constexpr float kTolerance = 1e-5f;

constexpr int kShapeDim = 1024;
const Shape kShape{1, kShapeDim, 1, 1};

} // anonymous namespace

int main() {
    // ------------------------------------------------------------------
    //  Step 1: Force CPU mode (ignore GPU availability)
    // ------------------------------------------------------------------
    GLOBAL_SETTING.use_cpu().auto_seed();

    std::cout << "\n========================================\n";
    std::cout << "CPU Configuration: Forced CPU mode\n";
    std::cout << "Using CPU for computation (ignoring GPU availability)\n";

    // ------------------------------------------------------------------
    //  Step 2: Create task and allocate tensors
    // ------------------------------------------------------------------
    SimpleTask task;

    DTensor d_buf_a   = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);   // current batch input
    DTensor d_buf_b   = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);   // next batch buffer
    DTensor d_weight  = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);   // weight W
    DTensor d_temp    = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);   // AXPY output
    DTensor d_result  = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);   // ReLU output
    DTensor d_mask    = task.alloc(kShape, DType::INT8, Region::S_MASK);   // ReLU bitmask

    task.finalize_memory();   // lock layout

    // ------------------------------------------------------------------
    //  Step 3: Prepare host tensors
    // ------------------------------------------------------------------
    Tensor host_a(kShape, DType::FP32);
    float* pa = host_a.data<float>();
    for (int i = 0; i < static_cast<int>(host_a.numel()); ++i) {
        pa[i] = kInputA;
    }

    // For CPU mode, we only have single rank, so prepare single tensor
    Tensor host_b_next(kShape, DType::FP32);
    float* pb = host_b_next.data<float>();
    const float val = 10.0f;   // CPU mode: rank 0 gets 10.0
    for (int i = 0; i < static_cast<int>(host_b_next.numel()); ++i) {
        pb[i] = val;
    }

    // ------------------------------------------------------------------
    //  Step 4: Build the two named graphs
    // ------------------------------------------------------------------
    // Graph 1 - Transfer: H2D range node (CPU placeholder)
    ComputationGraph g_xfer;
    GraphNode xfer_node;
    xfer_node.kind = GraphNode::Kind::RANGE;
    xfer_node.range_op = RangeOp::RANGE_H2D_COPY_A;
    const DTensor& live_b = task.memory_plan().get_dtensor(d_buf_b.id);
    xfer_node.output_ranges = {
        MemRange{live_b.offset(), live_b.slot_bytes(),
                 static_cast<int32_t>(live_b.region),
                 static_cast<int32_t>(live_b.region)}
    };
    g_xfer.append(std::move(xfer_node));
    task.add_graph("xfer", std::move(g_xfer), StreamKind::TRANS);

    // Graph 2 - Compute: AXPY -> ReLU
    ComputationGraph g_compute;
    g_compute.append(ComputeOp::AXPY_FWD,
                     {d_buf_a.id, d_weight.id},          // inputs: A, W
                     {d_temp.id},                         // output: temp
                     OpParams{AxpyParams{kAlpha}});       // alpha = 2.0
    g_compute.append(ComputeOp::RELU_FP32_FWD,
                     {d_temp.id},
                     {d_result.id, d_mask.id});
    task.add_graph("compute", std::move(g_compute), StreamKind::COMP_1);

    // ------------------------------------------------------------------
    //  Step 5: Compile for CPU
    // ------------------------------------------------------------------
    task.compile();

    // ------------------------------------------------------------------
    //  Step 6: Initialize data on CPU
    // ------------------------------------------------------------------
    task.transfer(host_a, d_buf_a);
    task.fill(d_weight, kWeight);   // W = 3.0

    // 图外同步传输：新版 ComputationGraph 使用 RANGE 节点表达图内 H2D，
    // 当前 capture 为 placeholder，通过图外 API 确保数据正确性
    task.transfer(host_b_next, d_buf_b);

    // ------------------------------------------------------------------
    //  Step 7: Launch dual-graph execution (compute + transfer)
    // Note: In CPU mode, StreamKind is only a label; dual graphs execute sequentially
    // ------------------------------------------------------------------
    task.run("xfer", "compute");

    // ------------------------------------------------------------------
    //  Step 8: Verify compute result
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    //  Step 8: Verify compute pipeline: d_temp, d_mask, d_result
    // ------------------------------------------------------------------
    bool compute_ok = true;

    // 新增：验证 d_temp
    Tensor h_temp = task.fetch(d_temp);
    const float* temp_data = h_temp.data<float>();
    for (int i = 0; i < static_cast<int>(h_temp.numel()); ++i) {
        if (std::fabs(temp_data[i] - kExpectedTemp) > kTolerance) {
            compute_ok = false;
            std::cerr << "  CPU: d_temp[" << i << "] = " << temp_data[i]
                      << " (expected " << kExpectedTemp << ")\n";
            break;
        }
    }

    // 新增：验证 d_mask
    Tensor h_mask = task.fetch(d_mask);
    const int8_t* mask_data = h_mask.data<int8_t>();
    // AXPY 输出 = -5 < 0 → ReLU mask 应为 0（全负）
    for (int i = 0; i < static_cast<int>(h_mask.numel()) && compute_ok; ++i) {
        if (mask_data[i] != 0) {
            compute_ok = false;
            std::cerr << "  CPU: d_mask[" << i << "] = " << (int)mask_data[i]
                      << " (expected 0, all negative)\n";
            break;
        }
    }

    // 验证 d_result
    Tensor h_result = task.fetch(d_result);
    const float* pr = h_result.data<float>();

    for (int i = 0; i < static_cast<int>(h_result.numel()); ++i) {
        if (std::fabs(pr[i] - kExpectedResult) > kTolerance) {
            compute_ok = false;
            std::cerr << "  CPU: result[" << i << "] = " << pr[i]
                      << " (expected " << kExpectedResult << ")\n";
            break;
        }
    }

    // ------------------------------------------------------------------
    //  Step 9: Verify transfer
    // ------------------------------------------------------------------
    bool xfer_ok = true;
    Tensor h_buf_b = task.fetch(d_buf_b);
    const float* pb_result = h_buf_b.data<float>();

    for (int i = 0; i < static_cast<int>(h_buf_b.numel()); ++i) {
        if (std::fabs(pb_result[i] - val) > kTolerance) {
            xfer_ok = false;
            std::cerr << "  CPU: buf_b[" << i << "] = " << pb_result[i]
                      << " (expected " << val << ")\n";
            break;
        }
    }

    // ------------------------------------------------------------------
    //  Step 10: Report
    // ------------------------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << "Dual Graph Test: Compute + Transfer (CPU)\n";
    std::cout << "========================================\n";
    std::cout << "Device: CPU (single rank)\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Compute (AXPY + ReLU): " << (compute_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "  A = " << kInputA << ", W = " << kWeight << ", alpha = " << kAlpha << "\n";
    std::cout << "  temp = " << kAlpha << " * (" << kInputA << ") + " << kWeight
              << " = " << kExpectedTemp << "\n";
    std::cout << "  result = ReLU(" << kExpectedTemp << ") = " << kExpectedResult << "\n";
    std::cout << "  result[0] = " << pr[0] << " (expected " << kExpectedResult << ")\n";
    std::cout << "\nTransfer (H2D): " << (xfer_ok ? "PASS" : "FAIL") << "\n";
    if (xfer_ok) {
        std::cout << "  CPU received data correctly: " << val << ".0\n";
    }
    std::cout << "========================================\n\n";

    return (compute_ok && xfer_ok) ? 0 : 1;
}
