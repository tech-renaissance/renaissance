/**
 * @file example_dual_graph.cpp
 * @brief Dual-graph example: overlapping compute and transfer on 4 ranks
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

constexpr int kPhysicalGpuIds[] = {1, 2, 4, 7};

} // anonymous namespace

int main() {
    // ------------------------------------------------------------------
    //  Step 1: Detect visible GPUs and configure
    // ------------------------------------------------------------------
    const int visible_gpu_count = GlobalRegistry::get_visible_gpu_count();

    if (visible_gpu_count >= 8) {
        GLOBAL_SETTING.use_gpu("1,2,4,7").auto_seed();
    } else if (visible_gpu_count > 0) {
        GLOBAL_SETTING.use_gpu("0").auto_seed();
    } else {
        GLOBAL_SETTING.use_cpu().auto_seed();
    }

    // ------------------------------------------------------------------
    //  Step 2: Create task and allocate tensors
    // ------------------------------------------------------------------
    SimpleTask task;

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    DTensor d_buf_a   = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);   // current batch input
    DTensor d_buf_b   = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);   // next batch buffer (multi-to-multi target)
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

    // Per-rank distinct data for the multi-to-multi transfer
    std::vector<Tensor> host_b_next;
    host_b_next.reserve(num_ranks);
    for (int rank = 0; rank < num_ranks; ++rank) {
        host_b_next.emplace_back(kShape, DType::FP32);
        float* p = host_b_next[rank].data<float>();
        const float val = static_cast<float>(10 + rank);   // rank0: 10.0, rank1: 11.0, ...
        for (int i = 0; i < static_cast<int>(host_b_next[rank].numel()); ++i) {
            p[i] = val;
        }
    }

    // ------------------------------------------------------------------
    //  Step 4: Build the two named graphs
    // ------------------------------------------------------------------
    // Graph 1 - Transfer stream: H2D DTensor-single node (async overlap)
    ComputationGraph g_xfer;
    GraphNode xfer_node;
    xfer_node.kind = GraphNode::Kind::RANGE;
    xfer_node.range_op = RangeOp::RANGE_H2D_COPY_DTENSOR;
    const DTensor& live_b = task.memory_plan().get_dtensor(d_buf_b.id);
    xfer_node.output_ranges = {
        MemRange{live_b.offset(), live_b.slot_bytes(),
                 static_cast<int32_t>(live_b.region),
                 static_cast<int32_t>(live_b.region)}
    };
    g_xfer.append(std::move(xfer_node));
    task.add_graph("xfer", std::move(g_xfer), StreamKind::TRANS);

    // Graph 2 - Compute stream: AXPY -> ReLU
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
    //  Step 5: Compile (one-shot: arena allocation + GPU capture)
    // ------------------------------------------------------------------
    task.compile();

    // ------------------------------------------------------------------
    //  Step 6: Initialize data on device
    // ------------------------------------------------------------------
    // Recommended: send A to rank 0, then broadcast
    task.transfer_to_rank(host_a, d_buf_a, 0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_buf_a);
    }
    // Alternative one-shot transfer:
    // task.transfer(host_a, d_buf_a);

    task.fill(d_weight, kWeight);   // W = 3.0 on all ranks
    task.fill(d_buf_b, -1.0f);         // pre-fill to distinguish from transfer result

    // TODO(BROKEN): fill_transfer_buffer removed (UTK v5.0).
    //   d_buf_b is 1024-element FP32 tensor (4KB) > StagingParamPool 256B limit.
    //   Replace with RANGE_H2D_COPY_A/B + StagingBufferPool for >256B DTensor.
    //   task.fill_transfer_buffer(host_b_next[0], d_buf_b);

    // ------------------------------------------------------------------
    //  Step 7: Launch dual-graph execution (compute || transfer)
    // ------------------------------------------------------------------
    task.run("xfer", "compute");

    // ------------------------------------------------------------------
    //  Step 8: Verify compute pipeline: d_temp, d_mask, d_result on ALL ranks
    // ------------------------------------------------------------------
    bool compute_ok = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        // AXPY intermediate: temp = alpha * A + W = 2 * (-4) + 3 = -5
        {
            Tensor h_temp = task.fetch_from_rank(d_temp, rank);
            const float* pt = h_temp.data<float>();
            for (int i = 0; i < static_cast<int>(h_temp.numel()); ++i) {
                if (std::fabs(pt[i] - kExpectedTemp) > kTolerance) {
                    compute_ok = false;
                    std::cerr << "Rank " << rank << " d_temp[" << i << "] mismatch: expected "
                              << kExpectedTemp << ", got " << pt[i] << "\n";
                    break;
                }
            }
        }
        if (!compute_ok) break;

        // ReLU output: ReLU(-5) = 0
        {
            Tensor h_result = task.fetch_from_rank(d_result, rank);
            const float* pr = h_result.data<float>();
            for (int i = 0; i < static_cast<int>(h_result.numel()); ++i) {
                if (std::fabs(pr[i] - kExpectedResult) > kTolerance) {
                    compute_ok = false;
                    std::cerr << "Rank " << rank << " d_result[" << i << "] mismatch: expected "
                              << kExpectedResult << ", got " << pr[i] << "\n";
                    break;
                }
            }
        }
        if (!compute_ok) break;

        // ReLU bitmask: all zeros (negative input)
        {
            Tensor h_mask = task.fetch_from_rank(d_mask, rank);
            const uint8_t* pm = static_cast<const uint8_t*>(h_mask.data<void>());
            for (int i = 0; i < static_cast<int>(h_mask.numel()); ++i) {
                if (pm[i] != 0) {
                    compute_ok = false;
                    std::cerr << "Rank " << rank << " d_mask[" << i << "] mismatch: expected 0, got "
                              << static_cast<int>(pm[i]) << "\n";
                    break;
                }
            }
        }
        if (!compute_ok) break;
    }

    // ------------------------------------------------------------------
    //  Step 9: Verify transfer overwrote pre-filled -1.0 with 10.0 on ALL ranks
    // ------------------------------------------------------------------
    bool xfer_ok = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_buf_b = task.fetch_from_rank(d_buf_b, rank);
        const float* pb = h_buf_b.data<float>();
        const float expected = 10.0f;

        for (int i = 0; i < static_cast<int>(h_buf_b.numel()); ++i) {
            if (std::fabs(pb[i] - expected) > kTolerance) {
                xfer_ok = false;
                std::cerr << "Rank " << rank << " d_buf_b[" << i << "] mismatch: expected "
                          << expected << ", got " << pb[i] << "\n";
                break;
            }
        }
        if (!xfer_ok) break;
    }

    // ------------------------------------------------------------------
    //  Step 10: Report
    // ------------------------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << "Dual Graph Test: Compute + Transfer Overlap\n";
    std::cout << "========================================\n";
    std::cout << "GPU Configuration: " << num_ranks << " rank(s)\n";
    if (num_ranks > 1 && visible_gpu_count >= 8) {
        std::cout << "Physical GPUs [1,2,4,7]\n";
        std::cout << "Rank Mapping:\n";
        for (int r = 0; r < num_ranks; ++r) {
            std::cout << "  rank " << r << " -> physical GPU " << kPhysicalGpuIds[r] << "\n";
        }
    }
    std::cout << "----------------------------------------\n";
    std::cout << "Compute Pipeline (AXPY -> ReLU): " << (compute_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "  Verified on all " << num_ranks << " rank(s): d_temp, d_mask, d_result\n";
    std::cout << "  A = " << kInputA << ", W = " << kWeight << ", alpha = " << kAlpha << "\n";
    std::cout << "  d_temp  = " << kAlpha << " * (" << kInputA << ") + " << kWeight
              << " = " << kExpectedTemp << "\n";
    std::cout << "  d_result = ReLU(" << kExpectedTemp << ") = " << kExpectedResult << "\n";
    std::cout << "\nTransfer (Async H2D via RANGE_H2D_COPY_DTENSOR): " << (xfer_ok ? "PASS" : "FAIL") << "\n";
    if (xfer_ok) {
        std::cout << "  Verified on all " << num_ranks << " rank(s): pre-filled -1.0 -> 10.0\n";
    }
    std::cout << "========================================\n\n";

    return (compute_ok && xfer_ok) ? 0 : 1;
}
