/**
 * @file example_axpy.cpp
 * @brief Simple AXPY example: C = alpha * A + B with adaptive GPU configuration
 * @version 4.20.1
 * @date 2026-04-20
 * @author Team Tech-Renaissance
 */

#include "renaissance.h"
#include <iostream>
#include <cmath>

using namespace tr;

namespace {

// --- Constants ---
constexpr float kAlpha     = 2.0f;
constexpr float kInputA    = 1.0f;
constexpr float kInputB    = 3.0f;
constexpr float kExpected  = kAlpha * kInputA + kInputB;  // 5.0
constexpr float kTolerance = 1e-5f;

constexpr int kShapeDim = 1024;
const Shape kShape{1, kShapeDim, 1, 1};

// Physical GPU mapping used when 8 visible GPUs are present
constexpr int kPhysicalGpuIds[] = {1, 2, 4, 7};

} // anonymous namespace

int main() {
    // ------------------------------------------------------------------
    //  Step 1: Detect visible GPUs and configure the runtime
    // ------------------------------------------------------------------
    const int visible_gpu_count = GlobalRegistry::get_visible_gpu_count();

    if (visible_gpu_count >= 8) {
        // 8-GPU system: use GPUs 1,2,4,7 for 4-rank parallel test
        GLOBAL_SETTING.use_gpu("1,2,4,7").auto_seed();
        std::cout << "\n========================================\n";
        std::cout << "GPU Configuration: 8-GPU system detected\n";
        std::cout << "Using GPUs [1,2,4,7] for 4-card parallel test\n";
        std::cout << "Rank Mapping:\n";
        for (int r = 0; r < 4; ++r) {
            std::cout << "  rank " << r << " -> physical GPU " << kPhysicalGpuIds[r] << "\n";
        }
    } else if (visible_gpu_count > 0) {
        // Fewer than 8 GPUs: single card on GPU 0
        GLOBAL_SETTING.use_gpu("0").auto_seed();
        std::cout << "\n========================================\n";
        std::cout << "GPU Configuration: " << visible_gpu_count << "-GPU(s) detected\n";
        std::cout << "Using GPU 0 for single-card test\n";
    } else {
        // No GPU: fall back to CPU
        GLOBAL_SETTING.use_cpu().auto_seed();
        std::cout << "\n========================================\n";
        std::cout << "No GPU detected, using CPU mode\n";
    }

    const int num_ranks = (visible_gpu_count >= 8) ? 4 : 1; // actual number of participating ranks

    // ------------------------------------------------------------------
    //  Step 2: Plan & draft the computation graph
    // ------------------------------------------------------------------
    SimpleTask task;

    DTensor d_a = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);
    DTensor d_b = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);
    DTensor d_c = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);
    task.finalize_memory();  // lock layout - no further allocations allowed

    ComputationGraph graph;
    graph.append(ComputeOp::AXPY_FWD,
                 {d_a.id, d_b.id},
                 {d_c.id},
                 OpParams{AxpyParams{kAlpha}});  // C = alpha * A + B
    task.add_graph("axpy", std::move(graph));

    // ------------------------------------------------------------------
    //  Step 3: Compile for real hardware
    // ------------------------------------------------------------------
    task.compile();

    // ------------------------------------------------------------------
    //  Step 4: Prepare host data and transfer to device(s)
    // ------------------------------------------------------------------
    Tensor h_a(kShape, DType::FP32);
    Tensor h_b(kShape, DType::FP32);

    float* pa = h_a.data<float>();
    float* pb = h_b.data<float>();
    for (int i = 0; i < static_cast<int>(h_a.numel()); ++i) {
        pa[i] = kInputA;
        pb[i] = kInputB;
    }

    // Recommended: separate transfer - send to rank 0 first, then broadcast
    task.transfer_to_rank(h_a, d_a, 0);   // H2D -> rank 0
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_a);   // NCCL broadcast to all ranks
    }
    task.transfer_to_rank(h_b, d_b, 0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_b);
    }

    // Alternative (commented): combined transfer that does H2D + broadcast in one call
    // task.transfer(h_a, d_a);
    // task.transfer(h_b, d_b);

    // ------------------------------------------------------------------
    //  Step 5: Execute and verify
    // ------------------------------------------------------------------
    task.run("axpy");

    bool all_passed = true;

    std::cout << "Verifying results from all " << num_ranks << " rank(s):\n";

    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_c = task.fetch_from_rank(d_c, rank);
        const float* pc = h_c.data<float>();

        bool rank_passed = true;
        for (int i = 0; i < static_cast<int>(h_c.numel()); ++i) {
            if (std::fabs(pc[i] - kExpected) > kTolerance) {
                rank_passed = false;
                all_passed = false;

                if (num_ranks > 1 && visible_gpu_count >= 8) {
                    std::cerr << "  Rank " << rank << " (physical GPU " << kPhysicalGpuIds[rank]
                              << "): FAIL - element[" << i << "] = " << pc[i]
                              << " (expected " << kExpected << ")\n";
                } else {
                    std::cerr << "  Rank " << rank
                              << ": FAIL - element[" << i << "] = " << pc[i]
                              << " (expected " << kExpected << ")\n";
                }
                break;
            }
        }

        if (rank_passed) {
            if (num_ranks > 1 && visible_gpu_count >= 8) {
                std::cout << "  Rank " << rank << " (physical GPU " << kPhysicalGpuIds[rank]
                          << "): PASS - all elements = " << pc[0] << "\n";
            } else {
                std::cout << "  Rank " << rank
                          << ": PASS - all elements = " << pc[0] << "\n";
            }
        }
    }

    // ------------------------------------------------------------------
    //  Step 6: Report final result
    // ------------------------------------------------------------------
    std::cout << "----------------------------------------\n";
    std::cout << "Computation: C = alpha * A + B = "
              << kAlpha << " * " << kInputA << " + " << kInputB << " = " << kExpected << "\n";
    std::cout << "Final Result: " << (all_passed ? "PASS" : "FAIL") << "\n";
    std::cout << "========================================\n\n";

    return all_passed ? 0 : 1;
}
