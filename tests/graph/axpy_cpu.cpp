/**
 * @file example_axpy_cpu.cpp
 * @brief Simple AXPY example: C = alpha * A + B with forced CPU execution
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
    //  Step 3: Compile for CPU hardware
    // ------------------------------------------------------------------
    task.compile();

    // ------------------------------------------------------------------
    //  Step 4: Prepare host data and transfer to CPU
    // ------------------------------------------------------------------
    Tensor h_a(kShape, DType::FP32);
    Tensor h_b(kShape, DType::FP32);

    float* pa = h_a.data<float>();
    float* pb = h_b.data<float>();
    for (int i = 0; i < static_cast<int>(h_a.numel()); ++i) {
        pa[i] = kInputA;
        pb[i] = kInputB;
    }

    // Direct transfer to CPU (no broadcast needed for single rank)
    task.transfer(h_a, d_a);
    task.transfer(h_b, d_b);

    // ------------------------------------------------------------------
    //  Step 5: Execute and verify
    // ------------------------------------------------------------------
    task.run("axpy");

    bool all_passed = true;

    std::cout << "Verifying results from CPU:\n";

    Tensor h_c = task.fetch(d_c);  // Fetch from CPU (rank 0)
    const float* pc = h_c.data<float>();

    bool rank_passed = true;
    for (int i = 0; i < static_cast<int>(h_c.numel()); ++i) {
        if (std::fabs(pc[i] - kExpected) > kTolerance) {
            rank_passed = false;
            all_passed = false;
            std::cerr << "  CPU: FAIL - element[" << i << "] = " << pc[i]
                      << " (expected " << kExpected << ")\n";
            break;
        }
    }

    if (rank_passed) {
        std::cout << "  CPU: PASS - all elements = " << pc[0] << "\n";
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
