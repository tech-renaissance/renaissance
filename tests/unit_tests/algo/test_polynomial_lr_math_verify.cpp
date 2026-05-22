/**
 * @file test_polynomial_lr_math_verify.cpp
 * @brief PolynomialLR数学精确性验证：与CLOSED.py逐行对比
 * @version 4.0.1
 * @date 2026-05-14
 * @author 技术觉醒团队
 * @note 验证目标：在每个last_epoch上，我们的实现与CLOSED.py的get_lr()返回完全相同的值
 */

#include "renaissance/algo/scheduler.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <sstream>

using namespace tr;

/**
 * @brief CLOSED.py的PolynomialLR.get_lr()的忠实复现
 *
 * 这个函数完全按照CLOSED.py第169-194行的逻辑实现，用于逐行验证我们的C++实现
 */
float closed_py_polynomial_lr_get_lr(
    int last_epoch,
    int total_steps,
    int warmup_steps,
    float base_lr,
    float start_lr,
    float power
) {
    if (last_epoch < warmup_steps) {
        // === Warmup 阶段：线性增长 ===
        float alpha;
        if (warmup_steps > 0) {
            alpha = static_cast<float>(last_epoch) / static_cast<float>(warmup_steps);
        } else {
            alpha = 1.0f;
        }
        return start_lr + (base_lr - start_lr) * alpha;
    } else {
        // === Polynomial Decay 阶段 ===
        int current_decay_step = last_epoch - warmup_steps;
        int total_decay_steps = total_steps - warmup_steps;

        float progress;
        if (total_decay_steps > 0) {
            progress = static_cast<float>(current_decay_step) / static_cast<float>(total_decay_steps);
            if (progress > 1.0f) progress = 1.0f;  // 防止超调
        } else {
            progress = 1.0f;
        }

        // 标准 polynomial decay: lr = base_lr × (1 - progress)^power
        float scale = std::pow(1.0f - progress, power);
        return base_lr * scale;
    }
}

/**
 * @brief 逐行对比验证
 */
bool verify_polynomial_lr_math() {
    std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  PolynomialLR 数学精确性验证 vs CLOSED.py                 ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl << std::endl;

    // === MLPerf标准配置（CLOSED.py第19-26行）===
    const int NUM_EPOCHS = 34;
    const int WARMUP_EPOCHS = 3;
    const int PER_GPU_BATCH = 512;
    const int REFERENCE_PER_GPU_BATCH = 256;
    const float REFERENCE_LR = 6.300f;
    const float WARMUP_START_LR = 0.0f;
    const float power = 2.0f;

    // 计算线性缩放的学习率（CLOSED.py第227行）
    float poly_base_lr = REFERENCE_LR * (static_cast<float>(PER_GPU_BATCH) / static_cast<float>(REFERENCE_PER_GPU_BATCH));

    std::cout << "=== MLPerf Standard Configuration ===" << std::endl;
    std::cout << "NUM_EPOCHS: " << NUM_EPOCHS << std::endl;
    std::cout << "WARMUP_EPOCHS: " << WARMUP_EPOCHS << std::endl;
    std::cout << "PER_GPU_BATCH: " << PER_GPU_BATCH << std::endl;
    std::cout << "poly_base_lr: " << poly_base_lr << " (linear scaling)" << std::endl;
    std::cout << "WARMUP_START_LR: " << WARMUP_START_LR << std::endl;
    std::cout << "power: " << power << std::endl << std::endl;

    // 假设每个epoch有100个update（batch）
    const int updates_per_epoch = 100;
    const int total_steps = NUM_EPOCHS * updates_per_epoch;
    const int warmup_steps = WARMUP_EPOCHS * updates_per_epoch;

    std::cout << "=== Step Configuration ===" << std::endl;
    std::cout << "updates_per_epoch: " << updates_per_epoch << std::endl;
    std::cout << "total_steps: " << total_steps << std::endl;
    std::cout << "warmup_steps: " << warmup_steps << std::endl << std::endl;

    // 创建我们的调度器
    PolynomialLR scheduler;
    scheduler.base_lr(poly_base_lr)
             .warmup(WARMUP_EPOCHS)
             .warmup_start_lr(WARMUP_START_LR)
             .power(power)
             .step_by_batch();
    scheduler.prepare(NUM_EPOCHS, updates_per_epoch);

    std::cout << "╔══════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  last_epoch  │   CLOSED.py   │  Our Impl     │   Diff        │ Status ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣" << std::endl;

    bool all_match = true;

    // 选择关键测试点
    std::vector<int> test_steps = {
        0,          // warmup起点
        50,         // warmup中点
        150,        // warmup终点前
        299,        // warmup最后一步
        300,        // decay起点
        850,        // decay中点
        1200,       // decay后期
        1699,       // decay最后一步
        1700,       // 超出范围（应被clamp）
        2000        // 更大超出范围
    };

    for (int last_epoch : test_steps) {
        // CLOSED.py的实现
        float closed_lr = closed_py_polynomial_lr_get_lr(
            last_epoch, total_steps, warmup_steps, poly_base_lr, WARMUP_START_LR, power
        );

        // 我们的实现
        float our_lr = scheduler.get_lr_by_batch(last_epoch);

        // 计算差异
        float diff = std::abs(closed_lr - our_lr);
        bool match = diff < 1e-6f;

        // 确定阶段
        std::string phase = (last_epoch < warmup_steps) ? "Warmup" : "Decay";

        // 格式化输出
        std::cout << "║   " << std::setw(4) << last_epoch
                  << "    │  " << std::fixed << std::setprecision(8) << closed_lr
                  << "  │  " << our_lr
                  << "  │  " << std::scientific << std::setprecision(2) << diff
                  << "  │ " << (match ? "✅" : "❌") << " " << phase << " ║" << std::endl;

        all_match &= match;
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════╝" << std::endl << std::endl;

    // === 特殊数学验证点 ===
    std::cout << "=== Special Mathematical Verification Points ===" << std::endl;

    // 1. warmup起点：last_epoch=0
    float lr_0_closed = closed_py_polynomial_lr_get_lr(0, total_steps, warmup_steps, poly_base_lr, WARMUP_START_LR, power);
    float lr_0_our = scheduler.get_lr_by_batch(0);
    std::cout << "Point 1 (warmup start): " << std::fixed << std::setprecision(8)
              << "CLOSED.py=" << lr_0_closed << ", Our=" << lr_0_our
              << ", Diff=" << std::abs(lr_0_closed - lr_0_our) << std::endl;
    std::cout << "  Expected: " << WARMUP_START_LR << " (WARMUP_START_LR)" << std::endl;
    bool p1_pass = std::abs(lr_0_closed - WARMUP_START_LR) < 1e-6f && std::abs(lr_0_our - WARMUP_START_LR) < 1e-6f;

    // 2. warmup终点：last_epoch=299 (warmup_steps-1)
    float lr_299_closed = closed_py_polynomial_lr_get_lr(299, total_steps, warmup_steps, poly_base_lr, WARMUP_START_LR, power);
    float lr_299_our = scheduler.get_lr_by_batch(299);
    std::cout << "Point 2 (warmup end): " << std::fixed << std::setprecision(8)
              << "CLOSED.py=" << lr_299_closed << ", Our=" << lr_299_our
              << ", Diff=" << std::abs(lr_299_closed - lr_299_our) << std::endl;
    std::cout << "  Expected: < base_lr (" << poly_base_lr << ")" << std::endl;
    bool p2_pass = std::abs(lr_299_closed - lr_299_our) < 1e-6f && lr_299_closed < poly_base_lr;

    // 3. decay起点：last_epoch=300 (warmup_steps)
    float lr_300_closed = closed_py_polynomial_lr_get_lr(300, total_steps, warmup_steps, poly_base_lr, WARMUP_START_LR, power);
    float lr_300_our = scheduler.get_lr_by_batch(300);
    std::cout << "Point 3 (decay start): " << std::fixed << std::setprecision(8)
              << "CLOSED.py=" << lr_300_closed << ", Our=" << lr_300_our
              << ", Diff=" << std::abs(lr_300_closed - lr_300_our) << std::endl;
    std::cout << "  Expected: " << poly_base_lr << " (base_lr)" << std::endl;
    bool p3_pass = std::abs(lr_300_closed - poly_base_lr) < 1e-6f && std::abs(lr_300_our - poly_base_lr) < 1e-6f;

    // 4. decay中点：last_epoch=850 (大致在总步数的中间)
    float lr_850_closed = closed_py_polynomial_lr_get_lr(850, total_steps, warmup_steps, poly_base_lr, WARMUP_START_LR, power);
    float lr_850_our = scheduler.get_lr_by_batch(850);
    std::cout << "Point 4 (decay mid): " << std::fixed << std::setprecision(8)
              << "CLOSED.py=" << lr_850_closed << ", Our=" << lr_850_our
              << ", Diff=" << std::abs(lr_850_closed - lr_850_our) << std::endl;
    // progress = (850-300)/(3400-300) = 550/3100 = 0.1774...
    // scale = (1-0.1774)^2 = 0.6772...
    // expected = 12.6 * 0.6772 = 8.534...
    float expected_850 = poly_base_lr * std::pow(1.0f - 550.0f/3100.0f, 2.0f);
    std::cout << "  Expected: " << expected_850 << " (base_lr * (1-550/3100)^2)" << std::endl;
    bool p4_pass = std::abs(lr_850_closed - expected_850) < 1e-4f && std::abs(lr_850_our - expected_850) < 1e-4f;

    // 5. decay终点：last_epoch=1699 (total_steps-1)
    float lr_1699_closed = closed_py_polynomial_lr_get_lr(1699, total_steps, warmup_steps, poly_base_lr, WARMUP_START_LR, power);
    float lr_1699_our = scheduler.get_lr_by_batch(1699);
    std::cout << "Point 5 (decay end): " << std::fixed << std::setprecision(10)
              << "CLOSED.py=" << lr_1699_closed << ", Our=" << lr_1699_our
              << ", Diff=" << std::abs(lr_1699_closed - lr_1699_our) << std::endl;
    std::cout << "  Expected: ~0.0 (approaching zero)" << std::endl;
    bool p5_pass = std::abs(lr_1699_closed) < 0.01f && std::abs(lr_1699_our) < 0.01f;

    std::cout << std::endl;

    // === 最终结论 ===
    bool math_perfect = all_match && p1_pass && p2_pass && p3_pass && p4_pass && p5_pass;

    std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
    if (math_perfect) {
        std::cout << "║  ✅ MATHEMATICAL PERFECT MATCH WITH CLOSED.PY               ║" << std::endl;
        std::cout << "║                                                              ║" << std::endl;
        std::cout << "║  Our PolynomialLR implementation is MATHEMATICALLY         ║" << std::endl;
        std::cout << "║  IDENTICAL to CLOSED.py at ALL tested steps!               ║" << std::endl;
    } else {
        std::cout << "║  ❌ MATHEMATICAL DISCREPANCY DETECTED                        ║" << std::endl;
    }
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;

    return math_perfect;
}

int main() {
    bool success = verify_polynomial_lr_math();
    return success ? 0 : 1;
}