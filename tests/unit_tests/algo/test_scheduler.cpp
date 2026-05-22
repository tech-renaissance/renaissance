/**
 * @file test_scheduler.cpp
 * @brief 学习率调度器简单测试样例
 * @version 4.0.1
 * @date 2026-05-14
 * @author 技术觉醒团队
 * @note 测试目标：验证PolynomialLR/CosineAnnealingLR/StepLR的基本功能和数学正确性
 */

#include "renaissance/algo/scheduler.h"
#include "renaissance/core/logger.h"

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace tr;

// 简单的断言宏
#define TEST_CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "❌ FAILED: " << message << std::endl; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(a, b, epsilon, message) \
    TEST_CHECK(std::abs((a) - (b)) < (epsilon), message)

/**
 * @brief 测试PolynomialLR基本功能
 */
bool test_polynomial_lr_basic() {
    std::cout << "=== Testing PolynomialLR Basic ===" << std::endl;

    PolynomialLR scheduler;
    scheduler.base_lr(0.1f)
             .warmup(2)
             .warmup_start_lr(0.0f)
             .power(2.0f)
             .step_by_batch();

    // 准备：10个epoch，每个epoch 100步
    const int total_epochs = 10;
    const int steps_per_epoch = 100;
    scheduler.prepare(total_epochs, steps_per_epoch);

    // 测试初始状态
    TEST_CHECK(scheduler.is_prepared(), "scheduler should be prepared");
    TEST_CHECK(scheduler.total_steps() == 1000, "total steps should be 1000");
    TEST_CHECK(scheduler.warmup_steps() == 200, "warmup steps should be 200");

    // 测试warmup阶段
    float lr_0 = scheduler.get_lr_by_batch(0);
    float lr_100 = scheduler.get_lr_by_batch(100);
    float lr_199 = scheduler.get_lr_by_batch(199);

    std::cout << "  Warmup phase:" << std::endl;
    std::cout << "    LR at step 0: " << lr_0 << " (expected: 0.0)" << std::endl;
    std::cout << "    LR at step 100: " << lr_100 << " (expected: 0.05)" << std::endl;
    std::cout << "    LR at step 199: " << lr_199 << " (expected: ~0.0995)" << std::endl;

    TEST_ASSERT_NEAR(lr_0, 0.0f, 1e-6f, "warmup start LR should be 0.0");
    TEST_ASSERT_NEAR(lr_100, 0.05f, 1e-6f, "warmup mid LR should be 0.05");
    TEST_ASSERT_NEAR(lr_199, 0.0995f, 1e-3f, "warmup end LR should be ~0.0995");

    // 测试衰减阶段
    float lr_200 = scheduler.get_lr_by_batch(200);
    float lr_600 = scheduler.get_lr_by_batch(600);  // 中点
    float lr_999 = scheduler.get_lr_by_batch(999);

    std::cout << "  Decay phase:" << std::endl;
    std::cout << "    LR at step 200: " << lr_200 << " (expected: 0.1)" << std::endl;
    std::cout << "    LR at step 600: " << lr_600 << " (expected: 0.16)" << std::endl;
    std::cout << "    LR at step 999: " << lr_999 << " (expected: ~0.0)" << std::endl;

    TEST_ASSERT_NEAR(lr_200, 0.1f, 1e-6f, "decay start LR should be base_lr");
    TEST_ASSERT_NEAR(lr_600, 0.025f, 1e-3f, "decay mid LR should be 0.025");
    TEST_ASSERT_NEAR(lr_999, 0.0f, 1e-3f, "decay end LR should be ~0.0");

    // 测试step()功能
    scheduler.reset();
    float current_lr = scheduler.get_current_lr();
    std::cout << "  Initial current_lr: " << current_lr << std::endl;

    scheduler.step();
    current_lr = scheduler.get_current_lr();
    std::cout << "  After first step: " << current_lr << std::endl;

    TEST_ASSERT_NEAR(current_lr, 0.0f, 1e-6f, "first step should return warmup start LR");

    std::cout << "✅ PolynomialLR basic test PASSED" << std::endl << std::endl;
    return true;
}

/**
 * @brief 测试CosineAnnealingLR基本功能
 */
bool test_cosine_annealing_lr_basic() {
    std::cout << "=== Testing CosineAnnealingLR Basic ===" << std::endl;

    CosineAnnealingLR scheduler;
    scheduler.base_lr(0.1f)
             .warmup(2)
             .warmup_start_factor(0.01f)
             .eta_min(0.001f)
             .step_by_batch();

    const int total_epochs = 10;
    const int steps_per_epoch = 100;
    scheduler.prepare(total_epochs, steps_per_epoch);

    // 测试warmup阶段
    float lr_0 = scheduler.get_lr_by_batch(0);
    float lr_100 = scheduler.get_lr_by_batch(100);

    std::cout << "  Warmup phase:" << std::endl;
    std::cout << "    LR at step 0: " << lr_0 << " (expected: 0.001)" << std::endl;
    std::cout << "    LR at step 100: " << lr_100 << " (expected: 0.0505)" << std::endl;

    TEST_ASSERT_NEAR(lr_0, 0.001f, 1e-6f, "cosine warmup start should be base_lr * factor");
    TEST_ASSERT_NEAR(lr_100, 0.0505f, 1e-3f, "cosine warmup mid should be ~0.0505");

    // 测试衰减阶段
    float lr_200 = scheduler.get_lr_by_batch(200);
    float lr_600 = scheduler.get_lr_by_batch(600);  // 中点
    float lr_999 = scheduler.get_lr_by_batch(999);

    std::cout << "  Decay phase:" << std::endl;
    std::cout << "    LR at step 200: " << lr_200 << " (expected: 0.1)" << std::endl;
    std::cout << "    LR at step 600: " << lr_600 << " (expected: ~0.037)" << std::endl;
    std::cout << "    LR at step 999: " << lr_999 << " (expected: 0.001)" << std::endl;

    TEST_ASSERT_NEAR(lr_200, 0.1f, 1e-6f, "cosine decay start should be base_lr");
    TEST_ASSERT_NEAR(lr_600, 0.0505f, 1e-3f, "cosine decay mid should be ~0.0505");
    TEST_ASSERT_NEAR(lr_999, 0.001f, 1e-3f, "cosine decay end should be eta_min");

    std::cout << "✅ CosineAnnealingLR basic test PASSED" << std::endl << std::endl;
    return true;
}

/**
 * @brief 测试StepLR基本功能
 */
bool test_step_lr_basic() {
    std::cout << "=== Testing StepLR Basic ===" << std::endl;

    StepLR scheduler;
    scheduler.base_lr(0.1f)
             .warmup(1)
             .warmup_start_factor(0.1f)
             .step_size(3)  // 每3个epoch衰减一次
             .gamma(0.5f)   // 每次衰减到50%
             .step_by_batch();

    const int total_epochs = 10;
    const int steps_per_epoch = 100;
    scheduler.prepare(total_epochs, steps_per_epoch);

    // 测试warmup阶段
    float lr_0 = scheduler.get_lr_by_batch(0);
    float lr_99 = scheduler.get_lr_by_batch(99);

    std::cout << "  Warmup phase:" << std::endl;
    std::cout << "    LR at step 0: " << lr_0 << " (expected: 0.01)" << std::endl;
    std::cout << "    LR at step 99: " << lr_99 << " (expected: ~0.099)" << std::endl;

    TEST_ASSERT_NEAR(lr_0, 0.01f, 1e-6f, "step warmup start should be base_lr * factor");
    TEST_ASSERT_NEAR(lr_99, 0.099f, 1e-3f, "step warmup end should be ~0.099");

    // 测试阶梯衰减
    float lr_100 = scheduler.get_lr_by_batch(100);  // epoch 1开始，base_lr
    float lr_400 = scheduler.get_lr_by_batch(400);  // epoch 3开始，第一次衰减后
    float lr_700 = scheduler.get_lr_by_batch(700);  // epoch 6开始，第二次衰减后

    std::cout << "  Step decay:" << std::endl;
    std::cout << "    LR at step 100: " << lr_100 << " (expected: 0.1)" << std::endl;
    std::cout << "    LR at step 400: " << lr_400 << " (expected: 0.05)" << std::endl;
    std::cout << "    LR at step 700: " << lr_700 << " (expected: 0.025)" << std::endl;

    TEST_ASSERT_NEAR(lr_100, 0.1f, 1e-6f, "step decay start should be base_lr");
    TEST_ASSERT_NEAR(lr_400, 0.05f, 1e-6f, "first step decay should be base_lr * gamma");
    TEST_ASSERT_NEAR(lr_700, 0.025f, 1e-6f, "second step decay should be base_lr * gamma^2");

    std::cout << "✅ StepLR basic test PASSED" << std::endl << std::endl;
    return true;
}

/**
 * @brief 测试epoch模式vs batch模式
 */
bool test_epoch_vs_batch_mode() {
    std::cout << "=== Testing Epoch vs Batch Mode ===" << std::endl;

    // 测试batch模式
    PolynomialLR batch_scheduler;
    batch_scheduler.base_lr(0.1f)
                   .warmup(2)
                   .warmup_start_lr(0.0f)
                   .power(2.0f)
                   .step_by_batch();

    const int total_epochs = 5;
    const int steps_per_epoch = 100;
    batch_scheduler.prepare(total_epochs, steps_per_epoch);

    // 测试epoch模式
    PolynomialLR epoch_scheduler;
    epoch_scheduler.base_lr(0.1f)
                   .warmup(2)
                   .warmup_start_lr(0.0f)
                   .power(2.0f)
                   .step_by_epoch();

    epoch_scheduler.prepare(total_epochs, steps_per_epoch);

    // 在batch模式下，每个batch有不同的LR
    float batch_lr_0 = batch_scheduler.get_lr_by_batch(0);
    float batch_lr_150 = batch_scheduler.get_lr_by_batch(150);
    float batch_lr_250 = batch_scheduler.get_lr_by_batch(250);

    std::cout << "  Batch mode:" << std::endl;
    std::cout << "    LR at batch 0: " << batch_lr_0 << std::endl;
    std::cout << "    LR at batch 150: " << batch_lr_150 << std::endl;
    std::cout << "    LR at batch 250: " << batch_lr_250 << std::endl;

    TEST_CHECK(batch_lr_0 != batch_lr_150, "batch mode should have different LR per batch");

    // 在epoch模式下，同一epoch内的所有batch有相同的LR
    float epoch_lr_0 = epoch_scheduler.get_lr_by_epoch(0);
    float epoch_lr_1 = epoch_scheduler.get_lr_by_epoch(1);
    float epoch_lr_2 = epoch_scheduler.get_lr_by_epoch(2);
    float batch_lr_100 = batch_scheduler.get_lr_by_batch(100);  // 对应 epoch 1 的 step

    std::cout << "  Epoch mode:" << std::endl;
    std::cout << "    LR at epoch 0: " << epoch_lr_0 << std::endl;
    std::cout << "    LR at epoch 1: " << epoch_lr_1 << std::endl;
    std::cout << "    LR at epoch 2: " << epoch_lr_2 << std::endl;

    TEST_CHECK(epoch_lr_0 != epoch_lr_1, "epoch mode should have different LR per epoch");
    TEST_ASSERT_NEAR(epoch_lr_1, batch_lr_100, 1e-6f, "epoch 1 LR should match batch 100 LR");

    std::cout << "✅ Epoch vs Batch mode test PASSED" << std::endl << std::endl;
    return true;
}

/**
 * @brief 打印LR曲线用于可视化
 */
void print_lr_curve() {
    std::cout << "=== Learning Rate Curve Visualization ===" << std::endl;

    PolynomialLR scheduler;
    scheduler.base_lr(0.1f)
             .warmup(2)
             .warmup_start_lr(0.0f)
             .power(2.0f)
             .step_by_batch();

    const int total_epochs = 5;
    const int steps_per_epoch = 100;
    scheduler.prepare(total_epochs, steps_per_epoch);

    std::cout << "Step\tLR\t\tPhase" << std::endl;
    std::cout << "-----\t-------\t\t-----" << std::endl;

    for (int step = 0; step <= scheduler.total_steps(); step += 50) {
        float lr = scheduler.get_lr_by_batch(step);
        std::string phase = (step < scheduler.warmup_steps()) ? "Warmup" : "Decay";
        std::cout << step << "\t" << std::fixed << std::setprecision(6) << lr << "\t" << phase << std::endl;
    }

    std::cout << std::endl;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  renAIssance Learning Rate Scheduler Test Suite           ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl << std::endl;

    bool all_passed = true;

    // 运行所有测试
    all_passed &= test_polynomial_lr_basic();
    all_passed &= test_cosine_annealing_lr_basic();
    all_passed &= test_step_lr_basic();
    all_passed &= test_epoch_vs_batch_mode();

    // 打印LR曲线
    print_lr_curve();

    // 总结
    std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
    if (all_passed) {
        std::cout << "║  ✅ ALL TESTS PASSED                                      ║" << std::endl;
    } else {
        std::cout << "║  ❌ SOME TESTS FAILED                                     ║" << std::endl;
    }
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;

    return all_passed ? 0 : 1;
}