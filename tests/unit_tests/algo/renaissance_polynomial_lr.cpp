/**
 * @file renaissance_polynomial_lr.cpp
 * @brief renAIssance框架的PolynomialLR调度器输出
 * @version 4.0.1
 * @date 2026-05-14
 * @author 技术觉醒团队
 * @note 完全按照CLOSED.py第354-366行的配置，输出每个batch的学习率到txt
 */

#include "renaissance/algo/scheduler.h"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace tr;

int main() {
    // === MLPerf标准配置（CLOSED.py第19-26行）===
    const float REFERENCE_LR = 6.300f;
    const float REFERENCE_PER_GPU_BATCH = 256.0f;
    const int WARMUP_EPOCHS = 3;
    const int NUM_EPOCHS = 34;
    const int PER_GPU_BATCH = 512;
    const float WARMUP_START_LR = 0.0f;

    // 线性缩放学习率（CLOSED.py第227行）
    float poly_base_lr = REFERENCE_LR * (PER_GPU_BATCH / REFERENCE_PER_GPU_BATCH);

    const int updates_per_epoch = 100;
    const int total_updates = NUM_EPOCHS * updates_per_epoch;
    const int warmup_updates = WARMUP_EPOCHS * updates_per_epoch;

    std::cout << "配置参数:" << std::endl;
    std::cout << "  NUM_EPOCHS: " << NUM_EPOCHS << std::endl;
    std::cout << "  WARMUP_EPOCHS: " << WARMUP_EPOCHS << std::endl;
    std::cout << "  poly_base_lr: " << poly_base_lr << std::endl;
    std::cout << "  WARMUP_START_LR: " << WARMUP_START_LR << std::endl;
    std::cout << "  total_updates: " << total_updates << std::endl;
    std::cout << "  warmup_updates: " << warmup_updates << std::endl;

    // 创建renAIssance的PolynomialLR调度器
    PolynomialLR scheduler;
    scheduler.base_lr(poly_base_lr)
             .warmup(WARMUP_EPOCHS)
             .warmup_start_lr(WARMUP_START_LR)
             .power(2.0f)
             .step_by_batch();
    scheduler.prepare(NUM_EPOCHS, updates_per_epoch);

    std::cout << "\nrenAIssance PolynomialLR创建完成" << std::endl;

    // 输出到文件
    std::ofstream outfile("renaissance_lr_output.txt");
    outfile << "renAIssance PolynomialLR学习率输出（与CLOSED.py完全一致的配置）\n";
    outfile << "========================================================================\n";
    outfile << "配置: total_updates=" << total_updates << ", warmup_updates=" << warmup_updates
            << ", base_lr=" << poly_base_lr << ", start_lr=" << WARMUP_START_LR << "\n";
    outfile << "========================================================================\n";
    outfile << std::left << std::setw(10) << "batch_id"
            << std::setw(10) << "phase"
            << std::setw(15) << "lr" << std::endl;
    outfile << "-" << std::endl;

    std::cout << "开始计算每个batch的学习率..." << std::endl;

    // 计算每个batch的学习率（与Python脚本完全相同的范围）
    for (int batch_id = 0; batch_id < total_updates + 100; batch_id++) {
        float lr = scheduler.get_lr_by_batch(batch_id);

        std::string phase = (batch_id < warmup_updates) ? "Warmup" : "Decay";

        // 写入文件（格式与Python脚本完全一致）
        outfile << std::left << std::setw(10) << batch_id
                << std::setw(10) << phase
                << std::fixed << std::setprecision(10) << std::setw(15) << lr << std::endl;

        // 每100个batch打印一次进度
        if (batch_id % 100 == 0) {
            std::cout << "  已处理batch " << batch_id << "/" << (total_updates + 100) << std::endl;
        }
    }

    outfile.close();

    std::cout << "\n✅ renAIssance学习率已输出到 renaissance_lr_output.txt" << std::endl;
    std::cout << "   输出格式：batch_id phase lr（共" << (total_updates + 100) << "行）" << std::endl;

    return 0;
}