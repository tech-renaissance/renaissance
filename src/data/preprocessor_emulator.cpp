/**
 * @file preprocessor_emulator.cpp
 * @brief Preprocessor模拟器实现
 * @version 3.8.0
 * @date 2026-01-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/preprocessor_emulator.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include <sstream>
#include <iomanip>
#include <chrono>
#include <iostream>

namespace tr {

// =============================================================================
// 配置接口
// =============================================================================

void PreprocessorEmulator::configure(const Config& config) {
    config_ = config;
    // 使用TR_WORKSPACE宏作为日志目录
    log_dir_ = TR_WORKSPACE;

    LOG_INFO << "PreprocessorEmulator configured: "
             << "workers=" << config_.num_workers
             << ", epochs=" << config_.num_epochs
             << ", log_dir=" << log_dir_
             << ", simulate_delay=" << config_.simulate_delay;
}

// =============================================================================
// 运行模拟
// =============================================================================

void PreprocessorEmulator::run(DataLoader& loader) {
    LOG_INFO << "Starting PreprocessorEmulator...";

    // 清理旧日志
    // TODO: 实现日志目录清理

    // 启动worker线程
    worker_threads_.clear();
    for (int i = 0; i < config_.num_workers; ++i) {
        worker_threads_.emplace_back(
            &PreprocessorEmulator::worker_func, this, i, std::ref(loader));
    }

    // 等待所有worker完成
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    worker_threads_.clear();

    LOG_INFO << "PreprocessorEmulator completed";
}

// =============================================================================
// Worker线程函数
// =============================================================================

void PreprocessorEmulator::worker_func(int worker_id, DataLoader& loader) {
    LOG_INFO << "Worker " << worker_id << " started";

    // 打开日志文件
    std::ostringstream oss;
    oss << log_dir_ << "/worker_" << worker_id << ".log";
    std::string log_file_path = oss.str();

    std::ofstream log_file(log_file_path);
    if (!log_file.is_open()) {
        TR_THROW(FileNotFoundError, "Failed to open log file: " << log_file_path);
    }

    // 消费样本
    int32_t label;
    const uint8_t* data_ptr;
    size_t data_size;
    size_t total_samples = 0;

    while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
        // 模拟预处理延迟（可选）
        if (config_.simulate_delay) {
            std::this_thread::sleep_for(std::chrono::microseconds(config_.delay_us));
        }

        // 写入日志：worker_id,size,label
        log_file << worker_id << "," << data_size << "," << label << "\n";
        total_samples++;
    }

    log_file.close();

    LOG_INFO << "Worker " << worker_id << " finished: "
             << total_samples << " samples processed";
}

} // namespace tr
