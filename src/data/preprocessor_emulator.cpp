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
namespace data {

// =============================================================================
// 配置接口
// =============================================================================

void PreprocessorEmulator::configure(const Config& config) {
    config_ = config;
    log_dir_ = config.log_dir;

    LOG_INFO << "PreprocessorEmulator configured: "
             << "workers=" << config_.num_workers
             << ", epochs=" << config_.num_epochs
             << ", log_dir=" << config_.log_dir
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
    oss << log_dir_ << "/worker_" << worker_id << "_log.txt";
    std::string log_file_path = oss.str();

    std::ofstream log_file(log_file_path);
    if (!log_file.is_open()) {
        TR_THROW(FileNotFoundError, "Failed to open log file: " << log_file_path);
    }

    // 记录数据集名称
    log_file << "# Dataset: " << loader.dataset_name() << "\n";
    log_file << "# Worker: " << worker_id << "\n";
    log_file << "# Format: worker_id,data_size,label\n";
    log_file.flush();

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

        // 写入日志
        log_file << worker_id << "," << data_size << "," << label << "\n";
        total_samples++;
    }

    log_file.close();

    LOG_INFO << "Worker " << worker_id << " finished: "
             << total_samples << " samples processed";
}

// =============================================================================
// 写入日志
// =============================================================================

void PreprocessorEmulator::write_log(int worker_id, size_t data_size, int32_t label) {
    std::lock_guard<std::mutex> lock(log_mutex_);

    // 创建日志文件
    std::ostringstream oss;
    oss << log_dir_ << "/worker_" << worker_id << "_log.txt";

    std::ofstream log_file(oss.str(), std::ios::app);
    if (log_file.is_open()) {
        log_file << worker_id << "," << data_size << "," << label << "\n";
    }
}

// =============================================================================
// 验证可复现性
// =============================================================================

bool PreprocessorEmulator::verify_reproducibility(
    const std::string& log_dir1,
    const std::string& log_dir2) {

    LOG_INFO << "Verifying reproducibility between "
             << log_dir1 << " and " << log_dir2;

    // 读取两次运行的日志
    struct LogRecord {
        int worker_id;
        size_t data_size;
        int32_t label;
    };

    std::vector<LogRecord> logs1, logs2;

    // 辅助函数：读取日志目录
    auto read_logs = [&](const std::string& dir) -> std::vector<LogRecord> {
        std::vector<LogRecord> records;

        // TODO: 扫描目录，读取所有worker_*.txt文件
        // 当前简化实现：假设worker_0到worker_15

        for (int i = 0; i < 16; ++i) {
            std::ostringstream oss;
            oss << dir << "/worker_" << i << "_log.txt";

            std::ifstream file(oss.str());
            if (!file.is_open()) {
                LOG_WARN << "Failed to open log file: " << oss.str();
                continue;
            }

            std::string line;
            while (std::getline(file, line)) {
                // 跳过注释行
                if (line.empty() || line[0] == '#') {
                    continue;
                }

                // 解析: worker_id,data_size,label
                std::istringstream iss(line);
                char comma;
                int worker_id;
                size_t data_size;
                int label;

                if (iss >> worker_id >> comma >> data_size >> comma >> label) {
                    records.push_back({worker_id, data_size, static_cast<int32_t>(label)});
                }
            }

            file.close();
        }

        LOG_INFO << "Read " << records.size() << " records from " << dir;
        return records;
    };

    // 读取两次日志
    logs1 = read_logs(log_dir1);
    logs2 = read_logs(log_dir2);

    // 验证数量
    if (logs1.size() != logs2.size()) {
        LOG_ERROR << "Log size mismatch: " << logs1.size() << " vs " << logs2.size();
        return false;
    }

    LOG_INFO << "Comparing " << logs1.size() << " records...";

    // 逐条对比
    for (size_t i = 0; i < logs1.size(); ++i) {
        if (logs1[i].worker_id != logs2[i].worker_id ||
            logs1[i].data_size != logs2[i].data_size ||
            logs1[i].label != logs2[i].label) {

            LOG_ERROR << "Mismatch at record " << i << ": "
                     << logs1[i].worker_id << "," << logs1[i].data_size << "," << logs1[i].label
                     << " vs "
                     << logs2[i].worker_id << "," << logs2[i].data_size << "," << logs2[i].label;
            return false;
        }
    }

    LOG_INFO << "✅ Reproducibility verified! All " << logs1.size()
             << " records match perfectly.";

    return true;
}

} // namespace data
} // namespace tr
