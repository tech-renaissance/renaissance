/**
 * @file preprocessor_emulator.h
 * @brief Preprocessor模拟器 - 用于验证随机可复现性
 * @version 3.8.0
 * @date 2026-01-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/data_loader.h"
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <fstream>

namespace tr {
namespace data {

/**
 * @class PreprocessorEmulator
 * @brief Preprocessor模拟器，用于验证DataLoader的随机可复现性
 *
 * 功能：
 * - 多线程worker模拟Preprocessor消费数据
 * - 记录每个worker读取的样本大小
 * - 输出日志用于验证可复现性
 *
 * 日志格式：worker_id,data_size
 */
class PreprocessorEmulator {
public:
    /**
     * @brief 配置参数
     */
    struct Config {
        int num_workers;           ///< Preprocessor线程数M
        int num_epochs;            ///< 运行epoch数
        std::string log_dir;       ///< 日志输出目录
        bool simulate_delay;       ///< 是否模拟预处理延迟
        uint64_t delay_us;         ///< 延迟时间（微秒）

        Config()
            : num_workers(16)
            , num_epochs(1)
            , log_dir(".")
            , simulate_delay(false)
            , delay_us(100) {}
    };

    /**
     * @brief 配置模拟器
     */
    void configure(const Config& config);

    /**
     * @brief 运行模拟
     * @param loader 数据加载器引用
     */
    void run(DataLoader& loader);

    /**
     * @brief 验证可复现性（静态方法）
     * @param log_dir1 第一次运行的日志目录
     * @param log_dir2 第二次运行的日志目录
     * @return 是否完全可复现
     */
    static bool verify_reproducibility(
        const std::string& log_dir1,
        const std::string& log_dir2);

private:
    /**
     * @brief Worker线程函数
     */
    void worker_func(int worker_id, DataLoader& loader);

    /**
     * @brief 写入日志
     */
    void write_log(int worker_id, size_t data_size, int32_t label);

    Config config_;
    std::vector<std::thread> worker_threads_;
    std::string log_dir_;
    std::mutex log_mutex_;
};

} // namespace data
} // namespace tr
