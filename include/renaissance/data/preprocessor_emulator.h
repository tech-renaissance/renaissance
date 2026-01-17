/**
 * @file preprocessor_emulator.h
 * @brief 预处理器模拟器（用于测试）
 * @details 模拟未来的Preprocessor，用于数据加载器测试
 * @version 3.7.0
 * @date 2026-01-09
 * @author 技术觉醒团队
 * @note 依赖项: data_loader_base.h
 */

#pragma once

#include "renaissance/data/data_loader_base.h"
#include <thread>
#include <vector>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <fstream>

namespace tr {
namespace data {

/**
 * @class PreprocessorEmulator
 * @brief 预处理器模拟器（测试工具）
 *
 * 核心功能：
 * - 从DataLoader并行读取数据
 * - 模拟预处理耗时（通过sleep）
 * - 统计每个标签的样本数
 * - 保存指定线程处理的第k张图片
 *
 * 使用场景：
 * - 测试DataLoader的多线程性能
 * - 测试DataLoader的随机性
 * - 调试数据加载流程
 * - 保存样本用于验证
 */
class PreprocessorEmulator {
public:
    /**
     * @brief 构造函数
     * @param loader 数据加载器指针（非拥有）
     * @param num_workers worker线程数（必须是2的幂，范围1~64）
     * @param simulate_ms 模拟每个样本的预处理时间（毫秒，0表示不sleep）
     *
     * 参数钳制规则：
     * - num_workers超过64 → WARNING + 钳制到64
     * - num_workers不是2的幂 → 自动向下取2的幂（不报WARNING）
     */
    PreprocessorEmulator(DataLoaderBase* loader, int num_workers = 8, int simulate_ms = 0);

    ~PreprocessorEmulator();

    // 禁止拷贝
    PreprocessorEmulator(const PreprocessorEmulator&) = delete;
    PreprocessorEmulator& operator=(const PreprocessorEmulator&) = delete;

    // =========================================================================
    // 控制接口
    // =========================================================================

    /**
     * @brief 启动所有worker线程
     * @details 必须在DataLoader::begin_epoch()之后调用
     */
    void start();

    /**
     * @brief 等待所有worker完成
     * @details 会阻塞直到所有worker线程退出
     */
    void join();

    // =========================================================================
    // 统计信息
    // =========================================================================

    /**
     * @brief 获取标签统计
     * @return 标签 -> 样本数的映射
     *
     * 线程安全：可以在worker运行时调用
     */
    std::map<int32_t, size_t> get_label_counts() const;

    /**
     * @brief 获取已处理的样本总数
     */
    size_t get_total_processed() const;

    // =========================================================================
    // 保存图片
    // =========================================================================

    /**
     * @brief 设置保存参数
     * @param worker_id 要保存的worker线程ID
     * @param sample_idx 该worker处理的第几个样本（从0开始）
     * @param output_path 输出文件路径（如"output.jpeg"）
     *
     * 调用此函数后，指定worker处理到第sample_idx个样本时，会自动保存
     */
    void save_sample_image(int worker_id, int sample_idx, const std::string& output_path);

private:
    // =========================================================================
    // Worker线程函数
    // =========================================================================

    void worker_thread(int worker_id);

    // =========================================================================
    // 成员变量
    // =========================================================================

    DataLoaderBase* loader_;      // 数据加载器（非拥有）
    int num_workers_;             // worker线程数（钳制到2的幂，1~64）
    int simulate_ms_;             // 模拟处理时间（毫秒）

    std::vector<std::thread> workers_;  // worker线程池
    std::atomic<bool> stop_flag_{false};  // 停止标志

    // 统计信息
    mutable std::mutex stats_mutex_;
    std::map<int32_t, size_t> label_counts_;  // 标签 -> 样本数
    std::atomic<size_t> total_processed_{0};  // 已处理样本总数

    // 保存图片配置
    std::mutex save_mutex_;
    int save_worker_id_ = -1;     // 要保存的worker ID
    int save_sample_idx_ = -1;    // 该worker的第几个样本
    std::string save_path_;       // 保存路径
    std::atomic<bool> save_done_{false};  // 是否已保存

    // =========================================================================
    // 内部方法
    // =========================================================================

    /**
     * @brief 参数钳制（必须是2的幂）
     */
    int clamp_to_power_of_two(int n, int max_val);
};

} // namespace data
} // namespace tr
