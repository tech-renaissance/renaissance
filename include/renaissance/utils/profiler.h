/**
 * @file profiler.h
 * @brief 性能分析器类声明
 * @details 性能分析器，用于方便计时和运算性能分析。
 * @version 3.6.10
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: 标准库
 * @note 所属系列: utils
 */

#pragma once

#include <chrono>
#include <string>

namespace tr {

// 前向声明
class Shape;

/**
 * @class Profiler
 * @brief 性能分析器类，用于计时和性能分析
 */
class Profiler {
public:
    Profiler();
    virtual ~Profiler();

    /**
     * @brief 启动计时器
     */
    void start();

    /**
     * @brief 停止计时器
     */
    void stop();

    /**
     * @brief 获取平均时间（毫秒）
     * @return 平均时间
     */
    double avg_time() const;

    /**
     * @brief 设置迭代次数
     * @param iterations 迭代次数
     */
    void set_iterations(int iterations);

    /**
     * @brief 获取总时间（毫秒）
     * @return 总时间
     */
    double total_time() const;

    /**
     * @brief 描述操作类型并计算FLOPs
     * @param operation_type 操作类型（如"mm", "conv_k3_s1_p1"等）
     * @param shape_a 输入张量a的形状
     * @param shape_b 输入张量b的形状
     */
    void describe_operation(const std::string& operation_type, Shape shape_a, Shape shape_b);

    /**
     * @brief 获取性能（GFLOPS）
     * @return 性能值
     */
    double get_performance() const;

private:
    bool timer_started_;                                           ///< 计时器是否已启动
    int iterations_;                                               ///< 迭代次数
    std::chrono::time_point<std::chrono::steady_clock> start_time_; ///< 开始时间
    std::chrono::time_point<std::chrono::steady_clock> end_time_;   ///< 结束时间
    double total_;                                                 ///< 总时间（毫秒）
    long long flops_;                                             ///< 浮点运算次数
};

} // namespace tr
