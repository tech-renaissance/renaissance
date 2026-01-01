/**
 * @file logger.h
 * @brief 日志器类声明
 * @details 轻量级、线程安全、可编译关闭的日志器，支持四级日志（DEBUG/INFO/WARN/ERROR）和模块化标记
 * @version 3.5.5
 * @date 2025-12-24
 * @author 技术觉醒团队
 * @note 依赖项:
 * @note 所属系列: base
 */

#pragma once

#include <string>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>

namespace tr {

// 前向声明
class TRException;

// 日志等级枚举
enum class LogLevel {
    DEBUG = 0,
    INFO  = 1,
    WARN  = 2,
    ERROR = 3
};

/**
 * @class Logger
 * @brief 日志器单例类
 * @details 提供线程安全的日志输出功能，支持控制台、文件输出和运行时级别控制
 */
class Logger {
public:
    /**
     * @brief 获取Logger单例实例
     * @return Logger单例引用
     */
    static Logger& instance() noexcept;

    /**
     * @brief 设置日志级别
     * @param level 日志级别，低于此级别的日志将被过滤
     */
    void set_level(LogLevel level) noexcept;

    /**
     * @brief 设置日志输出文件
     * @param filename 日志文件路径，空字符串表示输出到控制台
     */
    void set_output_file(const std::string& filename);

    /**
     * @brief 设置静默模式
     * @param quiet true表示不输出任何日志，false表示正常输出
     */
    void set_quiet_mode(bool quiet) noexcept;

    /**
     * @brief 核心日志方法
     * @param level 日志级别
     * @param module 模块名称
     * @param file 源文件名
     * @param line 行号
     * @param message 日志消息
     */
    void log(LogLevel level, const char* module, const char* file,
             int line, const std::string& message);

    /**
     * @brief 获取当前日志级别
     * @return 当前日志级别
     */
    LogLevel level() const noexcept { return current_level_; }

    /**
     * @brief 记录异常（便捷方法，用于catch块）
     * @param e TRException异常对象
     *
     * @note 此方法可在catch块中手动调用，用于记录已捕获的异常
     *       terminate handler会自动处理未捕获的异常，无需手动调用
     *
     * @deprecated V3.7.0后，异常不再自动记录Logger。此方法仅用于特殊场景。
     */
    void log_exception(const TRException& e);

private:
    Logger();
    ~Logger();

    // 禁止拷贝和赋值
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief 获取当前时间戳字符串
     * @return 格式化的时间戳（YYYY-MM-DD HH:MM:SS.mmm）
     */
    std::string get_timestamp() const;

    /**
     * @brief 将日志级别转换为字符串
     * @param level 日志级别
     * @return 级别字符串
     */
    const char* level_to_string(LogLevel level) const noexcept;

    LogLevel current_level_;      // 当前日志级别
    std::ofstream file_stream_;    // 文件输出流
    bool quiet_mode_;              // 静默模式标志
    mutable std::mutex mutex_;     // 线程安全锁
};

} // namespace tr

// ============================================================================
// 编译期开关（核心设计）
// ============================================================================

#ifndef TR_LOG_LEVEL
    #ifdef NDEBUG
        #define TR_LOG_LEVEL 2  // Release: 只保留WARN/ERROR
    #else
        #define TR_LOG_LEVEL 0  // Debug: 全开
    #endif
#endif

// ============================================================================
// 零开销日志宏
// ============================================================================

namespace tr {
namespace detail {

/**
 * @class LogStream
 * @brief 延迟求值辅助类（RAII）
 * @details 在析构时自动将缓冲区内容输出到Logger，支持流式操作
 */
class LogStream {
public:
    LogStream(LogLevel level, const char* module, const char* file, int line)
        : level_(level), module_(module), file_(file), line_(line) {}

    ~LogStream() {
        tr::Logger::instance().log(level_, module_, file_, line_, stream_.str());
    }

    /**
     * @brief 流式输出操作符
     * @tparam T 输出类型
     * @param value 输出值
     * @return LogStream引用，支持链式调用
     */
    template<typename T>
    LogStream& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

    /**
     * @brief 支持std::endl等操纵符
     * @param manip 操纵符
     * @return LogStream引用
     */
    LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        stream_ << manip;
        return *this;
    }

private:
    LogLevel level_;
    const char* module_;
    const char* file_;
    int line_;
    std::ostringstream stream_;
};

/**
 * @struct NullStream
 * @brief 空操作占位符
 * @details 用于编译期优化，被过滤的日志不产生任何代码
 */
struct NullStream {
    template<typename T>
    NullStream& operator<<(const T&) { return *this; }

    NullStream& operator<<(std::ostream& (*)(std::ostream&)) { return *this; }
};

} // namespace detail
} // namespace tr

// ============================================================================
// 编译期过滤宏（关键！）
// ============================================================================

#if TR_LOG_LEVEL <= 0
    #define TR_LOG_DEBUG(module) ::tr::detail::LogStream(::tr::LogLevel::DEBUG, module, __FILE__, __LINE__)
#else
    #define TR_LOG_DEBUG(module) if(0) ::tr::detail::NullStream()
#endif

#if TR_LOG_LEVEL <= 1
    #define TR_LOG_INFO(module) ::tr::detail::LogStream(::tr::LogLevel::INFO, module, __FILE__, __LINE__)
#else
    #define TR_LOG_INFO(module) if(0) ::tr::detail::NullStream()
#endif

#if TR_LOG_LEVEL <= 2
    #define TR_LOG_WARN(module) ::tr::detail::LogStream(::tr::LogLevel::WARN, module, __FILE__, __LINE__)
#else
    #define TR_LOG_WARN(module) if(0) ::tr::detail::NullStream()
#endif

// ERROR级别始终保留
#define TR_LOG_ERROR(module) ::tr::detail::LogStream(::tr::LogLevel::ERROR, module, __FILE__, __LINE__)

// ============================================================================
// 便捷宏（自动推断模块）
// ============================================================================

#define LOG_DEBUG TR_LOG_DEBUG("TR")
#define LOG_INFO  TR_LOG_INFO("TR")
#define LOG_WARN  TR_LOG_WARN("TR")
#define LOG_ERROR TR_LOG_ERROR("TR")
