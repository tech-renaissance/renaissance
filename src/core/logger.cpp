/**
 * @file logger.cpp
 * @brief 日志器类实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: core
 */

#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include <iomanip>
#include <thread>
#include <ctime>
#include <cstring>

namespace tr {

Logger& Logger::instance() noexcept {
    static Logger instance;
    return instance;
}

Logger::Logger() = default;

Logger::~Logger() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
}

void Logger::set_level(LogLevel level) noexcept {
    current_level_.store(level, std::memory_order_relaxed);
}

void Logger::set_output_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    if (!filename.empty()) {
        file_stream_.clear();  // 清除之前的错误状态
        file_stream_.open(filename, std::ios::app);
        if (!file_stream_.is_open()) {
            std::cerr << "[Logger] Failed to open log file: " << filename << std::endl;
        }
    }
}

void Logger::set_quiet_mode(bool quiet) noexcept {
    quiet_mode_.store(quiet, std::memory_order_relaxed);
}

void Logger::log(LogLevel level, const char* module, const char* file,
                 int line, const std::string& message) {
    // 运行时过滤（无锁读取）
    if (level < current_level_.load(std::memory_order_relaxed) ||
        quiet_mode_.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 构建日志行
    std::ostringstream oss;
    oss << "[" << get_timestamp() << "] "
        << "[" << level_to_string(level) << "] "
        << "[" << module << "] "
        << message;

    // DEBUG级别显示位置信息
    if (level == LogLevel::DEBUG) {
        // 提取文件名（去除路径）
        const char* filename = strrchr(file, '/');
        if (!filename) filename = strrchr(file, '\\');
        filename = filename ? filename + 1 : file;

        oss << " (" << filename << ":" << line << ")";
    }

    std::string log_line = oss.str();

    // 输出
    if (file_stream_.is_open()) {
        file_stream_ << log_line << std::endl;
    } else {
        if (level >= LogLevel::ERR) {
            std::cerr << log_line << std::endl;
        } else {
            std::cout << log_line << std::endl;
        }
    }
}

std::string Logger::get_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

const char* Logger::level_to_string(LogLevel level) const noexcept {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERR:   return "ERROR";  // 显示时仍为"ERROR"
        default: return "?????";
    }
}

void Logger::log_exception(const TRException& e) {
    // 4.20.1：使用线程安全的 get_contexts() 替代 deprecated 的 contexts()
    auto contexts = e.get_contexts();
    log(LogLevel::ERR, "EXCEPTION",
        contexts.empty() ? "" : contexts[0].file,
        0, e.what());
}

void Logger::init() {
    // 空实现：Logger的构造函数已完成所有初始化工作
}

void Logger::cleanup() {
    // 空实现：Logger的析构函数会处理资源清理
}

} // namespace tr
