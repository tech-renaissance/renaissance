/**
 * @file logger.cpp
 * @brief 日志器类实现
 * @details 实现线程安全的日志输出功能，支持控制台、文件输出和运行时级别控制
 * @version 3.5.5
 * @date 2025-12-24
 * @author 技术觉醒团队
 * @note 依赖项: logger.h
 * @note 所属系列: base
 */

#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include <iomanip>
#include <thread>
#include <ctime>
#include <cstring>

namespace tr {

Logger& Logger::instance() noexcept {
    static Logger instance;
    return instance;
}

Logger::Logger()
    : current_level_(LogLevel::INFO)
    , quiet_mode_(false) {
}

Logger::~Logger() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
}

void Logger::set_level(LogLevel level) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    current_level_ = level;
}

void Logger::set_output_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    if (!filename.empty()) {
        file_stream_.open(filename, std::ios::app);
        if (!file_stream_.is_open()) {
            std::cerr << "[Logger] Failed to open log file: " << filename << std::endl;
        }
    }
}

void Logger::set_quiet_mode(bool quiet) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    quiet_mode_ = quiet;
}

void Logger::log(LogLevel level, const char* module, const char* file,
                 int line, const std::string& message) {
    // 运行时过滤
    if (level < current_level_ || quiet_mode_) return;

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
        if (level >= LogLevel::ERROR) {
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
        case LogLevel::ERROR: return "ERROR";
        default: return "?????";
    }
}

void Logger::log_exception(const TRException& e) {
    log(LogLevel::ERROR, "EXCEPTION",
        e.contexts().empty() ? "" : e.contexts()[0].file,
        0, e.what());
}

} // namespace tr
