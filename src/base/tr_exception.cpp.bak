/**
 * @file tr_exception.cpp
 * @brief 框架异常类实现
 * @details 实现TRException基类的方法，包括自动日志记录功能
 * @version 3.5.5
 * @date 2025-12-24
 * @author 技术觉醒团队
 * @note 依赖项: tr_exception.h, logger.h
 * @note 所属系列: base
 */

#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <cstring>

namespace tr {

TRException::TRException(const std::string& message,
                         const char* file,
                         int line,
                         const char* func)
    : message_(message), file_(file), line_(line), func_(func) {
    auto_log();  // 关键改进：自动记录日志
}

const char* TRException::what() const noexcept {
    if (what_.empty()) {
        build_what();
    }
    return what_.c_str();
}

void TRException::build_what() const {
    std::ostringstream oss;
    oss << "[" << type() << "] " << message_;

    if (file_ && file_[0] != '\0') {
        // 提取文件名（去除路径）
        const char* filename = strrchr(file_, '/');
        if (!filename) filename = strrchr(file_, '\\');
        filename = filename ? filename + 1 : file_;

        oss << " (at " << filename << ":" << line_;
        if (func_ && func_[0] != '\0') {
            oss << " in " << func_ << "()";
        }
        oss << ")";
    }

    what_ = oss.str();
}

void TRException::auto_log() const {
    // 自动记录ERROR级别日志
    Logger::instance().log(LogLevel::ERROR, "EXCEPTION",
                           file_, line_, message_);
}

TRException::~TRException() noexcept = default;

} // namespace tr
