/**
 * @file tr_exception.cpp
 * @brief 异常类实现 + 全局terminate handler
 * @details 实现TRException的方法和全局terminate handler自动安装
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: tr_exception.h
 * @note 所属系列: core
 */

#include "renaissance/core/tr_exception.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <crtdbg.h>
#endif

namespace tr {

// ============================================================================
// 全局terminate handler（静态注册，自动生效）
// ============================================================================

namespace {

void framework_terminate_handler() noexcept {
    std::cerr << "\n";
    std::cerr << "===============================================================================\n";
    std::cerr << "            RENAISSANCE FRAMEWORK - FATAL ERROR\n";
    std::cerr << "===============================================================================\n";
    std::cerr << "\n";

    std::exception_ptr eptr = std::current_exception();
    if (eptr) {
        try {
            std::rethrow_exception(eptr);
        } catch (const TRException& e) {
            std::cerr << "Exception Type : " << e.type() << "\n";
            std::cerr << "Root Message   : " << e.message() << "\n";
            std::cerr << "\n";

            // 4.20.1：使用线程安全的get_contexts()，并添加OOM保护
            try {
                auto contexts = e.get_contexts();
                if (!contexts.empty()) {
                    std::cerr << "Call Stack (bottom to top):\n";
                    for (const auto& ctx : contexts) {
                        std::cerr << "  -> " << ctx.to_string() << "\n";
                    }
                    std::cerr << "\n";
                }
            } catch (...) {
                std::cerr << "Call Stack: (unavailable - likely out of memory)\n\n";
            }

            // 4.20.1：对what()加保护（可能因OOM失败）
            try {
                std::cerr << "Full Description:\n  " << e.what() << "\n";
            } catch (...) {
                std::cerr << "Full Description: (unavailable - likely out of memory)\n";
            }

        } catch (const std::exception& e) {
            // 4.20.1：对标准异常的what()也加保护
            try {
                std::cerr << "Standard Exception: " << e.what() << "\n";
            } catch (...) {
                std::cerr << "Standard Exception: (unable to display)\n";
            }
        } catch (...) {
            std::cerr << "Unknown Exception\n";
        }
    } else {
        std::cerr << "Terminate called without active exception\n";
    }

    std::cerr << "\n";
    std::cerr << "===============================================================================\n";
    std::cerr << "Program will now abort.\n";
    std::cerr << "===============================================================================\n\n";
    std::cerr << std::flush;

    // 禁用Windows错误对话框（只对当前进程有效）
    #ifdef _WIN32
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    #endif

    std::abort();
}

// 使用静态初始化自动注册（C++11保证线程安全）
struct TerminateHandlerRegistrar {
    TerminateHandlerRegistrar() {
        std::set_terminate(framework_terminate_handler);
    }
};

static TerminateHandlerRegistrar g_registrar;

} // anonymous namespace

// ============================================================================
// TRException实现
// ============================================================================

TRException::TRException(const char* type_name,
                         const std::string& message,
                         const char* file,
                         const char* func)
    : type_name_(type_name)
    , root_message_(message)
    , what_cached_(false)  // 4.20.1：显式初始化（C++11兼容）
{
    // 添加根上下文
    ExceptionContext root_ctx;
    root_ctx.file = basename(file);
    root_ctx.func = func;
    root_ctx.message = message;
    contexts_.push_back(root_ctx);
}

// 4.20.1：拷贝构造函数实现（MSVC C5272 警告修复）
TRException::TRException(const TRException& other) noexcept
    : std::exception(other),
      type_name_(other.type_name_),
      root_message_(other.root_message_),
      contexts_(other.contexts_),
      what_cached_(other.what_cached_.load(std::memory_order_acquire)),
      what_cache_(other.what_cache_) {
    // mutex_ 不能复制，使用默认构造的 mutex（每个对象独立）
    // 这是 noexcept 函数，不抛出异常
}

// 4.20.1：拷贝赋值运算符实现（MSVC C5272 警告修复）
TRException& TRException::operator=(const TRException& other) noexcept {
    if (this != &other) {
        // 复制所有可复制成员
        type_name_ = other.type_name_;
        root_message_ = other.root_message_;
        contexts_ = other.contexts_;
        what_cached_.store(other.what_cached_.load(std::memory_order_acquire),
                          std::memory_order_release);
        what_cache_ = other.what_cache_;
        // mutex_ 不能复制，保持当前对象的 mutex 独立性
    }
    return *this;
}

void TRException::add_context(const std::string& ctx_message,
                               const char* file,
                               const char* func) {
    std::lock_guard<std::mutex> lock(mutex_);  // 4.20.1：线程安全

    ExceptionContext ctx;
    ctx.file = basename(file);
    ctx.func = func;
    ctx.message = ctx_message;
    contexts_.push_back(ctx);

    // 清空缓存，强制重建
    what_cache_.clear();
    what_cached_.store(false, std::memory_order_release);  // 4.20.1：重置缓存标志
}

// 4.20.1：线程安全的get_contexts()实现
std::vector<ExceptionContext> TRException::get_contexts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return contexts_;  // 返回拷贝，安全
}

const char* TRException::what() const noexcept {
    // 4.20.1：快速路径（无锁检查，C++11兼容）
    if (what_cached_.load(std::memory_order_acquire)) {
        return what_cache_.c_str();
    }

    // 4.20.1：慢速路径（加锁构建）
    std::lock_guard<std::mutex> lock(mutex_);

    // 4.20.1：双检锁（其他线程可能已构建）
    if (what_cached_.load(std::memory_order_acquire)) {
        return what_cache_.c_str();
    }

    // 4.20.1：OOM安全保护
    try {
        std::ostringstream oss;
        oss << "[" << type_name_ << "] " << root_message_;

        // 构建Context Chain
        if (contexts_.size() > 1) {
            oss << "\n  Context Chain:";
            for (const auto& ctx : contexts_) {
                oss << "\n    -> " << ctx.message
                    << " (at " << ctx.file << " :: " << ctx.func << "())";
            }
        } else if (!contexts_.empty()) {
            // 单层上下文，简化输出
            const auto& ctx = contexts_[0];
            oss << " (at " << ctx.file << " :: " << ctx.func << "())";
        }

        what_cache_ = oss.str();
        what_cached_.store(true, std::memory_order_release);

    } catch (...) {
        // 4.20.1：OOM兜底 - 直接返回静态字符串，不依赖堆分配
        // 不写入 what_cache_，避免二次 bad_alloc
        static const char fallback[] =
            "[FATAL] Unable to format exception message (out of memory)";
        return fallback;  // 零分配，绝不抛异常
    }

    return what_cache_.c_str();
}

const char* TRException::basename(const char* path) {
    if (!path) return "";

    const char* name = path;
    for (const char* p = path; *p; ++p) {
        if (*p == '/' || *p == '\\') {
            name = p + 1;
        }
    }
    return name;
}

} // namespace tr
