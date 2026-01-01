/**
 * @file tr_exception.cpp
 * @brief 异常类实现 + 全局terminate handler
 * @details 实现TRException的方法和全局terminate handler自动安装
 * @version 3.7.0
 * @date 2026-01-01
 * @author 技术觉醒团队
 * @note 依赖项: tr_exception.h
 * @note 所属系列: base
 */

#include "renaissance/base/tr_exception.h"
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

            // 显示完整的Context Chain
            const auto& contexts = e.contexts();
            if (!contexts.empty()) {
                std::cerr << "Call Stack (bottom to top):\n";
                for (const auto& ctx : contexts) {
                    std::cerr << "  -> " << ctx.to_string() << "\n";
                }
                std::cerr << "\n";
            }

            std::cerr << "Full Description:\n  " << e.what() << "\n";

        } catch (const std::exception& e) {
            std::cerr << "Standard Exception: " << e.what() << "\n";
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
{
    // 添加根上下文
    ExceptionContext root_ctx;
    root_ctx.file = basename(file);
    root_ctx.func = func;
    root_ctx.message = message;
    contexts_.push_back(root_ctx);
}

void TRException::add_context(const std::string& ctx_message,
                               const char* file,
                               const char* func) {
    ExceptionContext ctx;
    ctx.file = basename(file);
    ctx.func = func;
    ctx.message = ctx_message;
    contexts_.push_back(ctx);

    // 清空缓存，强制重建
    what_cache_.clear();
}

const char* TRException::what() const noexcept {
    if (what_cache_.empty()) {
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
