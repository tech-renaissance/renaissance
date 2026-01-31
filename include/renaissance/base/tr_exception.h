/**
 * @file tr_exception.h
 * @brief 框架异常系统（渐进式增强架构）
 * @details 技术觉醒框架统一异常体系，支持Context Chain多层上下文，自动记录日志
 * @version 3.7.0
 * @date 2026-01-01
 * @author 技术觉醒团队
 * @note 依赖项:
 * @note 所属系列: base
 */

#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <vector>

namespace tr {

// ============================================================================
// 异常上下文（Context Chain支持）
// ============================================================================

/**
 * @struct ExceptionContext
 * @brief 异常上下文信息
 */
struct ExceptionContext {
    const char* file;      ///< 文件名（basename）
    const char* func;      ///< 函数名
    std::string message;   ///< 消息

    std::string to_string() const {
        std::ostringstream oss;
        oss << message << " (at " << file << " :: " << func << "())";
        return oss.str();
    }
};

// ============================================================================
// 核心异常类
// ============================================================================

/**
 * @class TRException
 * @brief 框架统一异常基类
 *
 * 核心特性：
 * - 支持Context Chain（多层上下文叠加）
 * - 延迟构建what()（性能优化）
 * - 不依赖Logger（解耦设计）
 */
class TRException : public std::exception {
public:
    /**
     * @brief 构造异常
     * @param type_name 异常类型名
     * @param message 错误消息
     * @param file 源文件名（__FILE__）
     * @param func 函数名（__func__）
     */
    TRException(const char* type_name,
                const std::string& message,
                const char* file,
                const char* func);

    virtual ~TRException() noexcept = default;

    /**
     * @brief 获取完整错误描述
     * @return 包含Context Chain的完整消息
     */
    const char* what() const noexcept override;

    /**
     * @brief 添加上下文信息（支持重新抛出）
     * @param ctx_message 上下文消息
     * @param file 文件名
     * @param func 函数名
     */
    void add_context(const std::string& ctx_message,
                     const char* file,
                     const char* func);

    // 访问器
    const char* type() const noexcept { return type_name_; }
    const std::string& message() const noexcept { return root_message_; }
    const std::vector<ExceptionContext>& contexts() const noexcept {
        return contexts_;
    }

protected:
    const char* type_name_;           ///< 异常类型（静态字符串）
    std::string root_message_;        ///< 根消息
    std::vector<ExceptionContext> contexts_; ///< 上下文链（从底到顶）
    mutable std::string what_cache_;  ///< what()缓存

    static const char* basename(const char* path);
};

// ============================================================================
// 具体异常类型
// ============================================================================

#define TR_DEFINE_EXCEPTION(ClassName) \
    class ClassName : public TRException { \
    public: \
        ClassName(const std::string& msg, const char* file, const char* func) \
            : TRException(#ClassName, msg, file, func) {} \
    }

TR_DEFINE_EXCEPTION(NotImplementedError);
TR_DEFINE_EXCEPTION(FileNotFoundError);
TR_DEFINE_EXCEPTION(ValueError);
TR_DEFINE_EXCEPTION(IndexError);
TR_DEFINE_EXCEPTION(TypeError);
TR_DEFINE_EXCEPTION(ZeroDivisionError);
TR_DEFINE_EXCEPTION(ShapeError);
TR_DEFINE_EXCEPTION(DeviceError);
TR_DEFINE_EXCEPTION(MemoryError);
TR_DEFINE_EXCEPTION(TimeoutError);  // P0-2修复：添加超时异常

#undef TR_DEFINE_EXCEPTION

// ============================================================================
// 辅助工具
// ============================================================================

namespace detail {

/**
 * @class MessageBuilder
 * @brief 支持流式语法的消息构建器
 */
class MessageBuilder {
public:
    template<typename T>
    MessageBuilder& operator<<(const T& value) {
        ss_ << value;
        return *this;
    }

    std::string str() const { return ss_.str(); }

private:
    std::ostringstream ss_;
};

} // namespace detail

} // namespace tr

// ============================================================================
// 宏定义（核心API）
// ============================================================================

/**
 * @brief 构建消息（内部辅助宏）
 */
#define TR_MSG(stream) \
    (::tr::detail::MessageBuilder() << stream).str()

/**
 * @brief 抛出异常（核心宏）
 *
 * 用法：TR_THROW(ValueError, "Invalid size: " << size)
 */
#define TR_THROW(ExceptionType, msg_stream) \
    throw ::tr::ExceptionType(TR_MSG(msg_stream), __FILE__, __func__)

/**
 * @brief 条件检查（最常用）
 *
 * 用法：TR_CHECK(x > 0, ValueError, "x must be positive, got " << x)
 */
#define TR_CHECK(condition, ExceptionType, msg_stream) \
    do { \
        if (!(condition)) { \
            TR_THROW(ExceptionType, msg_stream); \
        } \
    } while(0)

/**
 * @brief 重新抛出并添加上下文
 *
 * 用法：
 *   try { ... }
 *   catch (TRException& e) {
 *       TR_RETHROW(e, "While loading model from " << path);
 *   }
 */
#define TR_RETHROW(exception, ctx_stream) \
    do { \
        (exception).add_context(TR_MSG(ctx_stream), __FILE__, __func__); \
        throw; \
    } while(0)

// 便捷宏
#define TR_NOT_IMPLEMENTED(...) TR_THROW(NotImplementedError, TR_MSG(__VA_ARGS__))
#define TR_VALUE_ERROR(...) TR_THROW(ValueError, TR_MSG(__VA_ARGS__))
#define TR_SHAPE_ERROR(...) TR_THROW(ShapeError, TR_MSG(__VA_ARGS__))
#define TR_TYPE_ERROR(...) TR_THROW(TypeError, TR_MSG(__VA_ARGS__))
#define TR_INDEX_ERROR(...) TR_THROW(IndexError, TR_MSG(__VA_ARGS__))
#define TR_DEVICE_ERROR(...) TR_THROW(DeviceError, TR_MSG(__VA_ARGS__))
#define TR_FILE_NOT_FOUND(...) TR_THROW(FileNotFoundError, TR_MSG(__VA_ARGS__))
#define TR_ZERO_DIVISION(...) TR_THROW(ZeroDivisionError, TR_MSG(__VA_ARGS__))
#define TR_MEMORY_ERROR(...) TR_THROW(MemoryError, TR_MSG(__VA_ARGS__))
