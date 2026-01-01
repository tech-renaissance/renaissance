/**
 * @file tr_exception.h
 * @brief 框架异常类声明（含子类）
 * @details 技术觉醒框架统一异常体系，支持错误类型分类，自动记录日志，完全向后兼容
 * @version 3.5.5
 * @date 2025-12-24
 * @author 技术觉醒团队
 * @note 依赖项:
 * @note 所属系列: base
 */

#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace tr {

/**
 * @class TRException
 * @brief 框架基础异常类
 * @details 继承自std::exception，提供统一的异常接口，支持自动日志记录
 */
class TRException : public std::exception {
public:
    /**
     * @brief 构造异常对象
     * @param message 异常消息
     * @param file 源文件名（自动捕获）
     * @param line 行号（自动捕获）
     * @param func 函数名（自动捕获）
     */
    TRException(const std::string& message,
                const char* file = "",
                int line = 0,
                const char* func = "");

    virtual ~TRException() noexcept;

    /**
     * @brief 获取异常描述
     * @return 完整的异常描述字符串
     */
    const char* what() const noexcept override;

    /**
     * @brief 获取异常类型名称
     * @return 类型名称字符串
     */
    virtual const char* type() const noexcept { return "TRException"; }

    /**
     * @brief 获取异常消息
     * @return 异常消息引用
     */
    const std::string& message() const noexcept { return message_; }

    /**
     * @brief 获取源文件名
     * @return 文件名字符串
     */
    const char* file() const noexcept { return file_; }

    /**
     * @brief 获取行号
     * @return 行号
     */
    int line() const noexcept { return line_; }

protected:
    std::string message_;    // 异常消息
    const char* file_;       // 源文件名
    int line_;               // 行号
    const char* func_;       // 函数名
    mutable std::string what_; // 缓存的what()结果

    /**
     * @brief 构建完整的异常描述
     */
    void build_what() const;

    /**
     * @brief 自动记录到Logger
     */
    void auto_log() const;
};

// ============================================================================
// 异常类型定义（使用宏简化）
// ============================================================================

/**
 * @brief 定义异常类型的宏
 * @param ClassName 异常类名
 */
#define TR_DEFINE_EXCEPTION(ClassName) \
    class ClassName : public TRException { \
    public: \
        using TRException::TRException; \
        const char* type() const noexcept override { return #ClassName; } \
    }

// 定义9种异常类型
TR_DEFINE_EXCEPTION(NotImplementedError);
TR_DEFINE_EXCEPTION(FileNotFoundError);
TR_DEFINE_EXCEPTION(ValueError);
TR_DEFINE_EXCEPTION(IndexError);
TR_DEFINE_EXCEPTION(TypeError);
TR_DEFINE_EXCEPTION(ZeroDivisionError);
TR_DEFINE_EXCEPTION(ShapeError);
TR_DEFINE_EXCEPTION(DeviceError);
TR_DEFINE_EXCEPTION(MemoryError);

#undef TR_DEFINE_EXCEPTION

} // namespace tr

// ============================================================================
// 便捷宏（关键改进！）
// ============================================================================

namespace tr {
namespace detail {

/**
 * @brief 简单的变参拼接器
 * @tparam Args 参数类型包
 * @param oss 输出字符串流
 * @param args 可变参数
 */
inline void append_to_stream(std::ostringstream&) {}

template<typename T, typename... Args>
void append_to_stream(std::ostringstream& oss, const T& value, const Args&... args) {
    oss << value;
    append_to_stream(oss, args...);
}

/**
 * @brief 格式化异常消息
 * @tparam Args 可变参数类型
 * @param args 格式化参数
 * @return 格式化后的字符串
 */
template<typename... Args>
std::string format_exception_message(const Args&... args) {
    std::ostringstream oss;
    append_to_stream(oss, args...);
    return oss.str();
}

} // namespace detail
} // namespace tr

// 基础抛出宏
#define TR_THROW(ExceptionType, ...) \
    throw tr::ExceptionType( \
        ::tr::detail::format_exception_message(__VA_ARGS__), \
        __FILE__, __LINE__, __func__)

// 条件抛出宏
#define TR_CHECK(condition, ExceptionType, ...) \
    do { \
        if (!(condition)) { \
            TR_THROW(ExceptionType, __VA_ARGS__); \
        } \
    } while(0)

// 快捷宏（最常用场景）
#define TR_NOT_IMPLEMENTED(...) TR_THROW(NotImplementedError, __VA_ARGS__)
#define TR_VALUE_ERROR(...) TR_THROW(ValueError, __VA_ARGS__)
#define TR_SHAPE_ERROR(...) TR_THROW(ShapeError, __VA_ARGS__)
#define TR_TYPE_ERROR(...) TR_THROW(TypeError, __VA_ARGS__)
#define TR_INDEX_ERROR(...) TR_THROW(IndexError, __VA_ARGS__)
#define TR_DEVICE_ERROR(...) TR_THROW(DeviceError, __VA_ARGS__)
#define TR_FILE_NOT_FOUND_ERROR(...) TR_THROW(FileNotFoundError, __VA_ARGS__)
#define TR_ZERO_DIVISION_ERROR(...) TR_THROW(ZeroDivisionError, __VA_ARGS__)
#define TR_MEMORY_ERROR(...) TR_THROW(MemoryError, __VA_ARGS__)
