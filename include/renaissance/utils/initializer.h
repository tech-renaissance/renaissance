/**
 * @file initializer.h
 * @brief 全局单例初始化器
 * @details 控制所有单例的构造和析构顺序，避免未定义行为
 * @version 1.0.0
 * @date 2026-02-27
 * @author 技术觉醒团队
 * @note 所属系列: utils
 */

#pragma once

#include <string>
#include <functional>
#include "renaissance/base/rng.h"  // for manual_seed()

namespace tr {

/**
 * @class Initializer
 * @brief 全局单例初始化器
 *
 * 功能：
 * - 控制单例的构造顺序（通过 init() 方法）
 * - 控制单例的析构顺序（通过 ScopeGuard 的 RAII 机制）
 *
 * 构造顺序：
 * 1. Logger（priority 10）
 * 2. GlobalRegistry（priority 20）
 * 3. DeviceManager（priority 30）
 * 4. Preprocessor（priority 40）
 *
 * 析构顺序：
 * - 严格逆序：Preprocessor → DeviceManager → GlobalRegistry → Logger
 *
 * 使用方法：
 * \code
 * int main() {
 *     tr::Initializer::init();  // 构造所有单例
 *     // ... 使用单例
 *     return 0;  // 自动析构所有单例
 * }
 * \endcode
 */
class Initializer {
public:
    /**
     * @brief 初始化所有单例
     * @details 按优先级顺序调用各单例的 init() 方法
     *
     * @param deep_learning_device 深度学习设备类型（"GPU"/"CPU"/"CUDA"/"MUSA"），默认"GPU"
     *                             注意："CUDA"和"MUSA"会自动转换为"GPU"
     *                             在某些场景下（EDGE_ARM/EDGE_RISCV/CPU_CLOUD）会强制使用"CPU"
     *
     * @note 线程安全，多次调用只生效一次
     */
    static void init(const std::string& deep_learning_device = "GPU");

private:
    /**
     * @brief 作用域守卫（RAII）
     * @details 析构时按逆序清理所有单例
     */
    struct ScopeGuard {
        ~ScopeGuard();
    };

    /**
     * @brief 清理所有单例
     * @details 按优先级逆序调用各单例的 cleanup() 方法（如果有）
     */
    static void cleanup_all();

    // 静态成员：程序启动时构造，退出时析构
    inline static ScopeGuard guard_;
    inline static bool initialized_ = false;

    // 禁止实例化
    Initializer() = delete;
    Initializer(const Initializer&) = delete;
    Initializer& operator=(const Initializer&) = delete;
};

} // namespace tr

// ============================================================================
// 全局宏别名（简化用户调用）
// ============================================================================

/**
 * @brief 初始化 renAIssance 框架（宏别名）
 * @details 等同于调用 tr::Initializer::init()
 *
 * 使用方法：
 * \code
 * int main() {
 *     INIT_FRAMEWORK();           // 使用默认设备（GPU）
 *     INIT_FRAMEWORK("GPU");      // 显式指定GPU
 *     INIT_FRAMEWORK("CPU");      // 显式指定CPU
 *     INIT_FRAMEWORK("CUDA");     // CUDA会自动转换为GPU
 *     INIT_FRAMEWORK("MUSA");     // MUSA会自动转换为GPU
 *     // ... 使用框架功能 ...
 *     return 0;
 * }
 * \endcode
 */
#define INIT_FRAMEWORK(...) tr::Initializer::init(__VA_ARGS__)

/**
 * @brief 设置可复现性标志（便捷宏）
 * @param value 是否启用可复现性（true=启用，false=禁用）
 *
 * \par 使用示例：
 * \code
 * int main() {
 *     INIT_FRAMEWORK("GPU");
 *     ENSURE_REPRODUCIBILITY(true);         // 启用可复现性
 *     ENSURE_REPRODUCIBILITY(false);        // 禁用可复现性
 *
 *     // ... 使用框架功能 ...
 *     return 0;
 * }
 * \endcode
 *
 * \note 此宏会调用 GlobalRegistry::set_reproducibility_insurance()
 */
#define ENSURE_REPRODUCIBILITY(value) \
    do { \
        tr::GlobalRegistry::instance().set_reproducibility_insurance(value); \
    } while(0)

/**
 * @brief 设置全局随机数种子（便捷宏）
 * @param seed 种子值（任意无符号64位整数）
 *
 * \par 使用示例：
 * \code
 * int main() {
 *     INIT_FRAMEWORK("GPU");
 *     MANUAL_SEED(42);                    // 设置随机种子
 *     ENSURE_REPRODUCIBILITY(true);       // 启用可复现性
 *
 *     // ... 使用框架功能 ...
 *     return 0;
 * }
 * \endcode
 *
 * \note 此宏会调用 tr::manual_seed()
 * \note 应在 INIT_FRAMEWORK() 之后调用
 * \note 相同的seed会产生完全相同的随机数序列
 */
#define MANUAL_SEED(seed) \
    do { \
        tr::manual_seed(seed); \
    } while(0)
