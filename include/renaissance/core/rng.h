/**
 * @file rng.h
 * @brief 随机数生成器类声明
 * @details 基于Philox4x32-10的Counter-Based RNG
 *          核心特性：多线程可复现、高性能、跨平台
 *          使用Pimpl模式避免在头文件中包含<atomic>
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: 无
 * @note 所属系列: base
 */

#pragma once

#include <cstdint>
#include <memory>  // for std::unique_ptr

namespace tr {

/**
 * @class Generator
 * @brief 伪随机数生成器（基于Philox4x32-10）
 *
 * 核心设计：
 * - 状态仅由 seed 和 offset 决定
 * - 使用原子操作预留偏移量区间，支持无锁并行
 * - 相同(seed, offset)永远产生相同随机数序列
 *
 * 内存布局：使用Pimpl模式，避免在头文件中暴露std::atomic
 * - 原因：MUSA SDK的musa/std/atomic与标准库<atomic>存在命名空间冲突
 * - 解决方案：通过前置声明将std::atomic隐藏在.cpp实现文件中
 * - 效果：MUSA环境下编译renaissance.h时不会包含<atomic>
 *
 * 典型用法：
 * @code
 * // 全局设置（推荐使用 GlobalRegistry 接口）
 * tr::GlobalRegistry::instance().manual_seed(42);
 *
 * // 使用默认生成器
 * tr::cpu_rand_normal_float(ptr, count, 0.0f, 1.0f);
 *
 * // 使用独立生成器（推荐用于多线程数据加载）
 * tr::Generator gen(1234);
 * tr::cpu_rand_normal_float(ptr, count, 0.0f, 1.0f, gen);
 * @endcode
 */
class Generator {
public:
    /**
     * @brief 构造函数
     * @param seed 随机种子（默认0）
     */
    explicit Generator(uint64_t seed = 0) noexcept;

    /**
     * @brief 析构函数
     */
    ~Generator();

    // 禁止拷贝（原子成员不可拷贝）
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // 允许移动
    Generator(Generator&& other) noexcept;
    Generator& operator=(Generator&& other) noexcept;

    // =========================================================================
    // 种子管理
    // =========================================================================

    /**
     * @brief 设置种子（重置offset为0）
     * @param seed 新种子
     * @note 线程安全，但会短暂阻塞其他操作
     */
    void set_seed(uint64_t seed);

    /**
     * @brief 获取当前种子
     * @return 种子值
     */
    uint64_t seed() const noexcept;

    // =========================================================================
    // 状态管理（用于Checkpoint）
    // =========================================================================

    /**
     * @brief 获取完整状态
     * @return {seed, offset}
     */
    std::pair<uint64_t, uint64_t> get_state() const;

    /**
     * @brief 设置完整状态
     * @param seed 种子
     * @param offset 偏移量
     */
    void set_state(uint64_t seed, uint64_t offset);

    // =========================================================================
    // 偏移量管理（核心方法）
    // =========================================================================

    /**
     * @brief 原子预留偏移量区间
     * @param count 需要的随机数个数
     * @return 本次预留的起始offset
     *
     * 这是实现多线程可复现的关键：
     * - 调用者获得 [返回值, 返回值+count) 区间的独占使用权
     * - 即便多线程并发调用，每个调用获得的区间互不重叠
     * - 相同的调用序列产生相同的区间分配
     */
    uint64_t next_offset(uint64_t count);

    /**
     * @brief 获取当前偏移量（不修改状态）
     * @return 当前offset
     */
    uint64_t current_offset() const noexcept;

    // =========================================================================
    // 辅助方法
    // =========================================================================

    /**
     * @brief 生成[low, high]范围的随机整数（包含两端）
     * @param low 最小值（包含）
     * @param high 最大值（包含）
     * @return [low, high]范围内的随机整数
     *
     * @note 可复现：相同的seed和offset序列产生相同的结果
     * @note 用于Fisher-Yates洗牌等需要精确控制的场景
     */
    int random_int(int low, int high);

private:
    class Impl;  // Pimpl: 前向声明实现类
    std::unique_ptr<Impl> impl_;  // 指向实现的智能指针
};

// =============================================================================
// 全局函数
// =============================================================================

/**
 * @brief 获取默认全局生成器
 * @return Generator引用
 * @note 线程安全（Meyers单例）
 */
Generator& get_default_generator();

/**
 * @brief 设置全局随机种子
 * @param seed 种子值
 *
 * 等价于 get_default_generator().set_seed(seed)
 */
void rng_set_seed(uint64_t seed);

// =============================================================================
// CPU随机数生成函数（核心API）
// =============================================================================

/**
 * @brief 生成N个随机uint64整数
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param gen 生成器引用
 *
 * 多线程安全且可复现
 */
void cpu_rand_uint64(uint64_t* ptr, size_t count, Generator& gen);

/**
 * @brief 生成N个伯努利分布的INT8（0或1）
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param prob_one "1"的概率，范围[0, 1]
 * @param gen 生成器引用
 */
void cpu_rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one, Generator& gen);

/**
 * @brief 生成N个均匀分布的INT8
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param low 最小值（包含）
 * @param high 最大值（包含）
 * @param gen 生成器引用
 */
void cpu_rand_uniform_int8(int8_t* ptr, size_t count, int8_t low, int8_t high, Generator& gen);

/**
 * @brief 生成N个伯努利分布的INT32（0或1）
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param prob_one "1"的概率，范围[0, 1]
 * @param gen 生成器引用
 */
void cpu_rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one, Generator& gen);

/**
 * @brief 生成N个均匀分布的INT32
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param low 最小值（包含）
 * @param high 最大值（不包含）
 * @param gen 生成器引用
 * @note 范围为 [low, high)，与Python randint语义一致
 */
void cpu_rand_uniform_int32(int32_t* ptr, size_t count, int32_t low, int32_t high, Generator& gen);

/**
 * @brief 生成N个均匀分布的FP32
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param low 最小值（包含）
 * @param high 最大值（不包含）
 * @param gen 生成器引用
 */
void cpu_rand_uniform_float(float* ptr, size_t count, float low, float high, Generator& gen);

/**
 * @brief 生成N个正态分布的FP32
 * @param ptr 目标内存指针
 * @param count 元素个数
 * @param mean 均值
 * @param std 标准差
 * @param gen 生成器引用
 */
void cpu_rand_normal_float(float* ptr, size_t count, float mean, float std, Generator& gen);

// =============================================================================
// 便捷函数（使用默认生成器）
// =============================================================================

inline void cpu_rand_uint64(uint64_t* ptr, size_t count) {
    cpu_rand_uint64(ptr, count, get_default_generator());
}

inline void cpu_rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one) {
    cpu_rand_bernoulli_int8(ptr, count, prob_one, get_default_generator());
}

inline void cpu_rand_uniform_int8(int8_t* ptr, size_t count, int8_t low, int8_t high) {
    cpu_rand_uniform_int8(ptr, count, low, high, get_default_generator());
}

inline void cpu_rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one) {
    cpu_rand_bernoulli_int32(ptr, count, prob_one, get_default_generator());
}

inline void cpu_rand_uniform_int32(int32_t* ptr, size_t count, int32_t low, int32_t high) {
    cpu_rand_uniform_int32(ptr, count, low, high, get_default_generator());
}

inline void cpu_rand_uniform_float(float* ptr, size_t count, float low = 0.0f, float high = 1.0f) {
    cpu_rand_uniform_float(ptr, count, low, high, get_default_generator());
}

inline void cpu_rand_normal_float(float* ptr, size_t count, float mean = 0.0f, float std = 1.0f) {
    cpu_rand_normal_float(ptr, count, mean, std, get_default_generator());
}

} // namespace tr
