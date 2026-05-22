/**
 * @file rng.cpp
 * @brief 随机数生成器实现
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: rng.h, philox.h
 * @note 所属系列: base
 */

#include "renaissance/core/rng.h"
#include "renaissance/core/philox.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"

#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <mutex>
#include <iostream>
#include <iomanip>

// OpenMP支持检测
#if defined(_OPENMP)
    #include <omp.h>
    #define TR_USE_OPENMP 1
#else
    #define TR_USE_OPENMP 0
#endif

namespace tr {

// =============================================================================
// Generator::Impl 定义（Pimpl实现类）
// =============================================================================
//
// 重要说明：为什么使用Pimpl模式？
//
// 问题描述：
// - MUSA SDK的<atomic>实现(musa/std/atomic)与C++标准库的<atomic>存在命名空间冲突
// - 如果在头文件中包含<atomic>，MUSA环境编译时会触发大量编译错误
// - 即使通过条件编译(#ifdef TR_USE_MUSA)也无法完全解决，因为包含顺序影响
//
// 解决方案：
// - 在头文件(rng.h)中只前置声明"class Impl"，使用std::unique_ptr<Impl>
// - 在实现文件(rng.cpp)中定义Impl类，此时包含<atomic>是安全的
// - MUSA环境编译renaissance.h时看不到<atomic>，避免冲突
// - CPU/CUDA环境正常编译，不受影响
//
// 性能影响：
// - 增加一次指针解引用(indirection)，但现代CPU的分支预测可缓解
// - unique_ptr开销极小(8字节指针)，相比原来直接成员增加可忽略
// - 关键路径(next_offset)仍使用原子操作，性能不受影响

class Generator::Impl {
public:
    uint64_t seed_;
    std::atomic<uint64_t> offset_;
    std::mutex mutex_;

    explicit Impl(uint64_t seed) noexcept
        : seed_(seed)
        , offset_(0)
    {
    }
};

// =============================================================================
// Generator 实现
// =============================================================================

Generator::Generator(uint64_t seed) noexcept
    : impl_(new Impl(seed))
{
}

Generator::~Generator() = default;

Generator::Generator(Generator&& other) noexcept
    : impl_(std::move(other.impl_))
{
    // 移动后源对象的impl_自动置为nullptr
}

Generator& Generator::operator=(Generator&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        // 源对象的impl_自动置为nullptr
    }
    return *this;
}

void Generator::set_seed(uint64_t seed) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->seed_ = seed;
    impl_->offset_.store(0, std::memory_order_release);
    LOG_DEBUG << "Generator seed set to " << seed;
}

uint64_t Generator::seed() const noexcept {
    return impl_->seed_;
}

std::pair<uint64_t, uint64_t> Generator::get_state() const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    return {impl_->seed_, impl_->offset_.load(std::memory_order_acquire)};
}

void Generator::set_state(uint64_t seed, uint64_t offset) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->seed_ = seed;
    impl_->offset_.store(offset, std::memory_order_release);
}

uint64_t Generator::next_offset(uint64_t count) {
    // 原子加法，返回旧值
    // 这是实现并发安全的关键：每次调用获得独占的区间
    return impl_->offset_.fetch_add(count, std::memory_order_relaxed);
}

uint64_t Generator::current_offset() const noexcept {
    return impl_->offset_.load(std::memory_order_relaxed);
}

int Generator::random_int(int low, int high) {
    // 预留一个offset用于生成随机数
    uint64_t offset = next_offset(1);
    uint64_t seed_val = seed();

    // 生成一个64位随机数
    uint64_t random_val = detail::philox_uint64(seed_val, offset);

    // 映射到[low, high]范围
    // 注意：范围包含两端，所以range = high - low + 1
    uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
    uint64_t val = random_val % range;

    return static_cast<int>(low + static_cast<int64_t>(val));
}

// =============================================================================
// 全局函数实现
// ==============================================================================

Generator& get_default_generator() {
    // Meyers单例，线程安全
    static Generator instance(0);
    return instance;
}

void rng_set_seed(uint64_t seed) {
    get_default_generator().set_seed(seed);
    std::cout << "Random Seed: 0x" << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << seed << std::dec << std::endl;
}

// =============================================================================
// CPU随机数生成函数实现
// =============================================================================

namespace {

/**
 * @brief 计算并行任务的线程数
 */
inline int get_num_threads(size_t count) {
#if TR_USE_OPENMP
    // 小任务不值得并行
    if (count < 4096) return 1;

    int max_threads = omp_get_max_threads();
    // 每个线程至少处理1024个元素
    int ideal_threads = static_cast<int>((count + 1023) / 1024);
    return std::min(max_threads, ideal_threads);
#else
    (void)count;
    return 1;
#endif
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// uint64 生成
// -----------------------------------------------------------------------------

void cpu_rand_uint64(uint64_t* ptr, size_t count, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_uint64");
    }

    // 预留偏移量区间
    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        ptr[i] = detail::philox_uint64(seed, base_offset + i);
    }
}

// -----------------------------------------------------------------------------
// INT8 伯努利分布
// -----------------------------------------------------------------------------

void cpu_rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_bernoulli_int8");
    }
    if (prob_one < 0.0f || prob_one > 1.0f) {
        TR_VALUE_ERROR("prob_one must be in [0, 1], got " << prob_one);
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    // 将概率转换为阈值（uint32范围）
    uint32_t threshold = static_cast<uint32_t>(prob_one * 4294967296.0f);

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        uint32_t r[4];
        detail::philox_generate_4x32(seed, base_offset + i, r);
        ptr[i] = (r[0] < threshold) ? 1 : 0;
    }
}

// -----------------------------------------------------------------------------
// INT8 均匀分布
// -----------------------------------------------------------------------------

void cpu_rand_uniform_int8(int8_t* ptr, size_t count, int8_t low, int8_t high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_uniform_int8");
    }
    if (low > high) {
        TR_VALUE_ERROR("low (" << static_cast<int>(low)
                 << ") must be <= high (" << static_cast<int>(high) << ")");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    // 计算范围
    uint32_t range = static_cast<uint32_t>(high - low) + 1;

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        uint32_t r[4];
        detail::philox_generate_4x32(seed, base_offset + i, r);
        // 使用模运算映射到[0, range)，然后加low
        uint32_t val = r[0] % range;
        ptr[i] = static_cast<int8_t>(low + static_cast<int8_t>(val));
    }
}

// -----------------------------------------------------------------------------
// INT32 伯努利分布
// -----------------------------------------------------------------------------

void cpu_rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_bernoulli_int32");
    }
    if (prob_one < 0.0f || prob_one > 1.0f) {
        TR_VALUE_ERROR("prob_one must be in [0, 1], got " << prob_one);
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    uint32_t threshold = static_cast<uint32_t>(prob_one * 4294967296.0f);

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        uint32_t r[4];
        detail::philox_generate_4x32(seed, base_offset + i, r);
        ptr[i] = (r[0] < threshold) ? 1 : 0;
    }
}

// -----------------------------------------------------------------------------
// INT32 均匀分布
// -----------------------------------------------------------------------------

void cpu_rand_uniform_int32(int32_t* ptr, size_t count, int32_t low, int32_t high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_uniform_int32");
    }
    if (low > high) {
        TR_VALUE_ERROR("low (" << low << ") must be <= high (" << high << ")");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    // 使用64位计算避免溢出
    // 注意：范围是 [low, high)（左闭右开），与Python randint语义一致
    uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low);

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        uint32_t r[4];
        detail::philox_generate_4x32(seed, base_offset + i, r);
        // 组合两个32位数获得更大的范围
        uint64_t combined = (static_cast<uint64_t>(r[0]) << 32) | r[1];
        uint64_t val = combined % range;
        ptr[i] = static_cast<int32_t>(low + static_cast<int64_t>(val));
    }
}

// -----------------------------------------------------------------------------
// FP32 均匀分布
// -----------------------------------------------------------------------------

void cpu_rand_uniform_float(float* ptr, size_t count, float low, float high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_uniform_float");
    }
    if (low > high) {
        TR_VALUE_ERROR("low (" << low << ") must be <= high (" << high << ")");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    float scale = high - low;

    int num_threads = get_num_threads(count);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        float u = detail::philox_uniform_float(seed, base_offset + i);
        ptr[i] = low + u * scale;
    }
}

// -----------------------------------------------------------------------------
// FP32 正态分布
// -----------------------------------------------------------------------------

void cpu_rand_normal_float(float* ptr, size_t count, float mean, float std, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in cpu_rand_normal_float");
    }
    if (std < 0.0f) {
        TR_VALUE_ERROR("std must be >= 0, got " << std);
    }

    // Box-Muller每次生成2个数，所以消耗的offset是(count + 1) / 2
    uint64_t pairs_needed = (count + 1) / 2;
    uint64_t base_offset = gen.next_offset(pairs_needed);
    uint64_t seed = gen.seed();

    int num_threads = get_num_threads(count);

    // 处理成对的元素
    int64_t pair_count = static_cast<int64_t>(count / 2);

#if TR_USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < pair_count; ++i) {
        float n0, n1;
        detail::philox_normal_pair(seed, base_offset + i, &n0, &n1);
        ptr[i * 2] = mean + std * n0;
        ptr[i * 2 + 1] = mean + std * n1;
    }

    // 处理最后一个奇数元素（如果有）
    if (count % 2 == 1) {
        float n0, n1;
        detail::philox_normal_pair(seed, base_offset + pair_count, &n0, &n1);
        ptr[count - 1] = mean + std * n0;
        // n1被丢弃，但这是确定性的
    }
}

} // namespace tr
