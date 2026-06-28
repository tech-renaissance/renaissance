/**
 * @file sample_info.h
 * @brief 样本信息结构体（FULLY模式第二个epoch及以后使用）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include <cstdint>

namespace tr {

// 跨平台对齐宏（64字节对齐，避免false sharing）
#ifdef _WIN32
    #define ALIGN_64 __declspec(align(64))
#else
    #define ALIGN_64 __attribute__((aligned(64)))
#endif

// MSVC: 禁用C4324警告（结构体因对齐而填充），这是预期的性能优化
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4324)
#endif

/**
 * @brief 样本信息结构体（FULLY模式第二个epoch及以后使用）
 * @details 存储(label, data_ptr, data_size)，用于FULLY模式的样本记录和重放
 * @note 64字节对齐，避免false sharing（多个线程访问相邻元素时的缓存行竞争）
 */
struct ALIGN_64 SampleInfo {
    int32_t label;               // 标签
    const uint8_t* data_ptr;     // 数据指针（指向full_arena或buffer）
    size_t data_size;            // 数据大小
};

// 恢复MSVC警告设置
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

// 结构体大小验证（调试用）
static_assert(sizeof(SampleInfo) >= 16, "SampleInfo size must be at least 16 bytes");

} // namespace tr
