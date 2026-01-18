/**
 * @file sample_window.h
 * @brief Preprocessor样本窗口结构
 * @version 3.8.0
 * @date 2026-01-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace tr {
namespace data {

/**
 * @brief 样本窗口（Preprocessor读取接口）
 * @details 零拷贝设计，直接指向内部缓冲区
 */
struct SampleWindow {
    const uint8_t* data;    ///< 图片数据指针（JPEG字节流）
    size_t size;            ///< 字节数
    int32_t label;          ///< 标签（ImageNet: 0~999）
    bool ready;             ///< 是否就绪

    /**
     * @brief 默认构造函数
     */
    SampleWindow() : data(nullptr), size(0), label(-1), ready(false) {}

    /**
     * @brief 构造函数
     */
    SampleWindow(const uint8_t* ptr, size_t sz, int32_t lbl, bool r)
        : data(ptr), size(sz), label(lbl), ready(r) {}
};

} // namespace data
} // namespace tr
