/**
 * @file decode_strategy.h
 * @brief JPEG解码策略封装
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include <cstdint>

namespace tr {

/**
 * @struct DecodeStrategy
 * @brief JPEG解码策略（仅包含R2解码区域信息）
 *
 * 说明：
 * - need_decode=false: 非ImageNet，直接使用输入（已解码）
 * - use_partial=true: 局部解码R2区域（MCU对齐），PO从R2中提取R1
 * - use_partial=false: 完整解码整张图，PO从整图中提取R1
 *
 * 坐标系统：
 * - R2 (MCU对齐解码区域): decode_x, decode_y, decode_w, decode_h (相对于完整图像的绝对坐标)
 * - R1 (实际裁剪区域): 由PO内部管理，不包含在此结构体中
 *
 * PW职责：
 * - 根据use_partial选择完整解码或局部解码R2到D区
 * - 调用execute时传递input_ptr=D区, input_w=decode_w, input_h=decode_h
 *
 * PO职责：
 * - get_decode_strategy()时计算R1和R2，仅返回R2信息
 * - execute()时根据execute_from_full标志，自己计算从input中提取R1的偏移
 */
struct DecodeStrategy {
    bool need_decode = false;       ///< 是否需要解码（非ImageNet为false）
    bool use_partial = false;       ///< 局部解码vs完整解码

    // MCU对齐的解码窗口R2（8的倍数，相对于完整图像的绝对坐标）
    int32_t decode_x = 0;           ///< 解码起始X（MCU对齐，向下取整）
    int32_t decode_y = 0;           ///< 解码起始Y（MCU对齐，向下取整）
    int32_t decode_w = 0;           ///< 解码宽度（MCU对齐，向上取整）
    int32_t decode_h = 0;           ///< 解码高度（MCU对齐，向上取整）
};

} // namespace tr
