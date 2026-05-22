/**
 * @file shape_id.h
 * @brief 形状去重键 —— 用于 CUDA Graph 跨变体去重
 * @version 4.20.2
 * @date 2026-05-13
 * @author 技术觉醒团队
 * @note 依赖项: <cstdint>, <functional>
 * @note 所属系列: graph
 * @note ShapeId是显式(N,H,W,C)四元组，零碰撞零hash，确定性去重的唯一key
 */

#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace tr {

/**
 * @brief 形状去重键
 *
 * 显式四元组 (N, H, W, C)，用于 CUDA Graph 的跨变体去重：
 *   - 输入 ShapeId 相同 → 所有中间 DTensor shape 相同 → 同一 CapturedGraph → 自动复用
 *
 * 为什么不使用 hash 或指针？
 *   - hash：有碰撞风险，需处理冲突
 *   - MemoryPlan 指针：不同变体 MemoryPlan 地址不同，即使 shape 相同也无法去重
 *   - 显式四元组：确定性、可跨变体比较、零碰撞
 */
struct ShapeId {
    int32_t n = 0;  ///< batch 维度
    int32_t h = 0;  ///< 高度
    int32_t w = 0;  ///< 宽度
    int32_t c = 0;  ///< 通道数

    bool operator==(const ShapeId& o) const noexcept {
        return n == o.n && h == o.h && w == o.w && c == o.c;
    }

    bool operator!=(const ShapeId& o) const noexcept {
        return !(*this == o);
    }

    std::string to_string() const {
        return "{" + std::to_string(n) + "," + std::to_string(h)
               + "," + std::to_string(w) + "," + std::to_string(c) + "}";
    }
};

/**
 * @brief ShapeId 哈希函数 —— 用于 unordered_map 去重
 *
 * 采用 hash_combine 风格（golden ratio 0x9e3779b9），
 * 避免简单 XOR 的相互抵消问题。
 */
struct ShapeIdHash {
    size_t operator()(const ShapeId& s) const noexcept {
        size_t h = 0;
        h ^= std::hash<int32_t>{}(s.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(s.h) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(s.w) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(s.c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

/**
 * @brief 形状无关图专用 ShapeId
 *
 * TRANSFER, COMM, OPTIMIZER, EMA_UPDATE 等子图的拓扑不依赖输入形状。
 * 全部 6 变体共享同一 kShapeInvariant → Phase B 必然碰撞 → 全局复用 7 张图。
 */
constexpr ShapeId kShapeInvariant{0, 0, 0, 0};

/**
 * @brief 内存范围描述符 — (起始偏移, 字节大小, 起始区域ID, 结束区域ID)
 *
 * 用于 GraphNode 的 RANGE 态，描述 RangeOp 的操作范围。
 * 与 OpSegment（start/end 半开区间）不同，MemRange 用 (offset, size) 更能直接匹配 CUDA kernel 参数。
 *
 * V4.20.8 修改：增加区域ID信息以支持调试输出
 */
struct MemRange {
    uint64_t offset = 0;  ///< 起始字节偏移
    uint64_t size   = 0;  ///< 字节大小
    int32_t  start_region_id = -1;  ///< 起始区域ID
    int32_t  end_region_id = -1;    ///< 结束区域ID
};

} // namespace tr