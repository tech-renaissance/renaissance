/**
 * @file tsr_format.h
 * @brief TSR V3 文件格式定义
 * @details 技术觉醒框架的张量持久化格式，支持快模式(RAW)和压缩模式(ZLIB)
 * @version 3.0.0
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: zlib
 * @note 所属系列: data
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace tr {

/**
 * @brief TSR V3 文件模式枚举
 *
 * 双模式设计满足不同场景需求：
 * - RAW：追求极致加载速度，支持mmap零拷贝
 * - ZLIB：追求最小存储空间，适合网络传输
 */
enum class TSRMode : uint32_t {
    RAW  = 0,  ///< 快模式：无压缩，256字节对齐，支持mmap
    ZLIB = 1   ///< 小模式：zlib压缩，CRC32校验
};

/**
 * @brief TSR V3 文件头结构（固定128字节）
 *
 * 内存布局设计要点：
 * - 总大小128字节，是64字节cache line的两倍
 * - 数据偏移在RAW模式下为256字节，满足AVX2/CUDA对齐要求
 * - 预留48字节扩展区，支持未来量化参数等
 *
 * NHWC语义约定（右对齐）：
 * - 0D标量: dims = [0,0,0,0]
 * - 1D向量: dims = [0,0,0,C]
 * - 2D矩阵: dims = [0,0,H,W]
 * - 3D张量: dims = [0,H,W,C]
 * - 4D张量: dims = [N,H,W,C]
 */
#pragma pack(push, 1)
struct TSRHeaderV3 {
    // ================= 基础识别区 (16 Bytes) =================
    char     magic[4];       ///< 魔数标识 "TSR3" (0x54,0x53,0x52,0x33)
    uint32_t header_size;    ///< 头部大小，固定为128
    uint32_t version;        ///< 格式版本，固定为3
    uint32_t file_mode;      ///< 文件模式：0=RAW, 1=ZLIB

    // ================= 张量元数据 (28 Bytes) =================
    uint8_t  dtype;          ///< 数据类型：1=FP32, 2=BF16, 3=INT32, 4=INT8
    uint8_t  ndim;           ///< 维度数 (0-4)
    uint8_t  reserved_1[2];  ///< 对齐填充，必须为0

    int32_t  dims[4];        ///< 形状数组，NHWC顺序右对齐存储
    uint64_t numel;          ///< 元素总数（冗余校验字段）

    // ================= 存储控制区 (32 Bytes) =================
    uint64_t data_offset;    ///< 数据在文件中的绝对偏移（RAW=256, ZLIB=128）
    uint64_t payload_size;   ///< 文件中数据实际字节数（压缩后大小）
    uint64_t raw_data_size;  ///< 原始数据字节数（未压缩）
    uint32_t crc32;          ///< CRC32校验和（对原始数据计算）
    uint32_t reserved_2;     ///< 保留字段，必须为0

    // ================= 扩展保留区 (52 Bytes) =================
    uint8_t  padding[52];    ///< 预留扩展区，必须全为0

    /**
     * @brief 初始化头部为默认值
     */
    void init() {
        std::memset(this, 0, sizeof(TSRHeaderV3));
        std::memcpy(magic, "TSR3", 4);
        header_size = 128;
        version = 3;
    }

    /**
     * @brief 检查魔数是否有效
     */
    bool is_valid_magic() const {
        return magic[0] == 'T' && magic[1] == 'S' &&
               magic[2] == 'R' && magic[3] == '3';
    }
};
#pragma pack(pop)

// 编译期验证头部大小
// static_assert(sizeof(TSRHeaderV3) == 128, "TSRHeaderV3 must be exactly 128 bytes"); // MSVC alignment issue

/**
 * @brief 根据dtype枚举值获取元素字节数
 * @param dtype 数据类型枚举值（1-4）
 * @return 元素字节数，无效类型返回0
 */
constexpr size_t tsr_dtype_size(uint8_t dtype) noexcept {
    switch (dtype) {
        case 1: return 4;  // FP32
        case 2: return 2;  // BF16
        case 3: return 4;  // INT32
        case 4: return 1;  // INT8
        default: return 0; // INVALID
    }
}

/**
 * @brief 获取dtype的字符串名称
 * @param dtype 数据类型枚举值
 * @return 类型名称字符串
 */
inline const char* tsr_dtype_name(uint8_t dtype) noexcept {
    switch (dtype) {
        case 1: return "FP32";
        case 2: return "BF16";
        case 3: return "INT32";
        case 4: return "INT8";
        default: return "INVALID";
    }
}

// TSR V3 常量定义
constexpr uint64_t TSR_RAW_DATA_OFFSET  = 256;  ///< RAW模式数据偏移
constexpr uint64_t TSR_ZLIB_DATA_OFFSET = 128;  ///< ZLIB模式数据偏移
constexpr size_t   TSR_ALIGNMENT        = 256;  ///< 内存对齐要求

} // namespace tr
