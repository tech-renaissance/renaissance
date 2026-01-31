/**
 * @file file_handle.h
 * @brief 跨平台文件句柄RAII封装
 * @details 提供Windows和Linux平台的文件操作统一接口
 * @version 1.0.0
 * @date 2026-01-23
 * @author 技术觉醒团队
 * @note 所属系统: data
 */

#pragma once

#include <string>
#include <cstdint>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifdef ERROR
        #undef ERROR
    #endif
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace tr {

// =============================================================================
// DTS文件头结构（与make_dataset.py一致）
// =============================================================================

// MNIST/CIFAR DTS文件头（256 byte header）
#pragma pack(push, 1)
struct SmallDtsHeader {
    uint8_t  magic[4];
    uint8_t  version[4];
    uint8_t  dataset_type[8];
    uint32_t is_training;
    uint32_t compress_level;
    uint32_t val_set_prep;
    uint32_t num_classes;
    uint8_t  tensor_layout[4];
    uint32_t image_width;
    uint32_t image_height;
    uint32_t num_channels;
    uint8_t  color_channel_type[4];
    uint32_t num_samples;
    uint32_t num_volumes;
    uint32_t volume_id;
    uint32_t num_blocks;
    uint32_t dummy;  // 填充字段，保持与ImageNet DtsHeader布局一致
    uint64_t total_bytes;
    uint32_t header_size;
    uint64_t block_bytes;
    uint32_t bytes_per_block;
    uint32_t block_header_size;
    uint32_t pic_alignment;
    uint32_t max_pic_area;
    uint32_t max_pic_per_block;
    float    compression_ratio;
    float    normalize_mean[3];
    float    normalize_std[3];
    uint32_t crc_code;
};
#pragma pack(pop)

static_assert(sizeof(SmallDtsHeader) == 144, "SmallDtsHeader must be exactly 144 bytes");

// =============================================================================
// FileHandle类
// =============================================================================

/**
 * @class FileHandle
 * @brief 跨平台文件句柄RAII封装
 *
 * 功能：
 * - Windows: 使用CreateFileA/ReadFile/CloseHandle
 * - Linux: 使用open/pread/close
 * - RAII自动管理资源
 */
class FileHandle {
public:
#ifdef _WIN32
    using HandleType = HANDLE;
    static constexpr HandleType INVALID_VALUE = INVALID_HANDLE_VALUE;
#else
    using HandleType = int;
    static constexpr HandleType INVALID_VALUE = -1;
#endif

    /**
     * @brief 构造函数：打开文件
     * @param path 文件路径
     * @throws FileNotFoundError 如果文件打开失败
     */
    explicit FileHandle(const std::string& path);

    /**
     * @brief 析构函数：自动关闭文件
     */
    ~FileHandle();

    // 禁止拷贝
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // 允许移动
    FileHandle(FileHandle&& other) noexcept;
    FileHandle& operator=(FileHandle&& other) noexcept;

    /**
     * @brief 获取底层句柄
     * @return 原生文件句柄
     */
    HandleType get() const { return handle_; }

    /**
     * @brief 检查句柄是否有效
     * @return true=有效, false=无效
     */
    bool is_valid() const { return handle_ != INVALID_VALUE; }

private:
    HandleType handle_;
};

} // namespace tr
