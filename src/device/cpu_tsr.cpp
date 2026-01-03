/**
 * @file cpu_device_tsr.cpp
 * @brief CpuDevice TSR V3 导入导出实现
 * @details 实现张量的高效持久化，支持快模式和压缩模式
 * @version 3.6.9
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: zlib, cpu_device.h, tsr_format.h
 * @note 所属系列: device
 */

#include "renaissance/device/cpu_device.h"
#include "renaissance/data/tsr_format.h"
#include "renaissance/data/tensor.h"
#include "renaissance/data/storage.h"
#include "renaissance/data/shape.h"
#include "renaissance/base/dtype.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"

#include <fstream>
#include <vector>
#include <cstring>
#include <zlib.h>

// ============================================================================
// 平台相关的内存映射支持
// ============================================================================

#if defined(__linux__)
    #define TR_HAS_MMAP 1
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#elif defined(_WIN32)
    #define TR_HAS_MMAP 1
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #define TR_HAS_MMAP 0
#endif

namespace tr {

// ============================================================================
// 内存映射句柄封装（RAII）
// ============================================================================

#if TR_HAS_MMAP

/**
 * @brief 内存映射句柄类（RAII封装）
 *
 * 用于管理mmap资源的生命周期，确保正确释放。
 * 作为Storage的holder，当最后一个引用消失时自动unmap。
 */
class MmapHandle {
public:
#if defined(__linux__)
    // Linux实现
    MmapHandle(void* base, size_t size, int fd)
        : base_(base), size_(size), fd_(fd) {}
#elif defined(_WIN32)
    // Windows实现
    MmapHandle(void* base, size_t size, HANDLE file_handle, HANDLE mapping_handle)
        : base_(base), size_(size), fd_(-1),
          file_handle_(file_handle), mapping_handle_(mapping_handle) {}
#endif

    ~MmapHandle() {
#if defined(__linux__)
        if (base_ && base_ != MAP_FAILED) {
            munmap(base_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
#endif

#if defined(_WIN32)
        if (base_) {
            UnmapViewOfFile(base_);
        }
        if (mapping_handle_) {
            CloseHandle(mapping_handle_);
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
        }
#endif
    }

    // 禁止拷贝
    MmapHandle(const MmapHandle&) = delete;
    MmapHandle& operator=(const MmapHandle&) = delete;

    void* base() const { return base_; }
    size_t size() const { return size_; }

private:
    void* base_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;

#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#endif
};

#endif // TR_HAS_MMAP

// ============================================================================
// 导出实现
// ============================================================================

void CpuDevice::export_tensor(const Tensor& tensor, const std::string& filename,
                              bool compress) const {
    // -------------------------------------------------------------------------
    // 1. 输入验证
    // -------------------------------------------------------------------------
    if (!tensor.is_valid()) {
        TR_VALUE_ERROR("Cannot export invalid tensor (dtype=INVALID)");
    }
    if (!tensor.is_bound()) {
        TR_VALUE_ERROR("Cannot export unbound tensor (storage not attached)");
    }
    if (!tensor.is_cpu()) {
        TR_DEVICE_ERROR("export_tensor only supports CPU tensors, got "
                 << tensor.device_type().to_string());
    }
    if (tensor.numel() == 0) {
        TR_VALUE_ERROR("Cannot export empty tensor (numel=0)");
    }

    // -------------------------------------------------------------------------
    // 2. 构造文件头
    // -------------------------------------------------------------------------
    TSRHeaderV3 header;
    header.init();

    // 文件模式
    header.file_mode = compress ? static_cast<uint32_t>(TSRMode::ZLIB)
                                : static_cast<uint32_t>(TSRMode::RAW);

    // 数据类型（与框架DType枚举值对应）
    header.dtype = static_cast<uint8_t>(tensor.dtype());

    // 维度数
    header.ndim = static_cast<uint8_t>(tensor.ndim());

    // NHWC形状填充（右对齐）
    const Shape& shape = tensor.shape();

    // 初始化为0
    header.dims[0] = 0;
    header.dims[1] = 0;
    header.dims[2] = 0;
    header.dims[3] = 0;

    // 根据维度数填充（右对齐到NHWC）
    switch (header.ndim) {
        case 0:
            // 标量：全为0
            break;
        case 1:
            // 1D: [0,0,0,C]
            header.dims[3] = shape.c();
            break;
        case 2:
            // 2D: [0,0,H,W]
            header.dims[2] = shape.h();
            header.dims[3] = shape.w();
            break;
        case 3:
            // 3D: [0,H,W,C]
            header.dims[1] = shape.h();
            header.dims[2] = shape.w();
            header.dims[3] = shape.c();
            break;
        case 4:
            // 4D: [N,H,W,C]
            header.dims[0] = shape.n();
            header.dims[1] = shape.h();
            header.dims[2] = shape.w();
            header.dims[3] = shape.c();
            break;
        default:
            TR_VALUE_ERROR("Invalid tensor ndim: " << static_cast<int>(header.ndim));
    }

    // 元素总数和数据大小
    header.numel = static_cast<uint64_t>(tensor.numel());
    header.raw_data_size = static_cast<uint64_t>(tensor.nbytes());

    // -------------------------------------------------------------------------
    // 3. 打开输出文件
    // -------------------------------------------------------------------------
    std::ofstream fs(filename, std::ios::binary | std::ios::trunc);
    if (!fs.is_open()) {
        TR_FILE_NOT_FOUND("Failed to create file: " << filename);
    }

    // -------------------------------------------------------------------------
    // 4. 根据模式写入数据
    // -------------------------------------------------------------------------
    try {
        const void* data_ptr = tensor.data_ptr();
        size_t data_size = tensor.nbytes();

        if (compress) {
            // ===== ZLIB压缩模式 =====
            LOG_INFO << "Exporting tensor to " << filename
                     << " (ZLIB mode, shape=" << shape.to_string()
                     << ", dtype=" << dtype_name(tensor.dtype()) << ")";

            header.data_offset = TSR_ZLIB_DATA_OFFSET;  // 128

            // 压缩数据
            std::vector<uint8_t> compressed = compress_zlib(data_ptr, data_size);
            header.payload_size = static_cast<uint64_t>(compressed.size());

            // 计算原始数据的CRC32
            header.crc32 = calculate_crc32(data_ptr, data_size);

            // 写入头部（强制写入128字节）
            fs.write(reinterpret_cast<const char*>(&header), 128);

            // 写入压缩数据（紧随头部）
            fs.write(reinterpret_cast<const char*>(compressed.data()),
                     compressed.size());

            // 计算压缩率
            double ratio = (data_size > 0)
                ? (100.0 * compressed.size() / data_size)
                : 0.0;

            LOG_INFO << "Compressed: " << data_size << " -> "
                     << compressed.size() << " bytes (" << ratio << "%)";

        } else {
            // ===== RAW快速模式 =====
            LOG_INFO << "Exporting tensor to " << filename
                     << " (RAW mode, shape=" << shape.to_string()
                     << ", dtype=" << dtype_name(tensor.dtype()) << ")";

            header.data_offset = TSR_RAW_DATA_OFFSET;  // 256
            header.payload_size = header.raw_data_size;

            // RAW模式也计算CRC32（可选但推荐，便于数据完整性验证）
            header.crc32 = calculate_crc32(data_ptr, data_size);

            LOG_DEBUG << "CRC32 calculated: " << header.crc32;
            LOG_DEBUG << "Data size: " << data_size << " bytes";

            // 写入头部（强制写入128字节）
            fs.write(reinterpret_cast<const char*>(&header), 128);

            // 写入对齐填充（128字节零填充，使数据偏移达到256）
            constexpr size_t padding_size = TSR_RAW_DATA_OFFSET - 128;  // 固定128字节header
            // static_assert(padding_size == 128, "Padding should be 128 bytes"); // MSVC alignment issue

            char padding[padding_size] = {0};
            fs.write(padding, padding_size);

            // 写入原始数据
            fs.write(reinterpret_cast<const char*>(data_ptr), data_size);

            LOG_INFO << "Written: " << data_size << " bytes at offset "
                     << TSR_RAW_DATA_OFFSET;
        }

        fs.flush();

        if (!fs.good()) {
            TR_VALUE_ERROR("Write error occurred while exporting to: " << filename);
        }

        fs.close();
        LOG_INFO << "Export successful: " << filename;

    } catch (const TRException&) {
        fs.close();
        throw;  // 重新抛出框架异常
    } catch (const std::exception& e) {
        fs.close();
        TR_VALUE_ERROR("Export failed: " << e.what());
    }
}

// ============================================================================
// 导入实现
// ============================================================================

Tensor CpuDevice::import_tensor(const std::string& filename, bool using_mmap) {
    // -------------------------------------------------------------------------
    // 1. 打开文件并读取头部
    // -------------------------------------------------------------------------
    std::ifstream fs(filename, std::ios::binary | std::ios::ate);
    if (!fs.is_open()) {
        TR_FILE_NOT_FOUND("File not found: " << filename);
    }

    // 获取文件大小
    std::streamsize file_size = fs.tellg();
    fs.seekg(0, std::ios::beg);

    if (file_size < static_cast<std::streamsize>(sizeof(TSRHeaderV3))) {
        TR_VALUE_ERROR("File too small to be a valid TSR file: " << filename
                 << " (size=" << file_size << ", need at least " << sizeof(TSRHeaderV3) << ")");
    }

    // 读取头部
    TSRHeaderV3 header;
    fs.read(reinterpret_cast<char*>(&header), sizeof(header));

    LOG_DEBUG << "Header size in file: " << sizeof(header) << " bytes";
    LOG_DEBUG << "Data offset from header: " << header.data_offset << " bytes";

    if (fs.gcount() != sizeof(header)) {
        TR_VALUE_ERROR("Failed to read TSR header from: " << filename);
    }

    // -------------------------------------------------------------------------
    // 2. 验证头部
    // -------------------------------------------------------------------------
    validate_tsr_header(header);

    // 验证文件大小是否足够
    uint64_t required_size = header.data_offset + header.payload_size;
    if (static_cast<uint64_t>(file_size) < required_size) {
        TR_VALUE_ERROR("TSR file truncated: " << filename
                 << " (file_size=" << file_size << ", required=" << required_size << ")");
    }

    // -------------------------------------------------------------------------
    // 3. 解析形状
    // -------------------------------------------------------------------------
    Shape shape;
    DType dtype = static_cast<DType>(header.dtype);

    switch (header.ndim) {
        case 0:
            shape = Shape();  // 标量
            break;
        case 1:
            shape = Shape(header.dims[3]);  // 1D: (C)
            break;
        case 2:
            shape = Shape(header.dims[2], header.dims[3]);  // 2D: (H,W)
            break;
        case 3:
            shape = Shape(header.dims[1], header.dims[2], header.dims[3]);  // 3D: (H,W,C)
            break;
        case 4:
            shape = Shape(header.dims[0], header.dims[1],
                         header.dims[2], header.dims[3]);  // 4D: (N,H,W,C)
            break;
        default:
            TR_VALUE_ERROR("Invalid ndim in TSR file: "
                     << static_cast<int>(header.ndim));
    }

    LOG_INFO << "Importing tensor from " << filename
             << " (mode=" << (header.file_mode == 0 ? "RAW" : "ZLIB")
             << ", shape=" << shape.to_string()
             << ", dtype=" << dtype_name(dtype) << ")";

    // -------------------------------------------------------------------------
    // 4. 根据模式加载数据
    // -------------------------------------------------------------------------
    if (header.file_mode == static_cast<uint32_t>(TSRMode::RAW)) {
        // ===== RAW模式 =====
        fs.close();  // 关闭流，准备使用mmap或重新打开

#if TR_HAS_MMAP
        // ----- 方案A: mmap零拷贝（可选） -----
        if (using_mmap) {

#if defined(__linux__)
        // Linux mmap
        int fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1) {
            TR_FILE_NOT_FOUND("Failed to open file for mmap: " << filename);
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            TR_VALUE_ERROR("Failed to stat file: " << filename);
        }

        // 映射整个文件（只读，私有映射）
        void* map_base = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map_base == MAP_FAILED) {
            close(fd);
            TR_MEMORY_ERROR("mmap failed for file: " << filename);
        }

        // 创建MmapHandle（RAII管理）
        auto mmap_handle = std::make_shared<MmapHandle>(map_base, sb.st_size, fd);

#elif defined(_WIN32)
        // Windows内存映射文件
        HANDLE file_handle = CreateFileA(
            filename.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );

        if (file_handle == INVALID_HANDLE_VALUE) {
            TR_FILE_NOT_FOUND("Failed to open file for mapping: " << filename);
        }

        LARGE_INTEGER file_size_li;
        if (!GetFileSizeEx(file_handle, &file_size_li)) {
            CloseHandle(file_handle);
            TR_VALUE_ERROR("Failed to get file size: " << filename);
        }

        HANDLE mapping_handle = CreateFileMappingA(
            file_handle,
            nullptr,
            PAGE_READONLY,
            0, 0,
            nullptr
        );

        if (!mapping_handle) {
            CloseHandle(file_handle);
            TR_MEMORY_ERROR("CreateFileMapping failed for: " << filename);
        }

        void* map_base = MapViewOfFile(
            mapping_handle,
            FILE_MAP_READ,
            0, 0, 0
        );

        if (!map_base) {
            CloseHandle(mapping_handle);
            CloseHandle(file_handle);
            TR_MEMORY_ERROR("MapViewOfFile failed for: " << filename);
        }

        // 创建MmapHandle（RAII管理）
        auto mmap_handle = std::make_shared<MmapHandle>(
            map_base,
            static_cast<size_t>(file_size_li.QuadPart),
            file_handle,
            mapping_handle
        );
#endif

        // 数据指针（偏移256字节）
        void* data_ptr = static_cast<char*>(mmap_handle->base()) + header.data_offset;

        // 检查对齐（调试信息）
        if (reinterpret_cast<uintptr_t>(data_ptr) % TSR_ALIGNMENT != 0) {
            LOG_WARN << "mmap data not " << TSR_ALIGNMENT
                     << "-byte aligned (this may affect SIMD performance)";
        }

        // 可选：验证CRC32
        if (header.crc32 != 0) {
            uint32_t computed_crc = calculate_crc32(data_ptr, header.raw_data_size);
            if (computed_crc != header.crc32) {
                TR_VALUE_ERROR("CRC32 mismatch in RAW mode: file may be corrupted "
                         << "(expected=" << header.crc32 << ", got=" << computed_crc << ")");
            }
            LOG_DEBUG << "CRC32 verified: " << header.crc32;
        }

        // 创建shared_ptr<void>作为holder
        // 当Tensor/Storage销毁时，MmapHandle会自动释放映射
        std::shared_ptr<void> holder(
            data_ptr,
            [mmap_handle](void*) {
                // data_ptr不需要单独释放，MmapHandle析构时会处理
                // 这个lambda只是捕获mmap_handle以延长其生命周期
            }
        );

        // 创建Storage（持有模式，通过holder管理生命周期）
        auto storage = std::make_shared<Storage>(
            data_ptr,
            header.raw_data_size,
            DeviceType::cpu(),
            holder
        );

        // 创建Tensor
        Tensor tensor(shape, dtype, DeviceType::cpu(), storage, 0, false);

        LOG_INFO << "Loaded via mmap (zero-copy), size=" << header.raw_data_size << " bytes";
        return tensor;
        }  // end of using_mmap

        // ----- 方案B: 常规读取（不使用mmap或平台不支持） -----
        std::ifstream fs2(filename, std::ios::binary);
        if (!fs2.is_open()) {
            TR_FILE_NOT_FOUND("Failed to reopen file: " << filename);
        }

        // 创建目标张量
        Tensor tensor = zeros(shape, dtype);

        // 定位到数据偏移
        fs2.seekg(header.data_offset);

        // 读取数据
        fs2.read(reinterpret_cast<char*>(tensor.data_ptr()), header.raw_data_size);

        if (static_cast<size_t>(fs2.gcount()) != header.raw_data_size) {
            TR_VALUE_ERROR("Failed to read complete data from TSR file");
        }

        fs2.close();

        // 可选：验证CRC32
        if (header.crc32 != 0) {
            uint32_t computed_crc = calculate_crc32(tensor.data_ptr(), header.raw_data_size);
            if (computed_crc != header.crc32) {
                TR_VALUE_ERROR("CRC32 mismatch: file may be corrupted");
            }
        }

        LOG_INFO << "Loaded via read (fallback), size=" << header.raw_data_size << " bytes";
        return tensor;
#else
        // ----- 方案C: 平台不支持mmap，使用常规读取 -----
        std::ifstream fs2(filename, std::ios::binary);
        if (!fs2.is_open()) {
            TR_FILE_NOT_FOUND("Failed to reopen file: " << filename);
        }

        // 创建目标张量
        Tensor tensor = zeros(shape, dtype);

        // 定位到数据偏移
        fs2.seekg(header.data_offset);

        // 读取数据
        fs2.read(reinterpret_cast<char*>(tensor.data_ptr()), header.raw_data_size);

        if (static_cast<size_t>(fs2.gcount()) != header.raw_data_size) {
            TR_VALUE_ERROR("Failed to read complete data from TSR file");
        }

        fs2.close();

        // 可选：验证CRC32
        if (header.crc32 != 0) {
            uint32_t computed_crc = calculate_crc32(tensor.data_ptr(), header.raw_data_size);
            if (computed_crc != header.crc32) {
                TR_VALUE_ERROR("CRC32 mismatch: file may be corrupted");
            }
        }

        LOG_INFO << "Loaded via read (no mmap support), size=" << header.raw_data_size << " bytes";
        return tensor;
#endif  // TR_HAS_MMAP

    } else if (header.file_mode == static_cast<uint32_t>(TSRMode::ZLIB)) {
        // ===== ZLIB模式 =====

        // 读取压缩数据
        std::vector<uint8_t> compressed(header.payload_size);
        fs.seekg(header.data_offset);
        fs.read(reinterpret_cast<char*>(compressed.data()), header.payload_size);

        if (static_cast<size_t>(fs.gcount()) != header.payload_size) {
            TR_VALUE_ERROR("Failed to read compressed data from TSR file: " << filename);
        }
        fs.close();

        // 创建目标张量
        Tensor tensor = zeros(shape, dtype);

        // 解压到张量
        decompress_zlib(
            compressed.data(),
            compressed.size(),
            tensor.data_ptr(),
            header.raw_data_size
        );

        // CRC32校验（ZLIB模式必须验证）
        uint32_t computed_crc = calculate_crc32(tensor.data_ptr(), header.raw_data_size);
        if (computed_crc != header.crc32) {
            TR_VALUE_ERROR("CRC32 mismatch: file corrupted or decompression error "
                     << "(expected=" << header.crc32 << ", got=" << computed_crc << ")");
        }

        LOG_INFO << "Loaded and decompressed: " << compressed.size()
                 << " -> " << header.raw_data_size << " bytes, CRC32 OK";

        return tensor;

    } else {
        TR_VALUE_ERROR("Unknown TSR file mode: " << header.file_mode);
    }
}

// ============================================================================
// 辅助方法实现
// ============================================================================

void CpuDevice::validate_tsr_header(const TSRHeaderV3& header) const {
    // 魔数校验
    if (!header.is_valid_magic()) {
        TR_VALUE_ERROR("Invalid TSR magic number (expected 'TSR3')");
    }

    // 版本校验
    if (header.version != 3) {
        TR_VALUE_ERROR("Unsupported TSR version: " << header.version
                 << " (this implementation supports version 3)");
    }

    // 头部大小校验
    if (header.header_size != 128) {
        TR_VALUE_ERROR("Invalid TSR header size: " << header.header_size
                 << " (expected 128)");
    }

    // 模式校验
    if (header.file_mode != 0 && header.file_mode != 1) {
        TR_VALUE_ERROR("Invalid TSR file mode: " << header.file_mode
                 << " (expected 0=RAW or 1=ZLIB)");
    }

    // 数据类型校验（必须是1-4）
    if (header.dtype < 1 || header.dtype > 4) {
        TR_VALUE_ERROR("Invalid TSR dtype: " << static_cast<int>(header.dtype)
                 << " (expected 1=FP32, 2=BF16, 3=INT32, 4=INT8)");
    }

    // 维度数校验
    if (header.ndim > 4) {
        TR_VALUE_ERROR("Invalid TSR ndim: " << static_cast<int>(header.ndim)
                 << " (expected 0-4)");
    }

    // 数据偏移校验
    if (header.file_mode == 0 && header.data_offset != TSR_RAW_DATA_OFFSET) {
        TR_VALUE_ERROR("Invalid data_offset for RAW mode: " << header.data_offset
                 << " (expected " << TSR_RAW_DATA_OFFSET << ")");
    }
    if (header.file_mode == 1 && header.data_offset != TSR_ZLIB_DATA_OFFSET) {
        TR_VALUE_ERROR("Invalid data_offset for ZLIB mode: " << header.data_offset
                 << " (expected " << TSR_ZLIB_DATA_OFFSET << ")");
    }

    // 元素数量校验
    int64_t expected_numel = 1;
    bool has_dim = false;
    for (int i = 0; i < 4; ++i) {
        if (header.dims[i] > 0) {
            expected_numel *= header.dims[i];
            has_dim = true;
        } else if (header.dims[i] < 0) {
            TR_VALUE_ERROR("Negative dimension in TSR file: dims[" << i << "]="
                     << header.dims[i]);
        }
    }

    // 标量情况特殊处理（所有dims为0，numel应为1）
    if (!has_dim && header.ndim == 0) {
        expected_numel = 1;
    }

    if (expected_numel != static_cast<int64_t>(header.numel)) {
        TR_VALUE_ERROR("TSR numel mismatch: dims product=" << expected_numel
                 << ", header.numel=" << header.numel);
    }

    // 数据大小校验
    size_t elem_size = tsr_dtype_size(header.dtype);
    if (elem_size == 0) {
        TR_VALUE_ERROR("Unknown dtype size for dtype="
                 << static_cast<int>(header.dtype));
    }

    size_t expected_size = header.numel * elem_size;
    if (expected_size != header.raw_data_size) {
        TR_VALUE_ERROR("TSR raw_data_size mismatch: expected=" << expected_size
                 << " (" << header.numel << " * " << elem_size << "), got=" << header.raw_data_size);
    }

    // 保留字段校验（必须为0）
    if (header.reserved_1[0] != 0 || header.reserved_1[1] != 0) {
        LOG_WARN << "TSR reserved_1 fields are non-zero (may be from future version)";
    }
    if (header.reserved_2 != 0) {
        LOG_WARN << "TSR reserved_2 field is non-zero (may be from future version)";
    }
}

uint32_t CpuDevice::calculate_crc32(const void* data, size_t size) const {
    // 使用zlib提供的crc32函数
    // 初始化CRC为0
    uLong crc = crc32(0L, Z_NULL, 0);

    // 计算数据的CRC32
    crc = crc32(crc, static_cast<const Bytef*>(data), static_cast<uInt>(size));

    return static_cast<uint32_t>(crc);
}

std::vector<uint8_t> CpuDevice::compress_zlib(const void* data, size_t size) const {
    if (size == 0) {
        return std::vector<uint8_t>();
    }

    // 预估压缩后的最大大小
    // compressBound返回压缩后数据的最大可能大小
    uLongf compressed_bound = compressBound(static_cast<uLong>(size));
    std::vector<uint8_t> compressed(compressed_bound);

    uLongf compressed_size = compressed_bound;

    // 使用默认压缩级别（Z_DEFAULT_COMPRESSION = 6）
    int ret = compress2(
        compressed.data(),
        &compressed_size,
        static_cast<const Bytef*>(data),
        static_cast<uLong>(size),
        Z_DEFAULT_COMPRESSION
    );

    if (ret != Z_OK) {
        const char* err_msg = "unknown error";
        switch (ret) {
            case Z_MEM_ERROR: err_msg = "out of memory"; break;
            case Z_BUF_ERROR: err_msg = "buffer too small"; break;
            case Z_STREAM_ERROR: err_msg = "invalid compression level"; break;
        }
        TR_VALUE_ERROR("zlib compression failed: " << err_msg << " (code=" << ret << ")");
    }

    // 调整到实际压缩后的大小
    compressed.resize(compressed_size);

    return compressed;
}

void CpuDevice::decompress_zlib(const uint8_t* src, size_t src_size,
                                void* dst, size_t dst_size) const {
    if (src_size == 0 || dst_size == 0) {
        if (src_size == 0 && dst_size == 0) {
            return;  // 空数据，无需处理
        }
        TR_VALUE_ERROR("Invalid decompression parameters: src_size=" << src_size
                 << ", dst_size=" << dst_size);
    }

    uLongf uncompressed_size = static_cast<uLongf>(dst_size);

    int ret = uncompress(
        static_cast<Bytef*>(dst),
        &uncompressed_size,
        src,
        static_cast<uLong>(src_size)
    );

    if (ret != Z_OK) {
        const char* err_msg = "unknown error";
        switch (ret) {
            case Z_MEM_ERROR: err_msg = "out of memory"; break;
            case Z_BUF_ERROR: err_msg = "output buffer too small"; break;
            case Z_DATA_ERROR: err_msg = "compressed data corrupted"; break;
        }
        TR_VALUE_ERROR("zlib decompression failed: " << err_msg << " (code=" << ret << ")");
    }

    if (uncompressed_size != dst_size) {
        TR_VALUE_ERROR("Decompressed size mismatch: expected=" << dst_size
                 << ", actual=" << uncompressed_size);
    }
}

} // namespace tr
