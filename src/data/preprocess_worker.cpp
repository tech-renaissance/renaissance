/**
 * @file preprocess_worker.cpp
 * @brief 预处理工作器实现
 * @version 2.1.0（测试模式版）
 * @date 2026-02-17
 * @author 技术觉醒团队
 */

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <turbojpeg.h>

#ifdef _WIN32
    #include <windows.h>
#endif

#include "renaissance/data/preprocess_worker.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/global_registry.h"
#include "renaissance/base/philox.h"
#include <chrono>

namespace tr {

// ============================================================================
// 辅助函数
// ============================================================================

namespace {
    // MCU对齐辅助函数
    constexpr int MCU_SIZE = 16;

    int32_t align_down_mcu(int32_t value) {
        return (value / MCU_SIZE) * MCU_SIZE;
    }

    int32_t align_up_mcu(int32_t value) {
        return ((value + MCU_SIZE - 1) / MCU_SIZE) * MCU_SIZE;
    }

    // 64字节对齐
    size_t align_64(size_t s) {
        return (s + 63) & ~size_t(63);
    }

    // 4KB页对齐
    size_t align_4k(size_t s) {
        return (s + 4095) & ~size_t(4095);
    }

    /**
     * @brief STB JPEG解码（fallback函数）
     */
    bool decode_jpeg_with_stb(
        const uint8_t* jpeg_data,
        size_t jpeg_size,
        uint8_t* decode_buffer,
        size_t buffer_size,
        int& width,
        int& height
    ) {
        int stb_width, stb_height, stb_channels;
        stbi_uc* stb_data = stbi_load_from_memory(
            reinterpret_cast<const stbi_uc*>(jpeg_data),
            static_cast<int>(jpeg_size),
            &stb_width, &stb_height, &stb_channels,
            3  // 强制3通道RGB
        );

        if (!stb_data) {
            return false;  // STB也无法解码
        }

        width = stb_width;
        height = stb_height;
        int pitch = width * 3;  // RGB = 3通道

        // 验证缓冲区容量
        size_t required_size = pitch * height;
        if (required_size > buffer_size) {
            stbi_image_free(stb_data);
            return false;
        }

        // 复制到目标缓冲区
        for (int y = 0; y < height; ++y) {
            std::memcpy(decode_buffer + y * pitch,
                      stb_data + y * stb_width * 3,
                      pitch);
        }

        stbi_image_free(stb_data);
        return true;
    }
}

// ============================================================================
// 构造函数和析构函数
// ============================================================================

PreprocessWorker::PreprocessWorker(
    const Config& config,
    const std::vector<std::unique_ptr<PreprocessOperation>>& train_ops,
    const std::vector<std::unique_ptr<PreprocessOperation>>& val_ops)
    : config_(config)
    , rng_(0)  // 临时seed，延迟初始化
{
    LOG_DEBUG << "[PW " << config_.worker_id << " CONSTRUCTOR] "
              << "test_mode=" << (config_.test_mode ? "ON" : "OFF")
              << ", engine=" << config_.engine_id
              << ", pid=" << config_.pid_in_engine;

    // ==================== 1. 分配Workshop ====================
    allocate_workshop();

    // ==================== 2. 克隆PO列表 ====================
    train_ops_.reserve(train_ops.size());
    for (const auto& op : train_ops) {
        train_ops_.push_back(op->clone());
    }

    val_ops_.reserve(val_ops.size());
    for (const auto& op : val_ops) {
        val_ops_.push_back(op->clone());
    }

    LOG_DEBUG << "PW " << config_.worker_id << " cloned POs: "
              << train_ops_.size() << " train, "
              << val_ops_.size() << " val";

    // ==================== 3. 初始化TurboJPEG 3.x ====================
    tj_handle_ = tj3Init(TJINIT_DECOMPRESS);
    TR_CHECK(tj_handle_ != nullptr, MemoryError,
             "Failed to initialize TurboJPEG for PW " << config_.worker_id);

    // 设置优化标志
    tj3Set(tj_handle_, TJPARAM_FASTDCT, 1);
    tj3Set(tj_handle_, TJPARAM_FASTUPSAMPLE, 1);

    LOG_DEBUG << "PW " << config_.worker_id << " TurboJPEG 3.x initialized";

    LOG_INFO << "PW " << config_.worker_id << " created"
             << ", workshop=" << (workshop_size_ / (1024.0*1024.0)) << " MB";
}

PreprocessWorker::~PreprocessWorker() {
    // 释放Workshop
    free_workshop();

    // 释放TurboJPEG 3.x句柄
    if (tj_handle_) {
        tj3Destroy(tj_handle_);
        tj_handle_ = nullptr;
    }

    LOG_DEBUG << "PW " << config_.worker_id << " destroyed";
}

// ============================================================================
// 内存分配
// ============================================================================

void PreprocessWorker::allocate_workshop() {
    // ==================== 计算总大小（所有区对齐）====================
    workshop_size_ = align_64(config_.region_d_size) +
                     align_64(config_.region_ab_size) * 2 +
                     align_64(config_.region_t_size);

    // S区对齐到4KB页边界（NUMA优化）
    for (int i = 0; i < config_.num_region_s; ++i) {
        workshop_size_ += align_4k(config_.region_s_size);
    }

    workshop_size_ += align_64(config_.region_c_size);

    // 总大小对齐到4KB页
    workshop_size_ = align_4k(workshop_size_);

    LOG_DEBUG << "PW " << config_.worker_id << " allocating workshop: "
              << (workshop_size_ / (1024.0*1024.0)) << " MB";

    // ==================== 分配内存 ====================
#ifdef _WIN32
    workshop_ = static_cast<uint8_t*>(
        VirtualAlloc(NULL, workshop_size_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    );
    TR_CHECK(workshop_ != nullptr, MemoryError,
             "VirtualAlloc failed for PW " << config_.worker_id
             << "\n  Size: " << (workshop_size_ / (1024.0*1024.0)) << " MB"
             << "\n  Error: " << static_cast<int>(GetLastError()));
#else
    constexpr size_t PAGE_SIZE = 4096;
    int ret = posix_memalign(
        reinterpret_cast<void**>(&workshop_),
        PAGE_SIZE,
        workshop_size_
    );
    TR_CHECK(ret == 0 && workshop_ != nullptr, MemoryError,
             "posix_memalign failed for PW " << config_.worker_id
             << "\n  Size: " << (workshop_size_ / (1024.0*1024.0)) << " MB"
             << "\n  Return code: " << ret);
#endif

    // ==================== 划分各区（64字节对齐）====================
    uint8_t* ptr = workshop_;

    region_d_ = ptr;
    LOG_DEBUG << "PW " << config_.worker_id << " D区: " << static_cast<void*>(ptr)
              << ", size=" << (config_.region_d_size / 1024) << " KB";
    ptr += align_64(config_.region_d_size);

    region_a_ = ptr;
    LOG_DEBUG << "PW " << config_.worker_id << " A区: " << static_cast<void*>(ptr)
              << ", size=" << (config_.region_ab_size / 1024) << " KB";
    ptr += align_64(config_.region_ab_size);

    region_b_ = ptr;
    LOG_DEBUG << "PW " << config_.worker_id << " B区: " << static_cast<void*>(ptr)
              << ", size=" << (config_.region_ab_size / 1024) << " KB";
    ptr += align_64(config_.region_ab_size);

    if (config_.region_t_size > 0) {
        region_t_ = ptr;
        LOG_DEBUG << "PW " << config_.worker_id << " T区: " << static_cast<void*>(ptr)
                  << ", size=" << (config_.region_t_size / 1024) << " KB";
        ptr += align_64(config_.region_t_size);
    }

    // S区对齐到4KB页边界
    region_s_.resize(config_.num_region_s);
    for (int i = 0; i < config_.num_region_s; ++i) {
        region_s_[i] = ptr;
        LOG_DEBUG << "PW " << config_.worker_id << " S" << (i+1) << "区: "
                  << static_cast<void*>(ptr)
                  << ", size=" << (config_.region_s_size / (1024.0*1024.0)) << " MB";
        ptr += align_4k(config_.region_s_size);
    }

    if (config_.region_c_size > 0) {
        region_c_ = ptr;
        LOG_DEBUG << "PW " << config_.worker_id << " C区: " << static_cast<void*>(ptr)
                  << ", size=" << (config_.region_c_size / (1024.0*1024.0)) << " MB";
        ptr += align_64(config_.region_c_size);
    }

    // 验证指针未越界
    TR_CHECK(ptr <= workshop_ + workshop_size_, MemoryError,
             "Workshop allocation overflow: " << (ptr - workshop_)
             << " > " << workshop_size_);
}

void PreprocessWorker::free_workshop() {
    if (workshop_) {
#ifdef _WIN32
        VirtualFree(workshop_, 0, MEM_RELEASE);
#else
        free(workshop_);
#endif
        workshop_ = nullptr;

        LOG_DEBUG << "PW " << config_.worker_id << " workshop freed";
    }
}

// ============================================================================
// 参数更新
// ============================================================================

void PreprocessWorker::update_parameters(const PreprocessWorkerParameter& param) {
    param_ = param;

    // 重置样本计数器（每个phase开始时）
    total_samples_processed_ = 0;

    LOG_DEBUG << "PW " << config_.worker_id << " parameters updated: "
              << (param_.is_train ? "TRAIN" : "VAL");
}

// ============================================================================
// RNG初始化
// ============================================================================

void PreprocessWorker::ensure_rng_initialized() {
    if (!rng_initialized_) {
        uint64_t base_seed = get_default_generator().seed();
        uint64_t worker_seed = base_seed ^ (static_cast<uint64_t>(config_.worker_id) << 16);
        rng_.set_seed(worker_seed);
        rng_initialized_ = true;

        LOG_DEBUG << "PW " << config_.worker_id << " RNG initialized with seed=" << worker_seed;
    }
}

// ============================================================================
// 解码方法
// ============================================================================

bool PreprocessWorker::decode_full(
    const uint8_t* jpeg_data,
    size_t jpeg_size,
    int32_t& width,
    int32_t& height,
    size_t& stride
) {
    // 使用TurboJPEG 3.x API读取header
    if (tj3DecompressHeader(tj_handle_, jpeg_data, jpeg_size) != 0) {
        LOG_DEBUG << "PW " << config_.worker_id << " tj3DecompressHeader failed, trying STB fallback";

        #if TR_USE_STB
        int stb_width, stb_height;
        if (decode_jpeg_with_stb(jpeg_data, jpeg_size, region_d_,
                                 config_.region_d_size, stb_width, stb_height)) {
            width = stb_width;
            height = stb_height;
            stride = ((width * config_.num_color_channels + 63) / 64) * 64;
            return true;
        }
        #endif
        return false;
    }

    width = tj3Get(tj_handle_, TJPARAM_JPEGWIDTH);
    height = tj3Get(tj_handle_, TJPARAM_JPEGHEIGHT);

    // 计算stride（64字节对齐）
    stride = ((static_cast<size_t>(width) * config_.num_color_channels + 63) / 64) * 64;

    // 验证D区容量
    size_t required_size = stride * height;
    if (required_size > config_.region_d_size) {
        LOG_DEBUG << "PW " << config_.worker_id << " image too large for D区, trying STB fallback: "
                  << "need " << (required_size / (1024.0*1024.0)) << " MB, "
                  << "have " << (config_.region_d_size / (1024.0*1024.0)) << " MB";

        #if TR_USE_STB
        int stb_width, stb_height;
        if (decode_jpeg_with_stb(jpeg_data, jpeg_size, region_d_,
                                 config_.region_d_size, stb_width, stb_height)) {
            width = stb_width;
            height = stb_height;
            stride = ((width * config_.num_color_channels + 63) / 64) * 64;
            return true;
        }
        #endif
        return false;
    }

    // 使用TurboJPEG 3.x API完整解码
    if (tj3Decompress8(tj_handle_, jpeg_data, jpeg_size,
                       region_d_, static_cast<int>(stride), TJPF_RGB) != 0) {
        LOG_DEBUG << "PW " << config_.worker_id << " tj3Decompress8 failed, trying STB fallback";

        #if TR_USE_STB
        int stb_width, stb_height;
        if (decode_jpeg_with_stb(jpeg_data, jpeg_size, region_d_,
                                 config_.region_d_size, stb_width, stb_height)) {
            width = stb_width;
            height = stb_height;
            stride = ((width * config_.num_color_channels + 63) / 64) * 64;
            return true;
        }
        #endif
        return false;
    }

    return true;
}

bool PreprocessWorker::decode_partial(
    const uint8_t* jpeg_data,
    size_t jpeg_size,
    const DecodeStrategy& strategy,
    int32_t& decoded_width,
    int32_t& decoded_height,
    size_t& stride
) {
    // 步骤1：先读取JPEG头获取原始图像尺寸
    if (tj3DecompressHeader(tj_handle_, jpeg_data, jpeg_size) != 0) {
        LOG_DEBUG << "PW " << config_.worker_id << " tj3DecompressHeader failed in partial decode: "
                  << tj3GetErrorStr(tj_handle_);
        return false;
    }

    int original_width = tj3Get(tj_handle_, TJPARAM_JPEGWIDTH);
    int original_height = tj3Get(tj_handle_, TJPARAM_JPEGHEIGHT);

    LOG_DEBUG << "PW " << config_.worker_id
              << " partial decode: original=" << original_width << "x" << original_height
              << ", R2=(" << strategy.decode_x << "," << strategy.decode_y
              << "," << strategy.decode_w << "x" << strategy.decode_h << ")";

    // 步骤2：设置TurboJPEG裁剪区域（R2，MCU对齐）
    tjregion crop_region;
    crop_region.x = strategy.decode_x;
    crop_region.y = strategy.decode_y;
    crop_region.w = strategy.decode_w;
    crop_region.h = strategy.decode_h;

    if (tj3SetCroppingRegion(tj_handle_, crop_region) != 0) {
        LOG_DEBUG << "PW " << config_.worker_id << " tj3SetCroppingRegion failed: "
                  << tj3GetErrorStr(tj_handle_);
        return false;
    }

    // 步骤3：解码R2区域到D区（不做memmove，直接保留R2布局）
    decoded_width = strategy.decode_w;
    decoded_height = strategy.decode_h;
    stride = ((decoded_width * config_.num_color_channels + 63) / 64) * 64;

    size_t required_size = stride * decoded_height;
    if (required_size > config_.region_d_size) {
        LOG_DEBUG << "PW " << config_.worker_id << " R2 decode region too large, returning false for STB fallback";
        return false;
    }

    // 步骤4：解码MCU对齐的R2区域
    if (tj3Decompress8(tj_handle_, jpeg_data, jpeg_size,
                       region_d_, static_cast<int>(stride), TJPF_RGB) != 0) {
        LOG_DEBUG << "PW " << config_.worker_id << " tj3Decompress8 (partial) failed: "
                  << tj3GetErrorStr(tj_handle_);
        return false;
    }

    LOG_DEBUG << "PW " << config_.worker_id
              << " partial decode success: R2=" << decoded_width << "x" << decoded_height
              << " at D区, stride=" << stride;

    return true;
}

#if TR_USE_STB
bool PreprocessWorker::decode_with_stb(
    const uint8_t* jpeg_data,
    size_t jpeg_size,
    int32_t& width,
    int32_t& height,
    size_t& stride
) {
    int stb_width, stb_height, stb_channels;
    stbi_uc* stb_data = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(jpeg_data),
        static_cast<int>(jpeg_size),
        &stb_width, &stb_height, &stb_channels,
        3  // 强制3通道RGB
    );

    if (!stb_data) {
        LOG_ERROR << "PW " << config_.worker_id << " STB decode failed";
        return false;
    }

    width = stb_width;
    height = stb_height;
    stride = ((width * config_.num_color_channels + 63) / 64) * 64;

    // 验证D区容量
    size_t required_size = stride * height;
    if (required_size > config_.region_d_size) {
        LOG_ERROR << "PW " << config_.worker_id << " STB image too large for D区";
        stbi_image_free(stb_data);
        return false;
    }

    // STB返回的数据是紧密打包的，需要复制到D区（考虑stride）
    for (int y = 0; y < height; ++y) {
        const stbi_uc* src_row = stb_data + y * (width * 3);
        uint8_t* dst_row = region_d_ + y * stride;
        std::memcpy(dst_row, src_row, width * 3);
    }

    stbi_image_free(stb_data);
    return true;
}
#endif

// ============================================================================
// work() 核心方法（测试模式）
// ============================================================================

bool PreprocessWorker::work(
    int32_t label,
    const uint8_t* data_ptr,
    size_t data_size
) {
    // 确保RNG已初始化
    ensure_rng_initialized();

    const bool is_train = param_.is_train;
    const int res = is_train ? param_.current_train_resolution
                             : param_.current_val_resolution;
    const size_t sample_stride = ((res * config_.num_color_channels + 63) / 64) * 64;

    // ==================== 测试模式路径 ====================
    if (config_.test_mode) {
        // 测试模式：只执行一个PO，输出到A区
        const auto& ops = is_train ? train_ops_ : val_ops_;

        if (ops.empty()) {
            // 无操作：跳过
            total_samples_processed_++;
            return true;
        }

        // ==================== 获取图像尺寸 ====================
        int32_t image_width = config_.raw_image_width;
        int32_t image_height = config_.raw_image_height;

        if (config_.dataset_type == DatasetType::imagenet) {
            // ImageNet：必须先读JPEG头（使用TurboJPEG 3.x API）
            if (tj3DecompressHeader(tj_handle_, data_ptr, data_size) != 0) {
                // TurboJPEG读取头失败，尝试STB fallback
                LOG_DEBUG << "PW " << config_.worker_id << " failed to read JPEG header, trying STB fallback";

                #if TR_USE_STB
                int stb_w, stb_h;
                size_t dummy_stride;
                if (decode_jpeg_with_stb(data_ptr, data_size, region_d_,
                                         config_.region_d_size, stb_w, stb_h)) {
                    // STB读取成功
                    image_width = stb_w;
                    image_height = stb_h;
                } else {
                    // STB也失败
                    LOG_ERROR << "PW " << config_.worker_id << " both TurboJPEG and STB failed to read JPEG header";
                    total_samples_processed_++;
                    return true;  // 跳过损坏样本
                }
                #else
                LOG_ERROR << "PW " << config_.worker_id << " failed to read JPEG header and STB not available";
                total_samples_processed_++;
                return true;  // 跳过损坏样本
                #endif
            } else {
                // TurboJPEG 3.x读取头成功
                image_width = tj3Get(tj_handle_, TJPARAM_JPEGWIDTH);
                image_height = tj3Get(tj_handle_, TJPARAM_JPEGHEIGHT);
            }
        }

        // ==================== 获取解码策略 ====================
        DecodeStrategy strategy;
        const uint8_t* initial_ptr = nullptr;
        int32_t initial_w = 0, initial_h = 0;
        size_t initial_stride = 0;
        bool execute_from_full = false;  // 是否从完整解码执行

        if (!ops.empty()) {
            strategy = ops[0]->get_decode_strategy(
                image_width, image_height,
                config_.sdmp_factor, &rng_
            );

            // 检查是否需要存储DecodeStrategy（CenterCrop等需要）
            if (strategy.need_decode && strategy.use_partial) {
                // 尝试局部解码R2区域到D区
                int32_t decoded_w, decoded_h;
                if (decode_partial(data_ptr, data_size, strategy, decoded_w, decoded_h, initial_stride)) {
                    // 局部解码成功：D区包含R2解码结果
                    // PO会根据execute_from_full=false和strategy，自己计算从R2中提取R1的偏移
                    initial_ptr = region_d_;
                    initial_w = decoded_w;     // R2宽度（strategy.decode_w）
                    initial_h = decoded_h;     // R2高度（strategy.decode_h）
                    execute_from_full = false;
                } else {
                    // 局部解码失败，尝试STB完整解码
                    LOG_DEBUG << "PW " << config_.worker_id << " partial decode failed, trying STB fallback";
                    #if TR_USE_STB
                    int stb_w, stb_h;
                    if (decode_jpeg_with_stb(data_ptr, data_size, region_d_,
                                             config_.region_d_size, stb_w, stb_h)) {
                        // STB完整解码成功
                        initial_ptr = region_d_;
                        initial_w = stb_w;
                        initial_h = stb_h;
                        initial_stride = initial_w * config_.num_color_channels;
                        execute_from_full = true;  // 关键：通知PO使用完整解码算法
                    } else {
                        // 两者都失败
                        LOG_ERROR << "PW " << config_.worker_id << " both TurboJPEG and STB failed";
                        total_samples_processed_++;
                        return true;  // 跳过样本
                    }
                    #else
                    LOG_ERROR << "PW " << config_.worker_id << " partial decode failed and STB not available";
                    total_samples_processed_++;
                    return true;
                    #endif
                }
            } else if (strategy.need_decode && !strategy.use_partial) {
                // 完整解码
                if (!decode_full(data_ptr, data_size, initial_w, initial_h, initial_stride)) {
                    // 完整解码失败，尝试STB fallback
                    LOG_DEBUG << "PW " << config_.worker_id << " full decode failed, trying STB fallback";

                    #if TR_USE_STB
                    int stb_w, stb_h;
                    if (decode_jpeg_with_stb(data_ptr, data_size, region_d_,
                                             config_.region_d_size, stb_w, stb_h)) {
                        // STB完整解码成功
                        initial_ptr = region_d_;
                        initial_w = stb_w;
                        initial_h = stb_h;
                        initial_stride = initial_w * config_.num_color_channels;
                        execute_from_full = true;  // 关键：通知PO使用完整解码算法
                    } else {
                        // 两者都失败
                        LOG_ERROR << "PW " << config_.worker_id << " both TurboJPEG and STB failed";
                        total_samples_processed_++;
                        return true;  // 跳过样本
                    }
                    #else
                    LOG_ERROR << "PW " << config_.worker_id << " full decode failed and STB not available";
                    total_samples_processed_++;
                    return true;
                    #endif
                } else {
                    initial_ptr = region_d_;
                }
            } else {
                // 不需要解码（非ImageNet）：直接使用输入
                initial_ptr = data_ptr;
                initial_w = image_width;
                initial_h = image_height;
                initial_stride = image_width * config_.num_color_channels;
            }
        }

        // ==================== 执行第一个PO（测试模式只执行一个）====================
        int32_t out_w, out_h;
        ops[0]->execute(
            initial_ptr, initial_w, initial_h, initial_stride,
            region_a_,  // 测试模式：固定输出到A区
            out_w, out_h,
            sample_stride,
            &rng_,
            execute_from_full  // 关键：告诉PO是从完整解码还是局部解码执行
        );

        total_samples_processed_++;
        return true;
    }

    // ==================== 一般模式（暂不实现）====================
    LOG_ERROR << "PW " << config_.worker_id << " general mode not implemented yet";
    return false;
}

} // namespace tr
