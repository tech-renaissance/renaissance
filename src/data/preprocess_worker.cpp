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
#include "renaissance/data/engine_buffer.h"
#include "renaissance/data/do_nothing.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/global_registry.h"
#include "renaissance/base/philox.h"
#include "renaissance/base/rng.h"
#include <chrono>
#include <algorithm>  // for std::swap

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
    const std::vector<std::unique_ptr<PreprocessOperation>>& val_ops,
    PreprocessWorkerParameter* pwp_ptr)
    : config_(config)
    , engine_buffer_(config.engine_buffer)  // 保存EngineBuffer指针
    , rng_(0)  // 临时seed，延迟初始化
{
    // =========================================================================
    // 关键步骤：保存PW专用的初始种子（用于洗牌的可复现性）
    // =========================================================================
    //
    // 种子衍生的两层设计：
    //
    // 第一层（构造时）：从全局种子衍生PW专用种子
    //   initial_seed_ = global_seed ^ (worker_id << 32)
    //   - 每个PW有自己独特的初始种子
    //   - 从全局种子衍生，可复现
    //   - 不受其他过程的随机数生成影响
    //
    // 第二层（shuffle时）：从initial_seed_按epoch衍生洗牌种子
    //   shuffle_seed = initial_seed_ ^ (phase_id << 32)
    //   - 每个phase的洗牌结果不同
    //   - 每个PW的洗牌结果不同
    //   - 完全可复现
    //
    // 为什么这样设计？
    // - 确保每个phase的洗牌结果不同（phase_id参与）
    // - 确保不同PW的洗牌结果不同（worker_id通过initial_seed_参与）
    // - 确保完全可复现（相同的全局种子 → 相同的initial_seed_序列）
    // - 避免被其他过程的随机数生成干扰（构造时就锁定种子）
    //
    // 参考ImageNetLoaderDts：它在每次shuffle时调用get_default_generator().seed()
    // 我们的设计更严格：在构造时锁定PW专用种子，完全隔离外界干扰

    preprocessor_param_ptr_ = pwp_ptr;
    param_ = *pwp_ptr;
    uint64_t global_seed = get_default_generator().seed();
    initial_seed_ = global_seed ^ (static_cast<uint64_t>(config_.worker_id) << 32);

    // ==================== 从GlobalRegistry复制S区/C区单个样本对齐后大小 ====================
    s_c_region_stride_ = GlobalRegistry::instance().aligned_max_output_size();

    // ==================== 从GlobalRegistry复制数据集样本总数 ====================
    dataset_total_train_samples_ = GlobalRegistry::instance().num_train_samples();
    dataset_total_val_samples_ = GlobalRegistry::instance().num_val_samples();

    // ==================== 从GlobalRegistry复制Deployment模式标志 ====================
    is_deployment_mode_ = GlobalRegistry::instance().is_deployment_mode();

    // ==================== 从GlobalRegistry复制RHF标志 ====================
    train_with_rhf_ = GlobalRegistry::instance().train_with_rhf();
    val_with_rhf_ = GlobalRegistry::instance().val_with_rhf();

    LOG_DEBUG << "[PW " << config_.worker_id << " CONSTRUCTOR] "
              << "test_mode=" << (config_.test_mode ? "ON" : "OFF")
              << ", engine=" << config_.engine_id
              << ", pid=" << config_.pid_in_engine
              << ", engine_buffer=" << (engine_buffer_ ? "SET" : "NULL")
              << ", global_seed=" << global_seed
              << ", initial_seed=" << initial_seed_
              << ", s_c_region_stride_=" << s_c_region_stride_ << " bytes"
              << ", train_samples=" << dataset_total_train_samples_
              << ", val_samples=" << dataset_total_val_samples_
              << ", deployment_mode=" << (is_deployment_mode_ ? "ON" : "OFF")
              << ", train_with_rhf=" << (train_with_rhf_ ? "YES" : "NO")
              << ", val_with_rhf=" << (val_with_rhf_ ? "YES" : "NO");

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
	num_train_ops_ = static_cast<int>(train_ops_.size());
	num_val_ops_ = static_cast<int>(val_ops_.size());

    LOG_DEBUG << "PW " << config_.worker_id << " cloned POs: "
              << num_train_ops_ << " train, "
              << num_val_ops_ << " val";

    // ==================== 3. 创建内置DoNothing操作 ====================
    built_in_do_nothing_ = new DoNothing();
    LOG_DEBUG << "PW " << config_.worker_id << " built-in DoNothing created";

    // ==================== 4. 初始化TurboJPEG 3.x ====================
    tj_handle_ = tj3Init(TJINIT_DECOMPRESS);
    TR_CHECK(tj_handle_ != nullptr, MemoryError,
             "Failed to initialize TurboJPEG for PW " << config_.worker_id);

    // 设置优化标志
    tj3Set(tj_handle_, TJPARAM_FASTDCT, 1);
    tj3Set(tj_handle_, TJPARAM_FASTUPSAMPLE, 1);

    LOG_DEBUG << "PW " << config_.worker_id << " TurboJPEG 3.x initialized";

    // 打印容量配置
    LOG_DEBUG << "PW " << config_.worker_id << " capacity: max_s_samples=" << config_.max_s_samples
              << ", max_c_samples=" << config_.max_c_samples;

    // ==================== 4. 初始化标签向量 ====================
    if (config_.max_c_samples > 0) {
        c_label_vector_.resize(config_.max_c_samples);
        LOG_DEBUG << "PW " << config_.worker_id << " C区标签向量初始化: size=" << config_.max_c_samples;
    }

    if (config_.num_region_s > 0 && config_.max_s_samples > 0) {
        s_label_vectors_.resize(config_.num_region_s);
        for (int i = 0; i < config_.num_region_s; ++i) {
            s_label_vectors_[i].resize(config_.max_s_samples);
        }
        LOG_DEBUG << "PW " << config_.worker_id << " S区标签向量初始化: num=" << config_.num_region_s
                  << ", each_size=" << config_.max_s_samples;
    }

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

    // 释放内置DoNothing操作
    if (built_in_do_nothing_) {
        delete built_in_do_nothing_;
        built_in_do_nothing_ = nullptr;
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
    region_s_ptrs_.resize(config_.num_region_s);
    for (int i = 0; i < config_.num_region_s; ++i) {
        region_s_ptrs_[i] = ptr;
        LOG_DEBUG << "PW " << config_.worker_id << " S" << (i+1) << "区: "
                  << static_cast<void*>(ptr)
                  << ", size=" << (config_.region_s_size / (1024.0*1024.0)) << " MB";
        ptr += align_4k(config_.region_s_size);
    }

    if (config_.region_c_size > 0) {
        region_c_ptr_ = ptr;
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

void PreprocessWorker::update_parameters() {
    param_ = *preprocessor_param_ptr_;

    // 重置样本计数器（每个phase开始时）
    local_sample_id_ = 0;      // 重置已处理样本计数
    if (!param_.is_lazy_phase) {
		if (param_.is_train) {
			current_s_samples_ = 0;  // Busy train phase重置S区写入计数
		}
		else {
			current_c_samples_ = 0;  // Busy val phase重置C区写入计数
		}
    }

    // =========================================================================
    // 重新初始化TurboJPEG handle（修复Train/Val切换时的状态问题）
    // =========================================================================
    // 问题：sdmp=1时，Train Phase结束后直接进入Val Phase，TurboJPEG handle内部状态可能被破坏
    // 解决方案：每次参数更新时销毁并重新创建handle
    if (tj_handle_) {
        tj3Destroy(tj_handle_);
        tj_handle_ = nullptr;
    }
    tj_handle_ = tj3Init(TJINIT_DECOMPRESS);
    TR_CHECK(tj_handle_ != nullptr, MemoryError,
             "Failed to reinitialize TurboJPEG for PW " << config_.worker_id);
    tj3Set(tj_handle_, TJPARAM_FASTDCT, 1);

    LOG_DEBUG << "PW " << config_.worker_id << " TurboJPEG handle reinitialized";

    LOG_DEBUG << "PW " << config_.worker_id << " parameters updated: "
              << (param_.is_train ? "TRAIN" : "VAL");
}

// -----------------------------------------------------------------------------

void PreprocessWorker::set_deployment_mode_input_property(
    int width,
    int height,
    int num_channels
) {
    // ==================== Deployment模式检查 ====================
    TR_CHECK(is_deployment_mode_, ValueError,
             "set_deployment_mode_input_property() can only be called in deployment mode. "
             "is_deployment_mode_=" << is_deployment_mode_);

    // ==================== 参数验证 ====================
    TR_CHECK(width > 0 && height > 0, ValueError,
             "Invalid image dimensions: " << width << "x" << height);
    TR_CHECK(num_channels == 1 || num_channels == 3, ValueError,
             "Invalid num_channels: " << num_channels << " (expected 1 or 3)");

    // ==================== 更新config_中的图像属性 ====================
    config_.raw_image_width = width;
    config_.raw_image_height = height;
    config_.num_color_channels = num_channels;

    // ==================== Debug日志 ====================
    LOG_DEBUG << "PW " << config_.worker_id
              << " deployment_mode_input_property updated: "
              << width << "x" << height
              << ", channels=" << num_channels;
}

// ============================================================================
// S区槽位请求
// ============================================================================

uint8_t* PreprocessWorker::request_s_region_slot(int32_t label, int s_region_idx) {
    // ==================== 参数验证 ====================
    TR_CHECK(s_region_idx >= 0 && s_region_idx < config_.num_region_s, ValueError,
             "Invalid s_region_idx: " << s_region_idx
             << ", expected [0, " << (config_.num_region_s - 1) << "]");

    TR_CHECK(local_sample_id_ < config_.max_s_samples, ValueError,
             "S区已满: local_sample_id_=" << local_sample_id_
             << ", max_s_samples=" << config_.max_s_samples);

    TR_CHECK(s_c_region_stride_ > 0, ValueError,
             "s_c_region_stride_ not initialized: " << s_c_region_stride_);

    // ==================== 计算槽位索引 ====================
    // 所有S区使用相同的slot_index（与local_sample_id_对应）
    int slot_index = local_sample_id_;

    // ==================== 计算目标指针 ====================
    // 使用从GlobalRegistry复制的s_c_region_stride_（64字节对齐的单个样本大小）
    uint8_t* target_ptr = region_s_ptrs_[s_region_idx] + slot_index * s_c_region_stride_;

    // ==================== 保存标签到对应S区的标签向量 ====================
    TR_CHECK(s_region_idx >= 0 && s_region_idx < static_cast<int>(s_label_vectors_.size()),
             ValueError, "s_region_idx out of range: " << s_region_idx);
    TR_CHECK(slot_index < static_cast<int>(s_label_vectors_[s_region_idx].size()),
             ValueError, "slot_index out of range: " << slot_index);

    s_label_vectors_[s_region_idx][slot_index] = label;

    // ==================== Debug日志 ====================
    LOG_DEBUG << "PW " << config_.worker_id
              << " request_s_region_slot: s_region_idx=" << s_region_idx
              << ", slot_index=" << slot_index
              << ", label=" << label
              << ", s_c_region_stride_=" << s_c_region_stride_
              << ", target_ptr=" << static_cast<void*>(target_ptr);

    return target_ptr;
}

uint8_t* PreprocessWorker::request_c_region_slot(int32_t label) {
    // ==================== 参数验证 ====================
    TR_CHECK(local_sample_id_ < config_.max_c_samples, ValueError,
             "C区已满: local_sample_id_=" << local_sample_id_
             << ", max_c_samples=" << config_.max_c_samples);

    TR_CHECK(s_c_region_stride_ > 0, ValueError,
             "s_c_region_stride_ not initialized: " << s_c_region_stride_);

    TR_CHECK(region_c_ptr_ != nullptr, ValueError,
             "C区指针未初始化（CPVS未启用）");

    // ==================== 计算槽位索引 ====================
    int slot_index = local_sample_id_;

    // ==================== 计算目标指针 ====================
    // 使用从GlobalRegistry复制的s_c_region_stride_（64字节对齐的单个样本大小）
    uint8_t* target_ptr = region_c_ptr_ + slot_index * s_c_region_stride_;

    // ==================== 保存标签到C区标签向量 ====================
    TR_CHECK(slot_index < static_cast<int>(c_label_vector_.size()),
             ValueError, "slot_index out of range: " << slot_index);

    c_label_vector_[slot_index] = label;

    // ==================== Debug日志 ====================
    LOG_DEBUG << "PW " << config_.worker_id
              << " request_c_region_slot: slot_index=" << slot_index
              << ", label=" << label
              << ", s_c_region_stride_=" << s_c_region_stride_
              << ", target_ptr=" << static_cast<void*>(target_ptr);

    return target_ptr;
}

// ============================================================================
// EngineBuffer交互方法
// ============================================================================

uint8_t* PreprocessWorker::request_engine_buffer_slot(int32_t label) {
    // ==================== 参数验证 ====================
    TR_CHECK(engine_buffer_ != nullptr, ValueError,
             "engine_buffer_ is null (test mode or not configured)");

    // ==================== 计算写入位置 ====================
    auto [batch_id, position] = calculate_write_position();

    // ==================== 请求EngineBuffer写入槽位 ====================
    // 零拷贝设计：直接返回EngineBuffer内部内存指针
    // 批次边界保护：自动防止快Worker覆盖慢Worker数据
    uint8_t* write_ptr = engine_buffer_->request_write_slot(position, batch_id, label);

    // ==================== Debug日志 ====================
    LOG_DEBUG << "PW " << config_.worker_id
              << " request_engine_buffer_slot: label=" << label
              << ", batch_id=" << batch_id
              << ", position=" << position
              << ", write_ptr=" << static_cast<void*>(write_ptr);

    return write_ptr;
}

bool PreprocessWorker::notify_engine_buffer_sample_written() {
    // ==================== 参数验证 ====================
    TR_CHECK(engine_buffer_ != nullptr, ValueError,
             "engine_buffer_ is null (test mode or not configured)");

    // ==================== 调用EngineBuffer的notify方法 ====================
    // EngineBuffer会判断是否触发传输（batch满或其他条件）
    bool triggered = engine_buffer_->notify_sample_written();

    // ==================== Debug日志 ====================
    LOG_DEBUG << "PW " << config_.worker_id
              << " notify_engine_buffer_sample_written: local_sample_id_=" << local_sample_id_
              << ", triggered=" << triggered;

    return triggered;
}

void PreprocessWorker::no_more_samples() {
    // ==================== 参数验证 ====================
    TR_CHECK(engine_buffer_ != nullptr, ValueError,
             "engine_buffer_ is null (test mode or not configured)");

    // ==================== 转发到EngineBuffer ====================
    // 使用pid_in_engine作为worker_id（EngineBuffer视角）
    engine_buffer_->no_more_samples(config_.pid_in_engine);

    // ==================== Debug日志 ====================
    LOG_DEBUG << "PW " << config_.worker_id
              << " no_more_samples: pid_in_engine=" << config_.pid_in_engine;
}

// ============================================================================
// 从S区/C区复制样本到EngineBuffer
// ============================================================================

void PreprocessWorker::copy_sample_from_s_to_eb(int s_region_idx) {
    // ==================== 参数验证 ====================
    TR_CHECK(s_region_idx >= 0 && s_region_idx < config_.num_region_s, ValueError,
             "Invalid s_region_idx: " << s_region_idx);
    TR_CHECK(engine_buffer_ != nullptr, ValueError,
             "engine_buffer_ is null (test mode or not configured)");

    // ==================== 步骤1：从s_shuffled_indices_查询槽位编号 ====================
    TR_CHECK(local_sample_id_ < static_cast<int>(s_shuffled_indices_.size()), ValueError,
             "local_sample_id_ out of range: " << local_sample_id_);
    int slot_index = s_shuffled_indices_[local_sample_id_];

    // ==================== 步骤2：从s_label_vectors_获取标签 ====================
    TR_CHECK(s_region_idx < static_cast<int>(s_label_vectors_.size()), ValueError,
             "s_region_idx out of range: " << s_region_idx);
    TR_CHECK(slot_index < static_cast<int>(s_label_vectors_[s_region_idx].size()), ValueError,
             "slot_index out of range: " << slot_index);
    int32_t label = s_label_vectors_[s_region_idx][slot_index];

    // ==================== 步骤3：计算EngineBuffer写入位置 ====================
    auto [batch_id, position] = calculate_write_position();

    // ==================== 步骤4：向EngineBuffer申请写入slot ====================
    uint8_t* eb_ptr = engine_buffer_->request_write_slot(position, batch_id, label);
    TR_CHECK(eb_ptr != nullptr, ValueError, "EngineBuffer returned nullptr");

    // ==================== 步骤5：从S区复制图像数据 ====================
    // 计算当前resolution需要的字节数
    int current_resolution = param_.is_train ? param_.current_train_resolution
                                              : param_.current_val_resolution;
    size_t num_sample_bytes = current_resolution * current_resolution * config_.num_color_channels;

    // 计算S区源地址
    uint8_t* s_ptr = region_s_ptrs_[s_region_idx] + slot_index * s_c_region_stride_;

    // 复制数据（只复制当前resolution需要的字节数）
    std::memcpy(eb_ptr, s_ptr, num_sample_bytes);

    LOG_DEBUG << "PW " << config_.worker_id
              << " copy_sample_from_s_to_eb: s_region_idx=" << s_region_idx
              << ", slot_index=" << slot_index
              << ", label=" << label
              << ", bytes=" << num_sample_bytes;

    // ==================== 步骤6：通知EngineBuffer样本写入完成 ====================
    notify_engine_buffer_sample_written();
}

void PreprocessWorker::copy_sample_from_c_to_eb() {
    // ==================== 参数验证 ====================
    TR_CHECK(engine_buffer_ != nullptr, ValueError,
             "engine_buffer_ is null (test mode or not configured)");

    // ==================== 步骤1：local_sample_id_直接作为槽位编号 ====================
    int slot_index = local_sample_id_;

    // ==================== 步骤2：从c_label_vector_获取标签 ====================
    TR_CHECK(slot_index < static_cast<int>(c_label_vector_.size()), ValueError,
             "slot_index out of range: " << slot_index);
    int32_t label = c_label_vector_[slot_index];

    // ==================== 步骤3：计算EngineBuffer写入位置 ====================
    auto [batch_id, position] = calculate_write_position();

    // ==================== 步骤4：向EngineBuffer申请写入slot ====================
    uint8_t* eb_ptr = engine_buffer_->request_write_slot(position, batch_id, label);
    TR_CHECK(eb_ptr != nullptr, ValueError, "EngineBuffer returned nullptr");

    // ==================== 步骤5：从C区复制图像数据 ====================
    // 计算当前resolution需要的字节数
    int current_resolution = param_.is_train ? param_.current_train_resolution
                                              : param_.current_val_resolution;
    size_t num_sample_bytes = current_resolution * current_resolution * config_.num_color_channels;

    // 计算C区源地址
    uint8_t* c_ptr = region_c_ptr_ + slot_index * s_c_region_stride_;

    // 复制数据（只复制当前resolution需要的字节数）
    std::memcpy(eb_ptr, c_ptr, num_sample_bytes);

    LOG_DEBUG << "PW " << config_.worker_id
              << " copy_sample_from_c_to_eb: slot_index=" << slot_index
              << ", label=" << label
              << ", bytes=" << num_sample_bytes;

    // ==================== 步骤6：通知EngineBuffer样本写入完成 ====================
    notify_engine_buffer_sample_written();
}

// ============================================================================
// S区洗牌索引
// ============================================================================

void PreprocessWorker::shuffle_s_indices(int phase_id) {
    // =========================================================================
    // 步骤1：从GlobalRegistry复制原始索引向量
    // =========================================================================

    // 直接向量赋值（O(1)操作，复制整个向量）
    // fixed_s_original_indices_内容为[0, 1, 2, ..., max_s_samples_-1]
    s_shuffled_indices_ = GlobalRegistry::instance().fixed_s_original_indices();

    LOG_DEBUG << "PW " << config_.worker_id
              << " copied fixed_s_original_indices_ to s_shuffled_indices_: size="
              << s_shuffled_indices_.size();

    // =========================================================================
    // 步骤2：生成洗牌种子（两层种子衍生，保证可复现性和差异性）
    // =========================================================================

    // 两层种子衍生设计：
    //
    // 第一层（构造时）：从全局种子衍生PW专用种子
    //   initial_seed_ = global_seed ^ (worker_id << 32)
    //   - 每个PW有自己独特的初始种子
    //   - 在构造函数中保存，固定不变
    //   - 完全不受训练过程中随机数生成的影响
    //
    // 第二层（shuffle时）：从initial_seed_按epoch衍生洗牌种子
    //   shuffle_seed = initial_seed_ ^ (phase_id << 32)
    //   - 每个epoch的洗牌结果不同（phase_id参与）
    //   - 每个PW的洗牌结果不同（worker_id通过initial_seed_参与）
    //   - 完全可复现
    //
    // 为什么这样设计？
    // 1. 确保每个epoch的洗牌结果不同
    //    - phase_id参与shuffle_seed计算
    //    - epoch 0, 1, 2, ... 使用不同的洗牌序列
    //
    // 2. 确保不同PW的洗牌结果不同
    //    - worker_id通过initial_seed_参与shuffle_seed计算
    //    - PW 0, 1, 2, ... 使用不同的洗牌序列
    //
    // 3. 确保完全可复现
    //    - 相同的全局种子 → 相同的initial_seed_序列
    //    - 相同的phase_id → 相同的shuffle_seed
    //    - 完全确定的洗牌结果
    //
    // 4. 避免被其他过程的随机数生成干扰
    //    - initial_seed_在构造时保存，固定不变
    //    - 不依赖于get_default_generator().seed()的当前值
    //    - 即使训练过程中修改了全局seed，也不影响洗牌
    //
    // 示例（假设global_seed=42）：
    //   PW 0: initial_seed_ = 42 ^ (0 << 32) = 42
    //   PW 1: initial_seed_ = 42 ^ (1 << 32) = 42 ^ 0x100000000
    //   PW 2: initial_seed_ = 42 ^ (2 << 32) = 42 ^ 0x200000000
    //
    //   Epoch 0:
    //     PW 0: shuffle_seed = 42 ^ (0 << 32) = 42
    //     PW 1: shuffle_seed = (42 ^ 0x100000000) ^ (0 << 32)
    //     PW 2: shuffle_seed = (42 ^ 0x200000000) ^ (0 << 32)
    //
    //   Epoch 1:
    //     PW 0: shuffle_seed = 42 ^ (1 << 32) = 42 ^ 0x100000000
    //     PW 1: shuffle_seed = (42 ^ 0x100000000) ^ (1 << 32)
    //     PW 2: shuffle_seed = (42 ^ 0x200000000) ^ (1 << 32)

    uint64_t shuffle_seed = initial_seed_ ^ (static_cast<uint64_t>(phase_id) << 32);

    LOG_DEBUG << "PW " << config_.worker_id
              << " shuffle seed: initial=" << initial_seed_
              << ", phase_id=" << phase_id
              << ", shuffle_seed=" << shuffle_seed;

    // =========================================================================
    // 步骤3：Fisher-Yates洗牌（只洗前current_s_samples_个元素）
    // =========================================================================

    // =========================================================================
    // 关键概念：洗牌范围 vs 实际可读取范围
    // =========================================================================
    //
    // 重要：洗牌的范围和实际可读取的范围是S区实际已存储的总样本数，
    //       而不是max_s_samples_（最大容量）！
    //
    // 示例：
    //   max_s_samples_ = 1000（S区最大容量）
    //   current_s_samples_ = 300（当前epoch实际写入的样本数）
    //
    //   洗牌范围：
    //     - 只对前300个元素（索引0~299）进行洗牌
    //     - 后700个元素（索引300~999）保持原始顺序
    //
    //   读取范围：
    //     - 读取时只使用前300个洗牌后的索引
    //     - 后700个索引不会被使用（因为没有对应的样本数据）
    //
    // 为什么这样设计？
    //   - max_s_samples_是静态计算的容量上限
    //   - current_s_samples_是动态写入的样本数
    //   - 洗牌只对已写入的样本有意义
    //   - 未使用的位置不应该参与洗牌，也不应该被读取
    //
    // 计算逻辑：
    //   current_s_samples_应该在busy epoch结束时更新
    //   或者直接使用local_sample_id_（因为local_sample_id_从0开始计数）

    int n = current_s_samples_;

    if (n <= 1) {
        // 样本数<=1，无需洗牌
        LOG_DEBUG << "PW " << config_.worker_id
                  << " skip shuffle: current_s_samples_=" << n << " (<=1)";
        return;
    }

    // =========================================================================
    // Fisher-Yates洗牌算法（直接使用Philox，保证可复现性）
    // =========================================================================
    //
    // 参考ImageNetLoaderDts::shuffle_full_dataset()的正确实现：
    // - 直接调用philox_generate_4x32()
    // - 第二个参数offset=i（循环变量），确保每次迭代的随机数独立
    // - 不修改任何Generator状态，避免干扰其他随机数生成
    // - 完全独立，完全可复现
    //
    // 为什么不使用Generator::random_int()？
    // - Generator::random_int()会调用next_offset(1)，修改Generator状态
    // - 如果同一个Generator被用于其他用途（比如PO操作），
    //   洗牌的offset就会受到之前调用的影响，破坏可复现性！
    // - 直接使用Philox算法，以shuffle_seed为种子，以i为offset，
    //   可以确保洗牌完全独立，不受其他因素影响

    for (int i = n - 1; i > 0; --i) {
        // 生成4个随机数（Philox算法）
        // - shuffle_seed：洗牌种子（固定）
        // - i：offset（随循环变化，确保每次迭代不同）
        uint32_t r[4];
        detail::philox_generate_4x32(shuffle_seed, static_cast<uint64_t>(i), r);

        // 取模获取随机索引 j ∈ [0, i]
        uint32_t j = r[0] % (i + 1);

        // 交换s_shuffled_indices_[i]和s_shuffled_indices_[j]
        std::swap(s_shuffled_indices_[i], s_shuffled_indices_[j]);
    }

    LOG_DEBUG << "PW " << config_.worker_id
              << " shuffled s_shuffled_indices_: range=[0," << (n-1) << "]";
}

// ============================================================================
// 写入位置计算（零竞争设计）
// ============================================================================

std::pair<int, int> PreprocessWorker::calculate_write_position() const {
    // 核心公式（参考test_engine_buffer_emulator.cpp第85-90行）
    // global_seq = pid_in_engine + local_sample_id_ * num_workers_per_engine
    // batch_id = global_seq / local_batch_size
    // position = global_seq % local_batch_size

    const int M = config_.num_workers_per_engine;      // M: 每个Engine的PW数量
    const int j = config_.pid_in_engine;               // j: 该PW在Engine内的编号
    const int B = config_.local_batch_size;            // B: 批次大小

    // n: 该PW当前phase已处理的样本数（从0开始，每次end_sample()后递增）
    // 注意：此方法应在end_sample()调用后使用
    const int n = local_sample_id_;

    // 计算全局样本序号（跨所有PW的统一序号）
    // PW0的第0次调用 → global_seq = 0
    // PW1的第0次调用 → global_seq = 1
    // PW0的第1次调用 → global_seq = M
    // PW1的第1次调用 → global_seq = M + 1
    int global_seq = n * M + j;

    // 计算批次ID和批次内的位置
    int batch_id = global_seq / B;
    int position = global_seq % B;

    // Debug模式边界检查
#ifndef NDEBUG
    TR_CHECK(position >= 0 && position < B, ValueError,
             "Position out of range: position=" << position
             << ", batch_size=" << B
             << ", worker_id=" << config_.worker_id
             << ", local_sample_id_=" << n);
    TR_CHECK(batch_id >= 0, ValueError,
             "Invalid batch_id: " << batch_id);
#endif

    LOG_DEBUG << "PW " << config_.worker_id
              << " calculate_write_position: j=" << j
              << ", n=" << n
              << ", M=" << M
              << ", global_seq=" << global_seq
              << ", batch_id=" << batch_id
              << ", position=" << position;

    return {batch_id, position};
}

// ============================================================================
// 样本处理完成
// ============================================================================

void PreprocessWorker::end_sample() {
    // 递增本地样本计数器
    local_sample_id_++;

    // Debug模式下检查是否溢出
#ifndef NDEBUG
    // 合理性检查：local_sample_id_不应该超过phase的样本总数
    // 这里只是警告，因为实际样本数可能动态确定
    if (local_sample_id_ > 10000000) {  // 1000万样本是合理上限
        LOG_WARN << "PW " << config_.worker_id
                 << " local_sample_id_ is unusually large: " << local_sample_id_;
    }
#endif
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
// work() 核心方法（一般模式 - Busy Phase使用）
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
	const int num_color_channels = config_.num_color_channels;
	const int phase_id = param_.phase_id;
    const bool is_lazy_phase = param_.is_lazy_phase;
	const int sdmp_factor = config_.sdmp_factor;
	const bool using_cpvs = config_.using_cpvs;
	const bool has_random_horizontal_flip = is_train? train_with_rhf_: val_with_rhf_;
	const int num_ops = is_train? num_train_ops_: num_val_ops_;

    // 检查是否已分配对应的EngineBuffer（一般模式必需）
    TR_CHECK(engine_buffer_ != nullptr, ValueError,
             "engine_buffer_ is null (test mode or not configured)");

	// 选择OP链
    const auto& ops = is_train ? train_ops_ : val_ops_;
    if (ops.empty()) {
        TR_VALUE_ERROR("Illegal increment: no operation to execute (ops is empty)");
        return false;
    }

    // ==================== 获取图像尺寸 ====================
    int32_t image_width = config_.raw_image_width;
    int32_t image_height = config_.raw_image_height;
    const uint8_t* initial_ptr = nullptr;
    int32_t initial_w = 0, initial_h = 0;
    size_t initial_stride = 0;
    bool execute_from_full = true;
    if ((config_.dataset_type != DatasetType::imagenet) || is_deployment_mode_) {
		// 对于非ImageNet的数据集，无需解码，直接取用原始数据
        // 对于部署模式，由于专用的SampleLoader会负责解码，所以也不需要解码，可直接取用原始数据
        initial_ptr = data_ptr;
        initial_w = image_width;
        initial_h = image_height;
        initial_stride = image_width * config_.num_color_channels;
		execute_from_full = true;
    }
	else {  // ImageNet数据集的情况
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
                TR_VALUE_ERROR("Illegal increment: both TurboJPEG and STB failed to read JPEG header");
                return true;  // 跳过损坏样本
            }
            #else
            LOG_ERROR << "PW " << config_.worker_id << " failed to read JPEG header and STB not available";
            TR_VALUE_ERROR("Illegal increment: failed to read JPEG header and STB not available");
            return true;  // 跳过损坏样本
            #endif
        } else {
            // TurboJPEG 3.x读取头成功
            image_width = tj3Get(tj_handle_, TJPARAM_JPEGWIDTH);
            image_height = tj3Get(tj_handle_, TJPARAM_JPEGHEIGHT);
        }

    	// ==================== 获取解码策略 ====================
    	DecodeStrategy strategy;
    	execute_from_full = false;  // 是否从完整解码执行

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
                initial_ptr = region_d_;
                initial_w = decoded_w;
                initial_h = decoded_h;
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
                    TR_VALUE_ERROR("Illegal increment: both TurboJPEG partial decode and STB fallback failed");
                    return true;  // 跳过样本
                }
                #else
                LOG_ERROR << "PW " << config_.worker_id << " partial decode failed and STB not available";
                TR_VALUE_ERROR("Illegal increment: partial decode failed and STB not available");
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
                    TR_VALUE_ERROR("Illegal increment: both TurboJPEG full decode and STB fallback failed");
                    return true;  // 跳过样本
                }
                #else
                LOG_ERROR << "PW " << config_.worker_id << " full decode failed and STB not available";
                TR_VALUE_ERROR("Illegal increment: full decode failed and STB not available");
                return true;
                #endif
            } else {
                // TurboJPEG完整解码成功
                initial_ptr = region_d_;
            }
        } else {
            initial_ptr = data_ptr;
            initial_w = image_width;
            initial_h = image_height;
            initial_stride = image_width * config_.num_color_channels;
        }
	}

    // ==================== 确定对一个样本总共需要进行的任务总次数 ====================
	int num_preprocess_tasks = 1;  // 正常情况下，只需进行一次预处理
	if (is_train) {
		num_preprocess_tasks = sdmp_factor;
	}
	else {
		num_preprocess_tasks = using_cpvs? 2: 1;
	}

	for (int task_id = 0; task_id < num_preprocess_tasks; ++task_id) {

		// 确定本次任务的最终输出位置
		uint8_t* final_output_ptr = nullptr;
		if (is_train) {
			if (task_id == num_preprocess_tasks - 1) {  // 最后一个任务，必定是输出到EngineBuffer
				final_output_ptr = request_engine_buffer_slot(label);
			}
			else {
				final_output_ptr = request_s_region_slot(label, task_id);  // task_id就是目标S区的ID
			}
		}
		else {

			if (task_id == num_preprocess_tasks - 1) {  // 最后一个任务，必定是输出到EngineBuffer
				final_output_ptr = request_engine_buffer_slot(label);

				if (task_id == 1) {
					// CPVS优化：从C区直接复制到EngineBuffer
					// C区和EngineBuffer都是紧凑布局，可以直接memcpy
					uint8_t* c_region_source_ptr = region_c_ptr_ + local_sample_id_ * s_c_region_stride_;
					std::memcpy(final_output_ptr, c_region_source_ptr, res * res * num_color_channels);
					notify_engine_buffer_sample_written();
					break;  // 提前结束，本轮无需再预处理！
				}
			}
			else {
				final_output_ptr = request_c_region_slot(label);
			}
		}

		// 重要！接下来针对RandomHorizontalFlip进行输出路径优化！有可能可以节省一次内存复制！
		bool last_op_is_flip = false;
		int total_ops_before_flip = 0;
		if (has_random_horizontal_flip) {
			// 判断本次操作到底是否需要水平翻转
			// 由于Preprocessor已排序，最后一个必定是RandomHorizontalFlip
			last_op_is_flip = ops.back()->should_flip(&rng_);
			total_ops_before_flip = num_ops - 1;
		}
		else {
			total_ops_before_flip = num_ops;
		}

		if (has_random_horizontal_flip && num_ops == 1) {  // 第一个操作就是水平翻转（也就是只有水平翻转一个操作）
			// 这种情况不会出现在ImageNet，因为Preprocessor已禁止
    		int32_t out_w, out_h;
    		size_t output_stride = 0;  // 触发自动计算
			if (last_op_is_flip) {
    			ops[0]->execute(  // 注意，根据我们的设计，RandomHorizontalFlip的excute()方法表示"必定执行翻转"
        			initial_ptr, initial_w, initial_h, initial_stride,
        			final_output_ptr,
        			out_w, out_h,
        			output_stride,  // 自动计算stride
        			&rng_,
        			true,
					true  // 最终输出，使用紧凑布局
    			);
			}
			else {  // 不翻转，那就直接从输入复制到输出
    			built_in_do_nothing_->execute(
        			initial_ptr, initial_w, initial_h, initial_stride,
        			final_output_ptr,
        			out_w, out_h,
        			output_stride,  // 自动计算紧凑布局
        			&rng_,
        			true,
					true  // 最终输出，使用紧凑布局
    			);
			}
			continue;  // 总共只有一个操作，因此可以提前宣布结束本轮次
		}

		if (last_op_is_flip) {  // 最后一个操作是水平翻转，而且已知前面有其他操作
    		size_t input_stride = 0;  // 占位符：op_id==0时使用initial_stride，op_id>=1时使用上一次的output_stride
    		size_t output_stride = 0;  // 会触发自动计算
			uint8_t* src_ptr = nullptr;
			uint8_t* dest_ptr = nullptr;
			int32_t input_w = initial_w;
			int32_t input_h = initial_h;
			int32_t output_w = 0;
			int32_t output_h = 0;
			for (int op_id = 0; op_id < total_ops_before_flip; ++op_id) {
				output_stride = 0;  // 会触发自动计算
				output_w = 0;
				output_h = 0;
				if (src_ptr != region_a_) {  // AB区乒乓。如果输入不在A区，就必定输出到A区，否则输出到B区
					dest_ptr = region_a_;
				}
				else {
					dest_ptr = region_b_;
				}
				if (op_id == 0) {
    				ops[op_id]->execute(
        				initial_ptr, initial_w, initial_h, initial_stride,
        				dest_ptr,
        				output_w, output_h,
        				output_stride,
        				&rng_,
        				execute_from_full,
						false  // 非最终输出，采用stride布局
    				);
				}
				else {
    				ops[op_id]->execute(
        				src_ptr, input_w, input_h, input_stride,
        				dest_ptr,
        				output_w, output_h,
        				output_stride,
        				&rng_,
        				true,
						false  // 非最终输出，采用stride布局
    				);
				}
				input_stride = output_stride;
				input_w = output_w;
				input_h = output_h;
				src_ptr = dest_ptr;
			}
    		size_t flip_output_stride = 0;  // 触发自动计算（紧凑布局）
    		ops[total_ops_before_flip]->execute(  // 执行翻转
        		src_ptr, input_w, input_h, input_stride,
        		final_output_ptr,
        		output_w, output_h,
        		flip_output_stride,  // 自动计算紧凑布局
        		&rng_,
        		true,
				true  // 最终输出，采用紧凑布局
    		);
		}
		else {  // 最后一个操作不是水平翻转，总共有1个或以上的操作
    		size_t input_stride = 0;  // 占位符：op_id==0时使用initial_stride，op_id>=1时使用上一次的output_stride
    		size_t output_stride = 0;  // 会触发自动计算
			uint8_t* src_ptr = nullptr;
			uint8_t* dest_ptr = nullptr;
			int32_t input_w = initial_w;
			int32_t input_h = initial_h;
			int32_t output_w = 0;
			int32_t output_h = 0;
			for (int op_id = 0; op_id < total_ops_before_flip; ++op_id) {
				output_w = 0;
				output_h = 0;
				bool compact_layout = false;
				if (op_id == total_ops_before_flip - 1) {  // 最后一个操作，必定输出到最终输出位置
					dest_ptr = final_output_ptr;
					output_stride = 0;  // 触发自动计算（紧凑布局）
					compact_layout = true;  // 最终输出，紧凑布局
				}
				else {
					output_stride = 0;  // 触发自动计算（紧凑布局，因为写入AB区）
					if (src_ptr != region_a_) {  // AB区乒乓。如果输入不在A区，就必定输出到A区，否则输出到B区
						dest_ptr = region_a_;
					}
					else {
						dest_ptr = region_b_;
					}
					compact_layout = false;  // 非最终输出，stride布局
				}
				if (op_id == 0) {
    				ops[op_id]->execute(
        				initial_ptr, initial_w, initial_h, initial_stride,
        				dest_ptr,
        				output_w, output_h,
        				output_stride,
        				&rng_,
        				execute_from_full,
						compact_layout
    				);
				}
				else {
    				ops[op_id]->execute(
        				src_ptr, input_w, input_h, input_stride,
        				dest_ptr,
        				output_w, output_h,
        				output_stride,
        				&rng_,
        				true,
						compact_layout
    				);
				}
				input_stride = output_stride;
				input_w = output_w;
				input_h = output_h;
				src_ptr = dest_ptr;
			}
		}
	}


	if (is_train) {
		// 递增S区的写入计数。注意：所有S区的写入样本数是一致的，写入多个S区，只递增1！
		if (sdmp_factor > 1) current_s_samples_++;  // 这个值在lazy phase下不改变，不清零
	}
	else {
		// 递增C区的写入计数
		if (using_cpvs) current_c_samples_++;  // 这个值在lazy phase下不改变，不清零
	}

	end_sample();  // 处理完一个样本就应该调用end_sample()，只能在work()和work_lazy()中调用，不应该在其他地方调用

	// PW0打印样本完成信息
	/*
	if (config_.worker_id == 0 && !is_train && param_.phase_id == 0) {
		std::cout << "[PW0] Sample " << (local_sample_id_ - 1)
		          << " completed, current_c_samples_=" << current_c_samples_
		          << std::endl;
	}
	*/

    return true;
}

// ============================================================================
// work_test_mode() - 测试模式专用方法（与work()的测试模式完全一致）
// ============================================================================

bool PreprocessWorker::work_test_mode(
    int32_t label,
    const uint8_t* data_ptr,
    size_t data_size
) {
    // 确保RNG已初始化
    ensure_rng_initialized();

    const bool is_train = param_.is_train;
    const int res = is_train ? param_.current_train_resolution
                             : param_.current_val_resolution;
    size_t final_sample_stride = ((res * config_.num_color_channels + 63) / 64) * 64;

    // ==================== 测试模式路径 ====================
    // 测试模式：只执行一个PO，输出到A区
    const auto& ops = is_train ? train_ops_ : val_ops_;

    if (ops.empty()) {
        TR_VALUE_ERROR("Illegal increment: no operation to execute (ops is empty)");
        return false;
    }

    // ==================== 获取图像尺寸 ====================
    int32_t image_width = config_.raw_image_width;
    int32_t image_height = config_.raw_image_height;
    const uint8_t* initial_ptr = nullptr;
    int32_t initial_w = 0, initial_h = 0;
    size_t initial_stride = 0;
    bool execute_from_full = true;

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
                TR_VALUE_ERROR("Illegal increment: both TurboJPEG and STB failed to read JPEG header");
                return true;  // 跳过损坏样本
            }
            #else
            LOG_ERROR << "PW " << config_.worker_id << " failed to read JPEG header and STB not available";
            TR_VALUE_ERROR("Illegal increment: failed to read JPEG header and STB not available");
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
    execute_from_full = false;  // 是否从完整解码执行

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
                initial_ptr = region_d_;
                initial_w = decoded_w;
                initial_h = decoded_h;
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
                    TR_VALUE_ERROR("Illegal increment: both TurboJPEG partial decode and STB fallback failed");
                    return true;  // 跳过样本
                }
                #else
                LOG_ERROR << "PW " << config_.worker_id << " partial decode failed and STB not available";
                TR_VALUE_ERROR("Illegal increment: partial decode failed and STB not available");
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
                    TR_VALUE_ERROR("Illegal increment: both TurboJPEG full decode and STB fallback failed");
                    return true;  // 跳过样本
                }
                #else
                LOG_ERROR << "PW " << config_.worker_id << " full decode failed and STB not available";
                TR_VALUE_ERROR("Illegal increment: full decode failed and STB not available");
                return true;
                #endif
            } else {
                // TurboJPEG完整解码成功
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
        final_sample_stride,
        &rng_,
        execute_from_full  // 关键：告诉PO是从完整解码还是局部解码执行
    );

    return true;
}

// ============================================================================
// work_lazy() - Lazy phase预处理（从S区/C区读取）
// ============================================================================

void PreprocessWorker::work_lazy() {
    // ==================== 前置检查 ====================
    TR_CHECK(config_.sdmp_factor > 1 || config_.using_cpvs, ValueError,
             "work_lazy() called but sdmp_factor=" << config_.sdmp_factor
             << ", using_cpvs=" << config_.using_cpvs);
    TR_CHECK(engine_buffer_ != nullptr, ValueError,
             "engine_buffer_ is null (lazy phase requires EngineBuffer)");

    LOG_INFO << "PW " << config_.worker_id << " work_lazy starting: "
             << (param_.is_train ? "TRAIN" : "VAL");

    // ==================== 步骤1：复位local_sample_id_ ====================
    local_sample_id_ = 0;

    // ==================== 步骤2：判断is_train并执行对应逻辑 ====================
    if (param_.is_train) {
        // ==================== Lazy Train Phase ====================
        TR_CHECK(config_.sdmp_factor > 1, ValueError,
                 "Lazy train phase requires sdmp_factor > 1, got " << config_.sdmp_factor);

        // ==================== 步骤2.1：洗牌S区索引 ====================
        shuffle_s_indices(param_.phase_id);

        // ==================== 步骤2.2：迭代current_s_samples_次 ====================
        // 迭代current_s_samples_（busy train phase时S区实际存储的样本总数）
        int active_s_region_idx = param_.active_s_region_idx;
        TR_CHECK(active_s_region_idx >= 0 && active_s_region_idx < config_.num_region_s, ValueError,
                 "Invalid active_s_region_idx: " << active_s_region_idx);

        LOG_INFO << "PW " << config_.worker_id << " lazy train phase: "
                 << "iterating " << current_s_samples_ << " samples from S region " << active_s_region_idx;

        for (int i = 0; i < current_s_samples_; ++i) {
            // 从S区复制到EngineBuffer
            copy_sample_from_s_to_eb(active_s_region_idx);

            // 递增local_sample_id_
            end_sample();
        }

        // 通知EngineBuffer没有更多样本
        no_more_samples();

        LOG_INFO << "PW " << config_.worker_id << " lazy train phase completed: "
                 << current_s_samples_ << " samples transferred";

    } else {
        // ==================== Lazy Val Phase ====================
        // 迭代current_c_samples_（busy val phase时C区实际存储的样本总数）
        TR_CHECK(config_.using_cpvs, ValueError,
                 "Lazy val phase requires using_cpvs=true");

        LOG_INFO << "PW " << config_.worker_id << " lazy val phase: "
                 << "iterating " << current_c_samples_ << " samples from C region";

        for (int i = 0; i < current_c_samples_; ++i) {
            // 从C区复制到EngineBuffer
            copy_sample_from_c_to_eb();

            // 递增local_sample_id_
            end_sample();
        }

        // 通知EngineBuffer没有更多样本
        no_more_samples();

        LOG_INFO << "PW " << config_.worker_id << " lazy val phase completed: "
                 << current_c_samples_ << " samples transferred";
    }
}

} // namespace tr
