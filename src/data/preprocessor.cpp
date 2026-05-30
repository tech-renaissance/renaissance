/**
 * @file preprocessor.cpp
 * @brief 图像预处理器实现（V4.0 - 姜总工的新设计）
 * @version 4.0.0
 * @date 2026-01-22
 * @author 技术觉醒团队
 */

#include "renaissance/data/preprocessor.h"
#include "renaissance/data/mnist_loader_dts.h"
#include "renaissance/data/mnist_loader_raw.h"
#include "renaissance/data/cifar_loader_dts.h"
#include "renaissance/data/cifar_loader_raw.h"
#include "renaissance/data/imagenet_loader_dts.h"
#include "renaissance/data/imagenet_loader_raw.h"
#include "renaissance/data/sample_loader.h"
#include "renaissance/data/resize.h"
#include "renaissance/data/center_crop.h"
#include "renaissance/data/random_resized_crop.h"
#include "renaissance/data/fast_random_resized_crop.h"
#include "renaissance/data/random_crop.h"
#include "renaissance/data/random_horizontal_flip.h"
#include "renaissance/data/do_nothing.h"
#include "renaissance/data/gaussian_blur.h"
#include "renaissance/data/random_grayscale.h"
#include "renaissance/data/random_erasing.h"
#include "renaissance/data/normalize.h"
#include "renaissance/data/fused_normalization.h"
#include "renaissance/data/transfer_station.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/global_registry.h"
#include "renaissance/core/types.h"
#include "renaissance/core/rng.h"

// CUDA头文件
#if defined(TR_USE_CUDA)
    #include <cuda_runtime.h>
#endif

#include <turbojpeg.h>
#include <zlib.h>  // for crc32()

// STB Image声明（不定义IMPLEMENTATION，避免重复定义）
#if TR_USE_STB
    #include <stb_image.h>
#endif

#include <cstdint>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstring>
#ifndef _WIN32
    #include <pthread.h>
    #include <sched.h>
#endif
#include <errno.h>
#include <set>

#if defined(TR_SCENE_GPU_CLOUD)
#include <unistd.h>
#endif

#if defined(TR_SCENE_GPU_CLOUD) && defined(TR_USE_LIBNUMA)
#include <numa.h>
#endif

namespace tr {

// =============================================================================
// 单例
// =============================================================================

Preprocessor& Preprocessor::instance() {
    static Preprocessor instance;
    return instance;
}

// =============================================================================
// 构造函数和析构函数
// =============================================================================

Preprocessor::Preprocessor()
    : current_dataloader_(nullptr)
    , num_load_workers_(0)
    , num_preproc_workers_(0)
    , world_size_(-1)
    , batch_size_(32)
    , max_resolution_(-1)
    , num_color_channels_(3)
    , sdmp_factor_(1)
    , using_cpvs_(false)
    , using_progressive_resolution_(false)
    , partial_mode_(false)
    , sample_size_bytes_(0)
    , buffer_size_bytes_(0)
    , is_deployment_mode_(false)
    , train_iteration_id_(0)  // 标志DataLoader读取训练集的次数，不一定等于epoch数或phase数
    , val_iteration_id_(0)  // 标志DataLoader读取验证集的次数，不一定等于epoch数或phase数
    , train_phase_id_(0)  // train phase的编号，也是实际已完成的训练阶段的数量
    , val_phase_id_(0)  // val phase的编号，也是实际已完成的验证阶段的数量
    , is_lazy_phase_(false)
    , config_state_(ConfigState::Unconfigured)
    , dataset_type_(DatasetType::mnist)  // 默认值，会被config_dataset覆盖
    , imagenet_compression_level_(0)
    , train_transforms_set_(false)
    , val_transforms_set_(false)
    , suppress_info_logs_(false)
    , fast_mode_(false)
    , max_intermediate_resolution_(0)
    , workshop_size_calculated_(false)
    , global_initial_seed_(0)
{
    LOG_DEBUG << "Preprocessor constructed";
}

Preprocessor::~Preprocessor() {
    // transfer_station_instances_ 自动析构；alterable_transfer_station_ptrs_ 由 GlobalRegistry 管理
}

// =============================================================================
// 新 API 实现（流畅配置模式）
// =============================================================================

Setup Preprocessor::setup() {
    return Setup();
}

// =============================================================================
// 配置接口
// =============================================================================

void Preprocessor::configure(const Config& config) {
    config_ = config;

    // ==================== 参数验证 ====================
    if (config_.apply_crop && !config_.jpeg_decode) {
        TR_VALUE_ERROR("Invalid configuration: apply_crop=true requires jpeg_decode=true");
        throw std::runtime_error("apply_crop=true requires jpeg_decode=true");
    }

    // 初始化统计数组
    worker_sample_counts_.clear();
    worker_sample_counts_.resize(config_.num_workers, 0);

#ifdef TEST_WITHOUT_PW
// 以下解码分区和日志都仅用于测试

    // 分配解码缓冲区（如果启用解码）
    if (config_.jpeg_decode) {
        allocate_decode_buffers();
    }

    // 创建日志目录（如果启用日志）
    if (config_.enable_logging) {
        try {
            std::filesystem::create_directories(config_.log_dir);
        } catch (const std::exception& e) {
            TR_VALUE_ERROR("Failed to create log directory: " << config_.log_dir
                          << "\n  Error: " << e.what());
        }
    }

#endif  // #ifdef TEST_WITHOUT_PW
}

void Preprocessor::set_train_transforms([[maybe_unused]] const std::vector<std::unique_ptr<PreprocessOperation>>& train_transforms) {
#ifndef TEST_WITHOUT_PW
// 测试模式不需要transform，跳过

    // 检查状态：必须是PreprocessorConfigured状态
    if (config_state_ != ConfigState::PreprocessorConfigured) {
        TR_THROW(ValueError,
                 "set_train_transforms failed: invalid state machine state.\n"
                 "  Current state: " << state_name(config_state_) << "\n"
                 "  Expected state: PreprocessorConfigured\n"
                 "  Solution:\n"
                 "    Please complete the following steps in order:\n"
                 "      1. config_dataset()\n"
                 "      2. config_dataloader()\n"
                 "      3. config_preprocessor()\n"
                 "      4. commit()");
    }

    // 清空旧的transform模板
    train_ops_template_.clear();

    // 将所有ops存入vector以便处理（使用移动语义）
    std::vector<std::unique_ptr<PreprocessOperation>> temp_ops;
    temp_ops.reserve(train_transforms.size());
    for (auto& op : const_cast<std::vector<std::unique_ptr<PreprocessOperation>>&>(train_transforms)) {
        temp_ops.push_back(std::move(op));
    }

    if (temp_ops.size() == 0) {
        // 空参数包：检查数据集类型
        if (is_imagenet()) {
            // ImageNet不允许空操作
            TR_THROW(ValueError,
                     "set_train_transforms() requires at least one transform operation for ImageNet dataset.\n"
                     "  Empty transforms are not allowed for ImageNet.\n"
                     "  ImageNet requires at least one preprocessing operation (e.g., Resize, CenterCrop, RandomResizedCrop).\n"
                     "Solution:\n"
                     "  Provide at least one preprocessing operation.");
        } else {
            // 非ImageNet数据集：使用DoNothing作为占位操作
            temp_ops.push_back(std::make_unique<DoNothing>());
        }
    }

    // =========================================================================
    // PO链自动融合：提取三个记录类参数，过滤出真实PO
    // =========================================================================
    // RandomHorizontalFlip、RandomErasing、Normalize 都是占位记录类，
    // 此处统一提取参数，不加入 filtered_ops（即从PO链中移除）
    bool flip_enabled = false;
    bool erase_enabled = false;
    float erase_p = 0.0f;
    float erase_scale_min = 0.02f;
    float erase_scale_max = 0.33f;
    NormMode norm_mode = NormMode::NO_NORM;  // 默认值

    std::vector<std::unique_ptr<PreprocessOperation>> filtered_ops;
    for (auto& op : temp_ops) {
        if (auto* re = dynamic_cast<RandomErasing*>(op.get())) {
            erase_enabled = true;
            erase_p = re->get_p();
            erase_scale_min = re->scale_min();
            erase_scale_max = re->scale_max();
        } else if (auto* rhf = dynamic_cast<RandomHorizontalFlip*>(op.get())) {
            flip_enabled = true;
        } else if (auto* norm = dynamic_cast<Normalize*>(op.get())) {
            // 提取 Normalize 的 norm_mode（由 Setup::commit() 自动注入）
            norm_mode = norm->mode();
        } else {
            filtered_ops.push_back(std::move(op));
        }
    }

    GlobalRegistry::instance().set_random_erasing_p(erase_p);

    // 如果用户不提供Normalize，norm_mode默认为NO_NORM，FusedNormalization将使用NO_NORM preset

    // 验证第一个PO（对于ImageNet）
    if (!filtered_ops.empty() && is_imagenet()) {
        const auto* first_op = filtered_ops[0].get();
        bool is_do_nothing_op = Preprocessor::is_do_nothing(first_op);

        if (pw_test_mode_) {
            if (is_do_nothing_op) {
            } else if (!Preprocessor::is_crop_or_resize_op(first_op)) {
                TR_THROW(ValueError,
                         "For ImageNet, first transform must be CenterCrop, Resize, or DoNothing in test mode."
                         << " Got: " << typeid(*first_op).name());
            }
        } else {
            if (!Preprocessor::is_crop_or_resize_op(first_op)) {
                TR_THROW(ValueError,
                         "For ImageNet, first transform must be CenterCrop or Resize. "
                         << "Got: " << typeid(*first_op).name());
            }
        }
    }

    // 组装真实PO链（不含记录类）
    train_ops_template_.clear();
    for (auto& op : filtered_ops) {
        train_ops_template_.push_back(std::move(op));
    }

    // =========================================================================
    // 构造并注入 FusedNormalization 到PO链末尾
    // =========================================================================
    NormalizePreset preset = NormalizePreset::NO_NORM;
    switch (norm_mode) {
        case NormMode::NO_NORM:  preset = NormalizePreset::NO_NORM;  break;
        case NormMode::MNIST:    preset = NormalizePreset::MNIST;    break;
        case NormMode::CIFAR:    preset = NormalizePreset::CIFAR;    break;
        case NormMode::IMAGENET: preset = NormalizePreset::IMAGENET; break;
        case NormMode::MLPERF:   preset = NormalizePreset::MLPERF;   break;
    }

    bool use_amp = false;
    try {
        use_amp = GlobalRegistry::instance().using_amp();
    } catch (const TRException&) {
    }

    auto fused_norm = std::make_unique<FusedNormalization>(
        preset,
        use_amp,
        flip_enabled,
        erase_enabled,
        erase_p,
        erase_scale_min,
        erase_scale_max,
        0
    );
    train_ops_template_.push_back(std::move(fused_norm));

    // 设定颜色通道数、输出尺寸和输出stride（包含FusedNormalization）
    int output_size_temp = default_input_width_;
    int max_intermediate_res_train = 0;
    for (auto& op : train_ops_template_) {
        op->set_num_channels(num_color_channels_);
        if (op->is_resize() || op->is_crop()) {
            output_size_temp = op->get_output_size();
        }
        else {
            output_size_temp = op->inference_output_size(output_size_temp);
            op->set_output_size(output_size_temp);
        }

        // 计算AB区大小时排除FusedNormalization
        // FusedNormalization的输出是FP32/FP16，存储在S区/C区/TransformStation，不在AB区
        // AB区只存储中间PO（FusedNormalization之前）的uint8_t输出
        auto* fused_norm = dynamic_cast<FusedNormalization*>(op.get());
        if (fused_norm == nullptr) {
            max_intermediate_res_train = std::max(max_intermediate_res_train, output_size_temp);
        }

        op->calculate_stride();
    }
    int final_train_output_size = output_size_temp;

    // 验证最终输出尺寸（仍需检查max_resolution_约束，但FusedNormalization输出不在AB区）
    // FusedNormalization的输出存储在S区/C区/TransformStation，不影响AB区大小
    // 此检查确保用户设定的分辨率不超过系统的安全限制
    TR_CHECK(final_train_output_size <= max_resolution_, ValueError,
             "Train transforms final output size (" << final_train_output_size
             << ") exceeds max_resolution (" << max_resolution_ << ").\n"
             "  Note: FusedNormalization output is stored in S/C/TransformStation, not AB region.\n"
             "  However, max_resolution_ is a system-wide safety limit.\n"
             "  Solutions:\n"
             "    1. Increase max_resolution in config_preprocessor()\n"
             "    2. Reduce the output size of your transform operations\n"
             "    3. Check if Pad operation is unexpectedly enlarging the output");

    // 更新全局最大中间分辨率
    max_intermediate_resolution_ = std::max(max_intermediate_res_train, max_intermediate_resolution_);

    train_ops_template_[0]->set_as_first();

    // 设置 train_with_rhf（基于融合后的 flip_enabled，而非 flip_ops 是否为空）
    GlobalRegistry::instance().set_train_with_rhf(flip_enabled);

    // 标记train transforms已设置
    train_transforms_set_ = true;

    // 更新状态
    update_config_state();

#else  // #ifdef TEST_WITHOUT_PW
    // 测试模式：只设置标志，不处理transforms
    train_transforms_set_ = true;
    update_config_state();
    LOG_DEBUG << "Train transforms set (TEST_WITHOUT_PW mode)";
#endif  // #ifndef TEST_WITHOUT_PW
}

void Preprocessor::set_val_transforms([[maybe_unused]] const std::vector<std::unique_ptr<PreprocessOperation>>& val_transforms) {
#ifndef TEST_WITHOUT_PW
// 测试模式不需要transform，跳过

    // 检查状态：必须是PreprocessorConfigured或TransformsSet状态
    // （因为set_train_transforms可能已经调用过了）
    if (config_state_ != ConfigState::PreprocessorConfigured &&
        config_state_ != ConfigState::TransformsSet) {
        TR_THROW(ValueError,
                 "set_val_transforms failed: invalid state machine state.\n"
                 "  Current state: " << state_name(config_state_) << "\n"
                 "  Expected state: PreprocessorConfigured or TransformsSet\n"
                 "  Solution:\n"
                 "    Please complete the following steps in order:\n"
                 "      1. config_dataset()\n"
                 "      2. config_dataloader()\n"
                 "      3. config_preprocessor()\n"
                 "      4. commit()");
    }

    // 清空旧的transform模板
    val_ops_template_.clear();

    // 将所有ops存入vector以便处理（使用移动语义）
    std::vector<std::unique_ptr<PreprocessOperation>> temp_ops;
    temp_ops.reserve(val_transforms.size());
    for (auto& op : const_cast<std::vector<std::unique_ptr<PreprocessOperation>>&>(val_transforms)) {
        temp_ops.push_back(std::move(op));
    }

    if (temp_ops.size() == 0) {
        // 空参数包：检查数据集类型
        if (is_imagenet()) {
            // ImageNet不允许空操作
            TR_THROW(ValueError,
                     "set_val_transforms() requires at least one transform operation for ImageNet dataset.\n"
                     "  Empty transforms are not allowed for ImageNet.\n"
                     "  ImageNet validation requires at least one preprocessing operation (e.g., Resize, CenterCrop).\n"
                     "Solution:\n"
                     "  Provide at least one preprocessing operation.");
        } else {
            // 非ImageNet数据集：使用DoNothing作为占位操作
            temp_ops.push_back(std::make_unique<DoNothing>());
        }
    }

    // =========================================================================
    // PO链自动融合：提取记录类参数，验证集强制禁用RandomErasing
    // =========================================================================
    // RandomHorizontalFlip、Normalize 是占位记录类（参数会被提取）
    // RandomErasing 在验证集中明确禁止
    bool val_flip_enabled = false;
    NormMode val_norm_mode = NormMode::NO_NORM;

    std::vector<std::unique_ptr<PreprocessOperation>> filtered_ops;
    for (auto& op : temp_ops) {
        if (auto* re = dynamic_cast<RandomErasing*>(op.get())) {
            TR_THROW(ValueError,
                     "RandomErasing is NOT allowed in validation transforms.\n"
                     "  RandomErasing is a training-only data augmentation operation.\n"
                     "  RandomErasing p value: " << re->get_p() << "\n"
                     "Solution:\n"
                     "  Remove RandomErasing from validation transforms.\n"
                     "  RandomErasing should only be used in set_train_transforms().");
        } else if (auto* rhf = dynamic_cast<RandomHorizontalFlip*>(op.get())) {
            val_flip_enabled = true;
        } else if (auto* norm = dynamic_cast<Normalize*>(op.get())) {
            // 提取 Normalize 的 norm_mode（由 Setup::commit() 自动注入）
            val_norm_mode = norm->mode();
        } else {
            filtered_ops.push_back(std::move(op));
        }
    }

    // 验证第一个PO（对于ImageNet）
    if (!filtered_ops.empty() && is_imagenet()) {
        const auto* first_op = filtered_ops[0].get();
        bool is_do_nothing_op = Preprocessor::is_do_nothing(first_op);

        if (pw_test_mode_) {
            if (is_do_nothing_op) {
            } else if (!Preprocessor::is_crop_or_resize_op(first_op)) {
                TR_THROW(ValueError,
                         "For ImageNet, first transform must be CenterCrop, Resize, or DoNothing in test mode. "
                         << "Got: " << typeid(*first_op).name());
            }
        } else {
            if (!Preprocessor::is_crop_or_resize_op(first_op)) {
                TR_THROW(ValueError,
                         "For ImageNet validation, first transform must be CenterCrop or Resize. "
                         << "Got: " << typeid(*first_op).name());
            }
        }
    }

    // 组装真实PO链（不含记录类）
    val_ops_template_.clear();
    for (auto& op : filtered_ops) {
        val_ops_template_.push_back(std::move(op));
    }

    // =========================================================================
    // 构造并注入 FusedNormalization 到验证集PO链末尾
    // =========================================================================
    NormalizePreset val_preset = NormalizePreset::NO_NORM;
    switch (val_norm_mode) {
        case NormMode::NO_NORM:  val_preset = NormalizePreset::NO_NORM;  break;
        case NormMode::MNIST:    val_preset = NormalizePreset::MNIST;    break;
        case NormMode::CIFAR:    val_preset = NormalizePreset::CIFAR;    break;
        case NormMode::IMAGENET: val_preset = NormalizePreset::IMAGENET; break;
        case NormMode::MLPERF:   val_preset = NormalizePreset::MLPERF;   break;
    }

    bool val_use_amp = false;
    try {
        val_use_amp = GlobalRegistry::instance().using_amp();
    } catch (const TRException&) {
    }

    auto val_fused_norm = std::make_unique<FusedNormalization>(
        val_preset,
        val_use_amp,
        val_flip_enabled,
        false,  // 验证集不启用擦除
        0.0f,   // 擦除概率无效
        0.02f,  // 擦除比例无效
        0.33f,  // 擦除比例无效
        0
    );
    val_ops_template_.push_back(std::move(val_fused_norm));

    // 设定颜色通道数、输出尺寸和输出stride（包含FusedNormalization）
    int output_size_temp = default_input_width_;
    int max_intermediate_res_val = 0;
    for (auto& op : val_ops_template_) {
        op->set_num_channels(num_color_channels_);
        if (op->is_resize() || op->is_crop()) {
            output_size_temp = op->get_output_size();
        }
        else {
            output_size_temp = op->inference_output_size(output_size_temp);
            op->set_output_size(output_size_temp);
        }

        // 计算AB区大小时排除FusedNormalization
        // FusedNormalization的输出是FP32/FP16，存储在S区/C区/TransformStation，不在AB区
        // AB区只存储中间PO（FusedNormalization之前）的uint8_t输出
        auto* fused_norm = dynamic_cast<FusedNormalization*>(op.get());
        if (fused_norm == nullptr) {
            max_intermediate_res_val = std::max(max_intermediate_res_val, output_size_temp);
        }

        op->calculate_stride();
    }
    int final_val_output_size = output_size_temp;

    // 验证最终输出尺寸（仍需检查max_resolution_约束，但FusedNormalization输出不在AB区）
    // FusedNormalization的输出存储在S区/C区/TransformStation，不影响AB区大小
    // 此检查确保用户设定的分辨率不超过系统的安全限制
    TR_CHECK(final_val_output_size <= max_resolution_, ValueError,
             "Validation transforms final output size (" << final_val_output_size
             << ") exceeds max_resolution (" << max_resolution_ << ").\n"
             "  Note: FusedNormalization output is stored in S/C/TransformStation, not AB region.\n"
             "  However, max_resolution_ is a system-wide safety limit.\n"
             "  Solutions:\n"
             "    1. Increase max_resolution in config_preprocessor()\n"
             "    2. Reduce the output size of your transform operations\n"
             "    3. Check if Pad operation is unexpectedly enlarging the output");

    // 更新全局最大中间分辨率
    max_intermediate_resolution_ = std::max(max_intermediate_res_val, max_intermediate_resolution_);

    val_ops_template_[0]->set_as_first();

    // 设置 val_with_rhf（基于融合后的 val_flip_enabled）
    GlobalRegistry::instance().set_val_with_rhf(val_flip_enabled);

    // ==================== CPVS互斥检查 ====================
    if (using_cpvs_) {
        for (const auto& op : val_ops_template_) {
            TR_CHECK(!op->introduce_randomness(), ValueError,
                     "CPVS (Cached Preprocessed Validation Set) optimization "
                     "is incompatible with randomized validation transforms.\n"
                     "  Validation transform '" << op->name() << "' introduces randomness.\n"
                     "  Solutions:\n"
                     "    1. Use non-random transforms for validation (e.g., Resize, CenterCrop)\n"
                     "    2. Disable CPVS by setting using_cpvs=false in config_preprocessor()");
        }
    }

    // 标记val transforms已设置
    val_transforms_set_ = true;

    // 更新状态
    update_config_state();

#else  // #ifdef TEST_WITHOUT_PW
    // 测试模式：只设置标志，不处理transforms
    val_transforms_set_ = true;
    update_config_state();
#endif  // #ifndef TEST_WITHOUT_PW
}

// =============================================================================
// 运行预处理
// =============================================================================

void Preprocessor::run(DataLoader& loader) {

    // 记录开始时间（整个epoch）
    auto epoch_start_time = std::chrono::high_resolution_clock::now();

    // 重置统计
    total_samples_.store(0, std::memory_order_relaxed);
    std::fill(worker_sample_counts_.begin(), worker_sample_counts_.end(), 0);

#ifdef TEST_WITHOUT_PW
// 只有测试模式有日志

    // 清空旧的日志文件（如果启用日志）
    if (config_.enable_logging) {
        for (int i = 0; i < config_.num_workers; ++i) {
            std::ostringstream oss;
            oss << config_.log_dir << "/worker_" << i << ".csv";
            std::string log_path = oss.str();

            // 以空模式打开文件，会清空内容
            std::ofstream(log_path, std::ios::out | std::ios::trunc).close();
        }
    }
#endif  // #ifdef TEST_WITHOUT_PW

    int buffer_count = 0;

    // 重置buffer计数
    buffer_count_ = 0;

#ifndef TEST_WITHOUT_PW
// 测试模式没有PW和TransferStation，不需要同步

    // 【方案A 关键修复1】重置所有同步状态
    workers_finished_.store(0, std::memory_order_seq_cst);
    current_buffer_seq_.store(0, std::memory_order_seq_cst);
    stop_flag_.store(false, std::memory_order_seq_cst);

    // 【Core Dump修复】重置TransferStation同步屏障
    engine_reset_barrier_.store(0, std::memory_order_seq_cst);

    // 【方案A 关键修复2】添加内存屏障，确保所有线程看到一致的状态
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif  // #ifndef TEST_WITHOUT_PW

    // =========================================================================
    // Step 1.2：持久线程池模式（替代原来的创建-销毁模式）
    // =========================================================================

    // 启动持久线程池（只执行一次）
    start_worker_pool(loader);

    do {
        // 通知worker开始新buffer
        notify_workers_new_buffer();

        // 等待worker完成当前buffer
        bool fully_second = (!partial_mode_) && ((pw_param_.is_train? train_phase_id_: val_phase_id_) > 0);
        if (fully_second) {
            // 这里解释一下，因为FULLY MODE的第二个以后的phase都是从已有的存储中读取，没有buffer的概念，直到处理完整个数据集才结束，所以不应该设置超时
            wait_workers_complete_buffer(true);  // 永久等待
        }
        else {  // PARTIAL MODE或第一个phase（必定加载）
            wait_workers_complete_buffer(false);  // 20s超时
        }

        ++buffer_count;

        // 触发DataLoader加载下一个buffer（会等待当前buffer被消费完）
        if (loader.has_more_buffers()) {
            loader.load_next_buffer();
        } else {
            break;
        }

    } while (true);

    // 停止线程池
    stop_worker_pool();

    // 记录结束时间（整个epoch）
    auto epoch_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = epoch_end_time - epoch_start_time;

    // 输出统计信息
    size_t total = total_samples_.load(std::memory_order_acquire);
    buffer_count_ = buffer_count;  // 保存buffer计数到成员变量
    // LOG_INFO << "Preprocessor completed: " << total << " total samples";
    // LOG_INFO << "Total buffers processed: " << buffer_count;
    // LOG_INFO << "Total epoch time: " << elapsed.count() << " seconds";

    // 验证每个worker的样本数（整个epoch的累积）
    size_t min_count = SIZE_MAX;
    size_t max_count = 0;
    for (int i = 0; i < config_.num_workers; ++i) {
        size_t count = worker_sample_counts_[i];
        min_count = std::min(min_count, count);
        max_count = std::max(max_count, count);
        LOG_DEBUG << "Worker " << i << ": " << count << " samples (total)";
    }

    // 验证样本数均匀性（姜总工的要求：最多相差1）
    if (max_count - min_count > 1) {
        LOG_WARN << "Worker sample distribution is not uniform: difference="
                 << (max_count - min_count) << " (expected <= 1)";
    }
}

// =============================================================================
// Worker线程函数（姜总工的静态领取设计）
// =============================================================================
// 解码缓冲区管理
// =============================================================================

void Preprocessor::allocate_decode_buffers() {
#ifdef TEST_WITHOUT_PW
// 只有测试模式需要这个区。正常模式的解码由PW完成，不需要这个区

    free_decode_buffers();  // 先释放旧的

    worker_decode_buffers_.resize(config_.num_workers);

    // ==================== 计算crop缓冲区大小 ====================
    constexpr size_t SIMD_ALIGNMENT = 64;
    constexpr int NUM_CHANNELS = 3;
    constexpr int DEFAULT_OUTPUT_SIZE = 224;

    // 对齐的stride (向上取整到64字节)
    size_t crop_stride = ((DEFAULT_OUTPUT_SIZE * NUM_CHANNELS + SIMD_ALIGNMENT - 1) / SIMD_ALIGNMENT) * SIMD_ALIGNMENT;
    size_t crop_buffer_size = crop_stride * DEFAULT_OUTPUT_SIZE;

    for (int i = 0; i < config_.num_workers; ++i) {
        // ==================== 分配JPEG解码缓冲区 ====================
        worker_decode_buffers_[i].memory = static_cast<uint8_t*>(
            malloc(worker_decode_buffers_[i].size)
        );

        if (!worker_decode_buffers_[i].memory) {
            TR_MEMORY_ERROR("Failed to allocate decode buffer for worker " << i
                           << "\n  Requested size: " << worker_decode_buffers_[i].size);
        }

        // ✅ 创建持久handle（只创建一次，循环中复用）
        worker_decode_buffers_[i].handle = tjInitDecompress();
        if (!worker_decode_buffers_[i].handle) {
            TR_MEMORY_ERROR("Failed to initialize turbojpeg handle for worker " << i);
        }

        // ==================== 分配RandomResizedCrop输出缓冲区 ====================
        if (config_.apply_crop) {
            worker_decode_buffers_[i].crop_buffer_size = crop_buffer_size;
            #ifdef _WIN32
                worker_decode_buffers_[i].crop_output_buffer = static_cast<uint8_t*>(
                    _aligned_malloc(crop_buffer_size, SIMD_ALIGNMENT)  // 注意：Windows是(size, alignment)
                );
            #else
                worker_decode_buffers_[i].crop_output_buffer = static_cast<uint8_t*>(
                    aligned_alloc(SIMD_ALIGNMENT, crop_buffer_size)   // Linux是(alignment, size)
                );
            #endif

            if (!worker_decode_buffers_[i].crop_output_buffer) {
                TR_MEMORY_ERROR("Failed to allocate crop output buffer for worker " << i
                               << "\n  Requested size: " << crop_buffer_size);
            }

            // 初始化crop参数（使用默认值）
            worker_decode_buffers_[i].crop_params.output_size = DEFAULT_OUTPUT_SIZE;

            // 初始化随机数生成器（每个worker使用不同seed）
            worker_decode_buffers_[i].rng = FastRandom(42 + i);

            LOG_DEBUG << "Allocated 128MB decode buffer + " << (crop_buffer_size / 1024.0 / 1024.0)
                     << "MB crop buffer for worker " << i;
        } else {
            LOG_DEBUG << "Allocated 128MB decode buffer for worker " << i << " (crop disabled)";
        }
    }
#endif  // #ifdef TEST_WITHOUT_PW
}

void Preprocessor::free_decode_buffers() {
#ifdef TEST_WITHOUT_PW
// 只有测试模式有这个区

    for (auto& buf : worker_decode_buffers_) {
        // ✅ 销毁持久handle
        if (buf.handle) {
            tjDestroy(buf.handle);
            buf.handle = nullptr;
        }

        // ==================== 释放JPEG解码缓冲区 ====================
        if (buf.memory) {
            free(buf.memory);
            buf.memory = nullptr;
        }

        // ==================== 释放Crop输出缓冲区 ====================
        if (buf.crop_output_buffer) {
            #ifdef _WIN32
                _aligned_free(buf.crop_output_buffer);
            #else
                free(buf.crop_output_buffer);
            #endif
            buf.crop_output_buffer = nullptr;
        }
    }
    worker_decode_buffers_.clear();

    LOG_DEBUG << "Freed all decode buffers, crop buffers, and handles";
#endif  // #ifdef TEST_WITHOUT_PW
}

// =============================================================================
// RandomResizedCrop实现
// =============================================================================

void* Preprocessor::ResizerCache::get_resizer(size_t src_w, size_t src_h,
                                              size_t dst_w, size_t dst_h) {
#ifndef TEST_WITHOUT_PW
// 正常模式如果调用这个方法，就要报错
    TR_THROW(ValueError, "ResizerCache::get_resizer can only be used in TEST_WITHOUT_PW mode.");
#endif

    // 如果尺寸匹配，复用缓存的 resizer
    if (cached_resizer_ &&
        cached_src_w_ == src_w && cached_src_h_ == src_h &&
        cached_dst_w_ == dst_w && cached_dst_h_ == dst_h) {
        return cached_resizer_;
    }

    // 释放旧的 resizer
    if (cached_resizer_) {
        SimdRelease(cached_resizer_);
        cached_resizer_ = nullptr;
    }

    // 创建新的 resizer
    cached_resizer_ = SimdResizerInit(
        src_w, src_h, dst_w, dst_h,
        3,  // RGB 3通道
        SimdResizeChannelByte,
        SimdResizeMethodBilinear
    );

    if (!cached_resizer_) {
        std::ostringstream oss;
        oss << "Failed to create Simd Resizer: "
            << src_w << "x" << src_h << " -> " << dst_w << "x" << dst_h;
        LOG_ERROR << oss.str();
        throw std::runtime_error(oss.str());
    }

    cached_src_w_ = src_w;
    cached_src_h_ = src_h;
    cached_dst_w_ = dst_w;
    cached_dst_h_ = dst_h;

    return cached_resizer_;
}

void Preprocessor::apply_random_resized_crop(int worker_id,
                                            const uint8_t* decoded_ptr,
                                            int width,
                                            int height,
                                            size_t pitch) {
#ifndef TEST_WITHOUT_PW
// 正常模式如果调用这个方法，就要报错
    TR_THROW(ValueError, "ResizerCache::get_resizer can only be used in TEST_WITHOUT_PW mode.");
#endif

    constexpr int NUM_CHANNELS = 3;
    constexpr int MAX_ATTEMPTS = 10;

    auto& buf = worker_decode_buffers_[worker_id];
    const auto& params = buf.crop_params;

    // ==================== Step 1: 计算裁剪参数 ====================
    const float area = static_cast<float>(width * height);

    int crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;  // 初始化为默认值
    bool success = false;

    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        // 随机选择目标面积（对数均匀分布，匹配PyTorch）
        const float target_area = area * std::exp(buf.rng.uniform(params.log_scale_min, params.log_scale_max));

        // 随机选择宽高比（对数均匀分布）
        const float aspect_ratio = std::exp(buf.rng.uniform(params.log_ratio_min, params.log_ratio_max));

        // 计算裁剪尺寸
        crop_w = static_cast<int>(std::round(std::sqrt(target_area * aspect_ratio)));
        crop_h = static_cast<int>(std::round(std::sqrt(target_area / aspect_ratio)));

        // 检查是否在有效范围内
        if (crop_w > 0 && crop_w <= width && crop_h > 0 && crop_h <= height) {
            // 随机选择裁剪起始位置
            crop_x = static_cast<int>(buf.rng.uniform() * (width - crop_w + 1));
            crop_y = static_cast<int>(buf.rng.uniform() * (height - crop_h + 1));
            success = true;
            break;
        }
    }

    // ==================== Step 2: 回退策略（中心裁剪）====================
    if (!success) {
        const float in_ratio = static_cast<float>(width) / height;
        if (in_ratio < params.ratio_min) {
            crop_w = width;
            crop_h = static_cast<int>(std::round(width / params.ratio_min));
        } else if (in_ratio > params.ratio_max) {
            crop_h = height;
            crop_w = static_cast<int>(std::round(height * params.ratio_max));
        } else {
            crop_w = width;
            crop_h = height;
        }
        crop_x = (width - crop_w) / 2;
        crop_y = (height - crop_h) / 2;
    }

    // ==================== Step 3: 获取裁剪区域的起始指针 ====================
    const uint8_t* crop_src = decoded_ptr + crop_y * pitch + crop_x * NUM_CHANNELS;

    // ==================== Step 4: 获取或创建 Resizer ====================
    void* resizer = buf.resizer_cache.get_resizer(
        crop_w, crop_h,
        params.output_size, params.output_size
    );

    // ==================== Step 5: 计算 crop 输出缓冲区的 stride ====================
    constexpr size_t SIMD_ALIGNMENT = 64;
    size_t output_stride = ((params.output_size * NUM_CHANNELS + SIMD_ALIGNMENT - 1) / SIMD_ALIGNMENT) * SIMD_ALIGNMENT;

    // ==================== Step 6: 执行 Resize（从裁剪区域到输出）====================
    SimdResizerRun(
        resizer,
        crop_src, pitch,                // 源：裁剪区域（使用原始 stride）
        buf.crop_output_buffer, output_stride  // 目标：输出缓冲区
    );
}

// =============================================================================
// 统计信息
// =============================================================================

Preprocessor::Stats Preprocessor::get_stats() const {
    Stats stats;
    stats.total_samples = total_samples_.load(std::memory_order_acquire);
    stats.buffer_count = buffer_count_;
    stats.per_worker = worker_sample_counts_;  // 复制（非原子）
    return stats;
}

void Preprocessor::reset() {
    total_samples_.store(0, std::memory_order_relaxed);
    buffer_count_ = 0;
    std::fill(worker_sample_counts_.begin(), worker_sample_counts_.end(), 0);

    // 重置状态机
    config_state_ = ConfigState::Unconfigured;
    train_transforms_set_ = false;
    val_transforms_set_ = false;

    LOG_DEBUG << "Preprocessor stats and state machine reset";
}

int Preprocessor::steps_per_epoch() const {
    TR_CHECK(steps_per_epoch_ > 0, ValueError,
             "steps_per_epoch has not been calculated or is invalid. "
             "Make sure Setup::commit() has been called and completed successfully.");
    return steps_per_epoch_;
}

void Preprocessor::calculate_steps_per_epoch() {
    const auto& reg = GlobalRegistry::instance();

    // 从DataLoader获取训练集样本总数
    TR_CHECK(current_dataloader_ != nullptr, ValueError,
             "DataLoader not initialized. Cannot calculate steps_per_epoch.");

    const size_t total_train_samples = current_dataloader_->num_train_samples();
    TR_CHECK(total_train_samples > 0, ValueError,
             "Training set has zero samples. Cannot calculate steps_per_epoch.");

    // 计算全局batch size和每个epoch的步数
    const int world_size = reg.world_size();
    const int local_batch_size = reg.get_local_batch_size();
    const int global_batch_size = world_size * local_batch_size;

    TR_CHECK(global_batch_size > 0, ValueError,
             "Global batch size must be positive. Got: " << global_batch_size);

    // 计算每个epoch的步数（向上取整）
    steps_per_epoch_ = static_cast<int>((total_train_samples + global_batch_size - 1) / global_batch_size);

    LOG_INFO << "Calculated steps_per_epoch: " << steps_per_epoch_
             << " (total_samples: " << total_train_samples
             << ", global_batch_size: " << global_batch_size << ")";
}

// =============================================================================
// Step 1.2：线程持久化实现
// =============================================================================

void Preprocessor::start_worker_pool(DataLoader& loader) {
    worker_pool_.clear();
    worker_pool_.reserve(config_.num_workers);

    for (int i = 0; i < config_.num_workers; ++i) {
        worker_pool_.emplace_back([this, i, &loader]() {
            // Worker线程：直接调用worker_func_persistent()
            // worker_func_persistent()内部会持续处理多个buffer直到整个epoch结束
            worker_func_persistent(i, loader);
        });
    }

    // 等待所有线程启动完成
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// ---------------------------------------------------------------------------

void Preprocessor::stop_worker_pool() {
    // 设置停止标志
    stop_flag_.store(true, std::memory_order_release);

    // 唤醒所有等待的worker
    current_buffer_seq_.fetch_add(1, std::memory_order_acq_rel);

    // 等待所有线程退出
    for (auto& t : worker_pool_) {
        if (t.joinable()) {
            t.join();
        }
    }

    worker_pool_.clear();
    stop_flag_.store(false, std::memory_order_release);
}

// ---------------------------------------------------------------------------

void Preprocessor::notify_workers_new_buffer() {
    // 【方案A 关键修复】必须先重置计数器，防止抢跑的Worker的计数被覆盖
    workers_finished_.store(0, std::memory_order_release);

    // 确保计数器重置对所有线程可见后，再更新序号唤醒Worker
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // 最后再唤醒Worker
    int old_seq = current_buffer_seq_.fetch_add(1, std::memory_order_acq_rel);
    LOG_DEBUG << "[PREPROC] notify_workers_new_buffer(): old_seq=" << old_seq << ", new_seq=" << (old_seq + 1);
}

// ---------------------------------------------------------------------------

void Preprocessor::wait_workers_complete_buffer(bool wait_forever) {
    // 等待所有worker完成当前buffer
    int expected = config_.num_workers;
    auto start = std::chrono::steady_clock::now();
    const auto TIMEOUT = std::chrono::seconds(36000);  // 36000秒超时

    LOG_DEBUG << "[PREPROC] wait_workers_complete_buffer(): START, expected=" << expected
             << ", wait_forever=" << wait_forever;

    while (workers_finished_.load(std::memory_order_acquire) < expected) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (!wait_forever && elapsed > TIMEOUT) {
            int current = workers_finished_.load(std::memory_order_acquire);
            LOG_ERROR << "TIMEOUT waiting for workers to complete buffer! "
                      << "workers_finished_=" << current << "/" << expected
                      << " (waited " << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << "s)";

            TR_THROW(TimeoutError, "Workers timeout, possible deadlock detected");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 改为1ms，减少CPU占用
    }
}

// ---------------------------------------------------------------------------

#if TR_USE_STB
// =============================================================================
// STB Image备用解码函数（用于处理TurboJPEG无法解码的特殊JPEG格式）
// =============================================================================

/**
 * @brief 使用STB Image解码JPEG（备用方案）
 * @param jpeg_data JPEG数据指针
 * @param jpeg_size JPEG数据大小
 * @param decode_buffer 解码输出缓冲区
 * @param buffer_size 缓冲区大小
 * @param width 输出的图像宽度
 * @param height 输出的图像高度
 * @return true=成功, false=失败
 */
static bool decode_jpeg_with_stb(
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
    int pitch = ((width * 3 + 63) / 64) * 64;  // 64字节对齐

    // 验证缓冲区容量
    size_t required_size = pitch * height;
    if (required_size > buffer_size) {
        stbi_image_free(stb_data);
        return false;
    }

    // 复制到decode_buffer（STB返回的数据是紧密打包的）
    for (int row = 0; row < height; ++row) {
        const stbi_uc* src_row = stb_data + row * (width * 3);
        uint8_t* dst_row = decode_buffer + row * pitch;
        std::memcpy(dst_row, src_row, width * 3);
    }

    stbi_image_free(stb_data);
    return true;
}
#endif

// ---------------------------------------------------------------------------

void Preprocessor::worker_func_persistent(int worker_id, DataLoader& loader) {

#ifndef TEST_WITHOUT_PW
// 注意：如果是正常运作，那就需要绑核和PW和TransferStation，但如果是无PW的测试，以下就应该跳过

    // ==================== Step 0: CPU绑核（GPU_CLOUD + auto_binding）====================
#if defined(TR_SCENE_GPU_CLOUD)
    if (auto_cpu_binding_ && !selected_gpu_ids_.empty()) {
        bind_worker_to_cpu(worker_id);
    }
#endif

    // ==================== PW实例创建（各worker线程自己创建）====================
    // 检查并创建当前worker的PW实例
    {
        static std::mutex pw_create_mutex;
        std::lock_guard<std::mutex> lock(pw_create_mutex);

        // 确保pw_instances_有足够空间
        if (pw_instances_.size() < static_cast<size_t>(num_preproc_workers_)) {
            pw_instances_.resize(num_preproc_workers_);
        }

        // 如果当前worker的PW还没创建，就创建它
        if (!pw_instances_[worker_id]) {
            LOG_DEBUG << "[Worker " << worker_id << "] Creating PW instance, test_mode="
                      << (pw_test_mode_ ? "ON" : "OFF");

            // 构建PW配置
            PreprocessWorker::Config pw_config;
            pw_config.worker_id = worker_id;

            // 计算engine信息
            int num_workers_per_engine = num_preproc_workers_ / world_size_;
            pw_config.engine_id = worker_id % world_size_;
            pw_config.pid_in_engine = worker_id / world_size_;
            pw_config.num_workers_per_engine = num_workers_per_engine;

            pw_config.global_initial_seed = global_initial_seed_;

            // Workshop大小（从全局配置获取）
            pw_config.region_d_size = workshop_region_d_size_;
            pw_config.region_ab_size = workshop_region_ab_size_;
            pw_config.region_t_size = workshop_region_t_size_;
            pw_config.region_s_size = workshop_region_s_size_;
            pw_config.region_c_size = workshop_region_c_size_;
            pw_config.num_region_s = workshop_num_region_s_;

            // Preprocessor参数
            pw_config.local_batch_size = batch_size_;
            pw_config.world_size = world_size_;
            pw_config.sdmp_factor = sdmp_factor_;
            pw_config.using_cpvs = using_cpvs_;
            pw_config.using_progressive_resolution = using_progressive_resolution_;

            // 数据集信息
            pw_config.dataset_type = dataset_type_;
            pw_config.num_color_channels = num_color_channels_;
            pw_config.raw_image_width = default_input_width_;   // ImageNet不需要
            pw_config.raw_image_height = default_input_width_;  // ImageNet不需要

            // 测试模式
            pw_config.test_mode = pw_test_mode_;

            // SDMP/CPVS缓存容量（样本数）
            pw_config.max_s_samples = max_s_samples_;
            pw_config.max_c_samples = max_c_samples_;

            // TransferStation指针（V3.14.0 - 跨步分配：worker_id % world_size）
            // 对应关系：PW ID % world_size = TransferStation ID
            if (!transfer_station_instances_.empty() && pw_config.engine_id < static_cast<int>(transfer_station_instances_.size())) {
                pw_config.transfer_station = transfer_station_instances_[pw_config.engine_id].get();
                LOG_DEBUG << "[Worker " << worker_id << "] Assigned to TransferStation " << pw_config.engine_id;
            } else {
                pw_config.transfer_station = nullptr;
                LOG_WARN << "[Worker " << worker_id << "] TransferStation not available, set to NULL";
            }

            LOG_DEBUG << "[Worker " << worker_id << "] pw_config.test_mode = "
                      << (pw_config.test_mode ? "ON" : "OFF");

            // 创建PW实例（使用new而不是make_unique，因为unique_ptr不能直接拷贝）
            pw_instances_[worker_id] = std::unique_ptr<PreprocessWorker>(
                new PreprocessWorker(
                    pw_config,
                    train_ops_template_,  // 训练集PO列表
                    val_ops_template_,     // 验证集PO列表
                    &pw_param_
                )
            );

            LOG_DEBUG << "[Worker " << worker_id << "] PW instance created successfully";
#if defined(TR_SCENE_GPU_CLOUD)
            if (auto_cpu_binding_ && !selected_gpu_ids_.empty()) {
                int rank = worker_id % world_size_;
                int numa_node = GlobalRegistry::instance().staging_memory_numa_node(rank);
                int target_cpu = GlobalRegistry::instance().cpu_binding_map()[worker_id];
                auto tid = std::this_thread::get_id();
                std::cout << "[StagingDebug] PW created: "
                               << "worker_id=" << worker_id << ", "
                               << "tid=" << tid << ", "
                               << "RANK=" << rank << ", "
                               << "NUMA=" << numa_node << ", "
                               << "CPU=" << target_cpu << "\n";
            }
#endif
        }
    }

    // ==================== PW更新参数 ====================
    {
		if (pw_instances_[worker_id]) {
			pw_instances_[worker_id]->update_parameters();
		}
	}

    // ==================== TransferStation更新 ====================
    // 只有每个Engine的第一个PW（worker_id < world_size_）负责更新对应的TransferStation
    if (worker_id < world_size_) {
        int engine_id = worker_id % world_size_;  // engine_id = worker_id % world_size_
        if (transfer_station_instances_[engine_id]) {
            transfer_station_instances_[engine_id]->reset_and_update();
        }
        // 【Core Dump修复1】Leader完成重置后签到
        engine_reset_barrier_.fetch_add(1, std::memory_order_acq_rel);
    }

    // ==================== 【Core Dump修复1】同步屏障 ====================
    // 只有Leader（worker_id < world_size_）参与barrier同步
    // Follower直接跳过，无需等待
    if (worker_id < world_size_) {
        while (engine_reset_barrier_.load(std::memory_order_acquire) < world_size_) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    // 【Core Dump修复1】内存屏障：确保所有线程看到重置后的最新内存状态
    std::atomic_thread_fence(std::memory_order_acq_rel);
#endif  // #ifndef TEST_WITHOUT_PW



#ifdef TEST_WITHOUT_PW
// 注意：以下日志是仅用于测试的，所以是#ifdef TEST_WITHOUT_PW才启用

    // 打开CSV日志文件（如果启用）
    std::ofstream log_file;
    std::ofstream crc_file;
    if (config_.enable_logging) {
        // 打开worker_x.csv (记录worker_id, size, label, first_byte)
        std::ostringstream oss;
        oss << config_.log_dir << "/worker_" << worker_id << ".csv";
        std::string log_path = oss.str();

        log_file.open(log_path, std::ios::out | std::ios::trunc);
        if (!log_file.is_open()) {
            TR_FILE_NOT_FOUND("Failed to open log file: " << log_path);
        }

        // 如果启用CRC计算，打开crc_x.csv (记录crc32)
        if (config_.calc_crc) {
            std::ostringstream crc_oss;
            crc_oss << config_.log_dir << "/crc_" << worker_id << ".csv";
            std::string crc_path = crc_oss.str();

            crc_file.open(crc_path, std::ios::out | std::ios::trunc);
            if (!crc_file.is_open()) {
                TR_FILE_NOT_FOUND("Failed to open CRC file: " << crc_path);
            }
        }
    }
#endif  // #ifdef TEST_WITHOUT_PW

    size_t local_count = 0;
    size_t sample_count = 0;  // 总样本计数器（用于打印前10个样本的CRC）

    // =========================================================================
    // 【关键修复】恢复外层循环，让worker持续处理多个buffer直到整个epoch结束
    // =========================================================================
    while (true) {
        // 等待新buffer的信号
        int last_seen = current_buffer_seq_.load(std::memory_order_acquire);
        while (current_buffer_seq_.load(std::memory_order_acquire) == last_seen &&
               !stop_flag_.load(std::memory_order_acquire)) {  // 只要没有得到最终退出信号，就等待buffer更新
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        if (stop_flag_.load(std::memory_order_acquire)) {  // 得到最终退出信号
#ifndef TEST_WITHOUT_PW
			if (pw_instances_[worker_id]) {
				// 让PW向对应的TransferStation表明已无更多样本
				pw_instances_[worker_id]->no_more_samples();
			}
#endif  // #ifndef TEST_WITHOUT_PW
            break;
        }

        // 处理当前buffer的所有样本
        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;



		while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {


#ifdef TEST_WITHOUT_PW
// 所谓快速模式，只能在测试时使用

			// 快速模式：直接计数，不执行任何预处理
			if (fast_mode_ && !config_.jpeg_decode && !config_.calc_crc) {
				local_count++;
				total_samples_.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
#else
			// ==================== PW模式（V4.1）：调用PW处理样本 ====================
			if (pw_instances_[worker_id]) {
				// 调用PW的work方法处理样本
				pw_instances_[worker_id]->work(label, data_ptr, data_size);
				local_count++;
				total_samples_.fetch_add(1, std::memory_order_relaxed);
				sample_count++;
				continue;
			}

			// 如果PW没创建成功，报错
			LOG_ERROR << "Worker " << worker_id << " PW instance not created, skipping sample";
#endif



#ifdef TEST_WITHOUT_PW
// 计算CRC和日志输出仅用于测试。这里的JPEG解码也是与PW的解码功能不同的，是独立的测试，因此在正常模式要禁止

			int first_byte = -1;
			uint32_t crc_value = 0;  // CRC32校验值（如果启用）
			// 计算CRC32（如果启用）
			if (config_.calc_crc) {
				crc_value = crc32(0L, Z_NULL, 0);  // 初始化CRC
				crc_value = crc32(crc_value, data_ptr, static_cast<uInt>(data_size));

				// 【调试】打印前10个样本和第962-973个样本的CRC（覆盖第二个buffer的前10个）
				if (sample_count < 10 || (sample_count >= 961 && sample_count < 973)) {
					LOG_DEBUG << "[PREPROC CRC] Worker=" << worker_id
							 << ", Sample #" << (sample_count + 1)
							 << ", DataPtr=" << static_cast<const void*>(data_ptr)
							 << ", Size=" << data_size
							 << ", Label=" << label
							 << ", CRC32=0x" << std::hex << std::uppercase << crc_value << std::dec;
				}
			}

			// JPEG解码（如果启用）
			if (config_.jpeg_decode) {
				tjhandle handle = worker_decode_buffers_[worker_id].handle;
				if (handle) {
					int width, height, subsamp, colorspace;

					// 获取JPEG头信息
					if (tjDecompressHeader3(handle, data_ptr, static_cast<unsigned long>(data_size),
										  &width, &height, &subsamp, &colorspace) == 0) {

						// ✅ 64字节对齐的stride（与PW/test_po/fast_crop一致）
						int pitch = ((width * 3 + 63) / 64) * 64;

						// 解码到worker专属的128MB缓冲区
						uint8_t* decode_buffer = worker_decode_buffers_[worker_id].memory;

						// ✅ 使用优化flags（FASTDCT + NOREALLOC，提升30-50%性能）
						if (tjDecompress2(handle, data_ptr, static_cast<unsigned long>(data_size),
										 decode_buffer, width, pitch, height,
										 TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC) == 0) {
							// 记录解码后第一个字节
							first_byte = static_cast<int>(decode_buffer[0]);

							// ✅ RandomResizedCrop（如果启用）
							if (config_.apply_crop) {
								apply_random_resized_crop(worker_id, decode_buffer,
														 width, height, pitch);
								// 注意：crop结果在crop_output_buffer中，可用于后续处理
							}
						} else {
							// TurboJPEG解码失败，尝试使用STB备用解码
							#if TR_USE_STB
							uint8_t* stb_decode_buffer = worker_decode_buffers_[worker_id].memory;
							size_t stb_buffer_size = worker_decode_buffers_[worker_id].size;
							int stb_width, stb_height;
							if (decode_jpeg_with_stb(data_ptr, data_size,
													   stb_decode_buffer, stb_buffer_size,
													   stb_width, stb_height)) {
								// STB解码成功
								width = stb_width;
								height = stb_height;
								int stb_pitch = ((width * 3 + 63) / 64) * 64;  // 64字节对齐
								first_byte = static_cast<int>(stb_decode_buffer[0]);

								// ✅ RandomResizedCrop（如果启用）
								if (config_.apply_crop) {
									apply_random_resized_crop(worker_id, stb_decode_buffer,
															 width, height, stb_pitch);
								}
							} else {
								// STB备用解码也失败
								LOG_ERROR << "[Worker " << worker_id << "] Failed to decode JPEG: "
										 << "both TurboJPEG and STB failed (sample size=" << data_size
										 << ", label=" << label << "), skipping sample";
							}
							#endif
						}
					}
					// TurboJPEG读取头失败，尝试使用STB备用解码
					#if TR_USE_STB
					else {
						uint8_t* stb_decode_buffer = worker_decode_buffers_[worker_id].memory;
						size_t stb_buffer_size = worker_decode_buffers_[worker_id].size;
						int stb_width, stb_height;
						if (decode_jpeg_with_stb(data_ptr, data_size,
												   stb_decode_buffer, stb_buffer_size,
												   stb_width, stb_height)) {
							// STB解码成功
							int stb_pitch2 = ((stb_width * 3 + 63) / 64) * 64;  // 64字节对齐
							first_byte = static_cast<int>(stb_decode_buffer[0]);

							// ✅ RandomResizedCrop（如果启用）
							if (config_.apply_crop) {
								apply_random_resized_crop(worker_id, stb_decode_buffer,
														 stb_width, stb_height, stb_pitch2);
							}
						} else {
							// STB备用解码也失败
							LOG_ERROR << "[Worker " << worker_id << "] Failed to decode JPEG header: "
									 << "both TurboJPEG and STB failed (sample size=" << data_size
									 << ", label=" << label << "), skipping sample";
						}
					}
					#endif
				}
			}

			// 模拟预处理延迟（可选）
			if (config_.simulate_delay) {
				std::this_thread::sleep_for(
					std::chrono::microseconds(config_.delay_us)
				);
			}

			// 写入CSV日志（分离格式）
			if (config_.enable_logging) {
				// worker_x.csv: worker_id,size,label,first_byte (无表头)
				log_file << worker_id << "," << data_size << "," << label << "," << first_byte << "\n";

				// crc_x.csv: crc32 (无表头)
				if (config_.calc_crc) {
					crc_file << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << crc_value << "\n";
				}
			}

// 如果是正常模式，以下几个计数器已经递增并且continue了
			sample_count++;  // 增加总样本计数
			local_count++;
			total_samples_.fetch_add(1, std::memory_order_relaxed);
#endif  // #ifdef TEST_WITHOUT_PW
		}







        // =========================================================================
        // 当前buffer处理完毕，报告完成状态
        // =========================================================================
        int finished_count = workers_finished_.fetch_add(1, std::memory_order_acq_rel);
        if (worker_id == 0) {
            LOG_DEBUG << "[PW0] Buffer completed, workers_finished_ now=" << (finished_count + 1);
        }
    }

    // =========================================================================
    // 保存此worker的样本数（累加到整个epoch的总数，使用互斥锁保护）
    // =========================================================================
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        worker_sample_counts_[worker_id] += local_count;
    }

#ifdef TEST_WITHOUT_PW
    if (log_file.is_open()) {
        log_file.close();
    }
    if (crc_file.is_open()) {
        crc_file.close();
    }
#endif  // #ifdef TEST_WITHOUT_PW
}

// =============================================================================
// 新配置方法：辅助方法
// =============================================================================

namespace {
    // 辅助函数：解析数据集名称
    DatasetType parse_dataset_type(const std::string& dataset_name) {
        std::string name = dataset_name;
        // 转换为小写
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);

        if (name == "mnist") {
            return DatasetType::mnist;
        } else if (name == "cifar10" || name == "cifar_10" || name == "cifar-10") {
            return DatasetType::cifar_10;
        } else if (name == "cifar100" || name == "cifar_100" || name == "cifar-100") {
            return DatasetType::cifar_100;
        } else if (name == "imagenet" || name == "imageNet") {
            return DatasetType::imagenet;
        } else {
            TR_THROW(ValueError,
                     "Unknown dataset name: " << dataset_name
                     << ". Supported: mnist, cifar10, cifar100, imagenet");
        }
    }

    // 辅助函数：判断是否使用DTS格式
    bool is_dts_format([[maybe_unused]] DatasetType type, bool dts_format) {
        // ImageNet支持DTS和RAW
        // MNIST/CIFAR支持DTS和RAW
        return dts_format;
    }
} // anonymous namespace

void Preprocessor::check_state(ConfigState expected_state, const std::string& method_name) {
    if (config_state_ != expected_state) {
        // 构建详细的错误消息
        std::ostringstream oss;
        oss << method_name << " failed: invalid state machine state.\n";

        // 当前状态和期望状态
        oss << "  Current state: " << state_name(config_state_) << "\n";
        oss << "  Expected state: " << state_name(expected_state) << "\n";

        // 提供解决方案
        oss << "  Solution:\n";
        switch (expected_state) {
            case ConfigState::DatasetSelected:
                oss << "    Please call config_dataset() first.\n";
                break;
            case ConfigState::DataLoaderConfigured:
                oss << "    Please call config_dataset() followed by config_dataloader().\n";
                break;
            case ConfigState::PreprocessorConfigured:
                oss << "    Please complete the following steps in order:\n";
                oss << "      1. config_dataset()\n";
                oss << "      2. config_dataloader()\n";
                oss << "      3. config_preprocessor()\n";
                break;
            case ConfigState::Initialized:
                oss << "    Please complete all configuration steps:\n";
                oss << "      1. config_dataset()\n";
                oss << "      2. config_dataloader()\n";
                oss << "      3. config_preprocessor()\n";
                oss << "      4. set_train_transforms()\n";
                oss << "      5. set_val_transforms()\n";
                break;
            default:
                oss << "    Please check the configuration flow.\n";
                break;
        }

        TR_THROW(ValueError, oss.str());
    }
}

std::string Preprocessor::state_name(ConfigState state) {
    switch (state) {
        case ConfigState::Unconfigured: return "Unconfigured";
        case ConfigState::DatasetSelected: return "DatasetSelected";
        case ConfigState::DataLoaderConfigured: return "DataLoaderConfigured";
        case ConfigState::PreprocessorConfigured: return "PreprocessorConfigured";
        case ConfigState::TransformsSet: return "TransformsSet";
        case ConfigState::Initialized: return "Initialized";
        default: return "Unknown";
    }
}

void Preprocessor::update_config_state() {
    // 根据transforms设置情况更新状态
    if (train_transforms_set_ && val_transforms_set_) {
        config_state_ = ConfigState::Initialized;
    } else if (train_transforms_set_ || val_transforms_set_) {
        config_state_ = ConfigState::TransformsSet;
    }
    // 如果都没设置，保持PreprocessorConfigured状态
}

void Preprocessor::build_dataset_paths(const std::string& dataset_path,
                                       std::string& train_path,
                                       std::string& val_path) {
    /**
     * 构建训练集和验证集路径
     *
     * 规则：
     * - MNIST/CIFAR DTS: mnist_train.dts / mnist_test.dts, cifar10_train.dts / cifar10_test.dts, cifar100_train.dts / cifar100_test.dts
     * - MNIST/CIFAR RAW: dataset_path/train 和 dataset_path/val
     * - ImageNet RAW: dataset_path/train 和 dataset_path/val
     * - ImageNet DTS: imagenet_train_lv{N}.dts / imagenet_val_lv{N}.dts
     */

    namespace fs = std::filesystem;

    if (is_dts_format()) {
        // DTS格式：需要完整的文件路径
        if (dataset_type_ == DatasetType::imagenet) {
            // ImageNet DTS: imagenet_train_lv{N}.dts / imagenet_val_lv{N}.dts
            train_path = dataset_path + "/imagenet_train_lv" + std::to_string(imagenet_compression_level_) + ".dts";
            val_path = dataset_path + "/imagenet_val_lv" + std::to_string(imagenet_compression_level_) + ".dts";
        } else if (dataset_type_ == DatasetType::mnist) {
            // MNIST DTS: mnist_train.dts / mnist_test.dts
            train_path = dataset_path + "/mnist_train.dts";
            val_path = dataset_path + "/mnist_test.dts";
        } else if (dataset_type_ == DatasetType::cifar_10) {
            // CIFAR-10 DTS: cifar10_train.dts / cifar10_test.dts
            train_path = dataset_path + "/cifar10_train.dts";
            val_path = dataset_path + "/cifar10_test.dts";
        } else if (dataset_type_ == DatasetType::cifar_100) {
            // CIFAR-100 DTS: cifar100_train.dts / cifar100_test.dts
            train_path = dataset_path + "/cifar100_train.dts";
            val_path = dataset_path + "/cifar100_test.dts";
        } else {
            TR_VALUE_ERROR("Unknown dataset type: " << static_cast<int>(dataset_type_));
        }
    } else {
        // RAW格式：dataset_path就是目录（train和val在同一路径）
        train_path = dataset_path;
        val_path = dataset_path;
    }

    LOG_DEBUG << "Built dataset paths: train=" << train_path << ", val=" << val_path;
}

bool Preprocessor::should_jpeg_decode() const {
    // 判断是否需要JPEG解码
    // ImageNet需要解码，MNIST/CIFAR不需要（已经是解压后的数据）
    return dataset_type_ == DatasetType::imagenet;
}

bool Preprocessor::should_apply_crop() const {
    // 判断是否需要RandomResizedCrop
    // ImageNet训练集需要，验证集不需要（或使用中心裁剪）
    // 目前简化：只在训练集且是ImageNet时需要
    return dataset_type_ == DatasetType::imagenet;
}

// =============================================================================
// 新配置方法：辅助函数（用于config_dataloader）
// =============================================================================

bool Preprocessor::is_dts_format() const {
    // 判断是否使用DTS格式
    // DTS格式：imagenet_compression_level_ >= 0
    // RAW格式：imagenet_compression_level_ < 0
    return imagenet_compression_level_ >= 0;
}

bool Preprocessor::is_imagenet() const {
    // 判断是否是ImageNet数据集
    return dataset_type_ == DatasetType::imagenet;
}

void Preprocessor::wait_all_transfer_stations_consumed() {
#ifndef TEST_WITHOUT_PW
// 测试的情况下没有PW和TransferStation，因此此方法应该跳过

    // 等待所有TransferStation的两个buffer都可写（深度学习引擎读取完毕）
    // 永久等待，直到深度学习引擎消费完所有数据
    // 检查间隔1ms，在快速机器上分秒必争
    LOG_DEBUG << "Waiting for all TransferStations to be consumed by deep learning engine...";

    const auto CHECK_INTERVAL = std::chrono::milliseconds(1);  // 1ms检查间隔

    while (true) {
        // 检查所有TransferStation的两个buffer是否都可写
        bool all_consumed = true;
        for (size_t i = 0; i < transfer_station_instances_.size(); ++i) {
            if (!transfer_station_instances_[i]->both_buffers_writeable()) {
                all_consumed = false;
                break;
            }
        }

        if (all_consumed) {
            LOG_DEBUG << "All TransferStations consumed, ready to end phase";
            return;
        }

        // 短暂休眠后再次检查
        std::this_thread::sleep_for(CHECK_INTERVAL);
    }

#endif  // TEST_WITHOUT_PW
}

// =============================================================================
// 新配置方法：config_dataset
// =============================================================================

void Preprocessor::config_dataset(const std::string& dataset_name,
                                  bool dts_format,
                                  int compression_level) {
    config_dataset(parse_dataset_type(dataset_name), dts_format, compression_level);
}

void Preprocessor::config_dataset(DatasetType dataset_type,
                                  bool dts_format,
                                  int compression_level) {
    // 检查状态
    check_state(ConfigState::Unconfigured, "config_dataset");

    // 禁用Deployment模式
    TR_CHECK(!is_deployment_mode_, ValueError,
             "Cannot call config_dataset() in deployment mode");
    dataset_type_ = dataset_type;
    if (dts_format) {
        imagenet_compression_level_ = compression_level;
    } else {
        imagenet_compression_level_ = -1;  // RAW格式
    }
    switch (dataset_type_) {
        case DatasetType::mnist:
            if (dts_format) {
                current_dataloader_ = &MnistLoaderDts::instance();
            } else {
                current_dataloader_ = &MnistLoaderRaw::instance();
            }
            default_input_width_ = 28;
            break;
        case DatasetType::cifar_10:
        case DatasetType::cifar_100:
            if (dts_format) {
                current_dataloader_ = &CifarLoaderDts::instance();
                CifarLoaderDts::instance().set_detected_num_classes(
                    dataset_type_ == DatasetType::cifar_10 ? 10 : 100
                );
            } else {
                current_dataloader_ = &CifarLoaderRaw::instance();
                CifarLoaderRaw::instance().set_detected_num_classes(
                    dataset_type_ == DatasetType::cifar_10 ? 10 : 100
                );
            }
            default_input_width_ = 32;
            break;

        case DatasetType::imagenet:
            if (dts_format) {
                current_dataloader_ = &ImageNetLoaderDts::instance();
            } else {
                current_dataloader_ = &ImageNetLoaderRaw::instance();
            }
            break;
    }
    config_state_ = ConfigState::DatasetSelected;
    GlobalRegistry::instance().set_dataset_type(dataset_type_);
    size_t num_train = current_dataloader_->num_train_samples();
    size_t num_val = current_dataloader_->num_val_samples();
    GlobalRegistry::instance().set_num_train_samples(num_train);
    GlobalRegistry::instance().set_num_val_samples(num_val);
}

// =============================================================================
// 新配置方法：config_dataloader
// =============================================================================

void Preprocessor::config_dataloader(const std::string& dataset_path,
                                     int num_load_workers,
                                     int num_preproc_workers,
                                     bool partial_mode,
                                     bool shuffle_train,
                                     bool download) {
    // 检查状态
    check_state(ConfigState::DatasetSelected, "config_dataloader");

    // 保存线程数到成员变量
    num_load_workers_ = num_load_workers;
    num_preproc_workers_ = num_preproc_workers;
    partial_mode_ = partial_mode;

    // 注册到全局注册表
    GlobalRegistry::instance().set_num_load_workers(num_load_workers);
    GlobalRegistry::instance().set_num_preproc_workers(num_preproc_workers);
    GlobalRegistry::instance().set_shuffle_train(shuffle_train);

    TR_CHECK(current_dataloader_ != nullptr, ValueError,
             "DataLoader not selected. Please call config_dataset() first");

    // 构建train_path和val_path
    std::string train_path, val_path;
    build_dataset_paths(dataset_path, train_path, val_path);

    // 下载/解压（如果需要）
    if (download) {
        try {
            current_dataloader_->download(dataset_path);

            // MNIST/CIFAR RAW需要解压，ImageNet RAW和所有DTS不需要
            if (!is_dts_format() && !is_imagenet()) {
                current_dataloader_->extract(dataset_path);
            }
        } catch (TRException& e) {
            TR_RETHROW(e, "Failed to download/extract dataset");
        }
    }

    // 配置Loader（一次性操作：读取文件头/summary.bin）
    try {
        current_dataloader_->configure(
            num_load_workers,
            num_preproc_workers,
            train_path,
            val_path,
            shuffle_train,  // train_shuffle
            false,          // val_shuffle（强制关闭）
            false,          // enable_train_logging（强制关闭）
            false           // enable_val_logging（强制关闭）
        );
    } catch (TRException& e) {
        TR_RETHROW(e, "Failed to configure DataLoader");
    }

    // 设置加载模式
    // MNIST/CIFAR：强制FULLY（Loader内部已处理）
    // ImageNet：根据partial_mode参数
    try {
        current_dataloader_->set_train_mode(
            is_imagenet() ? (partial_mode ? LoadMode::PARTIAL : LoadMode::FULLY)
                          : LoadMode::FULLY
        );
        current_dataloader_->set_val_mode(
            is_imagenet() ? (partial_mode ? LoadMode::PARTIAL : LoadMode::FULLY)
                          : LoadMode::FULLY
        );
    } catch (TRException& e) {
        TR_RETHROW(e, "Failed to set load mode");
    }

    // 更新状态
    config_state_ = ConfigState::DataLoaderConfigured;
}

// =============================================================================
// 新配置方法：config_preprocessor
// =============================================================================

void Preprocessor::config_preprocessor(int num_color_channels,
                                       int sdmp_factor,
                                       bool using_cpvs,
                                       bool pw_test_mode,
                                       int max_intermediate_resolution,
                                       bool drop_last) {
    // 检查状态
    check_state(ConfigState::DataLoaderConfigured, "config_preprocessor");

    // 从 GlobalRegistry 读取 batch_size 和 max_resolution
    int batch_size = GlobalRegistry::instance().get_local_batch_size();
    int max_final_resolution = GlobalRegistry::instance().max_sample_resolution();
    bool using_progressive_resolution = GlobalRegistry::instance().using_progressive_resolution();

    // 从 GlobalRegistry 读取 world_size（由 use_gpu()/use_cpu() 自动设置）
    world_size_ = GlobalRegistry::instance().world_size();

    // ==================== 内存使用警告 ====================
    // 经验公式：sdmp_factor应该小于计算机内存除以187GB
    // 如果大于这个值，即使仍能运转，也会因为数据加载器的缓存被驱逐而导致性能大幅下降
    //
    // 示例计算：
    // - 512 GB内存: 512/187 ≈ 2.74，合理的最大取值是2
    // - 960 GB内存: 960/187 ≈ 5.13，合理的最大取值是5
    //
    global_initial_seed_ = get_default_generator().seed();
    // 保存配置
    batch_size_ = batch_size;
    max_resolution_ = max_final_resolution;
    [[maybe_unused]] bool user_specified_max_intermediate = (max_intermediate_resolution > 0);
    max_intermediate_resolution_ = (max_intermediate_resolution > 0 ? max_intermediate_resolution : max_final_resolution);

    if (dataset_type_ == DatasetType::mnist) {
        num_color_channels_ = 1;
    }
    else if (dataset_type_ == DatasetType::cifar_10 || dataset_type_ == DatasetType::cifar_100 || dataset_type_ == DatasetType::imagenet) {
        num_color_channels_ = 3;
    }
    else {
        num_color_channels_ = num_color_channels;
    }

    sdmp_factor_ = sdmp_factor;
    using_cpvs_ = using_cpvs;
    using_progressive_resolution_ = using_progressive_resolution;

    // 设置PW测试模式（在配置阶段）
    pw_test_mode_ = pw_test_mode;

    // 注册到全局注册表（fixed变量，但不包括world_size）
    // world_size 已在 commit() 中验证并注册
    // 注意：batch_size 和 max_resolution 不应该在这里设置
    // - batch_size 应由用户通过 GLOBAL_SETTING.local_batch_size() 设置
    // - max_resolution 应由用户通过 GLOBAL_SETTING.train_resolution()/val_resolution() 设置
    // 这里只从 GlobalRegistry 读取，不再写回
    GlobalRegistry::instance().set_train_crop_output(max_final_resolution);
    GlobalRegistry::instance().set_train_resize_output(max_final_resolution);
    GlobalRegistry::instance().set_val_crop_output(max_final_resolution);
    GlobalRegistry::instance().set_val_resize_output(max_final_resolution);
    GlobalRegistry::instance().set_current_resolution_train(max_final_resolution);  // 初始值
    GlobalRegistry::instance().set_current_resolution_val(max_final_resolution);  // 初始值
    GlobalRegistry::instance().set_num_color_channels(num_color_channels_);  // 使用修正后的值
    GlobalRegistry::instance().set_sdmp_factor(sdmp_factor);
    GlobalRegistry::instance().set_using_cpvs(using_cpvs);
    GlobalRegistry::instance().set_using_drop_last(drop_last);

    // 计算单个样本大小（FusedNormalization输出的FP32/FP16字节数）
    size_t max_res = static_cast<size_t>(max_final_resolution);
    size_t num_ch = static_cast<size_t>(num_color_channels_);  // 使用修正后的成员变量
    bool using_amp = false;
    try { using_amp = GlobalRegistry::instance().using_amp(); } catch (const TRException&) {}
    if (using_amp) {
        sample_size_bytes_ = max_res * max_res * 4 * sizeof(uint16_t);
    } else {
        sample_size_bytes_ = max_res * max_res * num_ch * sizeof(float);
    }

    // 计算单个缓冲区大小
    buffer_size_bytes_ = batch_size * sample_size_bytes_;

    // 调用旧的configure方法（兼容性）
    Config config;
    config.num_workers = num_preproc_workers_;  // 使用保存的成员变量
    config.jpeg_decode = should_jpeg_decode();  // 根据数据集类型判断
    config.apply_crop = should_apply_crop();    // 根据数据集类型判断
    config.enable_logging = false;              // 默认关闭日志
    config.simulate_delay = false;
    configure(config);  // 调用旧方法

    // 更新状态
    config_state_ = ConfigState::PreprocessorConfigured;
}

// =============================================================================
// 新配置方法：set_train/val_transforms（模板实现）
// =============================================================================

namespace {
    /**
     * @brief 判断PO类型是否需要解码
     * @param op PO指针
     * @return true=需要解码, false=不需要解码
     */
    bool is_crop_or_resize_op(const PreprocessOperation* op) {
        return op->is_crop() || op->is_resize();
    }

    /**
     * @brief 判断是否是RandomHorizontalFlip
     * @note 使用虚方法is_random_horizontal_flip()，避免dynamic_cast开销
     */
    bool is_random_horizontal_flip(const PreprocessOperation* op) {
        return op->is_random_horizontal_flip();
    }

    /**
     * @brief 判断是否是DoNothing
     */
    bool is_do_nothing(const PreprocessOperation* op) {
        return dynamic_cast<const DoNothing*>(op) != nullptr;
    }
} // anonymous namespace

// ============================================================================
// 辅助函数实现（静态成员函数）
// ============================================================================

bool Preprocessor::is_do_nothing(const PreprocessOperation* op) {
    return dynamic_cast<const DoNothing*>(op) != nullptr;
}

bool Preprocessor::is_crop_or_resize_op(const PreprocessOperation* op) {
    return op->is_crop() || op->is_resize();
}

bool Preprocessor::is_random_horizontal_flip(const PreprocessOperation* op) {
    return dynamic_cast<const RandomHorizontalFlip*>(op) != nullptr;
}



// ============================================================================
// 模板函数set_train_transforms和set_val_transforms已移至头文件
// 这样可以支持任意数量的PreprocessOperation参数（1~N个）
// ============================================================================


// ============================================================================
// 显式实例化已移除
// 模板函数将按需实例化，支持任意数量的PreprocessOperation参数（1~N个）
// ============================================================================

void Preprocessor::set_deployment_transforms() {
    // TODO: 后续实现
    TR_NOT_IMPLEMENTED("set_deployment_transforms not implemented yet");
}

// =============================================================================
// 新配置方法：config_deployment_mode
// =============================================================================

void Preprocessor::config_deployment_mode(int batch_size,
                                         int num_color_channels) {
    // 从 GlobalRegistry 读取 max_resolution
    int max_resolution = GlobalRegistry::instance().max_sample_resolution();

    // 检查状态
    check_state(ConfigState::Unconfigured, "config_deployment_mode");

    // 启用Deployment模式
    is_deployment_mode_ = true;

    // 注册到GlobalRegistry
    GlobalRegistry::instance().set_is_deployment_mode(true);

    // 绑定SampleLoader（一次性绑定）
    current_dataloader_ = &SampleLoader::instance();

    // 配置SampleLoader
    auto& sample_loader = static_cast<SampleLoader&>(*current_dataloader_);
    sample_loader.configure_memory_pool(256);  // 默认256MB

    // 配置Preprocessor（Deployment模式强制参数）
    // 从 GlobalRegistry 读取 world_size（Deployment模式应该已通过use_cpu()设置为1）
    world_size_ = GlobalRegistry::instance().world_size();
    batch_size_ = batch_size;
    max_resolution_ = max_resolution;
    num_color_channels_ = num_color_channels;
    size_t max_res = static_cast<size_t>(max_resolution);
    bool using_amp = false;
    try { using_amp = GlobalRegistry::instance().using_amp(); } catch (const TRException&) {}
    if (using_amp) {
        sample_size_bytes_ = max_res * max_res * 4 * sizeof(uint16_t);
    } else {
        sample_size_bytes_ = max_res * max_res * num_color_channels * sizeof(float);
    }
    buffer_size_bytes_ = batch_size * sample_size_bytes_;

    // Deployment模式：单线程、单worker
    num_load_workers_ = 1;
    num_preproc_workers_ = 1;

    // 调用旧的configure方法（兼容性）
    Config config;
    config.num_workers = 1;  // 单线程
    config.jpeg_decode = true;   // 需要解码JPEG
    config.apply_crop = false;   // 不需要随机裁剪
    config.enable_logging = false;
    configure(config);  // 调用旧方法

    // 更新状态
    config_state_ = ConfigState::PreprocessorConfigured;
}

// =============================================================================
// 高级封装方法：multi_thread_init(), train()和val()
// =============================================================================

void Preprocessor::multi_thread_init() {
#ifndef TEST_WITHOUT_PW
// 测试模式不需要PW和TransferStation，因此此方法应该跳过

    // 创建临时线程池（用于绑核等初始化操作）
    std::vector<std::thread> init_threads;
    init_threads.reserve(num_preproc_workers_);

    // TransferStation创建的互斥锁
    static std::mutex eb_create_mutex;

    // 展开所有worker线程
    for (int i = 0; i < num_preproc_workers_; ++i) {
        init_threads.emplace_back([this, i]() {
            // ==================== CPU绑核（GPU_CLOUD + auto_binding）====================
#if defined(TR_SCENE_GPU_CLOUD)
            if (auto_cpu_binding_ && !selected_gpu_ids_.empty()) {
                bind_worker_to_cpu(i);
            }
#endif

            // ==================== TransferStation创建（仅线程0~world_size-1）====================
            if (i < world_size_) {
                std::lock_guard<std::mutex> lock(eb_create_mutex);

                // 确保transfer_station_instances_有足够空间
                if (transfer_station_instances_.size() < static_cast<size_t>(world_size_)) {
                    transfer_station_instances_.resize(world_size_);
                }

                // 如果当前TransferStation还没创建，就创建它
                if (!transfer_station_instances_[i]) {

                    // 计算每个Engine的worker数
                    int num_workers_per_engine = num_preproc_workers_ / world_size_;

                    // 创建TransferStation实例（仅正常模式）
                    transfer_station_instances_[i] = std::unique_ptr<TransferStation>(
                        new TransferStation()  // 构造函数无参数，始终为正常模式
                    );

                    // 检查sample_size_bytes_是否在合理范围内
                    // FP32: 224*224*3*4 = 602112字节, FP16: 224*224*4*2 = 401408字节
                    TR_CHECK(sample_size_bytes_ > 0 && sample_size_bytes_ < 200ULL * 1024ULL * 1024ULL, ValueError,
                             "sample_size_bytes_ is invalid: " << sample_size_bytes_
                             << " (expected ~600KB for FP32 or ~400KB for FP16 224x224x3)");

                    // ✅ 注册到GlobalRegistry（在锁保护下，确保第i号EB注册到第i号位置）
                    {
                        auto& registry = GlobalRegistry::instance();
                        registry.set_transfer_station_ptr(i, transfer_station_instances_[i].get());
                        LOG_DEBUG << "TransferStation[" << i << "] registered to GlobalRegistry at index " << i;
                    }

                    transfer_station_instances_[i]->configure(
                        batch_size_,                      // local_batch_size
                        sample_size_bytes_,               // max_train_sample_bytes
                        sample_size_bytes_,               // max_val_sample_bytes
                        num_workers_per_engine,           // num_workers_per_engine
                        i                                 // engine_id（线程ID）
                    );

                    // 更新phase（从GlobalRegistry自动获取配置）
                    transfer_station_instances_[i]->reset_and_update();
                }
            }

            LOG_DEBUG << "Initialization thread " << i << " completed";
        });

        }

    // Join所有线程（必须！确保NUMA架构下的正确初始化）
    for (auto& t : init_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // 重置TransferStation同步屏障（multi_thread_init中的reset_and_update不计入barrier）
    engine_reset_barrier_.store(0, std::memory_order_release);

#endif  // #ifndef TEST_WITHOUT_PW

    LOG_INFO << "Preprocessor initialized.";
    // 设置初始化完成标志
    multi_thread_inited_ = true;
}

// -----------------------------------------------------------------------------

void Preprocessor::ensure_inited() {
    if (!workshop_size_calculated_) {
        calculate_workshop_sizes(max_intermediate_resolution_);
        workshop_size_calculated_ = true;
    }

    // 多线程初始化（如果还未初始化）
    if (!multi_thread_inited_) {
        multi_thread_init();
    }
}

void Preprocessor::train() {
    GlobalRegistry& gr_instance = GlobalRegistry::instance();
    // 检查状态
    check_state(ConfigState::Initialized, "train");

    // 检查Deployment模式
    TR_CHECK(!is_deployment_mode_, ValueError,
             "train() is not available in deployment mode");

    ensure_inited();

    if (sdmp_factor_ == 1 || train_phase_id_ % sdmp_factor_ == 0) {  // Busy train phase行为
        is_lazy_phase_ = false;

        // 更新运行时参数
        pw_param_.is_train = true;
        pw_param_.is_lazy_phase = false;
        pw_param_.active_s_region_idx = -1;  // Busy phase不从S区读取
        pw_param_.phase_id = train_phase_id_;
        pw_param_.current_train_resolution = gr_instance.current_resolution_train();
        pw_param_.current_val_resolution = gr_instance.current_resolution_val();

        // 开始训练阶段
        gr_instance.begin_train();

        // 训练一个epoch
        current_dataloader_->begin_epoch(train_iteration_id_, true);  // train_iteration_id_标志DataLoader读取训练集的次数，不一定等于epoch数或phase数
        this->run(*current_dataloader_);  // 原有run方法，包含完整预处理
        current_dataloader_->end_epoch();

        // 等待深度学习引擎读取完毕所有TransferStation的数据
        wait_all_transfer_stations_consumed();

        // 结束训练阶段
        gr_instance.end_train();

        // 输出时间统计（仅在测试模式下输出简单信息，详细统计由test程序负责）
        // 注意：test程序会计算更详细的统计信息（包括吞吐量），所以这里不再输出

        train_iteration_id_++;  // train_iteration_id_标志DataLoader读取训练集的次数，不一定等于epoch数或phase数
    }
    else {  // Lazy train phase行为
        is_lazy_phase_ = true;
        // 注意！这个时候是不需要递增train_iteration_id_的！因为并没有调用DataLoader！

        // 更新运行时参数
        pw_param_.is_train = true;
        pw_param_.is_lazy_phase = true;
        pw_param_.active_s_region_idx = (train_phase_id_ % sdmp_factor_) - 1;
        pw_param_.phase_id = train_phase_id_;
        // Lazy Phase不可以更新分辨率

        // ==================== 重置样本计数器 ====================
        total_samples_.store(0, std::memory_order_relaxed);

        // 开始训练阶段
        gr_instance.begin_train();

        // ==================== 展开Lazy Train Phase ====================
        // 参照worker_func_persistent的结构，但不涉及DataLoader

        // 【修复竞态条件】步骤1：先完成所有TransferStation的reset_and_update()
        for (int i = 0; i < world_size_; ++i) {
            int engine_id = i;
            if (transfer_station_instances_[engine_id]) {
                transfer_station_instances_[engine_id]->reset_and_update();
            }
        }

        // 【修复竞态条件】步骤2：然后所有worker再并行执行work_lazy()
        std::vector<std::thread> lazy_threads;
        for (int i = 0; i < num_preproc_workers_; ++i) {
            lazy_threads.emplace_back([this, i]() {
                // ==================== Step 1: CPU绑核（GPU_CLOUD + auto_binding）====================
#if defined(TR_SCENE_GPU_CLOUD)
                if (auto_cpu_binding_ && !selected_gpu_ids_.empty()) {
                    bind_worker_to_cpu(i);
                }
#endif

                // ==================== Step 2: PW更新参数 ====================
                if (pw_instances_[i]) {
                    pw_instances_[i]->update_parameters();
                }

                // ==================== Step 3: 调用PW的work_lazy() ====================
                // 注意：TransferStation已经在所有worker线程启动前完成reset_and_update()
                if (pw_instances_[i]) {
                    pw_instances_[i]->work_lazy();
                }

                LOG_DEBUG << "[Lazy Train Worker " << i << "] completed";
            });
        }

        // 等待所有lazy线程完成
        for (auto& t : lazy_threads) {
            if (t.joinable()) {
                t.join();
            }
        }

        // ==================== 设置正确的样本数 ====================
        // Lazy Phase从S区复制数据，没有通过DataLoader累加total_samples_
        // 所以需要手动设置为训练集样本总数
        total_samples_.store(current_dataloader_->num_train_samples(), std::memory_order_relaxed);

        // 等待深度学习引擎读取完毕所有TransferStation的数据
        wait_all_transfer_stations_consumed();

        // 结束训练阶段
        gr_instance.end_train();
        // 注意！这个时候是不需要递增train_iteration_id_的！因为并没有调用DataLoader！
    }

    train_phase_id_++;  // 只要调用了train()就必定递增
}

void Preprocessor::val() {
    GlobalRegistry& gr_instance = GlobalRegistry::instance();
    // 检查状态
    check_state(ConfigState::Initialized, "val");

    ensure_inited();

    if (!using_cpvs_ || val_phase_id_ == 0) {  // Busy val phase行为
        is_lazy_phase_ = false;

        // 更新运行时参数
        pw_param_.is_train = false;
        pw_param_.is_lazy_phase = false;
        pw_param_.active_s_region_idx = -1;  // Val phase不从S区读取
        pw_param_.phase_id = val_phase_id_;
        pw_param_.current_train_resolution = gr_instance.current_resolution_train();
        pw_param_.current_val_resolution = gr_instance.current_resolution_val();

        // 开始验证阶段
        gr_instance.begin_val();

        // 验证一个epoch（不递增iteration_id）
        if (is_deployment_mode_) {
            // Deployment模式：使用SampleLoader
            current_dataloader_->begin_epoch(0, false);
            this->run(*current_dataloader_);
            current_dataloader_->end_epoch();
        } else {
            // 普通模式：使用验证集
            current_dataloader_->begin_epoch(val_iteration_id_, false);  // val_iteration_id_标志DataLoader读取验证集的次数，不一定等于epoch数或phase数
            this->run(*current_dataloader_);
            current_dataloader_->end_epoch();
        }

        // 等待深度学习引擎读取完毕所有TransferStation的数据
        wait_all_transfer_stations_consumed();

        // 结束验证阶段
        gr_instance.end_val();

        // 输出时间统计（仅在测试模式下输出简单信息，详细统计由test程序负责）
        // 注意：test程序会计算更详细的统计信息（包括吞吐量），所以这里不再输出



        // 递增DataLoader调用计数
        val_iteration_id_++;  // val_iteration_id_标志DataLoader读取验证集的次数，不一定等于epoch数或phase数
    }
    else {  // Lazy val phase行为
        is_lazy_phase_ = true;
        // 注意！这个阶段是不需要递增iteration_id_的！因为并没有调用DataLoader！

        // 更新运行时参数
        pw_param_.is_train = false;
        pw_param_.is_lazy_phase = true;
        pw_param_.active_s_region_idx = -1;  // Val phase不从S区读取
        pw_param_.phase_id = val_phase_id_;
        // Lazy Phase不可以更新分辨率

        // ==================== 重置样本计数器 ====================
        total_samples_.store(0, std::memory_order_relaxed);

        // 开始验证阶段
        gr_instance.begin_val();

        // ==================== 展开Lazy Val Phase ====================
        // 参照worker_func_persistent的结构，但不涉及DataLoader

        // 【修复竞态条件】步骤1：先完成所有TransferStation的reset_and_update()
        for (int i = 0; i < world_size_; ++i) {
            int engine_id = i;
            if (transfer_station_instances_[engine_id]) {
                transfer_station_instances_[engine_id]->reset_and_update();
            }
        }

        // 【修复竞态条件】步骤2：然后所有worker再并行执行work_lazy()
        std::vector<std::thread> lazy_threads;
        for (int i = 0; i < num_preproc_workers_; ++i) {
            lazy_threads.emplace_back([this, i]() {
                // ==================== Step 1: CPU绑核（GPU_CLOUD + auto_binding）====================
#if defined(TR_SCENE_GPU_CLOUD)
                if (auto_cpu_binding_ && !selected_gpu_ids_.empty()) {
                    bind_worker_to_cpu(i);
                }
#endif

                // ==================== Step 2: PW更新参数 ====================
                if (pw_instances_[i]) {
                    pw_instances_[i]->update_parameters();
                }

                // ==================== Step 3: 调用PW的work_lazy() ====================
                // 注意：TransferStation已经在所有worker线程启动前完成reset_and_update()
                if (pw_instances_[i]) {
                    pw_instances_[i]->work_lazy();
                }

                LOG_DEBUG << "[Lazy Val Worker " << i << "] completed";
            });
        }

        // 等待所有lazy线程完成
        for (auto& t : lazy_threads) {
            if (t.joinable()) {
                t.join();
            }
        }

        // ==================== 设置正确的样本数 ====================
        // Lazy Phase从C区复制数据，没有通过DataLoader累加total_samples_
        // 所以需要手动设置为验证集样本总数
        total_samples_.store(current_dataloader_->num_val_samples(), std::memory_order_relaxed);

        // 等待深度学习引擎读取完毕所有TransferStation的数据
        wait_all_transfer_stations_consumed();

        // 结束验证阶段
        gr_instance.end_val();
        // 注意！这个阶段是不需要递增iteration_id_的！因为并没有调用DataLoader！
    }

    val_phase_id_++;  // 只要调用了val()就必定递增
}

// =============================================================================
// 测试方法：test_dataloader()和warmup()
// =============================================================================

namespace {
    /**
     * @brief 获取数据集大小（MB单位）
     * @param dataset_name 数据集名称
     * @param format 格式（"raw"或"dts"）
     * @param is_train true=训练集, false=验证集
     * @param compression_level DTS压缩级别（仅ImageNet有效，0-3）
     * @return 数据集大小（MB）
     */
    double get_dataset_size_mb(const std::string& dataset_name,
                                const std::string& format,
                                bool is_train,
                                int compression_level = 0) {
        // ImageNet
        if (dataset_name == "imagenet") {
            if (format == "raw") {
                return is_train ? 140102.280 : 6395.874;
            } else {  // DTS
                if (compression_level == 0) {
                    return is_train ? 140288.000 : 6416.000;
                } else if (compression_level == 1) {
                    return is_train ? 65680.000 : 2816.000;
                } else if (compression_level == 2) {
                    return is_train ? 65744.000 : 2832.000;
                } else if (compression_level == 3) {
                    return is_train ? 45632.000 : 1952.000;
                }
            }
        }
        // CIFAR-10（支持"cifar10"和"cifar-10"两种格式）
        else if (dataset_name == "cifar10" || dataset_name == "cifar-10") {
            if (format == "raw") {
                return is_train ? 146.532 : 29.3064;
            } else {  // DTS
                return is_train ? 146.532 : 29.306;
            }
        }
        // CIFAR-100（支持"cifar100"和"cifar-100"两种格式）
        else if (dataset_name == "cifar100" || dataset_name == "cifar-100") {
            if (format == "raw") {
                return is_train ? 146.532 : 29.3064;
            } else {  // DTS
                return is_train ? 146.532 : 29.306;
            }
        }
        // MNIST
        else if (dataset_name == "mnist") {
            if (format == "raw") {
                return is_train ? 44.9181 : 7.48634;
            } else {  // DTS
                return is_train ? 44.9181 : 7.48634;
            }
        }

        // 未知数据集
        return 0.0;
    }
}

void Preprocessor::test_dataloader() {
#ifdef TEST_WITHOUT_PW
    TR_CHECK(!is_deployment_mode_, ValueError,
             "test_dataloader() is not available in deployment mode");

    // 获取当前数据集信息
    const char* ds_name = current_dataloader_->dataset_name();
    std::string dataset_name = ds_name;

    // 转换为小写用于比较
    std::string dataset_name_lower = dataset_name;
    std::transform(dataset_name_lower.begin(), dataset_name_lower.end(),
                   dataset_name_lower.begin(), ::tolower);

    // 判断格式和压缩级别
    bool is_dts = (dynamic_cast<ImageNetLoaderDts*>(current_dataloader_) != nullptr) ||
                  (dynamic_cast<MnistLoaderDts*>(current_dataloader_) != nullptr) ||
                  (dynamic_cast<CifarLoaderDts*>(current_dataloader_) != nullptr);

    // 直接使用Preprocessor保存的compression_level（在config_dataset时设置）
    int compression_level = imagenet_compression_level_;

    std::string format = is_dts ? "dts" : "raw";

    // 测试训练集
    if (!suppress_info_logs_) {
        std::cout << "\n========================================\n"
                  << "Training Set Test\n"
                  << "========================================\n";
    }

    auto start = std::chrono::high_resolution_clock::now();
    current_dataloader_->begin_epoch(0, true);
    this->run_fast_without_processing();
    current_dataloader_->end_epoch();
    auto end = std::chrono::high_resolution_clock::now();

    // 计算用时
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double train_seconds = duration.count() / 1000.0;
    size_t train_samples = total_samples_.load();

    // 获取数据集大小并计算吞吐量
    double train_size_mb = get_dataset_size_mb(dataset_name_lower, format, true, compression_level);
    double train_size_gb = train_size_mb / 1024.0;
    double train_gb_per_sec = train_size_gb / train_seconds;

    // 验证完整性
    size_t expected_train_samples = current_dataloader_->num_train_samples();
    bool train_integrity_passed = (train_samples == expected_train_samples);

    // 打印结果
    if (!suppress_info_logs_) {
        std::cout << "Load time:        " << std::fixed << std::setprecision(3) << train_seconds << " s\n"
                  << "Total samples:    " << train_samples << "\n"
                  << "Expected samples: " << expected_train_samples << "\n"
                  << "Throughput:       " << std::fixed << std::setprecision(3)
                  << train_gb_per_sec << " GB/s\n"
                  << "Integrity:        " << (train_integrity_passed ? "PASSED" : "FAILED") << "\n";
    }

    // 测试验证集
    if (!suppress_info_logs_) {
        std::cout << "\n========================================\n"
                  << "Validation Set Test\n"
                  << "========================================\n";
    }

    start = std::chrono::high_resolution_clock::now();
    current_dataloader_->begin_epoch(0, false);
    this->run_fast_without_processing();
    current_dataloader_->end_epoch();
    end = std::chrono::high_resolution_clock::now();

    // 计算用时
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double val_seconds = duration.count() / 1000.0;
    size_t val_samples = total_samples_.load();

    // 获取数据集大小并计算吞吐量
    double val_size_mb = get_dataset_size_mb(dataset_name_lower, format, false, compression_level);
    double val_size_gb = val_size_mb / 1024.0;
    double val_gb_per_sec = val_size_gb / val_seconds;

    // 验证完整性
    size_t expected_val_samples = current_dataloader_->num_val_samples();
    bool val_integrity_passed = (val_samples == expected_val_samples);

    // 打印结果
    if (!suppress_info_logs_) {
        std::cout << "Load time:        " << std::fixed << std::setprecision(3) << val_seconds << " s\n"
                  << "Total samples:    " << val_samples << "\n"
                  << "Expected samples: " << expected_val_samples << "\n"
                  << "Throughput:       " << std::fixed << std::setprecision(3)
                  << val_gb_per_sec << " GB/s\n"
                  << "Integrity:        " << (val_integrity_passed ? "PASSED" : "FAILED") << "\n";
    }

    // 【关键】测试完成后重置DataLoader状态，释放FULLY模式内存
    // 目的：将DataLoader重置到"刚刚加载完文件头"的状态
    //       这样下次调用时才会重新加载数据
    current_dataloader_->reset_after_warmup();
#else
        TR_THROW(ValueError, "test_dataloader() can only be used in TEST_WITHOUT_PW mode.");
#endif  // #ifdef TEST_WITHOUT_PW
}

// =============================================================================
// 快速运行（不执行预处理）
// =============================================================================

void Preprocessor::run_fast_without_processing() {
#ifdef TEST_WITHOUT_PW
    /**
     * 快速运行（不执行预处理）
     *
     * 用途：
     * - test_dataloader：测试加载速度和完整性
     * - warmup：预热缓存
     *
     * 与run()的区别：
     * - run(): 解码JPEG + 随机裁剪 + 数据增强 + 归一化
     * - run_fast(): 直接丢弃样本，只计数
     */

    auto& loader = *current_dataloader_;

    // 重置统计
    total_samples_.store(0, std::memory_order_relaxed);
    std::fill(worker_sample_counts_.begin(), worker_sample_counts_.end(), 0);

    // 设置快速模式标志
    fast_mode_ = true;

    // 启动持久线程池（只执行一次）
    start_worker_pool(loader);

    // 主循环：处理所有buffers
    do {
        // 通知worker开始新buffer
        notify_workers_new_buffer();

        // 等待worker完成当前buffer
        bool fully_second = (!partial_mode_) && ((pw_param_.is_train? train_phase_id_: val_phase_id_) > 0);
        if (fully_second) {
            // 这里解释一下，因为FULLY MODE的第二个以后的phase都是从已有的存储中读取，没有buffer的概念，直到处理完整个数据集才结束，所以不应该设置超时
            wait_workers_complete_buffer(true);  // 永久等待
        }
        else {  // PARTIAL MODE或第一个phase（必定加载）
            wait_workers_complete_buffer(false);  // 20s超时
        }

        // 触发DataLoader加载下一个buffer
        if (loader.has_more_buffers()) {
            loader.load_next_buffer();
        } else {
            break;
        }
    } while (true);

    // 停止线程池
    stop_worker_pool();

    // 清除快速模式标志
    fast_mode_ = false;
#else
        TR_THROW(ValueError, "run_fast_without_processing() can only be used in TEST_WITHOUT_PW mode.");
#endif  // #ifdef TEST_WITHOUT_PW
}

// =============================================================================
// Workshop大小计算（按照PW2.md规范）
// =============================================================================

void Preprocessor::calculate_workshop_sizes([[maybe_unused]] int ref_resolution_for_ab) {
#ifndef TEST_WITHOUT_PW
// 测试模式不需要进行workshop大小计算，因为没有PW

    // ==================== 参数有效性检查 ====================
    TR_CHECK(max_resolution_ > 0 && max_resolution_ <= 10000, ValueError,
             "max_resolution_ is invalid: " << max_resolution_);
    TR_CHECK(num_color_channels_ > 0 && num_color_channels_ <= 10, ValueError,
             "num_color_channels_ is invalid: " << num_color_channels_);

    LOG_DEBUG << "calculate_workshop_sizes: max_resolution_=" << max_resolution_
              << ", num_color_channels_=" << num_color_channels_;

    // ==================== 辅助对齐函数 ====================
    auto align_64 = [](size_t s) -> size_t {
        return (s + 63) & ~size_t(63);
    };
    // 4KB页对齐（用于Workshop所有区域）
    auto align_4k = [](size_t s) -> size_t {
        return (s + 4095) & ~size_t(4095);
    };

    // ==================== 1. 计算D区大小 ====================
    // 根据list.md的实际测量数据（2026-02-22统计）
    if (is_imagenet()) {
        // ImageNet需要解码，D区大小取决于压缩级别
        switch (imagenet_compression_level_) {
            case 0:  // RAW或LV0
                // ImageNet DTS LV0解码后最大字节数为95073792 Bytes，对齐到2MB大页后是96468992 Bytes (92MB)
                workshop_region_d_size_ = 96468992;
                break;
            case 1:  // LV1
                // ImageNet DTS LV1解码后最大字节数为3046400 Bytes，对齐到2MB大页后是4194304 Bytes (4MB)
                workshop_region_d_size_ = 4194304;
                break;
            case 2:  // LV2
                // ImageNet DTS LV2解码后最大字节数为742400 Bytes，对齐到2MB大页后是2097152 Bytes (2MB)
                workshop_region_d_size_ = 2097152;
                break;
            case 3:  // LV3
                // ImageNet DTS LV3解码后最大字节数为742400 Bytes，对齐到2MB大页后是2097152 Bytes (2MB)
                workshop_region_d_size_ = 2097152;
                break;
            default:
                // 默认使用LV0的大小（最安全）
                workshop_region_d_size_ = 96468992;
                break;
        }
    } else {
        // MNIST/CIFAR等不需要JPEG解码，D区为0
        workshop_region_d_size_ = 0;
    }

    // ==================== 2. 计算AB区大小 ====================
    // AB区大小需要对齐到4KB页边界（NUMA优化）
    //
    // 关键设计说明：
    // 1. AB区只存储中间PO（FusedNormalization之前）的uint8_t输出
    // 2. FusedNormalization必定是PO链的最后一个操作
    // 3. FusedNormalization的输出是FP32/FP16，存储在S区/C区/TransformStation，不在AB区
    // 4. 因此ref_resolution_for_ab = max_intermediate_resolution_不包含FusedNormalization的输出尺寸
    // 5. max_intermediate_resolution_在set_train_transforms/set_val_transforms中计算时已排除FusedNormalization
    //
    // 按照EXP2.md第326-327行的公式
    // 注意：使用static_cast<size_t>确保乘法不会产生符号扩展问题
    size_t num_ch = static_cast<size_t>(num_color_channels_);
    size_t max_train_output = static_cast<size_t>(ref_resolution_for_ab) * ref_resolution_for_ab * num_ch;
    size_t max_val_output = max_train_output;
    size_t stride = ((ref_resolution_for_ab * num_ch + 63) / 64) * 64;  // 64字节对齐
    size_t ab_size = stride * ref_resolution_for_ab;
    workshop_region_ab_size_ = align_4k(ab_size);  // 对齐到4KB页边界

    LOG_DEBUG << "calculate_workshop_sizes: max_train_output=" << max_train_output
              << ", max_val_output=" << max_val_output
              << ", stride=" << stride
              << ", ab_size=" << ab_size
              << ", workshop_region_ab_size_=" << workshop_region_ab_size_;

    // ==================== 计算S区和C区共用大小（与FusedNormalization输出完全一致）====================
    // 这个值用于S区和C区，train和val共享
    //
    // 关键设计说明：
    // 1. 使用GlobalRegistry::max_sample_resolution()而非max_resolution_
    //    - max_resolution_是AB区的uint8_t数据大小（已废弃）
    //    - max_sample_resolution()是FusedNormalization输出的真正最终分辨率
    //    - 它会自动对比train_sample_resolution和val_sample_resolution并返回最大值
    //    - S区/C区存储的是FusedNormalization的输出，必须使用最终的分辨率
    //
    // 2. 考虑FusedNormalization的数据类型膨胀和通道填充
    //    - FP32模式：max_res × max_res × num_channels × sizeof(float)
    //    - FP16/AMP模式：max_res × max_res × 4 × sizeof(uint16_t)（固定4通道padding）
    //    - 这与FusedNormalization::calculate_stride()的计算完全一致
    //
    // 3. 紧凑布局，无需per-sample padding
    //    - FusedNormalization的calculate_stride()已保证紧凑布局
    //    - output_stride_ == compact_output_stride_，不存在SIMD对齐需求
    //    - FP32/FP16的stride在常见尺寸下天然64字节对齐（如224×12=2688=64×42）
    //    - 额外的对齐操作会造成S区/C区容量计算与实际输出大小不匹配

    auto& registry = GlobalRegistry::instance();
    int max_res = registry.max_sample_resolution();  // 自动对比train和val分辨率的最大值
    bool using_amp = registry.using_amp();

    size_t max_output_per_sample;
    if (using_amp) {
        // FP16 AMP模式：固定4通道，每个元素2字节
        max_output_per_sample = static_cast<size_t>(max_res) * max_res * 4 * sizeof(uint16_t);
    } else {
        // FP32模式：原始通道数，每个元素4字节（复用前面定义的num_ch）
        max_output_per_sample = static_cast<size_t>(max_res) * max_res * num_ch * sizeof(float);
    }

    registry.set_aligned_max_output_size(max_output_per_sample);

    LOG_DEBUG << "Registered aligned_max_output_size to GlobalRegistry: " << max_output_per_sample << " bytes"
              << " (max_res=" << max_res << ", channels=" << num_ch << ", AMP=" << using_amp << ")";

    // ==================== 3. 计算T区大小 ====================
    // T区默认为0，需要根据PO的require_temp()决定
    // 目前我们假设不需要T区（TODO: 检查PO是否require_temp）
    workshop_region_t_size_ = 0;
    if (workshop_region_t_size_ > 0) {
        workshop_region_t_size_ = align_4k(workshop_region_t_size_);  // 对齐到4KB页边界
    }

    // ==================== 4. 计算S区容量和大小 ====================
    // 按照EXP2.md第366-368行的公式
    // S区需要对齐到4KB页边界（NUMA优化）

    // 始终计算max_s_samples_（不管是否启用SDMP）
    TR_CHECK(current_dataloader_ != nullptr, ValueError,
             "calculate_workshop_sizes: current_dataloader_ is null");
    size_t num_train = current_dataloader_->num_train_samples();
    size_t num_train_per_engine = (num_train + world_size_ - 1) / world_size_;
    int num_workers_per_engine = num_preproc_workers_ / world_size_;
    max_s_samples_ = static_cast<int>((num_train_per_engine + num_workers_per_engine - 1) / num_workers_per_engine);

    // ==================== 向GlobalRegistry注册S区原始索引向量 ====================
    // 这个向量用于S区洗牌，所有PW共享同一个原始顺序定义
    std::vector<int> s_original_indices(max_s_samples_);
    for (int i = 0; i < max_s_samples_; ++i) {
        s_original_indices[i] = i;  // [0, 1, 2, ..., max_s_samples_-1]
    }
    GlobalRegistry::instance().set_fixed_s_original_indices(s_original_indices);

    LOG_DEBUG << "Registered fixed_s_original_indices_ to GlobalRegistry: size=" << max_s_samples_;

    // 只有启用SDMP时才分配S区内存
    if (sdmp_factor_ > 1) {
        // S区大小计算：使用FusedNormalization输出的紧凑字节数（无per-sample padding）
        workshop_region_s_size_ = max_s_samples_ * max_output_per_sample;
        workshop_region_s_size_ = align_4k(workshop_region_s_size_);  // 4KB页对齐
        workshop_num_region_s_ = sdmp_factor_ - 1;  // S区个数 = sdmp_factor - 1

        LOG_DEBUG << "S区计算: num_train=" << num_train
                  << ", per_engine=" << num_train_per_engine
                  << ", max_s_samples=" << max_s_samples_
                  << ", sample_bytes=" << max_output_per_sample << " bytes"
                  << ", s_size=" << (workshop_region_s_size_ / (1024.0*1024.0)) << " MB";
    } else {
        workshop_region_s_size_ = 0;
        workshop_num_region_s_ = 0;
    }

    // ==================== 5. 计算C区容量和大小 ====================
    // 按照EXP2.md第395-396行的公式
    // C区使用FusedNormalization输出的紧凑字节数（无per-sample padding），然后对整个C区进行4KB页对齐

    // 始终计算max_c_samples_（不管是否启用CPVS）
    size_t num_val = current_dataloader_->num_val_samples();
    size_t num_val_per_engine = (num_val + world_size_ - 1) / world_size_;
    max_c_samples_ = static_cast<int>((num_val_per_engine + num_workers_per_engine - 1) / num_workers_per_engine);

    // 只有启用CPVS时才分配C区内存
    if (using_cpvs_) {
        // C区大小计算：使用FusedNormalization输出的紧凑字节数（无per-sample padding）
        // 4KB页对齐确保NUMA性能
        workshop_region_c_size_ = max_c_samples_ * max_output_per_sample;
        workshop_region_c_size_ = align_4k(workshop_region_c_size_);  // 4KB页对齐

        LOG_DEBUG << "C区计算: num_val=" << num_val
                  << ", per_engine=" << num_val_per_engine
                  << ", max_c_samples=" << max_c_samples_
                  << ", sample_bytes=" << max_output_per_sample << " bytes"
                  << ", c_size=" << (workshop_region_c_size_ / (1024.0*1024.0)) << " MB";
    } else {
        workshop_region_c_size_ = 0;
    }
#endif  // #ifndef TEST_WITHOUT_PW
}

// =============================================================================
// PW管理方法（V4.1）
// =============================================================================

void Preprocessor::create_pw_instances([[maybe_unused]] bool is_train) {
#ifndef TEST_WITHOUT_PW
// 测试模式跳过此方法，因为不需要PW

    if (!workshop_size_calculated_) {
        calculate_workshop_sizes(max_intermediate_resolution_);
        workshop_size_calculated_ = true;
    }

    // 检查是否已经创建
    if (!pw_instances_.empty()) {
        return;  // 已创建，无需重复创建
    }

    // 确保pw_instances_有足够的空间
    pw_instances_.resize(num_preproc_workers_);

    // 计算每个worker对应的engine信息
    int num_workers_per_engine = num_preproc_workers_ / world_size_;

    // 为每个worker创建PW
    for (int i = 0; i < num_preproc_workers_; ++i) {
        // 构建PW配置
        PreprocessWorker::Config pw_config;
        pw_config.worker_id = i;
        pw_config.engine_id = i % world_size_;
        pw_config.pid_in_engine = i / world_size_;

        pw_config.global_initial_seed = global_initial_seed_;

        // 计算workshop大小
        pw_config.region_d_size = workshop_region_d_size_;
        pw_config.region_ab_size = workshop_region_ab_size_;
        pw_config.region_t_size = workshop_region_t_size_;
        pw_config.region_s_size = workshop_region_s_size_;
        pw_config.region_c_size = workshop_region_c_size_;
        pw_config.num_region_s = workshop_num_region_s_;

        // 从Preprocessor获取参数
        pw_config.local_batch_size = batch_size_;
        pw_config.world_size = world_size_;
        pw_config.sdmp_factor = sdmp_factor_;
        pw_config.using_cpvs = using_cpvs_;
        pw_config.using_progressive_resolution = using_progressive_resolution_;
        pw_config.num_workers_per_engine = num_workers_per_engine;

        // 数据集信息
        pw_config.dataset_type = dataset_type_;
        pw_config.num_color_channels = num_color_channels_;
        pw_config.raw_image_width = default_input_width_;   // ImageNet不需要
        pw_config.raw_image_height = default_input_width_;  // ImageNet不需要

        // 测试模式
        pw_config.test_mode = pw_test_mode_;

        // SDMP/CPVS缓存容量（样本数）
        pw_config.max_s_samples = max_s_samples_;
        pw_config.max_c_samples = max_c_samples_;

        // TransferStation指针（V3.14.0 - 跨步分配：worker_id % world_size）
        // 对应关系：PW ID % world_size = TransferStation ID
        if (!transfer_station_instances_.empty() && pw_config.engine_id < static_cast<int>(transfer_station_instances_.size())) {
            pw_config.transfer_station = transfer_station_instances_[pw_config.engine_id].get();
        } else {
            pw_config.transfer_station = nullptr;
        }

        // 创建PW实例（同时传入train_ops和val_ops）
        // 注意：PreprocessWorker会在内部克隆PO，所以可以传递const引用
        // 使用new而不是make_unique，因为make_unique无法处理const unique_ptr向量
        pw_instances_[i] = std::unique_ptr<PreprocessWorker>(
            new PreprocessWorker(
                pw_config,
                train_ops_template_,  // 训练集PO列表
                val_ops_template_,     // 验证集PO列表
                &pw_param_
            )
        );
    }
#endif  // #ifndef TEST_WITHOUT_PW
}

void Preprocessor::destroy_pw_instances() {
#ifndef TEST_WITHOUT_PW
// 测试模式跳过此方法，因为不需要PW

    if (pw_instances_.empty()) {
        return;  // 已销毁或从未创建
    }

    pw_instances_.clear();
#endif  // #ifndef TEST_WITHOUT_PW
}

// =============================================================================
// 设备配置方法：cpu_binding()
// =============================================================================

void Preprocessor::cpu_binding(bool enable) {
    // 步骤1：检查状态机（必须在PreprocessorConfigured状态）
    check_state(ConfigState::PreprocessorConfigured, "cpu_binding");

    // 步骤2：从GlobalRegistry读取GPU配置
    auto& registry = GlobalRegistry::instance();
    bool using_gpu = registry.using_gpu();

    if (using_gpu) {
        engine_device_ = "GPU";
        selected_gpu_ids_ = registry.gpu_ids();
    } else {
        engine_device_ = "CPU";
        selected_gpu_ids_.clear();
    }

    auto_cpu_binding_ = enable;

    // 步骤3：从 GlobalRegistry 读取 world_size（由 use_gpu()/use_cpu() 自动设置）
    world_size_ = GlobalRegistry::instance().world_size();

    // 步骤4：如果是GPU模式且auto_cpu_binding，计算绑核策略
#if defined(TR_SCENE_GPU_CLOUD)
    if (engine_device_ == "GPU" && auto_cpu_binding_) {
        calculate_cpu_binding_strategy();
    } else {
        auto_cpu_binding_ = false;  // 禁用绑核
    }
#else
    auto_cpu_binding_ = false;
#endif

    // 步骤6：注册到GlobalRegistry
    register_device_config();

    // 步骤7：打印设备配置信息
    std::cout << "Device configured: " << engine_device_
              << ", GPUs=" << (selected_gpu_ids_.empty() ? 0 : selected_gpu_ids_.size());
#if defined(TR_SCENE_GPU_CLOUD)
    std::cout << ", Auto CPU Binding: " << (auto_cpu_binding_ ? "True" : "False");
#endif
    std::cout << std::endl;

    // 步骤8：更新状态机
    config_state_ = ConfigState::PreprocessorConfigured;
    device_configured_ = true;
}

// =============================================================================
// 设备配置辅助方法
// =============================================================================

// 计算 Staging Buffer Pool 每个block的大小
// 公式：2 × (align_up(data_raw + 16, 256) + align_up(label_raw + 16, 256))
// +16 为 XNNPACK SIMD 越界保护，与 MemoryPlan::compute_tensor_layout 一致
// 注意：此函数在所有 GPU 场景（GPU_CLOUD / PC_CUDA / 嵌入式）下都需要
static size_t calculate_staging_buffer_size(GlobalRegistry& registry) {
    int res   = registry.max_sample_resolution();
    int ch    = registry.num_color_channels();
    int batch = registry.get_local_batch_size();
    bool amp  = registry.using_amp();

    size_t channel_bytes = amp ? 8ULL : static_cast<size_t>(ch) * 4ULL;
    size_t data_raw  = channel_bytes * static_cast<size_t>(res) * res * batch;
    size_t label_raw = 4ULL * static_cast<size_t>(batch);

    // FP32/INT32 slot = 2 * align_up_256(elems * 2 + 16); FP16 slot = align_up_256(elems * 2 + 16)
    size_t label_aligned = 2 * utils::align_up_256(label_raw / 2 + 16);
    size_t data_aligned;
    if (amp) {
        data_aligned = utils::align_up_256(data_raw + 16);
    } else {
        data_aligned = 2 * utils::align_up_256(data_raw / 2 + 16);
    }

    size_t total = 2ULL * (data_aligned + label_aligned);

    LOG_INFO << "Setup: staging buffer = "
             << (total / (1024 * 1024)) << " MB per GPU"
             << " (" << res << "x" << res << "x" << ch
             << ", batch=" << batch
             << ", " << (amp ? "AMP" : "FP32") << ")";
    return total;
}

#if defined(TR_SCENE_GPU_CLOUD)

// 辅助：通过 libnuma 获取指定 NUMA 节点的所有逻辑 CPU（升序排列）
static std::vector<int> get_cpus_for_numa_node(int node) {
#if defined(TR_USE_LIBNUMA)
    if (numa_available() < 0) {
        return {};
    }
    struct bitmask* mask = numa_allocate_cpumask();
    if (!mask) {
        return {};
    }
    if (numa_node_to_cpus(node, mask) < 0) {
        numa_free_cpumask(mask);
        return {};
    }
    std::vector<int> cpus;
    for (unsigned int i = 0; i < mask->size; ++i) {
        if (numa_bitmask_isbitset(mask, i)) {
            cpus.push_back(static_cast<int>(i));
        }
    }
    numa_free_cpumask(mask);
    return cpus;
#else
    (void)node;
    return {};
#endif
}

void Preprocessor::calculate_cpu_binding_strategy() {
    int total_workers = num_preproc_workers_;

    auto& registry = GlobalRegistry::instance();
    TR_CHECK(registry.has_staging_memory(), RuntimeError,
             "calculate_cpu_binding_strategy: staging memory not allocated, "
             "call allocate_staging_memory() first");

    // 获取系统在线 CPU 核心总数
    long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    TR_CHECK(ncpus > 0, RuntimeError,
             "Failed to detect online CPUs. sysconf returned: " << ncpus);

    std::vector<int> binding_map(total_workers);

    LOG_DEBUG << "CPU Binding Strategy (simple round-robin across all CPUs):";
    LOG_DEBUG << std::string(80, '-');
    LOG_DEBUG << std::left << std::setw(10) << "WorkerID"
              << std::setw(12) << "BindCore"
              << "Note";

    for (int w = 0; w < total_workers; ++w) {
        int target_cpu = w % static_cast<int>(ncpus);
        binding_map[w] = target_cpu;

        LOG_DEBUG << std::left << std::setw(10) << w
                  << std::setw(12) << target_cpu
                  << "  (CPU " << target_cpu << "/" << ncpus << ")";
    }
    LOG_DEBUG << std::string(80, '-');

    registry.set_cpu_binding_map(binding_map);
}

void Preprocessor::bind_worker_to_cpu(int worker_id) {
    const auto& binding_map = GlobalRegistry::instance().cpu_binding_map();

    TR_CHECK(worker_id >= 0 && worker_id < static_cast<int>(binding_map.size()),
             ValueError, "worker_id out of range: " << worker_id);

    int target_cpu = binding_map[worker_id];
    // === 安全守卫：防止无效CPU号 ===
    TR_CHECK(target_cpu >= 0, RuntimeError,
             "bind_worker_to_cpu: invalid CPU " << target_cpu
             << " for worker " << worker_id);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(target_cpu, &cpuset);

    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    TR_CHECK(ret == 0, DeviceError,
             "pthread_setaffinity_np failed for worker " << worker_id
             << " -> CPU " << target_cpu << ": " << strerror(errno));

    LOG_DEBUG << "PW[" << worker_id << "] -> CPU[" << target_cpu << "]";
}

#endif  // TR_SCENE_GPU_CLOUD

void Preprocessor::register_device_config() {
    auto& registry = GlobalRegistry::instance();

    // 注册CPU绑定配置（GPU配置已在测试代码中通过 GlobalRegistry::use_gpu() 设置）
    registry.set_cpu_binding_enabled(auto_cpu_binding_);

    // 注意：CPU绑定映射已在 calculate_cpu_binding_strategy() 中注册
}

void Preprocessor::init() {
    // 空实现：Preprocessor的实际初始化通过配置方法完成
}

void Preprocessor::cleanup() {
    // 空实现：Preprocessor的资源由析构函数自动处理
}

// ============================================================================
// Setup 类实现
// ============================================================================

// Setup 构造/析构
Setup::Setup()
    : state_(std::make_unique<State>()) {
}

Setup::~Setup() {
    if (state_ && !state_->committed) {
        LOG_WARN << "Setup object destroyed without calling commit(). "
                 << "Configuration was not applied.";
    }
}

Setup::Setup(Setup&& other) noexcept
    : state_(std::move(other.state_)) {
}

Setup& Setup::operator=(Setup&& other) noexcept {
    if (this != &other) {
        state_ = std::move(other.state_);
    }
    return *this;
}

// 链式调用实现
Setup& Setup::dataset(const std::string& name, const std::string& path) {
    state_->dataset_name = name;
    state_->dataset_path = path;
    return *this;
}

Setup& Setup::color_channels(int ch) {
    state_->color_channels = ch;
    return *this;
}

Setup& Setup::load_workers(int num) {
    state_->num_load_workers = num;
    return *this;
}

Setup& Setup::preprocess_workers(int num) {
    state_->num_preproc_workers = num;
    return *this;
}

Setup& Setup::using_dts_format(bool dts, int level) {
    state_->using_dts_format = dts;
    state_->dts_compression_level = dts? level: 0;
    return *this;
}

Setup& Setup::fully_mode(bool fully) {
    state_->partial_mode = !fully;
    return *this;
}

// 不建议使用这个API，建议使用fully_mode()来配置，因为默认就是partial mode
Setup& Setup::partial_mode(bool partial) {
    state_->partial_mode = partial;
    return *this;
}

Setup& Setup::shuffle_train(bool shuffle) {
    state_->shuffle_train = shuffle;
    return *this;
}

Setup& Setup::download(bool enable) {
    state_->download = enable;
    return *this;
}

Setup& Setup::sdmp_factor(int factor) {
    state_->sdmp_factor = factor;
    return *this;
}

Setup& Setup::using_cpvs(bool enable) {
    state_->using_cpvs = enable;
    return *this;
}

Setup& Setup::max_intermediate_resolution(int res) {
    state_->max_intermediate_resolution = res;
    return *this;
}

Setup& Setup::drop_last(bool enable) {
    state_->drop_last = enable;
    return *this;
}

Setup& Setup::cpu_binding(bool enable) {
    state_->cpu_binding = enable;
    return *this;
}

Setup& Setup::normalization(NormMode mode) {
    // 保存归一化模式到State中
    state_->norm_mode = mode;
    return *this;
}

// commit() 实现
void Setup::commit() {
    if (state_->committed) {
        LOG_WARN << "Setup::commit() called multiple times. Ignoring duplicate calls.";
        return;
    }

    // 验证必需参数
    if (state_->dataset_name.empty()) {
        TR_VALUE_ERROR("Dataset name not specified. "
                       "Please call .dataset() before commit().");
    }

    if (state_->dataset_path.empty()) {
        TR_VALUE_ERROR("Dataset path not specified. "
                       "Please call .dataset() with a valid path.");
    }

    if (state_->color_channels != 1 && state_->color_channels != 3) {
        TR_VALUE_ERROR("Invalid color channels: " << state_->color_channels
                       << ". Please call .color_channels() with 1 or 3.");
    }

    if (state_->num_load_workers <= 0) {
        TR_VALUE_ERROR("Invalid number of load workers: " << state_->num_load_workers
                       << ". Please call .num_load_workers() with a positive value.");
    }

    if (state_->num_preproc_workers <= 0) {
        TR_VALUE_ERROR("Invalid number of preprocess workers: " << state_->num_preproc_workers
                       << ". Please call .num_preproc_workers() with a positive value.");
    }

    // 从 GlobalRegistry 读取 device 配置
    auto& registry = GlobalRegistry::instance();

    // 尝试读取设备类型，如果 fixed_using_gpu 尚未设置，则设置默认值（CPU模式）
    // 这样可以支持不需要显式调用 GLOBAL_SETTING.use_gpu()/use_cpu() 的场景
    bool is_gpu_mode = false;
    try {
        is_gpu_mode = registry.using_gpu();
    } catch (const TRException&) {
        // fixed_using_gpu 尚未设置，设置为 CPU 模式（默认值）
        registry.use_cpu();
        is_gpu_mode = false;
    }

    std::string device_type = is_gpu_mode ? "GPU" : "CPU";

    // 检查 preprocess_workers 是否为 world_size 的整数倍
    int ws = registry.world_size();
    if (ws > 0 && state_->num_preproc_workers % ws != 0) {
        TR_VALUE_ERROR("Number of preprocess workers (" << state_->num_preproc_workers
                       << ") must be a multiple of world_size (" << ws << "). "
                       << "Got: " << state_->num_preproc_workers % ws << " remainder. "
                       << "Please adjust .preprocess_workers() to be divisible by the number of GPUs.");
    }

    auto& prep = Preprocessor::instance();

    // Step 1: Dataset
    prep.config_dataset(state_->dataset_name, state_->using_dts_format, state_->dts_compression_level);

    // Step 2: DataLoader
    {
        prep.config_dataloader(
            state_->dataset_path,
            state_->num_load_workers,
            state_->num_preproc_workers,
            state_->partial_mode,
            state_->shuffle_train,
            state_->download
        );
    }

    // Step 3: Preprocessor
    {
        prep.config_preprocessor(
            state_->color_channels,
            state_->sdmp_factor,
            state_->using_cpvs,
            false,
            state_->max_intermediate_resolution,
            state_->drop_last
        );
    }

    // Step 4: Device
    {
        // 从 GlobalRegistry 读取 GPU 配置
        bool using_gpu = false;
        try {
            using_gpu = registry.using_gpu();
        } catch (const TRException&) {
            // fixed_using_gpu 尚未设置，使用前面设置的默认值（CPU模式）
            using_gpu = false;
        }

        if (using_gpu) {
            prep.engine_device_ = "GPU";
            prep.selected_gpu_ids_ = registry.gpu_ids();
        } else {
            prep.engine_device_ = "CPU";
            prep.selected_gpu_ids_.clear();
        }
        prep.auto_cpu_binding_ = state_->cpu_binding;

        // 从 GlobalRegistry 读取 world_size（由 use_gpu()/use_cpu() 自动设置）
        prep.world_size_ = registry.world_size();

        // 分配 staging memory（所有场景：GPU / CPU / 嵌入式。GPU场景用 cudaHostAlloc，CPU场景用 malloc）
        {
            size_t bytes_per_gpu = calculate_staging_buffer_size(registry);
            registry.allocate_staging_memory(bytes_per_gpu);
        }

        // 绑核策略仅 GPU_CLOUD（依赖 libnuma）
#if defined(TR_SCENE_GPU_CLOUD)
        if (prep.auto_cpu_binding_) {
            prep.calculate_cpu_binding_strategy();
        }
#else
        prep.auto_cpu_binding_ = false;
#endif

        // 注册到GlobalRegistry
        prep.register_device_config();
    }

    // Step 5: Transforms
    {
        // 检查是否需要自动注入Normalize PO
        bool need_inject_train_norm = (state_->norm_mode != NormMode::NO_NORM);
        bool need_inject_val_norm = (state_->norm_mode != NormMode::NO_NORM);

        // 检查PO链中是否已经包含Normalize
        auto has_normalize_in_chain = [](const std::vector<std::unique_ptr<PreprocessOperation>>& ops) {
            for (const auto& op : ops) {
                if (dynamic_cast<Normalize*>(op.get())) {
                    return true;
                }
            }
            return false;
        };

        bool train_has_normalize = has_normalize_in_chain(state_->train_ops);
        bool val_has_normalize = has_normalize_in_chain(state_->val_ops);

        // 如果用户通过.normalization()指定了NormMode，但PO链中没有Normalize PO，自动注入
        if (need_inject_train_norm && !train_has_normalize) {
            auto norm_po = std::make_unique<Normalize>(state_->norm_mode);
            state_->train_ops.push_back(std::move(norm_po));
        }

        if (need_inject_val_norm && !val_has_normalize) {
            auto norm_po = std::make_unique<Normalize>(state_->norm_mode);
            state_->val_ops.push_back(std::move(norm_po));
        }

        prep.set_train_transforms(state_->train_ops);
        prep.set_val_transforms(state_->val_ops);
    }

    // Step 7: 计算每个epoch的步数（必须在所有配置完成后）
    {
        prep.calculate_steps_per_epoch();
    }

    prep.ensure_inited();

    state_->committed = true;
}

} // namespace tr
