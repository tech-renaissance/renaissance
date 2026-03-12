/**
 * @file test_epoch_crc.cpp
 * @brief 任意epoch和phase的CRC32测试
 * @details 测试指定epoch和phase的CRC32值
 *          每个epoch定义：先加载训练集、后加载验证集
 *          例如：epoch=1,phase=train 表示运行：
 *            - epoch0 train (不输出CRC)
 *            - epoch0 val (不输出CRC)
 *            - epoch1 train (输出CRC)
 * @version 1.0.0
 * @date 2026-02-07
 * @author 技术觉醒团队
 */

// 必须在所有包含之前设置Windows宏
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifdef ERROR
        #undef ERROR
    #endif
#endif

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <vector>

using namespace tr;

// =============================================================================
// 默认配置
// =============================================================================

static constexpr char DEFAULT_DATASET_PATH[] = "T:/Datasets";

// =============================================================================
// 辅助函数
// =============================================================================

/**
 * @brief 打印使用说明
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --dataset <NAME>     Dataset to test: mnist, cifar-10, cifar-100, imagenet\n"
              << "                       (case-insensitive, supports both cifar10/cifar-10)\n"
              << "  --format <FMT>       Format: raw, dts (case-insensitive)\n"
              << "  --phase <PHASE>      Phase to test: train, val (case-insensitive)\n"
              << "  --epoch <N>          Epoch ID to test (0, 1, 2, ...)\n"
              << "  --path <PATH>        Dataset root path (default: " << DEFAULT_DATASET_PATH << ")\n\n"
              << "Optional Options:\n"
              << "  --mode <MODE>        Mode: partial, fully (case-insensitive, default: partial)\n"
              << "  --shuffle            Enable data shuffling (default: disabled)\n"
              << "  --lv <0-3>           DTS compression level for ImageNet (default: 0)\n"
              << "  --workers <N>        Number of load workers (default: 16)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 16)\n"
              << "  --help               Show this help message\n\n"
              << "Epoch Definition:\n"
              << "  Each epoch consists of: train phase -> val phase\n"
              << "  Examples:\n"
              << "    --epoch 0 --phase train   Test first training set (no warm-up)\n"
              << "    --epoch 0 --phase val     Test first validation set (warmed by train)\n"
              << "    --epoch 1 --phase train   Test second training set (warmed by epoch0)\n"
              << "    --epoch 5 --phase val     Test 6th validation set (after 5 full epochs)\n\n"
              << "Note: FULLY mode only loads data once (epoch 0), later epochs just shuffle.\n"
              << "      CRC logs saved to: " << TR_WORKSPACE << "/logs/\n"
              << "      CSV format: worker_id,size,label,first_byte,crc32\n"
              << "Command Sample: /root/epfs/R/renaissance/build/bin/tests/pw/test_epoch_crc --dataset imagenet --format raw --phase train --epoch 2 --path /root/epfs/dataset/imagenet --mode partial --shuffle --lv 0 --workers 16 --preproc 8\n\n";
}

/**
 * @brief 运行单个phase（train或val）
 * @return 返回是否输出CRC
 */
bool run_phase(Preprocessor& prep, int epoch_id, bool is_train,
               DataLoader* loader, bool enable_crc_logging, int num_preproc_workers) {
    // 【调试】报告当前阶段
    std::cout << "  >>> Entering Epoch " << epoch_id
              << " " << (is_train ? "TRAIN" : "VAL") << " phase <<<\n" << std::flush;

    // 配置是否启用CRC计算和日志记录
    Preprocessor::Config config;
    config.num_workers = num_preproc_workers;  // 使用实际的preprocess worker数量
    config.enable_logging = enable_crc_logging;  // 关键：控制是否输出CRC
    config.calc_crc = enable_crc_logging;
    config.jpeg_decode = false;      // 禁用JPEG解码
    config.apply_crop = false;
    config.simulate_delay = false;

    // 【调试】打印配置信息
    std::cout << "  Config: enable_logging=" << config.enable_logging
              << ", calc_crc=" << config.calc_crc << "\n" << std::flush;

    prep.configure(config);

    // 运行phase
    if (is_train) {
        prep.train();
    } else {
        prep.val();
    }

    // 【调试】报告阶段完成
    std::cout << "  >>> Epoch " << epoch_id
              << " " << (is_train ? "TRAIN" : "VAL") << " phase completed <<<\n" << std::flush;

    return true;
}

/**
 * @brief 测试指定epoch和phase的CRC
 */
void test_epoch_crc(const std::string& dataset_name,
                   const std::string& format_str,
                   const std::string& mode_str,
                   const std::string& phase_str,
                   int target_epoch,
                   const std::string& dataset_path,
                   bool enable_shuffle,
                   int compression_level,
                   int num_load_workers,
                   int num_preproc_workers) {
    // 【调试】启用DEBUG日志级别
    Logger::instance().set_level(LogLevel::DEBUG);

    // 获取Preprocessor单例
    auto& prep = Preprocessor::instance();

    // 解析格式（转换为大写以支持大小写不敏感）
    std::string format_upper = format_str;
    std::transform(format_upper.begin(), format_upper.end(), format_upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    bool dts_format = (format_upper == "DTS");

    // 解析模式（转换为大写以支持大小写不敏感）
    std::string mode_upper = mode_str;
    std::transform(mode_upper.begin(), mode_upper.end(), mode_upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    bool partial_mode = (mode_upper == "PARTIAL");

    // 解析phase（转换为大写以支持大小写不敏感）
    std::string phase_upper = phase_str;
    std::transform(phase_upper.begin(), phase_upper.end(), phase_upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    bool target_is_train = (phase_upper == "TRAIN");

    // 使用新的Setup构建器API配置Preprocessor
    Preprocessor::setup()
        .dataset(dataset_name, dataset_path)
        .using_dts_format(dts_format, compression_level)
        .batch_size(32)  // TEST_WITHOUT_PW模式可以使用小batch
        .max_output_resolution(224)
        .color_channels(3)
        .load_workers(num_load_workers)
        .preprocess_workers(num_preproc_workers)
        .fully_mode(!partial_mode)
        .shuffle_train(enable_shuffle)
        .download(false)
        .sdmp_factor(1)
        .using_cpvs(false)
        .train_transforms(*std::make_unique<DoNothing>())
        .val_transforms(*std::make_unique<DoNothing>())
        .commit();

    // 获取DataLoader引用（用于获取样本数等信息）
    DataLoader* loader = nullptr;

    // 转换数据集名称为小写用于比较
    std::string dataset_lower = dataset_name;
    std::transform(dataset_lower.begin(), dataset_lower.end(), dataset_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // 标准化名称（移除连字符）
    std::string dataset_normalized = dataset_lower;
    size_t hyphen_pos;
    while ((hyphen_pos = dataset_normalized.find('-')) != std::string::npos) {
        dataset_normalized.replace(hyphen_pos, 1, "");
    }

    // 根据标准化后的名称选择DataLoader
    if (dataset_normalized == "mnist") {
        loader = dts_format ? (DataLoader*)&MnistLoaderDts::instance()
                            : (DataLoader*)&MnistLoaderRaw::instance();
    } else if (dataset_normalized == "cifar10") {
        // CIFAR-10和CIFAR-100使用相同的Loader类，Loader内部通过detected_num_classes_区分
        loader = dts_format ? (DataLoader*)&CifarLoaderDts::instance()
                            : (DataLoader*)&CifarLoaderRaw::instance();
    } else if (dataset_normalized == "cifar100") {
        // CIFAR-100和CIFAR-10使用相同的Loader类，Loader内部通过detected_num_classes_区分
        loader = dts_format ? (DataLoader*)&CifarLoaderDts::instance()
                            : (DataLoader*)&CifarLoaderRaw::instance();
    } else if (dataset_normalized == "imagenet") {
        loader = dts_format ? (DataLoader*)&ImageNetLoaderDts::instance()
                            : (DataLoader*)&ImageNetLoaderRaw::instance();
    }

    if (!loader) {
        std::cerr << "Error: Failed to get DataLoader instance\n";
        exit(1);
    }

    // 打印测试信息
    std::cout << "\n========================================================================\n";
    std::cout << "Epoch CRC Test: " << dataset_name << " " << format_str << " " << mode_str << "\n";
    std::cout << "========================================================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Dataset: " << loader->dataset_name() << "\n";
    std::cout << "  Format: " << (dts_format ? "DTS" : "RAW");
    if (dts_format && dataset_normalized == "imagenet") {
        std::cout << " LV" << compression_level;
    }
    std::cout << "\n";
    std::cout << "  Mode: " << (partial_mode ? "PARTIAL" : "FULLY") << "\n";
    std::cout << "  Shuffle: " << (enable_shuffle ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "  Load workers: " << num_load_workers << "\n";
    std::cout << "  Preprocess workers: " << num_preproc_workers << "\n";
    std::cout << "  Target: Epoch " << target_epoch << " " << (target_is_train ? "TRAIN" : "VAL") << "\n\n";

    std::cout << "Execution Plan:\n";
    int phase_count = 0;
    for (int epoch = 0; epoch <= target_epoch; ++epoch) {
        // 每个epoch先train后val
        // train phase
        if (epoch == target_epoch && target_is_train) {
            std::cout << "  [" << ++phase_count << "] Epoch " << epoch << " TRAIN  -> CRC LOGGING ENABLED\n";
        } else {
            std::cout << "  [" << ++phase_count << "] Epoch " << epoch << " TRAIN  -> (no CRC)\n";
        }
        // val phase
        if (epoch == target_epoch && !target_is_train) {
            std::cout << "  [" << ++phase_count << "] Epoch " << epoch << " VAL    -> CRC LOGGING ENABLED\n";
        } else if (epoch < target_epoch || target_is_train) {
            std::cout << "  [" << ++phase_count << "] Epoch " << epoch << " VAL    -> (no CRC)\n";
        }
    }
    std::cout << "\n";

    std::cout << "CSV Output Format:\n";
    std::cout << "  worker_id,size,label,first_byte,crc32\n\n";

    std::cout << "Running test...\n\n";

    // 执行测试流程
    auto start = std::chrono::high_resolution_clock::now();
    bool crc_logged = false;

    for (int epoch = 0; epoch <= target_epoch; ++epoch) {
        // Train phase
        bool enable_crc = (epoch == target_epoch && target_is_train);
        std::cout << "Running Epoch " << epoch << " TRAIN...\n" << std::flush;
        run_phase(prep, epoch, true, loader, enable_crc, num_preproc_workers);
        if (enable_crc) {
            std::cout << "  [CRC LOGGED]\n" << std::flush;
            crc_logged = true;
        }
        std::cout << "\n" << std::flush;

        // 如果已经打印了CRC，就退出
        if (crc_logged) {
            break;
        }

        // Val phase (only if not the last train phase)
        if (epoch < target_epoch || !target_is_train) {
            enable_crc = (epoch == target_epoch && !target_is_train);
            std::cout << "Running Epoch " << epoch << " VAL...\n" << std::flush;
            run_phase(prep, epoch, false, loader, enable_crc, num_preproc_workers);
            if (enable_crc) {
                std::cout << "  [CRC LOGGED]\n" << std::flush;
                crc_logged = true;
            }
            std::cout << "\n" << std::flush;

            // 如果已经打印了CRC，就退出
            if (crc_logged) {
                break;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 获取统计信息
    auto stats = prep.get_stats();
    size_t expected_samples = target_is_train ? loader->num_train_samples() : loader->num_val_samples();
    bool integrity_passed = (stats.total_samples == expected_samples);

    // 打印结果
    std::cout << "\n========================================================================\n";
    std::cout << "Test Results:\n";
    std::cout << "========================================================================\n";
    std::cout << "  Target: Epoch " << target_epoch << " " << (target_is_train ? "TRAIN" : "VAL") << "\n";
    std::cout << "  Total time:       " << std::fixed << std::setprecision(3) << elapsed.count() << " s\n";
    std::cout << "  Samples processed: " << stats.total_samples << "\n";
    std::cout << "  Expected samples:  " << expected_samples << "\n";
    std::cout << "  Integrity:        " << (integrity_passed ? "PASSED" : "FAILED") << "\n";
    std::cout << "  Throughput:       " << std::fixed << std::setprecision(1)
              << (static_cast<double>(stats.total_samples) / elapsed.count()) << " samples/s\n";
    std::cout << "\n";
    std::cout << "CSV logs saved to: " << TR_WORKSPACE << "/logs/\n";
    std::cout << "  (One file per worker, format: worker_<worker_id>.csv)\n";
    std::cout << "========================================================================\n\n";

    if (integrity_passed) {
        std::cout << "SUCCESS: Epoch " << target_epoch << " " << (target_is_train ? "TRAIN" : "VAL")
                  << " CRC test completed with integrity check PASSED!\n\n";
    } else {
        std::cout << "WARNING: Integrity check failed, but CSV logs were generated.\n\n";
    }
}

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char* argv[]) {
    // 【调试】在程序最开始就启用完整日志输出（必须在任何Logger调用之前）
    tr::Logger::instance().set_level(tr::LogLevel::DEBUG);

    // 默认配置
    std::string dataset_arg = "";
    std::string format_arg = "";
    std::string mode_arg = "partial";
    std::string phase_arg = "";
    std::string dataset_path = DEFAULT_DATASET_PATH;
    int target_epoch = -1;
    bool enable_shuffle = false;
    int compression_level = 0;
    int num_load_workers = 16;
    int num_preproc_workers = 16;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_arg = argv[++i];
        } else if (arg == "--format" && i + 1 < argc) {
            format_arg = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            mode_arg = argv[++i];
        } else if (arg == "--phase" && i + 1 < argc) {
            phase_arg = argv[++i];
        } else if (arg == "--epoch" && i + 1 < argc) {
            target_epoch = std::stoi(argv[++i]);
            if (target_epoch < 0) {
                std::cerr << "Error: --epoch must be >= 0\n";
                return 1;
            }
        } else if (arg == "--shuffle") {
            enable_shuffle = true;
        } else if (arg == "--lv" && i + 1 < argc) {
            compression_level = std::stoi(argv[++i]);
            if (compression_level < 0 || compression_level > 3) {
                std::cerr << "Error: LV must be between 0 and 3\n";
                return 1;
            }
        } else if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--workers" && i + 1 < argc) {
            num_load_workers = std::stoi(argv[++i]);
        } else if (arg == "--preproc" && i + 1 < argc) {
            num_preproc_workers = std::stoi(argv[++i]);
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 【第一句】初始化框架（必须在所有其他操作之前）
    INIT_FRAMEWORK("GPU");

    // 验证必需参数
    if (dataset_arg.empty() || format_arg.empty() || phase_arg.empty() || target_epoch < 0) {
        std::cerr << "Error: --dataset, --format, --phase, and --epoch are required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // 验证format参数
    if (format_arg != "raw" && format_arg != "dts") {
        std::cerr << "Error: --format must be 'raw' or 'dts'\n";
        return 1;
    }

    // 验证mode参数
    if (mode_arg != "partial" && mode_arg != "fully") {
        std::cerr << "Error: --mode must be 'partial' or 'fully'\n";
        return 1;
    }

    // 验证phase参数
    if (phase_arg != "train" && phase_arg != "val") {
        std::cerr << "Error: --phase must be 'train' or 'val'\n";
        return 1;
    }

    // 运行测试
    try {
        test_epoch_crc(dataset_arg, format_arg, mode_arg, phase_arg, target_epoch,
                      dataset_path, enable_shuffle, compression_level, num_load_workers, num_preproc_workers);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
