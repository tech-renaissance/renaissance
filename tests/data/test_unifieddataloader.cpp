/**
 * @file test_unifieddataloader.cpp
 * @brief UnifiedDataLoader综合性能测试程序
 * @details 支持所有数据集（MNIST/CIFAR-10/CIFAR-100/ImageNet）的RAW和DTS格式
 *          完整覆盖test_partial_mode.cpp和test_fully_mode.cpp的所有测试功能
 * @version 2.0.0
 * @date 2026-02-02
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <algorithm>
#include <thread>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

using namespace tr;

// =============================================================================
// 配置
// =============================================================================

static constexpr int DEFAULT_WORKERS = 8;
static constexpr int DEFAULT_PREPROCESS = 16;

// =============================================================================
// 辅助函数
// =============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --dataset_name <NAME>   Dataset name: mnist, cifar-10, cifar-100, imagenet (REQUIRED)\n\n"
              << "Optional Options:\n"
              << "  --path <PATH>            Dataset path (default: TR_WORKSPACE/datasets/<NAME>)\n"
              << "  --using_dts              Use DTS format (default: RAW)\n"
              << "  --fully_mode             Use FULLY mode instead of auto-selected mode (overrides default)\n"
              << "  --lv <0-3>               DTS compression level for ImageNet (default: 0, DTS only)\n"
              << "  --workers <N>            Number of IO workers (default: " << DEFAULT_WORKERS << ")\n"
              << "  --preprocess <N>         Number of preprocess workers (default: " << DEFAULT_PREPROCESS << ")\n"
              << "  --shuffle                Enable shuffle (default: disabled for integrity test)\n"
              << "  --log                    Enable CSV logging (default: disabled)\n"
              << "  --help                   Show this help message\n\n"
              << "Examples:\n"
              << "  # Test MNIST RAW (auto-selects FULLY mode)\n"
              << "  " << program_name << " --dataset_name mnist\n\n"
              << "  # Test CIFAR-10 DTS (auto-selects FULLY mode)\n"
              << "  " << program_name << " --dataset_name cifar-10 --using_dts\n\n"
              << "  # Test ImageNet DTS in PARTIAL mode (default)\n"
              << "  " << program_name << " --dataset_name imagenet --using_dts\n\n"
              << "  # Test ImageNet DTS in FULLY mode (override default)\n"
              << "  " << program_name << " --dataset_name imagenet --using_dts --fully_mode\n\n"
              << "  # Test ImageNet DTS with LV3 compression\n"
              << "  " << program_name << " --dataset_name imagenet --using_dts --lv 3 --workers 16\n\n"
              << "Note:\n"
              << "  - Both training and validation sets will be tested (first train, then val)\n"
              << "  - Load mode is auto-selected by default:\n"
              << "    * MNIST/CIFAR RAW: FULLY mode (only supported mode)\n"
              << "    * MNIST/CIFAR DTS: FULLY mode (recommended for small datasets)\n"
              << "    * ImageNet RAW/DTS: PARTIAL mode (default, recommended for large datasets)\n"
              << "  - Use --fully_mode to override auto-selection and force FULLY mode\n";
}

std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

void print_results(double time_sec, size_t total_samples,
                   size_t expected_samples, uint64_t dataset_size_bytes,
                   const std::string& split_name) {
    double samples_per_sec = total_samples / time_sec;
    double dataset_size_mb = dataset_size_bytes / (1024.0 * 1024.0);
    double mb_per_sec = dataset_size_mb / time_sec;

    std::cout << "\n========================================\n"
              << "Test Results: " << split_name << " Set\n"
              << "========================================\n"
              << std::fixed << std::setprecision(3)
              << "Load time:        " << time_sec << " s\n"
              << "Dataset size:     " << dataset_size_mb << " MB\n"
              << "Throughput:       " << mb_per_sec << " MB/s\n"
              << "Total samples:    " << total_samples << "\n"
              << "Expected samples: " << expected_samples << "\n"
              << "Samples/sec:      " << static_cast<int>(samples_per_sec) << "\n"
              << "Integrity:        " << (total_samples == expected_samples ? "PASSED" : "FAILED") << "\n"
              << "========================================\n";
}

// =============================================================================
// 主测试函数
// =============================================================================

int main(int argc, char** argv) {
    // =========================================================================
    // 解析命令行参数
    // =========================================================================

    std::string dataset_name;
    std::string dataset_path;
    bool using_dts = false;
    int lv = 0;
    int num_workers = DEFAULT_WORKERS;
    int num_preprocess = DEFAULT_PREPROCESS;
    bool fully_mode = false;
    bool shuffle = false;
    bool enable_logging = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dataset_name" && i + 1 < argc) {
            dataset_name = argv[++i];
        } else if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--using_dts") {
            using_dts = true;
        } else if (arg == "--fully_mode") {
            fully_mode = true;
        } else if (arg == "--lv" && i + 1 < argc) {
            lv = std::atoi(argv[++i]);
            if (lv < 0 || lv > 3) {
                std::cerr << "Error: LV must be between 0 and 3\n";
                return 1;
            }
        } else if (arg == "--workers" && i + 1 < argc) {
            num_workers = std::atoi(argv[++i]);
            if (num_workers < 1 || num_workers > 16) {
                std::cerr << "Error: Workers must be between 1 and 16\n";
                return 1;
            }
        } else if (arg == "--preprocess" && i + 1 < argc) {
            num_preprocess = std::atoi(argv[++i]);
            if (num_preprocess < 1 || num_preprocess > 128) {
                std::cerr << "Error: Preprocess workers must be between 1 and 128\n";
                return 1;
            }
        } else if (arg == "--shuffle") {
            shuffle = true;
        } else if (arg == "--log") {
            enable_logging = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // =========================================================================
    // 参数验证
    // =========================================================================

    if (dataset_name.empty()) {
        std::cerr << "Error: --dataset_name is required\n";
        print_usage(argv[0]);
        return 1;
    }

    // 标准化数据集名称
    dataset_name = to_lower(dataset_name);

    // 验证数据集名称
    if (dataset_name != "mnist" && dataset_name != "cifar-10" &&
        dataset_name != "cifar-100" && dataset_name != "imagenet") {
        std::cerr << "Error: Invalid dataset_name '" << dataset_name << "'\n";
        std::cerr << "       Supported datasets: mnist, cifar-10, cifar-100, imagenet\n";
        return 1;
    }

    // 设置默认路径
    if (dataset_path.empty()) {
        dataset_path = std::string(TR_WORKSPACE) + "/datasets/" + dataset_name;
    }

    // =========================================================================
    // 配置UnifiedDataLoader
    // =========================================================================

    std::cout << "========================================\n"
              << "UnifiedDataLoader Performance Test\n"
              << "========================================\n"
              << "Dataset:         " << dataset_name << "\n"
              << "Format:          " << (using_dts ? "DTS" : "RAW") << "\n"
              << "Path:            " << dataset_path << "\n"
              << "IO workers:      " << num_workers << "\n"
              << "Preprocess:      " << num_preprocess << "\n"
              << "Mode override:   " << (fully_mode ? "FULLY (manual)" : "Auto-selected") << "\n";

    if (using_dts && dataset_name == "imagenet") {
        std::cout << "Compression LV:  " << lv << "\n";
    }

    std::cout << "Shuffle:         " << (shuffle ? "enabled" : "disabled") << "\n"
              << "Logging:         " << (enable_logging ? "enabled" : "disabled") << "\n"
              << "========================================\n";

    try {
        // 获取UnifiedDataLoader单例
        UnifiedDataLoader& loader = UnifiedDataLoader::getInstance();

        // 选择数据集
        std::cout << "[1/6] Selecting dataset..." << std::endl;
        DatasetType dataset_type;

        if (dataset_name == "mnist") {
            dataset_type = DatasetType::mnist;
        } else if (dataset_name == "cifar-10") {
            dataset_type = DatasetType::cifar_10;
        } else if (dataset_name == "cifar-100") {
            dataset_type = DatasetType::cifar_100;
        } else {  // imagenet
            dataset_type = DatasetType::imagenet;
        }

        loader.select_dataset(dataset_type, using_dts ? "DTS" : "RAW", lv);

        // 配置loader（使用简化版configure，会自动选择默认模式）
        std::cout << "[2/6] Configuring loader..." << std::endl;
        loader.configure(dataset_path, num_workers, num_preprocess,
                        shuffle,  // shuffle_train
                        shuffle); // shuffle_val

        // 如果用户指定了--fully_mode，手动覆盖默认模式
        if (fully_mode) {
            std::cout << "[INFO] Overriding auto-selected mode to FULLY (user specified --fully_mode)" << std::endl;
            loader.set_train_mode(LoadMode::FULLY);
            loader.set_val_mode(LoadMode::FULLY);
        }

        std::cout << "[INFO] Loader configuration completed" << std::endl;

        // 配置Preprocessor
        std::cout << "\n[3/6] Configuring preprocessor..." << std::endl;
        Preprocessor& preproc = Preprocessor::getInstance();
        Preprocessor::Config preproc_config;
        preproc_config.num_workers = num_preprocess;
        preproc_config.jpeg_decode = (!using_dts && dataset_name == "imagenet");
        preproc_config.apply_crop = false;
        preproc_config.enable_logging = enable_logging;
        preproc_config.log_dir = std::string(TR_WORKSPACE) + "/logs";
        preproc.configure(preproc_config);

        std::cout << "[INFO] Preprocessor configuration completed" << std::endl;

        // =========================================================================
        // 测试训练集
        // =========================================================================

        std::cout << "\n========================================\n"
                  << "Testing Training Set\n"
                  << "========================================" << std::endl;

        // 开始epoch（包含第一个buffer的加载）
        std::cout << "[4/6] Beginning epoch 0 (training)..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        loader.begin_epoch(0, true);

        // 预期样本数
        size_t expected_samples = 0;
            if (dataset_name == "mnist") {
                expected_samples = 60000;
            } else if (dataset_name == "cifar-10" || dataset_name == "cifar-100") {
                expected_samples = 50000;
            } else if (dataset_name == "imagenet") {
                expected_samples = 1281167;
            }

            std::cout << "Expected samples: " << expected_samples << std::endl;

            // 运行Preprocessor
            std::cout << "\n[5/6] Running preprocessor..." << std::endl;
            preproc.run(loader);

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;

            std::cout << "[INFO] Preprocessing completed" << std::endl;

            // 获取统计信息
            Preprocessor::Stats stats = preproc.get_stats();

            // 计算数据集大小（手动计算：label + image）
            uint64_t dataset_size_bytes = 0;
            if (dataset_name == "imagenet") {
                if (using_dts) {
                    dataset_size_bytes = ImageNetLoaderDts::getInstance().get_current_dataset_size_bytes();
                } else {
                    dataset_size_bytes = ImageNetLoaderRaw::getInstance().get_current_dataset_size_bytes();
                }
            } else if (dataset_name == "mnist") {
                // MNIST: 1 byte label + 784 bytes image (28*28*1)
                size_t image_bytes = 28 * 28 * 1;
                dataset_size_bytes = expected_samples * (1 + image_bytes);
            } else if (dataset_name == "cifar-10" || dataset_name == "cifar-100") {
                // CIFAR: 1 byte label + 3072 bytes image (32*32*3)
                size_t image_bytes = 32 * 32 * 3;
                dataset_size_bytes = expected_samples * (1 + image_bytes);
            }

            // 结束epoch
            std::cout << "\n[6/6] Ending epoch..." << std::endl;
            loader.end_epoch();

            // 打印结果
            print_results(elapsed.count(), stats.total_samples, expected_samples,
                        dataset_size_bytes, "Training");

            // 打印buffer统计
            std::cout << "\nBuffer statistics:" << std::endl;
            std::cout << "  Total buffers processed: " << stats.buffer_count << std::endl;

        // 打印每个worker的样本数
        std::cout << "\nWorker sample distribution:" << std::endl;
        for (int i = 0; i < num_preprocess; ++i) {
            std::cout << "  Worker " << std::setw(2) << i << ": "
                     << std::setw(8) << stats.per_worker[i] << " samples" << std::endl;
        }

        // 验证完整性
        if (stats.total_samples != expected_samples) {
            std::cout << "\n[ERROR] Training set test FAILED!" << std::endl;
            std::cout << "   Expected: " << expected_samples << std::endl;
            std::cout << "   Got:      " << stats.total_samples << std::endl;
            return 1;
        }

        std::cout << "\n[SUCCESS] Training set test PASSED!" << std::endl;

        // =========================================================================
        // 测试验证集
        // =========================================================================

        std::cout << "\n========================================\n"
                  << "Testing Validation Set\n"
                  << "========================================" << std::endl;

        // 开始epoch
        std::cout << "[4/6] Beginning epoch 0 (validation)..." << std::endl;
        auto start_time_val = std::chrono::high_resolution_clock::now();
        loader.begin_epoch(0, false);

        // 预期样本数
        size_t expected_samples_val = 0;
        if (dataset_name == "mnist") {
            expected_samples_val = 10000;
        } else if (dataset_name == "cifar-10") {
            expected_samples_val = 10000;
        } else if (dataset_name == "cifar-100") {
            expected_samples_val = 10000;
        } else if (dataset_name == "imagenet") {
            expected_samples_val = 50000;
        }

        std::cout << "Expected samples: " << expected_samples_val << std::endl;

        // 运行Preprocessor
        std::cout << "\n[5/6] Running preprocessor..." << std::endl;
        preproc.run(loader);

        auto end_time_val = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_val = end_time_val - start_time_val;

        std::cout << "[INFO] Preprocessing completed" << std::endl;

        // 获取统计信息
        Preprocessor::Stats stats_val = preproc.get_stats();

        // 计算数据集大小（手动计算：label + image）
        uint64_t dataset_size_bytes_val = 0;
        if (dataset_name == "imagenet") {
            if (using_dts) {
                dataset_size_bytes_val = ImageNetLoaderDts::getInstance().get_current_dataset_size_bytes();
            } else {
                dataset_size_bytes_val = ImageNetLoaderRaw::getInstance().get_current_dataset_size_bytes();
            }
        } else if (dataset_name == "mnist") {
            // MNIST: 1 byte label + 784 bytes image (28*28*1)
            size_t image_bytes = 28 * 28 * 1;
            dataset_size_bytes_val = expected_samples_val * (1 + image_bytes);
        } else if (dataset_name == "cifar-10" || dataset_name == "cifar-100") {
            // CIFAR: 1 byte label + 3072 bytes image (32*32*3)
            size_t image_bytes = 32 * 32 * 3;
            dataset_size_bytes_val = expected_samples_val * (1 + image_bytes);
        }

        // 结束epoch
        std::cout << "\n[6/6] Ending epoch..." << std::endl;
        loader.end_epoch();

        // 打印结果
        print_results(elapsed_val.count(), stats_val.total_samples, expected_samples_val,
                    dataset_size_bytes_val, "Validation");

        // 打印buffer统计
        std::cout << "\nBuffer statistics:" << std::endl;
        std::cout << "  Total buffers processed: " << stats_val.buffer_count << std::endl;

        // 打印每个worker的样本数
        std::cout << "\nWorker sample distribution:" << std::endl;
        for (int i = 0; i < num_preprocess; ++i) {
            std::cout << "  Worker " << std::setw(2) << i << ": "
                     << std::setw(8) << stats_val.per_worker[i] << " samples" << std::endl;
        }

        // 验证完整性
        if (stats_val.total_samples != expected_samples_val) {
            std::cout << "\n[ERROR] Validation set test FAILED!" << std::endl;
            std::cout << "   Expected: " << expected_samples_val << std::endl;
            std::cout << "   Got:      " << stats_val.total_samples << std::endl;
            return 1;
        }

        std::cout << "\n[SUCCESS] Validation set test PASSED!" << std::endl;

        // =========================================================================
        // 总结
        // =========================================================================

        std::cout << "\n========================================\n"
                  << "ALL TESTS PASSED!\n"
                  << "========================================\n"
                  << "Dataset: " << dataset_name << " (" << (using_dts ? "DTS" : "RAW") << ")\n"
                  << "========================================\n";

        return 0;

    } catch (const TRException& e) {
        std::cout << "\n[ERROR] Exception caught:" << std::endl;
        std::cout << "  Type:    " << e.type() << std::endl;
        std::cout << "  Message: " << e.message() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cout << "\n[ERROR] Standard exception caught:" << std::endl;
        std::cout << "  Message: " << e.what() << std::endl;
        return 1;
    }
}
