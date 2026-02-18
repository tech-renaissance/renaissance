/**
 * @file test_dataloader_performance.cpp
 * @brief DataLoader性能测试
 * @details 测试DTS格式数据加载器的读取速度，对标method2_native.cpp
 * @version 4.0.0
 * @date 2026-01-17
 * @author 技术觉醒团队
 * @note 依赖项: libcurl
 * @note 所属系列: data
 */

// 必须在所有包含之前设置Windows宏
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX  // 防止min/max宏被定义
    #endif
    #ifdef ERROR
        #undef ERROR
    #endif
#endif

#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>
#include "renaissance.h"

using namespace tr;

// =============================================================================
// 配置
// =============================================================================

static constexpr char DEFAULT_DATASET_PATH[] = "T:/dataset/imagenet";
static constexpr int DEFAULT_WORKERS = 8;
static constexpr int DEFAULT_PREPROCESS = 16;

// =============================================================================
// 辅助函数
// =============================================================================

/**
 * @brief 打印使用说明
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --dts                Use DTS format (required)\n"
              << "  --train              Load training set (default: validation)\n"
              << "  --val                Load validation set (default)\n"
              << "  --lv <0-3>           DTS compression level (default: 0)\n"
              << "  --path <PATH>        Dataset path (default: " << DEFAULT_DATASET_PATH << ")\n"
              << "  --workers <N>        Number of loader workers (default: " << DEFAULT_WORKERS << ")\n"
              << "  --preprocess <N>     Number of preprocess workers (default: " << DEFAULT_PREPROCESS << ")\n"
              << "  --mode <MODE>        Load mode: partial or fully (default: partial)\n"
              << "  --shuffle            Enable shuffle (default: disabled for performance test)\n"
              << "  --help               Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dts --val --lv 0 --mode partial\n"
              << "  " << program_name << " --dts --train --lv 3 --workers 16 --mode fully\n"
              << "  " << program_name << " --dts --val --lv 0 --path /data/imagenet --mode fully\n";
}

/**
 * @brief 打印性能测试结果
 */
void print_results(double time_sec, size_t total_bytes, size_t total_samples) {
    double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
    double throughput_gb = total_gb / time_sec;
    double throughput_mb = total_bytes / (1024.0 * 1024.0) / time_sec;

    std::cout << "\n========================================\n"
              << "Performance Test Results\n"
              << "========================================\n"
              << std::fixed << std::setprecision(3)
              << "Load time:     " << time_sec << " s\n"
              << "Total bytes:   " << total_gb << " GB\n"
              << "Total samples: " << total_samples << "\n"
              << "Throughput:    " << throughput_gb << " GB/s\n"
              << "               " << throughput_mb << " MB/s\n"
              << "Samples/sec:   " << (total_samples / time_sec) << "\n"
              << "========================================\n";
}

// =============================================================================
// 主测试函数
// =============================================================================

int main(int argc, char** argv) {
    // =========================================================================
    // 解析命令行参数
    // =========================================================================

    bool use_dts = false;
    bool is_train = false;
    int lv = 0;
    std::string dataset_path = DEFAULT_DATASET_PATH;
    int num_workers = DEFAULT_WORKERS;
    int num_preprocess = DEFAULT_PREPROCESS;
    std::string mode_str = "partial";  // 默认 partial 模式
    bool shuffle = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dts") {
            use_dts = true;
        } else if (arg == "--train") {
            is_train = true;
        } else if (arg == "--val") {
            is_train = false;
        } else if (arg == "--lv" && i + 1 < argc) {
            lv = std::atoi(argv[++i]);
            if (lv < 0 || lv > 3) {
                std::cerr << "Error: LV must be between 0 and 3\n";
                return 1;
            }
        } else if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--workers" && i + 1 < argc) {
            num_workers = std::atoi(argv[++i]);
            if (num_workers < 1 || num_workers > 16) {
                std::cerr << "Error: Workers must be between 1 and 16\n";
                return 1;
            }
        } else if (arg == "--preprocess" && i + 1 < argc) {
            num_preprocess = std::atoi(argv[++i]);
            if (num_preprocess < 1 || num_preprocess > 64) {
                std::cerr << "Error: Preprocess workers must be between 1 and 64\n";
                return 1;
            }
        } else if (arg == "--mode" && i + 1 < argc) {
            mode_str = argv[++i];
            if (mode_str != "partial" && mode_str != "fully") {
                std::cerr << "Error: Mode must be 'partial' or 'fully'\n";
                return 1;
            }
        } else if (arg == "--shuffle") {
            shuffle = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // =========================================================================
    // 打印配置信息
    // =========================================================================

    std::cout << "\n========================================\n"
              << "DataLoader Performance Test\n"
              << "========================================\n"
              << "Dataset path: " << dataset_path << "\n"
              << "Format: " << (use_dts ? "DTS" : "Raw (not supported)") << "\n"
              << "Dataset: " << (is_train ? "Train" : "Validation") << "\n"
              << "Compression LV: " << lv << "\n"
              << "Loader workers: " << num_workers << "\n"
              << "Preprocess workers: " << num_preprocess << "\n"
              << "Load mode: " << mode_str << "\n"
              << "Shuffle: " << (shuffle ? "enabled" : "disabled") << "\n"
              << "========================================\n\n";

    if (!use_dts) {
        std::cerr << "Error: Only DTS format is supported in this test\n";
        return 1;
    }

    // =========================================================================
    // 构造DTS文件路径
    // =========================================================================

    std::string train_file = dataset_path + "/imagenet_train_lv" + std::to_string(lv) + ".dts";
    std::string val_file = dataset_path + "/imagenet_val_lv" + std::to_string(lv) + ".dts";

    std::string target_file = is_train ? train_file : val_file;
    LOG_INFO << "Target DTS file: " << target_file;

    // =========================================================================
    // 创建并配置DataLoader
    // =========================================================================

    try {
        auto& loader = ImageNetLoaderDts::instance();

        // 设置加载模式（根据命令行参数）
        LoadMode mode = (mode_str == "fully") ? LoadMode::FULLY : LoadMode::PARTIAL;
        // 只对当前测试的数据集设置 FULLY，另一个数据集使用 PARTIAL 以节省内存
        if (is_train) {
            loader.set_train_mode(mode);
            loader.set_val_mode(LoadMode::PARTIAL);  // 验证集始终 PARTIAL
        } else {
            loader.set_train_mode(LoadMode::PARTIAL);  // 训练集始终 PARTIAL
            loader.set_val_mode(mode);
        }

        // 配置加载器
        // 注意：必须提供训练集和验证集的完整路径，即使只测试其中一个
        loader.configure(
            num_workers,
            num_preprocess,
            train_file,    // train_path (训练集文件)
            val_file,      // val_path (验证集文件)
            shuffle,       // shuffle_train
            false,         // shuffle_val (验证集默认不乱序)
            false          // skip_first (第一个epoch也乱序，除非--skip-first)
        );

        // =========================================================================
        // 开始Epoch并计时
        // =========================================================================

        LOG_INFO << "Starting epoch...";

        auto start = std::chrono::high_resolution_clock::now();

        loader.begin_epoch(0, is_train);  // epoch_id=0, is_train

        // =========================================================================
        // 并行消费所有样本
        // =========================================================================

        std::atomic<size_t> total_samples{0};
        std::atomic<size_t> total_bytes{0};

        // 简单的并行消费（不使用OpenMP，避免额外依赖）
        std::vector<std::thread> preprocess_threads;

        for (int worker_id = 0; worker_id < num_preprocess; ++worker_id) {
            preprocess_threads.emplace_back([&, worker_id]() {
                size_t local_samples = 0;
                size_t local_bytes = 0;

                int32_t label;
                const uint8_t* data_ptr;
                size_t data_size;

                while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
                    local_samples++;
                    local_bytes += data_size;

                    // 可选：模拟预处理延迟
                    // std::this_thread::sleep_for(std::chrono::microseconds(100));
                }

                total_samples.fetch_add(local_samples, std::memory_order_relaxed);
                total_bytes.fetch_add(local_bytes, std::memory_order_relaxed);
            });
        }

        // 等待所有Preprocessor线程完成
        for (auto& t : preprocess_threads) {
            t.join();
        }

        loader.end_epoch();

        auto end = std::chrono::high_resolution_clock::now();

        // =========================================================================
        // 计算并打印结果
        // =========================================================================

        double time_sec = std::chrono::duration<double>(end - start).count();
        size_t total_samples_val = total_samples.load(std::memory_order_relaxed);
        size_t total_bytes_val = total_bytes.load(std::memory_order_relaxed);

        print_results(time_sec, total_bytes_val, total_samples_val);

        LOG_INFO << "Test completed successfully";

        return 0;

    } catch (const tr::TRException& e) {
        std::cerr << "TRException: " << e.what() << std::endl;
        LOG_ERROR << "Failed to run data loader: " << e.what();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        LOG_ERROR << "Standard exception: " << e.what();
        return 1;
    } catch (...) {
        // 捕获所有其他异常，包括可能的Windows SEH异常
        std::cerr << "Unknown exception caught - This may be a segmentation fault or access violation" << std::endl;
        LOG_ERROR << "Unknown exception - possibly memory access violation";
        return 1;
    }
}
