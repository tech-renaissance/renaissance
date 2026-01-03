/**
 * @file test_imagenet_loader.cpp
 * @brief ImageNet数据加载器集成测试
 * @details 测试DTS加载器和原始目录加载器
 * @version 3.7.0
 * @date 2026-01-09
 * @author 技术觉醒团队
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

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <filesystem>

using namespace tr::data;

// =============================================================================
// 配置
// =============================================================================

static constexpr char DEFAULT_DATASET_PATH[] = "I:/imagenet";

// =============================================================================
// 辅助函数
// =============================================================================

/**
 * @brief 打印使用说明
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --dts                Use DTS format (default: raw directory)\n"
              << "  --raw                Use raw directory format (default)\n"
              << "  --full               Use full load mode (default: partial)\n"
              << "  --partial            Use partial load mode (default)\n"
              << "  --train              Load training set (default)\n"
              << "  --val                Load validation set\n"
              << "  --lv <0-3>           DTS compression level (default: 0)\n"
              << "  --path <PATH>        Dataset path (default: " << DEFAULT_DATASET_PATH << ")\n"
              << "  --workers <N>        Number of loader workers (default: 8)\n"
              << "  --preprocess <N>     Number of preprocess workers (default: 16)\n"
              << "  --simulate <MS>      Simulate preprocess time in ms (default: 0)\n"
              << "  --save-worker <N>    Save Nth worker's sample (default: 0)\n"
              << "  --save-sample <N>    Save Nth sample (default: 0)\n"
              << "  --output <PATH>      Output JPEG path (default: output.jpeg)\n"
              << "  --shuffle            Enable shuffle (default: true)\n"
              << "  --no-shuffle         Disable shuffle\n"
              << "  --help               Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dts --train --lv 0\n"
              << "  " << program_name << " --dts --val --lv 0 --workers 4\n"
              << "  " << program_name << " --raw --val --workers 4\n";
}

/**
 * @brief 计算文件或目录总大小
 */
size_t calculate_total_size(const std::string& path, bool is_dts) {
    if (is_dts) {
        // DTS文件：直接获取文件大小
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return 0;
        }
        return static_cast<size_t>(file.tellg());
    } else {
        // 原始目录：递归计算所有JPEG文件大小
        size_t total = 0;
        namespace fs = std::filesystem;

        try {
            for (const auto& entry : fs::recursive_directory_iterator(path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpeg" || ext == ".jpg" || ext == ".png") {
                        total += entry.file_size();
                    }
                }
            }
        } catch (const fs::filesystem_error& e) {
            LOG_WARN << "Failed to calculate directory size: " << e.what();
        }

        return total;
    }
}

// =============================================================================
// 主测试函数
// =============================================================================

int main(int argc, char** argv) {
    // =========================================================================
    // 解析命令行参数
    // =========================================================================

    bool use_dts = false;
    bool full_load = false;
    bool is_train = true;
    int lv = 0;  // DTS压缩级别 (0-3)
    std::string dataset_path = DEFAULT_DATASET_PATH;
    int num_workers = 8;
    int num_preprocess = 16;
    int simulate_ms = 0;
    int save_worker_id = 0;
    int save_sample_idx = 0;
    std::string output_path = "output.jpeg";
    bool shuffle = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dts") {
            use_dts = true;
        } else if (arg == "--raw") {
            use_dts = false;
        } else if (arg == "--full") {
            full_load = true;
        } else if (arg == "--partial") {
            full_load = false;
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
        } else if (arg == "--preprocess" && i + 1 < argc) {
            num_preprocess = std::atoi(argv[++i]);
        } else if (arg == "--simulate" && i + 1 < argc) {
            simulate_ms = std::atoi(argv[++i]);
        } else if (arg == "--save-worker" && i + 1 < argc) {
            save_worker_id = std::atoi(argv[++i]);
        } else if (arg == "--save-sample" && i + 1 < argc) {
            save_sample_idx = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--shuffle") {
            shuffle = true;
        } else if (arg == "--no-shuffle") {
            shuffle = false;
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
              << "ImageNet Data Loader Test\n"
              << "========================================\n"
              << "Dataset path: " << dataset_path << "\n"
              << "Format: " << (use_dts ? "DTS" : "Raw Directory") << "\n"
              << "Load mode: " << (full_load ? "FULL" : "PARTIAL") << "\n"
              << "Dataset: " << (is_train ? "Train" : "Validation") << "\n"
              << "Loader workers: " << num_workers << "\n"
              << "Preprocessor workers: " << num_preprocess << "\n"
              << "Simulate time: " << simulate_ms << " ms\n"
              << "Shuffle: " << (shuffle ? "enabled" : "disabled") << "\n"
              << "Save config: worker=" << save_worker_id
              << ", sample=" << save_sample_idx
              << " -> " << output_path << "\n"
              << "========================================\n\n";

    // =========================================================================
    // 创建数据加载器
    // =========================================================================

    std::unique_ptr<DataLoaderBase> loader;

    try {
        if (use_dts) {
            // 构造DTS文件路径（支持LV0-LV3压缩级别）
            std::string dts_file;
            if (is_train) {
                dts_file = dataset_path + "/imagenet_train_lv" + std::to_string(lv) + ".dts";
            } else {
                dts_file = dataset_path + "/imagenet_val_lv" + std::to_string(lv) + ".dts";
            }

            LOG_INFO << "Loading DTS file: " << dts_file;

            auto dts_loader = std::make_unique<DtsDataLoader>(
                num_workers,
                full_load ? LoadMode::FULL : LoadMode::PARTIAL,
                false  // check_crc=false
            );

            if (!dts_loader->load(dts_file, is_train)) {
                LOG_ERROR << "Failed to load DTS file: " << dts_file;
                return 1;
            }

            loader = std::move(dts_loader);

        } else {
            // 原始目录加载
            std::string data_dir;
            if (is_train) {
                data_dir = dataset_path + "/train";
            } else {
                data_dir = dataset_path + "/val";
            }

            LOG_INFO << "Scanning directory: " << data_dir;

            auto raw_loader = std::make_unique<RawDataLoader>(num_workers);

            if (!raw_loader->load(data_dir, is_train)) {
                LOG_ERROR << "Failed to scan directory: " << data_dir;
                return 1;
            }

            loader = std::move(raw_loader);
        }

        LOG_INFO << "Dataset loaded successfully:"
                 << "\n  Samples: " << loader->num_samples()
                 << "\n  Classes: " << loader->num_classes();

    } catch (const tr::TRException& e) {
        LOG_ERROR << "Failed to create data loader: " << e.what();
        return 1;
    }

    // =========================================================================
    // 创建PreprocessorEmulator
    // =========================================================================

    PreprocessorEmulator emulator(loader.get(), num_preprocess, simulate_ms);
    emulator.save_sample_image(save_worker_id, save_sample_idx, output_path);

    // =========================================================================
    // 开始epoch并计时
    // =========================================================================

    LOG_INFO << "Starting epoch...";

    auto start = std::chrono::high_resolution_clock::now();

    loader->begin_epoch(0, shuffle, false);  // shuffle=shuffle, skip_first=false
    emulator.start();

    // =========================================================================
    // 等待完成
    // =========================================================================

    emulator.join();
    loader->end_epoch();

    auto end = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // 计算统计信息
    // =========================================================================

    double time_sec = std::chrono::duration<double>(end - start).count();
    size_t total_bytes = calculate_total_size(
        use_dts ? (is_train ? dataset_path + "/imagenet_train.dts"
                            : dataset_path + "/imagenet_val.dts")
                : (is_train ? dataset_path + "/train"
                            : dataset_path + "/val"),
        use_dts
    );
    double speed_mb = total_bytes / (1024.0 * 1024.0) / time_sec;

    size_t total_processed = emulator.get_total_processed();

    // =========================================================================
    // 输出结果
    // =========================================================================

    std::cout << "\n========================================\n"
              << "Test Results\n"
              << "========================================\n"
              << "Load time: " << time_sec << " s\n"
              << "Total bytes: " << total_bytes / (1024.0 * 1024 * 1024) << " GB\n"
              << "Speed: " << speed_mb << " MB/s\n"
              << "Total samples processed: " << total_processed << "\n"
              << "Samples per second: " << (total_processed / time_sec) << "\n"
              << "========================================\n\n";

    // 输出标签统计（前20个）
    const auto& label_counts = emulator.get_label_counts();
    std::cout << "Label distribution (first 20 classes):\n";
    int count = 0;
    for (const auto& [label, num] : label_counts) {
        if (count++ >= 20) break;
        std::cout << "  Label " << label << ": " << num << " samples\n";
    }
    if (label_counts.size() > 20) {
        std::cout << "  ... and " << (label_counts.size() - 20) << " more classes\n";
    }
    std::cout << "\n";

    // 检查保存的文件
    std::ifstream output_file(output_path, std::ios::binary | std::ios::ate);
    if (output_file.good()) {
        size_t output_size = output_file.tellg();
        std::cout << "Saved image: " << output_path
                  << " (" << output_size << " bytes)\n\n";
    } else {
        std::cout << "Warning: Failed to save image to " << output_path << "\n\n";
    }

    LOG_INFO << "Test completed successfully";

    return 0;
}
