/**
 * @file test_partial_mode.cpp
 * @brief PARTIAL模式完整性和速度测试
 * @details 测试V4.0新架构的PARTIAL模式，验证数据完整性和加载速度
 * @version 4.0.0
 * @date 2026-01-22
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>
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

/**
 * @brief 清空系统文件缓存
 * @details Windows: 使用SetSystemFileCacheSize清空Standby列表
 *          Linux: 使用drop_caches清空页面缓存
 * @return true=成功, false=失败
 */
bool clear_system_cache() {
#ifdef _WIN32
    // Windows: 使用NTDLL API清空系统缓存
    typedef BOOL (WINAPI *SetSystemFileCacheSize_t)(SIZE_T, SIZE_T, DWORD);

    HMODULE hNtdll = GetModuleHandleA("ntdll.dll");
    if (!hNtdll) {
        std::cerr << "[WARNING] Failed to get ntdll.dll handle\n";
        return false;
    }

    auto pSetSystemFileCacheSize = reinterpret_cast<SetSystemFileCacheSize_t>(
        GetProcAddress(hNtdll, "SetSystemFileCacheSize")
    );

    if (!pSetSystemFileCacheSize) {
        std::cerr << "[WARNING] SetSystemFileCacheSize not available\n";
        return false;
    }

    // 清空缓存: (-1, -1, 0) 表示最小和最大都设为-1
    SIZE_T minSize = static_cast<SIZE_T>(-1);
    SIZE_T maxSize = static_cast<SIZE_T>(-1);

    if (pSetSystemFileCacheSize(minSize, maxSize, 0)) {
        std::cout << "[INFO] System cache cleared successfully (Windows)\n";
        return true;
    } else {
        std::cerr << "[WARNING] Failed to clear system cache (may need admin rights)\n";
        return false;
    }

#else
    // Linux: 写入drop_caches
    std::cerr << "[WARNING] --clear-cache on Linux requires root privileges\n";
    std::cerr << "[INFO] Please run manually: echo 3 > /proc/sys/vm/drop_caches\n";
    return false;
#endif
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --dts                Use DTS format (required)\n"
              << "  --path <PATH>        Dataset path (REQUIRED)\n\n"
              << "Optional Options:\n"
              << "  --train              Test training set (default: validation)\n"
              << "  --val                 Test validation set (default)\n"
              << "  --lv <0-3>           DTS compression level (default: 0)\n"
              << "  --workers <N>        Number of IO workers (default: " << DEFAULT_WORKERS << ")\n"
              << "  --preprocess <N>     Number of preprocess workers (default: " << DEFAULT_PREPROCESS << ")\n"
              << "  --shuffle            Enable shuffle (default: disabled for integrity test)\n"
              << "  --decode             Enable JPEG decoding (default: disabled)\n"
              << "  --crop               Enable RandomResizedCrop (requires --decode, default: disabled)\n"
              << "  --log                Enable CSV logging (default: disabled)\n"
              << "  --clear-cache        Clear system cache before testing (for cold cache measurement)\n"
              << "  --help               Show this help message\n\n"
              << "Preprocessing Modes:\n"
              << "  Default:             No decoding, just count samples (fastest)\n"
              << "  --decode:            JPEG decode to RGB (baseline performance)\n"
              << "  --decode --crop:     JPEG decode + RandomResizedCrop to 224x224 (full pipeline)\n\n"
              << "Cache Performance:\n"
              << "  By default, this test measures WARM CACHE performance (fast, 50+ GB/s).\n"
              << "  Use --clear-cache to measure COLD CACHE performance (real I/O speed, 2-3 GB/s).\n"
              << "  For cold cache test, you may need admin rights on Windows.\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dts --path /data/imagenet --val --lv 0\n"
              << "  " << program_name << " --dts --path /data/imagenet --train --lv 3 --workers 4 --preprocess 24\n"
              << "  " << program_name << " --dts --path /data/imagenet --val --lv 0 --decode\n"
              << "  " << program_name << " --dts --path /data/imagenet --val --lv 0 --decode --crop\n"
              << "  " << program_name << " --dts --path /data/imagenet --val --lv 0 --log\n"
              << "  " << program_name << " --dts --path /data/imagenet --val --lv 0 --clear-cache  # Cold cache test\n";
}

void print_results(double time_sec, size_t total_samples,
                   size_t expected_samples, uint64_t dataset_size_bytes,
                   const std::string& dataset_name) {
    double samples_per_sec = total_samples / time_sec;
    double dataset_size_mb = dataset_size_bytes / (1024.0 * 1024.0);
    double mb_per_sec = dataset_size_mb / time_sec;

    std::cout << "\n========================================\n"
              << "PARTIAL Mode Test Results: " << dataset_name << "\n"
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

    bool use_dts = false;
    bool is_train = false;
    int lv = 0;
    std::string dataset_path;
    int num_workers = DEFAULT_WORKERS;
    int num_preprocess = DEFAULT_PREPROCESS;
    bool shuffle = false;
    bool enable_logging = false;  // 新增：日志开关
    bool clear_cache = false;     // 新增：清空缓存选项
    bool enable_decode = false;   // 新增：JPEG解码开关
    bool enable_crop = false;     // 新增：RandomResizedCrop开关

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
            if (num_preprocess < 1 || num_preprocess > 128) {
                std::cerr << "Error: Preprocess workers must be between 1 and 128\n";
                return 1;
            }
        } else if (arg == "--shuffle") {
            shuffle = true;
        } else if (arg == "--decode") {
            enable_decode = true;
        } else if (arg == "--crop") {
            enable_crop = true;
        } else if (arg == "--log") {
            enable_logging = true;
        } else if (arg == "--clear-cache") {
            clear_cache = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // =========================================================================
    // 参数验证
    // =========================================================================

    if (!use_dts) {
        std::cerr << "Error: --dts is required\n";
        print_usage(argv[0]);
        return 1;
    }

    if (dataset_path.empty()) {
        std::cerr << "Error: --path is required\n";
        print_usage(argv[0]);
        return 1;
    }

    // 验证crop依赖decode
    if (enable_crop && !enable_decode) {
        std::cerr << "Error: --crop requires --decode\n";
        return 1;
    }

    // 构建DTS文件路径
    std::string dts_file = dataset_path + "/" +
                           (is_train ? "imagenet_train" : "imagenet_val") +
                           "_lv" + std::to_string(lv) + ".dts";

    // =========================================================================
    // 测试开始
    // =========================================================================

    std::cout << "========================================\n"
              << "PARTIAL Mode Test: " << (is_train ? "Training" : "Validation") << " Set\n"
              << "========================================\n"
              << "DTS file:       " << dts_file << "\n"
              << "IO workers:     " << num_workers << "\n"
              << "Preprocess:     " << num_preprocess << "\n"
              << "Shuffle:        " << (shuffle ? "enabled" : "disabled") << "\n"
              << "JPEG decode:    " << (enable_decode ? "enabled" : "disabled") << "\n"
              << "RandomCrop:     " << (enable_crop ? "enabled" : "disabled") << "\n"
              << "Logging:        " << (enable_logging ? "enabled" : "disabled") << "\n"
              << "Cache mode:     " << (clear_cache ? "COLD (clear cache before test)" : "WARM (cached)") << "\n"
              << "========================================\n";

    // 如果需要清空缓存
    if (clear_cache) {
        std::cout << "\n[0/5] Clearing system cache for cold cache measurement..." << std::endl;
        if (!clear_system_cache()) {
            std::cerr << "\n[ERROR] Failed to clear system cache!" << std::endl;
            std::cerr << "[ERROR] Cold cache test requires administrator privileges." << std::endl;
            std::cerr << "[ERROR] Please run as administrator, or remove --clear-cache flag." << std::endl;
            std::cerr << "\nWindows: Right-click Command Prompt -> Run as Administrator" << std::endl;
            std::cerr << "Linux: Use 'sudo' or run as root" << std::endl;
            return 1;
        }
        // 等待2秒让缓存完全清空
        std::cout << "[INFO] System cache cleared successfully." << std::endl;
        std::cout << "[INFO] Waiting 2 seconds for cache to fully clear...\n" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    try {
        // 获取单例
        ImageNetLoaderDts& loader = ImageNetLoaderDts::getInstance();

        // 配置DataLoader（只配置需要测试的数据集）
        std::cout << "\n[1/5] Configuring loader..." << std::endl;

        // 根据测试目标只配置对应的数据集
        std::string train_path = is_train ? (dataset_path + "/imagenet_train_lv" + std::to_string(lv) + ".dts") : "";
        std::string val_path = is_train ? "" : (dataset_path + "/imagenet_val_lv" + std::to_string(lv) + ".dts");

        loader.configure(
            num_workers,
            num_preprocess,
            train_path,   // train_path（空字符串表示不配置）
            val_path,     // val_path（空字符串表示不配置）
            shuffle,      // shuffle_train
            shuffle,      // shuffle_val（与训练集一致）
            false         // skip_first
        );

        // 设置模式
        if (is_train) {
            loader.set_train_mode(LoadMode::PARTIAL);
        } else {
            loader.set_val_mode(LoadMode::PARTIAL);
        }

        std::cout << "Configuration completed" << std::endl;

        // 配置Preprocessor
        std::cout << "\n[2/5] Configuring preprocessor..." << std::endl;
        Preprocessor& preproc = Preprocessor::getInstance();
        Preprocessor::Config preproc_config;
        preproc_config.num_workers = num_preprocess;
        preproc_config.jpeg_decode = enable_decode;    // 设置JPEG解码
        preproc_config.apply_crop = enable_crop;        // 设置RandomResizedCrop
        preproc_config.enable_logging = enable_logging;
        preproc_config.log_dir = TR_WORKSPACE "/logs";
        preproc.configure(preproc_config);

        std::cout << "Preprocessor configuration completed" << std::endl;

        // 开始epoch（包含第一个buffer的加载）
        std::cout << "\n[3/5] Beginning epoch 0..." << std::endl;
        std::cout << "NOTE: This test measures CACHED performance (warm cache)." << std::endl;
        std::cout << "      For cold cache performance, reboot system or wait 10+ minutes." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();  // 在begin_epoch之前开始计时
        loader.begin_epoch(0, is_train);

        // 预期样本数
        size_t expected_samples = is_train ? 1281167 : 50000;
        std::cout << "Expected samples: " << expected_samples << std::endl;

        // 运行Preprocessor
        std::cout << "\n[4/5] Running preprocessor..." << std::endl;

        preproc.run(loader);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Preprocessing completed" << std::endl;

        // 获取统计信息
        Preprocessor::Stats stats = preproc.get_stats();

        // 获取数据集大小（在end_epoch之前，因为end_epoch会清空current_set_）
        uint64_t dataset_size_bytes = loader.get_current_dataset_size_bytes();

        // 结束epoch
        std::cout << "\n[5/5] Ending epoch..." << std::endl;
        loader.end_epoch();

        // 打印结果
        print_results(elapsed.count(), stats.total_samples, expected_samples,
                     dataset_size_bytes, is_train ? "Training" : "Validation");

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
        if (stats.total_samples == expected_samples) {
            std::cout << "\nINTEGRITY TEST PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << "\nINTEGRITY TEST FAILED!" << std::endl;
            std::cout << "   Expected: " << expected_samples << std::endl;
            std::cout << "   Got:      " << stats.total_samples << std::endl;
            return 1;
        }

    } catch (const TRException& e) {
        std::cout << "\nException caught:" << std::endl;
        std::cout << "  Type:    " << e.type() << std::endl;
        std::cout << "  Message: " << e.message() << std::endl;
        return 1;

    } catch (...) {
        std::cout << "\nUnknown exception caught" << std::endl;
        return 1;
    }

    return 0;
}
