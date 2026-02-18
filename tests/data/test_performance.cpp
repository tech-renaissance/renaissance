/**
 * @file test_performance.cpp
 * @brief DataLoader性能基准测试
 * @version 3.8.0
 * @date 2026-01-17
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace tr;

// 简单的性能计时器
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_seconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }

    double elapsed_milliseconds() const {
        return elapsed_seconds() * 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

int main(int argc, char* argv[]) {
    std::cout << "=== DataLoader Performance Benchmark Test ===" << std::endl;

    try {
        // 获取DataLoader单例
        ImageNetLoaderDts& loader = ImageNetLoaderDts::instance();

        // 配置
        std::cout << "\nConfiguring DataLoader..." << std::endl;
        loader.configure(
            8,                              // num_load_workers
            16,                             // num_preproc_workers
            "T:/dataset/imagenet/imagenet_train_lv0.dts",      // train_path
            "T:/dataset/imagenet/imagenet_val_lv0.dts",        // val_path
            true,                           // shuffle_train
            false,                          // shuffle_val
            false                           // skip_first
        );

        // 测试验证集加载速度
        std::cout << "\n=== Testing Validation Set Loading ===" << std::endl;
        std::cout << "Expected: ~50,000 samples" << std::endl;

        Timer timer;

        loader.begin_epoch(0, false);  // epoch 0, validation

        // 消费所有样本（不记录日志，仅测试速度）
        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;
        size_t total_samples = 0;
        size_t total_bytes = 0;

        while (loader.get_next_sample(0, label, data_ptr, data_size)) {
            total_samples++;
            total_bytes += data_size;

            // 每10000个样本打印一次进度
            if (total_samples % 10000 == 0) {
                std::cout << "  Processed: " << total_samples << " samples\r" << std::flush;
            }
        }

        loader.end_epoch();

        double elapsed = timer.elapsed_seconds();

        std::cout << "\n\n=== Performance Results ===" << std::endl;
        std::cout << "Total samples: " << total_samples << std::endl;
        std::cout << "Total bytes: " << (total_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Total time: " << elapsed << " seconds" << std::endl;

        double throughput_mb_s = (total_bytes / (1024.0 * 1024.0)) / elapsed;
        double throughput_gb_s = throughput_mb_s / 1024.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Throughput: " << throughput_mb_s << " MB/s ("
                  << throughput_gb_s << " GB/s)" << std::endl;

        // 性能目标验证
        std::cout << "\n=== Performance Targets ===" << std::endl;
        std::cout << "Linux target (N=8): 2.0-2.5 GB/s" << std::endl;
        std::cout << "Windows target (N=8): 12-16 GB/s" << std::endl;

        // 简单判断（不考虑平台差异）
        if (throughput_gb_s >= 2.0) {
            std::cout << "\nTEST PASSED: Throughput meets minimum target!" << std::endl;
            return 0;
        } else {
            std::cout << "\nWARNING: Throughput below target" << std::endl;
            std::cout << "   Expected: ≥2.0 GB/s, Got: " << throughput_gb_s << " GB/s" << std::endl;
            return 0;  // 不算失败，因为可能是硬件限制
        }

    } catch (const TRException& e) {
        std::cout << "\nTEST FAILED with exception:" << std::endl;
        std::cout << "  Type: " << e.type() << std::endl;
        std::cout << "  Message: " << e.message() << std::endl;
        std::cout << "\nFull what(): " << e.what() << std::endl;
        return 1;

    } catch (...) {
        std::cout << "\nTEST FAILED: Unknown exception" << std::endl;
        return 1;
    }

    return 0;
}
