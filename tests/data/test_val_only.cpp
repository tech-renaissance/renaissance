/**
 * @file test_val_only.cpp
 * @brief 仅测试验证集加载（快速测试）
 * @version 4.0.0
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

private:
    std::chrono::high_resolution_clock::time_point start_;
};

int main() {
    std::cout << "=== Validation Set ONLY Test ===" << std::endl;

    try {
        // 获取DataLoader单例
        ImageNetLoaderDts& loader = ImageNetLoaderDts::getInstance();

        // 配置（只配置val路径，train路径随便填）
        std::cout << "\nConfiguring DataLoader..." << std::endl;
        loader.configure(
            8,                                      // num_load_workers
            16,                                     // num_preproc_workers
            "T:/dataset/imagenet/imagenet_train_lv0.dts",  // train_path (不使用)
            "T:/dataset/imagenet/imagenet_val_lv0.dts",    // val_path (使用这个)
            true,                                   // shuffle_train
            false,                                  // shuffle_val
            false                                   // skip_first
        );

        // 测试验证集加载速度
        std::cout << "\n=== Testing Validation Set Loading ===" << std::endl;
        std::cout << "Expected: 50,000 samples" << std::endl;
        std::cout << "Mode: Validation (no shuffle)" << std::endl;

        Timer timer;

        loader.begin_epoch(0, false);  // epoch 0, validation

        // 消费所有样本
        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;
        size_t total_samples = 0;
        size_t total_bytes = 0;

        while (loader.get_next_sample(0, label, data_ptr, data_size)) {
            total_samples++;
            total_bytes += data_size;

            // 每5000个样本打印一次进度
            if (total_samples % 5000 == 0) {
                std::cout << "  Processed: " << total_samples << " samples\r" << std::flush;
            }
        }

        loader.end_epoch();

        double elapsed = timer.elapsed_seconds();

        std::cout << "\n\n=== Performance Results ===" << std::endl;
        std::cout << "Total samples: " << total_samples << std::endl;
        std::cout << "Total bytes: " << std::fixed << std::setprecision(2)
                  << (total_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(3)
                  << elapsed << " seconds" << std::endl;

        double throughput_mb_s = (total_bytes / (1024.0 * 1024.0)) / elapsed;
        double throughput_gb_s = throughput_mb_s / 1024.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Throughput: " << throughput_mb_s << " MB/s ("
                  << throughput_gb_s << " GB/s)" << std::endl;

        // 性能目标验证
        std::cout << "\n=== Performance Targets ===" << std::endl;
        std::cout << "Linux target (N=8): 2.0-2.5 GB/s" << std::endl;
        std::cout << "Windows target (N=8): 12-16 GB/s" << std::endl;

        // 样本数量验证
        std::cout << "\n=== Sample Count Verification ===" << std::endl;
        if (total_samples == 50000) {
            std::cout << "✅ Sample count CORRECT: 50,000" << std::endl;
        } else {
            std::cout << "⚠️  Sample count mismatch: expected 50,000, got "
                      << total_samples << std::endl;
        }

        // 吞吐量评估
        std::cout << "\n=== Performance Evaluation ===" << std::endl;
        if (throughput_gb_s >= 12.0) {
            std::cout << "✅ EXCELLENT: Throughput " << throughput_gb_s
                      << " GB/s meets Windows target!" << std::endl;
            return 0;
        } else if (throughput_gb_s >= 2.0) {
            std::cout << "✅ GOOD: Throughput " << throughput_gb_s
                      << " GB/s meets Linux target." << std::endl;
            return 0;
        } else {
            std::cout << "⚠️  WARNING: Throughput " << throughput_gb_s
                      << " GB/s is below target (≥2.0 GB/s)" << std::endl;
            return 0;  // 不算失败，继续报告
        }

    } catch (const TRException& e) {
        std::cout << "\n✗ TEST FAILED with exception:" << std::endl;
        std::cout << "  Type: " << e.type() << std::endl;
        std::cout << "  Message: " << e.message() << std::endl;
        std::cout << "\nFull what(): " << e.what() << std::endl;
        return 1;

    } catch (const std::exception& e) {
        std::cout << "\n✗ TEST FAILED with std::exception:" << std::endl;
        std::cout << "  what(): " << e.what() << std::endl;
        return 1;

    } catch (...) {
        std::cout << "\n✗ TEST FAILED: Unknown exception" << std::endl;
        return 1;
    }

    return 0;
}
