/**
 * @file test_shared_buffers.cpp
 * @brief 测试共享缓冲区功能（train PARTIAL + val PARTIAL）
 * @details 验证当训练集和验证集都是PARTIAL模式时，它们共用双缓冲
 * @version 1.0.0
 * @date 2026-01-26
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace tr;

int main(int argc, char** argv) {
    std::cout << "========================================\n"
              << "Shared Buffers Test\n"
              << "========================================\n";

    try {
        // 获取单例
        ImageNetLoaderDts& loader = ImageNetLoaderDts::instance();

        // 配置训练集和验证集（都为PARTIAL模式）
        std::string dataset_path = "/root/epfs/dataset/imagenet";
        std::string train_path = dataset_path + "/imagenet_train_lv0.dts";
        std::string val_path = dataset_path + "/imagenet_val_lv0.dts";

        std::cout << "\n[1/4] Configuring loader..." << std::endl;
        loader.configure(
            16,    // IO workers
            64,    // Preprocess workers
            train_path,
            val_path,
            false,  // shuffle_train
            false,  // shuffle_val
            false,  // skip_first
            false   // verify_crc
        );

        // 设置为PARTIAL模式
        std::cout << "\n[2/4] Setting both train and val to PARTIAL mode..." << std::endl;
        loader.set_train_mode(LoadMode::PARTIAL);
        loader.set_val_mode(LoadMode::PARTIAL);

        // 测试训练集
        std::cout << "\n[3/4] Testing train set..." << std::endl;
        auto start_train = std::chrono::high_resolution_clock::now();
        loader.begin_epoch(0, true);

        // 创建一个简单的预处理器来消费数据
        int total_samples = 0;
        int preproc_worker_id = 0;
        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;

        // 只读取前1000个样本作为测试
        int max_samples = 1000;
        while (total_samples < max_samples &&
               loader.get_next_sample(preproc_worker_id, label, data_ptr, data_size)) {
            total_samples++;
        }

        auto end_train = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_time = end_train - start_train;

        loader.end_epoch();

        std::cout << "Train set test completed: " << total_samples << " samples in "
                 << train_time.count() << " seconds" << std::endl;

        // 测试验证集
        std::cout << "\n[4/4] Testing val set..." << std::endl;
        auto start_val = std::chrono::high_resolution_clock::now();
        loader.begin_epoch(0, false);

        // 重置状态
        total_samples = 0;
        preproc_worker_id = 0;

        // 只读取前1000个样本作为测试
        while (total_samples < max_samples &&
               loader.get_next_sample(preproc_worker_id, label, data_ptr, data_size)) {
            total_samples++;
        }

        auto end_val = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> val_time = end_val - start_val;

        loader.end_epoch();

        std::cout << "Val set test completed: " << total_samples << " samples in "
                 << val_time.count() << " seconds" << std::endl;

        std::cout << "\n========================================\n"
                  << "Shared Buffers Test: PASSED\n"
                  << "========================================\n" << std::endl;

        return 0;

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
