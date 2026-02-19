/**
 * @file test_gpu_binding.cpp
 * @brief 测试GPU选择、CPU绑核和映射关系
 * @version 1.0.0
 * @date 2026-02-18
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace tr;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --gpus <ID,ID,...>  GPU IDs to use (e.g., 0,1,4,7)\n\n"
              << "Optional Options:\n"
              << "  --preproc <N>        Number of preprocess workers (default: 16)\n"
              << "  --cpu-bind          Enable CPU binding (default: disabled)\n"
              << "  --help              Show this help message\n\n"
              << "Description:\n"
              << "  This test verifies GPU selection and CPU binding strategy.\n"
              << "  It prints the mapping between engine_id (0~world_size-1)\n"
              << "  and real GPU IDs.\n\n"
              << "Examples:\n"
              << "  " << program_name << " --gpus 0,1,4,7\n"
              << "  " << program_name << " --gpus 0,2,4,6 --preproc 32 --cpu-bind\n";
}

std::vector<int> parse_gpu_ids(const std::string& gpu_str) {
    std::vector<int> gpu_ids;
    std::stringstream ss(gpu_str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        try {
            int gpu_id = std::stoi(token);
            if (gpu_id < 0) {
                std::cerr << "Error: GPU ID must be non-negative, got " << gpu_id << "\n";
                return {};
            }
            gpu_ids.push_back(gpu_id);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid GPU ID '" << token << "'\n";
            return {};
        }
    }

    return gpu_ids;
}

int main(int argc, char* argv[]) {
    // 默认配置
    std::vector<int> selected_gpus;
    int num_preproc_workers = 16;
    bool auto_cpu_binding = false;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--gpus" && i + 1 < argc) {
            std::string gpu_str = argv[++i];
            selected_gpus = parse_gpu_ids(gpu_str);
            if (selected_gpus.empty()) {
                std::cerr << "Error: Failed to parse GPU IDs\n";
                return 1;
            }
        } else if (arg == "--preproc" && i + 1 < argc) {
            num_preproc_workers = std::stoi(argv[++i]);
            if (num_preproc_workers < 1) {
                std::cerr << "Error: --preproc must be at least 1\n";
                return 1;
            }
        } else if (arg == "--cpu-bind") {
            auto_cpu_binding = true;
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 验证必需参数
    if (selected_gpus.empty()) {
        std::cerr << "Error: --gpus is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // 打印配置信息
    std::cout << "\n========================================\n"
              << "GPU Binding and Mapping Test\n"
              << "========================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Selected GPUs: ";
    for (size_t i = 0; i < selected_gpus.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << selected_gpus[i];
    }
    std::cout << "\n";
    std::cout << "  World size: " << selected_gpus.size() << "\n";
    std::cout << "  Preprocess workers: " << num_preproc_workers << "\n";
    std::cout << "  CPU binding: " << (auto_cpu_binding ? "Enabled" : "Disabled") << "\n\n";

    // 配置Preprocessor
    auto& prep = Preprocessor::instance();

    // 步骤1：配置数据集（使用ImageNet RAW作为示例）
    std::cout << "=== Step 1: Configuring Dataset ===\n";
    prep.config_dataset(DatasetType::imagenet, false, 0);  // RAW, no compression

    // 步骤2：配置DataLoader（使用虚拟路径，不实际加载数据）
    std::cout << "Configuring DataLoader...\n";
    prep.config_dataloader(
        "/root/datasets/imagenet",  // 虚拟路径
        1,                           // num_load_workers（测试用1个）
        num_preproc_workers,
        true,                        // partial_mode
        false,                       // shuffle_train
        false                        // download
    );

    // 步骤3：配置Preprocessor
    std::cout << "Configuring Preprocessor...\n";
    prep.config_preprocessor(
        selected_gpus.size(),       // world_size（匹配GPU数量）
        256,                        // batch_size
        224,                        // max_resolution
        3,                          // num_color_channels
        1,                          // sdmp_factor
        false,                      // using_cpvs
        true                        // pw_test_mode（测试模式）
    );

    // 步骤4：配置设备（指定GPU）
    std::cout << "Configuring Device...\n";
    prep.config_device("GPU", selected_gpus, auto_cpu_binding);

    // 步骤5：设置数据变换（使用Resize，测试模式不需要实际解码）
    std::cout << "Setting transforms (Resize only, test mode)...\n";
    prep.set_train_transforms(Resize(224));
    prep.set_val_transforms(Resize(224));

    // 步骤6：多线程初始化（包括CPU绑核）
    std::cout << "\n=== Step 2: Multi-thread Initialization ===\n";
    prep.multi_thread_init();

    // 步骤7：查询并打印映射关系
    std::cout << "\n=== Step 3: GPU Mapping Query ===\n";

    auto& registry = GlobalRegistry::instance();
    const auto& gpu_ids = registry.gpu_ids();
    int world_size = registry.world_size();

    std::cout << "\nEngine ID to Real GPU ID Mapping:\n";
    std::cout << "----------------------------------------\n";
    for (int engine_id = 0; engine_id < world_size; ++engine_id) {
        int real_gpu_id = gpu_ids[engine_id];
        std::cout << "  Engine " << engine_id << " → GPU " << real_gpu_id << "\n";
    }
    std::cout << "----------------------------------------\n\n";

    // 步骤8：打印CPU绑核策略（如果启用）
    if (auto_cpu_binding) {
        std::cout << "=== Step 4: CPU Binding Strategy ===\n";

        const auto& binding_map = registry.cpu_binding_map();

        std::cout << "\nWorker ID to CPU Core Binding:\n";
        std::cout << "----------------------------------------\n";
        std::cout << std::left << std::setw(12) << "Worker ID"
                  << std::setw(12) << "Engine ID"
                  << "CPU Core\n";
        std::cout << "----------------------------------------\n";

        for (int worker_id = 0; worker_id < num_preproc_workers; ++worker_id) {
            int engine_id = worker_id % world_size;
            int real_gpu_id = gpu_ids[engine_id];
            int cpu_core = binding_map[worker_id];

            std::cout << std::left << std::setw(12) << worker_id
                      << std::setw(12) << engine_id
                      << "(GPU " << real_gpu_id << ") → CPU " << cpu_core << "\n";

            // 只打印每个Engine的第一个worker
            if ((worker_id + 1) % (num_preproc_workers / world_size) == 0) {
                std::cout << "----------------------------------------\n";
            }
        }
    }

    // 汇总
    std::cout << "\n========================================\n"
              << "Test Completed Successfully\n"
              << "========================================\n"
              << "(No data loading/decoding performed)\n\n";

    return 0;
}
