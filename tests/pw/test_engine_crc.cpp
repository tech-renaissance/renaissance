/**
 * @file test_engine_crc.cpp
 * @brief EngineBuffer CRC保存测试：验证CRC32计算和CSV保存功能
 * @version 1.0.0
 * @date 2026-02-25
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <algorithm>
#include <vector>

using namespace tr;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --dataset <TYPE>    Dataset type: mnist, cifar10\n"
              << "  --path <PATH>       Dataset root path\n\n"
              << "Optional Options:\n"
              << "  --format <fmt>       Dataset format: raw, dts (default: raw)\n"
              << "  --mode <MODE>        DataLoader mode: fully, partial (default: partial)\n"
              << "  --batch-size <N>     Batch size (default: 256)\n"
              << "  --cpvs <true/false>  Enable CPVS (default: false)\n"
              << "  --sdmp <N>           SDMP factor (default: 1)\n"
              << "  --shuffle            Enable training set shuffling (default: disabled)\n"
              << "  --reproducible       Enable reproducible mode (default: disabled)\n"
              << "  --epoch <N>          Number of epochs to run (default: 1)\n"
              << "  --val-first          Run validation first (epoch 0)\n"
              << "  --resolution <N>     Output resolution (default: 224)\n"
              << "  --po-train1 <PO>     First train PO (Resize, CenterCrop, RandomResizedCrop, etc.)\n"
              << "  --po-train2 <PO>     Second train PO (optional)\n"
              << "  --po-train3 <PO>     Third train PO (optional)\n"
              << "  --po-val1 <PO>       First val PO (default: same as po-train1)\n"
              << "  --po-val2 <PO>       Second val PO (optional)\n"
              << "  --po-val3 <PO>       Third val PO (optional)\n"
              << "  --loaders <N>        Number of load workers (default: 16)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 16)\n"
              << "  --device <TYPE>      Device type: CPU, GPU (default: CPU)\n"
              << "  --gpu-ids <IDS>      GPU IDs (e.g., \"0,1,2,3\" or \"0-7\", default: \"0\")\n"
              << "  --seed <N>           Random seed (default: 42)\n"
              << "  --help               Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dataset mnist --path /data/mnist --epoch 2\n"
              << "  " << program_name << " --dataset cifar10 --path /data/cifar --val-first --epoch 3\n"
              << "  " << program_name << " --dataset cifar10 --path /data/cifar --po-train1 RandomResizedCrop --reproducible\n\n"
              << "Note: CRC files will be saved to " TR_WORKSPACE "/eb_[engine_id]_[train/val]_[epoch].csv\n"
              << "Command Sample: /root/epfs/R/renaissance/build/bin/tests/pw/test_engine_crc --dataset mnist --path /root/epfs/dataset/mnist --format raw --batch-size 256 --cpvs true --sdmp 2 --shuffle --reproducible --epoch 3 --resolution 28 --po-train1 DoNothing --po-train2 DoNothing --preproc 112 --device GPU --gpu-ids \"0,1,2,3,4,5,6,7\" --seed 42\n\n";
}

/**
 * @brief 检查PO是否为resize/crop类（需要resolution参数）
 */
bool is_resize_or_crop_po(const std::string& po_name) {
    std::string po_lower = po_name;
    std::transform(po_lower.begin(), po_lower.end(), po_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    return po_lower == "resize" ||
           po_lower == "centercrop" ||
           po_lower == "randomresizedcrop" ||
           po_lower == "fastrandomresizedcrop" ||
           po_lower == "randomcrop";
}

/**
 * @brief 解析PO名称并创建PO对象
 */
std::unique_ptr<PreprocessOperation> create_po(const std::string& po_name,
                                               float scale_min, float scale_max,
                                               float ratio_min, float ratio_max,
                                               float flip_prob) {
    std::string po_lower = po_name;
    std::transform(po_lower.begin(), po_lower.end(), po_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (po_lower == "resize" || po_lower == "centercrop" ||
        po_lower == "randomresizedcrop" || po_lower == "fastrandomresizedcrop" ||
        po_lower == "randomcrop") {
        std::cerr << "Error: PO '" << po_name << "' requires --resolution parameter\n";
        return nullptr;
    } else if (po_lower == "randomhorizontalflip") {
        return std::make_unique<RandomHorizontalFlip>(flip_prob);
    } else if (po_lower == "donothing") {
        return std::make_unique<DoNothing>();
    } else {
        std::cerr << "Error: Unknown PO type: " << po_name << "\n";
        std::cerr << "Supported: RandomHorizontalFlip, DoNothing\n";
        return nullptr;
    }
}

/**
 * @brief 创建resize/crop类PO（带resolution参数）
 */
std::unique_ptr<PreprocessOperation> create_resize_crop_po(const std::string& po_name, int resolution,
                                                          float scale_min, float scale_max,
                                                          float ratio_min, float ratio_max) {
    std::string po_lower = po_name;
    std::transform(po_lower.begin(), po_lower.end(), po_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (po_lower == "resize") {
        return std::make_unique<Resize>(resolution);
    } else if (po_lower == "centercrop") {
        return std::make_unique<CenterCrop>(resolution);
    } else if (po_lower == "randomresizedcrop") {
        return std::make_unique<RandomResizedCrop>(resolution, scale_min, scale_max,
                                                   ratio_min, ratio_max);
    } else if (po_lower == "fastrandomresizedcrop") {
        return std::make_unique<FastRandomResizedCrop>(resolution, scale_min, scale_max,
                                                        ratio_min, ratio_max);
    } else if (po_lower == "randomcrop") {
        return std::make_unique<RandomCrop>(resolution);
    } else {
        std::cerr << "Error: PO '" << po_name << "' is not a resize/crop operation\n";
        return nullptr;
    }
}

/**
 * @brief 计算单个PO的输出尺寸
 * @param input_size 输入尺寸
 * @param po_name PO名称
 * @param target_resolution resize/crop类PO的目标分辨率
 * @return 该PO的输出尺寸
 */
int calculate_single_po_output(int input_size, const std::string& po_name, int target_resolution) {
    if (po_name.empty()) {
        return input_size; // 空PO相当于DoNothing，保持输入尺寸
    }

    std::string po_lower = po_name;
    std::transform(po_lower.begin(), po_lower.end(), po_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // resize/crop类PO输出目标分辨率
    if (po_lower == "resize" || po_lower == "centercrop" ||
        po_lower == "randomresizedcrop" || po_lower == "fastrandomresizedcrop" ||
        po_lower == "randomcrop") {
        return target_resolution;
    }

    // 其他PO保持输入尺寸
    return input_size;
}

/**
 * @brief 计算 PO 链的最终输出尺寸（链式计算）
 * @param dataset_base_size 数据集原始尺寸（MNIST=28, CIFAR=32）
 * @param po_name1, po_name2, po_name3 三个PO名称
 * @param target_resolution resize/crop类PO的目标分辨率
 * @return PO3的最终输出尺寸
 */
int calculate_po_chain_final_output(int dataset_base_size,
                                    const std::string& po_name1,
                                    const std::string& po_name2,
                                    const std::string& po_name3,
                                    int target_resolution) {
    // PO1作用于数据集原始尺寸
    int size_after_po1 = calculate_single_po_output(dataset_base_size, po_name1, target_resolution);

    // PO2作用于PO1的输出
    int size_after_po2 = calculate_single_po_output(size_after_po1, po_name2, target_resolution);

    // PO3作用于PO2的输出
    int size_after_po3 = calculate_single_po_output(size_after_po2, po_name3, target_resolution);

    return size_after_po3;
}

int main(int argc, char* argv[]) {
    // 默认配置
    std::string dataset_type_str;
    std::string dataset_path;
    std::string format_arg = "raw";
    std::string mode_arg = "partial";
    int batch_size = 256;
    bool using_cpvs = false;
    int sdmp_factor = 1;
    bool shuffle_train = false;
    bool reproducible = false;
    int num_epochs = 1;
    bool val_first = false;
    int resolution = 224;
    std::string po_train1, po_train2, po_train3;
    std::string po_val1, po_val2, po_val3;
    int num_load_workers = 16;
    int num_preproc_workers = 16;
    std::string device_type = "CPU";
    std::string gpu_ids_str = "";
    uint64_t random_seed = 42;

    // RandomResizedCrop参数
    float scale_min = 0.08f;
    float scale_max = 1.0f;
    float ratio_min = 0.75f;
    float ratio_max = 1.3333333f;
    float flip_prob = 0.5f;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_type_str = argv[++i];
        } else if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--format" && i + 1 < argc) {
            format_arg = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            mode_arg = argv[++i];
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--cpvs" && i + 1 < argc) {
            std::string cpvs_str = argv[++i];
            std::transform(cpvs_str.begin(), cpvs_str.end(), cpvs_str.begin(), ::tolower);
            using_cpvs = (cpvs_str == "true");
        } else if (arg == "--sdmp" && i + 1 < argc) {
            sdmp_factor = std::stoi(argv[++i]);
        } else if (arg == "--shuffle") {
            shuffle_train = true;
        } else if (arg == "--reproducible") {
            reproducible = true;
        } else if (arg == "--epoch" && i + 1 < argc) {
            num_epochs = std::stoi(argv[++i]);
        } else if (arg == "--val-first") {
            val_first = true;
        } else if (arg == "--resolution" && i + 1 < argc) {
            resolution = std::stoi(argv[++i]);
        } else if (arg == "--po-train1" && i + 1 < argc) {
            po_train1 = argv[++i];
        } else if (arg == "--po-train2" && i + 1 < argc) {
            po_train2 = argv[++i];
        } else if (arg == "--po-train3" && i + 1 < argc) {
            po_train3 = argv[++i];
        } else if (arg == "--po-val1" && i + 1 < argc) {
            po_val1 = argv[++i];
        } else if (arg == "--po-val2" && i + 1 < argc) {
            po_val2 = argv[++i];
        } else if (arg == "--po-val3" && i + 1 < argc) {
            po_val3 = argv[++i];
        } else if (arg == "--loaders" && i + 1 < argc) {
            num_load_workers = std::stoi(argv[++i]);
        } else if (arg == "--preproc" && i + 1 < argc) {
            num_preproc_workers = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device_type = argv[++i];
            std::transform(device_type.begin(), device_type.end(), device_type.begin(),
                           [](unsigned char c) { return std::toupper(c); });
            if (device_type != "CPU" && device_type != "GPU") {
                std::cerr << "Error: --device must be 'CPU' or 'GPU'\n";
                return 1;
            }
        } else if (arg == "--gpu-ids" && i + 1 < argc) {
            gpu_ids_str = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            random_seed = std::stoull(argv[++i]);
        } else if (arg == "--scale-min" && i + 1 < argc) {
            scale_min = std::stof(argv[++i]);
        } else if (arg == "--scale-max" && i + 1 < argc) {
            scale_max = std::stof(argv[++i]);
        } else if (arg == "--ratio-min" && i + 1 < argc) {
            ratio_min = std::stof(argv[++i]);
        } else if (arg == "--ratio-max" && i + 1 < argc) {
            ratio_max = std::stof(argv[++i]);
        } else if (arg == "--prob" && i + 1 < argc) {
            flip_prob = std::stof(argv[++i]);
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 【第一句】初始化框架（必须在所有其他操作之前）
    INIT_FRAMEWORK(device_type);
    std::cout << "Framework initialized with device: " << device_type << "\n";

    // 【第二句】设置可复现性保险
    GlobalRegistry::instance().ensure_reproducibility(reproducible);
    if (reproducible) {
        std::cout << "Reproducible mode: ENABLED\n";
    } else {
        std::cout << "Reproducible mode: DISABLED (performance optimized)\n";
    }

    // 验证必需参数
    if (dataset_type_str.empty()) {
        std::cerr << "Error: --dataset is required\n";
        print_usage(argv[0]);
        return 1;
    }
    if (dataset_path.empty()) {
        std::cerr << "Error: --path is required\n";
        print_usage(argv[0]);
        return 1;
    }

    // 验证dataset类型（只支持mnist, cifar10, cifar100）
    std::transform(dataset_type_str.begin(), dataset_type_str.end(), dataset_type_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (dataset_type_str != "mnist" &&
        dataset_type_str != "cifar10" && dataset_type_str != "cifar-10" &&
        dataset_type_str != "cifar100" && dataset_type_str != "cifar-100") {
        std::cerr << "Error: Unsupported dataset type: " << dataset_type_str << "\n";
        std::cerr << "Supported: mnist, cifar10, cifar100\n";
        return 1;
    }

    // 验证format参数
    std::transform(format_arg.begin(), format_arg.end(), format_arg.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (format_arg != "raw" && format_arg != "dts") {
        std::cerr << "Error: --format must be 'raw' or 'dts'\n";
        return 1;
    }
    bool use_dts = (format_arg == "dts");

    // 验证mode参数
    std::transform(mode_arg.begin(), mode_arg.end(), mode_arg.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    bool partial_mode;
    if (mode_arg == "partial") {
        partial_mode = true;
    } else if (mode_arg == "fully") {
        partial_mode = false;
    } else {
        std::cerr << "Error: --mode must be 'partial' or 'fully'\n";
        return 1;
    }

    // 验证PO参数（PO1~3都是可选的）
    if (po_val1.empty()) {
        po_val1 = po_train1;
    }
    if (po_val2.empty()) {
        po_val2 = po_train2;
    }
    if (po_val3.empty()) {
        po_val3 = po_train3;
    }

    // 获取数据集原始尺寸
    int dataset_base_size = 0;
    if (dataset_type_str == "mnist") {
        dataset_base_size = 28;
    } else if (dataset_type_str.substr(0, 5) == "cifar") {
        dataset_base_size = 32;
    }

    // 计算max_resolution（链式计算PO1->PO2->PO3的最终输出）
    std::cout << "\n=== Calculating Max Resolution ===\n";
    std::cout << "Dataset base size: " << dataset_base_size << "\n";
    int train_final_output = calculate_po_chain_final_output(
        dataset_base_size, po_train1, po_train2, po_train3, resolution);
    int val_final_output = calculate_po_chain_final_output(
        dataset_base_size, po_val1, po_val2, po_val3, resolution);
    int calculated_max_resolution = std::max(train_final_output, val_final_output);
    std::cout << "Train PO chain final output: " << train_final_output << "\n";
    std::cout << "Val PO chain final output: " << val_final_output << "\n";
    std::cout << "Calculated max_resolution: " << calculated_max_resolution << "\n";

    // 输出配置信息
    std::cout << "\n=== Configuration Summary ===\n";
    std::cout << "Dataset: " << dataset_type_str;
    std::cout << "\nPath: " << dataset_path << "\n";
    std::cout << "Load workers: " << num_load_workers << "\n";
    std::cout << "Preprocess workers: " << num_preproc_workers << "\n";
    std::cout << "Shuffle train: " << (shuffle_train ? "enabled" : "disabled") << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "SDMP factor: " << sdmp_factor << "\n";
    std::cout << "CPVS: " << (using_cpvs ? "enabled" : "disabled") << "\n";
    std::cout << "Device: " << device_type << "\n";
    std::cout << "Test mode: false (NORMAL mode with EngineBuffer)\n";

    // 创建PO对象（所有PO都是可选的，缺失则用DoNothing）
    std::cout << "\n=== Setting Transforms ===\n";

    // Train PO 1~3（可选）
    std::unique_ptr<PreprocessOperation> train_po1, train_po2, train_po3;
    if (!po_train1.empty()) {
        std::cout << "Train PO 1: " << po_train1 << "\n";
        train_po1 = is_resize_or_crop_po(po_train1)
            ? create_resize_crop_po(po_train1, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_train1, scale_min, scale_max, ratio_min, ratio_max, flip_prob);
        if (!train_po1) return 1;
    } else {
        std::cout << "Train PO 1: DoNothing (default)\n";
        train_po1 = std::make_unique<DoNothing>();
    }

    if (!po_train2.empty()) {
        std::cout << "Train PO 2: " << po_train2 << "\n";
        train_po2 = is_resize_or_crop_po(po_train2)
            ? create_resize_crop_po(po_train2, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_train2, scale_min, scale_max, ratio_min, ratio_max, flip_prob);
        if (!train_po2) return 1;
    } else {
        std::cout << "Train PO 2: DoNothing (default)\n";
        train_po2 = std::make_unique<DoNothing>();
    }

    if (!po_train3.empty()) {
        std::cout << "Train PO 3: " << po_train3 << "\n";
        train_po3 = is_resize_or_crop_po(po_train3)
            ? create_resize_crop_po(po_train3, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_train3, scale_min, scale_max, ratio_min, ratio_max, flip_prob);
        if (!train_po3) return 1;
    } else {
        std::cout << "Train PO 3: DoNothing (default)\n";
        train_po3 = std::make_unique<DoNothing>();
    }

    // Val PO 1~3（可选）
    std::unique_ptr<PreprocessOperation> val_po1, val_po2, val_po3;
    if (!po_val1.empty()) {
        std::cout << "Val PO 1: " << po_val1 << "\n";
        val_po1 = is_resize_or_crop_po(po_val1)
            ? create_resize_crop_po(po_val1, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_val1, scale_min, scale_max, ratio_min, ratio_max, flip_prob);
        if (!val_po1) return 1;
    } else {
        std::cout << "Val PO 1: DoNothing (default)\n";
        val_po1 = std::make_unique<DoNothing>();
    }

    if (!po_val2.empty()) {
        std::cout << "Val PO 2: " << po_val2 << "\n";
        val_po2 = is_resize_or_crop_po(po_val2)
            ? create_resize_crop_po(po_val2, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_val2, scale_min, scale_max, ratio_min, ratio_max, flip_prob);
        if (!val_po2) return 1;
    } else {
        std::cout << "Val PO 2: DoNothing (default)\n";
        val_po2 = std::make_unique<DoNothing>();
    }

    if (!po_val3.empty()) {
        std::cout << "Val PO 3: " << po_val3 << "\n";
        val_po3 = is_resize_or_crop_po(po_val3)
            ? create_resize_crop_po(po_val3, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_val3, scale_min, scale_max, ratio_min, ratio_max, flip_prob);
        if (!val_po3) return 1;
    } else {
        std::cout << "Val PO 3: DoNothing (default)\n";
        val_po3 = std::make_unique<DoNothing>();
    }

    std::cout << "Random seed: " << random_seed << "\n";

    // 获取Preprocessor实例（后续调用train()和val()需要）
    auto& prep = Preprocessor::instance();

    // 设置current_resolution（必须在Setup之前设置）
    GlobalRegistry::instance().set_current_resolution_train(resolution);
    GlobalRegistry::instance().set_current_resolution_val(resolution);

    // 使用Setup构建器配置Preprocessor（支持变参模板，最多3个PO）
    Preprocessor::setup()
        .dataset(dataset_type_str, dataset_path)
        .using_dts_format(use_dts, 0)
        .batch_size(batch_size)
        .max_output_resolution(calculated_max_resolution)
        .color_channels(3)
        .load_workers(num_load_workers)
        .preprocess_workers(num_preproc_workers)
        .fully_mode(!partial_mode)
        .shuffle_train(shuffle_train)
        .download(false)
        .sdmp_factor(sdmp_factor)
        .using_cpvs(using_cpvs)
        .cpu_binding(false)
        .train_transforms(*train_po1, *train_po2, *train_po3)
        .val_transforms(*val_po1, *val_po2, *val_po3)
        .commit();

    // 设置全局随机种子
    manual_seed(random_seed);

    // 步骤5：运行多个epoch
    std::cout << "\n=== Running " << num_epochs << " Epoch(s) ===\n\n";

    for (int epoch = 0; epoch < num_epochs + 1; ++epoch) {

        // 设置当前epoch ID到GlobalRegistry
        GlobalRegistry::instance().set_user_epoch_id(epoch);

        // 如果是val-first模式，epoch=0时先运行验证
        if (epoch == 0) {
            if (val_first) {
                std::cout << "========== Epoch " << epoch << " ==========\n";
                std::cout << "\n[VAL - val-first mode]\n";
                prep.val();
            }
            continue;
        }
        std::cout << "========== Epoch " << epoch << " ==========\n";

        // 训练阶段
        std::cout << "\n[TRAIN]\n";
        prep.train();

        // 验证阶段
        std::cout << "\n[VAL]\n";
        prep.val();
    }

    // 汇总
    std::cout << "\n========================================\n";
    std::cout << "=== Test Completed Successfully ===\n";
    std::cout << "========================================\n";
    std::cout << "CRC files saved to: " TR_WORKSPACE "/\n\n";

    // 【V3.24.1更新】使用正常的return 0，触发Initializer::ScopeGuard析构
    // Initializer会按正确顺序清理所有单例，避免析构顺序问题
    std::cout.flush();
    return 0;

    // 【旧版本MSYS2兼容性修复】：
    // 在MSYS2环境中，mimalloc的atexit handler可能与ucrtbase冲突
    // 过去使用 _Exit(0) 跳过析构，现在Initializer解决了析构顺序问题
    // _Exit(0);  // 旧方案：跳过atexit handler
}
