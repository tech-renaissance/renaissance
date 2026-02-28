/**
 * @file test_pw_ultimate.cpp
 * @brief PW综合测试：普通模式（非test mode），完整测试Preprocessor功能
 * @version 1.0.0
 * @date 2026-02-21
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
              << "  --dataset <TYPE>    Dataset type: imagenet, mnist, cifar10, cifar100\n"
              << "  --path <PATH>       Dataset root path\n\n"
              << "Optional Options:\n"
              << "  --format <fmt>       Dataset format: raw, dts (default: raw)\n"
              << "  --mode <MODE>        DataLoader mode: fully, partial (default: partial)\n"
              << "  --shuffle            Enable shuffle train data (default: disabled)\n"
              << "  --lv <0-3>           DTS compression level (default: 0)\n"
              << "  --cpu-bind           Enable CPU binding (default: disabled)\n"
              << "  --batch-size <N>     Batch size (default: 256)\n"
              << "  --cpvs <true/false>  Enable CPVS (default: false)\n"
              << "  --sdmp <N>           SDMP factor (default: 1)\n"
              << "  --reproducible       Enable reproducible mode (default: disabled)\n"
              << "  --epoch <N>          Number of epochs to run (default: 1)\n"
              << "  --resolution <N>     Output resolution for resize/crop operations (default: 224)\n"
              << "                      This parameter is passed to resize/crop POs only.\n"
              << "                      The max_resolution for config_preprocessor is inferred\n"
              << "                      from the final output size of train and val transforms.\n"
              << "  --po-train1 <PO>     First train PO (MUST be resize/crop):\n"
              << "                      Resize, CenterCrop, RandomResizedCrop, FastRandomResizedCrop, RandomCrop\n"
              << "  --po-train2 <PO>     Second train PO (optional):\n"
              << "                      - If resize/crop: uses --resolution parameter\n"
              << "                      - If other: infers output from previous PO\n"
              << "                      Supported: RandomHorizontalFlip, ColorJitter, RandomRotation,\n"
              << "                                  RandomAutocontrast, RandomBrightness, GaussianBlur, GaussianNoise,\n"
              << "                                  RandomGrayscale, Pad, DoNothing\n"
              << "  --po-val1 <PO>       First val PO (MUST be resize/crop, default: same as po-train1)\n"
              << "  --po-val2 <PO>       Second val PO (optional, same rules as po-train2)\n"
              << "  --loaders <N>        Number of load workers (default: 16)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 16)\n"
              << "  --device <TYPE>      Device type: CPU, GPU (default: CPU)\n"
              << "  --gpu-ids <IDS>      GPU IDs (e.g., \"0,1,2,3\" or \"0-7\", default: \"0\")\n"
              << "  --seed <N>           Random seed (default: 42)\n"
              << "  --help               Show this help message\n\n"
              << "PO Parameters (RandomResizedCrop/FastRandomResizedCrop):\n"
              << "  --scale-min <F>      Min scale (default: 0.08)\n"
              << "  --scale-max <F>      Max scale (default: 1.0)\n"
              << "  --ratio-min <F>      Min ratio (default: 0.75)\n"
              << "  --ratio-max <F>      Max ratio (default: 1.333)\n"
              << "PO Parameters (RandomHorizontalFlip):\n"
              << "  --prob <F>           Flip probability (default: 0.5)\n"
              << "PO Parameters (ColorJitter):\n"
              << "  --brightness <F>     Brightness jitter (default: 0.0)\n"
              << "  --contrast <F>       Contrast jitter (default: 0.0)\n"
              << "  --saturation <F>     Saturation jitter (default: 0.0)\n"
              << "  --hue <F>            Hue jitter, range [0, 0.5] (default: 0.0)\n"
              << "PO Parameters (RandomRotation):\n"
              << "  --degrees <F>        Rotation angle range in degrees (default: 30.0)\n"
              << "  --fill <N>           Fill value for areas outside the image (default: 0)\n"
              << "PO Parameters (RandomAutocontrast):\n"
              << "  --autocontrast-p <F> Probability of applying autocontrast (default: 0.5)\n"
              << "PO Parameters (RandomBrightness):\n"
              << "  WARNING: RandomBrightness is NOT recommended for datasets other than MNIST!\n"
              << "  Shift range is hardcoded to [-7, 7], no configurable parameters.\n"
              << "PO Parameters (GaussianBlur):\n"
              << "  --blur-sigma-min <F> Minimum sigma for Gaussian blur (default: 0.1)\n"
              << "  --blur-sigma-max <F> Maximum sigma for Gaussian blur (default: 2.0)\n"
              << "PO Parameters (RandomGrayscale):\n"
              << "  --grayscale-p <F>   Probability of converting to grayscale (default: 0.1)\n"
              << "PO Parameters (GaussianNoise):\n"
              << "  --noise-mean <F>    Mean of Gaussian noise (default: 0.0)\n"
              << "  --noise-sigma <F>   Standard deviation of Gaussian noise (default: 25.5)\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dataset imagenet --path /data/imagenet --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip\n"
              << "  " << program_name << " --dataset imagenet --path /data/imagenet --po-train1 FastRandomResizedCrop --sdmp 2 --epoch 2\n"
              << "  " << program_name << " --dataset imagenet --path /data/imagenet --po-train1 RandomResizedCrop --po-train2 GaussianBlur --resolution 256\n"
              << "  " << program_name << " --dataset mnist --path /data/mnist --format raw --mode fully --po-train1 Resize\n"
              << "  " << program_name << " --dataset cifar10 --path /data/cifar --po-train1 CenterCrop --sdmp 3 --epoch 2\n"
              << "  " << program_name << " --dataset imagenet --path /data/imagenet --po-train1 RandomResizedCrop --reproducible\n\n"
              << "Note: This test uses Preprocessor NORMAL mode (with EngineBuffer)\n\n"
              << "Command Sample: /root/epfs/R/renaissance/build/bin/tests/pw/test_progressive_resolution --path /root/epfs/dataset/mnist --reproducible\n"
              << "Command Sample: /root/epfs/R/renaissance/build/bin/tests/pw/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode fully --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids \"0,1,2,3,4,5,6,7\" --epoch 28 --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true\n\n";
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
 * @param po_name PO名称
 * @param resolution 分辨率（仅用于resize/crop类PO）
 * @param scale_min, scale_max, ratio_min, ratio_max RandomResizedCrop参数
 * @param flip_prob RandomHorizontalFlip概率
 * @param brightness, contrast, saturation, hue ColorJitter参数
 * @param degrees, fill RandomRotation参数
 * @param autocontrast_p RandomAutocontrast概率
 * @param blur_sigma_min, blur_sigma_max GaussianBlur参数
 * @param grayscale_p RandomGrayscale概率
 * @param noise_mean, noise_sigma GaussianNoise参数
 * @return PO对象，如果类型未知则返回nullptr
 */
std::unique_ptr<PreprocessOperation> create_po(const std::string& po_name,
                                                   float scale_min, float scale_max,
                                                   float ratio_min, float ratio_max,
                                                   float flip_prob,
                                                   float brightness, float contrast,
                                                   float saturation, float hue,
                                                   float degrees, int fill,
                                                   float autocontrast_p,
                                                   float blur_sigma_min, float blur_sigma_max,
                                                   float grayscale_p,
                                                   float noise_mean, float noise_sigma) {
    std::string po_lower = po_name;
    std::transform(po_lower.begin(), po_lower.end(), po_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (po_lower == "resize") {
        std::cerr << "Error: PO '" << po_name << "' requires --resolution parameter\n";
        return nullptr;
    } else if (po_lower == "centercrop") {
        std::cerr << "Error: PO '" << po_name << "' requires --resolution parameter\n";
        return nullptr;
    } else if (po_lower == "randomresizedcrop") {
        std::cerr << "Error: PO '" << po_name << "' requires --resolution parameter\n";
        return nullptr;
    } else if (po_lower == "fastrandomresizedcrop") {
        std::cerr << "Error: PO '" << po_name << "' requires --resolution parameter\n";
        return nullptr;
    } else if (po_lower == "randomcrop") {
        std::cerr << "Error: PO '" << po_name << "' requires --resolution parameter\n";
        return nullptr;
    } else if (po_lower == "randomhorizontalflip") {
        return std::make_unique<RandomHorizontalFlip>(flip_prob);
    } else if (po_lower == "colorjitter") {
        return std::make_unique<ColorJitter>(brightness, contrast, saturation, hue);
    } else if (po_lower == "randomrotation") {
        return std::make_unique<RandomRotation>(degrees, static_cast<uint8_t>(fill));
    } else if (po_lower == "randomautocontrast") {
        return std::make_unique<RandomAutocontrast>(autocontrast_p);
    } else if (po_lower == "randombrightness") {
        return std::make_unique<RandomBrightness>();
    } else if (po_lower == "gaussianblur") {
        return std::make_unique<GaussianBlur>(blur_sigma_min, blur_sigma_max);
    } else if (po_lower == "randomgrayscale") {
        return std::make_unique<RandomGrayscale>(grayscale_p);
    } else if (po_lower == "gaussiannoise") {
        return std::make_unique<GaussianNoise>(noise_mean, noise_sigma, true);
    } else if (po_lower == "pad") {
        return std::make_unique<Pad>(4, std::vector<int>{0}, PaddingMode::CONSTANT);
    } else if (po_lower == "donothing") {
        return std::make_unique<DoNothing>();
    } else {
        std::cerr << "Error: Unknown PO type: " << po_name << "\n";
        std::cerr << "Supported: Resize, CenterCrop, RandomResizedCrop, FastRandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, RandomAutocontrast, RandomBrightness, GaussianBlur, GaussianNoise, RandomGrayscale, Pad, RandomCrop, DoNothing\n";
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
        std::cerr << "Resize/crop operations: Resize, CenterCrop, RandomResizedCrop, FastRandomResizedCrop, RandomCrop\n";
        return nullptr;
    }
}

/**
 * @brief 计算 PO 链的最终输出尺寸
 * @param po_name1 第一个PO名称（必须是resize/crop）
 * @param po_name2 第二个PO名称（可选，空字符串表示无）
 * @param resolution resolution参数（用于resize/crop类PO）
 * @param padding_ padding参数（用于Pad类PO）
 * @return 最终输出尺寸
 */
int calculate_po_chain_final_output(const std::string& po_name1,
                                     const std::string& po_name2,
                                     int resolution,
                                     int padding_ = 4) {
    std::string po1_lower = po_name1;
    std::transform(po1_lower.begin(), po1_lower.end(), po1_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // 第一个PO必须是resize/crop，输出就是resolution
    int current_size = resolution;

    // 如果没有第二个PO，返回当前尺寸
    if (po_name2.empty()) {
        return current_size;
    }

    // 处理第二个PO
    std::string po2_lower = po_name2;
    std::transform(po2_lower.begin(), po2_lower.end(), po2_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // 如果第二个PO是resize/crop类，输出是resolution
    if (po2_lower == "resize" || po2_lower == "centercrop" ||
        po2_lower == "randomresizedcrop" || po2_lower == "fastrandomresizedcrop" ||
        po2_lower == "randomcrop") {
        return resolution;
    }
    // 如果是Pad，输出是 input_size + 2*padding
    else if (po2_lower == "pad") {
        return current_size + 2 * padding_;
    }
    // 其他PO不改变尺寸
    else {
        return current_size;
    }
}

int main(int argc, char* argv[]) {
    // 默认配置
    std::string dataset_type_str;
    std::string dataset_path;
    std::string format_arg = "raw";
    std::string mode_arg = "partial";
    int compression_level = 0;
    bool shuffle_train = false;  // 是否打乱训练集数据
    bool auto_cpu_binding = false;
    int batch_size = 256;
    bool using_cpvs = false;
    int sdmp_factor = 1;
    bool reproducible = false;  // 可复现模式标志
    int num_epochs = 1;
    int resolution = 224;
    std::string po_train1, po_train2;
    std::string po_val1, po_val2;
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

    // ColorJitter参数
    float brightness = 0.2f;  // 亮度变化范围 [0.8, 1.2]
    float contrast = 0.2f;    // 对比度变化范围 [0.8, 1.2]
    float saturation = 0.2f;  // 饱和度变化范围 [0.8, 1.2]
    float hue = 0.1f;         // 色调变化范围 [-0.1, 0.1]

    // RandomRotation参数
    float degrees = 30.0f;    // 旋转角度范围 [-30, 30]
    int fill = 0;             // 填充值（黑色）

    // RandomAutocontrast参数
    float autocontrast_p = 0.5f;  // 自动对比度概率

    // GaussianBlur参数
    float blur_sigma_min = 0.1f;   // 高斯模糊最小sigma
    float blur_sigma_max = 2.0f;   // 高斯模糊最大sigma

    // RandomGrayscale参数
    float grayscale_p = 0.1f;      // 灰度化概率（PyTorch默认值）

    // GaussianNoise参数
    float noise_mean = 0.0f;   // 噪声均值
    float noise_sigma = 25.5f; // 噪声标准差（相当于[0,1]范围的0.1）

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
        } else if (arg == "--shuffle") {
            shuffle_train = true;
        } else if (arg == "--lv" && i + 1 < argc) {
            compression_level = std::stoi(argv[++i]);
            if (compression_level < 0 || compression_level > 3) {
                std::cerr << "Error: --lv must be between 0 and 3\n";
                return 1;
            }
        } else if (arg == "--cpu-bind") {
            auto_cpu_binding = true;
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--cpvs" && i + 1 < argc) {
            std::string cpvs_str = argv[++i];
            std::transform(cpvs_str.begin(), cpvs_str.end(), cpvs_str.begin(), ::tolower);
            using_cpvs = (cpvs_str == "true");
        } else if (arg == "--sdmp" && i + 1 < argc) {
            sdmp_factor = std::stoi(argv[++i]);
        } else if (arg == "--reproducible") {
            reproducible = true;
        } else if (arg == "--epoch" && i + 1 < argc) {
            num_epochs = std::stoi(argv[++i]);
        } else if (arg == "--resolution" && i + 1 < argc) {
            resolution = std::stoi(argv[++i]);
        } else if (arg == "--po-train1" && i + 1 < argc) {
            po_train1 = argv[++i];
        } else if (arg == "--po-train2" && i + 1 < argc) {
            po_train2 = argv[++i];
        } else if (arg == "--po-val1" && i + 1 < argc) {
            po_val1 = argv[++i];
        } else if (arg == "--po-val2" && i + 1 < argc) {
            po_val2 = argv[++i];
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
        } else if (arg == "--brightness" && i + 1 < argc) {
            brightness = std::stof(argv[++i]);
        } else if (arg == "--contrast" && i + 1 < argc) {
            contrast = std::stof(argv[++i]);
        } else if (arg == "--saturation" && i + 1 < argc) {
            saturation = std::stof(argv[++i]);
        } else if (arg == "--hue" && i + 1 < argc) {
            hue = std::stof(argv[++i]);
        } else if (arg == "--degrees" && i + 1 < argc) {
            degrees = std::stof(argv[++i]);
        } else if (arg == "--fill" && i + 1 < argc) {
            fill = std::stoi(argv[++i]);
        } else if (arg == "--autocontrast-p" && i + 1 < argc) {
            autocontrast_p = std::stof(argv[++i]);
        } else if (arg == "--blur-sigma-min" && i + 1 < argc) {
            blur_sigma_min = std::stof(argv[++i]);
        } else if (arg == "--blur-sigma-max" && i + 1 < argc) {
            blur_sigma_max = std::stof(argv[++i]);
        } else if (arg == "--grayscale-p" && i + 1 < argc) {
            grayscale_p = std::stof(argv[++i]);
        } else if (arg == "--noise-mean" && i + 1 < argc) {
            noise_mean = std::stof(argv[++i]);
        } else if (arg == "--noise-sigma" && i + 1 < argc) {
            noise_sigma = std::stof(argv[++i]);
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 【第一句】初始化框架（必须在所有其他操作之前）
    INIT_FRAMEWORK(device_type);
    std::cout << "Framework initialized with device: " << device_type << "\n";

    // 【第二句】设置可复现性保险（必须在所有其他操作之前）
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

    // 验证dataset类型
    std::transform(dataset_type_str.begin(), dataset_type_str.end(), dataset_type_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    DatasetType dataset_type;
    if (dataset_type_str == "imagenet") {
        dataset_type = DatasetType::imagenet;
    } else if (dataset_type_str == "mnist") {
        dataset_type = DatasetType::mnist;
    } else if (dataset_type_str == "cifar10" || dataset_type_str == "cifar-10") {
        dataset_type = DatasetType::cifar_10;
    } else if (dataset_type_str == "cifar100" || dataset_type_str == "cifar-100") {
        dataset_type = DatasetType::cifar_100;
    } else {
        std::cerr << "Error: Unsupported dataset type: " << dataset_type_str << "\n";
        std::cerr << "Supported: imagenet, mnist, cifar10, cifar100\n";
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

    // 验证PO参数
    if (po_train1.empty()) {
        std::cerr << "Error: --po-train1 is required\n";
        return 1;
    }

    // 验证PO1必须是resize/crop类（ImageNet的第一个PO必须是resize/crop）
    if (!is_resize_or_crop_po(po_train1)) {
        std::cerr << "Error: --po-train1 must be a resize/crop operation (Resize, CenterCrop, RandomResizedCrop, FastRandomResizedCrop, RandomCrop)\n";
        std::cerr << "Got: " << po_train1 << "\n";
        return 1;
    }

    if (po_val1.empty()) {
        po_val1 = po_train1;  // 默认同train
    }

    // 验证val PO1也必须是resize/crop类
    if (!is_resize_or_crop_po(po_val1)) {
        std::cerr << "Error: --po-val1 must be a resize/crop operation (Resize, CenterCrop, RandomResizedCrop, FastRandomResizedCrop, RandomCrop)\n";
        std::cerr << "Got: " << po_val1 << "\n";
        return 1;
    }

    // 计算正确的max_resolution（基于PO链最终输出）
    std::cout << "\n=== Calculating Max Resolution ===\n";
    int train_final_output = calculate_po_chain_final_output(po_train1, po_train2, resolution);
    int val_final_output = calculate_po_chain_final_output(po_val1, po_val2, resolution);
    int calculated_max_resolution = std::max(train_final_output, val_final_output);

    std::cout << "Train PO chain final output: " << train_final_output << "\n";
    std::cout << "Val PO chain final output: " << val_final_output << "\n";
    std::cout << "Calculated max_resolution: " << calculated_max_resolution << "\n";

    // 输出配置信息
    std::cout << "\n=== Configuration Summary ===\n";
    std::cout << "Dataset: " << dataset_type_str;
    std::cout << "\nFormat: " << (use_dts ? "DTS" : "RAW");
    if (use_dts) {
        std::cout << " LV" << compression_level;
    }
    std::cout << "\nPath: " << dataset_path << "\n";
    std::cout << "Load workers: " << num_load_workers << "\n";
    std::cout << "Preprocess workers: " << num_preproc_workers << "\n";
    std::cout << "Mode: " << (partial_mode ? "Partial" : "Fully") << "\n";
    std::cout << "Shuffle train: " << (shuffle_train ? "enabled" : "disabled") << "\n";
    std::cout << "Resolution parameter: " << resolution << "\n";
    std::cout << "Calculated max_resolution: " << calculated_max_resolution << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "SDMP factor: " << sdmp_factor << "\n";
    std::cout << "CPVS: " << (using_cpvs ? "enabled" : "disabled") << "\n";
    std::cout << "Device: " << device_type << "\n";
    if (device_type == "GPU" && !gpu_ids_str.empty()) {
        std::cout << "GPU IDs: " << gpu_ids_str << "\n";
    }
    std::cout << "CPU binding: " << (auto_cpu_binding ? "enabled" : "disabled") << "\n";
    std::cout << "Test mode: false (NORMAL mode with EngineBuffer)\n";

    // 创建PO对象
    std::cout << "\n=== Setting Transforms ===\n";

    // Train PO 1（必需）
    std::cout << "Train PO 1: " << po_train1 << " (" << resolution << ")\n";
    auto train_po1 = create_resize_crop_po(po_train1, resolution, scale_min, scale_max,
                                           ratio_min, ratio_max);
    if (!train_po1) return 1;

    // Train PO 2（可选，如果为空则使用DoNothing）
    std::unique_ptr<PreprocessOperation> train_po2;
    if (po_train2.empty()) {
        std::cout << "Train PO 2: DoNothing (default)\n";
        train_po2 = std::make_unique<DoNothing>();
    } else {
        std::cout << "Train PO 2: " << po_train2 << "\n";
        train_po2 = is_resize_or_crop_po(po_train2)
            ? create_resize_crop_po(po_train2, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_train2, scale_min, scale_max, ratio_min, ratio_max, flip_prob,
                       brightness, contrast, saturation, hue, degrees, fill, autocontrast_p,
                       blur_sigma_min, blur_sigma_max, grayscale_p, noise_mean, noise_sigma);
        if (!train_po2) return 1;
    }

    // Val PO 1（必需）
    std::cout << "Val PO 1: " << po_val1 << " (" << resolution << ")\n";
    auto val_po1 = create_resize_crop_po(po_val1, resolution, scale_min, scale_max,
                                         ratio_min, ratio_max);
    if (!val_po1) return 1;

    // Val PO 2（可选，如果为空则使用DoNothing）
    std::unique_ptr<PreprocessOperation> val_po2;
    if (po_val2.empty()) {
        std::cout << "Val PO 2: DoNothing (default)\n";
        val_po2 = std::make_unique<DoNothing>();
    } else {
        std::cout << "Val PO 2: " << po_val2 << "\n";
        val_po2 = is_resize_or_crop_po(po_val2)
            ? create_resize_crop_po(po_val2, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_val2, scale_min, scale_max, ratio_min, ratio_max, flip_prob,
                       brightness, contrast, saturation, hue, degrees, fill, autocontrast_p,
                       blur_sigma_min, blur_sigma_max, grayscale_p, noise_mean, noise_sigma);
        if (!val_po2) return 1;
    }

    std::cout << "Random seed: " << random_seed << "\n";

    // 获取Preprocessor实例（后续调用train()和val()需要）
    auto& prep = Preprocessor::instance();

    // 使用Setup构建器配置Preprocessor（统一调用，无需分支）
    Preprocessor::setup()
        .dataset(dataset_type_str, dataset_path)
        .using_dts_format(use_dts, compression_level)
        .batch_size(batch_size)
        .max_output_resolution(calculated_max_resolution)
        .color_channels(3)
        .num_load_workers(num_load_workers)
        .num_preproc_workers(num_preproc_workers)
        .partial_mode(partial_mode)
        .shuffle_train(shuffle_train)
        .download(false)
        .sdmp_factor(sdmp_factor)
        .using_cpvs(using_cpvs)
        .pw_test_mode(false)
        .cpu_binding(auto_cpu_binding)
        .device(device_type)
        .train_transforms(*train_po1, *train_po2)
        .val_transforms(*val_po1, *val_po2)
        .commit();

    // 设置全局随机种子
    manual_seed(random_seed);

    // 步骤5：运行多个epoch
    std::cout << "\n=== Running " << num_epochs << " Epoch(s) ===\n\n";

    double total_train_time = 0.0;
    double total_val_time = 0.0;
    size_t total_train_samples = 0;
    size_t total_val_samples = 0;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "========== Epoch " << epoch << " ==========\n";

        // Train phase
        std::cout << "\n[TRAIN]\n";
        auto start_train = std::chrono::high_resolution_clock::now();

        prep.train();

        auto end_train = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_time = end_train - start_train;
        total_train_time += train_time.count();

        auto stats = prep.get_stats();
        total_train_samples += stats.total_samples;
        std::cout << "  Time: " << std::fixed << std::setprecision(3)
                  << train_time.count() << " s"
                  << ", Samples: " << stats.total_samples
                  << ", Throughput: " << std::fixed << std::setprecision(1)
                  << (static_cast<double>(stats.total_samples) / train_time.count())
                  << " samples/s\n";

        // Val phase
        std::cout << "\n[VAL]\n";
        auto start_val = std::chrono::high_resolution_clock::now();

        prep.val();

        auto end_val = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> val_time = end_val - start_val;
        total_val_time += val_time.count();

        stats = prep.get_stats();
        total_val_samples += stats.total_samples;
        std::cout << "  Time: " << std::fixed << std::setprecision(3)
                  << val_time.count() << " s"
                  << ", Samples: " << stats.total_samples
                  << ", Throughput: " << std::fixed << std::setprecision(1)
                  << (static_cast<double>(stats.total_samples) / val_time.count())
                  << " samples/s\n";
    }

    // 汇总
    std::cout << "\n========================================\n";
    std::cout << "=== FINAL SUMMARY ===\n";
    std::cout << "========================================\n";
    std::cout << "Total train time: " << std::fixed << std::setprecision(3)
              << total_train_time << " s (" << total_train_samples << " samples)\n";
    std::cout << "Total val time: " << std::fixed << std::setprecision(3)
              << total_val_time << " s (" << total_val_samples << " samples)\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(3)
              << (total_train_time + total_val_time) << " s\n";
    std::cout << "Avg train time: " << std::fixed << std::setprecision(3)
              << (total_train_time / num_epochs) << " s\n";
    std::cout << "Avg val time: " << std::fixed << std::setprecision(3)
              << (total_val_time / num_epochs) << " s\n";
    std::cout << "Avg epoch time: " << std::fixed << std::setprecision(3)
              << ((total_train_time + total_val_time) / num_epochs) << " s\n";
    std::cout << "\n=== Test Completed Successfully ===\n\n";

    return 0;
}
