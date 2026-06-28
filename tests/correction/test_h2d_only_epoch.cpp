/**
 * @file test_h2d_only_epoch.cpp
 * @brief H2D-Only Epoch 测试：测量 Preprocessor → TransferStation → cudaMemcpy 每 epoch 耗时
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/correction
 *
 * 设计原则：
 * - 直接复用 test_pw_ultimate.cpp 的 PO 工厂函数（create_po / create_resize_crop_po）
 * - 使用 compile_h2d_only() / run_h2d_only() —— GPU 消费者真实存在，正常走读写信号量
 * - 不使用 FOR_TRANSFER_STATION_UNIT_TESTS_ONLY 宏
 * - BluePrint 自动适配：CIFAR-10 → fc(10), ImageNet → fc(1000)
 * - 输出包含 per-rank 和聚合带宽
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <algorithm>
#include <vector>

#ifdef _WIN32
    #ifdef min
        #undef min
    #endif
    #ifdef max
        #undef max
    #endif
#endif

using namespace tr;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --dataset <TYPE>    Dataset type: imagenet, cifar10\n"
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
              << "  --resolution <N>     Output resolution for resize/crop (default: 224)\n"
              << "  --po-train1 <PO>     First train PO (MUST be resize/crop):\n"
              << "                      Resize, CenterCrop, RandomResizedCrop, FastRandomResizedCrop, RandomCrop\n"
              << "  --po-train2 <PO>     Second train PO (optional)\n"
              << "  --po-val1 <PO>       First val PO (default: same as po-train1)\n"
              << "  --po-val2 <PO>       Second val PO (optional)\n"
              << "  --normalization <M>  Normalization mode (default: mlperf)\n"
              << "  --loaders <N>        Number of load workers (default: 16)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 16)\n"
              << "  --device <TYPE>      Device type: CPU, GPU (default: CPU)\n"
              << "  --gpu-ids <IDS>      GPU IDs (e.g., \"0,1,2,3\")\n"
              << "  --amp                Enable Automatic Mixed Precision (AMP) mode\n"
              << "  --seed <N>           Random seed (default: 42)\n"
              << "  --epoch <N>          Number of epochs to run (default: 1)\n"
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
              << "  --hue <F>            Hue jitter (default: 0.0)\n"
              << "PO Parameters (RandomRotation):\n"
              << "  --degrees <F>        Rotation degrees (default: 30.0)\n"
              << "  --fill <N>           Fill value (default: 0)\n"
              << "PO Parameters (RandomAutocontrast):\n"
              << "  --autocontrast-p <F> Probability (default: 0.5)\n"
              << "PO Parameters (GaussianBlur):\n"
              << "  --blur-sigma-min <F> Min sigma (default: 0.1)\n"
              << "  --blur-sigma-max <F> Max sigma (default: 2.0)\n"
              << "PO Parameters (RandomGrayscale):\n"
              << "  --grayscale-p <F>    Probability (default: 0.1)\n"
              << "PO Parameters (GaussianNoise):\n"
              << "  --noise-mean <F>     Mean (default: 0.0)\n"
              << "  --noise-sigma <F>    Stddev (default: 25.5)\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dataset imagenet --path /data/imagenet --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --normalization imagenet --device GPU --gpu-ids \"0,1,2,3\"\n"
              << "  " << program_name << " --dataset cifar10 --path /data/cifar10 --po-train1 CenterCrop --resolution 32 --normalization cifar --device GPU --gpu-ids \"0\"\n\n";
}

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
    (void)scale_min; (void)scale_max; (void)ratio_min; (void)ratio_max;
    (void)brightness; (void)contrast; (void)saturation; (void)hue;
    (void)degrees; (void)fill; (void)autocontrast_p;
    (void)blur_sigma_min; (void)blur_sigma_max; (void)grayscale_p;
    (void)noise_mean; (void)noise_sigma;

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
    } else if (po_lower == "randomscale") {
        return std::make_unique<RandomScale>(0.8f, 1.2f);
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
        std::cerr << "Supported: Resize, CenterCrop, RandomResizedCrop, FastRandomResizedCrop, "
                  << "RandomHorizontalFlip, ColorJitter, RandomRotation, RandomAutocontrast, "
                  << "RandomScale, GaussianBlur, GaussianNoise, RandomGrayscale, Pad, RandomCrop, DoNothing\n";
        return nullptr;
    }
}

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

int calculate_po_chain_final_output(const std::string& po_name1,
                                     const std::string& po_name2,
                                     int resolution,
                                     int padding_ = 4) {
    std::string po1_lower = po_name1;
    std::transform(po1_lower.begin(), po1_lower.end(), po1_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    int current_size = resolution;

    if (po_name2.empty()) {
        return current_size;
    }

    std::string po2_lower = po_name2;
    std::transform(po2_lower.begin(), po2_lower.end(), po2_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (po2_lower == "resize" || po2_lower == "centercrop" ||
        po2_lower == "randomresizedcrop" || po2_lower == "fastrandomresizedcrop" ||
        po2_lower == "randomcrop") {
        return resolution;
    } else if (po2_lower == "pad") {
        return current_size + 2 * padding_;
    } else {
        return current_size;
    }
}

NormMode parse_norm_mode(const std::string& str) {
    if (str.empty()) return NormMode::NO_NORM;

    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower == "imagenet") return NormMode::IMAGENET;
    if (lower == "mnist")    return NormMode::MNIST;
    if (lower == "cifar")    return NormMode::CIFAR;
    if (lower == "mlperf")   return NormMode::MLPERF;
    return NormMode::NO_NORM;
}

int main(int argc, char* argv[]) {
    std::string dataset_type_str;
    std::string dataset_path;
    std::string format_arg = "raw";
    std::string mode_arg = "partial";
    int compression_level = 0;
    bool shuffle_train = false;
    bool auto_cpu_binding = false;
    int batch_size = 256;
    bool using_cpvs = false;
    int sdmp_factor = 1;
    bool reproducible = false;
    int resolution = 224;
    std::string po_train1, po_train2;
    std::string po_val1, po_val2;
    std::string normalization_mode = "mlperf";
    int num_load_workers = 16;
    int num_preproc_workers = 16;
    std::string device_type = "CPU";
    std::string gpu_ids_str = "";
    bool using_amp = false;
    uint64_t random_seed = 42;
    int total_epochs_arg = 1;

    float scale_min = 0.08f;
    float scale_max = 1.0f;
    float ratio_min = 0.75f;
    float ratio_max = 1.3333333f;
    float flip_prob = 0.5f;

    float brightness = 0.2f;
    float contrast = 0.2f;
    float saturation = 0.2f;
    float hue = 0.1f;

    float degrees = 30.0f;
    int fill = 0;

    float autocontrast_p = 0.5f;

    float blur_sigma_min = 0.1f;
    float blur_sigma_max = 2.0f;

    float grayscale_p = 0.1f;

    float noise_mean = 0.0f;
    float noise_sigma = 25.5f;

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
            std::transform(cpvs_str.begin(), cpvs_str.end(), cpvs_str.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            using_cpvs = (cpvs_str == "true");
        } else if (arg == "--sdmp" && i + 1 < argc) {
            sdmp_factor = std::stoi(argv[++i]);
        } else if (arg == "--reproducible") {
            reproducible = true;
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
        } else if (arg == "--normalization" && i + 1 < argc) {
            normalization_mode = argv[++i];
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
        } else if (arg == "--amp") {
            using_amp = true;
        } else if (arg == "--seed" && i + 1 < argc) {
            random_seed = static_cast<uint64_t>(std::stoull(argv[++i]));
        } else if (arg == "--epoch" && i + 1 < argc) {
            total_epochs_arg = std::stoi(argv[++i]);
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

    std::transform(dataset_type_str.begin(), dataset_type_str.end(), dataset_type_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    DatasetType dataset_type;
    int num_classes;
    if (dataset_type_str == "imagenet") {
        dataset_type = DatasetType::imagenet;
        num_classes = 1000;
        if (resolution == 224) {} // default
    } else if (dataset_type_str == "cifar10" || dataset_type_str == "cifar-10") {
        dataset_type = DatasetType::cifar_10;
        num_classes = 10;
        if (resolution == 224) resolution = 32;
    } else {
        std::cerr << "Error: Unsupported dataset type: " << dataset_type_str << "\n";
        std::cerr << "Supported: imagenet, cifar10\n";
        return 1;
    }
    (void)dataset_type;

    NormMode norm = parse_norm_mode(normalization_mode);
    if (norm == NormMode::NO_NORM && normalization_mode != "no_norm" && !normalization_mode.empty()) {
        std::cerr << "Error: Unknown normalization mode: " << normalization_mode << "\n";
        return 1;
    }

    std::transform(format_arg.begin(), format_arg.end(), format_arg.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (format_arg != "raw" && format_arg != "dts") {
        std::cerr << "Error: --format must be 'raw' or 'dts'\n";
        return 1;
    }
    bool use_dts = (format_arg == "dts");

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

    if (po_train1.empty()) {
        std::cerr << "Error: --po-train1 is required\n";
        return 1;
    }
    if (!is_resize_or_crop_po(po_train1)) {
        std::cerr << "Error: --po-train1 must be a resize/crop operation\n";
        return 1;
    }
    if (po_val1.empty()) {
        po_val1 = po_train1;
    }

    int train_final_output = calculate_po_chain_final_output(po_train1, po_train2, resolution);
    int val_final_output = calculate_po_chain_final_output(po_val1, po_val2, resolution);
    int calculated_max_resolution = std::max(train_final_output, val_final_output);

    std::cout << "\n=== H2D-Only Epoch Test ===\n";
    std::cout << "Dataset:      " << dataset_type_str << "\n";
    std::cout << "Path:         " << dataset_path << "\n";
    std::cout << "Batch size:   " << batch_size << "\n";
    std::cout << "Resolution:   " << resolution << "\n";
    std::cout << "Max res:      " << calculated_max_resolution << "\n";
    std::cout << "AMP:          " << (using_amp ? "on" : "off") << "\n";
    std::cout << "Device:       " << device_type << "\n";
    if (!gpu_ids_str.empty()) {
        std::cout << "GPU IDs:      " << gpu_ids_str << "\n";
    }

    if (reproducible) {
        GLOBAL_SETTING.reproducible().local_batch_size(batch_size).manual_seed(random_seed);
    } else {
        GLOBAL_SETTING.local_batch_size(batch_size).manual_seed(random_seed);
    }
    if (using_amp) {
        GLOBAL_SETTING.amp(true);
    }

    if (device_type == "GPU") {
        if (!gpu_ids_str.empty()) {
            GLOBAL_SETTING.use_gpu(gpu_ids_str);
        } else {
            GLOBAL_SETTING.use_gpu();
        }
    } else {
        GLOBAL_SETTING.use_cpu();
    }

    GLOBAL_SETTING.train_resolution(calculated_max_resolution)
                 .val_resolution(calculated_max_resolution);

    auto train_po1 = create_resize_crop_po(po_train1, resolution, scale_min, scale_max,
                                           ratio_min, ratio_max);
    if (!train_po1) return 1;

    std::unique_ptr<PreprocessOperation> train_po2;
    if (po_train2.empty()) {
        train_po2 = std::make_unique<DoNothing>();
    } else {
        train_po2 = is_resize_or_crop_po(po_train2)
            ? create_resize_crop_po(po_train2, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_train2, scale_min, scale_max, ratio_min, ratio_max, flip_prob,
                       brightness, contrast, saturation, hue, degrees, fill, autocontrast_p,
                       blur_sigma_min, blur_sigma_max, grayscale_p, noise_mean, noise_sigma);
        if (!train_po2) return 1;
    }

    auto val_po1 = create_resize_crop_po(po_val1, resolution, scale_min, scale_max,
                                         ratio_min, ratio_max);
    if (!val_po1) return 1;

    std::unique_ptr<PreprocessOperation> val_po2;
    if (po_val2.empty()) {
        val_po2 = std::make_unique<DoNothing>();
    } else {
        val_po2 = is_resize_or_crop_po(po_val2)
            ? create_resize_crop_po(po_val2, resolution, scale_min, scale_max, ratio_min, ratio_max)
            : create_po(po_val2, scale_min, scale_max, ratio_min, ratio_max, flip_prob,
                       brightness, contrast, saturation, hue, degrees, fill, autocontrast_p,
                       blur_sigma_min, blur_sigma_max, grayscale_p, noise_mean, noise_sigma);
        if (!val_po2) return 1;
    }

    Preprocessor::setup()
        .dataset(dataset_type_str, dataset_path)
        .using_dts_format(use_dts, compression_level)
        .color_channels(3)
        .load_workers(num_load_workers)
        .preprocess_workers(num_preproc_workers)
        .fully_mode(!partial_mode)
        .shuffle_train(shuffle_train)
        .download(false)
        .sdmp_factor(sdmp_factor)
        .using_cpvs(using_cpvs)
        .cpu_binding(auto_cpu_binding)
        .normalization(norm)
        .train_transforms(*train_po1, *train_po2)
        .val_transforms(*val_po1, *val_po2)
        .commit();

    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)))
        .num_classes(num_classes)
        .loss(CrossEntropyLoss())
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .scheduler(StepLR().step_size(30).gamma(0.1f))
        .total_epochs(total_epochs_arg);

    task.compile_h2d_only();

    auto run_res = task.run_h2d_only();
    auto res = run_res.aggregate_train();

    std::cout << "\n=== Results ===\n";
    std::cout << "Batches:        " << res.batches << "\n";
    std::cout << std::fixed << std::setprecision(3);

    if (res.batches == 0) {
        std::cout << "Elapsed:        0.000 s (no batches)\n";
    } else {
        double elapsed_s = res.elapsed_us / 1e6;
        std::cout << "Elapsed:        " << elapsed_s << " s\n";
        std::cout << "Avg latency:    " << res.avg_lat_us / 1000.0 << " ms/batch\n";
        std::cout << "Per-rank bytes: ";
        double total_gb = static_cast<double>(res.total_bytes) / (1024.0 * 1024.0 * 1024.0);
        if (total_gb < 0.01)
            std::cout << res.total_bytes << " B";
        else
            std::cout << total_gb << " GB";
        std::cout << "\n";

        if (elapsed_s > 0.0 && res.total_bytes > 0) {
            std::cout << "Per-rank BW:    " << res.bandwidth_gbps << " GB/s\n";

            int num_ranks = GlobalRegistry::instance().world_size();
            size_t aggregate_bytes = res.total_bytes * static_cast<size_t>(num_ranks);
            double aggregate_gb = static_cast<double>(aggregate_bytes) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "Aggregate bytes: " << aggregate_gb << " GB\n";
            double aggregate_bw = static_cast<double>(aggregate_bytes) / elapsed_s;
            aggregate_bw /= (1024.0 * 1024.0 * 1024.0);
            std::cout << "Aggregate BW:   " << aggregate_bw << " GB/s\n";
        }
    }

    std::cout << "=== DONE ===\n\n";
    return 0;
}