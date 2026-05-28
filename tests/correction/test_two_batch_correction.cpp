/**
 * @file test_two_batch_correction.cpp
 * @brief Two-Batch H2D Correction Test
 * @version 1.0.0
 * @date 2026-05-25
 *
 * Design:
 *   - Compile with TR_TEST_TWO_BATCH_CORRECTION macro
 *   - Run 1 epoch with compile_h2d_only() / run_h2d_only()
 *   - Only the first 2 batches perform actual H2D transfer
 *   - Preprocessor processes all batches normally
 *   - After epoch, fetch I_A_LABEL, I_A_DATA, I_B_LABEL, I_B_DATA from every rank
 *   - Verify label range and first/last pixel non-zero
 *
 * Supports:
 *   - CPU / GPU / AMP
 *   - MNIST / CIFAR10 / IMAGENET
 *   - Multi-RANK
 */

#include "renaissance.h"
#include "renaissance/task/deep_learning_task.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>

#ifdef _WIN32
    #ifdef min
        #undef min
    #endif
    #ifdef max
        #undef max
    #endif
#endif

using namespace tr;

static float fp16_to_f32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1u;
    uint32_t exponent = static_cast<uint32_t>((h >> 10) & 0x1Fu);
    uint32_t mantissa = h & 0x3FFu;

    if (exponent == 0u) {
        if (mantissa == 0u) {
            uint32_t bits = sign << 31;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }
        float v = static_cast<float>(mantissa) * (1.0f / 1024.0f);
        v *= 6.103515625e-5f;
        return sign ? -v : v;
    }

    if (exponent == 0x1Fu) {
        uint32_t bits = (sign << 31) | 0x7F800000u | (mantissa << 13);
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    uint32_t f32_exp  = (exponent + 112u) << 23;
    uint32_t f32_mant = mantissa << 13;
    uint32_t bits = (sign << 31) | f32_exp | f32_mant;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static float read_pixel_avg(const Tensor& t, int pixel_idx) {
    int channels = t.shape().c();
    int64_t num_pixels = t.numel() / channels;
    if (pixel_idx < 0 || pixel_idx >= num_pixels) return 0.0f;
    int base = pixel_idx * channels;
    float sum = 0.0f;
    if (t.dtype() == DType::FP16) {
        const uint16_t* p = t.data<uint16_t>();
        for (int c = 0; c < channels; ++c) sum += fp16_to_f32(p[base + c]);
    } else {
        const float* p = t.data<float>();
        for (int c = 0; c < channels; ++c) sum += p[base + c];
    }
    return sum / static_cast<float>(channels);
}

static bool verify_labels(const Tensor& labels, int num_classes) {
    const int32_t* data = labels.data<int32_t>();
    int n = static_cast<int>(labels.numel());
    for (int i = 0; i < n; ++i) {
        if (data[i] < 0 || data[i] >= num_classes) return false;
    }
    return true;
}

static bool verify_first_last_pixel(const Tensor& data, const char* name) {
    int channels = data.shape().c();
    int64_t num_pixels = data.numel() / channels;
    if (num_pixels < 1) return false;
    float first = read_pixel_avg(data, 0);
    float last  = read_pixel_avg(data, static_cast<int>(num_pixels) - 1);
    const float eps = 1e-6f;
    bool ok = std::abs(first) > eps && std::abs(last) > eps;
    std::cout << "    [DEBUG " << name << "] shape=["
              << data.shape().n() << "," << data.shape().h() << ","
              << data.shape().w() << "," << data.shape().c() << "] dtype="
              << (data.dtype() == DType::FP16 ? "FP16" : "FP32")
              << " pixels=" << num_pixels
              << " first=" << first << " last=" << last
              << " -> " << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

int main(int argc, char* argv[]) {
    bool use_amp = false;
    bool use_cpu = false;
    std::string dataset = "cifar10";
    int custom_bs = 128;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--amp") { use_amp = true; }
        else if (arg == "--cpu") { use_cpu = true; }
        else if (arg == "--dataset" && i + 1 < argc) { dataset = argv[++i]; }
        else if (arg == "--bs" && i + 1 < argc) { custom_bs = std::stoi(argv[++i]); }
    }

    int resolution = 32;
    int channels = 3;
    int num_classes = 10;
    NormMode norm = NormMode::CIFAR;

    if (dataset == "imagenet") {
        resolution = 224; channels = 3; num_classes = 1000; norm = NormMode::MLPERF;
    } else if (dataset == "cifar10") {
        resolution = 32; channels = 3; num_classes = 10; norm = NormMode::CIFAR;
    } else if (dataset == "mnist") {
        resolution = 28; channels = 1; num_classes = 10; norm = NormMode::MNIST;
    } else {
        std::cerr << "ERROR: Unknown dataset: " << dataset << std::endl;
        return 1;
    }

    if (use_cpu) {
        GLOBAL_SETTING.use_cpu();
    } else {
        GLOBAL_SETTING.use_gpu();
    }
    GLOBAL_SETTING.local_batch_size(custom_bs)
                 .train_resolution(resolution)
                 .val_resolution(resolution);
    if (use_amp) GLOBAL_SETTING.amp(true);

    auto& reg = GlobalRegistry::instance();
    reg.set_num_color_channels(channels);

#ifdef _WIN32
    std::string base_path = "T:\\dataset\\";
#else
    std::string base_path = "/root/epfs/dataset/";
#endif
    std::string dataset_path = base_path + dataset;
    if (dataset == "cifar10") {
        std::string alt = base_path + "cifar-10";
        if (!std::filesystem::exists(dataset_path) && std::filesystem::exists(alt))
            dataset_path = alt;
    }
    if (!std::filesystem::exists(dataset_path)) {
        std::cerr << "ERROR: dataset path not found: " << dataset_path << std::endl;
        return 1;
    }

    int world_size = std::max(1, reg.world_size());
    int preproc_workers = (dataset == "imagenet") ? 16 : 8;
    // 确保预处理线程数是 world_size 的倍数
    if (preproc_workers % world_size != 0) {
        preproc_workers = ((preproc_workers / world_size) + 1) * world_size;
    }

    if (dataset == "imagenet") {
        PREPROCESSOR_SETTING
            .dataset(dataset, dataset_path)
            .color_channels(channels)
            .load_workers(8)
            .preprocess_workers(preproc_workers)
            .cpu_binding(false)
            .normalization(norm)
            .train_transforms(
                FastRandomResizedCrop(resolution, {0.08f, 1.0f}, {0.75f, 1.333f}),
                RandomHorizontalFlip())
            .val_transforms(
                Resize(256),
                CenterCrop(224))
            .partial_mode(true)
            .commit();
    } else {
        PREPROCESSOR_SETTING
            .dataset(dataset, dataset_path)
            .color_channels(channels)
            .load_workers(2)
            .preprocess_workers(preproc_workers)
            .cpu_binding(false)
            .normalization(norm)
            .train_transforms(DoNothing())
            .val_transforms(DoNothing())
            .commit();
    }

    int bs = reg.get_local_batch_size();
    int res = reg.train_sample_resolution_begin();
    int num_ranks = reg.world_size();
    int steps = Preprocessor::instance().steps_per_epoch();

    std::cout << "\n=== Two-Batch Correction Test ===" << std::endl;
    std::cout << "dataset=" << dataset
              << " amp=" << use_amp
              << " cpu=" << use_cpu
              << " bs=" << bs
              << " res=" << res
              << " ch=" << channels
              << " classes=" << num_classes
              << " ranks=" << num_ranks
              << " steps=" << steps << std::endl;

    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)));
    task.loss(CrossEntropyLoss());
    task.optimizer(SGD().momentum(0.9f).weight_decay(5e-4f));
    task.num_classes(num_classes);
    task.total_epochs(1);
    task.validate_every(1, 2);
    task.compile_h2d_only();

    auto result = task.run_h2d_only();

    std::cout << "\n--- Fetch & Verify ---" << std::endl;

    DTensor d_label_a, d_data_a, d_label_b, d_data_b;
    bool found_la = false, found_da = false, found_lb = false, found_db = false;
    for (const auto& d : task.memory_plan().dtensors()) {
        if (d.region == Region::I_A_LABEL) { d_label_a = d; found_la = true; }
        else if (d.region == Region::I_A_DATA)  { d_data_a  = d; found_da = true; }
        else if (d.region == Region::I_B_LABEL) { d_label_b = d; found_lb = true; }
        else if (d.region == Region::I_B_DATA)  { d_data_b  = d; found_db = true; }
    }

    if (!found_la || !found_da || !found_lb || !found_db) {
        std::cerr << "ERROR: Could not find input buffer dtensors" << std::endl;
        return 1;
    }

    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        std::cout << "Rank " << rank << ":" << std::endl;

        Tensor t_label_a = task.fetch_from_rank(d_label_a, rank);
        Tensor t_data_a  = task.fetch_from_rank(d_data_a, rank);
        Tensor t_label_b = task.fetch_from_rank(d_label_b, rank);
        Tensor t_data_b  = task.fetch_from_rank(d_data_b, rank);

        // Raw hex dump before verify — ensures data snapshot matches what verify reads
        if (t_data_a.dtype() == DType::FP16) {
            const uint16_t* pa = t_data_a.data<uint16_t>();
            std::cout << "    [RAW pre-verify] data_a first 8 u16: ";
            for (int i = 0; i < 8; ++i) std::cout << std::hex << pa[i] << " ";
            std::cout << std::dec << std::endl;
        }

        bool la_ok = verify_labels(t_label_a, num_classes);
        bool lb_ok = verify_labels(t_label_b, num_classes);
        bool da_ok = verify_first_last_pixel(t_data_a, "data_a");

        if (t_data_b.dtype() == DType::FP16) {
            const uint16_t* pb = t_data_b.data<uint16_t>();
            std::cout << "    [RAW pre-verify] data_b first 8 u16: ";
            for (int i = 0; i < 8; ++i) std::cout << std::hex << pb[i] << " ";
            std::cout << std::dec << std::endl;
        }

        bool db_ok = verify_first_last_pixel(t_data_b, "data_b");

        // Raw hex dump for debugging AMP — last 8 u16 for post-mortem
        if (t_data_a.dtype() == DType::FP16) {
            const uint16_t* pa = t_data_a.data<uint16_t>();
            const uint16_t* pb = t_data_b.data<uint16_t>();
            std::cout << "    [RAW] data_a last 8 u16: ";
            int na = static_cast<int>(t_data_a.numel());
            for (int i = na - 8; i < na; ++i) std::cout << std::hex << pa[i] << " ";
            std::cout << std::dec << std::endl;
            std::cout << "    [RAW] data_b last 8 u16: ";
            int nb = static_cast<int>(t_data_b.numel());
            for (int i = nb - 8; i < nb; ++i) std::cout << std::hex << pb[i] << " ";
            std::cout << std::dec << std::endl;
        }

        std::cout << "  label_a: " << (la_ok ? "PASS" : "FAIL")
                  << "  data_a: "  << (da_ok ? "PASS" : "FAIL")
                  << "  label_b: " << (lb_ok ? "PASS" : "FAIL")
                  << "  data_b: "  << (db_ok ? "PASS" : "FAIL") << std::endl;

        if (!la_ok || !lb_ok || !da_ok || !db_ok) all_pass = false;
    }

    std::cout << "\n=== OVERALL: " << (all_pass ? "PASS" : "FAIL") << " ===" << std::endl;
    return all_pass ? 0 : 1;
}
