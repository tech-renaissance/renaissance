/**
 * @file test_two_po.cpp
 * @brief 测试两个PreprocessOperation的链式处理
 * @version 1.0.0
 * @date 2026-02-21
 * @author 技术觉醒团队
 */

#include "renaissance/data/resize.h"
#include "renaissance/data/center_crop.h"
#include "renaissance/data/random_resized_crop.h"
#include "renaissance/data/random_horizontal_flip.h"
#include "renaissance/data/do_nothing.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"

#include <turbojpeg.h>

// STB Image备用解码（用于处理TurboJPEG无法解码的特殊JPEG格式）
#if TR_USE_STB
    #define STB_IMAGE_IMPLEMENTATION
    #include <stb_image.h>
#endif

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cmath>

#ifdef _WIN32
    #include <windows.h>
    #include <fcntl.h>
    #include <io.h>
#else
    #include <unistd.h>
#endif

using namespace tr;

// =============================================================================
// 配置
// =============================================================================

static constexpr int DEFAULT_OUTPUT_SIZE = 224;
static constexpr int DEFAULT_JPEG_QUALITY = 90;

// =============================================================================
// 辅助函数
// =============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --po1 <NAME>        First PO (Resize/CenterCrop/RandomResizedCrop/RandomHorizontalFlip/DoNothing)\n"
              << "  --po2 <NAME>        Second PO (Resize/CenterCrop/RandomResizedCrop/RandomHorizontalFlip/DoNothing)\n\n"
              << "Optional Options:\n"
              << "  --input <PATH>      Input image path (default: input.jpg)\n"
              << "  --output <PATH>     Output image path (default: workspace/output_two_po.jpg)\n"
              << "  --size <N>          Output size (default: 224)\n"
              << "  --quality <N>       JPEG quality for output (default: 90)\n"
              << "  --seed <N>          Random seed (default: 42)\n"
              << "  --help              Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --po1 RandomResizedCrop --po2 RandomHorizontalFlip --size 224\n"
              << "  " << program_name << " --po1 Resize --po2 CenterCrop --size 128 --seed 123\n";
}

/**
 * @brief 计算MCU对齐的stride
 */
size_t calculate_stride(int32_t width, int32_t channels) {
    constexpr size_t ALIGNMENT = 64;
    size_t raw_stride = static_cast<size_t>(width) * channels;
    return ((raw_stride + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
}

/**
 * @brief 读取JPEG文件到内存
 */
bool read_jpeg_file(const std::string& path, std::vector<uint8_t>& buffer) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "[ERROR] Failed to open file: " << path << "\n";
        return false;
    }

    // 获取文件大小
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        std::cerr << "[ERROR] Invalid file size: " << file_size << "\n";
        fclose(f);
        return false;
    }

    // 读取文件
    buffer.resize(file_size);
    size_t read_size = fread(buffer.data(), 1, file_size, f);
    fclose(f);

    if (read_size != static_cast<size_t>(file_size)) {
        std::cerr << "[ERROR] Failed to read complete file. Read: "
                  << read_size << ", Expected: " << file_size << "\n";
        return false;
    }

    std::cout << "[INFO] Read JPEG file: " << path
              << ", size: " << file_size << " bytes\n";
    return true;
}

/**
 * @brief 保存为JPEG文件
 */
bool save_jpeg_file(const std::string& path,
                    const uint8_t* data,
                    int32_t width,
                    int32_t height,
                    int32_t channels,
                    size_t stride,
                    int quality) {
    // 打开文件
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "[ERROR] Failed to create output file: " << path << "\n";
        return false;
    }

    // 计算需要的buffer大小
    size_t jpeg_size = 0;
    tjhandle tj = tj3Init(TJINIT_COMPRESS);
    if (!tj) {
        std::cerr << "[ERROR] Failed to initialize TurboJPEG for compression\n";
        fclose(f);
        return false;
    }

    // 设置压缩质量和子采样
    tj3Set(tj, TJPARAM_QUALITY, quality);
    int subsamp = TJSAMP_444;
    tj3Set(tj, TJPARAM_SUBSAMP, subsamp);

    // 估算buffer大小（TurboJPEG 3.x API）
    jpeg_size = tj3JPEGBufSize(width, height, subsamp);

    // 压缩（TurboJPEG 3.x API - tj3Compress8会自动分配buffer）
    unsigned char* jpeg_buffer = nullptr;
    if (tj3Compress8(tj, data, width, static_cast<int>(stride), height, TJPF_RGB,
                     &jpeg_buffer, &jpeg_size) != 0) {
        std::cerr << "[ERROR] JPEG compression failed: " << tj3GetErrorStr(tj) << "\n";
        tj3Destroy(tj);
        fclose(f);
        return false;
    }

    // 写入文件
    size_t written = fwrite(jpeg_buffer, 1, jpeg_size, f);
    fclose(f);

    // 释放TurboJPEG分配的buffer
    tj3Free(jpeg_buffer);
    tj3Destroy(tj);

    if (written != jpeg_size) {
        std::cerr << "[ERROR] Failed to write complete JPEG. Wrote: "
                  << written << ", Expected: " << jpeg_size << "\n";
        return false;
    }

    std::cout << "[INFO] Saved JPEG: " << path
              << ", size: " << width << "x" << height
              << ", jpeg_size: " << jpeg_size << " bytes\n";
    return true;
}

/**
 * @brief 完整解码JPEG
 */
bool decode_jpeg_full(tjhandle tj,
                     const uint8_t* jpeg_data,
                     size_t jpeg_size,
                     std::vector<uint8_t>& output,
                     int32_t& width,
                     int32_t& height,
                     size_t& stride) {
    // 读取JPEG头
    if (tj3DecompressHeader(tj, jpeg_data, jpeg_size) != 0) {
        std::cerr << "[WARN] TurboJPEG failed to read JPEG header, trying STB fallback...\n";
        #if TR_USE_STB
        // STB备用解码
        int stb_width, stb_height, stb_channels;
        stbi_uc* stb_data = stbi_load_from_memory(
            reinterpret_cast<const stbi_uc*>(jpeg_data),
            static_cast<int>(jpeg_size),
            &stb_width, &stb_height, &stb_channels,
            3  // 强制3通道RGB
        );

        if (!stb_data) {
            std::cerr << "[ERROR] Both TurboJPEG and STB failed to decode JPEG\n";
            TR_VALUE_ERROR("Failed to decode JPEG: both TurboJPEG and STB failed");
            return false;
        }

        width = stb_width;
        height = stb_height;
        constexpr int CHANNELS = 3;
        stride = calculate_stride(width, CHANNELS);

        size_t buffer_size = stride * height;
        output.resize(buffer_size);

        // 复制STB解码的数据到output
        for (int row = 0; row < height; ++row) {
            const stbi_uc* src_row = stb_data + row * (width * 3);
            uint8_t* dst_row = output.data() + row * stride;
            std::memcpy(dst_row, src_row, width * 3);
        }

        stbi_image_free(stb_data);
        std::cout << "[INFO] STB fallback decode successful: " << width << "x" << height << "\n";
        return true;
        #else
        std::cerr << "[ERROR] TurboJPEG failed to read JPEG header and STB fallback not available\n";
        return false;
        #endif
    }

    width = tj3Get(tj, TJPARAM_JPEGWIDTH);
    height = tj3Get(tj, TJPARAM_JPEGHEIGHT);

    std::cout << "[INFO] JPEG size: " << width << "x" << height << "\n";

    // 计算stride
    constexpr int CHANNELS = 3;
    stride = calculate_stride(width, CHANNELS);

    // 分配输出buffer
    size_t buffer_size = stride * height;
    output.resize(buffer_size);

    // 完整解码
    if (tj3Decompress8(tj, jpeg_data, jpeg_size,
                       output.data(), static_cast<int>(stride),
                       TJPF_RGB) != 0) {
        std::cerr << "[WARN] TurboJPEG decompression failed, trying STB fallback...\n";
        #if TR_USE_STB
        // STB备用解码
        int stb_width, stb_height, stb_channels;
        stbi_uc* stb_data = stbi_load_from_memory(
            reinterpret_cast<const stbi_uc*>(jpeg_data),
            static_cast<int>(jpeg_size),
            &stb_width, &stb_height, &stb_channels,
            3  // 强制3通道RGB
        );

        if (!stb_data) {
            std::cerr << "[ERROR] Both TurboJPEG and STB failed to decode JPEG\n";
            TR_VALUE_ERROR("Failed to decode JPEG: both TurboJPEG and STB failed");
            return false;
        }

        // STB可能返回不同尺寸，更新width/height
        width = stb_width;
        height = stb_height;
        stride = calculate_stride(width, CHANNELS);
        buffer_size = stride * height;
        output.resize(buffer_size);

        // 复制STB解码的数据到output
        for (int row = 0; row < height; ++row) {
            const stbi_uc* src_row = stb_data + row * (width * 3);
            uint8_t* dst_row = output.data() + row * stride;
            std::memcpy(dst_row, src_row, width * 3);
        }

        stbi_image_free(stb_data);
        std::cout << "[INFO] STB fallback decode successful: " << width << "x" << height << "\n";
        return true;
        #else
        std::cerr << "[ERROR] TurboJPEG decompression failed and STB fallback not available\n";
        return false;
        #endif
    }

    std::cout << "[INFO] Full decode complete, stride: " << stride << "\n";
    return true;
}

/**
 * @brief 局部解码JPEG
 */
bool decode_jpeg_partial(tjhandle tj,
                         const uint8_t* jpeg_data,
                         size_t jpeg_size,
                         const DecodeStrategy& strategy,
                         std::vector<uint8_t>& output,
                         int32_t& width,
                         int32_t& height,
                         size_t& stride) {
    // 设置裁剪区域（MCU对齐）
    tjregion crop_region;
    crop_region.x = strategy.decode_x;
    crop_region.y = strategy.decode_y;
    crop_region.w = strategy.decode_w;
    crop_region.h = strategy.decode_h;

    std::cout << "[INFO] Partial decode region: x=" << crop_region.x
              << ", y=" << crop_region.y
              << ", w=" << crop_region.w
              << ", h=" << crop_region.h << "\n";

    if (tj3SetCroppingRegion(tj, crop_region) != 0) {
        std::cerr << "[ERROR] tj3SetCroppingRegion failed: "
                  << tj3GetErrorStr(tj) << "\n";
        return false;
    }

    // 计算stride（基于decode_w）
    constexpr int CHANNELS = 3;
    width = strategy.decode_w;
    height = strategy.decode_h;
    stride = calculate_stride(width, CHANNELS);

    // 分配输出buffer
    size_t buffer_size = stride * height;
    output.resize(buffer_size);

    // 解码
    if (tj3Decompress8(tj, jpeg_data, jpeg_size,
                       output.data(), static_cast<int>(stride),
                       TJPF_RGB) != 0) {
        std::cerr << "[ERROR] JPEG partial decompression failed: "
                  << tj3GetErrorStr(tj) << "\n";
        return false;
    }

    std::cout << "[INFO] Partial decode complete, stride: " << stride << "\n";
    return true;
}

/**
 * @brief 创建PO实例
 */
std::unique_ptr<PreprocessOperation> create_po(const std::string& name, int size) {
    if (name == "Resize") {
        return std::make_unique<Resize>(size);
    } else if (name == "CenterCrop") {
        return std::make_unique<CenterCrop>(size);
    } else if (name == "RandomResizedCrop") {
        return std::make_unique<RandomResizedCrop>(size);
    } else if (name == "RandomHorizontalFlip") {
        return std::make_unique<RandomHorizontalFlip>();
    } else if (name == "DoNothing") {
        return std::make_unique<DoNothing>();
    } else {
        std::cerr << "[ERROR] Unknown PO: " << name << "\n";
        std::cerr << "[INFO] Supported POs: Resize, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, DoNothing\n";
        return nullptr;
    }
}

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char* argv[]) {
    // 解析参数
    std::string input_path = "input.jpg";
    std::string output_path = "workspace/output_two_po.jpg";
    std::string po1_name, po2_name;
    int output_size = DEFAULT_OUTPUT_SIZE;
    int jpeg_quality = DEFAULT_JPEG_QUALITY;
    uint64_t random_seed = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--po1" && i + 1 < argc) {
            po1_name = argv[++i];
        } else if (arg == "--po2" && i + 1 < argc) {
            po2_name = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            output_size = std::atoi(argv[++i]);
        } else if (arg == "--quality" && i + 1 < argc) {
            jpeg_quality = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            random_seed = std::stoull(argv[++i]);
        } else {
            std::cerr << "[ERROR] Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (po1_name.empty() || po2_name.empty()) {
        std::cerr << "[ERROR] --po1 and --po2 parameters are required\n";
        print_usage(argv[0]);
        return 1;
    }

    // 初始化Logger（启用INFO级别）
    Logger::instance().set_level(LogLevel::INFO);

    std::cout << "\n========== Two POs Chain Test ==========\n";
    std::cout << "PO1: " << po1_name << "\n";
    std::cout << "PO2: " << po2_name << "\n";
    std::cout << "Output size: " << output_size << "\n";
    std::cout << "Input: " << input_path << "\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "=========================================\n\n";

    try {
        // =====================================================================
        // Step 1: 读取JPEG文件
        // =====================================================================
        std::vector<uint8_t> jpeg_data;
        if (!read_jpeg_file(input_path, jpeg_data)) {
            return 1;
        }

        // =====================================================================
        // Step 2: 初始化TurboJPEG
        // =====================================================================
        tjhandle tj = tj3Init(TJINIT_DECOMPRESS);
        if (!tj) {
            std::cerr << "[ERROR] Failed to initialize TurboJPEG\n";
            return 1;
        }

        // 设置优化标志
        tj3Set(tj, TJPARAM_FASTDCT, 1);
        tj3Set(tj, TJPARAM_FASTUPSAMPLE, 1);

        // =====================================================================
        // Step 3: 创建PO1和PO2
        // =====================================================================
        auto po1 = create_po(po1_name, output_size);
        if (!po1) {
            tj3Destroy(tj);
            return 1;
        }

        auto po2 = create_po(po2_name, output_size);
        if (!po2) {
            tj3Destroy(tj);
            return 1;
        }

        std::cout << "[INFO] Created PO1: " << po1->name() << "\n";
        std::cout << "[INFO] Created PO2: " << po2->name() << "\n";

        // 初始化RNG（用于随机操作）
        Generator rng;
        rng.set_seed(random_seed);
        if (po1->introduce_randomness() || po2->introduce_randomness()) {
            std::cout << "[INFO] POs introduce randomness, using seed: " << random_seed << "\n";
        }

        // =====================================================================
        // Step 4: 获取PO1的解码策略
        // =====================================================================
        int32_t image_width = 0, image_height = 0;

        // 先读取JPEG头获取尺寸
        if (tj3DecompressHeader(tj, jpeg_data.data(), jpeg_data.size()) != 0) {
            std::cerr << "[ERROR] Failed to read JPEG header: "
                      << tj3GetErrorStr(tj) << "\n";
            tj3Destroy(tj);
            return 1;
        }

        image_width = tj3Get(tj, TJPARAM_JPEGWIDTH);
        image_height = tj3Get(tj, TJPARAM_JPEGHEIGHT);

        // 获取PO1的解码策略（sdmp_factor=1表示不使用SDMP）
        DecodeStrategy strategy1 = po1->get_decode_strategy(
            image_width, image_height, 1, &rng
        );

        std::cout << "[INFO] PO1 decode strategy: need_decode=" << strategy1.need_decode
                  << ", use_partial=" << strategy1.use_partial << "\n";

        // =====================================================================
        // Step 5: 解码JPEG并执行PO1
        // =====================================================================
        std::vector<uint8_t> po1_output;
        int32_t po1_w = 0, po1_h = 0;
        size_t po1_stride = 0;

        auto start_decode = std::chrono::high_resolution_clock::now();

        if (strategy1.use_partial) {
            // PO1使用局部解码
            std::vector<uint8_t> decoded_image;
            int32_t decoded_w = 0, decoded_h = 0;
            size_t decoded_stride = 0;

            if (!decode_jpeg_partial(tj, jpeg_data.data(), jpeg_data.size(),
                                    strategy1, decoded_image,
                                    decoded_w, decoded_h, decoded_stride)) {
                tj3Destroy(tj);
                return 1;
            }

            // 执行PO1：传递R2解码结果（从起始位置开始）
            po1_stride = calculate_stride(output_size, 3);
            po1_output.resize(po1_stride * output_size);

            auto start_po1 = std::chrono::high_resolution_clock::now();

            po1->execute(decoded_image.data(), decoded_w, decoded_h, decoded_stride,
                        po1_output.data(), po1_w, po1_h, po1_stride, &rng,
                        false);  // execute_from_full=false

            auto end_po1 = std::chrono::high_resolution_clock::now();
            double po1_ms = std::chrono::duration<double, std::milli>(
                end_po1 - start_po1).count();

            std::cout << "[INFO] PO1 execution time: " << std::fixed << std::setprecision(2)
                      << po1_ms << " ms\n";

        } else {
            // PO1使用完整解码
            std::vector<uint8_t> decoded_image;
            int32_t decoded_w = 0, decoded_h = 0;
            size_t decoded_stride = 0;

            if (!decode_jpeg_full(tj, jpeg_data.data(), jpeg_data.size(),
                                decoded_image, decoded_w, decoded_h, decoded_stride)) {
                tj3Destroy(tj);
                return 1;
            }

            // 执行PO1
            po1_stride = calculate_stride(output_size, 3);
            po1_output.resize(po1_stride * output_size);

            auto start_po1 = std::chrono::high_resolution_clock::now();

            po1->execute(decoded_image.data(), decoded_w, decoded_h, decoded_stride,
                        po1_output.data(), po1_w, po1_h, po1_stride, &rng,
                        true);  // execute_from_full=true

            auto end_po1 = std::chrono::high_resolution_clock::now();
            double po1_ms = std::chrono::duration<double, std::milli>(
                end_po1 - start_po1).count();

            std::cout << "[INFO] PO1 execution time: " << std::fixed << std::setprecision(2)
                      << po1_ms << " ms\n";
        }

        auto end_decode = std::chrono::high_resolution_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(
            end_decode - start_decode).count();

        std::cout << "[INFO] Decode + PO1 time: " << std::fixed << std::setprecision(2)
                  << decode_ms << " ms\n";
        std::cout << "[INFO] PO1 output: " << po1_w << "x" << po1_h << "\n";

        // 释放TurboJPEG句柄
        tj3Destroy(tj);

        // =====================================================================
        // Step 6: 执行PO2（不需要再解码）
        // =====================================================================
        std::vector<uint8_t> po2_output;
        int32_t po2_w = 0, po2_h = 0;
        size_t po2_stride = 0;

        // PO2的输出尺寸由PO1决定
        po2_stride = calculate_stride(po1_w, 3);
        po2_output.resize(po2_stride * po1_h);

        auto start_po2 = std::chrono::high_resolution_clock::now();

        po2->execute(po1_output.data(), po1_w, po1_h, po1_stride,
                    po2_output.data(), po2_w, po2_h, po2_stride, &rng,
                    true);  // execute_from_full=true（从PO1的完整输出处理）

        auto end_po2 = std::chrono::high_resolution_clock::now();
        double po2_ms = std::chrono::duration<double, std::milli>(
            end_po2 - start_po2).count();

        std::cout << "[INFO] PO2 execution time: " << std::fixed << std::setprecision(2)
                  << po2_ms << " ms\n";
        std::cout << "[INFO] PO2 output: " << po2_w << "x" << po2_h << "\n";

        // =====================================================================
        // Step 7: 保存结果
        // =====================================================================
        if (!save_jpeg_file(output_path, po2_output.data(),
                          po2_w, po2_h, 3, po2_stride, jpeg_quality)) {
            return 1;
        }

        std::cout << "\n[SUCCESS] Test completed successfully!\n";

    } catch (const TRException& e) {
        std::cerr << "\n[ERROR] Framework exception: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Standard exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
