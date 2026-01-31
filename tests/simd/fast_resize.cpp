/**
 * @file g.cpp
 * @brief Decode-as-Resize vs Full Decode + Simd Resize 性能对比
 * @details 对比两种图片缩放方案的性能
 * @version 1.0.0
 * @date 2026-01-28
 * @author 技术觉醒团队
 *
 * Method 1: Full Decode + Simd Resize (两步法)
 *   - 使用tjDecompress2解码全图到原始尺寸
 *   - 使用SimdResizer缩放到目标尺寸
 *
 * Method 2: Decode-as-Resize (一步法)
 *   - 使用tj3DecompressToY2Planner2在IDCT阶段直接缩放
 *   - 自动选择最优缩放因子(1/1, 1/2, 1/4, 1/8)
 *   - 如果缩放后仍不匹配,再用Simd微调
 *
 * 预期: Method 2在大图上有明显优势
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <climits>
#include <chrono>
#include <string>
#include <algorithm>
#include <iostream>
#include <vector>

#include <Simd/SimdLib.h>
#include <turbojpeg.h>
#include <jpeglib.h>

// ============================================================================
// Cross-platform aligned memory allocation
// ============================================================================
#ifdef _WIN32
    #include <malloc.h>
    #define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #include <cstdlib>
    #define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

// ============================================================================
// Constants
// ============================================================================
constexpr size_t SIMD_ALIGNMENT = 64;
constexpr int NUM_CHANNELS = 3;

// ============================================================================
// Helper functions
// ============================================================================
inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline size_t calculate_aligned_stride(int width, int channels, size_t alignment = SIMD_ALIGNMENT) {
    return align_up(static_cast<size_t>(width) * channels, alignment);
}

// ============================================================================
// Method 1: Full Decode + Simd Resize (两步法)
// ============================================================================
class FullDecodeAndResizeMethod {
private:
    tjhandle tj_handle_;
    uint8_t* full_decode_buffer_;
    size_t full_buffer_size_;
    size_t full_stride_;

public:
    FullDecodeAndResizeMethod()
        : tj_handle_(nullptr)
        , full_decode_buffer_(nullptr)
        , full_buffer_size_(0)
        , full_stride_(0)
    {
        tj_handle_ = tj3Init(TJINIT_DECOMPRESS);
        if (!tj_handle_) {
            fprintf(stderr, "Failed to initialize TurboJPEG\n");
            return;
        }

        // 设置FASTDCT标志
        tj3Set(tj_handle_, TJPARAM_FASTDCT, 1);
        tj3Set(tj_handle_, TJPARAM_FASTUPSAMPLE, 1);
    }

    ~FullDecodeAndResizeMethod() {
        if (full_decode_buffer_) {
            ALIGNED_FREE(full_decode_buffer_);
        }
        if (tj_handle_) {
            tj3Destroy(tj_handle_);
        }
    }

    FullDecodeAndResizeMethod(const FullDecodeAndResizeMethod&) = delete;
    FullDecodeAndResizeMethod& operator=(const FullDecodeAndResizeMethod&) = delete;

    bool process(const uint8_t* jpeg_buffer, size_t jpeg_size,
                 int target_width, int target_height,
                 uint8_t* output_buffer, size_t output_stride)
    {
        // Step 1: 读取JPEG头
        if (tj3DecompressHeader(tj_handle_, jpeg_buffer, jpeg_size) != 0) {
            fprintf(stderr, "tj3DecompressHeader failed: %s\n", tj3GetErrorStr(tj_handle_));
            return false;
        }

        int src_width = tj3Get(tj_handle_, TJPARAM_JPEGWIDTH);
        int src_height = tj3Get(tj_handle_, TJPARAM_JPEGHEIGHT);

        // Step 2: 解码全图到原始尺寸
        size_t needed_stride = calculate_aligned_stride(src_width, NUM_CHANNELS);
        size_t needed_size = needed_stride * src_height;

        if (needed_size > full_buffer_size_) {
            if (full_decode_buffer_) {
                ALIGNED_FREE(full_decode_buffer_);
            }
            full_decode_buffer_ = static_cast<uint8_t*>(
                ALIGNED_ALLOC(SIMD_ALIGNMENT, needed_size));
            if (!full_decode_buffer_) {
                fprintf(stderr, "Failed to allocate buffer\n");
                return false;
            }
            full_buffer_size_ = needed_size;
        }
        full_stride_ = needed_stride;

        if (tj3Decompress8(tj_handle_, jpeg_buffer, jpeg_size,
                          full_decode_buffer_, static_cast<int>(full_stride_), TJPF_RGB) != 0) {
            fprintf(stderr, "tj3Decompress8 failed: %s\n", tj3GetErrorStr(tj_handle_));
            return false;
        }

        // Step 3: 使用SimdResize缩放到目标尺寸
        void* resizer = SimdResizerInit(src_width, src_height,
                                        target_width, target_height,
                                        NUM_CHANNELS,
                                        SimdResizeChannelByte,
                                        SimdResizeMethodBilinear);
        if (!resizer) {
            fprintf(stderr, "SimdResizerInit failed\n");
            return false;
        }

        SimdResizerRun(resizer, full_decode_buffer_, full_stride_,
                      output_buffer, output_stride);

        SimdRelease(resizer);
        return true;
    }
};

// ============================================================================
// Method 2: Decode-as-Resize (一步法)
// ============================================================================
class DecodeAsResizeMethod {
private:
    tjhandle tj_handle_;
    uint8_t* temp_buffer_;
    size_t temp_buffer_size_;

    // 根据输入和输出尺寸，选择最优的缩放因子
    int select_optimal_scaling_factor(int src_width, int src_height,
                                      int target_width, int target_height) {
        // 获取可用的缩放因子
        const tjscalingfactor* scaling_factors = nullptr;
        int num_factors = 0;
        scaling_factors = tj3GetScalingFactors(&num_factors);

        if (!scaling_factors || num_factors == 0) {
            return 1;  // 默认1/1
        }

        // 策略：选择缩放后尺寸最接近但不小于目标尺寸的缩放因子
        // 如果原图已经小于目标，则选择放大(1/1)
        if (src_width <= target_width || src_height <= target_height) {
            return 1;  // 不缩放，后面会通过Simd放大
        }

        int best_factor_index = 0;
        int min_diff = INT_MAX;

        for (int i = 0; i < num_factors; ++i) {
            const auto& factor = scaling_factors[i];

            // 计算缩放后的尺寸
            int scaled_w = TJSCALED(src_width, factor);
            int scaled_h = TJSCALED(src_height, factor);

            // 确保缩放后不小于目标尺寸
            if (scaled_w >= target_width && scaled_h >= target_height) {
                // 计算与目标尺寸的差异
                int diff = (scaled_w - target_width) + (scaled_h - target_height);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_factor_index = i;
                }
            }
        }

        return best_factor_index;
    }

public:
    DecodeAsResizeMethod()
        : tj_handle_(nullptr)
        , temp_buffer_(nullptr)
        , temp_buffer_size_(0)
    {
        tj_handle_ = tj3Init(TJINIT_DECOMPRESS);
        if (!tj_handle_) {
            fprintf(stderr, "Failed to initialize TurboJPEG\n");
            return;
        }

        // 设置FASTDCT标志
        tj3Set(tj_handle_, TJPARAM_FASTDCT, 1);
        tj3Set(tj_handle_, TJPARAM_FASTUPSAMPLE, 1);
    }

    ~DecodeAsResizeMethod() {
        if (temp_buffer_) {
            ALIGNED_FREE(temp_buffer_);
        }
        if (tj_handle_) {
            tj3Destroy(tj_handle_);
        }
    }

    DecodeAsResizeMethod(const DecodeAsResizeMethod&) = delete;
    DecodeAsResizeMethod& operator=(const DecodeAsResizeMethod&) = delete;

    bool process(const uint8_t* jpeg_buffer, size_t jpeg_size,
                 int target_width, int target_height,
                 uint8_t* output_buffer, size_t output_stride)
    {
        // Step 1: 读取JPEG头
        if (tj3DecompressHeader(tj_handle_, jpeg_buffer, jpeg_size) != 0) {
            fprintf(stderr, "tj3DecompressHeader failed: %s\n", tj3GetErrorStr(tj_handle_));
            return false;
        }

        int src_width = tj3Get(tj_handle_, TJPARAM_JPEGWIDTH);
        int src_height = tj3Get(tj_handle_, TJPARAM_JPEGHEIGHT);

        // Step 2: 选择最优缩放因子
        const tjscalingfactor* scaling_factors = tj3GetScalingFactors(nullptr);
        int num_factors = 0;
        scaling_factors = tj3GetScalingFactors(&num_factors);

        int best_factor_index = select_optimal_scaling_factor(
            src_width, src_height, target_width, target_height);

        const tjscalingfactor& selected_factor = scaling_factors[best_factor_index];

        // Step 3: 设置缩放因子
        tj3SetScalingFactor(tj_handle_, selected_factor);

        // Step 4: 计算缩放后的尺寸
        int scaled_width = TJSCALED(src_width, selected_factor);
        int scaled_height = TJSCALED(src_height, selected_factor);

        // Step 5: 解码(应用IDCT缩放)
        size_t temp_stride = calculate_aligned_stride(scaled_width, NUM_CHANNELS);
        size_t temp_size = temp_stride * scaled_height;

        if (temp_size > temp_buffer_size_) {
            if (temp_buffer_) {
                ALIGNED_FREE(temp_buffer_);
            }
            temp_buffer_ = static_cast<uint8_t*>(
                ALIGNED_ALLOC(SIMD_ALIGNMENT, temp_size));
            if (!temp_buffer_) {
                fprintf(stderr, "Failed to allocate temp buffer\n");
                return false;
            }
            temp_buffer_size_ = temp_size;
        }

        if (tj3Decompress8(tj_handle_, jpeg_buffer, jpeg_size,
                          temp_buffer_, static_cast<int>(temp_stride), TJPF_RGB) != 0) {
            fprintf(stderr, "tj3Decompress8 failed: %s\n", tj3GetErrorStr(tj_handle_));
            return false;
        }

        // Step 6: 如果IDCT缩放后仍不精确匹配目标尺寸,用Simd微调
        if (scaled_width != target_width || scaled_height != target_height) {
            void* resizer = SimdResizerInit(scaled_width, scaled_height,
                                            target_width, target_height,
                                            NUM_CHANNELS,
                                            SimdResizeChannelByte,
                                            SimdResizeMethodBilinear);
            if (!resizer) {
                fprintf(stderr, "SimdResizerInit failed\n");
                return false;
            }

            SimdResizerRun(resizer, temp_buffer_, temp_stride,
                          output_buffer, output_stride);

            SimdRelease(resizer);
        } else {
            // 尺寸精确匹配，直接复制
            if (temp_stride == output_stride) {
                memcpy(output_buffer, temp_buffer_, temp_size);
            } else {
                // 需要逐行复制
                for (int y = 0; y < target_height; ++y) {
                    memcpy(output_buffer + y * output_stride,
                           temp_buffer_ + y * temp_stride,
                           target_width * NUM_CHANNELS);
                }
            }
        }

        return true;
    }
};

// ============================================================================
// Command line arguments
// ============================================================================
struct Args {
    std::string path;
    int target_width = 224;
    int target_height = 224;
    int iterations = 1000;

    bool parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--path" && i + 1 < argc) {
                path = argv[++i];
            } else if (arg == "--width" && i + 1 < argc) {
                target_width = std::atoi(argv[++i]);
            } else if (arg == "--height" && i + 1 < argc) {
                target_height = std::atoi(argv[++i]);
            } else if (arg == "--iter" && i + 1 < argc) {
                iterations = std::atoi(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                printf("Usage: %s --path <input.jpg> [--width W] [--height H] [--iter N]\n",
                       argv[0]);
                printf("  --path   Path to input JPEG file (required)\n");
                printf("  --width  Target width (default: 224)\n");
                printf("  --height Target height (default: 224)\n");
                printf("  --iter   Number of iterations (default: 1000)\n");
                printf("\nMethods:\n");
                printf("  Method 1: Full Decode + Simd Resize (two-step)\n");
                printf("  Method 2: Decode-as-Resize (one-step with IDCT scaling)\n");
                printf("\nExpected: Method 2 is faster for large images\n");
                return false;
            }
        }

        if (path.empty()) {
            fprintf(stderr, "Error: --path is required\n");
            return false;
        }
        return true;
    }
};

// ============================================================================
// Main program
// ============================================================================
int main(int argc, char* argv[]) {
    Args args;
    if (!args.parse(argc, argv)) {
        return 1;
    }

    // Load JPEG file
    FILE* fp = fopen(args.path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", args.path.c_str());
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    long jpeg_file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t* jpeg_data = static_cast<uint8_t*>(malloc(jpeg_file_size));
    if (!jpeg_data) {
        fclose(fp);
        return 1;
    }

    size_t read_size = fread(jpeg_data, 1, jpeg_file_size, fp);
    fclose(fp);

    if (static_cast<long>(read_size) != jpeg_file_size) {
        free(jpeg_data);
        return 1;
    }

    // Get image dimensions
    tjhandle temp_handle = tj3Init(TJINIT_DECOMPRESS);
    if (tj3DecompressHeader(temp_handle, jpeg_data, jpeg_file_size) != 0) {
        fprintf(stderr, "Failed to read JPEG header: %s\n", tj3GetErrorStr(temp_handle));
        free(jpeg_data);
        return 1;
    }
    int width = tj3Get(temp_handle, TJPARAM_JPEGWIDTH);
    int height = tj3Get(temp_handle, TJPARAM_JPEGHEIGHT);
    int subsamp = tj3Get(temp_handle, TJPARAM_SUBSAMP);
    int colorspace = tj3Get(temp_handle, TJPARAM_COLORSPACE);
    tj3Destroy(temp_handle);

    printf("=== Image Information ===\n");
    printf("Image:         %s\n", args.path.c_str());
    printf("Dimensions:    %d x %d\n", width, height);
    printf("Subsampling:   %s\n",
           subsamp == TJSAMP_444 ? "4:4:4" :
           subsamp == TJSAMP_422 ? "4:2:2" :
           subsamp == TJSAMP_420 ? "4:2:0" : "Other");
    printf("Target size:   %d x %d\n", args.target_width, args.target_height);
    printf("Iterations:    %d\n\n", args.iterations);

    // Allocate output buffer
    size_t output_stride = calculate_aligned_stride(args.target_width, NUM_CHANNELS);
    size_t output_size = output_stride * args.target_height;
    uint8_t* output_buffer = static_cast<uint8_t*>(
        ALIGNED_ALLOC(SIMD_ALIGNMENT, output_size));

    if (!output_buffer) {
        free(jpeg_data);
        return 1;
    }

    double avg_ms1 = 0.0;
    int8_t result1 = 0;

    // ========================================================================
    // Method 1: Full Decode + Simd Resize (two-step)
    // ========================================================================
    {
        printf("=== Method 1: Full Decode + Simd Resize (Two-Step) ===\n");
        FullDecodeAndResizeMethod method1;

        // Warmup
        for (int i = 0; i < 10; ++i) {
            method1.process(jpeg_data, jpeg_file_size,
                           args.target_width, args.target_height,
                           output_buffer, output_stride);
        }

        auto start1 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < args.iterations; ++i) {
            method1.process(jpeg_data, jpeg_file_size,
                           args.target_width, args.target_height,
                           output_buffer, output_stride);
            result1 ^= static_cast<int8_t>(output_buffer[0]);
        }

        auto end1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
        double total_ms1 = duration1.count() / 1000.0;
        avg_ms1 = total_ms1 / args.iterations;

        printf("Total time:     %.3f ms\n", total_ms1);
        printf("Average time:   %.4f ms/iter\n", avg_ms1);
        printf("Throughput:     %.1f images/sec\n", 1000.0 / avg_ms1);
        printf("Checksum:       %d\n\n", static_cast<int>(result1));
    }

    // ========================================================================
    // Method 2: Decode-as-Resize (one-step with IDCT scaling)
    // ========================================================================
    {
        printf("=== Method 2: Decode-as-Resize (One-Step IDCT Scaling) ===\n");
        DecodeAsResizeMethod method2;

        // Warmup
        for (int i = 0; i < 10; ++i) {
            method2.process(jpeg_data, jpeg_file_size,
                           args.target_width, args.target_height,
                           output_buffer, output_stride);
        }

        int8_t result2 = 0;

        auto start2 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < args.iterations; ++i) {
            method2.process(jpeg_data, jpeg_file_size,
                           args.target_width, args.target_height,
                           output_buffer, output_stride);
            result2 ^= static_cast<int8_t>(output_buffer[0]);
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
        double total_ms2 = duration2.count() / 1000.0;
        double avg_ms2 = total_ms2 / args.iterations;

        printf("Total time:     %.3f ms\n", total_ms2);
        printf("Average time:   %.4f ms/iter\n", avg_ms2);
        printf("Throughput:     %.1f images/sec\n", 1000.0 / avg_ms2);
        printf("Checksum:       %d\n\n", static_cast<int>(result2));

        // Calculate speedup
        printf("=== Summary ===\n");
        printf("Speedup (Method2 vs Method1): %.2fx\n", avg_ms1 / avg_ms2);
        printf("Time reduction:               %.1f%%\n", (1.0 - avg_ms2 / avg_ms1) * 100.0);

        if (result1 != result2) {
            printf("WARNING: Checksum mismatch! Method1=%d, Method2=%d\n",
                   static_cast<int>(result1), static_cast<int>(result2));
        }
    }

    ALIGNED_FREE(output_buffer);
    free(jpeg_data);

    return 0;
}
