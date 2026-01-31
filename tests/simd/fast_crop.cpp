#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
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
constexpr int MAX_ATTEMPTS = 10;
constexpr int MCU_SIZE = 8;

// ============================================================================
// Helper functions
// ============================================================================
inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline size_t calculate_aligned_stride(int width, int channels, size_t alignment = SIMD_ALIGNMENT) {
    return align_up(static_cast<size_t>(width) * channels, alignment);
}

inline int align_down_mcu(int value) {
    return (value / MCU_SIZE) * MCU_SIZE;
}

inline int align_up_mcu(int value) {
    return ((value + MCU_SIZE - 1) / MCU_SIZE) * MCU_SIZE;
}

// ============================================================================
// RandomResizedCrop parameters
// ============================================================================
struct CropParams {
    float scale_min = 0.08f;
    float scale_max = 1.0f;
    float ratio_min = 0.75f;
    float ratio_max = 1.3333333f;
    int output_size = 224;
};

// ============================================================================
// Fast random number generator
// ============================================================================
class FastRandom {
private:
    unsigned int seed_;

public:
    explicit FastRandom(unsigned int seed = 42) : seed_(seed) {
        srand(seed);
    }

    float uniform() {
        return static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + 1.0f);
    }

    float uniform(float min_val, float max_val) {
        return min_val + uniform() * (max_val - min_val);
    }

    void reset(unsigned int seed = 42) {
        seed_ = seed;
        srand(seed);
    }
};

// ============================================================================
// Crop region structure
// ============================================================================
struct CropRegion {
    int x, y, w, h;
};

// ============================================================================
// Generate random crop parameters (PyTorch-compatible)
// ============================================================================
bool generate_crop_params(int input_width, int input_height,
                         const CropParams& params, FastRandom& rng,
                         CropRegion& crop) {
    const float area = static_cast<float>(input_width * input_height);
    const float log_scale_min = std::log(params.scale_min);
    const float log_scale_max = std::log(params.scale_max);
    const float log_ratio_min = std::log(params.ratio_min);
    const float log_ratio_max = std::log(params.ratio_max);

    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        const float target_area = area * std::exp(rng.uniform(log_scale_min, log_scale_max));
        const float aspect_ratio = std::exp(rng.uniform(log_ratio_min, log_ratio_max));

        int w = static_cast<int>(std::round(std::sqrt(target_area * aspect_ratio)));
        int h = static_cast<int>(std::round(std::sqrt(target_area / aspect_ratio)));

        if (w > 0 && w <= input_width && h > 0 && h <= input_height) {
            crop.x = static_cast<int>(rng.uniform() * (input_width - w + 1));
            crop.y = static_cast<int>(rng.uniform() * (input_height - h + 1));
            crop.w = w;
            crop.h = h;
            return true;
        }
    }

    // Fallback: center crop
    const float in_ratio = static_cast<float>(input_width) / input_height;
    if (in_ratio < params.ratio_min) {
        crop.w = input_width;
        crop.h = static_cast<int>(std::round(input_width / params.ratio_min));
    } else if (in_ratio > params.ratio_max) {
        crop.h = input_height;
        crop.w = static_cast<int>(std::round(input_height * params.ratio_max));
    } else {
        crop.w = input_width;
        crop.h = input_height;
    }
    crop.x = (input_width - crop.w) / 2;
    crop.y = (input_height - crop.h) / 2;
    return true;
}

// ============================================================================
// Resizer cache
// ============================================================================
class ResizerCache {
private:
    void* cached_resizer_;
    size_t cached_src_w_, cached_src_h_;
    size_t cached_dst_w_, cached_dst_h_;

public:
    ResizerCache()
        : cached_resizer_(nullptr)
        , cached_src_w_(0), cached_src_h_(0)
        , cached_dst_w_(0), cached_dst_h_(0)
    {}

    ~ResizerCache() {
        if (cached_resizer_) {
            SimdRelease(cached_resizer_);
        }
    }

    ResizerCache(const ResizerCache&) = delete;
    ResizerCache& operator=(const ResizerCache&) = delete;

    void* get_resizer(size_t src_w, size_t src_h, size_t dst_w, size_t dst_h) {
        if (cached_resizer_ &&
            cached_src_w_ == src_w && cached_src_h_ == src_h &&
            cached_dst_w_ == dst_w && cached_dst_h_ == dst_h) {
            return cached_resizer_;
        }

        if (cached_resizer_) {
            SimdRelease(cached_resizer_);
            cached_resizer_ = nullptr;
        }

        cached_resizer_ = SimdResizerInit(
            src_w, src_h, dst_w, dst_h,
            NUM_CHANNELS,
            SimdResizeChannelByte,
            SimdResizeMethodBilinear
        );

        cached_src_w_ = src_w;
        cached_src_h_ = src_h;
        cached_dst_w_ = dst_w;
        cached_dst_h_ = dst_h;

        return cached_resizer_;
    }
};

// ============================================================================
// Method 1: Full Decode + Crop + Resize (baseline)
// ============================================================================
class FullDecodeMethod {
private:
    tjhandle tj_handle_;
    uint8_t* full_decode_buffer_;
    size_t full_buffer_size_;
    size_t full_stride_;

public:
    FullDecodeMethod()
        : tj_handle_(nullptr)
        , full_decode_buffer_(nullptr)
        , full_buffer_size_(0)
        , full_stride_(0)
    {
        tj_handle_ = tjInitDecompress();
    }

    ~FullDecodeMethod() {
        if (full_decode_buffer_) {
            ALIGNED_FREE(full_decode_buffer_);
        }
        if (tj_handle_) {
            tjDestroy(tj_handle_);
        }
    }

    FullDecodeMethod(const FullDecodeMethod&) = delete;
    FullDecodeMethod& operator=(const FullDecodeMethod&) = delete;

    bool process(const uint8_t* jpeg_buffer, size_t jpeg_size,
                 uint8_t* output_buffer, size_t output_stride,
                 const CropRegion& crop)
    {
        int width, height, subsamp, colorspace;
        tjDecompressHeader3(tj_handle_, jpeg_buffer, jpeg_size,
                           &width, &height, &subsamp, &colorspace);

        size_t needed_stride = calculate_aligned_stride(width, NUM_CHANNELS);
        size_t needed_size = needed_stride * height;

        if (needed_size > full_buffer_size_) {
            if (full_decode_buffer_) {
                ALIGNED_FREE(full_decode_buffer_);
            }
            full_decode_buffer_ = static_cast<uint8_t*>(
                ALIGNED_ALLOC(SIMD_ALIGNMENT, needed_size));
            full_buffer_size_ = needed_size;
        }
        full_stride_ = needed_stride;

        tjDecompress2(tj_handle_, jpeg_buffer, jpeg_size,
                     full_decode_buffer_, width, static_cast<int>(full_stride_), height,
                     TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC);

        // Crop + Resize
        const uint8_t* crop_src = full_decode_buffer_ + crop.y * full_stride_ + crop.x * NUM_CHANNELS;

        void* resizer = SimdResizerInit(crop.w, crop.h, 224, 224, NUM_CHANNELS,
                                        SimdResizeChannelByte, SimdResizeMethodBilinear);

        SimdResizerRun(resizer, crop_src, full_stride_,
                      output_buffer, output_stride);

        SimdRelease(resizer);
        return true;
    }
};

// ============================================================================
// Method 2: True Partial Decode using TurboJPEG 3.x API
// ============================================================================
class TruePartialDecodeMethod {
private:
    tjhandle tj_handle_;
    uint8_t* temp_buffer_;
    size_t temp_buffer_size_;

public:
    TruePartialDecodeMethod()
        : tj_handle_(nullptr)
        , temp_buffer_(nullptr)
        , temp_buffer_size_(0)
    {
        tj_handle_ = tj3Init(TJINIT_DECOMPRESS);
    }

    ~TruePartialDecodeMethod() {
        if (temp_buffer_) {
            ALIGNED_FREE(temp_buffer_);
        }
        if (tj_handle_) {
            tj3Destroy(tj_handle_);
        }
    }

    TruePartialDecodeMethod(const TruePartialDecodeMethod&) = delete;
    TruePartialDecodeMethod& operator=(const TruePartialDecodeMethod&) = delete;

    bool process(const uint8_t* jpeg_buffer, size_t jpeg_size,
                 uint8_t* output_buffer, size_t output_stride,
                 const CropRegion& crop)
    {
        // Read JPEG header first (required before setting crop region)
        if (tj3DecompressHeader(tj_handle_, jpeg_buffer, jpeg_size) != 0) {
            fprintf(stderr, "tj3DecompressHeader failed: %s\n", tj3GetErrorStr(tj_handle_));
            return false;
        }

        // Calculate MCU-aligned crop region
        int mcu_x = align_down_mcu(crop.x);
        int mcu_y = align_down_mcu(crop.y);
        int mcu_x_end = align_up_mcu(crop.x + crop.w);
        int mcu_y_end = align_up_mcu(crop.y + crop.h);

        // Set cropping region - this tells TurboJPEG to only decode the ROI
        tjregion crop_region;
        crop_region.x = mcu_x;
        crop_region.y = mcu_y;
        crop_region.w = mcu_x_end - mcu_x;
        crop_region.h = mcu_y_end - mcu_y;

        if (tj3SetCroppingRegion(tj_handle_, crop_region) != 0) {
            fprintf(stderr, "tj3SetCroppingRegion failed: %s\n", tj3GetErrorStr(tj_handle_));
            return false;
        }

        // Get cropped dimensions
        int mcu_width = crop_region.w;
        int mcu_height = crop_region.h;

        // Allocate buffer for decoded crop
        size_t crop_stride = calculate_aligned_stride(mcu_width, NUM_CHANNELS);
        size_t crop_size = crop_stride * mcu_height;

        if (crop_size > temp_buffer_size_) {
            if (temp_buffer_) {
                ALIGNED_FREE(temp_buffer_);
            }
            temp_buffer_ = static_cast<uint8_t*>(
                ALIGNED_ALLOC(SIMD_ALIGNMENT, crop_size));
            temp_buffer_size_ = crop_size;
        }

        // Decompress only the cropped region directly to RGB
        if (tj3Decompress8(tj_handle_, jpeg_buffer, jpeg_size,
                          temp_buffer_, static_cast<int>(crop_stride), TJPF_RGB) != 0) {
            fprintf(stderr, "tj3Decompress8 failed: %s\n", tj3GetErrorStr(tj_handle_));
            return false;
        }

        // Resize from MCU-aligned crop to exact crop, then to output
        const uint8_t* exact_crop_src = temp_buffer_ +
            (crop.y - mcu_y) * crop_stride + (crop.x - mcu_x) * NUM_CHANNELS;

        void* resizer = SimdResizerInit(crop.w, crop.h, 224, 224, NUM_CHANNELS,
                                        SimdResizeChannelByte, SimdResizeMethodBilinear);

        SimdResizerRun(resizer, exact_crop_src, crop_stride,
                      output_buffer, output_stride);

        SimdRelease(resizer);
        return true;
    }
};

// ============================================================================
// Command line arguments
// ============================================================================
struct Args {
    std::string path;
    int output_size = 224;
    int iterations = 1000;

    bool parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--path" && i + 1 < argc) {
                path = argv[++i];
            } else if (arg == "--size" && i + 1 < argc) {
                output_size = std::atoi(argv[++i]);
            } else if (arg == "--iter" && i + 1 < argc) {
                iterations = std::atoi(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                printf("Usage: %s --path <input.jpg> [--size <output_size>] [--iter <iterations>]\n",
                       argv[0]);
                printf("  --path   Path to input JPEG file (required)\n");
                printf("  --size   Output size (default: 224)\n");
                printf("  --iter   Number of iterations (default: 1000)\n");
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
    tjhandle temp_handle = tjInitDecompress();
    int width, height, subsamp, colorspace;
    tjDecompressHeader3(temp_handle, jpeg_data, jpeg_file_size,
                       &width, &height, &subsamp, &colorspace);
    tjDestroy(temp_handle);

    CropParams params;
    params.output_size = args.output_size;

    FastRandom rng(42);
    ResizerCache cache;

    size_t output_stride = calculate_aligned_stride(args.output_size, NUM_CHANNELS);
    size_t output_size = output_stride * args.output_size;
    uint8_t* output_buffer = static_cast<uint8_t*>(
        ALIGNED_ALLOC(SIMD_ALIGNMENT, output_size));

    if (!output_buffer) {
        free(jpeg_data);
        return 1;
    }

    // ========================================================================
    // Method 1: Full Decode + Crop + Resize
    // ========================================================================
    {
        printf("=== Method 1: Full Decode + Crop + Resize ===\n");
        FullDecodeMethod method1;

        // Warmup
        rng.reset(42);
        for (int i = 0; i < 10; ++i) {
            CropRegion crop;
            generate_crop_params(width, height, params, rng, crop);
            method1.process(jpeg_data, jpeg_file_size, output_buffer, output_stride, crop);
        }

        int8_t result1 = 0;
        rng.reset(42);

        auto start1 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < args.iterations; ++i) {
            CropRegion crop;
            generate_crop_params(width, height, params, rng, crop);
            method1.process(jpeg_data, jpeg_file_size, output_buffer, output_stride, crop);
            result1 ^= static_cast<int8_t>(output_buffer[0]);
        }

        auto end1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
        double total_ms1 = duration1.count() / 1000.0;
        double avg_ms1 = total_ms1 / args.iterations;

        printf("Total time: %.3f ms\n", total_ms1);
        printf("Average time per iteration: %.4f ms\n", avg_ms1);
        printf("Checksum: %d\n\n", static_cast<int>(result1));
    }

    // ========================================================================
    // Method 2: True Partial Decode + Crop + Resize
    // ========================================================================
    {
        printf("=== Method 2: True Partial Decode (MCU-aligned) ===\n");
        TruePartialDecodeMethod method2;

        // Warmup
        rng.reset(42);
        for (int i = 0; i < 10; ++i) {
            CropRegion crop;
            generate_crop_params(width, height, params, rng, crop);
            method2.process(jpeg_data, jpeg_file_size, output_buffer, output_stride, crop);
        }

        int8_t result2 = 0;
        rng.reset(42);

        auto start2 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < args.iterations; ++i) {
            CropRegion crop;
            generate_crop_params(width, height, params, rng, crop);
            method2.process(jpeg_data, jpeg_file_size, output_buffer, output_stride, crop);
            result2 ^= static_cast<int8_t>(output_buffer[0]);
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
        double total_ms2 = duration2.count() / 1000.0;
        double avg_ms2 = total_ms2 / args.iterations;

        printf("Total time: %.3f ms\n", total_ms2);
        printf("Average time per iteration: %.4f ms\n", avg_ms2);
        printf("Checksum: %d\n\n", static_cast<int>(result2));
    }

    ALIGNED_FREE(output_buffer);
    free(jpeg_data);

    return 0;
}
