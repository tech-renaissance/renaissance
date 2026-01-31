/**
 * @file d.cpp
 * @brief Expert OP's RandomResizedCrop Implementation using Simd C API
 * @details 使用 Simd C API (SimdResizerInit/Run)，实现预分配工作区和Resizer缓存机制
 *          支持零拷贝裁剪、64字节内存对齐、turbojpeg解码
 * @version 1.0.0
 * @date 2026-01-23
 * @author 技术觉醒团队
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <algorithm>
#include <iostream>

// Simd 库头文件
#include <Simd/SimdLib.h>

// libjpeg-turbo 头文件
#include <turbojpeg.h>

// ============================================================================
// 跨平台对齐内存分配
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
// 常量定义
// ============================================================================
constexpr size_t SIMD_ALIGNMENT = 64;  // AVX2 需要 32 字节，预留 64 字节以备 AVX-512
constexpr int NUM_CHANNELS = 3;        // RGB/BGR
constexpr int MAX_ATTEMPTS = 10;       // RandomResizedCrop 最大尝试次数

// ============================================================================
// 内存对齐辅助函数
// ============================================================================
inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline size_t calculate_aligned_stride(int width, int channels, size_t alignment = SIMD_ALIGNMENT) {
    return align_up(static_cast<size_t>(width) * channels, alignment);
}

// ============================================================================
// Resizer缓存管理器 - 避免重复创建Resizer对象
// ============================================================================
class ResizerCache {
private:
    void* cached_resizer_;
    size_t cached_src_w_, cached_src_h_;
    size_t cached_dst_w_, cached_dst_h_;

public:
    ResizerCache()
        : cached_resizer_(nullptr)
        , cached_src_w_(0)
        , cached_src_h_(0)
        , cached_dst_w_(0)
        , cached_dst_h_(0)
    {}

    ~ResizerCache() {
        if (cached_resizer_) {
            SimdRelease(cached_resizer_);
        }
    }

    // 禁止拷贝
    ResizerCache(const ResizerCache&) = delete;
    ResizerCache& operator=(const ResizerCache&) = delete;

    // 获取或创建 Resizer（缓存机制）
    void* get_resizer(size_t src_w, size_t src_h, size_t dst_w, size_t dst_h) {
        // 如果尺寸匹配，复用缓存的 resizer
        if (cached_resizer_ &&
            cached_src_w_ == src_w && cached_src_h_ == src_h &&
            cached_dst_w_ == dst_w && cached_dst_h_ == dst_h) {
            return cached_resizer_;
        }

        // 释放旧的 resizer
        if (cached_resizer_) {
            SimdRelease(cached_resizer_);
            cached_resizer_ = nullptr;
        }

        // 创建新的 resizer
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
// RandomResizedCrop 参数结构
// ============================================================================
struct RandomResizedCropParams {
    float scale_min = 0.08f;
    float scale_max = 1.0f;
    float ratio_min = 0.75f;
    float ratio_max = 1.3333333f;
    int output_size = 224;  // 输出正方形边长
};

// ============================================================================
// 快速随机数生成器封装
// ============================================================================
class FastRandom {
private:
    unsigned int seed_;

public:
    explicit FastRandom(unsigned int seed = 42) : seed_(seed) {
        srand(seed);
    }

    // 生成 [0, 1) 的浮点数
    float uniform() {
        return static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + 1.0f);
    }

    // 生成 [min, max) 的浮点数
    float uniform(float min_val, float max_val) {
        return min_val + uniform() * (max_val - min_val);
    }
};

// ============================================================================
// RandomResizedCrop 核心实现
// ============================================================================
class RandomResizedCrop {
private:
    ResizerCache& cache_;
    RandomResizedCropParams params_;
    FastRandom rng_;

    // 输出缓冲区信息
    size_t output_stride_;

public:
    RandomResizedCrop(ResizerCache& cache,
                      const RandomResizedCropParams& params = {})
        : cache_(cache)
        , params_(params)
        , rng_(42)
    {
        output_stride_ = calculate_aligned_stride(params_.output_size, NUM_CHANNELS);
    }

    // 计算裁剪区域 - 返回是否成功
    bool get_crop_params(int input_width, int input_height,
                         int& crop_x, int& crop_y,
                         int& crop_w, int& crop_h)
    {
        const float area = static_cast<float>(input_width * input_height);
        const float log_scale_min = std::log(params_.scale_min);
        const float log_scale_max = std::log(params_.scale_max);
        const float log_ratio_min = std::log(params_.ratio_min);
        const float log_ratio_max = std::log(params_.ratio_max);

        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            // 随机选择目标面积（对数均匀分布，匹配PyTorch）
            const float target_area = area * std::exp(rng_.uniform(log_scale_min, log_scale_max));

            // 随机选择宽高比（对数均匀分布）
            const float aspect_ratio = std::exp(rng_.uniform(log_ratio_min, log_ratio_max));

            // 计算裁剪尺寸
            int w = static_cast<int>(std::round(std::sqrt(target_area * aspect_ratio)));
            int h = static_cast<int>(std::round(std::sqrt(target_area / aspect_ratio)));

            // 检查是否在有效范围内
            if (w > 0 && w <= input_width && h > 0 && h <= input_height) {
                // 随机选择裁剪起始位置
                crop_x = static_cast<int>(rng_.uniform() * (input_width - w + 1));
                crop_y = static_cast<int>(rng_.uniform() * (input_height - h + 1));
                crop_w = w;
                crop_h = h;
                return true;
            }
        }

        // 回退策略：中心裁剪
        const float in_ratio = static_cast<float>(input_width) / input_height;
        if (in_ratio < params_.ratio_min) {
            crop_w = input_width;
            crop_h = static_cast<int>(std::round(input_width / params_.ratio_min));
        } else if (in_ratio > params_.ratio_max) {
            crop_h = input_height;
            crop_w = static_cast<int>(std::round(input_height * params_.ratio_max));
        } else {
            crop_w = input_width;
            crop_h = input_height;
        }
        crop_x = (input_width - crop_w) / 2;
        crop_y = (input_height - crop_h) / 2;
        return true;
    }

    // 执行 RandomResizedCrop
    // input_ptr: 输入图像数据（RGB/BGR，已对齐）
    // output_ptr: 输出图像数据（需预分配，已对齐）
    // input_width, input_height: 输入图像尺寸
    // input_stride: 输入图像行步长
    void operator()(const uint8_t* input_ptr,
                    uint8_t* output_ptr,
                    int input_width,
                    int input_height,
                    size_t input_stride)
    {
        // Step 1: 计算裁剪参数
        int crop_x, crop_y, crop_w, crop_h;
        get_crop_params(input_width, input_height, crop_x, crop_y, crop_w, crop_h);

        // Step 2: 获取裁剪区域的起始指针
        const uint8_t* crop_src = input_ptr + crop_y * input_stride + crop_x * NUM_CHANNELS;

        // Step 3: 获取或创建 Resizer
        void* resizer = cache_.get_resizer(
            crop_w, crop_h,
            params_.output_size, params_.output_size
        );

        // Step 4: 直接 Resize（从裁剪区域到输出）
        // Simd 的 Resizer 支持非连续内存（通过 stride 参数）
        SimdResizerRun(
            resizer,
            crop_src, input_stride,      // 源：裁剪区域（使用原始 stride）
            output_ptr, output_stride_   // 目标：输出缓冲区
        );
    }

    size_t get_output_stride() const { return output_stride_; }
    int get_output_size() const { return params_.output_size; }
};

// ============================================================================
// JPEG 解码器（使用 libjpeg-turbo）
// ============================================================================
class JpegDecoder {
private:
    tjhandle handle_;
    uint8_t* decoded_buffer_;
    int width_, height_;
    size_t stride_;
    size_t buffer_size_;

public:
    JpegDecoder() : handle_(nullptr), decoded_buffer_(nullptr),
                    width_(0), height_(0), stride_(0), buffer_size_(0) {
        handle_ = tjInitDecompress();
        if (!handle_) {
            exit(1);
        }
    }

    ~JpegDecoder() {
        if (decoded_buffer_) {
            ALIGNED_FREE(decoded_buffer_);
        }
        if (handle_) {
            tjDestroy(handle_);
        }
    }

    // 禁止拷贝
    JpegDecoder(const JpegDecoder&) = delete;
    JpegDecoder& operator=(const JpegDecoder&) = delete;

    bool decode(const char* filepath) {
        // 读取文件
        FILE* fp = fopen(filepath, "rb");
        if (!fp) {
            return false;
        }

        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        uint8_t* jpeg_buffer = static_cast<uint8_t*>(malloc(file_size));
        if (!jpeg_buffer) {
            fclose(fp);
            return false;
        }

        size_t read_size = fread(jpeg_buffer, 1, file_size, fp);
        fclose(fp);

        if (static_cast<long>(read_size) != file_size) {
            free(jpeg_buffer);
            return false;
        }

        // 获取 JPEG 信息
        int subsamp, colorspace;
        if (tjDecompressHeader3(handle_, jpeg_buffer, file_size,
                                &width_, &height_, &subsamp, &colorspace) != 0) {
            free(jpeg_buffer);
            return false;
        }

        // 计算对齐的 stride 并分配缓冲区
        stride_ = calculate_aligned_stride(width_, NUM_CHANNELS);
        buffer_size_ = stride_ * height_;

        if (decoded_buffer_) {
            ALIGNED_FREE(decoded_buffer_);
        }
        decoded_buffer_ = static_cast<uint8_t*>(
            ALIGNED_ALLOC(SIMD_ALIGNMENT, buffer_size_));

        if (!decoded_buffer_) {
            free(jpeg_buffer);
            return false;
        }

        // 解码 JPEG（使用 RGB 格式）
        if (tjDecompress2(handle_, jpeg_buffer, file_size,
                          decoded_buffer_, width_, static_cast<int>(stride_), height_,
                          TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC) != 0) {
            free(jpeg_buffer);
            return false;
        }

        free(jpeg_buffer);

        return true;
    }

    const uint8_t* data() const { return decoded_buffer_; }
    int width() const { return width_; }
    int height() const { return height_; }
    size_t stride() const { return stride_; }
};

// ============================================================================
// 命令行参数解析
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
// 主程序
// ============================================================================
int main(int argc, char* argv[]) {
    // 解析命令行参数
    Args args;
    if (!args.parse(argc, argv)) {
        return 1;
    }

    // Step 1: 解码 JPEG
    JpegDecoder decoder;
    if (!decoder.decode(args.path.c_str())) {
        return 1;
    }

    // Step 2: 创建 Resizer 缓存
    ResizerCache cache;

    // Step 3: 创建 RandomResizedCrop 处理器
    RandomResizedCropParams params;
    params.output_size = args.output_size;
    RandomResizedCrop cropper(cache, params);

    // Step 4: 预分配输出缓冲区
    size_t output_stride = cropper.get_output_stride();
    size_t output_buffer_size = output_stride * args.output_size;
    uint8_t* output_buffer = static_cast<uint8_t*>(
        ALIGNED_ALLOC(SIMD_ALIGNMENT, output_buffer_size));

    if (!output_buffer) {
        return 1;
    }

    // Step 5: 预热（避免首次运行的额外开销）
    for (int i = 0; i < 10; ++i) {
        cropper(decoder.data(), output_buffer,
                decoder.width(), decoder.height(), decoder.stride());
    }

    // Step 6: 基准测试
    int8_t result = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < args.iterations; ++i) {
        cropper(decoder.data(), output_buffer,
                decoder.width(), decoder.height(), decoder.stride());

        result ^= static_cast<int8_t>(output_buffer[0]);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Step 7: 计算并输出结果
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double total_ms = duration.count() / 1000.0;
    double avg_ms = total_ms / args.iterations;

    std::cout << avg_ms << std::endl;
    std::cout << static_cast<int>(result) << std::endl;

    // 清理
    ALIGNED_FREE(output_buffer);

    return 0;
}
