/**
 * @file a.cpp
 * @brief Expert CG's RandomResizedCrop Implementation using Simd C API
 * @details 使用 Simd C API 实现基本的 RandomResizedCrop
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
#include <iostream>
#include <algorithm>

#include <Simd/SimdLib.h>
#include <turbojpeg.h>

// ============================================================================
// 跨平台对齐内存分配
// ============================================================================
static inline void* aligned_malloc(size_t size, size_t alignment = 64) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size)) ptr = nullptr;
    return ptr;
#endif
}

static inline void aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ============================================================================
// 图像数据结构
// ============================================================================
struct ImageData {
    uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

// ============================================================================
// JPEG 解码（使用 libjpeg-turbo）
// ============================================================================
bool decode_jpeg(const std::string& path, ImageData& img) {
    tjhandle tj = tjInitDecompress();
    if (!tj) {
        return false;
    }

    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
        return false;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    unsigned char* jpeg_buf = (unsigned char*)malloc(size);
    fread(jpeg_buf, 1, size, file);
    fclose(file);

    int subsamp, colorspace;
    if (tjDecompressHeader3(tj, jpeg_buf, size, &img.width, &img.height, &subsamp, &colorspace)) {
        free(jpeg_buf);
        tjDestroy(tj);
        return false;
    }

    img.stride = img.width * 3;
    img.data = (uint8_t*)aligned_malloc(img.height * img.stride, 64);
    if (tjDecompress2(tj, jpeg_buf, size, img.data, img.width, 0, img.height, TJPF_RGB, TJFLAG_FASTDCT)) {
        aligned_free(img.data);
        img.data = nullptr;
        free(jpeg_buf);
        tjDestroy(tj);
        return false;
    }

    free(jpeg_buf);
    tjDestroy(tj);
    return true;
}

// ============================================================================
// 裁剪矩形结构
// ============================================================================
struct CropRect {
    int x, y, w, h;
};

// ============================================================================
// 随机裁剪参数生成
// ============================================================================
static inline CropRect get_random_crop(int img_w, int img_h,
                                       float scale_min, float scale_max,
                                       float ratio_min, float ratio_max) {
    const float area = static_cast<float>(img_w * img_h);
    for (int i = 0; i < 10; ++i) {
        float target_area = area * (scale_min + (scale_max - scale_min) * (rand() / (RAND_MAX + 1.0f)));
        float aspect = ratio_min + (ratio_max - ratio_min) * (rand() / (RAND_MAX + 1.0f));
        int w = static_cast<int>(std::round(std::sqrt(target_area * aspect)));
        int h = static_cast<int>(std::round(std::sqrt(target_area / aspect)));
        if (w <= img_w && h <= img_h)
            return {rand() % (img_w - w + 1), rand() % (img_h - h + 1), w, h};
    }
    int size = std::min(img_w, img_h);
    return {(img_w - size) / 2, (img_h - size) / 2, size, size};
}

// ============================================================================
// RandomResizedCrop 核心函数
// ============================================================================
void RandomResizedCrop(const uint8_t* input_ptr, int input_w, int input_h, int input_stride,
                       uint8_t* output_ptr, int output_size,
                       float scale_min = 0.08f, float scale_max = 1.0f,
                       float ratio_min = 0.75f, float ratio_max = 1.3333f)
{
    CropRect rect = get_random_crop(input_w, input_h, scale_min, scale_max, ratio_min, ratio_max);

    // 使用 Simd C API 的 Resize 函数
    // SimdResize 需要源和目标的指针和尺寸
    const uint8_t* src_data = input_ptr + rect.y * input_stride + rect.x * 3;

    // 创建 resizer
    void* resizer = SimdResizerInit(rect.w, rect.h, output_size, output_size, 3,
                                     SimdResizeChannelByte, SimdResizeMethodBilinear);

    if (resizer) {
        // 执行 resize
        SimdResizerRun(resizer, src_data, input_stride, output_ptr, output_size * 3);
        // 释放 resizer
        SimdRelease(resizer);
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv) {
    std::string path = "";
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--path=", 7) == 0)
            path = argv[i] + 7;
    }
    if (path.empty()) {
        return 1;
    }

    ImageData img;
    if (!decode_jpeg(path, img)) {
        return 1;
    }

    const int output_size = 224;
    size_t out_bytes = output_size * output_size * 3;
    uint8_t* output = (uint8_t*)aligned_malloc(out_bytes, 64);

    srand(1234);

    // warm up (not timed)
    RandomResizedCrop(img.data, img.width, img.height, img.stride, output, output_size);

    // timing start
    auto t0 = std::chrono::high_resolution_clock::now();
    int8_t result = 0;
    for (int i = 0; i < 1000; ++i) {
        RandomResizedCrop(img.data, img.width, img.height, img.stride, output, output_size);
        result ^= output[0];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;

    std::cout << ms << std::endl;
    std::cout << (int)result << std::endl;

    aligned_free(output);
    aligned_free(img.data);
    return 0;
}
